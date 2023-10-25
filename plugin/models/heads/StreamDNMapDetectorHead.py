import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob, xavier_init
from mmcv.runner import force_fp32
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmdet.models import build_loss

from mmdet.core import multi_apply, reduce_mean, build_assigner, build_sampler
from mmdet.models import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid
from ..utils.memory_buffer import StreamTensorMemory
from ..utils.query_update import MotionMLP

from ..utils.utils import gen_sineembed_for_position, SinePositionalEncoding      
from ..utils.query_denoising import CdnQueryGenerator
from ..utils.dn_memory_buffer import DNStreamTensorMemory

@HEADS.register_module(force=True)
class StreamDNMapDetectorHead(nn.Module):

    def __init__(self, 
                 num_queries,
                 num_classes=3,
                 in_channels=128,
                 embed_dims=256,
                 score_thr=0.1,
                 num_points=20,
                 coord_dim=2,
                 roi_size=(60, 30),
                 different_heads=True,
                 predict_refine=False,
                 bev_pos=None,
                 sync_cls_avg_factor=True,
                 bg_cls_weight=0.,
                 streaming_cfg=dict(),
                 transformer=dict(),
                 loss_cls=dict(), 
                 loss_reg=dict(),
                 assigner=dict(),
                 dn_cfg=dict(),
                 loss_dn_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=4.0
                 ),
                 loss_dn_reg=dict(
                    type='LinesL1Loss',
                    loss_weight=50.0,
                    beta=0.01,
                 ),
                 dn_iter=0,
                 dn_cls_num=3,
                ):
        super().__init__()
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.different_heads = different_heads
        self.predict_refine = predict_refine
        self.bev_pos = bev_pos
        self.num_points = num_points
        self.coord_dim = coord_dim
        
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.bg_cls_weight = bg_cls_weight
        
        if streaming_cfg:
            self.streaming_query = streaming_cfg['streaming']
        else:
            self.streaming_query = False
        if self.streaming_query:
            self.batch_size = streaming_cfg['batch_size']
            self.topk_query = streaming_cfg['topk']
            self.trans_loss_weight = streaming_cfg.get('trans_loss_weight', 0.0)
            self.query_memory = StreamTensorMemory(
                self.batch_size,
            )
            self.reference_points_memory = StreamTensorMemory(
                self.batch_size,
            )
            c_dim = 12

            self.query_update = MotionMLP(c_dim=c_dim, f_dim=self.embed_dims, identity=True)
            self.target_memory = StreamTensorMemory(self.batch_size)

            self.stream_dn_target_memory = DNStreamTensorMemory(self.batch_size)
            
        self.register_buffer('roi_size', torch.tensor(roi_size, dtype=torch.float32))
        origin = (-roi_size[0]/2, -roi_size[1]/2)
        self.register_buffer('origin', torch.tensor(origin, dtype=torch.float32))

        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)

        self.transformer = build_transformer(transformer)

        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)
        self.loss_dn_cls = build_loss(loss_dn_cls)
        self.loss_dn_reg = build_loss(loss_dn_reg)
        self.dn_cfg = dn_cfg
        self.assigner = build_assigner(assigner)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        
        self.dn_cls_num = dn_cls_num

        self._init_embedding()
        self._init_branch()
        self.init_weights()

        self.iter = 0
        self.dn_iter = dn_iter

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""

        for p in self.input_proj.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        xavier_init(self.reference_points_embed, distribution='uniform', bias=0.)

        self.transformer.init_weights()

        # init prediction branch
        for m in self.reg_branches:
            for param in m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

        # focal loss init
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            if isinstance(self.cls_branches, nn.ModuleList):
                for m in self.cls_branches:
                    if hasattr(m, 'bias'):
                        nn.init.constant_(m.bias, bias_init)
            else:
                m = self.cls_branches
                nn.init.constant_(m.bias, bias_init)
        
        if self.streaming_query:
            if isinstance(self.query_update, MotionMLP):
                self.query_update.init_weights()
            if hasattr(self, 'query_alpha'):
                for m in self.query_alpha:
                    for param in m.parameters():
                        if param.dim() > 1:
                            nn.init.zeros_(param)

    def _init_embedding(self):
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=self.embed_dims//2,
            normalize=True
        )
        self.bev_pos_embed = build_positional_encoding(positional_encoding)

        # query_pos_embed & query_embed
        self.query_embedding = nn.Embedding(self.num_queries,
                                            self.embed_dims)

        self.reference_points_embed = nn.Linear(self.embed_dims, self.num_points * 2)

        if self.dn_cfg is not None:
            # self.dn_bbox = Linear(self.embed_dims * 2, self.embed_dims)
            self.dn_pts = Linear(self.embed_dims//2, self.embed_dims//2)
            self.label_embedding = nn.Embedding(self.dn_cls_num, self.embed_dims//2)
            self.query_convert = Linear(self.embed_dims, self.embed_dims)
            # self.order_embedding = nn.Embedding(self.num_pts_per_vec, self.embed_dims)
            # self.order_convert = Linear(self.embed_dims, self.embed_dims)
            self.instance_convert = Linear(self.embed_dims*10, self.embed_dims//2)
    
    def _init_branch(self,):
        """Initialize classification branch and regression branch of head."""
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = Linear(self.embed_dims, self.cls_out_channels)

        reg_branch = [
            Linear(self.embed_dims, 2*self.embed_dims),
            nn.LayerNorm(2*self.embed_dims),
            nn.ReLU(),
            Linear(2*self.embed_dims, 2*self.embed_dims),
            nn.LayerNorm(2*self.embed_dims),
            nn.ReLU(),
            Linear(2*self.embed_dims, self.num_points * self.coord_dim),
        ]
        reg_branch = nn.Sequential(*reg_branch)

        num_layers = self.transformer.decoder.num_layers
        if self.different_heads:
            cls_branches = nn.ModuleList(
                [copy.deepcopy(cls_branch) for _ in range(num_layers)])
            reg_branches = nn.ModuleList(
                [copy.deepcopy(reg_branch) for _ in range(num_layers)])
        else:
            cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_layers)])
            reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_layers)])

        self.reg_branches = reg_branches
        self.cls_branches = cls_branches

    def _prepare_context(self, bev_features):
        """Prepare class label and vertex context."""
        device = bev_features.device

        # Add 2D coordinate grid embedding
        B, C, H, W = bev_features.shape
        bev_mask = bev_features.new_zeros(B, H, W)
        bev_pos_embeddings = self.bev_pos_embed(bev_mask) # (bs, embed_dims, H, W)
        bev_features = self.input_proj(bev_features) + bev_pos_embeddings # (bs, embed_dims, H, W)
    
        assert list(bev_features.shape) == [B, self.embed_dims, H, W]
        return bev_features

    def propagate(self, query_embedding, img_metas, return_loss=True):
        bs = query_embedding.shape[0]
        propagated_query_list = []
        prop_reference_points_list = []
        prev2curr_matrix_list = []

        # import ipdb; ipdb.set_trace()
        
        tmp = self.query_memory.get(img_metas)
        query_memory, pose_memory = tmp['tensor'], tmp['img_metas']

        tmp = self.reference_points_memory.get(img_metas)
        ref_pts_memory, pose_memory = tmp['tensor'], tmp['img_metas']

        if return_loss:
            target_memory = self.target_memory.get(img_metas)['tensor']
            trans_loss = query_embedding.new_zeros((1,))
            num_pos = 0

        is_first_frame_list = tmp['is_first_frame']

        for i in range(bs):
            is_first_frame = is_first_frame_list[i]
            if is_first_frame:
                padding = query_embedding.new_zeros((self.topk_query, self.embed_dims))
                propagated_query_list.append(padding)

                padding = query_embedding.new_zeros((self.topk_query, self.num_points, 2))
                prop_reference_points_list.append(padding)

                prev2curr_matrix_list.append(torch.eye(4, dtype=torch.float64).to(query_embedding.device))
            else:
                # use float64 to do precise coord transformation
                prev_e2g_trans = self.roi_size.new_tensor(pose_memory[i]['ego2global_translation'], dtype=torch.float64)
                prev_e2g_rot = self.roi_size.new_tensor(pose_memory[i]['ego2global_rotation'], dtype=torch.float64)
                curr_e2g_trans = self.roi_size.new_tensor(img_metas[i]['ego2global_translation'], dtype=torch.float64)
                curr_e2g_rot = self.roi_size.new_tensor(img_metas[i]['ego2global_rotation'], dtype=torch.float64)
                
                prev_e2g_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                prev_e2g_matrix[:3, :3] = prev_e2g_rot
                prev_e2g_matrix[:3, 3] = prev_e2g_trans

                curr_g2e_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
                curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

                prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
                pos_encoding = prev2curr_matrix.float()[:3].view(-1)

                prev2curr_matrix_list.append(prev2curr_matrix)

                prop_q = query_memory[i]
                query_memory_updated = self.query_update(
                    prop_q, # (topk, embed_dims)
                    pos_encoding.view(1, -1).repeat(len(query_memory[i]), 1)
                )
                propagated_query_list.append(query_memory_updated.clone())

                pred = self.reg_branches[-1](query_memory_updated).sigmoid() # (num_prop, 2*num_pts)
                assert list(pred.shape) == [self.topk_query, 2*self.num_points]

                if return_loss:
                    targets = target_memory[i]

                    weights = targets.new_ones((self.topk_query, 2*self.num_points))
                    bg_idx = torch.all(targets.view(self.topk_query, -1) == 0.0, dim=1)
                    num_pos = num_pos + (self.topk_query - bg_idx.sum())
                    weights[bg_idx, :] = 0.0

                    denormed_targets = targets * self.roi_size + self.origin # (topk, num_pts, 2)
                    denormed_targets = torch.cat([
                        denormed_targets,
                        denormed_targets.new_zeros((self.topk_query, self.num_points, 1)), # z-axis
                        denormed_targets.new_ones((self.topk_query, self.num_points, 1)) # 4-th dim
                    ], dim=-1) # (num_prop, num_pts, 4)
                    assert list(denormed_targets.shape) == [self.topk_query, self.num_points, 4]
                    curr_targets = torch.einsum('lk,ijk->ijl', prev2curr_matrix.float(), denormed_targets)
                    normed_targets = (curr_targets[..., :2] - self.origin) / self.roi_size # (num_prop, num_pts, 2)
                    normed_targets = torch.clip(normed_targets, min=0., max=1.).reshape(-1, 2*self.num_points)
                    # (num_prop, 2*num_pts)
                    trans_loss += self.loss_reg(pred, normed_targets, weights, avg_factor=1.0)
                
                # ref pts
                prev_ref_pts = ref_pts_memory[i]
                denormed_ref_pts = prev_ref_pts * self.roi_size + self.origin # (num_prop, num_pts, 2)
                assert list(prev_ref_pts.shape) == [self.topk_query, self.num_points, 2]
                denormed_ref_pts = torch.cat([
                    denormed_ref_pts,
                    denormed_ref_pts.new_zeros((self.topk_query, self.num_points, 1)), # z-axis
                    denormed_ref_pts.new_ones((self.topk_query, self.num_points, 1)) # 4-th dim
                ], dim=-1) # (num_prop, num_pts, 4)
                assert list(denormed_ref_pts.shape) == [self.topk_query, self.num_points, 4]

                curr_ref_pts = torch.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_ref_pts.double()).float()
                normed_ref_pts = (curr_ref_pts[..., :2] - self.origin) / self.roi_size # (num_prop, num_pts, 2)
                normed_ref_pts = torch.clip(normed_ref_pts, min=0., max=1.)

                prop_reference_points_list.append(normed_ref_pts)
                
        prop_query_embedding = torch.stack(propagated_query_list) # (bs, topk, embed_dims)
        prop_ref_pts = torch.stack(prop_reference_points_list) # (bs, topk, num_pts, 2)
        assert list(prop_query_embedding.shape) == [bs, self.topk_query, self.embed_dims]
        assert list(prop_ref_pts.shape) == [bs, self.topk_query, self.num_points, 2]
        
        init_reference_points = self.reference_points_embed(query_embedding).sigmoid() # (bs, num_q, 2*num_pts)
        init_reference_points = init_reference_points.view(bs, self.num_queries, self.num_points, 2) # (bs, num_q, num_pts, 2)
        memory_query_embedding = None

        if return_loss:
            trans_loss = self.trans_loss_weight * trans_loss / (num_pos + 1e-10)
            return query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, memory_query_embedding, is_first_frame_list, trans_loss, prev2curr_matrix_list
        else:
            return query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, memory_query_embedding, is_first_frame_list

    def denoise(self, gt_vecs_list, gt_labels_list, device):
        gt_pts_list = [gt_bboxes[:, 0, :].reshape(-1, 20, 2) for gt_bboxes in gt_vecs_list]

        for i in gt_labels_list:
            if len(i) == 0:
                import pdb; pdb.set_trace() 

        gt_bboxes_list = []
        for gt in gt_pts_list:
            gt_bboxes_list.append(torch.cat((gt.min(1)[0], gt.max(1)[0]), -1))

        if self.dn_cfg is not None:
            if self.dn_cfg.get('neg', False) is True:
                self.dn_generator = PNCdnQueryGenerator(**self.dn_cfg)
            else:
                self.dn_generator = CdnQueryGenerator(**self.dn_cfg)
    
            dn_query_content, input_query_bbox, input_query_pts, attn_mask, dn_meta, denoise_refers \
                            = self.dn_generator(gt_bboxes_list, gt_pts_list, gt_labels_list, self.label_embedding)
            pe_pts = self.dn_pts(gen_sineembed_for_position(denoise_refers.flatten(1, 2), self.embed_dims//4))
            pe_pts = pe_pts.reshape(pe_pts.size(0), -1, 20, pe_pts.size(2)).flatten(2)
            dn_query_pos = self.instance_convert(pe_pts)
            query = self.query_convert(torch.cat((dn_query_content, dn_query_pos), -1))

            if torch.isnan(query).any() or torch.isinf(query).any() or torch.isnan(denoise_refers).any() or torch.isinf(denoise_refers).any():
                import pdb; pdb.set_trace() 
                torch.isnan(denoise_refers.sum((2, 3))).any()
            
            return query, attn_mask, dn_meta, denoise_refers

    def stream_denoise(self, gt_vecs_list, gt_labels_list_x, cur_gt_vecs_list, cur_gt_labels_list_x, device, prev2curr_matrix_list, prop_query_embedding):
        pos_valid = []
        gt_pts_list = []
        gt_labels_list = []
        last_save_idx = {}

        for i, gt_bboxes in enumerate(gt_vecs_list):
            if gt_bboxes is not None:
                prev_ref_pts = gt_bboxes[:, 0, :].reshape(-1, 20, 2)
                denormed_ref_pts = prev_ref_pts * self.roi_size + self.origin # (num_prop, num_pts, 2)
                denormed_ref_pts = torch.cat([
                    denormed_ref_pts,
                    denormed_ref_pts.new_zeros((denormed_ref_pts.size(0), self.num_points, 1)), # z-axis
                    denormed_ref_pts.new_ones((denormed_ref_pts.size(0), self.num_points, 1)) # 4-th dim
                ], dim=-1) # (num_prop, num_pts, 4)

                prev2curr_matrix = prev2curr_matrix_list[i]
                curr_ref_pts = torch.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_ref_pts.double()).float()
                normed_ref_pts = (curr_ref_pts[..., :2] - self.origin) / self.roi_size # (num_prop, num_pts, 2)
                normed_ref_pts = torch.clip(normed_ref_pts, min=0., max=1.)
                select_idx = (((normed_ref_pts==0).sum((1, 2)) + (normed_ref_pts==1).sum((1, 2))) != 40)

                if select_idx.sum() == 0:
                    gt_pts_list.append(cur_gt_vecs_list[i][:, 0, :].reshape(-1, 20, 2))
                    gt_labels_list.append(cur_gt_labels_list_x[i])    
                else:
                    pos_valid.append(i)
                    gt_pts_list.append(normed_ref_pts[select_idx])
                    gt_labels_list.append(gt_labels_list_x[i][select_idx])
                    last_save_idx[i] = select_idx
            # if gt_bboxes is not None:
            #     pos_valid.append(i)
            #     gt_pts_list.append(gt_bboxes[:, 0, :].reshape(-1, 20, 2))
            #     gt_labels_list.append(gt_labels_list_x[i])
            else:
                gt_pts_list.append(cur_gt_vecs_list[i][:, 0, :].reshape(-1, 20, 2))
                gt_labels_list.append(cur_gt_labels_list_x[i])

        gt_bboxes_list = []
        for gt in gt_pts_list:
            gt_bboxes_list.append(torch.cat((gt.min(1)[0], gt.max(1)[0]), -1))

        if self.dn_cfg is not None:
            self.dn_generator = CdnQueryGenerator(**self.dn_cfg)
    
            dn_query_content, input_query_bbox, input_query_pts, attn_mask, dn_meta, denoise_refers \
                            = self.dn_generator(gt_bboxes_list, gt_pts_list, gt_labels_list, self.label_embedding, prop_query_embedding)
            pe_pts = self.dn_pts(gen_sineembed_for_position(denoise_refers.flatten(1, 2), self.embed_dims//4))
            pe_pts = pe_pts.reshape(pe_pts.size(0), -1, 20, pe_pts.size(2)).flatten(2)
            dn_query_pos = self.instance_convert(pe_pts)
            query = self.query_convert(torch.cat((dn_query_content, dn_query_pos), -1))

            if torch.isnan(query).any() or torch.isinf(query).any() or torch.isnan(denoise_refers).any() or torch.isinf(denoise_refers).any():
                import pdb; pdb.set_trace() 
                torch.isnan(denoise_refers.sum((2, 3))).any()

            # TODO 如果把prev gt转到当前帧再DN，则不需要query转换
            # for i in pos_valid:
            #     prev2curr_matrix = prev2curr_matrix_list[i]
            #     query_num_tmp = query[i].size(0)
            #     pos_encoding = prev2curr_matrix.float()[:3].view(-1)
            #     query[i] = self.query_update(query[i], pos_encoding.view(1, -1).repeat(query_num_tmp, 1))

            #     # ref pts
            #     prev_ref_pts = denoise_refers[i]
            #     denormed_ref_pts = prev_ref_pts * self.roi_size + self.origin # (num_prop, num_pts, 2)
            #     denormed_ref_pts = torch.cat([
            #         denormed_ref_pts,
            #         denormed_ref_pts.new_zeros((query_num_tmp, self.num_points, 1)), # z-axis
            #         denormed_ref_pts.new_ones((query_num_tmp, self.num_points, 1)) # 4-th dim
            #     ], dim=-1) # (num_prop, num_pts, 4)

            #     curr_ref_pts = torch.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_ref_pts.double()).float()
            #     normed_ref_pts = (curr_ref_pts[..., :2] - self.origin) / self.roi_size # (num_prop, num_pts, 2)
            #     denoise_refers[i] = torch.clip(normed_ref_pts, min=0., max=1.)

            # for i, j in enumerate(pos_valid):
            #     prev2curr_matrix = prev2curr_matrix_list[j]
            #     query_num_tmp = query[i].size(0)
            #     pos_encoding = prev2curr_matrix.float()[:3].view(-1)
            #     query[i] = self.query_update(query[i], pos_encoding.view(1, -1).repeat(query_num_tmp, 1))

            #     # ref pts
            #     prev_ref_pts = denoise_refers[i]
            #     denormed_ref_pts = prev_ref_pts * self.roi_size + self.origin # (num_prop, num_pts, 2)
            #     denormed_ref_pts = torch.cat([
            #         denormed_ref_pts,
            #         denormed_ref_pts.new_zeros((query_num_tmp, self.num_points, 1)), # z-axis
            #         denormed_ref_pts.new_ones((query_num_tmp, self.num_points, 1)) # 4-th dim
            #     ], dim=-1) # (num_prop, num_pts, 4)

            #     curr_ref_pts = torch.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_ref_pts.double()).float()
            #     normed_ref_pts = (curr_ref_pts[..., :2] - self.origin) / self.roi_size # (num_prop, num_pts, 2)
            #     denoise_refers[i] = torch.clip(normed_ref_pts, min=0., max=1.)
            
            # 一个batch有None，有需要dn的，分开处理
            # if len(pos_valid) != len(gt_vecs_list):
            #     tmp_query = []
            #     tmp_refers = []
            #     for i in range(len(gt_vecs_list)):
            #         if i in pos_valid:
            #             find_idx = pos_valid.index(i)
            #             tmp_query.append(query[find_idx])
            #             tmp_refers.append(denoise_refers[find_idx])
            #             # 如果dn query求的batch位置和整体位置不一致
            #             if i != find_idx:
            #                 dn_meta['known_bid'][dn_meta['known_bid']==find_idx] = i
            #         else:
            #             tmp_query.append(torch.zeros_like(query[0]))
            #             tmp_refers.append(torch.zeros_like(denoise_refers[0]))
            #     query = torch.stack(tmp_query, dim=0)
            #     denoise_refers = torch.stack(tmp_refers, dim=0)

            dn_meta['stream_valid'] = pos_valid
            dn_meta['last_valid_idx'] = last_save_idx
            return query, attn_mask, dn_meta, denoise_refers

    def forward_train(self, bev_features, img_metas, gts):
        '''
        Args:
            bev_feature (List[Tensor]): shape [B, C, H, W]
                feature in bev view
        Outs:
            preds_dict (list[dict]):
                lines (Tensor): Classification score of all
                    decoder layers, has shape
                    [bs, num_query, 2*num_points]
                scores (Tensor):
                    [bs, num_query,]
        '''

        bev_features = self._prepare_context(bev_features)

        bs, C, H, W = bev_features.shape
        img_masks = bev_features.new_zeros((bs, H, W))
        # pos_embed = self.positional_encoding(img_masks)
        pos_embed = None

        query_embedding = self.query_embedding.weight[None, ...].repeat(bs, 1, 1) # [B, num_q, embed_dims]
        input_query_num = self.num_queries

        # num query: self.num_query + self.topk
        if self.streaming_query:
            query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, memory_query, is_first_frame_list, trans_loss, prev2curr_matrix_list = \
                self.propagate(query_embedding, img_metas, return_loss=True)
            
        else:
            init_reference_points = self.reference_points_embed(query_embedding).sigmoid() # (bs, num_q, 2*num_pts)
            init_reference_points = init_reference_points.view(-1, self.num_queries, self.num_points, 2) # (bs, num_q, num_pts, 2)
            prop_query_embedding = None
            prop_ref_pts = None
            is_first_frame_list = [True for i in range(bs)]
        
        assert list(init_reference_points.shape) == [bs, self.num_queries, self.num_points, 2]
        assert list(query_embedding.shape) == [bs, self.num_queries, self.embed_dims]

        # stream dn target获取
        last_gt = self.stream_dn_target_memory.get(img_metas)
        last_gt_list, last_id_list, last_label_list = last_gt['gt'], last_gt['id_list'], last_gt['label_list']

        if self.iter >= self.dn_iter:
            # stream_dn_query, stream_self_attn_mask, stream_dn_meta, stream_denoise_refers \
            #             = self.stream_denoise(copy.deepcopy(last_gt_list), copy.deepcopy(last_label_list), 
            #             copy.deepcopy(gts[0]['lines']), copy.deepcopy(gts[0]['labels']), 
            #             bev_features.device, prev2curr_matrix_list, 
            #             prop_query_embedding=None)
            stream_dn_query, stream_self_attn_mask, stream_dn_meta, stream_denoise_refers \
                        = self.denoise(copy.deepcopy(last_gt_list), copy.deepcopy(last_label_list), 
                        copy.deepcopy(gts[0]['lines']), copy.deepcopy(gts[0]['labels']), 
                        bev_features.device, prev2curr_matrix_list, 
                        prop_query_embedding=None)
            dn_num = stream_dn_meta['pad_size']
        else:
            # if prop_query_embedding is not None:
            #     mask_size = query_embedding.size(1) + prop_query_embedding.size(1)
            # else:
            #     mask_size = query_embedding.size(1)
            # stream_self_attn_mask = torch.ones(mask_size, mask_size).to(bev_features.device) < 0
            stream_self_attn_mask = None
            stream_dn_query = None
            stream_denoise_refers = None
            dn_num = 0

        # 得到gt bbox和points，进行denoise预处理
        # if self.dn_cfg is not None:
        #     dn_query, self_attn_mask, dn_meta, denoise_refers \
        #                  = self.denoise(copy.deepcopy(gts[0]['lines']), copy.deepcopy(gts[0]['labels']), bev_features.device)
        #     dn_num = dn_meta['pad_size']

        # outs_dec: (num_layers, num_qs, bs, embed_dims)
        inter_queries, init_reference, inter_references = self.transformer(
            mlvl_feats=[bev_features,],
            mlvl_masks=[img_masks.type(torch.bool)],
            query_embed=query_embedding,
            prop_query=prop_query_embedding,
            mlvl_pos_embeds=[pos_embed], # not used
            memory_query=None,
            init_reference_points=init_reference_points,
            prop_reference_points=prop_ref_pts,
            reg_branches=self.reg_branches,
            cls_branches=self.cls_branches,
            predict_refine=self.predict_refine,
            is_first_frame_list=is_first_frame_list,
            query_key_padding_mask=query_embedding.new_zeros((bs, self.num_queries), dtype=torch.bool), # mask used in self-attn,
            dn_query=stream_dn_query,
            dn_refer=stream_denoise_refers,
            attn_masks=stream_self_attn_mask,
        )

        outputs = []
        for i, (queries) in enumerate(inter_queries):
            reg_points = inter_references[i][:, dn_num:] # (bs, num_q, num_points, 2)
            bs = reg_points.shape[0]
            reg_points = reg_points.view(bs, -1, 2*self.num_points) # (bs, num_q, 2*num_points)

            scores = self.cls_branches[i](queries[:, dn_num:]) # (bs, num_q, num_classes)

            reg_points_list = []
            scores_list = []
            for i in range(len(scores)):
                # padding queries should not be output
                reg_points_list.append(reg_points[i])
                scores_list.append(scores[i])

            pred_dict = {
                'lines': reg_points_list,
                'scores': scores_list
            }
            outputs.append(pred_dict)
        loss_dict, det_match_idxs, det_match_gt_idxs, gt_lines_list = self.loss(gts=gts, preds=outputs)
        
        if dn_num != 0:
            # DN输出
            outputs_dn = []
            for i, (queries) in enumerate(inter_queries):
                reg_points = inter_references[i][:, :dn_num] # (bs, num_q, num_points, 2)
                bs = reg_points.shape[0]
                reg_points = reg_points.view(bs, -1, 2*self.num_points) # (bs, num_q, 2*num_points)

                scores = self.cls_branches[i](queries[:, :dn_num]) # (bs, num_q, num_classes)

                reg_points_list = []
                scores_list = []
                for i in range(len(scores)):
                    # padding queries should not be output
                    reg_points_list.append(reg_points[i])
                    scores_list.append(scores[i])

                pred_dict = {
                    'lines': reg_points_list,
                    'scores': scores_list
                }
                outputs_dn.append(pred_dict)
            
            loss_dn_dict = self.loss_dn(gts=copy.deepcopy(gts[0]), last_gts=copy.deepcopy(last_gt_list), 
                                        last_ids=copy.deepcopy(last_id_list), last_labels=copy.deepcopy(last_label_list), \
                                        preds=outputs_dn, dn_meta=stream_dn_meta)
            loss_dict.update(loss_dn_dict)

        if self.streaming_query:
            query_list = []
            ref_pts_list = []
            gt_targets_list = []
            lines, scores = outputs[-1]['lines'], outputs[-1]['scores']
            gt_lines = gt_lines_list[-1] # take results from the last layer

            for i in range(bs):
                _lines = lines[i]
                _queries = inter_queries[-1][i][dn_num:]
                _scores = scores[i]
                _gt_targets = gt_lines[i] # (num_q or num_q+topk, 20, 2)
                assert len(_lines) == len(_queries)
                assert len(_lines) == len(_gt_targets)

                _scores, _ = _scores.max(-1)
                topk_score, topk_idx = _scores.topk(k=self.topk_query, dim=-1)

                _queries = _queries[topk_idx] # (topk, embed_dims)
                _lines = _lines[topk_idx] # (topk, 2*num_pts)
                _gt_targets = _gt_targets[topk_idx] # (topk, 20, 2)
                
                query_list.append(_queries)
                ref_pts_list.append(_lines.view(-1, self.num_points, 2))
                gt_targets_list.append(_gt_targets.view(-1, self.num_points, 2))

            self.query_memory.update(query_list, img_metas)
            self.reference_points_memory.update(ref_pts_list, img_metas)
            self.target_memory.update(gt_targets_list, img_metas)

            self.stream_dn_target_memory.update(gts[0]['lines'], gts[0]['id'], gts[0]['labels'], img_metas)

            loss_dict['trans_loss'] = trans_loss
        
        self.iter += 1
        # print(self.iter)
        return outputs, loss_dict, det_match_idxs, det_match_gt_idxs
    
    def forward_test(self, bev_features, img_metas):
        '''
        Args:
            bev_feature (List[Tensor]): shape [B, C, H, W]
                feature in bev view
        Outs:
            preds_dict (list[dict]):
                lines (Tensor): Classification score of all
                    decoder layers, has shape
                    [bs, num_query, 2*num_points]
                scores (Tensor):
                    [bs, num_query,]
        '''

        bev_features = self._prepare_context(bev_features)

        bs, C, H, W = bev_features.shape
        img_masks = bev_features.new_zeros((bs, H, W))
        # pos_embed = self.positional_encoding(img_masks)
        pos_embed = None

        query_embedding = self.query_embedding.weight[None, ...].repeat(bs, 1, 1) # [B, num_q, embed_dims]
        input_query_num = self.num_queries
        # num query: self.num_query + self.topk
        if self.streaming_query:
            query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, memory_query, is_first_frame_list = \
                self.propagate(query_embedding, img_metas, return_loss=False)
            
        else:
            init_reference_points = self.reference_points_embed(query_embedding).sigmoid() # (bs, num_q, 2*num_pts)
            init_reference_points = init_reference_points.view(-1, self.num_queries, self.num_points, 2) # (bs, num_q, num_pts, 2)
            prop_query_embedding = None
            prop_ref_pts = None
            is_first_frame_list = [True for i in range(bs)]
        
        assert list(init_reference_points.shape) == [bs, input_query_num, self.num_points, 2]
        assert list(query_embedding.shape) == [bs, input_query_num, self.embed_dims]

        # outs_dec: (num_layers, num_qs, bs, embed_dims)
        inter_queries, init_reference, inter_references = self.transformer(
            mlvl_feats=[bev_features,],
            mlvl_masks=[img_masks.type(torch.bool)],
            query_embed=query_embedding,
            prop_query=prop_query_embedding,
            mlvl_pos_embeds=[pos_embed], # not used
            memory_query=None,
            init_reference_points=init_reference_points,
            prop_reference_points=prop_ref_pts,
            reg_branches=self.reg_branches,
            cls_branches=self.cls_branches,
            predict_refine=self.predict_refine,
            is_first_frame_list=is_first_frame_list,
            query_key_padding_mask=query_embedding.new_zeros((bs, self.num_queries), dtype=torch.bool), # mask used in self-attn,
        )

        outputs = []
        for i, (queries) in enumerate(inter_queries):
            reg_points = inter_references[i] # (bs, num_q, num_points, 2)
            bs = reg_points.shape[0]
            reg_points = reg_points.view(bs, -1, 2*self.num_points) # (bs, num_q, 2*num_points)
            scores = self.cls_branches[i](queries) # (bs, num_q, num_classes)

            reg_points_list = []
            scores_list = []
            prop_mask_list = []
            for i in range(len(scores)):
                # padding queries should not be output
                reg_points_list.append(reg_points[i])
                scores_list.append(scores[i])
                prop_mask = scores.new_ones((len(scores[i]), ), dtype=torch.bool)
                prop_mask[-len(scores[i]):] = False
                # prop_mask[-self.num_queries:] = False
                prop_mask_list.append(prop_mask)

            pred_dict = {
                'lines': reg_points_list,
                'scores': scores_list,
                'prop_mask': prop_mask_list
            }
            outputs.append(pred_dict)
        
        if self.streaming_query:
            query_list = []
            ref_pts_list = []
            lines, scores = outputs[-1]['lines'], outputs[-1]['scores']
            for i in range(bs):
                _lines = lines[i]
                _queries = inter_queries[-1][i]
                _scores = scores[i]
                assert len(_lines) == len(_queries)
                _scores, _ = _scores.max(-1)
                topk_score, topk_idx = _scores.topk(k=self.topk_query, dim=-1)

                _queries = _queries[topk_idx] # (topk, embed_dims)
                _lines = _lines[topk_idx] # (topk, 2*num_pts)
                
                query_list.append(_queries)
                ref_pts_list.append(_lines.view(-1, self.num_points, 2))

            self.query_memory.update(query_list, img_metas)
            self.reference_points_memory.update(ref_pts_list, img_metas)

        return outputs

    @force_fp32(apply_to=('score_pred', 'lines_pred', 'gt_lines'))
    def _get_target_single(self,
                           score_pred,
                           lines_pred,
                           gt_labels,
                           gt_lines,
                           gt_bboxes_ignore=None):
        """
            Compute regression and classification targets for one image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                score_pred (Tensor): Box score logits from a single decoder layer
                    for one image. Shape [num_query, cls_out_channels].
                lines_pred (Tensor):
                    shape [num_query, 2*num_points]
                gt_labels (torch.LongTensor)
                    shape [num_gt, ]
                gt_lines (Tensor):
                    shape [num_gt, 2*num_points].
                
            Returns:
                tuple[Tensor]: a tuple containing the following for one sample.
                    - labels (LongTensor): Labels of each image.
                        shape [num_query, 1]
                    - label_weights (Tensor]): Label weights of each image.
                        shape [num_query, 1]
                    - lines_target (Tensor): Lines targets of each image.
                        shape [num_query, num_points, 2]
                    - lines_weights (Tensor): Lines weights of each image.
                        shape [num_query, num_points, 2]
                    - pos_inds (Tensor): Sampled positive indices for each image.
                    - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_pred_lines = len(lines_pred)
        # assigner and sampler
        assign_result, gt_permute_idx = self.assigner.assign(preds=dict(lines=lines_pred, scores=score_pred,),
                                             gts=dict(lines=gt_lines,
                                                      labels=gt_labels, ),
                                             gt_bboxes_ignore=gt_bboxes_ignore)
        sampling_result = self.sampler.sample(
            assign_result, lines_pred, gt_lines)
        num_gt = len(gt_lines)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_gt_inds = sampling_result.pos_assigned_gt_inds

        labels = gt_lines.new_full(
                (num_pred_lines, ), self.num_classes, dtype=torch.long) # (num_q, )
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_lines.new_ones(num_pred_lines) # (num_q, )

        lines_target = torch.zeros_like(lines_pred) # (num_q, 2*num_pts)
        lines_weights = torch.zeros_like(lines_pred) # (num_q, 2*num_pts)
        
        if num_gt > 0:
            if gt_permute_idx is not None: # using permute invariant label
                # gt_permute_idx: (num_q, num_gt)
                # pos_inds: which query is positive
                # pos_gt_inds: which gt each pos pred is assigned
                # single_matched_gt_permute_idx: which permute order is matched
                single_matched_gt_permute_idx = gt_permute_idx[
                    pos_inds, pos_gt_inds
                ]
                lines_target[pos_inds] = gt_lines[pos_gt_inds, single_matched_gt_permute_idx].type(
                    lines_target.dtype) # (num_q, 2*num_pts)
            else:
                lines_target[pos_inds] = sampling_result.pos_gt_bboxes.type(
                    lines_target.dtype) # (num_q, 2*num_pts)
        
        lines_weights[pos_inds] = 1.0 # (num_q, 2*num_pts)

        # normalization
        # n = lines_weights.sum(-1, keepdim=True) # (num_q, 1)
        # lines_weights = lines_weights / n.masked_fill(n == 0, 1) # (num_q, 2*num_pts)
        # [0, ..., 0] for neg ind and [1/npts, ..., 1/npts] for pos ind

        return (labels, label_weights, lines_target, lines_weights,
                pos_inds, neg_inds, pos_gt_inds)

    # @force_fp32(apply_to=('preds', 'gts'))
    def get_targets(self, preds, gts, gt_bboxes_ignore_list=None):
        """
            Compute regression and classification targets for a batch image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                preds (dict): 
                    - lines (Tensor): shape (bs, num_queries, 2*num_points)
                    - scores (Tensor): shape (bs, num_queries, num_class_channels)
                gts (dict):
                    - class_label (list[Tensor]): tensor shape (num_gts, )
                    - lines (list[Tensor]): tensor shape (num_gts, 2*num_points)
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                tuple: a tuple containing the following targets.
                    - labels_list (list[Tensor]): Labels for all images.
                    - label_weights_list (list[Tensor]): Label weights for all \
                        images.
                    - lines_targets_list (list[Tensor]): Lines targets for all \
                        images.
                    - lines_weight_list (list[Tensor]): Lines weights for all \
                        images.
                    - num_total_pos (int): Number of positive samples in all \
                        images.
                    - num_total_neg (int): Number of negative samples in all \
                        images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        # format the inputs
        gt_labels = gts['labels']
        gt_lines = gts['lines']

        lines_pred = preds['lines']

        (labels_list, label_weights_list,
        lines_targets_list, lines_weights_list,
        pos_inds_list, neg_inds_list,pos_gt_inds_list) = multi_apply(
            self._get_target_single, preds['scores'], lines_pred,
            gt_labels, gt_lines, gt_bboxes_ignore=gt_bboxes_ignore_list)
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        new_gts = dict(
            labels=labels_list, # list[Tensor(num_q, )], length=bs
            label_weights=label_weights_list, # list[Tensor(num_q, )], length=bs, all ones
            lines=lines_targets_list, # list[Tensor(num_q, 2*num_pts)], length=bs
            lines_weights=lines_weights_list, # list[Tensor(num_q, 2*num_pts)], length=bs
        )

        return new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list

    # @force_fp32(apply_to=('preds', 'gts'))
    def loss_single(self,
                    preds,
                    gts,
                    gt_bboxes_ignore_list=None,
                    reduction='none'):
        """
            Loss function for outputs from a single decoder layer of a single
            feature level.
            Args:
                preds (dict): 
                    - lines (Tensor): shape (bs, num_queries, 2*num_points)
                    - scores (Tensor): shape (bs, num_queries, num_class_channels)
                gts (dict):
                    - class_label (list[Tensor]): tensor shape (num_gts, )
                    - lines (list[Tensor]): tensor shape (num_gts, 2*num_points)
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components for outputs from
                    a single decoder layer.
        """

        # Get target for each sample
        new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list =\
            self.get_targets(preds, gts, gt_bboxes_ignore_list)

        # Batched all data
        # for k, v in new_gts.items():
        #     new_gts[k] = torch.stack(v, dim=0) # tensor (bs, num_q, ...)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                preds['scores'][0].new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # Classification loss
        # since the inputs needs the second dim is the class dim, we permute the prediction.
        pred_scores = torch.cat(preds['scores'], dim=0) # (bs*num_q, cls_out_channles)
        cls_scores = pred_scores.reshape(-1, self.cls_out_channels) # (bs*num_q, cls_out_channels)
        cls_labels = torch.cat(new_gts['labels'], dim=0).reshape(-1) # (bs*num_q, )
        cls_weights = torch.cat(new_gts['label_weights'], dim=0).reshape(-1) # (bs*num_q, )
        
        loss_cls = self.loss_cls(
            cls_scores, cls_labels, cls_weights, avg_factor=cls_avg_factor)
        
        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        pred_lines = torch.cat(preds['lines'], dim=0)   # (bs*query, 20*2)
        gt_lines = torch.cat(new_gts['lines'], dim=0)   # (bs*query, 20*2)
        line_weights = torch.cat(new_gts['lines_weights'], dim=0)  # (bs*query, 20*2)

        assert len(pred_lines) == len(gt_lines)
        assert len(gt_lines) == len(line_weights)

        loss_reg = self.loss_reg(
            pred_lines, gt_lines, line_weights, avg_factor=num_total_pos)

        loss_dict = dict(
            cls=loss_cls,
            reg=loss_reg,
        )

        return loss_dict, pos_inds_list, pos_gt_inds_list, new_gts['lines']

    # @force_fp32(apply_to=('preds', 'gts'))
    def loss_dn_single(self,
                    preds,
                    gts,
                    dn_meta,
                    gt_bboxes_ignore_list=None,
                    reduction='none',):
        """
            Loss function for outputs from a single decoder layer of a single
            feature level.
            Args:
                preds (dict): 
                    - lines (Tensor): shape (bs, num_queries, 2*num_points)
                    - scores (Tensor): shape (bs, num_queries, num_class_channels)
                gts (dict):
                    - class_label (list[Tensor]): tensor shape (num_gts, )
                    - lines (list[Tensor]): tensor shape (num_gts, 2*num_points)
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components for outputs from
                    a single decoder layer.
        """
        group = dn_meta['num_dn_group']
        gt_labels_list = [label for label in gts['labels']]
        cls_labels = torch.cat(gt_labels_list, dim=0).repeat(group) # (bs*num_q, )
        # cls_weights = torch.ones_like(cls_labels) # (bs*num_q, )
        cls_weights = dn_meta['loss_weight']

        gt_pts_list = [gt_line[:, 0, :] for gt_line in gts['lines']]
        gt_lines = torch.cat(gt_pts_list, 0).repeat(group, 1)
        # line_weights = torch.ones_like(gt_lines)
        line_weights = dn_meta['loss_weight'][:, None].repeat(1, gt_lines.size(1))

        # construct weighted avg_factor to match with the official DETR repo
        # TODO num_total_pos这里计算多了
        # import ipdb; ipdb.set_trace()
        num_total_pos = dn_meta['loss_weight'].sum().item()
        # num_total_pos = cls_labels.size(0)
        cls_avg_factor = num_total_pos * 1.0
        
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                preds['scores'][0].new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # Classification loss
        pred_scores = torch.stack(preds['scores'], dim=0) # (bs*num_q, cls_out_channles)
        cls_scores = pred_scores[(dn_meta['known_bid'].long(), dn_meta['map_known_indice'].long())] # (bs*num_q, cls_out_channels)
        
        loss_cls = self.loss_dn_cls(
            cls_scores, cls_labels, cls_weights, avg_factor=cls_avg_factor)
        
        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        # print(self.iter, "   ", num_total_pos,  "   ", cls_scores.shape,  "   ", cls_scores.shape)
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()   # 这行有问题报错

        pred_lines = torch.stack(preds['lines'], dim=0)
        pred_lines = pred_lines[(dn_meta['known_bid'].long(), dn_meta['map_known_indice'].long())] 

        assert len(pred_lines) == len(gt_lines)
        assert len(gt_lines) == len(line_weights)

        loss_reg = self.loss_dn_reg(
            pred_lines, gt_lines, line_weights, avg_factor=num_total_pos)

        loss_dict = dict(
            dn_cls=loss_cls,
            dn_reg=loss_reg,
        )

        return loss_dict

    @force_fp32(apply_to=('gt_lines_list', 'preds_dicts'))
    def loss_dn(self,
             gts,
             preds,
             last_gts=None, 
             last_ids=None, 
             last_labels=None,
             dn_meta=None,
             gt_bboxes_ignore=None,
             reduction='mean'):
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        # Since there might have multi layer
        # dn_meta_tmp = [dn_meta for _ in range(6)]
        losses = []

        new_gts = {}
        tmp_labels_list = []
        tmp_lines_list = []

        for i, ids in enumerate(last_ids):
            if ids is not None and i in dn_meta['stream_valid']:
                tmp_labels = []
                tmp_lines = []
                tmp_idx = -1
                for j in range(ids.size(0)):
                    # import ipdb; ipdb.set_trace()
                    if dn_meta['last_valid_idx'][i][j] == False:
                        continue
                    tmp_idx += 1
                    pos = (gts['id'][i]==ids[j]).nonzero().squeeze(-1)
                    if ids[j]==-1 or len(pos)==0:
                        # import ipdb; ipdb.set_trace()
                        tmp_lines.append(last_gts[i][j])
                        tmp_labels.append(last_labels[i][j])
                        # 需要把匹配不上的gt loss置0
                        # dn_meta['loss_weight'][i][j] = 0 
                        dn_meta['loss_weight'][i][tmp_idx] = 0 
                    else:
                        tmp_lines.append(gts['lines'][i][pos[0]])
                        tmp_labels.append(gts['labels'][i][pos[0]])
                tmp_labels_list.append(torch.stack(tmp_labels, 0))
                tmp_lines_list.append(torch.stack(tmp_lines, 0))
            else:
                tmp_labels_list.append(gts['labels'][i])
                tmp_lines_list.append(gts['lines'][i])    
        dn_meta['loss_weight'] = torch.cat(dn_meta['loss_weight']).repeat(dn_meta['num_dn_group'])
        # gts['id']
        # 匹配id=0 或者 匹配id在当前batch不存在时，loss weight应该为0
        # pos_idx = -1
        # single_group = dn_meta['pad_size'] // dn_meta['num_dn_group']
        # for i, ids in enumerate(last_ids):
        #     if ids is not None:
        #         tmp_labels = []
        #         tmp_lines = []
        #         for j in range(ids.size(0)):
        #             pos_idx += 1
        #             pos = (gts['id'][i]==ids[j]).nonzero().squeeze(-1)
        #             if ids[j]==-1 or len(pos)==0:
        #                 # import ipdb; ipdb.set_trace()
        #                 tmp_lines.append(last_gts[i][j])
        #                 tmp_labels.append(last_labels[i][j])
        #                 # 需要把匹配不上的gt loss置0
        #                 for k in range(dn_meta['num_dn_group']):
        #                     dn_meta['loss_weight'][pos_idx+k*single_group] = 0 
        #                 # import ipdb; ipdb.set_trace()
        #             else:
        #                 tmp_lines.append(gts['lines'][i][pos[0]])
        #                 tmp_labels.append(gts['labels'][i][pos[0]])
        #         tmp_labels_list.append(torch.stack(tmp_labels, 0))
        #         tmp_lines_list.append(torch.stack(tmp_lines, 0))
        new_gts['labels'] = tmp_labels_list
        new_gts['lines'] = tmp_lines_list
        new_gts = [copy.deepcopy(new_gts) for _ in range(6)]

        # multi_apply(self.loss_dn_single, preds, gts, dn_meta_tmp, reduction=reduction)
        for i in range(len(preds)):
            losses.append(self.loss_dn_single(preds[i], new_gts[i], dn_meta, reduction=reduction))
        # import ipdb; ipdb.set_trace()

        # Format the losses
        loss_dict = dict()
        # loss from the last decoder layer
        for k, v in losses[-1].items():
            loss_dict[k] = v
        
        # Loss from other decoder layers
        num_dec_layer = 0
        for loss in losses[:-1]:
            for k, v in loss.items():
                loss_dict[f'd{num_dec_layer}.{k}'] = v
            num_dec_layer += 1

        return loss_dict

    @force_fp32(apply_to=('gt_lines_list', 'preds_dicts'))
    def loss(self,
             gts,
             preds,
             gt_bboxes_ignore=None,
             reduction='mean'):
        """
            Loss Function.
            Args:
                gts (list[dict]): list length: num_layers
                    dict {
                        'label': list[tensor(num_gts, )], list length: batchsize,
                        'line': list[tensor(num_gts, 2*num_points)], list length: batchsize,
                        ...
                    }
                preds (list[dict]): list length: num_layers
                    dict {
                        'lines': tensor(bs, num_queries, 2*num_points),
                        'scores': tensor(bs, num_queries, class_out_channels),
                    }
                    
                gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                    which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        # Since there might have multi layer
        losses, pos_inds_lists, pos_gt_inds_lists, gt_lines_list = multi_apply(
            self.loss_single, preds, gts, reduction=reduction)

        # Format the losses
        loss_dict = dict()
        # loss from the last decoder layer
        for k, v in losses[-1].items():
            loss_dict[k] = v
        
        # Loss from other decoder layers
        num_dec_layer = 0
        for loss in losses[:-1]:
            for k, v in loss.items():
                loss_dict[f'd{num_dec_layer}.{k}'] = v
            num_dec_layer += 1

        return loss_dict, pos_inds_lists, pos_gt_inds_lists, gt_lines_list
    
    def post_process(self, preds_dict, tokens, thr=0.0):
        lines = preds_dict['lines'] # List[Tensor(num_queries, 2*num_points)]
        bs = len(lines)
        scores = preds_dict['scores'] # (bs, num_queries, 3)
        prop_mask = preds_dict['prop_mask']

        results = []
        for i in range(bs):
            tmp_vectors = lines[i]
            tmp_prop_mask = prop_mask[i]
            num_preds, num_points2 = tmp_vectors.shape
            tmp_vectors = tmp_vectors.view(num_preds, num_points2//2, 2)
            # focal loss
            if self.loss_cls.use_sigmoid:
                tmp_scores, tmp_labels = scores[i].max(-1)
                tmp_scores = tmp_scores.sigmoid()
                pos = tmp_scores > thr
            else:
                assert self.num_classes + 1 == self.cls_out_channels
                tmp_scores, tmp_labels = scores[i].max(-1)
                bg_cls = self.cls_out_channels
                pos = tmp_labels != bg_cls

            tmp_vectors = tmp_vectors[pos]
            tmp_scores = tmp_scores[pos]
            tmp_labels = tmp_labels[pos]
            tmp_prop_mask = tmp_prop_mask[pos]

            if len(tmp_scores) == 0:
                single_result = {
                'vectors': [],
                'scores': [],
                'labels': [],
                'prop_mask': [],
                'token': tokens[i]
            }
            else:
                single_result = {
                    'vectors': tmp_vectors.detach().cpu().numpy(),
                    'scores': tmp_scores.detach().cpu().numpy(),
                    'labels': tmp_labels.detach().cpu().numpy(),
                    'prop_mask': tmp_prop_mask.detach().cpu().numpy(),
                    'token': tokens[i]
                }
            results.append(single_result)
        
        return results

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        for k, v in self.__dict__.items():
            if isinstance(v, StreamTensorMemory):
                v.train(*args, **kwargs)
    
    def eval(self):
        super().eval()
        for k, v in self.__dict__.items():
            if isinstance(v, StreamTensorMemory):
                v.eval()

    def forward(self, *args, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            return self.forward_test(*args, **kwargs)