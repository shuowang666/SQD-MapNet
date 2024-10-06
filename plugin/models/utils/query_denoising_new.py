# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.core import bbox_xyxy_to_cxcywh
from .utils import inverse_sigmoid
import math
import numpy as np


def rotate_matrix(theta):
    cos = np.cos(theta)
    sin = np.sin(theta)

    return np.stack((np.stack((cos, -sin), 1), np.stack((sin, cos), 1)), 1)


class CdnQueryGenerator:
    def __init__(self,
                 hidden_dim=256,
                 num_classes=0,
                 num_queries=0,
                 noise_scale=dict(label=0.5, box=0.4, pt=0.0),
                 group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=None),
                 bev_h=200, bev_w=100,
                 pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
                 voxel_size=[0.3, 0.3],
                 num_pts_per_vec=20,
                 rotate_range=0.0,
                 froze_class=None,
                 class_spesific=None,
                 noise_decay=False,
                 **kwargs):
        self.num_queries = num_queries
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.label_noise_scale = noise_scale['label']
        self.box_noise_scale = noise_scale['box']
        self.pt_noise_scale = noise_scale['pt']
        self.dynamic_dn_groups = group_cfg.get('dynamic', False)
        if self.dynamic_dn_groups:
            assert 'num_dn_queries' in group_cfg, \
                'num_dn_queries should be set when using ' \
                'dynamic dn groups'
            self.num_dn = group_cfg['num_dn_queries']
        else:
            assert 'num_groups' in group_cfg, \
                'num_groups should be set when using ' \
                'static dn groups'
            self.num_dn = group_cfg['num_groups']
        assert isinstance(self.num_dn, int) and self.num_dn >= 1, \
            f'Expected the num in group_cfg to have type int. ' \
            f'Found {type(self.num_dn)} '
        self.pc_range = pc_range
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.voxel_size = voxel_size
        self.num_pts_per_vec = num_pts_per_vec
        self.rotate_range = rotate_range
        self.froze_class = froze_class
        self.class_spesific = class_spesific
        self.noise_decay = noise_decay

    def get_num_groups(self, group_queries=None):
        """
        Args:
            group_queries (int): Number of dn queries in one group.
        """
        if self.dynamic_dn_groups:
            assert group_queries is not None, \
                'group_queries should be provided when using ' \
                'dynamic dn groups'
            if group_queries == 0:
                num_groups = 1
            else:
                num_groups = self.num_dn // group_queries
        else:
            num_groups = self.num_dn
        if num_groups < 1:
            num_groups = 1
        return int(num_groups)

    def __call__(self,
                 gt_bboxes,
                 gt_pts,
                 gt_labels=None,
                 label_enc=None,
                 prop_query_embedding=None,
                 noise_scale_list=None):
        """

        Args:
            gt_bboxes (List[Tensor]): List of ground truth bboxes
                of the image, shape of each (num_gts, 4).
            gt_labels (List[Tensor]): List of ground truth labels
                of the image, shape of each (num_gts,), if None,

        Returns:
            TODO
        """
        if gt_labels is not None:
            assert len(gt_bboxes) == len(gt_labels), \
                f'the length of provided gt_labels ' \
                f'{len(gt_labels)} should be equal to' \
                f' that of gt_bboxes {len(gt_bboxes)}'

        batch_size = len(gt_bboxes)
        device = gt_bboxes[0].device

        # convert bbox
        gt_bboxes_list = []
        gt_pts_list = []
        loss_weight = []
        neglect_pos = []

        line_pos = []
        bound_pos = []
        ped_pos = []

        for label, bboxes, pts in zip(gt_labels, gt_bboxes, gt_pts):
            if self.froze_class is None:
                loss_weight.append(1 - ((bboxes[:, 0]==bboxes[:, 2]) | (bboxes[:, 1]==bboxes[:, 3])).long())
            else:
                loss_weight.append(1 - ((bboxes[:, 0]==bboxes[:, 2]) | (bboxes[:, 1]==bboxes[:, 3]) | (label!=self.froze_class)).long())  # 只计算某个类别的dn loss

            neglect_pos.append(((bboxes[:, 0]==bboxes[:, 2]) | (bboxes[:, 1]==bboxes[:, 3])).nonzero().squeeze(-1))

            pts_ = ((pts - bboxes[:, None, :2]) / (bboxes[:, None, 2:] - bboxes[:, None, :2])).clamp(min=0.0, max=1.0)
            gt_pts_list.append(pts_)

            bboxes_normalized = bbox_xyxy_to_cxcywh(bboxes)
            gt_bboxes_list.append(bboxes_normalized)

            # 保存不同类别线的位置
            line_pos.append((label == 1).long())
            ped_pos.append((label == 0).long())
            bound_pos.append((label == 2).long())

        known = [torch.ones(b.shape[0]).int() for b in gt_bboxes]
        known_num = [sum(k) for k in known]

        num_groups = self.get_num_groups(int(max(known_num)))
        assert num_groups >= 1

        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat(gt_labels)
        boxes = torch.cat(gt_bboxes_list)
        # choice one: 
        pt = torch.cat(gt_pts_list) 

        batch_idx = torch.cat([torch.full_like(torch.ones(t.shape[0]).long(), i) for i, t in enumerate(gt_bboxes)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(num_groups, 1).view(-1)
        known_labels = labels.repeat(num_groups, 1).view(-1)
        known_bid = batch_idx.repeat(num_groups, 1).view(-1)
        known_bboxs = boxes.repeat(num_groups, 1)
        known_pts = pt.repeat(num_groups, 1, 1)
        known_labels_expand = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if noise_scale_list is not None:
            noise_scale_list = torch.cat(noise_scale_list).repeat(num_groups)

        # 
        if self.class_spesific is not None:
            line_pos = torch.cat(line_pos).repeat(num_groups)
            ped_pos = torch.cat(ped_pos).repeat(num_groups)
            bound_pos = torch.cat(bound_pos).repeat(num_groups)

        single_pad = int(max(known_num)) 

        pad_size = int(single_pad * num_groups)
        if self.box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, : 2] = \
                known_bboxs[:, : 2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = \
                known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

            diff = torch.zeros_like(known_bboxs)
            diff[:, :2] = known_bboxs[:, 2:] / 2
            diff[:, 2:] = known_bboxs[:, 2:] / 2

            rand_sign = torch.randint_like(
                known_bboxs, low=0, high=2, dtype=torch.float32)
            rand_sign = rand_sign * 2.0 - 1.0
            rand_part = torch.rand_like(known_bboxs)
            rand_part *= rand_sign
            add = torch.mul(rand_sign, diff).to(device)

            if self.class_spesific:
                # import ipdb; ipdb.set_trace()
                noise = torch.mul(rand_part, diff).to(device)
                known_bbox_ += (noise*line_pos[:, None]*self.class_spesific[1] + noise*ped_pos[:, None]*self.class_spesific[0] + \
                                noise*bound_pos[:, None]*self.class_spesific[2])
            else:
                if self.noise_decay:
                    known_bbox_ += torch.mul(rand_part, diff).to(device) * self.box_noise_scale * noise_scale_list[:, None]
                else:
                    known_bbox_ += torch.mul(rand_part, diff).to(device) * self.box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)

            known_bbox_expand[:, :2] = \
                (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = \
                known_bbox_[:, 2:] - known_bbox_[:, :2]
        else:
            known_bbox_ = torch.zeros_like(known_bboxs)
            known_bbox_[:, : 2] = \
                known_bboxs[:, : 2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = \
                known_bboxs[:, :2] + known_bboxs[:, 2:] / 2

        if self.pt_noise_scale > 0:
            rand_sign = (torch.rand_like(known_pts) * 2.0 - 1.0) / 20
            known_pts += rand_sign.to(device) * self.pt_noise_scale
            known_pts = known_pts.clamp(min=0.0, max=1.0)

        # Rotate
        if self.rotate_range > 0:
            random_theta = (np.random.rand(known_bbox_.size(0)) * 2 - 1) * self.rotate_range * math.pi / 180
            R_matrix = rotate_matrix(random_theta)
            known_refers = (known_bbox_[:, None, :2] + known_pts * known_bbox_expand[:, None, 2:] - known_bbox_expand[:, None, :2]).permute(0, 2, 1)
            known_refers = torch.bmm(torch.from_numpy(R_matrix).to(torch.float32).to(device), known_refers).permute(0, 2, 1)
            known_refers = known_refers + known_bbox_expand[:, None, :2]
        else:
            known_refers = known_bbox_[:, None, :2] + known_pts * known_bbox_expand[:, None, 2:]

        if self.label_noise_scale > 0:
            p = torch.rand_like(known_labels_expand.float())
            chosen_indice = torch.nonzero(
                p < (self.label_noise_scale * 0.5)).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, self.num_classes)
            known_labels_expand.scatter_(0, chosen_indice, new_label)

        m = known_labels_expand.long().to(device)
        input_label_embed = label_enc(m)
        input_bbox_embed = known_bbox_expand
        padding_label = torch.zeros(pad_size, self.hidden_dim).to(device)
        padding_bbox = torch.zeros(pad_size, 4).to(device)
        padding_pts = torch.zeros(pad_size, self.num_pts_per_vec, 2).to(device)
        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)
        input_query_pts = padding_pts.repeat(batch_size, 1, 1, 1)
        denoise_refers = padding_pts.repeat(batch_size, 1, 1, 1)

        map_known_indice = torch.tensor([]).to(device)
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])
            map_known_indice = torch.cat([
                map_known_indice + single_pad * i
                for i in range(num_groups)
            ]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(),
                               map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(),
                              map_known_indice)] = input_bbox_embed
            input_query_pts[(known_bid.long(),
                              map_known_indice)] = known_pts
            denoise_refers[(known_bid.long(),
                              map_known_indice)] = known_refers

        if prop_query_embedding is not None:
            tgt_size = pad_size + self.num_queries + prop_query_embedding.size(1)
        else:
            tgt_size = pad_size + self.num_queries
        attn_mask = torch.ones(tgt_size, tgt_size).to(device) < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(num_groups):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1),
                          single_pad * (i + 1):pad_size] = True
            if i == num_groups - 1:
                attn_mask[single_pad * i:single_pad *
                          (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1),
                          single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad *
                          (i + 1), :single_pad * i] = True

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': num_groups,
            # 'post_dn': post_dn,
            'known_bid': known_bid.long(),
            'map_known_indice': map_known_indice,
            'loss_weight': loss_weight,
        }

        # 去掉完全的直线，分母为0
        for i, pos in enumerate(neglect_pos):
            if len(pos) != 0:
                for j in range(num_groups):
                    denoise_refers[i][pos+single_pad * j] = gt_pts[i][pos]

        return input_query_label, input_query_bbox, input_query_pts, attn_mask, dn_meta, denoise_refers
