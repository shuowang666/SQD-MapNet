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
                 pseudo_w=4/30,
                 wh_ratio=20.0,
                 rotate_range=0.0,
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
        self.pseudo_w = pseudo_w
        self.wh_ratio = wh_ratio
        self.rotate_range = rotate_range

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
                 label_enc=None,):
        """

        Args:
            gt_bboxes (List[Tensor]): List of ground truth bboxes
                of the image, shape of each (num_gts, 4).
            gt_labels (List[Tensor]): List of ground truth labels
                of the image, shape of each (num_gts,), if None,
                TODO:noisy_label would be None.

        Returns:
            TODO
        """
        # TODO: temp only support for CDN
        # TODO: temp assert gt_labels is not None and label_enc is not None
        if gt_labels is not None:
            assert len(gt_bboxes) == len(gt_labels), \
                f'the length of provided gt_labels ' \
                f'{len(gt_labels)} should be equal to' \
                f' that of gt_bboxes {len(gt_bboxes)}'
        # assert gt_labels is not None and label_enc is not None # TODO: adjust args
        batch_size = len(gt_bboxes)
        device = gt_bboxes[0].device

        # convert bbox
        gt_bboxes_list = []
        gt_pts_list = []
        # refer_list = []
        loss_weight = []
        neglect_pos = []

        for bboxes, pts in zip(gt_bboxes, gt_pts):
            # import ipdb; ipdb.set_trace()
            loss_weight.append(1 - ((bboxes[:, 0]==bboxes[:, 2]) | (bboxes[:, 1]==bboxes[:, 3])).long())
            neglect_pos.append(((bboxes[:, 0]==bboxes[:, 2]) | (bboxes[:, 1]==bboxes[:, 3])).nonzero().squeeze(-1))

            pts_ = ((pts - bboxes[:, None, :2]) / (bboxes[:, None, 2:] - bboxes[:, None, :2])).clamp(min=0.0, max=1.0)
            # pts_ = ((pts-pts.new_tensor([-15, -30]))/pts.new_tensor([30, 60])).clamp(min=0.0, max=1.0)
            gt_pts_list.append(pts_)
            # refer_list.append(refer)

            # bboxes = ((bboxes - bboxes.new_tensor([-15, -30, -15, -30])) / bboxes.new_tensor([30, 60, 30, 60])).clamp(min=0.0, max=1.0)
            bboxes_normalized = bbox_xyxy_to_cxcywh(bboxes)

            # bboxes[:, ::2] = ((bboxes[:, ::2]-self.pc_range[0]) / self.voxel_size[0]).floor()
            # bboxes[:, 1::2] = ((bboxes[:, 1::2]-self.pc_range[1]) / self.voxel_size[1]).floor()
            # factor = bboxes.new_tensor([self.bev_w, self.bev_h, self.bev_w, self.bev_h]).unsqueeze(0)
            # bboxes_normalized = bbox_xyxy_to_cxcywh(bboxes) / factor
            gt_bboxes_list.append(bboxes_normalized)

        known = [torch.ones(b.shape[0]).int() for b in gt_bboxes]
        known_num = [sum(k) for k in known]

        num_groups = self.get_num_groups(int(max(known_num)))
        # num_groups = 1
        assert num_groups >= 1

        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat(gt_labels)
        boxes = torch.cat(gt_bboxes_list)
        # choice one: 选择point gt位置
        pt = torch.cat(gt_pts_list) 
        # choice two: 选择矩形框斜对角线
        # pt = torch.cat((torch.linspace(0, 1, 20).unsqueeze(-1), torch.linspace(0, 1, 20).unsqueeze(-1)), -1).repeat(labels.size(0), 1, 1).to(device)
        
        # refers = torch.cat(refer_list)

        batch_idx = torch.cat([torch.full_like(torch.ones(t.shape[0]).long(), i) for i, t in enumerate(gt_bboxes)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)

        known_indice = known_indice.repeat(num_groups, 1).view(-1)
        known_labels = labels.repeat(num_groups, 1).view(-1)
        known_bid = batch_idx.repeat(num_groups, 1).view(-1)
        known_bboxs = boxes.repeat(num_groups, 1)
        known_pts = pt.repeat(num_groups, 1, 1)
        # known_refers = refers.repeat(num_groups, 1, 1)
        known_labels_expand = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()
        loss_weight = torch.cat(loss_weight).repeat(num_groups)

        if self.label_noise_scale > 0:
            p = torch.rand_like(known_labels_expand.float())
            chosen_indice = torch.nonzero(
                p < (self.label_noise_scale * 0.5)).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, self.num_classes)
            known_labels_expand.scatter_(0, chosen_indice, new_label)
        single_pad = int(max(known_num))  # TODO

        # plot时新加，不plot时注释掉
        # if self.pt_noise_scale > 0:
        #     rand_sign = (torch.rand_like(known_pts) * 2.0 - 1.0) / 20
        #     known_pts_ = known_pts + rand_sign.to(device) * self.pt_noise_scale
        #     known_pts_ = known_pts_.clamp(min=0.0, max=1.0)

        # pad_size应该表示pos/neg应该pad的数目
        pad_size = int(single_pad * num_groups)
        if self.box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs)
            # 又变回x, y, x, y形式
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

            # 原始不加筛选
            known_bbox_ += torch.mul(rand_part, diff).to(device) * self.box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)

            # 这里面是具体筛选代码
            # wh = known_bbox_[:, 2:] - known_bbox_[:, :2]
            # select_idx = ((diff[:, 0]<5/60) & (wh[:, 1]/wh[:, 0]>self.wh_ratio))
            # rand_sign_x = torch.randint_like(add[select_idx, :1], low=0, high=2, dtype=torch.float32)
            # rand_sign_x = rand_sign_x * 2.0 - 1.0
            # rand_part_x = torch.rand_like(add[select_idx, :1])
            # add[select_idx, ::2] += (self.pseudo_w * rand_part_x * rand_sign_x).repeat(1, 2)

            # select_idx = ((diff[:, 1]<5/120) & (wh[:, 0]/wh[:, 1]>self.wh_ratio))
            # rand_sign_y = torch.randint_like(add[select_idx, :1], low=0, high=2, dtype=torch.float32)
            # rand_sign_y = rand_sign_y * 2.0 - 1.0
            # rand_part_y = torch.rand_like(add[select_idx, :1])
            # add[select_idx, 1::2] += (self.pseudo_w / 2 * rand_part_y * rand_sign_y).repeat(1, 2)
            
            # 筛选时用
            # known_bbox_ = known_bbox_ + add * self.box_noise_scale
            # known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0)
            # 画图用
            # known_bbox_2 = known_bbox_ + add * self.box_noise_scale
            # known_bbox_2 = known_bbox_2.clamp(min=0.0, max=1.0)

            # # plot
            # import cv2 
            # import numpy as np
            # # 分别得到扰动前后点的位置
            # pt_before_noise = known_bbox_[:, None, :2] + known_pts * wh[:, None, :]
            # pt_before_noise[:, :, 0] *= 300
            # pt_before_noise[:, :, 1] *= 600
            # wh_ = known_bbox_2[:, 2:] - known_bbox_2[:, :2]
            # pt_after_noise = known_bbox_2[:, None, :2] + known_pts_ * wh_[:, None, :]
            # pt_after_noise[:, :, 0] *= 300
            # pt_after_noise[:, :, 1] *= 600

            # image1 = np.zeros((600, 300, 3))
            # image2 = np.zeros((600, 300, 3))
            # box_1 = known_bbox_[:known_num[0].item()].cpu().numpy()
            # label_1 = known_labels[:known_num[0].item()].cpu().numpy()
            # box_1[:, ::2] *= 300
            # box_1[:, 1::2] *= 600
            # box_1 = box_1.astype(int)
            # box_1_ = known_bbox_2[:known_num[0].item()].cpu().numpy()
            # box_1_[:, ::2] *= 300
            # box_1_[:, 1::2] *= 600
            # box_1_ = box_1_.astype(int)
            # pt_before_1 = pt_before_noise[:known_num[0].item()].cpu().numpy().astype(int)
            # pt_after_1 = pt_after_noise[:known_num[0].item()].cpu().numpy().astype(int)

            # for l in range(box_1.shape[0]):
            #     image1 = np.zeros((600, 300, 3))
            #     cv2.rectangle(image1, box_1[l, :2], box_1[l, 2:], (0, 0, 255))
            #     for k in range(20):
            #         cv2.circle(image1, pt_before_1[l, k], 2, (0, 0, 255), -1)
            #     cv2.rectangle(image1, box_1_[l, :2], box_1_[l, 2:], (255, 0, 0))
            #     for k in range(20):
            #         cv2.circle(image1, pt_after_1[l, k], 2, (255, 0, 0), -1)
            #     cv2.putText(image1, str(label_1[l]), (150, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            #     cv2.imwrite('vis/origin/image1_'+str(l)+'.jpg', image1)
            # box_2 = known_bbox_[known_num[0].item():known_num[0].item()+known_num[1].item()].cpu().numpy()
            # label_2 = known_labels[known_num[0].item():known_num[0].item()+known_num[1].item()].cpu().numpy()
            # box_2[:, ::2] *= 300
            # box_2[:, 1::2] *= 600
            # box_2 = box_2.astype(int)
            # box_2_ = known_bbox_2[known_num[0].item():known_num[0].item()+known_num[1].item()].cpu().numpy()
            # box_2_[:, ::2] *= 300
            # box_2_[:, 1::2] *= 600
            # box_2_ = box_2_.astype(int)
            # pt_before_2 = pt_before_noise[known_num[0].item():known_num[0].item()+known_num[1].item()].cpu().numpy().astype(int)
            # pt_after_2 = pt_after_noise[known_num[0].item():known_num[0].item()+known_num[1].item()].cpu().numpy().astype(int)
            # for l in range(box_2.shape[0]):
            #     image2 = np.zeros((600, 300, 3))
            #     cv2.rectangle(image2, box_2[l, :2], box_2[l, 2:], (0, 0, 255))
            #     for k in range(20):
            #         cv2.circle(image2, pt_before_2[l, k], 2, (0, 0, 255), -1)
            #     cv2.rectangle(image2, box_2_[l, :2], box_2_[l, 2:], (255, 0, 0))
            #     for k in range(20):
            #         cv2.circle(image2, pt_after_2[l, k], 2, (255, 0, 0), -1)
            #     cv2.putText(image2, str(label_2[l]), (150, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            #     cv2.imwrite('vis/origin/image2_'+str(l)+'.jpg', image2)

            # import ipdb; ipdb.set_trace()

            # 又转回x,y,w,h形式
            known_bbox_expand[:, :2] = \
                (known_bbox_[:, :2] + known_bbox_[:, 2:]) / 2
            known_bbox_expand[:, 2:] = \
                known_bbox_[:, 2:] - known_bbox_[:, :2]
        else:
            known_bbox_ = torch.zeros_like(known_bboxs)
            # 又变回x, y, x, y形式
            known_bbox_[:, : 2] = \
                known_bboxs[:, : 2] - known_bboxs[:, 2:] / 2
            known_bbox_[:, 2:] = \
                known_bboxs[:, :2] + known_bboxs[:, 2:] / 2
            
        if self.pt_noise_scale > 0:
            rand_sign = (torch.rand_like(known_pts) * 2.0 - 1.0) / 20
            known_pts += rand_sign.to(device) * self.pt_noise_scale
            known_pts = known_pts.clamp(min=0.0, max=1.0)

        # 进行旋转
        if self.rotate_range > 0:
            random_theta = (np.random.rand(known_bbox_.size(0)) * 2 - 1) * self.rotate_range * math.pi / 180
            R_matrix = rotate_matrix(random_theta)
            known_refers = (known_bbox_[:, None, :2] + known_pts * known_bbox_expand[:, None, 2:] - known_bbox_expand[:, None, :2]).permute(0, 2, 1)
            known_refers = torch.bmm(torch.from_numpy(R_matrix).to(torch.float32).to(device), known_refers).permute(0, 2, 1)
            known_refers = known_refers + known_bbox_expand[:, None, :2]

            # plot
            # import cv2
            # pt_before_1 = known_bbox_[:, None, :2] + known_pts * known_bbox_expand[:, None, 2:]
            # pt_before_1[:, :, 0] *= 300
            # pt_before_1[:, :, 1] *= 600
            # pt_before_1 = pt_before_1.cpu().numpy().astype(int)
            # pt_after_1 = known_refers.cpu().numpy()
            # pt_after_1[:, :, 0] *= 300
            # pt_after_1[:, :, 1] *= 600
            # pt_after_1 = pt_after_1.astype(int)
            # for l in range(known_refers.shape[0]):
            #     image1 = np.zeros((600, 300, 3))
            #     # cv2.rectangle(image1, box_1[l, :2], box_1[l, 2:], (0, 0, 255))
            #     for k in range(20):
            #         cv2.circle(image1, pt_before_1[l, k], 2, (0, 0, 255), -1)
            #     # cv2.rectangle(image1, box_1_[l, :2], box_1_[l, 2:], (255, 0, 0))
            #     for k in range(20):
            #         cv2.circle(image1, pt_after_1[l, k], 2, (255, 0, 0), -1)
            #     # cv2.putText(image1, str(label_1[l]), (150, 300), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            #     cv2.imwrite('vis/rotate/image1_'+str(l)+'.jpg', image1)
        else:
            known_refers = known_bbox_[:, None, :2] + known_pts * known_bbox_expand[:, None, 2:]
        
        # 鸡生蛋，蛋生鸡gt
        # known_refers = known_bbox_[:, None, :2] + known_pts * known_bbox_expand[:, None, 2:]


        # TODO 这里没有取 inverse_sigmoid
        m = known_labels_expand.long().to(device)
        input_label_embed = label_enc(m)
        # input_bbox_embed = inverse_sigmoid(known_bbox_expand, eps=1e-3)
        input_bbox_embed = known_bbox_expand
        padding_label = torch.zeros(pad_size, self.hidden_dim).to(device)
        padding_bbox = torch.zeros(pad_size, 4).to(device)
        padding_pts = torch.zeros(pad_size, 20, 2).to(device)
        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)
        input_query_pts = padding_pts.repeat(batch_size, 1, 1, 1)
        denoise_refers = padding_pts.repeat(batch_size, 1, 1, 1)

        # post_dn = []
        # for num in known_num:
        #     single_id = torch.cat([torch.tensor(range(num)) + i*single_pad for i in range(num_groups)])
        #     post_dn.append(single_id.long())

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

        # 设置斜对角线，而非gt point的相对位置
        # input_query_pts = torch.cat((torch.linspace(0, 1, 20).unsqueeze(-1), torch.linspace(0, 1, 20).unsqueeze(-1)), -1).unsqueeze(0).repeat(batch_size, input_query_bbox.size(1), 1, 1).to(device)

        dn_meta = {
            'pad_size': pad_size,
            'num_dn_group': num_groups,
            # 'post_dn': post_dn,
            'known_bid': known_bid.long(),
            'map_known_indice': map_known_indice,
            'loss_weight': loss_weight,
        }
        # if (loss_weight==0).sum() != 0:
        #     import ipdb; ipdb.set_trace()

        for i, pos in enumerate(neglect_pos):
            if len(pos) != 0:
                for j in range(num_groups):
                    denoise_refers[i][pos+single_pad * j] = gt_pts[i][pos]

        # import ipdb; ipdb.set_trace()

        return input_query_label, input_query_bbox, input_query_pts, attn_mask, dn_meta, denoise_refers
