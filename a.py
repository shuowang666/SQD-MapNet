
# import torch

# la = torch.load('/data/code/StreamMapNet/ckpts/fcos3d_vovnet_imgbackbone-remapped.pth')
# new_la = dict()
# for k in la.keys():
#     # import ipdb; ipdb.set_trace()
#     if 'neck' in k or 'fpn' in k:
#         import ipdb; ipdb.set_trace()
#     if k.split('.')[0] == 'img_backbone':
#         new_la['backbone.'+k] = la[k] 
#     else:
#         new_la[k] = la[k] 
        
# torch.save(new_la, '/data/code/StreamMapNet/ckpts/fcos3d_vovnet_imgbackbone-remapped_convert.pth')

# import pickle

# with open('/data/code/StreamMapNet/datasets/nuScenes/nuscenes_map_infos_val.pkl', 'rb') as f:
# # with open('/data/code/StreamMapNet/datasets/nuScenes/nuscenes_infos_temporal_val.pkl', 'rb') as f:
#     fi = pickle.load(f)
#     import ipdb; ipdb.set_trace()

import shutil
import os
import numpy as np
import cv2

path = '/data/code/StreamMapNet/final_res'
# select = ['singleframe', 'streammapnet', 'streamdn']
# name = 'n015-2018-07-11-11-54-16+0800_1531281445862460'
# name = 'n015-2018-07-18-11-41-49+0800_1531885321012467'
name = 'n015-2018-07-16-11-49-16+0800_1531713379362460'

# for s in select:
    # shutil.copyfile(os.path.join(path, s, name, 'pred.jpg'), os.path.join(path, 'paper_vis', s+'_pred3.jpg'))
    # shutil.copyfile(os.path.join(path, s, name, 'gt.jpg'), os.path.join(path, 'paper_vis', 'gt3.jpg'))

cam_img_name = ['CAM_FRONT_LEFT.jpg', 'CAM_FRONT.jpg', 'CAM_FRONT_RIGHT.jpg']
cam_img_name2 = ['CAM_BACK_LEFT.jpg', 'CAM_BACK.jpg', 'CAM_BACK_RIGHT.jpg']
tmp = []
for c in cam_img_name:
    tmp.append(cv2.imread(os.path.join(path, 'streammapnet', name, c)))
front_img = np.hstack(tmp)

tmp = []
for c in cam_img_name2:
    tmp.append(cv2.imread(os.path.join(path, 'streammapnet', name, c)))
back_img = np.hstack(tmp)
img = np.vstack((front_img, back_img))
cv2.imwrite(os.path.join(path, 'paper_vis', 'image3.jpg'), img)
# import ipdb; ipdb.set_trace()