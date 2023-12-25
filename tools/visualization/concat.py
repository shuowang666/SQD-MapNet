# 把不同方法的图片concat到一张对比

import cv2
import numpy as np
import os
from tqdm import tqdm


root_path = '/data/code/StreamMapNet/final_res'
single_path = os.path.join(root_path, 'singleframe')
streammapnet_path = os.path.join(root_path, 'streammapnet')
streamdn_path = os.path.join(root_path, 'streamdn')
dst_path = os.path.join(root_path, 'duibi')

all_dir = os.listdir(single_path)
# pbar = tqdm(all_dir)
for d in tqdm(all_dir):
    img_single = cv2.imread(os.path.join(single_path, d, 'pred.jpg'))
    gt = cv2.imread(os.path.join(single_path, d, 'gt.jpg'))
    img_streammapnet = cv2.imread(os.path.join(streammapnet_path, d, 'pred.jpg'))
    img_streamdn = cv2.imread(os.path.join(streamdn_path, d, 'pred.jpg'))

    img = np.vstack((gt, img_single, img_streammapnet, img_streamdn))
    cv2.putText(img, "GT", (10,200), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,0), 8)
    cv2.putText(img, "Single", (10,1100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,0), 8)
    cv2.putText(img, "SteamMapNet", (10,2100), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,0), 8)
    cv2.putText(img, "DN", (10,3000), cv2.FONT_HERSHEY_SIMPLEX, 4, (0,0,0), 8)
    cv2.imwrite(os.path.join(dst_path, d+'.jpg'), img)
    # import ipdb; ipdb.set_trace()