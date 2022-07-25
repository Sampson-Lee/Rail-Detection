<<<<<<< HEAD
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 21:49:26 2017

@author: zander
"""

import os, random, sys, json
import cv2
import hand_utils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time, tqdm
from IPython import embed
import pandas as pd
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

sys.path.append("..")
import torchvision.transforms as transforms
from data.dataloader import get_train_loader
from data.constant import raildb_row_anchor
sys.path.append("..")
from utils.evaluation import mask_2_inter, LaneEval

data_root = '/home/ssd7T/lxpData/rail/dataset/'
img_transforms = transforms.Compose([
    transforms.ToTensor(),
])

resolution_src = np.float32([1280, 720])

ROI_src = np.float32([[80, 720], [400, 250], [1200, 720], [800, 250]])
ROI_dst = np.float32([(0, 720), (0, 0), (1280, 720), (1280, 0)])

def validate(data_type):
    print(data_type)
    val_loader, _ = get_train_loader(1, data_root, griding_num=200, distributed=False, num_lanes=4, mode='val', type=data_type)
    t_all = []; preds = []; gts = []
    for i, data_label in enumerate(tqdm.tqdm(val_loader)):
        # if i==5: break
        _, _, inter_labels, _, names = data_label
        frame = cv2.resize(cv2.imread(os.path.join(data_root, names[0])), (1280, 720))
        
        t1 = time.time()
        # perform perspective transform
        M, Minv = hand_utils.get_M_Minv(ROI_src, ROI_dst)
        img_warped = cv2.warpPerspective(frame.copy(), M, frame.shape[1::-1], flags=cv2.INTER_LINEAR)

        # get the thresholded binary image
        img_bin = hand_utils.thresholding(img_warped)

        # perform detection
        left_fit, right_fit, [lefty, leftx], [righty, rightx] = hand_utils.find_line(img_bin)
        t2 = time.time()
        t_all.append(t2 - t1)

        # draw the detected laneline and the information
        image = Image.fromarray(frame)
        img_draw, warp_draw = hand_utils.draw_area(image, img_bin, ROI_src, Minv, left_fit, right_fit, lefty, leftx, righty, rightx)
        # plt.imshow(img_draw)
        # plt.pause(2)
        vis_path = os.path.join(data_root, 'hand-crafted/vis', names[0]).replace('pic', data_type)
        if not os.path.exists(os.path.dirname(vis_path)): os.makedirs(os.path.dirname(vis_path))
        cv2.imwrite(vis_path, img_draw)
        pred_path = os.path.join(data_root, 'hand-crafted/pred', names[0]).replace('pic', data_type)
        if not os.path.exists(os.path.dirname(pred_path)): os.makedirs(os.path.dirname(pred_path))
        cv2.imwrite(pred_path, warp_draw)

        # evaluation
        pred = [mask_2_inter(warp_draw, raildb_row_anchor)]
        gt = inter_labels.cpu().numpy()
        preds.append(pred)
        gts.append(gt)

    preds = np.concatenate(preds); gts = np.concatenate(gts)
    for i in range(1,21):
        LaneEval.pixel_thresh = i
        res = LaneEval.bench_all(preds, gts, raildb_row_anchor)
        res = json.loads(res)
        for r in res:
            print(r['name'], r['value']) 

    print('average time:', np.mean(t_all) / 1)
    print('average fps:',1 / np.mean(t_all))

    print('fastest time:', min(t_all) / 1)
    print('fastest fps:',1 / min(t_all))

    print('slowest time:', max(t_all) / 1)
    print('slowest fps:',1 / max(t_all))

validate('all')
# for i in ['all','sun','rain','night','line','cross','curve','slope','near','far']:
#     data_type = i
=======
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 21:49:26 2017

@author: zander
"""

import os, random, sys, json
import cv2
import hand_utils
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time, tqdm
from IPython import embed
import pandas as pd
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

sys.path.append("..")
import torchvision.transforms as transforms
from data.dataloader import get_train_loader
from data.constant import raildb_row_anchor
sys.path.append("..")
from utils.evaluation import mask_2_inter, LaneEval

data_root = '/home/ssd7T/lxpData/rail/dataset/'
img_transforms = transforms.Compose([
    transforms.ToTensor(),
])

resolution_src = np.float32([1280, 720])

ROI_src = np.float32([[80, 720], [400, 250], [1200, 720], [800, 250]])
ROI_dst = np.float32([(0, 720), (0, 0), (1280, 720), (1280, 0)])

def validate(data_type):
    print(data_type)
    val_loader, _ = get_train_loader(1, data_root, griding_num=200, distributed=False, num_lanes=4, mode='val', type=data_type)
    t_all = []; preds = []; gts = []
    for i, data_label in enumerate(tqdm.tqdm(val_loader)):
        # if i==5: break
        _, _, inter_labels, _, names = data_label
        frame = cv2.resize(cv2.imread(os.path.join(data_root, names[0])), (1280, 720))
        
        t1 = time.time()
        # perform perspective transform
        M, Minv = hand_utils.get_M_Minv(ROI_src, ROI_dst)
        img_warped = cv2.warpPerspective(frame.copy(), M, frame.shape[1::-1], flags=cv2.INTER_LINEAR)

        # get the thresholded binary image
        img_bin = hand_utils.thresholding(img_warped)

        # perform detection
        left_fit, right_fit, [lefty, leftx], [righty, rightx] = hand_utils.find_line(img_bin)
        t2 = time.time()
        t_all.append(t2 - t1)

        # draw the detected laneline and the information
        image = Image.fromarray(frame)
        img_draw, warp_draw = hand_utils.draw_area(image, img_bin, ROI_src, Minv, left_fit, right_fit, lefty, leftx, righty, rightx)
        # plt.imshow(img_draw)
        # plt.pause(2)
        vis_path = os.path.join(data_root, 'hand-crafted/vis', names[0]).replace('pic', data_type)
        if not os.path.exists(os.path.dirname(vis_path)): os.makedirs(os.path.dirname(vis_path))
        cv2.imwrite(vis_path, img_draw)
        pred_path = os.path.join(data_root, 'hand-crafted/pred', names[0]).replace('pic', data_type)
        if not os.path.exists(os.path.dirname(pred_path)): os.makedirs(os.path.dirname(pred_path))
        cv2.imwrite(pred_path, warp_draw)

        # evaluation
        pred = [mask_2_inter(warp_draw, raildb_row_anchor)]
        gt = inter_labels.cpu().numpy()
        preds.append(pred)
        gts.append(gt)

    preds = np.concatenate(preds); gts = np.concatenate(gts)
    for i in range(1,21):
        LaneEval.pixel_thresh = i
        res = LaneEval.bench_all(preds, gts, raildb_row_anchor)
        res = json.loads(res)
        for r in res:
            print(r['name'], r['value']) 

    print('average time:', np.mean(t_all) / 1)
    print('average fps:',1 / np.mean(t_all))

    print('fastest time:', min(t_all) / 1)
    print('fastest fps:',1 / min(t_all))

    print('slowest time:', max(t_all) / 1)
    print('slowest fps:',1 / max(t_all))

validate('all')
# for i in ['all','sun','rain','night','line','cross','curve','slope','near','far']:
#     data_type = i
>>>>>>> d175ba8a15a74cff363e8da114147f44311bfb42
#     validate(data_type)