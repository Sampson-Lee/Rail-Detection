import torch
from PIL import Image
import os
import pdb
import numpy as np
import cv2
import random
import csv
import pandas as pd

import data.mytransforms as mytransforms
from data.constant import raildb_row_anchor
# import mytransforms as mytransforms
# from constant import raildb_row_anchor

import torchvision.transforms as transforms
from IPython import embed
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def loader_func(path):
    return Image.open(path).resize((1280,720), Image.NEAREST)

class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, meta_file, img_transform = None, type='all'):
        super(LaneTestDataset, self).__init__()
        self.data_path = data_path
        self.meta_file = meta_file
        self.img_transform = img_transform
        self.type = type

        pd_reader = pd.read_csv(meta_file)
        random.seed(2022)
        rd_ind = list(range(len(pd_reader)))
        random.shuffle(rd_ind)
        self.pd_reader = pd_reader.reindex(index=rd_ind)
    
        len_image = int(len(self.pd_reader)*0.2)
        self.pd_reader = self.pd_reader.loc[len(self.pd_reader)-len_image:]
        if self.type != 'all': self.img_list = list(self.pd_reader['name'][self.pd_reader[self.type].astype(bool)])
        else: self.img_list = list(self.pd_reader['name'])
        print(str(self.type) + ' has {} testing'.format(len(self.img_list)))

    def __getitem__(self, index):
        # parse label and image name
        img_name = self.img_list[index]
        jpeg_name = 'pic/' + img_name[:-12] + '/' + img_name

        # read label and image
        img_path = os.path.join(self.data_path, jpeg_name)
        img = loader_func(img_path)
        
        img = self.img_transform(img)

        return img, jpeg_name

    def __len__(self):
        return len(self.img_list)


class LaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, meta_file, img_transform = None, target_transform = None, simu_transform = None, 
                    griding_num = 100, row_anchor = None, num_lanes = 4, mode='train', type='all'):
        super(LaneClsDataset, self).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.simu_transform = simu_transform
        self.data_path = data_path
        self.griding_num = griding_num
        self.num_lanes = num_lanes
        self.mode = mode
        self.type = type

        pd_reader = pd.read_csv(meta_file)
        random.seed(2022)
        rd_ind = list(range(len(pd_reader)))
        random.shuffle(rd_ind)
        self.pd_reader = pd_reader.reindex(index=rd_ind)
        print('we have totally {} images'.format(len(self.pd_reader['name'])))

        if self.mode == 'train':
            len_image = int(len(self.pd_reader)*0.8)
            self.pd_reader_train = self.pd_reader.iloc[:len_image]

            if self.type != 'all': self.img_list = list(self.pd_reader_train['name'][self.pd_reader_train[self.type].astype(bool)])
            else: self.img_list = list(self.pd_reader_train['name'])
            print(self.type + ' has {} training'.format(len(self.img_list)))
        
        elif self.mode == 'val':
            len_image = int(len(self.pd_reader)*0.2)
            self.pd_reader_val = self.pd_reader.iloc[len(self.pd_reader)-len_image:]

            if self.type != 'all': self.img_list = list(self.pd_reader_val['name'][self.pd_reader_val[self.type].astype(bool)])
            else: self.img_list = list(self.pd_reader_val['name'])
            print(str(self.type) + ' has {} validating'.format(len(self.img_list)))

        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def get_gridding(self, inter_label, num_cols, w):
        num_lane, num_row = inter_label.shape
        col_sample = np.linspace(0, w - 1, num_cols)

        to_pts = np.zeros((num_row, num_lane))
        for i in range(num_lane):
            pti = inter_label[i, :]
            to_pts[:,i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def get_intersaction(self, label):

        all_idx = np.zeros((self.num_lanes, len(self.row_anchor)))

        for i, r in enumerate(self.row_anchor):
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, self.num_lanes + 1):
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i] = pos

        return all_idx

    def __getitem__(self, index):
        # parse label and image name
        img_name = self.img_list[index]
        jpeg_name = 'pic/' + img_name[:-12] + '/' + img_name
        label_name = 'mask/' + img_name[:-12] + '/' + img_name.replace('jpeg', 'png')

        # read label and image
        label_path = os.path.join(self.data_path, label_name)
        label = loader_func(label_path)
        # print(label.size)

        img_path = os.path.join(self.data_path, jpeg_name)
        img = loader_func(img_path)

        # get the positions of intersactions between polyline and rowline (num_lanes, num_rows)
        inter_label = self.get_intersaction(label)
        # print(inter_label.shape)

        # get the coordinates of lanes at row anchors (num_rows, num_lanes)
        grid_label = self.get_gridding(inter_label, self.griding_num, label.size[0])
        # print(grid_label.shape)

        if self.simu_transform:
            img, label = self.simu_transform(img, label)
        img = self.img_transform(img)
        seg_label = self.target_transform(label)
        seg_label[seg_label>self.num_lanes] = 0
        assert (seg_label >= 0).all() & (seg_label < self.num_lanes+1).all()

        return img, grid_label, inter_label, seg_label, jpeg_name

    def __len__(self):
        return len(self.img_list)

if __name__ == "__main__":
    data_path = '/home/ssd7T/lxpData/rail/dataset/'
    meta_file = '/home/ssd7T/lxpData/rail/dataset/meta.csv'

    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        # transforms.Normalize(0.5, 0.5),
    ])
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((288, 800)),
        mytransforms.MaskToTensor(),
    ])
    simu_transform = mytransforms.Compose2([
        # mytransforms.RandomRotate(6),
        # mytransforms.RandomUDoffsetLABEL(100),
        # mytransforms.RandomLROffsetLABEL(200)
    ])

    all_dataset = LaneClsDataset(data_path, meta_file, img_transform = img_transform, target_transform = target_transform, 
                    simu_transform = simu_transform, griding_num=56, row_anchor = raildb_row_anchor, num_lanes=4, mode='val', type='far')

    all_loader = torch.utils.data.DataLoader(all_dataset, batch_size=1, shuffle=False, num_workers=1)

    col_sample = np.linspace(0, 800 - 1, 56)
    col_sample_w = col_sample[1] - col_sample[0]
    color_list = [(0,0,225), (255,0,0), (0,225,0), (255,0,225), (255,255,225), (0,255,255)]

    for ind, (img, grid_label, inter_label, label, jpeg_name) in enumerate(all_loader):
        image = (img[0].permute(1, 2, 0).numpy() * 255).astype(int)
        label = (label[0].numpy()).astype(int)
        label = np.repeat(np.expand_dims(label, axis=2), 3, axis=2)

        canvas = image.copy().astype(np.uint8)
        grid_label[grid_label == 56] = 0
        for i in range(grid_label[0].shape[0]):
            for j in range(grid_label[0].shape[1]):
                if grid_label[0][i,j] > 0:
                    point = (int(grid_label[0][i,j]*col_sample_w)-1, int(raildb_row_anchor[j]/2.5)-1)
                    
                    cv2.circle(canvas, point, 5, color_list[i], -1)

        for lb_value in range(4):
            label[label[:,:,0]==(lb_value+1)] = color_list[lb_value]

        arr_output = np.concatenate([image, label, canvas], axis=0)
        vis_path = data_path+jpeg_name[0].replace('pic', 'check')
        print(vis_path)
        if not os.path.exists(os.path.dirname(vis_path)): os.makedirs(os.path.dirname(vis_path))
        cv2.imwrite(vis_path, arr_output)
        break

