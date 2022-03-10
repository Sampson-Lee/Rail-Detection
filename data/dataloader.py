import torch, os
import numpy as np

import torchvision.transforms as transforms
import data.mytransforms as mytransforms
from data.constant import raildb_row_anchor
from data.dataset import LaneClsDataset, LaneTestDataset

def get_train_loader(batch_size, data_root, griding_num=56, distributed=True, num_lanes=4, mode='train', type='all'):
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((288, 800)),
        mytransforms.MaskToTensor(),
    ])

    img_transform = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    if mode=='train':
        simu_transform = mytransforms.Compose2([
            mytransforms.RandomRotate(6),
            mytransforms.RandomUDoffsetLABEL(100),
            mytransforms.RandomLROffsetLABEL(200)
        ])
    else:
        simu_transform = None

    train_dataset = LaneClsDataset(
                        data_root,
                        data_root+'meta.csv',
                        img_transform = img_transform, 
                        target_transform = target_transform, 
                        simu_transform = simu_transform, 
                        griding_num = griding_num, 
                        row_anchor = raildb_row_anchor, 
                        num_lanes = num_lanes,
                        mode = mode,
                        type = type,
                        )

    if distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=sampler_train, num_workers=4)

    return train_loader, len(raildb_row_anchor)

class SeqDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    '''
    Change the behavior of DistributedSampler to sequential distributed sampling.
    The sequential sampling helps the stability of multi-thread testing, which needs multi-thread file io.
    Without sequentially sampling, the file io on thread may interfere other threads.
    '''
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas, rank, shuffle)
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        num_per_rank = int(self.total_size // self.num_replicas)

        # sequential sampling
        indices = indices[num_per_rank * self.rank : num_per_rank * (self.rank + 1)]

        assert len(indices) == self.num_samples

        return iter(indices)
