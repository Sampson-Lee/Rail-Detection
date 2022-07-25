import torch, os, cv2, sys
import scipy.special, tqdm
import numpy as np
import torchvision.transforms as transforms

sys.path.append("..")

from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from utils.evaluation import grid_2_inter
from data.dataset import LaneTestDataset
from data.constant import raildb_row_anchor
from IPython import embed

color_list = [(0,0,225), (255,0,0), (0,225,0), (255,0,225), (255,255,225), (0,255,255), (255,255,0), (125,255,255)]
thickness_list = [1, 3, 5, 7, 9, 11, 13, 15]
thickness_list.reverse()

def deploy_images(loader, cfg):
    for i, data in enumerate(tqdm.tqdm(loader)):
        imgs, names = data
        imgs = imgs.cuda()
        with torch.no_grad():
            outs = net(imgs)

        vis = cv2.resize(cv2.imread(os.path.join(cfg.data_root, names[0])), (1280, 720))
        preds_inter = [grid_2_inter(out, cfg.griding_num) for out in outs]
            
        for i in range(preds_inter[0].shape[0]):
            points = [[int(x),int(y)] for (x,y) in zip(preds_inter[0][i], raildb_row_anchor) if x>=0]
            cv2.polylines(vis, (np.asarray([points])).astype(np.int32), False, color_list[i], thickness=thickness_list[i])

        vis_path = os.path.join(cfg.data_root, 'deploy', names[0]).replace('pic', cfg.type)
        if not os.path.exists(os.path.dirname(vis_path)): os.makedirs(os.path.dirname(vis_path))
        cv2.imwrite(vis_path, vis)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    args, cfg = merge_config()

    dist_print('start testing...')

    # load model
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']
    net = parsingNet(pretrained = False, backbone=cfg.backbone, cls_dim = (cfg.griding_num+1, len(raildb_row_anchor), cfg.num_lanes),).cuda()

    state_dict = torch.load(cfg.test_model, map_location='cpu')
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()

    # load data
    img_transforms = transforms.Compose([
        transforms.Resize((288, 800)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    cfg.type = 'all'
    dataset = LaneTestDataset(cfg.data_root, cfg.data_root+'meta.csv', img_transform = img_transforms, type=cfg.type)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
    deploy_images(loader, cfg)

    cfg.type = 'sun'
    dataset = LaneTestDataset(cfg.data_root, cfg.data_root+'meta.csv', img_transform = img_transforms, type=cfg.type)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
    deploy_images(loader, cfg)

    cfg.type = 'rain'
    dataset = LaneTestDataset(cfg.data_root, cfg.data_root+'meta.csv', img_transform = img_transforms, type=cfg.type)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
    deploy_images(loader, cfg)

    cfg.type = 'night'
    dataset = LaneTestDataset(cfg.data_root, cfg.data_root+'meta.csv', img_transform = img_transforms, type=cfg.type)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
    deploy_images(loader, cfg)

    cfg.type = 'line'
    dataset = LaneTestDataset(cfg.data_root, cfg.data_root+'meta.csv', img_transform = img_transforms, type=cfg.type)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
    deploy_images(loader, cfg)

    cfg.type = 'cross'
    dataset = LaneTestDataset(cfg.data_root, cfg.data_root+'meta.csv', img_transform = img_transforms, type=cfg.type)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
    deploy_images(loader, cfg)

    cfg.type = 'curve'
    dataset = LaneTestDataset(cfg.data_root, cfg.data_root+'meta.csv', img_transform = img_transforms, type=cfg.type)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
    deploy_images(loader, cfg)

    cfg.type = 'slope'
    dataset = LaneTestDataset(cfg.data_root, cfg.data_root+'meta.csv', img_transform = img_transforms, type=cfg.type)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
    deploy_images(loader, cfg)

    cfg.type = 'near'
    dataset = LaneTestDataset(cfg.data_root, cfg.data_root+'meta.csv', img_transform = img_transforms, type=cfg.type)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
    deploy_images(loader, cfg)

    cfg.type = 'far'
    dataset = LaneTestDataset(cfg.data_root, cfg.data_root+'meta.csv', img_transform = img_transforms, type=cfg.type)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle = False, num_workers=1)
    deploy_images(loader, cfg)

# python deploy.py ../configs/raildb.py --test_model /home/ssd7T/lxpData/rail/log/20220306_103237_lr_4e-04_b_64test/best_0.926.pth
