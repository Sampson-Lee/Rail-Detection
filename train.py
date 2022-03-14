from wsgiref import validate
import torch, os, datetime, copy, json, scipy, cv2
import numpy as np

from model.model import parsingNet
from data.dataloader import get_train_loader
from data.constant import raildb_row_anchor
from utils.evaluation import LaneEval, grid_2_inter

from utils.dist_utils import dist_print, dist_tqdm, is_main_process
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics

from utils.common import merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger

import time
from IPython import embed

color_list = [(0,0,225), (255,0,0), (0,225,0), (255,0,225), (255,255,225), (0,255,255), (255,255,0), (125,255,255)]
thickness_list = [1, 3, 5, 7, 9, 11, 13, 15]
thickness_list.reverse()

def inference(net, data_label):

    img, cls_label, _, _, _ = data_label
    img, cls_label = img.cuda(), cls_label.long().cuda()
    cls_out = net(img)
    return {'cls_out': cls_out, 'cls_label': cls_label}

def resolve_val_data(results):
    # input: (batch_size, num_gridding, num_cls_per_lane, num_of_lanes)
    # output: (batch_size, num_cls_per_lane, num_of_lanes)
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    return results

def calc_loss(loss_dict, results, logger, global_step):
    loss = 0

    for i in range(len(loss_dict['name'])):
        
        data_src = loss_dict['data_src'][i]

        datas = [results[src] for src in data_src]
        loss_cur = loss_dict['op'][i](*datas)

        if global_step % 20 == 0:
            # print(loss_cur)

            logger.add_scalar('loss/'+loss_dict['name'][i], loss_cur, global_step)

        loss += loss_cur * loss_dict['weight'][i]
    return loss

def train(net, train_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict):
    dist_print('*****************   Training   ***********************')
    net.train(mode=True)
    progress_bar = dist_tqdm(train_loader)
    t_data_0 = time.time()
    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()
        reset_metrics(metric_dict)
        global_step = epoch * len(train_loader) + b_idx

        t_net_0 = time.time()
        results = inference(net, data_label)

        loss = calc_loss(loss_dict, results, logger, global_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        t_net_1 = time.time()

        results = resolve_val_data(results)

        update_metrics(metric_dict, results)
        if global_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
        logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

        if hasattr(progress_bar, 'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                    data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                    net_time = '%.3f' % float(t_net_1 - t_net_0), 
                                    **kwargs)
        t_data_0 = time.time()

def validate(net, val_loader, logger, metric_dict, savefig=[]):
    dist_print('*****************   Validating   ***********************')
    net.train(mode=False)
    progress_bar = dist_tqdm(val_loader)
    t_data_0 = time.time()
    reset_metrics(metric_dict)

    preds = []; gts = []
    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()
        global_step = b_idx

        results = inference(net, data_label)
        preds_inter = [grid_2_inter(out, cfg.griding_num) for out in results['cls_out']]
        # print(pred)
        gt = data_label[2].cpu().numpy()
        # print(gt)
        
        if len(savefig)!=0:
            for idx, item in enumerate(data_label[-1]):
                vis = cv2.resize(cv2.imread(os.path.join(savefig[0], item)), (1280, 720))
                vis_mask = np.zeros_like(vis).astype(np.uint8)
                
                for i in range(preds_inter[idx].shape[0]):
                    points = [[int(x),int(y)] for (x,y) in zip(preds_inter[idx][i], raildb_row_anchor) if x>=0]
                    cv2.polylines(vis, (np.asarray([points])).astype(np.int32), False, color_list[i], thickness=thickness_list[i])
                    cv2.polylines(vis_mask, (np.asarray([points])).astype(np.int32), False, color_list[i], thickness=thickness_list[i])

                vis_path = os.path.join(savefig[0], 'row_based/vis', item).replace('pic', savefig[1])
                if not os.path.exists(os.path.dirname(vis_path)): os.makedirs(os.path.dirname(vis_path))
                cv2.imwrite(vis_path, vis)

                pred_path = os.path.join(savefig[0], 'row_based/pred', item).replace('pic', savefig[1])
                if not os.path.exists(os.path.dirname(pred_path)): os.makedirs(os.path.dirname(pred_path))
                cv2.imwrite(pred_path, vis)

        results = resolve_val_data(results)
        update_metrics(metric_dict, results)
        t_data_0 = time.time()

        for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
            logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
        acc_top1 = metric_dict['op'][0].get()

        if hasattr(progress_bar, 'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            progress_bar.set_postfix(**kwargs,
                                    data_time = '%.3f' % float(t_data_1 - t_data_0), 
                                    )
        
        preds.append(preds_inter)
        gts.append(gt)

    preds = np.concatenate(preds); gts = np.concatenate(gts)
    res = LaneEval.bench_all(preds, gts, raildb_row_anchor)
    res = json.loads(res)
    for r in res:
        dist_print(r['name'], r['value']) 
    
    return acc_top1


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

    args, cfg = merge_config()

    work_dir = get_work_dir(cfg)
    
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18','34','50','101','152','50next','101next','50wide','101wide']

    train_loader, cls_num_per_lane = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, distributed, cfg.num_lanes, mode='train', type=cfg.type)
    val_loader, _ = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, distributed, cfg.num_lanes, mode='val', type='all')
    val_sun_loader, _ = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, distributed, cfg.num_lanes, mode='val', type='sun')
    val_rain_loader, _ = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, distributed, cfg.num_lanes, mode='val', type='rain')
    val_night_loader, _ = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, distributed, cfg.num_lanes, mode='val', type='night')
    val_line_loader, _ = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, distributed, cfg.num_lanes, mode='val', type='line')
    val_cross_loader, _ = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, distributed, cfg.num_lanes, mode='val', type='cross')
    val_curve_loader, _ = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, distributed, cfg.num_lanes, mode='val', type='curve')
    val_slope_loader, _ = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, distributed, cfg.num_lanes, mode='val', type='slope')
    val_near_loader, _ = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, distributed, cfg.num_lanes, mode='val', type='near')
    val_far_loader, _ = get_train_loader(cfg.batch_size, cfg.data_root, cfg.griding_num, distributed, cfg.num_lanes, mode='val', type='far')

    net = parsingNet(pretrained = True, backbone=cfg.backbone, cls_dim = (cfg.griding_num+1, cls_num_per_lane, cfg.num_lanes)).cuda()

    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])
    optimizer = get_optimizer(net, cfg)

    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))
    metric_dict = get_metric_dict(cfg)
    loss_dict = get_loss_dict(cfg)
    logger = get_logger(work_dir, cfg)
    cp_projects(args.auto_backup, work_dir)
    
    best_acc = 0; best_epoch = 0; best_model = None
    for epoch in range(resume_epoch, cfg.epoch):
        train(net, train_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict)
        acc = validate(net, val_loader, logger, metric_dict)
        if acc > best_acc: best_acc, best_epoch, best_model = acc, epoch, copy.deepcopy(net)
        save_model(net, optimizer, epoch, work_dir, distributed)
    dist_print('*************    validate all      ***************')
    validate(best_model, val_loader, logger, metric_dict, savefig=[cfg.data_root, 'all'])
    dist_print('*************    validate sun      ***************')
    validate(best_model, val_sun_loader, logger, metric_dict, savefig=[cfg.data_root, 'sun'])
    dist_print('*************    validate rain      ***************')
    validate(best_model, val_rain_loader, logger, metric_dict, savefig=[cfg.data_root, 'rain'])
    dist_print('*************    validate night      ***************')
    validate(best_model, val_night_loader, logger, metric_dict, savefig=[cfg.data_root, 'night'])
    dist_print('*************    validate line      ***************')
    validate(best_model, val_line_loader, logger, metric_dict, savefig=[cfg.data_root, 'line'])
    dist_print('*************    validate cross      ***************')
    validate(best_model, val_cross_loader, logger, metric_dict, savefig=[cfg.data_root, 'cross'])
    dist_print('*************    validate curve      ***************')
    validate(best_model, val_curve_loader, logger, metric_dict, savefig=[cfg.data_root, 'curve'])
    dist_print('*************    validate slope      ***************')
    validate(best_model, val_slope_loader, logger, metric_dict, savefig=[cfg.data_root, 'slope'])
    dist_print('*************    validate near      ***************')
    validate(best_model, val_near_loader, logger, metric_dict, savefig=[cfg.data_root, 'near'])
    dist_print('*************    validate far      ***************')
    validate(best_model, val_far_loader, logger, metric_dict, savefig=[cfg.data_root, 'far'])
    logger.close()
    dist_print(best_acc, best_epoch)
    if is_main_process(): torch.save(best_model.state_dict(), os.path.join(work_dir, 'best_{:.3f}.pth'.format(best_acc)))
