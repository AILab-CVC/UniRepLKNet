import sys
import math
import os
import time
import random
import numpy as np
import logging
import argparse
import sklearn.metrics as metrics

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.multiprocessing as mp
import torch.cuda.amp as amp
import torch.distributed as dist
from torchvision import transforms
from tensorboardX import SummaryWriter

from pointnet2_ops import pointnet2_utils
from timm.scheduler import CosineLRScheduler
from util.config import load_cfg_from_cfg_file, merge_cfg_from_list
from util.util import AverageMeter, save_checkpoint, count_prompt_parameters, barrier
from util.rotate import rotate_point_clouds, rotate_point_clouds_batch, rotate_theta_phi
from dataset import data_transforms
from models.layers.utils import fps
from models.p2p_adaptor import P2P

# from models.p2p_raw import P2P

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudScaleAndTranslate(),
    ]
)

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

best_acc_cls = 0.0


def worker_init_fn(worker_id):
    random.seed(time.time() + worker_id)


def get_parser():
    parser = argparse.ArgumentParser(description='BPNet')
    parser.add_argument('--config', type=str, default='config/Pretrain_ModelNet40/pretrain-4096-ConvNeXt-L-1k.yaml', help='config file')
    # parser.add_argument('--config', type=str, default='config/ShapeNet/p2p_ResNet-18.yaml', help='config file')
    
    
    parser.add_argument('opts', help='see config/scannet/promptvit_scannet3d_5cm.yaml for all options', default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = merge_cfg_from_list(cfg, args.opts)
    return cfg


def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def main_process():
    return not args.multiprocessing_distributed or (
            args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)
    cudnn.benchmark = True

    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(os.path.join(args.save_path, 'model'), exist_ok=True)

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
    # Log for check version
    print(
        'torch.__version__:%s\ntorch.version.cuda:%s\ntorch.backends.cudnn.version:%s\ntorch.backends.cudnn.enabled:%s' % (
            torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), torch.backends.cudnn.enabled))

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    args.ngpus_per_node = len(args.train_gpu)
    if len(args.train_gpu) == 1:
        args.sync_bn = False
        args.distributed = False
        args.multiprocessing_distributed = False
        args.use_apex = False

    if args.multiprocessing_distributed:
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, args))
    else:
        main_worker(args.train_gpu, args.ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, argss):
    global args
    global best_acc_cls
    args = argss
    args.lr = float(args.lr)

    # if args.distributed:
    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])
    if args.multiprocessing_distributed:
        args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size,
                            rank=args.rank)

    model = P2P(args)
    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)

    model._fix_weight()  # 冻结部分权重
    param_grad, param_all, param_prompt, param_basemodel_grad, param_basemodel_nograd = count_prompt_parameters(model)
    if main_process():
        logger.info("Trainable Parameters: {}".format(param_grad))
        logger.info("All Parameters: {}".format(param_all))
        logger.info("Prompting Parameters: {}".format(param_prompt))
        logger.info("Base Model Trainable Parameters: {}".format(param_basemodel_grad))
        logger.info("Base Model Frozen Parameters: {}".format(param_basemodel_nograd))
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name)

    # ####################### Optimizer ####################### #
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineLRScheduler(optimizer, t_initial=args.epochs, lr_min=1e-6, warmup_lr_init=1e-6,
                                  warmup_t=args.warmup_epochs, cycle_limit=1, t_in_epochs=True)

    if args.distributed:
        torch.cuda.set_device(gpu)
        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.batch_size_val = int(args.batch_size_val / ngpus_per_node)
        args.workers = int(args.workers / ngpus_per_node)
        model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[gpu], find_unused_parameters=True)  # 
    else:
        model = model.cuda()

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            if main_process():
                logger.info("=> no weight found at '{}'".format(args.weight))
    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            current_iter = checkpoint['current_iter']
            best_acc_cls = checkpoint['best_acc_cls']
            if main_process():
                logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']}, best acc {checkpoint['best_acc_cls']})")
        else:
            if main_process():
                logger.info("=> no checkpoint found at '{}'".format(args.resume))

    # ####################### Data Loader ####################### #

    # from dataset.shapenet55 import ShapeNet
    # train_data = ShapeNet(config=args, split='train', whole=True)
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,
    #                                                                 shuffle=True) if args.distributed else None
    # train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
    #                                             shuffle=(train_sampler is None),
    #                                             num_workers=args.workers, pin_memory=True, sampler=train_sampler,
    #                                             drop_last=True, worker_init_fn=worker_init_fn)
    # if args.evaluate:
    #     val_data = ShapeNet(config=args, split='test')
    #     val_sampler = None
    #     val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
    #                                                 shuffle=False, num_workers=args.workers, pin_memory=True,
    #                                                 drop_last=False, sampler=val_sampler)
    from dataset.modelnet import ModelNet
    train_data = ModelNet(config=args, split='train')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,
                                                                    shuffle=True) if args.distributed else None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size,
                                                shuffle=(train_sampler is None),
                                                num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                                drop_last=True, worker_init_fn=worker_init_fn)
    if args.evaluate:
        val_data = ModelNet(config=args, split='test')
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
                                                    shuffle=False, num_workers=args.workers, pin_memory=True,
                                                    drop_last=False, sampler=val_sampler)

    barrier()
    # ####################### Train ####################### #
    for epoch in range(args.start_epoch, args.epochs):
        if args.amp:
            scaler = amp.GradScaler()
        else: 
            scaler = None
        
        if args.distributed:
            train_sampler.set_epoch(epoch)
        if args.data_name == 'modelnet' or args.data_name == 'scanobjectnn' or args.data_name == 'scanobjectnn_hardest' or args.data_name == 'shapenet':
            loss_train_cls, acc_train_cls, current_iter \
                = train_cls(train_loader, model, optimizer, scheduler, epoch, scaler)
        else:
            raise Exception('Dataset not supported yet'.format(args.data_name))
        epoch_log = epoch + 1
        if main_process():
            if args.data_name == 'modelnet' or args.data_name == 'scanobjectnn' or args.data_name == 'scanobjectnn_hardest' or args.data_name == 'shapenet':
                writer.add_scalar('loss_train_cls', loss_train_cls, epoch_log)
                writer.add_scalar('acc_train_cls', acc_train_cls, epoch_log)

        is_best = False
        # if (args.evaluate and (epoch_log % args.eval_freq == 0) and (epoch_log < args.epochs - args.last_epochs) and (epoch_log > 200)) or (
        #         args.evaluate and (epoch_log >= args.epochs - args.last_epochs)):
        if (args.evaluate and (epoch_log % args.eval_freq == 0) and (epoch_log < args.epochs - args.last_epochs)) or (
            args.evaluate and (epoch_log >= args.epochs - args.last_epochs)):
            if args.data_name == 'modelnet' or args.data_name == 'scanobjectnn' or args.data_name == 'scanobjectnn_hardest' or args.data_name == 'shapenet':
                loss_val_cls, acc_val_cls \
                    = validate_cls(val_loader, model)
            else:
                raise Exception('Dataset not supported yet'.format(args.data_name))

            if main_process():
                if args.data_name == 'modelnet' or args.data_name == 'scanobjectnn' or args.data_name == 'scanobjectnn_hardest' or args.data_name == 'shapenet':
                    writer.add_scalar('loss_val_cls', loss_val_cls, epoch_log)
                    writer.add_scalar('acc_val_cls', acc_val_cls, epoch_log)

                    # remember best iou and save checkpoint
                    is_best = acc_val_cls > best_acc_cls
                    best_acc_cls = max(best_acc_cls, acc_val_cls)
                    logger.info('Best Accuracy: %.4f' % (best_acc_cls))

        if (epoch_log % args.save_freq == 0) and main_process():
            save_checkpoint(
                {
                    'epoch': epoch_log,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_acc_cls': best_acc_cls,
                    'current_iter': current_iter
                }, is_best, os.path.join(args.save_path, 'model')
            )
    if main_process():
        writer.close()
        if args.data_name == 'modelnet' or args.data_name == 'scanobjectnn' or args.data_name == 'scanobjectnn_hardest' or args.data_name == 'shapenet':
            logger.info('==>Training done!\nBest Accuracy: %.3f' % (best_acc_cls))
        else:
            raise Exception('Dataset not supported yet'.format(args.data_name))


def train_cls(train_loader, model, optimizer, scheduler, epoch, scaler=None):
    torch.backends.cudnn.enabled = True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    
    labels = []
    preds = []

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    for i, batch_data in enumerate(train_loader):
        optimizer.zero_grad()
        data_time.update(time.time() - end)
        points = batch_data[0].cuda()
        label = batch_data[1].cuda()

        if args.npoints == 1024:
            point_all = 1200
        elif args.npoints == 2048:
            point_all = 2400
        elif args.npoints == 4096:
            point_all = 4800
        elif args.npoints == 8192:
            point_all = 8192
        else:
            raise NotImplementedError()
        if points.size(1) < point_all:
            point_all = points.size(1)
        
        fps_idx = pointnet2_utils.furthest_point_sample(points[:, :, :3].contiguous(), point_all)  # (B, npoint)
        fps_idx = fps_idx[:, np.random.choice(point_all, args.npoints, False)]
        points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()  # (B, N, 3)
        
        # random rotate the point cloud with a random init angle
        angle = torch.stack([torch.rand(points.size(0)) * 1.9 + 0.04,                       # 0.04 ~ 1.94pi
                             (torch.rand(points.size(0)) * 0.2 - 0.4)], dim=-1) * math.pi   # -0.4 ~ -0.2 pi
        rotation_matrix = rotate_theta_phi(angle)
        input_pc = rotate_point_clouds_batch(points, rotation_matrix, use_normals=args.use_normals).contiguous()   
        input_pc = train_transforms(input_pc)            

        if scaler is not None:
            # forward the model
            with amp.autocast(dtype=torch.float16):
                output = model(input_pc, original_pc=points) # B 40
                if args.distributed:
                    loss, accuracy = model.module.get_loss_acc(output, label, smoothing=args.label_smoothing)
                else:
                    loss, accuracy = model.get_loss_acc(output, label, smoothing=args.label_smoothing)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
        else:
            output = model(input_pc, original_pc=points) # B 40
            if args.distributed:
                loss, accuracy = model.module.get_loss_acc(output, label, smoothing=args.label_smoothing)
            else:
                loss, accuracy = model.get_loss_acc(output, label, smoothing=args.label_smoothing)
            loss.backward()
            optimizer.step()
            
        if args.scheduler != 'CosLR':
            scheduler.step()

        pred = torch.argmax(output, 1).detach().cpu().numpy()
        preds.append(pred)
        labels.append(label.cpu().numpy())
        loss_meter.update(loss.item(), args.batch_size)
        batch_time.update(time.time() - end)
        end = time.time()

        # Adjust lr
        current_iter = epoch * len(train_loader) + i + 1
        current_lr = optimizer.param_groups[0]["lr"]

        # calculate remain time
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Lr {learning_rate:.6f} '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'Accuracy {accuracy:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                          batch_time=batch_time, learning_rate=current_lr,
                                                          remain_time=remain_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))
        if main_process():
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('acc_train_batch', accuracy, current_iter)
            writer.add_scalar('learning_rate', current_lr, current_iter)
            
        # torch.cuda.empty_cache()
        
    if args.scheduler == 'CosLR':
        scheduler.step(epoch)

    avg_acc = metrics.accuracy_score(np.concatenate(labels), np.concatenate(preds)) * 100
    if main_process():
        logger.info(
            'Train result at epoch [{}/{}]: avgAcc {:.4f}.'.format(epoch + 1, args.epochs, avg_acc))
    return loss_meter.avg, avg_acc, current_iter


def validate_cls(val_loader, model):
    torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107
    loss_meter = AverageMeter()
    labels = []
    preds = []
    theta = np.linspace(0.1, 2, 9)
    phi = -0.35
    v_theta, v_phi = np.meshgrid(theta[:8], phi)
    angles = np.stack([v_theta, v_phi], axis=-1).reshape(-1, 2)
    angles = torch.from_numpy(angles) * math.pi
    rotation_matrixs = rotate_theta_phi(angles)

    model.eval()
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            points = batch_data[0].cuda()
            label = batch_data[1].cuda()

            points = fps(points, args.npoints)

            for rotation_matrix in rotation_matrixs:
                input_pc = rotate_point_clouds(points, rotation_matrix, use_normals=args.use_normals)
                output = model(input_pc, original_pc=points)

                if args.distributed:
                    loss, _ = model.module.get_loss_acc(output, label, smoothing=args.label_smoothing)
                else:
                    loss, _ = model.get_loss_acc(output, label, smoothing=args.label_smoothing)

                pred = torch.argmax(output, 1)
                preds.append(pred.detach().cpu().numpy())
                labels.append(label.cpu().numpy())
                loss_meter.update(loss.item(), args.batch_size * 8)
            
            # torch.cuda.empty_cache()

    avg_acc = metrics.accuracy_score(np.concatenate(labels), np.concatenate(preds)) * 100
    if main_process():
        logger.info(
            'Val result: avgAcc {:.4f}.'.format(avg_acc))
    return loss_meter.avg, avg_acc


if __name__ == '__main__':
    main()
