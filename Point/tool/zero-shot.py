import sys
import math
import os
import time
import random
import numpy as np
import logging
import argparse
import sklearn.metrics as metrics
from typing import OrderedDict

import cv2
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
import torch.utils.data

from tensorboardX import SummaryWriter
from tqdm import tqdm

from util.config import load_cfg_from_cfg_file, merge_cfg_from_list
from util.util import AverageMeter, count_prompt_parameters
from util.rotate import rotate_point_clouds, rotate_point_clouds_batch, rotate_theta_phi
from dataset import data_transforms
from models.layers.utils import fps
from models.p2p_adaptor import P2P

# from models.p2p_raw import P2P

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)

ScanObjectNN2ModelNet = {
    1: 2,
    5: 8,
    6: 12,
    8: 13,
    12: 30,
    14: 35
}

ShapeNet2ModelNet = {
    0: 0,
    3: 7,
    4: 8,
    6: 17,
    15: 33
}


def worker_init_fn(worker_id):
    random.seed(time.time() + worker_id)


def get_parser():
    parser = argparse.ArgumentParser(description='BPNet')

    parser.add_argument('--config', type=str, default='/cpfs01/user/penghaoyang/code/ICCV2023/config/Pretrain_ModelNet40/pretrain-2048-ConvNeXt-L-1k.yaml', help='config file')
    
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


def main():
    args = get_parser()
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
    main_worker(args)


def main_worker(args):

    args = args

    model = P2P(args)

    logger = get_logger()
    logger.info(args)
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))
    logger.info(model)

    model._fix_weight()  # 冻结部分权重
    param_grad, param_all, param_prompt, param_basemodel_grad, param_basemodel_nograd = count_prompt_parameters(model)

    logger.info("Trainable Parameters: {}".format(param_grad))
    logger.info("All Parameters: {}".format(param_all))
    logger.info("Prompting Parameters: {}".format(param_prompt))
    logger.info("Base Model Trainable Parameters: {}".format(param_basemodel_grad))
    logger.info("Base Model Frozen Parameters: {}".format(param_basemodel_nograd))
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    model = model.cuda()
    checkpoint = torch.load(args.model_path, map_location=lambda storage, loc: storage.cuda())
    state_dict = OrderedDict({key.replace("module.", ""): value for key, value in checkpoint['state_dict'].items()})
    model.load_state_dict(state_dict, strict=True)
    print("Successfully load the model weight")

    
    # ####################### Data Loader ####################### #
    # from dataset.modelnet import ModelNetFewShot

    # val_data = ModelNetFewShot(config=args, split='test')
    # val_sampler = None
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
    #                                             shuffle=False, num_workers=args.workers, pin_memory=True,
    #                                             drop_last=False, sampler=val_sampler)
    
    from dataset.scanobjectnn import ScanObjectNN
    args.data_root = "/cpfs01/user/penghaoyang/code/ICCV2023/data/ScanObjectNN/main_split"
    val_data = ScanObjectNN(config=args, subset='test')
    val_sampler = None
    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=args.batch_size_val,
        shuffle=False, num_workers=args.workers, pin_memory=True,
        drop_last=False, sampler=val_sampler
        )
    cat_id = ScanObjectNN2ModelNet
        
    # from dataset.shapenet import ShapeNetClsFewShot

    # val_data = ShapeNetClsFewShot(data_root=args.data_root, num_points=args.npoints, partition='test')
    # val_sampler = None
    # val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val,
    #                                         shuffle=False, num_workers=args.workers, pin_memory=True,
    #                                         drop_last=False, sampler=val_sampler)
    # cat_id = ShapeNet2ModelNet
    
    validate_cls(args, val_loader, model, cat_id)



def test_cls(args, model, val_data_loader):
    torch.backends.cudnn.enabled = False  # for cudnn bug at https://github.com/pytorch/pytorch/issues/4107

    # here we produce the point cloud after rotating into 80 angles
    theta = np.linspace(0, 2, 11)
    phi = np.linspace(-0.4, -0.2, 4)
    v_theta, v_phi = np.meshgrid(theta[:10], phi)
    angles = np.stack([v_theta, v_phi], axis=-1).reshape(-1, 2)
    angles = torch.from_numpy(angles) * torch.pi
    rotation_matrixs = rotate_theta_phi(angles)

    with torch.no_grad():
        model.eval()
        labels = []
        preds = []
        pbar = tqdm(total=len(val_data_loader))
        for i, batch_data in enumerate(val_data_loader):
            pbar.update(1)
            points = batch_data[0].cuda()

            points = fps(points, args.npoints)

            outputs = []
            # vote
            for rotation_matrix in rotation_matrixs:
                input_pc = rotate_point_clouds(points, rotation_matrix, use_normals=args.use_normals)
                output = model(input_pc, original_pc=points)
                outputs.append(output.detach().unsqueeze(0))

            outputs = torch.cat(outputs, dim=0).mean(0)
            preds.append(torch.argmax(outputs, 1).detach().cpu().numpy())
            labels.append(batch_data[1].numpy())
            torch.cuda.empty_cache()

        pbar.close()
        labels = np.concatenate(labels)
        preds = np.concatenate(preds)
        # np.save(join(args.save_folder, 'gt.npy'), labels)
        # np.save(join(args.save_folder, 'pred.npy'), preds)
        oAcc = metrics.accuracy_score(labels, preds) * 100
        mAcc = metrics.balanced_accuracy_score(labels, preds) * 100

        print("Test overall accuracy: ", oAcc)
        print("Test mean accuracy: ", mAcc)


def validate_cls(args, val_loader, model, cat_id):
    id2cat = {cat_id[t]: t for t in cat_id}
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

            # for rotation_matrix in rotation_matrixs:
            input_pc = rotate_point_clouds(points, rotation_matrixs[0], use_normals=args.use_normals)
            output = model(input_pc, original_pc=points)

            pred = torch.argmax(output, 1)
            cat_idx = [la in cat_id for la in label.cpu().numpy()]
            
            label = label.cpu().numpy()[cat_idx]
            pseudo_pred_list = pred.detach().cpu().numpy()[cat_idx]
            
            true_pred = []
            for la in pseudo_pred_list:
                if la in cat_id:
                    true_pred.append(id2cat[la])
                else:
                    true_pred.append(la)
            
            pred = np.array(true_pred, dtype=np.int64)
            
            preds.append(pred)
            labels.append(label)
        
            # torch.cuda.empty_cache()

    avg_acc = metrics.accuracy_score(np.concatenate(labels), np.concatenate(preds)) * 100
    
    print(avg_acc)


if __name__ == '__main__':
    main()
