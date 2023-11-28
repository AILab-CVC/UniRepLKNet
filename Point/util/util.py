import os
import torch
import torch.distributed as dist
import shutil
from os.path import join


def count_prompt_parameters(model):
    param_grad = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_all = sum(p.numel() for p in model.parameters())
    param_prompt = sum(p.numel() for n, p in model.named_parameters() if 'base_model' not in n)
    param_basemodel_grad = sum(p.numel() for n, p in model.named_parameters() if 'base_model' in n and p.requires_grad)
    param_basemodel_nograd = sum(p.numel() for n, p in model.named_parameters() if 'base_model' in n and not p.requires_grad)
    return param_grad, param_all, param_prompt, param_basemodel_grad, param_basemodel_nograd


def save_checkpoint(state, is_best, sav_path, filename='model_last.pth'):
    epoch = state['epoch']
    filename = join(sav_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(sav_path, 'model_best.pth'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def is_distributed():
    if not dist.is_available() or not dist.is_initialized():
        return False
    return True

def barrier():
    if not is_distributed():
        return
    torch.distributed.barrier()