# --------------------------------------------------------
# UniRepLKNet
# https://arxiv.org/abs/2311.15599
# https://github.com/AILab-CVC/UniRepLKNet
# Licensed under The Apache 2.0 License [see LICENSE for details]
# --------------------------------------------------------
import argparse
import os
import warnings
import mmcv
import torch
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model, load_state_dict)
from mmcv.utils import DictAction
from mmseg.models import build_segmentor
import mmcv_custom   # noqa: F401,F403
import mmseg_custom   # noqa: F401,F403


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('target', help='where to save the reparameterized weights')

    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')

    args = parser.parse_args()

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    print(cfg.model)

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    if hasattr(model, 'module'):
        load_state_dict(model.module, checkpoint['state_dict'], strict=False)
        model = model.module
    else:
        load_state_dict(model, checkpoint['state_dict'], strict=False)
    
    for m in model.modules():
        if hasattr(m, 'reparameterize_unireplknet'):
            m.reparameterize_unireplknet()

    result = {
        'state_dict': model.state_dict()
    }
    torch.save(result, args.target)


if __name__ == '__main__':
    main()
