# UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition
# Github source: https://github.com/AILab-CVC/UniRepLKNet
# Licensed under The Apache License 2.0 License [see LICENSE for details]
# Based on RepLKNet, ConvNeXt, timm, DINO and DeiT code bases
# https://github.com/DingXiaoH/RepLKNet-pytorch
# https://github.com/facebookresearch/ConvNeXt
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from timm.models.registry import register_model
from functools import partial
import torch.utils.checkpoint as checkpoint
try:
    from huggingface_hub import hf_hub_download
except:
    hf_hub_download = None      # install huggingface_hub if you would like to download models conveniently from huggingface

has_mmdet = False
has_mmseg = False
#   =============== for the ease of directly using this file in MMSegmentation and MMDetection.
#   =============== ignore the following two segments of code if you do not plan to do so
#   =============== delete one of the following two segments if you get a confliction
try:
    from mmseg.models.builder import BACKBONES as seg_BACKBONES
    from mmseg.utils import get_root_logger
    from mmcv.runner import _load_checkpoint
    has_mmseg = True
except ImportError:
    get_root_logger = None
    _load_checkpoint = None

# try:
#     from mmdet.models.builder import BACKBONES as det_BACKBONES
#     from mmdet.utils import get_root_logger
#     from mmcv.runner import _load_checkpoint
#     has_mmdet = True
# except ImportError:
#     get_root_logger = None
#     _load_checkpoint = None
#   ===========================================================================================


class GRNwithNHWC(nn.Module):
    """ GRN (Global Response Normalization) layer
    Originally proposed in ConvNeXt V2 (https://arxiv.org/abs/2301.00808)
    This implementation is more efficient than the original (https://github.com/facebookresearch/ConvNeXt-V2)
    We assume the inputs to this layer are (N, H, W, C)
    """
    def __init__(self, dim, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        if self.use_bias:
            self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        if self.use_bias:
            return (self.gamma * Nx + 1) * x + self.beta
        else:
            return (self.gamma * Nx + 1) * x


class NCHWtoNHWC(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1)


class NHWCtoNCHW(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2)

#================== This function decides which conv implementation (the native or iGEMM) to use
#   Note that iGEMM large-kernel conv impl will be used if
#       -   you attempt to do so (attempt_to_use_large_impl=True), and
#       -   it has been installed (follow https://github.com/AILab-CVC/UniRepLKNet), and
#       -   the conv layer is depth-wise, stride = 1, non-dilated, kernel_size > 5, and padding == kernel_size // 2
def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias,
               attempt_use_lk_impl=True):
    kernel_size = to_2tuple(kernel_size)
    if padding is None:
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)
    else:
        padding = to_2tuple(padding)
    need_large_impl = kernel_size[0] == kernel_size[1] and kernel_size[0] > 5 and padding == (kernel_size[0] // 2, kernel_size[1] // 2)

    if attempt_use_lk_impl and need_large_impl:
        print('---------------- trying to import iGEMM implementation for large-kernel conv')
        try:
            from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
            print('---------------- found iGEMM implementation ')
        except:
            DepthWiseConv2dImplicitGEMM = None
            print('---------------- found no iGEMM. use original conv. follow https://github.com/AILab-CVC/UniRepLKNet to install it.')
        if DepthWiseConv2dImplicitGEMM is not None and need_large_impl and in_channels == out_channels \
                and out_channels == groups and stride == 1 and dilation == 1:
            print(f'===== iGEMM Efficient Conv Impl, channels {in_channels}, kernel size {kernel_size} =====')
            return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=bias)


def get_bn(dim, use_sync_bn=False):
    if use_sync_bn:
        return nn.SyncBatchNorm(dim)
    else:
        return nn.BatchNorm2d(dim)

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = F.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)

def fuse_bn(conv, bn):
    conv_bias = 0 if conv.bias is None else conv.bias
    std = (bn.running_var + bn.eps).sqrt()
    return conv.weight * (bn.weight / std).reshape(-1, 1, 1, 1), bn.bias + (conv_bias - bn.running_mean) * bn.weight / std

def convert_dilated_to_nondilated(kernel, dilate_rate):
    identity_kernel = torch.ones((1, 1, 1, 1))
    if kernel.size(1) == 1:
        #   This is a DW kernel
        dilated = F.conv_transpose2d(kernel, identity_kernel, stride=dilate_rate)
        return dilated
    else:
        #   This is a dense or group-wise (but not DW) kernel
        slices = []
        for i in range(kernel.size(1)):
            dilated = F.conv_transpose2d(kernel[:,i:i+1,:,:], identity_kernel, stride=dilate_rate)
            slices.append(dilated)
        return torch.cat(slices, dim=1)

def merge_dilated_into_large_kernel(large_kernel, dilated_kernel, dilated_r):
    large_k = large_kernel.size(2)
    dilated_k = dilated_kernel.size(2)
    equivalent_kernel_size = dilated_r * (dilated_k - 1) + 1
    equivalent_kernel = convert_dilated_to_nondilated(dilated_kernel, dilated_r)
    rows_to_pad = large_k // 2 - equivalent_kernel_size // 2
    merged_kernel = large_kernel + F.pad(equivalent_kernel, [rows_to_pad] * 4)
    return merged_kernel


class DilatedReparamBlock(nn.Module):
    """
    Dilated Reparam Block proposed in UniRepLKNet (https://github.com/AILab-CVC/UniRepLKNet)
    We assume the inputs to this block are (N, C, H, W)
    """
    def __init__(self, channels, kernel_size, deploy, use_sync_bn=False, attempt_use_lk_impl=True):
        super().__init__()
        self.lk_origin = get_conv2d(channels, channels, kernel_size, stride=1,
                                    padding=kernel_size//2, dilation=1, groups=channels, bias=deploy,
                                    attempt_use_lk_impl=attempt_use_lk_impl)
        self.attempt_use_lk_impl = attempt_use_lk_impl

        #   Default settings. We did not tune them carefully. Different settings may work better.
        if kernel_size == 17:
            self.kernel_sizes = [5, 9, 3, 3, 3]
            self.dilates = [1, 2, 4, 5, 7]
        elif kernel_size == 15:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 5, 7]
        elif kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        elif kernel_size == 5:
            self.kernel_sizes = [3, 3]
            self.dilates = [1, 2]
        else:
            raise ValueError('Dilated Reparam Block requires kernel_size >= 5')

        if not deploy:
            self.origin_bn = get_bn(channels, use_sync_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__setattr__('dil_conv_k{}_{}'.format(k, r),
                                 nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=k, stride=1,
                                           padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels,
                                           bias=False))
                self.__setattr__('dil_bn_k{}_{}'.format(k, r), get_bn(channels, use_sync_bn=use_sync_bn))

    def forward(self, x):
        if not hasattr(self, 'origin_bn'):      # deploy mode
            return self.lk_origin(x)
        out = self.origin_bn(self.lk_origin(x))
        for k, r in zip(self.kernel_sizes, self.dilates):
            conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
            bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
            out = out + bn(conv(x))
        return out

    def merge_dilated_branches(self):
        if hasattr(self, 'origin_bn'):
            origin_k, origin_b = fuse_bn(self.lk_origin, self.origin_bn)
            for k, r in zip(self.kernel_sizes, self.dilates):
                conv = self.__getattr__('dil_conv_k{}_{}'.format(k, r))
                bn = self.__getattr__('dil_bn_k{}_{}'.format(k, r))
                branch_k, branch_b = fuse_bn(conv, bn)
                origin_k = merge_dilated_into_large_kernel(origin_k, branch_k, r)
                origin_b += branch_b
            merged_conv = get_conv2d(origin_k.size(0), origin_k.size(0), origin_k.size(2), stride=1,
                                    padding=origin_k.size(2)//2, dilation=1, groups=origin_k.size(0), bias=True,
                                    attempt_use_lk_impl=self.attempt_use_lk_impl)
            merged_conv.weight.data = origin_k
            merged_conv.bias.data = origin_b
            self.lk_origin = merged_conv
            self.__delattr__('origin_bn')
            for k, r in zip(self.kernel_sizes, self.dilates):
                self.__delattr__('dil_conv_k{}_{}'.format(k, r))
                self.__delattr__('dil_bn_k{}_{}'.format(k, r))


class UniRepLKNetBlock(nn.Module):

    def __init__(self,
                 dim,
                 kernel_size,
                 drop_path=0.,
                 layer_scale_init_value=1e-6,
                 deploy=False,
                 attempt_use_lk_impl=True,
                 with_cp=False,
                 use_sync_bn=False,
                 ffn_factor=4):
        super().__init__()
        self.with_cp = with_cp
        if deploy:
            print('------------------------------- Note: deploy mode')
        if self.with_cp:
            print('****** note with_cp = True, reduce memory consumption but may slow down training ******')

        if kernel_size == 0:
            self.dwconv = nn.Identity()
        elif kernel_size >= 7:
            self.dwconv = DilatedReparamBlock(dim, kernel_size, deploy=deploy,
                                              use_sync_bn=use_sync_bn,
                                              attempt_use_lk_impl=attempt_use_lk_impl)

        else:
            assert kernel_size in [3, 5]
            self.dwconv = get_conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=kernel_size // 2,
                                     dilation=1, groups=dim, bias=deploy,
                                     attempt_use_lk_impl=attempt_use_lk_impl)

        if deploy or kernel_size == 0:
            self.norm = nn.Identity()
        else:
            self.norm = get_bn(dim, use_sync_bn=use_sync_bn)

        self.se = SEBlock(dim, dim // 4)

        ffn_dim = int(ffn_factor * dim)
        self.pwconv1 = nn.Sequential(
            NCHWtoNHWC(),
            nn.Linear(dim, ffn_dim))
        self.act = nn.Sequential(
            nn.GELU(),
            GRNwithNHWC(ffn_dim, use_bias=not deploy))
        if deploy:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim),
                NHWCtoNCHW())
        else:
            self.pwconv2 = nn.Sequential(
                nn.Linear(ffn_dim, dim, bias=False),
                NHWCtoNCHW(),
                get_bn(dim, use_sync_bn=use_sync_bn))

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim),
                                  requires_grad=True) if (not deploy) and layer_scale_init_value is not None \
                                                         and layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def compute_residual(self, x):
        y = self.se(self.norm(self.dwconv(x)))
        y = self.pwconv2(self.act(self.pwconv1(y)))
        if self.gamma is not None:
            y = self.gamma.view(1, -1, 1, 1) * y
        return self.drop_path(y)

    def forward(self, inputs):

        def _f(x):
            return x + self.compute_residual(x)

        if self.with_cp and inputs.requires_grad:
            out = checkpoint.checkpoint(_f, inputs)
        else:
            out = _f(inputs)
        return out

    def reparameterize(self):
        if hasattr(self.dwconv, 'merge_dilated_branches'):
            self.dwconv.merge_dilated_branches()
        if hasattr(self.norm, 'running_var'):
            std = (self.norm.running_var + self.norm.eps).sqrt()
            if hasattr(self.dwconv, 'lk_origin'):
                self.dwconv.lk_origin.weight.data *= (self.norm.weight / std).view(-1, 1, 1, 1)
                self.dwconv.lk_origin.bias.data = self.norm.bias + (
                            self.dwconv.lk_origin.bias - self.norm.running_mean) * self.norm.weight / std
            else:
                conv = nn.Conv2d(self.dwconv.in_channels, self.dwconv.out_channels, self.dwconv.kernel_size,
                                 self.dwconv.padding, self.dwconv.groups, bias=True)
                conv.weight.data = self.dwconv.weight * (self.norm.weight / std).view(-1, 1, 1, 1)
                conv.bias.data = self.norm.bias - self.norm.running_mean * self.norm.weight / std
                self.dwconv = conv
            self.norm = nn.Identity()
        if self.gamma is not None:
            final_scale = self.gamma.data
            self.gamma = None
        else:
            final_scale = 1
        if self.act[1].use_bias and len(self.pwconv2) == 3:
            grn_bias = self.act[1].beta.data
            self.act[1].__delattr__('beta')
            self.act[1].use_bias = False
            linear = self.pwconv2[0]
            grn_bias_projected_bias = (linear.weight.data @ grn_bias.view(-1, 1)).squeeze()
            bn = self.pwconv2[2]
            std = (bn.running_var + bn.eps).sqrt()
            new_linear = nn.Linear(linear.in_features, linear.out_features, bias=True)
            new_linear.weight.data = linear.weight * (bn.weight / std * final_scale).view(-1, 1)
            linear_bias = 0 if linear.bias is None else linear.bias.data
            linear_bias += grn_bias_projected_bias
            new_linear.bias.data = (bn.bias + (linear_bias - bn.running_mean) * bn.weight / std) * final_scale
            self.pwconv2 = nn.Sequential(new_linear, self.pwconv2[1])



default_UniRepLKNet_A_F_P_kernel_sizes = ((3, 3),
                                      (13, 13),
                                      (13, 13, 13, 13, 13, 13),
                                      (13, 13))
default_UniRepLKNet_N_kernel_sizes = ((3, 3),
                                      (13, 13),
                                      (13, 13, 13, 13, 13, 13, 13, 13),
                                      (13, 13))
default_UniRepLKNet_T_kernel_sizes = ((3, 3, 3),
                                      (13, 13, 13),
                                      (13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3),
                                      (13, 13, 13))
default_UniRepLKNet_S_B_L_XL_kernel_sizes = ((3, 3, 3),
                                             (13, 13, 13),
                                             (13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3),
                                             (13, 13, 13))
UniRepLKNet_A_F_P_depths = (2, 2, 6, 2)
UniRepLKNet_N_depths = (2, 2, 8, 2)
UniRepLKNet_T_depths = (3, 3, 18, 3)
UniRepLKNet_S_B_L_XL_depths = (3, 3, 27, 3)

default_depths_to_kernel_sizes = {
    UniRepLKNet_A_F_P_depths: default_UniRepLKNet_A_F_P_kernel_sizes,
    UniRepLKNet_N_depths: default_UniRepLKNet_N_kernel_sizes,
    UniRepLKNet_T_depths: default_UniRepLKNet_T_kernel_sizes,
    UniRepLKNet_S_B_L_XL_depths: default_UniRepLKNet_S_B_L_XL_kernel_sizes
}

class UniRepLKNet(nn.Module):
    r""" UniRepLKNet
        A PyTorch impl of UniRepLKNet

    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: (3, 3, 27, 3)
        dims (int): Feature dimension at each stage. Default: (96, 192, 384, 768)
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
        kernel_sizes (tuple(tuple(int))): Kernel size for each block. None means using the default settings. Default: None.
        deploy (bool): deploy = True means using the inference structure. Default: False
        with_cp (bool): with_cp = True means using torch.utils.checkpoint to save GPU memory. Default: False
        init_cfg (dict): weights to load. The easiest way to use UniRepLKNet with for OpenMMLab family. Default: None
        attempt_use_lk_impl (bool): try to load the efficient iGEMM large-kernel impl. Setting it to False disabling the iGEMM impl. Default: True
        use_sync_bn (bool): use_sync_bn = True means using sync BN. Use it if your batch size is small. Default: False
    """
    def __init__(self,
                 in_chans=3,
                 num_classes=1000,
                 depths=(3, 3, 27, 3),
                 dims=(96, 192, 384, 768),
                 drop_path_rate=0.,
                 layer_scale_init_value=1e-6,
                 head_init_scale=1.,
                 kernel_sizes=None,
                 deploy=False,
                 with_cp=False,
                 init_cfg=None,
                 attempt_use_lk_impl=True,
                 use_sync_bn=False,
                 **kwargs
                 ):
        super().__init__()

        depths = tuple(depths)
        if kernel_sizes is None:
            if depths in default_depths_to_kernel_sizes:
                print('=========== use default kernel size ')
                kernel_sizes = default_depths_to_kernel_sizes[depths]
            else:
                raise ValueError('no default kernel size settings for the given depths, '
                                 'please specify kernel sizes for each block, e.g., '
                                 '((3, 3), (13, 13), (13, 13, 13, 13, 13, 13), (13, 13))')
        print(kernel_sizes)
        for i in range(4):
            assert len(kernel_sizes[i]) == depths[i], 'kernel sizes do not match the depths'

        self.with_cp = with_cp

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        print('=========== drop path rates: ', dp_rates)

        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(nn.Sequential(
            nn.Conv2d(in_chans, dims[0] // 2, kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0] // 2, eps=1e-6, data_format="channels_first"),
            nn.GELU(),
            nn.Conv2d(dims[0] // 2, dims[0], kernel_size=3, stride=2, padding=1),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")))

        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=3, stride=2, padding=1),
                LayerNorm(dims[i + 1], eps=1e-6, data_format="channels_first")))

        self.stages = nn.ModuleList()

        cur = 0
        for i in range(4):
            main_stage = nn.Sequential(
                *[UniRepLKNetBlock(dim=dims[i], kernel_size=kernel_sizes[i][j], drop_path=dp_rates[cur + j],
                                   layer_scale_init_value=layer_scale_init_value, deploy=deploy,
                                   attempt_use_lk_impl=attempt_use_lk_impl,
                                   with_cp=with_cp, use_sync_bn=use_sync_bn) for j in
                  range(depths[i])])
            self.stages.append(main_stage)
            cur += depths[i]

        last_channels = dims[-1]

        self.for_pretrain = init_cfg is None
        self.for_downstream = not self.for_pretrain     # there may be some other scenarios
        if self.for_downstream:
            assert num_classes is None

        if self.for_pretrain:
            self.init_cfg = None
            self.norm = nn.LayerNorm(last_channels, eps=1e-6)  # final norm layer
            self.head = nn.Linear(last_channels, num_classes)
            self.apply(self._init_weights)
            self.head.weight.data.mul_(head_init_scale)
            self.head.bias.data.mul_(head_init_scale)
            self.output_mode = 'logits'
        else:
            self.init_cfg = init_cfg        # OpenMMLab style init
            self.init_weights()
            self.output_mode = 'features'
            norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
            for i_layer in range(4):
                layer = norm_layer(dims[i_layer])
                layer_name = f'norm{i_layer}'
                self.add_module(layer_name, layer)


    #   load pretrained backbone weights in the OpenMMLab style
    def init_weights(self):

        def load_state_dict(module, state_dict, strict=False, logger=None):
            unexpected_keys = []
            own_state = module.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                    unexpected_keys.append(name)
                    continue
                if isinstance(param, torch.nn.Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    raise RuntimeError(
                        'While copying the parameter named {}, '
                        'whose dimensions in the model are {} and '
                        'whose dimensions in the checkpoint are {}.'.format(
                            name, own_state[name].size(), param.size()))
            missing_keys = set(own_state.keys()) - set(state_dict.keys())

            err_msg = []
            if unexpected_keys:
                err_msg.append('unexpected key in source state_dict: {}\n'.format(', '.join(unexpected_keys)))
            if missing_keys:
                err_msg.append('missing keys in source state_dict: {}\n'.format(', '.join(missing_keys)))
            err_msg = '\n'.join(err_msg)
            if err_msg:
                if strict:
                    raise RuntimeError(err_msg)
                elif logger is not None:
                    logger.warn(err_msg)
                else:
                    print(err_msg)

        logger = get_root_logger()
        assert self.init_cfg is not None
        ckpt_path = self.init_cfg['checkpoint']
        if ckpt_path is None:
            print('================ Note: init_cfg is provided but I got no init ckpt path, so skip initialization')
        else:
            ckpt = _load_checkpoint(ckpt_path, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt

            load_state_dict(self, _state_dict, strict=False, logger=logger)


    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        if self.output_mode == 'logits':
            for stage_idx in range(4):
                x = self.downsample_layers[stage_idx](x)
                x = self.stages[stage_idx](x)
            x = self.norm(x.mean([-2, -1]))
            x = self.head(x)
            return x
        elif self.output_mode == 'features':
            outs = []
            for stage_idx in range(4):
                x = self.downsample_layers[stage_idx](x)
                x = self.stages[stage_idx](x)
                outs.append(self.__getattr__(f'norm{stage_idx}')(x))
            return outs
        else:
            raise ValueError('Defined new output mode?')

    def reparameterize_unireplknet(self):
        for m in self.modules():
            if hasattr(m, 'reparameterize'):
                m.reparameterize()



class LayerNorm(nn.Module):
    r""" LayerNorm implementation used in ConvNeXt
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last", reshape_last_to_first=False):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)
        self.reshape_last_to_first = reshape_last_to_first

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


#   For easy use as backbone in MMDetection framework. Ignore these lines if you do not use MMDetection
if has_mmdet:
    @det_BACKBONES.register_module()
    class UniRepLKNetBackbone(UniRepLKNet):
        def __init__(self,
                     depths=(3, 3, 27, 3),
                     dims=(96, 192, 384, 768),
                     drop_path_rate=0.,
                     layer_scale_init_value=1e-6,
                     kernel_sizes=None,
                     deploy=False,
                     with_cp=False,
                     init_cfg=None,
                     attempt_use_lk_impl=False):
            assert init_cfg is not None
            super().__init__(in_chans=3, num_classes=None, depths=depths, dims=dims,
                             drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value,
                             kernel_sizes=kernel_sizes, deploy=deploy, with_cp=with_cp,
                             init_cfg=init_cfg, attempt_use_lk_impl=attempt_use_lk_impl, use_sync_bn=True)

#   For easy use as backbone in MMSegmentation framework. Ignore these lines if you do not use MMSegmentation
if has_mmseg:
    @seg_BACKBONES.register_module()
    class UniRepLKNetBackbone(UniRepLKNet):
        def __init__(self,
                     depths=(3, 3, 27, 3),
                     dims=(96, 192, 384, 768),
                     drop_path_rate=0.,
                     layer_scale_init_value=1e-6,
                     kernel_sizes=None,
                     deploy=False,
                     with_cp=False,
                     init_cfg=None,
                     attempt_use_lk_impl=False):
            assert init_cfg is not None
            super().__init__(in_chans=3, num_classes=None, depths=depths, dims=dims,
                             drop_path_rate=drop_path_rate, layer_scale_init_value=layer_scale_init_value,
                             kernel_sizes=kernel_sizes, deploy=deploy, with_cp=with_cp,
                             init_cfg=init_cfg, attempt_use_lk_impl=attempt_use_lk_impl, use_sync_bn=True)


model_urls = {
    #TODO: it seems that google drive does not support direct downloading with url? so where to upload the checkpoints other than huggingface? any suggestions?
}

huggingface_file_names = {
    "unireplknet_a_1k": "unireplknet_a_in1k_224_acc77.03.pth",
    "unireplknet_f_1k": "unireplknet_f_in1k_224_acc78.58.pth",
    "unireplknet_p_1k": "unireplknet_p_in1k_224_acc80.23.pth",
    "unireplknet_n_1k": "unireplknet_n_in1k_224_acc81.64.pth",
    "unireplknet_t_1k": "unireplknet_t_in1k_224_acc83.21.pth",
    "unireplknet_s_1k": "unireplknet_s_in1k_224_acc83.91.pth",
    "unireplknet_s_22k": "FILE LOST. REPRODUCING",
    "unireplknet_s_22k_to_1k": "unireplknet_s_in22k_to_in1k_384_acc86.44.pth",
    "unireplknet_b_22k": "unireplknet_b_in22k_pretrain.pth",
    "unireplknet_b_22k_to_1k": "unireplknet_b_in22k_to_in1k_384_acc87.40.pth",
    "unireplknet_l_22k": "unireplknet_l_in22k_pretrain.pth",
    "unireplknet_l_22k_to_1k": "unireplknet_l_in22k_to_in1k_384_acc87.88.pth",
    "unireplknet_xl_22k": "unireplknet_xl_in22k_pretrain.pth",
    "unireplknet_xl_22k_to_1k": "FILE LOST. REPRODUCING"
}

def load_with_key(model, key):
    # if huggingface hub is found, download from our huggingface repo
    if hf_hub_download is not None:
        repo_id = 'DingXiaoH/UniRepLKNet'
        cache_file = hf_hub_download(repo_id=repo_id, filename=huggingface_file_names[key])
        checkpoint = torch.load(cache_file, map_location='cpu')
    else:
        checkpoint = torch.hub.load_state_dict_from_url(url=model_urls[key], map_location="cpu", check_hash=True)
    if 'model' in checkpoint:
        checkpoint = checkpoint['model']
    model.load_state_dict(checkpoint)

def initialize_with_pretrained(model, model_name, in_1k_pretrained, in_22k_pretrained, in_22k_to_1k):
    if in_1k_pretrained:
        key = model_name + '_1k'
    elif in_22k_pretrained:
        key = model_name + '_22k'
    elif in_22k_to_1k:
        key = model_name + '_22k_to_1k'
    else:
        key = None
    if key:
        load_with_key(model, key)

@register_model
def unireplknet_a(in_1k_pretrained=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_A_F_P_depths, dims=(40, 80, 160, 320), **kwargs)
    initialize_with_pretrained(model, 'unireplknet_a', in_1k_pretrained, False, False)
    return model

@register_model
def unireplknet_f(in_1k_pretrained=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_A_F_P_depths, dims=(48, 96, 192, 384), **kwargs)
    initialize_with_pretrained(model, 'unireplknet_f', in_1k_pretrained, False, False)
    return model

@register_model
def unireplknet_p(in_1k_pretrained=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_A_F_P_depths, dims=(64, 128, 256, 512), **kwargs)
    initialize_with_pretrained(model, 'unireplknet_p', in_1k_pretrained, False, False)
    return model

@register_model
def unireplknet_n(in_1k_pretrained=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_N_depths, dims=(80, 160, 320, 640), **kwargs)
    initialize_with_pretrained(model, 'unireplknet_n', in_1k_pretrained, False, False)
    return model

@register_model
def unireplknet_t(in_1k_pretrained=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_T_depths, dims=(80, 160, 320, 640), **kwargs)
    initialize_with_pretrained(model, 'unireplknet_t', in_1k_pretrained, False, False)
    return model

@register_model
def unireplknet_s(in_1k_pretrained=False, in_22k_pretrained=False, in_22k_to_1k=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_S_B_L_XL_depths, dims=(96, 192, 384, 768), **kwargs)
    initialize_with_pretrained(model, 'unireplknet_s', in_1k_pretrained, in_22k_pretrained, in_22k_to_1k)
    return model

@register_model
def unireplknet_b(in_22k_pretrained=False, in_22k_to_1k=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_S_B_L_XL_depths, dims=(128, 256, 512, 1024), **kwargs)
    initialize_with_pretrained(model, 'unireplknet_b', False, in_22k_pretrained, in_22k_to_1k)
    return model

@register_model
def unireplknet_l(in_22k_pretrained=False, in_22k_to_1k=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_S_B_L_XL_depths, dims=(192, 384, 768, 1536), **kwargs)
    initialize_with_pretrained(model, 'unireplknet_l', False, in_22k_pretrained, in_22k_to_1k)
    return model

@register_model
def unireplknet_xl(in_22k_pretrained=False, in_22k_to_1k=False, **kwargs):
    model = UniRepLKNet(depths=UniRepLKNet_S_B_L_XL_depths, dims=(256, 512, 1024, 2048), **kwargs)
    initialize_with_pretrained(model, 'unireplknet_xl', False, in_22k_pretrained, in_22k_to_1k)
    return model



if __name__ == '__main__':
    #   Test case showing the equivalency of Structural Re-parameterization
    x = torch.randn(2, 4, 19, 19)
    layer = UniRepLKNetBlock(4, kernel_size=13, attempt_use_lk_impl=False)
    for n, p in layer.named_parameters():
        if 'beta' in n:
            torch.nn.init.ones_(p)
        else:
            torch.nn.init.normal_(p)
    for n, p in layer.named_buffers():
        if 'running_var' in n:
            print('random init var')
            torch.nn.init.uniform_(p)
            p.data += 2
        elif 'running_mean' in n:
            print('random init mean')
            torch.nn.init.uniform_(p)
    layer.gamma.data += 0.5
    layer.eval()
    origin_y = layer(x)
    layer.reparameterize()
    eq_y = layer(x)
    print(layer)
    print(eq_y - origin_y)
    print((eq_y - origin_y).abs().sum() / origin_y.abs().sum())