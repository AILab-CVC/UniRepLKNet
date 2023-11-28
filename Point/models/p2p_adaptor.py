import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
import numpy as np
from timm.models import create_model
from models.layers.encoder import ProjEnc


class BatchNormPoint(nn.Module):
    def __init__(self, feat_size, sync_bn=False):
        super().__init__()
        self.feat_size = feat_size
        self.sync_bn = sync_bn
        self.bn = nn.BatchNorm1d(feat_size)

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 3
        s1, s2, s3 = x.shape[0], x.shape[1], x.shape[2]
        assert s3 == self.feat_size
        if self.sync_bn:
            # 4d input for BatchNorm2dSync
            x = x.view(s1 * s2, self.feat_size, 1, 1)
            x = self.bn(x)
        else:
            x = x.view(s1 * s2, self.feat_size)
            x = self.bn(x)
        return x.view(s1, s2, s3)

class Adapter(nn.Module):
    def __init__(self, num_views, in_features):
        super().__init__()
        self.num_views = num_views
        self.in_features = in_features
        self.multi_view_feature_num = self.num_views * self.in_features
        self.avg_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.SE_module = nn.Sequential(
            nn.Linear(in_features=self.multi_view_feature_num,
                      out_features=self.multi_view_feature_num),
            nn.ReLU(),
            nn.Linear(in_features=self.multi_view_feature_num,
                      out_features=self.multi_view_feature_num),
            nn.Sigmoid()
        )

    def forward(self, feat: torch.Tensor):
        B, C, H, W = feat.shape
        feat = feat.reshape(B // self.num_views, self.num_views * C, H, W)
        feat_score = self.avg_pooling(feat).squeeze()
        feat_score = self.SE_module(feat_score)
        feat = feat * feat_score[..., None, None]
        return feat


class P2P(nn.Module):
    def __init__(self, cfg, is_test=False):
        super().__init__()
        self.cfg = cfg

        # self.pc_views = PCViews(4)
        # self.num_views = self.pc_views.num_views
        TRANS = -1.4
        self.views = nn.Parameter(
            torch.tensor(
                np.array([
                    [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
                    [[1 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
                    [[2 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
                    [[3 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
                    [[5 * np.pi / 4, -np.pi / 4, np.pi / 2], [0, 0, TRANS]],
                    [[5 * np.pi / 4, np.pi / 4, np.pi / 2], [0, 0, TRANS]],
                    [[7 * np.pi / 4, -np.pi / 4, np.pi / 2], [0, 0, TRANS]],
                    [[7 * np.pi / 4, np.pi / 4, np.pi / 2], [0, 0, TRANS]],
                    [[0, -np.pi / 2, np.pi / 2], [0, 0, TRANS]],
                    [[0, np.pi / 2, np.pi / 2], [0, 0, TRANS]]
                ]), dtype=torch.float32
            ), requires_grad=False)
        
        self.num_views = cfg.num_views

        self.enc = ProjEnc(cfg)

        self.base_model_name = cfg.base_model_variant
        if is_test:
            if 'resnet' in cfg.base_model_variant:
                self.base_model = create_model(cfg.base_model_variant, features_only=True)
                self.last_feat_hook_handle = self._set_last_feat_hook()
            else:
                self.base_model = create_model(cfg.base_model_variant)
        else:
            if cfg.checkpoint_path is not None:
                self.base_model = create_model(cfg.base_model_variant, checkpoint_path=cfg.checkpoint_path)
            else:
                # self.base_model = create_model(cfg.base_model_variant, pretrained=True)
                if 'resnet' in cfg.base_model_variant:
                    self.base_model = create_model(cfg.base_model_variant, pretrained=True, features_only=True)
                    self.last_feat_hook_handle = self._set_last_feat_hook()
                else:
                    # self.base_model = create_model(cfg.base_model_variant, pretrained=True)
                    self.base_model = create_model(cfg.base_model_variant, pretrained=False)

        if 'resnet' in cfg.base_model_variant:
            self.base_model.num_features = self.base_model.fc.in_features
        if cfg.head_type == 'mlp':
            from models.layers.head import MLPHead
            self.cls_head = MLPHead(cfg.num_features, cfg.classes, cfg.mlp_mid_channels, cfg.mlp_dropout_ratio)
        elif cfg.head_type == 'pooling_mlp':
            from models.layers.head import PoolingClsHead
            self.cls_head = PoolingClsHead(cfg)  # , cfg.mlp_mid_channels, cfg.mlp_dropout_ratio
        elif cfg.head_type == 'vit_cls_head':
            from models.layers.head import ViTClsHead
            self.cls_head = ViTClsHead(cfg)
        elif cfg.head_type == 'linear':
            self.cls_head = nn.Linear(cfg.view_feature, cfg.classes)
        elif cfg.head_type == 'token_cls_head':
            from models.layers.decoder import TokenClsHead
            self.cls_head = TokenClsHead(cfg)
        elif cfg.head_type == 'token_cls_head_v2':
            from models.layers.decoder import TokenClsHeadV2
            self.cls_head = TokenClsHeadV2(cfg)
        else:
            raise ValueError('cfg.head_type is not defined!')

        self.loss_ce = nn.CrossEntropyLoss()

    def _set_last_feat_hook(self):
        def layer_hook(module, inp, out):
            self.last_feats = out

        return self.base_model.layer4.register_forward_hook(layer_hook)

    def _fix_weight(self):
        for param in self.base_model.parameters():
            param.requires_grad = False
        # flexible learnable parameters
        if self.cfg.update_type is not None:
            for name, param in self.base_model.named_parameters():
                if self.cfg.update_type in name:
                    param.requires_grad = True
            print('Learnable {} parameters!'.format(self.cfg.update_type))

    def get_loss_acc(self, pred, gt, smoothing=True):
        gt = gt.contiguous().view(-1).long()

        if smoothing:
            eps = 0.2
            n_class = pred.size(1)

            one_hot = torch.zeros_like(pred).scatter(1, gt.view(-1, 1), 1)
            one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
            log_prb = F.log_softmax(pred, dim=1)

            loss_cls = -(one_hot * log_prb).sum(dim=1).mean()
        else:
            loss_cls = self.loss_ce(pred, gt.long())

        pred = pred.argmax(-1)
        acc = (pred == gt).sum() / float(gt.size(0))

        return loss_cls, acc * 100

    # @autocast(dtype=torch.float16)
    def forward(self, pc, original_pc):

        original_pc = torch.repeat_interleave(original_pc, self.num_views, dim=0)
        pc = self.point_transform(pc)
        img = self.enc(original_pc, pc) 
        if 'resnet' in self.base_model_name:
            out = self.base_model(img)
            out = self.last_feats
        else:
            out = self.base_model.forward_features(img) 
        out = self.cls_head(out)
        return out

    def point_transform(self, points: torch.Tensor):
        views = self.views[:self.num_views]
        angle = views[:, 0, :]
        self.rot_mat = euler2mat(angle).transpose(1, 2)
        self.translation = views[:, 1, :]
        self.translation = self.translation.unsqueeze(1)

        b = points.shape[0]
        v = self.translation.shape[0]

        points = torch.repeat_interleave(points, v, dim=0)
        rot_mat = self.rot_mat.repeat(b, 1, 1)
        translation = self.translation.repeat(b, 1, 1)

        points = torch.matmul(points, rot_mat)
        points = points - translation
        return points


def euler2mat(angle: torch.Tensor):
    """Convert euler angles to rotation matrix.
     :param angle: [3] or [b, 3]
     :return
        rotmat: [3] or [b, 3, 3]
    source
    https://github.com/ClementPinard/SfmLearner-Pytorch/blob/master/inverse_warp.py
    """

    if len(angle.size()) == 1:
        x, y, z = angle[0], angle[1], angle[2]
        _dim = 0
        _view = [3, 3]
    elif len(angle.size()) == 2:
        b, _ = angle.size()
        x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]
        _dim = 1
        _view = [b, 3, 3]

    else:
        assert False

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zero = z.detach() * 0
    one = zero.detach() + 1
    zmat = torch.stack([cosz, -sinz, zero,
                        sinz, cosz, zero,
                        zero, zero, one], dim=_dim).reshape(_view)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zero, siny,
                        zero, one, zero,
                        -siny, zero, cosy], dim=_dim).reshape(_view)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([one, zero, zero,
                        zero, cosx, -sinx,
                        zero, sinx, cosx], dim=_dim).reshape(_view)

    rot_mat = xmat @ ymat @ zmat
    # print(rot_mat)
    return rot_mat
