import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models import create_model
from models.layers.encoder import ProjEnc


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
        # self.views = self.views.float()
        self.num_views = cfg.num_views
        self.sub_img_size = int(cfg.img_size // math.sqrt(cfg.num_views))
    
        self.enc = ProjEnc(cfg)

        if is_test:
            self.base_model = create_model(cfg.base_model_variant)
        else:
            if cfg.checkpoint_path is not None:
                self.base_model = create_model(cfg.base_model_variant, checkpoint_path=cfg.checkpoint_path)
            else:
                self.base_model = create_model(cfg.base_model_variant, pretrained=True)
        
        if 'resnet' in cfg.base_model_variant:
            self.base_model.num_features = self.base_model.fc.in_features
        
        if cfg.head_type == 'mlp':
            from models.layers.head import MLPHead
            cls_head = MLPHead(self.base_model.num_features, cfg.classes, cfg.mlp_mid_channels, cfg.mlp_dropout_ratio)
        elif cfg.head_type == 'linear':
            cls_head = nn.Linear(self.base_model.num_features, cfg.classes)
        else:
            raise ValueError('cfg.head_type is not defined!')
        
        if 'convnext' in cfg.base_model_variant:
            self.base_model.head.fc = cls_head
        elif 'resnet' in cfg.base_model_variant:
            self.base_model.fc = cls_head
        else:
            self.base_model.head = cls_head
        self.projector = nn.Sequential(
            
        )
        self.loss_ce = nn.CrossEntropyLoss()

    def _fix_weight(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

        # learnable cls token
        if 'vit' in self.cfg.base_model_variant:
            self.base_model.cls_token.requires_grad = True

        # learnable cls head parameters
        if 'convnext' in self.cfg.base_model_variant:
            for param in self.base_model.head.fc.parameters():
                param.requires_grad = True
        elif 'resnet' in self.cfg.base_model_variant:
            for param in self.base_model.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.base_model.head.parameters():
                param.requires_grad = True

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

    def forward(self, pc, original_pc):
        
        original_pc = torch.repeat_interleave(original_pc, self.num_views, dim=0)
        # pc = self.pc_views.point_transform(pc)
        pc = self.point_transform(pc)
        
        img = self.enc(original_pc, pc)  # enc将点云投影为含有可学习颜色的图像
        
        img = F.interpolate(img, (self.sub_img_size, self.sub_img_size), mode='nearest-exact')
        b, c, h, w = img.shape
        img = img.reshape(b // self.num_views, self.num_views, c, h, w)
        multi_view_imgs = [img[:, i, ...] for i in range(img.shape[1])]
        img = torch.cat(
            [
                torch.cat([i for i in multi_view_imgs[:2]], dim=2),
                torch.cat([i for i in multi_view_imgs[2:]], dim=2)
                ],
            dim=3
        )
        out = self.base_model(img)  # base model为冻结住的预训练模型
        return out
    
    def point_transform(self, points: torch.Tensor):
        # view_idx = np.random.choice(self.views.shape[0], self.num_views, replace=False)
        # views = self.views[view_idx]
        views = self.views[:4]
        # angle = torch.tensor(views[:, 0, :]).float().cuda()
        # self.rot_mat = euler2mat(angle).transpose(1, 2)
        # self.translation = torch.tensor(views[:, 1, :]).float().cuda()
        # self.translation = self.translation.unsqueeze(1)
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

    # zero = torch.zeros([b], requires_grad=False, device=angle.device)[0]
    # one = torch.ones([b], requires_grad=False, device=angle.device)[0]
    zero = z.detach()*0
    one = zero.detach()+1
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


# class PCViews(nn.Module):
#     def __init__(self, num_views):
#         TRANS = -1.4
#         self.views = torch.from_numpy(
#             np.asarray([
#                 [[0 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
#                 [[1 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
#                 [[2 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
#                 [[3 * np.pi / 2, 0, np.pi / 2], [0, 0, TRANS]],
#                 [[5 * np.pi / 4, -np.pi / 4, np.pi / 2], [0, 0, TRANS]], 
#                 [[5 * np.pi / 4, np.pi / 4, np.pi / 2], [0, 0, TRANS]], 
#                 [[7 * np.pi / 4, -np.pi / 4, np.pi / 2], [0, 0, TRANS]], 
#                 [[7 * np.pi / 4, np.pi / 4, np.pi / 2], [0, 0, TRANS]],  
#                 [[0, -np.pi / 2, np.pi / 2], [0, 0, TRANS]],
#                 [[0, np.pi / 2, np.pi / 2], [0, 0, TRANS]]
#             ])
#         )
#         self.views = self.views.float().cuda()
#         self.num_views = num_views
        
#     def point_transform(self, points: torch.Tensor):
#         """
#         :param points: [batch, num_points, 3]
#         :return:
#         """
#         view_idx = np.random.choice(self.views.shape[0], self.num_views, replace=False)
#         views = self.views[view_idx]
        
#         # angle = torch.tensor(views[:, 0, :]).float().cuda()
#         # self.rot_mat = euler2mat(angle).transpose(1, 2)
#         # self.translation = torch.tensor(views[:, 1, :]).float().cuda()
#         # self.translation = self.translation.unsqueeze(1)
#         angle = views[:, 0, :]
#         self.rot_mat = euler2mat(angle).transpose(1, 2)
#         self.translation = views[:, 1, :]
#         self.translation = self.translation.unsqueeze(1)
        
#         b = points.shape[0]
#         v = self.translation.shape[0]
        
#         points = torch.repeat_interleave(points, v, dim=0)
#         rot_mat = self.rot_mat.repeat(b, 1, 1).to(points.device)
#         translation = self.translation.repeat(b, 1, 1).to(points.device)
        
#         points = torch.matmul(points, rot_mat)
#         points = points - translation
#         return points