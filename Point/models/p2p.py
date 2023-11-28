import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from timm.models import create_model
from models.layers.encoder import ProjEnc


class AdaptorCls(nn.Module):
    def __init__(self, num_views, in_features):
        super().__init__()
        self.num_views = num_views
        self.in_features = in_features
        self.adapter_ratio = 0.6
        self.fusion_init = 0.5
        self.dropout = 0.075
        self.fusion_ratio = nn.Parameter(torch.tensor([self.fusion_init] * self.num_views), requires_grad=True)
        
        # self.global_f = nn.Sequential(
        #         nn.BatchNorm3d(self.in_features),
        #         nn.Dropout(self.dropout),
        #         nn.Conv3d(in_channels=self.in_features,
        #                   out_channels=self.in_features,
        #                   kernel_size=(self.num_views, 1, 1)),
        #         nn.BatchNorm3d(self.in_features),
        #         nn.ReLU(),
        #         nn.Dropout(self.dropout))

        self.global_f = nn.Sequential(
                nn.BatchNorm2d(self.in_features),
                nn.Dropout(self.dropout),
                nn.Conv2d(in_channels=self.in_features,
                          out_channels=self.in_features,
                          kernel_size=(self.num_views, 1)),
                nn.BatchNorm2d(self.in_features),
                nn.ReLU(),
                nn.Dropout(self.dropout))

        # self.view_f = nn.Sequential(
        #         nn.Conv2d(in_channels=self.in_features,
        #                   out_channels=self.in_features,
        #                   kernel_size=1),
        #         nn.ReLU(),
        #         nn.Conv2d(in_channels=self.in_features,
        #                   out_channels=self.in_features * self.num_views,
        #                   kernel_size=1),
                # nn.ReLU())
        self.view_f = nn.Sequential(
                nn.Conv1d(in_channels=self.in_features,
                          out_channels=self.in_features,
                          kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(in_channels=self.in_features,
                          out_channels=self.in_features * self.num_views,
                          kernel_size=1),
                nn.ReLU())
        
    def forward(self, feat: torch.Tensor):
        # B, C, H, W = feat.shape
        # img_feat = feat.reshape(B // self.num_views, self.num_views, C, H, W).permute(0, 2, 1, 3, 4)  # batch x channel x num_views x H x W
        # res_feat = feat.reshape(B // self.num_views, self.num_views * C, H, W)
        B, C = feat.shape[0], feat.shape[1]
        img_feat = feat.reshape(B // self.num_views, self.num_views, C, -1).permute(0, 2, 1, 3)  # batch x channel x num_views x (H x W)
        res_feat = feat.reshape(B // self.num_views, self.num_views * C, -1)
        
        # Global feature
        # global_feat = self.global_f(img_feat * self.fusion_ratio.reshape(1, 1, -1, 1, 1))
        global_feat = self.global_f(img_feat * self.fusion_ratio.reshape(1, 1, -1, 1))
        
        # global_feat = global_feat.squeeze()
        # View-wise adapted features
        view_feat = self.view_f(global_feat)
        
        img_feat = view_feat * self.adapter_ratio + res_feat * (1 - self.adapter_ratio)
        return img_feat
    

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
                    self.base_model = create_model(cfg.base_model_variant, pretrained=True)
                
        
        if 'resnet' in cfg.base_model_variant:
            self.base_model.num_features = self.base_model.fc.in_features
        if cfg.head_type == 'mlp':
            from models.layers.head import MLPHead
            self.cls_head = MLPHead(cfg.num_features, cfg.classes, cfg.mlp_mid_channels, cfg.mlp_dropout_ratio)
        elif cfg.head_type == 'pooling_mlp':
            from models.layers.head import PoolingClsHead
            self.cls_head = PoolingClsHead(cfg.num_head_features, cfg.classes)  # , cfg.mlp_mid_channels, cfg.mlp_dropout_ratio
        elif cfg.head_type == 'vit_cls_head':
            from models.layers.head import ViTClsHead
            self.cls_head = ViTClsHead(cfg.num_head_features, cfg.classes)
        elif cfg.head_type == 'linear':
            self.cls_head = nn.Linear(self.base_model.num_features, cfg.classes)
        else:
            raise ValueError('cfg.head_type is not defined!')
        
        # if 'convnext' in cfg.base_model_variant:
        #     self.base_model.head.fc = self.cls_head
        # elif 'resnet' in cfg.base_model_variant:
        #     self.base_model.fc = self.cls_head
        # else:
        #     self.base_model.head = self.cls_head
        # self.projector = nn.Sequential(
            
        # )
        # self.adaptor = AdaptorCls(num_views=cfg.num_views, in_features=cfg.num_features)
        self.loss_ce = nn.CrossEntropyLoss()
    
    def _set_last_feat_hook(self):
        # TODO
        def layer_hook(module, inp, out):
            self.last_feats = out
        return self.base_model.layer4.register_forward_hook(layer_hook)

    def _fix_weight(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

        # learnable cls token
        # if 'vit' in self.cfg.base_model_variant:
        #     self.base_model.cls_token.requires_grad = True

        # # learnable cls head parameters
        # if 'convnext' in self.cfg.base_model_variant:
        #     for param in self.base_model.head.fc.parameters():
        #         param.requires_grad = True
        # elif 'resnet' in self.cfg.base_model_variant:
        #     for param in self.base_model.fc.parameters():
        #         param.requires_grad = True
        # else:
        #     for param in self.base_model.head.parameters():
        #         param.requires_grad = True

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
        
        # ######################### multi view start ##############################
        original_pc = torch.repeat_interleave(original_pc, self.num_views, dim=0)
        pc = self.point_transform(pc)
        # ########################## multi view end ###############################
        
        # ########################### raw p2p #############################
        img = self.enc(original_pc, pc)  # enc将点云投影为含有可学习颜色的图像
        # out = self.base_model(img)  # base model为冻结住的预训练模型
        # ######################### raw p2p end ###########################
        
        # ######################## cls head ###############################
        if 'resnet' in self.base_model_name:
            out = self.base_model(img)
            out = self.last_feats
        else:
            out = self.base_model.forward_features(img)  # base model为冻结住的预训练模型
        # out = self.adaptor(out)
        out = self.cls_head(out)
        # ###################### cls head end #############################
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