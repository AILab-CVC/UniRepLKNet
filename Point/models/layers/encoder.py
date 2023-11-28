import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter

from models.layers.utils import knn_point
from timm.models.resnet import BasicBlock, Bottleneck
from models.layers.fusion import MultiViewFusion, MultiViewFusionV2, MultiViewFusionSE
import torchvision.transforms as transforms

class ProjEnc(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.local_size = cfg.local_size
        self.trans_dim = cfg.trans_dim
        self.graph_dim = cfg.graph_dim
        self.imgblock_dim = cfg.imgblock_dim
        self.img_size = cfg.img_size
        self.obj_size = cfg.obj_size
        # self.sub_img_size = cfg.sub_img_size
        self.num_views = cfg.num_views
        self.atten_fusion = cfg.atten_fusion
        self.imagenet_mean = torch.Tensor(cfg.imagenet_default_mean)
        self.imagenet_std = torch.Tensor(cfg.imagenet_default_std)

        self.input_trans = nn.Conv1d(3, self.trans_dim, 1)
        self.graph_layer = nn.Sequential(nn.Conv2d(self.trans_dim * 2, self.graph_dim, kernel_size=1, bias=False),
                                         nn.GroupNorm(4, self.graph_dim),
                                         nn.LeakyReLU(negative_slope=0.2)
                                         )
        self.proj_layer = nn.Conv1d(self.graph_dim, self.graph_dim, kernel_size=1)

        self.img_block = nn.Sequential(
            BasicBlock(self.graph_dim, self.graph_dim),
            nn.Conv2d(self.graph_dim, self.graph_dim, kernel_size=1),
        )

        # ################# multi view feature fusion Start #########################
        if self.atten_fusion:
            # self.fusion = MultiViewFusionSE(cfg)   
            # self.fusion = MultiViewFusion(cfg)
            self.fusion = MultiViewFusionV2(cfg)
        # ################## multi view feature fusion End ##########################

        self.img_layer = nn.Conv2d(self.graph_dim, 3, kernel_size=1)

        self.offset = torch.Tensor([[-1, -1], [-1, 0], [-1, 1],
                                    [0, -1], [0, 0], [0, 1],
                                    [1, -1], [1, 0], [1, 1]])

    @staticmethod
    def get_graph_feature(coor_q: torch.Tensor, x_q: torch.Tensor, coor_k: torch.Tensor, x_k: torch.Tensor, k: int):
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            idx = knn_point(k, coor_k.transpose(1, 2).contiguous(), coor_q.transpose(1, 2).contiguous())  # B G k
            idx = idx.transpose(1, 2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature

    def forward(self, original_pc: torch.Tensor, pc: torch.Tensor):
        B, N, _ = pc.shape

        # calculate range
        pc_range = pc.max(dim=1)[0] - pc.min(dim=1)[0]  # B 3
        grid_size = pc_range[:, :2].max(dim=-1)[0] / (self.obj_size - 3)  # B,
        idx_xy = torch.floor(
            (pc[:, :, :2] - pc.min(dim=1)[0][:, :2].unsqueeze(dim=1)) / grid_size.unsqueeze(dim=1).unsqueeze(
                dim=2))  # B N 2
        idx_xy_dense = (idx_xy.unsqueeze(dim=2) + self.offset.unsqueeze(dim=0).unsqueeze(dim=0).to(pc.device)).view(
            idx_xy.size(0), N * 9, 2) + 1
        # B N 1 2 + 1 1 9 2 -> B N 9 2 -> B 9N 2
        idx_xy_dense_center = torch.floor(
            (idx_xy_dense.max(dim=1)[0] + idx_xy_dense.min(dim=1)[0]) / 2).int()  # B x N 每个点云object的2D中心
        offset_x = self.obj_size / 2 - idx_xy_dense_center[:, 0:1] - 1
        offset_y = self.obj_size / 2 - idx_xy_dense_center[:, 1:2] - 1
        idx_xy_dense_offset = idx_xy_dense + torch.cat([offset_x, offset_y], dim=1).unsqueeze(dim=1)

        # Get features
        original_pc = original_pc.transpose(1, 2).contiguous()  # B, 3, N
        f = self.input_trans(original_pc)  # batch x channel x npoints
        f = self.get_graph_feature(original_pc, f, original_pc, f, self.local_size)
        f = self.graph_layer(f)
        f = f.max(dim=-1, keepdim=False)[0]  # B C N

        f = self.proj_layer(f).transpose(1, 2).contiguous()  # B N C

        f_dense = f.unsqueeze(dim=2).expand(-1, -1, 9, -1).contiguous().view(f.size(0), N * 9, self.graph_dim)
        # B N 9 C -> B 9N C
        assert idx_xy_dense_offset.min() >= 0 and idx_xy_dense_offset.max() <= (self.obj_size - 1), str(
            idx_xy_dense_offset.min()) + '-' + str(idx_xy_dense_offset.max())

        # change idx to 1-dim
        new_idx_xy_dense = idx_xy_dense_offset[:, :, 0] * self.obj_size + idx_xy_dense_offset[:, :, 1]
        # scatter the features    
        out = scatter(f_dense, new_idx_xy_dense.long(), dim=1, reduce="sum")

        # need to pad
        if out.size(1) < self.obj_size * self.obj_size:
            delta = self.obj_size * self.obj_size - out.size(1)
            zero_pad = torch.zeros(out.size(0), delta, out.size(2)).to(out.device)
            res = torch.cat([out, zero_pad], dim=1).reshape((out.size(0), self.obj_size, self.obj_size, out.size(2)))
        else:
            res = out.reshape((out.size(0), self.obj_size, self.obj_size, out.size(2)))
        if self.obj_size < self.img_size:
            # pad to 256
            pad_size = self.img_size - self.obj_size
            zero_pad_h = torch.zeros(out.size(0), int(pad_size // 2), self.obj_size, out.size(2)).to(out.device)
            zero_pad_w = torch.zeros(out.size(0), self.img_size, int(pad_size // 2), out.size(2)).to(out.device)
            res = torch.cat([zero_pad_h, res, zero_pad_h], dim=1)
            res = torch.cat([zero_pad_w, res, zero_pad_w], dim=2)

        # B 224 224 C
        # ################# multi view feature fusion Start #########################
        res = res.permute(0, 3, 1, 2).contiguous()
   
        img_feat = self.img_block(res)
        if self.atten_fusion:
            img_feat = self.fusion(img_feat)
        # ################## multi view feature fusion End ##########################

        img = self.img_layer(img_feat)  # B 3 224 224

        mean_vec = self.imagenet_mean.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).to(img.device)  # 1 3 1 1
        std_vec = self.imagenet_std.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).to(img.device)  # 1 3 1 1
        # Normalize the pic        
        img = torch.sigmoid(img)
        img_norm = img.sub(mean_vec).div(std_vec)

        return img_norm


class ProjEncRAW(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.local_size = cfg.local_size
        self.trans_dim = cfg.trans_dim
        self.graph_dim = cfg.graph_dim
        self.imgblock_dim = cfg.imgblock_dim
        self.img_size = cfg.img_size
        self.obj_size = cfg.obj_size
        self.imagenet_mean = torch.Tensor(cfg.imagenet_default_mean)
        self.imagenet_std = torch.Tensor(cfg.imagenet_default_std)

        self.input_trans = nn.Conv1d(3, self.trans_dim, 1)
        self.graph_layer = nn.Sequential(nn.Conv2d(self.trans_dim*2, self.graph_dim, kernel_size=1, bias=False),
                                        nn.GroupNorm(4, self.graph_dim),
                                        nn.LeakyReLU(negative_slope=0.2)
                                        )
        self.proj_layer = nn.Conv1d(self.graph_dim, self.graph_dim, kernel_size=1)
        
        self.img_block = nn.Sequential(
            BasicBlock(self.graph_dim, self.graph_dim),
            nn.Conv2d(self.graph_dim, self.graph_dim, kernel_size=1),
        )
        self.img_layer = nn.Conv2d(self.graph_dim, 3, kernel_size=1)

        self.offset = torch.Tensor([[-1, -1], [-1, 0], [-1, 1], 
                                    [0, -1], [0, 0], [0, 1],
                                    [1, -1], [1, 0], [1, 1]])

    @staticmethod
    def get_graph_feature(coor_q, x_q, coor_k, x_k, k):
        batch_size = x_k.size(0)
        num_points_k = x_k.size(2)
        num_points_q = x_q.size(2)

        with torch.no_grad():
            idx = knn_point(k, coor_k.transpose(1, 2).contiguous(), coor_q.transpose(1, 2).contiguous()) # B G k
            idx = idx.transpose(1, 2).contiguous()
            assert idx.shape[1] == k
            idx_base = torch.arange(0, batch_size, device=x_q.device).view(-1, 1, 1) * num_points_k
            idx = idx + idx_base
            idx = idx.view(-1)
        num_dims = x_k.size(1)
        x_k = x_k.transpose(2, 1).contiguous()
        feature = x_k.view(batch_size * num_points_k, -1)[idx, :]
        feature = feature.view(batch_size, k, num_points_q, num_dims).permute(0, 3, 2, 1).contiguous()
        x_q = x_q.view(batch_size, num_dims, num_points_q, 1).expand(-1, -1, -1, k)
        feature = torch.cat((feature - x_q, x_q), dim=1)
        return feature
    
    def forward(self, original_pc, pc):
        B, N, _ = pc.shape
        
        # calculate range
        pc_range = pc.max(dim=1)[0] - pc.min(dim=1)[0]  # B 3
        grid_size = pc_range[:, :2].max(dim=-1)[0] / (self.obj_size - 3)  # B,
        idx_xy = torch.floor((pc[:, :, :2] - pc.min(dim=1)[0][:, :2].unsqueeze(dim=1)) / grid_size.unsqueeze(dim=1).unsqueeze(dim=2))  # B N 2
        idx_xy_dense = (idx_xy.unsqueeze(dim=2) + self.offset.unsqueeze(dim=0).unsqueeze(dim=0).to(pc.device)).view(idx_xy.size(0), N*9, 2) + 1
        # B N 1 2 + 1 1 9 2 -> B N 9 2 -> B 9N 2
        idx_xy_dense_center = torch.floor((idx_xy_dense.max(dim=1)[0] + idx_xy_dense.min(dim=1)[0]) / 2).int()
        offset_x = self.obj_size / 2 - idx_xy_dense_center[:, 0:1] - 1
        offset_y = self.obj_size / 2 - idx_xy_dense_center[:, 1:2] - 1
        idx_xy_dense_offset = idx_xy_dense + torch.cat([offset_x, offset_y], dim=1).unsqueeze(dim=1)

        # Get features
        original_pc = original_pc.transpose(1, 2).contiguous()      # B, 3, N
        f = self.input_trans(original_pc)
        f = self.get_graph_feature(original_pc, f, original_pc, f, self.local_size)
        f = self.graph_layer(f)
        f = f.max(dim=-1, keepdim=False)[0]       # B C N
        
        f = self.proj_layer(f).transpose(1, 2).contiguous()        # B N C

        f_dense = f.unsqueeze(dim=2).expand(-1, -1, 9, -1).contiguous().view(f.size(0), N * 9, self.graph_dim) 
        # B N 9 C -> B 9N C
        assert idx_xy_dense_offset.min() >= 0 and idx_xy_dense_offset.max() <= (self.obj_size-1), str(idx_xy_dense_offset.min()) + '-' + str(idx_xy_dense_offset.max())
        
        # change idx to 1-dim
        new_idx_xy_dense = idx_xy_dense_offset[:, :, 0] * self.obj_size + idx_xy_dense_offset[:, :, 1]
        # scatter the features    
        out = scatter(f_dense, new_idx_xy_dense.long(), dim=1, reduce="sum") 

        #need to pad 
        if out.size(1) < self.obj_size * self.obj_size: 
            delta = self.obj_size * self.obj_size - out.size(1) 
            zero_pad = torch.zeros(out.size(0), delta, out.size(2)).to(out.device) 
            res = torch.cat([out, zero_pad], dim=1).reshape((out.size(0), self.obj_size, self.obj_size, out.size(2))) 
        else: 
            res = out.reshape((out.size(0), self.obj_size, self.obj_size, out.size(2))) 
        if self.obj_size < self.img_size:
            # pad to 256
            pad_size = self.img_size - self.obj_size
            zero_pad_h = torch.zeros(out.size(0), int(pad_size // 2), self.obj_size, out.size(2)).to(out.device)
            zero_pad_w = torch.zeros(out.size(0), self.img_size, int(pad_size // 2), out.size(2)).to(out.device)
            res = torch.cat([zero_pad_h, res, zero_pad_h], dim=1)
            res = torch.cat([zero_pad_w, res, zero_pad_w], dim=2)
        # B 224 224 C
        img_feat = self.img_block(res.permute(0, 3, 1, 2).contiguous())
        img = self.img_layer(img_feat)  # B 3 224 224
        mean_vec = self.imagenet_mean.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).to(img.device)  # 1 3 1 1
        std_vec = self.imagenet_std.unsqueeze(dim=0).unsqueeze(dim=2).unsqueeze(dim=3).to(img.device)   # 1 3 1 1
        # Normalize the pic        
        img = nn.Sigmoid()(img)
        img_norm = img.sub(mean_vec).div(std_vec)

        return img_norm