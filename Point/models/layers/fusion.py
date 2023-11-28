import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.transformer import Transformer
from einops import rearrange


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


class SEModule(nn.Module):
    """ SE Module as defined in original SE-Nets with a few additions
    Additions include:
        * divisor can be specified to keep channels % div == 0 (default: 8)
        * reduction channels can be specified directly by arg (if rd_channels is set)
        * reduction channels can be specified by float rd_ratio (default: 1/16)
        * global max pooling can be added to the squeeze aggregation
        * customizable activation, normalization, and gate layer
    """

    def __init__(
            self, channels, rd_ratio=1. / 16, rd_channels=None, rd_divisor=8, add_maxpool=False,
            act_layer=nn.ReLU, norm_layer=None):
        super(SEModule, self).__init__()
        self.add_maxpool = add_maxpool
        if not rd_channels:
            rd_channels = make_divisible(channels * rd_ratio, rd_divisor, round_limit=0.)
        self.fc1 = nn.Conv2d(channels, rd_channels, kernel_size=1, bias=True)
        self.bn = norm_layer(rd_channels) if norm_layer else nn.Identity()
        self.act = act_layer(inplace=True)
        self.fc2 = nn.Conv2d(rd_channels, channels, kernel_size=1, bias=True)
        self.gate = nn.Sigmoid()

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        if self.add_maxpool:
            # experimental codepath, may remove or change
            x_se = 0.5 * x_se + 0.5 * x.amax((2, 3), keepdim=True)
        x_se = self.fc1(x_se)
        x_se = self.act(self.bn(x_se))
        x_se = self.fc2(x_se)
        return x * self.gate(x_se)


class MultiViewFusionSE(nn.Module):
    def __init__(self, cfg, act_layer=nn.ReLU, norm_layer=nn.BatchNorm2d):
        super(MultiViewFusionSE, self).__init__()

        inchannel = cfg.graph_dim * cfg.num_views
        outchannel = cfg.graph_dim * cfg.num_views
        self.num_views = cfg.num_views
        self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(inchannel)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=1, bias=False)
        self.bn2 = norm_layer(outchannel)

        self.se = SEModule(outchannel)

        self.act2 = act_layer(inplace=True)

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        x = x.reshape(B // self.num_views, self.num_views * C, H, W)
        shortcut = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.se(x)
        x += shortcut
        x = self.act2(x)
        x = x.reshape(B, C, H, W)
        return x


# class MultiViewFusion(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.img_size = cfg.img_size
#         self.graph_dim = cfg.graph_dim
#         self.num_views = cfg.num_views
#         self.interpolate_size = 16
#         # self.pooling = nn.AvgPool2d(kernel_size=14, stride=14)
#         # self.conv1 = nn.Sequential(
#         #     nn.Conv2d(in_channels=cfg.graph_dim, out_channels=128, kernel_size=1),
#         #     nn.ReLU()
#         #     )
#         self.conv1 = nn.Conv2d(in_channels=cfg.graph_dim, out_channels=128, kernel_size=14, stride=14, bias=False)
#         self.transformer = Transformer(128, depth=1, heads=8, dim_head=128, mlp_dim=256, selfatt=True)
#         self.conv2 = nn.Conv2d(in_channels=128, out_channels=cfg.graph_dim, kernel_size=1)
#     def forward(self, feats: torch.Tensor):
#         feats = self.conv1(feats)
#         B, C = feats.shape[:2]
#         # feats = self.pooling(feats)
#         feats = rearrange(feats.reshape(B // self.num_views, self.num_views, C, self.interpolate_size**2), 'B N C S -> B (N S) C')
#         feats = self.transformer(feats)
#         feats = rearrange(feats, 'B (N S) C -> (B N) C S', N=self.num_views).reshape(B, C, self.interpolate_size, self.interpolate_size)
#         feats = self.conv2(feats)
#         feats = F.interpolate(feats, size=self.img_size)
#         return feats


class MultiViewFusion(nn.Module):
    def __init__(self, cfg):
        super(MultiViewFusion, self).__init__()
        self.img_size = cfg.img_size
        self.graph_dim = cfg.graph_dim
        self.num_views = cfg.num_views
        self.kernel_size = 7
        self.sub_image_size = self.img_size // self.kernel_size
        self.conv1 = nn.Conv2d(in_channels=cfg.graph_dim, out_channels=128, kernel_size=self.kernel_size,
                               stride=self.kernel_size, bias=False)
        self.transformer = Transformer(128, depth=1, heads=8, dim_head=128, mlp_dim=256, selfatt=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=cfg.graph_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, feats: torch.Tensor):
        shortcut = feats
        feats = self.conv1(feats)
        B, C = feats.shape[:2]
        # feats = self.pooling(feats)
        feats = rearrange(feats.reshape(B // self.num_views, self.num_views, C, self.sub_image_size ** 2),
                          'B N C S -> B (N S) C')
        feats = self.transformer(feats)
        feats = rearrange(feats, 'B (N S) C -> (B N) C S', N=self.num_views).reshape(B, C, self.sub_image_size,
                                                                                     self.sub_image_size)
        feats = F.interpolate(feats, size=self.img_size)
        feats = self.conv2(feats)
        feats = feats + shortcut
        feats = self.relu(feats)
        return feats


class MultiViewFusionV2(nn.Module):
    def __init__(self, cfg):
        super(MultiViewFusionV2, self).__init__()
        self.img_size = cfg.img_size
        self.graph_dim = cfg.graph_dim
        self.num_views = cfg.num_views
        self.kernel_size = 7
        self.sub_image_size = self.img_size // self.kernel_size
        self.conv1 = nn.Conv2d(in_channels=cfg.graph_dim, out_channels=128, kernel_size=self.kernel_size,
                               stride=self.kernel_size, bias=False)
        self.transformer = Transformer(128, depth=1, heads=8, dim_head=128, mlp_dim=256, selfatt=True)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=cfg.graph_dim, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=cfg.graph_dim * 2, out_channels=cfg.graph_dim, kernel_size=1),
            nn.GELU()
        )

    def forward(self, feats: torch.Tensor):
        shortcut = feats
        feats = self.conv1(feats)
        B, C = feats.shape[:2]
        # feats = self.pooling(feats)
        feats = rearrange(feats.reshape(B // self.num_views, self.num_views, C, self.sub_image_size ** 2),
                          'B N C S -> B (N S) C')
        feats = self.transformer(feats)
        feats = rearrange(feats, 'B (N S) C -> (B N) C S', N=self.num_views).reshape(B, C, self.sub_image_size,
                                                                                     self.sub_image_size)
        feats = F.interpolate(feats, size=self.img_size)
        feats = self.conv2(feats)
        feats = self.conv3(torch.cat((feats, shortcut), dim=1))
        return feats


# class MultiViewFusion(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.img_size = cfg.img_size
#         self.graph_dim = cfg.graph_dim
#         self.num_views = cfg.num_views
#         self.interpolate_size = 16
#         self.pooling = nn.AvgPool2d(kernel_size=14, stride=14)
#         # self.conv1 = nn.Sequential(
#         #     nn.Conv2d(in_channels=cfg.graph_dim, out_channels=128, kernel_size=1),
#         #     nn.ReLU()
#         #     )
#         self.transformer = Transformer(64, depth=4, heads=8, dim_head=64, mlp_dim=256, selfatt=True)
#         # self.conv2 = nn.Conv2d(in_channels=128, out_channels=cfg.graph_dim, kernel_size=1)
#     def forward(self, feats: torch.Tensor):
#         # feats = self.conv1(feats)
#         B, C = feats.shape[:2]
#         # feats = F.interpolate(feats, size=self.interpolate_size)
#         feats = self.pooling(feats)
#         feats = rearrange(feats.reshape(B // self.num_views, self.num_views, C, self.interpolate_size**2), 'B N C S -> B (N S) C')
#         feats = self.transformer(feats)
#         feats = rearrange(feats, 'B (N S) C -> (B N) C S', N=self.num_views).reshape(B, C, self.interpolate_size, self.interpolate_size)
#         # feats = self.conv2(feats)
#         feats = F.interpolate(feats, size=self.img_size)
#         return feats
