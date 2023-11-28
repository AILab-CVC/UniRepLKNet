import torch
import torch.nn as nn
import torch.nn.functional as F
from models.layers.transformer import Transformer
from einops import rearrange

class SelectAdaptivePool2d(nn.Module):
    """Selectable global pooling layer with dynamic input kernel size
    """
    def __init__(self, output_size=1, pool_type='avg', flatten=False):
        super(SelectAdaptivePool2d, self).__init__()
        self.pool_type = pool_type or ''  # convert other falsy values to empty string for consistent TS typing
        self.flatten = nn.Flatten(1) if flatten else nn.Identity()
        if pool_type == '':
            self.pool = nn.Identity()  # pass through
            self.flatten = nn.Identity()
        elif pool_type == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(output_size)
        elif pool_type == 'max':
            self.pool = nn.AdaptiveMaxPool2d(output_size)
        else:
            assert False, 'Invalid pool type: %s' % pool_type

    def is_identity(self):
        return not self.pool_type

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        return x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + 'pool_type=' + self.pool_type \
               + ', flatten=' + str(self.flatten) + ')'


class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for channels of '2D' spatial NCHW tensors """
    def __init__(self, num_channels, eps=1e-6, affine=True):
        super().__init__(num_channels, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        # if self._fast_norm:
        #     x = fast_layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        # else:
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x

    
class MLPHead(nn.Module):
    def __init__(self, num_features, num_classes, mid_channels, dropout_ratio):
        super().__init__()
        self.mlp_head = nn.Sequential(
                nn.Linear(num_features, mid_channels[0]),
                nn.BatchNorm1d(mid_channels[0]),
                nn.ReLU(),
                nn.Dropout(dropout_ratio),
                nn.Linear(mid_channels[0], mid_channels[1]),
                nn.BatchNorm1d(mid_channels[1]),
                nn.ReLU(),
                nn.Dropout(dropout_ratio),
                nn.Linear(mid_channels[1], num_classes)
            )
        
    def forward(self, feats):
        return self.mlp_head(feats)
    

class PoolingClsHead(nn.Module):
    def __init__(self, cfg):  # , mid_channels, dropout_ratio
        super().__init__()
        self.num_views = cfg.num_views
        self.num_features = cfg.view_feature * self.num_views
        self.num_classes = cfg.classes
        self.adaptive_pooling = nn.AdaptiveAvgPool2d(output_size=1)
        self.norm = nn.LayerNorm(self.num_features)
        
        self.cls_head = nn.Linear(self.num_features, self.num_classes)
        # self.cls_head = nn.Sequential(
        #     nn.Linear(num_features, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, num_classes)
        # )
        
    def forward(self, feats: torch.Tensor):
        B, C, H, W = feats.shape
        feats = feats.reshape(B // self.num_views, self.num_views * C, H, W)
        feats = self.adaptive_pooling(feats).squeeze()
        feats = self.norm(feats)
        feats = self.cls_head(feats)
        return feats
    
    
class ViTClsHead(nn.Module):
    def __init__(self, cfg):  # , mid_channels, dropout_ratio
        super().__init__()
        self.num_views = cfg.num_views
        self.num_features = cfg.view_feature * self.num_views
        self.num_classes = cfg.classes
        self.norm = nn.LayerNorm(self.num_features)
        
        self.cls_head = nn.Linear(self.num_features, self.num_classes)
        
    def forward(self, feats: torch.Tensor):
        B = feats.shape[0]
        feats = feats.reshape(B // self.num_views, -1)
        feats = self.norm(feats)
        feats = self.cls_head(feats)
        return feats
