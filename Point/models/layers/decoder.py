import torch
from torch import nn, Tensor
from einops import rearrange
from timm.models.layers import trunc_normal_
from models.layers.transformer import Transformer


class TokenClsHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_classes = cfg.classes
        self.embed_dim = cfg.view_feature
        self.num_views = cfg.num_views
        self.num_patches = self.num_views * (cfg.num_patches ** 2)
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, self.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_classes, self.embed_dim))

        self.atten = Transformer(self.embed_dim, depth=2, heads=8, dim_head=128, mlp_dim=self.embed_dim * 2, selfatt=True)
        self.pos_drop = nn.Dropout(p=cfg.head_drop_rate)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, feats: Tensor):
        feats = rearrange(feats, '(B N) C H W -> B (N H W) C', N=self.num_views)
        B, N, C = feats.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        feats = torch.cat((cls_tokens, feats), dim=1)
        feats = feats + self.pos_embed
        feats = self.pos_drop(feats)

        feats = self.atten(feats)

        cls_logits = feats[:, 0:self.num_classes].mean(-1)
        return cls_logits


class TokenClsHeadV2(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_classes = cfg.classes
        self.embed_dim = cfg.view_feature
        self.num_views = cfg.num_views
        self.num_patches = self.num_views * (cfg.num_patches ** 2)

        self.projector = nn.Sequential(
            nn.Linear(self.embed_dim, 256),
            nn.GELU()
        )
        self.cls_token = nn.Parameter(torch.zeros(1, self.num_classes, 256))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + self.num_classes, 256))
        self.atten = Transformer(256, depth=2, heads=8, dim_head=128, mlp_dim=256 * 2, selfatt=True)
        self.pos_drop = nn.Dropout(p=cfg.head_drop_rate)

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, feats: Tensor):
        feats = rearrange(feats, '(B N) C H W -> B (N H W) C', N=self.num_views)
        B, N, C = feats.shape
        feats = self.projector(feats)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        feats = torch.cat((cls_tokens, feats), dim=1)
        feats = feats + self.pos_embed
        feats = self.pos_drop(feats)

        feats = self.atten(feats)

        cls_logits = feats[:, 0:self.num_classes].mean(-1)
        return cls_logits
    