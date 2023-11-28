import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding
from layers.Causal_Conv import CausalConv
from layers.Multi_Correlation import AutoCorrelation, AutoCorrelationLayer, CrossCorrelation, CrossCorrelationLayer, \
    MultiCorrelation
from layers.Corrformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, \
    my_Layernorm, series_decomp
from unireplknet import UniRepLKNet

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.node_num = configs.node_num
        self.node_list = configs.node_list  # node_num = node_list[0]*node_list[1]*node_list[2]...
        self.output_attention = configs.output_attention

        # Decomp
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Encoding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.root_path,
                                           configs.node_num, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.root_path,
                                           configs.node_num, configs.embed, configs.freq,
                                           configs.dropout)
        
        max13_kernel_sizes = [[3, 3, 3], [13, 13, 13], [13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3], [13, 13, 13]]

        default_kernel_sizes = [[3, 3, 3], [13, 13, 13], [13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3, 13, 3, 3], [11, 11, 11]]

        max11_kernel_sizes = [[3, 3, 3], [11, 11, 11], [11, 3, 3, 11, 3, 3, 11, 3, 3, 11, 3, 3, 11, 3, 3, 11, 3, 3, 11, 3, 3, 11, 3, 3, 11, 3, 3], [11, 11, 11]]

        max9_kernel_sizes = [[3, 3, 3], [9, 9, 9], [9, 3, 3, 9, 3, 3, 9, 3, 3, 9, 3, 3, 9, 3, 3, 9, 3, 3, 9, 3, 3, 9, 3, 3, 9, 3, 3], [9, 9, 9]]

        max7_kernel_sizes = [[3, 3, 3], [7, 7, 7], [7, 3, 3, 7, 3, 3, 7, 3, 3, 7, 3, 3, 7, 3, 3, 7, 3, 3, 7, 3, 3, 7, 3, 3, 7, 3, 3], [7, 7, 7]]

        all7_kernel_sizes = [[7] * 3, [7] * 3, [7] * 27, [7] * 3]
        self.encoder = UniRepLKNet(num_classes=1000, depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], drop_path_rate=0.4,
                    kernel_sizes=default_kernel_sizes,in_chans=1,
                custom_set='nolk', disable_iGEMM=True,
                with_cp=True)
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    MultiCorrelation(
                        AutoCorrelationLayer(
                            AutoCorrelation(True, configs.factor_temporal, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        CrossCorrelationLayer(
                            CrossCorrelation(
                                CausalConv(
                                    num_inputs=configs.d_model // configs.n_heads * (self.label_len + self.pred_len),
                                    num_channels=[configs.d_model // configs.n_heads * (self.label_len + self.pred_len)] \
                                                 * configs.dec_tcn_layers,
                                    kernel_size=3),
                                False, configs.factor_spatial, attention_dropout=configs.dropout,
                                output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        configs.node_num,
                        configs.node_list,
                        dropout=configs.dropout,
                    ),
                    MultiCorrelation(
                        AutoCorrelationLayer(
                            AutoCorrelation(False, configs.factor_temporal, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        CrossCorrelationLayer(
                            CrossCorrelation(
                                CausalConv(
                                    num_inputs=configs.d_model // configs.n_heads * (self.label_len + self.pred_len),
                                    num_channels=[configs.d_model // configs.n_heads * (self.label_len + self.pred_len)] \
                                                 * configs.dec_tcn_layers,
                                    kernel_size=3),
                                False, configs.factor_spatial, attention_dropout=configs.dropout,
                                output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        configs.node_num,
                        configs.node_list,
                        dropout=configs.dropout,
                    ),
                    configs.d_model,
                    configs.c_out,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )
        self.affine_weight = nn.Parameter(torch.ones(1, 1, configs.enc_in))
        self.affine_bias = nn.Parameter(torch.zeros(1, 1, configs.enc_in))
        self.d_model = configs.d_model
        trainables = [p for p in self.parameters() if p.requires_grad]
        print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in self.parameters()) / 1e6))
        print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
        
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # init & normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        x_enc = x_enc * self.affine_weight.repeat(1, 1, self.node_num) + self.affine_bias.repeat(1, 1, self.node_num)
        # decomp
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]]).cuda()
        seasonal_init, trend_init = self.decomp(x_enc)
        # decoder input init
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc
        B, L, D = x_enc.shape
        _, _, C = x_mark_enc.shape
        x_enc = x_enc.view(B, L, self.node_num, -1).permute(0, 2, 1, 3).contiguous() \
            .view(B * self.node_num, L, D // self.node_num)
        x_mark_enc = x_mark_enc.unsqueeze(1).repeat(1, self.node_num, 1, 1).view(B * self.node_num, L, C)
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = enc_out.reshape(self.node_num,1,192,192)
        enc_out = self.encoder(enc_out)
        enc_out = enc_out.reshape(self.node_num,-1,self.d_model)
        # dec
        B, L, D = seasonal_init.shape
        _, _, C = x_mark_dec.shape
        seasonal_init = seasonal_init.view(B, L, self.node_num, -1).permute(0, 2, 1, 3).contiguous() \
            .view(B * self.node_num, L, D // self.node_num)
        trend_init = trend_init.view(B, L, self.node_num, -1).permute(0, 2, 1, 3).contiguous() \
            .view(B * self.node_num, L, D // self.node_num)
        x_mark_dec = x_mark_dec.unsqueeze(1).repeat(1, self.node_num, 1, 1).view(B * self.node_num, L, C)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final
        dec_out = trend_part + seasonal_part
        dec_out = dec_out[:, -self.pred_len:, :] \
            .view(B, self.node_num, self.pred_len, D // self.node_num).permute(0, 2, 1, 3).contiguous() \
            .view(B, self.pred_len, D)  # B L D

        # scale back
        dec_out = dec_out - self.affine_bias.repeat(1, 1, self.node_num)
        dec_out = dec_out / (self.affine_weight.repeat(1, 1, self.node_num) + 1e-10)
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out  # [B, L, D]
