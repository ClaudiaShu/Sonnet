import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from sonnet.baselines.layers.Crossformer_EncDec import (
    scale_block,
    Encoder,
    Decoder,
    DecoderLayer,
)
from sonnet.baselines.layers.Embed import PatchEmbedding
from sonnet.baselines.layers.SelfAttention_Family import (
    AttentionLayer,
    FullAttention,
    TwoStageAttentionLayer,
)
from sonnet.lightning.lightning_module import BaseModel

from math import ceil


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class Model(BaseModel):
    """
    Paper link: https://openreview.net/pdf?id=vSVLM2j9eie
    """

    def __init__(self, configs, **kwargs) -> None:
        super(Model, self).__init__(configs, **kwargs)
        self.enc_in = configs.enc_in
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.seg_len = 12
        self.win_size = 2

        # The padding operation to handle invisible sgemnet length
        self.pad_in_len = ceil(1.0 * configs.seq_len / self.seg_len) * self.seg_len
        self.pad_out_len = ceil(1.0 * configs.pred_len / self.seg_len) * self.seg_len
        self.in_seg_num = self.pad_in_len // self.seg_len
        self.out_seg_num = ceil(
            self.in_seg_num / (self.win_size ** (configs.e_layers - 1))
        )
        self.head_nf = configs.d_model * self.out_seg_num

        # Embedding
        self.enc_value_embedding = PatchEmbedding(
            configs.d_model,
            self.seg_len,
            self.seg_len,
            self.pad_in_len - configs.seq_len,
            0,
        )
        self.enc_pos_embedding = nn.Parameter(
            torch.randn(1, configs.enc_in, self.in_seg_num, configs.d_model)
        )
        self.pre_norm = nn.LayerNorm(configs.d_model)

        # Encoder
        self.encoder = Encoder(
            [
                scale_block(
                    configs,
                    1 if l == 0 else self.win_size,
                    configs.d_model,
                    configs.n_heads,
                    configs.d_ff,
                    1,
                    configs.dropout,
                    self.in_seg_num
                    if l == 0
                    else ceil(self.in_seg_num / self.win_size**l),
                    configs.factor,
                )
                for l in range(configs.e_layers)
            ]
        )
        # Decoder
        self.dec_pos_embedding = nn.Parameter(
            torch.randn(
                1, configs.enc_in, (self.pad_out_len // self.seg_len), configs.d_model
            )
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    TwoStageAttentionLayer(
                        configs,
                        (self.pad_out_len // self.seg_len),
                        configs.factor,
                        configs.d_model,
                        configs.n_heads,
                        configs.d_ff,
                        configs.dropout,
                    ),
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=False,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    self.seg_len,
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    # activation=configs.activation,
                )
                for l in range(configs.e_layers + 1)
            ],
        )

    def forecast(self, x_enc):
        # embedding
        x_enc, n_vars = self.enc_value_embedding(x_enc.permute(0, 2, 1))
        x_enc = rearrange(
            x_enc, "(b d) seg_num d_model -> b d seg_num d_model", d=n_vars
        )
        x_enc += self.enc_pos_embedding
        x_enc = self.pre_norm(x_enc)
        enc_out, attns = self.encoder(x_enc)

        dec_in = repeat(
            self.dec_pos_embedding,
            "b ts_d l d -> (repeat b) ts_d l d",
            repeat=x_enc.shape[0],
        )
        dec_out = self.decoder(dec_in, enc_out)
        return dec_out

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
