import numpy as np

import torch
from torch import nn

from sonnet.baselines.layers.RevIN import RevIN
from sonnet.lightning.lightning_module import BaseModel


def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
):
    """
    A copy-paste from https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / np.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).tril(
            diagonal=0
        )
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


class Model(BaseModel):
    def __init__(self, configs, **kwargs):
        super(Model, self).__init__(configs, **kwargs)
        num_channels = configs.enc_in
        seq_len = configs.seq_len
        hid_dim = configs.d_model
        pred_horizon = configs.pred_len
        use_revin = configs.revin
        self.revin = RevIN(num_features=num_channels)
        self.compute_keys = nn.Linear(seq_len, hid_dim)
        self.compute_queries = nn.Linear(seq_len, hid_dim)
        self.compute_values = nn.Linear(seq_len, seq_len)
        self.linear_forecaster = nn.Linear(seq_len, pred_horizon)
        self.use_revin = use_revin

    def forward(self, x, flatten_output=False):
        # RevIN Normalization
        if self.use_revin:
            x_norm = self.revin(x, mode="norm").transpose(1, 2)  # (n, D, L)
        else:
            x_norm = x.transpose(1, 2)
        # Channel-Wise Attention
        queries = self.compute_queries(x_norm)  # (n, D, hid_dim)
        keys = self.compute_keys(x_norm)  # (n, D, hid_dim)
        values = self.compute_values(x_norm)  # (n, D, L)
        if hasattr(nn.functional, "scaled_dot_product_attention"):
            att_score = nn.functional.scaled_dot_product_attention(
                queries, keys, values
            )  # (n, D, L)
        else:
            att_score = scaled_dot_product_attention(queries, keys, values)  # (n, D, L)
        out = x_norm + att_score  # (n, D, L)
        # Linear Forecasting
        out = self.linear_forecaster(out)  # (n, D, H)
        # RevIN Denormalization
        if self.use_revin:
            out = self.revin(out.transpose(1, 2), mode="denorm").transpose(
                1, 2
            )  # (n, D, H)

        out = out.transpose(1, 2)
        if flatten_output:
            return out.reshape([out.shape[0], out.shape[1] * out.shape[2]])
        else:
            return out
