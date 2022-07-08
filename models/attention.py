import math
from typing import Union, List, Optional

import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
import einops


class LinearAttention(pl.LightningModule):
    def __init__(self, d: int, num_heads: int = 8,
                 d_k: Optional[int] = None, d_v: Optional[int] = None):
        super().__init__()

        self.d = d
        self.d_k = d_k if d_k is not None else self.d
        self.d_v = d_v if d_v is not None else self.d

        self.num_heads = num_heads
        self.head_dim = self.d // num_heads
        assert self.head_dim * num_heads == self.d, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = nn.Parameter(torch.empty((self.d, self.d)))
        self.k_proj_weight = nn.Parameter(torch.empty((self.d, self.d_k)))
        self.v_proj_weight = nn.Parameter(torch.empty((self.d, self.d_v)))

        self.in_proj_bias = nn.Parameter(torch.empty(3 * self.d))

        self.bias_k = nn.Parameter(torch.empty((1, 1, self.d)))
        self.bias_v = nn.Parameter(torch.empty((1, 1, self.d)))

    def forward(self, x):
        print(x.shape)

        out = x
        print(out.shape)
        return out


attn = LinearAttention(d=512)
x = torch.randn(64, 128, 512)
x_attn = attn(x)
print("ok")
