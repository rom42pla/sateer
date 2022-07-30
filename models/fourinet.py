import math
from collections import OrderedDict
from typing import Union

import functorch
import torch
from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F
import einops
from torch._C._autograd import ProfilerActivity
from torch.autograd import profiler
from torch.profiler import profile


class FouriEncoder(nn.Module):
    def __init__(self,
                 embeddings_dim: int,
                 num_encoders: int = 6,
                 dropout_p: float = 0.1,

                 ):
        super().__init__()

        # model architecture
        assert isinstance(num_encoders, int) and num_encoders >= 1, \
            f"there must be at least one encoder, not {num_encoders}"
        self.num_encoders = num_encoders
        assert isinstance(embeddings_dim, int) and embeddings_dim >= 1, \
            f"embeddings must be greater than 0, not {embeddings_dim}"
        self.embeddings_dim = embeddings_dim
        assert 0 <= dropout_p < 1, \
            f"dropout must be in [0, 1], not {dropout_p}"
        self.dropout_p = dropout_p

        # architecture
        self.encoder_blocks = nn.Sequential(OrderedDict([*[(f"enc{i}",
                                                            FouriEncoderBlock(in_features=self.embeddings_dim,
                                                                              mid_features=self.embeddings_dim * 4,
                                                                              out_features=self.embeddings_dim,
                                                                              dropout_p=self.dropout_p))
                                                           for i in range(self.num_encoders)],
                                                         # ("pool", nn.Linear(in_features=self.embeddings_dim,
                                                         #                    out_features=self.embeddings_dim)),
                                                         # ("act", nn.SELU()),
                                                         ]))

    def forward(self, x: torch.Tensor):
        # prepares the input
        assert x.shape[-1] == self.embeddings_dim
        assert len(x.shape) in {2, 3}
        is_batched = True if len(x.shape) == 3 else False
        if not is_batched:
            x = einops.rearrange(x, "s c -> () s c")
        input_shape = x.shape

        # encoders pass
        x = self.encoder_blocks(x)

        if not is_batched:
            x = einops.rearrange(x, "b s c -> (b s) c")
        assert x.shape == input_shape, \
            f"output shape {x.shape} is different from input shape {input_shape}"
        return x


class FouriEncoderBlock(nn.Module):

    def __init__(self,
                 in_features: int,
                 mid_features: int,
                 out_features: int,
                 dropout_p: Union[int, float] = 0, ):
        super().__init__()
        assert isinstance(in_features, int) and in_features >= 1
        assert isinstance(mid_features, int) and mid_features >= 1
        assert isinstance(out_features, int) and out_features >= 1
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        assert 0 <= dropout_p < 1
        self.dropout_p = dropout_p

        self.fourier_layer = FastFourierTransform()
        self.layer_norm_1 = nn.LayerNorm([in_features, ])
        self.feed_forward_layer = FouriFeedForward(in_features=self.in_features,
                                                   mid_features=self.mid_features,
                                                   out_features=self.out_features,
                                                   dropout_p=dropout_p)
        if self.in_features != self.out_features:
            self.up_projection = nn.Sequential(
                nn.Linear(in_features=self.in_features, out_features=self.mid_features),
                nn.SELU(),
                nn.Linear(in_features=self.mid_features, out_features=self.out_features)
            )
        self.layer_norm_2 = nn.LayerNorm([out_features, ])

    def forward(self, x):
        # fourier pass
        x_fourier = self.fourier_layer(x)
        x = self.layer_norm_1(x + x_fourier)
        # fc pass
        x_forwarded = self.feed_forward_layer(x)
        if self.in_features != self.out_features:
            x = self.up_projection(x)
        x = x + x_forwarded
        x = self.layer_norm_2(x)
        return x


class FouriDecoder(nn.Module):
    def __init__(self,
                 embeddings_dim: int,
                 num_decoders: int = 6,
                 dropout_p: float = 0.1,

                 num_heads: int = 4,
                 attention_type: str = "quadratic",
                 ):
        super().__init__()

        # model architecture
        assert isinstance(num_decoders, int) and num_decoders >= 1, \
            f"there must be at least one decoder, not {num_decoders}"
        self.num_decoders: int = num_decoders
        assert isinstance(embeddings_dim, int) and embeddings_dim >= 1, \
            f"embeddings must be greater than 0, not {embeddings_dim}"
        self.embeddings_dim = embeddings_dim
        assert 0 <= dropout_p < 1, \
            f"dropout must be in [0, 1], not {dropout_p}"
        self.dropout_p = dropout_p

        assert isinstance(num_heads, int) and num_heads >= 1
        self.num_heads: int = num_heads

        self.decoder_blocks = nn.ModuleList(
            [FouriDecoderBlock(in_features=self.embeddings_dim,
                               mid_features=self.embeddings_dim,
                               out_features=self.embeddings_dim,
                               dropout_p=self.dropout_p,
                               num_heads=self.num_heads,
                               attention_type=attention_type)
             for _ in range(self.num_decoders)])
        self.postprocessing = nn.Sequential(OrderedDict([
            ("pool", nn.Linear(in_features=self.embeddings_dim,
                               out_features=self.embeddings_dim)),
            ("act", nn.SELU()),
        ]))

    def forward(self, tgt: torch.Tensor, src: torch.Tensor):
        # prepares the input
        assert tgt.shape[-1] == src.shape[-1] == self.embeddings_dim
        assert len(src.shape) == len(tgt.shape)
        assert len(tgt.shape) in {2, 3} and len(src.shape) in {2, 3}

        is_batched = True if len(src.shape) == 3 else False
        if not is_batched:
            src = einops.rearrange(src, "s c -> () s c")
            tgt = einops.rearrange(tgt, "s c -> () s c")
        input_shape = tgt.shape

        # decoders pass
        for decoder_block in self.decoder_blocks:
            tgt = decoder_block(tgt=tgt, src=src)
        tgt = self.postprocessing(tgt)

        if not is_batched:
            tgt = einops.rearrange(tgt, "b s c -> (b s) c")
        assert tgt.shape == input_shape, \
            f"output shape {tgt.shape} is different from input shape {input_shape}"
        return tgt


class FouriDecoderBlock(nn.Module):

    def __init__(
            self,
            in_features: int,
            mid_features: int,
            out_features: int,
            dropout_p: Union[int, float] = 0,

            num_heads: int = 4,
            attention_type: str = "linear",
    ):
        super().__init__()
        assert isinstance(in_features, int) and in_features >= 1
        assert isinstance(mid_features, int) and mid_features >= 1
        assert isinstance(out_features, int) and out_features >= 1
        self.in_features: int = in_features
        self.mid_features: int = mid_features
        self.out_features: int = out_features
        assert 0 <= dropout_p < 1
        self.dropout_p: float = dropout_p
        assert isinstance(num_heads, int) and num_heads >= 1
        self.num_heads: int = num_heads

        self.fourier_layer = FastFourierTransform()
        self.layer_norm_1 = nn.LayerNorm([in_features, ])

        if attention_type == "linear":
            self.attention_fn: LinearMultiheadAttention = LinearMultiheadAttention(embeddings_dim=self.in_features,
                                                                                   num_heads=self.num_heads)
        elif attention_type == "quadratic":
            self.attention_fn: nn.MultiheadAttention = nn.MultiheadAttention(embed_dim=self.in_features,
                                                                             num_heads=self.num_heads,
                                                                             batch_first=True)
        else:
            raise NotImplementedError
        self.attention_type = attention_type

        self.layer_norm_2 = nn.LayerNorm([in_features, ])
        self.feed_forward_layer = FouriFeedForward(in_features=self.in_features,
                                                   mid_features=self.mid_features,
                                                   out_features=self.out_features,
                                                   dropout_p=dropout_p)
        if self.in_features != self.out_features:
            self.up_projection = nn.Sequential(
                nn.Linear(in_features=self.in_features, out_features=self.mid_features),
                nn.SELU(),
                nn.Linear(in_features=self.mid_features, out_features=self.out_features)
            )
        self.layer_norm_3 = nn.LayerNorm([out_features, ])

    def forward(self,
                tgt: torch.Tensor,
                src: torch.Tensor,
                ):
        # fourier pass
        tgt = self.layer_norm_1(tgt + self.fourier_layer(tgt))
        # attention
        if self.attention_type == "linear":
            attentions = self.attention_fn(tgt, src, src)
        elif self.attention_type == "quadratic":
            mask = torch.triu(torch.full((tgt.shape[1], src.shape[1]), float('-inf'), device=tgt.device), diagonal=1)
            attentions, _ = self.attention_fn(tgt, src, src, attn_mask=mask)
        else:
            raise NotImplementedError
        tgt = self.layer_norm_2(tgt + attentions)
        # fc pass
        tgt_fwd = self.feed_forward_layer(tgt)
        if self.in_features != self.out_features:
            tgt = self.up_projection(tgt)
        tgt = self.layer_norm_3(tgt + tgt_fwd)
        return tgt


class LinearMultiheadAttention(nn.Module):
    def __init__(
            self,
            embeddings_dim: int,
            num_heads: int,
            dropout_p: float = 0.0,
    ):
        super().__init__()
        assert isinstance(embeddings_dim, int) and embeddings_dim >= 1
        self.embeddings_dim: int = embeddings_dim
        assert isinstance(num_heads, int) and num_heads >= 1
        self.num_heads: int = num_heads
        assert 0 <= dropout_p < 1
        self.dropout_p: float = float(dropout_p)

        self.q_weights = nn.Parameter(torch.randn(self.num_heads, self.embeddings_dim, self.embeddings_dim))
        self.k_weights = nn.Parameter(torch.randn(self.num_heads, self.embeddings_dim, self.embeddings_dim))
        self.v_weights = nn.Parameter(torch.randn(self.num_heads, self.embeddings_dim, self.embeddings_dim))

        self.out_reshaper = nn.Sequential(
            Rearrange("b h s d -> b s d h"),
            nn.AdaptiveMaxPool2d(output_size=(self.embeddings_dim, 1)),
            Rearrange("b s d h -> b s (d h)"),
            # Rearrange("b h s d -> b s (h d)"),
        )

    def forward(self, q, k, v):
        qs = F.softmax(torch.einsum("bsd,hio->bhsd", q, self.q_weights), dim=-2)  # (b h s d)
        ks = F.softmax(torch.einsum("bsd,hio->bhsd", k, self.k_weights), dim=-2)  # (b h s d)
        vs = torch.einsum("bsd,hio->bhsd", v, self.v_weights)  # (b h s d)
        # gets global contexts
        global_contexts = torch.einsum("bhnk,bhmv->bhkv", [ks, vs])  # (b h d d)
        # gets the output
        out = torch.einsum("bhkv,bhsq->bhsv", global_contexts, qs)  # (b h s d)
        # reshapes the output
        out = self.out_reshaper(out)
        return out


class FastFourierTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = functorch.vmap(torch.fft.fftn)(x).real
        return x


class FouriFeedForward(nn.Module):
    def __init__(self, in_features: int, mid_features: int, out_features: int,
                 dropout_p: Union[int, float] = 0.1):
        super().__init__()
        assert isinstance(in_features, int) and in_features >= 1
        assert isinstance(mid_features, int) and mid_features >= 1
        assert isinstance(out_features, int) and out_features >= 1
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        assert 0 <= dropout_p < 1
        self.dropout_p = dropout_p

        self.linear_1 = nn.Linear(self.in_features, self.mid_features)
        # self.linear_1 = nn.Sequential(
        #     Rearrange("b s c -> b c s"),
        #     nn.Conv1d(in_channels=self.in_features, out_channels=self.mid_features,
        #               kernel_size=7, stride=1, padding=3),
        #     Rearrange("b c s -> b s c"),
        # )
        self.activation = nn.SELU()
        self.linear_2 = nn.Linear(self.mid_features, self.out_features)
        # self.linear_2 = nn.Sequential(
        #     Rearrange("b s c -> b c s"),
        #     nn.Conv1d(in_channels=self.mid_features, out_channels=self.out_features,
        #               kernel_size=5, stride=1, padding=2),
        #     Rearrange("b c s -> b s c"),
        # )
        self.dropout = nn.AlphaDropout(p=self.dropout_p)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    embeddings_dim, batch_size, sampling_rate, seconds = 512, 32, 128, 1
    batch = {
        "eegs": torch.randn(batch_size, seconds * sampling_rate, embeddings_dim, dtype=torch.float32),
        "labels": torch.ones(batch_size, 6, dtype=torch.long),
        "sampling_rates": torch.zeros(batch_size, dtype=torch.long) + sampling_rate,
    }
    batch_target = {
        "eegs": torch.randn(batch_size, 4, embeddings_dim, dtype=torch.float32),
    }

    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        with profiler.record_function("full attention"):
            _ = FouriDecoder(embeddings_dim, 8, attention_type="quadratic")(batch_target["eegs"], batch["eegs"])

        # with profiler.record_function("linear attention"):
        #     _ = FouriDecoder(embeddings_dim, 8, attention_type="linear")(batch_target["eegs"], batch["eegs"])

    print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time", row_limit=8))
