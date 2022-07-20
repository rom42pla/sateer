import math
from collections import OrderedDict
from typing import Union

import functorch
import torch
from einops.layers.torch import Rearrange
from torch import nn
import einops


class FouriEncoder(nn.Module):
    def __init__(self,
                 embeddings_dim: int,
                 num_encoders: int = 6,
                 dropout_p: float = 0.1,

                 use_masking: bool = True,
                 mask_perc_min: float = 0.05,
                 mask_perc_max: float = 0.15,
                 mask_start_index: int = 0,

                 add_positional_embeddings: bool = True,
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
        assert isinstance(add_positional_embeddings, bool)
        self.add_positional_embeddings = add_positional_embeddings

        # masking
        assert isinstance(use_masking, bool)
        self.use_masking = use_masking
        if self.use_masking is True:
            assert isinstance(mask_perc_min, float) and 0 <= mask_perc_min < 1
            assert isinstance(mask_perc_max, float) and 0 <= mask_perc_max < 1 and mask_perc_max >= mask_perc_min
            self.mask_perc_max, self.mask_perc_min = mask_perc_max, mask_perc_min
            assert isinstance(mask_start_index, int) and mask_start_index >= 0
            self.mask_start_index = mask_start_index

        # special tokens dict
        self.special_tokens = {
            token: i_token
            for i_token, token in enumerate(["[start]", "[end]", "[mask]"])
        }

        # architecture
        self.tokens_embedder = nn.Embedding(len(self.special_tokens), self.embeddings_dim)
        self.encoder_blocks = nn.Sequential(OrderedDict([*[(f"enc_{i}",
                                                            FouriEncoderBlock(in_features=self.embeddings_dim,
                                                                              mid_features=self.embeddings_dim,
                                                                              out_features=self.embeddings_dim,
                                                                              dropout_p=self.dropout_p))
                                                           for i in range(self.num_encoders)],
                                                         ("pooler", nn.Linear(in_features=self.embeddings_dim,
                                                                              out_features=self.embeddings_dim)),
                                                         ("act", nn.Tanh()),
                                                         ]))

    def forward(self, x: torch.Tensor):
        # prepares the input
        assert x.shape[-1] == self.embeddings_dim
        assert len(x.shape) in {2, 3}
        is_batched = True if len(x.shape) == 3 else False
        if not is_batched:
            x = einops.rearrange(x, "s c -> () s c")
        input_shape = x.shape

        # generates special tokens
        start_token, end_token, mask_token = self.tokens_embedder(torch.as_tensor([
            self.special_tokens["[start]"],
            self.special_tokens["[end]"],
            self.special_tokens["[mask]"],
        ], device=x.device))
        # eventually adds masking
        if self.training and self.use_masking:
            mask_rand = torch.rand((x.shape[0], x[:, self.mask_start_index:].shape[1]),
                                   dtype=x.dtype, device=x.device)
            mask = (mask_rand >= self.mask_perc_min) * (mask_rand <= self.mask_perc_max)
            if x.shape[1] != mask.shape[1]:
                mask = torch.cat(
                    [torch.zeros(mask.shape[0], self.mask_start_index, dtype=torch.bool, device=mask.device),
                     mask], dim=1)
            x[mask] = mask_token

        # adds start and end token
        x = torch.cat([start_token.repeat(x.shape[0], 1, 1),
                       x,
                       end_token.repeat(x.shape[0], 1, 1)], dim=1)

        # eventually adds positional embeddings
        if self.add_positional_embeddings:
            x = self.add_positional_embeddings_fn(x)

        # encoders pass
        x = self.encoder_blocks(x)[:, 1:-1]

        if not is_batched:
            x = einops.rearrange(x, "b s c -> (b s) c")
        assert x.shape == input_shape, \
            f"output shape {x.shape} is different from input shape {input_shape}"
        return x

    @staticmethod
    def add_positional_embeddings_fn(x: torch.Tensor):
        sequence_length, embeddings_dim = x.shape[-2], x.shape[-1]
        pe = torch.zeros(sequence_length, embeddings_dim, device=x.device)
        position = torch.arange(0, sequence_length, device=x.device).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embeddings_dim, 2, dtype=torch.float, device=x.device) *
                              -(math.log(10000.0) / embeddings_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        x = x + pe
        return x


class FouriEncoderBlock(nn.Module):

    def __init__(self, in_features: int, mid_features: int, out_features: int,
                 dropout_p: Union[int, float] = 0):
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
        self.feed_forward_layer = FouriEEGFeedForward(in_features=self.in_features,
                                                      mid_features=self.mid_features,
                                                      out_features=self.out_features,
                                                      dropout_p=dropout_p)
        if self.in_features != self.out_features:
            self.up_projection = nn.Sequential(
                nn.Linear(in_features=self.in_features, out_features=self.mid_features),
                nn.GELU(),
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


class FastFourierTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = functorch.vmap(torch.fft.fftn)(x).real
        return x


class FouriEEGFeedForward(nn.Module):
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
        self.activation = nn.GELU()
        self.linear_2 = nn.Linear(self.mid_features, self.out_features)
        # self.linear_2 = nn.Sequential(
        #     Rearrange("b s c -> b c s"),
        #     nn.Conv1d(in_channels=self.mid_features, out_channels=self.out_features,
        #               kernel_size=5, stride=1, padding=2),
        #     Rearrange("b c s -> b s c"),
        # )
        self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    embeddings_dim, batch_size, sampling_rate, seconds = 512, 2048, 128, 1
    batch = {
        "eegs": torch.randn(batch_size, seconds * sampling_rate, embeddings_dim, dtype=torch.float32),
        "labels": torch.ones(batch_size, 6, dtype=torch.long),
        "sampling_rates": torch.zeros(batch_size, dtype=torch.long) + sampling_rate,
    }
    model = FouriEncoder(embeddings_dim=embeddings_dim,
                         num_encoders=2, use_masking=True,
                         mask_perc_min=0.1, mask_perc_max=0.3)
    print("input shape", batch["eegs"].shape)
    out = model(batch["eegs"])
    print("output shape", out.shape)
    print(model)
