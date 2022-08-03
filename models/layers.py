import math
import warnings
from collections import OrderedDict
from typing import Union, List

import functorch
import torch
from einops.layers.torch import Rearrange
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
import torch.nn.functional as F
import einops
from torch._C._autograd import ProfilerActivity
from torch.autograd import profiler
from torch.profiler import profile
from torchaudio import transforms


class FouriEncoder(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_encoders: int = 6,
            dropout_p: float = 0.1,
            mixing_sublayer_type: str = "fourier",
            num_attention_heads: int = 8,
    ):
        super().__init__()

        # model architecture
        assert isinstance(num_encoders, int) and num_encoders >= 1, \
            f"there must be at least one encoder, not {num_encoders}"
        self.num_encoders = num_encoders
        assert isinstance(hidden_size, int) and hidden_size >= 1, \
            f"embeddings must be greater than 0, not {hidden_size}"
        self.hidden_size = hidden_size
        assert isinstance(num_attention_heads, int) and num_attention_heads >= 1
        self.num_attention_heads = num_attention_heads
        assert 0 <= dropout_p < 1, \
            f"dropout must be in [0, 1], not {dropout_p}"
        self.dropout_p = dropout_p
        assert isinstance(mixing_sublayer_type, str)
        self.mixing_sublayer_type = mixing_sublayer_type

        # architecture
        self.encoder_blocks = nn.Sequential(OrderedDict([*[(f"enc{i}",
                                                            FouriEncoderBlock(
                                                                in_features=self.hidden_size,
                                                                mid_features=self.hidden_size * 4,
                                                                out_features=self.hidden_size,
                                                                dropout_p=self.dropout_p,
                                                                mixing_sublayer_type=self.mixing_sublayer_type,
                                                                num_attention_heads=self.num_attention_heads,
                                                            ))
                                                           for i in range(self.num_encoders)],
                                                         # ("pool", nn.Linear(in_features=self.embeddings_dim,
                                                         #                    out_features=self.embeddings_dim)),
                                                         # ("act", nn.SELU()),
                                                         ]))

    def forward(self, x: torch.Tensor):
        # prepares the input
        assert x.shape[-1] == self.hidden_size
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

    def __init__(
            self,
            in_features: int,
            mid_features: int,
            out_features: int,
            dropout_p: Union[int, float] = 0,
            mixing_sublayer_type: str = "fourier",
            max_position_embeddings: int = 512,
            num_attention_heads: int = 8,
    ):
        super().__init__()
        assert isinstance(in_features, int) and in_features >= 1
        assert isinstance(mid_features, int) and mid_features >= 1
        assert isinstance(out_features, int) and out_features >= 1
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        assert isinstance(max_position_embeddings, int) and max_position_embeddings >= 1
        self.max_position_embeddings = max_position_embeddings
        assert isinstance(num_attention_heads, int) and num_attention_heads >= 1
        self.num_attention_heads = num_attention_heads
        assert 0 <= dropout_p < 1
        self.dropout_p = dropout_p
        mixing_sublayer_mapping = {
            "fourier": FastFourierTransform,
            "identity": IdentityTransform,
            "linear": LinearTransform,
            "attention": AttentionTransform,
        }
        assert isinstance(mixing_sublayer_type, str) and mixing_sublayer_type in mixing_sublayer_mapping.keys()
        self.mixing_sublayer_type = mixing_sublayer_type
        self.mixing_sublayer = mixing_sublayer_mapping[
            self.mixing_sublayer_type](hidden_size=self.in_features,
                                       max_position_embeddings=max_position_embeddings,
                                       dropout_p=dropout_p,
                                       num_attention_heads=self.num_attention_heads)

        self.layer_norm_1 = nn.LayerNorm([in_features, ])
        self.feed_forward_layer = FouriFeedForward(in_features=self.in_features,
                                                   mid_features=self.mid_features,
                                                   out_features=self.out_features,
                                                   dropout_p=self.dropout_p)
        if self.in_features != self.out_features:
            self.up_projection = nn.Sequential(
                nn.Linear(in_features=self.in_features, out_features=self.mid_features),
                nn.SELU(),
                nn.Linear(in_features=self.mid_features, out_features=self.out_features)
            )
        self.layer_norm_2 = nn.LayerNorm([out_features, ])

    def forward(self, x):
        # mixing sublayer
        x = self.layer_norm_1(x + self.mixing_sublayer(x))
        # feed-forward
        x = self.layer_norm_2((x if self.in_features == self.out_features else self.up_projection(x)) +
                              self.feed_forward_layer(x))
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
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        print(x.dtype)
        is_half = True if x.dtype == torch.float16 else False
        if is_half:
            x = x.type(torch.float32)
        x = functorch.vmap(torch.fft.fftn)(x).real
        if is_half:
            x = x.type(torch.float16)
        return x


class IdentityTransform(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class LinearTransform(nn.Module):
    def __init__(
            self,
            max_position_embeddings: int = 512,
            hidden_size: int = 768,
            **kwargd
    ):
        super().__init__()
        assert isinstance(max_position_embeddings, int) and max_position_embeddings >= 1
        self.max_position_embeddings = max_position_embeddings
        assert isinstance(hidden_size, int) and hidden_size >= 1
        self.hidden_size = hidden_size

        self.mat_hidden = nn.Parameter(torch.randn(
            (self.hidden_size, self.hidden_size),
        ))
        self.mat_seq = nn.Parameter(torch.randn(
            (self.max_position_embeddings, self.max_position_embeddings),
        ))

    def forward(self, x):
        x = torch.einsum(
            "bij,jk,ni->bnk",
            x,
            self.mat_hidden.to(x.device),
            self.mat_seq[:x.shape[1], :x.shape[1]].to(x.device)
        )
        return x


class AttentionTransform(nn.Module):
    def __init__(
            self,
            max_position_embeddings: int = 512,
            hidden_size: int = 768,
            num_attention_heads: int = 8,
            dropout_p: Union[int, float] = 0.1,
            **kwargs
    ):
        super().__init__()
        assert isinstance(max_position_embeddings, int) and max_position_embeddings >= 1
        self.max_position_embeddings = max_position_embeddings
        assert isinstance(hidden_size, int) and hidden_size >= 1
        self.hidden_size = hidden_size
        assert isinstance(num_attention_heads, int) and num_attention_heads >= 1
        self.num_attention_heads = num_attention_heads
        assert 0 <= dropout_p < 1
        self.dropout_p = dropout_p

        self.multihead_attn = nn.MultiheadAttention(
            self.hidden_size, self.num_attention_heads,
            dropout=self.dropout_p,
            batch_first=True)

    def forward(self, x):
        x, _ = self.multihead_attn(x, x, x)
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


class GetSinusoidalPositionalEmbeddings(nn.Module):
    def __init__(
            self,
            max_position_embeddings: int = 512
    ):
        super().__init__()
        assert isinstance(max_position_embeddings, int) and max_position_embeddings >= 1
        self.max_position_embeddings = max_position_embeddings

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 3  # (b s c)
        sequence_length, embeddings_dim = self.max_position_embeddings, x.shape[-1]
        pe = torch.zeros(sequence_length, embeddings_dim, device=x.device)
        position = torch.arange(0, sequence_length, device=x.device).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embeddings_dim, 2, dtype=torch.float, device=x.device) *
                              -(math.log(10000.0) / embeddings_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        del position, div_term
        pe = pe[:x.shape[1]].repeat(x.shape[0], 1, 1)[:, :x.shape[1]]
        assert pe.shape == x.shape
        return pe


class GetLearnedPositionalEmbeddings(nn.Module):
    def __init__(
            self,
            max_position_embeddings: int = 512,
            hidden_size: int = 768,
    ):
        super().__init__()
        assert isinstance(max_position_embeddings, int) and max_position_embeddings >= 1
        self.max_position_embeddings = max_position_embeddings
        assert isinstance(hidden_size, int) and hidden_size >= 1
        self.hidden_size = hidden_size

        self.embedder = nn.Embedding(self.max_position_embeddings, self.hidden_size)

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 3  # (b s c)
        pe = self.embedder(torch.arange(x.shape[1], device=x.device)).repeat(x.shape[0], 1, 1)
        assert pe.shape == x.shape
        return pe


class GetTokenTypeEmbeddings(nn.Module):
    def __init__(
            self,
            hidden_size: int = 768,
    ):
        super().__init__()
        assert isinstance(hidden_size, int) and hidden_size >= 1
        self.hidden_size = hidden_size

        self.embedder = nn.Embedding(2, self.hidden_size)

    def forward(self, x: torch.Tensor, special_tokens_indices: List[int]):
        assert len(x.shape) == 3  # (b s c)
        indices = torch.zeros(x.shape[1], dtype=torch.long, device=x.device)
        indices[special_tokens_indices] = 1
        pe = self.embedder(indices).repeat(x.shape[0], 1, 1)
        assert pe.shape == x.shape
        return pe


class AddGaussianNoise(nn.Module):
    def __init__(self, strength: float = 0.1):
        super().__init__()
        assert strength >= 0
        self.strength = strength

    def forward(self, x: torch.Tensor):
        noise = torch.normal(mean=torch.zeros_like(x, device=x.device, requires_grad=False) + x.mean(),
                             std=torch.zeros_like(x, device=x.device, requires_grad=False) + x.std())
        noise = noise * self.strength
        return x + noise


class MelSpectrogram(nn.Module):
    def __init__(self,
                 sampling_rate: Union[int, float, torch.Tensor],
                 window_size: Union[int, float] = 0.5,
                 window_stride: Union[int, float] = 0.05,
                 min_freq: int = 0,
                 max_freq: int = 50,
                 mels: int = 8,
                 ):
        super().__init__()
        # sampling rate
        assert any([isinstance(sampling_rate, t) for t in (int, float, torch.Tensor)])
        if isinstance(sampling_rate, torch.Tensor):
            if not torch.allclose(sampling_rate, sampling_rate[0]):
                raise NotImplementedError(f"all the elements in a batch must have the same sampling rate")
            sampling_rate = sampling_rate[0].detach().item()
        self.sampling_rate = sampling_rate
        # frequencies
        assert isinstance(min_freq, int) and isinstance(max_freq, int) and \
               0 <= min_freq <= max_freq
        self.min_freq: int = min_freq
        self.max_freq: int = max_freq
        assert isinstance(mels, int) \
               and mels > 0
        self.mels = mels

        # window
        assert window_size > 0
        self.window_size = math.floor(window_size * self.sampling_rate)
        assert window_stride > 0
        self.window_stride = math.floor(window_stride * self.sampling_rate)

    def forward(self, eegs: torch.Tensor):
        assert isinstance(eegs, torch.Tensor) and len(eegs.shape) in {2, 3}
        eegs = einops.rearrange(eegs, "s c -> c s" if len(eegs.shape) == 2 else "b s c -> b c s")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mel_fn = transforms.MelSpectrogram(
                sample_rate=self.sampling_rate,
                f_min=self.min_freq,
                f_max=self.max_freq,
                n_mels=self.mels,
                center=True,
                n_fft=128,
                normalized=True,
                power=1,
                win_length=self.window_size,
                hop_length=self.window_stride,
                pad=self.window_stride // 2,
            ).to(eegs.device)
            spectrogram = mel_fn(eegs.float())  # (b c m s)
        spectrogram = einops.rearrange(spectrogram, "b c m s -> b s c m")
        return spectrogram

    @staticmethod
    def plot_mel_spectrogram(spectrogram: torch.Tensor, scale: int = 2):
        assert len(spectrogram.shape) == 3  # s c m
        import matplotlib.pyplot as plt
        lines = int(math.ceil(math.sqrt(spectrogram.shape[1])))
        fig, axs = plt.subplots(nrows=lines, ncols=lines, figsize=(lines * scale * 1.5, lines * scale),
                                tight_layout=True)
        min_value, max_value = spectrogram.min(), spectrogram.max()

        for i_ax, ax in enumerate(axs.flat):
            if i_ax < spectrogram.shape[1]:
                im = ax.imshow(einops.rearrange(spectrogram[:, i_ax, :], "s m -> m s"),
                               vmin=min_value, vmax=max_value, aspect="auto", cmap=plt.get_cmap("hot"))
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax, orientation="vertical")
                ax.set_title(f"electrode {i_ax}")
                ax.set_xlabel("sample")
                ax.set_ylabel("mels")
                ax.invert_yaxis()
            else:
                ax.set_visible(False)
        plt.show(block=False)


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
