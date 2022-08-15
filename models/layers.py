import math
import warnings
from typing import Union, List, Dict

import functorch
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
import einops
from torch._C._autograd import ProfilerActivity
from torch.autograd import profiler
from torch.profiler import profile
from torchaudio import transforms


#
#
# class LinearEncoder(nn.Module):
#     def __init__(
#             self,
#             hidden_size: int,
#             num_encoders: int = 4,
#             num_attention_heads: int = 8,
#             dropout_p: float = 0.2,
#     ):
#         super().__init__()
#
#         # model architecture
#         assert isinstance(num_encoders, int) and num_encoders >= 1, \
#             f"there must be at least one encoder, not {num_encoders}"
#         self.num_encoders = num_encoders
#         assert isinstance(hidden_size, int) and hidden_size >= 1, \
#             f"embeddings must be greater than 0, not {hidden_size}"
#         self.hidden_size = hidden_size
#         assert isinstance(num_attention_heads, int) and num_attention_heads >= 1
#         self.num_attention_heads = num_attention_heads
#         assert 0 <= dropout_p < 1, \
#             f"dropout must be in [0, 1], not {dropout_p}"
#         self.dropout_p = dropout_p
#         assert isinstance(num_attention_heads, int) and num_attention_heads >= 1
#         self.num_attention_heads: int = num_attention_heads
#
#         # architecture
#         self.encoder_blocks = nn.ModuleList([
#             LinearEncoderBlock(
#                 in_features=self.hidden_size,
#                 out_features=self.hidden_size,
#                 dropout_p=self.dropout_p,
#                 num_attention_heads=self.num_attention_heads,
#             )
#             for _ in range(self.num_encoders)
#         ])
#
#     def forward(self, x: torch.Tensor):
#         # prepares the input
#         assert x.shape[-1] == self.hidden_size
#         input_shape = x.shape
#
#         # encoders pass
#         for encoder_block in self.encoder_blocks:
#             x = encoder_block(x)
#
#         assert x.shape == input_shape, \
#             f"output shape {x.shape} is different from input shape {input_shape}"
#         return x
#
#
# class LinearEncoderBlock(nn.Module):
#
#     def __init__(
#             self,
#             in_features: int,
#             out_features: int,
#             num_attention_heads: int = 4,
#             dropout_p: Union[int, float] = 0.2,
#     ):
#         super().__init__()
#         assert isinstance(in_features, int) and in_features >= 1
#         assert isinstance(out_features, int) and out_features >= 1
#         self.in_features: int = in_features
#         self.out_features: int = out_features
#         assert 0 <= dropout_p < 1
#         self.dropout_p: float = dropout_p
#         assert isinstance(num_attention_heads, int) and num_attention_heads >= 1
#         self.num_attention_heads: int = num_attention_heads
#
#         # section 1
#         self.attention_fn_1 = LinearMultiheadAttention(
#             hidden_size=self.in_features,
#             num_attention_heads=self.num_attention_heads,
#             dropout_p=self.dropout_p
#         )
#         self.layer_norm_1 = nn.LayerNorm([in_features, ])
#
#         # section 2
#         self.feed_forward_layer = FeedForward(
#             in_features=self.in_features,
#             out_features=self.out_features,
#             dropout_p=self.dropout_p
#         )
#         if self.in_features != self.out_features:
#             self.up_projection = nn.Sequential(
#                 nn.Linear(in_features=self.in_features, out_features=self.in_features * 4),
#                 nn.SELU(),
#                 nn.Linear(in_features=self.in_features * 4, out_features=self.out_features)
#             )
#         self.layer_norm_2 = nn.LayerNorm([out_features, ])
#
#     def forward(self, x):
#         # section 1
#         x = self.layer_norm_1(x + self.attention_fn_1(x, x, x))
#
#         # section 2
#         x = self.layer_norm_2((x if self.in_features == self.out_features else self.up_projection(x)) +
#                               self.feed_forward_layer(x))
#         return x
#
#
# class LinearDecoder(nn.Module):
#     def __init__(
#             self,
#             hidden_size: int,
#             num_decoders: int = 6,
#             dropout_p: float = 0.1,
#
#             num_attention_heads: int = 4,
#     ):
#         super().__init__()
#
#         # model architecture
#         assert isinstance(num_decoders, int) and num_decoders >= 1, \
#             f"there must be at least one decoder, not {num_decoders}"
#         self.num_decoders: int = num_decoders
#         assert isinstance(hidden_size, int) and hidden_size >= 1, \
#             f"embeddings must be greater than 0, not {hidden_size}"
#         self.hidden_size = hidden_size
#         assert 0 <= dropout_p < 1, \
#             f"dropout must be in [0, 1], not {dropout_p}"
#         self.dropout_p = dropout_p
#
#         assert isinstance(num_attention_heads, int) and num_attention_heads >= 1
#         self.num_attention_heads: int = num_attention_heads
#
#         self.decoder_blocks = nn.ModuleList([
#             LinearDecoderBlock(
#                 in_features=self.hidden_size,
#                 out_features=self.hidden_size,
#                 dropout_p=self.dropout_p,
#                 num_attention_heads=self.num_attention_heads,
#             )
#             for _ in range(self.num_decoders)
#         ])
#         self.postprocessing = nn.Sequential(OrderedDict([
#             ("pool", nn.Linear(in_features=self.hidden_size,
#                                out_features=self.hidden_size)),
#             ("act", nn.SELU()),
#         ]))
#
#     def forward(self, tgt: torch.Tensor, src: torch.Tensor):
#         # prepares the input
#         assert tgt.shape[-1] == src.shape[-1] == self.hidden_size
#         assert len(src.shape) == len(tgt.shape)
#
#         input_shape = tgt.shape
#
#         # decoders pass
#         for decoder_block in self.decoder_blocks:
#             tgt = decoder_block(tgt=tgt, src=src)
#         tgt = self.postprocessing(tgt)
#
#         assert tgt.shape == input_shape, \
#             f"output shape {tgt.shape} is different from input shape {input_shape}"
#         return tgt
#
#
# class LinearDecoderBlock(nn.Module):
#
#     def __init__(
#             self,
#             in_features: int,
#             out_features: int,
#             num_attention_heads: int = 4,
#             dropout_p: Union[int, float] = 0.2,
#     ):
#         super().__init__()
#         assert isinstance(in_features, int) and in_features >= 1
#         assert isinstance(out_features, int) and out_features >= 1
#         self.in_features: int = in_features
#         self.out_features: int = out_features
#         assert 0 <= dropout_p < 1
#         self.dropout_p: float = dropout_p
#         assert isinstance(num_attention_heads, int) and num_attention_heads >= 1
#         self.num_attention_heads: int = num_attention_heads
#
#         # section 1
#         self.attention_fn_1: LinearMultiheadAttention = LinearMultiheadAttention(
#             hidden_size=self.in_features,
#             num_attention_heads=self.num_attention_heads,
#             dropout_p=self.dropout_p,
#         )
#         self.layer_norm_1 = nn.LayerNorm([in_features, ])
#
#         # section 2
#         self.attention_fn_2: LinearMultiheadAttention = LinearMultiheadAttention(
#             hidden_size=self.in_features,
#             num_attention_heads=self.num_attention_heads,
#             dropout_p=self.dropout_p,
#         )
#         self.layer_norm_2 = nn.LayerNorm([in_features, ])
#
#         # section 3
#         self.feed_forward_layer = FeedForward(
#             in_features=self.in_features,
#             out_features=self.out_features,
#             dropout_p=self.dropout_p
#         )
#         if self.in_features != self.out_features:
#             self.up_projection = nn.Sequential(
#                 nn.Linear(in_features=self.in_features, out_features=self.in_features * 4),
#                 nn.SELU(),
#                 nn.Linear(in_features=self.in_features * 4, out_features=self.out_features)
#             )
#         self.layer_norm_3 = nn.LayerNorm([out_features, ])
#
#     def forward(
#             self,
#             tgt: torch.Tensor,
#             src: torch.Tensor,
#     ):
#         # section 1
#         tgt = self.layer_norm_1(tgt + self.attention_fn_1(tgt, tgt, tgt))
#
#         # section 2
#         tgt = self.layer_norm_2(tgt + self.attention_fn_2(tgt, src, src))
#
#         # section 3
#         tgt_fwd = self.feed_forward_layer(tgt)
#         if self.in_features != self.out_features:
#             tgt = self.up_projection(tgt)
#         tgt = self.layer_norm_3(tgt + tgt_fwd)
#         return tgt
#
#
# class LinearMultiheadAttention(nn.Module):
#     def __init__(
#             self,
#             hidden_size: int,
#             num_attention_heads: int,
#             dropout_p: float = 0.2,
#     ):
#         super().__init__()
#         assert isinstance(hidden_size, int) and hidden_size >= 1
#         self.hidden_size: int = hidden_size
#         assert isinstance(num_attention_heads, int) and num_attention_heads >= 1
#         self.num_attention_heads: int = num_attention_heads
#         assert 0 <= dropout_p < 1
#         self.dropout_p: float = float(dropout_p)
#
#         self.q_weights = nn.Parameter(torch.randn(self.num_attention_heads, self.hidden_size, self.hidden_size))
#         self.k_weights = nn.Parameter(torch.randn(self.num_attention_heads, self.hidden_size, self.hidden_size))
#         self.v_weights = nn.Parameter(torch.randn(self.num_attention_heads, self.hidden_size, self.hidden_size))
#
#         self.out_reshaper = nn.Sequential(
#             Rearrange("b h s d -> b s d h"),
#             nn.AdaptiveMaxPool2d(output_size=(self.hidden_size, 1)),
#             Rearrange("b s d h -> b s (d h)"),
#             # Rearrange("b h s d -> b s (h d)"),
#         )
#
#     def forward(self, q, k, v):
#         st = time.time()
#         for i_head in range(self.num_attention_heads):
#             print(k.shape, self.k_weights.shape)
#             ks = torch.matmul(k, self.k_weights[i_head])
#         print("matmul", time.time() - st, "seconds")
#
#         st = time.time()
#         ks = F.softmax(torch.einsum("bsd,hio->bhsd", k, self.k_weights), dim=-2)  # (b h s d)
#         print("einsum", time.time() - st, "seconds")
#         # ks = F.softmax(torch.einsum("bsd,hio->bhsd", k, self.k_weights), dim=-2)  # (b h s d)
#         # vs = torch.einsum("bsd,hio->bhsd", v, self.v_weights)  # (b h s d)
#         # gets global contexts
#         # global_contexts = torch.einsum("bhnk,bhmv->bhkv", [ks, vs])  # (b h d d)
#         # print(ks.shape, vs.shape, global_contexts.shape)
#         print(ks.shape)
#
#         exit()
#         del ks, vs
#         # gets the output
#         qs = F.softmax(torch.einsum("bsd,hio->bhsd", q, self.q_weights), dim=-2)  # (b h s d)
#         out = torch.einsum("bhkv,bhsq->bhsv", global_contexts, qs)  # (b h s d)
#         del qs
#         # reshapes the output
#         out = self.out_reshaper(out)
#         return out


class FastFourierTransform(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        initial_type = x.dtype
        is_half = True if initial_type in [torch.float16, torch.bfloat16] else False
        if is_half:
            x = x.type(torch.float32)
        x = functorch.vmap(torch.fft.fftn)(x).real
        if is_half:
            x = x.type(initial_type)
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
            **kwargs
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
            self.mat_hidden.type_as(x),
            self.mat_seq[:x.shape[1], :x.shape[1]].type_as(x)
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


class FeedForward(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout_p: Union[int, float] = 0.2
    ):
        super().__init__()
        assert isinstance(in_features, int) and in_features >= 1
        assert isinstance(out_features, int) and out_features >= 1
        self.in_features = in_features
        self.out_features = out_features
        assert 0 <= dropout_p < 1
        self.dropout_p = dropout_p

        self.linear_1 = nn.Linear(
            in_features=self.in_features,
            out_features=self.in_features * 4
        )
        self.activation = nn.SELU()
        self.linear_2 = nn.Linear(
            in_features=self.in_features * 4,
            out_features=self.out_features
        )
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
            max_position_embeddings: int = 1024
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


class GetUserEmbeddings(nn.Module):
    def __init__(self,
                 hidden_size: int):
        super().__init__()
        assert isinstance(hidden_size, int) and hidden_size >= 1, \
            f"embeddings must be greater than 0, not {hidden_size}"
        self.hidden_size = hidden_size
        self.embeddings: Dict[str, torch.Tensor] = {}

    def forward(self, ids: Union[List[Union[int, str]], torch.Tensor]) -> torch.Tensor:
        if isinstance(ids, torch.Tensor):
            ids = ids.clone().detach().tolist()
        for id in ids:
            if id not in self.embeddings.keys():
                self.add_id(id)
        embeddings = torch.stack([self.embeddings[id] for id in ids])
        return embeddings

    def add_id(self, id: Union[int, str]):
        assert id not in self.embeddings.keys()
        self.embeddings[id] = nn.Parameter(torch.randn(self.hidden_size, requires_grad=True))


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
    def __init__(
            self,
            sampling_rate: int,
            window_size: Union[int, float],
            window_stride: Union[int, float],
            mels: int = 8,
            min_freq: int = 0,
            max_freq: int = 50,
    ):
        super().__init__()
        # assertions
        assert isinstance(sampling_rate, int) and sampling_rate >= 1
        self.sampling_rate: int = sampling_rate
        assert isinstance(min_freq, int) and isinstance(max_freq, int) and \
               0 <= min_freq <= max_freq
        self.min_freq: int = min_freq
        self.max_freq: int = max_freq
        assert isinstance(mels, int) \
               and mels > 0
        self.mels: int = mels
        assert window_size > 0
        self.window_size: int = math.floor(window_size * self.sampling_rate)
        assert window_stride > 0
        self.window_stride: int = math.floor(window_stride * self.sampling_rate)

    def forward(
            self,
            eegs: torch.Tensor,
    ):
        assert len(eegs.shape) == 3
        window_size = min(self.window_size, eegs.shape[1])
        window_stride = min(self.window_stride, window_size // 2)
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        mel_fn = transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            f_min=self.min_freq,
            f_max=self.max_freq,
            n_mels=self.mels,
            center=True,
            # n_fft=192,
            n_fft=max(128, window_size),
            normalized=True,
            power=1,
            win_length=window_size,
            hop_length=window_stride,
            pad=window_stride//2,
        ).to(eegs.device).float()
        eegs = einops.rearrange(eegs, "b s c -> b c s")
        spectrogram = mel_fn(eegs)  # (b c m s)
        # spectrogram = transforms.AmplitudeToDB(stype="power")(spectrogram)
        spectrogram = einops.rearrange(spectrogram, "b c m s -> b s c m")
        return spectrogram

    @staticmethod
    def plot_mel_spectrogram(input_spectrogram: torch.Tensor, scale: int = 2):
        assert len(input_spectrogram.shape) == 3  # s c m
        import matplotlib.pyplot as plt
        spectrogram = input_spectrogram.clone().detach().cpu()
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
    embeddings_dim, batch_size, sampling_rate, seconds = 512, 1024, 128, 10
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
            _ = LinearDecoder(embeddings_dim, 4, num_attention_heads=8)(batch_target["eegs"], batch["eegs"])

        # with profiler.record_function("linear attention"):
        #     _ = FouriDecoder(embeddings_dim, 8, attention_type="linear")(batch_target["eegs"], batch["eegs"])

    print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time", row_limit=8))
