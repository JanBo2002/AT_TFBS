"""Shared model and model components for every configuration."""
from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_VERSION = '5.1.0-input-channel-dropout'

if torch.version.hip is not None:
    print("ROCm detected: forcing mathematical SDPA backend")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


# Model components
# ----------------

ActivationName = Literal["relu", "gelu"]


def make_activation(name: ActivationName) -> nn.Module:
    """Create a fresh activation module for one network location."""
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class RMSBatchNorm1d(nn.Module):
    """Channel-wise RMS normalization without mean subtraction.

    This follows the high-level behavior described for AlphaGenome: trainable
    scale/offset and an exponential moving average of per-channel variance used
    during inference. Input shape is [B, C, L].
    """

    def __init__(self, channels: int, eps: float = 1e-5, ema_decay: float = 0.9) -> None:
        super().__init__()
        self.eps = eps
        self.ema_decay = ema_decay
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean_square", torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"RMSBatchNorm1d expects [B,C,L], got {tuple(x.shape)}")
        if self.training:
            mean_square = x.float().pow(2).mean(dim=(0, 2))
            with torch.no_grad():
                self.running_mean_square.mul_(self.ema_decay).add_(
                    mean_square.detach(), alpha=1.0 - self.ema_decay
                )
        else:
            mean_square = self.running_mean_square
        normalized = x / torch.sqrt(mean_square.to(dtype=x.dtype)[None, :, None] + self.eps)
        return normalized * self.weight[None, :, None] + self.bias[None, :, None]


class RMSNormChannels(nn.Module):
    """Per-position RMS normalization across channels for [B,C,L] tensors."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(dim=1, keepdim=True) + self.eps)
        normalized = x / rms.to(dtype=x.dtype)
        return normalized * self.weight[None, :, None] + self.bias[None, :, None]


class WSConv1d(nn.Conv1d):
    """1D convolution with per-output-channel weight standardization."""

    def __init__(self, *args: Any, ws_eps: float = 1e-5, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.ws_eps = ws_eps
        self.gain = nn.Parameter(torch.ones(self.out_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        reduce_dims = tuple(range(1, weight.ndim))
        mean = weight.mean(dim=reduce_dims, keepdim=True)
        variance = weight.var(dim=reduce_dims, unbiased=False, keepdim=True)
        standardized = (weight - mean) / torch.sqrt(variance + self.ws_eps)
        fan_in = (self.in_channels // self.groups) * self.kernel_size[0]
        standardized = standardized * (self.gain[:, None, None] / math.sqrt(fan_in))
        return F.conv1d(
            x,
            standardized,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class ConvBlock(nn.Module):
    """RMSBatchNorm -> activation -> standardized Conv1D (or pointwise Linear)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        groups: int = 1,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Only odd kernel sizes are supported for same padding")
        self.norm = RMSBatchNorm1d(in_channels)
        self.activation = make_activation(activation)
        padding = dilation * (kernel_size - 1) // 2
        if kernel_size == 1:
            self.conv: nn.Module = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv = WSConv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                groups=groups,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.activation(self.norm(x)))


def pad_or_crop_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
    current = x.shape[1]
    if current == channels:
        return x
    if current > channels:
        return x[:, :channels, :]
    return F.pad(x, (0, 0, 0, channels - current), mode="constant", value=0.0)


class MotifConv1d(nn.Module):
    """Parallel 9-/12-bp input filters with a shared output length and channel budget."""

    def __init__(self, out_channels: int = 64, kernels: Sequence[int] = (9, 12)) -> None:
        super().__init__()
        self.kernels = tuple(kernels)
        if not self.kernels or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in self.kernels):
            raise ValueError("Input kernels must be a nonempty tuple of positive integers")
        if out_channels < len(self.kernels):
            raise ValueError("At least one output channel per motif branch is required")
        base, remainder = divmod(out_channels, len(self.kernels))
        self.branches = nn.ModuleList([
            nn.Conv1d(4, base + int(i < remainder), kernel_size=kernel)
            for i, kernel in enumerate(self.kernels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for kernel, conv in zip(self.kernels, self.branches, strict=True):
            # Same output length, also for the even kernel: 12 bp -> 5 left, 6 right.
            left = (kernel - 1) // 2
            outputs.append(conv(F.pad(x, (left, kernel - 1 - left))))
        return torch.cat(outputs, dim=1)


class DNAEmbedder(nn.Module):
    """Parallel 9-/12-bp input filters followed by MAA's 5-bp residual refinement."""

    def __init__(
        self, out_channels: int = 64, activation: ActivationName = "relu",
        kernels: Sequence[int] = (9, 12),
    ) -> None:
        super().__init__()
        self.initial = MotifConv1d(out_channels, kernels)
        self.residual = ConvBlock(
            out_channels, out_channels, kernel_size=5, activation=activation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.initial(x)
        return out + self.residual(out)


class DownresBlock(nn.Module):
    """AlphaGenome-style two-stage residual block with channel expansion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        self.first = ConvBlock(
            in_channels, out_channels, kernel_size=5, activation=activation
        )
        self.second = ConvBlock(
            out_channels, out_channels, kernel_size=5, activation=activation
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.first(x) + pad_or_crop_channels(x, self.out_channels)
        return out + self.second(out)


class UpresBlock(nn.Module):
    """MAA decoder block with optional nearest-repeat upsampling and U-Net skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ActivationName = "relu",
        upsample: bool = True,
    ) -> None:
        super().__init__()
        self.reduce = ConvBlock(
            in_channels, out_channels, kernel_size=5, activation=activation
        )
        self.skip_projection = ConvBlock(
            out_channels, out_channels, kernel_size=1, activation=activation
        )
        self.refine = ConvBlock(
            out_channels, out_channels, kernel_size=5, activation=activation
        )
        self.out_channels = out_channels
        self.upsample = bool(upsample)
        self.residual_scale = nn.Parameter(torch.tensor(0.9, dtype=torch.float32))

    def forward(self, x: torch.Tensor, unet_skip: torch.Tensor) -> torch.Tensor:
        out = self.reduce(x) + pad_or_crop_channels(x, self.out_channels)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode="nearest")
        out = out * self.residual_scale
        if out.shape[-1] != unet_skip.shape[-1]:
            raise ValueError(
                f"Decoder/skip length mismatch: {out.shape[-1]} vs {unet_skip.shape[-1]}"
            )
        out = out + self.skip_projection(unet_skip)
        return out + self.refine(out)


class DilatedMetaFormerBlock(nn.Module):
    """Depthwise dilated token mixer plus channel MLP, both residual."""

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        expansion: int = 2,
        dropout: float = 0.1,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.token_norm = RMSNormChannels(channels)
        self.token_activation = make_activation(activation)
        self.token_mixer = WSConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=channels,
        )
        self.channel_norm = RMSNormChannels(channels)
        self.channel_mlp = nn.Sequential(
            nn.Conv1d(channels, expansion * channels, kernel_size=1),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Conv1d(expansion * channels, channels, kernel_size=1),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.token_mixer(self.token_activation(self.token_norm(x))))
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class TransformerBlock(nn.Module):
    """4-head transformer block with selectable FFN activation."""

    def __init__(
        self,
        channels: int = 128,
        num_heads: int = 4,
        ffn_channels: int = 256,
        dropout: float = 0.1,
        activation: ActivationName = "relu",
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_channels),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ffn_channels, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,L] -> [B,L,C] for nn.MultiheadAttention
        sequence = x.transpose(1, 2)
        normalized = self.norm1(sequence)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        sequence = sequence + attended
        sequence = sequence + self.ffn(self.norm2(sequence))
        return sequence.transpose(1, 2)


# Model configuration and forward pass
# ------------------------------------

@dataclass
class ModelConfig:
    sequence_length: int = 256
    embed_channels: int = 64
    encoder_channels: tuple[int, int, int] = (80, 96, 128)
    bottleneck_length: int = 64
    motif_kernels: tuple[int, ...] = (9, 12)
    transformer_heads: int = 4
    transformer_ffn: int = 256
    dropout: float = 0.1
    channel_dropout: float = 0.0
    activation: ActivationName = "relu"
    classification_readout: str = "center"
    regression_nonnegative: bool = True


class HumanTFBindingModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.sequence_length % 4 != 0:
            raise ValueError("sequence_length must be divisible by 4 for two pool stages")
        if config.sequence_length // 4 != config.bottleneck_length:
            raise ValueError(
                "bottleneck_length must equal sequence_length / 4 for this architecture"
            )
        self.config = config
        if not 0.0 <= config.channel_dropout < 1.0:
            raise ValueError("channel_dropout must be in [0, 1)")
        c0 = config.embed_channels
        c1, c2, c4 = config.encoder_channels

        self.embedder = DNAEmbedder(c0, activation=config.activation, kernels=config.motif_kernels)
        # Drop complete channels across all sequence positions, during training.
        # Identity preserves the original computation and RNG stream at p=0.
        self.channel_dropout = (
            nn.Dropout1d(config.channel_dropout) if config.channel_dropout else nn.Identity()
        )
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.down1 = DownresBlock(c0, c1, activation=config.activation)
        self.down2 = DownresBlock(c1, c2, activation=config.activation)
        self.down3 = DownresBlock(c2, c4, activation=config.activation)

        self.position_embedding = nn.Parameter(
            torch.zeros(1, c4, config.bottleneck_length)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        self.bottleneck = nn.ModuleList(
            [
                DilatedMetaFormerBlock(
                    c4, kernel_size=5, dilation=1, dropout=config.dropout,
                    activation=config.activation,
                ),
                DilatedMetaFormerBlock(
                    c4, kernel_size=5, dilation=2, dropout=config.dropout,
                    activation=config.activation,
                ),
                TransformerBlock(
                    channels=c4,
                    num_heads=config.transformer_heads,
                    ffn_channels=config.transformer_ffn,
                    dropout=config.dropout,
                    activation=config.activation,
                ),
                TransformerBlock(
                    channels=c4,
                    num_heads=config.transformer_heads,
                    ffn_channels=config.transformer_ffn,
                    dropout=config.dropout,
                    activation=config.activation,
                ),
                DilatedMetaFormerBlock(
                    c4, kernel_size=5, dilation=4, dropout=config.dropout,
                    activation=config.activation,
                ),
            ]
        )

        self.up1 = UpresBlock(c4, c2, activation=config.activation, upsample=False)
        self.up2 = UpresBlock(c2, c1, activation=config.activation)
        self.up3 = UpresBlock(c1, c0, activation=config.activation)

        self.classification_head = nn.Conv1d(c0, 1, kernel_size=1)
        self.regression_head = nn.Conv1d(c0, 1, kernel_size=1)

    @staticmethod
    def gather_positions(track: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        if track.ndim != 2:
            raise ValueError(f"Expected [B,L] track, got {tuple(track.shape)}")
        indices = indices.clamp(min=0, max=track.shape[1] - 1)
        return track.gather(1, indices[:, None]).squeeze(1)

    def classify_track(
        self, track: torch.Tensor, center_indices: torch.Tensor
    ) -> torch.Tensor:
        mode = self.config.classification_readout
        if mode == "center":
            return self.gather_positions(track, center_indices)
        if mode == "local-logsumexp":
            width = 64
            if track.shape[1] < width:
                raise ValueError("Local readout requires at least 64 positions")
            start = (track.shape[1] - width) // 2
            return torch.logsumexp(track[:, start:start + width], dim=1) - math.log(width)
        if mode == "max":
            return track.max(dim=1).values
        if mode == "mean":
            return track.mean(dim=1)
        if mode == "logsumexp":
            return torch.logsumexp(track, dim=1) - math.log(track.shape[1])
        raise ValueError(f"Unknown classification readout: {mode}")

    def forward(
        self, sequence: torch.Tensor, center_indices: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if sequence.ndim != 3 or sequence.shape[1] != 4:
            raise ValueError(f"Expected input [B,4,L], got {tuple(sequence.shape)}")
        if sequence.shape[-1] != self.config.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.config.sequence_length}, got {sequence.shape[-1]}"
            )

        skip0 = self.channel_dropout(self.embedder(sequence))  # [B, c0, 256]
        x = self.down1(self.pool(skip0))                 # [B, 80,128]
        skip1 = x
        x = self.down2(self.pool(x))                     # [B, 96, 64]
        skip2 = x
        x = self.down3(x)                                # [B,128, 64]; no third pooling

        x = x + self.position_embedding
        for block in self.bottleneck:
            x = block(x)

        x = self.up1(x, skip2)                           # [B, 96, 64]
        x = self.up2(x, skip1)                           # [B, 80,128]
        x = self.up3(x, skip0)                           # [B, c0, 256]

        classification_track = self.classification_head(x).squeeze(1)
        regression_track_raw = self.regression_head(x).squeeze(1)
        regression_track = (
            F.softplus(regression_track_raw)
            if self.config.regression_nonnegative
            else regression_track_raw
        )
        classification_logit = self.classify_track(
            classification_track, center_indices
        )
        return {
            "classification_track": classification_track,
            "classification_logit": classification_logit,
            "regression_track": regression_track,
            "embedding": x,
        }


# Model builder and parameter count
# ---------------------------------

def maa(context, model_config=None, **params):
    """Build the original MAA model with configurable parallel input kernels."""
    unknown = set(params) - {f.name for f in fields(ModelConfig)}
    if unknown:
        raise ValueError(f"Unknown MAA predictor settings: {sorted(unknown)}")
    if model_config is None:
        params = dict(params)
        for key in ("motif_kernels", "encoder_channels"):
            if key in params:
                params[key] = tuple(params[key])
        model_config = ModelConfig(**params)
    return HumanTFBindingModel(model_config)

def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
