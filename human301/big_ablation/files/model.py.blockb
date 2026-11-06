"""Shared model and model components for every configuration."""
from __future__ import annotations

import math
import csv
import hashlib
import json
import re
from dataclasses import dataclass, fields
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_VERSION = '5.7.0-block-b-architecture-options'

if torch.version.hip is not None:
    print("ROCm detected: forcing mathematical SDPA backend")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


# Model components
# ----------------

ActivationName = Literal["relu", "gelu", "silu"]
NormalizationName = Literal["rmsbatch", "group"]
PoolingName = Literal["max", "avg", "attention"]


def make_activation(name: ActivationName) -> nn.Module:
    """Create a fresh activation module for one network location."""
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unknown activation: {name}")


class DropPath(nn.Module):
    """Stochastic depth: drops a whole residual branch per sample during training.

    Kept branches are rescaled by 1/(1 - rate). At rate 0 (the default) the
    module is a pure pass-through and consumes no random numbers.
    """

    def __init__(self, rate: float = 0.0) -> None:
        super().__init__()
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not 0.0 <= rate < 1.0:
            raise ValueError("stochastic_depth must be a number in [0, 1)")
        self.rate = float(rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rate == 0.0 or not self.training:
            return x
        keep = 1.0 - self.rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep).div_(keep)
        return x * mask

    def extra_repr(self) -> str:
        return f"rate={self.rate}"


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


def make_norm(kind: NormalizationName, channels: int, groups: int = 8) -> nn.Module:
    """Channel normalization for ConvBlock: batch/EMA RMS statistics or per-sequence GroupNorm."""
    if kind == "rmsbatch":
        return RMSBatchNorm1d(channels)
    if kind == "group":
        if type(groups) is not int or groups < 1 or channels % groups:
            raise ValueError(f"norm_groups={groups!r} must be a positive divisor of every channel count ({channels} channels)")
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"Unknown normalization: {kind}")


class ConvBlock(nn.Module):
    """Norm -> activation -> standardized Conv1D (or pointwise Linear).

    The default normalization is the original RMSBatchNorm1d (batch statistics in
    training, EMA statistics in evaluation); "group" uses nn.GroupNorm, i.e.
    statistics of the sequence itself in both modes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        dilation: int = 1,
        groups: int = 1,
        activation: ActivationName = "relu",
        norm: NormalizationName = "rmsbatch",
        norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Only odd kernel sizes are supported for same padding")
        self.norm = make_norm(norm, in_channels, norm_groups)
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


MotifActivation = Literal["none", "exp"]

# Upper bound of the pre-activation for the exponential first layer. exp(10)
# stays finite in float16 under AMP; the bound only acts as a safety net.
MOTIF_EXP_CLAMP = 10.0


class MotifConv1d(nn.Module):
    """Parallel 9-/12-bp input filters with a shared output length and channel budget.

    activation="exp" applies an exponential activation to the first-layer output
    (Koo & Ploenzke 2021), which favours complete PWM-like motifs per filter.
    rc_sharing=True learns half of each branch's filters and derives the other
    half as their reverse complements, so the first-layer feature set is strand
    symmetric by construction. Both options are off by default and then leave
    the original computation untouched.
    """

    def __init__(
        self, out_channels: int = 64, kernels: Sequence[int] = (9, 12),
        activation: MotifActivation = "none", rc_sharing: bool = False,
    ) -> None:
        super().__init__()
        self.kernels = tuple(kernels)
        if not self.kernels or any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in self.kernels):
            raise ValueError("Input kernels must be a nonempty tuple of positive integers")
        if out_channels < len(self.kernels):
            raise ValueError("At least one output channel per motif branch is required")
        if activation not in ("none", "exp"):
            raise ValueError(f"Unknown motif activation: {activation}")
        if type(rc_sharing) is not bool:
            raise ValueError("rc_sharing must be true or false")
        self.activation = activation
        self.rc_sharing = rc_sharing
        base, remainder = divmod(out_channels, len(self.kernels))
        widths = [base + int(i < remainder) for i in range(len(self.kernels))]
        if rc_sharing and any(width % 2 for width in widths):
            raise ValueError("rc_sharing needs an even channel budget per motif branch")
        self.branches = nn.ModuleList([
            nn.Conv1d(4, width // 2 if rc_sharing else width, kernel_size=kernel)
            for width, kernel in zip(widths, self.kernels)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = []
        for kernel, conv in zip(self.kernels, self.branches, strict=True):
            # Same output length, also for the even kernel: 12 bp -> 5 left, 6 right.
            left = (kernel - 1) // 2
            padded = F.pad(x, (left, kernel - 1 - left))
            if self.rc_sharing:
                # A<->T, C<->G and reversed positions: the reverse-complement filter.
                reverse = conv.weight.flip(dims=(1, 2))
                outputs.append(F.conv1d(padded, conv.weight, conv.bias))
                outputs.append(F.conv1d(padded, reverse, conv.bias))
            else:
                outputs.append(conv(padded))
        out = torch.cat(outputs, dim=1)
        if self.activation == "exp":
            out = torch.exp(out.float().clamp(max=MOTIF_EXP_CLAMP)).to(out.dtype)
        return out


class DNAEmbedder(nn.Module):
    """Parallel input filters followed by configurable 5-bp residual refinements."""

    def __init__(
        self, out_channels: int = 64, activation: ActivationName = "relu",
        kernels: Sequence[int] = (9, 12),
        residual_blocks: int = 1,
        motif_activation: MotifActivation = "none", motif_rc_sharing: bool = False,
        norm: NormalizationName = "rmsbatch", norm_groups: int = 8, drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if type(residual_blocks) is not int or residual_blocks < 0:
            raise ValueError("residual_blocks must be a nonnegative integer")
        self.residual_blocks = residual_blocks
        self.initial = MotifConv1d(out_channels, kernels, motif_activation, motif_rc_sharing)
        # Preserve the original parameter names and initialization order at depth 1.
        self.residual = ConvBlock(
            out_channels, out_channels, kernel_size=5, activation=activation, norm=norm, norm_groups=norm_groups,
        ) if residual_blocks else nn.Identity()
        self.additional_residuals = nn.ModuleList([
            ConvBlock(out_channels, out_channels, kernel_size=5, activation=activation, norm=norm, norm_groups=norm_groups)
            for _ in range(max(0, residual_blocks - 1))
        ])
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.initial(x)
        if self.residual_blocks:
            out = out + self.drop_path(self.residual(out))
        for block in self.additional_residuals:
            out = out + self.drop_path(block(out))
        return out


class DownresBlock(nn.Module):
    """AlphaGenome-style two-stage residual block with channel expansion."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ActivationName = "relu",
        norm: NormalizationName = "rmsbatch",
        norm_groups: int = 8,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.first = ConvBlock(
            in_channels, out_channels, kernel_size=5, activation=activation, norm=norm, norm_groups=norm_groups,
        )
        self.second = ConvBlock(
            out_channels, out_channels, kernel_size=5, activation=activation, norm=norm, norm_groups=norm_groups,
        )
        self.out_channels = out_channels
        # Stochastic depth only on true identity residuals: the channel-changing
        # first branch has no identity to fall back to and is never dropped.
        self.drop_path = DropPath(drop_path)
        self.drop_first = in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = self.first(x)
        if self.drop_first:
            first = self.drop_path(first)
        out = first + pad_or_crop_channels(x, self.out_channels)
        return out + self.drop_path(self.second(out))


class UpresBlock(nn.Module):
    """MAA decoder block with optional nearest-repeat upsampling and U-Net skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ActivationName = "relu",
        upsample: bool = True,
        norm: NormalizationName = "rmsbatch",
        norm_groups: int = 8,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.reduce = ConvBlock(
            in_channels, out_channels, kernel_size=5, activation=activation, norm=norm, norm_groups=norm_groups,
        )
        self.skip_projection = ConvBlock(
            out_channels, out_channels, kernel_size=1, activation=activation, norm=norm, norm_groups=norm_groups,
        )
        self.refine = ConvBlock(
            out_channels, out_channels, kernel_size=5, activation=activation, norm=norm, norm_groups=norm_groups,
        )
        self.out_channels = out_channels
        self.upsample = bool(upsample)
        self.residual_scale = nn.Parameter(torch.tensor(0.9, dtype=torch.float32))
        # As in DownresBlock: only same-width residual branches are dropped.
        self.drop_path = DropPath(drop_path)
        self.drop_reduce = in_channels == out_channels

    def forward(self, x: torch.Tensor, unet_skip: torch.Tensor) -> torch.Tensor:
        reduced = self.reduce(x)
        if self.drop_reduce:
            reduced = self.drop_path(reduced)
        out = reduced + pad_or_crop_channels(x, self.out_channels)
        if self.upsample:
            out = F.interpolate(out, scale_factor=2, mode="nearest")
        out = out * self.residual_scale
        if out.shape[-1] != unet_skip.shape[-1]:
            raise ValueError(
                f"Decoder/skip length mismatch: {out.shape[-1]} vs {unet_skip.shape[-1]}"
            )
        out = out + self.skip_projection(unet_skip)
        return out + self.drop_path(self.refine(out))


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
        drop_path: float = 0.0,
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
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.dropout(self.token_mixer(self.token_activation(self.token_norm(x)))))
        x = x + self.drop_path(self.channel_mlp(self.channel_norm(x)))
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
        drop_path: float = 0.0,
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
        self.drop_path = DropPath(drop_path)

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
        sequence = sequence + self.drop_path(attended)
        sequence = sequence + self.drop_path(self.ffn(self.norm2(sequence)))
        return sequence.transpose(1, 2)


class AttentionPool1d(nn.Module):
    """Enformer-style attention pooling over non-overlapping windows.

    Per channel, the positions of a window are weighted by a softmax over
    learned logits (1x1 convolution, initialized to 2 * identity, so that the
    initial behaviour is a soft maximum of the activations themselves).
    """

    def __init__(self, channels: int, window: int = 2) -> None:
        super().__init__()
        self.window = window
        self.logits = nn.Conv1d(channels, channels, kernel_size=1, bias=False)
        with torch.no_grad():
            self.logits.weight.copy_(2.0 * torch.eye(channels)[:, :, None])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        if length % self.window:
            raise ValueError(f"Length {length} is not divisible by the pooling window {self.window}")
        logits = self.logits(x).reshape(batch, channels, length // self.window, self.window)
        weights = torch.softmax(logits.float(), dim=-1).to(x.dtype)
        return (x.reshape(batch, channels, length // self.window, self.window) * weights).sum(dim=-1)


class SqueezeExcite(nn.Module):
    """Squeeze-and-excitation: each sequence reweights its own channels.

    The squeeze is a global maximum over positions by default (LegNet-Max
    style), optionally the mean; the excitation is a two-layer bottleneck MLP
    with a sigmoid gate.
    """

    def __init__(self, channels: int, reduction: int = 4, pool: str = "max",
                 activation: ActivationName = "relu") -> None:
        super().__init__()
        if type(reduction) is not int or reduction < 1:
            raise ValueError("se_reduction must be a positive integer")
        if pool not in ("max", "mean"):
            raise ValueError(f"Unknown SE squeeze: {pool}")
        self.pool = pool
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Conv1d(channels, hidden, kernel_size=1)
        self.activation = make_activation(activation)
        self.fc2 = nn.Conv1d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        squeezed = x.amax(dim=-1, keepdim=True) if self.pool == "max" else x.mean(dim=-1, keepdim=True)
        gate = torch.sigmoid(self.fc2(self.activation(self.fc1(squeezed))))
        return x * gate


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
    embedder_residual_blocks: int = 1
    motif_activation: MotifActivation = "none"
    motif_rc_sharing: bool = False
    decoder: bool = True
    normalization: NormalizationName = "rmsbatch"
    norm_groups: int = 8
    pooling: PoolingName = "max"
    pooling_stages: int = 2
    se_blocks: bool = False
    se_reduction: int = 4
    se_pool: str = "max"
    stochastic_depth: float = 0.0


class HumanTFBindingModel(nn.Module):
    def __init__(self, config: ModelConfig, *, pwm_matrices=None, pwm_manifest=None, pwm_target=None) -> None:
        super().__init__()
        if config.pooling_stages not in (1, 2):
            raise ValueError("pooling_stages must be 1 or 2")
        stride = 2 ** config.pooling_stages
        if config.sequence_length % stride != 0:
            raise ValueError(f"sequence_length must be divisible by {stride} for {config.pooling_stages} pool stage(s)")
        if config.sequence_length // stride != config.bottleneck_length:
            raise ValueError(
                f"bottleneck_length must equal sequence_length / {stride} for {config.pooling_stages} pool stage(s)"
            )
        if config.pooling not in ("max", "avg", "attention"):
            raise ValueError(f"Unknown pooling: {config.pooling}")
        if type(config.decoder) is not bool or type(config.se_blocks) is not bool:
            raise ValueError("decoder and se_blocks must be true or false")
        if pwm_target is not None and not config.decoder:
            raise ValueError("The PWM auxiliary target needs the decoder (base-resolution features)")
        self.config = config
        self.readout_stride = 1 if config.decoder else stride
        if not 0.0 <= config.channel_dropout < 1.0:
            raise ValueError("channel_dropout must be in [0, 1)")
        c0 = config.embed_channels + (len(pwm_matrices) if pwm_matrices is not None else 0)
        self.feature_channels = c0
        c1, c2, c4 = config.encoder_channels
        # Settings shared by every convolutional/residual block.
        block = dict(activation=config.activation, norm=config.normalization,
                     norm_groups=config.norm_groups, drop_path=config.stochastic_depth)

        if pwm_matrices is None:
            self.embedder = DNAEmbedder(
                c0, kernels=config.motif_kernels,
                residual_blocks=config.embedder_residual_blocks,
                motif_activation=config.motif_activation, motif_rc_sharing=config.motif_rc_sharing,
                **block,
            )
        else:
            if type(config.embedder_residual_blocks) is not int or config.embedder_residual_blocks != 1:
                raise ValueError("Variable embedder residual depth requires a model without PWM channels")
            self.embedder = PWMDNAEmbedder(
                config.embed_channels, config.motif_kernels, config.activation,
                pwm_matrices, pwm_manifest,
                motif_activation=config.motif_activation, motif_rc_sharing=config.motif_rc_sharing,
                norm=config.normalization, norm_groups=config.norm_groups, drop_path=config.stochastic_depth,
            )
        # Drop complete channels across all sequence positions, during training.
        # Identity preserves the original computation and RNG stream at p=0.
        self.channel_dropout = (
            nn.Dropout1d(config.channel_dropout) if config.channel_dropout else nn.Identity()
        )
        if config.pooling == "max":
            self.pool: nn.Module = nn.MaxPool1d(kernel_size=2, stride=2)
        elif config.pooling == "avg":
            self.pool = nn.AvgPool1d(kernel_size=2, stride=2)
        else:
            # One learned pooling module per stage (channel counts differ).
            self.pool = nn.ModuleList([AttentionPool1d(c0)] + ([AttentionPool1d(c1)] if config.pooling_stages == 2 else []))
        self.down1 = DownresBlock(c0, c1, **block)
        self.down2 = DownresBlock(c1, c2, **block)
        self.down3 = DownresBlock(c2, c4, **block)
        self.se_down = None
        if config.se_blocks:
            self.se_down = nn.ModuleList([
                SqueezeExcite(channels, config.se_reduction, config.se_pool, config.activation)
                for channels in (c1, c2, c4)
            ])

        self.position_embedding = nn.Parameter(
            torch.zeros(1, c4, config.bottleneck_length)
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

        self.bottleneck = nn.ModuleList(
            [
                DilatedMetaFormerBlock(
                    c4, kernel_size=5, dilation=1, dropout=config.dropout,
                    activation=config.activation, drop_path=config.stochastic_depth,
                ),
                DilatedMetaFormerBlock(
                    c4, kernel_size=5, dilation=2, dropout=config.dropout,
                    activation=config.activation, drop_path=config.stochastic_depth,
                ),
                TransformerBlock(
                    channels=c4,
                    num_heads=config.transformer_heads,
                    ffn_channels=config.transformer_ffn,
                    dropout=config.dropout,
                    activation=config.activation,
                    drop_path=config.stochastic_depth,
                ),
                TransformerBlock(
                    channels=c4,
                    num_heads=config.transformer_heads,
                    ffn_channels=config.transformer_ffn,
                    dropout=config.dropout,
                    activation=config.activation,
                    drop_path=config.stochastic_depth,
                ),
                DilatedMetaFormerBlock(
                    c4, kernel_size=5, dilation=4, dropout=config.dropout,
                    activation=config.activation, drop_path=config.stochastic_depth,
                ),
            ]
        )

        self.se_up = None
        if config.decoder:
            self.up1 = UpresBlock(c4, c2, upsample=False, **block)
            # With a single pooling stage the second decoder block stays at the
            # bottleneck resolution; only the last block returns to 256 bp.
            self.up2 = UpresBlock(c2, c1, upsample=(config.pooling_stages == 2), **block)
            self.up3 = UpresBlock(c1, c0, **block)
            if config.se_blocks:
                self.se_up = nn.ModuleList([
                    SqueezeExcite(channels, config.se_reduction, config.se_pool, config.activation)
                    for channels in (c2, c1, c0)
                ])
            head_channels = c0
        else:
            # No decoder: the heads read the bottleneck directly (stride 4 or 2
            # base pairs per position); the regression track is repeated back
            # to base resolution so that consumers see the usual length.
            head_channels = c4

        self.classification_head = nn.Conv1d(head_channels, 1, kernel_size=1)
        self.regression_head = nn.Conv1d(head_channels, 1, kernel_size=1)
        # Optional auxiliary target: predict fixed PWM score tracks from the
        # decoder output. Registered last, so every existing parameter keeps its
        # name, shape and initialization order when the option is off.
        self.pwm_target = None
        self.pwm_target_weight = 0.0
        if pwm_target is not None:
            matrices, manifest, weight = pwm_target
            if not isinstance(weight, (int, float)) or isinstance(weight, bool) or weight <= 0:
                raise ValueError("pwm_target_weight must be a positive number")
            self.pwm_target = PWMTargetHead(c0, matrices, manifest)
            self.pwm_target_weight = float(weight)

    def _pool(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        if isinstance(self.pool, nn.ModuleList):
            return self.pool[stage](x)
        return self.pool(x)

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
        x = self.down1(self._pool(skip0, 0))             # [B, 80,128]
        if self.se_down is not None:
            x = self.se_down[0](x)
        skip1 = x
        x = self.down2(self._pool(x, 1) if self.config.pooling_stages == 2 else x)  # [B, 96, 64] (or 128)
        if self.se_down is not None:
            x = self.se_down[1](x)
        skip2 = x
        x = self.down3(x)                                # [B,128, 64]; no third pooling
        if self.se_down is not None:
            x = self.se_down[2](x)

        x = x + self.position_embedding
        for block in self.bottleneck:
            x = block(x)

        if self.config.decoder:
            x = self.up1(x, skip2)                       # [B, 96, 64]
            if self.se_up is not None:
                x = self.se_up[0](x)
            x = self.up2(x, skip1)                       # [B, 80,128]
            if self.se_up is not None:
                x = self.se_up[1](x)
            x = self.up3(x, skip0)                       # [B, c0, 256]
            if self.se_up is not None:
                x = self.se_up[2](x)

        classification_track = self.classification_head(x).squeeze(1)
        regression_track_raw = self.regression_head(x).squeeze(1)
        regression_track = (
            F.softplus(regression_track_raw)
            if self.config.regression_nonnegative
            else regression_track_raw
        )
        if self.readout_stride > 1:
            regression_track = F.interpolate(
                regression_track[:, None], scale_factor=self.readout_stride, mode="nearest"
            ).squeeze(1)
        classification_logit = self.classify_track(
            classification_track, center_indices // self.readout_stride
        )
        result = {
            "classification_track": classification_track,
            "classification_logit": classification_logit,
            "regression_track": regression_track,
            "embedding": x,
        }
        if self.pwm_target is not None:
            result["auxiliary_loss"] = self.pwm_target_weight * self.pwm_target(x, sequence)
        return result


# Model builder and parameter count
# ---------------------------------

def _maa_config(model_config, params):
    unknown = set(params) - {f.name for f in fields(ModelConfig)}
    if unknown:
        raise ValueError(f"Unknown MAA predictor settings: {sorted(unknown)}")
    if model_config is None:
        params = dict(params)
        for key in ("motif_kernels", "encoder_channels"):
            if key in params:
                params[key] = tuple(params[key])
        model_config = ModelConfig(**params)
    return model_config


def maa(context, model_config=None, **params):
    """Build the original MAA model with configurable parallel input kernels."""
    return HumanTFBindingModel(_maa_config(model_config, params))


# Optional fixed PWM features after the learned input kernels.
# ------------------------------------------------------------------------

def _pwm_file(path: Path) -> tuple[str, list[list[float]], str]:
    """Read original position x A/C/G/T ln-log-odds; never re-log or rescale."""
    content = path.read_bytes()
    rows, header = [], ""
    for line_number, line in enumerate(content.decode("utf-8-sig").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(">"):
            if rows or header:
                raise ValueError(f"{path}:{line_number}: expected one PWM per file")
            header = line[1:].strip()
            continue
        try:
            numbers = [Decimal(token) for token in line.split()]
        except InvalidOperation as error:
            raise ValueError(f"{path}:{line_number}: invalid PWM row") from error
        if len(numbers) != 4 or any(not value.is_finite() for value in numbers):
            raise ValueError(f"{path}:{line_number}: four finite A/C/G/T scores required")
        row = [float(value) for value in numbers]
        # Validate the same file convention as PWM_Auswahl v1.1.0. This check
        # changes no values: the scanner uses the four original scores below.
        try:
            restored = [0.25 * math.exp(value) - 1e-5 for value in row]
        except OverflowError as error:
            raise ValueError(f"{path}:{line_number}: not supported ln-log-odds") from error
        if (any(not math.isfinite(value) or value < -1e-9 for value in restored)
                or abs(sum(max(0.0, value) for value in restored) - 1.0) > 0.0021):
            raise ValueError(
                f"{path}:{line_number}: expected ln((p + 1e-5) / 0.25), "
                "with row rounding tolerance 0.0021; refusing another PWM format"
            )
        rows.append(row)
    if not rows:
        raise ValueError(f"Empty PWM: {path}")
    return header, rows, hashlib.sha256(content).hexdigest()


def _load_pwm_bank(context, pwm_root: str, pwm_count: int, sequence_length: int, learned_channels: int):
    """Use the first K selected representatives, ordered by the stored rank."""
    if type(pwm_count) is not int or pwm_count < 1:
        raise ValueError("pwm_count must be a positive integer")
    tf = context.tf
    if not isinstance(tf, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", tf):
        raise ValueError("maa_pwm requires a single-TF run with a valid context.tf")
    assay_map = {"CHIP": "CHS", "CHS": "CHS", "GHTS": "GHTS"}
    if context.train_assay not in assay_map:
        raise ValueError(f"No PWM assay mapping for {context.train_assay!r}")
    assay = assay_map[context.train_assay]
    root = Path(pwm_root).expanduser().resolve()
    from data import validate_pwm_split_provenance
    split_provenance = validate_pwm_split_provenance(context, root, assay)
    identity = (str(root), tf, assay, pwm_count, sequence_length, learned_channels)
    if split_provenance is not None:
        identity += (split_provenance["sha256"],)
    artifacts = getattr(context, "artifacts", None)
    cached = artifacts.get("_maa_pwm_bank") if artifacts is not None else None
    if cached is not None:
        if cached[0] != identity:
            raise ValueError("The PWM bank cannot change within a training/evaluation run")
        return cached[1], cached[2]

    table = root / "selected.tsv"
    table_bytes = table.read_bytes()
    reader = csv.DictReader(table_bytes.decode("utf-8-sig").splitlines(), delimiter="\t")
    required = {"TF", "Assay", "Matrix_ID", "Status", "Approved", "Eligible",
                "Selected_order", "MEX_overall_rank", "Rank_motif_ID", "Width", "Copied_to"}
    if required - set(reader.fieldnames or []):
        raise ValueError(f"{table}: missing columns {sorted(required - set(reader.fieldnames or []))}")
    records = [row for row in reader if row["TF"] == tf and row["Assay"] == assay]
    if len(records) < pwm_count:
        raise ValueError(f"{tf}/{assay}: requested {pwm_count} ranked PWMs, found {len(records)}")
    try:
        records.sort(key=lambda row: int(row["Selected_order"]))
        orders = [int(row["Selected_order"]) for row in records]
        ranks = [int(row["MEX_overall_rank"]) for row in records]
    except (ValueError, TypeError) as error:
        raise ValueError(f"{table}: noninteger selection order or MEX rank for {tf}/{assay}") from error
    if orders != list(range(1, len(records) + 1)) or ranks != sorted(ranks) or min(ranks) < 1:
        raise ValueError(f"{table}: inconsistent selection order or MEX ranking for {tf}/{assay}")

    matrices, channels, seen = [], [], set()
    for channel, row in enumerate(records[:pwm_count]):
        if (row["Status"] != "selected" or row["Approved"].strip().lower() != "yes"
                or row["Eligible"].strip().lower() != "true"):
            raise ValueError(f"{tf}/{assay}: PWM {row['Matrix_ID']} is not approved and eligible")
        name = Path(row["Copied_to"]).name
        if not name.endswith(".pwm"):
            raise ValueError(f"Unexpected selected PWM filename: {name}")
        path = root / tf / assay / "selected" / name
        header, matrix, sha256 = _pwm_file(path)
        if len(matrix) != int(row["Width"]) or len(matrix) > sequence_length:
            raise ValueError(f"Invalid PWM width for input length {sequence_length}: {path}")
        numeric_key = tuple(tuple(position) for position in matrix)
        if numeric_key in seen:
            raise ValueError(f"Duplicate numeric PWM in selected inputs: {path}")
        seen.add(numeric_key)
        matrices.append(matrix)
        channels.append({
            "pwm_channel": channel + 1, "feature_channel_zero_based": channel + learned_channels,
            "selected_order": int(row["Selected_order"]), "matrix_id": row["Matrix_ID"],
            "mex_overall_rank": int(row["MEX_overall_rank"]),
            "rank_motif_id": row["Rank_motif_ID"], "original_header": header,
            "width": len(matrix), "file": str(path), "sha256": sha256,
        })
    manifest = {
        "schema_version": "2.0", "tf": tf, "train_assay": context.train_assay,
        "pwm_assay": assay, "pwm_count": pwm_count, "dna_input_channels": 4,
        "learned_channels": learned_channels, "fused_channels": learned_channels + pwm_count,
        "fusion_stage": "after_learned_motif_convolution_before_residual",
        "pwm_root": str(root), "selected_tsv_sha256": hashlib.sha256(table_bytes).hexdigest(),
        "format": "original_ln_log_odds_ACGT", "scoring": "sum_original_scores",
        "strands": "maximum_forward_reverse_complement_per_window",
        "alignment": "anchor_at_start_plus_floor((width-1)/2)",
        "invalid_windows": "sum_of_per_position_minimum_scores",
        "unknown_bases": "mask_any_window_containing_N",
        "trainable_pwm_weights": False, "channels": channels,
    }
    if split_provenance is not None:
        manifest["split_provenance"] = split_provenance
    if artifacts is not None:
        # Inference reuses this source-assay bank even when evaluating a
        # different target assay. File changes mid-run cannot replace it.
        artifacts["_maa_pwm_bank"] = (identity, matrices, manifest)
    return matrices, manifest


class FixedPWMScanner(nn.Module):
    """Exactly one fixed, both-strand score channel per original PWM."""

    def __init__(self, matrices, manifest) -> None:
        super().__init__()
        self.manifest = json.loads(json.dumps(manifest))
        self.pwm_count = len(matrices)
        grouped = {}
        for index, matrix in enumerate(matrices):
            weight = torch.tensor(matrix, dtype=torch.float32).transpose(0, 1).contiguous()
            grouped.setdefault(len(matrix), []).append((index, weight))
        self.widths = tuple(sorted(grouped))
        for width, entries in grouped.items():
            forward = torch.stack([weight for _, weight in entries])
            reverse = forward.flip(dims=(1, 2))  # A,C,G,T -> T,G,C,A and reverse positions.
            self.register_buffer(f"weights_{width}", torch.cat([forward, reverse], dim=0))
            self.register_buffer(f"indices_{width}", torch.tensor([i for i, _ in entries]))
            self.register_buffer(f"floor_{width}", forward.min(dim=1).values.sum(dim=1))
            self.register_buffer(f"mask_{width}", torch.ones(1, 1, width))

    def get_extra_state(self):
        # Plain JSON types remain compatible with torch.load(weights_only=True).
        return json.loads(json.dumps(self.manifest))

    def set_extra_state(self, state):
        if state != self.manifest:
            raise RuntimeError("Checkpoint PWM identities/ranks/checksums differ from this source PWM bank")

    @torch.no_grad()
    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.score_with_mask(sequence)[0]

    @torch.no_grad()
    def score_with_mask(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Scores as in forward plus a [B,K,L] mask of positions whose window is complete and N-free."""
        if sequence.ndim != 3 or sequence.shape[1] != 4:
            raise ValueError("FixedPWMScanner expects one-hot [B,4,L] in A,C,G,T order")
        # Score in float32 even in an AMP training/inference context.
        with torch.autocast(device_type=sequence.device.type, enabled=False):
            x = sequence.float()
            known = (x.sum(dim=1, keepdim=True) == 1).float()
            result = x.new_empty(x.shape[0], self.pwm_count, x.shape[-1])
            result_mask = torch.zeros(x.shape[0], self.pwm_count, x.shape[-1], dtype=torch.bool, device=x.device)
            for width in self.widths:
                if width > x.shape[-1]:
                    raise ValueError(f"PWM width {width} exceeds sequence length {x.shape[-1]}")
                weights = getattr(self, f"weights_{width}")
                indices = getattr(self, f"indices_{width}")
                floor = getattr(self, f"floor_{width}")[None, :, None]
                count = indices.numel()
                matches = F.conv1d(x, weights)
                scores = torch.maximum(matches[:, :count], matches[:, count:])
                valid = F.conv1d(known, getattr(self, f"mask_{width}")) == width
                scores = torch.where(valid, scores, floor)
                full = floor.expand(x.shape[0], count, x.shape[-1]).clone()
                left = (width - 1) // 2
                full[:, :, left:left + scores.shape[-1]] = scores
                result.index_copy_(1, indices, full)
                full_mask = torch.zeros(x.shape[0], count, x.shape[-1], dtype=torch.bool, device=x.device)
                full_mask[:, :, left:left + scores.shape[-1]] = valid.expand(-1, count, -1)
                result_mask.index_copy_(1, indices, full_mask)
            return result, result_mask


class PWMDNAEmbedder(nn.Module):
    """Concatenate learned kernel outputs and fixed PWM tracks, then refine."""

    def __init__(self, out_channels, kernels, activation, matrices, manifest,
                 motif_activation: MotifActivation = "none", motif_rc_sharing: bool = False,
                 norm: NormalizationName = "rmsbatch", norm_groups: int = 8, drop_path: float = 0.0) -> None:
        super().__init__()
        self.initial = MotifConv1d(out_channels, kernels, motif_activation, motif_rc_sharing)
        self.scanner = FixedPWMScanner(matrices, manifest)
        channels = out_channels + len(matrices)
        self.residual = ConvBlock(channels, channels, kernel_size=5, activation=activation, norm=norm, norm_groups=norm_groups)
        self.drop_path = DropPath(drop_path)

    def features(self, sequence: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.initial(sequence), self.scanner(sequence)], dim=1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        out = self.features(sequence)
        return out + self.drop_path(self.residual(out))


def _record_pwm_manifest(context, manifest, filename, key):
    recorder = getattr(context, "recorder", None)
    if recorder is not None:
        recorder.update(**{key: manifest})
    run_dir = getattr(context, "run_dir", None)
    if run_dir is not None:
        path = Path(run_dir) / filename
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError(f"Refusing to replace a different PWM manifest: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")


def maa_pwm(context, model_config=None, *, pwm_root: str, pwm_count: int, **params):
    """MAA with K PWM tracks after the learned kernels; unchanged fit interface."""
    config = _maa_config(model_config, params)
    matrices, manifest = _load_pwm_bank(
        context, pwm_root, pwm_count, config.sequence_length, config.embed_channels
    )
    model = HumanTFBindingModel(config, pwm_matrices=matrices, pwm_manifest=manifest)
    _record_pwm_manifest(context, manifest, "pwm_channels.json", "pwm_channels")
    print(f"PWM features: {context.tf}/{manifest['pwm_assay']}, "
          f"{config.embed_channels} learned kernel channels + {pwm_count} fixed PWM "
          f"channels = {model.feature_channels}; maximum over both strands.")
    return model


class PWMTargetHead(nn.Module):
    """Auxiliary regression of fixed PWM score tracks from the decoder output.

    The input stays the plain one-hot sequence; the network only receives the
    PWM scores as a training target. Each PWM's two-strand window score is
    standardized with its expected mean and standard deviation under a uniform
    base background (single-strand moments from the matrix, then the maximum of
    two approximately independent normal strands: mean + std/sqrt(pi), std *
    sqrt(1 - 1/pi)), so random windows score about N(0, 1) and motif hits stand
    out at several standard deviations; the loss is therefore on the same scale
    as the classification loss. Positions whose window leaves the sequence or
    contains N are excluded from the loss.
    """

    def __init__(self, channels: int, matrices, manifest) -> None:
        super().__init__()
        self.scanner = FixedPWMScanner(matrices, manifest)
        means, stds = [], []
        for matrix in matrices:
            position_means = [sum(row) / 4.0 for row in matrix]
            position_variances = [sum((value - mean) ** 2 for value in row) / 4.0 for row, mean in zip(matrix, position_means)]
            strand_mean, strand_std = sum(position_means), math.sqrt(sum(position_variances))
            means.append(strand_mean + strand_std / math.sqrt(math.pi))
            stds.append(strand_std * math.sqrt(1.0 - 1.0 / math.pi))
        self.register_buffer("target_mean", torch.tensor(means, dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(stds, dtype=torch.float32).clamp_min(1e-6))
        self.head = nn.Conv1d(channels, len(matrices), kernel_size=1)

    def forward(self, features: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            scores, valid = self.scanner.score_with_mask(sequence)
            target = (scores - self.target_mean[None, :, None]) / self.target_std[None, :, None]
            mask = valid.to(torch.float32)
        with torch.autocast(device_type=features.device.type, enabled=False):
            prediction = self.head(features.float())
            squared = (prediction - target) ** 2
            return (squared * mask).sum() / mask.sum().clamp_min(1.0)


def maa_pwm_target(context, model_config=None, *, pwm_root: str, pwm_count: int,
                   pwm_target_weight: float = 1.0, **params):
    """MAA with the K PWM score tracks as an auxiliary training target only.

    Unlike maa_pwm, the PWM scores never enter the input; the model must encode
    motif hits itself and is pulled towards motif dependence by the auxiliary
    loss (weight * MSE on the scaled tracks), which is added to the training
    objective. Inference is unchanged: the auxiliary head is ignored.
    """
    config = _maa_config(model_config, params)
    matrices, manifest = _load_pwm_bank(
        context, pwm_root, pwm_count, config.sequence_length, config.embed_channels
    )
    manifest = dict(manifest, usage="auxiliary_regression_target", target_weight=float(pwm_target_weight),
                    target_scaling="two_strand_max_standardized_by_uniform_background_moments_per_pwm",
                    masked_positions="incomplete_windows_and_windows_with_N",
                    fusion_stage="none_input_is_one_hot_only")
    model = HumanTFBindingModel(config, pwm_target=(matrices, manifest, pwm_target_weight))
    _record_pwm_manifest(context, manifest, "pwm_target.json", "pwm_target")
    print(f"PWM auxiliary target: {context.tf}/{manifest['pwm_assay']}, {pwm_count} fixed PWM tracks, "
          f"loss weight {float(pwm_target_weight)}; input stays one-hot only.")
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
