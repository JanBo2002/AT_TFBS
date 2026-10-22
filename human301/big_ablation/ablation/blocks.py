from __future__ import annotations

# Shared implementation extracted from the frozen 4.1.0 reference.
from .common import *

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

