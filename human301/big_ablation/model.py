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


SCRIPT_VERSION = '5.3.0-post-kernel-pwm-channels'

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
    def __init__(self, config: ModelConfig, *, pwm_matrices=None, pwm_manifest=None) -> None:
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
        c0 = config.embed_channels + (len(pwm_matrices) if pwm_matrices is not None else 0)
        self.feature_channels = c0
        c1, c2, c4 = config.encoder_channels

        if pwm_matrices is None:
            self.embedder = DNAEmbedder(c0, activation=config.activation, kernels=config.motif_kernels)
        else:
            self.embedder = PWMDNAEmbedder(
                config.embed_channels, config.motif_kernels, config.activation,
                pwm_matrices, pwm_manifest,
            )
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
    identity = (str(root), tf, assay, pwm_count, sequence_length, learned_channels)
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
        if sequence.ndim != 3 or sequence.shape[1] != 4:
            raise ValueError("FixedPWMScanner expects one-hot [B,4,L] in A,C,G,T order")
        # Score in float32 even in an AMP training/inference context.
        with torch.autocast(device_type=sequence.device.type, enabled=False):
            x = sequence.float()
            known = (x.sum(dim=1, keepdim=True) == 1).float()
            result = x.new_empty(x.shape[0], self.pwm_count, x.shape[-1])
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
            return result


class PWMDNAEmbedder(nn.Module):
    """Concatenate learned kernel outputs and fixed PWM tracks, then refine."""

    def __init__(self, out_channels, kernels, activation, matrices, manifest) -> None:
        super().__init__()
        self.initial = MotifConv1d(out_channels, kernels)
        self.scanner = FixedPWMScanner(matrices, manifest)
        channels = out_channels + len(matrices)
        self.residual = ConvBlock(channels, channels, kernel_size=5, activation=activation)

    def features(self, sequence: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.initial(sequence), self.scanner(sequence)], dim=1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        out = self.features(sequence)
        return out + self.residual(out)


def maa_pwm(context, model_config=None, *, pwm_root: str, pwm_count: int, **params):
    """MAA with K PWM tracks after the learned kernels; unchanged fit interface."""
    config = _maa_config(model_config, params)
    matrices, manifest = _load_pwm_bank(
        context, pwm_root, pwm_count, config.sequence_length, config.embed_channels
    )
    model = HumanTFBindingModel(config, pwm_matrices=matrices, pwm_manifest=manifest)
    recorder = getattr(context, "recorder", None)
    if recorder is not None:
        recorder.update(pwm_channels=manifest)
    run_dir = getattr(context, "run_dir", None)
    if run_dir is not None:
        path = Path(run_dir) / "pwm_channels.json"
        if path.exists():
            if json.loads(path.read_text()) != manifest:
                raise ValueError(f"Refusing to replace a different PWM manifest: {path}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"PWM features: {context.tf}/{manifest['pwm_assay']}, "
          f"{config.embed_channels} learned kernel channels + {pwm_count} fixed PWM "
          f"channels = {model.feature_channels}; maximum over both strands.")
    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable
