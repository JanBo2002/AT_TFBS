from __future__ import annotations

# Shared implementation extracted from the frozen 4.1.0 reference.
from .common import *
from .blocks import *

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
        c0 = config.embed_channels
        c1, c2, c4 = config.encoder_channels

        self.embedder = DNAEmbedder(c0, activation=config.activation, kernels=config.motif_kernels)
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

        skip0 = self.embedder(sequence)                  # [B, 64,256]
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
        x = self.up3(x, skip0)                           # [B, 64,256]

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

