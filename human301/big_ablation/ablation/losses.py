from __future__ import annotations

# Shared implementation extracted from the frozen 4.1.0 reference.
from .common import *
from .model import HumanTFBindingModel

TaskMode = Literal["classification", "regression", "multitask"]


RegressionMode = Literal["profile", "center", "max", "mean"]


CheckpointObjective = Literal[
    "auto",
    "classification-auprc",
    "regression-loss",
    "regression-profile-mae",
    "regression-profile-pearson",
]


BigWigAggregation = Literal["mean", "center-weighted", "max"]


@dataclass
class TrainingConfig:
    task_mode: TaskMode = "multitask"
    regression_mode: RegressionMode = "profile"
    regression_weight: float = 1.0
    regression_positive_only: bool = False
    smooth_l1_beta: float = 0.5


def scalar_from_track(
    track: torch.Tensor,
    center_indices: torch.Tensor,
    mode: Literal["center", "max", "mean"],
) -> torch.Tensor:
    if mode == "center":
        return HumanTFBindingModel.gather_positions(track, center_indices)
    if mode == "max":
        return track.max(dim=1).values
    if mode == "mean":
        return track.mean(dim=1)
    raise ValueError(f"Unknown scalar track mode: {mode}")


def compute_multitask_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    classification_criterion: nn.Module,
    training_config: TrainingConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    labels = batch["label"]
    classification_loss = classification_criterion(
        outputs["classification_logit"], labels
    )

    if training_config.task_mode == "classification":
        # Keep the regression head in the architecture for later multitask runs,
        # but do not include it in the optimization objective.
        regression_loss = classification_loss.new_zeros(())
        total = classification_loss
    else:
        prediction = outputs["regression_track"]
        target = batch["fe_profile"]
        valid_mask = batch["valid_mask"]
        mode = training_config.regression_mode

        if mode == "profile":
            elementwise = F.smooth_l1_loss(
                prediction,
                target,
                reduction="none",
                beta=training_config.smooth_l1_beta,
            )
            mask = valid_mask
            if training_config.regression_positive_only:
                mask = mask * labels[:, None]
            denominator = mask.sum().clamp_min(1.0)
            regression_loss = (elementwise * mask).sum() / denominator
        else:
            scalar_prediction = scalar_from_track(prediction, batch["center_index"], mode)
            scalar_target = scalar_from_track(target, batch["center_index"], mode)
            elementwise = F.smooth_l1_loss(
                scalar_prediction,
                scalar_target,
                reduction="none",
                beta=training_config.smooth_l1_beta,
            )
            if training_config.regression_positive_only:
                denominator = labels.sum().clamp_min(1.0)
                regression_loss = (elementwise * labels).sum() / denominator
            else:
                regression_loss = elementwise.mean()

        if training_config.task_mode == "regression":
            # Regression-only training: the classification head remains in the model,
            # but classification_loss is excluded from the optimization objective.
            total = regression_loss
        else:
            total = classification_loss + training_config.regression_weight * regression_loss
    return total, {
        "classification": classification_loss.detach(),
        "regression": regression_loss.detach(),
        "total": total.detach(),
    }

