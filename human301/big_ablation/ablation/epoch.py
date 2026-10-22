from __future__ import annotations

# Shared implementation extracted from the frozen 4.1.0 reference.
from .common import *
from .model import *
from .losses import *
from .metrics import *

def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in batch.items():
        result[key] = value.to(device, non_blocking=True) if torch.is_tensor(value) else value
    return result


@dataclass
class EpochResult:
    losses: dict[str, float]
    classification: dict[str, float]
    regression: dict[str, float]
    regression_positive: dict[str, float]
    regression_negative: dict[str, float]
    profile_regression: dict[str, float]
    profile_regression_positive: dict[str, float]
    profile_regression_negative: dict[str, float]
    profile_mae: float
    labels: np.ndarray
    probabilities: np.ndarray
    regression_targets: np.ndarray
    regression_predictions: np.ndarray
    sample_ids: list[str]
    chromosomes: list[str]
    window_starts: np.ndarray
    center_indices: np.ndarray
    full_regression_targets: np.ndarray
    full_regression_predictions: np.ndarray


def classification_forward(
    model: HumanTFBindingModel,
    sequence: torch.Tensor,
    center_indices: torch.Tensor,
    eval_crops: str = "multi",
    eval_rc: bool = True,
) -> dict[str, torch.Tensor]:
    if sequence.shape[-1] != STORED_LENGTH:
        return model(sequence, center_indices)
    length = model.config.sequence_length
    starts = evaluation_crop_starts(STORED_LENGTH, length, eval_crops)
    centers = torch.full((sequence.shape[0],), length // 2, dtype=torch.long, device=sequence.device)
    scores = []
    for start in starts:
        crop = sequence[:, :, start:start + length]
        scores.append(torch.sigmoid(model(crop, centers)["classification_logit"].float()))
        if eval_rc:
            reverse = crop[:, [3, 2, 1, 0], :].flip(-1)
            reverse_centers = length - 1 - centers
            scores.append(torch.sigmoid(model(reverse, reverse_centers)["classification_logit"].float()))
    probabilities = torch.stack(scores).mean(dim=0)
    return {
        "classification_logit": torch.logit(probabilities, eps=torch.finfo(torch.float32).eps),
        "classification_probability": probabilities,
    }


@torch.inference_mode()
def score_training_negatives(model, loader, device, eval_crops, eval_rc, use_amp=False):
    """Return scores in the training-negative pool order; no validation/test access."""
    model.eval()
    scores = []
    for raw_batch in loader:
        batch = move_batch_to_device(raw_batch, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            outputs = classification_forward(model, batch["sequence"], batch["center_index"], eval_crops, eval_rc)
        scores.append(outputs["classification_probability"].cpu().numpy())
    return np.concatenate(scores)


def run_epoch(
    model: HumanTFBindingModel,
    loader: DataLoader,
    device: torch.device,
    classification_criterion: nn.Module,
    training_config: TrainingConfig,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    use_amp: bool,
    gradient_clip: float,
    classification_threshold: float = 0.5,
    collect_full_profiles: bool = False,
    eval_crops: str = "multi",
    eval_rc: bool = True,
) -> EpochResult:
    training = optimizer is not None
    model.train(training)
    loss_sums = {"classification": 0.0, "regression": 0.0, "total": 0.0}
    sample_count = 0
    labels_all: list[np.ndarray] = []
    probabilities_all: list[np.ndarray] = []
    targets_all: list[np.ndarray] = []
    predictions_all: list[np.ndarray] = []
    sample_ids: list[str] = []
    chromosomes_all: list[str] = []
    window_starts_all: list[np.ndarray] = []
    center_indices_all: list[np.ndarray] = []
    full_targets_all: list[np.ndarray] = []
    full_predictions_all: list[np.ndarray] = []
    profile_stats_all = new_profile_accumulator()
    profile_stats_positive = new_profile_accumulator()
    profile_stats_negative = new_profile_accumulator()

    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for raw_batch in loader:
            batch = move_batch_to_device(raw_batch, device)
            batch_size = int(batch["label"].shape[0])
            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=amp_device_type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=use_amp,
            ):
                outputs = classification_forward(
                    model, batch["sequence"], batch["center_index"], eval_crops, eval_rc
                )
                loss, components = compute_multitask_loss(
                    outputs,
                    batch,
                    classification_criterion,
                    training_config,
                )

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                    optimizer.step()

            sample_count += batch_size
            for key in loss_sums:
                loss_sums[key] += float(components[key]) * batch_size

            probabilities = outputs.get("classification_probability")
            if probabilities is None:
                probabilities = torch.sigmoid(outputs["classification_logit"].float())
            labels_all.append(batch["label"].detach().cpu().numpy())
            probabilities_all.append(probabilities.detach().cpu().numpy())

            if "fe_profile" in batch:
                # Scalar regression metrics are reported at the center for profile mode;
                # otherwise they follow the configured scalar target.
                scalar_mode: Literal["center", "max", "mean"] = (
                    "center"
                    if training_config.regression_mode == "profile"
                    else training_config.regression_mode
                )
                scalar_target = scalar_from_track(
                    batch["fe_profile"], batch["center_index"], scalar_mode
                )
                scalar_prediction = scalar_from_track(
                    outputs["regression_track"], batch["center_index"], scalar_mode
                )
                targets_all.append(scalar_target.detach().cpu().numpy())
                predictions_all.append(scalar_prediction.detach().cpu().numpy())
            sample_ids.extend(list(raw_batch["sample_id"]))
            chromosomes_all.extend(list(raw_batch["chromosome"]))
            window_starts_all.append(batch["window_start"].detach().cpu().numpy())
            center_indices_all.append(batch["center_index"].detach().cpu().numpy())
            if "fe_profile" in batch:
                if collect_full_profiles:
                    full_targets_all.append(
                        batch["fe_profile"].detach().to(torch.float32).cpu().numpy()
                    )
                    full_predictions_all.append(
                        outputs["regression_track"].detach().to(torch.float32).cpu().numpy()
                    )

                update_profile_accumulator(
                    profile_stats_all,
                    batch["fe_profile"],
                    outputs["regression_track"],
                    batch["valid_mask"],
                )
                update_profile_accumulator(
                    profile_stats_positive,
                    batch["fe_profile"],
                    outputs["regression_track"],
                    batch["valid_mask"] * batch["label"][:, None],
                )
                update_profile_accumulator(
                    profile_stats_negative,
                    batch["fe_profile"],
                    outputs["regression_track"],
                    batch["valid_mask"] * (1.0 - batch["label"][:, None]),
                )

    labels_np = np.concatenate(labels_all).astype(np.int64)
    probabilities_np = np.concatenate(probabilities_all)
    targets_np = np.concatenate(targets_all) if targets_all else np.empty(0, dtype=np.float32)
    predictions_np = np.concatenate(predictions_all) if predictions_all else np.empty(0, dtype=np.float32)
    window_starts_np = np.concatenate(window_starts_all).astype(np.int64, copy=False)
    center_indices_np = np.concatenate(center_indices_all).astype(np.int64, copy=False)
    if collect_full_profiles and full_targets_all:
        full_targets_np = np.concatenate(full_targets_all, axis=0).astype(
            np.float32, copy=False
        )
        full_predictions_np = np.concatenate(full_predictions_all, axis=0).astype(
            np.float32, copy=False
        )
    else:
        full_targets_np = np.empty((0, 0), dtype=np.float32)
        full_predictions_np = np.empty((0, 0), dtype=np.float32)
    positive_mask = labels_np == 1
    negative_mask = labels_np == 0
    profile_all = finalize_profile_accumulator(profile_stats_all)
    profile_positive = finalize_profile_accumulator(profile_stats_positive)
    profile_negative = finalize_profile_accumulator(profile_stats_negative)
    return EpochResult(
        losses={key: value / max(1, sample_count) for key, value in loss_sums.items()},
        classification=classification_metrics(
            labels_np, probabilities_np, threshold=classification_threshold
        ),
        regression=regression_metrics_or_nan(targets_np, predictions_np),
        regression_positive=regression_metrics_or_nan(
            targets_np[positive_mask] if targets_np.size else targets_np,
            predictions_np[positive_mask] if predictions_np.size else predictions_np
        ),
        regression_negative=regression_metrics_or_nan(
            targets_np[negative_mask] if targets_np.size else targets_np,
            predictions_np[negative_mask] if predictions_np.size else predictions_np
        ),
        profile_regression=profile_all,
        profile_regression_positive=profile_positive,
        profile_regression_negative=profile_negative,
        profile_mae=profile_all["mae"],
        labels=labels_np,
        probabilities=probabilities_np,
        regression_targets=targets_np,
        regression_predictions=predictions_np,
        sample_ids=sample_ids,
        chromosomes=chromosomes_all,
        window_starts=window_starts_np,
        center_indices=center_indices_np,
        full_regression_targets=full_targets_np,
        full_regression_predictions=full_predictions_np,
    )

