from __future__ import annotations

# Shared implementation extracted from the frozen 4.1.0 reference.
from .common import *
from .losses import *

def safe_correlation(function, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return float("nan")
    result = function(y_true, y_pred)
    value = result.statistic if hasattr(result, "statistic") else result[0]
    return float(value)


def classification_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    specificity = tn / max(1, tn + fp)
    negative_predictive_value = tn / max(1, tn + fn)
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "specificity": float(specificity),
        "negative_predictive_value": float(negative_predictive_value),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "mcc": float(matthews_corrcoef(labels, predictions)),
        "true_negative": float(tn),
        "false_positive": float(fp),
        "false_negative": float(fn),
        "true_positive": float(tp),
    }
    if np.unique(labels).size == 2:
        metrics["auroc"] = float(roc_auc_score(labels, probabilities))
        metrics["auprc"] = float(average_precision_score(labels, probabilities))
    else:
        metrics["auroc"] = float("nan")
        metrics["auprc"] = float("nan")
    return metrics


def select_classification_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    strategy: str,
    fixed_threshold: float = 0.5,
) -> tuple[float, float]:
    """Choose a binary threshold using validation predictions only.

    The sweep is exact over all unique predicted probabilities and is O(n log n).
    The test chromosome is never used to select the threshold.
    """
    if strategy == "fixed":
        return float(fixed_threshold), float("nan")
    if np.unique(labels).size != 2:
        raise ValueError("Threshold selection requires both classes in validation data")

    order = np.argsort(-probabilities, kind="mergesort")
    sorted_prob = probabilities[order]
    sorted_label = labels[order].astype(np.int64)
    cumulative_tp = np.cumsum(sorted_label)
    cumulative_fp = np.cumsum(1 - sorted_label)
    group_ends = np.flatnonzero(
        np.r_[sorted_prob[1:] != sorted_prob[:-1], True]
    )

    tp = cumulative_tp[group_ends].astype(np.float64)
    fp = cumulative_fp[group_ends].astype(np.float64)
    positives = float(sorted_label.sum())
    negatives = float(sorted_label.size - sorted_label.sum())
    fn = positives - tp
    tn = negatives - fp
    thresholds = sorted_prob[group_ends].astype(np.float64)

    if strategy == "val-f1":
        scores = 2.0 * tp / np.maximum(1.0, 2.0 * tp + fp + fn)
    elif strategy == "val-balanced-accuracy":
        sensitivity = tp / np.maximum(1.0, tp + fn)
        specificity = tn / np.maximum(1.0, tn + fp)
        scores = 0.5 * (sensitivity + specificity)
    elif strategy == "val-mcc":
        numerator = tp * tn - fp * fn
        denominator = np.sqrt(
            np.maximum(1.0, (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        )
        scores = numerator / denominator
    else:
        raise ValueError(f"Unknown threshold strategy: {strategy}")

    best_score = float(np.nanmax(scores))
    best_indices = np.flatnonzero(np.isclose(scores, best_score, rtol=1e-12, atol=1e-12))
    # Prefer the equally good threshold closest to 0.5 for stable behavior.
    best_index = int(best_indices[np.argmin(np.abs(thresholds[best_indices] - 0.5))])
    return float(thresholds[best_index]), best_score


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    difference = predictions - targets
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(np.square(difference)))),
        "pearson": safe_correlation(pearsonr, targets, predictions),
        "spearman": safe_correlation(spearmanr, targets, predictions),
    }


def regression_metrics_or_nan(
    targets: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    """Regression metrics that remain well-defined for an empty subgroup."""
    if targets.size == 0:
        return {
            "n_samples": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
        }
    metrics = regression_metrics(targets, predictions)
    metrics["n_samples"] = float(targets.size)
    return metrics


def new_profile_accumulator() -> dict[str, float]:
    return {
        "n_values": 0.0,
        "sum_target": 0.0,
        "sum_prediction": 0.0,
        "sum_target_sq": 0.0,
        "sum_prediction_sq": 0.0,
        "sum_product": 0.0,
        "sum_abs_error": 0.0,
        "sum_sq_error": 0.0,
    }


def update_profile_accumulator(
    accumulator: dict[str, float],
    target: torch.Tensor,
    prediction: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Accumulate profile metrics without storing every 1024-position profile."""
    valid = mask > 0
    if not bool(valid.any()):
        return
    target_values = target[valid].detach().to(torch.float64)
    prediction_values = prediction[valid].detach().to(torch.float64)
    difference = prediction_values - target_values
    accumulator["n_values"] += float(target_values.numel())
    accumulator["sum_target"] += float(target_values.sum().cpu())
    accumulator["sum_prediction"] += float(prediction_values.sum().cpu())
    accumulator["sum_target_sq"] += float((target_values * target_values).sum().cpu())
    accumulator["sum_prediction_sq"] += float(
        (prediction_values * prediction_values).sum().cpu()
    )
    accumulator["sum_product"] += float((target_values * prediction_values).sum().cpu())
    accumulator["sum_abs_error"] += float(difference.abs().sum().cpu())
    accumulator["sum_sq_error"] += float((difference * difference).sum().cpu())


def finalize_profile_accumulator(accumulator: dict[str, float]) -> dict[str, float]:
    n = accumulator["n_values"]
    if n <= 0:
        return {
            "n_values": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson": float("nan"),
        }
    mean_target = accumulator["sum_target"] / n
    mean_prediction = accumulator["sum_prediction"] / n
    covariance = accumulator["sum_product"] / n - mean_target * mean_prediction
    variance_target = accumulator["sum_target_sq"] / n - mean_target * mean_target
    variance_prediction = (
        accumulator["sum_prediction_sq"] / n - mean_prediction * mean_prediction
    )
    denominator = math.sqrt(max(0.0, variance_target) * max(0.0, variance_prediction))
    pearson = covariance / denominator if denominator > 0 else float("nan")
    return {
        "n_values": float(n),
        "mae": accumulator["sum_abs_error"] / n,
        "rmse": math.sqrt(accumulator["sum_sq_error"] / n),
        "pearson": float(pearson),
    }


def validation_selection_score(
    result: EpochResult,
    task_mode: TaskMode,
    objective: CheckpointObjective,
) -> tuple[float, str]:
    """Return a score where larger is always better for checkpoint selection."""
    resolved = objective
    if resolved == "auto":
        resolved = (
            "regression-profile-pearson"
            if task_mode == "regression"
            else "classification-auprc"
        )

    if resolved == "classification-auprc":
        score = result.classification["auprc"]
    elif resolved == "regression-loss":
        score = -result.losses["regression"]
    elif resolved == "regression-profile-mae":
        score = -result.profile_regression["mae"]
    elif resolved == "regression-profile-pearson":
        score = result.profile_regression["pearson"]
    else:  # pragma: no cover - argparse restricts choices
        raise ValueError(f"Unknown checkpoint objective: {resolved}")

    if not math.isfinite(score):
        # The regression loss is normally finite from the first epoch and is a
        # robust fallback when a correlation is initially undefined.
        fallback = result.losses["regression"]
        if math.isfinite(fallback):
            return -fallback, f"{resolved} (fallback: regression-loss)"
        return -result.losses["total"], f"{resolved} (fallback: total-loss)"
    return float(score), resolved

