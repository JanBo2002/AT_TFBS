from __future__ import annotations

# Shared implementation extracted from the frozen 4.1.0 reference.
from .common import *
from .utils import *
from .losses import *
from .metrics import *
from .epoch import EpochResult

def result_summary(result: EpochResult) -> dict[str, Any]:
    if result.regression_targets.size == 0:
        return {"loss": result.losses, "classification": result.classification}
    return {
        "loss": result.losses,
        "classification": result.classification,
        "regression_scalar_all": result.regression,
        "regression_scalar_positive": result.regression_positive,
        "regression_scalar_negative": result.regression_negative,
        "regression_profile_all": result.profile_regression,
        "regression_profile_positive": result.profile_regression_positive,
        "regression_profile_negative": result.profile_regression_negative,
    }


def print_result(
    prefix: str, result: EpochResult, task_mode: TaskMode = "multitask"
) -> None:
    c = result.classification
    r = result.regression
    profile = result.profile_regression

    if task_mode == "regression":
        message = (
            f"{prefix}: loss={result.losses['total']:.4f} "
            f"FE-loss={result.losses['regression']:.4f} "
            f"FE-scalar-MAE={r['mae']:.4f} FE-scalar-Pearson={r['pearson']:.4f} "
            f"FE-profile-MAE={profile['mae']:.4f} "
            f"FE-profile-Pearson={profile['pearson']:.4f}"
        )
    else:
        message = (
            f"{prefix}: loss={result.losses['total']:.4f} "
            f"AUROC={c['auroc']:.4f} AUPRC={c['auprc']:.4f} "
            f"BalAcc={c['balanced_accuracy']:.4f} F1={c['f1']:.4f} "
            f"MCC={c['mcc']:.4f}"
        )
        if task_mode == "multitask":
            message += (
                f" FE-scalar-MAE={r['mae']:.4f} FE-scalar-Pearson={r['pearson']:.4f} "
                f"FE-profile-MAE={profile['mae']:.4f} "
                f"FE-profile-Pearson={profile['pearson']:.4f}"
            )
    print(message)


def write_predictions(
    path: Path, result: EpochResult, classification_threshold: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predicted_labels = (result.probabilities >= classification_threshold).astype(np.int64)
    has_fe = result.regression_targets.size == result.labels.size
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        columns = ["sample_id", "true_label", "binding_probability", "predicted_binary_label",
                   "classification_threshold", "correct"]
        if has_fe:
            columns += ["fe_target_scalar", "fe_prediction_scalar"]
        writer.writerow(columns)
        for i, (sample_id, label, probability, predicted_label) in enumerate(zip(
            result.sample_ids, result.labels, result.probabilities, predicted_labels, strict=True
        )):
            row = [sample_id, int(label), f"{float(probability):.8g}", int(predicted_label),
                   f"{float(classification_threshold):.8g}", int(int(label) == int(predicted_label))]
            if has_fe:
                row += [f"{float(result.regression_targets[i]):.8g}",
                        f"{float(result.regression_predictions[i]):.8g}"]
            writer.writerow(row)


def write_regression_predictions(path: Path, result: EpochResult) -> None:
    """Write one scalar FE target/prediction per test window."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "sample_id",
                "true_label",
                "fe_target_scalar",
                "fe_prediction_scalar",
                "absolute_error",
                "squared_error",
            ]
        )
        for sample_id, label, target, prediction in zip(
            result.sample_ids,
            result.labels,
            result.regression_targets,
            result.regression_predictions,
            strict=True,
        ):
            error = float(prediction) - float(target)
            writer.writerow(
                [
                    sample_id,
                    int(label),
                    f"{float(target):.8g}",
                    f"{float(prediction):.8g}",
                    f"{abs(error):.8g}",
                    f"{error * error:.8g}",
                ]
            )


def write_full_regression_profiles(
    npz_path: Path,
    index_path: Path,
    result: EpochResult,
) -> None:
    """Write complete 1024-position FE target/prediction profiles for each window."""
    if result.full_regression_predictions.size == 0:
        raise ValueError(
            "No full profiles were collected. Run the final test epoch with "
            "collect_full_profiles=True."
        )
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        sample_ids=np.asarray(result.sample_ids),
        chromosomes=np.asarray(result.chromosomes),
        window_starts=result.window_starts,
        center_indices=result.center_indices,
        genomic_centers=result.window_starts + result.center_indices,
        labels=result.labels,
        binding_probabilities=result.probabilities,
        fe_targets=result.full_regression_targets,
        fe_predictions=result.full_regression_predictions,
    )

    with index_path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "profile_index",
                "sample_id",
                "chromosome",
                "window_start_0based",
                "center_index_0based",
                "genomic_center_0based",
                "true_label",
                "binding_probability",
            ]
        )
        for index, values in enumerate(
            zip(
                result.sample_ids,
                result.chromosomes,
                result.window_starts,
                result.center_indices,
                result.labels,
                result.probabilities,
                strict=True,
            )
        ):
            sample_id, chromosome, start, center, label, probability = values
            writer.writerow(
                [
                    index,
                    sample_id,
                    chromosome,
                    int(start),
                    int(center),
                    int(start) + int(center),
                    int(label),
                    f"{float(probability):.8g}",
                ]
            )


def _read_chrom_sizes(path: Path) -> dict[str, int]:
    sizes: dict[str, int] = {}
    with path.open("rt", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 2:
                raise ValueError(
                    f"Invalid chromosome-size line {line_number} in {path}: {raw_line.rstrip()}"
                )
            try:
                size = int(fields[1])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid chromosome size at {path}:{line_number}: {fields[1]}"
                ) from exc
            if size <= 0:
                raise ValueError(
                    f"Chromosome size must be positive at {path}:{line_number}"
                )
            sizes[fields[0]] = size
    if not sizes:
        raise ValueError(f"No chromosome sizes found in {path}")
    return sizes


def _natural_chrom_key(name: str) -> tuple[int, int | str]:
    value = canonical_chrom(name)
    if value.isdigit():
        return (0, int(value))
    return (1, value)


def _output_chromosome_name(chromosome: str, prefix: str) -> str:
    canonical = canonical_chrom(chromosome)
    return f"{prefix}{canonical}" if prefix else canonical


def _lookup_chromosome_size(
    sizes: dict[str, int],
    canonical_name: str,
    output_name: str,
) -> int | None:
    candidates = [
        output_name,
        canonical_name,
        f"Chr{canonical_name}",
        f"chr{canonical_name}",
        f"chromosome{canonical_name}",
    ]
    for candidate in candidates:
        if candidate in sizes:
            return sizes[candidate]
    lower_sizes = {name.lower(): size for name, size in sizes.items()}
    for candidate in candidates:
        if candidate.lower() in lower_sizes:
            return lower_sizes[candidate.lower()]
    return None


def _aggregate_profiles_for_chromosome(
    window_starts: np.ndarray,
    profiles: np.ndarray,
    aggregation: BigWigAggregation,
) -> tuple[np.ndarray, np.ndarray]:
    """Map overlapping window profiles to one value per covered genomic base."""
    if profiles.ndim != 2 or profiles.shape[0] != window_starts.size:
        raise ValueError("Profile/window-start dimensions do not match")
    if profiles.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    profile_length = int(profiles.shape[1])
    offsets = np.arange(profile_length, dtype=np.int64)
    positions = (window_starts[:, None] + offsets[None, :]).reshape(-1)
    values = profiles.astype(np.float64, copy=False).reshape(-1)
    valid = np.isfinite(values) & (positions >= 0)
    positions = positions[valid]
    values = values[valid]
    if positions.size == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    order = np.argsort(positions, kind="mergesort")
    positions = positions[order]
    values = values[order]
    group_starts = np.r_[0, np.flatnonzero(positions[1:] != positions[:-1]) + 1]
    unique_positions = positions[group_starts]

    if aggregation == "max":
        aggregated = np.maximum.reduceat(values, group_starts)
    else:
        if aggregation == "mean":
            weights = np.ones(profile_length, dtype=np.float64)
        elif aggregation == "center-weighted":
            center = (profile_length - 1) / 2.0
            distance = np.abs(np.arange(profile_length, dtype=np.float64) - center)
            weights = 1.0 - distance / (center + 1.0)
            # Keep a small non-zero edge weight so singly covered edge bases remain visible.
            weights = np.maximum(weights, 1.0 / profile_length)
        else:  # pragma: no cover - argparse restricts choices
            raise ValueError(f"Unknown BigWig aggregation: {aggregation}")

        repeated_weights = np.broadcast_to(weights, profiles.shape).reshape(-1)[valid]
        repeated_weights = repeated_weights[order]
        weighted_sum = np.add.reduceat(values * repeated_weights, group_starts)
        weight_sum = np.add.reduceat(repeated_weights, group_starts)
        aggregated = weighted_sum / np.maximum(weight_sum, np.finfo(np.float64).eps)

    return unique_positions.astype(np.int64, copy=False), aggregated.astype(np.float32)


def write_regression_bigwig(
    path: Path,
    result: EpochResult,
    profiles: np.ndarray,
    aggregation: BigWigAggregation,
    chromosome_prefix: str,
    chrom_sizes_path: Path | None,
) -> dict[str, int]:
    """Write overlapping per-window profiles as an IGV-compatible BigWig track."""
    try:
        import pyBigWig  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "BigWig export requires pyBigWig. Install it in the active environment with: "
            "python -m pip install pyBigWig"
        ) from exc

    if profiles.size == 0:
        raise ValueError(
            "No full profiles were collected. Enable --export-bigwig or "
            "--export-full-regression-profiles for the final test pass."
        )
    if profiles.ndim != 2 or profiles.shape[0] != len(result.chromosomes):
        raise ValueError("BigWig profiles do not match the number of test windows")

    supplied_sizes = (
        _read_chrom_sizes(chrom_sizes_path.resolve())
        if chrom_sizes_path is not None
        else None
    )
    chromosome_array = np.asarray(result.chromosomes, dtype=str)
    output_data: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}

    for canonical_name in sorted(set(chromosome_array), key=_natural_chrom_key):
        mask = chromosome_array == canonical_name
        positions, values = _aggregate_profiles_for_chromosome(
            result.window_starts[mask], profiles[mask], aggregation
        )
        if positions.size == 0:
            continue

        output_name = _output_chromosome_name(canonical_name, chromosome_prefix)
        inferred_size = int(positions[-1]) + 1
        if supplied_sizes is None:
            chromosome_size = inferred_size
        else:
            found_size = _lookup_chromosome_size(
                supplied_sizes, canonical_name, output_name
            )
            if found_size is None:
                raise ValueError(
                    f"No chromosome size found for {output_name!r} in {chrom_sizes_path}"
                )
            chromosome_size = int(found_size)
            if inferred_size > chromosome_size:
                raise ValueError(
                    f"Predictions reach coordinate {inferred_size - 1} on {output_name}, "
                    f"but the supplied chromosome size is only {chromosome_size}."
                )

        keep = positions < chromosome_size
        output_data[output_name] = (positions[keep], values[keep], chromosome_size)

    if not output_data:
        raise ValueError("No covered genomic positions were available for BigWig export")

    path.parent.mkdir(parents=True, exist_ok=True)
    ordered_names = sorted(output_data, key=_natural_chrom_key)
    header = [(name, output_data[name][2]) for name in ordered_names]
    bigwig = pyBigWig.open(str(path), "w")
    try:
        bigwig.addHeader(header)
        for chromosome in ordered_names:
            positions, values, _ = output_data[chromosome]
            if positions.size == 0:
                continue
            # Fixed-step blocks are efficient, but a new block is required at every gap.
            boundaries = np.r_[
                0,
                np.flatnonzero(np.diff(positions) != 1) + 1,
                positions.size,
            ]
            for block_start, block_end in zip(boundaries[:-1], boundaries[1:], strict=True):
                # Chunk very long contiguous runs to avoid constructing huge Python lists.
                for chunk_start in range(int(block_start), int(block_end), 1_000_000):
                    chunk_end = min(chunk_start + 1_000_000, int(block_end))
                    bigwig.addEntries(
                        chromosome,
                        int(positions[chunk_start]),
                        values=values[chunk_start:chunk_end].astype(float).tolist(),
                        span=1,
                        step=1,
                    )
    finally:
        bigwig.close()

    return {
        chromosome: int(output_data[chromosome][0].size)
        for chromosome in ordered_names
    }


def write_regression_summary(path: Path, result: EpochResult, regression_mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "MACS3 FE regression result",
        "==========================",
        f"Regression mode used for training: {regression_mode}",
        "For profile mode, scalar metrics below refer to the center position.",
        "",
    ]
    groups = [
        ("Scalar metrics, all windows", result.regression),
        ("Scalar metrics, positive windows", result.regression_positive),
        ("Scalar metrics, negative windows", result.regression_negative),
        ("Profile metrics, all valid bases", result.profile_regression),
        ("Profile metrics, positive valid bases", result.profile_regression_positive),
        ("Profile metrics, negative valid bases", result.profile_regression_negative),
    ]
    for title, metrics in groups:
        lines.append(title)
        lines.append("-" * len(title))
        for key, value in metrics.items():
            if isinstance(value, float) and math.isfinite(value):
                lines.append(f"{key}: {value:.6f}")
            else:
                lines.append(f"{key}: {value}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_multitask_summary(
    path: Path,
    result: EpochResult,
    threshold_strategy: str,
    regression_mode: str,
    regression_weight: float,
) -> None:
    c = result.classification
    r = result.regression
    p = result.profile_regression
    lines = [
        "Joint TF-binding classification and MACS3 FE regression",
        "========================================================",
        f"Regression loss weight: {regression_weight}",
        f"Regression mode: {regression_mode}",
        f"Threshold strategy: {threshold_strategy}",
        "",
        "Classification",
        f"AUROC: {c['auroc']:.6f}",
        f"AUPRC: {c['auprc']:.6f}",
        f"Balanced accuracy: {c['balanced_accuracy']:.6f}",
        f"F1: {c['f1']:.6f}",
        f"MCC: {c['mcc']:.6f}",
        "",
        "FE regression (scalar readout)",
        f"MAE: {r['mae']:.6f}",
        f"RMSE: {r['rmse']:.6f}",
        f"Pearson: {r['pearson']:.6f}",
        f"Spearman: {r['spearman']:.6f}",
        "",
        "FE regression (full profile)",
        f"MAE: {p['mae']:.6f}",
        f"RMSE: {p['rmse']:.6f}",
        f"Pearson: {p['pearson']:.6f}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_confusion_matrix(path: Path, metrics: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["", "predicted_0", "predicted_1"])
        writer.writerow(
            [
                "true_0",
                int(metrics["true_negative"]),
                int(metrics["false_positive"]),
            ]
        )
        writer.writerow(
            [
                "true_1",
                int(metrics["false_negative"]),
                int(metrics["true_positive"]),
            ]
        )


def write_binary_summary(path: Path, metrics: dict[str, float], threshold_strategy: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Binary TF-binding classification result",
        "=======================================",
        f"Threshold strategy: {threshold_strategy}",
        f"Decision threshold: {metrics['threshold']:.8f}",
        f"AUROC: {metrics['auroc']:.6f}",
        f"AUPRC: {metrics['auprc']:.6f}",
        f"Accuracy: {metrics['accuracy']:.6f}",
        f"Balanced accuracy: {metrics['balanced_accuracy']:.6f}",
        f"Precision: {metrics['precision']:.6f}",
        f"Recall / sensitivity: {metrics['recall']:.6f}",
        f"Specificity: {metrics['specificity']:.6f}",
        f"F1: {metrics['f1']:.6f}",
        f"MCC: {metrics['mcc']:.6f}",
        "",
        "Confusion matrix counts",
        f"TN: {int(metrics['true_negative'])}",
        f"FP: {int(metrics['false_positive'])}",
        f"FN: {int(metrics['false_negative'])}",
        f"TP: {int(metrics['true_positive'])}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_classification_curves(
    output_dir: Path,
    result: EpochResult,
    task_mode: TaskMode,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Write ROC and precision-recall curves for the final test predictions."""
    if np.unique(result.labels).size != 2:
        raise ValueError("ROC and precision-recall curves require both classes")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Automatic ROC/PR plots require matplotlib. Install it with: "
            "python -m pip install matplotlib"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    chromosomes = ", ".join(sorted(set(result.chromosomes))) or "test split"
    diagnostic = task_mode == "regression"
    title_suffix = " (diagnostic only: untrained classification head)" if diagnostic else ""

    fpr, tpr, _ = roc_curve(result.labels, result.probabilities)
    auroc = roc_auc_score(result.labels, result.probabilities)
    roc_path = output_dir / "roc_curve.png"
    plt.figure(figsize=(6.4, 5.4))
    plt.plot(fpr, tpr, linewidth=2, label=f"AUROC = {auroc:.4f}")
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", label="Random classifier")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve - test {chromosomes}{title_suffix}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(result.labels, result.probabilities)
    auprc = average_precision_score(result.labels, result.probabilities)
    prevalence = float(np.mean(result.labels))
    pr_path = output_dir / "precision_recall_curve.png"
    plt.figure(figsize=(6.4, 5.4))
    plt.plot(recall, precision, linewidth=2, label=f"AUPRC = {auprc:.4f}")
    plt.plot(
        [0.0, 1.0],
        [prevalence, prevalence],
        linestyle="--",
        label=f"Positive prevalence = {prevalence:.4f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-recall curve - test {chromosomes}{title_suffix}")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.02)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=dpi, bbox_inches="tight")
    plt.close()

    if diagnostic:
        print(
            "WARNING: ROC/PR plots were written for a regression-only checkpoint. "
            "Its classification head was not trained, so these curves are diagnostic only."
        )
    return roc_path, pr_path

