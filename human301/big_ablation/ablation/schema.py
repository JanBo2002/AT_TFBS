"""One versioned, fixed CSV header for every experiment and workflow."""
SCHEMA_VERSION = "1.0"

METRICS = [
    "auprc", "auroc", "mcc", "f1", "accuracy", "balanced_accuracy", "precision",
    "recall", "specificity", "negative_predictive_value", "true_positive",
    "false_positive", "true_negative", "false_negative", "threshold", "brier_score",
    "log_loss", "sample_count", "positive_count", "negative_count", "pr_baseline",
]
TIMES = [
    "training_wall_s", "fit_loop_s", "train_total_s", "validation_total_s", "mining_total_s",
    "setup_s", "post_fit_s", "epoch_mean_s", "epoch_median_s", "epoch_std_s",
    "epoch_min_s", "epoch_max_s", "epoch_p95_s", "epoch_s", "train_s", "validation_s",
    "mining_s", "inference_s", "evaluation_wall_s", "total_wall_s", "samples_per_s",
]
COLUMNS = [
    "schema_version", "row_type", "experiment_id", "reference_id", "config_hash", "code_hash",
    "run_id", "model_id", "training_scope", "tf", "tf_ids_json", "train_assay", "test_assay",
    "direction", "seed", "epoch", "evaluation_id", "status", "included_in_summary",
    "summary_level", "summary_source", "statistic", "aggregation_unit", "n_records",
    "n_runs", "n_tfs", "n_seeds", "started_at", "finished_at", "hostname", "job_id",
    "array_task_id", "device", "device_name", "gpu_count", "torch_version", "python_version",
    "amp_enabled", "gpu_peak_memory_bytes", "parameter_count", "trainable_parameter_count",
    "input_length", "kernels_json", "channels_per_branch_json", "model_config_json",
    "training_config_json", "inference_config_json", "epochs_requested", "epochs_run",
    "best_epoch", "early_stopped", "learning_rate_used", "learning_rate_next", "batch_size",
    "eval_batch_size", "num_workers", "negatives_per_positive", "pos_weight", "weight_decay",
    "learning_rate_initial", "patience", "min_checkpoint_epoch", "checkpoint_objective",
    "checkpoint_path", "checkpoint_sha256", "threshold_strategy", "validation_selection_score",
    "train_cache_path", "train_cache_sha256", "test_cache_path", "test_cache_sha256", "split_hash",
    "train_positive", "train_negative", "val_positive", "val_negative", "test_positive",
    "test_negative", "samples_per_training_epoch", "sampling_hard_count", "sampling_cyclic_count",
    "negative_pool_coverage", "evaluation_crops_json", "evaluation_rc", "forward_views",
    "train_loss", "val_loss", "classification_loss", "regression_loss", "total_loss",
] + METRICS + [f"train_{m}" for m in METRICS] + [f"val_{m}" for m in METRICS] + TIMES + [
    "delta_auprc", "ratio_auprc", "log2_ratio_auprc", "win_fraction", "paired_reference_run_id",
    "metric_name", "metric_value", "metric_unit", "analysis_name", "error", "extra_json",
]
assert len(COLUMNS) == len(set(COLUMNS))

# Fields whose sum is meaningful. Performance metrics are never summed.
SUM_FIELDS = set(TIMES) - {"epoch_mean_s","epoch_median_s","epoch_std_s","epoch_min_s",
    "epoch_max_s","epoch_p95_s","samples_per_s"}
SUM_FIELDS |= {"epochs_run", "sample_count", "positive_count", "negative_count",
    "true_positive", "false_positive", "true_negative", "false_negative"}

NUMERIC = set(METRICS + TIMES + [f"train_{m}" for m in METRICS] + [f"val_{m}" for m in METRICS]) | {
    "parameter_count", "trainable_parameter_count", "epochs_requested", "epochs_run", "best_epoch",
    "gpu_peak_memory_bytes", "train_loss", "val_loss", "classification_loss", "regression_loss",
    "total_loss", "delta_auprc", "ratio_auprc", "log2_ratio_auprc", "win_fraction", "metric_value",
}
