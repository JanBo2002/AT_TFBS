from __future__ import annotations

# Shared implementation extracted from the frozen 4.1.0 reference.
from .common import *
from .utils import *
from .data import *
from .model import *
from .losses import *
from .metrics import *
from .epoch import *
from .reporting import *
from .loaders import *

def train_command(args: argparse.Namespace, *, recorder=None, model_factory=None) -> int:
    function_started = time.perf_counter()
    if not 0.0 <= args.reverse_complement_probability <= 1.0:
        raise ValueError("reverse-complement-probability must be in [0, 1]")
    if not 0.0 <= args.classification_threshold <= 1.0:
        raise ValueError("classification-threshold must be in [0, 1]")
    if args.learning_rate <= 0 or args.min_learning_rate < 0:
        raise ValueError("Learning rates must be non-negative and initial LR > 0")
    if args.min_learning_rate > args.learning_rate:
        raise ValueError("min-learning-rate cannot exceed learning-rate")
    if not 0.0 < args.adam_beta1 < 1.0 or not 0.0 < args.adam_beta2 < 1.0:
        raise ValueError("Adam beta values must lie strictly between 0 and 1")
    if args.regression_weight < 0:
        raise ValueError("regression-weight must be non-negative")
    if args.min_checkpoint_epoch < 1 or args.min_checkpoint_epoch > args.epochs:
        raise ValueError("min-checkpoint-epoch must be between 1 and epochs")
    if args.export_bigwig and args.regression_mode != "profile":
        raise ValueError("--export-bigwig requires --regression-mode profile")
    if args.export_target_bigwig and args.regression_mode != "profile":
        raise ValueError("--export-target-bigwig requires --regression-mode profile")
    if args.val_fraction <= 0:
        raise ValueError("Training requires --val-fraction > 0 for validation and threshold selection")
    if args.negatives_per_positive < 1 or args.eval_batch_size < 1:
        raise ValueError("Sampling ratio and evaluation batch size must be positive")
    if args.hard_negative_start_epoch < 2 or args.hard_negative_refresh < 1:
        raise ValueError("Hard-negative mining needs a warmup epoch and a positive refresh interval")
    seed_everything(args.seed)
    dataset_root = args.dataset_root.resolve() if args.dataset_root is not None else args.cache_path.resolve().parent
    cache_path = (
        args.cache_path.resolve()
        if args.cache_path is not None
        else dataset_root / "tf_binding_cache.npz"
    )
    if not cache_path.is_file():
        raise FileNotFoundError(f"Existing dataset cache not found: {cache_path}")
    cache = build_or_load_cache(
        dataset_root,
        cache_path,
        fe_transform=args.fe_transform,
        rebuild=args.rebuild_cache,
    )
    sequence_length, context_per_side = cache_geometry(cache)
    if sequence_length != 256:
        raise ValueError(
            f"This requested architecture expects 256 bp, but the dataset contains {sequence_length} bp"
        )
    human301 = "split" in cache
    if human301:
        if args.task_mode != "classification":
            raise ValueError("The Human301 BED cache has binary labels and no FE targets; use --task-mode classification")
        if args.export_full_regression_profiles or args.export_bigwig or args.export_target_bigwig:
            raise ValueError("The BED cache has no FE targets or trained FE head; regression exports are unavailable")
        audit = cache["summary"].get("sequence_audit", {})
        conflicts = [key for key, value in audit.items() if isinstance(value, dict)
                     and value.get("identical_or_RC_identical_sequences_with_both_labels", 0) > 0]
        if conflicts:
            raise ValueError("Conflicting sequence labels in the existing cache: " + ", ".join(conflicts))
        overlaps = [f"{key}={value}" for key, value in audit.items() if isinstance(value, int) and value > 0]
        if overlaps:
            print("WARNING: Shared sequences or reverse complements between cached splits: "
                  + ", ".join(overlaps) + ". The datasets remain unchanged.", file=sys.stderr)
        print("Window geometry: 256 bp from 301 bp; crop starts 0..45; shifts -22..+23; fresh uniform draw per access")
        print(f"Evaluation: crops {evaluation_crop_starts(301, 256, args.eval_crops)}, reverse complement={args.eval_rc}; all samples")
    else:
        if args.train_shift_max < 0:
            raise ValueError("--train-shift-max must not be negative")
        if args.eval_shift_max is None:
            args.eval_shift_max = args.train_shift_max
        if args.eval_shift_max < 0:
            raise ValueError("--eval-shift-max must not be negative")
        if args.train_shift_max > context_per_side:
            raise ValueError(
                f"--train-shift-max {args.train_shift_max} exceeds the context of this dataset "
                f"({context_per_side} bases per side); regenerate the dataset with a larger "
                "--context-per-side or lower the shift"
            )
        if args.eval_shift == "fixed" and args.eval_shift_max > context_per_side:
            raise ValueError(
                f"--eval-shift-max {args.eval_shift_max} exceeds the context of this dataset "
                f"({context_per_side} bases per side)"
            )
        print(
            f"Window geometry: crop {sequence_length} bp from {sequence_length + 2 * context_per_side} bp "
            f"context; train shift +/-{args.train_shift_max} ({args.train_shift_distribution}); "
            f"eval shift {args.eval_shift}"
            + (f" +/-{args.eval_shift_max}" if args.eval_shift == "fixed" else "")
        )

    splits = make_splits(
        cache,
        train_chromosomes=args.train_chromosomes,
        test_chromosomes=args.test_chromosomes,
        val_fraction=args.val_fraction,
        val_block_size=args.val_block_size,
        min_split_samples_per_class=args.min_split_samples_per_class,
        balanced_evaluation=args.balanced_evaluation,
        max_eval_positive_reduction=args.max_eval_positive_reduction,
        val_milp_time_limit=args.val_milp_time_limit,
    )
    if human301:
        args.train_chromosomes = tuple(sorted(set(cache["chromosomes"][splits["train"]].astype(str))))
        args.test_chromosomes = tuple(sorted(set(cache["chromosomes"][splits["test"]].astype(str))))
    evaluation_balance: dict[str, Any] = {"scope": "all_samples", "undersampled": False}
    if args.balanced_evaluation:
        # These selections are made exactly once. They remain unchanged across
        # all epochs so validation scores and the final test are comparable.
        splits["val"], evaluation_balance["validation"] = make_fixed_balanced_evaluation_split(
            cache,
            splits["val"],
            seed=args.seed + 10_001,
            split_name="validation",
            max_positive_reduction=args.max_eval_positive_reduction,
        )
        splits["test"], evaluation_balance["test"] = make_fixed_balanced_evaluation_split(
            cache,
            splits["test"],
            seed=args.seed + 20_003,
            split_name="test",
            max_positive_reduction=args.max_eval_positive_reduction,
        )
    labels = cache["labels"].astype(np.int64)
    for split_name, indices in splits.items():
        split_labels = labels[indices]
        positives = int(split_labels.sum())
        negatives = int(indices.size - positives)
        print(
            f"{split_name:>5}: n={indices.size:,}, positive={positives:,}, "
            f"negative={negatives:,}"
        )

    device = resolve_device(args.device)
    print(f"Device: {device}")
    if recorder is not None:
        split_hash = hashlib.sha256()
        for name, indices in splits.items():
            split_hash.update(name.encode())
            split_hash.update(indices.tobytes())
        recorder.update(device=str(device), split_hash=split_hash.hexdigest(),
            split_counts={name:{"positive":int(labels[indices].sum()),
                "negative":int(len(indices)-labels[indices].sum())} for name,indices in splits.items()})

    model_config = ModelConfig(
        sequence_length=sequence_length,
        bottleneck_length=sequence_length // 4,
        motif_kernels=tuple(args.motif_kernels),
        embed_channels=args.embed_channels,
        encoder_channels=tuple(args.encoder_channels),
        transformer_heads=args.heads,
        transformer_ffn=args.ffn_channels,
        dropout=args.dropout,
        activation=args.activation,
        classification_readout=args.classification_readout,
        regression_nonnegative=not args.allow_negative_regression_output,
    )
    model = (model_factory(model_config) if model_factory else HumanTFBindingModel(model_config)).to(device)
    print(f"Hidden activation: {model_config.activation.upper()}")
    total_parameters, trainable_parameters = count_parameters(model)
    if recorder is not None:
        recorder.update(trainable_parameter_count=trainable_parameters)
    print(
        f"Parameters: {total_parameters:,} total; {trainable_parameters:,} trainable"
    )

    print(f"Architecture: motif kernels {model_config.motif_kernels}; encoder 128/64/64; "
          "original five-block bottleneck at 64; decoder 64/128/256")

    train_dataset = TFBindingDataset(
        cache,
        splits["train"],
        reverse_complement_probability=args.reverse_complement_probability,
        shift_max=args.train_shift_max,
        shift_distribution=args.train_shift_distribution,
        seed=args.seed,
        training=True,
    )
    if human301:
        evaluation_shift = {
            "mode": "fixed_crop_average", "crop_starts": list(evaluation_crop_starts(301, 256, args.eval_crops)),
            "reverse_complement": args.eval_rc, "aggregation": "mean_probability",
        }
        val_dataset = TFBindingDataset(cache, splits["val"])
        test_dataset = TFBindingDataset(cache, splits["test"])
    else:
        # Val/Test bekommen je Fenster einen festen Shift (oder 0), damit die
        # Bewertung ueber Epochen und Laeufe hinweg dieselben Ausschnitte sieht.
        evaluation_shift: dict[str, Any] = {
            "mode": args.eval_shift,
            "shift_max": args.eval_shift_max if args.eval_shift == "fixed" else 0,
            "distribution": args.train_shift_distribution,
            "seed": args.seed + 30_007,
        }
        eval_rng = np.random.default_rng(evaluation_shift["seed"])
        fixed_val_shifts = draw_shifts(
            eval_rng, int(splits["val"].size), evaluation_shift["shift_max"], args.train_shift_distribution
        )
        fixed_test_shifts = draw_shifts(
            eval_rng, int(splits["test"].size), evaluation_shift["shift_max"], args.train_shift_distribution
        )
        evaluation_shift["validation_shifts"] = fixed_val_shifts.tolist()
        evaluation_shift["test_shifts"] = fixed_test_shifts.tolist()
        val_dataset = TFBindingDataset(cache, splits["val"], 0.0, fixed_shifts=fixed_val_shifts)
        test_dataset = TFBindingDataset(cache, splits["test"], 0.0, fixed_shifts=fixed_test_shifts)
    train_sampler = CyclicBalancedNegativeSampler(
        train_dataset, seed=args.seed, negatives_per_positive=args.negatives_per_positive,
        batch_size=args.batch_size, hard_negative_fraction=args.hard_negative_fraction,
    )
    required_epochs = train_sampler.min_epochs_for_positive_coverage
    if args.epochs < required_epochs:
        print(
            f"WARNING: {required_epochs} epochs are needed to expose every "
            f"training positive at least once; only {args.epochs} were requested.",
            file=sys.stderr,
        )
    if required_epochs > args.min_checkpoint_epoch:
        args.min_checkpoint_epoch = min(args.epochs, required_epochs)
        print(
            "Checkpoint selection and early stopping start at epoch "
            f"{args.min_checkpoint_epoch} so all training positives can be traversed."
        )
    train_loader = make_loader(
        train_dataset,
        args.batch_size,
        False,
        args.num_workers,
        device,
        sampler=train_sampler,
    )
    val_loader = make_loader(
        val_dataset, args.eval_batch_size, False, args.num_workers, device
    )
    test_loader = make_loader(
        test_dataset, args.eval_batch_size, False, args.num_workers, device
    )

    mining_loader = None
    if train_sampler.hard_count:
        mining_indices = splits["train"][train_sampler.pool.positions]
        mining_dataset = TFBindingDataset(cache, mining_indices)
        mining_loader = make_loader(mining_dataset, args.eval_batch_size, False, args.num_workers, device)

    train_labels = labels[splits["train"]]
    positive_count = int(train_labels.sum())
    negative_count = int(train_labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("The training split must contain both classes")
    # Weight the presented 1:k distribution, rather than the full 1:100 pool.
    pos_weight = torch.tensor(float(args.negatives_per_positive), dtype=torch.float32, device=device)
    classification_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    training_config = TrainingConfig(
        task_mode=args.task_mode,
        regression_mode=args.regression_mode,
        regression_weight=args.regression_weight,
        regression_positive_only=args.regression_positive_only,
        smooth_l1_beta=args.smooth_l1_beta,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs), eta_min=args.min_learning_rate
    )
    amp_enabled = args.amp and device.type == "cuda"

    if not amp_enabled:
        scaler = None
    elif hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=True)

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Output directory must be new or empty; existing runs are not overwritten")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    epochs_without_improvement = 0

    config_payload = {
        "script_version": SCRIPT_VERSION,
        "model": asdict(model_config),
        "parameter_count": total_parameters,
        "cache_sha256": sha256_file(cache_path),
        "dataset": cache.get("summary"),
        "training": asdict(training_config),
        "arguments": {key: value for key, value in vars(args).items() if key != "function"},
        "splits": {key: value.tolist() for key, value in splits.items()},
        "class_counts": {
            "train_positive": positive_count,
            "train_negative": negative_count,
            "sampling_scope": "global_positive_negative",
            "samples_per_training_epoch": train_sampler.samples_per_epoch,
            "negatives_per_positive": args.negatives_per_positive,
            "pos_weight": float(pos_weight.item()),
        },
        "evaluation_balance": evaluation_balance,
        "window_geometry": {
            "crop_length": sequence_length,
            "context_per_side": context_per_side,
            "context_length": int(cache["sequences"].shape[1]),
            "context_left": context_per_side,
            "context_right": int(cache["sequences"].shape[1]) - sequence_length - context_per_side,
            "classification_readout_position": "crop_center",
        },
        "train_shift": {
            "shift_max": args.train_shift_max,
            "distribution": args.train_shift_distribution,
            "redrawn_every_epoch": True,
            "redrawn_per_access": human301,
            "crop_start_range_inclusive": [0, 45] if human301 else None,
            "shift_range_inclusive": [-22, 23] if human301 else None,
        },
        "evaluation_shift": evaluation_shift,
    }
    with (output_dir / "configuration.json").open("wt", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2, default=json_default)
        handle.write("\n")

    if recorder is not None:
        recorder.update(setup_s=time.perf_counter()-function_started,
                        train_cache_sha256=config_payload["cache_sha256"])
    fit_loop_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        mining_started = time.perf_counter()
        if mining_loader is not None and epoch >= args.hard_negative_start_epoch and (epoch - args.hard_negative_start_epoch) % args.hard_negative_refresh == 0:
            mining_scores = score_training_negatives(model, mining_loader, device, args.eval_crops, args.eval_rc, amp_enabled)
            train_sampler.update_hard_negatives(mining_scores)
            print(f"Epoch {epoch:03d}: refreshed hard-negative candidates from training only")
        mining_seconds = time.perf_counter() - mining_started
        train_dataset.set_epoch(epoch)
        train_started = time.perf_counter()
        train_result = run_epoch(
            model,
            train_loader,
            device,
            classification_criterion,
            training_config,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=amp_enabled,
            gradient_clip=args.gradient_clip,
            eval_crops=args.eval_crops, eval_rc=args.eval_rc,
        )
        train_seconds = time.perf_counter() - train_started
        val_started = time.perf_counter()
        val_result = run_epoch(
            model,
            val_loader,
            device,
            classification_criterion,
            training_config,
            optimizer=None,
            scaler=None,
            use_amp=amp_enabled,
            gradient_clip=args.gradient_clip,
            eval_crops=args.eval_crops, eval_rc=args.eval_rc,
        )
        validation_seconds = time.perf_counter() - val_started
        print(f"  negatives seen={train_sampler.last_summary['negative_pool_coverage']:.1%}; "
              f"hard negatives={train_sampler.last_summary['hard_negative']:,}")
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        learning_rate_next = float(optimizer.param_groups[0]["lr"])
        elapsed = time.time() - start_time
        print_result(f"Epoch {epoch:03d} train", train_result, training_config.task_mode)
        print_result(f"Epoch {epoch:03d} val  ", val_result, training_config.task_mode)
        print(
            f"  learning_rate_used={learning_rate_used:.3e}; "
            f"learning_rate_next={learning_rate_next:.3e}; elapsed={elapsed:.1f}s"
        )

        row = {
            "epoch": epoch,
            "learning_rate_used": learning_rate_used,
            "learning_rate_next": learning_rate_next,
            "elapsed_seconds": elapsed,
            "train_seconds": train_seconds,
            "validation_seconds": validation_seconds,
            "mining_seconds": mining_seconds,
            "train": result_summary(train_result),
            "val": result_summary(val_result),
            "sampling": train_sampler.last_summary,
            "train_shift": train_dataset.shift_summary(),
        }
        history.append(row)
        if recorder is not None:
            recorder.epoch(row)
        with (output_dir / "history.json").open("wt", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2, default=json_default)
            handle.write("\n")

        score, objective_used = validation_selection_score(
            val_result,
            training_config.task_mode,
            args.checkpoint_objective,
        )
        if epoch < args.min_checkpoint_epoch:
            print(
                f"  checkpoint selection starts at epoch "
                f"{args.min_checkpoint_epoch}; current checkpoint skipped"
            )
            continue
        if score > best_score + args.min_delta:
            best_score = score
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "script_version": SCRIPT_VERSION,
                    "cache_sha256": config_payload["cache_sha256"],
                    "model_config": asdict(model_config),
                    "training_config": asdict(training_config),
                    "epoch": epoch,
                    "validation_score": score,
                    "checkpoint_objective": objective_used,
                },
                checkpoint_path,
            )
            print(
                f"  saved new best checkpoint: {checkpoint_path} "
                f"(objective={objective_used}, score={score:.6f})"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if recorder is not None:
        recorder.fit_loop_finished(time.perf_counter() - fit_loop_started, history)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    # Select the binary decision threshold exclusively on the held-out
    # validation split. The test chromosome remains untouched until final use.
    best_val_result = run_epoch(
        model,
        val_loader,
        device,
        classification_criterion,
        training_config,
        optimizer=None,
        scaler=None,
        use_amp=amp_enabled,
        gradient_clip=args.gradient_clip,
        eval_crops=args.eval_crops, eval_rc=args.eval_rc,
        classification_threshold=0.5,
    )
    if training_config.task_mode == "regression":
        classification_threshold = 0.5
        threshold_score = float("nan")
        threshold_strategy_used = "not-selected-regression-only"
        print(
            "Regression-only mode: binary threshold optimization is skipped; "
            "classification outputs are diagnostic only."
        )
    else:
        classification_threshold, threshold_score = select_classification_threshold(
            best_val_result.labels,
            best_val_result.probabilities,
            strategy=args.threshold_strategy,
            fixed_threshold=args.classification_threshold,
        )
        threshold_strategy_used = args.threshold_strategy
        print(
            f"Binary threshold: {classification_threshold:.6f} "
            f"(strategy={args.threshold_strategy}, validation_score={threshold_score:.6f})"
        )

    best_val_result.classification = classification_metrics(
        best_val_result.labels, best_val_result.probabilities, threshold=classification_threshold
    )
    write_predictions(output_dir / "validation_predictions.tsv", best_val_result, classification_threshold)
    with (output_dir / "validation_metrics.json").open("wt", encoding="utf-8") as handle:
        json.dump(result_summary(best_val_result), handle, indent=2, default=json_default)
        handle.write("\n")
    checkpoint["classification_threshold"] = classification_threshold
    checkpoint["evaluation_crop_mode"] = args.eval_crops
    checkpoint["evaluation_crop_starts"] = list(evaluation_crop_starts(301, 256, args.eval_crops))
    checkpoint["evaluation_reverse_complement"] = args.eval_rc
    checkpoint["negatives_per_positive"] = args.negatives_per_positive
    checkpoint["amp_enabled"] = amp_enabled
    torch.save(checkpoint, output_dir / "model_for_prediction.pt")
    if args.skip_test:
        print(f"Validation-only run completed; frozen model: {output_dir / 'model_for_prediction.pt'}")
        return 0

    collect_full_profiles = (
        args.export_full_regression_profiles
        or args.export_bigwig
        or args.export_target_bigwig
    )
    test_result = run_epoch(
        model,
        test_loader,
        device,
        classification_criterion,
        training_config,
        optimizer=None,
        scaler=None,
        use_amp=amp_enabled,
        gradient_clip=args.gradient_clip,
        eval_crops=args.eval_crops, eval_rc=args.eval_rc,
        classification_threshold=classification_threshold,
        collect_full_profiles=collect_full_profiles,
    )
    print_result("TEST " + ",".join(args.test_chromosomes), test_result, training_config.task_mode)
    test_summary = result_summary(test_result)
    test_summary["checkpoint_epoch"] = int(checkpoint["epoch"])
    test_summary["threshold_selection"] = {
        "strategy": threshold_strategy_used,
        "threshold": classification_threshold,
        "validation_objective_score": threshold_score,
    }
    with (output_dir / "test_metrics.json").open("wt", encoding="utf-8") as handle:
        json.dump(test_summary, handle, indent=2, default=json_default)
        handle.write("\n")
    write_predictions(
        output_dir / "binary_classification_results.tsv",
        test_result,
        classification_threshold,
    )
    # Keep the old generic filename as an identical convenience export.
    write_predictions(
        output_dir / "test_predictions.tsv",
        test_result,
        classification_threshold,
    )
    write_confusion_matrix(
        output_dir / "binary_confusion_matrix.tsv", test_result.classification
    )
    write_binary_summary(
        output_dir / "binary_classification_summary.txt",
        test_result.classification,
        threshold_strategy_used,
    )
    roc_plot_path, pr_plot_path = write_classification_curves(
        output_dir, test_result, training_config.task_mode
    )
    with (output_dir / "binary_classification_metrics.json").open(
        "wt", encoding="utf-8"
    ) as handle:
        json.dump(
            {
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "threshold_strategy": threshold_strategy_used,
                "threshold": classification_threshold,
                "validation_objective_score": threshold_score,
                "test_metrics": test_result.classification,
            },
            handle,
            indent=2,
            default=json_default,
        )
        handle.write("\n")
    if "fe_profiles" in cache:
        write_regression_predictions(
            output_dir / "regression_results.tsv", test_result
        )
        if args.export_full_regression_profiles:
            if training_config.task_mode == "classification":
                print(
                    "WARNING: Full FE profiles are being exported from a classification-only "
                    "checkpoint. The regression head received no training signal, so these FE "
                    "predictions are not biologically meaningful. Use a multitask checkpoint "
                    "for interpretable FE predictions."
                )
            write_full_regression_profiles(
                output_dir / "regression_profiles.npz",
                output_dir / "regression_profile_index.tsv",
                test_result,
            )

        if args.export_bigwig or args.export_target_bigwig:
            if training_config.task_mode == "classification":
                print(
                    "WARNING: BigWig export is using a classification-only checkpoint. "
                    "The regression track did not receive a regression training signal."
                )
            if args.export_bigwig:
                predicted_bigwig_path = (
                    args.bigwig_output.resolve()
                    if args.bigwig_output is not None
                    else output_dir / "predicted_fe_test.bw"
                )
                coverage = write_regression_bigwig(
                    predicted_bigwig_path,
                    test_result,
                    test_result.full_regression_predictions,
                    aggregation=args.bigwig_aggregation,
                    chromosome_prefix=args.bigwig_chrom_prefix,
                    chrom_sizes_path=args.chrom_sizes,
                )
                print(
                    f"Predicted FE BigWig: {predicted_bigwig_path} "
                    f"({sum(coverage.values()):,} covered bases)"
                )
            if args.export_target_bigwig:
                target_bigwig_path = (
                    args.target_bigwig_output.resolve()
                    if args.target_bigwig_output is not None
                    else output_dir / "observed_fe_test.bw"
                )
                target_coverage = write_regression_bigwig(
                    target_bigwig_path,
                    test_result,
                    test_result.full_regression_targets,
                    aggregation=args.bigwig_aggregation,
                    chromosome_prefix=args.bigwig_chrom_prefix,
                    chrom_sizes_path=args.chrom_sizes,
                )
                print(
                    f"Observed FE BigWig: {target_bigwig_path} "
                    f"({sum(target_coverage.values()):,} covered bases)"
                )
        with (output_dir / "regression_metrics.json").open(
            "wt", encoding="utf-8"
        ) as handle:
            json.dump(
                {
                    "checkpoint_epoch": int(checkpoint["epoch"]),
                    "regression_mode": training_config.regression_mode,
                    "regression_weight": training_config.regression_weight,
                    "scalar_all": test_result.regression,
                    "scalar_positive": test_result.regression_positive,
                    "scalar_negative": test_result.regression_negative,
                    "profile_all": test_result.profile_regression,
                    "profile_positive": test_result.profile_regression_positive,
                    "profile_negative": test_result.profile_regression_negative,
                },
                handle,
                indent=2,
                default=json_default,
            )
            handle.write("\n")
        write_regression_summary(
            output_dir / "regression_summary.txt",
            test_result,
            training_config.regression_mode,
        )
        write_multitask_summary(
            output_dir / "multitask_summary.txt",
            test_result,
            threshold_strategy_used,
            training_config.regression_mode,
            training_config.regression_weight,
        )
    print(f"Outputs written to {output_dir}")
    print("Binary predictions: binary_classification_results.tsv")
    print("Binary metrics: binary_classification_metrics.json")
    print(f"ROC plot: {roc_plot_path.name}")
    print(f"Precision-recall plot: {pr_plot_path.name}")
    if "fe_profiles" in cache:
        print("Regression predictions: regression_results.tsv")
        if args.export_full_regression_profiles:
            print("Full FE profiles: regression_profiles.npz")
            print("Full FE profile index: regression_profile_index.tsv")
        if args.export_bigwig:
            print("Predicted FE BigWig export completed.")
        if args.export_target_bigwig:
            print("Observed FE BigWig export completed.")
        print("Regression metrics: regression_metrics.json")
        print("Joint summary: multitask_summary.txt")
    print("Confusion matrix: binary_confusion_matrix.tsv")
    return 0

