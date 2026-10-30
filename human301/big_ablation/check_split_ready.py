#!/usr/bin/env python3
"""Read-only Human301 split check and optional CPU batch check.

Run from big_ablation with its at_env. Imports the project's current modules.
No sbatch calls, experiment runs, checkpoints or result files are created.
"""
from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import sys
import traceback
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


class NpyMember:
    def __init__(self, archive, key):
        self.handle = archive.open(key + ".npy")
        version = np.lib.format.read_magic(self.handle)
        if version == (1, 0):
            reader = np.lib.format.read_array_header_1_0
        elif version == (2, 0):
            reader = np.lib.format.read_array_header_2_0
        else:
            raise ValueError(f"Unsupported NPY header version: {version}")
        self.shape, fortran, self.dtype = reader(self.handle)
        require(not fortran and not self.dtype.hasobject,
                f"{key}: expected a C-order array without objects")
        self.offset = self.handle.tell()
        self.row_bytes = int(np.prod(self.shape[1:])) * self.dtype.itemsize
        expected = self.offset + int(np.prod(self.shape)) * self.dtype.itemsize
        require(archive.getinfo(key + ".npy").file_size == expected,
                f"{key}: array header and member size disagree")

    def read(self, count):
        size = count * self.row_bytes
        buffer = self.handle.read(size)
        require(len(buffer) == size, "Truncated NPY member")
        return np.frombuffer(buffer, dtype=self.dtype).reshape((count,) + self.shape[1:])

    def rows(self, indices):
        result = []
        for index in indices:
            self.handle.seek(self.offset + int(index) * self.row_bytes)
            result.append(self.read(1)[0].copy())
        return np.asarray(result, dtype=self.dtype)

    def close(self):
        self.handle.close()


def scan_cache(path, source, variant, definitions, canonical_chrom, negatives):
    """Scan complete labels/split/chromosome arrays with bounded memory."""
    names = ("train", "validation", "test")
    with zipfile.ZipFile(path) as archive, contextlib.ExitStack() as stack:
        members = {}
        for key in ("sequences", "labels", "split", "chromosomes", "starts", "ends", "sample_ids"):
            members[key] = NpyMember(archive, key)
            stack.callback(members[key].close)
        n = members["sequences"].shape[0]
        require(n > 0 and members["sequences"].shape == (n, 301), "Expected DNA shape [N,301]")
        require(members["sequences"].dtype == np.uint8, "Expected uint8 DNA")
        for key in members.keys() - {"sequences"}:
            require(members[key].shape == (n,), f"{key}: incorrect shape")
        with archive.open("summary_json.npy") as handle:
            summary = json.loads(str(np.load(handle, allow_pickle=False).item()))
        require(summary.get("split_variant") == variant, "Wrong split_variant in cache")
        require(summary.get("split_definition") == definitions[variant], "Wrong split definition")
        require(Path(summary["parent_cache"]["path"]).resolve() == source.resolve(), "Wrong parent cache path")
        counts = np.zeros((3, 2), dtype=np.int64)
        chrom_sets = [set() for _ in names]
        selected = {(s, y): [] for s in range(3) for y in (0, 1)}
        wanted = {(s, y): (negatives if s == 0 and y == 0 else 1)
                  for s in range(3) for y in (0, 1)}
        expected = {str(chrom): code for code, name in enumerate(names)
                    for chrom in definitions[variant][name]}
        for offset in range(0, n, 65536):
            size = min(65536, n - offset)
            labels = members["labels"].read(size)
            split = members["split"].read(size)
            chroms = members["chromosomes"].read(size)
            require(np.isin(labels, (0, 1)).all(), "Invalid labels")
            require(np.isin(split, (0, 1, 2)).all(), "Invalid split codes")
            for chrom in np.unique(chroms):
                normalized = canonical_chrom(chrom)
                require(normalized in expected, f"Undefined chromosome {chrom}")
                code = expected[normalized]
                require(np.all(split[chroms == chrom] == code),
                        f"Chromosome {chrom} assigned to the wrong split")
                chrom_sets[code].add(str(chrom))
            for code in range(3):
                for label in (0, 1):
                    mask = (split == code) & (labels == label)
                    counts[code, label] += np.count_nonzero(mask)
                    needed = wanted[code, label] - len(selected[code, label])
                    if needed > 0:
                        selected[code, label].extend((offset + np.flatnonzero(mask)[:needed]).tolist())
        for code, name in enumerate(names):
            stats = summary["splits"][name]
            require(counts[code].min() > 0, f"{name}: missing a class")
            require(int(stats["positive"]) == int(counts[code, 1]) and
                    int(stats["negative"]) == int(counts[code, 0]), f"{name}: summary counts disagree")
            require(set(stats["chromosomes"]) == chrom_sets[code], f"{name}: summary chromosomes disagree")
        require(all(not chrom_sets[i] & chrom_sets[j] for i in range(3) for j in range(i)),
                "Chromosomes overlap across splits")
        indices = np.asarray(sorted(i for group in selected.values() for i in group), dtype=np.int64)
        return summary, indices


def read_sample(path, indices):
    with zipfile.ZipFile(path) as archive, contextlib.ExitStack() as stack:
        result = {}
        for key in ("sequences", "labels", "split", "chromosomes", "starts", "ends", "sample_ids"):
            member = NpyMember(archive, key)
            stack.callback(member.close)
            result[key] = member.rows(indices)
        return result


def cpu_batch(config, config_path, row, target, indices, data, training, torch):
    """One tiny optimizer step, plus validation/test views, entirely in RAM."""
    cache = read_sample(target, indices)
    source_sample = read_sample(row["cache"], indices)
    for key in cache.keys() - {"split"}:
        require(np.array_equal(cache[key], source_sample[key]), f"Sample changed versus original: {key}")
    require(np.all(cache["sequences"] <= 4), "Invalid sampled DNA codes")
    context = SimpleNamespace(
        config=config, config_path=config_path, root=Path.cwd(), run_dir=Path.cwd() / "plans",
        tf=row["tf"], train_assay=row["assay"], seed=config["seeds"][0],
        cache_paths={row["assay"]: target}, artifacts={},
        recorder=SimpleNamespace(update=lambda **kwargs: None), device="cpu", workers=0,
    )
    training.invoke(config["data"], context)
    data.seed_everything(context.seed)
    model = training.invoke(config["predictor"], context).cpu()
    params = config["fitting"]["params"]
    defaults = vars(training.training_namespace())
    settings = {**defaults, **params}
    train = data.TFBindingDataset(cache, np.flatnonzero(cache["split"] == 0),
                                  reverse_complement_probability=settings["reverse_complement_probability"],
                                  seed=context.seed, training=True)
    sampler = data.CyclicBalancedNegativeSampler(train, context.seed,
        negatives_per_positive=settings["negatives_per_positive"], batch_size=settings["batch_size"])
    loader = data.make_loader(train, settings["batch_size"], False, 0, torch.device("cpu"), sampler)
    batch = next(iter(loader))
    require(batch["sequence"].shape[-1] == 256, "Training crop is not 256 bp")
    model.train()
    output = model(batch["sequence"], batch["center_index"])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(settings["negatives_per_positive"])))
    loss = criterion(output["classification_logit"], batch["label"])
    require(bool(torch.isfinite(loss)), "Non-finite classification loss")
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
        betas=(settings["adam_beta1"], settings["adam_beta2"]), eps=settings["adam_eps"],
        weight_decay=settings["weight_decay"])
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    require(grads and all(bool(torch.isfinite(g).all()) for g in grads), "Missing/non-finite gradients")
    torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"])
    optimizer.step()
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), "Non-finite model parameters")
    model.eval()
    with torch.inference_mode():
        for code in (1, 2):
            dataset = data.TFBindingDataset(cache, np.flatnonzero(cache["split"] == code))
            batch = next(iter(data.make_loader(dataset, 2, False, 0, torch.device("cpu"))))
            require(batch["sequence"].shape[-1] == 301, "Evaluation input is not 301 bp")
            result = training.classification_forward(model, batch["sequence"], batch["center_index"],
                                                       **config["inference"]["params"])
            probabilities = result["classification_probability"]
            require(bool(torch.isfinite(probabilities).all()) and
                    bool(((probabilities >= 0) & (probabilities <= 1)).all()), "Invalid evaluation scores")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("configs/caches.csv"))
    parser.add_argument("--tf", nargs="+", help="Optional TF subset")
    parser.add_argument("--batch-caches", type=int, default=1,
                        help="Smallest prepared caches to CPU-batch-test; default 1; 0 skips batch tests")
    args = parser.parse_args()
    require(args.batch_caches >= 0, "--batch-caches must be nonnegative")
    sys.path.insert(0, str(Path.cwd()))
    import torch
    import data
    import training
    torch.set_num_threads(1)
    config_path = args.config.resolve()
    config = training.load_config(config_path)
    variant = data.config_split_variant(config)
    require(variant in ("split_v2", "split_v3"), "Select a new-split config")
    require(config["workflow"]["factory"].split(":")[-1] == "supervised", "This checker expects supervised")
    require(config["fitting"] and config["fitting"]["factory"].split(":")[-1] == "baseline", "Expected baseline fitting")
    require(config["inference"]["factory"].split(":")[-1] == "fixed_views", "Expected fixed_views inference")
    for key in ("workflow", "data", "predictor", "fitting", "inference"):
        fn = training.factory(config[key])
        positional = (None, None) if key == "fitting" else ((None, None, None) if key == "inference" else (None,))
        inspect.signature(fn).bind(*positional, **config[key]["params"])
    fitting = config["fitting"]["params"]
    defaults = vars(training.training_namespace())
    require(not set(fitting) - set(defaults), "Unknown fitting parameters")
    k = fitting.get("negatives_per_positive", defaults["negatives_per_positive"])
    batch_size = fitting.get("batch_size", defaults["batch_size"])
    require(type(k) is int and k > 0 and batch_size >= k + 1 and batch_size % (k + 1) == 0,
            "Invalid batch size / negative sampling ratio")
    reference = config_path.parent / (config["reference"] + ".json")
    require(reference.is_file(), f"Missing Basic/reference config: {reference}")
    require(data.config_split_variant(training.load_config(reference)) == variant,
            "Reference config uses a different split")
    rows = data.read_manifest(args.manifest)
    manifest_tfs = {r["tf"] for r in rows}
    if args.tf:
        require(not set(args.tf) - {r["tf"] for r in rows}, "Requested TF absent from manifest")
        rows = [r for r in rows if r["tf"] in args.tf]
    profile = config.get("task_profile", "long")
    if "tfs" in config:
        selected_tfs = set(config["tfs"])
        selected_pairs = None
    else:
        filename = "training_priority_short.csv" if profile == "short" else "training_priority.csv"
        selected_pairs = data.read_priority(Path.cwd() / "configs" / filename)
        selected_tfs = {tf for tf, assay in selected_pairs}
    require(not selected_tfs - manifest_tfs,
            f"Config/priority list contains TFs absent from manifest: {sorted(selected_tfs - manifest_tfs)}")
    print(f"CONFIG OK: {config['name']} | {variant} | seeds={config['seeds']}", flush=True)
    print(f"TF selection from config/priority list: {len(selected_tfs)} TFs", flush=True)
    eligible = [r for r in rows if r["tf"] in selected_tfs and
                r["assay"] in set(config["train_assays"]) | set(config["test_assays"])]
    prepared = []
    needed = set(config["train_assays"]) | set(config["test_assays"])
    for row in eligible:
        target = data.split_cache_path(row["cache"], variant)
        if not target.is_file() or not target.with_suffix(".summary.json").is_file():
            continue
        summary = data.cache_summary(target)
        require(summary.get("split_variant") == variant, f"Wrong variant: {target}")
        prepared.append((row, target, sum(s["total"] for s in summary["splits"].values())))
    require(prepared, "No completed selected cache yet; rerun after preparation advances")
    ready_pairs = {(row["tf"], row["assay"]) for row, target, n in prepared}
    ready_tfs = sorted(tf for tf in {r["tf"] for r in eligible}
                       if all((tf, a) in ready_pairs for a in needed))
    print(f"Prepared: {len(prepared)}/{len(eligible)} selected TF/assay caches", flush=True)
    print(f"All required assays prepared: {len(ready_tfs)} TFs: {', '.join(ready_tfs) or '(none yet)'}", flush=True)
    for tf in sorted({tf for tf, assay in ready_pairs} - set(ready_tfs)):
        missing = sorted(assay for assay in needed if (tf, assay) not in ready_pairs)
        print(f"WAIT {tf}: still needs {', '.join(missing)}", flush=True)
    if selected_pairs is not None:
        planned = sum((tf, assay) in selected_pairs and assay in config["train_assays"]
                      for tf in ready_tfs for assay in needed) * len(config["seeds"])
    else:
        planned = len(ready_tfs) * len(config["train_assays"]) * len(config["seeds"])
    print(f"Ready source-training tasks with all target assays: {planned}", flush=True)
    batch_checked = 0
    for row, target, n in sorted(prepared, key=lambda entry: entry[2]):
        print(f"CHECK {row['tf']}/{row['assay']}: scanning labels, splits and chromosomes ...", flush=True)
        summary, indices = scan_cache(target, row["cache"], variant,
                                      data.SPLIT_DEFINITIONS, data.canonical_chrom, k)
        print(f"  OK: {n:,} rows; chromosome assignment and class counts match", flush=True)
        audit = {key: value for key, value in summary.get("sequence_audit", {}).items()
                 if "shared_sequences_or_RC" in key and value}
        if audit:
            print(f"  NOTE: recorded identical/RC sequences across splits: {audit}", flush=True)
        if batch_checked < args.batch_caches and row["assay"] in config["train_assays"]:
            print("  CPU: real training samples, forward/backward, AdamW, validation/test views ...", flush=True)
            cpu_batch(config, config_path, row, target, indices, data, training, torch)
            batch_checked += 1
            print("  CPU BATCH OK (temporary model discarded)", flush=True)
    print(f"PASS: {len(prepared)} completed caches checked; {batch_checked} CPU batch checks", flush=True)
    print("Scope: completed caches only. No full training, GPU/AMP, checkpoint writing or collector check.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        raise SystemExit(1)
