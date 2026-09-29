#!/usr/bin/env python3
"""Multi-task CNN/Transformer model for Arabidopsis TF binding.

The script is designed for datasets produced by the three-positive-group
version of build_tf_binding_dataset.py. Positive and negative examples are
paired within the TSS, genic, and noncoding groups during balanced sampling.
It automatically reads:

    DATASET_ROOT/
      metadata_positive_tss.tsv
      metadata_positive_genic.tsv
      metadata_positive_noncoding.tsv
      metadata_negative_tss.tsv
      metadata_negative_genic.tsv
      metadata_negative_noncoding.tsv
      positive/{tss,genic,noncoding}/<chromosome>/*.window.tsv[.gz]
      negative/<chromosome>/*.window.tsv[.gz]

Each window file is expected to contain at least the columns ``base`` and ``FE``.
The model jointly predicts:

1. a window-level TF-binding label (positive/negative), and
2. the MACS3 fold-enrichment signal, normally as a 1024-position profile.

Architecture
------------
Input [B, 4, 1024]
  -> AlphaGenome-like DNA embedder, 4 -> 64 channels, effective RF 19 bp
  -> four pool + residual down-resolution stages
  -> MetaFormer(d=1), MetaFormer(d=2), Transformer, Transformer,
     MetaFormer(d=4)
  -> four AlphaGenome-like up-resolution stages with U-Net skips
  -> two 1x1 output heads: binding-logit track and FE-regression track

Hidden activations are selectable between ReLU and GELU with --activation.

This is research code. Verify the target semantics and tune the loss weights on a
validation split before drawing biological conclusions.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Architekturvariante dieser Datei. Jede Variante ist eine eigene, unabhaengige
# Skriptdatei; die drei Schalter sind hier fest verdrahtet und werden in
# configuration.json festgehalten.
#   ARCH_ATTENTION_POSITION: "absolute" = gelernte absolute Positionseinbettung
#       auf den Bottleneck-Tokens plus nn.MultiheadAttention (Basis);
#       "relative" = keine absolute Einbettung, eigene Attention mit additivem,
#       je Kopf gelerntem Abstands-Bias (T5/Enformer-Muster, vorzeichenbehaftete
#       Abstandsklassen: exakt bis 8 Tokens, darueber logarithmisch gestaffelt).
#   ARCH_POOL_STAGES: 4 = Bottleneck mit 64 Tokens a 16 bp (Basis);
#       2 = Bottleneck mit 256 Tokens a 4 bp, Encoder-Kanaele (96, 128).
#   ARCH_POOL_TYPE: "max" = MaxPool1d (Basis); "attention" = gelerntes
#       Softmax-Pooling je Kanal (Enformer/Basenji2), Logits initial 2*Identitaet.
# ---------------------------------------------------------------------------
ARCH_ATTENTION_POSITION = "absolute"
ARCH_POOL_STAGES = 2
ARCH_POOL_TYPE = "max"
ARCH_VARIANT_NAME = "res4bp"
SCRIPT_VERSION = f"1.7.2-cyclic-positive-balanced-eval-arch-{ARCH_VARIANT_NAME}"
ARCH_ENCODER_CHANNELS = {4: (80, 96, 112, 128), 2: (96, 128)}[ARCH_POOL_STAGES]

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

import numpy as np
import torch
if torch.version.hip is not None:
    print("ROCm detected: forcing mathematical SDPA backend")
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

try:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_recall_curve,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )
except ImportError as exc:  # pragma: no cover - clear dependency error
    raise RuntimeError(
        "scikit-learn is required. Install with: python -m pip install scikit-learn"
    ) from exc

try:
    from scipy.stats import pearsonr, spearmanr
    from scipy.optimize import Bounds, LinearConstraint, milp
    from scipy import sparse
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "SciPy with scipy.optimize.milp is required (SciPy >= 1.9). "
        "Install with: python -m pip install -U scipy"
    ) from exc


# -----------------------------------------------------------------------------
# Reproducibility and utilities
# -----------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can be substantially slower, so only the CuDNN
    # settings are fixed here. Exact bitwise reproducibility still depends on
    # hardware and PyTorch version.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def canonical_chrom(name: str) -> str:
    """Map names such as Chr1, chr1 and chromosome1 to the canonical value 1."""
    value = str(name).strip()
    lower = value.lower()
    if lower.startswith("chromosome"):
        value = value[len("chromosome") :]
    elif lower.startswith("chr"):
        value = value[3:]
    value = value.strip()
    if value.isdigit():
        return str(int(value))
    aliases = {
        "mt": "M",
        "mitochondria": "M",
        "mitochondrion": "M",
        "cp": "C",
        "chloroplast": "C",
    }
    return aliases.get(value.lower(), value.upper())


def parse_chromosome_list(value: str) -> tuple[str, ...]:
    chroms = tuple(canonical_chrom(item) for item in value.split(",") if item.strip())
    if not chroms:
        raise argparse.ArgumentTypeError("At least one chromosome is required.")
    return chroms


def stable_fraction(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2**64 - 1)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


# -----------------------------------------------------------------------------
# Dataset parsing and cache
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class WindowRecord:
    sample_id: str
    label: int
    chromosome: str
    chromosome_raw: str
    window_start: int
    center_index: int
    window_length: int
    window_path: str
    positive_type: str
    negative_type: str
    # Datensaetze ab Generator 7.1 tragen um jedes Fenster ein Umfeld
    # (context_per_side Basen je Seite); die Datei hat context_length Zeilen.
    context_length: int = 0
    context_per_side: int = 0
    context_start: int = 0


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("rt", encoding="utf-8", newline="")


def resolve_metadata_files(dataset_root: Path) -> list[tuple[Path, str]]:
    """Resolve the builder's three positive and three negative metadata pools."""
    positive_paths = [
        (dataset_root / "metadata_positive_tss.tsv", "tss"),
        (dataset_root / "metadata_positive_genic.tsv", "genic"),
        (dataset_root / "metadata_positive_noncoding.tsv", "noncoding"),
    ]
    if all(path.is_file() for path, _ in positive_paths):
        selected_positives = positive_paths
    elif (dataset_root / "metadata_positive_all.tsv").is_file():
        selected_positives = [(dataset_root / "metadata_positive_all.tsv", "")]
    elif (dataset_root / "metadata_positive.tsv").is_file():
        # Alte Datensaetze enthielten ausschliesslich TSS-positive Beispiele.
        selected_positives = [(dataset_root / "metadata_positive.tsv", "tss")]
    else:
        expected = [str(path) for path, _ in positive_paths]
        expected.extend(
            [
                str(dataset_root / "metadata_positive_all.tsv"),
                str(dataset_root / "metadata_positive.tsv"),
            ]
        )
        raise FileNotFoundError(
            "No positive metadata found. Expected either all three files:\n  "
            + "\n  ".join(expected[:3])
            + "\nor one combined file:\n  "
            + "\n  ".join(expected[3:])
        )
    negative_paths = [
        (dataset_root / "metadata_negative_tss.tsv", "tss"),
        (dataset_root / "metadata_negative_genic.tsv", "genic"),
        (dataset_root / "metadata_negative_noncoding.tsv", "noncoding"),
    ]
    if all(path.is_file() for path, _ in negative_paths):
        selected_negatives = negative_paths
    elif (dataset_root / "metadata_negative_all.tsv").is_file():
        selected_negatives = [(dataset_root / "metadata_negative_all.tsv", "")]
    elif (dataset_root / "metadata_negative.tsv").is_file():
        # Backward-compatible fallback. The rows must contain negative_type or
        # sample_type so that the three sampling pools can still be recovered.
        selected_negatives = [(dataset_root / "metadata_negative.tsv", "")]
    else:
        expected = [str(path) for path, _ in negative_paths]
        expected.extend(
            [
                str(dataset_root / "metadata_negative_all.tsv"),
                str(dataset_root / "metadata_negative.tsv"),
            ]
        )
        raise FileNotFoundError(
            "No negative metadata found. Expected either all three files:\n  "
            + "\n  ".join(expected[:3])
            + "\nor one combined file:\n  "
            + "\n  ".join(expected[3:])
        )
    return [*selected_positives, *selected_negatives]


def read_metadata(dataset_root: Path) -> list[WindowRecord]:
    """Read paired TSS, genic, and noncoding positive/negative metadata."""
    records: list[WindowRecord] = []
    metadata_files = resolve_metadata_files(dataset_root)

    required = {
        "sample_id",
        "label",
        "chromosome",
        "window_start_0based",
        "center_index_0based",
        "window_length",
        "window_data_file",
    }

    for metadata_path, inferred_pool_type in metadata_files:
        with metadata_path.open("rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"No header found in {metadata_path}")
            absent = required.difference(reader.fieldnames)
            if absent:
                raise ValueError(
                    f"{metadata_path} is missing columns: {', '.join(sorted(absent))}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    label = int(row["label"])
                    window_length = int(row["window_length"])
                    center_index = int(row["center_index_0based"])
                    window_start = int(row["window_start_0based"])
                    context_per_side = int(row.get("context_per_side") or 0)
                    context_length = int(row.get("context_length") or window_length)
                    context_start = int(row.get("context_start_0based") or window_start)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid numeric metadata in {metadata_path}:{line_number}"
                    ) from exc
                if label not in (0, 1):
                    raise ValueError(
                        f"Invalid label {label} in {metadata_path}:{line_number}; expected 0/1"
                    )
                if (
                    window_length <= 0
                    or context_per_side < 0
                    or context_length != window_length + 2 * context_per_side
                    or context_start != window_start - context_per_side
                    or not 0 <= center_index < context_length
                ):
                    raise ValueError(
                        f"Invalid window geometry in {metadata_path}:{line_number}"
                    )
                rel_path = row["window_data_file"].strip()
                if not rel_path:
                    raise ValueError(
                        f"Empty window_data_file in {metadata_path}:{line_number}"
                    )
                if label == 1:
                    positive_type = str(row.get("positive_type", "")).strip().lower()
                    if not positive_type:
                        sample_type = str(row.get("sample_type", "")).strip().lower()
                        positive_type = sample_type.removeprefix("positive_")
                    if not positive_type:
                        positive_type = inferred_pool_type
                    if positive_type not in {"tss", "genic", "noncoding"}:
                        raise ValueError(
                            f"Cannot assign positive pool for {metadata_path}:{line_number}; "
                            "expected positive_type tss, genic, or noncoding"
                        )
                    negative_type = "positive"
                else:
                    positive_type = "negative"
                    negative_type = str(row.get("negative_type", "")).strip().lower()
                    if not negative_type:
                        sample_type = str(row.get("sample_type", "")).strip().lower()
                        negative_type = sample_type.removeprefix("negative_")
                    if not negative_type:
                        negative_type = inferred_pool_type
                    if negative_type not in {"tss", "genic", "noncoding"}:
                        raise ValueError(
                            f"Cannot assign negative pool for {metadata_path}:{line_number}; "
                            "expected negative_type tss, genic, or noncoding"
                        )
                records.append(
                    WindowRecord(
                        sample_id=row["sample_id"].strip(),
                        label=label,
                        chromosome=canonical_chrom(row["chromosome"]),
                        chromosome_raw=row["chromosome"].strip(),
                        window_start=window_start,
                        center_index=center_index,
                        window_length=window_length,
                        window_path=rel_path,
                        positive_type=positive_type,
                        negative_type=negative_type,
                        context_length=context_length,
                        context_per_side=context_per_side,
                        context_start=context_start,
                    )
                )

    if not records:
        raise ValueError(f"No samples found below {dataset_root}")
    lengths = {record.window_length for record in records}
    if len(lengths) != 1:
        raise ValueError(f"Mixed window lengths are unsupported: {sorted(lengths)}")
    contexts = {record.context_per_side for record in records}
    if len(contexts) != 1:
        raise ValueError(f"Mixed context sizes are unsupported: {sorted(contexts)}")
    return records


BASE_TO_CODE = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}
CODE_TO_BASE = np.array(["A", "C", "G", "T", "N"])


def read_window_file(
    path: Path,
    expected_length: int,
    fe_transform: Literal["none", "asinh"] = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Read base and FE columns from a generated window TSV or TSV.GZ."""
    sequence = np.empty(expected_length, dtype=np.uint8)
    fe = np.empty(expected_length, dtype=np.float32)
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header in window file {path}")
        needed = {"base", "FE"}
        absent = needed.difference(reader.fieldnames)
        if absent:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(absent))}")
        count = 0
        for count, row in enumerate(reader, start=1):
            if count > expected_length:
                raise ValueError(f"{path} has more than {expected_length} positions")
            base = row["base"].strip().upper()
            sequence[count - 1] = BASE_TO_CODE.get(base, 4)
            try:
                value = float(row["FE"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid FE value in {path} at row {count + 1}") from exc
            if not math.isfinite(value):
                value = 0.0
            fe[count - 1] = value
    if count != expected_length:
        raise ValueError(f"{path} has {count} positions; expected {expected_length}")
    if fe_transform == "asinh":
        fe = np.arcsinh(fe).astype(np.float32, copy=False)
    return sequence, fe


# Cache-Layout: Sequenzen/FE in Umfeldlaenge, Zuschnittlaenge und Umfeld als Skalare.
CACHE_FORMAT = "2-context"


def metadata_signature(dataset_root: Path) -> str:
    parts: list[str] = []
    for path, _ in resolve_metadata_files(dataset_root):
        stat = path.stat()
        parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def build_or_load_cache(
    dataset_root: Path,
    cache_path: Path,
    fe_transform: Literal["none", "asinh"],
    rebuild: bool,
) -> dict[str, np.ndarray]:
    records = read_metadata(dataset_root)
    signature = metadata_signature(dataset_root)

    if cache_path.is_file() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_signature = str(cached["source_signature"].item())
            cached_transform = str(cached["fe_transform"].item())
            cached_format = (
                str(cached["cache_format"].item()) if "cache_format" in cached.files else ""
            )
            if (
                cached_signature == signature
                and cached_transform == fe_transform
                and cached_format == CACHE_FORMAT
                and "negative_types" in cached.files
                and "positive_types" in cached.files
            ):
                return {key: cached[key].copy() for key in cached.files}
        print("Cache exists but is stale, uses a different FE transform or an old layout; rebuilding.")

    n_samples = len(records)
    window_length = records[0].window_length
    context_per_side = records[0].context_per_side
    context_length = records[0].context_length
    # Die Datei traegt das Umfeld; der Zuschnitt auf window_length passiert im Dataset.
    sequences = np.empty((n_samples, context_length), dtype=np.uint8)
    fe_profiles = np.empty((n_samples, context_length), dtype=np.float16)
    labels = np.empty(n_samples, dtype=np.uint8)
    chromosomes = np.empty(n_samples, dtype=f"<U{max(2, max(len(r.chromosome) for r in records))}")
    starts = np.empty(n_samples, dtype=np.int64)
    context_starts = np.empty(n_samples, dtype=np.int64)
    centers = np.empty(n_samples, dtype=np.int32)
    sample_ids = np.empty(n_samples, dtype=f"<U{max(1, max(len(r.sample_id) for r in records))}")
    negative_types = np.empty(
        n_samples,
        dtype=f"<U{max(len(r.negative_type) for r in records)}",
    )
    positive_types = np.empty(
        n_samples,
        dtype=f"<U{max(len(r.positive_type) for r in records)}",
    )

    print(f"Building cache from {n_samples:,} window files ...")
    for index, record in enumerate(records):
        path = dataset_root / record.window_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Window file listed in metadata does not exist: {path}"
            )
        seq, fe = read_window_file(path, record.context_length, fe_transform=fe_transform)
        sequences[index] = seq
        fe_profiles[index] = fe.astype(np.float16, copy=False)
        labels[index] = record.label
        chromosomes[index] = record.chromosome
        starts[index] = record.window_start
        context_starts[index] = record.context_start
        centers[index] = record.center_index
        sample_ids[index] = record.sample_id
        positive_types[index] = record.positive_type
        negative_types[index] = record.negative_type
        if (index + 1) % 1000 == 0 or index + 1 == n_samples:
            print(f"  parsed {index + 1:,}/{n_samples:,}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sequences": sequences,
        "fe_profiles": fe_profiles,
        "labels": labels,
        "chromosomes": chromosomes,
        "starts": starts,
        "context_starts": context_starts,
        "centers": centers,
        "sample_ids": sample_ids,
        "positive_types": positive_types,
        "negative_types": negative_types,
        "window_length": np.asarray(window_length),
        "context_per_side": np.asarray(context_per_side),
        "source_signature": np.asarray(signature),
        "fe_transform": np.asarray(fe_transform),
        "cache_format": np.asarray(CACHE_FORMAT),
    }
    np.savez_compressed(cache_path, **payload)
    print(f"Cache written to {cache_path}")
    return payload


def make_splits(
    cache: dict[str, np.ndarray],
    train_chromosomes: Sequence[str],
    test_chromosomes: Sequence[str],
    val_fraction: float,
    val_block_size: int,
    min_split_samples_per_group: int = 2,
    balanced_evaluation: bool = True,
    max_eval_positive_reduction: float = 1.0,
    val_milp_time_limit: float = 120.0,
) -> dict[str, np.ndarray]:
    """Select intact genomic blocks while covering all six biological strata.

    Blocks linked by overlapping windows form one indivisible component, even
    when the overlap crosses a nominal block boundary. An integer optimization
    chooses components near the desired validation fraction for each stratum.
    """
    chromosomes = cache["chromosomes"].astype(str)
    starts = cache["starts"].astype(np.int64)
    labels = cache["labels"].astype(np.int64)
    positive_types = cache["positive_types"].astype(str)
    negative_types = cache["negative_types"].astype(str)
    stratum_names = tuple(
        f"{label}_{group}"
        for label in ("positive", "negative")
        for group in POOL_NAMES
    )
    stratum_codes = {
        name: index for index, name in enumerate(stratum_names)
    }
    strata = np.asarray([
        stratum_codes.get(
            f"positive_{positive_types[index]}"
            if labels[index] == 1 else f"negative_{negative_types[index]}",
            -1,
        )
        for index in range(labels.size)
    ], dtype=np.int64)
    train_set = {canonical_chrom(chrom) for chrom in train_chromosomes}
    test_set = {canonical_chrom(chrom) for chrom in test_chromosomes}
    overlap = train_set.intersection(test_set)
    if overlap:
        raise ValueError(f"Train and test chromosomes overlap: {sorted(overlap)}")

    train_candidates = np.flatnonzero(np.isin(chromosomes, sorted(train_set)))
    test_indices = np.flatnonzero(np.isin(chromosomes, sorted(test_set)))
    if train_candidates.size == 0:
        raise ValueError(f"No samples found for train chromosomes {sorted(train_set)}")
    if test_indices.size == 0:
        raise ValueError(f"No samples found for test chromosomes {sorted(test_set)}")

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if val_block_size < 1:
        raise ValueError("val_block_size must be positive")
    if min_split_samples_per_group < 1:
        raise ValueError("min_split_samples_per_group must be positive")
    if not 0.0 <= max_eval_positive_reduction <= 1.0:
        raise ValueError("max_eval_positive_reduction must be in [0, 1]")
    if val_milp_time_limit <= 0:
        raise ValueError("val_milp_time_limit must be positive")
    if val_fraction == 0.0:
        return {
            "train": train_candidates,
            "val": np.empty(0, dtype=np.int64),
            "test": test_indices,
        }

    candidate_strata = strata[train_candidates]
    if np.any(candidate_strata < 0):
        raise ValueError("At least one training example has an unknown biological group")
    totals = np.bincount(
        candidate_strata[candidate_strata >= 0], minlength=len(stratum_names)
    )
    missing = [
        name for name, count in zip(stratum_names, totals) if count == 0
    ]
    if missing:
        raise RuntimeError(
            "Training chromosomes have no examples for: " + ", ".join(missing)
        )
    too_small = [
        f"{name} ({count})" for name, count in zip(stratum_names, totals)
        if count < 2 * min_split_samples_per_group
    ]
    if too_small:
        raise RuntimeError(
            f"At least {min_split_samples_per_group} examples of every group "
            "are required in both training and validation. Too few: "
            + ", ".join(too_small)
        )
    # Training draws at most one copy of each negative per epoch. If negatives
    # are fewer, the sampler cycles through all positives over later epochs.
    # Validation/test randomly keep as many positives as negatives if needed.
    deficits = [
        f"{name}: {int(totals[3 + i])} negative < {int(totals[i])} positive"
        for i, name in enumerate(POOL_NAMES)
        if totals[3 + i] < totals[i]
    ]
    if deficits:
        print(
            "Training groups with fewer negatives than positives: "
            + "; ".join(deficits)
            + ". Positives will be cycled between epochs without replacement "
            "inside an epoch.",
            file=sys.stderr,
        )

    # Every chromosome/block pair is atomic. Merge neighboring blocks if
    # windows from them overlap, preventing train/validation window leakage.
    candidate_chroms = chromosomes[train_candidates]
    candidate_starts = starts[train_candidates]
    block_numbers = candidate_starts // val_block_size
    block_keys = sorted(set(zip(candidate_chroms.tolist(), block_numbers.tolist())))
    block_ids = {key: index for index, key in enumerate(block_keys)}
    parents = list(range(len(block_keys)))

    def find(block_id: int) -> int:
        while parents[block_id] != block_id:
            parents[block_id] = parents[parents[block_id]]
            block_id = parents[block_id]
        return block_id

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parents[max(root_first, root_second)] = min(root_first, root_second)

    window_length = int(cache["sequences"].shape[1])
    for chromosome in sorted(set(candidate_chroms.tolist())):
        positions = np.flatnonzero(candidate_chroms == chromosome)
        positions = positions[np.argsort(candidate_starts[positions], kind="stable")]
        previous_end = -1
        previous_block = -1
        for position in positions:
            start = int(candidate_starts[position])
            block = block_ids[(chromosome, int(block_numbers[position]))]
            if start < previous_end:
                union(block, previous_block)
            if start + window_length > previous_end:
                previous_end = start + window_length
                previous_block = block

    roots = sorted({find(index) for index in range(len(block_keys))})
    root_columns = {root: index for index, root in enumerate(roots)}
    component_of_sample = np.asarray([
        root_columns[find(block_ids[(str(chromosome), int(block))])]
        for chromosome, block in zip(candidate_chroms, block_numbers)
    ], dtype=np.int64)
    counts = np.zeros((len(stratum_names), len(roots)), dtype=np.float64)
    for stratum, component in zip(candidate_strata, component_of_sample):
        counts[int(stratum), int(component)] += 1.0

    support = np.count_nonzero(counts, axis=1)
    indivisible = [
        f"{name} ({int(support[i])} component)"
        for i, name in enumerate(stratum_names)
        if support[i] < 2
    ]
    if indivisible:
        raise RuntimeError(
            "No leak-free training/validation split: all examples of "
            + ", ".join(indivisible)
            + f" are joined at --val-block-size {val_block_size}. "
            "Smaller blocks may help only when they separate those examples; "
            "overlapping input contexts still remain together."
        )

    n_components = len(roots)
    n_strata = len(stratum_names)
    n_variables = n_components + 2 * n_strata
    target_counts = val_fraction * totals
    # Minimize the sum of absolute deviations of validation fractions from the
    # requested fraction. The last 12 continuous variables are the deviations.
    objective = np.zeros(n_variables, dtype=np.float64)
    objective[n_components:] = np.tile(1.0 / totals, 2)
    equalities = np.zeros((n_strata, n_variables), dtype=np.float64)
    equalities[:, :n_components] = counts
    for stratum in range(n_strata):
        equalities[stratum, n_components + stratum] = -1.0
        equalities[stratum, n_components + n_strata + stratum] = 1.0

    constraints = [
        LinearConstraint(sparse.csr_matrix(equalities), target_counts, target_counts),
        LinearConstraint(
            sparse.csr_matrix(np.pad(counts, ((0, 0), (0, 2 * n_strata)))),
            np.full(n_strata, min_split_samples_per_group),
            totals - min_split_samples_per_group,
        ),
    ]
    # Training balances each epoch by cycling positives if negatives are scarce.
    # Validation uses the same optional reduction limit as the evaluator.
    pad = ((0, 0), (0, 2 * n_strata))
    if balanced_evaluation and max_eval_positive_reduction < 1.0:
        val_balance = counts[3:] - (1.0 - max_eval_positive_reduction) * counts[:3]
        constraints.append(LinearConstraint(
            sparse.csr_matrix(np.pad(val_balance, pad)),
            np.zeros(3), np.full(3, np.inf),
        ))
    upper_bounds = np.concatenate((np.ones(n_components), np.full(2 * n_strata, np.inf)))
    integrality = np.concatenate((np.ones(n_components), np.zeros(2 * n_strata)))
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(np.zeros(n_variables), upper_bounds),
        constraints=constraints,
        options={"time_limit": val_milp_time_limit, "mip_rel_gap": 0.01},
    )
    if result.x is None:
        reason = "time limit reached" if result.status == 1 else (
            "constraints are infeasible" if result.status == 2 else "solver failed"
        )
        raise RuntimeError(
            f"Genomic-block validation split failed ({reason}). "
            f"{n_components} overlap-connected components; training chromosome "
            f"totals: {dict(zip(stratum_names, totals.astype(int).tolist()))}; "
            f"component support: {dict(zip(stratum_names, support.tolist()))}. "
            f"Solver: {result.message}. "
            "If infeasible, check per-group counts and block sizes; if timed out, "
            "increase --val-milp-time-limit. No individual windows were reassigned."
        )
    selected_components = np.rint(result.x[:n_components]).astype(bool)
    validation_counts = counts @ selected_components.astype(np.float64)
    if (
        np.max(np.abs(result.x[:n_components] - selected_components)) > 1e-5
        or np.any(validation_counts < min_split_samples_per_group)
        or np.any(totals - validation_counts < min_split_samples_per_group)
        or (balanced_evaluation and max_eval_positive_reduction < 1.0 and np.any(
            (counts[3:] - (1.0 - max_eval_positive_reduction) * counts[:3])
            @ selected_components < -1e-5
        ))
    ):
        raise RuntimeError("The genomic-block optimizer returned an invalid split")

    val_mask = selected_components[component_of_sample]
    train_indices = train_candidates[~val_mask]
    val_indices = train_candidates[val_mask]
    print(
        f"Validation split: {int(selected_components.sum())}/{n_components} "
        f"non-overlapping block components; "
        f"per-group counts: {dict(zip(stratum_names, validation_counts.astype(int).tolist()))}"
    )
    return {"train": train_indices, "val": val_indices, "test": test_indices}


ONE_HOT_LOOKUP = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],  # A
        [0.0, 1.0, 0.0, 0.0],  # C
        [0.0, 0.0, 1.0, 0.0],  # G
        [0.0, 0.0, 0.0, 1.0],  # T
        [0.0, 0.0, 0.0, 0.0],  # N / unknown
    ],
    dtype=torch.float32,
)


ShiftDistribution = Literal["uniform", "gaussian"]


def draw_shifts(
    rng: np.random.Generator, count: int, shift_max: int, distribution: ShiftDistribution,
) -> np.ndarray:
    """Ganzzahlige Verschiebungen in [-shift_max, shift_max]."""
    if shift_max <= 0 or count <= 0:
        return np.zeros(count, dtype=np.int64)
    if distribution == "uniform":
        return rng.integers(-shift_max, shift_max + 1, size=count, dtype=np.int64)
    if distribution == "gaussian":
        values = np.rint(rng.normal(0.0, shift_max / 2.0, size=count)).astype(np.int64)
        return np.clip(values, -shift_max, shift_max)
    raise ValueError(f"Unknown shift distribution: {distribution}")


def cache_geometry(cache: dict[str, np.ndarray]) -> tuple[int, int]:
    """(Zuschnittlaenge, Umfeld je Seite) eines geladenen Caches."""
    context_length = int(cache["sequences"].shape[1])
    if "window_length" in cache:
        window_length = int(np.asarray(cache["window_length"]).item())
        context_per_side = int(np.asarray(cache["context_per_side"]).item())
    else:
        window_length, context_per_side = context_length, 0
    if window_length + 2 * context_per_side != context_length:
        raise ValueError("Cache geometry is inconsistent with the stored sequences")
    return window_length, context_per_side


class TFBindingDataset(Dataset[dict[str, Any]]):
    """Fenster mit Umfeld; jeder Zugriff schneidet window_length Basen aus.

    Der Zuschnitt beginnt bei context_per_side + shift. Im Training wird der
    Shift je Beispiel und Epoche neu gezogen (``set_epoch`` setzt den Seed),
    fuer Validierung und Test kommt er aus ``fixed_shifts`` (oder ist 0). Das
    Klassifikations-Readout liegt immer in der Mitte des Ausschnitts, damit
    das Modell die Bindestelle innerhalb von +/- shift_max selbst finden muss.
    """

    def __init__(
        self,
        cache: dict[str, np.ndarray],
        indices: np.ndarray,
        reverse_complement_probability: float = 0.0,
        shift_max: int = 0,
        shift_distribution: ShiftDistribution = "uniform",
        fixed_shifts: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        self.sequences = cache["sequences"]
        self.fe_profiles = cache["fe_profiles"]
        self.labels = cache["labels"]
        self.chromosomes = cache["chromosomes"]
        self.starts = cache["starts"]
        self.context_starts = (
            cache["context_starts"] if "context_starts" in cache else cache["starts"]
        )
        self.centers = cache["centers"]
        self.sample_ids = cache["sample_ids"]
        self.positive_types = cache["positive_types"]
        self.negative_types = cache["negative_types"]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.reverse_complement_probability = float(reverse_complement_probability)
        if not 0.0 <= self.reverse_complement_probability <= 1.0:
            raise ValueError("reverse_complement_probability must be in [0, 1]")
        self.window_length, self.context_per_side = cache_geometry(cache)
        self.shift_max = int(shift_max)
        self.shift_distribution: ShiftDistribution = shift_distribution
        if self.shift_max < 0:
            raise ValueError("shift_max must not be negative")
        if self.shift_max > self.context_per_side:
            raise ValueError(
                f"shift_max {self.shift_max} exceeds the dataset context of "
                f"{self.context_per_side} bases per side"
            )
        if fixed_shifts is not None:
            fixed_shifts = np.asarray(fixed_shifts, dtype=np.int64)
            if fixed_shifts.shape != (self.indices.size,):
                raise ValueError("fixed_shifts must hold one shift per dataset position")
            if np.any(np.abs(fixed_shifts) > self.context_per_side):
                raise ValueError("fixed_shifts exceed the dataset context")
        self.fixed_shifts = fixed_shifts
        self.seed = int(seed)
        self.epoch = 0
        self.epoch_shifts = np.zeros(self.indices.size, dtype=np.int64)
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Zieht die Shifts dieser Epoche fuer alle Positionen (reproduzierbar)."""
        self.epoch = int(epoch)
        if self.fixed_shifts is not None:
            self.epoch_shifts = self.fixed_shifts
            return
        rng = np.random.default_rng([self.seed, self.epoch, 7])
        self.epoch_shifts = draw_shifts(
            rng, self.indices.size, self.shift_max, self.shift_distribution
        )

    def shift_summary(self) -> dict[str, Any]:
        shifts = self.epoch_shifts[: self.indices.size]
        return {
            "shift_max": self.shift_max,
            "distribution": self.shift_distribution if self.fixed_shifts is None else "fixed",
            "mean_abs_shift": float(np.mean(np.abs(shifts))) if shifts.size else 0.0,
            "max_abs_shift": int(np.max(np.abs(shifts))) if shifts.size else 0,
        }

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        shift = int(self.epoch_shifts[item])
        crop_start = self.context_per_side + shift
        crop_end = crop_start + self.window_length
        codes = torch.from_numpy(
            self.sequences[index, crop_start:crop_end].astype(np.int64, copy=False)
        )
        fe = torch.from_numpy(
            self.fe_profiles[index, crop_start:crop_end].astype(np.float32, copy=False)
        )
        # Das Readout sitzt in der Ausschnittmitte, unabhaengig vom Shift.
        center = self.window_length // 2

        if self.reverse_complement_probability > 0.0 and random.random() < self.reverse_complement_probability:
            # Code permutation A,C,G,T,N -> T,G,C,A,N followed by sequence reversal.
            complement = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)
            codes = complement[codes].flip(0)
            fe = fe.flip(0)
            center = codes.numel() - 1 - center

        one_hot = ONE_HOT_LOOKUP[codes].transpose(0, 1).contiguous()  # [4, L]
        valid_mask = codes.ne(4).to(torch.float32)
        return {
            "sequence": one_hot,
            "fe_profile": fe,
            "valid_mask": valid_mask,
            "label": torch.tensor(float(self.labels[index]), dtype=torch.float32),
            "center_index": torch.tensor(center, dtype=torch.long),
            "sample_id": str(self.sample_ids[index]),
            "positive_type": str(self.positive_types[index]),
            "negative_type": str(self.negative_types[index]),
            "chromosome": str(self.chromosomes[index]),
            # Genomischer Start des tatsaechlich verwendeten Ausschnitts.
            "window_start": int(self.context_starts[index]) + crop_start,
            "shift": shift,
        }


POOL_NAMES = ("tss", "genic", "noncoding")
NEGATIVE_POOL_NAMES = POOL_NAMES
POSITIVE_POOL_NAMES = POOL_NAMES


def negative_pool_counts(
    cache: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, int]:
    labels = cache["labels"][indices].astype(np.int64)
    negative_types = cache["negative_types"][indices].astype(str)
    return {
        name: int(np.sum((labels == 0) & (negative_types == name)))
        for name in NEGATIVE_POOL_NAMES
    }


def positive_pool_counts(
    cache: dict[str, np.ndarray], indices: np.ndarray
) -> dict[str, int]:
    labels = cache["labels"][indices].astype(np.int64)
    positive_types = cache["positive_types"][indices].astype(str)
    return {
        name: int(np.sum((labels == 1) & (positive_types == name)))
        for name in POSITIVE_POOL_NAMES
    }


def make_fixed_balanced_evaluation_split(
    cache: dict[str, np.ndarray],
    indices: np.ndarray,
    seed: int,
    split_name: str,
    max_positive_reduction: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Create a fixed 1:1 split, paired within each biological group.

    If a negative pool is smaller than its positive group, randomly select
    exactly as many positives as negatives (seeded) and print a warning. The
    optional max_positive_reduction can impose a stricter limit; its default
    of 1.0 accepts every nonempty pool. The summary records group counts.
    """
    if not 0.0 <= max_positive_reduction <= 1.0:
        raise ValueError("max_positive_reduction must be in [0, 1]")
    indices = np.asarray(indices, dtype=np.int64)
    labels = cache["labels"][indices].astype(np.int64)
    negative_types = cache["negative_types"][indices].astype(str)
    positive_types = cache["positive_types"][indices].astype(str)
    if not np.any(labels == 1):
        raise ValueError(f"{split_name} contains no positive examples")

    positive_pools = {
        pool_name: indices[(labels == 1) & (positive_types == pool_name)]
        for pool_name in POOL_NAMES
    }
    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray] = []
    summary: dict[str, Any] = {
        "max_positive_reduction": float(max_positive_reduction),
        "groups": {},
        "positives_dropped_total": 0,
    }
    for pool_name in POOL_NAMES:
        positives = positive_pools[pool_name]
        pool = indices[(labels == 0) & (negative_types == pool_name)]
        if positives.size == 0:
            print(
                f"WARNING: {split_name}: no positives in group {pool_name!r}; "
                "this group is absent from the balanced evaluation.",
                file=sys.stderr,
            )
            summary["groups"][pool_name] = {
                "positives_available": 0,
                "negatives_available": int(pool.size),
                "pairs_used": 0,
                "positives_dropped": 0,
            }
            continue
        pairs = int(positives.size)
        dropped = 0
        if pool.size < positives.size:
            shortfall = (positives.size - pool.size) / positives.size
            if shortfall > max_positive_reduction:
                raise ValueError(
                    f"Cannot balance {split_name}: negative pool {pool_name!r} has "
                    f"{pool.size:,} samples but {positives.size:,} are required "
                    f"({shortfall:.1%} short, allowed {max_positive_reduction:.1%}). "
                    "Provide a larger negative pool or raise "
                    "--max-eval-positive-reduction."
                )
            pairs = int(pool.size)
            dropped = int(positives.size - pairs)
            positives = rng.choice(positives, size=pairs, replace=False)
            print(
                f"WARNING: {split_name}: negative pool {pool_name!r} has only "
                f"{pool.size:,} samples for {pairs + dropped:,} positives; "
                f"{dropped:,} positives ({shortfall:.1%}) were dropped at random "
                "to keep the 1:1 pairing.",
                file=sys.stderr,
            )
        selected_parts.append(positives)
        selected_parts.append(rng.choice(pool, size=pairs, replace=False))
        summary["groups"][pool_name] = {
            "positives_available": int(positive_pools[pool_name].size),
            "negatives_available": int(pool.size),
            "pairs_used": pairs,
            "positives_dropped": dropped,
        }
        summary["positives_dropped_total"] += dropped

    selected = np.concatenate(selected_parts).astype(np.int64, copy=False)
    if selected.size == 0:
        raise ValueError(
            f"Cannot evaluate {split_name}: there are no positive/negative pairs "
            "in any biological group."
        )
    return rng.permutation(selected), summary


class CyclicBalancedNegativeSampler(Sampler[int]):
    """Draw a distinct 1:1 pair per group and cycle unused samples over epochs.

    Each epoch uses min(positive_count, negative_count) examples of each class
    within each biological group. Scarce negatives are never duplicated within
    an epoch; instead the positive pool is traversed cyclically across epochs.
    """

    def __init__(self, dataset: TFBindingDataset, seed: int) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        global_indices = dataset.indices
        labels = dataset.labels[global_indices].astype(np.int64)
        positive_types = dataset.positive_types[global_indices].astype(str)
        negative_types = dataset.negative_types[global_indices].astype(str)
        self.positive_pool_positions = {
            pool_name: np.flatnonzero(
                (labels == 1) & (positive_types == pool_name)
            ).astype(np.int64)
            for pool_name in POOL_NAMES
        }
        self.negative_pool_positions = {
            pool_name: np.flatnonzero(
                (labels == 0) & (negative_types == pool_name)
            ).astype(np.int64)
            for pool_name in POOL_NAMES
        }
        self.quotas: dict[str, int] = {}
        for pool_name in POOL_NAMES:
            positives = self.positive_pool_positions[pool_name].size
            negatives = self.negative_pool_positions[pool_name].size
            if not positives or not negatives:
                raise ValueError(
                    f"Training pool {pool_name!r} needs both classes, found "
                    f"{positives:,} positives and {negatives:,} negatives."
                )
            self.quotas[pool_name] = int(min(positives, negatives))

        self.positions = {
            (label, name): pool
            for label, pools in (("positive", self.positive_pool_positions),
                                 ("negative", self.negative_pool_positions))
            for name, pool in pools.items()
        }
        self.min_epochs_for_positive_coverage = max(
            math.ceil(self.positive_pool_positions[name].size / self.quotas[name])
            for name in POOL_NAMES
        )
        for name in POOL_NAMES:
            if self.positive_pool_positions[name].size > self.quotas[name]:
                print(
                    f"Training pool {name!r}: {self.positive_pool_positions[name].size:,} "
                    f"positives, {self.negative_pool_positions[name].size:,} negatives; "
                    f"{self.quotas[name]:,} distinct pairs per epoch. "
                    "Positive examples rotate across epochs.",
                    file=sys.stderr,
                )

        self.rng = np.random.default_rng(self.seed)
        self.orders = {
            key: self.rng.permutation(positions)
            for key, positions in self.positions.items()
        }
        self.cursors = {key: 0 for key in self.positions}
        self.epoch = 0
        self.seen_positions = {key: set() for key in self.positions}
        self.last_summary: dict[str, Any] = {}

    def __len__(self) -> int:
        return 2 * sum(self.quotas.values())

    def _draw_unique(self, key: tuple[str, str], count: int) -> np.ndarray:
        selected: list[int] = []
        selected_set: set[int] = set()
        while len(selected) < count:
            if self.cursors[key] >= self.orders[key].size:
                self.orders[key] = self.rng.permutation(self.positions[key])
                self.cursors[key] = 0
            position = int(self.orders[key][self.cursors[key]])
            self.cursors[key] += 1
            # A draw that crosses a cycle boundary must still not duplicate an
            # example inside the same epoch.
            if position in selected_set:
                continue
            selected.append(position)
            selected_set.add(position)
        return np.asarray(selected, dtype=np.int64)

    def __iter__(self) -> Iterator[int]:
        positive_parts: list[np.ndarray] = []
        negative_parts: list[np.ndarray] = []
        for pool_name in POOL_NAMES:
            count = self.quotas[pool_name]
            for label, parts in (("positive", positive_parts),
                                 ("negative", negative_parts)):
                key = (label, pool_name)
                drawn = self._draw_unique(key, count)
                parts.append(drawn)
                self.seen_positions[key].update(int(value) for value in drawn)

        epoch_positions = np.concatenate(
            [*positive_parts, *negative_parts]
        ).astype(np.int64, copy=False)
        epoch_positions = self.rng.permutation(epoch_positions)
        self.epoch += 1
        pairs = sum(self.quotas.values())
        self.last_summary = {
            "epoch": self.epoch,
            "positive": pairs,
            "positive_by_group": self.quotas.copy(),
            "positive_pool_size": {
                name: int(self.positive_pool_positions[name].size)
                for name in POOL_NAMES
            },
            "positive_unique_seen": {
                name: len(self.seen_positions[("positive", name)])
                for name in POOL_NAMES
            },
            "positive_coverage": {
                name: len(self.seen_positions[("positive", name)])
                / self.positive_pool_positions[name].size
                for name in POOL_NAMES
            },
            "negative_total": pairs,
            "negative_drawn": self.quotas.copy(),
            "negative_unique_seen": {
                name: len(self.seen_positions[("negative", name)])
                for name in POOL_NAMES
            },
            "negative_pool_size": {
                name: int(self.negative_pool_positions[name].size)
                for name in POOL_NAMES
            },
            "negative_coverage": {
                name: len(self.seen_positions[("negative", name)])
                / self.negative_pool_positions[name].size
                for name in POOL_NAMES
            },
        }
        return iter(epoch_positions.tolist())


# -----------------------------------------------------------------------------
# AlphaGenome-inspired layers with selectable hidden activation
# -----------------------------------------------------------------------------


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


class DNAEmbedder(nn.Module):
    """4 -> 64 channels with a maximal effective receptive field of 19 bp."""

    def __init__(self, out_channels: int = 64, activation: ActivationName = "relu") -> None:
        super().__init__()
        self.initial = nn.Conv1d(4, out_channels, kernel_size=15, padding=7)
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
    """AlphaGenome-style up-resolution block with nearest-repeat and U-Net skip."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        activation: ActivationName = "relu",
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
        self.residual_scale = nn.Parameter(torch.tensor(0.9, dtype=torch.float32))

    def forward(self, x: torch.Tensor, unet_skip: torch.Tensor) -> torch.Tensor:
        out = self.reduce(x) + pad_or_crop_channels(x, self.out_channels)
        out = F.interpolate(out, scale_factor=2, mode="nearest") * self.residual_scale
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


class SoftmaxPooling1d(nn.Module):
    """Gelerntes Softmax-Pooling (Enformer/Basenji2) fuer [B,C,L]-Tensoren.

    Je Pooling-Fenster erhaelt jede Position ein gelerntes Logit (lineare
    Abbildung ueber die Kanaele, ohne Bias, initial 2 * Identitaet, also nahe am
    Max-Pooling); der Pool ist die softmax-gewichtete Summe der Positionen.
    """

    def __init__(self, channels: int, pool_size: int = 2, init_scale: float = 2.0) -> None:
        super().__init__()
        self.pool_size = int(pool_size)
        self.logit = nn.Linear(channels, channels, bias=False)
        with torch.no_grad():
            self.logit.weight.copy_(init_scale * torch.eye(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, length = x.shape
        if length % self.pool_size != 0:
            raise ValueError(f"Length {length} is not divisible by pool size {self.pool_size}")
        windows = x.view(batch, channels, length // self.pool_size, self.pool_size)
        # Logit je Kanal und Position aus allen Kanaelen derselben Position.
        logits = torch.einsum("bcwp,dc->bdwp", windows, self.logit.weight)
        weights = torch.softmax(logits, dim=-1)
        return (windows * weights).sum(dim=-1)


class RelativePositionBias(nn.Module):
    """Additiver Attention-Bias je Kopf aus vorzeichenbehafteten Abstandsklassen.

    T5-Schema: ``num_buckets`` Klassen, je Richtung die Haelfte; davon die
    erste Haelfte exakt (Abstand 0..7 Tokens), der Rest logarithmisch bis
    ``max_distance``. Vorzeichen erhalten die Orientierung (Query links oder
    rechts vom Key).
    """

    def __init__(self, num_heads: int, sequence_length: int, num_buckets: int = 32) -> None:
        super().__init__()
        self.num_buckets = int(num_buckets)
        self.max_distance = max(int(sequence_length), 2)
        self.bias = nn.Embedding(self.num_buckets, num_heads)
        nn.init.trunc_normal_(self.bias.weight, std=0.02)
        positions = torch.arange(sequence_length)
        relative = positions[None, :] - positions[:, None]  # key - query
        self.register_buffer(
            "buckets", self.bucket_indices(relative), persistent=False
        )

    def bucket_indices(self, relative_position: torch.Tensor) -> torch.Tensor:
        num_buckets = self.num_buckets // 2
        result = (relative_position > 0).long() * num_buckets
        distance = relative_position.abs()
        max_exact = num_buckets // 2
        is_small = distance < max_exact
        scaled = (
            torch.log(distance.float().clamp(min=1) / max_exact)
            / math.log(self.max_distance / max_exact)
            * (num_buckets - max_exact)
        )
        large = max_exact + scaled.long()
        large = torch.minimum(large, torch.full_like(large, num_buckets - 1))
        return result + torch.where(is_small, distance, large)

    def forward(self, length: int) -> torch.Tensor:
        buckets = self.buckets[:length, :length]
        return self.bias(buckets).permute(2, 0, 1).unsqueeze(0)  # [1, H, L, L]


class RelativeMultiheadSelfAttention(nn.Module):
    """Mehrkopf-Selbstattention mit additivem relativem Positions-Bias."""

    def __init__(
        self, channels: int, num_heads: int, dropout: float, sequence_length: int,
        num_buckets: int = 32,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.qkv = nn.Linear(channels, 3 * channels)
        self.output = nn.Linear(channels, channels)
        self.attention_dropout = nn.Dropout(dropout)
        self.relative_bias = RelativePositionBias(num_heads, sequence_length, num_buckets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)  # je [B, H, L, D]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores + self.relative_bias(length)
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value)  # [B, H, L, D]
        attended = attended.transpose(1, 2).reshape(batch, length, channels)
        return self.output(attended)


class TransformerBlock(nn.Module):
    """Transformer block; attention with absolute (nn.MultiheadAttention) or
    relative (additive distance bias) position handling."""

    def __init__(
        self,
        channels: int = 128,
        num_heads: int = 4,
        ffn_channels: int = 256,
        dropout: float = 0.1,
        activation: ActivationName = "relu",
        attention_position: str = "absolute",
        sequence_length: int = 64,
    ) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")
        self.attention_position = attention_position
        self.norm1 = nn.LayerNorm(channels)
        if attention_position == "relative":
            self.attention = RelativeMultiheadSelfAttention(
                channels, num_heads, dropout, sequence_length
            )
        elif attention_position == "absolute":
            self.attention = nn.MultiheadAttention(
                embed_dim=channels,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
        else:
            raise ValueError(f"Unknown attention position mode: {attention_position}")
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, ffn_channels),
            make_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(ffn_channels, channels),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B,C,L] -> [B,L,C]
        sequence = x.transpose(1, 2)
        normalized = self.norm1(sequence)
        if self.attention_position == "relative":
            attended = self.attention(normalized)
        else:
            attended, _ = self.attention(
                normalized,
                normalized,
                normalized,
                need_weights=False,
            )
        sequence = sequence + attended
        sequence = sequence + self.ffn(self.norm2(sequence))
        return sequence.transpose(1, 2)


@dataclass
class ModelConfig:
    sequence_length: int = 1024
    embed_channels: int = 64
    pool_stages: int = ARCH_POOL_STAGES
    encoder_channels: tuple[int, ...] = ARCH_ENCODER_CHANNELS
    bottleneck_length: int = 1024 // (2 ** ARCH_POOL_STAGES)
    transformer_heads: int = 4
    transformer_ffn: int = 256
    dropout: float = 0.1
    activation: ActivationName = "relu"
    classification_readout: str = "center"
    regression_nonnegative: bool = True
    attention_position: str = ARCH_ATTENTION_POSITION
    pool_type: str = ARCH_POOL_TYPE
    architecture_variant: str = ARCH_VARIANT_NAME


class ArabidopsisTFBindingModel(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        stages = int(config.pool_stages)
        factor = 2 ** stages
        if stages < 1:
            raise ValueError("pool_stages must be at least 1")
        if len(config.encoder_channels) != stages:
            raise ValueError("encoder_channels must list one width per pool stage")
        if config.sequence_length % factor != 0:
            raise ValueError(
                f"sequence_length must be divisible by {factor} for {stages} pool stages"
            )
        if config.sequence_length // factor != config.bottleneck_length:
            raise ValueError(
                f"bottleneck_length must equal sequence_length / {factor} for this architecture"
            )
        if config.pool_type not in ("max", "attention"):
            raise ValueError(f"Unknown pool type: {config.pool_type}")
        self.config = config
        c0 = config.embed_channels
        widths = [c0, *config.encoder_channels]
        bottleneck_channels = widths[-1]

        self.embedder = DNAEmbedder(c0, activation=config.activation)
        # Ein Pooling-Modul je Stufe; Max-Pooling hat keine Parameter, das
        # Softmax-Pooling braucht die Kanalzahl des gepoolten Tensors.
        self.pools = nn.ModuleList(
            [
                SoftmaxPooling1d(widths[index]) if config.pool_type == "attention"
                else nn.MaxPool1d(kernel_size=2, stride=2)
                for index in range(stages)
            ]
        )
        self.downs = nn.ModuleList(
            [
                DownresBlock(widths[index], widths[index + 1], activation=config.activation)
                for index in range(stages)
            ]
        )

        if config.attention_position == "absolute":
            self.position_embedding = nn.Parameter(
                torch.zeros(1, bottleneck_channels, config.bottleneck_length)
            )
            nn.init.trunc_normal_(self.position_embedding, std=0.02)
        else:
            self.position_embedding = None

        def transformer() -> TransformerBlock:
            return TransformerBlock(
                channels=bottleneck_channels,
                num_heads=config.transformer_heads,
                ffn_channels=config.transformer_ffn,
                dropout=config.dropout,
                activation=config.activation,
                attention_position=config.attention_position,
                sequence_length=config.bottleneck_length,
            )

        self.bottleneck = nn.ModuleList(
            [
                DilatedMetaFormerBlock(
                    bottleneck_channels, kernel_size=5, dilation=1, dropout=config.dropout,
                    activation=config.activation,
                ),
                DilatedMetaFormerBlock(
                    bottleneck_channels, kernel_size=5, dilation=2, dropout=config.dropout,
                    activation=config.activation,
                ),
                transformer(),
                transformer(),
                DilatedMetaFormerBlock(
                    bottleneck_channels, kernel_size=5, dilation=4, dropout=config.dropout,
                    activation=config.activation,
                ),
            ]
        )

        self.ups = nn.ModuleList(
            [
                UpresBlock(widths[index + 1], widths[index], activation=config.activation)
                for index in reversed(range(stages))
            ]
        )

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

        x = self.embedder(sequence)                      # [B, 64, 1024]
        skips = [x]
        for pool, down in zip(self.pools, self.downs):
            x = down(pool(x))                            # halbe Laenge je Stufe
            skips.append(x)
        skips.pop()                                      # der letzte Encoderausgang ist x selbst

        if self.position_embedding is not None:
            x = x + self.position_embedding
        for block in self.bottleneck:
            x = block(x)

        for up in self.ups:
            x = up(x, skips.pop())                       # zurueck auf [B, 64, 1024]

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


# -----------------------------------------------------------------------------
# Losses, metrics and training
# -----------------------------------------------------------------------------


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
        return ArabidopsisTFBindingModel.gather_positions(track, center_indices)
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


def run_epoch(
    model: ArabidopsisTFBindingModel,
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
                outputs = model(batch["sequence"], batch["center_index"])
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

            probabilities = torch.sigmoid(outputs["classification_logit"])
            labels_all.append(batch["label"].detach().cpu().numpy())
            probabilities_all.append(probabilities.detach().cpu().numpy())

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
    targets_np = np.concatenate(targets_all)
    predictions_np = np.concatenate(predictions_all)
    window_starts_np = np.concatenate(window_starts_all).astype(np.int64, copy=False)
    center_indices_np = np.concatenate(center_indices_all).astype(np.int64, copy=False)
    if collect_full_profiles:
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
            targets_np[positive_mask], predictions_np[positive_mask]
        ),
        regression_negative=regression_metrics_or_nan(
            targets_np[negative_mask], predictions_np[negative_mask]
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


def result_summary(result: EpochResult) -> dict[str, Any]:
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


def _seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    dataset: TFBindingDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    # Die Shifts werden je Epoche im Hauptprozess gezogen (set_epoch); mit
    # persistent_workers saehen Worker jedoch veraltete Shift-Tabellen, daher
    # keine persistenten Worker.
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        drop_last=False,
    )


def write_predictions(
    path: Path, result: EpochResult, classification_threshold: float
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    predicted_labels = (result.probabilities >= classification_threshold).astype(np.int64)
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "sample_id",
                "true_label",
                "binding_probability",
                "predicted_binary_label",
                "classification_threshold",
                "correct",
                "fe_target_scalar",
                "fe_prediction_scalar",
            ]
        )
        for sample_id, label, probability, predicted_label, target, prediction in zip(
            result.sample_ids,
            result.labels,
            result.probabilities,
            predicted_labels,
            result.regression_targets,
            result.regression_predictions,
            strict=True,
        ):
            writer.writerow(
                [
                    sample_id,
                    int(label),
                    f"{float(probability):.8g}",
                    int(predicted_label),
                    f"{float(classification_threshold):.8g}",
                    int(int(label) == int(predicted_label)),
                    f"{float(target):.8g}",
                    f"{float(prediction):.8g}",
                ]
            )


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


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def train_command(args: argparse.Namespace) -> int:
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
    seed_everything(args.seed)
    dataset_root = args.dataset_root.resolve()
    cache_path = (
        args.cache_path.resolve()
        if args.cache_path is not None
        else dataset_root / "tf_binding_cache.npz"
    )
    cache = build_or_load_cache(
        dataset_root,
        cache_path,
        fe_transform=args.fe_transform,
        rebuild=args.rebuild_cache,
    )
    sequence_length, context_per_side = cache_geometry(cache)
    if sequence_length != 1024:
        raise ValueError(
            f"This requested architecture expects 1024 bp, but the dataset contains {sequence_length} bp"
        )
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
        min_split_samples_per_group=args.min_split_samples_per_group,
        balanced_evaluation=args.balanced_evaluation,
        max_eval_positive_reduction=args.max_eval_positive_reduction,
        val_milp_time_limit=args.val_milp_time_limit,
    )
    evaluation_balance: dict[str, Any] = {}
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
            f"negative={negatives:,}, positive_pools={positive_pool_counts(cache, indices)}, "
            f"negative_pools={negative_pool_counts(cache, indices)}"
        )

    device = resolve_device(args.device)
    print(f"Device: {device}")

    model_config = ModelConfig(
        sequence_length=sequence_length,
        bottleneck_length=sequence_length // (2 ** ARCH_POOL_STAGES),
        dropout=args.dropout,
        activation=args.activation,
        classification_readout=args.classification_readout,
        regression_nonnegative=not args.allow_negative_regression_output,
    )
    model = ArabidopsisTFBindingModel(model_config).to(device)
    print(f"Hidden activation: {model_config.activation.upper()}")
    print(
        f"Architecture variant: {ARCH_VARIANT_NAME} (attention {ARCH_ATTENTION_POSITION}, "
        f"{ARCH_POOL_STAGES} pool stages -> {model_config.bottleneck_length} tokens, "
        f"{ARCH_POOL_TYPE} pooling)"
    )
    total_parameters, trainable_parameters = count_parameters(model)
    print(
        f"Parameters: {total_parameters:,} total; {trainable_parameters:,} trainable"
    )

    train_dataset = TFBindingDataset(
        cache,
        splits["train"],
        reverse_complement_probability=args.reverse_complement_probability,
        shift_max=args.train_shift_max,
        shift_distribution=args.train_shift_distribution,
        seed=args.seed,
    )
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
    train_sampler = CyclicBalancedNegativeSampler(train_dataset, seed=args.seed)
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
            f"{args.min_checkpoint_epoch} so positive pools can be traversed."
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
        val_dataset, args.batch_size, False, args.num_workers, device
    )
    test_loader = make_loader(
        test_dataset, args.batch_size, False, args.num_workers, device
    )

    train_labels = labels[splits["train"]]
    positive_count = int(train_labels.sum())
    negative_count = int(train_labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("The training split must contain both classes")
    # The sampler presents a 1:1 class distribution in every training epoch.
    # Applying the full-pool imbalance again through pos_weight would double
    # compensate the positive class.
    pos_weight = torch.tensor(1.0, dtype=torch.float32, device=device)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    epochs_without_improvement = 0

    config_payload = {
        "script_version": SCRIPT_VERSION,
        "architecture_variant": {
            "name": ARCH_VARIANT_NAME,
            "attention_position": ARCH_ATTENTION_POSITION,
            "pool_stages": ARCH_POOL_STAGES,
            "bottleneck_tokens": model_config.bottleneck_length,
            "bp_per_token": sequence_length // model_config.bottleneck_length,
            "pool_type": ARCH_POOL_TYPE,
            "encoder_channels": list(ARCH_ENCODER_CHANNELS),
            "parameters_total": total_parameters,
            "parameters_trainable": trainable_parameters,
        },
        "model": asdict(model_config),
        "training": asdict(training_config),
        "arguments": {key: value for key, value in vars(args).items() if key != "function"},
        "splits": {key: value.tolist() for key, value in splits.items()},
        "class_counts": {
            "train_positive": positive_count,
            "train_negative": negative_count,
            "train_positive_pools": positive_pool_counts(cache, splits["train"]),
            "train_negative_pools": negative_pool_counts(cache, splits["train"]),
            "samples_per_training_epoch": len(train_sampler),
            "pos_weight": float(pos_weight.item()),
        },
        "evaluation_balance": evaluation_balance,
        "window_geometry": {
            "crop_length": sequence_length,
            "context_per_side": context_per_side,
            "context_length": sequence_length + 2 * context_per_side,
            "classification_readout_position": "crop_center",
        },
        "train_shift": {
            "shift_max": args.train_shift_max,
            "distribution": args.train_shift_distribution,
            "redrawn_every_epoch": True,
        },
        "evaluation_shift": evaluation_shift,
    }
    with (output_dir / "configuration.json").open("wt", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2, default=json_default)
        handle.write("\n")

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        train_dataset.set_epoch(epoch)
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
        )
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
        )
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
            "train": result_summary(train_result),
            "val": result_summary(val_result),
            "sampling": train_sampler.last_summary,
            "train_shift": train_dataset.shift_summary(),
        }
        history.append(row)
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
        classification_threshold=classification_threshold,
        collect_full_profiles=collect_full_profiles,
    )
    print_result(
        f"TEST chr{','.join(args.test_chromosomes)}", test_result, training_config.task_mode
    )
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


def inspect_command(args: argparse.Namespace) -> int:
    dataset_root = args.dataset_root.resolve()
    cache_path = (
        args.cache_path.resolve()
        if args.cache_path is not None
        else dataset_root / "tf_binding_cache.npz"
    )
    cache = build_or_load_cache(
        dataset_root,
        cache_path,
        fe_transform=args.fe_transform,
        rebuild=args.rebuild_cache,
    )
    splits = make_splits(
        cache,
        train_chromosomes=args.train_chromosomes,
        test_chromosomes=args.test_chromosomes,
        val_fraction=args.val_fraction,
        val_block_size=args.val_block_size,
        min_split_samples_per_group=args.min_split_samples_per_group,
        val_milp_time_limit=args.val_milp_time_limit,
    )
    print(f"Cache: {cache_path}")
    print(f"Sequences: {cache['sequences'].shape}, dtype={cache['sequences'].dtype}")
    print(f"FE profiles: {cache['fe_profiles'].shape}, dtype={cache['fe_profiles'].dtype}")
    crop_length, context_per_side = cache_geometry(cache)
    print(f"Geometry: crop {crop_length} bp, context {context_per_side} bp per side")
    labels = cache["labels"].astype(np.int64)
    for split_name, indices in splits.items():
        split_labels = labels[indices]
        print(
            f"{split_name}: n={indices.size:,}, positives={int(split_labels.sum()):,}, "
            f"negatives={int(indices.size - split_labels.sum()):,}, "
            f"positive_pools={positive_pool_counts(cache, indices)}, "
            f"negative_pools={negative_pool_counts(cache, indices)}, "
            f"chromosomes={sorted(set(cache['chromosomes'][indices].astype(str)))}"
        )
    return 0


def smoke_test_command(args: argparse.Namespace) -> int:
    seed_everything(args.seed)
    config = ModelConfig(
        sequence_length=1024,
        bottleneck_length=1024 // (2 ** ARCH_POOL_STAGES),
        activation=args.activation,
        classification_readout="center",
    )
    model = ArabidopsisTFBindingModel(config)
    batch_size = args.batch_size
    sequence = torch.zeros(batch_size, 4, 1024)
    random_codes = torch.randint(0, 4, (batch_size, 1024))
    sequence.scatter_(1, random_codes[:, None, :], 1.0)
    centers = torch.full((batch_size,), 512, dtype=torch.long)
    with torch.no_grad():
        outputs = model(sequence, centers)
    assert outputs["classification_track"].shape == (batch_size, 1024)
    assert outputs["classification_logit"].shape == (batch_size,)
    assert outputs["regression_track"].shape == (batch_size, 1024)
    assert outputs["embedding"].shape == (batch_size, 64, 1024)
    total, trainable = count_parameters(model)
    print("Smoke test passed.")
    print(f"classification_track: {tuple(outputs['classification_track'].shape)}")
    print(f"regression_track:     {tuple(outputs['regression_track'].shape)}")
    print(f"final_embedding:      {tuple(outputs['embedding'].shape)}")
    print(f"parameters: {total:,} total; {trainable:,} trainable")
    return 0


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--fe-transform",
        choices=("none", "asinh"),
        default="none",
        help="Use 'none' when the supplied FE values are already asinh-transformed.",
    )
    parser.add_argument(
        "--train-chromosomes",
        type=parse_chromosome_list,
        default=("1", "2", "3", "4"),
        help="Comma-separated chromosome names.",
    )
    parser.add_argument(
        "--test-chromosomes",
        type=parse_chromosome_list,
        default=("5",),
        help="Comma-separated chromosome names.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument(
        "--val-milp-time-limit", type=float, default=120.0,
        help="Maximum seconds for genomic-block validation optimization (default: 120).",
    )
    parser.add_argument(
        "--val-block-size",
        type=int,
        default=100_000,
        help="Genomic block size used for deterministic validation splitting.",
    )
    parser.add_argument(
        "--min-split-samples-per-group",
        type=int,
        default=2,
        help=(
            "Minimum examples in both training and validation for each of the "
            "six positive/negative biological groups. Whole genomic blocks "
            "are retained; use a larger value for more reliable evaluation."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an AlphaGenome-inspired multi-task TF-binding model."
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="Parse/cache the generated dataset and show split statistics."
    )
    add_dataset_arguments(inspect_parser)
    inspect_parser.set_defaults(function=inspect_command)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the model.")
    add_dataset_arguments(train_parser)
    train_parser.add_argument("--output-dir", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=50)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--num-workers", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=3e-4)
    train_parser.add_argument("--min-learning-rate", type=float, default=1e-6)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--adam-beta1", type=float, default=0.9)
    train_parser.add_argument("--adam-beta2", type=float, default=0.999)
    train_parser.add_argument("--adam-eps", type=float, default=1e-8)
    train_parser.add_argument("--dropout", type=float, default=0.10)
    train_parser.add_argument(
        "--activation",
        choices=("relu", "gelu"),
        default="relu",
        help="Hidden activation used throughout Conv, MetaFormer, and Transformer blocks.",
    )
    train_parser.add_argument("--gradient-clip", type=float, default=1.0)
    train_parser.add_argument("--patience", type=int, default=10)
    train_parser.add_argument("--min-delta", type=float, default=1e-4)
    train_parser.add_argument(
        "--min-checkpoint-epoch",
        type=int,
        default=1,
        help=(
            "Do not select a best checkpoint or count early-stopping patience before "
            "this epoch. Increase this when the model should see several cyclic "
            "negative draws before checkpoint selection."
        ),
    )
    train_parser.add_argument("--seed", type=int, default=13)
    train_parser.add_argument("--device", default="auto")
    train_parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument(
        "--reverse-complement-probability",
        type=float,
        default=0.5,
        help=(
            "On-the-fly reverse-complement probability for training sequences; "
            "the FE target profile is reversed consistently."
        ),
    )
    train_parser.add_argument(
        "--train-shift-max",
        type=int,
        default=128,
        help=(
            "Random crop shift (bases) applied to every training window, drawn anew "
            "for each example in each epoch; the classification readout stays at the "
            "crop center, so a positive means 'binding site within +/- shift of the "
            "center'. Must not exceed the dataset's context_per_side. 0 disables."
        ),
    )
    train_parser.add_argument(
        "--train-shift-distribution",
        choices=("uniform", "gaussian"),
        default="uniform",
        help=(
            "uniform: every shift in [-max, max] equally likely (default); gaussian: "
            "sigma = max/2, clipped to [-max, max]."
        ),
    )
    train_parser.add_argument(
        "--eval-shift",
        choices=("fixed", "none"),
        default="fixed",
        help=(
            "fixed: validation and test windows get one fixed shift each, drawn once "
            "with a seed from the training distribution (default); none: centered crops "
            "(exact GC-matched windows)."
        ),
    )
    train_parser.add_argument(
        "--eval-shift-max",
        type=int,
        default=None,
        help="Shift range for --eval-shift fixed; defaults to --train-shift-max.",
    )
    train_parser.add_argument(
        "--balanced-evaluation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use fixed 1:1 validation/test subsets with equal positive/negative "
            "counts inside each of the tss, genic, and noncoding groups. Use "
            "--no-balanced-evaluation to evaluate every held-out sample."
        ),
    )
    train_parser.add_argument(
        "--max-eval-positive-reduction",
        type=float,
        default=1.0,
        help=(
            "With --balanced-evaluation: randomly keep as many positives as "
            "negatives in a group when negatives are scarce, and warn. Default "
            "1.0 permits any shortfall; a smaller value imposes a limit."
        ),
    )
    train_parser.add_argument(
        "--task-mode",
        choices=("classification", "regression", "multitask"),
        default="multitask",
        help=(
            "classification optimizes only binary BCE; regression optimizes only "
            "MACS3-FE; multitask (default) optimizes both objectives."
        ),
    )
    train_parser.add_argument(
        "--classification-readout",
        choices=("center", "max", "mean", "logsumexp"),
        default="center",
        help="The generated positive windows are summit-centered, so center is the default.",
    )
    train_parser.add_argument(
        "--regression-mode",
        choices=("profile", "center", "max", "mean"),
        default="profile",
        help="Profile uses all 1024 FE targets; scalar modes use one value per window.",
    )
    train_parser.add_argument(
        "--regression-weight",
        type=float,
        default=0.3,
        help=(
            "Weight of the regression loss in multitask mode. Values above 1 emphasize "
            "regression; ignored in regression-only mode."
        ),
    )
    train_parser.add_argument(
        "--checkpoint-objective",
        choices=(
            "auto",
            "classification-auprc",
            "regression-loss",
            "regression-profile-mae",
            "regression-profile-pearson",
        ),
        default="auto",
        help=(
            "Validation objective for best-checkpoint selection. auto uses AUPRC except "
            "in regression-only mode, where profile Pearson is maximized."
        ),
    )
    train_parser.add_argument(
        "--export-full-regression-profiles",
        action="store_true",
        help=(
            "Export complete per-base FE target and prediction arrays for the final test "
            "split as regression_profiles.npz, plus a TSV index."
        ),
    )
    train_parser.add_argument(
        "--export-bigwig",
        action="store_true",
        help=(
            "Aggregate overlapping test-window FE predictions and write an "
            "IGV-compatible BigWig track. Requires pyBigWig."
        ),
    )
    train_parser.add_argument(
        "--export-target-bigwig",
        action="store_true",
        help="Also export the observed FE targets as a comparison BigWig track.",
    )
    train_parser.add_argument(
        "--bigwig-output",
        type=Path,
        help="Optional output path for predicted FE BigWig (default: OUTPUT_DIR/predicted_fe_test.bw).",
    )
    train_parser.add_argument(
        "--target-bigwig-output",
        type=Path,
        help="Optional output path for observed FE BigWig (default: OUTPUT_DIR/observed_fe_test.bw).",
    )
    train_parser.add_argument(
        "--bigwig-aggregation",
        choices=("mean", "center-weighted", "max"),
        default="center-weighted",
        help=(
            "How overlapping window predictions are combined per genomic base. "
            "center-weighted reduces edge effects."
        ),
    )
    train_parser.add_argument(
        "--bigwig-chrom-prefix",
        default="Chr",
        help=(
            "Prefix for canonical chromosome names in BigWig, e.g. 'Chr' -> Chr5. "
            "Pass an empty string for names such as 5."
        ),
    )
    train_parser.add_argument(
        "--chrom-sizes",
        type=Path,
        help=(
            "Optional two-column chromosome sizes file. Without it, each BigWig "
            "chromosome length is inferred from the last covered prediction."
        ),
    )
    train_parser.add_argument(
        "--threshold-strategy",
        choices=("fixed", "val-mcc", "val-f1", "val-balanced-accuracy"),
        default="val-mcc",
        help="Select the final binary threshold on validation data only.",
    )
    train_parser.add_argument(
        "--classification-threshold",
        type=float,
        default=0.5,
        help="Used only when --threshold-strategy fixed.",
    )
    train_parser.add_argument("--regression-positive-only", action="store_true")
    train_parser.add_argument("--smooth-l1-beta", type=float, default=0.5)
    train_parser.add_argument(
        "--allow-negative-regression-output",
        action="store_true",
        help="Disable the final Softplus on the FE head.",
    )
    train_parser.set_defaults(function=train_command)

    smoke_parser = subparsers.add_parser(
        "smoke-test", help="Run a synthetic forward pass and shape assertions."
    )
    smoke_parser.add_argument("--batch-size", type=int, default=2)
    smoke_parser.add_argument("--seed", type=int, default=13)
    smoke_parser.add_argument(
        "--activation", choices=("relu", "gelu"), default="relu"
    )
    smoke_parser.set_defaults(function=smoke_test_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.function(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())