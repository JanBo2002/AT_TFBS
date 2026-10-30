"""Shared caches, datasets, sampling, augmentation and data loaders."""
from __future__ import annotations

import argparse
import bisect
import csv
import copy
import fcntl
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, get_worker_info

from sklearn.metrics import average_precision_score, matthews_corrcoef, roc_auc_score
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy import sparse

SCRIPT_VERSION = '5.4.1-split-v2-v3'


# Sequence caches and preparation
# -------------------------------

STORED_LENGTH = 301


BASES = "ACGTN"


ENCODE = np.full(256, 255, dtype=np.uint8)


for _i, _b in enumerate(BASES):
    ENCODE[ord(_b)] = ENCODE[ord(_b.lower())] = _i


COMPLEMENT = np.array([3, 2, 1, 0, 4], dtype=np.uint8)


SPLIT_NAMES = ("train", "validation", "test")


@dataclass(frozen=True)
class BedRecord:
    chrom: str
    start: int
    end: int
    center: int

    @property
    def key(self):
        return self.chrom, self.start, self.end


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_bed(path, expected_length=STORED_LENGTH):
    records, seen = [], set()
    with Path(path).open() as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line or line.startswith(("#", "track ", "browser ")):
                continue
            cols = line.split()
            if len(cols) < 3:
                raise ValueError(f"{path}:{lineno}: BED needs at least three columns")
            try:
                start, end = int(cols[1]), int(cols[2])
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: non-integer coordinates") from exc
            center = start + (expected_length - 1) // 2
            # In the supplied BED4 files column 4 is a coordinate, not an ID.
            if len(cols) >= 4 and re.fullmatch(r"-?\d+", cols[3]):
                center = int(cols[3])
            record = BedRecord(cols[0], start, end, center)
            if start < 0 or end - start != expected_length:
                raise ValueError(f"{path}:{lineno}: expected a valid {expected_length}-bp interval")
            if center != start + (expected_length - 1) // 2:
                raise ValueError(f"{path}:{lineno}: numeric BED4 center is not the window midpoint")
            if record.key in seen:
                raise ValueError(f"{path}:{lineno}: duplicate interval {record.key}")
            seen.add(record.key)
            records.append(record)
    if not records:
        raise ValueError(f"Empty BED: {path}")
    return records


def check_positive_negative_overlap(positives, negatives):
    groups = defaultdict(list)
    for row in positives:
        groups[row.chrom].append((row.start, row.end))
    for chrom, intervals in groups.items():
        intervals.sort()
        starts = [s for s, _ in intervals]
        max_ends = np.maximum.accumulate([e for _, e in intervals]).tolist()
        groups[chrom] = (starts, max_ends)
    for row in negatives:
        if row.chrom not in groups:
            continue
        starts, max_ends = groups[row.chrom]
        last = bisect.bisect_left(starts, row.end) - 1
        if last >= 0 and max_ends[last] > row.start:
            raise ValueError(f"Negative overlaps a positive interval: {row.key}")


def encode_sequence(sequence, context="sequence"):
    try:
        raw = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    except UnicodeEncodeError as exc:
        raise ValueError(f"Non-ASCII DNA in {context}") from exc
    codes = ENCODE[raw]
    if len(codes) != STORED_LENGTH:
        raise ValueError(f"{context}: expected {STORED_LENGTH} bases, got {len(codes)}")
    if np.any(codes == 255):
        invalid = sorted(set(sequence[i] for i in np.flatnonzero(codes == 255)))
        raise ValueError(f"{context}: unsupported bases {invalid}; expected A/C/G/T/N")
    if np.all(codes == 4):
        raise ValueError(f"{context}: entire sequence is N; refusing silent sample removal")
    return codes


class IndexedFasta:
    """Small standard-library reader for a plain FASTA and its .fai index."""
    def __init__(self, path):
        self.path = Path(path)
        if self.path.suffix in (".gz", ".bgz"):
            raise ValueError("Reference FASTA must be uncompressed; supply an ordinary .fa + .fai")
        self.index = {}
        fai = Path(str(self.path) + ".fai")
        if not fai.is_file():
            raise ValueError(f"Missing index {fai}; create it with samtools faidx")
        with fai.open() as handle:
            for line in handle:
                name, length, offset, line_bases, line_width, *_ = line.split()
                if name in self.index:
                    raise ValueError(f"Duplicate FASTA chromosome: {name}")
                values = tuple(map(int, (length, offset, line_bases, line_width)))
                if values[0] <= 0 or values[2] <= 0 or values[3] < values[2]:
                    raise ValueError(f"Invalid .fai line for {name}")
                self.index[name] = values
        self.handle = self.path.open("rb")

    def fetch(self, chrom, start, end):
        if chrom not in self.index:
            raise ValueError(f"Chromosome {chrom!r} is absent from {self.path}; check assembly and naming")
        length, offset, line_bases, line_width = self.index[chrom]
        if not 0 <= start < end <= length:
            raise ValueError(f"Interval {chrom}:{start}-{end} exceeds reference length {length}")
        byte_start = offset + (start // line_bases) * line_width + start % line_bases
        last = end - 1
        byte_end = offset + (last // line_bases) * line_width + last % line_bases + 1
        self.handle.seek(byte_start)
        sequence = self.handle.read(byte_end - byte_start).replace(b"\n", b"").replace(b"\r", b"")
        if len(sequence) != end - start:
            raise ValueError(f"FASTA/index mismatch for {chrom}:{start}-{end}")
        return sequence.decode("ascii")

    def close(self):
        self.handle.close()


def read_interval_fasta(path, records):
    """Match by coordinates, never by line order or numeric center alone."""
    found, header, chunks = {}, None, []
    def finish():
        if header is None:
            return
        match = re.fullmatch(r"([^:\s]+):(\d+)-(\d+)(?:\([+-]\))?", header)
        if match is None:
            raise ValueError(f"{path}: header {header!r} is not chr:start-end; do not guess BED order")
        key = (match[1], int(match[2]), int(match[3]))
        if key in found:
            raise ValueError(f"Duplicate FASTA coordinates: {key}")
        found[key] = "".join(chunks)
        if header.endswith("(-)"):
            raise ValueError("Use genomic forward-strand FASTA (bedtools getfasta without -s)")
    with Path(path).open() as handle:
        for lineno, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                finish()
                header, chunks = line[1:].split()[0], []
            else:
                if header is None:
                    raise ValueError(f"{path}:{lineno}: sequence before FASTA header")
                chunks.append(line)
    finish()
    expected = {r.key for r in records}
    if set(found) != expected:
        missing, extra = expected - set(found), set(found) - expected
        raise ValueError(f"{path}: BED/FASTA mismatch ({len(missing)} missing, {len(extra)} extra records)")
    return [found[r.key] for r in records]


def crop_bounds(stored_length, input_length):
    if not 1 <= input_length <= stored_length:
        raise ValueError("input_length must be between 1 and stored_length")
    return stored_length - input_length + 1


def evaluation_crop_starts(stored_length, input_length, mode):
    last = crop_bounds(stored_length, input_length) - 1
    center = last // 2
    if mode == "center":
        return (center,)
    if mode == "multi":
        return tuple(sorted({0, center, last - center, last}))
    raise ValueError(f"Unknown crop mode: {mode}")


class CyclicNegativePool:
    def __init__(self, positions, seed):
        self.positions = np.asarray(positions, dtype=np.int64)
        if len(self.positions) == 0 or len(np.unique(self.positions)) != len(self.positions):
            raise ValueError("Negative pool must be nonempty and unique")
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(self.positions)
        self.cursor = 0
        self.seen = set()

    def draw(self, count):
        if not 0 < count <= len(self.positions):
            raise ValueError("Cannot draw more distinct negatives than the pool contains")
        drawn, used = [], set()
        while len(drawn) < count:
            if self.cursor == len(self.order):
                self.order = self.rng.permutation(self.positions)
                self.cursor = 0
            value = int(self.order[self.cursor])
            self.cursor += 1
            if value not in used:
                drawn.append(value)
                used.add(value)
        self.seen.update(used)
        return np.array(drawn, dtype=np.int64)


def balanced_batches(positive_positions, negative_positions, negatives_per_positive, batch_size, rng):
    k = negatives_per_positive
    if k < 1 or batch_size < k + 1 or batch_size % (k + 1):
        raise ValueError("batch_size must be a positive multiple of negatives_per_positive + 1")
    positive = rng.permutation(np.asarray(positive_positions, dtype=np.int64))
    negative = np.asarray(negative_positions, dtype=np.int64)
    if len(negative) != k * len(positive):
        raise ValueError("Incorrect number of sampled negatives")
    if len(np.unique(positive)) != len(positive) or len(np.unique(negative)) != len(negative):
        raise ValueError("Duplicate samples in training epoch")
    per_batch = batch_size // (k + 1)
    return [rng.permutation(np.concatenate((positive[i:i + per_batch], negative[k * i:k * (i + per_batch)]))).tolist()
            for i in range(0, len(positive), per_batch)]


def counts(labels):
    positive = int(np.count_nonzero(labels == 1))
    return {"positive": positive, "negative": int(len(labels) - positive), "total": int(len(labels))}


def select_mcc_threshold(labels, scores):
    """Exact grouped-score MCC search; call on validation data only."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or set(labels.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Threshold selection needs finite scores and both binary classes")
    order = np.argsort(-scores, kind="stable")
    sorted_scores, sorted_labels = scores[order], labels[order]
    ends = np.r_[np.flatnonzero(sorted_scores[:-1] != sorted_scores[1:]), len(scores) - 1]
    tp = np.cumsum(sorted_labels, dtype=np.float64)[ends]
    fp = (ends + 1).astype(np.float64) - tp
    positives = float(labels.sum())
    negatives = len(labels) - positives
    fn, tn = positives - tp, negatives - fp
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, denom, out=np.zeros_like(tp), where=denom > 0)
    thresholds = sorted_scores[ends]
    thresholds = np.r_[np.nextafter(sorted_scores[0], np.inf), thresholds]
    mcc = np.r_[0.0, mcc]
    best = np.flatnonzero(np.isclose(mcc, mcc.max(), rtol=1e-12, atol=1e-12))
    index = best[np.argmin(np.abs(thresholds[best] - 0.5))]
    return float(thresholds[index]), float(mcc[index])


def classification_metrics(labels, scores, threshold):
    from sklearn.metrics import average_precision_score, roc_auc_score, matthews_corrcoef
    labels, scores = np.asarray(labels, dtype=np.int64), np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or set(labels.tolist()) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError("Evaluation needs finite scores and both binary classes")
    if np.any((scores < 0) | (scores > 1)) or not math.isfinite(threshold):
        raise ValueError("Scores must be probabilities; threshold must be finite")
    predicted = scores >= threshold
    tp = int(np.count_nonzero(predicted & (labels == 1)))
    fp = int(np.count_nonzero(predicted & (labels == 0)))
    fn = int(np.count_nonzero(~predicted & (labels == 1)))
    tn = int(np.count_nonzero(~predicted & (labels == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    bounded = np.clip(scores, 1e-7, 1 - 1e-7)
    return {"auprc": float(average_precision_score(labels, scores)),
            "auprc_method": "sklearn.average_precision_score (non-trapezoidal)",
            "auroc": float(roc_auc_score(labels, scores)), "threshold": float(threshold),
            "mcc": float(matthews_corrcoef(labels, predicted)), "precision": precision,
            "recall": recall, "specificity": specificity, "fpr": 1 - specificity,
            "balanced_accuracy": (recall + specificity) / 2,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "accuracy": (tp + tn) / len(labels), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "log_loss": float(-(labels * np.log(bounded) + (1 - labels) * np.log1p(-bounded)).mean()),
            "positive_prevalence": float(labels.mean()), "pr_baseline": float(labels.mean()),
            **counts(labels)}


def load_cache(path):
    with np.load(path, allow_pickle=False) as raw:
        cache = {k: raw[k] for k in raw.files}
    required = {"sequences", "labels", "split", "chromosomes", "starts", "ends", "sample_ids", "summary_json"}
    if required - set(cache):
        raise ValueError(f"Missing cache fields: {sorted(required - set(cache))}")
    sequences, labels, split = cache["sequences"], cache["labels"], cache["split"]
    n = len(labels)
    if sequences.dtype != np.uint8 or sequences.shape != (n, STORED_LENGTH) or np.any(sequences > 4):
        raise ValueError("Invalid cache DNA shape/type/codes")
    for key in ("labels", "split", "chromosomes", "starts", "ends", "sample_ids"):
        if cache[key].shape != (n,):
            raise ValueError(f"Invalid cache shape for {key}")
    if not np.all(np.isin(labels, [0, 1])) or not np.all(np.isin(split, [0, 1, 2])):
        raise ValueError("Invalid cache labels/splits")
    if np.any(cache["starts"] < 0) or np.any(cache["ends"] - cache["starts"] != STORED_LENGTH):
        raise ValueError("Invalid cached genomic coordinates")
    if len(np.unique(cache["sample_ids"])) != n:
        raise ValueError("Duplicate sample IDs")
    chrom_sets = []
    for code in range(3):
        index = np.flatnonzero(split == code)
        if set(labels[index].tolist()) != {0, 1}:
            raise ValueError(f"{SPLIT_NAMES[code]} needs both classes")
        chrom_sets.append(set(cache["chromosomes"][index].tolist()))
    if any(chrom_sets[i] & chrom_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Chromosome leakage between cache splits")
    cache["summary"] = json.loads(str(cache.pop("summary_json").item()))
    return cache


def prepare(args):
    names = ("train_positive", "train_negative", "test_positive", "test_negative")
    beds = {name: read_bed(getattr(args, name + "_bed")) for name in names}
    train_chroms = {r.chrom for name in names[:2] for r in beds[name]}
    test_chroms = {r.chrom for name in names[2:] for r in beds[name]}
    if train_chroms & test_chroms:
        raise ValueError(f"Train/test chromosome overlap: {sorted(train_chroms & test_chroms)}")
    val_chroms = set(args.val_chromosomes.split(","))
    if not val_chroms or val_chroms - train_chroms or not (train_chroms - val_chroms):
        raise ValueError("Validation chromosomes must be present in training and leave training chromosomes")
    check_positive_negative_overlap(beds["train_positive"], beds["train_negative"])
    check_positive_negative_overlap(beds["test_positive"], beds["test_negative"])
    fasta_paths = [getattr(args, name + "_fasta") for name in names]
    if args.genome_fasta and any(fasta_paths):
        raise ValueError("Choose reference FASTA or four interval FASTAs, not both")
    if not args.genome_fasta and not all(fasta_paths):
        raise ValueError("Supply --genome-fasta or all four --*-fasta inputs")
    reader = IndexedFasta(args.genome_fasta) if args.genome_fasta else None
    n = sum(len(rows) for rows in beds.values())
    sequences = np.empty((n, STORED_LENGTH), dtype=np.uint8)
    labels = np.empty(n, dtype=np.uint8)
    split = np.empty(n, dtype=np.uint8)
    records, source_hashes, cursor = [], {}, 0
    try:
        for name in names:
            path = getattr(args, name + "_bed")
            source_hashes[name + "_bed"] = {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            rows = beds[name]
            supplied = None
            if reader is None:
                path = getattr(args, name + "_fasta")
                supplied = read_interval_fasta(path, rows)
                source_hashes[name + "_fasta"] = {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for local, row in enumerate(rows):
                dna = reader.fetch(row.chrom, row.start, row.end) if reader else supplied[local]
                sequences[cursor] = encode_sequence(dna, f"{name}/{row.chrom}:{row.start}-{row.end}")
                labels[cursor] = int(name.endswith("positive"))
                split[cursor] = 2 if name.startswith("test") else int(row.chrom in val_chroms)
                records.append(row)
                cursor += 1
            print(f"Loaded {name}: {len(rows):,}", flush=True)
    finally:
        if reader:
            reader.close()
    if args.genome_fasta:
        source_hashes["genome_fasta"] = {"path": str(Path(args.genome_fasta).resolve()), "sha256": sha256_file(args.genome_fasta)}
        source_hashes["genome_fai"] = {"path": str(Path(str(args.genome_fasta) + ".fai").resolve()), "sha256": sha256_file(str(args.genome_fasta) + ".fai")}
    chromosomes = np.array([r.chrom for r in records])
    summary = {"format_version": "at301-v1", "genome_build": args.genome_build,
               "coordinate_convention": "0-based half-open", "stored_length": STORED_LENGTH,
               "sources": source_hashes, "splits": {}}
    for code, name in enumerate(SPLIT_NAMES):
        index = np.flatnonzero(split == code)
        if set(labels[index].tolist()) != {0, 1}:
            raise ValueError(f"{name} must contain positive and negative examples")
        n_fraction = (sequences[index] == 4).mean(axis=1)
        summary["splits"][name] = {
            **counts(labels[index]), "chromosomes": sorted(set(chromosomes[index].tolist())),
            "sequences_with_N": int(np.count_nonzero(n_fraction)),
            "max_N_fraction": float(n_fraction.max()),
            "mean_GC_fraction_ACGT_only": {},
        }
        for label, class_name in ((1, "positive"), (0, "negative")):
            dna = sequences[index[labels[index] == label]]
            gc = ((dna == 1) | (dna == 2)).sum(1) / (dna != 4).sum(1)
            summary["splits"][name]["mean_GC_fraction_ACGT_only"][class_name] = float(gc.mean())
    # Exact sequence duplicates, including reverse complements, can be detected
    # only after FASTA is supplied. Report them; never change fixed test labels.
    signature_sets = []
    sequence_audit = {}
    for code, name in enumerate(SPLIT_NAMES):
        index = np.flatnonzero(split == code)
        signatures = [min(sequences[i].tobytes(), COMPLEMENT[sequences[i][::-1]].tobytes()) for i in index]
        signature_sets.append(set(signatures))
        by_label = [{s for s, i in zip(signatures, index) if labels[i] == label} for label in (0, 1)]
        sequence_audit[name] = {"duplicates_or_RC_duplicates_within_split": len(signatures) - len(set(signatures)),
                                "identical_or_RC_identical_sequences_with_both_labels": len(by_label[0] & by_label[1])}
    for i in range(3):
        for j in range(i + 1, 3):
            sequence_audit[SPLIT_NAMES[i] + "_" + SPLIT_NAMES[j] + "_shared_sequences_or_RC"] = len(signature_sets[i] & signature_sets[j])
    summary["sequence_audit"] = sequence_audit
    summary["sequence_audit_action"] = "Reported only; no evaluation samples removed or relabeled"
    output = Path(args.output)
    if output.suffix != ".npz":
        raise ValueError("Output must end in .npz")
    summary_path = output.with_suffix(".summary.json")
    if not args.overwrite and (output.exists() or summary_path.exists()):
        raise ValueError("Output exists; use a new path or --overwrite explicitly")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, sequences=sequences, labels=labels, split=split,
                            chromosomes=chromosomes, starts=np.array([r.start for r in records], dtype=np.int64),
                            ends=np.array([r.end for r in records], dtype=np.int64),
                            sample_ids=np.array([f"{SPLIT_NAMES[int(s)]}:{int(y)}:{r.chrom}:{r.start}-{r.end}" for r, s, y in zip(records, split, labels)]),
                            summary_json=np.array(json.dumps(summary)))
    # Validate the serialized artifact before committing the output path.
    load_cache(temporary)
    temporary.replace(output)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["splits"], indent=2))
    print("Sequence audit:", json.dumps(sequence_audit, indent=2))
    print(f"Saved: {output}")


# Reproducibility and genomic utilities
# -------------------------------------

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


# Datasets, splits and sampling
# -----------------------------

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
    # Datensaetze ab Generator 7.1 tragen um jedes Fenster ein Umfeld
    # (context_per_side Basen je Seite); die Datei hat context_length Zeilen.
    context_length: int = 0
    context_per_side: int = 0
    context_start: int = 0


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("rt", encoding="utf-8", newline="")


def resolve_metadata_files(dataset_root: Path) -> list[tuple[Path, int]]:
    """Read the single positive and single negative metadata table."""
    files = [(dataset_root / "metadata_positive.tsv", 1),
             (dataset_root / "metadata_negative.tsv", 0)]
    missing = [str(path) for path, _ in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Pooled metadata missing: " + ", ".join(missing)
            + ". Generate a new genome-wide dataset with data_m33_pooled.py."
        )
    return files


def read_metadata(dataset_root: Path) -> list[WindowRecord]:
    """Read labels and geometry, without biological pool annotations."""
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

    for metadata_path, expected_label in metadata_files:
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
                if label != expected_label:
                    raise ValueError(
                        f"Label {label} conflicts with metadata class in {metadata_path}:{line_number}"
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


CACHE_FORMAT = "3-pooled-context"


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
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as existing:
            human_cache = "split" in existing.files
        if human_cache:
            if rebuild:
                raise ValueError("Training reads the existing Human301 cache; --rebuild-cache is not applicable")
            return load_cache(cache_path)
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
    min_split_samples_per_class: int = 2,
    balanced_evaluation: bool = True,
    max_eval_positive_reduction: float = 1.0,
    val_milp_time_limit: float = 120.0,
) -> dict[str, np.ndarray]:
    """Select intact genomic blocks while covering positive and negative classes.

    Blocks linked by overlapping windows form one indivisible component, even
    when the overlap crosses a nominal block boundary. An integer optimization
    chooses components near the desired validation fraction for each stratum.
    """
    if "split" in cache:
        if balanced_evaluation:
            raise ValueError("Human301 validation/test use every cached sample; use --no-balanced-evaluation")
        return {
            name: np.flatnonzero(cache["split"] == code).astype(np.int64)
            for code, name in enumerate(("train", "val", "test"))
        }
    chromosomes = cache["chromosomes"].astype(str)
    starts = cache["starts"].astype(np.int64)
    labels = cache["labels"].astype(np.int64)
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("All labels must be 0 or 1")
    stratum_names = ("positive", "negative")
    strata = 1 - labels
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
    if min_split_samples_per_class < 1:
        raise ValueError("min_split_samples_per_class must be positive")
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
        raise ValueError("At least one training example has an unknown class")
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
        if count < 2 * min_split_samples_per_class
    ]
    if too_small:
        raise RuntimeError(
            f"At least {min_split_samples_per_class} examples of each class "
            "are required in both training and validation. Too few: "
            + ", ".join(too_small)
        )
    # Training draws at most one copy of each negative per epoch. If negatives
    # are fewer, the sampler cycles through all positives over later epochs.
    # Validation/test randomly keep as many positives as negatives if needed.
    if totals[1] < totals[0]:
        print(
            f"Training has {int(totals[1]):,} negatives for {int(totals[0]):,} positives. "
            "Positive examples will rotate across epochs, without duplicates inside an epoch.",
            file=sys.stderr,
        )

    # Every chromosome/block pair is atomic. Merge neighboring blocks if
    # windows from them overlap, preventing train/validation window leakage.
    candidate_chroms = chromosomes[train_candidates]
    candidate_starts = starts[train_candidates]
    context_starts = cache.get("context_starts", starts)[train_candidates].astype(np.int64)
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
        positions = positions[np.argsort(context_starts[positions], kind="stable")]
        previous_end = -1
        previous_block = -1
        for position in positions:
            start = int(context_starts[position])
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
    # requested fraction. The last four continuous variables are the deviations.
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
            np.full(n_strata, min_split_samples_per_class),
            totals - min_split_samples_per_class,
        ),
    ]
    # Training balances each epoch by cycling positives if negatives are scarce.
    # Validation uses the same optional reduction limit as the evaluator.
    pad = ((0, 0), (0, 2 * n_strata))
    if balanced_evaluation and max_eval_positive_reduction < 1.0:
        val_balance = counts[1:] - (1.0 - max_eval_positive_reduction) * counts[:1]
        constraints.append(LinearConstraint(
            sparse.csr_matrix(np.pad(val_balance, pad)),
            np.zeros(1), np.full(1, np.inf),
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
            "If infeasible, check class counts and block sizes; if timed out, "
            "increase --val-milp-time-limit. No individual windows were reassigned."
        )
    selected_components = np.rint(result.x[:n_components]).astype(bool)
    validation_counts = counts @ selected_components.astype(np.float64)
    if (
        np.max(np.abs(result.x[:n_components] - selected_components)) > 1e-5
        or np.any(validation_counts < min_split_samples_per_class)
        or np.any(totals - validation_counts < min_split_samples_per_class)
        or (balanced_evaluation and max_eval_positive_reduction < 1.0 and np.any(
            (counts[1:] - (1.0 - max_eval_positive_reduction) * counts[:1])
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
        f"per-class counts: {dict(zip(stratum_names, validation_counts.astype(int).tolist()))}"
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
    if "split" in cache:
        if context_length != STORED_LENGTH:
            raise ValueError("Human301 caches must store 301-bp sequences")
        # The odd 45-bp excess is asymmetric: 22 bases left, 23 right.
        return 256, 22
    if "window_length" in cache:
        window_length = int(np.asarray(cache["window_length"]).item())
        context_per_side = int(np.asarray(cache["context_per_side"]).item())
    else:
        window_length, context_per_side = context_length, 0
    if window_length + 2 * context_per_side != context_length:
        raise ValueError("Cache geometry is inconsistent with the stored sequences")
    return window_length, context_per_side


class TFBindingDataset(Dataset[dict[str, Any]]):
    """MAA dataset with fresh Human301 crops and intact evaluation sequences.

    Human training crops start uniformly at 0..45, independently per access.
    Human evaluation returns all 301 bases; run_epoch applies the fixed views.
    The classification readout stays at the crop midpoint, also after shifting.
    MAA's FE-target handling remains available for caches supplying FE profiles.
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
        training: bool = False,
    ) -> None:
        self.sequences = cache["sequences"]
        self.fe_profiles = cache.get("fe_profiles")
        self.labels = cache["labels"]
        self.chromosomes = cache["chromosomes"]
        self.starts = cache["starts"]
        self.context_starts = cache.get("context_starts", cache["starts"])
        self.centers = cache.get("centers")
        self.sample_ids = cache["sample_ids"]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.human301 = "split" in cache
        self.training = bool(training)
        self.reverse_complement_probability = float(reverse_complement_probability)
        if not 0.0 <= self.reverse_complement_probability <= 1.0:
            raise ValueError("reverse_complement_probability must be in [0, 1]")
        self.window_length, self.context_per_side = cache_geometry(cache)
        self.shift_max = int(shift_max)
        self.shift_distribution = shift_distribution
        if self.shift_max < 0:
            raise ValueError("shift_max must not be negative")
        if not self.human301 and self.shift_max > self.context_per_side:
            raise ValueError("shift_max exceeds the context of this dataset")
        if fixed_shifts is not None:
            fixed_shifts = np.asarray(fixed_shifts, dtype=np.int64)
            if fixed_shifts.shape != (self.indices.size,):
                raise ValueError("fixed_shifts must hold one shift per dataset position")
            if not self.human301 and np.any(np.abs(fixed_shifts) > self.context_per_side):
                raise ValueError("fixed_shifts exceed the dataset context")
        self.fixed_shifts = fixed_shifts
        self.seed = int(seed)
        self.epoch = 0
        self.epoch_shifts = np.zeros(self.indices.size, dtype=np.int64)
        self.rng = None
        self.rng_worker = None
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.rng = None
        self.rng_worker = None
        if self.human301:
            return
        if self.fixed_shifts is not None:
            self.epoch_shifts = self.fixed_shifts
            return
        rng = np.random.default_rng([self.seed, self.epoch, 7])
        self.epoch_shifts = draw_shifts(
            rng, self.indices.size, self.shift_max, self.shift_distribution
        )

    def _augmentation_rng(self) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id + 1
        if self.rng is None or self.rng_worker != worker_id:
            self.rng = np.random.default_rng([self.seed, self.epoch, worker_id, 301])
            self.rng_worker = worker_id
        return self.rng

    def shift_summary(self) -> dict[str, Any]:
        if self.human301:
            return {
                "crop_start_min": 0, "crop_start_max": 45,
                "shift_min": -22, "shift_max": 23,
                "distribution": "uniform", "redrawn_per_access": True,
            }
        shifts = self.epoch_shifts[:self.indices.size]
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
        if self.human301:
            rng = self._augmentation_rng() if self.training else None
            crop_start = int(rng.integers(0, 46)) if self.training else 0
            crop_end = crop_start + self.window_length if self.training else STORED_LENGTH
            shift = crop_start - self.context_per_side if self.training else 0
            reverse = self.training and rng.random() < self.reverse_complement_probability
        else:
            shift = int(self.epoch_shifts[item])
            crop_start = self.context_per_side + shift
            crop_end = crop_start + self.window_length
            reverse = random.random() < self.reverse_complement_probability

        codes = torch.from_numpy(
            self.sequences[index, crop_start:crop_end].astype(np.int64, copy=False)
        )
        fe = None if self.fe_profiles is None else torch.from_numpy(
            self.fe_profiles[index, crop_start:crop_end].astype(np.float32, copy=False)
        )
        center = codes.numel() // 2
        if reverse:
            complement = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)
            codes = complement[codes].flip(0)
            if fe is not None:
                fe = fe.flip(0)
            center = codes.numel() - 1 - center

        result = {
            "sequence": ONE_HOT_LOOKUP[codes].transpose(0, 1).contiguous(),
            "valid_mask": codes.ne(4).to(torch.float32),
            "label": torch.tensor(float(self.labels[index]), dtype=torch.float32),
            "center_index": torch.tensor(center, dtype=torch.long),
            "sample_id": str(self.sample_ids[index]),
            "chromosome": str(self.chromosomes[index]),
            "window_start": int(self.context_starts[index]) + crop_start,
            "shift": shift,
        }
        if fe is not None:
            result["fe_profile"] = fe
        return result


def class_counts(cache: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, int]:
    labels = cache["labels"][indices].astype(np.int64)
    return {"positive": int(np.count_nonzero(labels == 1)),
            "negative": int(np.count_nonzero(labels == 0))}


def make_fixed_balanced_evaluation_split(
    cache: dict[str, np.ndarray], indices: np.ndarray, seed: int, split_name: str,
    max_positive_reduction: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select one reproducible global 1:1 evaluation subset, without annotations."""
    if not 0 <= max_positive_reduction <= 1:
        raise ValueError("max_positive_reduction must be in [0, 1]")
    indices = np.asarray(indices, dtype=np.int64)
    labels = cache["labels"][indices].astype(np.int64)
    positives = indices[labels == 1]
    negatives = indices[labels == 0]
    if not positives.size or not negatives.size:
        raise ValueError(f"{split_name} must contain both positive and negative examples")
    pairs = int(min(positives.size, negatives.size))
    dropped = int(positives.size - pairs)
    reduction = dropped / positives.size
    if reduction > max_positive_reduction:
        raise ValueError(
            f"Cannot balance {split_name}: {negatives.size:,} negatives for {positives.size:,} positives "
            f"({reduction:.1%} short, allowed {max_positive_reduction:.1%}). "
            "Raise --max-eval-positive-reduction or provide more negatives."
        )
    if dropped:
        print(
            f"WARNING: {split_name}: only {negatives.size:,} negatives for {positives.size:,} positives; "
            f"{dropped:,} positives were dropped at random to keep the global 1:1 ratio.",
            file=sys.stderr,
        )
    rng = np.random.default_rng(seed)
    selected = np.concatenate((rng.choice(positives, pairs, replace=False),
                               rng.choice(negatives, pairs, replace=False)))
    summary = {"scope": "global", "max_positive_reduction": max_positive_reduction,
               "positives_available": int(positives.size), "negatives_available": int(negatives.size),
               "pairs_used": pairs, "positives_dropped_total": dropped}
    return rng.permutation(selected), summary


class CyclicBalancedNegativeSampler(Sampler[list[int]]):
    """Use all positives, rotate distinct negatives, and emit balanced batches.

    Optional hard examples are selected exclusively from scored training
    negatives; remaining negatives continue through the cyclic random pool.
    """
    def __init__(self, dataset, seed, negatives_per_positive=10, batch_size=88, hard_negative_fraction=0.0):
        self.dataset = dataset
        labels = dataset.labels[dataset.indices].astype(np.int64)
        self.epoch = 0
        self.min_epochs_for_positive_coverage = 1
        self.positives = np.flatnonzero(labels == 1)
        self.samples_per_epoch = (negatives_per_positive + 1) * len(self.positives)
        self.pool = CyclicNegativePool(np.flatnonzero(labels == 0), seed + 71)
        self.k, self.batch_size = negatives_per_positive, batch_size
        self.rng = np.random.default_rng(seed + 83)
        if self.k * len(self.positives) > len(self.pool.positions):
            raise ValueError("Not enough distinct training negatives for requested per-epoch ratio")
        if self.k < 1 or self.batch_size < self.k + 1 or self.batch_size % (self.k + 1):
            raise ValueError("batch_size must be a multiple of negatives_per_positive + 1")
        if not 0 <= hard_negative_fraction < 1:
            raise ValueError("hard_negative_fraction must be at least zero and below one")
        self.hard_count = int(self.k * len(self.positives) * hard_negative_fraction)
        if hard_negative_fraction > 0 and self.hard_count == 0:
            raise ValueError("Hard-negative fraction is too small for this training set")
        self.hard_candidates = np.empty(0, dtype=np.int64)
        self.last_summary = {}

    def update_hard_negatives(self, scores):
        """Scores must follow pool.positions, which contains training negatives only."""
        scores = np.asarray(scores)
        if scores.shape != self.pool.positions.shape or not np.isfinite(scores).all():
            raise ValueError("One finite score per training negative is required")
        # Randomly draw the hard part from the highest-scoring candidates.
        bank_size = min(len(scores), 2 * self.hard_count)
        order = np.argsort(-scores, kind="stable")[:bank_size]
        self.hard_candidates = self.pool.positions[order]

    def _draw_random_without(self, count, excluded):
        """Continue the cyclic pool while keeping this epoch's negatives unique."""
        drawn, used = [], set(excluded)
        while len(drawn) < count:
            if self.pool.cursor == len(self.pool.order):
                self.pool.order = self.pool.rng.permutation(self.pool.positions)
                self.pool.cursor = 0
            value = int(self.pool.order[self.pool.cursor])
            self.pool.cursor += 1
            if value not in used:
                drawn.append(value)
                used.add(value)
        self.pool.seen.update(used)
        return np.asarray(drawn, dtype=np.int64)

    def __len__(self):
        return math.ceil(len(self.positives) / (self.batch_size // (self.k + 1)))

    def __iter__(self):
        # Keep this a generator: DataLoader can create an unused iterator when
        # starting workers. Pool/RNG state advances only when batches are read.
        self.epoch += 1
        total = self.k * len(self.positives)
        hard = np.empty(0, dtype=np.int64)
        if len(self.hard_candidates):
            hard = self.rng.choice(self.hard_candidates, size=self.hard_count, replace=False)
        if len(hard):
            random_negatives = self._draw_random_without(total - len(hard), hard)
            negatives = self.rng.permutation(np.concatenate((hard, random_negatives)))
        else:
            negatives = self.pool.draw(total)
        batches = balanced_batches(self.positives, negatives, self.k, self.batch_size, self.rng)
        self.last_summary = {"epoch": self.epoch, "scope": "global", "positive": len(self.positives),
                             "positive_pool_size": len(self.positives), "positive_coverage": 1.0,
                             "negative": len(negatives), "negative_total": len(negatives),
                             "hard_negative": len(hard), "random_negative": total - len(hard),
                             "negatives_seen": len(self.pool.seen), "negative_pool_size": len(self.pool.positions),
                             "negative_pool_coverage": len(self.pool.seen) / len(self.pool.positions)}
        yield from batches


# Data loaders and device selection
# ---------------------------------

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
    if isinstance(sampler, CyclicBalancedNegativeSampler):
        return DataLoader(
            dataset, batch_sampler=sampler, num_workers=num_workers,
            pin_memory=device.type == "cuda", persistent_workers=False,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
        )
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


# Cache manifests and task priority
# ---------------------------------

def read_manifest(path: Path) -> list[dict]:
    path = path.resolve()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["tf", "assay", "cache"]:
            raise ValueError("Manifest header must be exactly: tf,assay,cache")
        rows = []
        keys = set()
        for row in reader:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+", row["tf"]):
                raise ValueError(f"Invalid TF name: {row['tf']}")
            assay = row["assay"].upper()
            assay = {"CHS": "CHIP", "CHIPSEQ": "CHIP", "GHT-SELEX": "GHTS"}.get(assay, assay)
            if assay not in {"CHIP", "GHTS"}:
                raise ValueError(f"Unknown assay {assay}")
            cache = Path(row["cache"]).expanduser()
            cache = cache.resolve() if cache.is_absolute() else (path.parent / cache).resolve()
            key = (row["tf"], assay)
            if key in keys:
                raise ValueError(f"Duplicate manifest entry: {key}")
            if not cache.is_file():
                raise FileNotFoundError(cache)
            keys.add(key)
            rows.append({"tf": row["tf"], "assay": assay, "cache": cache})
    if not rows:
        raise ValueError("Manifest has no caches")
    return rows


def discover(root: Path, destination: Path) -> None:
    """Infer TF/assay only from recognizable Human301 cache paths or sources.

    Ambiguity is an error, not an opportunity to guess a TF assignment.
    """
    import numpy as np
    candidates = []
    for path in sorted(root.resolve().rglob("*.npz")):
        try:
            with np.load(path, allow_pickle=False) as raw:
                if not {"split", "summary_json", "sequences"}.issubset(raw.files):
                    continue
                summary = json.loads(str(raw["summary_json"].item()))
        except (OSError, ValueError):
            continue
        if summary.get("split_variant", "original") != "original":
            continue
        texts = [str(path.relative_to(root.resolve()))]
        texts += [str(v.get("path", "")) for v in summary.get("sources", {}).values() if isinstance(v, dict)]
        assignments = set()
        for text in texts:
            # Known layouts: TF/CHIP/dataset.npz, CHIP/TF/dataset.npz,
            # TF_CHIP.npz, or BED names containing those two tokens.
            tokens = [t for t in re.split(r"[/_.\\-]+", text) if t]
            for i, token in enumerate(tokens):
                assay = {"CHIP": "CHIP", "CHS": "CHIP", "GHTS": "GHTS"}.get(token.upper())
                if assay:
                    for adjacent in (i-1, i+1):
                        if 0 <= adjacent < len(tokens):
                            tf = tokens[adjacent]
                            if re.fullmatch(r"[A-Z][A-Z0-9]{1,15}", tf) and tf not in {"CHIP","CHS","GHTS","DNA","TRAIN","TEST","TFBS"}:
                                assignments.add((tf, assay))
        if len(assignments) != 1:
            raise ValueError(f"Cannot unambiguously assign {path}. Enter it manually in a tf,assay,cache manifest.")
        tf, assay = assignments.pop()
        candidates.append({"tf": tf, "assay": assay, "cache": str(path.resolve())})
    if not candidates:
        raise ValueError(f"No Human301 caches found below {root}")
    keys = [(r["tf"],r["assay"]) for r in candidates]
    if len(set(keys)) != len(keys):
        raise ValueError("Multiple caches for the same TF/assay; choose one explicitly in the manifest")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open("w",newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["tf","assay","cache"])
        writer.writeheader()
        writer.writerows(candidates)


SPLIT_VARIANTS = ("original", "split_v2", "split_v3")
SPLIT_ALIASES = {"mirrored": "split_v2", "mixed": "split_v3"}
SPLIT_DEFINITIONS = {
    "split_v2": {
        "train": [4, 6, 8, 10, 12, 14, 18, 20, 22],
        "validation": [2, 16],
        "test": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21],
    },
    "split_v3": {
        "train": [2, 5, 6, 10, 11, 13, 15, 18, 20],
        "validation": [1, 8],
        "test": [3, 4, 7, 9, 12, 14, 16, 17, 19, 21, 22],
    },
}


def validate_split_variant(value):
    if not isinstance(value, str):
        raise ValueError(f"split_variant must be one of {SPLIT_VARIANTS}, got {value!r}")
    value = SPLIT_ALIASES.get(value, value)
    if value not in SPLIT_VARIANTS:
        raise ValueError(f"split_variant must be one of {SPLIT_VARIANTS}, got {value!r}")
    return value


def config_split_variant(config):
    spec = config.get("data", {})
    variant = validate_split_variant(spec.get("params", {}).get("split_variant", "original"))
    if variant != "original" and spec.get("factory") not in (
        "ablation.providers:human301", "data:human301",
    ):
        raise ValueError("Chromosome split variants require the human301 data factory")
    return variant


def split_cache_path(source, variant):
    variant = validate_split_variant(variant)
    source = Path(source).resolve()
    if variant == "original":
        return source
    return source.parent / "splits" / variant / source.name


def cache_summary(path):
    with np.load(path, allow_pickle=False) as raw:
        return json.loads(str(raw["summary_json"].item()))


def resolve_split_cache_paths(paths, variant="original"):
    """Resolve once at planning; also accept already-resolved task-table paths."""
    variant = validate_split_variant(variant)
    if variant == "original":
        # Keep legacy paths, cache bytes and cached split assignments untouched.
        return dict(paths)
    resolved = {}
    for assay, raw in paths.items():
        source = Path(raw).resolve()
        source_summary = cache_summary(source)
        existing_variant = validate_split_variant(source_summary.get("split_variant", "original"))
        if existing_variant == variant:
            target, summary = source, source_summary
        elif existing_variant != "original":
            raise ValueError(f"Cannot use {existing_variant} cache as {variant}: {source}")
        else:
            target = split_cache_path(source, variant)
            if not target.is_file():
                raise FileNotFoundError(
                    f"Split cache missing: {target}. Run python data.py prepare-splits "
                    "--manifest configs/caches.csv first (submit_array.sh does this automatically)."
                )
            summary = cache_summary(target)
            if summary.get("parent_cache", {}).get("sha256") != sha256_file(source):
                raise ValueError(f"Split cache belongs to different source bytes: {target}")
        if (validate_split_variant(summary.get("split_variant")) != variant
                or summary.get("split_definition") != SPLIT_DEFINITIONS[variant]):
            raise ValueError(f"Unexpected chromosome split definition: {target}")
        resolved[assay] = target
    return resolved


def _variant_split(chromosomes, variant):
    definition = SPLIT_DEFINITIONS[validate_split_variant(variant)]
    mapping = {str(chrom): code for code, name in enumerate(SPLIT_NAMES)
               for chrom in definition[name]}
    if (sum(len(v) for v in definition.values()) != 22
            or len(mapping) != 22 or set(mapping) != {str(i) for i in range(1, 23)}):
        raise ValueError(f"Split must assign each autosome exactly once: {variant}")
    normalized = [canonical_chrom(c) for c in chromosomes]
    unsupported = set(normalized) - mapping.keys()
    if unsupported:
        raise ValueError(f"{variant} only defines autosomes 1-22; found {sorted(unsupported)}")
    return np.asarray([mapping[c] for c in normalized], dtype=np.uint8)


def _variant_summary(cache, split, source, source_hash, variant):
    summary = copy.deepcopy(cache["summary"])
    summary.update(
        split_variant=variant, split_definition=copy.deepcopy(SPLIT_DEFINITIONS[variant]),
        parent_cache={"path": str(source), "sha256": source_hash},
        sample_id_note="Stable source identifiers; split prefixes in IDs describe the original source, not the new assignment.",
        source_assignment_note="sources lists the original input files; current assignments are in split_definition and the split array.",
    )
    summary["splits"] = {}
    signatures_by_split = []
    audit = {}
    for code, name in enumerate(SPLIT_NAMES):
        index = np.flatnonzero(split == code)
        labels = cache["labels"][index]
        if set(labels.tolist()) != {0, 1}:
            raise ValueError(f"{source}: {variant}/{name} needs positive and negative samples")
        dna = cache["sequences"][index]
        nf = (dna == 4).mean(axis=1)
        stats = dict(counts(labels), chromosomes=sorted(set(cache["chromosomes"][index].tolist())),
                     sequences_with_N=int(np.count_nonzero(nf)),
                     max_N_fraction=float(nf.max()), mean_GC_fraction_ACGT_only={})
        for label, label_name in ((1, "positive"), (0, "negative")):
            selected = dna[labels == label]
            gc = ((selected == 1) | (selected == 2)).sum(1) / (selected != 4).sum(1)
            stats["mean_GC_fraction_ACGT_only"][label_name] = float(gc.mean())
        summary["splits"][name] = stats
        signatures = [hashlib.sha256(min(seq.tobytes(), COMPLEMENT[seq[::-1]].tobytes())).digest()
                      for seq in dna]
        signature_set = set(signatures)
        by_label = [{s for s, y in zip(signatures, labels) if y == label} for label in (0, 1)]
        audit[name] = {
            "duplicates_or_RC_duplicates_within_split": len(signatures) - len(signature_set),
            "identical_or_RC_identical_sequences_with_both_labels": len(by_label[0] & by_label[1]),
        }
        signatures_by_split.append(signature_set)
    for i in range(3):
        for j in range(i + 1, 3):
            audit[f"{SPLIT_NAMES[i]}_{SPLIT_NAMES[j]}_shared_sequences_or_RC"] = len(
                signatures_by_split[i] & signatures_by_split[j])
    summary["sequence_audit"] = audit
    summary["sequence_audit_action"] = "Reported only; no samples removed, duplicated or relabeled"
    return summary


def _atomic_cache_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(value, handle, indent=2, default=json_default)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        Path(temp).unlink(missing_ok=True)


def prepare_split_variants(manifest, variants=("split_v2", "split_v3"), *, overwrite=False, report=None):
    """Prepare every TF/assay in the original manifest; never rewrite originals."""
    variants = tuple(dict.fromkeys(validate_split_variant(v) for v in variants))
    rows = read_manifest(Path(manifest))
    records = []
    for row in rows:
        source = Path(row["cache"])
        cache = load_cache(source)
        if cache["summary"].get("split_variant", "original") != "original":
            raise ValueError(f"Preparation requires the original manifest, got {source}")
        source_hash = sha256_file(source)
        for variant in variants:
            if variant == "original":
                record = dict(tf=row["tf"], assay=row["assay"], split_variant=variant,
                              cache=str(source), status="unchanged", splits=cache["summary"].get("splits", {}))
            else:
                target = split_cache_path(source, variant)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.with_suffix(target.suffix + ".lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    expected_split = _variant_split(cache["chromosomes"], variant)
                    reuse = target.is_file() and not overwrite
                    if reuse:
                        existing = load_cache(target)
                        summary = existing["summary"]
                        if (summary.get("parent_cache", {}).get("sha256") != source_hash
                                or summary.get("split_variant") != variant
                                or summary.get("split_definition") != SPLIT_DEFINITIONS[variant]
                                or not np.array_equal(existing["split"], expected_split)
                                or any(not np.array_equal(existing[k], cache[k])
                                       for k in cache if k not in ("summary", "split"))):
                            raise ValueError(f"Existing split cache differs from its source/definition: {target}. "
                                             "Use a fresh source copy or --overwrite when no runs use this variant.")
                        status = "reused"
                        del existing
                    else:
                        summary = _variant_summary(cache, expected_split, source, source_hash, variant)
                        payload = {k: v for k, v in cache.items() if k != "summary"}
                        payload.update(split=expected_split, summary_json=np.asarray(json.dumps(summary)))
                        fd, temp = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
                        try:
                            with os.fdopen(fd, "wb") as handle:
                                np.savez_compressed(handle, **payload)
                            load_cache(Path(temp))
                            if sha256_file(source) != source_hash:
                                raise ValueError(f"Source changed during preparation: {source}")
                            os.replace(temp, target)
                        finally:
                            Path(temp).unlink(missing_ok=True)
                        status = "created"
                    _atomic_cache_json(target.with_suffix(".summary.json"), summary)
                record = dict(tf=row["tf"], assay=row["assay"], split_variant=variant,
                              cache=str(target), status=status, source_cache=str(source),
                              source_sha256=source_hash, splits=summary["splits"])
            records.append(record)
            sizes = '; '.join(f"{name}: +{s['positive']}/-{s['negative']}"
                              for name, s in record['splits'].items() if 'positive' in s and 'negative' in s)
            print(f"{row['tf']}/{row['assay']} | {variant}: {record['status']} | {sizes} | {record['cache']}", flush=True)
            if report is not None:
                _atomic_cache_json(report, {"manifest": str(Path(manifest).resolve()), "records": records})
    return records


def human301(context, split_variant="original"):
    split_variant = validate_split_variant(split_variant)
    paths = resolve_split_cache_paths(context.cache_paths, split_variant)
    if split_variant != "original":
        predictor_params = getattr(context, "config", {}).get("predictor", {}).get("params", {})
        if "pwm_root" in predictor_params:
            assay = {"CHIP": "CHS", "GHTS": "GHTS"}[context.train_assay]
            validate_pwm_split_provenance(context, predictor_params["pwm_root"], assay)
        context.cache_paths = dict(paths)
        context.recorder.update(split_variant=split_variant,
                                train_cache_path=str(paths[context.train_assay]))
    return paths


def validate_pwm_split_provenance(context, pwm_root, assay):
    """Require an explicit split-specific provenance record for resplit PWMs."""
    variant = config_split_variant(getattr(context, "config", {}))
    if variant == "original":
        return None
    path = Path(pwm_root) / context.tf / assay / "split_provenance.json"
    if not path.is_file():
        raise ValueError(
            f"{variant} PWM run needs {path}. Rebuild/select PWMs for this split and "
            "record split_variant, fit_chromosomes and selection_chromosomes; "
            "original-split PWMs must not be silently reused."
        )
    record = json.loads(path.read_text())
    if validate_split_variant(record.get("split_variant")) != variant:
        raise ValueError(f"PWM split provenance mismatch: {path}")
    for key, allowed in (
        ("fit_chromosomes", set(SPLIT_DEFINITIONS[variant]["train"])),
        ("selection_chromosomes", set(SPLIT_DEFINITIONS[variant]["train"])
                                 | set(SPLIT_DEFINITIONS[variant]["validation"])),
    ):
        values = record.get(key)
        if not isinstance(values, list):
            raise ValueError(f"PWM provenance needs a {key} list: {path}")
        normalized = {canonical_chrom(c) for c in values}
        if not normalized <= {str(c) for c in allowed}:
            raise ValueError(f"PWM {key} includes chromosomes outside the allowed source split: {path}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "record": record}


def make_split_configs(config_paths, variants=("split_v2", "split_v3")):
    """Copy selected recipes and their references; preserve all fitting parameters."""
    variants = tuple(dict.fromkeys(validate_split_variant(v) for v in variants))
    if "original" in variants:
        raise ValueError("Use original configs directly; only split_v2/split_v3 copies are generated")
    written = []
    visiting = set()
    done = set()

    def make(raw, variant):
        path = Path(raw)
        if not path.is_file():
            name = str(raw) if str(raw).endswith(".json") else str(raw) + ".json"
            path = Path(__file__).resolve().parent / "configs" / name
        path = path.resolve()
        key = (path, variant)
        if key in done:
            return
        if key in visiting:
            raise ValueError(f"Circular config references: {path}")
        visiting.add(key)
        base = json.loads(path.read_text())
        if config_split_variant(base) != "original":
            raise ValueError(f"Generate split copies from an original config: {path}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", base["name"]):
            raise ValueError(f"Unsafe experiment name: {path}")
        reference = base["reference"]
        if not isinstance(reference, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", reference):
            raise ValueError(f"Unsafe reference name: {path}")
        if reference != base["name"]:
            make(path.parent / (reference + ".json"), variant)
        result = copy.deepcopy(base)
        result["name"] += "_" + variant
        result["reference"] += "_" + variant
        result["description"] = base.get("description", "") + " Chromosome split: " + variant + "."
        result["data"]["params"]["split_variant"] = variant
        destination = path.parent / (result["name"] + ".json")
        if destination.exists():
            if json.loads(destination.read_text()) != result:
                raise FileExistsError(f"Different config already exists: {destination}")
        else:
            with destination.open("x") as handle:
                json.dump(result, handle, indent=2)
                handle.write("\n")
        written.append(destination)
        print(destination, flush=True)
        visiting.remove(key)
        done.add(key)

    for raw in config_paths:
        for variant in variants:
            make(raw, variant)
    return written

def read_priority(path: Path) -> dict[tuple[str, str], int]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["priority", "tf", "assay"]:
            raise ValueError("Priority header must be exactly: priority,tf,assay")
        ranks = {}
        for expected, row in enumerate(reader, 1):
            assay = {"CHS": "CHIP"}.get(row["assay"], row["assay"])
            key = (row["tf"], assay)
            if int(row["priority"]) != expected or not row["tf"]:
                raise ValueError("Priority numbers must be consecutive, starting at 1")
            if assay not in {"CHIP", "GHTS"} or key in ranks:
                raise ValueError(f"Invalid or duplicate priority entry: {key}")
            ranks[key] = expected
    if not ranks:
        raise ValueError("Priority list is empty")
    return ranks


def order_tasks(rows: list[dict], path: Path) -> list[dict]:
    ranks = read_priority(path)
    # Stable sort: repetitions retain seed order. Unknown pairs follow the list
    # in manifest order. Custom multi-TF planners keep their own task order.
    return sorted(rows, key=lambda row: ranks.get((row["tf"], row["train_assay"]), len(ranks) + 1))


# Cache preparation command
# -------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("train-positive", "train-negative", "test-positive", "test-negative"):
        prep.add_argument("--" + name + "-bed", required=True, type=Path)
        prep.add_argument("--" + name + "-fasta", type=Path)
    prep.add_argument("--genome-fasta", type=Path)
    prep.add_argument("--genome-build", required=True, help="Explicit assembly provenance; not inferred from chromosomes")
    prep.add_argument("--val-chromosomes", default="chr5,chr7")
    prep.add_argument("--output", required=True, type=Path)
    prep.add_argument("--overwrite", action="store_true")
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--cache", required=True, type=Path)
    variants = sub.add_parser("prepare-splits", help="Prepare chromosome splits for every TF/assay in the original manifest")
    variants.add_argument("--manifest", required=True, type=Path)
    variants.add_argument("--variants", nargs="+", choices=SPLIT_VARIANTS, default=["split_v2", "split_v3"])
    variants.add_argument("--report", type=Path)
    variants.add_argument("--overwrite", action="store_true", help="Replace variant caches only; use only when no runs use them")
    recipes = sub.add_parser("make-split-configs", help="Create split-specific copies of configs and their references")
    recipes.add_argument("--config", required=True, nargs="+", type=Path)
    recipes.add_argument("--variants", nargs="+", choices=("split_v2", "split_v3"), default=["split_v2", "split_v3"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "prepare-splits":
        prepare_split_variants(args.manifest, args.variants, overwrite=args.overwrite, report=args.report)
    elif args.command == "make-split-configs":
        make_split_configs(args.config, args.variants)
    else:
        print(json.dumps(load_cache(args.cache)["summary"], indent=2))


if __name__ == "__main__":
    main()
