#!/usr/bin/env python3
"""Prepare fixed BED/FASTA train/test sets for maa301.py (NumPy only).

BED coordinates are interpreted as 0-based, half-open. Use either an indexed,
uncompressed reference FASTA or four interval FASTAs whose headers are
chr:start-end (for example, bedtools getfasta output). No row-order guessing,
GC matching, resampling of evaluation sets, or signal-profile targets.
"""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

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
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        print(json.dumps(load_cache(args.cache)["summary"], indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
