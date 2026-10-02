#!/usr/bin/env python3
"""Genomweiter ChIP-seq-Datensatz mit gemeinsamem GC-Matching.

Alle eindeutigen Summits oberhalb von --summit-min-value werden als positive
Kandidaten betrachtet. Ein positives 1024-bp-Kernfenster liegt exakt auf dem
Summit und muss vollstaendig auf dem Chromosom liegen. Es gibt keine Auswahl
nach TSS-Abstand, Genmitgliedschaft oder Herkunftsgruppe.

Negative Kandidaten umfassen ALLE vollstaendigen Fensterstarts in 1-bp-Schritten
auf den ausgewaehlten Chromosomen. Nur die bisherigen harten Kriterien filtern:
max(-log10(p)) <= --pval-threshold im Kernfenster, keine Ueberlappung mit einem
positiven Kernfenster und messbarer GC-Anteil. Ueberlappende Negative bleiben
zunaechst im SQLite-Pool. Der GCMatcher waehlt daraus gemeinsam moeglichst viele
untereinander nicht ueberlappende Fenster passend zur GC-Verteilung ALLER
Positiven (gleiche CDF-/Quantiltoleranzen, keine feste Zielzahl, Heuristik).

Optional werden negative Zentren in Genkoerpern oder bis zu 1000 bp von
Genen/TSS beim GC-Matching bevorzugt. Die Lage ist kein harter Filter:
entfernte Kandidaten bleiben im gemeinsamen Pool und die GC-Grenzen gelten
unveraendert. GFF und TSS-BED dienen ausschliesslich dieser Prioritaet.

Die Fensterdateien enthalten weiterhin standardmaessig 196 bp Kontext je
Seite (1416 bp insgesamt). Filter und GC-Matching betreffen nur den Kern;
der Kontext dient dem zufaelligen Zuschnitt im Training. Genomische
Koordinaten sind 0-basiert und halb-offen. Ausgaben: metadata_positive.tsv
und metadata_negative.tsv; es werden keine Herkunftsgruppen eingefuehrt.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import math
import re
import shlex
import shutil
import sys
import tempfile
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from gc_matching import MATCHER_VERSION, CandidateStore, GCMatcher, WindowCandidate

LOGGER = logging.getLogger("tf_binding_dataset_pooled")
SCRIPT_VERSION = "8.1.0-pooled-gc-soft-location-preference"

@dataclass(frozen=True)
class Summit:
    chrom_key: str
    chrom_bigwig: str
    position: int
    score: float
    interval_start: int
    interval_end: int

    @property
    def key(self) -> tuple[str, int]:
        return (self.chrom_key, self.position)


@dataclass(frozen=True)
class ChromBundle:
    chrom_key: str
    fasta_name: str
    fe_name: str
    pval_name: str
    summit_name: str
    length: int


@dataclass(frozen=True)
class PositiveSample:
    summit: Summit
    bundle: ChromBundle
    window_start: int
    window_end: int

    @property
    def center(self) -> int:
        return self.summit.position

    @property
    def summit_index(self) -> int:
        return self.summit.position - self.window_start


@dataclass(frozen=True)
class NegativeSample:
    bundle: ChromBundle
    center: int
    window_start: int
    window_end: int
    selection_pval_max: float
    gc_target_fraction: float
    gc_fraction_at_selection: float
    gc_absolute_difference: float
    gc_match_tolerance: float | None = None
    origin_ids: str | None = None
    location_preferred: bool = False
    center_distance_to_gene_bp: int | None = None
    center_distance_to_tss_bp: int | None = None


@dataclass(frozen=True)
class LocationPreference:
    """Vektorisierte Abstaende vom Kernfensterzentrum zu annotierten Basen."""
    mode: str
    distance_bp: int
    gene_intervals: Mapping[str, tuple[np.ndarray, np.ndarray]]
    tss_positions: Mapping[str, np.ndarray]

    def classify(self, chrom: str, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        centers = np.asarray(centers, dtype=np.int64)
        gene_distance = np.full(centers.shape, -1, dtype=np.int64)
        tss_distance = np.full(centers.shape, -1, dtype=np.int64)
        intervals = self.gene_intervals.get(chrom)
        if intervals is not None and len(intervals[0]):
            starts, ends = intervals
            previous = np.searchsorted(starts, centers, side="right") - 1
            have_previous = previous >= 0
            gene_distance[have_previous] = np.maximum(
                centers[have_previous] - (ends[previous[have_previous]] - 1), 0)
            following = previous + 1
            have_following = following < len(starts)
            right = starts[following[have_following]] - centers[have_following]
            current = gene_distance[have_following]
            gene_distance[have_following] = np.where(current < 0, right, np.minimum(current, right))
        positions = self.tss_positions.get(chrom)
        if positions is not None and len(positions):
            following = np.searchsorted(positions, centers, side="left")
            previous = following - 1
            have_previous = previous >= 0
            tss_distance[have_previous] = centers[have_previous] - positions[previous[have_previous]]
            have_following = following < len(positions)
            right = positions[following[have_following]] - centers[have_following]
            current = tss_distance[have_following]
            tss_distance[have_following] = np.where(current < 0, right, np.minimum(current, right))
        preferred = np.zeros(centers.shape, dtype=bool)
        if self.mode in {"gene", "gene-or-tss"}:
            preferred |= (gene_distance >= 0) & (gene_distance <= self.distance_bp)
        if self.mode in {"tss", "gene-or-tss"}:
            preferred |= (tss_distance >= 0) & (tss_distance <= self.distance_bp)
        return gene_distance, tss_distance, preferred


class ChromResolver:
    """Loest Chromosomennamen robust zwischen 1, Chr1 und chr1 auf."""

    def __init__(self, names: Iterable[str], source_name: str):
        self.source_name = source_name
        self.names = tuple(str(x) for x in names)
        self.exact = set(self.names)
        self.by_key: dict[str, str] = {}
        for name in self.names:
            key = canonical_chrom(name)
            if key in self.by_key and self.by_key[key] != name:
                raise ValueError(
                    f"Mehrdeutige Chromosomennamen in {source_name}: "
                    f"{self.by_key[key]!r} und {name!r} haben denselben "
                    f"kanonischen Namen {key!r}."
                )
            self.by_key[key] = name

    def resolve(self, requested: str) -> str:
        if requested in self.exact:
            return requested
        key = canonical_chrom(requested)
        try:
            return self.by_key[key]
        except KeyError as exc:
            raise KeyError(
                f"Chromosom {requested!r} (kanonisch {key!r}) fehlt in "
                f"{self.source_name}. Verfuegbar: {', '.join(self.names[:20])}"
            ) from exc


class NumpyJSONEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def canonical_chrom(name: str) -> str:
    value = str(name).strip()
    lower = value.lower()
    if lower.startswith("chromosome"):
        value = value[len("chromosome") :]
    elif lower.startswith("chr"):
        value = value[3:]
    value = value.strip()
    if re.fullmatch(r"0*\d+", value):
        return str(int(value))
    aliases = {
        "mt": "M",
        "mitochondria": "M",
        "mitochondrion": "M",
        "cp": "C",
        "pt": "C",
        "plastid": "C",
        "chloroplast": "C",
    }
    return aliases.get(value.lower(), value.upper())


def safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    safe = safe.strip("._")
    return safe or "unnamed"


def open_dependencies() -> tuple[Any, Any]:
    try:
        import pyBigWig  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Das Python-Paket 'pyBigWig' fehlt. Installation z.B. mit:\n"
            "  conda install -c conda-forge pybigwig\n"
            "oder:\n"
            "  python -m pip install pyBigWig"
        ) from exc
    try:
        import pysam  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Das Python-Paket 'pysam' fehlt. Installation z.B. mit:\n"
            "  conda install -c bioconda pysam\n"
            "oder:\n"
            "  python -m pip install pysam"
        ) from exc
    return pyBigWig, pysam


def open_bigwig(pybigwig_module: Any, path: Path) -> Any:
    handle = pybigwig_module.open(str(path))
    if handle is None:
        raise OSError(f"BigWig konnte nicht geoeffnet werden: {path}")
    if hasattr(handle, "isBigWig") and not handle.isBigWig():
        handle.close()
        raise ValueError(f"Datei ist keine BigWig: {path}")
    return handle


def build_chrom_bundles(
    chromosome_keys: Sequence[str],
    fasta: Any,
    fe_bw: Any,
    pval_bw: Any,
    summit_bw: Any,
) -> dict[str, ChromBundle]:
    fasta_resolver = ChromResolver(fasta.references, "FASTA")
    fe_chroms: Mapping[str, int] = fe_bw.chroms()
    pval_chroms: Mapping[str, int] = pval_bw.chroms()
    summit_chroms: Mapping[str, int] = summit_bw.chroms()
    fe_resolver = ChromResolver(fe_chroms.keys(), "FE-BigWig")
    pval_resolver = ChromResolver(pval_chroms.keys(), "p-Wert-BigWig")
    summit_resolver = ChromResolver(summit_chroms.keys(), "Summit-BigWig")

    bundles: dict[str, ChromBundle] = {}
    fasta_lengths = dict(zip(fasta.references, fasta.lengths))
    for key in sorted(set(chromosome_keys)):
        fasta_name = fasta_resolver.resolve(key)
        fe_name = fe_resolver.resolve(key)
        pval_name = pval_resolver.resolve(key)
        summit_name = summit_resolver.resolve(key)
        lengths = {
            "FASTA": int(fasta_lengths[fasta_name]),
            "FE-BigWig": int(fe_chroms[fe_name]),
            "p-Wert-BigWig": int(pval_chroms[pval_name]),
            "Summit-BigWig": int(summit_chroms[summit_name]),
        }
        if len(set(lengths.values())) != 1:
            details = ", ".join(f"{source}={length}" for source, length in lengths.items())
            raise ValueError(
                f"Uneinheitliche Chromosomenlaengen fuer {key}: {details}. "
                "Alle Dateien muessen auf derselben Referenz beruhen."
            )
        bundles[key] = ChromBundle(
            chrom_key=key,
            fasta_name=fasta_name,
            fe_name=fe_name,
            pval_name=pval_name,
            summit_name=summit_name,
            length=lengths["FASTA"],
        )
    return bundles


def load_summits(
    summit_bw: Any,
    bundles: Mapping[str, ChromBundle],
    min_value: float,
    wide_interval_policy: str,
) -> tuple[dict[str, list[Summit]], int]:
    result: dict[str, list[Summit]] = {}
    wide_count = 0
    for key, bundle in bundles.items():
        intervals = summit_bw.intervals(bundle.summit_name)
        summits: list[Summit] = []
        if intervals is not None:
            for start, end, score in intervals:
                start = int(start)
                end = int(end)
                score = float(score)
                if not math.isfinite(score) or score <= min_value:
                    continue
                width = end - start
                if width <= 0:
                    raise ValueError(
                        f"Ungueltiges Summit-Intervall in {bundle.summit_name}: "
                        f"[{start}, {end})."
                    )
                if width != 1:
                    wide_count += 1
                    if wide_interval_policy == "error":
                        raise ValueError(
                            "Die Summit-BigWig enthaelt ein Intervall mit mehr "
                            f"als einer Base: {bundle.summit_name}:{start}-{end}. "
                            "Erwartet werden 1-bp-Summits. Falls Plateaus bewusst "
                            "vorliegen, --wide-summit-policy midpoint verwenden."
                        )
                    position = start + (width - 1) // 2
                else:
                    position = start
                summits.append(
                    Summit(
                        chrom_key=key,
                        chrom_bigwig=bundle.summit_name,
                        position=position,
                        score=score,
                        interval_start=start,
                        interval_end=end,
                    )
                )
        summits.sort(key=lambda x: (x.position, -x.score))
        # Sicherheitshalber gleiche Summit-Koordinaten auf den hoechsten Wert reduzieren.
        deduplicated: dict[int, Summit] = {}
        for summit in summits:
            previous = deduplicated.get(summit.position)
            if previous is None or summit.score > previous.score:
                deduplicated[summit.position] = summit
        result[key] = sorted(deduplicated.values(), key=lambda x: x.position)
    return result, wide_count


def is_valid_window(center: int, window_size: int, chrom_length: int) -> tuple[bool, int, int]:
    half = window_size // 2
    start = center - half
    end = start + window_size
    return start >= 0 and end <= chrom_length, start, end


def bigwig_values(
    bigwig: Any,
    chrom: str,
    start: int,
    end: int,
    missing_value: float,
) -> np.ndarray:
    expected = end - start
    if expected < 0:
        raise ValueError(f"Ungueltiges BigWig-Intervall [{start}, {end}).")
    if expected == 0:
        return np.empty(0, dtype=np.float32)
    try:
        values = bigwig.values(chrom, int(start), int(end), numpy=True)
    except TypeError:
        values = bigwig.values(chrom, int(start), int(end))
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        array = np.full(expected, missing_value, dtype=np.float32)
    if array.size != expected:
        raise ValueError(
            f"BigWig lieferte fuer {chrom}:{start}-{end} {array.size} statt "
            f"{expected} Werte."
        )
    nan_mask = np.isnan(array)
    if np.any(nan_mask):
        array = array.copy()
        array[nan_mask] = missing_value
    return array


def rolling_max(values: np.ndarray, window_size: int) -> np.ndarray:
    if window_size <= 0 or window_size > values.size:
        raise ValueError("Ungueltige Fenstergroesse fuer rolling_max.")
    output = np.empty(values.size - window_size + 1, dtype=np.float32)
    queue: deque[int] = deque()
    for index, value in enumerate(values):
        while queue and values[queue[-1]] <= value:
            queue.pop()
        queue.append(index)
        oldest_allowed = index - window_size + 1
        while queue and queue[0] < oldest_allowed:
            queue.popleft()
        if index >= window_size - 1:
            output[index - window_size + 1] = values[queue[0]]
    return output


def merge_intervals(intervals: Sequence[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    if not intervals:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    ordered = sorted(intervals)
    merged: list[list[int]] = []
    for start, end in ordered:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return (
        np.asarray([item[0] for item in merged], dtype=np.int64),
        np.asarray([item[1] for item in merged], dtype=np.int64),
    )


def overlap_mask(
    starts: np.ndarray,
    ends: np.ndarray,
    merged_starts: np.ndarray,
    merged_ends: np.ndarray,
) -> np.ndarray:
    if merged_starts.size == 0:
        return np.zeros(starts.size, dtype=bool)
    indices = np.searchsorted(merged_starts, ends, side="left") - 1
    valid = indices >= 0
    result = np.zeros(starts.size, dtype=bool)
    result[valid] = merged_ends[indices[valid]] > starts[valid]
    return result


def gc_fraction_for_sequence(sequence: str) -> float:
    upper = sequence.upper()
    gc_count = upper.count("G") + upper.count("C")
    canonical_count = sum(upper.count(base) for base in "ACGT")
    return float(gc_count / canonical_count) if canonical_count else math.nan


def sample_gc_fractions(samples: Sequence[Any], fasta: Any) -> np.ndarray:
    fractions = np.empty(len(samples), dtype=np.float64)
    for index, sample in enumerate(samples):
        sequence = fasta.fetch(
            sample.bundle.fasta_name,
            int(sample.window_start),
            int(sample.window_end),
        )
        fraction = gc_fraction_for_sequence(sequence)
        if not math.isfinite(fraction):
            raise ValueError(
                f"Fenster {sample.bundle.fasta_name}:{sample.window_start}-"
                f"{sample.window_end} enthaelt keine A/C/G/T-Basen; GC-Matching "
                "ist nicht moeglich."
            )
        fractions[index] = fraction
    return fractions


def validate_window_start_range(
    bundle: ChromBundle,
    first_start: int,
    stop_start: int,
    window_size: int,
) -> int:
    if first_start < 0 or stop_start <= first_start:
        raise ValueError("Ungueltiger Bereich fuer Fensterstarts.")
    fetch_end = stop_start - 1 + window_size
    if fetch_end > bundle.length:
        raise ValueError("Fensterstartbereich reicht ueber das Chromosomenende hinaus.")
    return fetch_end


def window_pval_maxima(
    bundle: ChromBundle,
    pval_bw: Any,
    first_start: int,
    stop_start: int,
    window_size: int,
) -> np.ndarray:
    fetch_end = validate_window_start_range(
        bundle, first_start, stop_start, window_size
    )
    pval = bigwig_values(
        pval_bw,
        bundle.pval_name,
        first_start,
        fetch_end,
        missing_value=0.0,
    )
    pval_maxima = rolling_max(pval, window_size)
    if pval_maxima.size != stop_start - first_start:
        raise AssertionError("Interner Fehler: falsche Zahl berechneter p-Wert-Profile.")
    return pval_maxima


def window_gc_fractions(
    bundle: ChromBundle,
    fasta: Any,
    first_start: int,
    stop_start: int,
    window_size: int,
) -> np.ndarray:
    fetch_end = validate_window_start_range(
        bundle, first_start, stop_start, window_size
    )
    sequence = fasta.fetch(bundle.fasta_name, int(first_start), int(fetch_end)).upper()
    expected = fetch_end - first_start
    if len(sequence) != expected:
        raise ValueError(
            f"FASTA lieferte fuer {bundle.fasta_name}:{first_start}-{fetch_end} "
            f"{len(sequence)} statt {expected} Basen."
        )
    bases = np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)
    is_gc = (bases == ord("G")) | (bases == ord("C"))
    is_canonical = (
        (bases == ord("A"))
        | (bases == ord("C"))
        | (bases == ord("G"))
        | (bases == ord("T"))
    )
    gc_prefix = np.empty(bases.size + 1, dtype=np.int64)
    canonical_prefix = np.empty(bases.size + 1, dtype=np.int64)
    gc_prefix[0] = 0
    canonical_prefix[0] = 0
    np.cumsum(is_gc, dtype=np.int64, out=gc_prefix[1:])
    np.cumsum(is_canonical, dtype=np.int64, out=canonical_prefix[1:])
    gc_counts = gc_prefix[window_size:] - gc_prefix[:-window_size]
    canonical_counts = (
        canonical_prefix[window_size:] - canonical_prefix[:-window_size]
    )
    gc_fractions = np.full(gc_counts.size, np.nan, dtype=np.float64)
    np.divide(
        gc_counts,
        canonical_counts,
        out=gc_fractions,
        where=canonical_counts > 0,
    )

    expected_windows = stop_start - first_start
    if gc_fractions.size != expected_windows:
        raise AssertionError("Interner Fehler: falsche Zahl berechneter GC-Profile.")
    return gc_fractions


def window_metric_arrays(
    bundle: ChromBundle,
    fasta: Any,
    pval_bw: Any,
    first_start: int,
    stop_start: int,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Liefert p-Wert-Maximum und GC-Anteil fuer alle Starts in [first, stop)."""
    pval_maxima = window_pval_maxima(
        bundle, pval_bw, first_start, stop_start, window_size
    )
    gc_fractions = window_gc_fractions(
        bundle, fasta, first_start, stop_start, window_size
    )
    return pval_maxima, gc_fractions


def intervals_by_chrom(
    samples: Sequence[Any],
    chrom_keys: Iterable[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom_key in chrom_keys:
        intervals = [
            (sample.window_start, sample.window_end)
            for sample in samples
            if sample.bundle.chrom_key == chrom_key
        ]
        result[chrom_key] = merge_intervals(intervals)
    return result


def select_positive_samples(
    summits_by_chrom: Mapping[str, Sequence[Summit]],
    bundles: Mapping[str, ChromBundle],
    window_size: int,
) -> tuple[list[PositiveSample], list[dict[str, Any]], dict[str, int]]:
    """Ein positives Fenster fuer jeden gueltigen Summit, ohne Annotationsfilter."""
    positives: list[PositiveSample] = []
    audit: list[dict[str, Any]] = []
    stats: Counter[str] = Counter()
    for chrom_key, summits in sorted(summits_by_chrom.items()):
        bundle = bundles[chrom_key]
        for summit in summits:
            stats["summits_considered"] += 1
            valid, start, end = is_valid_window(summit.position, window_size, bundle.length)
            if valid:
                positives.append(PositiveSample(summit, bundle, start, end))
                stats["positive_samples"] += 1
            else:
                stats["excluded_window_outside_chromosome"] += 1
            audit.append({
                "chromosome": bundle.fasta_name,
                "summit_position_0based": summit.position,
                "summit_score": summit.score,
                "window_start_0based": start,
                "window_end_0based_exclusive": end,
                "status": "included_positive" if valid else "excluded_window_outside_chromosome",
            })
    positives.sort(key=lambda sample: (sample.bundle.chrom_key, sample.window_start))
    stats.setdefault("positive_samples", 0)
    stats.setdefault("excluded_window_outside_chromosome", 0)
    return positives, audit, dict(stats)


def load_location_preference(
    args: argparse.Namespace, bundles: Mapping[str, ChromBundle],
) -> tuple[LocationPreference | None, dict[str, Any]]:
    """Liest Annotation nur fuer die weiche Bevorzugung negativer Zentren."""
    mode = args.negative_location_preference
    summary: dict[str, Any] = {
        "mode": mode, "distance_bp": args.negative_preference_distance,
        "distance_reference": "original_core_window_center_to_nearest_annotated_base",
        "within_gene_distance_bp": 0, "distance_threshold_inclusive": True,
        "hard_location_filter": False,
        "strategy": "preferred_first_within_gc_bin_then_distant_fallback",
    }
    if mode == "none":
        return None, summary
    gene_ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
    tss_points: dict[str, list[int]] = defaultdict(list)

    def open_annotation(path: Path) -> Any:
        return gzip.open(path, "rt", encoding="utf-8") if path.name.lower().endswith(".gz") else path.open("rt", encoding="utf-8")

    if mode in {"gene", "gene-or-tss"}:
        accepted = {value.strip().lower() for value in args.gene_feature_types.split(",") if value.strip()}
        counts: Counter[str] = Counter()
        ignored = 0
        with open_annotation(args.gene_annotation) as handle:
            for line_number, line in enumerate(handle, 1):
                if line.startswith("##FASTA"):
                    break
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.rstrip("\n\r").split("\t")
                if len(fields) != 9:
                    raise ValueError(f"{args.gene_annotation}:{line_number}: GFF3/GTF mit 9 Feldern erwartet.")
                if fields[2].lower() not in accepted:
                    continue
                chrom = canonical_chrom(fields[0])
                if chrom not in bundles:
                    ignored += 1
                    continue
                try:
                    start, end = int(fields[3]) - 1, int(fields[4])
                except ValueError as exc:
                    raise ValueError(f"{args.gene_annotation}:{line_number}: Ungueltige Genkoordinaten.") from exc
                if not 0 <= start < end <= bundles[chrom].length:
                    raise ValueError(f"{args.gene_annotation}:{line_number}: Gen ausserhalb des Chromosoms: [{start}, {end}).")
                gene_ranges[chrom].append((start, end))
                counts[fields[2]] += 1
        if not gene_ranges:
            raise ValueError("Keine passenden Gene auf den ausgewaehlten Chromosomen gefunden.")
        summary["gene_annotation"] = str(args.gene_annotation)
        summary["gene_feature_types"] = sorted(accepted)
        summary["loaded_gene_features"] = sum(counts.values())
        summary["gene_feature_counts"] = dict(counts)
        summary["gene_features_on_other_chromosomes"] = ignored
    if mode in {"tss", "gene-or-tss"}:
        for path, expected_strand in ((args.tss_plus, "+"), (args.tss_minus, "-")):
            loaded, ignored = 0, 0
            with open_annotation(path) as handle:
                for line_number, line in enumerate(handle, 1):
                    stripped = line.strip()
                    if not stripped or stripped.startswith(("#", "track", "browser")):
                        continue
                    fields = stripped.split()
                    if len(fields) < 6:
                        raise ValueError(f"{path}:{line_number}: Mindestens BED6 erwartet.")
                    chrom = canonical_chrom(fields[0])
                    if chrom not in bundles:
                        ignored += 1
                        continue
                    try:
                        start, end = int(fields[1]), int(fields[2])
                    except ValueError as exc:
                        raise ValueError(f"{path}:{line_number}: Ungueltige TSS-Koordinaten.") from exc
                    if fields[5] != expected_strand:
                        raise ValueError(f"{path}:{line_number}: Erwarteter TSS-Strang {expected_strand!r}.")
                    if not 0 <= start < end <= bundles[chrom].length:
                        raise ValueError(f"{path}:{line_number}: TSS ausserhalb des Chromosoms.")
                    # BED ist halb-offen: 5'-Base am Plus-Start bzw. Minus-Ende-1.
                    tss_points[chrom].append(start if expected_strand == "+" else end - 1)
                    loaded += 1
            summary[f"tss_{'plus' if expected_strand == '+' else 'minus'}"] = {
                "path": str(path), "loaded_records": loaded, "records_on_other_chromosomes": ignored}
        if not tss_points:
            raise ValueError("Keine TSS auf den ausgewaehlten Chromosomen gefunden.")
    genes = {chrom: merge_intervals(ranges) for chrom, ranges in gene_ranges.items()}
    tss = {chrom: np.unique(np.asarray(points, dtype=np.int64)) for chrom, points in tss_points.items()}
    summary["chromosomes_with_genes"] = sorted(genes)
    summary["chromosomes_with_tss"] = sorted(tss)
    summary["unique_tss_positions"] = sum(len(points) for points in tss.values())
    return LocationPreference(mode, args.negative_preference_distance, genes, tss), summary


def build_dense_genome_pool(
    positive_samples: Sequence[PositiveSample],
    bundles: Mapping[str, ChromBundle], fasta: Any, pval_bw: Any,
    window_size: int, pval_threshold: float, scan_chunk_size: int,
    store: CandidateStore,
    location_preference: LocationPreference | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prueft jeden vollstaendigen Start genau einmal, ungeachtet seiner Lage."""
    exclusions = intervals_by_chrom(positive_samples, bundles)
    audit: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    # Begrenzte Listen verhindern, dass alle Kandidaten gleichzeitig im RAM liegen.
    chunk_size = min(scan_chunk_size, 50_000)
    for chrom_key, bundle in sorted(bundles.items()):
        stop_start = max(0, bundle.length - window_size + 1)
        stats = {key: 0 for key in (
            "tested_windows", "pval_valid_windows", "hard_valid_windows",
            "gc_measurable_windows", "rejected_pval", "rejected_positive_overlap",
            "rejected_unmeasurable_gc", "preferred_candidate_windows",
        )}
        before = store.count
        LOGGER.info("Pruefe %s: %s moegliche negative Starts", bundle.fasta_name, f"{stop_start:,}")
        for first in range(0, stop_start, chunk_size):
            stop = min(stop_start, first + chunk_size)
            maxima, gc = window_metric_arrays(bundle, fasta, pval_bw, first, stop, window_size)
            starts = np.arange(first, stop, dtype=np.int64)
            pval_valid = maxima <= pval_threshold
            no_positive_overlap = ~overlap_mask(starts, starts + window_size, *exclusions[chrom_key])
            hard_valid = pval_valid & no_positive_overlap
            measurable = hard_valid & np.isfinite(gc)
            preferred = (location_preference.classify(chrom_key, starts + window_size // 2)[2]
                         if location_preference is not None else np.zeros(starts.shape, dtype=bool))
            stats["preferred_candidate_windows"] += int(np.count_nonzero(measurable & preferred))
            stats["tested_windows"] += int(starts.size)
            stats["pval_valid_windows"] += int(np.count_nonzero(pval_valid))
            stats["hard_valid_windows"] += int(np.count_nonzero(hard_valid))
            stats["gc_measurable_windows"] += int(np.count_nonzero(measurable))
            # Die Ablehnungszaehler sind disjunkt: erst p-Wert, dann Ueberlappung, dann GC.
            stats["rejected_pval"] += int(np.count_nonzero(~pval_valid))
            stats["rejected_positive_overlap"] += int(np.count_nonzero(pval_valid & ~no_positive_overlap))
            stats["rejected_unmeasurable_gc"] += int(np.count_nonzero(hard_valid & ~np.isfinite(gc)))
            store.add_many([WindowCandidate(
                chrom_key=chrom_key, window_start=int(starts[index]),
                selection_pval_max=float(maxima[index]), gc_fraction=float(gc[index]),
                origin_region=bundle.fasta_name,
                location_preferred=bool(preferred[index]),
            ) for index in np.flatnonzero(measurable)])
        totals.update(stats)
        audit.append({"chromosome": bundle.fasta_name, "chromosome_length": bundle.length,
                      **stats, "pool_candidate_windows": store.count - before})
    store.finish()
    summary = {"pool_mode": "all_genomic_starts_1bp_then_hard_filters",
               "candidate_pool_size": store.count, "candidate_grid_step": 1, **dict(totals)}
    if summary["tested_windows"] != (summary["rejected_pval"]
            + summary["rejected_positive_overlap"] + summary["rejected_unmeasurable_gc"] + store.count):
        raise AssertionError("Kandidatenbilanz ist unvollstaendig.")
    return audit, summary


def materialize_matched_samples(
    matches: Sequence[tuple[int, WindowCandidate, float]],
    bundles: Mapping[str, ChromBundle], window_size: int,
    location_preference: LocationPreference | None = None,
) -> tuple[list[NegativeSample], list[dict[str, Any]]]:
    negatives: list[NegativeSample] = []
    audit: list[dict[str, Any]] = []
    for candidate_id, candidate, target_gc in matches:
        bundle = bundles[candidate.chrom_key]
        start, end = candidate.window_start, candidate.window_start + window_size
        origin = f"{bundle.fasta_name}:{start}-{end}"
        difference = abs(candidate.gc_fraction - target_gc)
        gene_distance = tss_distance = None
        if location_preference is not None:
            genes, tss, preferred = location_preference.classify(
                candidate.chrom_key, np.asarray([start + window_size // 2]))
            gene_distance = int(genes[0]) if genes[0] >= 0 else None
            tss_distance = int(tss[0]) if tss[0] >= 0 else None
            if bool(preferred[0]) != candidate.location_preferred:
                raise AssertionError("Lageprioritaet hat sich zwischen Pool und Auswahl geaendert.")
        negatives.append(NegativeSample(
            bundle=bundle, center=start + window_size // 2, window_start=start, window_end=end,
            selection_pval_max=candidate.selection_pval_max, gc_target_fraction=float(target_gc),
            gc_fraction_at_selection=candidate.gc_fraction, gc_absolute_difference=difference,
            origin_ids=origin,
            location_preferred=candidate.location_preferred,
            center_distance_to_gene_bp=gene_distance,
            center_distance_to_tss_bp=tss_distance,
        ))
        audit.append({"candidate_id": candidate_id, "chromosome": bundle.fasta_name,
                      "window_start_0based": start, "window_end_0based_exclusive": end,
                      "target_gc_fraction": target_gc, "matched_gc_fraction": candidate.gc_fraction,
                      "absolute_gc_difference": difference, "selection_pval_max": candidate.selection_pval_max,
                      "gc_target_role": "post_selection_quantile_audit_only",
                      "negative_location_preferred": int(candidate.location_preferred),
                      "center_distance_to_gene_bp": gene_distance,
                      "center_distance_to_tss_bp": tss_distance})
    negatives.sort(key=lambda sample: (sample.bundle.chrom_key, sample.window_start))
    return negatives, audit


def unique_sample_ids(samples: Sequence[Any]) -> dict[int, str]:
    ids: dict[int, str] = {}
    seen: set[str] = set()
    for sample in samples:
        group = "positive" if isinstance(sample, PositiveSample) else "negative"
        sample_id = f"{group}_{safe_filename(sample.bundle.fasta_name)}_{sample.window_start}_{sample.window_end}"
        if sample_id in seen:
            raise AssertionError(f"Doppeltes Fenster innerhalb einer Klasse: {sample_id}")
        seen.add(sample_id)
        ids[id(sample)] = sample_id
    return ids


def sample_base_metadata(
    sample: Any, sample_id: str, sample_type: str, window_size: int,
    center_index: int | None = None,
) -> dict[str, Any]:
    positive = isinstance(sample, PositiveSample)
    metadata = {
        "sample_id": sample_id, "label": int(positive),
        "group": "positive" if positive else "negative", "sample_type": sample_type,
        "chromosome": sample.bundle.fasta_name,
        "window_start_0based": sample.window_start, "window_end_0based_exclusive": sample.window_end,
        "window_start_1based": sample.window_start + 1, "window_end_1based_inclusive": sample.window_end,
        "window_length": window_size, "center_position_0based": sample.center,
        "center_position_1based": sample.center + 1,
        "center_index_0based": window_size // 2 if center_index is None else center_index,
        "origin_ids": f"{sample.bundle.fasta_name}:{sample.window_start}-{sample.window_end}",
        "summit_position_0based": sample.summit.position if positive else None,
        "summit_position_1based": sample.summit.position + 1 if positive else None,
        "summit_score": sample.summit.score if positive else None,
        "summit_index_0based": sample.summit_index if positive else None,
        "genomic_shift_center_minus_summit": 0 if positive else None,
        "selection_pval_max": None if positive else sample.selection_pval_max,
        "gc_target_fraction": None if positive else sample.gc_target_fraction,
        "gc_fraction_at_selection": None if positive else sample.gc_fraction_at_selection,
        "gc_absolute_difference": None if positive else sample.gc_absolute_difference,
        "gc_target_role": None if positive else "post_selection_quantile_audit_only",
        "negative_location_preferred": None if positive else int(sample.location_preferred),
        "center_distance_to_gene_bp": None if positive else sample.center_distance_to_gene_bp,
        "center_distance_to_tss_bp": None if positive else sample.center_distance_to_tss_bp,
    }
    return metadata


def write_candidate_store(
    path: Path, store: CandidateStore, bundles: Mapping[str, ChromBundle],
    window_size: int,
) -> None:
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "candidate_id", "chromosome", "window_start_0based",
            "window_end_0based_exclusive", "gc_fraction", "selection_pval_max",
            "origin_ids",
            "negative_location_preferred",
        ])
        for candidate_id, chrom, start, pval, gc, preferred in store.db.execute(
            "SELECT id,chrom,start,pval,gc,location_preferred FROM candidates ORDER BY chrom,start,id"
        ):
            writer.writerow([
                candidate_id, bundles[chrom].fasta_name, start,
                start + window_size, f"{gc:.17g}", f"{pval:.17g}",
                store.source_names(candidate_id),
                int(preferred),
            ])


def write_gc_capacity_plot(
    output_dir: Path, group: str, positive_gc: Sequence[float],
    store: CandidateStore, bin_width: float, window_size: int, tolerance: float,
) -> dict[str, Any]:
    """Zeigt die kumulative GC-Kapazitaet sofort nach Aufbau des Pools.

    Oben je GC-Schwelle die maximal gestuetzte Zahl Negativer von unten
    (GC <= t) und von oben (GC >= t), streng und mit CDF-Toleranz; die
    tiefste Stelle ist der Engpass. Unten die Positivverteilung und die
    Kapazitaet je Bereich fuer sich als optimistischer Vergleich.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("GC-Diagramme benoetigen matplotlib.") from exc
    directory = output_dir / "distributions"
    directory.mkdir(parents=True, exist_ok=True)
    capacity = store.gc_capacity(window_size, positive_gc, tolerance)
    edges = np.r_[0.0, store.edges, 1.0]
    n_bins = len(edges) - 1
    p_counts = np.bincount(capacity["positive_bins"], minlength=n_bins)
    n_pos = capacity["positive_count"]
    p_frac = p_counts / n_pos
    cdf = np.cumsum(p_frac)
    low = capacity["capacity_gc_at_most"]
    high = capacity["capacity_gc_at_least"]
    low_keys = sorted(low)
    high_keys = sorted(high)
    low_x = np.array([edges[k + 1] for k in low_keys])
    high_x = np.array([edges[k] for k in high_keys])

    def supported(side_keys: Sequence[int], values: Mapping[int, int],
                  from_low: bool, eps: float) -> np.ndarray:
        out = []
        for k in side_keys:
            share = cdf[k] if from_low else 1.0 - (cdf[k - 1] if k else 0.0)
            need = share - eps
            out.append(values[k] / need if need > 0 else np.nan)
        return np.asarray(out, dtype=np.float64)

    table_rows = [{
        "bin_index": row["bin_index"], "bin_start": float(edges[row["bin_index"]]),
        "bin_end": float(edges[row["bin_index"] + 1]),
        **{key: (None if isinstance(value, float) and math.isinf(value) else value)
           for key, value in row.items() if key != "bin_index"},
    } for row in capacity["rows"]]
    table = directory / f"{group}_gc_capacity.tsv"
    write_table(table, table_rows)

    relevant = np.flatnonzero(p_counts > 0)
    margin = max(0.02, 2 * bin_width)
    x_min = max(0.0, edges[relevant[0]] - margin)
    x_max = min(1.0, edges[relevant[-1] + 1] + margin)
    plot = directory / f"{group}_gc_capacity.svg"
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1.3, 0.8]})
    ax = axes[0]
    for eps, style, alpha in ((capacity["strict_tolerance"], "-", 1.0),
                              (tolerance, "--", 0.9)):
        label_eps = "streng" if eps == 0.0 else f"CDF-Toleranz {eps:g}"
        ax.plot(low_x, supported(low_keys, low, True, eps), style, color="#0072B2",
                alpha=alpha, linewidth=2, label=f"von unten (GC ≤ t), {label_eps}")
        ax.plot(high_x, supported(high_keys, high, False, eps), style, color="#D55E00",
                alpha=alpha, linewidth=2, label=f"von oben (GC ≥ t), {label_eps}")
    for value, color, text, offset in (
        (capacity["bound_strict"], "#0072B2",
         f"Obergrenze streng: {capacity['bound_strict']:,.0f}", (4, -11)),
        (capacity["bound_at_tolerance"], "#7B61A8",
         f"Obergrenze bei Toleranz {tolerance:g}: {capacity['bound_at_tolerance']:,.0f}", (4, 3)),
        (n_pos, "#555555", f"Positive: {n_pos:,}", (4, 3)),
    ):
        if math.isfinite(value) and value > 0:
            ax.axhline(value, color=color, linestyle=":", linewidth=1.2)
            ax.annotate(text, xy=(x_min, value), xytext=offset,
                        textcoords="offset points", fontsize=8.5, color=color)
    for key, color in (("bottleneck_strict", "#0072B2"),
                       ("bottleneck_at_tolerance", "#7B61A8")):
        info = capacity[key]
        if info["bin_index"] is not None:
            x = edges[info["bin_index"] + 1] if info["side"] == "low" else edges[info["bin_index"]]
            ax.axvline(x, color=color, linewidth=0.9, alpha=0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Maximal gestützte Zahl Negativer")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.grid(alpha=0.25, which="both")

    ax2 = axes[1]
    ax2.stairs(p_frac, edges, linewidth=2.5, color="#0072B2", label=f"Positive (n={n_pos})")
    ax2.set_ylabel("Relativer GC-Anteil\nder Positiven")
    ax2.set_xlabel("GC-Schwelle t")
    ax2.grid(axis="y", alpha=0.25)
    standalone = np.asarray(capacity["standalone_bin_capacity"], dtype=np.float64)
    ax3 = ax2.twinx()
    ax3.stairs(np.where(standalone > 0, standalone, np.nan), edges, linewidth=1.4,
               color="#7B61A8", linestyle="--",
               label="Kandidaten je Bereich, nicht überlappend (nur dieser Bereich)")
    ax3.set_yscale("log")
    ax3.set_ylabel("Fenster je GC-Bereich", color="#7B61A8")
    handles, labels = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    ax2.legend(handles + handles3, labels + labels3, loc="upper right", fontsize=8.5)
    ax2.set_xlim(x_min, x_max)
    fig.suptitle(f"{group}: kumulative GC-Kapazität der Negativkandidaten")
    fig.text(0.5, 0.01,
             "Kurve = Kapazität nicht überlappender Kandidaten mit GC ≤ t (bzw. ≥ t), geteilt "
             "durch den Anteil der Positiven dort.\nDie niedrigste Stelle ist der Engpass; keine "
             "gültige Auswahl kann größer sein als das Minimum"
             + ("; pro TSS höchstens ein Negativ ist nicht enthalten." if group == "tss" else "."),
             ha="center", va="bottom", fontsize=8.5, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(plot)
    plt.close(fig)
    LOGGER.info(
        "%s: GC-Kapazitaet - Obergrenze streng %.0f, bei Toleranz %.2f %.0f, "
        "Engpass %s (%d Positive umgebucht); Diagramm: %s",
        group, capacity["bound_strict"], tolerance, capacity["bound_at_tolerance"],
        capacity["bottleneck_at_tolerance"], capacity["positives_reassigned_to_nearest_bin"],
        plot,
    )
    capacity["outputs"] = {
        "capacity_table": str(table.relative_to(output_dir)),
        "capacity_svg": str(plot.relative_to(output_dir)),
    }
    return capacity


def write_gc_coverage_plots(
    output_dir: Path, group: str, positive_gc: Sequence[float],
    matches: Sequence[tuple[int, WindowCandidate, float]],
    store: CandidateStore, capacities: Sequence[int],
    bin_width: float, ks_distance: float,
    capacity: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Schreibt GC-Vergleich und Ergebnisdiagramm nach dem Matching."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("GC-Diagramme benoetigen matplotlib.") from exc
    directory = output_dir / "distributions"
    pos = np.asarray(positive_gc, dtype=float)
    neg = np.asarray([candidate.gc_fraction for _, candidate, _ in matches])
    edges = np.r_[0.0, store.edges, 1.0]
    centers = (edges[:-1] + edges[1:]) / 2.0
    n_bins = len(centers)
    p_counts = np.bincount(np.searchsorted(store.edges, pos, side="right"), minlength=n_bins)
    s_counts = np.bincount(np.searchsorted(store.edges, neg, side="right"), minlength=n_bins)
    c_counts = np.asarray(capacity["raw_bin_counts"], dtype=np.int64)
    capacity_rows = capacity["rows"]
    p_frac = p_counts / len(pos)
    s_frac = s_counts / len(neg)
    c_frac = c_counts / store.count

    def finite_or_none(value: Any) -> Any:
        return None if isinstance(value, float) and math.isinf(value) else value

    rows = [{
        "bin_index": b, "bin_start": float(edges[b]), "bin_end": float(edges[b + 1]),
        "bin_center": float(centers[b]), "positive_count": int(p_counts[b]),
        "candidate_pool_count": int(c_counts[b]), "selected_count": int(s_counts[b]),
        "nonoverlapping_capacity_upper_bound": int(capacities[b]),
        "cumulative_capacity_gc_at_most_bin": capacity_rows[b]["cumulative_capacity_gc_at_most_bin"],
        "cumulative_capacity_gc_at_least_bin": capacity_rows[b]["cumulative_capacity_gc_at_least_bin"],
        "supported_negatives_from_low_tail": finite_or_none(
            capacity_rows[b]["supported_negatives_from_low_tail"]),
        "supported_negatives_from_high_tail": finite_or_none(
            capacity_rows[b]["supported_negatives_from_high_tail"]),
        "supported_negatives_from_low_tail_strict": finite_or_none(
            capacity_rows[b]["supported_negatives_from_low_tail_strict"]),
        "supported_negatives_from_high_tail_strict": finite_or_none(
            capacity_rows[b]["supported_negatives_from_high_tail_strict"]),
        "positive_fraction": float(p_frac[b]),
        "candidate_pool_fraction": float(c_frac[b]), "selected_fraction": float(s_frac[b]),
        "candidate_minus_positive_fraction": float(c_frac[b] - p_frac[b]),
        "selected_minus_positive_fraction": float(s_frac[b] - p_frac[b]),
    } for b in range(n_bins)]
    data_path = directory / f"{group}_gc_before_after.tsv"
    raw_path = directory / f"{group}_gc_values.tsv"
    write_table(data_path, rows)
    write_table(raw_path, [
        *({"set": "positive", "gc_fraction": float(x)} for x in pos),
        *({"set": "selected", "gc_fraction": float(x)} for x in neg),
    ], ["set", "gc_fraction"])

    after = directory / f"{group}_gc_after_matching.svg"
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1.1, 1.2, 0.6]})
    # Eine breite volle und eine schmale gestrichelte Linie bleiben auch bei
    # exakt deckungsgleichen Kurven beide sichtbar.
    axes[0].stairs(p_frac, edges, label=f"Positive (n={len(pos)})",
                   linewidth=3, color="#0072B2", zorder=2)
    axes[0].stairs(s_frac, edges, label=f"Negative (n={len(neg)})",
                   linewidth=1.8, linestyle="--", color="#D55E00", zorder=3)
    axes[0].set_ylabel("Relativer Anteil je GC-Bereich")
    axes[0].legend()
    axes[0].set_title(f"{group}: GC nach Matching (gebinnter CDF-Abstand {ks_distance:.4f})")
    for values, label, color, width, style in (
        (pos, "Positive", "#0072B2", 3, "-"),
        (neg, "Negative", "#D55E00", 1.8, "--"),
    ):
        ordered = np.sort(values)
        axes[1].step(ordered, np.arange(1, len(ordered) + 1) / len(ordered),
                     where="post", label=label, color=color,
                     linewidth=width, linestyle=style)
    axes[1].set_ylabel("Kumulative Verteilung (ECDF)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    support = np.unique(np.concatenate((pos, neg)))
    pos_cdf = np.searchsorted(np.sort(pos), support, side="right") / len(pos)
    neg_cdf = np.searchsorted(np.sort(neg), support, side="right") / len(neg)
    difference = neg_cdf - pos_cdf
    axes[2].axhline(0, linestyle=":", linewidth=1.1, color="#555555")
    axes[2].fill_between(support, difference, 0, step="post",
                         color="#CC79A7", alpha=0.22)
    axes[2].step(support, difference, where="post", color="#9A4A82", linewidth=1.8)
    axes[2].set_ylabel("Negativ minus\nPositiv (ECDF)")
    axes[2].set_xlabel("GC-Anteil")
    axes[2].set_title(f"Kumulierte Abweichung (Maximum: {np.max(np.abs(difference)):.4f})")
    axes[2].grid(alpha=0.25)
    margin = max(0.02, 2 * bin_width)
    axes[2].set_xlim(max(0, min(np.min(pos), np.min(neg)) - margin),
                     min(1, max(np.max(pos), np.max(neg)) + margin))
    fig.tight_layout()
    fig.savefig(after)
    plt.close(fig)
    return rows, {
        "comparison_table": str(data_path.relative_to(output_dir)),
        "raw_gc_values": str(raw_path.relative_to(output_dir)),
        "capacity_table": capacity["outputs"]["capacity_table"],
        "capacity_svg": capacity["outputs"]["capacity_svg"],
        "after_matching_svg": str(after.relative_to(output_dir)),
    }


def ensure_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            if any(path.iterdir()):
                raise FileExistsError(
                    f"Ausgabeordner existiert und ist nicht leer: {path}. "
                    "--overwrite verwenden, um ihn zu ersetzen."
                )
        else:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def format_scalar(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.10g}"
    return str(value)


def dinucleotide_shuffle_sequence(
    sequence: str,
    rng: np.random.Generator,
    max_attempts: int = 20,
) -> str:
    """Randomisiert eine Sequenz als Euler-Pfad mit exakt gleichen Dinukleotiden."""
    if len(sequence) < 3:
        return sequence

    expected_dinucleotides = Counter(zip(sequence, sequence[1:]))
    shuffled = sequence
    for _ in range(max_attempts):
        outgoing: dict[str, list[str]] = defaultdict(list)
        for left, right in zip(sequence, sequence[1:]):
            outgoing[left].append(right)
        for targets in outgoing.values():
            rng.shuffle(targets)

        stack = [sequence[0]]
        reversed_path: list[str] = []
        while stack:
            current = stack[-1]
            targets = outgoing.get(current)
            if targets:
                stack.append(targets.pop())
            else:
                reversed_path.append(stack.pop())
        shuffled = "".join(reversed(reversed_path))

        if len(shuffled) != len(sequence):
            raise AssertionError("Interner Fehler: Dinukleotid-Shuffle hat falsche Laenge.")
        if Counter(zip(shuffled, shuffled[1:])) != expected_dinucleotides:
            raise AssertionError("Interner Fehler: Dinukleotide wurden nicht erhalten.")
        if shuffled[0] != sequence[0] or shuffled[-1] != sequence[-1]:
            raise AssertionError("Interner Fehler: Sequenzenden wurden nicht erhalten.")
        if shuffled != sequence:
            break
    return shuffled


def sample_shuffle_rng(
    seed: int,
    group: str,
    chromosome: str,
    sample_id: str,
) -> np.random.Generator:
    # Der stabile Hash macht jedes Sample unabhaengig von der Schreibreihenfolge.
    payload = f"{seed}\0{group}\0{chromosome}\0{sample_id}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return np.random.default_rng(derived_seed)


def write_window_file(
    path: Path,
    sequence: str,
    fe_signal: np.ndarray,
    pval_signal: np.ndarray,
    context_start: int,
    center_index: int,
    core_start_index: int,
    core_end_index: int,
    compress: bool,
) -> None:
    """Schreibt das innere Fenster samt Umfeld; eine Zeile je Basenposition.

    ``context_start`` ist die genomische 0-basierte Position der ersten Zeile
    (auch wenn sie aufgefuellt ist), ``center_index`` der Index des Fenster-
    zentrums, ``core_start_index``/``core_end_index`` das innere Fenster.
    """
    opener = gzip.open if compress else open
    mode = "wt"
    with opener(path, mode, encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            [
                "window_index_0based",
                "relative_to_center",
                "genomic_position_0based",
                "genomic_position_1based",
                "base",
                "FE",
                "minus_log10_p",
                "is_center",
                "in_core_window",
            ]
        )
        for index, base in enumerate(sequence):
            genomic_position = context_start + index
            writer.writerow(
                [
                    index,
                    index - center_index,
                    genomic_position,
                    genomic_position + 1,
                    base,
                    f"{float(fe_signal[index]):.8g}",
                    f"{float(pval_signal[index]):.8g}",
                    1 if index == center_index else 0,
                    1 if core_start_index <= index < core_end_index else 0,
                ]
            )


def write_info_file(
    path: Path, metadata: Mapping[str, Any],
    class_definitions: Mapping[str, str],
) -> None:
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["key", "value"])
        for name, definition in class_definitions.items():
            writer.writerow([f"definition_{name}", definition])
        for key, value in metadata.items():
            writer.writerow([key, format_scalar(value)])


@dataclass(frozen=True)
class ContextData:
    """Sequenz und Signale des inneren Fensters samt Umfeld."""

    sequence: str
    fe: np.ndarray
    pval: np.ndarray
    context_start: int
    context_end: int
    padded_left: int
    padded_right: int
    core_start_index: int
    core_end_index: int

    @property
    def core_sequence(self) -> str:
        return self.sequence[self.core_start_index:self.core_end_index]

    @property
    def core_fe(self) -> np.ndarray:
        return self.fe[self.core_start_index:self.core_end_index]

    @property
    def core_pval(self) -> np.ndarray:
        return self.pval[self.core_start_index:self.core_end_index]

    @property
    def flank_pval_max(self) -> float:
        flanks = np.concatenate(
            (self.pval[:self.core_start_index], self.pval[self.core_end_index:])
        )
        return float(np.max(flanks)) if flanks.size else 0.0

    @property
    def flank_fe_max(self) -> float:
        flanks = np.concatenate(
            (self.fe[:self.core_start_index], self.fe[self.core_end_index:])
        )
        return float(np.max(flanks)) if flanks.size else 0.0


def fetch_sample_data(
    sample: Any, fasta: Any, fe_bw: Any, pval_bw: Any, context_per_side: int,
) -> ContextData:
    """Holt Fenster plus Umfeld; ueber das Chromosom hinaus wird mit N/0 aufgefuellt."""
    if context_per_side < 0:
        raise ValueError("Umfeld je Seite darf nicht negativ sein.")
    window_start = int(sample.window_start)
    window_end = int(sample.window_end)
    chrom_length = int(sample.bundle.length)
    context_start = window_start - context_per_side
    context_end = window_end + context_per_side
    fetch_start = max(0, context_start)
    fetch_end = min(chrom_length, context_end)
    padded_left = fetch_start - context_start
    padded_right = context_end - fetch_end
    sequence = fasta.fetch(sample.bundle.fasta_name, fetch_start, fetch_end).upper()
    expected = fetch_end - fetch_start
    if len(sequence) != expected:
        raise ValueError(
            f"FASTA lieferte fuer {sample.bundle.fasta_name}:"
            f"{fetch_start}-{fetch_end} {len(sequence)} statt {expected} Basen."
        )
    fe = bigwig_values(
        fe_bw, sample.bundle.fe_name, fetch_start, fetch_end, missing_value=0.0,
    )
    pval = bigwig_values(
        pval_bw, sample.bundle.pval_name, fetch_start, fetch_end, missing_value=0.0,
    )
    if padded_left or padded_right:
        sequence = "N" * padded_left + sequence + "N" * padded_right
        fe = np.concatenate((np.zeros(padded_left), fe, np.zeros(padded_right)))
        pval = np.concatenate((np.zeros(padded_left), pval, np.zeros(padded_right)))
    core_start_index = window_start - context_start
    core_end_index = core_start_index + (window_end - window_start)
    if len(sequence) != context_end - context_start:
        raise AssertionError("Interner Fehler: Umfeldlaenge stimmt nicht.")
    return ContextData(
        sequence=sequence, fe=np.asarray(fe, dtype=np.float64),
        pval=np.asarray(pval, dtype=np.float64),
        context_start=context_start, context_end=context_end,
        padded_left=padded_left, padded_right=padded_right,
        core_start_index=core_start_index, core_end_index=core_end_index,
    )


def write_samples(
    output_dir: Path,
    sample_type: str,
    samples: Sequence[Any],
    sample_ids: Mapping[int, str],
    fasta: Any,
    fe_bw: Any,
    pval_bw: Any,
    window_size: int,
    pval_threshold: float,
    compress: bool,
    dinucleotide_shuffle: bool,
    dinucleotide_shuffle_seed: int,
    class_definitions: Mapping[str, str],
    context_per_side: int,
) -> list[dict[str, Any]]:
    output_subdirs = {"positive": Path("positive"), "negative": Path("negative")}
    try:
        output_subdir = output_subdirs[sample_type]
    except KeyError as exc:
        raise ValueError(f"Unbekannter Ausgabe-Sample-Typ: {sample_type}") from exc
    group = sample_type
    metadata_rows: list[dict[str, Any]] = []
    for index, sample in enumerate(samples, start=1):
        chrom_folder = output_dir / output_subdir / safe_filename(sample.bundle.fasta_name)
        chrom_folder.mkdir(parents=True, exist_ok=True)
        sample_id = sample_ids[id(sample)]
        extension = ".window.tsv.gz" if compress else ".window.tsv"
        window_path = chrom_folder / f"{sample_id}{extension}"
        info_path = chrom_folder / f"{sample_id}.info.tsv"

        context = fetch_sample_data(sample, fasta, fe_bw, pval_bw, context_per_side)
        sequence = context.sequence
        fe = context.fe
        pval = context.pval
        original_sequence = sequence
        if dinucleotide_shuffle:
            shuffle_rng = sample_shuffle_rng(
                seed=dinucleotide_shuffle_seed,
                group=sample_type,
                chromosome=sample.bundle.fasta_name,
                sample_id=sample_id,
            )
            sequence = dinucleotide_shuffle_sequence(sequence, shuffle_rng)
        changed_bases = sum(
            original_base != shuffled_base
            for original_base, shuffled_base in zip(original_sequence, sequence)
        )
        center_index = context.core_start_index + window_size // 2
        write_window_file(
            path=window_path,
            sequence=sequence,
            fe_signal=fe,
            pval_signal=pval,
            context_start=context.context_start,
            center_index=center_index,
            core_start_index=context.core_start_index,
            core_end_index=context.core_end_index,
            compress=compress,
        )
        metadata = sample_base_metadata(
            sample, sample_id, sample_type, window_size, center_index,
        )
        # Alle Kennzahlen und Pruefungen beziehen sich auf das innere Fenster;
        # das Umfeld wird nur diagnostisch beschrieben.
        core_original = original_sequence[context.core_start_index:context.core_end_index]
        core_sequence = sequence[context.core_start_index:context.core_end_index]
        core_fe = context.core_fe
        core_pval = context.core_pval
        canonical_bases = sum(core_original.count(base) for base in "ACGT")
        gc_bases = core_original.count("G") + core_original.count("C")
        gc_fraction = float(gc_bases / canonical_bases) if canonical_bases else math.nan
        flank_pval_max = context.flank_pval_max
        metadata.update(
            {
                "context_start_0based": context.context_start,
                "context_end_0based_exclusive": context.context_end,
                "context_length": len(sequence),
                "context_per_side": context_per_side,
                "context_padded_left": context.padded_left,
                "context_padded_right": context.padded_right,
                "core_window_index_start": context.core_start_index,
                "core_window_index_end_exclusive": context.core_end_index,
                "GC_bases": int(gc_bases),
                "canonical_bases": int(canonical_bases),
                "GC_fraction": gc_fraction,
                "FE_min": float(np.min(core_fe)),
                "FE_max": float(np.max(core_fe)),
                "FE_mean": float(np.mean(core_fe)),
                "pval_signal_min": float(np.min(core_pval)),
                "pval_signal_max": float(np.max(core_pval)),
                "pval_signal_mean": float(np.mean(core_pval)),
                "pval_negative_threshold": pval_threshold,
                "flank_pval_max": flank_pval_max,
                "flank_pval_exceeds_threshold": (
                    int(flank_pval_max > pval_threshold) if group == "negative" else None
                ),
                "flank_FE_max": context.flank_fe_max,
                "N_bases": int(core_sequence.count("N")),
                "N_fraction": float(core_sequence.count("N") / len(core_sequence)),
                "context_N_bases": int(sequence.count("N")),
                "sequence_orientation": (
                    "dinucleotide_shuffled_from_genomic_plus"
                    if dinucleotide_shuffle
                    else "genomic_plus"
                ),
                "sequence_shuffle_method": (
                    "exact_dinucleotide_eulerian"
                    if dinucleotide_shuffle
                    else "none"
                ),
                "sequence_shuffle_seed": (
                    dinucleotide_shuffle_seed if dinucleotide_shuffle else None
                ),
                "sequence_changed_bases": int(changed_bases),
                "sequence_changed_fraction": float(changed_bases / len(sequence)),
                "dinucleotide_counts_preserved": 1 if dinucleotide_shuffle else None,
                "sequence_matches_reference_coordinates": (
                    0 if dinucleotide_shuffle else 1
                ),
                "window_data_file": str(window_path.relative_to(output_dir)),
                "info_file": str(info_path.relative_to(output_dir)),
            }
        )
        if group == "negative" and float(np.max(core_pval)) > pval_threshold:
            raise AssertionError(
                f"Negatives Fenster {sample_id} verletzt nach erneutem Auslesen "
                f"den p-Wert-Filter: {float(np.max(core_pval))} > {pval_threshold}."
            )
        if isinstance(sample, NegativeSample):
            if not math.isfinite(gc_fraction):
                raise AssertionError(
                    f"GC-gematchtes Fenster {sample_id} enthaelt keine A/C/G/T-Basen."
                )
            difference = abs(gc_fraction - sample.gc_target_fraction)
            if (
                sample.gc_match_tolerance is not None
                and difference > sample.gc_match_tolerance + 1e-12
            ):
                raise AssertionError(
                    f"GC-gematchtes Fenster {sample_id} verletzt nach erneutem "
                    f"Auslesen die GC-Toleranz: {difference} > "
                    f"{sample.gc_match_tolerance}."
                )
            if not math.isclose(
                gc_fraction,
                sample.gc_fraction_at_selection,
                abs_tol=1e-12,
                rel_tol=0.0,
            ):
                raise AssertionError(
                    f"GC-Anteil von Fenster {sample_id} hat sich zwischen Auswahl "
                    "und Ausgabe geaendert."
                )
        write_info_file(info_path, metadata, class_definitions)
        metadata_rows.append(metadata)
        if index % 1000 == 0 or index == len(samples):
            LOGGER.info("%s: %d/%d Fenster geschrieben", sample_type, index, len(samples))
    return metadata_rows


def write_table(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields: list[str] = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
        fieldnames = fields
    with path.open("wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_scalar(row.get(key)) for key in fieldnames})


def write_igv_bed(path: Path, samples: Sequence[Any], sample_ids: Mapping[int, str], group: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    color = "0,160,0" if group == "positive" else "200,0,0"
    with path.open("wt", encoding="utf-8", newline="") as handle:
        handle.write(f'track name="{group}_windows" description="{group} windows" visibility=2 itemRgb="On"\n')
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for sample in samples:
            writer.writerow([sample.bundle.fasta_name, sample.window_start, sample.window_end,
                             sample_ids[id(sample)], 1000, ".", sample.center, sample.center + 1, color])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alle Summits und genomweite Negative mit gemeinsamem GC-Matching.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    for flag, description in (
        ("fe-bigwig", "MACS3-FE-BigWig"), ("pval-bigwig", "MACS3 -log10(p)-BigWig"),
        ("summit-bigwig", "1-bp-Peak-Summits mit Summit-Scores"),
        ("fasta", "Referenzgenom FASTA"), ("fai", "FASTA-Index (.fai)"),
        ("output-dir", "Neuer Datensatzordner"),
    ):
        parser.add_argument(f"--{flag}", type=Path, required=True, help=description)
    parser.add_argument("--chromosomes", help="Kommagetrennte Chromosomen; ohne Angabe alle aus der FASTA. Alle Tracks muessen diese enthalten.")
    parser.add_argument("--negative-location-preference", choices=("gene-or-tss", "gene", "tss", "none"),
                        default="gene-or-tss", help="Weiche Bevorzugung innerhalb der GC-Bereiche, kein Lagefilter.")
    parser.add_argument("--negative-preference-distance", type=int, default=1000,
                        help="Maximaler Abstand vom ORIGINALEN Kernfensterzentrum zu Gen/TSS; im Gen 0, Grenze inklusive.")
    parser.add_argument("--gene-annotation", type=Path, help="GFF3/GTF, auch gzip; nur fuer Gennaehe.")
    parser.add_argument("--gene-feature-types", default="gene,pseudogene,transposable_element_gene")
    parser.add_argument("--tss-plus", type=Path, help="BED6 mit Plus-TSS; nur fuer TSS-Naehe.")
    parser.add_argument("--tss-minus", type=Path, help="BED6 mit Minus-TSS; nur fuer TSS-Naehe.")
    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument("--context-per-side", type=int, default=196,
                        help="Kontextreserve je Seite; Filter und GC-Matching gelten nur fuer den Kern.")
    parser.add_argument("--pval-threshold", type=float, default=1.3,
                        help="Maximaler -log10(p)-Wert an jeder Base eines negativen Kernfensters.")
    parser.add_argument("--summit-min-value", type=float, default=0.0,
                        help="Summit-Scores muessen endlich und groesser als dieser Wert sein.")
    parser.add_argument("--wide-summit-policy", choices=("error", "midpoint"), default="error")
    parser.add_argument("--gc-bin-width", type=float, default=0.005)
    parser.add_argument("--gc-ks-tolerance", type=float, default=0.06,
                        help="Maximaler kumulativer GC-Abstand an den Bereichsgrenzen.")
    parser.add_argument("--gc-quantile-tolerance", type=float, default=0.015)
    parser.add_argument("--candidate-scan-chunk-size", type=int, default=1_000_000)
    parser.add_argument("--save-candidate-pools", action="store_true",
                        help="Exportiert auch den vollstaendigen hart gefilterten Pool als TSV (kann sehr gross sein).")
    parser.add_argument("--tmp-dir", type=Path, help="Arbeitsordner fuer SQLite; sonst TMPDIR/Systemstandard.")
    parser.add_argument("--uncompressed-window-files", action="store_true")
    parser.add_argument("--dinucleotide-shuffle", action="store_true")
    parser.add_argument("--dinucleotide-shuffle-seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    paths = [args.fe_bigwig, args.pval_bigwig, args.summit_bigwig, args.fasta, args.fai]
    if args.negative_location_preference in {"gene", "gene-or-tss"}:
        if args.gene_annotation is None:
            raise ValueError("--gene-annotation ist fuer die gewaehlte Lagepraeferenz erforderlich.")
        if not any(value.strip() for value in args.gene_feature_types.split(",")):
            raise ValueError("--gene-feature-types darf nicht leer sein.")
        paths.append(args.gene_annotation)
    if args.negative_location_preference in {"tss", "gene-or-tss"}:
        if args.tss_plus is None or args.tss_minus is None:
            raise ValueError("--tss-plus und --tss-minus sind fuer TSS-Bevorzugung erforderlich.")
        paths.extend([args.tss_plus, args.tss_minus])
    if args.negative_preference_distance < 0:
        raise ValueError("--negative-preference-distance darf nicht negativ sein.")
    missing = [str(path) for path in paths
               if not path.is_file()]
    if missing:
        raise FileNotFoundError("Eingabedateien fehlen: " + ", ".join(missing))
    if args.window_size <= 0:
        raise ValueError("--window-size muss positiv sein.")
    if not 0 <= args.context_per_side <= args.window_size // 2:
        raise ValueError("--context-per-side muss zwischen 0 und halber Fensterlaenge liegen.")
    if not math.isfinite(args.pval_threshold) or not math.isfinite(args.summit_min_value):
        raise ValueError("p-Wert- und Summit-Schwellen muessen endlich sein.")
    if not math.isfinite(args.gc_bin_width) or not 0 < args.gc_bin_width < 1:
        raise ValueError("--gc-bin-width muss zwischen 0 und 1 liegen.")
    for flag, value in (("gc-ks-tolerance", args.gc_ks_tolerance),
                        ("gc-quantile-tolerance", args.gc_quantile_tolerance)):
        if not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"--{flag} muss zwischen 0 und 1 liegen.")
    if args.candidate_scan_chunk_size < 1 or args.dinucleotide_shuffle_seed < 0:
        raise ValueError("Scan-Chunk muss positiv und Shuffle-Seed nichtnegativ sein.")
    if args.chromosomes is not None and not any(x.strip() for x in args.chromosomes.split(",")):
        raise ValueError("--chromosomes darf nicht leer sein.")
    if args.tmp_dir is not None:
        args.tmp_dir.mkdir(parents=True, exist_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    validate_args(args)
    pyBigWig, pysam = open_dependencies()
    class_definitions = {
        "positive": "Ein vollstaendiges, auf einem gueltigen Peak-Summit zentriertes Kernfenster; keine TSS-/Genpositionsregeln.",
        "negative": "Genomweit gewaehltes Kernfenster ohne positives Kernfenster-Overlap, mit p-Wert-Maximum unter der Schwelle und gemeinsamem GC-Matching.",
    }
    fe_bw = pval_bw = summit_bw = fasta = store = None
    try:
        fe_bw = open_bigwig(pyBigWig, args.fe_bigwig)
        pval_bw = open_bigwig(pyBigWig, args.pval_bigwig)
        summit_bw = open_bigwig(pyBigWig, args.summit_bigwig)
        fasta = pysam.FastaFile(str(args.fasta), filepath_index=str(args.fai))
        chrom_keys = [canonical_chrom(x.strip()) for x in args.chromosomes.split(",") if x.strip()] if args.chromosomes else [canonical_chrom(x) for x in fasta.references]
        bundles = build_chrom_bundles(chrom_keys, fasta, fe_bw, pval_bw, summit_bw)
        location_preference, preference_summary = load_location_preference(args, bundles)
        LOGGER.info("Negative Lageprioritaet: %s, Abstand <= %d bp vom Kernzentrum",
                    args.negative_location_preference, args.negative_preference_distance)
        summits, wide_count = load_summits(summit_bw, bundles, args.summit_min_value, args.wide_summit_policy)
        positives, positive_audit, positive_summary = select_positive_samples(summits, bundles, args.window_size)
        if not positives:
            raise RuntimeError("Keine gueltigen positiven Summit-Fenster vorhanden.")
        positive_gc = sample_gc_fractions(positives, fasta)
        ensure_output_dir(args.output_dir, args.overwrite)
        LOGGER.info("%s positive Summit-Fenster; gemeinsames GC-Matching", f"{len(positives):,}")
        with tempfile.TemporaryDirectory(prefix="tf_binding_pooled_gc_", dir=args.tmp_dir) as work:
            store = CandidateStore(Path(work) / "pooled.sqlite", "pooled", np.arange(args.gc_bin_width, 1.0, args.gc_bin_width))
            try:
                candidate_audit, negative_summary = build_dense_genome_pool(
                    positives, bundles, fasta, pval_bw, args.window_size, args.pval_threshold,
                    args.candidate_scan_chunk_size, store,
                    location_preference,
                )
                if store.count == 0:
                    raise RuntimeError("Keine negativen Kandidaten erfuellen p-Wert-, Overlap- und GC-Kriterien.")
                gc_capacity = write_gc_capacity_plot(args.output_dir, "pooled", positive_gc, store,
                                                      args.gc_bin_width, args.window_size, args.gc_ks_tolerance)
                LOGGER.info("Bevorzugte negative Kandidaten: %s von %s (%.2f%%)",
                            f"{negative_summary['preferred_candidate_windows']:,}", f"{store.count:,}",
                            100 * negative_summary["preferred_candidate_windows"] / store.count)
                matcher = GCMatcher(args.gc_bin_width, args.gc_ks_tolerance, args.window_size,
                                    args.gc_quantile_tolerance, prioritize_location=location_preference is not None)
                matches, match_summary, capacities = matcher.match(positive_gc, store, positives)
                negative_summary.update(match_summary)
                negatives, match_audit = materialize_matched_samples(matches, bundles, args.window_size, location_preference)
                preference_summary.update({
                    "preferred_candidate_windows": negative_summary["preferred_candidate_windows"],
                    "candidate_pool_size": store.count,
                    "preferred_candidate_fraction": negative_summary["preferred_candidate_windows"] / store.count,
                    "preferred_selected_windows": match_summary["location_preferred_selected"],
                    "selected_total": len(negatives),
                    "preferred_selected_fraction": match_summary["location_preferred_selected_fraction"],
                    "gc_priority_fallback_used": match_summary["location_preference_fallback_used"],
                })
                negative_summary["location_preference"] = preference_summary
                LOGGER.info("Ausgewaehlte bevorzugte Negative: %s von %s (%.2f%%)",
                            f"{match_summary['location_preferred_selected']:,}", f"{len(negatives):,}",
                            100 * match_summary["location_preferred_selected_fraction"])
                balance_rows, plot_paths = write_gc_coverage_plots(
                    args.output_dir, "pooled", positive_gc, matches, store, capacities,
                    args.gc_bin_width, match_summary["gc_binned_cdf_distance"], gc_capacity,
                )
                if args.save_candidate_pools:
                    directory = args.output_dir / "candidate_pools"
                    directory.mkdir(exist_ok=True)
                    write_candidate_store(directory / "negative_candidates.tsv", store, bundles, args.window_size)
                negative_summary["gc_comparison_outputs"] = plot_paths
                negative_summary["gc_capacity"] = {key: gc_capacity[key] for key in (
                    "distinct_starts", "nonoverlapping_total", "bound_strict", "bound_at_tolerance",
                    "bottleneck_strict", "bottleneck_at_tolerance",
                )}
            finally:
                store.close()
                store = None
        ids = unique_sample_ids([*positives, *negatives])
        metadata_by_class = {}
        for group, samples in (("positive", positives), ("negative", negatives)):
            rows = write_samples(args.output_dir, group, samples, ids, fasta, fe_bw, pval_bw,
                                 args.window_size, args.pval_threshold, not args.uncompressed_window_files,
                                 args.dinucleotide_shuffle, args.dinucleotide_shuffle_seed, class_definitions,
                                 args.context_per_side)
            metadata_by_class[group] = rows
            write_table(args.output_dir / f"metadata_{group}.tsv", rows)
            write_igv_bed(args.output_dir / "igv" / f"{group}_windows.bed", samples, ids, group)
        write_table(args.output_dir / "sample_classes.tsv",
                    [{"sample_type": name, "definition": definition} for name, definition in class_definitions.items()])
        write_table(args.output_dir / "audit" / "positive_summit_selection.tsv", positive_audit)
        write_table(args.output_dir / "audit" / "negative_candidates_by_chromosome.tsv", candidate_audit)
        write_table(args.output_dir / "audit" / "negative_gc_matching_balance.tsv", balance_rows)
        write_table(args.output_dir / "audit" / "negative_gc_selection.tsv", match_audit)
        summary = {
            "script_version": SCRIPT_VERSION, "gc_matcher_version": MATCHER_VERSION,
            "command": shlex.join([sys.executable, str(Path(__file__).resolve()), *(sys.argv[1:] if argv is None else argv)]),
            "arguments": vars(args), "sample_class_definitions": class_definitions,
            "selection": {"positive": "all_valid_summits", "negative": "all_genomic_starts_1bp",
                          "tss_gene_filters": False, "gc_matching_reference": "all_positive_core_windows",
                          "gc_matching_scope": "pooled", "selection_and_filters_span": "core_window_only",
                          "positive_center_policy": "summit", "candidate_grid_step": 1},
            "negative_location_preference": preference_summary,
            "counts": {"positive_total": len(positives), "negative_total": len(negatives),
                       "summits_considered": positive_summary["summits_considered"],
                       "summits_not_in_positive_output": positive_summary["excluded_window_outside_chromosome"],
                       "negative_candidate_pool": negative_summary["candidate_pool_size"],
                       "wide_summit_intervals": wide_count},
            "window_geometry": {"window_length": args.window_size, "context_per_side": args.context_per_side,
                                "context_length": args.window_size + 2 * args.context_per_side},
            "chromosomes": {key: {"fasta_name": bundle.fasta_name, "length": bundle.length} for key, bundle in bundles.items()},
            "positive_selection": positive_summary, "negative_selection": negative_summary,
            "context_flank_pval_exceedances": sum(int(row["flank_pval_exceeds_threshold"]) for row in metadata_by_class["negative"]),
        }
        # Metadaten und GC-Ergebnis muessen dieselben ORIGINALEN Kernfenster beschreiben.
        for group, expected in (("positive", positive_gc), ("negative", [sample.gc_fraction_at_selection for sample in negatives])):
            observed = [row["GC_fraction"] for row in metadata_by_class[group]]
            if not np.allclose(observed, expected, atol=1e-12, rtol=0):
                raise AssertionError(f"GC-Verteilung von {group} hat sich beim Schreiben geaendert.")
        with (args.output_dir / "summary.json").open("wt", encoding="utf-8") as handle:
            json.dump(summary, handle, cls=NumpyJSONEncoder, indent=2)
            handle.write("\n")
        (args.output_dir / ".complete").write_text(SCRIPT_VERSION + "\n", encoding="utf-8")
        LOGGER.info("Fertig: %s Positive, %s Negative -> %s", len(positives), len(negatives), args.output_dir)
        return 0
    finally:
        if store is not None:
            store.close()
        for handle in (fe_bw, pval_bw, summit_bw, fasta):
            if handle is not None:
                handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
