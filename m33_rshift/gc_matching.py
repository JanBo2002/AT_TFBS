#!/usr/bin/env python3
"""Kandidatenpool und GC-Matching fuer genomische Sequenzfenster."""

from __future__ import annotations

import math
import sqlite3
from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

MATCHER_VERSION = "1.4-cumulative-gc-capacity"


@dataclass(frozen=True)
class WindowCandidate:
    chrom_key: str
    window_start: int
    selection_pval_max: float
    gc_fraction: float
    gene: Any | None = None
    tss: Any | None = None
    origin_region: str | None = None
    offset_probability: float = 0.0


def gc_balance_metrics(
    reference: Sequence[float],
    comparison: Sequence[float],
) -> dict[str, float | None]:
    """Berechnet kontinuierliche, bin-unabhaengige GC-Balancekennzahlen."""
    ref = np.asarray(reference, dtype=np.float64)
    comp = np.asarray(comparison, dtype=np.float64)
    if ref.size == 0 or comp.size == 0:
        raise ValueError("GC-Balance kann nicht fuer leere Verteilungen berechnet werden.")
    if not np.all(np.isfinite(ref)) or not np.all(np.isfinite(comp)):
        raise ValueError("GC-Balance erfordert ausschliesslich endliche Werte.")

    pooled_variance = 0.5 * (
        float(np.var(ref, ddof=1)) if ref.size > 1 else 0.0
    ) + 0.5 * (
        float(np.var(comp, ddof=1)) if comp.size > 1 else 0.0
    )
    pooled_sd = math.sqrt(max(0.0, pooled_variance))
    mean_difference = float(np.mean(comp) - np.mean(ref))
    standardized_mean_difference = (
        mean_difference / pooled_sd
        if pooled_sd > 0.0
        else (0.0 if mean_difference == 0.0 else None)
    )
    support = np.unique(np.concatenate([ref, comp]))
    ref_ecdf = np.searchsorted(np.sort(ref), support, side="right") / ref.size
    comp_ecdf = np.searchsorted(np.sort(comp), support, side="right") / comp.size
    quantile_count = max(int(ref.size), int(comp.size), 2)
    probabilities = np.linspace(0.0, 1.0, quantile_count)
    quantile_difference = float(
        np.mean(
            np.abs(
                np.quantile(ref, probabilities)
                - np.quantile(comp, probabilities)
            )
        )
    )
    return {
        "reference_mean": float(np.mean(ref)),
        "comparison_mean": float(np.mean(comp)),
        "mean_difference_comparison_minus_reference": mean_difference,
        "standardized_mean_difference": (
            float(standardized_mean_difference)
            if standardized_mean_difference is not None
            else None
        ),
        "kolmogorov_smirnov_distance": float(
            np.max(np.abs(ref_ecdf - comp_ecdf))
        ),
        "mean_absolute_quantile_difference": quantile_difference,
    }


def greedy_nonoverlapping_count(sorted_starts: np.ndarray, window_size: int) -> int:
    """Maximale Zahl nicht ueberlappender Fenster gleicher Laenge.

    Fuer gleich lange Fenster ist die Greedy-Wahl nach Startposition (erstes
    Fenster nehmen, dann das erste ab Start + window_size) nachweislich die
    groesstmoegliche ueberlappungsfreie Auswahl; die Anzahl ist eindeutig.
    """
    count = 0
    index = 0
    size = int(sorted_starts.size)
    while index < size:
        count += 1
        index = int(np.searchsorted(
            sorted_starts, sorted_starts[index] + window_size, side="left"
        ))
    return count


def cumulative_capacity_bound(
    positive_bins: np.ndarray,
    n_bins: int,
    low: dict[int, int],
    high: dict[int, int],
    total: int,
    tolerance: float,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    """Obergrenze fuer die Zahl der Negativen bei gebinnter CDF-Toleranz.

    Jede gueltige Auswahl der Groesse N enthaelt mindestens N * (F_pos(t) - eps)
    Fenster mit GC <= t und hoechstens C_low(t) solche Fenster; entsprechend
    von oben. Das Minimum ueber alle Schwellen ist eine notwendige Grenze,
    die Schwelle mit dem Minimum der Engpass.
    """
    positive_counts = np.bincount(positive_bins, minlength=n_bins).astype(np.float64)
    cdf = np.cumsum(positive_counts) / positive_counts.sum()
    bound = float(total)
    bottleneck: dict[str, Any] = {"side": "total", "bin_index": None, "capacity": int(total)}
    rows: list[dict[str, Any]] = []
    for k in range(n_bins):
        fraction_low = float(cdf[k])                              # Anteil mit gc_bin <= k
        fraction_high = 1.0 - (float(cdf[k - 1]) if k else 0.0)  # Anteil mit gc_bin >= k
        row: dict[str, Any] = {
            "bin_index": k, "positive_cdf_at_upper_edge": fraction_low,
            "cumulative_capacity_gc_at_most_bin": None,
            "supported_negatives_from_low_tail": None,
            "cumulative_capacity_gc_at_least_bin": None,
            "supported_negatives_from_high_tail": None,
        }
        if k in low:
            need = fraction_low - tolerance
            supported = low[k] / need if need > 0 else math.inf
            row["cumulative_capacity_gc_at_most_bin"] = int(low[k])
            row["supported_negatives_from_low_tail"] = supported
            if supported < bound:
                bound = supported
                bottleneck = {"side": "low", "bin_index": k, "capacity": int(low[k])}
        if k in high:
            need = fraction_high - tolerance
            supported = high[k] / need if need > 0 else math.inf
            row["cumulative_capacity_gc_at_least_bin"] = int(high[k])
            row["supported_negatives_from_high_tail"] = supported
            if supported < bound:
                bound = supported
                bottleneck = {"side": "high", "bin_index": k, "capacity": int(high[k])}
        rows.append(row)
    return bound, bottleneck, rows


class CandidateStore:
    """Disk-backed candidate set; overlapping starts remain available for matching."""

    def __init__(self, path: Path, group: str, edges: np.ndarray):
        self.group = group
        self.edges = np.asarray(edges, dtype=np.float64)
        self.db = sqlite3.connect(str(path))
        self.db.execute("PRAGMA journal_mode=OFF")
        self.db.execute("PRAGMA synchronous=OFF")
        self.db.execute("PRAGMA temp_store=FILE")
        self.db.execute(
            "CREATE TABLE candidates (id INTEGER PRIMARY KEY, chrom TEXT NOT NULL, "
            "start INTEGER NOT NULL, pval REAL NOT NULL, gc REAL NOT NULL, "
            "gc_bin INTEGER NOT NULL, source_id INTEGER NOT NULL, "
            "origin_region TEXT, offset_probability REAL NOT NULL, "
            "match_key INTEGER NOT NULL, UNIQUE(chrom, start, match_key))"
        )
        self.db.execute(
            "CREATE TABLE origins (candidate_id INTEGER NOT NULL, source_id INTEGER NOT NULL, "
            "UNIQUE(candidate_id, source_id))"
        )
        self.sources: list[Any] = []
        self.source_indices: dict[int, int] = {}
        self.count = 0

    def _source_index(self, source: Any | None) -> int:
        if source is None:
            return -1
        key = id(source)
        if key not in self.source_indices:
            self.source_indices[key] = len(self.sources)
            self.sources.append(source)
        return self.source_indices[key]

    def add_many(self, candidates: Sequence[WindowCandidate]) -> int:
        inserted = 0
        cursor = self.db.cursor()
        for candidate in candidates:
            source = candidate.tss if self.group == "tss" else candidate.gene
            source_id = self._source_index(source)
            match_key = source_id + 1 if self.group == "tss" else 0
            gc_bin = int(np.searchsorted(self.edges, candidate.gc_fraction, side="right"))
            cursor.execute(
                "INSERT OR IGNORE INTO candidates "
                "(chrom,start,pval,gc,gc_bin,source_id,origin_region,offset_probability,match_key) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (candidate.chrom_key, candidate.window_start,
                 candidate.selection_pval_max, candidate.gc_fraction,
                 gc_bin, source_id, candidate.origin_region,
                 candidate.offset_probability, match_key),
            )
            if cursor.rowcount:
                candidate_id = int(cursor.lastrowid)
                inserted += 1
            else:
                candidate_id = int(cursor.execute(
                    "SELECT id FROM candidates WHERE chrom=? AND start=? AND match_key=?",
                    (candidate.chrom_key, candidate.window_start, match_key),
                ).fetchone()[0])
            if source_id >= 0:
                cursor.execute(
                    "INSERT OR IGNORE INTO origins(candidate_id,source_id) VALUES (?,?)",
                    (candidate_id, source_id),
                )
        self.count += inserted
        return inserted

    def finish(self) -> None:
        self.db.commit()
        self.db.execute(
            "CREATE INDEX candidates_by_gc ON candidates(gc_bin,chrom,start)"
        )
        self.db.commit()

    def bin_counts(self) -> np.ndarray:
        """Rohzahl der Kandidatenzeilen je GC-Bereich (alle Zeilen, auch je TSS)."""
        counts = np.zeros(len(self.edges) + 1, dtype=np.int64)
        for gc_bin, count in self.db.execute(
            "SELECT gc_bin, COUNT(*) FROM candidates GROUP BY gc_bin"
        ):
            counts[int(gc_bin)] = int(count)
        if int(counts.sum()) != self.count:
            raise AssertionError("Kandidatengroesse und GC-Zaehlung weichen ab.")
        return counts

    def start_arrays(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Je Chromosom sortierte Startpositionen und GC-Bins, jede Position einmal."""
        chroms = [str(row[0]) for row in self.db.execute(
            "SELECT DISTINCT chrom FROM candidates ORDER BY chrom"
        )]
        arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for chrom in chroms:
            # UNIQUE(chrom,start,match_key) liefert die Starts sortiert; identische
            # Koordinaten mit mehreren TSS werden nur einmal gewertet.
            cursor = self.db.execute(
                "SELECT start, MIN(gc_bin), MAX(gc_bin) FROM candidates WHERE chrom=? "
                "GROUP BY start ORDER BY start", (chrom,)
            )
            start_chunks: list[np.ndarray] = []
            bin_chunks: list[np.ndarray] = []
            while True:
                rows = cursor.fetchmany(1_000_000)
                if not rows:
                    break
                block = np.asarray(rows, dtype=np.int64)
                if np.any(block[:, 1] != block[:, 2]):
                    raise AssertionError("Identische Fenster haben verschiedene GC-Bereiche.")
                start_chunks.append(block[:, 0])
                bin_chunks.append(block[:, 1].astype(np.int16))
            if start_chunks:
                arrays[chrom] = (np.concatenate(start_chunks), np.concatenate(bin_chunks))
        return arrays

    def gc_capacity(
        self, window_size: int, positive_gc: Sequence[float], tolerance: float,
        strict_tolerance: float = 0.0,
    ) -> dict[str, Any]:
        """Kumulative GC-Kapazitaet des Pools gegen die positive GC-Verteilung.

        Fuer jede GC-Schwelle t wird die maximale Zahl ueberlappungsfreier
        Kandidaten mit GC <= t (C_low) bzw. GC >= t (C_high) exakt bestimmt.
        Daraus folgen Obergrenzen fuer die Zahl der Negativen, streng
        (strict_tolerance) und mit der CDF-Toleranz des Matchers (tolerance),
        jeweils mit der Schwelle, an der die Grenze bindet. Die Grenzen sind
        notwendige Bedingungen fuer jede Auswahl, keine Zusage der Heuristik.
        """
        if window_size <= 0:
            raise ValueError("Fensterlaenge muss positiv sein.")
        positive = np.asarray(positive_gc, dtype=np.float64)
        if positive.size == 0:
            raise ValueError("GC-Kapazitaet benoetigt positive Referenzwerte.")
        n_bins = len(self.edges) + 1
        positive_bins = np.searchsorted(self.edges, positive, side="right")
        arrays = self.start_arrays()
        distinct_starts = int(sum(starts.size for starts, _ in arrays.values()))
        if distinct_starts == 0:
            raise RuntimeError(f"Keine negativen Kandidaten fuer GC-Kapazitaet von {self.group}.")
        # Kapazitaet je Bereich fuer sich (optimal gepackt, ohne Konkurrenz der
        # Nachbarbereiche) - dieselbe Groesse wie im Matcher.
        standalone = np.zeros(n_bins, dtype=np.int64)
        for starts, bins in arrays.values():
            for b in np.unique(bins):
                standalone[int(b)] += greedy_nonoverlapping_count(starts[bins == b], window_size)
        available = np.flatnonzero(standalone > 0)
        # Wie im Matcher: Positive in Bereichen ohne jeden Kandidaten werden dem
        # naechsten Bereich mit Kandidaten zugerechnet.
        reassigned = 0
        present = np.bincount(positive_bins, minlength=n_bins) > 0
        for b in np.flatnonzero(present & (standalone == 0)):
            nearest = int(min(available, key=lambda x: (abs(int(x) - int(b)), int(x))))
            reassigned += int(np.count_nonzero(positive_bins == b))
            positive_bins[positive_bins == b] = nearest
        occupied = np.flatnonzero(np.bincount(positive_bins, minlength=n_bins) > 0)
        # Schwellen nur dort, wo sich F_pos aendert.
        thresholds_low = [int(k) for k in occupied if k < occupied[-1]]
        thresholds_high = [int(k) for k in occupied if k > occupied[0]]
        total = sum(greedy_nonoverlapping_count(starts, window_size) for starts, _ in arrays.values())
        low: dict[int, int] = {}
        high: dict[int, int] = {}
        for k in thresholds_low:
            low[k] = sum(greedy_nonoverlapping_count(starts[bins <= k], window_size)
                         for starts, bins in arrays.values())
        for k in thresholds_high:
            high[k] = sum(greedy_nonoverlapping_count(starts[bins >= k], window_size)
                          for starts, bins in arrays.values())
        strict_bound, strict_bottleneck, strict_rows = cumulative_capacity_bound(
            positive_bins, n_bins, low, high, total, strict_tolerance
        )
        tolerant_bound, tolerant_bottleneck, rows = cumulative_capacity_bound(
            positive_bins, n_bins, low, high, total, tolerance
        )
        for row, strict_row in zip(rows, strict_rows):
            row["supported_negatives_from_low_tail_strict"] = (
                strict_row["supported_negatives_from_low_tail"]
            )
            row["supported_negatives_from_high_tail_strict"] = (
                strict_row["supported_negatives_from_high_tail"]
            )
        return {
            "group": self.group,
            "window_size": int(window_size),
            "positive_count": int(positive.size),
            "positive_bins": positive_bins,
            "positives_reassigned_to_nearest_bin": reassigned,
            "raw_bin_counts": self.bin_counts(),
            "distinct_starts": distinct_starts,
            "standalone_bin_capacity": standalone,
            "nonoverlapping_total": int(total),
            "capacity_gc_at_most": low,
            "capacity_gc_at_least": high,
            "strict_tolerance": float(strict_tolerance),
            "bound_strict": strict_bound,
            "bottleneck_strict": strict_bottleneck,
            "tolerance": float(tolerance),
            "bound_at_tolerance": tolerant_bound,
            "bottleneck_at_tolerance": tolerant_bottleneck,
            "rows": rows,
        }

    def iter_bin(self, gc_bin: int, prioritize_offset: bool = False) -> Iterable[tuple[Any, ...]]:
        ordering = "offset_probability DESC,chrom,start,id" if prioritize_offset else "chrom,start,id"
        return self.db.execute(
            "SELECT id,chrom,start,pval,gc,source_id FROM candidates "
            f"WHERE gc_bin=? ORDER BY {ordering}", (gc_bin,)
        )

    def selected_candidate(self, row: tuple[Any, ...]) -> WindowCandidate:
        _id, chrom, start, pval, gc, source_id = row
        source = self.sources[source_id] if source_id >= 0 else None
        return WindowCandidate(
            chrom_key=str(chrom), window_start=int(start),
            selection_pval_max=float(pval), gc_fraction=float(gc),
            gene=source if self.group == "genic" else None,
            tss=source if self.group == "tss" else None,
        )

    def source_names(self, candidate_id: int) -> str:
        if self.group == "noncoding":
            row = self.db.execute(
                "SELECT origin_region FROM candidates WHERE id=?", (candidate_id,)
            ).fetchone()
            return str(row[0]) if row and row[0] else "intergenic_region"
        source_ids = [row[0] for row in self.db.execute(
            "SELECT source_id FROM origins WHERE candidate_id=? ORDER BY source_id",
            (candidate_id,),
        )]
        return ",".join(
            self.sources[index].name if self.group == "tss"
            else self.sources[index].gene_id for index in source_ids
        )

    def close(self) -> None:
        self.db.close()


class GCMatcher:
    """GC-Matching fuer jeweils eine positive und negative Beispielgruppe.

    Aus allen gueltigen Kandidaten werden moeglichst viele Negative gewaehlt,
    ohne ueberlappende Fenster oder doppelte TSS. Die GC-Verteilung der
    ausgewaehlten Fenster muss die positive Verteilung innerhalb der
    eingestellten CDF- und Quantilgrenzen treffen. Es gibt keine feste
    Zielzahl; die deterministische Auswahl garantiert kein globales Maximum.
    """

    # Knappe GC-Bereiche werden zuerst belegt; bessere ueberlappende Fenster
    # koennen bereits ausgewaehlte Kandidaten desselben Bereichs ersetzen.

    def __init__(self, gc_bin_width: float, ks_tolerance: float, window_size: int,
                 quantile_tolerance: float = 0.015):
        self.bin_width = gc_bin_width
        self.ks_tolerance = ks_tolerance
        self.window_size = window_size
        self.quantile_tolerance = quantile_tolerance

    @staticmethod
    def ks_distance(reference: np.ndarray, selected: np.ndarray) -> float:
        if selected.size == 0:
            return math.inf
        support = np.unique(np.concatenate((reference, selected)))
        return float(np.max(np.abs(
            np.searchsorted(np.sort(reference), support, side="right") / reference.size
            - np.searchsorted(np.sort(selected), support, side="right") / selected.size
        )))

    @staticmethod
    def binned_cdf_distance(reference: np.ndarray, selected: np.ndarray,
                            edges: np.ndarray) -> float:
        if selected.size == 0:
            return math.inf
        p = np.bincount(np.searchsorted(edges, reference, side="right"),
                        minlength=len(edges) + 1) / len(reference)
        n = np.bincount(np.searchsorted(edges, selected, side="right"),
                        minlength=len(edges) + 1) / len(selected)
        return float(np.max(np.abs(np.cumsum(p) - np.cumsum(n))))

    @staticmethod
    def quantile_distance(reference: np.ndarray, selected: np.ndarray) -> float:
        if selected.size == 0:
            return math.inf
        q = np.linspace(0, 1, max(len(reference), len(selected), 2))
        return float(np.mean(np.abs(np.quantile(reference, q)
                                    - np.quantile(selected, q))))

    def _repair_balance(
        self,
        reference: np.ndarray,
        selected_rows: list[tuple[int, WindowCandidate, float]],
        edges: np.ndarray,
    ) -> list[tuple[int, WindowCandidate, float]] | None:
        """Behält möglichst viele der bereits konfliktfreien Fenster mit passender GC-CDF."""
        if not selected_rows:
            return None
        n_bins = len(edges) + 1
        groups: list[list[tuple[int, WindowCandidate, float]]] = [
            [] for _ in range(n_bins)
        ]
        for row in selected_rows:
            groups[int(np.searchsorted(edges, row[1].gc_fraction, side="right"))].append(row)
        capacities = [len(group) for group in groups]
        reference_counts = np.bincount(
            np.searchsorted(edges, reference, side="right"), minlength=n_bins
        )
        reference_cdf = np.cumsum(reference_counts) / len(reference)

        # Fuer jede moegliche Groesse pruefen, ob sich die vorhandenen Fenster
        # auf die GC-Bereiche verteilen lassen, ohne die CDF-Grenze zu verletzen.
        for size in range(len(selected_rows), 0, -1):
            lower: list[int] = []
            upper: list[int] = []
            lo = hi = 0
            for b, capacity in enumerate(capacities):
                lo = max(lo, math.ceil(size * (reference_cdf[b] - self.ks_tolerance) - 1e-10))
                hi = min(hi + capacity, math.floor(size * (reference_cdf[b] + self.ks_tolerance) + 1e-10))
                if lo > hi:
                    break
                lower.append(lo)
                upper.append(hi)
            if len(lower) != n_bins or not lower[-1] <= size <= upper[-1]:
                continue

            # Rueckwaerts zu einer moeglichst referenznahen Verteilung.
            kept_per_bin = [0] * n_bins
            prefix = size
            for b in range(n_bins - 1, -1, -1):
                previous_lo = lower[b - 1] if b else 0
                previous_hi = upper[b - 1] if b else 0
                minimum = max(previous_lo, prefix - capacities[b])
                maximum = min(previous_hi, prefix)
                desired = round(size * reference_cdf[b - 1]) if b else 0
                previous_prefix = min(max(desired, minimum), maximum)
                kept_per_bin[b] = prefix - previous_prefix
                prefix = previous_prefix

            keep_ids: set[int] = set()
            for group, count in zip(groups, kept_per_bin):
                if count == len(group):
                    keep_ids.update(row[0] for row in group)
                elif count:
                    keep_ids.update(
                        row[0] for row in sorted(
                            group,
                            key=lambda row: (abs(row[1].gc_fraction - row[2]), row[0]),
                        )[:count]
                    )
            retained = [row for row in selected_rows if row[0] in keep_ids]
            values = np.asarray([row[1].gc_fraction for row in retained])
            if (self.binned_cdf_distance(reference, values, edges) <= self.ks_tolerance
                    and self.quantile_distance(reference, values) <= self.quantile_tolerance):
                return retained
        return None

    def match(
        self,
        positive_gc: Sequence[float],
        store: CandidateStore,
        occupied: Sequence[Any],
    ) -> tuple[list[tuple[int, WindowCandidate, float]], dict[str, Any], list[int]]:
        reference = np.asarray(positive_gc, dtype=np.float64)
        if reference.size == 0 or not np.all(np.isfinite(reference)):
            raise ValueError("Positive GC-Referenz ist leer oder ungueltig.")
        if store.count == 0:
            raise RuntimeError(f"Kein gueltiger Negativkandidat fuer {store.group}.")
        n_bins = len(store.edges) + 1
        positive_bins = np.searchsorted(store.edges, reference, side="right")
        positive_counts = np.bincount(positive_bins, minlength=n_bins)
        occupied_intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for sample in occupied:
            occupied_intervals[sample.bundle.chrom_key].append(
                (int(sample.window_start), int(sample.window_end))
            )
        occupied_by_chrom: dict[str, tuple[list[int], list[int]]] = {}
        for chrom, ranges in occupied_intervals.items():
            merged: list[list[int]] = []
            for start, end in sorted(ranges):
                if not merged or start > merged[-1][1]:
                    merged.append([start, end])
                else:
                    merged[-1][1] = max(merged[-1][1], end)
            occupied_by_chrom[chrom] = (
                [item[0] for item in merged],
                [item[1] for item in merged],
            )

        def overlaps_occupied(chrom: str, start: int) -> bool:
            blocked = occupied_by_chrom.get(chrom)
            if not blocked:
                return False
            starts, ends = blocked
            index = bisect_left(starts, start + self.window_size) - 1
            return index >= 0 and ends[index] > start
        # Pro GC-Bereich ist die Greedy-Wahl nach rechtem Fensterrand bei
        # gleich langen Fenstern eine unabhaengige obere Kapazitaetsschranke.
        capacities = np.zeros(n_bins, dtype=np.int64)
        for gc_bin in range(n_bins):
            last_chrom = None
            last_end = -1
            for _id, chrom, start, _pval, _gc, _source in store.iter_bin(gc_bin):
                if chrom != last_chrom:
                    last_chrom, last_end = chrom, -1
                if start < last_end:
                    continue
                if overlaps_occupied(chrom, start):
                    continue
                capacities[gc_bin] += 1
                last_end = start + self.window_size
        available = np.flatnonzero(capacities > 0)
        if available.size == 0:
            raise RuntimeError(f"Keine nicht ueberlappenden Negativkandidaten fuer {store.group}.")
        target_bins = positive_bins.copy()
        missing_bins = np.flatnonzero((positive_counts > 0) & (capacities == 0))
        for b in missing_bins:
            nearest = min(available, key=lambda x: (abs(int(x) - int(b)), int(x)))
            target_bins[positive_bins == b] = nearest
        target_counts = np.bincount(target_bins, minlength=n_bins)
        fractions = target_counts / reference.size
        active = np.flatnonzero(target_counts > 0)
        if not active.size:
            raise AssertionError("Keine besetzten positiven GC-Bereiche.")
        # Echte raeumliche Obergrenze ueber ALLE GC-Bereiche. Einzelne GC-
        # Bereiche liefern zusaetzliche Schranken, aber innerhalb der
        # erlaubten CDF-Abweichung muss ein seltener Bereich nicht exakt
        # entsprechend seiner positiven Haeufigkeit besetzt werden.
        global_capacity = 0
        last_chrom, last_end = None, -1
        for chrom, start in store.db.execute(
            "SELECT chrom,start FROM candidates ORDER BY chrom,start"
        ):
            if chrom != last_chrom:
                last_chrom, last_end = chrom, -1
            if start >= last_end and not overlaps_occupied(chrom, start):
                global_capacity += 1
                last_end = start + self.window_size
        upper = global_capacity
        for b in active:
            minimum_fraction = fractions[b] - 2 * self.ks_tolerance
            if minimum_fraction > 0:
                upper = min(upper, int(capacities[b] / minimum_fraction))
        if upper <= 0:
            raise RuntimeError(
                f"GC-Matching fuer {store.group}: Mindestens ein positiver "
                "GC-Bereich besitzt kein zulaessiges negatives Fenster. "
                "Mit dieser GC-Aufloesung ist keine ausgewogene Auswahl moeglich."
            )
        attempts: list[dict[str, Any]] = []
        target_size = upper
        best: list[tuple[int, WindowCandidate, float]] = []
        best_ks = math.inf
        while target_size >= 1:
            exact_quotas = target_size * fractions
            quotas = np.floor(exact_quotas).astype(np.int64)
            remaining = target_size - int(quotas.sum())
            priority = sorted(active, key=lambda b: (
                -(exact_quotas[b] - quotas[b]), b
            ))
            for b in priority[:remaining]:
                quotas[b] += 1
            # Enge GC-Bereiche zuerst reservieren, damit benachbarte
            # ueberlappende Fenster sie nicht verdraengen.
            bin_order = sorted(active, key=lambda b: (
                capacities[b] / max(1, quotas[b]), capacities[b], b
            ))
            selected_rows: list[tuple[int, WindowCandidate, float]] = []
            selected_by_chrom: dict[str, dict[int, tuple[int, int]]] = defaultdict(dict)
            selected_targets: list[float] = []
            selected_source_ids: list[int] = []
            used_tss: set[int] = set()
            obtained = np.zeros(n_bins, dtype=np.int64)
            for b in bin_order:
                if quotas[b] == 0:
                    continue
                bin_reference = np.sort(reference[target_bins == b])
                bin_targets = np.quantile(
                    bin_reference,
                    (np.arange(quotas[b]) + 0.5) / max(1, quotas[b]),
                ) if quotas[b] else np.empty(0)
                for row in store.iter_bin(int(b), prioritize_offset=store.group == "tss"):
                    candidate_id, chrom, start, _pval, gc, source_id = row
                    if overlaps_occupied(chrom, start):
                        continue
                    buckets = selected_by_chrom[chrom]
                    block = start // self.window_size
                    conflicts = {
                        buckets[adjacent][1]
                        for adjacent in (block - 1, block, block + 1)
                        if adjacent in buckets
                        and abs(start - buckets[adjacent][0]) < self.window_size
                    }
                    if len(conflicts) == 1:
                        old_index = next(iter(conflicts))
                        old_id, old_candidate, _ = selected_rows[old_index]
                        old_source = selected_source_ids[old_index]
                        if (np.searchsorted(store.edges, old_candidate.gc_fraction,
                                            side="right") == b
                                and candidate_id != old_id
                                and (store.group != "tss" or source_id == old_source
                                     or source_id not in used_tss)
                                and abs(gc - selected_targets[old_index]) + 1e-12
                                < abs(old_candidate.gc_fraction - selected_targets[old_index])):
                            old_block = old_candidate.window_start // self.window_size
                            del selected_by_chrom[old_candidate.chrom_key][old_block]
                            buckets[block] = (start, old_index)
                            if store.group == "tss":
                                used_tss.discard(old_source)
                                used_tss.add(source_id)
                            selected_rows[old_index] = (
                                int(candidate_id), store.selected_candidate(row), float(gc)
                            )
                            selected_source_ids[old_index] = source_id
                        continue
                    if conflicts or obtained[b] >= quotas[b]:
                        continue
                    if store.group == "tss" and source_id in used_tss:
                        continue
                    buckets[block] = (start, len(selected_rows))
                    if store.group == "tss":
                        used_tss.add(source_id)
                    selected_rows.append((int(candidate_id), store.selected_candidate(row), float(gc)))
                    selected_targets.append(float(bin_targets[obtained[b]]))
                    selected_source_ids.append(source_id)
                    obtained[b] += 1
            values = np.asarray([item[1].gc_fraction for item in selected_rows])
            ks = self.binned_cdf_distance(reference, values, store.edges)
            quantile_error = self.quantile_distance(reference, values)
            attempts.append({"requested": target_size, "obtained": len(selected_rows),
                             "binned_cdf_distance": ks,
                             "mean_absolute_quantile_difference": quantile_error})
            acceptable = (ks <= self.ks_tolerance
                          and quantile_error <= self.quantile_tolerance)
            if not acceptable:
                repaired = self._repair_balance(reference, selected_rows, store.edges)
                if repaired is not None:
                    attempts[-1]["gc_repair_removed"] = len(selected_rows) - len(repaired)
                    attempts[-1]["gc_repair_retained"] = len(repaired)
                    selected_rows = repaired
                    values = np.asarray([item[1].gc_fraction for item in selected_rows])
                    ks = self.binned_cdf_distance(reference, values, store.edges)
                    quantile_error = self.quantile_distance(reference, values)
                    acceptable = True
            if acceptable and len(selected_rows) > len(best):
                best, best_ks = selected_rows, ks
            if len(selected_rows) == target_size and acceptable:
                break
            # Auch ein nicht besetzter GC-Bereich darf den naechsten Versuch
            # nicht auf null setzen: andere Bereiche koennen die CDF-Grenze
            # weiterhin einhalten.
            next_size = min(target_size - 1, int(target_size * 0.9))
            if best and next_size <= len(best):
                break
            target_size = next_size
        if not best:
            raise RuntimeError(
                f"Fuer {store.group} wurde keine ueberlappungsfreie Auswahl "
                f"mit GC-CDF-Abstand <= {self.ks_tolerance} und mittlerem "
                f"Quantilabstand <= {self.quantile_tolerance} gefunden. "
                f"Versuche: {attempts[-3:]}"
            )
        chosen = np.asarray([item[1].gc_fraction for item in best])
        starts_by_chrom: dict[str, list[int]] = defaultdict(list)
        selected_tss: set[int] = set()
        for _candidate_id, candidate, _target in best:
            if overlaps_occupied(candidate.chrom_key, candidate.window_start):
                raise AssertionError("GC-Auswahl ueberlappt ein gesperrtes Fenster.")
            starts_by_chrom[candidate.chrom_key].append(candidate.window_start)
            if store.group == "tss":
                if candidate.tss is None or id(candidate.tss) in selected_tss:
                    raise AssertionError("TSS-Herkunft in GC-Auswahl doppelt oder fehlend.")
                selected_tss.add(id(candidate.tss))
        for chrom, starts in starts_by_chrom.items():
            ordered_starts = sorted(starts)
            if any(b - a < self.window_size for a, b in zip(ordered_starts, ordered_starts[1:])):
                raise AssertionError(f"GC-Auswahl enthaelt ueberlappende Fenster auf {chrom}.")
        if (self.binned_cdf_distance(reference, chosen, store.edges) > self.ks_tolerance
                or self.quantile_distance(reference, chosen) > self.quantile_tolerance):
            raise AssertionError("GC-Auswahl verletzt die geprueften Verteilungsgrenzen.")
        summary = {
            "selection_mode": "maximum_cardinality_gc_constrained_heuristic",
            "global_optimum_guaranteed": False,
            "positive_samples_unchanged": True,
            "candidate_pool_size": store.count,
            "created_total": len(best),
            "selected_without_overlap": True,
            "positive_gc_count": len(reference),
            "gc_binned_cdf_tolerance": self.ks_tolerance,
            "gc_binned_cdf_distance": best_ks,
            "gc_mean_quantile_tolerance": self.quantile_tolerance,
            "gc_mean_quantile_distance": self.quantile_distance(reference, chosen),
            "gc_raw_ks_distance_diagnostic_only": self.ks_distance(reference, chosen),
            "per_window_gc_targets": "post_selection_quantile_audit_only",
            "gc_search_size_upper_bound": upper,
            "candidate_global_nonoverlap_upper_bound": global_capacity,
            "positive_gc_bins_without_negative_capacity": missing_bins.tolist(),
            "gc_bins": [
                {"bin_index": int(b), "positive_count": int(positive_counts[b]),
                 "nonoverlapping_capacity_upper_bound": int(capacities[b]),
                 "selected_count": int(np.count_nonzero(
                     np.searchsorted(store.edges, chosen, side="right") == b
                 ))}
                for b in range(n_bins)
            ],
            "search_attempts": attempts,
            "continuous_balance_after_matching": gc_balance_metrics(reference, chosen),
        }
        ordered = sorted(best, key=lambda item: (item[1].gc_fraction, item[0]))
        target_quantiles = np.quantile(
            reference, (np.arange(len(ordered)) + 0.5) / len(ordered)
        )
        targets = {item[0]: float(target) for item, target in zip(ordered, target_quantiles)}
        best = [(item[0], item[1], targets[item[0]]) for item in best]
        return best, summary, capacities.tolist()
