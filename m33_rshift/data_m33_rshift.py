#!/usr/bin/env python3
"""
Erzeugt einen ChIP-seq-Datensatz fuer TF-Bindestellenvorhersage.

Die positiven TSS-, genischen und nichtgenischen Fenster entsprechen der
Version 6.1: TSS-nahe Summits werden TSS zugeordnet, andere Summits im Abstand
von mindestens --min-nontss-summit-distance bp werden anhand ihrer Summit-Base
als genisch/nichtgenisch klassifiziert. Alle positiven Fenster sind auf den
Summit zentriert. BED/BigWig-Koordinaten sind 0-basiert und halb-offen.

Version 7.1: Jede Fensterdatei traegt links und rechts --context-per-side
zusaetzliche Basen (Standard 196, also 1416 Zeilen). Auswahl, GC-Matching,
p-Wert-Filter und Ueberlappungspruefung gelten weiterhin nur fuer das innere
1024-bp-Fenster; das Umfeld ist Reserve fuer den zufaelligen Zuschnitt im
Training (Shift je Epoche). Positive Fenster liegen exakt auf dem Summit; die
fruehere Gleichverteilung des Zentrums (--positive-shift-max) entfaellt.
Ragt das Umfeld ueber das Chromosom hinaus, wird es mit N und Signal 0
aufgefuellt. Fuer Negative wird nur diagnostisch gezaehlt, wie oft im Umfeld
ein p-Wert ueber der Schwelle liegt.

Version 7.0: Negative Kandidaten werden fuer alle gueltigen Fensterstarts
(1-bp-Schritt) aufgebaut, auch wenn sie sich untereinander ueberlappen.
Fuer TSS werden alle gueltigen Offsets mit positiver Offsetwahrscheinlichkeit
betrachtet. Ein eigenstaendiger GCMatcher waehlt getrennt fuer jede der drei
Gruppen moeglichst viele, sich nicht ueberlappende negative Fenster aus, deren
kumulative GC-Verteilung an den GC-Bereichsgrenzen hoechstens
--gc-ks-tolerance und deren mittlerer Quantilabstand hoechstens
--gc-quantile-tolerance von der positiven Referenz abweicht.
Jeder ausgewaehlte negative TSS hat eine eigene TSS.
Es gibt keine vorgegebene Negativzahl; die Heuristik garantiert kein globales
Maximum.

Die GC-Kapazitaetsgrafik bestimmt fuer jede GC-Schwelle t die maximale Zahl
ueberlappungsfreier Kandidaten mit GC <= t bzw. GC >= t (Greedy nach Start,
fuer gleich lange Fenster exakt) und teilt sie durch den Anteil der Positiven
dort. Das Minimum ueber alle Schwellen ist eine Obergrenze fuer die Zahl der
Negativen, die Schwelle mit dem Minimum der Engpass; beides wird streng und mit
der CDF-Toleranz des Matchers ausgewiesen. Das Ergebnis zeigt danach
Histogramm, ECDF und kumulative Abweichung fuer Positive und Negative.
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
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from urllib.parse import unquote

import numpy as np

from gc_matching import MATCHER_VERSION, CandidateStore, GCMatcher, WindowCandidate


LOGGER = logging.getLogger("tf_binding_dataset")
SCRIPT_VERSION = "7.1.0-context-flanks-summit-centered"


@dataclass(frozen=True)
class TSS:
    chrom_input: str
    chrom_key: str
    start: int
    end: int
    name: str
    bed_score: str
    strand: str
    cage_score: float | None
    position: int
    source_file: str
    line_number: int


@dataclass(frozen=True)
class Gene:
    chrom_input: str
    chrom_key: str
    start: int
    end: int
    gene_id: str
    name: str
    strand: str
    biotype: str | None
    feature_type: str
    source_file: str
    line_number: int


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
    tss: TSS
    summit: Summit
    candidate_count: int
    bundle: ChromBundle
    window_start: int
    window_end: int
    window_shift: int
    genomic_offset: int
    oriented_offset: int
    summit_genomic_offset: int
    summit_oriented_offset: int

    @property
    def center(self) -> int:
        return self.summit.position + self.window_shift

    @property
    def summit_index(self) -> int:
        return self.summit.position - self.window_start


@dataclass(frozen=True)
class AnnotationPositiveSample:
    positive_type: str
    summit: Summit
    bundle: ChromBundle
    window_start: int
    window_end: int
    window_shift: int
    nearest_tss: TSS
    nearest_tss_distance: int
    gene: Gene | None = None
    overlapping_gene_count: int = 0

    @property
    def center(self) -> int:
        return self.summit.position + self.window_shift

    @property
    def summit_index(self) -> int:
        return self.summit.position - self.window_start


@dataclass(frozen=True)
class NegativeSample:
    tss: TSS
    bundle: ChromBundle
    center: int
    window_start: int
    window_end: int
    genomic_offset: int
    oriented_offset: int
    selection_pval_max: float
    offset_observed_count: int
    offset_estimated_probability: float
    offset_support_mode: str
    origin_ids: str | None = None


@dataclass(frozen=True)
class AnnotationNegativeSample:
    negative_type: str
    bundle: ChromBundle
    center: int
    window_start: int
    window_end: int
    selection_pval_max: float
    gc_target_fraction: float
    gc_fraction_at_selection: float
    gc_absolute_difference: float
    gc_match_tolerance: float | None
    gene: Gene | None = None
    gc_bin_index: int | None = None
    gc_bin_start: float | None = None
    gc_bin_end: float | None = None
    gc_bin_width: float | None = None
    gc_assignment_mode: str | None = None
    gc_target_bin_quota: int | None = None
    origin_ids: str | None = None


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


def parse_optional_float(value: str) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_tss_bed(path: Path, expected_strand: str) -> list[TSS]:
    records: list[TSS] = []
    seen_ids: Counter[str] = Counter()
    with path.open("rt", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("track") or line.startswith("browser"):
                continue
            fields = raw_line.rstrip("\n\r").split("\t")
            if len(fields) < 6:
                fields = line.split()
            if len(fields) < 6:
                raise ValueError(
                    f"{path}:{line_number}: Mindestens BED6 erwartet, "
                    f"gefunden wurden {len(fields)} Felder."
                )
            chrom, start_text, end_text, name, bed_score, strand = fields[:6]
            try:
                start = int(start_text)
                end = int(end_text)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: start/end sind keine ganzen Zahlen."
                ) from exc
            if start < 0 or end <= start:
                raise ValueError(
                    f"{path}:{line_number}: Ungueltiges BED-Intervall "
                    f"[{start}, {end})."
                )
            if strand != expected_strand:
                raise ValueError(
                    f"{path}:{line_number}: Strang {strand!r}, erwartet "
                    f"wurde {expected_strand!r}."
                )
            if not name or name == ".":
                name = f"TSS_{canonical_chrom(chrom)}_{start}_{strand}"
            cage_score = parse_optional_float(fields[6]) if len(fields) >= 7 else None
            # Fuer mehrbasige BED-Intervalle wird die 5'-Position verwendet.
            position = start if strand == "+" else end - 1
            record = TSS(
                chrom_input=chrom,
                chrom_key=canonical_chrom(chrom),
                start=start,
                end=end,
                name=name,
                bed_score=bed_score,
                strand=strand,
                cage_score=cage_score,
                position=position,
                source_file=str(path),
                line_number=line_number,
            )
            records.append(record)
            seen_ids[name] += 1

    if not records:
        raise ValueError(f"Keine TSS-Eintraege in {path} gefunden.")
    duplicate_ids = [name for name, count in seen_ids.items() if count > 1]
    if duplicate_ids:
        LOGGER.warning(
            "%s enthaelt %d doppelte TSS-IDs. Dateinamen werden bei Bedarf "
            "um Koordinaten ergaenzt.",
            path,
            len(duplicate_ids),
        )
    return records


def open_text_auto(path: Path) -> Any:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("rt", encoding="utf-8")


def parse_annotation_attributes(text: str) -> dict[str, str]:
    """Liest sowohl GFF3- als auch GTF-Attribute."""
    attributes: dict[str, str] = {}
    for raw_item in text.strip().strip(";").split(";"):
        item = raw_item.strip()
        if not item:
            continue
        if "=" in item:
            key, value = item.split("=", 1)
            attributes[key.strip()] = unquote(value.strip())
            continue
        match = re.match(r'^([^\s]+)\s+"(.*)"$', item)
        if match:
            attributes[match.group(1)] = match.group(2)
            continue
        fields = item.split(None, 1)
        attributes[fields[0]] = fields[1].strip('"') if len(fields) == 2 else ""
    return attributes


def resolve_gene_annotation_path(requested: Path | None, script_path: Path) -> Path:
    if requested is not None:
        return requested

    folder = script_path.resolve().parent
    patterns = ("*.gff3", "*.gff3.gz", "*.gff", "*.gff.gz", "*.gtf", "*.gtf.gz")
    candidates = sorted({path for pattern in patterns for path in folder.glob(pattern)})
    if len(candidates) == 1:
        LOGGER.info("Genannotation automatisch gefunden: %s", candidates[0])
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            "Keine Genannotation neben dem Skript gefunden. Eine GFF3/GTF-Datei "
            "dort ablegen oder mit --gene-annotation angeben."
        )
    raise ValueError(
        "Mehrere moegliche Genannotationsdateien neben dem Skript gefunden; "
        "bitte mit --gene-annotation eindeutig auswaehlen:\n  "
        + "\n  ".join(str(path) for path in candidates)
    )


def read_gene_annotation(
    path: Path,
    feature_types: Sequence[str],
) -> tuple[list[Gene], dict[str, Any]]:
    accepted = {value.strip().lower() for value in feature_types if value.strip()}
    if not accepted:
        raise ValueError("Mindestens ein Gen-Feature-Typ muss angegeben werden.")

    records: list[Gene] = []
    seen_keys: set[tuple[str, int, int, str]] = set()
    feature_counts: Counter[str] = Counter()
    with open_text_auto(path) as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip() or raw_line.startswith("#"):
                continue
            fields = raw_line.rstrip("\n\r").split("\t")
            if len(fields) != 9:
                raise ValueError(
                    f"{path}:{line_number}: GFF3/GTF mit genau 9 Feldern erwartet."
                )
            (
                chrom,
                _source,
                feature_type,
                start_text,
                end_text,
                _score,
                strand,
                _phase,
                raw_attrs,
            ) = fields
            feature_counts[feature_type] += 1
            if feature_type.lower() not in accepted:
                continue
            try:
                start_1based = int(start_text)
                end_1based = int(end_text)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: start/end sind keine ganzen Zahlen."
                ) from exc
            if start_1based < 1 or end_1based < start_1based:
                raise ValueError(
                    f"{path}:{line_number}: Ungueltiges 1-basiertes Intervall "
                    f"[{start_1based}, {end_1based}]."
                )
            if strand not in ("+", "-", ".", "?"):
                raise ValueError(f"{path}:{line_number}: Ungueltiger Strang {strand!r}.")

            attributes = parse_annotation_attributes(raw_attrs)
            gene_id = (
                attributes.get("ID")
                or attributes.get("gene_id")
                or attributes.get("locus_tag")
                or attributes.get("Name")
                or f"gene_{canonical_chrom(chrom)}_{start_1based}_{end_1based}"
            )
            name = (
                attributes.get("Name")
                or attributes.get("gene_name")
                or attributes.get("locus_tag")
                or gene_id
            )
            biotype = (
                attributes.get("biotype")
                or attributes.get("gene_biotype")
                or attributes.get("gene_type")
            )
            start = start_1based - 1
            end = end_1based
            key = (canonical_chrom(chrom), start, end, gene_id)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            records.append(
                Gene(
                    chrom_input=chrom,
                    chrom_key=key[0],
                    start=start,
                    end=end,
                    gene_id=gene_id,
                    name=name,
                    strand=strand,
                    biotype=biotype,
                    feature_type=feature_type,
                    source_file=str(path),
                    line_number=line_number,
                )
            )

    if not records:
        available = ", ".join(
            f"{name}={count}" for name, count in feature_counts.most_common(20)
        )
        raise ValueError(
            f"Keine Features der Typen {sorted(accepted)} in {path} gefunden. "
            f"Vorhandene Typen: {available or 'keine'}"
        )
    records.sort(
        key=lambda gene: (
            gene.chrom_key,
            gene.start,
            gene.end,
            gene.gene_id,
            gene.line_number,
        )
    )
    summary = {
        "path": str(path),
        "accepted_feature_types": sorted(accepted),
        "loaded_genes": len(records),
        "all_feature_counts": dict(feature_counts),
    }
    return records, summary


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
    tss_chrom_keys: Sequence[str],
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
    for key in sorted(set(tss_chrom_keys)):
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


def tss_conflict_key(tss: TSS, summit: Summit) -> tuple[Any, ...]:
    cage = tss.cage_score if tss.cage_score is not None else -math.inf
    return (abs(summit.position - tss.position), -cage, tss.name, tss.position)


def select_positive_samples(
    tss_records: Sequence[TSS],
    summits_by_chrom: Mapping[str, Sequence[Summit]],
    bundles: Mapping[str, ChromBundle],
    max_distance: int,
    window_size: int,
) -> tuple[list[PositiveSample], list[dict[str, Any]], dict[str, int]]:
    proposals_by_summit: dict[tuple[str, int], list[tuple[TSS, Summit, int]]] = defaultdict(list)
    audit: list[dict[str, Any]] = []
    stats = Counter()

    summit_positions: dict[str, list[int]] = {
        chrom: [summit.position for summit in summits]
        for chrom, summits in summits_by_chrom.items()
    }

    for tss in tss_records:
        summits = summits_by_chrom.get(tss.chrom_key, ())
        positions = summit_positions.get(tss.chrom_key, [])
        left = bisect_left(positions, tss.position - max_distance)
        right = bisect_right(positions, tss.position + max_distance)
        candidates = list(summits[left:right])
        if not candidates:
            stats["no_summit_in_range"] += 1
            audit.append(
                {
                    "tss_id": tss.name,
                    "chromosome": tss.chrom_input,
                    "tss_position_0based": tss.position,
                    "strand": tss.strand,
                    "status": "no_summit_in_range",
                    "summit_position_0based": "",
                    "summit_score": "",
                    "candidate_count": 0,
                    "reason": f"kein Summit innerhalb von +/-{max_distance} bp",
                }
            )
            continue
        best = max(
            candidates,
            key=lambda summit: (
                summit.score,
                -abs(summit.position - tss.position),
                -summit.position,
            ),
        )
        proposals_by_summit[best.key].append((tss, best, len(candidates)))
        stats["tss_with_candidate"] += 1

    winners: list[tuple[TSS, Summit, int]] = []
    for summit_key, proposals in proposals_by_summit.items():
        proposals_sorted = sorted(
            proposals,
            key=lambda item: tss_conflict_key(item[0], item[1]),
        )
        winner = proposals_sorted[0]
        winners.append(winner)
        if len(proposals_sorted) > 1:
            stats["summit_conflicts"] += 1
            stats["duplicate_summit_losers"] += len(proposals_sorted) - 1
            winner_tss = winner[0]
            for loser_tss, loser_summit, loser_count in proposals_sorted[1:]:
                audit.append(
                    {
                        "tss_id": loser_tss.name,
                        "chromosome": loser_tss.chrom_input,
                        "tss_position_0based": loser_tss.position,
                        "strand": loser_tss.strand,
                        "status": "duplicate_summit_loser",
                        "summit_position_0based": loser_summit.position,
                        "summit_score": loser_summit.score,
                        "candidate_count": loser_count,
                        "reason": f"Summit bereits der TSS {winner_tss.name} zugeordnet",
                    }
                )

    # Eine feste Reihenfolge entkoppelt die reproduzierbare Zufallsziehung von
    # der Einfuegereihenfolge der Summit-Vorschlaege.
    winners.sort(
        key=lambda item: (
            item[0].chrom_key,
            item[1].position,
            item[0].strand,
            item[0].position,
            item[0].name,
            item[0].source_file,
            item[0].line_number,
        )
    )
    positive: list[PositiveSample] = []
    for tss, summit, candidate_count in winners:
        bundle = bundles[tss.chrom_key]
        # Positive Fenster liegen exakt auf dem Summit; der zufaellige Versatz
        # passiert erst im Training (Zuschnitt aus dem Umfeld).
        valid, start, end = is_valid_window(summit.position, window_size, bundle.length)
        if not valid:
            stats["window_outside_chromosome"] += 1
            audit.append(
                {
                    "tss_id": tss.name,
                    "chromosome": tss.chrom_input,
                    "tss_position_0based": tss.position,
                    "strand": tss.strand,
                    "status": "window_outside_chromosome",
                    "summit_position_0based": summit.position,
                    "summit_score": summit.score,
                    "candidate_count": candidate_count,
                    "reason": f"Fenster [{start}, {end}) liegt ausserhalb des Chromosoms",
                }
            )
            continue

        center = summit.position
        window_shift = 0

        summit_genomic_offset = summit.position - tss.position
        summit_oriented_offset = (
            summit_genomic_offset if tss.strand == "+" else -summit_genomic_offset
        )
        genomic_offset = center - tss.position
        oriented_offset = genomic_offset if tss.strand == "+" else -genomic_offset
        positive.append(
            PositiveSample(
                tss=tss,
                summit=summit,
                candidate_count=candidate_count,
                bundle=bundle,
                window_start=start,
                window_end=end,
                window_shift=window_shift,
                genomic_offset=genomic_offset,
                oriented_offset=oriented_offset,
                summit_genomic_offset=summit_genomic_offset,
                summit_oriented_offset=summit_oriented_offset,
            )
        )
        stats["positive_samples"] += 1

    positive.sort(
        key=lambda sample: (
            sample.bundle.fasta_name,
            sample.window_start,
            sample.tss.strand,
            sample.tss.name,
        )
    )
    return positive, audit, dict(stats)


def select_annotation_positive_samples(
    tss_records: Sequence[TSS],
    genes: Sequence[Gene],
    summits_by_chrom: Mapping[str, Sequence[Summit]],
    positive_tss_samples: Sequence[PositiveSample],
    bundles: Mapping[str, ChromBundle],
    min_tss_distance: int,
    window_size: int,
) -> tuple[
    list[AnnotationPositiveSample],
    list[AnnotationPositiveSample],
    list[dict[str, Any]],
    dict[str, int],
]:
    """Klassifiziert alle nicht-TSS-positiven Summits anhand ihrer Summit-Base.

    Ein Summit wird erst dann als genisch oder nichtgenisch betrachtet, wenn
    sein minimaler Abstand zu *jeder* TSS mindestens ``min_tss_distance`` ist.
    Genmitgliedschaft verwendet 0-basierte, halb-offene Genintervalle und damit
    exakt ``gene.start <= summit.position < gene.end``.
    """
    tss_by_chrom_position: dict[str, dict[int, list[TSS]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for tss in tss_records:
        tss_by_chrom_position[tss.chrom_key][tss.position].append(tss)
    tss_positions = {
        chrom_key: sorted(by_position)
        for chrom_key, by_position in tss_by_chrom_position.items()
    }
    genes_by_chrom: dict[str, list[Gene]] = defaultdict(list)
    for gene in genes:
        if gene.chrom_key in bundles:
            genes_by_chrom[gene.chrom_key].append(gene)
    for chrom_genes in genes_by_chrom.values():
        chrom_genes.sort(
            key=lambda gene: (
                gene.start,
                gene.end,
                gene.gene_id,
                gene.source_file,
                gene.line_number,
            )
        )

    positive_tss_by_summit = {
        sample.summit.key: sample for sample in positive_tss_samples
    }
    positive_genic: list[AnnotationPositiveSample] = []
    positive_noncoding: list[AnnotationPositiveSample] = []
    audit: list[dict[str, Any]] = []
    stats = Counter()

    for chrom_key in sorted(summits_by_chrom):
        bundle = bundles[chrom_key]
        positions = tss_positions.get(chrom_key, [])
        by_position = tss_by_chrom_position.get(chrom_key, {})
        chrom_genes = genes_by_chrom.get(chrom_key, [])
        active_genes: list[Gene] = []
        next_gene_index = 0

        for summit in summits_by_chrom[chrom_key]:
            stats["summits_considered"] += 1
            while (
                next_gene_index < len(chrom_genes)
                and chrom_genes[next_gene_index].start <= summit.position
            ):
                active_genes.append(chrom_genes[next_gene_index])
                next_gene_index += 1
            active_genes = [
                gene for gene in active_genes if gene.end > summit.position
            ]
            containing_genes = [
                gene
                for gene in active_genes
                if gene.start <= summit.position < gene.end
            ]
            containing_genes.sort(
                key=lambda gene: (
                    gene.end - gene.start,
                    gene.start,
                    gene.end,
                    gene.gene_id,
                    gene.source_file,
                    gene.line_number,
                )
            )

            insertion = bisect_left(positions, summit.position)
            nearest_positions: list[int] = []
            if insertion < len(positions):
                nearest_positions.append(positions[insertion])
            if insertion > 0:
                nearest_positions.append(positions[insertion - 1])
            nearest_tss_candidates = [
                tss
                for position in nearest_positions
                for tss in by_position[position]
            ]
            if not nearest_tss_candidates:
                raise AssertionError(
                    f"Interner Fehler: keine TSS auf Datenchromosom {chrom_key}."
                )
            nearest_tss = min(
                nearest_tss_candidates,
                key=lambda tss: (
                    abs(summit.position - tss.position),
                    -(tss.cage_score if tss.cage_score is not None else -math.inf),
                    tss.name,
                    tss.source_file,
                    tss.line_number,
                ),
            )
            nearest_distance = abs(summit.position - nearest_tss.position)
            common_audit: dict[str, Any] = {
                "chromosome": bundle.fasta_name,
                "summit_position_0based": summit.position,
                "summit_score": summit.score,
                "nearest_tss_id": nearest_tss.name,
                "nearest_tss_position_0based": nearest_tss.position,
                "nearest_tss_distance": nearest_distance,
                "overlapping_gene_count": len(containing_genes),
                "gene_id": containing_genes[0].gene_id if containing_genes else None,
            }

            positive_tss = positive_tss_by_summit.get(summit.key)
            if positive_tss is not None:
                stats["positive_tss"] += 1
                audit.append(
                    {
                        **common_audit,
                        "positive_type": "tss",
                        "window_start_0based": positive_tss.window_start,
                        "window_end_0based_exclusive": positive_tss.window_end,
                        "window_shift": positive_tss.window_shift,
                        "status": "included_positive_tss",
                        "reason": "",
                    }
                )
                continue

            if nearest_distance < min_tss_distance:
                stats["excluded_tss_proximal"] += 1
                audit.append(
                    {
                        **common_audit,
                        "positive_type": None,
                        "window_start_0based": None,
                        "window_end_0based_exclusive": None,
                        "window_shift": None,
                        "status": "excluded_tss_proximal",
                        "reason": (
                            f"Abstand {nearest_distance} bp ist kleiner als "
                            f"{min_tss_distance} bp; Summit wurde nicht als "
                            "positives TSS-Beispiel ausgewaehlt"
                        ),
                    }
                )
                continue

            valid, start, end = is_valid_window(
                summit.position, window_size, bundle.length
            )
            if not valid:
                stats["excluded_window_outside_chromosome"] += 1
                audit.append(
                    {
                        **common_audit,
                        "positive_type": None,
                        "window_start_0based": start,
                        "window_end_0based_exclusive": end,
                        "window_shift": None,
                        "status": "excluded_window_outside_chromosome",
                        "reason": "Unverschobenes Summit-Fenster liegt ausserhalb des Chromosoms",
                    }
                )
                continue

            window_shift = 0

            positive_type = "genic" if containing_genes else "noncoding"
            sample = AnnotationPositiveSample(
                positive_type=positive_type,
                summit=summit,
                bundle=bundle,
                window_start=start,
                window_end=end,
                window_shift=window_shift,
                nearest_tss=nearest_tss,
                nearest_tss_distance=nearest_distance,
                gene=containing_genes[0] if containing_genes else None,
                overlapping_gene_count=len(containing_genes),
            )
            if positive_type == "genic":
                positive_genic.append(sample)
                stats["positive_genic"] += 1
            else:
                positive_noncoding.append(sample)
                stats["positive_noncoding"] += 1
            audit.append(
                {
                    **common_audit,
                    "positive_type": positive_type,
                    "window_start_0based": start,
                    "window_end_0based_exclusive": end,
                    "window_shift": window_shift,
                    "status": f"included_positive_{positive_type}",
                    "reason": "",
                }
            )

    sort_key = lambda sample: (
        sample.bundle.fasta_name,
        sample.window_start,
        sample.summit.position,
    )
    positive_genic.sort(key=sort_key)
    positive_noncoding.sort(key=sort_key)
    stats["positive_total"] = (
        stats["positive_tss"]
        + stats["positive_genic"]
        + stats["positive_noncoding"]
    )
    stats["summits_not_in_positive_output"] = (
        stats["summits_considered"] - stats["positive_total"]
    )
    return positive_genic, positive_noncoding, audit, dict(stats)


def estimate_offset_distribution(
    offsets: Sequence[int],
    max_distance: int,
    method: str,
    bandwidth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    support = np.arange(-max_distance, max_distance + 1, dtype=np.int32)
    counts = np.zeros(support.size, dtype=np.int64)
    if offsets:
        values = np.asarray(offsets, dtype=np.int64)
        if np.any(values < -max_distance) or np.any(values > max_distance):
            raise ValueError("Offset ausserhalb der definierten Distanz gefunden.")
        counts = np.bincount(values + max_distance, minlength=support.size).astype(np.int64)
    if counts.sum() == 0:
        probabilities = np.zeros_like(support, dtype=np.float64)
        return support, counts, probabilities

    if method == "empirical":
        density = counts.astype(np.float64)
    elif method == "kde":
        if bandwidth <= 0:
            raise ValueError("--kde-bandwidth muss groesser als 0 sein.")
        radius = min(max_distance, max(1, int(math.ceil(4.0 * bandwidth))))
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / bandwidth) ** 2)
        kernel /= kernel.sum()
        density = np.convolve(counts.astype(np.float64), kernel, mode="same")
    else:
        raise ValueError(f"Unbekannte Verteilungsmethode: {method}")

    probabilities = density / density.sum()
    return support, counts, probabilities


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


def candidate_pval_profile(
    tss: TSS,
    bundle: ChromBundle,
    pval_bw: Any,
    window_size: int,
    max_distance: int,
    pval_threshold: float,
    positive_intervals: tuple[np.ndarray, np.ndarray],
    exclude_positive_overlap: bool,
) -> tuple[np.ndarray, np.ndarray]:
    half = window_size // 2
    support = np.arange(-max_distance, max_distance + 1, dtype=np.int64)
    broad_start = tss.position - max_distance - half
    broad_length = window_size + 2 * max_distance
    broad_end = broad_start + broad_length

    # Ausserhalb des Chromosoms bleibt +inf stehen und kann daher nie negativ sein.
    pval = np.full(broad_length, np.inf, dtype=np.float32)
    clipped_start = max(0, broad_start)
    clipped_end = min(bundle.length, broad_end)
    if clipped_start < clipped_end:
        fetched = bigwig_values(
            pval_bw,
            bundle.pval_name,
            clipped_start,
            clipped_end,
            missing_value=0.0,
        )
        insert_start = clipped_start - broad_start
        pval[insert_start : insert_start + fetched.size] = fetched

    max_by_genomic_offset = rolling_max(pval, window_size)
    if max_by_genomic_offset.size != support.size:
        raise AssertionError("Interner Fehler: falsche Anzahl von Offset-Fenstern.")
    max_by_oriented_offset = (
        max_by_genomic_offset
        if tss.strand == "+"
        else max_by_genomic_offset[::-1].copy()
    )
    valid = max_by_oriented_offset <= pval_threshold

    if exclude_positive_overlap and np.any(valid):
        genomic_offsets = support if tss.strand == "+" else -support
        centers = tss.position + genomic_offsets
        starts = centers - half
        ends = starts + window_size
        merged_starts, merged_ends = positive_intervals
        valid &= ~overlap_mask(starts, ends, merged_starts, merged_ends)

    return max_by_oriented_offset.astype(np.float32, copy=False), valid


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


def gc_bin_indices(values: Sequence[float], bin_width: float) -> tuple[np.ndarray, int]:
    bin_count = int(math.ceil(1.0 / bin_width))
    array = np.asarray(values, dtype=np.float64)
    indices = np.floor(array / bin_width + 1e-12).astype(np.int64)
    return np.clip(indices, 0, bin_count - 1), bin_count


def merged_gene_intervals(
    genes: Sequence[Gene],
    bundles: Mapping[str, ChromBundle],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom_key, bundle in bundles.items():
        intervals = [
            (max(0, gene.start), min(bundle.length, gene.end))
            for gene in genes
            if gene.chrom_key == chrom_key and gene.start < bundle.length and gene.end > 0
        ]
        if not intervals:
            raise ValueError(
                f"Die Genannotation enthaelt auf Datenchromosom {bundle.fasta_name!r} "
                "keine Genintervalle. Chromosomennamen und Assembly-Kompatibilitaet pruefen."
            )
        result[chrom_key] = merge_intervals(intervals)
    return result


def intergenic_regions(
    merged_starts: np.ndarray,
    merged_ends: np.ndarray,
    chrom_length: int,
    window_size: int,
) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    cursor = 0
    for start, end in zip(merged_starts.tolist(), merged_ends.tolist()):
        if start - cursor >= window_size:
            regions.append((cursor, start))
        cursor = max(cursor, end)
    if chrom_length - cursor >= window_size:
        regions.append((cursor, chrom_length))
    return regions


def scan_filtered_candidate_region(
    bundle: ChromBundle,
    fasta: Any,
    pval_bw: Any,
    first_start: int,
    stop_start: int,
    window_size: int,
    grid_phase: int,
    pval_threshold: float,
    excluded_starts: np.ndarray,
    excluded_ends: np.ndarray,
    scan_chunk_size: int,
    gene: Gene | None,
    candidate_sink: Callable[[Sequence[WindowCandidate]], int] | None = None,
    candidate_step: int | None = None,
    origin_region: str | None = None,
) -> tuple[list[WindowCandidate], dict[str, int]]:
    """Scannt Fensterstarts; ein Sink erlaubt ueberlappende, grosse Pools."""
    candidates: list[WindowCandidate] = []
    stats = {
        "tested_windows": 0,
        "pval_valid_windows": 0,
        "hard_valid_windows": 0,
        "gc_measurable_windows": 0,
    }
    if stop_start <= first_start:
        return candidates, stats

    step = window_size if candidate_step is None else candidate_step
    for chunk_first in range(first_start, stop_start, min(scan_chunk_size, 50_000) if candidate_sink else scan_chunk_size):
        chunk_stop = min(stop_start, chunk_first + (min(scan_chunk_size, 50_000) if candidate_sink else scan_chunk_size))
        first_relative = (grid_phase - chunk_first) % step
        relative_indices = np.arange(
            first_relative,
            chunk_stop - chunk_first,
            step,
            dtype=np.int64,
        )
        if relative_indices.size == 0:
            continue
        pval_maxima, gc_fractions = window_metric_arrays(
            bundle=bundle,
            fasta=fasta,
            pval_bw=pval_bw,
            first_start=chunk_first,
            stop_start=chunk_stop,
            window_size=window_size,
        )
        starts = chunk_first + relative_indices
        ends = starts + window_size
        pval_subset = pval_maxima[relative_indices]
        gc_subset = gc_fractions[relative_indices]
        pval_valid = pval_subset <= pval_threshold
        nonoverlapping = ~overlap_mask(
            starts,
            ends,
            excluded_starts,
            excluded_ends,
        )
        hard_valid = pval_valid & nonoverlapping
        measurable = hard_valid & np.isfinite(gc_subset)
        stats["tested_windows"] += int(relative_indices.size)
        stats["pval_valid_windows"] += int(np.count_nonzero(pval_valid))
        stats["hard_valid_windows"] += int(np.count_nonzero(hard_valid))
        stats["gc_measurable_windows"] += int(np.count_nonzero(measurable))
        chunk_candidates: list[WindowCandidate] = []
        for relative_index in np.flatnonzero(measurable).tolist():
            chunk_candidates.append(
                WindowCandidate(
                    chrom_key=bundle.chrom_key,
                    window_start=int(starts[relative_index]),
                    selection_pval_max=float(pval_subset[relative_index]),
                    gc_fraction=float(gc_subset[relative_index]),
                    gene=gene,
                    origin_region=origin_region,
                )
            )
        if candidate_sink is None:
            candidates.extend(chunk_candidates)
        elif chunk_candidates:
            candidate_sink(chunk_candidates)
    return candidates, stats


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
        ])
        for candidate_id, chrom, start, pval, gc in store.db.execute(
            "SELECT id,chrom,start,pval,gc FROM candidates ORDER BY chrom,start,id"
        ):
            writer.writerow([
                candidate_id, bundles[chrom].fasta_name, start,
                start + window_size, f"{gc:.17g}", f"{pval:.17g}",
                store.source_names(candidate_id),
            ])


def build_dense_tss_pool(
    tss_records: Sequence[TSS], positive_samples: Sequence[Any],
    distributions: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    bundles: Mapping[str, ChromBundle], fasta: Any, pval_bw: Any,
    window_size: int, max_distance: int, pval_threshold: float,
    exclude_positive_overlap: bool, store: CandidateStore,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positive_keys = {
        (sample.tss.source_file, sample.tss.line_number)
        for sample in positive_samples if isinstance(sample, PositiveSample)
    }
    excluded = intervals_by_chrom(positive_samples, bundles)
    audit: list[dict[str, Any]] = []
    with_candidates = 0
    for tss in sorted(tss_records, key=lambda x: (
        x.chrom_key, x.strand, x.position, x.name, x.line_number
    )):
        if (tss.source_file, tss.line_number) in positive_keys:
            continue
        support, _counts, probabilities = distributions[tss.strand]
        if not np.any(probabilities > 0):
            audit.append({"tss_id": tss.name, "status": "no_positive_offset_distribution_for_strand"})
            continue
        bundle = bundles[tss.chrom_key]
        maxima, valid = candidate_pval_profile(
            tss, bundle, pval_bw, window_size, max_distance, pval_threshold,
            excluded[tss.chrom_key], exclude_positive_overlap,
        )
        valid &= probabilities > 0
        indices = np.flatnonzero(valid)
        if indices.size:
            half = window_size // 2
            offsets = support[indices] if tss.strand == "+" else -support[indices]
            starts = tss.position + offsets - half
            first = int(np.min(starts))
            stop = int(np.max(starts)) + 1
            gc_values = window_gc_fractions(bundle, fasta, first, stop, window_size)
            rows = [WindowCandidate(
                chrom_key=tss.chrom_key, window_start=int(start),
                selection_pval_max=float(maxima[index]),
                gc_fraction=float(gc_values[int(start) - first]), tss=tss,
                offset_probability=float(probabilities[index]),
            ) for index, start in zip(indices, starts)
                if math.isfinite(float(gc_values[int(start) - first]))]
            store.add_many(rows)
            if rows:
                with_candidates += 1
        audit.append({
            "tss_id": tss.name, "chromosome": bundle.fasta_name,
            "strand": tss.strand, "tested_offsets": int(np.count_nonzero(probabilities)),
            "valid_offsets": int(indices.size),
            "status": "included_in_pool" if indices.size else "no_valid_negative_offset",
        })
    store.finish()
    return audit, {
        "selection_mode": "all_valid_tss_offsets_then_gc_match",
        "tss_with_candidates": with_candidates,
        "candidate_pool_size": store.count,
    }


def build_dense_genic_pool(
    genes: Sequence[Gene], previous_samples: Sequence[Any],
    bundles: Mapping[str, ChromBundle], fasta: Any, pval_bw: Any,
    window_size: int, pval_threshold: float, placement: str,
    scan_chunk_size: int, store: CandidateStore,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    exclusions = intervals_by_chrom(previous_samples, bundles)
    audit: list[dict[str, Any]] = []
    totals: Counter[str] = Counter()
    half = window_size // 2
    for gene in genes:
        bundle = bundles.get(gene.chrom_key)
        if bundle is None:
            continue
        if placement == "fully-contained":
            first = max(0, gene.start)
            stop = min(bundle.length - window_size, gene.end - window_size) + 1
        else:
            first_center = max(half, gene.start)
            last_center = min(bundle.length - window_size + half, gene.end - 1)
            first, stop = first_center - half, last_center - half + 1
        before = store.count
        _unused, stats = scan_filtered_candidate_region(
            bundle, fasta, pval_bw, first, stop, window_size, 0,
            pval_threshold, *exclusions[gene.chrom_key], scan_chunk_size,
            gene, candidate_sink=store.add_many, candidate_step=1,
        ) if stop > first else ([], {key: 0 for key in (
            "tested_windows", "pval_valid_windows", "hard_valid_windows",
            "gc_measurable_windows",
        )})
        totals.update(stats)
        audit.append({
            "gene_id": gene.gene_id, "chromosome": bundle.fasta_name,
            "gene_start_0based": gene.start, "gene_end_0based_exclusive": gene.end,
            **stats, "pool_candidate_windows": store.count - before,
            "available_gc_bins": "", "status": (
                "included_in_pool" if store.count > before else "no_new_candidate"
            ), "reason": "",
        })
    store.finish()
    return audit, {"pool_mode": "all_valid_overlapping_genic_starts",
                   "candidate_pool_size": store.count, "placement": placement,
                   **dict(totals)}


def build_dense_noncoding_pool(
    genes: Sequence[Gene], previous_samples: Sequence[Any],
    bundles: Mapping[str, ChromBundle], fasta: Any, pval_bw: Any,
    window_size: int, pval_threshold: float, scan_chunk_size: int,
    store: CandidateStore,
) -> dict[str, Any]:
    gene_intervals = merged_gene_intervals(genes, bundles)
    exclusions = intervals_by_chrom(previous_samples, bundles)
    totals: Counter[str] = Counter()
    gap_count = 0
    for chrom_key, bundle in sorted(bundles.items()):
        starts, ends = gene_intervals[chrom_key]
        for first, gap_end in intergenic_regions(starts, ends, bundle.length, window_size):
            gap_count += 1
            _unused, stats = scan_filtered_candidate_region(
                bundle, fasta, pval_bw, first, gap_end - window_size + 1,
                window_size, 0, pval_threshold, *exclusions[chrom_key],
                scan_chunk_size, None, candidate_sink=store.add_many,
                candidate_step=1,
                origin_region=f"{bundle.fasta_name}:{first}-{gap_end}",
            )
            totals.update(stats)
    store.finish()
    return {"pool_mode": "all_valid_overlapping_intergenic_starts",
            "candidate_pool_size": store.count, "intergenic_regions": gap_count,
            **dict(totals)}


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


def materialize_matched_samples(
    matches: Sequence[tuple[int, WindowCandidate, float]],
    negative_type: str,
    bundles: Mapping[str, ChromBundle],
    positive_gc_fractions: Sequence[float],
    window_size: int,
    gc_bin_width: float,
    method: str,
    origin_lookup: Callable[[int], str] | None = None,
) -> tuple[
    list[AnnotationNegativeSample],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    positive_values = np.asarray(positive_gc_fractions, dtype=np.float64)
    positive_bins, bin_count = gc_bin_indices(positive_values, gc_bin_width)
    positive_counts = np.bincount(
        positive_bins, minlength=bin_count
    ).astype(np.int64)
    realized_gc = [candidate.gc_fraction for _, candidate, _ in matches]
    if realized_gc:
        realized_bins, _ = gc_bin_indices(realized_gc, gc_bin_width)
        realized_counts = np.bincount(
            realized_bins, minlength=bin_count
        ).astype(np.int64)
    else:
        realized_counts = np.zeros(bin_count, dtype=np.int64)
    positive_fractions = positive_counts / max(1, int(positive_counts.sum()))
    realized_fractions = realized_counts / max(1, int(realized_counts.sum()))
    half = window_size // 2
    samples: list[AnnotationNegativeSample] = []
    selection_rows: list[dict[str, Any]] = []

    for candidate_id, candidate, target_gc in matches:
        bin_index = int(gc_bin_indices([candidate.gc_fraction], gc_bin_width)[0][0])
        bundle = bundles[candidate.chrom_key]
        difference = abs(candidate.gc_fraction - target_gc)
        samples.append(
            AnnotationNegativeSample(
                negative_type=negative_type,
                bundle=bundle,
                center=candidate.window_start + half,
                window_start=candidate.window_start,
                window_end=candidate.window_start + window_size,
                selection_pval_max=candidate.selection_pval_max,
                gc_target_fraction=float(target_gc),
                gc_fraction_at_selection=candidate.gc_fraction,
                gc_absolute_difference=float(difference),
                gc_match_tolerance=None,
                gene=candidate.gene,
                gc_bin_index=bin_index,
                gc_bin_start=float(bin_index * gc_bin_width),
                gc_bin_end=float(min(1.0, (bin_index + 1) * gc_bin_width)),
                gc_bin_width=float(gc_bin_width),
                gc_assignment_mode=(
                    "distribution_matching_quantile_audit_pairing"
                    if method == "distribution" else method
                ),
                gc_target_bin_quota=None,
                origin_ids=origin_lookup(candidate_id) if origin_lookup else None,
            )
        )
        selection_rows.append(
            {
                "candidate_id": int(candidate_id),
                "chromosome": bundle.fasta_name,
                "window_start_0based": candidate.window_start,
                "window_end_0based_exclusive": candidate.window_start + window_size,
                "gene_id": candidate.gene.gene_id if candidate.gene is not None else None,
                "target_gc_fraction": float(target_gc),
                "matched_gc_fraction": candidate.gc_fraction,
                "absolute_gc_difference": float(difference),
                "selection_pval_max": candidate.selection_pval_max,
            }
        )

    samples.sort(
        key=lambda sample: (
            sample.bundle.fasta_name,
            sample.window_start,
            sample.gene.gene_id if sample.gene is not None else "",
        )
    )
    histogram_rows = []
    for bin_index in range(bin_count):
        if positive_counts[bin_index] or realized_counts[bin_index]:
            row = {
                "bin_index": int(bin_index),
                "bin_start": float(bin_index * gc_bin_width),
                "bin_end": float(min(1.0, (bin_index + 1) * gc_bin_width)),
                "positive_count": int(positive_counts[bin_index]),
                "realized_count": int(realized_counts[bin_index]),
                "positive_fraction": float(positive_fractions[bin_index]),
                "realized_fraction": float(realized_fractions[bin_index]),
                "fraction_difference_realized_minus_positive": float(
                    realized_fractions[bin_index] - positive_fractions[bin_index]
                ),
            }
            if negative_type == "genic":
                row.update(
                    {
                        "realized_genic_count": int(realized_counts[bin_index]),
                    }
                )
            else:
                row.update(
                    {
                        "realized_noncoding_count": int(realized_counts[bin_index]),
                    }
                )
            histogram_rows.append(row)
    return samples, selection_rows, histogram_rows


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


def offset_filename_token(offset: int) -> str:
    sign = "m" if offset < 0 else "p"
    return f"{sign}{abs(int(offset)):03d}"


def sample_type_name(sample: Any) -> str:
    if isinstance(sample, PositiveSample):
        return "positive_tss"
    if isinstance(sample, AnnotationPositiveSample):
        return f"positive_{sample.positive_type}"
    if isinstance(sample, NegativeSample):
        return "negative_tss"
    if isinstance(sample, AnnotationNegativeSample):
        return f"negative_{sample.negative_type}"
    raise TypeError(f"Unbekannter Sample-Typ: {type(sample).__name__}")


def sample_group_name(sample: Any) -> str:
    return (
        "positive"
        if isinstance(sample, (PositiveSample, AnnotationPositiveSample))
        else "negative"
    )


def unique_sample_ids(samples: Sequence[Any]) -> dict[int, str]:
    used: Counter[tuple[str, str, str]] = Counter()
    result: dict[int, str] = {}
    for sample in samples:
        sample_type = sample_type_name(sample)
        chrom = safe_filename(sample.bundle.fasta_name)
        if isinstance(sample, NegativeSample):
            tss_base = safe_filename(sample.tss.name)
            base = f"{tss_base}__offset_{offset_filename_token(sample.oriented_offset)}"
        elif isinstance(sample, PositiveSample):
            base = safe_filename(sample.tss.name)
        elif isinstance(sample, AnnotationPositiveSample):
            base = f"summit_{sample.summit.position}"
        elif sample.gene is not None:
            base = safe_filename(sample.gene.gene_id)
        else:
            base = f"intergenic_{sample.window_start}_{sample.window_end}"
        key = (sample_type, chrom, base)
        used[key] += 1
        if used[key] == 1:
            sample_id = base
        else:
            if isinstance(sample, (PositiveSample, NegativeSample)):
                strand_text = "plus" if sample.tss.strand == "+" else "minus"
                suffix = f"tss_{sample.tss.position}_{strand_text}"
            elif isinstance(sample, AnnotationPositiveSample):
                suffix = f"summit_{sample.summit.position}"
            else:
                suffix = f"window_{sample.window_start}_{sample.window_end}"
            sample_id = f"{base}__{suffix}_{used[key]}"
        result[id(sample)] = sample_id
    return result


def format_scalar(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.10g}"
    return str(value)


def positive_class_definitions(
    max_summit_distance: int, min_nontss_distance: int,
) -> dict[str, str]:
    """Kurze, auch in den Sample-Ausgaben sichtbare Klassendefinitionen."""
    return {
        "positive_tss": (
            f"Zugeordneter Peak-Summit hoechstens {max_summit_distance} bp "
            "von einer TSS entfernt."
        ),
        "positive_genic": (
            f"Nicht-TSS-Summit mindestens {min_nontss_distance} bp "
            "von jeder TSS entfernt; Summit-Base liegt in einem Gen."
        ),
        "positive_noncoding": (
            f"Nicht-TSS-Summit mindestens {min_nontss_distance} bp "
            "von jeder TSS entfernt; Summit-Base liegt ausserhalb annotierter Genintervalle."
        ),
    }


def sample_base_metadata(
    sample: Any,
    sample_id: str,
    sample_type: str,
    window_size: int,
    center_index: int | None = None,
) -> dict[str, Any]:
    group = sample_group_name(sample)
    tss = sample.tss if isinstance(sample, (PositiveSample, NegativeSample)) else None
    gene = (
        sample.gene
        if isinstance(sample, (AnnotationPositiveSample, AnnotationNegativeSample))
        else None
    )
    metadata: dict[str, Any] = {
        "sample_id": sample_id,
        "label": 1 if group == "positive" else 0,
        "group": group,
        "sample_type": sample_type,
        "positive_type": sample_type.removeprefix("positive_") if group == "positive" else None,
        "negative_type": sample_type.removeprefix("negative_") if group == "negative" else None,
        "origin_ids": getattr(sample, "origin_ids", None),
        "chromosome": sample.bundle.fasta_name,
        "chromosome_input_bed": tss.chrom_input if tss is not None else None,
        "window_start_0based": sample.window_start,
        "window_end_0based_exclusive": sample.window_end,
        "window_start_1based": sample.window_start + 1,
        "window_end_1based_inclusive": sample.window_end,
        "window_length": window_size,
        "center_position_0based": sample.center,
        "center_position_1based": sample.center + 1,
        "center_index_0based": window_size // 2 if center_index is None else int(center_index),
        "tss_id": tss.name if tss is not None else None,
        "tss_start_0based": tss.start if tss is not None else None,
        "tss_end_0based_exclusive": tss.end if tss is not None else None,
        "tss_position_0based": tss.position if tss is not None else None,
        "tss_position_1based": tss.position + 1 if tss is not None else None,
        "tss_strand": tss.strand if tss is not None else None,
        "tss_cage_score": tss.cage_score if tss is not None else None,
        "gene_id": gene.gene_id if gene is not None else None,
        "gene_name": gene.name if gene is not None else None,
        "gene_feature_type": gene.feature_type if gene is not None else None,
        "gene_biotype": gene.biotype if gene is not None else None,
        "gene_chromosome_input": gene.chrom_input if gene is not None else None,
        "gene_start_0based": gene.start if gene is not None else None,
        "gene_end_0based_exclusive": gene.end if gene is not None else None,
        "gene_strand": gene.strand if gene is not None else None,
        "genomic_offset_center_minus_tss": (
            sample.genomic_offset if isinstance(sample, (PositiveSample, NegativeSample)) else None
        ),
        "strand_oriented_offset": (
            sample.oriented_offset if isinstance(sample, (PositiveSample, NegativeSample)) else None
        ),
        "absolute_tss_center_distance": (
            abs(sample.genomic_offset)
            if isinstance(sample, (PositiveSample, NegativeSample))
            else None
        ),
        "nearest_tss_id": (
            sample.nearest_tss.name
            if isinstance(sample, AnnotationPositiveSample)
            else None
        ),
        "nearest_tss_position_0based": (
            sample.nearest_tss.position
            if isinstance(sample, AnnotationPositiveSample)
            else None
        ),
        "nearest_tss_strand": (
            sample.nearest_tss.strand
            if isinstance(sample, AnnotationPositiveSample)
            else None
        ),
        "nearest_tss_summit_distance": (
            sample.nearest_tss_distance
            if isinstance(sample, AnnotationPositiveSample)
            else None
        ),
        "overlapping_gene_count": (
            sample.overlapping_gene_count
            if isinstance(sample, AnnotationPositiveSample)
            else None
        ),
        "sequence_orientation": "genomic_plus",
    }
    if isinstance(sample, PositiveSample):
        oriented_shift = (
            sample.window_shift
            if sample.tss.strand == "+"
            else -sample.window_shift
        )
        metadata.update(
            {
                "summit_position_0based": sample.summit.position,
                "summit_position_1based": sample.summit.position + 1,
                "summit_score": sample.summit.score,
                "summits_within_tss_range": sample.candidate_count,
                "summit_index_0based": sample.summit_index,
                "genomic_shift_center_minus_summit": sample.window_shift,
                "strand_oriented_shift_center_minus_summit": oriented_shift,
                "genomic_offset_summit_minus_tss": sample.summit_genomic_offset,
                "strand_oriented_offset_summit_minus_tss": sample.summit_oriented_offset,
                "absolute_tss_summit_distance": abs(sample.summit_genomic_offset),
            }
        )
    elif isinstance(sample, AnnotationPositiveSample):
        metadata.update(
            {
                "summit_position_0based": sample.summit.position,
                "summit_position_1based": sample.summit.position + 1,
                "summit_score": sample.summit.score,
                "summits_within_tss_range": None,
                "summit_index_0based": sample.summit_index,
                "genomic_shift_center_minus_summit": sample.window_shift,
                "strand_oriented_shift_center_minus_summit": None,
                "genomic_offset_summit_minus_tss": None,
                "strand_oriented_offset_summit_minus_tss": None,
                "absolute_tss_summit_distance": sample.nearest_tss_distance,
                "selection_pval_max": None,
                "offset_observed_count": None,
                "offset_estimated_probability": None,
                "offset_support_mode": None,
            }
        )
    elif isinstance(sample, NegativeSample):
        metadata.update(
            {
                "summit_position_0based": None,
                "summit_position_1based": None,
                "summit_score": None,
                "summits_within_tss_range": None,
                "summit_index_0based": None,
                "genomic_shift_center_minus_summit": None,
                "strand_oriented_shift_center_minus_summit": None,
                "genomic_offset_summit_minus_tss": None,
                "strand_oriented_offset_summit_minus_tss": None,
                "absolute_tss_summit_distance": None,
                "selection_pval_max": sample.selection_pval_max,
                "offset_observed_count": sample.offset_observed_count,
                "offset_estimated_probability": sample.offset_estimated_probability,
                "offset_support_mode": sample.offset_support_mode,
            }
        )
    else:
        metadata.update(
            {
                "summit_position_0based": None,
                "summit_position_1based": None,
                "summit_score": None,
                "summits_within_tss_range": None,
                "summit_index_0based": None,
                "genomic_shift_center_minus_summit": None,
                "strand_oriented_shift_center_minus_summit": None,
                "genomic_offset_summit_minus_tss": None,
                "strand_oriented_offset_summit_minus_tss": None,
                "absolute_tss_summit_distance": None,
                "selection_pval_max": sample.selection_pval_max,
                "offset_observed_count": None,
                "offset_estimated_probability": None,
                "offset_support_mode": None,
                "gc_target_fraction": sample.gc_target_fraction,
                "gc_fraction_at_selection": sample.gc_fraction_at_selection,
                "gc_absolute_difference": sample.gc_absolute_difference,
                "gc_match_tolerance": sample.gc_match_tolerance,
                "gc_bin_index": sample.gc_bin_index,
                "gc_bin_start": sample.gc_bin_start,
                "gc_bin_end": sample.gc_bin_end,
                "gc_bin_width": sample.gc_bin_width,
                "gc_assignment_mode": sample.gc_assignment_mode,
                "gc_target_bin_quota": sample.gc_target_bin_quota,
            }
        )
    return metadata


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
    output_subdirs = {
        "positive_tss": Path("positive") / "tss",
        "positive_genic": Path("positive") / "genic",
        "positive_noncoding": Path("positive") / "noncoding",
        "negative_tss": Path("negative") / "tss",
        "negative_genic": Path("negative") / "genic",
        "negative_noncoding": Path("negative") / "noncoding",
    }
    try:
        output_subdir = output_subdirs[sample_type]
    except KeyError as exc:
        raise ValueError(f"Unbekannter Ausgabe-Sample-Typ: {sample_type}") from exc
    group = "positive" if sample_type.startswith("positive_") else "negative"
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
        if isinstance(sample, AnnotationNegativeSample):
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


def write_igv_bed(
    path: Path,
    samples: Sequence[Any],
    sample_ids: Mapping[int, str],
    sample_type: str,
) -> None:
    colors = {
        "positive_tss": "0,180,0",
        "positive_genic": "0,120,210",
        "positive_noncoding": "0,160,160",
        "negative_tss": "220,0,0",
        "negative_genic": "230,140,0",
        "negative_noncoding": "120,80,180",
    }
    color = colors[sample_type]
    with path.open("wt", encoding="utf-8", newline="") as handle:
        handle.write(
            f'track name="{sample_type}_windows" description="{sample_type} windows" '
            'visibility=2 itemRgb="On"\n'
        )
        for sample in samples:
            sample_id = sample_ids[id(sample)]
            score = 1000
            marker_position = (
                sample.summit.position
                if isinstance(sample, (PositiveSample, AnnotationPositiveSample))
                else sample.center
            )
            if isinstance(sample, (PositiveSample, NegativeSample)):
                strand = sample.tss.strand
            elif sample.gene is not None:
                strand = sample.gene.strand
            else:
                strand = "."
            if strand not in ("+", "-", "."):
                strand = "."
            thick_start = marker_position
            thick_end = marker_position + 1
            fields = [
                sample.bundle.fasta_name,
                sample.window_start,
                sample.window_end,
                sample_id,
                score,
                strand,
                thick_start,
                thick_end,
                color,
            ]
            handle.write("\t".join(map(str, fields)) + "\n")


def write_offset_outputs(
    output_dir: Path,
    positive_samples: Sequence[PositiveSample],
    distributions: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    method: str,
    bandwidth: float,
) -> None:
    dist_dir = output_dir / "distributions"
    dist_dir.mkdir(parents=True, exist_ok=True)
    for strand, label in (("+", "plus"), ("-", "minus")):
        samples = [sample for sample in positive_samples if sample.tss.strand == strand]
        offset_rows = [
            {
                "tss_id": sample.tss.name,
                "chromosome": sample.bundle.fasta_name,
                "strand": strand,
                "tss_position_0based": sample.tss.position,
                "summit_position_0based": sample.summit.position,
                "summit_score": sample.summit.score,
                "window_center_position_0based": sample.center,
                "genomic_shift_center_minus_summit": sample.window_shift,
                "genomic_offset_center_minus_tss": sample.genomic_offset,
                "strand_oriented_offset": sample.oriented_offset,
                "genomic_offset_summit_minus_tss": sample.summit_genomic_offset,
                "strand_oriented_offset_summit_minus_tss": sample.summit_oriented_offset,
                "absolute_summit_tss_distance": abs(sample.summit_genomic_offset),
            }
            for sample in samples
        ]
        write_table(dist_dir / f"positive_offsets_{label}.tsv", offset_rows)
        support, counts, probabilities = distributions[strand]
        distribution_rows = [
            {
                "strand": strand,
                "offset": int(offset),
                "observed_count": int(count),
                "estimated_probability": float(probability),
                "method": method,
                "kde_bandwidth": bandwidth if method == "kde" else None,
            }
            for offset, count, probability in zip(support, counts, probabilities)
        ]
        write_table(
            dist_dir / f"offset_distribution_{label}.tsv",
            distribution_rows,
        )

        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(support, counts, width=1.0, alpha=0.35, label="beobachtet")
            if probabilities.sum() > 0:
                scaled = probabilities * max(1, counts.sum())
                ax.plot(support, scaled, linewidth=2, label=f"geschaetzt ({method})")
            ax.set_xlabel("Strandnormalisierter Summit-TSS-Offset [bp]")
            ax.set_ylabel("Anzahl / skalierte Dichte")
            ax.set_title(f"Offset-Verteilung: {label}-Strang")
            ax.legend()
            fig.tight_layout()
            fig.savefig(dist_dir / f"offset_distribution_{label}.png", dpi=160)
            plt.close(fig)
        except ImportError:
            LOGGER.warning(
                "matplotlib ist nicht installiert; Verteilungsplots werden uebersprungen."
            )


def metadata_field_order() -> list[str]:
    return [
        "sample_id",
        "label",
        "group",
        "sample_type",
        "positive_type",
        "negative_type",
        "origin_ids",
        "chromosome",
        "chromosome_input_bed",
        "window_start_0based",
        "window_end_0based_exclusive",
        "window_start_1based",
        "window_end_1based_inclusive",
        "window_length",
        "center_position_0based",
        "center_position_1based",
        "center_index_0based",
        "context_start_0based",
        "context_end_0based_exclusive",
        "context_length",
        "context_per_side",
        "context_padded_left",
        "context_padded_right",
        "core_window_index_start",
        "core_window_index_end_exclusive",
        "tss_id",
        "tss_start_0based",
        "tss_end_0based_exclusive",
        "tss_position_0based",
        "tss_position_1based",
        "tss_strand",
        "tss_cage_score",
        "gene_id",
        "gene_name",
        "gene_feature_type",
        "gene_biotype",
        "gene_chromosome_input",
        "gene_start_0based",
        "gene_end_0based_exclusive",
        "gene_strand",
        "summit_position_0based",
        "summit_position_1based",
        "summit_score",
        "summits_within_tss_range",
        "summit_index_0based",
        "genomic_shift_center_minus_summit",
        "strand_oriented_shift_center_minus_summit",
        "genomic_offset_summit_minus_tss",
        "strand_oriented_offset_summit_minus_tss",
        "absolute_tss_summit_distance",
        "genomic_offset_center_minus_tss",
        "strand_oriented_offset",
        "absolute_tss_center_distance",
        "nearest_tss_id",
        "nearest_tss_position_0based",
        "nearest_tss_strand",
        "nearest_tss_summit_distance",
        "overlapping_gene_count",
        "selection_pval_max",
        "offset_observed_count",
        "offset_estimated_probability",
        "offset_support_mode",
        "gc_target_fraction",
        "gc_fraction_at_selection",
        "gc_absolute_difference",
        "gc_match_tolerance",
        "gc_bin_index",
        "gc_bin_start",
        "gc_bin_end",
        "gc_bin_width",
        "gc_assignment_mode",
        "gc_target_bin_quota",
        "GC_bases",
        "canonical_bases",
        "GC_fraction",
        "sequence_orientation",
        "sequence_shuffle_method",
        "sequence_shuffle_seed",
        "sequence_changed_bases",
        "sequence_changed_fraction",
        "dinucleotide_counts_preserved",
        "sequence_matches_reference_coordinates",
        "FE_min",
        "FE_max",
        "FE_mean",
        "pval_signal_min",
        "pval_signal_max",
        "pval_signal_mean",
        "pval_negative_threshold",
        "flank_pval_max",
        "flank_pval_exceeds_threshold",
        "flank_FE_max",
        "context_N_bases",
        "N_bases",
        "N_fraction",
        "window_data_file",
        "info_file",
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Erzeugt je drei TSS-nahe, genische und nichtkodierende positive "
            "und negative Gruppen. Jede Negativgruppe wird separat gegen die "
            "GC-Verteilung ihrer korrespondierenden Positivgruppe gematcht."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {SCRIPT_VERSION}")
    required = parser.add_argument_group("Pflichtargumente")
    required.add_argument("--fe-bigwig", type=Path, required=True, help="MACS3-FE-BigWig")
    required.add_argument(
        "--pval-bigwig",
        type=Path,
        required=True,
        help="MACS3 p-Wert-BigWig mit -log10(p)",
    )
    required.add_argument(
        "--summit-bigwig",
        type=Path,
        required=True,
        help="BigWig mit 1-bp-Peak-Summits und Summit-Scores",
    )
    required.add_argument("--tss-plus", type=Path, required=True, help="TSS-BED fuer +")
    required.add_argument("--tss-minus", type=Path, required=True, help="TSS-BED fuer -")
    required.add_argument("--fasta", type=Path, required=True, help="TAIR10 FASTA")
    required.add_argument("--fai", type=Path, required=True, help="TAIR10 FASTA-Index (.fai)")
    required.add_argument("--output-dir", type=Path, required=True, help="Ausgabeordner")

    parser.add_argument(
        "--gene-annotation",
        type=Path,
        default=None,
        help=(
            "GFF3/GTF-Genannotation; ohne Angabe wird genau eine *.gff3[.gz], "
            "*.gff[.gz] oder *.gtf[.gz] neben dem Skript automatisch verwendet"
        ),
    )
    parser.add_argument(
        "--gene-feature-types",
        default="gene,pseudogene,transposable_element_gene",
        help="Kommagetrennte GFF3/GTF-Feature-Typen, die als Gene gelten",
    )

    parser.add_argument("--window-size", type=int, default=1024)
    parser.add_argument(
        "--context-per-side",
        type=int,
        default=196,
        help=(
            "Zusaetzliche Basen links und rechts des Fensters in jeder Fensterdatei "
            "(Reserve fuer den zufaelligen Zuschnitt im Training). Auswahl, "
            "GC-Matching, p-Wert-Filter und Ueberlappung gelten nur fuer das "
            "innere Fenster; Umfeld ausserhalb des Chromosoms wird mit N/0 aufgefuellt."
        ),
    )
    parser.add_argument("--max-summit-distance", type=int, default=300)
    parser.add_argument(
        "--min-nontss-summit-distance",
        type=int,
        default=600,
        help=(
            "Minimaler Abstand eines positiven genischen oder nichtgenischen "
            "Summits zu jeder TSS; die Grenze selbst ist eingeschlossen"
        ),
    )
    parser.add_argument(
        "--pval-threshold",
        type=float,
        default=1.3,
        help="Negative Fenster duerfen nirgendwo einen groesseren -log10(p)-Wert haben",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--offset-distribution",
        choices=("kde", "empirical"),
        default="kde",
        help="Methode zur Schaetzung der strandweisen Offset-Verteilung",
    )
    parser.add_argument(
        "--kde-bandwidth",
        type=float,
        default=15.0,
        help="Bandbreite der diskreten Gaussian-KDE in bp",
    )
    parser.add_argument(
        "--summit-min-value",
        type=float,
        default=0.0,
        help="Nur Summit-Werte groesser als dieser Wert werden verwendet",
    )
    parser.add_argument(
        "--wide-summit-policy",
        choices=("error", "midpoint"),
        default="error",
        help="Behandlung von Summit-BigWig-Intervallen mit Laenge >1",
    )
    parser.add_argument(
        "--allow-positive-overlap",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gc-bin-width",
        type=float,
        default=0.005,
        help=(
            "GC-Aufloesung fuer Verteilungsauswahl, Audit und Diagramme"
        ),
    )
    parser.add_argument(
        "--gc-ks-tolerance", type=float, default=0.06,
        help="Maximaler kumulativer GC-Abstand an den Grenzen der GC-Bereiche",
    )
    parser.add_argument(
        "--gc-quantile-tolerance", type=float, default=0.015,
        help="Maximaler mittlerer absoluter Abstand der GC-Quantile",
    )
    parser.add_argument(
        "--save-candidate-pools",
        action="store_true",
        help=(
            "Speichert die vollstaendigen hart gefilterten Kandidatenpools "
            "zusaetzlich unter candidate_pools/"
        ),
    )
    parser.add_argument(
        "--genic-window-placement",
        choices=("center-within-gene", "fully-contained"),
        default="center-within-gene",
        help=(
            "Genische Platzierung: Fensterzentrum im Gen oder gesamtes Fenster "
            "im Gen; center-within-gene erfasst auch Gene kuerzer als das Fenster"
        ),
    )
    parser.add_argument(
        "--genic-count",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--genic-seed",
        type=int,
        default=43,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--noncoding-count",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--noncoding-per-positive",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--candidate-scan-chunk-size",
        "--noncoding-scan-chunk-size",
        dest="candidate_scan_chunk_size",
        type=int,
        default=1_000_000,
        help=(
            "Maximale Zahl zusammenhaengender Fensterstarts pro Scan-Chunk; "
            "der alte Name --noncoding-scan-chunk-size bleibt als Alias erhalten"
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Aus Kompatibilitaetsgruenden akzeptiert; dichter Scan laeuft derzeit sequenziell",
    )
    parser.add_argument(
        "--noncoding-seed",
        type=int,
        default=44,
        help=argparse.SUPPRESS,
    )
    # Alte Optionen werden fuer nachvollziehbare Fehlermeldungen weiterhin
    # akzeptiert, beeinflussen das neue GC-Matching aber nicht.
    parser.add_argument("--gc-tolerance", type=float, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--genic-candidate-step", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--noncoding-candidate-step", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--noncoding-pool-multiplier", type=int, default=None, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--uncompressed-window-files",
        action="store_true",
        help="Schreibt .tsv statt .tsv.gz fuer jedes Fenster",
    )
    parser.add_argument(
        "--dinucleotide-shuffle",
        action="store_true",
        help=(
            "Randomisiert jede ausgegebene DNA-Sequenz unter exakter "
            "Erhaltung ihrer Dinukleotid-Zusammensetzung"
        ),
    )
    parser.add_argument(
        "--dinucleotide-shuffle-seed",
        type=int,
        default=42,
        help="Seed fuer die reproduzierbare Dinukleotid-Shuffle-Kontrolle",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    input_paths = [
        args.fe_bigwig,
        args.pval_bigwig,
        args.summit_bigwig,
        args.tss_plus,
        args.tss_minus,
        args.fasta,
        args.fai,
        args.gene_annotation,
    ]
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Folgende Eingabedateien fehlen:\n  " + "\n  ".join(missing))
    if args.window_size <= 0:
        raise ValueError("--window-size muss groesser als 0 sein.")
    if args.context_per_side < 0:
        raise ValueError("--context-per-side darf nicht negativ sein.")
    if args.context_per_side > args.window_size // 2:
        raise ValueError(
            "--context-per-side darf hoechstens die halbe Fensterlaenge betragen; "
            "groessere Verschiebungen im Training wuerden den Summit aus dem Ausschnitt schieben."
        )
    if args.seed < 0 or args.dinucleotide_shuffle_seed < 0:
        raise ValueError("Zufalls-Seeds duerfen nicht negativ sein.")
    if args.max_summit_distance < 0:
        raise ValueError("--max-summit-distance darf nicht negativ sein.")
    if args.min_nontss_summit_distance < 0:
        raise ValueError("--min-nontss-summit-distance darf nicht negativ sein.")
    if not math.isfinite(args.pval_threshold):
        raise ValueError("--pval-threshold muss endlich sein.")
    if args.offset_distribution == "kde" and args.kde_bandwidth <= 0:
        raise ValueError("--kde-bandwidth muss groesser als 0 sein.")
    if not math.isfinite(args.gc_bin_width) or not 0.0 < args.gc_bin_width <= 1.0:
        raise ValueError("--gc-bin-width muss groesser als 0 und hoechstens 1 sein.")
    if not math.isfinite(args.gc_ks_tolerance) or not 0.0 < args.gc_ks_tolerance < 1.0:
        raise ValueError("--gc-ks-tolerance muss zwischen 0 und 1 liegen.")
    if not math.isfinite(args.gc_quantile_tolerance) or not 0.0 < args.gc_quantile_tolerance < 1.0:
        raise ValueError("--gc-quantile-tolerance muss zwischen 0 und 1 liegen.")
    if args.allow_positive_overlap:
        raise ValueError("Negative Fenster duerfen positive Fenster nicht ueberlappen; --allow-positive-overlap ist nicht mehr zulaessig.")
    if args.candidate_scan_chunk_size <= 0:
        raise ValueError("--candidate-scan-chunk-size muss groesser als 0 sein.")
    if args.workers <= 0:
        raise ValueError("--workers muss groesser als 0 sein.")
    if args.genic_seed < 0 or args.noncoding_seed < 0:
        raise ValueError("Zufalls-Seeds duerfen nicht negativ sein.")
    deprecated_target_counts = {
        "--genic-count": args.genic_count,
        "--noncoding-count": args.noncoding_count,
        "--noncoding-per-positive": args.noncoding_per_positive,
    }
    supplied_target_counts = [
        name for name, value in deprecated_target_counts.items() if value is not None
    ]
    if supplied_target_counts:
        LOGGER.warning("Veraltete Zielzahl-Optionen %s werden ignoriert; die Anzahl ergibt sich aus GC-Balance und Ueberlappung.", ", ".join(supplied_target_counts))
    deprecated_pool_options = {
        "--gc-tolerance": args.gc_tolerance,
        "--genic-candidate-step": args.genic_candidate_step,
        "--noncoding-candidate-step": args.noncoding_candidate_step,
        "--noncoding-pool-multiplier": args.noncoding_pool_multiplier,
    }
    supplied_deprecated = [
        name for name, value in deprecated_pool_options.items() if value is not None
    ]
    if supplied_deprecated:
        LOGGER.warning(
            "Die veralteten Optionen %s werden ignoriert. Das neue Verfahren "
            "prueft alle zulaessigen Fensterstarts und verwendet "
            "eigenstaendiges verteilungsbasiertes GC-Matching.",
            ", ".join(supplied_deprecated),
        )
    if args.workers != 1:
        LOGGER.warning("--workers wird beim dichten Kandidatenscan derzeit ignoriert.")


def main(argv: Sequence[str] | None = None) -> int:
    command_argv = sys.argv if argv is None else [sys.argv[0], *argv]

    saved_command = shlex.join(
        [
            sys.executable,
            str(Path(command_argv[0]).resolve()),
            *command_argv[1:],
        ]
    )
    working_directory = str(Path.cwd())

    args = parse_args(argv)
    args.gene_annotation = resolve_gene_annotation_path(
        args.gene_annotation,
        Path(command_argv[0]),
    )
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if not callable(getattr(CandidateStore, "gc_capacity", None)):
        raise RuntimeError(
            "gc_matching.py ist zu alt: Bitte data_m_3_3.py und gc_matching.py "
            "gemeinsam aktualisieren."
        )
    validate_args(args)
    class_definitions = positive_class_definitions(
        args.max_summit_distance, args.min_nontss_summit_distance
    )
    for sample_type, definition in class_definitions.items():
        LOGGER.info("%s: %s", sample_type, definition)
    pyBigWig, pysam = open_dependencies()
    LOGGER.info(
        "GC-Matching: eigene, verteilungsbasierte Auswahl ohne R (Matcher %s)",
        MATCHER_VERSION,
    )
    LOGGER.info("Lese TSS-Dateien")
    tss_plus = read_tss_bed(args.tss_plus, "+")
    tss_minus = read_tss_bed(args.tss_minus, "-")
    tss_records = tss_plus + tss_minus
    LOGGER.info("TSS: %d plus, %d minus", len(tss_plus), len(tss_minus))
    gene_feature_types = [
        value.strip() for value in args.gene_feature_types.split(",") if value.strip()
    ]
    LOGGER.info("Lese Genannotation: %s", args.gene_annotation)
    genes, gene_annotation_summary = read_gene_annotation(
        args.gene_annotation,
        gene_feature_types,
    )
    LOGGER.info("%d Genintervalle geladen", len(genes))

    match_temp_dir = tempfile.TemporaryDirectory(prefix="tf_binding_gc_match_")
    match_work_dir = Path(match_temp_dir.name)
    fe_bw = pval_bw = summit_bw = fasta = None
    try:
        fe_bw = open_bigwig(pyBigWig, args.fe_bigwig)
        pval_bw = open_bigwig(pyBigWig, args.pval_bigwig)
        summit_bw = open_bigwig(pyBigWig, args.summit_bigwig)
        fasta = pysam.FastaFile(str(args.fasta), filepath_index=str(args.fai))

        bundles = build_chrom_bundles(
            [tss.chrom_key for tss in tss_records],
            fasta,
            fe_bw,
            pval_bw,
            summit_bw,
        )
        LOGGER.info("Chromosomenzuordnung erfolgreich fuer: %s", ", ".join(sorted(bundles)))
        genes_per_data_chrom = Counter(
            gene.chrom_key for gene in genes if gene.chrom_key in bundles
        )
        missing_annotation_chroms = [
            bundle.fasta_name
            for chrom_key, bundle in bundles.items()
            if genes_per_data_chrom[chrom_key] == 0
        ]
        if missing_annotation_chroms:
            raise ValueError(
                "Die Genannotation enthaelt fuer folgende Datenchromosomen keine "
                "Gene: "
                + ", ".join(missing_annotation_chroms)
                + ". Assembly und Chromosomennamen pruefen."
            )
        gene_annotation_summary.update(
            {
                "genes_on_data_chromosomes": int(sum(genes_per_data_chrom.values())),
                "genes_ignored_on_other_chromosomes": int(
                    len(genes) - sum(genes_per_data_chrom.values())
                ),
                "genes_per_data_chromosome": dict(genes_per_data_chrom),
            }
        )

        summits_by_chrom, wide_summit_count = load_summits(
            summit_bw=summit_bw,
            bundles=bundles,
            min_value=args.summit_min_value,
            wide_interval_policy=args.wide_summit_policy,
        )
        summit_count = sum(len(items) for items in summits_by_chrom.values())
        LOGGER.info("%d Summit-Positionen geladen", summit_count)

        positives, positive_audit, positive_stats = select_positive_samples(
            tss_records=tss_records,
            summits_by_chrom=summits_by_chrom,
            bundles=bundles,
            max_distance=args.max_summit_distance,
            window_size=args.window_size,
        )
        LOGGER.info("%d positive TSS-Beispiele ausgewaehlt", len(positives))
        LOGGER.info(
            "Positive Fenster sind auf den Summit zentriert; Umfeld je Seite: %d bp",
            args.context_per_side,
        )
        if not positives:
            raise RuntimeError(
                "Keine positiven Beispiele gefunden. Chromosomennamen, Summit-Werte "
                "und --max-summit-distance pruefen."
            )

        (
            positive_genic,
            positive_noncoding,
            positive_summit_audit,
            positive_group_stats,
        ) = select_annotation_positive_samples(
            tss_records=tss_records,
            genes=genes,
            summits_by_chrom=summits_by_chrom,
            positive_tss_samples=positives,
            bundles=bundles,
            min_tss_distance=args.min_nontss_summit_distance,
            window_size=args.window_size,
        )
        LOGGER.info(
            "Zusaetzliche positive Summit-Gruppen: %d genisch, %d nichtgenisch",
            len(positive_genic),
            len(positive_noncoding),
        )
        if not positive_genic:
            raise RuntimeError(
                "Keine positiven genischen Summits gefunden; gruppenweises "
                "GC-Matching fuer genische Negative ist nicht moeglich."
            )
        if not positive_noncoding:
            raise RuntimeError(
                "Keine positiven nichtgenischen Summits gefunden; gruppenweises "
                "GC-Matching fuer nichtkodierende Negative ist nicht moeglich."
            )
        all_positive_samples: list[Any] = [
            *positives,
            *positive_genic,
            *positive_noncoding,
        ]

        # Die Negativverteilung basiert immer auf dem echten Summit-TSS-Offset,
        # nicht auf dem optional verschobenen positiven Fensterzentrum.
        offsets_plus = [
            sample.summit_oriented_offset
            for sample in positives
            if sample.tss.strand == "+"
        ]
        offsets_minus = [
            sample.summit_oriented_offset
            for sample in positives
            if sample.tss.strand == "-"
        ]
        distributions = {
            "+": estimate_offset_distribution(
                offsets_plus,
                args.max_summit_distance,
                args.offset_distribution,
                args.kde_bandwidth,
            ),
            "-": estimate_offset_distribution(
                offsets_minus,
                args.max_summit_distance,
                args.offset_distribution,
                args.kde_bandwidth,
            ),
        }

        positive_gc_fractions = {
            "tss": sample_gc_fractions(positives, fasta),
            "genic": sample_gc_fractions(positive_genic, fasta),
            "noncoding": sample_gc_fractions(positive_noncoding, fasta),
        }
        matcher = GCMatcher(args.gc_bin_width, args.gc_ks_tolerance,
                            args.window_size, args.gc_quantile_tolerance)
        gc_edges = np.arange(args.gc_bin_width, 1.0, args.gc_bin_width)
        ensure_output_dir(args.output_dir, args.overwrite)

        tss_pool = CandidateStore(match_work_dir / "tss.sqlite", "tss", gc_edges)
        negative_audit, negative_summary = build_dense_tss_pool(
            tss_records, all_positive_samples, distributions, bundles, fasta,
            pval_bw, args.window_size, args.max_summit_distance,
            args.pval_threshold, True, tss_pool,
        )
        tss_gc_capacity = write_gc_capacity_plot(
            args.output_dir, "tss", positive_gc_fractions["tss"], tss_pool,
            args.gc_bin_width, args.window_size, args.gc_ks_tolerance,
        )
        tss_matches, tss_match_summary, tss_capacities = matcher.match(
            positive_gc_fractions["tss"], tss_pool, all_positive_samples,
        )
        negatives: list[NegativeSample] = []
        tss_match_rows: list[dict[str, Any]] = []
        for candidate_id, candidate, target_gc in tss_matches:
            tss = candidate.tss
            if tss is None:
                raise AssertionError("TSS-Kandidat ohne Herkunft.")
            support, counts, probabilities = distributions[tss.strand]
            center = candidate.window_start + args.window_size // 2
            genomic_offset = center - tss.position
            oriented_offset = genomic_offset if tss.strand == "+" else -genomic_offset
            offset_index = oriented_offset + args.max_summit_distance
            if not 0 <= offset_index < support.size:
                raise AssertionError("TSS-Offset ausserhalb des Supports.")
            negatives.append(NegativeSample(
                tss=tss, bundle=bundles[candidate.chrom_key], center=center,
                window_start=candidate.window_start,
                window_end=candidate.window_start + args.window_size,
                genomic_offset=genomic_offset, oriented_offset=oriented_offset,
                selection_pval_max=candidate.selection_pval_max,
                offset_observed_count=int(counts[offset_index]),
                offset_estimated_probability=float(probabilities[offset_index]),
                offset_support_mode="gc_matched_from_all_valid_offsets",
                origin_ids=tss_pool.source_names(candidate_id),
            ))
            tss_match_rows.append({
                "candidate_id": candidate_id, "chromosome": bundles[candidate.chrom_key].fasta_name,
                "window_start_0based": candidate.window_start,
                "window_end_0based_exclusive": candidate.window_start + args.window_size,
                "tss_id": tss.name, "origin_ids": tss_pool.source_names(candidate_id),
                "strand_oriented_offset": oriented_offset,
                "offset_estimated_probability": float(probabilities[offset_index]),
                "target_gc_fraction": target_gc,
                "matched_gc_fraction": candidate.gc_fraction,
                "absolute_gc_difference": abs(candidate.gc_fraction - target_gc),
                "selection_pval_max": candidate.selection_pval_max,
            })
        negatives.sort(key=lambda sample: (
            sample.bundle.fasta_name, sample.window_start, sample.tss.name
        ))
        negative_summary.update(tss_match_summary)
        negative_summary["candidate_pool_saved"] = args.save_candidate_pools
        negative_summary["selected_offset_counts_by_strand"] = {
            strand: dict(Counter(
                sample.oriented_offset for sample in negatives
                if sample.tss.strand == strand
            )) for strand in ("+", "-")
        }

        genic_pool = CandidateStore(match_work_dir / "genic.sqlite", "genic", gc_edges)
        genic_audit, genic_summary = build_dense_genic_pool(
            genes, [*all_positive_samples, *negatives], bundles,
            fasta, pval_bw, args.window_size, args.pval_threshold,
            args.genic_window_placement, args.candidate_scan_chunk_size,
            genic_pool,
        )
        genic_gc_capacity = write_gc_capacity_plot(
            args.output_dir, "genic", positive_gc_fractions["genic"], genic_pool,
            args.gc_bin_width, args.window_size, args.gc_ks_tolerance,
        )
        genic_matches, genic_match_summary, genic_capacities = matcher.match(
            positive_gc_fractions["genic"], genic_pool,
            [*all_positive_samples, *negatives],
        )
        genic_negatives, genic_match_rows, genic_gc_histogram = materialize_matched_samples(
            genic_matches, "genic", bundles, positive_gc_fractions["genic"],
            args.window_size, args.gc_bin_width, "distribution",
            origin_lookup=genic_pool.source_names,
        )
        for row in genic_match_rows:
            row["origin_ids"] = genic_pool.source_names(row["candidate_id"])
        genic_summary.update(genic_match_summary)
        genic_summary["gc_histogram"] = genic_gc_histogram
        genic_summary["candidate_pool_saved"] = args.save_candidate_pools

        noncoding_pool = CandidateStore(
            match_work_dir / "noncoding.sqlite", "noncoding", gc_edges,
        )
        noncoding_summary = build_dense_noncoding_pool(
            genes, [*all_positive_samples, *negatives, *genic_negatives],
            bundles, fasta, pval_bw, args.window_size, args.pval_threshold,
            args.candidate_scan_chunk_size, noncoding_pool,
        )
        noncoding_gc_capacity = write_gc_capacity_plot(
            args.output_dir, "noncoding", positive_gc_fractions["noncoding"],
            noncoding_pool, args.gc_bin_width, args.window_size, args.gc_ks_tolerance,
        )
        noncoding_matches, noncoding_match_summary, noncoding_capacities = matcher.match(
            positive_gc_fractions["noncoding"], noncoding_pool,
            [*all_positive_samples, *negatives, *genic_negatives],
        )
        noncoding_negatives, noncoding_match_rows, noncoding_gc_histogram = materialize_matched_samples(
            noncoding_matches, "noncoding", bundles,
            positive_gc_fractions["noncoding"], args.window_size,
            args.gc_bin_width, "distribution",
            origin_lookup=noncoding_pool.source_names,
        )
        for row in noncoding_match_rows:
            row["origin_ids"] = noncoding_pool.source_names(row["candidate_id"])
        noncoding_summary.update(noncoding_match_summary)
        noncoding_summary["gc_histogram"] = noncoding_gc_histogram
        noncoding_summary["candidate_pool_saved"] = args.save_candidate_pools

        write_table(
            args.output_dir / "sample_classes.tsv",
            [
                {"sample_type": name, "definition": definition}
                for name, definition in class_definitions.items()
            ],
            ["sample_type", "definition"],
        )

        command_file = args.output_dir / "run_command.txt"
        with command_file.open("wt", encoding="utf-8") as handle:
            handle.write(f"# Arbeitsverzeichnis: {working_directory}\n")
            handle.write(f"# Aufgeloeste Genannotation: {args.gene_annotation}\n")
            handle.write(saved_command)
            handle.write("\n")

        (args.output_dir / "positive").mkdir(parents=True, exist_ok=True)
        (args.output_dir / "negative" / "tss").mkdir(parents=True, exist_ok=True)
        (args.output_dir / "negative" / "genic").mkdir(parents=True, exist_ok=True)
        (args.output_dir / "negative" / "noncoding").mkdir(parents=True, exist_ok=True)
        (args.output_dir / "igv").mkdir(parents=True, exist_ok=True)
        (args.output_dir / "audit").mkdir(parents=True, exist_ok=True)
        LOGGER.info("Erzeuge GC-Ergebnisdiagramme")
        plot_results = {}
        for group, store, positive_gc, matches, capacities, gc_capacity, group_summary in (
            ("tss", tss_pool, positive_gc_fractions["tss"], tss_matches,
             tss_capacities, tss_gc_capacity, negative_summary),
            ("genic", genic_pool, positive_gc_fractions["genic"], genic_matches,
             genic_capacities, genic_gc_capacity, genic_summary),
            ("noncoding", noncoding_pool, positive_gc_fractions["noncoding"],
             noncoding_matches, noncoding_capacities, noncoding_gc_capacity,
             noncoding_summary),
        ):
            rows, plot_paths = write_gc_coverage_plots(
                args.output_dir, group, positive_gc, matches, store, capacities,
                args.gc_bin_width, group_summary["gc_binned_cdf_distance"],
                gc_capacity,
            )
            group_summary["gc_matching_balance_by_plot_bin"] = rows
            group_summary["gc_capacity"] = {
                "candidate_distinct_starts": gc_capacity["distinct_starts"],
                "candidate_nonoverlapping_total": gc_capacity["nonoverlapping_total"],
                "positives_reassigned_to_nearest_bin": (
                    gc_capacity["positives_reassigned_to_nearest_bin"]
                ),
                "bound_strict": (
                    None if math.isinf(gc_capacity["bound_strict"])
                    else float(gc_capacity["bound_strict"])
                ),
                "bottleneck_strict": gc_capacity["bottleneck_strict"],
                "tolerance": gc_capacity["tolerance"],
                "bound_at_tolerance": (
                    None if math.isinf(gc_capacity["bound_at_tolerance"])
                    else float(gc_capacity["bound_at_tolerance"])
                ),
                "bottleneck_at_tolerance": gc_capacity["bottleneck_at_tolerance"],
                "selected_count": len(matches),
                "selected_share_of_bound_at_tolerance": (
                    None if not math.isfinite(gc_capacity["bound_at_tolerance"])
                    or gc_capacity["bound_at_tolerance"] <= 0
                    else len(matches) / float(gc_capacity["bound_at_tolerance"])
                ),
            }
            group_summary["gc_comparison_outputs"] = plot_paths
            plot_results[group] = rows
        tss_match_audit = plot_results["tss"]
        genic_match_audit = plot_results["genic"]
        noncoding_audit = plot_results["noncoding"]
        if args.save_candidate_pools:
            candidate_pool_dir = args.output_dir / "candidate_pools"
            candidate_pool_dir.mkdir(parents=True, exist_ok=True)
            for group, store in (("tss", tss_pool), ("genic", genic_pool),
                                 ("noncoding", noncoding_pool)):
                write_candidate_store(
                    candidate_pool_dir / f"negative_{group}_candidates.tsv",
                    store, bundles, args.window_size,
                )

        all_samples: list[Any] = [
            *all_positive_samples,
            *negatives,
            *genic_negatives,
            *noncoding_negatives,
        ]
        sample_ids = unique_sample_ids(all_samples)
        compress = not args.uncompressed_window_files

        positive_tss_metadata = write_samples(
            output_dir=args.output_dir,
            sample_type="positive_tss",
            samples=positives,
            sample_ids=sample_ids,
            fasta=fasta,
            fe_bw=fe_bw,
            pval_bw=pval_bw,
            window_size=args.window_size,
            pval_threshold=args.pval_threshold,
            compress=compress,
            dinucleotide_shuffle=args.dinucleotide_shuffle,
            dinucleotide_shuffle_seed=args.dinucleotide_shuffle_seed,
            class_definitions=class_definitions,
            context_per_side=args.context_per_side,
        )
        positive_genic_metadata = write_samples(
            output_dir=args.output_dir,
            sample_type="positive_genic",
            samples=positive_genic,
            sample_ids=sample_ids,
            fasta=fasta,
            fe_bw=fe_bw,
            pval_bw=pval_bw,
            window_size=args.window_size,
            pval_threshold=args.pval_threshold,
            compress=compress,
            dinucleotide_shuffle=args.dinucleotide_shuffle,
            dinucleotide_shuffle_seed=args.dinucleotide_shuffle_seed,
            class_definitions=class_definitions,
            context_per_side=args.context_per_side,
        )
        positive_noncoding_metadata = write_samples(
            output_dir=args.output_dir,
            sample_type="positive_noncoding",
            samples=positive_noncoding,
            sample_ids=sample_ids,
            fasta=fasta,
            fe_bw=fe_bw,
            pval_bw=pval_bw,
            window_size=args.window_size,
            pval_threshold=args.pval_threshold,
            compress=compress,
            dinucleotide_shuffle=args.dinucleotide_shuffle,
            dinucleotide_shuffle_seed=args.dinucleotide_shuffle_seed,
            class_definitions=class_definitions,
            context_per_side=args.context_per_side,
        )
        negative_tss_metadata = write_samples(
            output_dir=args.output_dir,
            sample_type="negative_tss",
            samples=negatives,
            sample_ids=sample_ids,
            fasta=fasta,
            fe_bw=fe_bw,
            pval_bw=pval_bw,
            window_size=args.window_size,
            pval_threshold=args.pval_threshold,
            compress=compress,
            dinucleotide_shuffle=args.dinucleotide_shuffle,
            dinucleotide_shuffle_seed=args.dinucleotide_shuffle_seed,
            class_definitions=class_definitions,
            context_per_side=args.context_per_side,
        )
        negative_genic_metadata = write_samples(
            output_dir=args.output_dir,
            sample_type="negative_genic",
            samples=genic_negatives,
            sample_ids=sample_ids,
            fasta=fasta,
            fe_bw=fe_bw,
            pval_bw=pval_bw,
            window_size=args.window_size,
            pval_threshold=args.pval_threshold,
            compress=compress,
            dinucleotide_shuffle=args.dinucleotide_shuffle,
            dinucleotide_shuffle_seed=args.dinucleotide_shuffle_seed,
            class_definitions=class_definitions,
            context_per_side=args.context_per_side,
        )
        negative_noncoding_metadata = write_samples(
            output_dir=args.output_dir,
            sample_type="negative_noncoding",
            samples=noncoding_negatives,
            sample_ids=sample_ids,
            fasta=fasta,
            fe_bw=fe_bw,
            pval_bw=pval_bw,
            window_size=args.window_size,
            pval_threshold=args.pval_threshold,
            compress=compress,
            dinucleotide_shuffle=args.dinucleotide_shuffle,
            dinucleotide_shuffle_seed=args.dinucleotide_shuffle_seed,
            class_definitions=class_definitions,
            context_per_side=args.context_per_side,
        )

        fields = metadata_field_order()
        positive_all_metadata = [
            *positive_tss_metadata,
            *positive_genic_metadata,
            *positive_noncoding_metadata,
        ]
        write_table(
            args.output_dir / "metadata_positive_tss.tsv",
            positive_tss_metadata,
            fields,
        )
        write_table(
            args.output_dir / "metadata_positive_genic.tsv",
            positive_genic_metadata,
            fields,
        )
        write_table(
            args.output_dir / "metadata_positive_noncoding.tsv",
            positive_noncoding_metadata,
            fields,
        )
        write_table(
            args.output_dir / "metadata_positive_all.tsv",
            positive_all_metadata,
            fields,
        )
        # Kompatibilitaetsname fuer bestehende Trainingsaufrufe.
        write_table(
            args.output_dir / "metadata_positive.tsv",
            positive_all_metadata,
            fields,
        )
        write_table(
            args.output_dir / "metadata_negative_tss.tsv",
            negative_tss_metadata,
            fields,
        )
        write_table(
            args.output_dir / "metadata_negative_genic.tsv",
            negative_genic_metadata,
            fields,
        )
        write_table(
            args.output_dir / "metadata_negative_noncoding.tsv",
            negative_noncoding_metadata,
            fields,
        )
        write_table(
            args.output_dir / "metadata_negative_all.tsv",
            [
                *negative_tss_metadata,
                *negative_genic_metadata,
                *negative_noncoding_metadata,
            ],
            fields,
        )
        write_table(
            args.output_dir / "audit" / "positive_selection.tsv",
            positive_audit,
            [
                "tss_id",
                "chromosome",
                "tss_position_0based",
                "strand",
                "status",
                "summit_position_0based",
                "summit_score",
                "candidate_count",
                "reason",
            ],
        )
        write_table(
            args.output_dir / "audit" / "positive_summit_classification.tsv",
            positive_summit_audit,
            [
                "chromosome",
                "summit_position_0based",
                "summit_score",
                "nearest_tss_id",
                "nearest_tss_position_0based",
                "nearest_tss_distance",
                "overlapping_gene_count",
                "gene_id",
                "positive_type",
                "window_start_0based",
                "window_end_0based_exclusive",
                "window_shift",
                "status",
                "reason",
            ],
        )
        write_table(
            args.output_dir / "audit" / "negative_tss_rejections.tsv",
            negative_audit,
            [
                "tss_id",
                "chromosome",
                "strand",
                "tested_offsets",
                "valid_offsets",
                "status",
                "reason",
            ],
        )
        write_table(
            args.output_dir / "audit" / "negative_genic_rejections.tsv",
            genic_audit,
            [
                "gene_id",
                "chromosome",
                "gene_start_0based",
                "gene_end_0based_exclusive",
                "tested_windows",
                "pval_valid_windows",
                "hard_valid_windows",
                "gc_measurable_windows",
                "pool_candidate_windows",
                "available_gc_bins",
                "status",
                "reason",
            ],
        )
        write_table(
            args.output_dir / "audit" / "negative_tss_gc_matching_balance.tsv",
            tss_match_audit,
            [
                "bin_index",
                "bin_start",
                "bin_end",
                "bin_center",
                "positive_count",
                "candidate_pool_count",
                "selected_count",
                "nonoverlapping_capacity_upper_bound",
                "cumulative_capacity_gc_at_most_bin",
                "cumulative_capacity_gc_at_least_bin",
                "supported_negatives_from_low_tail",
                "supported_negatives_from_high_tail",
                "supported_negatives_from_low_tail_strict",
                "supported_negatives_from_high_tail_strict",
                "positive_fraction",
                "candidate_pool_fraction",
                "selected_fraction",
                "candidate_minus_positive_fraction",
                "selected_minus_positive_fraction",
            ],
        )
        write_table(
            args.output_dir / "audit" / "negative_genic_gc_matching_balance.tsv",
            genic_match_audit,
            [
                "bin_index",
                "bin_start",
                "bin_end",
                "bin_center",
                "positive_count",
                "candidate_pool_count",
                "selected_count",
                "nonoverlapping_capacity_upper_bound",
                "cumulative_capacity_gc_at_most_bin",
                "cumulative_capacity_gc_at_least_bin",
                "supported_negatives_from_low_tail",
                "supported_negatives_from_high_tail",
                "supported_negatives_from_low_tail_strict",
                "supported_negatives_from_high_tail_strict",
                "positive_fraction",
                "candidate_pool_fraction",
                "selected_fraction",
                "candidate_minus_positive_fraction",
                "selected_minus_positive_fraction",
            ],
        )
        write_table(
            args.output_dir / "audit" / "negative_noncoding_gc_matching_balance.tsv",
            noncoding_audit,
            [
                "bin_index",
                "bin_start",
                "bin_end",
                "bin_center",
                "positive_count",
                "candidate_pool_count",
                "selected_count",
                "nonoverlapping_capacity_upper_bound",
                "cumulative_capacity_gc_at_most_bin",
                "cumulative_capacity_gc_at_least_bin",
                "supported_negatives_from_low_tail",
                "supported_negatives_from_high_tail",
                "supported_negatives_from_low_tail_strict",
                "supported_negatives_from_high_tail_strict",
                "positive_fraction",
                "candidate_pool_fraction",
                "selected_fraction",
                "candidate_minus_positive_fraction",
                "selected_minus_positive_fraction",
            ],
        )
        matched_selection_fields = [
            "candidate_id",
            "chromosome",
            "window_start_0based",
            "window_end_0based_exclusive",
            "gene_id",
            "origin_ids",
            "target_gc_fraction",
            "matched_gc_fraction",
            "absolute_gc_difference",
            "selection_pval_max",
        ]
        write_table(
            args.output_dir / "audit" / "negative_tss_matchranges_selection.tsv",
            tss_match_rows,
            [
                "candidate_id",
                "chromosome",
                "window_start_0based",
                "window_end_0based_exclusive",
                "tss_id",
                "origin_ids",
                "strand_oriented_offset",
                "offset_estimated_probability",
                "target_gc_fraction",
                "matched_gc_fraction",
                "absolute_gc_difference",
                "selection_pval_max",
            ],
        )
        write_table(
            args.output_dir / "audit" / "negative_genic_matchranges_selection.tsv",
            genic_match_rows,
            matched_selection_fields,
        )
        write_table(
            args.output_dir / "audit" / "negative_noncoding_matchranges_selection.tsv",
            noncoding_match_rows,
            matched_selection_fields,
        )
        write_igv_bed(
            args.output_dir / "igv" / "positive_tss_windows.bed",
            positives,
            sample_ids,
            "positive_tss",
        )
        write_igv_bed(
            args.output_dir / "igv" / "positive_genic_windows.bed",
            positive_genic,
            sample_ids,
            "positive_genic",
        )
        write_igv_bed(
            args.output_dir / "igv" / "positive_noncoding_windows.bed",
            positive_noncoding,
            sample_ids,
            "positive_noncoding",
        )
        write_igv_bed(
            args.output_dir / "igv" / "negative_tss_windows.bed",
            negatives,
            sample_ids,
            "negative_tss",
        )
        write_igv_bed(
            args.output_dir / "igv" / "negative_genic_windows.bed",
            genic_negatives,
            sample_ids,
            "negative_genic",
        )
        write_igv_bed(
            args.output_dir / "igv" / "negative_noncoding_windows.bed",
            noncoding_negatives,
            sample_ids,
            "negative_noncoding",
        )
        write_offset_outputs(
            output_dir=args.output_dir,
            positive_samples=positives,
            distributions=distributions,
            method=args.offset_distribution,
            bandwidth=args.kde_bandwidth,
        )
        write_table(
            args.output_dir / "distributions" / "genic_gc_histogram.tsv",
            genic_summary["gc_histogram"],
            [
                "bin_index",
                "bin_start",
                "bin_end",
                "positive_count",
                "realized_count",
                "realized_genic_count",
                "positive_fraction",
                "realized_fraction",
                "fraction_difference_realized_minus_positive",
            ],
        )
        write_table(
            args.output_dir / "distributions" / "noncoding_gc_histogram.tsv",
            noncoding_summary["gc_histogram"],
            [
                "bin_index",
                "bin_start",
                "bin_end",
                "positive_count",
                "realized_count",
                "realized_noncoding_count",
                "positive_fraction",
                "realized_fraction",
                "fraction_difference_realized_minus_positive",
            ],
        )

        context_flank_pval_exceedances = {
            name: int(sum(1 for row in rows if row.get("flank_pval_exceeds_threshold")))
            for name, rows in (
                ("negative_tss", negative_tss_metadata),
                ("negative_genic", negative_genic_metadata),
                ("negative_noncoding", negative_noncoding_metadata),
            )
        }
        for name, count in context_flank_pval_exceedances.items():
            LOGGER.info(
                "%s: %d Fenster mit p-Wert ueber der Schwelle im Umfeld ausserhalb "
                "des inneren Fensters (nur Diagnose, keine Filterung)",
                name, count,
            )
        chrom_map_rows = [
            {
                "canonical_chromosome": key,
                "fasta": bundle.fasta_name,
                "fe_bigwig": bundle.fe_name,
                "pval_bigwig": bundle.pval_name,
                "summit_bigwig": bundle.summit_name,
                "length": bundle.length,
                "output_folder": safe_filename(bundle.fasta_name),
            }
            for key, bundle in sorted(bundles.items())
        ]
        write_table(args.output_dir / "chromosome_mapping.tsv", chrom_map_rows)

        summary = {
            "script_version": SCRIPT_VERSION,
            "positive_class_definitions": class_definitions,
            "inputs": {
                "fe_bigwig": args.fe_bigwig,
                "pval_bigwig": args.pval_bigwig,
                "summit_bigwig": args.summit_bigwig,
                "tss_plus": args.tss_plus,
                "tss_minus": args.tss_minus,
                "fasta": args.fasta,
                "fai": args.fai,
                "gene_annotation": args.gene_annotation,
            },
            "parameters": {
                "window_size": args.window_size,
                "context_per_side": args.context_per_side,
                "context_length": args.window_size + 2 * args.context_per_side,
                "center_index_0based": args.context_per_side + args.window_size // 2,
                "core_window_index_start": args.context_per_side,
                "core_window_index_end_exclusive": args.context_per_side + args.window_size,
                "context_role": "reserve_for_random_crop_shift_in_training_only",
                "selection_and_filters_span": "core_window_only",
                "context_padding_outside_chromosome": "N_and_zero_signal",
                "positive_tss_center_policy": "summit",
                "positive_nontss_center_policy": "summit",
                "context_flank_pval_exceedances": context_flank_pval_exceedances,
                "max_summit_distance": args.max_summit_distance,
                "min_nontss_summit_distance": args.min_nontss_summit_distance,
                "nontss_distance_boundary_inclusive": True,
                "positive_genic_membership_rule": "summit_base_within_gene_interval",
                "positive_noncoding_membership_rule": "summit_base_outside_all_gene_intervals",
                "pval_threshold": args.pval_threshold,
                "negative_tss_selection_mode": "all_valid_offsets_at_most_one_selected_per_tss",
                "legacy_seed_ignored": args.seed,
                "offset_distribution": args.offset_distribution,
                "kde_bandwidth": args.kde_bandwidth,
                "summit_min_value": args.summit_min_value,
                "wide_summit_policy": args.wide_summit_policy,
                "exclude_positive_overlap": not args.allow_positive_overlap,
                "gene_feature_types": gene_feature_types,
                "gc_bin_width": args.gc_bin_width,
                "gc_bin_width_role": "candidate_quotas_and_plots",
                "gc_ks_tolerance": args.gc_ks_tolerance,
                "gc_quantile_tolerance": args.gc_quantile_tolerance,
                "gc_matching_tool": "GCMatcher",
                "gc_matching_deterministic": True,
                "gc_matching_covariate": "gc_fraction",
                "gc_matching_pairs": {
                    "negative_tss": "positive_tss",
                    "negative_genic": "positive_genic",
                    "negative_noncoding": "positive_noncoding",
                },
                "gc_matching_count_mode": "maximize_without_fixed_target_subject_to_ks_tolerance",
                "gc_matching_retention_rule": "disjoint_windows_with_origin_constraints",
                "gc_capacity_method": "greedy_nonoverlapping_count_per_gc_threshold_from_both_tails",
                "gc_capacity_bound": "min_over_thresholds_of_capacity_divided_by_positive_share_minus_tolerance",
                "gc_comparison_plots": "cumulative_capacity_and_selected_vs_positive_ecdf",
                "gc_plot_format": "SVG",
                "genic_window_placement": args.genic_window_placement,
                "candidate_grid_step": 1,
                "candidate_scan_chunk_size": args.candidate_scan_chunk_size,
                "workers_used": 1,
                "save_candidate_pools": args.save_candidate_pools,
                "window_files_compressed": compress,
                "dinucleotide_shuffle": args.dinucleotide_shuffle,
                "dinucleotide_shuffle_seed": args.dinucleotide_shuffle_seed,
                "dinucleotide_shuffle_scope": (
                    "all_output_sequences"
                    if args.dinucleotide_shuffle
                    else "disabled"
                ),
                "signals_changed_by_dinucleotide_shuffle": False,
            },
            "counts": {
                "tss_plus": len(tss_plus),
                "tss_minus": len(tss_minus),
                "summits": summit_count,
                "wide_summit_intervals": wide_summit_count,
                "positive": len(all_positive_samples),
                "positive_tss": len(positives),
                "positive_tss_plus": len(offsets_plus),
                "positive_tss_minus": len(offsets_minus),
                "positive_genic": len(positive_genic),
                "positive_noncoding": len(positive_noncoding),
                "negative_total": (
                    len(negatives) + len(genic_negatives) + len(noncoding_negatives)
                ),
                "negative_tss": len(negatives),
                "negative_tss_plus": sum(1 for x in negatives if x.tss.strand == "+"),
                "negative_tss_minus": sum(1 for x in negatives if x.tss.strand == "-"),
                "negative_genic": len(genic_negatives),
                "negative_noncoding": len(noncoding_negatives),
                "annotated_genes_total": len(genes),
                "annotated_genes_on_data_chromosomes": sum(
                    1 for gene in genes if gene.chrom_key in bundles
                ),
            },
            "positive_selection": {
                "tss_selection": positive_stats,
                "all_summit_classification": positive_group_stats,
                "all_loaded_summits_included": (
                    positive_group_stats["summits_not_in_positive_output"] == 0
                ),
            },
            "positive_gc_distribution": {
                positive_type: {
                    "count": len(fractions),
                    "min": float(np.min(fractions)),
                    "max": float(np.max(fractions)),
                    "mean": float(np.mean(fractions)),
                    "median": float(np.median(fractions)),
                }
                for positive_type, fractions in positive_gc_fractions.items()
            },
            "negative_tss_selection": negative_summary,
            "negative_genic_selection": genic_summary,
            "negative_noncoding_selection": noncoding_summary,
            "gene_annotation": gene_annotation_summary,
            "chromosome_mapping": chrom_map_rows,
        }
        with (args.output_dir / "run_summary.json").open("wt", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False, cls=NumpyJSONEncoder)
            handle.write("\n")

        LOGGER.info("Fertig. Ausgabe: %s", args.output_dir)
        return 0
    finally:
        for handle in (fe_bw, pval_bw, summit_bw, fasta):
            if handle is not None:
                try:
                    handle.close()
                except Exception:
                    LOGGER.debug("Fehler beim Schliessen eines Eingabe-Handles", exc_info=True)
        for pool_name in ("tss_pool", "genic_pool", "noncoding_pool"):
            pool = locals().get(pool_name)
            if pool is not None:
                pool.close()
        match_temp_dir.cleanup()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        LOGGER.error("Abgebrochen.")
        raise SystemExit(130)
    except Exception as exc:
        LOGGER.error("%s", exc)
        if logging.getLogger().isEnabledFor(logging.DEBUG):
            LOGGER.exception("Details")
        raise SystemExit(1)
