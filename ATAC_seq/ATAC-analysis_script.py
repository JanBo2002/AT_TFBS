#!/usr/bin/env python3
"""
ATAC-seq batch pipeline (sister script of ChIP-analysis_script.py).

The read-level filter chain is IDENTICAL to the ChIP pipeline (fastp -> Bowtie2 ->
MAPQ + chromosome whitelist -> samtools fixmate/markdup), so ChIP and ATAC datasets are
processed on the same basis. ATAC-seq has no input control, therefore peak calling and
quality control are different: no control BAM, no SPP/NSC/RSC, no fold-enrichment tracks.
Instead the pipeline reports the ATAC-specific metrics (organelle fraction, fragment size
distribution, TSS enrichment, PBC1/PBC2, depth-standardised peak count and FRiP).

Pipeline scope (per experiment = one dataset / GEO series or sample):
  1.  Read and validate YAML experiment configs
  2.  Create experiment output folders
  3.  Resolve/download FASTQ files (local / SRA / ENA / URL / Zenodo)
  4.  FastQC before trimming
  5.  Trim reads with fastp (same parameters as ChIP; Nextera adapter explicit for SE)
  6.  FastQC after trimming
  7.  Align with Bowtie2 + samtools (same parameters as ChIP; index must contain organelles)
  8.  samtools idxstats on the UNFILTERED BAM -> organelle / nuclear read fraction
  9.  Filter (MAPQ, chromosome whitelist) and remove duplicates (same chain as ChIP)
      + library complexity: NRF (markdup stats, as in ChIP) and PBC1/PBC2 (ENCODE definition)
  10. Fragment size distribution (PE only): nucleosome-free / mono- / di-nucleosomal fractions
  11. Tn5-shifted BAM (+4/-5) -> cut-site bigWig; CPM fragment bigWig (as in ChIP)
  12. TSS enrichment score (ENCODE-style) from the cut-site signal around annotated TSS
  13. MACS3 peak calling WITHOUT control (PE: BAMPE; SE: --nomodel --shift -100 --extsize 200)
  14. Greenscreen/blacklist filtering of peaks with bedtools (as in ChIP)
  15. FRiP (same definition as ChIP) + peak count/FRiP on a fixed number of fragments
  16. deepTools fingerprint, optional ataqv metrics
  17. IDR for biological replicates, consensus peak set (as in ChIP)
  18. Per-experiment summary, per-sample metrics JSON, MultiQC custom tables, MultiQC report
  Batch level (after all experiments):
  19. QC summary table over all samples with status tiers (cohort percentiles from 5 samples on)
  20. Peak overlap (Jaccard) matrix and merged peak set across experiments
  21. ataqv viewer (mkarv) and one MultiQC report over all experiments

Usage:
  python ATAC-analysis_script.py --config-dir configs --outdir results [--threads N]
  python ATAC-analysis_script.py --outdir results --summary-only     (batch summary only)
"""

import argparse
import gzip
import itertools
import json
import math
import re
import shlex
import shutil
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import yaml


ALLOWED_LAYOUTS = {"SE", "PE"}
ALLOWED_SOURCES = {"local", "sra", "geo", "ena", "url", "zenodo"}
AUTO_LAYOUT = "AUTO"   # allowed for source=geo: layout is taken from the SRA metadata at download time
CONFIG_SUFFIXES = {".yaml", ".yml"}
DOWNLOAD_RETRY_DELAYS = [30, 60, 120, 300]
NEXTERA_ADAPTER = "CTGTCTCTTATACACATCT"

# Fixed reference for Arabidopsis thaliana (TAIR10.1, Ensembl-style chromosome names 1-5, Mt, Pt).
# Set the paths ONCE here; a config needs a `reference:` block only to deviate from these values
# (e.g. another assembly). genome_size: nuclear chromosomes 1-5 of TAIR10.1 = 119,146,348 bp.
DEFAULT_REFERENCE = {
    "genome_index": "/path/to/bowtie2_index/TAIR10.1.atlas",          # Bowtie2 index prefix, MUST contain Mt and Pt
    "genome_size": "1.19e8",                                          # MACS3 -g, same value as in the ChIP configs
    "mask_bed": "/path/to/arabidopsis_greenscreen_20inputs.bed",      # Greenscreen (Klasfeld et al. 2022); None = no mask
    "annotation_gtf": "/path/to/Araport11.gtf",                       # TSS source for the TSS enrichment; None = skip
    "tss_bed": None,                                                  # ready-made TSS BED instead of the GTF
}

DEFAULT_PARAMS = {
    "threads": 6,
    "min_mapq": 30,                      # identical to ChIP pipeline; Arabidopsis ATAC-seq: reads below quality 30 discarded in
                                         # Hellens et al. 2023 (Sci Data 10:490)
    "macs_qvalue": 0.01,                 # identical to ChIP pipeline
    "macs_se_shift": -100,               # cut-site centring for SE data; shift 100 / extsize 200 with MACS2 and g 1.2e8 for Arabidopsis
                                         # ATAC-seq in Hellens et al. 2023 (Sci Data 10:490)
    "macs_se_extsize": 200,
    "bowtie2_extra": "--dovetail",       # PE only: keep mates that overhang each other (short ATAC fragments); Bowtie2 with dovetailing
                                         # for Arabidopsis ATAC-seq in Hellens et al. 2023 (Sci Data 10:490). "" = exact ChIP command.
    "idr_threshold": 0.05,
    "download_workers": 1,
    "keep_chroms": ["1", "2", "3", "4", "5"],      # nuclear chromosomes, names exactly as in the index
    "organelle_chroms": ["Mt", "Pt"],              # organelle contigs, names as in the index (ChrM/ChrC in TAIR style)
    "fastp_extra": "--dont_eval_duplication --detect_adapter_for_pe --cut_front --cut_tail --cut_mean_quality 20 --length_required 20",
    "nextera_adapter": NEXTERA_ADAPTER,  # passed explicitly for SE data (PE uses overlap detection as in ChIP)
    "subsample_fragments": 5_000_000,    # fixed depth for comparable peak counts / FRiP (= 10 M paired reads); working value at the
                                         # usable_fragments "good" level, see note at DEFAULT_QC_THRESHOLDS; 0 disables
    "subsample_seed": 42,
    "tss_window": 2000,                  # bp on each side of the TSS
    "tss_flank": 100,                    # bp at each window end used as background
    "tss_bin_size": 10,
    "tss_feature": "gene",               # GTF/GFF feature used to derive TSS positions
    "fragment_max": 1000,                # upper limit of the fragment size histogram
    "nfr_max": 100,                      # nucleosome-free fragments: <= nfr_max
    "mono_min": 180, "mono_max": 247,    # mono-nucleosomal fragments
    "di_min": 315, "di_max": 473,        # di-nucleosomal fragments
    "run_ataqv": True,
    "run_fingerprint": True,             # deepTools plotFingerprint (optional QC; skipped on failure)
    "pooled_peaks": True,                # MACS3 on the merged replicate BAM (additional peak set; consensus stays IDR-based)
    "macs_signal_tracks": True,          # MACS3 fold-enrichment and -log10(p) bigWigs over the local background (all replicates combined)
    "fingerprint_timeout": 3600,         # seconds; plotFingerprint is killed after this and reported as a warning
    "sort_memory": "2G",
}

# QC tiers (Arabidopsis-specific where published):
#   usable_fragments   : 10 M aligned nuclear reads detect >92 % of the accessible regions found with 100 M reads
#                        (Lu et al. 2017, NAR 45:e41); 5 M fragments = 10 M paired reads. The Arabidopsis reference
#                        studies were sequenced deeper (Maher et al. 2018, Plant Cell 30:15: 31-90 M nuclear reads;
#                        Sijacic et al. 2018, Plant J 94:215: >15 M filtered reads per replicate).
#   FRiP               : "high-quality ATAC-seq data from Arabidopsis thaliana typically has a FRiP score >35%"
#                        (Schmitz et al. 2022, Plant Cell 34:503). Reference comparison only: >= 0.35 is "good",
#                        below is "below_reference" (never "poor"), because FRiP also depends on peak calling and depth.
#   tss_enrichment     : no published Arabidopsis threshold; the score depends on the TSS set used (Grandi et al.
#                        2022, Nat Protoc 17:1518). Reported without a tier unless thresholds are set in a config.
#   organelle_fraction : crude/sucrose-sedimented nuclei ~50 % organellar reads, INTACT >90 % nuclear reads
#                        (Maher et al. 2018); FANS lowers organellar reads from >50 % to ~30 % (Lu et al. 2017).
#   alignment_rate     : ENCODE ATAC-seq standard (>95 %, >80 % acceptable); species-independent here because the
#                        index contains all genome sequences.
#   NRF/PBC1/PBC2      : ENCODE library-complexity bands for ChIP-seq, as in the ChIP pipeline (ENCODE's ATAC-seq
#                        bands are stricter: acceptable from 0.7, ideal above 0.9).
# The overall status is the worst tier over OVERALL_METRICS. FRiP below the reference lowers the status to at
# most "ok", never to "poor"; tss_enrichment only counts if thresholds are configured. Organelle fraction,
# alignment rate and library complexity are tiered and listed in "flagged" but do not change the status.
# subsample_fragments (DEFAULT_PARAMS) should sit at or below the usable_fragments "good" level: only samples
# with more usable fragments than the target are standardised (column fixed_depth_applied).
# Tiers are recomputed in the batch summary from the stored metrics, so changing thresholds here and running
# --summary-only re-rates all experiments without reprocessing.
OVERALL_METRICS = ["usable_fragments", "FRiP", "tss_enrichment"]
TIER_METRICS = ["organelle_fraction", "alignment_rate", "usable_fragments", "NRF", "PBC1", "PBC2", "tss_enrichment", "FRiP"]
COHORT_MIN_SAMPLES = 5   # cohort percentiles in the batch table are only reported from this number of samples on
DEFAULT_QC_THRESHOLDS = {
    "organelle_fraction": {"good_max": 0.30, "ok_max": 0.55},
    "alignment_rate": {"good_min": 0.95, "ok_min": 0.80},
    "usable_fragments": {"good_min": 5_000_000, "ok_min": 2_500_000},
    "NRF": {"good_min": 0.80, "ok_min": 0.50},
    "PBC1": {"good_min": 0.80, "ok_min": 0.50},
    "PBC2": {"good_min": 3.0, "ok_min": 1.0},
    "FRiP": {"reference": 0.35},
}

DEFAULT_SAVE = {
    "raw_fastq": False,
    "trimmed_fastq": False,
    "fastqc": True,
    "sorted_bam": False,
    "filtered_bam": False,
    "shifted_bam": False,
    "pooled_bam": False,
    "subsampled_bam": False,
    "cpm_bigwig": True,
    "cutsite_bigwig": True,
    "peaks_raw": True,
    "peaks_filtered": True,              # always kept in practice: FRiP, IDR and consensus need the filtered peaks
    "qc_reports": True,                  # MultiQC report per experiment
}

EXPERIMENT_SUBDIRS = [
    "downloads", "trim", "fastqc/raw", "fastqc/trimmed", "align", "peaks", "peaks_fixed_depth",
    "signal", "pooled", "qc/idxstats", "qc/flagstat", "qc/pbc", "qc/fragsize", "qc/tss", "qc/frip",
    "qc/fingerprint", "qc/ataqv", "idr", "consensus", "logs", "reports", "tmp",
]

# Column order of the per-sample metrics table (per-experiment MultiQC table and batch summary).
METRIC_COLUMNS = [
    "sample", "experiment_id", "sample_id", "rep_id", "layout", "accession",
    "read_length_raw", "read_length_trimmed", "raw_reads", "reads_after_trimming", "trimming_loss_fraction",
    "alignment_rate", "unique_alignment_fraction", "multi_alignment_fraction",
    "mapped_reads_unfiltered", "organelle_fraction", "nuclear_fraction",
    "reads_after_mapq_whitelist", "duplicate_rate", "NRF", "PBC1", "PBC2", "estimated_library_size",
    "usable_fragments",
    "median_fragment_size", "nfr_fraction", "mono_fraction", "di_fraction", "long_fraction", "nfr_mono_ratio",
    "tss_enrichment", "tss_source",
    "peaks_raw", "peaks_filtered", "FRiP",
    "fixed_depth_applied", "fixed_depth_fragments", "peaks_fixed_depth", "FRiP_fixed_depth",
    "greenscreen",
]


# ----------------------------------------------------------------------------------------------
# Small helpers (identical to the ChIP pipeline where they exist there)
# ----------------------------------------------------------------------------------------------

def timestamp() -> str:
    return datetime.now().strftime("%d.%m.%Y %H:%M")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as fh:
        return sum(1 for line in fh if line.strip())


def count_gzip_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        return sum(1 for line in fh if line.strip())


def fmt(value: Any, digits: int = 4) -> str:
    """Format a metric for TSV output (NA for missing values)."""
    if value is None:
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        if math.isnan(value):
            return "NA"
        return f"{value:.{digits}f}"
    return str(value)


def parse_markdup_stats(path: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if ":" not in line:
                continue
            key, value = line.strip().split(":", 1)
            value = value.strip()
            if value.isdigit():
                stats[key] = int(value)
    return stats


def compute_library_complexity(markdup_stats: Path, thresholds: dict[str, Any]) -> dict[str, Any]:
    """NRF from samtools markdup stats, exactly as in the ChIP pipeline (status tiers from config)."""

    stats = parse_markdup_stats(markdup_stats)
    total = stats.get("READ", 0)
    nonredundant = stats.get("WRITTEN", 0)
    duplicates = stats.get("DUPLICATE TOTAL", max(total - nonredundant, 0))
    nrf = nonredundant / total if total > 0 else 0.0

    return {
        "total_reads": total,
        "nonredundant_reads": nonredundant,
        "duplicate_reads": duplicates,
        "duplicate_rate": duplicates / total if total > 0 else 0.0,
        "NRF": nrf,
        "estimated_library_size": stats.get("ESTIMATED_LIBRARY_SIZE") or None,
        "status": tier_status("NRF", nrf, thresholds),
        "markdup_stats": markdup_stats,
    }


def read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/object.")
    return data


def require_mapping(obj: Any, name: str) -> dict[str, Any]:
    if not isinstance(obj, dict):
        raise ValueError(f"'{name}' must be a mapping/object.")
    return obj


def require_list(obj: Any, name: str) -> list[Any]:
    if not isinstance(obj, list) or not obj:
        raise ValueError(f"'{name}' must be a non-empty list.")
    return obj


def require_key(mapping: dict[str, Any], key: str, where: str) -> Any:
    value = mapping.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required key '{where}.{key}'.")
    return value


def validate_number(value: Any, name: str, cast_type: type) -> None:
    try:
        cast_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"'{name}' must be {cast_type.__name__}-compatible, got {value!r}.") from exc


def as_non_empty_string_list(value: Any, name: str) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise ValueError(f"'{name}' must be a string or a non-empty list of strings.")

    cleaned = [str(item).strip() for item in values if str(item).strip()]
    if not cleaned:
        raise ValueError(f"'{name}' must not be empty.")
    return cleaned


# ----------------------------------------------------------------------------------------------
# QC status tiers
# ----------------------------------------------------------------------------------------------

def tier_status(metric: str, value: Any, thresholds: dict[str, Any]) -> str:
    """Map a metric value to good / ok / poor using the configured thresholds."""

    spec = thresholds.get(metric)
    if spec is None or value is None:
        return "NA"
    if isinstance(value, float) and math.isnan(value):
        return "NA"
    if "good_max" in spec:
        if value <= float(spec["good_max"]):
            return "good"
        if value <= float(spec["ok_max"]):
            return "ok"
        return "poor"
    if "good_min" in spec:
        if value >= float(spec["good_min"]):
            return "good"
        if value >= float(spec["ok_min"]):
            return "ok"
        return "poor"
    if "reference" in spec:
        return "good" if value >= float(spec["reference"]) else "below_reference"
    return "NA"


def merge_thresholds(custom: dict[str, Any] | None) -> dict[str, Any]:
    """Default thresholds plus per-config overrides; an override replaces the default of that metric completely."""

    merged = {metric: dict(spec) for metric, spec in DEFAULT_QC_THRESHOLDS.items()}
    for metric, spec in (custom or {}).items():
        spec = require_mapping(spec, f"qc_thresholds.{metric}")
        if not any(k in spec for k in ("good_min", "good_max", "reference")):
            raise ValueError(f"'qc_thresholds.{metric}' needs good_min/ok_min, good_max/ok_max or reference.")
        merged[metric] = dict(spec)
    return merged


# ----------------------------------------------------------------------------------------------
# Config handling
# ----------------------------------------------------------------------------------------------

def validate_sample(sample: dict[str, Any], default_layout: str, where: str) -> None:
    layout = str(sample.get("layout", default_layout)).upper()
    source = str(require_key(sample, "source", where)).lower()
    if source not in ALLOWED_SOURCES:
        raise ValueError(f"'{where}.source' must be one of {sorted(ALLOWED_SOURCES)}, got {source!r}.")
    sample["source"] = source

    if layout == AUTO_LAYOUT and source != "geo":
        raise ValueError(f"'{where}.layout' AUTO is only allowed for source='geo' (layout is then read from SRA).")
    if layout not in ALLOWED_LAYOUTS | {AUTO_LAYOUT}:
        raise ValueError(f"'{where}.layout' must be one of {sorted(ALLOWED_LAYOUTS)} (or AUTO for source='geo'), got {layout!r}.")
    sample["layout"] = layout

    fastq1 = sample.get("fastq1")
    fastq2 = sample.get("fastq2")
    fastq_url_1 = sample.get("fastq_url_1")
    fastq_url_2 = sample.get("fastq_url_2")

    if source == "local":
        if not fastq1:
            raise ValueError(f"'{where}.fastq1' is required when source='local'.")
        if layout == "PE" and not fastq2:
            raise ValueError(f"'{where}.fastq2' is required for PE local input.")
    elif source == "sra":
        accessions = as_non_empty_string_list(require_key(sample, "accession", where), f"{where}.accession")
        sample["accessions"] = accessions
        sample["accession"] = accessions[0] if len(accessions) == 1 else accessions
    elif source == "geo":
        gsm_ids = as_non_empty_string_list(require_key(sample, "accession", where), f"{where}.accession")
        bad = [g for g in gsm_ids if not re.fullmatch(r"GSM\d+", g)]
        if bad:
            raise ValueError(f"'{where}.accession' must contain GEO sample ids (GSM...) for source='geo', got {bad}.")
        sample["geo_accessions"] = gsm_ids
        sample["accession"] = gsm_ids[0] if len(gsm_ids) == 1 else gsm_ids
    else:
        if not fastq_url_1:
            raise ValueError(f"'{where}.fastq_url_1' is required when source={source!r}.")
        if layout == "PE" and not fastq_url_2:
            raise ValueError(f"'{where}.fastq_url_2' is required for PE {source!r} input.")


def get_sample_layout(sample: dict[str, Any]) -> str:
    return str(sample["layout"]).upper()


def merge_defaults(data: dict[str, Any]) -> None:
    data["reference"] = {**DEFAULT_REFERENCE, **require_mapping(data.get("reference", {}) or {}, "reference")}
    data["params"] = {**DEFAULT_PARAMS, **require_mapping(data.get("params", {}) or {}, "params")}
    data["save"] = {**DEFAULT_SAVE, **require_mapping(data.get("save", {}) or {}, "save")}
    data["_qc_thresholds_custom"] = dict(data.get("qc_thresholds") or {})
    data["qc_thresholds"] = merge_thresholds(data.get("qc_thresholds"))
    data["metadata"] = require_mapping(data.get("metadata", {}) or {}, "metadata")


def validate_config(data: dict[str, Any]) -> None:
    require_key(data, "experiment_id", "root")
    require_key(data, "sample_id", "root")

    layout = str(require_key(data, "layout", "root")).upper()
    if layout not in ALLOWED_LAYOUTS | {AUTO_LAYOUT}:
        raise ValueError(f"'layout' must be one of {sorted(ALLOWED_LAYOUTS)} or AUTO (GEO samples only), got {layout!r}.")
    data["layout"] = layout

    reference = require_mapping(data["reference"], "reference")
    hint = " (set DEFAULT_REFERENCE at the top of the script or a 'reference:' block in the config)"
    genome_index = str(require_key(reference, "genome_index", "reference"))
    require_key(reference, "genome_size", "reference")
    validate_number(reference["genome_size"], "reference.genome_size", float)
    if not any(Path(f"{genome_index}.1.{ext}").is_file() for ext in ("bt2", "bt2l")):
        raise ValueError(f"'reference.genome_index' is not a Bowtie2 index prefix: {genome_index}{hint}")
    for optional_file in ("mask_bed", "annotation_gtf", "tss_bed"):
        value = reference.get(optional_file)
        if value and not Path(str(value)).is_file():
            raise ValueError(f"'reference.{optional_file}' does not exist: {value}{hint}")

    params = require_mapping(data["params"], "params")
    for key, cast_type in [
        ("threads", int), ("min_mapq", int), ("macs_qvalue", float), ("macs_se_shift", int),
        ("macs_se_extsize", int), ("idr_threshold", float), ("download_workers", int),
        ("subsample_fragments", int), ("subsample_seed", int), ("tss_window", int), ("tss_flank", int),
        ("tss_bin_size", int), ("fragment_max", int), ("nfr_max", int), ("mono_min", int), ("mono_max", int),
        ("di_min", int), ("di_max", int),
    ]:
        validate_number(params.get(key), f"params.{key}", cast_type)
    if int(params["download_workers"]) < 1:
        raise ValueError("'params.download_workers' must be >= 1.")
    if int(params["tss_window"]) <= 2 * int(params["tss_flank"]):
        raise ValueError("'params.tss_window' must be larger than twice 'params.tss_flank'.")
    if int(params["tss_window"]) % int(params["tss_bin_size"]) or int(params["tss_flank"]) % int(params["tss_bin_size"]):
        raise ValueError("'params.tss_window' and 'params.tss_flank' must be multiples of 'params.tss_bin_size'.")
    require_list(params.get("keep_chroms"), "params.keep_chroms")
    params["keep_chroms"] = [str(c) for c in params["keep_chroms"]]
    organelles = params.get("organelle_chroms") or []
    if not isinstance(organelles, list):
        raise ValueError("'params.organelle_chroms' must be a list (may be empty).")
    params["organelle_chroms"] = [str(c) for c in organelles]

    replicates = require_list(require_key(data, "replicates", "root"), "replicates")
    seen_rep_ids: set[str] = set()
    for idx, rep in enumerate(replicates, start=1):
        rep = require_mapping(rep, f"replicates[{idx}]")
        rep_id = str(require_key(rep, "rep_id", f"replicates[{idx}]")).strip()
        if rep_id in seen_rep_ids:
            raise ValueError(f"Duplicate rep_id {rep_id!r}.")
        seen_rep_ids.add(rep_id)
        sample = require_mapping(require_key(rep, "sample", f"replicates[{idx}]"), f"replicates[{idx}].sample")
        validate_sample(sample, layout, f"replicates[{idx}].sample")


def find_config_files(config_dir: Path) -> list[Path]:
    if not config_dir.is_dir():
        raise ValueError(f"Config directory does not exist: {config_dir}")
    return sorted(p for p in config_dir.iterdir() if p.is_file() and p.suffix.lower() in CONFIG_SUFFIXES)


def make_experiment_dirs(outdir: Path, experiment_id: str) -> Path:
    experiment_dir = outdir / experiment_id
    for subdir in EXPERIMENT_SUBDIRS:
        (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
    return experiment_dir


def load_config(path: Path, outdir: Path) -> dict[str, Any]:
    try:
        data = read_yaml(path)
        merge_defaults(data)
        validate_config(data)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc

    experiment_id = str(data["experiment_id"])
    experiment_dir = make_experiment_dirs(outdir, experiment_id)
    shutil.copy2(path, experiment_dir / "config_used.yaml")

    data["config_path"] = path
    data["experiment_dir"] = experiment_dir
    return data


def load_configs(config_dir: Path, outdir: Path) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for path in find_config_files(config_dir):
        cfg = load_config(path, outdir)
        if cfg["experiment_id"] in seen_ids:
            raise ValueError(f"{path}: duplicate experiment_id {cfg['experiment_id']!r}.")
        seen_ids.add(cfg["experiment_id"])
        configs.append(cfg)
    if not configs:
        raise ValueError(f"No YAML config files found in {config_dir}")
    return configs


# ----------------------------------------------------------------------------------------------
# Command execution and downloads (identical to the ChIP pipeline)
# ----------------------------------------------------------------------------------------------

def run_command(cmd: str, log_path: Path | None = None) -> None:
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT, text=True)
    else:
        result = subprocess.run(cmd, shell=True)

    if result.returncode != 0:
        msg = f"Command failed with exit code {result.returncode}: {cmd}"
        if log_path:
            msg += f"\nLog: {log_path}"
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if lines:
                    msg += "\nLast log lines:\n" + "\n".join(lines[-40:])
        raise RuntimeError(msg)


def run_capture(cmd: str) -> str:
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}\n{result.stderr}")
    return result.stdout.strip()


def run_download_command(cmd: str, log_path: Path) -> None:
    """Run a network download command with backoff for transient remote failures."""

    for attempt in range(len(DOWNLOAD_RETRY_DELAYS) + 1):
        try:
            run_command(cmd, log_path)
            return
        except RuntimeError:
            if attempt == len(DOWNLOAD_RETRY_DELAYS):
                raise

            delay = DOWNLOAD_RETRY_DELAYS[attempt]
            print(
                f"{timestamp()}  Download command failed; retrying in {delay}s "
                f"(attempt {attempt + 2}/{len(DOWNLOAD_RETRY_DELAYS) + 1}) .."
            )
            time.sleep(delay)


def cleanup_stale_sra_lock(tmp_dir: Path, accession: str) -> None:
    """Remove an interrupted SRA prefetch staging directory when a lock remains."""

    accession_dir = tmp_dir / accession
    if not accession_dir.exists():
        return

    lock_files = list(accession_dir.glob("*.lock"))
    if not lock_files:
        return

    print(
        f"{timestamp()}  Removing stale SRA lock for {accession}: "
        f"{', '.join(str(path) for path in lock_files)}"
    )
    shutil.rmtree(accession_dir)


def run_sra_prefetch(accession: str, tmp_dir: Path, log_path: Path) -> None:
    """Download one SRA accession and recover from stale prefetch locks."""

    cmd = (
        f"export TMPDIR={shlex.quote(str(tmp_dir))} && "
        f"prefetch {shlex.quote(accession)} --output-directory {shlex.quote(str(tmp_dir))}"
    )

    for attempt in range(len(DOWNLOAD_RETRY_DELAYS) + 1):
        cleanup_stale_sra_lock(tmp_dir, accession)
        try:
            run_command(cmd, log_path)
            return
        except RuntimeError:
            if attempt == len(DOWNLOAD_RETRY_DELAYS):
                raise

            delay = DOWNLOAD_RETRY_DELAYS[attempt]
            print(
                f"{timestamp()}  SRA prefetch failed for {accession}; retrying in {delay}s "
                f"(attempt {attempt + 2}/{len(DOWNLOAD_RETRY_DELAYS) + 1}) .."
            )
            time.sleep(delay)


def sample_label(experiment_id: str, rep_id: str) -> str:
    return f"{experiment_id}.{rep_id}"


def sample_accession_text(sample: dict[str, Any]) -> str:
    source = str(sample["source"])
    if source == "sra":
        return ",".join(as_non_empty_string_list(sample.get("accessions", sample["accession"]), "accession"))
    if source == "geo":
        gsm = ",".join(sample.get("geo_accessions", []))
        runs = sample.get("accessions")
        return f"{gsm}:{','.join(runs)}" if runs else gsm
    if source == "local":
        return Path(str(sample["fastq1"])).name
    return str(sample.get("fastq_url_1", ""))


def parse_pysradb_table(text: str) -> list[dict[str, str]]:
    """Parse a tab-separated pysradb table into a list of row dicts (empty if no header)."""

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    header = [col.strip() for col in lines[0].split("\t")]
    rows = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < 2:
            continue
        rows.append({header[i] if i < len(header) else f"col{i}": fields[i].strip() for i in range(len(fields))})
    return rows


def resolve_geo_runs(gsm_ids: list[str], label: str, experiment_dir: Path) -> tuple[list[str], str | None, dict[str, Any]]:
    """GSM -> SRR via pysradb (SRA metadata); returns runs, library layout (PE/SE/None) and provenance.

    Primary route: `pysradb metadata --detailed GSM...` (runs + library_layout + strategy in one table).
    Fallback: `pysradb gsm-to-srr GSM...` (runs only). Results are written to downloads/<label>.geo_resolution.tsv.
    """

    if shutil.which("pysradb") is None:
        raise RuntimeError(
            f"{label}: source 'geo' needs pysradb (conda install -c bioconda pysradb). "
            "Alternatively resolve the GSM ids yourself and use source: sra with the SRR accessions."
        )

    logs_dir = experiment_dir / "logs"
    downloads_dir = experiment_dir / "downloads"
    logs_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    runs: list[str] = []
    layouts: set[str] = set()
    strategies: set[str] = set()
    provenance_rows: list[dict[str, str]] = []

    for gsm in gsm_ids:
        rows: list[dict[str, str]] = []
        try:
            out = run_capture(f"pysradb metadata --detailed {shlex.quote(gsm)}")
            rows = parse_pysradb_table(out)
        except RuntimeError as exc:
            print(f"\t\t  NOTE: pysradb metadata failed for {gsm}; trying gsm-to-srr. ({str(exc).splitlines()[0]})")

        run_col = next((c for c in (rows[0].keys() if rows else []) if c.lower() in {"run_accession", "run", "run_acc"}), None)
        alias_col = next((c for c in (rows[0].keys() if rows else []) if "experiment_alias" in c.lower()), None)
        layout_col = next((c for c in (rows[0].keys() if rows else []) if "library_layout" in c.lower()), None)
        strategy_col = next((c for c in (rows[0].keys() if rows else []) if "library_strategy" in c.lower()), None)

        gsm_runs: list[str] = []
        if rows and run_col:
            for row in rows:
                if alias_col and row.get(alias_col) and row[alias_col] != gsm and gsm not in row[alias_col]:
                    continue
                run = row.get(run_col, "")
                if re.fullmatch(r"[SED]RR\d+", run) and run not in gsm_runs:
                    gsm_runs.append(run)
                    if layout_col and row.get(layout_col):
                        layouts.add(row[layout_col].upper())
                    if strategy_col and row.get(strategy_col):
                        strategies.add(row[strategy_col])
                    provenance_rows.append({
                        "gsm": gsm, "run": run,
                        "layout": row.get(layout_col, "") if layout_col else "",
                        "strategy": row.get(strategy_col, "") if strategy_col else "",
                    })
        if not gsm_runs:
            out = run_capture(f"pysradb gsm-to-srr {shlex.quote(gsm)}")
            for row in parse_pysradb_table(out):
                for value in row.values():
                    if re.fullmatch(r"[SED]RR\d+", value) and value not in gsm_runs:
                        gsm_runs.append(value)
                        provenance_rows.append({"gsm": gsm, "run": value, "layout": "", "strategy": ""})
        if not gsm_runs:
            raise RuntimeError(f"{label}: no SRA runs found for {gsm} (does the GEO sample have raw data in SRA?).")
        runs.extend(r for r in gsm_runs if r not in runs)

    with (downloads_dir / f"{label}.geo_resolution.tsv").open("w", encoding="utf-8") as fh:
        fh.write("gsm\trun\tlibrary_layout\tlibrary_strategy\n")
        for row in provenance_rows:
            fh.write(f"{row['gsm']}\t{row['run']}\t{row['layout']}\t{row['strategy']}\n")

    detected: str | None = None
    if layouts == {"PAIRED"}:
        detected = "PE"
    elif layouts == {"SINGLE"}:
        detected = "SE"
    elif len(layouts) > 1:
        raise RuntimeError(f"{label}: runs of {gsm_ids} have mixed library layouts {sorted(layouts)}; split them into separate samples.")

    info = {"runs": runs, "layout": detected, "strategies": sorted(strategies)}
    print(f"\t\t  GEO {','.join(gsm_ids)} -> {','.join(runs)} (layout: {detected or 'unknown'}; strategy: {', '.join(sorted(strategies)) or 'unknown'})")
    if strategies and not any("ATAC" in st.upper() for st in strategies):
        print(f"\t\t  WARNING: SRA library strategy for {','.join(gsm_ids)} is {sorted(strategies)}, not ATAC-seq.")
    return runs, detected, info


def resolve_fastqs(sample: dict[str, Any], layout: str, label: str, experiment_dir: Path, threads: int) -> tuple[Path, Path | None]:
    """Return local FASTQ paths, downloading the sample first when needed."""

    downloads_dir = experiment_dir / "downloads"
    tmp_dir = experiment_dir / "tmp" / label
    logs_dir = experiment_dir / "logs"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    source = sample["source"]

    if source == "geo":
        runs, detected, _info = resolve_geo_runs(sample["geo_accessions"], label, experiment_dir)
        if layout == AUTO_LAYOUT:
            if detected is None:
                raise RuntimeError(f"{label}: library layout could not be read from SRA; set layout: PE or SE for this sample.")
            layout = detected
        elif detected and detected != layout:
            raise RuntimeError(
                f"{label}: config layout is {layout}, but SRA reports {detected} for {sample['geo_accessions']}. "
                "Fix the layout in the config (or use layout: AUTO)."
            )
        sample["layout"] = layout
        sample["accessions"] = runs
        sample["sra_layout"] = detected
        source = "sra"

    if source == "local":
        fq1 = Path(sample["fastq1"])
        fq2 = Path(sample["fastq2"]) if layout == "PE" else None
        if not fq1.exists():
            raise FileNotFoundError(f"FASTQ not found: {fq1}")
        if fq2 and not fq2.exists():
            raise FileNotFoundError(f"FASTQ not found: {fq2}")
        return fq1, fq2

    fq1 = downloads_dir / f"{label}.R1.raw.fastq.gz"
    fq2 = downloads_dir / f"{label}.R2.raw.fastq.gz" if layout == "PE" else None

    if fq1.exists() and (layout != "PE" or (fq2 and fq2.exists())):
        print(f"{timestamp()}  Reusing existing download for {label}: {fq1}")
        return fq1, fq2

    print(f"{timestamp()}  Downloading {label} from {source}: in progress ..")

    if source in {"ena", "url", "zenodo"}:
        run_download_command(f'curl -L "{sample["fastq_url_1"]}" -o {fq1}', logs_dir / f"{label}.download.R1.log")
        if layout == "PE":
            run_download_command(f'curl -L "{sample["fastq_url_2"]}" -o {fq2}', logs_dir / f"{label}.download.R2.log")

    elif source == "sra":
        accessions = as_non_empty_string_list(sample.get("accessions", sample["accession"]), "accession")
        run_fq1s: list[Path] = []
        run_fq2s: list[Path] = []

        for accession in accessions:
            for stale_path in [
                tmp_dir / f"{accession}.fastq",
                tmp_dir / f"{accession}_1.fastq",
                tmp_dir / f"{accession}_2.fastq",
                tmp_dir / f"{accession}.R1.fastq.gz",
                tmp_dir / f"{accession}.R2.fastq.gz",
            ]:
                stale_path.unlink(missing_ok=True)

            run_sra_prefetch(accession, tmp_dir, logs_dir / f"{label}.{accession}.prefetch.log")
            run_command(
                " ".join([
                    "fasterq-dump",
                    f"$(find {tmp_dir} -name '{accession}.sra' -o -name '{accession}.sralite*' | head -n1)",
                    f"--outdir {tmp_dir}",
                    f"--temp {tmp_dir}",
                    f"--threads {threads}",
                    "--split-files" if layout == "PE" else "",
                ]),
                logs_dir / f"{label}.{accession}.fasterq_dump.log",
            )

            run_fq1 = tmp_dir / f"{accession}.R1.fastq.gz"
            run_fq1s.append(run_fq1)
            if layout == "PE":
                run_fq2 = tmp_dir / f"{accession}.R2.fastq.gz"
                run_fq2s.append(run_fq2)
                if not (tmp_dir / f"{accession}_1.fastq").exists() or not (tmp_dir / f"{accession}_2.fastq").exists():
                    raise RuntimeError(
                        f"{label}.{accession}: config layout is PE, but fasterq-dump did not produce "
                        f"{accession}_1.fastq and {accession}_2.fastq. Check the SRA library layout."
                    )
                run_command(f"pigz -p {threads} -c {tmp_dir}/{accession}_1.fastq > {run_fq1}")
                run_command(f"pigz -p {threads} -c {tmp_dir}/{accession}_2.fastq > {run_fq2}")
                run_command(f"rm -f {tmp_dir}/{accession}_1.fastq {tmp_dir}/{accession}_2.fastq")
            else:
                if not (tmp_dir / f"{accession}.fastq").exists():
                    raise RuntimeError(
                        f"{label}.{accession}: config layout is SE, but fasterq-dump did not produce "
                        f"{accession}.fastq. Check the SRA library layout."
                    )
                run_command(f"pigz -p {threads} -c {tmp_dir}/{accession}.fastq > {run_fq1}")
                run_command(f"rm -f {tmp_dir}/{accession}.fastq")

            run_command(f"rm -rf {tmp_dir}/{accession}")

        run_command(f"cat {' '.join(str(path) for path in run_fq1s)} > {fq1}")
        if layout == "PE":
            run_command(f"cat {' '.join(str(path) for path in run_fq2s)} > {fq2}")
        if len(accessions) > 1:
            print(f"\t\t  Merged {len(accessions)} SRA runs for {label}: {', '.join(accessions)}")

    print(f"\t\t  Download complete: {fq1}")
    return fq1, fq2


def start_downloads(
    data: dict[str, Any],
    executor: ThreadPoolExecutor,
    futures: dict[tuple[str, str], Future[tuple[Path, Path | None]]] | None = None,
    limit: int | None = None,
) -> dict[tuple[str, str], Future[tuple[Path, Path | None]]]:
    """Start remote FASTQ downloads before CPU-heavy processing begins."""

    experiment_id = str(data["experiment_id"])
    experiment_dir = Path(data["experiment_dir"])
    threads = int(data["params"]["threads"])
    futures = dict(futures or {})
    started = 0

    for rep in data["replicates"]:
        sample = rep["sample"]
        if sample["source"] == "local":
            continue
        rep_id = str(rep["rep_id"])
        key = (experiment_id, rep_id)
        if key in futures:
            continue
        label = sample_label(experiment_id, rep_id)
        futures[key] = executor.submit(resolve_fastqs, sample, get_sample_layout(sample), label, experiment_dir, threads)
        started += 1
        if limit is not None and started >= limit:
            break

    if started:
        workers = int(data["params"]["download_workers"])
        print(f"{timestamp()}  Started {started} remote FASTQ download job(s) for {experiment_id} with {workers} worker(s).")

    return futures


# ----------------------------------------------------------------------------------------------
# Read QC, trimming, alignment (identical parameters to the ChIP pipeline)
# ----------------------------------------------------------------------------------------------

def run_fastqc(fq1: Path, fq2: Path | None, layout: str, label: str, outdir: Path, threads: int, stage: str) -> Path:
    sample_fastqc_dir = outdir / "fastqc" / stage / label
    sample_fastqc_dir.mkdir(parents=True, exist_ok=True)

    print(f"{timestamp()}  FastQC {stage} for {label}: in progress ..")

    inputs = f"{fq1} {fq2}" if layout == "PE" and fq2 else str(fq1)
    cmd = f"export JAVA_TOOL_OPTIONS='-Djava.awt.headless=true' && fastqc -t {threads} -o {sample_fastqc_dir} {inputs}"
    run_command(cmd, outdir / "logs" / f"{label}.fastqc.{stage}.log")
    return sample_fastqc_dir


def trim_reads(fq1: Path, fq2: Path | None, layout: str, label: str, experiment_dir: Path, params: dict[str, Any]) -> tuple[Path, Path | None, Path, Path]:
    """Trim adapters and low-quality bases with fastp.

    Parameters are identical to the ChIP pipeline. ATAC libraries carry Nextera adapters and
    many fragments shorter than the read length; for PE data fastp removes read-through
    adapter by overlap analysis (same as ChIP), for SE data the Nextera sequence is passed
    explicitly because overlap analysis is not available.
    """

    trim_dir = experiment_dir / "trim"
    trim_dir.mkdir(parents=True, exist_ok=True)

    t1 = trim_dir / f"{label}.R1.trim.fastq.gz"
    t2 = trim_dir / f"{label}.R2.trim.fastq.gz" if layout == "PE" else None
    json_report = trim_dir / f"{label}.fastp.json"
    html_report = trim_dir / f"{label}.fastp.html"

    threads = int(params["threads"])
    fastp_extra = str(params.get("fastp_extra", ""))
    adapter = str(params.get("nextera_adapter", "") or "")

    print(f"{timestamp()}  Trimming reads for {label}: in progress ..")

    if layout == "PE":
        cmd = " ".join([
            "fastp",
            "-i", str(fq1),
            "-I", str(fq2),
            "-o", str(t1),
            "-O", str(t2),
            "--thread", str(threads),
            fastp_extra,
            "--json", str(json_report),
            "--html", str(html_report),
        ])
    else:
        cmd = " ".join([
            "fastp",
            "-i", str(fq1),
            "-o", str(t1),
            "--thread", str(threads),
            fastp_extra,
            f"--adapter_sequence {adapter}" if adapter else "",
            "--json", str(json_report),
            "--html", str(html_report),
        ])

    run_command(cmd, experiment_dir / "logs" / f"{label}.fastp.log")
    return t1, t2, json_report, html_report


def parse_fastp_json(path: Path) -> dict[str, Any]:
    """Read counts and read lengths before/after trimming from the fastp JSON report."""

    result: dict[str, Any] = {
        "raw_reads": None, "reads_after_trimming": None, "trimming_loss_fraction": None,
        "read_length_raw": None, "read_length_trimmed": None,
    }
    try:
        with path.open("r", encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, ValueError):
        return result

    summary = report.get("summary", {})
    before = summary.get("before_filtering", {})
    after = summary.get("after_filtering", {})
    raw = before.get("total_reads")
    kept = after.get("total_reads")
    result["raw_reads"] = raw
    result["reads_after_trimming"] = kept
    if raw:
        result["trimming_loss_fraction"] = 1.0 - (kept or 0) / raw
    lengths_raw = [before.get("read1_mean_length"), before.get("read2_mean_length")]
    lengths_trim = [after.get("read1_mean_length"), after.get("read2_mean_length")]
    lengths_raw = [float(x) for x in lengths_raw if x]
    lengths_trim = [float(x) for x in lengths_trim if x]
    result["read_length_raw"] = round(mean(lengths_raw), 1) if lengths_raw else None
    result["read_length_trimmed"] = round(mean(lengths_trim), 1) if lengths_trim else None
    return result


def align_bowtie2(t1: Path, t2: Path | None, layout: str, label: str, data: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Align trimmed reads with Bowtie2 (identical parameters to the ChIP pipeline), sort and index.

    -X 2000 keeps multi-nucleosomal ATAC fragments; the index MUST contain the organelle
    genomes, otherwise chloroplast/mitochondrial reads are forced onto nuclear chromosomes.
    """

    experiment_dir = Path(data["experiment_dir"])
    align_dir = experiment_dir / "align"
    align_dir.mkdir(parents=True, exist_ok=True)

    params = data["params"]
    reference = data["reference"]
    threads = int(params["threads"])
    sort_threads = max(1, threads // 2)

    bam = align_dir / f"{label}.sorted.bam"
    bai = Path(f"{bam}.bai")
    log_bt2 = experiment_dir / "logs" / f"{label}.bowtie2.log"

    print(f"{timestamp()}  Aligning {label} with Bowtie2: in progress ..")

    if layout == "PE":
        cmd = " ".join([
            "bowtie2",
            "-x", str(reference["genome_index"]),
            "-1", str(t1),
            "-2", str(t2),
            "-p", str(threads),
            "--no-mixed",
            "--no-discordant",
            "-X 2000",
            str(params.get("bowtie2_extra", "") or ""),
            f"2> {log_bt2}",
            "| samtools view -bS -",
            f"| samtools sort -@ {sort_threads} -o {bam}",
        ])
    else:
        cmd = " ".join([
            "bowtie2",
            "-x", str(reference["genome_index"]),
            "-U", str(t1),
            "-p", str(threads),
            f"2> {log_bt2}",
            "| samtools view -bS -",
            f"| samtools sort -@ {sort_threads} -o {bam}",
        ])

    run_command(cmd)
    run_command(f"samtools index {bam}")

    if log_bt2.exists():
        with log_bt2.open() as fh:
            for line in fh:
                if "overall alignment rate" in line:
                    print(f"\t\t  Bowtie2: {line.strip()}")

    return bam, bai, log_bt2


def parse_bowtie2_log(path: Path, layout: str) -> dict[str, Any]:
    """Alignment statistics from the Bowtie2 summary (overall, unique, multi, discordant)."""

    result: dict[str, Any] = {
        "alignment_rate": None, "unique_alignment_fraction": None,
        "multi_alignment_fraction": None, "discordant_fraction": None,
    }
    if not path.exists():
        return result
    text = path.read_text(encoding="utf-8", errors="replace")

    total_match = re.search(r"^\s*(\d+) reads; of these:", text, re.MULTILINE)
    total = int(total_match.group(1)) if total_match else 0

    overall = re.search(r"([\d.]+)% overall alignment rate", text)
    if overall:
        result["alignment_rate"] = float(overall.group(1)) / 100.0

    if layout == "PE":
        unique = re.search(r"(\d+) \([\d.]+%\) aligned concordantly exactly 1 time", text)
        multi = re.search(r"(\d+) \([\d.]+%\) aligned concordantly >1 times", text)
        discordant = re.search(r"(\d+) \([\d.]+%\) aligned discordantly 1 time", text)
    else:
        unique = re.search(r"(\d+) \([\d.]+%\) aligned exactly 1 time", text)
        multi = re.search(r"(\d+) \([\d.]+%\) aligned >1 times", text)
        discordant = None

    if total > 0:
        if unique:
            result["unique_alignment_fraction"] = int(unique.group(1)) / total
        if multi:
            result["multi_alignment_fraction"] = int(multi.group(1)) / total
        if discordant:
            result["discordant_fraction"] = int(discordant.group(1)) / total
        elif layout == "PE":
            no_concordant = re.search(r"(\d+) \([\d.]+%\) aligned concordantly 0 times", text)
            if no_concordant and int(no_concordant.group(1)) == 0:
                result["discordant_fraction"] = 0.0
    return result


# ----------------------------------------------------------------------------------------------
# Chromosome naming helpers (identical alias logic to the ChIP pipeline)
# ----------------------------------------------------------------------------------------------

def chrom_aliases_for_target(chrom: str) -> set[str]:
    """Return common Arabidopsis chromosome aliases for a target chrom name."""

    aliases = {chrom}
    bare = chrom[3:] if chrom.lower().startswith("chr") else chrom

    if bare in {"1", "2", "3", "4", "5"}:
        aliases.update({bare, f"Chr{bare}", f"chr{bare}", f"chromosome_{bare}", f"Chromosome_{bare}"})
    elif bare.lower() in {"mt", "m", "mitochondria", "mitochondrion"}:
        aliases.update({chrom, "Mt", "mt", "ChrM", "chrM", "M", "mitochondria", "mitochondrion"})
    elif bare.lower() in {"pt", "c", "cp", "chloroplast"}:
        aliases.update({chrom, "Pt", "pt", "ChrC", "chrC", "C", "chloroplast"})

    return aliases


def bam_contigs(bam: Path) -> list[str]:
    out = run_capture(f"samtools view -H {shlex.quote(str(bam))} | awk '$1==\"@SQ\"{{sub(\"SN:\",\"\",$2); print $2}}'")
    return [line.strip() for line in out.splitlines() if line.strip()]


def contig_alias_map(contigs: list[str]) -> dict[str, str]:
    """Map every known alias (and the name itself) to the contig name used in the BAM."""

    alias_to_target: dict[str, str] = {}
    for contig in contigs:
        for alias in chrom_aliases_for_target(contig):
            alias_to_target.setdefault(alias, contig)
    return alias_to_target


def is_mitochondrion(name: str) -> bool:
    return "Mt" in chrom_aliases_for_target(name) and name not in {"1", "2", "3", "4", "5"}


# ----------------------------------------------------------------------------------------------
# Organelle fraction, filtering, library complexity
# ----------------------------------------------------------------------------------------------

def parse_idxstats(path: Path) -> list[tuple[str, int, int]]:
    rows: list[tuple[str, int, int]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4 or fields[0] == "*":
                continue
            rows.append((fields[0], int(fields[1]), int(fields[2])))
    return rows


def organelle_metrics(idxstats_rows: list[tuple[str, int, int]], keep_chroms: list[str], organelle_chroms: list[str]) -> dict[str, Any]:
    """Fraction of mapped reads on organelle contigs vs nuclear chromosomes (from the unfiltered BAM)."""

    organelle_aliases: set[str] = set()
    for name in organelle_chroms:
        organelle_aliases.update(chrom_aliases_for_target(name))
    keep = set(keep_chroms)

    mapped_total = sum(mapped for _, _, mapped in idxstats_rows)
    organelle = sum(mapped for contig, _, mapped in idxstats_rows if contig in organelle_aliases)
    nuclear = sum(mapped for contig, _, mapped in idxstats_rows if contig in keep)
    per_contig = {contig: mapped for contig, _, mapped in idxstats_rows if contig in organelle_aliases}

    return {
        "mapped_reads_unfiltered": mapped_total,
        "organelle_reads": organelle,
        "organelle_fraction": organelle / mapped_total if mapped_total else None,
        "nuclear_reads": nuclear,
        "nuclear_fraction": nuclear / mapped_total if mapped_total else None,
        "organelle_reads_per_contig": per_contig,
    }


def samtools_idxstats(bam: Path, label: str, data: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    """samtools idxstats on the UNFILTERED sorted BAM -> organelle fraction, whitelist check."""

    experiment_dir = Path(data["experiment_dir"])
    params = data["params"]
    outdir = experiment_dir / "qc" / "idxstats"
    outdir.mkdir(parents=True, exist_ok=True)
    out_txt = outdir / f"{label}.idxstats"
    run_command(f"samtools idxstats {bam} > {out_txt}")

    rows = parse_idxstats(out_txt)
    contigs = {contig for contig, _, _ in rows}
    missing = [c for c in params["keep_chroms"] if c not in contigs]
    if missing:
        raise RuntimeError(
            f"{label}: params.keep_chroms entries {missing} are not present in the alignment index. "
            f"Available contigs: {', '.join(sorted(contigs))}. Adjust keep_chroms/organelle_chroms to the index naming."
        )
    metrics = organelle_metrics(rows, params["keep_chroms"], params["organelle_chroms"])
    if metrics["organelle_fraction"] is not None:
        print(f"\t\t  Organelle fraction (unfiltered): {metrics['organelle_fraction']:.3f}; nuclear fraction: {metrics['nuclear_fraction']:.3f}")
    return out_txt, metrics


def compute_pbc(bam: Path, layout: str, label: str, data: dict[str, Any]) -> dict[str, Any]:
    """ENCODE library complexity (NRF/PBC1/PBC2) on the MAPQ/whitelist-filtered BAM BEFORE deduplication.

    Fragments are keyed by chromosome, outer coordinates and strand (PE) or 5' position and
    strand (SE). TotalFragments, DistinctFragments, OneFragment (seen once), TwoFragments
    (seen twice); NRF = Distinct/Total, PBC1 = One/Distinct, PBC2 = One/Two.
    """

    experiment_dir = Path(data["experiment_dir"])
    params = data["params"]
    pbc_dir = experiment_dir / "qc" / "pbc"
    tmp_dir = experiment_dir / "tmp"
    pbc_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = pbc_dir / f"{label}.pbc.tsv"
    sort_mem = str(params.get("sort_memory", "2G"))

    if layout == "PE":
        key_cmd = (
            f"samtools view -f 0x42 -F 0x904 {shlex.quote(str(bam))} "
            "| awk 'BEGIN{OFS=\"\\t\"}{s=int($2/16)%2; a=$4+0; b=$8+0; if(a>b){t=a;a=b;b=t} print $3,a,b,s}'"
        )
    else:
        key_cmd = (
            f"samtools view -F 0x904 {shlex.quote(str(bam))} "
            "| awk 'BEGIN{OFS=\"\\t\"}{s=int($2/16)%2; print $3,$4,s}'"
        )

    summary_awk = (
        "awk 'BEGIN{mt=0;m0=0;m1=0;m2=0} ($1==1){m1++} ($1==2){m2++} {m0++; mt+=$1} "
        "END{nrf=(mt>0)?m0/mt:0; pbc1=(m0>0)?m1/m0:0; pbc2=(m2>0)?m1/m2:-1; "
        "printf \"TotalFragments\\tDistinctFragments\\tOneFragment\\tTwoFragments\\tNRF\\tPBC1\\tPBC2\\n\"; "
        "printf \"%d\\t%d\\t%d\\t%d\\t%.6f\\t%.6f\\t%.6f\\n\", mt,m0,m1,m2,nrf,pbc1,pbc2}'"
    )
    cmd = (
        f"{key_cmd} | LC_ALL=C sort -S {sort_mem} -T {shlex.quote(str(tmp_dir))} | uniq -c | {summary_awk} "
        f"> {shlex.quote(str(out_tsv))}"
    )
    run_command(cmd, experiment_dir / "logs" / f"{label}.pbc.log")

    result: dict[str, Any] = {"PBC1": None, "PBC2": None, "pbc_total_fragments": None, "pbc_distinct_fragments": None, "path": out_tsv}
    lines = [line for line in out_tsv.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) >= 2:
        fields = lines[1].split("\t")
        if len(fields) >= 7:
            result["pbc_total_fragments"] = int(fields[0])
            result["pbc_distinct_fragments"] = int(fields[1])
            result["PBC1"] = float(fields[5])
            pbc2 = float(fields[6])
            result["PBC2"] = pbc2 if pbc2 >= 0 else None
    return result


def filter_and_dedup(bam: Path, layout: str, label: str, data: dict[str, Any]) -> tuple[Path, Path, Path, dict[str, Any]]:
    """Filter by MAPQ, keep the chromosome whitelist, remove PCR duplicates (identical to the ChIP pipeline).

    The chromosome whitelist drops organelle reads; their fraction has already been recorded
    from the unfiltered BAM by samtools_idxstats. PBC1/PBC2 are computed on the filtered,
    not yet deduplicated BAM before it is deleted.
    """

    experiment_dir = Path(data["experiment_dir"])
    align_dir = experiment_dir / "align"
    align_dir.mkdir(parents=True, exist_ok=True)

    params = data["params"]
    threads = int(params["threads"])
    sort_threads = max(1, threads // 2)
    min_mapq = int(params["min_mapq"])
    chroms = " ".join(shlex.quote(str(chrom)) for chrom in params["keep_chroms"])

    out_bam = align_dir / f"{label}.filtered.dedup.bam"
    out_bai = Path(f"{out_bam}.bai")
    stats_txt = align_dir / f"{label}.markdup_stats.txt"
    tmp_pfx = align_dir / label

    print(f"{timestamp()}  Filtering and deduplicating {label} (MAPQ>={min_mapq}): in progress ..")

    if layout == "PE":
        run_command(
            f"samtools view -b -q {min_mapq} {bam} {chroms} "
            f"| samtools sort -n -@ {sort_threads} -o {tmp_pfx}.tmp.namesort.bam"
        )
        run_command(f"samtools fixmate -m {tmp_pfx}.tmp.namesort.bam {tmp_pfx}.tmp.fixmate.bam")
        run_command(f"samtools sort -@ {sort_threads} -o {tmp_pfx}.tmp.coordsort.bam {tmp_pfx}.tmp.fixmate.bam")
        pbc = compute_pbc(Path(f"{tmp_pfx}.tmp.coordsort.bam"), layout, label, data)
        run_command(f"samtools markdup -r -f {stats_txt} {tmp_pfx}.tmp.coordsort.bam {out_bam}")
        run_command(f"rm -f {tmp_pfx}.tmp.namesort.bam {tmp_pfx}.tmp.fixmate.bam {tmp_pfx}.tmp.coordsort.bam")
    else:
        run_command(
            f"samtools view -b -q {min_mapq} {bam} {chroms} "
            f"| samtools sort -@ {sort_threads} -o {tmp_pfx}.tmp.coordsort.bam"
        )
        pbc = compute_pbc(Path(f"{tmp_pfx}.tmp.coordsort.bam"), layout, label, data)
        run_command(f"samtools markdup -r --mode s -f {stats_txt} {tmp_pfx}.tmp.coordsort.bam {out_bam}")
        run_command(f"rm -f {tmp_pfx}.tmp.coordsort.bam")

    run_command(f"samtools index {out_bam}")
    print(f"\t\t  Deduplicated BAM: {out_bam}")
    if pbc["PBC1"] is not None:
        print(f"\t\t  PBC1: {pbc['PBC1']:.3f}  PBC2: {fmt(pbc['PBC2'], 2)}")

    return out_bam, out_bai, stats_txt, pbc


def samtools_flagstat(bam: Path, label: str, data: dict[str, Any]) -> Path:
    outdir = Path(data["experiment_dir"]) / "qc" / "flagstat"
    outdir.mkdir(parents=True, exist_ok=True)
    out_txt = outdir / f"{label}.flagstat.txt"
    run_command(f"samtools flagstat {bam} > {out_txt}")
    return out_txt


def count_usable_fragments(bam: Path, layout: str) -> int:
    """Fragments after all filters: read1 of proper pairs (PE) or reads (SE), as in the ChIP FRiP."""

    if layout == "PE":
        return int(run_capture(f"samtools view -c -f 0x42 -F 0x904 {shlex.quote(str(bam))}"))
    return int(run_capture(f"samtools view -c -F 0x904 {shlex.quote(str(bam))}"))


# ----------------------------------------------------------------------------------------------
# Fragment size distribution (ATAC-specific)
# ----------------------------------------------------------------------------------------------

def fragment_size_distribution(bam: Path, label: str, data: dict[str, Any]) -> dict[str, Any]:
    """Insert size histogram of proper pairs; NFR / mono / di-nucleosomal fractions (PE only)."""

    experiment_dir = Path(data["experiment_dir"])
    params = data["params"]
    frag_dir = experiment_dir / "qc" / "fragsize"
    frag_dir.mkdir(parents=True, exist_ok=True)
    hist_tsv = frag_dir / f"{label}.fragsize.tsv"
    fragment_max = int(params["fragment_max"])

    print(f"{timestamp()}  Fragment size distribution for {label}: in progress ..")
    cmd = (
        f"samtools view -f 0x42 -F 0x904 {shlex.quote(str(bam))} "
        f"| awk -v max={fragment_max} '{{t=$9; if(t<0)t=-t; if(t>0 && t<=max) h[t]++}} END{{for(i in h) print i\"\\t\"h[i]}}' "
        f"| sort -k1,1n > {shlex.quote(str(hist_tsv))}"
    )
    run_command(cmd, experiment_dir / "logs" / f"{label}.fragsize.log")

    hist: dict[int, int] = {}
    for line in hist_tsv.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t")
        if len(fields) == 2 and fields[0].isdigit():
            hist[int(fields[0])] = int(fields[1])

    total = sum(hist.values())
    result: dict[str, Any] = {
        "fragments_in_histogram": total, "median_fragment_size": None, "nfr_fraction": None,
        "mono_fraction": None, "di_fraction": None, "long_fraction": None, "nfr_mono_ratio": None,
        "histogram": hist, "path": hist_tsv,
    }
    if total == 0:
        return result

    nfr = sum(c for s, c in hist.items() if s <= int(params["nfr_max"]))
    mono = sum(c for s, c in hist.items() if int(params["mono_min"]) <= s <= int(params["mono_max"]))
    di = sum(c for s, c in hist.items() if int(params["di_min"]) <= s <= int(params["di_max"]))
    long_frag = sum(c for s, c in hist.items() if s > int(params["di_max"]))

    cumulative = 0
    median_size = None
    for size in sorted(hist):
        cumulative += hist[size]
        if cumulative >= total / 2:
            median_size = size
            break

    result.update({
        "median_fragment_size": median_size,
        "nfr_fraction": nfr / total,
        "mono_fraction": mono / total,
        "di_fraction": di / total,
        "long_fraction": long_frag / total,
        "nfr_mono_ratio": nfr / mono if mono > 0 else None,
    })

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        sizes = sorted(hist)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(sizes, [hist[s] / total for s in sizes], lw=1.2)
        ax.set_xlabel("Fragment length (bp)")
        ax.set_ylabel("Fraction of fragments")
        ax.set_title(f"{label}: NFR {result['nfr_fraction']:.2f}, mono {result['mono_fraction']:.2f}, di {result['di_fraction']:.2f}")
        ax.set_xlim(0, fragment_max)
        fig.tight_layout()
        fig.savefig(frag_dir / f"{label}.fragsize.png", dpi=120)
        plt.close(fig)
    except Exception as exc:  # plotting is optional
        print(f"\t\t  NOTE: fragment size plot skipped ({exc}).")

    print(f"\t\t  NFR fraction: {result['nfr_fraction']:.3f}; mono-nucleosomal: {result['mono_fraction']:.3f}; median: {median_size} bp")
    return result


# ----------------------------------------------------------------------------------------------
# Signal tracks: Tn5-shifted cut sites and CPM fragment coverage
# ----------------------------------------------------------------------------------------------

def log_issue(data: dict[str, Any], level: str, message: str) -> None:
    """Print a prominent ERROR/WARNING line and keep it for the experiment summary (the run continues)."""

    line = f"{level}: {message}"
    print(f"\n{'!' * 8} {line}\n", flush=True)
    data.setdefault("_issues", []).append(line)


def check_mask_bed(data: dict[str, Any], bam: Path) -> Path | None:
    """Validate the greenscreen/blacklist BED against the BAM contig names, once per experiment.

    Chromosome names that differ from the alignment index (e.g. Chr1 vs 1) would make bedtools and
    bamCoverage match nothing without any error. Names are therefore checked explicitly: known
    aliases are converted into a renamed copy (WARNING), unresolvable names or a missing/empty
    file raise a loud ERROR and the mask is not applied, which is also recorded in the QC tables.
    """

    if "_mask_bed_effective" in data:
        return data["_mask_bed_effective"]

    reference = data["reference"]
    mask_bed = reference.get("mask_bed")
    effective: Path | None = None
    status = "none"

    if not mask_bed:
        status = "none"
        print(f"{timestamp()}  NOTE: no reference.mask_bed given; peaks are not greenscreen-filtered.")
    elif not Path(mask_bed).is_file() or Path(mask_bed).stat().st_size == 0:
        status = "not_applied"
        log_issue(data, "ERROR", f"mask_bed {mask_bed} is missing or empty; greenscreen filter NOT applied.")
    else:
        bed_chroms: list[str] = []
        n_regions = 0
        with Path(mask_bed).open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if not line.strip() or line.startswith(("#", "track", "browser")):
                    continue
                fields = line.rstrip("\n").split("\t")
                if len(fields) < 3:
                    continue
                n_regions += 1
                if fields[0] not in bed_chroms:
                    bed_chroms.append(fields[0])
        contigs = bam_contigs(bam)
        alias_map = contig_alias_map(contigs)
        matched = [c for c in bed_chroms if c in contigs]
        mappable = {c: alias_map[c] for c in bed_chroms if c not in contigs and c in alias_map}
        unresolved = [c for c in bed_chroms if c not in contigs and c not in alias_map]

        if n_regions == 0:
            status = "not_applied"
            log_issue(data, "ERROR", f"mask_bed {mask_bed} contains no BED regions; greenscreen filter NOT applied.")
        elif not matched and not mappable:
            status = "not_applied"
            log_issue(
                data, "ERROR",
                f"mask_bed chromosome names {bed_chroms} do not match the alignment index ({', '.join(contigs)}); "
                "greenscreen filter and bigWig blacklist NOT applied. Rename the chromosomes in the BED file.",
            )
        elif mappable:
            renamed = Path(data["experiment_dir"]) / "qc" / "greenscreen_renamed.bed"
            with Path(mask_bed).open("r", encoding="utf-8", errors="replace") as src, renamed.open("w", encoding="utf-8") as dst:
                for line in src:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) >= 3 and fields[0] in mappable:
                        fields[0] = mappable[fields[0]]
                        line = "\t".join(fields) + "\n"
                    dst.write(line)
            effective = renamed
            status = "renamed"
            log_issue(
                data, "WARNING",
                f"mask_bed chromosome names were converted to the index naming ({', '.join(f'{a}->{b}' for a, b in mappable.items())}); "
                f"using {renamed}" + (f"; unresolved names ignored: {unresolved}" if unresolved else ""),
            )
        else:
            effective = Path(mask_bed)
            status = "applied"
            if unresolved:
                log_issue(data, "WARNING", f"mask_bed contains chromosome names not in the index, their regions are ignored: {unresolved}")
            print(f"{timestamp()}  Greenscreen mask OK: {n_regions} regions on {', '.join(matched)} ({mask_bed})")

    data["_mask_bed_effective"] = effective
    data["_mask_status"] = status
    return effective


def mask_option(data: dict[str, Any]) -> str:
    mask_bed = data.get("_mask_bed_effective")
    if mask_bed and Path(mask_bed).is_file():
        return f"--blackListFileName {shlex.quote(str(mask_bed))}"
    return ""


def bam_to_bigwig(bam: Path, label: str, data: dict[str, Any], layout: str) -> Path:
    """CPM-normalised fragment coverage bigWig (identical to the ChIP pipeline)."""

    experiment_dir = Path(data["experiment_dir"])
    signal_dir = experiment_dir / "signal"
    signal_dir.mkdir(parents=True, exist_ok=True)
    threads = int(data["params"]["threads"])
    bw = signal_dir / f"{label}.cpm.bw"

    print(f"{timestamp()}  Generating CPM bigWig for {label}: in progress ..")
    cmd = " ".join([
        "bamCoverage",
        "-b", str(bam),
        "-o", str(bw),
        "--normalizeUsing CPM",
        "--extendReads" if layout == "PE" else "",
        mask_option(data),
        "--binSize 10",
        "-p", str(threads),
    ])
    run_command(cmd, experiment_dir / "logs" / f"{label}.bamCoverage.log")
    return bw


def tn5_shift_bam(bam: Path, label: str, layout: str, data: dict[str, Any]) -> Path | None:
    """Shift read starts by +4/-5 (Tn5 insertion offset) with deepTools alignmentSieve (paired-end only).

    alignmentSieve --ATACshift works on fragments and drops single-end reads, so for SE data the
    cut-site track is built from the unshifted 5' read ends instead (the 4-5 bp offset is
    irrelevant at the 10-bp resolution of the TSS score). If the shifted BAM comes back empty
    (deepTools version differences), the unshifted BAM is used with a warning.
    """

    if layout != "PE":
        print(f"{timestamp()}  Tn5 shift skipped for {label} (single-end): cut sites are taken from unshifted 5' read ends.")
        return None

    experiment_dir = Path(data["experiment_dir"])
    align_dir = experiment_dir / "align"
    tmp_dir = experiment_dir / "tmp"
    threads = int(data["params"]["threads"])
    unsorted = tmp_dir / f"{label}.shifted.unsorted.bam"
    shifted = align_dir / f"{label}.shifted.bam"

    print(f"{timestamp()}  Tn5 shift (+4/-5) for {label}: in progress ..")
    try:
        run_command(
            f"alignmentSieve -b {shlex.quote(str(bam))} -o {shlex.quote(str(unsorted))} --ATACshift -p {threads}",
            experiment_dir / "logs" / f"{label}.alignmentSieve.log",
        )
        run_command(f"samtools sort -@ {max(1, threads // 2)} -o {shlex.quote(str(shifted))} {shlex.quote(str(unsorted))}")
        run_command(f"samtools index {shlex.quote(str(shifted))}")
    except RuntimeError as exc:
        print(f"\t\t  WARNING: alignmentSieve failed for {label}; cut sites are taken from the unshifted BAM. ({str(exc).splitlines()[0]})")
        unlink_if(unsorted)
        unlink_if(shifted)
        unlink_if(f"{shifted}.bai")
        return None
    unsorted.unlink(missing_ok=True)

    n_shifted = int(run_capture(f"samtools view -c {shlex.quote(str(shifted))}"))
    if n_shifted == 0:
        print(f"\t\t  WARNING: shifted BAM for {label} is empty; cut sites are taken from the unshifted BAM.")
        unlink_if(shifted)
        unlink_if(f"{shifted}.bai")
        return None
    return shifted


def cutsite_bigwig(shifted_bam: Path, label: str, data: dict[str, Any]) -> Path:
    """1-bp cut-site track (5' base of every read, Tn5-shifted for PE), CPM-normalised."""

    experiment_dir = Path(data["experiment_dir"])
    signal_dir = experiment_dir / "signal"
    threads = int(data["params"]["threads"])
    bw = signal_dir / f"{label}.cutsites.cpm.bw"

    print(f"{timestamp()}  Generating cut-site bigWig for {label}: in progress ..")
    cmd = " ".join([
        "bamCoverage",
        "-b", str(shifted_bam),
        "-o", str(bw),
        "--Offset 1",
        "--binSize 1",
        "--normalizeUsing CPM",
        mask_option(data),
        "-p", str(threads),
    ])
    run_command(cmd, experiment_dir / "logs" / f"{label}.cutsites.bamCoverage.log")
    return bw


# ----------------------------------------------------------------------------------------------
# TSS enrichment (ATAC-specific)
# ----------------------------------------------------------------------------------------------

def tss_bed_from_annotation(annotation: Path, contigs: list[str], keep_chroms: list[str], out_bed: Path, feature: str) -> int:
    """Derive unique TSS positions (BED6) from a GTF/GFF3 annotation; chrom names mapped to the BAM naming."""

    alias_map = contig_alias_map(contigs)
    keep = set(keep_chroms)
    features_wanted = [feature, "transcript", "mRNA", "gene"]
    tss_by_feature: dict[str, set[tuple[str, int, int, str, str]]] = {f: set() for f in features_wanted}

    opener = gzip.open if str(annotation).endswith(".gz") else open
    with opener(annotation, "rt", encoding="utf-8", errors="replace") as fh:  # type: ignore[arg-type]
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 8 or fields[2] not in tss_by_feature:
                continue
            chrom = alias_map.get(fields[0], fields[0])
            if chrom not in keep:
                continue
            start, end, strand = int(fields[3]), int(fields[4]), fields[6]
            if strand == "-":
                tss = end - 1
            else:
                tss = start - 1
                strand = "+"
            name = ""
            attr_match = re.search(r'(?:gene_id|ID|Parent|transcript_id)[ =]"?([^";]+)"?', fields[8] if len(fields) > 8 else "")
            if attr_match:
                name = attr_match.group(1)
            tss_by_feature[fields[2]].add((chrom, tss, tss + 1, name or f"{chrom}:{tss}", strand))

    chosen: set[tuple[str, int, int, str, str]] = set()
    for f in features_wanted:
        if tss_by_feature[f]:
            chosen = tss_by_feature[f]
            break

    unique = {(c, s, e, strand): name for c, s, e, name, strand in sorted(chosen)}
    out_bed.parent.mkdir(parents=True, exist_ok=True)
    with out_bed.open("w", encoding="utf-8") as out:
        for (c, s, e, strand), name in sorted(unique.items(), key=lambda item: (item[0][0], item[0][1])):
            out.write(f"{c}\t{s}\t{e}\t{name}\t0\t{strand}\n")
    return len(unique)


def validate_tss_bed(tss_bed: Path, data: dict[str, Any], bam: Path) -> tuple[Path | None, str]:
    """Check a user-supplied TSS BED against the BAM contigs (same logic as the mask check).

    Known chromosome aliases (e.g. Chr1 -> 1) are converted in a copy, regions outside the nuclear
    chromosomes (keep_chroms) are dropped, and a missing strand column is reported because the profile
    can then not be oriented. If no region matches the index, an ERROR is logged and TSS metrics are skipped.
    """

    contigs = bam_contigs(bam)
    alias_map = contig_alias_map(contigs)
    keep = set(data["params"]["keep_chroms"])
    out_bed = Path(data["experiment_dir"]) / "qc" / "tss" / "tss_from_bed.bed"
    out_bed.parent.mkdir(parents=True, exist_ok=True)

    n_in = n_kept = n_renamed = n_unstranded = 0
    unresolved: set[str] = set()
    with tss_bed.open("r", encoding="utf-8", errors="replace") as src, out_bed.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            n_in += 1
            chrom = fields[0]
            target = chrom if chrom in contigs else alias_map.get(chrom)
            if target is None:
                unresolved.add(chrom)
                continue
            if target not in keep:
                continue
            if target != chrom:
                n_renamed += 1
                fields[0] = target
            if len(fields) < 6 or fields[5] not in ("+", "-"):
                n_unstranded += 1
            dst.write("\t".join(fields) + "\n")
            n_kept += 1

    if n_kept == 0:
        log_issue(data, "ERROR", f"tss_bed {tss_bed}: no region on the nuclear chromosomes of the index "
                                 f"(unresolved names: {sorted(unresolved) or '-'}); TSS enrichment NOT computed.")
        return None, "not_applied"
    status = "tss_bed"
    if n_renamed:
        log_issue(data, "WARNING", f"tss_bed {tss_bed}: chromosome names of {n_renamed} regions converted to the index naming.")
        status = "tss_bed_renamed"
    if unresolved:
        log_issue(data, "WARNING", f"tss_bed {tss_bed}: chromosome names not in the index, regions ignored: {sorted(unresolved)}")
    if n_unstranded:
        log_issue(data, "WARNING", f"tss_bed {tss_bed}: {n_unstranded} of {n_kept} regions without strand in column 6; "
                                   "their profiles are not oriented, which blurs the TSS enrichment.")
    print(f"{timestamp()}  TSS BED from reference.tss_bed: {n_kept} of {n_in} regions on nuclear chromosomes ({out_bed})")
    return out_bed, status


def prepare_tss_bed(data: dict[str, Any], bam: Path) -> Path | None:
    """Return the TSS BED for this experiment, cached per experiment.

    Priority: reference.tss_bed (validated, see validate_tss_bed); otherwise TSS are derived from
    reference.annotation_gtf. If tss_bed is set, the annotation is not read at all.
    """

    reference = data["reference"]
    experiment_dir = Path(data["experiment_dir"])
    cached = data.get("_tss_bed")
    if cached is not None:
        return cached if cached else None

    tss_bed = reference.get("tss_bed")
    if tss_bed:
        if reference.get("annotation_gtf"):
            print(f"{timestamp()}  NOTE: reference.tss_bed is set; the annotation is not used for TSS positions.")
        bed, status = validate_tss_bed(Path(tss_bed), data, bam)
        data["_tss_bed"] = bed if bed else ""
        data["_tss_source"] = status
        return bed

    annotation = reference.get("annotation_gtf")
    if not annotation:
        print(f"{timestamp()}  NOTE: no reference.tss_bed / reference.annotation_gtf given; TSS enrichment and ataqv TSS metrics are skipped.")
        data["_tss_bed"] = ""
        data["_tss_source"] = "none"
        return None

    out_bed = experiment_dir / "qc" / "tss" / "tss.bed"
    n = tss_bed_from_annotation(Path(annotation), bam_contigs(bam), data["params"]["keep_chroms"], out_bed, str(data["params"]["tss_feature"]))
    if n == 0:
        log_issue(data, "ERROR", f"no TSS positions derived from {annotation} (feature '{data['params']['tss_feature']}' or "
                                 "chromosome naming?); TSS enrichment NOT computed.")
        data["_tss_bed"] = ""
        data["_tss_source"] = "not_applied"
        return None
    print(f"{timestamp()}  TSS BED with {n} unique positions: {out_bed}")
    data["_tss_bed"] = out_bed
    data["_tss_source"] = "annotation"
    return out_bed


def aggregate_profile_from_matrix(matrix_gz: Path) -> list[float]:
    """Mean signal per bin over all regions of a deepTools computeMatrix output."""

    sums: list[float] = []
    n_rows = 0
    with gzip.open(matrix_gz, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("@") or not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            values = []
            for item in fields[6:]:
                try:
                    v = float(item)
                except ValueError:
                    v = 0.0
                values.append(0.0 if math.isnan(v) else v)
            if not values:
                continue
            if not sums:
                sums = [0.0] * len(values)
            if len(values) != len(sums):
                continue
            for i, v in enumerate(values):
                sums[i] += v
            n_rows += 1
    if n_rows == 0:
        return []
    return [s / n_rows for s in sums]


def tss_score_from_profile(profile: list[float], flank_bins: int, smooth_bins: int) -> dict[str, Any]:
    """ENCODE-style TSS enrichment: max of the (smoothed) profile divided by the mean of the window flanks."""

    if not profile or len(profile) <= 2 * flank_bins:
        return {"tss_enrichment": None, "tss_background": None, "tss_center": None}
    background = mean(profile[:flank_bins] + profile[-flank_bins:])
    if background <= 0:
        return {"tss_enrichment": None, "tss_background": background, "tss_center": None}
    enrichment = [v / background for v in profile]
    smooth = max(1, smooth_bins)
    smoothed = [mean(enrichment[max(0, i - smooth // 2): min(len(enrichment), i + smooth // 2 + 1)]) for i in range(len(enrichment))]
    return {
        "tss_enrichment": max(smoothed),
        "tss_background": background,
        "tss_center": enrichment[len(enrichment) // 2],
    }


def tss_enrichment(cutsite_bw: Path, tss_bed: Path, label: str, data: dict[str, Any]) -> dict[str, Any]:
    experiment_dir = Path(data["experiment_dir"])
    params = data["params"]
    tss_dir = experiment_dir / "qc" / "tss"
    threads = int(params["threads"])
    window = int(params["tss_window"])
    bin_size = int(params["tss_bin_size"])
    flank_bins = int(params["tss_flank"]) // bin_size
    smooth_bins = max(1, 50 // bin_size)

    matrix = tss_dir / f"{label}.tss_matrix.gz"
    profile_png = tss_dir / f"{label}.tss_profile.png"
    profile_data = tss_dir / f"{label}.tss_profile.tsv"
    score_tsv = tss_dir / f"{label}.tss_enrichment.tsv"

    print(f"{timestamp()}  TSS enrichment for {label}: in progress ..")
    run_command(
        " ".join([
            "computeMatrix reference-point --referencePoint TSS",
            "-S", shlex.quote(str(cutsite_bw)),
            "-R", shlex.quote(str(tss_bed)),
            "-b", str(window), "-a", str(window),
            "--binSize", str(bin_size),
            "--missingDataAsZero",
            "-p", str(threads),
            "-o", shlex.quote(str(matrix)),
        ]),
        experiment_dir / "logs" / f"{label}.computeMatrix.log",
    )
    try:
        run_command(
            " ".join([
                "plotProfile", "-m", shlex.quote(str(matrix)),
                "-o", shlex.quote(str(profile_png)),
                "--outFileNameData", shlex.quote(str(profile_data)),
                "--plotTitle", shlex.quote(label),
            ]),
            experiment_dir / "logs" / f"{label}.plotProfile.log",
        )
    except RuntimeError as exc:
        print(f"\t\t  NOTE: plotProfile failed, score is still computed from the matrix ({str(exc).splitlines()[0]}).")

    result = tss_score_from_profile(aggregate_profile_from_matrix(matrix), flank_bins, smooth_bins)
    with score_tsv.open("w", encoding="utf-8") as fh:
        fh.write("metric\tvalue\n")
        for key in ("tss_enrichment", "tss_background", "tss_center"):
            fh.write(f"{key}\t{fmt(result[key], 6)}\n")
    if result["tss_enrichment"] is not None:
        print(f"\t\t  TSS enrichment: {result['tss_enrichment']:.2f}")
    return result


# ----------------------------------------------------------------------------------------------
# Peak calling without control, greenscreen filter, FRiP, fixed-depth metrics
# ----------------------------------------------------------------------------------------------

def call_peaks_macs3(bam: Path, peak_name: str, layout: str, data: dict[str, Any], peaks_dir: Path, bedgraph: bool = False) -> tuple[Path, Path]:
    """MACS3 peak calling WITHOUT control (ATAC-seq has no input).

    PE: fragment-based BAMPE mode, parameters as in the ChIP pipeline (-q, --keep-dup all, summits).
    SE: ATAC convention --nomodel --shift -100 --extsize 200 (cut-site centring) instead of the
    ChIP nucleosome-length fallback.
    """

    experiment_dir = Path(data["experiment_dir"])
    peaks_dir.mkdir(parents=True, exist_ok=True)
    params = data["params"]
    reference = data["reference"]

    raw_peaks = peaks_dir / f"{peak_name}.raw.narrowPeak"
    summits = peaks_dir / f"{peak_name}.summits.bed"

    print(f"{timestamp()}  Calling peaks with MACS3 (no control) for {peak_name}: in progress ..")
    cmd = [
        "macs3 callpeak",
        "-t", shlex.quote(str(bam)),
        "-f", "BAMPE" if layout == "PE" else "BAM",
        "-g", str(reference["genome_size"]),
        "-q", str(params["macs_qvalue"]),
        "--call-summits",
        "--keep-dup all",
        "--outdir", shlex.quote(str(peaks_dir)),
        "-n", shlex.quote(peak_name),
    ]
    if layout != "PE":
        cmd += ["--nomodel", "--shift", str(int(params["macs_se_shift"])), "--extsize", str(int(params["macs_se_extsize"]))]
    if bedgraph:
        cmd += ["-B", "--SPMR"]          # pileup and local-lambda bedGraphs (per million reads) for the signal tracks
    run_command(" ".join(cmd), experiment_dir / "logs" / f"{peak_name}.macs3.log")

    run_command(f"mv {shlex.quote(str(peaks_dir / (peak_name + '_peaks.narrowPeak')))} {shlex.quote(str(raw_peaks))}")
    run_command(f"mv {shlex.quote(str(peaks_dir / (peak_name + '_summits.bed')))} {shlex.quote(str(summits))}")

    print(f"\t\t  Called {count_lines(raw_peaks)} raw peaks.")
    return raw_peaks, summits


def greenscreen_filter(peaks_raw: Path, peak_name: str, data: dict[str, Any], peaks_dir: Path) -> Path:
    """Remove peaks overlapping artefact regions with bedtools intersect -v (identical to the ChIP pipeline)."""

    filtered_peaks = peaks_dir / f"{peak_name}.greenscreen.narrowPeak"
    mask_bed = data.get("_mask_bed_effective")

    if mask_bed and Path(mask_bed).is_file() and Path(mask_bed).stat().st_size > 0:
        run_command(f"bedtools intersect -v -a {shlex.quote(str(peaks_raw))} -b {shlex.quote(str(mask_bed))} > {shlex.quote(str(filtered_peaks))}")
        n_before = count_lines(peaks_raw)
        n_after = count_lines(filtered_peaks)
        print(f"\t\t  Greenscreen: removed {n_before - n_after}/{n_before} peaks, {n_after} retained.")
        if n_before > 0 and n_before == n_after:
            print(f"\t\t  NOTE: no peak of {peak_name} overlapped a greenscreen region (possible, but check the mask if this happens for every sample).")
    else:
        shutil.copy2(peaks_raw, filtered_peaks)
        print(f"\t\t  Greenscreen filter not applied ({data.get('_mask_status', 'none')}); all {count_lines(filtered_peaks)} raw peaks passed through.")

    return filtered_peaks


def compute_frip(bam: Path, peaks: Path, out_tsv: Path, layout: str) -> dict[str, Any]:
    """FRiP = fragments in peaks / usable fragments (same definition as the ChIP pipeline)."""

    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    if layout == "PE":
        total = int(run_capture(f"samtools view -c -F 0x904 -f 0x42 {shlex.quote(str(bam))}"))
        in_peaks = int(run_capture(
            f"samtools view -b -F 0x904 -f 0x42 {shlex.quote(str(bam))} "
            f"| bedtools intersect -u -abam stdin -b {shlex.quote(str(peaks))} "
            f"| samtools view -c -"
        ))
    else:
        total = int(run_capture(f"samtools view -c -F 0x904 {shlex.quote(str(bam))}"))
        in_peaks = int(run_capture(
            f"samtools view -b -F 0x904 {shlex.quote(str(bam))} "
            f"| bedtools intersect -u -abam stdin -b {shlex.quote(str(peaks))} "
            f"| samtools view -c -"
        ))

    frip = in_peaks / total if total > 0 else 0.0
    with out_tsv.open("w", encoding="utf-8") as fh:
        fh.write("metric\tvalue\n")
        fh.write(f"usable_fragments\t{total}\n")
        fh.write(f"fragments_in_peaks\t{in_peaks}\n")
        fh.write(f"FRiP\t{frip:.6f}\n")
    return {"usable_fragments": total, "fragments_in_peaks": in_peaks, "FRiP": frip, "path": out_tsv}


def fixed_depth_metrics(bam: Path, layout: str, label: str, usable: int, data: dict[str, Any]) -> dict[str, Any]:
    """Peak count and FRiP on a fixed number of fragments, so datasets of different depth are comparable."""

    params = data["params"]
    target = int(params["subsample_fragments"])
    result: dict[str, Any] = {"fixed_depth_applied": False, "fixed_depth_fragments": None, "peaks_fixed_depth": None, "FRiP_fixed_depth": None}
    if target <= 0:
        return result

    experiment_dir = Path(data["experiment_dir"])
    peaks_dir = experiment_dir / "peaks_fixed_depth"
    tmp_dir = experiment_dir / "tmp"
    threads = int(params["threads"])
    seed = int(params["subsample_seed"])

    if usable <= target:
        print(f"\t\t  Fixed-depth metrics: {usable} usable fragments <= target {target}; using full data.")
        return result

    fraction = target / usable
    frac_digits = f"{fraction:.6f}".split(".")[1]
    sub_bam = tmp_dir / f"{label}.sub{target}.bam"
    print(f"{timestamp()}  Fixed-depth metrics for {label}: subsampling {usable} -> ~{target} fragments ..")
    run_command(f"samtools view -b -s {seed}.{frac_digits} -@ {max(1, threads // 2)} {shlex.quote(str(bam))} > {shlex.quote(str(sub_bam))}")
    run_command(f"samtools index {shlex.quote(str(sub_bam))}")

    sub_name = f"{label}.fixed_depth"
    raw_peaks, _ = call_peaks_macs3(sub_bam, sub_name, layout, data, peaks_dir)
    filtered = greenscreen_filter(raw_peaks, sub_name, data, peaks_dir)
    frip = compute_frip(sub_bam, filtered, experiment_dir / "qc" / "frip" / f"{sub_name}.frip.tsv", layout)

    result.update({
        "fixed_depth_applied": True,
        "fixed_depth_fragments": frip["usable_fragments"],
        "peaks_fixed_depth": count_lines(filtered),
        "FRiP_fixed_depth": frip["FRiP"],
        "subsampled_bam": sub_bam,
    })
    print(f"\t\t  Fixed depth ({frip['usable_fragments']} fragments): {result['peaks_fixed_depth']} peaks, FRiP {frip['FRiP']:.3f}")
    return result


def plot_fingerprint(bam: Path, label: str, data: dict[str, Any]) -> tuple[Path | None, Path | None]:
    """deepTools fingerprint without control: how strongly reads concentrate in a small genome fraction.

    Optional QC step (params.run_fingerprint). A failure or timeout is reported as a warning and
    does not stop the pipeline; the fingerprint is not used for the QC tiers.
    """

    params = data["params"]
    if not bool(params.get("run_fingerprint", True)):
        return None, None

    experiment_dir = Path(data["experiment_dir"])
    threads = int(params["threads"])
    timeout_s = int(params.get("fingerprint_timeout", 3600))
    outdir = experiment_dir / "qc" / "fingerprint"
    outdir.mkdir(parents=True, exist_ok=True)
    pdf = outdir / f"{label}.fingerprint.pdf"
    tsv = outdir / f"{label}.fingerprint.tsv"
    metrics = outdir / f"{label}.fingerprint.qcmetrics.tsv"

    print(f"{timestamp()}  Fingerprint for {label}: in progress ..")
    cmd = " ".join([
        f"timeout {timeout_s}",
        "plotFingerprint",
        "-b", shlex.quote(str(bam)),
        "--labels", shlex.quote(label),
        "--plotFile", shlex.quote(str(pdf)),
        "--outRawCounts", shlex.quote(str(tsv)),
        "--outQualityMetrics", shlex.quote(str(metrics)),
        "-p", str(threads),
    ])
    try:
        run_command(cmd, experiment_dir / "logs" / f"{label}.fingerprint.log")
    except RuntimeError as exc:
        print(f"\t\t  WARNING: plotFingerprint failed or timed out for {label}; continuing without fingerprint. "
              f"Set params.run_fingerprint: false to skip it. ({str(exc).splitlines()[0]})")
        return None, None
    return pdf, tsv


def run_ataqv(bam: Path, peaks: Path, tss_bed: Path | None, label: str, data: dict[str, Any]) -> Path | None:
    """ataqv metrics (optional; used by the batch-level mkarv viewer). Failures do not stop the pipeline."""

    params = data["params"]
    if not bool(params.get("run_ataqv", True)) or shutil.which("ataqv") is None:
        if bool(params.get("run_ataqv", True)):
            print("\t\t  NOTE: ataqv not found in PATH; skipping.")
        return None

    experiment_dir = Path(data["experiment_dir"])
    outdir = experiment_dir / "qc" / "ataqv"
    outdir.mkdir(parents=True, exist_ok=True)
    autosomes = outdir / "autosomal_reference.txt"
    autosomes.write_text("\n".join(params["keep_chroms"]) + "\n", encoding="utf-8")
    metrics_json = outdir / f"{label}.ataqv.json.gz"

    contigs = set(bam_contigs(bam))
    mito = next((name for name in params["organelle_chroms"] if is_mitochondrion(name)), None)
    mito_in_bam = next((c for c in contigs if mito and c in chrom_aliases_for_target(mito)), None)

    cmd = [
        "ataqv",
        "--name", shlex.quote(label),
        "--metrics-file", shlex.quote(str(metrics_json)),
        "--peak-file", shlex.quote(str(peaks)),
        "--autosomal-reference-file", shlex.quote(str(autosomes)),
        "--ignore-read-groups",
    ]
    if tss_bed:
        cmd += ["--tss-file", shlex.quote(str(tss_bed))]
    if mito_in_bam:
        cmd += ["--mitochondrial-reference-name", shlex.quote(mito_in_bam)]
    cmd += ["arabidopsis", shlex.quote(str(bam))]

    print(f"{timestamp()}  ataqv for {label}: in progress ..")
    try:
        run_command(" ".join(cmd) + f" > {shlex.quote(str(outdir / (label + '.ataqv.txt')))}", experiment_dir / "logs" / f"{label}.ataqv.log")
    except RuntimeError as exc:
        print(f"\t\t  WARNING: ataqv failed for {label}; continuing. {str(exc).splitlines()[0]}")
        return None
    return metrics_json


# ----------------------------------------------------------------------------------------------
# IDR and consensus peaks (identical to the ChIP pipeline)
# ----------------------------------------------------------------------------------------------

def run_idr(peak_files: list[Path], data: dict[str, Any]) -> Path | None:
    if len(peak_files) < 2:
        return None
    if shutil.which("idr") is None:
        log_issue(data, "ERROR", "idr not found in PATH; IDR skipped, consensus falls back to the first replicate.")
        return None

    sample_id = str(data["sample_id"])
    experiment_dir = Path(data["experiment_dir"])
    idr_dir = experiment_dir / "idr"
    idr_dir.mkdir(parents=True, exist_ok=True)

    print(f"{timestamp()}  IDR analysis for {sample_id} ({len(peak_files)} replicates): in progress ..")

    sorted_peaks: list[Path] = []
    for i, peak_file in enumerate(peak_files, start=1):
        sorted_peak = idr_dir / f"{sample_id}.rep{i}.sorted.narrowPeak"
        run_command(f"sort -k8,8nr {shlex.quote(str(peak_file))} > {shlex.quote(str(sorted_peak))}")
        sorted_peaks.append(sorted_peak)

    idr_outputs: list[Path] = []
    for rep_a, rep_b in itertools.combinations(sorted_peaks, 2):
        pair_label = f"{rep_a.stem}__vs__{rep_b.stem}"
        idr_out = idr_dir / f"{sample_id}.{pair_label}.idr.narrowPeak"
        idr_log = idr_dir / f"{sample_id}.{pair_label}.idr.log"
        terminal_log = idr_dir / f"{sample_id}.{pair_label}.idr.stderr.log"
        cmd = " ".join([
            "idr",
            "--samples", str(rep_a), str(rep_b),
            "--input-file-type narrowPeak",
            "--rank p.value",
            "--output-file", str(idr_out),
            "--plot",
            "--log-output-file", str(idr_log),
        ])
        try:
            run_command(cmd, terminal_log)
        except RuntimeError as exc:
            log_text = ""
            for path in [terminal_log, idr_log]:
                if path.exists():
                    log_text += "\n" + path.read_text(encoding="utf-8", errors="replace")
            if "Peak files must contain at least 20 peaks post-merge" in f"{exc}\n{log_text}":
                print(f"\t\t  Skipping IDR for {pair_label}: fewer than 20 merged peaks.")
                continue
            raise
        if idr_out.exists() and idr_out.stat().st_size > 0:
            idr_outputs.append(idr_out)

    if not idr_outputs:
        print("\t\t  IDR skipped: no replicate pair had enough merged peaks.")
        return None

    if len(idr_outputs) == 1:
        merged = idr_outputs[0]
    else:
        merged = idr_dir / f"{sample_id}.all_idr.merged.bed"
        run_command(
            f"cat {' '.join(str(p) for p in idr_outputs)} "
            f"| sort -k1,1 -k2,2n "
            f"| bedtools merge -i stdin -c 5,9 -o max,max "
            f"> {merged}"
        )

    final_idr = idr_dir / f"{sample_id}.idr.narrowPeak"
    shutil.copy2(merged, final_idr)
    print(f"\t\t  Consensus IDR peaks: {count_lines(final_idr)} -> {final_idr}")
    return final_idr


def make_consensus_peaks(source_peaks: Path, data: dict[str, Any]) -> tuple[Path, Path]:
    sample_id = str(data["sample_id"])
    consensus_dir = Path(data["experiment_dir"]) / "consensus"
    consensus_dir.mkdir(parents=True, exist_ok=True)

    consensus = consensus_dir / f"{sample_id}.consensus.narrowPeak"
    summits = consensus_dir / f"{sample_id}.consensus.summits.bed"
    igv_bed = consensus_dir / f"{sample_id}.consensus.IGV.bed"
    shutil.copy2(source_peaks, consensus)

    if consensus.stat().st_size > 0:
        run_command(
            "awk 'BEGIN {OFS=\"\\t\"} "
            "{ "
            "score = (NF >= 5 && $5 >= 0 && $5 <= 1000) ? $5 : ((NF >= 4 && $4 >= 0 && $4 <= 1000) ? $4 : 0); "
            "name = (NF >= 4 && $4 !~ /^[0-9.]+$/) ? $4 : \"peak_\" NR; "
            "print $1, $2, $3, name, score, (NF >= 6 ? $6 : \".\") "
            "}' "
            f"{consensus} > {igv_bed}"
        )
        run_command(
            "awk 'BEGIN {OFS=\"\\t\"} "
            "{ "
            "if (NF >= 10 && $10 >= 0) summit = $2 + $10; "
            "else summit = int(($2 + $3) / 2); "
            "score = (NF >= 5 && $5 >= 0 && $5 <= 1000) ? $5 : ((NF >= 4 && $4 >= 0 && $4 <= 1000) ? $4 : 0); "
            "name = (NF >= 4 && $4 !~ /^[0-9.]+$/) ? $4 : \"peak_\" NR; "
            "print $1, summit, summit+1, name, score "
            "}' "
            f"{consensus} > {summits}"
        )
    else:
        summits.touch()
        igv_bed.touch()

    return consensus, summits


# ----------------------------------------------------------------------------------------------
# MACS3 signal tracks (fold enrichment and -log10 p over the local background)
# ----------------------------------------------------------------------------------------------

def bam_chrom_sizes(bam: Path) -> list[tuple[str, int]]:
    out = run_capture(f"samtools view -H {shlex.quote(str(bam))} | awk '$1==\"@SQ\"{{sub(\"SN:\",\"\",$2); sub(\"LN:\",\"\",$3); print $2\"\\t\"$3}}'")
    sizes = []
    for line in out.splitlines():
        fields = line.split("\t")
        if len(fields) == 2 and fields[1].isdigit():
            sizes.append((fields[0], int(fields[1])))
    return sizes


def bedgraph_to_bigwig(bedgraph: Path, chrom_sizes: list[tuple[str, int]], out_bw: Path, tmp_dir: Path) -> bool:
    """Clip to chromosome ends, sort, and write a bigWig (pyBigWig; falls back to bedGraphToBigWig)."""

    sizes_file = tmp_dir / f"{out_bw.stem}.chrom.sizes"
    clipped = tmp_dir / f"{out_bw.stem}.clipped.bdg"
    order = sorted(chrom_sizes, key=lambda item: item[0])
    sizes_file.write_text("".join(f"{c}\t{n}\n" for c, n in order), encoding="utf-8")
    run_command(
        "awk 'BEGIN{OFS=\"\\t\"} NR==FNR{len[$1]=$2; next} ($1 in len){e=($3>len[$1])?len[$1]:$3; if(e>$2) print $1,$2,e,$4}' "
        f"{shlex.quote(str(sizes_file))} {shlex.quote(str(bedgraph))} | LC_ALL=C sort -k1,1 -k2,2n > {shlex.quote(str(clipped))}"
    )
    try:
        import pyBigWig  # bundled with deepTools

        bw = pyBigWig.open(str(out_bw), "w")
        bw.addHeader(order)
        chunk_chroms: list[str] = []
        chunk_starts: list[int] = []
        chunk_ends: list[int] = []
        chunk_values: list[float] = []
        with clipped.open("r", encoding="utf-8") as fh:
            for line in fh:
                c, a, b, v = line.rstrip("\n").split("\t")
                chunk_chroms.append(c)
                chunk_starts.append(int(a))
                chunk_ends.append(int(b))
                chunk_values.append(float(v))
                if len(chunk_chroms) >= 500_000:
                    bw.addEntries(chunk_chroms, chunk_starts, ends=chunk_ends, values=chunk_values)
                    chunk_chroms, chunk_starts, chunk_ends, chunk_values = [], [], [], []
            if chunk_chroms:
                bw.addEntries(chunk_chroms, chunk_starts, ends=chunk_ends, values=chunk_values)
        bw.close()
        ok = True
    except ImportError:
        if shutil.which("bedGraphToBigWig") is None:
            ok = False
        else:
            run_command(f"bedGraphToBigWig {shlex.quote(str(clipped))} {shlex.quote(str(sizes_file))} {shlex.quote(str(out_bw))}")
            ok = True
    clipped.unlink(missing_ok=True)
    sizes_file.unlink(missing_ok=True)
    return ok


def macs_signal_tracks(bam: Path, name: str, data: dict[str, Any], work_dir: Path) -> dict[str, Path]:
    """Fold-enrichment and -log10(Poisson p) bigWigs from the MACS3 pileup against its local lambda.

    Without a control, MACS3 estimates the local background from the sample itself (genome-wide
    lambda and the 10-kb llocal window), so the tracks show signal relative to the local
    background, the ATAC analogue of the ChIP pipeline's fold-enrichment track (ENCODE ATAC
    produces the same two tracks). Requires the bedGraphs of a callpeak run with -B --SPMR.
    """

    experiment_dir = Path(data["experiment_dir"])
    signal_dir = experiment_dir / "signal"
    tmp_dir = experiment_dir / "tmp"
    treat = work_dir / f"{name}_treat_pileup.bdg"
    ctrl = work_dir / f"{name}_control_lambda.bdg"
    result: dict[str, Path] = {}
    if not treat.exists() or not ctrl.exists():
        log_issue(data, "ERROR", f"MACS3 bedGraphs for {name} not found; FE/-log10p tracks skipped.")
        return result

    sizes = bam_chrom_sizes(bam)
    sample_id = str(data["sample_id"])

    # --SPMR scaled the pileups to "per million fragments"; the Poisson p-values must be computed on
    # the real counts, so bdgcmp gets the number of fragments / 1e6 as scaling factor (as in the
    # ENCODE pipeline). Fold enrichment is a ratio and unaffected by the scaling.
    n_tags = None
    xls = work_dir / f"{name}_peaks.xls"
    if xls.exists():
        for line in xls.read_text(encoding="utf-8", errors="replace").splitlines():
            # BAMPE mode: "# total fragments in treatment: N"; BAM/SE mode: "# total tags in treatment: N"
            # and "# tags after filtering in treatment: N" (identical with --keep-dup all)
            if line.startswith(("# tags after filtering in treatment:", "# fragments after filtering in treatment:")):
                n_tags = int(line.split(":")[1].strip())
                break
            if line.startswith(("# total tags in treatment:", "# total fragments in treatment:")) and n_tags is None:
                n_tags = int(line.split(":")[1].strip())
    if not n_tags:
        paired = int(run_capture(f"samtools view -c -f 0x1 {shlex.quote(str(bam))} | head -c 20")) > 0
        n_tags = count_usable_fragments(bam, "PE" if paired else "SE")
    scaling = max(n_tags, 1) / 1e6

    for method, suffix in (("FE", "FE"), ("ppois", "log10p")):
        bdg = work_dir / f"{name}.{suffix}.bdg"
        run_command(
            f"macs3 bdgcmp -t {shlex.quote(str(treat))} -c {shlex.quote(str(ctrl))} -m {method} "
            f"-p 0.00001 -S {scaling:.6f} -o {shlex.quote(str(bdg))}",
            experiment_dir / "logs" / f"{name}.bdgcmp.{suffix}.log",
        )
        out_bw = signal_dir / f"{sample_id}.{suffix}.bw"
        if bedgraph_to_bigwig(bdg, sizes, out_bw, tmp_dir):
            result[suffix] = out_bw
        else:
            log_issue(data, "WARNING", f"neither pyBigWig nor bedGraphToBigWig available; {suffix} track for {sample_id} not written.")
        bdg.unlink(missing_ok=True)
    treat.unlink(missing_ok=True)
    ctrl.unlink(missing_ok=True)
    if result:
        print(f"\t\t  MACS3 signal tracks: {', '.join(p.name for p in result.values())}")
    return result


# ----------------------------------------------------------------------------------------------
# Pooled replicate outputs (merged BAM -> CPM and cut-site bigWigs, optional pooled peak set)
# ----------------------------------------------------------------------------------------------

def make_pooled_outputs(data: dict[str, Any], replicate_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge the filtered BAMs of all replicates and derive pooled signal tracks (as the ChIP pipeline
    does for its pooled track) plus, optionally, a pooled MACS3 peak set.

    The pooled peak set does not replace the IDR consensus; it is the better choice for shallow
    datasets, where single replicates are depth-limited. Overlap of the pooled peaks with every
    single-replicate peak set is reported so the support of the extra peaks is visible.
    """

    result: dict[str, Any] = {}
    sample_id = str(data["sample_id"])
    experiment_dir = Path(data["experiment_dir"])
    params = data["params"]
    threads = int(params["threads"])
    pooled_dir = experiment_dir / "pooled"
    signal_dir = experiment_dir / "signal"
    pooled_dir.mkdir(parents=True, exist_ok=True)

    layouts = {r["sample"]["layout"] for r in replicate_results}
    layout = "PE" if layouts == {"PE"} else "SE"
    bams = [Path(r["sample"]["filtered_bam"]) for r in replicate_results]
    missing = [b for b in bams if not b.exists()]
    if missing:
        log_issue(data, "ERROR", f"pooled outputs skipped, filtered BAMs missing: {missing}")
        return result

    if len(replicate_results) < 2:
        # Single replicate: no pooling, but the MACS3 signal tracks are still produced from it.
        if bool(params.get("macs_signal_tracks", True)):
            call_peaks_macs3(bams[0], f"{sample_id}.signal", layout, data, pooled_dir, bedgraph=True)
            result["signal_tracks"] = macs_signal_tracks(bams[0], f"{sample_id}.signal", data, pooled_dir)
            for leftover in pooled_dir.glob(f"{sample_id}.signal*"):
                leftover.unlink(missing_ok=True)
        return result

    print(f"{timestamp()}  Pooling {len(bams)} replicates of {sample_id}: in progress ..")
    pooled_bam = pooled_dir / f"{sample_id}.pooled.filtered.dedup.bam"
    run_command(f"samtools merge -f -@ {threads} -o {shlex.quote(str(pooled_bam))} {' '.join(shlex.quote(str(b)) for b in bams)}")
    run_command(f"samtools index {shlex.quote(str(pooled_bam))}")
    result["pooled_bam"] = pooled_bam
    result["pooled_fragments"] = count_usable_fragments(pooled_bam, layout)

    cpm_bw = signal_dir / f"{sample_id}.pooled.cpm.bw"
    run_command(
        " ".join([
            "bamCoverage", "-b", shlex.quote(str(pooled_bam)), "-o", shlex.quote(str(cpm_bw)),
            "--normalizeUsing CPM", "--extendReads" if layout == "PE" else "", mask_option(data),
            "--binSize 10", "-p", str(threads),
        ]),
        experiment_dir / "logs" / f"{sample_id}.pooled.bamCoverage.log",
    )
    result["pooled_cpm_bigwig"] = cpm_bw

    shifted = [r.get("shifted_bam") for r in replicate_results]
    cut_source = pooled_bam
    if all(shifted) and all(Path(str(b)).exists() for b in shifted):
        pooled_shifted = pooled_dir / f"{sample_id}.pooled.shifted.bam"
        run_command(f"samtools merge -f -@ {threads} -o {shlex.quote(str(pooled_shifted))} {' '.join(shlex.quote(str(b)) for b in shifted)}")
        run_command(f"samtools index {shlex.quote(str(pooled_shifted))}")
        cut_source = pooled_shifted
        result["pooled_shifted_bam"] = pooled_shifted
    cut_bw = signal_dir / f"{sample_id}.pooled.cutsites.cpm.bw"
    run_command(
        " ".join([
            "bamCoverage", "-b", shlex.quote(str(cut_source)), "-o", shlex.quote(str(cut_bw)),
            "--Offset 1", "--binSize 1", "--normalizeUsing CPM", mask_option(data), "-p", str(threads),
        ]),
        experiment_dir / "logs" / f"{sample_id}.pooled.cutsites.bamCoverage.log",
    )
    result["pooled_cutsite_bigwig"] = cut_bw
    print(f"\t\t  Pooled tracks: {cpm_bw.name}, {cut_bw.name} ({result['pooled_fragments']} fragments)")

    want_tracks = bool(params.get("macs_signal_tracks", True))
    if bool(params.get("pooled_peaks", True)) or want_tracks:
        peak_name = f"{sample_id}.pooled"
        raw_peaks, _ = call_peaks_macs3(pooled_bam, peak_name, layout, data, pooled_dir, bedgraph=want_tracks)
        if want_tracks:
            result["signal_tracks"] = macs_signal_tracks(pooled_bam, peak_name, data, pooled_dir)
    if bool(params.get("pooled_peaks", True)):
        filtered = greenscreen_filter(raw_peaks, peak_name, data, pooled_dir)
        n_pooled = count_lines(filtered)
        support = {}
        for r in replicate_results:
            rep_peaks = Path(r["filtered_peaks"])
            n_overlap = int(run_capture(
                f"bedtools intersect -u -a {shlex.quote(str(filtered))} -b {shlex.quote(str(rep_peaks))} | wc -l"
            )) if n_pooled else 0
            support[r["rep_id"]] = n_overlap / n_pooled if n_pooled else None
        frip = compute_frip(pooled_bam, filtered, experiment_dir / "qc" / "frip" / f"{peak_name}.frip.tsv", layout)
        result.update({"pooled_peaks": filtered, "pooled_peak_count": n_pooled, "pooled_support": support, "pooled_FRiP": frip["FRiP"]})
        print(f"\t\t  Pooled peaks: {n_pooled} (FRiP {frip['FRiP']:.3f}); supported by " +
              ", ".join(f"{k}: {fmt(v, 2)}" for k, v in support.items()))

    if not data["save"].get("pooled_bam", False):
        for key in ("pooled_bam", "pooled_shifted_bam"):
            bam = result.get(key)
            if bam:
                unlink_if(bam)
                unlink_if(f"{bam}.bai")
    return result


# ----------------------------------------------------------------------------------------------
# Cleanup
# ----------------------------------------------------------------------------------------------

def unlink_if(path: Any) -> None:
    if path and Path(path).exists():
        Path(path).unlink()


def cleanup_sample_files(result: dict[str, Any], sample: dict[str, Any], save: dict[str, Any]) -> None:
    if sample["source"] != "local" and not save["raw_fastq"]:
        for key in ["raw_fastq1", "raw_fastq2"]:
            unlink_if(result.get(key))
    if not save["trimmed_fastq"]:
        for key in ["trimmed_fastq1", "trimmed_fastq2"]:
            unlink_if(result.get(key))


def cleanup_replicate_outputs(rep_result: dict[str, Any], data: dict[str, Any]) -> None:
    save = data["save"]
    if not save["peaks_raw"]:
        unlink_if(rep_result.get("raw_peaks"))
    if not save["peaks_filtered"]:
        print("\t\t  NOTE: keeping filtered peaks although save option is false; FRiP/IDR/consensus need them.")
    if not save["cpm_bigwig"]:
        unlink_if(rep_result.get("cpm_bigwig"))
    if not save["cutsite_bigwig"]:
        unlink_if(rep_result.get("cutsite_bigwig"))
    if not save["subsampled_bam"]:
        sub_bam = rep_result.get("fixed_depth", {}).get("subsampled_bam")
        unlink_if(sub_bam)
        if sub_bam:
            unlink_if(f"{sub_bam}.bai")


def cleanup_final_bam_files(replicate_results: list[dict[str, Any]], data: dict[str, Any]) -> None:
    save = data["save"]
    for rep_result in replicate_results:
        sample_result = rep_result["sample"]
        if not save["sorted_bam"]:
            for key in ["sorted_bam", "sorted_bai"]:
                unlink_if(sample_result.get(key))
        if not save["filtered_bam"]:
            for key in ["filtered_bam", "filtered_bai"]:
                unlink_if(sample_result.get(key))
        if not save["shifted_bam"]:
            shifted = rep_result.get("shifted_bam")
            unlink_if(shifted)
            if shifted:
                unlink_if(f"{shifted}.bai")


def cleanup_tmp_dir(data: dict[str, Any]) -> None:
    tmp_dir = Path(data["experiment_dir"]) / "tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------------------------------
# Per-sample / per-replicate processing
# ----------------------------------------------------------------------------------------------

def process_sample(
    data: dict[str, Any],
    rep: dict[str, Any],
    download_futures: dict[tuple[str, str], Future[tuple[Path, Path | None]]] | None = None,
) -> dict[str, Any]:
    experiment_id = str(data["experiment_id"])
    experiment_dir = Path(data["experiment_dir"])
    params = data["params"]
    save = data["save"]
    thresholds = data["qc_thresholds"]

    rep_id = str(rep["rep_id"])
    label = sample_label(experiment_id, rep_id)
    sample = rep["sample"]
    layout = get_sample_layout(sample)
    threads = int(params["threads"])

    future = download_futures.get((experiment_id, rep_id)) if download_futures else None
    if future:
        print(f"{timestamp()}  Waiting for FASTQ download for {label} if needed ..")
        fq1, fq2 = future.result()
    else:
        fq1, fq2 = resolve_fastqs(sample, layout, label, experiment_dir, threads)
    layout = get_sample_layout(sample)   # AUTO (GEO) is replaced by the SRA layout during download

    raw_fastqc_dir = run_fastqc(fq1, fq2, layout, label, experiment_dir, threads, "raw") if save["fastqc"] else None
    t1, t2, fastp_json, fastp_html = trim_reads(fq1, fq2, layout, label, experiment_dir, params)
    trimmed_fastqc_dir = run_fastqc(t1, t2, layout, label, experiment_dir, threads, "trimmed") if save["fastqc"] else None
    read_stats = parse_fastp_json(fastp_json)
    if read_stats["read_length_raw"]:
        print(f"\t\t  Mean read length: {read_stats['read_length_raw']} bp raw, {read_stats['read_length_trimmed']} bp after trimming")

    sorted_bam, sorted_bai, bowtie2_log = align_bowtie2(t1, t2, layout, label, data)
    align_stats = parse_bowtie2_log(bowtie2_log, layout)
    idxstats_txt, organelle = samtools_idxstats(sorted_bam, label, data)

    filtered_bam, filtered_bai, markdup_stats, pbc = filter_and_dedup(sorted_bam, layout, label, data)
    library_complexity = compute_library_complexity(markdup_stats, thresholds)
    flagstat_txt = samtools_flagstat(filtered_bam, label, data)
    usable = count_usable_fragments(filtered_bam, layout)
    print(f"\t\t  Usable fragments after all filters: {usable}")

    fragsize = fragment_size_distribution(filtered_bam, label, data) if layout == "PE" else {}

    result = {
        "label": label,
        "rep_id": rep_id,
        "layout": layout,
        "accession": sample_accession_text(sample),
        "raw_fastq1": fq1,
        "raw_fastq2": fq2,
        "trimmed_fastq1": t1,
        "trimmed_fastq2": t2,
        "fastp_json": fastp_json,
        "fastp_html": fastp_html,
        "fastqc_raw": raw_fastqc_dir,
        "fastqc_trimmed": trimmed_fastqc_dir,
        "sorted_bam": sorted_bam,
        "sorted_bai": sorted_bai,
        "bowtie2_log": bowtie2_log,
        "idxstats": idxstats_txt,
        "filtered_bam": filtered_bam,
        "filtered_bai": filtered_bai,
        "markdup_stats": markdup_stats,
        "flagstat": flagstat_txt,
        "read_stats": read_stats,
        "align_stats": align_stats,
        "organelle": organelle,
        "library_complexity": library_complexity,
        "pbc": pbc,
        "usable_fragments": usable,
        "fragsize": fragsize,
    }

    cleanup_sample_files(result, sample, save)
    return result


def process_replicate(
    data: dict[str, Any],
    rep: dict[str, Any],
    download_futures: dict[tuple[str, str], Future[tuple[Path, Path | None]]] | None = None,
) -> dict[str, Any]:
    rep_id = str(rep["rep_id"])
    experiment_dir = Path(data["experiment_dir"])

    print(f"{timestamp()}  Processing replicate {rep_id}")
    sample_out = process_sample(data, rep, download_futures)
    check_mask_bed(data, Path(sample_out["filtered_bam"]))
    label = sample_out["label"]
    layout = sample_out["layout"]
    filtered_bam = Path(sample_out["filtered_bam"])

    peaks_dir = experiment_dir / "peaks"
    raw_peaks, summits = call_peaks_macs3(filtered_bam, label, layout, data, peaks_dir)
    filtered_peaks = greenscreen_filter(raw_peaks, label, data, peaks_dir)

    cpm_bw = bam_to_bigwig(filtered_bam, label, data, layout)
    shifted_bam = tn5_shift_bam(filtered_bam, label, layout, data)
    cutsite_bw = cutsite_bigwig(shifted_bam if shifted_bam else filtered_bam, label, data)

    tss_bed = prepare_tss_bed(data, filtered_bam)
    tss = tss_enrichment(cutsite_bw, tss_bed, label, data) if tss_bed else {"tss_enrichment": None, "tss_background": None, "tss_center": None}

    frip = compute_frip(filtered_bam, filtered_peaks, experiment_dir / "qc" / "frip" / f"{label}.frip.tsv", layout)
    fixed_depth = fixed_depth_metrics(filtered_bam, layout, label, int(sample_out["usable_fragments"]), data)
    fingerprint_pdf, fingerprint_tsv = plot_fingerprint(filtered_bam, label, data)
    ataqv_json = run_ataqv(filtered_bam, filtered_peaks, tss_bed, label, data)

    result = {
        "rep_id": rep_id,
        "sample": sample_out,
        "raw_peaks": raw_peaks,
        "summits": summits,
        "filtered_peaks": filtered_peaks,
        "raw_peak_count": count_lines(raw_peaks),
        "filtered_peak_count": count_lines(filtered_peaks),
        "cpm_bigwig": cpm_bw,
        "shifted_bam": shifted_bam,
        "cutsite_bigwig": cutsite_bw,
        "tss": tss,
        "frip": frip,
        "fixed_depth": fixed_depth,
        "fingerprint_pdf": fingerprint_pdf,
        "fingerprint_tsv": fingerprint_tsv,
        "ataqv_json": ataqv_json,
    }
    result["metrics"] = collect_metrics(data, result)

    cleanup_replicate_outputs(result, data)
    return result


def collect_metrics(data: dict[str, Any], rep_result: dict[str, Any]) -> dict[str, Any]:
    """Flat per-sample metrics dictionary (the row of the QC tables)."""

    s = rep_result["sample"]
    lc = s["library_complexity"]
    fs = s.get("fragsize", {}) or {}
    fd = rep_result["fixed_depth"]
    thresholds = data["qc_thresholds"]

    metrics: dict[str, Any] = {
        "sample": s["label"],
        "experiment_id": str(data["experiment_id"]),
        "sample_id": str(data["sample_id"]),
        "rep_id": rep_result["rep_id"],
        "layout": s["layout"],
        "accession": s["accession"],
        **{k: s["read_stats"].get(k) for k in ("read_length_raw", "read_length_trimmed", "raw_reads", "reads_after_trimming", "trimming_loss_fraction")},
        **{k: s["align_stats"].get(k) for k in ("alignment_rate", "unique_alignment_fraction", "multi_alignment_fraction", "discordant_fraction")},
        "mapped_reads_unfiltered": s["organelle"]["mapped_reads_unfiltered"],
        "organelle_fraction": s["organelle"]["organelle_fraction"],
        "nuclear_fraction": s["organelle"]["nuclear_fraction"],
        "organelle_reads_per_contig": s["organelle"]["organelle_reads_per_contig"],
        "reads_after_mapq_whitelist": lc["total_reads"],
        "duplicate_rate": lc["duplicate_rate"],
        "NRF": lc["NRF"],
        "PBC1": s["pbc"]["PBC1"],
        "PBC2": s["pbc"]["PBC2"],
        "estimated_library_size": lc["estimated_library_size"],
        "usable_fragments": s["usable_fragments"],
        "median_fragment_size": fs.get("median_fragment_size"),
        "nfr_fraction": fs.get("nfr_fraction"),
        "mono_fraction": fs.get("mono_fraction"),
        "di_fraction": fs.get("di_fraction"),
        "long_fraction": fs.get("long_fraction"),
        "nfr_mono_ratio": fs.get("nfr_mono_ratio"),
        "tss_enrichment": rep_result["tss"].get("tss_enrichment"),
        "tss_source": data.get("_tss_source", "none"),
        "peaks_raw": rep_result["raw_peak_count"],
        "peaks_filtered": rep_result["filtered_peak_count"],
        "FRiP": rep_result["frip"]["FRiP"],
        "fixed_depth_applied": fd["fixed_depth_applied"],
        "fixed_depth_fragments": fd["fixed_depth_fragments"] if fd["fixed_depth_applied"] else s["usable_fragments"],
        "peaks_fixed_depth": fd["peaks_fixed_depth"] if fd["fixed_depth_applied"] else rep_result["filtered_peak_count"],
        "FRiP_fixed_depth": fd["FRiP_fixed_depth"] if fd["fixed_depth_applied"] else rep_result["frip"]["FRiP"],
        "greenscreen": data.get("_mask_status", "none"),
        "metadata": dict(data.get("metadata", {})),
        "qc_thresholds": thresholds,
        "qc_thresholds_custom": dict(data.get("_qc_thresholds_custom", {})),
    }
    metrics["status"] = status_summary(metrics, thresholds)
    return metrics


def status_summary(metrics: dict[str, Any], thresholds: dict[str, Any]) -> dict[str, Any]:
    """Per-metric tiers and an overall status (worst tier over the tiered metrics)."""

    tiers: dict[str, str] = {}
    for metric in TIER_METRICS:
        tiers[metric] = tier_status(metric, metrics.get(metric), thresholds)
    return {"tiers": tiers, **overall_from_tiers(tiers)}


def overall_from_tiers(tiers: dict[str, str]) -> dict[str, Any]:
    """Overall status = worst tier over OVERALL_METRICS (FRiP below the reference counts as "ok");
    "flagged" lists every metric rated poor and FRiP below the reference."""

    order = {"poor": 0, "ok": 1, "good": 2}
    core = ["ok" if tiers.get(m) == "below_reference" else tiers.get(m, "NA") for m in OVERALL_METRICS]
    rated = [t for t in core if t in order]
    overall = min(rated, key=lambda t: order[t]) if rated else "NA"
    flagged = sorted(m if t == "poor" else f"{m}:below_reference" for m, t in tiers.items() if t in ("poor", "below_reference"))
    return {"overall": overall, "flagged": flagged}


# ----------------------------------------------------------------------------------------------
# Reports
# ----------------------------------------------------------------------------------------------

def write_mqc_table(path: Path, rows: list[dict[str, Any]], section_id: str, section_name: str, description: str) -> None:
    columns = [c for c in METRIC_COLUMNS if c != "sample"] + ["status"]
    with path.open("w", encoding="utf-8") as out:
        out.write(f"# id: '{section_id}'\n")
        out.write("# plot_type: 'table'\n")
        out.write(f"# section_name: '{section_name}'\n")
        out.write(f"# description: '{description}'\n")
        out.write("# headers:\n")
        for col in ("alignment_rate", "organelle_fraction", "nuclear_fraction", "duplicate_rate", "NRF", "PBC1", "nfr_fraction", "mono_fraction", "FRiP", "FRiP_fixed_depth"):
            out.write(f"#   {col}:\n#     format: '{{:.3f}}'\n")
        out.write("sample\t" + "\t".join(columns) + "\n")
        for row in rows:
            values = [fmt(row.get(col)) if col != "status" else row["status"]["overall"] for col in columns]
            out.write(f"{row['sample']}\t" + "\t".join(values) + "\n")


def write_fragment_size_mqc(path: Path, replicate_results: list[dict[str, Any]], fragment_max: int) -> None:
    rows = []
    for r in replicate_results:
        fs = r["sample"].get("fragsize") or {}
        hist = fs.get("histogram") or {}
        total = sum(hist.values())
        if total:
            rows.append((r["sample"]["label"], [hist.get(i, 0) / total for i in range(1, fragment_max + 1)]))
    if not rows:
        return
    with path.open("w", encoding="utf-8") as out:
        out.write("# id: 'atac_fragment_sizes'\n")
        out.write("# plot_type: 'linegraph'\n")
        out.write("# section_name: 'ATAC fragment size distribution'\n")
        out.write("# description: 'Fraction of proper-pair fragments per insert size (filtered, deduplicated BAM).'\n")
        out.write("# pconfig:\n#     xlab: 'Fragment length (bp)'\n#     ylab: 'Fraction of fragments'\n")
        out.write("sample\t" + "\t".join(str(i) for i in range(1, fragment_max + 1)) + "\n")
        for label, values in rows:
            out.write(label + "\t" + "\t".join(f"{v:.6g}" for v in values) + "\n")


def write_summary(
    data: dict[str, Any],
    replicate_results: list[dict[str, Any]],
    consensus: Path,
    consensus_summits: Path,
    idr_peaks: Path | None,
    pooled: dict[str, Any] | None = None,
) -> Path:
    experiment_id = str(data["experiment_id"])
    pooled = pooled or {}
    reports_dir = Path(data["experiment_dir"]) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary = reports_dir / f"{experiment_id}.summary.txt"

    metrics_rows = [r["metrics"] for r in replicate_results]
    for row in metrics_rows:
        with (reports_dir / f"{row['sample']}.metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(row, fh, indent=2, default=str)

    frips = [r["frip"]["FRiP"] for r in replicate_results]
    peak_counts = [r["filtered_peak_count"] for r in replicate_results]

    with summary.open("w", encoding="utf-8") as fh:
        fh.write(f"experiment_id\t{experiment_id}\n")
        fh.write(f"sample_id\t{data['sample_id']}\n")
        fh.write(f"layout\t{','.join(sorted({r['sample']['layout'] for r in replicate_results}))}\n")
        fh.write(f"genome_index\t{data['reference']['genome_index']}\n")
        fh.write(f"genome_size\t{data['reference']['genome_size']}\n")
        fh.write(f"replicates\t{len(replicate_results)}\n")
        fh.write("control_used\tno (ATAC-seq)\n")
        fh.write(f"greenscreen\t{data.get('_mask_status', 'none')}\t{data.get('_mask_bed_effective') or '-'}\n")
        for issue in data.get("_issues", []):
            fh.write(f"issue\t{issue}\n")
        fh.write(f"consensus_peaks\t{consensus}\n")
        fh.write(f"consensus_summits\t{consensus_summits}\n")
        fh.write(f"idr_peaks\t{idr_peaks if idr_peaks else 'not_run'}\n")
        fh.write(f"pooled_fragments\t{pooled.get('pooled_fragments', 'NA')}\n")
        fh.write(f"pooled_cpm_bigwig\t{pooled.get('pooled_cpm_bigwig', 'NA')}\n")
        fh.write(f"pooled_cutsite_bigwig\t{pooled.get('pooled_cutsite_bigwig', 'NA')}\n")
        fh.write(f"pooled_peaks\t{pooled.get('pooled_peaks', 'not_run')}\n")
        fh.write(f"pooled_peak_count\t{pooled.get('pooled_peak_count', 'NA')}\n")
        fh.write(f"pooled_FRiP\t{fmt(pooled.get('pooled_FRiP'))}\n")
        for suffix, path in (pooled.get("signal_tracks") or {}).items():
            fh.write(f"macs_signal_{suffix}\t{path}\n")
        for rep_id, frac in (pooled.get("pooled_support") or {}).items():
            fh.write(f"pooled_peaks_supported_by_{rep_id}\t{fmt(frac)}\n")
        fh.write(f"mean_FRiP\t{mean(frips):.6f}\n")
        fh.write(f"sd_FRiP\t{stdev(frips):.6f}\n" if len(frips) > 1 else "sd_FRiP\tNA\n")
        fh.write(f"mean_filtered_peak_count\t{mean(peak_counts):.2f}\n")
        fh.write(f"sd_filtered_peak_count\t{stdev(peak_counts):.2f}\n" if len(peak_counts) > 1 else "sd_filtered_peak_count\tNA\n")
        for key, value in data.get("metadata", {}).items():
            fh.write(f"metadata.{key}\t{value}\n")
        fh.write("\nper_replicate\n")
        fh.write("sample\t" + "\t".join(c for c in METRIC_COLUMNS if c != "sample") + "\tstatus\tflagged\n")
        for row in metrics_rows:
            values = [fmt(row.get(c)) for c in METRIC_COLUMNS if c != "sample"]
            fh.write(f"{row['sample']}\t" + "\t".join(values) + f"\t{row['status']['overall']}\t{','.join(row['status']['flagged']) or '-'}\n")

    write_mqc_table(
        reports_dir / "atac_qc_table_mqc.tsv", metrics_rows, "atac_qc_table", "ATAC-seq QC table",
        "Per-sample ATAC-seq metrics (organelle fraction from the unfiltered BAM; NRF from samtools markdup; PBC1/PBC2 ENCODE definition; TSS enrichment ENCODE-style; fixed-depth peaks/FRiP on a subsample).",
    )
    write_fragment_size_mqc(reports_dir / "atac_fragment_sizes_mqc.tsv", replicate_results, int(data["params"]["fragment_max"]))

    print(f"\t\t  Summary: {summary}")
    return summary


def run_multiqc(target_dir: Path, out_dir: Path, title: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{timestamp()}  Running MultiQC: {title}")
    cmd = " ".join([
        "multiqc",
        shlex.quote(str(target_dir)),
        "-o", shlex.quote(str(out_dir)),
        "--title", shlex.quote(title),
        "-f",
    ])
    run_command(cmd, out_dir / "multiqc.log")
    report = next(out_dir.glob("*multiqc_report.html"), out_dir / "multiqc_report.html")
    print(f"\t\t  MultiQC report: {report}")
    return report


# ----------------------------------------------------------------------------------------------
# Experiment driver
# ----------------------------------------------------------------------------------------------

def run_experiment(
    data: dict[str, Any],
    download_executor: ThreadPoolExecutor,
    download_futures: dict[tuple[str, str], Future[tuple[Path, Path | None]]] | None = None,
    next_data: dict[str, Any] | None = None,
) -> dict[tuple[str, str], Future[tuple[Path, Path | None]]]:
    start = datetime.now()
    experiment_dir = Path(data["experiment_dir"])

    print(f"\n{'=' * 70}")
    print(f"  ATAC-seq pipeline | experiment: {data['experiment_id']}")
    print(f"  layout: {data['layout']} | replicates: {len(data['replicates'])}")
    print(f"{'=' * 70}\n")

    download_futures = start_downloads(data, download_executor, download_futures)
    next_download_futures = start_downloads(next_data, download_executor, limit=2) if next_data else {}

    replicate_results = [process_replicate(data, rep, download_futures) for rep in data["replicates"]]
    peak_files = [r["filtered_peaks"] for r in replicate_results]

    idr_peaks = run_idr(peak_files, data)
    consensus_source = idr_peaks if idr_peaks else peak_files[0]
    consensus, consensus_summits = make_consensus_peaks(consensus_source, data)
    pooled = make_pooled_outputs(data, replicate_results)
    write_summary(data, replicate_results, consensus, consensus_summits, idr_peaks, pooled)

    cleanup_final_bam_files(replicate_results, data)
    cleanup_tmp_dir(data)

    if data["save"].get("qc_reports", True):
        run_multiqc(experiment_dir, experiment_dir / "multiqc", f"ATAC-seq {data['experiment_id']}")

    elapsed = datetime.now() - start
    issues = data.get("_issues", [])
    if issues:
        print(f"\n{'!' * 8} {len(issues)} issue(s) in {data['experiment_id']} (also listed in the summary file):")
        for issue in issues:
            print(f"{'!' * 8}   {issue}")
    print(f"\n{timestamp()}  Experiment complete: {data['experiment_id']} ({elapsed})")
    return {k: v for k, v in next_download_futures.items() if k not in download_futures}


# ----------------------------------------------------------------------------------------------
# Batch level: QC summary over all experiments, peak overlap, viewer, MultiQC
# ----------------------------------------------------------------------------------------------

def percentile_ranks(values: list[tuple[str, float]], higher_is_better: bool = True) -> dict[str, float]:
    """Cohort percentile (0-100) of each sample for one metric; 100 = best in the cohort."""

    valid = [(s, v) for s, v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    n = len(valid)
    if n == 0:
        return {}
    ranks: dict[str, float] = {}
    for sample, value in valid:
        worse = sum(1 for _, other in valid if (other < value if higher_is_better else other > value))
        ties = sum(1 for _, other in valid if other == value) - 1
        ranks[sample] = 100.0 * (worse + 0.5 * ties) / (n - 1) if n > 1 else 100.0
    return ranks


def batch_qc_summary(outdir: Path) -> Path | None:
    metric_files = sorted(outdir.glob("*/reports/*.metrics.json"))
    if not metric_files:
        print(f"{timestamp()}  Batch summary: no metrics files found under {outdir}.")
        return None

    rows: list[dict[str, Any]] = []
    for path in metric_files:
        try:
            with path.open("r", encoding="utf-8") as fh:
                rows.append(json.load(fh))
        except (OSError, ValueError) as exc:
            print(f"\t\t  WARNING: could not read {path}: {exc}")

    for r in rows:   # re-rate with the thresholds currently in the script (+ per-config overrides)
        r["status"] = status_summary(r, merge_thresholds(r.get("qc_thresholds_custom")))

    rank_specs = [
        ("tss_enrichment", True), ("FRiP_fixed_depth", True), ("usable_fragments", True),
        ("organelle_fraction", False), ("NRF", True), ("PBC1", True),
    ]
    if len(rows) >= COHORT_MIN_SAMPLES:
        ranks = {metric: percentile_ranks([(r["sample"], r.get(metric)) for r in rows], better) for metric, better in rank_specs}
    else:
        ranks = {metric: {} for metric, _ in rank_specs}
        print(f"{timestamp()}  NOTE: {len(rows)} sample(s) < {COHORT_MIN_SAMPLES}; cohort percentiles are not reported.")
    meta_keys = sorted({k for r in rows for k in (r.get("metadata") or {}).keys()})
    tier_metrics = TIER_METRICS

    summary = outdir / "atac_qc_summary.tsv"
    header = (
        METRIC_COLUMNS
        + [f"metadata.{k}" for k in meta_keys]
        + [f"tier.{m}" for m in tier_metrics]
        + ["status", "flagged"]
        + [f"cohort_pct.{m}" for m, _ in rank_specs]
    )
    order = {"good": 0, "ok": 1, "poor": 2, "NA": 3}
    rows.sort(key=lambda r: (order.get(r["status"]["overall"], 3), -(r.get("tss_enrichment") or 0.0)))
    with summary.open("w", encoding="utf-8") as out:
        out.write("\t".join(header) + "\n")
        for r in rows:
            values = [fmt(r.get(c)) for c in METRIC_COLUMNS]
            values += [str((r.get("metadata") or {}).get(k, "")) for k in meta_keys]
            values += [r["status"]["tiers"].get(m, "NA") for m in tier_metrics]
            values += [r["status"]["overall"], ",".join(r["status"]["flagged"]) or "-"]
            values += [fmt(ranks[m].get(r["sample"]), 1) for m, _ in rank_specs]
            out.write("\t".join(values) + "\n")

    print(f"{timestamp()}  Batch QC summary ({len(rows)} samples): {summary}")
    good = sum(1 for r in rows if r["status"]["overall"] == "good")
    ok = sum(1 for r in rows if r["status"]["overall"] == "ok")
    poor = sum(1 for r in rows if r["status"]["overall"] == "poor")
    print(f"\t\t  status: {good} good, {ok} ok, {poor} poor")
    return summary


def batch_peak_overlap(outdir: Path) -> None:
    """Pairwise Jaccard overlap of consensus peak sets and a merged peak set across experiments."""

    consensus_files = sorted(outdir.glob("*/consensus/*.consensus.narrowPeak"))
    consensus_files = [p for p in consensus_files if p.stat().st_size > 0]
    if len(consensus_files) < 2:
        return

    batch_dir = outdir / "consensus_all"
    tmp_dir = batch_dir / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    sorted_files: dict[str, Path] = {}
    for path in consensus_files:
        exp_id = path.parent.parent.name
        sorted_bed = tmp_dir / f"{exp_id}.sorted.bed"
        run_command(f"cut -f1-3 {shlex.quote(str(path))} | sort -k1,1 -k2,2n > {shlex.quote(str(sorted_bed))}")
        sorted_files[exp_id] = sorted_bed

    print(f"{timestamp()}  Peak overlap matrix over {len(sorted_files)} experiments ..")
    ids = sorted(sorted_files)
    matrix: dict[tuple[str, str], float] = {}
    for a, b in itertools.combinations(ids, 2):
        try:
            out = run_capture(f"bedtools jaccard -a {shlex.quote(str(sorted_files[a]))} -b {shlex.quote(str(sorted_files[b]))}")
            fields = out.splitlines()[-1].split("\t")
            matrix[(a, b)] = matrix[(b, a)] = float(fields[2])
        except (RuntimeError, IndexError, ValueError) as exc:
            print(f"\t\t  WARNING: jaccard failed for {a} vs {b}: {str(exc).splitlines()[0]}")
            matrix[(a, b)] = matrix[(b, a)] = float("nan")

    with (batch_dir / "peak_jaccard_matrix.tsv").open("w", encoding="utf-8") as fh:
        fh.write("experiment\t" + "\t".join(ids) + "\n")
        for a in ids:
            fh.write(a + "\t" + "\t".join("1.0000" if a == b else fmt(matrix.get((a, b)), 4) for b in ids) + "\n")

    run_command(
        f"cat {' '.join(shlex.quote(str(p)) for p in sorted_files.values())} | sort -k1,1 -k2,2n "
        f"| bedtools merge -i stdin > {shlex.quote(str(batch_dir / 'merged_peaks.bed'))}"
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\t\t  Jaccard matrix and merged peak set: {batch_dir}")


def batch_ataqv_viewer(outdir: Path) -> None:
    json_files = sorted(outdir.glob("*/qc/ataqv/*.ataqv.json.gz"))
    if not json_files or shutil.which("mkarv") is None:
        return
    viewer_dir = outdir / "ataqv_viewer"
    print(f"{timestamp()}  Building ataqv viewer for {len(json_files)} samples ..")
    try:
        run_command(
            f"mkarv --force {shlex.quote(str(viewer_dir))} {' '.join(shlex.quote(str(p)) for p in json_files)}",
            outdir / "ataqv_viewer.mkarv.log",
        )
        print(f"\t\t  ataqv viewer: {viewer_dir}/index.html")
    except RuntimeError as exc:
        print(f"\t\t  WARNING: mkarv failed: {str(exc).splitlines()[0]}")


def run_batch_summary(outdir: Path, with_multiqc: bool = True) -> None:
    print(f"\n{'=' * 70}\n  Batch summary over {outdir}\n{'=' * 70}\n")
    batch_qc_summary(outdir)
    batch_peak_overlap(outdir)
    batch_ataqv_viewer(outdir)
    if with_multiqc:
        try:
            run_multiqc(outdir, outdir / "multiqc_all", "ATAC-seq all experiments")
        except RuntimeError as exc:
            print(f"\t\t  WARNING: batch MultiQC failed: {str(exc).splitlines()[0]}")


# ----------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ATAC-seq batch pipeline from YAML configs.")
    parser.add_argument("--config-dir", type=Path, help="Folder with .yaml/.yml experiment configs.")
    parser.add_argument("--outdir", default=Path("results"), type=Path, help="Batch output folder.")
    parser.add_argument("--threads", type=int, default=None, help="Override params.threads of all configs (e.g. $SLURM_CPUS_PER_TASK).")
    parser.add_argument("--summary-only", action="store_true", help="Only rebuild the batch-level summary from existing results.")
    parser.add_argument("--no-batch-multiqc", action="store_true", help="Skip the MultiQC report over all experiments.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip experiments that already have reports/<experiment_id>.summary.txt in --outdir (batch summary still covers them).")
    args = parser.parse_args()
    if not args.summary_only and args.config_dir is None:
        parser.error("--config-dir is required unless --summary-only is given.")
    return args


def main() -> None:
    args = parse_args()

    if args.summary_only:
        run_batch_summary(args.outdir, with_multiqc=not args.no_batch_multiqc)
        return

    try:
        configs = load_configs(args.config_dir, args.outdir)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    if args.threads:
        for cfg in configs:
            cfg["params"]["threads"] = int(args.threads)

    print(f"\nParsed {len(configs)} config(s) successfully.")

    if args.skip_existing:
        remaining = []
        for cfg in configs:
            summary = Path(cfg["experiment_dir"]) / "reports" / f"{cfg['experiment_id']}.summary.txt"
            if summary.exists():
                print(f"{timestamp()}  Skipping {cfg['experiment_id']}: already processed ({summary}).")
            else:
                remaining.append(cfg)
        configs = remaining
        if not configs:
            print(f"{timestamp()}  Nothing to process; rebuilding the batch summary only.")
            run_batch_summary(args.outdir, with_multiqc=not args.no_batch_multiqc)
            return

    try:
        download_workers = max(int(cfg["params"]["download_workers"]) for cfg in configs)
        preloaded_futures: dict[tuple[str, str], Future[tuple[Path, Path | None]]] = {}
        with ThreadPoolExecutor(max_workers=download_workers) as download_executor:
            for idx, data in enumerate(configs):
                next_data = configs[idx + 1] if idx + 1 < len(configs) else None
                preloaded_futures = run_experiment(data, download_executor, preloaded_futures, next_data)
    except (RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    run_batch_summary(args.outdir, with_multiqc=not args.no_batch_multiqc)


if __name__ == "__main__":
    main()