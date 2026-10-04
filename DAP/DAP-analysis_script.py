#!/usr/bin/env python3
"""
DAP-seq / ampDAP-seq batch pipeline for Arabidopsis thaliana TAIR10.1.

Independent sister script of ChIP-analysis_script.py. DAP-seq incubates an
in-vitro-produced TF with a genomic DNA library; native DNA methylation is
retained in DAP and removed by pre-assay amplification in ampDAP.
Each experiment declares assay, TF, gDNA library and control_type explicitly.
Supported controls: halo_mock, input_library, none. Controls are reused within
an experiment only. The `ip` YAML key and output names remain compatible.
Reference paths are fixed once in DEFAULT_REFERENCE, not in dataset configs.

Pipeline scope:
  1.  Read and validate YAML experiment configs
  2.  Create experiment output folders
  3.  Resolve/download FASTQ files for all IP and control samples
  4.  Run FastQC before trimming
  5.  Trim reads with fastp
  6.  Run FastQC after trimming
  7.  Align reads to reference genome with Bowtie2 + samtools
  8.  Filter low-quality reads and remove PCR duplicates with samtools
  9.  Call peaks with MACS3, with matched control(s) or explicitly without control
  10. Remove artefact / blacklist regions with bedtools
  11. Create CPM-normalised and fold-enrichment bigWig tracks
  12. Compute FRiP
  13. Compute NSC/RSC strand cross-correlation
  14. Run IDR for replicate pairs; skip for a single replicate
  15. Create consensus peak set
  16. Write per-experiment summary
  17. Run one MultiQC report per experiment

Pooled signal tracks also run for a single replicate. Without a control, CPM,
fingerprint, MACS3 FE and p-value tracks remain available; bamCompare log2 tracks
are explicitly skipped. QC metrics do not veto samples.
Usage:
  python DAP-analysis_script.py --config-dir configs --outdir results
  python DAP-analysis_script.py --config-dir configs --outdir results --validate-only
Requires Python >=3.10, PyYAML and the same command-line tools as the ChIP script.
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
ALLOWED_SOURCES = {"local", "sra", "ena", "url", "zenodo"}
ALLOWED_ASSAYS = {"dap", "ampdap"}
ALLOWED_CONTROL_TYPES = {"halo_mock", "input_library", "none"}
CONFIG_SUFFIXES = {".yaml", ".yml"}
DOWNLOAD_RETRY_DELAYS = [30, 60, 120, 300]

# Paths shared with the current ATAC script. Change these once here when needed.
DEFAULT_REFERENCE = {
    "genome_index": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/general_data/genome/TAIR10.1.atlas",
    "genome_size": "1.19e8",
    "mask_bed": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/general_data/masks/arabidopsis_greenscreen_20inputs.bed",
    "genome_fasta": "/data/projects/SFB_A03/jan/AT_TFBS/raw_data/general_data/genome/TAIR10.1.atlas.fa",
}

DEFAULT_PARAMS = {
    "threads": 6,
    "min_mapq": 30,
    "macs_qvalue": 0.01,
    "macs_extsize": 200,  # Configurable last resort; measured SPP values take priority.
    "idr_threshold": 0.05,
    "download_workers": 1,
    "macs_signal_subsample_control": True,
    "keep_chroms": ["1", "2", "3", "4", "5"],
    "organelle_chroms": ["Mt", "Pt"],
    "nrf_thresholds": None,  # Raw NRF; optional user-defined {good_min, moderate_min}.
    "peak_count_notice_max": 60000,  # Informational only; no automatic exclusion.
    "macs_qvalue_no_control": None,  # None inherits macs_qvalue; no hidden stricter cutoff.
    "macs_signal_normalize_chrom_names": True,
    "fastp_extra": "--dont_eval_duplication --detect_adapter_for_pe --cut_front --cut_tail --cut_mean_quality 20 --length_required 20",
}

DEFAULT_SAVE = {
    "raw_fastq": False,
    "trimmed_fastq": False,
    "fastqc": True,
    "sorted_bam": False,
    "filtered_bam": False,
    "control_filtered_bam": False,
    "cpm_bigwig_ip": True,
    "cpm_bigwig_control": True,
    "fold_enrichment_bigwig": True,
    "pooled_log2_bigwig": True,
    "pooled_macs3_fe_bigwig": True,
    "peaks_raw": True,
    "peaks_filtered": True,
    "logs": True,
    "qc_reports": True,
}


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


def compute_library_complexity(markdup_stats: Path, thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    stats = parse_markdup_stats(markdup_stats)
    total = stats.get("READ", 0)
    nonredundant = stats.get("WRITTEN", 0)
    duplicates = stats.get("DUPLICATE TOTAL", max(total - nonredundant, 0))
    nrf = nonredundant / total if total > 0 else 0.0

    status = "not_interpreted"
    if thresholds is not None and total > 0:
        if nrf >= float(thresholds["good_min"]):
            status = "good"
        elif nrf >= float(thresholds["moderate_min"]):
            status = "moderate"
        else:
            status = "poor"

    return {
        "total_reads": total,
        "nonredundant_reads": nonredundant,
        "duplicate_reads": duplicates,
        "duplicate_rate": duplicates / total if total > 0 else 0.0,
        "NRF": nrf,
        "estimated_library_size": stats.get("ESTIMATED_LIBRARY_SIZE", 0),
        "status": status,
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


def validate_sample(sample: dict[str, Any], default_layout: str, where: str) -> None:
    layout = str(sample.get("layout", default_layout)).upper()
    if layout not in ALLOWED_LAYOUTS:
        raise ValueError(f"'{where}.layout' must be one of {sorted(ALLOWED_LAYOUTS)}, got {layout!r}.")
    sample["layout"] = layout

    source = str(require_key(sample, "source", where)).lower()
    if source not in ALLOWED_SOURCES:
        raise ValueError(f"'{where}.source' must be one of {sorted(ALLOWED_SOURCES)}, got {source!r}.")
    sample["source"] = source

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
    else:
        if not fastq_url_1:
            raise ValueError(f"'{where}.fastq_url_1' is required when source={source!r}.")
        if layout == "PE" and not fastq_url_2:
            raise ValueError(f"'{where}.fastq_url_2' is required for PE {source!r} input.")


def control_samples(rep: dict[str, Any]) -> list[dict[str, Any]]:
    controls = rep.get("controls")
    if controls is not None:
        return controls
    raw = rep.get("control")
    return raw if isinstance(raw, list) else ([raw] if raw else [])


def get_sample_layout(sample: dict[str, Any]) -> str:
    return str(sample["layout"]).upper()


def control_label_role(controls: list[dict[str, Any]], index: int) -> str:
    return "control" if len(controls) == 1 else f"control{index + 1}"


def merge_defaults(data: dict[str, Any]) -> None:
    if "reference" in data:
        raise ValueError("Reference is fixed in DEFAULT_REFERENCE; remove 'reference' from the dataset config.")
    data["reference"] = dict(DEFAULT_REFERENCE)
    data["params"] = {**DEFAULT_PARAMS, **require_mapping(data.get("params", {}), "params")}
    data["save"] = {**DEFAULT_SAVE, **require_mapping(data.get("save", {}), "save")}


def report_error(data: dict[str, Any], message: str) -> None:
    """Expose recoverable failures in the terminal and the experiment report."""
    print(f"{timestamp()}  Error: {message}")
    data.setdefault("issues", []).append(message)
    if data.get("experiment_dir"):
        path = Path(data["experiment_dir"]) / "reports" / "errors.tsv"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"{timestamp()}\t{message.replace(chr(9), ' ').replace(chr(10), ' ')}\n")


def check_identifier(value: Any, name: str) -> str:
    text = str(value).strip()
    if text in {"", ".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
        raise ValueError(f"'{name}' must be a filename-safe identifier (letters, numbers, '_', '-', '.').")
    return text


def sample_context(sample: dict[str, Any], data: dict[str, Any], where: str) -> None:
    """An omitted sample-level value inherits the declared experiment context."""
    for key in ("assay", "gdna_library"):
        actual = str(sample.get(key, data[key])).strip()
        if key == "assay":
            actual = actual.lower()
        if actual != data[key]:
            raise ValueError(f"'{where}.{key}'={actual!r} differs from experiment {data[key]!r}; split these datasets.")
        sample[key] = actual
    if sample.get("preparation_batch") and data.get("preparation_batch") not in {None, "unknown"}:
        if str(sample["preparation_batch"]) != data["preparation_batch"]:
            raise ValueError(f"'{where}.preparation_batch' differs from the experiment batch.")


def normalize_mask_bed(data: dict[str, Any]) -> Path:
    """Check the mandatory mask and normalize common TAIR/Ensembl aliases."""
    reference = data["reference"]
    original = Path(reference.get("mask_bed") or "")
    if not original.is_file() or original.stat().st_size == 0:
        report_error(data, f"Missing or empty Greenscreen mask: {original}")
        raise FileNotFoundError(f"A non-empty Greenscreen mask is required: {original}")
    keep = [str(chrom) for chrom in data["params"]["keep_chroms"]]
    aliases = {alias: chrom for chrom in keep for alias in chrom_aliases_for_target(chrom)}
    records: list[str] = []
    renamed = 0
    unknown: set[str] = set()
    with original.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            if not line.strip() or line.startswith(("#", "track", "browser")):
                continue
            fields = line.split()
            try:
                start, end = int(fields[1]), int(fields[2])
                if start < 0 or end <= start:
                    raise ValueError("invalid interval")
            except (ValueError, IndexError) as exc:
                report_error(data, f"Invalid mask BED record at {original}:{lineno}")
                raise ValueError(f"Invalid mask BED record at {original}:{lineno}") from exc
            target = aliases.get(fields[0])
            if target is None:
                unknown.add(fields[0])
                continue
            renamed += int(fields[0] != target)
            fields[0] = target
            records.append("\t".join(fields) + "\n")
    if not records:
        report_error(data, f"Mask chromosomes do not match keep_chroms={keep}; observed {sorted(unknown)}.")
        raise ValueError("No mask intervals match the retained chromosomes.")
    if unknown:
        print(f"{timestamp()}  Mask: ignoring contigs outside the whitelist: {sorted(unknown)}")
    if renamed or unknown:
        normalized = Path(data["experiment_dir"]) / "reports" / "greenscreen.normalized.bed"
        normalized.parent.mkdir(parents=True, exist_ok=True)
        normalized.write_text("".join(records), encoding="utf-8")
        reference["mask_bed_original"] = str(original)
        reference["mask_bed"] = str(normalized)
        print(f"{timestamp()}  Mask: normalized chromosome names in {renamed} intervals.")
        return normalized
    return original


def check_reference_files(data: dict[str, Any]) -> None:
    """Fail before downloading reads if reference files are unavailable."""
    reference = data["reference"]
    prefix = str(reference["genome_index"])
    parts = ("1", "2", "3", "4", "rev.1", "rev.2")
    if not any(all(Path(f"{prefix}.{part}.{suffix}").is_file() and Path(f"{prefix}.{part}.{suffix}").stat().st_size > 0 for part in parts) for suffix in ("bt2", "bt2l")):
        raise FileNotFoundError(f"Incomplete Bowtie2 index: {prefix} (edit DEFAULT_REFERENCE if necessary).")
    fasta = Path(reference["genome_fasta"])
    if not fasta.is_file() or fasta.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty reference FASTA: {fasta}")
    fai = Path(f"{fasta}.fai")
    if fai.is_file():
        contigs = {line.split("\t")[0] for line in fai.read_text().splitlines() if line.strip()}
    else:
        with fasta.open(encoding="utf-8") as fh:
            contigs = {line[1:].split()[0] for line in fh if line.startswith(">")}
    missing = set(map(str, data["params"]["keep_chroms"])) - contigs
    if missing:
        raise ValueError(f"Reference chromosome names do not match keep_chroms: missing {sorted(missing)}.")
    normalize_mask_bed(data)


def sample_outputs(rep_result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [("ip", rep_result["ip"]), *(("control", c) for c in rep_result.get("controls", []))]


def validate_config(data: dict[str, Any]) -> None:
    for key in ("experiment_id", "sample_id"):
        data[key] = check_identifier(require_key(data, key, "root"), key)
    for key, allowed in (("assay", ALLOWED_ASSAYS), ("control_type", ALLOWED_CONTROL_TYPES)):
        value = str(require_key(data, key, "root")).strip().lower()
        if value not in allowed:
            raise ValueError(f"'{key}' must be one of {sorted(allowed)}, got {value!r}.")
        data[key] = value
    for key in ("tf", "gdna_library"):
        data[key] = str(require_key(data, key, "root")).strip()
        if not data[key] or "\n" in data[key] or "\t" in data[key]:
            raise ValueError(f"'{key}' must be a non-empty one-line identifier.")
    data["tf_family"] = str(data.get("tf_family") or "unknown")
    data["preparation_batch"] = str(data.get("preparation_batch") or "unknown")
    if "gdna_metadata" in data:
        require_mapping(data["gdna_metadata"], "gdna_metadata")

    layout = str(require_key(data, "layout", "root")).upper()
    if layout not in ALLOWED_LAYOUTS:
        raise ValueError(f"'layout' must be one of {sorted(ALLOWED_LAYOUTS)}, got {layout!r}.")
    data["layout"] = layout

    reference = require_mapping(require_key(data, "reference", "root"), "reference")
    require_key(reference, "genome_index", "reference")
    require_key(reference, "genome_size", "reference")

    params = require_mapping(data["params"], "params")
    validate_number(params.get("threads"), "params.threads", int)
    validate_number(params.get("min_mapq"), "params.min_mapq", int)
    validate_number(params.get("macs_qvalue"), "params.macs_qvalue", float)
    if "macs_signal_pvalue" in params:
        validate_number(params.get("macs_signal_pvalue"), "params.macs_signal_pvalue", float)
    validate_number(params.get("macs_extsize"), "params.macs_extsize", int)
    validate_number(params.get("idr_threshold"), "params.idr_threshold", float)
    validate_number(params.get("download_workers"), "params.download_workers", int)
    validate_number(params.get("peak_count_notice_max"), "params.peak_count_notice_max", int)
    require_list(params.get("organelle_chroms"), "params.organelle_chroms")
    for key in ("macs_signal_subsample_control", "macs_signal_normalize_chrom_names"):
        if not isinstance(params[key], bool):
            raise ValueError(f"'params.{key}' must be a YAML boolean.")
    if int(params["threads"]) < 1 or int(params["macs_extsize"]) < 1:
        raise ValueError("threads and macs_extsize must be positive.")
    if not 0 <= int(params["min_mapq"]) <= 255:
        raise ValueError("min_mapq must be between 0 and 255.")
    for key in ("macs_qvalue", "macs_qvalue_no_control", "macs_signal_pvalue", "idr_threshold"):
        if params.get(key) is not None:
            validate_number(params[key], f"params.{key}", float)
            if not 0 < float(params[key]) <= 1:
                raise ValueError(f"'params.{key}' must be >0 and <=1.")
    if params["nrf_thresholds"] is not None:
        thresholds = require_mapping(params["nrf_thresholds"], "params.nrf_thresholds")
        good = float(require_key(thresholds, "good_min", "nrf_thresholds"))
        moderate = float(require_key(thresholds, "moderate_min", "nrf_thresholds"))
        if not 0 <= moderate <= good <= 1:
            raise ValueError("NRF thresholds must satisfy 0 <= moderate_min <= good_min <= 1.")
    for key, value in data["save"].items():
        if not isinstance(value, bool):
            raise ValueError(f"'save.{key}' must be a YAML boolean.")
    if int(params["download_workers"]) < 1:
        raise ValueError("'params.download_workers' must be >= 1.")
    require_list(params.get("keep_chroms"), "params.keep_chroms")

    replicates = require_list(require_key(data, "replicates", "root"), "replicates")
    seen_rep_ids: set[str] = set()
    for idx, rep in enumerate(replicates, start=1):
        rep = require_mapping(rep, f"replicates[{idx}]")
        rep_id = check_identifier(require_key(rep, "rep_id", f"replicates[{idx}]"), "rep_id")
        rep["rep_id"] = rep_id
        if rep_id in seen_rep_ids:
            raise ValueError(f"Duplicate rep_id {rep_id!r}.")
        seen_rep_ids.add(rep_id)

        ip = require_mapping(require_key(rep, "ip", f"replicates[{idx}]"), f"replicates[{idx}].ip")
        sample_context(rep, data, f"replicates[{idx}]")
        raw_control = rep.get("control", data.get("control"))
        if data["control_type"] == "none":
            if raw_control not in (None, [], {}):
                raise ValueError("control_type='none' cannot include control samples.")
            controls = []
        elif isinstance(raw_control, list):
            controls = require_list(raw_control, f"replicates[{idx}].control")
            controls = [
                require_mapping(control, f"replicates[{idx}].control[{control_idx}]")
                for control_idx, control in enumerate(controls, start=1)
            ]
        else:
            if raw_control is None:
                raise ValueError(f"replicates[{idx}]: control_type={data['control_type']!r} requires a control.")
            controls = [require_mapping(raw_control, f"replicates[{idx}].control")]

        validate_sample(ip, layout, f"replicates[{idx}].ip")
        sample_context(ip, data, f"replicates[{idx}].ip")
        for control_idx, control in enumerate(controls, start=1):
            where = f"replicates[{idx}].control"
            if len(controls) > 1:
                where = f"{where}[{control_idx}]"
            validate_sample(control, layout, where)
            sample_context(control, data, where)
            if control.get("control_type", data["control_type"]) != data["control_type"]:
                raise ValueError(f"'{where}.control_type' differs from the experiment control type.")
            if sample_identity_key(ip) == sample_identity_key(control):
                raise ValueError(f"'{where}' and the TF pulldown refer to the same input files/runs.")
        rep["controls"] = controls


def find_config_files(config_dir: Path) -> list[Path]:
    if not config_dir.is_dir():
        raise ValueError(f"Config directory does not exist: {config_dir}")
    return sorted(p for p in config_dir.iterdir() if p.is_file() and p.suffix.lower() in CONFIG_SUFFIXES)


def make_experiment_dirs(outdir: Path, experiment_id: str) -> Path:
    experiment_dir = outdir / experiment_id
    for subdir in [
        "downloads", "trim", "fastqc/raw", "fastqc/trimmed",
        "align/ip", "align/control", "peaks", "signal",
        "qc/frip", "qc/spp", "qc/fingerprint", "qc/flagstat",
        "idr", "consensus", "logs", "reports", "tmp",
    ]:
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
    for path in find_config_files(config_dir):
        configs.append(load_config(path, outdir))
    if not configs:
        raise ValueError(f"No YAML config files found in {config_dir}")
    ids = [str(cfg["experiment_id"]) for cfg in configs]
    if len(ids) != len(set(ids)):
        raise ValueError("experiment_id must be unique across configs to avoid output collisions.")
    return configs


def run_command(cmd: str, log_path: Path | None = None) -> None:
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], stdout=log, stderr=subprocess.STDOUT, text=True)
    else:
        result = subprocess.run(["bash", "-o", "pipefail", "-c", cmd])

    if result.returncode != 0:
        msg = f"Command failed with exit code {result.returncode}: {cmd}"
        if log_path:
            msg += f"\nLog: {log_path}"
            if log_path.exists():
                lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
                if lines:
                    msg += "\nLast log lines:\n" + "\n".join(lines[-40:])
        raise RuntimeError(msg)


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


def run_capture(cmd: str) -> str:
    result = subprocess.run(["bash", "-o", "pipefail", "-c", cmd], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {cmd}\n{result.stderr}")
    return result.stdout.strip()


def sample_label(experiment_id: str, rep_id: str, role: str) -> str:
    return f"{experiment_id}.{rep_id}.{role}"


def sample_download_key(rep: dict[str, Any], label_role: str) -> tuple[str, str]:
    return str(rep["rep_id"]), label_role


def sample_identity_key(sample: dict[str, Any]) -> tuple[str, ...]:
    """Return a stable key for reusing identical sample inputs."""

    source = str(sample["source"])
    layout = get_sample_layout(sample)
    if source == "local":
        values = [source, str(Path(sample["fastq1"]).resolve())]
        if layout == "PE":
            values.append(str(Path(sample["fastq2"]).resolve()))
        return tuple([layout, *values])
    if source == "sra":
        return tuple([layout, source, *as_non_empty_string_list(sample.get("accessions", sample["accession"]), "accession")])
    return layout, source, str(sample["fastq_url_1"]), str(sample.get("fastq_url_2", ""))


def resolve_fastqs(sample: dict[str, Any], layout: str, label: str, experiment_dir: Path, threads: int) -> tuple[Path, Path | None]:
    """Return local FASTQ paths, downloading the sample first when needed."""

    downloads_dir = experiment_dir / "downloads"
    tmp_dir = experiment_dir / "tmp" / label
    logs_dir = experiment_dir / "logs"
    downloads_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    source = sample["source"]

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
    """Start all remote FASTQ downloads before CPU-heavy processing begins."""

    experiment_id = str(data["experiment_id"])
    experiment_dir = Path(data["experiment_dir"])
    threads = int(data["params"]["threads"])
    # Carry preloaded futures from the previous experiment into this one.
    futures = dict(futures or {})
    futures_by_sample: dict[tuple[str, ...], Future[tuple[Path, Path | None]]] = {}
    started = 0

    for rep in data["replicates"]:
        rep_id = str(rep["rep_id"])
        sample_jobs: list[tuple[str, str, dict[str, Any]]] = [("ip", "ip", rep["ip"])]
        controls = control_samples(rep)
        sample_jobs.extend(
            ("control", control_label_role(controls, idx), control)
            for idx, control in enumerate(controls)
        )

        for role, label_role, sample in sample_jobs:
            if sample["source"] == "local":
                continue
            layout = get_sample_layout(sample)
            label = sample_label(experiment_id, rep_id, label_role)
            key = sample_download_key(rep, label_role)
            # Reuse shared controls and samples preloaded from the previous experiment.
            identity = sample_identity_key(sample) if role == "control" else (rep_id, role)
            future = futures.get(key) or futures_by_sample.get(identity)
            if future is None:
                future = executor.submit(
                    resolve_fastqs,
                    sample,
                    layout,
                    label,
                    experiment_dir,
                    threads,
                )
                started += 1
                futures_by_sample[identity] = future
            else:
                print(f"{timestamp()}  Reusing queued FASTQ download for {label}.")
                futures_by_sample[identity] = future
            futures[key] = future
            # For next-experiment preloading, only queue the requested first jobs.
            if limit is not None and started >= limit:
                break
        if limit is not None and started >= limit:
            break

    if futures:
        workers = int(data["params"]["download_workers"])
        print(f"{timestamp()}  Started {len(futures_by_sample)} unique remote FASTQ download job(s) for {len(futures)} sample use(s) with {workers} worker(s).")

    return futures


def run_fastqc(fq1: Path, fq2: Path | None, layout: str, label: str, outdir: Path, threads: int, stage: str) -> Path:
    """Generate per-base quality profiles with FastQC."""

    # FastQC reports: per-base quality, GC content, duplication levels, adapter content.
    # Running FastQC before and after trimming makes adapter removal and quality filtering visible.

    sample_fastqc_dir = outdir / "fastqc" / stage / label
    sample_fastqc_dir.mkdir(parents=True, exist_ok=True)

    print(f"{timestamp()}  FastQC {stage} for {label}: in progress ..")

    inputs = f"{fq1} {fq2}" if layout == "PE" and fq2 else str(fq1)
    cmd = f"export JAVA_TOOL_OPTIONS='-Djava.awt.headless=true' && fastqc -t {threads} -o {sample_fastqc_dir} {inputs}"
    run_command(cmd, outdir / "logs" / f"{label}.fastqc.{stage}.log")

    print(f"\t\t  FastQC output in: {sample_fastqc_dir}")
    return sample_fastqc_dir


def trim_reads(fq1: Path, fq2: Path | None, layout: str, label: str, experiment_dir: Path, params: dict[str, Any]) -> tuple[Path, Path | None, Path, Path]:
    """Trim sequencing adapters and low-quality bases using fastp."""

    # Why trim?
    # Raw reads often contain adapter sequences (synthetic DNA from library prep)
    # and low-quality bases at the 3' end. Adapters will prevent alignment.
    # fastp auto-detects common adapters and trims bases below Q20 (99% accuracy).
    # It also produces a handy HTML report — open it in a browser after running!

    trim_dir = experiment_dir / "trim"
    trim_dir.mkdir(parents=True, exist_ok=True)

    t1 = trim_dir / f"{label}.R1.trim.fastq.gz"
    t2 = trim_dir / f"{label}.R2.trim.fastq.gz" if layout == "PE" else None
    json_report = trim_dir / f"{label}.fastp.json"
    html_report = trim_dir / f"{label}.fastp.html"

    threads = int(params["threads"])
    fastp_extra = str(params.get("fastp_extra", ""))

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
            "--json", str(json_report),
            "--html", str(html_report),
        ])

    run_command(cmd, experiment_dir / "logs" / f"{label}.fastp.log")
    print(f"\t\t  Open {html_report} to check trimming quality.")

    return t1, t2, json_report, html_report


def align_bowtie2(t1: Path, t2: Path | None, layout: str, label: str, role: str, data: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Align trimmed reads to the reference genome using Bowtie2, sort and index the BAM."""

    # Why Bowtie2?
    # Fast, memory-efficient short-read aligner using an FM-index (Burrows-Wheeler transform).
    # --no-mixed / --no-discordant (PE): discard reads where only one mate maps or mates
    #   map too far apart — these are likely artefacts.
    # We pipe bowtie2 -> samtools view (SAM->BAM) -> samtools sort (coordinate order).
    # samtools index builds the .bai index needed for random access.

    experiment_dir = Path(data["experiment_dir"])
    align_dir = experiment_dir / "align" / role
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


def filter_and_dedup(bam: Path, layout: str, label: str, role: str, data: dict[str, Any]) -> tuple[Path, Path, Path]:
    """Filter by MAPQ, keep canonical chromosomes, and remove PCR duplicates with samtools."""

    # Why these filters?
    # 1) MAPQ >= 30: high confidence that the read maps to a unique location.
    #    Reads with lower MAPQ come from repetitive regions and create false peaks.
    # 2) Chromosome whitelist: keep canonical chromosomes and drop organellar DNA when configured.
    # 3) PCR duplicate removal: fragments amplified more than once get identical
    #    coordinates. Keeping them inflates signal and can create spurious peaks.
    #    samtools markdup -r removes them.
    #
    # For PE: name-sort -> fixmate (adds mate info) -> coord-sort -> markdup
    # For SE: coord-sort -> markdup --mode s

    experiment_dir = Path(data["experiment_dir"])
    align_dir = experiment_dir / "align" / role
    align_dir.mkdir(parents=True, exist_ok=True)

    params = data["params"]
    threads = int(params["threads"])
    sort_threads = max(1, threads // 2)
    min_mapq = int(params["min_mapq"])
    chroms = " ".join(str(chrom) for chrom in params["keep_chroms"])

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
        run_command(f"samtools markdup -r -f {stats_txt} {tmp_pfx}.tmp.coordsort.bam {out_bam}")
        run_command(f"rm -f {tmp_pfx}.tmp.namesort.bam {tmp_pfx}.tmp.fixmate.bam {tmp_pfx}.tmp.coordsort.bam")
    else:
        run_command(
            f"samtools view -b -q {min_mapq} {bam} {chroms} "
            f"| samtools sort -@ {sort_threads} -o {tmp_pfx}.tmp.coordsort.bam"
        )
        run_command(f"samtools markdup -r --mode s -f {stats_txt} {tmp_pfx}.tmp.coordsort.bam {out_bam}")
        run_command(f"rm -f {tmp_pfx}.tmp.coordsort.bam")

    run_command(f"samtools index {out_bam}")
    print(f"\t\t  Deduplicated BAM: {out_bam}")
    print(f"\t\t  Duplication stats: {stats_txt}")

    return out_bam, out_bai, stats_txt


def samtools_flagstat(bam: Path, label: str, data: dict[str, Any]) -> Path:
    """Run samtools flagstat to count mapped/unmapped/paired reads."""

    outdir = Path(data["experiment_dir"]) / "qc" / "flagstat"
    outdir.mkdir(parents=True, exist_ok=True)
    out_txt = outdir / f"{label}.flagstat.txt"
    run_command(f"samtools flagstat {bam} > {out_txt}")
    return out_txt


def read_filter_metrics(bam: Path, label: str, data: dict[str, Any]) -> dict[str, Any]:
    """Measure MAPQ loss within retained chromosomes, independently of deduplication."""
    chroms = " ".join(shlex.quote(str(c)) for c in data["params"]["keep_chroms"])
    qb = shlex.quote(str(bam))
    idxstats = Path(data["experiment_dir"]) / "qc" / "idxstats" / f"{label}.idxstats.tsv"
    idxstats.parent.mkdir(parents=True, exist_ok=True)
    run_command(f"samtools idxstats {qb} > {shlex.quote(str(idxstats))}")
    contigs = {line.split("\t")[0] for line in idxstats.read_text().splitlines() if line.strip()}
    organelles = [str(c) for c in data["params"]["organelle_chroms"] if str(c) in contigs]
    total = int(run_capture(f"samtools view -c -F 0x904 {qb}"))
    before = int(run_capture(f"samtools view -c -F 0x904 {qb} {chroms}"))
    after = int(run_capture(f"samtools view -c -F 0x904 -q {int(data['params']['min_mapq'])} {qb} {chroms}"))
    organelle_reads = int(run_capture(f"samtools view -c -F 0x904 {qb} " + " ".join(shlex.quote(c) for c in organelles))) if organelles else 0
    return {"mapped_reads_unfiltered": total, "mapped_reads_before_mapq": before,
            "mapped_reads_after_mapq": after, "mapq_reads_lost": before - after,
            "mapq_loss_fraction": (before - after) / before if before else None,
            "organelle_reads": organelle_reads,
            "organelle_fraction": organelle_reads / total if total else None,
            "idxstats": idxstats}


def call_peaks_macs3(ip_bam: Path, control_bams: list[Path], rep_id: str, data: dict[str, Any], fmt: str,
                     fragment_length: int | None = None) -> tuple[Path, Path]:
    """Model first for SE; use measured library fragments for the model fallback."""

    # Why MACS3?
    # MACS is the standard TF peak caller. It models the local background and finds
    # windows with significant enrichment. Mock and input controls capture different
    # technical backgrounds; without control, MACS estimates lambda from treatment.

    experiment_id = str(data["experiment_id"])
    experiment_dir = Path(data["experiment_dir"])
    peaks_dir = experiment_dir / "peaks"
    peaks_dir.mkdir(parents=True, exist_ok=True)

    params = data["params"]
    reference = data["reference"]
    peak_name = f"{experiment_id}.{rep_id}"
    raw_peaks = peaks_dir / f"{peak_name}.raw.narrowPeak"
    summits = peaks_dir / f"{peak_name}.summits.bed"

    print(f"{timestamp()}  Calling peaks with MACS3 for {peak_name}: in progress ..")

    base_cmd = [
        "macs3 callpeak",
        "-t", shlex.quote(str(ip_bam)),
        "-f", fmt,
        "-g", str(reference["genome_size"]),
        "-q", str(params["macs_qvalue_no_control"] if not control_bams and params.get("macs_qvalue_no_control") is not None else params["macs_qvalue"]),
        "--call-summits",
        "--keep-dup all",
        "--outdir", shlex.quote(str(peaks_dir)),
        "-n", shlex.quote(peak_name),
    ]
    if control_bams:
        base_cmd.extend(["-c", " ".join(shlex.quote(str(bam)) for bam in control_bams)])
    metrics = {"mode": "paired_end_fragments" if fmt == "BAMPE" else "macs_model",
               "fragment_length_bp": None, "control_used": bool(control_bams)}
    data.setdefault("peak_calling", {})[rep_id] = metrics
    log_path = experiment_dir / "logs" / f"{peak_name}.macs3.log"

    try:
        run_command(" ".join(base_cmd), log_path)
    except RuntimeError as exc:
        log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        model_failed = any(text in log_text.lower() for text in (
            "needs at least 100 paired peaks", "can't build the model", "cannot build the model",
            "not enough paired peaks", "no proper paired peaks", "not enough pairs"))
        if fmt != "BAM" or not model_failed:
            raise

        extsize = int(fragment_length or params["macs_extsize"])
        report_error(data, f"MACS3 SE model failed for {peak_name}; retrying with --nomodel --extsize {extsize} (library fragment length).")
        metrics.update(mode="nomodel_fallback", fragment_length_bp=extsize)
        shutil.copy2(log_path, log_path.with_name(f"{peak_name}.macs3.model_attempt.log"))
        retry_cmd = " ".join([*base_cmd, "--nomodel", "--extsize", str(extsize)])
        try:
            run_command(retry_cmd, log_path)
        except RuntimeError as retry_exc:
            raise retry_exc from exc

    if metrics["mode"] == "macs_model" and log_path.exists():
        match = re.search(r"predicted fragment length is\s+(\d+)", log_path.read_text(errors="replace"), re.I)
        if match:
            metrics["fragment_length_bp"] = int(match.group(1))
    (peaks_dir / f"{peak_name}_peaks.narrowPeak").replace(raw_peaks)
    (peaks_dir / f"{peak_name}_summits.bed").replace(summits)

    print(f"\t\t  Called {count_lines(raw_peaks)} raw peaks. Summits in: {summits}")
    return raw_peaks, summits


def greenscreen_filter(peaks_raw: Path, rep_id: str, data: dict[str, Any]) -> Path:
    """Remove peaks overlapping known artefact regions using bedtools intersect."""

    # The reference Greenscreen also masks mapping artefacts in naked gDNA.
    # bedtools intersect -v keeps only peaks that do NOT overlap the mask.

    experiment_id = str(data["experiment_id"])
    experiment_dir = Path(data["experiment_dir"])
    reference = data["reference"]
    filtered_peaks = experiment_dir / "peaks" / f"{experiment_id}.{rep_id}.greenscreen.narrowPeak"

    mask_bed = normalize_mask_bed(data)
    print(f"{timestamp()}  Greenscreen/blacklist filtering for {experiment_id}.{rep_id}: in progress ..")

    with peaks_raw.open() as fh:
        peak_chroms = {line.split()[0] for line in fh if line.strip() and not line.startswith(("#", "track", "browser"))}
    expected = set(map(str, data["params"]["keep_chroms"]))
    if peak_chroms - expected:
        report_error(data, f"Peak chromosome names do not match the configured reference: {sorted(peak_chroms - expected)}.")
        raise ValueError("Peak/mask chromosome mismatch; refusing unmasked peaks.")
    run_command(f"bedtools intersect -v -a {shlex.quote(str(peaks_raw))} -b {shlex.quote(str(mask_bed))} > {shlex.quote(str(filtered_peaks))}")
    n_before = count_lines(peaks_raw)
    n_after = count_lines(filtered_peaks)
    data.setdefault("greenscreen", {})[rep_id] = {"before": n_before, "after": n_after, "removed": n_before - n_after}
    print(f"\t\t  Removed {n_before - n_after}/{n_before} peaks overlapping the mask. {n_after} retained.")

    return filtered_peaks


def bam_to_bigwig(bam: Path, label: str, data: dict[str, Any], layout: str) -> Path:
    """Convert filtered BAM to a CPM-normalised bigWig signal track with deepTools."""

    # bigWig is a compressed, indexed format for continuous genomic signals.
    # CPM normalisation makes samples with different sequencing depths comparable.

    experiment_dir = Path(data["experiment_dir"])
    signal_dir = experiment_dir / "signal"
    signal_dir.mkdir(parents=True, exist_ok=True)

    params = data["params"]
    reference = data["reference"]
    threads = int(params["threads"])
    bw = signal_dir / f"{label}.cpm.bw"
    bl = f"--blackListFileName {reference['mask_bed']}" if reference.get("mask_bed") and Path(reference["mask_bed"]).is_file() else ""

    print(f"{timestamp()}  Generating CPM bigWig for {label}: in progress ..")

    if layout == "PE":
        cmd = " ".join([
            "bamCoverage",
            "-b", str(bam),
            "-o", str(bw),
            "--normalizeUsing CPM",
            "--extendReads",
            bl,
            "--binSize 10",
            "-p", str(threads),
        ])
    else:
        cmd = " ".join([
            "bamCoverage",
            "-b", str(bam),
            "-o", str(bw),
            "--normalizeUsing CPM",
            bl,
            "--binSize 10",
            "-p", str(threads),
        ])

    run_command(cmd, experiment_dir / "logs" / f"{label}.bamCoverage.log")
    return bw


def fold_enrichment_bigwig(ip_bam: Path, control_bam: Path, rep_id: str, data: dict[str, Any]) -> Path:
    """Compute a log2(IP / control) fold-enrichment bigWig using deepTools bamCompare."""

    # Dividing IP by input cancels background and highlights regions specifically
    # enriched in the TF pulldown. log2 ratio + pseudocount=1 centers unchanged regions at 0.

    experiment_id = str(data["experiment_id"])
    experiment_dir = Path(data["experiment_dir"])
    signal_dir = experiment_dir / "signal"
    signal_dir.mkdir(parents=True, exist_ok=True)

    params = data["params"]
    reference = data["reference"]
    threads = int(params["threads"])
    bw = signal_dir / f"{experiment_id}.{rep_id}.log2_ip_vs_control.bw"
    bl = f"--blackListFileName {reference['mask_bed']}" if reference.get("mask_bed") and Path(reference["mask_bed"]).is_file() else ""

    print(f"{timestamp()}  Fold-enrichment bigWig for {experiment_id}.{rep_id}: in progress ..")

    cmd = " ".join([
        "bamCompare",
        "-b1", str(ip_bam),
        "-b2", str(control_bam),
        "--operation log2",
        "--pseudocount 1",
        "--normalizeUsing CPM",
        "--scaleFactorsMethod None",
        bl,
        "--binSize 10",
        "-p", str(threads),
        "-o", str(bw),
    ])
    run_command(cmd, experiment_dir / "logs" / f"{experiment_id}.{rep_id}.bamCompare.log")
    return bw


def merged_control_bam(control_outputs: list[dict[str, Any]], rep_id: str, data: dict[str, Any]) -> dict[str, Any] | None:
    """Return one BAM for tools that accept only a single control BAM."""

    if not control_outputs:
        return None
    if len(control_outputs) == 1:
        return control_outputs[0]

    experiment_id = str(data["experiment_id"])
    experiment_dir = Path(data["experiment_dir"])
    align_dir = experiment_dir / "align" / "control"
    align_dir.mkdir(parents=True, exist_ok=True)

    label = sample_label(experiment_id, rep_id, "control_merged")
    bam = align_dir / f"{label}.filtered.dedup.bam"
    bai = Path(f"{bam}.bai")
    input_bams = [Path(control["filtered_bam"]) for control in control_outputs]

    if input_bams:
        print(f"{timestamp()}  Merging {len(input_bams)} controls for {experiment_id}.{rep_id}: in progress ..")
        run_command(
            " ".join([
                "samtools merge -f",
                shlex.quote(str(bam)),
                " ".join(shlex.quote(str(path)) for path in input_bams),
            ]),
            experiment_dir / "logs" / f"{label}.samtools_merge.log",
        )
        run_command(f"samtools index {shlex.quote(str(bam))}")

    return {
        "label": label,
        "filtered_bam": bam,
        "filtered_bai": bai,
        "source_controls": control_outputs,
    }


def replicates_for_pooled_bigwig(replicate_results: list[dict[str, Any]], data: dict[str, Any]) -> list[dict[str, Any]]:
    # Future QC gates belong here, e.g. FRiP >= 0.1 or NSC >= 1.05.
    return replicate_results


def pooled_log2_bigwig(replicate_results: list[dict[str, Any]], data: dict[str, Any]) -> Path | None:
    """Compute a pooled log2(IP/control) bigWig from all replicate BAMs."""

    selected = replicates_for_pooled_bigwig(replicate_results, data)
    if not selected:
        return None
    if data["control_type"] == "none":
        print(f"{timestamp()}  Pooled bamCompare log2 track skipped: control_type=none.")
        return None

    experiment_id = str(data["experiment_id"])
    experiment_dir = Path(data["experiment_dir"])
    signal_dir = experiment_dir / "signal"
    pooled_dir = experiment_dir / "tmp" / "pooled_log2"
    signal_dir.mkdir(parents=True, exist_ok=True)
    pooled_dir.mkdir(parents=True, exist_ok=True)

    name = f"{experiment_id}.pooled_replicates"
    ip_bams = [Path(r["ip"]["filtered_bam"]) for r in selected]
    control_bams = list(dict.fromkeys(
        Path(control["filtered_bam"])
        for r in selected
        for control in r.get("controls", [])
    ))
    pooled_ip = pooled_dir / f"{name}.ip.filtered.dedup.bam"
    pooled_control = pooled_dir / f"{name}.control.filtered.dedup.bam"
    bw = signal_dir / f"{name}.log2_ip_vs_control.bw"

    print(f"{timestamp()}  Pooled log2(IP/control) bigWig for {experiment_id}: in progress ..")

    run_command(
        " ".join([
            "samtools merge -f",
            shlex.quote(str(pooled_ip)),
            " ".join(shlex.quote(str(path)) for path in ip_bams),
        ]),
        experiment_dir / "logs" / f"{name}.pooled_ip.samtools_merge.log",
    )
    run_command(f"samtools index {shlex.quote(str(pooled_ip))}")
    run_command(
        " ".join([
            "samtools merge -f",
            shlex.quote(str(pooled_control)),
            " ".join(shlex.quote(str(path)) for path in control_bams),
        ]),
        experiment_dir / "logs" / f"{name}.pooled_control.samtools_merge.log",
    )
    run_command(f"samtools index {shlex.quote(str(pooled_control))}")

    params = data["params"]
    reference = data["reference"]
    threads = int(params["threads"])
    bl = f"--blackListFileName {reference['mask_bed']}" if reference.get("mask_bed") and Path(reference["mask_bed"]).is_file() else ""
    cmd = " ".join([
        "bamCompare",
        "-b1", str(pooled_ip),
        "-b2", str(pooled_control),
        "--operation log2",
        "--pseudocount 1",
        "--normalizeUsing CPM",
        "--scaleFactorsMethod None",
        bl,
        "--binSize 10",
        "-p", str(threads),
        "-o", str(bw),
    ])
    run_command(cmd, experiment_dir / "logs" / f"{name}.pooled_bamCompare.log")

    for path in [pooled_ip, Path(f"{pooled_ip}.bai"), pooled_control, Path(f"{pooled_control}.bai")]:
        if path.exists():
            path.unlink()

    return bw


def chrom_sizes_file(data: dict[str, Any]) -> Path:
    """Return a chromosome sizes file, creating it from genome_fasta.fai when needed."""

    reference = data["reference"]
    if reference.get("chrom_sizes"):
        chrom_sizes = Path(reference["chrom_sizes"])
        if chrom_sizes.is_file():
            return chrom_sizes
        raise FileNotFoundError(f"reference.chrom_sizes does not exist: {chrom_sizes}")

    genome_fasta = reference.get("genome_fasta")
    if not genome_fasta:
        raise ValueError("reference.genome_fasta or reference.chrom_sizes is required for MACS3 FE bigWig output.")

    fasta = Path(genome_fasta)
    fai = Path(f"{fasta}.fai")
    if not fai.exists():
        run_command(f"samtools faidx {shlex.quote(str(fasta))}")

    out = Path(data["experiment_dir"]) / "tmp" / "reference.chrom.sizes"
    out.parent.mkdir(parents=True, exist_ok=True)
    run_command(f"cut -f1,2 {shlex.quote(str(fai))} > {shlex.quote(str(out))}")
    return out


def read_chrom_sizes(chrom_sizes: Path) -> dict[str, str]:
    """Read a two-column chrom.sizes file as chromosome -> size strings."""

    sizes: dict[str, str] = {}
    with chrom_sizes.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 2:
                continue
            sizes[fields[0]] = fields[1]
    return sizes


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


def chrom_alias_file(chrom_sizes: Path, data: dict[str, Any]) -> Path:
    """Create an alias table that maps common MACS/BAM chromosome names to chrom.sizes names."""

    tmp_dir = Path(data["experiment_dir"]) / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out = tmp_dir / "reference.chrom.aliases.tsv"
    sizes = read_chrom_sizes(chrom_sizes)

    alias_to_target: dict[str, str] = {}
    for chrom in sizes:
        for alias in chrom_aliases_for_target(chrom):
            alias_to_target.setdefault(alias, chrom)

    with out.open("w", encoding="utf-8") as fh:
        for alias, target in sorted(alias_to_target.items()):
            fh.write(f"{alias}\t{target}\n")

    return out


def normalize_bedgraph_chrom_names(input_bdg: Path, chrom_sizes: Path, data: dict[str, Any], label: str) -> Path:
    """Rewrite bedGraph chromosome names to match the chrom.sizes file before bigWig conversion."""

    params = data["params"]
    if not bool(params.get("macs_signal_normalize_chrom_names", True)):
        return input_bdg

    experiment_dir = Path(data["experiment_dir"])
    tmp_dir = input_bdg.parent
    aliases = chrom_alias_file(chrom_sizes, data)
    normalized = tmp_dir / f"{label}.chrom_normalized.bdg"

    run_command(
        "awk 'BEGIN{OFS=\"\\t\"} "
        "NR==FNR{alias[$1]=$2; next} "
        "$1==\"track\" || $1==\"browser\" {next} "
        "{if ($1 in alias) {$1=alias[$1]; print $0}}' "
        f"{shlex.quote(str(aliases))} {shlex.quote(str(input_bdg))} > {shlex.quote(str(normalized))}",
        experiment_dir / "logs" / f"{label}.normalize_chrom_names.log",
    )
    return normalized


def bam_to_tagalign(bam: Path, layout: str, tagalign: Path, data: dict[str, Any]) -> Path:
    """Convert a filtered BAM to ENCODE-style tagAlign input for MACS signal tracks."""

    experiment_dir = Path(data["experiment_dir"])
    params = data["params"]
    threads = int(params["threads"])
    tmp_dir = tagalign.parent
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if layout == "PE":
        namesort_bam = tmp_dir / f"{tagalign.stem}.namesort.bam"
        run_command(
            f"samtools sort -n -@ {max(1, threads // 2)} -o {shlex.quote(str(namesort_bam))} {shlex.quote(str(bam))}",
            experiment_dir / "logs" / f"{tagalign.stem}.namesort.log",
        )
        cmd = (
            f"LC_COLLATE=C bedtools bamtobed -bedpe -mate1 -i {shlex.quote(str(namesort_bam))} "
            "| awk 'BEGIN{OFS=\"\\t\"} "
            "$1==$4 {printf \"%s\\t%s\\t%s\\tN\\t1000\\t%s\\n%s\\t%s\\t%s\\tN\\t1000\\t%s\\n\", "
            "$1,$2,$3,$9,$4,$5,$6,$10}' "
            f"| gzip -nc > {shlex.quote(str(tagalign))}"
        )
        run_command(cmd, experiment_dir / "logs" / f"{tagalign.stem}.bam2tagalign.log")
        namesort_bam.unlink(missing_ok=True)
    else:
        cmd = (
            f"bedtools bamtobed -i {shlex.quote(str(bam))} "
            "| awk 'BEGIN{OFS=\"\\t\"}{$4=\"N\";$5=\"1000\";print $0}' "
            f"| gzip -nc > {shlex.quote(str(tagalign))}"
        )
        run_command(cmd, experiment_dir / "logs" / f"{tagalign.stem}.bam2tagalign.log")

    return tagalign


def parse_spp_fragment_length(spp_txt: Path) -> int | None:
    """Parse the estimated fragment length from phantompeakqualtools output."""

    if not spp_txt.exists() or spp_txt.stat().st_size == 0:
        return None

    with spp_txt.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                continue
            value = fields[2].split(",", 1)[0].strip()
            if value.lstrip("-").isdigit() and int(value) > 0:
                return int(value)
    return None


def select_fragment_length(spp_txt: Path, data: dict[str, Any], label: str,
                           other_estimates: list[int] | None = None) -> tuple[int, str]:
    fraglen = parse_spp_fragment_length(spp_txt)
    if fraglen:
        return fraglen, "spp"
    estimates = [int(n) for n in other_estimates or [] if n and int(n) > 0]
    if estimates:
        value = round(mean(estimates))
        report_error(data, f"No pooled SPP estimate for {label}; using mean replicate SPP estimate {value} bp.")
        return value, "mean_replicate_spp"
    fallback = int(data["params"]["macs_extsize"])
    report_error(data, f"No usable SPP fragment length for {label}; using configured macs_extsize={fallback} bp.")
    return fallback, "configured_fallback"


def estimate_pooled_fragment_length(pooled_ip: Path, layout: str, data: dict[str, Any], name: str) -> int:
    out_tsv = compute_nsc_rsc(pooled_ip, name, data, layout)
    value, source = select_fragment_length(out_tsv, data, name, data.get("replicate_spp_fragment_lengths"))
    data["pooled_fragment_length"] = {"bp": value, "source": source}
    print(f"{timestamp()}  Pooled fragment length: {value} bp ({source}).")
    return value


def clean_macs_bedgraph_to_bigwig(input_bdg: Path, output_bw: Path, chrom_sizes: Path, data: dict[str, Any], label: str) -> Path:
    """Clip, sort, de-overlap, and convert a MACS bedGraph to bigWig."""

    experiment_dir = Path(data["experiment_dir"])
    tmp_dir = input_bdg.parent
    normalized_bdg = normalize_bedgraph_chrom_names(input_bdg, chrom_sizes, data, label)
    clipped = tmp_dir / f"{label}.clipped.bdg"
    sorted_bdg = tmp_dir / f"{label}.sorted.bdg"

    run_command(
        f"bedtools slop -i {shlex.quote(str(normalized_bdg))} -g {shlex.quote(str(chrom_sizes))} -b 0 "
        "| awk 'BEGIN{OFS=\"\\t\"}{if ($3 != -1 && $2 < $3) print $0}' "
        f"> {shlex.quote(str(clipped))}",
        experiment_dir / "logs" / f"{label}.clip_bedgraph.log",
    )
    run_command(
        f"LC_COLLATE=C sort -k1,1 -k2,2n {shlex.quote(str(clipped))} "
        "| awk 'BEGIN{OFS=\"\\t\"}{if (NR==1 || prev_chr!=$1 || prev_end<=$2) "
        "{print $0}; prev_chr=$1; prev_end=$3}' "
        f"> {shlex.quote(str(sorted_bdg))}",
        experiment_dir / "logs" / f"{label}.sort_bedgraph.log",
    )
    run_command(
        f"bedGraphToBigWig {shlex.quote(str(sorted_bdg))} {shlex.quote(str(chrom_sizes))} {shlex.quote(str(output_bw))}",
        experiment_dir / "logs" / f"{label}.bedGraphToBigWig.log",
    )
    return output_bw


def subsample_control_tagalign_if_needed(ip_ta: Path, control_ta: Path, layout: str, data: dict[str, Any], name: str) -> Path:
    """Subsample control tagAlign to IP depth, matching ENCODE signal-track behavior."""

    params = data["params"]
    if not bool(params.get("macs_signal_subsample_control", True)):
        return control_ta

    experiment_dir = Path(data["experiment_dir"])
    ip_lines = count_gzip_lines(ip_ta)
    control_lines = count_gzip_lines(control_ta)
    if control_lines <= ip_lines or ip_lines <= 0:
        print(f"\t\t  Control tagAlign not subsampled ({control_lines} control lines <= {ip_lines} IP lines).")
        return control_ta

    subsampled = control_ta.parent / f"{name}.control.subsampled.tagAlign.gz"
    print(f"\t\t  Subsampling control tagAlign from {control_lines} to {ip_lines} lines.")

    if layout == "PE":
        target_pairs = max(1, ip_lines // 2)
        cmd = (
            f"gzip -cd {shlex.quote(str(control_ta))} "
            "| sed 'N;s/\\n/\\t/' "
            f"| shuf -n {target_pairs} "
            "| awk 'BEGIN{OFS=\"\\t\"}{print $1,$2,$3,$4,$5,$6; print $7,$8,$9,$10,$11,$12}' "
            f"| gzip -nc > {shlex.quote(str(subsampled))}"
        )
    else:
        cmd = (
            f"gzip -cd {shlex.quote(str(control_ta))} "
            f"| shuf -n {ip_lines} "
            f"| gzip -nc > {shlex.quote(str(subsampled))}"
        )

    run_command(cmd, experiment_dir / "logs" / f"{name}.control_subsample_tagalign.log")
    return subsampled


def pooled_macs3_fe_bigwig(replicate_results: list[dict[str, Any]], data: dict[str, Any]) -> Path | None:
    """Compute pooled MACS3 FE/p-value tracks, including n=1 and no-control assays."""

    selected = replicates_for_pooled_bigwig(replicate_results, data)
    if not selected:
        return None

    experiment_id = str(data["experiment_id"])
    experiment_dir = Path(data["experiment_dir"])
    signal_dir = experiment_dir / "signal"
    pooled_dir = experiment_dir / "tmp" / "pooled_macs3_fe"
    signal_dir.mkdir(parents=True, exist_ok=True)
    pooled_dir.mkdir(parents=True, exist_ok=True)

    name = f"{experiment_id}.pooled_replicates.macs3"
    layouts = [str(r["ip"]["layout"]).upper() for r in selected]
    layout = "PE" if all(item == "PE" for item in layouts) else "SE"
    ip_bams = [Path(r["ip"]["filtered_bam"]) for r in selected]
    controls = list({Path(control["filtered_bam"]): control
                     for r in selected for control in r.get("controls", [])}.values())
    control_bams = [Path(control["filtered_bam"]) for control in controls]
    control_layout = "PE" if controls and all(c["layout"] == "PE" for c in controls) else "SE"
    pooled_ip = pooled_dir / f"{name}.ip.filtered.dedup.bam"
    pooled_control = pooled_dir / f"{name}.control.filtered.dedup.bam"
    ip_ta = pooled_dir / f"{name}.ip.tagAlign.gz"
    control_ta = pooled_dir / f"{name}.control.tagAlign.gz"
    fc_bw = signal_dir / f"{name}.FE.bw"
    pval_bw = signal_dir / f"{name}.pval.bw"

    print(f"{timestamp()}  ENCODE-style pooled MACS3 FE bigWig for {experiment_id}: in progress ..")
    run_command(
        " ".join([
            "samtools merge -f",
            shlex.quote(str(pooled_ip)),
            " ".join(shlex.quote(str(path)) for path in ip_bams),
        ]),
        experiment_dir / "logs" / f"{name}.pooled_ip.samtools_merge.log",
    )
    run_command(f"samtools index {shlex.quote(str(pooled_ip))}")
    if control_bams:
        run_command(
            " ".join(["samtools merge -f", shlex.quote(str(pooled_control)),
                      " ".join(shlex.quote(str(path)) for path in control_bams)]),
            experiment_dir / "logs" / f"{name}.pooled_control.samtools_merge.log",
        )
        run_command(f"samtools index {shlex.quote(str(pooled_control))}")

    fraglen = estimate_pooled_fragment_length(pooled_ip, layout, data, name)
    bam_to_tagalign(pooled_ip, layout, ip_ta, data)
    macs_control_ta = None
    if control_bams:
        bam_to_tagalign(pooled_control, control_layout, control_ta, data)
        macs_control_ta = subsample_control_tagalign_if_needed(ip_ta, control_ta, control_layout, data, name)

    reference = data["reference"]
    params = data["params"]
    pval_thresh = params.get("macs_signal_pvalue", params["macs_qvalue"])
    prefix = pooled_dir / name
    macs_log = experiment_dir / "logs" / f"{name}.macs3_signal_callpeak.log"
    run_command(
        " ".join([
            "macs3 callpeak",
            "-t", shlex.quote(str(ip_ta)),
            *(["-c", shlex.quote(str(macs_control_ta))] if macs_control_ta else []),
            "-f BED",
            "-n", shlex.quote(name),
            "-g", str(reference["genome_size"]),
            "-p", str(pval_thresh),
            "--nomodel",
            "--shift 0",
            "--extsize", str(fraglen),
            "--keep-dup all",
            "-B",
            "--SPMR",
            "--outdir", shlex.quote(str(pooled_dir)),
        ]),
        macs_log,
    )

    treat_pileup = pooled_dir / f"{name}_treat_pileup.bdg"
    control_lambda = pooled_dir / f"{name}_control_lambda.bdg"
    run_command(
        f"macs3 bdgcmp -t {shlex.quote(str(treat_pileup))} -c {shlex.quote(str(control_lambda))} "
        f"--o-prefix {shlex.quote(str(prefix))} -m FE",
        experiment_dir / "logs" / f"{name}.macs3_bdgcmp_fe.log",
    )
    sval = count_gzip_lines(ip_ta) / 1_000_000.0
    run_command(
        f"macs3 bdgcmp -t {shlex.quote(str(treat_pileup))} -c {shlex.quote(str(control_lambda))} "
        f"--o-prefix {shlex.quote(str(prefix))} -m ppois -S {sval}",
        experiment_dir / "logs" / f"{name}.macs3_bdgcmp_ppois.log",
    )

    chrom_sizes = chrom_sizes_file(data)
    clean_macs_bedgraph_to_bigwig(pooled_dir / f"{name}_FE.bdg", fc_bw, chrom_sizes, data, f"{name}.FE")
    clean_macs_bedgraph_to_bigwig(pooled_dir / f"{name}_ppois.bdg", pval_bw, chrom_sizes, data, f"{name}.pval")

    print(f"\t\t  MACS3 FE bigWig: {fc_bw}")
    print(f"\t\t  MACS3 p-value bigWig: {pval_bw}")
    return fc_bw


def compute_frip(bam: Path, peaks: Path, rep_id: str, data: dict[str, Any], layout: str) -> dict[str, Any]:
    """Calculate what fraction of all mapped reads land inside the called peaks."""

    # FRiP = reads_in_peaks / total_mapped_reads.
    # For PE we count read1 of proper pairs to get one count per fragment.

    experiment_id = str(data["experiment_id"])
    experiment_dir = Path(data["experiment_dir"])
    frip_dir = experiment_dir / "qc" / "frip"
    frip_dir.mkdir(parents=True, exist_ok=True)
    out_tsv = frip_dir / f"{experiment_id}.{rep_id}.frip.tsv"

    print(f"{timestamp()}  Computing FRiP for {experiment_id}.{rep_id}: in progress ..")

    if layout == "PE":
        total = int(run_capture(f"samtools view -c -F 0x904 -f 0x42 {bam}"))
        in_peaks = int(run_capture(
            f"samtools view -b -F 0x904 -f 0x42 {bam} "
            f"| bedtools intersect -u -abam stdin -b {peaks} "
            f"| samtools view -c -"
        ))
    else:
        total = int(run_capture(f"samtools view -c -F 0x904 {bam}"))
        in_peaks = int(run_capture(
            f"samtools view -b -F 0x904 {bam} "
            f"| bedtools intersect -u -abam stdin -b {peaks} "
            f"| samtools view -c -"
        ))

    frip = in_peaks / total if total > 0 else 0.0
    with out_tsv.open("w") as fh:
        fh.write("metric\tvalue\n")
        fh.write(f"total_mapped_reads\t{total}\n")
        fh.write(f"reads_in_peaks\t{in_peaks}\n")
        fh.write(f"FRiP\t{frip:.6f}\n")

    return {"total_mapped_reads": total, "reads_in_peaks": in_peaks, "FRiP": frip, "path": out_tsv}


def compute_nsc_rsc(bam: Path, label: str, data: dict[str, Any], layout: str) -> Path:
    """Compute NSC/RSC quality scores via strand cross-correlation with SPP."""

    # Opposite strands provide an estimate of the gDNA library fragment length.
    # Shifting the minus-strand pile by fragment size aligns it with the plus strand.
    # The correlation at that shift indicates signal quality.

    experiment_dir = Path(data["experiment_dir"])
    params = data["params"]
    threads = int(params["threads"])
    spp_dir = experiment_dir / "qc" / "spp"
    spp_dir.mkdir(parents=True, exist_ok=True)

    out_tsv = spp_dir / f"{label}.spp.tsv"
    out_pdf = spp_dir / f"{label}.spp.pdf"
    log_path = experiment_dir / "logs" / f"{label}.spp.log"
    out_tsv.unlink(missing_ok=True)

    print(f"{timestamp()}  NSC/RSC for {label}: in progress ..")

    if layout == "PE":
        cmd = (
            f"TMPBAM=$(mktemp --suffix=.bam); "
            f"samtools view -b -f 0x40 {bam} > $TMPBAM && "
            f"samtools index $TMPBAM && "
            f"Rscript $(which run_spp.R) -c=$TMPBAM -p={threads} "
            f"-savp={out_pdf} -out={out_tsv} -rf; "
            f"status=$?; rm -f $TMPBAM ${{TMPBAM}}.bai; exit $status"
        )
    else:
        cmd = (
            f"Rscript $(which run_spp.R) -c={bam} -p={threads} "
            f"-savp={out_pdf} -out={out_tsv} -rf"
        )

    try:
        run_command(cmd, log_path)
    except RuntimeError:
        out_tsv.unlink(missing_ok=True)
        report_error(data, f"SPP failed for {label}; NSC/RSC unavailable. See {log_path}.")
    return out_tsv


def plot_fingerprint(ip_bam: Path, control_bam: Path | None, rep_id: str, data: dict[str, Any]) -> tuple[Path, Path]:
    """Generate a deepTools plotFingerprint cumulative coverage curve."""

    # A diagonal line = uniform coverage, expected for input/control DNA.
    # A steep curve indicates concentrated coverage; it is descriptive QC only.

    experiment_id = str(data["experiment_id"])
    experiment_dir = Path(data["experiment_dir"])
    params = data["params"]
    threads = int(params["threads"])
    outdir = experiment_dir / "qc" / "fingerprint"
    outdir.mkdir(parents=True, exist_ok=True)

    pdf = outdir / f"{experiment_id}.{rep_id}.fingerprint.pdf"
    tsv = outdir / f"{experiment_id}.{rep_id}.fingerprint.tsv"

    cmd = " ".join([
        "plotFingerprint",
        "-b", shlex.quote(str(ip_bam)), *([shlex.quote(str(control_bam))] if control_bam else []),
        "--labels", shlex.quote(f"DAP_{rep_id}"), *([shlex.quote(data["control_type"])] if control_bam else []),
        "--plotFile", str(pdf),
        "--outRawCounts", str(tsv),
        "-p", str(threads),
    ])
    run_command(cmd, experiment_dir / "logs" / f"{experiment_id}.{rep_id}.fingerprint.log")
    return pdf, tsv


def run_idr(peak_files: list[Path], data: dict[str, Any]) -> Path | None:
    """Apply the configured IDR threshold; preserve narrowPeak/summit columns."""
    if len(peak_files) < 2:
        data["idr_status"] = "not_run_single_replicate"
        return None
    sample_id = str(data["sample_id"])
    idr_dir = Path(data["experiment_dir"]) / "idr"
    idr_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(data["params"]["idr_threshold"])
    print(f"{timestamp()}  IDR for {sample_id}: {len(peak_files)} replicates, threshold={threshold}.")
    sorted_peaks = []
    for i, peaks in enumerate(peak_files, 1):
        sorted_peak = idr_dir / f"{sample_id}.rep{i}.sorted.narrowPeak"
        run_command(f"sort -k8,8nr {shlex.quote(str(peaks))} > {shlex.quote(str(sorted_peak))}")
        sorted_peaks.append(sorted_peak)
    outputs = []
    completed_pairs = 0
    for a, b in itertools.combinations(sorted_peaks, 2):
        pair = f"{a.stem}__vs__{b.stem}"
        out = idr_dir / f"{sample_id}.{pair}.idr.narrowPeak"
        log = idr_dir / f"{sample_id}.{pair}.idr.log"
        terminal = idr_dir / f"{sample_id}.{pair}.idr.stderr.log"
        out.unlink(missing_ok=True)
        cmd = " ".join([
            "idr", "--samples", shlex.quote(str(a)), shlex.quote(str(b)),
            "--input-file-type narrowPeak", "--rank p.value",
            "--idr-threshold", str(threshold),
            "--output-file", shlex.quote(str(out)), "--plot",
            "--log-output-file", shlex.quote(str(log)),
        ])
        try:
            run_command(cmd, terminal)
        except RuntimeError as exc:
            text = str(exc) + "\n" + "\n".join(path.read_text(errors="replace") for path in [terminal, log] if path.exists())
            if "Peak files must contain at least 20 peaks post-merge" in text:
                report_error(data, f"IDR skipped for {pair}: fewer than 20 merged peaks.")
                continue
            raise
        if not out.is_file():
            raise FileNotFoundError(f"IDR did not write its output: {out}")
        completed_pairs += 1
        outputs.append(out)
    if not completed_pairs:
        data["idr_status"] = "skipped_insufficient_peaks"
        report_error(data, "No IDR pair could be evaluated; consensus falls back to replicate 1.")
        return None
    # Keep the original pairwise-union policy for >2 replicates, but retain
    # all ten narrowPeak columns and distinct summits instead of bedtools merge.
    records: dict[tuple[str, int, int, int], tuple[float, list[str]]] = {}
    cutoff = -math.log10(threshold)
    for output in outputs:
        with output.open() as fh:
            for line in fh:
                if not line.strip() or line.startswith("#"):
                    continue
                fields = line.split()
                if len(fields) < 12:
                    raise ValueError(f"Malformed IDR narrowPeak row in {output}")
                global_score = float(fields[11])
                if global_score < cutoff:  # explicit check of -log10(global IDR)
                    continue
                key = (fields[0], int(fields[1]), int(fields[2]), int(fields[9]))
                if key not in records or global_score > records[key][0]:
                    records[key] = (global_score, fields[:10])
    final_idr = idr_dir / f"{sample_id}.idr.narrowPeak"
    with final_idr.open("w") as fh:
        for i, (_, fields) in enumerate((records[k] for k in sorted(records)), 1):
            fields[3] = f"{sample_id}.idr_peak_{i}"
            fh.write("\t".join(fields) + "\n")
    data["idr_status"] = "completed"
    data["idr_pairs_completed"] = completed_pairs
    print(f"{timestamp()}  IDR peaks passing the threshold: {count_lines(final_idr)} -> {final_idr}")
    # An evaluated but empty IDR set remains empty; do not silently use rep 1.
    return final_idr


def make_consensus_peaks(source_peaks: Path, data: dict[str, Any]) -> tuple[Path, Path]:
    """Copy final peaks to consensus location and generate a summit BED file."""

    # The consensus is the input for downstream motif analysis.
    # Summit BED centers downstream sequence extraction on the best peak position.

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


def cleanup_sample_files(result: dict[str, Any], sample: dict[str, Any], role: str, save: dict[str, Any]) -> None:
    if sample["source"] != "local" and not save["raw_fastq"]:
        for key in ["raw_fastq1", "raw_fastq2"]:
            path = result.get(key)
            if path and Path(path).exists():
                Path(path).unlink()

    if not save["trimmed_fastq"]:
        for key in ["trimmed_fastq1", "trimmed_fastq2"]:
            path = result.get(key)
            if path and Path(path).exists():
                Path(path).unlink()

def cleanup_replicate_outputs(rep_result: dict[str, Any], data: dict[str, Any]) -> None:
    save = data["save"]

    if not save["peaks_raw"]:
        path = rep_result.get("raw_peaks")
        if path and Path(path).exists():
            Path(path).unlink()

    if not save["peaks_filtered"]:
        print("\t\t  NOTE: keeping filtered peaks although save option is false; FRiP/IDR/consensus need them.")

    if not save["cpm_bigwig_ip"]:
        path = rep_result.get("ip_cpm_bigwig")
        if path and Path(path).exists():
            Path(path).unlink()

    if not save["cpm_bigwig_control"]:
        paths = rep_result.get("control_cpm_bigwig")
        if paths and not isinstance(paths, list):
            paths = [paths]
        for path in paths or []:
            if path and Path(path).exists():
                Path(path).unlink()

    if not save["fold_enrichment_bigwig"]:
        path = rep_result.get("fold_enrichment_bigwig")
        if path and Path(path).exists():
            Path(path).unlink()

def cleanup_final_bam_files(replicate_results: list[dict[str, Any]], data: dict[str, Any]) -> None:
    """Delete large BAM intermediates after all downstream analyses are complete."""

    save = data["save"]

    for rep_result in replicate_results:
        sample_results = sample_outputs(rep_result)
        if rep_result.get("control") and "source_controls" in rep_result["control"]:
            sample_results.append(("control", rep_result["control"]))

        for role, sample_result in sample_results:

            if not save["sorted_bam"]:
                for key in ["sorted_bam", "sorted_bai"]:
                    path = sample_result.get(key)
                    if path and Path(path).exists():
                        Path(path).unlink()

            keep_filtered = save["filtered_bam"] if role == "ip" else save["control_filtered_bam"]
            if not keep_filtered:
                for key in ["filtered_bam", "filtered_bai"]:
                    path = sample_result.get(key)
                    if path and Path(path).exists():
                        Path(path).unlink()


def cleanup_tmp_dir(data: dict[str, Any]) -> None:
    """Delete experiment temporary files after all downstream analyses are complete."""

    tmp_dir = Path(data["experiment_dir"]) / "tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)


def process_sample(
    data: dict[str, Any],
    rep: dict[str, Any],
    role: str,
    download_futures: dict[tuple[str, str], Future[tuple[Path, Path | None]]] | None = None,
    control_cache: dict[tuple[str, ...], dict[str, Any]] | None = None,
    sample: dict[str, Any] | None = None,
    label_role: str | None = None,
) -> dict[str, Any]:
    experiment_id = str(data["experiment_id"])
    experiment_dir = Path(data["experiment_dir"])
    params = data["params"]
    save = data["save"]

    label_role = label_role or role
    label = sample_label(experiment_id, str(rep["rep_id"]), label_role)
    sample = sample or rep[role]
    layout = get_sample_layout(sample)
    threads = int(params["threads"])
    identity = sample_identity_key(sample)

    if role == "control" and control_cache is not None and identity in control_cache:
        cached = control_cache[identity]
        print(f"{timestamp()}  Reusing processed control for {label}: {cached['label']}")
        return cached

    future = download_futures.get(sample_download_key(rep, label_role)) if download_futures else None
    if future:
        print(f"{timestamp()}  Waiting for FASTQ download for {label} if needed ..")
        fq1, fq2 = future.result()
    else:
        fq1, fq2 = resolve_fastqs(sample, layout, label, experiment_dir, threads)

    raw_fastqc_dir = run_fastqc(fq1, fq2, layout, label, experiment_dir, threads, "raw") if save["fastqc"] else None
    t1, t2, fastp_json, fastp_html = trim_reads(fq1, fq2, layout, label, experiment_dir, params)
    trimmed_fastqc_dir = run_fastqc(t1, t2, layout, label, experiment_dir, threads, "trimmed") if save["fastqc"] else None

    sorted_bam, sorted_bai, bowtie2_log = align_bowtie2(t1, t2, layout, label, role, data)
    filter_metrics = read_filter_metrics(sorted_bam, label, data)
    filtered_bam, filtered_bai, markdup_stats = filter_and_dedup(sorted_bam, layout, label, role, data)
    library_complexity = compute_library_complexity(markdup_stats, params["nrf_thresholds"])
    flagstat_txt = samtools_flagstat(filtered_bam, label, data)
    usable_reads = int(run_capture(f"samtools view -c -F 0x904 {shlex.quote(str(filtered_bam))}"))
    if usable_reads == 0:
        raise ValueError(f"{label}: no mapped reads remain after filtering/deduplication.")
    filter_metrics["usable_reads"] = usable_reads
    spp_txt = compute_nsc_rsc(filtered_bam, label, data, layout)
    spp_fragment_length = parse_spp_fragment_length(spp_txt)

    result = {
        "label": label,
        "layout": layout,
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
        "filtered_bam": filtered_bam,
        "filtered_bai": filtered_bai,
        "markdup_stats": markdup_stats,
        "library_complexity": library_complexity,
        "flagstat": flagstat_txt,
        "spp": spp_txt,
        "spp_fragment_length": spp_fragment_length,
        "filter_metrics": filter_metrics,
        "gdna_library": sample["gdna_library"],
        "assay": sample["assay"],
    }

    cleanup_sample_files(result, sample, role, save)
    if role == "control" and control_cache is not None:
        control_cache[identity] = result
    return result


def process_replicate(
    data: dict[str, Any],
    rep: dict[str, Any],
    download_futures: dict[tuple[str, str], Future[tuple[Path, Path | None]]] | None = None,
    control_cache: dict[tuple[str, ...], dict[str, Any]] | None = None,
    control_cpm_cache: dict[Path, Path] | None = None,
) -> dict[str, Any]:
    rep_id = str(rep["rep_id"])

    print(f"{timestamp()}  Processing replicate {rep_id}: TF pulldown ({data['tf']})")
    ip_outputs = process_sample(data, rep, "ip", download_futures, control_cache)

    controls = control_samples(rep)
    control_outputs: list[dict[str, Any]] = []
    for control_idx, control in enumerate(controls):
        label_role = control_label_role(controls, control_idx)
        print(f"{timestamp()}  Processing replicate {rep_id}: {label_role}")
        control_outputs.append(
            process_sample(
                data,
                rep,
                "control",
                download_futures,
                control_cache,
                sample=control,
                label_role=label_role,
            )
        )
    combined_control = merged_control_bam(control_outputs, rep_id, data)
    macs3_control_bams = [Path(control["filtered_bam"]) for control in control_outputs]
    macs3_layouts = [ip_outputs["layout"], *(control["layout"] for control in control_outputs)]
    macs3_fmt = "BAMPE" if all(layout == "PE" for layout in macs3_layouts) else "BAM"

    fragment_length = ip_outputs["spp_fragment_length"]
    fragment_source = "spp" if fragment_length else "not_required_for_BAMPE"
    if macs3_fmt == "BAM":
        fragment_length, fragment_source = select_fragment_length(ip_outputs["spp"], data, ip_outputs["label"])
    ip_outputs["macs_fallback_fragment_length"] = fragment_length
    ip_outputs["fragment_length_source"] = fragment_source
    raw_peaks, summits = call_peaks_macs3(ip_outputs["filtered_bam"], macs3_control_bams, rep_id, data, macs3_fmt, fragment_length)
    filtered_peaks = greenscreen_filter(raw_peaks, rep_id, data)

    ip_cpm = bam_to_bigwig(ip_outputs["filtered_bam"], ip_outputs["label"], data, ip_outputs["layout"]) if data["save"]["cpm_bigwig_ip"] else None
    control_cpm = None
    if control_outputs and data["save"]["cpm_bigwig_control"]:
        control_cpm = []
        for control in control_outputs:
            control_bam = Path(control["filtered_bam"])
            if control_cpm_cache is not None and control_bam in control_cpm_cache:
                control_bw = control_cpm_cache[control_bam]
                print(f"{timestamp()}  Reusing CPM bigWig for shared control: {control_bw}")
            else:
                control_bw = bam_to_bigwig(control_bam, control["label"], data, control["layout"])
                if control_cpm_cache is not None:
                    control_cpm_cache[control_bam] = control_bw
            control_cpm.append(control_bw)
    fold_bw = fold_enrichment_bigwig(ip_outputs["filtered_bam"], combined_control["filtered_bam"], rep_id, data) if combined_control and data["save"]["fold_enrichment_bigwig"] else None
    if combined_control is None:
        print(f"{timestamp()}  Control CPM / bamCompare tracks skipped for {rep_id}: control_type=none.")

    frip = compute_frip(ip_outputs["filtered_bam"], filtered_peaks, rep_id, data, ip_outputs["layout"])
    fingerprint_pdf, fingerprint_tsv = plot_fingerprint(ip_outputs["filtered_bam"], combined_control["filtered_bam"] if combined_control else None, rep_id, data)

    result = {
        "rep_id": rep_id,
        "ip": ip_outputs,
        "control": combined_control,
        "controls": control_outputs,
        "raw_peaks": raw_peaks,
        "summits": summits,
        "filtered_peaks": filtered_peaks,
        "raw_peak_count": count_lines(raw_peaks),
        "filtered_peak_count": count_lines(filtered_peaks),
        "ip_cpm_bigwig": ip_cpm,
        "control_cpm_bigwig": control_cpm,
        "fold_enrichment_bigwig": fold_bw,
        "frip": frip,
        "fingerprint_pdf": fingerprint_pdf,
        "fingerprint_tsv": fingerprint_tsv,
    }

    limit = int(data["params"]["peak_count_notice_max"])
    result["peak_count_notice"] = "above_notice_max" if limit > 0 and result["filtered_peak_count"] > limit else "none"
    if result["peak_count_notice"] != "none":
        print(f"{timestamp()}  NOTE: {rep_id} has >{limit} filtered peaks; retained for review (no QC gate).")

    cleanup_replicate_outputs(result, data)
    return result


def metric_text(value: Any) -> str:
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return "NA"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def parse_spp_scores(path: Path) -> dict[str, float | None]:
    scores: dict[str, float | None] = {"NSC": None, "RSC": None}
    if path.is_file():
        for line in path.read_text(errors="replace").splitlines():
            fields = line.split("\t")
            if line.startswith("#") or len(fields) < 10:
                continue
            for key, index in (("NSC", 8), ("RSC", 9)):
                try:
                    value = float(fields[index].split(",")[0])
                    scores[key] = value if math.isfinite(value) else None
                except ValueError:
                    pass
            break
    return scores


def write_summary(
    data: dict[str, Any], replicate_results: list[dict[str, Any]],
    consensus: Path, consensus_summits: Path, idr_peaks: Path | None,
    pooled_log2_bw: Path | None = None, pooled_macs3_fe_bw: Path | None = None,
) -> Path:
    experiment_id = str(data["experiment_id"])
    reports = Path(data["experiment_dir"]) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    summary = reports / f"{experiment_id}.summary.txt"
    control_used = data["control_type"] != "none"
    frips = [r["frip"]["FRiP"] for r in replicate_results]
    peak_counts = [r["filtered_peak_count"] for r in replicate_results]
    metadata = {key: data[key] for key in (
        "experiment_id", "sample_id", "assay", "control_type", "tf", "tf_family", "gdna_library", "preparation_batch", "layout")}
    metadata.update(
        reference=data["reference"], gdna_metadata=data.get("gdna_metadata", {}),
        control_used=control_used, replicates=len(replicate_results),
        consensus_source=data["consensus_source"], idr_status=data.get("idr_status", "not_run"),
        consensus_peaks=consensus, consensus_summits=consensus_summits,
        idr_peaks=idr_peaks, idr_threshold=data["params"]["idr_threshold"],
        pooled_log2_bigwig=pooled_log2_bw, pooled_macs3_fe_bigwig=pooled_macs3_fe_bw,
        pooled_macs3_pval_bigwig=(pooled_macs3_fe_bw.with_name(pooled_macs3_fe_bw.name.replace(".FE.bw", ".pval.bw")) if pooled_macs3_fe_bw else None),
        control_track_status=("available_or_disabled_by_save" if control_used else "skipped_no_control"),
        background_source=("matched_control" if control_used else "macs_treatment_local_lambda"),
        macs_qvalue=(data["params"]["macs_qvalue_no_control"] if not control_used and data["params"].get("macs_qvalue_no_control") is not None else data["params"]["macs_qvalue"]),
        consensus_peak_count=count_lines(consensus),
        mean_FRiP=mean(frips), sd_FRiP=stdev(frips) if len(frips)>1 else None,
        mean_filtered_peak_count=mean(peak_counts), sd_filtered_peak_count=stdev(peak_counts) if len(peak_counts)>1 else None,
        pooled_fragment_length=data.get("pooled_fragment_length"),
        recoverable_error_count=len(data.get("issues", [])), issues=data.get("issues", []),
    )
    sample_rows = []
    seen: set[str] = set()
    prefix = ["sample", "experiment", "assay", "control_type", "tf", "tf_family", "gdna_library", "preparation_batch", "rep_id", "role", "layout"]
    metric_columns = ["mapped_reads_unfiltered", "mapped_reads_before_mapq", "mapped_reads_after_mapq", "mapq_reads_lost", "mapq_loss_fraction", "organelle_reads", "organelle_fraction", "usable_reads", "total_reads", "nonredundant_reads", "duplicate_reads", "duplicate_rate", "NRF", "status", "estimated_library_size", "spp_fragment_length", "NSC", "RSC"]
    for rep in replicate_results:
        for role, sample in sample_outputs(rep):
            if sample["label"] in seen:
                continue
            seen.add(sample["label"])
            values = {**sample["library_complexity"], **sample["filter_metrics"], **parse_spp_scores(sample["spp"]), "spp_fragment_length": sample["spp_fragment_length"]}
            row = dict(zip(prefix, [sample["label"], experiment_id, data["assay"], data["control_type"], data["tf"], data["tf_family"], data["gdna_library"], data["preparation_batch"], rep["rep_id"], role, sample["layout"]]))
            row.update({key: values.get(key) for key in metric_columns})
            sample_rows.append(row)
    metrics_path = reports / f"{experiment_id}.sample_metrics.tsv"
    with metrics_path.open("w") as fh:
        fh.write("\t".join(prefix + metric_columns) + "\n")
        for row in sample_rows:
            fh.write("\t".join(metric_text(row[key]) for key in prefix + metric_columns) + "\n")
    with summary.open("w") as fh:
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, default=str, ensure_ascii=False)
            fh.write(f"{key}\t{metric_text(value)}\n")
        fh.write(f"genome_index\t{data['reference']['genome_index']}\n")
        fh.write(f"genome_size\t{data['reference']['genome_size']}\n")
        fh.write(f"sample_metrics\t{metrics_path}\n")
        fh.write("\nlibrary_complexity\n")
        fh.write("sample\trole\ttotal_reads\tnonredundant_reads\tduplicate_reads\tduplicate_rate\tNRF\tstatus\testimated_library_size\n")
        for row in sample_rows:
            keys = ["sample", "role", "total_reads", "nonredundant_reads", "duplicate_reads", "duplicate_rate", "NRF", "status", "estimated_library_size"]
            fh.write("\t".join(metric_text(row[key]) for key in keys) + "\n")
        fh.write("\nper_replicate\n")
        fh.write("rep_id\traw_peaks\tfiltered_peaks\tFRiP\tip_bam\tcontrol_bams\tfiltered_peaks_path\tmacs_mode\tfragment_length_bp\tfragment_length_source\tpeak_count_notice\tmapped_reads_before_mapq\tmapped_reads_after_mapq\tmapq_loss_fraction\n")
        for r in replicate_results:
            controls = ",".join(str(c["filtered_bam"]) for c in r["controls"]) or "none"
            calling = data.get("peak_calling", {}).get(r["rep_id"], {})
            row = [r["rep_id"], r["raw_peak_count"], r["filtered_peak_count"], r["frip"]["FRiP"], r["ip"]["filtered_bam"], controls, r["filtered_peaks"], calling.get("mode"), calling.get("fragment_length_bp"), r["ip"].get("fragment_length_source"), r["peak_count_notice"]]
            row.extend(r["ip"]["filter_metrics"].get(key) for key in ("mapped_reads_before_mapq", "mapped_reads_after_mapq", "mapq_loss_fraction"))
            fh.write("\t".join(metric_text(v) for v in row) + "\n")
    payload = {**metadata, "sample_metrics": sample_rows, "replicate_results": replicate_results, "peak_calling": data.get("peak_calling", {}), "greenscreen": data.get("greenscreen", {}), "parameters": data["params"], "save": data["save"]}
    (reports / f"{experiment_id}.summary.json").write_text(json.dumps(payload, indent=2, default=str, ensure_ascii=False) + "\n", encoding="utf-8")
    data["sample_metric_rows"] = sample_rows
    data["replicate_results"] = replicate_results
    data["idr_peaks_count"] = count_lines(idr_peaks) if idr_peaks else None
    print(f"{timestamp()}  Summary: {summary}")
    return summary


def run_experiment(
    data: dict[str, Any],
    download_executor: ThreadPoolExecutor,
    download_futures: dict[tuple[str, str], Future[tuple[Path, Path | None]]] | None = None,
    next_data: dict[str, Any] | None = None,
) -> dict[tuple[str, str], Future[tuple[Path, Path | None]]]:
    start = datetime.now()

    print(f"\n{'=' * 70}")
    print(f"  {data['assay'].upper()}-seq pipeline | experiment: {data['experiment_id']} | TF: {data['tf']}")
    print(f"  control: {data['control_type']} | gDNA library: {data['gdna_library']}")
    print(f"  layout: {data['layout']} | replicates: {len(data['replicates'])}")
    print(f"{'=' * 70}\n")

    # Queue all downloads for this experiment, then warm up two jobs from the next one.
    download_futures = start_downloads(data, download_executor, download_futures)
    next_download_futures = start_downloads(next_data, download_executor, limit=2) if next_data else {}
    control_cache: dict[tuple[str, ...], dict[str, Any]] = {}
    control_cpm_cache: dict[Path, Path] = {}
    replicate_results = [
        process_replicate(data, rep, download_futures, control_cache, control_cpm_cache)
        for rep in data["replicates"]
    ]
    peak_files = [r["filtered_peaks"] for r in replicate_results]

    idr_peaks = run_idr(peak_files, data)
    consensus_source = idr_peaks if idr_peaks else peak_files[0]
    data["consensus_source"] = (
        "idr_pairwise_union" if idr_peaks and len(peak_files) > 2 else
        "idr" if idr_peaks else
        "single_replicate_q_peaks" if len(peak_files) == 1 else
        "replicate_1_fallback_idr_unavailable"
    )
    consensus, consensus_summits = make_consensus_peaks(consensus_source, data)
    data["replicate_spp_fragment_lengths"] = [r["ip"]["spp_fragment_length"] for r in replicate_results if r["ip"]["spp_fragment_length"]]
    pooled_log2_bw = pooled_log2_bigwig(replicate_results, data) if data["save"].get("pooled_log2_bigwig", True) else None
    pooled_macs3_fe_bw = pooled_macs3_fe_bigwig(replicate_results, data) if data["save"].get("pooled_macs3_fe_bigwig", True) else None
    write_summary(data, replicate_results, consensus, consensus_summits, idr_peaks, pooled_log2_bw, pooled_macs3_fe_bw)

    cleanup_final_bam_files(replicate_results, data)
    cleanup_tmp_dir(data)

    if data["save"].get("qc_reports", True):
        run_multiqc(data)

    elapsed = datetime.now() - start
    print(f"\n{timestamp()}  Experiment complete: {data['experiment_id']} ({elapsed})")
    # The next loop iteration receives only the futures belonging to its experiment.
    return next_download_futures


def run_multiqc(data: dict[str, Any]) -> Path:
    """Create DAP metadata/QC tables and run MultiQC for this experiment."""
    experiment_dir = Path(data["experiment_dir"])
    reports = experiment_dir / "reports"
    multiqc_dir = experiment_dir / "multiqc"
    multiqc_dir.mkdir(parents=True, exist_ok=True)
    rows = data.get("sample_metric_rows", [])
    if rows:
        path = reports / "dap_sample_metrics_mqc.tsv"
        columns = list(rows[0])
        with path.open("w") as fh:
            fh.write("# plot_type: 'table'\n# section_name: 'DAP-seq sample QC'\n")
            fh.write("# description: 'DAP/ampDAP metadata, MAPQ loss and raw library-complexity metrics. No automatic QC exclusions.'\n")
            fh.write("# headers:\n#   NRF:\n#     format: '{:.3f}'\n#   mapq_loss_fraction:\n#     format: '{:.2%}'\n#   organelle_fraction:\n#     format: '{:.2%}'\n#   duplicate_rate:\n#     format: '{:.2%}'\n")
            fh.write("\t".join(columns) + "\n")
            for row in rows:
                fh.write("\t".join(metric_text(row[c]) for c in columns) + "\n")
    path = reports / "dap_peaks_mqc.tsv"
    columns = ["sample", "experiment", "assay", "control_type", "tf", "gdna_library", "raw_peaks", "filtered_peaks", "FRiP", "idr_peaks", "consensus_source", "peak_count_notice"]
    with path.open("w") as fh:
        fh.write("# plot_type: 'table'\n# section_name: 'DAP-seq MACS3 / IDR'\n")
        fh.write("# description: 'Peak counts, FRiP, control assignment and consensus provenance from the DAP-seq pipeline.'\n")
        fh.write("\t".join(columns) + "\n")
        for rep in data.get("replicate_results", []):
            row = [rep["rep_id"], data["experiment_id"], data["assay"], data["control_type"], data["tf"], data["gdna_library"], rep["raw_peak_count"], rep["filtered_peak_count"], rep["frip"]["FRiP"], data.get("idr_peaks_count"), data["consensus_source"], rep["peak_count_notice"]]
            fh.write("\t".join(metric_text(v) for v in row) + "\n")
    print(f"{timestamp()}  Running MultiQC for {data['experiment_id']}.")
    run_command(f"multiqc {shlex.quote(str(experiment_dir))} -o {shlex.quote(str(multiqc_dir))} -f", multiqc_dir / "multiqc.log")
    return multiqc_dir / "multiqc_report.html"



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Arabidopsis DAP-seq / ampDAP-seq from YAML configs.")
    parser.add_argument("--config-dir", required=True, type=Path, help="Folder with dataset .yaml/.yml configs.")
    parser.add_argument("--outdir", default=Path("results"), type=Path, help="Batch output folder.")
    parser.add_argument("--threads", type=int, help="Override params.threads in every config.")
    parser.add_argument("--validate-only", action="store_true", help="Check configs, local FASTQs and fixed references; do not download or analyze reads.")
    return parser.parse_args()

def check_required_tools(configs: list[dict[str, Any]]) -> None:
    required = {"bowtie2", "samtools", "fastp", "macs3", "bedtools", "plotFingerprint"}
    for cfg in configs:
        save = cfg["save"]
        if save["fastqc"]:
            required.add("fastqc")
        if save["cpm_bigwig_ip"] or (save["cpm_bigwig_control"] and cfg["control_type"] != "none"):
            required.add("bamCoverage")
        if cfg["control_type"] != "none" and (save["fold_enrichment_bigwig"] or save["pooled_log2_bigwig"]):
            required.add("bamCompare")
        if save["pooled_macs3_fe_bigwig"]:
            required.add("bedGraphToBigWig")
        if save["qc_reports"]:
            required.add("multiqc")
        for rep in cfg["replicates"]:
            for sample in [rep["ip"], *control_samples(rep)]:
                if sample["source"] == "sra":
                    required.update({"prefetch", "fasterq-dump", "pigz"})
                elif sample["source"] in {"ena", "url", "zenodo"}:
                    required.add("curl")
        if len(cfg["replicates"]) > 1:
            required.add("idr")
    missing = sorted(tool for tool in required if shutil.which(tool) is None)
    if missing:
        raise FileNotFoundError("Missing command-line tools: " + ", ".join(missing) + ". Activate the analysis environment before running.")
    # SPP is recoverable; missing R/SPP is recorded by compute_nsc_rsc at runtime.


def main() -> None:
    args = parse_args()
    try:
        configs = load_configs(args.config_dir, args.outdir)
        if args.threads is not None:
            if args.threads < 1:
                raise ValueError("--threads must be positive.")
            for cfg in configs:
                cfg["params"]["threads"] = args.threads
        for cfg in configs:
            check_reference_files(cfg)
            for rep in cfg["replicates"]:
                for sample in [rep["ip"], *control_samples(rep)]:
                    if sample["source"] == "local":
                        for key in ["fastq1", *(["fastq2"] if sample["layout"] == "PE" else [])]:
                            path = Path(sample[key])
                            if not path.is_file() or path.stat().st_size == 0:
                                raise FileNotFoundError(f"{cfg['experiment_id']}: missing or empty local FASTQ: {path}")
            effective = {k: cfg[k] for k in ["experiment_id", "sample_id", "assay", "control_type", "tf", "tf_family", "gdna_library", "preparation_batch", "layout", "reference", "params", "save", "replicates"]}
            (Path(cfg["experiment_dir"]) / "config_effective.yaml").write_text(yaml.safe_dump(effective, sort_keys=False), encoding="utf-8")
        print(f"{timestamp()}  Parsed and validated {len(configs)} config(s).")
        if args.validate_only:
            print("Validation complete; no reads downloaded or processed.")
            return
        check_required_tools(configs)
        workers = max(int(cfg["params"]["download_workers"]) for cfg in configs)
        preloaded: dict[tuple[str, str], Future[tuple[Path, Path | None]]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for idx, data in enumerate(configs):
                next_data = configs[idx + 1] if idx + 1 < len(configs) else None
                preloaded = run_experiment(data, executor, preloaded, next_data)
    except (ValueError, RuntimeError, OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
