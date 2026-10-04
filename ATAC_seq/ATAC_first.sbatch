#!/usr/bin/env bash
# Auf dem Login-Knoten starten: bash ATAC_first.sbatch
# Der Launcher reicht ein Array (eine YAML pro Task) und danach einen QC-Job ein.
# Diese Ressourcen gelten fuer die Array-Tasks, nicht fuer den Launcher.
#SBATCH --job-name=ATAC_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --chdir=/data/projects/SFB_A03/jan/AT_TFBS/ATAC_seq

set -euo pipefail

MODE="${1:---launch}"
PROJECT_DIR="${PROJECT_DIR:-/data/projects/SFB_A03/jan/AT_TFBS/ATAC_seq}"
CONFIG_DIR="${CONFIG_DIR:-$PROJECT_DIR/configs}"
SCRIPT="${SCRIPT:-$PROJECT_DIR/ATAC-analysis_script.py}"
OUTDIR="${OUTDIR:-/data/projects/SFB_A03/jan/AT_TFBS/raw_data/ATAC}"
CONDA_ROOT="${CONDA_ROOT:-/home/jbossenz/miniforge3}"
CONDA_ENV="${CONDA_ENV:-at_env}"
PARTITION="${PARTITION:-cpu}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
JOB_MEM="${JOB_MEM:-32G}"
JOB_TIME="${JOB_TIME:-24:00:00}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
SUMMARY_CPUS="${SUMMARY_CPUS:-4}"
SUMMARY_MEM="${SUMMARY_MEM:-16G}"
SUMMARY_TIME="${SUMMARY_TIME:-04:00:00}"
CONFIGS="${CONFIGS:-}"             # leer: alle YAMLs; sonst Dateistaemme mit Leerzeichen
SKIP_EXISTING="${SKIP_EXISTING:-1}" # 0: auch bereits fertige Datensaetze neu verarbeiten
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"   # 1: nur einen Job fuer die Batch-Auswertung einreichen
LOG_DIR="$PROJECT_DIR/slurm"

fail() { echo "FEHLER: $*" >&2; exit 1; }

activate_env() {
    [[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]] || fail "Conda fehlt: $CONDA_ROOT"
    unset PYTHONHOME PYTHONPATH || true
    set +u
    source "$CONDA_ROOT/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    set -u
    export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
}

check_tools() {
    local tool
    local -a missing=()
    for tool in fastqc fastp bowtie2 bowtie2-inspect samtools macs3 bedtools \
                bamCoverage alignmentSieve computeMatrix plotProfile plotFingerprint \
                idr multiqc prefetch fasterq-dump pigz java; do
        command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
    done
    [[ ${#missing[@]} -eq 0 ]] || fail "Programme fehlen in $CONDA_ENV/PATH: ${missing[*]}"
    python -c 'import yaml, matplotlib; print("PyYAML und matplotlib: OK")'
    command -v ataqv >/dev/null 2>&1 || echo "Hinweis: ataqv fehlt; optionaler QC-Schritt wird uebersprungen."
}

setup_job() {
    THREADS="${SLURM_CPUS_PER_TASK:-$CPUS_PER_TASK}"
    export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"
    mkdir -p "$OUTDIR/tmp_slurm"
    TMPDIR="$(mktemp -d "$OUTDIR/tmp_slurm/job-${SLURM_JOB_ID:-manual}-${SLURM_ARRAY_TASK_ID:-summary}.XXXXXX")"
    export TMPDIR
    cd "$PROJECT_DIR"
    echo "Start: $(date -Is) | Host: $(hostname) | Job: ${SLURM_JOB_ID:-manual} | Threads: $THREADS"
}

if [[ "$MODE" == "--worker" || "$MODE" == "--summary" ]]; then
    [[ $# -eq 2 ]] || fail "Interner Aufruf braucht einen Manifest-Pfad."
    MANIFEST="$2"
    [[ -f "$MANIFEST" ]] || fail "Manifest fehlt: $MANIFEST"
    LAUNCH_DIR="$(dirname "$MANIFEST")"
    SCRIPT="$LAUNCH_DIR/ATAC-analysis_script.py"
    [[ -f "$SCRIPT" ]] || fail "Pipeline-Snapshot fehlt: $SCRIPT"
    activate_env
    setup_job
    if [[ "$MODE" == "--worker" ]]; then
        [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[1-9][0-9]*$ ]] || fail "Worker muss als Array-Task laufen."
        finish_worker() {
            local rc=$?
            trap - EXIT
            printf '%s\n' "$rc" > "$LAUNCH_DIR/status/task-${SLURM_ARRAY_TASK_ID}.exitcode"
            rm -rf -- "$TMPDIR"
            exit "$rc"
        }
        trap finish_worker EXIT
        check_tools
        # Nur den Experiment-Treiber der unveraenderten Pipeline aufrufen.
        # main() wuerde auch gemeinsame Batch-Dateien schreiben und parallel kollidieren.
        python - "$SCRIPT" "$MANIFEST" "$OUTDIR" "$THREADS" "$SLURM_ARRAY_TASK_ID" "$SKIP_EXISTING" <<'PY'
import json
import runpy
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

script, manifest, outdir, threads, task_id, skip = sys.argv[1:]
entry = json.loads(Path(manifest).read_text())["configs"][int(task_id) - 1]
pipeline = runpy.run_path(script, run_name="atac_array_worker")
cfg = pipeline["load_config"](Path(entry["path"]), Path(outdir))
cfg["params"]["threads"] = int(threads)
print(f"Array task {task_id}: {cfg['experiment_id']} ({entry['path']})", flush=True)
summary = Path(cfg["experiment_dir"]) / "reports" / f"{cfg['experiment_id']}.summary.txt"
if skip == "1" and summary.is_file():
    print(f"Bereits verarbeitet, ueberspringe: {summary}", flush=True)
else:
    with ThreadPoolExecutor(max_workers=int(cfg["params"]["download_workers"])) as executor:
        pipeline["run_experiment"](cfg, executor)
PY
    else
        trap 'rm -rf -- "$TMPDIR"' EXIT
        # afterany: erst nachdem ALLE Array-Tasks beendet sind, auch bei Fehlern.
        python "$SCRIPT" --outdir "$OUTDIR" --summary-only
        python - "$MANIFEST" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
entries = json.loads(manifest.read_text())["configs"]
failed = []
for i, entry in enumerate(entries, 1):
    status = manifest.parent / "status" / f"task-{i}.exitcode"
    code = status.read_text().strip() if status.is_file() else "missing (e.g. cancelled)"
    print(f"Task {i}: {entry['experiment_id']}: exitcode={code}")
    if code != "0":
        failed.append(entry["experiment_id"])
if failed:
    raise SystemExit("Batch-Bericht erstellt, aber Tasks fehlgeschlagen/unvollstaendig: " + ", ".join(failed))
PY
    fi
    echo "Fertig: $(date -Is)"
    exit 0
fi

[[ "$MODE" == "--launch" && $# -le 1 ]] || fail "Start: bash ATAC_first.sbatch"
[[ -z "${SLURM_JOB_ID:-}" ]] || fail "Launcher mit bash auf dem Login-Knoten starten, nicht mit sbatch oder innerhalb einer Allocation."
command -v sbatch >/dev/null 2>&1 || fail "sbatch fehlt. Auf dem Slurm-Login-Knoten starten."
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || fail "MAX_PARALLEL muss positiv sein."
[[ "$CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]] || fail "CPUS_PER_TASK muss positiv sein."
[[ "$SUMMARY_CPUS" =~ ^[1-9][0-9]*$ ]] || fail "SUMMARY_CPUS muss positiv sein."
[[ "$SKIP_EXISTING" =~ ^[01]$ && "$SUMMARY_ONLY" =~ ^[01]$ ]] || fail "SKIP_EXISTING und SUMMARY_ONLY muessen 0 oder 1 sein."
[[ -d "$PROJECT_DIR" && -f "$SCRIPT" ]] || fail "Projekt/Pipeline fehlt: $PROJECT_DIR / $SCRIPT"
[[ "$SUMMARY_ONLY" == "1" || -d "$CONFIG_DIR" ]] || fail "Config-Ordner fehlt: $CONFIG_DIR"
activate_env
mkdir -p "$LOG_DIR" "$OUTDIR"
LAUNCH_DIR="$(mktemp -d "$LOG_DIR/atac-launch-$(date +%Y%m%d-%H%M%S).XXXXXX")"
mkdir -p "$LAUNCH_DIR/configs" "$LAUNCH_DIR/status"
cp -- "${BASH_SOURCE[0]}" "$LAUNCH_DIR/ATAC_first.sbatch"
cp -- "$SCRIPT" "$LAUNCH_DIR/ATAC-analysis_script.py"
MANIFEST="$LAUNCH_DIR/manifest.json"
cd "$PROJECT_DIR"

if [[ "$SUMMARY_ONLY" == "1" ]]; then
    printf '{"configs": []}\n' > "$MANIFEST"
else
    check_tools
    # Alle Configs vor der ersten Einreichung pruefen und unveraenderlich kopieren.
    # Gleicher Index kann je Config unterschiedliche Chromosomen-Listen haben.
    python - "$SCRIPT" "$CONFIG_DIR" "$CONFIGS" "$LAUNCH_DIR" <<'PY'
import json
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

script, config_dir, selection, launch_dir = sys.argv[1:]
pipeline = runpy.run_path(script, run_name="atac_launcher_preflight")
config_dir, launch_dir = Path(config_dir), Path(launch_dir)
paths = []
if selection.strip():
    for stem in selection.split():
        matches = [config_dir / f"{stem}{suffix}" for suffix in (".yml", ".yaml")
                   if (config_dir / f"{stem}{suffix}").is_file()]
        if len(matches) != 1:
            raise SystemExit(f"FEHLER: Config fehlt oder mehrdeutig: {stem}")
        paths.append(matches[0])
else:
    paths = sorted(p for p in config_dir.iterdir() if p.suffix.lower() in (".yml", ".yaml"))
if not paths:
    raise SystemExit(f"FEHLER: keine Configs in {config_dir}")
seen_ids, index_names, entries = set(), {}, []
for path in paths:
    try:
        cfg = pipeline["read_yaml"](path)
        pipeline["merge_defaults"](cfg)
        pipeline["validate_config"](cfg)
        exp_id = str(cfg["experiment_id"])
        if not exp_id or exp_id in (".", "..") or "/" in exp_id or exp_id in seen_ids:
            raise ValueError(f"ungueltige/doppelte experiment_id: {exp_id!r}")
        seen_ids.add(exp_id)
        index = str(cfg["reference"]["genome_index"])
        parts = ("1", "2", "3", "4", "rev.1", "rev.2")
        if not any(all(Path(f"{index}.{part}.{suffix}").is_file() for part in parts)
                   for suffix in ("bt2", "bt2l")):
            raise ValueError(f"Bowtie2-Index unvollstaendig: {index}")
        if index not in index_names:
            index_names[index] = {line.split()[0] for line in subprocess.check_output(
                ["bowtie2-inspect", "--names", index], text=True).splitlines() if line.strip()}
        expected = set(cfg["params"]["keep_chroms"] + cfg["params"]["organelle_chroms"])
        if absent := sorted(expected - index_names[index]):
            raise ValueError(f"Chromosomen fehlen im Index: {', '.join(absent)}")
        for rep in cfg["replicates"]:
            sample = rep["sample"]
            if sample["source"] == "local":
                for key in ("fastq1", "fastq2") if sample["layout"] == "PE" else ("fastq1",):
                    if not Path(sample[key]).is_file() or Path(sample[key]).stat().st_size == 0:
                        raise ValueError(f"Lokale FASTQ fehlt/ist leer: {sample[key]}")
        snapshot = launch_dir / "configs" / path.name
        shutil.copy2(path, snapshot)
        entries.append({"path": str(snapshot), "experiment_id": exp_id})
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"FEHLER in {path.name}: {exc}") from None
    print(f"Task {len(entries)}: {exp_id} | Config und Referenz: OK")
(launch_dir / "manifest.json").write_text(json.dumps({"configs": entries}, indent=2) + "\n")
PY
fi

# sbatch uebernimmt die gesetzten Einstellungen fuer Worker und Summary-Job.
export PROJECT_DIR CONFIG_DIR OUTDIR CONDA_ROOT CONDA_ENV PARTITION CPUS_PER_TASK
export SKIP_EXISTING
common=(--parsable --partition="$PARTITION" --nodes=1 --ntasks=1 --chdir="$PROJECT_DIR" --export=ALL)
dependency=()
if [[ "$SUMMARY_ONLY" != "1" ]]; then
    count="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["configs"]))' "$MANIFEST")"
    array_id="$(sbatch "${common[@]}" --job-name=ATAC_dataset \
        --array="1-${count}%${MAX_PARALLEL}" --cpus-per-task="$CPUS_PER_TASK" \
        --mem="$JOB_MEM" --time="$JOB_TIME" \
        --output="$LOG_DIR/ATAC_dataset-%A_%a.out" --error="$LOG_DIR/ATAC_dataset-%A_%a.err" \
        "$LAUNCH_DIR/ATAC_first.sbatch" --worker "$MANIFEST")"
    array_id="${array_id%%;*}"
    [[ "$array_id" =~ ^[0-9]+$ ]] || fail "Unerwartete Array-Job-ID: $array_id"
    dependency=(--dependency="afterany:$array_id")
    echo "Array eingereicht: $array_id | $count Datensaetze | maximal $MAX_PARALLEL parallel"
fi
if ! summary_id="$(sbatch "${common[@]}" "${dependency[@]}" --job-name=ATAC_batch_QC \
    --cpus-per-task="$SUMMARY_CPUS" --mem="$SUMMARY_MEM" --time="$SUMMARY_TIME" \
    --output="$LOG_DIR/ATAC_batch_QC-%j.out" --error="$LOG_DIR/ATAC_batch_QC-%j.err" \
    "$LAUNCH_DIR/ATAC_first.sbatch" --summary "$MANIFEST")"; then
    fail "Batch-QC-Job konnte nicht eingereicht werden. Bereits eingereichte Array-Tasks laufen weiter. Danach: SUMMARY_ONLY=1 bash ATAC_first.sbatch"
fi
echo "Batch-QC-Job: ${summary_id%%;*}"
echo "Config-/Pipeline-Snapshot und Task-Zuordnung: $MANIFEST"
echo "Ergebnisse: $OUTDIR | Logs: $LOG_DIR"
