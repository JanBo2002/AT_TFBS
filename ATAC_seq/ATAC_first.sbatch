#!/usr/bin/env bash
#SBATCH --job-name=ATAC_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --chdir=/data/projects/SFB_A03/jan/AT_TFBS/ATAC_seq
#SBATCH --output=/data/projects/SFB_A03/jan/AT_TFBS/ATAC_seq/slurm/%x-%j.out
#SBATCH --error=/data/projects/SFB_A03/jan/AT_TFBS/ATAC_seq/slurm/%x-%j.err


set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/data/projects/SFB_A03/jan/AT_TFBS/ATAC_seq}"
CONFIG_DIR="$PROJECT_DIR/configs"
SCRIPT="$PROJECT_DIR/ATAC-analysis_script.py"
OUTDIR="$PROJECT_DIR/raw_data/ATAC"
CONDA_ROOT="${CONDA_ROOT:-/home/jbossenz/miniforge3}"
CONDA_ENV="${CONDA_ENV:-at_env}"
THREADS="${SLURM_CPUS_PER_TASK:-8}"
CONFIGS="${CONFIGS:-}"              # leer = alle Configs in CONFIG_DIR; sonst Dateistaemme, durch Leerzeichen getrennt
SKIP_EXISTING="${SKIP_EXISTING:-1}" # 1 = fertige Experimente ueberspringen
SUMMARY_ONLY="${SUMMARY_ONLY:-0}"   # 1 = keine Verarbeitung, nur Batch-Auswertung neu bauen

fail() { echo "FEHLER: $*" >&2; exit 1; }
[[ -f "$SCRIPT" ]] || fail "Skript fehlt: $SCRIPT"
[[ -d "$CONFIG_DIR" ]] || fail "Config-Ordner fehlt: $CONFIG_DIR"
[[ -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]] || fail "Conda fehlt: $CONDA_ROOT"

# Eigene Miniforge-Umgebung auch im nicht-interaktiven SLURM-Job aktivieren.
unset PYTHONHOME PYTHONPATH || true
set +u
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export TMPDIR="$OUTDIR/tmp_slurm"

cd "$PROJECT_DIR"
mkdir -p "$OUTDIR" "$TMPDIR"
echo "Start: $(date -Is) | Host: $(hostname) | Job: ${SLURM_JOB_ID:-manual}"
echo "Conda: $CONDA_ENV | Python: $(command -v python) | Threads: $THREADS"
echo "Ausgabe: $OUTDIR"

if [[ "$SUMMARY_ONLY" == "1" ]]; then
    python "$SCRIPT" --outdir "$OUTDIR" --summary-only
    echo "Fertig (nur Batch-Auswertung): $(date -Is)"
    exit 0
fi

# Fehlende Programme vor den Downloads erkennen.
missing=()
for tool in fastqc fastp bowtie2 bowtie2-inspect samtools macs3 bedtools \
            bamCoverage alignmentSieve computeMatrix plotProfile plotFingerprint \
            idr multiqc prefetch fasterq-dump pigz java; do
    command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
[[ ${#missing[@]} -eq 0 ]] || fail "Programme fehlen in $CONDA_ENV/PATH: ${missing[*]}"
python -c 'import yaml, matplotlib; print("PyYAML und matplotlib: OK")'
command -v ataqv >/dev/null 2>&1 || echo "Hinweis: ataqv fehlt; dieser optionale QC-Schritt wird uebersprungen."

# Config-Auswahl: ohne CONFIGS laeuft der ganze Ordner; mit CONFIGS werden nur die genannten
# Dateistaemme (ohne .yml/.yaml) in einen temporaeren Ordner ausserhalb von OUTDIR kopiert.
RUN_CONFIG_DIR="$CONFIG_DIR"
if [[ -n "$CONFIGS" ]]; then
    RUN_CONFIG_DIR="$(mktemp -d "${TMPDIR}/selected-configs-${SLURM_JOB_ID:-manual}.XXXXXX")"
    trap 'rm -rf -- "$RUN_CONFIG_DIR"' EXIT
    for stem in $CONFIGS; do
        config=""
        for name in "$stem.yml" "$stem.yaml"; do
            [[ -f "$CONFIG_DIR/$name" ]] && { config="$CONFIG_DIR/$name"; break; }
        done
        [[ -n "$config" ]] || fail "Config fehlt: $CONFIG_DIR/$stem.yml bzw. .yaml"
        cp -- "$config" "$RUN_CONFIG_DIR/"
        echo "Config: $config"
    done
fi

# Dieselbe Config-Validierung wie in der Pipeline (inkl. Referenzdateien aus DEFAULT_REFERENCE),
# zusaetzlich Vollstaendigkeit des Bowtie2-Index und Abgleich der Chromosomennamen
# (keep_chroms + organelle_chroms) mit dem Index, bevor FASTQs heruntergeladen werden.
python - "$SCRIPT" "$RUN_CONFIG_DIR" <<'PY'
import runpy
import subprocess
import sys
from pathlib import Path

pipeline = runpy.run_path(sys.argv[1], run_name="atac_preflight")
config_dir = Path(sys.argv[2])
configs = sorted(p for p in config_dir.iterdir() if p.suffix.lower() in (".yml", ".yaml"))
if not configs:
    raise SystemExit(f"FEHLER: keine Configs in {config_dir}")
checked_indexes = set()
for path in configs:
    try:
        cfg = pipeline["read_yaml"](path)
        pipeline["merge_defaults"](cfg)
        pipeline["validate_config"](cfg)
        index = str(cfg["reference"]["genome_index"])
        parts = ("1", "2", "3", "4", "rev.1", "rev.2")
        if not any(all(Path(f"{index}.{part}.{suffix}").is_file() for part in parts)
                   for suffix in ("bt2", "bt2l")):
            raise ValueError(f"Bowtie2-Index fehlt oder ist unvollstaendig: {index}")
        if index not in checked_indexes:
            # bowtie2-inspect gibt die vollstaendige FASTA-Kopfzeile aus (z. B. "1 dna:chromosome ...");
            # Bowtie2 selbst benutzt im SAM nur das erste Wort, deshalb hier ebenfalls.
            names = {line.split()[0] for line in subprocess.check_output(
                ["bowtie2-inspect", "--names", index], text=True).splitlines() if line.strip()}
            expected = set(cfg["params"]["keep_chroms"] + cfg["params"]["organelle_chroms"])
            if absent := sorted(expected - names):
                raise ValueError(f"Chromosomen fehlen im Index: {', '.join(absent)}; "
                                 "Index und keep_chroms/organelle_chroms im Skript pruefen.")
            checked_indexes.add(index)
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"FEHLER in {path.name}: {exc}") from None
    print(f"Config und Referenz: OK ({path.name})")
PY

# Ein Aufruf fuer alle ausgewaehlten Experimente: Verarbeitung nacheinander, danach gemeinsame
# QC-Tabelle, Peakvergleich, ataqv-Viewer und MultiQC-Bericht ueber alles in OUTDIR.
skip_flag=()
python "$SCRIPT" \
    --config-dir "$RUN_CONFIG_DIR" \
    --outdir "$OUTDIR" \
    --threads "$THREADS"

echo "Fertig: $(date -Is)"
