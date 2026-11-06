#!/usr/bin/env bash
# Version: 2026-11-06-block-b-guest-launcher-4
#
# Reicht Configs mit einem Befehl über das unveränderte submit_array.sh ein – auch von
# einem zweiten Nutzerkonto aus, das den Projektordner und die Conda-Umgebung at_env
# mitbenutzt. Die Jobs laufen unter dem Konto, das dieses Script startet, gleichzeitig mit
# den Arrays des anderen Kontos; die Ergebnisse landen wie gewohnt in results/.
#
#   bash submit_block_b.sh                  # alle Block-B-Configs (B01, B03–B12)
#   bash submit_block_b.sh --dry-run        # nur prüfen, nichts einreichen
#   bash submit_block_b.sh --account=NAME   # mit Slurm-Account
#   bash submit_block_b.sh B04_gelu_middle B05_silu_middle   # eigene Auswahl
#
# Übersprungen werden Configs mit vorhandenem results-Ordner und Configs, für die schon
# ein Array läuft/wartet (auch unter dem anderen Konto, soweit squeue es anzeigt).
# Conda: JAN_CONDA_ROOT (gesetzt) > configs/conda_root.txt (vom Installer) > DEFAULT_CONDA_ROOT.
set -euo pipefail

ABLATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_CONDA_ROOT="/home/jbossenz/miniforge3"

dry_run=0
account_option=()
configs=()
for value in "$@"; do
    case "$value" in
        --dry-run) dry_run=1 ;;
        --account=?*) account_option=("$value") ;;
        --help|-h) sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
        -*) echo "Unbekannte Option: $value" >&2; exit 2 ;;
        *) configs+=("$value") ;;
    esac
done
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Abgebrochen: auf dem Login-Knoten starten, nicht in einem Slurm-Job." >&2
    exit 1
fi

# --- Umgebung: immer at_env des Projektbesitzers, unabhängig vom startenden Konto.
if [[ -z "${JAN_CONDA_ROOT:-}" && -r "$ABLATION_ROOT/configs/conda_root.txt" ]]; then
    JAN_CONDA_ROOT="$(head -n 1 "$ABLATION_ROOT/configs/conda_root.txt")"
fi
export JAN_CONDA_ROOT="${JAN_CONDA_ROOT:-$DEFAULT_CONDA_ROOT}"
PY="${ABLATION_PYTHON:-$JAN_CONDA_ROOT/envs/at_env/bin/python}"
# Conda-/Python-Einstellungen des startenden Kontos würden in die Jobs vererbt
# (submit_array.sh reicht mit --export=ALL ein) und dürfen at_env nicht überlagern.
unset PYTHONPATH PYTHONHOME CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL CONDA_EXE CONDA_PYTHON_EXE \
      CONDA_PROMPT_MODIFIER _CE_M _CE_CONDA
for index in 1 2 3 4 5 6 7 8 9; do unset "CONDA_PREFIX_$index"; done
export PYTHONNOUSERSITE=1
export ABLATION_PYTHON="$PY"
export PATH="$(dirname -- "$PY"):$PATH"
# Neue Dateien gruppenweit les-/schreibbar, wo keine Default-ACLs wirken (Slurm übernimmt die umask).
umask 002

echo "Projekt:      $ABLATION_ROOT"
echo "Conda-Wurzel: $JAN_CONDA_ROOT"
echo "Interpreter:  $PY"
echo "Konto:        $(id -un)"

if [[ ! -x "$PY" ]]; then
    echo "Abgebrochen: Python-Interpreter nicht ausführbar: $PY" >&2
    echo "Entweder fehlen Rechte auf die Conda-Umgebung (siehe Freigabe) oder JAN_CONDA_ROOT stimmt nicht." >&2
    exit 1
fi
for executable in sbatch squeue; do
    command -v "$executable" >/dev/null 2>&1 || { echo "Abgebrochen: $executable fehlt." >&2; exit 1; }
done
for file in submit_array.sh run_experiment.py training.py model.py data.py train_array.sbatch collect_config.sbatch configs/caches.csv; do
    [[ -r "$ABLATION_ROOT/$file" ]] || { echo "Abgebrochen: $file fehlt oder ist nicht lesbar." >&2; exit 1; }
done

# --- Job-Skripte: Conda-Aktivierung muss unter diesem Konto genauso funktionieren.
home_conda='(\$HOME|\$\{HOME\}|~)/[^ "]*(miniforge|miniconda|anaconda|conda)'
uses_conda=0
for file in train_array.sbatch collect_config.sbatch; do
    script="$ABLATION_ROOT/$file"
    # Kommentare ignorieren; $HOME als Default in ${JAN_CONDA_ROOT:-...} oder ${JAN_CONDA_ROOT:=...} ist unkritisch (JAN_CONDA_ROOT ist exportiert).
    offending="$(grep -nE "$home_conda" "$script" | grep -vE '^[0-9]+:[[:space:]]*#' | grep -vE 'JAN_CONDA_ROOT:?[-=]' || true)"
    if [[ "${SKIP_SBATCH_CHECK:-0}" != "1" && -n "$offending" ]]; then
        echo "Abgebrochen: $file sucht Conda im Home des startenden Kontos:" >&2
        printf '%s\n' "$offending" >&2
        echo "Unter einem anderen Konto zeigt das ins falsche Home. Die Zeile auf \$JAN_CONDA_ROOT oder einen" >&2
        echo "absoluten Pfad umstellen (oder, falls sie nie greift, mit SKIP_SBATCH_CHECK=1 erneut starten)." >&2
        exit 1
    fi
    if [[ -n "$(grep -vE '^[[:space:]]*#' "$script" | grep -E 'conda|JAN_CONDA_ROOT' || true)" ]]; then uses_conda=1; fi
done
if (( uses_conda )); then
    # Die Aktivierung, die die Job-Skripte verwenden, einmal mit diesem Konto ausführen.
    if ! check="$(bash -c 'set -e; source "$JAN_CONDA_ROOT/etc/profile.d/conda.sh"; conda activate "$JAN_CONDA_ROOT/envs/at_env"; python -c "import sys, torch, numpy; print(sys.executable)"' 2>&1)"; then
        echo "Abgebrochen: Die Conda-Aktivierung der Job-Skripte scheitert unter Konto $(id -un):" >&2
        printf '%s\n' "$check" | tail -n 5 >&2
        echo "Prüfen: Freigabe auf $JAN_CONDA_ROOT (siehe Anleitung) und ob JAN_CONDA_ROOT stimmt." >&2
        exit 1
    fi
    echo "Job-Umgebung: source \$JAN_CONDA_ROOT/etc/profile.d/conda.sh; conda activate \$JAN_CONDA_ROOT/envs/at_env -> $(printf '%s\n' "$check" | tail -n 1)"
fi

# --- Vorprüfung und Einreichung (Python aus at_env; reicht über submit_array.sh ein).
exec "$PY" - "$ABLATION_ROOT" "$dry_run" ${account_option[@]+"${account_option[@]}"} -- ${configs[@]+"${configs[@]}"} <<'LAUNCHER'
import csv, hashlib, importlib, json, os, pwd, stat, subprocess, sys, tempfile
from pathlib import Path

DEFAULT_CONFIGS = [
    "B01_no_decoder_middle", "B03_groupnorm_middle", "B04_gelu_middle", "B05_silu_middle",
    "B06_avgpool_middle", "B07_attention_pool_middle", "B08_se_max_middle", "B09_bottleneck128_middle",
    "B10_encoder_64_64_64_middle", "B11_encoder_96_128_192_middle", "B12_stochastic_depth_0p1_middle",
]
root, dry_run = Path(sys.argv[1]), sys.argv[2] == "1"
separator = sys.argv.index("--")
account, names = sys.argv[3:separator], sys.argv[separator + 1:] or DEFAULT_CONFIGS
me = pwd.getpwuid(os.getuid()).pw_name
problems, notes = [], []

# Python-Umgebung: torch-Build und alle Projektmodule mit ihren Abhängigkeiten importierbar?
try:
    import torch
    build = ("CUDA " + torch.version.cuda) if torch.version.cuda else (
        ("ROCm " + torch.version.hip) if getattr(torch.version, "hip", None) else None)
    print(f"torch:        {torch.__version__} ({build or 'nur CPU-Build'})")
    if build is None:
        notes.append("torch in at_env ist ein reiner CPU-Build; GPU-Training wäre so nicht möglich")
except Exception as error:
    problems.append(f"torch ist mit diesem Interpreter nicht importierbar: {error}")
sys.path.insert(0, str(root))
sys.dont_write_bytecode = True
for module in ("model", "data", "training"):
    try:
        importlib.import_module(module)
    except Exception as error:
        problems.append(f"{module}.py ist nicht importierbar: {type(error).__name__}: {error}")

# Block-B-Code installiert? (Programmdateien = Stand eines Manifests in certified_code/)
installed = False
for manifest in sorted((root / "certified_code").glob("*.json")) if (root / "certified_code").is_dir() else []:
    try:
        files = json.loads(manifest.read_text())["files"]
        installed = installed or all(hashlib.sha256((root / n).read_bytes()).hexdigest() == spec["sha256"]
                                     for n, spec in files.items())
    except (OSError, ValueError, KeyError):
        pass
if not installed:
    problems.append("Block-B-Code ist nicht installiert (python install_block_b_update.pyz --target ... ausführen)")

# Schreibrechte, die submit_array.sh und die Jobs brauchen
for sub in ("results", "plans", "configs", "results/Results_all"):
    directory = root / sub
    if not directory.is_dir():  # legt submit_array.sh bzw. collect selbst an; nichts vorab erzeugen
        if not os.access(directory.parent, os.W_OK | os.X_OK):
            problems.append(f"keine Schreibrechte in {directory.parent}")
        continue
    try:
        fd, temporary = tempfile.mkstemp(prefix=".write_test_", dir=directory)
        os.close(fd)
        os.unlink(temporary)
    except OSError as error:
        problems.append(f"keine Schreibrechte in {directory}: {error}")
for path in [root / "plans" / ".dispatcher.lock", root / "results" / "Results_all" / "schema.json"]:
    if path.exists() and not os.access(path, os.R_OK | os.W_OK):
        problems.append(f"keine Schreibrechte auf {path}")
unreadable = []
with (root / "configs" / "caches.csv").open(newline="") as handle:
    for row in csv.DictReader(handle):
        if not os.access(row["cache"], os.R_OK):
            unreadable.append(row["cache"])
if unreadable:
    problems.append("Caches nicht lesbar:\n    " + "\n    ".join(sorted(set(unreadable))[:20]))

# Laufende Arrays (alle Konten, soweit squeue sie zeigt): Sammeljob h301_collect_<Config>
active = {}
try:
    result = subprocess.run(["squeue", "--noheader", "--name=" + ",".join("h301_collect_" + n for n in names),
                             "--format=%i|%j|%Z|%u"], capture_output=True, text=True, timeout=90)
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split("|", 3)]
        if (len(fields) == 4 and fields[1].startswith("h301_collect_") and fields[2]
                and Path(fields[2]).resolve() == root.resolve()):
            active.setdefault(fields[1][len("h301_collect_"):], []).append(f"Job {fields[0]} von {fields[3]}")
    if result.returncode:
        notes.append("squeue lieferte einen Fehler; laufende Arrays anderer Konten nicht geprüft")
except (OSError, subprocess.SubprocessError) as error:
    notes.append(f"squeue nicht abfragbar ({error}); laufende Arrays anderer Konten nicht geprüft")

selected, missing, missing_reference = [], [], {}
for name in names:
    path = root / "configs" / (name + ".json")
    if not path.is_file():
        missing.append(name)
        continue
    config = json.loads(path.read_text())
    if config.get("name") != name:
        problems.append(f"Config-Name und Dateiname weichen ab: {path}")
    pwm_root = config.get("predictor", {}).get("params", {}).get("pwm_root")
    if pwm_root and not os.access(pwm_root if os.path.isabs(pwm_root) else root / pwm_root, os.R_OK | os.X_OK):
        problems.append(f"{name}: PWM-Ordner nicht lesbar: {pwm_root}")
    variant = config.get("data", {}).get("params", {}).get("split_variant", "original")
    if variant not in ("original", None):
        notes.append(f"{name}: split_variant {variant} – submit_array.sh bereitet dafür Split-Caches vor "
                     "und braucht Schreibrechte in den Cache-Ordnern")
    if (root / "results" / name).exists():
        print(f"Übersprungen: {name} (results/{name} existiert bereits; Ergänzen nur über submit_array.sh --restart)")
        continue
    if name in active:
        print(f"Übersprungen: {name} (Array läuft/wartet bereits: {', '.join(active[name])})")
        continue
    reference = config.get("reference")
    if reference and reference != name and not (root / "results" / "Results_all" / f"{reference}.csv").is_file():
        missing_reference.setdefault(reference, []).append(name)
    selected.append(name)
if missing:
    problems.append("Configs fehlen in configs/: " + ", ".join(missing))
for reference, dependent in missing_reference.items():
    notes.append(f"Referenz {reference} hat noch keine results/Results_all/{reference}.csv; die Vergleichsspalten von "
                 f"{', '.join(dependent)} bleiben leer, bis sie vorliegt (danach run_experiment.py collect --config <Name>)")
for note in notes:
    print("Hinweis: " + note)
if problems:
    print("Vorprüfung fehlgeschlagen:\n  - " + "\n  - ".join(problems), file=sys.stderr)
    sys.exit(1)
if not selected:
    print("Nichts einzureichen.")
    sys.exit(0)
print("Einzureichen: " + ", ".join(selected))
command = ["bash", str(root / "submit_array.sh"), *selected, *account]
print("Befehl: bash submit_array.sh " + " ".join([*selected, *account]), flush=True)
if dry_run:
    print("Trockenlauf; nichts eingereicht.")
    sys.exit(0)

status = subprocess.run(command, cwd=root, stdin=subprocess.DEVNULL).returncode
# submit_array.sh legt seinen Batch-Ordner mit mkdtemp (nur Besitzer) an; für das andere
# freigegebene Konto die eigenen Batch-Ordner mit Plan und Slurm-Logs öffnen.
for batch in (root / "plans").glob("batch_array_*"):
    try:
        info = batch.lstat()
        if stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid():
            os.chmod(batch, stat.S_IMODE(info.st_mode) | stat.S_IRWXG)
    except OSError:
        pass
if status == 0:
    print(f"Jobs laufen unter Konto {me}, gleichzeitig mit den Arrays anderer Konten; Status: squeue -u {me}")
sys.exit(status)
LAUNCHER
