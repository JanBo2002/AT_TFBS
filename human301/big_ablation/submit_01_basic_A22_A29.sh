#!/usr/bin/env bash
# In big_ablation ablegen und im aktivierten at_env mit bash starten.
# Reicht nur 01_basic_0p1cdo und A22-A29 aus diesem Chat ein.
set -euo pipefail

ABLATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$ABLATION_ROOT"
export ABLATION_PYTHON="${ABLATION_PYTHON:-python3}"

configs=(
  01_basic_0p1cdo
  A22_embed_32_short
  A23_embed_64_short
  A24_embed_128_short
  A25_embed_256_short
  A26_narrow_32_deep_2_short
  A27_wide_128_flat_0_short
  A28_param_32_enc_80_114_128_short
  A29_param_128_enc_56_74_128_short
)

check_only=0
account_options=()
for option in "$@"; do
  case "$option" in
    --check) check_only=1 ;;
    --account=?*)
      if ((${#account_options[@]})); then
        echo "Bitte nur einen --account=NAME angeben." >&2
        exit 2
      fi
      account_options+=("$option")
      ;;
    --help|-h)
      echo "Aufruf: bash submit_01_basic_A22_A29.sh [--check] [--account=NAME]"
      echo "--check: Configs und Embedder-Unterstützung prüfen, ohne Jobs einzureichen."
      exit 0
      ;;
    *) echo "Unbekanntes Argument: $option (siehe --help)." >&2; exit 2 ;;
  esac
done

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  echo "Dieses Skript mit bash auf dem Login-Knoten starten." >&2
  exit 2
fi
if [[ ! -f submit_array.sh ]]; then
  echo "submit_array.sh fehlt. Dieses Skript im Ordner big_ablation ablegen." >&2
  exit 2
fi
for config in "${configs[@]}"; do
  if [[ ! -f "configs/$config.json" ]]; then
    echo "Abgebrochen vor dem Einreichen: configs/$config.json fehlt." >&2
    exit 2
  fi
done

# Kurze Prüfung auf der CPU: keine Trainingsläufe und keine Quelldateiänderungen.
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
"$ABLATION_PYTHON" -B - "${configs[@]}" <<'CHECK_CONFIGS'
import ast
import inspect
from pathlib import Path
import sys

root = Path.cwd()


def require(condition, message):
    if not condition:
        raise ValueError(message)


try:
    import torch
    import model
    import training

    torch.set_num_threads(1)
    require("embedder_residual_blocks" in model.ModelConfig.__dataclass_fields__,
            "model.py unterstützt die Embedder-Tiefe für A26/A27 noch nicht.")
    forwarding = ast.parse(inspect.getsource(training.train_command))
    require(any(isinstance(node, ast.keyword) and node.arg == "embedder_residual_blocks"
                for node in ast.walk(forwarding)),
            "training.py gibt embedder_residual_blocks noch nicht an ModelConfig weiter.")

    for name in sys.argv[1:]:
        config = training.load_config(root / "configs" / (name + ".json"))
        profile = "long" if name == "01_basic_0p1cdo" else "short"
        require(config["name"] == name, f"Abweichender Config-Name: {name}")
        require(config.get("task_profile") == profile,
                f"{name}: task_profile muss {profile!r} sein.")
        require(config["seeds"] == [13], f"{name}: erwartet wird nur Seed 13.")
        require(config["fitting"]["params"]["epochs"] == 100,
                f"{name}: erwartet werden maximal 100 Epochen.")
        params = config["predictor"]["params"]
        require(params.get("dropout") == 0.1 and params.get("channel_dropout") == 0.1,
                f"{name}: dropout und channel_dropout müssen jeweils 0.1 sein.")
        expected_depth = (2 if name.startswith("A26_") else
                          0 if name.startswith("A27_") else 1)
        require(params.get("embedder_residual_blocks", 1) == expected_depth,
                f"{name}: erwartet werden {expected_depth} Embedder-Residualblöcke.")
        net = training.invoke(config["predictor"], None).cpu()
        require(getattr(net.embedder, "residual_blocks", None) == expected_depth,
                f"{name}: das Modell verwendet nicht die konfigurierte Embedder-Tiefe.")
        require(len(net.embedder.additional_residuals) == max(0, expected_depth - 1),
                f"{name}: unpassende Zahl zusätzlicher Embedder-Blöcke.")
        require(isinstance(net.embedder.residual, torch.nn.Identity) == (expected_depth == 0),
                f"{name}: der erste Embedder-Residualblock passt nicht zur Tiefe.")
        print(f"OK: {name} | Profil {profile} | Embedder-Tiefe {expected_depth}", flush=True)
        del net
except Exception as error:
    raise SystemExit(f"Abgebrochen vor dem Einreichen: {error}\n"
                     "Bei Importfehlern bitte zuerst at_env aktivieren.")
CHECK_CONFIGS

if ((check_only)); then
  echo "Alle neun Configs geprüft; keine Jobs eingereicht."
  exit 0
fi

echo "Reiche alle neun Configs gemeinsam über submit_array.sh ein."
exec bash "$ABLATION_ROOT/submit_array.sh" "${configs[@]}" "${account_options[@]}"
