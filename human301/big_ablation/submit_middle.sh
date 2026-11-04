#!/usr/bin/env bash
# Version: 2026-11-04-middle-migration
# Stellt alle Short-Configs auf das Middle-Profil um (Umbenennung _short -> _middle,
# Metadaten der vorhandenen Läufe, Referenzen) und reicht über submit_array.sh --restart
# nur die fehlenden Läufe ein (6 zusätzliche TFs x 2 Assays je Seed).
#
# Aufruf auf dem Login-Knoten, aus dem big_ablation-Ordner oder mit vollem Pfad:
#   bash submit_middle.sh --dry-run                 # nur prüfen und berichten
#   bash submit_middle.sh --account=NAME            # alle Short-Configs umstellen und einreichen
#   bash submit_middle.sh --account=NAME A22_embed_32_short A26_narrow_32_deep_2_short
#   bash submit_middle.sh --no-submit               # nur umstellen, nicht einreichen
#   bash submit_middle.sh --include-unrun ...       # auch Short-Configs ohne results-Ordner
set -euo pipefail
ABLATION_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "Abgebrochen: submit_middle.sh auf dem Login-Knoten starten, nicht in einem Slurm-Job." >&2
    exit 1
fi
if [[ ! -f "$ABLATION_ROOT/migrate_short_to_middle.py" ]]; then
    echo "Abgebrochen: migrate_short_to_middle.py fehlt neben submit_middle.sh." >&2
    exit 1
fi
cd "$ABLATION_ROOT"
exec "${ABLATION_PYTHON:-python3}" "$ABLATION_ROOT/migrate_short_to_middle.py" "$@"
