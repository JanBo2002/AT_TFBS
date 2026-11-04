#!/usr/bin/env bash
#SBATCH --job-name=hum301_collect
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --no-requeue
set -euo pipefail
if [[ $# -lt 2 || $# -gt 3 || ! "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]]; then
    echo "Aufruf: collect_config.sbatch /absolut/big_ablation KONFIG [STATUS_JSON]" >&2
    exit 2
fi
ABLATION_ROOT="$1"
if [[ ! -f "$ABLATION_ROOT/configs/$2.json" ]]; then
    echo "Konfiguration fehlt: $ABLATION_ROOT/configs/$2.json" >&2
    exit 2
fi
"${ABLATION_PYTHON:-python3}" - "$ABLATION_ROOT" "$2" "${3:-}" <<'HUM301_COLLECT_FALLBACK'
import json
from pathlib import Path
import subprocess
import sys


# HUM301_CONTINUATION_COLLECT_V2
CONTINUATION_HELPER = '# EXPERIMENT_CONTINUATION_V1\ndef _experiment_continuation(experiment, config, selected):\n    """Accept a documented rename/TF extension with identical scientific settings."""\n    path = experiment / "continuation.json"\n    if not path.exists():\n        return None\n    state = json.loads(path.read_text())\n    ignored = {"name", "description", "tfs", "task_profile"}\n    scientific = lambda value: {k:v for k,v in value.items() if k not in ignored}\n    source = state["source_config"]\n    if (state.get("schema_version") != 1 or state["target_config"] != config\n            or state["target_name"] != config["name"]\n            or scientific(source) != scientific(config)):\n        raise ValueError("Continuation changes more than the experiment name or TF selection")\n    imported = state["imported_runs"]\n    if not set(imported) <= {meta["run_id"] for meta in selected}:\n        raise ValueError("A completed run recorded by the continuation is missing")\n    for meta in selected:\n        actual = meta["config"]\n        if scientific(actual) != scientific(config) or recipe_digest(actual) != meta["config_hash"]:\n            raise ValueError("Continuation contains an incompatible or inconsistent training recipe")\n        if meta["run_id"] in imported:\n            expected = imported[meta["run_id"]]\n            if (actual["name"] != state["source_name"] or\n                    any(meta.get(key) != expected[key] for key in\n                        ("config_hash", "code_hash", "tf", "train_assay", "seed"))):\n                raise ValueError("Original run identity or provenance changed during continuation")\n        elif (actual["name"] != config["name"] or meta["config_hash"] != recipe_digest(config)\n                or meta["code_hash"] != state["continuation_code_hash"]):\n            raise ValueError("An undocumented experiment was added to the continuation")\n    return state\n\n\n'
CONTINUATION_PATCHES = [('    if len(config_hashes)>1 or (len(code_hashes)>1 and not allow_code_version_mismatch):', '    continuation = _experiment_continuation(experiment, config, [metadata[r] for r in active])\n    if (len(config_hashes)>1 and continuation is None) or (len(code_hashes)>1 and not allow_code_version_mismatch):'), ('        row["included_in_summary"] = row["run_id"] in active and row.get("status")=="completed"', '        row["included_in_summary"] = row["run_id"] in active and row.get("status")=="completed"\n        if continuation is not None:\n            row["experiment_id"] = config["name"]'), ('    all_rows = rows+aggregated', '    if continuation is not None:\n        for row in aggregated:\n            if row["summary_level"] == "experiment":\n                if len(config_hashes)>1: row["config_hash"] = None\n                if len(code_hashes)>1: row["code_hash"] = None\n    all_rows = rows+aggregated'), ('    write_report(experiment,config,rows,aggregated,csv_path)', '    write_report(experiment,config,rows,aggregated,csv_path)\n    if continuation is not None:\n        with (experiment/"evaluation.md").open("a") as report:\n            report.write("\\nContinuation: " + continuation["source_name"] + " -> " + config["name"] +\n                ". Imported completed runs: " + str(len(continuation["imported_runs"])) +\n                ". Original run/model/evaluation IDs and hashes are retained; CSV experiment labels use the current name. "\n                "See continuation.json for provenance.\\n")')]


def collect_continuation(root, name):
    import ast
    import sys
    sys.path.insert(0, str(root))
    import training
    state = json.loads((root/'results'/name/'continuation.json').read_text())
    if training.package_hash(root) != state['continuation_code_hash']:
        raise ValueError('Python implementation changed after continuation was prepared')
    text = (root/'training.py').read_text()
    anchor = 'def _collect(root, experiment, config, destination, *, allow_code_version_mismatch=False):'
    if text.count(anchor) != 1:
        raise ValueError('Collection implementation changed; continuation adapter not applied')
    text = text.replace(anchor, CONTINUATION_HELPER + anchor, 1)
    for before, after in CONTINUATION_PATCHES:
        if text.count(before) != 1:
            raise ValueError('Collection implementation changed; continuation adapter not applied')
        text = text.replace(before, after, 1)
    selected = [node for node in ast.parse(text).body
                if isinstance(node, ast.FunctionDef) and node.name in {'_experiment_continuation', '_collect'}]
    if len(selected) != 2:
        raise ValueError('Continuation collection adapter is incomplete')
    module = ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[]))
    namespace = dict(vars(training))
    exec(compile(module, str(root/'collect_config.sbatch') + ':continuation', 'exec'), namespace)
    original = training._collect
    training._collect = namespace['_collect']
    try:
        training.collect(root, name)
    finally:
        training._collect = original
    return 0


def main(root, name, completion):
    root = Path(root).resolve()
    if completion:
        path = Path(completion)
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            state = {}
        outputs = (root / 'results' / 'Results_all' / (name + '.csv'),
                   root / 'results' / name / 'evaluation.md')
        if state.get('summary_status') == 'completed' and all(output.is_file() for output in outputs):
            print(f'{name}: Zusammenfassung wurde bereits direkt im letzten Lauf erstellt.')
            return 0
    if (root / 'results' / name / 'continuation.json').is_file():
        return collect_continuation(root, name)
    return subprocess.run([sys.executable, str(root / 'run_experiment.py'), 'collect', '--config', name]).returncode


raise SystemExit(main(*sys.argv[1:]))
HUM301_COLLECT_FALLBACK
