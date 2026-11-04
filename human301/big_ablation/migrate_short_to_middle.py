#!/usr/bin/env python3
"""Stellt Short-Configs (15 TFs) auf das Middle-Profil (21 TFs) um und benennt sie um.

Für jede ausgewählte Config NAME_short:
  * configs/NAME_short.json  -> configs/NAME_middle.json   (name, task_profile=middle, Beschreibung)
  * results/NAME_short/      -> results/NAME_middle/       (Ordner wird umbenannt, Läufe bleiben)
  * Metadaten der vorhandenen Läufe (run.json, config.json, records.jsonl,
    evaluations/*/*/metrics.json, training/*.json) erhalten den neuen Namen,
    den neuen Rezept-Hash und die neuen Pfade
  * Configs, die diese Config als "reference" nennen, zeigen auf den neuen Namen
  * Results_all/NAME_short.csv wird in die Sicherung verschoben (wird neu erzeugt)
Anschließend reicht submit_array.sh --restart nur die fehlenden Läufe ein
(6 TFs x 2 Assays je Seed); alle vorhandenen Läufe bleiben erhalten.

Aufruf (Login-Knoten, im big_ablation-Ordner):
    python migrate_short_to_middle.py [--dry-run] [--no-submit] [--account=NAME] [CONFIG ...]
Ohne CONFIG-Angabe werden alle Short-Configs aus configs/ bearbeitet.
Configs mit laufenden oder wartenden Jobs werden übersprungen; das Script
später für diese erneut aufrufen. Short-Configs ohne results-Ordner werden nur
mit --include-unrun umgestellt (dann startet das volle Middle-Training).
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SELECTION_KEYS = {"description", "seeds", "train_assays", "test_assays", "tfs", "task_profile"}
ASSAY_DIR = {"CHIP": "train_CHIP", "GHTS": "train_GHTS"}
RUN_JSON_FILES = ("run.json", "config.json", "training/configuration.json", "training/prediction_metadata.json")


class Abort(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Abort(message)


# Hilfsfunktionen (ohne Import von training.py: kein torch auf dem Login-Knoten nötig)
# ----------------------------------------------------------------------------------

def json_text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def recipe_hash(config):
    """Entspricht training.recipe_digest (Auswahlschlüssel ausgenommen)."""
    return hashlib.sha256(json_text({k: v for k, v in config.items() if k not in SELECTION_KEYS}).encode()).hexdigest()


def read_priority(path):
    require(path.is_file(), f"Prioritätsliste fehlt: {path}")
    pairs = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == ["priority", "tf", "assay"], f"{path.name}: Kopfzeile muss priority,tf,assay sein")
        for expected, row in enumerate(reader, 1):
            assay = {"CHS": "CHIP"}.get(row["assay"], row["assay"])
            require(row["tf"] and row["priority"].isdigit() and int(row["priority"]) == expected,
                    f"{path.name}: Prioritäten müssen lückenlos bei 1 beginnen")
            require(assay in ASSAY_DIR, f"{path.name}: ungültiges Assay {row['assay']!r}")
            require((row["tf"], assay) not in pairs, f"{path.name}: doppelter Eintrag {row['tf']}/{row['assay']}")
            pairs.append((row["tf"], assay))
    require(pairs, f"{path.name}: Liste ist leer")
    return pairs


def check_code_support():
    training = (ROOT / "training.py").read_text()
    require('"middle"' in training and "TASK_PROFILES" in training and "run_recipe_hash" in training,
            "training.py kennt das Profil 'middle' noch nicht; zuerst das Middle-Update installieren.")
    require("profile_priority_file" in (ROOT / "run_experiment.py").read_text(),
            "run_experiment.py verwendet noch die alte Profil-Zuordnung; zuerst das Middle-Update installieren.")
    require((ROOT / "submit_array.sh").is_file(), "submit_array.sh fehlt im Projektordner.")


def atomic_write_text(path, text):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        Path(temporary).write_text(text)
        if path.exists():
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_write_json(path, value):
    atomic_write_text(path, json.dumps(value, indent=2) + "\n")


def new_name(name):
    return name[:-len("_short")] + "_middle" if name.endswith("_short") else name + "_middle"


def rewrite_strings(value, replacements):
    """Ersetzt alte Ergebnispfade in allen Zeichenketten eines JSON-Dokuments."""
    if isinstance(value, str):
        for pattern, replacement in replacements:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, list):
        return [rewrite_strings(item, replacements) for item in value]
    if isinstance(value, dict):
        return {key: rewrite_strings(item, replacements) for key, item in value.items()}
    return value


# Zustand
# -------

def run_completed(run_dir, test_assays):
    try:
        meta = json.loads((run_dir / "run.json").read_text())
    except (OSError, ValueError):
        return False
    if meta.get("status") != "completed" or not (run_dir / "records.jsonl").is_file():
        return False
    for assay in test_assays:
        evaluation = run_dir / "evaluations" / "standard" / assay
        try:
            metrics = json.loads((evaluation / "metrics.json").read_text())
        except (OSError, ValueError):
            return False
        if metrics.get("status") != "completed" or not (evaluation / "predictions.tsv").is_file():
            return False
    return True


def load_all_configs():
    configs = {}
    for path in sorted((ROOT / "configs").glob("*.json")):
        try:
            config = json.loads(path.read_text())
        except ValueError as error:
            raise Abort(f"Config nicht lesbar: {path}: {error}")
        require(isinstance(config, dict) and config.get("name") == path.stem,
                f"Name in der Datei stimmt nicht mit dem Dateinamen überein: {path}")
        configs[path.stem] = config
    return configs


def is_short_config(config):
    profile = config.get("task_profile", "long")
    if "tfs" in config:
        return False
    return profile == "short" or (profile == "middle" and config["name"].endswith("_short"))


def inspect(name, config, configs, middle_pairs):
    target = new_name(name)
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", target) is not None, f"Ungültiger neuer Name: {target}")
    require(target not in configs and not (ROOT / "configs" / (target + ".json")).exists(),
            f"{name}: Ziel-Config existiert bereits: configs/{target}.json")
    require(not (ROOT / "results" / target).exists(), f"{name}: Ziel-Ordner existiert bereits: results/{target}")
    for assay in config["train_assays"]:
        require(assay in ASSAY_DIR, f"{name}: unbekanntes Trainings-Assay {assay!r}")
    experiment = ROOT / "results" / name
    require(not experiment.is_symlink(), f"{name}: results/{name} ist ein Link; keine automatische Umstellung.")
    stored = None
    if (experiment / "config.json").is_file():
        stored = json.loads((experiment / "config.json").read_text())
        require(recipe_hash(stored) == recipe_hash(config),
                f"{name}: results/{name}/config.json enthält ein anderes Modell-/Trainingsrezept als configs/{name}.json.")
    done, missing = [], []
    for tf, assay in middle_pairs:
        if assay not in config["train_assays"]:
            continue
        for seed in config["seeds"]:
            run_dir = experiment / "runs" / tf / ASSAY_DIR[assay] / f"seed_{int(seed):02d}"
            (done if run_completed(run_dir, config["test_assays"]) else missing).append((tf, assay, int(seed)))
    dependents = sorted(other for other, spec in configs.items() if spec.get("reference") == name and other != name)
    return dict(name=name, target=target, config=config, experiment=experiment, stored=stored,
                done=done, missing=missing, dependents=dependents)


# Laufende Jobs
# -------------

def jobs_by_config():
    """Ordnet laufende/wartende Projektjobs den Configs zu; None = Zuordnung unbekannt."""
    if not shutil.which("squeue"):
        return {}, []
    env = dict(os.environ, LC_ALL="C")
    result = subprocess.run(["squeue", "--me", "--noheader", "--format=%i|%j|%Z"],
                            capture_output=True, text=True, env=env, timeout=90)
    require(result.returncode == 0, "squeue fehlgeschlagen: " + (result.stderr.strip() or result.stdout.strip()))
    owners = {}
    for submission in (ROOT / "plans").glob("*/submission.json"):
        try:
            state = json.loads(submission.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(state, dict) and state.get("array_id") is not None:
            owners[str(state["array_id"])] = set(state.get("requested_configs", state.get("configs", [])))
    busy, unknown = {}, []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split("|", 2)]
        if len(fields) != 3 or not fields[2] or Path(fields[2]).resolve() != ROOT:
            continue
        job, job_name = fields[0], fields[1]
        parent = job.split("_", 1)[0]
        if job_name.startswith("h301_collect_"):
            busy.setdefault(job_name[len("h301_collect_"):], []).append(f"{job} ({job_name})")
        elif job_name.startswith(("h301_array_", "hum301_array")) or re.match(r"A\d+_", job_name):
            owned = owners.get(parent)
            if owned is None:
                unknown.append(f"{job} ({job_name})")
            for config in owned or ():
                busy.setdefault(config, []).append(f"{job} ({job_name})")
    return busy, list(dict.fromkeys(unknown))


# Umstellung einer Config
# -----------------------

def run_files(run_dir):
    files = [run_dir / name for name in RUN_JSON_FILES]
    files += sorted(run_dir.glob("evaluations/*/*/metrics.json"))
    files.append(run_dir / "records.jsonl")
    return [path for path in files if path.is_file()]


def migrate(report, name_map, backup):
    name, target, config = report["name"], report["target"], dict(report["config"])
    experiment = report["experiment"]
    config_backup = backup / "configs"
    config_backup.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "configs" / (name + ".json"), config_backup / (name + ".json"))

    config["name"] = target
    config["reference"] = name_map.get(config["reference"], config["reference"])
    config["task_profile"] = "middle"
    note = " Extended to the middle task profile (renamed from " + name + ")."
    if note.strip() not in config.get("description", ""):
        config["description"] = (config.get("description", "").rstrip() + note).strip()
    new_hash = recipe_hash(config)

    if experiment.exists():
        old_prefix = str(experiment)
        new_prefix = str(ROOT / "results" / target)
        replacements = [(re.compile(re.escape(old_prefix) + r"(?=/|$)"), new_prefix)]
        for run_json in sorted(experiment.glob("runs/*/*/*/run.json")):
            run_dir = run_json.parent
            for path in run_files(run_dir):
                relative = path.relative_to(ROOT / "results")
                destination = backup / "results" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, destination)
                if path.name == "records.jsonl":
                    lines = []
                    for line in path.read_text().splitlines():
                        if not line.strip():
                            continue
                        row = rewrite_strings(json.loads(line), replacements)
                        if row.get("experiment_id") == name:
                            row["experiment_id"] = target
                        if row.get("reference_id") in name_map:
                            row["reference_id"] = name_map[row["reference_id"]]
                        if "config_hash" in row:
                            row["config_hash"] = new_hash
                        lines.append(json.dumps(row, allow_nan=False, default=str))
                    atomic_write_text(path, "".join(line + "\n" for line in lines))
                    continue
                document = rewrite_strings(json.loads(path.read_text()), replacements)
                if path.name == "run.json":
                    document["experiment_id"] = target
                    document["reference_id"] = name_map.get(document.get("reference_id"), document.get("reference_id"))
                    document["config_hash"] = new_hash
                    if isinstance(document.get("config"), dict):
                        document["config"] = dict(document["config"], name=target, reference=config["reference"],
                                                  task_profile="middle", description=config["description"])
                elif path.name == "config.json":
                    document.update(name=target, reference=config["reference"], task_profile="middle",
                                    description=config["description"])
                elif path.name == "metrics.json" and document.get("experiment_id") == name:
                    document["experiment_id"] = target
                atomic_write_json(path, document)
        if (experiment / "config.json").is_file():
            shutil.copy2(experiment / "config.json", config_backup / (name + ".results.config.json"))
            stored = dict(report["stored"], name=target, reference=config["reference"], task_profile="middle",
                          description=config["description"])
            atomic_write_json(experiment / "config.json", stored)
        os.rename(experiment, ROOT / "results" / target)
        summary = ROOT / "results" / "Results_all" / (name + ".csv")
        if summary.is_file():
            (backup / "Results_all").mkdir(parents=True, exist_ok=True)
            shutil.move(str(summary), str(backup / "Results_all" / summary.name))

    atomic_write_json(ROOT / "configs" / (target + ".json"), config)
    (ROOT / "configs" / (name + ".json")).unlink()


def update_references(name_map, migrated, backup):
    """Configs außerhalb der Auswahl, die eine umbenannte Config als Referenz nennen."""
    changed = []
    for path in sorted((ROOT / "configs").glob("*.json")):
        if path.stem in migrated:
            continue
        config = json.loads(path.read_text())
        if config.get("reference") not in name_map:
            continue
        (backup / "configs").mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup / "configs" / path.name)
        config["reference"] = name_map[config["reference"]]
        atomic_write_json(path, config)
        stored = ROOT / "results" / path.stem / "config.json"
        if stored.is_file():
            shutil.copy2(stored, backup / "configs" / (path.stem + ".results.config.json"))
            atomic_write_json(stored, dict(json.loads(stored.read_text()), reference=config["reference"]))
        changed.append(path.stem)
    return changed


def ordered(reports):
    """Referenzen vor abhängigen Configs, sonst alphabetisch."""
    names = {report["name"]: report for report in reports}
    result = []
    def visit(name, trail=()):
        if name in result or name not in names:
            return
        require(name not in trail, "Zirkuläre Referenzen zwischen den Configs: " + " -> ".join(trail + (name,)))
        visit(names[name]["config"].get("reference"), trail + (name,))
        result.append(name)
    for name in sorted(names):
        visit(name)
    return [names[name] for name in result]


# Hauptprogramm
# -------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="*", metavar="CONFIG", help="Short-Configs aus configs/ (Standard: alle)")
    parser.add_argument("--dry-run", action="store_true", help="Nur prüfen und berichten; nichts ändern")
    parser.add_argument("--no-submit", action="store_true", help="Nur umstellen und umbenennen; keine Jobs einreichen")
    parser.add_argument("--account", help="Slurm-Account, wird an submit_array.sh durchgereicht")
    parser.add_argument("--include-unrun", action="store_true",
                        help="Auch Short-Configs ohne results-Ordner umstellen (startet dann alle 42 Aufgaben)")
    args = parser.parse_args()
    require(not os.environ.get("SLURM_JOB_ID"), "Dieses Script auf dem Login-Knoten starten, nicht in einem Slurm-Job.")
    check_code_support()
    short_pairs = read_priority(ROOT / "configs" / "training_priority_short.csv")
    middle_pairs = read_priority(ROOT / "configs" / "training_priority_middle.csv")
    dropped = [pair for pair in short_pairs if pair not in middle_pairs]
    require(not dropped, "Middle-Liste enthält nicht alle Short-Paare: " + ", ".join(f"{tf}/{a}" for tf, a in dropped))

    (ROOT / "plans").mkdir(exist_ok=True)
    with (ROOT / "plans" / ".dispatcher.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Abort("submit_array.sh läuft gerade in diesem Projekt; später erneut versuchen.") from None
        configs = load_all_configs()
        if args.configs:
            names = list(dict.fromkeys(args.configs))
            for name in names:
                require(name in configs, f"Config fehlt: configs/{name}.json")
                require(is_short_config(configs[name]),
                        f"{name}: keine Short-Config (task_profile {configs[name].get('task_profile', 'long')!r}).")
        else:
            names = [name for name, config in configs.items() if is_short_config(config)]
        require(names, "Keine Short-Configs gefunden.")

        print("Prüfe Warteschlange ...", flush=True)
        busy, unknown = jobs_by_config()
        require(not unknown, "Trainingsjobs ohne Config-Zuordnung in plans/*/submission.json: " + "; ".join(unknown)
                + ". Vor der Umstellung muss klar sein, welche Configs diese Jobs betreffen.")
        skipped = {name: busy[name] for name in names if name in busy}
        selected = [name for name in names if name not in busy]
        unrun = [name for name in selected if not (ROOT / "results" / name).exists()]
        if unrun and not args.include_unrun:
            selected = [name for name in selected if name not in unrun]
        reports = ordered([inspect(name, configs[name], configs, middle_pairs) for name in selected])
        name_map = {report["name"]: report["target"] for report in reports}
        # Referenzen auf Configs, die in einem früheren Aufruf bereits umbenannt wurden.
        for report in reports:
            reference = report["config"].get("reference")
            if reference not in configs and reference not in name_map and new_name(reference) in configs:
                name_map[reference] = new_name(reference)
        # Abhängige Configs, deren Referenz umbenannt wird, aber selbst laufen: später erneut aufrufen.
        for report in reports:
            blocked = [other for other in report["dependents"] if other in busy]
            require(not blocked, f"{report['name']}: wird von laufenden Configs referenziert ({', '.join(blocked)}); "
                    "erst nach deren Ende umstellen.")

        print(f"\nShort-Profil: {len({tf for tf, _ in short_pairs})} TFs, Middle-Profil: {len({tf for tf, _ in middle_pairs})} TFs.")
        for name, jobs in skipped.items():
            print(f"\n{name}: übersprungen, Jobs aktiv/wartend: {', '.join(jobs)} (später erneut aufrufen)")
        if unrun and not args.include_unrun:
            print("\nOhne Ergebnisse, nicht umgestellt (dafür --include-unrun angeben): " + ", ".join(unrun))
        for report in reports:
            print(f"\n{report['name']} -> {report['target']}:")
            if not report["experiment"].exists():
                print(f"  Ergebnisse: noch keine; nach der Umstellung werden alle {len(report['missing'])} Aufgaben trainiert.")
            else:
                total = len(report["done"]) + len(report["missing"])
                print(f"  Abgeschlossene Läufe im Middle-Umfang: {len(report['done'])} / {total}")
                short_missing = [(tf, a, s) for tf, a, s in report["missing"] if (tf, a) in short_pairs]
                if short_missing:
                    print("  Warnung: auch Short-Läufe fehlen/unvollständig: "
                          + ", ".join(f"{tf}/{a}/s{s}" for tf, a, s in short_missing))
                if report["missing"]:
                    print("  Fehlende Läufe: " + ", ".join(f"{tf}/{a}/s{s}" for tf, a, s in report["missing"]))
                else:
                    print("  Keine fehlenden Läufe; nur Umbenennung und neue Zusammenfassung.")
            if report["config"].get("reference") in name_map:
                print(f"  Referenz: {report['config']['reference']} -> {name_map[report['config']['reference']]}")
            others = [d for d in report["dependents"] if d not in name_map]
            if others:
                print(f"  Referenz wird angepasst in: {', '.join(others)}")

        if not reports:
            print("\nNichts zu tun.")
            return 0
        if args.dry_run:
            print("\nPrüfung erfolgreich; keine Änderungen ausgeführt (--dry-run).")
            return 0

        backup = ROOT / "plans" / "middle_migration_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup.mkdir(parents=True, exist_ok=False)
        (backup / "name_map.json").write_text(json.dumps(name_map, indent=2) + "\n")
        # Protokoll der verwendeten Middle-Liste; submit_array.sh bleibt unverändert.
        shutil.copy2(ROOT / "configs" / "training_priority_middle.csv", backup / "training_priority_middle.csv")
        print(f"\nSicherung: {backup}", flush=True)
        for report in reports:
            migrate(report, name_map, backup)
            print(f"Umgestellt: {report['name']} -> {report['target']}", flush=True)
        changed = update_references(name_map, set(name_map), backup)
        if changed:
            print("Referenzen angepasst in: " + ", ".join(changed))

    targets = [report["target"] for report in reports]
    command = ["bash", "submit_array.sh", "--restart", *targets]
    if args.account:
        command.append("--account=" + args.account)
    env = dict(os.environ, ABLATION_ALLOW_CODE_VERSION_MISMATCH="1")
    print("\nEinreichung (ergänzt nur fehlende Läufe; fertige Configs bekommen nur eine neue Zusammenfassung):")
    print("  ABLATION_ALLOW_CODE_VERSION_MISMATCH=1 " + " ".join(command))
    if args.no_submit:
        return 0
    print("\nStarte submit_array.sh ...", flush=True)
    completed = subprocess.run(command, cwd=ROOT, env=env)
    if completed.returncode:
        print("submit_array.sh hat einen Fehler gemeldet; die Umstellung der Dateien ist trotzdem abgeschlossen. "
              "Den obigen Befehl nach Behebung erneut ausführen.", file=sys.stderr)
    return completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Abort, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"Abgebrochen: {exc}", file=sys.stderr)
        raise SystemExit(1)
