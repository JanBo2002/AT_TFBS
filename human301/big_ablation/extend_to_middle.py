#!/usr/bin/env python3
"""Erweitert fertige Short-Configs (15 TFs) auf das Middle-Profil (21 TFs).

Die Config behält Namen, Modell- und Trainingsrezept und ihren results-Ordner;
nur task_profile wechselt von "short" auf "middle". Danach ergänzt
submit_array.sh --restart ausschließlich die fehlenden Läufe (6 TFs x 2 Assays
je Seed); die vorhandenen 30 Läufe bleiben unverändert erhalten.

Aufruf (Login-Knoten, im big_ablation-Ordner):
    python extend_to_middle.py [--dry-run] [--submit] [--account=NAME] CONFIG [CONFIG ...]
"""
from __future__ import annotations

import argparse
import csv
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


class Abort(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise Abort(message)


# Hilfsfunktionen (bewusst ohne Import von training.py: kein torch auf dem Login-Knoten nötig)
# -----------------------------------------------------------------------------------------

def recipe_hash(config):
    reduced = {k: v for k, v in config.items() if k not in SELECTION_KEYS}
    return hashlib.sha256(json.dumps(reduced, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def package_hash(root):
    result = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        if "results" in path.relative_to(root).parts or "__pycache__" in path.parts:
            continue
        result.update(str(path.relative_to(root)).encode())
        result.update(path.read_bytes())
    return result.hexdigest()


def read_priority(path):
    """Gleiche Prüfung wie data.read_priority; liefert geordnete (tf, assay)-Paare."""
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
    experiment = (ROOT / "run_experiment.py").read_text()
    require('"middle"' in training and "TASK_PROFILES" in training,
            "training.py kennt das Profil 'middle' noch nicht; zuerst die aktualisierten Programmdateien installieren.")
    require("profile_priority_file" in experiment,
            "run_experiment.py verwendet noch die alte Profil-Zuordnung; zuerst die aktualisierten Programmdateien installieren.")
    require("training_priority_middle.csv" in (ROOT / "submit_array.sh").read_text(),
            "submit_array.sh kopiert die Middle-Prioritätsliste noch nicht; zuerst die aktualisierte Version installieren.")


def atomic_write_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    os.close(fd)
    try:
        Path(temporary).write_text(json.dumps(value, indent=2) + "\n")
        if path.exists():
            shutil.copymode(path, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


# Zustand einer Config
# --------------------

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


def inspect_config(name, middle_pairs, short_pairs):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name) is not None, f"Ungültiger Config-Name: {name}")
    path = ROOT / "configs" / (name + ".json")
    require(path.is_file(), f"Config fehlt: {path}")
    config = json.loads(path.read_text())
    require(config.get("name") == name, f"Name in der Datei stimmt nicht mit dem Dateinamen überein: {path}")
    require("tfs" not in config, f"{name}: verwendet eine feste TF-Liste (tfs) statt eines task_profile; nichts zu erweitern.")
    profile = config.get("task_profile", "long")
    require(profile in ("short", "middle"),
            f"{name}: task_profile ist {profile!r}; erweitert werden nur Short-Configs (oder bereits umgestellte Middle-Configs).")
    for assay in config["train_assays"]:
        require(assay in ASSAY_DIR, f"{name}: unbekanntes Trainings-Assay {assay!r}")

    experiment = ROOT / "results" / name
    stored = None
    if (experiment / "config.json").is_file():
        stored = json.loads((experiment / "config.json").read_text())
        require(recipe_hash(stored) == recipe_hash(config),
                f"{name}: results/{name}/config.json enthält ein anderes Modell-/Trainingsrezept als configs/{name}.json; "
                "die Erweiterung würde beim Training abgewiesen.")
    if (experiment / "versions.txt").is_file():
        match = re.search(r"^code_sha256:\s*([0-9a-f]{64})", (experiment / "versions.txt").read_text(), re.MULTILINE)
        recorded_code = match.group(1) if match else None
    else:
        recorded_code = None

    done, missing = [], []
    for tf, assay in middle_pairs:
        if assay not in config["train_assays"]:
            continue
        for seed in config["seeds"]:
            run_dir = experiment / "runs" / tf / ASSAY_DIR[assay] / f"seed_{int(seed):02d}"
            (done if run_completed(run_dir, config["test_assays"]) else missing).append((tf, assay, int(seed)))
    expected_short = [(tf, assay, int(seed)) for tf, assay in short_pairs
                      if assay in config["train_assays"] for seed in config["seeds"]]
    short_missing = [task for task in expected_short if task in missing]
    return dict(name=name, path=path, config=config, profile=profile, experiment=experiment, stored=stored,
                recorded_code=recorded_code, done=done, missing=missing, short_missing=short_missing)


# Laufende Jobs
# -------------

def active_jobs_for(names):
    """Jobs dieses Projekts, die eine der Configs trainieren oder zusammenfassen."""
    if not shutil.which("squeue"):
        return []
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
    blocking = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split("|", 2)]
        if len(fields) != 3 or not fields[2] or Path(fields[2]).resolve() != ROOT:
            continue
        job, job_name = fields[0], fields[1]
        parent = job.split("_", 1)[0]
        if job_name.startswith("h301_collect_"):
            if job_name[len("h301_collect_"):] in names:
                blocking.append(f"{job} ({job_name})")
        elif job_name.startswith(("h301_array_", "hum301_array")) or re.match(r"A\d+_", job_name):
            owned = owners.get(parent)
            if owned is None or owned & set(names):
                blocking.append(f"{job} ({job_name}; Configs: {', '.join(sorted(owned)) if owned else 'unbekannt'})")
    return list(dict.fromkeys(blocking))


# Hauptprogramm
# -------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", metavar="CONFIG", help="Namen der Short-Configs aus configs/")
    parser.add_argument("--dry-run", action="store_true", help="Nur prüfen und berichten; nichts ändern")
    parser.add_argument("--submit", action="store_true",
                        help="Nach der Umstellung direkt 'bash submit_array.sh --restart ...' starten")
    parser.add_argument("--account", help="Slurm-Account, wird an submit_array.sh durchgereicht")
    args = parser.parse_args()
    require(not os.environ.get("SLURM_JOB_ID"), "Dieses Script auf dem Login-Knoten starten, nicht in einem Slurm-Job.")
    names = list(dict.fromkeys(args.configs))
    require(len(names) == len(args.configs), "Configs mehrfach angegeben.")

    check_code_support()
    short_pairs = read_priority(ROOT / "configs" / "training_priority_short.csv")
    middle_pairs = read_priority(ROOT / "configs" / "training_priority_middle.csv")
    dropped = [pair for pair in short_pairs if pair not in middle_pairs]
    require(not dropped, "Middle-Liste enthält nicht alle Short-Paare; so würde die Erweiterung Aufgaben entfernen: "
            + ", ".join(f"{tf}/{assay}" for tf, assay in dropped))
    added_tfs = list(dict.fromkeys(tf for tf, _ in middle_pairs if tf not in {tf for tf, _ in short_pairs}))
    print(f"Short-Profil: {len({tf for tf, _ in short_pairs})} TFs, {len(short_pairs)} Paare. "
          f"Middle-Profil: {len({tf for tf, _ in middle_pairs})} TFs, {len(middle_pairs)} Paare. "
          f"Neu: {', '.join(added_tfs)}")

    reports = [inspect_config(name, middle_pairs, short_pairs) for name in names]
    blocking = active_jobs_for(set(names))
    require(not blocking, "Für diese Configs sind noch Jobs aktiv oder wartend; laufende Aufgaben würden nach einer "
            "Config-Änderung mit 'Configuration changed since planning' abbrechen. Zuerst abwarten oder scancel: "
            + "; ".join(blocking))

    current_code = package_hash(ROOT)
    needs_version_flag = False
    for report in reports:
        name = report["name"]
        print(f"\n{name}:")
        print(f"  task_profile: {report['profile']}" + (" (bereits middle)" if report["profile"] == "middle" else " -> middle"))
        if report["recorded_code"] and report["recorded_code"] != current_code:
            needs_version_flag = True
            print("  Hinweis: Die bisherigen Läufe wurden mit einer anderen Code-Version trainiert; "
                  "die Ergänzung braucht ABLATION_ALLOW_CODE_VERSION_MISMATCH=1 (wird im Befehl unten gesetzt).")
        if not report["experiment"].exists():
            print(f"  Ergebnisse: noch keine (results-Ordner fehlt); nach der Umstellung werden alle "
                  f"{len(report['missing'])} Aufgaben des Middle-Umfangs trainiert.")
            continue
        print(f"  Abgeschlossene Läufe im Middle-Umfang: {len(report['done'])} / {len(report['done']) + len(report['missing'])}")
        if report["short_missing"]:
            print("  Warnung: Auch Läufe des Short-Umfangs fehlen oder sind unvollständig: "
                  + ", ".join(f"{tf}/{assay}/s{seed}" for tf, assay, seed in report["short_missing"]))
        if report["missing"]:
            print("  Fehlende Läufe (werden mit --restart ergänzt): "
                  + ", ".join(f"{tf}/{assay}/s{seed}" for tf, assay, seed in report["missing"]))
        else:
            print("  Keine fehlenden Läufe; --restart würde nur die Zusammenfassung neu schreiben.")

    to_change = [report for report in reports if report["profile"] == "short"]
    if args.dry_run:
        print("\nPrüfung erfolgreich; keine Änderungen ausgeführt (--dry-run).")
    elif to_change:
        backup = ROOT / "plans" / "middle_update_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
        backup.mkdir(parents=True, exist_ok=False)
        note = " Extended to the middle task profile (adds " + ", ".join(added_tfs) + ")."
        for report in to_change:
            config = dict(report["config"])
            config["task_profile"] = "middle"
            if note.strip() not in config.get("description", ""):
                config["description"] = (config.get("description", "").rstrip() + note).strip()
            shutil.copy2(report["path"], backup / report["path"].name)
            atomic_write_json(report["path"], config)
            if report["stored"] is not None:
                stored = dict(report["stored"], task_profile="middle", description=config["description"])
                shutil.copy2(report["experiment"] / "config.json", backup / (report["name"] + ".results.config.json"))
                atomic_write_json(report["experiment"] / "config.json", stored)
            print(f"Umgestellt: configs/{report['name']}.json (Sicherung: {backup})")
    else:
        print("\nAlle angegebenen Configs sind bereits auf middle; nichts geändert.")

    command = ["bash", "submit_array.sh", "--restart", *names]
    if args.account:
        command.append("--account=" + args.account)
    env = dict(os.environ)
    prefix = ""
    if needs_version_flag:
        env["ABLATION_ALLOW_CODE_VERSION_MISMATCH"] = "1"
        prefix = "ABLATION_ALLOW_CODE_VERSION_MISMATCH=1 "
    print("\nNächster Schritt (ergänzt nur die fehlenden Läufe, vorhandene bleiben erhalten):")
    print("  " + prefix + " ".join(command))
    if args.submit and not args.dry_run:
        print("\nStarte submit_array.sh ...", flush=True)
        completed = subprocess.run(command, cwd=ROOT, env=env)
        return completed.returncode
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (Abort, OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        print(f"Abgebrochen: {exc}", file=sys.stderr)
        raise SystemExit(1)
