#!/usr/bin/env python3
"""Erzeugt ein kleines synthetisches big_ablation-Testprojekt (CPU, wenige Minuten je Config).

    python make_synthetic_project.py --code /pfad/zu/big_ablation --output /tmp/hum301_synthetic

Legt an: ein Zufallsgenom mit eingestreutem Motiv, BED-Dateien, zwei human301-Caches
(CHIP/GHTS für den TF "FAKE") über `data.py prepare`, eine synthetische PWM-Wurzel
(3 PWMs, selected.tsv) und Mini-Configs T00_basic sowie T30–T35 (2 Epochen).
Danach z. B.:
    cd /tmp/hum301_synthetic
    python run_experiment.py run --config T00_basic --manifest configs/caches.csv --device cpu --workers 0
    python run_experiment.py run --config T35_tta  --manifest configs/caches.csv --device cpu --workers 0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def write_fasta(genome, path, line=60):
    with path.open("w") as fa, Path(str(path) + ".fai").open("w") as fai:
        offset = 0
        for name, seq in genome.items():
            header = f">{name}\n"
            fa.write(header)
            offset += len(header)
            s = "".join(seq)
            body = "\n".join(s[i:i + line] for i in range(0, len(s), line)) + "\n"
            fa.write(body)
            fai.write(f"{name}\t{len(s)}\t{offset}\t{line}\t{line + 1}\n")
            offset += len(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--code", required=True, type=Path, help="Ordner mit model.py, data.py, training.py, run_experiment.py")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-config", type=Path, default=None,
                        help="Vorlage (Standard: configs/01_basic_0p1cdo.json im Code-Ordner)")
    args = parser.parse_args()
    code = args.code.resolve()
    out = args.output.resolve()
    if out.exists():
        parser.exit(1, f"Zielordner existiert bereits: {out}\n")
    (out / "configs").mkdir(parents=True)
    for name in ("model.py", "data.py", "training.py", "run_experiment.py"):
        shutil.copy2(code / name, out / name)
    for name in ("training_priority.csv", "training_priority_short.csv", "training_priority_middle.csv"):
        if (code / "configs" / name).is_file():
            shutil.copy2(code / "configs" / name, out / "configs" / name)

    rng = np.random.default_rng(1)
    raw = out / "raw"
    raw.mkdir()
    chroms = {f"chr{i}": 160000 for i in (1, 2, 3, 4, 5, 7)} | {"chr6": 110000}
    motif = "TGACGTCA"
    genome = {name: rng.choice(list("ACGT"), size=length) for name, length in chroms.items()}

    def make(path_pos, path_neg, chrom_list, n_pos, n_neg):
        pos_rows, neg_rows = [], []
        for chrom in chrom_list:
            slots = list(range(0, chroms[chrom] - 301, 311))
            for st in slots[:n_pos]:
                pos = st + 150 - 4 + int(rng.integers(-10, 11))
                genome[chrom][pos:pos + 8] = list(motif)
                if rng.random() < 0.5:
                    pos2 = st + 60 + int(rng.integers(0, 30))
                    genome[chrom][pos2:pos2 + 8] = list(motif)
                pos_rows.append(f"{chrom}\t{st}\t{st + 301}\n")
            for st in slots[n_pos:n_pos + n_neg]:
                neg_rows.append(f"{chrom}\t{st}\t{st + 301}\n")
        Path(path_pos).write_text("".join(pos_rows))
        Path(path_neg).write_text("".join(neg_rows))

    make(raw / "train_pos.bed", raw / "train_neg.bed", ["chr1", "chr2", "chr3", "chr4", "chr5", "chr7"], 36, 420)
    make(raw / "test_pos.bed", raw / "test_neg.bed", ["chr6"], 40, 300)
    write_fasta(genome, raw / "genome.fa")
    for assay, val in (("CHIP", "chr5,chr7"), ("GHTS", "chr7,chr5")):
        subprocess.run([sys.executable, str(out / "data.py"), "prepare",
                        "--train-positive-bed", str(raw / "train_pos.bed"), "--train-negative-bed", str(raw / "train_neg.bed"),
                        "--test-positive-bed", str(raw / "test_pos.bed"), "--test-negative-bed", str(raw / "test_neg.bed"),
                        "--genome-fasta", str(raw / "genome.fa"), "--genome-build", "synthetic", "--val-chromosomes", val,
                        "--output", str(out / "caches" / "FAKE" / assay / "dataset.npz")], check=True, cwd=out)
    with (out / "configs" / "caches.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["tf", "assay", "cache"])
        for assay in ("CHIP", "GHTS"):
            writer.writerow(["FAKE", assay, str(out / "caches" / "FAKE" / assay / "dataset.npz")])

    pwms = out / "pwms"

    def write_pwm(path, probs, name):
        lines = [f">{name}"] + ["\t".join(f"{math.log((p + 1e-5) / 0.25):.6f}" for p in row) for row in probs]
        path.write_text("\n".join(lines) + "\n")

    def motif_probs(seq, strength=0.85):
        return [[strength if "ACGT"[i] == base else (1 - strength) / 3 for i in range(4)] for base in seq]

    def random_probs(width):
        rows = []
        for _ in range(width):
            d = rng.dirichlet([0.5] * 4)
            rows.append(list(d / d.sum()))
        return rows

    rows = []
    for assay in ("CHS", "GHTS"):
        sel = pwms / "FAKE" / assay / "selected"
        sel.mkdir(parents=True)
        for order, (name, probs) in enumerate([("M1_motif", motif_probs(motif)), ("M2_rand10", random_probs(10)),
                                                ("M3_rand12", random_probs(12))], 1):
            write_pwm(sel / f"{name}.pwm", probs, name)
            rows.append(dict(TF="FAKE", Assay=assay, Matrix_ID=name, Status="selected", Approved="yes", Eligible="True",
                             Selected_order=order, MEX_overall_rank=order, Rank_motif_ID=f"{name}.rank", Width=len(probs),
                             Copied_to=str(sel / f"{name}.pwm")))
    with (pwms / "selected.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    base_path = args.base_config or (code / "configs" / "01_basic_0p1cdo.json")
    base = json.loads(base_path.read_text())
    base.pop("task_profile", None)
    base["tfs"] = ["FAKE"]
    base["fitting"]["params"].update(epochs=2, batch_size=44, num_workers=0, eval_batch_size=64)

    def cfg(name, desc, **changes):
        c = json.loads(json.dumps(base))
        c["name"], c["description"], c["reference"] = name, desc, "T00_basic"
        for path, value in changes.items():
            node = c
            keys = path.split("/")
            for k in keys[:-1]:
                node = node[k]
            node[keys[-1]] = value
        (out / "configs" / f"{name}.json").write_text(json.dumps(c, indent=2) + "\n")

    cfg("T00_basic", "tiny basic")
    cfg("T30_exp", "exp first layer", **{"predictor/params/motif_activation": "exp"})
    cfg("T31_rc", "rc sharing", **{"predictor/params/motif_rc_sharing": True})
    cfg("T32_pwmtarget", "pwm target", **{"predictor/factory": "ablation.builders:maa_pwm_target",
        "predictor/params/pwm_root": str(pwms), "predictor/params/pwm_count": 3, "predictor/params/pwm_target_weight": 1.0})
    cfg("T33_shift8", "shift 8", **{"fitting/params/train_shift_max": 8})
    cfg("T34_shift0", "shift 0", **{"fitting/params/train_shift_max": 0})
    cfg("T35_tta", "tta all crops, reuse T00 checkpoints", **{"fitting": {"factory": "ablation.fitting:reuse_checkpoint",
        "params": {"source_config": "T00_basic"}},
        "inference/params": {"eval_crops": "all", "eval_rc": True, "allow_crop_mode_change": True}})
    print(f"Synthetisches Projekt angelegt: {out}")
    print("Reihenfolge: zuerst T00_basic (liefert die Checkpoints für T35_tta), dann die übrigen.")


if __name__ == "__main__":
    main()
