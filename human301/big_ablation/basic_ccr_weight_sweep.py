#!/usr/bin/env python3
"""Basic + CCR: AP for both cross-assay scenarios and 10% score weights.

Uses the original prediction paths in comparison_summary.csv (00_basic,
output_ccr, seed 13). Scores are aligned by exact chromosome/start/end;
labels must agree. No thresholds, fitting, normalization or calibration.

One CSV: per-TF rows and one unweighted TF mean below each scenario.
The same tables are printed as Markdown. Missing/invalid inputs are reported
in this CSV, never replaced with zero AP. Existing overlap files are read only.

Requires numpy, pandas and scikit-learn. Run in the existing at_env environment:
    python basic_ccr_weight_sweep.py
"""

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


RESULTS = Path(
    "/data/projects/SFB_A03/jan/AT_TFBS/human301/big_ablation/results"
)
DEFAULT_SUMMARY = RESULTS / "prediction_overlap" / "comparison_summary.csv"
SCENARIOS = ("CHS_to_GHTS", "GHTS_to_CHS")
TFS = (
    "BATF2", "CREB3L3", "CTCF", "ELF3", "FLI1", "FOSL2", "GABPA", "GLI4",
    "LEF1", "LEUTX", "MAX", "MGA", "MYF6", "NFKB1", "NR1H4", "PAX7",
    "RFX5", "RORB", "SOX2", "SRY", "TERF1", "USF3", "VDR", "YY1", "ZBED9",
    "ZBTB8A", "ZFP3", "ZNF696", "ZNF772", "ZNF773",
)
MIX_COLUMNS = [(b, "AP_Basic{}_CCR{}".format(b, 100 - b))
               for b in range(90, 0, -10)]
AP_COLUMNS = ["AP_Basic", "AP_CCR"] + [name for _, name in MIX_COLUMNS]
FIELDNAMES = ["scenario", "TF"] + AP_COLUMNS + [
    "status", "n_sequences", "n_positive", "n_negative", "message"
]


def labels_and_scores(frame, label_column, score_column, source):
    labels = pd.to_numeric(frame[label_column], errors="raise").to_numpy()
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("{}: labels must be 0 or 1".format(source))
    scores = pd.to_numeric(frame[score_column], errors="raise").to_numpy(
        dtype=np.float64, copy=True
    )
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("{}: scores must be finite probabilities in [0, 1]".format(source))
    return labels.astype(np.uint8), scores


def coordinate_index(chrom, start, end, source):
    if chrom.isna().any() or chrom.str.contains(r"\s|^$", regex=True).any():
        raise ValueError("{}: invalid chromosome".format(source))
    if (start < 0).any() or (end <= start).any():
        raise ValueError("{}: invalid genomic interval".format(source))
    index = pd.MultiIndex.from_arrays([chrom, start, end], names=["chrom", "start", "end"])
    if not index.is_unique:
        raise ValueError("{}: duplicate genomic coordinates".format(source))
    return index


def load_pair(basic_file, ccr_file):
    """Return labels and two score arrays in the same exact sequence order."""
    basic = pd.read_csv(
        basic_file, sep="\t",
        usecols=["sample_id", "true_label", "binding_probability"],
        dtype={"sample_id": "string"},
    )
    coords = basic["sample_id"].str.extract(r"(?:^|:)([^:\s]+):(\d+)-(\d+)$")
    if coords.isna().any().any():
        raise ValueError("Basic: sample_id must end in chrom:start-end")
    basic_index = coordinate_index(
        coords[0], coords[1].astype(np.int64), coords[2].astype(np.int64), "Basic"
    )
    labels, basic_scores = labels_and_scores(
        basic, "true_label", "binding_probability", "Basic"
    )
    del basic, coords

    ccr = pd.read_csv(
        ccr_file,
        usecols=["bed_chrom", "bed_start", "bed_end", "label", "pred_proba"],
        dtype={"bed_chrom": "string", "bed_start": np.int64, "bed_end": np.int64},
    )
    ccr_index = coordinate_index(
        ccr["bed_chrom"], ccr["bed_start"], ccr["bed_end"], "CCR"
    )
    ccr_labels, ccr_scores = labels_and_scores(ccr, "label", "pred_proba", "CCR")
    if len(basic_index) != len(ccr_index):
        raise ValueError("Basic and CCR contain different sequence counts")
    order = ccr_index.get_indexer(basic_index)
    if (order < 0).any():
        raise ValueError("Basic and CCR contain different genomic coordinates")
    if not np.array_equal(labels, ccr_labels[order]):
        raise ValueError("Basic and CCR labels disagree after coordinate alignment")
    if np.unique(labels).size != 2:
        raise ValueError("Both positive and negative labels are required")
    return labels, basic_scores, ccr_scores[order]


def ap_weight_sweep(labels, basic_scores, ccr_scores):
    values = {
        "AP_Basic": float(average_precision_score(labels, basic_scores)),
        "AP_CCR": float(average_precision_score(labels, ccr_scores)),
    }
    for basic_percent, column in MIX_COLUMNS:
        mixed = ((basic_percent / 100.0) * basic_scores
                 + ((100 - basic_percent) / 100.0) * ccr_scores)
        values[column] = float(average_precision_score(labels, mixed))
    return values


def load_manifest(path):
    summary = pd.read_csv(path)
    required = {
        "own_model", "reference_model", "scenario", "TF", "seed", "status",
        "own_file", "reference_file", "n_sequences", "n_positive", "n_negative",
        "ap_own", "ap_reference",
    }
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError("Summary is missing columns: {}".format(", ".join(missing)))
    selected = summary[
        summary["own_model"].eq("00_basic")
        & summary["reference_model"].eq("output_ccr")
        & summary["scenario"].isin(SCENARIOS)
    ].copy()
    if selected.duplicated(["TF", "scenario"]).any():
        raise ValueError("Summary has duplicate Basic/CCR rows for a TF/scenario")
    unknown = set(selected["TF"]) - set(TFS)
    if unknown:
        raise ValueError("Unexpected TFs in summary: {}".format(sorted(unknown)))
    return {(row["TF"], row["scenario"]): row for row in selected.to_dict("records")}


def evaluate_pair(tf, scenario, manifest_row):
    result = {"TF": tf, "scenario": scenario, "status": "ok", "message": ""}
    if manifest_row is None:
        result.update(status="missing", message="Basic/CCR pair absent from summary")
        return result
    try:
        if manifest_row["status"] != "ok":
            raise ValueError("Earlier analysis status is {}".format(manifest_row["status"]))
        if int(manifest_row["seed"]) != 13:
            raise ValueError("Expected seed 13")
        basic_file = Path(manifest_row["own_file"])
        ccr_file = Path(manifest_row["reference_file"])
        missing = [str(p) for p in (basic_file, ccr_file) if not p.is_file()]
        if missing:
            raise FileNotFoundError("Missing prediction file(s): " + " | ".join(missing))
        y, basic, ccr = load_pair(basic_file, ccr_file)
        counts = {"n_sequences": int(y.size), "n_positive": int(y.sum()),
                  "n_negative": int(y.size - y.sum())}
        for name, actual in counts.items():
            if actual != int(manifest_row[name]):
                raise ValueError("{} differs from earlier analysis".format(name))
        values = ap_weight_sweep(y, basic, ccr)
        for actual, saved in [(values["AP_Basic"], manifest_row["ap_own"]),
                              (values["AP_CCR"], manifest_row["ap_reference"])]:
            if not np.isclose(actual, float(saved), rtol=1e-6, atol=1e-8):
                raise ValueError("Single-model AP differs from earlier analysis: {} vs {}".format(
                    actual, saved))
        result.update(counts)
        result.update(values)
    except FileNotFoundError as exc:
        result.update(status="missing", message=str(exc))
    except (ValueError, TypeError, OSError, pd.errors.ParserError) as exc:
        result.update(status="error", message=str(exc))
    return result


def with_means(details):
    rows = []
    for scenario in SCENARIOS:
        group = [row for row in details if row["scenario"] == scenario]
        successful = [row for row in group if row["status"] == "ok"]
        mean = {
            "scenario": scenario, "TF": "DURCHSCHNITT ({} TFs)".format(len(successful)),
            "status": "ok" if len(successful) == len(TFS) else "partial",
            "message": "Unweighted TF mean; {}/{} successful pairs".format(
                len(successful), len(TFS)),
        }
        if successful:
            mean.update({column: float(np.mean([r[column] for r in successful]))
                         for column in AP_COLUMNS})
        rows.extend(group)
        rows.append(mean)
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", dir=str(path.parent),
            prefix=path.name + ".", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(temporary), str(path))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def print_tables(rows):
    headers = ["TF", "AP Basic", "AP CCR"] + [
        "{}:{}".format(b, 100 - b) for b, _ in MIX_COLUMNS
    ]
    print("\nAlle Gewichtungen: Basic:CCR in %. Metrik: Average Precision (AP).")
    for scenario in SCENARIOS:
        print("\n**{}**\n".format(scenario.replace("_to_", " → ")))
        print("| " + " | ".join(headers) + " |")
        print("| " + " | ".join(["---"] + ["---:"] * len(AP_COLUMNS)) + " |")
        for row in rows:
            if row["scenario"] != scenario:
                continue
            cells = [row["TF"]] + [
                ("{:.3f}".format(row[c]).replace(".", ",") if c in row else "—")
                for c in AP_COLUMNS
            ]
            print("| " + " | ".join(cells) + " |")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=None,
                        help="Default: basic_ccr_weight_sweep.csv beside the summary")
    args = parser.parse_args(argv)
    output = args.output or args.summary.parent / "basic_ccr_weight_sweep.csv"
    if args.summary.resolve() == output.resolve():
        parser.error("Output must differ from the input summary")
    try:
        manifest = load_manifest(args.summary)
    except (OSError, ValueError) as exc:
        print("Cannot read summary: {}".format(exc), file=sys.stderr)
        return 1
    missing_paths = sorted({
        str(row[key]) for row in manifest.values() if row["status"] == "ok"
        for key in ("own_file", "reference_file") if not Path(row[key]).is_file()
    })
    print("Preflight: {} summary pairs; {} expected; {} missing prediction files".format(
        len(manifest), len(TFS) * len(SCENARIOS), len(missing_paths)), flush=True)
    for path in missing_paths:
        print("MISSING: " + path, file=sys.stderr, flush=True)
    details = []
    for scenario in SCENARIOS:
        for tf in TFS:
            print("[{}] {} {}".format(len(details) + 1, scenario, tf), flush=True)
            row = evaluate_pair(tf, scenario, manifest.get((tf, scenario)))
            details.append(row)
            if row["status"] != "ok":
                print("{}: {}".format(row["status"].upper(), row["message"]),
                      file=sys.stderr, flush=True)
            # Preserve completed work if a long server job is interrupted.
            write_csv(output, with_means(details))
    rows = with_means(details)
    print_tables(rows)
    bad = sum(row["status"] != "ok" for row in details)
    print("\nCSV: {}\nSuccessful pairs: {}/{}".format(output, len(details) - bad, len(details)),
          flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
