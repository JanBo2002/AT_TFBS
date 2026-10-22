"""Deterministic single-writer CSV export, paired comparisons and summaries."""
from __future__ import annotations

import csv
import fcntl
import json
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from .recorder import atomic_json
from .schema import COLUMNS, NUMERIC, SCHEMA_VERSION, SUM_FIELDS


def number(value):
    if value in (None, ""): return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError,TypeError): return None


def truth(value):
    return value is True or str(value).lower() == "true"


def tf_set(rows):
    names=set()
    for row in rows:
        if row.get("tf"):
            names.add(row["tf"])
        elif row.get("tf_ids_json"):
            names.update(json.loads(row["tf_ids_json"]))
    return names


def summaries(records, level, source, keys):
    groups = defaultdict(list)
    for row in records:
        if row["row_type"] == source and truth(row.get("included_in_summary")):
            groups[tuple(row.get(k,"") for k in keys)].append(row)
    result = []
    for group_key, rows in groups.items():
        common = {k:rows[0].get(k) for k in ("schema_version","experiment_id","reference_id","config_hash","code_hash")}
        common.update(dict(zip(keys,group_key)))
        common.update(row_type="summary",status="completed",included_in_summary=True,
            summary_level=level,summary_source=source,n_records=len(rows),
            n_runs=len({r["run_id"] for r in rows}),n_tfs=len(tf_set(rows)),
            n_seeds=len({r.get("seed") for r in rows if r.get("seed") is not None}))
        for statistic in ("mean","std","median","q25","q75","min","max","sum"):
            row = dict(common,statistic=statistic)
            row["aggregation_unit"] = "unique_runs" if statistic == "sum" else ("tf_mean_over_seeds" if level=="experiment" else "seed")
            for field in NUMERIC:
                if statistic == "sum" and field not in SUM_FIELDS: continue
                values = [number(r.get(field)) for r in rows]
                values = [v for v in values if v is not None]
                if not values: continue
                if field=="win_fraction":
                    by_tf_delta=defaultdict(list)
                    for r in rows:
                        delta=number(r.get("delta_auprc"))
                        if delta is not None: by_tf_delta[r.get("tf",r["run_id"])].append(delta)
                    values=[float(np.mean(v)>0) for v in by_tf_delta.values()]
                    if not values: continue
                elif level == "experiment" and statistic != "sum":
                    by_tf = defaultdict(list)
                    for r in rows:
                        value = number(r.get(field))
                        if value is not None: by_tf[r.get("tf",r["run_id"])].append(value)
                    values = [float(np.mean(v)) for v in by_tf.values()]
                if statistic == "std":
                    value = float(np.std(values,ddof=1)) if len(values)>1 else None
                elif statistic == "mean": value = float(np.mean(values))
                elif statistic == "median": value = float(np.median(values))
                elif statistic == "q25": value = float(np.percentile(values,25))
                elif statistic == "q75": value = float(np.percentile(values,75))
                elif statistic == "min": value = min(values)
                elif statistic == "max": value = max(values)
                else: value = sum(values)
                row[field] = value
            result.append(row)
    return result


def pair_key(row):
    return tuple(str(row.get(k,"")) for k in ("tf","train_assay","test_assay","seed","evaluation_id",
        "test_cache_sha256","split_hash","evaluation_crops_json","evaluation_rc"))


def add_reference_comparisons(rows, root, reference_id, experiment_id):
    reference_file = root/"results"/"Results_all"/f"{reference_id}.csv"
    if experiment_id == reference_id:
        baseline = [r for r in rows if r["row_type"]=="evaluation" and truth(r.get("included_in_summary"))]
    elif reference_file.exists():
        with reference_file.open(newline="") as handle:
            baseline = [r for r in csv.DictReader(handle) if r["row_type"]=="evaluation" and truth(r.get("included_in_summary"))]
    else:
        return
    lookup = {pair_key(r):r for r in baseline}
    for row in rows:
        if row["row_type"] != "evaluation": continue
        ref = lookup.get(pair_key(row))
        if ref is None: continue
        value, reference = number(row.get("auprc")),number(ref.get("auprc"))
        if value is None or reference is None: continue
        row["paired_reference_run_id"] = ref["run_id"]
        row["delta_auprc"] = value-reference
        row["win_fraction"] = float(value>reference)
        if reference>0:
            row["ratio_auprc"] = value/reference
            if value>0: row["log2_ratio_auprc"] = math.log2(value/reference)


def collect(root: Path, experiment_id: str):
    experiment = root/"results"/experiment_id
    config = json.loads((experiment/"config.json").read_text())
    destination = root/"results"/"Results_all"
    destination.mkdir(parents=True,exist_ok=True)
    with (destination/f".{experiment_id}.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _collect(root,experiment,config,destination)


def _collect(root, experiment, config, destination):
    rows,metadata = [],{}
    for path in sorted((experiment/"runs").rglob("run.json")):
        meta = json.loads(path.read_text())
        metadata[meta["run_id"]] = meta
        records_path = path.parent/"records.jsonl"
        if records_path.exists():
            for line in records_path.read_text().splitlines():
                if line.strip(): rows.append(json.loads(line))
        if not any(r["run_id"]==meta["run_id"] and r["row_type"]=="run" for r in rows):
            row = {k:v for k,v in meta.items() if k in COLUMNS}
            row.update(schema_version=SCHEMA_VERSION,row_type="run",epochs_run=meta.get("epochs_run",0),
                status=meta["status"],error=meta.get("error",""))
            rows.append(row)
    if not rows:
        raise ValueError(f"No recorded runs for {config['name']}")
    # A repeated attempt with the same TF/source/seed is not an additional seed.
    # Pick the latest completed attempt, retain every attempt in the raw CSV.
    selected = {}
    for meta in sorted(metadata.values(),key=lambda m:m["started_at"]):
        if meta["status"] != "completed": continue
        key=(meta.get("training_scope","single_tf"),meta.get("tf"),meta.get("tf_ids_json"),meta["train_assay"],meta["seed"],meta["config_hash"])
        selected[key]=meta["run_id"]
    active = set(selected.values())
    code_hashes = {metadata[r]["code_hash"] for r in active}
    config_hashes = {metadata[r]["config_hash"] for r in active}
    if len(code_hashes)>1 or len(config_hashes)>1:
        raise ValueError("Selected runs mix code/config versions. Use a distinct experiment name for a changed implementation.")
    for row in rows:
        row["included_in_summary"] = row["run_id"] in active and row.get("status")=="completed"
    # Enrich evaluation rows with source-run descriptors without copying runtime
    # totals onto every target-assay row.
    run_lookup = {r["run_id"]:r for r in rows if r["row_type"]=="run"}
    descriptive = {"epochs_requested","epochs_run","best_epoch","parameter_count","trainable_parameter_count",
        "kernels_json","channels_per_branch_json","input_length","batch_size","eval_batch_size","num_workers",
        "learning_rate_initial","weight_decay","patience","min_checkpoint_epoch","negatives_per_positive",
        "pos_weight","checkpoint_objective","validation_selection_score","amp_enabled","torch_version",
        "device_name","train_positive","train_negative","val_positive","val_negative"}
    for row in rows:
        if row["row_type"]=="evaluation":
            source = run_lookup.get(row["run_id"],{})
            for key in descriptive:
                if row.get(key) is None: row[key]=source.get(key)
    add_reference_comparisons(rows,root,config["reference"],config["name"])
    aggregated = []
    aggregated += summaries(rows,"tf","run",["tf","tf_ids_json","training_scope","train_assay"])
    aggregated += summaries(rows,"tf","evaluation",["tf","train_assay","test_assay","direction","evaluation_id"])
    aggregated += summaries(rows,"experiment","run",["train_assay"])
    aggregated += summaries(rows,"experiment","run",[])
    aggregated += summaries(rows,"experiment","evaluation",["train_assay","test_assay","direction","evaluation_id"])
    all_rows = rows+aggregated
    chosen = [metadata[r] for r in active]
    if chosen:
        started=min(datetime.fromisoformat(m["started_at"]) for m in chosen)
        finished=max(datetime.fromisoformat(m["finished_at"]) for m in chosen)
        all_rows.append({"schema_version":SCHEMA_VERSION,"row_type":"summary","experiment_id":config["name"],
            "reference_id":config["reference"],"summary_level":"experiment","summary_source":"run",
            "statistic":"span","aggregation_unit":"calendar_span","status":"completed","included_in_summary":True,
            "metric_name":"experiment_elapsed_wall_s","metric_value":(finished-started).total_seconds(),
            "metric_unit":"s","n_runs":len(chosen)})
    csv_path=destination/f"{config['name']}.csv"
    tmp=csv_path.with_name(csv_path.name+f".{os.getpid()}.tmp")
    with tmp.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=COLUMNS,extrasaction="raise")
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k:("" if v is None else v) for k,v in row.items()})
    tmp.replace(csv_path)
    write_report(experiment,config,rows,aggregated,csv_path)
    atomic_json(destination/"schema.json",{"schema_version":SCHEMA_VERSION,"columns":COLUMNS,
        "row_types":["run","epoch","evaluation","analysis","summary"],
        "time_unit":"seconds","memory_unit":"bytes","auprc_definition":"sklearn average_precision_score"})
    print(f"Collected {config['name']}: {len(rows)} raw rows, {len(aggregated)} summary rows -> {csv_path}")
    return csv_path


def fmt(value,digits=4):
    n=number(value)
    return "—" if n is None else f"{n:.{digits}f}"


def write_report(experiment,config,rows,aggregated,csv_path):
    runs=[r for r in rows if r["row_type"]=="run"]
    selected=[r for r in runs if truth(r.get("included_in_summary"))]
    lines=[f"# {config['name']}","",config.get("description",""),"",
        f"Reference: `{config['reference']}`. Selected completed runs: {len(selected)}. Recorded attempts: {len(runs)}.",
        f"Standardized CSV: `{csv_path.relative_to(experiment.parent.parent)}`.","",
        "## Test metrics", "", "Seeds are averaged per TF before aggregation across TFs.","",
        "| Direction | TFs | Runs | AUPRC mean | AUPRC median | AUROC mean | MCC mean | Δ AUPRC mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    means=[r for r in aggregated if r["summary_level"]=="experiment" and r["summary_source"]=="evaluation" and r["statistic"]=="mean"]
    for row in means:
        median=next(r for r in aggregated if r["summary_level"]=="experiment" and r["summary_source"]=="evaluation" and r["statistic"]=="median" and r.get("direction")==row["direction"])
        lines.append(f"| {row['direction']} | {row['n_tfs']} | {row['n_runs']} | {fmt(row.get('auprc'))} | {fmt(median.get('auprc'))} | {fmt(row.get('auroc'))} | {fmt(row.get('mcc'))} | {fmt(row.get('delta_auprc'))} |")
    lines += ["","## Runtime per TF and source assay","",
        "Training totals count each trained model once. Epoch time includes training, validation, mining and epoch overhead before checkpoint/report writing.","",
        "| TF | Source | Seeds | Epochs mean | Epoch seconds mean | Training seconds sum | Inference seconds sum |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for row in [r for r in aggregated if r["summary_level"]=="tf" and r["summary_source"]=="run" and r["statistic"]=="mean"]:
        summed=next(r for r in aggregated if r["summary_level"]=="tf" and r["summary_source"]=="run" and r["statistic"]=="sum" and r["tf"]==row["tf"] and r["train_assay"]==row["train_assay"])
        lines.append(f"| {row['tf']} | {row['train_assay']} | {row['n_seeds']} | {fmt(row.get('epochs_run'),1)} | {fmt(row.get('epoch_mean_s'),2)} | {fmt(summed.get('training_wall_s'),2)} | {fmt(summed.get('inference_s'),2)} |")
    total=next((r for r in aggregated if r["summary_level"]=="experiment" and r["summary_source"]=="run" and not r.get("train_assay") and r["statistic"]=="sum"),{})
    lines += ["",f"Summed training time: {fmt(total.get('training_wall_s'),2)} s. Summed job time including inference: {fmt(total.get('total_wall_s'),2)} s.",
        "Calendar span is recorded separately in the CSV and can include gaps between job submissions.","",
        "## Attempts", "", "| TF | Source | Seed | Run | Status | Included |", "|---|---|---:|---|---|---|"]
    for row in runs:
        lines.append(f"| {row.get('tf','')} | {row.get('train_assay','')} | {row.get('seed','')} | {row['run_id']} | {row.get('status','')} | {row.get('included_in_summary',False)} |")
    lines += ["","## Interpretation","",
        "AUPRC is average precision. MCC/F1 use the threshold selected on source validation only.",
        "Raw `evaluation` rows contain individual TF/source/target/seed results. `run` rows contain runtime totals; `epoch` rows contain the complete epoch trajectory.",
        "Repeated attempts are retained; only the latest completed attempt per TF/source/seed contributes to summaries.",""]
    path=experiment/"evaluation.md"
    tmp=path.with_name(path.name+f".{os.getpid()}.tmp")
    tmp.write_text("\n".join(lines))
    tmp.replace(path)
