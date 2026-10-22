"""Optional planners remove the assumption of one separately trained model per TF."""
from __future__ import annotations

import json


def multi_tf(config,manifest,tf_filter=None,train_assays=None,seeds=None):
    """One task per source assay and seed, containing every selected TF.

    Select this planner together with a multi-TF workflow/data/fit factory. This
    only plans the tasks; no multi-TF learning algorithm is implemented here.
    """
    tfs=list(dict.fromkeys(r["tf"] for r in manifest))
    if tf_filter: tfs=[t for t in tfs if t in tf_filter]
    lookup={(r["tf"],r["assay"]):r["cache"] for r in manifest}
    rows=[]
    for source in train_assays or config["train_assays"]:
        paths={}
        for tf in tfs:
            for assay in set(config["test_assays"])|{source}:
                if (tf,assay) not in lookup: raise ValueError(f"Missing {tf}/{assay}")
                paths[f"{tf}|{assay}"]=str(lookup[(tf,assay)])
        for seed in seeds or config["seeds"]:
            rows.append({"tf":"","tf_ids_json":json.dumps(tfs),"training_scope":"multi_tf",
                "train_assay":source,"seed":seed,"cache_paths_json":json.dumps(paths,sort_keys=True)})
    return rows
