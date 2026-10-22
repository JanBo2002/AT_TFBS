"""Integration tests use generated caches, never the user's scientific data."""
from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from ablation.collect import collect, summaries
from ablation.schema import COLUMNS
from ablation.verification import synthetic_cache
from ablation.planning import multi_tf
from ablation.specs import recipe_digest


class PackageTests(unittest.TestCase):
    def test_additional_seeds_keep_recipe_identity(self):
        config=json.loads((ROOT/"configs"/"00_basic.json").read_text())
        changed=json.loads(json.dumps(config))
        changed["seeds"]=[1,2,3]
        changed["description"]="More repetitions"
        self.assertEqual(recipe_digest(config),recipe_digest(changed))
        changed["predictor"]["params"]["motif_kernels"]=[7,11]
        self.assertNotEqual(recipe_digest(config),recipe_digest(changed))
    def test_seed_then_tf_macro_average(self):
        rows=[]
        for tf,seed,auprc in [("TFX",1,.1),("TFX",2,.9),("TFY",1,.8)]:
            rows.append(dict(schema_version="1.0",experiment_id="test",reference_id="ref",config_hash="x",
                code_hash="x",row_type="evaluation",included_in_summary=True,tf=tf,seed=seed,
                run_id=f"{tf}_{seed}",train_assay="CHIP",test_assay="GHTS",auprc=auprc))
        result=summaries(rows,"experiment","evaluation",["train_assay","test_assay"])
        mean=next(r for r in result if r["statistic"]=="mean")
        self.assertAlmostEqual(mean["auprc"],.65)
        self.assertFalse(any(r.get("auprc") is not None for r in result if r["statistic"]=="sum"))

    def test_three_configs_differ_only_in_name_and_input_kernels(self):
        configs=[json.loads(p.read_text()) for p in sorted((ROOT/"configs").glob("*.json"))]
        self.assertEqual(len(configs),3)
        for config in configs:
            config.pop("name")
            config["predictor"]["params"]["motif_kernels"]=[9,12]
        self.assertEqual(configs[0],configs[1])
        self.assertEqual(configs[1],configs[2])

    def test_win_fraction_uses_tf_mean_delta(self):
        rows=[]
        for tf,seed,delta in [("TFX",1,.2),("TFX",2,-.1),("TFY",1,-.2)]:
            rows.append(dict(schema_version="1.0",experiment_id="test",reference_id="ref",config_hash="x",
                code_hash="x",row_type="evaluation",included_in_summary=True,tf=tf,seed=seed,
                run_id=f"{tf}_{seed}",train_assay="CHIP",test_assay="GHTS",delta_auprc=delta,win_fraction=float(delta>0)))
        result=summaries(rows,"experiment","evaluation",["train_assay","test_assay"])
        self.assertEqual(next(r for r in result if r["statistic"]=="mean")["win_fraction"],.5)

    def test_slurm_spooled_script_uses_explicit_project_root(self):
        with tempfile.TemporaryDirectory(prefix="slurm_spool_check_") as temp:
            temp=Path(temp)
            fake=temp/"fake_python"
            fake.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$ABLATION_CAPTURE"\n')
            fake.chmod(0o755)
            capture=temp/"args.txt"
            env=dict(os.environ,ABLATION_PYTHON=str(fake),ABLATION_CAPTURE=str(capture),SLURM_ARRAY_TASK_ID="3")
            for name,args in [("train_task.sbatch",[str(temp/"tasks.csv"),"3",str(ROOT)]),
                              ("train_array.sbatch",[str(temp/"tasks.csv"),str(ROOT)]),
                              ("collect_all.sbatch",[str(ROOT)])]:
                spool=temp/"slurm_script"
                shutil.copyfile(ROOT/name,spool)
                subprocess.run(["bash",str(spool),*args],env=env,check=True)
                self.assertEqual(capture.read_text().splitlines()[0],str(ROOT/"run_experiment.py"))

    def test_multi_tf_planner_creates_shared_training_tasks(self):
        config=json.loads((ROOT/"configs"/"00_basic.json").read_text())
        manifest=[{"tf":tf,"assay":a,"cache":Path(f"/{tf}/{a}.npz")} for tf in ("CREB3L3","MAX") for a in ("CHIP","GHTS")]
        tasks=multi_tf(config,manifest)
        self.assertEqual(len(tasks),2)
        self.assertTrue(all(r["training_scope"]=="multi_tf" for r in tasks))
        self.assertEqual(json.loads(tasks[0]["tf_ids_json"]),["CREB3L3","MAX"])

    def test_full_pipeline_all_variants_and_retry(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory(prefix="ablation_pipeline_") as temp:
            temp=Path(temp)
            package=temp/"big_ablation"
            shutil.copytree(ROOT,package,ignore=shutil.ignore_patterns("__pycache__","results","verification.json"))
            for p in (package/"configs").glob("*.json"):
                config=json.loads(p.read_text())
                config["fitting"]["params"].update(epochs=2,num_workers=0,eval_batch_size=8,amp=False)
                p.write_text(json.dumps(config,indent=2)+"\n")
            manifest=package/"configs"/"caches.csv"
            with manifest.open("w",newline="") as handle:
                writer=csv.writer(handle);writer.writerow(["tf","assay","cache"])
                for index,assay in enumerate(("CHIP","GHTS")):
                    path=temp/"cache"/"CREB3L3"/assay/"dataset.npz"
                    path.parent.mkdir(parents=True)
                    synthetic_cache(path,seed=100+index)
                    writer.writerow(["CREB3L3",assay,path])
            spec=importlib.util.spec_from_file_location("_integration_ablation_runner",package/"run_experiment.py")
            runner=importlib.util.module_from_spec(spec);sys.modules[spec.name]=runner;spec.loader.exec_module(runner)
            buffer=io.StringIO()
            old_argv=sys.argv
            try:
                with contextlib.redirect_stdout(buffer),contextlib.redirect_stderr(buffer):
                    sys.argv=["run_experiment.py","run","--all-configs","--manifest",str(manifest),"--device","cpu","--workers","0"]
                    self.assertEqual(runner.main(),0)
                headers=[]
                for config_path in sorted((package/"configs").glob("*.json")):
                    name=json.loads(config_path.read_text())["name"]
                    path=package/"results"/"Results_all"/f"{name}.csv"
                    with path.open(newline="") as handle:
                        reader=csv.DictReader(handle);headers.append(reader.fieldnames);rows=list(reader)
                    self.assertEqual(headers[-1],COLUMNS)
                    runs=[r for r in rows if r["row_type"]=="run"]
                    epochs=[r for r in rows if r["row_type"]=="epoch"]
                    evaluations=[r for r in rows if r["row_type"]=="evaluation"]
                    self.assertEqual(len(runs),2)
                    self.assertEqual(len(epochs),4)
                    self.assertEqual(len(evaluations),4)
                    self.assertTrue(all(r["status"]=="completed" for r in runs))
                    self.assertTrue(all(r["epochs_run"]=="2" for r in runs))
                    self.assertTrue(all(r["val_auprc"]!="" and r["train_loss"]!="" for r in runs))
                    self.assertTrue(all(r["val_threshold"]==r["threshold"] for r in runs))
                    self.assertTrue(all(r["training_wall_s"]=="" for r in evaluations))
                    self.assertTrue(all(r["delta_auprc"]!="" for r in evaluations))
                    self.assertEqual({r["direction"] for r in evaluations},{"CHIP_to_CHIP","CHIP_to_GHTS","GHTS_to_CHIP","GHTS_to_GHTS"})
                    self.assertTrue((package/"results"/name/"evaluation.md").is_file())
                    total=next(r for r in rows if r["row_type"]=="summary" and r["summary_level"]=="experiment"
                        and r["summary_source"]=="run" and r["statistic"]=="sum" and r["train_assay"]=="")
                    self.assertAlmostEqual(float(total["training_wall_s"]),sum(float(r["training_wall_s"]) for r in runs))
                self.assertEqual(headers[0],headers[1]);self.assertEqual(headers[1],headers[2])
                # A successful retry must not inflate the number of seeds/models.
                with contextlib.redirect_stdout(buffer),contextlib.redirect_stderr(buffer):
                    sys.argv=["run_experiment.py","run","--config","00_basic","--manifest",str(manifest),"--train-assay","CHIP","--device","cpu","--workers","0"]
                    self.assertEqual(runner.main(),0)
                with (package/"results"/"Results_all"/"00_basic.csv").open(newline="") as handle:
                    rows=list(csv.DictReader(handle))
                runs=[r for r in rows if r["row_type"]=="run"]
                self.assertEqual(len(runs),3)
                self.assertEqual(sum(r["included_in_summary"]=="True" for r in runs),2)
            except Exception:
                print(buffer.getvalue()[-12000:])
                raise
            finally:
                sys.argv=old_argv


if __name__=="__main__": unittest.main()
