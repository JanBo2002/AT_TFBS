"""Numerical regression checks against the unchanged uploaded reference."""
from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from . import hum301_data
from .data import TFBindingDataset, CyclicBalancedNegativeSampler
from .defaults import training_namespace
from .epoch import classification_forward
from .model import HumanTFBindingModel, ModelConfig
from .recorder import atomic_json, file_hash
from .training import train_command
from .utils import seed_everything


def reference_module(root):
    sys.modules.setdefault("hum301_data",hum301_data)
    name="_hum301_frozen_reference"
    spec=importlib.util.spec_from_file_location(name,root/"reference"/"hum301_motif64.py")
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def synthetic_cache(path: Path, seed=101):
    """Tiny valid cache for software checks; never scientific experiment data."""
    rng=np.random.default_rng(seed)
    labels=np.array([1,1]+[0]*20+[1,1,0,0]+[1,1,0,0],dtype=np.uint8)
    split=np.array([0]*22+[1]*4+[2]*4,dtype=np.uint8)
    n=len(labels)
    starts=np.arange(n,dtype=np.int64)*1000
    np.savez_compressed(path,sequences=rng.integers(0,4,size=(n,301),dtype=np.uint8),labels=labels,
        split=split,chromosomes=np.array([f"chr{int(s)+1}" for s in split]),starts=starts,ends=starts+301,
        sample_ids=np.array([f"synthetic_{seed}_{i}" for i in range(n)]),
        summary_json=np.array(json.dumps({"format_version":"at301-v1","sequence_audit":{},"splits":{}})))


def same_state(left,right):
    a,b=left.state_dict(),right.state_dict()
    if a.keys()!=b.keys(): raise AssertionError("State-dict keys changed")
    for name in a:
        if not torch.equal(a[name],b[name]):
            raise AssertionError(f"State mismatch: {name}; max difference {float((a[name]-b[name]).abs().max())}")


def same_outputs(a,b):
    if a.keys()!=b.keys(): raise AssertionError("Output names changed")
    for key in a:
        if not torch.equal(a[key],b[key]): raise AssertionError(f"Output mismatch: {key}")


def verify(root: Path, output: Path):
    previous_threads=torch.get_num_threads()
    torch.set_num_threads(2)
    reference=reference_module(root)
    checks=[]
    try:
        original=reference.build_parser().parse_args(["train","--cache","CACHE","--output-dir","OUTPUT"])
        current=vars(training_namespace())
        for key,value in current.items():
            if getattr(original,key)!=value: raise AssertionError(f"Changed default {key}")
        checks.append("all original argparse training defaults preserved")
        for seed in (1,13,42):
            reference.seed_everything(seed)
            old=reference.HumanTFBindingModel(reference.ModelConfig())
            seed_everything(seed)
            new=HumanTFBindingModel(ModelConfig())
            same_state(old,new)
        checks.append("state-dict names, initial weights and buffers bitwise equal for seeds 1/13/42")
        old.eval();new.eval()
        generator=torch.Generator().manual_seed(99)
        codes=torch.randint(0,4,(2,256),generator=generator)
        sequence=torch.nn.functional.one_hot(codes,4).transpose(1,2).float()
        centers=torch.tensor([128,127])
        with torch.no_grad(): same_outputs(old(sequence,centers),new(sequence,centers))
        checks.append("all forward outputs bitwise equal, including center indices 128 and 127")
        full=torch.nn.functional.one_hot(torch.randint(0,4,(2,301),generator=generator),4).transpose(1,2).float()
        for crops in ("center","multi"):
            for rc in (False,True):
                with torch.no_grad():
                    a=reference.classification_forward(old,full,centers,crops,rc)
                    b=classification_forward(new,full,centers,crops,rc)
                same_outputs(a,b)
        checks.append("center/multi crop and RC probability averaging bitwise equal")
        labels=torch.tensor([0.,1.])
        for epoch in range(3):
            opt_old=torch.optim.AdamW(old.parameters(),lr=3e-4,weight_decay=1e-4)
            opt_new=torch.optim.AdamW(new.parameters(),lr=3e-4,weight_decay=1e-4)
            old.train();new.train()
            for model,opt in ((old,opt_old),(new,opt_new)):
                opt.zero_grad(set_to_none=True)
                torch.manual_seed(1000+epoch)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(model(sequence,centers)["classification_logit"],labels,pos_weight=torch.tensor(10.))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
            same_state(old,new)
        checks.append("dropout, gradients, clipping, AdamW updates and normalization buffers bitwise equal")
        with tempfile.TemporaryDirectory(prefix="hum301_verify_") as temp:
            temp=Path(temp)
            cache_path=temp/"dataset.npz"
            synthetic_cache(cache_path)
            cache=hum301_data.load_cache(cache_path)
            indices=np.flatnonzero(cache["split"]==0)
            old_data=reference.TFBindingDataset(cache,indices,0.5,seed=13,training=True)
            new_data=TFBindingDataset(cache,indices,0.5,seed=13,training=True)
            for epoch in (0,1,5):
                old_data.set_epoch(epoch);new_data.set_epoch(epoch)
                for index in (0,1,2,10,0):
                    a,b=old_data[index],new_data[index]
                    for key in a:
                        if torch.is_tensor(a[key]):
                            if not torch.equal(a[key],b[key]): raise AssertionError(f"Dataset mismatch {key}")
                        elif a[key]!=b[key]: raise AssertionError(f"Dataset mismatch {key}")
            checks.append("fresh random crops, RC augmentation and per-worker RNG match")
            old_sampler=reference.CyclicBalancedNegativeSampler(old_data,seed=13,negatives_per_positive=10,batch_size=88)
            new_sampler=CyclicBalancedNegativeSampler(new_data,seed=13,negatives_per_positive=10,batch_size=88)
            for epoch in range(5):
                a,b=list(old_sampler),list(new_sampler)
                if a!=b: raise AssertionError("Sampler order changed")
            checks.append("cyclic negative selection and batch order match over five epochs")
            # A full training trajectory catches scheduler/checkpoint/threshold
            # regressions that a forward-only smoke check cannot detect.
            original=reference.build_parser().parse_args(["train","--cache",str(cache_path),"--output-dir",str(temp/"old"),
                "--epochs","2","--num-workers","0","--eval-batch-size","8","--device","cpu","--no-amp","--skip-test"])
            new_args=copy.deepcopy(original)
            new_args.output_dir=temp/"new"
            new_args.motif_kernels=(9,12)
            class Recorder:
                def update(self,**kwargs): pass
                def epoch(self,row): pass
                def fit_loop_finished(self,*args): pass
            buffer=io.StringIO()
            try:
                with contextlib.redirect_stdout(buffer),contextlib.redirect_stderr(buffer):
                    reference.train_command(original)
                    train_command(new_args,recorder=Recorder())
            except Exception:
                print(buffer.getvalue()[-5000:])
                raise
            a=torch.load(temp/"old"/"model_for_prediction.pt",map_location="cpu",weights_only=True)
            b=torch.load(temp/"new"/"model_for_prediction.pt",map_location="cpu",weights_only=True)
            for key in a["model_state"]:
                if not torch.equal(a["model_state"][key],b["model_state"][key]): raise AssertionError(f"Training state differs: {key}")
            for key in ("epoch","validation_score","classification_threshold","checkpoint_objective","evaluation_crop_mode","evaluation_reverse_complement"):
                if a[key]!=b[key]: raise AssertionError(f"Checkpoint metadata differs: {key}")
            if (temp/"old"/"validation_predictions.tsv").read_bytes()!=(temp/"new"/"validation_predictions.tsv").read_bytes():
                raise AssertionError("Validation predictions differ")
            old_history=json.loads((temp/"old"/"history.json").read_text())
            new_history=json.loads((temp/"new"/"history.json").read_text())
            for old_row,new_row in zip(old_history,new_history):
                for key in ("train","val","sampling","train_shift","learning_rate_used","learning_rate_next"):
                    if old_row[key]!=new_row[key]: raise AssertionError(f"Epoch history differs: {key}")
            checks.append("two full training epochs: weights, LR schedule, metrics, selected checkpoint and validation-MCC threshold bitwise equal")
        variants=[]
        for path in sorted((root/"configs").glob("*.json")):
            c=json.loads(path.read_text())
            params=c["predictor"]["params"]
            params={k:tuple(v) if k in {"motif_kernels","encoder_channels"} else v for k,v in params.items()}
            model=HumanTFBindingModel(ModelConfig(**params)).eval()
            with torch.no_grad(): result=model(sequence,centers)
            if result["classification_track"].shape!=(2,256): raise AssertionError("Invalid track shape")
            widths=[branch.out_channels for branch in model.embedder.initial.branches]
            if sum(widths)!=64: raise AssertionError("Input channel budget changed")
            if len(widths)==4 and widths!=[16]*4: raise AssertionError("Four-branch channel allocation changed")
            variants.append({"name":c["name"],"kernels":list(model.config.motif_kernels),"channels":widths,
                "parameters":sum(p.numel() for p in model.parameters())})
        checks.append("all three configurations forward successfully; channel allocation 32/32 or 16/16/16/16")
        report={"status":"passed","device":"cpu","torch_version":str(torch.__version__),
            "reference_sha256":file_hash(root/"reference"/"hum301_motif64.py"),
            "data_dependency_sha256":file_hash(root/"ablation"/"hum301_data.py"),
            "checks":checks,"variants":variants,
            "limits":"Synthetic regression tests; no scientific TF training data were available here. GPU/ROCm full-run reproducibility depends on the original environment."}
        atomic_json(output,report)
        print(json.dumps(report,indent=2))
        return report
    finally:
        torch.set_num_threads(previous_threads)
