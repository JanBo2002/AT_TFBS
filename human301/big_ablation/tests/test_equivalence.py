"""Gleichheitstests: neue Programmdateien gegen die Originale bei Standardeinstellungen.

1. Modell: gleiche Parameter (Namen, Formen, Werte bei gleichem Seed) und bit-gleiche
   Ausgaben in train()- und eval()-Modus für die bestehenden Config-Varianten.
2. Datensatz: gleiche Zuschnitte und RC-Entscheidungen (gleicher RNG-Strom) beim
   Standard-Shift 23; neue Bereiche bei Shift 8 und 0.
3. Auswertungs-Crops: "multi"/"center" unverändert, "all" = 46 Starts.
4. PWM-Scanner: forward bit-gleich zum Original.
"""
import importlib.util, sys, json, math
from pathlib import Path
import numpy as np
import torch

def load(name, path, alias):
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module

import argparse
_a = argparse.ArgumentParser(); _a.add_argument('--orig', required=True, help='Ordner mit den Originaldateien (model.py, data.py)'); _a.add_argument('--new', required=True, help='Ordner mit den neuen Dateien'); _a.add_argument('--pwm-root', default=None)
_args = _a.parse_args()
ORIG = Path(_args.orig).resolve(); NEW = Path(_args.new).resolve()
# model.py importiert "data" nur lazy (validate_pwm_split_provenance); beide Varianten separat laden.
sys.path.insert(0, str(NEW))
data_new = load('data', NEW / 'data.py', 'data')
model_new = load('model', NEW / 'model.py', 'model')
data_old = load('data_old', ORIG / 'data.py', 'data_old')
model_old = load('model_old', ORIG / 'model.py', 'model_old')

failures = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""))
    if not ok: failures.append(name)

# ---------------------------------------------------------------- 1. Modell
variants = {
    "basic 9/12 x64": dict(),
    "A26 depth 2": dict(embed_channels=32, embedder_residual_blocks=2),
    "A27 depth 0 x128": dict(embed_channels=128, embedder_residual_blocks=0),
    "A05 kernel 7": dict(motif_kernels=(7,), dropout=0.0, channel_dropout=0.0),
    "A14 3/6/9/12 x128 cdo0.2": dict(motif_kernels=(3, 6, 9, 12), embed_channels=128, channel_dropout=0.2),
    "A28 enc 80/114/128 x32": dict(embed_channels=32, encoder_channels=(80, 114, 128)),
}
for label, params in variants.items():
    torch.manual_seed(13); old = model_old.HumanTFBindingModel(model_old.ModelConfig(**params))
    torch.manual_seed(13); new = model_new.HumanTFBindingModel(model_new.ModelConfig(**params))
    so, sn = old.state_dict(), new.state_dict()
    same_keys = list(so) == list(sn)
    same_values = same_keys and all(torch.equal(so[k], sn[k]) for k in so)
    check(f"state_dict identisch: {label}", same_values, f"{len(so)} Tensoren")
    x = torch.randn(3, 4, 256); x = (x == x.max(dim=1, keepdim=True).values).float()
    centers = torch.full((3,), 128, dtype=torch.long)
    for mode in ("train", "eval"):
        getattr(old, mode)(); getattr(new, mode)()
        torch.manual_seed(5); oo = old(x, centers)
        torch.manual_seed(5); on = new(x, centers)
        same = all(torch.equal(oo[k], on[k]) for k in oo) and set(oo) == set(on)
        check(f"forward bit-gleich ({mode}): {label}", same)
    # Rückwärts: gleiche Gradienten
    old.train(); new.train()
    torch.manual_seed(9); l1 = old(x, centers)["classification_logit"].sum(); l1.backward()
    torch.manual_seed(9); l2 = new(x, centers)["classification_logit"].sum(); l2.backward()
    g_same = all((po.grad is None and pn.grad is None) or (po.grad is not None and pn.grad is not None and torch.equal(po.grad, pn.grad))
                 for po, pn in zip(old.parameters(), new.parameters()))
    check(f"Gradienten bit-gleich: {label}", g_same)

# ------------------------------------------------------------- 2. Datensatz
rng = np.random.default_rng(0)
n = 50
cache = {
    "sequences": rng.integers(0, 4, size=(n, 301)).astype(np.uint8), "labels": rng.integers(0, 2, size=n).astype(np.uint8),
    "chromosomes": np.array(["chr1"] * n), "starts": np.arange(n) * 1000, "sample_ids": np.array([f"s{i}" for i in range(n)]),
    "split": np.zeros(n, dtype=np.uint8),
}
idx = np.arange(n)
def draws(module, shift_max, seed=3, rounds=4):
    ds = module.TFBindingDataset(cache, idx, reverse_complement_probability=0.5, shift_max=shift_max, seed=seed, training=True)
    out = []
    for epoch in range(rounds):
        ds.set_epoch(epoch)
        for i in range(n):
            item = ds[i]; out.append((item["shift"], int(item["center_index"]), item["sequence"].numpy().tobytes()))
    return out
check("Dataset Standard-Shift 23: Zuschnitte/RC bit-gleich", draws(data_old, 23) == draws(data_new, 23), f"{4*n} Zugriffe")
d8 = draws(data_new, 8); shifts8 = {s for s, _, _ in d8}
check("Dataset Shift 8: Verschiebungen in [-8, 8]", min(shifts8) >= -8 and max(shifts8) <= 8, f"beobachtet {min(shifts8)}..{max(shifts8)}")
d0 = draws(data_new, 0); shifts0 = {s for s, _, _ in d0}
check("Dataset Shift 0: immer zentriert (22)", shifts0 == {0})
ds0 = data_new.TFBindingDataset(cache, idx, 0.0, shift_max=0, seed=1, training=True)
seq0 = ds0[0]["sequence"].numpy().argmax(0); check("Shift 0 entspricht Positionen 22..277", np.array_equal(seq0, cache["sequences"][0][22:278]))
check("shift_summary neu bei 23 = Original", data_new.TFBindingDataset(cache, idx, shift_max=23, training=True).shift_summary() == data_old.TFBindingDataset(cache, idx, shift_max=23, training=True).shift_summary())
print("   shift_summary(8):", data_new.TFBindingDataset(cache, idx, shift_max=8, training=True).shift_summary())
# Evaluationsmodus (training=False) unverändert
dso = data_old.TFBindingDataset(cache, idx, training=False); dsn = data_new.TFBindingDataset(cache, idx, training=False)
check("Dataset Evaluation (301 bp) bit-gleich", all(torch.equal(dso[i]["sequence"], dsn[i]["sequence"]) for i in range(n)))

# ---------------------------------------------------------- 3. Crop-Modi
for mode in ("center", "multi"):
    check(f"evaluation_crop_starts {mode} unverändert", data_old.evaluation_crop_starts(301, 256, mode) == data_new.evaluation_crop_starts(301, 256, mode))
starts_all = data_new.evaluation_crop_starts(301, 256, "all")
check("evaluation_crop_starts all = 0..45 (46 Starts)", starts_all == tuple(range(46)))

# ------------------------------------------------------ 4. PWM-Scanner
matrices = [[[float(v) for v in row] for row in rng.normal(size=(w, 4))] for w in (8, 10, 12)]
manifest = {"schema_version": "2.0", "channels": []}
so = model_old.FixedPWMScanner(matrices, manifest); sn = model_new.FixedPWMScanner(matrices, manifest)
x = torch.randn(4, 4, 256); x = (x == x.max(dim=1, keepdim=True).values).float(); x[0, :, 10:14] = 0  # N-Basen
check("FixedPWMScanner forward bit-gleich", torch.equal(so(x), sn(x)))

print("\nFEHLER:", failures if failures else "keine")
sys.exit(1 if failures else 0)
