"""Funktionstests der neuen Optionen (exp-Aktivierung, RC-Sharing, PWM-Hilfsziel, Scanner-Maske)."""
import importlib.util, sys, math, json, types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import argparse
_a = argparse.ArgumentParser(); _a.add_argument('--orig', required=True); _a.add_argument('--new', required=True); _a.add_argument('--pwm-root', required=True, help='PWM-Wurzel mit selected.tsv und FAKE/CHS/selected/*.pwm (synthetisch, siehe make_synthetic_project.py)')
_args = _a.parse_args()
ORIG = Path(_args.orig).resolve(); NEW = Path(_args.new).resolve()
sys.path.insert(0, str(NEW))
def load(path, alias):
    spec = importlib.util.spec_from_file_location(alias, path); m = importlib.util.module_from_spec(spec)
    sys.modules[alias] = m; spec.loader.exec_module(m); return m
data_new = load(NEW / 'data.py', 'data'); model = load(NEW / 'model.py', 'model'); model_old = load(ORIG / 'model.py', 'model_old')

failures = []
def check(name, ok, detail=""):
    print(("PASS " if ok else "FAIL ") + name + (f" — {detail}" if detail else ""))
    if not ok: failures.append(name)

def onehot(batch, length, seed=0, n_positions=()):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 4, (batch, length), generator=g)
    x = F.one_hot(codes, 4).permute(0, 2, 1).float()
    for b, start, end in n_positions: x[b, :, start:end] = 0
    return x
def revcomp(x): return x[:, [3, 2, 1, 0], :].flip(-1)

# ------------------------------------------------ exp-Aktivierung
torch.manual_seed(1)
conv = model.MotifConv1d(64, (9, 12), activation="exp")
x = onehot(5, 256)
out = conv(x)
check("exp: Ausgabe strikt positiv und endlich", bool((out > 0).all()) and bool(torch.isfinite(out).all()), f"max {out.max():.3f}")
# Begrenzung: große Vorsaktivierung läuft nicht über
with torch.no_grad():
    for c in conv.branches: c.weight.mul_(50)
out_big = conv(x)
check("exp: Begrenzung greift (max = exp(10))", bool(torch.isfinite(out_big).all()) and abs(float(out_big.max()) - math.exp(10)) < 1e-2)
torch.manual_seed(1); conv = model.MotifConv1d(64, (9, 12), activation="exp")
with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
    out_amp = conv(x)
check("exp: unter Autocast endlich", bool(torch.isfinite(out_amp.float()).all()), f"dtype {out_amp.dtype}")
# Standard "none" = Original-MotifConv1d
torch.manual_seed(2); a = model.MotifConv1d(64, (9, 12)); torch.manual_seed(2); b = model_old.MotifConv1d(64, (9, 12))
check("MotifConv1d Standard bit-gleich zum Original", torch.equal(a(x), b(x)))
# Modell mit exp: Vorwärts/Rückwärts
torch.manual_seed(3); m = model.HumanTFBindingModel(model.ModelConfig(motif_activation="exp")); m.train()
centers = torch.full((5,), 128, dtype=torch.long)
loss = m(x, centers)["classification_logit"].sum(); loss.backward()
check("exp-Modell: Gradienten endlich", all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))

# ------------------------------------------------ RC-Sharing
torch.manual_seed(4); conv = model.MotifConv1d(64, (9, 11), rc_sharing=True)
learned = sum(p.numel() for n, p in conv.named_parameters() if n.endswith("weight"))
torch.manual_seed(4); plain = model.MotifConv1d(64, (9, 11))
plain_learned = sum(p.numel() for n, p in plain.named_parameters() if n.endswith("weight"))
check("RC-Sharing: halbe Filtergewichte", learned * 2 == plain_learned, f"{learned} vs {plain_learned}")
check("RC-Sharing: 64 Ausgabekanäle", conv(x).shape[1] == 64)
# Äquivarianz bei ungeraden Kerneln: out(rc(x)) = swap(fwd<->rc)(flip(out(x)))
y = conv(x); y_rc = conv(revcomp(x))
blocks = []
for width in (32, 32):  # je Branch: [16 fwd | 16 rc]
    blocks.append((width // 2, width // 2))
swapped = []
offset = 0
for f, r in blocks:
    swapped.append(y[:, offset + f: offset + f + r]); swapped.append(y[:, offset: offset + f]); offset += f + r
expected = torch.cat(swapped, dim=1).flip(-1)
check("RC-Sharing: strangsymmetrisch (ungerade Kernel 9/11)", torch.allclose(y_rc, expected, atol=1e-5), f"max |diff| {float((y_rc-expected).abs().max()):.2e}")
# Mit Kernel 12 (gerade) ist die Symmetrie wie beim Original um eine Position versetzt — nur prüfen, dass es läuft
torch.manual_seed(5); conv12 = model.MotifConv1d(64, (9, 12), rc_sharing=True); check("RC-Sharing mit 9/12 lauffähig", conv12(x).shape == (5, 64, 256))
try:
    model.MotifConv1d(63, (9, 12), rc_sharing=True); check("RC-Sharing: ungerades Budget abgelehnt", False)
except ValueError as e: check("RC-Sharing: ungerades Budget abgelehnt", True, str(e))
torch.manual_seed(6); m = model.HumanTFBindingModel(model.ModelConfig(motif_rc_sharing=True)); m.train()
loss = m(x, centers)["classification_logit"].sum(); loss.backward()
check("RC-Modell: Vorwärts/Rückwärts ok", all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()))
sd_keys = list(m.state_dict())
check("RC-Modell: Parameternamen unverändert (nur Formen)", sd_keys == list(model.HumanTFBindingModel(model.ModelConfig()).state_dict()))

# ------------------------------------------------ Scanner-Maske
rng = np.random.default_rng(0)
matrices = [[[float(v) for v in row] for row in rng.normal(size=(w, 4))] for w in (8, 10, 12)]
sc = model.FixedPWMScanner(matrices, {"schema_version": "2.0"})
xs = onehot(3, 256, seed=9, n_positions=[(0, 100, 103)])
scores, valid = sc.score_with_mask(xs)
check("score_with_mask: forward bit-gleich", torch.equal(scores, sc(xs)))
sc_old = model_old.FixedPWMScanner(matrices, {"schema_version": "2.0"})
check("Scanner neu == Original", torch.equal(sc(xs), sc_old(xs)))
# Maske: Breite 8 -> left 3, gültige Positionen 3..251 (249 Fenster), für Probe 0 zusätzlich Fenster über N (100..102) ungültig
k8 = valid[1, 0]; check("Maske Breite 8: 249 gültige Positionen ohne N", int(k8.sum()) == 249 and bool(k8[3]) and not bool(k8[2]) and bool(k8[251]) and not bool(k8[252]))
k8n = valid[0, 0]; check("Maske: Fenster über N ungültig", int(k8n.sum()) == 249 - 10, f"{int(k8n.sum())}")

# ------------------------------------------------ PWM-Hilfsziel
head = model.PWMTargetHead(64, matrices, {"schema_version": "2.0"})
feats = torch.randn(3, 64, 256, requires_grad=True)
aux = head(feats, xs)
check("PWMTargetHead: skalarer endlicher Verlust", aux.ndim == 0 and bool(torch.isfinite(aux)))
aux.backward(); check("PWMTargetHead: Gradient erreicht die Merkmale", feats.grad is not None and bool(feats.grad.abs().sum() > 0))
# Standardisierung: Zufallsfenster ~ N(0,1)
with torch.no_grad():
    sco, val = head.scanner.score_with_mask(onehot(64, 256, seed=11))
    z = (sco - head.target_mean[None, :, None]) / head.target_std[None, :, None]
    zz = z[val]
check("Ziel standardisiert (Zufallssequenz: Mittel ~0, Std ~1)", abs(float(zz.mean())) < 0.1 and 0.85 < float(zz.std()) < 1.15, f"mean {float(zz.mean()):.3f} std {float(zz.std()):.3f}")
# Modell über Builder, ohne Recorder/Run-Dir, mit Attrappen-Kontext
ctx = types.SimpleNamespace(tf="FAKE", train_assay="CHIP", config={}, artifacts={}, recorder=None, run_dir=None)
m = model.maa_pwm_target(ctx, pwm_root=_args.pwm_root, pwm_count=3, pwm_target_weight=2.0, **{"motif_kernels": [9, 12]})
m.train(); out = m(x, centers)
check("maa_pwm_target: auxiliary_loss im Ausgang", "auxiliary_loss" in out and bool(torch.isfinite(out["auxiliary_loss"])), f"{float(out['auxiliary_loss']):.3f}")
check("maa_pwm_target: Gewicht 2 angewendet", abs(float(out["auxiliary_loss"]) - 2.0 * float(m.pwm_target(out["embedding"], x))) < 1e-5)
base_keys = list(model.HumanTFBindingModel(model.ModelConfig()).state_dict())
keys = list(m.state_dict())
check("maa_pwm_target: bestehende Parameter unverändert benannt, neue hinten", keys[:len(base_keys)] == base_keys, f"+{len(keys)-len(base_keys)} Einträge")
ctx2 = types.SimpleNamespace(tf="FAKE", train_assay="CHIP", config={}, artifacts={}, recorder=None, run_dir=None)
m2 = model.maa_pwm_target(ctx2, pwm_root=_args.pwm_root, pwm_count=3, pwm_target_weight=2.0, **{"motif_kernels": [9, 12]})
m2.load_state_dict(m.state_dict(), strict=True); check("maa_pwm_target: state_dict rund (extra_state Manifest)", True)
m.eval(); out_eval = m(x, centers); check("maa_pwm_target: eval liefert Logits", "classification_logit" in out_eval)
# Loss-Komposition in training.py
training = load(NEW / 'training.py', 'training')
tc = training.TrainingConfig(task_mode="classification") if hasattr(training, "TrainingConfig") else None
crit = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(10.0))
batch = {"label": torch.tensor([1., 0., 1., 0., 1.])}
m.train(); out = m(x, centers)
total, comps = training.compute_multitask_loss(out, batch, crit, tc)
check("compute_multitask_loss: total = Klassifikation + Hilfsziel", abs(float(total) - float(comps["classification"]) - float(out["auxiliary_loss"])) < 1e-5)
out_plain = model.HumanTFBindingModel(model.ModelConfig())(x, centers)
total2, comps2 = training.compute_multitask_loss(out_plain, batch, crit, tc)
check("compute_multitask_loss ohne Hilfsziel unverändert", float(total2) == float(comps2["classification"]))

print("\nFEHLER:", failures if failures else "keine")
sys.exit(1 if failures else 0)
