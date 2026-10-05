#!/usr/bin/env python3
"""Compact DNA classifier for fixed, imbalanced 301-bp datasets.

Default: random 256-bp training crops, three encoder stages, one Transformer,
global max+mean pooling; no decoder or profile regression. Train uses cyclic
1:5 batches with pos_weight=5; validation/test retain every supplied example.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from at301_data import (STORED_LENGTH, SPLIT_NAMES, CyclicNegativePool, balanced_batches,
                        classification_metrics, counts, crop_bounds, evaluation_crop_starts,
                        load_cache, select_mcc_threshold, sha256_file)

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    raise SystemExit("PyTorch is required for maa301.py. Activate your PyTorch environment; data preparation uses NumPy only.")

VERSION = "1.0.0-at301-fixed-test"
ONE_HOT = np.vstack((np.eye(4, dtype=np.float32), np.zeros((1, 4), dtype=np.float32)))


class RMSBatchNorm1d(nn.Module):
    """Same RMS normalization and EMA behavior as the supplied maa(6).py."""
    def __init__(self, channels, eps=1e-5, ema_decay=0.9):
        super().__init__()
        self.eps, self.ema_decay = eps, ema_decay
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.register_buffer("running_mean_square", torch.ones(channels))

    def forward(self, x):
        if self.training:
            mean_square = x.float().square().mean(dim=(0, 2))
            with torch.no_grad():
                self.running_mean_square.mul_(self.ema_decay).add_(mean_square.detach(), alpha=1 - self.ema_decay)
        else:
            mean_square = self.running_mean_square
        out = x / torch.sqrt(mean_square.to(x.dtype)[None, :, None] + self.eps)
        return out * self.weight[None, :, None] + self.bias[None, :, None]


class WSConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gain = nn.Parameter(torch.ones(self.out_channels))

    def forward(self, x):
        dims = tuple(range(1, self.weight.ndim))
        weight = (self.weight - self.weight.mean(dim=dims, keepdim=True)) / torch.sqrt(self.weight.var(dim=dims, unbiased=False, keepdim=True) + 1e-5)
        fan_in = (self.in_channels // self.groups) * self.kernel_size[0]
        weight = weight * (self.gain[:, None, None] / math.sqrt(fan_in))
        return F.conv1d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def activation(name):
    return nn.GELU() if name == "gelu" else nn.ReLU()


class ConvBlock(nn.Module):
    def __init__(self, incoming, outgoing, kernel=5, act="relu"):
        super().__init__()
        self.norm = RMSBatchNorm1d(incoming)
        self.act = activation(act)
        self.conv = WSConv1d(incoming, outgoing, kernel_size=kernel, padding=kernel // 2)

    def forward(self, x):
        return self.conv(self.act(self.norm(x)))


class DNAEmbedder(nn.Module):
    def __init__(self, channels, act):
        super().__init__()
        self.initial = nn.Conv1d(4, channels, kernel_size=15, padding=7)
        self.residual = ConvBlock(channels, channels, act=act)

    def forward(self, x):
        x = self.initial(x)
        return x + self.residual(x)


class DownresBlock(nn.Module):
    def __init__(self, incoming, outgoing, act):
        super().__init__()
        self.outgoing = outgoing
        self.first = ConvBlock(incoming, outgoing, act=act)
        self.second = ConvBlock(outgoing, outgoing, act=act)

    def forward(self, x):
        residual = x[:, :self.outgoing] if x.shape[1] > self.outgoing else F.pad(x, (0, 0, 0, self.outgoing - x.shape[1]))
        x = self.first(x) + residual
        return x + self.second(x)


class TransformerBlock(nn.Module):
    def __init__(self, channels, heads, ffn_channels, dropout, act):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(channels), nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(channels, ffn_channels), activation(act),
                                 nn.Dropout(dropout), nn.Linear(ffn_channels, channels), nn.Dropout(dropout))

    def forward(self, x):
        x = x.transpose(1, 2)
        normalized = self.norm1(x)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        x = x + attended
        return (x + self.ffn(self.norm2(x))).transpose(1, 2)


@dataclass
class ModelConfig:
    input_length: int = 256
    embed_channels: int = 32
    encoder_channels: tuple = (48, 64, 96)
    transformer_blocks: int = 1
    heads: int = 4
    ffn_channels: int = 192
    head_channels: int = 64
    dropout: float = 0.2
    activation: str = "relu"


class TFBindingModel301(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not 8 <= config.input_length <= STORED_LENGTH:
            raise ValueError("input_length must be 8..301")
        if len(config.encoder_channels) != 3 or any(c <= 0 for c in config.encoder_channels):
            raise ValueError("Exactly three positive encoder channel counts are required")
        if config.embed_channels < 1 or config.ffn_channels < 1 or config.head_channels < 1:
            raise ValueError("Embedding, FFN and head channel counts must be positive")
        if not 0 <= config.dropout < 1 or config.activation not in ("relu", "gelu"):
            raise ValueError("Invalid dropout or activation")
        channels = config.encoder_channels[-1]
        if config.heads <= 0 or channels % config.heads or config.transformer_blocks < 1:
            raise ValueError("Invalid attention heads or block count")
        self.config = config
        self.embedder = DNAEmbedder(config.embed_channels, config.activation)
        self.pool = nn.MaxPool1d(2, 2)
        previous = config.embed_channels
        encoders = []
        for outgoing in config.encoder_channels:
            encoders.append(DownresBlock(previous, outgoing, config.activation))
            previous = outgoing
        self.encoders = nn.ModuleList(encoders)
        self.position_embedding = nn.Parameter(torch.zeros(1, channels, config.input_length // 8))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.transformers = nn.ModuleList([TransformerBlock(channels, config.heads, config.ffn_channels,
                                                           config.dropout, config.activation)
                                          for _ in range(config.transformer_blocks)])
        self.readout_norm = nn.LayerNorm(2 * channels)
        self.classifier = nn.Sequential(nn.Linear(2 * channels, config.head_channels), activation(config.activation),
                                        nn.Dropout(config.dropout), nn.Linear(config.head_channels, 1))

    def forward(self, sequence):
        if sequence.ndim != 3 or sequence.shape[1:] != (4, self.config.input_length):
            raise ValueError(f"Expected [B,4,{self.config.input_length}], got {tuple(sequence.shape)}")
        x = self.embedder(sequence)
        for block in self.encoders:
            x = block(self.pool(x))
        x = x + self.position_embedding
        for block in self.transformers:
            x = block(x)
        pooled = torch.cat((x.amax(dim=2), x.mean(dim=2)), dim=1)
        return self.classifier(self.readout_norm(pooled)).squeeze(1)


class SequenceDataset(Dataset):
    def __init__(self, cache, indices, input_length=256, training=False, seed=1, rc_probability=0.5):
        self.sequences, self.labels = cache["sequences"], cache["labels"]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.input_length, self.training, self.seed = input_length, training, seed
        self.rc_probability = rc_probability
        if not 0 <= rc_probability <= 1:
            raise ValueError("RC probability must be 0..1")
        self.set_epoch(0)

    def set_epoch(self, epoch):
        rng = np.random.default_rng([self.seed, epoch, 301])
        self.crop_starts = rng.integers(0, crop_bounds(STORED_LENGTH, self.input_length), size=len(self.indices))
        self.rc_flags = rng.random(len(self.indices)) < self.rc_probability

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        index = int(self.indices[position])
        codes = self.sequences[index]
        if self.training:
            start = int(self.crop_starts[position])
            codes = codes[start:start + self.input_length]
        one_hot = ONE_HOT[codes].T.copy()
        if self.training and self.rc_flags[position]:
            one_hot = one_hot[[3, 2, 1, 0], ::-1].copy()
        return torch.from_numpy(one_hot), torch.tensor(float(self.labels[index]), dtype=torch.float32)


class EpochBatchSampler:
    def __init__(self, labels, negatives_per_positive, batch_size, seed):
        self.positives = np.flatnonzero(labels == 1)
        self.pool = CyclicNegativePool(np.flatnonzero(labels == 0), seed + 71)
        self.k, self.batch_size = negatives_per_positive, batch_size
        self.rng = np.random.default_rng(seed + 83)
        if self.k * len(self.positives) > len(self.pool.positions):
            raise ValueError("Not enough distinct training negatives for requested per-epoch ratio")
        if self.k < 1 or self.batch_size < self.k + 1 or self.batch_size % (self.k + 1):
            raise ValueError("batch_size must be a multiple of negatives_per_positive + 1")
        self.last_summary = {}

    def __len__(self):
        return math.ceil(len(self.positives) / (self.batch_size // (self.k + 1)))

    def __iter__(self):
        negatives = self.pool.draw(self.k * len(self.positives))
        batches = balanced_batches(self.positives, negatives, self.k, self.batch_size, self.rng)
        self.last_summary = {"positive": len(self.positives), "negative": len(negatives),
                             "negatives_seen": len(self.pool.seen), "negative_pool_size": len(self.pool.positions),
                             "negative_pool_coverage": len(self.pool.seen) / len(self.pool.positions)}
        return iter(batches)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, sample_count = 0.0, 0
    for sequence, label in loader:
        sequence, label = sequence.to(device), label.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(sequence)
        loss = criterion(logits, label)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
        optimizer.step()
        loss_sum += loss.item() * len(label)
        sample_count += len(label)
    return loss_sum / sample_count


@torch.inference_mode()
def predict(model, loader, device, crop_mode="multi", include_rc=True):
    model.eval()
    starts = evaluation_crop_starts(STORED_LENGTH, model.config.input_length, crop_mode)
    labels, scores = [], []
    for full_sequence, label in loader:
        full_sequence = full_sequence.to(device)
        total = torch.zeros(len(label), device=device)
        for start in starts:
            sequence = full_sequence[:, :, start:start + model.config.input_length]
            total += model(sequence).sigmoid()
            if include_rc:
                total += model(sequence[:, [3, 2, 1, 0], :].flip(-1)).sigmoid()
        total /= len(starts) * (2 if include_rc else 1)
        scores.append(total.cpu().numpy())
        labels.append(label.numpy().astype(np.int64))
    return np.concatenate(labels), np.concatenate(scores)


def dump_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save_predictions(path, cache, indices, scores, threshold):
    with Path(path).open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["sample_id", "chromosome", "start_0based", "end_exclusive", "label", "binding_score", "prediction"])
        for index, score in zip(indices, scores):
            writer.writerow([cache["sample_ids"][index], cache["chromosomes"][index],
                             int(cache["starts"][index]), int(cache["ends"][index]),
                             int(cache["labels"][index]), float(score), int(score >= threshold)])


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable")
    return torch.device(name)


def train(args):
    if args.epochs < 1 or args.patience < 1 or args.learning_rate <= 0 or args.min_delta < 0:
        raise ValueError("Invalid training hyperparameters")
    if not 0 <= args.dropout < 1 or args.num_workers < 0 or args.weight_decay < 0:
        raise ValueError("Invalid dropout/workers/weight decay")
    if args.eval_batch_size < 1 or not 0 <= args.min_learning_rate <= args.learning_rate:
        raise ValueError("Invalid evaluation batch size or minimum learning rate")
    seed_everything(args.seed)
    cache = load_cache(args.cache)
    # Report shared sequences across splits and continue with the fixed data.
    # Conflicting labels retain their existing error behavior.
    audit = cache["summary"].get("sequence_audit", {})
    overlap_issues = [key for key, value in audit.items() if isinstance(value, int) and value > 0]
    label_issues = [key for key, value in audit.items() if isinstance(value, dict) and value.get("identical_or_RC_identical_sequences_with_both_labels", 0) > 0]
    if label_issues:
        raise ValueError("Resolve sequence overlap/conflicting sequence labels before training: " + ", ".join(label_issues))
    if overlap_issues:
        print("WARNING: Shared sequences or reverse complements between dataset splits: "
              + ", ".join(f"{key}={audit[key]}" for key in overlap_issues)
              + ". Training continues with the unchanged datasets.", flush=True)
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError("Output directory must be new or empty; existing runs are never overwritten")
    device = resolve_device(args.device)
    config = ModelConfig(input_length=args.input_length, embed_channels=args.embed_channels,
                         encoder_channels=tuple(args.encoder_channels), transformer_blocks=args.transformer_blocks,
                         heads=args.heads, ffn_channels=args.ffn_channels, head_channels=args.head_channels,
                         dropout=args.dropout, activation=args.activation)
    model = TFBindingModel301(config).to(device)
    indices = {name: np.flatnonzero(cache["split"] == code) for code, name in enumerate(SPLIT_NAMES)}
    train_data = SequenceDataset(cache, indices["train"], args.input_length, training=True,
                                 seed=args.seed, rc_probability=args.rc_probability)
    sampler = EpochBatchSampler(cache["labels"][indices["train"]], args.negatives_per_positive, args.batch_size, args.seed)
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": device.type == "cuda"}
    # Workers are recreated each epoch so they receive the new crop/RC arrays.
    train_loader = DataLoader(train_data, batch_sampler=sampler, **loader_kwargs)
    val_data = SequenceDataset(cache, indices["validation"], args.input_length)
    val_loader = DataLoader(val_data, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs)
    pos_weight = float(args.negatives_per_positive)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_learning_rate)
    metadata = {"version": VERSION, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "model": asdict(config), "parameter_count": sum(p.numel() for p in model.parameters()),
                "cache_sha256": sha256_file(args.cache), "dataset": cache["summary"],
                "training_ratio": f"1:{args.negatives_per_positive}", "pos_weight": pos_weight,
                "evaluation_balance": "original, no undersampling", "selection_metric": "validation average precision",
                "threshold_selection": "validation MCC after loading the selected checkpoint",
                "evaluation_crop_starts": evaluation_crop_starts(STORED_LENGTH, args.input_length, args.eval_crops),
                "evaluation_reverse_complement": args.eval_rc, "device": str(device), "torch_version": str(torch.__version__)}
    output.mkdir(parents=True, exist_ok=True)
    dump_json(output / "configuration.json", metadata)
    print(f"Device: {device}; parameters: {metadata['parameter_count']:,}; training pos_weight: {pos_weight:g}")
    for name in SPLIT_NAMES:
        print(name, counts(cache["labels"][indices[name]]))
    best_ap, best_epoch, stale, history = -math.inf, 0, 0, []
    checkpoint_path = output / "best_model.pt"
    beginning = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        tick = time.perf_counter()
        train_data.set_epoch(epoch)
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        y_val, p_val = predict(model, val_loader, device, args.eval_crops, args.eval_rc)
        val_metrics = classification_metrics(y_val, p_val, 0.5)
        ap = val_metrics["auprc"]
        improved = ap > best_ap + args.min_delta
        if improved:
            best_ap, best_epoch, stale = ap, epoch, 0
            temp_checkpoint = checkpoint_path.with_suffix(".pt.tmp")
            torch.save({"state_dict": model.state_dict(), "model_config": asdict(config), "epoch": epoch,
                        "validation_average_precision": ap, "cache_sha256": metadata["cache_sha256"]}, temp_checkpoint)
            temp_checkpoint.replace(checkpoint_path)
        else:
            stale += 1
        history.append({"epoch": epoch, "training_loss": loss, "validation": val_metrics,
                        "sampling": sampler.last_summary, "learning_rate": optimizer.param_groups[0]["lr"],
                        "seconds": time.perf_counter() - tick, "selected_checkpoint": improved})
        dump_json(output / "history.json", history)
        print(f"Epoch {epoch:03d}: loss={loss:.4f}; val AP={ap:.5f}; negatives seen={sampler.last_summary['negative_pool_coverage']:.1%}", flush=True)
        scheduler.step()
        if stale >= args.patience:
            print(f"Early stopping; selected epoch {best_epoch}")
            break
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    y_val, p_val = predict(model, val_loader, device, args.eval_crops, args.eval_rc)
    threshold, validation_mcc = select_mcc_threshold(y_val, p_val)
    validation_metrics = classification_metrics(y_val, p_val, threshold)
    # Exploratory validation runs can skip the fixed test altogether.
    test_metrics = None
    if not args.skip_test:
        test_data = SequenceDataset(cache, indices["test"], args.input_length)
        test_loader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False, **loader_kwargs)
        y_test, p_test = predict(model, test_loader, device, args.eval_crops, args.eval_rc)
        test_metrics = classification_metrics(y_test, p_test, threshold)
        save_predictions(output / "test_predictions.tsv", cache, indices["test"], p_test, threshold)
    results = {"checkpoint_epoch": best_epoch, "epochs_run": len(history), "training_minutes": (time.perf_counter() - beginning) / 60,
               "threshold": threshold, "validation_mcc": validation_mcc, "validation": validation_metrics, "test": test_metrics,
               "test_evaluated": not args.skip_test, "sampling_final": sampler.last_summary}
    dump_json(output / "binary_classification_metrics.json", results)
    checkpoint["classification_threshold"] = threshold
    checkpoint["evaluation_crop_starts"] = metadata["evaluation_crop_starts"]
    checkpoint["evaluation_crop_mode"] = args.eval_crops
    checkpoint["evaluation_reverse_complement"] = args.eval_rc
    torch.save(checkpoint, output / "model_for_prediction.pt")
    save_predictions(output / "validation_predictions.tsv", cache, indices["validation"], p_val, threshold)
    (output / ".complete").write_text(VERSION + "\n")
    print(json.dumps(results, indent=2))


def evaluate(args):
    """Evaluate a frozen checkpoint on the full fixed test; do not retune it."""
    cache = load_cache(args.cache)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("cache_sha256") != sha256_file(args.cache):
        raise ValueError("Checkpoint/cache provenance mismatch; use the original cache")
    for field in ("classification_threshold", "evaluation_crop_mode", "evaluation_reverse_complement"):
        if field not in checkpoint:
            raise ValueError("Use model_for_prediction.pt with its frozen threshold and inference settings")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise ValueError("Evaluation output directory must be new or empty")
    if args.eval_batch_size < 1 or args.num_workers < 0:
        raise ValueError("Invalid evaluation loader parameters")
    model = TFBindingModel301(ModelConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    index = np.flatnonzero(cache["split"] == 2)
    data = SequenceDataset(cache, index, model.config.input_length)
    loader = DataLoader(data, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=device.type == "cuda")
    labels, scores = predict(model, loader, device, checkpoint["evaluation_crop_mode"], checkpoint["evaluation_reverse_complement"])
    threshold = checkpoint["classification_threshold"]
    results = {"checkpoint_epoch": checkpoint["epoch"], "threshold": threshold,
               "checkpoint_sha256": sha256_file(args.checkpoint), "cache_sha256": checkpoint["cache_sha256"],
               "evaluation_crop_mode": checkpoint["evaluation_crop_mode"],
               "evaluation_reverse_complement": checkpoint["evaluation_reverse_complement"],
               "test": classification_metrics(labels, scores, threshold)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dump_json(args.output_dir / "binary_classification_metrics.json", results)
    save_predictions(args.output_dir / "test_predictions.tsv", cache, index, scores, threshold)
    (args.output_dir / ".complete").write_text(VERSION + "\n")
    print(json.dumps(results, indent=2))


def smoke_test(args):
    """Real PyTorch forward/backward, batch sampling and inference check."""
    seed_everything(1)
    device = resolve_device(args.device)
    for length in (256, 301):
        model = TFBindingModel301(ModelConfig(input_length=length)).to(device)
        x = F.one_hot(torch.randint(0, 4, (6, length), device=device), 4).permute(0, 2, 1).float()
        target = torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.float32, device=device)
        logits = model(x)
        assert logits.shape == (6,) and torch.isfinite(logits).all()
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(5.0, device=device))
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        with torch.inference_mode():
            model.eval()
            assert torch.isfinite(model(x)).all()
        print(f"PASS PyTorch {length}-bp forward/backward: {float(loss.detach()):.6f}")
    cache = {"sequences": np.random.default_rng(1).integers(0, 4, size=(36, 301), dtype=np.uint8),
             "labels": np.array([1] * 6 + [0] * 30, dtype=np.uint8)}
    data = SequenceDataset(cache, np.arange(36), training=True)
    sampler = EpochBatchSampler(cache["labels"], 5, 12, 1)
    batches = list(sampler)
    assert len(batches) == 3 and len(set(i for b in batches for i in b)) == 36
    for b in batches:
        assert cache["labels"][b].sum() == 2
    model = TFBindingModel301(ModelConfig()).to(device)
    evaluation = SequenceDataset(cache, np.arange(36))
    y, p = predict(model, DataLoader(evaluation, batch_size=12), device)
    assert len(y) == 36 and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    print("PASS balanced batches and 8-view inference")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=VERSION)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--cache", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--input-length", type=int, choices=(256, 301), default=256)
    p.add_argument("--negatives-per-positive", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=48, help="Multiple of negatives-per-positive + 1; default = 8 positives + 40 negatives")
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--eval-crops", choices=("center", "multi"), default="multi")
    p.add_argument("--eval-rc", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-test", action="store_true", help="Compare on validation only; evaluate the chosen frozen checkpoint later")
    p.add_argument("--rc-probability", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--min-learning-rate", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--embed-channels", type=int, default=32)
    p.add_argument("--encoder-channels", type=int, nargs=3, default=(48, 64, 96))
    p.add_argument("--transformer-blocks", type=int, default=1)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--ffn-channels", type=int, default=192)
    p.add_argument("--head-channels", type=int, default=64)
    p.add_argument("--activation", choices=("relu", "gelu"), default="relu")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="auto")
    p.add_argument("--num-workers", type=int, default=0)
    smoke = sub.add_parser("smoke-test")
    smoke.add_argument("--device", default="cpu")
    evaluation = sub.add_parser("evaluate", help="Full fixed test with frozen checkpoint threshold and inference settings")
    evaluation.add_argument("--cache", required=True, type=Path)
    evaluation.add_argument("--checkpoint", required=True, type=Path)
    evaluation.add_argument("--output-dir", required=True, type=Path)
    evaluation.add_argument("--device", default="auto")
    evaluation.add_argument("--eval-batch-size", type=int, default=128)
    evaluation.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        smoke_test(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc)) from exc
