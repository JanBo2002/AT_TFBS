from __future__ import annotations

# Shared implementation extracted from the frozen 4.1.0 reference.
from .common import *
from .utils import seed_everything
from .data import CyclicBalancedNegativeSampler

def _seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_loader(
    dataset: TFBindingDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    sampler: Sampler[int] | None = None,
) -> DataLoader:
    # Die Shifts werden je Epoche im Hauptprozess gezogen (set_epoch); mit
    # persistent_workers saehen Worker jedoch veraltete Shift-Tabellen, daher
    # keine persistenten Worker.
    if isinstance(sampler, CyclicBalancedNegativeSampler):
        return DataLoader(
            dataset, batch_sampler=sampler, num_workers=num_workers,
            pin_memory=device.type == "cuda", persistent_workers=False,
            worker_init_fn=_seed_worker if num_workers > 0 else None,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        drop_last=False,
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device

