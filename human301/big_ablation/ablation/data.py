from __future__ import annotations

# Shared implementation extracted from the frozen 4.1.0 reference.
from .common import *
from .utils import *

@dataclass(frozen=True)
class WindowRecord:
    sample_id: str
    label: int
    chromosome: str
    chromosome_raw: str
    window_start: int
    center_index: int
    window_length: int
    window_path: str
    # Datensaetze ab Generator 7.1 tragen um jedes Fenster ein Umfeld
    # (context_per_side Basen je Seite); die Datei hat context_length Zeilen.
    context_length: int = 0
    context_per_side: int = 0
    context_start: int = 0


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("rt", encoding="utf-8", newline="")


def resolve_metadata_files(dataset_root: Path) -> list[tuple[Path, int]]:
    """Read the single positive and single negative metadata table."""
    files = [(dataset_root / "metadata_positive.tsv", 1),
             (dataset_root / "metadata_negative.tsv", 0)]
    missing = [str(path) for path, _ in files if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Pooled metadata missing: " + ", ".join(missing)
            + ". Generate a new genome-wide dataset with data_m33_pooled.py."
        )
    return files


def read_metadata(dataset_root: Path) -> list[WindowRecord]:
    """Read labels and geometry, without biological pool annotations."""
    records: list[WindowRecord] = []
    metadata_files = resolve_metadata_files(dataset_root)

    required = {
        "sample_id",
        "label",
        "chromosome",
        "window_start_0based",
        "center_index_0based",
        "window_length",
        "window_data_file",
    }

    for metadata_path, expected_label in metadata_files:
        with metadata_path.open("rt", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"No header found in {metadata_path}")
            absent = required.difference(reader.fieldnames)
            if absent:
                raise ValueError(
                    f"{metadata_path} is missing columns: {', '.join(sorted(absent))}"
                )
            for line_number, row in enumerate(reader, start=2):
                try:
                    label = int(row["label"])
                    window_length = int(row["window_length"])
                    center_index = int(row["center_index_0based"])
                    window_start = int(row["window_start_0based"])
                    context_per_side = int(row.get("context_per_side") or 0)
                    context_length = int(row.get("context_length") or window_length)
                    context_start = int(row.get("context_start_0based") or window_start)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid numeric metadata in {metadata_path}:{line_number}"
                    ) from exc
                if label not in (0, 1):
                    raise ValueError(
                        f"Invalid label {label} in {metadata_path}:{line_number}; expected 0/1"
                    )
                if (
                    window_length <= 0
                    or context_per_side < 0
                    or context_length != window_length + 2 * context_per_side
                    or context_start != window_start - context_per_side
                    or not 0 <= center_index < context_length
                ):
                    raise ValueError(
                        f"Invalid window geometry in {metadata_path}:{line_number}"
                    )
                rel_path = row["window_data_file"].strip()
                if not rel_path:
                    raise ValueError(
                        f"Empty window_data_file in {metadata_path}:{line_number}"
                    )
                if label != expected_label:
                    raise ValueError(
                        f"Label {label} conflicts with metadata class in {metadata_path}:{line_number}"
                    )
                records.append(
                    WindowRecord(
                        sample_id=row["sample_id"].strip(),
                        label=label,
                        chromosome=canonical_chrom(row["chromosome"]),
                        chromosome_raw=row["chromosome"].strip(),
                        window_start=window_start,
                        center_index=center_index,
                        window_length=window_length,
                        window_path=rel_path,
                        context_length=context_length,
                        context_per_side=context_per_side,
                        context_start=context_start,
                    )
                )

    if not records:
        raise ValueError(f"No samples found below {dataset_root}")
    lengths = {record.window_length for record in records}
    if len(lengths) != 1:
        raise ValueError(f"Mixed window lengths are unsupported: {sorted(lengths)}")
    contexts = {record.context_per_side for record in records}
    if len(contexts) != 1:
        raise ValueError(f"Mixed context sizes are unsupported: {sorted(contexts)}")
    return records


BASE_TO_CODE = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
}


CODE_TO_BASE = np.array(["A", "C", "G", "T", "N"])


def read_window_file(
    path: Path,
    expected_length: int,
    fe_transform: Literal["none", "asinh"] = "none",
) -> tuple[np.ndarray, np.ndarray]:
    """Read base and FE columns from a generated window TSV or TSV.GZ."""
    sequence = np.empty(expected_length, dtype=np.uint8)
    fe = np.empty(expected_length, dtype=np.float32)
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"No header in window file {path}")
        needed = {"base", "FE"}
        absent = needed.difference(reader.fieldnames)
        if absent:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(absent))}")
        count = 0
        for count, row in enumerate(reader, start=1):
            if count > expected_length:
                raise ValueError(f"{path} has more than {expected_length} positions")
            base = row["base"].strip().upper()
            sequence[count - 1] = BASE_TO_CODE.get(base, 4)
            try:
                value = float(row["FE"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid FE value in {path} at row {count + 1}") from exc
            if not math.isfinite(value):
                value = 0.0
            fe[count - 1] = value
    if count != expected_length:
        raise ValueError(f"{path} has {count} positions; expected {expected_length}")
    if fe_transform == "asinh":
        fe = np.arcsinh(fe).astype(np.float32, copy=False)
    return sequence, fe


CACHE_FORMAT = "3-pooled-context"


def metadata_signature(dataset_root: Path) -> str:
    parts: list[str] = []
    for path, _ in resolve_metadata_files(dataset_root):
        stat = path.stat()
        parts.append(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def build_or_load_cache(
    dataset_root: Path,
    cache_path: Path,
    fe_transform: Literal["none", "asinh"],
    rebuild: bool,
) -> dict[str, np.ndarray]:
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as existing:
            human_cache = "split" in existing.files
        if human_cache:
            if rebuild:
                raise ValueError("Training reads the existing Human301 cache; --rebuild-cache is not applicable")
            return load_cache(cache_path)
    records = read_metadata(dataset_root)
    signature = metadata_signature(dataset_root)

    if cache_path.is_file() and not rebuild:
        with np.load(cache_path, allow_pickle=False) as cached:
            cached_signature = str(cached["source_signature"].item())
            cached_transform = str(cached["fe_transform"].item())
            cached_format = (
                str(cached["cache_format"].item()) if "cache_format" in cached.files else ""
            )
            if (
                cached_signature == signature
                and cached_transform == fe_transform
                and cached_format == CACHE_FORMAT
            ):
                return {key: cached[key].copy() for key in cached.files}
        print("Cache exists but is stale, uses a different FE transform or an old layout; rebuilding.")

    n_samples = len(records)
    window_length = records[0].window_length
    context_per_side = records[0].context_per_side
    context_length = records[0].context_length
    # Die Datei traegt das Umfeld; der Zuschnitt auf window_length passiert im Dataset.
    sequences = np.empty((n_samples, context_length), dtype=np.uint8)
    fe_profiles = np.empty((n_samples, context_length), dtype=np.float16)
    labels = np.empty(n_samples, dtype=np.uint8)
    chromosomes = np.empty(n_samples, dtype=f"<U{max(2, max(len(r.chromosome) for r in records))}")
    starts = np.empty(n_samples, dtype=np.int64)
    context_starts = np.empty(n_samples, dtype=np.int64)
    centers = np.empty(n_samples, dtype=np.int32)
    sample_ids = np.empty(n_samples, dtype=f"<U{max(1, max(len(r.sample_id) for r in records))}")
    print(f"Building cache from {n_samples:,} window files ...")
    for index, record in enumerate(records):
        path = dataset_root / record.window_path
        if not path.is_file():
            raise FileNotFoundError(
                f"Window file listed in metadata does not exist: {path}"
            )
        seq, fe = read_window_file(path, record.context_length, fe_transform=fe_transform)
        sequences[index] = seq
        fe_profiles[index] = fe.astype(np.float16, copy=False)
        labels[index] = record.label
        chromosomes[index] = record.chromosome
        starts[index] = record.window_start
        context_starts[index] = record.context_start
        centers[index] = record.center_index
        sample_ids[index] = record.sample_id
        if (index + 1) % 1000 == 0 or index + 1 == n_samples:
            print(f"  parsed {index + 1:,}/{n_samples:,}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sequences": sequences,
        "fe_profiles": fe_profiles,
        "labels": labels,
        "chromosomes": chromosomes,
        "starts": starts,
        "context_starts": context_starts,
        "centers": centers,
        "sample_ids": sample_ids,
        "window_length": np.asarray(window_length),
        "context_per_side": np.asarray(context_per_side),
        "source_signature": np.asarray(signature),
        "fe_transform": np.asarray(fe_transform),
        "cache_format": np.asarray(CACHE_FORMAT),
    }
    np.savez_compressed(cache_path, **payload)
    print(f"Cache written to {cache_path}")
    return payload


def make_splits(
    cache: dict[str, np.ndarray],
    train_chromosomes: Sequence[str],
    test_chromosomes: Sequence[str],
    val_fraction: float,
    val_block_size: int,
    min_split_samples_per_class: int = 2,
    balanced_evaluation: bool = True,
    max_eval_positive_reduction: float = 1.0,
    val_milp_time_limit: float = 120.0,
) -> dict[str, np.ndarray]:
    """Select intact genomic blocks while covering positive and negative classes.

    Blocks linked by overlapping windows form one indivisible component, even
    when the overlap crosses a nominal block boundary. An integer optimization
    chooses components near the desired validation fraction for each stratum.
    """
    if "split" in cache:
        if balanced_evaluation:
            raise ValueError("Human301 validation/test use every cached sample; use --no-balanced-evaluation")
        return {
            name: np.flatnonzero(cache["split"] == code).astype(np.int64)
            for code, name in enumerate(("train", "val", "test"))
        }
    chromosomes = cache["chromosomes"].astype(str)
    starts = cache["starts"].astype(np.int64)
    labels = cache["labels"].astype(np.int64)
    if np.any((labels != 0) & (labels != 1)):
        raise ValueError("All labels must be 0 or 1")
    stratum_names = ("positive", "negative")
    strata = 1 - labels
    train_set = {canonical_chrom(chrom) for chrom in train_chromosomes}
    test_set = {canonical_chrom(chrom) for chrom in test_chromosomes}
    overlap = train_set.intersection(test_set)
    if overlap:
        raise ValueError(f"Train and test chromosomes overlap: {sorted(overlap)}")

    train_candidates = np.flatnonzero(np.isin(chromosomes, sorted(train_set)))
    test_indices = np.flatnonzero(np.isin(chromosomes, sorted(test_set)))
    if train_candidates.size == 0:
        raise ValueError(f"No samples found for train chromosomes {sorted(train_set)}")
    if test_indices.size == 0:
        raise ValueError(f"No samples found for test chromosomes {sorted(test_set)}")

    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0, 1)")
    if val_block_size < 1:
        raise ValueError("val_block_size must be positive")
    if min_split_samples_per_class < 1:
        raise ValueError("min_split_samples_per_class must be positive")
    if not 0.0 <= max_eval_positive_reduction <= 1.0:
        raise ValueError("max_eval_positive_reduction must be in [0, 1]")
    if val_milp_time_limit <= 0:
        raise ValueError("val_milp_time_limit must be positive")
    if val_fraction == 0.0:
        return {
            "train": train_candidates,
            "val": np.empty(0, dtype=np.int64),
            "test": test_indices,
        }

    candidate_strata = strata[train_candidates]
    if np.any(candidate_strata < 0):
        raise ValueError("At least one training example has an unknown class")
    totals = np.bincount(
        candidate_strata[candidate_strata >= 0], minlength=len(stratum_names)
    )
    missing = [
        name for name, count in zip(stratum_names, totals) if count == 0
    ]
    if missing:
        raise RuntimeError(
            "Training chromosomes have no examples for: " + ", ".join(missing)
        )
    too_small = [
        f"{name} ({count})" for name, count in zip(stratum_names, totals)
        if count < 2 * min_split_samples_per_class
    ]
    if too_small:
        raise RuntimeError(
            f"At least {min_split_samples_per_class} examples of each class "
            "are required in both training and validation. Too few: "
            + ", ".join(too_small)
        )
    # Training draws at most one copy of each negative per epoch. If negatives
    # are fewer, the sampler cycles through all positives over later epochs.
    # Validation/test randomly keep as many positives as negatives if needed.
    if totals[1] < totals[0]:
        print(
            f"Training has {int(totals[1]):,} negatives for {int(totals[0]):,} positives. "
            "Positive examples will rotate across epochs, without duplicates inside an epoch.",
            file=sys.stderr,
        )

    # Every chromosome/block pair is atomic. Merge neighboring blocks if
    # windows from them overlap, preventing train/validation window leakage.
    candidate_chroms = chromosomes[train_candidates]
    candidate_starts = starts[train_candidates]
    context_starts = cache.get("context_starts", starts)[train_candidates].astype(np.int64)
    block_numbers = candidate_starts // val_block_size
    block_keys = sorted(set(zip(candidate_chroms.tolist(), block_numbers.tolist())))
    block_ids = {key: index for index, key in enumerate(block_keys)}
    parents = list(range(len(block_keys)))

    def find(block_id: int) -> int:
        while parents[block_id] != block_id:
            parents[block_id] = parents[parents[block_id]]
            block_id = parents[block_id]
        return block_id

    def union(first: int, second: int) -> None:
        root_first, root_second = find(first), find(second)
        if root_first != root_second:
            parents[max(root_first, root_second)] = min(root_first, root_second)

    window_length = int(cache["sequences"].shape[1])
    for chromosome in sorted(set(candidate_chroms.tolist())):
        positions = np.flatnonzero(candidate_chroms == chromosome)
        positions = positions[np.argsort(context_starts[positions], kind="stable")]
        previous_end = -1
        previous_block = -1
        for position in positions:
            start = int(context_starts[position])
            block = block_ids[(chromosome, int(block_numbers[position]))]
            if start < previous_end:
                union(block, previous_block)
            if start + window_length > previous_end:
                previous_end = start + window_length
                previous_block = block

    roots = sorted({find(index) for index in range(len(block_keys))})
    root_columns = {root: index for index, root in enumerate(roots)}
    component_of_sample = np.asarray([
        root_columns[find(block_ids[(str(chromosome), int(block))])]
        for chromosome, block in zip(candidate_chroms, block_numbers)
    ], dtype=np.int64)
    counts = np.zeros((len(stratum_names), len(roots)), dtype=np.float64)
    for stratum, component in zip(candidate_strata, component_of_sample):
        counts[int(stratum), int(component)] += 1.0

    support = np.count_nonzero(counts, axis=1)
    indivisible = [
        f"{name} ({int(support[i])} component)"
        for i, name in enumerate(stratum_names)
        if support[i] < 2
    ]
    if indivisible:
        raise RuntimeError(
            "No leak-free training/validation split: all examples of "
            + ", ".join(indivisible)
            + f" are joined at --val-block-size {val_block_size}. "
            "Smaller blocks may help only when they separate those examples; "
            "overlapping input contexts still remain together."
        )

    n_components = len(roots)
    n_strata = len(stratum_names)
    n_variables = n_components + 2 * n_strata
    target_counts = val_fraction * totals
    # Minimize the sum of absolute deviations of validation fractions from the
    # requested fraction. The last four continuous variables are the deviations.
    objective = np.zeros(n_variables, dtype=np.float64)
    objective[n_components:] = np.tile(1.0 / totals, 2)
    equalities = np.zeros((n_strata, n_variables), dtype=np.float64)
    equalities[:, :n_components] = counts
    for stratum in range(n_strata):
        equalities[stratum, n_components + stratum] = -1.0
        equalities[stratum, n_components + n_strata + stratum] = 1.0

    constraints = [
        LinearConstraint(sparse.csr_matrix(equalities), target_counts, target_counts),
        LinearConstraint(
            sparse.csr_matrix(np.pad(counts, ((0, 0), (0, 2 * n_strata)))),
            np.full(n_strata, min_split_samples_per_class),
            totals - min_split_samples_per_class,
        ),
    ]
    # Training balances each epoch by cycling positives if negatives are scarce.
    # Validation uses the same optional reduction limit as the evaluator.
    pad = ((0, 0), (0, 2 * n_strata))
    if balanced_evaluation and max_eval_positive_reduction < 1.0:
        val_balance = counts[1:] - (1.0 - max_eval_positive_reduction) * counts[:1]
        constraints.append(LinearConstraint(
            sparse.csr_matrix(np.pad(val_balance, pad)),
            np.zeros(1), np.full(1, np.inf),
        ))
    upper_bounds = np.concatenate((np.ones(n_components), np.full(2 * n_strata, np.inf)))
    integrality = np.concatenate((np.ones(n_components), np.zeros(2 * n_strata)))
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(np.zeros(n_variables), upper_bounds),
        constraints=constraints,
        options={"time_limit": val_milp_time_limit, "mip_rel_gap": 0.01},
    )
    if result.x is None:
        reason = "time limit reached" if result.status == 1 else (
            "constraints are infeasible" if result.status == 2 else "solver failed"
        )
        raise RuntimeError(
            f"Genomic-block validation split failed ({reason}). "
            f"{n_components} overlap-connected components; training chromosome "
            f"totals: {dict(zip(stratum_names, totals.astype(int).tolist()))}; "
            f"component support: {dict(zip(stratum_names, support.tolist()))}. "
            f"Solver: {result.message}. "
            "If infeasible, check class counts and block sizes; if timed out, "
            "increase --val-milp-time-limit. No individual windows were reassigned."
        )
    selected_components = np.rint(result.x[:n_components]).astype(bool)
    validation_counts = counts @ selected_components.astype(np.float64)
    if (
        np.max(np.abs(result.x[:n_components] - selected_components)) > 1e-5
        or np.any(validation_counts < min_split_samples_per_class)
        or np.any(totals - validation_counts < min_split_samples_per_class)
        or (balanced_evaluation and max_eval_positive_reduction < 1.0 and np.any(
            (counts[1:] - (1.0 - max_eval_positive_reduction) * counts[:1])
            @ selected_components < -1e-5
        ))
    ):
        raise RuntimeError("The genomic-block optimizer returned an invalid split")

    val_mask = selected_components[component_of_sample]
    train_indices = train_candidates[~val_mask]
    val_indices = train_candidates[val_mask]
    print(
        f"Validation split: {int(selected_components.sum())}/{n_components} "
        f"non-overlapping block components; "
        f"per-class counts: {dict(zip(stratum_names, validation_counts.astype(int).tolist()))}"
    )
    return {"train": train_indices, "val": val_indices, "test": test_indices}


ONE_HOT_LOOKUP = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],  # A
        [0.0, 1.0, 0.0, 0.0],  # C
        [0.0, 0.0, 1.0, 0.0],  # G
        [0.0, 0.0, 0.0, 1.0],  # T
        [0.0, 0.0, 0.0, 0.0],  # N / unknown
    ],
    dtype=torch.float32,
)


ShiftDistribution = Literal["uniform", "gaussian"]


def draw_shifts(
    rng: np.random.Generator, count: int, shift_max: int, distribution: ShiftDistribution,
) -> np.ndarray:
    """Ganzzahlige Verschiebungen in [-shift_max, shift_max]."""
    if shift_max <= 0 or count <= 0:
        return np.zeros(count, dtype=np.int64)
    if distribution == "uniform":
        return rng.integers(-shift_max, shift_max + 1, size=count, dtype=np.int64)
    if distribution == "gaussian":
        values = np.rint(rng.normal(0.0, shift_max / 2.0, size=count)).astype(np.int64)
        return np.clip(values, -shift_max, shift_max)
    raise ValueError(f"Unknown shift distribution: {distribution}")


def cache_geometry(cache: dict[str, np.ndarray]) -> tuple[int, int]:
    """(Zuschnittlaenge, Umfeld je Seite) eines geladenen Caches."""
    context_length = int(cache["sequences"].shape[1])
    if "split" in cache:
        if context_length != STORED_LENGTH:
            raise ValueError("Human301 caches must store 301-bp sequences")
        # The odd 45-bp excess is asymmetric: 22 bases left, 23 right.
        return 256, 22
    if "window_length" in cache:
        window_length = int(np.asarray(cache["window_length"]).item())
        context_per_side = int(np.asarray(cache["context_per_side"]).item())
    else:
        window_length, context_per_side = context_length, 0
    if window_length + 2 * context_per_side != context_length:
        raise ValueError("Cache geometry is inconsistent with the stored sequences")
    return window_length, context_per_side


class TFBindingDataset(Dataset[dict[str, Any]]):
    """MAA dataset with fresh Human301 crops and intact evaluation sequences.

    Human training crops start uniformly at 0..45, independently per access.
    Human evaluation returns all 301 bases; run_epoch applies the fixed views.
    The classification readout stays at the crop midpoint, also after shifting.
    MAA's FE-target handling remains available for caches supplying FE profiles.
    """

    def __init__(
        self,
        cache: dict[str, np.ndarray],
        indices: np.ndarray,
        reverse_complement_probability: float = 0.0,
        shift_max: int = 0,
        shift_distribution: ShiftDistribution = "uniform",
        fixed_shifts: np.ndarray | None = None,
        seed: int = 0,
        training: bool = False,
    ) -> None:
        self.sequences = cache["sequences"]
        self.fe_profiles = cache.get("fe_profiles")
        self.labels = cache["labels"]
        self.chromosomes = cache["chromosomes"]
        self.starts = cache["starts"]
        self.context_starts = cache.get("context_starts", cache["starts"])
        self.centers = cache.get("centers")
        self.sample_ids = cache["sample_ids"]
        self.indices = np.asarray(indices, dtype=np.int64)
        self.human301 = "split" in cache
        self.training = bool(training)
        self.reverse_complement_probability = float(reverse_complement_probability)
        if not 0.0 <= self.reverse_complement_probability <= 1.0:
            raise ValueError("reverse_complement_probability must be in [0, 1]")
        self.window_length, self.context_per_side = cache_geometry(cache)
        self.shift_max = int(shift_max)
        self.shift_distribution = shift_distribution
        if self.shift_max < 0:
            raise ValueError("shift_max must not be negative")
        if not self.human301 and self.shift_max > self.context_per_side:
            raise ValueError("shift_max exceeds the context of this dataset")
        if fixed_shifts is not None:
            fixed_shifts = np.asarray(fixed_shifts, dtype=np.int64)
            if fixed_shifts.shape != (self.indices.size,):
                raise ValueError("fixed_shifts must hold one shift per dataset position")
            if not self.human301 and np.any(np.abs(fixed_shifts) > self.context_per_side):
                raise ValueError("fixed_shifts exceed the dataset context")
        self.fixed_shifts = fixed_shifts
        self.seed = int(seed)
        self.epoch = 0
        self.epoch_shifts = np.zeros(self.indices.size, dtype=np.int64)
        self.rng = None
        self.rng_worker = None
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)
        self.rng = None
        self.rng_worker = None
        if self.human301:
            return
        if self.fixed_shifts is not None:
            self.epoch_shifts = self.fixed_shifts
            return
        rng = np.random.default_rng([self.seed, self.epoch, 7])
        self.epoch_shifts = draw_shifts(
            rng, self.indices.size, self.shift_max, self.shift_distribution
        )

    def _augmentation_rng(self) -> np.random.Generator:
        worker = get_worker_info()
        worker_id = 0 if worker is None else worker.id + 1
        if self.rng is None or self.rng_worker != worker_id:
            self.rng = np.random.default_rng([self.seed, self.epoch, worker_id, 301])
            self.rng_worker = worker_id
        return self.rng

    def shift_summary(self) -> dict[str, Any]:
        if self.human301:
            return {
                "crop_start_min": 0, "crop_start_max": 45,
                "shift_min": -22, "shift_max": 23,
                "distribution": "uniform", "redrawn_per_access": True,
            }
        shifts = self.epoch_shifts[:self.indices.size]
        return {
            "shift_max": self.shift_max,
            "distribution": self.shift_distribution if self.fixed_shifts is None else "fixed",
            "mean_abs_shift": float(np.mean(np.abs(shifts))) if shifts.size else 0.0,
            "max_abs_shift": int(np.max(np.abs(shifts))) if shifts.size else 0,
        }

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        if self.human301:
            rng = self._augmentation_rng() if self.training else None
            crop_start = int(rng.integers(0, 46)) if self.training else 0
            crop_end = crop_start + self.window_length if self.training else STORED_LENGTH
            shift = crop_start - self.context_per_side if self.training else 0
            reverse = self.training and rng.random() < self.reverse_complement_probability
        else:
            shift = int(self.epoch_shifts[item])
            crop_start = self.context_per_side + shift
            crop_end = crop_start + self.window_length
            reverse = random.random() < self.reverse_complement_probability

        codes = torch.from_numpy(
            self.sequences[index, crop_start:crop_end].astype(np.int64, copy=False)
        )
        fe = None if self.fe_profiles is None else torch.from_numpy(
            self.fe_profiles[index, crop_start:crop_end].astype(np.float32, copy=False)
        )
        center = codes.numel() // 2
        if reverse:
            complement = torch.tensor([3, 2, 1, 0, 4], dtype=torch.long)
            codes = complement[codes].flip(0)
            if fe is not None:
                fe = fe.flip(0)
            center = codes.numel() - 1 - center

        result = {
            "sequence": ONE_HOT_LOOKUP[codes].transpose(0, 1).contiguous(),
            "valid_mask": codes.ne(4).to(torch.float32),
            "label": torch.tensor(float(self.labels[index]), dtype=torch.float32),
            "center_index": torch.tensor(center, dtype=torch.long),
            "sample_id": str(self.sample_ids[index]),
            "chromosome": str(self.chromosomes[index]),
            "window_start": int(self.context_starts[index]) + crop_start,
            "shift": shift,
        }
        if fe is not None:
            result["fe_profile"] = fe
        return result


def class_counts(cache: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, int]:
    labels = cache["labels"][indices].astype(np.int64)
    return {"positive": int(np.count_nonzero(labels == 1)),
            "negative": int(np.count_nonzero(labels == 0))}


def make_fixed_balanced_evaluation_split(
    cache: dict[str, np.ndarray], indices: np.ndarray, seed: int, split_name: str,
    max_positive_reduction: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select one reproducible global 1:1 evaluation subset, without annotations."""
    if not 0 <= max_positive_reduction <= 1:
        raise ValueError("max_positive_reduction must be in [0, 1]")
    indices = np.asarray(indices, dtype=np.int64)
    labels = cache["labels"][indices].astype(np.int64)
    positives = indices[labels == 1]
    negatives = indices[labels == 0]
    if not positives.size or not negatives.size:
        raise ValueError(f"{split_name} must contain both positive and negative examples")
    pairs = int(min(positives.size, negatives.size))
    dropped = int(positives.size - pairs)
    reduction = dropped / positives.size
    if reduction > max_positive_reduction:
        raise ValueError(
            f"Cannot balance {split_name}: {negatives.size:,} negatives for {positives.size:,} positives "
            f"({reduction:.1%} short, allowed {max_positive_reduction:.1%}). "
            "Raise --max-eval-positive-reduction or provide more negatives."
        )
    if dropped:
        print(
            f"WARNING: {split_name}: only {negatives.size:,} negatives for {positives.size:,} positives; "
            f"{dropped:,} positives were dropped at random to keep the global 1:1 ratio.",
            file=sys.stderr,
        )
    rng = np.random.default_rng(seed)
    selected = np.concatenate((rng.choice(positives, pairs, replace=False),
                               rng.choice(negatives, pairs, replace=False)))
    summary = {"scope": "global", "max_positive_reduction": max_positive_reduction,
               "positives_available": int(positives.size), "negatives_available": int(negatives.size),
               "pairs_used": pairs, "positives_dropped_total": dropped}
    return rng.permutation(selected), summary


class CyclicBalancedNegativeSampler(Sampler[list[int]]):
    """Use all positives, rotate distinct negatives, and emit balanced batches.

    Optional hard examples are selected exclusively from scored training
    negatives; remaining negatives continue through the cyclic random pool.
    """
    def __init__(self, dataset, seed, negatives_per_positive=10, batch_size=88, hard_negative_fraction=0.0):
        self.dataset = dataset
        labels = dataset.labels[dataset.indices].astype(np.int64)
        self.epoch = 0
        self.min_epochs_for_positive_coverage = 1
        self.positives = np.flatnonzero(labels == 1)
        self.samples_per_epoch = (negatives_per_positive + 1) * len(self.positives)
        self.pool = CyclicNegativePool(np.flatnonzero(labels == 0), seed + 71)
        self.k, self.batch_size = negatives_per_positive, batch_size
        self.rng = np.random.default_rng(seed + 83)
        if self.k * len(self.positives) > len(self.pool.positions):
            raise ValueError("Not enough distinct training negatives for requested per-epoch ratio")
        if self.k < 1 or self.batch_size < self.k + 1 or self.batch_size % (self.k + 1):
            raise ValueError("batch_size must be a multiple of negatives_per_positive + 1")
        if not 0 <= hard_negative_fraction < 1:
            raise ValueError("hard_negative_fraction must be at least zero and below one")
        self.hard_count = int(self.k * len(self.positives) * hard_negative_fraction)
        if hard_negative_fraction > 0 and self.hard_count == 0:
            raise ValueError("Hard-negative fraction is too small for this training set")
        self.hard_candidates = np.empty(0, dtype=np.int64)
        self.last_summary = {}

    def update_hard_negatives(self, scores):
        """Scores must follow pool.positions, which contains training negatives only."""
        scores = np.asarray(scores)
        if scores.shape != self.pool.positions.shape or not np.isfinite(scores).all():
            raise ValueError("One finite score per training negative is required")
        # Randomly draw the hard part from the highest-scoring candidates.
        bank_size = min(len(scores), 2 * self.hard_count)
        order = np.argsort(-scores, kind="stable")[:bank_size]
        self.hard_candidates = self.pool.positions[order]

    def _draw_random_without(self, count, excluded):
        """Continue the cyclic pool while keeping this epoch's negatives unique."""
        drawn, used = [], set(excluded)
        while len(drawn) < count:
            if self.pool.cursor == len(self.pool.order):
                self.pool.order = self.pool.rng.permutation(self.pool.positions)
                self.pool.cursor = 0
            value = int(self.pool.order[self.pool.cursor])
            self.pool.cursor += 1
            if value not in used:
                drawn.append(value)
                used.add(value)
        self.pool.seen.update(used)
        return np.asarray(drawn, dtype=np.int64)

    def __len__(self):
        return math.ceil(len(self.positives) / (self.batch_size // (self.k + 1)))

    def __iter__(self):
        # Keep this a generator: DataLoader can create an unused iterator when
        # starting workers. Pool/RNG state advances only when batches are read.
        self.epoch += 1
        total = self.k * len(self.positives)
        hard = np.empty(0, dtype=np.int64)
        if len(self.hard_candidates):
            hard = self.rng.choice(self.hard_candidates, size=self.hard_count, replace=False)
        if len(hard):
            random_negatives = self._draw_random_without(total - len(hard), hard)
            negatives = self.rng.permutation(np.concatenate((hard, random_negatives)))
        else:
            negatives = self.pool.draw(total)
        batches = balanced_batches(self.positives, negatives, self.k, self.batch_size, self.rng)
        self.last_summary = {"epoch": self.epoch, "scope": "global", "positive": len(self.positives),
                             "positive_pool_size": len(self.positives), "positive_coverage": 1.0,
                             "negative": len(negatives), "negative_total": len(negatives),
                             "hard_negative": len(hard), "random_negative": total - len(hard),
                             "negatives_seen": len(self.pool.seen), "negative_pool_size": len(self.pool.positions),
                             "negative_pool_coverage": len(self.pool.seen) / len(self.pool.positions)}
        yield from batches

