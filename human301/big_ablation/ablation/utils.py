from __future__ import annotations

# Shared implementation extracted from the frozen 4.1.0 reference.
from .common import *

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic algorithms can be substantially slower, so only the CuDNN
    # settings are fixed here. Exact bitwise reproducibility still depends on
    # hardware and PyTorch version.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def canonical_chrom(name: str) -> str:
    """Map names such as Chr1, chr1 and chromosome1 to the canonical value 1."""
    value = str(name).strip()
    lower = value.lower()
    if lower.startswith("chromosome"):
        value = value[len("chromosome") :]
    elif lower.startswith("chr"):
        value = value[3:]
    value = value.strip()
    if value.isdigit():
        return str(int(value))
    aliases = {
        "mt": "M",
        "mitochondria": "M",
        "mitochondrion": "M",
        "cp": "C",
        "chloroplast": "C",
    }
    return aliases.get(value.lower(), value.upper())


def parse_chromosome_list(value: str) -> tuple[str, ...]:
    chroms = tuple(canonical_chrom(item) for item in value.split(",") if item.strip())
    if not chroms:
        raise argparse.ArgumentTypeError("At least one chromosome is required.")
    return chroms


def stable_fraction(text: str) -> float:
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(2**64 - 1)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")

