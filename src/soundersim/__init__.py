"""soundersim: radar sounder simulator for Earth's ice sheets."""

__version__ = "0.1.0"

from .output import build_dataset, combine, save  # noqa: E402
from .simulate import simulate  # noqa: E402

__all__ = ["__version__", "build_dataset", "combine", "save", "simulate"]
