"""Arlo: a tiny decision head for small models.

Attach it to any model that gives hidden states; it answers "which option?" and "yes or no?".
"""

from .adapters import AttrEncoder, FnEncoder, HFEncoder, as_encoder
from .config import ArloConfig
from .decider import ArloDecider
from .engine import Arlo, MultiHeadPointer, binary_loss, choice_loss
from .metrics import evaluate
from .training import train_arlo

__version__ = "0.1.0"

__all__ = [
    "Arlo", "ArloConfig", "ArloDecider", "MultiHeadPointer", "choice_loss", "binary_loss",
    "train_arlo", "evaluate", "FnEncoder", "HFEncoder", "AttrEncoder", "as_encoder",
]
