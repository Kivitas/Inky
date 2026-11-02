"""Inky: a ~50M-parameter recurrent language model for creative writing (untrained)."""

from .config import InkyConfig
from .arlo import Arlo
from .modeling import InkyCache, InkyModel, InkyOutput
from .metrics import binary_report, choice_report, format_report, print_report, text_report
from .optim import HybridOptimizer, Muon, build_hybrid_optimizer, split_parameters
from .scratch import reduce_scratch
from .style import STOCK_PHRASES, build_bad_word_ids

__all__ = [
    "InkyConfig",
    "InkyModel",
    "InkyCache",
    "InkyOutput",
    "Arlo",
    "HybridOptimizer",
    "Muon",
    "build_hybrid_optimizer",
    "split_parameters",
    "binary_report",
    "choice_report",
    "text_report",
    "format_report",
    "print_report",
    "reduce_scratch",
    "STOCK_PHRASES",
    "build_bad_word_ids",
]
