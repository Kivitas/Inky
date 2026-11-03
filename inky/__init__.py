"""Inky: a ~50M-parameter recurrent language model for creative writing (untrained)."""

from .config import InkyConfig
from .arlo import Arlo
from .modeling import InkyCache, InkyModel, InkyOutput
from .data import doc_boundary_mask, pack_sequences_with_segments
from .evaluation import early_exit_drift, perplexity_by_loops
from .metrics import (
    baseline_reports,
    binary_report,
    choice_report,
    compare_to_reference,
    format_report,
    print_report,
    text_report,
)
from .optim import (
    HybridOptimizer,
    Muon,
    assign_parameter_groups,
    build_hybrid_optimizer,
    format_group_counts,
    split_parameters,
)
from .scratch import reduce_scratch
from .style import STOCK_PHRASES, NgramBanIndex, build_bad_word_ids, soft_penalty_token_ids

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
    "NgramBanIndex",
    "soft_penalty_token_ids",
    "assign_parameter_groups",
    "format_group_counts",
    "pack_sequences_with_segments",
    "doc_boundary_mask",
    "baseline_reports",
    "compare_to_reference",
    "perplexity_by_loops",
    "early_exit_drift",
]
