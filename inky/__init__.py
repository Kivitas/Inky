"""Inky: a ~50M-parameter recurrent language model for creative writing (untrained)."""

from .arlo import Arlo
from .config import (
    REASONING_MODES,
    REASONING_SCRATCH_BUDGETS,
    STYLE_MODES,
    InkyConfig,
    resolve_reasoning_loops,
    resolve_reasoning_scratch_budget,
)
from .data import doc_boundary_mask, pack_sequences_with_segments
from .diagnostics import format_diagnostics, loop_diagnostics
from .evaluation import bits_per_byte, bytes_per_token, early_exit_drift, perplexity_by_loops
from .grpo import (
    ContinualGRPOTrainer,
    UserPreferenceProfile,
    compute_group_relative_advantages,
    grpo_loss,
    sequence_log_probs,
)
from .jepa import (
    EnergyCostCritic,
    JEPAController,
    JEPAControlSignals,
    LatentStateEncoder,
    LatentWorldModelPredictor,
)
from .layers import GatedDeltaNetMixer
from .memory_bank import ProductKeyMemoryBank, QuantizedValueTable
from .metrics import (
    baseline_reports,
    binary_report,
    choice_report,
    compare_to_reference,
    format_report,
    print_report,
    text_report,
)
from .modeling import InkyCache, InkyModel, InkyOutput
from .optim import (
    HybridOptimizer,
    Muon,
    assign_parameter_groups,
    build_hybrid_optimizer,
    format_group_counts,
    split_parameters,
)
from .scratch import reduce_scratch
from .style import (
    FORMAL_PHRASES,
    FORMAL_WORDS,
    INFORMAL_PHRASES,
    INFORMAL_WORDS,
    STOCK_PHRASES,
    STYLE_PROFILES,
    NgramBanIndex,
    build_bad_word_ids,
    build_style_controls,
    get_style_profile,
    soft_penalty_token_ids,
)

__all__ = [
    "InkyConfig",
    "REASONING_MODES",
    "REASONING_SCRATCH_BUDGETS",
    "STYLE_MODES",
    "resolve_reasoning_loops",
    "resolve_reasoning_scratch_budget",
    "InkyModel",
    "InkyCache",
    "InkyOutput",
    "Arlo",
    "GatedDeltaNetMixer",
    "ProductKeyMemoryBank",
    "QuantizedValueTable",
    "JEPAController",
    "JEPAControlSignals",
    "LatentStateEncoder",
    "LatentWorldModelPredictor",
    "EnergyCostCritic",
    "ContinualGRPOTrainer",
    "UserPreferenceProfile",
    "compute_group_relative_advantages",
    "grpo_loss",
    "sequence_log_probs",
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
    "FORMAL_WORDS",
    "FORMAL_PHRASES",
    "INFORMAL_WORDS",
    "INFORMAL_PHRASES",
    "STYLE_PROFILES",
    "get_style_profile",
    "build_style_controls",
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
    "bits_per_byte",
    "bytes_per_token",
    "loop_diagnostics",
    "format_diagnostics",
]
