"""Arlo, the writing decision maker for Inky, ported from EOS's Rhen (model.py: RhenLatentDecisionEngine,
MultiHeadRhenPointerHead).

Rhen answers typed questions over a state without generating text. Inky ports the two
primitives that fit writing:

  - Choice: pick one of N options (candidate openings, endings, tones, revisions). Returns a
    probability for each option.
  - Noul:   a yes/no judgment about the state (for example "does this paragraph need revising?").
    Returns p_true.

Pipeline:
  1. The language model encodes the context (and the question, if any) causally.
  2. K latent decision slots cross-attend to those hidden states and refine each other through
     a gated update (Rhen channels A and D). The slots are reset for every decision.
  3. The question's final state is shifted by a summary of the slots (Rhen channel B).
  4. A multi-head bilinear pointer scores the query against each option (Rhen's pointer head).

The engine only reads the model's hidden states. It never changes the language-model path.

Not ported from Rhen: Score, MultiChoice, Range, Extract and Cascade primitives; the calibration
and conformal layers; abstention; the Hopfield trajectory memory; KV modulation; and the ACT
confidence feedback. Those either serve non-writing tasks or depend on calibration data.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import InkyConfig
from .layers import RMSNorm


class MultiHeadPointer(nn.Module):
    """Scores a query against options with several bilinear heads, mixed by learned weights.

    Each head projects both sides to a low rank and normalises them, so every head compares
    the two in its own subspace. The head weights are softmax-normalised.
    """

    def __init__(self, hidden_size: int, num_heads: int, rank: int, eps: float):
        super().__init__()
        self.rank = rank
        self.q_projs = nn.ModuleList(nn.Linear(hidden_size, rank, bias=False) for _ in range(num_heads))
        self.o_projs = nn.ModuleList(nn.Linear(hidden_size, rank, bias=False) for _ in range(num_heads))
        self.norm_q = nn.ModuleList(RMSNorm(rank, eps) for _ in range(num_heads))
        self.norm_o = nn.ModuleList(RMSNorm(rank, eps) for _ in range(num_heads))
        self.head_logits = nn.Parameter(torch.zeros(num_heads))

    def forward(self, query: torch.Tensor, options: torch.Tensor) -> torch.Tensor:
        """query [B, d], options [B, N, d] -> scores [B, N]."""
        weights = torch.softmax(self.head_logits, dim=0)
        scores = query.new_zeros(options.shape[:2])
        for h in range(len(self.q_projs)):
            q = self.norm_q[h](self.q_projs[h](query))                    # [B, r]
            o = self.norm_o[h](self.o_projs[h](options))                  # [B, N, r]
            s = torch.einsum("br,bnr->bn", q, o) / math.sqrt(self.rank)
            scores = scores + weights[h] * s
        return scores


class Arlo(nn.Module):
    """Latent decision slots that read the model's states and feed a Choice or Noul answer."""

    def __init__(self, config: InkyConfig):
        super().__init__()
        d, r = config.hidden_size, config.arlo_dim
        self.refine_steps = config.arlo_refine_steps
        self.scale = r ** -0.5

        # K learnable base slots (Rhen's hypothesis bank). Read by the optimiser as AdamW
        # parameters, see inky/optim.py ADAMW_ONLY_NAMES.
        self.slot_base = nn.Parameter(torch.randn(config.arlo_slots, r) * 0.02)

        # Channel A: slots (queries) read the hidden states (keys and values).
        self.q_proj = nn.Linear(r, r, bias=False)
        self.k_proj = nn.Linear(d, r, bias=False)
        self.v_proj = nn.Linear(d, r, bias=False)
        # Channel D: gated slot update between refinement steps.
        self.slot_gate = nn.Linear(2 * r, r)
        # Channel B: steer the query with a summary of the slots.
        self.out_proj = nn.Linear(r, d, bias=False)
        self.out_norm = RMSNorm(d, config.rms_norm_eps)

        self.pointer = MultiHeadPointer(d, config.arlo_heads, config.arlo_pointer_rank, config.rms_norm_eps)
        self.binary_head = nn.Linear(d, 1)

    def read(self, states: torch.Tensor) -> torch.Tensor:
        """states [B, S, d] -> slot summaries [B, K, r]."""
        B = states.shape[0]
        slots = self.slot_base.unsqueeze(0).expand(B, -1, -1)
        k = self.k_proj(states)
        v = self.v_proj(states)
        for _ in range(self.refine_steps):
            attn = torch.softmax(torch.matmul(self.q_proj(slots), k.transpose(-1, -2)) * self.scale, dim=-1)
            z = torch.matmul(attn, v)                                     # [B, K, r]
            alpha = torch.sigmoid(self.slot_gate(torch.cat([slots, z], dim=-1)))
            slots = alpha * slots + (1.0 - alpha) * z
        return slots

    def summarise(self, query: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """query [B, d] shifted by the slots' summary -> [B, d]."""
        steer = self.out_norm(self.out_proj(self.read(states).mean(dim=1)))
        return query + steer

    def choice_scores(self, query: torch.Tensor, options: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """query [B, d], options [B, N, d], states [B, S, d] -> option scores [B, N] (pre-softmax)."""
        return self.pointer(self.summarise(query, states), options)

    def binary_logit(self, query: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """query [B, d], states [B, S, d] -> logit [B]."""
        return self.binary_head(self.summarise(query, states)).squeeze(-1)
