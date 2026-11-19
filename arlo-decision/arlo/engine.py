"""The Arlo engine: latent decision slots that read hidden states and answer a Choice or a Yes/No.

This module is pure tensors in, tensors out. It never touches a language model, so it works on hidden
states from any source.

  1. K latent decision slots cross-attend to the hidden states and refine each other through a gated
     update. The slots are reset for every decision.
  2. The query (the last state of the question) is shifted by a summary of the slots.
  3. Choice: a multi-head bilinear pointer scores the query against each option.
     Yes/No:  a linear head on the shifted query gives one logit.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ArloConfig
from .layers import RMSNorm


class MultiHeadPointer(nn.Module):
    """Scores a query against options with several bilinear heads, mixed by softmax-normalised weights."""

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
            q = self.norm_q[h](self.q_projs[h](query))
            o = self.norm_o[h](self.o_projs[h](options))
            scores = scores + weights[h] * torch.einsum("br,bnr->bn", q, o) / math.sqrt(self.rank)
        return scores


def choice_loss(scores: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over option scores: scores [B, N] logits, gold [B] indices -> scalar."""
    return F.cross_entropy(scores, gold.long())


def binary_loss(logit: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with logits: logit [B], gold [B] 0/1 -> scalar."""
    return F.binary_cross_entropy_with_logits(logit, gold.to(logit.dtype))


class Arlo(nn.Module):
    """Latent decision slots over hidden states. Feeds a Choice or a Yes/No answer."""

    def __init__(self, config: ArloConfig):
        super().__init__()
        self.config = config
        d, r = config.hidden_size, config.dim
        self.refine_steps = config.refine_steps
        self.scale = r ** -0.5

        self.slot_base = nn.Parameter(torch.randn(config.slots, r) * 0.02)
        self.q_proj = nn.Linear(r, r, bias=False)
        self.k_proj = nn.Linear(d, r, bias=False)
        self.v_proj = nn.Linear(d, r, bias=False)
        self.slot_gate = nn.Linear(2 * r, r)
        self.out_proj = nn.Linear(r, d, bias=False)
        self.out_norm = RMSNorm(d, config.norm_eps)
        self.pointer = MultiHeadPointer(d, config.heads, config.pointer_rank, config.norm_eps)
        self.binary_head = nn.Linear(d, 1)

    def read(self, states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """states [B, S, d] (+ optional bool mask [B, S], True = keep) -> slot summaries [B, K, r]."""
        B = states.shape[0]
        slots = self.slot_base.unsqueeze(0).expand(B, -1, -1)
        k = self.k_proj(states)
        v = self.v_proj(states)
        for _ in range(self.refine_steps):
            logits = torch.matmul(self.q_proj(slots), k.transpose(-1, -2)) * self.scale
            if mask is not None:
                logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))
            z = torch.matmul(torch.softmax(logits, dim=-1), v)
            alpha = torch.sigmoid(self.slot_gate(torch.cat([slots, z], dim=-1)))
            slots = alpha * slots + (1.0 - alpha) * z
        return slots

    def summarise(self, query: torch.Tensor, states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """query [B, d] shifted by the slots that match it best -> [B, d]."""
        slots = self.out_proj(self.read(states, mask))
        match = torch.einsum("bd,bkd->bk", query, slots) / math.sqrt(query.shape[-1])
        steer = self.out_norm(torch.einsum("bk,bkd->bd", torch.softmax(match, dim=-1), slots))
        return query + steer

    def choice_scores(
        self, query: torch.Tensor, options: torch.Tensor, states: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """query [B, d], options [B, N, d], states [B, S, d] -> option logits [B, N].

        The shifted query attends over the options and moves again, so an option's score depends on the
        other options. Scores are invariant to the order of the options.
        """
        shifted = self.summarise(query, states, mask)
        match = torch.einsum("bd,bnd->bn", shifted, options) / math.sqrt(shifted.shape[-1])
        option_context = torch.einsum("bn,bnd->bd", torch.softmax(match, dim=-1), options)
        return self.pointer(shifted + option_context, options)

    def binary_logit(self, query: torch.Tensor, states: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """query [B, d], states [B, S, d] -> yes/no logit [B]."""
        return self.binary_head(self.summarise(query, states, mask)).squeeze(-1)

    def training_loss(
        self,
        query: torch.Tensor,
        options: torch.Tensor | None,
        states: torch.Tensor,
        gold: torch.Tensor,
        task: str,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """task "choice": options [B, N, d], gold [B] indices. task "binary": gold [B] 0/1, options ignored."""
        if task == "choice":
            return choice_loss(self.choice_scores(query, options, states, mask), gold)
        if task == "binary":
            return binary_loss(self.binary_logit(query, states, mask), gold)
        raise ValueError(f"unknown task {task!r}; expected 'choice' or 'binary'")
