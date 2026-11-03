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


def choice_loss(scores: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over option scores: scores [B, N] logits, gold [B] long indices -> scalar."""
    return F.cross_entropy(scores, gold.long())


def binary_loss(logit: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy with logits: logit [B], gold [B] float 0/1 -> scalar."""
    return F.binary_cross_entropy_with_logits(logit, gold.to(logit.dtype))


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
        """query [B, d] shifted by the slots' summary -> [B, d].

        The slots are not averaged. Each slot is projected to the model width and weighted by its
        match with the query (softmax over slots), so the slot that fits this query dominates. With a
        single slot the weight is 1 and the result is the same as a plain projection of that slot.
        """
        slots = self.out_proj(self.read(states))                                       # [B, K, d]
        match = torch.einsum("bd,bkd->bk", query, slots) / math.sqrt(query.shape[-1])
        weights = torch.softmax(match, dim=-1)
        steer = self.out_norm(torch.einsum("bk,bkd->bd", weights, slots))
        return query + steer

    def choice_scores(self, query: torch.Tensor, options: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """query [B, d], options [B, N, d], states [B, S, d] -> option scores [B, N] (pre-softmax).

        The query is shifted by the slot summary and then by an attention summary of the option set:
        the shifted query attends over the N options, and the attended vector moves it again. So an
        option's score depends on the other options, not only on itself. The scores are invariant to
        the order of the options (permutation-equivariant per option).
        """
        shifted = self.summarise(query, states)
        match = torch.einsum("bd,bnd->bn", shifted, options) / math.sqrt(shifted.shape[-1])
        option_context = torch.einsum("bn,bnd->bd", torch.softmax(match, dim=-1), options)
        return self.pointer(shifted + option_context, options)

    def binary_logit(self, query: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """query [B, d], states [B, S, d] -> logit [B]."""
        return self.binary_head(self.summarise(query, states)).squeeze(-1)

    def training_loss(
        self,
        query: torch.Tensor,
        options: torch.Tensor,
        states: torch.Tensor,
        gold: torch.Tensor,
        task: str,
    ) -> torch.Tensor:
        """Decision loss with gradients. task "choice": options [B, N, d], gold [B] indices.
        task "binary": gold [B] float 0/1, options is ignored. Returns a scalar tensor.
        Nothing here runs an optimiser step; the caller owns training."""
        if task == "choice":
            return choice_loss(self.choice_scores(query, options, states), gold)
        if task == "binary":
            return binary_loss(self.binary_logit(query, states), gold)
        raise ValueError(f"unknown Arlo training task {task!r}; expected 'choice' or 'binary'")


def _ids_2d(ids) -> torch.Tensor:
    return torch.as_tensor(ids, dtype=torch.long).reshape(1, -1)


def option_ids(option, index: int = 0) -> torch.Tensor:
    """One decision option as [1, L] long ids. An empty option is rejected (ARLO-03): it has no
    token states to pool, so it cannot be scored."""
    ids = _ids_2d(option)
    if ids.shape[1] == 0:
        raise ValueError(f"option {index} is empty; every decision option needs at least one token")
    return ids


def mean_pool_options(h: torch.Tensor, prefix_len: int, lengths) -> torch.Tensor:
    """Mean of each option's own token states (ARLO-03).

    ``h`` is [N, T, d]: row i holds the prefix (positions 0..prefix_len-1) followed by option i, which
    occupies positions prefix_len..lengths[i]-1. Right padding after lengths[i] is excluded from the
    mean. Returns [N, d]. Every option must have at least one token (see ``option_ids``).
    """
    N, T, _ = h.shape
    pos = torch.arange(T, device=h.device).unsqueeze(0)                           # [1, T]
    lens = torch.as_tensor(list(lengths), device=h.device).unsqueeze(1)           # [N, 1]
    mask = ((pos >= prefix_len) & (pos < lens)).unsqueeze(-1).to(h.dtype)         # [N, T, 1]
    return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)


def _example_features(model, example: dict, task: str):
    """Frozen-LM features for one labelled example: (query [1, d], states [1, P, d], options or None).

    Mirrors decide_choice and decide_binary: the prefix is context plus optional question, each
    option is appended to the prefix, and the options run in one right-padded batch. Each option is
    the mean of its own token states (``mean_pool_options``). The query is the prefix's last state.
    Callers run this under no_grad, so the language model gets no gradients.
    """
    context = _ids_2d(example["context"])
    question = example.get("question")
    prefix = context if question is None else torch.cat([context, _ids_2d(question)], dim=1)
    P = prefix.shape[1]
    if task == "binary":
        h = model(prefix, return_hidden=True).hidden
        return h[:, -1], h[:, :P], None
    opts = [option_ids(opt, i) for i, opt in enumerate(example["options"])]
    if not opts:
        raise ValueError("a choice example needs at least one option")
    seqs = [torch.cat([prefix, o], dim=1)[0] for o in opts]
    lengths = [s.shape[0] for s in seqs]
    batch = torch.zeros((len(seqs), max(lengths)), dtype=torch.long, device=prefix.device)
    for i, s in enumerate(seqs):
        batch[i, :lengths[i]] = s
    h = model(batch, return_hidden=True).hidden
    option_states = mean_pool_options(h, P, lengths).unsqueeze(0)                  # [1, N, d]
    return h[:1, P - 1], h[:1, :P], option_states


def train_arlo(
    model,
    batches,
    task: str = "choice",
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    epochs: int = 1,
) -> list:
    """Train only the Arlo decision engine on labelled decision examples.

    ``batches`` is an iterable of batches; each batch is a list of examples (dicts):
      - task "choice":  {"context": ids, "options": [ids, ...], "gold": index, "question": ids (optional)}
      - task "binary":  {"context": ids, "gold": 0 or 1, "question": ids (optional)}
    ``ids`` is a 1-D or [1, L] integer tensor or list. The loss is the same ``training_loss`` the
    decision heads use (cross-entropy over options, or BCE with logits).

    Every parameter of ``model`` outside ``model.arlo`` is frozen for the run (requires_grad off, and
    the language model runs under no_grad), so only Arlo changes. The model's train/eval mode and the
    requires_grad flags are restored afterwards, even on error. Returns the mean loss of each step.
    """
    arlo = getattr(model, "arlo", None)
    if arlo is None:
        raise RuntimeError("train_arlo needs a model built with enable_arlo=True")
    if task not in ("choice", "binary"):
        raise ValueError(f"unknown Arlo training task {task!r}; expected 'choice' or 'binary'")
    batches = [list(batch) for batch in batches]
    if not any(batches):
        raise ValueError("train_arlo needs at least one non-empty batch")

    arlo_ids = {id(p) for p in arlo.parameters()}
    frozen = [p for p in model.parameters() if id(p) not in arlo_ids and p.requires_grad]
    was_training = model.training
    optimiser = torch.optim.AdamW(arlo.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    try:
        for p in frozen:
            p.requires_grad_(False)
        model.eval()
        for _ in range(epochs):
            for batch in batches:
                if not batch:
                    continue
                terms = []
                for example in batch:
                    with torch.no_grad():
                        query, states, options = _example_features(model, example, task)
                    gold = int(example["gold"])
                    if task == "choice":
                        n = options.shape[1]
                        if not 0 <= gold < n:
                            raise ValueError(f"choice gold {gold} is outside the {n} options")
                        terms.append(arlo.training_loss(
                            query, options, states, torch.tensor([gold]), task="choice"))
                    else:
                        if gold not in (0, 1):
                            raise ValueError(f"binary gold must be 0 or 1, got {gold}")
                        terms.append(arlo.training_loss(
                            query, query.unsqueeze(1), states, torch.tensor([float(gold)]), task="binary"))
                loss = torch.stack(terms).mean()
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                losses.append(float(loss.detach()))
    finally:
        for p in frozen:
            p.requires_grad_(True)
        model.train(was_training)
    return losses
