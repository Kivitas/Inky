"""Inky language model: pre layers, a looped recurrent core with PonderNet halting, post layers."""

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel

from .config import InkyConfig
from .arlo import Arlo
from .metrics import binary_report, choice_report, print_report, text_report
from .layers import (
    InkyBlock,
    KVEntry,
    PonderHalting,
    RecurrentCore,
    RMSNorm,
    RotaryEmbedding,
    geometric_halting_weights,
    geometric_prior,
    pondernet_kl,
)
from .scratch import reduce_scratch
from .style import banned_by_bad_words, banned_by_no_repeat_ngram


class InkyCache:
    """Per-sequence decoding state.

    One latent-KV entry per pre layer, per post layer, and per (loop, recurrent layer), plus the
    SSM highway's running prefix sum for each loop. The loop count is fixed at the first call.
    """

    def __init__(self, config: InkyConfig):
        self.pre: List[Optional[KVEntry]] = [None] * config.num_pre_layers
        self.post: List[Optional[KVEntry]] = [None] * config.num_post_layers
        self.num_loops: Optional[int] = None
        self.core: List[Optional[list]] = []
        self.prefix: List[Optional[torch.Tensor]] = []
        self.seq_len = 0


@dataclass
class InkyOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    past: Optional[InkyCache] = None
    moe_aux_loss: Optional[torch.Tensor] = None
    mean_loops: Optional[torch.Tensor] = None  # expected loop count per token (ACT)
    hidden: Optional[torch.Tensor] = None      # final hidden states, set when return_hidden=True
    loop_applications: Optional[int] = None    # token-loop pairs actually computed (early exit only)
    full_loop_applications: Optional[int] = None  # tokens x loops, for the same input


class InkyModel(PreTrainedModel):
    config_class = InkyConfig
    base_model_prefix = "inky"

    def __init__(self, config: InkyConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = RotaryEmbedding(config.qk_rope_head_dim, config.max_position_embeddings, config.rope_theta)
        self.pre_layers = nn.ModuleList(InkyBlock(config) for _ in range(config.num_pre_layers))
        self.core = RecurrentCore(config)
        self.post_layers = nn.ModuleList(InkyBlock(config) for _ in range(config.num_post_layers))
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.halting = PonderHalting(config.hidden_size, config.ponder_prior_steps)
        self.arlo = Arlo(config) if config.enable_arlo else None
        self.post_init()
        # Start the halting gate at its prior. Set after post_init, which would reset the bias.
        self.halting.reset_prior()

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(0.0, std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(0.0, std)

    # ------------------------------------------------------------------ helpers

    def count_parameters(self) -> Dict[str, int]:
        """Parameter counts by section. The embedding is tied to the output head, so it is counted once."""
        def n(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        parts = {
            "embedding (tied with LM head)": n(self.embed_tokens),
            "pre_layers": n(self.pre_layers),
            "recurrent_core": n(self.core),
            "post_layers": n(self.post_layers),
            "norm + halting": n(self.norm) + n(self.halting),
            "decision engine": n(self.arlo) if self.arlo is not None else 0,
        }
        parts["total"] = n(self)
        return parts

    def _choose_loops(self, cache: Optional[InkyCache], num_loops: Optional[int]) -> int:
        if cache is not None and cache.num_loops is not None:
            if num_loops is not None and num_loops != cache.num_loops:
                raise ValueError(
                    f"num_loops={num_loops} differs from the {cache.num_loops} loops this cache was filled with"
                )
            return cache.num_loops
        if num_loops is not None:
            return int(num_loops)
        if self.training:
            # Sample a depth per batch so the shared core learns to work at any number of loops.
            return random.randint(self.config.min_recurrent_loops, self.config.max_recurrent_loops)
        return self.config.max_recurrent_loops

    def _project(self, h: torch.Tensor) -> torch.Tensor:
        logits = F.linear(h, self.embed_tokens.weight).float()
        cap = self.config.final_logit_softcapping
        if cap > 0.0:
            logits = cap * torch.tanh(logits / cap)
        return logits

    def _chunk_ce(self, h: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self._project(h)
        valid = (y != -100).float()
        ce = F.cross_entropy(logits, y, ignore_index=-100, reduction="sum")
        z = (torch.logsumexp(logits, dim=-1).pow(2) * valid).sum()
        return ce, z

    def _lm_loss(self, hidden: torch.Tensor, labels: torch.Tensor, chunk_size: int = 1024):
        """Chunked next-token cross-entropy. Full-vocabulary logits are never kept for the whole sequence."""
        h = hidden[:, :-1].reshape(-1, hidden.shape[-1])
        y = labels[:, 1:].reshape(-1)
        ce_total = hidden.new_zeros((), dtype=torch.float32)
        z_total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, h.shape[0], chunk_size):
            hc, yc = h[start:start + chunk_size], y[start:start + chunk_size]
            if self.training and torch.is_grad_enabled():
                ce, z = checkpoint(self._chunk_ce, hc, yc, use_reentrant=False)
            else:
                ce, z = self._chunk_ce(hc, yc)
            ce_total = ce_total + ce
            z_total = z_total + z
        n_valid = (y != -100).sum().clamp(min=1).float()
        return ce_total / n_valid, z_total / n_valid

    def _chunk_weighted_ce(self, h: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        logits = self._project(h)
        ce = F.cross_entropy(logits, y, ignore_index=-100, reduction="none")
        return (ce * w).sum()

    def _weighted_step_ce(
        self,
        loop_states: Sequence[torch.Tensor],
        weights: torch.Tensor,
        labels: torch.Tensor,
        chunk_size: int = 1024,
    ) -> torch.Tensor:
        """Next-token cross-entropy of every loop state, weighted by its halting probability.

        This trains each loop's state to predict on its own, so a shorter loop budget gives a
        usable output and not only a partly trained one. Divided by the number of valid tokens.
        """
        y = labels[:, 1:].reshape(-1)
        n_valid = (y != -100).sum().clamp(min=1).float()
        total = weights.new_zeros((), dtype=torch.float32)
        D = self.config.hidden_size
        for n, state in enumerate(loop_states):
            hn = self.norm(state)[:, :-1].reshape(-1, D)
            wn = weights[n][:, :-1].reshape(-1).float()
            for start in range(0, hn.shape[0], chunk_size):
                hc = hn[start:start + chunk_size]
                yc = y[start:start + chunk_size]
                wc = wn[start:start + chunk_size]
                if self.training and torch.is_grad_enabled():
                    total = total + checkpoint(self._chunk_weighted_ce, hc, yc, wc, use_reentrant=False)
                else:
                    total = total + self._chunk_weighted_ce(hc, yc, wc)
        return total / n_valid

    # ------------------------------------------------------------------ forward

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past: Optional[InkyCache] = None,
        use_cache: bool = False,
        num_loops: Optional[int] = None,
        last_only: bool = False,
        return_hidden: bool = False,
    ) -> InkyOutput:
        """Causal forward pass.

        ``labels`` (same shape as ``input_ids``; -100 to ignore) returns the training loss and no
        logits. Without labels, logits are returned; ``last_only=True`` keeps only the final position.
        Pass ``past=InkyCache(...)`` (or ``use_cache=True``) to decode incrementally. The loop count
        is fixed by the first call of a cached sequence.
        """
        B, S = input_ids.shape
        cache = past
        if cache is None and use_cache:
            cache = InkyCache(self.config)
        past_len = cache.seq_len if cache is not None else 0
        if past_len + S > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {past_len + S} exceeds max_position_embeddings "
                f"({self.config.max_position_embeddings})"
            )

        cos, sin = self.rope(past_len, S)
        h = self.embed_tokens(input_ids)
        moe_aux = h.new_zeros((), dtype=torch.float32)

        for i, layer in enumerate(self.pre_layers):
            h, present, aux = layer(h, cos, sin, cache.pre[i] if cache is not None else None)
            if cache is not None:
                cache.pre[i] = present
            moe_aux = moe_aux + aux

        loops = self._choose_loops(cache, num_loops)
        if cache is not None and cache.num_loops is None:
            cache.num_loops = loops
            cache.core = [None] * loops
            cache.prefix = [None] * loops

        # PonderNet halting over the recurrent loops. Each loop emits lambda_n(t); the halting
        # distribution p_n(t) puts stop mass on each loop, and the output is the expectation of the
        # loop states under p. Every loop still runs, so cached decoding matches the full forward
        # pass exactly. This version saves no compute (see README, Known issues).
        h_loop, prev_mem = h, None
        lambdas: List[torch.Tensor] = []
        loop_states: List[torch.Tensor] = []

        for step in range(1, loops + 1):
            idx = step - 1
            h_next, mem, new_prefix, presents, aux = self.core(
                h_loop,
                step,
                cos,
                sin,
                past=cache.core[idx] if cache is not None else None,
                prefix_sum=cache.prefix[idx] if cache is not None else None,
                prev_mem=prev_mem,
                past_len=past_len,
            )
            if cache is not None:
                cache.core[idx] = presents
                cache.prefix[idx] = new_prefix
            prev_mem = mem
            moe_aux = moe_aux + aux

            lambdas.append(self.halting(h_next))
            loop_states.append(h_next)
            h_loop = h_next

        weights = geometric_halting_weights(lambdas)                 # [L, B, S], sums to 1 over L
        states = torch.stack(loop_states, dim=0).float()             # [L, B, S, D]
        h_out = (weights.unsqueeze(-1) * states).sum(dim=0)          # [B, S, D]
        loop_index = torch.arange(1, loops + 1, device=h.device, dtype=torch.float32).view(-1, 1, 1)
        mean_loops = (weights * loop_index).sum(dim=0).mean()        # expected loops per token

        h = h_out.to(h.dtype)
        for i, layer in enumerate(self.post_layers):
            h, present, aux = layer(h, cos, sin, cache.post[i] if cache is not None else None)
            if cache is not None:
                cache.post[i] = present
            moe_aux = moe_aux + aux

        if cache is not None:
            cache.seq_len += S

        h = self.norm(h)

        if labels is not None:
            ce, z = self._lm_loss(h, labels)
            step_ce = self._weighted_step_ce(loop_states, weights, labels)
            prior = geometric_prior(loops, self.halting.prior_lambda, device=h.device)
            kl = pondernet_kl(weights, prior)
            loss = (
                ce
                + self.config.z_loss_coef * z
                + moe_aux
                + self.config.ponder_step_loss_coef * step_ce
                + self.config.ponder_kl_coef * kl
            )
            return InkyOutput(loss=loss, past=cache, moe_aux_loss=moe_aux, mean_loops=mean_loops)

        if return_hidden:
            return InkyOutput(hidden=h, past=cache, moe_aux_loss=moe_aux, mean_loops=mean_loops)
        logits = self._project(h[:, -1:] if last_only else h)
        return InkyOutput(logits=logits, past=cache, moe_aux_loss=moe_aux, mean_loops=mean_loops)

    @torch.no_grad()
    def forward_early_exit(
        self,
        input_ids: torch.Tensor,
        halt_threshold: Optional[float] = None,
        num_loops: Optional[int] = None,
        return_hidden: bool = False,
        last_only: bool = False,
    ) -> InkyOutput:
        """Inference with per-token early exit (batch size 1, no cache, no training).

        A token exits as soon as its cumulative halting mass reaches ``halt_threshold``. The mass not
        yet assigned is given to its exit state, so the output stays a probability-weighted mix. With
        ``halt_threshold=None`` no token exits early, and the output matches ``forward`` on the same
        input (checked in tests). Exited tokens keep their state and their stored keys and values, so
        the remaining tokens attend to them without recomputing them.
        """
        if input_ids.shape[0] != 1:
            raise ValueError("early exit needs batch size 1")
        if halt_threshold is not None and not 0.0 < halt_threshold <= 1.0:
            raise ValueError(f"halt_threshold must be in (0, 1], got {halt_threshold}")
        S = input_ids.shape[1]
        if S > self.config.max_position_embeddings:
            raise ValueError(f"sequence length {S} exceeds max_position_embeddings")
        self.eval()
        cfg = self.config
        L = num_loops if num_loops is not None else cfg.max_recurrent_loops

        cos, sin = self.rope(0, S)
        h = self.embed_tokens(input_ids)
        for layer in self.pre_layers:
            h, _, _ = layer(h, cos, sin, None)

        state = h.clone()
        stores = [
            (
                torch.zeros(1, S, cfg.kv_lora_rank, device=h.device, dtype=h.dtype),
                torch.zeros(1, 1, S, cfg.qk_rope_head_dim, device=h.device, dtype=h.dtype),
            )
            for _ in self.core.layers
        ]
        active = torch.arange(S, device=h.device)
        surv = torch.ones(S, device=h.device, dtype=torch.float32)   # probability of not stopped yet
        cum = torch.zeros(S, device=h.device, dtype=torch.float32)   # halting mass assigned so far
        out = torch.zeros(1, S, cfg.hidden_size, device=h.device, dtype=torch.float32)
        exit_loop = torch.full((S,), float(L), device=h.device)
        ssm_mem = None
        computed = 0

        for n in range(1, L + 1):
            if active.numel() == 0:
                break
            out_act, ssm_mem = self.core.forward_subset(state, n, cos, sin, active, stores, ssm_mem)
            computed += int(active.numel())
            lam = self.halting(out_act).squeeze(0)                    # [K]
            if n < L:
                w = surv[active] * lam
                surv_new = surv[active] * (1.0 - lam)
            else:
                w = surv[active]                                       # the remainder on the last loop
                surv_new = torch.zeros_like(w)
            out[0, active] += w.unsqueeze(-1) * out_act[0].float()
            cum[active] += w
            surv[active] = surv_new
            state[0, active] = out_act[0]

            if n == L or halt_threshold is None:
                exit_now = torch.full_like(active, n == L, dtype=torch.bool)
            else:
                exit_now = cum[active] >= halt_threshold
            if bool(exit_now.any()):
                ex = active[exit_now]
                out[0, ex] += surv[ex].unsqueeze(-1) * state[0, ex].float()   # unassigned mass
                surv[ex] = 0.0
                exit_loop[ex] = float(n)
            active = active[~exit_now]

        hidden = out.to(h.dtype)
        for layer in self.post_layers:
            hidden, _, _ = layer(hidden, cos, sin, None)
        hidden = self.norm(hidden)
        mean_loops = exit_loop.mean()
        common = dict(
            mean_loops=mean_loops,
            loop_applications=computed,
            full_loop_applications=S * L,
        )
        if return_hidden:
            return InkyOutput(hidden=hidden, **common)
        logits = self._project(hidden[:, -1:] if last_only else hidden)
        return InkyOutput(logits=logits, **common)

    # ------------------------------------------------------------------ writing decisions

    @torch.no_grad()
    def decide_choice(
        self,
        context_ids: torch.Tensor,
        options: Sequence[torch.Tensor],
        question_ids: Optional[torch.Tensor] = None,
        gold: Optional[int] = None,
        report: bool = False,
        display: bool = False,
    ):
        """Probability of each option, from the decision engine (Rhen Choice).

        ``context_ids`` is [1, S]. ``options`` is a list of [1, L_i] tensors, one per option.
        Each option is scored after the context and the optional question. Because the model is
        causal, the context states are the same for every option.
        Returns a tensor of shape [N] that sums to 1.
        """
        if self.arlo is None:
            raise RuntimeError("decision engine is disabled (enable_arlo=False)")
        self.eval()
        prefix = context_ids if question_ids is None else torch.cat([context_ids, question_ids], dim=1)
        P = prefix.shape[1]
        seqs = [torch.cat([prefix, opt], dim=1)[0] for opt in options]
        lengths = [s.shape[0] for s in seqs]
        # One batched forward. Options are right-padded with token 0, which is always a valid id
        # (config.pad_token_id may lie outside a small vocabulary). The model is causal, so the padding
        # cannot change any position before it, and each option is read at its own last token.
        batch = torch.zeros((len(seqs), max(lengths)), dtype=prefix.dtype, device=prefix.device)
        for i, s in enumerate(seqs):
            batch[i, :lengths[i]] = s
        h = self(batch, return_hidden=True).hidden
        query = h[:1, P - 1]          # the last prefix token is the same for every option
        states = h[:1, :P]
        last = torch.tensor(lengths, device=h.device) - 1
        option_states = h[torch.arange(len(seqs), device=h.device), last]     # [N, D]
        scores = self.arlo.choice_scores(query, option_states.unsqueeze(0), states)
        probs = torch.softmax(scores, dim=-1)[0]
        if not (report or display):
            return probs
        gold_t = None if gold is None else torch.tensor([gold])
        rep = choice_report(probs.unsqueeze(0), gold_t)
        if display:
            print_report(rep)
        return probs, rep

    @torch.no_grad()
    def decide_binary(
        self,
        context_ids: torch.Tensor,
        question_ids: Optional[torch.Tensor] = None,
        gold: Optional[int] = None,
        report: bool = False,
        display: bool = False,
    ):
        """Probability that the answer is yes, from the decision engine (Rhen Noul). Returns a scalar.

        With ``report=True`` or ``display=True`` the output is also summarised by ``binary_report``.
        Pass ``gold`` (0 or 1) to get the confusion metrics.
        """
        if self.arlo is None:
            raise RuntimeError("decision engine is disabled (enable_arlo=False)")
        self.eval()
        prefix = context_ids if question_ids is None else torch.cat([context_ids, question_ids], dim=1)
        h = self(prefix, return_hidden=True).hidden
        p = torch.sigmoid(self.arlo.binary_logit(h[:, -1], h))[0]
        if not (report or display):
            return p
        rep = binary_report(p.unsqueeze(0), None if gold is None else [gold])
        if display:
            print_report(rep)
        return p, rep

    # ------------------------------------------------------------------ training utilities

    @torch.no_grad()
    def update_moe_balance(self) -> None:
        """Move the MoE balance biases once. Call after each optimizer step during training."""
        for module in self.modules():
            if hasattr(module, "update_balance_bias"):
                module.update_balance_bias()

    # ------------------------------------------------------------------ generation

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        eos_token_id: Optional[int] = None,
        num_loops: Optional[int] = None,
        no_repeat_ngram_size: int = 4,
        bad_word_ids: Optional[Sequence[Sequence[int]]] = None,
        reduce_scratch_blocks: bool = False,
        report: bool = False,
        display: bool = False,
    ):
        """Sample a continuation.

        ``bad_word_ids`` blocks token sequences (see ``style.build_bad_word_ids``).
        ``no_repeat_ngram_size`` blocks any repeated n-gram (0 turns it off).
        ``reduce_scratch_blocks`` applies PENCIL-style reduction: when the model closes a scratch
        block, the notes are removed from the sequence and the cache is rebuilt. It needs batch
        size 1. Tokens in the prompt are never removed.
        """
        self.eval()
        cfg = self.config
        if reduce_scratch_blocks and input_ids.shape[0] != 1:
            raise ValueError("scratch reduction needs batch size 1")
        prompt_len = input_ids.shape[1]
        cache = InkyCache(cfg)
        seq = input_ids
        out = self(seq, past=cache, num_loops=num_loops, last_only=True)
        for _ in range(max_new_tokens):
            if cache.seq_len >= cfg.max_position_embeddings:
                break
            logits = _apply_repetition_penalty(out.logits[:, -1, :], seq, repetition_penalty)
            logits = _block_tokens(logits, seq, no_repeat_ngram_size, bad_word_ids)
            if do_sample:
                next_tok = _sample(logits, temperature, top_k, top_p)
            else:
                next_tok = logits.argmax(dim=-1)
            seq = torch.cat([seq, next_tok[:, None]], dim=1)
            if eos_token_id is not None and bool((next_tok == eos_token_id).all()):
                break
            if reduce_scratch_blocks and int(next_tok[0]) == cfg.scratch_close_id:
                reduced = reduce_scratch(
                    seq[0].tolist(), cfg.scratch_open_id, cfg.scratch_sep_id, cfg.scratch_close_id, start=prompt_len
                )
                if len(reduced) != seq.shape[1]:
                    # The notes are gone, so the cache for the remaining text is rebuilt from scratch.
                    # The loop depth is kept, so the text after the close uses the same depth as before.
                    loops = cache.num_loops
                    seq = torch.tensor([reduced], dtype=seq.dtype, device=seq.device)
                    cache = InkyCache(cfg)
                    out = self(seq, past=cache, num_loops=loops, last_only=True)
                    continue
            out = self(next_tok[:, None], past=cache, last_only=True)
        if not (report or display):
            return seq
        rep = text_report(seq[0, prompt_len:], bad_word_ids)
        if display:
            print_report(rep)
        return seq, rep


def _block_tokens(
    logits: torch.Tensor,
    seq: torch.Tensor,
    no_repeat_ngram_size: int,
    bad_word_ids: Optional[Sequence[Sequence[int]]],
) -> torch.Tensor:
    if not no_repeat_ngram_size and not bad_word_ids:
        return logits
    logits = logits.clone()
    for b in range(seq.shape[0]):
        ids = seq[b].tolist()
        banned = banned_by_no_repeat_ngram(ids, no_repeat_ngram_size)
        if bad_word_ids:
            banned |= banned_by_bad_words(ids, bad_word_ids)
        if banned:
            logits[b, sorted(banned)] = float("-inf")
    return logits


def _apply_repetition_penalty(logits: torch.Tensor, seq: torch.Tensor, penalty: float) -> torch.Tensor:
    if penalty == 1.0:
        return logits
    picked = torch.gather(logits, 1, seq)
    picked = torch.where(picked < 0, picked * penalty, picked / penalty)
    return logits.scatter(1, seq, picked)


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    logits = logits / max(temperature, 1e-5)
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
