"""Inky language model: pre layers, a looped recurrent core with PonderNet halting, post layers."""

import warnings
import weakref
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel

from .arlo import Arlo, mean_pool_options, option_ids
from .config import InkyConfig
from .data import DEFAULT_RESERVED_IDS
from .layers import (
    InkyBlock,
    KVEntry,
    LatentKV,
    MoRRouter,
    PonderHalting,
    RecurrentCore,
    RMSNorm,
    RotaryEmbedding,
    geometric_halting_weights,
    geometric_prior,
    pondernet_kl,
)
from .metrics import binary_report, choice_report, print_report, text_report
from .scratch import DEFAULT_MAX_NOTE_TOKENS, ScratchTracker, reduce_scratch  # noqa: F401
from .style import NgramBanIndex, banned_by_bad_words

# Default for ``eos_token_id`` in generate / self_speculative_generate: use ``config.eos_token_id``.
# An explicit None means "never stop at EOS"; an int means "stop at this id".
_USE_CONFIG_EOS = object()


def _claim(entry, cache: "InkyCache"):
    """Marks a freshly built latent entry as owned by ``cache`` (see ``_writable_past``). Other entries pass through."""
    if isinstance(entry, LatentKV):
        entry.owner = weakref.ref(cache)
    return entry


def _writable_past(entry, cache: "InkyCache"):
    """The entry to hand to a layer as ``past``: an entry that ``cache`` itself produced may be extended in
    place (its buffer has spare capacity, so no per-token concatenation). An entry that came from another
    cache (a fork shares the lists of its parent), a snapshot or a plain tuple is not writable, so the layer
    copies it on the first append and the owner of the buffer is never overwritten."""
    if isinstance(entry, LatentKV) and entry.owner is not None and entry.owner() is cache:
        return entry.alias_writable()
    return entry


class InkyCache:
    """Per-sequence decoding state.

    One latent-KV entry per pre layer, per post layer, and per (loop, recurrent layer), plus the
    SSM highway's running prefix sum for each loop (None when ``use_highway`` is off). The loop count is
    fixed at the first call.

    Latent-attention entries are ``LatentKV``: a preallocated buffer that grows geometrically (up to
    ``max_position_embeddings``) plus a valid length. A decode step writes the new latents into the buffer
    instead of concatenating the whole cache. They unpack as ``(c_kv, k_pe)``, are never edited in place
    once built, and only the cache that produced an entry extends its buffer in place; a fork or snapshot
    shares the buffer safely because anything else copies on its first append.
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
    # MoE balance-free aux: pre and post layers summed, recurrent layers averaged over the loops run.
    moe_aux_loss: Optional[torch.Tensor] = None
    # Expected number of loops per token under the halting distribution, sum_n n * p_n(t), averaged over
    # tokens. Same definition in forward and forward_early_exit (in [1, L]). Under routing_mode='mor' it is
    # the mean of the routed depths (each token's realised loop count, capped by num_loops).
    mean_loops: Optional[torch.Tensor] = None
    hidden: Optional[torch.Tensor] = None      # final hidden states, set when return_hidden=True
    # Token-loop pairs actually computed, and tokens x loops for the same input. Set by forward_early_exit
    # and, under routing_mode='mor', by forward (token rows run through the recurrent block stack).
    loop_applications: Optional[int] = None
    full_loop_applications: Optional[int] = None
    # Realised exit loop per token, averaged (early exit only). Equals L when nothing exits early.
    # Not the same quantity as mean_loops: it is the loop a token stopped at, not the expected value.
    mean_exit_loop: Optional[torch.Tensor] = None
    # Set when labels are given: detached, unweighted loss terms. Total loss is
    # ce + z_loss_coef * z_loss + moe_aux + ponder_step_loss_coef * step_ce + ponder_kl_coef * ponder_kl.
    # Under routing_mode='mor': ce + z_loss_coef * z_loss + moe_aux + mor_capacity_aux_coef * mor_balance
    # (step_ce and ponder_kl are reported as 0).
    loss_parts: Optional[Dict[str, torch.Tensor]] = None


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
        mor = config.routing_mode == "mor"
        # Under MoR routing the depth router replaces the PonderNet gate, so the gate is not created.
        self.halting = None if mor else PonderHalting(config.hidden_size, config.ponder_prior_steps)
        self.arlo = Arlo(config) if config.enable_arlo else None
        self.mor_router = MoRRouter(config) if mor else None
        self.post_init()
        # Start the halting gate at its prior. Set after post_init, which would reset the bias.
        if self.halting is not None:
            self.halting.reset_prior()
        if self.mor_router is not None:
            self.mor_router.reset_prior(config.mor_target)
        # ARLO-01: the decision engine is an untrained prior until train_arlo has run. A gradient hook on
        # each Arlo parameter sets the flag on the first backward pass (train_arlo does one per step), so
        # arlo.py does not need to change. The flag is not saved with the weights (see ``arlo_trained``).
        self._arlo_trained = False
        if self.arlo is not None:
            for p in self.arlo.parameters():
                p.register_post_accumulate_grad_hook(self._on_arlo_grad)

    def _on_arlo_grad(self, param: torch.Tensor) -> None:
        self._arlo_trained = True

    @property
    def arlo_trained(self) -> bool:
        """True once the Arlo parameters have received a gradient in this process (train_arlo does this).

        Not saved with the weights: a model reloaded from a checkpoint reports False until its decision
        engine is trained again (or ``mark_arlo_trained`` is called after loading trained weights).
        """
        return bool(self._arlo_trained)

    def mark_arlo_trained(self) -> None:
        """Declare the decision engine trained, e.g. after loading weights trained elsewhere."""
        self._arlo_trained = True

    def _warn_if_arlo_untrained(self) -> None:
        if not self._arlo_trained:
            warnings.warn(
                "decision engine (Arlo) has not been trained in this process; its outputs are an untrained "
                "prior. Run inky.arlo.train_arlo first, or call mark_arlo_trained() if the weights were trained "
                "elsewhere (ARLO-01).",
                UserWarning,
                stacklevel=4,     # helper -> decide_* -> torch.no_grad wrapper -> caller
            )

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
            "norm + halting": n(self.norm) + (n(self.halting) if self.halting is not None else 0),
            "decision engine": n(self.arlo) if self.arlo is not None else 0,
        }
        if self.mor_router is not None:
            parts["mor_router"] = n(self.mor_router)
        parts["total"] = n(self)
        return parts

    def _choose_loops(
        self,
        cache: Optional[InkyCache],
        num_loops: Optional[int],
        generator: Optional[torch.Generator] = None,
    ) -> int:
        if cache is not None and cache.num_loops is not None:
            if num_loops is not None and num_loops != cache.num_loops:
                raise ValueError(
                    f"num_loops={num_loops} differs from the {cache.num_loops} loops this cache was filled with"
                )
            return cache.num_loops
        if num_loops is not None:
            return int(num_loops)
        if self.config.routing_mode == "mor":
            return self.config.max_recurrent_loops      # the router sets each token's depth; this is the cap
        if self.training:
            # Sample a depth per batch so the shared core learns to work at any number of loops.
            # Drawn from torch's RNG (or ``generator``), so torch.manual_seed makes runs reproducible.
            draw = torch.randint(
                self.config.min_recurrent_loops, self.config.max_recurrent_loops + 1, (1,), generator=generator
            )
            return int(draw)
        return self.config.max_recurrent_loops

    def _state_rms(self, x: torch.Tensor) -> torch.Tensor:
        """Per-token RMS of ``x`` over the last dimension (kept as a size-1 dim), in float32."""
        return x.float().pow(2).mean(-1, keepdim=True).add(self.config.rms_norm_eps).sqrt()

    def _mix_loop_states(self, states: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Halting-weighted mixture of the loop states. ``states`` [L, B, S, D] (float32), ``weights`` [L, B, S].

        With ``loop_mix_normalize`` (default) the loop states are mixed by direction and the result is given the
        p-weighted mean RMS: ``(sum_n p_n h_n / rms_n) * (sum_n p_n rms_n)``. Loop n then contributes in
        proportion to ``p_n`` however large its state is (the stream grows with every loop), and the output
        keeps the stream's scale. When all loop states have the same RMS this equals ``sum_n p_n h_n``.
        Without it the mixture is the plain weighted sum."""
        w = weights.unsqueeze(-1)
        if not self.config.loop_mix_normalize:
            return (w * states).sum(dim=0)
        rms = self._state_rms(states)                                    # [L, B, S, 1]
        return (w * states / rms).sum(dim=0) * (w * rms).sum(dim=0)

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

    def _run_head(self, fn, *args):
        """Run a per-chunk loss function, recomputing it in backward when ``lm_loss_recompute`` is set.

        Recomputing keeps only one chunk of logits alive (the default). Turning it off removes the
        second head pass in backward (CORE-08) and keeps each chunk's logits for backward instead.
        """
        if self.training and torch.is_grad_enabled() and self.config.lm_loss_recompute:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    def _lm_loss(self, hidden: torch.Tensor, labels: torch.Tensor, chunk_size: int = 1024):
        """Chunked next-token cross-entropy. Full-vocabulary logits are never kept for the whole sequence
        when ``lm_loss_recompute`` is on (the default)."""
        h = hidden[:, :-1].reshape(-1, hidden.shape[-1])
        y = labels[:, 1:].reshape(-1)
        ce_total = hidden.new_zeros((), dtype=torch.float32)
        z_total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, h.shape[0], chunk_size):
            hc, yc = h[start:start + chunk_size], y[start:start + chunk_size]
            ce, z = self._run_head(self._chunk_ce, hc, yc)
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
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Next-token cross-entropy of every loop state, weighted by its halting probability.

        This trains each loop's state to predict on its own, so a shorter loop budget gives a
        usable output and not only a partly trained one. Divided by the number of valid tokens.

        The halting weights are detached here: the gate is trained by the main mixture loss and the
        ponder KL, and this term only trains the loop states. With ``step_ce_token_fraction`` < 1, a
        random subset of target positions (drawn once, shared by all loops) is scored. The sum is
        rescaled by (positions / subset size), so its expectation equals the full-position value.
        """
        weights = weights.detach()
        y_all = labels[:, 1:].reshape(-1)
        n_valid = (y_all != -100).sum().clamp(min=1).float()
        N = y_all.shape[0]
        fraction = self.config.step_ce_token_fraction
        D = self.config.hidden_size
        if fraction < 1.0:
            k = max(1, min(N, round(fraction * N)))
            sel = torch.randperm(N, generator=generator)[:k].sort().values.to(y_all.device)
            y = y_all[sel]
            scale = float(N) / float(k)
        else:
            sel = None
            y = y_all
            scale = 1.0
        M = y.shape[0]
        total = weights.new_zeros((), dtype=torch.float32)
        for n, state in enumerate(loop_states):
            flat = state[:, :-1].reshape(-1, D)
            wn = weights[n][:, :-1].reshape(-1).float()
            if sel is not None:
                flat = flat[sel]
                wn = wn[sel]
            hn = self.norm(flat)
            for start in range(0, M, chunk_size):
                hc = hn[start:start + chunk_size]
                yc = y[start:start + chunk_size]
                wc = wn[start:start + chunk_size]
                total = total + self._run_head(self._chunk_weighted_ce, hc, yc, wc)
        return total * (scale / n_valid)

    # ------------------------------------------------------------------ MoR routing

    def _mor_route(
        self,
        h: torch.Tensor,
        loops: int,
        keep: Optional[torch.Tensor],
        generator: Optional[torch.Generator],
    ):
        """Depth router. Returns ``(depth [B, S] long, gate [B, S] float or None, probs [B, S, n_depths])``.

        The logits come from each token's own state only (causal). Training samples the depth with
        Gumbel-max from ``logits / mor_temperature`` (``mor_sample_depth=False`` takes the argmax instead);
        eval always takes the argmax. ``probs`` is the tempered softmax; ``gate`` is the probability of the
        chosen depth (None when ``mor_gate_scale`` is off) and carries the router's gradient. Depth is
        capped at ``loops`` and set to 0 at padded positions (``keep`` False), which never run.
        """
        cfg = self.config
        scaled = self.mor_router(h) / cfg.mor_temperature
        if self.training and cfg.mor_sample_depth:
            dev = generator.device if generator is not None else scaled.device
            u = torch.rand(scaled.shape, generator=generator, device=dev).to(scaled.device)
            noise = -torch.log(-torch.log(u.clamp(1e-10, 1.0 - 1e-7)))
            choice = (scaled.detach() + noise).argmax(dim=-1)
        else:
            choice = scaled.detach().argmax(dim=-1)
        probs = torch.softmax(scaled, dim=-1)
        gate = probs.gather(-1, choice.unsqueeze(-1)).squeeze(-1) if cfg.mor_gate_scale else None
        depth = (choice + cfg.min_recurrent_loops).clamp(max=loops)
        if keep is not None:
            depth = depth * keep.to(depth.dtype)
        return depth, gate, probs

    def _mor_balance(self, probs: torch.Tensor, keep: Optional[torch.Tensor]) -> torch.Tensor:
        """Depth-balance loss: (mean expected depth - mor_target)^2, with the expectation taken under the
        router's soft distribution so that it is differentiable. Averaged over real tokens only."""
        cfg = self.config
        depths = torch.arange(
            cfg.min_recurrent_loops, cfg.max_recurrent_loops + 1, device=probs.device, dtype=torch.float32
        )
        expected = (probs * depths).sum(-1)                                   # [B, S]
        if keep is None:
            mean_depth = expected.mean()
        else:
            real = keep.to(expected.dtype)
            mean_depth = (expected * real).sum() / real.sum().clamp(min=1.0)
        return (mean_depth - cfg.mor_target).pow(2)

    def _mor_recurrence(self, h, loops, cache, past_len, cos, sin, attn_mask, segment_ids, valid_mask, keep,
                        generator):
        """Routed recurrence for ``routing_mode='mor'`` (see ``RecurrentCore.forward_mor``).

        Returns ``(h_out, core_aux, depth, probs, rows_computed)`` and fills the per-loop cache entries
        and highway prefix sums of ``cache`` (when given).
        """
        depth, gate, probs = self._mor_route(h, loops, keep, generator)
        past_core = past_prefix = None
        if cache is not None:
            past_core = [
                [_writable_past(e, cache) for e in cache.core[i]] if cache.core[i] is not None else None
                for i in range(loops)
            ]
            past_prefix = list(cache.prefix)
        h_out, aux, rows, entries, prefixes = self.core.forward_mor(
            h, depth, gate, loops, cos, sin,
            past_core=past_core, past_prefix=past_prefix, past_len=past_len,
            attn_mask=attn_mask, segment_ids=segment_ids, valid_mask=valid_mask,
            gather=self.config.mor_gather, want_cache=cache is not None,
        )
        if cache is not None:
            for i, ents in enumerate(entries):
                cache.core[i] = [_claim(e, cache) for e in ents]
                cache.prefix[i] = prefixes[i]
        return h_out, aux, depth, probs, rows

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
        attention_mask: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> InkyOutput:
        """Causal forward pass.

        ``labels`` (same shape as ``input_ids``; -100 to ignore) returns the training loss and no
        logits. Without labels, logits are returned; ``last_only=True`` keeps only the final position.
        Pass ``past=InkyCache(...)`` (or ``use_cache=True``) to decode incrementally. The loop count
        is fixed by the first call of a cached sequence.

        Packing (uncached passes only, i.e. no ``past`` and no ``use_cache``):
          * ``segment_ids`` [B, S] (int): tokens attend only within their own segment, and the state
            highway's running mean restarts at each segment. Packed documents then match running each
            document on its own.
          * ``attention_mask`` [B, S] (bool or 0/1, True = real token): padded keys are not attended to,
            and padding is left out of the state highway's running mean (CORE-10), so a padded row matches
            its unpadded run at the real positions. Padded positions still produce outputs, which the
            labels should ignore. The mean loop count (``mean_loops``) and the halting KL average over the
            real positions only. The router's z-loss in the MoE layers still averages over every position.
            With left padding, the last pad position predicts the first real token, so that target label
            must also be -100 (the loss does not infer it from the mask).
        ``generator`` (optional) drives the random loop-depth draw and the step-CE token subset.

        Depth routing (``config.routing_mode``). ``"ponder"`` (default): PonderNet, every token runs every
        loop and the output is the halting-weighted mix of the loop states. ``"mor"``: token-choice
        Mixture-of-Recursions. After the pre layers a router picks a depth d(t) per token from that token's
        own state (Gumbel-max sample in training, argmax in eval and decode; capped by ``num_loops``, which
        defaults to ``max_recurrent_loops`` and is fixed by the first call of a cached sequence). Loop n
        runs only the tokens with d(t) >= n: their attention queries, latent projections, output
        projection and MoE run on the gathered active rows, so compute scales with the sum of the depths.
        An exited token keeps its state frozen and keeps the latents of its last active loop as the keys and
        values other tokens see in deeper loops (stale reuse). The step modulation and the SSM highway see
        the frozen state of exited tokens: the highway's running mean at loop n averages every earlier
        token's loop-n modulated state, active or not, but only active tokens use its output. An active
        token's update is ``h + gate * (core_out - h)`` with gate the router probability of its depth
        (``mor_gate_scale``). The output state of a token is its state after its last active loop (no
        halting mixture). The loss is CE + z-loss on that final state, the MoE aux, and
        ``mor_capacity_aux_coef * (E[depth] - target)^2``; ``loss_parts`` reports step_ce and ponder_kl as
        0. A cached step routes the new token, runs it for its d loops only, and fills the deeper loops'
        caches with its last latent and its frozen highway contribution, so cached decoding equals the full
        forward. ``attention_mask`` / ``segment_ids`` are supported (padded positions get depth 0 and never
        run). ``ssm_mixer_layers`` is rejected at config time. Layer-drop (``loop_dropout_prob``) skips a
        layer for all active tokens of a loop; an exited token whose layer-j latent was never written keeps
        a zero key/value there (training only).
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
        attn_mask = self._packing_mask(B, S, attention_mask, segment_ids, input_ids.device, cache is not None)
        # Uncached passes only (checked in _packing_mask): padding is excluded from the highway's mean.
        valid_mask = None
        keep = None
        if attention_mask is not None:
            keep = attention_mask.to(device=input_ids.device, dtype=torch.bool)
            valid_mask = keep

        cos, sin = self.rope(past_len, S)
        h = self.embed_tokens(input_ids)
        if self.training and self.config.embedding_dropout > 0.0:
            h = F.dropout(h, p=self.config.embedding_dropout, training=True)     # TRAIN-04
        moe_aux = h.new_zeros((), dtype=torch.float32)

        for i, layer in enumerate(self.pre_layers):
            h, present, aux = layer(h, cos, sin, _writable_past(cache.pre[i], cache) if cache is not None else None,
                                    attn_mask=attn_mask, segment_ids=segment_ids)
            if cache is not None:
                cache.pre[i] = _claim(present, cache)
            moe_aux = moe_aux + aux

        loops = self._choose_loops(cache, num_loops, generator)
        if cache is not None and cache.num_loops is None:
            cache.num_loops = loops
            cache.core = [None] * loops
            cache.prefix = [None] * loops

        mor = self.config.routing_mode == "mor"
        mor_probs = mor_rows = None
        if mor:
            # Mixture-of-Recursions: route, then run each loop on the active tokens only (see forward docstring).
            h_out, core_aux, mor_depth, mor_probs, mor_rows = self._mor_recurrence(
                h, loops, cache, past_len, cos, sin, attn_mask, segment_ids, valid_mask, keep, generator
            )
            moe_aux = moe_aux + core_aux
            per_token_loops = mor_depth.to(torch.float32)
            if keep is not None:
                real = keep.to(per_token_loops.dtype)
                mean_loops = (per_token_loops * real).sum() / real.sum().clamp(min=1.0)
            else:
                mean_loops = per_token_loops.mean()
        else:
            # PonderNet halting over the recurrent loops. Each loop emits lambda_n(t); the halting
            # distribution p_n(t) puts stop mass on each loop, and the output is the expectation of the
            # loop states under p. Every loop still runs, so cached decoding matches the full forward
            # pass exactly. This version saves no compute (see README, Known issues).
            h_loop, prev_mem = h, None
            lambdas: List[torch.Tensor] = []
            loop_states: List[torch.Tensor] = []
            core_aux = h.new_zeros((), dtype=torch.float32)   # summed over loops, averaged below

            for step in range(1, loops + 1):
                idx = step - 1
                h_next, mem, new_prefix, presents, aux = self.core(
                    h_loop,
                    step,
                    cos,
                    sin,
                    past=(
                        [_writable_past(e, cache) for e in cache.core[idx]]
                        if cache is not None and cache.core[idx] is not None else None
                    ),
                    prefix_sum=cache.prefix[idx] if cache is not None else None,
                    prev_mem=prev_mem,
                    past_len=past_len,
                    attn_mask=attn_mask,
                    segment_ids=segment_ids,
                    valid_mask=valid_mask,
                )
                if cache is not None:
                    cache.core[idx] = [_claim(e, cache) for e in presents]
                    cache.prefix[idx] = new_prefix
                prev_mem = mem
                core_aux = core_aux + aux

                lambdas.append(self.halting(h_next))
                loop_states.append(h_next)
                h_loop = h_next
            # The recurrent layers run once per loop. Averaging keeps the aux at the scale of one pass, so
            # it does not grow with the loop count. Each layer's own term is still its per-loop mean.
            moe_aux = moe_aux + core_aux / loops

            weights = geometric_halting_weights(lambdas)                 # [L, B, S], sums to 1 over L
            states = torch.stack(loop_states, dim=0).float()             # [L, B, S, D]
            h_out = self._mix_loop_states(states, weights)               # [B, S, D]
            loop_index = torch.arange(1, loops + 1, device=h.device, dtype=torch.float32).view(-1, 1, 1)
            per_token_loops = (weights * loop_index).sum(dim=0)           # [B, S] expected loops per token
            if keep is not None:
                real = keep.to(per_token_loops.dtype)
                mean_loops = (per_token_loops * real).sum() / real.sum().clamp(min=1.0)
            else:
                mean_loops = per_token_loops.mean()

        h = h_out.to(h.dtype)
        for i, layer in enumerate(self.post_layers):
            h, present, aux = layer(h, cos, sin, _writable_past(cache.post[i], cache) if cache is not None else None,
                                    attn_mask=attn_mask, segment_ids=segment_ids)
            if cache is not None:
                cache.post[i] = _claim(present, cache)
            moe_aux = moe_aux + aux

        if cache is not None:
            cache.seq_len += S

        h = self.norm(h)

        extra = {}
        if mor:
            extra = dict(loop_applications=mor_rows, full_loop_applications=B * S * loops)

        if labels is not None and mor:
            ce, z = self._lm_loss(h, labels)
            balance = self._mor_balance(mor_probs, keep)
            loss = ce + self.config.z_loss_coef * z + moe_aux + self.config.mor_capacity_aux_coef * balance
            zero = ce.detach() * 0.0
            parts = {
                "ce": ce.detach(),
                "step_ce": zero,
                "z_loss": z.detach(),
                "moe_aux": moe_aux.detach(),
                "ponder_kl": zero,
                "mor_balance": balance.detach(),
            }
            return InkyOutput(loss=loss, past=cache, moe_aux_loss=moe_aux, mean_loops=mean_loops,
                              loss_parts=parts, **extra)

        if labels is not None:
            ce, z = self._lm_loss(h, labels)
            step_ce = self._weighted_step_ce(loop_states, weights, labels, generator=generator)
            prior = geometric_prior(loops, self.halting.prior_lambda, device=h.device)
            kl = pondernet_kl(weights, prior, mask=keep)
            loss = (
                ce
                + self.config.z_loss_coef * z
                + moe_aux
                + self.config.ponder_step_loss_coef * step_ce
                + self.config.ponder_kl_coef * kl
            )
            parts = {
                "ce": ce.detach(),
                "step_ce": step_ce.detach(),
                "z_loss": z.detach(),
                "moe_aux": moe_aux.detach(),
                "ponder_kl": kl.detach(),
            }
            return InkyOutput(loss=loss, past=cache, moe_aux_loss=moe_aux, mean_loops=mean_loops,
                              loss_parts=parts)

        if return_hidden:
            return InkyOutput(hidden=h, past=cache, moe_aux_loss=moe_aux, mean_loops=mean_loops, **extra)
        logits = self._project(h[:, -1:] if last_only else h)
        return InkyOutput(logits=logits, past=cache, moe_aux_loss=moe_aux, mean_loops=mean_loops, **extra)

    def _packing_mask(
        self,
        B: int,
        S: int,
        attention_mask: Optional[torch.Tensor],
        segment_ids: Optional[torch.Tensor],
        device: torch.device,
        cached: bool,
    ) -> Optional[torch.Tensor]:
        """Validates the packing arguments and builds the key-padding mask [B, 1, S, S] (or None).

        The diagonal is always allowed, so a padded query row never has every key masked (that would
        give NaN in attention). Causality is applied by the attention itself.
        """
        if attention_mask is None and segment_ids is None:
            return None
        if cached:
            raise ValueError(
                "attention_mask and segment_ids need an uncached forward pass (no past, no use_cache)"
            )
        if segment_ids is not None and tuple(segment_ids.shape) != (B, S):
            raise ValueError(f"segment_ids must have shape {(B, S)}, got {tuple(segment_ids.shape)}")
        if attention_mask is None:
            return None
        if tuple(attention_mask.shape) != (B, S):
            raise ValueError(f"attention_mask must have shape {(B, S)}, got {tuple(attention_mask.shape)}")
        keep = attention_mask.to(device=device, dtype=torch.bool)
        eye = torch.eye(S, dtype=torch.bool, device=device)
        return keep[:, None, None, :] | eye

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

        Loop statistics: ``mean_loops`` is the expected loop count sum_n n * p_n(t) over the mass each
        token actually assigned to loop n (the unassigned remainder counts at its exit loop). It is the
        same quantity ``forward`` reports, and equals it when ``halt_threshold=None``. ``mean_exit_loop``
        is the loop each token stopped at, averaged over tokens.

        Under ``routing_mode='mor'`` the exit loop of a token is simply its router depth (capped by
        ``num_loops``): the call runs the routed forward pass, which already skips exited tokens, and
        ``halt_threshold`` must be None (there is no halting mass to threshold). ``mean_loops`` and
        ``mean_exit_loop`` are both the mean routed depth.
        """
        if input_ids.shape[0] != 1:
            raise ValueError("early exit needs batch size 1")
        if self.config.routing_mode == "mor":
            if halt_threshold is not None:
                raise ValueError(
                    "halt_threshold is not used with routing_mode='mor': a token exits at its router depth"
                )
            self.eval()
            L = int(num_loops) if num_loops is not None else self.config.max_recurrent_loops
            out = self.forward(input_ids, num_loops=L, return_hidden=True)
            common = dict(
                mean_loops=out.mean_loops,
                mean_exit_loop=out.mean_loops,
                loop_applications=out.loop_applications,
                full_loop_applications=out.full_loop_applications,
            )
            if return_hidden:
                return InkyOutput(hidden=out.hidden, **common)
            h = out.hidden
            return InkyOutput(logits=self._project(h[:, -1:] if last_only else h), **common)
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
        # loop_mix_normalize: ``out`` accumulates sum_n p_n h_n / rms_n and ``scale_acc`` sum_n p_n rms_n, and the
        # two are multiplied at the end (see _mix_loop_states); otherwise ``out`` is the plain sum_n p_n h_n.
        norm_mix = cfg.loop_mix_normalize
        scale_acc = torch.zeros(S, device=h.device, dtype=torch.float32)
        exit_loop = torch.full((S,), float(L), device=h.device)
        expected_loops = torch.zeros(S, device=h.device, dtype=torch.float32)  # sum_n n * (mass at loop n)
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
            if norm_mix:
                r = self._state_rms(out_act[0])                           # [K, 1]
                out[0, active] += w.unsqueeze(-1) * out_act[0].float() / r
                scale_acc[active] += w * r[:, 0]
            else:
                out[0, active] += w.unsqueeze(-1) * out_act[0].float()
            cum[active] += w
            expected_loops[active] += w * float(n)
            surv[active] = surv_new
            state[0, active] = out_act[0]

            if n == L or halt_threshold is None:
                exit_now = torch.full_like(active, n == L, dtype=torch.bool)
            else:
                exit_now = cum[active] >= halt_threshold
            if bool(exit_now.any()):
                ex = active[exit_now]
                st = state[0, ex].float()
                if norm_mix:                                                   # unassigned mass
                    r = self._state_rms(st)
                    out[0, ex] += surv[ex].unsqueeze(-1) * st / r
                    scale_acc[ex] += surv[ex] * r[:, 0]
                else:
                    out[0, ex] += surv[ex].unsqueeze(-1) * st
                expected_loops[ex] += surv[ex] * float(n)                      # ...counted at its exit loop
                surv[ex] = 0.0
                exit_loop[ex] = float(n)
            active = active[~exit_now]

        if norm_mix:
            out = out * scale_acc.view(1, S, 1)
        hidden = out.to(h.dtype)
        for layer in self.post_layers:
            hidden, _, _ = layer(hidden, cos, sin, None)
        hidden = self.norm(hidden)
        common = dict(
            mean_loops=expected_loops.mean(),
            mean_exit_loop=exit_loop.mean(),
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

        ``context_ids`` is [1, S]. ``options`` is a list of [1, L_i] tensors, one per option; every option
        needs at least one token (ValueError otherwise). Each option is scored after the context and the
        optional question, and is represented by the mean of its own token states (padding excluded).
        Because the model is causal, the context states are the same for every option.
        Returns a tensor of shape [N] that sums to 1. The model's train/eval mode is restored afterwards.
        """
        if self.arlo is None:
            raise RuntimeError("decision engine is disabled (enable_arlo=False)")
        self._warn_if_arlo_untrained()
        if len(options) == 0:
            raise ValueError("decide_choice needs at least one option")
        opts = [option_ids(opt, i) for i, opt in enumerate(options)]
        was_training = self.training
        self.eval()
        try:
            prefix = context_ids if question_ids is None else torch.cat([context_ids, question_ids], dim=1)
            P = prefix.shape[1]
            seqs = [torch.cat([prefix, o.to(prefix.device)], dim=1)[0] for o in opts]
            lengths = [s.shape[0] for s in seqs]
            # One batched forward. Options are right-padded with token 0, which is always a valid id
            # (config.pad_token_id may lie outside a small vocabulary). The model is causal, so the padding
            # cannot change any position before it, and each option's mean covers only its own tokens.
            batch = torch.zeros((len(seqs), max(lengths)), dtype=torch.long, device=prefix.device)
            for i, s in enumerate(seqs):
                batch[i, :lengths[i]] = s
            h = self(batch, return_hidden=True).hidden
            query = h[:1, P - 1]          # the last prefix token is the same for every option
            states = h[:1, :P]
            option_states = mean_pool_options(h, P, lengths)                  # [N, D]
            scores = self.arlo.choice_scores(query, option_states.unsqueeze(0), states)
        finally:
            self.train(was_training)
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
        Pass ``gold`` (0 or 1) to get the confusion metrics. The model's train/eval mode is restored afterwards.
        """
        if self.arlo is None:
            raise RuntimeError("decision engine is disabled (enable_arlo=False)")
        self._warn_if_arlo_untrained()
        was_training = self.training
        self.eval()
        try:
            prefix = context_ids if question_ids is None else torch.cat([context_ids, question_ids], dim=1)
            h = self(prefix, return_hidden=True).hidden
            p = torch.sigmoid(self.arlo.binary_logit(h[:, -1], h))[0]
        finally:
            self.train(was_training)
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

    @torch.no_grad()
    def reset_moe_load(self) -> None:
        """Discard the MoE routing counts collected since the last balance update.

        The counters fill on every forward pass, including eval and the pass of a skipped step, so call this
        after any pass whose routing should not move the balance biases. ``train`` does so after each eval and
        after each skipped step."""
        for module in self.modules():
            if hasattr(module, "reset_expert_load"):
                module.reset_expert_load()

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
        eos_token_id: Union[int, None, object] = _USE_CONFIG_EOS,
        num_loops: Optional[int] = None,
        no_repeat_ngram_size: int = 4,
        bad_word_ids: Optional[Sequence[Sequence[int]]] = None,
        reduce_scratch_blocks: bool = False,
        report: bool = False,
        display: bool = False,
        stock_word_penalty: float = 0.0,
        soft_penalty_ids: Optional[Sequence[int]] = None,
        repeat_window: Optional[int] = None,
        count_prose_only: bool = False,
        scratch_max_note_tokens: Optional[int] = DEFAULT_MAX_NOTE_TOKENS,
        blocked_token_ids: Optional[Sequence[int]] = None,
    ):
        """Sample a continuation.

        ``blocked_token_ids`` are never emitted (set to -inf). None (the default) blocks the reserved
        special ids of the tokenizer, ``DEFAULT_RESERVED_IDS`` (49974 to 50002), minus the scratch ids,
        which generation uses through ``reduce_scratch_blocks``, and minus the configured EOS id, which
        must stay allowed (DATA-05). Pass an explicit list to block other ids, or ``[]`` to block nothing.
        The EOS id (``eos_token_id``, when given) and the configured ``config.eos_token_id`` are never
        blocked, even if they are in the list. Ids outside the vocabulary are ignored.
        ``eos_token_id`` stops generation once every row emits it: left at its default it is
        ``config.eos_token_id`` (DOC-05), an int overrides it, and an explicit ``None`` never stops at EOS.
        ``bad_word_ids`` blocks token sequences (see ``style.build_bad_word_ids``).
        ``no_repeat_ngram_size`` blocks any repeated n-gram (0 turns it off; 1 is rejected, because it
        would ban every token already seen).
        ``repeat_window`` (default None = the whole history) limits the repetition penalty and the
        no-repeat rule to the last ``repeat_window`` tokens, prompt included.
        ``stock_word_penalty`` (default 0) subtracts this value from the logit of every id in
        ``soft_penalty_ids`` (see ``style.soft_penalty_token_ids``). It is a soft preference, not a ban.
        ``reduce_scratch_blocks`` applies PENCIL-style reduction: when the model closes a scratch
        block, the notes are removed from the sequence. It needs batch size 1. Tokens in the prompt
        are never removed. Notes longer than ``scratch_max_note_tokens`` (None = no cap) are not
        treated as notes. Structurally invalid markers (a separator with no open block, a close with
        no separator) are masked out. After a close, the cache is rolled back to the block's open
        marker and only the text after it is recomputed, so no full re-prefill happens.
        ``count_prose_only`` (needs ``reduce_scratch_blocks`` to differ) counts only prose tokens
        against ``max_new_tokens``, so notes do not use up the budget. A note still open at the stop
        point is kept in the output.

        Each batch row keeps an incremental no-repeat n-gram index, updated with every new token, so
        the decode loop does not rebuild it from the whole sequence at each step.
        """
        self.eval()
        cfg = self.config
        if eos_token_id is _USE_CONFIG_EOS:
            eos_token_id = cfg.eos_token_id      # DOC-05: stop at the configured <|endoftext|> by default
        if reduce_scratch_blocks and input_ids.shape[0] != 1:
            raise ValueError("scratch reduction needs batch size 1")
        if no_repeat_ngram_size == 1:
            raise ValueError(
                "no_repeat_ngram_size=1 would ban every token already seen; use 0 to turn the rule off "
                "or a value >= 2"
            )
        if no_repeat_ngram_size < 0:
            raise ValueError(f"no_repeat_ngram_size must be >= 0, got {no_repeat_ngram_size}")
        if repeat_window is not None and repeat_window < 1:
            raise ValueError(f"repeat_window must be None or >= 1, got {repeat_window}")
        if scratch_max_note_tokens is not None and scratch_max_note_tokens < 0:
            raise ValueError(f"scratch_max_note_tokens must be None or >= 0, got {scratch_max_note_tokens}")
        if stock_word_penalty < 0:
            raise ValueError(f"stock_word_penalty must be >= 0, got {stock_word_penalty}")
        penalty_mask = None
        if stock_word_penalty > 0:
            ids_list = sorted({int(t) for t in (soft_penalty_ids or ())})
            if not ids_list:
                raise ValueError("stock_word_penalty > 0 needs soft_penalty_ids")
            if ids_list[0] < 0 or ids_list[-1] >= cfg.vocab_size:
                raise ValueError("soft_penalty_ids must be token ids inside the vocabulary")
            penalty_mask = torch.zeros(cfg.vocab_size, dtype=torch.float32, device=input_ids.device)
            penalty_mask[ids_list] = 1.0

        blocked = _blocked_ids(cfg, blocked_token_ids, eos_token_id)
        prompt_len = input_ids.shape[1]
        cache = InkyCache(cfg)
        seq = input_ids
        # Per-row token history (Python ints). Appended to once per new token; never re-listed.
        history: List[List[int]] = [row.tolist() for row in input_ids]
        ngram = (
            [_make_ngram_index(no_repeat_ngram_size, row, repeat_window) for row in history]
            if no_repeat_ngram_size else None
        )
        tail_len = max((len(s) - 1 for s in bad_word_ids), default=0) if bad_word_ids else 0
        scratch = (
            ScratchTracker(cfg.scratch_open_id, cfg.scratch_sep_id, cfg.scratch_close_id,
                           scratch_max_note_tokens, prefix=history[0])
            if reduce_scratch_blocks else None
        )
        # Before each open marker: the cache state and the logits that predicted the marker's position.
        # Keyed by the marker's index in seq. Used to roll back the cache when its block closes.
        snapshots: Dict[int, Tuple[dict, InkyOutput]] = {}
        out = self(seq, past=cache, num_loops=num_loops, last_only=True)
        steps = 0
        while True:
            if count_prose_only and scratch is not None:
                if scratch.prose_count(prompt_len) >= max_new_tokens:
                    break
            elif steps >= max_new_tokens:
                break
            steps += 1
            if cache.seq_len >= cfg.max_position_embeddings:
                break
            logits = _apply_repetition_penalty(out.logits[:, -1, :], seq, repetition_penalty, repeat_window)
            if penalty_mask is not None:
                logits = logits - stock_word_penalty * penalty_mask
            logits = _block_tokens(logits, ngram, history, tail_len, bad_word_ids)
            logits = _mask_ids(logits, blocked)
            if scratch is not None:
                logits = _mask_ids(logits, scratch.banned_markers())
            if do_sample:
                next_tok = _sample(logits, temperature, top_k, top_p)
            else:
                next_tok = logits.argmax(dim=-1)
            new_tokens = next_tok.tolist()
            if scratch is not None and new_tokens[0] == cfg.scratch_open_id:
                # cache.seq_len == seq.shape[1] here, so this is the state before the marker is fed.
                snapshots[seq.shape[1]] = (_cache_snapshot(cache), out)
            seq = torch.cat([seq, next_tok[:, None]], dim=1)
            for b, token in enumerate(new_tokens):
                history[b].append(token)
                if ngram is not None:
                    ngram[b].update(token)
            if eos_token_id is not None and bool((next_tok == eos_token_id).all()):
                break
            if scratch is not None:
                cut = scratch.feed(new_tokens[0])
                if cut is not None:
                    # Notes were removed from index `cut` onward. Everything before it is unchanged,
                    # so the cache is rolled back to the block's open marker and only the text after
                    # it is recomputed (the loop count is part of the cache and is kept).
                    snap, out_before = snapshots.pop(cut)
                    snapshots = {k: v for k, v in snapshots.items() if k in scratch.open_positions()}
                    _cache_restore(cache, snap)
                    seq = torch.tensor([scratch.out], dtype=seq.dtype, device=seq.device)
                    history[0] = list(scratch.out)
                    if ngram is not None:
                        ngram[0] = _make_ngram_index(no_repeat_ngram_size, scratch.out, repeat_window)
                    if seq.shape[1] > cut:
                        out = self(seq[:, cut:], past=cache, last_only=True)
                    else:
                        out = out_before
                    continue
                snapshots = {k: v for k, v in snapshots.items() if k in scratch.open_positions()}
            out = self(next_tok[:, None], past=cache, last_only=True)
        if not (report or display):
            return seq
        rep = text_report(seq[0, prompt_len:], bad_word_ids)
        if display:
            print_report(rep)
        return seq, rep


def _cache_snapshot(cache: InkyCache) -> dict:
    """Shallow copy of the cache state. The forward pass replaces entries and never edits them in place."""
    return {
        "pre": list(cache.pre),
        "post": list(cache.post),
        "core": list(cache.core),
        "prefix": list(cache.prefix),
        "seq_len": cache.seq_len,
    }


def _cache_restore(cache: InkyCache, snap: dict) -> None:
    cache.pre = list(snap["pre"])
    cache.post = list(snap["post"])
    cache.core = list(snap["core"])
    cache.prefix = list(snap["prefix"])
    cache.seq_len = snap["seq_len"]


def _blocked_ids(cfg: InkyConfig, explicit, eos_token_id: Optional[int]) -> List[int]:
    """Sorted token ids that generate must never emit (DATA-05).

    Default (``explicit`` None): the reserved special ids minus the scratch ids and minus the configured
    EOS id. An explicit list replaces the default. The EOS id (``eos_token_id`` and ``cfg.eos_token_id``)
    is always removed from the result, so the stop token stays allowed. Ids outside the vocabulary are
    dropped, since they cannot be indexed.
    """
    scratch = {cfg.scratch_open_id, cfg.scratch_sep_id, cfg.scratch_close_id}
    if explicit is None:
        ids = set(DEFAULT_RESERVED_IDS) - scratch - {cfg.eos_token_id}
    else:
        ids = {int(t) for t in explicit}
    ids.discard(cfg.eos_token_id)
    if eos_token_id is not None:
        ids.discard(int(eos_token_id))
    return sorted(t for t in ids if 0 <= t < cfg.vocab_size)


def _mask_ids(logits: torch.Tensor, ids) -> torch.Tensor:
    """Sets the given token ids to -inf (returns a copy; ``logits`` is not modified)."""
    if not ids:
        return logits
    logits = logits.clone()
    logits[:, sorted(ids)] = float("-inf")
    return logits


class _WindowedNgramIndex:
    """No-repeat index that counts only n-grams lying fully inside the last ``window`` tokens.

    Same answers as ``NgramBanIndex`` when the window covers the whole sequence. Grams that leave
    the window are removed as new tokens arrive, so the index stays O(window) in size.
    """

    def __init__(self, n: int, ids: Iterable[int] = (), window: Optional[int] = None):
        if window is not None and window < 1:
            raise ValueError(f"window must be None or >= 1, got {window}")
        self.n = n
        self.window = window
        self._seq: List[int] = []
        # (n-1)-token prefix -> {next token: number of windowed occurrences}
        self._next: Dict[Tuple[int, ...], Dict[int, int]] = {}
        for token in ids:
            self.update(token)

    def __len__(self) -> int:
        return len(self._seq)

    def _gram_at(self, start: int) -> Tuple[Tuple[int, ...], int]:
        n = self.n
        return tuple(self._seq[start:start + n - 1]), self._seq[start + n - 1]

    def _add(self, start: int) -> None:
        key, nxt = self._gram_at(start)
        counts = self._next.setdefault(key, {})
        counts[nxt] = counts.get(nxt, 0) + 1

    def _remove(self, start: int) -> None:
        key, nxt = self._gram_at(start)
        counts = self._next[key]
        counts[nxt] -= 1
        if counts[nxt] == 0:
            del counts[nxt]
        if not counts:
            del self._next[key]

    def update(self, token: int) -> None:
        self._seq.append(token)
        n, w, L = self.n, self.window, len(self._seq)
        if n < 2 or (w is not None and w < n):
            return
        if L >= n:
            self._add(L - n)                     # the gram that this token completes
        if w is not None and L - w - 1 >= 0:
            self._remove(L - w - 1)              # the gram that has just left the window

    def banned(self) -> Set[int]:
        n = self.n
        if n <= 0 or len(self._seq) < n:
            return set()
        if n == 1:
            return set(self._seq if self.window is None else self._seq[-self.window:])
        tail = tuple(self._seq[-(n - 1):])
        return set(self._next.get(tail, {}).keys())


def _make_ngram_index(n: int, ids: Iterable[int], window: Optional[int]):
    if window is None:
        return NgramBanIndex(n, ids)
    return _WindowedNgramIndex(n, ids, window)


def _block_tokens(
    logits: torch.Tensor,
    ngram: Optional[List[NgramBanIndex]],
    history: List[List[int]],
    tail_len: int,
    bad_word_ids: Optional[Sequence[Sequence[int]]],
) -> torch.Tensor:
    """Sets banned next tokens to -inf. ``ngram[b]`` is row b's incremental no-repeat index (None if off).

    Bad-word sequences only look at the last ``tail_len`` tokens of a row (the longest prefix they need),
    so the row history is sliced to that tail rather than scanned in full.
    """
    if ngram is None and not bad_word_ids:
        return logits
    logits = logits.clone()
    for b in range(logits.shape[0]):
        banned = ngram[b].banned() if ngram is not None else set()
        if bad_word_ids:
            tail = history[b][-tail_len:] if tail_len > 0 else []
            banned |= banned_by_bad_words(tail, bad_word_ids)
        if banned:
            logits[b, sorted(banned)] = float("-inf")
    return logits


def _apply_repetition_penalty(
    logits: torch.Tensor, seq: torch.Tensor, penalty: float, window: Optional[int] = None
) -> torch.Tensor:
    """Penalises every token seen in ``seq`` (or only the last ``window`` tokens when given)."""
    if penalty == 1.0:
        return logits
    ctx = seq if window is None else seq[:, -window:]
    picked = torch.gather(logits, 1, ctx)
    picked = torch.where(picked < 0, picked * penalty, picked / penalty)
    return logits.scatter(1, ctx, picked)


def _candidate_distribution(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Kept candidates and their probabilities: ``(indices [B, K], probs [B, K])``.

    With ``top_k`` set, only the k largest logits are read (torch.topk), and top-p works inside
    those k values, so the full vocabulary is never sorted (GEN-05). Without ``top_k`` and with
    ``top_p < 1``, the full vocabulary is sorted as before. Ties at the k-th logit keep exactly k
    tokens here, where the old mask kept every tie.
    """
    logits = logits / max(temperature, 1e-5)
    V = logits.size(-1)
    if top_k and 0 < top_k < V:
        vals, idx = torch.topk(logits, top_k, dim=-1)          # sorted, largest first
    elif top_p < 1.0:
        vals, idx = torch.sort(logits, descending=True, dim=-1)
    else:
        vals = logits
        idx = torch.arange(V, device=logits.device).unsqueeze(0).expand(logits.shape[0], V)
    if top_p < 1.0:
        cum = torch.softmax(vals, dim=-1).cumsum(dim=-1)
        remove = cum > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        vals = vals.masked_fill(remove, float("-inf"))
    return idx, torch.softmax(vals, dim=-1)


def _sample(logits: torch.Tensor, temperature: float, top_k: int, top_p: float) -> torch.Tensor:
    idx, probs = _candidate_distribution(logits, temperature, top_k, top_p)
    pick = torch.multinomial(probs, num_samples=1)
    return idx.gather(-1, pick).squeeze(-1)
