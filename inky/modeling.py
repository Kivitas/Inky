"""Inky language model: pre layers, a looped recurrent core with PonderNet halting, post layers."""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import PreTrainedModel

from .arlo import Arlo
from .cache import InkyCache, InkyOutput, _claim, _writable_past  # noqa: F401
from .config import REASONING_MODES, STYLE_MODES, InkyConfig, resolve_reasoning_loops
from .decision import InkyDecisionMixin
from .generation import (  # noqa: F401  (the generation helpers stay importable from here)
    _USE_CONFIG_EOS,
    InkyGenerationMixin,
    _apply_repetition_penalty,
    _blocked_ids,
    _cache_restore,
    _cache_snapshot,
    _candidate_distribution,
    _sample,
    _WindowedNgramIndex,
)
from .jepa import JEPAController
from .layers import (
    InkyBlock,
    ModeConditioner,
    MoRRouter,
    PonderHalting,
    RecurrentCore,
    RMSNorm,
    RotaryEmbedding,
    geometric_halting_weights,
    geometric_prior,
    pondernet_kl,
)
from .memory_bank import ProductKeyMemoryBank
from .scratch import DEFAULT_MAX_NOTE_TOKENS, ScratchTracker, reduce_scratch  # noqa: F401


class InkyModel(InkyGenerationMixin, InkyDecisionMixin, PreTrainedModel):
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
        # routing_mode='last' has no halting mixture, so it has no gate either.
        self.halting = None if config.routing_mode in ("mor", "last") else PonderHalting(
            config.hidden_size, config.ponder_prior_steps
        )
        self.arlo = Arlo(config) if config.enable_arlo else None
        self.mor_router = MoRRouter(config) if mor else None
        self.mode_cond = (
            ModeConditioner(config.hidden_size)
            if getattr(config, "use_mode_conditioning", False)
            else None
        )
        self.memory_bank = (
            ProductKeyMemoryBank(config)
            if getattr(config, "enable_memory_bank", False)
            else None
        )
        self.jepa = (
            JEPAController(config)
            if getattr(config, "enable_jepa", False)
            else None
        )
        self.post_init()
        # Start the halting gate at its prior and thought gate at neutral. Set after post_init.
        self.core.reset_thought_gate()
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
        if self.mode_cond is not None:
            parts["mode_conditioning"] = n(self.mode_cond)
        if self.memory_bank is not None:
            parts["memory_bank"] = n(self.memory_bank)
        if self.jepa is not None:
            parts["jepa_controller"] = n(self.jepa)
        parts["total"] = n(self)
        return parts

    def _choose_loops(
        self,
        cache: Optional[InkyCache],
        num_loops: Optional[int],
        generator: Optional[torch.Generator] = None,
        reasoning_mode: Optional[str] = None,
    ) -> int:
        eff_mode = reasoning_mode if reasoning_mode is not None else getattr(self.config, "reasoning_mode", None)
        resolved = resolve_reasoning_loops(eff_mode, num_loops)
        if cache is not None and cache.num_loops is not None:
            if resolved is not None and resolved != cache.num_loops:
                raise ValueError(
                    f"num_loops={resolved} differs from the {cache.num_loops} loops this cache was filled with"
                )
            return cache.num_loops
        if resolved is not None:
            return int(resolved)
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
        reasoning_mode: Optional[str] = None,
        style_mode: Optional[str] = None,
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

        eff_reasoning = reasoning_mode if reasoning_mode is not None else getattr(self.config, "reasoning_mode", None)
        eff_style = style_mode if style_mode is not None else getattr(self.config, "style_mode", None)
        if eff_reasoning is not None and eff_reasoning not in REASONING_MODES:
            raise ValueError(
                f"reasoning_mode must be one of {tuple(REASONING_MODES)} or None, got {eff_reasoning!r}"
            )
        if eff_style is not None and eff_style not in STYLE_MODES:
            raise ValueError(f"style_mode must be one of {STYLE_MODES} or None, got {eff_style!r}")
        if self.mode_cond is not None:
            h = self.mode_cond(h, reasoning_mode=eff_reasoning, style_mode=eff_style)

        loops = self._choose_loops(cache, num_loops, generator, reasoning_mode=eff_reasoning)
        if cache is not None and cache.num_loops is None:
            cache.num_loops = loops
            cache.core = [None] * loops
            cache.prefix = [None] * loops

        prior_steps = float(REASONING_MODES[eff_reasoning]) if eff_reasoning is not None else None
        conv_weight = 1.0 if getattr(self.config, "convergence_halting", False) else 0.0

        mor = self.config.routing_mode == "mor"
        last_mode = self.config.routing_mode == "last"
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
            # routing_mode='last' runs the same loops but has no gate: the output is the last loop's state.
            h_loop, prev_mem = h, None
            lambdas: List[torch.Tensor] = []
            loop_states: List[torch.Tensor] = []
            jepa_z_preds: List[torch.Tensor] = []
            core_aux = h.new_zeros((), dtype=torch.float32)   # summed over loops, averaged below

            for step in range(1, loops + 1):
                idx = step - 1
                jepa_ctrl = (
                    self.jepa(h_loop, step=step, h_goal_source=h)
                    if self.jepa is not None
                    else None
                )
                if jepa_ctrl is not None:
                    jepa_z_preds.append(jepa_ctrl.z_pred)
                    expert_bias = jepa_ctrl.moe_bias
                    fetch_gate = jepa_ctrl.fetch_gate
                else:
                    expert_bias = None
                    fetch_gate = None

                if self.memory_bank is not None:
                    u_t, _ = self.memory_bank(
                        h_loop, step=step, input_ids=input_ids, fetch_gate=fetch_gate
                    )
                else:
                    u_t = h_loop

                h_next, mem, new_prefix, presents, aux = self.core(
                    u_t,
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
                    h0=h,                                  # the prelude output (used by loop_input_injection)
                    expert_bias=expert_bias,
                )
                if cache is not None:
                    cache.core[idx] = [_claim(e, cache) for e in presents]
                    cache.prefix[idx] = new_prefix
                prev_mem = mem
                core_aux = core_aux + aux

                if not last_mode:
                    lam = self.halting(
                        h_next,
                        prev_h=h_loop,
                        prior_steps=prior_steps,
                        convergence_weight=conv_weight,
                    )
                    if jepa_ctrl is not None:
                        lam = 0.5 * (lam + jepa_ctrl.p_halt.to(lam.dtype))
                    lambdas.append(lam)
                    loop_states.append(h_next)
                h_loop = h_next
            # The recurrent layers run once per loop. Averaging keeps the aux at the scale of one pass, so
            # it does not grow with the loop count. Each layer's own term is still its per-loop mean.
            moe_aux = moe_aux + core_aux / loops

            if last_mode:
                h_out = h_loop.float()
                per_token_loops = torch.full((B, S), float(loops), device=h.device)
            else:
                weights = geometric_halting_weights(lambdas)             # [L, B, S], sums to 1 over L
                states = torch.stack(loop_states, dim=0).float()         # [L, B, S, D]
                h_out = self._mix_loop_states(states, weights)           # [B, S, D]
                loop_index = torch.arange(1, loops + 1, device=h.device, dtype=torch.float32).view(-1, 1, 1)
                per_token_loops = (weights * loop_index).sum(dim=0)       # [B, S] expected loops per token
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
        elif last_mode:
            extra = dict(loop_applications=B * S * loops, full_loop_applications=B * S * loops)

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

        if labels is not None and last_mode:
            ce, z = self._lm_loss(h, labels)
            loss = ce + self.config.z_loss_coef * z + moe_aux
            zero = ce.detach() * 0.0
            parts = {
                "ce": ce.detach(),
                "step_ce": zero,
                "z_loss": z.detach(),
                "moe_aux": moe_aux.detach(),
                "ponder_kl": zero,
            }
            return InkyOutput(loss=loss, past=cache, moe_aux_loss=moe_aux, mean_loops=mean_loops,
                              loss_parts=parts)

        if labels is not None:
            ce, z = self._lm_loss(h, labels)
            step_ce = self._weighted_step_ce(loop_states, weights, labels, generator=generator)
            eff_lam = (1.0 / prior_steps) if prior_steps is not None else self.halting.prior_lambda
            prior = geometric_prior(loops, eff_lam, device=h.device)
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
            if self.jepa is not None and len(loop_states) >= 2:
                j_loss = self.jepa.jepa_loss(jepa_z_preds[:-1], loop_states[1:])
                loss = loss + float(getattr(self.config, "jepa_loss_coef", 0.1)) * j_loss
                parts["jepa_loss"] = j_loss.detach()
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
        reasoning_mode: Optional[str] = None,
        style_mode: Optional[str] = None,
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
        eff_reasoning = reasoning_mode if reasoning_mode is not None else getattr(self.config, "reasoning_mode", None)
        eff_style = style_mode if style_mode is not None else getattr(self.config, "style_mode", None)
        if eff_reasoning is not None and eff_reasoning not in REASONING_MODES:
            raise ValueError(
                f"reasoning_mode must be one of {tuple(REASONING_MODES)} or None, got {eff_reasoning!r}"
            )
        if eff_style is not None and eff_style not in STYLE_MODES:
            raise ValueError(f"style_mode must be one of {STYLE_MODES} or None, got {eff_style!r}")
        resolved_loops = resolve_reasoning_loops(eff_reasoning, num_loops)
        if self.config.routing_mode in ("mor", "last"):
            if halt_threshold is not None:
                raise ValueError(
                    f"halt_threshold is not used with routing_mode={self.config.routing_mode!r}: "
                    + ("a token exits at its router depth" if self.config.routing_mode == "mor"
                       else "there is no halting gate, every token runs every loop")
                )
            self.eval()
            L = int(resolved_loops) if resolved_loops is not None else self.config.max_recurrent_loops
            out = self.forward(
                input_ids, num_loops=L, return_hidden=True, reasoning_mode=eff_reasoning, style_mode=eff_style
            )
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
        if self.config.attention_type != "mla":
            raise NotImplementedError("forward_early_exit needs attention_type='mla' (it reuses latent keys)")
        if self.config.local_window_loops:
            raise NotImplementedError("forward_early_exit does not support local_window_loops (global keys only)")
        if halt_threshold is not None and not 0.0 < halt_threshold <= 1.0:
            raise ValueError(f"halt_threshold must be in (0, 1], got {halt_threshold}")
        S = input_ids.shape[1]
        if S > self.config.max_position_embeddings:
            raise ValueError(f"sequence length {S} exceeds max_position_embeddings")
        self.eval()
        cfg = self.config
        L = int(resolved_loops) if resolved_loops is not None else cfg.max_recurrent_loops
        prior_steps = float(REASONING_MODES[eff_reasoning]) if eff_reasoning is not None else None
        conv_weight = 1.0 if getattr(cfg, "convergence_halting", False) else 0.0

        cos, sin = self.rope(0, S)
        h = self.embed_tokens(input_ids)
        for layer in self.pre_layers:
            h, _, _ = layer(h, cos, sin, None)
        if self.mode_cond is not None:
            h = self.mode_cond(h, reasoning_mode=eff_reasoning, style_mode=eff_style)

        h0 = h                                                       # the prelude output (loop_input_injection)
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
            out_act, ssm_mem = self.core.forward_subset(state, n, cos, sin, active, stores, ssm_mem, h0=h0)
            computed += int(active.numel())
            lam = self.halting(
                out_act,
                prev_h=state[:, active],
                prior_steps=prior_steps,
                convergence_weight=conv_weight,
            ).squeeze(0)                    # [K]
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
