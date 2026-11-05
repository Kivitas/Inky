"""Configuration for Inky: a ~50M-parameter recurrent language model for creative writing.

Inky keeps the language-model path of EOS (parameter-shared recurrent core, latent-KV
attention, shared-expert sparse MoE, SSM memory highway, PonderNet halting) at a smaller scale.
The Rhen decision layer is ported as Arlo (inky/arlo.py) and is built when ``enable_arlo`` is set.

Inky also adds PENCIL-style scratch reduction (arXiv:2503.14337): the model may draft a
scratch block (scratch_open ... scratch_sep ... scratch_close), and the reduction step removes
the block from context, keeping only the text that follows it. The three scratch ids sit after
the 50,000 tokens of EOS_Tokenizer and must be added to that tokenizer as special tokens.
"""

from transformers import PretrainedConfig


class InkyConfig(PretrainedConfig):
    model_type = "inky"

    def __init__(
        self,
        vocab_size: int = 50003,           # 50,000 EOS_Tokenizer ids + 3 scratch special tokens
        scratch_open_id: int = 50000,      # opens a scratch block (PENCIL [CALL])
        scratch_sep_id: int = 50001,       # ends the scratch notes (PENCIL [SEP])
        scratch_close_id: int = 50002,     # closes the block and triggers reduction (PENCIL [RETURN])
        hidden_size: int = 384,
        # Attention (Multi-Latent Attention with decoupled RoPE)
        num_attention_heads: int = 6,
        qk_nope_head_dim: int = 48,
        qk_rope_head_dim: int = 16,
        v_head_dim: int = 64,
        q_lora_rank: int = 192,
        kv_lora_rank: int = 128,
        # Opt-in per-head RMSNorm on the query and key parts (nope and rope) before the dot product.
        # False keeps the architecture and parameter count unchanged. Cost when True: the key norm is
        # nonlinear, so the absorbed cached-decode path can no longer score against the latent cache
        # directly. It up-projects the WHOLE latent cache to per-head k_nope ([B, H, T, nope]) every
        # decode step, which costs O(T * H * nope * kv_rank) per step and a transient [B, H, T, nope]
        # tensor. That removes most of the decode-time benefit of absorption for long contexts.
        qk_norm: bool = False,
        # Which QK-norm to use when qk_norm is True. "head" (the default, unchanged behaviour): per-head
        # norm of q_nope, q_pe, k_nope, k_pe; the k_nope norm blocks absorption, as described above.
        # "latent": no k_nope norm (the latent c_kv is already RMS-normed); q_nope, q_pe and k_pe are
        # normed. The key up-projection can then still be folded into the query, so the cached decode
        # path stays cheap and matches the full path exactly. Ignored when qk_norm is False.
        qk_norm_mode: str = "head",
        # Phase B comparison arms (all default to the existing architecture; none has been trained or compared).
        # attention_type: "mla" (multi-latent attention above) or "gqa" (plain grouped-query attention with
        # num_kv_heads key/value heads of width qk_nope_head_dim + qk_rope_head_dim and v_head_dim, RoPE on the
        # last qk_rope_head_dim dims). At this size MLA's cache saving is ~30 MiB at 8k, while its absorbed decode
        # costs ~2x the attention FLOPs of plain heads. GQA supports the cache, packing and masks, but not
        # routing_mode="mor" or forward_early_exit (those need the latent layout). num_kv_heads is ignored for "mla".
        attention_type: str = "mla",
        num_kv_heads: int = 2,
        # attention_gate: a sigmoid gate per head, computed from the block input, multiplies the attention output
        # before the output projection (the gated-attention design of the Qwen team, reported at 1.7B and 15B).
        attention_gate: bool = False,
        # Phase D: local attention in some loops. A query in a loop listed in ``local_window_loops`` (1-based loop
        # indices) attends only to the last ``local_window`` positions, itself included; the other loops, and the
        # pre and post layers, stay global. The weights are shared across loops, so this only changes the mask.
        # Cached decode then reads ``local_window`` cached positions instead of all of them in those loops (the
        # cache itself still keeps every position). At 8k context the 14 global attention applications are about
        # three quarters of the decode FLOPs. Needs attention_type "mla" or "gqa" without routing_mode="mor", and
        # is rejected by forward_early_exit. () or local_window=None keeps every loop global. A selective-SSM layer
        # (ssm_mixer_layers) ignores the window: its state already decays.
        local_window: int | None = None,
        local_window_loops: tuple = (),
        # Recurrent depth: pre layers -> recurrent core (shared over loops) -> post layers
        num_pre_layers: int = 1,
        num_recurrent_layers: int = 2,
        num_post_layers: int = 1,
        max_recurrent_loops: int = 6,      # effective depth 1 + 2 * 6 + 1 = 14 layer applications
        min_recurrent_loops: int = 2,      # lower bound of the loop count sampled in training
        loop_dropout_prob: float = 0.1,
        use_step_modulation: bool = True,  # FiLM scale/shift per loop step
        ssm_state_dim: int = 64,           # State-space memory highway width
        # PonderNet halting (per token). The stop probability at each loop is learned; the halting
        # distribution is pulled towards a geometric prior with mean ponder_prior_steps loops.
        ponder_prior_steps: float = 3.0,   # expected loops under the prior (lambda_prior = 1 / this)
        ponder_kl_coef: float = 1e-3,      # weight of KL(halting || prior)
        ponder_step_loss_coef: float = 1.0,  # weight of the per-loop LM loss, weighted by halting probability
        # Depth routing. "ponder" (the default) is the PonderNet path above: every token runs every loop and
        # the output is a halting-weighted mix. "mor" is opt-in Mixture-of-Recursions routing (arXiv
        # 2507.10524, token-choice variant): a router assigns each token a depth d in
        # [min_recurrent_loops, max_recurrent_loops] from that token's own state (causal), loop n runs only
        # the tokens with d >= n (attention queries, key/value projections and the MoE run on the active
        # rows only, so training compute really drops), an exited token keeps its state and its last
        # latents (stale key/value reuse), and the output is the state after each token's last active loop.
        # Under "mor" the PonderNet halting gate, its KL and the per-loop CE (step_ce) are not used, and
        # num_loops is a cap on the depth. See layers.MoRRouter and InkyModel.forward for the semantics.
        # routing_mode="last" is the third option: every token runs every loop, there is no halting gate, and the
        # output is the last loop's state (training still draws the loop count in [min, max]; eval uses the max).
        # No PonderNet KL and no per-loop loss (step_ce and ponder_kl are reported as 0), as in recurrent-depth models
        # that train on random depth only.
        routing_mode: str = "ponder",
        # loop_input_injection: each loop's block input also gets gain * (the prelude output), with a per-channel
        # gain that starts at 0, so the model starts identical to one without it. The simplest form of the
        # prelude re-injection used by recurrent-depth models.
        loop_input_injection: bool = False,
        mor_router_hidden: int | None = None,   # None: the router is one Linear(hidden, depths); int: MLP width
        mor_capacity_aux_coef: float = 1e-2,    # weight of the depth-balance loss ((E[depth] - target)^2)
        mor_gate_scale: bool = True,            # scale an active token's update by its router probability
        mor_temperature: float = 1.0,           # softmax / Gumbel temperature of the depth router
        mor_target_depth: float | None = None,  # target mean depth; None = ponder_prior_steps
        mor_sample_depth: bool = True,          # train: Gumbel-max sampled depth (False: argmax, as in eval)
        mor_gather: bool = True,                # False: masked reference path (all rows computed, same result)
        # Fraction of target positions used by the per-loop LM loss each step (1.0 = every position).
        # The subset is drawn once per forward pass and shared by all loops, and the loss is scaled so its
        # expectation matches the full version. Lower values save a full-vocabulary projection per loop.
        step_ce_token_fraction: float = 0.25,
        # Shared-expert fine-grained MoE
        intermediate_size: int = 704,      # per-expert SwiGLU width
        # Width of the always-on shared expert(s); None = intermediate_size. With a separate width, "fewer, larger
        # routed experts" fits the same budget: shared 704 + 4 routed x 1408 with top-1 has the same active width
        # (2,112) and the same parameters per block as shared 704 + 8 routed x 704 with top-2.
        shared_intermediate_size: int | None = None,
        num_shared_experts: int = 1,
        num_routed_experts: int = 8,
        num_active_routed: int = 2,
        router_z_loss_coef: float = 1e-3,
        balance_rate: float = 1e-3,        # aux-loss-free congestion-bias rate
        # How the congestion bias moves each optimizer step. "sign": every bias moves by +-balance_rate towards
        # the uniform load (the DeepSeek-V3 rule, so a step of 1e-3 can overcome a score gap of ~1e-2 in tens of
        # steps). "proportional": the step is balance_rate * (uniform - load), at most ~1/num_routed_experts of
        # that, which is the original Inky rule and is much slower to remove an early imbalance.
        balance_update: str = "sign",
        # MoE dispatch: on a non-CPU device, batches of at most this many tokens run every routed expert
        # on every token (sync-free, no host read of expert counts; ~num_routed/top_k times the routed
        # compute). Larger batches (training, prefill) use the grouped path. 0 disables the dense path.
        moe_dense_max_tokens: int = 64,
        # Ablation switches. use_moe=False replaces every MoE feed-forward with ONE dense SwiGLU whose width
        # is dense_intermediate_size; None picks (num_shared_experts + top-k routed) * intermediate_size,
        # which equals the MoE's ACTIVE feed-forward parameters per token (the router's num_routed * hidden
        # weights are not counted). use_highway=False does not create the StateSpaceMemoryHighway at all.
        # Both default to True, which is the standard architecture (50,219,202 parameters).
        use_moe: bool = True,
        dense_intermediate_size: int | None = None,
        use_highway: bool = True,
        # Initial value of the learned gain on the highway's output (``h + gain * RMSNorm(out_proj(mem))``). The
        # RMSNorm output has unit scale whatever the stream's scale is, and the token embeddings start at RMS 0.02,
        # so a fixed 0.1 made the highway about as large as the stream itself at initialisation (125% of the loop-1
        # input, measured). 0.0 (default) starts the model without the highway and lets the gain grow if the
        # highway helps; 0.1 reproduces the earlier fixed scale. Ignored when use_highway is False.
        highway_gain_init: float = 0.0,
        # Loop mixture. The PonderNet output is sum_n p_n * h_n over loop states whose scale grows with n (RMS
        # 0.06 -> 0.28 over six loops at init), so p_n alone does not say how much loop n contributes. True mixes
        # the directions h_n / rms(h_n) with p_n and multiplies by the p-weighted mean RMS, so each loop's share
        # is p_n and the output keeps the stream's scale. False is the plain sum_n p_n * h_n.
        loop_mix_normalize: bool = True,
        # Context and numerics
        max_position_embeddings: int = 8192,
        rope_theta: float = 500000.0,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        final_logit_softcapping: float = 30.0,
        z_loss_coef: float = 1e-4,
        gradient_checkpointing: bool = False,  # recompute block activations in backward (training)
        # Recompute the LM head in backward for each loss chunk (CORE-08). True keeps the memory of
        # one chunk of logits at a time, at the cost of a second head pass per step. False keeps the
        # chunk logits for backward: one head pass, but about one fp32 copy of N x vocab per chunk.
        lm_loss_recompute: bool = True,
        # Writing decision engine (Rhen's Choice and Noul primitives). Reads hidden states only.
        enable_arlo: bool = False,         # off by default: Arlo is not trained by train.py yet
        arlo_slots: int = 8,               # K latent decision slots
        arlo_dim: int = 128,               # slot width (Rhen rhen_hidden_dim)
        arlo_heads: int = 4,               # pointer heads
        arlo_pointer_rank: int = 64,
        arlo_refine_steps: int = 2,        # slot refinement passes (gated update)
        eos_token_id: int = 49974,         # <|endoftext|> in EOS_Tokenizer
        pad_token_id: int = 49974,
        ssm_mixer_layers: tuple = (),      # recurrent-core layer indices using SelectiveSSMMixer, not attention
        # Dropout (TRAIN-04). Both are active only in train mode and both default to 0 (no dropout).
        # embedding_dropout drops token-embedding entries once per forward pass. residual_dropout drops
        # each block's attention and MoE branch outputs before they are added to the residual stream.
        embedding_dropout: float = 0.0,
        residual_dropout: float = 0.0,
        **kwargs,
    ):
        for name, value in (("embedding_dropout", embedding_dropout), ("residual_dropout", residual_dropout)):
            if not 0.0 <= float(value) < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {value}")
        self.embedding_dropout = float(embedding_dropout)
        self.residual_dropout = float(residual_dropout)
        self.gradient_checkpointing = gradient_checkpointing
        self.lm_loss_recompute = bool(lm_loss_recompute)
        self.enable_arlo = enable_arlo
        self.arlo_slots = arlo_slots
        self.arlo_dim = arlo_dim
        self.arlo_heads = arlo_heads
        self.arlo_pointer_rank = arlo_pointer_rank
        self.arlo_refine_steps = arlo_refine_steps
        self.vocab_size = vocab_size
        self.scratch_open_id = scratch_open_id
        self.scratch_sep_id = scratch_sep_id
        self.scratch_close_id = scratch_close_id
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_norm = bool(qk_norm)
        if qk_norm_mode not in ("head", "latent"):
            raise ValueError(f"qk_norm_mode must be 'head' or 'latent', got {qk_norm_mode!r}")
        self.qk_norm_mode = qk_norm_mode
        if attention_type not in ("mla", "gqa"):
            raise ValueError(f"attention_type must be 'mla' or 'gqa', got {attention_type!r}")
        self.attention_type = attention_type
        if int(num_kv_heads) < 1:
            raise ValueError(f"num_kv_heads must be >= 1, got {num_kv_heads}")
        self.num_kv_heads = int(num_kv_heads)
        if attention_type == "gqa" and num_attention_heads % self.num_kv_heads != 0:
            raise ValueError(
                f"attention_type='gqa' needs num_attention_heads ({num_attention_heads}) divisible by "
                f"num_kv_heads ({self.num_kv_heads})"
            )
        self.attention_gate = bool(attention_gate)
        self.loop_input_injection = bool(loop_input_injection)
        if local_window is not None:
            local_window = int(local_window)
            if local_window < 1:
                raise ValueError(f"local_window must be None or >= 1, got {local_window}")
        self.local_window = local_window
        self.local_window_loops = tuple(sorted({int(i) for i in local_window_loops}))
        if self.local_window_loops:
            if local_window is None:
                raise ValueError("local_window_loops needs local_window to be set")
            bad = [i for i in self.local_window_loops if not 1 <= i <= max_recurrent_loops]
            if bad:
                raise ValueError(f"local_window_loops {bad} out of range 1..{max_recurrent_loops}")
            if routing_mode == "mor":
                raise ValueError("local_window_loops is not supported with routing_mode='mor'")
        self.num_pre_layers = num_pre_layers
        self.num_recurrent_layers = num_recurrent_layers
        self.num_post_layers = num_post_layers
        self.max_recurrent_loops = max_recurrent_loops
        self.min_recurrent_loops = min_recurrent_loops
        self.loop_dropout_prob = loop_dropout_prob
        self.use_step_modulation = use_step_modulation
        self.ssm_state_dim = ssm_state_dim
        self.ponder_prior_steps = ponder_prior_steps
        self.ponder_kl_coef = ponder_kl_coef
        self.ponder_step_loss_coef = ponder_step_loss_coef
        if routing_mode not in ("ponder", "mor", "last"):
            raise ValueError(f"routing_mode must be 'ponder', 'mor' or 'last', got {routing_mode!r}")
        if routing_mode == "mor" and attention_type != "mla":
            raise ValueError("routing_mode='mor' needs attention_type='mla' (it gathers latent keys and values)")
        self.routing_mode = routing_mode
        if mor_router_hidden is not None:
            mor_router_hidden = int(mor_router_hidden)
            if mor_router_hidden < 1:
                raise ValueError(f"mor_router_hidden must be None or >= 1, got {mor_router_hidden}")
        self.mor_router_hidden = mor_router_hidden
        if float(mor_capacity_aux_coef) < 0.0:
            raise ValueError(f"mor_capacity_aux_coef must be >= 0, got {mor_capacity_aux_coef}")
        self.mor_capacity_aux_coef = float(mor_capacity_aux_coef)
        self.mor_gate_scale = bool(mor_gate_scale)
        if not float(mor_temperature) > 0.0:
            raise ValueError(f"mor_temperature must be > 0, got {mor_temperature}")
        self.mor_temperature = float(mor_temperature)
        if mor_target_depth is not None:
            mor_target_depth = float(mor_target_depth)
            if mor_target_depth < 1.0:
                raise ValueError(f"mor_target_depth must be None or >= 1, got {mor_target_depth}")
        self.mor_target_depth = mor_target_depth
        self.mor_sample_depth = bool(mor_sample_depth)
        self.mor_gather = bool(mor_gather)
        if routing_mode == "mor":
            if min_recurrent_loops < 1 or max_recurrent_loops < min_recurrent_loops:
                raise ValueError(
                    "routing_mode='mor' needs 1 <= min_recurrent_loops <= max_recurrent_loops, got "
                    f"{min_recurrent_loops} and {max_recurrent_loops}"
                )
            if tuple(ssm_mixer_layers):
                raise ValueError(
                    "routing_mode='mor' does not support ssm_mixer_layers: a selective-SSM mixer carries a "
                    "state across every token, so it cannot skip exited tokens"
                )
        if not 0.0 < float(step_ce_token_fraction) <= 1.0:
            raise ValueError(f"step_ce_token_fraction must be in (0, 1], got {step_ce_token_fraction}")
        self.step_ce_token_fraction = float(step_ce_token_fraction)
        self.intermediate_size = intermediate_size
        if shared_intermediate_size is not None:
            shared_intermediate_size = int(shared_intermediate_size)
            if shared_intermediate_size < 1:
                raise ValueError(f"shared_intermediate_size must be None or >= 1, got {shared_intermediate_size}")
        self.shared_intermediate_size = shared_intermediate_size
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.num_active_routed = num_active_routed
        self.router_z_loss_coef = router_z_loss_coef
        self.balance_rate = balance_rate
        if balance_update not in ("sign", "proportional"):
            raise ValueError(f"balance_update must be 'sign' or 'proportional', got {balance_update!r}")
        self.balance_update = balance_update
        self.highway_gain_init = float(highway_gain_init)
        self.loop_mix_normalize = bool(loop_mix_normalize)
        if int(moe_dense_max_tokens) < 0:
            raise ValueError(f"moe_dense_max_tokens must be >= 0, got {moe_dense_max_tokens}")
        self.moe_dense_max_tokens = int(moe_dense_max_tokens)
        self.use_moe = bool(use_moe)
        self.use_highway = bool(use_highway)
        if dense_intermediate_size is not None:
            dense_intermediate_size = int(dense_intermediate_size)
            if dense_intermediate_size < 1:
                raise ValueError(f"dense_intermediate_size must be None or >= 1, got {dense_intermediate_size}")
        self.dense_intermediate_size = dense_intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.final_logit_softcapping = final_logit_softcapping
        self.z_loss_coef = z_loss_coef
        self.ssm_mixer_layers = tuple(int(i) for i in ssm_mixer_layers)
        bad = [i for i in self.ssm_mixer_layers if not 0 <= i < num_recurrent_layers]
        if bad:
            raise ValueError(
                f"ssm_mixer_layers {bad} out of range for {num_recurrent_layers} recurrent layers"
            )
        # The LM head is always tied to the input embedding (see InkyModel.forward).
        # from_pretrained passes the saved value back in, so drop it here to avoid a duplicate.
        kwargs.pop("tie_word_embeddings", None)
        super().__init__(
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=True,
            **kwargs,
        )

    @property
    def dense_ffn_size(self) -> int:
        """Width of the dense SwiGLU used when ``use_moe=False``.

        ``dense_intermediate_size`` if set, else the MoE's active width: the shared experts plus the top-k
        routed experts, each ``intermediate_size`` wide, so the dense FFN has the same number of
        feed-forward parameters per token as the MoE it replaces (router excluded).
        """
        if self.dense_intermediate_size is not None:
            return self.dense_intermediate_size
        top_k = min(self.num_active_routed, self.num_routed_experts)
        return self.num_shared_experts * self.shared_ffn_size + top_k * self.intermediate_size

    @property
    def shared_ffn_size(self) -> int:
        """Width of each shared expert: ``shared_intermediate_size`` if set, else ``intermediate_size``."""
        return self.intermediate_size if self.shared_intermediate_size is None else self.shared_intermediate_size

    @property
    def mor_target(self) -> float:
        """Target mean depth of the MoR router: ``mor_target_depth``, else ``ponder_prior_steps``, clipped
        into [min_recurrent_loops, max_recurrent_loops]."""
        target = self.ponder_prior_steps if self.mor_target_depth is None else self.mor_target_depth
        return float(min(max(float(target), self.min_recurrent_loops), self.max_recurrent_loops))

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim
