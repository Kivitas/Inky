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
        # Shared-expert fine-grained MoE
        intermediate_size: int = 704,      # per-expert SwiGLU width
        num_shared_experts: int = 1,
        num_routed_experts: int = 8,
        num_active_routed: int = 2,
        router_z_loss_coef: float = 1e-3,
        balance_rate: float = 1e-3,        # aux-loss-free congestion-bias rate
        # Context and numerics
        max_position_embeddings: int = 8192,
        rope_theta: float = 500000.0,
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        final_logit_softcapping: float = 30.0,
        z_loss_coef: float = 1e-4,
        gradient_checkpointing: bool = False,  # recompute block activations in backward (training)
        # Writing decision engine (Rhen's Choice and Noul primitives). Reads hidden states only.
        enable_arlo: bool = True,
        arlo_slots: int = 8,               # K latent decision slots
        arlo_dim: int = 128,               # slot width (Rhen rhen_hidden_dim)
        arlo_heads: int = 4,               # pointer heads
        arlo_pointer_rank: int = 64,
        arlo_refine_steps: int = 2,        # slot refinement passes (gated update)
        eos_token_id: int = 49974,         # <|endoftext|> in EOS_Tokenizer
        pad_token_id: int = 49974,
        ssm_mixer_layers: tuple = (),      # recurrent-core layer indices that use SelectiveSSMMixer instead of attention
        **kwargs,
    ):
        self.gradient_checkpointing = gradient_checkpointing
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
        self.intermediate_size = intermediate_size
        self.num_shared_experts = num_shared_experts
        self.num_routed_experts = num_routed_experts
        self.num_active_routed = num_active_routed
        self.router_z_loss_coef = router_z_loss_coef
        self.balance_rate = balance_rate
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
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim
