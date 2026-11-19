"""Comprehensive tests for PKM Memory Bank, JEPA Executive Controller, Continual GRPO RL,
GatedDeltaNet hybrid linear mixers, and end-to-end ultra_sparse_jepa InkyModel integration."""

import torch
import torch.nn.functional as F

from inky import (
    REASONING_MODES,
    REASONING_SCRATCH_BUDGETS,
    ContinualGRPOTrainer,
    GatedDeltaNetMixer,
    InkyConfig,
    InkyModel,
    JEPAController,
    ProductKeyMemoryBank,
    QuantizedValueTable,
    UserPreferenceProfile,
    compute_group_relative_advantages,
    grpo_loss,
    resolve_reasoning_scratch_budget,
    sequence_log_probs,
)


def _tiny_ultra_config(**overrides) -> InkyConfig:
    defaults = dict(
        vocab_size=256,
        scratch_open_id=253,
        scratch_sep_id=254,
        scratch_close_id=255,
        eos_token_id=252,
        pad_token_id=252,
        hidden_size=32,
        num_attention_heads=4,
        qk_nope_head_dim=8,
        qk_rope_head_dim=4,
        v_head_dim=8,
        q_lora_rank=16,
        kv_lora_rank=16,
        num_pre_layers=1,
        num_recurrent_layers=4,
        num_post_layers=1,
        ssm_mixer_layers=(0, 1, 2),
        linear_mixer_type="gated_deltanet",
        ssm_state_dim=8,
        intermediate_size=32,
        num_shared_experts=1,
        num_routed_experts=4,
        num_active_routed=2,
        pkm_sub_keys=8,
        pkm_key_dim=16,
        pkm_mem_dim=16,
        pkm_top_k=2,
        pkm_query_steps=(1, 3, 5),
        pkm_storage_dtype="int8",
        jepa_latent_dim=16,
        jepa_horizon=3,
        jepa_num_actions=4,
        max_position_embeddings=128,
    )
    defaults.update(overrides)
    return InkyConfig.ultra_sparse_jepa(**defaults)


def test_pkm_memory_bank_sublinear_and_quantized_tables() -> None:
    torch.manual_seed(42)

    # 1. QuantizedValueTable in float32, int8, and int4 modes
    num_slots, mem_dim = 64, 16
    write_idx = torch.tensor([0, 7, 19, 63], dtype=torch.long)
    write_vals = torch.randn(4, mem_dim, dtype=torch.float32)

    for dtype_mode, tol in (("float32", 1e-6), ("int8", 0.05), ("int4", 0.35)):
        table = QuantizedValueTable(
            num_slots=num_slots,
            mem_dim=mem_dim,
            storage_dtype=dtype_mode,
            init_std=0.01,
        )
        table.write_slots(write_idx, write_vals)
        fetched = table.fetch(write_idx.view(2, 2), dtype=torch.float32)
        assert fetched.shape == (2, 2, mem_dim)
        assert torch.allclose(fetched.reshape(4, mem_dim), write_vals, atol=tol)
        assert table.storage_bytes > 0

    # 2. ProductKeyMemoryBank Cartesian key decomposition (M = S^2) & exact Top-K over K^2
    s_keys, key_dim, top_k = 12, 16, 4
    bank = ProductKeyMemoryBank(
        hidden_size=32,
        sub_keys=s_keys,
        key_dim=key_dim,
        mem_dim=16,
        top_k=top_k,
        query_steps=(1, 3),
        routing_mode="pkm",
        storage_dtype="int8",
    )
    assert bank.num_slots == s_keys * s_keys

    h = torch.randn(2, 5, 32, requires_grad=True)
    h_norm = bank.norm(h)
    top_indices, top_scores = bank.pkm_route(h_norm)
    assert top_indices.shape == (2, 5, top_k)
    assert top_scores.shape == (2, 5, top_k)

    # Verify pkm_route selects exact global Top-K over all M = S^2 Cartesian key pairs
    with torch.no_grad():
        q = bank.q_proj(h_norm)
        q1, q2 = q.chunk(2, dim=-1)
        s1 = F.linear(q1, bank.c1)  # [B, L, S]
        s2 = F.linear(q2, bank.c2)  # [B, L, S]
        full_scores = (s1.unsqueeze(-1) + s2.unsqueeze(-2)).reshape(2, 5, s_keys * s_keys)
        exact_scores, exact_indices = torch.topk(full_scores, k=top_k, dim=-1)
        assert torch.allclose(top_scores, exact_scores, atol=1e-5)
        assert torch.equal(top_indices, exact_indices)

    # 3. O(1) deterministic multi-head rolling n-gram hash_route with non-negative indices
    input_ids = torch.randint(0, 200, (2, 5), dtype=torch.long)
    hash_idx, hash_scores = bank.hash_route(input_ids)
    assert hash_idx.shape == (2, 5, top_k)
    assert hash_scores.shape == (2, 5, top_k)
    assert bool((hash_idx >= 0).all() and (hash_idx < bank.num_slots).all())

    # 4. Gated residual injection u_t = h_t + g_t * m_hat_t on query_steps only
    fetch_gate = torch.ones(2, 5)
    u_queried, info_q = bank(h, step=1, input_ids=input_ids, fetch_gate=fetch_gate)
    assert info_q["queried"] is True
    assert info_q["indices"] is not None
    assert u_queried.shape == h.shape
    assert not torch.allclose(u_queried, h)

    u_skipped, info_skip = bank(h, step=2, input_ids=input_ids, fetch_gate=fetch_gate)
    assert info_skip["queried"] is False
    assert info_skip["indices"] is None
    assert torch.equal(u_skipped, h)

    # Also verify hash routing mode forward pass
    hash_bank = ProductKeyMemoryBank(
        hidden_size=32,
        sub_keys=s_keys,
        key_dim=key_dim,
        mem_dim=16,
        top_k=top_k,
        query_steps=(1,),
        routing_mode="hash",
        storage_dtype="int4",
    )
    u_hash, info_hash = hash_bank(h, step=1, input_ids=input_ids)
    assert info_hash["queried"] is True
    assert u_hash.shape == h.shape


def test_jepa_executive_controller_rollout_critic_and_loss() -> None:
    torch.manual_seed(7)
    b, s, d_model, d_z, num_experts = 2, 6, 32, 16, 8
    controller = JEPAController(
        hidden_size=d_model,
        latent_dim=d_z,
        action_dim=8,
        num_actions=4,
        horizon=3,
        num_routed_experts=num_experts,
        convergence_eps=0.05,
        mem_threshold=0.5,
        lambda_compute=0.05,
    )
    controller.train()

    h_0 = torch.randn(b, s, d_model, requires_grad=True)
    h_t = torch.randn(b, s, d_model, requires_grad=True)

    # Verify E_phi causal goal encoding and P_psi 3-step rollout + C_omega energy critic
    signals = controller(h_t, step=2, h_goal_source=h_0)
    assert signals.z_t.shape == (b, s, d_z)
    assert signals.z_pred.shape == (b, s, d_z)
    assert signals.trajectory.shape == (3, b, s, d_z)
    assert signals.energy.shape == (b, s)
    assert signals.fetch_gate.shape == (b, s)
    assert signals.moe_bias.shape == (b, s, num_experts)
    assert signals.p_halt.shape == (b, s)
    assert signals.converged.shape == (b, s)
    assert signals.converged.dtype == torch.bool
    assert bool(((signals.p_halt >= 0.0) & (signals.p_halt <= 1.0)).all())

    # Verify linear compute penalty lambda_compute * t in C_omega energy critic
    with torch.no_grad():
        z_goal = controller.encoder.encode_goal(h_0)
        e_step0 = controller.critic.energy(signals.z_pred, z_goal, step=0)
        e_step3 = controller.critic.energy(signals.z_pred, z_goal, step=3)
        expected_diff = 3.0 * controller.critic.lambda_compute
        assert torch.allclose(e_step3 - e_step0, torch.full_like(e_step0, expected_diff), atol=1e-5)

    # Verify jepa_loss (stop-gradient target + VICReg variance/covariance) & gradient flow
    h_next1 = torch.randn(b, s, d_model, requires_grad=True)
    h_next2 = torch.randn(b, s, d_model, requires_grad=True)
    sig2 = controller(h_next1, step=3, h_goal_source=h_0)
    loss = controller.jepa_loss([signals.z_pred, sig2.z_pred], [h_next1, h_next2])
    assert loss.ndim == 0
    assert torch.isfinite(loss) and float(loss.item()) > 0.0

    loss.backward()
    assert h_t.grad is not None and torch.isfinite(h_t.grad).all()
    assert controller.predictor.action_embeddings.grad is not None
    assert controller.encoder.state_proj.weight.grad is not None
    # Stop-gradient on h_next2 target means h_next2 receives no gradient from jepa_loss
    assert h_next2.grad is None

    # Verify mobile 1.3 GB budget calculation at 4-bit quantization
    footprint = controller.memory_footprint_bytes(bits=4)
    assert footprint["bits"] == 4.0
    assert footprint["mobile_ceiling_bytes"] == 1_300_000_000.0
    assert footprint["total_bytes"] <= footprint["mobile_ceiling_bytes"]
    assert footprint["within_mobile_ceiling"] == 1.0


def test_grpo_continual_user_preference_rl() -> None:
    torch.manual_seed(11)

    # 1. UserPreferenceProfile feedback recording, scoring, and state_dict round-trip
    profile = UserPreferenceProfile(preferred_length=10.0, length_tolerance=8.0, ema_alpha=0.3)
    liked_resp = [10, 20, 30, 40, 50, 60, 70, 80]
    disliked_resp = [90, 91, 92, 93, 94, 95, 96, 97]

    profile.record_feedback(liked_resp, reward=1.5, style_mode="formal")
    profile.record_feedback(disliked_resp, reward=-1.5, style_mode="informal")

    score_liked = profile.score_response(liked_resp, style_mode="formal")
    score_disliked = profile.score_response(disliked_resp, style_mode="formal")
    assert score_liked > score_disliked
    assert profile.preferred_style_mode == "formal"

    saved_state = profile.state_dict()
    restored = UserPreferenceProfile()
    restored.load_state_dict(saved_state)
    assert restored.preferred_style_mode == "formal"
    assert abs(restored.score_response(liked_resp, style_mode="formal") - score_liked) < 1e-6

    # 2. compute_group_relative_advantages & grpo_loss with non-negative Schulman KL
    rewards = torch.tensor([1.5, 0.2, -0.5, -1.2], dtype=torch.float32)
    adv = compute_group_relative_advantages(rewards)
    assert adv.shape == (4,)
    assert abs(float(adv.mean().item())) < 1e-5
    assert abs(float(adv.std(unbiased=False).item()) - 1.0) < 1e-4

    g, seq_len, vocab = 4, 7, 64
    prompt_len = 3
    logits = torch.randn(g, seq_len, vocab, requires_grad=True)
    ref_logits = (logits.detach() + 0.3 * torch.randn(g, seq_len, vocab))
    token_ids = torch.randint(0, vocab, (g, seq_len), dtype=torch.long)

    policy_lp, mask = sequence_log_probs(logits, token_ids, prompt_len=prompt_len)
    old_lp = policy_lp.detach() - 0.05
    ref_lp, _ = sequence_log_probs(ref_logits, token_ids, prompt_len=prompt_len)

    loss, metrics = grpo_loss(
        policy_log_probs=policy_lp,
        old_log_probs=old_lp,
        advantages=adv,
        ref_log_probs=ref_lp,
        mask=mask,
        clip_eps=0.2,
        kl_beta=0.05,
    )
    assert torch.isfinite(loss)
    assert metrics["kl"] >= 0.0
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()

    # 3. ContinualGRPOTrainer step_on_group & learn_from_conversation across modes
    cfg = _tiny_ultra_config(
        num_recurrent_layers=1,
        ssm_mixer_layers=(),
        enable_jepa=False,
        enable_memory_bank=False,
    )
    model = InkyModel(cfg)
    trainer = ContinualGRPOTrainer(
        model=model,
        profile=profile,
        lr=5e-4,
        grpo_epochs=1,
        ref_ema_decay=0.99,
    )
    prompt = torch.tensor([[5, 12]], dtype=torch.long)
    candidates = [
        [10, 20, 30],
        [90, 91, 92],
    ]

    for r_mode, s_mode in (("low", "formal"), ("high", "informal"), ("max", "formal")):
        step_stats = trainer.step_on_group(
            prompt_ids=prompt,
            responses=candidates,
            explicit_rewards=[1.0, -1.0],
            reasoning_mode=r_mode,
            style_mode=s_mode,
        )
        assert step_stats["kl"] >= 0.0
        assert len(step_stats["rewards"]) == 2

    conv_stats = trainer.learn_from_conversation(
        prompt_ids=prompt,
        user_feedback=1.0,
        chosen_response=[10, 20],
        group_size=2,
        max_new_tokens=2,
        reasoning_mode="low",
        style_mode="informal",
    )
    assert len(conv_stats["responses"]) == 2
    assert conv_stats["step"] == trainer.step_count


def test_hybrid_linear_gated_deltanet_and_ultra_sparse_jepa_model() -> None:
    torch.manual_seed(23)

    # 1. GatedDeltaNetMixer prefill, single-token cached decode equivalence, and segment_ids reset
    mixer_cfg = _tiny_ultra_config()
    mixer = GatedDeltaNetMixer(mixer_cfg).eval()
    x = torch.randn(2, 6, mixer_cfg.hidden_size)

    with torch.no_grad():
        full_out, full_present = mixer(x)
        assert full_out.shape == x.shape

        # Step-by-step cached decode matches full prefill
        past = None
        step_outs = []
        for t in range(x.shape[1]):
            y_t, past = mixer(x[:, t : t + 1], past=past)
            step_outs.append(y_t)
        decoded_out = torch.cat(step_outs, dim=1)
        assert torch.allclose(full_out, decoded_out, atol=1e-5)
        assert torch.allclose(full_present[0], past[0], atol=1e-5)

        # Document boundary reset via segment_ids: second segment [t >= 3] matches running alone
        seg_ids = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 2, 2, 2]], dtype=torch.long)
        seg_out, _ = mixer(x, segment_ids=seg_ids)
        doc2_alone, _ = mixer(x[:, 3:])
        assert torch.allclose(seg_out[:, 3:], doc2_alone, atol=1e-5)

    # 2. InkyConfig.hybrid_linear_128k("1:3") and ("1:7")
    cfg_1_3 = InkyConfig.hybrid_linear_128k("1:3")
    cfg_1_7 = InkyConfig.hybrid_linear_128k("1:7")
    assert cfg_1_3.max_position_embeddings == 131072
    assert cfg_1_3.num_recurrent_layers == 4
    assert cfg_1_3.ssm_mixer_layers == (0, 1, 2)
    assert cfg_1_3.linear_mixer_type == "gated_deltanet"
    assert cfg_1_7.max_position_embeddings == 131072
    assert cfg_1_7.num_recurrent_layers == 8
    assert cfg_1_7.ssm_mixer_layers == (0, 1, 2, 3, 4, 5, 6)

    # 3. Tiny InkyModel with InkyConfig.ultra_sparse_jepa(...)
    cfg = _tiny_ultra_config()
    model = InkyModel(cfg)
    assert model.jepa is not None
    assert model.memory_bank is not None
    assert model.core.additive_step_embed is not None
    assert sum(isinstance(b.attn, GatedDeltaNetMixer) for b in model.core.layers) == 3

    # Verify reasoning mode loop counts ("low"=2, "high"=5, "max"=10) & extended scratchpad budgets
    assert REASONING_MODES == {"low": 2, "high": 5, "max": 10}
    assert REASONING_SCRATCH_BUDGETS["low"] < REASONING_SCRATCH_BUDGETS["high"] < REASONING_SCRATCH_BUDGETS["max"]
    for mode, expected_loops in REASONING_MODES.items():
        assert resolve_reasoning_scratch_budget(mode, 256) == REASONING_SCRATCH_BUDGETS[mode]
        model.eval()
        with torch.no_grad():
            ids = torch.randint(0, 100, (1, 4), dtype=torch.long)
            out_mode = model(ids, reasoning_mode=mode, style_mode="formal", use_cache=True)
            assert out_mode.past is not None
            assert out_mode.past.num_loops == expected_loops

    # Verify training forward + backward with jepa_loss in out.loss_parts
    model.train()
    train_ids = torch.randint(0, 100, (2, 6), dtype=torch.long)
    out = model(train_ids, labels=train_ids, reasoning_mode="low", style_mode="informal")
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.loss_parts is not None
    assert "jepa_loss" in out.loss_parts
    assert torch.isfinite(out.loss_parts["jepa_loss"])
    out.loss.backward()
    assert model.core.additive_step_embed.grad is not None

    # Verify generate in both "formal" and "informal" style modes
    model.eval()
    prompt = torch.tensor([[10, 20, 30]], dtype=torch.long)
    for style in ("formal", "informal"):
        gen = model.generate(
            prompt,
            max_new_tokens=2,
            do_sample=False,
            reasoning_mode="low",
            style_mode=style,
        )
        assert gen.shape == (1, 5)


def test_jepa_config_wires_compute_penalty() -> None:
    from inky.jepa import JEPAController
    cfg = InkyConfig.ultra_sparse_jepa(jepa_compute_penalty=0.123)
    assert abs(JEPAController(cfg).critic.lambda_compute - 0.123) < 1e-9


def test_linear_mixers_fp32_carry_and_moe_expert_bias_steering() -> None:
    from inky.layers import SelectiveSSMMixer, SharedExpertMoE

    torch.manual_seed(3)
    cfg = _tiny_ultra_config()
    x = torch.randn(2, 6, cfg.hidden_size).bfloat16()
    seg = torch.tensor([[0, 0, 1, 1, 1, 2], [0, 0, 0, 1, 1, 1]])
    for cls in (GatedDeltaNetMixer, SelectiveSSMMixer):
        m = cls(cfg).to(torch.bfloat16).eval()
        with torch.no_grad():
            _, p = m(x)
            assert p[0].dtype == torch.float32          # prefill carry kept in fp32
            _, p = m(x[:, :1], past=p)
            assert p[0].dtype == torch.float32          # single-token decode carry kept in fp32
            out, _ = m(x, segment_ids=seg)
            alone, _ = m(x[:, 2:5])
            assert torch.allclose(out[0, 2:5].float(), alone[0].float(), atol=5e-2)

    moe = SharedExpertMoE(cfg).eval()
    bias = torch.zeros(1, 5, cfg.num_routed_experts)
    bias[..., 3] = 50.0
    with torch.no_grad():
        moe(torch.randn(1, 5, cfg.hidden_size), expert_bias=bias)
    assert int(moe.load_counts.argmax()) == 3
    moe.update_balance_bias()
    assert moe.balance_bias[3] < 0 < moe.balance_bias[0]
    assert torch.allclose(moe.balance_bias.abs(), torch.full_like(moe.balance_bias, cfg.balance_rate))
