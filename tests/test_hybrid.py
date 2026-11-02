"""Hybrid recurrent core: selective SSM mixer in some recurrent layers, attention in the rest."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel  # noqa: E402
from inky.layers import MultiLatentAttention, SelectiveSSMMixer  # noqa: E402

OPEN, SEP, CLOSE = 508, 509, 510


def tiny_config(**overrides) -> InkyConfig:
    kwargs = dict(
        vocab_size=512,
        scratch_open_id=OPEN,
        scratch_sep_id=SEP,
        scratch_close_id=CLOSE,
        hidden_size=64,
        num_attention_heads=4,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        q_lora_rank=32,
        kv_lora_rank=24,
        intermediate_size=96,
        num_routed_experts=4,
        num_active_routed=2,
        ssm_state_dim=16,
        max_recurrent_loops=4,
        min_recurrent_loops=2,
        loop_dropout_prob=0.0,
        max_position_embeddings=256,
        arlo_dim=16,
        arlo_pointer_rank=8,
        arlo_slots=4,
        eos_token_id=1,
        pad_token_id=1,
    )
    kwargs.update(overrides)
    return InkyConfig(**kwargs)


def hybrid_config() -> InkyConfig:
    return tiny_config(ssm_mixer_layers=(0,))


def seeded_model(config: InkyConfig) -> InkyModel:
    torch.manual_seed(0)
    model = InkyModel(config)
    model.eval()
    return model


def n_params(model: InkyModel) -> int:
    return sum(p.numel() for p in model.parameters())


def test_hybrid_mixer_is_causal_late_token_does_not_change_early_logits():
    model = seeded_model(hybrid_config())
    ids = torch.randint(2, 500, (2, 16))
    changed = ids.clone()
    changed[:, -1] = (changed[:, -1] + 7) % 500 + 2
    with torch.no_grad():
        a = model(ids, num_loops=4).logits
        b = model(changed, num_loops=4).logits
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)
    assert not torch.allclose(a[:, -1], b[:, -1], atol=1e-5)


def test_hybrid_cached_decoding_matches_full_forward():
    model = seeded_model(hybrid_config())
    ids = torch.randint(2, 512, (2, 18))
    with torch.no_grad():
        full = model(ids, num_loops=4).logits

        prefix = 10
        out = model(ids[:, :prefix], use_cache=True, num_loops=4)
        cache = out.past
        step_logits = [out.logits]
        for t in range(prefix, ids.shape[1]):
            out = model(ids[:, t:t + 1], past=cache)
            step_logits.append(out.logits)
        incremental = torch.cat(step_logits, dim=1)

    assert torch.allclose(full[:, :prefix], incremental[:, :prefix], atol=1e-4)
    assert torch.allclose(full[:, prefix:], incremental[:, prefix:], atol=1e-4)


def test_default_config_builds_attention_only():
    config = tiny_config()
    assert config.ssm_mixer_layers == ()
    model = seeded_model(config)
    mixers = [m for m in model.modules() if isinstance(m, SelectiveSSMMixer)]
    assert mixers == []
    attn = [m for m in model.modules() if isinstance(m, MultiLatentAttention)]
    assert len(attn) == config.num_pre_layers + config.num_recurrent_layers + config.num_post_layers


def test_hybrid_has_fewer_parameters_than_attention_only():
    attention_only = n_params(seeded_model(tiny_config()))
    hybrid = seeded_model(hybrid_config())
    hybrid_total = n_params(hybrid)
    assert sum(isinstance(m, SelectiveSSMMixer) for m in hybrid.modules()) == 1
    assert hybrid_total < attention_only
    print(f"\nparameters: attention-only={attention_only} hybrid={hybrid_total}")
