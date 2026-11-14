"""Regression tests for Phase 1 critical bug fixes and integrity patches."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inky import InkyConfig, InkyModel  # noqa: E402
from inky.data import DEFAULT_MIN_CHARS, prepare_corpus  # noqa: E402
from inky.layers import geometric_prior, pondernet_kl  # noqa: E402
from inky.optim import HybridOptimizer  # noqa: E402
from inky.train import train  # noqa: E402
from scripts import train as train_script  # noqa: E402


def _tiny_config(**overrides) -> InkyConfig:
    defaults = dict(
        vocab_size=64,
        scratch_open_id=61,
        scratch_sep_id=62,
        scratch_close_id=63,
        eos_token_id=1,
        pad_token_id=0,
        hidden_size=32,
        num_attention_heads=2,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=16,
        q_lora_rank=16,
        kv_lora_rank=16,
        num_pre_layers=1,
        num_recurrent_layers=1,
        num_post_layers=1,
        min_recurrent_loops=2,
        max_recurrent_loops=2,
        intermediate_size=32,
        num_shared_experts=1,
        num_routed_experts=2,
        num_active_routed=1,
        ssm_state_dim=8,
        max_position_embeddings=64,
    )
    defaults.update(overrides)
    return InkyConfig(**defaults)


def test_pondernet_kl_zero_weight_has_zero_gradient():
    w = torch.tensor([[[1.0]], [[0.0]]], requires_grad=True)  # [L=2, B=1, S=1]
    q = geometric_prior(2, 0.5)
    kl = pondernet_kl(w, q)
    kl.backward()
    assert torch.isfinite(w.grad).all()
    assert float(w.grad[1, 0, 0]) == pytest.approx(0.0, abs=1e-7)


def test_prepare_corpus_with_segments_returns_doc_ids():
    docs = [
        "First short story about a quiet river.",
        "Second short story about a bright lantern.",
    ]
    tokens, doc_ids = prepare_corpus(
        docs,
        lambda s: [ord(c) % 50 + 2 for c in s],
        seq_len=16,
        eos_id=1,
        with_segments=True,
    )
    assert tokens.shape == doc_ids.shape
    assert tokens.ndim == 2 and tokens.shape[1] == 16
    assert int(doc_ids.max()) >= 1


def test_scripts_train_min_chars_default_matches_data_module(tmp_path):
    args = train_script.parse_args(["--dry-run", "--out-dir", str(tmp_path)])
    assert args.min_chars == DEFAULT_MIN_CHARS == 20


def test_generate_batched_eos_pads_finished_rows():
    cfg = _tiny_config()
    model = InkyModel(cfg).eval()
    calls = {"n": 0}

    def fake_forward(input_ids, past=None, num_loops=None, last_only=True, **kwargs):
        calls["n"] += 1
        b = input_ids.shape[0]
        logits = torch.full((b, 1, cfg.vocab_size), -10.0)
        if calls["n"] == 1:
            logits[0, 0, cfg.eos_token_id] = 10.0
            logits[1, 0, 5] = 10.0
        elif calls["n"] == 2:
            logits[0, 0, 7] = 10.0
            logits[1, 0, cfg.eos_token_id] = 10.0
        else:
            logits[:, 0, 9] = 10.0
        if past is not None:
            past.seq_len += input_ids.shape[1]
        return SimpleNamespace(logits=logits)

    model.forward = fake_forward  # type: ignore[method-assign]
    out = model.generate(
        torch.tensor([[2, 3], [2, 3]]),
        max_new_tokens=6,
        do_sample=False,
        no_repeat_ngram_size=0,
        blocked_token_ids=[],
    )
    # Row 0 emitted EOS at step 1, so step 2 must be padded (0), and loop stops at step 2 when row 1 emits EOS.
    assert out.shape == (2, 4)
    assert out[0].tolist() == [2, 3, cfg.eos_token_id, cfg.pad_token_id]
    assert out[1].tolist() == [2, 3, 5, cfg.eos_token_id]


def test_generate_empty_scratch_block_does_not_infinite_loop():
    cfg = _tiny_config()
    model = InkyModel(cfg).eval()
    # Emit OPEN, then SEP, then CLOSE (0 answer tokens), then token 5, then EOS.
    scripted = [cfg.scratch_open_id, cfg.scratch_sep_id, cfg.scratch_close_id, 5, cfg.eos_token_id]
    idx = {"i": 0}

    def fake_forward(input_ids, past=None, num_loops=None, last_only=True, **kwargs):
        tok = scripted[min(idx["i"], len(scripted) - 1)]
        idx["i"] += 1
        logits = torch.full((1, 1, cfg.vocab_size), -10.0)
        logits[0, 0, tok] = 10.0
        logits[0, 0, 5] = 2.0  # fallback when scratch_open_id is masked
        if past is not None:
            past.seq_len += input_ids.shape[1]
        return SimpleNamespace(logits=logits)

    model.forward = fake_forward  # type: ignore[method-assign]
    out = model.generate(
        torch.tensor([[2, 3]]),
        max_new_tokens=6,
        do_sample=False,
        no_repeat_ngram_size=0,
        reduce_scratch_blocks=True,
        blocked_token_ids=[],
    )
    assert out[0, :2].tolist() == [2, 3]
    assert cfg.scratch_open_id not in out[0].tolist()


def test_grad_scaler_backs_off_on_nonfinite_loss_skip(monkeypatch):
    cfg = _tiny_config()
    model = InkyModel(cfg)
    opt = HybridOptimizer(model, adamw_lr=1e-3, muon_lr=1e-2)
    batches = [torch.randint(2, 50, (2, 8))]

    orig_forward = model.forward
    call_count = {"n": 0}

    def nan_first_forward(*args, **kwargs):
        call_count["n"] += 1
        out = orig_forward(*args, **kwargs)
        if call_count["n"] == 1:
            out.loss = out.loss * float("nan")
        return out

    model.forward = nan_first_forward  # type: ignore[method-assign]

    class FakeScaler:
        def __init__(self):
            self._scale = torch.tensor(65536.0)
            self.updates = []

        def is_enabled(self):
            return True

        def get_scale(self):
            return float(self._scale)

        def get_backoff_factor(self):
            return 0.5

        def scale(self, loss):
            return loss

        def unscale_(self, _opt):
            pass

        def step(self, _opt):
            pass

        def update(self, new_scale=None):
            if new_scale is not None:
                self._scale = torch.tensor(float(new_scale))
            self.updates.append(float(self._scale))

    fake = FakeScaler()
    monkeypatch.setattr("inky.train._make_scaler", lambda _amp, _dev: fake)
    history = train(model, opt, batches, steps=2, max_skipped_steps=5, log_every=0)
    assert history[0]["skipped"] is True
    assert history[0]["skip_reason"] == "nonfinite_loss"
    assert fake.updates[0] == 32768.0
    assert history[1]["skipped"] is False
