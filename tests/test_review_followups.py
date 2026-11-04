"""Tests for the review follow-ups: TRAIN-04 early stopping, TRAIN-05 Moonlight LR warning, TRAIN-08 training-draw
NLL, DOC-05 default EOS stop in generate, and ARLO-01 untrained decision-engine warning.

Run with: python -m pytest tests/test_review_followups.py -q -p no:cacheprovider
"""

import importlib.util
import json
import math
import sys
import warnings
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import HybridOptimizer, InkyConfig, InkyModel  # noqa: E402
from inky.arlo import train_arlo  # noqa: E402
from inky.evaluation import evaluate, perplexity  # noqa: E402
from inky.train import _patience_state, read_train_state, train  # noqa: E402


def tiny_config(**overrides) -> InkyConfig:
    kw = dict(
        vocab_size=512,
        scratch_open_id=508,
        scratch_sep_id=509,
        scratch_close_id=510,
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
    kw.update(overrides)
    return InkyConfig(**kw)


def seeded_model(config: InkyConfig) -> InkyModel:
    torch.manual_seed(0)
    return InkyModel(config)


def batches(n: int = 2, seed: int = 1):
    gen = torch.Generator().manual_seed(seed)
    return [torch.randint(2, 500, (2, 16), generator=gen) for _ in range(n)]


def _quiet(_message):
    return None


# ----------------------------------------------------------------------------- TRAIN-05


def test_moonlight_with_default_muon_lr_warns_in_train():
    model = seeded_model(tiny_config())
    opt = HybridOptimizer(model, muon_lr=0.02, scale_mode="moonlight")
    with pytest.warns(UserWarning, match="moonlight.*muon_lr=0.02.*3-4x"):
        train(model, opt, batches(), steps=1, log_fn=_quiet)


def test_moonlight_warning_names_the_adamw_scale_fix():
    model = seeded_model(tiny_config())
    opt = HybridOptimizer(model, muon_lr=0.02, scale_mode="moonlight")
    with pytest.warns(UserWarning, match="AdamW-scale"):
        train(model, opt, batches(), steps=1, log_fn=_quiet)


@pytest.mark.parametrize("mode, lr", [("shape", 0.02), ("moonlight", 6e-4), ("moonlight", 1e-3)])
def test_no_moonlight_warning_for_shape_mode_or_adamw_scale_lr(mode, lr):
    model = seeded_model(tiny_config())
    opt = HybridOptimizer(model, muon_lr=lr, scale_mode=mode)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train(model, opt, batches(), steps=1, log_fn=_quiet)
    assert not [w for w in caught if "moonlight" in str(w.message)], [str(w.message) for w in caught]


# ----------------------------------------------------------------------------- TRAIN-04 early stopping


def _frozen_optimizer(model):
    # Tiny learning rates keep the weights nearly fixed so each test run is cheap. They do NOT make the eval NLL
    # constant (the MoE balance bias still moves), so the early-stopping tests never depend on the real NLL:
    # they script it with ``_script_eval`` instead.
    return HybridOptimizer(model, adamw_lr=1e-12, muon_lr=1e-12)


def _script_eval(monkeypatch, nlls):
    """Make ``inky.train.evaluate`` return the given NLLs in order, one per eval call."""
    seq = iter(nlls)

    def fake(model, batches, num_loops=None, ignore_id=None, **kw):
        nll = next(seq)
        return {"nll": nll, "perplexity": math.exp(nll) if math.isfinite(nll) else float("inf"),
                "tokens": 1, "batches": 1, "mean_loops": 1.0}

    monkeypatch.setattr("inky.train.evaluate", fake)


def _run(monkeypatch, nlls, steps=50, tmp_path=None, logs=None, **kw):
    _script_eval(monkeypatch, nlls)
    model = seeded_model(tiny_config())
    return train(model, _frozen_optimizer(model), batches(), steps=steps, eval_batches=batches(1, 5),
                 eval_every=1, checkpoint_dir=tmp_path / "run" if tmp_path else None,
                 log_fn=logs.append if logs is not None else _quiet, **kw)


def _evals(history):
    return [r for r in history if r["kind"] == "eval"]


def test_early_stopping_is_off_by_default(monkeypatch):
    history = _run(monkeypatch, [1.0] * 6, steps=6)
    assert len(_evals(history)) == 6
    assert not any(r.get("early_stopped") for r in history)


def test_early_stopping_never_improving_stops_after_patience_evals(monkeypatch):
    logs = []
    history = _run(monkeypatch, [1.0, 1.0, 1.0, 1.0], logs=logs, early_stop_patience=2)
    evals = _evals(history)
    # Step 1 is the best (no earlier eval); steps 2 and 3 are two evals without improvement, so the run stops at 3.
    assert [e["step"] for e in evals] == [1, 2, 3]
    assert [e["is_best"] for e in evals] == [True, False, False]
    assert evals[-1].get("early_stopped") is True
    assert history[-1]["kind"] == "eval" and history[-1]["step"] == 3
    assert [r["step"] for r in history if r["kind"] == "train"] == [1, 2, 3]   # no training step after the stop
    assert any("early stop at step 3" in line for line in logs)
    assert any("stopped early at step 3 of 50" in line for line in logs)


def test_early_stopping_improve_then_plateau(monkeypatch):
    history = _run(monkeypatch, [3.0, 2.0, 1.0, 1.0, 1.1, 0.9, 5.0], early_stop_patience=2)
    evals = _evals(history)
    # Best at step 3 (1.0); steps 4 and 5 do not improve, so the run stops at step 5.
    assert [e["step"] for e in evals] == [1, 2, 3, 4, 5]
    assert [e["is_best"] for e in evals] == [True, True, True, False, False]
    assert evals[-1].get("early_stopped") is True


def test_early_stopping_counter_resets_on_a_new_best(monkeypatch):
    history = _run(monkeypatch, [3.0, 3.0, 2.0, 2.5, 2.6, 2.7], early_stop_patience=2)
    assert [e["step"] for e in _evals(history)] == [1, 2, 3, 4, 5]     # the improvement at 3 reset the count


def test_early_stopping_does_not_trigger_while_the_loss_keeps_improving(monkeypatch):
    history = _run(monkeypatch, [5.0, 4.0, 3.0, 2.0, 1.0], steps=5, early_stop_patience=1)
    assert history[-1]["step"] == 5 and not any(r.get("early_stopped") for r in history)


def test_early_stopping_writes_a_final_checkpoint_at_the_stop_step(monkeypatch, tmp_path):
    _run(monkeypatch, [1.0, 1.0, 1.0], tmp_path=tmp_path, early_stop_patience=1)
    assert (tmp_path / "run" / "step_2" / "model").is_dir()
    assert not (tmp_path / "run" / "step_3").exists()
    assert read_train_state(tmp_path / "run" / "step_2")["step"] == 2


def test_early_stopping_logs_the_train_line_on_the_stop_step(monkeypatch):
    logs = []
    _run(monkeypatch, [1.0, 1.0, 1.0], logs=logs, early_stop_patience=1, log_every=100)
    # step 1 is neither a multiple of log_every nor the last step, but the stop step must still get its train line.
    stop_idx = next(i for i, line in enumerate(logs) if "early stop at step 2" in line)
    assert any(line.startswith("step 2/") for line in logs[stop_idx:]), logs
    assert not any(line.startswith("step 1/") for line in logs)


def test_early_stopping_counter_is_rebuilt_on_resume(monkeypatch, tmp_path):
    _run(monkeypatch, [2.0, 2.0], steps=2, tmp_path=tmp_path)
    # Evals at steps 1 (best), 2 (no improvement) are saved. Resuming with patience 2: the next eval (step 3)
    # is the second without improvement, so the run stops at step 3.
    _script_eval(monkeypatch, [2.0, 2.0])
    model2 = seeded_model(tiny_config())
    history = train(model2, _frozen_optimizer(model2), batches(), steps=50, eval_batches=batches(1, 5),
                    eval_every=1, early_stop_patience=2, resume_from=tmp_path / "run" / "step_2", log_fn=_quiet)
    assert [e["step"] for e in _evals(history)] == [1, 2, 3]
    assert history[-1]["step"] == 3


def test_patience_state_is_rebuilt_from_eval_nll_not_is_best():
    # An older history without is_best: the minimum is at the third eval, two finite evals follow it.
    hist = [{"kind": "train", "step": 1, "loss": 1.0}] + [
        {"kind": "eval", "step": s, "eval_nll": v} for s, v in enumerate([3.0, 2.0, 1.0, 1.5, 1.2], start=1)
    ]
    count, ref = _patience_state(hist)
    assert (count, ref) == (2, 1.0)
    assert _patience_state([]) == (0, math.inf)
    nan_hist = hist + [{"kind": "eval", "step": 9, "eval_nll": float("nan")}]
    assert _patience_state(nan_hist)[0] == 2                 # non-finite evals are not counted


def test_resume_from_a_history_without_is_best_still_counts_patience(monkeypatch, tmp_path):
    _run(monkeypatch, [2.0, 1.0, 1.5], steps=3, tmp_path=tmp_path)
    ckpt = tmp_path / "run" / "step_3"
    state = read_train_state(ckpt)
    for rec in state["history"]:
        rec.pop("is_best", None)
    (ckpt / "train_state.json").write_text(json.dumps(state))
    _script_eval(monkeypatch, [1.6, 1.7])
    model2 = seeded_model(tiny_config())
    history = train(model2, _frozen_optimizer(model2), batches(), steps=50, eval_batches=batches(1, 5),
                    eval_every=1, early_stop_patience=2, resume_from=ckpt, log_fn=_quiet)
    # One eval (step 3) already follows the best (step 2), so step 4 is the second and the run stops there.
    assert history[-1]["step"] == 4 and _evals(history)[-1].get("early_stopped") is True


def test_min_delta_requires_a_large_enough_improvement(monkeypatch):
    # With min_delta 0.5, 0.9 and 0.7 are not enough of a gain over the reference 1.0, so the run stops after 2 evals
    # without improvement (steps 2 and 3).
    history = _run(monkeypatch, [1.0, 0.9, 0.7, 0.6, 0.5], early_stop_patience=2, early_stop_min_delta=0.5)
    evals = _evals(history)
    assert [e["step"] for e in evals] == [1, 2, 3]
    # is_best (and so best.pt) still follows every strict new minimum.
    assert [e["is_best"] for e in evals] == [True, True, True]


def test_min_delta_zero_counts_any_strict_improvement(monkeypatch):
    history = _run(monkeypatch, [1.0, 0.99, 0.98, 0.97], steps=4, early_stop_patience=1, early_stop_min_delta=0.0)
    assert history[-1]["step"] == 4 and not any(r.get("early_stopped") for r in history)


def test_min_delta_reference_moves_only_on_counted_improvements(monkeypatch):
    # Reference 1.0; 0.4 beats it by more than 0.5 (counted, reference 0.4); 0.3 and 0.2 do not beat 0.4 by 0.5.
    history = _run(monkeypatch, [1.0, 0.4, 0.3, 0.2, 0.1], early_stop_patience=2, early_stop_min_delta=0.5)
    assert [e["step"] for e in _evals(history)] == [1, 2, 3, 4]


def test_min_delta_is_stored_in_the_train_config_and_validated(monkeypatch, tmp_path):
    _run(monkeypatch, [1.0, 0.5], steps=2, tmp_path=tmp_path, early_stop_patience=3, early_stop_min_delta=0.25)
    cfg = json.loads((tmp_path / "run" / "step_2" / "train_config.json").read_text())
    assert cfg["early_stop_min_delta"] == 0.25 and cfg["early_stop_patience"] == 3
    model = seeded_model(tiny_config())
    with pytest.raises(ValueError, match="early_stop_min_delta"):
        train(model, _frozen_optimizer(model), batches(), steps=1, early_stop_min_delta=-0.1)


def test_early_stopping_argument_validation():
    model = seeded_model(tiny_config())
    opt = _frozen_optimizer(model)
    with pytest.raises(ValueError, match="early_stop_patience"):
        train(model, opt, batches(), steps=2, early_stop_patience=0, eval_batches=batches(1), eval_every=1)
    with pytest.raises(ValueError, match="needs eval_batches"):
        train(model, opt, batches(), steps=2, early_stop_patience=1)
    with pytest.raises(ValueError, match="needs eval_batches"):
        train(model, opt, batches(), steps=2, early_stop_patience=1, eval_batches=batches(1), eval_every=0)


def _load_train_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("inky_train_script", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_early_stop_flags_parse_and_default_off(tmp_path):
    script = _load_train_script()
    base = ["--dry-run", "--out-dir", str(tmp_path)]
    args = script.parse_args(base)
    assert args.early_stop_patience is None and args.early_stop_min_delta == 0.0
    args = script.parse_args(base + ["--early-stop-patience", "3", "--early-stop-min-delta", "0.01"])
    assert args.early_stop_patience == 3 and args.early_stop_min_delta == 0.01


@pytest.mark.parametrize("extra", [
    ["--early-stop-patience", "0"],
    ["--early-stop-min-delta", "-1"],
    ["--early-stop-patience", "2"],            # not a dry run here, so --eval-every stays 0
])
def test_cli_early_stop_flags_reject_bad_values(tmp_path, extra):
    script = _load_train_script()
    argv = ["--out-dir", str(tmp_path), "--corpus", "x.txt", "--tokenizer-path", "t"] + extra
    with pytest.raises(SystemExit):
        script.parse_args(argv)


def test_cli_passes_early_stop_flags_to_train(monkeypatch, tmp_path):
    script = _load_train_script()
    seen = {}

    def fake_train(*a, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(script, "train", fake_train)
    script.main(["--dry-run", "--out-dir", str(tmp_path), "--steps", "2", "--early-stop-patience", "4",
                 "--early-stop-min-delta", "0.02"])
    assert seen["early_stop_patience"] == 4 and seen["early_stop_min_delta"] == 0.02


# ----------------------------------------------------------------------------- TRAIN-08 training-draw report


def test_evaluate_default_output_is_unchanged():
    model = seeded_model(tiny_config())
    out = evaluate(model, batches())
    assert set(out) == {"nll", "perplexity", "tokens", "batches", "mean_loops"}
    assert out["nll"] == perplexity(model, batches())["nll"]


def test_evaluate_train_draw_is_the_mean_of_the_per_depth_nlls():
    model = seeded_model(tiny_config(min_recurrent_loops=2, max_recurrent_loops=5))
    data = batches()
    out = evaluate(model, data, train_draw=True)
    per_depth = [perplexity(model, data, num_loops=d)["nll"] for d in (2, 3, 4, 5)]
    assert out["train_draw_depths"] == [2, 3, 4, 5]
    assert out["nll_train_draw"] == pytest.approx(sum(per_depth) / 4, abs=1e-6)
    assert out["perplexity_train_draw"] == pytest.approx(math.exp(out["nll_train_draw"]), rel=1e-9)
    assert out["nll"] == evaluate(model, data)["nll"]                 # the headline value is unchanged
    assert not any("mean_depth" in k for k in out)                    # the blended fields are gone


def test_evaluate_train_draw_with_a_single_depth_equals_that_depth():
    model = seeded_model(tiny_config(min_recurrent_loops=3, max_recurrent_loops=3))
    data = batches()
    out = evaluate(model, data, train_draw=True)
    assert out["nll_train_draw"] == pytest.approx(perplexity(model, data, num_loops=3)["nll"], abs=1e-6)


def test_evaluate_train_draw_accepts_a_generator_of_batches():
    model = seeded_model(tiny_config())
    data = batches()
    out = evaluate(model, (b for b in data), train_draw=True)
    assert out["nll_train_draw"] == pytest.approx(evaluate(model, data, train_draw=True)["nll_train_draw"], abs=1e-9)


# ----------------------------------------------------------------------------- DOC-05 default EOS stop


def test_generate_stops_at_config_eos_when_no_eos_is_passed():
    # Find the token the model emits first, then make it the configured EOS. No eos_token_id is passed to generate.
    prompt = torch.randint(2, 500, (1, 5))
    kw = dict(do_sample=False, repetition_penalty=1.0)
    probe = seeded_model(tiny_config(eos_token_id=1))
    first = int(probe.generate(prompt, max_new_tokens=1, **kw)[0, -1])

    model = seeded_model(tiny_config(eos_token_id=first))          # same weights, configured EOS = first token
    out = model.generate(prompt, max_new_tokens=8, **kw)
    assert out.shape == (1, 6), out.shape
    assert int(out[0, -1]) == first


def test_explicit_eos_overrides_the_config_default():
    prompt = torch.randint(2, 500, (1, 5))
    kw = dict(do_sample=False, repetition_penalty=1.0)
    probe = seeded_model(tiny_config(eos_token_id=1))
    first = int(probe.generate(prompt, max_new_tokens=1, **kw)[0, -1])
    model = seeded_model(tiny_config(eos_token_id=511))           # configured EOS never emitted here
    out = model.generate(prompt, max_new_tokens=8, eos_token_id=first, **kw)
    assert out.shape == (1, 6) and int(out[0, -1]) == first


# ----------------------------------------------------------------------------- ARLO-01 untrained warning


def _decision_model():
    torch.manual_seed(0)
    return InkyModel(tiny_config(enable_arlo=True))


def _binary_examples(n: int = 6, seed: int = 7):
    gen = torch.Generator().manual_seed(seed)
    return [
        {"context": torch.randint(2, 500, (1, 9), generator=gen), "gold": int(i % 2)}
        for i in range(n)
    ]


def test_decide_warns_until_arlo_is_trained():
    model = _decision_model()
    ctx = torch.randint(2, 500, (1, 9))
    assert model.arlo_trained is False
    with pytest.warns(UserWarning, match="ARLO-01"):
        model.decide_binary(ctx)
    with pytest.warns(UserWarning, match="ARLO-01"):
        model.decide_choice(ctx, [torch.randint(2, 500, (1, 2))])


def test_train_arlo_clears_the_untrained_warning():
    model = _decision_model()
    train_arlo(model, [_binary_examples()], task="binary", lr=1e-2, epochs=1)
    assert model.arlo_trained is True
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.decide_binary(torch.randint(2, 500, (1, 9)))
    assert not [w for w in caught if "ARLO-01" in str(w.message)]


def test_mark_arlo_trained_silences_the_warning():
    model = _decision_model()
    model.mark_arlo_trained()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.decide_binary(torch.randint(2, 500, (1, 9)))
    assert not [w for w in caught if "ARLO-01" in str(w.message)]


def test_model_without_arlo_raises_and_does_not_warn():
    model = seeded_model(tiny_config(enable_arlo=False))
    assert model.arlo_trained is False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(RuntimeError, match="disabled"):
            model.decide_binary(torch.randint(2, 500, (1, 9)))
    assert not [w for w in caught if "ARLO-01" in str(w.message)]
