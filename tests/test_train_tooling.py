"""Tests for the training tooling: batched Muon, --compile fallback, --set / --config-json overrides, the vocabulary
check and tokens/sec logging.

Run with: python -m pytest tests/test_train_tooling.py -q -p no:cacheprovider
"""

import importlib.util
import io
import json
import sys
import warnings
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inky import HybridOptimizer, InkyConfig, InkyModel, Muon  # noqa: E402
from inky.train import train  # noqa: E402


def _load_train_script():
    spec = importlib.util.spec_from_file_location("inky_train_script_tooling", ROOT / "scripts" / "train.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def tiny_config(**overrides) -> InkyConfig:
    kw = dict(
        vocab_size=512, scratch_open_id=509, scratch_sep_id=510, scratch_close_id=511, hidden_size=64,
        num_attention_heads=4, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=16, q_lora_rank=32,
        kv_lora_rank=24, intermediate_size=96, num_routed_experts=4, num_active_routed=2, ssm_state_dim=16,
        max_recurrent_loops=4, min_recurrent_loops=2, loop_dropout_prob=0.0, max_position_embeddings=256,
        arlo_dim=16, arlo_pointer_rank=8, arlo_slots=4, eos_token_id=1, pad_token_id=1,
    )
    kw.update(overrides)
    return InkyConfig(**kw)


def seeded_model(**overrides) -> InkyModel:
    torch.manual_seed(0)
    return InkyModel(tiny_config(**overrides))


def windows(n: int = 8, seq: int = 16, seed: int = 1) -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    return torch.randint(2, 500, (n, seq), generator=gen)


def _quiet(_message):
    return None


# ----------------------------------------------------------------------------- 1. batched Muon


def _fill_grads(opt: HybridOptimizer, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    for p in opt.params:
        p.grad = torch.randn(p.shape, generator=gen)


def _round_trip(state):
    """Through torch.save/load like a real checkpoint (optimiser load_state_dict can otherwise alias live tensors)."""
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False)


def _pair(**kw):
    a, b = seeded_model(), seeded_model()
    return (a, HybridOptimizer(a, muon_batched=False, **kw)), (b, HybridOptimizer(b, muon_batched=True, **kw))


def test_the_tiny_model_has_same_shape_muon_matrices():
    # Guards the equivalence test below: it must actually exercise the stacked path.
    model = seeded_model()
    opt = HybridOptimizer(model)
    shapes = [tuple(p.shape) for g in opt.muon.param_groups for p in g["params"]]
    assert len(shapes) > len(set(shapes))


@pytest.mark.parametrize("scale_mode", ["shape", "moonlight"])
def test_batched_muon_matches_per_matrix_over_several_steps(scale_mode):
    (ma, oa), (mb, ob) = _pair(scale_mode=scale_mode, muon_lr=0.01 if scale_mode == "shape" else 1e-3)
    for step in range(3):
        _fill_grads(oa, 100 + step)
        _fill_grads(ob, 100 + step)
        oa.step(max_grad_norm=1.0)
        ob.step(max_grad_norm=1.0)
    for (na, pa), (nb, pb) in zip(ma.named_parameters(), mb.named_parameters()):
        assert na == nb
        assert torch.allclose(pa, pb, atol=1e-5), na
    for pa, pb in zip(oa.muon.param_groups[0]["params"], ob.muon.param_groups[0]["params"]):
        assert torch.allclose(oa.muon.state[pa]["momentum_buffer"], ob.muon.state[pb]["momentum_buffer"], atol=1e-6)


def test_batched_muon_handles_missing_grads_and_non_nesterov():
    shapes = [(8, 16), (8, 16), (16, 8), (8, 16)]
    gen = torch.Generator().manual_seed(0)
    base = [torch.randn(s, generator=gen) for s in shapes]
    grads = [torch.randn(s, generator=gen) for s in shapes]

    def run(batched):
        params = [torch.nn.Parameter(t.clone()) for t in base]
        opt = Muon(params, lr=0.02, nesterov=False, weight_decay=0.1, batched=batched)
        for p, g in zip(params, grads):
            p.grad = g.clone()
        params[1].grad = None                      # one parameter has no gradient this step
        opt.step()
        return params

    for pa, pb in zip(run(False), run(True)):
        assert torch.allclose(pa, pb, atol=1e-5)


def test_old_per_parameter_state_loads_into_a_batched_optimizer_and_vice_versa():
    (ma, oa), (mb, ob) = _pair()
    for step in range(2):
        _fill_grads(oa, 7 + step)
        oa.step(max_grad_norm=1.0)
    old_state = _round_trip(oa.state_dict())       # written by the per-matrix path: the "old" format
    assert "batched" not in json.dumps(list(old_state["muon"].keys()))
    assert all(set(v) == {"momentum_buffer"} for v in old_state["muon"]["state"].values())

    mb.load_state_dict(ma.state_dict())
    ob.load_state_dict(old_state)
    new_state = ob.state_dict()
    assert new_state["muon"]["param_groups"] == old_state["muon"]["param_groups"]
    assert set(new_state["muon"]["state"]) == set(old_state["muon"]["state"])

    # Continuing from the loaded state, both paths agree again.
    _fill_grads(oa, 50)
    _fill_grads(ob, 50)
    oa.step()
    ob.step()
    for pa, pb in zip(ma.parameters(), mb.parameters()):
        assert torch.allclose(pa, pb, atol=1e-5)

    # A batched optimizer's state loads into a per-matrix one too (same format).
    other = seeded_model()
    other_opt = HybridOptimizer(other, muon_batched=False)
    other_opt.load_state_dict(_round_trip(ob.state_dict()))


def test_batched_is_opt_in():
    assert Muon([torch.nn.Parameter(torch.zeros(4, 4))]).batched is False
    assert HybridOptimizer(seeded_model()).muon.batched is False
    assert HybridOptimizer(seeded_model(), muon_batched=True).muon.batched is True


# ----------------------------------------------------------------------------- 2. --compile


def _losses(history):
    return [r["loss"] for r in history if r["kind"] == "train"]


def _run_train(model_kwargs=None, **kw):
    model = seeded_model(**(model_kwargs or {}))
    return train(model, HybridOptimizer(model, adamw_lr=1e-3, muon_lr=0.01), windows(), steps=3, batch_size=2,
                 log_fn=_quiet, **kw)


def test_compile_is_off_by_default(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("torch.compile must not be called by default")

    monkeypatch.setattr(torch, "compile", boom)
    assert len(_run_train()) == 3


def test_compile_failure_at_wrap_time_falls_back_to_eager_with_a_warning(monkeypatch):
    eager = _losses(_run_train())

    def boom(*a, **k):
        raise RuntimeError("no compiler here")

    monkeypatch.setattr(torch, "compile", boom)
    with pytest.warns(RuntimeWarning, match="torch.compile failed.*eager"):
        history = _run_train(compile_model=True)
    assert _losses(history) == eager


def test_compile_failure_on_the_first_step_falls_back_and_rewinds_the_rng(monkeypatch):
    eager = _losses(_run_train())
    calls = []

    def lazy_boom(model):
        def forward(**kwargs):
            calls.append(1)
            torch.rand(3)                          # consume global RNG before failing, like a half-run forward
            raise RuntimeError("inductor exploded")
        return forward

    monkeypatch.setattr(torch, "compile", lazy_boom)
    with pytest.warns(RuntimeWarning, match="first step"):
        history = _run_train(compile_model=True)
    assert len(calls) == 1                         # tried once, then eager for the rest of the run
    assert _losses(history) == eager


def test_working_compile_wrapper_is_used_and_recorded(monkeypatch, tmp_path):
    eager = _losses(_run_train())
    used = []

    def passthrough(model):
        def forward(**kwargs):
            used.append(1)
            return model(**kwargs)
        return forward

    monkeypatch.setattr(torch, "compile", passthrough)
    history = _run_train(compile_model=True, checkpoint_dir=tmp_path, checkpoint_every=3)
    assert len(used) == 3 and _losses(history) == eager
    cfg = json.loads((tmp_path / "step_3" / "train_config.json").read_text())
    assert cfg["compile_model"] is True


def test_script_passes_compile_and_skips_it_where_a_dry_run_cannot_compile(monkeypatch, tmp_path):
    script = _load_train_script()
    seen = []
    monkeypatch.setattr(script, "train", lambda *a, **kw: seen.append(kw) or [])

    args = ["--dry-run", "--steps", "1", "--compile"]
    monkeypatch.setattr(script, "compile_supported_for_dry_run", lambda: False)
    script.main(args + ["--out-dir", str(tmp_path / "a")])
    monkeypatch.setattr(script, "compile_supported_for_dry_run", lambda: True)
    script.main(args + ["--out-dir", str(tmp_path / "b")])
    script.main(["--dry-run", "--steps", "1", "--out-dir", str(tmp_path / "c")])
    assert [kw["compile_model"] for kw in seen] == [False, True, False]


def test_script_batched_muon_flag(monkeypatch, tmp_path):
    script = _load_train_script()
    seen = []

    def fake_train(model, optimizer, *a, **kw):
        seen.append(optimizer.muon.batched)
        return []

    monkeypatch.setattr(script, "train", fake_train)
    script.main(["--dry-run", "--steps", "1", "--out-dir", str(tmp_path / "a"), "--muon-batched"])
    script.main(["--dry-run", "--steps", "1", "--out-dir", str(tmp_path / "b")])
    assert seen == [True, False]


# ----------------------------------------------------------------------------- 3. --set / --config-json


def _parse(script, tmp_path, *extra):
    return script.parse_args(["--dry-run", "--out-dir", str(tmp_path)] + list(extra))


def test_set_parses_literals_and_collects_repeats(tmp_path):
    script = _load_train_script()
    args = _parse(script, tmp_path, "--set", "qk_norm=True", "--set", "vocab_size=32003",
                  "--set", "rope_theta=5e5", "--set", "num_routed_experts=8")
    assert args.config_overrides == {"qk_norm": True, "vocab_size": 32003, "rope_theta": 500000.0,
                                     "num_routed_experts": 8}
    assert _parse(script, tmp_path).config_overrides == {}


def test_set_accepts_lowercase_words_and_later_values_win(tmp_path):
    script = _load_train_script()
    args = _parse(script, tmp_path, "--set", "qk_norm=true", "--set", "qk_norm=False")
    assert args.config_overrides == {"qk_norm": False}


@pytest.mark.parametrize("item, message", [
    ("no_such_field=1", "unknown InkyConfig field"),
    ("hidden_sze=64", "Did you mean hidden_size"),
    ("hidden_size=1.5", "expects an int"),
    ("qk_norm=1", "expects a bool"),
    ("hidden_size", "key=value"),
    ("hidden_size=[1, 2", "Python literal"),
])
def test_set_errors_are_clear(tmp_path, capsys, item, message):
    script = _load_train_script()
    with pytest.raises(SystemExit):
        _parse(script, tmp_path, "--set", item)
    assert message in capsys.readouterr().err


def test_config_json_is_loaded_and_set_wins(tmp_path):
    script = _load_train_script()
    path = tmp_path / "variant.json"
    path.write_text(json.dumps({"num_routed_experts": 8, "qk_norm": True, "ssm_mixer_layers": [0]}))
    args = _parse(script, tmp_path, "--config-json", str(path), "--set", "num_routed_experts=2")
    assert args.config_overrides == {"num_routed_experts": 2, "qk_norm": True, "ssm_mixer_layers": (0,)}


def test_config_json_accepts_a_saved_train_config_and_rejects_bad_files(tmp_path, capsys):
    script = _load_train_script()
    saved = tmp_path / "saved.json"
    saved.write_text(json.dumps({"model_config": {"num_routed_experts": 8}, "steps": 3}))
    assert _parse(script, tmp_path, "--config-json", str(saved)).config_overrides == {"num_routed_experts": 8}

    for name, text, message in (("bad.json", "{", "cannot read"), ("list.json", "[1]", "JSON object"),
                                ("unk.json", json.dumps({"nope": 1}), "unknown InkyConfig field")):
        (tmp_path / name).write_text(text)
        with pytest.raises(SystemExit):
            _parse(script, tmp_path, "--config-json", str(tmp_path / name))
        assert message in capsys.readouterr().err
    with pytest.raises(SystemExit):
        _parse(script, tmp_path, "--config-json", str(tmp_path / "missing.json"))


def test_fields_are_read_from_the_config_signature_so_new_keys_work(tmp_path, monkeypatch):
    # Keys other agents add to InkyConfig (use_moe, use_highway, qk_norm_mode, ...) need no change here.
    script = _load_train_script()
    fields = dict(script.config_fields())
    assert "hidden_size" in fields and "self" not in fields and "kwargs" not in fields
    monkeypatch.setattr(script, "config_fields", lambda: {**fields, "use_moe": True, "qk_norm_mode": "per_head"})
    args = _parse(script, tmp_path, "--set", "use_moe=False", "--set", 'qk_norm_mode="shared"')
    assert args.config_overrides == {"use_moe": False, "qk_norm_mode": "shared"}
    with pytest.raises(SystemExit):
        _parse(script, tmp_path, "--set", "use_moe=3")


def test_scratch_ids_follow_vocab_size_unless_given():
    script = _load_train_script()
    assert script.derive_scratch_ids({"vocab_size": 600}) == {
        "vocab_size": 600, "scratch_open_id": 597, "scratch_sep_id": 598, "scratch_close_id": 599}
    explicit = {"vocab_size": 600, "scratch_close_id": 590}
    assert script.derive_scratch_ids(explicit) == explicit
    assert script.derive_scratch_ids({"hidden_size": 64}) == {"hidden_size": 64}


def test_dry_run_main_applies_overrides_and_scratch_ids(monkeypatch, tmp_path):
    script = _load_train_script()
    seen = {}

    def fake_train(model, optimizer, *a, **kw):
        seen["config"] = model.config
        seen["extra"] = kw["extra_config"]
        return []

    monkeypatch.setattr(script, "train", fake_train)
    script.main(["--dry-run", "--steps", "1", "--out-dir", str(tmp_path), "--set", "num_routed_experts=6",
                 "--set", "vocab_size=600", "--set", "qk_norm=True"])
    cfg = seen["config"]
    assert (cfg.num_routed_experts, cfg.vocab_size, cfg.qk_norm) == (6, 600, True)
    assert (cfg.scratch_open_id, cfg.scratch_sep_id, cfg.scratch_close_id) == (597, 598, 599)
    assert seen["extra"]["config_overrides"]["num_routed_experts"] == 6
    saved = json.loads((tmp_path / "train_config.json").read_text())
    assert saved["model_config"]["vocab_size"] == 600 and saved["config_overrides"]["qk_norm"] is True


def test_invalid_override_values_are_reported_not_a_traceback(tmp_path):
    script = _load_train_script()
    with pytest.raises(SystemExit, match="invalid config"):
        script.main(["--dry-run", "--steps", "1", "--out-dir", str(tmp_path), "--set", "step_ce_token_fraction=0.0"])


def test_two_step_dry_run_with_overrides_and_resume_honours_the_saved_config(tmp_path, capsys):
    script = _load_train_script()
    out = tmp_path / "run"
    script.main(["--dry-run", "--steps", "2", "--out-dir", str(out), "--set", "num_routed_experts=4",
                 "--set", "intermediate_size=128"])
    step2 = out / "checkpoints" / "step_2"
    ckpt_cfg = json.loads((step2 / "train_config.json").read_text())
    assert ckpt_cfg["model_config"]["intermediate_size"] == 128
    assert ckpt_cfg["config_overrides"] == {"num_routed_experts": 4, "intermediate_size": 128}
    assert json.loads((out / "train_config.json").read_text())["model_config"]["intermediate_size"] == 128
    assert InkyConfig.from_pretrained(step2 / "model").intermediate_size == 128

    # Resume with no overrides: the checkpoint's config is used (the dry-run default would be 96 and not load).
    script.main(["--dry-run", "--steps", "3", "--out-dir", str(out), "--resume", str(step2)])
    step3 = out / "checkpoints" / "step_3"
    assert InkyConfig.from_pretrained(step3 / "model").intermediate_size == 128

    # Resume with a conflicting override: refused, naming the field.
    with pytest.raises(SystemExit, match="would change the model config.*intermediate_size"):
        script.main(["--dry-run", "--steps", "4", "--out-dir", str(out), "--resume", str(step3),
                     "--set", "intermediate_size=96"])
    # A matching override is fine.
    script.main(["--dry-run", "--steps", "4", "--out-dir", str(out), "--resume", str(step3),
                 "--set", "intermediate_size=128"])
    assert (out / "checkpoints" / "step_4").is_dir()
    capsys.readouterr()


# ----------------------------------------------------------------------------- 4. vocabulary check


def test_vocab_mismatch_error_tells_how_to_fix_it(capsys):
    script = _load_train_script()
    config = InkyConfig()                           # vocab_size 50003, scratch ids 50000..50002
    with pytest.raises(SystemExit) as info:
        script.check_tokenizer_vocab(config, 32000)
    message = str(info.value)
    assert "--set vocab_size=32003" in message
    assert "tokenizer size plus 3 scratch ids" in message
    assert "appended after the tokenizer ids" in message
    assert script.check_tokenizer_vocab(config, 50000) == []


def test_vocab_check_modes_and_scratch_collisions():
    script = _load_train_script()
    config = InkyConfig()
    assert len(script.check_tokenizer_vocab(config, 32000, mode="warn")) == 1   # size only; scratch ids are in range
    short = tiny_config(vocab_size=600, scratch_open_id=10, scratch_sep_id=11, scratch_close_id=12)
    problems = script.check_tokenizer_vocab(short, 597, mode="warn")
    assert len(problems) == 1 and "collide with real tokens" in problems[0]
    with pytest.raises(SystemExit):
        script.check_tokenizer_vocab(short, 597)


def test_real_run_uses_the_check_and_set_vocab_size_fixes_it(monkeypatch, tmp_path):
    script = _load_train_script()
    corpus = tmp_path / "c.txt"
    corpus.write_text("x " * 200)

    class FakeTokenizer:
        def __len__(self):
            return 32000

        def encode(self, text, add_special_tokens=False):
            return [2 + (ord(c) % 100) for c in text]

    monkeypatch.setattr(script, "_load_tokenizer", lambda path: FakeTokenizer())
    base = ["--corpus", str(corpus), "--tokenizer-path", "tok", "--out-dir", str(tmp_path / "o")]
    with pytest.raises(SystemExit, match="--set vocab_size=32003"):
        script._build_setup(script.parse_args(base))
    config, _texts, _enc, _eos = script._build_setup(script.parse_args(base + ["--set", "vocab_size=32003"]))
    assert (config.vocab_size, config.scratch_open_id, config.scratch_close_id) == (32003, 32000, 32002)
    script._build_setup(script.parse_args(base + ["--tokenizer-vocab-check", "off"]))   # no exit


# ----------------------------------------------------------------------------- 5. throughput logging


def test_history_and_log_line_carry_tokens_per_second_and_tokens_seen():
    model = seeded_model()
    logs = []
    history = train(model, HybridOptimizer(model, adamw_lr=1e-3, muon_lr=0.01), windows(), steps=3, batch_size=2,
                    grad_accum_steps=2, log_every=1, log_fn=logs.append)
    records = [r for r in history if r["kind"] == "train"]
    assert [r["tokens_seen"] for r in records] == [2 * 2 * 16 * k for k in (1, 2, 3)]
    assert all(r["tokens_per_sec"] > 0 for r in records)
    lines = [m for m in logs if m.startswith("step ") and "loss" in m and "tok/s" in m]
    assert len(lines) == 3 and "tokens_seen 192" in lines[-1]


def test_tokens_seen_continues_across_a_resume(tmp_path):
    def run(steps, **kw):
        model = seeded_model()
        return train(model, HybridOptimizer(model, adamw_lr=1e-3, muon_lr=0.01), windows(), steps=steps,
                     batch_size=2, checkpoint_dir=tmp_path, checkpoint_every=2, log_fn=_quiet, **kw)

    first = run(2)
    assert first[-1]["tokens_seen"] == 2 * 2 * 16
    resumed = run(4, resume_from=tmp_path / "step_2")
    assert [r["tokens_seen"] for r in resumed if r["kind"] == "train"] == [32, 64, 96, 128]


def test_skipped_steps_still_report_throughput():
    model = seeded_model()
    real_forward = model.forward
    state = {"n": 0}

    def forward(*a, **kw):
        out = real_forward(*a, **kw)
        state["n"] += 1
        if state["n"] == 1:
            out.loss = out.loss * float("nan")
        return out

    model.forward = forward
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        history = train(model, HybridOptimizer(model, adamw_lr=1e-3, muon_lr=0.01), windows(), steps=2,
                        batch_size=2, log_fn=_quiet)
    first, second = [r for r in history if r["kind"] == "train"]
    assert first["skipped"] and first["tokens_per_sec"] > 0 and first["tokens_seen"] == 32
    assert not second["skipped"] and second["tokens_seen"] == 64
