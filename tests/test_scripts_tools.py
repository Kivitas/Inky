"""Tests for scripts/bench.py, scripts/ablate.py and scripts/sample.py. Everything runs on CPU with tiny models."""

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest
import torch

from inky import InkyConfig, InkyModel

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_inky_script_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


bench = _load("bench")
ablate = _load("ablate")
sample = _load("sample")


# ----------------------------------------------------------------------------- bench

def test_bench_tiny_writes_json_and_restores_threads(tmp_path, capsys):
    threads_before = torch.get_num_threads()
    out = tmp_path / "bench.json"
    result = bench.main(["--tiny", "--context-lengths", "8,16", "--decode-tokens", "3", "--repeats", "2",
                         "--num-loops", "2", "--threads", "1", "--json", str(out)])
    assert torch.get_num_threads() == threads_before
    assert [r["context"] for r in result["results"]] == [8, 16]
    for row in result["results"]:
        assert row["prefill_s"] > 0 and row["prefill_tokens_per_s"] > 0
        assert row["decode_tokens"] == 3 and row["decode_ms_per_token"] > 0 and row["decode_tokens_per_s"] > 0
    assert result["num_loops"] == 2 and result["threads"] == 1
    assert json.loads(out.read_text())["params"]["total"] == result["params"]["total"]
    assert "| context |" in capsys.readouterr().out


def test_bench_set_overrides_and_skips_long_contexts():
    base = bench.main(["--tiny", "--context-lengths", "8", "--decode-tokens", "1", "--repeats", "1"])
    more = bench.main(["--tiny", "--set", "num_routed_experts=8", "--set", "max_position_embeddings=64",
                       "--context-lengths", "8,100", "--decode-tokens", "1", "--repeats", "1"])
    assert more["params"]["total"] > base["params"]["total"]
    assert [r["context"] for r in more["results"]] == [8]
    assert more["skipped"][0]["context"] == 100


def test_parse_overrides_literals_strings_and_unknown_keys():
    for module in (bench, ablate):
        parsed = module.parse_overrides(["qk_norm=True", "hidden_size=32", "rope_theta=1e4"])
        assert parsed == {"qk_norm": True, "hidden_size": 32, "rope_theta": 1e4}
        with pytest.raises(ValueError, match="no such parameter"):
            module.parse_overrides(["definitely_not_a_key=1"])
        with pytest.raises(ValueError, match="key=value"):
            module.parse_overrides(["hidden_size"])


def test_bench_rejects_bad_set_key():
    with pytest.raises(SystemExit):
        bench.main(["--tiny", "--set", "definitely_not_a_key=1"])


# ----------------------------------------------------------------------------- ablate

TINY_RUN = ["--synthetic", "--tiny", "--steps", "3", "--batch-size", "2", "--seq-len", "16",
            "--synthetic-windows", "16", "--eval-fraction", "0.25"]


def test_ablate_synthetic_end_to_end(tmp_path):
    names = "baseline,dense,no_highway,qk_norm,qk_norm_latent,fixed_depth"
    result = ablate.main([*TINY_RUN, "--variants", names, "--out-dir", str(tmp_path)])
    by_name = {r["variant"]: r for r in result["variants"]}
    assert list(by_name) == names.split(",")
    keys = ablate.config_keys()
    expected_skips = {"dense": "use_moe", "no_highway": "use_highway", "qk_norm_latent": "qk_norm_mode"}
    for name, key in expected_skips.items():
        if key in keys:
            assert by_name[name]["status"] == "ok", name
        else:
            assert by_name[name]["status"] == "skipped" and key in by_name[name]["reason"], name
    for name in ("baseline", "qk_norm", "fixed_depth"):
        row = by_name[name]
        assert row["status"] == "ok" and row["steps_run"] == 3
        assert torch.isfinite(torch.tensor(row["final_train_ce"])) and row["eval_perplexity"] > 1
        assert row["tokens_per_s"] > 0 and row["params"] > 0
    base = by_name["baseline"]["dense_settings"]
    assert base["dense_intermediate_total_match"] == (1 + 4) * 96      # tiny: 1 shared + 4 routed experts, width 96
    assert base["dense_intermediate_active_match"] == (1 + 2) * 96
    assert base["active_params_est"] < base["total_params"]
    assert by_name["fixed_depth"]["overrides"] == {"min_recurrent_loops": 4}
    saved = json.loads((tmp_path / "ablation.json").read_text())
    assert [r["variant"] for r in saved["variants"]] == names.split(",")
    table = (tmp_path / "ablation.md").read_text()
    assert "| baseline | ok |" in table and "| variant |" in table


def test_ablate_identical_seeds_are_reproducible():
    args = [*TINY_RUN, "--variants", "baseline", "--steps", "2"]
    a = ablate.main(args)["variants"][0]
    b = ablate.main(args)["variants"][0]
    assert a["eval_nll"] == b["eval_nll"] and a["final_train_ce"] == b["final_train_ce"]


def test_ablate_vocab_variant_uses_scratch_ids_and_skips_when_eos_does_not_fit():
    cfg = InkyConfig(**{**ablate.TINY_KWARGS, **ablate.resolve_variant("vocab32k", ablate.TINY_KWARGS)})
    assert (cfg.vocab_size, cfg.scratch_open_id, cfg.scratch_sep_id, cfg.scratch_close_id) == (
        32003, 32000, 32001, 32002)
    assert cfg.eos_token_id < cfg.scratch_open_id
    # A base EOS id above the 32k vocabulary cannot be shared with the data, so the variant is skipped.
    big = ["--set", "vocab_size=50003", "--set", "scratch_open_id=50000", "--set", "scratch_sep_id=50001",
           "--set", "scratch_close_id=50002", "--set", "eos_token_id=40000", "--set", "pad_token_id=40000"]
    result = ablate.main([*TINY_RUN, "--variants", "vocab32k", "--steps", "1", *big])
    assert result["variants"][0]["status"] == "skipped" and "EOS" in result["variants"][0]["reason"]


def test_ablate_relative_variants_follow_set_overrides():
    base = {"max_recurrent_loops": 4, "min_recurrent_loops": 2, "num_routed_experts": 4}
    assert ablate.resolve_variant("fewer_loops", base) == {"max_recurrent_loops": 2, "min_recurrent_loops": 2}
    assert ablate.resolve_variant("more_experts", base) == {"num_routed_experts": 8}
    assert ablate.resolve_variant("fixed_depth", base) == {"min_recurrent_loops": 4}


def test_ablate_list_variants_and_unknown_variant(capsys):
    assert ablate.main(["--list-variants"]) is None
    assert "baseline" in capsys.readouterr().out.split()
    with pytest.raises(SystemExit):
        ablate.main([*TINY_RUN, "--variants", "no_such_variant"])
    with pytest.raises(SystemExit):
        ablate.main(["--tiny"])          # neither --synthetic nor a corpus


def test_ablate_corpus_mode_with_a_tokenizers_json(tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers

    rng = random.Random(0)
    words = [f"w{i}" for i in range(40)]
    tok = Tokenizer(models.WordLevel({w: i + 4 for i, w in enumerate(words)}, unk_token="w0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok_path = tmp_path / "tokenizer.json"
    tok.save(str(tok_path))
    corpus = tmp_path / "docs.jsonl"
    with corpus.open("w", encoding="utf-8") as f:
        for _ in range(80):
            f.write(json.dumps({"text": " ".join(rng.choice(words) for _ in range(120))}) + "\n")
    result = ablate.main(["--corpus", str(corpus), "--tokenizer-path", str(tmp_path), "--tiny", "--steps", "2",
                          "--batch-size", "2", "--seq-len", "16", "--min-chars", "20", "--variants", "baseline"])
    row = result["variants"][0]
    assert row["status"] == "ok" and row["eval_nll"] > 0 and result["data"].startswith("corpus")


# ----------------------------------------------------------------------------- sample

def test_sample_dry_run(capsys):
    text = sample.main(["--dry-run", "--max-new-tokens", "6", "--seed", "1"])
    assert isinstance(text, str)
    assert capsys.readouterr().out.startswith("Once upon a time")
    assert sample.main(["--dry-run", "--max-new-tokens", "6", "--seed", "1"]) == text      # seeded


def test_sample_dry_run_with_every_control():
    text = sample.main(["--dry-run", "--max-new-tokens", "8", "--temperature", "0.7", "--top-p", "0.9",
                        "--top-k", "20", "--repetition-penalty", "1.2", "--no-repeat-ngram-size", "3",
                        "--repeat-window", "32", "--bad-words", "xyz,qq", "--default-bad-words",
                        "--reduce-scratch-blocks", "--num-loops", "2"])
    assert isinstance(text, str)
    assert isinstance(sample.main(["--dry-run", "--greedy", "--max-new-tokens", "4"]), str)


def test_sample_rejects_ngram_size_one_and_missing_args():
    with pytest.raises(SystemExit):
        sample.main(["--dry-run", "--no-repeat-ngram-size", "1"])
    with pytest.raises(SystemExit):
        sample.main([])


def test_char_codec_round_trips():
    codec = sample.CharCodec()
    assert codec.decode(codec.encode("Hello, world! é")) == "Hello, world! é"
    assert max(codec.encode("zz€")) < sample.TINY_VOCAB - 3


def test_bad_words_are_encoded_with_variants():
    codec = sample.CharCodec()
    ids = sample.build_bad_words(codec, "foo, bar", use_default=False)
    assert ids and codec.encode("foo") in ids and codec.encode(" bar") in ids
    assert sample.build_bad_words(codec, None, use_default=False) is None
    assert len(sample.build_bad_words(codec, None, use_default=True)) > 0


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("ckpt")
    torch.manual_seed(0)
    model = InkyModel(InkyConfig(**sample.TINY_KWARGS)).eval()
    model.save_pretrained(root / "pretrained")
    model.save_pretrained(root / "checkpoints" / "step_3" / "model")
    model.save_pretrained(root / "checkpoints" / "step_10" / "model")
    torch.save({"model": model.state_dict(), "config": model.config.to_dict(), "step": 3, "eval_nll": 1.0,
                "eval_perplexity": 2.7}, root / "best.pt")
    return root, model


def test_load_model_accepts_every_checkpoint_layout(tiny_checkpoint):
    root, model = tiny_checkpoint
    reference = model.state_dict()
    for path in (root / "pretrained", root / "checkpoints" / "step_10", root / "best.pt", root,
                 root / "checkpoints"):
        loaded = sample.load_model(str(path))
        assert not loaded.training
        for key, value in reference.items():
            assert torch.equal(loaded.state_dict()[key], value), (path, key)
    with pytest.raises(SystemExit):
        sample.load_model(str(root / "missing"))
    (root / "empty").mkdir(exist_ok=True)
    with pytest.raises(SystemExit):
        sample.load_model(str(root / "empty"))


def test_sample_from_checkpoint_and_tokenizers_json(tiny_checkpoint, tmp_path, capsys):
    from tokenizers import Tokenizer, models, pre_tokenizers

    root, _ = tiny_checkpoint
    words = ["the", "keeper", "said", "lamp", "sea", "night", "light", "old"]
    tok = Tokenizer(models.WordLevel({w: i + 4 for i, w in enumerate(words)}, unk_token="the"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok_file = tmp_path / "tokenizer.json"
    tok.save(str(tok_file))
    for tokenizer_arg in (tok_file, tmp_path):                   # the JSON file, or the directory that holds it
        text = sample.main(["--checkpoint", str(root / "pretrained"), "--tokenizer-path", str(tokenizer_arg),
                            "--prompt", "the keeper said", "--max-new-tokens", "5", "--seed", "3",
                            "--bad-words", "sea"])
        assert isinstance(text, str)
        assert capsys.readouterr().out.startswith("the keeper said")
    big = Tokenizer(models.WordLevel({f"w{i}": i for i in range(600)}, unk_token="w0"))
    big.pre_tokenizer = pre_tokenizers.Whitespace()
    big_file = tmp_path / "big.json"
    big.save(str(big_file))
    with pytest.raises(SystemExit, match="vocabulary"):
        sample.main(["--checkpoint", str(root / "pretrained"), "--tokenizer-path", str(big_file)])
