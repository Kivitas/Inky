"""Train Inky from text files.

    python scripts/train.py --corpus data/*.jsonl --tokenizer-path path/to/tokenizer --out-dir runs/inky
    python scripts/train.py --corpus book.txt --tokenizer-path path/to/tokenizer --resume runs/inky/checkpoints/step_500
    python scripts/train.py --dry-run --steps 3 --out-dir runs/dry

Pipeline: read the corpus, then data.prepare_corpus (clean, quality filter, dedupe, encode, pack into
windows), then inky.train.train. Checkpoints go to <out-dir>/checkpoints (step_N directories, best.pt).

Corpus: .jsonl files are read with --text-field (default "text"); any other file is one document.
The tokenizer is external to this repository. It is loaded lazily with transformers.AutoTokenizer from
--tokenizer-path, and it must agree with InkyConfig.vocab_size.

Config overrides (any InkyConfig field, so ablation variants launch from the command line):
    --set use_moe=False --set vocab_size=32003 --set qk_norm=True      (repeatable; values are Python literals)
    --config-json variant.json                                         (a JSON object of the same keys)
--set wins over --config-json, which wins over the dropout flags. Unknown keys are rejected. If vocab_size is
overridden without the scratch ids, they follow it as vocab_size-3, -2, -1 (they must sit after the tokenizer
ids). The config used is saved in train_config.json (model_config and config_overrides), and --resume reuses the
checkpoint's config: an override that differs from it is an error, never a silent change.

--compile wraps the training forward in torch.compile and falls back to eager (with a warning) if that fails.
--muon-batched runs Newton-Schulz once per matrix shape instead of once per matrix (same results, faster).

--dry-run needs neither a corpus nor a tokenizer. It builds a tiny config, generates a random-character
corpus with a character-level encoder, and runs the same prepare_corpus and train steps for a few steps.
"""

import argparse
import ast
import difflib
import inspect
import json
import math
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from inky import HybridOptimizer, InkyConfig, InkyModel  # noqa: E402
from inky.data import iter_jsonl, prepare_corpus  # noqa: E402
from inky.evaluation import split_heldout  # noqa: E402
from inky.train import train  # noqa: E402

DRY_RUN_VOCAB = 512


def dry_run_config(**overrides) -> InkyConfig:
    """A few-thousand-parameter-scale config that runs on a CPU in seconds. ``overrides`` go to InkyConfig."""
    kwargs = dict(
        vocab_size=DRY_RUN_VOCAB,
        scratch_open_id=DRY_RUN_VOCAB - 3,
        scratch_sep_id=DRY_RUN_VOCAB - 2,
        scratch_close_id=DRY_RUN_VOCAB - 1,
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


def _fail(message: str) -> None:
    sys.exit(f"error: {message}")


class ConfigOverrideError(ValueError):
    """A --set / --config-json problem, reported through the argument parser."""


SCRATCH_ID_KEYS = ("scratch_open_id", "scratch_sep_id", "scratch_close_id")
_LITERAL_WORDS = {"true": True, "false": False, "none": None, "null": None}


def config_fields() -> Dict[str, Any]:
    """Every InkyConfig constructor field with its default, read from the signature so new fields just work."""
    params = inspect.signature(InkyConfig.__init__).parameters
    return {
        name: param.default
        for name, param in params.items()
        if name != "self" and param.kind not in (param.VAR_KEYWORD, param.VAR_POSITIONAL)
    }


def _coerce_override(key: str, value: Any, default: Any) -> Any:
    """Check ``value`` against the type of the field's default (when it has one) and return it, normalised."""
    if default is None or default is inspect.Parameter.empty:
        return value
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigOverrideError(f"{key} expects a bool (True or False), got {value!r}")
    elif isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigOverrideError(f"{key} expects an int, got {value!r}")
    elif isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigOverrideError(f"{key} expects a number, got {value!r}")
        value = float(value)
    elif isinstance(default, str):
        if not isinstance(value, str):
            raise ConfigOverrideError(f"{key} expects a string, got {value!r}")
    elif isinstance(default, (tuple, list)):
        if not isinstance(value, (tuple, list)):
            raise ConfigOverrideError(f"{key} expects a list or tuple, got {value!r}")
        value = type(default)(value)
    return value


def validate_overrides(raw: Dict[str, Any], source: str) -> Dict[str, Any]:
    """Reject unknown keys and wrongly typed values; return the normalised overrides."""
    fields = config_fields()
    out: Dict[str, Any] = {}
    for key, value in raw.items():
        if key not in fields:
            near = difflib.get_close_matches(key, list(fields), n=3, cutoff=0.6)
            hint = f" Did you mean {', '.join(near)}?" if near else ""
            raise ConfigOverrideError(f"{source}: unknown InkyConfig field {key!r}.{hint}")
        try:
            out[key] = _coerce_override(key, value, fields[key])
        except ConfigOverrideError as exc:
            raise ConfigOverrideError(f"{source}: {exc}") from None
    return out


def parse_set_items(items: Iterable[str]) -> Dict[str, Any]:
    """Parse ``key=value`` strings. Values go through ast.literal_eval (true/false/none are also accepted)."""
    raw: Dict[str, Any] = {}
    for item in items:
        key, sep, text = item.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ConfigOverrideError(f"--set expects key=value, got {item!r}")
        lowered = text.strip().lower()
        if lowered in _LITERAL_WORDS:
            value = _LITERAL_WORDS[lowered]
        else:
            try:
                value = ast.literal_eval(text.strip())
            except (ValueError, SyntaxError):
                if isinstance(config_fields().get(key), str):
                    value = text.strip()      # string-valued field: a bare word such as mor is fine
                else:
                    raise ConfigOverrideError(
                        f"--set {key}: cannot read {text!r} as a Python literal (strings need quotes)"
                    ) from None
        raw[key] = value
    return validate_overrides(raw, "--set")


def load_config_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object of config overrides. A ``model_config`` object (a train_config.json) is unwrapped."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ConfigOverrideError(f"--config-json: cannot read {str(path)!r}: {exc}") from None
    if not isinstance(data, dict):
        raise ConfigOverrideError(f"--config-json: {str(path)!r} must hold a JSON object")
    if "model_config" in data and "model_config" not in config_fields() and isinstance(data["model_config"], dict):
        data = data["model_config"]
    return validate_overrides(data, f"--config-json {path}")


def derive_scratch_ids(overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Overrides plus scratch ids following ``vocab_size`` (its last three ids) when it is set and they are not."""
    out = dict(overrides)
    if "vocab_size" in out and not any(k in out for k in SCRATCH_ID_KEYS):
        v = int(out["vocab_size"])
        out.update(scratch_open_id=v - 3, scratch_sep_id=v - 2, scratch_close_id=v - 1)
    return out


def check_tokenizer_vocab(config: InkyConfig, tokenizer_size: int, mode: str = "error") -> List[str]:
    """Problems between the tokenizer and the config's vocabulary layout. Mode "error" exits on any."""
    n = tokenizer_size
    problems: List[str] = []
    if config.vocab_size != n + 3:
        problems.append(
            f"tokenizer has {n} ids but InkyConfig.vocab_size is {config.vocab_size}; vocab_size must be the "
            f"tokenizer size plus 3 scratch ids = {n + 3}. The scratch ids must be appended after the tokenizer ids. "
            f"Fix: pass --set vocab_size={n + 3} (the scratch ids then default to {n}, {n + 1}, {n + 2})."
        )
    bad = [f"{k}={getattr(config, k)}" for k in SCRATCH_ID_KEYS if not n <= getattr(config, k) < config.vocab_size]
    if bad:
        problems.append(
            f"scratch ids must be appended after the tokenizer ids (>= {n}) and below vocab_size "
            f"({config.vocab_size}); got {', '.join(bad)}, which collide with real tokens."
        )
    if problems and mode == "error":
        _fail(" ".join(problems) + " (--tokenizer-vocab-check warn or off skips this check)")
    return problems


def compile_supported_for_dry_run() -> bool:
    """torch.compile on CPU needs a C++ compiler and is not usable on Windows here, so --dry-run skips it there."""
    if sys.platform == "win32":
        return False
    return any(shutil.which(cc) for cc in ("g++", "gcc", "clang", "cc"))


def _load_tokenizer(path: str):
    """Load the external tokenizer. transformers is imported here so that --dry-run never needs it."""
    try:
        from transformers import AutoTokenizer
    except ImportError:
        _fail("loading a tokenizer needs the 'transformers' package (pip install transformers)")
    try:
        return AutoTokenizer.from_pretrained(path)
    except Exception as exc:  # any load failure is reported with the path
        _fail(f"cannot load tokenizer from {path!r}: {exc}")


def _iter_corpus_texts(paths: List[str], text_field: str) -> Iterator[str]:
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            _fail(f"corpus file not found: {path}")
        if path.suffix.lower() == ".jsonl":
            yield from iter_jsonl(path, field=text_field)
        else:
            yield path.read_text(encoding="utf-8")


def _dry_run_texts(seed: int, n_docs: int = 40) -> Iterable[str]:
    """Random-character documents long enough to pass the default min_chars filter."""
    rng = random.Random(seed)
    alphabet = "abcdefghijklmnopqrstuvwxyz     "
    for _ in range(n_docs):
        length = rng.randint(260, 400)
        yield "".join(rng.choice(alphabet) for _ in range(length)).strip()


def _config_kwargs(args) -> Dict[str, Any]:
    """Dropout flags, then --config-json / --set overrides (later wins), with scratch ids following vocab_size."""
    kwargs = dict(embedding_dropout=args.embedding_dropout, residual_dropout=args.residual_dropout)
    kwargs.update(derive_scratch_ids(getattr(args, "config_overrides", {})))
    return kwargs


def _make_config(args) -> InkyConfig:
    kwargs = _config_kwargs(args)
    try:
        return dry_run_config(**kwargs) if args.dry_run else InkyConfig(**kwargs)
    except (TypeError, ValueError, AssertionError) as exc:
        _fail(f"invalid config: {exc}")


def _resume_config(args, requested: InkyConfig) -> InkyConfig:
    """The config of the checkpoint being resumed. Explicit overrides that disagree with it are an error."""
    model_dir = Path(args.resume) / "model"
    try:
        saved = InkyConfig.from_pretrained(model_dir)
    except Exception as exc:  # noqa: BLE001 - any unreadable checkpoint config is reported with its path
        _fail(f"cannot read the checkpoint config in {model_dir}: {exc}")
    explicit = dict(getattr(args, "config_overrides", {}))
    for key in ("embedding_dropout", "residual_dropout"):
        if getattr(args, key) != 0.0:
            explicit.setdefault(key, getattr(args, key))
    diffs = [
        f"{key}: checkpoint {getattr(saved, key, None)!r} vs requested {getattr(requested, key, None)!r}"
        for key in explicit
        if getattr(saved, key, None) != getattr(requested, key, None)
    ]
    if diffs:
        _fail(
            "--resume would change the model config ("
            + "; ".join(diffs)
            + "). Drop those overrides to continue with the checkpoint's config, or start a new run."
        )
    return saved


def _build_setup(args):
    """Return ``(config, texts, encode, eos_id)`` for a real run or for the dry run."""
    config = _make_config(args)
    if args.resume is not None:
        config = _resume_config(args, config)
    if args.dry_run:
        encode = lambda text: [3 + ord(c) % (config.vocab_size - 4) for c in text]  # noqa: E731
        return config, _dry_run_texts(args.seed), encode, config.eos_token_id

    if not args.corpus:
        _fail("--corpus is required (or use --dry-run)")
    if not args.tokenizer_path:
        _fail("--tokenizer-path is required (or use --dry-run)")
    tokenizer = _load_tokenizer(args.tokenizer_path)
    if args.tokenizer_vocab_check != "off":
        problems = check_tokenizer_vocab(config, len(tokenizer), args.tokenizer_vocab_check)
        for problem in problems:
            print(f"warning: {problem}")
    texts = _iter_corpus_texts(args.corpus, args.text_field)
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)  # noqa: E731
    return config, texts, encode, config.eos_token_id


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="tiny config and random corpus; no tokenizer needed")
    parser.add_argument("--corpus", nargs="+", help="one or more .jsonl or .txt files")
    parser.add_argument("--text-field", default="text", help="JSONL field holding the document text")
    parser.add_argument("--tokenizer-path", help="directory or name readable by transformers.AutoTokenizer")
    parser.add_argument("--out-dir", type=Path, default=None, help="run directory (checkpoints go under it)")
    parser.add_argument("--steps", type=int, default=1000, help="total optimiser steps")
    parser.add_argument(
        "--batch-size", type=int, default=None, help="sequences per micro-batch (default 8, dry run 2)"
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=1,
        help="micro-batches per optimiser step (effective batch = batch-size * grad-accum)",
    )
    parser.add_argument(
        "--amp",
        choices=["none", "bf16", "fp16"],
        default="none",
        help="mixed precision for the forward pass (fp16 needs CUDA)",
    )
    parser.add_argument(
        "--eval-by-depth",
        action="store_true",
        help="at each eval, also log perplexity at every loop count in the training range",
    )
    parser.add_argument("--seq-len", type=int, default=None, help="tokens per window (default 1024, dry run 32)")
    parser.add_argument("--lr", type=float, default=3e-3, help="AdamW learning rate (base, before warmup/decay)")
    parser.add_argument("--muon-lr", type=float, default=0.02, help="Muon learning rate (base)")
    parser.add_argument("--warmup-steps", type=int, default=None, help="default: 5%% of --steps, at least 1")
    parser.add_argument("--decay-fraction", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loop-curriculum", action="store_true", help="ramp the recurrent loop count up")
    parser.add_argument("--max-skipped-steps", type=int, default=50, help="consecutive skipped steps before abort")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=0, help="eval every N steps (0 = no eval)")
    parser.add_argument("--diagnose-every", type=int, default=0,
                        help="every N steps log stream RMS per loop, loop-to-loop change, highway gain and expert "
                             "load (one extra eval-mode forward pass; 0 = off)")
    parser.add_argument("--early-stop-patience", type=int, default=None,
                        help="stop after N consecutive evals without a new best eval NLL (needs --eval-every > 0; "
                             "default off)")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0,
                        help="an eval counts as an improvement only if its NLL is below the best by more than this "
                             "(default 0.0)")
    parser.add_argument("--eval-fraction", type=float, default=0.02, help="share of windows held out for eval")
    parser.add_argument(
        "--ckpt-every", type=int, default=0, help="checkpoint every N steps (the last step is always saved)"
    )
    parser.add_argument("--keep-last", type=int, default=3, help="checkpoints to keep (0 keeps all)")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--min-chars", type=int, default=200, help="quality filter: minimum document length")
    parser.add_argument("--resume", type=Path, default=None, help="a step_N checkpoint directory to resume from")
    parser.add_argument("--embedding-dropout", type=float, default=0.0,
                        help="dropout on the token embeddings, train mode only, in [0, 1) (default 0 = off)")
    parser.add_argument("--residual-dropout", type=float, default=0.0,
                        help="dropout on each block's branch outputs before the residual add, train mode only, "
                             "in [0, 1) (default 0 = off)")
    parser.add_argument("--set", dest="set_items", action="append", default=[], metavar="KEY=VALUE",
                        help="override an InkyConfig field, e.g. --set use_moe=False --set vocab_size=32003 "
                             "(repeatable; the value is a Python literal)")
    parser.add_argument("--config-json", type=Path, default=None, metavar="PATH",
                        help="JSON object of InkyConfig overrides (--set wins over it)")
    parser.add_argument("--tokenizer-vocab-check", choices=["error", "warn", "off"], default="error",
                        help="when the tokenizer size + 3 scratch ids differs from config.vocab_size: exit (default), "
                             "warn, or skip the check")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the training forward; falls back to eager with a warning on failure")
    parser.add_argument("--muon-batched", action="store_true",
                        help="batch the Muon Newton-Schulz step over same-shape matrices (same results, faster)")
    args = parser.parse_args(argv)

    if args.out_dir is None:
        parser.error("--out-dir is required")
    for flag, value in (("--embedding-dropout", args.embedding_dropout),
                        ("--residual-dropout", args.residual_dropout)):
        if not 0.0 <= value < 1.0:
            parser.error(f"{flag} must be in [0, 1), got {value}")
    if args.dry_run:
        args.batch_size = args.batch_size or 2
        args.seq_len = args.seq_len or 32
        args.eval_every = args.eval_every or 1
        args.ckpt_every = args.ckpt_every or 1
    else:
        args.batch_size = args.batch_size or 8
        args.seq_len = args.seq_len or 1024
    if args.steps < 1:
        parser.error("--steps must be >= 1")
    if args.grad_accum < 1:
        parser.error("--grad-accum must be >= 1")
    if args.diagnose_every < 0:
        parser.error("--diagnose-every must be >= 0")
    if args.early_stop_patience is not None:
        if args.early_stop_patience < 1:
            parser.error("--early-stop-patience must be >= 1")
        if args.eval_every <= 0:
            parser.error("--early-stop-patience needs --eval-every > 0")
    if not (math.isfinite(args.early_stop_min_delta) and args.early_stop_min_delta >= 0):
        parser.error("--early-stop-min-delta must be a finite number >= 0")
    args.amp_dtype = None if args.amp == "none" else args.amp
    try:
        overrides = load_config_json(args.config_json) if args.config_json is not None else {}
        overrides.update(parse_set_items(args.set_items))
    except ConfigOverrideError as exc:
        parser.error(str(exc))
    args.config_overrides = overrides
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    config, texts, encode, eos_id = _build_setup(args)
    if args.seq_len < 2 or args.seq_len > config.max_position_embeddings:
        _fail(f"--seq-len must be in [2, {config.max_position_embeddings}], got {args.seq_len}")

    if args.dry_run:
        print(f"dry run: tiny config (vocab {config.vocab_size}), random-character corpus, out-dir {args.out_dir}")
    windows = prepare_corpus(texts, encode, args.seq_len, eos_id, min_chars=args.min_chars)
    print(f"corpus: {windows.shape[0]} windows of {args.seq_len} tokens")
    if windows.shape[0] == 0:
        _fail("the corpus produced no full windows; add more text or lower --seq-len")

    train_windows, eval_batches = windows, None
    if args.eval_every > 0:
        if windows.shape[0] < 2:
            _fail("eval needs at least 2 windows, so the held-out split leaves some for training")
        train_idx, eval_idx = split_heldout(list(range(windows.shape[0])), fraction=args.eval_fraction, seed=args.seed)
        train_windows = windows[train_idx]
        held = windows[eval_idx]
        eval_batches = [held[i:i + args.batch_size] for i in range(0, held.shape[0], args.batch_size)]
        print(f"split: {len(train_idx)} train windows, {len(eval_idx)} held out for eval")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.dry_run else "cpu")
    torch.manual_seed(args.seed)
    model = InkyModel(config).to(device)
    print(f"model: {model.count_parameters()['total'] / 1e6:.2f} M parameters on {device}")
    optimizer = HybridOptimizer(model, adamw_lr=args.lr, muon_lr=args.muon_lr, muon_batched=args.muon_batched)

    compile_model = args.compile
    if compile_model and args.dry_run and not compile_supported_for_dry_run():
        print("--compile skipped for --dry-run: torch.compile is not usable here (needs a C++ compiler, not Windows)")
        compile_model = False
    args.out_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model_config": config.to_dict(),
        "config_overrides": dict(args.config_overrides),
        "resumed_from": str(args.resume) if args.resume is not None else None,
    }
    (args.out_dir / "train_config.json").write_text(json.dumps(run_config, indent=2, sort_keys=True, default=str))

    history = train(
        model,
        optimizer,
        train_windows,
        steps=args.steps,
        batch_size=args.batch_size,
        eval_batches=eval_batches,
        eval_every=args.eval_every,
        warmup_steps=args.warmup_steps,
        decay_fraction=args.decay_fraction,
        max_grad_norm=args.max_grad_norm,
        use_loop_curriculum=args.loop_curriculum,
        log_every=args.log_every,
        checkpoint_dir=args.out_dir / "checkpoints",
        checkpoint_every=args.ckpt_every,
        keep_last=args.keep_last,
        max_skipped_steps=args.max_skipped_steps,
        resume_from=args.resume,
        seed=args.seed,
        grad_accum_steps=args.grad_accum,
        amp_dtype=args.amp_dtype,
        eval_by_depth=args.eval_by_depth,
        diagnose_every=args.diagnose_every,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        compile_model=compile_model,
        extra_config={"config_overrides": dict(args.config_overrides), "muon_batched": args.muon_batched},
    )

    train_records = [r for r in history if r.get("kind") == "train"]
    eval_records = [r for r in history if r.get("kind") == "eval"]
    if train_records:
        last = train_records[-1]
        print(f"finished at step {last['step']}: loss {last['loss']:.4f}, ce {last['ce']:.4f}")
    if eval_records:
        best = min(eval_records, key=lambda r: r["eval_nll"])
        print(f"best eval nll {best['eval_nll']:.4f} at step {best['step']}")
    print(f"checkpoints in {args.out_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
