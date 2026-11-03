"""Train Inky from text files.

    python scripts/train.py --corpus data/*.jsonl --tokenizer-path path/to/tokenizer --out-dir runs/inky
    python scripts/train.py --corpus book.txt --tokenizer-path path/to/tokenizer --resume runs/inky/checkpoints/step_500
    python scripts/train.py --dry-run --steps 3 --out-dir runs/dry

Pipeline: read the corpus, then data.prepare_corpus (clean, quality filter, dedupe, encode, pack into
windows), then inky.train.train. Checkpoints go to <out-dir>/checkpoints (step_N directories, best.pt).

Corpus: .jsonl files are read with --text-field (default "text"); any other file is one document.
The tokenizer is external to this repository. It is loaded lazily with transformers.AutoTokenizer from
--tokenizer-path, and it must agree with InkyConfig.vocab_size.

--dry-run needs neither a corpus nor a tokenizer. It builds a tiny config, generates a random-character
corpus with a character-level encoder, and runs the same prepare_corpus and train steps for a few steps.
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Iterable, Iterator, List, Optional

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


def _build_setup(args):
    """Return ``(config, texts, encode, eos_id)`` for a real run or for the dry run."""
    dropout = dict(embedding_dropout=args.embedding_dropout, residual_dropout=args.residual_dropout)
    if args.dry_run:
        config = dry_run_config(**dropout)
        encode = lambda text: [3 + ord(c) % (config.vocab_size - 4) for c in text]  # noqa: E731
        return config, _dry_run_texts(args.seed), encode, config.eos_token_id

    if not args.corpus:
        _fail("--corpus is required (or use --dry-run)")
    if not args.tokenizer_path:
        _fail("--tokenizer-path is required (or use --dry-run)")
    tokenizer = _load_tokenizer(args.tokenizer_path)
    config = InkyConfig(**dropout)
    if len(tokenizer) > config.vocab_size:
        _fail(f"tokenizer has {len(tokenizer)} tokens but InkyConfig.vocab_size is {config.vocab_size}")
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
    parser.add_argument("--batch-size", type=int, default=None, help="sequences per micro-batch (default 8, dry run 2)")
    parser.add_argument("--grad-accum", type=int, default=1, help="micro-batches per optimiser step (effective batch = batch-size * grad-accum)")
    parser.add_argument("--amp", choices=["none", "bf16", "fp16"], default="none", help="mixed precision for the forward pass (fp16 needs CUDA)")
    parser.add_argument("--eval-by-depth", action="store_true", help="at each eval, also log perplexity at every loop count in the training range")
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
    parser.add_argument("--eval-fraction", type=float, default=0.02, help="share of windows held out for eval")
    parser.add_argument("--ckpt-every", type=int, default=0, help="checkpoint every N steps (the last step is always saved)")
    parser.add_argument("--keep-last", type=int, default=3, help="checkpoints to keep (0 keeps all)")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--min-chars", type=int, default=200, help="quality filter: minimum document length")
    parser.add_argument("--resume", type=Path, default=None, help="a step_N checkpoint directory to resume from")
    parser.add_argument("--embedding-dropout", type=float, default=0.0,
                        help="dropout on the token embeddings, train mode only, in [0, 1) (default 0 = off)")
    parser.add_argument("--residual-dropout", type=float, default=0.0,
                        help="dropout on each block's branch outputs before the residual add, train mode only, "
                             "in [0, 1) (default 0 = off)")
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
    args.amp_dtype = None if args.amp == "none" else args.amp
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
    optimizer = HybridOptimizer(model, adamw_lr=args.lr, muon_lr=args.muon_lr)

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
