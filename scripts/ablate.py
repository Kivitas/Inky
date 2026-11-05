"""Ablation runner for the table in docs/TRAINING_PLAN.md (section 4).

    python scripts/ablate.py --synthetic --tiny --steps 20 --out-dir runs/ablate
    python scripts/ablate.py --corpus data/*.jsonl --tokenizer-path path/to/tokenizer.json --steps 500 \
        --variants baseline,dense,no_highway,qk_norm --out-dir runs/ablate
    python scripts/ablate.py --list-variants

Each variant is a dict of InkyConfig overrides applied on top of the base config (the default InkyConfig, or the
tiny one with --tiny, plus any ``--set key=value``). For every variant the script builds the model with the same
seed, reports its parameter count, trains it for ``--steps`` steps with ``inky.train.train`` on the same
windows with the same seed, then scores the same held-out windows. The result is a markdown table (printed, and
written with --markdown or into --out-dir) and a JSON file (--json or --out-dir).

Variants are built lazily. A variant whose override keys InkyConfig does not accept yet (for example ``use_moe``
before the config grows it) is skipped with a message and listed as skipped in the table, never run with the key
silently ignored. A variant that cannot use the chosen data (for example a 32k vocabulary against a 50k tokenizer)
is skipped the same way.

Data: --synthetic generates a small seeded random-token corpus with a learnable bigram structure, so the script
needs no files. Otherwise give --corpus (.jsonl with --text-field, or plain text files) and --tokenizer-path (a
tokenizers JSON file, a directory holding tokenizer.json, or anything transformers.AutoTokenizer can load).

"Parameter-matched dense settings" are analytic: with S shared and R routed experts of width I (SwiGLU, 3*h*I
parameters each), a dense FFN of width (S+R)*I matches the MoE's total FFN parameters and (S+A)*I matches the
per-token active FFN parameters (A = active routed experts). The router is ignored. Rows 3 and 8 of the plan are
covered by the vocabulary and fixed-depth variants; optimiser, halting and scratch rows are not config toggles
and are out of scope here.
"""

import argparse
import ast
import inspect
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from inky import HybridOptimizer, InkyConfig, InkyModel  # noqa: E402
from inky.data import iter_jsonl, prepare_corpus  # noqa: E402
from inky.evaluation import evaluate, split_heldout  # noqa: E402
from inky.train import train  # noqa: E402

TINY_VOCAB = 512
TINY_KWARGS: Dict[str, Any] = dict(
    vocab_size=TINY_VOCAB, scratch_open_id=TINY_VOCAB - 3, scratch_sep_id=TINY_VOCAB - 2,
    scratch_close_id=TINY_VOCAB - 1, hidden_size=64, num_attention_heads=4, qk_nope_head_dim=12,
    qk_rope_head_dim=4, v_head_dim=16, q_lora_rank=32, kv_lora_rank=24, intermediate_size=96,
    num_routed_experts=4, num_active_routed=2, ssm_state_dim=16, max_recurrent_loops=4, min_recurrent_loops=2,
    loop_dropout_prob=0.0, max_position_embeddings=256, arlo_dim=16, arlo_pointer_rank=8, arlo_slots=4,
    eos_token_id=1, pad_token_id=1,
)

Overrides = Dict[str, Any]


def _vocab_variant(size: int) -> Callable[[Overrides], Overrides]:
    """A vocabulary of ``size`` text ids plus 3 scratch ids. EOS/PAD move to the same offset as 49974 in 50003."""
    def build(base: Overrides) -> Overrides:
        total = size + 3
        return dict(vocab_size=total, scratch_open_id=size, scratch_sep_id=size + 1, scratch_close_id=size + 2,
                    eos_token_id=total - 29, pad_token_id=total - 29)
    return build


def _fewer_loops(base: Overrides) -> Overrides:
    top = max(1, int(base.get("max_recurrent_loops", 6)) - 2)
    return dict(max_recurrent_loops=top, min_recurrent_loops=min(int(base.get("min_recurrent_loops", 2)), top))


def _more_experts(base: Overrides) -> Overrides:
    return dict(num_routed_experts=2 * int(base.get("num_routed_experts", 8)))


def _fixed_depth(base: Overrides) -> Overrides:
    return dict(min_recurrent_loops=int(base.get("max_recurrent_loops", 6)))


def _big_experts(base: Overrides) -> Overrides:
    """Half as many routed experts, each twice as wide: same expert parameters and the same active width."""
    ff = int(base.get("intermediate_size", 704))
    routed = int(base.get("num_routed_experts", 8))
    active = int(base.get("num_active_routed", 2))
    return dict(shared_intermediate_size=ff, intermediate_size=2 * ff,
                num_routed_experts=max(2, routed // 2), num_active_routed=max(1, active // 2))


def _core_shape(layers: int) -> Callable[[Overrides], Overrides]:
    """``layers`` unique recurrent layers with fewer loops, at about the same number of layer applications."""
    def build(base: Overrides) -> Overrides:
        base_layers = int(base.get("num_recurrent_layers", 2))
        base_loops = int(base.get("max_recurrent_loops", 6))
        applications = base_layers * base_loops
        loops = max(1, round(applications / layers))
        min_loops = min(int(base.get("min_recurrent_loops", 2)), loops)
        prior = min(float(base.get("ponder_prior_steps", 3.0)), float(loops))
        return dict(num_recurrent_layers=layers, max_recurrent_loops=loops, min_recurrent_loops=min_loops,
                    ponder_prior_steps=prior)
    return build


def _local_window(base: Overrides) -> Overrides:
    """A window of one eighth of the context (1,024 at 8k) in every loop except the last, which stays global."""
    window = max(1, int(base.get("max_position_embeddings", 8192)) // 8)
    loops = int(base.get("max_recurrent_loops", 6))
    return dict(local_window=window, local_window_loops=tuple(range(1, loops)))


def _compose(*specs) -> Callable[[Overrides], Overrides]:
    """Merge variant specs (dicts or functions of the base) in order; later keys win."""
    def build(base: Overrides) -> Overrides:
        out: Overrides = {}
        for spec in specs:
            out.update(dict(spec(base) if callable(spec) else spec))
        return out
    return build


# name -> overrides, or a function of the base overrides returning overrides (so relative variants follow --set).
VARIANTS: Dict[str, Union[Overrides, Callable[[Overrides], Overrides]]] = {
    "baseline": {},
    "dense": {"use_moe": False},                                          # plan row 1
    "no_highway": {"use_highway": False},                                 # row 4
    "qk_norm": {"qk_norm": True},                                         # row 5
    "qk_norm_latent": {"qk_norm": True, "qk_norm_mode": "latent"},        # row 5, latent-space key norm
    "vocab16k": _vocab_variant(16000),                                    # row 3
    "vocab32k": _vocab_variant(32000),                                    # row 3
    "fewer_loops": _fewer_loops,                                          # two fewer maximum loops
    "fixed_depth": _fixed_depth,                                          # row 8: always train at the maximum depth
    "more_experts": _more_experts,                                        # twice the routed experts, same top-k
    "mor": {"routing_mode": "mor"},                                       # row 6: token-choice depth routing
    "gqa": {"attention_type": "gqa"},                                     # plain grouped-query attention vs MLA
    "attn_gate": {"attention_gate": True},                                # does a per-head sigmoid output gate help
    "last_loop": {"routing_mode": "last"},                                # last loop state, no halting mixture
    "inject": {"loop_input_injection": True},                             # re-inject the prelude output into each loop
    "big_experts": _big_experts,                                          # fewer, wider routed experts, same params
    "core3x4": _core_shape(3),                                            # 3 unique recurrent layers x 4 loops
    "core4x3": _core_shape(4),                                            # 4 unique recurrent layers x 3 loops
    "local_window": _local_window,                                        # local attention in all loops but the last
    "hybrid_ssm": {"ssm_mixer_layers": (1,)},                             # second recurrent layer is a selective SSM
    "v16k_core3x4": _compose(_vocab_variant(16000), _core_shape(3)),      # 16k vocabulary with the 3x4 core
    "v16k_core4x3": _compose(_vocab_variant(16000), _core_shape(4)),      # 16k vocabulary with the 4x3 core
}


def config_keys() -> set:
    """Names InkyConfig.__init__ declares. PretrainedConfig swallows unknown kwargs, so try/except is not enough."""
    params = inspect.signature(InkyConfig.__init__).parameters
    return {n for n, p in params.items() if n != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}


def parse_overrides(items: Optional[List[str]]) -> Overrides:
    """``["key=value", ...]`` to a dict. Values go through ast.literal_eval, else they stay strings."""
    keys = config_keys()
    out: Overrides = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        key = key.strip()
        if key not in keys:
            raise ValueError(f"--set {key}: InkyConfig has no such parameter")
        try:
            out[key] = ast.literal_eval(raw.strip())
        except (ValueError, SyntaxError):
            out[key] = raw.strip()
    return out


def resolve_variant(name: str, base: Overrides) -> Overrides:
    spec = VARIANTS[name]
    return dict(spec(base) if callable(spec) else spec)


def dense_equivalents(cfg: InkyConfig, total_params: int) -> Dict[str, Any]:
    """Analytic parameter-matched dense settings for a MoE config (see the module docstring)."""
    shared, routed, active, width = (int(cfg.num_shared_experts), int(cfg.num_routed_experts),
                                     int(cfg.num_active_routed), int(cfg.intermediate_size))
    if getattr(cfg, "use_moe", True) is False:
        return {"moe": False, "total_params": total_params, "active_params_est": total_params,
                "dense_intermediate_total_match": None, "dense_intermediate_active_match": None}
    blocks = cfg.num_pre_layers + cfg.num_recurrent_layers + cfg.num_post_layers
    per_expert = 3 * cfg.hidden_size * width
    inactive = blocks * max(0, routed - active) * per_expert
    return {"moe": True, "total_params": total_params, "active_params_est": total_params - inactive,
            "dense_intermediate_total_match": (shared + routed) * width,
            "dense_intermediate_active_match": (shared + active) * width}


# --------------------------------------------------------------------------------------------------- data

def synthetic_windows(n: int, seq_len: int, data_vocab: int, eos_id: int, seed: int) -> torch.Tensor:
    """Seeded random-token windows in [2, data_vocab). 70% of tokens follow a fixed random bigram map, so a model
    that trains at all gets a loss below the uniform ln(V). The EOS id is excluded so it never lands mid-window."""
    if data_vocab < 6:
        raise ValueError(f"synthetic data needs a vocabulary of at least 6 ids, got {data_vocab}")
    gen = torch.Generator().manual_seed(seed)
    succ = torch.randint(2, data_vocab, (data_vocab,), generator=gen)
    out = torch.empty(n, seq_len, dtype=torch.long)
    out[:, 0] = torch.randint(2, data_vocab, (n,), generator=gen)
    for t in range(1, seq_len):
        follow = torch.rand(n, generator=gen) < 0.7
        out[:, t] = torch.where(follow, succ[out[:, t - 1]], torch.randint(2, data_vocab, (n,), generator=gen))
    out[out == eos_id] = 2 if eos_id != 2 else 3
    return out


class Encoder:
    """encode(text) -> ids and the size of the id space, for either a tokenizers JSON or a transformers tokenizer."""

    def __init__(self, encode: Callable[[str], List[int]], size: int):
        self.encode, self.size = encode, size


def load_encoder(path: str) -> Encoder:
    p = Path(path)
    json_file = p if p.is_file() else (p / "tokenizer.json" if (p / "tokenizer.json").is_file() else None)
    if json_file is not None:
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(json_file))
        return Encoder(lambda text: tok.encode(text, add_special_tokens=False).ids, tok.get_vocab_size())
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(path)
    except Exception as exc:
        raise SystemExit(f"error: cannot load tokenizer from {path!r}: {exc}") from exc
    return Encoder(lambda text: tok.encode(text, add_special_tokens=False), len(tok))


def corpus_texts(paths: List[str], text_field: str):
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            raise SystemExit(f"error: corpus file not found: {path}")
        if path.suffix.lower() == ".jsonl":
            yield from iter_jsonl(path, field=text_field)
        else:
            yield path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------------------- running

def _quiet(_: str) -> None:
    pass


def run_variant(name: str, config: InkyConfig, train_windows: torch.Tensor, eval_batches: List[torch.Tensor],
                args: argparse.Namespace, device: torch.device) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    model = InkyModel(config).to(device)
    counts = model.count_parameters()
    row: Dict[str, Any] = {"variant": name, "status": "ok", "params": counts["total"],
                           "dense_settings": dense_equivalents(config, counts["total"])}
    optimizer = HybridOptimizer(model, adamw_lr=args.lr, muon_lr=args.muon_lr)
    start = time.perf_counter()
    history = train(model, optimizer, train_windows.to(device), steps=args.steps, batch_size=args.batch_size,
                    use_loop_curriculum=args.loop_curriculum, grad_accum_steps=args.grad_accum, seed=args.seed,
                    log_every=max(1, args.steps), log_fn=_quiet if not args.verbose else print,
                    max_grad_norm=args.max_grad_norm)
    elapsed = time.perf_counter() - start
    records = [r for r in history if r.get("kind") == "train" and not r.get("skipped")]
    tail = records[-min(5, len(records)):]
    row["steps_run"] = len(records)
    row["skipped_steps"] = sum(1 for r in history if r.get("kind") == "train" and r.get("skipped"))
    row["final_loss"] = records[-1]["loss"] if records else None
    row["final_train_ce"] = sum(r["ce"] for r in tail) / len(tail) if tail else None
    row["train_seconds"] = elapsed
    row["tokens_per_s"] = args.steps * args.batch_size * args.grad_accum * train_windows.shape[1] / elapsed
    if eval_batches:
        model.eval()
        scored = evaluate(model, [b.to(device) for b in eval_batches])
        row["eval_nll"], row["eval_perplexity"] = scored["nll"], scored["perplexity"]
        row["eval_mean_loops"] = scored.get("mean_loops")
    return row


def format_markdown(result: Dict[str, Any]) -> str:
    def num(x, fmt):
        return "n/a" if x is None else format(x, fmt)

    a = result["args"]
    lines = [f"Ablation: {a['steps']} steps, batch {a['batch_size']} x seq {a['seq_len']}, seed {a['seed']}, "
             f"data {result['data']}",
             "",
             "| variant | status | params (M) | active est. (M) | dense width (total / active) | train CE | "
             "eval NLL | eval PPL | tok/s |",
             "| :--- | :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |"]
    for r in result["variants"]:
        if r["status"] != "ok":
            lines.append(f"| {r['variant']} | skipped: {r['reason']} | | | | | | | |")
            continue
        d = r["dense_settings"]
        dense = ("n/a (dense)" if not d["moe"]
                 else f"{d['dense_intermediate_total_match']} / {d['dense_intermediate_active_match']}")
        lines.append(f"| {r['variant']} | ok | {r['params'] / 1e6:.3f} | "
                     f"{d['active_params_est'] / 1e6:.3f} | {dense} | "
                     f"{num(r.get('final_train_ce'), '.4f')} | {num(r.get('eval_nll'), '.4f')} | "
                     f"{num(r.get('eval_perplexity'), '.2f')} | {num(r.get('tokens_per_s'), '.0f')} |")
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-variants", action="store_true", help="print the variant names and exit")
    parser.add_argument("--variants", default=None, help="comma-separated variant names (default: all)")
    parser.add_argument("--synthetic", action="store_true", help="built-in random-token corpus; no files needed")
    parser.add_argument("--synthetic-windows", type=int, default=64, help="windows in the synthetic corpus")
    parser.add_argument("--corpus", nargs="+", help=".jsonl or text files")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--tokenizer-path", help="tokenizers JSON file, directory with tokenizer.json, or HF name")
    parser.add_argument("--min-chars", type=int, default=200)
    parser.add_argument("--tiny", action="store_true", help="start from a tiny base config (seconds on a CPU)")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="base InkyConfig override applied to every variant, repeatable")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--loop-curriculum", action="store_true")
    parser.add_argument("--eval-fraction", type=float, default=0.25, help="share of windows held out")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true", help="print train()'s own log lines")
    parser.add_argument("--out-dir", type=Path, default=None, help="writes ablation.md and ablation.json here")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.list_variants:
        return args
    if not args.synthetic and not (args.corpus and args.tokenizer_path):
        parser.error("give --synthetic, or --corpus together with --tokenizer-path")
    if args.steps < 1 or args.batch_size < 1 or args.grad_accum < 1 or args.seq_len < 2:
        parser.error("--steps, --batch-size, --grad-accum must be >= 1 and --seq-len >= 2")
    names = list(VARIANTS) if args.variants is None else [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variant(s) {unknown}; choose from {list(VARIANTS)}")
    args.variant_names = names
    try:
        args.base_overrides = {**(TINY_KWARGS if args.tiny else {}), **parse_overrides(args.overrides)}
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Optional[List[str]] = None) -> Optional[Dict[str, Any]]:
    args = parse_args(argv)
    if args.list_variants:
        for name in VARIANTS:
            print(name)
        return None
    device = torch.device(args.device)
    base = args.base_overrides
    base_config = InkyConfig(**base)
    if args.seq_len > base_config.max_position_embeddings:
        raise SystemExit(f"error: --seq-len {args.seq_len} exceeds max_position_embeddings "
                         f"({base_config.max_position_embeddings})")

    # 1. Lazily resolve variants; skip any whose keys InkyConfig does not accept yet.
    keys = config_keys()
    results: List[Dict[str, Any]] = []
    candidates: Dict[str, Overrides] = {}
    for name in args.variant_names:
        extra = resolve_variant(name, base)
        missing = sorted(k for k in extra if k not in keys)
        if missing:
            reason = f"InkyConfig does not accept {', '.join(missing)} yet"
            print(f"skipping variant {name!r}: {reason}")
            results.append({"variant": name, "status": "skipped", "reason": reason, "overrides": extra})
            continue
        candidates[name] = {**base, **extra}

    # 2. The data: identical for every variant.
    data_eos = base_config.eos_token_id
    if args.synthetic:
        data_vocab = min([c.get("scratch_open_id", base_config.scratch_open_id) for c in candidates.values()]
                         or [base_config.scratch_open_id])
        windows = synthetic_windows(args.synthetic_windows, args.seq_len, data_vocab, data_eos, args.seed)
        data_desc = f"synthetic ({windows.shape[0]} windows, ids < {data_vocab})"
    else:
        enc = load_encoder(args.tokenizer_path)
        data_vocab = enc.size
        windows = prepare_corpus(corpus_texts(args.corpus, args.text_field), enc.encode, args.seq_len, data_eos,
                                 min_chars=args.min_chars)
        data_desc = f"corpus ({windows.shape[0]} windows, tokenizer size {enc.size})"
    if windows.shape[0] < 2:
        raise SystemExit("error: need at least 2 windows (one to train on, one held out)")
    train_idx, eval_idx = split_heldout(list(range(windows.shape[0])), fraction=args.eval_fraction, seed=args.seed)
    train_windows, held = windows[train_idx], windows[eval_idx]
    eval_batches = [held[i:i + args.batch_size] for i in range(0, held.shape[0], args.batch_size)]

    # 3. Build, train and score each runnable variant.
    for name in args.variant_names:
        if name not in candidates:
            continue
        config = InkyConfig(**candidates[name])
        if data_vocab > config.scratch_open_id:
            reason = f"data ids reach {data_vocab} but this vocabulary only has {config.scratch_open_id} text ids"
        elif data_eos >= config.vocab_size or data_eos in (config.scratch_open_id, config.scratch_sep_id,
                                                              config.scratch_close_id):
            reason = f"data EOS id {data_eos} is not a text id of this vocabulary"
        else:
            reason = None
        if reason is not None:
            print(f"skipping variant {name!r}: {reason}")
            results.append({"variant": name, "status": "skipped", "reason": reason})
            continue
        print(f"running variant {name!r} ...")
        row = run_variant(name, config, train_windows, eval_batches, args, device)
        row["overrides"] = {k: v for k, v in candidates[name].items() if base.get(k, "<unset>") != v}
        results.append(row)
        print(f"  params {row['params'] / 1e6:.3f} M, train CE {row['final_train_ce']:.4f}, "
              f"eval NLL {row.get('eval_nll', float('nan')):.4f}, {row['tokens_per_s']:.0f} tok/s")

    order = {n: i for i, n in enumerate(args.variant_names)}
    results.sort(key=lambda r: order[r["variant"]])
    result = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                       if k in ("steps", "batch_size", "grad_accum", "seq_len", "lr", "muon_lr", "seed",
                                "loop_curriculum", "eval_fraction", "tiny", "overrides")},
              "data": data_desc, "variants": results}
    table = format_markdown(result)
    print()
    print(table)
    json_path = args.json or (args.out_dir / "ablation.json" if args.out_dir else None)
    md_path = args.markdown or (args.out_dir / "ablation.md" if args.out_dir else None)
    for path, text in ((json_path, json.dumps(result, indent=2, default=str)), (md_path, table + "\n")):
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(text, encoding="utf-8")
            print(f"wrote {path}")
    return result


if __name__ == "__main__":
    main()
