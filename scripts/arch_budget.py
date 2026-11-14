"""Compare architecture variants at matched compute before training: parameters, FLOPs, decode cost and KV cache.

    python scripts/arch_budget.py --tiny --variants baseline,dense,mor       # seconds on a CPU
    python scripts/arch_budget.py --variants baseline,more_experts --prefill-len 512 --extrapolate-to 8192
    python scripts/arch_budget.py --tiny --variants baseline,fewer_loops --set num_routed_experts=8 --decode-ms \
        --json runs/budget.json --markdown runs/budget.md

Variants are the named entries of ``ablate.VARIANTS`` (run ``ablate.py --list-variants``). Each one is built as
``InkyConfig(**{**base, **resolve_variant(name, base)})``, where ``base`` is the tiny config with ``--tiny`` and the
``--set`` overrides, and is measured on CPU with a fixed seed:

* ``params``: the total parameter count, and the share held by the tied embedding.
* ``body_flops_per_token``: forward FLOPs of the body (no LM head) on ``--prefill-len`` random tokens, divided by
  the length; attention is counted at that length without a causal discount (an upper bound for prefill).
  ``head_flops_per_token`` is analytic, ``2 * vocab_size * hidden_size``.
* ``decode_flops_by_context``: FLOPs of one cached decode step after a prefill of each ``--contexts`` length. A line
  is fitted through the first and last measured points, which gives a context-independent part (the intercept) and
  an attention part (the slope per cached token), and the value extrapolated to ``--extrapolate-to``. A context
  that leaves no room for the decode token (``>= max_position_embeddings``) is skipped and listed.
* ``cache_bytes_per_token``: fp32 bytes of the latent KV cache per cached token after the largest prefill, from the
  ``LatentKV`` entries. Non-latent entries (selective-SSM state) are constant in length, so they are reported as
  ``cache_fixed_bytes`` and left out of the per-token figure. The bf16 column is half of the fp32 one.
* ``decode_ms`` (with ``--decode-ms``): the median wall-clock time of up to 5 cached decode steps at the largest
  measured context, on CPU with the current torch thread count.

A variant that cannot be built or run is recorded as ``{"name": ..., "error": ...}`` and the others still run.

SDPA attention. ``torch.utils.flop_counter`` has no formula for the CPU flash-attention op that
``F.scaled_dot_product_attention`` dispatches to, so it counts such attention as 0 FLOPs. Prefill attention (both
attention types) and the whole of grouped-query attention use SDPA, while the multi-latent decode step uses explicit
matmuls, which are counted. Without a correction a GQA variant would look far cheaper to decode than it is, so this
script registers the standard formula, ``2 * b * heads * s_q * s_k * (d_qk + d_v)``, for that op.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts (ablate)

import torch  # noqa: E402
from ablate import TINY_KWARGS, VARIANTS, parse_overrides, resolve_variant  # noqa: E402
from torch.utils.flop_counter import FlopCounterMode  # noqa: E402

from inky import InkyConfig, InkyModel  # noqa: E402
from inky.layers import LatentKV  # noqa: E402


def _sdpa_cpu_flops(q_shape, k_shape, v_shape, *args, **kwargs) -> int:
    """FLOPs of one SDPA call (scores and weighted values), counted on the query heads and without a causal discount."""
    b, heads, s_q, d_qk = q_shape
    s_k, d_v = k_shape[-2], v_shape[-1]
    return 2 * b * heads * s_q * s_k * d_qk + 2 * b * heads * s_q * s_k * d_v


SDPA_FLOP_MAPPING = {torch.ops.aten._scaled_dot_product_flash_attention_for_cpu: _sdpa_cpu_flops}


def flop_counter() -> FlopCounterMode:
    """A FLOP counter that also counts SDPA attention (see the module docstring)."""
    return FlopCounterMode(display=False, custom_mapping=SDPA_FLOP_MAPPING)


EMBEDDING_KEY = "embedding (tied with LM head)"
DEFAULT_PREFILL = 512
DECODE_TIMING_STEPS = 5


def latent_cache_bytes(cache) -> Tuple[int, int]:
    """(bytes of latent KV entries, bytes of other entries). Each LatentKV entry is already cut to its valid length."""
    layers = list(cache.pre) + list(cache.post)
    for loop in cache.core:
        layers += list(loop or [])
    latent = fixed = 0
    for entry in layers:
        if entry is None:
            continue
        nbytes = sum(t.numel() * t.element_size() for t in entry[:2])
        if isinstance(entry, LatentKV):
            latent += nbytes
        else:
            fixed += nbytes
    return latent, fixed


def _random_ids(gen: torch.Generator, vocab_open: int, n: int) -> torch.Tensor:
    return torch.randint(2, max(3, vocab_open), (1, n), generator=gen)


@torch.no_grad()
def measure_variant(name: str, base: Dict[str, Any], args: argparse.Namespace, contexts: List[int]) -> Dict[str, Any]:
    cfg = InkyConfig(**{**base, **resolve_variant(name, base)})
    torch.manual_seed(args.seed)
    model = InkyModel(cfg).eval()
    counts = model.count_parameters()
    loops = cfg.max_recurrent_loops
    limit = cfg.max_position_embeddings
    gen = torch.Generator().manual_seed(args.seed)

    prefill = args.prefill_len if args.prefill_len is not None else min(DEFAULT_PREFILL, limit)
    with flop_counter() as fc:
        model(_random_ids(gen, cfg.scratch_open_id, prefill), return_hidden=True, num_loops=loops)
    body = fc.get_total_flops() / prefill
    head = 2.0 * cfg.vocab_size * cfg.hidden_size
    head_passes = 1.0 + (
        loops * cfg.step_ce_token_fraction
        if cfg.routing_mode == "ponder" and cfg.ponder_step_loss_coef > 0
        else 0.0
    )
    train_body = body * (4.0 if cfg.gradient_checkpointing else 3.0)
    train_head = head * head_passes * (4.0 if cfg.lm_loss_recompute else 3.0)
    train_flops = train_body + train_head

    measured = [t for t in contexts if t < limit]
    skipped = [t for t in contexts if t >= limit]
    largest = measured[-1] if measured else None
    decode: Dict[int, float] = {}
    cache_per_token = cache_fixed = None
    for t in measured:
        out = model(_random_ids(gen, cfg.scratch_open_id, t), use_cache=True, last_only=True, num_loops=loops)
        if t == largest:
            latent, fixed = latent_cache_bytes(out.past)
            cache_per_token, cache_fixed = latent / t, fixed        # batch size is 1
        with flop_counter() as fc:
            model(_random_ids(gen, cfg.scratch_open_id, 1), past=out.past, last_only=True)
        decode[t] = fc.get_total_flops()

    slope = intercept = extrapolated = None
    if len(measured) >= 2:
        t0, t1 = measured[0], measured[-1]
        slope = (decode[t1] - decode[t0]) / (t1 - t0)
        intercept = decode[t0] - slope * t0
        extrapolated = intercept + slope * args.extrapolate_to

    decode_ms = None
    if args.decode_ms and largest is not None:
        out = model(_random_ids(gen, cfg.scratch_open_id, largest), use_cache=True, last_only=True, num_loops=loops)
        steps = min(DECODE_TIMING_STEPS, limit - largest)     # decode positions must stay below the limit
        times = []
        for _ in range(steps):
            nxt = _random_ids(gen, cfg.scratch_open_id, 1)
            t0 = time.perf_counter()
            model(nxt, past=out.past, last_only=True)
            times.append(time.perf_counter() - t0)
        decode_ms = statistics.median(times) * 1e3

    bytes_fp32 = None if cache_per_token is None else cache_per_token * args.extrapolate_to / 2**20
    return {
        "name": name,
        "prefill_len": prefill,
        "params": counts["total"],
        "embedding_share": counts[EMBEDDING_KEY] / counts["total"],
        "body_flops_per_token": body,
        "head_flops_per_token": head,
        "total_flops_per_token": body + head,
        "train_flops_per_token": train_flops,
        "measured_contexts": measured,
        "skipped_contexts": skipped,
        "largest_context": largest,
        "decode_flops_by_context": decode,
        "decode_flops_largest": decode.get(largest),
        "decode_fixed_flops": intercept,
        "decode_attention_flops_per_context_token": slope,
        "extrapolate_to": args.extrapolate_to,
        "decode_flops_extrapolated": extrapolated,
        "cache_bytes_per_token": cache_per_token,
        "cache_fixed_bytes": cache_fixed,
        "cache_mib_fp32_extrapolated": bytes_fp32,
        "cache_mib_bf16_extrapolated": None if bytes_fp32 is None else bytes_fp32 / 2,
        "decode_ms": decode_ms,
        "decode_ms_context": largest if decode_ms is not None else None,
    }


def _fmt(x: Optional[float], spec: str) -> str:
    return "n/a" if x is None else format(x, spec)


def format_markdown(rows: List[Dict[str, Any]], args: argparse.Namespace) -> str:
    prefill = "min(512, max_position_embeddings)" if args.prefill_len is None else f"{args.prefill_len} tokens"
    lines = [f"Architecture budget: prefill {prefill}, contexts {args.contexts}, "
             f"extrapolated to {args.extrapolate_to}, seed {args.seed}"
             f"{', tiny base' if args.tiny else ''}",
             "",
             "| variant | params (M) | embedding % | body MFLOP/token | head MFLOP/token | total MFLOP/token | "
             "decode MFLOP at largest measured context | decode MFLOP at --extrapolate-to | "
             "cache MiB (bf16) at --extrapolate-to | decode ms |",
             "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]

    def mflop(x: Optional[float]) -> str:
        return _fmt(None if x is None else x / 1e6, ".3f")

    for r in rows:
        if "error" in r:
            continue
        lines.append(
            f"| {r['name']} | {_fmt(r['params'] / 1e6, '.3f')} | {_fmt(100 * r['embedding_share'], '.1f')} | "
            f"{mflop(r['body_flops_per_token'])} | {mflop(r['head_flops_per_token'])} | "
            f"{mflop(r['total_flops_per_token'])} | {mflop(r['decode_flops_largest'])} | "
            f"{mflop(r['decode_flops_extrapolated'])} | "
            f"{_fmt(r['cache_mib_bf16_extrapolated'], '.3f')} | {_fmt(r['decode_ms'], '.2f')} |")
    errors = [r for r in rows if "error" in r]
    if errors:
        lines += ["", "Errored variants (not in the table):"]
        lines += [f"- {r['name']}: {r['error']}" for r in errors]
    notes = [r for r in rows if "error" not in r and r["skipped_contexts"]]
    if notes:
        lines += [""]
        lines += [f"- {r['name']}: skipped contexts {r['skipped_contexts']} (they leave no room for the decode token "
                  f"under max_position_embeddings)" for r in notes]
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--variants", default="baseline", help="comma-separated names from ablate.VARIANTS")
    parser.add_argument("--tiny", action="store_true", help="start from the tiny config instead of the default one")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="base InkyConfig override applied to every variant, repeatable")
    parser.add_argument("--prefill-len", type=int, default=None,
                        help="tokens used to measure body FLOPs (default: 512, capped at max_position_embeddings)")
    parser.add_argument("--contexts", default="256,1024,2048",
                        help="comma-separated cached context lengths for the decode step (default 256,1024,2048)")
    parser.add_argument("--extrapolate-to", type=int, default=8192,
                        help="context for the extrapolated decode and cache")
    parser.add_argument("--decode-ms", action="store_true",
                        help="also time cached decode steps (median of up to 5) at the largest context")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None, help="write the rows to this JSON file")
    parser.add_argument("--markdown", type=Path, default=None, help="write the table to this markdown file")
    args = parser.parse_args(argv)

    args.variant_names = [v.strip() for v in args.variants.split(",") if v.strip()]
    if not args.variant_names:
        parser.error("--variants needs at least one name")
    unknown = [v for v in args.variant_names if v not in VARIANTS]
    if unknown:
        parser.error(f"unknown variant(s) {unknown}; choose from {list(VARIANTS)}")
    try:
        args.contexts = sorted({int(x) for x in args.contexts.split(",") if x.strip()})
    except ValueError:
        parser.error("--contexts needs comma-separated integers")
    if not args.contexts or min(args.contexts) < 1:
        parser.error("--contexts needs positive integers")
    if (args.prefill_len is not None and args.prefill_len < 1) or args.extrapolate_to < 1:
        parser.error("--prefill-len and --extrapolate-to must be >= 1")
    try:
        args.base_overrides = {**(TINY_KWARGS if args.tiny else {}), **parse_overrides(args.overrides)}
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    rows: List[Dict[str, Any]] = []
    for name in args.variant_names:
        try:
            rows.append(measure_variant(name, args.base_overrides, args, args.contexts))
        except Exception as exc:  # one broken variant must not stop the comparison
            rows.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
    table = format_markdown(rows, args)
    print(table)
    result: Dict[str, Any] = {
        "variants": rows,
        "args": {"variants": args.variant_names, "tiny": args.tiny, "set": args.overrides,
                 "prefill_len": args.prefill_len, "contexts": args.contexts,
                 "extrapolate_to": args.extrapolate_to, "decode_ms": args.decode_ms, "seed": args.seed},
    }
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.json}")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(table + "\n", encoding="utf-8")
        print(f"wrote {args.markdown}")
    return result


if __name__ == "__main__":
    main()
