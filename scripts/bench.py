"""Measure prefill time and per-token decode time of an Inky model at several context lengths.

    python scripts/bench.py                                   # default config, contexts 256 and 1024
    python scripts/bench.py --context-lengths 256,1024,4096 --threads 8 --repeats 5 --json bench.json
    python scripts/bench.py --num-loops 4 --set qk_norm=True --set num_routed_experts=16
    python scripts/bench.py --tiny --context-lengths 16,32 --decode-tokens 4 --repeats 1   # seconds on a CPU

The model is built from ``InkyConfig`` (plus ``--set key=value`` overrides, parsed with ast.literal_eval; a value
that is not a Python literal is kept as a string) and is untrained, which does not matter for timing. For each
context length the script times one cached prefill pass of that many tokens, then ``--decode-tokens`` single-token
cached decode steps, and repeats that ``--repeats`` times. The reported time is the median over the repeats.

Peak memory is the CUDA peak allocation on a GPU. On CPU it is the peak resident set of the whole process, which
never goes down, so later rows include the earlier rows. It is None where the platform does not provide it.
"""

import argparse
import ast
import inspect
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from inky import InkyConfig, InkyModel  # noqa: E402

TINY_VOCAB = 512
TINY_KWARGS: Dict[str, Any] = dict(
    vocab_size=TINY_VOCAB, scratch_open_id=TINY_VOCAB - 3, scratch_sep_id=TINY_VOCAB - 2,
    scratch_close_id=TINY_VOCAB - 1, hidden_size=64, num_attention_heads=4, qk_nope_head_dim=12,
    qk_rope_head_dim=4, v_head_dim=16, q_lora_rank=32, kv_lora_rank=24, intermediate_size=96,
    num_routed_experts=4, num_active_routed=2, ssm_state_dim=16, max_recurrent_loops=4, min_recurrent_loops=2,
    loop_dropout_prob=0.0, max_position_embeddings=256, arlo_dim=16, arlo_pointer_rank=8, arlo_slots=4,
    eos_token_id=1, pad_token_id=1,
)


def config_keys() -> set:
    """Names InkyConfig.__init__ declares. PretrainedConfig swallows unknown kwargs, so a try/except is not enough."""
    params = inspect.signature(InkyConfig.__init__).parameters
    return {n for n, p in params.items() if n != "self" and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)}


def parse_overrides(items: Optional[List[str]]) -> Dict[str, Any]:
    """``["key=value", ...]`` to a dict. Values go through ast.literal_eval, else they stay strings."""
    keys = config_keys()
    out: Dict[str, Any] = {}
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


def peak_memory_mb(device: torch.device) -> Optional[float]:
    """CUDA peak allocation, else the process's peak resident set (monotonic), else None."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 2**20
    try:
        import resource  # POSIX
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 2**20 if sys.platform == "darwin" else rss / 1024
    except ImportError:
        pass
    try:  # Windows, without psutil
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        counters = Counters()
        counters.cb = ctypes.sizeof(Counters)
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / 2**20
    except Exception:  # best effort only
        pass
    return None


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def bench_context(model: InkyModel, context: int, decode_tokens: int, repeats: int, batch_size: int,
                  num_loops: Optional[int], device: torch.device, seed: int) -> Dict[str, Any]:
    """Time a cached prefill of ``context`` tokens and ``decode_tokens`` single-token steps after it."""
    cfg = model.config
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(2, max(3, cfg.scratch_open_id), (batch_size, context), generator=gen).to(device)
    n_decode = max(0, min(decode_tokens, cfg.max_position_embeddings - context))
    prefill_times: List[float] = []
    decode_times: List[float] = []
    for _ in range(repeats):
        _sync(device)
        t0 = time.perf_counter()
        out = model(ids, use_cache=True, num_loops=num_loops, last_only=True)
        _sync(device)
        prefill_times.append(time.perf_counter() - t0)
        cache, nxt = out.past, out.logits[:, -1].argmax(-1, keepdim=True)
        if n_decode:
            t0 = time.perf_counter()
            for _ in range(n_decode):
                out = model(nxt, past=cache, last_only=True)  # the loop count is fixed by the prefill call
                nxt = out.logits[:, -1].argmax(-1, keepdim=True)
            _sync(device)
            decode_times.append((time.perf_counter() - t0) / n_decode)
    prefill = statistics.median(prefill_times)
    row: Dict[str, Any] = {
        "context": context, "batch_size": batch_size, "repeats": repeats,
        "prefill_s": prefill, "prefill_tokens_per_s": batch_size * context / prefill,
        "decode_tokens": n_decode, "decode_ms_per_token": None, "decode_tokens_per_s": None,
        "peak_memory_mb": peak_memory_mb(device),
    }
    if decode_times:
        per_token = statistics.median(decode_times)
        row["decode_ms_per_token"] = per_token * 1e3
        row["decode_tokens_per_s"] = batch_size / per_token
    return row


def format_table(result: Dict[str, Any]) -> str:
    def num(x, fmt):
        return "n/a" if x is None else format(x, fmt)

    lines = [f"params {result['params']['total'] / 1e6:.2f} M | device {result['device']} | threads {result['threads']}"
             f" | num_loops {result['num_loops']} | batch {result['batch_size']}",
             "| context | prefill (s) | prefill tok/s | decode ms/token | decode tok/s | peak mem (MiB) |",
             "| ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in result["results"]:
        lines.append(f"| {r['context']} | {num(r['prefill_s'], '.3f')} | {num(r['prefill_tokens_per_s'], '.1f')} | "
                     f"{num(r['decode_ms_per_token'], '.2f')} | {num(r['decode_tokens_per_s'], '.1f')} | "
                     f"{num(r['peak_memory_mb'], '.0f')} |")
    for skipped in result["skipped"]:
        lines.append(f"skipped context {skipped['context']}: {skipped['reason']}")
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--context-lengths", default="256,1024",
                        help="comma-separated prefill lengths (default 256,1024)")
    parser.add_argument("--decode-tokens", type=int, default=32, help="cached decode steps timed per repeat")
    parser.add_argument("--repeats", type=int, default=3, help="timed repeats per context; the median is reported")
    parser.add_argument("--num-loops", type=int, default=None, help="recurrent loops (default: the model maximum)")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU threads (default: leave as is)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="InkyConfig override, repeatable (value parsed with ast.literal_eval)")
    parser.add_argument("--tiny", action="store_true", help="start from a tiny config instead of the default one")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", type=Path, default=None, help="write the results to this JSON file")
    args = parser.parse_args(argv)
    try:
        args.contexts = [int(x) for x in args.context_lengths.split(",") if x.strip()]
        args.config_overrides = parse_overrides(args.overrides)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.contexts or min(args.contexts) < 1:
        parser.error("--context-lengths needs positive integers")
    if args.repeats < 1 or args.batch_size < 1 or args.decode_tokens < 0:
        parser.error("--repeats and --batch-size must be >= 1 and --decode-tokens >= 0")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be >= 1")
    return args


def main(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    args = parse_args(argv)
    device = torch.device(args.device)
    previous_threads = torch.get_num_threads()
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    try:
        torch.manual_seed(args.seed)
        config = InkyConfig(**{**(TINY_KWARGS if args.tiny else {}), **args.config_overrides})
        model = InkyModel(config).to(device).eval()
        if args.num_loops is not None and not 1 <= args.num_loops <= config.max_recurrent_loops:
            raise SystemExit(f"error: --num-loops must be in [1, {config.max_recurrent_loops}]")
        result: Dict[str, Any] = {
            "config_overrides": {**(TINY_KWARGS if args.tiny else {}), **args.config_overrides},
            "params": model.count_parameters(), "device": str(device), "threads": torch.get_num_threads(),
            "num_loops": args.num_loops if args.num_loops is not None else config.max_recurrent_loops,
            "batch_size": args.batch_size, "results": [], "skipped": [],
        }
        # One short warm-up so the first timed row does not pay for lazy initialisation.
        with torch.no_grad():
            warm = torch.randint(2, max(3, config.scratch_open_id), (args.batch_size, min(8, args.contexts[0])))
            model(warm.to(device), use_cache=True, num_loops=args.num_loops, last_only=True)
        for context in args.contexts:
            if context > config.max_position_embeddings:
                result["skipped"].append({"context": context, "reason":
                                          f"exceeds max_position_embeddings ({config.max_position_embeddings})"})
                continue
            result["results"].append(bench_context(model, context, args.decode_tokens, args.repeats, args.batch_size,
                                                   args.num_loops, device, args.seed))
    finally:
        torch.set_num_threads(previous_threads)
    print(format_table(result))
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.json}")
    return result


if __name__ == "__main__":
    main()
