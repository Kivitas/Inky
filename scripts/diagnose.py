"""Report stream scale, loop-to-loop change, highway share and expert load for an Inky model.

    python scripts/diagnose.py                                  # default config, untrained, random tokens
    python scripts/diagnose.py --tiny --seq-len 64              # seconds on a CPU
    python scripts/diagnose.py --set highway_gain_init=0.1 --set loop_mix_normalize=False   # the earlier behaviour
    python scripts/diagnose.py --checkpoint runs/inky/checkpoints/step_500 --json diag.json

The input is a batch of random token ids (seeded), so the numbers describe the architecture's scales, not how it
reads text. That is what you want at initialisation. For a trained checkpoint random ids are still a fixed probe
that can be compared across steps, but expert load on real text is better read from ``train --diagnose-every``.
See ``inky.diagnostics`` for what each column means.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))   # sibling scripts (bench, sample)

import torch  # noqa: E402

from inky import InkyConfig, InkyModel  # noqa: E402
from inky.diagnostics import format_diagnostics, loop_diagnostics  # noqa: E402


def build_model(args) -> InkyModel:
    if args.checkpoint:
        from sample import load_model
        return load_model(args.checkpoint, args.device)
    from bench import TINY_KWARGS, parse_overrides
    kwargs = dict(TINY_KWARGS) if args.tiny else {}
    kwargs.update(parse_overrides(args.set))
    torch.manual_seed(args.seed)
    return InkyModel(InkyConfig(**kwargs)).to(args.device).eval()


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", help="a save_pretrained dir, step_N dir, best.pt or checkpoints dir")
    parser.add_argument("--tiny", action="store_true", help="a few-thousand-parameter config (plumbing check)")
    parser.add_argument("--set", action="append", metavar="KEY=VALUE", help="InkyConfig override (repeatable)")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-loops", type=int, default=None, help="default: the model's maximum")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", metavar="PATH", help="also write the numbers to this JSON file")
    args = parser.parse_args(argv)
    if args.seq_len < 2 or args.batch_size < 1:
        parser.error("--seq-len must be >= 2 and --batch-size >= 1")
    if args.checkpoint and (args.tiny or args.set):
        parser.error("--tiny and --set build a new model; do not combine them with --checkpoint")

    try:
        model = build_model(args)
    except ValueError as exc:
        parser.error(str(exc))
    cfg = model.config
    if args.seq_len > cfg.max_position_embeddings:
        parser.error(f"--seq-len {args.seq_len} exceeds max_position_embeddings {cfg.max_position_embeddings}")
    gen = torch.Generator().manual_seed(args.seed)
    ids = torch.randint(2, max(3, cfg.scratch_open_id), (args.batch_size, args.seq_len), generator=gen)
    result = loop_diagnostics(model, ids.to(args.device), num_loops=args.num_loops)
    print(format_diagnostics(result))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
