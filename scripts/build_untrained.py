"""Build the untrained Inky model, print its parameter budget, and optionally save it.

    python scripts/build_untrained.py                     # print the breakdown only
    python scripts/build_untrained.py --out checkpoints/inky-50m-init
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="seed for the random initialisation")
    parser.add_argument("--out", type=Path, default=None, help="directory to save config and weights")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    config = InkyConfig()
    model = InkyModel(config)

    print(f"Inky, untrained (seed {args.seed})")
    print(f"  context window         {config.max_position_embeddings} tokens")
    print(f"  effective depth        {config.num_pre_layers} + {config.num_recurrent_layers} x "
          f"{config.max_recurrent_loops} + {config.num_post_layers} = "
          f"{config.num_pre_layers + config.num_recurrent_layers * config.max_recurrent_loops + config.num_post_layers} "
          f"layer applications (training samples {config.min_recurrent_loops}..{config.max_recurrent_loops} loops)")
    print("  parameters")
    for name, count in model.count_parameters().items():
        print(f"    {name:32s} {count / 1e6:8.2f} M")
    print(f"  fp32 weight size       {model.count_parameters()['total'] * 4 / 2**20:.0f} MiB")
    print(f"  bf16 weight size       {model.count_parameters()['total'] * 2 / 2**20:.0f} MiB")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.out)
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
