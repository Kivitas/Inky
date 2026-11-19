"""Benchmark GatedDeltaNet prefill (recurrent vs chunked), SSM scan and MoE dispatch on CPU.
Usage: PYTHONPATH=. python scripts/bench_mixers.py"""
import time, torch
from inky.config import InkyConfig
from inky.layers import GatedDeltaNetMixer, SelectiveSSMMixer, SharedExpertMoE


def t(fn, n=3):
    fn(); s = time.process_time()
    for _ in range(n): fn()
    return (time.process_time() - s) / n * 1e3


@torch.no_grad()
def main():
    torch.set_num_threads(1); torch.manual_seed(0)
    cfg = InkyConfig()
    d = cfg.hidden_size
    g = GatedDeltaNetMixer(cfg).eval()
    for S in (128, 512):
        x = torch.randn(2, S, d)
        g.recurrent_max_tokens = 10**9; a = t(lambda: g(x), 1)
        g.recurrent_max_tokens = 1; b = t(lambda: g(x))
        print(f"GDN prefill S={S}: loop {a:.1f} ms  chunked {b:.1f} ms  x{a/b:.1f}")
    s = SelectiveSSMMixer(cfg).eval()
    for S in (128, 512):
        x = torch.randn(2, S, d)
        print(f"SSM S={S}: {t(lambda: s(x)):.1f} ms")
    m = SharedExpertMoE(cfg).eval()
    for N in (64, 1024):
        x = torch.randn(1, N, d)
        print(f"MoE N={N}: {t(lambda: m(x)):.1f} ms")


if __name__ == "__main__":
    main()
