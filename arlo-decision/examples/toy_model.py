"""End to end: a toy small model + Arlo. Train a decision head, then decide.

Run:  python examples/toy_model.py
"""

import torch
import torch.nn as nn

from arlo import ArloDecider, evaluate, train_arlo

VOCAB, D = 32, 48


class ToyLM(nn.Module):
    """Stand-in for your small model: ids [B, T] -> hidden states [B, T, D]."""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D)
        self.rnn = nn.GRU(D, D, batch_first=True)   # causal

    def forward(self, ids):
        return self.rnn(self.emb(ids))[0]


torch.manual_seed(0)
lm = ToyLM()

# Task: given a context, pick the option that repeats the context's first token.
def make(n):
    out = []
    for _ in range(n):
        ctx = torch.randint(2, VOCAB, (6,)).tolist()
        right = [ctx[0]]
        wrong = [t for t in torch.randint(2, VOCAB, (3,)).tolist() if t != ctx[0]][:2] or [2]
        opts = [[w] for w in wrong] + [right]
        out.append({"context": ctx, "options": opts, "gold": len(opts) - 1})
    return out


data = make(160)
arlo = ArloDecider(lm, hidden_size=D, slots=4, dim=32, heads=2, pointer_rank=16)
train = [data[i:i + 16] for i in range(0, 128, 16)]
losses = train_arlo(arlo, train, task="choice", lr=3e-3, epochs=8)
print(f"loss {losses[0]:.3f} -> {losses[-1]:.3f}")
print("held-out:", evaluate(arlo, data[128:], task="choice"))
print("probs:", arlo.choice(data[130]["context"], data[130]["options"]))
