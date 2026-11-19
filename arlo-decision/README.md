# Arlo

A tiny decision head for small models. Arlo reads a model's hidden states and answers two kinds of
question, without generating any text:

- **Choice**: which of N options is best? Returns a probability for each option.
- **Yes/No**: does this hold? Returns P(yes).

Arlo is about 0.4 M parameters at width 384 (8 slots, width 128). It never changes the model it reads.
You train only Arlo, on labelled decisions, with the model frozen.

Arlo was extracted from the Inky project, where it was ported from the
Choice and Noul primitives of Rhen. The maths is unchanged: with the same weights it gives the same scores
as the original (checked numerically). This package adds an encoder-agnostic interface, padding masks,
standalone checkpoints and tests.

**Status:** Arlo starts untrained. Its answers mean nothing until you train it on your own labelled
decisions. It warns once per call while untrained. No pretrained weights ship with this repository.

## Install

```bash
pip install git+https://github.com/kivitas/arlo-decision.git      # or: pip install -e . from a clone
pip install "arlo-decision[hf]"                                    # optional, for Hugging Face models
```

Requires Python 3.10+ and PyTorch.

## Use with your model

Arlo needs one thing from your model: **token ids in, hidden states `[B, T, hidden_size]` out.** The model
should be causal (decoder or recurrent). Any `nn.Module` or function with that shape works as is.

```python
from arlo import ArloDecider, train_arlo

decider = ArloDecider(my_model, hidden_size=384)       # my_model(ids) -> [B, T, 384]

# Train the head. Examples are token-id lists. The model stays frozen.
choice_data = [{"context": ctx_ids, "options": [opt_a_ids, opt_b_ids, opt_c_ids], "gold": 1}, ...]
train_arlo(decider, [choice_data[i:i+16] for i in range(0, len(choice_data), 16)], "choice", epochs=5)

probs = decider.choice(ctx_ids, [opt_a_ids, opt_b_ids, opt_c_ids])   # tensor [3], sums to 1
best  = decider.pick(ctx_ids, [opt_a_ids, opt_b_ids, opt_c_ids])     # index

# Yes/No: examples are {"context": ids, "gold": 0 or 1}
train_arlo(decider, yn_batches, "binary")
p_yes = decider.yes_no(ctx_ids, question=question_ids)               # scalar
go    = decider.decide(ctx_ids, threshold=0.7)                       # bool

decider.save("arlo.pt")                                              # Arlo weights only
decider = ArloDecider.load("arlo.pt", my_model)
```

`question` is optional extra ids appended after the context. In a choice, each option is scored after the
context and question, and is represented by the mean of its own token states.

### Adapters

If your model does not return a bare tensor, wrap it:

```python
from arlo import HFEncoder, AttrEncoder, FnEncoder

ArloDecider(HFEncoder(hf_model), hidden_size=hf_model.config.hidden_size)     # last hidden layer
ArloDecider(AttrEncoder(model, "hidden", return_hidden=True), hidden_size=384) # model(ids, return_hidden=True).hidden
ArloDecider(FnEncoder(lambda ids: ...), hidden_size=384)
```

A custom encoder may take `(input_ids, attention_mask)`; name the second parameter `attention_mask` and Arlo
passes a bool mask (True = real token), which matters for non-causal encoders.

### Hidden states you already have

The engine takes tensors, so you can skip the decider and the encoder:

```python
from arlo import Arlo, ArloConfig
arlo = Arlo(ArloConfig(hidden_size=384))
scores = arlo.choice_scores(query, options, states)   # [B, d], [B, N, d], [B, S, d] -> [B, N] logits
logit  = arlo.binary_logit(query, states)             # -> [B]
loss   = arlo.training_loss(query, options, states, gold, task="choice")
```

## Configuration

`ArloConfig(hidden_size, slots=8, dim=128, heads=4, pointer_rank=64, refine_steps=2)`. `hidden_size` must
match your model. Pass the others as keyword arguments to `ArloDecider` to make Arlo smaller or larger.

## How it works

1. K latent decision slots cross-attend to the hidden states and refine each other through a gated update.
   They are reset for every decision.
2. The query (the last state of the context or question) is shifted by the slots that match it best.
3. Choice: a multi-head bilinear pointer scores the query against each option. Option scores depend on the
   other options and do not depend on option order. Yes/No: a linear head gives one logit.

Not included: Rhen's Score, Range, Extract and Cascade primitives, its calibration layers, abstention and
trajectory memory.

## Example and tests

```bash
python examples/toy_model.py     # trains Arlo on a toy GRU
python -m pytest -q
```

## Licence

MIT, copyright (c) 2026 Kivitas. See `LICENSE`.
