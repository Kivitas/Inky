# Inky

A ~50M-parameter recurrent language model for creative and text writing, with an 8k context
window. It is built to run locally. This repository contains the untrained model code, the
writing-controls, and tests. It has no training data and no training loop. Those come later.

Inky takes its language-model path from the EOS project (`Desktop/New folder/EOS LLM`) and scales
it down. Inky adds PENCIL-style scratch reduction from arXiv:2503.14337, and Arlo, a writing decision
maker ported from Rhen's Choice and Noul primitives. Arlo reads the model's states
and answers "which option" or "yes or no" questions. It doesn't generate text, and it never
changes the language-model path. Rhen's other primitives (Score, Range, Extract, Cascade) and its
calibration layers are not ported.

## Parameter budget (50.62M)

| Section | Parameters |
| :--- | ---: |
| Embedding, tied with LM head (50,003 × 384) | 19.20 M |
| Pre layer (1 block) | 7.74 M |
| Recurrent core (2 blocks, shared across loops) | 15.54 M |
| Post layer (1 block) | 7.74 M |
| Norms and halting unit | < 0.01 M |
| Decision engine (8 slots, width 128, 4 pointer heads) | 0.40 M |

Each block is a latent-KV attention layer (about 0.44 M) plus a shared-expert MoE: one shared
expert and eight routed experts, with two routed experts active per token (about 7.3 M).

Effective depth is 1 + 2 × 6 + 1 = **14 layer applications**. Training samples between 2 and 6
loops per batch. Inference uses 6 loops unless `num_loops` is passed.

## Architecture

```
tokens → embedding
       → pre layer
       → recurrent core, looped up to 6 times:
           FiLM step modulation (per loop index)
           state-space memory highway (causal running mean, carried across loops)
           2 × block (latent-KV attention + shared-expert MoE), loop dropout 0.1
           damped residual update
           ACT halting: per-token weights that sum to 1 over the loops
       → post layer
       → RMSNorm → tied LM head (logit soft-cap 30)
```

| Component | What it does | Source in EOS |
| :--- | :--- | :--- |
| Multi-latent attention | Keys and values come from one 128-d latent per token. The RoPE key is shared across heads. The KV cache stores only the latent. | `MultiLatentAttention` |
| Absorbed decoding | For one new token, the key projection is folded into the query, so cached latents are never up-projected. Matches the full path. | MLA matrix absorption |
| Shared-expert MoE | One always-on SwiGLU expert plus 8 routed experts (top-2, sigmoid router). Aux-loss-free balancing via a bias buffer. | `SharedExpertFineGrainedMoE` |
| Recurrent core | One set of shared layers applied repeatedly. Gives depth without extra parameters. | `RecurrentCoreBlock` |
| Step modulation | FiLM scale and shift per loop index. Starts as the identity. | `StepModulation` |
| State-space highway | Carries a per-position memory across loops. Its causal running mean is cached for decoding. | `StateSpaceMemoryHighway` |
| ACT halting | Each token stops adding weight once its halting mass reaches 0.9. Ponder cost is a training penalty. | `EntropyGatedACTHalting` |
| Chunked loss | Next-token cross-entropy and z-loss in chunks, so full 8k × 50k logits never exist at once. | `chunked_ce` |
| Scratch reduction | The model may draft notes, then close the block. Reduction deletes the notes and keeps the text. | PENCIL (arXiv:2503.14337) |

### Scratch reduction (PENCIL)

PENCIL's rule is `C [CALL] T [SEP] A [RETURN] ⇒ C A`: intermediate thoughts `T` are deleted from
context, and only the answer `A` stays. The paper applies this to logic puzzles. Inky applies the
same rule to writing: the model drafts notes between `scratch_open` and `scratch_sep`, writes the
text, and closes with `scratch_close`. The notes are removed, so the 8k window holds only prose.

- The reduction is `inky.scratch.reduce_scratch`. It handles nested blocks, leaves unclosed blocks
  in place, and never edits the prompt.
- `generate(..., reduce_scratch_blocks=True)` rebuilds the cache for the shortened context after
  each reduction, so the cache always matches the text the model sees. It needs batch size 1.
- The three scratch ids (50000–50002) must be added to `EOS_Tokenizer` as special tokens. The
  model has no way to learn this behaviour until it is trained. Until then it is inert.

Honest caveat: PENCIL's gains are measured on puzzles with a long reasoning trace. Whether it helps
prose is a hypothesis to test after training.

### Prose controls

`generate` has three decoding controls aimed at writing quality:

- `bad_word_ids`: blocks token sequences. Build them with
  `build_bad_word_ids(tokenizer_encode)`. The default list in `inky/style.py` covers stock
  phrases that show up often in AI-written text (for example "delve", "tapestry", "furthermore",
  "it's important to note").
- `no_repeat_ngram_size` (default 4): blocks any 4-token phrase from repeating.
- `repetition_penalty` (default 1.1), `top_p`, `top_k`, and `temperature` are unchanged.

These controls improve variety, but they do not guarantee anything about how a text will score on
an AI detector. Detector scores are unreliable: see the sources below.

## Files

```
inky/config.py       InkyConfig (all sizes and switches, scratch ids)
inky/layers.py       RMSNorm, RoPE, latent-KV attention, MoE, recurrent core, halting
inky/modeling.py     InkyModel, InkyCache (KV and SSM state), forward, generate
inky/scratch.py      reduce_scratch (PENCIL reduction rule)
inky/style.py        STOCK_PHRASES, bad-word and n-gram blocking
inky/arlo.py     Arlo (latent slots, pointer heads) for Choice and Noul
inky/optim.py        Muon + AdamW hybrid optimiser (ported from EOS)
scripts/build_untrained.py   build the model, print the budget, optionally save it
tests/test_inky.py   structural tests (model, scratch, prose controls)
tests/test_optim.py  optimiser routing, orthogonalisation, loss decrease
tests/test_decision.py  Arlo: distributions, equivariance, LM path untouched
```

## Usage

```bash
pip install -r requirements.txt
python -m pytest tests -q
python scripts/build_untrained.py --out checkpoints/inky-50m-init
```

```python
import torch
from inky import InkyConfig, InkyModel, build_bad_word_ids

torch.manual_seed(0)
model = InkyModel(InkyConfig())                   # untrained
print(model.count_parameters()["total"])          # about 50.2M

model.eval()
bad = build_bad_word_ids(lambda s: tokenizer.encode(s, add_special_tokens=False))
out = model.generate(
    prompt_ids,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.95,
    bad_word_ids=bad,
    reduce_scratch_blocks=True,                   # batch size 1
)
```

Tokenizer: `EOS_Tokenizer/` in the EOS project has 50,000 ids, and `<|endoftext|>` is id 49974.
Add three special tokens (`<|scratch|>` = 50000, `<|scratch_sep|>` = 50001, `<|/scratch|>` = 50002)
so that the scratch ids exist. Then load it with `tokenizers.Tokenizer.from_file` or
`PreTrainedTokenizerFast(tokenizer_file=...)`.

## Measured on CPU (8 threads, fp32, untrained)

| Context | Prefill | Decode per token |
| :--- | ---: | ---: |
| 1,024 | 2.3 s | 131 ms |
| 4,096 | 11.6 s | 169 ms |
| 8,186 | 25.0 s | 185 ms |

These figures were measured before the scratch and style changes. Those changes add no model
compute. Reduction re-runs the prefill for the shortened context. Timings describe one machine and
say nothing about quality.

## Limits

- **Untrained.** Output is random until it is trained on a text corpus. Nothing here is a claim
  about the quality of Inky's writing.
- **Scratch needs training.** The reduction mechanism is in place, but the model does not know
  when to use the scratch markers yet.
- **Batching needs equal lengths.** There is no padding mask. Use batch size 1 for generation and
  for scratch reduction.
- **Training at 8k is expensive.** Train at shorter lengths first, then extend. Gradient
  checkpointing is available via `InkyConfig(gradient_checkpointing=True)`.

## Sources for the style controls

- Perplexity and burstiness, the statistical signals detectors use, and their limits:
  [How do AI detectors work?](https://scribbr.com/ai-tools/how-do-ai-detectors-work/)
- Stock transitions and overused words in AI text: [AI patterns in writing](https://searchatlas.com/blog/ai-patterns-in-writing/)
- Detector reliability: [Counter Turing Test (arXiv:2310.05030)](https://arxiv.org/pdf/2310.05030)
- Scratch reduction: [PENCIL: Long Thoughts with Short Memory (arXiv:2503.14337)](https://arxiv.org/abs/2503.14337)
