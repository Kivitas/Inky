# Inky

A ~50M-parameter recurrent language model for creative and text writing, with an 8k context
window. It is built to run locally. This repository contains the untrained model code, a training loop (smoke-tested only), the
writing-controls, and tests. There is no training data and no training run yet.

Inky takes its language-model path from the EOS project (`Desktop/New folder/EOS LLM`) and scales
it down. Inky adds PENCIL-style scratch reduction from arXiv:2503.14337, and Arlo, a writing decision
maker ported from Rhen's Choice and Noul primitives. Arlo reads the model's states
and answers "which option" or "yes or no" questions. It doesn't generate text, and it never
changes the language-model path. Arlo has a training loss (`choice_loss`, `binary_loss`), but no
training has run yet, so its answers are untrained. Rhen's other primitives (Score, Range, Extract, Cascade) and its
calibration layers are not ported.

## Parameter budget (50.61M)

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
loops per batch. Inference uses 6 loops unless `num_loops` is passed. Training always runs every loop.
For inference, `forward_early_exit(ids, halt_threshold=...)` lets tokens stop early (batch size 1, no
generation cache). It cuts the token-loop work, but the output drifts from the full forward pass, so
check it against the evaluation harness before using it (see below).

## Architecture

```
tokens → embedding
       → pre layer
       → recurrent core, looped up to 6 times:
           FiLM step modulation (per loop index)
           state-space memory highway (causal running mean, carried across loops)
           2 × block (latent-KV attention + shared-expert MoE), loop dropout 0.1
           damped residual update
           PonderNet halting: per-token weights over the loops, sum to 1
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
| PonderNet halting | Each loop emits a stop probability; the halting distribution is pulled to a geometric prior (mean 3 loops) by a KL term, and each loop's state is trained on its own, weighted by its halting probability. Training runs every loop. Inference can exit early per token (`forward_early_exit`). | PonderNet (arXiv 2107.05407, see docs/research/act.md) |
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
  each reduction, so the cache matches the text the model sees. The rebuild uses the default loop
  count rather than the `num_loops` the caller passed, so text after a reduction can be computed at
  a different depth (see Known issues). It needs batch size 1.
- The three scratch ids (50000–50002) must be added to `EOS_Tokenizer` as special tokens. The
  model has no way to learn this behaviour until it is trained. Until then it is inert.

Honest caveat: PENCIL's gains are measured on puzzles with a long reasoning trace. Whether it helps
prose is a hypothesis to test after training.

### Prose controls

`generate` has three decoding controls aimed at writing quality:

- `bad_word_ids`: blocks token sequences. Build them with
  `build_bad_word_ids(tokenizer_encode)`. The default list in `inky/style.py` is a hand-written
  starting point of stock phrases (for example "delve", "tapestry", "furthermore", "it's important
  to note"). It is not a validated detector.
- `no_repeat_ngram_size` (default 4): blocks any 4-token phrase from repeating.
- `repetition_penalty` (default 1.1), `top_p`, `top_k`, and `temperature` are unchanged.

These controls are intended to add variety. They have not been measured, and they do not guarantee
anything about how a text will score on an AI detector. Detector scores are unreliable: see the sources below.

## Files

```
inky/config.py       InkyConfig (all sizes and switches, scratch ids)
inky/layers.py       RMSNorm, RoPE, latent-KV attention, MoE, recurrent core, PonderNet halting
inky/modeling.py     InkyModel, InkyCache (KV and SSM state), forward, generate
inky/scratch.py      reduce_scratch (PENCIL reduction rule)
inky/style.py        STOCK_PHRASES, bad-word and n-gram blocking
inky/arlo.py         Arlo (latent slots, pointer heads) for Choice and Noul
inky/optim.py        Muon + AdamW hybrid optimiser (ported from EOS)
inky/data.py         text cleaning, quality filter, near-duplicate removal, packing
inky/train.py        training loop, schedule, loop curriculum, checkpoints
inky/evaluation.py   held-out evaluation
inky/metrics.py      evaluation reports
inky/quantize.py     weight quantization
inky/speculative.py  self-speculative decoding
inky/distill.py      distillation loss
scripts/build_untrained.py   build the model, print the budget, optionally save it
tests/test_inky.py   structural tests (model, scratch, prose controls)
tests/test_optim.py  optimiser routing, orthogonalisation, loss decrease
tests/test_decision.py  Arlo: distributions, equivariance, LM path untouched
tests/test_data.py, tests/test_train.py, tests/test_evaluation.py, tests/test_quantize.py,
tests/test_speculative.py, tests/test_distill.py, tests/test_metrics.py
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
print(model.count_parameters()["total"])          # about 50.6M

model.eval()
bad = build_bad_word_ids(lambda s: tokenizer.encode(s, add_special_tokens=False))
out = model.generate(
    prompt_ids,
    max_new_tokens=512,
    temperature=0.8,
    top_p=0.95,
    bad_word_ids=bad,
    eos_token_id=model.config.eos_token_id,       # stop at <|endoftext|>
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

## Known issues

High-severity findings from the critique files in `docs/critique/`. Each one is open.

- DATA-01: the quality filter rejects Hindi, Thai and Chinese text ([data_docs.md](docs/critique/data_docs.md)).
- DATA-02: dedupe and packing hold all data in memory and do not scale to a real corpus ([data_docs.md](docs/critique/data_docs.md)).
- TEST-01: the speculative-decoding tests pass on a degenerate output and cannot detect a broken decoder ([data_docs.md](docs/critique/data_docs.md)).
- TEST-02: the bad-word test cannot fail, so the ban is unverified ([data_docs.md](docs/critique/data_docs.md)).
- CORE-01: partly addressed. Inference-only early exit with stored keys and values is in `forward_early_exit`. It cuts token-loop work to about a third at threshold 0.5 on the untrained model, with a large output drift (relative 0.2 to 0.5). Training is unchanged, and the drift is unmeasured on trained weights ([core.md](docs/critique/core.md), [act.md](docs/research/act.md)).
- CORE-02: each output is a weighted mix over loop states, so changing `num_loops` changes the function, not only the depth ([core.md](docs/critique/core.md)).

## Sources for the style controls

- Perplexity and burstiness, the statistical signals detectors use, and their limits:
  [How do AI detectors work?](https://scribbr.com/ai-tools/how-do-ai-detectors-work/)
- Stock transitions and overused words in AI text: [AI patterns in writing](https://searchatlas.com/blog/ai-patterns-in-writing/)
- Detector reliability: [Counter Turing Test (arXiv:2310.05030)](https://arxiv.org/pdf/2310.05030)
- Scratch reduction: [PENCIL: Long Thoughts with Short Memory (arXiv:2503.14337)](https://arxiv.org/abs/2503.14337)
