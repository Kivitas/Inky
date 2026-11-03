# Inky

A ~50M-parameter recurrent language model for creative and text writing, with an 8k context
window, meant to run locally.

**Status: untrained.** This repository contains the model code, a training loop with a command-line
entry point (smoke-tested, including `--dry-run`, never run on a real corpus), decoding controls, an
evaluation harness, and 490 tests. There is no training corpus, no trained weights, no baseline
comparison, and no measured quality. Nothing here is a claim about how Inky writes.

Inky takes its language-model path from the EOS project (`Desktop/New folder/EOS LLM`) and scales
it down. It adds PENCIL-style scratch reduction (arXiv:2503.14337) and Arlo, a writing decision
maker ported from the Choice and Noul primitives in Rhen. Arlo reads hidden states and answers
"which option" or "yes or no" questions. It does not generate text, and it is not part of the
language-model path. It is off by default (`enable_arlo=False`). `train_arlo` can fit its
parameters on labelled examples with the language model frozen, but no trained decision
checkpoint exists, so its answers are untrained. The other Rhen primitives (Score, Range, Extract,
Cascade) and its calibration layers are not ported.

## Install

```bash
pip install -e .[dev]        # runtime dependencies plus pytest; run from the repository root
```

`pyproject.toml` pins `torch>=2.13,<3` and `transformers>=5.13,<6`. For a CPU-only PyTorch, install
it first from the PyTorch CPU index (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).
`requirements.txt` lists the same runtime dependencies.

## Parameter budget

Counts from `scripts/build_untrained.py` and `model.count_parameters()`:

| Section | Default (`enable_arlo=False`) | With Arlo (`enable_arlo=True`) |
| :--- | ---: | ---: |
| Embedding, tied with LM head (50,003 × 384) | 19.20 M | 19.20 M |
| Pre layer (1 block) | 7.74 M | 7.74 M |
| Recurrent core (2 blocks, shared across loops) | 15.54 M | 15.54 M |
| Post layer (1 block) | 7.74 M | 7.74 M |
| Norms and halting unit | < 0.01 M | < 0.01 M |
| Decision engine (8 slots, width 128, 4 pointer heads) | 0 | 0.40 M |
| **Total** | **50.22 M** (50,219,201) | **50.61 M** (50,614,854) |

Each block is a latent-KV attention layer (about 0.44 M) plus a shared-expert MoE: one shared
expert and eight routed experts, with two routed experts active per token (about 7.3 M).

Effective depth is 1 + 2 × 6 + 1 = **14 layer applications**. Training samples between 2 and 6
loops per batch (or follows a ramp with `--loop-curriculum`). Inference uses 6 loops unless
`num_loops` is passed. Training always runs every loop.

## Architecture

```
tokens → embedding
       → pre layer
       → recurrent core, looped up to 6 times:
           FiLM step modulation (per loop index)
           state-space memory highway (causal running mean, carried across loops)
           2 × block (latent-KV attention + shared-expert MoE), layer dropout 0.1
           damped residual update
           PonderNet halting: per-token weights over the loops, sum to 1
       → post layer
       → RMSNorm → tied LM head (logit soft-cap 30)
```

| Component | What it does | Source in EOS |
| :--- | :--- | :--- |
| Multi-latent attention | Keys and values come from one 128-d latent per token. The RoPE key is shared across heads. The KV cache stores only the latent. | `MultiLatentAttention` |
| Absorbed decoding | For one new token, the key projection is folded into the query, so cached latents are never up-projected. Matches the full path in fp32 tests. | MLA matrix absorption |
| Shared-expert MoE | One always-on SwiGLU expert plus 8 routed experts (top-2, sigmoid router). Aux-loss-free balancing via a bias buffer. | `SharedExpertFineGrainedMoE` |
| Recurrent core | One set of shared layers applied repeatedly. Gives depth without extra parameters. | `RecurrentCoreBlock` |
| Step modulation | FiLM scale and shift per loop index. Starts as the identity. | `StepModulation` |
| State-space highway | Carries a per-position memory across loops. Its causal running mean is cached for decoding. Optional selective-SSM mixer (`ssm_mixer_layers`, off by default) uses a chunked scan. | `StateSpaceMemoryHighway` |
| PonderNet halting | Each loop emits a stop probability. The halting distribution is pulled to a geometric prior (mean 3 loops) by a KL term. The output is the halting-weighted mix of loop states. | PonderNet (arXiv 2107.05407, see docs/research/act.md) |
| Per-loop loss | Each loop's state is also trained to predict the next token, weighted by its halting probability (`step_ce_token_fraction`, default 0.25). | new in Inky |
| Chunked loss | Next-token cross-entropy and z-loss in chunks, so full 8k × 50k logits never exist at once. | `chunked_ce` |
| Scratch reduction | The model may draft notes, then close the block. Reduction deletes the notes and keeps the text. | PENCIL (arXiv:2503.14337) |

Inference can also exit early per token with `forward_early_exit(ids, halt_threshold=...)`. It is
batch size 1, has no training path, and its speed and output drift are unmeasured on trained weights.

### Scratch reduction (PENCIL)

PENCIL's rule is `C [CALL] T [SEP] A [RETURN] ⇒ C A`: intermediate thoughts `T` are deleted from
context, and only the answer `A` stays. The paper applies this to logic puzzles. Inky applies the
same rule to writing: the model drafts notes between `scratch_open` and `scratch_sep`, writes the
text, and closes with `scratch_close`. The notes are removed, so the 8k window holds only prose.

- The reduction is `inky.scratch.reduce_scratch`. It handles nested blocks, leaves unclosed blocks
  and stray markers as ordinary text, and never edits the prompt.
- Notes are capped at `max_note_tokens` (default 256, `None` for no cap). A block whose notes grow
  past the cap keeps its text instead of deleting it.
- `generate(..., reduce_scratch_blocks=True)` rebuilds the cache for the shortened context after
  each reduction, at the same loop count as before, and recomputes only the text after the marker.
  It needs batch size 1.
- The three scratch ids (50000–50002) must be added to the EOS tokenizer as special tokens. The
  model has no way to learn this behaviour until it is trained. Until then it is inert.

Honest caveat: PENCIL's gains are measured on puzzles with long reasoning traces. Whether it helps
prose is a hypothesis to test after training.

### Prose controls

`generate` has these decoding controls:

- `bad_word_ids`: blocks token sequences. Build them with
  `build_bad_word_ids(encode)`. The default phrase list in `inky/style.py` is a hand-written
  starting point. Matching is on token ids, so other spellings can get through.
- `blocked_token_ids`: never emitted. The default (`None`) blocks the reserved ids (the EOS
  tokenizer's special tokens and the scratch ids); pass a list to change that.
- `no_repeat_ngram_size` (default 4): blocks any repeated n-gram in the sequence (prompt included).
  `1` is rejected, because it would ban every token already seen. `0` turns the rule off.
- `repeat_window` (opt-in, default `None`): limits the repetition penalty and the no-repeat rule to
  the last N tokens. The default uses the whole history.
- `repetition_penalty` (default 1.1), `top_p`, `top_k` (default 50), and `temperature`. With top-k,
  ties at the k-th logit keep exactly k candidates.
- `count_prose_only` (opt-in, needs scratch reduction): `max_new_tokens` counts only prose tokens,
  not notes. The default counts notes.
- `stock_word_penalty` (default 0, off): a soft logit penalty for single stock words
  (`style.soft_penalty_token_ids`). It is opt-in.

These controls are intended to add variety. They have not been measured on any model, and they do
not guarantee anything about how a text will score on an AI-text detector. Detector scores are
unreliable (see the sources below).

**Detector evasion is not a validated capability of this project.** Nothing here has been tested
against an AI-text detector, and the style controls should not be read as a way to get past one.

## Training

Command-line entry point: `scripts/train.py`. Run `python scripts/train.py --help` for all flags.

```bash
python scripts/train.py --dry-run --steps 3 --out-dir runs/dry          # no corpus or tokenizer needed
python scripts/train.py --corpus data/*.jsonl --tokenizer-path path/to/tokenizer --out-dir runs/inky
python scripts/train.py --corpus book.txt --tokenizer-path path/to/tokenizer --resume runs/inky/checkpoints/step_500
```

Flags worth knowing:

- `--grad-accum N` (effective batch = `--batch-size` × N) and `--amp {none,bf16,fp16}` (default none).
  fp16 needs CUDA, which is not available here, so that path is untested.
- `--loop-curriculum` ramps the loop count up over training. `--eval-by-depth` also logs perplexity at
  every training depth.
- `--embedding-dropout` and `--residual-dropout` (both default 0). Residual dropout applies to each
  block application, so when enabled the shared core drops the same blocks at every loop.
- `--min-chars` (default 20), `--ckpt-every`, `--keep-last`, `--resume`, `--max-skipped-steps`.
- `--warmup-steps` defaults to 5% of `--steps`.

Pipeline: `data.prepare_corpus` (clean, quality filter, streaming near-duplicate removal, drop of
documents that contain a reserved id, encode, pack), then `inky.train.train`. Checkpoints go to
`<out-dir>/checkpoints/step_N/` (weights, optimiser state, step, history, `rng.pt`,
`train_config.json`). `best.pt` holds the lowest-eval-NLL weights. Only the newest `--keep-last`
step directories are kept.

Training is not run anywhere in this repository. The dry run uses random characters and a
character-level encoder, and it only checks that the code runs.

Other training code:

- `inky.arlo.train_arlo(model, examples, ...)` trains only the Arlo parameters on labelled choice or
  yes/no examples. Every other parameter is frozen, and the language model runs under `no_grad`.
- Left padding: the shifted loss reads `labels[:, 1:]`, so the first label of each row must be -100.
  Check this before using left padding.

## Usage

```bash
pip install -e .[dev]
python -m pytest tests -q
python scripts/build_untrained.py --out checkpoints/inky-50m-init
```

The tokenizer is not in this repository. It is the `EOS_Tokenizer/` folder of the EOS project
(50,000 ids; `<|endoftext|>` is id 49974). Inky's scratch markers need three more ids, 50000 to 50002,
which are not in that tokenizer yet. Add them before using `reduce_scratch_blocks=True`.

```python
import torch
from tokenizers import Tokenizer
from inky import InkyConfig, InkyModel, build_bad_word_ids

TOKENIZER_FILE = "path/to/EOS_Tokenizer/tokenizer.json"   # external; not part of this repository
tokenizer = Tokenizer.from_file(TOKENIZER_FILE)

def encode(text: str) -> list:
    return tokenizer.encode(text, add_special_tokens=False).ids

config = InkyConfig()                       # enable_arlo=False by default
torch.manual_seed(0)
model = InkyModel(config).eval()            # untrained: the text it writes is random
print(model.count_parameters()["total"])    # 50,219,201 for the default config (about 50.22M)

prompt_ids = torch.tensor([encode("The lighthouse keeper said")])
out = model.generate(
    prompt_ids,
    max_new_tokens=64,
    temperature=0.8,
    top_p=0.95,
    bad_word_ids=build_bad_word_ids(encode),
    eos_token_id=config.eos_token_id,       # stop at <|endoftext|>
    reduce_scratch_blocks=True,             # batch size 1; needs the scratch ids in the tokenizer
)
print(tokenizer.decode(out[0, prompt_ids.shape[1]:].tolist()))
```

## Measured on CPU (8 threads, fp32, untrained)

| Context | Prefill | Decode per token |
| :--- | ---: | ---: |
| 1,024 | 2.3 s | 131 ms |
| 4,096 | 11.6 s | 169 ms |
| 8,186 | 25.0 s | 185 ms |

These figures are from an earlier commit and have not been re-measured. They describe one machine
and say nothing about quality.

The chunked selective-SSM scan (off by default) was measured at 1.8× faster than the token loop at
2048 tokens. At 8192 it keeps about 2 GiB of activations per SSM application, so it needs gradient
checkpointing.

## Files

```
inky/__init__.py        public exports
inky/config.py          InkyConfig (sizes, switches, dropout, scratch ids, Arlo settings)
inky/layers.py          RMSNorm, RoPE, latent-KV attention, MoE, recurrent core, highway, selective SSM mixer, PonderNet halting
inky/modeling.py        InkyModel, InkyCache, forward, forward_early_exit, decide_*, generate
inky/scratch.py         reduce_scratch (PENCIL reduction rule, note cap)
inky/style.py           stock phrases, bad-word and n-gram blocking, soft penalties
inky/arlo.py            Arlo (latent slots, pointer heads, decision losses, train_arlo) for Choice and Noul
inky/optim.py           Muon + AdamW hybrid optimiser (ported from EOS), parameter routing
inky/train.py           training loop: WSD schedule, loop curriculum, grad accumulation, AMP option, eval, skip on non-finite loss, atomic checkpoints, resume
inky/data.py            text cleaning, quality filter, streaming MinHash near-duplicate removal, reserved-id drop, packing
inky/distill.py         distillation loss (not wired into training)
inky/evaluation.py      held-out split, perplexity, loop statistics, early-exit drift
inky/metrics.py         reports for binary decisions, choices and generated text
inky/quantize.py        int8 and int4 weight-only quantization (inference)
inky/speculative.py     self-speculative greedy decoding (no writing controls)
scripts/build_untrained.py   build the model, print the budget, optionally save it
scripts/train.py        command-line training entry point (--dry-run available)
tests/                  pytest suite (run with `python -m pytest tests -q`; 490 tests collected)
docs/critique/          critique findings and their status (STATUS.md is the index)
docs/research/          research notes (ACT / PonderNet, training)
```

## Limits

- **Untrained.** Output is random until the model is trained on a text corpus. No quality claim
  is made or measured.
- **No corpus, no trained weights, no baseline.** A training run needs a licensed or public-domain
  corpus and a compute budget. The project has neither yet.
- **Scratch needs training.** The reduction mechanism is in place, but the model does not know
  when to use the scratch markers.
- **Batching.** `forward` accepts an `attention_mask` for uncached passes, and `segment_ids` for
  packed documents. Padding is excluded from the highway's running mean on uncached passes, but the
  MoE router's z-loss still counts pad positions (CORE-10). Generation and scratch reduction need
  batch size 1 in practice.
- **Training at 8k is expensive.** Train at shorter lengths first, then extend. Gradient
  checkpointing is available via `InkyConfig(gradient_checkpointing=True)`.

## Known issues

Open items, checked against the current code. The full list with evidence is in
[docs/critique/STATUS.md](docs/critique/STATUS.md).

Project-level:

- **No trained weights and no training data.** No quality, perplexity or preference result exists.
- **No baseline comparison and no evaluation run.** The harness exists, but nothing has been scored
  against a reference model (GEN-14).
- **No LICENSE file.** The licence choice is the owner's to make.
- **The tokenizer is external.** It lives in the EOS project, and the three scratch ids must be added.
- **Early-exit saving is unmeasured.** `forward_early_exit` cuts loop work on the untrained model
  (about a third at threshold 0.5), with a large output drift. The training forward runs every loop
  by design (CORE-01, accepted design).

Behaviour changes from the recent fixes (these change results compared with earlier runs):

- **Default `min_chars` is 20**, not 200. Short texts (haiku, short dialogue) now enter the corpus.
- **`step_ce_token_fraction` defaults to 0.25.** Each loop's loss is a sampled estimate of the full
  value, so loss curves are not comparable with runs that used the full loss.
- **Scratch notes are capped at 256 tokens.** A longer block keeps its text.
- **Reserved ids.** `prepare_corpus` drops any document that contains a reserved id, and `generate`
  blocks them by default.
- **Top-k sampling ties.** Ties at the k-th logit keep exactly k candidates.
- **`no_repeat_ngram_size=1` raises** instead of silently banning every seen token.
- **Arlo is off by default** and `decide_*` restores the train/eval mode it found.
- **Residual dropout** (when enabled) applies on every loop, so the shared core drops the same
  blocks at each application.
- **Quantized decoding is slow on CPU by default.** Weights are dequantized on every call. Set
  `cache_dequantized=True` for the fast path (GEN-04).

Model and decoding (from `docs/critique/`):

- **Scratch needs training.** The model does not yet use the scratch markers.
- **Repetition penalty and no-repeat rule use the whole history by default** (GEN-06, GEN-07).
  `repeat_window` is opt-in.
- **Blocks match token ids.** Bad-word and stock-phrase blocks miss other spellings. The decoded-text
  check exists in `text_report` only (GEN-08, GEN-09).
- **Speculative decoding is greedy only** and has no writing controls (GEN-03).
- **Loop depth differs between training and evaluation by default.** Training draws a depth from 2 to
  6 per batch, while evaluation and generation use 6 (TRAIN-08). `eval_by_depth` is opt-in.
- **The output is a halting-weighted mixture of loop states,** so `num_loops` changes the function,
  not only the depth (CORE-02, accepted design).
- **Arlo.** `decide_*` does not warn when the engine is untrained (ARLO-01). Its query comes only from
  the prefix (ARLO-03), and each option's prefix is recomputed rather than cached (ARLO-02).
- **Position encoding.** RoPE uses 16 dimensions with base 5e5 at 8k (CORE-05, accepted design).
- **Training gaps.** No early stopping (TRAIN-04). The Moonlight scale option has no warning about its
  learning rate (TRAIN-05). `distillation_loss` is not chunked and not wired into training (TRAIN-15).
- **fp16 AMP is untested** here (no CUDA).
- **Dedupe is probabilistic.** A near-duplicate pair at similarity 0.8 can be missed (about 1e-7 per
  pair), but no non-duplicate is dropped.

## Sources for the style controls

- Perplexity and burstiness, the statistical signals detectors use, and their limits:
  [How do AI detectors work?](https://scribbr.com/ai-tools/how-do-ai-detectors-work/)
- Stock transitions and overused words in AI text: [AI patterns in writing](https://searchatlas.com/blog/ai-patterns-in-writing/)
- Detector reliability: [Counter Turing Test (arXiv:2310.05030)](https://arxiv.org/pdf/2310.05030)
- Scratch reduction: [PENCIL: Long Thoughts with Short Memory (arXiv:2503.14337)](https://arxiv.org/abs/2503.14337)
