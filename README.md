# Inky

A ~50M-parameter recurrent language model for creative and text writing, with an 8k context
window, meant to run locally.

**Status: untrained.** This repository contains the model code, a training loop with a command-line
entry point (smoke-tested, including `--dry-run`, never run on a real corpus), decoding controls, an
evaluation harness, tooling scripts (`train`, `bench`, `ablate`, `sample`, `diagnose`, `arch_budget`), and 834 tests. There is no training corpus, no trained weights, no baseline
comparison, and no measured quality. Nothing here is a claim about how Inky writes.

Inky takes its language-model path from the EOS project (`Desktop/New folder/EOS LLM`) and scales
it down. It adds PENCIL-style scratch reduction (arXiv:2503.14337) and Arlo, a writing decision
maker ported from the Choice and Noul primitives in Rhen. Arlo reads hidden states and answers
"which option" or "yes or no" questions. It does not generate text, and it is not part of the
language-model path. It is off by default (`enable_arlo=False`). `train_arlo` can fit its
parameters on labelled examples with the language model frozen, but no trained decision
checkpoint exists, so its answers are untrained. `decide_*` warns while the engine is untrained;
`train_arlo` clears the warning, and so does `model.mark_arlo_trained()` after you load trained
Arlo weights. The other Rhen primitives (Score, Range, Extract,
Cascade) and its calibration layers are not ported.

**Licence:** proprietary, all rights reserved (see `LICENSE`). This is a private repository; access
does not grant any right to use, copy, modify or distribute it.

## Install

```bash
pip install -e .[dev]        # runtime dependencies plus pytest and ruff; run from the repository root
```

`pyproject.toml` pins `torch>=2.13,<3` and `transformers>=5.13,<6`. For a CPU-only PyTorch, install
it first from the PyTorch CPU index (`pip install torch --index-url https://download.pytorch.org/whl/cpu`).
`requirements.txt` lists the same runtime dependencies. The dev extras pin `ruff>=0.17,<0.18`; CI
runs `python -m ruff check .` and the tests on Ubuntu and Windows (Python 3.12 and 3.13).

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
| **Total** | **50.22 M** (50,219,202) | **50.61 M** (50,614,855) |

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
| State-space highway | Carries a per-position memory across loops. Its causal running mean is cached for decoding. It re-enters the stream through a learned scalar gain that starts at 0 (`highway_gain_init`), so the model begins without it. Optional selective-SSM mixer (`ssm_mixer_layers`, off by default) uses a chunked scan. | `StateSpaceMemoryHighway` |
| PonderNet halting | Each loop emits a stop probability. The halting distribution is pulled to a geometric prior (mean 3 loops) by a KL term. The output is the halting-weighted mix of loop states, mixed by direction and rescaled to the weighted mean RMS (`loop_mix_normalize`), so each loop's share is its halting probability. | PonderNet (arXiv 2107.05407, see docs/research/act.md) |
| Per-loop loss | Each loop's state is also trained to predict the next token, weighted by its halting probability (`step_ce_token_fraction`, default 0.25). | new in Inky |
| Chunked loss | Next-token cross-entropy and z-loss in chunks, so full 8k × 50k logits never exist at once. | `chunked_ce` |
| Scratch reduction | The model may draft notes, then close the block. Reduction deletes the notes and keeps the text. | PENCIL (arXiv:2503.14337) |

Config switches that change the attention, MoE and routing profile (all defaults keep the
50,219,202-parameter architecture above):

- `qk_norm` (opt-in, default `False`) and `qk_norm_mode` (`"head"` default, or `"latent"`):
  RMS-normalises queries and keys. With `"head"`, the key norm blocks absorption, so decoding
  up-projects the whole cached latent on every step and costs more per token. With `"latent"`, there
  is no `k_nope` norm (the latent is already RMS-normed) and `q_nope`, `q_pe` and `k_pe` are normed,
  so the key up-projection can still be folded into the query and cached decode matches the full
  path. `qk_norm_mode` is ignored when `qk_norm` is off. Both are untested on trained weights.
- `use_moe` (default `True`): `False` replaces every MoE feed-forward with one dense SwiGLU of width
  `dense_intermediate_size`. `None` (default) means (shared + top-k routed) × `intermediate_size`
  = 2,112, the MoE's active feed-forward width per token (router not counted).
- `use_highway` (default `True`): `False` does not build the state-space highway at all.
- Parameter counts with the switches off (from `model.count_parameters()`): no highway 50,165,505;
  dense 30,742,722; dense and no highway 30,689,025. The dense variant is smaller than the MoE
  default because it matches active, not total, feed-forward parameters. These are ablation
  switches; no variant has been trained or compared.
- `highway_gain_init` (default `0.0`): initial value of the learned scalar `out_gain` on the highway's
  output, `h + out_gain * RMSNorm(out_proj(mem))`. The RMSNorm output has unit scale whatever the
  stream's scale is, and the embeddings start at RMS 0.02, so the earlier fixed factor of 0.1 made the
  highway as large as the stream at initialisation (about 120% of the loop-1 input, measured with
  `scripts/diagnose.py`). With 0.0 the model starts without the highway and the gain opens only if it
  helps; while it is exactly 0 the highway's other weights get no gradient and the gain does. `0.1`
  reproduces the earlier scale.
- `loop_mix_normalize` (default `True`): the PonderNet mixture is `(sum_n p_n h_n / rms_n) * (sum_n p_n rms_n)`
  instead of `sum_n p_n h_n`. The stream grows with every loop, so in the plain sum a late loop counts for more
  than its `p_n`; the normalised form gives each loop exactly its halting share and keeps the output at the
  stream's scale. It is identical to the plain sum when all loop states have the same RMS. `False` is the plain
  sum. `forward_early_exit` follows the same setting.
- `balance_update` (default `"sign"`): how the MoE balance bias moves each optimizer step. `"sign"` moves every
  bias by exactly `balance_rate` towards uniform load (the DeepSeek-V3 rule); `"proportional"` is the earlier
  rule, whose step is at most `balance_rate / num_routed_experts`. In a toy run (tiny model, 8 experts, 200 steps,
  one seed) the expert-load spread (coefficient of variation) fell from 1.46 at step 10 to 0.35 at step 200
  with `"sign"` and 1e-3, against 1.19 with `"proportional"`. Neither rule prevented the collapse in the first
  steps there, and nothing here is measured on real data. Training also clears the MoE routing counters after
  each eval and each skipped step, so those passes no longer feed the next bias update.

Diagnostics: `python scripts/diagnose.py` (or `inky.loop_diagnostics(model, ids)`) reports the stream RMS
entering and leaving each loop, the loop-to-loop change, the highway's share, the halting weights and the
expert load, and `python scripts/train.py ... --diagnose-every N` logs the same during training. At
initialisation on random tokens, stream RMS grows 4.8x over the six loops with the earlier settings and 2.0x
with these defaults, and the core-layer expert-load coefficient of variation is 0.26 and 0.24 instead of 0.66
and 0.45. These are scale checks at initialisation, not quality results.

Comparison arms (Phases B, D and E of the architecture plan). Each one defaults to the architecture above,
adds no parameters until it is switched on, and has not been trained or compared. They exist so the
comparisons can be run: `scripts/ablate.py` has a variant for each and `scripts/arch_budget.py` tabulates
their cost. All of them keep cached decoding equal to the full forward pass (tested).

- `attention_type="gqa"` (with `num_kv_heads`, default 2): plain grouped-query attention instead of
  multi-latent attention. Queries and keys have 64 dims with RoPE on the last 16, values 64. Its cache is a
  `LatentKV` of flattened keys and values: 56 MiB against 31.5 MiB in bf16 at 8k. Decode attention costs 21.5
  KFLOP per cached token against 45.7 for the absorbed latent form (analytic and measured), so decode at 8k is
  about 294 MFLOP against 494. A CPU decode step at 4,096 tokens of context took 180 ms against 208 ms for the
  baseline (one run, untrained fp32). It supports the cache, packing and padding masks; `routing_mode="mor"`
  and `forward_early_exit` need the latent layout and are rejected. Its quality against MLA is unmeasured.
- `attention_gate=True`: a sigmoid per head, computed from the block input, multiplies the attention output
  (the gated-attention design of the Qwen team, reported at 1.7B dense and 15B MoE, not at this size). It adds
  `heads * (hidden + 1)` parameters per attention layer and works with both attention types and MoR.
- `routing_mode="last"`: no halting gate; the output is the last loop's state, and training still draws the
  loop count. The loss is the cross-entropy plus the small terms; `step_ce` and `ponder_kl` are reported as 0.
  With one loop it equals the PonderNet output (tested).
- `loop_input_injection=True`: every loop's block input also gets `inject_gain * (prelude output)`, with a
  per-channel gain that starts at 0, so the model starts identical to one without it. This is the simplest
  form of the re-injection used in recurrent-depth models.
- `shared_intermediate_size`: width of the shared expert, separate from the routed experts' `intermediate_size`.
  With it, 4 routed experts of width 1,408 with top-1 and a shared expert of 704 have the same active width and
  the same expert parameters per block as the default 8 x 704 with top-2 (the `big_experts` variant).
- `local_window` with `local_window_loops` (1-based loop numbers): in those loops a query attends to the last
  `local_window` positions only; the other loops and the pre and post layers stay global. The weights are
  shared, so only the mask changes, and one-token decode reads `local_window` cached positions instead of all of
  them (the cache still keeps every position). It works with MLA and GQA, with packing and with gradient
  checkpointing; it is rejected with MoR, and `forward_early_exit` raises. The `local_window` variant uses a
  window of 1/8 of the context (1,024 at 8k) in every loop but the last. That cuts decode at 8k from about 494 to
  260 MFLOP, and a CPU decode step at 4,096 tokens of context took 89 ms against 208 ms (one run, untrained
  fp32). That is the speed side only: what the window costs in quality is unmeasured.
- Hybrid layers: `ssm_mixer_layers` (a selective-SSM mixer in place of attention) now supports packed
  documents through `segment_ids`: the state is reset at each document boundary, and a packed row matches its
  documents run separately (tested at chunk edges). Padding masks (`attention_mask`) are still not supported.
  The `hybrid_ssm` variant makes the second recurrent layer an SSM mixer.

Cost tables: `python scripts/arch_budget.py --variants baseline,gqa,local_window` prints parameters, forward
FLOPs per token, decode FLOPs at several contexts (and extrapolated), and the KV cache. torch's FLOP counter
counts the CPU attention op (`scaled_dot_product_attention`) as zero, which hides GQA's attention entirely and
part of the prefill attention for both types, so the script registers the standard formula for it. Bits per byte
(`inky.bits_per_byte`, `inky.bytes_per_token`, `evaluate(..., bytes_per_token=...)`) is the loss to compare
across vocabulary sizes, where per-token perplexity is not comparable.
- `moe_dense_max_tokens` (default 64): the sync-free dense MoE dispatch is used only for small token
  counts (at most this many) on non-CPU devices. Larger batches use the grouped path. CPU always uses
  the grouped path.

Inference can also exit early per token with `forward_early_exit(ids, halt_threshold=...)`. It is
batch size 1, has no training path, and its speed and output drift are unmeasured on trained weights.

### Mixture-of-Recursions routing (`routing_mode="mor"`)

`routing_mode` is `"ponder"` (default, the PonderNet path above), `"last"` (see the comparison arms) or
`"mor"` (opt-in, Mixture-of-Recursions,
arXiv 2507.10524, token-choice variant). Fields: `mor_router_hidden` (default `None`: one linear
layer; an int gives an MLP), `mor_capacity_aux_coef` (default 0.01), `mor_gate_scale` (default
`True`), `mor_temperature` (default 1.0), `mor_target_depth` (default `None`, meaning
`ponder_prior_steps`, clipped into the loop range), `mor_sample_depth` (default `True`) and
`mor_gather` (default `True`). With `"mor"` the default config has 50,220,742 parameters (the router
replaces the halting unit).

Semantics:

- A causal router reads each token's own state (after the pre layers) and picks a depth between
  `min_recurrent_loops` and `max_recurrent_loops`. It sees no later token, so cached decoding can
  route the new token.
- Training samples the depth with Gumbel-max (`mor_sample_depth=True`); eval takes the argmax. The gate
  is the router probability of the chosen depth, which scales an active token's update
  (`mor_gate_scale`) and carries the router's gradient.
- In loop n only the tokens with depth >= n run (attention queries, key/value projections and the MoE
  work on the active rows only). An exited token keeps its state and its last latents, which later
  loops reuse as stale keys and values. The output is each token's state after its last active loop.
  `num_loops` is a cap on depth.
- Loss = `ce + z_loss_coef * z + moe_aux + mor_capacity_aux_coef * (E[depth] - target)^2`. The
  PonderNet halting gate, its KL and the per-loop CE are not used.

Limits: `ssm_mixer_layers` is rejected with `"mor"`; early exit with `halt_threshold` is not
supported (a token exits at its router depth); the gather path reads counts on the host, which may
graph-break under `torch.compile`; attention pads queries to the busiest row of each batch.
`mor_gather=False` is a masked reference path with the same results at full compute.

**Untrained and unmeasured.** MoR has not been trained, timed or compared with PonderNet. Nothing here
claims it matches PonderNet quality or saves wall-clock time.

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
- `--diagnose-every N` (opt-in, default 0): every N steps run `inky.loop_diagnostics` on the step's last
  micro-batch and add a `"diagnostics"` record to the history (stream RMS per loop, loop-to-loop change,
  highway gain, expert load), plus a log line. It costs one eval-mode forward pass and changes nothing else.
- `--early-stop-patience N` and `--early-stop-min-delta D` (opt-in, both off by default): stop after
  N consecutive evals without an eval-NLL improvement of more than D. This needs `--eval-every` and
  held-out batches. `best.pt` is unaffected.
- `train` warns when `scale_mode="moonlight"` is combined with a Muon LR that looks like the
  shape-mode default (0.02), since Moonlight scaling expects AdamW-sized learning rates (TRAIN-05).
- `--min-chars` (default 20), `--ckpt-every`, `--keep-last`, `--resume`, `--max-skipped-steps`.
- `--set KEY=VALUE` (repeatable, any `InkyConfig` field, value is a Python literal) and
  `--config-json PATH` (a JSON object of the same keys) launch ablation variants, for example
  `--set use_moe=False --set vocab_size=32003`. `--set` wins over `--config-json`, which wins over the
  dropout flags. Unknown keys are rejected. If `vocab_size` is overridden without the scratch ids,
  they become `vocab_size-3`, `-2`, `-1`. The overrides and the full config go to `train_config.json`.
- `--resume` reuses the checkpoint's config. An override that differs from it is an error, never a
  silent change.
- `--tokenizer-vocab-check {error,warn,off}` (default `error`): `config.vocab_size` must equal the
  tokenizer size + 3 scratch ids, with the scratch ids at or above the tokenizer size. A mismatch exits
  (or warns, or is skipped). The fix it prints is `--set vocab_size=<n+3>`.
- `--muon-batched` runs Newton-Schulz once per matrix shape instead of once per matrix. It gives the
  same results to float tolerance and the same state-dict format; its speed-up is not measured here.
- `--compile` wraps the training forward in `torch.compile`. If compilation fails it warns and
  falls back to eager. `--dry-run` skips it on Windows (it needs a C++ compiler there). It has not
  been run on CUDA, and its speed-up is unmeasured.
- Training records carry `tokens_per_sec` and `tokens_seen` (cumulative, including skipped steps and
  resumed history); the log line prints both.
- `--warmup-steps` defaults to 5% of `--steps`.

Pipeline: `data.prepare_corpus` (clean, quality filter, streaming near-duplicate removal, drop of
documents that contain a reserved id, encode, pack), then `inky.train.train`. Checkpoints go to
`<out-dir>/checkpoints/step_N/` (weights, optimiser state, step, history, `rng.pt`,
`train_config.json`). `best.pt` holds the lowest-eval-NLL weights. Only the newest `--keep-last`
step directories are kept.

Training is not run anywhere in this repository (and nothing has been tested on CUDA). The dry run uses random characters and a
character-level encoder, and it only checks that the code runs.

Other training code:

- `inky.arlo.train_arlo(model, examples, ...)` trains only the Arlo parameters on labelled choice or
  yes/no examples. Every other parameter is frozen, and the language model runs under `no_grad`.
- Left padding: the shifted loss reads `labels[:, 1:]`, so the first label of each row must be -100.
  Check this before using left padding.

## Usage

```bash
pip install -e .[dev]
python -m pytest tests -q               # 834 tests collected
python -m ruff check .                  # lint, also run in CI
python scripts/build_untrained.py --out checkpoints/inky-50m-init
```

### Tooling scripts

```bash
# Decode and prefill timing (median over repeats; untrained weights, timing only)
python scripts/bench.py --context-lengths 256,1024,4096 --threads 8 --repeats 5 --json bench.json
python scripts/bench.py --tiny --context-lengths 16,32 --decode-tokens 4 --repeats 1   # seconds on a CPU

# Ablation runner for the table in docs/TRAINING_PLAN.md
python scripts/ablate.py --list-variants
python scripts/ablate.py --synthetic --tiny --steps 20 --out-dir runs/ablate
python scripts/ablate.py --corpus data/*.jsonl --tokenizer-path path/to/tokenizer.json --steps 500 \
    --variants baseline,dense,no_highway,qk_norm --out-dir runs/ablate

# Cost of architecture variants: parameters, FLOPs per token, decode FLOPs by context, KV cache
python scripts/arch_budget.py --variants baseline,gqa,local_window --contexts 2048,4096 --extrapolate-to 8192
python scripts/arch_budget.py --tiny --variants baseline,dense,core3x4 --json budget.json

# Architecture scale report: stream RMS per loop, loop change, highway share, expert load
python scripts/diagnose.py --tiny --seq-len 64
python scripts/diagnose.py --set highway_gain_init=0.1 --set loop_mix_normalize=False   # the earlier behaviour

# Generate from a checkpoint with the prose controls
python scripts/sample.py --dry-run
python scripts/sample.py --checkpoint runs/inky/checkpoints/step_500 --tokenizer-path tokenizer.json \
    --prompt "The lighthouse keeper said" --max-new-tokens 120 --temperature 0.8 --top-p 0.95 \
    --repetition-penalty 1.1 --no-repeat-ngram-size 4 --bad-words "tapestry,delve" --default-bad-words
```

- `scripts/bench.py`: times one cached prefill and `--decode-tokens` cached decode steps at each
  `--context-lengths` entry (default 256,1024), median of `--repeats`. Takes `--set KEY=VALUE`,
  `--num-loops`, `--threads`, `--batch-size`, `--tiny`, `--device`, `--json`. Peak memory is the CUDA
  peak on a GPU; on CPU it is the peak resident set of the process, which never decreases.
- `scripts/ablate.py`: variants are `baseline`, `dense`, `no_highway`, `qk_norm`, `qk_norm_latent`,
  `vocab16k`, `vocab32k`, `fewer_loops`, `fixed_depth`, `more_experts`, `mor` (`--list-variants`). Each trains
  on the same windows and seed and is scored on the same held-out windows; the output is a markdown
  table plus JSON (`--out-dir`, `--json`, `--markdown`). `--synthetic` uses a seeded random-token
  corpus (no files; it only checks the plumbing). A variant the config or data cannot support is
  listed as skipped, never run with a key ignored. The optimiser, halting and scratch rows are not
  config toggles and are out of scope. No real ablation has been run.
- `scripts/sample.py`: `--checkpoint` accepts a `save_pretrained` dir, a `step_N` dir, `best.pt`, or a
  checkpoints dir (`best.pt` wins, else the newest step). `--tokenizer-path` is a tokenizers JSON file
  or a directory holding `tokenizer.json`. Flags: `--prompt`, `--max-new-tokens`, `--temperature`,
  `--top-p`, `--top-k`, `--repetition-penalty`, `--no-repeat-ngram-size`, `--repeat-window`,
  `--bad-words`, `--default-bad-words`, `--reduce-scratch-blocks`, `--num-loops`, `--greedy`,
  `--seed`, `--device`, `--dry-run`. An untrained model writes noise.

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
print(model.count_parameters()["total"])    # 50,219,202 for the default config (about 50.22M)

prompt_ids = torch.tensor([encode("The lighthouse keeper said")])
out = model.generate(
    prompt_ids,
    max_new_tokens=64,
    temperature=0.8,
    top_p=0.95,
    bad_word_ids=build_bad_word_ids(encode),
    # eos_token_id defaults to config.eos_token_id, so generation already stops at <|endoftext|>.
    # Pass eos_token_id=None to never stop at EOS.
    reduce_scratch_blocks=True,             # batch size 1; needs the scratch ids in the tokenizer
)
print(tokenizer.decode(out[0, prompt_ids.shape[1]:].tolist()))
```

Evaluation: `evaluate(model, batches)` scores at the model's maximum depth by default. With
`train_draw=True` it also reports `train_draw_depths`, `nll_train_draw` and `perplexity_train_draw`:
the mean over the integer depths from the minimum to the maximum training depth, which is what a
training batch sees on average. The extra scoring passes cost time, so the flag is off by default.

## Decode speed (CPU)

The latent-KV decode cache is a growable buffer (`LatentKV` in `inky/layers.py`): a decode step
writes the new latents into spare capacity (geometric growth, up to `max_position_embeddings`)
instead of calling `torch.cat` on the whole cache every token. Forks and snapshots copy on their
first append, so shared buffers are never overwritten. A measurement on CPU at 2,000 tokens of context
gave about 88 ms per token with the growable cache against about 92 ms before, so the change is small,
not a large speed-up.

The earlier table (131 / 169 / 185 ms per token at 1,024 / 4,096 / 8,186 tokens, with 2.3 to 25 s
of prefill) came from an older commit. It is stale and has not been measured again since. Re-measure
with `scripts/bench.py` before quoting any speed. Every timing describes one machine and untrained
weights, and says nothing about quality.

The chunked selective-SSM scan (off by default) was measured at 1.8× faster than the token loop at
2048 tokens. At 8192 it keeps about 2 GiB of activations per SSM application, so it needs gradient
checkpointing.

## Files

```
inky/__init__.py        public exports
inky/config.py          InkyConfig (sizes, switches, dropout, scratch ids, Arlo settings)
inky/layers.py          RMSNorm, RoPE, latent-KV attention (growable LatentKV cache), MoE, recurrent core, highway, selective SSM mixer, PonderNet halting, MoR router
inky/modeling.py        InkyModel: forward, forward_early_exit, loss terms, MoR routing
inky/cache.py           InkyCache (decoding state) and InkyOutput
inky/generation.py      InkyGenerationMixin: generate (sampling, prose controls, scratch reduction) and its helpers
inky/decision.py        InkyDecisionMixin: decide_choice, decide_binary, the Arlo "trained" flag
inky/scratch.py         reduce_scratch (PENCIL reduction rule, note cap)
inky/style.py           stock phrases, bad-word and n-gram blocking, soft penalties
inky/arlo.py            Arlo (latent slots, pointer heads, decision losses, train_arlo) for Choice and Noul
inky/optim.py           Muon + AdamW hybrid optimiser (ported from EOS), parameter routing, optional batched Newton-Schulz
inky/train.py           training loop: WSD schedule, loop curriculum, grad accumulation, AMP option, eval, skip on non-finite loss, atomic checkpoints, resume
inky/data.py            text cleaning, quality filter, streaming MinHash near-duplicate removal, reserved-id drop, packing
inky/distill.py         distillation loss (not wired into training)
inky/evaluation.py      held-out split, perplexity, loop statistics, early-exit drift
inky/diagnostics.py     loop_diagnostics: stream RMS per loop, loop change, highway share, expert load
inky/metrics.py         reports for binary decisions, choices and generated text
inky/quantize.py        int8 and int4 weight-only quantization (inference)
inky/speculative.py     self-speculative greedy decoding (no writing controls)
scripts/build_untrained.py   build the model, print the budget, optionally save it
scripts/train.py        command-line training entry point (--dry-run available)
scripts/bench.py        prefill and per-token decode timing
scripts/ablate.py       ablation runner (named variants, --synthetic mode)
scripts/sample.py       text generation from a checkpoint with the prose controls
scripts/diagnose.py     architecture scale report (stream RMS per loop, highway gain, expert load)
scripts/arch_budget.py  cost table for named variants (params, FLOPs, decode FLOPs by context, KV cache)
LICENSE                 proprietary, all rights reserved
tests/                  pytest suite (run with `python -m pytest tests -q`; 834 tests collected)
docs/critique/          critique findings and their status (STATUS.md is the index)
docs/research/          research notes (ACT / PonderNet, training)
```

## Limits

- **Untrained.** Output is random until the model is trained on a text corpus. No quality claim
  is made or measured.
- **No corpus, no trained weights, no baseline.** A training run needs a licensed or public-domain
  corpus and a compute budget. The project has neither yet.
- **No CUDA testing, no distributed training.** Nothing has been run on CUDA. Multi-GPU training (DDP)
  is not implemented.
- **Ablation switches, comparison arms and MoR are untrained.** `use_moe`, `use_highway`, `qk_norm_mode`,
  `routing_mode="mor"` and `"last"`, `attention_type="gqa"`, `attention_gate`, `loop_input_injection`,
  `shared_intermediate_size`, `local_window_loops` and the `scripts/ablate.py` variants exist, and every variant
  has been run end to end for a few steps on random tokens, but none has been trained on text or compared.
  Which of them to keep, and deleting the ones that lose (the plan's Phase C), waits for those results.
- **Scratch needs training.** The reduction mechanism is in place, but the model does not know
  when to use the scratch markers.
- **Batching.** `forward` accepts an `attention_mask` for uncached passes, and `segment_ids` for
  packed documents. Padding is excluded from the highway's running mean on uncached passes, but the
  MoE router's z-loss still counts pad positions (CORE-10). The selective-SSM mixer packs documents with
  `segment_ids` but has no padding mask. Generation and scratch reduction need batch size 1 in practice.
- **Training at 8k is expensive.** Train at shorter lengths first, then extend. Gradient
  checkpointing is available via `InkyConfig(gradient_checkpointing=True)`.

## Review and plan

- [docs/REVIEW.md](docs/REVIEW.md): engineering review, findings, and prioritised next steps.
- [docs/TRAINING_PLAN.md](docs/TRAINING_PLAN.md): training plan, token-budget guideline, corpus
  options, stages and ablation table. This is a plan, not a result.

## Known issues

Open items, checked against the current code. The full list with evidence is in
[docs/critique/STATUS.md](docs/critique/STATUS.md).

Project-level:

- **No trained weights and no training data.** No quality, perplexity or preference result exists.
- **No baseline comparison and no evaluation run.** The harness exists, but nothing has been scored
  against a reference model (GEN-14).
- **Licence.** A `LICENSE` file is present: proprietary, all rights reserved, private repository.
  (The earlier "no LICENSE file" item is resolved.)
- **The tokenizer is external.** It lives in the EOS project, and the three scratch ids must be added.
- **Decode speed is unmeasured on the current commit.** The growable cache measured about 88 against
  about 92 ms per token at 2,000 tokens of context on CPU. The older 131-185 ms table is stale.
- **Early-exit saving is unmeasured.** `forward_early_exit` cuts loop work on the untrained model
  (about a third at threshold 0.5), with a large output drift. The training forward runs every loop
  by design (CORE-01, accepted design).

Behaviour changes from the recent fixes (these change results compared with earlier runs):

- **Phase A architecture fixes.** The highway output now has a learned gain that starts at 0 (+1 parameter,
  50,219,202 in total), the PonderNet mixture is scale-preserving, and the MoE balance rule is `"sign"`. The
  old behaviour is `--set highway_gain_init=0.1 --set loop_mix_normalize=False --set balance_update=proportional`.
  Nothing has been trained either way; runs made before this change are not comparable with later ones.

- **Default `min_chars` is 20**, not 200. Short texts (haiku, short dialogue) now enter the corpus.
- **`step_ce_token_fraction` defaults to 0.25.** Each loop's loss is a sampled estimate of the full
  value, so loss curves are not comparable with runs that used the full loss.
- **Scratch notes are capped at 256 tokens.** A longer block keeps its text.
- **Reserved ids.** `prepare_corpus` drops any document that contains a reserved id, and `generate`
  blocks them by default.
- **Top-k sampling ties.** Ties at the k-th logit keep exactly k candidates.
- **`no_repeat_ngram_size=1` raises** instead of silently banning every seen token.
- **`generate` stops at EOS by default.** The default `eos_token_id` now means
  `config.eos_token_id` (DOC-05). Before, it never stopped unless you passed an id. Pass
  `eos_token_id=None` to keep the old never-stop behaviour. Speculative decoding follows the same
  rule.
- **Vocabulary size is a config choice.** `vocab_size` defaults to 50,003 (the embedding is 38% of the
  parameters). A 16k or 32k vocabulary is available through `--set` or `scripts/ablate.py`, but no
  ablation has been run. The training CLI refuses a tokenizer whose size + 3 differs from `vocab_size`.
- **MoE dispatch.** The sync-free dense path is used only up to `moe_dense_max_tokens` (default 64)
  tokens on non-CPU devices; larger inputs use the grouped path.
- **Evaluation.** `evaluate(train_draw=True)` also reports `nll_train_draw` and
  `perplexity_train_draw`.
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
- **Arlo.** Its query comes only from the prefix (ARLO-03), and each option's prefix is recomputed
  rather than cached (ARLO-02). The untrained warning (ARLO-01) is fixed: `decide_*` warns until
  `train_arlo` runs or `mark_arlo_trained()` is called.
- **Position encoding.** RoPE uses 16 dimensions with base 5e5 at 8k (CORE-05, accepted design).
- **Training gaps.** `distillation_loss` is not chunked and not wired into training (TRAIN-15).
  Fixed in this change set: early stopping is available but opt-in (`early_stop_patience`,
  `early_stop_min_delta`; TRAIN-04), and the Moonlight scale option warns about a shape-mode learning
  rate (TRAIN-05).
- **fp16 AMP is untested** here (no CUDA).
- **Dedupe is probabilistic.** A near-duplicate pair at similarity 0.8 can be missed (about 1e-7 per
  pair), but no non-duplicate is dropped.

## Sources for the style controls

- Perplexity and burstiness, the statistical signals detectors use, and their limits:
  [How do AI detectors work?](https://scribbr.com/ai-tools/how-do-ai-detectors-work/)
- Stock transitions and overused words in AI text: [AI patterns in writing](https://searchatlas.com/blog/ai-patterns-in-writing/)
- Detector reliability: [Counter Turing Test (arXiv:2310.05030)](https://arxiv.org/pdf/2310.05030)
- Scratch reduction: [PENCIL: Long Thoughts with Short Memory (arXiv:2503.14337)](https://arxiv.org/abs/2503.14337)
