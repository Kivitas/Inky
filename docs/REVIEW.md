# Engineering review

Scope: the Inky repository as of the current working tree on `main` (last commit 711efeb, plus uncommitted changes). This is a review of the code,
the evidence, and the open decisions. It makes no claim about writing quality, because no model has
been trained or scored.

## Summary

Inky is a ~50M-parameter recurrent language model with an 8k window. The code is complete enough to
build, run and test (713 tests collected), and the training loop has been smoke-tested with `--dry-run` only.
The main gap is evidence: there is no corpus, no trained weights, no baseline run (the dense switch
exists but was never trained) and no measured quality. Most design choices are therefore unvalidated, and the next work should test them rather
than add to them.

## Findings

### Parameter budget

- The embedding is 19.2M of 50.2M parameters (38%), because of the 50k vocabulary tied with the LM
  head. A smaller vocabulary (16-32k) would free a large share of the budget for depth or width.
- The recurrent core is 15.5M parameters, and the model applies it up to 6 times, so most of the
  compute goes to the shared core and the MoE experts.

### Decode speed

- Decode was measured at 131-185 ms per token on CPU (8 threads, fp32, untrained) at an earlier
  commit. That table is stale and has not been measured again since; do not quote it.
- Two suspected causes were the per-token re-concatenation of the latent cache and a host sync in the
  MoE dispatch. Both changes are now in (see "Done in this change set"). The growable `LatentKV` cache
  measured only about 88 against about 92 ms per token on CPU at 2,000 tokens of context, so the cache
  concat was not the main cost. The remaining cost has not been profiled.

### Architecture choices that need evidence

- **Halting.** PonderNet halting (arXiv 2107.05407, 2021) is used to weight loop outputs. Newer
  Mixture-of-Recursions routing (arXiv 2507.10524) saves training compute by routing tokens through
  fewer recursion steps. Inky now implements it as opt-in `routing_mode="mor"`, but it is untrained
  and unmeasured, and the two have not been compared. PonderNet's KL term and geometric prior add
  training complexity.
- **MoE at this size.** Work reported in arXiv 2605.11689 (found via web search, not independently
  verified) suggests MoE is doubtful below about 20M
  active parameters under short training budgets. Inky sits near that scale, and whether it falls
  above or below the line depends on how the embedding is counted, so a dense baseline is needed to
  show that MoE helps at all.
- **Muon.** Per-matrix Newton-Schulz orthogonalisation adds per-step cost. A NeurIPS 2026 benchmark
  (neurips.cc/virtual/2026/161439; found via web search, not independently verified) reports that it can lose on wall-clock time at tiny scale. Inky
  uses the hybrid Muon and AdamW optimiser ported from EOS without a comparison against AdamW alone.
  An optional batched Newton-Schulz step (`--muon-batched`) now exists; its speed-up is unmeasured.
- **QK-norm and gated attention.** IMU-1 (arXiv 2602.02522; found via web search, not independently
  verified) reports these as stabilisers. Inky now has an opt-in `qk_norm` flag (default off, so no
  default behaviour changed), with `qk_norm_mode` `"head"` or `"latent"` (the latent mode keeps the
  cached decode path absorbed), and still has no gated attention. QK-norm is untested on trained
  weights.
- **Highway.** The state-space memory highway is a causal running mean carried across loops. It has
  not been ablated, so its contribution is unknown. It now enters the stream through a learned gain that
  starts at 0 (see "Phase A" below). `use_highway=False` (50,165,505 parameters) and the
  `no_highway` variant in `scripts/ablate.py` exist for the test.
- **Depth mismatch.** Training draws 2 to 6 loops per batch, while inference uses 6 by default. The
  model is therefore evaluated at a depth it was only partly trained for, unless `eval_by_depth` is
  used.

### Process and project gaps

- **No baseline run.** `use_moe=False` builds a dense variant (30,742,722 parameters, which matches
  the MoE's active feed-forward width, not its total), but nothing has been trained or scored against
  it or a published reference, so no claim of improvement is possible yet.
- **No CUDA testing.** Everything was run on CPU. `--amp fp16`, `--compile` and the MoE dense dispatch
  path on a device have never run on a GPU.
- **No corpus.** Training needs a licensed or public-domain corpus. None is in the repository.
- **No post-training stage.** There is no instruction or style tuning, and no preference data.
- **Licence: resolved.** A `LICENSE` file is present: proprietary, all rights reserved, private
  repository.
- **No distributed training.** DDP is not implemented, so training runs on a single device. A
  `--compile` flag now exists (falls back to eager on failure); it is untested on CUDA and unmeasured.
- **Lint and CI** now run in CI on Ubuntu and Windows (see below). Nothing else about the pipeline
  is validated, because no training run exists.

## Prioritised next steps

1. **Dense baseline first.** Train a dense model of matched total and active size on the same tokens
   and evaluation. Every later claim depends on this.
2. **Decode speed.** The cache is now growable and the MoE dispatch sync is reduced (see below).
   Re-measure with `scripts/bench.py` on the current commit at 1k, 4k and 8k context, then profile
   what is left.
3. **Ablations.** `scripts/ablate.py` has the variants (`dense`, `no_highway`, `qk_norm`,
   `qk_norm_latent`, `vocab16k`, `vocab32k`, `fixed_depth`, ...). None has been run on a real corpus.
   The vocabulary reduction is a config choice (`--set vocab_size=...`), not yet ablated. Each run
   needs a held-out perplexity and the same token budget.
4. **QK-norm.** Test `qk_norm` (head) and `qk_norm_latent` against the default on the same token
   budget; head mode costs more per token in decode.
5. **Mixture-of-Recursions routing.** `routing_mode="mor"` exists. Compare it against PonderNet on
   training compute and held-out loss before adopting it. Known limits: no `ssm_mixer_layers`, no
   `halt_threshold` early exit, host reads in the gather path may graph-break under `torch.compile`,
   attention pads to the busiest row.
6. **Reproducibility.** Lint and CI are done; pin the data manifest and seeds for each ablation so
   that results can be reproduced. Multi-device (DDP) training is not implemented.

## Done in this change set

These changes are in the working tree (not yet committed). They are code changes only; none has been
measured on trained weights.

Phase A (architecture scale and balance fixes). Measured at initialisation on random tokens with
`scripts/diagnose.py`, not on trained weights:

- **Highway gain** (`highway_gain_init`, default 0.0). The highway added `0.1 * RMSNorm(...)` to a stream whose
  RMS is 0.02 to 0.3, i.e. 120% of the loop-1 input, and its per-parameter gradient was about 47x that of the
  core MoE. It now starts switched off behind a learned scalar (+1 parameter).
- **Scale-preserving loop mixture** (`loop_mix_normalize`, default True). The stream RMS grew 4.8x over six
  loops (0.058 to 0.278), so `p_n` was not each loop's real share of the plain sum. With the gain at 0 the
  growth is 2.0x, and the mixture now gives each loop its halting share. `forward_early_exit` matches `forward`.
- **MoE balance rule** (`balance_update`, default `"sign"`). The old step was at most `balance_rate / 8`. A
  toy run (tiny model, 200 steps, one seed) removed the imbalance with the sign rule and not with the old
  one. `train` now also clears the routing counters after each eval and each skipped step: before, eval
  batches doubled the counts the next update saw (checked by removing the reset: 192 instead of 96 assignments).
- **Diagnostics** (`inky/diagnostics.py`, `scripts/diagnose.py`, `train --diagnose-every`).
- At initialisation the core-layer expert-load coefficient of variation fell from 0.66 and 0.45 to 0.26 and 0.24.
  That fits the old highway pushing a near-constant vector into the router, but no training run confirms it.

The remaining architecture items (matched-compute comparison of vocabulary size and unique layers against
loops, MoE against dense, MLA against GQA, attention output gate, loop mechanism, input injection) are
comparisons that need training runs and are not started.

Done in the latest round:

- **Growable decode cache** (`LatentKV` in `inky/layers.py`): no per-token `torch.cat`. Measured
  about 88 against about 92 ms per token on CPU at 2,000 tokens, so a small gain only.
- **Latent QK-norm** (`qk_norm_mode="latent"`): keeps absorbed cached decode with `qk_norm` on.
- **Dense baseline switch** (`use_moe=False`, `dense_intermediate_size`) and **highway switch**
  (`use_highway=False`). Parameter counts: default 50,219,202; no highway 50,165,505; dense
  30,742,722; dense and no highway 30,689,025. Nothing has been trained with them.
- **Mixture-of-Recursions routing** (`routing_mode="mor"`, `mor_*` fields): token-choice causal depth
  router, active tokens only per loop, stale-KV reuse. Untrained and unmeasured.
- **Batched Muon** (`--muon-batched`) and **`--compile`** (falls back to eager).
- **Training CLI**: `--set`, `--config-json`, resume refuses a changed config,
  `--tokenizer-vocab-check`, `tokens_per_sec` / `tokens_seen` logging.
- **Scripts**: `scripts/bench.py`, `scripts/ablate.py`, `scripts/sample.py`.
- **LICENSE**: proprietary, all rights reserved, private repository.

Earlier in this change set:

- `qk_norm` config flag (opt-in, default off). In `"head"` mode decode up-projects the latent cache
  each step, so it costs more per token.
- MoE dispatch: the sync-free dense path is used only for small token counts on non-CPU devices
  (`moe_dense_max_tokens`, default 64); the grouped path is used otherwise. Decode speed on the new
  path has not been re-measured.
- Opt-in early stopping (`early_stop_patience`, `early_stop_min_delta`, CLI `--early-stop-patience`
  and `--early-stop-min-delta`; TRAIN-04).
- Moonlight learning-rate warning (TRAIN-05).
- `evaluate(train_draw=True)` also reports `nll_train_draw` and `perplexity_train_draw`.
- `generate` EOS default (behaviour change, DOC-05): the default `eos_token_id` now means
  `config.eos_token_id`, and an explicit `None` means never stop at EOS. Speculative decoding follows
  the same rule.
- Arlo untrained warning (ARLO-01), cleared by `train_arlo` or `mark_arlo_trained()`.
- Ruff lint in CI, with an Ubuntu and Windows matrix.

Still not done: no corpus, no trained weights, no baseline or ablation run, no CUDA testing, no DDP.

See also `docs/TRAINING_PLAN.md` for the planned training stages and ablation table, and the
[critique status index](critique/STATUS.md) for the open findings.
