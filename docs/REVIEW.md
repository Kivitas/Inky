# Engineering review

Scope: the Inky repository as of the current `main` (commit 7a69fc1). This is a review of the code,
the evidence, and the open decisions. It makes no claim about writing quality, because no model has
been trained or scored.

## Summary

Inky is a ~50M-parameter recurrent language model with an 8k window. The code is complete enough to
build, run and test (over 518 tests), and the training loop has been smoke-tested with `--dry-run` only.
The main gap is evidence: there is no corpus, no trained weights, no dense baseline and no measured
quality. Most design choices are therefore unvalidated, and the next work should test them rather
than add to them.

## Findings

### Parameter budget

- The embedding is 19.2M of 50.2M parameters (38%), because of the 50k vocabulary tied with the LM
  head. A smaller vocabulary (16-32k) would free a large share of the budget for depth or width.
- The recurrent core is 15.5M parameters, and the model applies it up to 6 times, so most of the
  compute goes to the shared core and the MoE experts.

### Decode speed

- Decode was measured at 131-185 ms per token on CPU (8 threads, fp32, untrained). This is a stale
  measurement from an earlier commit and has not been repeated.
- The two likely causes are the per-token re-concatenation of the latent cache in
  `MultiLatentAttention.forward` (the cache concat), which copies the whole cache on every step, and
  a host sync in `SharedExpertMoE._dispatch_routed`, which stalls the device once per token.
- The MoE dispatch change is now in (see "Done in this change set"). The cache concat is unchanged,
  and neither has been profiled on the current code.

### Architecture choices that need evidence

- **Halting.** PonderNet halting (arXiv 2107.05407, 2021) is used to weight loop outputs. Newer
  Mixture-of-Recursions routing (arXiv 2507.10524) saves training compute by routing tokens through
  fewer recursion steps. Inky has not compared the two, and PonderNet's KL term and geometric prior
  add training complexity.
- **MoE at this size.** Work reported in arXiv 2605.11689 (found via web search, not independently
  verified) suggests MoE is doubtful below about 20M
  active parameters under short training budgets. Inky sits near that scale, and whether it falls
  above or below the line depends on how the embedding is counted, so a dense baseline is needed to
  show that MoE helps at all.
- **Muon.** Per-matrix Newton-Schulz orthogonalisation adds per-step cost. A NeurIPS 2026 benchmark
  (neurips.cc/virtual/2026/161439; found via web search, not independently verified) reports that it can lose on wall-clock time at tiny scale. Inky
  uses the hybrid Muon and AdamW optimiser ported from EOS without a comparison against AdamW alone.
- **QK-norm and gated attention.** IMU-1 (arXiv 2602.02522; found via web search, not independently
  verified) reports these as stabilisers. Inky now has an opt-in `qk_norm` flag (default off, so no
  default behaviour changed) and still has no gated attention. QK-norm is untested on trained
  weights.
- **Highway.** The state-space memory highway is a causal running mean carried across loops. It has
  not been ablated, so its contribution is unknown.
- **Depth mismatch.** Training draws 2 to 6 loops per batch, while inference uses 6 by default. The
  model is therefore evaluated at a depth it was only partly trained for, unless `eval_by_depth` is
  used.

### Process and project gaps

- **No baseline.** Nothing has been scored against a dense model of the same size or a published
  reference, so no claim of improvement is possible yet.
- **No corpus.** Training needs a licensed or public-domain corpus. None is in the repository.
- **No post-training stage.** There is no instruction or style tuning, and no preference data.
- **No LICENSE.** The licence choice is the owner's decision. The repository cannot be reused
  until it is made.
- **No distributed training and no `torch.compile`.** Training runs on a single device without
  compilation, which limits throughput.
- **Lint and CI** now run in CI on Ubuntu and Windows (see below). Nothing else about the pipeline
  is validated, because no training run exists.

## Prioritised next steps

1. **Dense baseline first.** Train a dense model of matched total and active size on the same tokens
   and evaluation. Every later claim depends on this.
2. **Decode speed.** Preallocate the attention cache instead of re-concatenating it (the MoE dispatch
   sync is already reduced; see below). Re-measure on the current commit at 1k, 4k and 8k context.
3. **Ablations.** MoE against dense at matched active parameters; vocabulary 16-32k against 50k; the
   highway on and off. Each needs a held-out perplexity and the same token budget.
4. **QK-norm.** The `qk_norm` flag exists (off by default). Test it against the default on the same
   token budget, and note that decode costs more per token with it on.
5. **Mixture-of-Recursions-style routing.** Compare against PonderNet on training compute and
   held-out loss before adopting it.
6. **LICENSE.** Owner decision. Required before any outside use.
7. **Reproducibility.** Lint and CI are done; pin the data manifest and seeds for each ablation so
   that results can be reproduced.

## Done in this change set

These changes are in the working tree (not yet committed). They are code changes only; none has been
measured on trained weights.

- `qk_norm` config flag (opt-in, default off). Decode up-projects the latent cache each step when it
  is on, so it costs more per token.
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

See also `docs/TRAINING_PLAN.md` for the planned training stages and ablation table, and the
[critique status index](critique/STATUS.md) for the open findings.
