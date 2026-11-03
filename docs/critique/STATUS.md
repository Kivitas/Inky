# Critique status

Snapshot of the findings in the five critique files, checked against the code on 2026-11-03
(commit f7299ea plus uncommitted changes, including the second round of fixes). Each file has a
status table at its top with the evidence for every finding. The finding text in those files is
unchanged, and their line numbers are from an older commit.

Status categories:

- **FIXED**: the fix is in the code. Where the fix is opt-in, the row says so, and the default
  behaviour is unchanged.
- **OPEN**: the problem remains, or is only partly fixed (the row says which part).
- **ACCEPTED-DESIGN**: a deliberate design choice, not a defect. It is documented, and the code was
  not changed for it.
- **NOT-A-BUG**: checked, and the claim does not hold for the code as designed.
- **UNVERIFIED**: could not be checked here (none remain).

## Counts

| File | FIXED | OPEN | ACCEPTED-DESIGN | NOT-A-BUG | Total |
| :--- | ---: | ---: | ---: | ---: | ---: |
| [core.md](core.md) | 8 | 4 | 3 | 1 | 16 |
| [data_docs.md](data_docs.md) | 14 | 1 | 0 | 0 | 15 |
| [arlo_metrics.md](arlo_metrics.md) | 10 | 2 | 0 | 0 | 12 |
| [generation.md](generation.md) | 10 | 4 | 0 | 0 | 14 |
| [training.md](training.md) | 12 | 3 | 0 | 0 | 15 |
| **All** | **54** | **14** | **3** | **1** | **72** |

Fixed by opt-in only (the default is unchanged): GEN-04 (`cache_dequantized`), GEN-06 and GEN-07
(`repeat_window`), GEN-13 (`count_prose_only`), TRAIN-04 (dropout, no early stopping). CORE-08 is
listed as OPEN because its default (`lm_loss_recompute=True`) still runs the second head pass.

Tests: 490 collected and 490 passed (`python -B -m pytest tests -q -p no:cacheprovider`, 2 warnings,
about 5 minutes on CPU, run 2026-11-03).

Passing tests show that the code runs. They do not show that the open items are fixed. DOC-05 still
has a test that does not set an EOS id, so it cannot detect the default-stop problem.

## Open items by theme

- **Measured quality is absent.** No trained weights, no corpus, no baseline, no evaluation run
  (GEN-14, and the project-level items in README Known issues).
- **Scratch and decoding.** Scratch needs training to be used. Speculative decoding is greedy only
  with no writing controls (GEN-03). Bad-word and stock-phrase blocks match token ids only (GEN-08,
  GEN-09).
- **Speed and dtype.** The default quantized path dequantizes on every call (GEN-04, fast path is
  opt-in). The default LM loss still recomputes the head in backward (CORE-08).
- **Model design (accepted).** The output is a halting-weighted mixture (CORE-02). Training runs every
  loop, so early exit saves no training compute (CORE-01). RoPE uses 16 dimensions (CORE-05).
  The RMSNorm on the highway output is unablated (CORE-14, NOT-A-BUG).
- **Model design (open).** The MoE router z-loss still averages over pad positions (CORE-10, partly
  fixed in the highway). The train-mode cached forward has no assertion (CORE-15). The decode
  prefix is recomputed per option (CORE-16).
- **Decision engine (Arlo).** `decide_*` does not warn when untrained (ARLO-01). The prefix is
  recomputed per option (ARLO-02). The query comes only from the prefix (ARLO-03).
- **Training.** Train and evaluation depth differ by default (TRAIN-08). The Moonlight LR has no
  guard (TRAIN-05). No early stopping (TRAIN-04). `distillation_loss` is neither chunked nor wired in
  (TRAIN-15).
- **Docs and tests.** The default EOS stop in `generate` is still unset (DOC-05).

Behaviour changes from the fixes (see README "Behaviour changes"): default `min_chars` 20, default
`step_ce_token_fraction` 0.25, scratch notes capped at 256 tokens, reserved ids dropped from the data
and blocked in `generate`, top-k ties keep exactly k candidates, `no_repeat_ngram_size=1` raises,
Arlo off by default.
