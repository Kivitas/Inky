# Training and optimisation critique

Scope: `inky/optim.py`, `inky/train.py`, `inky/distill.py`, `inky/config.py`, and the training path of `inky/modeling.py` and `inky/layers.py`. No source files were edited and no training was run.

Verification: the existing suite passes (`101 passed`). Items marked **confirmed (run)** were checked with a short script that calls the project's own functions. Items marked **confirmed (read)** were verified from the code but not executed. Items marked **read** are reasoning from the code only.

Ordered by importance.

---

## TRAIN-01 (high): no resume path; optimiser state and schedule position are not saved

- **Where:** `inky/train.py:166-167` (checkpoint writes only `model.save_pretrained`); `inky/train.py:121` (loop always starts at step 1).
- **What is wrong:** A checkpoint holds only the model weights. The Muon momentum buffers, AdamW moments, `_base_lrs`, the step counter, the loop-curriculum position, the batch cycle position and the RNG state are all lost. `train()` has no `start_step` argument, so a resumed run re-runs warmup and the curriculum from zero. `HybridOptimizer.state_dict()` exists (`optim.py:237`) but is never called.
- **Why it matters:** A multi-day run cannot survive a crash without a restart that changes the optimiser dynamics. Early resumes after warmup would also spike the loss, because Adam's second-moment estimates restart from zero.
- **Fix:** Save `{"model": ..., "optimizer": optimizer.state_dict(), "step": step, "history": history, "rng": ...}` in each checkpoint (`torch.save` next to `save_pretrained`). Add `start_step` (or `resume_from`) to `train()` so the schedules and the data cursor resume at the right place. Add a test that a 10-step run and a 5+5 resumed run give the same weights.
- **Status:** confirmed (read).

## TRAIN-02 (high): the bulk of the parameters get no weight decay

- **Where:** `inky/optim.py:186` (`muon_weight_decay: float = 0.0`), `optim.py:142` (decay only applied when `wd > 0`), `optim.py:148-173` (`split_parameters`).
- **What is wrong:** Every 2D matrix goes to Muon, and Muon's weight decay defaults to 0 and is never overridden anywhere in the repo. Only the tied embedding (in the AdamW decay group) gets `adamw_weight_decay=0.1`. Running `split_parameters` on the default config gave:
  - Muon: 31,400,704 params (62%) with wd = 0. These are the attention projections, expert SwiGLUs, router, SSM highway and halting matrices.
  - AdamW decay: 19,207,552 params (the embedding only).
  - AdamW no decay: 6,598 params (norms, biases, scalars).
- **Why it matters:** The `HybridOptimizer` docstring says norms and scalars get no decay, which implies matrices do. The intent is not implemented. With no weight decay, no dropout on the matrices and a small creative-writing corpus, overfitting is the expected outcome (see TRAIN-04).
- **Fix:** Pass a non-zero `muon_weight_decay` (about 0.1 is the common Moonlight/Muon setting; tune it). Decide explicitly whether the router and halting gate (TRAIN-14) should decay. Add a test that asserts the wd value per group.
- **Status:** confirmed (run).

## TRAIN-03 (high): no guard against NaN or Inf gradients or loss

- **Where:** `inky/optim.py:228-235` (`HybridOptimizer.step`); `inky/train.py:132-136`.
- **What is wrong:** `clip_grad_norm_` returns a non-finite norm when any gradient is non-finite, and the step is still taken. Muon's Newton-Schulz iteration then propagates NaN into every matrix, and the loop continues. Nothing checks `loss` or `grad_norm` before the step, and checkpoints are written on schedule (`train.py:166`), so a poisoned model can be saved and reported as a finished run.
- **Why it matters:** One bad batch (a rare token, a fp overflow, a router collapse) silently destroys the run. A bf16 or long run makes this more likely.
- **Fix:** In `train()`, check `torch.isfinite(loss)` and the grad norm. If either is non-finite, zero the grads, skip `optimizer.step`, count the skip, and stop after N consecutive skips. Also log the count in the history. Add a test that injects a NaN loss and checks the weights are unchanged.
- **Status:** confirmed (read).

## TRAIN-04 (high): overfitting risk with almost no regularisation, and no model selection

- **Where:** `inky/layers.py:400-402` (the only dropout: a whole recurrent layer is skipped with probability 0.1, decided once per batch, per layer, per loop); `inky/config.py:39` (`loop_dropout_prob=0.1`); `inky/train.py:154-163` (eval is logged only).
- **What is wrong:** There is no attention, residual, MLP or embedding dropout. Layer skipping is decided per batch, so every sample in a batch shares it. The eval loss is printed but no "best" checkpoint is kept, there is no early stopping, and the final checkpoint is the one that is kept. The `loop_dropout_prob` name suggests loop-level dropout, but the code drops individual layers.
- **Why it matters:** A 50M-parameter model with 38% of its parameters in the tied embedding is large for a creative-writing corpus. Without regularisation and with no selection on held-out loss, a long run will memorise.
- **Fix:** Add optional residual and embedding dropout (default off, sweep on). Keep the best eval checkpoint by `eval_nll` and report it. Add early stopping on eval loss. Rename `loop_dropout_prob` to `layer_dropout_prob` or make it per-sample. Compare train and eval loss in the logs.
- **Status:** confirmed (read).

## TRAIN-05 (medium): Muon's `moonlight` scale and the default LR do not match

- **Where:** `inky/optim.py:185` (`muon_lr=0.02`), `optim.py:75-80` (`moonlight` scale), `optim.py:139` (`alpha = lr * scale`); `PLAN.md:138` ("moonlight gives ~2x update size").
- **What is wrong:** The default Muon LR is 0.02. In `shape` mode the per-element RMS of the update is about `lr / sqrt(max(rows, cols))`, which gives about 9.5e-4 for a 384x384 matrix. That is close to AdamW's 6e-4, so the default is consistent. Moonlight's scale is designed for LRs in AdamW units, so with `lr=0.02` it is far too large. Measured per-element RMS at `lr=0.02`:
  - 384x384 `shape`: 9.5e-4. `moonlight`: 3.7e-3 (about 3.9x).
  - 704x384 `shape`: 9.8e-4. `moonlight`: 3.8e-3 (about 3.9x).

  The PLAN note says "about 2x", which is wrong; the measured factor is about 4x.
- **Why it matters:** Anyone who enables `scale_mode="moonlight"` (the option is added so it can be tuned) gets an update roughly six times AdamW's, which is likely to diverge or make the loss curve misleading.
- **Fix:** When `moonlight` is used, set `muon_lr` to an AdamW-scale value (about `adamw_lr`, 6e-4 to 1e-3) and make `train()` or `HybridOptimizer` refuse or warn on the combination `moonlight` with `muon_lr > 0.005`. Correct the PLAN.md figure.
- **Status:** confirmed (run).

## TRAIN-06 (medium): no warmup by default, so the first step runs at full LR from random init

- **Where:** `inky/train.py:80` (`warmup_steps: int = 0`), `inky/train.py:36` (`warm = 1.0` when warmup is 0).
- **What is wrong:** With the default, `wsd_lr_scale(0, 1000, 0) == 1.0`, so the first optimiser step uses the full Muon LR (0.02) and the full AdamW LR (6e-4) on a freshly initialised model. When warmup is on, step 1 gets scale 0 (`wsd_lr_scale(0, 1000, 100) == 0.0`), so the first step is wasted but harmless.
- **Why it matters:** Muon with LR 0.02 at step 1, with no warmup, is the most common cause of an early loss spike or divergence in these setups.
- **Fix:** Make the default warmup a fraction of the run (for example `max(100, 0.02 * steps)`), or make `warmup_steps` required. Document the chosen value next to `decay_fraction`.
- **Status:** confirmed (run).

## TRAIN-07 (medium): a final checkpoint is not written unless `checkpoint_every > 0`

- **Where:** `inky/train.py:110-111` and `inky/train.py:166`; docstring at `inky/train.py:94-95`.
- **What is wrong:** The condition is `ckpt_root is not None and checkpoint_every > 0 and (step % checkpoint_every == 0 or is_last)`. The docstring says checkpoints are written "at the last step", but if `checkpoint_every` is 0 (its default) nothing is saved, not even the last step. Passing only `checkpoint_dir` silently produces no output.
- **Why it matters:** A caller who sets a directory and forgets the interval gets no weights and no error.
- **Fix:** `if ckpt_root is not None and (is_last or (checkpoint_every > 0 and step % checkpoint_every == 0)):`. Add a test for this case.
- **Status:** confirmed (read).

## TRAIN-08 (medium): training depth is random and unseeded when the curriculum is off, and does not match eval

- **Where:** `inky/modeling.py:103-106` (`random.randint(min, max)` during training, `max` at eval); `inky/train.py:83` (`use_loop_curriculum=False` by default); `inky/modeling.py:3` (`import random`).
- **What is wrong:** With the default settings the model trains at a random depth 2..6 per batch, drawn from Python's `random`, which `torch.manual_seed` does not control. Evaluation and generation always use depth 6. The model is therefore trained on a distribution it is not evaluated on, and runs are not reproducible from the seed.
- **Why it matters:** Reported eval numbers depend on an unseeded draw and on a train/eval mismatch. Ablations (PLAN step 8) would be noisy.
- **Fix:** Use a `torch.Generator` seeded from the run seed for the depth draw, or make the curriculum the default. Document that eval uses the maximum depth and check that the training distribution covers it.
- **Status:** confirmed (read).

## TRAIN-09 (medium): router z-loss is counted once per recurrent loop, so its weight changes during training

- **Where:** `inky/modeling.py:211` (`moe_aux = moe_aux + aux` inside the loop); `inky/layers.py:398-406` (the core sums its layers' aux); `inky/layers.py:266` (`z_loss` per layer).
- **What is wrong:** The core's MoE aux (router z-loss) is added on every loop iteration, so the core's share is multiplied by the loop count (2 to 6). Pre and post layers are counted once. With the curriculum the effective weight therefore rises from about 2x to 6x the configured `router_z_loss_coef` for the core, which changes the objective partway through training.
- **Why it matters:** It is a hidden schedule on the loss, and the value is not what `router_z_loss_coef=1e-3` suggests.
- **Fix:** Either average the core aux over the loop count or add it once per call, and make the choice explicit in the config comment. Log `moe_aux_loss` separately in `train()` so its size can be seen.
- **Status:** read (the code paths are clear; the size of the effect is not measured).

## TRAIN-10 (medium): balance bias is driven by the last loop's routing only

- **Where:** `inky/layers.py:255-257` (`expert_load` overwritten on every call); `inky/layers.py:270-272` (`update_balance_bias`); `inky/modeling.py:326-330`; `inky/train.py:136`.
- **What is wrong:** The core's MoE module is called once per loop. Each call overwrites `expert_load`, so after a forward pass the buffer holds only the last loop's counts. `update_moe_balance()` is applied once per optimiser step, so it reacts to one loop's routing and ignores the others. Eval and generation forwards also overwrite the buffer, which does not affect training only because `update_moe_balance` runs right after the training forward.
- **Why it matters:** The aux-loss-free balancing for the shared core is driven by a noisy, partial signal. Expert collapse in the core would be detected late or not at all.
- **Fix:** Accumulate per-expert counts across all calls in the step (in training mode only), and reset them in `update_balance_bias`. Consider a separate buffer per call site.
- **Status:** read.

## TRAIN-11 (medium): the data path holds the whole training set in memory and replays it in the same order

- **Where:** `inky/train.py:101-104` (`batches = list(train_batches)`, `itertools.cycle`); `inky/data.py:109` (`pack_sequences` joins documents with EOS, no attention masking across documents).
- **What is wrong:** `train()` materialises all batches in RAM, and `itertools.cycle` replays the same order every epoch with no reshuffle. Packed sequences let attention cross document boundaries. The loss also includes positions whose target is the EOS or the first token of the next document.
- **Why it matters:** Memory will not scale past a small corpus. Fixed order across epochs slightly hurts generalisation. Cross-document attention is a common source of quality loss on packed creative text.
- **Fix:** Accept an iterable that is re-created per epoch (or a `DataLoader` with a seeded sampler), reshuffle each epoch, and store the epoch and cursor in the checkpoint (TRAIN-01). Decide and document whether document-boundary masking is wanted.
- **Status:** confirmed (read).

## TRAIN-12 (medium): no gradient accumulation and no mixed precision; per-step GPU syncs

- **Where:** `inky/train.py:130-135` (one forward and backward per step, fp32); `inky/train.py:141` (`float(loss.detach())` each step); `inky/layers.py:400` (`float(torch.rand(()))` per layer per loop).
- **What is wrong:** There is no `autocast` or `GradScaler` and no accumulation argument, so the effective batch size is limited by memory. The per-layer `float(torch.rand(()))` forces a CPU-GPU sync up to 12 times per forward pass, and the loss is also synced every step.
- **Why it matters:** A 50M model on a single GPU can use bf16 autocast for roughly a 2x speed-up. Without accumulation the batch size needed for a stable LR may not fit.
- **Fix:** Add `grad_accum_steps` (scale the loss and step every N micro-batches), an `amp_dtype` option using `torch.autocast`, and generate the layer-skip mask with a tensor op and no `float()`. Sync only at log steps.
- **Status:** confirmed (read).

## TRAIN-13 (low): the LR schedule reaches zero on the final step, which makes the last update a no-op

- **Where:** `inky/train.py:40-45` (`remaining = max(0, total_steps - 1 - step)` gives 0 at the last step, so `decay = 0`).
- **What is wrong:** `wsd_lr_scale(999, 1000, 100) == 0.0` and `wsd_lr_scale(998, 1000, 100) == 0.005`. The final optimiser step has learning rate zero, so it changes only the optimiser's running buffers. The final checkpoint and last eval therefore reflect the state after 999 effective steps.
- **Why it matters:** It is a small waste, but a checkpoint is labelled as the end of training and its last step does nothing.
- **Fix:** Use a floor such as 0.1 (or `(total_steps - step) / decay_len`) so the last step has a positive LR, and add a test for the endpoint values.
- **Status:** confirmed (run).

## TRAIN-14 (low): Muon is applied to the router and the halting gate, which are gating vectors

- **Where:** `inky/layers.py:237` (`router`, shape 8x384); `inky/layers.py:418` (`HaltingUnit.gate`, shape 1x384); both are 2D, not embeddings, so `split_parameters` (`optim.py:166`) sends them to Muon.
- **What is wrong:** Orthogonalising a 1-row matrix reduces to normalising the gradient, so the update size no longer follows the gradient size. The halting gate's weight is effectively moved at a fixed rate regardless of how informative the gradient is, and the router's 8 directions are forced to equal magnitude.
- **Why it matters:** Gate and router learning dynamics change in ways that are hard to predict, and the halting head controls compute.
- **Fix:** Exclude `router` and `halting.gate` from Muon by name (add them to `ADAMW_ONLY_NAMES`) and use AdamW for them. Add a test that checks their group.
- **Status:** read (the normalisation argument is analytic; not measured).

## TRAIN-15 (low): distillation loss is not wired into training and is memory-heavy; config notes

- **Where:** `inky/distill.py:43-45` (full `[N, V]` float32 copies of student and teacher logits, no chunking, unlike `_lm_loss` in `modeling.py:122`); `inky/distill.py:57-60`; `inky/train.py:131` (`train()` has no teacher argument).
- **What is wrong:**
  - For B=8, S=1024 and V=50003, one fp32 `[N, V]` tensor is about 1.6 GB, and `s`, `t`, `log_s` and `log_t` coexist, so the peak is several GB.
  - No check that the teacher's vocabulary matches the student's (50003).
  - `train()` has no distillation path, so `distillation_loss` is only callable manually.
  - Config: the tied embedding is 19.2M of 50.6M parameters (`config.py:21`, `vocab_size=50003`), which is 38% of the model. This is a deliberate choice, but it should be recorded because it shapes the budget.
- **Why it matters:** Phase 3 would run out of memory on a real batch, and a vocabulary mismatch would fail late or silently.
- **Fix:** Chunk the distillation over positions the same way `_lm_loss` does, assert `teacher.shape[-1] == student.shape[-1]`, and add a `teacher` option to `train()` when Phase 3 starts.
- **Status:** read (memory estimate is arithmetic; vocabulary and wiring checked by reading).

---

## Items checked that are fine

- **Clip order:** the global norm is computed once over both groups after `backward()` and before both optimiser steps (`optim.py:228-235`). This matches the stated EOS coupling.
- **Muon math:** the Nesterov update, decoupled decay and the Newton-Schulz orthogonalisation (`optim.py:32-54`, `optim.py:115-145`) are implemented as described; the RMS numbers above match the expected `1/sqrt(max)` behaviour.
- **Parameter routing:** the tied embedding and `slot_base`/`step_embedding` go to AdamW (`optim.py:153-171`), and norms and biases get no decay.
- **Loss composition:** the next-token shift in `_lm_loss` (`modeling.py:124-125`) is consistent with `distill.shift_for_next_token`; the z-loss is applied to the softcapped logits and averaged over valid tokens (`modeling.py:115-137`).
- **ACT:** the halting weights sum to 1 per token, and the ponder term is differentiable through the halting head (`modeling.py:213-232`).
- **Config numbers:** the 50.6M parameter count is consistent with the stated sizes, and the default `max_recurrent_loops` (6) and `min_recurrent_loops` (2) match the curriculum defaults.

## Missing pieces for a real run (summary)

- No resume or optimiser-state checkpointing (TRAIN-01).
- No NaN or Inf handling (TRAIN-03).
- No mixed precision, no gradient accumulation, no per-step sync control (TRAIN-12).
- No training entry point in `scripts/` (only `build_untrained.py` exists); the loop is reachable only from Python.
- No logging to a file or experiment tracker; history is in memory only.
- No data reshuffling across epochs and no held-out split inside `train()` (TRAIN-11).
