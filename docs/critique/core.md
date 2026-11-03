# Critique: model core

## Status (checked against the code, 2026-11-03)

Checked against the working tree at commit f7299ea plus uncommitted changes. **Line numbers in the
findings below are from an older commit and may not match the current code.** The finding text is
kept as written. Legend: **FIXED** = the fix is in the code (a note says which part, for example opt-in). **OPEN** = the problem remains or is only partly fixed (the note says which part). **ACCEPTED-DESIGN** = a deliberate design choice, not a defect; it is documented and was not changed. **NOT-A-BUG** = checked, and the claim does not hold for the code as designed. **UNVERIFIED** = could not be checked here.

Counts for this file: 8 FIXED, 4 OPEN, 3 ACCEPTED-DESIGN, 1 NOT-A-BUG (of 16).

| ID | Severity | Status | Evidence in current code |
| :--- | :--- | :--- | :--- |
| CORE-01 | high | ACCEPTED-DESIGN | Training forward runs every loop, by design (the halting weights mix all loop states). `forward_early_exit` (inference, batch size 1) skips loop work; its savings on trained weights are unmeasured. |
| CORE-02 | high | ACCEPTED-DESIGN | Output is a halting-weighted mix of loop states, by design. The per-loop step CE (`step_ce_token_fraction`, default 0.25) trains each state, which mitigates the dependence on loop count. |
| CORE-03 | medium | FIXED | The linear loop penalty is gone. PonderNet KL to a geometric prior (mean 3 loops), coefficient 1e-3. |
| CORE-04 | medium | FIXED | `load_counts` accumulates over every call and is reset in `update_balance_bias`. |
| CORE-05 | medium | ACCEPTED-DESIGN | `qk_rope_head_dim = 16` and `rope_theta = 5e5` are unchanged, by design. Not checked at 8k on trained weights. |
| CORE-06 | medium | FIXED | `apply_rope` computes in at least fp32 and returns the input dtype (`layers.py:64-73`). |
| CORE-07 | low | FIXED | `_absorbed_attention` computes the scores in at least fp32 (`layers.py:213-216`). |
| CORE-08 | medium | OPEN | `lm_loss_recompute` defaults to True, so `_lm_loss` still checkpoints each chunk and the head runs twice in backward. Setting it to False removes the second pass but keeps each chunk's logits for backward. The default is unchanged. |
| CORE-09 | low | FIXED | Core aux is divided by the loop count (`core_aux / loops`). |
| CORE-10 | medium | OPEN | Partly fixed. Uncached passes pass `attention_mask` to the highway as `valid_mask`, so its running mean excludes pads. The MoE router z-loss still averages over pad positions (`layers.py:428`). |
| CORE-11 | low | FIXED | The scratch rebuild passes the cache's `num_loops`. Test: `test_scratch_rebuild_keeps_the_loop_count`. |
| CORE-12 | medium | FIXED | Chunked scan (`SelectiveSSMMixer._chunked_scan`). Measured on CPU: 1.8x faster than the token loop at 2048. At 8192 it keeps about 2 GiB of activations per SSM application, so it needs gradient checkpointing (`InkyConfig(gradient_checkpointing=True)`). |
| CORE-13 | low | FIXED | Routed experts are dispatched with one sort and one host read of the counts per MoE call. |
| CORE-14 | low | NOT-A-BUG | Design choice: RMSNorm on `out_proj(mem)`. No ablation has been run, so the choice is unmeasured. |
| CORE-15 | low | OPEN | Depth draw and layer drop now use torch's RNG, so `manual_seed` controls them. No assertion for a train-mode cached forward. |
| CORE-16 | low | OPEN | The n-gram index is incremental (fixed). `decide_choice` is one batched forward, but the prefix is still recomputed per option, with no cache. |

Scope: `inky/layers.py` (all), `inky/modeling.py` (`InkyModel.forward`, `_lm_loss`, `generate`, `_sample`), `inky/config.py` defaults.
No files under `inky/` or `tests/` were edited. Repro scripts lived in the session scratchpad and ran from the project directory with `PYTHONDONTWRITEBYTECODE=1`, so no `__pycache__` was written. The existing tests were not run.

Confirmation key: **Repro** = run on a tiny random-init model (hidden 64, 3 loops) or by arithmetic on the real config. **Read** = from code only.

## Summary

- The ACT halting is a soft mixture, not an early exit. Every token runs every loop, so halting saves no compute at training or inference. The output also depends on the loop budget in a way that is not just "more depth".
- The core is correct where it matters most for causality: fp32 cached decode matches the full forward to 2.4e-7, and the causal running mean in the highway uses the right denominators.
- Main risks for the 50M creative-writing run: wasted loops and diluted depth (CORE-01, CORE-02), a balance signal that only sees the last loop (CORE-04), and a RoPE table too small for 8k (CORE-06).

## Checked and found OK

- **ACT weights sum to 1.** Each step's weight is either `p` or the remaining mass, and the last loop takes `1 - accum`. Halted tokens get 0. Repro: the weights for budget 2 and budget 6 both sum to 1.0 (modeling.py:215-230).
- **Causality with cache.** Repro: fp32 incremental decode vs full forward, max abs diff 2.4e-7. Highway running mean (layers.py:347-351) and the prefix sum are correct. MLA mask for `past_len > 0` (layers.py:132) is correct. Absorbed decode matches prefill in fp32 (3.3e-8).
- **Gradient of `torch.norm` at zero** (velocity in HaltingUnit): repro gives grad 0, no NaN on torch 2.13 CPU.
- **Shifted LM loss** (`_lm_loss`, modeling.py:124-125) and the softcap are correct.

---

## CORE-01: ACT never exits early, so halting costs full compute (high)

- **Where:** modeling.py:195-230 (loop runs for all `loops` steps regardless of `halted`); modeling.py:227 (`h_out` accumulates every step); layers.py:413-424 (HaltingUnit).
- **What is wrong:** `halted` only zeroes the weights (`w = where(halted, 0, w)`). The core still runs for every token on every loop and writes KV for all of them. Inference therefore always pays `max_recurrent_loops` (6) passes, which is 14 layer applications per token. Repro: `num_loops=3` produced exactly 3 core applications, so halting never shortens a forward pass.
- **Why it matters:** The design brief is a 50M model where compute and latency matter. Adaptive depth is supposed to be the saving, and here it is pure reweighting. Training cost is set by the sampled loop count alone (uniform 2 to 6, mean 4), and generation cost is fixed at 6.
- **Fix:** Pick one. (a) Remove ACT and use a fixed loop count, which is simpler and honest about cost. (b) Implement real early exit: halted tokens stop updating `h` and their KV entries for later loops are copied forward from their last computed state. This keeps the cache valid for other tokens' attention. Either way, add a logged metric for mean executed loops per token, not just the mixture weights.
- **Confirmed:** Repro (spy on `m.core.forward`, count = 3 for `num_loops=3`). Design consequence read from code.

## CORE-02: Output is a mixture over loop states, so loop budget changes the function and depth is diluted (high)

- **Where:** modeling.py:222-227 (`w = p`, final `w = 1 - accum`, `h_out` is a weighted sum); modeling.py:179 and config.py:37-43 (budget and threshold).
- **What is wrong:** Each token's output is a convex combination of the loop states, not the last state. With the init halting bias (p ~ 0.37 measured), budget 6 gives weights `[0.37, 0.37, 0.26, 0, 0, 0]`, so about 37% of every token's output is the first loop's state, and loops 4 to 6 get zero weight. With budget 2 the weights are `[0.37, 0.63]`. The same weights from the same `p` mean that changing `num_loops` changes the mixture itself, not just the depth. Training samples the budget uniformly per batch (train.py curriculum, modeling.py:105), so the model has to be invariant to a quantity that changes its output distribution.
- **Why it matters:** Generation uses the default budget of 6, which the model sees only about a fifth of the time in training. Shallow-state leakage also works against the depth that the recurrence is for. For creative writing, the outputs at budget 6 may be bland or mismatched compared with the budget it was trained at most often.
- **Fix:** Use the last loop's state (or a learned final-state readout) and let halting only gate compute, which ties to CORE-01. If mixture ACT is kept, normalise per budget or make the threshold budget-aware, and evaluate perplexity at each budget in `num_loops` in {2..6} to see the spread.
- **Confirmed:** Repro (measured per-loop p at init: 0.368, 0.368, 0.368, 0.367, 0.367, 0.366; weight table from a re-implemented ACT rule fed with those p values). The model's own `mean_loops` at init is 1.897, which matches the weight table.

## CORE-03: Ponder penalty pushes toward early exit, against the depth the recurrence needs (medium)

- **Where:** modeling.py:215 (`ponder` accumulation), modeling.py:247 (`ponder_loss_coef * mean_loops`), config.py:45 (`ponder_loss_coef = 1e-3`).
- **What is wrong:** The penalty counts loops per token as if they cost compute, but under CORE-01 every loop runs anyway. Its only effect is to push `p` up, which moves weight onto early loop states, which is the dilution in CORE-02. It also includes padding tokens in the mean (see CORE-10). At 1e-3 the gradient is small, so the gate probably barely moves from its init. The result is an unexplained knob with a bias in the wrong direction.
- **Why it matters:** The halting gate learns almost nothing useful, and the one term that acts on it is aimed at shallowness.
- **Fix:** Remove the ponder term, or tie it to real executed compute (CORE-01). Log the distribution of `p` and the mean executed loops per validation set so the gate's behaviour is visible.
- **Confirmed:** Read for the direction; measured `mean_loops` = 1.897 at init on the tiny model. Gradient magnitude not measured.

## CORE-04: MoE balance bias is driven by the last loop call only (medium)

- **Where:** layers.py:255-257 (`expert_load` overwritten on every call), layers.py:269-272 (`update_balance_bias` reads it), modeling.py:326-330 (`update_moe_balance`).
- **What is wrong:** The core's MoE module is called once per loop. Each call overwrites `expert_load`, so the balance update sees only the final call of the step, and the core's shared experts are balanced only against that loop's routing. Loop dropout (layers.py:400-402) skips calls and leaves the buffer stale. Gradient accumulation would have the same problem, since only the last micro-batch survives. Eval forwards also overwrite the buffer.
- **Why it matters:** Routing at loop 1 and loop 6 can differ a lot (the states are different). The bias can end up balanced for one loop and collapsed for the others. That is a collapse risk for the shared experts in the core, which are the most-used weights in the model.
- **Fix:** Accumulate counts across all calls in a step (`expert_counts += bincount`), reset in `update_balance_bias`, and only accumulate under `self.training`. This also gives the right normalisation under loop dropout.
- **Confirmed:** Repro. Three calls in one forward gave loads `[0.083, 0.333, 0.333, 0.25]`, then `[0.0, 0.333, 0.333, 0.333]`, then `[0.0, 0.333, 0.25, 0.417]`, so the buffer changes on every call and keeps only the last.

## CORE-05: RoPE on 16 dimensions is too weak for 8k positions (medium)

- **Where:** config.py:37-38 and 55 (`qk_rope_head_dim = 16`, `rope_theta = 5e5`, `max_position_embeddings = 8192`); layers.py:30-41 (RotaryEmbedding); layers.py:93-95 (only the RoPE slice gets position).
- **What is wrong:** Position enters only through the 16-dim RoPE slice, which has 8 frequency pairs. With base 5e5, the lowest frequencies barely rotate over the context. Computed over 8191 positions: pair 0 rotates 8191 rad, pair 3 rotates 60 rad, pair 5 rotates 2.25 rad, pair 6 rotates 0.44 rad, pair 7 rotates 0.08 rad. Five of the eight pairs carry little or no usable position signal at long range. The 48-dim nope path is position-free by design.
- **Why it matters:** Long-range coherence across several thousand tokens of a story (names, callbacks, tone) relies on the model knowing relative distance. This setup gives a weak, mostly aliased position signal at 8k.
- **Fix:** Raise `qk_rope_head_dim` to 32 or 64 (trading some nope width), or lower `rope_theta` to about 1e4 for this slice. Check with a simple long-range probe (a key token at position 100 vs 4000) before and after.
- **Confirmed:** Repro. The inv_freq and angle values come from printing the actual formula with the config values. Effect on model quality not measured.

## CORE-06: RoPE tables stay fp32 and mix dtypes with bf16 activations (medium)

- **Where:** layers.py:37-38 (cos/sin buffers are fp32), layers.py:49-51 (`apply_rope` promotes), layers.py:99 and 127-128 (k_pe and the concatenated q/k inherit fp32), layers.py:102-104 (cache stores fp32 k_pe).
- **What is wrong:** `apply_rope` with bf16 input and fp32 tables returns fp32 (repro). Two results: (a) with bf16 weights and fp32 tables, prefill crashes in SDPA with "Expected query, key, and value to have the same dtype" (repro); (b) under autocast the cache stores fp32 `c_kv` and `k_pe` (repro: both fp32), doubling cache memory and moving attention to fp32.
- **Why it matters:** `from_pretrained(..., torch_dtype=bf16)` casts buffers too, so the crash probably does not show up on the normal path. But any partial cast (e.g. only Linear layers in bf16 for a quantized or mixed run) fails in a non-obvious way, and the autocast path gives different cache dtypes from the pure-bf16 path.
- **Fix:** Cast cos/sin to `x.dtype` inside `apply_rope`, or hold the tables in fp32 and cast at use. Add a test that runs prefill and decode under bf16 weights with fp32 tables.
- **Confirmed:** Repro (`repro_dtype`): crash message and cache dtypes as above.

## CORE-07: Absorbed decode computes attention scores in the activation dtype (low)

- **Where:** layers.py:151-156 (`q_c`, `scores` as matmuls in activation dtype), layers.py:156 (softmax cast to float only after the matmul), layers.py:157 (probs cast back to bf16).
- **What is wrong:** The dot products with `c_kv` and `k_pe` are accumulated in bf16 and only then converted for softmax. The prefill path uses SDPA, which accumulates in fp32. So decode and prefill disagree by rounding, and the gap grows with the magnitude of the scores. Long contexts (8k) and large trained logits make this worse than the test below.
- **Why it matters:** Generation is always decode. A consistent small error at every step of an 8k-token story compounds into a different distribution from the one the model was trained on.
- **Fix:** Compute `q_c`, `scores` and `out_c` in fp32 (cast `c_kv`, `k_pe`, `q_*` to float), keep only the final output in the activation dtype.
- **Confirmed:** Repro at random init: bf16 decode vs prefill max abs diff 1.8e-4 against an output scale of 0.04 (about 0.5%). Fp32 gives 3.3e-8. The size on a trained model is not measured.

## CORE-08: Training cost: LM head is about 40% of per-token FLOPs and runs twice (medium)

- **Where:** modeling.py:108-113 (`_project`), modeling.py:115-137 (`_chunk_ce` with checkpoint per 1024-token chunk), config.py:21 (vocab 50,003), hidden 384.
- **What is wrong:** The tied head is 384 x 50,003, about 19.2M parameters, which is roughly 40% of the model's parameters. Per token that is about 19M multiply-adds against roughly 29M for the active body: a layer application is about 2.9M active params and the average is about 10 applications (mean 4 loops). The checkpoint in `_lm_loss` recomputes the head in backward, so the head runs twice per step.
- **Why it matters:** This is the biggest single line of training compute, and it is spent on a head that is mostly idle for creative prose. The 8k sequence adds per-token head cost linearly.
- **Fix:** Keep the chunking for memory, but drop the recompute if memory allows (store bf16 logits per chunk, or raise the chunk size and skip checkpoint). Consider a smaller vocabulary or a factorised head. Measure the step time split before choosing.
- **Confirmed:** Read plus parameter arithmetic from config values. Step time not measured.

## CORE-09: Auxiliary MoE losses are summed per layer application, so their weight depends on the sampled loop count (low)

- **Where:** modeling.py:211 and 237 (`moe_aux` summed over every layer application), layers.py:266 (router z-loss per call), layers.py:288-293 (`aux` from each block), modeling.py:94-106 (random loop count), config.py:51 (`router_z_loss_coef = 1e-3`).
- **What is wrong:** The z-loss is computed on each MoE call, so there are about 1 + 2L + 1 calls for loop count L. The total aux weight therefore grows with the sampled loop count and with the curriculum (train.py `loop_curriculum`). The number is not normalised by the number of applications.
- **Why it matters:** The effective regularisation strength changes from batch to batch and over the schedule. Loss curves are not comparable across runs with different curricula.
- **Fix:** Average the z-loss over calls (or over loops) rather than summing, and log `moe_aux` separately so its share of the loss is visible.
- **Confirmed:** Read.

## CORE-10: No attention mask or padding support; padding leaks into the ponder mean and the highway (medium)

- **Where:** modeling.py:141-150 (`forward` has no `attention_mask`); modeling.py:243 (`ponder.mean()` over all positions); layers.py:347-351 (highway running mean over all positions, including pads); modeling.py:378 (EOS stops only when all rows finished).
- **What is wrong:** There is no mask input anywhere, so left-padded batches attend to pad tokens and the causal running mean in the highway includes them. Right padding is safe for causal attention, but the ponder mean still includes pads. `generate` keeps decoding finished rows (no per-row stop) and cannot batch prompts of different lengths safely.
- **Why it matters:** Batched evaluation and batched generation are common ways to get throughput. Silent contamination from pads is hard to notice.
- **Fix:** Add an `attention_mask` (or require right padding and assert it), mask pads out of the highway cumsum and the ponder/loss means, and track per-row EOS in `generate` and pad finished rows explicitly.
- **Confirmed:** Read (no mask parameter in the signature; ponder and cumsum over all positions).

## CORE-11: Scratch reduction silently changes the loop count (low)

- **Where:** modeling.py:380-389 (reduction rebuilds the cache), specifically modeling.py:388 (`self(seq, past=cache, last_only=True)` drops `num_loops`).
- **What is wrong:** When the caller passed `num_loops`, the first call uses it but the rebuild after scratch reduction falls back to the default (max loops). The depth changes mid-generation. The rebuild also re-prefills the whole remaining sequence in one pass, which at 8k is a full quadratic attention pass at each reduction.
- **Why it matters:** The model's text after a reduction is computed at a different depth from the text before it, which is an unannounced behaviour change for anyone who tuned `num_loops`.
- **Fix:** Pass `num_loops=loops_used` on the rebuild (store the cache's `num_loops`, or pass the argument through).
- **Confirmed:** Read.

## CORE-12: Selective SSM mixer is a Python loop over positions with large fp32 saved state (medium; dormant by default)

- **Where:** layers.py:198-203 (per-token Python loop; the state `h` is [B, d, N] per step), layers.py:186-206; config.py:70 (`ssm_mixer_layers` default empty).
- **What is wrong:** The recurrence runs in a Python loop of length S for every application. Autograd keeps the state for each step. At d=384, N=64 in fp32 that is about 98 KB per token per application per batch element, so about 0.8 GB per 8k sequence per SSM application, and about 5 GB at 6 loops. In a bf16 model, the recurrent state also accumulates rounding over up to 8k steps.
- **Why it matters:** Enabling this mixer at 8k would blow up training memory and make steps very slow (8k iterations x loops). It is only dormant because the default is off.
- **Fix:** Use a chunked parallel scan (the recurrence is linear and diagonal, so chunks of 64 with a cumulative-decay trick work), keep the state in fp32, and add a test at S = 2048 for memory and speed before enabling.
- **Confirmed:** Read plus memory estimate from shapes. Not run at 8k.

## CORE-13: MoE dispatch forces host syncs in every expert and every call (low)

- **Where:** layers.py:259-264 (`torch.where(idx == e)` and `numel() == 0` per expert), layers.py:256 (`bincount`).
- **What is wrong:** Each routed expert does a data-dependent `nonzero` and a Python branch on its size. On CUDA that is a host sync per expert per call, roughly 8 per MoE call times about 14 calls per forward, so about 110 syncs per forward.
- **Why it matters:** At this model size the GPU is mostly idle waiting on the host, so step time is dominated by synchronisation rather than FLOPs.
- **Fix:** Use a sorted, grouped dispatch (one sort by expert, then `split` by counts once, with the counts moved to host in one transfer), or a padded fixed-capacity dispatch with no per-expert branching.
- **Confirmed:** Read. Sync count not profiled.

## CORE-14: Highway RMSNorm rescales a slowly varying causal mean to fixed magnitude (low)

- **Where:** layers.py:347-361 (running mean `h_mean` over all past positions, `mem`, and `out = h + 0.1 * self.norm(out_proj(mem))`).
- **What is wrong:** The highway input is a causal running mean, which changes very slowly after a few hundred positions (it is diluted by 1/t). The RMSNorm at the end rescales whatever small change is left to unit RMS. So after the first few hundred tokens, the highway adds an almost position-independent vector of fixed magnitude to every token, with little token-specific information.
- **Why it matters:** The "memory" the highway is supposed to carry for 8k contexts carries little information beyond a global average. It may be doing little harm, but it is not doing what its name suggests.
- **Fix:** Use a local or exponentially decayed mean (a fixed-rate EMA) instead of the global mean, or remove the RMSNorm so the magnitude of the change survives. Ablate the highway (`0.0 *` on the output) on a held-out long-context set to see whether it helps at all.
- **Confirmed:** Read. Effect on quality not measured.

## CORE-15: Loop dropout and random loop counts are not seed-controlled and break cache consistency in train mode (low)

- **Where:** layers.py:400-402 (CPU `torch.rand` per layer per loop, drops whole layers, presents set to None); modeling.py:105 (`random.randint` on Python's RNG).
- **What is wrong:** Loop depth uses Python's `random` while the layer dropout uses torch's CPU RNG, so runs are not reproducible from `torch.manual_seed` alone. A forward in train mode with a cache returns `None` presents for dropped layers, and a later decode call then computes attention without those layers' history, which is a silent mismatch.
- **Why it matters:** Reproducibility of ablations, and any train-mode cached evaluation (e.g. a quick sample during training) gives outputs that do not match eval mode.
- **Fix:** Use one seeded `torch.Generator` for both depth and layer dropout, and assert `not self.training` when `past` is provided, or disable loop dropout when a cache is in use.
- **Confirmed:** Read.

## CORE-16: Decision scoring recomputes the full prefix per option, and n-gram blocking is O(N) Python per step (low)

- **Where:** modeling.py:281-285 (`decide_choice` runs one full forward per option, despite the comment saying context states are shared); style.py:65-76 and modeling.py:405-412 (`banned_by_no_repeat_ngram` rescans the whole sequence in Python each generated token).
- **What is wrong:** Each option costs a full forward over the context, so decision scoring is O(options x context). The n-gram blocker is O(sequence) Python work per token, which adds up over an 8k generation.
- **Why it matters:** Both are avoidable overheads at generation time. The n-gram cost grows with length and was not measured.
- **Fix:** Score options from one cached context (`past=` the prefix cache, then one-step-per-option or batched options). Maintain the n-gram index incrementally in `generate` (a dict from (n-1)-gram to next tokens, updated per token).
- **Confirmed:** Read.

---

## Test coverage gaps

Existing tests cover decode-vs-full consistency (fp32), ACT mean-loop bounds, and balance-bias movement. Nothing covers: ACT weights summing to 1 directly, the per-call `expert_load` behaviour (CORE-04), bf16 prefill vs decode (CORE-07), dtype with partial casts (CORE-06), scratch reduction keeping `num_loops` (CORE-11), padding (CORE-10), or the SSM mixer at length (CORE-12). The repro scripts for CORE-01, 02, 04, 06 and 07 are small enough to become tests.

## Memory at 8k context (estimate)

- **Decode cache:** 14 cached layer applications (pre 1, core 2 x 6 loops, post 1), each storing `c_kv` (128) and `k_pe` (16) per token. About 33 MB per 8k sequence in bf16, or about 66 MB if the cache is fp32 under autocast (CORE-06). Small.
- **Training activations:** Dominated by 14 layer applications per token with no gradient checkpointing by default (`gradient_checkpointing=False`). Not measured. Turning it on is probably the cheapest way to fit 8k batches, at about one extra forward per block.
