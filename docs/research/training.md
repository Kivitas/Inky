# Training research for Inky: Muon weight decay, stability guards, exact resume

Date of research: 2026-11-02. Scope: current practice (2024 to 2026) for Inky's hybrid Muon + AdamW optimiser.

Verification key: **[V]** = claim confirmed on the fetched page. **[S]** = search snippet only, not opened. **[I]** = my inference or recommendation, not stated by a source.

---

## 1. Weight decay for Muon

### 1.1 PyTorch `torch.optim.Muon` documentation (2.15 page)
- Link: https://docs.pytorch.org/docs/2.15/generated/torch.optim.Muon.html
- Year: not shown on the page (the class is in current PyTorch stable docs) [unverified year].
- Confirmed [V]: signature `Muon(params, lr=0.001, weight_decay=0.1, momentum=0.95, nesterov=True, ns_coefficients=(3.4445, -4.775, 2.0315), eps=1e-07, ns_steps=5, adjust_lr_fn=None)`. Weight decay default is 0.1, and the algorithm is described as using decoupled weight decay.
- Confirmed [V]: "Muon is an optimizer for 2D parameters of neural network hidden layers." Biases and embeddings should be optimised by AdamW. The page does not name the output head.
- Confirmed [V]: `adjust_lr_fn` accepts `"original"`, `"match_rms_adamw"`, `"spectral_unclamped"`; default `None` falls back to `"original"`. `"match_rms_adamw"` scales the update by 0.2 * sqrt(max(A, B)) so that AdamW learning rate and weight decay can be reused. Scaling detail from the Moonlight paper below and the docs snippet [S].
- Confidence: high for the defaults and the parameter split.

### 1.2 Moonlight: "Muon is Scalable for LLM Training" (Liu et al., 2025)
- Link: https://arxiv.org/abs/2502.16982 (HTML read: https://arxiv.org/html/2502.16982v1)
- Year: 2025.
- Confirmed [V]: update rule W_t = W_{t-1} - eta_t (0.2 * O_t * sqrt(max(A, B)) + lambda * W_{t-1}). Weight decay is added outside the orthogonalised update, which is decoupled decay. The paper cites Loshchilov and Hutter (2019) for this.
- Confirmed [V]: the paper says the original Muon was missing weight decay and that adding it, together with a per-parameter update scale, are the two techniques needed to scale Muon. The numeric lambda for Muon is not stated in the text [not stated].
- Confirmed [V]: Moonlight pretraining uses weight decay 0.1 for all stages (the Moonlight-A AdamW baseline is said to follow the same settings; applying 0.1 to AdamW there is an inference).
- Confirmed [V]: AdamW handles RMSNorm, LM head and embedding parameters; Muon handles matrix parameters. RMSNorm gamma receives weight decay, which the paper says is "crucial for ensuring training stability." Biases are not discussed.
- Confirmed [V]: at 800M parameters and 100B tokens, vanilla Muon (no decay) converged faster early but "some model weights grew too large over time." Adding decay gave lower validation loss than both vanilla Muon and AdamW in that over-trained regime. The lambda used there is not stated.
- Confirmed [V]: Muon beats AdamW at small scale, but the gain diminishes as models scale.
- Confidence: high that decay is needed for long or over-trained runs; medium that 0.1 is the right value for Inky.

### 1.3 Other reported Muon weight-decay values (snippets only)
- Muown: row-norm control for Muon, 2026, https://arxiv.org/pdf/2605.10797. Reports lambda = 0.1 as optimal over {0.01, 0.03, 0.1, 0.3} [S]. Not opened.
- NVIDIA Emerging Optimizers, https://docs.nvidia.com/nemo/emerging-optimizers/0.2.0/apidocs/orthogonalized-optimizers.html. Default weight_decay = 0.01 [S]. Not opened.
- Keras Muon, https://keras3.posit.co/reference/optimizer_muon.html. Default weight_decay = 0.1 [S]. Not opened.
- ICML 2025 workshop paper on Muon orthogonalisation, https://icml.cc/virtual/2025/47855. Reports that 0.1 underperforms during the first 80% of training but outperforms lower values afterward, and proposes clipping and scheduling decay [S]. Not opened. Treat as unverified.

### 1.4 Is decay needed at small scale?
- No source fully settles this. Moonlight's evidence (section 1.2) shows the need appears once runs are long or over-trained, not from the very small runs in its ablations [V for the 800M run only].
- Confidence: low for small Inky-scale runs. Recommendation: use 0.1 by default, and run a short sweep over {0.01, 0.1} on the target model [I].

---

## 2. Training stability guards

### 2.1 ZClip: adaptive spike mitigation (2025)
- Link: https://arxiv.org/abs/2504.02507 (code: https://github.com/paperwave/ZClip)
- Year: 2025 (April).
- Confirmed [V] from abstract: z-score anomaly detection on gradient norms to clip large spikes; threshold adapts to gradient-norm statistics; aims to prevent loss spikes without changing normal training.
- Confirmed [V] from README: defaults `z_thresh = 2.5`, EMA `alpha = 0.97`, `warmup_steps = 25`, `clip_factor = 1.0`, `max_grad_norm = 1.0` (optional cap). Call order: `zclip.step(model)` after `loss.backward()` and before `optimizer.step()`.
- Not stated [V]: the exact clipping formula in default mode, and whether any step is skipped.
- Confidence: medium. Use it as an optional add-on, not the baseline.

### 2.2 AdaGC: adaptive per-tensor gradient clipping (2025; ICML 2026 listing)
- Link: https://arxiv.org/html/2502.11034v2 (abstract: https://arxiv.org/abs/2502.11034)
- Year: 2025 (arXiv); listed at ICML 2026 [V from search listing].
- Confirmed [V]: per-tensor scale h = min(lambda_rel * gamma_{t-1,i} / ||g_{t,i}||, 1.0), with gamma updated as an EMA of the clipped norm: gamma_t = beta * gamma_{t-1} + (1 - beta) * ||clipped g||.
- Confirmed [V] defaults (Table 9): `lambda_rel = 1.04`, `beta = 0.99`, `T_start = 100` (warm-up with global clipping). Grid search on Llama-2 7B chose these values.
- Confirmed [V] baseline: GlobalGC with `lambda_abs = 1.0` (Table 10).
- Confirmed [V] results: Llama-2 7B zero-shot average 49.69% (GlobalGC) vs 51.01% (AdaGC); Mixtral 8x1B 47.74% vs 49.01%. Spike score drops to 0 with AdaGC in those runs. The text's +2.48% ERNIE claim does not match its table (34.38 vs 34.73) [V].
- Not stated [V]: any NaN or skip handling. AdaGC scales gradients; it does not skip steps.
- Confidence: medium-high for the defaults and the global clip value of 1.0; medium for the benefit size, since the results come from the authors' own experiments.

### 2.3 Skipping on loss spikes
- Source: LING MoE report, "Every FLOP Counts," https://arxiv.org/pdf/2503.05139 (2025). Abstract [V]: refinement of "training anomaly handling" is one of its contributions.
- Snippet [S]: when a spike is detected, the update is skipped and the data is re-injected into later batches; the learning rate is reduced if the spike persists. Spike threshold not stated. Full text not opened.
- Source: K2-V2 report, https://arxiv.org/pdf/2512.06201 (2025/2026). Snippet [S]: distinguishes narrow spikes, which training continues through, from wide spike clusters that triggered a rollback. Not opened.
- Confidence: low on exact thresholds; medium on the general pattern (skip plus data re-injection, rollback for clusters).

### 2.4 Skipping on NaN or Inf loss or gradients
- Practitioner guide (apxml, "diagnosing loss spikes") [S]: check loss and gradients for NaN or Inf before the optimiser step and skip the update if they appear. Not opened.
- PyTorch forum thread on skipping NaN-loss steps under DDP, https://discuss.pytorch.org/t/how-can-we-skip-a-step-with-nan-loss-in-the-training-step-when-using-distributed-data-parallel-ddp/177999. Verified [V]: no accepted answer; the suggestions are to find the source of the NaN and to use `nan_to_num`. It does not cover DDP coordination.
- Confidence: high that a NaN check before `optimizer.step()` is standard practice; the DDP coordination rule below is my recommendation, not a sourced fact [I].

---

## 3. Checkpointing for exact resume

### 3.1 Resuming from checkpoints (apxml course, 2025)
- Link: https://apxml.com/courses/how-to-build-a-large-language-model/chapter-19-checkpointing-fault-tolerance/resuming-training-from-checkpoints
- Year: not shown on page [unverified].
- Confirmed [V]: restoring the "entire training state" needs model weights, optimiser state (moment estimates), scheduler state, and the iteration counter (`start_iter = iteration + 1`).
- Confirmed [V] but hedged on the page: data-loader progress ("potentially" needed; re-create the loader for the epoch and skip batches; call `sampler.set_epoch(epoch)` for `DistributedSampler`) and RNG state. Torch RNG is restored in code; Python and NumPy RNG appear only as commented-out code.
- Not covered [V]: GradScaler (AMP) state, CUDA RNG state, sampler internal state.
- Confidence: high for model, optimiser, scheduler and step; medium for the rest.

### 3.2 RLinf checkpoint resume guide (third-party framework docs)
- Link: https://rlinf.readthedocs.io/en/latest/rst_source/guides/resume.html
- Year: not shown on page [unverified].
- Confirmed [V]: Megatron backend saves `distrib_optim.pt` (optimiser), `model_optim_rng.pt` (model shards, optimiser, RNG) and `data/data.pt` (dataloader state, so no samples are skipped or repeated). FSDP/DCP saves optimiser, scheduler and RNG in `dcp_checkpoint`; the dataloader state is restored, but no separate file name is given.
- Confirmed [V]: the page requires the same model, FSDP version, world size, sharding and rank-to-shard mapping for local-shard resume.
- Confirmed [V]: the page's only check is that the next logged step after resume is the expected one. Note that its own example is internally inconsistent (resumes from step 10, checks for step 30).
- Confidence: medium. It is a framework guide, not a primary paper.

### 3.3 Megatron Bridge checkpointing docs (snippet only)
- Link: https://docs.nvidia.com/nemo/megatron-bridge/0.6.0/training/checkpointing.html [S]. States that loading only a pretrained checkpoint does not restore optimiser, RNG, dataloader or scheduler state, and that RNG save and load default to True. Not opened.

### 3.4 Minimum state for exact resume (synthesis)
Sources above agree on the core items. Items marked [I] are my additions.
1. Model parameters (all shards).
2. Optimiser state for both groups: AdamW moments and step counts; Muon momentum buffers [I, as Muon is stateful].
3. LR scheduler state and the global step, plus tokens consumed.
4. Data position: sampler or loader state, epoch, and shuffle seed. Sources say this is needed; the exact fields are [I].
5. RNG state: Python, NumPy, torch CPU, and CUDA (sources cover torch and partly Python/NumPy; CUDA is [I]).
6. GradScaler state if mixed precision uses fp16 [I].
7. Guard state: skipped-step counter, consecutive-skip counter, and EMA statistics used by ZClip or AdaGC [I].
8. A config and world-size fingerprint, because the sources require the same world size and sharding [V for the requirement].

---

## 4. Recommended defaults for Inky (hybrid Muon + AdamW)

Confidence is given per line. These are recommendations; they are not verified end-to-end.

**Parameter groups**
- Muon: 2D hidden-layer matrices only. Confidence high (PyTorch docs [V]).
- AdamW: embeddings, LM head, RMSNorm gamma, biases [V for embeddings, head and norms per Moonlight; biases [V] as "AdamW" per PyTorch docs].

**Weight decay (decoupled)**
- Muon group: 0.1 (PyTorch default [V]; Moonlight uses 0.1 [V for pretraining]). Sweep {0.01, 0.1} for short runs. Confidence medium.
- AdamW matrix-like group (embeddings, LM head): 0.1 [I], following the Moonlight setting. Confidence low-medium.
- RMSNorm gamma: apply decay as Moonlight does, 0.1 [V for the statement; value follows the setting]. Confidence medium.
- Biases: 0 [I]. No source in this research tests decay on biases. Confidence medium.

**Muon update scale**
- Use `adjust_lr_fn="match_rms_adamw"` if Inky wants to reuse AdamW learning rate and decay values; otherwise keep `"original"` [V for options; choice is [I]]. Confidence medium.

**Clipping**
- Baseline: global gradient-norm clip at 1.0 across all parameters [V: AdaGC's GlobalGC baseline uses lambda_abs = 1.0]. Confidence medium-high.
- Optional, only if spikes persist: ZClip with z_thresh 2.5, alpha 0.97, warmup 25 [V from README] or AdaGC with lambda_rel 1.04, beta 0.99, T_start 100 [V]. Confidence medium.

**NaN and Inf handling**
- Before `optimizer.step()`, check that the loss and the total gradient norm are finite. If not, zero the gradients, skip the optimiser step, and do not advance the scheduler [S for the check; scheduler detail is [I]]. Confidence high for the check.
- In DDP, make the skip decision a collective: all-reduce the non-finite flag so every rank skips together [I]. Confidence medium.
- Count consecutive skips. Halt, or roll back to the last checkpoint, after N consecutive skips; N = 10 is a placeholder [I]. Confidence low.
- Log every skip with step, loss and grad norm [I].

**Loss spikes (not NaN)**
- Do not skip by default. Log and continue. If a spike is isolated, continue; if spikes cluster, roll back to the last good checkpoint [S: K2-V2 pattern]. Optional: skip and re-inject data as in the LING report [S]. Confidence medium-low.

**Checkpoint contents**
- Everything in section 3.4. Save atomically (write to a temp file, then rename) and keep the last few checkpoints [I].
- Acceptance test [I]: resume from a checkpoint and check that the next N losses match an uninterrupted run exactly in deterministic mode, instead of only checking the step number as RLinf does [V for RLinf's weaker check].

---

## Sources

- PyTorch `torch.optim.Muon` docs, 2.15: https://docs.pytorch.org/docs/2.15/generated/torch.optim.Muon.html
- Moonlight, "Muon is Scalable for LLM Training," 2025: https://arxiv.org/abs/2502.16982 and https://arxiv.org/html/2502.16982v1
- Muown, row-norm control for Muon, 2026 [S]: https://arxiv.org/pdf/2605.10797
- NVIDIA Emerging Optimizers, orthogonalized optimizers [S]: https://docs.nvidia.com/nemo/emerging-optimizers/0.2.0/apidocs/orthogonalized-optimizers.html
- Keras Muon [S]: https://keras3.posit.co/reference/optimizer_muon.html
- ICML 2025 Muon orthogonalisation workshop paper [S]: https://icml.cc/virtual/2025/47855
- ZClip, 2025: https://arxiv.org/abs/2504.02507 and https://github.com/paperwave/ZClip
- AdaGC, 2025 (ICML 2026 listing): https://arxiv.org/html/2502.11034v2 and https://icml.cc/virtual/2026/poster/61050
- SPAM, 2025 [S]: https://arxiv.org/abs/2501.06842v1
- LING MoE report, 2025: https://arxiv.org/pdf/2503.05139
- K2-V2 report, 2025/2026 [S]: https://arxiv.org/pdf/2512.06201
- apxml, diagnosing loss spikes [S]: https://apxml.com/courses/how-to-build-a-large-language-model/chapter-24-identifying-mitigating-training-instabilities/diagnosing-loss-spikes
- PyTorch forum, skipping NaN-loss steps under DDP: https://discuss.pytorch.org/t/how-can-we-skip-a-step-with-nan-loss-in-the-training-step-when-using-distributed-data-parallel-ddp/177999
- apxml, resuming training from checkpoints: https://apxml.com/courses/how-to-build-a-large-language-model/chapter-19-checkpointing-fault-tolerance/resuming-training-from-checkpoints
- RLinf, checkpoint resume: https://rlinf.readthedocs.io/en/latest/rst_source/guides/resume.html
- NVIDIA Megatron Bridge, checkpointing [S]: https://docs.nvidia.com/nemo/megatron-bridge/0.6.0/training/checkpointing.html
- Loshchilov and Hutter, decoupled weight decay (2019), cited by Moonlight; arXiv ID from memory, not opened: https://arxiv.org/abs/1711.05101
