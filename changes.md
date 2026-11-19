# Inky Architecture & Reasoning Upgrade Summary (`changes.md`)

This document summarizes all architectural upgrades, reasoning/style modes, external memory systems, predictive controllers, continual reinforcement learning pipelines, and audit phases implemented in **Inky**.

---

## 1. Multi-Tier Reasoning Modes (`low`, `high`, `max`) & Improved Looping

Implemented in `inky/config.py`, `inky/layers.py`, `inky/modeling.py`, `inky/generation.py`, and `scripts/sample.py`:

| Reasoning Mode | Recurrent Thinking Loops ($N$) | Halting Prior ($\mu$) | Extended Scratchpad Budget (`REASONING_SCRATCH_BUDGETS`) | Description |
| :--- | :---: | :---: | :---: | :--- |
| **`"low"`** | **`2` loops** | `2.0` | **`384` tokens** | Fast, low-latency latent pass with extended note allowance. |
| **`"high"`** | **`5` loops** | `5.0` | **`768` tokens** | Deep 5-loop iterative refinement with balanced halting distribution. |
| **`"max"`** | **`10` loops** | `10.0` | **`1,536` tokens** | Maximum 10-loop deep deductive thinking + $6\times$ the default 256-token PENCIL scratchpad. Budgets take effect with `reduce_scratch_blocks`. |

### Recurrent Looping Improvements
- **Deep 10-Loop Extrapolation & Progressive Depth Damping (`LoopStepModulation`, `RecurrentCore._mix_update`)**:
  - Clamps step indices safely beyond `max_recurrent_loops` and scales deeper loop updates by $L_{\max} / t$, allowing any checkpoint to run 10 loops in `"max"` mode without index errors or activation variance blowup while preserving exact prefill/cached-decode equivalence.
- **Additive Loop-Step Transition ($u_t + e_t$)**:
  - Supports explicit additive depth embeddings $e_t \in \mathbb{R}^{d_{\text{model}}}$ (`use_additive_step_embedding=True`) before passing memory-enriched state $u_t$ into the weight-tied recurrent core $h_{t+1} = f_{\theta_{\mathcal{C}}}(u_t + e_t, \text{KV\_Cache}_{1:n}^{(t)})$.
- **Per-Token, Per-Channel Thought Refinement Gate (`use_thought_gate=True`)**:
  - Learned sigmoid gate over $[h_{\text{in}} \| h_{\text{new}}]$ (zero-initialized so $2\sigma(0) = 1.0$ at startup) that allows resolved token features to stay locked while unresolved reasoning dimensions keep refining across 5–10 loops.
- **Mode-Aware & Convergence-Aware PonderNet Halting (`convergence_halting=True`)**:
  - Dynamically shifts the halting logit prior to match `"low"` (2), `"high"` (5), or `"max"` (10) loops and modulates halting probability by relative state velocity $\|h_n - h_{n-1}\|_2 / \|h_n\|_2$ so tokens still revising their thoughts continue pondering.

---

## 2. Human-Like Prose Engine & Register Modes (`formal`, `informal`)

Implemented in `inky/style.py`, `inky/layers.py` (`ModeConditioner`), `inky/generation.py`, and `scripts/sample.py`:

- **`"formal"` Mode (`STYLE_PROFILES["formal"]`)**:
  - **Sampling & Cadence**: Composed, precise decoding (`temperature=0.70`, `top_p=0.90`, `top_k=40`, `min_p=0.05`, `repetition_penalty=1.12`, `no_repeat_ngram_size=4`).
  - **Register Controls**: Hard-blocks casual phrases (`INFORMAL_PHRASES`: `"you know"`, `"kind of"`, `"sort of"`, `"no way"`, `"for real"`, `"by the way"`, `"to be honest"`) + AI clichés (`STOCK_PHRASES`), and soft-penalizes colloquialisms/slang (`INFORMAL_WORDS`, 22 entries, e.g. `"gonna"`, `"wanna"`, `"gotta"`, `"kinda"`, `"yeah"`, `"nah"`, `"lol"`, `"omg"`, `"btw"`, `"tbh"`, `"ain't"`, `"stuff"`, `"totally"`, `"literally"`, `"basically"`) + `STOCK_WORDS`.
- **`"informal"` Mode (`STYLE_PROFILES["informal"]`)**:
  - **Sampling & Cadence**: Warm, natural conversational burstiness (`temperature=0.85`, `top_p=0.95`, `top_k=60`, `min_p=0.03`, `repetition_penalty=1.08`, `no_repeat_ngram_size=4`).
  - **Register Controls**: Hard-blocks stiff bureaucratic/academic connectors (`FORMAL_PHRASES`: `"in accordance with"`, `"with respect to"`, `"prior to"`, `"subsequent to"`, `"it is imperative"`, `"one must"`, `"for the purpose of"`, `"in the event that"`) + `STOCK_PHRASES`, and soft-penalizes stilted transitions (`FORMAL_WORDS`, 20 entries, e.g. `"heretofore"`, `"herein"`, `"thereof"`, `"whereby"`, `"notwithstanding"`, `"henceforth"`, `"aforementioned"`, `"consequently"`, `"subsequently"`, `"nevertheless"`, `"furthermore"`, `"moreover"`, `"thus"`, `"hence"`, `"shall"`, `"whilst"`, `"utilize"`, `"commence"`) + `STOCK_WORDS`.
- **Profile application**: `generate(style_mode=...)` now applies the profile's temperature, `top_p`, `top_k`, repetition penalty and n-gram size (explicit non-default arguments win), not just `min_p` and the word lists. Stock-word penalty: 1.5 (formal) / 2.0 (informal).
- **Log-Space `min_p` Truncation & Internal FiLM Conditioning (`ModeConditioner`)**:
  - Filters tail tokens below $\text{min\_p} \cdot p_{\max}$ in log-space (`_candidate_distribution`) and conditions hidden representations via zero-initialized FiLM embeddings (`reasoning_embed` for `low`/`high`/`max`, `style_embed` for `formal`/`informal`).

---

## 3. Hybrid Linear Layers (`1:3` / `1:7` Gated DeltaNet & Selective SSM) + 128k MLA

Implemented in `inky/layers.py` (`GatedDeltaNetMixer`, `SelectiveSSMMixer`, `MultiLatentAttention`) and `inky/config.py`:

- **`GatedDeltaNetMixer` ($\mathcal{O}(1)$ Fast-Weight Matrix State)**:
  - Maintains a constant-size per-head associative matrix state $S_t \in \mathbb{R}^{H \times d_h \times d_h}$ updated via the Gated Delta Rule:
    $$S_t = \alpha_t S_{t-1} + \beta_t \left(v_t - S_{t-1} k_t\right) k_t^\top, \qquad y_t = S_t q_t$$
  - Supports document-boundary state resets via `segment_ids` (prefill only; combining with a cache raises `ValueError`) and $\mathcal{O}(1)$ single-token cached decoding. Prefill uses a chunkwise WY-transform (chunk 64, fp32, exact vs the per-token reference to ~1e-16 in double); the per-token loop remains as the decode path and test reference. State is carried in fp32.
- **Fast $\mathcal{O}(1)$ `SelectiveSSMMixer` Decode & FP32 Carry**:
  - Per-channel diagonal state `[B, hidden, ssm_state_dim]`; chunked scan rewritten without the `[B,C,d,N]` tensor (~6.7x faster on CPU, see `scripts/bench_mixers.py`). Single-token decode step bypasses chunked scan matrix construction when $S = 1$, keeping recurrent carry state $h$ in `float32` across chunks and decode steps.
- **128k Context Hybrid Presets (`InkyConfig.hybrid_linear_128k`)**:
  - Combines Multi-Head Latent Attention (MLA, compressed latent KV cache `LatentKV`) with linear recurrent layers at a **`1:3`** ($3$ linear : $1$ MLA) or **`1:7`** ($7$ linear : $1$ MLA) ratio with `max_position_embeddings = 131072` (128k context).

---

## 4. Ultra-Sparse MoE with Shared Expert & JEPA Router Steering

Implemented in `inky/layers.py` (`SharedExpertMoE`) and `inky/config.py` (`InkyConfig.ultra_sparse_jepa`):

- **Shared + Ultra-Sparse Routed Experts**:
  - Combines always-on `shared_experts` with up to 32+ fine-grained `routed_experts` (top-2 active per token) and DeepSeek-V3 auxiliary-loss-free sign congestion balancing (`balance_update="sign"`).
- **Dynamic Executive Expert Bias ($b_{\text{moe}}$)**:
  - `SharedExpertMoE.forward` accepts `expert_bias` ($b_{\text{moe}} \in \mathbb{R}^{B \times S \times E_{\text{routed}}}$) from the JEPA controller to steer tokens (added to router logits, so it changes routing weights as well as selection) toward specialized logic, code, or language experts based on simulated latent trajectories.

---

## 5. Isolated External Memory Bank ($\mathcal{B}$) with Sub-Linear Product-Key Memory (PKM)

Implemented in `inky/memory_bank.py` (`ProductKeyMemoryBank`, `QuantizedValueTable`):

1. **System Topology & Storage Decoupling**:
   - Decouples the ~50M active Reasoning Core ($\mathcal{C}$) in VRAM from the large-capacity External Memory Bank ($\mathcal{B}$) stored in pinned host CPU RAM or `mmap`-backed storage (`QuantizedValueTable`), supporting **`"int4"`** (packed 2-values-per-byte), **`"int8"`**, and **`"float32"`** quantization.
2. **Sub-Linear $\mathcal{O}(\sqrt{M})$ Cartesian Product-Key Indexing**:
   - Factorizes $M = S^2$ slots into two compact sub-key codebooks $C^{(1)}, C^{(2)} \in \mathbb{R}^{S \times (d_k / 2)}$.
   - Projects normalized hidden state $\tilde{h}_t = \text{RMSNorm}(h_t)$ into $q_t = W_q \tilde{h}_t = [q_t^{(1)} \| q_t^{(2)}]$, computes sub-scores $s^{(1)} = C^{(1)} q_t^{(1)}$ and $s^{(2)} = C^{(2)} q_t^{(2)}$ in $\mathcal{O}(S \cdot d_k)$, and selects exact global Top-$K$ winners $\mathcal{I}^*$ over the $K^2$ Cartesian pairs $\mathcal{I}_1 \times \mathcal{I}_2$.
   - Also supports $\mathcal{O}(1)$ deterministic multi-head polynomial rolling $m$-gram hash routing (`routing_mode="hash"`).
3. **Sparse Value Gather & Attention Aggregation**:
   - Transfers (mmap supported; `pkm_pin_memory` stages gathers through a pinned buffer on CUDA, untested without a GPU) strictly $K \cdot d_{\text{mem}}$ values per token from host RAM/NVMe, dequantizes on-device, and aggregates with scaled softmax weights:
     $$\alpha_j = \frac{\exp(z_j / \sqrt{d_k})}{\sum_{r=1}^K \exp(z_r / \sqrt{d_k})}, \qquad m_t = \sum_{j=1}^K \alpha_j v_{i_j}$$
4. **Gated Residual Injection into the Looped Core**:
   - At designated loop steps $t \in \mathcal{T}_{\text{query}}$, fuses retrieved memory via token-conditioned sigmoid gating with a zero-init learned per-loop-step bias (`pkm_step_gate`), multiplied by JEPA's `fetch_gate` $g_{\text{mem}}$. Defaults: bank `top_k=8`; `InkyConfig` `pkm_top_k=4`, `pkm_query_steps=(1,3,5)`:
     $$\hat{m}_t = W_{\text{out}} m_t, \qquad g_t = \sigma\!\left(W_g [\tilde{h}_t \| \hat{m}_t] + b_g\right), \qquad u_t = h_t + g_t \odot \hat{m}_t$$

---

## 6. Joint Embedding Predictive Architecture (JEPA) Executive Controller

Implemented in `inky/jepa.py` (`JEPAController`, `LatentStateEncoder`, `LatentWorldModelPredictor`, `EnergyCostCritic`, `JEPAControlSignals`):

1. **Latent State & Causal Goal Encoder ($E_\phi$)**:
   - Compresses $h_t \in \mathbb{R}^{d_{\text{model}}}$ into semantic latent state $z_t = \text{RMSNorm}(W_e h_t) \in \mathbb{R}^{d_z}$ ($d_z \le 256$) and derives a strictly causal goal vector $z_{\text{goal}}$ from the running mean of prelude states.
2. **Latent World-Model Predictor ($P_\psi$)**:
   - 2-layer SiLU MLP + GRU gated transition model rolling out candidate action trajectories $\hat{z}_{t+1:t+H}$ ($H=3$ lookahead horizon) in latent space without running the full Transformer block.
3. **Energy / Cost Critic ($C_\omega$)**:
   - Scores predicted trajectories against $z_{\text{goal}}$ with L2 + cosine energy and linear compute cost:
     $$E(\hat{z}, z_{\text{goal}}, t) = \|\hat{z}_{t+H} - z_{\text{goal}}\|_2^2 + \alpha_{\text{cos}}\!\left(1 - \cos(\hat{z}_{t+H}, z_{\text{goal}})\right) + \lambda_{\text{compute}} \cdot t$$
   - Selects optimal action plan $a_t^* = \arg\min_{a_t} E(\hat{z}, z_{\text{goal}}, t)$ with straight-through softmin differentiability during training.
4. **Executive Control Signals & Self-Supervised VICReg Loss**:
   - Emits per-loop **Bank Fetch Gate** (hard 0/1 in eval, straight-through in training, when $0.5(\text{energy}+\sigma(\text{uncertainty}))>\tau_{\text{mem}}$), **MoE Expert Bias** ($b_{\text{moe}}$), and **Adaptive Loop Halting** ($p_{\text{halt}} = \sigma(w_h^\top z_t)$, boosted upon latent convergence $\|\hat{z}_{t+1} - z_t\|_2 < \epsilon$; averaged with the model's halting gate). $\alpha_{\text{cos}}$ (`jepa_alpha_cos`), softmin temperature, VICReg coefficients and gate temperature are configurable; the fetch-gate energy is squashed as $E/(1+E)$ so `uncertainty_head` receives gradient; `jepa_compute_penalty` sets $\lambda_{\text{compute}}$. The JEPA loss pairs each loop's prediction with that loop's output state.
   - Trains via stop-gradient multi-step latent prediction + VICReg variance/covariance anti-collapse regularization (`jepa_loss`), and reports footprint via `memory_footprint_bytes` (`active_ram_bytes` = controller + ~50M core; `within_mobile_ceiling` checks active RAM against 1.3 GB; `within_total_ceiling` checks the total).

---

## 7. Continual User-Preference Reinforcement Learning with GRPO

Implemented in `inky/grpo.py` (`ContinualGRPOTrainer`, `UserPreferenceProfile`, `compute_group_relative_advantages`, `grpo_loss`, `sequence_log_probs`):

- **`UserPreferenceProfile`**:
  - Tracks user likes/dislikes across conversations (preferred vocabulary, bigrams, response length EMA, and `"formal"` vs `"informal"` register affinity) via `record_feedback` and scores candidate completions via `score_response`.
- **Group Relative Policy Optimization (GRPO)**:
  - Samples a group of $G$ candidate responses per prompt, computes group-normalized advantages:
    $$A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G) + \epsilon}$$
  - Optimizes the clipped surrogate GRPO loss with the unbiased non-negative Schulman reverse-KL penalty against reference policy $\pi_{\text{ref}}$:
    $$\mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}}) = \frac{\pi_{\text{ref}}}{\pi_\theta} - \log\frac{\pi_{\text{ref}}}{\pi_\theta} - 1$$
  - Provides `learn_from_conversation(...)` so every conversation turn adapts the model toward the user's preferences.

---

## 8. Audit Phases 1–4 Bug Fixes, Optimizations & Tooling

- **Phase 1 (Integrity & Bug Fixes)**:
  - Fixed `GradScaler` scale backoff on non-finite loss skips (`inky/train.py`), `train_arlo` target tensor device placement (`inky/arlo.py`), document-level train/eval split before sequence packing (`scripts/train.py` via `inky.evaluation.split_heldout`, then `prepare_corpus(..., with_segments=True)`), batched EOS padding & zero-answer scratchpad infinite-loop guard (`inky/generation.py`), `pondernet_kl` zero-weight gradient spike (`inky/layers.py`), and default `qk_norm_mode="latent"` (`inky/config.py`).
- **Phase 2 (Optimizer, Speculative Cache Truncation & Compute Accounting)**:
  - Routed (AdamW; encoder/predictor/bank projections stay on Muon) `a_gate`, `readout`, `lm_head`, `thought_gate`, `reasoning_embed`, `style_embed`, and PKM/JEPA heads to AdamW and per-group AdamW/Muon gradient clipping (`inky/optim.py`; the fp16 GradScaler path in `inky/train.py` still clips globally).
  - Added $\mathcal{O}(1)$ zero-copy `LatentKV.truncate` (returns a trimmed view) and `InkyCache.truncate` for self-speculative decoding rollback (`inky/layers.py`, `inky/cache.py`, `inky/speculative.py`); rollback applies only without highway/SSM mixers.
  - Added `train_flops_per_token` accounting in `scripts/arch_budget.py`.
- **Phase 3 & 4 (Ablation Baselines, Chunked Distillation & Packaging)**:
  - Added `"dense_unrolled"`, `"dense_param_match"`, `"mqa"`, and `"human_reasoning"` variants in `scripts/ablate.py`.
  - Added memory-bounded `chunked_distillation_loss` in `inky/distill.py` (chunks are checkpointed under grad).
  - Added `--gradient-checkpointing` to `scripts/train.py` and `[project.scripts]` CLI entrypoints (`inky-train`, `inky-sample`, `inky-ablate`, `inky-diagnose`, `inky-bench`) in `pyproject.toml`.

---

## 9. Quality Pass (Loops, Style, Mixers, PKM/JEPA, GRPO)

- **Generation/style**: one fp32 logit copy per decode step; `BadWordIndex` (~99x faster bad-word lookup); validated sampling args and `style_overrides`; fused `ModeConditioner` FiLM; configurable `loop_beta_min/max`, `deep_loop_damping`, `style_profile_overrides`.
- **Halting**: fixed NaN gradient in convergence-aware halting when a token's state is unchanged (epsilon inside the sqrt).
- **GRPO**: zero-variance/NaN groups yield zero advantage and skip the update; non-finite steps are skipped (`nonfinite_skips`); `length_norm`, `ref_mode="copy"|"params"|"none"` (always under `no_grad`); `UserPreferenceProfile.decay/save/load`.
- **Training**: fp16 GradScaler path clips per optimizer group (`_clip_per_group`).
- **Tests**: new suites `test_loops_and_style_quality`, `test_mixers_parallel`, `test_pkm_jepa_quality`, `test_grpo_robust`, `test_distill_chunked`, `test_train_scaler_clip`.
- **Known gaps**: MLA and hybrid/MoE presets unchanged; no GPU/Triton benchmarks; with both `num_loops` and a mode set, the halting prior still follows the mode; `ref_mode="params"` does not snapshot buffers (KL differs ~5e-3 from `"copy"`); `generate` profile precedence compares against signature defaults.
