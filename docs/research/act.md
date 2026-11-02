# Adaptive computation in looped / recurrent-depth LMs: research notes for Inky

Date of research: 2026-11-02. Scope: 2016 to 2026 work on (1) per-token early exit that saves real compute, including KV cache handling; (2) budget-invariance of outputs; (3) PonderNet- and ACT-style halting with a sensible prior. Also ACT's known failure modes.

**Verification key.** "Verified" means the claim was read on the fetched page (abstract, HTML, or PDF text as noted). "Snippet" means it came only from search-result text and was not opened. "Unverified" means it is from memory or a secondary summary and was not confirmed on a page.

Note: the arXiv abstract pages do not contain full text. Where a claim needed the body, the HTML version was fetched. The ACT PDF could not be read through the fetch tool, so ACT mechanics are unverified.

---

## Findings

### F1. AdaPonderLM: self-supervised token-wise early exit with KV reuse
- **Link:** [arXiv 2603.01914 (abstract)](https://arxiv.org/abs/2603.01914), [HTML v2](https://arxiv.org/html/2603.01914v2)
- **Year:** 2026 (March)
- **Mechanism:** Each recurrent iteration i has its own two-layer MLP gate. A sigmoid gives a keep probability s_i; a token is pruned when s_i < tau = 1e-4. A persistent binary mask makes pruning monotonic. Training is two-stage: CE loss only for 20k steps, then CE + lambda * L_ponder, where L_ponder is the mean of the smallest k-fraction of gate values (k = 0.1, lambda = 0.1 by default). The paper says a single MLP shared across iterations was unstable, so each iteration gets its own. KV reuse: for pruned tokens, keys and values are overwritten with the previous iteration's values wherever the mask is 0. (Verified.)
- **Halting prior:** No explicit prior on the halting distribution. The sparsity term is the only regularizer. (Verified.)
- **Real compute saved?** Not as described. The KV procedure computes new K/V for all tokens and then overwrites the pruned ones, and the paper reports no wall-clock speedup. My reading of the text: the reported savings are FLOP-accounting savings, not skipped kernels. Unverified on code, since no code was checked.
- **Numbers (verified from Table 1, Pile, 26B tokens):** 70M: AdaPonderLM 3.8x inference FLOPs, val PPL 14.32, vs PonderLM 4x / 14.40, Pause Token 4x / 16.27, Loop Transformer 4x / 14.98. 410M: AdaPonderLM 3.8x / PPL 9.87, PonderLM 4x / 9.72. The abstract's "about 10%" compute reduction does not match the table, which shows about 5% (3.8x vs 4x). The text is internally inconsistent on this figure.
- **Budget-invariance:** Not addressed. Output depends on the gate outcome, which depends on the training-time sparsity target.
- **Code complexity:** Medium. Per-iteration gate MLPs, a monotonic mask, a two-stage training schedule with a warm-up on k, and KV overwrite logic.
- **Fit at 50M with KV cache:** Plausible. The smallest tested model is 70M, and the gates add "only slightly" to parameters (no exact figure given). The KV reuse design is cache-compatible.

### F2. Mixture-of-Recursions (MoR): per-token recursion depth via routers
- **Link:** [arXiv 2507.10524 (abstract)](https://arxiv.org/abs/2507.10524), [HTML](https://arxiv.org/html/2507.10524). Secondary listing: NeurIPS 2025 per search snippet (unverified).
- **Year:** 2025 (July arXiv)
- **Mechanism (verified):** Two router designs. Expert-choice: each recursion depth picks its top-k tokens via a score g = G(theta^T H), with sigmoid and a linear router performing best in the paper's ablation. Capacity shrinks with depth (for N_r = 3, 3/3, 2/3, 1/3 of tokens). Token-choice: a router assigns each token one depth once, top-1 over softmax or sigmoid, with an MLP router performing best. A token can only be re-evaluated at depth r+1 if it was selected at r (hierarchical filtering).
- **KV cache (verified):** Recursion-wise caching stores KV only for tokens routed to a given depth, and attention is restricted to those cached tokens. Recursive KV sharing computes KV once at the first step and reuses it at all deeper steps. The paper says sharing "leads to quite large performance degradation in expert-choice routing under a fixed token setting," but can help token-choice routing.
- **Real compute saved (verified):** Yes, in throughput terms. The paper reports up to about 2.06x vanilla throughput at max batch size for 360M models (mean 256 generated tokens, quality trade-off noted). Training: about 25% fewer training FLOPs and 25% lower peak memory at N_r = 2, with slight quality differences.
- **Quality (verified):** At equal 16.5e18 FLOPs, expert-choice MoR with N_r = 2 reaches 43.1% average few-shot accuracy vs 42.3% for vanilla, with about half the parameters.
- **Budget-invariance:** Not addressed as a goal. Under expert-choice, top-k capacity fixes the compute budget, and the output depends on which tokens are routed. The paper says adjusting top-k after training is hard, because router scores separate almost perfectly with the auxiliary loss. This is a stated limitation (verified).
- **Code complexity:** Medium-high. Variable token subsets per depth need gather/scatter, per-depth KV bookkeeping, and capacity logic. A reference repository is linked from the arXiv listing (not inspected).
- **Fit at 50M with KV cache:** Possible but tight. The smallest tested model is 135M (verified), and the router overhead at 50M has not been measured.

### F3. PonderNet: geometric-prior halting trained end-to-end
- **Link:** [arXiv 2107.05407 (abstract)](https://arxiv.org/abs/2107.05407), [HTML v2](https://arxiv.org/html/2107.05407v2)
- **Year:** 2021
- **Mechanism (verified, Sections 2.2 to 2.5):** Conditional halting probability lambda_n at each step; unconditional p_n = lambda_n * prod_{j<n}(1 - lambda_j). Loss L = sum_n p_n * l(y, y_hat_n) + beta * KL(p || p_G(lambda_p)), with beta = 0.01 in the experiments.
- **Prior (verified):** Geometric prior with parameter lambda_p, which biases toward an expected number of steps 1/lambda_p. The paper says lambda_p is easy to interpret as the inverse of the expected number of steps, gives no fixed rule for choosing it, and reports that lambda_p = 0.9 failed on parity while 0.1 settled near 3 steps. Appendix B sweeps 10 values in (0, 1].
- **Training unroll (verified):** N is the smallest step where cumulative halting probability exceeds 1 - eps (eps = 0.05). Leftover probability can be assigned to the last step.
- **Inference (verified):** Halting is sampled from the Bernoulli at each step, so exit is stochastic. The output is the prediction at the halting step. Computing the expected prediction over steps is possible but requires running all steps, which removes the savings.
- **Stated limitations (verified):** The prior is a soft KL regularizer, not a hard compute bound; the unroll cap N is what limits compute. Chaining PonderNet modules makes network size grow exponentially (Appendix A.2).
- **KV cache for halted tokens:** Not discussed in the paper (verified by absence in the fetched text).
- **Budget-invariance:** Not addressed. The output is taken at a sampled step, so it depends on the sampled exit.
- **Code complexity:** Low. The loss is a sum over steps plus a KL term, and the prior is one scalar.
- **Fit at 50M with KV cache:** The loss is trivially compatible with 50M. The KV question is open and must be solved separately (see F1 for a reuse scheme).

### F4. Adaptive Computation Time (ACT), Graves 2016
- **Link:** [arXiv 1603.08983 (abstract)](https://arxiv.org/abs/1603.08983). The PDF and HTML could not be read by the fetch tool, so the body-level claims below are unverified.
- **Year:** 2016
- **Mechanism (unverified on page; from memory):** Halting probabilities per step accumulate until a threshold of 1 - eps, with a remainder term for the last step, and the output is a weighted average of step outputs. A ponder cost tau * (number of steps) is added to the loss.
- **Verified from abstract:** ACT is deterministic and differentiable, adds no noise to parameter gradients, and needs minimal architecture changes. On four synthetic tasks it gives large gains. On character-level Wikipedia language modelling it "does not yield large gains in performance," and extra computation went to hard transitions such as spaces and sentence ends.
- **Code complexity:** Low.
- **Fit at 50M with KV cache:** Low-risk on size, but the weighted-average output means every step must be computed for every token. The abstract's language-model result shows little gain, so the case for ACT on a 50M LM is weak.

### F5. Recurrent-depth latent reasoning (test-time scaling)
- **Link:** [arXiv 2502.05171 (abstract)](https://arxiv.org/abs/2502.05171)
- **Year:** 2025 (February)
- **Verified from abstract:** A recurrent block is iterated to arbitrary test-time depth. The proof-of-concept model has 3.5B parameters, trained on 800B tokens, and reaches compute "equivalent to 50 billion parameters" at test time. The abstract makes no claim about per-token exit.
- **Snippet only (not opened):** Zero-shot per-token early exit using KL between successive recurrent steps; for missing KV, attend to the deepest available cache entry, following Elbayad et al. (2019).
- **Fit at 50M:** Scale is far above 50M. Useful as a reference for KV handling, not as a template.

### F6. ADEPT and early-exit survey (snippets only)
- **ADEPT:** [arXiv 2601.03700](https://arxiv.org/html/2601.03700v1), 2026. Snippet only: token-level early exit in prefill and generation, KV for skipped layers generated in a decoupled way, up to 25% efficiency gain in generation. Unverified.
- **Survey:** [arXiv 2509.05915](https://arxiv.org/abs/2509.05915), 2025. Snippet only: per-token early exit can create system-level bottlenecks (for example batch idling) that offset savings. Unverified.

### F7. Mixture-of-Depths (context only)
- **Link:** [arXiv 2404.02258](https://arxiv.org/abs/2404.02258), 2024. Snippet only: a fixed per-layer token capacity k; skipped tokens pass through the residual connection. Unverified beyond the snippet. Useful as a fixed-budget comparator, which is what budget-invariance needs.

---

## Answers to the three questions

**(1) Per-token early exit that saves real compute, including the cache.**
- Verified designs that avoid computing exited tokens at all: MoR (F2), with real throughput gains (up to about 2.06x at 360M). Its cache handling is recursion-wise caching, and its shared-KV variant degrades under expert-choice.
- AdaPonderLM (F1) reuses cached KV for halted tokens, but as described its algorithm still computes K/V for all tokens. Its published savings are FLOP-accounting (about 5% in its own table), not wall-clock.
- Any cache scheme must answer: what K/V does a halted token expose to later tokens? The options verified in the sources are (a) reuse the last computed K/V (F1), (b) keep K/V only at the depths where a token was active and restrict attention (F2), or (c) share first-step K/V (F2, degrades under expert-choice).

**(2) Budget-invariance (output independent of the chosen number of loops).**
- No verified source here establishes budget-invariance. MoR (F2) states that output depends on routing and that changing top-k after training is hard. AdaPonderLM (F1) and PonderNet (F3) both depend on sampled or gated exit.
- Recommendation derived from these sources, not from a verified paper: train with a fixed depth readout and a loss that makes the head's output agree across depths, or train the exit head so that the output at the chosen depth matches the output at full depth. This is unverified and needs a source check before adoption.

**(3) PonderNet- or ACT-style halting with a sensible prior.**
- PonderNet (F3) gives a principled, verified prior: geometric with lambda_p, where the expected number of steps is 1/lambda_p, with a KL term weighted by beta = 0.01. The paper gives no guidance beyond a sweep, so set lambda_p from the target average depth.
- ACT (F4) has no prior; it uses a ponder cost tau, which is the source of its sensitivity (see below).

---

## ACT failure modes

Verified from the fetched or searched text:
1. **Sensitivity to the time penalty tau.** The original paper says the model's behaviour is quite sensitive to tau and that the right value is not obvious (search snippet from a secondary summary; original body not read, so treat as partially verified).
2. **Runaway computation without a cap.** Without constraints, the network keeps computing as long as it can. A hard cap M on steps is a common fix (search snippet; unverified on the original page).
3. **Weak gains on language modelling.** Verified from the abstract: ACT "does not yield large gains in performance" on character-level Wikipedia LM.
4. **Biased gradients for the cost.** Attributed in search snippets to the PonderNet authors' critique: the gradient for computation cost only back-propagates through the last step, which biases the estimate. Not confirmed in the PonderNet text that was fetched, so unverified.
5. **Output as a weighted average.** Every step must be computed for each token, so the weighted average gives no compute saving at inference. Verified against PonderNet's description of ACT's output (search snippet; unverified on the ACT page).

---

## Ranked recommendation for Inky (at most three)

1. **PonderNet-style geometric-prior halting plus a hard unroll cap, with KV reuse from F1.** Trade-off: lowest code complexity and a principled, interpretable prior (lambda_p = 1/expected steps). Cons: inference-time exit is stochastic unless you accept full-depth compute for the expected prediction; KV reuse for halted tokens is taken from F1 and is not measured on code; budget-invariance is not solved.
2. **MoR-style expert-choice router with recursion-wise KV caching.** Trade-off: the only verified option with real throughput gains (up to about 2.06x at 360M). Cons: highest code complexity (gather/scatter, per-depth KV bookkeeping), a fixed top-k budget means outputs depend on capacity, and the 50M fit is untested.
3. **AdaPonderLM-style per-iteration gates with monotonic mask and KV overwrite.** Trade-off: small, cache-compatible design with a two-stage schedule and reported PPL close to baselines at 70M and 410M. Cons: savings as described are FLOP accounting (about 5%) rather than wall-clock; the paper's own numbers conflict on the headline reduction; no budget-invariance.

Recommended order for Inky if one option is adopted: start with option 1 (lowest risk) and add the MoR routing only if throughput measurements show a need.

---

## Sources

- [AdaPonderLM, arXiv 2603.01914 (2026), abstract](https://arxiv.org/abs/2603.01914)
- [AdaPonderLM, HTML v2 (2026)](https://arxiv.org/html/2603.01914v2)
- [Mixture-of-Recursions, arXiv 2507.10524 (2025), abstract](https://arxiv.org/abs/2507.10524)
- [Mixture-of-Recursions, HTML (2025)](https://arxiv.org/html/2507.10524)
- [PonderNet, arXiv 2107.05407 (2021), abstract](https://arxiv.org/abs/2107.05407)
- [PonderNet, HTML v2 (2021)](https://arxiv.org/html/2107.05407v2)
- [Adaptive Computation Time, arXiv 1603.08983 (2016), abstract](https://arxiv.org/abs/1603.08983)
- [Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach, arXiv 2502.05171 (2025), abstract](https://arxiv.org/abs/2502.05171)
- [ADEPT, arXiv 2601.03700 (2026), snippet only](https://arxiv.org/html/2601.03700v1)
- [Accelerating LLM Inference via Early-Exiting Algorithms, arXiv 2509.05915 (2025), snippet only](https://arxiv.org/abs/2509.05915)
- [Mixture-of-Depths, arXiv 2404.02258 (2024), snippet only](https://arxiv.org/abs/2404.02258)
