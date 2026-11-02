# Inky: improvement plan (cost-first)

Status: Inky is an untrained 50.6M-parameter model with its code and tests in place. Nothing
about its writing quality has been measured yet. This plan orders work by cost and by how much
it tells us. It assumes no training data has been chosen yet.

## 1. What the research says (as of this search)

**Architecture trends**

- Hybrids of attention and cheap recurrent layers are the main direction. NVIDIA's Nemotron 3
  mixes mostly Mamba-2 layers with MoE layers and a few attention layers, and reports up to 3.3×
  inference throughput and a 1M-token context. Source:
  [Nemotron 3 (arXiv:2512.20856)](https://arxiv.org/pdf/2512.20856v1).
- Mamba-3 (ICLR 2026) reports a 0.6-point accuracy gain at 1.5B over Gated DeltaNet, and comparable
  perplexity to Mamba-2 at half the state size. Source:
  [ICLR 2026 listing](https://iclr.cc/virtual/2026/oral/10010353).
- Looped and recurrent-depth models are an active 2026 topic: allocating loop compute per token,
  stability of recurrent dynamics, and test-time loop scaling. Inky already has a looped core, so
  this is the most relevant area for it. Sources:
  [Allocating Recurrent Compute in Looped LMs (arXiv:2608.18230)](https://arxiv.org/pdf/2608.18230),
  [Gated Recurrent Transformers (arXiv:2608.15062)](https://arxiv.org/pdf/2608.15062),
  [Right Direction, Wrong Step (arXiv:2609.16665)](https://arxiv.org/pdf/2609.16665).
  The Ouro "2–3× parameter efficiency" figure came from a secondary survey, so I haven't verified it.

**Optimiser**

- Moonlight (Moonshot) reports Muon matches AdamW at about 52% of the training FLOPs, using weight
  decay and a per-parameter update-scale fix. These are self-reported, and I found no independent
  frontier-scale replication. Source:
  [Moonlight 16B A3B model card](https://huggingface.co/moonshotai/Moonlight-16B-A3B).
- A January 2026 post argues Muon's update shrinks as matrix width grows, so the scale fix matters.
  Source: [Scaling Is Not Plug-and-Play](https://huggingface.co/blog/bird-of-paradise/scaling-is-not-plug-and-play).

**Inference cost**

- Quantization is the cheapest first step, and it needs no training. Post-training methods such as
  GPTQ and AWQ apply. Source:
  [LLM Inference Optimization (2026)](https://www.morphllm.com/llm-inference-optimization).
- Speculative decoding: a small drafter proposes tokens and the target checks them. A 2026 paper
  combines quantized targets, quantization-aware distillation and speculative decoding. Source:
  [arXiv:2607.04244](https://arxiv.org/pdf/2607.04244).
- Decoding is memory-bandwidth-bound, so smaller KV caches help. Source: the same guide.

**Small-model training cost**

- An Apple study found that maximising GPU memory use is not cost-optimal for training 100M–2B
  models, so cheaper, lower-utilisation setups can win. Source:
  [Computational Bottlenecks of Training SLMs](https://arxiv.org/html/2410.19456v1).
- Vendor cost figures ($500–$5,000 to train a sub-1B model) are unverified marketing numbers.
  I haven't used them for any decision here.

## 2. How Inky compares

| Trend | Inky today | Gap |
| :--- | :--- | :--- |
| Hybrid attention + cheap recurrence | Attention in every block; a small SSM highway across loops | No SSM *layer*. Candidate for a Nemotron-style hybrid core. |
| Looped depth | Yes: 6 loops, ACT halting, FiLM per loop | Halting and loop count are untested. Per-token loop allocation is an open question. |
| Sparse experts | Yes: 1 shared + 8 routed, top-2 | Fine. Expert count is untested. |
| Latent KV cache | Yes, with absorbed decoding | Cache is already small. KV quantization adds little. |
| Muon optimiser | Yes, with EOS's update scale | Moonlight's scale fix is not applied. |
| Low-cost inference | fp32 weights only (~190 MB) | No quantization path. Untested. |
| Speculative decoding | No | Loops give a natural cheap drafter (see §3). |
| Scratch reduction, prose controls | Yes (untrained) | Need a measurement to show they help. |

## 3. Plan, ordered by cost

Each step has a check that decides whether to keep it. Costs are relative effort on this
machine (CPU, no training data yet).

### Phase 0: no training data needed (days, CPU only)

1. **Evaluation harness first.** Without it, no other step can be judged. Build a held-out split
   loader, perplexity, and the existing `metrics.py` reports. *Check:* the harness runs on random
   tokens and reports numbers that match hand calculations.
2. **Inference quantization.** Add int8 and 4-bit weight-only quantization for the linear layers.
   The 50M model drops from ~97 MB (bf16) to ~24 MB (4-bit). *Check:* perplexity change on the
   harness, and tokens per second, once a trained model exists. Until then, check size and speed only.
3. **Self-speculative decoding (a research idea, untested).** Use the same weights with fewer loops
   as the drafter and the full loop count as the verifier. Greedy decoding has to match exactly,
   and the draft acceptance rate has to be high enough to pay off. *Check:* the output matches
   plain decoding bit for bit, and the acceptance rate is measured.
4. **Muon update-scale option.** Add Moonlight's scale fix behind a flag, default off. *Check:* the
   existing optimiser tests pass, and the loss curve on a fixed batch is compared with and without it.

### Phase 1: needs a data decision (your call)

5. **Training loop with checkpointing and logging.** Use the hybrid optimiser, the loop-count
   sampling already in the model, and a cosine or WSD schedule. Start at short context (1,024)
   and extend to 8k later. *Check:* training loss falls and the harness perplexity falls.
6. **Loop curriculum.** Start with few loops and increase them over training. This cuts early
   compute. *Check:* same final perplexity for less total compute.
7. **Data quality over quantity.** Deduplicate, filter, and keep a clean held-out split. For a
   50M model this usually matters more than model size. *Check:* the harness score per data slice.

### Phase 2: ablations (only after Phase 1 produces a baseline)

8. **Loop count, expert count, scratch on/off, Arlo on/off.** Each is one run with the harness.
   *Check:* keep a component only if it beats the baseline on perplexity or on Arlo's metrics.
9. **Hybrid core.** Replace some attention blocks in the recurrent core with a state-space layer,
   following Nemotron 3's pattern. *Check:* perplexity and tokens per second against the baseline.

### Phase 3: only if quality is short of the goal

10. **Distillation from a larger teacher.** This is the main lever for quality at a small size, but
    it needs teacher outputs and real compute. Defer it until the baseline is measured.

## 4. Cost levers, ranked by cost saved per unit of effort

1. Smaller, cleaner data (Phase 1, step 7). Usually the largest effect at this size.
2. Short-context training first, then 8k (step 5). Attention cost grows with context.
3. Loop curriculum (step 6). Compute scales with the number of loops.
4. Weight quantization for inference (step 2). Near-free, and it cuts memory.
5. Self-speculative decoding (step 3). Untested, so the gain is unknown.
6. Hybrid core (step 9). Cuts attention cost, but needs ablations first.

## 5. Risks and open questions

- **No measured quality.** Every claim about Inky's writing is pending the harness and a trained run.
- **Looped models are harder to train.** Recurrent dynamics can be unstable. Monitor loss spikes
  and the loop-to-loop change in hidden states.
- **Speculative decoding with loops** may accept too few drafts to help. Measure before building on it.
- **Vendor and secondary figures** above are not verified. Treat them as leads, not facts.

## 6. What I need from you

- A **data source** for Phase 1 (licensed or public-domain writing, and how much).
- **Compute**: a GPU, or CPU only. This decides whether the 8k context is practical.
- Whether to proceed through **Phase 0** now. It needs no data and no training.

## 7. Status (code written and tested; no training run)

| Step | Module | Status |
| :--- | :--- | :--- |
| 1 Evaluation harness | `inky/evaluation.py` | Done, tested |
| 2 Quantization | `inky/quantize.py` | Done, tested on the tiny config |
| 3 Self-speculative decoding | `inky/speculative.py` | Done; output matches greedy; rejection path tested |
| 4 Muon scale option | `inky/optim.py` (`scale_mode`) | Done, default unchanged; "moonlight" gives ~2× update size, needs LR tuning |
| 5 Training loop | `inky/train.py` | Done; smoke-tested only (3 steps) |
| 6 Loop curriculum | `inky/train.py` (`loop_curriculum`) | Done, tested |
| 7 Data pipeline | `inky/data.py` | Done; near-duplicate threshold 0.8 (30-word docs with one changed word are kept) |
| 8 Ablations | not built | Needs training runs; deferred |
| 9 Selective SSM mixer | `inky/layers.py` (`SelectiveSSMMixer`), `ssm_mixer_layers` | Done; off by default; tested |
| 10 Distillation loss | `inky/distill.py` | Done, tested; no teacher or training yet |

Test suite: 101 passing.
