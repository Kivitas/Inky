# Training plan

**Status: plan, not result.** Nothing in this document has been run. Every number is a planning
assumption, a guideline, or a target to be measured. No model has been trained, and no quality claim
is made.

## 1. Token budget

Target model: Inky at ~50M parameters (50.22M total, of which 19.2M is the tied embedding).

| Quantity | Value | Status |
| :--- | :--- | :--- |
| Guideline range | 1-5B training tokens | Heuristic, not measured |
| Lower bound (about 20 tokens per parameter) | ~1B tokens | Heuristic |
| Upper bound (about 100 tokens per parameter) | ~5B tokens | Heuristic |

The range is a guideline taken from the common practice of training small models well past the
compute-optimal ratio. It is not derived from Inky's own loss curves, and it should be replaced by
measured scaling once a dense baseline exists. Plan to run the lower bound first, then decide from
the held-out curve whether more tokens are worth the compute.

## 2. Corpus sourcing

No corpus is in the repository, and the choice of sources is the owner's decision. Options:

- **Public-domain books.** Older public-domain texts (for example Project Gutenberg). Check the
  copyright status in the jurisdiction that applies, because the rules differ by country and some
  editions carry their own copyright.
- **Licensed data.** Text licensed for training, with the licence recorded for each source.
- **Owner-written text.** Text written or owned by the project, with a clear grant.
- **Excluded.** Scraped web text whose licence is unknown, and any corpus that cannot be
  documented. The project should not train on material it cannot describe.

For every source, record the licence, the date of retrieval, and the filter applied. The existing
pipeline (`inky/data.py`) already does cleaning, a quality filter, near-duplicate removal and packing,
so the sources only need to be listed in a manifest.

## 3. Training stages

Each stage is a plan. The step counts and learning rates are placeholders to be set from the
dense baseline.

1. **Short-context pretraining, 1024 tokens.** Most of the token budget goes here, because it is the
   cheapest per token. Use the WSD schedule already in `inky/train.py`, with the loop curriculum
   turned on so that depth rises from 2 to 6.
2. **Extend to 8k.** Continue from the stage 1 checkpoint at 8192 tokens, with a small share of the
   budget (for example 10-15%, to be decided). The RoPE base is already set for 8k
   (CORE-05, accepted design). Gradient checkpointing is available if memory is short.
3. **Anneal.** A short decay phase on the highest-quality subset of the corpus, at a low learning
   rate, to finish the run.
4. **Style and instruction tuning.** Supervised fine-tuning on writing-task examples with a clear
   licence, then, if needed, a preference stage. This stage does not exist yet, and its data source
   is not chosen.

Checkpoints are saved under `<out-dir>/checkpoints/step_N/`, and `best.pt` keeps the lowest held-out
NLL. Each stage starts from the previous stage's checkpoint and logs its own held-out evaluation.

## 4. Ablation table

Run each row against the same token budget and the same held-out split. The dense baseline is row 1,
and every other row is compared with it.

| # | Variant | Question it answers | Compare against | Status |
| :-- | :--- | :--- | :--- | :--- |
| 1 | Dense transformer, matched total and active size | Does recurrence or MoE help at all? | Itself | Planned |
| 2 | MoE (shared expert plus routed experts) | Does MoE beat dense at this active-parameter scale and short budget? | Row 1 | Planned |
| 3 | Vocabulary 16k, 32k, 50k | How much budget does the embedding cost, and what does it buy? | 50k | Planned |
| 4 | Highway on and off | Does the causal running mean add anything? | On | Planned |
| 5 | QK-norm on and off | Does it stabilise training or change loss? | Off | Planned (`qk_norm` flag exists, default off) |
| 6 | PonderNet halting against Mixture-of-Recursions-style routing | Which saves more training compute at equal loss? | PonderNet | Planned |
| 7 | Muon plus AdamW against AdamW alone | Does Muon pay for its per-step cost at this scale? | AdamW | Planned |
| 8 | Training depth fixed at 6 against 2-6 with curriculum | Does training at mixed depth reduce the train and inference mismatch? | Fixed 6 | Planned |
| 9 | Scratch reduction on and off | Does PENCIL-style scratch help prose? | Off | Planned, after training teaches the markers |

The table lists questions, not results. Each row needs held-out perplexity at the trained depth and
at depth 6, and the metrics in section 5.

## 5. Evaluation metrics

- **Held-out perplexity.** On a split held out before training, reported per depth (`eval_by_depth`)
  and at the final depth. This is the primary metric for the ablations.
- **Repetition.** Rate of repeated 4-grams and the distinct-n ratio on generated text, at fixed
  decoding settings. The current no-repeat rule should be reported alongside, so that the
  repetition figure is not just a measure of that rule.
- **Human preference.** Blind pairwise comparisons on a fixed set of writing prompts, judged against
  a dense baseline and against an established small model. Record the number of raters, the prompts,
  and the agreement rate. Preference results are only meaningful with enough prompts to separate the
  systems.
- **Operational metrics.** Training throughput, decode time per token at 1k, 4k and 8k, and the
  output drift of early exit against full depth. These are not quality metrics, but they decide
  whether a variant is usable.

Detector scores are not a metric in this plan. The project makes no claim about AI-text detection
(see the README).

## 6. Order of work

1. Build the dense baseline (row 1) on a small public-domain corpus, to check the pipeline end to end.
2. Measure the token budget curve on the baseline, and choose the target within the 1-5B range.
3. Run rows 2-4, then rows 5-8, against the same budget.
4. Run stages 2-3 on the chosen variant, and only then stage 4.
5. Re-measure decode speed after the cache and dispatch changes, and record the result in the
   README with its commit.
