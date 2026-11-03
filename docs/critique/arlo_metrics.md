# Critique: Arlo decision maker and metrics

## Status (checked against the code, 2026-11-03)

Checked against the working tree at commit f7299ea plus uncommitted changes. **Line numbers in the
findings below are from an older commit and may not match the current code.** The finding text is
kept as written. Legend: **FIXED** = the fix is in the code (a note says which part, for example opt-in). **OPEN** = the problem remains or is only partly fixed (the note says which part). **ACCEPTED-DESIGN** = a deliberate design choice, not a defect; it is documented and was not changed. **NOT-A-BUG** = checked, and the claim does not hold for the code as designed. **UNVERIFIED** = could not be checked here.

Counts for this file: 10 FIXED, 2 OPEN (of 12).

| ID | Severity | Status | Evidence in current code |
| :--- | :--- | :--- | :--- |
| ARLO-01 | high | FIXED | `train_arlo` trains only the Arlo parameters, with the language model frozen (`arlo.py:217-284`). `enable_arlo` defaults to False. `decide_*` still does not warn when the engine is untrained, so use its output only after `train_arlo`. |
| ARLO-02 | high | OPEN | The options run in one batched forward instead of N forwards. The prefix is still recomputed for each option, with no cache. |
| MET-01 | high | FIXED | A class with gold counts as 0 in macro precision, recall and F1 (`choice_report`). |
| MET-02 | high | FIXED | F-beta and per-option F1 are 0 when precision and recall are both 0. |
| ARLO-03 | medium | OPEN | The query still comes only from the prefix (`h[:1, P - 1]`, `modeling.py:559`). `choice_scores` lets the options attend to each other, which is a partial mitigation. |
| ARLO-04 | medium | FIXED | `summarise` weights the projected slots by their match with the query (softmax over slots), not a plain mean. |
| ARLO-05 | medium | FIXED | `decide_choice` and `decide_binary` restore the previous train/eval mode (`was_training` in `modeling.py`). |
| MET-03 | medium | FIXED | Confidence is computed for the decision made (`x if pred == 1 else 1 - x`). |
| MET-04 | medium | FIXED | Balanced accuracy is None unless recall and specificity are both defined. |
| MET-07 | medium | FIXED | `split_heldout` raises `ValueError` when the train part would be empty (`evaluation.py:47-48`). |
| MET-05 | low | FIXED | Gold out of range, an empty choice set and binary labels other than 0 or 1 raise `ValueError` (`metrics.py:76, 168, 189`). |
| MET-06 | low | FIXED | Top-1 uses the same argsort order as the ranks. |

Scope: `inky/arlo.py`, `decide_choice` and `decide_binary` in `inky/modeling.py`, `inky/metrics.py`, `inky/evaluation.py`.
No files under `inky/` or `tests/` were edited. Repro scripts ran from the session scratchpad with `PYTHONDONTWRITEBYTECODE=1`, so no `__pycache__` was written to the project.

Entries are ordered by importance. "Confirmed" means a repro was run. "Read" means the finding comes from reading the code only.

---

## ARLO-01 (high): the decision path has no training signal, so its output is an untrained prior

- Where: `inky/arlo.py:91-115` (read, summarise, choice_scores, binary_logit), `inky/modeling.py:245-248` (the only loss; Arlo is not in it), `inky/config.py:62` (`enable_arlo=True` by default), `tests/test_inky.py:74` (the test exempts `arlo.` parameters from the gradient check "by design").
- What is wrong: no loss touches the Arlo parameters. Every `decide_*` call is `@torch.no_grad()`. A model ships with `enable_arlo=True`, and its `decide_choice` and `decide_binary` return numbers from random weights.
- Why it matters: the README and the docstrings present Choice and Noul as working decision outputs. Any probability a user reads is noise. The metrics then measure noise, and the report looks precise.
- Repro (confirmed): after an LM loss backward, 0 of 27 `arlo.*` parameters have a gradient. On an untrained tiny model, `decide_binary` gives p between 0.47 and 0.62 across 6 random contexts. `decide_choice` per-option means stay fixed (0.14 to 0.39), with std across 20 contexts of 0.008 to 0.04. The output is almost a constant prior.
- Fix: add a decision loss on labelled data (cross-entropy over options for Choice, BCE for Noul). Train the Arlo parameters in that stage, or with a separate optimiser on frozen LM states. Until a decision checkpoint exists, either default `enable_arlo` to False or make `decide_*` raise or warn "untrained". Add a test that a few decision steps reduce the loss on a toy set. The current test only checks that the LM path is unchanged.

## ARLO-02 (high): decide_choice runs one full forward per option, recomputing the shared prefix

- Where: `inky/modeling.py:281-285`.
- What is wrong: the loop calls `self(torch.cat([prefix, opt]))` once per option. The prefix (context plus question) is recomputed N times, and no KV cache is used, although the method's docstring says the context states are identical for every option.
- Why it matters: cost is N full forwards of length P+L. For an 8k context and 8 candidate openings, that is 8 times the prefix work for a single decision. The code also has no batching.
- Repro (confirmed): 6 options with a 40-token prefix gave 6 forward calls.
- Fix: run the prefix once with the cache (`past`) to get `query` and `states`. Then score the options in one batched pass over the cached prefix: right-pad the options, pass an attention mask, and gather each option's last real token. Keep the current output as a reference test.

## MET-01 (high): macro averages in choice_report drop classes that have zero true positives

- Where: `inky/metrics.py:195` (per-option F1 is None when precision and recall are undefined), `:198-208` (macro_precision and macro_recall skip None, and macro_f1 skips None).
- What is wrong: a class with gold support but no predictions has precision None and recall 0. Its F1 is None and is dropped from `macro_f1`. A class that is predicted but never correct is dropped from `macro_recall`. The three macro values therefore average over different sets of classes, and classes the model fails on leave the average.
- Why it matters: the display can show a high macro F1 for a model that gets half its items wrong.
- Repro (confirmed): 2 items, gold [0, 2], predictions [0, 1]. `top1_accuracy` is 0.5, but `macro_f1` is 1.0 (only option 0 counts). A second case with gold [0,1,2,2] and all predictions equal to 0 gave `macro_precision` 0.25 and `macro_f1` 0.4, which do not agree with each other.
- Fix: follow the zero_division=0 convention for any class with gold support, so the class counts as 0 in precision, recall and F1. Exclude a class only when it has neither gold nor predictions. Print the number of classes in each average, and show per-class support next to the macro line.

## MET-02 (high): F1, F2 and F0.5 show n/a when the true value is 0

- Where: `inky/metrics.py:81-86` (`f_beta` returns None when `precision*beta^2 + recall == 0`).
- What is wrong: when precision and recall are both 0, F-beta is 0 by the standard limit, but the code returns None and the display prints "n/a". The same happens in choice per-option F1 at line 195.
- Why it matters: "n/a" tells the reader the metric is undefined, not that the model got every positive wrong. It hides the worst case.
- Repro (confirmed): 3 items, p = [0.7, 0.6, 0.2], gold [0, 0, 1]. The output is `precision` = 0.0 (tp 0, fp 2), `recall` = 0.0, `f1` = None, `f2` = None. Precision is not None here, so the None comes from the zero denominator.
- Fix: when precision and recall are both defined and both 0, return 0.0. When precision is undefined (no predicted positives), keep n/a for precision and state the reason in the display (for example "no predicted positives").

## ARLO-03 (medium): the Choice query does not depend on the options, and options are compared only through a dot product

- Where: `inky/modeling.py:284` (`query = h[:, P-1]` comes from the prefix only), `inky/modeling.py:285` (each option is represented by its last-token state), `inky/arlo.py:109-111` (`choice_scores` gets the query, the options and the context states, and does not look at the options as a set).
- What is wrong: the query is fixed for all options, so the options never influence it. Each option is scored independently from one last-token vector, and the softmax is the only interaction between options. Options of different lengths are not normalised: a longer option is judged by its final token alone. An empty option silently returns the prefix's last state.
- Why it matters: the module is described as choosing among candidates, but the scoring is closer to independent per-option scoring. Pairwise contrasts (for example "which opening is more concise") cannot be learned. Length effects go into the score unnoticed.
- Fix: pool each option's states (mean over its tokens, with a mask), and let the slots or the query attend over the set of option states so options can be compared. Reject empty options with a ValueError. Document the scoring rule in the docstring.

## ARLO-04 (medium): the K latent slots are averaged to one vector before use

- Where: `inky/arlo.py:106` (`self.read(states).mean(dim=1)`).
- What is wrong: `read` refines K slots through two gated steps, then `summarise` replaces them with their mean. The per-slot structure is discarded, so the slot count and refinement steps change only the mean. The slot gate and the attention over states are learned, but their result is collapsed linearly.
- Why it matters: the design presents K "hypotheses" that compete, yet nothing selects among them. The extra compute and parameters do little beyond a learned pooling.
- Fix: either use the slots directly (for example, attend from the query to the K slots, or flatten them into a learned projection), or simplify `read` to one attention pool and remove the slot machinery. Add an ablation: arlo_slots=1 against K=8 on the same decision set. Keep only what beats it.

## ARLO-05 (medium): decide_choice and decide_binary switch the model to eval mode and never restore it

- Where: `inky/modeling.py:276` and `:312` (`self.eval()` with no restore). The return type also depends on the flags at `:287-294` and `:316-321`.
- What is wrong: calling a decision method during training turns off dropout and other train-only behaviour for the rest of the run. `evaluation._run` restores the mode in a `finally` block, but these methods do not. The return type is a tensor by default, but a tuple with `report` or `display`.
- Why it matters: a silent mode change can skew training results. The variable return type makes the API error-prone.
- Repro (confirmed): `model.train()` followed by `decide_binary(...)` left `model.training` False.
- Fix: save `was_training = self.training`, call `self.eval()`, and restore it in a `finally` block, as `_run` does. Split the API into a plain probability method and a `*_with_report` method, so the return type is fixed.

## MET-03 (medium): binary confidence ignores the decision threshold

- Where: `inky/metrics.py:49-50`. `preds` uses `threshold`, but `confidence = max(x, 1-x)` assumes a threshold of 0.5.
- What is wrong: at threshold 0.8, p = 0.6 is predicted "no", so the confidence in that decision is P(no) = 0.4. The code reports 0.6. `mean_confidence` and the ECE bins (`:108`) use the same wrong value.
- Why it matters: the calibration figure describes a different decision from the one made.
- Repro (confirmed): p = [0.6, 0.6, 0.9, 0.1], gold [1, 0, 1, 0], threshold 0.8. Output: `decisions` [0, 0, 1, 0], `mean_confidence` 0.75, `ece` 0.10.
- Fix: `confidence = x if pred == 1 else 1 - x`, and compute ECE from that value.

## MET-04 (medium): balanced_accuracy and binary display substitute a single rate silently

- Where: `inky/metrics.py:101-102` (balanced accuracy averages whichever of recall and specificity exist), `:103` (Youden's J returns None in that case), `:108` (ECE uses the same substitution indirectly).
- What is wrong: when a set has no positives, recall is None, and balanced accuracy becomes the specificity alone. With no negatives it becomes recall alone. The display prints a number under the name "balanced accuracy" that is not balanced.
- Why it matters: a model that predicts "no" for everything on an all-negative set shows balanced accuracy 1.0, which is not a meaningful value.
- Repro (confirmed): gold [0,0,0] gives `recall` None, `specificity` 1.0, `balanced_accuracy` 1.0, `roc_auc` None.
- Fix: return None unless both recall and specificity are defined. Do not substitute silently.

## MET-07 (medium): evaluation.py reports mean_loops unweighted and counts padding tokens in perplexity

- Where: `inky/evaluation.py:8-9` (docstring says every token is weighted equally), `:72-79` (cross-entropy over all targets, with no ignore mask), `:92` (`mean_loops` is an unweighted mean of batch means), `:40-43` (`split_heldout` gives an empty train split when `n == 1`, with no error).
- What is wrong: `mean_loops` is averaged per batch, not per token or per sequence, so a short final batch counts as much as a full one. The NLL path has no pad-token mask, so padded batches inflate the token count and bias perplexity. `split_heldout` allows an empty training set.
- Why it matters: perplexity is only correct if the batches are unpadded. The docstring claims an equal per-token weighting that the loop count does not have.
- Fix: accept an optional `pad_id` or attention mask, and exclude those positions from both the NLL sum and the token count. Weight `mean_loops` by the token count of each batch. Raise in `split_heldout` if the train split would be empty. Add a test with padded batches that checks perplexity is unchanged after the mask is applied.

## MET-05 (low): weak validation of gold labels in choice_report and binary_report

- Where: `inky/metrics.py:165-167` (length check only), `:180-181` (`torch.tensor(g)` in the one-hot index), `:184-186` (confusion matrix uses `matrix[a][b]` directly), `:69-75` (binary labels are not checked against {0, 1}), `:106` (log loss uses truthiness, so a label of 2 counts as positive there but not in tp/fn).
- What is wrong: a negative gold index does not wrap silently in `choice_report` (the earlier `order.index` raises a `ValueError` that reads "-1 is not in list"), but the message does not say what went wrong. An empty item set raises `IndexError` because `torch.tensor([])` is a float tensor. A binary label of 2 gives inconsistent counts across metrics.
- Repro (confirmed): gold [-1, 1] raises `ValueError: -1 is not in list`. An empty item set raises `IndexError: tensors used as indices must be long...`.
- Fix: validate that choice gold lies in [0, K) and that binary labels lie in {0, 1}, with clear messages. Handle n == 0 explicitly by returning None fields.

## MET-06 (low): top-1 accuracy and rank-based metrics use different tie-breaking

- Where: `inky/metrics.py:153` (`argmax` gives the predicted index), `:170-171` (`argsort` gives the ranks), `:173` (top-1 accuracy from `pred`).
- What is wrong: with tied probabilities, `argmax` and `argsort` can choose different options. Then `top1_accuracy` and the rank-1 count disagree, and MRR and top-k can count the same item differently.
- Why it matters: small, but untrained near-uniform outputs (see ARLO-01) make ties more likely, and the report would show two different "top-1" numbers.
- Repro: read only. Not run.
- Fix: derive the prediction from the same ordering used for the ranks (for example, `order[0]`), or define the tie rule once and use it everywhere.

---

## Summary of what was run

- Repro scripts (scratchpad, not in the project): `C:\Users\SIRISH~1\AppData\Local\Temp\claude\C--Users-Sirisha-Rajana-AppData-Local-Claude-Data-anthropicpbc-claude-fnn82j28hfe8t-scratch-workspaces-bb9c0a75-e132-4b82-8492-a648e136af22-03c06548-7135-4b3d-991f-1566897bbfed-scratch-2026-11-02-9396-uyl6jt\627ccfef-fa24-4866-a818-e529a3172221\scratchpad\repro.py` and `repro2.py`.
- Confirmed by repro: ARLO-01, ARLO-02, ARLO-05, MET-01, MET-02, MET-03, MET-04, MET-05.
- Read only: ARLO-03, ARLO-04, MET-06, MET-07.
