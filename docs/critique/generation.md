# Generation and writing controls: critique

## Status (checked against the code, 2026-11-03)

Checked against the working tree at commit f7299ea plus uncommitted changes. **Line numbers in the
findings below are from an older commit and may not match the current code.** The finding text is
kept as written. Legend: **FIXED** = the fix is in the code (a note says which part, for example opt-in). **OPEN** = the problem remains or is only partly fixed (the note says which part). **ACCEPTED-DESIGN** = a deliberate design choice, not a defect; it is documented and was not changed. **NOT-A-BUG** = checked, and the claim does not hold for the code as designed. **UNVERIFIED** = could not be checked here.

Counts for this file: 10 FIXED, 4 OPEN (of 14).

| ID | Severity | Status | Evidence in current code |
| :--- | :--- | :--- | :--- |
| GEN-01 | high | FIXED | Notes are capped at 256 tokens (`DEFAULT_MAX_NOTE_TOKENS`). A block over the cap keeps its text, and an unpaired open stays as text. Measured: a stray open followed by 300 tokens of prose keeps the prose. The original repro `[1, 2, OPEN, 10, 11, 12, SEP, 13, CLOSE, 14] -> [1, 2, 13, 14]` is a well-formed block, so it is NOT-A-BUG for that input. |
| GEN-02 | high | FIXED | The rebuild after a scratch close passes the cache's `num_loops`. |
| GEN-03 | high | OPEN | The re-prefill cost is fixed (committed caches and forks). The speculative path is still greedy only, with no writing controls; its docstring now says so. |
| GEN-04 | high | FIXED | Opt-in: `cache_dequantized=True` keeps a float copy and is the fast path. The default still dequantizes on every call (`quantize.py:149-167`), so default quantized decoding is slow on CPU. |
| GEN-05 | high | FIXED | With `top_k` set (the default is 50), only the k largest logits are read (`torch.topk`). The full vocabulary is sorted only when `top_k=0` and `top_p < 1`. |
| GEN-06 | medium | FIXED | Opt-in: `repeat_window` limits the repetition penalty to the last N tokens. The default (None) still applies it to the whole sequence, prompt included. |
| GEN-07 | medium | FIXED | Opt-in: the same `repeat_window` limits the no-repeat rule. The default (None) still uses the whole history. |
| GEN-08 | medium | OPEN | Single stock words are no longer hard-banned, and a soft penalty exists, but its default is 0. Phrases such as "embark on" are still hard-banned, with no narration-only scope. |
| GEN-09 | medium | OPEN | `text_report` adds a decoded-text phrase check. Generation still bans token sequences only. |
| GEN-10 | medium | FIXED | A scratch close no longer re-prefills the whole sequence. Only the text after the marker is recomputed (the `generate` docstring, `modeling.py:660-666`). |
| GEN-11 | low | FIXED | `NgramBanIndex` is incremental, and the bad-word check looks only at the tail of the row. |
| GEN-12 | low | FIXED | `no_repeat_ngram_size=1` raises `ValueError` (`modeling.py:672`). |
| GEN-13 | low | FIXED | Opt-in: `count_prose_only=True` counts only prose toward `max_new_tokens`. The default still counts notes. |
| GEN-14 | low | OPEN | An evaluation harness exists (`inky/evaluation.py`, `inky/metrics.py`), but there is no trained model to score the defaults on. |

Scope: `inky/modeling.py` (`generate`, `_block_tokens`, `_apply_repetition_penalty`, `_sample`),
`inky/style.py`, `inky/scratch.py`, `inky/speculative.py`, `inky/quantize.py`. No file under `inky/`
or `tests/` was changed.

Measurement setup: CPU (torch 2.13.0+cpu). Repro and timing script kept outside the project. The
generation tests (`tests/test_speculative.py`, `tests/test_quantize.py`, `tests/test_inky.py`) pass
(31 tests), but they run a tiny random model (vocab 512, 64 hidden) with a scripted sampler in
the scratch case.

"Confirmed (measured)" means a run on this machine. "Confirmed (trace)" means the code path was
followed and an instrumented call showed the behaviour. "Read only" means the claim comes from code
and arithmetic, not a run.

---

## GEN-01 - Scratch reduction deletes prose after a stray scratch_open

- **Severity:** high
- **File:** `inky/scratch.py:31-38` (the `sep` and `close` branches), called from `inky/modeling.py:380-383`
- **What is wrong:** A frame is pushed on every `open`. A later `sep` pairs with the top frame even
  when that frame is a stray open with no separator. When a `close` then arrives, the code deletes
  `out[frame["open"] : frame["sep"] + 1]`, which is everything from the stray open to that separator,
  including ordinary prose. Repro: `reduce_scratch([1, 2, OPEN, 10, 11, 12, SEP, 13, CLOSE, 14], OPEN, SEP, CLOSE, start=2)` returns `[1, 2, 13, 14]`. Tokens 10, 11, 12 (a sentence the user would keep) are gone. (Confirmed, measured.)
- **Why it matters for creative writing:** The model is untrained for scratch markers (README,
  "Scratch needs training"), so stray markers are the expected case, not an edge case. Silent loss
  of written text in a long chapter is the worst failure a writing tool can have, and nothing
  reports it.
- **Fix:** (a) Cap note length (for example 256 tokens). If a block grows past the cap, unpair its
  frame and keep its tokens as text. (b) Refuse a `sep` when no `open` is pending in the current
  turn, using the same logit-mask approach as `bad_word_ids`, so stray separators are never
  sampled. (c) Add a test for `OPEN, text, SEP, text, CLOSE` and for a lone `SEP` after prose.

## GEN-02 - Scratch rebuild drops `num_loops`, so loop depth changes mid-text

- **Severity:** high
- **File:** `inky/modeling.py:388` (`out = self(seq, past=cache, last_only=True)`), with `_choose_loops` at `inky/modeling.py:94-106`
- **What is wrong:** The first prefill passes `num_loops`, but the rebuild after a scratch close
  does not. The new `InkyCache` has `num_loops=None`, so `_choose_loops` falls back to
  `config.max_recurrent_loops` (6 by default). In an instrumented run with `num_loops=2`, the calls
  to `_choose_loops` were `[2, None, None, ...]`, and the rebuild received `None`, so the text after
  the first scratch close is produced at the default depth. (Confirmed, trace.)
- **Why it matters for creative writing:** A user who sets a shallow depth for speed or for a
  consistent voice gets two depths in one passage, after any scratch close. The mismatch is silent
  and no test covers `num_loops` with scratch enabled.
- **Fix:** Pass `num_loops=cache.num_loops` (or the original `num_loops`) in the rebuild call. Add a
  test that asserts `_choose_loops` sees the same value across the rebuild.

## GEN-03 - Speculative path re-prefills the whole context every round and ignores the writing controls

- **Severity:** high
- **File:** `inky/speculative.py:78` (drafter prefill), `:88` (verifier prefill), `:76-93` (round loop); docstring at `:9-11`, `:51-52`
- **What is wrong:** Each round builds a new drafter cache and a new verifier cache over the whole
  sequence. Each round therefore costs about 2L token-forwards to accept about `n+1` tokens. Measured
  on a tiny model, 120 new tokens after a 40-token prompt: greedy `generate` fed 160 tokens through
  the model, speculative decoding fed 4848 (30x). The loop never exits early, so cost grows with
  context: at 8k each round is an 8k prefill, twice. Separately, the function is greedy only. It has
  no repetition penalty, no no-repeat, no bad-word blocking and no scratch reduction, so any writing
  use of it returns the repetitive greedy text the controls exist to prevent. The equality claim
  (identical to greedy) is checked only on the tiny model; prefill and incremental decode can
  disagree on argmax ties in float arithmetic, and that was not tested. (Confirmed, measured.)
- **Why it matters for creative writing:** The file presents this as a speedup. It is slower than
  plain decoding at any context length, and it cannot produce the output the writing controls are
  meant to shape. The tests pass, so the problem is not visible in CI.
- **Fix:** Implement the cache truncation that the docstring lists as future work (roll back KV
  entries and SSM prefix sums to the accepted length). Until then, mark the function
  "not for use with writing controls", raise if any control is set, and do not present it as a
  speed path in the README.

## GEN-04 - Quantized linears dequantize the full weight on every forward, so CPU decoding gets slower

- **Severity:** high
- **File:** `inky/quantize.py:84-95` (`dequantize`), `:102-105` (`forward`), `:97-100` (`weight` property); `inky/layers.py:149` (absorbed MLA decode reads `kv_b_proj.weight`); `inky/quantize.py:13` (`DEFAULT_SKIP_NAMES` does not skip `kv_b_proj`)
- **What is wrong:** Weights are stored as int8 or packed int4, but each call to `forward`
  reconstructs a full float matrix (for int4, via bit unpacking, stack, reshape and a scale
  multiply). On the decode shape (1 x 384 input, 384 x 1536 weight), measured per call: float
  `Linear` 0.18 ms, int8 `QuantLinear` 2.67 ms (15x slower), int4 `QuantLinear` 8.8 ms (50x slower).
  The absorbed MLA decode also reads `.weight`, which dequantizes `kv_b_proj` again for every layer,
  loop and token. Because `kv_b_proj` is not in the skip list, it is quantized and then dequantized
  on every step. (Confirmed, measured for the linear layer; the `kv_b_proj` path is confirmed by
  reading.)
- **Why it matters for creative writing:** The quantization path saves storage (about 190 MB fp32
  down) but makes decoding slower on CPU, which is where this project runs. The PLAN.md "low-cost
  inference" goal is not met by the current implementation, and no test measures speed.
- **Fix:** For CPU decode, keep a dequantized float copy cached after the first call (memory cost,
  speed win), or use a real int8 matmul kernel (for example `torch._int_mm` or a dynamic-quantized
  linear). Precompute `w_uk` and `w_uv` once in eval mode rather than per step. Add `kv_b_proj` to
  the skip list or handle it explicitly. Add a decode-speed check to `tests/test_quantize.py`.
  Also note that symmetric int4 with max/7 scales and no calibration has not been checked for
  quality on trained weights (read only).

## GEN-05 - Sampler sorts the full 50,003-token vocabulary on every step

- **Severity:** high
- **File:** `inky/modeling.py:426-440` (`_sample`); the full sort is at `:432`, and `top_k` at `:428-430` already reduces the set to 50 values
- **What is wrong:** With `top_p < 1` (default 0.95), the code sorts all 50,003 logits, runs softmax
  and cumsum over all of them, and then scatters back into a full-vocabulary tensor with `full_like`.
  `top_k` is applied first, so only 50 values can survive, but the sort still covers the whole
  vocabulary. Measured on this machine: 21.6 ms per call with top_k=50 and top_p=0.95, versus 9.9 ms
  with top_k only. Both figures are large next to the README's 131 to 185 ms decode per token (a
  different machine, so only the ratio is meaningful). (Confirmed, measured.)
- **Why it matters for creative writing:** This runs on every generated token and is a
  double-digit share of decode time on CPU. It is also the default configuration.
- **Fix:** Use `torch.topk(logits, k)` once, then apply top-p on those k sorted values, sample on
  the k-element distribution, and map the index back through the topk indices. Sampling only the
  k values avoids the full-vocabulary softmax and scatter. Add a test that the new path matches the
  old one for a fixed seed.

## GEN-06 - Repetition penalty 1.1 applies to every token in the prompt and history, including punctuation and names

- **Severity:** medium
- **File:** `inky/modeling.py:371` (call site, `seq` includes the prompt), `:418-423` (`_apply_repetition_penalty`); default at `:343`
- **What is wrong:** The penalty is set-based: every token that has appeared once, anywhere in the
  sequence, gets the same treatment. For a positive logit z it becomes z / 1.1, which removes about
  9% of z. A logit of 20 loses about 1.8 nats, and the division by temperature 0.8 that follows
  makes the effective shift about 11%. In an 8k context almost every common token (the, a, comma,
  period, the character names) has already appeared, so the penalty mostly reshuffles the
  distribution toward rare tokens rather than reducing repetition. (Read only; arithmetic from the
  code.)
- **Why it matters for creative writing:** Names, articles and punctuation get penalised in the
  place where a writer most needs them to be predictable. This pushes the prose toward unusual
  word choices (a thesaurus effect) and makes character names drift. The penalty is applied to the
  prompt, so a user's own names and phrasing are penalised too.
- **Fix:** Default to `repetition_penalty=1.0` and rely on the n-gram rule for loop control, or
  apply the penalty only to the last 64 to 128 tokens, exempt punctuation and stop words, and
  compare against a measured repetition metric (distinct-n or repeated-n-gram ratio) on a held-out
  set before picking a default.

## GEN-07 - No-repeat 4-gram rule uses the whole history on token ids, so it blocks legitimate reuse

- **Severity:** medium
- **File:** `inky/style.py:65-76` (`banned_by_no_repeat_ngram`), `inky/modeling.py:402-412` (`_block_tokens`); default `no_repeat_ngram_size=4` at `modeling.py:346`
- **What is wrong:** The rule bans the next token whenever the last three tokens have occurred
  before anywhere in the sequence, including the prompt. With BPE ids this is roughly "the same
  three-token phrase may not be continued the same way twice". Three problems follow. (1) Legitimate
  reuse is blocked: dialogue tags ("she said"), refrains in verse, character tags, deliberate
  callbacks. (2) The window never shrinks, so the number of forced deviations grows with the length
  of the story. (3) Matching is on ids, so the same words with a different leading space or
  capital letter tokenise differently and escape the rule, which makes the behaviour inconsistent.
  (Read only; the cost side is measured in GEN-10.)
- **Why it matters for creative writing:** A 4-gram rule over the full history is a blunt tool for
  prose. It creates awkward forced word choices late in a long chapter, and it is most damaging in
  exactly the places fiction repeats on purpose.
- **Fix:** Limit the rule to a sliding window (for example the last 256 tokens), raise n to 6 to 8
  for prose, and exempt short function-word n-grams. Consider a soft penalty in place of a hard
  ban. Measure with a repeated-n-gram metric and a human read on a fixed set of prompts that
  include refrains and dialogue.

## GEN-08 - Stock-phrase list hard-bans common words that writers need

- **Severity:** medium
- **File:** `inky/style.py:11-29` (`STOCK_PHRASES`), applied as a hard `-inf` through `inky/modeling.py:412-414`
- **What is wrong:** Several entries are ordinary words that fiction uses literally or as genre
  vocabulary: "realm" (fantasy staple), "tapestry" (a physical object in historical or descriptive
  prose), "testament" ("Old Testament", or "a testament to her courage" in any register), "embark
  on" (for example "embark on a quest"), "intricate" (a lace pattern, a clockwork mechanism),
  "underscore" (the character, or a verb in any setting), "pivotal" (a pivotal moment is also what
  thriller writers actually write), and "moreover" or "furthermore", which are normal in
  period dialogue. These are banned in every context, including dialogue and the user's own
  prompt, and there is no per-project override other than replacing the whole list. (Read only.)
- **Why it matters for creative writing:** Hard bans on single common words force the model into
  substitutes that are often worse ("the kingdom" for "the realm"), and they make the tool unusable
  for historical fiction, fantasy and academic-voice characters. The docstring itself says the list
  is "not a validated detector", so the default should not ban words that are good prose in some
  contexts.
- **Fix:** Turn the list into a soft penalty (for example -2 nats) rather than a hard ban, apply it
  to narration only (not inside quoted dialogue), and remove single common words (realm, tapestry,
  testament, intricate, underscore) from the default. Ship the default as an empty or minimal list
  and let each project opt in. Measure with a human-rated sample of outputs before and after.

## GEN-09 - Bad-word blocking works on token ids only, so text variants escape it, and the report shares the blind spot

- **Severity:** medium
- **File:** `inky/style.py:32-45` (`build_bad_word_ids` variants), `:48-62` (`banned_by_bad_words`); `inky/metrics.py:239-241` (the `blocked_phrase_hits` report uses the same token-level check)
- **What is wrong:** The list tries four surface forms per phrase (bare, capitalised, with a space,
  with a space and capital), but every other casing ("DELVE", "De lve"), curly apostrophes ("It’s
  important to note") and any tokenisation the variants did not cover will escape. For multi-token
  entries only the last token is blocked, and only after the earlier tokens match, so a phrase
  whose early tokens are produced differently is never blocked. Single-token entries are banned
  everywhere, including inside longer words if the tokenizer splits them that way (the tokenizer
  is not in this repo, so this last point is unverified). The `blocked_phrase_hits` report calls the
  same token-level function, so it cannot see these leaks either and will report zero hits for
  banned text that appears in another form. (Read only.)
- **Why it matters for creative writing:** The user believes the phrases are blocked, and the
  report says they are not present, when some are. Both claims come from the same blind spot.
- **Fix:** Add a text-level check after decoding (case-insensitive regex with word boundaries and
  both apostrophe forms) that flags a hit and resamples or penalises the next token. Make the
  report use the same text-level check, and measure the leak rate on sampled outputs.

## GEN-10 - Scratch reduction re-prefills the whole context every time a block closes

- **Severity:** medium
- **File:** `inky/modeling.py:380-389` (`reduce_scratch` call and cache rebuild at `:386-388`)
- **What is wrong:** Each close that changes the sequence throws the cache away and runs a full
  prefill over the reduced context. The README reports 25.0 s prefill at 8,186 tokens on the
  author's machine, so each scratch close can cost about that much. A causal model makes this
  avoidable: the cached keys and values before the `scratch_open` position are unchanged by the
  removal, so the cache can be truncated to that position and only the text after it recomputed.
  The SSM prefix sums need the same truncation. (Read only; the cost is from the README figures.)
- **Why it matters for creative writing:** A model that writes several notes in a long chapter pays
  a full-context prefill per note. This is the feature that is supposed to keep long writing
  workable, and it is the costliest path in `generate`.
- **Fix:** Store the cache length at each `scratch_open`. On a close, truncate the cache to that
  length and prefill only the tokens after the block. This needs the same KV and prefix-sum
  truncation as GEN-03. Add a test that the truncated-cache output matches the full-rebuild output
  on the tiny model.

## GEN-11 - Per-token n-gram blocking is O(context) in Python and calls `tolist()` each step

- **Severity:** low
- **File:** `inky/style.py:71-76`, `inky/modeling.py:408-410`
- **What is wrong:** Every generated token rebuilds a Python list from the tensor (`seq[b].tolist()`)
  and scans the whole history, creating a tuple per position. Measured on random ids (the cheap
  case, since repeats are rare): 0.52 ms per token at 1k context, 1.18 ms at 4k, 2.50 ms at 8k. The
  stock bad-word check is negligible (0.04 to 0.07 ms per token for 60 sequences). On real text,
  where repeats are common, the set insertions add more. A 1,000-token reply at 8k context spends
  about 1.5 to 2 s here. This is small next to the README's 131 to 185 ms per decode step, so it is a
  low-severity cost. (Confirmed, measured.)
- **Why it matters for creative writing:** It grows linearly with the length of the story, so it is
  the one control whose cost scales with the thing the user cares about most. Fixing GEN-07 (a
  window) also fixes this.
- **Fix:** Keep an incremental dictionary from (n-1)-gram to next-token set, updated with one entry
  per generated token. Build it once from the prompt. Look-ups become O(1) per step. Keep the tensor
  on the device and avoid the per-step `tolist()`.

## GEN-12 - Default stacking is not validated, and `no_repeat_ngram_size=1` bans every token already seen

- **Severity:** low
- **File:** defaults at `inky/modeling.py:340-346`; `inky/style.py:67-69` (n=1 path); `inky/modeling.py:427` (temperature before top-k)
- **What is wrong:** The defaults combine temperature 0.8, top_k 50, top_p 0.95, repetition penalty
  1.1 and no-repeat 4, plus the stock bans. The penalty runs on raw logits before temperature, so its
  effect is scaled by 1/0.8. top_k=50 and top_p=0.95 overlap heavily on a 50k vocabulary. None of
  this combination has been measured on a trained model (PLAN.md says "need a measurement"). Also,
  `no_repeat_ngram_size=1` returns `set(ids)`, which bans every token already in the prompt and the
  output. That is almost unusable for prose, and the function accepts it without a warning. (Read
  only.)
- **Why it matters for creative writing:** The controls interact, so tuning one at a time gives
  misleading results. A user who sets n=1 gets degenerate text with no error.
- **Fix:** Reject n=1 (or document it as a degenerate setting). Choose defaults with a small grid
  over temperature, top_p, repetition penalty and n on a fixed prompt set, scored with a
  repetition metric and a human read. Record the chosen values next to the README claims.

## GEN-13 - `max_new_tokens` counts scratch notes, so the visible text can be much shorter than requested

- **Severity:** low
- **File:** `inky/modeling.py:368` (`for _ in range(max_new_tokens)`), `:380-389` (reduction after the budget has been spent)
- **What is wrong:** The budget counts every sampled token, including notes that the reduction
  removes afterwards. A request for 500 tokens of prose with several scratch blocks can return far
  fewer prose tokens, and nothing reports the difference. (Read only.)
- **Why it matters for creative writing:** The user gets a short chapter with no error, and may
  retry with a larger budget that then spends more on notes.
- **Fix:** Count only tokens that survive reduction against the budget, or return the number of
  notes and visible tokens in the `report` output so the difference is visible.

## GEN-14 - No evaluation harness checks any of these defaults on a trained model

- **Severity:** low
- **File:** `README.md` ("Limits" and "Prose controls" sections); `PLAN.md` (rows "Scratch reduction, prose controls": "Need a measurement")
- **What is wrong:** Every default in this review (temperature, top_p, top_k, repetition penalty,
  no-repeat size, stock list, scratch reduction) is untested on a trained model. The tests use a
  tiny random model, a scripted sampler, and structural checks only. `text_report` measures
  structure, not whether the output reads well. (Read only.)
- **Why it matters for creative writing:** The controls may be making the prose worse, and the
  project has no way to tell. Every entry above is a claim that needs that measurement.
- **Fix:** Build a small fixed prompt set (fiction, dialogue, verse, a long-context continuation).
  Score each setting with repeated-n-gram rate, distinct-n, stock-phrase hit rate under the
  text-level check (GEN-09), scratch leak rate (GEN-01), and a blind human preference test. Use
  those results to set the defaults.

---

## Summary by priority

1. GEN-01 scratch reduction deletes prose after a stray open (confirmed, measured).
2. GEN-02 `num_loops` lost on scratch rebuild (confirmed, trace).
3. GEN-03 speculative path does about 30x the model work and ignores the writing controls (confirmed, measured).
4. GEN-04 quantized linears dequantize every step, 15x (int8) to 50x (int4) slower per call (confirmed, measured).
5. GEN-05 full-vocabulary sort every token, about 22 ms (confirmed, measured).
6. GEN-06 to GEN-09: the prose controls (penalty, n-gram, stock list, bad-word matching) need a measured default and a text-level check.
7. GEN-10 scratch rebuild re-prefills the full context on every close.
8. GEN-11 to GEN-14: lower-impact costs and the missing evaluation.
