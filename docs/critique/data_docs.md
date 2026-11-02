# Critique: data pipeline, config, docs and tests

Scope: `inky/data.py`, `inky/config.py`, `README.md`, `PLAN.md`, `tests/`, plus the EOS tokenizer at
`C:\Users\Sirisha Rajana\Desktop\New folder\EOS LLM\EOS_Tokenizer` (read only). Where a finding came from
running code, the measurement is given. Nothing under `inky/`, `tests/`, `README.md` or `PLAN.md` was edited.

Measurements used: the default model has 50,614,854 parameters (`count_parameters`). The EOS tokenizer has
50,000 ids: 49,974 BPE entries plus 26 added special tokens at ids 49974-49999.

## Answer to the tokenizer question

For English creative prose the tokenizer is efficient: a 106-character sample gave about 1.2 to 1.3 tokens
per word. The problems are elsewhere. The vocabulary was built for a multi-purpose model. About 14% of
entries contain code or markup characters (a crude regex, not a validated classification). It has 34
programming keywords and many code identifiers (`defaultValue`, `collectionView`, `-module`, `ffffff`). It has
26 special tokens for chat, tool calls, vision, fill-in-the-middle and repo context. Inky never uses these,
and nothing blocks them during generation. Non-English text is expensive: Hindi costs 1.20 tokens per
character and Chinese 1.38, against 0.22 for English. There are only 7 Devanagari tokens.

## Findings, most important first

### DATA-01 (high): quality_filter rejects non-Latin scripts
- `inky/data.py:17` (BASIC_PUNCTUATION), `inky/data.py:56-58` (symbol ratio).
- The symbol count is `not ch.isalnum() and not ch.isspace() and ch not in BASIC_PUNCTUATION`. Combining
  marks (Devanagari vowel signs, Thai vowels) are not `isalnum()`, and ideographic punctuation
  (。，、) is not in the set.
- Measured: a 336-character Hindi paragraph has a symbol ratio of 0.33 and is rejected. A Thai sample has a
  ratio of 0.19 and is rejected. A 992-character distinct Chinese document has a ratio of 0.12 and is
  rejected. An English curly-quote and em-dash paragraph has a ratio of 0.02 and passes.
- Why it matters: the filter silently removes whole languages from the corpus. The docstring says "letters",
  which is not what the code counts. The data is meant for a writing model, so the bias is a real problem.
- Fix: count only symbol-like categories (`unicodedata.category` starting with `S`, or `P` outside an allowed
  set) instead of "not isalnum". Treat `M*` as letters. Add typographic and CJK punctuation to the allowed set
  (`“”‘’—…。，、《》`). Add tests with Hindi, Thai, Chinese and Japanese samples that must pass.

### DATA-02 (high): dedupe and packing do not scale to a real corpus
- `inky/data.py:85-106` (dedupe keeps every kept document's 5-gram set in memory and compares each new
  document against all of them). `inky/data.py:109-124` (pack builds a Python list of every token).
  `inky/data.py:142-157` (`prepare_corpus` holds all encoded documents as lists at once).
- Measured: 300 documents of 3,000 words take a 160 MB peak in dedupe, about 170 bytes per shingle. That
  extrapolates to tens of GB for a 1 GB text corpus. The packing list costs roughly 8 bytes of pointer per
  token, plus a 28-byte int object for each token the encoder creates. `prepare_corpus` keeps the encoded
  lists alive as well, so the peak is several times the final tensor.
- Why it matters: PLAN.md step 7 and Phase 1 need a real corpus. The docstring warns only about the
  quadratic time, not about memory, so the risk is not visible.
- Fix: use MinHash with LSH banding for near-duplicates, and store only signatures. Stream the
  encode-and-pack step into a preallocated `numpy` or `torch` memmap or into shards. Document the memory
  per token in `data.py`.

### TEST-01 (high): speculative-decoding tests cannot detect a broken implementation
- `tests/test_speculative.py:43-47` (`seeded_model`), `:61-66`, `:81-86`, `:89-109` (rejection path).
- Measured: with the test's seed and prompt, the greedy continuation is token 7 repeated 12 times. The
  drafter agrees with it every time, so `acceptance_rate` is 1.0 (9 of 9 and 10 of 10). The rejection test
  also sees only token 7, so it cannot show that a wrong draft was rejected.
- Why it matters: "output matches greedy" passes for any decoder that emits 7 repeatedly, and PLAN.md
  section 7 (`PLAN.md:137`) cites these tests as the evidence that speculative decoding is correct.
- Fix: choose a seed or scale the weights so the greedy output has at least 4 distinct tokens. Assert
  `len(set(expected[0, 5:].tolist())) > 3` in the test. In the rejection test, assert that the verifier's
  output differs from the drafter's raw argmax at some position.

### TEST-02 (high): the bad-word test passes whether or not the ban works
- `tests/test_inky.py:246-251` (`test_bad_words_are_never_generated`).
- Measured: with `bad_word_ids=None`, token 77 is also never generated (the greedy output is 2, 3, 3, 3...).
  The assertion `77 not in out` is therefore always true for this model.
- Why it matters: the test can never fail, so it gives no protection to the prose-control feature that the
  README advertises.
- Fix: force the target into the output. For example, set the LM-head bias or raise the embedding row for
  token 77, then assert it appears without the ban and does not appear with it.

### DOC-01 (high): README says there is no training loop
- `README.md:4-5` ("It has no training data and no training loop. Those come later.")
- `inky/train.py` exists (warmup-stable-decay schedule, loop curriculum, checkpoints, `train()`),
  `tests/test_train.py` tests it, and PLAN.md section 7 marks the training loop "Done".
- Why it matters: a reader will believe the training code is missing. The file list in the README also omits it.
- Fix: rewrite the sentence to "There is no training data and no training run yet. `inky/train.py` has a
  smoke-tested loop."

### DATA-03 (medium): creative forms and layout are damaged by the filters and by clean_text
- Length and repetition rules: `inky/data.py:42` and `:147` (`min_chars=200` default in `quality_filter`
  and `prepare_corpus`). `inky/data.py:60-65` (repeated-line rule).
- Layout: `inky/data.py:34` (`re.sub(r"[ \t]+", " ", text)`).
- Measured: a three-line haiku, a 100-character micro-fiction and a 10-line poem with a 4-line refrain are
  all rejected. A 19-line chapter with 9 `* * *` scene breaks (9 of 19 lines repeat) is rejected. Every copy
  of a repeated line counts, so separators and refrains count against the document. `clean_text` turns
  `"    He stood.\n        She ran."` into `" He stood.\n She ran."`, so indentation in verse, screenplays and
  block quotes is lost, and the pipeline has no option to keep it.
- Why it matters: the writing model loses short forms and structured prose, which is the target use. The
  docstring discloses the collapse, but no option keeps the layout.
- Fix: make `min_chars` a per-source option, not a global. Exclude separator lines (only `*`, `-`, `#`, `~`)
  and lines under about 40 characters from the repeated-line count, or count only the extra copies. Collapse
  runs of spaces inside a line but keep leading indentation (`re.sub(r"(?<=\S)[ \t]+", " ", ...)`), and add a
  test for verse.

### DOC-02 (medium): README numbers and file list do not match the code
- `README.md:119` ("`print(model.count_parameters()["total"])  # about 50.2M`"). The measured total is 50,614,854.
- `README.md:14` gives "50.62M". The code gives 50.61M. The table components add to 50.62 only after rounding.
- `README.md:89-103` (Files list) omits `inky/data.py`, `inky/train.py`, `inky/evaluation.py`,
  `inky/metrics.py`, `inky/quantize.py`, `inky/speculative.py`, `inky/distill.py` and seven test files
  (`test_data`, `test_train`, `test_evaluation`, `test_quantize`, `test_speculative`, `test_distill`,
  `test_metrics`).
- `config.py:3-5` says "The Rhen decision layer is not included", while `config.py:62-65` turns Arlo on by
  default. Arlo is a port of Choice and Noul, so the module docstring contradicts itself.
- Why it matters: a user who runs the usage example sees 50.6M and thinks the README is wrong.
- Fix: change the usage comment to "about 50.6M" and the header to 50.61M. Regenerate the Files list from
  the tree. Change the config docstring to "Arlo (Choice and Noul primitives) is included; the other
  Rhen layers are not."

### DOC-03 (medium): PLAN.md section 2 describes work that section 7 already finished
- `PLAN.md:62` ("fp32 weights only (~190 MB). No quantization path. Untested.")
- `PLAN.md:63` ("Speculative decoding | No")
- Section 7 (`PLAN.md:136-137`) lists `inky/quantize.py` and `inky/speculative.py` as "Done, tested".
- `PLAN.md:61` says Moonlight's scale fix is "not applied". Section 7 says it is available behind a flag,
  default off. That is still true, but the row should say so.
- Why it matters: the plan is the document that decides what to do next, and it contradicts itself.
- Fix: update the "Inky today" column to match section 7, and state the default for each flag.

### DOC-04 (medium): README presents unmeasured effects as facts
- `README.md:86-87` ("These controls improve variety"). `PLAN.md:64` says the same controls "Need a
  measurement to show they help."
- `README.md:80-82` ("The default list ... covers stock phrases that show up often in AI-written text").
  `inky/style.py:3-5` says the list "is a starting point, not a validated detector."
- Why it matters: the README is the first thing a reader sees, and it states an effect that no test or
  measurement supports.
- Fix: write "These controls are intended to add variety. They have not been measured." Keep the caveat
  about detectors. Describe the list as a hand-written starting point, not as a measured pattern.

### DOC-05 (medium): generate ignores the configured EOS token, and the test named for it does not test it
- `inky/modeling.py:344` (`eos_token_id: Optional[int] = None`) and `:378` (EOS stop only when passed).
  `config.eos_token_id` (49974) is never read by `generate`.
- `README.md:123-131` (usage example) does not pass `eos_token_id`. Generation continues for the full
  `max_new_tokens` after `<|endoftext|>`.
- `tests/test_inky.py:140-147` is named `test_generate_appends_tokens_and_stops_at_eos`. It never sets an
  EOS and asserts length 13, so it passes if EOS stopping is broken.
- Fix: default `eos_token_id` to `self.config.eos_token_id`. Rename the test, or make it script the model to
  emit the EOS id and assert that output stops there.

### DOC-06 (medium): scratch reduction silently changes the loop count
- `inky/modeling.py:381-388`: after a reduction, the cache is rebuilt with `InkyCache(cfg)` and
  `self(seq, past=cache, last_only=True)`, without passing `num_loops`.
- Measured: `generate(..., num_loops=2, reduce_scratch_blocks=True)` on the tiny config records loop counts
  2 and then 4 (the model maximum). The text is produced at a different depth after the first reduction.
- `README.md:67-68` says the reduction "rebuilds the cache for the shortened context ... so the cache always
  matches the text the model sees." That is true for the text but not for the depth.
- Fix: pass the loop count from the first cache when rebuilding. Add a test that asserts the loop count is
  unchanged after reduction.

### DATA-04 (medium): bad input passes silently or crashes far from its cause
- `inky/data.py:133` (`open(path, "r", encoding="utf-8")`), `:137` (`json.loads`), `:139` (`yield obj[field]`).
- Measured: a file written with a UTF-8 BOM (the default for some Windows editors, and this project is
  developed on Windows) raises `JSONDecodeError: Unexpected UTF-8 BOM` on line 1. A record `{"text": null}`
  yields `None`, and `clean_text(None)` then fails with a `TypeError` later in the pipeline.
- Empty input: `prepare_corpus([])` and `prepare_corpus(["   "] * 5)` both return a `(0, seq_len)` tensor with
  no warning, the same as a fully filtered corpus. `quality_filter(" " * 300)` returns `True` on its own
  (`inky/data.py:53-66`). The pipeline is safe only because `clean_text` runs first.
- Fix: open with `encoding="utf-8-sig"`. Skip or report records whose field is not a `str`, with the line
  number. Log or raise when zero documents survive a stage. Make `quality_filter` return False when there is
  no non-whitespace text.

### DATA-05 (medium): the EOS tokenizer is not blocked from emitting chat, tool or vision tokens
- `config.py:21` (`vocab_size=50003`), `config.py:68` (`eos_token_id=49974`).
- `README.md:133-136` (tokenizer instructions). `inky/modeling.py:371-372` (no special-token mask in
  `generate`).
- The tokenizer file defines `<|im_start|>`, `<|im_end|>`, `<|user|>`, `<|assistant|>`, `<|system|>`,
  `<tool_call>`, the vision tokens, the fill-in-the-middle tokens and `<|thought|>`. The tokenizer folder also
  ships a `chat_template.jinja` that uses them. The README tells users to add three more scratch tokens, but
  `<|thought|>` already exists and plays a similar role.
- Why it matters: after training, the model can write chat or tool markup into a story. Nothing in the
  code or docs warns about this.
- Fix: in `generate`, block every id from the reserved special range unless it is the EOS or scratch id.
  Document the reserved ids in the README. Explain why `<|thought|>` is not reused for scratch.

### DATA-06 (medium): packed training windows let stories attend to each other
- `inky/data.py:109-124` (documents are joined with EOS and cut into windows).
- `inky/layers.py:130-133` (causal attention over the whole window, with no document mask).
- Nothing in `pack_sequences`, `prepare_corpus` or the README says that attention crosses document
  boundaries.
- Why it matters: for creative writing, one story will learn to continue into the next unrelated story.
  This is a quality issue, not a crash, and it is invisible in the tests.
- Fix: either build a block-diagonal mask from document ids (a small change to the attention mask), or
  document the behaviour and keep documents short. Add a test that tokens after an EOS do not change
  when the previous document changes.

### TEST-03 (low): several tests are too loose to fail
- `tests/test_inky.py:167`: `all(p.grad is None or torch.isfinite(p.grad).all() ...)` passes when no gradient
  exists at all. A broken checkpoint path would still pass. Assert `p.grad is not None` for trainable
  parameters that the loss depends on.
- `tests/test_inky.py:131-137`: the ACT test only checks `1.0 <= mean_loops <= max`. A halting bug that
  made every token take one loop would pass. Assert a change in mean loops when `act_threshold` changes.
- `tests/test_inky.py:150-158`: the name says the bias moves "towards uniform load", but the test only
  checks that some bias is non-zero. Check the sign against the load.
- `tests/test_inky.py:54-57`: the 45M to 55M band is wide enough to hide a 0.4M README mismatch (DOC-02).
  Assert the exact count and keep the expected value in one place.
- No test in `tests/test_data.py` covers empty input, BOM files, `null` text, non-Latin scripts, verse
  layout or the near-duplicate boundary. PLAN.md:141 says 30-word documents with one changed word are kept,
  which the README does not say. Add one test for each.
