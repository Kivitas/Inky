"""Tests for inky/metrics.py against hand-computed values. Run with: python -m pytest tests -q"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inky import InkyConfig, InkyModel  # noqa: E402
from inky.metrics import binary_report, choice_report, format_report, text_report  # noqa: E402

OPEN, SEP, CLOSE = 508, 509, 510


def approx(a, b, tol=1e-4):
    return a is not None and abs(a - b) < tol


def test_binary_confusion_and_rates_match_hand_computation():
    # threshold 0.5 -> predictions [1, 1, 0, 0, 1]; gold [1, 0, 0, 1, 1]
    r = binary_report([0.9, 0.8, 0.3, 0.2, 0.6], [1, 0, 0, 1, 1])
    assert (r["tp"], r["fp"], r["tn"], r["fn"]) == (2, 1, 1, 1)
    assert approx(r["accuracy"], 0.6)
    assert approx(r["precision"], 2 / 3)
    assert approx(r["recall"], 2 / 3)
    assert approx(r["specificity"], 0.5)
    assert approx(r["f1"], 2 / 3)
    assert approx(r["fpr"], 0.5)
    assert approx(r["fnr"], 1 / 3)
    assert approx(r["mcc"], 1 / 6)
    assert approx(r["balanced_accuracy"], (2 / 3 + 0.5) / 2)
    assert approx(r["prevalence"], 0.6)
    # pairs (positive, negative): 0.9 beats both negatives, 0.6 beats 0.3, 0.2 beats neither -> 3 of 6
    assert approx(r["roc_auc"], 3 / 6)


def test_binary_without_labels_shows_confidence_but_no_confusion():
    r = binary_report([0.9, 0.1])
    assert r["tp"] is None and r["recall"] is None and r["f1"] is None
    assert approx(r["mean_confidence"], 0.9)
    assert r["predicted_yes_rate"] == 0.5


def test_binary_no_positives_gives_none_not_a_crash():
    r = binary_report([0.1, 0.2], [0, 0])
    assert r["recall"] is None           # no gold positives
    assert approx(r["specificity"], 1.0)


def test_choice_accuracy_mrr_nll_and_macro_scores():
    probs = [[0.7, 0.2, 0.1], [0.1, 0.3, 0.6], [0.2, 0.5, 0.3]]
    r = choice_report(probs, [0, 2, 0])
    assert approx(r["top1_accuracy"], 2 / 3)
    assert approx(r["top2_accuracy"], 2 / 3)
    assert approx(r["top3_accuracy"], 1.0)
    assert approx(r["mrr"], (1 + 1 + 1 / 3) / 3)
    assert approx(r["nll"], -(math.log(0.7) + math.log(0.6) + math.log(0.2)) / 3)
    assert approx(r["brier"], (0.14 + 0.26 + 0.98) / 3)
    assert r["confusion_matrix"] == [[1, 1, 0], [0, 0, 0], [0, 0, 1]]
    opt0, opt1, opt2 = r["per_option"]
    assert (opt0["tp"], opt0["fp"], opt0["fn"], opt0["tn"]) == (1, 0, 1, 1)
    assert approx(opt0["recall"], 0.5) and approx(opt0["f1"], 2 / 3)
    assert opt1["recall"] is None and opt1["precision"] == 0.0   # option 1 never gold
    assert approx(opt2["f1"], 1.0)
    assert approx(r["macro_precision"], 2 / 3)
    assert approx(r["macro_recall"], 0.75)
    assert approx(r["micro_f1"], 2 / 3)


def test_choice_macro_scores_keep_classes_with_gold_support():
    # gold [0, 2], predictions [0, 1] over three options: option 2 has gold but is never predicted
    r = choice_report([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1]], [0, 2])
    assert approx(r["top1_accuracy"], 0.5)
    assert r["macro_f1"] < 1.0
    assert approx(r["macro_f1"], 0.5)           # options 0 (F1 1) and 2 (F1 0); option 1 is predicted only
    assert approx(r["macro_recall"], 0.5)       # options 0 and 2 have gold
    assert approx(r["macro_precision"], 1 / 3)  # option 1 (predicted only) and option 2 count as 0
    assert r["per_option"][2]["f1"] == 0.0


def test_f_beta_is_zero_when_precision_and_recall_are_both_zero():
    r = binary_report([0.7, 0.6, 0.2], [0, 0, 1])
    assert r["precision"] == 0.0 and r["recall"] == 0.0
    assert r["f1"] == 0.0 and r["f2"] == 0.0 and r["f0_5"] == 0.0
    # no predicted positives: precision is undefined, so F-beta stays None
    r = binary_report([0.1, 0.2], [1, 0])
    assert r["precision"] is None and r["recall"] == 0.0 and r["f1"] is None


def test_binary_confidence_follows_the_decision_threshold():
    r = binary_report([0.6, 0.6, 0.9, 0.1], [1, 0, 1, 0], threshold=0.8)
    assert r["decisions"] == [0, 0, 1, 0]
    assert approx(r["mean_confidence"], (0.4 + 0.4 + 0.9 + 0.9) / 4)


def test_balanced_accuracy_needs_both_recall_and_specificity():
    r = binary_report([0.1, 0.2, 0.3], [0, 0, 0])   # no positives: recall undefined
    assert r["balanced_accuracy"] is None


def test_choice_without_labels_reports_confidence_only():
    r = choice_report([[0.6, 0.4], [0.2, 0.8]], None)
    assert r["predicted"] == [0, 1]
    assert approx(r["mean_top_probability"], 0.7)
    assert "top1_accuracy" not in r


def test_text_indicators_match_hand_computation():
    r = text_report([5, 6, 7, 5, 6, 7, 8])
    assert approx(r["distinct_1"], 4 / 7)
    assert approx(r["distinct_2"], 4 / 6)
    assert approx(r["distinct_3"], 4 / 5)
    assert r["repeated_4gram_rate"] == 0.0
    assert approx(r["type_token_ratio"], 4 / 7)
    assert r["blocked_phrase_hits"] is None
    r = text_report([5, 6, 7, 5, 6, 7, 8], bad_word_ids=[[7, 8]])
    assert r["blocked_phrase_hits"] == 1


def test_format_report_displays_the_named_indicators():
    text = format_report(binary_report([0.9, 0.8, 0.3, 0.2, 0.6], [1, 0, 0, 1, 1]))
    for word in ("true positives", "true negatives", "false positives", "false negatives",
                 "recall", "F1", "confusion matrix", "Matthews"):
        assert word in text, word
    text = format_report(choice_report([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]], [0, 2]))
    assert "per option" in text and "macro F1" in text


def tiny_model() -> InkyModel:
    torch.manual_seed(0)
    config = InkyConfig(
        vocab_size=512, scratch_open_id=OPEN, scratch_sep_id=SEP, scratch_close_id=CLOSE,
        hidden_size=64, num_attention_heads=4, qk_nope_head_dim=12, qk_rope_head_dim=4, v_head_dim=16,
        q_lora_rank=32, kv_lora_rank=24, intermediate_size=96, num_routed_experts=4, num_active_routed=2,
        ssm_state_dim=16, max_recurrent_loops=4, min_recurrent_loops=2, loop_dropout_prob=0.0,
        max_position_embeddings=256, arlo_dim=16, arlo_pointer_rank=8, arlo_slots=4,
    )
    return InkyModel(config).eval()


def test_decision_outputs_carry_reports_when_asked():
    model = tiny_model()
    ctx = torch.randint(2, 500, (1, 8))
    p, rep = model.decide_binary(ctx, gold=1, report=True)
    assert rep["kind"] == "binary" and rep["items"] == 1
    assert rep["tp"] + rep["fn"] == 1          # the single gold positive is either TP or FN
    opts = [torch.randint(2, 500, (1, 2)), torch.randint(2, 500, (1, 3))]
    probs, rep = model.decide_choice(ctx, opts, gold=0, report=True)
    assert rep["kind"] == "choice" and rep["options"] == 2
    assert approx(float(probs.sum()), 1.0, 1e-5)


def test_generation_report_is_returned_and_excludes_prompt():
    model = tiny_model()
    prompt = torch.tensor([[2, 3, 4]])
    seq, rep = model.generate(prompt, max_new_tokens=6, do_sample=False, repetition_penalty=1.0,
                              no_repeat_ngram_size=0, report=True)
    assert rep["kind"] == "text" and rep["tokens"] == seq.shape[1] - 3


if __name__ == "__main__":
    tests = [value for name, value in list(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"{len(tests)} tests passed")
