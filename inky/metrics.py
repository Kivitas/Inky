"""Reports and displays for Inky's outputs.

Three kinds of output get a report:

  - binary decisions (Noul):   confusion counts, recall, precision, F1, specificity, MCC, AUC,
                               log loss, Brier score, calibration error.
  - choice decisions (Choice): top-k accuracy, MRR, NLL, per-option TP/FP/FN/TN, macro and
                               micro precision, recall and F1, and the K x K confusion matrix.
  - generated text:            distinct-n, repetition rate, type-token ratio, and hits on
                               blocked phrases.

Confusion-based metrics need gold labels. Without labels, a decision still reports its own
confidence, entropy and margin, and the confusion fields show "n/a".

Every metric returns None when its denominator is zero. The display prints that as "n/a"
rather than hiding it.
"""

import math
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from .style import banned_by_bad_words

NA = None


def _div(num: float, den: float) -> Optional[float]:
    return None if den == 0 else num / den


def _to_list(x) -> list:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().reshape(-1).tolist()
    return list(x)


# --------------------------------------------------------------------------- binary decisions

def binary_report(probs, labels=None, threshold: float = 0.5, ece_bins: int = 10) -> Dict[str, object]:
    """Report for yes/no decisions. ``probs`` are P(yes) and ``labels`` are 0/1 gold answers.

    Pass ``labels=None`` to get confidence, entropy and the predicted decision only.
    """
    p = _to_list(probs)
    n = len(p)
    preds = [1 if x >= threshold else 0 for x in p]
    confidence = [max(x, 1.0 - x) for x in p]
    entropy = [-(x * math.log(x + 1e-12) + (1 - x) * math.log(1 - x + 1e-12)) / math.log(2) for x in p]
    report: Dict[str, object] = {
        "kind": "binary",
        "items": n,
        "threshold": threshold,
        "predicted_yes_rate": _div(sum(preds), n),
        "mean_confidence": _div(sum(confidence), n),
        "mean_entropy_bits": _div(sum(entropy), n),
        "decisions": preds,
    }
    if labels is None:
        for key in ("tp", "tn", "fp", "fn", "accuracy", "precision", "recall", "specificity",
                    "npv", "fpr", "fnr", "f1", "f2", "f0_5", "balanced_accuracy", "mcc",
                    "youden_j", "prevalence", "roc_auc", "log_loss", "brier", "ece"):
            report[key] = NA
        report["labels"] = "not given: confusion metrics need gold labels"
        return report

    y = [int(v) for v in _to_list(labels)]
    if len(y) != n:
        raise ValueError(f"got {n} probabilities but {len(y)} labels")
    tp = sum(1 for a, b in zip(preds, y) if a == 1 and b == 1)
    tn = sum(1 for a, b in zip(preds, y) if a == 0 and b == 0)
    fp = sum(1 for a, b in zip(preds, y) if a == 1 and b == 0)
    fn = sum(1 for a, b in zip(preds, y) if a == 0 and b == 1)
    precision = _div(tp, tp + fp)
    recall = _div(tp, tp + fn)
    specificity = _div(tn, tn + fp)
    npv = _div(tn, tn + fn)

    def f_beta(beta: float) -> Optional[float]:
        if precision is None or recall is None:
            return None
        den = beta * beta * precision + recall
        return None if den == 0 else (1 + beta * beta) * precision * recall / den

    mcc_den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    recalls = [v for v in (recall, specificity) if v is not None]
    report.update({
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": _div(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "npv": npv,
        "fpr": _div(fp, fp + tn),
        "fnr": _div(fn, fn + tp),
        "f1": f_beta(1.0),
        "f2": f_beta(2.0),
        "f0_5": f_beta(0.5),
        "balanced_accuracy": sum(recalls) / len(recalls) if recalls else None,
        "mcc": None if mcc_den == 0 else (tp * tn - fp * fn) / mcc_den,
        "youden_j": None if recall is None or specificity is None else recall + specificity - 1,
        "prevalence": _div(sum(y), n),
        "roc_auc": roc_auc(p, y),
        "log_loss": _div(-sum(math.log(max(x if t else 1 - x, 1e-12)) for x, t in zip(p, y)), n),
        "brier": _div(sum((x - t) ** 2 for x, t in zip(p, y)), n),
        "ece": expected_calibration_error(confidence, [int(a == b) for a, b in zip(preds, y)], ece_bins),
        "labels": "given",
    })
    report["confusion"] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
    return report


def roc_auc(scores: Sequence[float], labels: Sequence[int]) -> Optional[float]:
    """Area under the ROC curve via the rank statistic (ties count half). None if one class only."""
    pos = [s for s, t in zip(scores, labels) if t == 1]
    neg = [s for s, t in zip(scores, labels) if t == 0]
    if not pos or not neg:
        return None
    wins = 0.0
    for a in pos:
        for b in neg:
            wins += 1.0 if a > b else 0.5 if a == b else 0.0
    return wins / (len(pos) * len(neg))


def expected_calibration_error(confidence: Sequence[float], correct: Sequence[int], bins: int = 10) -> Optional[float]:
    """Mean gap between confidence and accuracy, weighted by bin size."""
    n = len(confidence)
    if n == 0:
        return None
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidence) if (lo < c <= hi) or (b == 0 and c == 0.0)]
        if not idx:
            continue
        acc = sum(correct[i] for i in idx) / len(idx)
        conf = sum(confidence[i] for i in idx) / len(idx)
        total += len(idx) / n * abs(acc - conf)
    return total


# --------------------------------------------------------------------------- choice decisions

def choice_report(probs, gold=None, top_k: Sequence[int] = (2, 3)) -> Dict[str, object]:
    """Report for N-way choices. ``probs`` is [items, options]. ``gold`` holds the correct index per item."""
    P = probs.detach().cpu() if isinstance(probs, torch.Tensor) else torch.as_tensor(probs, dtype=torch.float32)
    if P.ndim != 2:
        raise ValueError("probs must be [items, options]")
    n, K = P.shape
    pred = P.argmax(dim=-1).tolist()
    report: Dict[str, object] = {
        "kind": "choice",
        "items": n,
        "options": K,
        "predicted": pred,
        "mean_top_probability": float(P.max(dim=-1).values.mean()) if n else None,
    }
    if gold is None:
        report["labels"] = "not given: accuracy and per-option metrics need gold labels"
        return report

    g = [int(v) for v in _to_list(gold)]
    if len(g) != n:
        raise ValueError(f"got {n} items but {len(g)} gold labels")
    ranks = []
    for i, gi in enumerate(g):
        order = torch.argsort(P[i], descending=True).tolist()
        ranks.append(order.index(gi) + 1)
    correct = sum(1 for a, b in zip(pred, g) if a == b)
    report["top1_accuracy"] = _div(correct, n)
    for k in top_k:
        if k <= K:
            report[f"top{k}_accuracy"] = _div(sum(1 for r in ranks if r <= k), n)
    report["mrr"] = _div(sum(1.0 / r for r in ranks), n)
    report["nll"] = _div(-sum(math.log(max(float(P[i, g[i]]), 1e-12)) for i in range(n)), n)
    report["mean_gold_probability"] = _div(sum(float(P[i, g[i]]) for i in range(n)), n)
    onehot = torch.zeros_like(P)
    onehot[torch.arange(n), torch.tensor(g)] = 1.0
    report["brier"] = _div(float(((P - onehot) ** 2).sum(dim=-1).sum()), n)

    matrix = [[0] * K for _ in range(K)]
    for a, b in zip(g, pred):
        matrix[a][b] += 1
    per_option = []
    precisions, recalls = [], []
    for k in range(K):
        tp = matrix[k][k]
        fn = sum(matrix[k]) - tp
        fp = sum(matrix[r][k] for r in range(K)) - tp
        tn = n - tp - fp - fn
        p_k, r_k = _div(tp, tp + fp), _div(tp, tp + fn)
        f1_k = None if p_k is None or r_k is None or p_k + r_k == 0 else 2 * p_k * r_k / (p_k + r_k)
        per_option.append({"option": k, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                           "precision": p_k, "recall": r_k, "f1": f1_k})
        if p_k is not None:
            precisions.append(p_k)
        if r_k is not None:
            recalls.append(r_k)
    macro_p = sum(precisions) / len(precisions) if precisions else None
    macro_r = sum(recalls) / len(recalls) if recalls else None
    f1s = [o["f1"] for o in per_option if o["f1"] is not None]
    report.update({
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "micro_precision": _div(correct, n),   # micro TP = FP = FN = errors, so micro P = R = F1 = accuracy
        "micro_recall": _div(correct, n),
        "micro_f1": _div(correct, n),
        "per_option": per_option,
        "confusion_matrix": matrix,
        "labels": "given",
    })
    return report


# --------------------------------------------------------------------------- generated text

def text_report(
    token_ids,
    bad_word_ids: Optional[Sequence[Sequence[int]]] = None,
) -> Dict[str, object]:
    """Indicators for a generated sequence of token ids (the new tokens only, not the prompt)."""
    ids = [int(x) for x in _to_list(token_ids)]
    n = len(ids)
    report: Dict[str, object] = {"kind": "text", "tokens": n}
    for k in (1, 2, 3):
        grams = [tuple(ids[i:i + k]) for i in range(n - k + 1)]
        report[f"distinct_{k}"] = _div(len(set(grams)), len(grams))
    four = [tuple(ids[i:i + 4]) for i in range(n - 3)]
    report["repeated_4gram_rate"] = None if not four else 1.0 - len(set(four)) / len(four)
    report["type_token_ratio"] = _div(len(set(ids)), n)
    top = Counter(ids).most_common(1)
    report["most_frequent_token_share"] = _div(top[0][1], n) if top else None

    hits = 0
    if bad_word_ids:
        for i in range(n):
            if ids[i] in banned_by_bad_words(ids[:i], bad_word_ids):
                hits += 1
    report["blocked_phrase_hits"] = hits if bad_word_ids else None
    return report


# --------------------------------------------------------------------------- display

_LABELS = {
    "tp": "true positives", "tn": "true negatives", "fp": "false positives", "fn": "false negatives",
    "accuracy": "accuracy", "precision": "precision", "recall": "recall (sensitivity)",
    "specificity": "specificity", "npv": "negative predictive value", "fpr": "false positive rate",
    "fnr": "false negative rate", "f1": "F1", "f2": "F2", "f0_5": "F0.5",
    "balanced_accuracy": "balanced accuracy", "mcc": "Matthews correlation", "youden_j": "Youden's J",
    "prevalence": "prevalence (yes in gold)", "predicted_yes_rate": "predicted yes rate",
    "roc_auc": "ROC AUC", "log_loss": "log loss", "brier": "Brier score", "ece": "calibration error (ECE)",
    "mean_confidence": "mean confidence", "mean_entropy_bits": "mean entropy (bits)",
    "top1_accuracy": "top-1 accuracy", "top2_accuracy": "top-2 accuracy", "top3_accuracy": "top-3 accuracy",
    "mrr": "mean reciprocal rank", "nll": "negative log likelihood",
    "mean_gold_probability": "mean probability of gold option", "mean_top_probability": "mean top probability",
    "macro_precision": "macro precision", "macro_recall": "macro recall", "macro_f1": "macro F1",
    "micro_precision": "micro precision", "micro_recall": "micro recall", "micro_f1": "micro F1",
    "tokens": "tokens", "distinct_1": "distinct-1", "distinct_2": "distinct-2", "distinct_3": "distinct-3",
    "repeated_4gram_rate": "repeated 4-gram rate", "type_token_ratio": "type-token ratio",
    "most_frequent_token_share": "most frequent token share", "blocked_phrase_hits": "blocked phrase hits",
}


def _fmt(value) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def format_report(report: Dict[str, object]) -> str:
    """Plain-text display of any report from this module."""
    kind = report.get("kind", "report")
    lines = [f"== {kind} report =="]
    if "labels" in report and isinstance(report["labels"], str) and report["labels"] != "given":
        lines.append(f"note: {report['labels']}")

    skip = {"kind", "decisions", "predicted", "per_option", "confusion_matrix", "confusion", "labels", "threshold"}
    rows = [(k, v) for k, v in report.items() if k not in skip and k in _LABELS]
    width = max((len(_LABELS[k]) for k, _ in rows), default=10)
    for key, value in rows:
        lines.append(f"  {_LABELS[key]:<{width}}  {_fmt(value)}")

    if kind == "binary" and report.get("labels") == "given":
        c = report["confusion"]
        lines += ["", "  confusion matrix          gold yes   gold no",
                  f"  predicted yes             {c['tp']:>8}   {c['fp']:>7}",
                  f"  predicted no              {c['fn']:>8}   {c['tn']:>7}"]
    if kind == "choice" and report.get("labels") == "given":
        lines += ["", "  per option     TP     FP     FN     TN   precision   recall      F1"]
        for o in report["per_option"]:
            lines.append(f"  option {o['option']:<4}{o['tp']:>6} {o['fp']:>6} {o['fn']:>6} {o['tn']:>6}"
                         f"   {_fmt(o['precision']):>9}  {_fmt(o['recall']):>7}  {_fmt(o['f1']):>7}")
        lines += ["", "  confusion matrix (rows = gold, columns = predicted)"]
        for r, row in enumerate(report["confusion_matrix"]):
            lines.append(f"  gold {r:<3}" + "".join(f"{v:>6}" for v in row))
    return "\n".join(lines)


def print_report(report: Dict[str, object]) -> None:
    print(format_report(report))
