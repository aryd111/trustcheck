"""Measure the model, including whether its confidence can be trusted.

Accuracy alone does not tell you when to trust a model. So we also check:
when the model says it is confident, is it actually right more often?
If yes, the "not sure, ask a human" flag is meaningful.
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score


def evaluate(checker, texts, labels, threshold=None):
    if threshold is None:
        threshold = checker.threshold
    labels = np.array(labels)
    p_spam = checker.spam_probability(texts)
    preds = (p_spam >= 0.5).astype(int)
    confidence = np.where(preds == 1, p_spam, 1 - p_spam)
    confident = confidence >= threshold

    report = {
        "accuracy": accuracy_score(labels, preds),
        "spam_precision": precision_score(labels, preds, zero_division=0),
        "spam_recall": recall_score(labels, preds, zero_division=0),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1]).tolist(),
        "threshold": threshold,
        "share_confident": float(confident.mean()),
        "accuracy_when_confident": _acc(labels[confident], preds[confident]),
        "accuracy_when_unsure": _acc(labels[~confident], preds[~confident]),
    }

    # Calibration table: group predictions by confidence, compare to reality.
    bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    table = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence >= lo) & (confidence < hi)
        if mask.any():
            table.append({
                "range": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                "count": int(mask.sum()),
                "avg_confidence": float(confidence[mask].mean()),
                "actual_accuracy": float((preds[mask] == labels[mask]).mean()),
            })
    report["calibration"] = table
    return report


def _acc(y_true, y_pred):
    return float((y_true == y_pred).mean()) if len(y_true) else float("nan")


def print_report(report):
    tn, fp = report["confusion_matrix"][0]
    fn, tp = report["confusion_matrix"][1]
    print(f"Accuracy:          {report['accuracy']:.1%}")
    print(f"Spam precision:    {report['spam_precision']:.1%}  (of messages flagged spam, how many really were)")
    print(f"Spam recall:       {report['spam_recall']:.1%}  (of real spam, how much we caught)")
    print(f"Mistakes:          {fp} normal messages wrongly flagged, {fn} spam messages missed")
    print()
    print(f"Trust check (threshold {report['threshold']:.0%}):")
    print(f"  Model was confident on {report['share_confident']:.1%} of messages")
    print(f"  Accuracy when confident: {report['accuracy_when_confident']:.1%}")
    print(f"  Accuracy when unsure:    {report['accuracy_when_unsure']:.1%}")
    print()
    print("Calibration (does the model's confidence match reality?)")
    print(f"  {'confidence':<12}{'messages':>10}{'avg conf':>10}{'actual acc':>12}")
    for row in report["calibration"]:
        print(f"  {row['range']:<12}{row['count']:>10}{row['avg_confidence']:>10.1%}{row['actual_accuracy']:>12.1%}")
