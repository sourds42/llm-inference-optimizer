"""
Precision/recall/F1/severity-accuracy/etc. over a list of
{"ground_truth": {...}, "prediction": {...}} pairs (see review_task.py).
Pure Python, no GPU -- fully unit-testable.

"Completeness" is defined honestly as the fraction of items that produced
a well-formed, parseable has_bug verdict -- a robustness proxy, not a
multi-issue-coverage metric, since every item in this dataset carries at
most one issue (see src/review_eval/dataset.py).
"""
from __future__ import annotations


def score_reviews(per_item: list) -> dict:
    n = len(per_item)
    well_formed = [r for r in per_item if r["prediction"].get("has_bug") is not None]
    completeness = round(len(well_formed) / n, 3) if n else 0.0

    tp = fp = fn = tn = 0
    severity_correct = severity_total = 0
    category_correct = category_total = 0
    critical_gt = critical_hits = 0

    for r in well_formed:
        gt, pred = r["ground_truth"], r["prediction"]
        gt_bug, pred_bug = gt["has_bug"], pred["has_bug"]

        if gt_bug and pred_bug:
            tp += 1
        elif gt_bug and not pred_bug:
            fn += 1
        elif not gt_bug and pred_bug:
            fp += 1
        else:
            tn += 1

        if gt_bug and pred_bug:
            severity_total += 1
            if pred.get("severity") == gt.get("severity"):
                severity_correct += 1
            category_total += 1
            if pred.get("category") == gt.get("category"):
                category_correct += 1

        if gt_bug and gt.get("severity") == "critical":
            critical_gt += 1
            if pred_bug:
                critical_hits += 1

    precision = round(tp / (tp + fp), 3) if (tp + fp) else 0.0
    recall = round(tp / (tp + fn), 3) if (tp + fn) else 0.0
    f1 = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) else 0.0
    false_positive_rate = round(fp / (fp + tn), 3) if (fp + tn) else 0.0

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "false_positive_rate": false_positive_rate,
        "severity_accuracy": round(severity_correct / severity_total, 3) if severity_total else None,
        "category_accuracy": round(category_correct / category_total, 3) if category_total else None,
        "critical_recall": round(critical_hits / critical_gt, 3) if critical_gt else None,
        "completeness": completeness,
        "n_items": n, "n_well_formed": len(well_formed),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }
