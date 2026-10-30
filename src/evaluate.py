import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve, precision_score, recall_score, f1_score)


def threshold_for_recall(y_true, scores, target_recall: float = 0.9) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    ok = np.where(recall[:-1] >= target_recall)[0]

    return float(thresholds[ok[-1]]) if len(ok) else float(thresholds[0])


def precision_at_recall(y_true, scores, target_recall: float = 0.9) -> float:
    precision, recall, _ = precision_recall_curve(y_true, scores)
    mask = recall >= target_recall

    return float(precision[mask].max()) if mask.any() else float("nan")


def precision_at_base_rate(recall: float, fpr: float, base_rate: float) -> float:
    tp = recall * base_rate
    fp = fpr * (1 - base_rate)

    return tp / (tp + fp) if (tp + fp) > 0 else float("nan")


def evaluate(y_true, scores, threshold: float, base_rates=(0.01, 0.05)) -> dict:
    y_true = np.asarray(y_true)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)

    result = {
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "average_precision": average_precision_score(y_true, scores),
        "precision_at_recall_0.9": precision_at_recall(y_true, scores, 0.9),
        "false_positive_rate": fpr,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }

    for rate in base_rates:
        result[f"precision_at_{rate:.0%}_phishing"] = precision_at_base_rate(recall, fpr, rate)

    return result
