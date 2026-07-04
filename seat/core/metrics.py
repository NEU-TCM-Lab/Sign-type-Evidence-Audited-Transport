from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_recall_fscore_support, roc_auc_score

from common import LABELS, assert_label_order


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    logits = logits.astype(np.float64)
    return 1.0 / (1.0 + np.exp(-logits))


def _validate_arrays(y_true: np.ndarray, y_prob: np.ndarray, label_order: list[str]) -> None:
    assert_label_order(label_order)
    if y_true.shape != y_prob.shape:
        raise AssertionError(f"y_true/y_prob shape mismatch: {y_true.shape} vs {y_prob.shape}")
    if y_true.ndim != 2 or y_true.shape[1] != len(LABELS):
        raise AssertionError(f"Expected [N,{len(LABELS)}] arrays, got {y_true.shape}")


def find_best_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    label_order: list[str] = LABELS,
    low: float = 0.05,
    high: float = 0.95,
    step: float = 0.005,
) -> dict[str, float]:
    """Per-class threshold that maximises that class's binary F1. Default grid keeps the
    historical 181-point (step 0.005) sweep; pass step=0.01 for the spec's 0.05..0.95/0.01 grid."""
    _validate_arrays(y_true, y_prob, label_order)
    thresholds: dict[str, float] = {}
    n_pts = int(round((high - low) / step)) + 1
    candidates = np.unique(np.concatenate([np.linspace(low, high, n_pts), np.array([0.5])]))
    for i, label in enumerate(label_order):
        if len(np.unique(y_true[:, i])) < 2:
            thresholds[label] = 0.5
            continue
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in candidates:
            pred = (y_prob[:, i] >= threshold).astype(np.int64)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true[:, i].astype(np.int64),
                pred,
                average="binary",
                zero_division=0,
            )
            if f1 > best_f1:
                best_f1 = float(f1)
                best_threshold = float(threshold)
        thresholds[label] = best_threshold
    return thresholds


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: dict[str, float] | None = None,
    label_order: list[str] = LABELS,
) -> dict:
    _validate_arrays(y_true, y_prob, label_order)
    if thresholds is None:
        thresholds = {label: 0.5 for label in label_order}
    per_label = {}
    values_by_metric: dict[str, list[float]] = {
        "auroc": [],
        "ap": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }
    valid_cols = []
    y_pred = np.zeros_like(y_true, dtype=np.int64)
    for i, label in enumerate(label_order):
        threshold = float(thresholds[label])
        y_pred[:, i] = (y_prob[:, i] >= threshold).astype(np.int64)
        unique = np.unique(y_true[:, i])
        if len(unique) < 2:
            row = {
                "threshold": threshold,
                "single_class": True,
                "support_pos": int(y_true[:, i].sum()),
                "support_neg": int((1 - y_true[:, i]).sum()),
                "auroc": float("nan"),
                "ap": float("nan"),
                "f1": float("nan"),
                "precision": float("nan"),
                "recall": float("nan"),
            }
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true[:, i].astype(np.int64),
                y_pred[:, i],
                average="binary",
                zero_division=0,
            )
            row = {
                "threshold": threshold,
                "single_class": False,
                "support_pos": int(y_true[:, i].sum()),
                "support_neg": int((1 - y_true[:, i]).sum()),
                "auroc": float(roc_auc_score(y_true[:, i], y_prob[:, i])),
                "ap": float(average_precision_score(y_true[:, i], y_prob[:, i])),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
            valid_cols.append(i)
        per_label[label] = row
        for metric in values_by_metric:
            values_by_metric[metric].append(row[metric])

    macro = {}
    for metric, values in values_by_metric.items():
        with np.errstate(all="ignore"):
            macro[metric] = float(np.nanmean(np.asarray(values, dtype=np.float64)))

    if valid_cols:
        yt = y_true[:, valid_cols].reshape(-1).astype(np.int64)
        yp = y_pred[:, valid_cols].reshape(-1).astype(np.int64)
        yscore = y_prob[:, valid_cols].reshape(-1)
        precision, recall, f1, _ = precision_recall_fscore_support(yt, yp, average="binary", zero_division=0)
        micro = {
            "auroc": float(roc_auc_score(yt, yscore)) if len(np.unique(yt)) > 1 else float("nan"),
            "ap": float(average_precision_score(yt, yscore)) if len(np.unique(yt)) > 1 else float("nan"),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }
    else:
        micro = {metric: float("nan") for metric in values_by_metric}

    # weighted-F1: per-class F1 weighted by positive support over valid (>=2-class) columns.
    if valid_cols:
        f1s = np.asarray([per_label[label_order[i]]["f1"] for i in valid_cols], dtype=np.float64)
        supp = np.asarray([per_label[label_order[i]]["support_pos"] for i in valid_cols], dtype=np.float64)
        weighted_f1 = float(np.average(f1s, weights=supp)) if supp.sum() > 0 else float(np.mean(f1s))
    else:
        weighted_f1 = float("nan")

    # sample-averaged F1: F1 over the label set of each sample, averaged over samples.
    sample_f1 = float(f1_score(y_true.astype(np.int64), y_pred, average="samples", zero_division=0))

    return {
        "label_order": label_order,
        "per_label": per_label,
        "macro": macro,
        "micro": micro,
        "weighted_f1": weighted_f1,
        "sample_f1": sample_f1,
        "num_samples": int(y_true.shape[0]),
        "num_valid_metric_labels": int(len(valid_cols)),
    }


def per_class_rows(experiment_name: str, metrics: dict) -> list[dict]:
    """Flatten compute_metrics() per_label into CSV rows:
    experiment_name,class_name,support_pos,support_neg,threshold,precision,recall,f1,auroc,ap."""
    rows = []
    for label in metrics["label_order"]:
        r = metrics["per_label"][label]
        rows.append({
            "experiment_name": experiment_name,
            "class_name": label,
            "support_pos": r["support_pos"],
            "support_neg": r["support_neg"],
            "threshold": round(float(r["threshold"]), 4),
            "precision": _fmt(r["precision"]),
            "recall": _fmt(r["recall"]),
            "f1": _fmt(r["f1"]),
            "auroc": _fmt(r["auroc"]),
            "ap": _fmt(r["ap"]),
        })
    return rows


def _fmt(v) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if np.isnan(f):
        return ""
    return f"{f:.4f}"
