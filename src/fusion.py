"""Fit fusion parameters on validation predictions, then evaluate held-out test data."""
import numpy as np
from settings import LABELS
from data_validation import alignment_order, validate_split_payloads


def sigmoid(x):
    x = np.clip(np.asarray(x, dtype=float), -700, 700)
    return 1 / (1 + np.exp(-x))


def binary_f1(y, pred):
    y, pred = np.asarray(y).astype(bool), np.asarray(pred).astype(bool)
    tp = np.sum(y & pred); fp = np.sum(~y & pred); fn = np.sum(y & ~pred)
    return float(2 * tp / max(2 * tp + fp + fn, 1))


def best_threshold(y, prob, step=0.01):
    best = (0.5, -1.0)
    for threshold in np.unique(np.r_[np.arange(0.05, 0.950001, step), 0.5]):
        score = binary_f1(y, prob >= threshold)
        if score > best[1]: best = (float(threshold), score)
    return best


def validate_prediction_dump(dump):
    payloads = {split: {"split": split, "ids": dump[f"{split}_ids"],
                        "labels": dump[f"{split}_labels"], "label_order": dump["label_order"]}
                for split in ("train", "val", "test") if f"{split}_ids" in dump}
    if not {"val", "test"} <= payloads.keys():
        raise ValueError("Prediction dump must contain validation and test identities/labels")
    validate_split_payloads(payloads)
    for split, payload in payloads.items():
        if dump[f"{split}_logits"].shape != payload["labels"].shape:
            raise ValueError(f"{split}: prediction/label shape mismatch")


def aligned_logits(reference, other, split):
    return other[f"{split}_logits"][alignment_order(reference, other, id_key=f"{split}_ids", label_key=f"{split}_labels")]


def select_fusion(val_logits, add_val_logits, val_labels, classes=None,
                  gammas=(0.0, 0.2, 0.5, 1.0, 1.5, 2.0), margin=0.0, step=0.01):
    if classes is not None and set(classes) - set(LABELS):
        raise ValueError("Unknown fusion class")
    if 0.0 not in gammas:
        raise ValueError("Fusion candidates must include the no-fusion baseline")
    selected = []
    for c, label in enumerate(LABELS):
        threshold, baseline_score = best_threshold(val_labels[:, c], sigmoid(val_logits[:, c]), step)
        best = {"label": label, "gamma": 0.0, "threshold": threshold, "val_f1": baseline_score}
        if classes is None or label in classes:
            for gamma in gammas:
                threshold, score = best_threshold(val_labels[:, c], sigmoid(val_logits[:, c] + gamma * add_val_logits[:, c]), step)
                if score > best["val_f1"] and score > baseline_score + margin / 100:
                    best = {"label": label, "gamma": float(gamma), "threshold": threshold, "val_f1": score}
        selected.append(best)
    return selected


def evaluate_selected(test_logits, add_test_logits, test_labels, selected):
    return [binary_f1(test_labels[:, c], sigmoid(test_logits[:, c] + cfg["gamma"] * add_test_logits[:, c]) >= cfg["threshold"])
            for c, cfg in enumerate(selected)]
