"""Fit a corrector on training predictions; select using validation only."""
import argparse
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score
from settings import LABELS
from data_validation import validate_split_payloads


class Corrector(nn.Module):
    def __init__(self, W_init=None):
        super().__init__()
        self.a = nn.Parameter(torch.ones(len(LABELS)))
        self.b = nn.Parameter(torch.zeros(len(LABELS)))
        self.W = nn.Parameter(torch.zeros(len(LABELS), len(LABELS)) if W_init is None else torch.tensor(W_init, dtype=torch.float32))

    def forward(self, probs):
        p = probs.clamp(1e-4, 1 - 1e-4)
        return self.a * torch.log(p / (1 - p)) + self.b + (p - 0.5) @ self.W.t()


def tune_thr(labels, probs):
    from fusion import best_threshold
    return np.array([best_threshold(labels[:, c], probs[:, c])[0] for c in range(len(LABELS))])


def macro(labels, probs, thresholds):
    return float(np.mean([f1_score(labels[:, c], probs[:, c] >= thresholds[c], zero_division=0) for c in range(len(LABELS))]))


def select_corrector(train_probs, train_labels, val_probs, val_labels, epochs=400, seed=42):
    y = np.asarray(train_labels, dtype=float)
    intersection = y.T @ y
    union = y.sum(0)[:, None] + y.sum(0)[None, :] - intersection
    coocc = intersection / np.maximum(union, 1)
    thresholds = tune_thr(val_labels, val_probs)
    best = {"name": "independent", "val_f1": macro(val_labels, val_probs, thresholds), "thresholds": thresholds, "model": None}
    candidates = [("learned_scratch", None, 1e-3), ("learned_jaccard_init", coocc, 1e-3), ("learned_strong_reg", None, 1e-2), ("learned_weak_reg", None, 1e-4)]
    for name, initial, decay in candidates:
        torch.manual_seed(seed)
        model = Corrector(initial)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=decay)
        x = torch.tensor(train_probs, dtype=torch.float32)
        target = torch.tensor(train_labels, dtype=torch.float32)
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(x), target)
            loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad(): probs = torch.sigmoid(model(torch.tensor(val_probs, dtype=torch.float32))).numpy()
        thresholds = tune_thr(val_labels, probs)
        score = macro(val_labels, probs, thresholds)
        if score > best["val_f1"]:
            best = {"name": name, "val_f1": score, "thresholds": thresholds, "model": model}
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npz", required=True, help="Ptr/Ytr/Pv/Yv/Pt/Yt, train/val/test_ids and label_order")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    with np.load(args.npz, allow_pickle=False) as archive: d = dict(archive)
    needed = {"Ptr", "Ytr", "Pv", "Yv", "Pt", "Yt", "train_ids", "val_ids", "test_ids", "label_order"}
    if needed - d.keys(): raise ValueError(f"Training predictions and split identities are required: {sorted(needed - d.keys())}")
    validate_split_payloads({s: {"split": s, "ids": d[f"{s}_ids"], "labels": d[key], "label_order": d["label_order"]} for s, key in (("train", "Ytr"), ("val", "Yv"), ("test", "Yt"))})
    for probs, labels in (("Ptr", "Ytr"), ("Pv", "Yv"), ("Pt", "Yt")):
        if d[probs].shape != d[labels].shape or not np.isfinite(d[probs]).all() or ((d[probs] < 0) | (d[probs] > 1)).any():
            raise ValueError("Expected finite probability arrays with matching label shapes")
    best = select_corrector(d["Ptr"], d["Ytr"], d["Pv"], d["Yv"], args.epochs, args.seed)
    test_probs = d["Pt"]
    if best["model"] is not None:
        with torch.no_grad(): test_probs = torch.sigmoid(best["model"](torch.tensor(test_probs, dtype=torch.float32))).numpy()
    print(f"Validation-selected corrector: {best['name']} (val F1={best['val_f1']:.4f})")
    print(f"Held-out test macro-F1: {macro(d['Yt'], test_probs, best['thresholds']):.4f}")


if __name__ == "__main__":
    main()
