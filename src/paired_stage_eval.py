from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from settings import OUTPUTS_DIR
from data_validation import alignment_order, validate_split_payloads
from fusion import validate_prediction_dump


LABELS = [
    "TonguePale",
    "TipSideRed",
    "Spot",
    "Ecchymosis",
    "Crack",
    "Toothmark",
    "FurThick",
    "FurYellow",
]
GAMMAS = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def f1_score(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    den = 2 * tp + fp + fn
    return 2 * tp / den if den > 0 else 0.0


def best_threshold(y: np.ndarray, prob: np.ndarray, step: float) -> tuple[float, float]:
    best_f1 = -1.0
    best_thr = 0.5
    for thr in np.arange(0.05, 0.95 + 1e-9, step):
        f1 = f1_score(y, (prob >= thr).astype(np.int64))
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, best_f1


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        dump = {k: data[k] for k in data.files}
    validate_prediction_dump(dump)
    return dump


def order_logits(ref: dict[str, np.ndarray], other: dict[str, np.ndarray], split: str) -> np.ndarray:
    order = alignment_order(ref, other, id_key=f"{split}_ids", label_key=f"{split}_labels")
    return other[f"{split}_logits"][order]



def eval_probs(y_val: np.ndarray, p_val: np.ndarray, y_test: np.ndarray, p_test: np.ndarray, step: float) -> tuple[float, list[float], list[float]]:
    test_f1s = []
    thresholds = []
    for class_idx in range(y_val.shape[1]):
        threshold, _ = best_threshold(y_val[:, class_idx], p_val[:, class_idx], step)
        thresholds.append(threshold)
        pred = (p_test[:, class_idx] >= threshold).astype(np.int64)
        test_f1s.append(f1_score(y_test[:, class_idx], pred))
    return float(np.mean(test_f1s) * 100), [x * 100 for x in test_f1s], thresholds


def fuse_color_pale(coocc: dict[str, np.ndarray], color: dict[str, np.ndarray], step: float) -> tuple[np.ndarray, np.ndarray, float]:
    y_val = coocc["val_labels"]
    base_val_logits = coocc["val_logits"].copy()
    base_test_logits = coocc["test_logits"].copy()
    color_val = order_logits(coocc, color, "val")[:, 0]
    color_test = order_logits(coocc, color, "test")[:, 0]

    best_gamma = 0.0
    best_val_f1 = -1.0
    for gamma in GAMMAS:
        pale_prob = sigmoid(base_val_logits[:, 0] + gamma * color_val)
        _, val_f1 = best_threshold(y_val[:, 0], pale_prob, step)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_gamma = gamma

    base_val_logits[:, 0] = base_val_logits[:, 0] + best_gamma * color_val
    base_test_logits[:, 0] = base_test_logits[:, 0] + best_gamma * color_test
    return sigmoid(base_val_logits), sigmoid(base_test_logits), best_gamma


def mean_std(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": round(float(values.mean()), 4),
        "std": round(float(values.std(ddof=1)), 4) if len(values) > 1 else 0.0,
    }


def bootstrap_ci(diff: np.ndarray, n_boot: int, rng_seed: int) -> dict[str, float]:
    rng = np.random.default_rng(rng_seed)
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(diff), size=len(diff))
        samples.append(float(diff[idx].mean()))
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return {"lo": round(float(lo), 4), "hi": round(float(hi), 4)}


def sign_count(diff: np.ndarray) -> dict[str, int]:
    return {
        "positive": int((diff > 1e-12).sum()),
        "zero": int((np.abs(diff) <= 1e-12).sum()),
        "negative": int((diff < -1e-12).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--n-boot", type=int, default=20000)
    parser.add_argument("--rng-seed", type=int, default=20260702)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    outputs_dir = args.outputs_dir.resolve()
    color = load_npz(outputs_dir / "color_pale.npz")
    rows = []

    for seed in args.seeds:
        base = load_npz(outputs_dir / f"base8_seed{seed}.npz")
        coocc = load_npz(outputs_dir / f"coocc8_seed{seed}.npz")
        y_val = base["val_labels"]
        y_test = base["test_labels"]

        base_macro, _, _ = eval_probs(y_val, sigmoid(base["val_logits"]), y_test, sigmoid(base["test_logits"]), args.threshold_step)
        coocc_val_logits = order_logits(base, coocc, "val")
        coocc_test_logits = order_logits(base, coocc, "test")
        coocc_macro, _, _ = eval_probs(y_val, sigmoid(coocc_val_logits), y_test, sigmoid(coocc_test_logits), args.threshold_step)

        coocc_for_color = dict(coocc)
        coocc_for_color["val_logits"] = coocc_val_logits
        coocc_for_color["test_logits"] = coocc_test_logits
        coocc_for_color["val_ids"] = base["val_ids"]
        coocc_for_color["test_ids"] = base["test_ids"]
        coocc_for_color["val_labels"] = y_val
        coocc_for_color["test_labels"] = y_test
        fused_val_prob, fused_test_prob, gamma = fuse_color_pale(coocc_for_color, color, args.threshold_step)
        fused_macro, _, _ = eval_probs(y_val, fused_val_prob, y_test, fused_test_prob, args.threshold_step)

        rows.append(
            {
                "seed": seed,
                "baseline": round(base_macro, 4),
                "plus_c2": round(coocc_macro, 4),
                "plus_c2_c1": round(fused_macro, 4),
                "c2_gain": round(coocc_macro - base_macro, 4),
                "c1_gain": round(fused_macro - coocc_macro, 4),
                "color_gamma": gamma,
            }
        )

    base_arr = np.array([r["baseline"] for r in rows], dtype=float)
    c2_arr = np.array([r["plus_c2"] for r in rows], dtype=float)
    full_arr = np.array([r["plus_c2_c1"] for r in rows], dtype=float)
    c2_gain = c2_arr - base_arr
    c1_gain = full_arr - c2_arr

    summary = {
        "seeds": args.seeds,
        "stage_stats": {
            "baseline": mean_std(base_arr),
            "plus_c2": mean_std(c2_arr),
            "plus_c2_c1": mean_std(full_arr),
        },
        "paired_gains": {
            "baseline_to_c2": {
                "mean": round(float(c2_gain.mean()), 4),
                "std": round(float(c2_gain.std(ddof=1)), 4),
                "bootstrap_ci95": bootstrap_ci(c2_gain, args.n_boot, args.rng_seed),
                "sign_count": sign_count(c2_gain),
            },
            "c2_to_c2_c1": {
                "mean": round(float(c1_gain.mean()), 4),
                "std": round(float(c1_gain.std(ddof=1)), 4),
                "bootstrap_ci95": bootstrap_ci(c1_gain, args.n_boot, args.rng_seed + 1),
                "sign_count": sign_count(c1_gain),
            },
        },
        "rows": rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary["stage_stats"], indent=2))
    print(json.dumps(summary["paired_gains"], indent=2))


if __name__ == "__main__":
    main()
