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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def f1_score(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    den = 2 * tp + fp + fn
    return 2 * tp / den if den > 0 else 0.0


def prf(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    tp = float(((pred == 1) & (y == 1)).sum())
    fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


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


def align_like(base: dict[str, np.ndarray], other: dict[str, np.ndarray], split: str) -> np.ndarray:
    order = alignment_order(base, other, id_key=f"{split}_ids", label_key=f"{split}_labels")
    return other[f"{split}_logits"][order]



def aggregate(logits: list[np.ndarray], mode: str) -> np.ndarray:
    stack = np.stack(logits, axis=0)
    if mode == "logit_mean":
        return stack.mean(axis=0)
    if mode == "prob_mean":
        return sigmoid(stack).mean(axis=0)
    if mode == "prob_bottom2_mean":
        return np.sort(sigmoid(stack), axis=0)[:2].mean(axis=0)
    raise ValueError(f"unsupported aggregate mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--aggregate", choices=["logit_mean", "prob_mean", "prob_bottom2_mean"], default="logit_mean")
    parser.add_argument("--color-npz", default="color_pale.npz")
    parser.add_argument("--pale-gamma", type=float, default=0.0)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    args = parser.parse_args()

    outputs_dir = args.outputs_dir.resolve()
    base_paths = [outputs_dir / f"coocc8_seed{seed}.npz" for seed in args.seeds]
    missing = [str(path) for path in base_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing coocc npz files: {missing}")

    runs = [load_npz(path) for path in base_paths]
    color = load_npz(outputs_dir / args.color_npz)
    ref = runs[0]
    y_val = ref["val_labels"]
    y_test = ref["test_labels"]

    for run in runs[1:] + [color]:
        for split in ("val", "test"):
            if set(map(str, run[f"{split}_ids"])) != set(map(str, ref[f"{split}_ids"])):
                raise ValueError(f"id set mismatch for split={split}")

    val_logits = [align_like(ref, run, "val") for run in runs]
    test_logits = [align_like(ref, run, "test") for run in runs]
    val_fused = aggregate(val_logits, args.aggregate)
    test_fused = aggregate(test_logits, args.aggregate)

    if args.pale_gamma != 0:
        color_val = align_like(ref, color, "val")[:, 0]
        color_test = align_like(ref, color, "test")[:, 0]
        if args.aggregate == "logit_mean":
            val_fused[:, 0] = val_fused[:, 0] + args.pale_gamma * color_val
            test_fused[:, 0] = test_fused[:, 0] + args.pale_gamma * color_test
            val_fused = sigmoid(val_fused)
            test_fused = sigmoid(test_fused)
        else:
            eps = 1e-6
            val_p = np.clip(val_fused[:, 0], eps, 1 - eps)
            test_p = np.clip(test_fused[:, 0], eps, 1 - eps)
            val_logit = np.log(val_p / (1 - val_p))
            test_logit = np.log(test_p / (1 - test_p))
            val_fused[:, 0] = sigmoid(val_logit + args.pale_gamma * color_val)
            test_fused[:, 0] = sigmoid(test_logit + args.pale_gamma * color_test)
    elif args.aggregate == "logit_mean":
        val_fused = sigmoid(val_fused)
        test_fused = sigmoid(test_fused)

    rows = []
    val_f1s = []
    test_f1s = []
    for class_idx, label in enumerate(LABELS):
        threshold, val_f1 = best_threshold(y_val[:, class_idx], val_fused[:, class_idx], args.threshold_step)
        pred = (test_fused[:, class_idx] >= threshold).astype(np.int64)
        precision, recall, test_f1 = prf(y_test[:, class_idx], pred)
        val_f1s.append(val_f1)
        test_f1s.append(test_f1)
        rows.append(
            {
                "class_name": label,
                "support_pos": int(y_test[:, class_idx].sum()),
                "support_neg": int((1 - y_test[:, class_idx]).sum()),
                "threshold": round(threshold, 4),
                "val_f1": round(val_f1 * 100, 4),
                "precision": round(precision * 100, 4),
                "recall": round(recall * 100, 4),
                "test_f1": round(test_f1 * 100, 4),
            }
        )

    result = {
        "seeds": args.seeds,
        "aggregate": args.aggregate,
        "pale_gamma": args.pale_gamma,
        "threshold_step": args.threshold_step,
        "val_macro_f1": round(float(np.mean(val_f1s) * 100), 4),
        "test_macro_f1": round(float(np.mean(test_f1s) * 100), 4),
        "per_class": rows,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"seeds={args.seeds} aggregate={args.aggregate} pale_gamma={args.pale_gamma} "
        f"val_macro_f1={result['val_macro_f1']:.2f} test_macro_f1={result['test_macro_f1']:.2f}"
    )


if __name__ == "__main__":
    main()
