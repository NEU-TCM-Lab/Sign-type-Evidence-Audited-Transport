from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from common import FEATURES_DIR, LABELS, assert_label_order, load_feature_file, write_json
from metrics import compute_metrics, sigmoid_np
from model import MaskPoolBBoxHead
from train_head import CachedFeatureDataset, collect_logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained mask-pooling classification head.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--export-predictions", action="store_true", help="Write sample-level private prediction CSVs locally")
    parser.add_argument("--out-dir", type=Path, default=None)
    return parser.parse_args()


def write_predictions(path: Path, payload: dict, probs, thresholds: dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = payload["labels"].numpy()
    preds = probs.copy()
    for i, label in enumerate(LABELS):
        preds[:, i] = (probs[:, i] >= thresholds[label]).astype(int)
    header = ["id", "image_path"]
    header += [f"{label}_true" for label in LABELS]
    header += [f"{label}_prob" for label in LABELS]
    header += [f"{label}_pred" for label in LABELS]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row_idx, sample_id in enumerate(payload["ids"]):
            writer.writerow(
                [sample_id, payload["image_paths"][row_idx]]
                + [int(labels[row_idx, i]) for i in range(len(LABELS))]
                + [float(probs[row_idx, i]) for i in range(len(LABELS))]
                + [int(preds[row_idx, i]) for i in range(len(LABELS))]
            )


def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert_label_order(checkpoint["label_order"])
    thresholds = checkpoint["thresholds"]
    for label in LABELS:
        if label not in thresholds:
            raise AssertionError(f"Checkpoint thresholds missing label {label}")

    model_config = checkpoint["model_config"]
    model = MaskPoolBBoxHead(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    out_dir = args.out_dir or args.checkpoint.parent
    for split in args.splits:
        payload = load_feature_file(split, tag=args.tag, features_dir=args.features_dir)
        dataset = CachedFeatureDataset(payload)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=torch.cuda.is_available())
        logits, y_true, loss = collect_logits(model, loader, device)
        adjustment = checkpoint.get("logit_adjustment")
        if adjustment and adjustment.get("eval_apply"):
            # Match the validation-time scores used to select these thresholds.
            import numpy as np
            logits = logits - np.asarray(adjustment["adjustment"])[None, :]
        probs = sigmoid_np(logits)
        metrics = compute_metrics(y_true, probs, thresholds, LABELS)
        metrics["loss"] = loss
        write_json(out_dir / f"metrics_{split}.json", metrics)
        if args.export_predictions:
            write_predictions(out_dir / f"predictions_{split}.csv", payload, probs, thresholds)
        print(f"{split}: wrote metrics and predictions to {out_dir}")


if __name__ == "__main__":
    main()
