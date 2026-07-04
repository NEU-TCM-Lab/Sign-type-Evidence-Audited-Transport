from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from common import FEATURES_DIR, LABELS, RUNS_DIR, assert_label_order, load_feature_file, set_seed, timestamp, write_json
from metrics import compute_metrics, find_best_thresholds, sigmoid_np
from model import MaskPoolBBoxHead
from losses import LogitAdjuster, build_loss, compute_pos_prior, loss_uses_logit_adjustment


class CachedFeatureDataset(Dataset):
    def __init__(self, payload: dict, limit: int | None = None) -> None:
        assert_label_order(payload["label_order"])
        n = payload["labels"].shape[0] if limit is None else min(limit, payload["labels"].shape[0])
        self.global_features = payload["global_features"][:n].float()
        self.mask_features = payload["mask_features"][:n].float()
        self.bbox_norm = payload["bbox_norm"][:n].float()
        self.labels = payload["labels"][:n].float()
        if self.labels.shape[1] != len(LABELS):
            raise AssertionError(f"Bad labels shape: {tuple(self.labels.shape)}")

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return self.global_features[idx], self.mask_features[idx], self.bbox_norm[idx], self.labels[idx]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BCE multi-label head on cached Qwen3-VL pooled features.")
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--feature-mode",
        choices=["global", "global_mask", "global_mask_bbox"],
        default="global_mask_bbox",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    # --- Stage-1 strong-baseline knobs (all default to the historical behaviour) ---
    parser.add_argument(
        "--loss",
        choices=["bce", "asl", "logit_adjusted_bce", "asl_logit_adjusted"],
        default="bce",
        help="Loss: bce | asl | logit_adjusted_bce | asl_logit_adjusted.",
    )
    parser.add_argument("--asl-gamma-pos", type=float, default=0.0)
    parser.add_argument("--asl-gamma-neg", type=float, default=4.0)
    parser.add_argument("--asl-clip", type=float, default=0.05)
    parser.add_argument("--la-tau", type=float, default=1.0, help="Logit-adjustment temperature tau.")
    parser.add_argument(
        "--la-eval-apply",
        action="store_true",
        help="Also apply logit adjustment at val/test metric time (default: train-only).",
    )
    pos_grp = parser.add_mutually_exclusive_group()
    pos_grp.add_argument("--pos-weight", dest="pos_weight", action="store_true",
                         help="Use clamped neg/pos BCE pos_weight reweighting (historical default).")
    pos_grp.add_argument("--no-pos-weight", dest="pos_weight", action="store_false",
                         help="Disable pos_weight reweighting (clean baseline / pair with logit adjustment).")
    parser.set_defaults(pos_weight=True)
    parser.add_argument(
        "--selection-metric",
        choices=["macro_auroc", "macro_f1", "macro_ap", "val_loss"],
        default="macro_auroc",
        help="Best-checkpoint criterion on validation (historical default: macro_auroc).",
    )
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--threshold-step", type=float, default=0.005,
                        help="Per-class F1 threshold search step on val (spec uses 0.01).")
    return parser.parse_args()


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=torch.cuda.is_available())


def collect_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
    adjuster: nn.Module | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return RAW (un-adjusted) logits + labels + mean per-element loss.

    Back-compat: with criterion=None it reports the historical BCE-sum loss. When a criterion
    is given, the monitoring loss is computed on adjuster(logits) so train/val loss are comparable.
    Metrics are always derived from the raw logits returned here."""
    bce_sum = nn.BCEWithLogitsLoss(reduction="sum")
    logits_all = []
    labels_all = []
    total_loss = 0.0
    total_items = 0
    model.eval()
    with torch.inference_mode():
        for global_features, mask_features, bbox_norm, labels in loader:
            global_features = global_features.to(device, non_blocking=True)
            mask_features = mask_features.to(device, non_blocking=True)
            bbox_norm = bbox_norm.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(global_features, mask_features, bbox_norm)
            if criterion is None:
                total_loss += float(bce_sum(logits, labels).item())
                total_items += int(labels.numel())
            else:
                adj = adjuster(logits) if adjuster is not None else logits
                total_loss += float(criterion(adj, labels).item()) * int(labels.shape[0])
                total_items += int(labels.shape[0])
            logits_all.append(logits.cpu().numpy())
            labels_all.append(labels.cpu().numpy())
    return np.concatenate(logits_all, axis=0), np.concatenate(labels_all, axis=0), total_loss / max(total_items, 1)


def score_for_early_stop(metrics: dict, val_loss: float, selection_metric: str = "macro_auroc") -> float:
    key = {"macro_auroc": "auroc", "macro_f1": "f1", "macro_ap": "ap"}.get(selection_metric)
    if selection_metric == "val_loss":
        return -float(val_loss)
    score = metrics["macro"][key]
    if score is None or not np.isfinite(score):
        # fall back through auroc -> ap -> -val_loss so selection never crashes on degenerate cols
        for fallback in ("auroc", "ap"):
            score = metrics["macro"][fallback]
            if score is not None and np.isfinite(score):
                break
        else:
            score = -val_loss
    return float(score)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_name = args.run_name or timestamp()
    run_dir = RUNS_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    train_payload = load_feature_file("train", tag=args.tag, features_dir=args.features_dir)
    val_payload = load_feature_file("val", tag=args.tag, features_dir=args.features_dir)
    train_dataset = CachedFeatureDataset(train_payload, limit=args.limit_train)
    val_dataset = CachedFeatureDataset(val_payload, limit=args.limit_val)
    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = make_loader(val_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    feature_dim = int(train_dataset.global_features.shape[1])
    model = MaskPoolBBoxHead(
        feature_dim=feature_dim,
        num_labels=len(LABELS),
        dropout=args.dropout,
        feature_mode=args.feature_mode,
    ).to(device)

    train_labels = train_dataset.labels
    pos = train_labels.sum(dim=0)
    neg = train_labels.shape[0] - pos
    if args.pos_weight:
        pos_weight = torch.clamp(neg / torch.clamp(pos, min=1.0), min=0.2, max=5.0).to(device)
    else:
        pos_weight = None
    criterion = build_loss(
        args.loss,
        pos_weight=pos_weight,
        asl_gamma_pos=args.asl_gamma_pos,
        asl_gamma_neg=args.asl_gamma_neg,
        asl_clip=args.asl_clip,
    )
    # Logit adjustment: subtract tau*log((1-pi)/pi) from logits before the loss.
    adjuster = None
    if loss_uses_logit_adjustment(args.loss):
        pi = compute_pos_prior(train_labels)
        adjuster = LogitAdjuster(pi, tau=args.la_tau).to(device)
        print(f"logit-adjustment tau={args.la_tau} adjustment={adjuster.adjustment.cpu().numpy().round(3).tolist()}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = -float("inf")
    best_epoch = -1
    bad_epochs = 0
    history = []

    write_json(
        run_dir / "config.json",
        {
            "args": vars(args),
            "label_order": LABELS,
            "feature_dim": feature_dim,
            "feature_mode": args.feature_mode,
            "pos_weight": pos_weight.detach().cpu().numpy().tolist() if pos_weight is not None else None,
            "loss": args.loss,
            "asl": {"gamma_pos": args.asl_gamma_pos, "gamma_neg": args.asl_gamma_neg, "clip": args.asl_clip},
            "logit_adjustment": (
                {"tau": args.la_tau, "eval_apply": args.la_eval_apply,
                 "adjustment": adjuster.adjustment.detach().cpu().numpy().tolist()}
                if adjuster is not None else None
            ),
            "selection_metric": args.selection_metric,
            "scheduler": args.scheduler,
            "seed": args.seed,
            "train_count": len(train_dataset),
            "val_count": len(val_dataset),
        },
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_items = 0
        for global_features, mask_features, bbox_norm, labels in tqdm(train_loader, desc=f"train:{epoch}", leave=False):
            global_features = global_features.to(device, non_blocking=True)
            mask_features = mask_features.to(device, non_blocking=True)
            bbox_norm = bbox_norm.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(global_features, mask_features, bbox_norm)
            if logits.shape != labels.shape:
                raise AssertionError(f"Logits/labels shape mismatch: {tuple(logits.shape)} vs {tuple(labels.shape)}")
            adj_logits = adjuster(logits) if adjuster is not None else logits
            loss = criterion(adj_logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * int(labels.shape[0])
            running_items += int(labels.shape[0])
        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(running_items, 1)
        val_logits, val_y, val_loss = collect_logits(model, val_loader, device, criterion=criterion, adjuster=adjuster)
        val_prob = sigmoid_np(val_logits)
        thresholds = find_best_thresholds(val_y, val_prob, LABELS, step=args.threshold_step)
        val_metrics = compute_metrics(val_y, val_prob, thresholds, LABELS)
        score = score_for_early_stop(val_metrics, val_loss, args.selection_metric)
        # raw vs logit-adjusted metrics (both recorded; thresholds tuned on raw probs)
        val_metrics_adj = None
        if adjuster is not None:
            adj_prob = sigmoid_np(val_logits - adjuster.adjustment.detach().cpu().numpy())
            thr_adj = find_best_thresholds(val_y, adj_prob, LABELS, step=args.threshold_step)
            val_metrics_adj = compute_metrics(val_y, adj_prob, thr_adj, LABELS)
        per_class_f1 = {lab: round(val_metrics["per_label"][lab]["f1"], 3)
                        if not np.isnan(val_metrics["per_label"][lab]["f1"]) else None for lab in LABELS}
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "early_stop_score": score,
            "val_macro": val_metrics["macro"],
            "val_macro_f1": val_metrics["macro"]["f1"],
            "val_micro_f1": val_metrics["micro"]["f1"],
            "val_sample_f1": val_metrics["sample_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "per_class_f1": per_class_f1,
            "thresholds": thresholds,
        }
        if val_metrics_adj is not None:
            row["val_macro_f1_adjusted"] = val_metrics_adj["macro"]["f1"]
        history.append(row)
        write_json(run_dir / "history.json", history)
        print(
            f"epoch={epoch} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"macroF1={val_metrics['macro']['f1']:.4f} microF1={val_metrics['micro']['f1']:.4f} "
            f"sampleF1={val_metrics['sample_f1']:.4f} macroAUROC={val_metrics['macro']['auroc']} "
            f"[{args.selection_metric}={score:.4f}] per_class_f1={per_class_f1}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            bad_epochs = 0
            checkpoint = {
                "model_state": model.state_dict(),
                "label_order": LABELS,
                "thresholds": thresholds,
                "feature_dim": feature_dim,
                "model_config": {
                    "feature_dim": feature_dim,
                    "bbox_dim": 9,
                    "bbox_hidden": 256,
                    "hidden_dim": 1024,
                    "num_labels": len(LABELS),
                    "dropout": args.dropout,
                    "feature_mode": args.feature_mode,
                },
                "feature_mode": args.feature_mode,
                "epoch": epoch,
                "val_metrics": val_metrics,
                "val_loss": val_loss,
                "early_stop_score": score,
                "loss": args.loss,
                "selection_metric": args.selection_metric,
                "logit_adjustment": (
                    {"tau": args.la_tau, "eval_apply": args.la_eval_apply,
                     "adjustment": adjuster.adjustment.detach().cpu().numpy().tolist()}
                    if adjuster is not None else None
                ),
            }
            torch.save(checkpoint, run_dir / "best_head.pt")
            write_json(run_dir / "best_val_metrics.json", val_metrics)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}")
                break

    write_json(
        run_dir / "summary.json",
        {
            "best_epoch": best_epoch,
            "best_score": best_score,
            "feature_mode": args.feature_mode,
            "label_order": LABELS,
        },
    )
    print(f"Best checkpoint: {run_dir / 'best_head.pt'}")


if __name__ == "__main__":
    main()
