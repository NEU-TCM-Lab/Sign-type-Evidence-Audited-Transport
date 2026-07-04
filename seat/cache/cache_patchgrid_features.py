from __future__ import annotations

# Cache the FULL frozen patch-token grid (not pooled) + per-patch mask weights, for the
# OT-readout validation gate: does an 8-attribute OT/attention readout beat mask-pool+MLP
# on the SAME frozen features? Backbone is frozen; only the readout head trains downstream.
# Reuses the geometry-aligned preprocessing from cache_backbone_features.py.

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from common import (
    DATASET_ROOT, FEATURES_DIR, LABELS, MANIFEST_DIR,
    assert_label_order, ensure_dirs, read_jsonl, split_manifest_path, write_json,
)
from cache_backbone_features import (
    BACKBONE_SPEC, load_backbone, preprocess_image, mask_grid_weights, resolve_dtype,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache full patch-grid + mask weights (frozen).")
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONE_SPEC.keys()))
    p.add_argument("--model-id", type=str, default=None)
    p.add_argument("--tag", required=True)
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    p.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def grid_feature_path(split: str, tag: str, features_dir: Path) -> Path:
    return Path(features_dir) / f"{split}_patchgrid_{tag}.pt"


@torch.inference_mode()
def extract_split(args, spec, vision, mean, std, device, dtype, split: str) -> dict:
    size, patch = int(spec["size"]), int(spec["patch"])
    grid = size // patch
    n_patches = grid * grid

    rows = read_jsonl(split_manifest_path(split, args.manifest_dir))
    if args.limit is not None:
        rows = rows[: args.limit]
    out_path = grid_feature_path(split, args.tag, args.features_dir)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite")

    patch_feats, mask_ws, bboxes, labels, ids = [], [], [], [], []
    fallback_count = 0
    t0 = time.perf_counter()

    for start in tqdm(range(0, len(rows), args.batch_size), desc=f"grid:{args.backbone}:{split}"):
        batch = rows[start : start + args.batch_size]
        pix, bw = [], []
        for row in batch:
            assert_label_order(row["label_order"])
            pix.append(preprocess_image(args.dataset_root / row["image_path"], size, mean, std))
            w, _s, fb = mask_grid_weights(args.dataset_root / row["mask_path"], size, patch, grid)
            bw.append((w, fb))
        pixel_values = torch.cat(pix, dim=0).to(device=device, dtype=dtype)
        extra = {"interpolate_pos_encoding": True} if args.backbone == "dinov2" else {}
        hs = vision(pixel_values=pixel_values, **extra).last_hidden_state
        specials = int(hs.shape[1]) - n_patches
        if specials < 0 or specials > 8:
            raise AssertionError(f"{args.backbone}: tokens {hs.shape[1]} vs grid^2 {n_patches}")
        patch_tokens = hs[:, specials:, :]  # [B,n_patches,D]

        for i, row in enumerate(batch):
            w, fb = bw[i]
            if fb:
                fallback_count += 1
            patch_feats.append(patch_tokens[i].to(torch.float16).cpu())   # [n_patches,D]
            mask_ws.append(w.to(torch.float16))                            # [n_patches]
            bboxes.append(torch.tensor(row["bbox_norm"], dtype=torch.float32))
            labels.append(torch.tensor(row["labels"], dtype=torch.float32))
            ids.append(row["id"])

    feature_dim = int(patch_feats[0].shape[-1])
    payload = {
        "split": split, "label_order": LABELS, "backbone": args.backbone,
        "model_id": spec["model_id"], "input_size": size, "patch_size": patch,
        "grid": grid, "tokens": n_patches, "feature_dim": feature_dim,
        "patch_features": torch.stack(patch_feats, dim=0),   # [N,n_patches,D] fp16
        "mask_weights": torch.stack(mask_ws, dim=0),         # [N,n_patches] fp16
        "bbox_norm": torch.stack(bboxes, dim=0),
        "labels": torch.stack(labels, dim=0),
        "ids": ids, "mask_fallback_count": fallback_count,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    write_json(out_path.with_suffix(".summary.json"), {
        "split": split, "path": str(out_path), "backbone": args.backbone,
        "count": len(ids), "feature_dim": feature_dim, "grid": grid, "tokens": n_patches,
        "fallback_count": fallback_count, "elapsed_sec": time.perf_counter() - t0,
        "tensor_gb": round(payload["patch_features"].numel() * 2 / 1e9, 2),
    })
    print(f"[grid {args.backbone}:{split}] N={len(ids)} tokens={n_patches} dim={feature_dim} "
          f"~{payload['patch_features'].numel()*2/1e9:.2f}GB fallback={fallback_count}")
    return payload["patch_features"].numel() * 2 / 1e9


def main() -> None:
    ensure_dirs()
    args = parse_args()
    spec = dict(BACKBONE_SPEC[args.backbone])
    if args.model_id:
        spec["model_id"] = args.model_id
    args.features_dir.mkdir(parents=True, exist_ok=True)
    dtype = resolve_dtype()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.backbone} <- {spec['model_id']} on {device} dtype={dtype}")
    _model, vision, mean, std = load_backbone(spec, dtype, device)
    total = sum(extract_split(args, spec, vision, mean, std, device, dtype, s) for s in args.splits)
    print(f"Done. tag={args.tag} total ~{total:.2f}GB of patch features")


if __name__ == "__main__":
    main()
