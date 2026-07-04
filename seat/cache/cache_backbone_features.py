from __future__ import annotations

# Cache frozen pooled features from non-Qwen vision backbones (CLIP / SigLIP2 / DINOv2),
# producing the SAME payload schema as cache_pooled_features.py so train_head.py /
# evaluate_head.py / run_experiments.py work unchanged. The ONLY variable vs the Qwen
# pipeline is the backbone: same manifest (same 3371/843/895 samples), same mask-pooling,
# same downstream head -> a fair "why MLLM vs CLIP/SigLIP/DINO" ablation.
#
# Geometry alignment (the only real risk): we do our OWN square resize (no center crop)
# for BOTH image and mask, derive the patch grid from config, and mask-pool with avg_pool2d
# at kernel=patch_size. This guarantees the mask grid == the patch-token grid.

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

from common import (
    DATASET_ROOT,
    FEATURES_DIR,
    LABELS,
    MANIFEST_DIR,
    assert_label_order,
    ensure_dirs,
    read_jsonl,
    split_feature_path,
    split_manifest_path,
    write_json,
)


# Each backbone: HF id, square input edge S, patch P (S % P == 0), and how many leading
# special tokens to drop from last_hidden_state (CLS/register). Native resolutions chosen
# so S is divisible by P (clean patch grid, exact mask alignment).
BACKBONE_SPEC: dict[str, dict[str, Any]] = {
    "clip": {
        "model_id": "openai/clip-vit-large-patch14-336",
        "size": 336, "patch": 14,        # 24x24 = 576 tokens, +1 CLS
        "vision_attr": "vision_model",
    },
    "siglip2": {
        "model_id": "google/siglip2-so400m-patch16-384",
        "size": 384, "patch": 16,        # 24x24 = 576 tokens, no CLS
        "vision_attr": "vision_model",
    },
    "dinov2": {
        "model_id": "facebook/dinov2-large",
        "size": 518, "patch": 14,        # 37x37 = 1369 tokens, +1 CLS
        "vision_attr": None,             # Dinov2Model itself returns last_hidden_state
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cache frozen pooled features from CLIP/SigLIP2/DINOv2.")
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONE_SPEC.keys()))
    p.add_argument("--model-id", type=str, default=None, help="Override the HF model id.")
    p.add_argument("--tag", required=True, help="Output tag, e.g. clipL336_seed42.")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    p.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    p.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    p.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="Debug: cap rows per split.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def resolve_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def load_backbone(spec: dict[str, Any], dtype: torch.dtype, device: torch.device):
    proc = AutoImageProcessor.from_pretrained(spec["model_id"])
    model = AutoModel.from_pretrained(spec["model_id"], dtype=dtype, low_cpu_mem_usage=True)
    model.to(device).eval()
    for prm in model.parameters():
        prm.requires_grad_(False)
    vision = getattr(model, spec["vision_attr"]) if spec["vision_attr"] else model
    mean = torch.tensor(proc.image_mean, dtype=torch.float32).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, dtype=torch.float32).view(1, 3, 1, 1)
    return model, vision, mean, std


def preprocess_image(path: Path, size: int, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    with Image.open(path) as img:
        img = img.convert("RGB").resize((size, size), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0          # [S,S,3]
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)       # [1,3,S,S]
    return (t - mean) / std


def mask_grid_weights(path: Path, size: int, patch: int, grid: int) -> tuple[torch.Tensor, float, bool]:
    # Identical square resize as the image -> avg_pool to patch grid. Top-left aligned,
    # same remainder handling as a patch-embed conv (here size % patch == 0, so exact).
    with Image.open(path) as m:
        m = m.convert("L").resize((size, size), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(m, dtype=np.float32)
    if arr.max(initial=0.0) > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    pooled = F.avg_pool2d(torch.from_numpy(arr).view(1, 1, size, size), kernel_size=patch, stride=patch)
    if tuple(pooled.shape[-2:]) != (grid, grid):
        raise AssertionError(f"mask pooled grid {tuple(pooled.shape[-2:])} != ({grid},{grid}) for {path}")
    w = pooled.reshape(-1)
    s = float(w.sum().item())
    fallback = (not np.isfinite(s)) or s <= 0.0
    return w, s, fallback


@torch.inference_mode()
def extract_split(args, spec, model, vision, mean, std, device, dtype, split: str) -> dict:
    size, patch = int(spec["size"]), int(spec["patch"])
    grid = size // patch
    if grid * patch != size:
        raise AssertionError(f"size {size} not divisible by patch {patch}")
    n_patches = grid * grid

    rows = read_jsonl(split_manifest_path(split, args.manifest_dir))
    if args.limit is not None:
        rows = rows[: args.limit]

    out_path = split_feature_path(split, tag=args.tag, features_dir=args.features_dir)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists; pass --overwrite")

    g_feats, m_feats, bboxes, labels, ids, img_paths = [], [], [], [], [], []
    fallback_count = 0
    feature_dim = 0
    t0 = time.perf_counter()

    for start in tqdm(range(0, len(rows), args.batch_size), desc=f"{args.backbone}:{split}"):
        batch = rows[start : start + args.batch_size]
        pix = []
        bw = []  # per-sample (weights, fallback)
        for row in batch:
            assert_label_order(row["label_order"])
            pix.append(preprocess_image(args.dataset_root / row["image_path"], size, mean, std))
            w, _s, fb = mask_grid_weights(args.dataset_root / row["mask_path"], size, patch, grid)
            bw.append((w, fb))
        pixel_values = torch.cat(pix, dim=0).to(device=device, dtype=dtype)

        extra = {"interpolate_pos_encoding": True} if args.backbone == "dinov2" else {}
        hs = vision(pixel_values=pixel_values, **extra).last_hidden_state  # [B,T,D]
        T = int(hs.shape[1])
        specials = T - n_patches
        if specials < 0 or specials > 8:
            raise AssertionError(f"{args.backbone}: token count {T} vs grid^2 {n_patches} (specials={specials})")
        patch_tokens = hs[:, specials:, :].float()  # [B,n_patches,D]
        if int(patch_tokens.shape[1]) != n_patches:
            raise AssertionError(f"patch token count {patch_tokens.shape[1]} != {n_patches}")

        for i, row in enumerate(batch):
            tok = patch_tokens[i]                      # [n_patches,D]
            global_feat = tok.mean(dim=0)
            w, fb = bw[i]
            if fb:
                mask_feat = global_feat
                fallback_count += 1
            else:
                wd = w.to(device=device, dtype=torch.float32)
                mask_feat = (tok * wd[:, None]).sum(dim=0) / wd.sum()
            g_feats.append(global_feat.to(torch.float16).cpu())
            m_feats.append(mask_feat.to(torch.float16).cpu())
            bboxes.append(torch.tensor(row["bbox_norm"], dtype=torch.float32))
            labels.append(torch.tensor(row["labels"], dtype=torch.float32))
            ids.append(row["id"])
            img_paths.append(row["image_path"])

    feature_dim = int(g_feats[0].numel())
    payload = {
        "split": split,
        "label_order": LABELS,
        "backbone": args.backbone,
        "model_id": spec["model_id"],
        "input_size": size,
        "patch_size": patch,
        "merge_size": 1,
        "grid": grid,
        "tokens": n_patches,
        "feature_dim": feature_dim,
        "global_features": torch.stack(g_feats, dim=0),
        "mask_features": torch.stack(m_feats, dim=0),
        "bbox_norm": torch.stack(bboxes, dim=0),
        "labels": torch.stack(labels, dim=0),
        "ids": ids,
        "image_paths": img_paths,
        "mask_fallback_count": fallback_count,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    summary = {
        "split": split,
        "path": str(out_path),
        "backbone": args.backbone,
        "model_id": spec["model_id"],
        "count": len(ids),
        "feature_dim": feature_dim,
        "input_size": size,
        "patch_size": patch,
        "grid": grid,
        "tokens": n_patches,
        "fallback_count": fallback_count,
        "elapsed_sec": time.perf_counter() - t0,
    }
    write_json(out_path.with_suffix(".summary.json"), summary)
    print(f"[{args.backbone}:{split}] count={len(ids)} dim={feature_dim} tokens={n_patches} fallback={fallback_count}")
    return summary


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
    model, vision, mean, std = load_backbone(spec, dtype, device)
    summaries = [extract_split(args, spec, model, vision, mean, std, device, dtype, s) for s in args.splits]
    write_json(args.features_dir / f"cache_summary_{args.tag}.json", summaries)
    print(f"Done. Wrote features for tag={args.tag} to {args.features_dir}")


if __name__ == "__main__":
    main()
