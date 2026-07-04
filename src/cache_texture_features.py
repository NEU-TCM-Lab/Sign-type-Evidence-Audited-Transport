from __future__ import annotations

# Per-patch TEXTURE / LOCAL-CONTRAST features for Ecchymosis (瘀斑 = dark/purple stasis spots), which
# are LOCAL ANOMALIES rather than a global color. Absolute color failed; here we encode how a patch
# DEVIATES from its own tongue and how textured/edgy it is:
#   - tongue-normalized color deviation  : patch_LAB_mean - mask-weighted tongue_LAB_mean        (3ch)
#   - local contrast                     : patch_LAB_mean - 3x3 grid-neighborhood mean           (3ch)
#   - texture                            : Sobel grad-mag patch mean & std, Laplacian abs mean    (3ch)
#   - spot edge                          : within-patch L range (max-min)                         (1ch)
# => 10ch, aligned to the 37x37 DINOv2 grid (518 square resize, 14px patch). GridDataset/train-ready.

import argparse, time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import torch

from common import DATASET_ROOT, FEATURES_DIR, LABELS, read_jsonl, split_manifest_path
from cache_backbone_features import mask_grid_weights

SIZE, PATCH, GRID = 518, 14, 37


def tex_grid(ipath: Path, mpath: Path) -> np.ndarray:
    with Image.open(ipath) as img:
        rgb = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[..., 0] / 255.0; a = (lab[..., 1] - 128) / 128.0; b = (lab[..., 2] - 128) / 128.0
    lab_n = np.stack([L, a, b], -1)                                          # [S,S,3]
    pm = lab_n.reshape(GRID, PATCH, GRID, PATCH, 3).mean(axis=(1, 3))        # [37,37,3] patch mean LAB
    Lrange = (lab_n[..., 0].reshape(GRID, PATCH, GRID, PATCH).max((1, 3))
              - lab_n[..., 0].reshape(GRID, PATCH, GRID, PATCH).min((1, 3)))  # [37,37] within-patch L range

    # mask-weighted tongue mean -> color deviation
    w, _s, _fb = mask_grid_weights(mpath, SIZE, PATCH, GRID)
    wv = w.numpy().reshape(GRID, GRID, 1).clip(0, 1)
    tongue_mean = (pm * wv).sum((0, 1)) / max(wv.sum(), 1e-6)                # [3]
    dev = pm - tongue_mean[None, None, :]                                    # [37,37,3]

    # local 3x3 grid-neighborhood contrast
    blur = cv2.blur(pm, (3, 3))                                              # [37,37,3]
    contrast = pm - blur                                                     # [37,37,3]

    # texture on grayscale: Sobel magnitude + Laplacian
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3); gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gmag = np.sqrt(gx * gx + gy * gy)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    gmag_p = gmag.reshape(GRID, PATCH, GRID, PATCH)
    grad_mean = gmag_p.mean((1, 3)); grad_std = gmag_p.std((1, 3))           # [37,37]
    lap_mean = lap.reshape(GRID, PATCH, GRID, PATCH).mean((1, 3))            # [37,37]

    feat = np.concatenate([dev, contrast,
                           grad_mean[..., None], grad_std[..., None], lap_mean[..., None],
                           Lrange[..., None]], axis=-1)                      # [37,37,10]
    return feat.reshape(GRID * GRID, 10).astype(np.float16), w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["val", "test", "train"])
    ap.add_argument("--tag", default="tex_seed42")
    a = ap.parse_args()
    mdir = Path("/root/autodl-tmp/TongueDx2_Qwen3VL4B_maskpool_cls/artifacts/manifests")
    for split in a.splits:
        rows = read_jsonl(split_manifest_path(split, mdir))
        t0 = time.perf_counter()
        feats, masks, labels, ids = [], [], [], []
        for i, row in enumerate(rows):
            f, w = tex_grid(DATASET_ROOT / row["image_path"], DATASET_ROOT / row["mask_path"])
            feats.append(torch.from_numpy(f)); masks.append(w.to(torch.float16))
            labels.append(torch.tensor(row["labels"], dtype=torch.float32)); ids.append(row["id"])
            if (i + 1) % 500 == 0:
                print(f"  {split} {i+1}/{len(rows)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        pf = torch.stack(feats)
        payload = {"split": split, "label_order": LABELS, "backbone": "texture", "grid": GRID,
                   "tokens": GRID * GRID, "feature_dim": int(pf.shape[-1]),
                   "patch_features": pf, "mask_weights": torch.stack(masks),
                   "labels": torch.stack(labels), "ids": ids, "mask_fallback_count": 0,
                   "bbox_norm": torch.zeros(len(ids), 9)}
        out = FEATURES_DIR / f"{split}_patchgrid_{a.tag}.pt"
        torch.save(payload, out)
        print(f"[{split}] {len(ids)} -> {out} ({time.perf_counter()-t0:.0f}s) feat={tuple(pf.shape)}", flush=True)


if __name__ == "__main__":
    main()
