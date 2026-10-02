from __future__ import annotations

# Cache per-patch COLOR features on the SAME 37x37 grid as the DINOv2 cache, for the two color-defined
# rare signs (TonguePale = whole-tongue paleness; Ecchymosis = localized purple spots). DINOv2 is
# semantic and under-encodes color; an explicit color pathway is the missing modality. Output matches
# the DINOv2 grid-cache schema (patch_features, mask_weights, labels, ids, feature_dim) so GridDataset
# and train_readout work directly with --tag color_seed42.
# Per patch (14x14, 518 square resize): mean of LAB(L,a,b) + HSV(H,S,V) + std(L,S) = 8 color channels.

import argparse, time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

import torch
from common import DATASET_ROOT, FEATURES_DIR, MANIFEST_DIR, LABELS, read_jsonl, split_manifest_path
from geometry import resize_image
from data_validation import identity_metadata
from cache_backbone_features import mask_grid_weights

SIZE, PATCH, GRID = 518, 14, 37
import os
EXTREMES = os.environ.get("COLOR_EXTREMES", "0") == "1"   # add localized within-patch min/max channels


def color_grid(path: Path, geometry: str = "square", extremes: bool | None = None) -> np.ndarray:
    with Image.open(path) as img:
        rgb = np.asarray(resize_image(img.convert("RGB"), SIZE, geometry), dtype=np.uint8)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)        # L 0-255, a,b 0-255 (offset 128)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)        # H 0-179, S,V 0-255
    chans = np.stack([lab[..., 0]/255, (lab[..., 1]-128)/128, (lab[..., 2]-128)/128,
                      hsv[..., 0]/179, hsv[..., 1]/255, hsv[..., 2]/255], axis=-1)  # [S,S,6]
    g = chans.reshape(GRID, PATCH, GRID, PATCH, 6)
    mean = g.mean(axis=(1, 3))                                           # [37,37,6]
    std = g[..., [0, 4]].std(axis=(1, 3))                                # [37,37,2] L,S texture
    parts = [mean, std]
    if EXTREMES if extremes is None else extremes:   # localized within-patch extremes for small spots (Ecchymosis = dark/purple/saturated)
        gm = g.transpose(0, 2, 1, 3, 4).reshape(GRID, GRID, PATCH * PATCH, 6)
        minLb = gm[..., [0, 2]].min(axis=2)                              # darkest, most-blue(-b) pixel
        maxaS = gm[..., [1, 4]].max(axis=2)                             # most-red(+a), most-saturated pixel
        parts += [minLb, maxaS]                                          # +4 channels
    feat = np.concatenate(parts, axis=-1).reshape(GRID * GRID, -1)
    return feat.astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    ap.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    ap.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    ap.add_argument("--geometry", choices=["square", "arpad"], default="square")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["val", "test", "train"])
    ap.add_argument("--tag", default="color_seed42")
    ap.add_argument("--color-extremes", action="store_true", default=EXTREMES)
    a = ap.parse_args()
    mdir = a.manifest_dir
    a.features_dir.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        out = a.features_dir / f"{split}_patchgrid_{a.tag}.pt"
        if out.exists() and not a.overwrite:
            raise FileExistsError(f"{out} exists; pass --overwrite")
        rows = read_jsonl(split_manifest_path(split, mdir))
        t0 = time.perf_counter()
        feats, masks, labels, ids = [], [], [], []
        fallback_count = 0
        for i, row in enumerate(rows):
            feats.append(torch.from_numpy(color_grid(a.dataset_root / row["image_path"], a.geometry, a.color_extremes)))
            w, _s, _fb = mask_grid_weights(a.dataset_root / row["mask_path"], SIZE, PATCH, GRID, a.geometry)
            fallback_count += int(_fb)
            masks.append(w.to(torch.float16))
            labels.append(torch.tensor(row["labels"], dtype=torch.float32))
            ids.append(row["id"])
            if (i + 1) % 500 == 0:
                print(f"  {split} {i+1}/{len(rows)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        pf_stack = torch.stack(feats)
        payload = {
            "split": split, "label_order": LABELS, "backbone": "color", "grid": GRID,
            "tokens": GRID * GRID, "feature_dim": int(pf_stack.shape[-1]),
            "patch_features": pf_stack, "mask_weights": torch.stack(masks),
            "labels": torch.stack(labels), "ids": ids, "mask_fallback_count": 0,
            "bbox_norm": torch.zeros(len(ids), 9),
        }
        payload.update(identity_metadata(rows))
        payload["geometry"] = a.geometry
        payload["color_extremes"] = a.color_extremes
        payload["mask_fallback_count"] = fallback_count
        torch.save(payload, out)
        print(f"[{split}] {len(ids)} -> {out} ({time.perf_counter()-t0:.0f}s) feat={tuple(payload['patch_features'].shape)}", flush=True)


if __name__ == "__main__":
    main()
