from __future__ import annotations
# Real ResNet-34 Grad-CAM for the traditional-CNN interpretability baseline (Panel A).
# Model: artifacts/runs/resnet34_pseudo_seed42/best.pt (torchvision resnet34, fc->Linear(512,8)),
# trained on bbox-cropped tongue @384px, ImageNet-normalised. test macro-F1 = 0.6866.
# gradcam(iid, cls) returns a 518x518 CAM in the SAME full-image frame the DINOv2 figure uses,
# so Panel A (CNN Grad-CAM) and Panel B (our patch evidence) are geometrically aligned.
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision import models
from PIL import Image

from settings import RESNET_CHECKPOINT, DATASET_ROOT, MANIFEST_DIR
CKPT = RESNET_CHECKPOINT
ROOT = DATASET_ROOT
MAN = MANIFEST_DIR / "test.jsonl"
IMEAN = np.array([0.485, 0.456, 0.406], np.float32); ISTD = np.array([0.229, 0.224, 0.225], np.float32)
S = 518
_man = None

def get_manifest():
    global _man
    if _man is None:
        with MAN.open(encoding="utf-8") as f:
            _man = {r["id"]: r for r in (json.loads(line) for line in f if line.strip())}
    return _man


def _build():
    m = models.resnet34(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 8)
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["state_dict"]); m.eval()
    return m


_MODEL = None


def gradcam(iid: str, cls: int, dev="cpu"):
    """Return (cam518 [518,518] float in [0,1] on full-image frame, None outside padded bbox)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _build().to(dev)
    row = get_manifest()[iid]
    img = Image.open(ROOT / row["image_path"]).convert("RGB")
    W, H = img.size
    x1, y1, x2, y2 = [int(round(v)) for v in row["sam2_bbox_px"]]
    px, py = int(round((x2 - x1) * 0.08)), int(round((y2 - y1) * 0.08))   # same 8% pad as training
    bx0, by0 = max(0, x1 - px), max(0, y1 - py); bx1, by1 = min(W, x2 + px), min(H, y2 + py)
    crop = img.crop((bx0, by0, bx1, by1)).resize((384, 384), Image.BILINEAR)
    arr = (np.asarray(crop, np.float32) / 255.0 - IMEAN) / ISTD
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(dev)

    feats = {}
    def fwd(_m, _i, o):
        feats["a"] = o; o.register_hook(lambda g: feats.__setitem__("g", g))
    h = _MODEL.layer4.register_forward_hook(fwd)
    x.requires_grad_(True)
    logit = _MODEL(x)[0, cls]
    _MODEL.zero_grad(); logit.backward()
    h.remove()
    A, G = feats["a"][0], feats["g"][0]                        # 512 x 12 x 12
    w = G.mean(dim=(1, 2))
    cam = torch.relu((w[:, None, None] * A).sum(0)).detach().cpu().numpy()   # 12 x 12
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-9)
    # place CAM back onto the full-image 518 frame at the padded-bbox location
    import cv2
    sx, sy = S / W, S / H
    fx0, fy0, fx1, fy1 = int(bx0 * sx), int(by0 * sy), int(bx1 * sx), int(by1 * sy)
    cam_box = cv2.resize(cam, (max(1, fx1 - fx0), max(1, fy1 - fy0)), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((S, S), np.float32)
    canvas[fy0:fy1, fx0:fx1] = cam_box
    return canvas


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("sample_id")
    parser.add_argument("class_index", type=int, choices=range(8))
    args = parser.parse_args()
    iid, cls = args.sample_id, args.class_index
    c = gradcam(iid, cls)
    print("cam", c.shape, "max", c.max(), "coverage", float((c > 0.5).mean()))
