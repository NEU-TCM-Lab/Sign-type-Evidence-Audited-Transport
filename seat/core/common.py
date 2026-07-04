from __future__ import annotations

import csv
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path("/root/autodl-tmp/TongueDx2_pseudo_v1_qwen3vl4b_sam2")
LABEL_ROOT = Path("/root/autodl-tmp/TongueDx2/list")
MODEL_PATH = Path("/root/autodl-tmp/modelscope_cache/Qwen/Qwen3-VL-4B-Instruct")

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MANIFEST_DIR = ARTIFACTS_DIR / "manifests"
FEATURES_DIR = ARTIFACTS_DIR / "features"
RUNS_DIR = ARTIFACTS_DIR / "runs"

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

SPLIT_CSV = {
    "train": "train_fold1.csv",
    "val": "val_fold1.csv",
    "test": "test.csv",
}

EXPECTED_COUNTS = {
    "train": 3371,
    "val": 843,
    "test": 895,
}

IMAGE_PROMPT = "<|vision_start|><|image_pad|><|vision_end|>"


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def ensure_dirs() -> None:
    for path in (MANIFEST_DIR, FEATURES_DIR, RUNS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clean_for_json(data), f, ensure_ascii=True, indent=2)
        f.write("\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(clean_for_json(row), ensure_ascii=True) + "\n")


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def assert_label_order(label_order: list[str]) -> None:
    if list(label_order) != LABELS:
        raise AssertionError(f"Label order mismatch: got {label_order}, expected {LABELS}")


def split_manifest_path(split: str, manifest_dir: str | Path = MANIFEST_DIR) -> Path:
    return Path(manifest_dir) / f"{split}.jsonl"


def split_feature_path(split: str, tag: str | None = None, features_dir: str | Path = FEATURES_DIR) -> Path:
    suffix = f"_{tag}" if tag else ""
    return Path(features_dir) / f"{split}_pooled{suffix}.pt"


def label_order_path(out_dir: str | Path = ARTIFACTS_DIR) -> Path:
    return Path(out_dir) / "label_order.json"


def bbox_norm_xyxy(bbox_xyxy: list[float] | tuple[float, ...], image_width: int, image_height: int) -> list[float]:
    if len(bbox_xyxy) != 4:
        raise ValueError(f"Expected xyxy bbox with 4 values, got {bbox_xyxy}")
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    if image_width <= 0 or image_height <= 0:
        raise ValueError(f"Invalid image size: {image_width}x{image_height}")
    if not (x2 > x1 and y2 > y1):
        raise ValueError(f"Invalid bbox xyxy: {bbox_xyxy}")
    nx1 = x1 / image_width
    ny1 = y1 / image_height
    nx2 = x2 / image_width
    ny2 = y2 / image_height
    nw = (x2 - x1) / image_width
    nh = (y2 - y1) / image_height
    ncx = (x1 + x2) * 0.5 / image_width
    ncy = (y1 + y2) * 0.5 / image_height
    narea = nw * nh
    return [nx1, ny1, nx2, ny2, ncx, ncy, nw, nh, narea]


def load_feature_file(split: str, tag: str | None = None, features_dir: str | Path = FEATURES_DIR) -> dict[str, Any]:
    path = split_feature_path(split, tag=tag, features_dir=features_dir)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert_label_order(payload["label_order"])
    labels = payload["labels"]
    if tuple(labels.shape[1:]) != (len(LABELS),):
        raise AssertionError(f"{path} has bad labels shape {tuple(labels.shape)}")
    for key in ("global_features", "mask_features"):
        if payload[key].shape[0] != labels.shape[0]:
            raise AssertionError(f"{path} {key} row count does not match labels")
    if payload["bbox_norm"].shape != (labels.shape[0], 9):
        raise AssertionError(f"{path} bbox_norm shape is {tuple(payload['bbox_norm'].shape)}, expected {(labels.shape[0], 9)}")
    return payload


def env_python_hint() -> str:
    return "/root/autodl-tmp/conda/envs/vlm-vllm/bin/python"
