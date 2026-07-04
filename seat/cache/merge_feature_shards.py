from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cache_pooled_features import shard_feature_path
from common import (
    EXPECTED_COUNTS,
    FEATURES_DIR,
    LABELS,
    MANIFEST_DIR,
    assert_label_order,
    read_jsonl,
    split_feature_path,
    split_manifest_path,
    write_json,
)


TENSOR_KEYS = [
    "global_features",
    "mask_features",
    "bbox_norm",
    "labels",
    "grid_hw",
    "resized_hw",
    "actual_tokens",
    "mask_weight_sum",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge deterministic cached feature shards into one split cache.")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-visual-tokens", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def token_stats(tokens: list[int]) -> dict[str, Any]:
    if not tokens:
        return {"min": None, "mean": None, "max": None, "p50": None, "p90": None, "p95": None}
    arr = np.asarray(tokens, dtype=np.float64)
    return {
        "min": int(arr.min()),
        "mean": float(arr.mean()),
        "max": int(arr.max()),
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
    }


def summary_path(feature_path: Path) -> Path:
    return feature_path.with_suffix(".summary.json")


def validate_final(path: Path, expected_ids: list[str], max_visual_tokens: int | None) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert_label_order(payload["label_order"])
    if list(payload["ids"]) != expected_ids:
        raise AssertionError(f"{path} ids do not match manifest order")
    if payload["labels"].shape != (len(expected_ids), len(LABELS)):
        raise AssertionError(f"{path} labels shape is {tuple(payload['labels'].shape)}")
    tokens = [int(v) for v in payload["actual_tokens"].tolist()]
    if max_visual_tokens is not None and max(tokens) > max_visual_tokens:
        raise AssertionError(f"{path} token max {max(tokens)} > {max_visual_tokens}")
    return make_summary(payload, path, source_num_shards=int(payload.get("source_num_shards", 1)))


def make_summary(payload: dict[str, Any], out_path: Path, source_num_shards: int) -> dict[str, Any]:
    tokens = [int(v) for v in payload["actual_tokens"].tolist()]
    return {
        "split": payload["split"],
        "path": str(out_path),
        "partial": False,
        "count": int(len(payload["ids"])),
        "selected_count": int(len(payload["selected_ids"])),
        "feature_dim": int(payload["feature_dim"]),
        "fallback_count": int(sum(bool(v) for v in payload["mask_fallback"])),
        "label_order": LABELS,
        "patch_size": int(payload["patch_size"]),
        "merge_size": int(payload["merge_size"]),
        "max_visual_tokens": payload.get("max_visual_tokens"),
        "max_pixels": payload.get("max_pixels"),
        "token_stats": token_stats(tokens),
        "selected_ids": payload["selected_ids"],
        "source_num_shards": int(source_num_shards),
    }


def main() -> None:
    args = parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")

    manifest_rows = read_jsonl(split_manifest_path(args.split, args.manifest_dir))
    expected_ids = [row["id"] for row in manifest_rows]
    if len(expected_ids) != EXPECTED_COUNTS[args.split]:
        raise AssertionError(f"Manifest count mismatch for {args.split}: {len(expected_ids)}")

    out_path = split_feature_path(args.split, tag=args.tag, features_dir=args.features_dir)
    if out_path.exists() and not args.overwrite:
        summary = validate_final(out_path, expected_ids, args.max_visual_tokens)
        write_json(summary_path(out_path), summary)
        print(f"{args.split}: final cache already exists and is valid: {out_path}")
        return

    rows_by_id: dict[str, dict[str, Any]] = {}
    metadata: dict[str, Any] | None = None
    base_path = split_feature_path(args.split, tag=args.tag, features_dir=args.features_dir)

    for shard_index in range(args.num_shards):
        shard_path = shard_feature_path(base_path, args.num_shards, shard_index)
        if not shard_path.exists():
            raise FileNotFoundError(shard_path)
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        assert_label_order(payload["label_order"])
        if payload["split"] != args.split:
            raise AssertionError(f"{shard_path} split={payload['split']} expected {args.split}")
        if int(payload.get("num_shards", args.num_shards)) != args.num_shards:
            raise AssertionError(f"{shard_path} num_shards mismatch")
        if int(payload.get("shard_index", shard_index)) != shard_index:
            raise AssertionError(f"{shard_path} shard_index mismatch")
        if metadata is None:
            metadata = payload
        else:
            for key in ("label_order", "model_path", "image_prompt", "patch_size", "merge_size", "max_visual_tokens", "max_pixels"):
                if payload.get(key) != metadata.get(key):
                    raise AssertionError(f"{shard_path} metadata mismatch for {key}")
        for idx, sample_id in enumerate(payload["ids"]):
            if sample_id in rows_by_id:
                raise AssertionError(f"Duplicate id across shards: {sample_id}")
            rows_by_id[sample_id] = {
                "image_path": payload["image_paths"][idx],
                "mask_fallback": bool(payload["mask_fallback"][idx]),
                **{key: payload[key][idx].cpu() for key in TENSOR_KEYS},
            }

    missing = [sample_id for sample_id in expected_ids if sample_id not in rows_by_id]
    extra = [sample_id for sample_id in rows_by_id if sample_id not in set(expected_ids)]
    if missing or extra:
        raise AssertionError(f"Shard merge id mismatch: missing={len(missing)} extra={len(extra)}")
    if metadata is None:
        raise AssertionError("No shards loaded")

    ordered = [rows_by_id[sample_id] for sample_id in expected_ids]
    payload = {
        "split": args.split,
        "label_order": LABELS,
        "model_path": metadata["model_path"],
        "image_prompt": metadata["image_prompt"],
        "patch_size": int(metadata["patch_size"]),
        "merge_size": int(metadata["merge_size"]),
        "max_visual_tokens": metadata.get("max_visual_tokens"),
        "max_pixels": metadata.get("max_pixels"),
        "processor_size": metadata.get("processor_size"),
        "feature_dim": int(metadata["feature_dim"]),
        "global_features": torch.stack([row["global_features"] for row in ordered], dim=0),
        "mask_features": torch.stack([row["mask_features"] for row in ordered], dim=0),
        "bbox_norm": torch.stack([row["bbox_norm"] for row in ordered], dim=0),
        "labels": torch.stack([row["labels"] for row in ordered], dim=0),
        "ids": expected_ids,
        "image_paths": [row["image_path"] for row in ordered],
        "grid_hw": torch.stack([row["grid_hw"] for row in ordered], dim=0),
        "resized_hw": torch.stack([row["resized_hw"] for row in ordered], dim=0),
        "actual_tokens": torch.stack([row["actual_tokens"] for row in ordered], dim=0).to(torch.int32),
        "mask_weight_sum": torch.stack([row["mask_weight_sum"] for row in ordered], dim=0).to(torch.float32),
        "mask_fallback": [row["mask_fallback"] for row in ordered],
        "selected_ids": expected_ids,
        "full_selected_ids": expected_ids,
        "num_shards": 1,
        "shard_index": 0,
        "source_num_shards": int(args.num_shards),
    }
    payload["token_stats"] = token_stats([int(v) for v in payload["actual_tokens"].tolist()])
    if args.max_visual_tokens is not None and payload["token_stats"]["max"] > args.max_visual_tokens:
        raise AssertionError(f"Token max {payload['token_stats']['max']} > {args.max_visual_tokens}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    write_json(summary_path(out_path), make_summary(payload, out_path, args.num_shards))
    print(f"merged {args.num_shards} shards into {out_path}")


if __name__ == "__main__":
    main()
