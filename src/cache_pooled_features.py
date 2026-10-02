from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from data_validation import identity_metadata

from common import (
    DATASET_ROOT,
    FEATURES_DIR,
    IMAGE_PROMPT,
    LABELS,
    MANIFEST_DIR,
    MODEL_PATH,
    assert_label_order,
    ensure_dirs,
    read_jsonl,
    split_feature_path,
    split_manifest_path,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cache frozen Qwen3-VL global and mask-guided pooled features.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--max-visual-tokens", type=int, default=None)
    parser.add_argument("--subset-size", type=int, default=None, help="Random subset size; requires --shuffle-seed.")
    parser.add_argument("--shuffle-seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit per split; mutually exclusive with --subset-size.")
    parser.add_argument("--tag", type=str, default=None, help="Optional output tag, e.g. mt768_train1000_seed42.")
    parser.add_argument("--smallest-first", action="store_true", help="Debug helper: process smallest images first before applying --limit.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=0, help="Save partial cache every N newly processed samples.")
    parser.add_argument("--resume", action="store_true", help="Resume from an existing final or partial cache.")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of deterministic row-order shards for this split.")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based shard index to process when --num-shards > 1.")
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def image_resampling(name: str):
    try:
        return getattr(Image.Resampling, name)
    except AttributeError:
        return getattr(Image, name)


def partial_feature_path(final_path: Path) -> Path:
    return final_path.with_name(final_path.stem + ".partial.pt")


def shard_feature_path(final_path: Path, num_shards: int, shard_index: int) -> Path:
    if num_shards <= 1:
        return final_path
    width = max(2, len(str(num_shards - 1)))
    shard_suffix = f".shard{shard_index:0{width}d}-of-{num_shards:0{width}d}"
    return final_path.with_name(final_path.stem + shard_suffix + final_path.suffix)


def summary_path(feature_path: Path, partial: bool = False) -> Path:
    suffix = ".partial.summary.json" if partial else ".summary.json"
    return feature_path.with_suffix(suffix)


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


def processor_size_kwargs(processor, max_visual_tokens: int | None) -> tuple[dict[str, Any], int | None]:
    if max_visual_tokens is None:
        return {}, None
    patch_size = int(processor.image_processor.patch_size)
    merge_size = int(processor.image_processor.merge_size)
    max_pixels = int(max_visual_tokens * (patch_size * merge_size) ** 2)
    shortest_edge = int(processor.image_processor.size["shortest_edge"])
    return {"size": {"longest_edge": max_pixels, "shortest_edge": shortest_edge}}, max_pixels


def mask_weights_for_grid(
    mask_path: Path,
    image_size: tuple[int, int],
    resized_hw: tuple[int, int],
    grid_hw: tuple[int, int],
    patch_size: int,
    merge_size: int,
) -> torch.Tensor:
    image_width, image_height = image_size
    resized_h, resized_w = resized_hw
    h_grid, w_grid = grid_hw
    kernel = patch_size * merge_size
    if resized_h != h_grid * kernel or resized_w != w_grid * kernel:
        raise AssertionError(
            f"Bad resize/grid relation: resized_hw={resized_hw}, grid_hw={grid_hw}, kernel={kernel}"
        )

    with Image.open(mask_path) as mask:
        mask = mask.convert("L")
        if mask.size != (image_width, image_height):
            raise AssertionError(f"Mask/image size mismatch for {mask_path}: mask={mask.size}, image={image_size}")
        # PIL resize takes (width, height); arrays below are [height, width].
        mask = mask.resize((resized_w, resized_h), resample=image_resampling("BILINEAR"))
        arr = np.asarray(mask, dtype=np.float32)

    if arr.max(initial=0.0) > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    mask_tensor = torch.from_numpy(arr).view(1, 1, resized_h, resized_w)
    pooled = F.avg_pool2d(mask_tensor, kernel_size=kernel, stride=kernel)
    if tuple(pooled.shape[-2:]) != (h_grid, w_grid):
        raise AssertionError(f"Mask pooled shape {tuple(pooled.shape[-2:])} != grid {(h_grid, w_grid)}")
    return pooled.squeeze(0).squeeze(0).contiguous().reshape(-1)


def load_model_and_processor(model_path: Path, dtype: torch.dtype, device: torch.device):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, processor


def select_rows(args: argparse.Namespace, split: str, patch_size: int, merge_size: int) -> list[dict[str, Any]]:
    if args.limit is not None and args.subset_size is not None:
        raise ValueError("--limit and --subset-size are mutually exclusive")
    if args.subset_size is not None and args.shuffle_seed is None:
        raise ValueError("--subset-size requires --shuffle-seed")
    if args.limit is not None and args.tag is None:
        raise ValueError("Use --tag when --limit is set so debug caches cannot overwrite full caches.")
    if args.subset_size is not None and args.tag is None:
        raise ValueError("Use --tag when --subset-size is set so subset caches cannot overwrite full caches.")

    rows = read_jsonl(split_manifest_path(split, args.manifest_dir))
    if args.smallest_first:
        factor = patch_size * merge_size
        rows = sorted(
            rows,
            key=lambda row: (
                ((int(row["image_height"]) + factor - 1) // factor)
                * ((int(row["image_width"]) + factor - 1) // factor)
            ),
        )
    if args.subset_size is not None:
        rows = rows[:]
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(rows)
        rows = rows[: args.subset_size]
    if args.limit is not None:
        rows = rows[: args.limit]
    return rows


def empty_accumulator(split: str, args: argparse.Namespace, patch_size: int, merge_size: int, max_pixels: int | None) -> dict[str, Any]:
    return {
        "split": split,
        "label_order": LABELS,
        "model_path": str(args.model_path),
        "image_prompt": IMAGE_PROMPT,
        "patch_size": patch_size,
        "merge_size": merge_size,
        "max_visual_tokens": args.max_visual_tokens,
        "max_pixels": max_pixels,
        "processor_size": None,
        "feature_dim": 0,
        "global_features": [],
        "mask_features": [],
        "bbox_norm": [],
        "labels": [],
        "ids": [],
        "image_paths": [],
        "grid_hw": [],
        "resized_hw": [],
        "actual_tokens": [],
        "mask_weight_sum": [],
        "mask_fallback": [],
        "selected_ids": [],
        "full_selected_ids": [],
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
    }


def unpack_payload(payload: dict[str, Any]) -> dict[str, Any]:
    unpacked = payload.copy()
    for key in ("global_features", "mask_features", "bbox_norm", "labels", "grid_hw", "resized_hw"):
        value = unpacked.get(key, [])
        if isinstance(value, torch.Tensor):
            unpacked[key] = [row.cpu() for row in value]
    if isinstance(unpacked.get("mask_weight_sum"), torch.Tensor):
        unpacked["mask_weight_sum"] = [float(v) for v in unpacked["mask_weight_sum"].cpu().tolist()]
    if isinstance(unpacked.get("actual_tokens"), torch.Tensor):
        unpacked["actual_tokens"] = [int(v) for v in unpacked["actual_tokens"].cpu().tolist()]
    return unpacked


def finalize_payload(acc: dict[str, Any]) -> dict[str, Any]:
    payload = acc.copy()
    n = len(payload["ids"])
    feature_dim = int(payload["global_features"][0].numel()) if n else 0
    payload["feature_dim"] = feature_dim
    payload["global_features"] = torch.stack(payload["global_features"], dim=0) if n else torch.empty(0, 2560, dtype=torch.float16)
    payload["mask_features"] = torch.stack(payload["mask_features"], dim=0) if n else torch.empty(0, 2560, dtype=torch.float16)
    payload["bbox_norm"] = torch.stack(payload["bbox_norm"], dim=0) if n else torch.empty(0, 9)
    payload["labels"] = torch.stack(payload["labels"], dim=0) if n else torch.empty(0, len(LABELS))
    payload["grid_hw"] = torch.stack(payload["grid_hw"], dim=0) if n else torch.empty(0, 2, dtype=torch.int32)
    payload["resized_hw"] = torch.stack(payload["resized_hw"], dim=0) if n else torch.empty(0, 2, dtype=torch.int32)
    payload["actual_tokens"] = torch.tensor(payload["actual_tokens"], dtype=torch.int32)
    payload["mask_weight_sum"] = torch.tensor(payload["mask_weight_sum"], dtype=torch.float32)
    payload["token_stats"] = token_stats(payload["actual_tokens"].tolist())
    return payload


def make_summary(payload: dict[str, Any], out_path: Path, selected_count: int, partial: bool, elapsed_sec: float | None = None) -> dict[str, Any]:
    actual_tokens = payload["actual_tokens"].tolist() if isinstance(payload["actual_tokens"], torch.Tensor) else payload["actual_tokens"]
    summary = {
        "split": payload["split"],
        "path": str(out_path),
        "partial": partial,
        "count": int(len(payload["ids"])),
        "selected_count": int(selected_count),
        "feature_dim": int(payload["feature_dim"]),
        "fallback_count": int(sum(bool(v) for v in payload["mask_fallback"])),
        "label_order": LABELS,
        "patch_size": int(payload["patch_size"]),
        "merge_size": int(payload["merge_size"]),
        "max_visual_tokens": payload.get("max_visual_tokens"),
        "max_pixels": payload.get("max_pixels"),
        "token_stats": token_stats([int(v) for v in actual_tokens]),
        "selected_ids": payload.get("selected_ids", []),
        "full_selected_count": len(payload.get("full_selected_ids", [])),
        "num_shards": int(payload.get("num_shards", 1)),
        "shard_index": int(payload.get("shard_index", 0)),
    }
    if elapsed_sec is not None:
        summary["elapsed_sec"] = elapsed_sec
        summary["images_per_sec"] = len(payload["ids"]) / elapsed_sec if elapsed_sec > 0 else None
    return summary


def save_payload(acc: dict[str, Any], out_path: Path, selected_count: int, partial: bool, elapsed_sec: float | None = None) -> None:
    save_path = partial_feature_path(out_path) if partial else out_path
    payload = finalize_payload(acc)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, save_path)
    write_json(summary_path(save_path, partial=False), make_summary(payload, save_path, selected_count, partial, elapsed_sec))


def load_resume_accumulator(
    args: argparse.Namespace,
    split: str,
    out_path: Path,
    selected_ids: list[str],
    patch_size: int,
    merge_size: int,
    max_pixels: int | None,
) -> tuple[dict[str, Any], bool]:
    final_exists = out_path.exists()
    partial_path = partial_feature_path(out_path)
    if args.overwrite:
        return empty_accumulator(split, args, patch_size, merge_size, max_pixels), False
    if final_exists:
        if args.resume:
            payload = torch.load(out_path, map_location="cpu", weights_only=False)
            if list(payload.get("selected_ids", payload.get("ids", []))) != selected_ids:
                raise AssertionError(f"Existing final cache selected ids differ: {out_path}")
            return unpack_payload(payload), True
        raise FileExistsError(f"{out_path} exists; pass --overwrite or --resume")
    if args.resume and partial_path.exists():
        payload = torch.load(partial_path, map_location="cpu", weights_only=False)
        existing_selected = list(payload.get("selected_ids", selected_ids))
        if existing_selected != selected_ids:
            raise AssertionError(f"Existing partial cache selected ids differ: {partial_path}")
        return unpack_payload(payload), False
    return empty_accumulator(split, args, patch_size, merge_size, max_pixels), False


def extract_batch(
    args: argparse.Namespace,
    batch_rows: list[dict[str, Any]],
    model,
    processor,
    device: torch.device,
    processor_kwargs: dict[str, Any],
    patch_size: int,
    merge_size: int,
    image_token_id: int,
) -> dict[str, list[Any]]:
    images = []
    image_sizes = []
    for row in batch_rows:
        assert_label_order(row["label_order"])
        image_path = args.dataset_root / row["image_path"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_size = image.size
            if image_size != (int(row["image_width"]), int(row["image_height"])):
                raise AssertionError(f"Manifest image size mismatch for {image_path}: {image_size} vs manifest")
            images.append(image.copy())
            image_sizes.append(image_size)

    inputs = processor(
        text=[IMAGE_PROMPT] * len(images),
        images=images,
        padding=True,
        return_tensors="pt",
        **processor_kwargs,
    )
    image_grid_thw_cpu = inputs["image_grid_thw"]
    image_grid_thw = image_grid_thw_cpu.to(device)
    pixel_values = inputs["pixel_values"].to(device)
    prompt_image_tokens = (inputs["input_ids"] == image_token_id).sum(dim=1).cpu().tolist()

    expected_tokens_per_image = []
    grid_hw_per_image = []
    resized_hw_per_image = []
    for row, thw, prompt_tokens in zip(batch_rows, image_grid_thw_cpu.tolist(), prompt_image_tokens):
        t_raw, h_raw, w_raw = [int(v) for v in thw]
        if t_raw != 1:
            raise AssertionError(f"Expected image T=1, got image_grid_thw={thw} for {row['image_path']}")
        if h_raw % merge_size != 0 or w_raw % merge_size != 0:
            raise AssertionError(f"Raw grid not divisible by merge size: grid={thw}, merge={merge_size}")
        h_grid = h_raw // merge_size
        w_grid = w_raw // merge_size
        actual_tokens = h_grid * w_grid
        expected_tokens = t_raw * h_raw * w_raw // (merge_size * merge_size)
        if actual_tokens != expected_tokens:
            raise AssertionError(f"Token formula mismatch for {row['image_path']}: {actual_tokens} vs {expected_tokens}")
        if args.max_visual_tokens is not None and actual_tokens > args.max_visual_tokens:
            raise AssertionError(
                f"{row['image_path']} has actual_tokens={actual_tokens} > max_visual_tokens={args.max_visual_tokens}"
            )
        if int(prompt_tokens) != actual_tokens:
            raise AssertionError(
                f"Prompt image token count mismatch for {row['image_path']}: prompt={prompt_tokens}, actual={actual_tokens}, grid={thw}"
            )
        expected_tokens_per_image.append(actual_tokens)
        grid_hw_per_image.append((h_grid, w_grid))
        resized_hw_per_image.append((h_raw * patch_size, w_raw * patch_size))

    with torch.inference_mode():
        image_embeds, _deepstack = model.get_image_features(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
    if len(image_embeds) != len(batch_rows):
        raise AssertionError(f"Expected {len(batch_rows)} image embedding chunks, got {len(image_embeds)}")

    out: dict[str, list[Any]] = {
        "global_features": [],
        "mask_features": [],
        "bbox_norm": [],
        "labels": [],
        "ids": [],
        "image_paths": [],
        "grid_hw": [],
        "resized_hw": [],
        "actual_tokens": [],
        "mask_weight_sum": [],
        "mask_fallback": [],
    }
    for row, image_size, tokens, expected_tokens, grid_hw, resized_hw in zip(
        batch_rows, image_sizes, image_embeds, expected_tokens_per_image, grid_hw_per_image, resized_hw_per_image
    ):
        if tokens.ndim != 2:
            raise AssertionError(f"Expected [tokens, dim], got {tuple(tokens.shape)} for {row['image_path']}")
        if int(tokens.shape[0]) != expected_tokens:
            raise AssertionError(
                f"Feature token grid mismatch for {row['image_path']}: features={tokens.shape[0]}, expected={expected_tokens}"
            )
        tokens_fp32 = tokens.float()
        global_feat = tokens_fp32.mean(dim=0)
        mask_path = args.dataset_root / row["mask_path"]
        weights = mask_weights_for_grid(
            mask_path=mask_path,
            image_size=image_size,
            resized_hw=resized_hw,
            grid_hw=grid_hw,
            patch_size=patch_size,
            merge_size=merge_size,
        )
        if int(weights.numel()) != expected_tokens:
            raise AssertionError(f"Mask weights length {weights.numel()} != tokens {expected_tokens} for {row['image_path']}")
        weight_sum = float(weights.sum().item())
        fallback = not np.isfinite(weight_sum) or weight_sum <= 0.0
        if fallback:
            mask_feat = global_feat
        else:
            weights_dev = weights.to(device=device, dtype=torch.float32)
            mask_feat = (tokens_fp32 * weights_dev[:, None]).sum(dim=0) / weights_dev.sum()

        feature_output_dtype = getattr(args, "feature_output_dtype", "float16")
        if feature_output_dtype == "float32":
            out["global_features"].append(global_feat.cpu())
            out["mask_features"].append(mask_feat.cpu())
        elif feature_output_dtype == "float16":
            out["global_features"].append(global_feat.to(dtype=torch.float16).cpu())
            out["mask_features"].append(mask_feat.to(dtype=torch.float16).cpu())
        else:
            raise ValueError(f"Unsupported feature_output_dtype: {feature_output_dtype}")
        out["bbox_norm"].append(torch.tensor(row["bbox_norm"], dtype=torch.float32))
        out["labels"].append(torch.tensor(row["labels"], dtype=torch.float32))
        out["ids"].append(row["id"])
        out["image_paths"].append(row["image_path"])
        out["grid_hw"].append(torch.tensor(grid_hw, dtype=torch.int32))
        out["resized_hw"].append(torch.tensor(resized_hw, dtype=torch.int32))
        out["actual_tokens"].append(int(expected_tokens))
        out["mask_weight_sum"].append(weight_sum)
        out["mask_fallback"].append(bool(fallback))
    return out


def extend_accumulator(acc: dict[str, Any], batch_out: dict[str, list[Any]]) -> None:
    for key, values in batch_out.items():
        acc[key].extend(values)


def extract_split(args: argparse.Namespace, split: str, model, processor, device: torch.device) -> dict:
    final_out_path = split_feature_path(split, tag=args.tag, features_dir=args.features_dir)
    out_path = shard_feature_path(final_out_path, args.num_shards, args.shard_index)
    patch_size = int(processor.image_processor.patch_size)
    merge_size = int(processor.image_processor.merge_size)
    image_token_id = int(model.config.image_token_id)
    processor_kwargs, max_pixels = processor_size_kwargs(processor, args.max_visual_tokens)

    full_rows = select_rows(args, split, patch_size, merge_size)
    full_selected_ids = [row["id"] for row in full_rows]
    rows = full_rows[args.shard_index :: args.num_shards]
    identity_by_id = {row["id"]: row for row in rows}
    selected_ids = [row["id"] for row in rows]
    acc, already_complete = load_resume_accumulator(args, split, out_path, selected_ids, patch_size, merge_size, max_pixels)
    acc["selected_ids"] = selected_ids
    acc["full_selected_ids"] = full_selected_ids
    acc["num_shards"] = int(args.num_shards)
    acc["shard_index"] = int(args.shard_index)
    acc["processor_size"] = processor_kwargs.get("size")
    if already_complete and len(acc["ids"]) == len(rows):
        payload = finalize_payload(acc)
        summary = make_summary(payload, out_path, len(rows), partial=False, elapsed_sec=0.0)
        write_json(summary_path(out_path), summary)
        print(f"{split}: final cache already complete at {out_path}")
        return summary

    completed_ids = set(acc["ids"])
    if len(completed_ids) != len(acc["ids"]):
        raise AssertionError(f"Resume cache contains duplicate ids for {split}")
    remaining = [row for row in rows if row["id"] not in completed_ids]
    start_time = time.perf_counter()
    newly_done = 0

    for start in tqdm(range(0, len(remaining), args.batch_size), desc=f"cache:{split}"):
        batch_rows = remaining[start : start + args.batch_size]
        batch_out = extract_batch(
            args=args,
            batch_rows=batch_rows,
            model=model,
            processor=processor,
            device=device,
            processor_kwargs=processor_kwargs,
            patch_size=patch_size,
            merge_size=merge_size,
            image_token_id=image_token_id,
        )
        extend_accumulator(acc, batch_out)
        newly_done += len(batch_rows)
        if args.save_every and newly_done % args.save_every == 0:
            save_payload(acc, out_path, len(rows), partial=True, elapsed_sec=time.perf_counter() - start_time)

    if len(acc["ids"]) != len(rows):
        raise AssertionError(f"{split} cache incomplete: {len(acc['ids'])} / {len(rows)}")
    if len(set(acc["ids"])) != len(acc["ids"]):
        raise AssertionError(f"{split} cache has duplicate ids")
    payload = finalize_payload(acc)
    payload.update(identity_metadata([identity_by_id[sample_id] for sample_id in payload["ids"]]))
    summary = make_summary(payload, out_path, len(rows), partial=False, elapsed_sec=time.perf_counter() - start_time)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    write_json(summary_path(out_path), summary)
    return summary


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")
    args.features_dir.mkdir(parents=True, exist_ok=True)
    dtype = resolve_dtype(args.dtype)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_path} on {device} with dtype={dtype}")
    model, processor = load_model_and_processor(args.model_path, dtype=dtype, device=device)
    summaries = [extract_split(args, split, model, processor, device) for split in args.splits]
    write_json(args.features_dir / (f"cache_summary_{args.tag}.json" if args.tag else "cache_summary.json"), summaries)
    print(f"Wrote pooled features to {args.features_dir}")


if __name__ == "__main__":
    main()
