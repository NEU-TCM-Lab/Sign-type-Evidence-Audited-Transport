from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm
import hashlib
from data_validation import relative_dataset_path, audit_manifest_splits

from common import (
    ARTIFACTS_DIR,
    DATASET_ROOT,
    EXPECTED_COUNTS,
    LABELS,
    LABEL_ROOT,
    MANIFEST_DIR,
    SPLIT_CSV,
    bbox_norm_xyxy,
    ensure_dirs,
    label_order_path,
    read_csv_rows,
    read_json,
    split_manifest_path,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build split manifests for frozen Qwen3-VL mask-pooling classification.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--label-root", type=Path, default=LABEL_ROOT)
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=list(SPLIT_CSV))
    parser.add_argument("--group-field", help="Patient/subject field in annotations or label CSV")
    parser.add_argument("--hash-images", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit after validation.")
    return parser.parse_args()


def load_label_rows(label_root: Path, split: str) -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    csv_path = label_root / SPLIT_CSV[split]
    rows = read_csv_rows(csv_path)
    by_path: dict[str, dict[str, str]] = {}
    duplicates = []
    for row in rows:
        image_path = str(row["image_path"]).strip()
        if image_path in by_path:
            duplicates.append(image_path)
        by_path[image_path] = row
    if duplicates:
        raise AssertionError(f"{csv_path} has duplicate image_path values: {duplicates[:5]}")
    return rows, by_path


def label_vector(row: dict[str, str], image_path: str) -> list[float]:
    values = []
    for label in LABELS:
        if label not in row:
            raise AssertionError(f"Missing label column {label} for {image_path}")
        raw = str(row[label]).strip()
        if raw not in {"0", "1", "0.0", "1.0"}:
            raise AssertionError(f"Bad label value for {image_path} {label}: {raw!r}")
        values.append(float(int(float(raw))))
    return values


def validate_split_alignment(split: str, ann_rows: list[dict], csv_rows: list[dict[str, str]]) -> dict:
    ann_paths = [str(row["original_relative_path"]) for row in ann_rows]
    csv_paths = [str(row["image_path"]).strip() for row in csv_rows]
    ann_set = set(ann_paths)
    csv_set = set(csv_paths)
    if len(ann_paths) != len(ann_set):
        raise AssertionError(f"{split} annotations contain duplicate original_relative_path values")
    if len(csv_paths) != len(csv_set):
        raise AssertionError(f"{split} CSV contains duplicate image_path values")
    if ann_set != csv_set:
        missing_in_ann = sorted(csv_set - ann_set)[:10]
        missing_in_csv = sorted(ann_set - csv_set)[:10]
        raise AssertionError(
            f"{split} annotation/CSV path sets differ; missing_in_ann={missing_in_ann}, missing_in_csv={missing_in_csv}"
        )
    expected = EXPECTED_COUNTS.get(split)
    if expected is not None and len(ann_rows) != expected:
        raise AssertionError(f"{split} count is {len(ann_rows)}, expected {expected}")
    return {
        "split": split,
        "count": len(ann_rows),
        "sets_equal": True,
        "same_order": ann_paths == csv_paths,
        "duplicate_paths": 0,
    }


def build_split(args: argparse.Namespace, split: str) -> dict:
    ann_path = args.dataset_root / "annotations" / f"{split}_pseudo.json"
    ann_rows = read_json(ann_path)
    csv_rows, labels_by_path = load_label_rows(args.label_root, split)
    summary = validate_split_alignment(split, ann_rows, csv_rows)

    out_rows = []
    iterator = ann_rows if args.limit is None else ann_rows[: args.limit]
    for ann in tqdm(iterator, desc=f"manifest:{split}"):
        original_relative_path = str(ann["original_relative_path"])
        label_row = labels_by_path[original_relative_path]

        image_rel = relative_dataset_path(ann["image_path"])
        mask_rel = relative_dataset_path(ann["pseudo_mask_path"])
        image_path = args.dataset_root / image_rel
        mask_path = args.dataset_root / mask_rel
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)

        with Image.open(image_path) as image:
            image_width, image_height = image.size
        with Image.open(mask_path) as mask:
            mask_width, mask_height = mask.size
        if (mask_width, mask_height) != (image_width, image_height):
            raise AssertionError(
                f"Mask/image size mismatch for {original_relative_path}: image={(image_width, image_height)}, mask={(mask_width, mask_height)}"
            )

        for path in (image_path, mask_path):
            if not path.resolve().is_relative_to(args.dataset_root.resolve()):
                raise ValueError("Dataset symlink escapes the dataset root")
        bbox = ann["sam2_bbox_px"]
        labels = label_vector(label_row, original_relative_path)
        row = {
                "id": str(ann["id"]),
                "split": split,
                "original_relative_path": original_relative_path,
                "image_path": str(image_rel),
                "mask_path": str(mask_rel),
                "image_width": image_width,
                "image_height": image_height,
                "qwen_bbox_px": ann["qwen_bbox_px"],
                "sam2_bbox_px": bbox,
                "bbox_norm": bbox_norm_xyxy(bbox, image_width, image_height),
                "labels": labels,
                "labels_dict": {label: labels[i] for i, label in enumerate(LABELS)},
                "label_order": LABELS,
                "pseudo_status": ann.get("pseudo_status"),
                "sam2_score": ann.get("sam2_score"),
                "mask_area": ann.get("mask_area"),
            }
        if args.group_field:
            group = ann.get(args.group_field, label_row.get(args.group_field))
            if group is None or not str(group).strip():
                raise ValueError("Patient/group field missing from an annotation")
            row["group_id"] = str(group)
        if args.hash_images:
            with image_path.open("rb") as f:
                row["image_sha256"] = hashlib.file_digest(f, "sha256").hexdigest()
        out_rows.append(row)

    out_path = split_manifest_path(split, args.manifest_dir)
    write_jsonl(out_path, out_rows)
    summary["written_count"] = len(out_rows)
    summary["manifest_path"] = str(out_path)
    return summary


def main() -> None:
    ensure_dirs()
    args = parse_args()
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    write_json(label_order_path(ARTIFACTS_DIR), LABELS)
    summaries = [build_split(args, split) for split in args.splits]
    from common import read_jsonl
    available = {s: read_jsonl(split_manifest_path(s, args.manifest_dir)) for s in SPLIT_CSV
                 if split_manifest_path(s, args.manifest_dir).is_file()}
    audit = audit_manifest_splits(available)
    write_json(args.manifest_dir / "split_audit.json", audit)
    write_json(args.manifest_dir / "summary.json", {"label_order": LABELS, "splits": summaries})
    print(f"Wrote manifests to {args.manifest_dir}")


if __name__ == "__main__":
    main()
