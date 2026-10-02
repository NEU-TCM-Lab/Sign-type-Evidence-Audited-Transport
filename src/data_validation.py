"""Checks for split contamination and safe cache alignment; no training dependency."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
import numpy as np
from settings import LABELS


def relative_dataset_path(value: str) -> Path:
    text = str(value).replace("\\", "/")
    path = PurePosixPath(text)
    if not text or path.is_absolute() or re.match(r"^[A-Za-z]:", text) or ".." in path.parts:
        raise ValueError("Dataset image/mask paths must be relative and stay inside the dataset root")
    return Path(*path.parts)


def _unique(values, context):
    values = [str(x) for x in values]
    if any(not x.strip() or x == "None" for x in values):
        raise ValueError(f"{context}: missing sample identity")
    if len(values) != len(set(values)):
        raise ValueError(f"{context}: duplicate sample identities")
    return values


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def alignment_order(reference: dict, other: dict, *, id_key="ids", label_key="labels") -> list[int]:
    if list(reference.get("label_order", LABELS)) != LABELS or list(other.get("label_order", LABELS)) != LABELS:
        raise ValueError("Cache label order mismatch")
    ref = _unique(reference[id_key], "reference cache")
    ids = _unique(other[id_key], "additional cache")
    if set(ref) != set(ids):
        raise ValueError("Cache sample ID sets differ")
    index = {sample_id: i for i, sample_id in enumerate(ids)}
    order = [index[x] for x in ref]
    if label_key in reference and label_key in other:
        if not np.array_equal(_array(reference[label_key]), _array(other[label_key])[order]):
            raise ValueError("Aligned caches disagree on ground-truth labels")
    return order


def validate_split_payloads(payloads: dict[str, dict]) -> dict:
    checked = []
    for split, payload in payloads.items():
        if payload.get("split", split) != split:
            raise ValueError(f"{split}: cache split metadata mismatch")
        if list(payload["label_order"]) != LABELS:
            raise ValueError(f"{split}: label order mismatch")
        ids = _unique(payload["ids"], split)
        if tuple(payload["labels"].shape) != (len(ids), len(LABELS)):
            raise ValueError(f"{split}: cache label/identity shape mismatch")
        labels = _array(payload["labels"])
        if not np.isfinite(labels).all() or not np.isin(labels, (0, 1)).all():
            raise ValueError(f"{split}: expected finite binary labels")
        if "patch_features" in payload:
            shape = tuple(payload["patch_features"].shape)
            if len(shape) != 3 or shape[0] != len(ids) or tuple(payload["mask_weights"].shape) != shape[:2]:
                raise ValueError(f"{split}: patch features and mask weights disagree")
            if shape[-1] != payload["feature_dim"]:
                raise ValueError(f"{split}: feature dimension metadata mismatch")
    for key in ("geometry", "feature_dim"):
        values = {p[key] for p in payloads.values() if key in p}
        if len(values) > 1:
            raise ValueError(f"Splits have different {key}")
    for key in ("ids", "image_paths", "original_relative_paths", "image_sha256", "group_ids"):
        available = [(s, p[key]) for s, p in payloads.items() if key in p]
        if len(available) < 2:
            continue
        seen = set()
        for split, values in available:
            if len(values) != len(payloads[split]["ids"]):
                raise ValueError(f"{split}: {key} length mismatch")
            # Multiple images of one subject within a split are allowed.
            canonical = {str(v).replace("\\", "/") for v in values}
            if key != "group_ids" and len(canonical) != len(values):
                raise ValueError(f"{split}: duplicate identities in {key}")
            if any(not v.strip() or v == "None" for v in canonical):
                raise ValueError(f"{split}: missing identities in {key}")
            if seen & canonical:
                raise ValueError(f"Cross-split overlap detected in {key} ({split})")
            seen |= canonical
        checked.append(key)
    return {"checked_identity_fields": checked,
            "group_metadata_complete": all("group_ids" in p for p in payloads.values()),
            "image_hash_metadata_complete": all("image_sha256" in p for p in payloads.values())}


def identity_metadata(rows: list[dict]) -> dict:
    result = {"image_paths": [r["image_path"] for r in rows]}
    for field, key in (("original_relative_path", "original_relative_paths"),
                       ("image_sha256", "image_sha256"), ("group_id", "group_ids")):
        if rows and all(r.get(field) is not None for r in rows):
            result[key] = [str(r[field]) for r in rows]
    return result


def audit_manifest_splits(manifests: dict[str, list[dict]], dataset_root: Path | None = None,
                          hash_images: bool = False, group_field: str | None = None) -> dict:
    seen = {k: set() for k in ("id", "image_path", "original_relative_path", "image_sha256", "group_id")}
    counts = {}
    grouped = 0
    hashed = 0
    for split, rows in manifests.items():
        if not rows:
            raise ValueError(f"{split}: empty manifest")
        _unique([r["id"] for r in rows], split)
        current = {k: set() for k in seen}
        for row in rows:
            if row.get("split", split) != split or list(row["label_order"]) != LABELS:
                raise ValueError(f"{split}: manifest metadata mismatch")
            if len(row["labels"]) != len(LABELS) or any(v not in (0, 1, 0.0, 1.0) for v in row["labels"]):
                raise ValueError(f"{split}: invalid binary labels")
            for field in ("image_path", "mask_path"):
                relative_dataset_path(row[field])
            for field in ("id", "image_path", "original_relative_path", "image_sha256"):
                if field == "image_sha256" and hash_images:
                    continue
                if row.get(field) is not None:
                    value = str(row[field]).replace("\\", "/")
                    if value in current[field]:
                        raise ValueError(f"{split}: duplicate {field}")
                    current[field].add(value)
            group = row.get(group_field) if group_field else row.get("group_id")
            if group_field and (group is None or not str(group).strip()):
                raise ValueError(f"{split}: required group field is missing")
            if group is not None:
                current["group_id"].add(str(group)); grouped += 1
            if hash_images:
                if dataset_root is None:
                    raise ValueError("Image hashing requires --dataset-root")
                path = (dataset_root / relative_dataset_path(row["image_path"])).resolve()
                if not path.is_relative_to(dataset_root.resolve()):
                    raise ValueError("Image symlink escapes the dataset root")
                with path.open("rb") as f:
                    digest = hashlib.file_digest(f, "sha256").hexdigest()
                if digest in current["image_sha256"]:
                    raise ValueError(f"{split}: duplicate image content")
                current["image_sha256"].add(digest); hashed += 1
        for field in seen:
            if seen[field] & current[field]:
                raise ValueError(f"Cross-split overlap detected in {field} ({split})")
            seen[field] |= current[field]
        counts[split] = len(rows)
    return {"split_counts": counts, "cross_split_overlap": False,
            "image_hashes_checked": hashed, "group_metadata_complete": grouped == sum(counts.values()),
            "limits": "No patient-level conclusion without complete group metadata; byte hashes do not detect near-duplicates."}
