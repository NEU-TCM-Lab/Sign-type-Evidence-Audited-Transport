"""Audit local manifests without exporting sample identifiers or patient data."""
import argparse
import json
from pathlib import Path
from settings import DATASET_ROOT, MANIFEST_DIR
from data_validation import audit_manifest_splits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--hash-images", action="store_true", help="Check exact image duplicates across splits")
    parser.add_argument("--group-field", help="Patient/subject identity field present in every manifest row")
    args = parser.parse_args()
    manifests = {}
    for split in ("train", "val", "test"):
        path = args.manifest_dir / f"{split}.jsonl"
        with path.open(encoding="utf-8") as f:
            manifests[split] = [json.loads(line) for line in f if line.strip()]
    print(json.dumps(audit_manifest_splits(manifests, args.dataset_root, args.hash_images, args.group_field), indent=2))


if __name__ == "__main__":
    main()
