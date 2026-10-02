"""Check tracked release files for credentials and sample-level artifacts."""
import argparse
import csv
import json
import re
import subprocess
from pathlib import Path
from settings import PROJECT_ROOT

SECRET_PATTERNS = [re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),
                   re.compile(r"github_pat_[A-Za-z0-9_]{40,}"),
                   re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
                   re.compile(r"\bAKIA[0-9A-Z]{16}\b")]
PRIVATE_KEYS = {"patient_id", "subject_id", "sample_id", "image_path", "image_paths", "ids", "predictions"}


def private_json_keys(value):
    if isinstance(value, dict):
        return any(key.lower() in PRIVATE_KEYS or private_json_keys(item) for key, item in value.items())
    if isinstance(value, list): return any(private_json_keys(item) for item in value)
    return False


def scan_files(root: Path, names: list[str]) -> list[dict]:
    findings = []
    for name in names:
        path = root / name
        if not path.is_file(): continue
        if path.suffix.lower() in {".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".jsonl", ".jpg", ".jpeg", ".png", ".zip"} or path.name.startswith(".env"):
            findings.append({"file": name, "kind": "private_or_unreviewed_artifact"})
            continue
        try: text = path.read_text(encoding="utf-8-sig")
        except (UnicodeError, OSError): continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            findings.append({"file": name, "kind": "credential_pattern"})
        if path.suffix.lower() == ".json" and private_json_keys(json.loads(text)):
            findings.append({"file": name, "kind": "sample_level_json"})
        if path.suffix.lower() == ".csv":
            header = next(csv.reader(text.splitlines()), [])
            if any(key.lower() in PRIVATE_KEYS for key in header):
                findings.append({"file": name, "kind": "sample_level_csv"})
    return findings


def scan_history(root: Path) -> tuple[int, list[dict]]:
    """Read Git objects without exporting credential values or extracting archives."""
    revisions = subprocess.check_output(["git", "rev-list", "--all"], cwd=root).decode().splitlines()
    findings = []
    for revision in revisions:
        listing = subprocess.check_output(["git", "ls-tree", "-r", "-z", revision], cwd=root)
        entries = []
        for entry in listing.split(b"\0"):
            if not entry: continue
            metadata, name = entry.split(b"\t", 1)
            mode, kind, object_id = metadata.split()
            if kind == b"blob": entries.append((object_id.decode(), name.decode("utf-8")))
        objects = subprocess.check_output(["git", "cat-file", "--batch"], cwd=root,
                                          input="".join(object_id + "\n" for object_id, _ in entries).encode())
        offset = 0
        for _, name in entries:
            end = objects.index(b"\n", offset)
            size = int(objects[offset:end].split()[-1]); start = end + 1
            data = objects[start:start + size]; offset = start + size + 1
            suffix = Path(name).suffix.lower()
            kind = None
            if suffix in {".pt", ".pth", ".ckpt", ".safetensors", ".npy", ".npz", ".jsonl", ".jpg", ".jpeg", ".png", ".zip"} or Path(name).name.startswith(".env"):
                kind = "private_or_unreviewed_artifact"
            else:
                try: text = data.decode("utf-8-sig")
                except UnicodeError: continue
                if any(pattern.search(text) for pattern in SECRET_PATTERNS): kind = "credential_pattern"
                if suffix == ".json" and private_json_keys(json.loads(text)): kind = "sample_level_json"
                if suffix == ".csv" and any(key.lower() in PRIVATE_KEYS for key in next(csv.reader(text.splitlines()), [])):
                    kind = "sample_level_csv"
            if kind: findings.append({"revision": revision, "file": name, "kind": kind})
    return len(revisions), findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", action="store_true", help="Also inspect all locally available Git revisions")
    args = parser.parse_args()
    files = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=PROJECT_ROOT).decode("utf-8").split("\0")
    findings = scan_files(PROJECT_ROOT, [name for name in files if name])
    revisions, history_findings = scan_history(PROJECT_ROOT) if args.history else (0, [])
    print(json.dumps({"files_checked": len([name for name in files if name]), "findings": findings,
                      "revisions_checked": revisions, "history_findings": history_findings,
                      "limits": "Heuristic release scan; does not prove absence of all PII or contamination in unavailable datasets."}, indent=2))
    if findings or history_findings: raise SystemExit(1)


if __name__ == "__main__":
    main()
