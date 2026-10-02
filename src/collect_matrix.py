"""Collect fixed-run metrics without selecting runs by test performance."""
import argparse
import csv
import json
from pathlib import Path
from settings import RUNS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-glob", required=True)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for run in sorted(args.runs_dir.glob(args.runs_glob)):
        path = run / "metrics_test.json"
        if not path.is_file(): raise FileNotFoundError(f"Run has no completed test metrics: {run.name}")
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"run": run.name, **{k: metrics["macro"].get(k) for k in ("f1", "auroc", "ap")}})
    if not rows: raise ValueError("No completed runs match the pattern")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "f1", "auroc", "ap"])
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
