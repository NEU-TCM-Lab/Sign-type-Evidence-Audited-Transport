"""Train pooled heads, then evaluate their validation-selected checkpoints."""
import argparse
import subprocess
import sys
from pathlib import Path
from settings import PROJECT_ROOT, FEATURES_DIR, RUNS_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--comparison-name", required=True)
    parser.add_argument("--feature-modes", nargs="+", choices=["global", "global_mask", "global_mask_bbox"], default=["global", "global_mask", "global_mask_bbox"])
    parser.add_argument("--features-dir", type=Path, default=FEATURES_DIR)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    for mode in args.feature_modes:
        name = f"{args.comparison_name}_{mode}_seed{args.seed}"
        subprocess.run([sys.executable, str(PROJECT_ROOT / "src/train_head.py"), "--tag", args.tag,
                        "--features-dir", str(args.features_dir), "--feature-mode", mode,
                        "--seed", str(args.seed), "--run-name", name], check=True)
        subprocess.run([sys.executable, str(PROJECT_ROOT / "src/evaluate_head.py"), "--tag", args.tag,
                        "--features-dir", str(args.features_dir), "--checkpoint", str(RUNS_DIR / name / "best_head.pt")], check=True)


if __name__ == "__main__":
    main()
