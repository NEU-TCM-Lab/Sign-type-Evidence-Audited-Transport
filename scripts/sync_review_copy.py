"""Keep the non-package seat/ review copy identical to runnable src/ modules."""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NEW_ROLES = {
    "core": ["settings.py", "data_validation.py", "checkpoints.py", "fusion.py"],
    "cache": ["geometry.py"],
    "evaluation": ["fuse_npz.py", "collect_matrix.py", "audit_dataset.py", "audit_release.py"],
    "training": ["run_experiments.py"],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    paths = {path for path in (ROOT / "seat").rglob("*.py")}
    paths |= {ROOT / "seat" / role / name for role, names in NEW_ROLES.items() for name in names}
    for destination in sorted(paths):
        source = ROOT / "src" / destination.name
        if args.check:
            if not destination.is_file() or source.read_bytes() != destination.read_bytes():
                raise SystemExit(f"Review copy differs: {destination.relative_to(ROOT)}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
    print(f"{'Checked' if args.check else 'Synchronized'} {len(paths)} review modules")


if __name__ == "__main__":
    main()
