import argparse
import json
from pathlib import Path
import numpy as np
from fusion import validate_prediction_dump, aligned_logits, select_fusion, evaluate_selected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--add", type=Path, required=True)
    parser.add_argument("--fuse-classes", nargs="*")
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()
    def load(path):
        with np.load(path, allow_pickle=False) as f: dump = dict(f)
        validate_prediction_dump(dump)
        return dump
    base, other = load(args.base), load(args.add)
    val, test = aligned_logits(base, other, "val"), aligned_logits(base, other, "test")
    selected = select_fusion(base["val_logits"], val, base["val_labels"], args.fuse_classes, margin=args.margin)
    scores = evaluate_selected(base["test_logits"], test, base["test_labels"], selected)
    result = {"selected_on": "validation", "test_macro_f1": float(np.mean(scores)),
              "classes": [{**cfg, "test_f1": score} for cfg, score in zip(selected, scores)]}
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
