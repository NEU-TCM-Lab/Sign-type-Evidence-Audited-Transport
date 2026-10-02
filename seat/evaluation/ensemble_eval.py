from __future__ import annotations

# Combine OT readouts WITH the DINOv2 attention baseline by probability ensembling.
# Rationale: OT readouts underperform on macro-F1 but have HIGHER macro-AP (better ranking) and are
# stronger on the baseline's weak classes (Pale) -> decorrelated errors -> ensemble can beat 73.45.
# This is the OT+DINOv2 system that pushes the score, with OT as a genuine contributing member.
# No training: reconstruct each run's model, get val+test probs, average, calibrate on val, eval test.

import argparse, json
import numpy as np
import torch

from common import FEATURES_DIR, RUNS_DIR, LABELS
from metrics import find_best_thresholds, compute_metrics
from train_readout import GridDataset, grid_path
from ot_readout_head import AttrReadoutHead
from ot_sign_head import SignOTHead
from ot_gen_head import OTGenHead
from checkpoints import load_trained_model, load_run_payload
from data_validation import validate_split_payloads


def build_model(run, feature_dim, device):
    return load_trained_model(run, device, feature_dim)[0]



@torch.inference_mode()
def probs(m, pf, mw, device, bs=128):
    out = []
    for i in range(0, pf.shape[0], bs):
        o = m(pf[i:i+bs].to(device), mw[i:i+bs].to(device))
        out.append(torch.sigmoid(o).cpu())
    return torch.cat(out).numpy()


def evalp(pv, y_v, pt, y_t):
    thr = find_best_thresholds(y_v, pv, step=0.01)
    mt = compute_metrics(y_t, pt, thr)
    return mt["macro"]["f1"] * 100, {l: mt["per_label"][l]["f1"] * 100 for l in LABELS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run names to ensemble (first = anchor baseline)")
    ap.add_argument("--weights", nargs="+", type=float, default=None)
    ap.add_argument("--tag", default="dinov2grid_seed42")
    ap.add_argument("--baseline", type=float, default=73.45)
    a = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    plv = load_run_payload(a.runs[0], a.tag, "val")
    plt_ = load_run_payload(a.runs[0], a.tag, "test")
    validate_split_payloads({"val": plv, "test": plt_})
    dv, dt = GridDataset(plv), GridDataset(plt_)
    y_v, y_t = dv.y.numpy(), dt.y.numpy()
    fdim = int(plv["feature_dim"])
    w = np.array(a.weights, float) if a.weights else np.ones(len(a.runs))
    if len(w) != len(a.runs) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError("Provide one finite nonnegative weight per run, with a positive sum")
    w = w / w.sum()

    pv_each, pt_each = [], []
    print(f"baseline to beat = {a.baseline:.2f}\n{'run':28} {'w':>5} {'solo macroF1':>12} {'Pale':>6} {'Ecchy':>6}")
    for run, wi in zip(a.runs, w):
        from data_validation import alignment_order
        rv = load_run_payload(run, a.tag, "val")
        rt = load_run_payload(run, a.tag, "test")
        validate_split_payloads({"val": rv, "test": rt})
        ov, ot = alignment_order(plv, rv), alignment_order(plt_, rt)
        m = build_model(run, int(rv["feature_dim"]), device)
        pv = probs(m, rv["patch_features"][ov].float(), rv["mask_weights"][ov].float(), device)
        pt = probs(m, rt["patch_features"][ot].float(), rt["mask_weights"][ot].float(), device)
        pv_each.append(pv); pt_each.append(pt)
        mf, perc = evalp(pv, y_v, pt, y_t)
        print(f"{run:28} {wi:5.2f} {mf:12.2f} {perc['TonguePale']:6.1f} {perc['Ecchymosis']:6.1f}")

    pv_ens = sum(wi * p for wi, p in zip(w, pv_each))
    pt_ens = sum(wi * p for wi, p in zip(w, pt_each))
    mf, perc = evalp(pv_ens, y_v, pt_ens, y_t)
    print(f"\n=== ENSEMBLE ({'+'.join(a.runs)}) ===")
    print(f"macro-F1 = {mf:.2f}   vs baseline {a.baseline:.2f}  -> {'BEATS +%.2f' % (mf-a.baseline) if mf>a.baseline else 'below %.2f' % (mf-a.baseline)}")
    print("per-class: " + "  ".join(f"{l[:5]}={perc[l]:.1f}" for l in LABELS))


if __name__ == "__main__":
    main()
