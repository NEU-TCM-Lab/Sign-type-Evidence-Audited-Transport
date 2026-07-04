from __future__ import annotations

# STRONGER color-counterfactual audit for the color-OT pale detector.
# Instead of one random sign vector, we enumerate ALL 2^3 sign combinations on the (L,a,S) channels
# within the tongue mask. The clinical "un-pale" direction is (L-, a+, S+) = (-1,+1,+1); the other 7
# sign vectors are matched non-clinical controls of equal per-channel magnitude. We report the
# directed counterfactual-AUC vs. the mean +/- std over the 7 controls, and whether the directed
# effect exceeds every individual control. This is the version reported in the paper.
# color channels: [L, a, b, H, S, V, Lstd, Sstd]

import argparse, json, itertools
import numpy as np
import torch
from common import FEATURES_DIR, RUNS_DIR
from train_readout import GridDataset, grid_path
from color_faithfulness import load_color, pale_logit, PALE

iL, ia, iS = 0, 1, 4
CLINICAL = (-1, +1, +1)   # un-pale: darker L, more red a, more saturated S


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="color_pale"); ap.add_argument("--tag", default="color_seed42")
    ap.add_argument("--deltas", type=float, nargs="+", default=[0.05, 0.1, 0.2, 0.3])
    a = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pl = torch.load(grid_path("test", a.tag, FEATURES_DIR), map_location="cpu", weights_only=False)
    ds = GridDataset(pl); pf, mw, y = ds.pf.float(), ds.mw, ds.y.numpy()
    m = load_color(a.run, dev)
    pos = y[:, PALE] == 1
    base = pale_logit(m, pf, mw, dev)
    inm = (mw > 0).float()
    dl = a.deltas

    def auc(drops):
        return float(np.trapezoid(drops, dl) / (dl[-1] - dl[0]))

    def direction_auc(sign):
        drops = []
        for d in dl:
            p = pf.clone()
            p[..., iL] += d * sign[0] * inm; p[..., ia] += d * sign[1] * inm; p[..., iS] += d * sign[2] * inm
            drops.append(float((base[pos] - pale_logit(m, p, mw, dev)[pos]).mean()))
        return auc(drops), drops

    def perimg_auc(sign):   # per-image AUC over Pale-positive images
        drops = []
        for d in dl:
            p = pf.clone()
            p[..., iL] += d * sign[0] * inm; p[..., ia] += d * sign[1] * inm; p[..., iS] += d * sign[2] * inm
            drops.append((base - pale_logit(m, p, mw, dev))[pos])
        return np.trapezoid(np.stack(drops, 0), dl, axis=0) / (dl[-1] - dl[0])

    clin_auc, clin_drops = direction_auc(CLINICAL)
    controls = {}
    control_drops = {}
    for s in itertools.product([-1, 1], repeat=3):
        if s == CLINICAL:
            continue
        key = "".join("+" if v > 0 else "-" for v in s)
        auc_s, drops_s = direction_auc(s)
        controls[key] = auc_s
        control_drops[key] = drops_s
    cvals = list(controls.values())

    # image-level bootstrap 95% CI of (directed - mean-control) per-image AUC difference
    dir_pi = perimg_auc(CLINICAL)
    ctrl_pi = np.mean(np.stack([perimg_auc(s) for s in itertools.product([-1, 1], repeat=3)
                                if s != CLINICAL], 0), 0)
    delta_pi = dir_pi - ctrl_pi
    rng = np.random.default_rng(0); n = len(delta_pi)
    boot = [float(np.mean(delta_pi[rng.integers(0, n, n)])) for _ in range(10000)]
    ci_lo, ci_hi = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    out = {
        "deltas": dl,
        "clinical_dir": "L-,a+,S+",
        "clinical_auc": clin_auc,
        "clinical_drops": clin_drops,
        "control_aucs": controls,
        "control_drops": control_drops,
        "control_mean": float(np.mean(cvals)),
        "control_std": float(np.std(cvals)),
        "directed_minus_control_mean": clin_auc - float(np.mean(cvals)),
        "directed_exceeds_all_controls": bool(clin_auc > max(cvals)),
        "delta_bootstrap_ci95": [ci_lo, ci_hi],
        "n_pale_pos": int(pos.sum()),
    }
    print(f"image-level Delta 95% bootstrap CI = [{ci_lo:.3f}, {ci_hi:.3f}]")
    print(f"clinical AUC = {clin_auc:.3f}  ({pos.sum()} Pale+ images)")
    print(f"7 controls: " + ", ".join(f"{k}={v:.3f}" for k, v in controls.items()))
    print(f"control mean +/- std = {out['control_mean']:.3f} +/- {out['control_std']:.3f}")
    print(f"directed - control_mean = {out['directed_minus_control_mean']:.3f}")
    print(f"directed exceeds all 7 controls: {out['directed_exceeds_all_controls']}")
    d = RUNS_DIR / a.run / "explain"; d.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(d / "color_counterfactual_multi.json", "w"), indent=2)
    print(f"-> {d/'color_counterfactual_multi.json'}")


if __name__ == "__main__":
    main()
