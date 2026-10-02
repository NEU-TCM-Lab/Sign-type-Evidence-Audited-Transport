from __future__ import annotations

# CLINICAL COLOR COUNTERFACTUAL faithfulness test for the color-OT pale detector.
# Pale tongue is a HOLISTIC color property (lighter L, less red -a, less saturated -S), so patch-deletion
# is the wrong probe. Instead we perturb the tongue-region color along the clinically-defined pale axis:
#   "make less pale" = L-=d (darker), a+=d (more red), S+=d (more saturated)  [opposite of the pale direction]
# and measure the drop in the Pale logit. If the directed clinical perturbation drops the pale confidence
# far more than a random-channel perturbation of equal magnitude, the model FAITHFULLY grounds its pale
# decision in the interpretable color axis (a causal, clinically-meaningful explanation).
# color channels: [L, a, b, H, S, V, Lstd, Sstd]

import argparse, json
import numpy as np
import torch
from common import FEATURES_DIR, RUNS_DIR, LABELS
from train_readout import GridDataset, grid_path
from color_faithfulness import load_color, pale_logit, PALE

# indices into the 8-ch color feature
iL, ia, iS = 0, 1, 4


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
    inm = (mw > 0).float().unsqueeze(-1)                       # perturb tongue-region patches only
    rng = np.random.default_rng(0)
    print(f"=== COLOR COUNTERFACTUAL on Pale ({pos.sum()} Pale+ test images) ===")
    print(f"{'delta':>6} {'drop(un-pale dir)':>18} {'drop(random dir)':>17}")
    res = {"clinical": [], "random": []}
    for d in a.deltas:
        # clinical un-pale direction
        pf_c = pf.clone()
        pf_c[..., iL] -= d * inm.squeeze(-1); pf_c[..., ia] += d * inm.squeeze(-1); pf_c[..., iS] += d * inm.squeeze(-1)
        drop_c = float((base[pos] - pale_logit(m, pf_c, mw, dev)[pos]).mean())
        # random direction of equal per-channel magnitude on the same 3 channels
        signs = torch.tensor(rng.choice([-1.0, 1.0], size=(1, 1, 3)), dtype=torch.float32)
        pf_r = pf.clone()
        for j, ch in enumerate([iL, ia, iS]):
            pf_r[..., ch] += d * signs[0, 0, j] * inm.squeeze(-1)
        drop_r = float((base[pos] - pale_logit(m, pf_r, mw, dev)[pos]).mean())
        res["clinical"].append(drop_c); res["random"].append(drop_r)
        print(f"{d:>6.2f} {drop_c:>18.3f} {drop_r:>17.3f}")
    auc = {n: float(np.trapezoid(v, a.deltas) / (a.deltas[-1] - a.deltas[0])) for n, v in res.items()}
    print(f"\ncounterfactual-AUC  clinical={auc['clinical']:.3f}  random={auc['random']:.3f}  "
          f"ratio={auc['clinical']/max(abs(auc['random']),1e-6):.1f}x")
    print(f"VERDICT: pale decision is {'FAITHFULLY grounded in the clinical color axis' if auc['clinical']>2*abs(auc['random']) and auc['clinical']>0 else 'not clearly color-grounded'}")
    out = RUNS_DIR / a.run / "explain"; out.mkdir(parents=True, exist_ok=True)
    json.dump({"deltas": a.deltas, "drops": res, "auc": auc}, open(out / "color_counterfactual.json", "w"), indent=2)
    print(f"-> {out/'color_counterfactual.json'}")


if __name__ == "__main__":
    main()
