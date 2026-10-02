from __future__ import annotations

# Faithfulness (deletion) test for the COLOR-OT pale-tongue detector's evidence map.
# For TonguePale-positive test images, mask the top-k patches the color-OT evidence heatmap highlights
# (zero their mask_weights), re-forward, and measure the drop in the Pale logit. A faithful / load-bearing
# evidence map -> large drop vs. masking random patches. This validates that the color evidence is
# spatially concentrated and causally drives the pale decision (interpretability that "stands firm").

import argparse, json
import numpy as np
import torch
from common import FEATURES_DIR, RUNS_DIR, LABELS
from train_readout import GridDataset, grid_path
from ot_gen_head import OTGenHead
from checkpoints import load_trained_model

PALE = LABELS.index("TonguePale")


def load_color(run, device):
    model, config = load_trained_model(run, device)
    if config["readout"] != "otgen":
        raise ValueError("Color faithfulness requires an otgen checkpoint")
    return model



@torch.inference_mode()
def pale_logit(m, pf, mw, device, bs=128):
    out = []
    for i in range(0, pf.shape[0], bs):
        out.append(m(pf[i:i+bs].to(device), mw[i:i+bs].to(device))[:, PALE].cpu())
    return torch.cat(out).numpy()


@torch.inference_mode()
def pale_heat(m, pf, mw, device, bs=128):
    hs = []
    for i in range(0, pf.shape[0], bs):
        _, info = m(pf[i:i+bs].to(device), mw[i:i+bs].to(device), return_plan=True)
        hs.append(info["ot_heat"][:, PALE].flatten(1).cpu())   # [b, N]
    return torch.cat(hs).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="color_pale"); ap.add_argument("--tag", default="color_seed42")
    ap.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 40])
    a = ap.parse_args()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pl = torch.load(grid_path("test", a.tag, FEATURES_DIR), map_location="cpu", weights_only=False)
    ds = GridDataset(pl); pf, mw, y = ds.pf, ds.mw, ds.y.numpy()
    m = load_color(a.run, dev)
    pos = y[:, PALE] == 1
    base = pale_logit(m, pf, mw, dev)
    heat = pale_heat(m, pf, mw, dev)                                # [N,P] color-OT evidence
    rng = np.random.default_rng(0)
    rand = rng.random(heat.shape)
    res = {"color-OT": [], "random": []}
    print(f"=== COLOR-OT pale evidence deletion test ({pos.sum()} Pale+ test images) ===")
    print(f"{'k':>4} {'drop(color-OT)':>15} {'drop(random)':>13}")
    for k in a.ks:
        for name, imp in [("color-OT", heat), ("random", rand)]:
            topk = torch.tensor(imp).topk(k, dim=1).indices
            mw2 = mw.clone(); mw2.scatter_(1, topk, 0.0)
            masked = pale_logit(m, pf, mw2, dev)
            res[name].append(float((base[pos] - masked[pos]).mean()))
        print(f"{k:>4} {res['color-OT'][-1]:>15.3f} {res['random'][-1]:>13.3f}")
    auc = {n: float(np.trapezoid(v, a.ks) / (a.ks[-1] - a.ks[0])) for n, v in res.items()}
    print(f"\ndeletion-AUC  color-OT={auc['color-OT']:.3f}  random={auc['random']:.3f}  "
          f"ratio={auc['color-OT']/max(auc['random'],1e-6):.1f}x")
    print(f"VERDICT: color evidence is {'LOAD-BEARING / faithful' if auc['color-OT']>3*max(auc['random'],1e-3) else 'weakly faithful'}")
    out = RUNS_DIR / a.run / "explain"; out.mkdir(parents=True, exist_ok=True)
    json.dump({"per_k": res, "ks": a.ks, "auc": auc}, open(out / "color_faithfulness.json", "w"), indent=2)
    print(f"-> {out/'color_faithfulness.json'}")


if __name__ == "__main__":
    main()
