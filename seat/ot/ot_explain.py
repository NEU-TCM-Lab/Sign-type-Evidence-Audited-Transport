from __future__ import annotations

# OT evidence explanation + faithfulness (deletion) test for a trained SignOTHead run.
#   1) per-label OT transport heatmaps [8,37,37] for sample images (saved as .npz, + PNG if matplotlib).
#   2) DELETION TEST: mask the top-k patches a method deems important (random / global-attention /
#      OT-transport map) by zeroing their mask_weights, re-forward, measure the logit drop. A more
#      faithful map -> larger drop. Target ordering: OT > attention > random.
# Reuses GridDataset + grid_path from train_readout, SignOTHead, common.LABELS.

import argparse, json
from pathlib import Path
import numpy as np
import torch

from common import FEATURES_DIR, RUNS_DIR, LABELS
from train_readout import GridDataset, grid_path
from ot_sign_head import SignOTHead


def load_model(run, feature_dim, device):
    s = json.loads((RUNS_DIR / run / "summary.json").read_text())
    cfg = s.get("signot") or {}
    m = SignOTHead(feature_dim=feature_dim, num_labels=len(LABELS),
                   proj_dim=cfg.get("proj_dim", 256), branch=cfg.get("branch", "both"),
                   evidence=cfg.get("evidence", "ot"), ot_k=cfg.get("ot_k", 4),
                   use_mask_mass=cfg.get("use_mask_mass", False),
                   beta_init=cfg.get("beta_init", 0.1),
                   beta_per_class=cfg.get("beta_per_class", True),
                   ot_demand=cfg.get("ot_demand", "attention"), ot_relax=cfg.get("ot_relax", "unbalanced"),
                   uot_rho=cfg.get("uot_rho", 0.1), partial_m=cfg.get("partial_m", 0.7),
                   partial_tau=cfg.get("partial_tau", 0.5), mass_gate=cfg.get("mass_gate", True)).to(device)
    sd = torch.load(RUNS_DIR / run / "best.pt", map_location=device)["model_state"]
    m.load_state_dict(sd, strict=False); m.eval()
    return m, cfg


@torch.inference_mode()
def forward_all(m, pf, mw, device, bs=128, want_heat=False):
    fins, oth, ath = [], [], []
    for i in range(0, pf.shape[0], bs):
        f, w = pf[i:i+bs].to(device), mw[i:i+bs].to(device)
        if want_heat:
            fin, info = m(f, w, return_plan=True)
            oth.append(info.get("ot_heat").flatten(2).cpu() if info.get("ot_heat") is not None else None)
            ath.append(info.get("attn_heat").flatten(2).cpu() if info.get("attn_heat") is not None else None)
        else:
            fin = m(f, w)
        fins.append(fin.cpu())
    out = torch.cat(fins)
    oth = torch.cat(oth) if want_heat and oth[0] is not None else None
    ath = torch.cat(ath) if want_heat and ath[0] is not None else None
    return out, oth, ath


def deletion_test(m, pf, mw, y, device, ks=(5, 10, 20, 40)):
    N, P = mw.shape
    base, ot_heat, attn_heat = forward_all(m, pf, mw, device, want_heat=True)   # [N,8], [N,8,P]
    rng = np.random.default_rng(0)
    rand_imp = torch.tensor(rng.random((N, len(LABELS), P)), dtype=torch.float32)
    selectors = {"random": rand_imp, "attention": attn_heat, "OT": ot_heat}
    # results[sel][k] = list of per-label mean logit drop (over positive samples)
    res = {s: {k: [] for k in ks} for s in selectors}
    for l in range(len(LABELS)):
        pos = (y[:, l] == 1)
        if pos.sum() < 5:
            continue
        for sname, imp in selectors.items():
            if imp is None:
                continue
            impl = imp[:, l, :]                          # [N,P]
            for k in ks:
                topk = impl.topk(k, dim=1).indices         # [N,k]
                mw2 = mw.clone()
                mw2.scatter_(1, topk, 0.0)
                masked, _, _ = forward_all(m, pf, mw2, device)   # [N,8]
                drop = (base[:, l] - masked[:, l])[pos]
                res[sname][k].append(float(drop.mean()))
    return res, selectors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run-name under artifacts/runs (a signot global+OT run)")
    ap.add_argument("--tag", default="dinov2grid_seed42")
    ap.add_argument("--split", default="test")
    ap.add_argument("--save-heat", type=int, default=8, help="save heatmaps for first N samples")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pl = torch.load(grid_path(a.split, a.tag, FEATURES_DIR), map_location="cpu", weights_only=False)
    ds = GridDataset(pl)
    pf, mw, y = ds.pf, ds.mw, ds.y
    feature_dim = int(pl["feature_dim"])
    m, cfg = load_model(a.run, feature_dim, device)
    out_dir = Path(a.out_dir or (RUNS_DIR / a.run / "explain")); out_dir.mkdir(parents=True, exist_ok=True)

    # --- heatmaps for a few samples ---
    if cfg.get("branch", "both") in {"ot", "both"}:
        with torch.inference_mode():
            fin, info = m(pf[:a.save_heat].to(device), mw[:a.save_heat].to(device), return_plan=True)
        np.savez(out_dir / "heatmaps.npz",
                 ot_heat=info["ot_heat"].cpu().numpy() if info.get("ot_heat") is not None else np.array([]),
                 attn_heat=info["attn_heat"].cpu().numpy() if info.get("attn_heat") is not None else np.array([]),
                 ids=np.array(list(pl["ids"])[:a.save_heat]), labels=y[:a.save_heat].numpy(),
                 probs=torch.sigmoid(fin).cpu().numpy())
        print(f"[heat] saved {a.save_heat} samples -> {out_dir/'heatmaps.npz'}")

    # --- deletion faithfulness test ---
    res, sels = deletion_test(m, pf, mw, y, device)
    ks = sorted(next(iter(res.values())).keys())
    print(f"\n=== DELETION TEST (mean logit drop over positive samples, macro over labels) — run={a.run} ===")
    print("selector   " + "  ".join(f"k={k:>3}" for k in ks) + "    AUC")
    summary = {}
    for s in ["random", "attention", "OT"]:
        if any(res[s][k] for k in ks):
            means = [float(np.mean(res[s][k])) if res[s][k] else float("nan") for k in ks]
            auc = float(np.trapz(means, ks) / (ks[-1] - ks[0]))
            summary[s] = {"per_k": dict(zip(map(str, ks), means)), "auc": auc}
            print(f"{s:10} " + "  ".join(f"{v:5.3f}" for v in means) + f"   {auc:6.3f}")
    verdict = ("OT" in summary and "attention" in summary and "random" in summary
               and summary["OT"]["auc"] > summary["attention"]["auc"] > summary["random"]["auc"])
    print(f"\nfaithfulness ordering OT > attention > random: {'PASS' if verdict else 'CHECK'}")
    json.dump({"run": a.run, "deletion": summary, "ordering_pass": verdict},
              open(out_dir / "faithfulness.json", "w"), indent=2)
    print(f"-> {out_dir/'faithfulness.json'}")


if __name__ == "__main__":
    main()
