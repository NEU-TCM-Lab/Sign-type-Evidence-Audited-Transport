from __future__ import annotations

# Three decisive reviews on whether the OT branch carries ANY value over the global attention readout.
#   1) Post-hoc fusion oracle: final = global + alpha*OT, sweep alpha on VAL-calibrated thresholds,
#      eval on TEST. If no alpha beats the global baseline -> OT logits carry no useful predictive info.
#   2) OT-vs-attention heatmap similarity (cosine / Spearman / Jensen-Shannon), per class.
#   3) Per-class analysis: per-class oracle alpha + per-class F1 gain table -> is OT useful for ANY class?
#
# Uses the trained SignOTHead's return_plan outputs (global_logit, ot_logit, ot_heat, attn_heat).
# Reuses metrics.find_best_thresholds / compute_metrics and ot_explain.load_model.

import argparse, json
from pathlib import Path
import numpy as np
import torch

from common import FEATURES_DIR, RUNS_DIR, LABELS
from metrics import find_best_thresholds, compute_metrics
from train_readout import GridDataset, grid_path
from ot_explain import load_model
from checkpoints import load_run_payload

ALPHAS = [-1.0, -0.5, 0.0, 0.1, 0.2, 0.5, 1.0]


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


@torch.inference_mode()
def collect(m, pf, mw, device, bs=128):
    gl, ol, oh, ah = [], [], [], []
    for i in range(0, pf.shape[0], bs):
        f, w = pf[i:i+bs].to(device), mw[i:i+bs].to(device)
        _, info = m(f, w, return_plan=True)
        gl.append(info["global_logit"].cpu()); ol.append(info["ot_logit"].cpu())
        oh.append(info["ot_heat"].flatten(2).cpu()); ah.append(info["attn_heat"].flatten(2).cpu())
    return (torch.cat(gl).numpy(), torch.cat(ol).numpy(),
            torch.cat(oh).numpy(), torch.cat(ah).numpy())   # [N,8],[N,8],[N,8,P],[N,8,P]


def macro_and_perclass(gl_v, ol_v, y_v, gl_t, ol_t, y_t, alpha, step=0.01):
    pv = sig(gl_v + alpha * ol_v); pt = sig(gl_t + alpha * ol_t)
    thr = find_best_thresholds(y_v, pv, step=step)
    mt = compute_metrics(y_t, pt, thr)
    perc = {l: mt["per_label"][l]["f1"] * 100 for l in LABELS}
    return mt["macro"]["f1"] * 100, perc


def _binary_f1(y, pred):
    tp = float(((pred == 1) & (y == 1)).sum()); fp = float(((pred == 1) & (y == 0)).sum())
    fn = float(((pred == 0) & (y == 1)).sum())
    denom = 2*tp + fp + fn
    return 2*tp / denom if denom > 0 else 0.0


def best_thr_f1(y, p, step=0.01):
    # per-single-class threshold maximizing binary F1; returns (best_thr, best_f1)
    if len(np.unique(y)) < 2:
        return 0.5, 0.0
    cand = np.unique(np.concatenate([np.arange(0.05, 0.95 + 1e-9, step), [0.5]]))
    bt, bf = 0.5, -1.0
    for t in cand:
        f = _binary_f1(y, (p >= t).astype(np.int64))
        if f > bf:
            bf, bt = f, float(t)
    return bt, bf


def spearman(a, b):
    ra = a.argsort().argsort().astype(np.float64); rb = b.argsort().argsort().astype(np.float64)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra*ra).sum() * (rb*rb).sum())
    return float((ra*rb).sum() / d) if d > 0 else 0.0


def js_div(p, q):
    p = p / max(p.sum(), 1e-12); q = q / max(q.sum(), 1e-12); m = 0.5*(p+q)
    def kl(a, b):
        mask = a > 0
        return float((a[mask] * np.log(a[mask] / np.clip(b[mask], 1e-12, None))).sum())
    return 0.5*kl(p, m) + 0.5*kl(q, m)   # in nats, 0..ln2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", default="dinov2grid_seed42")
    ap.add_argument("--baseline", type=float, default=73.45, help="global-only DINOv2 calibrated macro-F1 to beat")
    a = ap.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    plv = load_run_payload(a.run, a.tag, "val")
    plt_ = load_run_payload(a.run, a.tag, "test")
    dv, dt = GridDataset(plv), GridDataset(plt_)
    feature_dim = int(plv["feature_dim"])
    m, cfg = load_model(a.run, feature_dim, device)
    print(f"run={a.run}  cfg: evidence={cfg.get('evidence')} relax={cfg.get('ot_relax')} "
          f"demand={cfg.get('ot_demand')} rho={cfg.get('uot_rho')} mass_gate={cfg.get('mass_gate')} "
          f"frozen={cfg.get('freeze_global')}")

    gl_v, ol_v, _, _ = collect(m, dv.pf, dv.mw, device)
    gl_t, ol_t, oh_t, ah_t = collect(m, dt.pf, dt.mw, device)
    y_v, y_t = dv.y.numpy(), dt.y.numpy()

    # ---------- Review 1: post-hoc fusion oracle ----------
    print(f"\n=== [1] VALIDATION-SELECTED POST-HOC FUSION  final = global + alpha*OT  (VAL-calibrated -> TEST macro-F1) ===")
    print(f"baseline (global-only DINOv2) = {a.baseline:.2f}")
    print(f"{'alpha':>6} | {'test macroF1':>12} | {'vs baseline':>11}")
    macro_by_alpha = {}; perc_by_alpha = {}
    for al in ALPHAS:
        mf, perc = macro_and_perclass(gl_v, ol_v, y_v, gl_t, ol_t, y_t, al)
        macro_by_alpha[al] = mf; perc_by_alpha[al] = perc
        tag = "  <- alpha=0 (global only of this run)" if al == 0 else ""
        print(f"{al:>6.2f} | {mf:>12.2f} | {mf-a.baseline:>+11.2f}{tag}")
    val_macro_by_alpha = {al: float(np.mean([best_thr_f1(y_v[:, li], sig(gl_v + al*ol_v)[:, li])[1]
                              for li in range(len(LABELS))])) for al in ALPHAS}
    best_al = max(val_macro_by_alpha, key=val_macro_by_alpha.get)
    beat = macro_by_alpha[best_al] > a.baseline
    print(f"best alpha={best_al} -> {macro_by_alpha[best_al]:.2f}  "
          f"{'BEATS' if beat else 'does NOT beat'} baseline {a.baseline:.2f}  "
          f"=> OT logits {'carry useful info' if beat else 'carry NO useful predictive info beyond global'}")

    # ---------- Review 2: OT vs attention heatmap similarity ----------
    print(f"\n=== [2] OT-vs-ATTENTION HEATMAP SIMILARITY (mean over test samples, per class) ===")
    print(f"{'label':>12} | {'cosine':>7} | {'spearman':>8} | {'JS(nats)':>8}")
    sims = {}
    for li, lab in enumerate(LABELS):
        cos_l, sp_l, js_l = [], [], []
        for n in range(oh_t.shape[0]):
            o = oh_t[n, li]; h = ah_t[n, li]
            if o.sum() <= 0 or h.sum() <= 0:
                continue
            cos_l.append(float(o @ h / (np.linalg.norm(o)*np.linalg.norm(h) + 1e-12)))
            sp_l.append(spearman(o, h)); js_l.append(js_div(o, h))
        sims[lab] = (float(np.mean(cos_l)), float(np.mean(sp_l)), float(np.mean(js_l)))
        print(f"{lab:>12} | {sims[lab][0]:>7.3f} | {sims[lab][1]:>8.3f} | {sims[lab][2]:>8.4f}")
    mc = np.mean([v[0] for v in sims.values()]); ms = np.mean([v[1] for v in sims.values()]); mj = np.mean([v[2] for v in sims.values()])
    print(f"{'MEAN':>12} | {mc:>7.3f} | {ms:>8.3f} | {mj:>8.4f}   (cos~1 & JS~0 => OT just copies attention)")

    # ---------- Review 3: per-class analysis + per-class oracle ----------
    print(f"\n=== [3] PER-CLASS ANALYSIS: does OT help ANY class? ===")
    base_perc = perc_by_alpha[0.0]                          # alpha=0 per-class F1 (this run's global)
    # per-class oracle: choose alpha per class to maximize that class VAL F1, report TEST F1
    pv = {al: sig(gl_v + al*ol_v) for al in ALPHAS}; pt = {al: sig(gl_t + al*ol_t) for al in ALPHAS}
    print(f"{'label':>12} | {'global(a=0)':>11} | {'bestAlpha':>9} | {'OT-fused':>8} | {'gain':>6}")
    pcg_perc = {}; any_help = []
    for li, lab in enumerate(LABELS):
        best_a, best_f1 = 0.0, -1.0
        for al in ALPHAS:
            _, f1v = best_thr_f1(y_v[:, li], pv[al][:, li])
            if f1v > best_f1:
                best_f1, best_a = f1v, al
        thr_t, _ = best_thr_f1(y_v[:, li], pv[best_a][:, li])
        f1t = _binary_f1(y_t[:, li], (pt[best_a][:, li] >= thr_t).astype(np.int64)) * 100
        pcg_perc[lab] = f1t; gain = f1t - base_perc[lab]
        if best_a != 0.0 and gain > 0.05:
            any_help.append((lab, best_a, gain))
        print(f"{lab:>12} | {base_perc[lab]:>11.2f} | {best_a:>9.2f} | {f1t:>8.2f} | {gain:>+6.2f}")
    pcg_macro = float(np.mean(list(pcg_perc.values())))
    print(f"\nper-class-gated ORACLE macro-F1 (upper bound) = {pcg_macro:.2f}  vs baseline {a.baseline:.2f}")
    if any_help:
        print("OT helps these classes (oracle): " + ", ".join(f"{l}(a={al:+.1f},+{g:.2f})" for l, al, g in any_help))
        print("=> per-class gated fusion is worth trying IF the gated-oracle macro beats baseline.")
    else:
        print("OT helps NO class under per-class oracle => STOP the OT classification route.")

    out = {"run": a.run, "review1_macro_by_alpha": macro_by_alpha, "best_alpha": best_al, "selected_on": "validation",
           "beats_baseline": bool(beat), "review2_similarity_mean": {"cosine": mc, "spearman": ms, "js": mj},
           "review2_per_class": sims, "review3_perclass_gated_oracle_macro": pcg_macro,
           "review3_classes_helped": any_help}
    od = RUNS_DIR / a.run / "review"; od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / "ot_review.json", "w"), indent=2)
    print(f"\n-> {od/'ot_review.json'}")


if __name__ == "__main__":
    main()
