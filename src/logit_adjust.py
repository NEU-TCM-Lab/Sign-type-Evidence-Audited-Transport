from __future__ import annotations
# Plan-1: post-hoc Logit Adjustment (Menon et al., ICLR'21) on G6 per-sign calibrated logprobs.
# logit'_i = logit(p_i) + delta_i ; decide by sign(logit') (i.e. threshold 0.5 on adjusted prob).
# Variants: (a) prior-based delta = -tau*log(prior) [1 param tau]; (b) constrained delta = a+b*log(prior)
# [2 params]; (c) free per-class margin = plan-1 threshold [8 params]. Tune all on val, eval test.
import argparse
import numpy as np
from sklearn.metrics import f1_score
from scipy.special import logit as _logit, expit

LABELS = ["TonguePale", "TipSideRed", "Spot", "Ecchymosis", "Crack", "Toothmark", "FurThick", "FurYellow"]
DESC = ["淡白", "舌尖红", "点刺", "瘀斑", "裂纹", "齿痕", "厚苔", "黄苔"]


def logit(p):
    return _logit(np.clip(p, 1e-4, 1 - 1e-4))


def macro(Y, score, thr=0.0):
    pred = (score >= thr).astype(int)
    return np.mean([f1_score(Y[:, i], pred[:, i], zero_division=0) for i in range(8)])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--npz", required=True); a = ap.parse_args()
    d = np.load(a.npz); Pv, Yv, Pt, Yt = d["Pv"], d["Yv"], d["Pt"], d["Yt"]
    Lv, Lt = logit(Pv), logit(Pt)
    prior = Yv.mean(0)                                  # class frequency
    lp = np.log(np.clip(prior, 1e-4, 1))

    # (a) prior-based, tune tau
    best_a = (-1, 0)
    for tau in np.linspace(-2, 2, 81):
        s = macro(Yv, Lv - tau * lp)
        if s > best_a[0]: best_a = (s, tau)
    tau = best_a[1]; mac_a = macro(Yt, Lt - tau * lp)

    # (b) constrained delta = c + b*lp, grid c,b
    best_b = (-1, 0, 0)
    for c in np.linspace(-2, 2, 41):
        for b in np.linspace(-2, 2, 41):
            s = macro(Yv, Lv + (c + b * lp))
            if s > best_b[0]: best_b = (s, c, b)
    _, c, bb = best_b; mac_b = macro(Yt, Lt + (c + bb * lp))

    # (c) free per-class margin (= per-class threshold tuned on val)
    delta = np.zeros(8)
    for i in range(8):
        best = (-1, 0)
        for dd in np.linspace(-4, 4, 161):
            f = f1_score(Yv[:, i], (Lv[:, i] + dd >= 0).astype(int), zero_division=0)
            if f > best[0]: best = (f, dd)
        delta[i] = best[1]
    pred_c = (Lt + delta >= 0).astype(int)
    mac_c = np.mean([f1_score(Yt[:, i], pred_c[:, i], zero_division=0) for i in range(8)])

    base = macro(Yt, Lt)                                 # no adjustment (threshold 0.5)
    print("=== Logit Adjustment on G6 probs (test MacroF1) ===")
    print(f"  no-adjust(0.5)          : {base*100:.2f}")
    print(f"  (a) prior tau={tau:+.2f}      : {mac_a*100:.2f}")
    print(f"  (b) c={c:+.2f} b={bb:+.2f}      : {mac_b*100:.2f}")
    print(f"  (c) free per-class margin: {mac_c*100:.2f}")
    print(f"  G6 hard output ref       : 70.80")
    print("  per-class free-margin F1:", {DESC[i]: round(f1_score(Yt[:, i], pred_c[:, i], zero_division=0)*100, 1) for i in range(8)})


if __name__ == "__main__":
    main()
