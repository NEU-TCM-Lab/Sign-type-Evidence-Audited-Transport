from __future__ import annotations
# Give co-occurrence its STRONGEST shot: a learned structured corrector (mini-CRF / C-Tran-style
# label message-passing) over the generative model's 8 calibrated per-sign probs. Learns ARBITRARY
# pairwise label dependencies (incl. NEGATIVE / conditional) that fixed-Jaccard rescoring missed.
#   logit_out_i = a_i * logit(p_i) + b_i + sum_j W_ij * (p_j - 0.5)
# Fit on val (BCE + weight decay), tune per-class thresholds on val, eval test.
# Compare: independent baseline vs Jaccard-init vs learned-from-scratch. If learned ALSO fails to
# beat independent -> airtight proof label dependencies are exhausted (not a crude-method artifact).

import argparse
import numpy as np
import torch
from torch import nn
from sklearn.metrics import f1_score

LABELS = ["TonguePale", "TipSideRed", "Spot", "Ecchymosis", "Crack", "Toothmark", "FurThick", "FurYellow"]


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


class Corrector(nn.Module):
    def __init__(self, W_init=None, no_offdiag=False):
        super().__init__()
        self.a = nn.Parameter(torch.ones(8))
        self.b = nn.Parameter(torch.zeros(8))
        W0 = torch.zeros(8, 8) if W_init is None else torch.tensor(W_init, dtype=torch.float32)
        self.W = nn.Parameter(W0)
        self.no_offdiag = no_offdiag

    def forward(self, P):                       # P: [N,8] probs
        Lg = torch.log(P.clamp(1e-4, 1 - 1e-4) / (1 - P).clamp(1e-4, 1 - 1e-4))
        msg = (P - 0.5) @ self.W.t()
        if self.no_offdiag:
            msg = msg * 0
        return self.a * Lg + self.b + msg       # output logits


def tune_thr(Y, prob):
    thr = np.zeros(8)
    for i in range(8):
        best, bt = -1, 0.5
        for t in np.linspace(0.05, 0.95, 91):
            f = f1_score(Y[:, i], (prob[:, i] >= t).astype(int), zero_division=0)
            if f > best: best, bt = f, t
        thr[i] = bt
    return thr


def macro(Y, prob, thr):
    return float(np.mean([f1_score(Y[:, i], (prob[:, i] >= thr[i]).astype(int), zero_division=0) for i in range(8)]))


def fit_eval(Pv, Yv, Pt, Yt, W_init, wd, no_off=False, epochs=400):
    m = Corrector(W_init=W_init, no_offdiag=no_off)
    opt = torch.optim.Adam(m.parameters(), lr=0.05, weight_decay=wd)
    lossf = nn.BCEWithLogitsLoss()
    Pvt, Yvt = torch.tensor(Pv, dtype=torch.float32), torch.tensor(Yv, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(m(Pvt), Yvt); loss.backward(); opt.step()
    with torch.no_grad():
        pv = torch.sigmoid(m(Pvt)).numpy()
        pt = torch.sigmoid(m(torch.tensor(Pt, dtype=torch.float32))).numpy()
    thr = tune_thr(Yv, pv)
    return macro(Yt, pt, thr), pt, thr, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    args = ap.parse_args()
    d = np.load(args.npz)
    Pv, Yv, Pt, Yt, C = d["Pv"], d["Yv"], d["Pt"], d["Yt"], d["C"]

    # independent baseline (no structure)
    thr0 = tune_thr(Yv, Pv)
    base = macro(Yt, Pt, thr0)

    results = {"independent": base}
    cands = [("learned_scratch", None, 1e-3, False),
             ("learned_jaccard_init", C, 1e-3, False),
             ("learned_strong_reg", None, 1e-2, False),
             ("learned_weak_reg", None, 1e-4, False)]
    best = ("independent", base, None, thr0)
    for name, Wi, wd, no in cands:
        mac, pt, thr, model = fit_eval(Pv, Yv, Pt, Yt, Wi, wd, no)
        results[name] = mac
        if mac > best[1]:
            best = (name, mac, model, thr)

    print("=== structured corrector (test MacroF1) ===")
    for k, v in results.items():
        print(f"  {k:24s}: {v*100:.2f}")
    print(f"BEST: {best[0]} = {best[1]*100:.2f}  (independent baseline {base*100:.2f}, Δ={ (best[1]-base)*100:+.2f})")
    # per-class of best
    if best[2] is not None:
        with torch.no_grad():
            pt = torch.sigmoid(best[2](torch.tensor(Pt, dtype=torch.float32))).numpy()
        pc = {LABELS[i]: round(f1_score(Yt[:, i], (pt[:, i] >= best[3][i]).astype(int), zero_division=0) * 100, 2) for i in range(8)}
        print("per-class (best):", pc)
        W = best[2].W.detach().numpy()
        print("learned W (label->label, neg=anti-corr):")
        for i in range(8):
            row = " ".join(f"{W[i,j]:+.2f}" for j in range(8))
            print(f"  {LABELS[i]:11s} {row}")


if __name__ == "__main__":
    main()
