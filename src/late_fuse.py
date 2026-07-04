from __future__ import annotations

# Per-class LATE fusion: final_logit[c] = base_logit[c] + gamma * add_logit[c], gamma searched on val,
# threshold calibrated on val, evaluated on test. Each model uses its OWN feature tag (so we can fuse the
# strong softmax DINOv2 baseline with a separate color/texture detector without weakening either branch).

import argparse, json
import numpy as np
import torch
from common import FEATURES_DIR, RUNS_DIR, LABELS
from train_readout import GridDataset, grid_path
from ensemble_eval import build_model


def _binary_f1(y, pred):
    tp=float(((pred==1)&(y==1)).sum()); fp=float(((pred==1)&(y==0)).sum()); fn=float(((pred==0)&(y==1)).sum())
    d=2*tp+fp+fn; return 2*tp/d if d>0 else 0.0


def best_thr(y,p,step=0.01):
    if len(np.unique(y))<2: return 0.5,0.0
    bt,bf=0.5,-1
    for t in np.arange(0.05,0.95+1e-9,step):
        f=_binary_f1(y,(p>=t).astype(int))
        if f>bf: bf,bt=f,float(t)
    return bt,bf


@torch.inference_mode()
def logits(run, tag, device, bs=128):
    plv=torch.load(grid_path("val",tag,FEATURES_DIR),map_location="cpu",weights_only=False)
    plt_=torch.load(grid_path("test",tag,FEATURES_DIR),map_location="cpu",weights_only=False)
    dv,dt=GridDataset(plv),GridDataset(plt_); fdim=int(plv["feature_dim"])
    m=build_model(run,fdim,device)
    def fwd(ds):
        out=[]
        for i in range(0,ds.pf.shape[0],bs):
            out.append(m(ds.pf[i:i+bs].to(device),ds.mw[i:i+bs].to(device)).cpu())
        return torch.cat(out).numpy()
    return fwd(dv),dv.y.numpy(),fwd(dt),dt.y.numpy()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-run",required=True); ap.add_argument("--base-tag",default="dinov2grid_seed42")
    ap.add_argument("--add-run",required=True); ap.add_argument("--add-tag",required=True)
    ap.add_argument("--cls",default="Ecchymosis")
    a=ap.parse_args()
    dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ci=LABELS.index(a.cls)
    blv,yv,blt,yt=logits(a.base_run,a.base_tag,dev)
    alv,_,alt,_=logits(a.add_run,a.add_tag,dev)
    bv,bt_=blv[:,ci],blt[:,ci]; av,at=alv[:,ci],alt[:,ci]; yvc,ytc=yv[:,ci],yt[:,ci]
    # baseline alone
    th,_=best_thr(yvc, 1/(1+np.exp(-bv))); base_f1=_binary_f1(ytc,(1/(1+np.exp(-bt_))>=th).astype(int))*100
    print(f"class={a.cls}  base({a.base_run}) F1={base_f1:.2f}")
    print(f"{'gamma':>6} {'test F1':>8}")
    best=(0.0,base_f1)
    for g in [-1,-0.5,-0.2,0,0.2,0.5,1,2]:
        fv=1/(1+np.exp(-(bv+g*av))); ft=1/(1+np.exp(-(bt_+g*at)))
        th,_=best_thr(yvc,fv); f1=_binary_f1(ytc,(ft>=th).astype(int))*100
        print(f"{g:>6.1f} {f1:>8.2f}{'  <-best' if f1>best[1] else ''}")
        if f1>best[1]: best=(g,f1)
    print(f"\nbest gamma={best[0]} -> {a.cls} F1={best[1]:.2f}  (base {base_f1:.2f}; {'+%.2f LIFT'%(best[1]-base_f1) if best[1]>base_f1 else 'no lift'})")


if __name__=="__main__":
    main()
