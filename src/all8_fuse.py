from __future__ import annotations

# Connect back to ALL 8 classes: DINOv2 softmax baseline + per-class-gated late fusion with a color
# detector. final_logit[c] = base[c] + gamma_c * color[c], gamma_c chosen per class on VAL (0 where
# color doesn't help). Reports test macro-F1 + per-class. The OT/color contribution is purely additive
# and interpretable; baseline stays intact for classes color can't help.

import argparse
import numpy as np
import torch
from common import FEATURES_DIR, LABELS
from train_readout import GridDataset, grid_path
from ensemble_eval import build_model
from checkpoints import load_run_payload
from data_validation import alignment_order, validate_split_payloads
from fusion import select_fusion, evaluate_selected

GAMMAS = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]


def _f1(y, pred):
    tp=float(((pred==1)&(y==1)).sum()); fp=float(((pred==1)&(y==0)).sum()); fn=float(((pred==0)&(y==1)).sum())
    d=2*tp+fp+fn; return 2*tp/d if d>0 else 0.0


def best_thr(y,p,step=0.01):
    if len(np.unique(y))<2: return 0.5,0.0
    bt,bf=0.5,-1
    for t in np.arange(0.05,0.95+1e-9,step):
        f=_f1(y,(p>=t).astype(int))
        if f>bf: bf,bt=f,float(t)
    return bt,bf


@torch.inference_mode()
def get_logits(run, tag, device, bs=128):
    plv=load_run_payload(run,tag,"val")
    plt_=load_run_payload(run,tag,"test")
    validate_split_payloads({"val": plv, "test": plt_})
    dv,dt=GridDataset(plv),GridDataset(plt_); m=build_model(run,int(plv["feature_dim"]),device)
    def fwd(ds):
        o=[]
        for i in range(0,ds.pf.shape[0],bs): o.append(m(ds.pf[i:i+bs].to(device),ds.mw[i:i+bs].to(device)).cpu())
        return torch.cat(o).numpy()
    return fwd(dv),dv.y.numpy(),fwd(dt),dt.y.numpy(),plv,plt_


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base-run",default="signot_global_only"); ap.add_argument("--base-tag",default="dinov2grid_seed42")
    ap.add_argument("--add-run",required=True); ap.add_argument("--add-tag",required=True)
    ap.add_argument("--margin",type=float,default=1.0,help="require this much VAL-F1 gain to accept color fusion (robust gating)")
    ap.add_argument("--fuse-classes",nargs="*",default=None,help="restrict color fusion to these a-priori classes (e.g. TonguePale)")
    a=ap.parse_args()
    dev=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    blv,yv,blt,yt,bv_payload,bt_payload=get_logits(a.base_run,a.base_tag,dev)
    alv,_,alt,_,av_payload,at_payload=get_logits(a.add_run,a.add_tag,dev)
    alv=alv[alignment_order(bv_payload,av_payload)]
    alt=alt[alignment_order(bt_payload,at_payload)]
    sig=lambda x:1/(1+np.exp(-x))
    selected=select_fusion(blv,alv,yv,a.fuse_classes,margin=a.margin)
    selected_test=evaluate_selected(blt,alt,yt,selected)
    base_f1=[]; fuse_f1=[]; gammas=[]
    print(f"{'label':>12} {'base':>6} {'fused':>6} {'gamma':>5}")
    for c,l in enumerate(LABELS):
        thb,_=best_thr(yv[:,c],sig(blv[:,c])); bf=_f1(yt[:,c],(sig(blt[:,c])>=thb).astype(int))*100
        # select gamma + threshold on VAL only (require >margin val gain over gamma=0), eval on TEST
        bg=selected[c]["gamma"]
        tf=selected_test[c]*100
        base_f1.append(bf); fuse_f1.append(tf); gammas.append(bg)
        print(f"{l:>12} {bf:6.2f} {tf:6.2f} {bg:5.1f}{'  <-color' if bg>0 else ''}")
    print(f"\nMACRO  base={np.mean(base_f1):.2f}  ->  per-class-gated fused={np.mean(fuse_f1):.2f}  "
          f"(+{np.mean(fuse_f1)-np.mean(base_f1):.2f})   [Dr.Tongue 72.87]")


if __name__=="__main__":
    main()
