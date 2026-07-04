from __future__ import annotations
# Faithfulness (deletion) of the ResNet-34 Grad-CAM itself: occlude the top-k Grad-CAM patches in the
# ResNet's OWN input and measure the drop in its OWN sign logit, vs random patches. Apples-to-apples
# WITHIN the CNN. Compare the confidence-retained curve to our DINOv2 attention (deletion AUC 1.878).
import json, numpy as np, torch
from pathlib import Path
from PIL import Image
from resnet_gradcam import _build, _man, ROOT, IMEAN, ISTD

LABELS=["TonguePale","TipSideRed","Spot","Ecchymosis","Crack","Toothmark","FurThick","FurYellow"]
SIGN="Toothmark"; s=LABELS.index(SIGN); G=12; dev="cuda" if torch.cuda.is_available() else "cpu"
model=_build().to(dev)

def prep(iid):
    row=_man[iid]; img=Image.open(ROOT/row["image_path"]).convert("RGB"); W,H=img.size
    x1,y1,x2,y2=[int(round(v)) for v in row["sam2_bbox_px"]]
    px,py=int((x2-x1)*0.08),int((y2-y1)*0.08)
    crop=img.crop((max(0,x1-px),max(0,y1-py),min(W,x2+px),min(H,y2+py))).resize((384,384),Image.BILINEAR)
    a=(np.asarray(crop,np.float32)/255-IMEAN)/ISTD
    return torch.from_numpy(a).permute(2,0,1).unsqueeze(0).to(dev)

def gradcam12(x):
    feats={}
    def fwd(_m,_i,o): feats["a"]=o; o.register_hook(lambda g:feats.__setitem__("g",g))
    h=model.layer4.register_forward_hook(fwd); x=x.clone().requires_grad_(True)
    lo=model(x)[0,s]; model.zero_grad(); lo.backward(); h.remove()
    A,Gr=feats["a"][0],feats["g"][0]; w=Gr.mean((1,2))
    cam=torch.relu((w[:,None,None]*A).sum(0)).detach().cpu().numpy()
    return cam  # 12x12

ids=[iid for iid,r in _man.items() if r["labels"][s]==1][:120]
ks=[2,5,10,20]; keep_gc={k:[] for k in ks}; keep_rnd={k:[] for k in ks}
rng=np.random.default_rng(0)
mean_px=torch.tensor((0-IMEAN)/ISTD,dtype=torch.float32).view(1,3,1,1).to(dev)  # normalised mean = 0 input
for iid in ids:
    x=prep(iid)
    with torch.no_grad(): base=float(model(x)[0,s])
    if base<0: continue                                  # only where CNN predicts the sign
    cam=gradcam12(x).ravel(); order=np.argsort(cam)[::-1]
    for k in ks:
        for sel,store in [(order[:k],keep_gc),(rng.permutation(G*G)[:k],keep_rnd)]:
            xm=x.clone()
            for p in sel:
                r,c=divmod(int(p),G); xm[:,:,r*32:(r+1)*32,c*32:(c+1)*32]=mean_px
            with torch.no_grad(): lo=float(model(xm)[0,s])
            store[k].append(base-lo)
gc=[np.mean(keep_gc[k]) for k in ks]; rd=[np.mean(keep_rnd[k]) for k in ks]
auc_gc=float(np.trapz(gc,ks)/(ks[-1]-ks[0])); auc_rd=float(np.trapz(rd,ks)/(ks[-1]-ks[0]))
print(f"ResNet Grad-CAM deletion ({SIGN}, n={len(keep_gc[ks[0]])} CNN-positive):")
print("  k        "+"  ".join(f"{k:>5}" for k in ks))
print("  gradcam  "+"  ".join(f"{v:5.2f}" for v in gc))
print("  random   "+"  ".join(f"{v:5.2f}" for v in rd))
print(f"  deletion-AUC  gradcam={auc_gc:.3f}  random={auc_rd:.3f}  ratio={auc_gc/max(abs(auc_rd),1e-6):.1f}x")
json.dump({"ks":ks,"gradcam":gc,"random":rd,"auc":{"gradcam":auc_gc,"random":auc_rd}},
          open(Path(__file__).parent.parent/"outputs"/f"gradcam_deletion_{SIGN}.json","w"),indent=2)
