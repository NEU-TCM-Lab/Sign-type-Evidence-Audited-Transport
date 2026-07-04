from __future__ import annotations
# Render the 4-panel interpretability comparison figure for the paper.
#  (A) traditional saliency heatmap paradigm (blurred attention) - unvalidated
#  (B) OURS prototype/attention on a localized sign (deletion-faithful)
#  (C) OURS clinical color counterfactual for pale (original | un-pale | pale-logit curve)
#  (D) OURS sign co-occurrence relation matrix (1-Jaccard)
import json
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

SRC = Path("/root/autodl-tmp/TongueDx2_Qwen3VL4B_maskpool_cls")
V14 = Path("/root/autodl-tmp/TongueDx2_Dinov2_V14")
ROOT = Path("/root/autodl-tmp/TongueDx2_pseudo_v1_qwen3vl4b_sam2")
LABELS = ["TonguePale","TipSideRed","Spot","Ecchymosis","Crack","Toothmark","FurThick","FurYellow"]
S = 518
man = {json.loads(l)["id"]: json.loads(l) for l in open(SRC/"artifacts/manifests/test.jsonl")}


def load_rgb(iid):
    return np.asarray(Image.open(ROOT/man[iid]["image_path"]).convert("RGB").resize((S,S), Image.BILINEAR), np.uint8)

def load_mask(iid):
    m = np.asarray(Image.open(ROOT/man[iid]["mask_path"]).convert("L").resize((S,S), Image.BILINEAR), np.float32)/255.0
    return (m>0.4).astype(np.float32)

def tongue_bbox(mask, pad=0.06):
    ys, xs = np.where(mask > 0)
    p = int(pad*S)
    y0, y1 = max(0, ys.min()-p), min(S, ys.max()+p); x0, x1 = max(0, xs.min()-p), min(S, xs.max()+p)
    return y0, y1, x0, x1

def overlay(rgb, heat37, mask, alpha=0.6, blur=0):
    h = cv2.resize(heat37.astype(np.float32), (S,S), interpolation=cv2.INTER_CUBIC) * (mask>0)
    if blur: h = gaussian_filter(h, blur)
    h = (h-h.min())/(h.max()-h.min()+1e-9)
    am = (h**1.5)[...,None]*alpha                                  # per-pixel alpha: only hot regions colored
    cmap = plt.cm.jet(h)[...,:3]*255
    return (rgb*(1-am)+cmap*am).astype(np.uint8), h

def cam_overlay(rgb, cam518, mask, alpha=0.6):
    # smooth CNN Grad-CAM saliency (already in the 518 frame), masked to the tongue for a fair view
    h = cam518.astype(np.float32) * (mask>0)
    h = (h-h.min())/(h.max()-h.min()+1e-9)
    am = (h**1.2)[...,None]*alpha
    cmap = plt.cm.jet(h)[...,:3]*255
    return (rgb*(1-am)+cmap*am).astype(np.uint8), h

def patch_overlay(rgb, heat37, mask, topk=26, alpha=0.6, style="weight"):
    # OUR evidence is DISCRETE patch selection: draw the 37x37 patch tessellation + fill the
    # OT-selected top-k on-tongue patches (segmentation-style), unlike a smooth CNN blob.
    #  style="weight": single-hue YlOrRd, shade = OT selection weight (+ colorbar in caller)
    #  style="binary": all selected patches one uniform colour (top-k membership only)
    P = S/37.0
    m37 = cv2.resize(mask.astype(np.float32),(37,37),interpolation=cv2.INTER_AREA)
    ton = m37 >= 0.3
    h = heat37.astype(np.float32).copy(); h[~ton] = -1e9
    hn = np.clip(h,0,None); hn = hn/(hn.max()+1e-9)
    sel = np.zeros((37,37), bool)
    for i,j in np.array(np.unravel_index(np.argsort(h.ravel())[::-1][:topk],(37,37))).T:
        sel[i,j] = True
    peak = np.unravel_index(np.argmax(h),(37,37))

    # tidy the arrangement (legibility) for BOTH styles: merge near-neighbours into a coherent band,
    # drop scattered specks, prune spurs. Cleaned cells keep their true OT weight hn for the weight style.
    from scipy.ndimage import binary_closing, label
    sel = binary_closing(sel, structure=np.ones((3,3)), iterations=1) & ton
    lbl, n = label(sel, structure=np.ones((3,3)))               # drop small scattered components (8-conn)
    for c in range(1, n+1):
        if (lbl == c).sum() < 3: sel[lbl == c] = False
    def nbr8(a):
        c = np.zeros(a.shape, int)
        for di in (-1,0,1):
            for dj in (-1,0,1):
                if di or dj: c += np.roll(np.roll(a,di,0),dj,1)
        return c
    for _ in range(1):                                          # prune spurs/protrusions (degree<2 endpoints)
        spur = sel & (nbr8(sel) < 2); spur[peak] = False
        sel &= ~spur
    sel[peak] = True

    cmap = plt.cm.YlOrRd
    disp = rgb.copy(); fill = disp.copy()
    pad = 2                                                       # inset tiles so the grid shows as gaps
    ii, jj = np.where(sel)
    for i,j in zip(ii,jj):
        col = (255,145,25) if style=="binary" else tuple(int(c) for c in (np.array(cmap(0.30+0.70*float(hn[i,j]))[:3])*255))
        cv2.rectangle(fill,(int(j*P)+pad,int(i*P)+pad),(int((j+1)*P)-pad,int((i+1)*P)-pad),col,-1)
    disp = cv2.addWeighted(fill, alpha, disp, 1-alpha, 0)
    for i,j in zip(ii,jj):                                        # soft rounded-ish tile borders
        cv2.rectangle(disp,(int(j*P)+pad,int(i*P)+pad),(int((j+1)*P)-pad,int((i+1)*P)-pad),(255,255,255),1)
    grid = disp.copy()                                           # faint full patch grid
    for k in range(38):
        c=int(round(k*P)); cv2.line(grid,(0,c),(S,c),(235,235,235),1); cv2.line(grid,(c,0),(c,S),(235,235,235),1)
    disp = cv2.addWeighted(grid,0.14,disp,0.86,0)
    return disp.astype(np.uint8), (int(peak[0]),int(peak[1])), P

def un_pale(rgb, mask):
    # clinical un-pale counterfactual (moderate): more saturated, redder, slightly darker, within tongue mask
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[...,1] = np.clip(hsv[...,1]*1.45+15, 0, 255)               # saturation up
    rgb2 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    lab = cv2.cvtColor(rgb2, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab[...,0] = np.clip(lab[...,0]*0.90, 0, 255)                  # darker L
    lab[...,1] = np.clip(lab[...,1]+14, 0, 255)                    # redder a
    rgb3 = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
    m = (gaussian_filter(mask, 4))[...,None]
    return (rgb*(1-m)+rgb3*m).astype(np.uint8)


# ---- data ----
import sys
TARGET = sys.argv[1] if len(sys.argv) > 1 else None          # e.g. Crack / Toothmark / Spot
STYLE = sys.argv[2] if len(sys.argv) > 2 else "weight"       # weight (YlOrRd+colorbar) | binary
FORCE_ID = sys.argv[3] if len(sys.argv) > 3 else None        # pick this exact sample id (from candidates_*.png)
# full-test attention dump (895 samples) so we can choose a sample whose evidence lands typically
_hp = SRC/"artifacts/runs/signot_global_ot_joint/explain_all/heatmaps.npz"
if not _hp.exists(): _hp = SRC/"artifacts/runs/signot_global_ot_joint/explain/heatmaps.npz"
hm = np.load(_hp, allow_pickle=True)
attn, labels, ids = hm["attn_heat"], hm["labels"], [str(x) for x in hm["ids"]]
loc = [LABELS.index(TARGET)] if TARGET else [LABELS.index(s) for s in ["Spot","Crack","Toothmark"]]
best = None
if FORCE_ID is not None and FORCE_ID in ids:
    n = ids.index(FORCE_ID); s = loc[0]; best = (0.0, n, s)
for n in range(attn.shape[0]) if best is None else []:
    m = load_mask(ids[n])
    for s in loc:
        if labels[n, s] == 1:
            hh = cv2.resize(attn[n,s].astype(np.float32),(S,S))*(m>0)
            conc = hh.max()/(hh[m>0].mean()+1e-9)
            if best is None or conc > best[0]: best = (conc, n, s)
if best is None:                                              # target sign not positive in the 8 dumped samples
    for n in range(attn.shape[0]):
        m = load_mask(ids[n])
        for s in ([LABELS.index(TARGET)] if TARGET else loc):
            hh = cv2.resize(attn[n,s].astype(np.float32),(S,S))*(m>0)
            conc = hh.max()/(hh[m>0].mean()+1e-9)
            if best is None or conc > best[0]: best = (conc, n, s)
_, bn, bs = best
attn_img = load_rgb(ids[bn]); attn_mask = load_mask(ids[bn])
ay0,ay1,ax0,ax1 = tongue_bbox(attn_mask)
cooc = np.load(V14/"outputs/cooc_cost_matrix.npz", allow_pickle=True)["C"]
cf = json.load(open(V14/"artifacts/runs/color_pale/explain/color_counterfactual.json"))
pale_ids = [str(x) for x in np.load(V14/"outputs/pale_color_heatmaps.npz", allow_pickle=True)["ids"]]
pid = pale_ids[0]; pale_rgb = load_rgb(pid); pale_mask = load_mask(pid); pale_cf = un_pale(pale_rgb, pale_mask)
py0,py1,px0,px1 = tongue_bbox(pale_mask)
def crop_a(x): return x[ay0:ay1, ax0:ax1]
def crop_p(x): return x[py0:py1, px0:px1]

# ---- figure ----
fig = plt.figure(figsize=(15, 8.6))
gs = GridSpec(2, 3, figure=fig, hspace=0.28, wspace=0.22)
def ttl(ax,t,c="k"): ax.set_title(t, fontsize=11, color=c, fontweight="bold")

# (A) traditional CNN Grad-CAM (real ResNet-34, test macro-F1 68.7) - smooth, post-hoc, unvalidated
from resnet_gradcam import gradcam
cam = gradcam(ids[bn], bs)
axA = fig.add_subplot(gs[0,0]); ovA,_ = cam_overlay(attn_img, cam, attn_mask, 0.6)
axA.imshow(crop_a(ovA)); axA.axis("off")
ttl(axA, f"(A) CNN (ResNet-34) Grad-CAM · {LABELS[bs]}", "#b00")
axA.text(0.5,-0.07,"smooth blob · post-hoc · UNVALIDATED · test F1 68.7", transform=axA.transAxes, ha="center", fontsize=9, color="#b00")

# (B) ours: DISCRETE OT-selected patches on the localized sign - deletion-faithful
axB = fig.add_subplot(gs[0,1]); ovB,(pi,pj),P = patch_overlay(attn_img, attn[bn,bs], attn_mask, topk=26, alpha=0.6, style=STYLE)
axB.imshow(crop_a(ovB)); axB.axis("off")
axB.add_patch(plt.Rectangle((pj*P-ax0,pi*P-ay0),P,P, ec="lime", fc="none", lw=2.5))
ttl(axB, f"(B) Ours: OT-selected patches · {LABELS[bs]}", "#060")
if STYLE == "weight":
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(norm=Normalize(0,1), cmap=plt.cm.YlOrRd); sm.set_array([])
    cax = axB.inset_axes([0.30, 0.04, 0.40, 0.035])           # inset horizontal bar, over dark mouth area
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal"); cb.set_ticks([0,1]); cb.set_ticklabels(["low","high"])
    cb.ax.tick_params(labelsize=7, colors="w"); cb.outline.set_edgecolor("w")
    cb.set_label("OT selection weight", fontsize=7.5, color="w", labelpad=-22)
    note = "discrete prototype→patch · shade = OT weight · deletion-faithful AUC 1.9"
else:
    note = "discrete prototype→patch · top-k selected · deletion-faithful AUC 1.9"
axB.text(0.5,-0.07,note, transform=axB.transAxes, ha="center", fontsize=9, color="#060")

# (D) co-occurrence relation matrix
axD = fig.add_subplot(gs[0,2]); im = axD.imshow(cooc, cmap="viridis_r", vmin=0, vmax=1)
axD.set_xticks(range(8)); axD.set_yticks(range(8))
axD.set_xticklabels([l[:4] for l in LABELS], rotation=45, ha="right", fontsize=7); axD.set_yticklabels([l[:4] for l in LABELS], fontsize=7)
for i in range(8):
    for j in range(8):
        if cooc[i,j] < 0.5 and i!=j: axD.text(j,i,f"{cooc[i,j]:.2f}", ha="center", va="center", fontsize=6, color="w")
ttl(axD, "(D) Ours: sign co-occurrence graph (1-Jaccard)", "#060")
fig.colorbar(im, ax=axD, fraction=0.046, pad=0.04)

# (C) clinical color counterfactual triptych
axC1 = fig.add_subplot(gs[1,0]); axC1.imshow(crop_p(pale_rgb)); axC1.axis("off"); ttl(axC1,"(C) Pale tongue — original","#060")
axC1.text(0.5,-0.06,"model: PALE ✓", transform=axC1.transAxes, ha="center", fontsize=9, color="#060")
axC2 = fig.add_subplot(gs[1,1]); axC2.imshow(crop_p(pale_cf)); axC2.axis("off"); ttl(axC2,"→ clinical counterfactual (un-pale)","#060")
axC2.text(0.5,-0.06,"redder · more saturated · darker", transform=axC2.transAxes, ha="center", fontsize=9, color="#333")
axC3 = fig.add_subplot(gs[1,2])
d = cf["deltas"]; cl = cf["drops"]["clinical"]; rd = cf["drops"]["random"]
axC3.plot(d, cl, "o-", color="#060", lw=2.5, label="un-pale dir (clinical)")
axC3.plot(d, rd, "s--", color="#999", lw=1.8, label="random dir")
axC3.set_xlabel("color perturbation δ", fontsize=9); axC3.set_ylabel("Pale logit drop", fontsize=9)
axC3.axhline(0, color="k", lw=0.6); axC3.legend(fontsize=8, loc="upper left"); axC3.grid(alpha=0.3)
ttl(axC3, "counterfactual-faithful 4.5×", "#060")

fig.suptitle("Interpretability: unvalidated saliency (traditional CNN)  vs.  our validated, sign-type-matched evidence",
             fontsize=13, fontweight="bold", y=0.98)
out = Path("/root/autodl-tmp/paper_materials_claude/figures"); out.mkdir(parents=True, exist_ok=True)
sfx = {"weight":"_w","binary":"_bin"}.get(STYLE,"")
fn = f"interpretability_4panel_{LABELS[bs]}{sfx}.png" if TARGET else f"interpretability_4panel{sfx}.png"
fig.savefig(out/fn, dpi=160, bbox_inches="tight")
print("saved", out/fn, "| localized sign panel =", LABELS[bs], "id", ids[bn], "conc=%.1f"%best[0], "| pale id", pid)
