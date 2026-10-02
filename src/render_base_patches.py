from __future__ import annotations
"""Render local illustrative figures. Sample selection does not establish test performance."""
import argparse
from pathlib import Path
from settings import PROJECT_ROOT, DATASET_ROOT, RUNS_DIR, MANIFEST_DIR, OUTPUTS_DIR, FIGURES_DIR, LABELS
from common import read_jsonl

def main():
    from pathlib import Path
    from settings import LABELS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--sign", choices=LABELS, default="Toothmark")
    _args = parser.parse_args()
    # Four ALIGNED building-block images on the calibrated square-resize 518x518 frame (sample 006218,
    # Toothmark) for the overview figure: (1) base image, (2) 37x37 patch-tessellation, (3) CNN Grad-CAM
    # heatmap, (4) OT/attention-selected patches. Same 518 base -> all overlays register exactly.
    from pathlib import Path
    import numpy as np
    import cv2, torch
    from PIL import Image
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import binary_closing, label

    from common import DATASET_ROOT, MANIFEST_DIR, read_jsonl, split_manifest_path
    import resnet_gradcam as RG

    LABELS = ["TonguePale","TipSideRed","Spot","Ecchymosis","Crack","Toothmark","FurThick","FurYellow"]
    S, PATCH, GRID = 518, 14, 37
    TID, SIGN = _args.sample_id, _args.sign
    SRC = PROJECT_ROOT
    OUT = _args.out_dir or FIGURES_DIR / "panels"
    OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    man = {r["id"]: r for r in read_jsonl(split_manifest_path("test", MANIFEST_DIR))}


    def sq_rgb(iid):
        return np.asarray(Image.open(DATASET_ROOT/man[iid]["image_path"]).convert("RGB").resize((S, S), Image.BILINEAR), np.uint8)
    def sq_mask(iid):
        m = np.asarray(Image.open(DATASET_ROOT/man[iid]["mask_path"]).convert("L").resize((S, S), Image.BILINEAR), np.float32)/255.0
        return (m > 0.4).astype(np.float32)
    def save(a, n): Image.fromarray(a).save(OUT/n); print("saved", OUT/n, a.shape[:2])


    def patch_grid(rgb, grid=GRID, thick=1, alpha=0.5):
        g = rgb.copy(); P = S/grid
        for k in range(grid+1):
            c = min(S-1, int(round(k*P)))
            cv2.line(g, (0, c), (S, c), (255, 255, 255), thick); cv2.line(g, (c, 0), (c, S), (255, 255, 255), thick)
        return cv2.addWeighted(g, alpha, rgb, 1-alpha, 0)


    def ot_patch_demo(rgb, heat37, mask, grid=12, topk=10, thick=3, alpha=0.55):
        # coarse demo tessellation: pool the real 37x37 attention to a grid x grid grid, pick top-k on-tongue cells
        P = S/grid
        h = cv2.resize(heat37.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
        m = cv2.resize(mask.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
        ton = m >= 0.3; hh = h.copy(); hh[~ton] = -1e9
        sel = np.zeros((grid, grid), bool)
        for i, j in np.array(np.unravel_index(np.argsort(hh.ravel())[::-1][:topk], (grid, grid))).T: sel[i, j] = True
        peak = np.unravel_index(np.argmax(hh), (grid, grid))
        disp = rgb.copy(); fill = disp.copy(); ii, jj = np.where(sel)
        for i, j in zip(ii, jj):
            cv2.rectangle(fill, (int(j*P)+thick, int(i*P)+thick), (int((j+1)*P)-thick, int((i+1)*P)-thick), (255, 145, 25), -1)
        disp = cv2.addWeighted(fill, alpha, disp, 1-alpha, 0)
        for i, j in zip(ii, jj):
            cv2.rectangle(disp, (int(j*P)+thick, int(i*P)+thick), (int((j+1)*P)-thick, int((i+1)*P)-thick), (255, 255, 255), 2)
        # full coarse grid, thicker white lines
        g = disp.copy()
        for k in range(grid+1):
            c = min(S-1, int(round(k*P))); cv2.line(g, (0, c), (S, c), (255, 255, 255), thick); cv2.line(g, (c, 0), (c, S), (255, 255, 255), thick)
        disp = cv2.addWeighted(g, 0.35, disp, 0.65, 0)
        cv2.rectangle(disp, (int(peak[1]*P)+thick, int(peak[0]*P)+thick), (int((peak[1]+1)*P)-thick, int((peak[0]+1)*P)-thick), (60, 255, 60), 4)
        return disp.astype(np.uint8)


    def cam_overlay(rgb, cam, mask, alpha=0.6):
        h = cam.astype(np.float32)*(mask > 0); h = (h-h.min())/(h.max()-h.min()+1e-9)
        am = (h**1.2)[..., None]*alpha; return (rgb*(1-am)+plt.cm.jet(h)[..., :3]*255*am).astype(np.uint8)


    def ot_patch(rgb, heat37, mask, topk=26, alpha=0.6):
        P = S/GRID; m37 = cv2.resize(mask.astype(np.float32), (GRID, GRID), interpolation=cv2.INTER_AREA); ton = m37 >= 0.3
        h = heat37.astype(np.float32).copy(); h[~ton] = -1e9
        sel = np.zeros((GRID, GRID), bool)
        for i, j in np.array(np.unravel_index(np.argsort(h.ravel())[::-1][:topk], (GRID, GRID))).T: sel[i, j] = True
        peak = np.unravel_index(np.argmax(h), (GRID, GRID))
        sel = binary_closing(sel, structure=np.ones((3, 3)), iterations=1) & ton
        lbl, n = label(sel, structure=np.ones((3, 3)))
        for c in range(1, n+1):
            if (lbl == c).sum() < 3: sel[lbl == c] = False
        sel[peak] = True
        disp = rgb.copy(); fill = disp.copy(); pad = 2; ii, jj = np.where(sel)
        for i, j in zip(ii, jj):
            cv2.rectangle(fill, (int(j*P)+pad, int(i*P)+pad), (int((j+1)*P)-pad, int((i+1)*P)-pad), (255, 145, 25), -1)
        disp = cv2.addWeighted(fill, alpha, disp, 1-alpha, 0)
        for i, j in zip(ii, jj):
            cv2.rectangle(disp, (int(j*P)+pad, int(i*P)+pad), (int((j+1)*P)-pad, int((i+1)*P)-pad), (255, 255, 255), 1)
        g = disp.copy()
        for k in range(GRID+1):
            c = int(round(k*P)); cv2.line(g, (0, c), (S, c), (235, 235, 235), 1); cv2.line(g, (c, 0), (c, S), (235, 235, 235), 1)
        disp = cv2.addWeighted(g, 0.14, disp, 0.86, 0)
        cv2.rectangle(disp, (int(peak[1]*P), int(peak[0]*P)), (int((peak[1]+1)*P), int((peak[0]+1)*P)), (60, 255, 60), 3)
        return disp.astype(np.uint8)


    def mask_vis(mask, grid=None, thick=3, fill=205):
        # pseudo tongue mask as grayscale (tongue=light gray, off-tongue=black); blocky at coarse grid + thick white grid
        if grid:
            m = cv2.resize(mask.astype(np.float32), (grid, grid), interpolation=cv2.INTER_AREA)
            vis = cv2.resize((m*fill).astype(np.uint8), (S, S), interpolation=cv2.INTER_NEAREST)
        else:
            vis = (mask*fill).astype(np.uint8)
        vis = np.repeat(vis[..., None], 3, axis=2)
        if grid:
            P = S/grid
            for k in range(grid+1):
                c = min(S-1, int(round(k*P))); cv2.line(vis, (0, c), (S, c), (255, 255, 255), thick); cv2.line(vis, (c, 0), (c, S), (255, 255, 255), thick)
        return vis


    DEMO_GRID = 12   # coarse demo tessellation (not the real 37x37) for presentation clarity
    rgb = sq_rgb(TID); mask = sq_mask(TID)
    # (0) pseudo tongue mask: clean full-res + coarse demo (blocky, thick white grid)
    save(mask_vis(mask), "pseudo_mask_518.png")
    save(mask_vis(mask, grid=DEMO_GRID, thick=3), "pseudo_mask_518_demo.png")
    # (1) base
    save(rgb, "base_518.png")
    # (2) patch tessellation — REAL 37x37, and a coarse DEMO grid with thick white lines
    save(patch_grid(rgb, GRID, thick=1), "patchgrid_518.png")
    save(patch_grid(rgb, DEMO_GRID, thick=3, alpha=0.6), "patchgrid_518_demo.png")
    # (3) CNN Grad-CAM (square 518 frame)
    cam = RG.gradcam(TID, LABELS.index(SIGN))
    save(cam_overlay(rgb, cam, mask), "cnn_heatmap_518.png")
    # (4) OT/attention-selected patches (square-resize model attention dump): real 37x37 + coarse demo
    hp = RUNS_DIR/"signot_global_ot_joint/explain_all/heatmaps.npz"
    hm = np.load(hp, allow_pickle=False); ids = [str(x) for x in hm["ids"]]
    attn = hm["attn_heat"][ids.index(TID), LABELS.index(SIGN)]
    save(ot_patch(rgb, attn, mask), "ot_patch_518.png")
    save(ot_patch_demo(rgb, attn, mask, grid=DEMO_GRID, topk=10, thick=3), "ot_patch_518_demo.png")
    print("DONE")


if __name__ == "__main__":
    main()
