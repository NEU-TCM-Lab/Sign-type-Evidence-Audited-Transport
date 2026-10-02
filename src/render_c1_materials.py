from __future__ import annotations
"""Render local illustrative figures. Sample selection does not establish test performance."""
import argparse
from pathlib import Path
from settings import PROJECT_ROOT, DATASET_ROOT, RUNS_DIR, MANIFEST_DIR, OUTPUTS_DIR, FIGURES_DIR, LABELS
from common import read_jsonl

def main():
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--sample-id", required=True)
    _args = parser.parse_args()
    # C1 (Injecting Modality Structure) building-block assets, derived from the same square-resize 518
    # base image (sample 006218). Produces: LAB/HSV 8-channel montage, coarse patch-colour mosaic,
    # OT-prototype top-k patch selection, and the top-k peak bar chart. Presentation/demo style.
    from pathlib import Path
    import numpy as np
    import cv2
    from PIL import Image
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from common import DATASET_ROOT, MANIFEST_DIR, read_jsonl, split_manifest_path

    S, GRID = 518, 37
    TID = _args.sample_id
    DEMO = 12
    OUT = _args.out_dir or FIGURES_DIR / "panels"
    OUT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    man = {r["id"]: r for r in read_jsonl(split_manifest_path("test", MANIFEST_DIR))}


    def sq_rgb(iid):
        return np.asarray(Image.open(DATASET_ROOT/man[iid]["image_path"]).convert("RGB").resize((S, S), Image.BILINEAR), np.uint8)
    def sq_mask(iid):
        m = np.asarray(Image.open(DATASET_ROOT/man[iid]["mask_path"]).convert("L").resize((S, S), Image.BILINEAR), np.float32)/255.0
        return (m > 0.4).astype(np.float32)
    def save(a, n): Image.fromarray(a).save(OUT/n); print("saved", OUT/n)


    def color8(rgb):
        # per-patch 8ch on 37x37: mean L,a,b,H,S,V + std(L),std(S) — same as cache_color_features
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
        chans = np.stack([lab[..., 0]/255, (lab[..., 1]-128)/128, (lab[..., 2]-128)/128,
                          hsv[..., 0]/179, hsv[..., 1]/255, hsv[..., 2]/255], -1)     # [S,S,6]
        P = S//GRID
        g = chans[:GRID*P, :GRID*P].reshape(GRID, P, GRID, P, 6)
        mean = g.mean((1, 3)); std = g[..., [0, 4]].std((1, 3))
        return np.concatenate([mean, std], -1)                                        # [37,37,8]


    rgb = sq_rgb(TID); mask = sq_mask(TID)
    m37 = cv2.resize(mask, (GRID, GRID), interpolation=cv2.INTER_AREA) >= 0.3
    feat = color8(rgb)                                                                # [37,37,8]

    # ---------- (1) LAB/HSV 8-channel montage ----------
    names = ["L", "a (green-red)", "b (blue-yellow)", "H", "S", "V", "std(L)", "std(S)"]
    cmaps = ["gray", "coolwarm", "coolwarm", "hsv", "viridis", "gray", "magma", "magma"]
    fig, axes = plt.subplots(2, 4, figsize=(9.5, 5.2))
    for c, ax in enumerate(axes.ravel()):
        ch = feat[..., c].copy().astype(float); ch[~m37] = np.nan
        cm = plt.get_cmap(cmaps[c]).copy(); cm.set_bad("#eeeeee")
        ax.imshow(ch, cmap=cm); ax.set_title(names[c], fontsize=11, fontweight="bold"); ax.axis("off")
    fig.suptitle("LAB/HSV per-patch color features (8 ch)", fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(OUT/"c1_8channels.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print("saved", OUT/"c1_8channels.png")

    # ---------- (2) coarse patch-colour mosaic (LAB/HSV box swatches) ----------
    def color_mosaic(rgb, mask, grid=DEMO, thick=3):
        pooled = cv2.resize(rgb, (grid, grid), interpolation=cv2.INTER_AREA)
        mg = cv2.resize(mask, (grid, grid), interpolation=cv2.INTER_AREA) >= 0.3
        pooled[~mg] = 0                                                               # off-tongue black
        vis = cv2.resize(pooled, (S, S), interpolation=cv2.INTER_NEAREST)
        P = S/grid
        for k in range(grid+1):
            c = min(S-1, int(round(k*P))); cv2.line(vis, (0, c), (S, c), (255, 255, 255), thick); cv2.line(vis, (c, 0), (c, S), (255, 255, 255), thick)
        return vis
    save(color_mosaic(rgb, mask), "c1_color_swatches.png")

    # ---------- (3) OT prototype + top-k selection ----------
    # color-prototype response: cosine similarity of each tongue patch's 8ch to a colour-salient prototype
    # (the highest-saturation tongue patch), then take the top-k peaks — illustrates "OT prototype->patch + top-k".
    F = feat.reshape(-1, 8); ton = m37.reshape(-1)
    Fn = F / (np.linalg.norm(F, axis=1, keepdims=True) + 1e-9)
    proto_idx = np.where(ton, feat[..., 4].reshape(-1), -1).argmax()                  # most-saturated tongue patch
    resp = (Fn @ Fn[proto_idx]); resp[~ton] = -1e9                                    # [1369]
    TOPK = 10
    top = np.argsort(resp)[::-1][:TOPK]; peak = int(top[0])

    def topk_patches(rgb, sel_flat, peak_flat, grid=GRID, thick=2):
        P = S/grid; disp = rgb.copy(); fill = disp.copy()
        for f in sel_flat:
            i, j = divmod(int(f), grid)
            cv2.rectangle(fill, (int(j*P)+1, int(i*P)+1), (int((j+1)*P)-1, int((i+1)*P)-1), (255, 145, 25), -1)
        disp = cv2.addWeighted(fill, 0.6, disp, 0.4, 0)
        for f in sel_flat:
            i, j = divmod(int(f), grid); cv2.rectangle(disp, (int(j*P), int(i*P)), (int((j+1)*P), int((i+1)*P)), (255, 255, 255), thick)
        i, j = divmod(int(peak_flat), grid); cv2.rectangle(disp, (int(j*P), int(i*P)), (int((j+1)*P), int((i+1)*P)), (60, 255, 60), 3)
        return disp.astype(np.uint8)
    save(topk_patches(rgb, top, peak), "c1_topk_patches.png")

    # ---------- (4) top-k peak bar chart ----------
    vals = np.sort(resp[ton])[::-1][:30]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    colors = ["#ff9119" if i < TOPK else "#cccccc" for i in range(len(vals))]
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axvline(TOPK-0.5, color="#060", ls="--", lw=1.5)
    ax.set_xlabel("tongue patches (sorted)", fontsize=10); ax.set_ylabel("color-prototype response", fontsize=10)
    ax.set_title(f"top-k peak detector (k={TOPK})", fontsize=11, fontweight="bold", color="#060")
    ax.set_xticks([]); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT/"c1_topk_bars.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print("saved", OUT/"c1_topk_bars.png")
    print("DONE")


if __name__ == "__main__":
    main()
