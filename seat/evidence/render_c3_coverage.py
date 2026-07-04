from __future__ import annotations
# C3 panel (d): evidence-type coverage matrix. rows = 8 signs, cols = 3 evidence types,
# shade = how strongly that sign is GROUNDED by that evidence (grounded / partial / not used).
# Message: every sign has >=1 grounded source; none relies only on spurious cues.
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

OUT = Path("/root/autodl-tmp/paper_materials_claude/paper_codex/latex/figures/panels")
signs = ["TonguePale", "TipSideRed", "Spot", "Ecchymosis", "Crack", "Toothmark", "FurThick", "FurYellow"]
cols = ["Spatial", "Color", "Relational"]
# 2 = grounded, 1 = partial, 0 = not used
M = [
    [0, 2, 1],   # Tongue color : holistic paleness -> spatial N/A; color 4.5x counterfactual grounds it
    [2, 1, 1],   # Tip red
    [2, 1, 1],   # Spots
    [2, 0, 1],   # Ecchymosis  : spatial grounded; color measured NOT useful (gamma=0)
    [2, 0, 2],   # Cracks      : spatial + strong co-occurrence
    [2, 0, 2],   # Teeth-marks : deletion-faithful 1.88 + co-occ 0.47/0.42
    [2, 0, 2],   # Fur thickness
    [2, 1, 1],   # Fur color
]
FACE = {2: "#3a7d34", 1: "#8ec44e", 0: "#e4eed6"}
DOT = {2: "#ffffff", 1: "#f2f7e9", 0: "#bacb9f"}
LABL = {2: "grounded", 1: "partial", 0: "not used"}

nr, nc = len(signs), len(cols)
cw, ch, gx, gy = 1.0, 0.66, 0.46, 0.16
fig, ax = plt.subplots(figsize=(4.7, 5.0))
for r in range(nr):
    y = (nr - 1 - r) * (ch + gy)
    ax.text(-0.28, y + ch/2, signs[r], ha="right", va="center", fontsize=10.5)
    for c in range(nc):
        x = c * (cw + gx)
        lv = M[r][c]
        ax.add_patch(FancyBboxPatch((x, y), cw, ch, boxstyle="round,pad=0.0,rounding_size=0.10",
                                    linewidth=0, facecolor=FACE[lv]))
        ax.add_patch(Circle((x + cw/2, y + ch/2), 0.055, facecolor=DOT[lv], edgecolor="none", zorder=3))
for c in range(nc):
    ax.text(c * (cw + gx) + cw/2, nr * (ch + gy) - gy + 0.08, cols[c], ha="center", va="bottom",
            fontsize=10, fontweight="bold")
# legend
ly = -0.80
for i, lv in enumerate([2, 1, 0]):
    lx = i * 1.95
    ax.add_patch(FancyBboxPatch((lx, ly), 0.32, 0.30, boxstyle="round,pad=0.0,rounding_size=0.06",
                                linewidth=0, facecolor=FACE[lv]))
    ax.text(lx + 0.42, ly + 0.15, LABL[lv], ha="left", va="center", fontsize=9.5)
ax.set_xlim(-2.05, max(nc * (cw + gx) - gx, 2 * 1.95 + 1.4) + 0.1)
ax.set_ylim(ly - 0.25, nr * (ch + gy) + 0.35)
ax.set_aspect("equal"); ax.axis("off")
fig.savefig(OUT / "c3_coverage_matrix.png", dpi=200, bbox_inches="tight")
print("saved", OUT / "c3_coverage_matrix.png")
