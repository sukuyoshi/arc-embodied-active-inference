import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "media_gallery"

ARC_PALETTE = [
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
]
ARC_CMAP = ListedColormap(ARC_PALETTE)

def build_safe_card_thumbnail(out_path):
    fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor("#0B0F19")
    ax.set_facecolor("#0B0F19")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    outer_rect = mpatches.Rectangle((1.5, 1.5), 97, 47, linewidth=1.5, edgecolor="#1E293B", facecolor="#0B0F19")
    ax.add_patch(outer_rect)

    left_grid = np.array([
        [0, 3, 0],
        [3, 4, 3],
        [0, 3, 0]
    ])
    for r in range(3):
        for c in range(3):
            color_idx = left_grid[r, c]
            fc = ARC_PALETTE[color_idx]
            gx = 7 + c * 4.5
            gy = 28 - r * 4.5
            g_rect = mpatches.Rectangle((gx, gy), 4.0, 4.0, linewidth=0.6, edgecolor="#334155", facecolor=fc)
            ax.add_patch(g_rect)

    right_grid = np.array([
        [2, 9, 2],
        [4, 4, 2],
        [4, 2, 9]
    ])
    for r in range(3):
        for c in range(3):
            color_idx = right_grid[r, c]
            fc = ARC_PALETTE[color_idx]
            gx = 79 + c * 4.5
            gy = 28 - r * 4.5
            g_rect = mpatches.Rectangle((gx, gy), 4.0, 4.0, linewidth=0.6, edgecolor="#334155", facecolor=fc)
            ax.add_patch(g_rect)

    ax.text(13.7, 12, "Task 00d62c1b", color="#64748B", fontsize=5.8, fontweight="bold", ha="center", va="center")
    ax.text(85.7, 12, "Task 3c9b0459", color="#64748B", fontsize=5.8, fontweight="bold", ha="center", va="center")

    center_bg = mpatches.FancyBboxPatch((26.5, 3.5), 47, 43, boxstyle="round,pad=0.5,rounding_size=1.2", facecolor="#111827", edgecolor="#38BDF8", linewidth=1.6)
    ax.add_patch(center_bg)

    ax.text(50, 40.5, "EAICE", color="#38BDF8", fontsize=16.0, fontweight="bold", ha="center", va="center")
    ax.text(50, 33.2, "MORTAL ACTIVE INFERENCE", color="#F8FAFC", fontsize=8.6, fontweight="bold", ha="center", va="center")
    ax.text(50, 27.5, "ARC-AGI-2 Physical Grounding", color="#94A3B8", fontsize=7.2, ha="center", va="center")

    tag1 = mpatches.FancyBboxPatch((29.5, 17.5), 41, 6.8, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor="#1E293B", edgecolor="#10B981", linewidth=1.0)
    ax.add_patch(tag1)
    ax.text(50, 20.9, "459K Params  |  20W Neuromorphic", color="#F1F5F9", fontsize=6.8, fontweight="bold", ha="center", va="center")

    tag2 = mpatches.FancyBboxPatch((29.5, 9.2), 41, 6.8, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor="#1E293B", edgecolor="#A855F7", linewidth=1.0)
    ax.add_patch(tag2)
    ax.text(50, 12.6, "1-Step Exact Match  |  100% Invariant", color="#F1F5F9", fontsize=6.8, fontweight="bold", ha="center", va="center")

    ax.text(50, 5.8, "ARC Prize 2026 Paper Track", color="#64748B", fontsize=5.8, ha="center", va="center")

    fig.savefig(out_path, dpi=100)
    plt.close(fig)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "card_thumbnail_560x280.png"
    build_safe_card_thumbnail(out_file)
    print(json.dumps({
        "status": "success",
        "generated": str(out_file)
    }, indent=2))

if __name__ == "__main__":
    main()
