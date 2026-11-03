import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent / "media_gallery"

def build_universal_card_thumbnail(out_path):
    fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor("#0B0F19")
    ax.set_facecolor("#0B0F19")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis("off")

    outer_rect = mpatches.Rectangle((1.5, 1.5), 97, 47, linewidth=1.5, edgecolor="#1E293B", facecolor="#0B0F19")
    ax.add_patch(outer_rect)

    ax.plot([7, 19], [34, 34], color="#38BDF8", linewidth=1.2, alpha=0.6)
    ax.plot([13, 13], [28, 40], color="#38BDF8", linewidth=1.2, alpha=0.6)
    ax.plot([9, 17], [30, 38], color="#60A5FA", linewidth=1.0, linestyle="--", alpha=0.5)
    circle_phys = mpatches.Circle((13, 34), 3.2, edgecolor="#38BDF8", facecolor="#0F172A", linewidth=1.4)
    ax.add_patch(circle_phys)
    ax.text(13, 22.5, "Embodied\nPhysics (3D)", color="#64748B", fontsize=5.8, fontweight="bold", ha="center", va="center", linespacing=1.2)

    nodes = [(83, 37), (89, 31), (82, 29), (88, 39), (85, 33)]
    for n1 in nodes:
        for n2 in nodes:
            if n1 != n2:
                ax.plot([n1[0], n2[0]], [n1[1], n2[1]], color="#A855F7", linewidth=0.7, alpha=0.45)
    for nx, ny in nodes:
        cn = mpatches.Circle((nx, ny), 1.3, edgecolor="#C084FC", facecolor="#3B0764", linewidth=1.0)
        ax.add_patch(cn)
    ax.text(85.5, 22.5, "Spiking\nSubstrate", color="#64748B", fontsize=5.8, fontweight="bold", ha="center", va="center", linespacing=1.2)

    center_bg = mpatches.FancyBboxPatch((26.5, 3.5), 47, 43, boxstyle="round,pad=0.5,rounding_size=1.2", facecolor="#111827", edgecolor="#38BDF8", linewidth=1.6)
    ax.add_patch(center_bg)

    ax.text(50, 41.5, "EAICE", color="#38BDF8", fontsize=16.5, fontweight="bold", ha="center", va="center")
    ax.text(50, 34.6, "MORTAL ACTIVE INFERENCE", color="#F8FAFC", fontsize=8.6, fontweight="bold", ha="center", va="center")
    ax.text(50, 29.2, "Neuromorphic Foundation for ARC-AGI-2", color="#94A3B8", fontsize=6.8, ha="center", va="center")

    tag1 = mpatches.FancyBboxPatch((28.5, 19.5), 43, 5.8, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor="#1E293B", edgecolor="#10B981", linewidth=1.0)
    ax.add_patch(tag1)
    ax.text(50, 22.4, "459K Params  |  20W Neuromorphic Equiv.", color="#F1F5F9", fontsize=6.6, fontweight="bold", ha="center", va="center")

    tag2 = mpatches.FancyBboxPatch((28.5, 12.2), 43, 5.8, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor="#1E293B", edgecolor="#A855F7", linewidth=1.0)
    ax.add_patch(tag2)
    ax.text(50, 15.1, "Life-Information Theory  |  SWR Replay", color="#F1F5F9", fontsize=6.6, fontweight="bold", ha="center", va="center")

    tag3 = mpatches.FancyBboxPatch((28.5, 4.9), 43, 5.8, boxstyle="round,pad=0.2,rounding_size=0.6", facecolor="#1E293B", edgecolor="#F59E0B", linewidth=1.0)
    ax.add_patch(tag3)
    ax.text(50, 7.8, "Standalone Zero-Simulator Reproduction", color="#F1F5F9", fontsize=6.5, fontweight="bold", ha="center", va="center")

    fig.savefig(out_path, dpi=100)
    plt.close(fig)

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = OUT_DIR / "card_thumbnail_560x280.png"
    build_universal_card_thumbnail(out_file)
    print(json.dumps({
        "status": "success",
        "generated": str(out_file)
    }, indent=2))

if __name__ == "__main__":
    main()
