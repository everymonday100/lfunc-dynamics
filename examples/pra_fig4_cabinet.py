#!/usr/bin/env python3
r"""
pra_fig4_cabinet.py — Fig. 4 для PRA: σ_tick vs route с quadrant boundaries.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CACHE = Path(__file__).resolve().parent.parent.parent
FIG_DIR = Path("pra_figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "mathtext.fontset": "dejavuserif",
})

def fig4():
    print("Fig. 4: CoreCabinet σ_tick vs route")
    cabinet = json.load(open(CACHE / "lfunc-dynamics" / "cabinet_verdicts.json",
                            encoding="utf-8"))
    theta = cabinet["theta_tick"]
    rows = cabinet["rows"]
    
    categories = {"clean": [], "suspect": [], "dimer": []}
    for r in rows:
        rt = r["route"]
        st = r["sigma_tick"]
        if "dimer" in rt:
            categories["dimer"].append(st)
        elif "suspect" in rt:
            categories["suspect"].append(st)
        else:
            categories["clean"].append(st)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    positions = [1, 2, 3]
    labels = ["Clean\n($g_{\\min}\\geq 0.05$)",
              "Suspect\n($0.02<g_{\\min}<0.05$)",
              "Dimer\n($g_{\\min}<0.02$)"]
    colors = ["tab:orange", "tab:red", "tab:purple"]
    
    bp = ax.boxplot([categories["clean"], categories["suspect"], categories["dimer"]],
                    positions=positions, patch_artist=True,
                    widths=0.6, showmeans=True, meanline=True)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    
    for i, (cat, color) in enumerate(zip(["clean", "suspect", "dimer"], colors)):
        x = np.random.normal(positions[i], 0.04, size=len(categories[cat]))
        ax.scatter(x, categories[cat], c=color, s=30, alpha=0.7,
                   edgecolors="k", linewidths=0.5, zorder=3)
    
    ax.axhline(theta, color="k", ls="--", lw=2, alpha=0.7,
               label=f"$\\theta_{{\\rm tick}} = {theta:.3f}$ (95th pctile)")
    
    ax.text(0.5, 0.02, "reliable", fontsize=11, weight="bold",
            color="tab:green", ha="center", va="center")
    ax.text(2.5, 0.08, "overconfident", fontsize=11, weight="bold",
            color="tab:red", ha="center", va="center")
    
    ax.set_ylabel(r"$\sigma_{\rm tick}$ (tick-to-tick dispersion)")
    ax.set_title("CoreCabinet: tick dispersion separates routes")
    ax.set_ylim(-0.01, 0.12)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper left")
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pra_fig4_cabinet.png")
    print("  saved: pra_fig4_cabinet.png")
    plt.close(fig)

if __name__ == "__main__":
    fig4()