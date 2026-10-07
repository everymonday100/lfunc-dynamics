#!/usr/bin/env python3
r"""
fig_corecabinet_architecture.py — обзорная схема CoreCabinet.
Левая половина: архитектура (input → features → K×T grid → two-stage aggregation → quadrant map → outputs).
Правая половина: два параллельных примера (quantum vs thermal) с разными sigma_tick и разными весами.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from pathlib import Path

FIG_DIR = Path("pra_figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "mathtext.fontset": "dejavuserif",
})

def draw_box(ax, x, y, w, h, text, color="lightblue", fontsize=8, weight="normal", alpha=0.9):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor="black", linewidth=1.2, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, weight=weight, wrap=True)

def draw_arrow(ax, x1, y1, x2, y2, color="black", style="->", lw=1.5):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw))

def fig_corecabinet():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    ax.set_title("CoreCabinet: Tick-Latched Temporal Ensemble for Calibrated Confidence",
                 fontsize=14, weight="bold", pad=20)

    # ============ ЛЕВАЯ ПОЛОВИНА: АРХИТЕКТУРА ============
    
    # 1. INPUT
    draw_box(ax, 0.3, 7.0, 2.2, 1.2, "Snapshot\nφ(x)\n(1D field)", "lightyellow", fontsize=9, weight="bold")
    ax.text(1.4, 6.7, "analog BH shot", ha="center", fontsize=7, color="gray")
    
    # Стрелка к features
    draw_arrow(ax, 2.5, 7.6, 3.2, 7.6)
    
    # 2. FEATURES
    feat_text = ("Feature vector (9):\n"
                 "E_out, E_in, m_j\n"
                 "m_out, m_in, R\n"
                 "bog_dev, kms_var\n"
                 "r_opt")
    draw_box(ax, 3.2, 6.5, 2.3, 2.2, feat_text, "lightgreen", fontsize=7)
    ax.text(4.35, 6.2, "physics-grounded", ha="center", fontsize=7, color="gray")
    
    # Стрелка к K×T grid
    draw_arrow(ax, 5.5, 7.6, 6.2, 7.6)
    
    # 3. K×T GRID (4×4)
    ax.text(7.35, 8.9, "K×T Grid (K=4 cores, T=4 ticks)", ha="center", fontsize=9, weight="bold")
    grid_x, grid_y = 6.2, 6.5
    cell_w, cell_h = 0.5, 0.5
    for t in range(4):
        for k in range(4):
            b = t * 4 + k
            color = "lightblue" if b % 2 == 0 else "lightcyan"
            draw_box(ax, grid_x + k * cell_w, grid_y + (3 - t) * cell_h,
                     cell_w - 0.05, cell_h - 0.05, f"b={b}", color, fontsize=6)
    # Подписи осей
    ax.text(6.0, 7.6, "tick\nt=0", ha="center", va="center", fontsize=6, rotation=90)
    ax.text(6.0, 6.6, "t=3", ha="center", va="center", fontsize=6, rotation=90)
    ax.text(7.35, 6.3, "core k=0 → 3", ha="center", fontsize=6)
    
    # Стрелка к aggregation
    draw_arrow(ax, 8.5, 7.6, 9.2, 7.6)
    
    # 4. TWO-STAGE AGGREGATION
    ax.text(10.35, 8.9, "Two-Stage Aggregation", ha="center", fontsize=9, weight="bold")
    
    # Stage 1: within-tick medians
    for t in range(4):
        y_pos = 7.5 - t * 0.5
        draw_box(ax, 9.2, y_pos, 1.5, 0.4, f"μ_t=med_k(ŷ_k,t)\nd_t=std_k(ŷ_k,t)", 
                 "lightyellow", fontsize=6)
        draw_arrow(ax, 8.5, 7.6 - t * 0.5, 9.2, y_pos + 0.2, color="blue", lw=1)
    
    # Stage 2: across-tick median
    draw_box(ax, 11.2, 7.0, 2.0, 1.2, 
             "ŷ = median_t(μ_t)\ns_tick = mean_t(d_t)\ns_ep = std(all cells)", 
             "lightcoral", fontsize=7, weight="bold")
    draw_arrow(ax, 10.7, 7.6, 11.2, 7.6)
    
    # Стрелка к quadrant map
    draw_arrow(ax, 13.2, 7.6, 13.8, 7.6)
    
    # 5. QUADRANT MAP
    ax.text(14.5, 8.9, "Quadrant Map", ha="center", fontsize=9, weight="bold")
    quad_size = 0.9
    quad_x, quad_y = 13.8, 6.5
    quads = [
        ("reliable", "rel_w<0.5\ns_tick<θ", "lightgreen"),
        ("overconfident", "rel_w<0.5\ns_tick≥θ", "yellow"),
        ("honest_low", "rel_w≥0.5\ns_tick≥θ", "lightcoral"),
        ("calibration_artifact", "rel_w≥0.5\ns_tick<θ", "orange"),
    ]
    for i, (name, cond, color) in enumerate(quads):
        row, col = i // 2, i % 2
        x = quad_x + col * (quad_size + 0.1)
        y = quad_y + (1 - row) * (quad_size + 0.1)
        draw_box(ax, x, y, quad_size, quad_size, f"{name}\n{cond}", color, fontsize=6, weight="bold")
    
    # Стрелка к outputs
    draw_arrow(ax, 13.2, 6.0, 11.2, 5.5)
    
    # 6. OUTPUTS
    ax.text(10.35, 5.8, "Outputs", ha="center", fontsize=9, weight="bold")
    outputs = [
        ("prediction ŷ", "lightblue"),
        ("sigma_tick", "lightcoral"),
        ("weight = 1/π", "lightgreen"),
        ("action", "lightyellow"),
    ]
    for i, (name, color) in enumerate(outputs):
        draw_box(ax, 9.2 + i * 1.1, 4.5, 1.0, 0.8, name, color, fontsize=7, weight="bold")
    
    # ============ ПРАВАЯ ПОЛОВИНА: ДВА ПРИМЕРА ============
    
    ax.text(8.0, 4.0, "Two Example Shots", ha="center", fontsize=11, weight="bold")
    
    # Quantum shot
    draw_box(ax, 0.3, 2.5, 3.5, 1.3, 
             "QUANTUM SHOT\nn_pairs=3, m_j=8.5\nbog_dev=0.3, kms_var=0.4", 
             "lightblue", fontsize=8, weight="bold")
    draw_arrow(ax, 3.8, 3.15, 4.5, 3.15, color="blue", lw=2)
    draw_box(ax, 4.5, 2.7, 2.5, 0.9, 
             "σ_tick = 0.30\n(high dispersion)", "lightcoral", fontsize=8, weight="bold")
    draw_arrow(ax, 7.0, 3.15, 7.7, 3.15, color="blue", lw=2)
    draw_box(ax, 7.7, 2.7, 2.5, 0.9, 
             "overconfident\nweight = 1/0.8 = 1.25", "yellow", fontsize=8, weight="bold")
    draw_arrow(ax, 10.2, 3.15, 10.9, 3.15, color="blue", lw=2)
    draw_box(ax, 10.9, 2.7, 2.5, 0.9, 
             "action: short ODE\ncheck (s_max=0.5)", "lightyellow", fontsize=7)
    
    # Thermal shot
    draw_box(ax, 0.3, 0.8, 3.5, 1.3, 
             "THERMAL SHOT\nn_pairs=0, m_j=2.1\nbog_dev=1.2, kms_var=2.1", 
             "lightgray", fontsize=8, weight="bold")
    draw_arrow(ax, 3.8, 1.45, 4.5, 1.45, color="orange", lw=2)
    draw_box(ax, 4.5, 1.0, 2.5, 0.9, 
             "σ_tick = 0.09\n(low dispersion)", "lightgreen", fontsize=8, weight="bold")
    draw_arrow(ax, 7.0, 1.45, 7.7, 1.45, color="orange", lw=2)
    draw_box(ax, 7.7, 1.0, 2.5, 0.9, 
             "reliable\nweight = 1/1.0 = 1.0", "lightgreen", fontsize=8, weight="bold")
    draw_arrow(ax, 10.2, 1.45, 10.9, 1.45, color="orange", lw=2)
    draw_box(ax, 10.9, 1.0, 2.5, 0.9, 
             "action: standard\nprotocol", "lightgreen", fontsize=7)
    
    # Легенда внизу
    ax.text(8.0, 0.2, 
            "Key insight: σ_tick separates quantum (high) from thermal (low) → quadrant map → HT weights → 75% shot economy",
            ha="center", fontsize=9, weight="bold", color="darkred",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="black"))
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_corecabinet_architecture.png", dpi=300)
    print("Saved: pra_figures/fig_corecabinet_architecture.png")
    plt.close(fig)

if __name__ == "__main__":
    fig_corecabinet()