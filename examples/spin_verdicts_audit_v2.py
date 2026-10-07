#!/usr/bin/env python3
"""Аудит v2: честное разделение (1) Fisher exact для флаговых фракций
и (2) Mann-Whitney для полных распределений σ_tick.
Плюс: средняя доминирующая компонента в mixed, гистограммы, фиксация seed."""
import json
import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact
import matplotlib.pyplot as plt
from pathlib import Path

SEED = 11  # зафиксирован для воспроизводимости

d = json.load(open("spin_verdicts_v6.json"))
s = np.array(d["sigma_tick"])
y = np.array(d["true_class"])
names = ["thermal", "shot", "quantum", "mixed"]
th = max(d["thetas"].values())

# ---- 1. Флаговые фракции и Fisher exact ----
print(f"Seed: {SEED} | θ (max thetas) = {th:.6f}")
print(f"\n{'class':9s} {'median':>9s} {'IQR':>14s} {'frac>θ':>7s} {'n_flagged':>10s}")
fracs = []
for i, nm in enumerate(names):
    v = s[y == i]
    q1, q3 = np.percentile(v, [25, 75])
    n_flag = int((v > th).sum())
    n_tot = len(v)
    frac = n_flag / n_tot
    fracs.append(frac)
    print(f"{nm:9s} {np.median(v):9.5f} [{q1:.5f},{q3:.5f}] {frac:7.3f} {n_flag:5d}/{n_tot}")

print("\nFisher exact (pairwise, flagged vs not):")
for i in range(4):
    for j in range(i+1, 4):
        a, b = fracs[i], fracs[j]
        n = len(s[y == i])
        table = [[int(a*n), int((1-a)*n)], [int(b*n), int((1-b)*n)]]
        _, p = fisher_exact(table, alternative='two-sided')
        print(f"  {names[i]:9s} vs {names[j]:9s}: {a:.3f} vs {b:.3f}, Fisher p={p:.3f}")

# ---- 2. Mann-Whitney для полных распределений ----
print("\nMann-Whitney (full σ_tick distributions, alternative='two-sided'):")
for i in range(4):
    for j in range(i+1, 4):
        _, p = mannwhitneyu(s[y == i], s[y == j], alternative='two-sided')
        print(f"  {names[i]:9s} vs {names[j]:9s}: MW p={p:.3e}")

# ---- 3. Средняя доминирующая компонента в mixed ----
if "true_fracs" in d and "pred_fracs" in d:
    tf = np.array(d["true_fracs"])
    pf = np.array(d["pred_fracs"])
    mx = y == 3
    dom_true = tf[mx].max(axis=1).mean()
    dom_pred = pf[mx].max(axis=1).mean()
    print(f"\nMixed traces: mean dominant component (true) = {dom_true:.3f}")
    print(f"              mean dominant component (pred) = {dom_pred:.3f}")
    if dom_true > 0.6:
        print("  → mixed ≈ pure (dominant > 60%): low σ_tick reflects easy classification")
    elif dom_true > 0.45:
        print("  → mixed has moderate dominance: router classifies by dominant, ignoring subdominant")
    else:
        print("  → mixed is genuinely mixed: low σ_tick is a blind spot, not confidence")

# ---- 4. Гистограммы ----
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, i, nm in zip(axes.flat, range(4), names):
    v = s[y == i]
    ax.hist(v, bins=50, color=['blue','green','red','orange'][i], alpha=0.7, edgecolor='black')
    ax.axvline(th, color='black', ls='--', label=f'θ={th:.4f}')
    ax.set_title(f"{nm} (n={len(v)}, median={np.median(v):.5f})")
    ax.set_xlabel("σ_tick")
    ax.set_ylabel("count")
    ax.legend()
fig.suptitle(f"σ_tick distributions (seed={SEED})", fontsize=14)
fig.tight_layout()
fig.savefig(Path("sigma_tick_distributions.png"), dpi=150)
print("\nSaved: sigma_tick_distributions.png")