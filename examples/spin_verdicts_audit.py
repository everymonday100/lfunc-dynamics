#!/usr/bin/env python3
"""Аудит sigma_tick из spin_verdicts_v6.json: медианы/IQR/доли вместо mean±std,
попарные MW-тесты, сверка breakdown декомпозиции с логом."""
import json
import numpy as np
from scipy.stats import mannwhitneyu

d = json.load(open("spin_verdicts_v6.json"))
s = np.array(d["sigma_tick"]); y = np.array(d["true_class"])
names = ["thermal", "shot", "quantum", "mixed"]
th = max(d["thetas"].values())  # консервативный порог для доли выше theta

print(f"{'class':9s} {'median':>9s} {'IQR':>14s} {'frac>0':>7s} {'frac>theta':>11s} {'mean':>8s}")
for i, nm in enumerate(names):
    v = s[y == i]
    q1, q3 = np.percentile(v, [25, 75])
    print(f"{nm:9s} {np.median(v):9.4f} [{q1:.4f},{q3:.4f}] "
          f"{np.mean(v > 0):7.3f} {np.mean(v > th):11.3f} {v.mean():8.4f}")

print("\nПопарные MW (alternative='greater', строка против столбца):")
for i in range(4):
    for j in range(4):
        if i == j: continue
        p = mannwhitneyu(s[y == i], s[y == j], alternative="greater")[1]
        if p < 0.05:
            print(f"  {names[i]:9s} > {names[j]:9s}: p={p:.2e}")

# сверка breakdown декомпозиции с логом
pf = np.array(d["pred_fracs"]); tf = np.array(d["true_fracs"]); mx = y == 3
dt, dp = tf[mx].argmax(1), pf[mx].argmax(1)
print("\nBreakdown декомпозиции (mixed, disjoint test):")
for k, nm in enumerate(names[:3]):
    n = int((dt == k).sum()); ok = int(((dt == k) & (dp == k)).sum())
    print(f"  true-dom {nm:9s}: {n:3d}, correct {ok:3d}")
print(f"  dominant accuracy = {(dt == dp).mean():.3f}; "
      f"MAE fracs = {np.mean(np.abs(pf[mx] - tf[mx])):.3f}")