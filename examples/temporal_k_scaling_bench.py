#!/usr/bin/env python3
r"""temporal_k_scaling_bench.py — AFTER-5: noise of the within-tick std estimate.

Для фиксированного входа x истинный разброс предсказаний по ядрам s(x)
оценивается выборочным std по K ядрам. Относительный шум этой оценки
теоретически 1/sqrt(2(K-1)) для i.i.d. гауссовских ячеек; усреднение по
T тикам даёт 1/sqrt(2(K-1)T) для sigma_tick. Бенч измеряет эмпирический
относительный шум субсэмплированием из пула M=128 ядер и сверяет с теорией
и гауссовским контролем.
"""
import json
import numpy as np
import torch
from pathlib import Path

from lfunc_dynamics.temporal_layer import BasePredictor

M_CORES = 128      # пул ядер (прокси популяции)
N_INPUTS = 200     # число входов x
R_SUB = 300        # субсэмплов на (x, K)
KS = (4, 8, 16, 32)
T_TICKS = 4
SEED = 0


def subset_rel_noise(preds: np.ndarray, K: int, R: int,
                     order_cache: np.ndarray) -> float:
    """Относительный шум выборочного std по субсэмплам размера K."""
    ds = preds[order_cache[:R, :K]].std(axis=1, ddof=1)
    ds = ds[ds > 0]
    if ds.size < 2:
        return np.nan
    return float(ds.std() / ds.mean())


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    # ---- пул ядер ----
    pool = []
    for m in range(M_CORES):
        torch.manual_seed(SEED + 1000 + m)
        pool.append(BasePredictor(in_dim=32, hidden=64, out_dim=1).eval())

    # ---- кэш порядков субсэмплирования (без возвратов) ----
    order = rng.random((R_SUB, M_CORES)).argsort(axis=1)

    # ---- эмпирика на пуле сетей ----
    emp_pool = {K: [] for K in KS}
    with torch.no_grad():
        for i in range(N_INPUTS):
            x = torch.randn(1, 32)
            preds = np.array([float(core(x)[0].item()) for core in pool],
                             dtype=np.float64)
            if preds.std(ddof=1) <= 0:
                continue
            for K in KS:
                emp_pool[K].append(subset_rel_noise(preds, K, R_SUB, order))
    emp_pool = {K: float(np.nanmean(v)) for K, v in emp_pool.items()}

    # ---- гауссовский контроль ----
    emp_gauss = {K: [] for K in KS}
    for _ in range(N_INPUTS):
        g = rng.normal(0.0, 1.0, M_CORES)
        for K in KS:
            emp_gauss[K].append(subset_rel_noise(g, K, R_SUB, order))
    emp_gauss = {K: float(np.nanmean(v)) for K, v in emp_gauss.items()}

    # ---- теория ----
    theory = {K: 1.0 / np.sqrt(2.0 * (K - 1)) for K in KS}
    theory_tick = {K: 1.0 / np.sqrt(2.0 * (K - 1) * T_TICKS) for K in KS}

    print(f"{'K':>3} | {'theory':>7} | {'pool of nets':>12} | "
          f"{'gauss control':>13} | {'sigma_tick (T=4)':>16}")
    for K in KS:
        print(f"{K:>3} | {theory[K]:7.3f} | {emp_pool[K]:12.3f} | "
              f"{emp_gauss[K]:13.3f} | {theory_tick[K]:16.3f}")

    verdict = dict(stage="k_scaling", seed=SEED, M=M_CORES, N=N_INPUTS,
                   R=R_SUB, T=T_TICKS,
                   Ks=list(KS),
                   theory=[theory[K] for K in KS],
                   theory_sigma_tick_T4=[theory_tick[K] for K in KS],
                   emp_pool=[emp_pool[K] for K in KS],
                   emp_gauss=[emp_gauss[K] for K in KS])
    out = Path("temporal_verdicts_k_scaling.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=1)
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()