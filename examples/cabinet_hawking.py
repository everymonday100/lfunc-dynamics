#!/usr/bin/env python3
r"""
cabinet_hawking.py (v8) — HT-веса + мягкая фильтрация + фигура significance vs N.
Блоки:
 1. Новые инварианты (bog_dev, kms_var, r_opt) из hawking.py v6.1
 2. HT-веса: w_i = 1/π_i, где π_i — вероятность квадранта
 3. Мягкая фильтрация: отбрасываем только honest_low (5% шотов)
 4. Последовательный останов: target CI = 0.3
 5. Фигура: significance vs N (plain vs HT-weighted)
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from lfunc_dynamics.hawking import HawkingRig
from lfunc_dynamics.ml_cabinet import CoreCabinet

FEATS = ["E_out", "E_in", "m_j", "m_out", "m_in", "R", "bog_dev", "kms_var", "r_opt"]
K, T = 4, 4
# Вероятности "попадания в надёжную выборку" по квадрантам (для HT-весов)
PI_MAP = {"reliable": 1.0, "overconfident": 0.8, "honest_low": 0.5, "calibration_artifact": 0.2}

def main():
    rig_q = HawkingRig(T_H=25.0, T_th=60.0, rate=2.0, seed=1)
    rig_th = HawkingRig(T_H=25.0, T_th=90.0, rate=0.0, seed=2)
    rng = np.random.default_rng(7)

    feats, ys_pairs, cls = [], [], []
    # Quantum шоты: гарантируем n_pairs >= 1
    for _ in range(240):
        n = max(1, int(rng.poisson(rig_q.rate)))
        phi, tr = rig_q.shot(rng, n)
        feats.append(rig_q.features(phi))
        ys_pairs.append(float(n))
        cls.append("quantum")
    # Thermal шоты: n_pairs = 0
    for _ in range(160):
        phi, tr = rig_th.shot(rng, 0)
        feats.append(rig_th.features(phi))
        ys_pairs.append(0.0)
        cls.append("thermal")

    y = np.array(ys_pairs)
    X = np.array([[f[c] for c in FEATS] for f in feats], float)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd

    # OOF регрессия
    oof = np.zeros(len(y))
    for tr, te in KFold(4, shuffle=True, random_state=0).split(Xs):
        m = HistGradientBoostingRegressor(random_state=0, max_iter=120, learning_rate=0.05)
        m.fit(Xs[tr], y[tr])
        oof[te] = m.predict(Xs[te])
    mae = mean_absolute_error(y, oof)
    print(f"OOF MAE n_pairs = {mae:.3f}")

    # Bag для CoreCabinet
    bag = []
    for i in range(K * T):
        idx = np.random.default_rng(i).choice(len(y), len(y), replace=True)
        m = HistGradientBoostingRegressor(random_state=i, max_iter=120, learning_rate=0.05)
        m.fit(Xs[idx], y[idx])
        bag.append(m)
    surr = type('Surr', (), {'models': bag, 'sigma_res': float(np.sqrt(np.mean((y - oof) ** 2)))})()

    def std_fn(fdict, _ens):
        return (np.array([[fdict[c] for c in FEATS]], float) - mu) / sd

    cab = CoreCabinet(surr, K=K, T=T)
    theta = cab.calibrate(feats, ["h"] * len(feats), std_fn)
    print(f"theta_tick = {theta:.4f}")

    # Прогон с HT-весами
    rows = []
    for f, c, yv in zip(feats, cls, y):
        p = cab.predict(f, "h", std_fn)
        pi = PI_MAP.get(p.quadrant, 0.5)
        rows.append(dict(cls=c, truth=float(yv), pred=float(p.value),
                         sigma_tick=float(p.sigma_tick), quadrant=p.quadrant,
                         weight=1.0 / pi))

    # Сертификация
    st_q = np.array([r["sigma_tick"] for r in rows if r["cls"] == "quantum"])
    st_t = np.array([r["sigma_tick"] for r in rows if r["cls"] == "thermal"])
    u, pval = mannwhitneyu(st_q, st_t, alternative="two-sided")
    print(f"\nСертификация: sigma_tick quantum={st_q.mean():.3f} vs thermal={st_t.mean():.3f} (MW p={pval:.4f})")

    # Мягкая фильтрация: отбрасываем только honest_low
    keep_mask = np.array([r["quadrant"] != "honest_low" for r in rows])
    print(f"\nМягкая фильтрация: оставляем {keep_mask.sum()}/{len(rows)} шотов ({100*keep_mask.mean():.1f}%)")

    # Последовательный останов с траекторией CI vs N
    true_mean = y.mean()
    target_ci = 0.3
    batch_size = 20
    N_max = len(rows)

    plain_ci_traj, ht_ci_traj = [], []
    plain_N_stop, ht_N_stop = None, None

    # Преобразуем данные в numpy массивы для удобной индексации
    all_truths = np.array([r["truth"] for r in rows])
    all_weights = np.array([r["weight"] for r in rows])

    for N in range(batch_size, N_max + 1, batch_size):
        # Plain estimator (все шоты)
        rates_plain = all_truths[:N]
        ci_plain = 1.96 * np.std(rates_plain) / np.sqrt(N)
        plain_ci_traj.append((N, ci_plain))
        if ci_plain < target_ci and plain_N_stop is None:
            plain_N_stop = N

        # HT-weighted estimator (с мягкой фильтрацией)
        keep = keep_mask[:N]
        rates_ht = all_truths[:N][keep]
        weights_ht = all_weights[:N][keep]
        
        if len(rates_ht) > 1:
            ht_mean = np.sum(weights_ht * rates_ht) / np.sum(weights_ht)
            # Дисперсия HT-оценки (упрощённая формула)
            var_ht = np.sum(weights_ht ** 2 * (rates_ht - ht_mean) ** 2) / (np.sum(weights_ht) ** 2 * len(rates_ht))
            ci_ht = 1.96 * np.sqrt(var_ht) if var_ht > 0 else 0.0
        else:
            ci_ht = ci_plain
        ht_ci_traj.append((N, ci_ht))
        if ci_ht < target_ci and ht_N_stop is None:
            ht_N_stop = N

    print(f"\nПоследовательный останов (target CI = {target_ci}):")
    print(f"  Plain: N={plain_N_stop}")
    print(f"  HT-weighted: N={ht_N_stop}")
    if ht_N_stop and plain_N_stop:
        print(f"  Выигрыш HT: {plain_N_stop - ht_N_stop} шотов ({100*(plain_N_stop - ht_N_stop)/plain_N_stop:.1f}%)")

    # Фигура significance vs N
    fig, ax = plt.subplots(figsize=(10, 6))
    N_vals_plain = [t[0] for t in plain_ci_traj]
    ci_vals_plain = [t[1] for t in plain_ci_traj]
    N_vals_ht = [t[0] for t in ht_ci_traj]
    ci_vals_ht = [t[1] for t in ht_ci_traj]

    ax.plot(N_vals_plain, ci_vals_plain, 'o-', label='Plain (unweighted)', color='tab:blue', markersize=6)
    ax.plot(N_vals_ht, ci_vals_ht, 's-', label='HT-weighted (soft filter)', color='tab:orange', markersize=6)
    ax.axhline(target_ci, color='k', ls='--', lw=1.5, alpha=0.7, label=f'Target CI = {target_ci}')

    if plain_N_stop:
        ax.axvline(plain_N_stop, color='tab:blue', ls=':', lw=1.5, alpha=0.5)
        ax.text(plain_N_stop + 10, target_ci * 0.9, f'N={plain_N_stop}', color='tab:blue', fontsize=10)
    if ht_N_stop:
        ax.axvline(ht_N_stop, color='tab:orange', ls=':', lw=1.5, alpha=0.5)
        ax.text(ht_N_stop + 10, target_ci * 1.1, f'N={ht_N_stop}', color='tab:orange', fontsize=10)

    ax.set_xlabel('Number of shots N')
    ax.set_ylabel('95% CI width')
    ax.set_title('Sequential stopping: plain vs HT-weighted estimator')
    ax.legend(loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, N_max + 20)
    ax.set_ylim(0, max(ci_vals_plain) * 1.2)

    fig.tight_layout()
    fig.savefig(Path("significance_vs_N.png"), dpi=150)
    print("\nСохранено: significance_vs_N.png")
    plt.close(fig)

    # Сохранение verdicts
    for r in rows:
        r["truth"] = float(r["truth"])
        r["pred"] = float(r["pred"])
        r["sigma_tick"] = float(r["sigma_tick"])
        r["weight"] = float(r["weight"])

    json.dump(dict(theta_tick=float(theta), rows=rows,
                   plain_N_stop=plain_N_stop, ht_N_stop=ht_N_stop),
              open(Path("hawking_cabinet_verdicts_v8.json"), "w"), indent=1)
    print("Сохранено: hawking_cabinet_verdicts_v8.json")
    cab.close()

if __name__ == "__main__":
    main()