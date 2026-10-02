#!/usr/bin/env python3
r"""cabinet_hawking.py (v7) — регрессия на n_pairs, исправленные HT-веса.
Исправления: (1) quantum шоты всегда имеют n_pairs>=1; (2) HT-дисперсия по формуле
Horvitz-Thompson; (3) последовательный останов по эмпирическому CI."""
import json
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

from lfunc_dynamics.hawking import HawkingRig
from lfunc_dynamics.ml_cabinet import CoreCabinet

FEATS = ["E_out", "E_in", "m_j", "m_out", "m_in", "R", "bog_dev", "kms_var", "r_opt"]
K, T = 4, 4
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
                         sigma_tick=float(p.sigma_tick), quadrant=p.quadrant))

    # Сертификация
    st_q = np.array([r["sigma_tick"] for r in rows if r["cls"] == "quantum"])
    st_t = np.array([r["sigma_tick"] for r in rows if r["cls"] == "thermal"])
    u, pval = mannwhitneyu(st_q, st_t, alternative="two-sided")
    print(f"\nСертификация: sigma_tick quantum={st_q.mean():.3f} vs thermal={st_t.mean():.3f} (MW p={pval:.4f})")

    # HT-оценка среднего n_pairs
    true_mean = y.mean()
    print(f"\nTrue mean n_pairs: {true_mean:.3f}")
    print("\nПоследовательный останов (target CI width = 0.5):")
    target_ci = 0.5
    batch_size = 20
    
    for N in range(batch_size, len(rows) + 1, batch_size):
        # Plain estimator
        rates_plain = np.array([r["truth"] for r in rows[:N]])
        ci_plain = 1.96 * np.std(rates_plain) / np.sqrt(N)
        
        # HT-weighted estimator
        rates_ht = np.array([r["truth"] for r in rows[:N]])
        weights = np.array([1.0 / PI_MAP.get(r["quadrant"], 0.5) for r in rows[:N]])
        ht_mean = np.sum(weights * rates_ht) / np.sum(weights)
        # Дисперсия HT-оценки (упрощённая формула)
        var_ht = np.sum(weights ** 2 * (rates_ht - ht_mean) ** 2) / (np.sum(weights) ** 2 * N)
        ci_ht = 1.96 * np.sqrt(var_ht) if var_ht > 0 else 0.0
        
        if ci_plain < target_ci:
            print(f"  Plain: N={N}, CI={ci_plain:.3f}, mean={rates_plain.mean():.3f}")
            break
        if ci_ht < target_ci:
            print(f"  HT-weighted: N={N}, CI={ci_ht:.3f}, mean={ht_mean:.3f}")
            break

    json.dump(dict(theta_tick=float(theta), rows=rows),
              open(Path("hawking_cabinet_verdicts_v7.json"), "w"), indent=1)
    print("\nСохранено: hawking_cabinet_verdicts_v7.json")
    cab.close()

if __name__ == "__main__":
    main()