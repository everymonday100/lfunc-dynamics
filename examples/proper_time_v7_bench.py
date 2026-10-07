#!/usr/bin/env python3
r"""proper_time_v7_bench.py (v8) — reviewer round-7 closures.

E1' K sweep 0.1..1.0 step 0.1: per-K contrast delta_r(gapped-smooth) with
    bootstrap CI; sign-flip location vs mean-field K_c and finite-N caveat.
E2' bootstrap CI width vs sem width ratio (heavy-tail check).
E3' recoverability from h + buffer (CV-ridge): erased / recent / random.
E4' per-dimension Cohen's d on h windows (gap vs pre), fraction |d|>0.5;
    equal-length smooth control.
E5' power sim calibrated to observed contrast (a = 0.5/1.0/1.5 x observed).
E6' episode-averaged periodogram of the gap window (8 freqs), low/high ratio.
E7' empirical null spectrum of the phase coherence matrix (200 sims of
    independent rotators, same omega distribution, T=16): lambda+_emp.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from proper_time_bench import episode, delta_r, clean  # noqa: E402

from lfunc_dynamics.proper_time import TemporalMemory  # noqa: E402

N_EP = 30
A, B = 0.3, 1.2
K_C = 2.0 * (B - A) / np.pi


def boot_ci_delta(rs, other=None, B_=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B_):
        i1 = rng.integers(0, len(rs), len(rs))
        m1 = delta_r([rs[i] for i in i1])[0]
        if other is None:
            vals.append(m1)
        else:
            i2 = rng.integers(0, len(other), len(other))
            vals.append(m1 - delta_r([other[i] for i in i2])[0])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def cv_r2(X, y, k=4, lam=1.0, seed=0):
    """CV-ridge R2 (multi-output); protects against n<<p overfitting."""
    idx = np.random.default_rng(seed).permutation(len(y))
    folds = np.array_split(idx, k)
    pred = np.zeros_like(y, dtype=float)
    for f in folds:
        tr = np.setdiff1d(idx, f)
        A = np.hstack([X[tr], np.ones((len(tr), 1))])
        G = A.T @ A + lam * np.eye(A.shape[1])
        w = np.linalg.solve(G, A.T @ y[tr])
        Bm = np.hstack([X[f], np.ones((len(f), 1))])
        pred[f] = Bm @ w
    den = np.sum((y - y.mean(0)) ** 2)
    return float(1 - np.sum((y - pred) ** 2) / den) if den > 0 else 0.0


def cohens_d_vec(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    var = ((na - 1) * a.var(0, ddof=1) + (nb - 1) * b.var(0, ddof=1)) / (na + nb - 2)
    return (a.mean(0) - b.mean(0)) / np.sqrt(np.maximum(var, 1e-12))


def main():
    rng = np.random.default_rng(7)
    res = {"K_c_meanfield": float(K_C)}

    # ---- E1': K sweep with per-K significance ----
    sweep = {}
    for K in np.arange(0.1, 1.001, 0.1):
        Kf = float(round(K, 1))
        gp = [episode("gapped", rng, i % 5, coupling=Kf) for i in range(N_EP)]
        sm = [episode("white1", rng, i % 5, coupling=Kf) for i in range(N_EP)]
        lo, hi = boot_ci_delta(gp, sm, B_=300)
        sweep[Kf] = [delta_r(gp)[0] - delta_r(sm)[0], lo, hi]
    res["E1_sweep_contrast_ci"] = sweep
    flips = [Kf for Kf, v in sweep.items() if v[1] > 0 or v[2] < 0]
    res["E1_significant_Ks"] = flips
    signs = {Kf: np.sign(v[0]) for Kf, v in sweep.items()}
    res["E1_sign_flip_between"] = [Kf for Kf in sorted(signs)
                                   if signs[Kf] != signs.get(round(Kf - 0.1, 1), signs[Kf])][:1]

    # ---- E2': bootstrap vs sem ----
    gp = [episode("gapped", rng, i % 5) for i in range(40)]
    sm = [episode("white1", rng, i % 5) for i in range(40)]
    m_g, se_g, n_g = delta_r(gp)
    lo, hi = boot_ci_delta(gp, B_=1000)
    res["E2_sem_width"] = 2 * 1.96 * se_g
    res["E2_boot_width"] = hi - lo
    res["E2_ratio_boot_sem"] = (hi - lo) / max(2 * 1.96 * se_g, 1e-12)

    # ---- E3': recoverability from h + buffer ----
    mem = TemporalMemory(capacity=8, dim=8)
    pushed, h_last = [], None
    for i in range(20):
        ep = episode("white1", rng, i % 5)
        v = ep["h_mean"][:8].astype(np.float32)    # push-вектор: 8-мерная проекция состояния
        mem.push(v); pushed.append(v)
        h_last = ep["h_mean"].astype(np.float64)   # полное 32-мерное скрытое состояние
    buf = mem._buf.flatten().astype(np.float64)    # 64
    X = np.tile(np.concatenate([buf, h_last]), (12, 1)) + rng.normal(0, 1e-6, (12, 96))
    res["E3_r2_erased_hbuf"] = cv_r2(X, np.array(pushed[:12]))
    Xr = np.tile(np.concatenate([buf, h_last]), (8, 1)) + rng.normal(0, 1e-6, (8, 96))
    res["E3_r2_recent_hbuf"] = cv_r2(Xr, np.array(pushed[12:]))
    res["E3_r2_random_hbuf"] = cv_r2(X, rng.normal(0, 1, (12, 8)))

    # ---- E4': per-dim effect sizes on h windows ----
    for tag, reg in (("gapped", "gapped"), ("smooth_control", "white1")):
        rows = [episode(reg, rng, i % 5) for i in range(20)]
        pre = np.array([r["h_traj"][8:24].mean(0) for r in rows])
        inn = np.array([r["h_traj"][24:40].mean(0) for r in rows])
        d = cohens_d_vec(inn, pre)
        res[f"E4_median_abs_d_{tag}"] = float(np.median(np.abs(d)))
        res[f"E4_frac_d_gt05_{tag}"] = float(np.mean(np.abs(d) > 0.5))

    # ---- E5': power sim calibrated to observed contrast ----
    base = [r["r"] for r in sm]
    L = min(len(b) for b in base)
    base = [b[:L] for b in base]
    tpl = (np.mean([r["r"][:L] for r in gp], 0) - np.mean(base, 0))
    for a in (0.5, 1.0, 1.5):
        for n in (20, 40):
            hd = hv = 0
            for _ in range(50):
                A_ = np.array([base[i] for i in rng.integers(0, len(base), n)])
                Bv = np.array([base[i] + a * tpl for i in rng.integers(0, len(base), n)])
                dA = A_[:, 24:40].mean(1) - A_[:, 8:24].mean(1)
                dB = Bv[:, 24:40].mean(1) - Bv[:, 8:24].mean(1)
                t = (dB.mean() - dA.mean()) / np.sqrt(dB.var(ddof=1) / n + dA.var(ddof=1) / n)
                hd += int(abs(t) > 2.0)
                M = np.vstack([A_, Bv]); y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
                from proper_time_bench import pca_proj, loo_lda
                hv += int(loo_lda(pca_proj(M, 3), y) > 0.7)
            res[f"E5_a{a}_n{n}"] = [hd / 50, hv / 50]

    # ---- E6': episode-averaged periodogram of gap window ----
    for tag, reg in (("smooth", "white1"), ("gapped", "gapped"),
                     ("gapnoise", "gapnoise"), ("ou", "ou1")):
        rows = [episode(reg, rng, i % 5) for i in range(40)]
        pows = []
        for r in rows:
            w = r["u_in"][24:40]
            w = w - w.mean()
            if np.var(w) < 1e-12:
                pows.append(np.zeros(8))
                continue
            F = np.abs(np.fft.rfft(w)[1:9]) ** 2
            pows.append(F / F.sum())
        P = np.mean(pows, 0)
        res[f"E6_psd_{tag}"] = [float(x) for x in P]
        res[f"E6_lowhigh_{tag}"] = float(P[:2].mean() / max(P[2:].mean(), 1e-12))

    # ---- E7': empirical null spectrum of phase coherence matrix ----
    T_win, N_osc = 16, 8
    null_max = []
    for _ in range(200):
        w = rng.uniform(A, B, N_osc)
        th0 = rng.uniform(0, 2 * np.pi, N_osc)
        Th = (th0[None, :] + w[None, :] * np.arange(T_win)[:, None]) % (2 * np.pi)
        C = np.abs(np.exp(1j * Th[:, :, None] - 1j * Th[:, None, :]).mean(0))
        null_max.append(float(np.max(np.linalg.eigvalsh(C))))
    res["E7_lambda_emp_95"] = float(np.percentile(null_max, 95))
    res["E7_lambda_emp_mean"] = float(np.mean(null_max))
    for tag, reg in (("smooth", "white1"), ("gapped", "gapped"), ("drift", "drift")):
        rows = [episode(reg, rng, i % 5) for i in range(30)]
        ev0 = []
        for r in rows:
            Th = np.asarray(r["theta_hist"][24:40], float)
            C = np.abs(np.exp(1j * Th[:, :, None] - 1j * Th[:, None, :]).mean(0))
            ev0.append(float(np.max(np.linalg.eigvalsh(C))))
        res[f"E7_ev0_{tag}"] = float(np.mean(ev0))
        res[f"E7_frac_above_null_{tag}"] = float(np.mean(np.array(ev0) > res["E7_lambda_emp_95"]))

    res["T2_status"] = ("dynamical arrow weak (AUC<=0.64, exp-fit R2=0.06); "
                        "arrow = irreversibility flag pending E3'")
    res["selfhood_status"] = ("candidate temporal dynamics; temporal selfhood "
                              "not yet established (reviewer round 7 framing accepted)")

    verdict = clean(dict(stage="after6_v8", **res))
    out = Path("proper_time_verdicts_v7.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=1)
    print(json.dumps(verdict, indent=1))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()