#!/usr/bin/env python3
r"""proper_time_bench.py (v5) — fixed-length episodes (gaps exist), paired
bootstrap + permutation null on delta R2, tau residual diagnostics + internal
tau head, power analysis, vector r[t] feature with permutation LOO-LDA,
LDA sanity simulation, second-moment order parameter, gapnoise reframed as
regime-switch control.
"""
import json
from pathlib import Path

import numpy as np

from lfunc_dynamics.proper_time import RecurrentCell, ProperTimeProbe

T_REC = 64
B_BOOT, B_PERM = 300, 200


# --------------------------------------------------------------------------
# utilities
# --------------------------------------------------------------------------
def ranks(x):
    x = np.asarray(x, float)
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def pcorr_spearman(x, y, z):
    rx, ry, rz = ranks(x), ranks(y), ranks(z)

    def pc(a, b):
        a = a - a.mean(); b = b - b.mean()
        na, nb = float(np.sqrt(a @ a)), float(np.sqrt(b @ b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float((a @ b) / (na * nb))


def auc(score, y):
    pos, neg = score[y == 1], score[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    var = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    return float((a.mean() - b.mean()) / np.sqrt(var)) if var > 0 else 0.0


def exp_fit_stats(d, n=24):
    """tau + качество экспоненциальной модели: R2 лог-фита и lag-1 остатков."""
    d = np.asarray(d[:n], float)
    d = d[d > 0]
    if d.size < 8:
        return np.nan, np.nan, np.nan
    s = np.log(d + 1e-9)
    t = np.arange(len(s))
    slope, icpt = np.polyfit(t, s, 1)
    resid = s - (icpt + slope * t)
    ss_tot = np.sum((s - s.mean()) ** 2)
    r2 = float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else 0.0
    lag1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if resid.size > 3 else np.nan
    tau = float(-1.0 / slope) if slope < -1e-6 else np.inf
    return tau, r2, lag1


def r2fit(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(ytr), 1))])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    B = np.hstack([(Xte - mu) / sd, np.ones((len(yte), 1))])
    pred = B @ w
    den = np.sum((yte - yte.mean()) ** 2)
    return float(1 - np.sum((yte - pred) ** 2) / den) if den > 0 else 0.0


def paired_boot_delta(Xi0, Xe0, y0, Xi1, Xe1, y1, ntr, B=B_BOOT, seed=0):
    """Paired bootstrap: один ресэмпл -> обе руки переобучены -> delta R2."""
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(B):
        idx = rng.integers(0, ntr, ntr)
        ri = r2fit(Xi0[idx], y0[idx], Xi1, y1)
        re = r2fit(Xe0[idx], y0[idx], Xe1, y1)
        ds.append(ri - re)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def perm_delta_p(Xi0, Xe0, y0, Xi1, Xe1, y1, ntr, obs, B=B_PERM, seed=0):
    """Нулевая: target перемешан -> распределение delta R2 между руками."""
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(B):
        yp = rng.permutation(y0[:ntr])
        ri = r2fit(Xi0[:ntr], yp, Xi1, y1)
        re = r2fit(Xe0[:ntr], yp, Xe1, y1)
        if ri - re >= obs:
            cnt += 1
    return (cnt + 1) / (B + 1)


def perm_p(Xtr, ytr, Xte, yte, obs, B=B_PERM, seed=0):
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(B):
        yp = rng.permutation(ytr)
        if r2fit(Xtr, yp, Xte, yte) >= obs:
            cnt += 1
    return (cnt + 1) / (B + 1)


def loo_lda(F, y):
    correct = 0
    for i in range(len(y)):
        mask = np.ones(len(y), bool); mask[i] = False
        classes = np.unique(y[mask])
        mu = np.array([F[mask][y[mask] == c].mean(0) for c in classes])
        var = np.vstack([F[mask][y[mask] == c].var(0) for c in classes]).mean(0) + 1e-9
        if classes[np.argmin(((F[i] - mu) ** 2 / var).sum(1))] == y[i]:
            correct += 1
    return correct / len(y)


def pca_proj(M, k=3):
    M = M - M.mean(0)
    U, S, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, :k] * S[:k]


def delta_r(rs, pre=(8, 24), during=(24, 40)):
    vals = []
    for r in rs:
        rj = r["r"]
        if rj.size >= during[1]:
            vals.append(float(rj[during[0]:during[1]].mean() - rj[pre[0]:pre[1]].mean()))
    if not vals:
        return float("nan"), float("nan"), 0
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals))), len(vals)


def power_n(delta, sd, alpha=0.05, power=0.8):
    if not np.isfinite(delta) or not np.isfinite(sd) or delta <= 0 or sd <= 0:
        return int(1e9)
    return int(np.ceil(2 * (1.96 + 0.84) ** 2 * sd ** 2 / delta ** 2))


# --------------------------------------------------------------------------
# episodes: FIXED length so the gap window (24..40) is inside every episode
# --------------------------------------------------------------------------
def episode(regime, rng, wseed, warm=False):
    cell = RecurrentCell(seed=wseed)
    cell.h = rng.normal(0.0, 3.0, cell.n)
    shadow = cell.perturbed_copy()
    level = 3.0 if regime.endswith("3") else 1.0
    spec = "ou" if regime.startswith("ou") else "white"
    if warm:
        for _ in range(32):
            xb = rng.normal(0.0, 1.0, cell.n)
            cell.step(xb); shadow.step(xb)
    probe = ProperTimeProbe()
    if regime == "drift":
        probe.set_drift(0.3)
    ou = np.zeros(cell.n)
    h_sum = np.zeros(cell.n)
    cum_ns, stalls, d_gap = 0, [], []
    for t in range(T_REC):
        if spec == "white":
            x = rng.normal(0.0, level, cell.n)
        else:
            ou = 0.9 * ou + rng.normal(0.0, level * np.sqrt(1 - 0.81), cell.n)
            x = ou.copy()
        if regime == "gapped" and 24 <= t < 40:
            x = np.zeros(cell.n)
        if regime == "gapnoise" and 24 <= t < 40:
            x = rng.normal(0.0, 0.15, cell.n)
        st = int(50_000 * rng.lognormal(0.0, 0.3))
        stalls.append(st); cum_ns += st
        h = cell.step(x); h_sum = h_sum + h
        probe.update(h, shadow.step(x))
        if 24 <= t < 40:
            d_gap.append(probe.d_series[-1])
    snap = probe.snapshot()
    d = np.asarray(probe.d_series, float)
    tau, r2_exp, lag1 = exp_fit_stats(d)
    return dict(reg=regime, level=level, spec=spec, snap=snap,
                r=np.asarray(probe.bank.history, float),
                r2=np.asarray(probe.bank.history2, float),
                d=d, h_mean=h_sum / T_REC,
                c_fwd=probe.transient_centroid(),
                E=float(d[:20].sum() - d[-20:].sum()),
                tau=tau, r2_exp=r2_exp, resid_lag1=lag1,
                cum=cum_ns, mean_stall=float(np.mean(stalls)),
                n_stall_big=int(np.sum(np.asarray(stalls) > 100_000)),
                d_gap=float(np.mean(d_gap)) if d_gap else float("nan"))


# --------------------------------------------------------------------------
def main():
    rng = np.random.default_rng(7)
    rows = []
    for seed_off in (0, 900):
        for reg in ("white1", "white3"):
            for i in range(40):
                rows.append(episode(reg, rng, (seed_off + i) % 5))
    for reg in ("ou1", "ou3", "strong", "gapped", "gapnoise", "drift"):
        for i in range(40):
            rows.append(episode(reg, rng, i % 5))
    warm = [episode("white1", rng, i % 5, warm=True) for i in range(60)]
    res = {}

    # ---- T1 ----
    whites = [r for r in rows if r["spec"] == "white"]
    W0, W1 = whites[:80], whites[80:]
    OU = [r for r in rows if r["spec"] == "ou"]

    def mats(rs):
        Xi = np.array([[r["snap"]["lam"], r["snap"]["n_ev"], r["snap"]["erased"],
                        r["snap"]["sat"], r["snap"]["coh"]] for r in rs], float)
        Xe = np.array([[r["mean_stall"], r["n_stall_big"]] for r in rs], float)
        y = np.array([r["level"] for r in rs], float)
        return Xi, Xe, y

    Xi0, Xe0, y0 = mats(W0)
    Xi1, Xe1, y1 = mats(W1)
    XiO, XeO, yO = mats(OU)
    ntr = 56
    for arm, X0, X1, XO in (("int", Xi0, Xi1, XiO), ("ext", Xe0, Xe1, XeO)):
        obs1 = r2fit(X0[:ntr], y0[:ntr], X1, y1)
        res[f"r2_{arm}_seed_transfer"] = obs1
        res[f"r2_{arm}_spectral_transfer"] = r2fit(X0[:ntr], y0[:ntr], XO, yO)
        res[f"perm_p_{arm}"] = perm_p(X0[:ntr], y0[:ntr], X1, y1, obs1)
    d_obs = res["r2_int_seed_transfer"] - res["r2_ext_seed_transfer"]
    lo, hi = paired_boot_delta(Xi0, Xe0, y0, Xi1, Xe1, y1, ntr)
    res["delta_r2_seed_transfer"] = d_obs
    res["delta_r2_paired_ci"] = [lo, hi]
    res["perm_p_delta"] = perm_delta_p(Xi0, Xe0, y0, Xi1, Xe1, y1, ntr, d_obs)

    # ---- T4 ----
    er = np.array([r["snap"]["erased"] for r in rows])
    sa = np.array([r["snap"]["sat"] for r in rows], float)
    st = np.array([r["snap"]["n_ev"] for r in rows], float)  # steps фиксирован в v5; контроль активности = n_ev
    res["spearman_raw"] = float(np.corrcoef(ranks(er), ranks(sa))[0, 1])
    res["spearman_partial_activity"] = pcorr_spearman(er, sa, st)

    # ---- T2 ----
    allr = rows + warm
    y_cold = np.zeros(len(allr)); y_cold[:len(rows)] = 1.0
    res["auc_centroid"] = auc(-np.array([r["c_fwd"] for r in allr]), y_cold)
    res["auc_energy"] = auc(np.array([r["E"] for r in allr]), y_cold)
    taus_all = np.array([r["tau"] if np.isfinite(r["tau"]) else 1e3 for r in allr])
    res["auc_slope"] = auc(-taus_all, y_cold)
    res["d_centroid"] = cohens_d([r["c_fwd"] for r in rows], [r["c_fwd"] for r in warm])
    tc = [r["tau"] for r in rows if np.isfinite(r["tau"])]
    tw = [r["tau"] for r in warm if np.isfinite(r["tau"])]
    res["tau_cold_median"] = float(np.median(tc)) if tc else float("nan")
    res["tau_warm_median"] = float(np.median(tw)) if tw else float("nan")
    res["tau_fit_r2_median"] = float(np.median([r["r2_exp"] for r in rows
                                                if np.isfinite(r["r2_exp"])]))
    res["tau_resid_lag1_median"] = float(np.median([r["resid_lag1"] for r in rows
                                                    if np.isfinite(r["resid_lag1"])]))
    # internal tau head: hidden -> log tau
    ct = [r for r in rows if np.isfinite(r["tau"])]
    Xh = np.array([r["h_mean"] for r in ct])
    yl = np.log([r["tau"] for r in ct])
    m = int(0.6 * len(yl))
    mu, sd = Xh[:m].mean(0), Xh[:m].std(0) + 1e-12
    A = np.hstack([(Xh[:m] - mu) / sd, np.ones((m, 1))])
    w, *_ = np.linalg.lstsq(A, yl[:m], rcond=None)
    Bm = np.hstack([(Xh[m:] - mu) / sd, np.ones((len(yl) - m, 1))])
    den = np.sum((yl[m:] - yl[m:].mean()) ** 2)
    res["r2_tau_head"] = float(1 - np.sum((yl[m:] - Bm @ w) ** 2) / den) if den > 0 else 0.0

    # ---- T3 ----
    for g, k in (("smooth", "white1"), ("strong", "strong"), ("gapped", "gapped"),
                 ("gapnoise", "gapnoise"), ("drift", "drift")):
        rs = [r for r in rows if r["reg"] == k]
        mm, se, nn = delta_r(rs)
        res[f"delta_r_{g}"] = [mm, se, nn]
        res[f"d_gap_{g}"] = float(np.mean([r["d_gap"] for r in rs]))
        res[f"delta_r2_{g}"] = [float(np.mean([r["r2"][24:40].mean() - r["r2"][8:24].mean()
                                               for r in rs]))]
    mg, sg, ng = delta_r([r for r in rows if r["reg"] == "gapped"])
    ms, ss, ns = delta_r([r for r in rows if r["reg"] == "white1"])
    sd_pool = np.sqrt(((ng - 1) * (sg * np.sqrt(ng)) ** 2 + (ns - 1) * (ss * np.sqrt(ns)) ** 2)
                      / max(ng + ns - 2, 1)) if ng > 0 and ns > 0 else np.nan
    res["power_n_observed_effect"] = power_n(abs(mg - ms), sd_pool)
    res["power_n_delta_005"] = power_n(0.05, sd_pool)
    # vector feature: full r[t] (+r2[t]) window, PCA->3, LOO-LDA + permutation p
    cls3 = [r for r in rows if r["reg"] in ("white1", "gapped", "drift")]
    Mv = np.array([np.concatenate([r["r"][8:56], r["r2"][8:56]]) for r in cls3])  # (160, 96)
    y3 = np.array([{"white1": 0, "gapped": 1, "drift": 2}[r["reg"]] for r in cls3])
    Fp = pca_proj(Mv, 3)
    acc_obs = loo_lda(Fp, y3)
    rngp = np.random.default_rng(11)
    cnt = 0
    for _ in range(B_PERM):
        if loo_lda(Fp, rngp.permutation(y3)) >= acc_obs:
            cnt += 1
    res["vec_lda_acc"] = acc_obs
    res["vec_lda_perm_p"] = (cnt + 1) / (B_PERM + 1)
    # LDA sanity simulation (procedure calibration)
    rng2 = np.random.default_rng(3)
    Fz = rng2.normal(0, 1, (120, 2))
    yz = np.repeat([0, 1, 2], 40)
    res["lda_sanity_chance"] = loo_lda(Fz, yz)
    Fs = Fz + np.repeat([[0.0, 0.0], [1.5, 0.0], [-0.75, 1.3]], 40, axis=0)
    res["lda_sanity_separated"] = loo_lda(Fs, yz)

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for g, k in (("smooth", "white1"), ("gapped", "gapped"),
                     ("drift", "drift"), ("strong", "strong")):
            rs = [r for r in rows if r["reg"] == k]
            L = min(len(r["r"]) for r in rs)
            plt.plot(np.mean([r["r"][:L] for r in rs], axis=0), label=f"r {g}")
        for g, k in (("smooth", "white1"), ("gapped", "gapped")):
            rs = [r for r in rows if r["reg"] == k]
            L = min(len(r["r2"]) for r in rs)
            plt.plot(np.mean([r["r2"][:L] for r in rs], axis=0), "--",
                     label=f"r2 {g}")
        plt.axvspan(24, 40, color="grey", alpha=0.15)
        plt.legend(); plt.xlabel("step"); plt.ylabel("order parameter")
        plt.title("Kuramoto r[t] and r2[t] by regime (gap shaded)")
        plt.savefig("coherence_curves.png", dpi=150); plt.close()
        print("Saved: coherence_curves.png")
    except Exception as exc:
        print(f"[plot skipped] {exc}")

    verdict = dict(stage="after6_v5", T_REC=T_REC, **res)
    out = Path("proper_time_verdicts.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=1)
    print(json.dumps(verdict, indent=1))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()