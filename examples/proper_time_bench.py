#!/usr/bin/env python3
r"""proper_time_bench.py (v6) — internal time: finalized protocol.

T1 double dissociation (paired bootstrap + permutation null on delta R2).
T4 shadow/free consistency under fixed length (length artifact removed).
T2 arrow: tau program CLOSED NEGATIVE (fit R2 reported); internal event
   localization via CUSUM change-point on d[t] (no external clock).
T3 gap signature: scalar delta_r (direction: decoherence drop), vector
   r[t]+r2[t] LOO-LDA with permutation p, power simulation with injected
   template, spectral manipulation check (input lag-1 in gap window),
   phase-matrix spectrum (decoherence vs clustering).
"""
import json
from pathlib import Path

import numpy as np

from lfunc_dynamics.proper_time import RecurrentCell, ProperTimeProbe

T_REC = 64
B_BOOT, B_PERM = 300, 200


# --------------------------------------------------------------------------
def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if np.isnan(f) else f
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


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
    rxy, rxz, ryz = pc(rx, ry), pc(rx, rz), pc(ry, rz)
    den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    return float((rxy - rxz * ryz) / den) if den > 1e-12 else 0.0


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
    d = np.asarray(d[:n], float); d = d[d > 0]
    if d.size < 8:
        return np.nan, np.nan, np.nan
    s = np.log(d + 1e-9); t = np.arange(len(s))
    slope, icpt = np.polyfit(t, s, 1)
    resid = s - (icpt + slope * t)
    ss_tot = np.sum((s - s.mean()) ** 2)
    r2 = float(1 - np.sum(resid ** 2) / ss_tot) if ss_tot > 0 else 0.0
    lag1r = float(np.corrcoef(resid[:-1], resid[1:])[0, 1]) if resid.size > 3 else np.nan
    tau = float(-1.0 / slope) if slope < -1e-6 else np.inf
    return tau, r2, lag1r


def r2fit(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(ytr), 1))])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    B = np.hstack([(Xte - mu) / sd, np.ones((len(yte), 1))])
    pred = B @ w
    den = np.sum((yte - yte.mean()) ** 2)
    return float(1 - np.sum((yte - pred) ** 2) / den) if den > 0 else 0.0


def paired_boot_delta(Xi0, Xe0, y0, Xi1, Xe1, y1, ntr, B=B_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(B):
        idx = rng.integers(0, ntr, ntr)
        ds.append(r2fit(Xi0[idx], y0[idx], Xi1, y1) - r2fit(Xe0[idx], y0[idx], Xe1, y1))
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def perm_delta_p(Xi0, Xe0, y0, Xi1, Xe1, y1, ntr, obs, B=B_PERM, seed=0):
    rng = np.random.default_rng(seed)
    cnt = 0
    for _ in range(B):
        yp = rng.permutation(y0[:ntr])
        if r2fit(Xi0[:ntr], yp, Xi1, y1) - r2fit(Xe0[:ntr], yp, Xe1, y1) >= obs:
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


def power_n(delta, sd):
    if not np.isfinite(delta) or not np.isfinite(sd) or delta <= 0 or sd <= 0:
        return int(1e9)
    return int(np.ceil(2 * (1.96 + 0.84) ** 2 * sd ** 2 / delta ** 2))


def lag1(x):
    x = np.asarray(x, float)
    if x.size < 4 or np.var(x) <= 0:
        return 0.0
    return float(np.corrcoef(x[:-1], x[1:])[0, 1])


def change_point(d):
    """Внутренняя локализация события: CUSUM по d[t], без внешних часов."""
    d = np.asarray(d, float)
    if d.size < 16:
        return -1
    cs = np.cumsum(d - d.mean())
    return int(np.argmax(np.abs(cs)))


def phase_spectrum(theta_hist, lo=24, hi=40):
    """Спектр матрицы попарных фазовых когерентностей: decoherence vs clustering."""
    Th = np.asarray(theta_hist[lo:hi], float)
    if Th.shape[0] < 4:
        return np.nan, np.nan
    N = Th.shape[1]
    C = np.abs(np.exp(1j * Th[:, :, None] - 1j * Th[:, None, :]).mean(0))
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    n_sig = int(np.sum(ev > 1.5 * ev.mean()))
    return n_sig, float(ev[0] / ev.sum())


def power_sim_vector(base_curves, template, ns=(10, 20, 40),
                     amps=(0.25, 0.5, 1.0), reps=50, seed=5):
    """Мощность векторного теста: инжекция известного сдвига формы."""
    rng = np.random.default_rng(seed)
    L = min(len(b) for b in base_curves)
    base = [b[:L] for b in base_curves]
    tpl = template[:L]
    out = {}
    for n in ns:
        for a in amps:
            hd = hv = 0
            for _ in range(reps):
                A = np.array([base[i] for i in rng.integers(0, len(base), n)])
                Bv = np.array([base[i] + a * tpl for i in rng.integers(0, len(base), n)])
                dA = A[:, 24:40].mean(1) - A[:, 8:24].mean(1)
                dB = Bv[:, 24:40].mean(1) - Bv[:, 8:24].mean(1)
                t = (dB.mean() - dA.mean()) / np.sqrt(dB.var(ddof=1) / n + dA.var(ddof=1) / n)
                if abs(t) > 2.0:
                    hd += 1
                M = np.vstack([A, Bv])
                y = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
                if loo_lda(pca_proj(M, 3), y) > 0.7:
                    hv += 1
            out[f"n{n}_a{a}"] = [hd / reps, hv / reps]
    return out


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
    cum_ns, stalls, d_gap, u_in = 0, [], [], []
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
        u_in.append(float(np.linalg.norm(x)))
        st = int(50_000 * rng.lognormal(0.0, 0.3))
        stalls.append(st); cum_ns += st
        h = cell.step(x); h_sum = h_sum + h
        probe.update(h, shadow.step(x))
        if 24 <= t < 40:
            d_gap.append(probe.d_series[-1])
    snap = probe.snapshot()
    d = np.asarray(probe.d_series, float)
    tau, r2_exp, lag1r = exp_fit_stats(d)
    n_sig, ev0 = phase_spectrum(probe.bank.theta_hist)
    return dict(reg=regime, level=level, spec=spec, snap=snap,
                r=np.asarray(probe.bank.history, float),
                r2=np.asarray(probe.bank.history2, float),
                d=d, u_in=np.asarray(u_in, float), h_mean=h_sum / T_REC,
                c_fwd=probe.transient_centroid(),
                E=float(d[:20].sum() - d[-20:].sum()),
                tau=tau, r2_exp=r2_exp, resid_lag1=lag1r,
                ph_nsig=n_sig, ph_ev0=ev0, cp=change_point(d),
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

    # ---- T4 (fixed length: length artifact removed) ----
    er = np.array([r["snap"]["erased"] for r in rows])
    sa = np.array([r["snap"]["sat"] for r in rows], float)
    nev = np.array([r["snap"]["n_ev"] for r in rows], float)
    res["spearman_raw"] = float(np.corrcoef(ranks(er), ranks(sa))[0, 1])
    res["spearman_partial_activity"] = pcorr_spearman(er, sa, nev)

    # ---- T2: arrow; tau program closed negative; internal event localization ----
    allr = rows + warm
    y_cold = np.zeros(len(allr)); y_cold[:len(rows)] = 1.0
    res["auc_centroid"] = auc(-np.array([r["c_fwd"] for r in allr]), y_cold)
    res["auc_energy"] = auc(np.array([r["E"] for r in allr]), y_cold)
    taus_all = np.array([r["tau"] if np.isfinite(r["tau"]) else 1e3 for r in allr])
    res["auc_slope"] = auc(-taus_all, y_cold)
    res["d_centroid"] = cohens_d([r["c_fwd"] for r in rows], [r["c_fwd"] for r in warm])
    res["tau_fit_r2_median"] = float(np.median([r["r2_exp"] for r in rows
                                                if np.isfinite(r["r2_exp"])]))
    res["tau_resid_lag1_median"] = float(np.median([r["resid_lag1"] for r in rows
                                                    if np.isfinite(r["resid_lag1"])]))
    gp = [r for r in rows if r["reg"] == "gapped"]
    res["changepoint_err_steps"] = float(np.mean([abs(r["cp"] - 24) for r in gp]))

    # ---- T3 ----
    for g, k in (("smooth", "white1"), ("strong", "strong"), ("gapped", "gapped"),
                 ("gapnoise", "gapnoise"), ("drift", "drift")):
        rs = [r for r in rows if r["reg"] == k]
        m, se, n = delta_r(rs)
        res[f"delta_r_{g}"] = [m, se, n]
        res[f"d_gap_{g}"] = float(np.mean([r["d_gap"] for r in rs]))
        res[f"input_lag1_gap_{g}"] = float(np.mean([lag1(r["u_in"][24:40]) for r in rs]))
        res[f"ph_ev0_{g}"] = float(np.mean([r["ph_ev0"] for r in rs]))
    mg, sg, ng = delta_r(gp)
    ms, ss, ns_ = delta_r([r for r in rows if r["reg"] == "white1"])
    sd_pool = np.sqrt(((ng - 1) * (sg * np.sqrt(ng)) ** 2 + (ns_ - 1) * (ss * np.sqrt(ns_)) ** 2)
                      / max(ng + ns_ - 2, 1))
    res["power_n_observed_effect"] = power_n(abs(mg - ms), sd_pool)
    cls3 = [r for r in rows if r["reg"] in ("white1", "gapped", "drift")]
    Mv = np.array([np.concatenate([r["r"][8:56], r["r2"][8:56]]) for r in cls3])
    y3 = np.array([{"white1": 0, "gapped": 1, "drift": 2}[r["reg"]] for r in cls3])
    Fp = pca_proj(Mv, 3)
    acc_obs = loo_lda(Fp, y3)
    rngp = np.random.default_rng(11)
    cnt = sum(1 for _ in range(B_PERM) if loo_lda(Fp, rngp.permutation(y3)) >= acc_obs)
    res["vec_lda_acc"] = acc_obs
    res["vec_lda_perm_p"] = (cnt + 1) / (B_PERM + 1)
    rng2 = np.random.default_rng(3)
    Fz = rng2.normal(0, 1, (120, 2)); yz = np.repeat([0, 1, 2], 40)
    res["lda_sanity_chance"] = loo_lda(Fz, yz)
    res["lda_sanity_separated"] = loo_lda(Fz + np.repeat([[0, 0], [1.5, 0], [-0.75, 1.3]], 40, axis=0), yz)
    # power simulation with injected shape template
    L = min(min(len(r["r"]) for r in gp), min(len(r["r"]) for r in rows if r["reg"] == "white1"))
    tpl = (np.mean([r["r"][:L] for r in gp], 0) - np.mean([r["r"][:L] for r in rows if r["reg"] == "white1"], 0))
    res["power_sim"] = power_sim_vector([r["r"] for r in rows if r["reg"] == "white1"], tpl)

    # ---- plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for g, k in (("smooth", "white1"), ("gapped", "gapped"),
                     ("drift", "drift"), ("strong", "strong")):
            rs = [r for r in rows if r["reg"] == k]
            Lp = min(len(r["r"]) for r in rs)
            plt.plot(np.mean([r["r"][:Lp] for r in rs], 0), label=f"r {g}")
        for g, k in (("smooth", "white1"), ("gapped", "gapped")):
            rs = [r for r in rows if r["reg"] == k]
            Lp = min(len(r["r2"]) for r in rs)
            plt.plot(np.mean([r["r2"][:Lp] for r in rs], 0), "--", label=f"r2 {g}")
        plt.axvspan(24, 40, color="grey", alpha=0.15)
        plt.legend(); plt.xlabel("step"); plt.ylabel("order parameter")
        plt.title("Kuramoto r[t] and r2[t] by regime (gap shaded)")
        plt.savefig("coherence_curves.png", dpi=150); plt.close()
        print("Saved: coherence_curves.png")
    except Exception as exc:
        print(f"[plot skipped] {exc}")

    verdict = clean(dict(stage="after6_v6", T_REC=T_REC, **res))
    out = Path("proper_time_verdicts.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=1)
    print(json.dumps(verdict, indent=1))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()