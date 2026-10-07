#!/usr/bin/env python3
r"""proper_time_bench.py (v4.1) — reviewer-refined protocol, single-file rebuild.

T1 seed/spectral transfer + bootstrap CI (fixed train) + permutation p.
T4 Spearman partial corr(erased, sat | steps).
T2 AUC + sign for centroid; transient energy; relaxation tau; Cohen's d.
T3 r[t] trajectories; robust gap-window rise test (quartiles, adaptive
   episode length); normalized time-to-peak; LOO-LDA on 2 features;
   coherence curves plot (matplotlib optional).
"""
import json
from pathlib import Path

import numpy as np

from lfunc_dynamics.proper_time import RecurrentCell, ProperTimeProbe

T_MAX, LAM_TARGET = 128, 40.0
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
        return float((a @ b) / np.sqrt((a @ a) * (b @ b)))
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


def relax_tau(d, n=24):
    d = np.asarray(d[:n], float)
    d = d[d > 0]
    if d.size < 8:
        return np.nan
    s = np.log(d + 1e-9)
    slope = np.polyfit(np.arange(len(s)), s, 1)[0]
    return float(-1.0 / slope) if slope < -1e-6 else np.inf


def r2fit(Xtr, ytr, Xte, yte):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    A = np.hstack([(Xtr - mu) / sd, np.ones((len(ytr), 1))])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    B = np.hstack([(Xte - mu) / sd, np.ones((len(yte), 1))])
    pred = B @ w
    den = np.sum((yte - yte.mean()) ** 2)
    return float(1 - np.sum((yte - pred) ** 2) / den) if den > 0 else 0.0


def boot_ci(Xtr, ytr, Xte, yte, B=B_BOOT, seed=0):
    """CI тестового R2 при ФИКСИРОВАННОЙ обученной модели (ресэмпл теста)."""
    rng = np.random.default_rng(seed)
    n = len(yte)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(yte[idx])) < 2:
            continue
        vals.append(r2fit(Xtr, ytr, Xte[idx], yte[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


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


def delta_r(rs):
    """Рост когерентности: вторая четверть траектории минус первая.
    Робастно к адаптивной длине эпизода."""
    vals = []
    for r in rs:
        rj = r["r"]
        if rj.size >= 10:
            q = max(1, rj.size // 4)
            vals.append(float(rj[q:2 * q].mean() - rj[:q].mean()))
    if not vals:
        return float("nan"), float("nan")
    return float(np.mean(vals)), float(np.std(vals) / np.sqrt(len(vals)))


# --------------------------------------------------------------------------
# episodes
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
    cum_ns, stalls, hs, t = 0, [], [], 0
    while t < T_MAX:
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
        h = cell.step(x); hs.append(h.copy())
        probe.update(h, shadow.step(x))
        t += 1
        if probe.lam >= LAM_TARGET and t >= 8:
            break
    snap = probe.snapshot()
    d = np.asarray(probe.d_series, float)
    E = float(d[:20].sum() - d[-20:].sum()) if d.size >= 40 else 0.0
    return dict(reg=regime, level=level, spec=spec, snap=snap,
                r=np.asarray(probe.bank.history, float), d=d,
                c_fwd=probe.transient_centroid(), E=E, tau=relax_tau(d),
                cum=cum_ns, steps=t, mean_stall=float(np.mean(stalls)),
                n_stall_big=int(np.sum(np.asarray(stalls) > 100_000)))


# --------------------------------------------------------------------------
# main
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

    # ---- T1: seed / spectral transfer ----
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
    res = {}
    for arm, X0, X1, XO in (("int", Xi0, Xi1, XiO), ("ext", Xe0, Xe1, XeO)):
        obs1 = r2fit(X0[:ntr], y0[:ntr], X1, y1)
        obsO = r2fit(X0[:ntr], y0[:ntr], XO, yO)
        lo, hi = boot_ci(X0[:ntr], y0[:ntr], X1, y1)
        res[f"r2_{arm}_seed_transfer"] = obs1
        res[f"r2_{arm}_spectral_transfer"] = obsO
        res[f"ci_seed_transfer_{arm}"] = [lo, hi]
        res[f"perm_p_{arm}"] = perm_p(X0[:ntr], y0[:ntr], X1, y1, obs1)
    res["delta_r2_seed_transfer"] = (res["r2_int_seed_transfer"]
                                     - res["r2_ext_seed_transfer"])

    # ---- T4: Spearman partial ----
    er = np.array([r["snap"]["erased"] for r in rows])
    sa = np.array([r["snap"]["sat"] for r in rows], float)
    st = np.array([r["snap"]["steps"] for r in rows], float)
    res["spearman_raw"] = float(np.corrcoef(ranks(er), ranks(sa))[0, 1])
    res["spearman_partial_steps"] = pcorr_spearman(er, sa, st)

    # ---- T2: arrow candidates ----
    allr = rows + warm
    y_cold = np.zeros(len(allr)); y_cold[:len(rows)] = 1.0
    f_cent = -np.array([r["c_fwd"] for r in allr])
    f_E = np.array([r["E"] for r in allr])
    taus = np.array([r["tau"] if np.isfinite(r["tau"]) else 1e3 for r in allr])
    f_sl = -taus
    res["auc_centroid"] = auc(f_cent, y_cold)
    res["auc_energy"] = auc(f_E, y_cold)
    res["auc_slope"] = auc(f_sl, y_cold)
    res["d_centroid"] = cohens_d([r["c_fwd"] for r in rows],
                                 [r["c_fwd"] for r in warm])
    res["d_energy"] = cohens_d([r["E"] for r in rows], [r["E"] for r in warm])
    tc = [r["tau"] for r in rows if np.isfinite(r["tau"])]
    tw = [r["tau"] for r in warm if np.isfinite(r["tau"])]
    res["tau_cold_median"] = float(np.median(tc)) if tc else float("nan")
    res["tau_warm_median"] = float(np.median(tw)) if tw else float("nan")

    # ---- T3: gap-rise test, curves, minimal classifier ----
    for g, k in (("smooth", "white1"), ("strong", "strong"), ("gapped", "gapped"),
                 ("gapnoise", "gapnoise"), ("drift", "drift")):
        rs = [r for r in rows if r["reg"] == k]
        m, se = delta_r(rs)
        res[f"delta_r_{g}"] = [m, se]
    curve_regs = (("smooth", "white1"), ("gapped", "gapped"),
                  ("drift", "drift"), ("strong", "strong"))
    L = min(min(len(r["r"]) for r in rows if r["reg"] == k) for _, k in curve_regs)
    res["mean_r_curves"] = {g: [float(np.mean([r["r"][i] for r in rows if r["reg"] == k]))
                                for i in range(L)] for g, k in curve_regs}
    res["time_to_peak_norm"] = {
        g: float(np.mean([np.argmax(r["r"]) / max(len(r["r"]), 1)
                          for r in rows if r["reg"] == k]))
        for g, k in (("smooth", "white1"), ("gapped", "gapped"), ("drift", "drift"))}
    cls = [r for r in rows if r["reg"] in ("white1", "gapped", "drift")]
    F2 = np.array([[np.polyfit(np.arange(max(len(r["r"]) // 2, 2)),
                               r["r"][:len(r["r"]) // 2], 1)[0], r["r"].max()]
                   for r in cls])
    y3 = np.array([{"white1": 0, "gapped": 1, "drift": 2}[r["reg"]] for r in cls])
    res["loo_lda_2feat_acc"] = loo_lda(F2, y3)
    res["corr_odom_work"] = float(np.corrcoef([r["snap"]["lam"] for r in rows],
                                              [r["level"] for r in rows])[0, 1])

    # ---- plot coherence curves (optional) ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for g, k in curve_regs:
            rs = [r for r in rows if r["reg"] == k]
            Lp = min(len(r["r"]) for r in rs)
            plt.plot(np.mean([r["r"][:Lp] for r in rs], axis=0), label=g)
        plt.legend(); plt.xlabel("step"); plt.ylabel("r")
        plt.title("Kuramoto coherence r[t] by regime")
        plt.savefig("coherence_curves.png", dpi=150); plt.close()
        print("Saved: coherence_curves.png")
    except Exception as exc:
        print(f"[plot skipped] {exc}")

    verdict = dict(stage="after6_v4", **res)
    out = Path("proper_time_verdicts.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=1)
    print(json.dumps({k: v for k, v in verdict.items()
                      if k != "mean_r_curves"}, indent=1))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()