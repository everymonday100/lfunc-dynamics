#!/usr/bin/env python3
r"""proper_time_bench.py (v4) — reviewer-refined protocol.

T1 seed/spectral transfer + bootstrap CI + permutation p:
   internal arm must predict drive level across simulator seeds AND across
   spectral shape (white -> OU, same std); external arm must fail.
T4 Spearman partial corr(erased, sat | steps); rule: >=0.5 proxy valid,
   <=0.2 link entirely via episode length.
T2 AUC + sign for centroid; transient energy; exponential relaxation fit
   d[t] ~ A exp(-t/tau); Cohen's d per candidate; best feature by AUC.
T3 r[t] trajectories; gap-window rise test Delta r (during - pre) for
   gapped / gapnoise / drift / smooth; normalized time-to-peak;
   LOO-LDA on (slope_first_half, max_r) as formalization of visual split.
"""
import json
import numpy as np
from pathlib import Path

from lfunc_dynamics.proper_time import RecurrentCell, ProperTimeProbe

T_MAX, LAM_TARGET = 128, 40.0
B_BOOT, B_PERM = 300, 200


# ---------------- utilities ----------------
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
    na, nb = len(a), len(b)
    var = ((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2)
    return float((np.mean(a) - np.mean(b)) / np.sqrt(var)) if var > 0 else 0.0


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


def boot_ci(Xte, yte, B=B_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    n = len(yte)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(yte[idx])) < 2:
            continue
        vals.append(r2fit(Xte, yte, Xte[idx], yte[idx]))
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


# ---------------- episodes ----------------
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
    r = np.asarray(probe.bank.history, float)
    d = np.asarray(probe.d_series, float)
    E = float(d[:20].sum() - d[-20:].sum()) if d.size >= 40 else 0.0
    pf = ProperTimeProbe(use_shadow=False)
    for h in hs:
        pf.update(h)
    return dict(reg=regime, level=level, spec=spec, snap=snap, r=r, d=d,
                c_fwd=probe.transient_centroid(), E=E,
                tau=relax_tau(d), cum=cum_ns, steps=t,
                mean_stall=float(np.mean(stalls)),
                n_stall_big=int(np.sum(np.asarray(stalls) > 100_000)))


def main():
    rng = np.random.default_rng(7)
    rows = []
    for g, seed_off in ((0, 0), (1, 900)):
        for reg in ("white1", "white3"):
            for i in range(40):
                rows.append(episode(reg, rng, (seed_off + i) % 5))
    for reg in ("ou1", "ou3", "strong", "gapped", "gapnoise", "drift"):
        for i in range(40):
            rows.append(episode(reg, rng, i % 5))
    warm = [episode("white1", rng, i % 5, warm=True) for i in range(60)]

    W0 = [r for r in rows if r["spec"] == "white" and r["reg"] in ("white1", "white3")][:80]
    W1 = [r for r in rows if r["spec"] == "white" and r["reg"] in ("white1", "white3")][80:]
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
        lo, hi = boot_ci(X1, y1)
        res[f"r2_{arm}_seed_transfer"] = obs1
        res[f"r2_{arm}_spectral_transfer"] = obsO
        res[f"ci_seed_transfer_{arm}"] = [lo, hi]
        res[f"perm_p_{arm}"] = perm_p(X0[:ntr], y0[:ntr], X1, y1, obs1)
    dR = res["r2_int_seed_transfer"] - res["r2_ext_seed_transfer"]
    res["delta_r2_seed_transfer"] = dR

    # ---- T4 ----
    er = np.array([r["snap"]["erased"] for r in rows])
    sa = np.array([r["snap"]["sat"] for r in rows], float)
    st = np.array([r["snap"]["steps"] for r in rows], float)
    res["spearman_raw"] = float(np.corrcoef(ranks(er), ranks(sa))[0, 1])
    res["spearman_partial_steps"] = pcorr_spearman(er, sa, st)

    # ---- T2 ----
    cold = rows
    y_cold = np.ones(len(cold) + len(warm)); y_cold[len(cold):] = 0
    allr = cold + warm
    f_cent = -np.array([r["c_fwd"] for r in allr])
    f_E = np.array([r["E"] for r in allr])
    f_sl = -np.array([relax_tau(r["d"]) if np.isfinite(relax_tau(r["d"])) else 1e3
                      for r in allr])
    res["auc_centroid"] = auc(f_cent, y_cold)
    res["auc_energy"] = auc(f_E, y_cold)
    res["auc_slope"] = auc(f_sl, y_cold)
    res["d_centroid"] = cohens_d([r["c_fwd"] for r in cold], [r["c_fwd"] for r in warm])
    res["d_energy"] = cohens_d([r["E"] for r in cold], [r["E"] for r in warm])
    tc = [r["tau"] for r in cold if np.isfinite(r["tau"])]
    tw = [r["tau"] for r in warm if np.isfinite(r["tau"])]
    res["tau_cold_median"] = float(np.median(tc)) if tc else np.nan
    res["tau_warm_median"] = float(np.median(tw)) if tw else np.nan

    # ---- T3 ----
    def delta_r(rs):
        vals = []
        for r in rs:
            rj = r["r"]
            if rj.size >= 40:
                vals.append(rj[23:39].mean() - rj[7:23].mean())
        return float(np.mean(vals)), float(np.std(vals) / np.sqrt(max(len(vals), 1)))
    for g in ("smooth", "strong", "gapped", "gapnoise", "drift"):
        key = "white1" if g == "smooth" else g
        rs = [r for r in rows if r["reg"] == key]
        m, se = delta_r(rs)
        res[f"delta_r_{g}"] = [m, se]
    L = min(len(r["r"]) for r in rows if r["reg"] in ("white1", "gapped", "drift", "strong"))
    res["mean_r_curves"] = {g: [float(np.mean([r["r"][i] for r in rows if r["reg"] == k]))
                                for i in range(L)]
                            for g, k in (("smooth", "white1"), ("gapped", "gapped"),
                                         ("drift", "drift"), ("strong", "strong"))}
    ttp = {g: float(np.mean([np.argmax(r["r"]) / max(len(r["r"]), 1)
                             for r in rows if r["reg"] == k]))
           for g, k in (("smooth", "white1"), ("gapped", "gapped"), ("drift", "drift"))}
    res["time_to_peak_norm"] = ttp
    cls = [r for r in rows if r["reg"] in ("white1", "gapped", "drift")]
    F2 = np.array([[np.polyfit(np.arange(max(len(r["r"]) // 2, 2)),
                               r["r"][:len(r["r"]) // 2], 1)[0], r["r"].max()]
                   for r in cls])
    y3 = np.array([{"white1": 0, "gapped": 1, "drift": 2}[r["reg"]] for r in cls])
    res["loo_lda_2feat_acc"] = loo_lda(F2, y3)
    res["corr_odom_work"] = float(np.corrcoef([r["snap"]["lam"] for r in rows],
                                              [r["level"] for r in rows])[0, 1])

    verdict = dict(stage="after6_v4", **res)
    out = Path("proper_time_verdicts.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=1)
    print(json.dumps({k: v for k, v in verdict.items()
                      if k != "mean_r_curves"}, indent=1))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()