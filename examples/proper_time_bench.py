#!/usr/bin/env python3
r"""proper_time_bench.py (v3) — AFTER-6 under reviewer controls.

T1 double dissociation (P4): linear probes on INTERNAL features
(lam, n_ev, erased, sat, coh) vs EXTERNAL features (mean stall, big-stall
count) for two targets: work (drive scale) and coordinate (cum_ns).
Internal arm must win work and lose coordinate; external arm mirrors.
T2 arrow (P3): far-init cold start extends relaxation transient; episode
window defined INTERNALLY (stop at odometer target), centroid shift vs
fixed-64 window reported; warm (burn-in) control stays ~0.5.
T3 gap specificity (P1): 'drift' regime detunes natural frequencies with
same amplitude but WITHOUT drive gap; coherence rise must be gap-specific.
T4 shadow/free consistency (P2): corr(erased_bits, saturation count).
"""
import json
import numpy as np
from pathlib import Path

from lfunc_dynamics.proper_time import RecurrentCell, ProperTimeProbe

T_MAX, T_FIXED, LAM_TARGET = 128, 64, 40.0
REGIMES = ("smooth", "strong", "gapped", "drift")
DRIVE = {"smooth": 1.0, "strong": 3.0, "gapped": 1.0, "drift": 1.0}


def episode(regime, rng, wseed, warm=False):
    cell = RecurrentCell(seed=wseed)
    cell.h = rng.normal(0.0, 3.0, cell.n)          # далёкий старт: протяжённый транзиент
    shadow = cell.perturbed_copy()
    drive = DRIVE[regime]
    if warm:                                        # burn-in поглощает транзиент
        for _ in range(32):
            xb = rng.normal(0.0, drive, cell.n)
            cell.step(xb); shadow.step(xb)
    probe = ProperTimeProbe()
    if regime == "drift":
        probe.set_drift(0.3)                        # контроль P1: дрейф без разрыва
    cum_ns, stalls, hs, t = 0, [], [], 0
    while t < T_MAX:
        x = rng.normal(0.0, drive, cell.n)
        if regime == "gapped" and 24 <= t < 40:
            x = np.zeros(cell.n)
        st = int(50_000 * rng.lognormal(0.0, 0.3))
        stalls.append(st); cum_ns += st
        h = cell.step(x); hs.append(h.copy())
        probe.update(h, shadow.step(x))
        t += 1
        if probe.lam >= LAM_TARGET and t >= 8:      # ВНУТРЕННЕЕ окно по одометру
            break
    snap = probe.snapshot(); snap["gaps"] = probe.gap_steps_post()
    c_int = probe.transient_centroid()
    pf = ProperTimeProbe(use_shadow=False)          # центроид на фикс-окне (контроль P3)
    for h in hs[:T_FIXED]:
        pf.update(h)
    pr = ProperTimeProbe(use_shadow=False)
    for h in reversed(hs):
        pr.update(h)
    return dict(reg=regime, snap=snap, c_int=c_int, c_fix=pf.transient_centroid(),
                c_rev=pr.transient_centroid(), cum=cum_ns, drive=drive,
                mean_stall=float(np.mean(stalls)),
                n_stall_big=int(np.sum(np.asarray(stalls) > 100_000)))


def r2lin(X, y, ntr=72):
    Xtr, Xte, ytr, yte = X[:ntr], X[ntr:], y[:ntr], y[ntr:]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    A = np.hstack([(Xtr - mu) / sd, np.ones((ntr, 1))])
    w, *_ = np.linalg.lstsq(A, ytr, rcond=None)
    B = np.hstack([(Xte - mu) / sd, np.ones((len(yte), 1))])
    pred = B @ w
    return float(1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2))


def main():
    rng = np.random.default_rng(7)
    cold = [episode(REGIMES[i % 4], rng, i % 5) for i in range(120)]
    warm = [episode("smooth", rng, i % 5, warm=True) for i in range(60)]

    # ---- T1: двойная диссоциация ----
    X_int = np.array([[r["snap"]["lam"], r["snap"]["n_ev"], r["snap"]["erased"],
                       r["snap"]["sat"], r["snap"]["coh"]] for r in cold], float)
    X_ext = np.array([[r["mean_stall"], r["n_stall_big"]] for r in cold], float)
    yW = np.array([r["drive"] for r in cold], float)
    yC = np.array([r["cum"] for r in cold], float)

    # ---- T2: стрела ----
    cent_cold = float(np.mean([r["c_int"] for r in cold]))
    cent_warm = float(np.mean([r["c_int"] for r in warm]))
    acc_fwd = float(np.mean([r["c_int"] < 0.5 for r in cold]))
    shift = float(np.mean([r["c_int"] - r["c_fix"] for r in cold]))

    # ---- T3: специфичность разрыва ----
    coh = {g: float(np.mean([r["snap"]["coh"] for r in cold if r["reg"] == g]))
           for g in REGIMES}
    gaps = {g: float(np.mean([r["snap"]["gaps"] for r in cold if r["reg"] == g]))
            for g in REGIMES}

    # ---- T4: согласованность тени и бесплатного счётчика ----
    corr_sf = float(np.corrcoef([r["snap"]["erased"] for r in cold],
                                [r["snap"]["sat"] for r in cold])[0, 1])

    verdict = dict(
        stage="after6_v3",
        r2_internal_work=r2lin(X_int, yW), r2_external_work=r2lin(X_ext, yW),
        r2_internal_coord=r2lin(X_int, yC), r2_external_coord=r2lin(X_ext, yC),
        corr_odom_work=float(np.corrcoef([r["snap"]["lam"] for r in cold], yW)[0, 1]),
        corr_odom_coordtime=float(np.corrcoef([r["snap"]["lam"] for r in cold], yC)[0, 1]),
        centroid_cold=cent_cold, centroid_warm=cent_warm,
        arrow_acc_fwd=acc_fwd, centroid_shift_int_vs_fixed=shift,
        coherence_by_regime=coh, gap_steps_by_regime=gaps,
        corr_shadow_free=corr_sf,
    )
    out = Path("proper_time_verdicts.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=1)
    print(json.dumps(verdict, indent=1))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()