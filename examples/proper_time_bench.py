#!/usr/bin/env python3
r"""proper_time_bench.py (v2) — AFTER-6: three decisive tests of internal time.

T1 decoupling: odometer must NOT correlate with coordinate (sim stall) time,
   but MUST correlate with drive strength (work performed).
T2 arrow: transient-centroid asymmetry. Cold-start episodes relax onto the
   driven attractor => top-step centroid early (<0.5); time-reversed replay
   puts it late (>0.5). Warm-start (stationary) episodes are the control:
   centroid ~0.5 in both directions (no arrow without irreversibility).
T3 rhythm phenomenology: injected input gaps are caught by the relative gap
   counter (no clock involved); Kuramoto coherence separates regimes because
   coupling re-syncs the bank while the drive is absent (gap => coherence rise).

Irreversibility counters: shadow-trajectory erased bits (Landauer magnitude,
episodic calibration) and the free overwrite counter (online arrow).

Requires src/lfunc_dynamics/proper_time.py v2.
"""
import json
import numpy as np
from pathlib import Path

from lfunc_dynamics.proper_time import RecurrentCell, ProperTimeProbe

T = 64            # записываемых шагов на эпизод
BURN = 32         # burn-in для тёплого (стационарного) контроля
N_COLD = 120      # холодные эпизоды: 40 на режим
N_WARM = 60       # тёплый контроль (smooth)
REGIMES = ("smooth", "strong", "gapped")


def episode(regime, rng, wseed, warm=False):
    """Один эпизод: траектория RNN под режимом + независимое координатное время."""
    cell = RecurrentCell(seed=wseed)
    shadow = cell.perturbed_copy()
    drive = {"smooth": 1.0, "strong": 3.0, "gapped": 1.0}[regime]

    def draw(t):
        x = rng.normal(0.0, drive, cell.n)
        if regime == "gapped" and 24 <= t < 40:
            x = np.zeros(cell.n)
        return x

    # burn-in: выход на привлекаемое состояние -> стационарная траектория
    if warm:
        for t in range(BURN):
            xb = rng.normal(0.0, drive, cell.n)
            cell.step(xb); shadow.step(xb)

    probe = ProperTimeProbe()          # use_shadow=True: лендауэровский счёт
    cum_ns = 0
    hs = []
    for t in range(T):
        x = draw(t)
        # координатное время: scheduler-шум, независимый от внутренней динамики
        cum_ns += int(50_000 * rng.lognormal(0.0, 0.3))
        h = cell.step(x)
        hs.append(h.copy())
        probe.update(h, shadow.step(x))
    probe.note_overwrite(1)            # эмуляция push в ring buffer (бесплатная стрела)

    # стрела: обращённый прогон той же траектории
    rev = ProperTimeProbe(use_shadow=False)
    for h in reversed(hs):
        rev.update(h)

    snap = probe.snapshot()
    snap["gaps"] = probe.gap_steps_post()
    return dict(reg=regime, warm=warm, snap=snap,
                c_fwd=probe.transient_centroid(),
                c_rev=rev.transient_centroid(),
                cum=cum_ns, drive=drive)


def main():
    rng = np.random.default_rng(7)
    cold = [episode(REGIMES[i % 3], rng, wseed=i % 5, warm=False)
            for i in range(N_COLD)]
    warm = [episode("smooth", rng, wseed=i % 5, warm=True)
            for i in range(N_WARM)]

    # ---- T1: отделение от координатного времени, связь с работой ----
    lams = np.array([r["snap"]["lam"] for r in cold])
    cums = np.array([r["cum"] for r in cold], float)
    drs = np.array([r["drive"] for r in cold])
    corr_time = float(np.corrcoef(lams, cums)[0, 1])
    corr_work = float(np.corrcoef(lams, drs)[0, 1])

    # ---- T2: стрела через центроид транзиента (+ стационарный контроль) ----
    cent_cold = float(np.mean([r["c_fwd"] for r in cold]))
    cent_warm = float(np.mean([r["c_fwd"] for r in warm]))
    acc_fwd = float(np.mean([r["c_fwd"] < 0.5 for r in cold]))
    acc_rev = float(np.mean([r["c_rev"] > 0.5 for r in cold]))
    acc_warm = float(np.mean([abs(r["c_fwd"] - 0.5) < 0.15 for r in warm]))

    # ---- T3: феноменология ритмов (разрывы и когерентность без часов) ----
    coh = {g: float(np.mean([r["snap"]["coh"] for r in cold if r["reg"] == g]))
           for g in REGIMES}
    gaps = {g: float(np.mean([r["snap"]["gaps"] for r in cold if r["reg"] == g]))
            for g in REGIMES}

    # ---- необратимость: лендауэровские биты и бесплатный счётчик ----
    erased_mean = float(np.mean([r["snap"]["erased"] for r in cold]))
    overwrites = int(sum(r["snap"]["overwrites"] for r in cold))

    verdict = dict(
        stage="after6_v2", T=T, burn=BURN,
        n_cold=N_COLD, n_warm=N_WARM,
        corr_odom_coordtime=corr_time,
        corr_odom_work=corr_work,
        centroid_cold_mean=cent_cold,
        centroid_warm_mean=cent_warm,
        arrow_acc_fwd=acc_fwd,
        arrow_acc_rev=acc_rev,
        warm_control_ok=acc_warm,
        coherence_by_regime=coh,
        gap_steps_by_regime=gaps,
        mean_erased_bits=erased_mean,
        overwrite_counter=overwrites,
    )
    out = Path("proper_time_verdicts.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=1)
    print(json.dumps(verdict, indent=1))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()