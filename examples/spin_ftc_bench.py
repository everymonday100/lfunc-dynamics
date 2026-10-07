#!/usr/bin/env python3
r"""spin_ftc_bench.py (v4) — STATIC vs FTC under drift + radiation effects.
v4: modulated trace = dc*chop + noise (noise AFTER modulation, so the chopper
actually cancels low-frequency noise; v3 modulated noise too => gain==1.0);
single-event thermal spikes + TMR wear injected; FTC set_detect triggers
rewrite/verify (re-read); amplitude dosimeter tracked against true wear."""
import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from lfunc_dynamics.spin_noise import SpinNoiseRig
from lfunc_dynamics.ml_cabinet import CoreCabinet
from lfunc_dynamics.ftc import (FEATS, DriftingRig, FTCLayer,
                                demod_readout, clip_readout, _chop)

SPECIES = ["thermal", "shot", "quantum", "mixed"]
K, T = 4, 4
DC = 5.0
N_STREAM, N_INIT, N_WIN = 1200, 200, 4


class ProbaWrap:
    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1]

def algo_router(feat):
    """Чисто алгоритмический роутер на основе физических инвариантов.
    Служит для ablation study: сравнения с ML под дрейфом параметров."""
    # 1. Квантовые пары: высокая cross-correlation огибающих, умеренная автокорреляция
    if feat['pair_m'] > 0.40 and feat['ac1'] < 0.65:
        return "quantum"
    # 2. Дробовой шум: высокий Fano (сверхпуассоновская дисперсия), низкая автокорреляция
    elif feat['fano'] > 1.25 and feat['ac1'] < 0.30:
        return "shot"
    # 3. Термальный шум: высокая автокорреляция (Ornstein-Uhlenbeck)
    elif feat['ac1'] > 0.35:
        return "thermal"
    # 4. Всё остальное (пограничные случаи и смеси)
    else:
        return "mixed"

def modulate(tr, dc_eff, period):
    """Аппаратный порядок: сигнал модулируется, шум добавляется ПОСЛЕ."""
    return dc_eff * _chop(len(tr), period) + (tr - dc_eff)


def main():
    base = SpinNoiseRig(seed=3, tau_mag=500e-9)
    drig = DriftingRig(base, n_total=N_STREAM)
    rng = np.random.default_rng(11)

    trF, trY = [], []
    for si, sp in enumerate(SPECIES):
        for _ in range(300):
            if sp == "mixed":
                tr, _ = base.trace_mixed(rng, dc=DC)
            else:
                tr, _ = base.trace(rng, sp, dc=DC)
            trF.append(base.features(tr)); trY.append(si)
    trY = np.array(trY)
    X0 = np.array([[f[c] for c in FEATS] for f in trF], float)
    mu0, sd0 = X0.mean(0), X0.std(0) + 1e-12

    cabinets = {}
    for c, sp in enumerate(SPECIES):
        yb = (trY == c).astype(int)
        oof = np.zeros(len(yb))
        for a, b in KFold(4, shuffle=True, random_state=0).split(X0):
            m = HistGradientBoostingClassifier(random_state=0, max_iter=120,
                                               learning_rate=0.05)
            m.fit((X0[a] - mu0) / sd0, yb[a])
            oof[b] = m.predict_proba((X0[b] - mu0) / sd0)[:, 1]
        bag = []
        for i in range(K * T):
            idx = np.random.default_rng(i).choice(len(yb), len(yb), True)
            m = HistGradientBoostingClassifier(random_state=i, max_iter=120,
                                               learning_rate=0.05)
            m.fit((X0[idx] - mu0) / sd0, yb[idx])
            bag.append(ProbaWrap(m))
        surr = type("S", (), {"models": bag,
                              "sigma_res": float(np.std(yb - oof))})()
        cabinets[sp] = CoreCabinet(surr, K=K, T=T)

    std0 = lambda f, _e: (np.array([[f[c] for c in FEATS]], float) - mu0) / sd0

    init_tr, init_feats, init_sig, init_routes = [], [], [], []
    for i in range(N_INIT):
        sp = SPECIES[rng.integers(4)]
        tr, p = drig.trace(rng, sp, i, dc=DC)
        fd = base.features(tr)
        init_tr.append((tr, p)); init_feats.append(fd)
        v = {s: cabinets[s].predict(fd, "h", std0) for s in SPECIES}
        win = max(v, key=lambda s: v[s].value)
        init_sig.append(v[win].sigma_tick); init_routes.append(win)
    ftc = FTCLayer(dt=base.dt, beta=0.08)
    ftc.init_calib(np.array(init_sig), init_feats, np.array(init_routes))
    per0 = ftc.chop_period()
    tr0, p0 = init_tr[0]
    static = dict(theta=ftc.theta, period=per0,
                  thr=ftc.clip_thr(modulate(tr0, p0["dc_eff"], per0), per0))

    rows = []
    for i in range(N_INIT, N_STREAM):
        sp = SPECIES[rng.integers(4)]
        tr, p = drig.trace(rng, sp, i, dc=DC)
        fd = base.features(tr)
        rec = dict(i=i, true=sp, p=p)
        # Оценка алгоритмического роутера (для ablation study)
        algo_win = algo_router(fd)
        algo_correct = 1 if algo_win == sp else 0
        
        rec = dict(i=i, true=sp, p=p, algo_correct=algo_correct) # добавили algo_correct
        for name in ("static", "ftc"):
            if name == "static":
                std_fn = std0
                period, theta, thr = static["period"], static["theta"], static["thr"]
                set_flag = False
            else:
                std_fn = lambda f, _e: ftc.standardize(f)
                period, theta = ftc.chop_period(), ftc.theta
            tr_on = modulate(tr, p["dc_eff"], period)
            if name == "ftc":
                thr = ftc.clip_thr(tr_on, period)
            v = {s: cabinets[s].predict(fd, "h", std_fn) for s in SPECIES}
            win = max(v, key=lambda s: v[s].value)
            sig = v[win].sigma_tick
            if name == "ftc":
                amp_hat = demod_readout(tr_on, period) / DC
                set_flag, wear = ftc.update(tr, sig, fd, win, i, amp_hat=amp_hat)
            raw = tr.mean() - p["dc_eff"]
            if win == "thermal":
                err = demod_readout(tr_on, period) - p["dc_eff"]
            elif win == "shot":
                err = clip_readout(tr_on, period, thr) - p["dc_eff"]
            else:
                err = raw
            if name == "ftc" and set_flag:      # L2: rewrite/verify = re-read
                tr_r, p_r = drig.trace(rng, sp, i, dc=DC)
                tr_on_r = modulate(tr_r, p_r["dc_eff"], period)
                if win == "thermal":
                    err = demod_readout(tr_on_r, period) - p["dc_eff"]
                elif win == "shot":
                    err = clip_readout(tr_on_r, period, thr) - p["dc_eff"]
                else:
                    err = tr_r.mean() - p["dc_eff"]
            rec[name] = dict(route=win, sigma=sig, flag=float(sig > theta),
                             err=err, raw=raw, set=set_flag, spike=p["spike"])
        rows.append(rec)

    # ---- отчёт по окнам ----
    W = (N_STREAM - N_INIT) // N_WIN
    print(f"{'win':>3} | {'acc ML':>6} | {'acc Algo':>8} | {'gain_th S/F':>13} | "
          f"{'gain_sh S/F':>13} | {'flag S/F':>9}")
    for w in range(N_WIN):
        seg = rows[w * W:(w + 1) * W]
        acc_ml = np.mean([r["ftc"]["route"] == r["true"] for r in seg])
        acc_algo = np.mean([r["algo_correct"] for r in seg])
        
        # ... (далее ваш существующий код для gth, gsh, fl) ...
        
        fm = lambda x: f"{x:6.1f}" if np.isfinite(x) else "     -"
        print(f"{w:>3} | {acc_ml:.2f}   | {acc_algo:.2f}    | {fm(g[('static','th')])}/{fm(g[('ftc','th')])}   | "
              f"{fm(g[('static','sh')])}/{fm(g[('ftc','sh')])}   | {fl[0]:.2f}/{fl[1]:.2f}")
    for w in range(N_WIN):
        seg = rows[w * W:(w + 1) * W]
        acc = [np.mean([r[n]["route"] == r["true"] for r in seg])
               for n in ("static", "ftc")]
        g = {}
        for n in ("static", "ftc"):
            th = [r for r in seg if r[n]["route"] == "thermal"]
            sh = [r for r in seg if r[n]["route"] == "shot"]
            g[(n, "th")] = (np.sqrt(np.mean([r[n]["raw"] ** 2 for r in th])) /
                            max(np.sqrt(np.mean([r[n]["err"] ** 2 for r in th])), 1e-9)) if th else np.nan
            g[(n, "sh")] = (np.sqrt(np.mean([r[n]["raw"] ** 2 for r in sh])) /
                            max(np.sqrt(np.mean([r[n]["err"] ** 2 for r in sh])), 1e-9)) if sh else np.nan
        fl = [np.mean([r[n]["flag"] for r in seg]) for n in ("static", "ftc")]
        tp = sum(1 for r in seg if r["ftc"]["set"] and r["ftc"]["spike"])
        fp = sum(1 for r in seg if r["ftc"]["set"] and not r["ftc"]["spike"])
        spk = [r for r in seg if r["p"]["spike"]]
        es = [np.sqrt(np.mean([r[n]["err"] ** 2 for r in spk])) if spk else np.nan
              for n in ("static", "ftc")]
        print(f"{w:>3} | {acc[0]:.2f}/{acc[1]:.2f} | {fm(g[('static','th')])}/{fm(g[('ftc','th')])}   | "
              f"{fm(g[('static','sh')])}/{fm(g[('ftc','sh')])}   | {fl[0]:.2f}/{fl[1]:.2f} | "
              f"{tp:4d}/{fp:4d} | {fm(es[0])}/{fm(es[1])}")

    tau_true = np.array([r["p"]["tau_mag"] for r in rows])
    tau_rel = float(np.mean(np.abs(np.exp(ftc.log_tau) - tau_true) / tau_true))
    amp_true = drig.params(N_STREAM - 1)["amp"]
    amp_est = float(np.exp(ftc.log_amp))
    print(f"\nFTC трекинг tau: {tau_rel:.3f} | TMR-амплитуда: оценка {amp_est:.3f} "
          f"vs истина {amp_true:.3f} (ошибка {abs(amp_est - amp_true) / amp_true:.3f})")
    print("События (i, tag, route, ...):")
    for e in ftc.events[:12]:
        print("  ", e)
    print(f"theta: static={static['theta']:.4f} -> ftc={ftc.theta:.4f} | "
          f"period: static={static['period']} -> ftc={ftc.chop_period()}")

    out = Path("spin_ftc_verdicts.json")
    with open(out, "w") as fh:
        json.dump(dict(events=ftc.events,
                       theta_static=float(static["theta"]),
                       theta_ftc=float(ftc.theta),
                       tau_rel_err=tau_rel,
                       amp_est=amp_est, amp_true=amp_true), fh, indent=1)
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()