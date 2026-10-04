#!/usr/bin/env python3
r"""spin_noise_bench.py (v4) — регрессионная декомпозиция смешанных трасс.
Цель декомпозиции — доли дисперсии компонент (наблюдаемая величина),
модель — регрессия на симплексе с OOF-нормировкой. Классификация маршрутов
остаётся на CoreCabinet; декомпозиция выбирает комбинацию очистки."""
import json
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import KFold

from lfunc_dynamics.spin_noise import SpinNoiseRig
from lfunc_dynamics.ml_cabinet import CoreCabinet

FEATS = ["var", "skew", "kurt", "ac1", "hf", "fano", "pair_m", "pair_dev"]
SPECIES = ["thermal", "shot", "quantum", "mixed"]
K, T = 4, 4
DC = 5.0


class ProbaWrap:
    def __init__(self, clf):
        self.clf = clf
    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1]


def main():
    rig = SpinNoiseRig(seed=3)
    rng = np.random.default_rng(11)
    fdicts, ycls, fracs, raws = [], [], [], []
    onehot = {"thermal": (1.0, 0.0, 0.0), "shot": (0.0, 1.0, 0.0), "quantum": (0.0, 0.0, 1.0)}

    for si, sp in enumerate(["thermal", "shot", "quantum"]):
        for _ in range(150):
            tr, nz = rig.trace(rng, sp, dc=DC)
            fdicts.append(rig.features(tr)); ycls.append(si)
            fracs.append(onehot[sp]); raws.append(tr)
    for _ in range(150):
        w = rng.uniform(0.4, 1.0, size=3)
        tr, info = rig.trace_mixed(rng, dc=DC, weights=w)
        fdicts.append(rig.features(tr)); ycls.append(3)
        fracs.append(info["fracs"]); raws.append(tr)

    ycls = np.array(ycls)
    F = np.array(fracs)
    raws = np.array(raws)
    X = np.array([[f[c] for c in FEATS] for f in fdicts], float)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd

    def std_fn(fdict, _ens):
        return (np.array([[fdict[c] for c in FEATS]], float) - mu) / sd

    # ---- 1) Классификация маршрутов (CoreCabinet, 4 OvR-кабинета) ----
    cabinets, thetas = {}, {}
    for c, sp in enumerate(SPECIES):
        yb = (ycls == c).astype(int)
        oof = np.zeros(len(yb))
        for tr, te in KFold(4, shuffle=True, random_state=0).split(Xs):
            m = HistGradientBoostingClassifier(random_state=0, max_iter=150, learning_rate=0.05)
            m.fit(Xs[tr], yb[tr]); oof[te] = m.predict_proba(Xs[te])[:, 1]
        bag = []
        for i in range(K * T):
            idx = np.random.default_rng(i).choice(len(yb), len(yb), replace=True)
            m = HistGradientBoostingClassifier(random_state=i, max_iter=150, learning_rate=0.05)
            m.fit(Xs[idx], yb[idx]); bag.append(ProbaWrap(m))
        surr = type("S", (), {"models": bag, "sigma_res": float(np.std(yb - oof))})()
        cab = CoreCabinet(surr, K=K, T=T)
        thetas[sp] = cab.calibrate(fdicts, ["h"] * len(fdicts), std_fn)
        cabinets[sp] = cab

    preds, sticks = [], []
    for fd in fdicts:
        vals = {sp: cabinets[sp].predict(fd, "h", std_fn) for sp in SPECIES}
        win = max(vals, key=lambda s: vals[s].value)
        preds.append(win); sticks.append(vals[win].sigma_tick)
    preds = np.array(preds); sticks = np.array(sticks)

    print("Классификация маршрутов (accuracy):")
    for si, sp in enumerate(SPECIES):
        print(f"  {sp:8s}: {(preds[ycls == si] == sp).mean():.3f}")

    # ---- 2) Регрессионная декомпозиция (доли дисперсии) ----
    oofF = np.zeros((len(ycls), 3))
    for c in range(3):
        oof = np.zeros(len(ycls))
        for tr, te in KFold(4, shuffle=True, random_state=0).split(Xs):
            m = HistGradientBoostingRegressor(random_state=0, max_iter=150, learning_rate=0.05)
            m.fit(Xs[tr], F[tr, c]); oof[te] = m.predict(Xs[te])
        oofF[:, c] = oof
    oofF = np.clip(oofF, 0.0, None)
    oofF /= oofF.sum(1, keepdims=True) + 1e-12
    mixed = ycls == 3
    mae_all = float(np.mean(np.abs(oofF - F)))
    mae_mixed = float(np.mean(np.abs(oofF[mixed] - F[mixed])))
    dom_true = F[mixed].argmax(1)
    dom_pred = oofF[mixed].argmax(1)
    acc_dom = float((dom_true == dom_pred).mean())
    print(f"\nДекомпозиция: OOF MAE долей = {mae_all:.3f} (mixed: {mae_mixed:.3f})")
    print(f"  доминирующая компонента на mixed: accuracy = {acc_dom:.3f}")
    names3 = ["thermal", "shot", "quantum"]
    for d in range(3):
        n = int((dom_true == d).sum())
        ok = int(((dom_true == d) & (dom_pred == d)).sum())
        print(f"    истинно домин. {names3[d]:8s}: {n:3d} шотов, верно {ok:3d}")

    # ---- 3) Witness: sigma_tick по классам ----
    print("\nsigma_tick победившего кабинета:")
    for si, sp in enumerate(SPECIES):
        print(f"  {sp:8s}: {sticks[ycls == si].mean():.4f} ± {sticks[ycls == si].std():.4f}")
    s_cl = sticks[(ycls == 0) | (ycls == 1)]
    s_mx = sticks[mixed]
    s_qu = sticks[ycls == 2]
    print(f"  mixed vs classical: MW p={mannwhitneyu(s_mx, s_cl, alternative='greater')[1]:.2e}")
    print(f"  mixed vs quantum  : MW p={mannwhitneyu(s_mx, s_qu, alternative='less')[1]:.2e}")

    # ---- 4) Очистка: маршрут → комбинация трактов ----
    print("\nОчистка DC-оценки (RMSE, истина 5.0):")
    def rmse(e): return float(np.sqrt(np.mean(np.square(e))))
    cases = [("thermal", False), ("shot", True), ("mixed", True)]
    for sp, clip in cases:
        si = SPECIES.index(sp)
        e0, e1 = [], []
        for _ in range(150):
            if sp == "mixed":
                w = rng.uniform(0.4, 1.0, size=3)
                x0, _ = rig.trace_mixed(rng, dc=DC, weights=w, chopped=False)
                x1, _ = rig.trace_mixed(rng, dc=DC, weights=w, chopped=True)
            else:
                x0, _ = rig.trace(rng, sp, dc=DC, chopped=False)
                x1, _ = rig.trace(rng, sp, dc=DC, chopped=True)
            e0.append(rig.readout(x0) - DC)
            e1.append(rig.readout(x1, clip=clip) - DC)
        r0, r1 = rmse(e0), rmse(e1)
        print(f"  {sp:8s}: до={r0:.4f} -> после={r1:.4f} (выигрыш {r0/r1:.1f}x)")
    print("  quantum : очистка отклонена (witness-only)")

    out = Path("spin_verdicts_v4.json")
    json.dump(dict(thetas={k: float(v) for k, v in thetas.items()},
                   preds=preds.tolist(), sigma_tick=sticks.tolist(),
                   true_class=ycls.tolist(), true_fracs=F.tolist(),
                   pred_fracs=oofF.tolist(),
                   mae_fracs=mae_all, acc_dominant=acc_dom),
              open(out, "w"), indent=1)
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()