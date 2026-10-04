#!/usr/bin/env python3
r"""spin_noise_bench.py — ватчдог CoreCabinet на шумах спинтроники.
Проверки: (1) классификация видов; (2) sigma_tick концентрируется на квантовом
виде (фазовая защёлка = граница решения); (3) очистка улучшает оценку DC
только для thermal и shot; quantum — witness-only."""
import json
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold

from lfunc_dynamics.spin_noise import SpinNoiseRig
from lfunc_dynamics.ml_cabinet import CoreCabinet

FEATS = ["var", "skew", "kurt", "ac1", "hf", "fano", "pair_m", "pair_dev"]
SPECIES = ["thermal", "shot", "quantum"]
K, T = 4, 4
DC = 5.0


class ProbaWrap:
    """Адаптер: ячейки CoreCabinet требуют .predict(X) -> скаляр."""
    def __init__(self, clf):
        self.clf = clf
    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1]


def main():
    rig = SpinNoiseRig(seed=3)
    rng = np.random.default_rng(11)
    fdicts, y, raws = [], [], []
    for si, sp in enumerate(SPECIES):
        for _ in range(150):
            tr, nz = rig.trace(rng, sp, dc=DC)
            fdicts.append(rig.features(tr)); y.append(si); raws.append(tr)
    y = np.array(y)
    X = np.array([[f[c] for c in FEATS] for f in fdicts], float)
    mu, sd = X.mean(0), X.std(0) + 1e-12
    Xs = (X - mu) / sd

    def std_fn(fdict, _ens):
        return (np.array([[fdict[c] for c in FEATS]], float) - mu) / sd

    cabinets, thetas = {}, {}
    for c, sp in enumerate(SPECIES):
        yb = (y == c).astype(int)
        oof = np.zeros(len(yb))
        for tr, te in KFold(4, shuffle=True, random_state=0).split(Xs):
            m = HistGradientBoostingClassifier(random_state=0, max_iter=150,
                                               learning_rate=0.05)
            m.fit(Xs[tr], yb[tr])
            oof[te] = m.predict_proba(Xs[te])[:, 1]
        bag = []
        for i in range(K * T):
            idx = np.random.default_rng(i).choice(len(yb), len(yb), replace=True)
            m = HistGradientBoostingClassifier(random_state=i, max_iter=150,
                                               learning_rate=0.05)
            m.fit(Xs[idx], yb[idx])
            bag.append(ProbaWrap(m))
        surr = type("S", (), {"models": bag, "sigma_res": float(np.std(yb - oof))})()
        cab = CoreCabinet(surr, K=K, T=T)
        thetas[sp] = cab.calibrate(fdicts, ["h"] * len(fdicts), std_fn)
        cabinets[sp] = cab
    print("theta_tick:", {k: round(v, 4) for k, v in thetas.items()})

    preds, sticks, quads = [], [], []
    for fd in fdicts:
        vals = {sp: cabinets[sp].predict(fd, "h", std_fn) for sp in SPECIES}
        win = max(vals, key=lambda s: vals[s].value)
        preds.append(win)
        sticks.append(vals[win].sigma_tick)
        quads.append(vals[win].quadrant)
    preds = np.array(preds); sticks = np.array(sticks)

    print("\nКлассификация (accuracy по видам):")
    for si, sp in enumerate(SPECIES):
        acc = (preds[y == si] == sp).mean()
        print(f"  {sp:8s}: {acc:.3f}")

    print("\nsigma_tick победившего кабинета, по истинному виду:")
    for si, sp in enumerate(SPECIES):
        print(f"  {sp:8s}: {sticks[y == si].mean():.4f}")
    sq = sticks[y == 2]; sc = sticks[y != 2]
    u, p = mannwhitneyu(sq, sc, alternative="greater")
    print(f"  quantum vs classical: MW p={p:.2e}  <- witness квантового вида")

    # --- Очистка: честное сравнение readout-трактов ---
    print("\nОчистка DC-оценки (RMSE, истина 5.0):")
    for sp, clip in [("thermal", False), ("shot", True)]:
        e0, e1 = [], []
        for _ in range(150):
            x0, _ = rig.trace(rng, sp, dc=DC, chopped=False)   # прямой тракт
            x1, _ = rig.trace(rng, sp, dc=DC, chopped=True)    # чоппер (+ клип для shot)
            e0.append(rig.readout(x0) - DC)
            e1.append(rig.readout(x1, clip=clip) - DC)
        r0 = np.sqrt(np.mean(np.square(e0))); r1 = np.sqrt(np.mean(np.square(e1)))
        print(f"  {sp:8s}: до={r0:.4f} -> после={r1:.4f} (выигрыш {r0/r1:.1f}x)")
    print("  quantum : очистка отклонена (witness-only)")

    out = Path("spin_verdicts.json")
    json.dump(dict(thetas={k: float(v) for k, v in thetas.items()},
                   preds=preds.tolist(), sigma_tick=sticks.tolist(),
                   quadrants=quads, true=y.tolist()),
              open(out, "w"), indent=1)
    print(f"\nСохранено: {out}")


if __name__ == "__main__":
    main()