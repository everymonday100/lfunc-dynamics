#!/usr/bin/env python3
r"""spin_noise_bench.py (v6) — disjoint train/test bench for the spintronic watchdog.
Train: 1800 traces (450/class); test: 600 disjoint traces (150/class).
Blocks:
 1. simplex coverage of mixed test traces (Dirichlet fracs);
 2. route classification (4 OvR CoreCabinets), train -> test;
 3. variance-fraction decomposition regressor, train -> test;
 4. sigma_tick witness stats on test;
 5. cleaning gains (chopper / MAD clip / composed);
 6. verdicts dump (context-managed, no ResourceWarning).
Requires spin_noise.py v2.4 (trace_mixed returns dict(fracs=..., weights=...)).
"""
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
ONEHOT = {"thermal": (1.0, 0.0, 0.0), "shot": (0.0, 1.0, 0.0),
          "quantum": (0.0, 0.0, 1.0)}
K, T = 4, 4
DC = 5.0
TR_PER, TE_PER = 450, 150


class ProbaWrap:
    """Adapter: CoreCabinet cells need .predict(X) -> 1-D array."""
    def __init__(self, clf):
        self.clf = clf

    def predict(self, X):
        return self.clf.predict_proba(X)[:, 1]


def _gen(rig, rng, n, species=None):
    """n traces; species=None -> mixed with Dirichlet variance fracs."""
    feats, labels, fracs = [], [], []
    for _ in range(n):
        if species is None:
            tr, info = rig.trace_mixed(rng, dc=DC)
            fr, lab = info["fracs"], 3
        else:
            tr, _ = rig.trace(rng, species, dc=DC)
            fr, lab = ONEHOT[species], SPECIES.index(species)
        feats.append(rig.features(tr))
        labels.append(lab)
        fracs.append(fr)
    return feats, np.array(labels, int), np.array(fracs, float)


def main():
    rig = SpinNoiseRig(seed=3)
    rng = np.random.default_rng(11)

    # ---- generation: disjoint train / test ----
    trF, trY, trFr, teF, teY, teFr = [], [], [], [], [], []
    for sp in ["thermal", "shot", "quantum"]:
        f, y, fr = _gen(rig, rng, TR_PER, sp); trF += f; trY.append(y); trFr.append(fr)
        f, y, fr = _gen(rig, rng, TE_PER, sp); teF += f; teY.append(y); teFr.append(fr)
    f, y, fr = _gen(rig, rng, TR_PER); trF += f; trY.append(y); trFr.append(fr)
    f, y, fr = _gen(rig, rng, TE_PER); teF += f; teY.append(y); teFr.append(fr)
    trY = np.concatenate(trY); trFr = np.concatenate(trFr)
    teY = np.concatenate(teY); teFr = np.concatenate(teFr)

    Xtr = np.array([[f[c] for c in FEATS] for f in trF], float)
    Xte = np.array([[f[c] for c in FEATS] for f in teF], float)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-12
    Xtr_s, Xte_s = (Xtr - mu) / sd, (Xte - mu) / sd

    def std_fn(fdict, _ens):
        return (np.array([[fdict[c] for c in FEATS]], float) - mu) / sd

    # ---- 1) simplex coverage (test mixed) ----
    Fm = teFr[teY == 3]
    assert Fm.ndim == 2 and Fm.shape[1] == 3, \
        f"fracs must be (N,3): check ONEHOT triples and info['fracs']; got {Fm.shape}"
    dom = Fm.argmax(axis=1)
    print(f"Simplex coverage (mixed test): thermal-dom {np.mean(dom == 0):.2f}, "
          f"shot-dom {np.mean(dom == 1):.2f}, quantum-dom {np.mean(dom == 2):.2f}")

    # ---- 2) route classification cabinets (train -> test) ----
    cabinets, thetas, preds, sticks = {}, {}, [], []
    for c, sp in enumerate(SPECIES):
        yb = (trY == c).astype(int)
        oof = np.zeros(len(yb))
        for tr, te in KFold(4, shuffle=True, random_state=0).split(Xtr_s):
            m = HistGradientBoostingClassifier(random_state=0, max_iter=150,
                                               learning_rate=0.05)
            m.fit(Xtr_s[tr], yb[tr]); oof[te] = m.predict_proba(Xtr_s[te])[:, 1]
        bag = []
        for i in range(K * T):
            idx = np.random.default_rng(i).choice(len(yb), len(yb), replace=True)
            m = HistGradientBoostingClassifier(random_state=i, max_iter=150,
                                               learning_rate=0.05)
            m.fit(Xtr_s[idx], yb[idx])
            bag.append(ProbaWrap(m))
        surr = type("S", (), {"models": bag,
                              "sigma_res": float(np.std(yb - oof))})()
        cab = CoreCabinet(surr, K=K, T=T)
        thetas[sp] = cab.calibrate(trF, ["h"] * len(trF), std_fn)
        cabinets[sp] = cab
    for fd in teF:
        vals = {sp: cabinets[sp].predict(fd, "h", std_fn) for sp in SPECIES}
        win = max(vals, key=lambda s: vals[s].value)
        preds.append(win); sticks.append(vals[win].sigma_tick)
    preds = np.array(preds); sticks = np.array(sticks)

    print("\nRoute accuracy (disjoint test, 150/class):")
    for si, sp in enumerate(SPECIES):
        print(f"  {sp:8s}: {(preds[teY == si] == sp).mean():.3f}")

    # ---- 3) decomposition regressor (train -> test) ----
    pred_fr = np.zeros((len(teY), 3))
    for c in range(3):
        m = HistGradientBoostingRegressor(random_state=0, max_iter=150,
                                          learning_rate=0.05)
        m.fit(Xtr_s, trFr[:, c])
        pred_fr[:, c] = m.predict(Xte_s)
    pred_fr = np.clip(pred_fr, 0.0, None)
    pred_fr /= pred_fr.sum(1, keepdims=True) + 1e-12
    mx = teY == 3
    mae_mix = float(np.mean(np.abs(pred_fr[mx] - teFr[mx])))
    dom_pred = pred_fr[mx].argmax(1)
    acc_dom = float((dom_pred == dom).mean())
    print(f"\nDecomposition (test): MAE fracs on mixed = {mae_mix:.3f}; "
          f"dominant accuracy = {acc_dom:.3f}")
    for d in range(3):
        n = int((dom == d).sum()); ok = int(((dom == d) & (dom_pred == d)).sum())
        print(f"  true-dom {SPECIES[d]:8s}: {n:3d}, correct {ok:3d}")

    # ---- 4) witness ----
    print("\nsigma_tick on test (winning cabinet):")
    for si, sp in enumerate(SPECIES):
        s = sticks[teY == si]
        print(f"  {sp:8s}: {s.mean():.4f} +/- {s.std():.4f}")
    s_cl = sticks[(teY == 0) | (teY == 1)]
    p_mc = mannwhitneyu(sticks[mx], s_cl, alternative="greater")[1]
    p_mq = mannwhitneyu(sticks[mx], sticks[teY == 2], alternative="less")[1]
    print(f"  mixed vs classical: MW p={p_mc:.2e}")
    print(f"  mixed vs quantum  : MW p={p_mq:.2e}")

    # ---- 5) cleaning ----
    print("\nCleaning DC estimate (RMSE, truth 5.0):")
    for sp, clip in [("thermal", False), ("shot", True), ("mixed", True)]:
        e0, e1 = [], []
        for _ in range(150):
            if sp == "mixed":
                x0, _ = rig.trace_mixed(rng, dc=DC, chopped=False)
                x1, _ = rig.trace_mixed(rng, dc=DC, chopped=True)
            else:
                x0, _ = rig.trace(rng, sp, dc=DC, chopped=False)
                x1, _ = rig.trace(rng, sp, dc=DC, chopped=True)
            e0.append(rig.readout(x0) - DC)
            e1.append(rig.readout(x1, clip=clip) - DC)
        r0 = float(np.sqrt(np.mean(np.square(e0))))
        r1 = float(np.sqrt(np.mean(np.square(e1))))
        print(f"  {sp:8s}: before={r0:.4f} -> after={r1:.4f} (gain {r0/r1:.1f}x)")
    print("  quantum : cleaning declined (witness-only)")

    # ---- 6) dump ----
    out = Path("spin_verdicts_v6.json")
    with open(out, "w") as fh:
        json.dump(dict(thetas={k: float(v) for k, v in thetas.items()},
                       preds=preds.tolist(), sigma_tick=sticks.tolist(),
                       true_class=teY.tolist(), true_fracs=teFr.tolist(),
                       pred_fracs=pred_fr.tolist(),
                       mae_fracs_mixed=mae_mix, acc_dominant=acc_dom),
                  fh, indent=1)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()