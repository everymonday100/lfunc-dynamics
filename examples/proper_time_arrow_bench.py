#!/usr/bin/env python3
r"""proper_time_arrow_bench.py (v2) — single-regime protocol, reviewer fixes.

Part A (confound, smooth only): Spearman(Lambda, n_erased) within regime for
  SurpriseMemoryV2 (q=0.3, model-based surprise, per-episode quantile
  threshold) vs FIFO (constant erasures = bookkeeping).
Part B (behavioral arrow): transition-asymmetry classifier on shuffled pairs,
  CV split BY EPISODE, three populations:
    cold : far-init episodes (transient present);
    warm : burn-in stationary driven episodes (no transient);
    iid  : states drawn i.i.d. from a stationary pool (kills positional and
           marginal-distribution bias; chance guaranteed if no artifact).
  Features: states / +erasure events / +dLambda; ablation by CIRCULAR TIME
  SHIFT of event/dLambda sequences within episode (distribution preserved).
Level-2 memo (do not forget): fix K_slow/K_fast = 8 first; initialize h_slow
  from h_fast at episode start; only then optimize the ratio.
"""
import json
from pathlib import Path

import numpy as np

from lfunc_dynamics.proper_time import TemporalMemory, SurpriseMemoryV2, RecurrentCell
from proper_time_bench import episode, ranks

REGIME = "smooth"
N_CONF = 120
N_ARROW = 60
CAP, DIM, Q = 8, 8, 0.3
N_PAIRS = 8


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    ra, rb = ranks(a), ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def run_memories(h_traj):
    T = len(h_traj)
    fifo = TemporalMemory(capacity=CAP, dim=DIM)
    sur = SurpriseMemoryV2(capacity=CAP, dim=DIM, state_dim=h_traj.shape[1])
    n_fifo = 0
    for t in range(T):
        v = h_traj[t][:DIM].astype(np.float32)
        if len(fifo) == CAP:
            n_fifo += 1
        fifo.push(v)
        sur.push_step(h_traj[t], v)  # <-- push_step вместо push
    dl = np.concatenate([[0.0], np.linalg.norm(np.diff(h_traj, axis=0), axis=1)])
    return n_fifo, sur.erased, np.asarray(sur.events, int), dl


def stationary_pool(seed=3, n=4000):
    cell = RecurrentCell(seed=seed)
    rng = np.random.default_rng(seed)
    hs = []
    for _ in range(n):
        hs.append(cell.step(rng.normal(0.0, 1.0, cell.n)).copy())
    return np.asarray(hs[500:])          # отбрасываем транзиент


def pair_dataset(h_trajs, ev_seqs, dl_seqs, n_pairs=N_PAIRS, seed=0):
    rng = np.random.default_rng(seed)
    eids, X, E, D, y = [], [], [], [], []
    for eid, (H, ev, dl) in enumerate(zip(h_trajs, ev_seqs, dl_seqs)):
        T = len(H)
        ts = rng.choice(T - 1, size=min(n_pairs, T - 1), replace=False)
        for t in ts:
            for fwd in (1, 0):
                u, v = (H[t], H[t + 1]) if fwd else (H[t + 1], H[t])
                eids.append(eid)
                X.append(np.concatenate([u, v]))
                E.append(float(ev[t])); D.append(float(dl[t])); y.append(fwd)
    order = rng.permutation(len(y))      # пары перемешаны
    return (np.array(eids)[order], np.array(X)[order], np.array(E)[order],
            np.array(D)[order], np.array(y)[order])


def cv_acc_episode(X, y, eids, k=5, lam=1.0, seed=0):
    eps = np.random.default_rng(seed).permutation(np.unique(eids))
    correct = tot = 0
    for f in np.array_split(eps, k):
        te = np.isin(eids, f); tr = ~te
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
        A = np.hstack([(X[tr] - mu) / sd, np.ones((tr.sum(), 1))])
        G = A.T @ A + lam * np.eye(A.shape[1])
        w = np.linalg.solve(G, A.T @ (2 * y[tr] - 1))
        B = np.hstack([(X[te] - mu) / sd, np.ones((te.sum(), 1))])
        correct += int(((B @ w > 0).astype(int) == y[te]).sum()); tot += int(te.sum())
    return correct / tot


def boot_ci_acc(X, y, eids, B=150, seed=0):
    rng = np.random.default_rng(seed)
    eps = np.unique(eids)
    vals = []
    for _ in range(B):
        sam = rng.choice(eps, len(eps), replace=True)
        mask = np.concatenate([np.where(eids == e)[0] for e in sam])
        vals.append(cv_acc_episode(X[mask], y[mask], eids[mask], k=3))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def shift_within(vals, eids, seed):
    rng = np.random.default_rng(seed)
    out = vals.copy()
    for e in np.unique(eids):
        m = eids == e
        out[m] = np.roll(vals[m], int(rng.integers(1, 8)))
    return out


def main():
    rng = np.random.default_rng(7)
    res = {"regime": REGIME, "q": Q}

    # ---- Part A: confound within single regime ----
    lams, fifo_n, sur_n = [], [], []
    conf_trajs = []
    for i in range(N_CONF):
        ep = episode(REGIME, rng, i % 5)
        nf, ns, _ev, _dl = run_memories(ep["h_traj"])
        lams.append(ep["snap"]["lam"]); fifo_n.append(nf); sur_n.append(ns)
        conf_trajs.append(ep["h_traj"])
    res["spearman_fifo"] = spearman(lams, fifo_n)
    res["spearman_surprise_v2"] = spearman(lams, sur_n)
    res["fifo_erasure_std"] = float(np.std(fifo_n))
    res["surprise_erasure_std"] = float(np.std(sur_n))
    res["level1_success"] = bool(abs(res["spearman_surprise_v2"]) < 0.5)

    # ---- Part B: behavioral arrow, three populations ----
    cold_H, cold_E, cold_D = [], [], []
    for i in range(N_ARROW):
        ep = episode(REGIME, rng, (i + 200) % 5)
        _nf, _ns, ev, dl = run_memories(ep["h_traj"])
        cold_H.append(ep["h_traj"]); cold_E.append(ev); cold_D.append(dl)
    warm_H, warm_E, warm_D = [], [], []
    for i in range(N_ARROW):
        ep = episode(REGIME, rng, (i + 300) % 5, warm=True)
        _nf, _ns, ev, dl = run_memories(ep["h_traj"])
        warm_H.append(ep["h_traj"]); warm_E.append(ev); warm_D.append(dl)
    pool = stationary_pool()
    rngp = np.random.default_rng(11)
    iid_H, iid_E, iid_D = [], [], []
    ev_pool = np.concatenate(cold_E); dl_pool = np.concatenate(cold_D)
    for i in range(N_ARROW):
        H = pool[rngp.integers(0, len(pool), 64)]
        iid_H.append(H)
        iid_E.append(ev_pool[rngp.integers(0, len(ev_pool), 64)])
        iid_D.append(dl_pool[rngp.integers(0, len(dl_pool), 64)])

    out = {}
    for tag, (Hs, Es, Ds) in (("cold", (cold_H, cold_E, cold_D)),
                              ("warm", (warm_H, warm_E, warm_D)),
                              ("iid", (iid_H, iid_E, iid_D))):
        eids, X, E, D, y = pair_dataset(Hs, Es, Ds)
        out[f"acc_{tag}_states"] = cv_acc_episode(X, y, eids)
        out[f"acc_{tag}_states_ci"] = list(boot_ci_acc(X, y, eids))
        if tag == "cold":
            out["acc_cold_plus_erasure"] = cv_acc_episode(np.hstack([X, E[:, None]]), y, eids)
            out["acc_cold_plus_dlam"] = cv_acc_episode(np.hstack([X, D[:, None]]), y, eids)
            out["acc_cold_erasure_timeshift"] = cv_acc_episode(
                np.hstack([X, shift_within(E, eids, 5)[:, None]]), y, eids)
            out["acc_cold_dlam_timeshift"] = cv_acc_episode(
                np.hstack([X, shift_within(D, eids, 6)[:, None]]), y, eids)
    res.update(out)
    lo, hi = out["acc_cold_states_ci"]
    wlo, whi = out["acc_warm_states_ci"]
    ilo, ihi = out["acc_iid_states_ci"]
    res["arrow_discovered"] = bool(lo > 0.5)
    res["controls_ok"] = bool(wlo <= 0.5 <= whi and ilo <= 0.5 <= ihi)
    res["erasure_used_aligned"] = bool(out["acc_cold_plus_erasure"] -
                                       out["acc_cold_erasure_timeshift"] > 0.02)

    verdict = dict(stage="arrow_level1_v2", **res)
    outp = Path("proper_time_verdicts_arrow.json")
    with open(outp, "w") as fh:
        json.dump(verdict, fh, indent=1)
    print(json.dumps(verdict, indent=1))
    print(f"Saved: {outp}")


if __name__ == "__main__":
    main()