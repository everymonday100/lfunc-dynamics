#!/usr/bin/env python3
r"""proper_time_e3_bench.py — E3''' final: irreversibility with buffer attention.

Runs episodes THROUGH TemporalEnsemble (persist_memory=True). For every
i >= BUF_CAP, push(i) overwrites v_{i-BUF_CAP}, so aligned pairs exist:
  fresh : decode v_i        (present in snapshot i; last written slot)
  erased: decode v_{i-CAP}  (overwritten by push i; absent from snapshot i)
  random: decode noise
Feature sets:
  buf_last     : the slot written at push i, read FROM the snapshot (8)
                 -> positive control: identity map, must be learnable
  buf_all      : full buffer snapshot (CAP*8 = 64)
  h            : hidden mean returned by the model (64)
  buf_last_h   : concatenation (72)
n = N_EP - BUF_CAP = 192 well-posed samples per task.
Decision rules:
  buf_last_to_fresh > 0.8 : decoder works (positive control passes)
  h_to_fresh        > 0.3 : Level-1 attention live (h carries buffer content)
  max(*_to_erased)  < 0.1 : erasure real => irreversibility established
The verdict also reports level1_present = hasattr(model, 'mem_query_proj'),
so a missing Level-1 patch in temporal_layer.py is detected by the bench.
"""
import json
from pathlib import Path

import numpy as np
import torch

from lfunc_dynamics.temporal_layer import BasePredictor, TemporalEnsemble

N_EP = 200
BUF_CAP = 8
DIM = 8


def cv_r2(X, y, k=5, lam=0.01, seed=0):
    """CV-ridge R2 (multi-output); weak regularization, n >> p here."""
    idx = np.random.default_rng(seed).permutation(len(y))
    folds = np.array_split(idx, k)
    pred = np.zeros_like(y, dtype=float)
    for f in folds:
        tr = np.setdiff1d(idx, f)
        A = np.hstack([X[tr], np.ones((len(tr), 1))])
        G = A.T @ A + lam * np.eye(A.shape[1])
        w = np.linalg.solve(G, A.T @ y[tr])
        Bm = np.hstack([X[f], np.ones((len(f), 1))])
        pred[f] = Bm @ w
    den = np.sum((y - y.mean(0)) ** 2)
    return float(1 - np.sum((y - pred) ** 2) / den) if den > 0 else 0.0


def main():
    rng = np.random.default_rng(42)
    torch.manual_seed(0)
    factory = lambda: BasePredictor(in_dim=32, hidden=64, out_dim=1)
    model = TemporalEnsemble(factory, n_cores=4, n_ticks=4,
                             coherence_budget_ms=25.0,
                             memory_capacity=BUF_CAP)
    level1 = hasattr(model, "mem_query_proj")

    h_list, v_list, snaps = [], [], []
    for i in range(N_EP):
        x = torch.randn(1, 32)
        out = model(x, persist_memory=True, return_diagnostics=True)
        h_list.append(out["h_mean"][0].detach().numpy().astype(np.float64))
        v_list.append(np.asarray(out["e_t"], dtype=np.float64))   # pushed vector
        snaps.append(model.memory._buf.copy())                    # after push i

    H = np.array(h_list)                                          # (N, 64)
    V = np.array(v_list)                                          # (N, 8)
    slot = np.arange(N_EP) % BUF_CAP                              # slot written at i
    BUF_LAST = np.array([snaps[i][slot[i]] for i in range(N_EP)])  # (N, 8) from snapshot
    BUF_ALL = np.array([s.flatten() for s in snaps])               # (N, 64)

    idx = np.arange(BUF_CAP, N_EP)
    y_fresh = V[idx]
    y_erased = V[idx - BUF_CAP]
    y_rand = rng.normal(0, 1, (len(idx), DIM))

    feats = {
        "buf_last": BUF_LAST[idx],
        "buf_all": BUF_ALL[idx],
        "h": H[idx],
        "buf_last_h": np.hstack([BUF_LAST[idx], H[idx]]),
    }
    results = {}
    for fname, X in feats.items():
        results[f"{fname}_to_fresh"] = cv_r2(X, y_fresh)
        results[f"{fname}_to_erased"] = cv_r2(X, y_erased)
        results[f"{fname}_to_random"] = cv_r2(X, y_rand)

    lines = []
    ok_ctrl = results["buf_last_to_fresh"] > 0.8
    lines.append(("PASS" if ok_ctrl else "FAIL") +
                 f" positive control: buf_last->fresh R2={results['buf_last_to_fresh']:.3f}")
    ok_h = results["h_to_fresh"] > 0.3
    lines.append(("PASS" if ok_h else "FAIL") +
                 f" h carries buffer: h->fresh R2={results['h_to_fresh']:.3f}" +
                 ("" if ok_h else " (Level 1 not active?)"))
    ok_irr = max(results[f"{f}_to_erased"] for f in feats) < 0.1
    lines.append(("PASS" if ok_irr else "FAIL") +
                 " irreversibility: max *_to_erased < 0.1 " +
                 str({f: round(results[f'{f}_to_erased'], 3) for f in feats}))
    if not level1:
        lines.append("WARN temporal_layer.py has NO mem_query_proj: "
                     "Level-1 patch not applied; h_to_fresh ~ 0 expected")

    verdict = {
        "stage": "e3_level1_final",
        "n_pairs": int(len(idx)),
        "level1_present": bool(level1),
        "results": results,
        "verdict_lines": lines,
        "arrow_status": ("Irreversibility established (erasure unreadable, "
                         "decoder validated)" if (ok_ctrl and ok_irr)
                         else "Irreversibility not demonstrated"),
    }
    out = Path("proper_time_verdicts_e3.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=2)

    print(f"E3''' final, n={len(idx)}, level1_present={level1}")
    print("=" * 64)
    for k, v in results.items():
        print(f"{k:20s}: {v:7.3f}")
    print("=" * 64)
    for ln in lines:
        print(ln)
    print("Arrow status:", verdict["arrow_status"])
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()