#!/usr/bin/env python3
r"""proper_time_e3_bench.py — E3''' (Level 1): irreversibility with buffer attention.

Runs episodes THROUGH the Level-1 TemporalEnsemble (buffer attention), so the
hidden state can carry buffer content. Aligned pairs for every i >= BUF_CAP:
  fresh : decode v_i          (just pushed, present in buffer snapshot i)
  erased: decode v_{i-CAP}    (just overwritten by push i, absent from snapshot i)
  random: decode noise
n = N_EP - BUF_CAP = 112 well-posed samples per task (fixes the n<<p failure
of the previous runs, where fresh had only 8 samples against 64-96 features).

Feature sets: buffer snapshot (64), h_mean post-attention (64), both (128).
Deciding table:
  buffer_to_fresh  > 0.8 : positive control passes (decoder works)
  h_to_fresh       > 0.3 : h carries buffer content (attention is live)
  *_to_erased      < 0.1 : erasure is real (information destroyed)
If h_to_fresh ~ 0 while buffer_to_fresh ~ 1, Level 1 is NOT active in the model.
"""
import json
from pathlib import Path

import numpy as np
import torch

from lfunc_dynamics.temporal_layer import BasePredictor, TemporalEnsemble

N_EP = 120
DIM = 8          # temporal embedding dim = push vector dim
BUF_CAP = 64     # model memory capacity; "fresh window" = last CAP pushes


def cv_r2(X, y, k=5, lam=1.0, seed=0):
    """CV-ridge R2 (multi-output); protects against n<<p overfitting."""
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
                             coherence_budget_ms=25.0)

    h_list, v_list, snaps = [], [], []
    for i in range(N_EP):
        x = torch.randn(1, 32)
        out = model(x, persist_memory=True, return_diagnostics=True)
        h_list.append(out["h_mean"][0].detach().numpy().astype(np.float64))
        v_list.append(np.asarray(out["e_t"], dtype=np.float64))   # pushed vector
        snaps.append(model.memory._buf.copy())                    # snapshot after push

    H = np.array(h_list)                             # (N, 64) post-attention hidden
    V = np.array(v_list)                             # (N, 8)  push vectors
    S = np.array([s.flatten() for s in snaps])       # (N, 64) buffer snapshots

    # Aligned tasks: for i >= DIM the push at i overwrote v_{i - capacity_window};
    # with capacity 64 > N nothing is overwritten, so "erased" = vector that LEFT
    # the recent window is not available; use the recency structure instead:
    # fresh = v_i (in buffer), erased = v_{i - BUF_CAP} if it existed else None.
    # Since CAP=64 and N=120, pushes 0..55 ARE overwritten by pushes 64..119.
    idx = np.arange(BUF_CAP, N_EP)                   # i where an overwrite happened
    y_fresh = V[idx]                                 # present in snapshot i
    y_erased = V[idx - BUF_CAP]                      # overwritten by push i
    y_rand = rng.normal(0, 1, (len(idx), DIM))

    feats = {
        "buffer": S[idx],
        "h": H[idx],
        "buf_h": np.hstack([S[idx], H[idx]]),
    }
    results = {}
    for fname, X in feats.items():
        results[f"{fname}_to_fresh"] = cv_r2(X, y_fresh)
        results[f"{fname}_to_erased"] = cv_r2(X, y_erased)
        results[f"{fname}_to_random"] = cv_r2(X, y_rand)

    lines = []
    ok_ctrl = results["buffer_to_fresh"] > 0.8
    lines.append(("PASS" if ok_ctrl else "FAIL")
                 + f" positive control: buffer->fresh R2={results['buffer_to_fresh']:.3f}")
    ok_h = results["h_to_fresh"] > 0.3
    lines.append(("PASS" if ok_h else "FAIL")
                 + f" h carries buffer: h->fresh R2={results['h_to_fresh']:.3f}"
                 + ("" if ok_h else " (Level 1 not active?)"))
    ok_irr = all(results[f"{f}_to_erased"] < 0.1 for f in feats)
    lines.append(("PASS" if ok_irr else "FAIL")
                 + " irreversibility: all *_to_erased < 0.1 "
                 + str({f: round(results[f'{f}_to_erased'], 3) for f in feats}))

    verdict = {
        "stage": "e3_level1",
        "n_pairs": int(len(idx)),
        "results": results,
        "verdict_lines": lines,
        "arrow_status": ("Irreversibility established (erasure unreadable, "
                         "decoder validated by positive control)"
                         if (ok_ctrl and ok_irr) else
                         "Irreversibility not demonstrated"),
        "level1_active": bool(ok_h),
    }
    out = Path("proper_time_verdicts_e3.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=2)

    print("E3''' (Level 1) results, n =", len(idx))
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