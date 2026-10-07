#!/usr/bin/env python3
r"""proper_time_e3_bench.py — E3'': positive control for irreversibility.

Tests whether erased buffer vectors are recoverable from hidden state + buffer.
Three feature sets: buffer-only (64), h-only (32), buffer+h (96).
Three targets: fresh (currently in buffer), erased (overwritten), random.

Deciding table:
  R²(buffer→fresh) ≈ 1   : decoder works (positive control passes)
  R²(h→fresh) > 0.5      : h carries buffer information
  R²(any→erased) ≈ 0     : irreversibility is real (info destroyed)

If R²(h→fresh) ≈ 0: buffer and h live in different subspaces; arrow-through-h
question is architecturally undecidable in this setup.
"""
import json
from pathlib import Path

import numpy as np

from lfunc_dynamics.proper_time import TemporalMemory
from proper_time_bench import episode


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
    N_EP = 120
    BUF_CAP = 8
    DIM = 8  # push-vector dimension (matches TemporalMemory default)

    # Generate episodes and collect (h_mean, push_vec) pairs
    h_list, v_list = [], []
    for i in range(N_EP):
        ep = episode("white1", rng, i % 5)
        h = ep["h_mean"].astype(np.float64)  # (32,)
        v = ep["h_mean"][:DIM].astype(np.float32)  # (8,) — what gets pushed
        h_list.append(h)
        v_list.append(v)

    # Fill ring buffer sequentially
    mem = TemporalMemory(capacity=BUF_CAP, dim=DIM)
    for v in v_list:
        mem.push(v)

    # Identify fresh (currently in buffer) and erased (overwritten)
    fresh_indices = list(range(N_EP - BUF_CAP, N_EP))
    erased_indices = list(range(N_EP - BUF_CAP))

    # Build feature matrices
    buf_flat = mem._buf.flatten().astype(np.float64)  # (64,)
    h_stack = np.array(h_list)  # (N_EP, 32)
    v_stack = np.array(v_list)  # (N_EP, 8)

    # Three feature sets
    X_buf = np.tile(buf_flat, (N_EP, 1))  # (N_EP, 64)
    X_h = h_stack.copy()  # (N_EP, 32)
    X_buf_h = np.hstack([X_buf, X_h])  # (N_EP, 96)

    # Targets: fresh, erased, random
    y_fresh = v_stack[fresh_indices]  # (BUF_CAP, 8)
    y_erased = v_stack[erased_indices]  # (N_EP - BUF_CAP, 8)
    y_random = rng.normal(0, 1, (N_EP, DIM)).astype(np.float32)  # (N_EP, 8)

    # Replicate targets to match N_EP for consistent CV
    y_fresh_rep = np.tile(y_fresh, (N_EP // BUF_CAP, 1))[:N_EP]
    y_erased_rep = np.tile(y_erased, (N_EP // (N_EP - BUF_CAP) + 1, 1))[:N_EP]

    # Compute R² for each (features, target) pair
    results = {
        "buffer_to_fresh": cv_r2(X_buf, y_fresh_rep),
        "buffer_to_erased": cv_r2(X_buf, y_erased_rep),
        "buffer_to_random": cv_r2(X_buf, y_random),
        "h_to_fresh": cv_r2(X_h, y_fresh_rep),
        "h_to_erased": cv_r2(X_h, y_erased_rep),
        "h_to_random": cv_r2(X_h, y_random),
        "buf_h_to_fresh": cv_r2(X_buf_h, y_fresh_rep),
        "buf_h_to_erased": cv_r2(X_buf_h, y_erased_rep),
        "buf_h_to_random": cv_r2(X_buf_h, y_random),
    }

    # Decision logic
    verdict_lines = []
    if results["buffer_to_fresh"] > 0.8:
        verdict_lines.append("✓ Positive control PASSED: buffer→fresh R² > 0.8")
    else:
        verdict_lines.append("✗ Positive control FAILED: buffer→fresh R² < 0.8 (decoder broken)")

    if results["h_to_fresh"] > 0.5:
        verdict_lines.append("✓ h carries buffer info: h→fresh R² > 0.5")
    else:
        verdict_lines.append("✗ h and buffer in different subspaces: h→fresh R² < 0.5")

    if results["buf_h_to_erased"] < 0.1:
        verdict_lines.append("✓ Irreversibility REAL: buf+h→erased R² ≈ 0 (info destroyed)")
    else:
        verdict_lines.append("✗ Irreversibility questionable: buf+h→erased R² > 0.1 (info recoverable)")

    verdict = {
        "stage": "e3_doubleprime",
        "n_episodes": N_EP,
        "buffer_capacity": BUF_CAP,
        "n_fresh": BUF_CAP,
        "n_erased": N_EP - BUF_CAP,
        "results": results,
        "verdict_lines": verdict_lines,
        "arrow_status": (
            "Irreversibility established" if results["buf_h_to_erased"] < 0.1 else
            "Irreversibility not demonstrated"
        ),
    }

    out = Path("proper_time_verdicts_e3.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=2)

    print("E3'' Results:")
    print("=" * 60)
    for k, v in results.items():
        print(f"{k:20s}: {v:7.3f}")
    print("=" * 60)
    print("\nVerdict:")
    for line in verdict_lines:
        print(line)
    print(f"\nArrow status: {verdict['arrow_status']}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()