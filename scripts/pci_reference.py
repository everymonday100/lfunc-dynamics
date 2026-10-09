#!/usr/bin/env python3
r"""pci_reference.py — PCI reference implementation.

Single canonical PCI computation used by ALL integration scripts.
Definition: LZ complexity of binarized response, normalized by
the maximum LZ for a binary sequence of the same length and entropy.

CASATI-ALIGNED: PCI = LZ(binarized) / LZ_max, where
  LZ_max = n / log2(n) for binary sequence of length n
  (upper bound for LZ complexity; see Casarotto et al. 2016)

This module replaces all ad-hoc compute_pci_from_traj implementations.
"""
import numpy as np


def lz_complexity(bits):
    """LZ76 phrase count via incremental dictionary."""
    s = "".join("1" if b else "0" for b in bits)
    seen = set()
    c = 0
    i = 0
    n = len(s)
    while i < n:
        j = i + 1
        while j <= n and s[i:j] in seen:
            j += 1
        seen.add(s[i:j])
        c += 1
        i = j
    return float(c)


def pci_from_traj(traj, n_neurons=16, seed=0):
    """PCI from a single trajectory: shape (T, K, n_hidden) or (T, n_features).
    Returns PCI value in [0, 1] (capped at 1.0)."""
    traj = np.asarray(traj, float)
    if traj.ndim == 3:
        flat = traj[:, :, :n_neurons].flatten()
    elif traj.ndim == 2:
        flat = traj[:, :n_neurons].flatten()
    else:
        flat = traj.flatten()
    
    if not np.all(np.isfinite(flat)):
        return float("nan")
    
    bits = (flat > np.median(flat)).astype(np.uint8)
    m = len(bits)
    if m < 10:
        return float("nan")
    
    c = lz_complexity(bits)
    # Casarotto-aligned normalization: LZ_max = n / log2(n)
    lz_max = m / np.log2(m)
    pci = c / lz_max
    return float(min(pci, 1.0))  # cap at 1.0


def pci_batch(trajs, n_neurons=16):
    """PCI over multiple episodes: shape (N, T, K, n) or (N, T, n).
    Returns (mean_pci, list_of_per_episode_pci)."""
    vals = []
    for ep in range(trajs.shape[0]):
        v = pci_from_traj(trajs[ep], n_neurons=n_neurons)
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return float("nan"), []
    return float(np.mean(vals)), vals