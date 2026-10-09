#!/usr/bin/env python3
r"""test_pci_consistency.py — verify PCI is deterministic and reproducible.

Runs PCI on a FIXED random trajectory with fixed seed. If two calls
return different numbers, there is a bug. This must pass before any
integration comparison is valid.
"""
import numpy as np
from pci_reference import pci_from_traj, pci_batch


def main():
    # Fixed trajectory, fixed seed
    rng = np.random.default_rng(42)
    T, K, n = 512, 4, 64
    traj = rng.normal(0, 1, (T, K, n))
    
    # Call twice — must be identical
    pci_1 = pci_from_traj(traj)
    pci_2 = pci_from_traj(traj)
    
    print(f"PCI call 1: {pci_1:.6f}")
    print(f"PCI call 2: {pci_2:.6f}")
    print(f"Consistent: {abs(pci_1 - pci_2) < 1e-10}")
    
    # Batch call
    trajs = rng.normal(0, 1, (10, T, K, n))
    mean_pci, per_ep = pci_batch(trajs)
    print(f"Batch mean: {mean_pci:.6f} (std={np.std(per_ep):.6f})")
    
    # Historical reference values for sanity check:
    # step0 expected ~1.4 (old normalization, may differ)
    # step0.5 sweep expected ~0.07 (old normalization, may differ)
    # After this fix, ALL future runs use this module.
    print("\nNOTE: previous PCI values (1.484, 0.995, 0.073) came from")
    print("three different normalization schemes. This module is the")
    print("single source of truth going forward. Historical values are")
    print("NOT comparable with this one.")


if __name__ == "__main__":
    main()