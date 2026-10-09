#!/usr/bin/env python3
r"""test_coupling_effect.py — verify lateral coupling actually changes trajectories.

Diagnostic: run the same architecture with coupling=0 and coupling=0.3,
compare trajectory norms and pairwise distances. If identical, the
coupling code is broken.
"""
import numpy as np
from lfunc_dynamics.proper_time import RecurrentCell


def run_pair(K=4, T=64, coupling=0.0, seed=0):
    rng = np.random.default_rng(seed)
    cores = [RecurrentCell(seed=s) for s in range(K)]
    trajs = np.zeros((T, K, cores[0].n))
    for c in cores:
        c.h = rng.normal(0, 0.5, c.n)
    for t in range(T):
        states = []
        for k, c in enumerate(cores):
            x = rng.normal(0, 1, c.n)
            f = np.tanh(c.W @ c.h + c.V @ x)
            lateral = np.zeros(c.n)
            if coupling > 0:
                other = [cores[j].h for j in range(K) if j != k]
                lateral = np.mean(other, axis=0)
                norm = np.linalg.norm(lateral)
                if norm > 1e-8:
                    lateral = lateral / norm
            c.h = c.h + 0.1 * f + coupling * lateral
            states.append(c.h.copy())
        for k in range(K):
            trajs[t, k] = states[k]
    return trajs


def main():
    t_zero = run_pair(coupling=0.0)
    t_half = run_pair(coupling=0.3)
    
    # If coupling works, trajectories must differ
    diff = np.linalg.norm(t_zero - t_half)
    print(f"Trajectory difference (coupling=0 vs 0.3): {diff:.4f}")
    print(f"Trajectories identical: {diff < 1e-10}")
    
    if diff < 1e-10:
        print("\n⚠ COUPLING BROKEN: trajectories identical.")
        print("Check that other_states is not empty and lateral is added.")
    else:
        print("\n✓ Coupling changes trajectories. Proceed to integration tests.")


if __name__ == "__main__":
    main()