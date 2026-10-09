#!/usr/bin/env python3
r"""scripts/selfhood_step0_baselines.py — Step 0: close measurement gaps.

Three gaps that currently make the selfhood roadmap untestable:

  A. warm-A3 at a=0.7
     Is T-breaking stationary (on warm/stationary episodes) or just a
     cold-start relaxation transient? Tests A3 on warm episodes under
     the asymmetric update, both full and half-split. Gate:
         warm-A3 CI excludes 0   ->  stationary T-breaking confirmed,
                                     T1 (asymmetric update) becomes viable
         warm-A3 CI contains 0   ->  T-breaking is boundary-only, T1
                                     does not create a persistent arrow
     Cold A3 at a=0.7 is reproduced as a sanity check.

  B. Negative baselines on T0 for ch2, PCI, causal density
     Without these, positive values on future architectures (T3/T4/T5)
     are uninterpretable. All three metrics are run on the base
     architecture (RecurrentCell, K=4 independent cores, no lateral
     coupling, no self-model). Expected: low across the board.
     B1. ch2 (fractal resonance): spectral proxy for IIT Phi.
         ch2 = |P_beta^2 / (P_delta * P_theta)|^(1/3)
         Threshold for Phi ~ 4.3 bits: ch2 >= 0.95.
     B2. PCI (Perturbational Complexity Index): Lempel-Ziv of
         binarized response matrix normalized by entropy, on
         standardized input perturbations. High (~1) = rich response.
     B3. Causal density (cd): mean Transfer Entropy between K cores.
         On independent cores driven by a common input, cd should be
         near zero; this is the baseline against which lateral
         coupling (T3) will be measured.

  C. Toy validation of sigma estimator
     A 3-state cyclic Markov chain with known entropy production rate
     sigma_theory = (alpha - beta) * log(alpha/beta) per step. We
     estimate sigma from simulated trajectories and verify relative
     error < 15%. Gate: if the estimator works here, it is licensed
     to measure sigma on the real architecture under Route 3.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from lfunc_dynamics.proper_time import RecurrentCell


# ==========================================================================
#  A. WARM-A3 AT a=0.7  (stationary vs transient T-breaking)
# ==========================================================================

def asym_episode(cell, T, a, burn_in, rng):
    """Run RecurrentCell with asymmetric update h' = a*h + (1-a)*tanh(...).
    Returns trajectory AFTER burn_in."""
    cell.h = rng.normal(0.0, 1.0, cell.n) * 3.0   # cold init
    traj = []
    for t in range(T + burn_in):
        x = rng.normal(0.0, 1.0, cell.n)
        f = np.tanh(cell.W @ cell.h + cell.V @ x)
        cell.h = a * cell.h + (1.0 - a) * f
        if t >= burn_in:
            traj.append(cell.h.copy())
    return np.asarray(traj)


def a3_moment(d):
    """Third-order reversal-odd moment of step magnitudes."""
    d = np.asarray(d, float)
    if d.size < 4:
        return np.nan
    m = d.size - 1
    val = float(np.mean(d[:m] * d[1:m+1] * (d[:m] - d[1:m+1])))
    mu = float(np.mean(d)) + 1e-12
    return val / (mu ** 3)


def boot_ci(vals, B=300, seed=0):
    rng = np.random.default_rng(seed)
    v = np.asarray(vals, float)
    if v.size == 0:
        return [np.nan, np.nan]
    means = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(B)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def run_part_A():
    """A3 at a=0.0 and a=0.7, cold and warm (burn-in)."""
    T, burn_in, N = 128, 64, 80
    results = {}

    for a in (0.0, 0.7):
        for label, bi in (("cold", 0), ("warm", burn_in)):
            rng = np.random.default_rng(7)
            cell = RecurrentCell(seed=0)
            a3_full, a3_first, a3_second = [], [], []
            for i in range(N):
                h = asym_episode(cell, T, a, bi, rng)
                d = np.linalg.norm(np.diff(h, axis=0), axis=1)
                half = len(d) // 2
                a3_full.append(a3_moment(d))
                a3_first.append(a3_moment(d[:half]))
                a3_second.append(a3_moment(d[half:]))
            results[f"A3_a{a}_{label}_full"] = [
                float(np.nanmean(a3_full)), boot_ci(a3_full)]
            results[f"A3_a{a}_{label}_first"] = [
                float(np.nanmean(a3_first)), boot_ci(a3_first)]
            results[f"A3_a{a}_{label}_second"] = [
                float(np.nanmean(a3_second)), boot_ci(a3_second)]

    # Gate A: stationary T-breaking at a=0.7?
    lo, hi = results["A3_a0.7_warm_full"][1]
    results["stationary_t_breaking"] = bool(lo > 0 or hi < 0)
    results["stationary_t_breaking_ci"] = [lo, hi]
    return results


# ==========================================================================
#  B. NEGATIVE BASELINES ON T0
# ==========================================================================

def collect_trajectories(K=4, T=512, N=60, seed=0):
    """K independent RecurrentCells, each driven by its own noise.
    Returns array of shape (N, T, K, n_hidden)."""
    rng = np.random.default_rng(seed)
    cells = [RecurrentCell(seed=s) for s in range(K)]
    all_traj = np.zeros((N, T, K, cells[0].n), dtype=np.float32)
    for ep in range(N):
        for k, cell in enumerate(cells):
            cell.h = rng.normal(0.0, 0.5, cell.n)
            for t in range(T):
                x = rng.normal(0.0, 1.0, cell.n)
                f = np.tanh(cell.W @ cell.h + cell.V @ x)
                cell.h = cell.h + 0.1 * f    # standard small-step update
                all_traj[ep, t, k] = cell.h.copy()
    return all_traj


# --- B1: ch2 (fractal resonance, proxy for IIT Phi) ---
def compute_ch2(trajs, fs=1.0):
    """ch2 = |P_beta^2 / (P_delta * P_theta)|^(1/3), averaged over neurons
    and episodes. trajs: (N, T, K, n_hidden)."""
    N, T, K, n = trajs.shape
    # PSD along time axis
    psd = np.abs(np.fft.rfft(trajs, axis=1)) ** 2          # (N, T//2+1, K, n)
    freqs = np.fft.rfftfreq(T, d=1.0/fs)
    # band masks
    band_d = (freqs >= 0.5/fs_scale(T)) & (freqs <= 4.0/fs_scale(T))
    band_t = (freqs > 4.0/fs_scale(T))  & (freqs <= 8.0/fs_scale(T))
    band_b = (freqs > 13.0/fs_scale(T)) & (freqs <= 30.0/fs_scale(T))
    if band_b.sum() == 0 or band_d.sum() == 0 or band_t.sum() == 0:
        return float("nan"), {"note": "bands empty at this T"}
    P_d = psd[:, band_d, :, :].mean(axis=1)                 # (N, K, n)
    P_t = psd[:, band_t, :, :].mean(axis=1)
    P_b = psd[:, band_b, :, :].mean(axis=1)
    ch2 = np.abs((P_b ** 2) / (P_d * P_t + 1e-30)) ** (1.0/3.0)
    vals = ch2.flatten()
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)), {"n_samples": int(vals.size),
                                   "ci": boot_ci(vals)}


def fs_scale(T):
    """Normalize frequency bins by number of samples so band definitions
    are scale-invariant. fs=1 in normalized units -> freq max = 0.5."""
    return float(T)


# --- B2: PCI (Perturbational Complexity Index) ---
def compute_pci(cells, N_pert=80, T=64, n_neurons=16, seed=0):
    """PCI = LZ(binarized response) / LZ_max, LZ_max = m / log2(m).
    Response subsampled to n_neurons per core to keep runtime sane.
    T0 expectation: near 1.0 (pure differentiation, independent cores)."""
    rng = np.random.default_rng(seed)
    K = len(cells)
    n = cells[0].n
    pci_vals = []
    for _ in range(N_pert):
        response = np.zeros((K, T, n_neurons), dtype=np.float32)
        kick = rng.normal(0.0, 2.0, n)
        for k, cell in enumerate(cells):
            cell.h = rng.normal(0.0, 0.5, n) + kick
            for t in range(T):
                x = rng.normal(0.0, 1.0, n) if t > 0 else np.zeros(n)
                f = np.tanh(cell.W @ cell.h + cell.V @ x)
                cell.h = cell.h + 0.1 * f
                response[k, t] = cell.h[:n_neurons]
        flat = response.flatten()
        bits = (flat > float(np.median(flat))).astype(np.uint8)
        m = len(bits)
        p1 = float(bits.mean())
        Hb = -(p1 * np.log2(p1 + 1e-12) + (1 - p1) * np.log2(1 - p1 + 1e-12))
        c = lz_complexity(bits)
        pci_vals.append(float(c * np.log2(m) / (m * max(Hb, 1e-6))))
    return float(np.mean(pci_vals)), {"ci": boot_ci(pci_vals)}


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


# --- B3: Causal density via simple binning TE ---
def transfer_entropy_binning(source, target, k_hist=1, bins=8):
    """TE(source -> target) via equal-frequency binning, in bits.
    TE = H(tf|tp) - H(tf|tp,sp)
       = [H(tf,tp) - H(tp)] - [H(tf,tp,sp) - H(tp,sp)]
    All entropies are computed on BINNED (integer) series."""
    T = min(len(source), len(target))
    if T < k_hist + 20:
        return 0.0
    sf = np.asarray(source[k_hist:T-1], float)
    sp = np.asarray(source[k_hist-1:T-2], float)
    tf = np.asarray(target[k_hist:T-1], float)
    tp = np.asarray(target[k_hist-1:T-2], float)

    def bin1d(x):
        ranks = stats.rankdata(x, method="ordinal")
        b = np.floor((ranks - 1) * bins / len(ranks)).astype(np.int64)
        return np.clip(b, 0, bins - 1)

    tf_b, tp_b, sp_b = bin1d(tf), bin1d(tp), bin1d(sp)

    def H(*arrs):
        A = (np.column_stack(arrs) if len(arrs) > 1
             else np.asarray(arrs[0])[:, None]).astype(np.int64)
        ndim = A.shape[1]
        idx = np.ravel_multi_index(A.T, [bins] * ndim)
        counts = np.bincount(idx, minlength=bins ** ndim)
        p = counts[counts > 0] / counts.sum()
        return float(-np.sum(p * np.log2(p)))

    te = (H(tf_b, tp_b) - H(tp_b)) - (H(tf_b, tp_b, sp_b) - H(tp_b, sp_b))
    return float(max(te, 0.0))


def compute_causal_density(trajs):
    """Mean TE over ordered core pairs, computed PER EPISODE and averaged
    (avoids artificial jumps at episode concatenation)."""
    N, T, K, n = trajs.shape
    ts = trajs.mean(axis=-1)                      # (N, T, K)
    te_vals = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            per_ep = [transfer_entropy_binning(ts[e, :, i], ts[e, :, j])
                      for e in range(N)]
            te_vals.append(float(np.mean(per_ep)))
    return float(np.mean(te_vals)), {"n_pairs": K * (K - 1),
                                     "ci_across_pairs": boot_ci(te_vals)}


def run_part_B():
    trajs = collect_trajectories(K=4, T=512, N=60, seed=0)
    ch2_val, ch2_info = compute_ch2(trajs)
    cells = [RecurrentCell(seed=s) for s in range(4)]
    pci_val, pci_info = compute_pci(cells)
    cd_val, cd_info = compute_causal_density(trajs)

    results = {
        "ch2": ch2_val, "ch2_ci": ch2_info.get("ci"),
        "ch2_threshold_human": 0.95,
        "ch2_below_threshold": bool(np.isfinite(ch2_val) and ch2_val < 0.95),
        "pci": pci_val, "pci_ci": pci_info.get("ci"),
        "pci_at_differentiation_ceiling": bool(np.isfinite(pci_val) and pci_val > 0.9),
        "causal_density": cd_val, "cd_ci": cd_info.get("ci"),
        "cd_near_zero": bool(np.isfinite(cd_val) and cd_val < 0.05),
    }
    results["baselines_negative"] = bool(
        results["ch2_below_threshold"]
        and results["pci_at_differentiation_ceiling"]
        and results["cd_near_zero"]
    )
    results["baseline_signature"] = (
        "pure differentiation, no integration (pci~1, ch2~0, cd~0)"
        if results["baselines_negative"] else "unexpected signature"
    )
    return results


# ==========================================================================
#  C. TOY VALIDATION OF SIGMA ESTIMATOR
# ==========================================================================

def toy_sigma_3state_cycle(alpha=0.6, beta=0.3, T=50000, seed=0):
    """3-state cycle 0->1->2->0 with forward rate alpha, backward rate beta.
    Stationary distribution is uniform by symmetry.
    Theory: sigma per step = (alpha - beta) * log(alpha/beta).
    We estimate sigma from data via plug-in transition matrix and pi."""
    rng = np.random.default_rng(seed)
    states = np.zeros(T+1, dtype=int)
    for t in range(T):
        s = states[t]
        r = rng.random()
        if r < alpha:
            states[t+1] = (s + 1) % 3
        elif r < alpha + beta:
            states[t+1] = (s - 1) % 3
        else:
            states[t+1] = s

    # Empirical transition matrix
    P = np.zeros((3, 3))
    for t in range(T):
        P[states[t], states[t+1]] += 1
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / (row_sums + 1e-30)

    # Empirical stationary distribution
    pi = np.array([np.sum(states == s) for s in range(3)], float) / T

    # Plug-in sigma
    sigma = 0.0
    for i in range(3):
        for j in range(3):
            if P[i, j] > 0 and P[j, i] > 0 and pi[i] > 0 and pi[j] > 0:
                sigma += pi[i] * P[i, j] * np.log(
                    (P[i, j] * pi[i]) / (P[j, i] * pi[j]))

    sigma_theory = (alpha - beta) * np.log(alpha / beta)
    rel_err = abs(sigma - sigma_theory) / (abs(sigma_theory) + 1e-12)

    return {
        "alpha": alpha, "beta": beta,
        "sigma_theory": float(sigma_theory),
        "sigma_eval": float(sigma),
        "relative_error": float(rel_err),
        "validation_passed": bool(rel_err < 0.15),
    }


def run_part_C():
    results = {}
    # Three operating points: strong asymmetry, moderate, near-zero
    for (a, b, tag) in [(0.6, 0.3, "strong"),
                        (0.45, 0.35, "moderate"),
                        (0.4, 0.4, "zero")]:
        results[tag] = toy_sigma_3state_cycle(alpha=a, beta=b)
    # Global gate: pass on strong AND zero (correct zero) AND moderate
    results["estimator_valid"] = (
        results["strong"]["validation_passed"]
        and results["moderate"]["validation_passed"]
        and results["zero"]["validation_passed"]
    )
    return results


# ==========================================================================
#  MAIN
# ==========================================================================

def main():
    verdict = {
        "stage": "selfhood_step0_baselines",
        "A_warm_a3": run_part_A(),
        "B_baselines": run_part_B(),
        "C_toy_sigma": run_part_C(),
    }
    verdict["overall_status"] = {
        "A_stationary_t_breaking_resolved": verdict["A_warm_a3"]["stationary_t_breaking"],
        "B_baselines_established": verdict["B_baselines"]["baselines_negative"],
        "C_sigma_estimator_validated": verdict["C_toy_sigma"]["estimator_valid"],
        "step0_complete": (
            verdict["B_baselines"]["baselines_negative"]
            and verdict["C_toy_sigma"]["estimator_valid"]
        ),
        "next_step_decision": (
            "T1 viable (stationary T-breaking confirmed)"
            if verdict["A_warm_a3"]["stationary_t_breaking"] else
            "Skip T1; go to T2 (temporal self-model) or T4 (closed loop)"
        ),
    }

    out = Path("proper_time_verdicts_selfhood_step0.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(json.dumps(verdict, indent=2))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()