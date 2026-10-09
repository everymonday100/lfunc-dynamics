#!/usr/bin/env python3
r"""selfhood_step0.5_delayed_coupling.py — Step 0.5: delayed coupling.

Delayed coupling creates genuine transfer entropy without homogenization:
cores influence each other with a delay τ=5-10, preserving individuality.

Three architectures:
  T0: independent cores (baseline)
  T0.5_delayed_c0.3_tau5:  coupling=0.3, delay=5
  T0.5_delayed_c0.3_tau10: coupling=0.3, delay=10
  T0.5_delayed_c0.5_tau5:  coupling=0.5, delay=5

Metrics use pci_reference.py (canonical).
Relative comparison: report DELTA from T0 baseline, not absolute thresholds.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

from pci_reference import pci_batch
from lfunc_dynamics.proper_time import RecurrentCell


class DelayedCoupledCore:
    """Core with delayed lateral coupling: lateral(t) = mean(h_j(t-τ))."""
    def __init__(self, seed, n=64, coupling=0.3, delay=5):
        self.cell = RecurrentCell(seed=seed)
        self.n = self.cell.n
        self.coupling = coupling
        self.delay = delay
        self.history = []  # ring buffer of past h
    
    def step(self, x, other_histories=None):
        f = np.tanh(self.cell.W @ self.cell.h + self.cell.V @ x)
        lateral = np.zeros(self.n)
        if other_histories is not None and len(other_histories) > 0:
            # other_histories: list of histories from other cores
            # take the state delay steps ago from each
            past_states = []
            for hist in other_histories:
                if len(hist) >= self.delay:
                    past_states.append(hist[-self.delay])
            if len(past_states) > 0:
                lateral = np.mean(past_states, axis=0)
                norm = np.linalg.norm(lateral)
                if norm > 1e-8:
                    lateral = lateral / norm
        self.cell.h = self.cell.h + 0.1 * f + self.coupling * lateral
        self.history.append(self.cell.h.copy())
        if len(self.history) > self.delay + 10:
            self.history = self.history[-(self.delay + 10):]
        return self.cell.h.copy()
    
    def reset(self, rng):
        self.cell.h = rng.normal(0, 0.5, self.n)
        self.history = []


def compute_ch2(trajs):
    """ch2 spectral proxy (same as before)."""
    N, T, K, n = trajs.shape
    if not np.all(np.isfinite(trajs)):
        return float("nan"), {}
    psd = np.abs(np.fft.rfft(trajs, axis=1)) ** 2
    freqs = np.fft.rfftfreq(T)
    def fs_scale(T): return float(T)
    band_d = (freqs >= 0.5/fs_scale(T)) & (freqs <= 4.0/fs_scale(T))
    band_t = (freqs > 4.0/fs_scale(T)) & (freqs <= 8.0/fs_scale(T))
    band_b = (freqs > 13.0/fs_scale(T)) & (freqs <= 30.0/fs_scale(T))
    if not (band_b.any() and band_d.any() and band_t.any()):
        return float("nan"), {}
    P_d = psd[:, band_d].mean(axis=1)
    P_t = psd[:, band_t].mean(axis=1)
    P_b = psd[:, band_b].mean(axis=1)
    ch2 = np.abs((P_b**2) / (P_d * P_t + 1e-30)) ** (1.0/3)
    vals = ch2[np.isfinite(ch2)]
    if len(vals) == 0:
        return float("nan"), {}
    return float(vals.mean()), {}


def transfer_entropy_binning(source, target, k_hist=1, bins=8):
    """TE(source→target) via equal-frequency binning (same as before)."""
    T = min(len(source), len(target))
    if T < k_hist + 20:
        return 0.0
    if not (np.all(np.isfinite(source)) and np.all(np.isfinite(target))):
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
    """Mean TE over ordered core pairs."""
    N, T, K, n = trajs.shape
    if not np.all(np.isfinite(trajs)):
        return float("nan")
    ts = trajs.mean(axis=-1)
    te_vals = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            per_ep = [transfer_entropy_binning(ts[e, :, i], ts[e, :, j])
                      for e in range(N)]
            te_vals.append(float(np.mean(per_ep)))
    return float(np.mean(te_vals)) if te_vals else float("nan")


def run_architecture(core_factory, K=4, T=512, N=40, seed=0):
    """Run K cores for N episodes, return (N, T, K, n)."""
    rng = np.random.default_rng(seed)
    cores = [core_factory(s) for s in range(K)]
    n = cores[0].n
    trajs = np.zeros((N, T, K, n), dtype=np.float64)
    for ep in range(N):
        for c in cores:
            c.reset(rng)
        for t in range(T):
            states = []
            for k, c in enumerate(cores):
                x = rng.normal(0, 1, n)
                other_hists = [cores[j].history for j in range(K) if j != k]
                h_new = c.step(x, other_histories=other_hists)
                states.append(h_new)
            for k in range(K):
                trajs[ep, t, k] = states[k]
    return trajs


def measure(trajs, tag):
    ch2_val, _ = compute_ch2(trajs)
    pci_val, _ = pci_batch(trajs)
    cd_val = compute_causal_density(trajs)
    return {"tag": tag, "ch2": ch2_val, "pci": pci_val, "causal_density": cd_val}


def main():
    print("Running T0 (independent)...")
    t0 = run_architecture(
        lambda s: DelayedCoupledCore(seed=s, coupling=0.0, delay=5),
        K=4, T=512, N=40)
    r_t0 = measure(t0, "T0_independent")
    
    results = {"T0_independent": r_t0}
    
    for (c, tau) in [(0.3, 5), (0.3, 10), (0.5, 5)]:
        tag = f"delayed_c{c}_tau{tau}"
        print(f"Running {tag}...")
        trajs = run_architecture(
            lambda s: DelayedCoupledCore(seed=s, coupling=c, delay=tau),
            K=4, T=512, N=40)
        r = measure(trajs, tag)
        results[tag] = r
    
    # RELATIVE comparison: delta from T0
    def delta(r):
        return {
            "d_ch2": r["ch2"] - r_t0["ch2"],
            "d_pci": r["pci"] - r_t0["pci"],
            "d_cd": r["causal_density"] - r_t0["causal_density"],
        }
    
    for tag, r in results.items():
        if tag != "T0_independent":
            r["delta_from_T0"] = delta(r)
            # Integration criterion: cd INCREASES without pci DROP
            r["integration_by_relative"] = bool(
                r["delta_from_T0"]["d_cd"] > 0 and
                r["delta_from_T0"]["d_pci"] > -0.05
            )
    
    verdict = {
        "stage": "selfhood_step0.5_delayed_coupling",
        "results": results,
        "note": (
            "Relative comparison, not absolute thresholds. "
            "Integration = cd increases without pci drop. "
            "PCI computed by pci_reference.py (canonical). "
            "ch2 is a spectral proxy; threshold 0.5 from human TMS-EEG "
            "is NOT applicable to 4-core systems."
        ),
    }
    
    out = Path("proper_time_verdicts_delayed_coupling.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=2)
    
    print("\n" + "=" * 70)
    print(f"{'Arch':<25} {'ch2':>8} {'pci':>8} {'cd':>8} {'d_cd':>8} {'integ?':>8}")
    print("-" * 70)
    for tag, r in results.items():
        ch2_s = f"{r['ch2']:8.4f}" if np.isfinite(r['ch2']) else "     nan"
        pci_s = f"{r['pci']:8.4f}" if np.isfinite(r['pci']) else "     nan"
        cd_s = f"{r['causal_density']:8.4f}" if np.isfinite(r['causal_density']) else "     nan"
        if tag == "T0_independent":
            d_cd_s = "  base"
            integ_s = "  -"
        else:
            d_cd_s = f"{r['delta_from_T0']['d_cd']:+8.4f}"
            integ_s = "  YES" if r["integration_by_relative"] else "  NO"
        print(f"{r['tag']:<25} {ch2_s} {pci_s} {cd_s} {d_cd_s} {integ_s}")
    print("=" * 70)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()