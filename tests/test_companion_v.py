import numpy as np
from lfunc_dynamics.invariants import compute_r2, compute_tau_H, compute_conv

def test_crystal_r2():
    """Проверка r2 для идеального кристалла."""
    N = 50
    w = np.arange(N, dtype=float)
    r2 = compute_r2(w)
    assert np.isfinite(r2)
    assert r2 > 0 

def test_tau_H_finite():
    """Проверка, что tau_H вычисляется для возмущенного кристалла."""
    N = 30
    rng = np.random.default_rng(42)
    w = np.arange(N, dtype=float) + rng.normal(0, 0.05, N)
    w -= w.mean()
    tau = compute_tau_H(w, s_max=1.0)
    assert np.isfinite(tau)
    assert tau > 0

def test_conv_finite():
    """Проверка вычисления conv."""
    N = 30
    rng = np.random.default_rng(42)
    w = np.arange(N, dtype=float) + rng.normal(0, 0.05, N)
    w -= w.mean()
    conv = compute_conv(w, s_max=1.0)
    assert np.isfinite(conv)
    assert conv > 0
