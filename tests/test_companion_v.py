import numpy as np
# Импортируем из нашего пакета
from lfunc_dynamics.invariants import compute_r2, compute_tau_H 

def test_crystal_r2():
    """Проверка r2 для идеального кристалла (должно быть близко к 1.0 после нормировки, 
       но для сырых данных зависит от N. Здесь просто проверяем, что функция работает)."""
    N = 50
    w = np.arange(N, dtype=float) # Идеальный кристалл
    r2 = compute_r2(w)
    assert np.isfinite(r2)
    # Для единичной решетки r2 должен быть положительным и конечным
    assert r2 > 0 

def test_tau_H_finite():
    """Проверка, что tau_H вычисляется для возмущенного кристалла."""
    N = 30
    w = np.arange(N, dtype=float) + np.random.normal(0, 0.05, N)
    w -= w.mean() # Центрирование
    tau = compute_tau_H(w, s_max=1.0) # Берем малый горизонт для скорости теста
    assert np.isfinite(tau)
    assert tau > 0