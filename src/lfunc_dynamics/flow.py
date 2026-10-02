import numpy as np
from scipy.integrate import solve_ivp

def forces(y):
    """Сила кулоновского отталкивания: 2 * Σ 1/(x_k - x_j)."""
    d = y[:, None] - y[None, :]
    np.fill_diagonal(d, np.inf)
    return (2.0 / d).sum(1)

def coulomb_energy(y):
    """Энергия Кулона: H = -Σ_{i<j} log|x_i - x_j|."""
    i, j = np.triu_indices(len(y), k=1)
    return -np.sum(np.log(np.abs(y[i] - y[j])))

def integrate_dysion(w, s_max=2.0, n_steps=100):
    """
    Интегрирует поток Дайсона dx/ds = forces(x).
    Возвращает массив времен и массив конфигураций.
    """
    t_eval = np.linspace(0.0, s_max, n_steps)
    sol = solve_ivp(
        lambda t, y: forces(y), 
        [0.0, s_max], 
        w,
        t_eval=t_eval, 
        method="RK45", 
        rtol=1e-8, 
        atol=1e-10
    )
    return sol.t, sol.y