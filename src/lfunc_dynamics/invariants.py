import numpy as np
from .flow import forces, coulomb_energy, integrate_dysion

def compute_tau_H(w, s_max=2.0):
    """Энергетическое полувремя релаксации на стандартном горизонте."""
    t, y = integrate_dysion(w, s_max=s_max)
    H = np.array([coulomb_energy(y[:, i]) for i in range(len(t))])
    
    H0, H_final = H[0], H[-1]
    H_half = (H0 + H_final) / 2.0
    
    if H0 > H_half > H_final:
        return float(np.interp(H_half, H[::-1], t[::-1]))
    return np.nan

def compute_conv(w):
    """Выпуклость диссипации (Dissipation Convexity)."""
    P0 = 0.5 * np.sum(forces(w) ** 2)
    t, y = integrate_dysion(w, s_max=2.0) # Используем стандартный горизонт
    H0 = coulomb_energy(y[:, 0])
    H_final = coulomb_energy(y[:, -1])
    drop = H0 - H_final
    if drop > 1e-9:
        return 2 * P0 / drop
    return np.nan

def compute_r2(w):
    """Нормализованный спектральный зазор лапласиана."""
    N = len(w)
    d = w[:, None] - w[None, :]
    np.fill_diagonal(d, np.inf)
    W = 2.0 / d ** 2
    L = np.diag(W.sum(1)) - W
    lam = np.linalg.eigvalsh(L)
    return float(lam[1] * N / (4 * np.pi ** 2))