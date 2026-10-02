import numpy as np

def unfold(g, Q, d):
    """Unfolding для L-функций степени d."""
    g = np.asarray(g, float)
    if d == 1:
        return (g / (2 * np.pi)) * (np.log(g * Q / (2 * np.pi)) - 1.0)
    return ((d * g / (2 * np.pi)) * (np.log(g / (2 * np.pi)) - 1.0) 
            + (g / (2 * np.pi)) * np.log(Q))

def spec_h0(eps, kappa_star=np.pi/2):
    """Вычисление w_low и спектра."""
    S = np.abs(np.fft.rfft(eps)) ** 2
    kap = 2 * np.pi * np.fft.rfftfreq(len(eps))
    w_low = float(S[kap <= kappa_star].sum() / S.sum()) if S.sum() > 0 else np.nan
    return S, kap, w_low