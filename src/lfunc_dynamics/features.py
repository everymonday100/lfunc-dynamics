"""Извлечение признаков окна: спектральные, локальные, фазовые."""
import numpy as np
from .flow import forces
from .invariants import compute_r2

def window_features(w):
    w = np.asarray(w, float)
    g = np.diff(w)
    gbar = g.mean()
    eps = g / gbar - 1.0
    F = np.fft.rfft(eps)
    kap = 2 * np.pi * np.fft.rfftfreq(len(eps))
    S = np.abs(F) ** 2
    w_low = float(S[kap <= np.pi / 2].sum() / S.sum()) if S.sum() > 0 else np.nan
    ph = np.angle(F)
    M = len(F)
    j = np.arange(1, M // 2)
    jj = 2 * j
    m = jj < M
    j, jj = j[m], jj[m]
    denom = (np.abs(F[j]) ** 2 * np.abs(F[jj])) + 1e-12
    bic = float(np.mean(np.abs(F[j] ** 2 * np.conj(F[jj])) / denom))
    qpc = float(np.mean(np.cos(2 * ph[j] - ph[jj])))
    sd = eps.std()
    skew = float(np.mean(eps ** 3) / sd ** 3) if sd > 0 else 0.0
    kurt = float(np.mean(eps ** 4) / sd ** 4 - 3.0) if sd > 0 else 0.0
    P0 = 0.5 * float(np.sum(forces(w) ** 2))
    return dict(
        h0=1.0 - w_low,
        r2=compute_r2(w),
        log_mingap=float(np.log(max(g.min(), 1e-6))),
        log_P0=float(np.log(max(P0, 1e-12))),
        skew=skew, kurt=kurt, bic=bic, qpc=qpc,
    )