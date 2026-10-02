r"""
ml_cabinet.py — CoreCabinet: tick-latched temporal ensemble inference.
Теория см. docstring-блок в начале модуля; разбиение b = t*K + k детерминировано.
"""
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

_G_MODELS = None

def _init_worker(models):
    global _G_MODELS
    _G_MODELS = models

def _cell(args):
    b, X, desync_s = args
    if desync_s > 0.0:
        time.sleep(desync_s)
    t0 = time.perf_counter()
    y = float(_G_MODELS[b].predict(X)[0])
    t1 = time.perf_counter()
    return b, y, t0, t1

QUADRANT_ACTION = {
    "reliable": "standard protocol; no verification needed",
    "overconfident": "downgrade confidence; short ODE check (s_max=0.5)",
    "honest_low": "ODE fallback; append measurement to active-learning cache",
    "calibration_artifact": "flag bag for refit; use ODE for this shot",
}

@dataclass
class CabinetPrediction:
    value: float
    ci: tuple
    sigma_ep: float
    sigma_tick: float
    tick_pattern: list
    quadrant: str
    action: str
    stale_ticks: list
    latch_lags_ms: list
    K: int
    T: int
    wall_ms: float

def make_standardizer(surrogate):
    def fn(feats, ensemble):
        X = surrogate._matrix([feats], [ensemble])
        return (X - surrogate._mu) / surrogate._sd
    return fn

class CoreCabinet:
    def __init__(self, surrogate, K=4, T=4, desync_ms=0.0, coh_budget_ms=25.0):
        self.models = list(surrogate.models)
        if len(self.models) != K * T:
            raise ValueError(f"bag size {len(self.models)} != K*T={K*T}")
        self.s_res = surrogate.sigma_res
        self.K, self.T = K, T
        self.desync = desync_ms / 1000.0
        self.coh = coh_budget_ms / 1000.0
        self.theta_tick = None
        self._pool = None

    def _pool_get(self):
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self.K,
                initializer=_init_worker, initargs=(self.models,))
        return self._pool

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    def predict_cells(self, X):
        """T последовательных волн по K параллельных ячеек; возврат (P, T0, T1)."""
        pool = self._pool_get()
        P = np.zeros((self.T, self.K))
        T0 = np.zeros((self.T, self.K))
        T1 = np.zeros((self.T, self.K))
        for t in range(self.T):
            futs = {}
            for k in range(self.K):
                b = t * self.K + k
                futs[k] = pool.submit(_cell, (b, X, k * self.desync))
            for k, f in futs.items():
                b, y, a, c = f.result()
                P[t, k] = y; T0[t, k] = a; T1[t, k] = c
        return P, T0, T1

    def aggregate(self, P, T1):
        stale, mus, ds, lags = [], [], [], []
        for t in range(self.T):
            lag = float(T1[t].max() - T1[t].min())
            lags.append(lag)
            if lag > self.coh:
                stale.append(t)
            mus.append(float(np.median(P[t])))
            ds.append(float(np.std(P[t])))
        keep = [t for t in range(self.T) if t not in stale] or list(range(self.T))
        value = float(np.median([mus[t] for t in keep]))
        s_ep = float(np.std(P))
        s_tick = float(np.mean([ds[t] for t in keep]))
        half = 1.96 * float(np.hypot(s_ep, self.s_res))
        return value, (value - half, value + half), s_ep, s_tick, ds, stale, lags

    def quadrant(self, value, half, s_tick):
        rel_w = half / max(abs(value), 0.5)
        th = self.theta_tick if self.theta_tick is not None else np.inf
        if rel_w < 0.5 and s_tick < th:
            return "reliable"
        if rel_w < 0.5 and s_tick >= th:
            return "overconfident"
        if rel_w >= 0.5 and s_tick >= th:
            return "honest_low"
        return "calibration_artifact"

    def predict(self, feats, ensemble, std_fn):
        t_start = time.perf_counter()
        X = std_fn(feats, ensemble)
        P, T0, T1 = self.predict_cells(X)
        value, ci, s_ep, s_tick, pattern, stale, lags = self.aggregate(P, T1)
        half = (ci[1] - ci[0]) / 2.0
        q = self.quadrant(value, half, s_tick)
        wall = (time.perf_counter() - t_start) * 1000.0
        return CabinetPrediction(
            value=value, ci=ci, sigma_ep=s_ep, sigma_tick=s_tick,
            tick_pattern=pattern, quadrant=q, action=QUADRANT_ACTION[q],
            stale_ticks=stale, latch_lags_ms=[1000 * l for l in lags],
            K=self.K, T=self.T, wall_ms=wall)

    def calibrate(self, feats_list, ensembles, std_fn):
        s_ticks = []
        for f, e in zip(feats_list, ensembles):
            X = std_fn(f, e)
            P, _, T1 = self.predict_cells(X)
            s_ticks.append(float(P.std(axis=1).mean()))
        self.theta_tick = float(np.quantile(s_ticks, 0.95))
        return self.theta_tick