r"""proper_time.py — AFTER-6 prototype: internal time of a computational process.
Duration = odometer/event-clock/rhythm-phase pattern (label-free);
arrow = contraction-erasure counter; rhythm = oscillator bank driven by work.
Coordinate time is never a feature (debug only)."""
import numpy as np


class RecurrentCell:
    def __init__(self, n=32, seed=0):
        rng = np.random.default_rng(seed)
        self.W = 0.9 * rng.standard_normal((n, n)) / np.sqrt(n)
        self.V = rng.standard_normal((n, n)) / np.sqrt(n)
        self.n = n
        self.h = np.zeros(n)

    def step(self, x):
        self.h = np.tanh(self.W @ self.h + self.V @ x)
        return self.h

    def perturbed_copy(self, delta=1e-3, seed=0):
        c = RecurrentCell.__new__(RecurrentCell)
        c.W, c.V, c.n = self.W, self.V, self.n
        rng = np.random.default_rng(seed)
        d = rng.standard_normal(self.n); d /= np.linalg.norm(d)
        c.h = self.h + delta * d
        return c


class RhythmBank:
    def __init__(self, n=8, seed=0):
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(0.3, 1.2, n)
        self.kap = 0.3 * self.w          # частотно-зависимая связь: когерентность читает паттерн
        self.theta = rng.uniform(0, 2 * np.pi, n)
        self.t = 0

    def update(self, u):
        self.theta = (self.theta + self.w + self.kap * u) % (2 * np.pi)
        self.t += 1

    def coherence(self):
        phi = (self.theta - self.w * self.t) % (2 * np.pi)
        return float(np.abs(np.mean(np.exp(1j * phi))))


class ProperTimeProbe:
    def __init__(self, eps=0.05, gap_tol=0.02, delta0=1e-3):
        self.eps, self.gap_tol, self.delta0 = eps, gap_tol, delta0
        self.reset()   # AFTER-6 fix: инициализируем состояние при создании

    def reset(self):
        self.bank = RhythmBank()
        self.lam = 0.0; self.n_ev = 0; self.erased = 0.0; self.gaps = 0
        self.prev_h = None; self.prev_dsh = None; self.d_series = []

    def update(self, h, h_shadow):
        if self.prev_h is None:
            self.prev_h = h.copy()
            self.prev_dsh = float(np.linalg.norm(h - h_shadow))
            return self.snapshot()
        d = float(np.linalg.norm(h - self.prev_h)); self.prev_h = h.copy()
        self.lam += d; self.d_series.append(d)
        if d > self.eps: self.n_ev += 1
        if d < self.gap_tol: self.gaps += 1
        dsh = float(np.linalg.norm(h - h_shadow))
        if self.prev_dsh > 1e-12 and dsh > 1e-12:
            self.erased += max(0.0, -np.log2(dsh / self.prev_dsh))
        self.prev_dsh = dsh
        self.bank.update(d)
        return self.snapshot()

    def snapshot(self):
        return dict(lam=self.lam, n_ev=self.n_ev, erased=self.erased,
                    coh=self.bank.coherence(), gaps=self.gaps)

    def arrow_slope(self):
        d = np.asarray(self.d_series)
        if d.size < 4: return 0.0
        return float(np.polyfit(np.arange(d.size), np.log(d + 1e-9), 1)[0])