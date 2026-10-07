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
    """Курамото-связанные осцилляторы с общим приводом работы и опциональным
    дрейфом собственных частот (контрольный режим для специфичности разрыва).
    coherence() — параметр порядка Курaмото r = |mean exp(i*theta)|."""
    def __init__(self, n=8, seed=0, coupling=0.15):
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(0.3, 1.2, n)
        self.kap = 0.3 * self.w
        self.K = coupling
        self.gamma = 0.0          # дрейф частот (контроль P1), 0 = без дрейфа
        self.t = 0                # внутренний счётчик шагов банка
        self.theta = rng.uniform(0, 2 * np.pi, n)
        self.history = []                 # траектория параметра порядка r[t]

    def update(self, u):
        w_eff = self.w * (1.0 + self.gamma * self.t / 64.0)
        mf = np.mean(np.exp(1j * self.theta))
        coup = self.K * np.imag(mf * np.exp(-1j * self.theta))
        self.theta = (self.theta + w_eff + self.kap * u + coup) % (2 * np.pi)
        self.t += 1
        self.history.append(self.coherence())

    def coherence(self):
        return float(np.abs(np.mean(np.exp(1j * self.theta))))


class ProperTimeProbe:
    def __init__(self, eps=0.05, gap_frac=0.25, delta0=1e-3, use_shadow=True):
        self.eps, self.gap_frac, self.delta0 = eps, gap_frac, delta0
        self.use_shadow = use_shadow
        self.overwrites = 0          # бесплатная стрела: перезаписи буфера
        self.reset()

    def reset(self):
        self.bank = RhythmBank()
        self.lam = 0.0; self.n_ev = 0; self.erased = 0.0
        self.prev_h = None; self.prev_dsh = None
        self.d_series = []
        self.sat = 0; self.steps = 0

    def note_overwrite(self, k=1):
        self.overwrites += k

    def update(self, h, h_shadow=None):
        if self.prev_h is None:
            self.prev_h = h.copy()
            if h_shadow is not None:
                self.prev_dsh = float(np.linalg.norm(h - h_shadow))
            return self.snapshot()
        d = float(np.linalg.norm(h - self.prev_h)); self.prev_h = h.copy()
        self.sat += int(np.sum(np.abs(h) > 0.98))   # бесплатное стирание (насыщение)
        self.steps += 1
        self.lam += d; self.d_series.append(d)
        if d > self.eps: self.n_ev += 1
        if self.use_shadow and h_shadow is not None:
            dsh = float(np.linalg.norm(h - h_shadow))
            if self.prev_dsh and self.prev_dsh > 1e-12 and dsh > 1e-12:
                self.erased += max(0.0, -np.log2(dsh / self.prev_dsh))
            self.prev_dsh = dsh
        self.bank.update(d)
        return self.snapshot()

    def snapshot(self):
        return dict(lam=self.lam, n_ev=self.n_ev, erased=self.erased,
                    coh=self.bank.coherence(), overwrites=self.overwrites,
                    sat=self.sat, steps=self.steps)

    def set_drift(self, gamma: float):
        self.bank.gamma = gamma

    def gap_steps_post(self):
        """Разрыв = шаги с активностью ниже четверти медианы эпизода."""
        d = np.asarray(self.d_series)
        if d.size < 8: return 0
        return int(np.sum(d < self.gap_frac * np.median(d)))

    def transient_centroid(self, top_frac=0.1):
        """Стрела: центроид самых крупных шагов. Транзиент в начале -> <0.5;
        для стационарного процесса (без стрелы) -> 0.5 в обе стороны."""
        d = np.asarray(self.d_series)
        if d.size < 8: return 0.5
        k = max(2, int(top_frac * d.size))
        idx = np.argsort(d)[-k:]
        return float(np.mean(idx) / (d.size - 1))