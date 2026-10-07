r"""
spin_noise.py (v2.2) — добавлен метод trace_mixed() для суперпозиции всех трёх шумов.
"""
import numpy as np


class SpinNoiseRig:
    def __init__(self, fs=1.0e9, n=2048, tau_mag=50.0e-9, tau_f=0.4e-9,
                 sigma_th=1.0, shot_rate=0.35, shot_amp=0.9, fano=1.0,
                 zp_amp=0.35, pair_rate=3.0, pair_amp=0.35,
                 chop_period=128, seed=0):
        self.fs, self.n = float(fs), int(n)
        self.dt = 1.0 / self.fs
        self.t = np.arange(self.n) * self.dt
        self.k = np.fft.rfftfreq(self.n, self.dt)
        self.phi = np.exp(-self.dt / tau_mag)
        self.sigma_th = sigma_th
        self.sigma_ou = sigma_th * np.sqrt(1.0 - self.phi ** 2)
        self.phi_f = np.exp(-self.dt / tau_f)
        self.lam = shot_rate
        self.gain_f = 1.0 / (1.0 - self.phi_f)
        self.shot_rate, self.shot_amp, self.fano = shot_rate, shot_amp, fano
        self.zp_amp, self.pair_rate, self.pair_amp = zp_amp, pair_rate, pair_amp
        half = int(chop_period) // 2
        self.chop = np.where((np.arange(self.n) // half) % 2 == 0, 1.0, -1.0)
        self.carriers = 2 * np.pi * np.array([60e6, 100e6, 140e6, 180e6])
        self.delays = np.array([8e-9, 16e-9, 24e-9])
        self.env_w = 3e-9

    # ---------- генераторы ----------
    def thermal(self, rng):
        x = np.empty(self.n)
        x[0] = self.sigma_th * rng.standard_normal()
        for i in range(1, self.n):
            x[i] = self.phi * x[i - 1] + self.sigma_ou * rng.standard_normal()
        return x

    def shot(self, rng):
        counts = rng.poisson(self.lam, self.n).astype(float)
        if self.fano != 1.0:
            counts = self.lam + (counts - self.lam) * np.sqrt(self.fano)
        q = np.zeros(self.n)
        for i in range(1, self.n):
            q[i] = self.phi_f * q[i - 1] + counts[i - 1]
        q -= self.lam * self.gain_f
        return self.shot_amp * q / np.sqrt(self.lam * self.gain_f + 1e-12)

    def quantum(self, rng):
        x = self.zp_amp * rng.standard_normal(self.n)
        for _ in range(int(rng.poisson(self.pair_rate))):
            w = float(rng.choice(self.carriers))
            dl = float(rng.choice(self.delays))
            t1 = rng.uniform(20e-9, self.t[-1] - dl - 20e-9)
            t2 = t1 + dl
            th = rng.uniform(0, 2 * np.pi)
            g1 = np.exp(-(self.t - t1) ** 2 / (2 * self.env_w ** 2))
            g2 = np.exp(-(self.t - t2) ** 2 / (2 * self.env_w ** 2))
            x += self.pair_amp * (g1 * np.cos(w * (self.t - t1) + th)
                                  + g2 * np.cos(w * (self.t - t2) + th))
        return x

    def trace(self, rng, species="thermal", dc=5.0, chopped=False):
        noise = {"thermal": self.thermal, "shot": self.shot,
                 "quantum": self.quantum}[species](rng)
        return dc * (self.chop if chopped else 1.0) + noise, noise

    def components(self, rng):
        return self.thermal(rng), self.shot(rng), self.quantum(rng)

    def trace_mixed(self, rng, dc=5.0, fracs=None, weights=None, chopped=False):
        """Суперпозиция, покрывающая ВЕСЬ симплекс смесей.
        fracs — целевые доли дисперсии (Dirichlet(1,1,1), если None);
        амплитуды обратно решаются из единичных дисперсий компонент:
        w_i = sqrt(f_i / var_i). Квантовый буст (до ~2.9x) возникает сам
        из малой единичной дисперсии пар, а не задаётся вручную.
        weights=... — legacy-путь v2.3, оставлен для совместимости."""
        n_th, n_sh, n_qu = self.components(rng)
        v_unit = np.array([n_th.var(), n_sh.var(), n_qu.var()]) + 1e-12
        if weights is not None:
            w = np.asarray(weights, float)
        else:
            if fracs is None:
                fracs = rng.dirichlet(np.ones(3))
            fracs = np.asarray(fracs, float)
            fracs = fracs / fracs.sum()
            w = np.sqrt(fracs / v_unit)
        noise = w[0] * n_th + w[1] * n_sh + w[2] * n_qu
        v = w ** 2 * v_unit
        fracs_real = v / v.sum()
        return dc * (self.chop if chopped else 1.0) + noise, dict(fracs=fracs_real, weights=w)

    # ---------- признаки ----------
    def _env(self, x):
        full = np.fft.fft(x)
        full[1:self.n // 2] *= 2.0
        full[self.n // 2 + 1:] = 0.0
        return np.abs(np.fft.ifft(full))

    def features(self, x):
        d = np.diff(x)
        sd = d.std() + 1e-12
        skew = float(((d - d.mean()) ** 3).mean() / sd ** 3)
        kurt = float(((d - d.mean()) ** 4).mean() / sd ** 4) - 3.0
        ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        sp = np.abs(np.fft.rfft(x - x.mean())) ** 2
        hf = float(sp[len(sp) // 2:].mean() / (sp.mean() + 1e-12))
        fano = float(d.var() / (np.abs(d).mean() ** 2 + 1e-12))
        xc = x - x.mean()
        pair_ms = []
        for w in self.carriers:
            mask = np.exp(-(self.k - w / (2 * np.pi)) ** 2 / (2 * 20e6 ** 2))
            band = np.fft.irfft(np.fft.rfft(xc) * mask, self.n)
            eb = self._env(band)
            eb = eb - eb.mean()
            denom = np.sqrt(np.mean(eb ** 2)) + 1e-12
            vals = []
            for dl in self.delays:
                kk = int(dl * self.fs)
                vals.append(float(np.mean(eb[:-kk] * eb[kk:])) / denom ** 2)
            pair_ms.append(max(vals))
        pair_m = float(max(pair_ms))
        pair_dev = float(np.std(pair_ms) / (abs(np.mean(pair_ms)) + 1e-12))
        return dict(var=float(x.var()), skew=skew, kurt=kurt, ac1=ac1,
                    hf=hf, fano=fano, pair_m=pair_m, pair_dev=pair_dev)

    # ---------- readout ----------
    def readout(self, x, clip=False):
        y = x * self.chop
        if clip:
            med = np.median(y)
            r = y - med
            s = 1.4826 * np.median(np.abs(r)) + 1e-12
            y = med + np.clip(r, -3.0 * s, 3.0 * s)
        return float(y.mean())