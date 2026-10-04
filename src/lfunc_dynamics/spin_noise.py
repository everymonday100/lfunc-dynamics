r"""
spin_noise.py — синтетические трассы считывания спинтроники (MgO-MTJ)
с тремя видами шума и физическими дискриминаторами для ватчдога CoreCabinet.

Виды:
  thermal  — Орнштейн-Уленбек (магнонный) + найквистовский пол; гауссов,
             коррелированный, лоренцевский спектр, классический detailed balance.
  shot     — составной пуассоновский процесс туннельных событий (rate = I/e,
             фактор Фано), импульсы сформированы однополюсным фильтром tau_f;
             дискретный, скошенный, T-независимый.
  quantum  — T-независимый пол нулевых колебаний ПЛЮС двухмодовые сжатые пары
             магнонов с ОБЩЕЙ случайной фазой (фазовая защёлка). Общая фаза —
             неклассическая подпись: невидима для степенных статистик, видима
             фазоинвариантным joint matched filter по огибающей.

Дискриминаторы (признаки):
  ac1      лаго-1 автокорреляция            -> thermal
  skew/kurt моменты приращений             -> shot
  fano     var/mean^2 приращений            -> shot (дискретность)
  hf       доля мощности верхней полосы     -> quantum (нулевой пол)
  pair_m   фазоинвариантный matched filter  -> quantum (пары)
  pair_dev разброс pair_m по несущим        -> квантовая Боголюбов-подобная константность
"""
import numpy as np


class SpinNoiseRig:
    def __init__(self, fs=1.0e9, n=2048, tau_mag=50.0e-9, tau_f=0.4e-9,
                 sigma_th=1.0, shot_rate=0.35, shot_q=0.9, fano=1.0,
                 zp_amp=0.35, pair_rate=3.0, pair_amp=0.35,
                 chop_period=128, seed=0):
        self.fs, self.n = float(fs), int(n)
        self.dt = 1.0 / self.fs
        self.t = np.arange(self.n) * self.dt
        self.phi_mag = np.exp(-self.dt / tau_mag)
        self.sigma_ou = sigma_th * np.sqrt(1.0 - np.exp(-2.0 * self.dt / tau_mag))
        self.phi_f = np.exp(-self.dt / tau_f)
        self.shot_rate, self.shot_amp, self.fano = shot_rate, shot_amp, fano
        self.zp_amp, self.pair_rate, self.pair_amp = zp_amp, pair_rate, pair_amp
        self.carriers = 2 * np.pi * np.array([60e6, 100e6, 140e6, 180e6])
        self.delays = np.array([8e-9, 16e-9, 24e-9])
        self.env_w = 3e-9
        self.lam = shot_rate
        self.gain_f = 1.0 / (1.0 - self.phi_f)          # ансамблевый DC-гейн фильтра
        half = chop_period // 2
        self.chop = np.where((np.arange(self.n) // half) % 2 == 0, 1.0, -1.0)

    # ---------- генераторы ----------
    def thermal(self, rng):
        x = np.zeros(self.n)
        for i in range(1, self.n):
            x[i] = x[i - 1] * self.phi_mag + self.sigma_ou * rng.standard_normal()
        return x

    def shot(self, rng):
        counts = rng.poisson(self.lam, self.n).astype(float)
        if self.fano != 1.0:
            counts = self.lam + (counts - self.lam) * np.sqrt(self.fano)
        q = np.zeros(self.n)
        for i in range(1, self.n):
            q[i] = self.phi_f * q[i - 1] + counts[i - 1]
        q -= self.lam * self.gain_f        # АНСАМБЛЕВОЕ центрирование, не выборочное
        return self.shot_q * q / np.sqrt(self.lam * self.gain_f + 1e-12)

    def quantum(self, rng):
        x = self.zp_amp * rng.standard_normal(self.n)
        n_pairs = int(rng.poisson(self.pair_rate))
        for _ in range(n_pairs):
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

    def trace(self, rng, species, dc=5.0, chopped=False):
        noise = {"thermal": self.thermal, "shot": self.shot,
                 "quantum": self.quantum}[species](rng)
        return dc * (self.chop if chopped else 1.0) + noise, noise

    # ---------- признаки ----------
    def _envelope(self, x):
        from scipy.signal import hilbert
        return np.abs(hilbert(x - x.mean()))

    def features(self, x):
        d = np.diff(x)
        sd = d.std() + 1e-12
        skew = float(((d - d.mean()) ** 3).mean() / sd ** 3)
        kurt = float(((d - d.mean()) ** 4).mean() / sd ** 4) - 3.0
        ac1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        sp = np.abs(np.fft.rfft(x - x.mean())) ** 2
        hf = float(sp[len(sp) // 2:].mean() / (sp.mean() + 1e-12))
        fano = float(d.var() / (np.abs(d).mean() ** 2 + 1e-12))
        env = self._envelope(x)
        env = env - env.mean()
        e0 = float(np.sqrt((env ** 2).mean()) + 1e-12)
        pair_ms = []
        for w in self.carriers:
            band = np.abs(np.fft.irfft(
                np.fft.rfft(x - x.mean()) *
                np.exp(-(np.fft.rfftfreq(self.n, self.dt) - w / (2 * np.pi)) ** 2
                       / (2 * 20e6 ** 2)), self.n))
            eb = band - band.mean()
            vals = []
            for dl in self.delays:
                k = int(dl * self.fs)
                vals.append(float(np.mean(eb[:-k] * eb[k:])) /
                            (np.sqrt(np.mean(eb ** 2)) * np.sqrt(np.mean(eb[k:] ** 2)) + 1e-12))
            pair_ms.append(max(vals))
        pair_m = float(max(pair_ms))
        pair_dev = float(np.std(pair_ms) / (np.mean(pair_ms) + 1e-12))
        return dict(var=float(x.var()), skew=skew, kurt=kurt, ac1=ac1,
                    hf=hf, fano=fano, pair_m=pair_m, pair_dev=pair_dev)

    # ---------- очистка (только классические виды) ----------
    def clean_thermal(self, x):
        alpha = 1.0 - self.phi_mag
        s = np.zeros_like(x)
        s[0] = x[0]
        for i in range(1, self.n):
            s[i] = s[i - 1] + alpha * (x[i] - s[i - 1])
        return x - (s - s.mean())

    def clean_shot(self, x):
        half = 4
        med = np.array([np.median(x[max(0, i - half):i + half + 1])
                        for i in range(self.n)])
        r = x - med
        thr = 3.0 * np.std(r)
        out = x.copy()
        m = np.abs(r) > thr
        out[m] = med[m]
        return out

    def clean(self, x, species):
        """Квантовый вид НЕ чистится классически: только свидетельство."""
        if species == "thermal":
            return self.clean_thermal(x)
        if species == "shot":
            return self.clean_shot(x)
        return x  # quantum: witness-only

    def readout(self, x, clip=False):
        """Синхронное детектирование: y = x*chop, среднее = оценка DC.
        clip=True подавляет разреженные дробовые импульсы (MAD-порог)."""
        y = x * self.chop
        if clip:
            med = np.median(y)
            r = y - med
            s = 1.4826 * np.median(np.abs(r)) + 1e-12
            y = med + np.clip(r, -3.0 * s, 3.0 * s)
        return float(y.mean())
        