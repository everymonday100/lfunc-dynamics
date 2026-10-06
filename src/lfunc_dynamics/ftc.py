r"""ftc.py (v1.3) — Floating Temporal Corrections + radiation-aware layer.
v1.3: DriftingRig injects single-event thermal spikes (Curie-crossing) and
TMR wear (barrier displacement damage); FTCLayer gains set_detect (run-length
+ amplitude stationarity break), a TMR amplitude dosimeter, and healing
ladder L1 adapt / L2 SET rewrite-verify / L3 wear quarantine."""
import numpy as np

FEATS = ["var", "skew", "kurt", "ac1", "hf", "fano", "pair_m", "pair_dev"]


class DriftingRig:
    def __init__(self, rig, n_total=1200, g_tau=0.4, g_lam=1.2, g_F=0.25,
                 spike_rate=0.03, spike_amp=8.0, spike_width=40e-9,
                 wear_tmr=0.3):
        self.rig, self.n_total = rig, n_total
        self.g = (g_tau, g_lam, g_F)
        self.spike_rate, self.spike_amp = spike_rate, spike_amp
        self.spike_width, self.wear_tmr = spike_width, wear_tmr

    def params(self, i):
        f = i / max(1, self.n_total - 1)
        gt, gl, gf = self.g
        return dict(tau_mag=50e-9 * (1 + gt * f),
                    lam=0.35 * (1 + gl * f),
                    fano=1.0 + gf * f,
                    amp=1.0 - self.wear_tmr * f)

    def trace(self, rng, species, i, dc=5.0, chopped=False):
        p, r = self.params(i), self.rig
        old = (r.phi, r.sigma_ou, r.lam, r.fano)
        r.phi = np.exp(-r.dt / p["tau_mag"])
        r.sigma_ou = r.sigma_th * np.sqrt(1.0 - r.phi ** 2)
        r.lam, r.fano = p["lam"], p["fano"]
        spike = False
        try:
            dc_eff = dc * p["amp"]
            if species == "mixed":
                out, _ = r.trace_mixed(rng, dc=dc_eff, chopped=chopped)
            elif species in ("thermal", "shot", "quantum"):
                out, _ = r.trace(rng, species, dc=dc_eff, chopped=chopped)
            else:
                raise ValueError(species)
            if rng.random() < self.spike_rate:   # термический пик (Кюри-транзиент)
                t0 = rng.uniform(0.0, r.t[-1] - 10 * self.spike_width)
                g = np.exp(-(r.t - t0) ** 2 / (2 * self.spike_width ** 2))
                out = out + self.spike_amp * g * float(rng.choice([-1.0, 1.0]))
                spike = True
        finally:
            r.phi, r.sigma_ou, r.lam, r.fano = old
        p["spike"] = spike
        p["dc_eff"] = dc * p["amp"]
        return out, p


class FTCLayer:
    def __init__(self, dt=1e-9, beta=0.05, eta=0.05, target_flag=0.08,
                 tol_tau=0.15, tol_d=3.0, set_amp=6.0, set_run=30,
                 wear_tol=0.15):
        self.dt, self.beta, self.eta = dt, beta, eta
        self.target_flag, self.tol_tau, self.tol_d = target_flag, tol_tau, tol_d
        self.set_amp, self.set_run, self.wear_tol = set_amp, set_run, wear_tol
        self.log_tau0 = np.log(50e-9)
        self.log_tau = self.log_tau0
        self.log_amp = 0.0
        self.theta = None
        self.mu = self.sd = None
        self.ref, self.ref_sd, self.ema = {}, {}, {}
        self.level = 0
        self.events = []

    def init_calib(self, sigmas, feat_rows, routes):
        self.theta = float(np.quantile(sigmas, 0.95))
        F = np.array([[f[c] for c in FEATS] for f in feat_rows], float)
        self.mu, self.sd = F.mean(0), F.std(0) + 1e-12
        for rt in set(routes):
            m = routes == rt if isinstance(routes, np.ndarray) else \
                np.array([r == rt for r in routes])
            self.ref[rt] = F[m].mean(0)
            self.ref_sd[rt] = F[m].std(0) + 1e-12
            self.ema[rt] = self.ref[rt].copy()

    def standardize(self, feat):
        v = (np.array([feat[c] for c in FEATS], float) - self.mu) / self.sd
        return v.reshape(1, -1)

    def set_detect(self, x):
        """SEU-транзиент: амплитуда > 6 MAD И пробег >= set_run отсчётов
        (дробовые импульсы дают пробег 1-3, пик Кюри - десятки)."""
        r = x - np.median(x)
        s = 1.4826 * float(np.median(np.abs(r))) + 1e-12
        m = (np.abs(r) > 4.0 * s).astype(np.int8)
        d = np.diff(np.concatenate(([0], m, [0])))
        st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        longest = int((en - st).max()) if st.size else 0
        amp = float(np.max(np.abs(r)) / s)
        return amp, longest

    def update(self, x, sigma, feat, route, i, amp_hat=None):
        fv = np.array([feat[c] for c in FEATS], float)
        self.mu = (1 - self.beta) * self.mu + self.beta * fv
        self.sd = (1 - self.beta) * self.sd + self.beta * np.abs(fv - self.mu)
        flag = float(sigma > self.theta)
        self.theta *= np.exp(self.eta * (flag - self.target_flag))
        if route in self.ema:
            self.ema[route] = ((1 - self.beta) * self.ema[route]
                               + self.beta * fv)
        if route == "thermal":
            m = x.mean()
            a = min(max(float(np.corrcoef(x[:-1] - m, x[1:] - m)[0, 1]), 0.05), 0.999)
            self.log_tau = ((1 - self.beta) * self.log_tau
                            + self.beta * np.log(-self.dt / np.log(a)))
        if amp_hat is not None and amp_hat > 0:
            self.log_amp = ((1 - self.beta) * self.log_amp
                            + self.beta * np.log(amp_hat))
        a, L = self.set_detect(x)
        set_flag = bool(a > self.set_amp and L >= self.set_run)
        wear = 1.0 - float(np.exp(self.log_amp))
        d_tau = abs(self.log_tau - self.log_tau0)
        d_route = 0.0
        if route in self.ema:
            d_route = float(np.max(np.abs(self.ema[route] - self.ref[route])
                                   / self.ref_sd[route]))
        if set_flag:
            self.events.append((i, "SET", route, round(a, 1), L))
        if wear > self.wear_tol and not any(e[1] == "WEAR" for e in self.events):
            self.events.append((i, "WEAR", route, round(wear, 3)))
        if (d_tau > self.tol_tau or d_route > self.tol_d) and self.level < 1:
            self.level = 1
            self.events.append((i, "ADAPT", route, round(d_tau, 3),
                                round(d_route, 2)))
        return set_flag, wear

    def chop_period(self):
        tau = np.exp(self.log_tau)
        p = int(1.0 / (6 * (1.0 / (2 * np.pi * tau)) * self.dt))
        return int(min(max(p, 32), 256))

    def clip_thr(self, x_on, period):
        y = x_on * _chop(len(x_on), period)
        r = y - np.median(y)
        return 3.0 * 1.4826 * float(np.median(np.abs(r))) + 1e-12


def _chop(n, period):
    half = max(1, period // 2)
    return np.where((np.arange(n) // half) % 2 == 0, 1.0, -1.0)


def demod_readout(x_on, period):
    return float((x_on * _chop(len(x_on), period)).mean())


def clip_readout(x_on, period, thr):
    y = x_on * _chop(len(x_on), period)
    med = np.median(y)
    r = y - med
    return float((med + np.clip(r, -thr, thr)).mean())