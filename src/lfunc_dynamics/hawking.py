r"""
hawking.py (v6.1) — синтетические шоты аналоговой ЧД с инвариантами Боголюбова/KMS.
Исправлено: shot() принимает опциональный n_pairs_override.
"""
import numpy as np

class HawkingRig:
    def __init__(self, nx=512, L=1.0, k_out=40.0, T_H=25.0, T_th=60.0,
                 sigma=-1.0, x0=0.45, w=0.12, seed=0, rate=2.0):
        self.nx, self.L = nx, L
        self.x = np.linspace(-L, L, nx)
        self.k_out = k_out
        self.r_grid = np.linspace(1.0, 2.2, 7)
        self.T_H = T_H
        self.T_th = T_th
        self.sigma = sigma
        self.x0 = x0
        self.w = w
        self.rate = rate
        self.w_out = np.exp(-(self.x - x0) ** 2 / (2 * w ** 2))
        self.w_in = np.exp(-(self.x + x0) ** 2 / (2 * w ** 2))
        self.k = 2 * np.pi * np.fft.rfftfreq(nx, d=(2 * L / (nx - 1)))
        self.n_th = self._occ(T_th)
        self.n_H = self._occ(T_H)

    def _occ(self, T):
        return np.where(self.k > 8.0,
                        1.0 / np.expm1(np.clip(self.k, 1e-9, None) / T), 0.0)

    def _thermal(self, rng):
        phi = np.zeros(self.nx)
        for win in (self.w_out, self.w_in):
            noise = rng.standard_normal(self.nx) * win
            phi += np.fft.irfft(np.fft.rfft(noise) * np.sqrt(self.n_th), self.nx)
        return phi

    def shot(self, rng, n_pairs_override=None):
        phi = self._thermal(rng)
        if n_pairs_override is None:
            n_pairs = int(rng.poisson(self.rate))
        else:
            n_pairs = n_pairs_override
        e_pair = 0.0
        ks = self.k[1:]
        mband = (ks > 10.0) & (ks < 120.0)
        p = self.n_H[1:] * mband
        p = p / p.sum() if p.sum() > 0 else np.ones_like(p) / len(p)
        r_true = 1.6
        for _ in range(n_pairs):
            ke = float(rng.choice(ks, p=p))
            th = rng.uniform(0, 2 * np.pi)
            phi += (self.w_out * np.cos(ke * (self.x - self.x0) + th)
                    + self.sigma * self.w_in
                    * np.cos(r_true * ke * (self.x + self.x0) + th))
            e_pair += 1.0
        return phi, dict(T_H=self.T_H, rate=float(n_pairs), n_pairs=n_pairs, pair_energy=e_pair)

    def features(self, phi):
        sp = np.fft.rfft(phi)
        best_mj, r_opt = 0.0, self.r_grid[0]
        for r in self.r_grid:
            mj_r = 0.0
            for kk in np.linspace(0.6 * self.k_out, 1.3 * self.k_out, 12):
                Tc = (self.w_out * np.cos(kk * (self.x - self.x0))
                      + self.sigma * self.w_in * np.cos(r * kk * (self.x + self.x0)))
                Ts = (self.w_out * np.sin(kk * (self.x - self.x0))
                      + self.sigma * self.w_in * np.sin(r * kk * (self.x + self.x0)))
                val = float(np.hypot(phi @ Tc, phi @ Ts)) / np.sqrt(np.sum(Tc ** 2) + 1e-12)
                mj_r = max(mj_r, val)
            if mj_r > best_mj:
                best_mj, r_opt = mj_r, r

        K_bins = np.linspace(0.6 * self.k_out, 1.3 * self.k_out, 12)
        bog_ratios, kms_ratios = [], []
        m_j_total, m_out_total, m_in_total = 0.0, 0.0, 0.0
        for kk in K_bins:
            Tc = (self.w_out * np.cos(kk * (self.x - self.x0))
                  + self.sigma * self.w_in * np.cos(r_opt * kk * (self.x + self.x0)))
            Ts = (self.w_out * np.sin(kk * (self.x - self.x0))
                  + self.sigma * self.w_in * np.sin(r_opt * kk * (self.x + self.x0)))
            mj = float(np.hypot(phi @ Tc, phi @ Ts)) / np.sqrt(np.sum(Tc ** 2) + 1e-12)
            oc = self.w_out * np.cos(kk * (self.x - self.x0))
            os_ = self.w_out * np.sin(kk * (self.x - self.x0))
            mo = float(np.hypot(phi @ oc, phi @ os_)) / np.sqrt(np.sum(oc ** 2) + 1e-12)
            ic = self.w_in * np.cos(r_opt * kk * (self.x + self.x0))
            is_ = self.w_in * np.sin(r_opt * kk * (self.x + self.x0))
            mi = float(np.hypot(phi @ ic, phi @ is_)) / np.sqrt(np.sum(ic ** 2) + 1e-12)
            m_j_total = max(m_j_total, mj)
            m_out_total = max(m_out_total, mo)
            m_in_total = max(m_in_total, mi)
            bog_ratios.append(mj ** 2 / (mo * mi + 1e-9))
            kms_ratios.append(mo / (mi + 1e-9))

        R = m_j_total ** 2 / (m_out_total ** 2 + m_in_total ** 2 + 1e-9)
        bog_dev = float(np.std(bog_ratios))
        kms_var = float(np.var(kms_ratios))

        return dict(E_out=m_out_total ** 2, E_in=m_in_total ** 2,
                    m_j=m_j_total, m_out=m_out_total, m_in=m_in_total,
                    R=R, bog_dev=bog_dev, kms_var=kms_var, r_opt=r_opt)