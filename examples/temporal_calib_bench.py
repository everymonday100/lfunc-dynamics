#!/usr/bin/env python3
r"""temporal_calib_bench.py — AFTER-3/4: calibration of P(s_tick > theta).

Протокол:
 1. Сбор s_tick + hidden на calibration (seed 1) и held-out (seed 2) эпизодах.
 2. Диагностика нуль-инфляции/скошенности: frac(s<1e-9), медиана, 95pct, skew.
 3. theta = 95-й эмпирический перцентиль calibration-набора (НЕ mean+1.645*std).
 4. Три оценщика P(s_tick>theta) на held-out:
    (a) гауссовский хвост из calibration mean/std  — отвергнутая статистика;
    (b) перенос эмпирического квантиля              — популяционный уровень;
    (c) CalibHead (BCE по hidden)                   — поэкземплярный уровень.
 5. Метрики: flagged fraction hold vs 0.05, ECE(10 bins), Brier, AUC,
    медианная относительная ошибка raw- vs log-шкалы MSE-head на хвосте.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lfunc_dynamics.temporal_layer import BasePredictor, TemporalEnsemble

EPS = 1e-9
N_CAL, N_HOLD = 400, 400


def phi_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


class CalibHead(nn.Module):
    """P(s_tick > theta) из hidden-состояния: sigmoid + BCE."""
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(),
                                 nn.Linear(32, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(h).squeeze(-1))


def collect(model, n_episodes: int, seed: int, batch: int = 16):
    torch.manual_seed(seed)
    s_ticks, H = [], []
    for _ in range(n_episodes):
        x = torch.randn(batch, 32)
        out = model(x, persist_memory=False, return_diagnostics=True)
        s_ticks.append(out["s_tick"])
        H.append(out["h_mod"].mean(0).numpy())   # episode-level hidden
    return np.array(s_ticks, float), np.array(H, float)


def train_calib_head(H, y, steps=400, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    head = CalibHead(H.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    Ht = torch.tensor(H, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    for _ in range(steps):
        loss = F.binary_cross_entropy(head(Ht), yt)
        opt.zero_grad(); loss.backward(); opt.step()
    return head


def train_mse_head(H, t, steps=400, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    head = nn.Sequential(nn.Linear(H.shape[1], 32), nn.GELU(), nn.Linear(32, 1))
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    Ht = torch.tensor(H, dtype=torch.float32)
    tt = torch.tensor(t, dtype=torch.float32)
    for _ in range(steps):
        loss = F.mse_loss(head(Ht).squeeze(-1), tt)
        opt.zero_grad(); loss.backward(); opt.step()
    return head


def ece(p, y, nbins=10):
    edges = np.linspace(0.0, 1.0, nbins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
    acc = 0.0
    for b in range(nbins):
        m = idx == b
        if m.sum() == 0:
            continue
        acc += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(acc)


def auc(scores, y):
    pos, neg = scores[y == 1], scores[y == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def main():
    factory = lambda: BasePredictor(in_dim=32, hidden=64, out_dim=1)
    model = TemporalEnsemble(factory, n_cores=4, n_ticks=4,
                             coherence_budget_ms=0.2, timing_mode="sim",
                             seed=0, stall_prob=0.1)

    s_cal, H_cal = collect(model, N_CAL, seed=1)
    s_hold, H_hold = collect(model, N_HOLD, seed=2)

    # ---- диагностика нуль-инфляции / скошенности ----
    fz_cal = float(np.mean(s_cal < EPS))
    fz_hold = float(np.mean(s_hold < EPS))
    skew_cal = float((((s_cal - s_cal.mean()) / s_cal.std()) ** 3).mean()) \
        if s_cal.std() > 0 else 0.0

    # ---- калибровка theta эмпирическим квантилем ----
    theta = float(np.percentile(s_cal, 95.0))
    y_cal = (s_cal > theta).astype(float)
    y_hold = (s_hold > theta).astype(float)
    ff_hold = float(y_hold.mean())

    # ---- (a) гауссовский хвост из mean/std (отвергнутая статистика) ----
    mu, sd = float(s_cal.mean()), float(s_cal.std())
    p_gauss = 1.0 - phi_cdf((theta - mu) / sd)

    # ---- (c) поэкземплярный CalibHead ----
    head = train_calib_head(H_cal, y_cal)
    with torch.no_grad():
        p_hold = head(torch.tensor(H_hold, dtype=torch.float32)).numpy()
    e_cal = ece(p_hold, y_hold)
    a_cal = auc(p_hold, y_hold)
    brier_head = float(np.mean((p_hold - y_hold) ** 2))
    brier_const = float(np.mean((y_cal.mean() - y_hold) ** 2))
    brier_gauss = float(np.mean((p_gauss - y_hold) ** 2))

    # ---- raw vs log шкала MSE-head (урок нуль-инфляции) ----
    h_raw = train_mse_head(H_cal, s_cal)
    h_log = train_mse_head(H_cal, np.log(s_cal + 1e-3))
    nz = s_hold > EPS
    with torch.no_grad():
        Ht = torch.tensor(H_hold, dtype=torch.float32)
        pr_raw = h_raw(Ht).squeeze(-1).numpy()
        pr_log = np.exp(h_log(Ht).squeeze(-1).numpy()) - 1e-3
    rel_raw = float(np.median(np.abs(pr_raw[nz] - s_hold[nz]) / s_hold[nz]))
    rel_log = float(np.median(np.abs(pr_log[nz] - s_hold[nz]) / s_hold[nz]))

    verdict = dict(
        stage="after34", seed=0, K=4, T=4,
        n_cal=N_CAL, n_hold=N_HOLD,
        frac_zero_cal=fz_cal, frac_zero_hold=fz_hold, skew_cal=skew_cal,
        theta=theta,
        flag_frac_cal=float(y_cal.mean()), flag_frac_hold=ff_hold,
        quantile_transfer_gap=abs(ff_hold - 0.05),
        p_gaussian_tail=p_gauss,
        gaussian_miscalib_gap=abs(p_gauss - 0.05),
        ece_calib_head=e_cal, auc_calib_head=a_cal,
        brier_head=brier_head, brier_const=brier_const, brier_gauss=brier_gauss,
        med_rel_err_raw_head=rel_raw, med_rel_err_log_head=rel_log,
    )
    out = Path("temporal_verdicts_calib.json")
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=1)
    print(json.dumps(verdict, indent=1))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()