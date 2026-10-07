r"""temporal_layer.py (v1.0, AFTER-1) — tick-latched temporal layer for continuous AI.

AFTER-1 adds (isolated improvement #1 of 5):
 - TemporalMemory.save()/load(): ring buffer persists to .npz between runs;
 - main() performs a reload test (context_vector must match) and records
   memory_path / reload_match / context vectors in the verdict.

Everything else is identical to the BEFORE stage (v0.9-adapted) so the
before/after comparison stays clean:
 - deterministic simulated timing (seeded lognormal jitter + stalls);
 - per-sample within-tick dispersion retained (CoreCabinet contract);
 - self-report head reads the FiLM-modulated hidden state;
 - verdicts dump mirroring spin/hawking benches.

Remaining improvements (each gets its own isolated verdict):
 #2 end-to-end conditioner training on a temporally structured task;
 #3 baseline arm without conditioner (ablation);
 #4 calibration of P(s_tick > theta) on held-out episodes;
 #5 K = 8/16 vs theory 1/sqrt(2(K-1)).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Records and embeddings
# ---------------------------------------------------------------------------
@dataclass
class TickRecord:
    """One tick: K parallel cells finishing at slightly different times."""
    tick_index: int
    cell_times_ns: List[int]
    cell_predictions: List[float]
    coherent: bool
    dispersion_per_sample: Optional[np.ndarray] = None

    @property
    def duration_ns(self) -> int:
        if not self.cell_times_ns:
            return 0
        return max(self.cell_times_ns) - min(self.cell_times_ns)

    @property
    def completion_ns(self) -> int:
        return max(self.cell_times_ns) if self.cell_times_ns else 0

    @property
    def dispersion(self) -> float:
        if len(self.cell_predictions) < 2:
            return 0.0
        return float(np.std(self.cell_predictions))


@dataclass
class TemporalEmbedding:
    """Aggregated temporal signal for one episode."""
    tick_durations_ns: List[int] = field(default_factory=list)
    tick_dispersions: List[float] = field(default_factory=list)
    staleness_flags: List[int] = field(default_factory=list)
    cumulative_ns: int = 0
    tick_count: int = 0

    def to_vector(self, max_ticks: int = 16, max_ns: float = 1e9) -> np.ndarray:
        """Fixed-size 8-feature vector (durations, dispersions, staleness, time)."""
        if not self.tick_durations_ns:
            return np.zeros(8, dtype=np.float32)
        d = np.asarray(self.tick_durations_ns, dtype=np.float64)
        s = np.asarray(self.tick_dispersions, dtype=np.float64)
        st = np.asarray(self.staleness_flags, dtype=np.float64)
        return np.array([
            d.mean() / max_ns,
            d.std() / max_ns,
            s.mean() if len(s) else 0.0,
            s.std() if len(s) else 0.0,
            st.mean() if len(st) else 0.0,
            self.tick_count / max_ticks,
            self.cumulative_ns / max_ns,
            np.log1p(self.cumulative_ns) / np.log1p(max_ns),
        ], dtype=np.float32)

    def summary(self) -> str:
        md = np.mean(self.tick_dispersions) if self.tick_dispersions else 0.0
        return (f"ticks={self.tick_count} cum={self.cumulative_ns / 1e6:.2f}ms "
                f"mean_disp={md:.4f} stale={int(np.sum(self.staleness_flags))}")


# ---------------------------------------------------------------------------
# 2. Temporal memory (ring buffer; AFTER-1: persists to disk)
# ---------------------------------------------------------------------------
class TemporalMemory:
    """Ring buffer of temporal embeddings; survives across episodes AND runs."""

    def __init__(self, capacity: int = 64, dim: int = 8):
        self.capacity = capacity
        self.dim = dim
        self._buf = np.zeros((capacity, dim), dtype=np.float32)
        self._ptr = 0
        self._size = 0

    def push(self, vec: np.ndarray) -> None:
        self._buf[self._ptr] = vec
        self._ptr = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def recent(self, n: int = 8) -> np.ndarray:
        if self._size == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        n = min(n, self._size)
        idx = [(self._ptr - 1 - i) % self.capacity for i in range(n)]
        idx.reverse()
        return self._buf[idx]

    def context_vector(self) -> np.ndarray:
        """Mean of recent history, or zeros if empty."""
        r = self.recent(n=8)
        if r.shape[0] == 0:
            return np.zeros(self.dim, dtype=np.float32)
        return r.mean(axis=0)

    def __len__(self) -> int:
        return self._size

    # ---- AFTER-1: persistence ----
    def save(self, path: str) -> None:
        """Persist ring buffer to disk (npz)."""
        np.savez_compressed(
            path,
            buf=self._buf,
            ptr=np.array([self._ptr]),
            size=np.array([self._size]),
            capacity=np.array([self.capacity]),
            dim=np.array([self.dim]),
        )

    @classmethod
    def load(cls, path: str) -> "TemporalMemory":
        """Restore ring buffer from disk."""
        data = np.load(path)
        mem = cls(capacity=int(data["capacity"][0]), dim=int(data["dim"][0]))
        mem._buf = data["buf"].astype(np.float32)
        mem._ptr = int(data["ptr"][0])
        mem._size = int(data["size"][0])
        return mem


# ---------------------------------------------------------------------------
# 3. Base predictor
# ---------------------------------------------------------------------------
class BasePredictor(nn.Module):
    """Small MLP returning (output, hidden) so the temporal layer can modulate."""

    def __init__(self, in_dim: int = 32, hidden: int = 64, out_dim: int = 1):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.head(h), h


# ---------------------------------------------------------------------------
# 4. Temporal conditioner (FiLM over the hidden state)
# ---------------------------------------------------------------------------
class TemporalConditioner(nn.Module):
    """h' = gamma * h + beta, initialized near identity."""

    def __init__(self, temporal_dim: int = 8, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(temporal_dim, 32), nn.GELU(), nn.Linear(32, 2 * hidden_dim))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        with torch.no_grad():
            self.net[-1].bias[:hidden_dim] = 1.0

    def forward(self, h: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        gb = self.net(e_t)
        gamma, beta = gb.chunk(2, dim=-1)
        return gamma * h + beta


# ---------------------------------------------------------------------------
# 5. Self-report head
# ---------------------------------------------------------------------------
class TemporalSelfReport(nn.Module):
    """Predicts tick dispersion from the (modulated) hidden state."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


# ---------------------------------------------------------------------------
# 6. Tick-latched ensemble with temporal awareness
# ---------------------------------------------------------------------------
class TemporalEnsemble(nn.Module):
    """K cores x T ticks; records timings, builds embedding, FiLM-injects it."""

    def __init__(self, base_factory: Callable[[], BasePredictor],
                 n_cores: int = 4, n_ticks: int = 4,
                 coherence_budget_ms: float = 0.2,
                 in_dim: int = 32, hidden_dim: int = 64, temporal_dim: int = 8,
                 memory_capacity: int = 64, timing_mode: str = "sim",
                 seed: int = 0, stall_prob: float = 0.1):
        super().__init__()
        self.n_cores = n_cores
        self.n_ticks = n_ticks
        self.coherence_budget_ns = int(coherence_budget_ms * 1e6)
        self.hidden_dim = hidden_dim
        self.temporal_dim = temporal_dim
        self.timing_mode = timing_mode
        self.stall_prob = stall_prob
        self._rng = np.random.default_rng(seed)
        self.cores = nn.ModuleList([base_factory() for _ in range(n_cores)])
        self.conditioner = TemporalConditioner(temporal_dim, hidden_dim)
        self.self_report = TemporalSelfReport(hidden_dim)
        self.memory = TemporalMemory(capacity=memory_capacity, dim=temporal_dim)
        self.readout = nn.Linear(hidden_dim, 1)
        self._theta: Optional[float] = None

    @staticmethod
    def _now_ns() -> int:
        return time.perf_counter_ns()

    def _sim_cell_time_ns(self) -> int:
        base = 50_000  # 50 us nominal per cell
        t = base * self._rng.lognormal(0.0, 0.3)
        if self._rng.random() < self.stall_prob:
            t *= self._rng.uniform(10.0, 100.0)
        return int(t)

    def forward(self, x: torch.Tensor, persist_memory: bool = True,
                return_diagnostics: bool = False):
        batch = x.shape[0]
        records: List[TickRecord] = []
        h_accum = torch.zeros(batch, self.hidden_dim, device=x.device)
        y_accum = torch.zeros(batch, 1, device=x.device)
        episode_start = self._now_ns()

        for t in range(self.n_ticks):
            cell_times, cell_preds, cell_hiddens, cell_vecs = [], [], [], []
            for k in range(self.n_cores):
                if self.timing_mode == "wall":
                    t0 = self._now_ns()
                    y_k, h_k = self.cores[k](x)
                    dt = self._now_ns() - t0
                else:
                    y_k, h_k = self.cores[k](x)
                    dt = self._sim_cell_time_ns()
                cell_times.append(dt)
                cell_preds.append(float(y_k.mean().item()))
                cell_hiddens.append(h_k)
                cell_vecs.append(y_k)
            stack = torch.stack(cell_vecs, dim=0)               # (K, batch, 1)
            d_per_sample = stack.std(dim=0).squeeze(-1).detach().cpu().numpy()
            h_tick = torch.stack(cell_hiddens, dim=0).mean(dim=0)
            y_tick = stack.mean(dim=0)                          # (batch, 1)
            h_accum = h_accum + h_tick
            y_accum = y_accum + y_tick
            dur = max(cell_times) - min(cell_times)
            records.append(TickRecord(
                tick_index=t, cell_times_ns=cell_times,
                cell_predictions=cell_preds,
                coherent=dur <= self.coherence_budget_ns,
                dispersion_per_sample=d_per_sample))

        h_mean = h_accum / self.n_ticks
        y_raw = (y_accum / self.n_ticks).squeeze(-1)            # (batch,)

        e_obj = TemporalEmbedding(
            tick_durations_ns=[r.duration_ns for r in records],
            tick_dispersions=[r.dispersion for r in records],
            staleness_flags=[0 if r.coherent else 1 for r in records],
            cumulative_ns=self._now_ns() - episode_start,
            tick_count=self.n_ticks)
        e_np = e_obj.to_vector(max_ticks=self.n_ticks * 4)
        e_t = torch.from_numpy(e_np).to(x.device)

        h_mod = self.conditioner(h_mean, e_t)
        y_final = self.readout(h_mod).squeeze(-1)
        s_pred = self.self_report(h_mod)   # читает МОДУЛИРОВАННЫЙ hidden

        if persist_memory:
            self.memory.push(e_np)

        s_tick = float(np.mean(e_obj.tick_dispersions)) if e_obj.tick_dispersions else 0.0
        out = {"y": y_final, "y_raw": y_raw, "s_tick": s_tick, "e_t": e_np,
               "self_report": s_pred, "cumulative_ns": e_obj.cumulative_ns}
        if return_diagnostics:
            out["records"] = records
            out["embedding"] = e_obj
        return out

    def calibrate_theta(self, s_ticks, percentile: float = 95.0) -> float:
        self._theta = float(np.percentile(s_ticks, percentile))
        return self._theta

    def flagged_fraction(self, s_ticks) -> float:
        if self._theta is None:
            raise ValueError("theta not calibrated")
        return float(np.mean(np.asarray(s_ticks) > self._theta))

    def temporal_context(self) -> np.ndarray:
        return self.memory.context_vector()


# ---------------------------------------------------------------------------
# 7. Tests for functional temporality
# ---------------------------------------------------------------------------
def test_self_report_learns(model: TemporalEnsemble, n_steps: int = 200):
    opt = torch.optim.Adam(model.self_report.parameters(), lr=1e-3)
    losses = []
    for _ in range(n_steps):
        x = torch.randn(16, 32)
        out = model(x, persist_memory=False)
        target = torch.full((16,), out["s_tick"])
        loss = F.mse_loss(out["self_report"], target)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses[0], losses[-1]


def test_conditioning_changes_output(model: TemporalEnsemble) -> float:
    x = torch.randn(8, 32)
    out = model(x, persist_memory=False)
    return (out["y"] - out["y_raw"]).abs().mean().item()


def test_memory_persists(model: TemporalEnsemble, n_episodes: int = 5) -> int:
    for _ in range(n_episodes):
        model(torch.randn(4, 32), persist_memory=True)
    return len(model.memory)


# ---------------------------------------------------------------------------
# 8. Main: verdict dump + AFTER-1 reload test
# ---------------------------------------------------------------------------
def main(stage: str = "after1", n_episodes: int = 40, seed: int = 0,
         out: Optional[str] = None):
    if out is None:
        out = f"temporal_verdicts_{stage}.json"
    torch.manual_seed(seed); np.random.seed(seed)
    factory = lambda: BasePredictor(in_dim=32, hidden=64, out_dim=1)
    model = TemporalEnsemble(factory, n_cores=4, n_ticks=4,
                             coherence_budget_ms=0.2, timing_mode="sim",
                             seed=seed, stall_prob=0.1)

    s_ticks, stale = [], 0
    for _ in range(n_episodes):
        o = model(torch.randn(8, 32), return_diagnostics=True)
        s_ticks.append(o["s_tick"])
        stale += sum(0 if r.coherent else 1 for r in o["records"])

    l0, l1 = test_self_report_learns(model)
    diff = test_conditioning_changes_output(model)
    size = test_memory_persists(model)

    # ---- AFTER-1: persist memory and verify reload ----
    mem_path = f"temporal_memory_{stage}.npz"
    model.memory.save(mem_path)
    reloaded = TemporalMemory.load(mem_path)
    ctx_before = model.memory.context_vector()
    ctx_after = reloaded.context_vector()
    reload_match = bool(np.allclose(ctx_before, ctx_after, atol=1e-6))

    verdict = dict(
        stage=stage, seed=seed, K=model.n_cores, T=model.n_ticks,
        selfreport_mse_first=l0, selfreport_mse_last=l1,
        selfreport_ratio=l1 / max(l0, 1e-12),
        conditioning_diff=diff, memory_size=size,
        s_tick_mean=float(np.mean(s_ticks)),
        s_tick_std=float(np.std(s_ticks)),
        stale_ticks_total=int(stale),
        memory_path=mem_path, reload_match=reload_match,
        context_before=ctx_before.tolist(),
        context_after=ctx_after.tolist(),
        s_ticks=s_ticks,
    )
    with open(out, "w") as fh:
        json.dump(verdict, fh, indent=1)
    print(json.dumps({k: v for k, v in verdict.items()
                      if k not in ("s_ticks",)}, indent=1))
    print(f"Saved: {out}")


if __name__ == "__main__":
    import sys
    stage = sys.argv[1] if len(sys.argv) > 1 else "after1"
    main(stage=stage)