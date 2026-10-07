r"""ml_cabinet.py — CoreCabinet: tick-latched temporal ensemble for calibrated confidence.

Контракты вызова (оба поддерживаются):
  (A) CoreCabinet(surr, K=K, T=T)
      surr — объект с атрибутами surr.models (список B моделей) и
             surr.sigma_res (скаляр, residual std для эпистемической шкалы).
      Используется в spin_noise_bench.py и spin_ftc_bench.py.

  (B) CoreCabinet(base_estimators=[m1,...,mB], K=K, T=T)
      base_estimators — список из B = K*T sklearn-совместимых моделей.
      Используется в standalone-демо.

predict(X, ensemble, std_fn) -> объект с атрибутами:
  value, sigma_tick, quadrant, prediction, ci_lower, ci_upper
"""
import numpy as np
from typing import List, Optional


class CabinetResult:
    """Контейнер результата предсказания CoreCabinet."""
    def __init__(self, prediction, ci_lower, ci_upper, sigma_tick, quadrant, value):
        self.prediction = prediction
        self.ci_lower = ci_lower
        self.ci_upper = ci_upper
        self.sigma_tick = sigma_tick
        self.quadrant = quadrant
        self.value = value


class CoreCabinet:
    """Tick-latched temporal ensemble.

    Parameters
    ----------
    base_estimators : list | object
        Либо список из B = K*T моделей, либо объект-суррогат с атрибутом .models.
    K, T : int
        Размерность сетки (cores x ticks).
    theta : float, optional
        Порог sigma_tick (калибруется через calibrate()).
    rel_w_threshold : float
        Порог относительной ширины доверительного интервала (default 0.5).
    """

    def __init__(self, base_estimators, K: int = 4, T: int = 4,
                 theta: Optional[float] = None, rel_w_threshold: float = 0.5):
        self.K = K
        self.T = T
        self.B = K * T
        self.theta = theta
        self.rel_w_threshold = rel_w_threshold

        # Универсальный разбор: surr.models или список
        if hasattr(base_estimators, "models"):
            self.models = list(base_estimators.models)
            self.sigma_res = float(getattr(base_estimators, "sigma_res", 1.0))
        else:
            self.models = list(base_estimators)
            self.sigma_res = 1.0

        if len(self.models) != self.B:
            raise ValueError(
                f"CoreCabinet expects B=K*T={self.B} base estimators, "
                f"got {len(self.models)}"
            )

    # ---- ядро: прогон всех B ячеек ----
    def _run_cells(self, X_std: np.ndarray) -> np.ndarray:
        """Возвращает матрицу предсказаний формы (B, n_samples)."""
        n = X_std.shape[0]
        P = np.zeros((self.B, n))
        for b, m in enumerate(self.models):
            try:
                p = m.predict(X_std)
                if p.ndim == 2:
                    # Классификатор -> берём вероятность положительного класса
                    p = p[:, 1]
                P[b] = p
            except Exception as e:
                raise RuntimeError(f"Model {b} failed: {e}") from e
        return P

    # ---- двухэтапная агрегация ----
    def _aggregate(self, P: np.ndarray) -> dict:
        n = P.shape[1]
        mu_t = np.zeros((self.T, n))
        d_t = np.zeros((self.T, n))
        for t in range(self.T):
            cells = P[t * self.K:(t + 1) * self.K]  # (K, n)
            mu_t[t] = np.median(cells, axis=0)
            d_t[t] = np.std(cells, axis=0)

        prediction = np.median(mu_t, axis=0)          # across-tick median
        sigma_tick = np.mean(d_t, axis=0)             # mean within-tick std
        sigma_ep = np.std(P, axis=0)                  # epistemic scale
        return dict(prediction=prediction, sigma_tick=sigma_tick,
                    sigma_ep=sigma_ep)

    # ---- квадрант ----
    def _quadrant(self, prediction, sigma_tick, sigma_ep) -> str:
        ci_half = 2.0 * sigma_ep
        rel_w = ci_half / max(abs(prediction), 0.5)
        if rel_w < self.rel_w_threshold and sigma_tick < self.theta:
            return "reliable"
        if rel_w < self.rel_w_threshold and sigma_tick >= self.theta:
            return "overconfident"
        if rel_w >= self.rel_w_threshold and sigma_tick >= self.theta:
            return "honest_low"
        return "calibration_artifact"

    # ---- главный интерфейс, совместимый со старыми бенчами ----
    def predict(self, X, ensemble="h", std_fn=None):
        """Предсказание для одного образца.

        Parameters
        ----------
        X : dict | np.ndarray
            Словарь признаков (если std_fn сам их извлекает) или вектор (n_features,).
        ensemble : str
            Идентификатор ансамбля (передаётся в std_fn).
        std_fn : callable
            Функция стандартизации std_fn(X, ensemble) -> (1, n_features).

        Returns
        -------
        CabinetResult
        """
        # 1. Стандартизация через std_fn (контракт бенчей)
        if std_fn is not None:
            X_std = std_fn(X, ensemble)
        else:
            X_std = np.asarray(X, dtype=float)

        if X_std.ndim == 1:
            X_std = X_std.reshape(1, -1)

        # 2. Прогон ячеек и агрегация
        P = self._run_cells(X_std)
        agg = self._aggregate(P)

        pred = float(agg["prediction"][0])
        sigma_tick = float(agg["sigma_tick"][0])
        sigma_ep = float(agg["sigma_ep"][0])

        # 3. Доверительный интервал
        ci_lower = pred - 2.0 * sigma_ep
        ci_upper = pred + 2.0 * sigma_ep

        # 4. Квадрант и «ценность» для роутинга
        quadrant = self._quadrant(pred, sigma_tick, sigma_ep)
        value_map = {"reliable": 1.0, "overconfident": 0.7,
                     "honest_low": 0.3, "calibration_artifact": 0.5}
        value = value_map.get(quadrant, 0.5)

        return CabinetResult(prediction=pred, ci_lower=ci_lower,
                             ci_upper=ci_upper, sigma_tick=sigma_tick,
                             quadrant=quadrant, value=value)

    # ---- калибровка theta на валидации ----
    def calibrate(self, X_list, ensemble="h", std_fn=None,
                  target_percentile: float = 0.95):
        """Установить theta как target_percentile-квантиль sigma_tick на X_list.

        X_list — итерируемый набор входов (словари признаков или векторы).
        Возвращает float (новое значение theta).
        """
        ticks = []
        for x in X_list:
            r = self.predict(x, ensemble=ensemble, std_fn=std_fn)
            ticks.append(r.sigma_tick)
        self.theta = float(np.percentile(ticks, target_percentile * 100))
        return self.theta