"""Bagged-суррогат conv с уровнем уверенности и правилами действий."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold

GROUPS = {
    "G0_spectral": ["h0", "r2"],
    "G1_local": ["h0", "r2", "log_mingap", "log_P0", "skew"],
    "G2_phase": ["h0", "r2", "log_mingap", "log_P0", "skew", "kurt", "bic", "qpc"],
}

class ConvSurrogate:
    def __init__(self, group="G2_phase", n_bag=16, seed=0):
        self.group = group
        self.cols = GROUPS[group]
        self.n_bag = n_bag
        self.rng = np.random.default_rng(seed)

    def _matrix(self, feats_list, ensembles):
        enc = sorted(set(ensembles))
        self._enc = enc if not hasattr(self, "_enc") else self._enc
        rows = []
        for f, e in zip(feats_list, ensembles):
            r = [f[c] for c in self.cols]
            r += [1.0 if e == k else 0.0 for k in self._enc]
            rows.append(r)
        return np.array(rows, float)

    def fit(self, feats_list, y, ensembles):
        X = self._matrix(feats_list, ensembles)
        y = np.asarray(y, float)
        self._mu, self._sd = X.mean(0), X.std(0) + 1e-12
        Xs = (X - self._mu) / self._sd
        n = len(y)
        self.models = []
        for _ in range(self.n_bag):
            idx = self.rng.choice(n, n, replace=True)
            m = HistGradientBoostingRegressor(random_state=int(self.rng.integers(1e6)),
                                              max_iter=150, learning_rate=0.05,
                                              min_samples_leaf=5, max_depth=3)
            m.fit(Xs[idx], y[idx])
            self.models.append(m)
        nn = NearestNeighbors(n_neighbors=5).fit(Xs)
        d, _ = nn.kneighbors(Xs)
        self._ood_thr = float(np.quantile(d.mean(1), 0.95))
        self._nn = nn
        # алеаторная сигма: out-of-fold RMSE одиночной модели
        oof = np.zeros(n)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(Xs):
            m = HistGradientBoostingRegressor(random_state=0, max_iter=150,
                                              learning_rate=0.05,
                                              min_samples_leaf=5, max_depth=3)
            m.fit(Xs[tr], y[tr])
            oof[te] = m.predict(Xs[te])
        self.sigma_res = float(np.sqrt(np.mean((y - oof) ** 2)))
        self.cv_mae = float(np.mean(np.abs(y - oof)))
        return self

    def predict(self, feats, ensemble):
        X = self._matrix([feats], [ensemble])
        Xs = (X - self._mu) / self._sd
        preds = np.array([m.predict(Xs)[0] for m in self.models])
        pred, s_ep = float(preds.mean()), float(preds.std())
        half = 1.96 * np.hypot(s_ep, self.sigma_res)
        d, _ = self._nn.kneighbors(Xs)
        ood = bool(d.mean(1)[0] > self._ood_thr)
        relw = half / max(abs(pred), 0.5)
        if not ood and relw < 0.5:
            conf = "HIGH"
        elif (not ood and relw >= 0.5) or (ood and relw < 0.5):
            conf = "MEDIUM"
        else:
            conf = "LOW"
        return dict(pred=pred, s_ep=s_ep, ci=(pred - half, pred + half),
                    ood=ood, rel_width=relw, confidence=conf,
                    action=self._recommend(pred, conf, feats))

    @staticmethod
    def _recommend(pred, conf, feats):
        mg = np.exp(feats["log_mingap"])
        if conf == "LOW":
            return ("НЕ ДОВЕРЯТЬ: прогнать детерминированный движок "
                    "lfunc_dynamics.compute_conv(w) и дописать результат в кэш "
                    "(активное обучение).")
        act = ""
        if pred > 5.0:
            act = (f"ДЕФЕКТ: conv>5 при min gap={mg:.4f} — похож на димерную пару; "
                   "исключить окно из статистик / в ловушке: локальное охлаждение пары.")
        elif 2.15 <= pred <= 2.55:
            act = ("Профиль диссипации суррогатно/термального типа (conv~2.3): "
                   "подготовка не арифметически-жёсткая; в ловушке ожидать "
                   "front-loaded сброс энергии.")
        elif 1.4 <= pred < 2.15:
            act = ("Арифметический класс (conv~1.7-1.9): стандартный протокол "
                   "кристаллизации, аномалий не обнаружено.")
        else:
            act = ("Аномально низкая диссипация: проверить окно (перерелаксированное "
                   "начальное состояние); верифицировать ОДУ.")
        if conf == "MEDIUM":
            act += " Дополнительно: короткая верификация ОДУ (s_max=0.5)."
        return act