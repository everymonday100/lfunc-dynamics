#!/usr/bin/env python3
r"""
ml_conv_v3.py (fix2) — суррогат conv на СТАНДАРТНОМ горизонте s=2.
ИСПРАВЛЕНИЯ fix2:
 1. Baseline: индекс label->путь строится сканированием lfunc_zeros/*.json
    (стемы файлов не совпадают с rec["label"] -> было 0/30, FileNotFoundError).
 2. Все вердикты (правило димеров + модель + уверенность + действие)
    сохраняются в ml_verdicts.json.
 3. stdout line-buffered: вывод не "зависает" перед долгими секциями.
Сохранено из fix1: conv пересчитывается compute_conv(w, s_max=2.0) и кэшируется
в conv_cache_s2.json; димеры (min gap < 0.02) маршрутизируются правилом;
интерактивный фолбэк в ОДУ (--interactive).
"""
import json, sys, glob
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

from lfunc_dynamics.cascade import unfold
from lfunc_dynamics.features import window_features
from lfunc_dynamics.invariants import compute_conv
from lfunc_dynamics.ml import ConvSurrogate, GROUPS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

CACHE = Path(__file__).resolve().parent.parent.parent
REPO = Path(__file__).resolve().parent.parent
CONV_CACHE = CACHE / "conv_cache_s2.json"
EXTRA_TRAIN = CACHE / "ml_train_extra.json"
VERDICTS = REPO / "ml_verdicts.json"
DIMER_GAP = 0.02
CLEAN_GAP = 0.05

# ---------------------------------------------------------------- загрузка окон
def build_label_index():
    """Индекс rec["label"] -> путь файла для ВСЕХ окон в lfunc_zeros/."""
    idx = {}
    for p in sorted(glob.glob(str(CACHE / "lfunc_zeros" / "*.json"))):
        try:
            rec = json.load(open(p, encoding="utf-8"))
            lab = rec.get("label")
            if lab:
                idx[lab] = Path(p)
        except Exception:
            continue
    return idx

def load_window(label, fam, idx):
    if fam == "GL(3)":
        cache = json.load(open(CACHE / "gl3_zeros_cache.json", encoding="utf-8"))
        rec = cache[label]
        zs = np.array(rec["zeros"], float)
        return unfold(zs[50:351], rec["Q"], 3)
    p = idx.get(label)
    if p is None:
        raise FileNotFoundError(f"window {label} not found in lfunc_zeros index")
    rec = json.load(open(p, encoding="utf-8"))
    gam = np.array(rec["positive_zeros"], float)
    gam = gam[gam > 0.3]
    if label.startswith("artin_"):
        d, n = 2, 301
    else:
        d = 1 if rec["family"] in ("U", "USp") else 2
        n = min(301, len(gam))
    u = unfold(gam, int(rec["conductor"]), d)
    c = len(u) // 2
    w = u[c - n // 2: c - n // 2 + n]
    return w - w[0]

def rigid_surrogate(N, s, rng):
    M = N - 1
    kap = 2 * np.pi * np.fft.rfftfreq(M)
    amp = np.sqrt(kap); amp[0] = 0.0
    re, im = rng.standard_normal(M // 2 + 1), rng.standard_normal(M // 2 + 1)
    if M % 2 == 0:
        im[-1] = 0.0
    eps = np.fft.irfft(amp * (re + 1j * im), M)
    eps *= s / eps.std()
    g = np.clip(1.0 + eps, 0.1, None)
    return np.concatenate([[0.0], np.cumsum(g)])

# ---------------------------------------------------------------- правила и датасет
def rule_gate(f):
    mg = np.exp(f["log_mingap"])
    if mg < DIMER_GAP:
        return (f"DIMER (min gap={mg:.4f} < {DIMER_GAP}): exclude window from "
                f"statistics; in trap: local cooling of the pair. No ODE needed.")
    return None

def build_dataset():
    print("Сбор датасета (conv пересчитывается на s=2 и кэшируется)...")
    idx = build_label_index()
    print(f"  индекс lfunc_zeros: {len(idx)} меток")
    inv = json.load(open(CACHE / "next_invariants_results.json", encoding="utf-8"))
    ccache = json.load(open(CONV_CACHE, encoding="utf-8")) if CONV_CACHE.exists() else {}
    feats, ys, ens, labels, wins = [], [], [], [], []
    n_err, first_err = 0, None
    for fam, rows in inv.items():
        ok = 0
        for r in rows:
            label = r.get("label")
            if not label:
                continue
            try:
                w = load_window(label, fam, idx)
            except Exception as e:
                n_err += 1
                if first_err is None:
                    first_err = f"{label}: {e!r}"
                continue
            if label not in ccache:
                ccache[label] = float(compute_conv(w, s_max=2.0))
            feats.append(window_features(w))
            ys.append(ccache[label])
            ens.append(fam); labels.append(label); wins.append(w)
            ok += 1
        print(f"  {fam}: загружено {ok}/{len(rows)} окон")
    if first_err:
        print(f"  первая ошибка загрузки: {first_err}")
    if n_err:
        print(f"  всего ошибок загрузки: {n_err}")
    json.dump(ccache, open(CONV_CACHE, "w", encoding="utf-8"), indent=1)
    print(f"  кэш conv сохранён: {CONV_CACHE.name} ({len(ccache)} записей)")
    return feats, ys, ens, labels, wins

# ---------------------------------------------------------------- абляция
def ablation(X_all, y):
    print("\n=== АБЛЯЦИЯ (только чистые окна, 5-fold CV x4) ===")
    for g in GROUPS:
        ml_, rl_, mr_, rr_ = [], [], [], []
        for seed in range(4):
            ol = np.zeros(len(y)); orf = np.zeros(len(y))
            for tr, te in KFold(5, shuffle=True, random_state=seed).split(X_all[g]):
                m = Ridge(alpha=1.0); m.fit(X_all[g][tr], y[tr])
                ol[te] = m.predict(X_all[g][te])
                m = RandomForestRegressor(n_estimators=50, max_depth=3,
                                          min_samples_leaf=5, random_state=seed)
                m.fit(X_all[g][tr], y[tr])
                orf[te] = m.predict(X_all[g][te])
            ml_.append(mean_absolute_error(y, ol)); rl_.append(r2_score(y, ol))
            mr_.append(mean_absolute_error(y, orf)); rr_.append(r2_score(y, orf))
        print(f"{g:12s}: Ridge MAE={np.mean(ml_):.3f} R2={np.mean(rl_):+.3f} | "
              f"RF MAE={np.mean(mr_):.3f} R2={np.mean(rr_):+.3f}")

# ---------------------------------------------------------------- вердикты
def save_verdicts(model, feats, ys, ens, labels):
    verdicts = []
    for f, yv, fam, lab in zip(feats, ys, ens, labels):
        mg = float(np.exp(f["log_mingap"]))
        gate = rule_gate(f)
        if gate:
            verdicts.append(dict(label=lab, fam=fam, min_gap=mg, conv=yv,
                                 route="rule:dimer", action=gate))
        else:
            res = model.predict(f, fam)
            zone = "clean" if mg >= CLEAN_GAP else "suspect(0.02-0.05)"
            verdicts.append(dict(label=lab, fam=fam, min_gap=mg, conv=yv,
                                 route="model:" + zone,
                                 pred=res["pred"], ci=list(res["ci"]),
                                 confidence=res["confidence"], ood=res["ood"],
                                 action=res["action"]))
    json.dump(verdicts, open(VERDICTS, "w", encoding="utf-8"), indent=2)
    print(f"\nВсе вердикты сохранены: {VERDICTS.name} ({len(verdicts)} записей)")

# ---------------------------------------------------------------- интерактив
def interactive_menu(model, feats, ys, ens, labels, wins):
    print("\n" + "=" * 70)
    print("ИНТЕРАКТИВНЫЙ РЕЖИМ: фолбэк в ОДУ (y/n/a/A)")
    print("=" * 70)
    extra = json.load(open(EXTRA_TRAIN, encoding="utf-8")) if EXTRA_TRAIN.exists() else []
    yes_all = no_all = False
    n_yes = n_no = 0
    for i, (f, fam, label) in enumerate(zip(feats, ens, labels)):
        if rule_gate(f):
            continue
        res = model.predict(f, fam)
        if not (res["confidence"] == "LOW" or res["ood"]):
            continue
        if yes_all:
            choice = "y"
        elif no_all:
            choice = "n"
        else:
            print(f"\n[{label}] ({fam}) conf={res['confidence']} OOD={res['ood']} "
                  f"pred={res['pred']:.3f} CI=[{res['ci'][0]:.3f}, {res['ci'][1]:.3f}]")
            choice = input("  Передать в ОДУ? y/n/a(all yes)/A(all no): ").strip().lower()
        if choice in ("y", "a"):
            if choice == "a":
                yes_all = True
            true = compute_conv(wins[i], s_max=2.0)
            half = (res["ci"][1] - res["ci"][0]) / 2
            verdict = ("вне CI модели -> профиль НЕ арифметический "
                       "(суррогат/термальный) или дефект"
                       if abs(true - res["pred"]) > 2 * half else
                       "согласуется с арифметическим классом")
            print(f"  ОДУ: conv={true:.3f} | вердикт: {verdict}")
            extra.append({"label": label, "conv": float(true)})
            n_yes += 1
        else:
            if choice == "A":
                no_all = True
            n_no += 1
    json.dump(extra, open(EXTRA_TRAIN, "w", encoding="utf-8"), indent=1)
    print(f"\nИТОГО: в ОДУ передано {n_yes}, пропущено {n_no}; "
          f"дообучение сохранено в {EXTRA_TRAIN.name}")

# ---------------------------------------------------------------- main
def main():
    interactive = "--interactive" in sys.argv
    feats, ys, ens, labels, wins = build_dataset()
    y = np.array(ys)
    mg = np.array([np.exp(f["log_mingap"]) for f in feats])

    clean = mg >= CLEAN_GAP
    dimer = mg < DIMER_GAP
    print(f"\nВсего окон: {len(y)} | чистых: {clean.sum()} | димеров: {dimer.sum()}")
    print(f"conv чистых: mean={y[clean].mean():.3f}, std={y[clean].std():.3f}, "
          f"median={np.median(y[clean]):.3f}")

    fc = [f for f, m in zip(feats, clean) if m]
    yc = y[clean]; ec = [e for e, m in zip(ens, clean) if m]
    X_all = {g: np.array([[f[c] for c in GROUPS[g]] for f in fc]) for g in GROUPS}
    ablation(X_all, yc)

    print("\n=== ФИНАЛЬНАЯ МОДЕЛЬ (G2, чистые окна) ===")
    model = ConvSurrogate(group="G2_phase", n_bag=16).fit(fc, yc, ec)
    print(f"CV MAE={model.cv_mae:.3f}, sigma_res={model.sigma_res:.3f}")

    rng = np.random.default_rng(7)
    print("\n=== ДЕМОНСТРАЦИОННЫЕ КАРТОЧКИ ===")
    for fam in ["baseline", "Artin", "GL(3)"]:
        idxs = [k for k in range(len(labels)) if ens[k] == fam and clean[k]]
        if not idxs:
            print(f"⚠️ {fam}: нет чистых окон в датасете")
            continue
        k = idxs[0]
        res = model.predict(feats[k], fam)
        print(f"\n[{labels[k]}] ({fam}) min gap={mg[k]:.4f}")
        print(f"  conv: истинное={ys[k]:.2f} | предсказано={res['pred']:.2f} "
              f"CI95=[{res['ci'][0]:.2f}, {res['ci'][1]:.2f}]")
        print(f"  уверенность={res['confidence']} OOD={res['ood']}")
        print(f"  ДЕЙСТВИЕ: {res['action']}")

    di = [k for k in range(len(labels)) if dimer[k]]
    if di:
        k = di[0]
        print(f"\n[{labels[k]}] ({ens[k]}) min gap={mg[k]:.4f}  (правило, не модель)")
        print(f"  conv истинное={ys[k]:.2f}")
        print(f"  ДЕЙСТВИЕ: {rule_gate(feats[k])}")

    w_s = rigid_surrogate(301, 0.37, rng)
    f_s = window_features(w_s)
    true_s = compute_conv(w_s, s_max=2.0)
    res = model.predict(f_s, "synthetic")
    print(f"\n[synthetic rigid surrogate] min gap={np.exp(f_s['log_mingap']):.4f}")
    print(f"  conv: истинное(ОДУ)={true_s:.2f} | предсказано={res['pred']:.2f} "
          f"CI95=[{res['ci'][0]:.2f}, {res['ci'][1]:.2f}]")
    print(f"  уверенность={res['confidence']} OOD={res['ood']}")
    half = (res["ci"][1] - res["ci"][0]) / 2
    print(f"  ВЕРДИКТ: {'измерение вне CI -> не арифметический класс' if abs(true_s - res['pred']) > 2*half else 'в пределах CI'}")
    print(f"  ДЕЙСТВИЕ: {res['action']}")

    save_verdicts(model, feats, ys, ens, labels)

    if interactive:
        interactive_menu(model, feats, ys, ens, labels, wins)

if __name__ == "__main__":
    main()