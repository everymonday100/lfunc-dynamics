#!/usr/bin/env python3
r"""cabinet_bench.py — бенчмарк CoreCabinet на 90 окнах: точность, квадранты,
разделение s_tick по классам маршрутов, throughput против single-shot."""
import json, time
import numpy as np
from pathlib import Path
from scipy.stats import mannwhitneyu

from lfunc_dynamics.cascade import unfold
from lfunc_dynamics.features import window_features
from lfunc_dynamics.ml import ConvSurrogate
from lfunc_dynamics.ml_cabinet import CoreCabinet, make_standardizer

CACHE = Path(__file__).resolve().parent.parent.parent
K, T = 4, 4

def load_window(label, fam):
    if fam == "GL(3)":
        cache = json.load(open(CACHE / "gl3_zeros_cache.json", encoding="utf-8"))
        rec = cache[label]
        zs = np.array(rec["zeros"], float)
        return unfold(zs[50:351], rec["Q"], 3)
    rec = json.load(open(CACHE / "lfunc_zeros" / f"{label}.json", encoding="utf-8"))
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

def main():
    verdicts = json.load(open(CACHE / "lfunc-dynamics" / "ml_verdicts.json",
                              encoding="utf-8"))
    feats, ys, ens, labels, routes = [], [], [], [], []
    for v in verdicts:
        try:
            w = load_window(v["label"], v["fam"])
        except Exception:
            continue
        feats.append(window_features(w))
        ys.append(v["conv"]); ens.append(v["fam"])
        labels.append(v["label"]); routes.append(v["route"])
    y = np.array(ys)
    mg = np.array([np.exp(f["log_mingap"]) for f in feats])
    clean = mg >= 0.05
    print(f"Окон: {len(y)} | чистых: {clean.sum()}")

    fc = [f for f, m in zip(feats, clean) if m]
    ec = [e for e, m in zip(ens, clean) if m]
    yc = y[clean]
    surr = ConvSurrogate(group="G2_phase", n_bag=K * T).fit(fc, yc, ec)
    std_fn = make_standardizer(surr)
    cab = CoreCabinet(surr, K=K, T=T)

    print("Калибровка theta_tick на чистых OOF-окнах...")
    theta = cab.calibrate(fc, ec, std_fn)
    print(f"theta_tick = {theta:.4f}")

    print("\nПрогон кабинета по всем окнам...")
    t0 = time.perf_counter()
    rows = []
    for f, e, lab, rt, yv in zip(feats, ens, labels, routes, ys):
        p = cab.predict(f, e, std_fn)
        rows.append(dict(label=lab, fam=e, route=rt, conv=yv,
                         pred=p.value, ci=list(p.ci), sigma_tick=p.sigma_tick,
                         quadrant=p.quadrant, action=p.action,
                         stale=p.stale_ticks, wall_ms=p.wall_ms))
    wall_cab = time.perf_counter() - t0

    single = []
    t0 = time.perf_counter()
    for f, e in zip(feats, ens):
        X = std_fn(f, e)
        preds = np.array([m.predict(X)[0] for m in surr.models])
        single.append(preds.mean())
    wall_single = time.perf_counter() - t0
    single = np.array(single)

    pred_cab = np.array([r["pred"] for r in rows])
    mae_cab = np.mean(np.abs(pred_cab - y))
    mae_single = np.mean(np.abs(single - y))
    print(f"\nMAE cabinet  = {mae_cab:.4f}")
    print(f"MAE single   = {mae_single:.4f}")
    print(f"throughput: cabinet {len(y)/wall_cab:.1f} win/s | "
          f"single {len(y)/wall_single:.1f} win/s")

    st_clean = [r["sigma_tick"] for r in rows if r["route"] == "model:clean"]
    st_susp = [r["sigma_tick"] for r in rows
               if r["route"].startswith("model:suspect")]
    if st_susp:
        u, p = mannwhitneyu(st_susp, st_clean, alternative="greater")
        print(f"\ns_tick: clean {np.mean(st_clean):.4f} vs suspect "
              f"{np.mean(st_susp):.4f} | Mann-Whitney p={p:.4f}")
        print("Гипотеза: паттерн шума растёт на границе домена "
              "(подтверждена)" if p < 0.05 else
              "Гипотеза: разделение не значимо (паттерн шума однороден)")

    print("\nКвадранты уверенности:")
    for q in ["reliable", "overconfident", "honest_low", "calibration_artifact"]:
        n = sum(1 for r in rows if r["quadrant"] == q)
        print(f"  {q:22s}: {n}")

    stale_all = [r["label"] for r in rows if r["stale"]]
    print(f"\nstale-такты: {len(stale_all)} окон {stale_all[:5]}")

    out = CACHE / "lfunc-dynamics" / "cabinet_verdicts.json"
    json.dump(dict(theta_tick=theta, K=K, T=T, rows=rows),
              open(out, "w", encoding="utf-8"), indent=1)
    print(f"Сохранено: {out.name}")
    cab.close()

if __name__ == "__main__":
    main()