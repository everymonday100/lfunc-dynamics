#!/usr/bin/env python3
r"""
pra_figures.py (v4) — три фигуры для Physical Review Applied.
v4: все тексты Fig. 3 на английском (через ml.ConvSurrogate._recommend);
    относительный путь pra_figures/; suspect-точки красные.
"""
import json, glob, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

from lfunc_dynamics.flow import integrate_dyson, coulomb_energy
from lfunc_dynamics.cascade import unfold
from lfunc_dynamics.features import window_features
from lfunc_dynamics.ml import ConvSurrogate

CACHE = Path(__file__).resolve().parent.parent.parent
FIG_DIR = Path("pra_figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "mathtext.fontset": "dejavuserif",
})

COLORS = {"baseline": "0.55", "Artin": "tab:purple", "GL(3)": "tab:red"}
DIMER_LABELS = {"27a1", "32a1", "49a1", "64a1", "artin_2.257.3t2.a.a"}

def load_window(label, fam):
    if fam == "GL(3)":
        cache = json.load(open(CACHE / "gl3_zeros_cache.json", encoding="utf-8"))
        rec = cache[label]
        zs = np.array(rec["zeros"], float)
        return unfold(zs[50:351], rec["Q"], 3)
    for cand in sorted(glob.glob(str(CACHE / "lfunc_zeros" / "*.json"))):
        rec = json.load(open(cand, encoding="utf-8"))
        if rec.get("label") == label:
            break
    else:
        raise FileNotFoundError(f"Window {label} not found")
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

def join_tau():
    ext = json.load(open(CACHE / "spectral_dynamic_extended.json", encoding="utf-8"))
    tau12 = {}
    for path, fam in [("nlevel_scan.json", "baseline"),
                      ("artin_flow_scan.json", "Artin"),
                      ("gl3_scan.json", "GL(3)")]:
        pp = CACHE / path
        if not pp.exists():
            continue
        data = json.load(open(pp, encoding="utf-8"))
        rows = data.get("rows", data) if isinstance(data, dict) else data
        for r in rows:
            lab = r.get("label")
            if lab is None:
                continue
            t = r["1"].get("tau12") if (fam == "baseline" and "1" in r) \
                else (r.get("tau12_meas") or r.get("tau_meas"))
            if t is not None and np.isfinite(t):
                tau12[lab] = (fam, t)
    xs, ys, fams, labs = [], [], [], []
    for r in ext:
        lab = r["label"]
        if lab in tau12 and r["ensemble"] == tau12[lab][0] and np.isfinite(r["tauH"]):
            xs.append(tau12[lab][1]); ys.append(r["tauH"])
            fams.append(r["ensemble"]); labs.append(lab)
    return np.array(xs), np.array(ys), fams, labs

def fig1():
    print("Fig. 1: Master curve + two-channel independence")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    for fam, label in [("baseline", "2-11-1.1-c1-0-0"),
                       ("Artin", "artin_2.104.3t2.b.a"),
                       ("GL(3)", "100a1")]:
        w = load_window(label, fam)
        t, y = integrate_dyson(w, s_max=2.0, n_steps=100)
        H = np.array([coulomb_energy(y[:, i]) for i in range(len(t))])
        ax.plot(t, (H - H[-1]) / (H[0] - H[-1]), color=COLORS[fam], lw=2.2, label=fam)
    ax.axhline(0.5, color="k", ls="--", lw=1, alpha=0.6, label=r"$\tilde{H}=0.5$")
    ax.axvline(0.94, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("Dyson time $s$")
    ax.set_ylabel(r"$\tilde{H}(s)=(H-H_\infty)/(H_0-H_\infty)$")
    ax.set_title("(a) Universal energy relaxation")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)

    ax = axes[1]
    xs, ys, fams, labs = join_tau()
    dim = np.array([l in DIMER_LABELS for l in labs])
    m_nd = ~dim
    for fam in ["baseline", "Artin", "GL(3)"]:
        m = np.array([f == fam for f in fams]) & m_nd
        if m.any():
            ax.scatter(xs[m], ys[m], c=COLORS[fam], s=35, alpha=0.7,
                       edgecolors="k", linewidths=0.5, label=fam)
    if dim.any():
        ax.scatter(xs[dim], ys[dim], facecolors="none", edgecolors="k",
                   s=90, linewidths=1.5, label="dimer defects")
    s, ic = np.polyfit(xs[m_nd], ys[m_nd], 1)
    xl = np.linspace(xs.min(), xs.max(), 50)
    ax.plot(xl, s * xl + ic, "k--", lw=1.5, alpha=0.7, label="fit (no dimers)")
    r_all = np.corrcoef(xs, ys)[0, 1]
    r_nd = np.corrcoef(xs[m_nd], ys[m_nd])[0, 1]
    ax.set_title("(b) Two-channel independence\n$r=%+.3f$ (all), $r=%+.3f$ (no dimers)"
                 % (r_all, r_nd))
    ax.set_xlabel(r"$\tau_{1/2}^{(n=1)}$ (gap onset, pipeline units)")
    ax.set_ylabel(r"$\tau_H$ (energy half-time, pipeline units)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pra_fig1_masterH_indep.png")
    print("  saved: pra_fig1_masterH_indep.png")
    plt.close(fig)

def fig2():
    print("Fig. 2: ML predictor accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    ext = json.load(open(CACHE / "spectral_dynamic_extended.json", encoding="utf-8"))
    rows = [{"ensemble": r.get("ensemble", "unknown"), "r2": r["r2"],
             "h0": r.get("h0", np.nan), "Q": r.get("Q", 0),
             "n_zeros": r.get("n_zeros_total", 0), "tau_H": r["tauH"]}
            for r in ext if np.isfinite(r.get("tauH", np.nan))
            and np.isfinite(r.get("r2", np.nan))]
    df = pd.DataFrame(rows)
    df["ensemble"] = df["ensemble"].astype("category")
    X, y = df[["ensemble", "r2", "h0", "Q", "n_zeros"]], df["tau_H"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    model = Pipeline([("prep", ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["ensemble"]),
         ("num", "passthrough", ["r2", "h0", "Q", "n_zeros"])])),
        ("reg", HistGradientBoostingRegressor(random_state=42, max_iter=200,
                                              learning_rate=0.05))])
    model.fit(Xtr, ytr)
    yp = model.predict(Xte)
    mae = mean_absolute_error(yte, yp)
    ax.scatter(yte, yp, c="tab:blue", s=30, alpha=0.6, edgecolors="k", linewidths=0.3)
    lims = [min(yte.min(), yp.min()), max(yte.max(), yp.max())]
    ax.plot(lims, lims, "k--", lw=1.5, alpha=0.7, label="perfect prediction")
    ax.set_xlabel(r"True $\tau_H$"); ax.set_ylabel(r"Predicted $\tau_H$")
    ax.set_title("(a) $\\tau_H$ predictor (MAE = %.5f)" % mae)
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    verdicts = json.load(open(CACHE / "lfunc-dynamics" / "ml_verdicts.json",
                              encoding="utf-8"))
    tc_clean, pc_clean, el_clean, eh_clean = [], [], [], []
    tc_susp, pc_susp, el_susp, eh_susp = [], [], [], []
    for v in verdicts:
        if v["route"] == "model:clean":
            tc_clean.append(v["conv"]); pc_clean.append(v["pred"])
            el_clean.append(v["pred"] - v["ci"][0]); eh_clean.append(v["ci"][1] - v["pred"])
        elif v["route"] == "model:suspect(0.02-0.05)":
            tc_susp.append(v["conv"]); pc_susp.append(v["pred"])
            el_susp.append(v["pred"] - v["ci"][0]); eh_susp.append(v["ci"][1] - v["pred"])
    tc_clean, pc_clean, el_clean, eh_clean = map(np.array, (tc_clean, pc_clean, el_clean, eh_clean))
    tc_susp, pc_susp, el_susp, eh_susp = map(np.array, (tc_susp, pc_susp, el_susp, eh_susp))
    if len(tc_clean) > 0:
        ax.errorbar(tc_clean, pc_clean, yerr=[el_clean, eh_clean], fmt="o",
                    color="tab:orange", markersize=4, alpha=0.6, ecolor="tab:orange",
                    elinewidth=1, capsize=2, capthick=0.5,
                    label="clean ($g_{\\min}\\geq 0.05$)")
    if len(tc_susp) > 0:
        ax.errorbar(tc_susp, pc_susp, yerr=[el_susp, eh_susp], fmt="o",
                    color="tab:red", markersize=6, alpha=0.9, ecolor="tab:red",
                    elinewidth=1.5, capsize=2, capthick=1,
                    label="suspect ($0.02<g_{\\min}<0.05$)")
    allv = np.concatenate([tc_clean, tc_susp, pc_clean, pc_susp])
    lims = [min(allv.min() - 0.5, 0.0), allv.max() + 0.5]
    ax.plot(lims, lims, "k--", lw=1.5, alpha=0.7, label="perfect prediction")
    mae_c = mean_absolute_error(np.concatenate([tc_clean, tc_susp]),
                                np.concatenate([pc_clean, pc_susp]))
    ax.set_xlabel("True conv"); ax.set_ylabel("Predicted conv")
    ax.set_title("(b) conv predictor (MAE = %.3f)" % mae_c)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pra_fig2_ml_accuracy.png")
    print("  saved: pra_fig2_ml_accuracy.png")
    plt.close(fig)

def _chevron(ax, x, y, w, h, tip, fc, ec, lw=1.6):
    pts = [(x, y), (x + w - tip, y), (x + w, y + h / 2.0),
           (x + w - tip, y + h), (x, y + h)]
    ax.add_patch(plt.Polygon(pts, closed=True, facecolor=fc,
                             edgecolor=ec, linewidth=lw, zorder=2))

def fig3():
    print("Fig. 3: Detection pipeline (English)")
    verdicts = json.load(open(CACHE / "lfunc-dynamics" / "ml_verdicts.json",
                              encoding="utf-8"))
    dimer_ex = next((v for v in verdicts if v["route"] == "rule:dimer"), None)
    ood_ex = next((v for v in verdicts if v.get("ood")), None)
    clean_ex = next((v for v in verdicts
                     if v["route"] == "model:clean" and v["confidence"] == "HIGH"), None)
    rows = [("Clean window", clean_ex, "tab:green"),
            ("Dimer defect", dimer_ex, "tab:red"),
            ("OOD / surrogate", ood_ex, "tab:orange")]

    fig, ax = plt.subplots(figsize=(12, 7.2))
    ax.axis("off")
    W1, W2, W3 = 3.5, 3.5, 3.9
    X1, X2, X3 = 0.3, 4.3, 8.3
    TIP, H = 0.4, 1.55
    for i, (title, ex, color) in enumerate(rows):
        if ex is None:
            continue
        y = 5.35 - i * 2.35
        ax.text(X1, y + H + 0.18, title, fontsize=12, weight="bold", color=color)
        _chevron(ax, X1, y, W1, H, TIP, "lightgray", "black")
        ax.text(X1 + 0.18, y + H - 0.22, ex["label"], fontsize=9.5,
                weight="bold", va="top")
        ax.text(X1 + 0.18, y + H - 0.62, "(%s)" % ex["fam"], fontsize=9,
                weight="bold", va="top")
        ax.text(X1 + 0.18, y + H - 1.02, "min gap = %.4f" % ex["min_gap"],
                fontsize=9, va="top")
        if ex["route"] == "rule:dimer":
            head = "RULE\n(dimer detected)"
            params = "min gap %.4f < 0.02" % ex["min_gap"]
            fc = "lightcoral"
        else:
            head = "MODEL conf=%s\nOOD=%s" % (ex["confidence"], ex["ood"])
            params = "pred=%.2f, CI=[%.2f, %.2f]" % (ex["pred"], ex["ci"][0], ex["ci"][1])
            fc = "lightblue" if ex["confidence"] == "HIGH" else "lightyellow"
        _chevron(ax, X2, y, W2, H, TIP, fc, "black")
        ax.text(X2 + 0.18, y + H - 0.22, head, fontsize=9.5, weight="bold", va="top")
        ax.text(X2 + 0.18, y + 0.35, params, fontsize=8, va="top")
        _chevron(ax, X3, y, W3, H, TIP, "white", color, lw=2.2)
        ax.text(X3 + 0.18, y + H - 0.15, "ACTION", fontsize=10,
                weight="bold", color=color, va="top")
        all_lines = textwrap.fill(ex["action"], width=46).split("\n")
        lines = all_lines[:5]
        if len(all_lines) > 5:
            lines[-1] = lines[-1][:43] + "..."
        ax.text(X3 + 0.18, y + H - 0.45, "\n".join(lines), fontsize=7.2,
                va="top", linespacing=0.95)
    ax.set_xlim(0, 12.4)
    ax.set_ylim(0.6, 7.6)
    ax.set_title("Defect detection pipeline: rule → model → action",
                 fontsize=14, weight="bold", pad=18)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pra_fig3_detection_pipeline.png")
    print("  saved: pra_fig3_detection_pipeline.png")
    plt.close(fig)

def main():
    print("=" * 70)
    print("pra_figures.py (v4, English) — PRA figures")
    print("=" * 70)
    fig1(); fig2(); fig3()
    print("\nDone: pra_figures/pra_fig{1,2,3}_*.png")

if __name__ == "__main__":
    main()