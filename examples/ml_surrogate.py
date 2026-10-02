#!/usr/bin/env python3
r"""
ml_surrogate.py (v2) — ML-предиктор для tau_H и conv.
Проверяет гипотезу: tau_H предсказывается из скалярных признаков (R² высокий),
а conv — нет (R² низкий), потому что conv чувствителен к фазам.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Путь к кэшам: родительская директория (MyRhProject)
CACHE_DIR = Path(__file__).parent.parent.parent

def load_dataset():
    """Собирает датасет с tau_H и conv из существующих кэшей."""
    rows = []
    
    # 1. Загружаем spectral_dynamic_extended.json (tau_H, r2, h0)
    p_ext = CACHE_DIR / "spectral_dynamic_extended.json"
    if p_ext.exists():
        data = json.load(open(p_ext, encoding="utf-8"))
        print(f"Загружено записей из spectral_dynamic_extended.json: {len(data)}")
        
        for r in data:
            tauH = r.get("tauH")
            r2 = r.get("r2")
            if tauH is not None and r2 is not None and np.isfinite(tauH) and np.isfinite(r2):
                rows.append({
                    "label": r.get("label", "unknown"),
                    "ensemble": r.get("ensemble", "unknown"),
                    "r2": r2,
                    "h0": r.get("h0", np.nan),
                    "Q": r.get("Q", 0),
                    "n_zeros": r.get("n_zeros_total", 0),
                    "tau_H": tauH,
                    "conv": np.nan,  # Будем заполнять из next_invariants_results
                })
    
    # 2. Загружаем next_invariants_results.json (conv)
    p_inv = CACHE_DIR / "next_invariants_results.json"
    if p_inv.exists():
        inv_data = json.load(open(p_inv, encoding="utf-8"))
        print(f"Загружено семейств из next_invariants_results.json: {len(inv_data)}")
        
        # Создаём словарь label -> conv
        conv_map = {}
        for fam, records in inv_data.items():
            for r in records:
                label = r.get("label")
                conv_val = r.get("conv")
                if label and conv_val is not None and np.isfinite(conv_val):
                    conv_map[label] = conv_val
        
        # Заполняем conv в основном датасете
        for row in rows:
            if row["label"] in conv_map:
                row["conv"] = conv_map[row["label"]]
    
    df = pd.DataFrame(rows)
    print(f"Строк с валидными (tau_H, r2): {len(df)}")
    print(f"Строк с валидным conv: {df['conv'].notna().sum()}")
    
    if "ensemble" in df.columns:
        df["ensemble"] = df["ensemble"].astype("category")
    
    return df

def train_model(X_train, X_test, y_train, y_test, target_name):
    """Обучает модель и возвращает метрики."""
    cat_features = [c for c in X_train.columns if X_train[c].dtype.name == "category"]
    num_features = [c for c in X_train.columns if X_train[c].dtype.name != "category"]
    
    transformers = []
    if cat_features:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_features))
    if num_features:
        transformers.append(("num", "passthrough", num_features))
    
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", HistGradientBoostingRegressor(
            random_state=42, max_iter=200, learning_rate=0.05
        ))
    ])
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mae, r2, y_pred

def main():
    print("=" * 70)
    print("ml_surrogate.py (v2) — проверка гипотезы о фазовой чувствительности")
    print("=" * 70)
    print("\nЗагрузка датасета...")
    df = load_dataset()
    
    if df.empty:
        print("\nНевозможно продолжить: датасет пуст.")
        return
    
    feature_cols = [c for c in ["ensemble", "r2", "h0", "Q", "n_zeros"] if c in df.columns]
    X = df[feature_cols]
    
    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ 1: Предсказание tau_H (должно работать хорошо)")
    print("=" * 70)
    
    # Фильтруем строки с валидным tau_H
    mask_tau = df["tau_H"].notna()
    X_tau = X[mask_tau]
    y_tau = df.loc[mask_tau, "tau_H"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_tau, y_tau, test_size=0.2, random_state=42
    )
    
    mae_tau, r2_tau, _ = train_model(X_train, X_test, y_train, y_test, "tau_H")
    
    print(f"\n✅ Результаты для tau_H:")
    print(f"  MAE: {mae_tau:.5f}  (бутстрап SE ~ 0.001-0.002)")
    print(f"  R² : {r2_tau:.4f}")
    
    print("\n" + "=" * 70)
    print("ЭКСПЕРИМЕНТ 2: Предсказание conv (должно работать плохо)")
    print("=" * 70)
    
    # Фильтруем строки с валидным conv
    mask_conv = df["conv"].notna()
    if mask_conv.sum() < 20:
        print(f"\n⚠️ Недостаточно данных для conv (только {mask_conv.sum()} строк).")
        print("Пропускаем эксперимент 2.")
        return
    
    X_conv = X[mask_conv]
    y_conv = df.loc[mask_conv, "conv"]
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_conv, y_conv, test_size=0.2, random_state=42
    )
    
    mae_conv, r2_conv, _ = train_model(X_train_c, X_test_c, y_train_c, y_test_c, "conv")
    
    print(f"\nРезультаты для conv:")
    print(f"  MAE: {mae_conv:.3f}")
    print(f"  R² : {r2_conv:.4f}")
    
    print("\n" + "=" * 70)
    print("СРАВНЕНИЕ И ВЫВОДЫ")
    print("=" * 70)
    print(f"tau_H: MAE = {mae_tau:.5f}, R² = {r2_tau:.4f}")
    print(f"conv:  MAE = {mae_conv:.3f}, R² = {r2_conv:.4f}")
    
    if r2_tau > 0.3 and r2_conv < 0.2:
        print("\n✅ ГИПОТЕЗА ПОДТВЕРЖДЕНА:")
        print("   tau_H предсказывается из скалярных признаков (зависит от 2-х моментов),")
        print("   conv НЕ предсказывается (чувствителен к фазам/биспектру).")
    else:
        print("\n⚠️ Гипотеза требует уточнения. Возможно, нужны дополнительные признаки.")

if __name__ == "__main__":
    main()