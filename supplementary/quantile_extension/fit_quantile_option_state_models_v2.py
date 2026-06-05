"""
10a_fit_quantile_option_state_models_v2.py

V2 fixes after pilot diagnostics:
1. Add Greeks to spread model features:
   OPTION_DELTA, OPTION_GAMMA, OPTION_VEGA, OPTION_THETA.
2. Save median-model residuals for controlled residual noise in simulation.
3. Save configuration values for quantile widening and clipping.

Run:
    py src\10a_fit_quantile_option_state_models_v2.py --mode pilot
    py src\10a_fit_quantile_option_state_models_v2.py --mode final
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_pinball_loss,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
OUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = OUT_DIR / "quantile_option_state_models_v2"
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

REAL_PROXY_FILE = DATA_DIR / "transitions_daily_top1_final_with_spy_2010_2023_with_regime_proxies.parquet"
REAL_BASE_FILE = DATA_DIR / "transitions_daily_top1_final_with_spy_2010_2023.parquet"
HMM_NPZ_FILE = OUT_DIR / "hmm_regime_params_pilot.npz"

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
SPREAD_EPS = 1e-6
TRADING_DAYS = 252

# Pilot-2 defaults based on the first pilot diagnosis.
DEFAULT_IV_WIDEN_SCALE = 1.25
DEFAULT_SPREAD_WIDEN_SCALE = 1.75
DEFAULT_IV_RESIDUAL_SCALE = 0.35
DEFAULT_SPREAD_RESIDUAL_SCALE = 0.50


def load_real_data() -> pd.DataFrame:
    path = REAL_PROXY_FILE if REAL_PROXY_FILE.exists() else REAL_BASE_FILE
    if not path.exists():
        raise FileNotFoundError(f"Cannot find real transition file:\n{REAL_PROXY_FILE}\n{REAL_BASE_FILE}")
    print(f"Loading real data: {path}")
    return pd.read_parquet(path)


def gaussian_pdf_1d(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    x = x.reshape(-1, 1)
    means = means.reshape(1, -1)
    variances = np.maximum(variances.reshape(1, -1), 1e-12)
    pdf = np.exp(-0.5 * (x - means) ** 2 / variances) / np.sqrt(2.0 * np.pi * variances)
    return np.maximum(pdf, 1e-300)


def viterbi(x: np.ndarray, pi: np.ndarray, A: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    T = len(x)
    K = len(pi)
    B = gaussian_pdf_1d(x, means, variances)
    logB = np.log(B)
    logA = np.log(np.maximum(A, 1e-300))
    logpi = np.log(np.maximum(pi, 1e-300))

    delta = np.zeros((T, K))
    psi = np.zeros((T, K), dtype=int)
    delta[0] = logpi + logB[0]

    for t in range(1, T):
        vals = delta[t - 1][:, None] + logA
        psi[t] = np.argmax(vals, axis=0)
        delta[t] = np.max(vals, axis=0) + logB[t]

    states = np.zeros(T, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def add_regime_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "REGIME_ID" in out.columns:
        out["REGIME_ID"] = pd.to_numeric(out["REGIME_ID"], errors="coerce").fillna(1).astype(int)
        return out

    if not HMM_NPZ_FILE.exists():
        print("HMM params not found; fallback REGIME_ID=1.")
        out["REGIME_ID"] = 1
        return out

    hmm = np.load(HMM_NPZ_FILE, allow_pickle=True)
    pi, A, means, variances = hmm["pi"], hmm["A"], hmm["means"], hmm["variances"]

    daily = out[["QUOTE_DATE", "SPY_CLOSE"]].drop_duplicates("QUOTE_DATE").sort_values("QUOTE_DATE")
    daily["LOG_RET"] = np.log(daily["SPY_CLOSE"].astype(float) / daily["SPY_CLOSE"].astype(float).shift(1))
    valid = daily.dropna(subset=["LOG_RET"]).copy()

    states = viterbi(valid["LOG_RET"].values.astype(float), pi, A, means, variances)
    valid["REGIME_ID"] = states

    daily = daily.merge(valid[["QUOTE_DATE", "REGIME_ID"]], on="QUOTE_DATE", how="left")
    daily["REGIME_ID"] = daily["REGIME_ID"].ffill().bfill().fillna(1).astype(int)
    out = out.merge(daily[["QUOTE_DATE", "REGIME_ID"]], on="QUOTE_DATE", how="left")
    out["REGIME_ID"] = out["REGIME_ID"].fillna(1).astype(int)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["QUOTE_DATE", "EPISODE_ID"]).reset_index(drop=True)

    if "SPY_MONEYNESS" not in out.columns:
        out["SPY_MONEYNESS"] = out["SPY_CLOSE"].astype(float) / out["STRIKE"].astype(float)
    if "SPY_LOG_MONEYNESS" not in out.columns:
        out["SPY_LOG_MONEYNESS"] = np.log(out["SPY_MONEYNESS"].astype(float))
    if "OPTION_MID_OVER_SPY" not in out.columns:
        out["OPTION_MID_OVER_SPY"] = out["OPTION_MID"].astype(float) / out["SPY_CLOSE"].astype(float)
    if "SPY_RET_LAG1" not in out.columns:
        out["SPY_RET_LAG1"] = out.groupby("EPISODE_ID")["SPY_CLOSE"].pct_change().fillna(0.0)

    # Daily rolling return proxies if missing.
    missing_daily = any(c not in out.columns for c in ["SPY_LOG_RET_DAILY", "ROLLING_RV_20D", "ROLLING_ABS_RET_10D"])
    if missing_daily:
        daily = out[["QUOTE_DATE", "SPY_CLOSE"]].drop_duplicates("QUOTE_DATE").sort_values("QUOTE_DATE")
        daily["SPY_LOG_RET_DAILY"] = np.log(daily["SPY_CLOSE"].astype(float) / daily["SPY_CLOSE"].astype(float).shift(1))
        daily["ROLLING_RV_10D"] = daily["SPY_LOG_RET_DAILY"].rolling(10, min_periods=3).std() * math.sqrt(TRADING_DAYS)
        daily["ROLLING_RV_20D"] = daily["SPY_LOG_RET_DAILY"].rolling(20, min_periods=5).std() * math.sqrt(TRADING_DAYS)
        daily["ROLLING_ABS_RET_10D"] = daily["SPY_LOG_RET_DAILY"].abs().rolling(10, min_periods=3).mean()
        daily = daily.fillna(method="bfill").fillna(0.0)
        merge_cols = ["QUOTE_DATE", "SPY_LOG_RET_DAILY", "ROLLING_RV_10D", "ROLLING_RV_20D", "ROLLING_ABS_RET_10D"]
        out = out.drop(columns=[c for c in merge_cols if c in out.columns and c != "QUOTE_DATE"], errors="ignore")
        out = out.merge(daily[merge_cols], on="QUOTE_DATE", how="left")

    out["IV_LEVEL"] = pd.to_numeric(out.get("OPTION_IV", np.nan), errors="coerce")
    out.loc[out["IV_LEVEL"] > 3.0, "IV_LEVEL"] = out.loc[out["IV_LEVEL"] > 3.0, "IV_LEVEL"] / 100.0

    if "OPTION_SPREAD_PCT" in out.columns:
        out["SPREAD_PROXY"] = pd.to_numeric(out["OPTION_SPREAD_PCT"], errors="coerce")
    elif "SPREAD_PROXY" not in out.columns:
        out["SPREAD_PROXY"] = np.nan

    out["LOG_SPREAD_PROXY"] = np.log(np.maximum(pd.to_numeric(out["SPREAD_PROXY"], errors="coerce"), 0.0) + SPREAD_EPS)

    required_numeric = [
        "REGIME_ID", "DTE", "SPY_LOG_MONEYNESS", "SPY_MONEYNESS", "ROLLING_RV_20D",
        "ROLLING_ABS_RET_10D", "SPY_RET_LAG1", "IV_LEVEL", "OPTION_MID_OVER_SPY",
        "SPREAD_PROXY", "LOG_SPREAD_PROXY", "OPTION_DELTA", "OPTION_GAMMA",
        "OPTION_VEGA", "OPTION_THETA"
    ]

    for c in required_numeric:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return out


def make_models(mode: str):
    df = add_features(add_regime_labels(load_real_data()))

    train = df[df["SPLIT"].astype(str).str.lower().eq("train")].copy()
    val = df[df["SPLIT"].astype(str).str.lower().isin(["val", "valid", "validation"])].copy()
    if val.empty:
        print("Validation split not found. Using test split for diagnostics only.")
        val = df[df["SPLIT"].astype(str).str.lower().eq("test")].copy()
    if val.empty:
        val = train.sample(frac=0.2, random_state=42)
        train = train.drop(index=val.index)

    if mode == "pilot":
        max_train = min(len(train), 60000)
        max_val = min(len(val), 25000)
        train = train.sample(n=max_train, random_state=42) if len(train) > max_train else train
        val = val.sample(n=max_val, random_state=43) if len(val) > max_val else val

    iv_features = [
        "REGIME_ID", "DTE", "SPY_LOG_MONEYNESS", "SPY_MONEYNESS",
        "ROLLING_RV_20D", "ROLLING_ABS_RET_10D", "SPY_RET_LAG1"
    ]

    # V2: spread model gets generated/recomputed Greeks and price ratio.
    spread_features = iv_features + [
        "IV_LEVEL", "OPTION_MID_OVER_SPY",
        "OPTION_DELTA", "OPTION_GAMMA", "OPTION_VEGA", "OPTION_THETA"
    ]

    train_iv = train.dropna(subset=iv_features + ["IV_LEVEL"]).copy()
    val_iv = val.dropna(subset=iv_features + ["IV_LEVEL"]).copy()
    train_spread = train.dropna(subset=spread_features + ["LOG_SPREAD_PROXY"]).copy()
    val_spread = val.dropna(subset=spread_features + ["LOG_SPREAD_PROXY"]).copy()

    common_params = dict(
        max_iter=300 if mode == "final" else 100,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        min_samples_leaf=30,
        random_state=42,
    )

    rows = []
    iv_models = {}
    spread_models = {}

    for q in QUANTILES:
        print(f"Training IV quantile {q:.2f}")
        m_iv = HistGradientBoostingRegressor(loss="quantile", quantile=q, **common_params)
        m_iv.fit(train_iv[iv_features], train_iv["IV_LEVEL"])
        iv_models[q] = m_iv
        joblib.dump(m_iv, MODEL_DIR / f"iv_q{int(q*1000):03d}.joblib")

        pred_iv = m_iv.predict(val_iv[iv_features])
        rows.append({
            "TARGET": "IV_LEVEL",
            "QUANTILE": q,
            "PINBALL": mean_pinball_loss(val_iv["IV_LEVEL"], pred_iv, alpha=q),
            "MAE": mean_absolute_error(val_iv["IV_LEVEL"], pred_iv),
            "RMSE": mean_squared_error(val_iv["IV_LEVEL"], pred_iv) ** 0.5,
            "R2": r2_score(val_iv["IV_LEVEL"], pred_iv),
        })

        print(f"Training spread quantile {q:.2f}")
        m_sp = HistGradientBoostingRegressor(loss="quantile", quantile=q, **common_params)
        m_sp.fit(train_spread[spread_features], train_spread["LOG_SPREAD_PROXY"])
        spread_models[q] = m_sp
        joblib.dump(m_sp, MODEL_DIR / f"spread_q{int(q*1000):03d}.joblib")

        pred_sp = m_sp.predict(val_spread[spread_features])
        rows.append({
            "TARGET": "LOG_SPREAD_PROXY",
            "QUANTILE": q,
            "PINBALL": mean_pinball_loss(val_spread["LOG_SPREAD_PROXY"], pred_sp, alpha=q),
            "MAE": mean_absolute_error(val_spread["LOG_SPREAD_PROXY"], pred_sp),
            "RMSE": mean_squared_error(val_spread["LOG_SPREAD_PROXY"], pred_sp) ** 0.5,
            "R2": r2_score(val_spread["LOG_SPREAD_PROXY"], pred_sp),
        })

    # Coverage diagnostics.
    iv_q05 = iv_models[0.05].predict(val_iv[iv_features])
    iv_q95 = iv_models[0.95].predict(val_iv[iv_features])
    iv_lo, iv_hi = np.minimum(iv_q05, iv_q95), np.maximum(iv_q05, iv_q95)
    iv_cov = float(np.mean((val_iv["IV_LEVEL"].values >= iv_lo) & (val_iv["IV_LEVEL"].values <= iv_hi)))

    sp_q05 = spread_models[0.05].predict(val_spread[spread_features])
    sp_q95 = spread_models[0.95].predict(val_spread[spread_features])
    sp_lo, sp_hi = np.minimum(sp_q05, sp_q95), np.maximum(sp_q05, sp_q95)
    sp_cov = float(np.mean((val_spread["LOG_SPREAD_PROXY"].values >= sp_lo) & (val_spread["LOG_SPREAD_PROXY"].values <= sp_hi)))

    coverage = pd.DataFrame([
        {"TARGET": "IV_LEVEL", "INTERVAL": "Q05-Q95", "COVERAGE": iv_cov, "N_VAL": len(val_iv)},
        {"TARGET": "LOG_SPREAD_PROXY", "INTERVAL": "Q05-Q95", "COVERAGE": sp_cov, "N_VAL": len(val_spread)},
    ])

    # V2 residuals from median models for controlled residual-noise injection in 10b.
    iv_med_pred_train = iv_models[0.50].predict(train_iv[iv_features])
    iv_residual = train_iv["IV_LEVEL"].values - iv_med_pred_train
    spread_med_pred_train = spread_models[0.50].predict(train_spread[spread_features])
    spread_residual = train_spread["LOG_SPREAD_PROXY"].values - spread_med_pred_train

    np.save(MODEL_DIR / "iv_median_residuals.npy", iv_residual)
    np.save(MODEL_DIR / "spread_median_residuals.npy", spread_residual)

    config = {
        "mode": mode,
        "quantiles": QUANTILES,
        "iv_features": iv_features,
        "spread_features": spread_features,
        "spread_eps": SPREAD_EPS,
        "iv_clip": [float(train_iv["IV_LEVEL"].quantile(0.01)), float(train_iv["IV_LEVEL"].quantile(0.99))],
        "spread_clip": [float(train["SPREAD_PROXY"].quantile(0.01)), float(train["SPREAD_PROXY"].quantile(0.99))],
        "iv_widen_scale": DEFAULT_IV_WIDEN_SCALE,
        "spread_widen_scale": DEFAULT_SPREAD_WIDEN_SCALE,
        "iv_residual_scale": DEFAULT_IV_RESIDUAL_SCALE,
        "spread_residual_scale": DEFAULT_SPREAD_RESIDUAL_SCALE,
        "train_rows_iv": int(len(train_iv)),
        "train_rows_spread": int(len(train_spread)),
        "validation_rows_iv": int(len(val_iv)),
        "validation_rows_spread": int(len(val_spread)),
    }

    (MODEL_DIR / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics = pd.DataFrame(rows)
    csv_path = OUT_DIR / f"quantile_model_validation_v2_{mode}.csv"
    xlsx_path = OUT_DIR / f"quantile_model_validation_v2_{mode}.xlsx"
    metrics.to_csv(csv_path, index=False)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Quantile_Metrics", index=False)
        coverage.to_excel(writer, sheet_name="Coverage", index=False)
        pd.DataFrame([config]).to_excel(writer, sheet_name="Config", index=False)

    print(f"Saved models: {MODEL_DIR}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {xlsx_path}")
    print("Coverage:")
    print(coverage)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "final"], default="pilot")
    args = parser.parse_args()
    make_models(args.mode)


if __name__ == "__main__":
    main()
