"""
10a_fit_quantile_option_state_models.py

Fit Gradient Boosting Quantile models for option-state simulation.

Outputs:
    outputs/quantile_option_state_models/*.joblib
    outputs/quantile_option_state_models/model_config.json
    outputs/quantile_model_validation.xlsx
    outputs/quantile_model_validation.csv

Run:
    py src\10a_fit_quantile_option_state_models.py --mode pilot
    py src\10a_fit_quantile_option_state_models.py --mode final
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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_pinball_loss


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
OUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = OUT_DIR / "quantile_option_state_models"
OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

REAL_PROXY_FILE = DATA_DIR / "transitions_daily_top1_final_with_spy_2010_2023_with_regime_proxies.parquet"
REAL_BASE_FILE = DATA_DIR / "transitions_daily_top1_final_with_spy_2010_2023.parquet"
HMM_NPZ_FILE = OUT_DIR / "hmm_regime_params_pilot.npz"

QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
SPREAD_EPS = 1e-6
TRADING_DAYS = 252


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
    b = np.exp(-0.5 * (x - means) ** 2 / variances) / np.sqrt(2 * np.pi * variances)
    return np.maximum(b, 1e-300)


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
        return out
    if not HMM_NPZ_FILE.exists():
        print("HMM params not found; REGIME_ID set to 1 (medium) as fallback.")
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

    # Daily return-based proxies if missing
    if "SPY_LOG_RET_DAILY" not in out.columns or "ROLLING_RV_20D" not in out.columns:
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
        out["SPREAD_PROXY"] = 0.01

    out["LOG_SPREAD_PROXY"] = np.log(np.maximum(out["SPREAD_PROXY"].astype(float), 0.0) + SPREAD_EPS)

    for c in [
        "REGIME_ID", "DTE", "SPY_LOG_MONEYNESS", "SPY_MONEYNESS", "ROLLING_RV_20D",
        "ROLLING_ABS_RET_10D", "SPY_RET_LAG1", "IV_LEVEL", "OPTION_MID_OVER_SPY",
        "SPREAD_PROXY", "LOG_SPREAD_PROXY"
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return out


def make_models(mode: str):
    df = load_real_data()
    df = add_regime_labels(df)
    df = add_features(df)

    train = df[df["SPLIT"].astype(str).str.lower().eq("train")].copy()
    val = df[df["SPLIT"].astype(str).str.lower().isin(["val", "validation"])].copy()
    if val.empty:
        print("Validation split not found. Using test split for diagnostics only.")
        val = df[df["SPLIT"].astype(str).str.lower().eq("test")].copy()
    if val.empty:
        val = train.sample(frac=0.2, random_state=42)
        train = train.drop(index=val.index)

    if mode == "pilot":
        max_train = min(len(train), 50000)
        train = train.sample(n=max_train, random_state=42) if len(train) > max_train else train
        val = val.sample(n=min(len(val), 20000), random_state=43) if len(val) > 20000 else val

    iv_features = ["REGIME_ID", "DTE", "SPY_LOG_MONEYNESS", "SPY_MONEYNESS", "ROLLING_RV_20D", "ROLLING_ABS_RET_10D", "SPY_RET_LAG1"]
    spread_features = iv_features + ["IV_LEVEL", "OPTION_MID_OVER_SPY"]

    train_iv = train.dropna(subset=iv_features + ["IV_LEVEL"]).copy()
    train_spread = train.dropna(subset=spread_features + ["LOG_SPREAD_PROXY"]).copy()
    val_iv = val.dropna(subset=iv_features + ["IV_LEVEL"]).copy()
    val_spread = val.dropna(subset=spread_features + ["LOG_SPREAD_PROXY"]).copy()

    config = {
        "quantiles": QUANTILES,
        "iv_features": iv_features,
        "spread_features": spread_features,
        "spread_eps": SPREAD_EPS,
        "iv_clip": [float(train_iv["IV_LEVEL"].quantile(0.01)), float(train_iv["IV_LEVEL"].quantile(0.99))],
        "spread_clip": [float(train["SPREAD_PROXY"].quantile(0.01)), float(train["SPREAD_PROXY"].quantile(0.99))],
    }

    rows = []
    iv_models = {}
    spread_models = {}

    common_params = dict(
        max_iter=250 if mode == "final" else 80,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=0.1,
        min_samples_leaf=30,
        random_state=42,
    )

    for q in QUANTILES:
        print(f"Training IV quantile {q:.2f}")
        m = HistGradientBoostingRegressor(loss="quantile", quantile=q, **common_params)
        m.fit(train_iv[iv_features], train_iv["IV_LEVEL"])
        iv_models[q] = m
        joblib.dump(m, MODEL_DIR / f"iv_q{int(q*1000):03d}.joblib")
        pred = m.predict(val_iv[iv_features])
        rows.append({
            "TARGET": "IV_LEVEL", "QUANTILE": q,
            "PINBALL": mean_pinball_loss(val_iv["IV_LEVEL"], pred, alpha=q),
            "MAE": mean_absolute_error(val_iv["IV_LEVEL"], pred),
            "RMSE": mean_squared_error(val_iv["IV_LEVEL"], pred) ** 0.5,
            "R2": r2_score(val_iv["IV_LEVEL"], pred),
        })

        print(f"Training spread quantile {q:.2f}")
        sm = HistGradientBoostingRegressor(loss="quantile", quantile=q, **common_params)
        sm.fit(train_spread[spread_features], train_spread["LOG_SPREAD_PROXY"])
        spread_models[q] = sm
        joblib.dump(sm, MODEL_DIR / f"spread_q{int(q*1000):03d}.joblib")
        spred = sm.predict(val_spread[spread_features])
        rows.append({
            "TARGET": "LOG_SPREAD_PROXY", "QUANTILE": q,
            "PINBALL": mean_pinball_loss(val_spread["LOG_SPREAD_PROXY"], spred, alpha=q),
            "MAE": mean_absolute_error(val_spread["LOG_SPREAD_PROXY"], spred),
            "RMSE": mean_squared_error(val_spread["LOG_SPREAD_PROXY"], spred) ** 0.5,
            "R2": r2_score(val_spread["LOG_SPREAD_PROXY"], spred),
        })

    # 90% coverage using q05/q95
    q05_iv = iv_models[0.05].predict(val_iv[iv_features])
    q95_iv = iv_models[0.95].predict(val_iv[iv_features])
    iv_lo = np.minimum(q05_iv, q95_iv)
    iv_hi = np.maximum(q05_iv, q95_iv)
    iv_cov = np.mean((val_iv["IV_LEVEL"].values >= iv_lo) & (val_iv["IV_LEVEL"].values <= iv_hi))

    q05_sp = spread_models[0.05].predict(val_spread[spread_features])
    q95_sp = spread_models[0.95].predict(val_spread[spread_features])
    sp_lo = np.minimum(q05_sp, q95_sp)
    sp_hi = np.maximum(q05_sp, q95_sp)
    sp_cov = np.mean((val_spread["LOG_SPREAD_PROXY"].values >= sp_lo) & (val_spread["LOG_SPREAD_PROXY"].values <= sp_hi))

    coverage = pd.DataFrame([
        {"TARGET": "IV_LEVEL", "INTERVAL": "Q05-Q95", "COVERAGE": iv_cov, "N_VAL": len(val_iv)},
        {"TARGET": "LOG_SPREAD_PROXY", "INTERVAL": "Q05-Q95", "COVERAGE": sp_cov, "N_VAL": len(val_spread)},
    ])

    config["train_rows_iv"] = int(len(train_iv))
    config["train_rows_spread"] = int(len(train_spread))
    config["validation_rows_iv"] = int(len(val_iv))
    config["validation_rows_spread"] = int(len(val_spread))
    config["mode"] = mode

    (MODEL_DIR / "model_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    metrics = pd.DataFrame(rows)
    metrics_path = OUT_DIR / f"quantile_model_validation_{mode}.csv"
    xlsx_path = OUT_DIR / f"quantile_model_validation_{mode}.xlsx"
    metrics.to_csv(metrics_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Quantile_Metrics", index=False)
        coverage.to_excel(writer, sheet_name="Coverage", index=False)
        pd.DataFrame([config]).to_excel(writer, sheet_name="Config", index=False)

    print(f"Saved: {metrics_path}")
    print(f"Saved: {xlsx_path}")
    print(f"Saved models: {MODEL_DIR}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "final"], default="pilot")
    args = parser.parse_args()
    make_models(args.mode)


if __name__ == "__main__":
    main()
