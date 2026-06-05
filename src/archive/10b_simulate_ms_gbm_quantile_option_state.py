"""
10b_simulate_ms_gbm_quantile_option_state.py

Simulate HMM-MSGBM paths with Gradient Boosting Quantile-generated option states.

Outputs:
    data/simulated/ms_gbm_quantile_option_state_pilot.parquet
    data/simulated/ms_gbm_quantile_option_state_final_n5000.parquet
    outputs/simulator_validation_quantile_<mode>.xlsx

Run:
    py src\10b_simulate_ms_gbm_quantile_option_state.py --mode pilot --n_sim 1000 --seed 42
    py src\10b_simulate_ms_gbm_quantile_option_state.py --mode final --n_sim 5000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
SIM_DIR = PROJECT_DIR / "data" / "simulated"
OUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = OUT_DIR / "quantile_option_state_models"
SIM_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

REAL_PROXY_FILE = DATA_DIR / "transitions_daily_top1_final_with_spy_2010_2023_with_regime_proxies.parquet"
REAL_BASE_FILE = DATA_DIR / "transitions_daily_top1_final_with_spy_2010_2023.parquet"
HMM_NPZ_FILE = OUT_DIR / "hmm_regime_params_pilot.npz"

TRADING_DAYS = 252
DT = 1.0 / TRADING_DAYS
RISK_FREE_RATE = 0.0
DIVIDEND_YIELD = 0.0
MIN_OPTION_PRICE = 0.01


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_price_greeks(S: float, K: float, tau_days: float, sigma: float, r: float = 0.0, q: float = 0.0):
    tau = max(float(tau_days) / TRADING_DAYS, 1e-8)
    sigma = max(float(sigma), 1e-6)
    S = max(float(S), 1e-8)
    K = max(float(K), 1e-8)

    if tau_days <= 0:
        price = max(S - K, 0.0)
        delta = 1.0 if S > K else 0.0
        return max(price, MIN_OPTION_PRICE), delta, 0.0, 0.0, 0.0

    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    disc_q = math.exp(-q * tau)
    disc_r = math.exp(-r * tau)
    price = S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
    delta = disc_q * norm_cdf(d1)
    gamma = disc_q * norm_pdf(d1) / (S * sigma * sqrt_tau)
    vega = S * disc_q * norm_pdf(d1) * sqrt_tau / 100.0
    theta = (
        -(S * disc_q * norm_pdf(d1) * sigma) / (2.0 * sqrt_tau)
        - r * K * disc_r * norm_cdf(d2)
        + q * S * disc_q * norm_cdf(d1)
    ) / TRADING_DAYS
    return max(price, MIN_OPTION_PRICE), float(delta), float(gamma), float(vega), float(theta)


def percentile_rank(values: pd.Series, train_values: pd.Series) -> pd.Series:
    train = pd.Series(train_values).dropna().sort_values().values
    if len(train) == 0:
        return pd.Series(np.nan, index=values.index)
    arr = pd.to_numeric(values, errors="coerce").values
    return pd.Series(np.searchsorted(train, arr, side="right") / len(train), index=values.index)


def load_real_data() -> pd.DataFrame:
    path = REAL_PROXY_FILE if REAL_PROXY_FILE.exists() else REAL_BASE_FILE
    if not path.exists():
        raise FileNotFoundError(f"Cannot find real transition file:\n{REAL_PROXY_FILE}\n{REAL_BASE_FILE}")
    print(f"Loading real data: {path}")
    return pd.read_parquet(path)


def add_basic_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)
    if "SPY_MONEYNESS" not in out.columns:
        out["SPY_MONEYNESS"] = out["SPY_CLOSE"].astype(float) / out["STRIKE"].astype(float)
    if "SPY_LOG_MONEYNESS" not in out.columns:
        out["SPY_LOG_MONEYNESS"] = np.log(out["SPY_MONEYNESS"].astype(float))
    if "OPTION_MID_OVER_SPY" not in out.columns:
        out["OPTION_MID_OVER_SPY"] = out["OPTION_MID"].astype(float) / out["SPY_CLOSE"].astype(float)
    if "SPY_RET_LAG1" not in out.columns:
        out["SPY_RET_LAG1"] = out.groupby("EPISODE_ID")["SPY_CLOSE"].pct_change().fillna(0.0)
    if "OPTION_SPREAD_PCT" not in out.columns and "SPREAD_PROXY" in out.columns:
        out["OPTION_SPREAD_PCT"] = out["SPREAD_PROXY"]
    return out


def load_models():
    config_path = MODEL_DIR / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot find quantile model config: {config_path}. Run 10a first.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    qs = [float(x) for x in config["quantiles"]]
    iv_models = {}
    spread_models = {}
    for q in qs:
        key = int(q * 1000)
        iv_models[q] = joblib.load(MODEL_DIR / f"iv_q{key:03d}.joblib")
        spread_models[q] = joblib.load(MODEL_DIR / f"spread_q{key:03d}.joblib")
    return config, qs, iv_models, spread_models


def predict_quantile_value(models, qs, x_df: pd.DataFrame, u: float) -> float:
    preds = np.array([models[q].predict(x_df)[0] for q in qs], dtype=float)
    preds = np.sort(preds)  # fix quantile crossing
    u = float(np.clip(u, min(qs), max(qs)))
    return float(np.interp(u, qs, preds))


def rolling_rv(returns: list[float], window: int, min_n: int = 3) -> float:
    vals = np.array(returns[-window:], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < min_n:
        return 0.0
    return float(np.std(vals, ddof=1) * math.sqrt(TRADING_DAYS))


def rolling_abs(returns: list[float], window: int, min_n: int = 3) -> float:
    vals = np.array(returns[-window:], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < min_n:
        return 0.0
    return float(np.mean(np.abs(vals)))


def make_feature_df(feature_names, vals: dict) -> pd.DataFrame:
    return pd.DataFrame([{c: vals.get(c, 0.0) for c in feature_names}])


def summarize_returns(df: pd.DataFrame, label: str) -> dict:
    ret = np.log(df["SPY_NEXT_CLOSE"].astype(float) / df["SPY_CLOSE"].astype(float)).replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "DATASET": label,
        "N_RETURNS": len(ret),
        "MEAN_DAILY_RETURN": ret.mean(),
        "STD_DAILY_RETURN": ret.std(),
        "ANN_VOL": ret.std() * math.sqrt(TRADING_DAYS),
        "SKEW": ret.skew(),
        "KURTOSIS": ret.kurtosis(),
        "Q05": ret.quantile(0.05),
        "Q50": ret.quantile(0.50),
        "Q95": ret.quantile(0.95),
    }


def simulate(n_sim: int, seed: int, mode: str):
    rng = np.random.default_rng(seed)
    real = add_basic_features(load_real_data())
    train = real[real["SPLIT"].astype(str).str.lower().eq("train")].copy()
    if train.empty:
        raise ValueError("No train split found in real data.")

    if not HMM_NPZ_FILE.exists():
        raise FileNotFoundError(f"Cannot find HMM params: {HMM_NPZ_FILE}. Run 08a first.")
    hmm = np.load(HMM_NPZ_FILE, allow_pickle=True)
    pi, A, means, variances = hmm["pi"], hmm["A"], hmm["means"], hmm["variances"]
    sigmas_daily = np.sqrt(variances)
    sigmas_ann = sigmas_daily * math.sqrt(TRADING_DAYS)

    config, qs, iv_models, spread_models = load_models()
    iv_features = config["iv_features"]
    spread_features = config["spread_features"]
    iv_clip = config["iv_clip"]
    spread_clip = config["spread_clip"]
    spread_eps = float(config.get("spread_eps", 1e-6))

    # Templates and fallback medians
    episodes = {eid: g.sort_values("QUOTE_DATE").reset_index(drop=True) for eid, g in train.groupby("EPISODE_ID")}
    ep_ids = list(episodes.keys())
    if not ep_ids:
        raise ValueError("No episodes found in train data.")

    real_iv = pd.to_numeric(train["OPTION_IV"], errors="coerce")
    real_iv = real_iv.where(real_iv <= 3.0, real_iv / 100.0)
    real_spread = pd.to_numeric(train.get("OPTION_SPREAD_PCT", train.get("SPREAD_PROXY", 0.01)), errors="coerce")
    iv_train_values = real_iv.dropna()
    spread_train_values = real_spread.dropna()

    rows = []
    base_date = pd.Timestamp("2000-01-03")
    labels = ["low_vol", "medium_vol", "high_vol"]

    for ep_idx in range(n_sim):
        tpl = episodes[ep_ids[int(rng.integers(0, len(ep_ids)))]]
        first = tpl.iloc[0]

        dte0 = int(max(5, min(90, float(first.get("DTE", 30)))))
        tpl_len = int(len(tpl))
        n_steps = int(max(3, min(tpl_len, dte0, 60)))

        S0 = float(first["SPY_CLOSE"])
        m0 = float(first.get("SPY_MONEYNESS", S0 / float(first["STRIKE"])))
        if not np.isfinite(m0) or m0 <= 0:
            m0 = 1.0
        K = S0 / m0

        # Simulate regime and price path
        z = int(rng.choice(len(pi), p=pi / pi.sum()))
        regimes = [z]
        S = [S0]
        rets = [0.0]
        for t in range(n_steps):
            sigma_ann = float(sigmas_ann[z])
            mu_daily = float(means[z])
            eps = rng.normal()
            next_S = S[-1] * math.exp((mu_daily - 0.5 * sigma_ann * sigma_ann * DT) + sigma_ann * math.sqrt(DT) * eps)
            next_S = max(next_S, 1e-6)
            S.append(next_S)
            rets.append(math.log(next_S / S[-2]))
            z = int(rng.choice(len(pi), p=A[z] / A[z].sum()))
            regimes.append(z)

        # Generate node IV/spread/price/greeks
        node = []
        u_episode = rng.uniform(0.05, 0.95)
        prev_ret = 0.0

        for t in range(n_steps + 1):
            dte = max(dte0 - t, 0)
            mny = S[t] / K
            log_mny = math.log(max(mny, 1e-12))
            rv20 = rolling_rv(rets[: t + 1], 20, min_n=3)
            rv10 = rolling_rv(rets[: t + 1], 10, min_n=3)
            abs10 = rolling_abs(rets[: t + 1], 10, min_n=3)
            reg = regimes[t]

            feature_vals = {
                "REGIME_ID": reg,
                "DTE": dte,
                "SPY_LOG_MONEYNESS": log_mny,
                "SPY_MONEYNESS": mny,
                "ROLLING_RV_20D": rv20,
                "ROLLING_RV_10D": rv10,
                "ROLLING_ABS_RET_10D": abs10,
                "SPY_RET_LAG1": prev_ret,
            }

            u_t = 0.7 * u_episode + 0.3 * rng.uniform(0.05, 0.95)
            u_t = float(np.clip(u_t, 0.05, 0.95))
            x_iv = make_feature_df(iv_features, feature_vals)
            iv = predict_quantile_value(iv_models, qs, x_iv, u_t)
            iv = float(np.clip(iv, iv_clip[0], iv_clip[1]))

            price, delta, gamma, vega, theta = bs_call_price_greeks(S[t], K, dte, iv, RISK_FREE_RATE, DIVIDEND_YIELD)
            feature_vals["IV_LEVEL"] = iv
            feature_vals["OPTION_MID_OVER_SPY"] = price / S[t]

            v_t = 0.5 * u_t + 0.5 * rng.uniform(0.05, 0.95)
            v_t = float(np.clip(v_t, 0.05, 0.95))
            x_sp = make_feature_df(spread_features, feature_vals)
            log_sp = predict_quantile_value(spread_models, qs, x_sp, v_t)
            spread = math.exp(log_sp) - spread_eps
            spread = float(np.clip(spread, spread_clip[0], spread_clip[1]))

            node.append({
                "S": S[t], "DTE": dte, "REGIME_ID": reg, "REGIME_LABEL": labels[reg] if reg < len(labels) else str(reg),
                "IV": iv, "SPREAD": spread, "PRICE": price, "DELTA": delta, "GAMMA": gamma, "VEGA": vega, "THETA": theta,
                "MONEYNESS": mny, "LOG_MONEYNESS": log_mny, "RV10": rv10, "RV20": rv20, "ABS10": abs10, "RET_LAG1": prev_ret,
            })
            prev_ret = rets[t] if t < len(rets) else prev_ret

        for t in range(n_steps):
            cur, nxt = node[t], node[t + 1]
            rows.append({
                "EPISODE_ID": f"ms_gbm_quantile_{mode}_{ep_idx:06d}",
                "SPLIT": "train",
                "QUOTE_DATE": base_date + pd.Timedelta(days=int(t)),
                "NEXT_QUOTE_DATE": base_date + pd.Timedelta(days=int(t + 1)),
                "SPY_CLOSE": cur["S"],
                "SPY_NEXT_CLOSE": nxt["S"],
                "SPY_DS": nxt["S"] - cur["S"],
                "STRIKE": K,
                "DTE": cur["DTE"],
                "NEXT_DTE": nxt["DTE"],
                "OPTION_MID": cur["PRICE"],
                "NEXT_OPTION_MID": nxt["PRICE"],
                "DOPTION": nxt["PRICE"] - cur["PRICE"],
                "OPTION_DELTA": cur["DELTA"],
                "OPTION_GAMMA": cur["GAMMA"],
                "OPTION_VEGA": cur["VEGA"],
                "OPTION_THETA": cur["THETA"],
                "OPTION_IV": cur["IV"],
                "OPTION_SPREAD_PCT": cur["SPREAD"],
                "SPREAD_PROXY": cur["SPREAD"],
                "SPY_MONEYNESS": cur["MONEYNESS"],
                "SPY_LOG_MONEYNESS": cur["LOG_MONEYNESS"],
                "OPTION_MID_OVER_SPY": cur["PRICE"] / cur["S"],
                "SPY_RET_LAG1": cur["RET_LAG1"],
                "SIM_REGIME_ID": cur["REGIME_ID"],
                "SIM_REGIME_LABEL": cur["REGIME_LABEL"],
                "REGIME_ID": cur["REGIME_ID"],
                "SPY_LOG_RET_DAILY": rets[t] if t < len(rets) else 0.0,
                "ROLLING_RV_10D": cur["RV10"],
                "ROLLING_RV_20D": cur["RV20"],
                "ROLLING_ABS_RET_10D": cur["ABS10"],
                "IV_LEVEL": cur["IV"],
            })

    sim = pd.DataFrame(rows)
    sim["IV_PERCENTILE"] = percentile_rank(sim["IV_LEVEL"], iv_train_values)
    sim["SPREAD_PERCENTILE"] = percentile_rank(sim["SPREAD_PROXY"], spread_train_values)

    out_path = SIM_DIR / ("ms_gbm_quantile_option_state_pilot.parquet" if mode == "pilot" else f"ms_gbm_quantile_option_state_final_n{n_sim}.parquet")
    sim.to_parquet(out_path, index=False)
    print(f"Saved simulated data: {out_path}")

    # Validation workbook
    ret_summary = pd.DataFrame([
        summarize_returns(sim, "MS-GBM Quantile sim")
    ])

    state_summary = []
    for col in ["OPTION_IV", "OPTION_SPREAD_PCT", "OPTION_DELTA", "OPTION_GAMMA", "OPTION_VEGA", "OPTION_THETA", "SPY_MONEYNESS", "DTE", "OPTION_MID_OVER_SPY"]:
        state_summary.append({
            "VARIABLE": col,
            "MEAN": pd.to_numeric(sim[col], errors="coerce").mean(),
            "STD": pd.to_numeric(sim[col], errors="coerce").std(),
            "Q05": pd.to_numeric(sim[col], errors="coerce").quantile(0.05),
            "Q50": pd.to_numeric(sim[col], errors="coerce").quantile(0.50),
            "Q95": pd.to_numeric(sim[col], errors="coerce").quantile(0.95),
        })
    state_summary = pd.DataFrame(state_summary)

    ep_rv = (
        sim.assign(LOG_RET=np.log(sim["SPY_NEXT_CLOSE"] / sim["SPY_CLOSE"]))
        .groupby("EPISODE_ID")["LOG_RET"].std().mul(math.sqrt(TRADING_DAYS))
        .reset_index(name="EPISODE_RV")
    )
    ep_summary = pd.DataFrame([{
        "EPISODES": sim["EPISODE_ID"].nunique(),
        "MEAN_EPISODE_RV": ep_rv["EPISODE_RV"].mean(),
        "STD_EPISODE_RV": ep_rv["EPISODE_RV"].std(),
        "Q05_EPISODE_RV": ep_rv["EPISODE_RV"].quantile(0.05),
        "Q50_EPISODE_RV": ep_rv["EPISODE_RV"].quantile(0.50),
        "Q95_EPISODE_RV": ep_rv["EPISODE_RV"].quantile(0.95),
    }])

    xlsx = OUT_DIR / f"simulator_validation_quantile_{mode}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        ret_summary.to_excel(writer, sheet_name="Return_Stats", index=False)
        state_summary.to_excel(writer, sheet_name="State_Summary", index=False)
        ep_summary.to_excel(writer, sheet_name="Episode_RV", index=False)
        sim.head(5000).to_excel(writer, sheet_name="Sample", index=False)
    print(f"Saved validation: {xlsx}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "final"], default="pilot")
    parser.add_argument("--n_sim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    n_sim = args.n_sim if args.n_sim is not None else (1000 if args.mode == "pilot" else 5000)
    simulate(n_sim=n_sim, seed=args.seed, mode=args.mode)


if __name__ == "__main__":
    main()
