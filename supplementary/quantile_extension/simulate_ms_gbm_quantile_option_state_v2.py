"""
10b_simulate_ms_gbm_quantile_option_state_v2.py

V2 fixes after pilot diagnostics:
1. Preserve DTE path from the sampled real episode template.
2. Generate IV first, recompute Black--Scholes price/Greeks, then generate spread using Greeks.
3. Apply controlled quantile widening for IV and spread.
4. Add controlled residual noise from median-model residuals.
5. Keep clipping to real train q1%-q99% ranges.

Run pilot-2:
    py src\10b_simulate_ms_gbm_quantile_option_state_v2.py --mode pilot --n_sim 1000 --seed 42

Run final after pilot-2 is accepted:
    py src\10b_simulate_ms_gbm_quantile_option_state_v2.py --mode final --n_sim 5000 --seed 42
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
MODEL_DIR = OUT_DIR / "quantile_option_state_models_v2"
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

    return max(float(price), MIN_OPTION_PRICE), float(delta), float(gamma), float(vega), float(theta)


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

    for c in ["OPTION_IV", "OPTION_SPREAD_PCT", "SPY_MONEYNESS", "SPY_LOG_MONEYNESS"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return out


def load_models():
    config_path = MODEL_DIR / "model_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Cannot find v2 quantile model config: {config_path}. Run 10a_v2 first.")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    qs = [float(x) for x in config["quantiles"]]

    iv_models = {}
    spread_models = {}
    for q in qs:
        key = int(q * 1000)
        iv_models[q] = joblib.load(MODEL_DIR / f"iv_q{key:03d}.joblib")
        spread_models[q] = joblib.load(MODEL_DIR / f"spread_q{key:03d}.joblib")

    iv_resid_path = MODEL_DIR / "iv_median_residuals.npy"
    sp_resid_path = MODEL_DIR / "spread_median_residuals.npy"
    iv_resid = np.load(iv_resid_path) if iv_resid_path.exists() else np.array([0.0])
    sp_resid = np.load(sp_resid_path) if sp_resid_path.exists() else np.array([0.0])

    iv_resid = iv_resid[np.isfinite(iv_resid)]
    sp_resid = sp_resid[np.isfinite(sp_resid)]
    if len(iv_resid) == 0:
        iv_resid = np.array([0.0])
    if len(sp_resid) == 0:
        sp_resid = np.array([0.0])

    return config, qs, iv_models, spread_models, iv_resid, sp_resid


def quantile_predict(models, qs, x_df: pd.DataFrame, u: float, widen_scale: float) -> float:
    preds = np.array([models[q].predict(x_df)[0] for q in qs], dtype=float)
    preds = np.sort(preds)  # fix quantile crossing

    # Post-hoc widening around the median to correct under-dispersion.
    median = float(np.interp(0.50, qs, preds))
    preds = median + float(widen_scale) * (preds - median)
    preds = np.sort(preds)

    u = float(np.clip(u, min(qs), max(qs)))
    return float(np.interp(u, qs, preds))


def make_feature_df(feature_names, vals: dict) -> pd.DataFrame:
    return pd.DataFrame([{c: vals.get(c, 0.0) for c in feature_names}])


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


def percentile_rank(values: pd.Series, train_values: pd.Series) -> pd.Series:
    train = pd.Series(train_values).dropna().sort_values().values
    if len(train) == 0:
        return pd.Series(np.nan, index=values.index)
    arr = pd.to_numeric(values, errors="coerce").values
    return pd.Series(np.searchsorted(train, arr, side="right") / len(train), index=values.index)


def summarize_series(series: pd.Series, label: str) -> dict:
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return {
        "VARIABLE": label,
        "N": len(s),
        "MEAN": s.mean(),
        "STD": s.std(),
        "Q05": s.quantile(0.05),
        "Q50": s.quantile(0.50),
        "Q95": s.quantile(0.95),
    }


def simulate(n_sim: int, seed: int, mode: str, iv_widen: float | None, spread_widen: float | None,
             iv_noise: float | None, spread_noise: float | None):
    rng = np.random.default_rng(seed)

    real = add_basic_features(load_real_data())
    train = real[real["SPLIT"].astype(str).str.lower().eq("train")].copy()
    if train.empty:
        raise ValueError("No train split found in real data.")

    if not HMM_NPZ_FILE.exists():
        raise FileNotFoundError(f"Cannot find HMM params: {HMM_NPZ_FILE}. Run HMM calibration first.")

    hmm = np.load(HMM_NPZ_FILE, allow_pickle=True)
    pi, A, means, variances = hmm["pi"], hmm["A"], hmm["means"], hmm["variances"]

    sigmas_daily = np.sqrt(variances)
    sigmas_ann = sigmas_daily * math.sqrt(TRADING_DAYS)

    config, qs, iv_models, spread_models, iv_resid, spread_resid = load_models()

    iv_features = config["iv_features"]
    spread_features = config["spread_features"]

    iv_widen_scale = float(config.get("iv_widen_scale", 1.25) if iv_widen is None else iv_widen)
    spread_widen_scale = float(config.get("spread_widen_scale", 1.75) if spread_widen is None else spread_widen)
    iv_resid_scale = float(config.get("iv_residual_scale", 0.35) if iv_noise is None else iv_noise)
    spread_resid_scale = float(config.get("spread_residual_scale", 0.50) if spread_noise is None else spread_noise)

    iv_clip = [float(config["iv_clip"][0]), float(config["iv_clip"][1])]
    spread_clip = [float(config["spread_clip"][0]), float(config["spread_clip"][1])]
    spread_eps = float(config.get("spread_eps", 1e-6))

    # Use real train templates and preserve their DTE paths.
    templates = {}
    for eid, g in train.groupby("EPISODE_ID"):
        gg = g.sort_values("QUOTE_DATE").reset_index(drop=True).copy()
        if len(gg) >= 3 and gg["DTE"].notna().all():
            templates[eid] = gg
    ep_ids = list(templates.keys())
    if not ep_ids:
        raise ValueError("No usable train episode templates found.")

    real_iv = pd.to_numeric(train["OPTION_IV"], errors="coerce")
    real_iv = real_iv.where(real_iv <= 3.0, real_iv / 100.0)
    real_spread = pd.to_numeric(train.get("OPTION_SPREAD_PCT", train.get("SPREAD_PROXY", np.nan)), errors="coerce")

    labels = ["low_vol", "medium_vol", "high_vol"]
    # Synthetic dates are reused across episodes to avoid pandas Timestamp overflow.
    base_date = pd.Timestamp("2000-01-03")
    rows = []

    print("Simulation settings:")
    print(f"  n_sim={n_sim}, seed={seed}, mode={mode}")
    print(f"  IV widen={iv_widen_scale}, spread widen={spread_widen_scale}")
    print(f"  IV residual scale={iv_resid_scale}, spread residual scale={spread_resid_scale}")
    print("  DTE path: preserved from real template")

    for ep_idx in range(n_sim):
        tpl = templates[ep_ids[int(rng.integers(0, len(ep_ids)))]].copy()
        tpl = tpl.reset_index(drop=True)

        # Preserve full template length, but avoid pathological very long episodes if any.
        n_steps = int(min(len(tpl), 90))
        tpl = tpl.iloc[:n_steps].copy()

        first = tpl.iloc[0]
        S0 = float(first["SPY_CLOSE"])
        m0 = float(first.get("SPY_MONEYNESS", S0 / float(first["STRIKE"])))
        if not np.isfinite(m0) or m0 <= 0:
            m0 = 1.0
        K = S0 / m0

        # V2: DTE nodes exactly from template rows + final NEXT_DTE.
        dte_nodes = []
        for t in range(n_steps):
            dte_nodes.append(float(tpl.iloc[t]["DTE"]))
        if "NEXT_DTE" in tpl.columns and pd.notna(tpl.iloc[n_steps - 1].get("NEXT_DTE", np.nan)):
            dte_nodes.append(float(tpl.iloc[n_steps - 1]["NEXT_DTE"]))
        else:
            dte_nodes.append(max(float(tpl.iloc[n_steps - 1]["DTE"]) - 1.0, 0.0))
        dte_nodes = [max(float(x), 0.0) for x in dte_nodes]

        # Simulate regime path and underlying path.
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

        # Generate node-level option state.
        node = []
        u_episode = rng.uniform(0.05, 0.95)
        prev_ret = 0.0

        for t in range(n_steps + 1):
            dte = dte_nodes[t]
            mny = S[t] / K
            log_mny = math.log(max(mny, 1e-12))
            rv20 = rolling_rv(rets[: t + 1], 20, min_n=3)
            rv10 = rolling_rv(rets[: t + 1], 10, min_n=3)
            abs10 = rolling_abs(rets[: t + 1], 10, min_n=3)
            reg = int(regimes[t])

            base_vals = {
                "REGIME_ID": reg,
                "DTE": dte,
                "SPY_LOG_MONEYNESS": log_mny,
                "SPY_MONEYNESS": mny,
                "ROLLING_RV_20D": rv20,
                "ROLLING_RV_10D": rv10,
                "ROLLING_ABS_RET_10D": abs10,
                "SPY_RET_LAG1": prev_ret,
            }

            # IV quantile generation with widening + residual noise.
            u_t = 0.7 * u_episode + 0.3 * rng.uniform(0.05, 0.95)
            u_t = float(np.clip(u_t, 0.05, 0.95))
            x_iv = make_feature_df(iv_features, base_vals)
            iv = quantile_predict(iv_models, qs, x_iv, u_t, iv_widen_scale)
            if iv_resid_scale > 0:
                iv += iv_resid_scale * float(rng.choice(iv_resid))
            iv = float(np.clip(iv, iv_clip[0], iv_clip[1]))

            # Recompute price and Greeks before spread prediction.
            price, delta, gamma, vega, theta = bs_call_price_greeks(S[t], K, dte, iv, RISK_FREE_RATE, DIVIDEND_YIELD)

            spread_vals = dict(base_vals)
            spread_vals.update({
                "IV_LEVEL": iv,
                "OPTION_MID_OVER_SPY": price / S[t],
                "OPTION_DELTA": delta,
                "OPTION_GAMMA": gamma,
                "OPTION_VEGA": vega,
                "OPTION_THETA": theta,
            })

            # Correlated spread quantile to preserve partial IV-spread dependence.
            v_t = 0.5 * u_t + 0.5 * rng.uniform(0.05, 0.95)
            v_t = float(np.clip(v_t, 0.05, 0.95))
            x_sp = make_feature_df(spread_features, spread_vals)
            log_spread = quantile_predict(spread_models, qs, x_sp, v_t, spread_widen_scale)
            if spread_resid_scale > 0:
                log_spread += spread_resid_scale * float(rng.choice(spread_resid))
            spread = math.exp(log_spread) - spread_eps
            spread = float(np.clip(spread, spread_clip[0], spread_clip[1]))

            node.append({
                "S": S[t],
                "DTE": dte,
                "REGIME_ID": reg,
                "REGIME_LABEL": labels[reg] if reg < len(labels) else str(reg),
                "IV": iv,
                "SPREAD": spread,
                "PRICE": price,
                "DELTA": delta,
                "GAMMA": gamma,
                "VEGA": vega,
                "THETA": theta,
                "MONEYNESS": mny,
                "LOG_MONEYNESS": log_mny,
                "RV10": rv10,
                "RV20": rv20,
                "ABS10": abs10,
                "RET_LAG1": prev_ret,
            })

            prev_ret = rets[t] if t < len(rets) else prev_ret

        # Transition-level rows.
        for t in range(n_steps):
            cur = node[t]
            nxt = node[t + 1]
            # Keep synthetic dates inside pandas' valid nanosecond timestamp range.
            # The episode identifier is already unique, so dates do not need to be unique across episodes.
            quote_date = base_date + pd.Timedelta(days=int(t))
            next_quote_date = base_date + pd.Timedelta(days=int(t + 1))

            rows.append({
                "EPISODE_ID": f"ms_gbm_quantile_v2_{mode}_{ep_idx:06d}",
                "SPLIT": "train",
                "QUOTE_DATE": quote_date,
                "NEXT_QUOTE_DATE": next_quote_date,

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
                "IV_LEVEL": cur["IV"],

                "OPTION_SPREAD_PCT": cur["SPREAD"],
                "SPREAD_PROXY": cur["SPREAD"],

                "SPY_MONEYNESS": cur["MONEYNESS"],
                "SPY_LOG_MONEYNESS": cur["LOG_MONEYNESS"],
                "OPTION_MID_OVER_SPY": cur["PRICE"] / cur["S"],

                "SPY_RET_LAG1": cur["RET_LAG1"],
                "SPY_LOG_RET_DAILY": rets[t] if t < len(rets) else 0.0,
                "ROLLING_RV_10D": cur["RV10"],
                "ROLLING_RV_20D": cur["RV20"],
                "ROLLING_ABS_RET_10D": cur["ABS10"],

                "SIM_REGIME_ID": cur["REGIME_ID"],
                "SIM_REGIME_LABEL": cur["REGIME_LABEL"],
                "REGIME_ID": cur["REGIME_ID"],
            })

    sim = pd.DataFrame(rows)
    sim["IV_PERCENTILE"] = percentile_rank(sim["IV_LEVEL"], real_iv)
    sim["SPREAD_PERCENTILE"] = percentile_rank(sim["SPREAD_PROXY"], real_spread)

    out_name = "ms_gbm_quantile_option_state_v2_pilot.parquet" if mode == "pilot" else f"ms_gbm_quantile_option_state_v2_final_n{n_sim}.parquet"
    out_path = SIM_DIR / out_name
    sim.to_parquet(out_path, index=False)
    print(f"Saved simulated data: {out_path}")

    # Simple validation workbook.
    validation_rows = []
    for col in ["DTE", "OPTION_IV", "OPTION_SPREAD_PCT", "OPTION_DELTA", "OPTION_GAMMA", "OPTION_VEGA", "OPTION_THETA", "SPY_MONEYNESS", "OPTION_MID_OVER_SPY"]:
        validation_rows.append(summarize_series(sim[col], col))
    state_summary = pd.DataFrame(validation_rows)

    ep_rv = (
        sim.assign(LOG_RET=np.log(sim["SPY_NEXT_CLOSE"] / sim["SPY_CLOSE"]))
        .groupby("EPISODE_ID")["LOG_RET"]
        .std()
        .mul(math.sqrt(TRADING_DAYS))
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

    xlsx = OUT_DIR / f"simulator_validation_quantile_v2_{mode}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        state_summary.to_excel(writer, sheet_name="State_Summary", index=False)
        ep_summary.to_excel(writer, sheet_name="Episode_RV", index=False)
        pd.DataFrame([{
            "mode": mode,
            "n_sim": n_sim,
            "seed": seed,
            "iv_widen_scale": iv_widen_scale,
            "spread_widen_scale": spread_widen_scale,
            "iv_residual_scale": iv_resid_scale,
            "spread_residual_scale": spread_resid_scale,
            "sim_file": str(out_path),
        }]).to_excel(writer, sheet_name="Config", index=False)
        sim.head(5000).to_excel(writer, sheet_name="Sample", index=False)

    print(f"Saved validation: {xlsx}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "final"], default="pilot")
    parser.add_argument("--n_sim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # Optional tuning knobs for pilot-2.
    parser.add_argument("--iv_widen", type=float, default=None)
    parser.add_argument("--spread_widen", type=float, default=None)
    parser.add_argument("--iv_noise", type=float, default=None)
    parser.add_argument("--spread_noise", type=float, default=None)

    args = parser.parse_args()
    n_sim = args.n_sim if args.n_sim is not None else (1000 if args.mode == "pilot" else 5000)

    simulate(
        n_sim=n_sim,
        seed=args.seed,
        mode=args.mode,
        iv_widen=args.iv_widen,
        spread_widen=args.spread_widen,
        iv_noise=args.iv_noise,
        spread_noise=args.spread_noise,
    )


if __name__ == "__main__":
    main()
