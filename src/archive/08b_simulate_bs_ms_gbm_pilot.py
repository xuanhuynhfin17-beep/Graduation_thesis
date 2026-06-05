"""
08b_simulate_bs_ms_gbm_pilot.py

Pilot simulators for pretraining:
    Version A: single-regime Black-Scholes / GBM
    Version B: HMM-calibrated Markov-switching GBM
    Version C: Markov-switching GBM with observable regime proxy features

Inputs:
    data/processed/transitions_daily_top1_final_with_spy_2010_2023.parquet
    outputs/hmm_regime_params_pilot.npz
    data/processed/transitions_daily_top1_final_with_spy_2010_2023_with_regime_proxies.parquet
        created by 08a

Outputs:
    data/processed/sim_bs_pretrain_pilot.parquet
    data/processed/sim_ms_gbm_pretrain_pilot.parquet
    data/processed/sim_ms_gbm_proxy_pretrain_pilot.parquet
    outputs/simulator_validation_pilot.xlsx

Run:
    py src\\08b_simulate_bs_ms_gbm_pilot.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import math
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

TRANSITIONS_FILE = "transitions_daily_top1_final_with_spy_2010_2023.parquet"
REAL_PROXY_FILE = "transitions_daily_top1_final_with_spy_2010_2023_with_regime_proxies.parquet"
HMM_NPZ_FILE = "hmm_regime_params_pilot.npz"


def _candidate_project_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, Path.cwd(), Path.cwd().parent, Path("/mnt/data")]
    out = []
    for p in candidates:
        p = p.resolve()
        if p not in out:
            out.append(p)
    return out


def find_existing_file(filename: str, relative_dirs: Iterable[str]) -> Path:
    checked = []
    for base in _candidate_project_dirs():
        for rel in relative_dirs:
            p = base / rel / filename if rel else base / filename
            checked.append(p)
            if p.exists():
                return p
    raise FileNotFoundError("Could not find " + filename + ". Checked:\n" + "\n".join(map(str, checked)))


TRANSITIONS_PATH = find_existing_file(TRANSITIONS_FILE, ["data/processed", "processed", ""])
REAL_PROXY_PATH = find_existing_file(REAL_PROXY_FILE, ["data/processed", "processed", ""])
HMM_NPZ_PATH = find_existing_file(HMM_NPZ_FILE, ["outputs", ""])

if (Path(__file__).resolve().parent.parent / "data" / "processed").exists():
    PROJECT_DIR = Path(__file__).resolve().parent.parent
elif (Path.cwd() / "data" / "processed").exists():
    PROJECT_DIR = Path.cwd()
else:
    PROJECT_DIR = Path("/mnt/data")

DATA_PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SIM_BS_PATH = DATA_PROCESSED_DIR / "sim_bs_pretrain_pilot.parquet"
SIM_MS_PATH = DATA_PROCESSED_DIR / "sim_ms_gbm_pretrain_pilot.parquet"
SIM_MS_PROXY_PATH = DATA_PROCESSED_DIR / "sim_ms_gbm_proxy_pretrain_pilot.parquet"
VALIDATION_XLSX = OUTPUT_DIR / "simulator_validation_pilot.xlsx"


# ============================================================
# CONFIG
# ============================================================

N_SIM_EPISODES = 5000
RANDOM_SEED = 123

TRADING_DAYS = 252
DT = 1.0 / TRADING_DAYS

RISK_FREE_RATE = 0.0
DIVIDEND_YIELD = 0.0

MIN_OPTION_PRICE = 0.01
DEFAULT_SPREAD_PCT = 0.01


# ============================================================
# BS FUNCTIONS
# ============================================================

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
        gamma = 0.0
        vega = 0.0
        theta = 0.0
        return price, delta, gamma, vega, theta

    sqrt_tau = math.sqrt(tau)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau

    disc_q = math.exp(-q * tau)
    disc_r = math.exp(-r * tau)

    price = S * disc_q * norm_cdf(d1) - K * disc_r * norm_cdf(d2)
    delta = disc_q * norm_cdf(d1)
    gamma = disc_q * norm_pdf(d1) / (S * sigma * sqrt_tau)
    vega = S * disc_q * norm_pdf(d1) * sqrt_tau / 100.0  # per 1 vol point
    theta = (
        -(S * disc_q * norm_pdf(d1) * sigma) / (2.0 * sqrt_tau)
        - r * K * disc_r * norm_cdf(d2)
        + q * S * disc_q * norm_cdf(d1)
    ) / TRADING_DAYS

    return max(price, MIN_OPTION_PRICE), float(delta), float(gamma), float(vega), float(theta)


# ============================================================
# FEATURE HELPERS
# ============================================================

def percentile_rank_from_train(values: pd.Series, train_values: pd.Series) -> pd.Series:
    train = pd.Series(train_values).dropna().sort_values().values
    if len(train) == 0:
        return pd.Series(np.nan, index=values.index)
    arr = pd.to_numeric(values, errors="coerce").values
    ranks = np.searchsorted(train, arr, side="right") / len(train)
    return pd.Series(ranks, index=values.index)


def add_proxy_features_to_sim(df: pd.DataFrame, real_proxy_train: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)

    out["SPY_LOG_RET_DAILY"] = np.log(out["SPY_CLOSE"].astype(float) / out.groupby("EPISODE_ID")["SPY_CLOSE"].shift(1).astype(float))
    out["SPY_LOG_RET_DAILY"] = out["SPY_LOG_RET_DAILY"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    out["ROLLING_RV_10D"] = (
        out.groupby("EPISODE_ID")["SPY_LOG_RET_DAILY"]
        .rolling(10, min_periods=3)
        .std()
        .reset_index(level=0, drop=True)
        * math.sqrt(TRADING_DAYS)
    )
    out["ROLLING_RV_20D"] = (
        out.groupby("EPISODE_ID")["SPY_LOG_RET_DAILY"]
        .rolling(20, min_periods=5)
        .std()
        .reset_index(level=0, drop=True)
        * math.sqrt(TRADING_DAYS)
    )
    out["ROLLING_ABS_RET_10D"] = (
        out.groupby("EPISODE_ID")["SPY_LOG_RET_DAILY"]
        .rolling(10, min_periods=3)
        .apply(lambda x: np.mean(np.abs(x)), raw=True)
        .reset_index(level=0, drop=True)
    )

    out["IV_LEVEL"] = out["OPTION_IV"].astype(float)
    out["SPREAD_PROXY"] = out["OPTION_SPREAD_PCT"].astype(float)

    for c in ["ROLLING_RV_10D", "ROLLING_RV_20D", "ROLLING_ABS_RET_10D"]:
        med = real_proxy_train[c].median() if c in real_proxy_train.columns else 0.0
        if pd.isna(med):
            med = 0.0
        out[c] = out[c].fillna(med)

    out["IV_PERCENTILE"] = percentile_rank_from_train(out["IV_LEVEL"], real_proxy_train["IV_LEVEL"])
    out["SPREAD_PERCENTILE"] = percentile_rank_from_train(out["SPREAD_PROXY"], real_proxy_train["SPREAD_PROXY"])

    for c in [
        "SPY_LOG_RET_DAILY",
        "ROLLING_RV_10D",
        "ROLLING_RV_20D",
        "ROLLING_ABS_RET_10D",
        "IV_LEVEL",
        "IV_PERCENTILE",
        "SPREAD_PROXY",
        "SPREAD_PERCENTILE",
    ]:
        out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return out


# ============================================================
# SIMULATION
# ============================================================

def build_episode_reference(real_train: pd.DataFrame):
    # Episode lengths and initial rows from real train data.
    ep_lengths = real_train.groupby("EPISODE_ID").size().values
    initial_rows = real_train.sort_values(["EPISODE_ID", "QUOTE_DATE"]).groupby("EPISODE_ID").head(1).reset_index(drop=True)

    iv_values = pd.to_numeric(real_train["OPTION_IV"], errors="coerce")
    iv_values = iv_values.where(iv_values <= 3.0, iv_values / 100.0)
    iv_median = float(iv_values.median())
    spread_median = float(pd.to_numeric(real_train.get("OPTION_SPREAD_PCT", pd.Series([DEFAULT_SPREAD_PCT])), errors="coerce").median())
    if not np.isfinite(spread_median):
        spread_median = DEFAULT_SPREAD_PCT

    return ep_lengths, initial_rows, iv_median, spread_median


def simulate_dataset(
    kind: str,
    real_train: pd.DataFrame,
    hmm_params: dict,
    n_episodes: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ep_lengths, initial_rows, train_iv_median, spread_median = build_episode_reference(real_train)

    # Single-regime BS volatility: train return realized vol or median IV.
    price = (
        real_train[["QUOTE_DATE", "SPY_CLOSE"]]
        .drop_duplicates("QUOTE_DATE")
        .sort_values("QUOTE_DATE")
        .reset_index(drop=True)
    )
    train_logret = np.log(price["SPY_CLOSE"].astype(float) / price["SPY_CLOSE"].astype(float).shift(1)).dropna()
    bs_sigma = float(train_logret.std() * math.sqrt(TRADING_DAYS))
    if not np.isfinite(bs_sigma) or bs_sigma <= 0:
        bs_sigma = train_iv_median
    bs_mu_daily = float(train_logret.mean()) if len(train_logret) else 0.0

    A = hmm_params["A"]
    pi = hmm_params["pi"]
    means = hmm_params["means"]
    sigmas = np.sqrt(hmm_params["variances"])

    all_rows = []
    base_date = pd.Timestamp("2000-01-03")

    for ep_idx in range(n_episodes):
        ref = initial_rows.iloc[int(rng.integers(0, len(initial_rows)))]

        S0 = float(ref["SPY_CLOSE"])
        m0 = float(ref["SPY_MONEYNESS"]) if "SPY_MONEYNESS" in ref and np.isfinite(ref["SPY_MONEYNESS"]) else 1.0
        if not np.isfinite(m0) or m0 <= 0:
            m0 = 1.0
        K = S0 / m0

        dte0 = int(max(10, min(60, float(ref["DTE"]) if "DTE" in ref else 30)))
        ep_len_sample = int(ep_lengths[int(rng.integers(0, len(ep_lengths)))])
        n_steps = int(max(5, min(dte0, ep_len_sample, 45)))

        S = [S0]
        regimes = []

        if kind == "bs":
            for t in range(n_steps):
                sigma = bs_sigma
                mu_daily = bs_mu_daily
                eps = rng.normal()
                next_S = S[-1] * math.exp((mu_daily - 0.5 * sigma * sigma * DT) + sigma * math.sqrt(DT) * eps)
                S.append(max(next_S, 1e-6))
                regimes.append(0)
        elif kind.startswith("ms"):
            z = int(rng.choice(len(pi), p=pi))
            for t in range(n_steps):
                sigma_ann = float(sigmas[z] * math.sqrt(TRADING_DAYS))
                mu_daily = float(means[z])
                eps = rng.normal()
                next_S = S[-1] * math.exp((mu_daily - 0.5 * sigma_ann * sigma_ann * DT) + sigma_ann * math.sqrt(DT) * eps)
                S.append(max(next_S, 1e-6))
                regimes.append(z)
                z = int(rng.choice(len(pi), p=A[z]))
        else:
            raise ValueError("kind must be 'bs' or 'ms'")

        prev_ret = 0.0
        for t in range(n_steps):
            dte = dte0 - t
            next_dte = max(dte - 1, 0)

            St = S[t]
            Snext = S[t + 1]
            reg = regimes[t]

            if kind == "bs":
                iv = bs_sigma
            else:
                iv = float(sigmas[reg] * math.sqrt(TRADING_DAYS))

            price_t, delta_t, gamma_t, vega_t, theta_t = bs_call_price_greeks(
                St, K, dte, iv, r=RISK_FREE_RATE, q=DIVIDEND_YIELD
            )
            price_next, _, _, _, _ = bs_call_price_greeks(
                Snext, K, next_dte, iv, r=RISK_FREE_RATE, q=DIVIDEND_YIELD
            )

            # Keep dates within pandas' valid timestamp range.
            # Simulated episodes do not need globally unique calendar dates; EPISODE_ID
            # separates episodes, and the environment sorts by EPISODE_ID and QUOTE_DATE.
            quote_date = base_date + pd.Timedelta(days=int(t))
            next_quote_date = base_date + pd.Timedelta(days=int(t + 1))

            row = {
                "EPISODE_ID": f"{kind}_pilot_{ep_idx:06d}",
                "SPLIT": "train",
                "QUOTE_DATE": quote_date,
                "NEXT_QUOTE_DATE": next_quote_date,
                "SPY_CLOSE": St,
                "SPY_NEXT_CLOSE": Snext,
                "SPY_DS": Snext - St,
                "STRIKE": K,
                "DTE": dte,
                "NEXT_DTE": next_dte,
                "OPTION_MID": price_t,
                "NEXT_OPTION_MID": price_next,
                "DOPTION": price_next - price_t,
                "OPTION_DELTA": delta_t,
                "OPTION_GAMMA": gamma_t,
                "OPTION_VEGA": vega_t,
                "OPTION_THETA": theta_t,
                "OPTION_IV": iv,
                "OPTION_SPREAD_PCT": spread_median,
                "SPY_MONEYNESS": St / K,
                "SPY_LOG_MONEYNESS": math.log(St / K),
                "OPTION_MID_OVER_SPY": price_t / St,
                "SPY_RET_LAG1": prev_ret,
                "SIM_REGIME_ID": reg,
                "SIM_REGIME_LABEL": ["low_vol", "medium_vol", "high_vol"][reg] if kind != "bs" else "single_bs",
            }
            all_rows.append(row)

            prev_ret = (Snext / St) - 1.0

    df = pd.DataFrame(all_rows)
    return df


def summarize_returns(df: pd.DataFrame, label: str) -> dict:
    # Use unique simulated transitions; for real data, duplicate dates are okay after drop_duplicates.
    if "QUOTE_DATE" in df.columns and label.startswith("Real"):
        tmp = df[["QUOTE_DATE", "SPY_CLOSE"]].drop_duplicates("QUOTE_DATE").sort_values("QUOTE_DATE")
        ret = np.log(tmp["SPY_CLOSE"].astype(float) / tmp["SPY_CLOSE"].astype(float).shift(1)).dropna()
    else:
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


# ============================================================
# MAIN
# ============================================================

print("Loading real transitions:")
print(TRANSITIONS_PATH)
real = pd.read_parquet(TRANSITIONS_PATH)
real_proxy = pd.read_parquet(REAL_PROXY_PATH)
real_train = real[real["SPLIT"].eq("train")].copy()
real_proxy_train = real_proxy[real_proxy["SPLIT"].eq("train")].copy()

print("Loading HMM parameters:")
print(HMM_NPZ_PATH)
hmm_npz = np.load(HMM_NPZ_PATH, allow_pickle=True)
hmm_params = {
    "pi": hmm_npz["pi"],
    "A": hmm_npz["A"],
    "means": hmm_npz["means"],
    "variances": hmm_npz["variances"],
}

print("Simulating BS pilot data...")
sim_bs = simulate_dataset("bs", real_train, hmm_params, N_SIM_EPISODES, RANDOM_SEED)
print("Simulating Markov-switching GBM pilot data...")
sim_ms = simulate_dataset("ms", real_train, hmm_params, N_SIM_EPISODES, RANDOM_SEED + 1)
sim_ms_proxy = add_proxy_features_to_sim(sim_ms, real_proxy_train)

sim_bs.to_parquet(SIM_BS_PATH, index=False)
sim_ms.to_parquet(SIM_MS_PATH, index=False)
sim_ms_proxy.to_parquet(SIM_MS_PROXY_PATH, index=False)

print("Saved:")
print(SIM_BS_PATH)
print(SIM_MS_PATH)
print(SIM_MS_PROXY_PATH)

# Validation summaries.
validation_rows = [
    summarize_returns(real_train, "Real train"),
    summarize_returns(sim_bs, "BS sim"),
    summarize_returns(sim_ms, "MS-GBM sim"),
]
return_summary = pd.DataFrame(validation_rows)

regime_counts = (
    sim_ms.groupby("SIM_REGIME_LABEL")
    .agg(
        ROWS=("SIM_REGIME_LABEL", "count"),
        MEAN_IV=("OPTION_IV", "mean"),
        MEAN_RET=("SPY_RET_LAG1", "mean"),
    )
    .reset_index()
)

episode_rv = []
for name, d in [("BS sim", sim_bs), ("MS-GBM sim", sim_ms)]:
    rv = (
        d.assign(LOG_RET=np.log(d["SPY_NEXT_CLOSE"] / d["SPY_CLOSE"]))
        .groupby("EPISODE_ID")["LOG_RET"]
        .std()
        .mul(math.sqrt(TRADING_DAYS))
        .reset_index(name="EPISODE_RV")
    )
    rv["DATASET"] = name
    episode_rv.append(rv)
episode_rv = pd.concat(episode_rv, ignore_index=True)

episode_rv_summary = (
    episode_rv.groupby("DATASET")
    .agg(
        EPISODES=("EPISODE_ID", "nunique"),
        MEAN_EPISODE_RV=("EPISODE_RV", "mean"),
        STD_EPISODE_RV=("EPISODE_RV", "std"),
        Q05_EPISODE_RV=("EPISODE_RV", lambda x: pd.Series(x).quantile(0.05)),
        Q50_EPISODE_RV=("EPISODE_RV", lambda x: pd.Series(x).quantile(0.50)),
        Q95_EPISODE_RV=("EPISODE_RV", lambda x: pd.Series(x).quantile(0.95)),
    )
    .reset_index()
)

with pd.ExcelWriter(VALIDATION_XLSX, engine="openpyxl") as writer:
    return_summary.to_excel(writer, sheet_name="Return_Stats", index=False)
    regime_counts.to_excel(writer, sheet_name="MS_Regime_Counts", index=False)
    episode_rv_summary.to_excel(writer, sheet_name="Episode_RV_Summary", index=False)
    sim_bs.head(2000).to_excel(writer, sheet_name="BS_Sample", index=False)
    sim_ms.head(2000).to_excel(writer, sheet_name="MS_GBM_Sample", index=False)

print("Saved simulator validation:")
print(VALIDATION_XLSX)
print(return_summary)
