"""
08a_fit_hmm_regimes_pilot.py

Pilot HMM calibration for regime-switching pretraining.

What it does:
    1. Loads the real SPY option transition dataset.
    2. Extracts unique daily SPY prices from the TRAIN split only.
    3. Fits a 3-state Gaussian HMM to train log returns using a small
       self-contained EM implementation.
    4. Sorts regimes by volatility: low / medium / high.
    5. Saves HMM parameters and train regime classification.
    6. Builds a real transition dataset with observable regime proxy features
       for later "Version C" regime-proxy pretraining.

Outputs:
    outputs/hmm_regime_params_pilot.xlsx
    outputs/hmm_regime_params_pilot.npz
    data/processed/transitions_daily_top1_final_with_spy_2010_2023_with_regime_proxies.parquet

Run:
    py src\\08a_fit_hmm_regimes_pilot.py
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

HMM_XLSX_PATH = OUTPUT_DIR / "hmm_regime_params_pilot.xlsx"
HMM_NPZ_PATH = OUTPUT_DIR / "hmm_regime_params_pilot.npz"
REAL_PROXY_PATH = DATA_PROCESSED_DIR / "transitions_daily_top1_final_with_spy_2010_2023_with_regime_proxies.parquet"


# ============================================================
# CONFIG
# ============================================================

N_REGIMES = 3
MAX_ITER = 250
TOL = 1e-7
RANDOM_RESTARTS = 5
RANDOM_SEED = 42

TRADING_DAYS = 252


# ============================================================
# HMM IMPLEMENTATION
# ============================================================

def gaussian_pdf_1d(x: np.ndarray, means: np.ndarray, variances: np.ndarray) -> np.ndarray:
    """Return B[t, k] = N(x_t | mean_k, var_k)."""
    x = x.reshape(-1, 1)
    means = means.reshape(1, -1)
    variances = np.maximum(variances.reshape(1, -1), 1e-12)
    z = (x - means) ** 2 / variances
    b = np.exp(-0.5 * z) / np.sqrt(2.0 * np.pi * variances)
    return np.maximum(b, 1e-300)


def forward_backward_scaled(x: np.ndarray, pi: np.ndarray, A: np.ndarray, means: np.ndarray, variances: np.ndarray):
    T = len(x)
    K = len(pi)

    B = gaussian_pdf_1d(x, means, variances)

    alpha = np.zeros((T, K))
    beta = np.zeros((T, K))
    scales = np.zeros(T)

    alpha[0] = pi * B[0]
    scales[0] = alpha[0].sum()
    if scales[0] <= 0:
        scales[0] = 1e-300
    alpha[0] /= scales[0]

    for t in range(1, T):
        alpha[t] = (alpha[t - 1] @ A) * B[t]
        scales[t] = alpha[t].sum()
        if scales[t] <= 0:
            scales[t] = 1e-300
        alpha[t] /= scales[t]

    beta[-1] = 1.0
    for t in range(T - 2, -1, -1):
        beta[t] = A @ (B[t + 1] * beta[t + 1])
        beta[t] /= scales[t + 1]

    gamma = alpha * beta
    gamma /= np.maximum(gamma.sum(axis=1, keepdims=True), 1e-300)

    xi = np.zeros((T - 1, K, K))
    for t in range(T - 1):
        numerator = alpha[t, :, None] * A * (B[t + 1] * beta[t + 1])[None, :]
        denom = numerator.sum()
        if denom <= 0:
            denom = 1e-300
        xi[t] = numerator / denom

    loglik = float(np.sum(np.log(scales)))
    return gamma, xi, loglik


def initialize_hmm(x: np.ndarray, K: int, rng: np.random.Generator):
    # Initialize by grouping observations according to absolute return quantiles.
    abs_x = np.abs(x)
    qs = np.quantile(abs_x, np.linspace(0, 1, K + 1))
    means = np.zeros(K)
    variances = np.zeros(K)

    for k in range(K):
        if k == K - 1:
            mask = (abs_x >= qs[k]) & (abs_x <= qs[k + 1])
        else:
            mask = (abs_x >= qs[k]) & (abs_x < qs[k + 1])
        if mask.sum() < 5:
            sample = x
        else:
            sample = x[mask]
        means[k] = sample.mean() + rng.normal(0, x.std() * 0.05)
        variances[k] = max(sample.var(), x.var() * 0.05, 1e-8)

    # High diagonal transition matrix to encourage persistent regimes.
    A = np.full((K, K), 0.05 / max(K - 1, 1))
    np.fill_diagonal(A, 0.95)
    A = A / A.sum(axis=1, keepdims=True)

    pi = np.full(K, 1.0 / K)
    return pi, A, means, variances


def fit_gaussian_hmm_1d(x: np.ndarray, K: int, max_iter: int, tol: float, n_restarts: int, seed: int):
    rng = np.random.default_rng(seed)
    best = None

    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    for restart in range(n_restarts):
        pi, A, means, variances = initialize_hmm(x, K, rng)

        rows = []
        prev_loglik = -np.inf

        for it in range(max_iter):
            gamma, xi, loglik = forward_backward_scaled(x, pi, A, means, variances)

            weights = gamma.sum(axis=0)
            weights = np.maximum(weights, 1e-12)

            pi = gamma[0]
            A = xi.sum(axis=0) / np.maximum(gamma[:-1].sum(axis=0)[:, None], 1e-12)
            A = A / A.sum(axis=1, keepdims=True)

            means = (gamma * x[:, None]).sum(axis=0) / weights
            variances = (gamma * (x[:, None] - means[None, :]) ** 2).sum(axis=0) / weights
            variances = np.maximum(variances, 1e-10)

            rows.append({"RESTART": restart, "ITER": it, "LOGLIK": loglik})

            if abs(loglik - prev_loglik) < tol:
                break
            prev_loglik = loglik

        gamma, xi, loglik = forward_backward_scaled(x, pi, A, means, variances)
        result = {
            "restart": restart,
            "pi": pi.copy(),
            "A": A.copy(),
            "means": means.copy(),
            "variances": variances.copy(),
            "gamma": gamma.copy(),
            "loglik": loglik,
            "fit_log": pd.DataFrame(rows),
        }
        if best is None or result["loglik"] > best["loglik"]:
            best = result

    return best


def sort_hmm_by_vol(params: dict) -> dict:
    sigmas = np.sqrt(params["variances"])
    order = np.argsort(sigmas)

    sorted_params = {
        "pi": params["pi"][order],
        "A": params["A"][np.ix_(order, order)],
        "means": params["means"][order],
        "variances": params["variances"][order],
        "gamma": params["gamma"][:, order],
        "loglik": params["loglik"],
        "fit_log": params["fit_log"],
        "order": order,
    }
    sorted_params["pi"] = sorted_params["pi"] / sorted_params["pi"].sum()
    sorted_params["A"] = sorted_params["A"] / sorted_params["A"].sum(axis=1, keepdims=True)
    return sorted_params


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
        for k in range(K):
            vals = delta[t - 1] + logA[:, k]
            psi[t, k] = int(np.argmax(vals))
            delta[t, k] = vals[psi[t, k]] + logB[t, k]

    states = np.zeros(T, dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


# ============================================================
# PROXY FEATURES
# ============================================================

def percentile_rank_from_train(values: pd.Series, train_values: pd.Series) -> pd.Series:
    train = pd.Series(train_values).dropna().sort_values().values
    if len(train) == 0:
        return pd.Series(np.nan, index=values.index)
    arr = pd.to_numeric(values, errors="coerce").values
    ranks = np.searchsorted(train, arr, side="right") / len(train)
    return pd.Series(ranks, index=values.index)


def add_real_regime_proxies(transitions: pd.DataFrame) -> pd.DataFrame:
    df = transitions.copy()
    df = df.sort_values(["QUOTE_DATE", "EPISODE_ID"]).reset_index(drop=True)

    # Build unique daily SPY price table.
    price = (
        df[["QUOTE_DATE", "SPY_CLOSE"]]
        .drop_duplicates("QUOTE_DATE")
        .sort_values("QUOTE_DATE")
        .reset_index(drop=True)
    )
    price["SPY_LOG_RET_DAILY"] = np.log(price["SPY_CLOSE"].astype(float) / price["SPY_CLOSE"].astype(float).shift(1))
    price["ROLLING_RV_10D"] = price["SPY_LOG_RET_DAILY"].rolling(10, min_periods=3).std() * math.sqrt(TRADING_DAYS)
    price["ROLLING_RV_20D"] = price["SPY_LOG_RET_DAILY"].rolling(20, min_periods=5).std() * math.sqrt(TRADING_DAYS)
    price["ROLLING_ABS_RET_10D"] = price["SPY_LOG_RET_DAILY"].abs().rolling(10, min_periods=3).mean()
    price[["ROLLING_RV_10D", "ROLLING_RV_20D", "ROLLING_ABS_RET_10D"]] = price[
        ["ROLLING_RV_10D", "ROLLING_RV_20D", "ROLLING_ABS_RET_10D"]
    ].fillna(method="bfill").fillna(0.0)

    df = df.merge(
        price[["QUOTE_DATE", "SPY_LOG_RET_DAILY", "ROLLING_RV_10D", "ROLLING_RV_20D", "ROLLING_ABS_RET_10D"]],
        on="QUOTE_DATE",
        how="left",
    )

    if "OPTION_IV" in df.columns:
        df["IV_LEVEL"] = pd.to_numeric(df["OPTION_IV"], errors="coerce")
        df.loc[df["IV_LEVEL"] > 3.0, "IV_LEVEL"] = df.loc[df["IV_LEVEL"] > 3.0, "IV_LEVEL"] / 100.0
    else:
        df["IV_LEVEL"] = np.nan

    if "OPTION_SPREAD_PCT" in df.columns:
        df["SPREAD_PROXY"] = pd.to_numeric(df["OPTION_SPREAD_PCT"], errors="coerce")
    else:
        df["SPREAD_PROXY"] = 0.0

    train_mask = df["SPLIT"].eq("train")
    df["IV_PERCENTILE"] = percentile_rank_from_train(df["IV_LEVEL"], df.loc[train_mask, "IV_LEVEL"])
    df["SPREAD_PERCENTILE"] = percentile_rank_from_train(df["SPREAD_PROXY"], df.loc[train_mask, "SPREAD_PROXY"])

    proxy_cols = [
        "SPY_LOG_RET_DAILY",
        "ROLLING_RV_10D",
        "ROLLING_RV_20D",
        "ROLLING_ABS_RET_10D",
        "IV_LEVEL",
        "IV_PERCENTILE",
        "SPREAD_PROXY",
        "SPREAD_PERCENTILE",
    ]

    for c in proxy_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        train_median = df.loc[train_mask, c].median()
        if pd.isna(train_median):
            train_median = 0.0
        df[c] = df[c].fillna(train_median)

    return df


# ============================================================
# MAIN
# ============================================================

print("Loading transitions:")
print(TRANSITIONS_PATH)
transitions = pd.read_parquet(TRANSITIONS_PATH)
transitions = transitions.sort_values(["QUOTE_DATE", "EPISODE_ID"]).reset_index(drop=True)

# Unique train daily SPY prices.
train_dates = transitions[transitions["SPLIT"].eq("train")]
daily_train = (
    train_dates[["QUOTE_DATE", "SPY_CLOSE"]]
    .drop_duplicates("QUOTE_DATE")
    .sort_values("QUOTE_DATE")
    .reset_index(drop=True)
)
daily_train["LOG_RET"] = np.log(
    daily_train["SPY_CLOSE"].astype(float) / daily_train["SPY_CLOSE"].astype(float).shift(1)
)
daily_train = daily_train.dropna(subset=["LOG_RET"]).reset_index(drop=True)

returns = daily_train["LOG_RET"].astype(float).values
print(f"Train daily observations for HMM: {len(returns)}")

best = fit_gaussian_hmm_1d(
    x=returns,
    K=N_REGIMES,
    max_iter=MAX_ITER,
    tol=TOL,
    n_restarts=RANDOM_RESTARTS,
    seed=RANDOM_SEED,
)
hmm = sort_hmm_by_vol(best)

states = viterbi(returns, hmm["pi"], hmm["A"], hmm["means"], hmm["variances"])
gamma = hmm["gamma"]

labels = ["low_vol", "medium_vol", "high_vol"]
regime_params = pd.DataFrame({
    "REGIME_ID": np.arange(N_REGIMES),
    "REGIME_LABEL": labels[:N_REGIMES],
    "MU_DAILY": hmm["means"],
    "SIGMA_DAILY": np.sqrt(hmm["variances"]),
    "MU_ANNUALIZED": hmm["means"] * TRADING_DAYS,
    "SIGMA_ANNUALIZED": np.sqrt(hmm["variances"]) * math.sqrt(TRADING_DAYS),
    "INITIAL_PROB": hmm["pi"],
})

transition_matrix = pd.DataFrame(
    hmm["A"],
    columns=[f"TO_{labels[j]}" for j in range(N_REGIMES)],
)
transition_matrix.insert(0, "FROM_REGIME", labels[:N_REGIMES])

classification = daily_train.copy()
classification["VITERBI_REGIME_ID"] = states
classification["VITERBI_REGIME_LABEL"] = [labels[i] for i in states]
for k in range(N_REGIMES):
    classification[f"POSTERIOR_{labels[k]}"] = gamma[:, k]

fit_log = hmm["fit_log"]
fit_summary = pd.DataFrame([{
    "N_REGIMES": N_REGIMES,
    "MAX_ITER": MAX_ITER,
    "TOL": TOL,
    "RANDOM_RESTARTS": RANDOM_RESTARTS,
    "BEST_LOGLIK": hmm["loglik"],
    "TRAIN_RETURN_MEAN_DAILY": returns.mean(),
    "TRAIN_RETURN_SIGMA_DAILY": returns.std(),
    "TRAIN_RETURN_SIGMA_ANNUALIZED": returns.std() * math.sqrt(TRADING_DAYS),
}])

# Save HMM parameters.
np.savez(
    HMM_NPZ_PATH,
    pi=hmm["pi"],
    A=hmm["A"],
    means=hmm["means"],
    variances=hmm["variances"],
    labels=np.array(labels[:N_REGIMES]),
    trading_days=np.array([TRADING_DAYS]),
)

with pd.ExcelWriter(HMM_XLSX_PATH, engine="openpyxl") as writer:
    regime_params.to_excel(writer, sheet_name="Regime_Params", index=False)
    transition_matrix.to_excel(writer, sheet_name="Transition_Matrix", index=False)
    classification.to_excel(writer, sheet_name="Train_Regime_Classification", index=False)
    fit_summary.to_excel(writer, sheet_name="Fit_Summary", index=False)
    fit_log.to_excel(writer, sheet_name="Fit_Log", index=False)

print("Saved HMM parameters:")
print(HMM_XLSX_PATH)
print(HMM_NPZ_PATH)

# Add observable regime proxy features to real transitions.
real_proxy = add_real_regime_proxies(transitions)
real_proxy.to_parquet(REAL_PROXY_PATH, index=False)

print("Saved real transitions with regime proxy features:")
print(REAL_PROXY_PATH)

print("\nRegime parameters:")
print(regime_params)

print("\nTransition matrix:")
print(transition_matrix)
