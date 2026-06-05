"""
08d_compare_pretraining_pilot.py

Comparison and regime-based analysis for the regime-switching pretraining pilot.

Input:
    outputs/pretraining_regime_switching_pilot_ppo.xlsx
    data/processed/transitions_daily_top1_final_with_spy_2010_2023.parquet

Output:
    outputs/pretraining_regime_switching_pilot_comparison.xlsx

Run:
    py src\\08d_compare_pretraining_pilot.py
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

PRETRAIN_XLSX_FILE = "pretraining_regime_switching_pilot_ppo.xlsx"
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


PRETRAIN_XLSX_PATH = find_existing_file(PRETRAIN_XLSX_FILE, ["outputs", ""])
TRANSITIONS_PATH = find_existing_file(TRANSITIONS_FILE, ["data/processed", "processed", ""])

if (Path(__file__).resolve().parent.parent / "data" / "processed").exists():
    PROJECT_DIR = Path(__file__).resolve().parent.parent
elif (Path.cwd() / "data" / "processed").exists():
    PROJECT_DIR = Path.cwd()
else:
    PROJECT_DIR = Path("/mnt/data")

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_XLSX = OUTPUT_DIR / "pretraining_regime_switching_pilot_comparison.xlsx"


# ============================================================
# METRICS
# ============================================================

def cvar_95(x):
    x = pd.Series(x).dropna()
    if x.empty:
        return np.nan
    q = x.quantile(0.05)
    return x[x <= q].mean()


def sharpe_like(x):
    x = pd.Series(x).dropna()
    s = x.std()
    if pd.isna(s) or s == 0:
        return np.nan
    return x.mean() / s


def make_metrics(df, group_cols):
    d = df.copy()
    for c in [
        "TOTAL_TC", "TOTAL_TURNOVER", "AVG_HEDGE", "AVG_DELTA",
        "AVG_ADJUSTMENT_FROM_DELTA", "NO_TRADE_RATE", "TRAINING_TIME_MIN"
    ]:
        if c not in d.columns:
            d[c] = np.nan

    return (
        d.groupby(group_cols, dropna=False)
        .agg(
            EPISODES=("EPISODE_ID", "nunique"),
            MEAN_PNL=("TERMINAL_PNL", "mean"),
            STD_PNL=("TERMINAL_PNL", "std"),
            MEDIAN_PNL=("TERMINAL_PNL", "median"),
            MIN_PNL=("TERMINAL_PNL", "min"),
            MAX_PNL=("TERMINAL_PNL", "max"),
            CVAR_95=("TERMINAL_PNL", cvar_95),
            SHARPE_LIKE=("TERMINAL_PNL", sharpe_like),
            MEAN_TC=("TOTAL_TC", "mean"),
            MEAN_TURNOVER=("TOTAL_TURNOVER", "mean"),
            AVG_HEDGE=("AVG_HEDGE", "mean"),
            AVG_ADJUSTMENT_FROM_DELTA=("AVG_ADJUSTMENT_FROM_DELTA", "mean"),
            NO_TRADE_RATE=("NO_TRADE_RATE", "mean"),
            TRAINING_TIME_MIN=("TRAINING_TIME_MIN", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def summarize_across_seeds(metrics_by_seed):
    d = metrics_by_seed.dropna(subset=["SEED"]).copy()
    if d.empty:
        return pd.DataFrame()
    return (
        d.groupby(["EXPERIMENT", "SPLIT"])
        .agg(
            N_SEEDS=("SEED", "nunique"),
            MEAN_OF_MEAN_PNL=("MEAN_PNL", "mean"),
            STD_OF_MEAN_PNL=("MEAN_PNL", "std"),
            MEAN_OF_CVAR_95=("CVAR_95", "mean"),
            STD_OF_CVAR_95=("CVAR_95", "std"),
            MEAN_OF_SHARPE_LIKE=("SHARPE_LIKE", "mean"),
            MEAN_TC=("MEAN_TC", "mean"),
            MEAN_TURNOVER=("MEAN_TURNOVER", "mean"),
            AVG_HEDGE=("AVG_HEDGE", "mean"),
            NO_TRADE_RATE=("NO_TRADE_RATE", "mean"),
            TRAINING_TIME_MIN=("TRAINING_TIME_MIN", "mean"),
        )
        .reset_index()
        .sort_values(["SPLIT", "EXPERIMENT"])
    )


# ============================================================
# REGIME FEATURES
# ============================================================

def assign_tercile(series, q1, q2):
    def label(x):
        if pd.isna(x):
            return "unknown"
        if x <= q1:
            return "low"
        if x <= q2:
            return "medium"
        return "high"
    return series.apply(label)


def build_episode_regimes(transitions):
    df = transitions.copy().sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)

    if {"SPY_CLOSE", "SPY_NEXT_CLOSE"}.issubset(df.columns):
        df["SPY_RET"] = df["SPY_NEXT_CLOSE"].astype(float) / df["SPY_CLOSE"].astype(float) - 1.0
    elif {"SPY_DS", "SPY_CLOSE"}.issubset(df.columns):
        df["SPY_RET"] = df["SPY_DS"].astype(float) / df["SPY_CLOSE"].astype(float)
    else:
        raise ValueError("Cannot compute SPY returns.")

    if "OPTION_IV" in df.columns:
        df["OPTION_IV_DECIMAL"] = pd.to_numeric(df["OPTION_IV"], errors="coerce")
        mask = df["OPTION_IV_DECIMAL"] > 3.0
        df.loc[mask, "OPTION_IV_DECIMAL"] = df.loc[mask, "OPTION_IV_DECIMAL"] / 100.0
    else:
        df["OPTION_IV_DECIMAL"] = np.nan

    if "SPY_MONEYNESS" in df.columns:
        df["MONEYNESS"] = pd.to_numeric(df["SPY_MONEYNESS"], errors="coerce")
    elif {"SPY_CLOSE", "STRIKE"}.issubset(df.columns):
        df["MONEYNESS"] = df["SPY_CLOSE"].astype(float) / df["STRIKE"].astype(float)
    else:
        df["MONEYNESS"] = np.nan

    ep = (
        df.groupby(["EPISODE_ID", "SPLIT"])
        .agg(
            START_DATE=("QUOTE_DATE", "first"),
            END_DATE=("NEXT_QUOTE_DATE", "last") if "NEXT_QUOTE_DATE" in df.columns else ("QUOTE_DATE", "last"),
            N_STEPS=("SPY_RET", "count"),
            START_DTE=("DTE", "first") if "DTE" in df.columns else ("SPY_RET", "count"),
            AVG_DTE=("DTE", "mean") if "DTE" in df.columns else ("SPY_RET", "count"),
            AVG_MONEYNESS=("MONEYNESS", "mean"),
            REALIZED_VOL_ANN=("SPY_RET", lambda x: math.sqrt(252.0) * pd.Series(x).dropna().std()),
            AVG_IV=("OPTION_IV_DECIMAL", "mean"),
        )
        .reset_index()
    )
    ep["REALIZED_VOL_ANN"] = ep["REALIZED_VOL_ANN"].fillna(0.0)

    train = ep[ep["SPLIT"].eq("train")]
    rv1, rv2 = train["REALIZED_VOL_ANN"].quantile([1/3, 2/3]).values
    iv1, iv2 = train["AVG_IV"].quantile([1/3, 2/3]).values

    ep["RV_REGIME"] = assign_tercile(ep["REALIZED_VOL_ANN"], rv1, rv2)
    ep["IV_REGIME"] = assign_tercile(ep["AVG_IV"], iv1, iv2)

    def moneyness_label(m):
        if pd.isna(m):
            return "unknown"
        if m < 0.98:
            return "OTM"
        if m <= 1.02:
            return "ATM"
        return "ITM"

    def dte_label(d):
        if pd.isna(d):
            return "unknown"
        if d <= 14:
            return "short"
        if d <= 30:
            return "medium"
        return "long"

    ep["MONEYNESS_REGIME"] = ep["AVG_MONEYNESS"].apply(moneyness_label)
    ep["DTE_REGIME"] = ep["START_DTE"].apply(dte_label)

    thresholds = pd.DataFrame([
        {"REGIME": "RV_REGIME", "LOW_MAX": rv1, "MEDIUM_MAX": rv2, "SOURCE": "train"},
        {"REGIME": "IV_REGIME", "LOW_MAX": iv1, "MEDIUM_MAX": iv2, "SOURCE": "train"},
        {"REGIME": "MONEYNESS_REGIME", "LOW_MAX": 0.98, "MEDIUM_MAX": 1.02, "SOURCE": "fixed"},
        {"REGIME": "DTE_REGIME", "LOW_MAX": 14, "MEDIUM_MAX": 30, "SOURCE": "fixed"},
    ])

    return ep, thresholds


# ============================================================
# MAIN
# ============================================================

print("Loading pilot results:")
print(PRETRAIN_XLSX_PATH)
episodes = pd.read_excel(PRETRAIN_XLSX_PATH, sheet_name="Episode_Results")
metrics_by_seed_existing = pd.read_excel(PRETRAIN_XLSX_PATH, sheet_name="Metrics_By_Seed")

print("Loading real transitions:")
print(TRANSITIONS_PATH)
transitions = pd.read_parquet(TRANSITIONS_PATH)

episode_regimes, regime_thresholds = build_episode_regimes(transitions)

merged = episodes.merge(
    episode_regimes[[
        "EPISODE_ID",
        "SPLIT",
        "REALIZED_VOL_ANN",
        "AVG_IV",
        "AVG_MONEYNESS",
        "START_DTE",
        "RV_REGIME",
        "IV_REGIME",
        "MONEYNESS_REGIME",
        "DTE_REGIME",
    ]],
    on=["EPISODE_ID", "SPLIT"],
    how="left",
)

overall_metrics = make_metrics(merged, ["EXPERIMENT", "ALGORITHM", "SPLIT", "STRATEGY"])
metrics_by_seed = make_metrics(merged, ["EXPERIMENT", "ALGORITHM", "SEED", "SPLIT", "STRATEGY"])
summary = summarize_across_seeds(metrics_by_seed)

regime_dims = ["RV_REGIME", "IV_REGIME", "MONEYNESS_REGIME", "DTE_REGIME"]
regime_metrics = {}
for col in regime_dims:
    regime_metrics[col] = make_metrics(merged, ["EXPERIMENT", "ALGORITHM", "SPLIT", col, "STRATEGY"])

# Compact test table.
test_summary = summary[summary["SPLIT"].eq("test")].copy()
baseline_test = overall_metrics[
    overall_metrics["SPLIT"].eq("test")
    & overall_metrics["ALGORITHM"].astype(str).str.lower().eq("baseline")
].copy()

# Differences vs E0 no pretraining for RL experiments.
e0 = test_summary[test_summary["EXPERIMENT"].eq("E0_no_pretrain")]
if not e0.empty:
    base = e0.iloc[0]
    test_summary["DELTA_MEAN_PNL_VS_E0"] = test_summary["MEAN_OF_MEAN_PNL"] - base["MEAN_OF_MEAN_PNL"]
    test_summary["DELTA_CVAR_VS_E0"] = test_summary["MEAN_OF_CVAR_95"] - base["MEAN_OF_CVAR_95"]
    test_summary["DELTA_TC_VS_E0"] = test_summary["MEAN_TC"] - base["MEAN_TC"]
    test_summary["DELTA_TURNOVER_VS_E0"] = test_summary["MEAN_TURNOVER"] - base["MEAN_TURNOVER"]

# High-RV compact.
rv = regime_metrics["RV_REGIME"]
high_rv_test = rv[(rv["SPLIT"].eq("test")) & (rv["RV_REGIME"].eq("high"))].copy()

with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
    merged.to_excel(writer, sheet_name="Episode_Results_With_Regime", index=False)
    overall_metrics.to_excel(writer, sheet_name="Overall_Metrics", index=False)
    metrics_by_seed.to_excel(writer, sheet_name="Metrics_By_Seed_Recalc", index=False)
    summary.to_excel(writer, sheet_name="Experiment_Summary_Recalc", index=False)
    test_summary.to_excel(writer, sheet_name="Test_Summary", index=False)
    baseline_test.to_excel(writer, sheet_name="Baseline_Test", index=False)
    episode_regimes.to_excel(writer, sheet_name="Episode_Regimes", index=False)
    regime_thresholds.to_excel(writer, sheet_name="Regime_Thresholds", index=False)
    high_rv_test.to_excel(writer, sheet_name="High_RV_Test", index=False)
    for col, d in regime_metrics.items():
        d.to_excel(writer, sheet_name=col.replace("_REGIME", "") + "_Metrics", index=False)

print("Saved comparison:")
print(OUTPUT_XLSX)
print("\nTest summary:")
print(test_summary)
