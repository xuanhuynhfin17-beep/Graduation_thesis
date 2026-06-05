import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

transitions = pd.read_parquet(
    DATA_DIR / "transitions_daily_top1_final_with_spy_2010_2023.parquet"
)

CONTRACT_MULTIPLIER = 100
TRANSACTION_COST_RATE = 0.0005


def run_baseline_v2(transitions_df, strategy="no_hedge"):
    df = transitions_df.copy()
    df = df.sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)

    if strategy == "no_hedge":
        df["HEDGE"] = 0.0
    elif strategy == "delta":
        df["HEDGE"] = df["OPTION_DELTA"].astype(float)
    else:
        raise ValueError("strategy must be 'no_hedge' or 'delta'")

    df["PREV_HEDGE"] = df.groupby("EPISODE_ID")["HEDGE"].shift(1).fillna(0.0)
    df["TRADE_SIZE"] = df["HEDGE"] - df["PREV_HEDGE"]

    df["TC_REBALANCE_PER_SHARE"] = (
        TRANSACTION_COST_RATE
        * df["SPY_CLOSE"]
        * df["TRADE_SIZE"].abs()
    )

    df["OPTION_PNL_PER_SHARE"] = -df["DOPTION"]
    df["STOCK_PNL_PER_SHARE"] = df["HEDGE"] * df["SPY_DS"]

    df["STEP_PNL_PER_SHARE_BEFORE_FINAL_LIQ"] = (
        df["OPTION_PNL_PER_SHARE"]
        + df["STOCK_PNL_PER_SHARE"]
        - df["TC_REBALANCE_PER_SHARE"]
    )

    df["IS_LAST_TRANSITION"] = (
        df.groupby("EPISODE_ID").cumcount()
        == df.groupby("EPISODE_ID")["EPISODE_ID"].transform("count") - 1
    )

    df["TC_FINAL_LIQ_PER_SHARE"] = 0.0
    last_mask = df["IS_LAST_TRANSITION"]

    df.loc[last_mask, "TC_FINAL_LIQ_PER_SHARE"] = (
        TRANSACTION_COST_RATE
        * df.loc[last_mask, "SPY_NEXT_CLOSE"]
        * df.loc[last_mask, "HEDGE"].abs()
    )

    df["STEP_PNL_PER_SHARE"] = (
        df["STEP_PNL_PER_SHARE_BEFORE_FINAL_LIQ"]
        - df["TC_FINAL_LIQ_PER_SHARE"]
    )

    df["STEP_PNL"] = df["STEP_PNL_PER_SHARE"] * CONTRACT_MULTIPLIER

    df["TRANSACTION_COST"] = (
        df["TC_REBALANCE_PER_SHARE"]
        + df["TC_FINAL_LIQ_PER_SHARE"]
    ) * CONTRACT_MULTIPLIER

    df["TURNOVER"] = df["TRADE_SIZE"].abs()
    df.loc[last_mask, "TURNOVER"] += df.loc[last_mask, "HEDGE"].abs()

    episode_result = (
        df.groupby(["EPISODE_ID", "SPLIT"])
        .agg(
            START_DATE=("QUOTE_DATE", "first"),
            END_DATE=("NEXT_QUOTE_DATE", "last"),
            N_STEPS=("STEP_PNL", "count"),
            TERMINAL_PNL=("STEP_PNL", "sum"),
            TOTAL_TC=("TRANSACTION_COST", "sum"),
            TOTAL_TURNOVER=("TURNOVER", "sum"),
            AVG_HEDGE=("HEDGE", "mean"),
            STD_HEDGE=("HEDGE", "std"),
        )
        .reset_index()
    )

    episode_result["STRATEGY"] = strategy
    return df, episode_result


def cvar_95(x):
    x = pd.Series(x).dropna()
    q = x.quantile(0.05)
    return x[x <= q].mean()


def sharpe_like(x):
    x = pd.Series(x).dropna()
    std = x.std()
    if std == 0 or pd.isna(std):
        return np.nan
    return x.mean() / std


nohedge_steps, nohedge_ep = run_baseline_v2(transitions, strategy="no_hedge")
delta_steps, delta_ep = run_baseline_v2(transitions, strategy="delta")

baseline_results = pd.concat([nohedge_ep, delta_ep], ignore_index=True)

metrics = (
    baseline_results
    .groupby(["SPLIT", "STRATEGY"])
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
    )
    .reset_index()
)

print(metrics)

baseline_results.to_parquet(
    OUTPUT_DIR / "baseline_episode_results_with_spy.parquet",
    index=False
)

with pd.ExcelWriter(
    OUTPUT_DIR / "baseline_metrics_final_with_spy.xlsx",
    engine="openpyxl"
) as writer:
    baseline_results.to_excel(writer, sheet_name="Episode_Results", index=False)
    metrics.to_excel(writer, sheet_name="Metrics", index=False)
print("\nSaved:")
print(OUTPUT_DIR / "baseline_metrics_final.xlsx")