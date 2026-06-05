"""
06_regime_based_evaluation_v3c_algorithms.py

Regime-based evaluation for residual-delta V3C algorithm comparison.

Purpose:
    Compare Delta vs PPO vs SAC vs TD3 across volatility regimes:
        - low volatility
        - medium volatility
        - high volatility

Inputs expected:
    1) data/processed/transitions_daily_top1_final_with_spy_2010_2023.parquet
    2) outputs/comparison_algorithms_residual_delta_v3c_ppo_multiseed.xlsx
    3) outputs/comparison_algorithms_residual_delta_v3c_sac_multiseed.xlsx
    4) outputs/comparison_algorithms_residual_delta_v3c_td3_multiseed.xlsx

Output:
    outputs/regime_based_evaluation_v3c_algorithms.xlsx

Main idea:
    - Compute episode-level realized volatility and average implied volatility
      from the transition data.
    - Assign each episode to low / medium / high volatility regimes.
    - Merge regimes into Episode_Results from PPO/SAC/TD3 Excel outputs.
    - Recompute metrics by:
          SPLIT, VOL_REGIME, ALGORITHM, STRATEGY
      and separately by IV_REGIME.

Academic default:
    Regime cutoffs are estimated from TRAIN episodes only, then applied to
    train/val/test. This avoids defining regimes using test-set information.
    If you want balanced buckets within test only, set THRESHOLD_SOURCE = "test".
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# PATH HELPERS
# ============================================================

TRANSITIONS_FILE = "transitions_daily_top1_final_with_spy_2010_2023.parquet"

PPO_XLSX = "comparison_algorithms_residual_delta_v3c_ppo_multiseed.xlsx"
SAC_XLSX = "comparison_algorithms_residual_delta_v3c_sac_multiseed.xlsx"
TD3_XLSX = "comparison_algorithms_residual_delta_v3c_td3_multiseed.xlsx"

OUTPUT_XLSX = "regime_based_evaluation_v3c_algorithms.xlsx"

# Use train thresholds for clean out-of-sample regime definition.
# Options: "train", "test", "all"
THRESHOLD_SOURCE = "train"

# Which regime grouping to treat as the main one in plots/tables.
# Options: "RV_REGIME" or "IV_REGIME"
MAIN_REGIME_COL = "RV_REGIME"


def _candidate_project_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    candidates = [
        here.parent,
        here.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
        Path("/mnt/data"),
    ]
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
    checked_msg = "\n".join(str(p) for p in checked)
    raise FileNotFoundError(f"Could not find {filename}. Checked:\n{checked_msg}")


TRANSITIONS_PATH = find_existing_file(
    TRANSITIONS_FILE,
    relative_dirs=["data/processed", "processed", ""],
)

PPO_PATH = find_existing_file(
    PPO_XLSX,
    relative_dirs=["outputs", ""],
)
SAC_PATH = find_existing_file(
    SAC_XLSX,
    relative_dirs=["outputs", ""],
)
TD3_PATH = find_existing_file(
    TD3_XLSX,
    relative_dirs=["outputs", ""],
)

if (Path(__file__).resolve().parent.parent / "data" / "processed").exists():
    PROJECT_DIR = Path(__file__).resolve().parent.parent
elif (Path.cwd() / "data" / "processed").exists():
    PROJECT_DIR = Path.cwd()
else:
    PROJECT_DIR = Path("/mnt/data")

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = OUTPUT_DIR / OUTPUT_XLSX


# ============================================================
# METRICS HELPERS
# ============================================================

def cvar_95(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    if x.empty:
        return np.nan
    q = x.quantile(0.05)
    tail = x[x <= q]
    return tail.mean() if not tail.empty else np.nan


def sharpe_like(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    std = x.std()
    if std == 0 or pd.isna(std):
        return np.nan
    return x.mean() / std


def make_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    # Make missing optional fields safe.
    optional_defaults = {
        "TOTAL_TC": np.nan,
        "TOTAL_TURNOVER": np.nan,
        "AVG_HEDGE": np.nan,
        "AVG_DELTA": np.nan,
        "AVG_ADJUSTMENT_FROM_DELTA": np.nan,
        "NO_TRADE_RATE": np.nan,
        "ACTION_NEAR_NEG1_RATE": np.nan,
        "ACTION_NEAR_POS1_RATE": np.nan,
        "TRAINING_TIME_MIN": np.nan,
    }

    d = df.copy()
    for c, default_value in optional_defaults.items():
        if c not in d.columns:
            d[c] = default_value

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
            AVG_DELTA=("AVG_DELTA", "mean"),
            AVG_ADJUSTMENT_FROM_DELTA=("AVG_ADJUSTMENT_FROM_DELTA", "mean"),
            NO_TRADE_RATE=("NO_TRADE_RATE", "mean"),
            ACTION_NEAR_NEG1_RATE=("ACTION_NEAR_NEG1_RATE", "mean"),
            ACTION_NEAR_POS1_RATE=("ACTION_NEAR_POS1_RATE", "mean"),
            TRAINING_TIME_MIN=("TRAINING_TIME_MIN", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def make_seed_summary(metrics_by_seed_regime: pd.DataFrame, regime_col: str) -> pd.DataFrame:
    # Only RL rows have valid seeds.
    d = metrics_by_seed_regime.dropna(subset=["SEED"]).copy()
    if d.empty:
        return pd.DataFrame()

    return (
        d.groupby(["ALGORITHM", "SPLIT", regime_col], dropna=False)
        .agg(
            N_SEEDS=("SEED", "nunique"),
            MEAN_OF_MEAN_PNL=("MEAN_PNL", "mean"),
            STD_OF_MEAN_PNL=("MEAN_PNL", "std"),
            MEAN_OF_CVAR_95=("CVAR_95", "mean"),
            STD_OF_CVAR_95=("CVAR_95", "std"),
            MEAN_OF_SHARPE_LIKE=("SHARPE_LIKE", "mean"),
            STD_OF_SHARPE_LIKE=("SHARPE_LIKE", "std"),
            MEAN_TC=("MEAN_TC", "mean"),
            MEAN_TURNOVER=("MEAN_TURNOVER", "mean"),
            AVG_HEDGE=("AVG_HEDGE", "mean"),
            AVG_ADJUSTMENT_FROM_DELTA=("AVG_ADJUSTMENT_FROM_DELTA", "mean"),
            NO_TRADE_RATE=("NO_TRADE_RATE", "mean"),
            TRAINING_TIME_MIN=("TRAINING_TIME_MIN", "mean"),
        )
        .reset_index()
        .sort_values(["SPLIT", regime_col, "ALGORITHM"])
    )


# ============================================================
# LOAD DATA
# ============================================================

print("Loading transitions:")
print(TRANSITIONS_PATH)

transitions = pd.read_parquet(TRANSITIONS_PATH)
transitions = transitions.sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)

print("Loading Excel Episode_Results:")
print(PPO_PATH)
print(SAC_PATH)
print(TD3_PATH)


def read_episode_results(path: Path, algorithm_hint: str) -> pd.DataFrame:
    d = pd.read_excel(path, sheet_name="Episode_Results")

    # Ensure ALGORITHM column exists.
    if "ALGORITHM" not in d.columns:
        d["ALGORITHM"] = algorithm_hint

    # Some older outputs may have baseline duplicated in each algorithm file.
    # Keep as-is for now; we will deduplicate later.
    return d


ppo_ep = read_episode_results(PPO_PATH, "PPO")
sac_ep = read_episode_results(SAC_PATH, "SAC")
td3_ep = read_episode_results(TD3_PATH, "TD3")

all_ep_raw = pd.concat([ppo_ep, sac_ep, td3_ep], ignore_index=True, sort=False)

# Normalize algorithm labels.
if "ALGORITHM" not in all_ep_raw.columns:
    all_ep_raw["ALGORITHM"] = np.nan

all_ep_raw["ALGORITHM"] = all_ep_raw["ALGORITHM"].fillna("baseline")
all_ep_raw["STRATEGY"] = all_ep_raw["STRATEGY"].astype(str)

# Deduplicate:
# - baseline rows appear in each Excel file; keep one copy per EPISODE/SPLIT/STRATEGY.
# - RL rows are unique by ALGORITHM/SEED/EPISODE/STRATEGY.
baseline_mask = all_ep_raw["ALGORITHM"].astype(str).str.lower().eq("baseline")
baseline_rows = (
    all_ep_raw[baseline_mask]
    .drop_duplicates(subset=["EPISODE_ID", "SPLIT", "STRATEGY"])
    .copy()
)

rl_rows = (
    all_ep_raw[~baseline_mask]
    .drop_duplicates(subset=["ALGORITHM", "SEED", "EPISODE_ID", "SPLIT", "STRATEGY"])
    .copy()
)

all_ep = pd.concat([baseline_rows, rl_rows], ignore_index=True, sort=False)

print("\nEpisode results loaded:")
print(all_ep.groupby(["ALGORITHM", "SPLIT"])["EPISODE_ID"].nunique())


# ============================================================
# COMPUTE EPISODE REGIMES FROM TRANSITIONS
# ============================================================

df = transitions.copy()

# Daily underlying return. Prefer SPY_NEXT_CLOSE / SPY_CLOSE - 1.
if {"SPY_CLOSE", "SPY_NEXT_CLOSE"}.issubset(df.columns):
    df["SPY_RET"] = df["SPY_NEXT_CLOSE"].astype(float) / df["SPY_CLOSE"].astype(float) - 1.0
elif {"SPY_DS", "SPY_CLOSE"}.issubset(df.columns):
    df["SPY_RET"] = df["SPY_DS"].astype(float) / df["SPY_CLOSE"].astype(float)
else:
    raise ValueError("Could not compute SPY return. Need SPY_NEXT_CLOSE/SPY_CLOSE or SPY_DS/SPY_CLOSE.")

# Clean IV. If IV is stored as 25 instead of 0.25, convert to decimal.
if "OPTION_IV" in df.columns:
    df["OPTION_IV_DECIMAL"] = pd.to_numeric(df["OPTION_IV"], errors="coerce")
    df.loc[df["OPTION_IV_DECIMAL"] > 3.0, "OPTION_IV_DECIMAL"] = (
        df.loc[df["OPTION_IV_DECIMAL"] > 3.0, "OPTION_IV_DECIMAL"] / 100.0
    )
else:
    df["OPTION_IV_DECIMAL"] = np.nan

episode_regime = (
    df.groupby(["EPISODE_ID", "SPLIT"])
    .agg(
        START_DATE=("QUOTE_DATE", "first"),
        END_DATE=("NEXT_QUOTE_DATE", "last") if "NEXT_QUOTE_DATE" in df.columns else ("QUOTE_DATE", "last"),
        N_STEPS=("SPY_RET", "count"),
        MEAN_SPY_RET=("SPY_RET", "mean"),
        STD_DAILY_RET=("SPY_RET", "std"),
        REALIZED_VOL_ANN=("SPY_RET", lambda x: np.sqrt(252.0) * pd.Series(x).dropna().std()),
        AVG_IV=("OPTION_IV_DECIMAL", "mean"),
        AVG_DTE=("DTE", "mean") if "DTE" in df.columns else ("SPY_RET", "count"),
        AVG_ABS_LOG_MONEYNESS=("SPY_LOG_MONEYNESS", lambda x: pd.Series(x).abs().mean()) if "SPY_LOG_MONEYNESS" in df.columns else ("SPY_RET", "count"),
    )
    .reset_index()
)

# If realized vol for very short/constant episodes is NaN, use 0.
episode_regime["REALIZED_VOL_ANN"] = episode_regime["REALIZED_VOL_ANN"].fillna(0.0)


def compute_tercile_thresholds(values: pd.Series) -> tuple[float, float]:
    v = pd.Series(values).dropna()
    if v.empty:
        return np.nan, np.nan
    return float(v.quantile(1/3)), float(v.quantile(2/3))


def assign_regime(values: pd.Series, q_low: float, q_high: float) -> pd.Series:
    def label(x):
        if pd.isna(x):
            return "unknown"
        if x <= q_low:
            return "low"
        if x <= q_high:
            return "medium"
        return "high"

    return values.apply(label)


if THRESHOLD_SOURCE == "train":
    threshold_base = episode_regime[episode_regime["SPLIT"] == "train"]
elif THRESHOLD_SOURCE == "test":
    threshold_base = episode_regime[episode_regime["SPLIT"] == "test"]
elif THRESHOLD_SOURCE == "all":
    threshold_base = episode_regime
else:
    raise ValueError("THRESHOLD_SOURCE must be 'train', 'test', or 'all'.")

rv_low, rv_high = compute_tercile_thresholds(threshold_base["REALIZED_VOL_ANN"])
iv_low, iv_high = compute_tercile_thresholds(threshold_base["AVG_IV"])

episode_regime["RV_REGIME"] = assign_regime(episode_regime["REALIZED_VOL_ANN"], rv_low, rv_high)
episode_regime["IV_REGIME"] = assign_regime(episode_regime["AVG_IV"], iv_low, iv_high)

regime_thresholds = pd.DataFrame([
    {
        "REGIME_TYPE": "REALIZED_VOL_ANN",
        "THRESHOLD_SOURCE": THRESHOLD_SOURCE,
        "LOW_MAX": rv_low,
        "MEDIUM_MAX": rv_high,
        "HIGH_MIN_EXCLUSIVE": rv_high,
    },
    {
        "REGIME_TYPE": "AVG_IV",
        "THRESHOLD_SOURCE": THRESHOLD_SOURCE,
        "LOW_MAX": iv_low,
        "MEDIUM_MAX": iv_high,
        "HIGH_MIN_EXCLUSIVE": iv_high,
    },
])

regime_counts = (
    episode_regime
    .groupby(["SPLIT", "RV_REGIME", "IV_REGIME"], dropna=False)
    .size()
    .reset_index(name="EPISODES")
)

print("\nRegime thresholds:")
print(regime_thresholds)

print("\nRegime counts:")
print(regime_counts)


# ============================================================
# MERGE REGIMES INTO EPISODE RESULTS
# ============================================================

regime_cols = [
    "EPISODE_ID",
    "SPLIT",
    "REALIZED_VOL_ANN",
    "AVG_IV",
    "RV_REGIME",
    "IV_REGIME",
    "AVG_DTE",
    "AVG_ABS_LOG_MONEYNESS",
]

merged = all_ep.merge(
    episode_regime[regime_cols],
    on=["EPISODE_ID", "SPLIT"],
    how="left",
    validate="many_to_one",
)

missing_regime = merged["RV_REGIME"].isna().sum()
if missing_regime:
    print(f"Warning: {missing_regime} rows have missing regime labels after merge.")

# Put baseline strategy labels cleanly.
merged.loc[merged["STRATEGY"].eq("delta"), "ALGORITHM"] = "baseline"
merged.loc[merged["STRATEGY"].eq("no_hedge"), "ALGORITHM"] = "baseline"


# ============================================================
# REGIME METRICS
# ============================================================

# Overall metrics by algorithm and strategy, for reference.
overall_metrics = make_metrics(merged, ["ALGORITHM", "SPLIT", "STRATEGY"])

# Regime metrics by realized volatility.
rv_metrics = make_metrics(merged, ["SPLIT", "RV_REGIME", "ALGORITHM", "STRATEGY"])

# Regime metrics by average implied volatility.
iv_metrics = make_metrics(merged, ["SPLIT", "IV_REGIME", "ALGORITHM", "STRATEGY"])

# Metrics by seed and regime for RL algorithms.
rv_metrics_by_seed = make_metrics(
    merged,
    ["ALGORITHM", "SEED", "SPLIT", "RV_REGIME", "STRATEGY"],
)

iv_metrics_by_seed = make_metrics(
    merged,
    ["ALGORITHM", "SEED", "SPLIT", "IV_REGIME", "STRATEGY"],
)

rv_seed_summary = make_seed_summary(rv_metrics_by_seed, "RV_REGIME")
iv_seed_summary = make_seed_summary(iv_metrics_by_seed, "IV_REGIME")

# Test-only compact view for thesis tables.
test_rv_compact = rv_metrics[rv_metrics["SPLIT"].eq("test")].copy()
test_iv_compact = iv_metrics[iv_metrics["SPLIT"].eq("test")].copy()

# Create a ranking table for test / realized-vol regimes.
ranking_rows = []
for regime, g in test_rv_compact.groupby("RV_REGIME"):
    # Exclude no_hedge for algorithm ranking if present.
    gg = g[~g["STRATEGY"].eq("no_hedge")].copy()

    for metric, ascending in [
        ("MEAN_PNL", False),
        ("CVAR_95", False),
        ("SHARPE_LIKE", False),
        ("MEAN_TC", True),
        ("MEAN_TURNOVER", True),
    ]:
        ranked = gg.sort_values(metric, ascending=ascending)
        if not ranked.empty:
            top = ranked.iloc[0]
            ranking_rows.append({
                "SPLIT": "test",
                "RV_REGIME": regime,
                "METRIC": metric,
                "BEST_ALGORITHM": top["ALGORITHM"],
                "BEST_STRATEGY": top["STRATEGY"],
                "BEST_VALUE": top[metric],
            })

test_rv_rankings = pd.DataFrame(ranking_rows)


# ============================================================
# SAVE OUTPUT
# ============================================================

with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    merged.to_excel(writer, sheet_name="Episode_Results_With_Regime", index=False)
    episode_regime.to_excel(writer, sheet_name="Episode_Regime", index=False)
    regime_thresholds.to_excel(writer, sheet_name="Regime_Thresholds", index=False)
    regime_counts.to_excel(writer, sheet_name="Regime_Counts", index=False)

    overall_metrics.to_excel(writer, sheet_name="Overall_Metrics", index=False)
    rv_metrics.to_excel(writer, sheet_name="RV_Regime_Metrics", index=False)
    iv_metrics.to_excel(writer, sheet_name="IV_Regime_Metrics", index=False)

    rv_metrics_by_seed.to_excel(writer, sheet_name="RV_Metrics_By_Seed", index=False)
    iv_metrics_by_seed.to_excel(writer, sheet_name="IV_Metrics_By_Seed", index=False)

    rv_seed_summary.to_excel(writer, sheet_name="RV_Seed_Summary", index=False)
    iv_seed_summary.to_excel(writer, sheet_name="IV_Seed_Summary", index=False)

    test_rv_compact.to_excel(writer, sheet_name="Test_RV_Compact", index=False)
    test_iv_compact.to_excel(writer, sheet_name="Test_IV_Compact", index=False)
    test_rv_rankings.to_excel(writer, sheet_name="Test_RV_Rankings", index=False)

print("\nSaved regime-based evaluation:")
print(OUTPUT_PATH)

print("\nKey sheets:")
print("  - Test_RV_Compact")
print("  - RV_Regime_Metrics")
print("  - RV_Seed_Summary")
print("  - Test_RV_Rankings")
