"""
06b_extended_regime_evaluation_v3c_algorithms.py

Extended regime-based evaluation for the residual-delta V3C algorithm comparison.

Regime dimensions:
    1. Realized volatility regime: low / medium / high
    2. Implied volatility regime: low / medium / high
    3. Moneyness regime for call options: OTM / ATM / ITM
    4. DTE regime: short / medium / long

Main normal-cost inputs expected in outputs/:
    - comparison_algorithms_residual_delta_v3c_ppo_multiseed.xlsx
    - comparison_algorithms_residual_delta_v3c_sac_multiseed.xlsx
    - comparison_algorithms_residual_delta_v3c_td3_multiseed.xlsx

Optional input:
    - comparison_sac_v3c_paper_like_100k_3seeds.xlsx
    - comparison_sac_v3c_paper_like_100k_3seeds.xlsx.xlsx

The optional file is included as SAC-A tuned if found.

Output:
    outputs/extended_regime_evaluation_v3c_algorithms.xlsx

Academic default:
    RV and IV regime thresholds are estimated from TRAIN episodes only,
    then applied to train/val/test. This avoids defining regimes using
    test-set information.

Run:
    py src\\06b_extended_regime_evaluation_v3c_algorithms.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

TRANSITIONS_FILE = "transitions_daily_top1_final_with_spy_2010_2023.parquet"

REQUIRED_RESULT_FILES = {
    "PPO": "comparison_algorithms_residual_delta_v3c_ppo_multiseed.xlsx",
    "SAC": "comparison_algorithms_residual_delta_v3c_sac_multiseed.xlsx",
    "TD3": "comparison_algorithms_residual_delta_v3c_td3_multiseed.xlsx",
}

OPTIONAL_RESULT_FILES = {
    "SAC-A": [
        "comparison_sac_v3c_paper_like_100k_3seeds.xlsx",
        "comparison_sac_v3c_paper_like_100k_3seeds.xlsx.xlsx",
    ],
}

OUTPUT_XLSX = "extended_regime_evaluation_v3c_algorithms.xlsx"

# RV/IV terciles are estimated from this split and applied to all splits.
# Options: "train", "test", "all"
THRESHOLD_SOURCE = "train"

# Moneyness buckets for call options using average S/K over episode.
OTM_MAX_MONEYNESS = 0.98
ATM_MAX_MONEYNESS = 1.02

# DTE buckets using START_DTE.
SHORT_DTE_MAX = 14
MEDIUM_DTE_MAX = 30


# ============================================================
# PATH HELPERS
# ============================================================

def _candidate_project_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    candidates = [
        here.parent,
        here.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
        Path("/mnt/data"),
    ]

    out: list[Path] = []
    for p in candidates:
        p = p.resolve()
        if p not in out:
            out.append(p)
    return out


def find_existing_file(filename: str, relative_dirs: Iterable[str]) -> Path:
    checked: list[Path] = []
    for base in _candidate_project_dirs():
        for rel in relative_dirs:
            p = base / rel / filename if rel else base / filename
            checked.append(p)
            if p.exists():
                return p

    checked_msg = "\n".join(str(p) for p in checked)
    raise FileNotFoundError(f"Could not find {filename}. Checked:\n{checked_msg}")


def find_optional_file(filenames: list[str], relative_dirs: Iterable[str]) -> Path | None:
    for filename in filenames:
        try:
            return find_existing_file(filename, relative_dirs)
        except FileNotFoundError:
            continue
    return None


TRANSITIONS_PATH = find_existing_file(
    TRANSITIONS_FILE,
    relative_dirs=["data/processed", "processed", ""],
)

RESULT_PATHS: dict[str, Path] = {}
for label, filename in REQUIRED_RESULT_FILES.items():
    RESULT_PATHS[label] = find_existing_file(filename, relative_dirs=["outputs", ""])

for label, filenames in OPTIONAL_RESULT_FILES.items():
    p = find_optional_file(filenames, relative_dirs=["outputs", ""])
    if p is not None:
        RESULT_PATHS[label] = p

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
# METRICS
# ============================================================

def cvar_95(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    if x.empty:
        return np.nan
    q = x.quantile(0.05)
    tail = x[x <= q]
    return float(tail.mean()) if not tail.empty else np.nan


def sharpe_like(x: pd.Series) -> float:
    x = pd.Series(x).dropna()
    std = x.std()
    if std == 0 or pd.isna(std):
        return np.nan
    return float(x.mean() / std)


def make_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    d = df.copy()

    defaults = {
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

    for c, v in defaults.items():
        if c not in d.columns:
            d[c] = v

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


def summarize_rl_across_seeds(metrics_by_seed: pd.DataFrame, regime_col: str) -> pd.DataFrame:
    d = metrics_by_seed.dropna(subset=["SEED"]).copy()
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


def make_baseline_compact(metrics: pd.DataFrame, regime_col: str) -> pd.DataFrame:
    b = metrics[metrics["ALGORITHM"].astype(str).str.lower().eq("baseline")].copy()
    if b.empty:
        return b

    out = pd.DataFrame({
        "ALGORITHM": b["STRATEGY"],
        "SPLIT": b["SPLIT"],
        regime_col: b[regime_col],
        "N_SEEDS": np.nan,
        "MEAN_OF_MEAN_PNL": b["MEAN_PNL"],
        "STD_OF_MEAN_PNL": np.nan,
        "MEAN_OF_CVAR_95": b["CVAR_95"],
        "STD_OF_CVAR_95": np.nan,
        "MEAN_OF_SHARPE_LIKE": b["SHARPE_LIKE"],
        "STD_OF_SHARPE_LIKE": np.nan,
        "MEAN_TC": b["MEAN_TC"],
        "MEAN_TURNOVER": b["MEAN_TURNOVER"],
        "AVG_HEDGE": b["AVG_HEDGE"],
        "AVG_ADJUSTMENT_FROM_DELTA": b["AVG_ADJUSTMENT_FROM_DELTA"],
        "NO_TRADE_RATE": b["NO_TRADE_RATE"],
        "TRAINING_TIME_MIN": b["TRAINING_TIME_MIN"],
    })

    return out


def make_thesis_compact(metrics: pd.DataFrame, seed_summary: pd.DataFrame, regime_col: str) -> pd.DataFrame:
    baseline = make_baseline_compact(metrics, regime_col)
    rl = seed_summary.copy()
    if not rl.empty:
        rl = rl[[
            "ALGORITHM",
            "SPLIT",
            regime_col,
            "N_SEEDS",
            "MEAN_OF_MEAN_PNL",
            "STD_OF_MEAN_PNL",
            "MEAN_OF_CVAR_95",
            "STD_OF_CVAR_95",
            "MEAN_OF_SHARPE_LIKE",
            "STD_OF_SHARPE_LIKE",
            "MEAN_TC",
            "MEAN_TURNOVER",
            "AVG_HEDGE",
            "AVG_ADJUSTMENT_FROM_DELTA",
            "NO_TRADE_RATE",
            "TRAINING_TIME_MIN",
        ]]
    out = pd.concat([baseline, rl], ignore_index=True, sort=False)
    return out.sort_values(["SPLIT", regime_col, "ALGORITHM"])


def make_rankings(compact: pd.DataFrame, regime_col: str, split: str = "test") -> pd.DataFrame:
    d = compact[compact["SPLIT"].eq(split)].copy()
    d = d[~d["ALGORITHM"].eq("no_hedge")].copy()

    rows = []
    for regime, g in d.groupby(regime_col, dropna=False):
        for metric, ascending in [
            ("MEAN_OF_MEAN_PNL", False),
            ("MEAN_OF_CVAR_95", False),
            ("MEAN_OF_SHARPE_LIKE", False),
            ("MEAN_TC", True),
            ("MEAN_TURNOVER", True),
        ]:
            gg = g.dropna(subset=[metric]).copy()
            if gg.empty:
                continue
            top = gg.sort_values(metric, ascending=ascending).iloc[0]
            rows.append({
                "SPLIT": split,
                "REGIME_COL": regime_col,
                "REGIME": regime,
                "METRIC": metric,
                "BEST_ALGORITHM": top["ALGORITHM"],
                "BEST_VALUE": top[metric],
            })
    return pd.DataFrame(rows)


# ============================================================
# LOAD EPISODE RESULTS
# ============================================================

print("Transitions:")
print(TRANSITIONS_PATH)
print("\nResult files:")
for k, v in RESULT_PATHS.items():
    print(f"  {k}: {v}")


def read_episode_results(path: Path, label: str) -> pd.DataFrame:
    d = pd.read_excel(path, sheet_name="Episode_Results")

    if "ALGORITHM" not in d.columns:
        d["ALGORITHM"] = label

    # Preserve baseline rows, but force RL rows to requested label.
    baseline_mask = d["STRATEGY"].astype(str).isin(["delta", "no_hedge"]) | d["ALGORITHM"].astype(str).str.lower().eq("baseline")
    d.loc[baseline_mask, "ALGORITHM"] = "baseline"
    d.loc[~baseline_mask, "ALGORITHM"] = label

    if "SOURCE_FILE" not in d.columns:
        d["SOURCE_FILE"] = path.name

    return d


all_sources = []
for label, path in RESULT_PATHS.items():
    all_sources.append(read_episode_results(path, label))

all_ep_raw = pd.concat(all_sources, ignore_index=True, sort=False)

# Deduplicate baseline rows that appear in every source file.
baseline_mask = all_ep_raw["ALGORITHM"].astype(str).str.lower().eq("baseline")
baseline_rows = (
    all_ep_raw[baseline_mask]
    .drop_duplicates(subset=["EPISODE_ID", "SPLIT", "STRATEGY"])
    .copy()
)

# RL rows are unique by algorithm, seed and episode.
rl_rows = (
    all_ep_raw[~baseline_mask]
    .drop_duplicates(subset=["ALGORITHM", "SEED", "EPISODE_ID", "SPLIT", "STRATEGY"])
    .copy()
)

all_ep = pd.concat([baseline_rows, rl_rows], ignore_index=True, sort=False)

print("\nEpisode rows by algorithm/split:")
print(all_ep.groupby(["ALGORITHM", "SPLIT"])["EPISODE_ID"].nunique())


# ============================================================
# LOAD TRANSITIONS AND COMPUTE EPISODE REGIME FEATURES
# ============================================================

transitions = pd.read_parquet(TRANSITIONS_PATH)
transitions = transitions.sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)
df = transitions.copy()

if {"SPY_CLOSE", "SPY_NEXT_CLOSE"}.issubset(df.columns):
    df["SPY_RET"] = df["SPY_NEXT_CLOSE"].astype(float) / df["SPY_CLOSE"].astype(float) - 1.0
elif {"SPY_DS", "SPY_CLOSE"}.issubset(df.columns):
    df["SPY_RET"] = df["SPY_DS"].astype(float) / df["SPY_CLOSE"].astype(float)
else:
    raise ValueError("Cannot compute SPY returns. Need SPY_NEXT_CLOSE/SPY_CLOSE or SPY_DS/SPY_CLOSE.")

if "OPTION_IV" in df.columns:
    df["OPTION_IV_DECIMAL"] = pd.to_numeric(df["OPTION_IV"], errors="coerce")
    mask_pct = df["OPTION_IV_DECIMAL"] > 3.0
    df.loc[mask_pct, "OPTION_IV_DECIMAL"] = df.loc[mask_pct, "OPTION_IV_DECIMAL"] / 100.0
else:
    df["OPTION_IV_DECIMAL"] = np.nan

if "SPY_MONEYNESS" in df.columns:
    df["MONEYNESS"] = pd.to_numeric(df["SPY_MONEYNESS"], errors="coerce")
elif {"SPY_CLOSE", "STRIKE"}.issubset(df.columns):
    df["MONEYNESS"] = df["SPY_CLOSE"].astype(float) / df["STRIKE"].astype(float)
else:
    df["MONEYNESS"] = np.nan

if "SPY_LOG_MONEYNESS" in df.columns:
    df["ABS_LOG_MONEYNESS"] = pd.to_numeric(df["SPY_LOG_MONEYNESS"], errors="coerce").abs()
else:
    df["ABS_LOG_MONEYNESS"] = np.log(df["MONEYNESS"]).abs()

episode_regime = (
    df.groupby(["EPISODE_ID", "SPLIT"])
    .agg(
        START_DATE=("QUOTE_DATE", "first"),
        END_DATE=("NEXT_QUOTE_DATE", "last") if "NEXT_QUOTE_DATE" in df.columns else ("QUOTE_DATE", "last"),
        N_STEPS=("SPY_RET", "count"),
        START_DTE=("DTE", "first") if "DTE" in df.columns else ("SPY_RET", "count"),
        AVG_DTE=("DTE", "mean") if "DTE" in df.columns else ("SPY_RET", "count"),
        AVG_MONEYNESS=("MONEYNESS", "mean"),
        START_MONEYNESS=("MONEYNESS", "first"),
        AVG_ABS_LOG_MONEYNESS=("ABS_LOG_MONEYNESS", "mean"),
        MEAN_SPY_RET=("SPY_RET", "mean"),
        STD_DAILY_RET=("SPY_RET", "std"),
        REALIZED_VOL_ANN=("SPY_RET", lambda x: np.sqrt(252.0) * pd.Series(x).dropna().std()),
        AVG_IV=("OPTION_IV_DECIMAL", "mean"),
    )
    .reset_index()
)

episode_regime["REALIZED_VOL_ANN"] = episode_regime["REALIZED_VOL_ANN"].fillna(0.0)


# ============================================================
# ASSIGN REGIMES
# ============================================================

def compute_terciles(series: pd.Series) -> tuple[float, float]:
    s = pd.Series(series).dropna()
    if s.empty:
        return np.nan, np.nan
    return float(s.quantile(1 / 3)), float(s.quantile(2 / 3))


def apply_tercile_label(series: pd.Series, q1: float, q2: float) -> pd.Series:
    def label(x):
        if pd.isna(x):
            return "unknown"
        if x <= q1:
            return "low"
        if x <= q2:
            return "medium"
        return "high"
    return series.apply(label)


if THRESHOLD_SOURCE == "train":
    threshold_base = episode_regime[episode_regime["SPLIT"].eq("train")]
elif THRESHOLD_SOURCE == "test":
    threshold_base = episode_regime[episode_regime["SPLIT"].eq("test")]
elif THRESHOLD_SOURCE == "all":
    threshold_base = episode_regime
else:
    raise ValueError("THRESHOLD_SOURCE must be 'train', 'test', or 'all'.")

rv_low, rv_high = compute_terciles(threshold_base["REALIZED_VOL_ANN"])
iv_low, iv_high = compute_terciles(threshold_base["AVG_IV"])

episode_regime["RV_REGIME"] = apply_tercile_label(episode_regime["REALIZED_VOL_ANN"], rv_low, rv_high)
episode_regime["IV_REGIME"] = apply_tercile_label(episode_regime["AVG_IV"], iv_low, iv_high)


def moneyness_regime(m):
    if pd.isna(m):
        return "unknown"
    if m < OTM_MAX_MONEYNESS:
        return "OTM"
    if m <= ATM_MAX_MONEYNESS:
        return "ATM"
    return "ITM"


def dte_regime(dte):
    if pd.isna(dte):
        return "unknown"
    if dte <= SHORT_DTE_MAX:
        return "short"
    if dte <= MEDIUM_DTE_MAX:
        return "medium"
    return "long"


episode_regime["MONEYNESS_REGIME"] = episode_regime["AVG_MONEYNESS"].apply(moneyness_regime)
episode_regime["START_MONEYNESS_REGIME"] = episode_regime["START_MONEYNESS"].apply(moneyness_regime)
episode_regime["DTE_REGIME"] = episode_regime["START_DTE"].apply(dte_regime)

regime_thresholds = pd.DataFrame([
    {
        "REGIME_TYPE": "RV_REGIME",
        "THRESHOLD_SOURCE": THRESHOLD_SOURCE,
        "LOW_MAX": rv_low,
        "MEDIUM_MAX": rv_high,
        "HIGH_MIN_EXCLUSIVE": rv_high,
    },
    {
        "REGIME_TYPE": "IV_REGIME",
        "THRESHOLD_SOURCE": THRESHOLD_SOURCE,
        "LOW_MAX": iv_low,
        "MEDIUM_MAX": iv_high,
        "HIGH_MIN_EXCLUSIVE": iv_high,
    },
    {
        "REGIME_TYPE": "MONEYNESS_REGIME",
        "THRESHOLD_SOURCE": "fixed",
        "LOW_LABEL": "OTM if S/K < 0.98",
        "MEDIUM_LABEL": "ATM if 0.98 <= S/K <= 1.02",
        "HIGH_LABEL": "ITM if S/K > 1.02",
    },
    {
        "REGIME_TYPE": "DTE_REGIME",
        "THRESHOLD_SOURCE": "fixed",
        "LOW_LABEL": f"short if DTE <= {SHORT_DTE_MAX}",
        "MEDIUM_LABEL": f"medium if {SHORT_DTE_MAX} < DTE <= {MEDIUM_DTE_MAX}",
        "HIGH_LABEL": f"long if DTE > {MEDIUM_DTE_MAX}",
    },
])

regime_count_cols = ["SPLIT", "RV_REGIME", "IV_REGIME", "MONEYNESS_REGIME", "DTE_REGIME"]
regime_counts = (
    episode_regime
    .groupby(regime_count_cols, dropna=False)
    .size()
    .reset_index(name="EPISODES")
)

print("\nRegime thresholds:")
print(regime_thresholds)

print("\nRegime counts by split/regime combination:")
print(regime_counts.head(20))


# ============================================================
# MERGE REGIMES INTO EPISODE RESULTS
# ============================================================

regime_cols = [
    "EPISODE_ID",
    "SPLIT",
    "REALIZED_VOL_ANN",
    "AVG_IV",
    "AVG_MONEYNESS",
    "START_MONEYNESS",
    "START_DTE",
    "AVG_DTE",
    "RV_REGIME",
    "IV_REGIME",
    "MONEYNESS_REGIME",
    "START_MONEYNESS_REGIME",
    "DTE_REGIME",
]

merged = all_ep.merge(
    episode_regime[regime_cols],
    on=["EPISODE_ID", "SPLIT"],
    how="left",
    validate="many_to_one",
)

missing = merged["RV_REGIME"].isna().sum()
if missing:
    print(f"Warning: {missing} episode result rows have missing regime labels.")

# Ensure baseline labels are clean.
merged.loc[merged["STRATEGY"].astype(str).eq("delta"), "ALGORITHM"] = "baseline"
merged.loc[merged["STRATEGY"].astype(str).eq("no_hedge"), "ALGORITHM"] = "baseline"


# ============================================================
# BUILD METRICS FOR EACH REGIME DIMENSION
# ============================================================

regime_dimensions = [
    "RV_REGIME",
    "IV_REGIME",
    "MONEYNESS_REGIME",
    "DTE_REGIME",
]

overall_metrics = make_metrics(merged, ["ALGORITHM", "SPLIT", "STRATEGY"])

metrics_by_regime = {}
metrics_by_seed_by_regime = {}
seed_summary_by_regime = {}
compact_by_regime = {}
rankings_by_regime = {}

for regime_col in regime_dimensions:
    metrics = make_metrics(merged, ["SPLIT", regime_col, "ALGORITHM", "STRATEGY"])
    metrics_by_seed = make_metrics(merged, ["ALGORITHM", "SEED", "SPLIT", regime_col, "STRATEGY"])
    seed_summary = summarize_rl_across_seeds(metrics_by_seed, regime_col)
    compact = make_thesis_compact(metrics, seed_summary, regime_col)
    rankings = make_rankings(compact, regime_col, split="test")

    metrics_by_regime[regime_col] = metrics
    metrics_by_seed_by_regime[regime_col] = metrics_by_seed
    seed_summary_by_regime[regime_col] = seed_summary
    compact_by_regime[regime_col] = compact
    rankings_by_regime[regime_col] = rankings


all_test_rankings = pd.concat(
    [df.assign(REGIME_DIMENSION=k) for k, df in rankings_by_regime.items()],
    ignore_index=True,
    sort=False,
)

test_compact_all = pd.concat(
    [
        df[df["SPLIT"].eq("test")].assign(REGIME_DIMENSION=k, REGIME=df[k])
        for k, df in compact_by_regime.items()
    ],
    ignore_index=True,
    sort=False,
)

# Keep compact all in a consistent column order.
preferred_cols = [
    "REGIME_DIMENSION",
    "REGIME",
    "ALGORITHM",
    "SPLIT",
    "N_SEEDS",
    "MEAN_OF_MEAN_PNL",
    "STD_OF_MEAN_PNL",
    "MEAN_OF_CVAR_95",
    "STD_OF_CVAR_95",
    "MEAN_OF_SHARPE_LIKE",
    "STD_OF_SHARPE_LIKE",
    "MEAN_TC",
    "MEAN_TURNOVER",
    "AVG_HEDGE",
    "AVG_ADJUSTMENT_FROM_DELTA",
    "NO_TRADE_RATE",
    "TRAINING_TIME_MIN",
]
test_compact_all = test_compact_all[[c for c in preferred_cols if c in test_compact_all.columns]]


# ============================================================
# SAVE OUTPUT
# ============================================================

with pd.ExcelWriter(OUTPUT_PATH, engine="openpyxl") as writer:
    merged.to_excel(writer, sheet_name="Episode_Results_Regimes", index=False)
    episode_regime.to_excel(writer, sheet_name="Episode_Regime_Features", index=False)
    regime_thresholds.to_excel(writer, sheet_name="Regime_Thresholds", index=False)
    regime_counts.to_excel(writer, sheet_name="Regime_Counts", index=False)
    overall_metrics.to_excel(writer, sheet_name="Overall_Metrics", index=False)

    for regime_col in regime_dimensions:
        short = regime_col.replace("_REGIME", "")
        metrics_by_regime[regime_col].to_excel(writer, sheet_name=f"{short}_Metrics", index=False)
        metrics_by_seed_by_regime[regime_col].to_excel(writer, sheet_name=f"{short}_Metrics_By_Seed", index=False)
        seed_summary_by_regime[regime_col].to_excel(writer, sheet_name=f"{short}_Seed_Summary", index=False)
        compact_by_regime[regime_col].to_excel(writer, sheet_name=f"{short}_Thesis_Compact", index=False)
        rankings_by_regime[regime_col].to_excel(writer, sheet_name=f"{short}_Test_Rankings", index=False)

    test_compact_all.to_excel(writer, sheet_name="Test_Compact_All_Regimes", index=False)
    all_test_rankings.to_excel(writer, sheet_name="Test_Rankings_All", index=False)

print("\nSaved extended regime evaluation:")
print(OUTPUT_PATH)

print("\nMost useful sheets:")
print("  - Test_Compact_All_Regimes")
print("  - RV_Thesis_Compact")
print("  - IV_Thesis_Compact")
print("  - MONEYNESS_Thesis_Compact")
print("  - DTE_Thesis_Compact")
print("  - Test_Rankings_All")
