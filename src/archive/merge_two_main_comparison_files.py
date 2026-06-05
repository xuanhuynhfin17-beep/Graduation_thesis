"""
merge_two_main_comparison_files.py

Merge two Excel outputs from the main comparison robustness run, typically:

1. PPO/SAC recovered file:
   outputs/main_comparison_v3c_hyperparam_scenarios_100k_5seeds_PPO_SAC_partial_RECOVERED_FROM_MODELS.xlsx

2. TD3-only file:
   outputs/main_comparison_v3c_hyperparam_scenarios_100k_5seeds_TD3_only.xlsx

The script creates one merged Excel file with recomputed summary sheets.

Usage option A: automatic default filenames
    python merge_two_main_comparison_files.py

Usage option B: explicit filenames
    python merge_two_main_comparison_files.py ^
        --file1 outputs/your_ppo_sac_file.xlsx ^
        --file2 outputs/your_td3_file.xlsx ^
        --out outputs/main_comparison_v3c_hyperparam_scenarios_100k_5seeds_MERGED.xlsx

Notes:
- The script does not retrain anything.
- It merges saved Excel results only.
- Scenario_Summary and Test_Ranking are recomputed from Metrics_By_Seed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


DEFAULT_FILE1_CANDIDATES = [
    Path("outputs/main_comparison_v3c_hyperparam_scenarios_100k_5seeds_PPO_SAC_partial_RECOVERED_FROM_MODELS.xlsx"),
    Path("outputs/main_comparison_v3c_hyperparam_scenarios_100k_5seeds_PPO_SAC_partial.xlsx"),
    Path("outputs/main_comparison_v3c_hyperparam_scenarios_100k_5seeds.xlsx"),
]

DEFAULT_FILE2_CANDIDATES = [
    Path("outputs/main_comparison_v3c_hyperparam_scenarios_100k_5seeds_TD3_only.xlsx"),
]

DEFAULT_OUT = Path("outputs/main_comparison_v3c_hyperparam_scenarios_100k_5seeds_MERGED.xlsx")


def find_first_existing(candidates: list[Path], label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    msg = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        f"Could not find {label}. Checked:\n{msg}\n\n"
        f"Pass explicit paths with --file1 and --file2 if your filenames differ."
    )


def read_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    xl = pd.ExcelFile(path)
    if sheet_name not in xl.sheet_names:
        return pd.DataFrame()
    return pd.read_excel(path, sheet_name=sheet_name)


def concat_sheets(
    files: list[Path],
    sheet_name: str,
    dedupe_subset: list[str] | None = None,
) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = read_sheet(f, sheet_name)
        if not df.empty:
            df["SOURCE_FILE"] = f.name
            dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True, sort=False)

    if dedupe_subset:
        subset = [c for c in dedupe_subset if c in out.columns]
        if subset:
            out = out.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
            return out

    return out.drop_duplicates(keep="first").reset_index(drop=True)


def recompute_scenario_summary(metrics_by_seed: pd.DataFrame) -> pd.DataFrame:
    if metrics_by_seed.empty:
        return pd.DataFrame()

    df = metrics_by_seed.copy()

    if "SEED" in df.columns:
        df = df.dropna(subset=["SEED"]).copy()

    group_cols = [c for c in ["ALGORITHM", "VARIANT", "SPLIT"] if c in df.columns]
    if not group_cols:
        raise ValueError("Metrics_By_Seed does not contain enough grouping columns.")

    agg_spec = {}

    if "SEED" in df.columns:
        agg_spec["N_SEEDS"] = ("SEED", "nunique")

    metric_map = {
        "MEAN_OF_MEAN_PNL": ("MEAN_PNL", "mean"),
        "STD_OF_MEAN_PNL": ("MEAN_PNL", "std"),
        "MEAN_OF_CVAR_95": ("CVAR_95", "mean"),
        "STD_OF_CVAR_95": ("CVAR_95", "std"),
        "MEAN_OF_SHARPE_LIKE": ("SHARPE_LIKE", "mean"),
        "STD_OF_SHARPE_LIKE": ("SHARPE_LIKE", "std"),
        "MEAN_TC": ("MEAN_TC", "mean"),
        "MEAN_TURNOVER": ("MEAN_TURNOVER", "mean"),
        "AVG_HEDGE": ("AVG_HEDGE", "mean"),
        "AVG_ADJUSTMENT_FROM_DELTA": ("AVG_ADJUSTMENT_FROM_DELTA", "mean"),
        "NO_TRADE_RATE": ("NO_TRADE_RATE", "mean"),
        "AVG_NO_TRADE_RATE": ("NO_TRADE_RATE", "mean"),
        "ACTION_NEAR_NEG1_RATE": ("ACTION_NEAR_NEG1_RATE", "mean"),
        "ACTION_NEAR_POS1_RATE": ("ACTION_NEAR_POS1_RATE", "mean"),
        "TRAINING_TIME_MIN": ("TRAINING_TIME_MIN", "mean"),
    }

    for out_col, (in_col, func) in metric_map.items():
        if in_col in df.columns:
            agg_spec[out_col] = (in_col, func)

    summary = df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()

    # Test split ranks. Higher is better for mean PnL, CVaR, and Sharpe-like.
    rank_map = {
        "MEAN_OF_MEAN_PNL": "RANK_MEAN_PNL",
        "MEAN_OF_CVAR_95": "RANK_CVAR_95",
        "MEAN_OF_SHARPE_LIKE": "RANK_SHARPE_LIKE",
        "MEAN_TC": "RANK_LOW_TC",
        "MEAN_TURNOVER": "RANK_LOW_TURNOVER",
    }

    for metric, rank_col in rank_map.items():
        summary[rank_col] = np.nan
        if metric not in summary.columns or "SPLIT" not in summary.columns:
            continue

        test_mask = summary["SPLIT"].astype(str).str.lower().eq("test")
        ascending = metric in ["MEAN_TC", "MEAN_TURNOVER"]
        summary.loc[test_mask, rank_col] = summary.loc[test_mask, metric].rank(
            ascending=ascending,
            method="min",
        )

    sort_cols = [c for c in ["SPLIT", "ALGORITHM", "VARIANT"] if c in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols).reset_index(drop=True)

    return summary


def build_test_ranking(scenario_summary: pd.DataFrame) -> pd.DataFrame:
    if scenario_summary.empty:
        return pd.DataFrame()

    if "SPLIT" not in scenario_summary.columns:
        return pd.DataFrame()

    test = scenario_summary[
        scenario_summary["SPLIT"].astype(str).str.lower().eq("test")
    ].copy()

    # Main thesis-friendly order: Sharpe-like first, then mean PnL, then CVaR.
    sort_cols = [c for c in ["RANK_SHARPE_LIKE", "RANK_MEAN_PNL", "RANK_CVAR_95"] if c in test.columns]
    if sort_cols:
        test = test.sort_values(sort_cols).reset_index(drop=True)
    else:
        test = test.reset_index(drop=True)

    flags = []
    for _, row in test.iterrows():
        notes = []

        mean_pnl = row.get("MEAN_OF_MEAN_PNL", np.nan)
        cvar = row.get("MEAN_OF_CVAR_95", np.nan)
        tc = row.get("MEAN_TC", np.nan)
        turnover = row.get("MEAN_TURNOVER", np.nan)

        if pd.notna(mean_pnl) and mean_pnl > 0:
            notes.append("positive mean PnL")
        if pd.notna(cvar) and cvar > -400:
            notes.append("better CVaR region")
        elif pd.notna(cvar) and cvar < -700:
            notes.append("weak CVaR region")
        if pd.notna(tc) and tc < 60:
            notes.append("moderate TC")
        elif pd.notna(tc) and tc > 80:
            notes.append("high TC")
        if pd.notna(turnover) and turnover < 2.6:
            notes.append("moderate turnover")
        elif pd.notna(turnover) and turnover > 3.0:
            notes.append("high turnover")

        flags.append("; ".join(notes) if notes else "mixed risk-cost profile")

    test["INTERPRETATION_FLAGS"] = flags
    return test


def build_best_by_algorithm(test_ranking: pd.DataFrame) -> pd.DataFrame:
    if test_ranking.empty or "ALGORITHM" not in test_ranking.columns:
        return pd.DataFrame()

    sort_cols = [c for c in ["RANK_SHARPE_LIKE", "RANK_MEAN_PNL", "RANK_CVAR_95"] if c in test_ranking.columns]
    if not sort_cols:
        return test_ranking.groupby("ALGORITHM", as_index=False).head(1)

    return (
        test_ranking.sort_values(["ALGORITHM"] + sort_cols)
        .groupby("ALGORITHM", as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file1", type=str, default=None, help="First Excel file, usually PPO/SAC recovered.")
    parser.add_argument("--file2", type=str, default=None, help="Second Excel file, usually TD3-only.")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT), help="Output merged Excel file.")
    args = parser.parse_args()

    file1 = Path(args.file1) if args.file1 else find_first_existing(DEFAULT_FILE1_CANDIDATES, "PPO/SAC file")
    file2 = Path(args.file2) if args.file2 else find_first_existing(DEFAULT_FILE2_CANDIDATES, "TD3-only file")
    out_path = Path(args.out)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    files = [file1, file2]

    print("Merging:")
    for f in files:
        print(" -", f)

    scenario_config = concat_sheets(
        files,
        "Scenario_Config",
        dedupe_subset=["VARIANT", "ALGORITHM"],
    )

    episode_results = concat_sheets(
        files,
        "Episode_Results",
        dedupe_subset=["VARIANT", "ALGORITHM", "SEED", "EPISODE_ID", "SPLIT", "STRATEGY"],
    )

    metrics_raw = concat_sheets(
        files,
        "Metrics",
        dedupe_subset=["VARIANT", "ALGORITHM", "SPLIT", "STRATEGY"],
    )

    metrics_by_seed = concat_sheets(
        files,
        "Metrics_By_Seed",
        dedupe_subset=["VARIANT", "ALGORITHM", "SEED", "SPLIT", "STRATEGY"],
    )

    training_log = concat_sheets(
        files,
        "Training_Log",
        dedupe_subset=["VARIANT", "ALGORITHM", "SEED", "MODEL_PATH"],
    )

    recovery_log = concat_sheets(
        files,
        "Recovery_Log",
        dedupe_subset=["VARIANT", "ALGORITHM", "SEED", "MODEL_PATH", "STATUS"],
    )

    # Optional sampled step rows. This is not needed for thesis tables but useful for audit.
    step_sample = concat_sheets(
        files,
        "Step_Results_Sample",
        dedupe_subset=["VARIANT", "ALGORITHM", "SEED", "EPISODE_ID", "QUOTE_DATE", "STRATEGY"],
    )

    scenario_summary = recompute_scenario_summary(metrics_by_seed)
    test_ranking = build_test_ranking(scenario_summary)
    best_by_algorithm = build_best_by_algorithm(test_ranking)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        scenario_config.to_excel(writer, sheet_name="Scenario_Config", index=False)
        episode_results.to_excel(writer, sheet_name="Episode_Results", index=False)
        metrics_raw.to_excel(writer, sheet_name="Metrics_Raw_Merged", index=False)
        metrics_by_seed.to_excel(writer, sheet_name="Metrics_By_Seed", index=False)
        scenario_summary.to_excel(writer, sheet_name="Scenario_Summary", index=False)
        test_ranking.to_excel(writer, sheet_name="Test_Ranking", index=False)
        best_by_algorithm.to_excel(writer, sheet_name="Best_By_Algorithm", index=False)
        training_log.to_excel(writer, sheet_name="Training_Log", index=False)

        if not recovery_log.empty:
            recovery_log.to_excel(writer, sheet_name="Recovery_Log", index=False)

        # Excel has row limits; keep sample only if reasonably sized.
        if not step_sample.empty:
            max_rows = 900_000
            if len(step_sample) > max_rows:
                step_sample = step_sample.sample(max_rows, random_state=42)
            step_sample.to_excel(writer, sheet_name="Step_Results_Sample", index=False)

    print("\nMerged file written:")
    print(out_path)

    print("\nSeed counts by algorithm/scenario:")
    if not metrics_by_seed.empty and {"ALGORITHM", "VARIANT", "SEED"}.issubset(metrics_by_seed.columns):
        print(metrics_by_seed.groupby(["ALGORITHM", "VARIANT"])["SEED"].nunique())

    print("\nTest ranking preview:")
    preview_cols = [
        "ALGORITHM", "VARIANT", "N_SEEDS",
        "MEAN_OF_MEAN_PNL", "MEAN_OF_CVAR_95",
        "MEAN_OF_SHARPE_LIKE", "MEAN_TC", "MEAN_TURNOVER",
        "INTERPRETATION_FLAGS",
    ]
    preview_cols = [c for c in preview_cols if c in test_ranking.columns]
    print(test_ranking[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()
