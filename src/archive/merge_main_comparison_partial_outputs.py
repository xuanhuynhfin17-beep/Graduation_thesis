"""
merge_main_comparison_partial_outputs.py

Merge a PPO/SAC partial run and a TD3-only run from:
05_main_comparison_v3c_5seed_hyperparam_scenarios*.py

Typical input files:
    outputs/main_comparison_v3c_hyperparam_scenarios_100k_5seeds_PPO_SAC_partial.xlsx
    outputs/main_comparison_v3c_hyperparam_scenarios_100k_5seeds_TD3_only.xlsx

Output:
    outputs/main_comparison_v3c_hyperparam_scenarios_100k_5seeds_MERGED.xlsx

Run:
    python merge_main_comparison_partial_outputs.py
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np


OUTPUT_DIR = Path("outputs")

PPO_SAC_FILE = OUTPUT_DIR / "main_comparison_v3c_hyperparam_scenarios_100k_5seeds_PPO_SAC_partial.xlsx"
TD3_FILE = OUTPUT_DIR / "main_comparison_v3c_hyperparam_scenarios_100k_5seeds_TD3_only.xlsx"

MERGED_FILE = OUTPUT_DIR / "main_comparison_v3c_hyperparam_scenarios_100k_5seeds_MERGED.xlsx"


def read_sheet_if_exists(path: Path, sheet_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    xl = pd.ExcelFile(path)
    if sheet_name not in xl.sheet_names:
        return pd.DataFrame()
    return pd.read_excel(path, sheet_name=sheet_name)


def concat_and_dedupe(dfs: list[pd.DataFrame], subset_priority: list[str] | None = None) -> pd.DataFrame:
    dfs = [df for df in dfs if df is not None and not df.empty]
    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True, sort=False)

    if subset_priority:
        subset = [c for c in subset_priority if c in out.columns]
        if subset:
            out = out.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
            return out

    out = out.drop_duplicates(keep="first").reset_index(drop=True)
    return out


def recompute_scenario_summary(metrics_by_seed: pd.DataFrame) -> pd.DataFrame:
    if metrics_by_seed.empty:
        return pd.DataFrame()

    df = metrics_by_seed.copy()
    if "SEED" in df.columns:
        df = df.dropna(subset=["SEED"])

    group_cols = [c for c in ["ALGORITHM", "VARIANT", "SPLIT"] if c in df.columns]
    if not group_cols:
        return pd.DataFrame()

    agg_map = {}
    if "SEED" in df.columns:
        agg_map["N_SEEDS"] = ("SEED", "nunique")

    metric_pairs = {
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
        "AVG_NO_TRADE_RATE": ("NO_TRADE_RATE", "mean"),
        "NO_TRADE_RATE": ("NO_TRADE_RATE", "mean"),
        "ACTION_NEAR_NEG1_RATE": ("ACTION_NEAR_NEG1_RATE", "mean"),
        "ACTION_NEAR_POS1_RATE": ("ACTION_NEAR_POS1_RATE", "mean"),
        "TRAINING_TIME_MIN": ("TRAINING_TIME_MIN", "mean"),
    }

    for out_col, (in_col, func) in metric_pairs.items():
        if in_col in df.columns:
            agg_map[out_col] = (in_col, func)

    summary = df.groupby(group_cols, dropna=False).agg(**agg_map).reset_index()

    # Add test-split ranks where columns exist.
    for col in ["MEAN_OF_MEAN_PNL", "MEAN_OF_CVAR_95", "MEAN_OF_SHARPE_LIKE"]:
        rank_col = "RANK_" + col.replace("MEAN_OF_", "")
        summary[rank_col] = np.nan
        if col in summary.columns and "SPLIT" in summary.columns:
            mask = summary["SPLIT"].astype(str).str.lower().eq("test")
            summary.loc[mask, rank_col] = summary.loc[mask, col].rank(
                ascending=False,
                method="min",
            )

    sort_cols = [c for c in ["SPLIT", "ALGORITHM", "VARIANT"] if c in summary.columns]
    if sort_cols:
        summary = summary.sort_values(sort_cols).reset_index(drop=True)

    return summary


def make_test_ranking(scenario_summary: pd.DataFrame) -> pd.DataFrame:
    if scenario_summary.empty or "SPLIT" not in scenario_summary.columns:
        return pd.DataFrame()

    test = scenario_summary[
        scenario_summary["SPLIT"].astype(str).str.lower().eq("test")
    ].copy()

    sort_cols = []
    for c in ["RANK_SHARPE_LIKE", "RANK_MEAN_PNL", "RANK_CVAR_95"]:
        if c in test.columns:
            sort_cols.append(c)

    if sort_cols:
        test = test.sort_values(sort_cols).reset_index(drop=True)
    else:
        test = test.reset_index(drop=True)

    flags = []
    for _, row in test.iterrows():
        notes = []
        if "MEAN_OF_MEAN_PNL" in row and pd.notna(row["MEAN_OF_MEAN_PNL"]) and row["MEAN_OF_MEAN_PNL"] > 0:
            notes.append("positive mean PnL")
        if "MEAN_OF_CVAR_95" in row and pd.notna(row["MEAN_OF_CVAR_95"]) and row["MEAN_OF_CVAR_95"] > -400:
            notes.append("better CVaR region")
        if "MEAN_TC" in row and pd.notna(row["MEAN_TC"]) and row["MEAN_TC"] < 60:
            notes.append("moderate TC")
        if "MEAN_TURNOVER" in row and pd.notna(row["MEAN_TURNOVER"]) and row["MEAN_TURNOVER"] < 2.6:
            notes.append("moderate turnover")
        flags.append("; ".join(notes) if notes else "watch / weak risk-cost profile")

    test["INTERPRETATION_FLAGS"] = flags
    return test


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = [PPO_SAC_FILE, TD3_FILE]
    print("Merging files:")
    for f in files:
        print(" -", f)
        if not f.exists():
            raise FileNotFoundError(
                f"Could not find {f}\n"
                "Edit PPO_SAC_FILE and TD3_FILE at the top of this script if your filenames differ."
            )

    scenario_config = concat_and_dedupe(
        [read_sheet_if_exists(f, "Scenario_Config") for f in files],
        subset_priority=["VARIANT", "ALGORITHM"],
    )

    episode_results = concat_and_dedupe(
        [read_sheet_if_exists(f, "Episode_Results") for f in files],
        subset_priority=["VARIANT", "ALGORITHM", "SEED", "EPISODE_ID", "SPLIT", "STRATEGY"],
    )

    metrics_by_seed = concat_and_dedupe(
        [read_sheet_if_exists(f, "Metrics_By_Seed") for f in files],
        subset_priority=["VARIANT", "ALGORITHM", "SEED", "SPLIT", "STRATEGY"],
    )

    training_log = concat_and_dedupe(
        [read_sheet_if_exists(f, "Training_Log") for f in files],
        subset_priority=["VARIANT", "ALGORITHM", "SEED", "MODEL_PATH"],
    )

    # Existing Metrics sheet is kept as reference, but Scenario_Summary is recomputed from Metrics_By_Seed.
    metrics_raw = concat_and_dedupe(
        [read_sheet_if_exists(f, "Metrics") for f in files],
        subset_priority=["VARIANT", "ALGORITHM", "SPLIT", "STRATEGY"],
    )

    scenario_summary = recompute_scenario_summary(metrics_by_seed)
    test_ranking = make_test_ranking(scenario_summary)

    with pd.ExcelWriter(MERGED_FILE, engine="openpyxl") as writer:
        scenario_config.to_excel(writer, sheet_name="Scenario_Config", index=False)
        episode_results.to_excel(writer, sheet_name="Episode_Results", index=False)
        metrics_raw.to_excel(writer, sheet_name="Metrics_Raw_Merged", index=False)
        metrics_by_seed.to_excel(writer, sheet_name="Metrics_By_Seed", index=False)
        scenario_summary.to_excel(writer, sheet_name="Scenario_Summary", index=False)
        test_ranking.to_excel(writer, sheet_name="Test_Ranking", index=False)
        training_log.to_excel(writer, sheet_name="Training_Log", index=False)

    print(f"\nMerged file written to:\n{MERGED_FILE}")

    print("\nScenario counts:")
    if not metrics_by_seed.empty:
        print(metrics_by_seed.groupby(["ALGORITHM", "VARIANT"])["SEED"].nunique())

    print("\nTest ranking preview:")
    preview_cols = [
        "ALGORITHM", "VARIANT",
        "N_SEEDS",
        "MEAN_OF_MEAN_PNL", "MEAN_OF_CVAR_95",
        "MEAN_OF_SHARPE_LIKE", "MEAN_TC", "MEAN_TURNOVER",
    ]
    preview_cols = [c for c in preview_cols if c in test_ranking.columns]
    print(test_ranking[preview_cols])


if __name__ == "__main__":
    main()
