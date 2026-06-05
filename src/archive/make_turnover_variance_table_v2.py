"""
make_turnover_variance_table_v2.py

Compute turnover variance diagnostics for RL hedging experiments.

Recommended turnover definition for this thesis
-----------------------------------------------
Use the experiment-accounting turnover column if available:
    U_i = sum_t TURNOVER_{i,t}

In the full factorial step-level parquet, TURNOVER is not always equal to
abs(HEDGE - PREV_HEDGE) on the final row of an episode. On the final row, the
experiment appears to include the closing/flattening hedge trade, so
    TURNOVER_final = abs(HEDGE - PREV_HEDGE) + abs(HEDGE)
This matches total implementation turnover used by the evaluation script.

For diagnostic transparency, this script also computes PURE_HEDGE_TURNOVER from
abs(HEDGE - PREV_HEDGE) when HEDGE and PREV_HEDGE are available.

Outputs
-------
turnover_episode_values.csv
turnover_variance_by_seed.csv
turnover_variance_summary.csv
table_turnover_variance_diagnostics.tex
turnover_variance_diagnostics.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

GROUP_CANDIDATES = ["VARIANT", "ALGORITHM", "SEED", "SPLIT", "STRATEGY"]
EPISODE_CANDIDATES = ["EPISODE_ID", "episode_id", "episode", "path_id", "OPTION_ID", "CONTRACT_ID"]
HEDGE_CANDIDATES = ["HEDGE", "hedge", "TARGET_HEDGE", "target_hedge"]
PREV_HEDGE_CANDIDATES = ["PREV_HEDGE", "prev_hedge", "HEDGE_PREV", "previous_hedge"]
TURNOVER_CANDIDATES = ["TURNOVER", "turnover", "STEP_TURNOVER", "step_turnover", "TOTAL_TURNOVER", "total_turnover"]


def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise ValueError(f"Cannot find any of {candidates}. Available columns: {list(df.columns)}")
    return None


def read_input(path: Path, sheet: str | None = None) -> tuple[pd.DataFrame, str]:
    suffix = path.suffix.lower()
    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path), "parquet"
    if suffix == ".csv":
        return pd.read_csv(path), "csv"
    if suffix in [".xlsx", ".xls"]:
        xl = pd.ExcelFile(path)
        if sheet:
            return pd.read_excel(path, sheet_name=sheet), f"excel:{sheet}"
        preferred = ["Step_Results", "Step_Results_Full", "Full_Step_Results", "Episode_Results", "Step_Results_Sample"]
        for s in preferred:
            if s in xl.sheet_names:
                return pd.read_excel(path, sheet_name=s), f"excel:{s}"
        return pd.read_excel(path, sheet_name=xl.sheet_names[0]), f"excel:{xl.sheet_names[0]}"
    raise ValueError(f"Unsupported file type: {path.suffix}")


def filter_split(df: pd.DataFrame, split: str | None) -> pd.DataFrame:
    if split is None:
        return df
    split_col = find_col(df, ["SPLIT", "split", "dataset_split"], required=False)
    if split_col is None:
        return df
    mask = df[split_col].astype(str).str.lower() == split.lower()
    return df.loc[mask].copy() if mask.any() else df


def group_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in GROUP_CANDIDATES if c in df.columns]


def summarize_episode_turnover(episode: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    if not groups:
        groups = ["__ALL__"]
        episode["__ALL__"] = "ALL"

    summary = (
        episode.groupby(groups, dropna=False)
        .agg(
            EPISODES=("EPISODE_TURNOVER", "count"),
            MEAN_TURNOVER=("EPISODE_TURNOVER", "mean"),
            MEDIAN_TURNOVER=("EPISODE_TURNOVER", "median"),
            STD_TURNOVER=("EPISODE_TURNOVER", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
            VAR_TURNOVER=("EPISODE_TURNOVER", lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else 0.0),
            P95_TURNOVER=("EPISODE_TURNOVER", lambda x: float(np.quantile(x, 0.95))),
            MAX_TURNOVER=("EPISODE_TURNOVER", "max"),
        )
        .reset_index()
    )
    summary["TURNOVER_CV"] = np.where(
        summary["MEAN_TURNOVER"].abs() > 1e-12,
        summary["STD_TURNOVER"] / summary["MEAN_TURNOVER"].abs(),
        np.nan,
    )
    return summary


def compute_from_step_level(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    episode_col = find_col(df, EPISODE_CANDIDATES)
    turnover_col = find_col(df, TURNOVER_CANDIDATES, required=False)
    hedge_col = find_col(df, HEDGE_CANDIDATES, required=False)
    prev_col = find_col(df, PREV_HEDGE_CANDIDATES, required=False)

    work = df.copy()
    groups = group_columns(work)
    ep_groups = groups + [episode_col]

    diagnostics = {"turnover_source": None, "final_closeout_detected": None}

    if turnover_col is not None:
        work[turnover_col] = pd.to_numeric(work[turnover_col], errors="coerce")
        work["STEP_TURNOVER_ACCOUNTING"] = work[turnover_col]
        diagnostics["turnover_source"] = turnover_col
    elif hedge_col and prev_col:
        work[hedge_col] = pd.to_numeric(work[hedge_col], errors="coerce")
        work[prev_col] = pd.to_numeric(work[prev_col], errors="coerce")
        work["STEP_TURNOVER_ACCOUNTING"] = (work[hedge_col] - work[prev_col]).abs()
        diagnostics["turnover_source"] = "abs(HEDGE-PREV_HEDGE)"
    else:
        raise ValueError("Step-level input needs TURNOVER or HEDGE + PREV_HEDGE.")

    if hedge_col and prev_col:
        work[hedge_col] = pd.to_numeric(work[hedge_col], errors="coerce")
        work[prev_col] = pd.to_numeric(work[prev_col], errors="coerce")
        work["PURE_STEP_TURNOVER"] = (work[hedge_col] - work[prev_col]).abs()
        if turnover_col is not None:
            diff = (work["STEP_TURNOVER_ACCOUNTING"] - work["PURE_STEP_TURNOVER"]).abs()
            diagnostics["rows_turnover_differs_from_absdiff"] = int((diff > 1e-12).sum())
            diagnostics["max_turnover_absdiff_difference"] = float(diff.max())
            # If date exists, check if differences happen on episode-final rows.
            date_col = find_col(work, ["QUOTE_DATE", "date", "DATE"], required=False)
            if date_col:
                is_last = work.groupby(ep_groups)[date_col].transform("max").eq(work[date_col])
                diagnostics["diff_rows_on_final_step"] = int(((diff > 1e-12) & is_last).sum())
                diagnostics["diff_rows_not_final_step"] = int(((diff > 1e-12) & (~is_last)).sum())
                diagnostics["final_closeout_detected"] = diagnostics["diff_rows_on_final_step"] > 0 and diagnostics["diff_rows_not_final_step"] == 0

    episode = (
        work.groupby(ep_groups, dropna=False)
        .agg(
            EPISODE_TURNOVER=("STEP_TURNOVER_ACCOUNTING", "sum"),
            MEAN_STEP_TURNOVER=("STEP_TURNOVER_ACCOUNTING", "mean"),
            VAR_STEP_TURNOVER=("STEP_TURNOVER_ACCOUNTING", lambda x: float(np.var(x, ddof=1)) if len(x) > 1 else 0.0),
            N_STEPS=("STEP_TURNOVER_ACCOUNTING", "size"),
        )
        .reset_index()
    )

    if "PURE_STEP_TURNOVER" in work.columns:
        pure = (
            work.groupby(ep_groups, dropna=False)
            .agg(PURE_EPISODE_TURNOVER=("PURE_STEP_TURNOVER", "sum"))
            .reset_index()
        )
        episode = episode.merge(pure, on=ep_groups, how="left")
        episode["TERMINAL_CLOSEOUT_TURNOVER"] = episode["EPISODE_TURNOVER"] - episode["PURE_EPISODE_TURNOVER"]

    seed_level = summarize_episode_turnover(episode, groups)
    return episode, seed_level, diagnostics


def compute_from_episode_level(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    episode_col = find_col(df, EPISODE_CANDIDATES, required=False)
    turnover_col = find_col(df, ["TOTAL_TURNOVER", "total_turnover", "TURNOVER", "turnover"])
    groups = group_columns(df)

    keep = groups + [turnover_col]
    if episode_col:
        keep.append(episode_col)
    work = df[keep].copy()
    work[turnover_col] = pd.to_numeric(work[turnover_col], errors="coerce")
    work = work.dropna(subset=[turnover_col]).copy()
    work = work.rename(columns={turnover_col: "EPISODE_TURNOVER"})
    if episode_col and episode_col != "EPISODE_ID":
        work = work.rename(columns={episode_col: "EPISODE_ID"})

    seed_level = summarize_episode_turnover(work, groups)
    return work, seed_level, {"turnover_source": turnover_col, "final_closeout_detected": "unknown_episode_level"}


def aggregate_across_seeds(seed_level: pd.DataFrame) -> pd.DataFrame:
    if "VARIANT" in seed_level.columns:
        group_cols = [c for c in ["VARIANT", "ALGORITHM", "SPLIT"] if c in seed_level.columns]
    else:
        group_cols = [c for c in ["ALGORITHM", "SPLIT", "STRATEGY"] if c in seed_level.columns]
    if not group_cols:
        return seed_level.copy()

    agg = (
        seed_level.groupby(group_cols, dropna=False)
        .agg(
            N_SEEDS=("SEED", "nunique") if "SEED" in seed_level.columns else ("EPISODES", "count"),
            EPISODES=("EPISODES", "sum"),
            MEAN_TURNOVER=("MEAN_TURNOVER", "mean"),
            STD_MEAN_TURNOVER=("MEAN_TURNOVER", "std"),
            VAR_TURNOVER=("VAR_TURNOVER", "mean"),
            STD_VAR_TURNOVER=("VAR_TURNOVER", "std"),
            TURNOVER_CV=("TURNOVER_CV", "mean"),
            P95_TURNOVER=("P95_TURNOVER", "mean"),
            MAX_TURNOVER=("MAX_TURNOVER", "mean"),
        )
        .reset_index()
    )
    return agg


def make_latex_table(summary: pd.DataFrame, outpath: Path) -> None:
    table = summary.copy()
    if "SPLIT" in table.columns:
        mask = table["SPLIT"].astype(str).str.lower() == "test"
        if mask.any():
            table = table.loc[mask].copy()

    label_col = "VARIANT" if "VARIANT" in table.columns else ("STRATEGY" if "STRATEGY" in table.columns else None)
    if label_col is None:
        table["Policy"] = "ALL"
        label_col = "Policy"

    lines = []
    lines.append(r"\begin{table}[!htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Turnover variance diagnostics by policy.}")
    lines.append(r"\label{tab:turnover-variance-diagnostics}")
    lines.append(r"\begin{threeparttable}")
    lines.append(r"\small")
    lines.append(r"\begin{adjustbox}{max width=\textwidth,center}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Policy & Episodes & Mean turnover & Var(turnover) & CV & P95 turnover & Max turnover \\")
    lines.append(r"\midrule")

    def fmt(x, digits=4):
        return "--" if pd.isna(x) else f"{float(x):.{digits}f}"

    for _, row in table.iterrows():
        lines.append(
            f"{row[label_col]} & "
            f"{int(row['EPISODES'])} & "
            f"{fmt(row['MEAN_TURNOVER'])} & "
            f"{fmt(row['VAR_TURNOVER'])} & "
            f"{fmt(row['TURNOVER_CV'])} & "
            f"{fmt(row['P95_TURNOVER'])} & "
            f"{fmt(row['MAX_TURNOVER'])} \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{adjustbox}")
    lines.append(r"\begin{tablenotes}[flushleft]")
    lines.append(r"\footnotesize")
    lines.append(r"\item Notes: Episode turnover is $U_i=\sum_t \mathrm{TURNOVER}_{i,t}$ using the experiment-accounting turnover field. In the step-level parquet, this includes the terminal close-out trade on the final episode row. A high cross-episode variance of $U_i$ indicates unstable execution intensity.")
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{threeparttable}")
    lines.append(r"\end{table}")
    outpath.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", default="outputs/turnover_variance")
    parser.add_argument("--sheet", default=None)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df, source = read_input(input_path, sheet=args.sheet)
    split = None if str(args.split).lower() == "all" else args.split
    df = filter_split(df, split)

    # Identify granularity.
    hedge_col = find_col(df, HEDGE_CANDIDATES, required=False)
    prev_col = find_col(df, PREV_HEDGE_CANDIDATES, required=False)
    turnover_col = find_col(df, TURNOVER_CANDIDATES, required=False)
    episode_col = find_col(df, EPISODE_CANDIDATES, required=False)

    if episode_col and (hedge_col or prev_col or turnover_col) and "QUOTE_DATE" in df.columns:
        episode, seed_level, diagnostics = compute_from_step_level(df)
        mode = "step_level"
    elif turnover_col:
        episode, seed_level, diagnostics = compute_from_episode_level(df)
        mode = "episode_level"
    else:
        raise ValueError("Input does not contain enough columns to compute turnover variance.")

    summary = aggregate_across_seeds(seed_level)

    episode.to_csv(outdir / "turnover_episode_values.csv", index=False)
    seed_level.to_csv(outdir / "turnover_variance_by_seed.csv", index=False)
    summary.to_csv(outdir / "turnover_variance_summary.csv", index=False)
    make_latex_table(summary, outdir / "table_turnover_variance_diagnostics.tex")

    with pd.ExcelWriter(outdir / "turnover_variance_diagnostics.xlsx") as writer:
        episode.to_excel(writer, sheet_name="Episode_Turnover", index=False)
        seed_level.to_excel(writer, sheet_name="Variance_By_Seed", index=False)
        summary.to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame([diagnostics]).to_excel(writer, sheet_name="Diagnostics", index=False)

    print(f"Source: {source}")
    print(f"Mode: {mode}")
    print(f"Rows after split filter: {len(df):,}")
    print(f"Diagnostics: {diagnostics}")
    print(f"Wrote: {outdir / 'turnover_variance_summary.csv'}")
    print(f"Wrote: {outdir / 'table_turnover_variance_diagnostics.tex'}")
    print(f"Wrote: {outdir / 'turnover_variance_diagnostics.xlsx'}")


if __name__ == "__main__":
    main()
