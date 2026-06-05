"""
10g_compare_e4_quantile_algorithms.py

Compare E4 quantile option-state pretraining across PPO, TD3 and SAC.

Inputs:
    outputs/pretraining_quantile_option_state_results_<mode>.xlsx
        - PPO E4 results from 10d
    outputs/pretraining_quantile_option_state_td3_sac_results_<mode>.xlsx
        - TD3/SAC E4 results from 10f

Outputs:
    outputs/e4_quantile_algorithm_robustness_<mode>.xlsx
    tables/chapter5_e4_quantile_algorithm_robustness.tex

Run:
    py src\10g_compare_e4_quantile_algorithms.py --mode final
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "outputs"
TABLE_DIR = PROJECT_DIR / "tables"

OUT_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)


def load_ppo(mode: str) -> pd.DataFrame:
    path = OUT_DIR / f"pretraining_quantile_option_state_results_{mode}.xlsx"
    if not path.exists():
        print(f"Warning: PPO E4 results not found: {path}")
        return pd.DataFrame()
    d = pd.read_excel(path, sheet_name="Experiment_Summary")
    d = d.copy()
    d["ALGORITHM"] = "PPO"
    d["EXPERIMENT"] = "E4_quantile_option_state_PPO"
    return normalize(d)


def load_td3_sac(mode: str) -> pd.DataFrame:
    path = OUT_DIR / f"pretraining_quantile_option_state_td3_sac_results_{mode}.xlsx"
    if not path.exists():
        print(f"Warning: TD3/SAC E4 results not found: {path}")
        return pd.DataFrame()
    d = pd.read_excel(path, sheet_name="Algorithm_Summary")
    return normalize(d)


def normalize(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    if "SPLIT" not in out.columns:
        out["SPLIT"] = "test"
    if "ALGORITHM" not in out.columns:
        out["ALGORITHM"] = "Unknown"
    if "EXPERIMENT" not in out.columns:
        out["EXPERIMENT"] = "E4_quantile_option_state_" + out["ALGORITHM"].astype(str)

    cols = [
        "EXPERIMENT", "ALGORITHM", "SPLIT", "N_SEEDS", "MEAN_OF_MEAN_PNL",
        "STD_OF_MEAN_PNL", "MEAN_OF_CVAR_95", "STD_OF_CVAR_95",
        "MEAN_OF_SHARPE_LIKE", "MEAN_TC", "MEAN_TURNOVER",
        "NO_TRADE_RATE", "AVG_ADJUSTMENT_FROM_DELTA",
        "ACTION_NEAR_NEG1_RATE", "ACTION_NEAR_POS1_RATE",
        "TRAINING_TIME_MIN"
    ]

    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols]


def fmt(x, nd=2):
    if pd.isna(x):
        return "--"
    return f"{float(x):.{nd}f}"


def make_latex(summary: pd.DataFrame) -> str:
    d = summary[summary["SPLIT"].astype(str).str.lower().eq("test")].copy()
    order = {"PPO": 0, "TD3": 1, "SAC": 2}
    d["_ORDER"] = d["ALGORITHM"].map(order).fillna(99)
    d = d.sort_values("_ORDER")

    rows = []
    for _, r in d.iterrows():
        rows.append(
            f"{r['ALGORITHM']} & {fmt(r['N_SEEDS'],0)} & {fmt(r['MEAN_OF_MEAN_PNL'])} & "
            f"{fmt(r['MEAN_OF_CVAR_95'])} & {fmt(r['MEAN_OF_SHARPE_LIKE'],3)} & "
            f"{fmt(r['MEAN_TC'])} & {fmt(r['MEAN_TURNOVER'])} & {fmt(r['NO_TRADE_RATE'],3)} & "
            f"{fmt(r['AVG_ADJUSTMENT_FROM_DELTA'],4)} \\\\"
        )

    return r"""
\begin{table}[H]
\centering
\scriptsize
\begin{threeparttable}
\caption{Algorithm robustness under quantile option-state pretraining.}
\label{tab:e4-quantile-algorithm-robustness}
\begin{tabularx}{\textwidth}{lrrrrrrrr}
\toprule
Algorithm & Seeds & Mean PnL & CVaR95 & Sharpe-like & TC & Turnover & No-trade & Avg. residual \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\footnotesize
\item The table applies the same quantile option-state pretraining regime to each algorithm. It is intended as a robustness check for algorithm--pretraining interaction, not as a replacement for the main algorithm comparison.
\end{tablenotes}
\end{threeparttable}
\end{table}
""".strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "final"], default="final")
    args = parser.parse_args()

    parts = []
    ppo = load_ppo(args.mode)
    td3sac = load_td3_sac(args.mode)
    if not ppo.empty:
        parts.append(ppo)
    if not td3sac.empty:
        parts.append(td3sac)
    if not parts:
        raise FileNotFoundError("No E4 quantile algorithm results found.")

    summary = pd.concat(parts, ignore_index=True)
    summary = summary.drop_duplicates(subset=["ALGORITHM", "SPLIT"], keep="last")

    xlsx = OUT_DIR / f"e4_quantile_algorithm_robustness_{args.mode}.xlsx"
    tex_path = TABLE_DIR / "chapter5_e4_quantile_algorithm_robustness.tex"

    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Algorithm_Robustness", index=False)

    tex_path.write_text(make_latex(summary) + "\n", encoding="utf-8")

    print(f"Saved: {xlsx}")
    print(f"Saved: {tex_path}")
    print(summary[summary["SPLIT"].astype(str).str.lower().eq("test")].to_string(index=False))


if __name__ == "__main__":
    main()
