"""
10e_compare_pretraining_quantile_extension.py

Combine previous pretraining results with E4 quantile option-state pretraining.

Inputs:
    outputs/thesis_final_tables_and_plots.xlsx   (preferred, if available)
    outputs/pretraining_regime_switching_pilot_comparison.xlsx (fallback)
    outputs/pretraining_quantile_option_state_results_final.xlsx
        or outputs/pretraining_quantile_option_state_results_pilot.xlsx

Outputs:
    outputs/pretraining_transfer_extended_quantile_summary.xlsx
    tables/chapter5_pretraining_quantile_extension.tex

Run:
    py src\10e_compare_pretraining_quantile_extension.py --mode final
    py src\10e_compare_pretraining_quantile_extension.py --mode pilot
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_DIR / "outputs"
TABLE_DIR = PROJECT_DIR / "tables"
TABLE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

THESIS_TABLES = OUT_DIR / "thesis_final_tables_and_plots.xlsx"
OLD_PRETRAIN = OUT_DIR / "pretraining_regime_switching_pilot_comparison.xlsx"


def read_old_pretraining() -> pd.DataFrame:
    # Preferred: current thesis summary workbook.
    if THESIS_TABLES.exists():
        try:
            d = pd.read_excel(THESIS_TABLES, sheet_name="Pretrain_Test")
            d = d.copy()
            d["SOURCE"] = "Pretrain_Test"
            if "EXPERIMENT" in d.columns:
                return normalize_pretrain_test(d)
        except Exception:
            pass
        try:
            d = pd.read_excel(THESIS_TABLES, sheet_name="Final_Pretraining")
            d = d.copy()
            d["SOURCE"] = "Final_Pretraining"
            return normalize_final_pretraining(d)
        except Exception:
            pass

    # Fallback: old comparison workbook.
    if OLD_PRETRAIN.exists():
        for sheet in ["Test_Summary", "Experiment_Summary_Recalc", "Experiment_Summary"]:
            try:
                d = pd.read_excel(OLD_PRETRAIN, sheet_name=sheet)
                d = d[d.get("SPLIT", "test").astype(str).str.lower().eq("test")] if "SPLIT" in d.columns else d
                d["SOURCE"] = sheet
                return normalize_pretrain_test(d)
            except Exception:
                continue

    print("Warning: could not find old pretraining summary. Only E4 will be reported.")
    return pd.DataFrame()


def normalize_pretrain_test(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["EXPERIMENT"] = d["EXPERIMENT"] if "EXPERIMENT" in d.columns else d.get("MODEL_LABEL", "Unknown")
    out["SPLIT"] = d["SPLIT"] if "SPLIT" in d.columns else "test"

    # Seed-level summary style
    mapping = {
        "MEAN_OF_MEAN_PNL": ["MEAN_OF_MEAN_PNL", "MEAN_PNL"],
        "STD_OF_MEAN_PNL": ["STD_OF_MEAN_PNL", "STD_PNL"],
        "MEAN_OF_CVAR_95": ["MEAN_OF_CVAR_95", "CVAR_95"],
        "MEAN_OF_SHARPE_LIKE": ["MEAN_OF_SHARPE_LIKE", "SHARPE_LIKE"],
        "MEAN_TC": ["MEAN_TC", "TOTAL_TC"],
        "MEAN_TURNOVER": ["MEAN_TURNOVER", "TOTAL_TURNOVER"],
        "NO_TRADE_RATE": ["NO_TRADE_RATE"],
        "N_SEEDS": ["N_SEEDS"],
    }
    for new, candidates in mapping.items():
        out[new] = np.nan
        for c in candidates:
            if c in d.columns:
                out[new] = d[c]
                break
    return out


def normalize_final_pretraining(d: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["EXPERIMENT"] = d["MODEL_LABEL"] if "MODEL_LABEL" in d.columns else d.get("EXPERIMENT", "Unknown")
    out["SPLIT"] = d["SPLIT"] if "SPLIT" in d.columns else "test"
    out["MEAN_OF_MEAN_PNL"] = d.get("MEAN_PNL", np.nan)
    out["STD_OF_MEAN_PNL"] = d.get("STD_PNL", np.nan)
    out["MEAN_OF_CVAR_95"] = d.get("CVAR_95", np.nan)
    out["MEAN_OF_SHARPE_LIKE"] = d.get("SHARPE_LIKE", np.nan)
    out["MEAN_TC"] = d.get("MEAN_TC", np.nan)
    out["MEAN_TURNOVER"] = d.get("MEAN_TURNOVER", np.nan)
    out["NO_TRADE_RATE"] = d.get("NO_TRADE_RATE", np.nan)
    out["N_SEEDS"] = d.get("N_SEEDS", np.nan)
    return out


def read_e4(mode: str) -> pd.DataFrame:
    path = OUT_DIR / f"pretraining_quantile_option_state_results_{mode}.xlsx"
    if not path.exists():
        alt = OUT_DIR / "pretraining_quantile_option_state_results_final.xlsx"
        if alt.exists():
            path = alt
    if not path.exists():
        raise FileNotFoundError(f"Cannot find E4 results: {path}")

    d = pd.read_excel(path, sheet_name="Experiment_Summary")
    d = d[d["SPLIT"].astype(str).str.lower().eq("test")].copy()
    d["EXPERIMENT"] = "E4_ms_gbm_quantile_option_state_pretrain"
    return normalize_pretrain_test(d)


def clean_label(x: str) -> str:
    mapping = {
        "E0_no_pretrain": "No pretraining",
        "E1_bs_pretrain": "BS pretraining",
        "E2_ms_gbm_pretrain": "MS-GBM pretraining",
        "E3_ms_gbm_proxy_pretrain": "MS-GBM + proxies",
        "E4_ms_gbm_quantile_option_state_pretrain": "MS-GBM + quantile option-state",
    }
    return mapping.get(str(x), str(x).replace("_", " "))


def make_latex_table(summary: pd.DataFrame) -> str:
    d = summary.copy()
    d = d[d["SPLIT"].astype(str).str.lower().eq("test")] if "SPLIT" in d.columns else d
    d["LABEL"] = d["EXPERIMENT"].map(clean_label)
    cols = ["LABEL", "N_SEEDS", "MEAN_OF_MEAN_PNL", "MEAN_OF_CVAR_95", "MEAN_OF_SHARPE_LIKE", "MEAN_TC", "MEAN_TURNOVER", "NO_TRADE_RATE"]

    def fmt(x, nd=2):
        if pd.isna(x):
            return "--"
        return f"{float(x):.{nd}f}"

    rows = []
    for _, r in d[cols].iterrows():
        rows.append(
            f"{r['LABEL']} & {fmt(r['N_SEEDS'],0)} & {fmt(r['MEAN_OF_MEAN_PNL'])} & {fmt(r['MEAN_OF_CVAR_95'])} & "
            f"{fmt(r['MEAN_OF_SHARPE_LIKE'],3)} & {fmt(r['MEAN_TC'])} & {fmt(r['MEAN_TURNOVER'])} & {fmt(r['NO_TRADE_RATE'],3)} \\\\"
        )

    return r"""
\begin{table}[H]
\centering
\scriptsize
\begin{threeparttable}
\caption{Extended pretraining comparison including quantile option-state simulation.}
\label{tab:pretraining-quantile-extension}
\begin{tabularx}{\textwidth}{L{4.1cm}rrrrrrr}
\toprule
Pretraining method & Seeds & Mean PnL & CVaR95 & Sharpe-like & TC & Turnover & No-trade \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\footnotesize
\item The quantile option-state simulator augments HMM-MSGBM paths with quantile-regression-generated implied volatility and spread, followed by Black--Scholes recomputation of option prices and Greeks. Metrics are reported on the real held-out test set.
\end{tablenotes}
\end{threeparttable}
\end{table}
""".strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "final"], default="final")
    args = parser.parse_args()

    old = read_old_pretraining()
    e4 = read_e4(args.mode)
    summary = pd.concat([old, e4], ignore_index=True, sort=False)

    # Remove exact duplicates, keep last occurrence for E4.
    summary = summary.drop_duplicates(subset=["EXPERIMENT", "SPLIT"], keep="last")

    xlsx = OUT_DIR / f"pretraining_transfer_extended_quantile_summary_{args.mode}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Extended_Summary", index=False)

    tex = make_latex_table(summary)
    tex_path = TABLE_DIR / "chapter5_pretraining_quantile_extension.tex"
    tex_path.write_text(tex + "\n", encoding="utf-8")

    print(f"Saved: {xlsx}")
    print(f"Saved: {tex_path}")
    print(summary)


if __name__ == "__main__":
    main()
