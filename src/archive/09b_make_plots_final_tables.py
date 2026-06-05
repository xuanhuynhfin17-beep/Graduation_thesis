"""
09b_make_plots_final_tables.py

Purpose:
    Create thesis-ready plots and final summary workbook using:
        - outputs/thesis_bootstrap_paired_results.xlsx from 09a
        - existing summary sheets from result workbooks

Outputs:
    outputs/thesis_final_tables_and_plots.xlsx
    outputs/thesis_plots/*.png

Run:
    py src\\09b_make_plots_final_tables.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def candidate_project_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, Path.cwd(), Path.cwd().parent, Path("/mnt/data")]
    out = []
    for p in candidates:
        p = p.resolve()
        if p not in out:
            out.append(p)
    return out


def find_file(filename: str, relative_dirs: Iterable[str] = ("outputs", "")) -> Path | None:
    for base in candidate_project_dirs():
        for rel in relative_dirs:
            p = base / rel / filename if rel else base / filename
            if p.exists():
                return p
    return None


if (Path(__file__).resolve().parent.parent / "outputs").exists():
    PROJECT_DIR = Path(__file__).resolve().parent.parent
elif (Path.cwd() / "outputs").exists():
    PROJECT_DIR = Path.cwd()
else:
    PROJECT_DIR = Path("/mnt/data")

OUTPUT_DIR = PROJECT_DIR / "outputs"
PLOT_DIR = OUTPUT_DIR / "thesis_plots"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

BOOTSTRAP_XLSX = find_file("thesis_bootstrap_paired_results.xlsx", ("outputs", ""))
FRONTIER_XLSX = find_file("risk_cost_frontier_ppo_td3_v3c_100k_3seeds.xlsx", ("outputs", ""))
REGIME_XLSX = find_file("extended_regime_evaluation_v3c_algorithms.xlsx", ("outputs", ""))
PRETRAIN_COMP_XLSX = find_file("pretraining_regime_switching_pilot_comparison_5000_150k_FINAL.xlsx", ("outputs", ""))
OUT_XLSX = OUTPUT_DIR / "thesis_final_tables_and_plots.xlsx"


def read_xlsx(path: Path | None, sheet: str) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_excel(path, sheet_name=sheet)
    except Exception:
        return pd.DataFrame()


def savefig(path: Path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print("Saved plot:", path)


# Load final tables from 09a.
bootstrap = read_xlsx(BOOTSTRAP_XLSX, "Bootstrap_All")
paired = read_xlsx(BOOTSTRAP_XLSX, "Paired_vs_Delta")
final_main = read_xlsx(BOOTSTRAP_XLSX, "Final_Main_Comparison")
final_ablation = read_xlsx(BOOTSTRAP_XLSX, "Final_Ablation")
final_frontier = read_xlsx(BOOTSTRAP_XLSX, "Final_Risk_Cost_Frontier")
final_highcost = read_xlsx(BOOTSTRAP_XLSX, "Final_High_Cost")
final_quadratic = read_xlsx(BOOTSTRAP_XLSX, "Final_Quadratic_Impact")
final_adaptive = read_xlsx(BOOTSTRAP_XLSX, "Final_Adaptive_Band")
final_pretrain = read_xlsx(BOOTSTRAP_XLSX, "Final_Pretraining")

# Existing compact regime/pretraining sheets.
rv_compact = read_xlsx(REGIME_XLSX, "RV_Thesis_Compact")
iv_compact = read_xlsx(REGIME_XLSX, "IV_Thesis_Compact")
mny_compact = read_xlsx(REGIME_XLSX, "MONEYNESS_Thesis_Compact")
dte_compact = read_xlsx(REGIME_XLSX, "DTE_Thesis_Compact")
pretrain_test = read_xlsx(PRETRAIN_COMP_XLSX, "Test_Summary")
pretrain_high_rv = read_xlsx(PRETRAIN_COMP_XLSX, "High_RV_Test")

# -----------------------------
# Plots
# -----------------------------

# Main comparison: Mean PnL with CI.
if not final_main.empty and "MEAN_PNL" in final_main.columns:
    order = ["Delta hedge", "PPO V3C", "TD3 V3C", "SAC original", "SAC-A tuned", "SAC-C low entropy"]
    d = final_main[final_main["MODEL_LABEL"].isin(order)].copy()
    d["ORDER"] = d["MODEL_LABEL"].map({x: i for i, x in enumerate(order)})
    d = d.sort_values("ORDER")
    plt.figure(figsize=(10, 5.5))
    plt.bar(d["MODEL_LABEL"], d["MEAN_PNL"])
    if {"MEAN_CI_LOW", "MEAN_CI_HIGH"}.issubset(d.columns):
        yerr = np.vstack([d["MEAN_PNL"] - d["MEAN_CI_LOW"], d["MEAN_CI_HIGH"] - d["MEAN_PNL"]])
        plt.errorbar(d["MODEL_LABEL"], d["MEAN_PNL"], yerr=yerr, fmt="none", capsize=3)
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean terminal PnL")
    plt.title("Main Algorithm Comparison: Mean PnL with Bootstrap 95% CI")
    savefig(PLOT_DIR / "main_comparison_mean_pnl_ci.png")

    plt.figure(figsize=(10, 5.5))
    plt.bar(d["MODEL_LABEL"], d["CVAR_95"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("CVaR95 terminal PnL")
    plt.title("Main Algorithm Comparison: Tail Risk")
    savefig(PLOT_DIR / "main_comparison_cvar.png")

# Risk-cost frontier.
frontier = read_xlsx(FRONTIER_XLSX, "Test_Frontier")
if not frontier.empty:
    mean_col = "MEAN_OF_MEAN_PNL" if "MEAN_OF_MEAN_PNL" in frontier.columns else "MEAN_PNL"
    cvar_col = "MEAN_OF_CVAR_95" if "MEAN_OF_CVAR_95" in frontier.columns else "CVAR_95"
    plt.figure(figsize=(8, 6))
    for alg, g in frontier.groupby("ALGORITHM"):
        g = g.sort_values("DELTA_RISK_PENALTY")
        plt.plot(g[mean_col], g[cvar_col], marker="o", label=alg)
        for _, r in g.iterrows():
            plt.annotate(f"{r['DELTA_RISK_PENALTY']:.1f}", (r[mean_col], r[cvar_col]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.xlabel("Mean terminal PnL")
    plt.ylabel("CVaR95 terminal PnL")
    plt.title("Risk-Cost Frontier: PPO vs TD3")
    plt.legend()
    savefig(PLOT_DIR / "risk_cost_frontier_mean_vs_cvar.png")

# Ablation.
if not final_ablation.empty:
    d = final_ablation[~final_ablation["MODEL_LABEL"].isin(["Delta hedge", "No hedge"])].copy()
    d = d.sort_values("MEAN_PNL", ascending=False)
    plt.figure(figsize=(11, 5.5))
    plt.bar(d["MODEL_LABEL"], d["MEAN_PNL"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("Mean terminal PnL")
    plt.title("Ablation Study: Mean PnL by Variant")
    savefig(PLOT_DIR / "ablation_mean_pnl.png")

    d2 = d.sort_values("CVAR_95", ascending=False)
    plt.figure(figsize=(11, 5.5))
    plt.bar(d2["MODEL_LABEL"], d2["CVAR_95"])
    plt.xticks(rotation=35, ha="right")
    plt.ylabel("CVaR95 terminal PnL")
    plt.title("Ablation Study: CVaR95 by Variant")
    savefig(PLOT_DIR / "ablation_cvar.png")

# Pretraining.
if not final_pretrain.empty:
    order = ["E0_no_pretrain", "E1_bs_pretrain", "E2_ms_gbm_pretrain", "E3_ms_gbm_proxy_pretrain"]
    d = final_pretrain[final_pretrain["MODEL_LABEL"].isin(order)].copy()
    d["ORDER"] = d["MODEL_LABEL"].map({x: i for i, x in enumerate(order)})
    d = d.sort_values("ORDER")
    plt.figure(figsize=(10, 5.5))
    plt.bar(d["MODEL_LABEL"], d["MEAN_PNL"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("Mean terminal PnL")
    plt.title("Pretraining Transfer: Mean PnL")
    savefig(PLOT_DIR / "pretraining_mean_pnl.png")
    plt.figure(figsize=(10, 5.5))
    plt.bar(d["MODEL_LABEL"], d["CVAR_95"])
    plt.xticks(rotation=25, ha="right")
    plt.ylabel("CVaR95 terminal PnL")
    plt.title("Pretraining Transfer: Tail Risk")
    savefig(PLOT_DIR / "pretraining_cvar.png")

# RV regime compact plot.
if not rv_compact.empty:
    mean_col = "MEAN_PNL" if "MEAN_PNL" in rv_compact.columns else "MEAN_OF_MEAN_PNL" if "MEAN_OF_MEAN_PNL" in rv_compact.columns else None
    label_col = "ALGORITHM" if "ALGORITHM" in rv_compact.columns else "STRATEGY" if "STRATEGY" in rv_compact.columns else None
    if mean_col and label_col and "RV_REGIME" in rv_compact.columns:
        d = rv_compact.copy()
        if "SPLIT" in d.columns:
            d = d[d["SPLIT"].eq("test")]
        pivot = d.pivot_table(index="RV_REGIME", columns=label_col, values=mean_col, aggfunc="mean")
        if not pivot.empty:
            pivot.plot(kind="bar", figsize=(10, 5.5))
            plt.ylabel("Mean terminal PnL")
            plt.title("Regime Analysis: Realized Volatility Regimes")
            plt.xticks(rotation=0)
            savefig(PLOT_DIR / "regime_rv_mean_pnl.png")

# -----------------------------
# Final workbook
# -----------------------------
with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
    workbook = writer.book
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})

    def write(df: pd.DataFrame, sheet: str):
        if df is None or df.empty:
            df = pd.DataFrame({"MESSAGE": ["No data available."]})
        df.to_excel(writer, sheet_name=sheet[:31], index=False)
        ws = writer.sheets[sheet[:31]]
        ws.freeze_panes(1, 0)
        ws.set_row(0, None, header_fmt)
        for i, col in enumerate(df.columns):
            max_len = max([len(str(col))] + [len(str(x)) for x in df[col].head(100).fillna("")])
            ws.set_column(i, i, min(max(max_len + 2, 10), 36))

    index = pd.DataFrame([
        {"SHEET": "Final_Main", "DESCRIPTION": "Main test comparison."},
        {"SHEET": "Final_Ablation", "DESCRIPTION": "Ablation study."},
        {"SHEET": "Final_Frontier", "DESCRIPTION": "Risk-cost frontier."},
        {"SHEET": "Final_Robustness", "DESCRIPTION": "High cost and quadratic impact."},
        {"SHEET": "Final_Adaptive", "DESCRIPTION": "Adaptive no-trade band."},
        {"SHEET": "Final_Pretraining", "DESCRIPTION": "Pretraining transfer."},
        {"SHEET": "Paired_vs_Delta", "DESCRIPTION": "Paired test differences versus Delta."},
    ])
    write(index, "Index")
    write(final_main, "Final_Main")
    write(final_ablation, "Final_Ablation")
    write(final_frontier, "Final_Frontier")
    robustness = pd.concat([
        final_highcost.assign(SETTING="High cost"),
        final_quadratic.assign(SETTING="Quadratic impact"),
    ], ignore_index=True, sort=False)
    write(robustness, "Final_Robustness")
    write(final_adaptive, "Final_Adaptive")
    write(final_pretrain, "Final_Pretraining")
    write(paired[paired.get("SPLIT", pd.Series()).eq("test")] if not paired.empty else paired, "Paired_vs_Delta")
    write(rv_compact, "Regime_RV")
    write(iv_compact, "Regime_IV")
    write(mny_compact, "Regime_Moneyness")
    write(dte_compact, "Regime_DTE")
    write(pretrain_test, "Pretrain_Test")
    write(pretrain_high_rv, "Pretrain_High_RV")
    write(bootstrap, "Bootstrap_All")

print("Saved:", OUT_XLSX)
print("Plots:", PLOT_DIR)
