"""
10c_validate_quantile_simulator.py

Validate state distribution matching for MS-GBM + Gradient Boosting Quantile simulator.

Outputs:
    outputs/quantile_simulator_state_match.xlsx
    outputs/quantile_simulator_state_match.csv
    outputs/quantile_simulator_figures/*.png

Run:
    py src\10c_validate_quantile_simulator.py --sim data\simulated\ms_gbm_quantile_option_state_final_n5000.parquet
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import wasserstein_distance
except Exception:
    wasserstein_distance = None


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
SIM_DIR = PROJECT_DIR / "data" / "simulated"
OUT_DIR = PROJECT_DIR / "outputs"
FIG_DIR = OUT_DIR / "quantile_simulator_figures"
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(exist_ok=True)

REAL_PROXY_FILE = DATA_DIR / "transitions_daily_top1_final_with_spy_2010_2023_with_regime_proxies.parquet"
REAL_BASE_FILE = DATA_DIR / "transitions_daily_top1_final_with_spy_2010_2023.parquet"
DEFAULT_SIM = SIM_DIR / "ms_gbm_quantile_option_state_final_n5000.parquet"


def load_real() -> pd.DataFrame:
    path = REAL_PROXY_FILE if REAL_PROXY_FILE.exists() else REAL_BASE_FILE
    if not path.exists():
        raise FileNotFoundError("Cannot find real transition file.")
    df = pd.read_parquet(path)
    return df[df["SPLIT"].astype(str).str.lower().eq("train")].copy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "SPY_MONEYNESS" not in out.columns:
        out["SPY_MONEYNESS"] = out["SPY_CLOSE"].astype(float) / out["STRIKE"].astype(float)
    if "SPY_LOG_MONEYNESS" not in out.columns:
        out["SPY_LOG_MONEYNESS"] = np.log(out["SPY_MONEYNESS"].astype(float))
    if "OPTION_MID_OVER_SPY" not in out.columns:
        out["OPTION_MID_OVER_SPY"] = out["OPTION_MID"].astype(float) / out["SPY_CLOSE"].astype(float)
    out["OPTION_IV"] = pd.to_numeric(out.get("OPTION_IV", np.nan), errors="coerce")
    out.loc[out["OPTION_IV"] > 3.0, "OPTION_IV"] = out.loc[out["OPTION_IV"] > 3.0, "OPTION_IV"] / 100.0
    if "OPTION_SPREAD_PCT" not in out.columns:
        if "SPREAD_PROXY" in out.columns:
            out["OPTION_SPREAD_PCT"] = out["SPREAD_PROXY"]
        else:
            out["OPTION_SPREAD_PCT"] = np.nan
    return out


def metric_row(var: str, real: pd.Series, sim: pd.Series) -> dict:
    r = pd.to_numeric(real, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s = pd.to_numeric(sim, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0 or len(s) == 0:
        wd = np.nan
    elif wasserstein_distance is not None:
        wd = float(wasserstein_distance(r.values, s.values))
    else:
        wd = abs(float(r.mean()) - float(s.mean()))
    return {
        "VARIABLE": var,
        "REAL_N": len(r),
        "SIM_N": len(s),
        "REAL_MEAN": r.mean(),
        "SIM_MEAN": s.mean(),
        "ABS_MEAN_DIFF": abs(r.mean() - s.mean()),
        "REAL_STD": r.std(),
        "SIM_STD": s.std(),
        "STD_RATIO_SIM_REAL": s.std() / r.std() if r.std() != 0 else np.nan,
        "REAL_Q05": r.quantile(0.05),
        "SIM_Q05": s.quantile(0.05),
        "REAL_Q50": r.quantile(0.50),
        "SIM_Q50": s.quantile(0.50),
        "REAL_Q95": r.quantile(0.95),
        "SIM_Q95": s.quantile(0.95),
        "WASSERSTEIN": wd,
    }


def episode_rv(df: pd.DataFrame) -> pd.Series:
    d = df.copy()
    d["LOG_RET"] = np.log(d["SPY_NEXT_CLOSE"].astype(float) / d["SPY_CLOSE"].astype(float))
    return d.groupby("EPISODE_ID")["LOG_RET"].std() * math.sqrt(252)


def plot_hist(real: pd.Series, sim: pd.Series, var: str, out_path: Path):
    r = pd.to_numeric(real, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    s = pd.to_numeric(sim, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) == 0 or len(s) == 0:
        return
    lo = np.nanquantile(pd.concat([r, s]), 0.01)
    hi = np.nanquantile(pd.concat([r, s]), 0.99)
    r = r[(r >= lo) & (r <= hi)]
    s = s[(s >= lo) & (s <= hi)]
    plt.figure(figsize=(8, 5))
    plt.hist(r, bins=50, alpha=0.5, density=True, label="Real train")
    plt.hist(s, bins=50, alpha=0.5, density=True, label="Simulated")
    plt.xlabel(var)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", type=str, default=str(DEFAULT_SIM))
    args = parser.parse_args()

    real = add_features(load_real())
    sim_path = Path(args.sim)
    if not sim_path.is_absolute():
        sim_path = PROJECT_DIR / sim_path
    if not sim_path.exists():
        raise FileNotFoundError(f"Cannot find simulated file: {sim_path}")

    sim = add_features(pd.read_parquet(sim_path))
    print(f"Loaded real train: {real.shape}")
    print(f"Loaded sim: {sim.shape}")

    variables = [
        "OPTION_IV", "OPTION_SPREAD_PCT", "OPTION_DELTA", "OPTION_GAMMA", "OPTION_VEGA",
        "OPTION_THETA", "SPY_MONEYNESS", "SPY_LOG_MONEYNESS", "DTE", "OPTION_MID_OVER_SPY"
    ]

    rows = []
    for var in variables:
        if var in real.columns and var in sim.columns:
            rows.append(metric_row(var, real[var], sim[var]))
            plot_hist(real[var], sim[var], var, FIG_DIR / f"{var.lower()}_real_vs_sim.png")

    rows.append(metric_row("EPISODE_RV", episode_rv(real), episode_rv(sim)))
    metrics = pd.DataFrame(rows).sort_values("VARIABLE")

    csv_path = OUT_DIR / "quantile_simulator_state_match.csv"
    xlsx_path = OUT_DIR / "quantile_simulator_state_match.xlsx"
    metrics.to_csv(csv_path, index=False)
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="State_Match", index=False)
        sim.head(5000).to_excel(writer, sheet_name="Sim_Sample", index=False)

    print(f"Saved: {csv_path}")
    print(f"Saved: {xlsx_path}")
    print(f"Saved figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
