"""
09a_bootstrap_ci_paired_delta.py

Purpose:
    1. Bootstrap confidence intervals for Mean terminal PnL and CVaR95.
    2. Paired episode-level difference versus Delta hedge.

Why this script exists:
    Thesis result workbooks are large. This script reads only Episode_Results,
    normalizes multi-seed results to one value per EPISODE_ID by averaging across
    seeds, and caches extracted sheets as parquet for faster reruns.

Output:
    outputs/thesis_bootstrap_paired_results.xlsx
    outputs/thesis_cache/*.parquet

Run:
    py src\\09a_bootstrap_ci_paired_delta.py

Notes:
    - First run can be slow because .xlsx files are large.
    - Reruns are faster because Episode_Results sheets are cached as parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable
import numpy as np
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

BOOTSTRAP_N = 2000
RANDOM_SEED = 42
ONLY_SPLIT = "test"   # set to None to include train/val/test

# Set INCLUDE_HEAVY_TABLES=False if you only want main/pretraining tables first.
INCLUDE_HEAVY_TABLES = True

INPUT_WORKBOOKS = [
    # Main comparison
    {
        "analysis": "main_comparison",
        "source": "main_ppo",
        "filename": "comparison_algorithms_residual_delta_v3c_ppo_multiseed.xlsx",
        "label_override": {"PPO": "PPO V3C"},
        "keep_baseline": True,
    },
    {
        "analysis": "main_comparison",
        "source": "main_td3",
        "filename": "comparison_algorithms_residual_delta_v3c_td3_multiseed.xlsx",
        "label_override": {"TD3": "TD3 V3C"},
        "keep_baseline": False,
    },
    {
        "analysis": "main_comparison",
        "source": "main_sac_original",
        "filename": "comparison_algorithms_residual_delta_v3c_sac_multiseed.xlsx",
        "label_override": {"SAC": "SAC original"},
        "keep_baseline": False,
    },
    {
        "analysis": "main_comparison",
        "source": "sac_a_tuned",
        "filename": "comparison_sac_v3c_paper_like_100k_3seeds.xlsx.xlsx",
        "label_override": {"SAC": "SAC-A tuned"},
        "keep_baseline": False,
    },
    {
        "analysis": "main_comparison",
        "source": "sac_c_low_entropy",
        "filename": "comparison_sac_v3c_low_target_entropy_100k_3seeds.xlsx",
        "label_override": {"SAC": "SAC-C low entropy"},
        "keep_baseline": False,
    },

    # Method / robustness / extensions
    {"analysis": "ablation", "source": "ablation", "filename": "ablation_ppo_v3c_100k_3seeds.xlsx", "keep_baseline": True},
    {"analysis": "risk_cost_frontier", "source": "frontier", "filename": "risk_cost_frontier_ppo_td3_v3c_100k_3seeds.xlsx", "keep_baseline": True},
    {"analysis": "high_cost", "source": "high_cost", "filename": "comparison_ppo_td3_residual_delta_v3c_highcost_100k_3seeds.xlsx", "keep_baseline": True},
    {"analysis": "quadratic_impact", "source": "quadratic", "filename": "comparison_ppo_td3_residual_delta_v3c_quadratic_impact_100k_3seeds.xlsx", "keep_baseline": True},
    {"analysis": "adaptive_band", "source": "adaptive_band", "filename": "comparison_ppo_td3_adaptive_no_trade_band_v3c_100k_3seeds.xlsx", "keep_baseline": True},
    {"analysis": "pretraining", "source": "pretraining", "filename": "pretraining_regime_switching_pilot_ppo_5000_150k_FINAL.xlsx", "keep_baseline": True},
]


# ============================================================
# PATH HELPERS
# ============================================================

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
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = OUTPUT_DIR / "thesis_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_XLSX = OUTPUT_DIR / "thesis_bootstrap_paired_results.xlsx"


# ============================================================
# METRICS
# ============================================================

def cvar_95(x) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    q = np.quantile(arr, 0.05)
    tail = arr[arr <= q]
    return float(tail.mean()) if tail.size else np.nan


def sharpe_like(x) -> float:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return np.nan
    sd = arr.std(ddof=1)
    return float(arr.mean() / sd) if sd != 0 else np.nan


def bootstrap_mean_cvar(x, n_boot=BOOTSTRAP_N, seed=RANDOM_SEED) -> dict:
    arr = np.asarray(x, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return {"N_EPISODES": 0, "MEAN_PNL": np.nan, "MEAN_CI_LOW": np.nan, "MEAN_CI_HIGH": np.nan, "CVAR_95": np.nan, "CVAR_CI_LOW": np.nan, "CVAR_CI_HIGH": np.nan, "STD_PNL": np.nan, "SHARPE_LIKE": np.nan}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = arr[idx]
    boot_means = samples.mean(axis=1)
    boot_cvars = np.array([cvar_95(row) for row in samples])
    return {
        "N_EPISODES": n,
        "MEAN_PNL": float(arr.mean()),
        "MEAN_CI_LOW": float(np.quantile(boot_means, 0.025)),
        "MEAN_CI_HIGH": float(np.quantile(boot_means, 0.975)),
        "CVAR_95": cvar_95(arr),
        "CVAR_CI_LOW": float(np.nanquantile(boot_cvars, 0.025)),
        "CVAR_CI_HIGH": float(np.nanquantile(boot_cvars, 0.975)),
        "STD_PNL": float(arr.std(ddof=1)) if n > 1 else np.nan,
        "SHARPE_LIKE": sharpe_like(arr),
    }


def bootstrap_paired_diff(diff, n_boot=BOOTSTRAP_N, seed=RANDOM_SEED) -> dict:
    arr = np.asarray(diff, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n == 0:
        return {"N_MATCHED_EPISODES": 0, "MEAN_DIFF_VS_DELTA": np.nan, "MEAN_DIFF_CI_LOW": np.nan, "MEAN_DIFF_CI_HIGH": np.nan, "MEDIAN_DIFF_VS_DELTA": np.nan, "PCT_OUTPERFORM_DELTA": np.nan, "CVAR95_OF_DIFF": np.nan}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = arr[idx].mean(axis=1)
    return {
        "N_MATCHED_EPISODES": n,
        "MEAN_DIFF_VS_DELTA": float(arr.mean()),
        "MEAN_DIFF_CI_LOW": float(np.quantile(boot_means, 0.025)),
        "MEAN_DIFF_CI_HIGH": float(np.quantile(boot_means, 0.975)),
        "MEDIAN_DIFF_VS_DELTA": float(np.median(arr)),
        "PCT_OUTPERFORM_DELTA": float((arr > 0).mean()),
        "CVAR95_OF_DIFF": cvar_95(arr),
    }


# ============================================================
# DATA LOADING AND NORMALIZATION
# ============================================================

def read_episode_results_with_cache(path: Path, source: str) -> pd.DataFrame:
    cache_path = CACHE_DIR / f"{source}_Episode_Results.parquet"
    # Refresh cache when workbook is newer.
    if cache_path.exists() and cache_path.stat().st_mtime >= path.stat().st_mtime:
        return pd.read_parquet(cache_path)

    print(f"Reading Episode_Results from {path.name} ...")
    cols = None
    df = pd.read_excel(path, sheet_name="Episode_Results", usecols=cols)
    df.to_parquet(cache_path, index=False)
    print(f"Cached: {cache_path}")
    return df


def label_row(row: pd.Series, spec: dict) -> str:
    strategy = str(row.get("STRATEGY", ""))
    alg = str(row.get("ALGORITHM", ""))

    if strategy == "delta":
        return "Delta hedge"
    if strategy == "no_hedge":
        return "No hedge"

    overrides = spec.get("label_override", {}) or {}
    if alg in overrides:
        return overrides[alg]

    if pd.notna(row.get("VARIANT", np.nan)):
        var = str(row.get("VARIANT"))
        if var not in ["baseline", "nan"]:
            return var

    if pd.notna(row.get("FRONTIER_VARIANT", np.nan)) and pd.notna(row.get("DELTA_RISK_PENALTY", np.nan)):
        var = str(row.get("FRONTIER_VARIANT"))
        if var != "baseline":
            return f"{alg} penalty={float(row['DELTA_RISK_PENALTY']):.1f}"

    if pd.notna(row.get("EXPERIMENT", np.nan)):
        exp = str(row.get("EXPERIMENT"))
        if exp != "baseline":
            return exp

    if alg and alg not in ["baseline", "nan"]:
        return alg
    return strategy


def normalize_episode_df(raw: pd.DataFrame, spec: dict) -> pd.DataFrame:
    df = raw.copy()
    if not spec.get("keep_baseline", True) and "ALGORITHM" in df.columns:
        df = df[df["ALGORITHM"].astype(str).str.lower().ne("baseline")].copy()
    if ONLY_SPLIT is not None:
        df = df[df["SPLIT"].astype(str).eq(ONLY_SPLIT)].copy()

    df["ANALYSIS"] = spec["analysis"]
    df["SOURCE"] = spec["source"]
    df["MODEL_LABEL"] = df.apply(lambda r: label_row(r, spec), axis=1)

    # Average across seeds at episode level to avoid treating seed evaluations on the
    # same episode as independent observations.
    key_cols = ["ANALYSIS", "SOURCE", "MODEL_LABEL", "SPLIT", "EPISODE_ID"]
    for c in ["ALGORITHM", "VARIANT", "FRONTIER_VARIANT", "DELTA_RISK_PENALTY", "EXPERIMENT"]:
        if c in df.columns:
            key_cols.insert(-2, c)

    metrics = [
        "TERMINAL_PNL", "TOTAL_TC", "TOTAL_TURNOVER", "AVG_HEDGE", "AVG_DELTA",
        "AVG_ADJUSTMENT_FROM_DELTA", "NO_TRADE_RATE", "TRAINING_TIME_MIN", "AVG_ADAPTIVE_BAND",
    ]
    metrics = [c for c in metrics if c in df.columns]
    episode_level = df.groupby(key_cols, dropna=False)[metrics].mean().reset_index()
    return episode_level


# ============================================================
# MAIN PROCESS
# ============================================================

all_episode = []
for spec in INPUT_WORKBOOKS:
    if not INCLUDE_HEAVY_TABLES and spec["analysis"] not in ["main_comparison", "pretraining"]:
        continue
    path = find_file(spec["filename"], ("outputs", ""))
    if path is None:
        print(f"Skipping missing file: {spec['filename']}")
        continue
    raw = read_episode_results_with_cache(path, spec["source"])
    all_episode.append(normalize_episode_df(raw, spec))

if not all_episode:
    raise RuntimeError("No input Episode_Results sheets found.")

episodes = pd.concat(all_episode, ignore_index=True, sort=False)

# Bootstrap CI.
bootstrap_rows = []
group_cols = [c for c in ["ANALYSIS", "SOURCE", "MODEL_LABEL", "SPLIT", "ALGORITHM", "VARIANT", "FRONTIER_VARIANT", "DELTA_RISK_PENALTY", "EXPERIMENT"] if c in episodes.columns]
for keys, g in episodes.groupby(group_cols, dropna=False):
    if not isinstance(keys, tuple):
        keys = (keys,)
    row = dict(zip(group_cols, keys))
    row.update(bootstrap_mean_cvar(g["TERMINAL_PNL"].values))
    for c in ["TOTAL_TC", "TOTAL_TURNOVER", "AVG_HEDGE", "AVG_DELTA", "AVG_ADJUSTMENT_FROM_DELTA", "NO_TRADE_RATE", "AVG_ADAPTIVE_BAND"]:
        if c in g.columns:
            row[f"MEAN_{c}"] = g[c].mean()
    bootstrap_rows.append(row)
bootstrap_df = pd.DataFrame(bootstrap_rows).sort_values(["ANALYSIS", "SPLIT", "MODEL_LABEL"])

# Paired delta differences.
paired_rows = []
for (analysis, split), d in episodes.groupby(["ANALYSIS", "SPLIT"], dropna=False):
    delta = d[d["MODEL_LABEL"].eq("Delta hedge")][["EPISODE_ID", "TERMINAL_PNL"]].rename(columns={"TERMINAL_PNL": "DELTA_PNL"})
    if delta.empty:
        continue
    for label, g in d.groupby("MODEL_LABEL"):
        if label in ["Delta hedge", "No hedge"]:
            continue
        model = g[["EPISODE_ID", "TERMINAL_PNL"]].rename(columns={"TERMINAL_PNL": "MODEL_PNL"})
        m = model.merge(delta, on="EPISODE_ID", how="inner")
        if m.empty:
            continue
        diff = m["MODEL_PNL"].values - m["DELTA_PNL"].values
        row = {"ANALYSIS": analysis, "SPLIT": split, "MODEL_LABEL": label}
        row.update(bootstrap_paired_diff(diff))
        row["MODEL_MEAN_PNL"] = m["MODEL_PNL"].mean()
        row["DELTA_MEAN_PNL"] = m["DELTA_PNL"].mean()
        paired_rows.append(row)
paired_df = pd.DataFrame(paired_rows).sort_values(["ANALYSIS", "SPLIT", "MODEL_LABEL"])

# Compact thesis sheets.
final_tables = {}
for analysis in bootstrap_df["ANALYSIS"].dropna().unique():
    d = bootstrap_df[bootstrap_df["ANALYSIS"].eq(analysis)].copy()
    if ONLY_SPLIT is not None:
        d = d[d["SPLIT"].eq(ONLY_SPLIT)].copy()
    cols = [
        "MODEL_LABEL", "SPLIT", "MEAN_PNL", "MEAN_CI_LOW", "MEAN_CI_HIGH",
        "CVAR_95", "CVAR_CI_LOW", "CVAR_CI_HIGH", "STD_PNL", "SHARPE_LIKE",
        "MEAN_TOTAL_TC", "MEAN_TOTAL_TURNOVER", "MEAN_AVG_HEDGE",
        "MEAN_AVG_ADJUSTMENT_FROM_DELTA", "MEAN_NO_TRADE_RATE", "MEAN_AVG_ADAPTIVE_BAND",
        "DELTA_RISK_PENALTY", "VARIANT", "EXPERIMENT",
    ]
    cols = [c for c in cols if c in d.columns]
    final_tables[analysis] = d[cols].sort_values("MEAN_PNL", ascending=False)

# Write output.
with pd.ExcelWriter(OUT_XLSX, engine="xlsxwriter") as writer:
    workbook = writer.book
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#D9EAF7", "border": 1})

    def write(df: pd.DataFrame, sheet: str):
        df.to_excel(writer, sheet_name=sheet[:31], index=False)
        ws = writer.sheets[sheet[:31]]
        ws.freeze_panes(1, 0)
        ws.set_row(0, None, header_fmt)
        for i, col in enumerate(df.columns):
            max_len = max([len(str(col))] + [len(str(x)) for x in df[col].head(100).fillna("")])
            ws.set_column(i, i, min(max(max_len + 2, 10), 36))

    index = pd.DataFrame([
        {"SHEET": "Bootstrap_All", "DESCRIPTION": "Bootstrap 95% confidence intervals for Mean PnL and CVaR95."},
        {"SHEET": "Paired_vs_Delta", "DESCRIPTION": "Episode-level paired differences versus Delta hedge."},
        {"SHEET": "Final_*", "DESCRIPTION": "Compact test-set thesis tables by analysis section."},
        {"SHEET": "Episode_Level_Cache", "DESCRIPTION": "Episode-level averaged data used for inference."},
    ])
    write(index, "Index")
    write(bootstrap_df, "Bootstrap_All")
    write(paired_df, "Paired_vs_Delta")
    for analysis, df in final_tables.items():
        sheet = "Final_" + analysis.replace("_", " ").title().replace(" ", "_")
        write(df, sheet)
    write(episodes.head(20000), "Episode_Level_Cache")

print("Saved:", OUT_XLSX)
print("Cache directory:", CACHE_DIR)
