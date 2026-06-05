"""
09_make_dataset_summary_latex.py

Create a thesis-ready Dataset Summary table for Chapter 4.

What this script does
---------------------
1. Loads the final processed transition-level parquet file.
2. Detects the split column, episode column, date column, option identifier columns if available.
3. Computes summary statistics by train/validation/test split:
   - number of daily transitions
   - number of option episodes
   - date range
   - average episode length
   - median episode length
   - number of unique option contracts, if identifiable
4. Saves:
   - outputs/dataset_summary_by_split.csv
   - outputs/dataset_summary_by_split.xlsx
   - tables/chapter4_dataset_summary.tex

How to run
----------
Place this file in:
    D:\SPY_Option_Hedging_Thesis\src\

Then run from project root:
    py src\09_make_dataset_summary_latex.py

If your final transition file has a different name, change DATA_PATH below.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


# ============================================================
# 1. Paths
# ============================================================

PROJECT_DIR = Path(r"D:\SPY_Option_Hedging_Thesis")

# Main final transition file used by your thesis.
# Change this path if your final file is different.
DATA_PATH = PROJECT_DIR / "data" / "processed" / "transitions_daily_top1_final_with_spy_2010_2023.parquet"

# Alternative if you want to summarize the regime-proxy version instead:
# DATA_PATH = PROJECT_DIR / "data" / "processed" / "transitions_daily_top1_final_with_spy_2010_2023_with_regime_proxies.parquet"

OUTPUT_DIR = PROJECT_DIR / "outputs"
TABLE_DIR = PROJECT_DIR / "tables"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Helper functions
# ============================================================

def find_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    """Find a column by trying exact, lowercase, and uppercase matches."""
    cols = list(df.columns)
    col_map_lower = {c.lower(): c for c in cols}
    col_map_upper = {c.upper(): c for c in cols}

    for cand in candidates:
        if cand in cols:
            return cand
        if cand.lower() in col_map_lower:
            return col_map_lower[cand.lower()]
        if cand.upper() in col_map_upper:
            return col_map_upper[cand.upper()]

    if required:
        raise ValueError(
            f"Could not find any of these columns: {candidates}\n"
            f"Available columns are:\n{cols}"
        )
    return None


def latex_escape(s: object) -> str:
    """Escape a few characters that commonly break LaTeX tables."""
    if pd.isna(s):
        return ""
    text = str(s)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
        "$": r"\$",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def fmt_int(x: object) -> str:
    if pd.isna(x):
        return ""
    return f"{int(x):,}"


def fmt_float(x: object, ndigits: int = 2) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x):,.{ndigits}f}"


def normalize_split_name(x: object) -> str:
    s = str(x).strip().lower()
    if s in {"train", "training"}:
        return "Train"
    if s in {"val", "valid", "validation"}:
        return "Validation"
    if s in {"test", "testing"}:
        return "Test"
    return str(x)


def split_order_key(x: str) -> int:
    order = {"Train": 0, "Validation": 1, "Test": 2}
    return order.get(str(x), 99)


# ============================================================
# 3. Load data
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Cannot find DATA_PATH:\n{DATA_PATH}\n\n"
        "Edit DATA_PATH in this script to point to your final processed parquet file."
    )

df = pd.read_parquet(DATA_PATH)

print(f"Loaded: {DATA_PATH}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")


# ============================================================
# 4. Detect important columns
# ============================================================

split_col = find_col(
    df,
    ["SPLIT", "split", "DATA_SPLIT", "dataset_split", "SET", "set"],
    required=True,
)

episode_col = find_col(
    df,
    [
        "EPISODE_ID", "episode_id", "EPISODE", "episode",
        "OPTION_EPISODE_ID", "option_episode_id",
        "CONTRACT_ID", "contract_id",
        "OPTION_ID", "option_id",
        "option_key", "OPTION_KEY",
    ],
    required=True,
)

date_col = find_col(
    df,
    [
        "QUOTE_DATE", "quote_date", "DATE", "date",
        "TRADE_DATE", "trade_date",
        "t", "timestamp",
    ],
    required=True,
)

# Optional option identifier for unique contract count.
# If not available, we use episode_col as a proxy.
option_id_col = find_col(
    df,
    [
        "OPTION_ID", "option_id",
        "CONTRACT_ID", "contract_id",
        "OPTION_KEY", "option_key",
        "TICKER_OPTION_ID", "ticker_option_id",
        "ROOT_SYMBOL", "root_symbol",
        "SYMBOL", "symbol",
    ],
    required=False,
)

if option_id_col is None:
    option_id_col = episode_col

print("\nDetected columns:")
print(f"  split_col    = {split_col}")
print(f"  episode_col  = {episode_col}")
print(f"  date_col     = {date_col}")
print(f"  option_id_col= {option_id_col}")


# ============================================================
# 5. Clean fields
# ============================================================

work = df.copy()

work["_SPLIT_NORM"] = work[split_col].map(normalize_split_name)
work["_DATE"] = pd.to_datetime(work[date_col], errors="coerce")

# Drop rows without split/date/episode only for summary computation.
summary_base = work.dropna(subset=["_SPLIT_NORM", "_DATE", episode_col]).copy()


# ============================================================
# 6. Compute split summary
# ============================================================

# Episode lengths in number of transitions.
episode_lengths = (
    summary_base
    .groupby(["_SPLIT_NORM", episode_col], dropna=False)
    .size()
    .reset_index(name="EPISODE_LENGTH")
)

episode_stats = (
    episode_lengths
    .groupby("_SPLIT_NORM")
    .agg(
        N_EPISODES=(episode_col, "nunique"),
        AVG_EPISODE_LENGTH=("EPISODE_LENGTH", "mean"),
        MEDIAN_EPISODE_LENGTH=("EPISODE_LENGTH", "median"),
        MIN_EPISODE_LENGTH=("EPISODE_LENGTH", "min"),
        MAX_EPISODE_LENGTH=("EPISODE_LENGTH", "max"),
    )
    .reset_index()
)

split_summary = (
    summary_base
    .groupby("_SPLIT_NORM")
    .agg(
        N_TRANSITIONS=(episode_col, "size"),
        N_UNIQUE_OPTIONS=(option_id_col, "nunique"),
        START_DATE=("_DATE", "min"),
        END_DATE=("_DATE", "max"),
    )
    .reset_index()
)

summary = split_summary.merge(episode_stats, on="_SPLIT_NORM", how="left")
summary = summary.sort_values("_SPLIT_NORM", key=lambda s: s.map(split_order_key))

# Add total row
total_episode_lengths = (
    summary_base
    .groupby(episode_col, dropna=False)
    .size()
    .reset_index(name="EPISODE_LENGTH")
)

total_row = pd.DataFrame([{
    "_SPLIT_NORM": "Total",
    "N_TRANSITIONS": len(summary_base),
    "N_UNIQUE_OPTIONS": summary_base[option_id_col].nunique(),
    "START_DATE": summary_base["_DATE"].min(),
    "END_DATE": summary_base["_DATE"].max(),
    "N_EPISODES": summary_base[episode_col].nunique(),
    "AVG_EPISODE_LENGTH": total_episode_lengths["EPISODE_LENGTH"].mean(),
    "MEDIAN_EPISODE_LENGTH": total_episode_lengths["EPISODE_LENGTH"].median(),
    "MIN_EPISODE_LENGTH": total_episode_lengths["EPISODE_LENGTH"].min(),
    "MAX_EPISODE_LENGTH": total_episode_lengths["EPISODE_LENGTH"].max(),
}])

summary = pd.concat([summary, total_row], ignore_index=True)

# Format dates for CSV/Excel readability
summary_export = summary.copy()
for col in ["START_DATE", "END_DATE"]:
    summary_export[col] = pd.to_datetime(summary_export[col]).dt.strftime("%Y-%m-%d")

csv_path = OUTPUT_DIR / "dataset_summary_by_split.csv"
xlsx_path = OUTPUT_DIR / "dataset_summary_by_split.xlsx"

summary_export.to_csv(csv_path, index=False)
summary_export.to_excel(xlsx_path, index=False)

print(f"\nSaved CSV:  {csv_path}")
print(f"Saved XLSX: {xlsx_path}")


# ============================================================
# 7. Create LaTeX table
# ============================================================

latex_rows = []
for _, row in summary_export.iterrows():
    latex_rows.append(
        " & ".join([
            latex_escape(row["_SPLIT_NORM"]),
            fmt_int(row["N_TRANSITIONS"]),
            fmt_int(row["N_EPISODES"]),
            fmt_int(row["N_UNIQUE_OPTIONS"]),
            latex_escape(row["START_DATE"]),
            latex_escape(row["END_DATE"]),
            fmt_float(row["AVG_EPISODE_LENGTH"], 2),
            fmt_float(row["MEDIAN_EPISODE_LENGTH"], 1),
        ]) + r" \\"
    )

latex_table = r"""
% ============================================================
% Dataset summary table
% Generated by src/09_make_dataset_summary_latex.py
% ============================================================

\subsection{Dataset Summary}
\label{sec:dataset-summary-table}

Table~\ref{tab:ch4-dataset-summary} reports the final transition-level dataset used in the empirical experiments. 
The table summarizes the number of daily hedging transitions, option episodes, unique option identifiers and chronological date range for each split. 
Episode length is measured as the number of daily hedging transitions in an option episode.

\begin{table}[H]
\centering
\scriptsize
\begin{threeparttable}
\caption{Dataset summary by split.}
\label{tab:ch4-dataset-summary}
\begin{tabularx}{\textwidth}{L{2.0cm}rrrrL{1.9cm}L{1.9cm}rr}
\toprule
Split & Transitions & Episodes & Unique options & Start date & End date & Avg. length & Median length \\
\midrule
""" + "\n".join(latex_rows) + r"""
\bottomrule
\end{tabularx}
\begin{tablenotes}
\footnotesize
\item The table is computed from the final processed transition file. 
A transition corresponds to one daily hedging step from \(t\) to \(t+1\). 
The unique-option count uses the available option identifier column; if no separate option identifier is available, the episode identifier is used as a proxy.
\end{tablenotes}
\end{threeparttable}
\end{table}
""".strip()

tex_path = TABLE_DIR / "chapter4_dataset_summary.tex"
tex_path.write_text(latex_table + "\n", encoding="utf-8")

print(f"Saved LaTeX table: {tex_path}")

print("\nDataset summary:")
print(summary_export.to_string(index=False))
