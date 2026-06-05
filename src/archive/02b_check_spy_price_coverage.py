import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

SPY_PATH = RAW_DIR / "SPY_daily_raw_2010_2023.csv"
TRANSITIONS_PATH = PROCESSED_DIR / "transitions_daily_top1_final_2010_2023.parquet"

spy = pd.read_csv(SPY_PATH)
transitions = pd.read_parquet(TRANSITIONS_PATH)

spy.columns = [str(c).strip().replace(" ", "_").upper() for c in spy.columns]

print("SPY columns:")
print(spy.columns.tolist())

date_col = "DATE"
price_col = "CLOSE"

if date_col not in spy.columns:
    raise ValueError("Không tìm thấy cột DATE trong SPY file.")

if price_col not in spy.columns:
    raise ValueError("Không tìm thấy cột CLOSE trong SPY file.")

spy = spy[[date_col, price_col]].copy()
spy = spy.rename(columns={date_col: "QUOTE_DATE", price_col: "SPY_CLOSE"})

spy["QUOTE_DATE"] = pd.to_datetime(spy["QUOTE_DATE"], errors="coerce")
spy["SPY_CLOSE"] = pd.to_numeric(spy["SPY_CLOSE"], errors="coerce")

spy = spy.dropna(subset=["QUOTE_DATE", "SPY_CLOSE"])
spy = spy.sort_values("QUOTE_DATE").drop_duplicates("QUOTE_DATE")

print("\n===== SPY FILE SUMMARY =====")
print("Rows:", len(spy))
print("Date range:", spy["QUOTE_DATE"].min(), "->", spy["QUOTE_DATE"].max())
print("Duplicate dates:", spy["QUOTE_DATE"].duplicated().sum())

print("\nRows by year:")
print(spy["QUOTE_DATE"].dt.year.value_counts().sort_index())

print("\nFirst rows:")
print(spy.head())

print("\n===== BASIC PRICE SCALE CHECK =====")
check_date = pd.Timestamp("2010-01-04")
row = spy[spy["QUOTE_DATE"] == check_date]

if not row.empty:
    price = row["SPY_CLOSE"].iloc[0]
    print(f"SPY close on {check_date.date()}: {price}")

    if price < 100:
        print("WARNING: Price looks adjusted. For option hedging, use raw unadjusted Close.")
    else:
        print("Price scale looks closer to raw SPY close.")
else:
    print("2010-01-04 not found in SPY file.")

print("\n===== COVERAGE AGAINST TRANSITIONS =====")

transitions["QUOTE_DATE"] = pd.to_datetime(transitions["QUOTE_DATE"])
transitions["NEXT_QUOTE_DATE"] = pd.to_datetime(transitions["NEXT_QUOTE_DATE"])

needed_dates = pd.Index(
    pd.concat([
        transitions["QUOTE_DATE"],
        transitions["NEXT_QUOTE_DATE"]
    ]).dropna().unique()
)

spy_dates = pd.Index(spy["QUOTE_DATE"].unique())

missing_dates = needed_dates.difference(spy_dates)

print("Needed transition dates:", len(needed_dates))
print("SPY available dates:", len(spy_dates))
print("Missing dates needed by transitions:", len(missing_dates))

if len(missing_dates) > 0:
    print("\nMissing dates:")
    print(pd.Series(missing_dates).sort_values().head(100).to_string(index=False))
else:
    print("Coverage check passed: SPY file has all transition dates.")

print("\n===== COMPARE SPY_CLOSE VS OPTION UNDERLYING_LAST =====")

tmp = transitions.merge(
    spy,
    on="QUOTE_DATE",
    how="left"
)

tmp["SPY_MINUS_UNDERLYING_LAST"] = tmp["SPY_CLOSE"] - tmp["UNDERLYING_LAST"]
tmp["SPY_OVER_UNDERLYING_LAST"] = tmp["SPY_CLOSE"] / tmp["UNDERLYING_LAST"]

print("\nDifference summary:")
print(tmp["SPY_MINUS_UNDERLYING_LAST"].describe())

print("\nRatio summary:")
print(tmp["SPY_OVER_UNDERLYING_LAST"].describe())

print("\nSample comparison:")
print(
    tmp[
        [
            "QUOTE_DATE",
            "SPY_CLOSE",
            "UNDERLYING_LAST",
            "SPY_MINUS_UNDERLYING_LAST",
            "SPY_OVER_UNDERLYING_LAST"
        ]
    ].head(20)
)