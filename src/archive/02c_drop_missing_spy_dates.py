import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

SPY_PATH = RAW_DIR / "SPY_daily_raw_2010_2023.csv"

EPISODES_PATH = PROCESSED_DIR / "episodes_daily_top1_final_2010_2023.parquet"
TRANSITIONS_PATH = PROCESSED_DIR / "transitions_daily_top1_final_2010_2023.parquet"

OUTPUT_EPISODES_PATH = PROCESSED_DIR / "episodes_daily_top1_final_with_spy_2010_2023.parquet"
OUTPUT_TRANSITIONS_PATH = PROCESSED_DIR / "transitions_daily_top1_final_with_spy_2010_2023.parquet"

episodes = pd.read_parquet(EPISODES_PATH)
transitions = pd.read_parquet(TRANSITIONS_PATH)

spy = pd.read_csv(SPY_PATH)
spy.columns = [str(c).strip().replace(" ", "_").upper() for c in spy.columns]

spy = spy[["DATE", "CLOSE"]].copy()
spy = spy.rename(columns={"DATE": "QUOTE_DATE", "CLOSE": "SPY_CLOSE"})
spy["QUOTE_DATE"] = pd.to_datetime(spy["QUOTE_DATE"], errors="coerce")
spy["SPY_CLOSE"] = pd.to_numeric(spy["SPY_CLOSE"], errors="coerce")
spy = spy.dropna(subset=["QUOTE_DATE", "SPY_CLOSE"])
spy = spy.sort_values("QUOTE_DATE").drop_duplicates("QUOTE_DATE")

transitions["QUOTE_DATE"] = pd.to_datetime(transitions["QUOTE_DATE"])
transitions["NEXT_QUOTE_DATE"] = pd.to_datetime(transitions["NEXT_QUOTE_DATE"])

spy_dates = set(spy["QUOTE_DATE"])

bad_transition_mask = (
    ~transitions["QUOTE_DATE"].isin(spy_dates)
    | ~transitions["NEXT_QUOTE_DATE"].isin(spy_dates)
)

bad_transitions = transitions[bad_transition_mask].copy()
bad_episode_ids = set(bad_transitions["EPISODE_ID"].unique())

print("===== MISSING SPY DATE IMPACT =====")
print("Bad transitions:", len(bad_transitions))
print("Bad episodes:", len(bad_episode_ids))

print("\nBad transitions by split:")
print(bad_transitions["SPLIT"].value_counts())

print("\nMissing current quote dates:")
print(
    sorted(
        transitions.loc[~transitions["QUOTE_DATE"].isin(spy_dates), "QUOTE_DATE"]
        .dropna()
        .unique()
    )
)

print("\nMissing next quote dates:")
print(
    sorted(
        transitions.loc[~transitions["NEXT_QUOTE_DATE"].isin(spy_dates), "NEXT_QUOTE_DATE"]
        .dropna()
        .unique()
    )
)

print("\nSample bad transitions:")
cols = [
    "EPISODE_ID", "SPLIT",
    "QUOTE_DATE", "NEXT_QUOTE_DATE",
    "DTE", "NEXT_DTE",
    "UNDERLYING_LAST", "NEXT_UNDERLYING_LAST",
    "OPTION_MID", "NEXT_OPTION_MID"
]
cols = [c for c in cols if c in bad_transitions.columns]
print(bad_transitions[cols].head(50))

# Drop whole episodes affected by missing SPY trading dates
episodes_clean = episodes[~episodes["EPISODE_ID"].isin(bad_episode_ids)].copy()
transitions_clean = transitions[~transitions["EPISODE_ID"].isin(bad_episode_ids)].copy()

print("\n===== BEFORE DROPPING =====")
print("Episodes:", episodes["EPISODE_ID"].nunique())
print("Transitions:", len(transitions))
print(transitions["SPLIT"].value_counts())

print("\n===== AFTER DROPPING =====")
print("Episodes:", episodes_clean["EPISODE_ID"].nunique())
print("Transitions:", len(transitions_clean))
print(transitions_clean["SPLIT"].value_counts())

# Merge SPY_t
transitions_clean = transitions_clean.merge(
    spy.rename(columns={"SPY_CLOSE": "SPY_CLOSE"}),
    on="QUOTE_DATE",
    how="left"
)

# Merge SPY_{t+1}
spy_next = spy.rename(columns={
    "QUOTE_DATE": "NEXT_QUOTE_DATE",
    "SPY_CLOSE": "SPY_NEXT_CLOSE"
})

transitions_clean = transitions_clean.merge(
    spy_next,
    on="NEXT_QUOTE_DATE",
    how="left"
)

transitions_clean["SPY_DS"] = (
    transitions_clean["SPY_NEXT_CLOSE"] - transitions_clean["SPY_CLOSE"]
)

# Add SPY moneyness
transitions_clean["SPY_MONEYNESS"] = (
    transitions_clean["SPY_CLOSE"] / transitions_clean["STRIKE"]
)

import numpy as np
transitions_clean["SPY_LOG_MONEYNESS"] = np.log(
    transitions_clean["SPY_CLOSE"] / transitions_clean["STRIKE"]
)

# Check missing after merge
print("\nMissing SPY_CLOSE after merge:", transitions_clean["SPY_CLOSE"].isna().sum())
print("Missing SPY_NEXT_CLOSE after merge:", transitions_clean["SPY_NEXT_CLOSE"].isna().sum())

# Compare external SPY with option underlying quote
transitions_clean["SPY_MINUS_UNDERLYING_LAST"] = (
    transitions_clean["SPY_CLOSE"] - transitions_clean["UNDERLYING_LAST"]
)

transitions_clean["SPY_OVER_UNDERLYING_LAST"] = (
    transitions_clean["SPY_CLOSE"] / transitions_clean["UNDERLYING_LAST"]
)

print("\n===== SPY_CLOSE VS UNDERLYING_LAST AFTER CLEANING =====")
print("Difference:")
print(transitions_clean["SPY_MINUS_UNDERLYING_LAST"].describe())

print("\nRatio:")
print(transitions_clean["SPY_OVER_UNDERLYING_LAST"].describe())

# Merge SPY into episodes
episodes_clean["QUOTE_DATE"] = pd.to_datetime(episodes_clean["QUOTE_DATE"])

episodes_clean = episodes_clean.merge(
    spy,
    on="QUOTE_DATE",
    how="left"
)

episodes_clean["SPY_MONEYNESS"] = episodes_clean["SPY_CLOSE"] / episodes_clean["STRIKE"]
episodes_clean["SPY_LOG_MONEYNESS"] = np.log(
    episodes_clean["SPY_CLOSE"] / episodes_clean["STRIKE"]
)

print("\nMissing episode SPY_CLOSE:", episodes_clean["SPY_CLOSE"].isna().sum())

if transitions_clean["SPY_CLOSE"].isna().sum() > 0:
    raise ValueError("Still missing SPY_CLOSE after cleaning.")

if transitions_clean["SPY_NEXT_CLOSE"].isna().sum() > 0:
    raise ValueError("Still missing SPY_NEXT_CLOSE after cleaning.")

if episodes_clean["SPY_CLOSE"].isna().sum() > 0:
    raise ValueError("Still missing episode SPY_CLOSE after cleaning.")

episodes_clean.to_parquet(OUTPUT_EPISODES_PATH, index=False)
transitions_clean.to_parquet(OUTPUT_TRANSITIONS_PATH, index=False)

print("\nSaved episodes with SPY:")
print(OUTPUT_EPISODES_PATH)

print("\nSaved transitions with SPY:")
print(OUTPUT_TRANSITIONS_PATH)