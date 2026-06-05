import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"

episodes_path = DATA_DIR / "episodes_daily_top1_final_2010_2023.parquet"
transitions_path = DATA_DIR / "transitions_daily_top1_final_2010_2023.parquet"

episodes = pd.read_parquet(episodes_path)
transitions = pd.read_parquet(transitions_path)

print("Episodes shape:", episodes.shape)
print("Transitions shape:", transitions.shape)

print("\nTransitions by split:")
print(transitions["SPLIT"].value_counts())

print("\nEpisodes by split:")
print(episodes.groupby("SPLIT")["EPISODE_ID"].nunique())

print("\nColumns:")
print(transitions.columns.tolist())