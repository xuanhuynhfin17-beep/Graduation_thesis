"""
05e_ablation_ppo_v3c.py

Ablation study for the residual-delta V3C hedging environment.

Purpose:
    Show which components of the V3C design matter:
        1. Direct PPO hedge learning
        2. Residual-delta action
        3. No-trade band
        4. Downside loss penalty
        5. Delta-risk penalty

Main experiment:
    Algorithm: PPO only
    Seeds: [1, 2, 3]
    Timesteps: 100,000 per seed per variant

Output:
    outputs/ablation_ppo_v3c_100k_3seeds.xlsx
    outputs/ablation_ppo_v3c_100k_3seeds_step_results.parquet

Run:
    py src\\05e_ablation_ppo_v3c.py

If runtime is too long, quick test:
    SEEDS = [1]
    TOTAL_TIMESTEPS = 10_000
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import gymnasium as gym
import numpy as np
import pandas as pd
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


# ============================================================
# PATH HELPERS
# ============================================================

TRANSITIONS_FILE = "transitions_daily_top1_final_with_spy_2010_2023.parquet"


def _candidate_project_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    candidates = [
        here.parent,
        here.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
        Path("/mnt/data"),
    ]

    out: list[Path] = []
    for p in candidates:
        p = p.resolve()
        if p not in out:
            out.append(p)
    return out


def find_existing_file(filename: str, relative_dirs: Iterable[str]) -> Path:
    checked: list[Path] = []
    for base in _candidate_project_dirs():
        for rel in relative_dirs:
            p = base / rel / filename if rel else base / filename
            checked.append(p)
            if p.exists():
                return p

    checked_msg = "\n".join(str(p) for p in checked)
    raise FileNotFoundError(f"Could not find {filename}. Checked:\n{checked_msg}")


TRANSITIONS_PATH = find_existing_file(
    TRANSITIONS_FILE,
    relative_dirs=["data/processed", "processed", ""],
)

if (Path(__file__).resolve().parent.parent / "data" / "processed").exists():
    PROJECT_DIR = Path(__file__).resolve().parent.parent
elif (Path.cwd() / "data" / "processed").exists():
    PROJECT_DIR = Path.cwd()
else:
    PROJECT_DIR = Path("/mnt/data")

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

CONTRACT_MULTIPLIER = 100

LINEAR_TRANSACTION_COST_RATE = 0.0005
QUADRATIC_IMPACT_RATE = 0.0

HEDGE_MIN = 0.0
HEDGE_MAX = 1.0

DEFAULT_ADJUSTMENT_LIMIT = 0.10
DEFAULT_NO_TRADE_BAND = 0.02

REWARD_SCALE_MODE = "option_mid"
MIN_REWARD_SCALE = 1.0

TOTAL_TIMESTEPS = 100_000
SEEDS = [1, 2, 3]

MODEL_DIR = OUTPUT_DIR / "ablation_ppo_v3c_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_OUTPUT_PATH = OUTPUT_DIR / "ablation_ppo_v3c_with_direct_no_trade_100k_3seeds.xlsx"
STEP_RESULTS_PATH = OUTPUT_DIR / "ablation_ppo_v3c_with_direct_no_trade_100k_3seeds_step_results.parquet"

PPO_POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[256, 256],
        vf=[256, 256],
    ),
    activation_fn=th.nn.Tanh,
)

BASE_FEATURE_COLS = [
    "DTE",
    "SPY_LOG_MONEYNESS",
    "SPY_MONEYNESS",
    "OPTION_MID_OVER_SPY",
    "SPY_RET_LAG1",
    "OPTION_DELTA",
    "OPTION_GAMMA",
    "OPTION_VEGA",
    "OPTION_THETA",
    "OPTION_IV",
    "OPTION_SPREAD_PCT",
]

# Ablation variants.
# The full V3C row should reproduce the main PPO V3C environment.
ABLATION_VARIANTS = [
    {
        "VARIANT": "direct_ppo",
        "DESCRIPTION": "Direct PPO learns full hedge ratio from scratch.",
        "RESIDUAL_DELTA": False,
        "ADJUSTMENT_LIMIT": None,
        "NO_TRADE_BAND": 0.00,
        "DOWNSIDE_PENALTY": 0.00,
        "DELTA_RISK_PENALTY": 0.00,
        "EXTRA_COST_PENALTY": 0.00,
    },
    {
        "VARIANT": "direct_ppo_no_trade",
        "DESCRIPTION": "Direct PPO learns full hedge ratio with fixed no-trade band.",
        "RESIDUAL_DELTA": False,
        "ADJUSTMENT_LIMIT": None,
        "NO_TRADE_BAND": DEFAULT_NO_TRADE_BAND,
        "DOWNSIDE_PENALTY": 0.00,
        "DELTA_RISK_PENALTY": 0.00,
        "EXTRA_COST_PENALTY": 0.00,
    },
    {
        "VARIANT": "residual_only",
        "DESCRIPTION": "Residual-delta action only, no no-trade band and no risk penalty.",
        "RESIDUAL_DELTA": True,
        "ADJUSTMENT_LIMIT": DEFAULT_ADJUSTMENT_LIMIT,
        "NO_TRADE_BAND": 0.00,
        "DOWNSIDE_PENALTY": 0.00,
        "DELTA_RISK_PENALTY": 0.00,
        "EXTRA_COST_PENALTY": 0.00,
    },
    {
        "VARIANT": "residual_no_trade",
        "DESCRIPTION": "Residual-delta action plus fixed no-trade band.",
        "RESIDUAL_DELTA": True,
        "ADJUSTMENT_LIMIT": DEFAULT_ADJUSTMENT_LIMIT,
        "NO_TRADE_BAND": DEFAULT_NO_TRADE_BAND,
        "DOWNSIDE_PENALTY": 0.00,
        "DELTA_RISK_PENALTY": 0.00,
        "EXTRA_COST_PENALTY": 0.00,
    },
    {
        "VARIANT": "residual_no_trade_downside",
        "DESCRIPTION": "Residual-delta plus no-trade plus downside loss penalty.",
        "RESIDUAL_DELTA": True,
        "ADJUSTMENT_LIMIT": DEFAULT_ADJUSTMENT_LIMIT,
        "NO_TRADE_BAND": DEFAULT_NO_TRADE_BAND,
        "DOWNSIDE_PENALTY": 0.50,
        "DELTA_RISK_PENALTY": 0.00,
        "EXTRA_COST_PENALTY": 0.00,
    },
    {
        "VARIANT": "full_v3c",
        "DESCRIPTION": "Full V3C: residual-delta plus no-trade, downside penalty and delta-risk penalty.",
        "RESIDUAL_DELTA": True,
        "ADJUSTMENT_LIMIT": DEFAULT_ADJUSTMENT_LIMIT,
        "NO_TRADE_BAND": DEFAULT_NO_TRADE_BAND,
        "DOWNSIDE_PENALTY": 0.50,
        "DELTA_RISK_PENALTY": 0.50,
        "EXTRA_COST_PENALTY": 0.00,
    },
]


# ============================================================
# DATA LOADING AND FEATURE ENGINEERING
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)

    if "OPTION_MID" in df.columns and "SPY_CLOSE" in df.columns:
        df["OPTION_MID_OVER_SPY"] = (
            df["OPTION_MID"].astype(float) / df["SPY_CLOSE"].astype(float)
        )

    if "SPY_CLOSE" in df.columns:
        df["SPY_RET_LAG1"] = (
            df.groupby("EPISODE_ID")["SPY_CLOSE"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    if "SPY_MONEYNESS" not in df.columns and {"SPY_CLOSE", "STRIKE"}.issubset(df.columns):
        df["SPY_MONEYNESS"] = df["SPY_CLOSE"].astype(float) / df["STRIKE"].astype(float)

    if "SPY_LOG_MONEYNESS" not in df.columns and {"SPY_CLOSE", "STRIKE"}.issubset(df.columns):
        df["SPY_LOG_MONEYNESS"] = np.log(
            df["SPY_CLOSE"].astype(float) / df["STRIKE"].astype(float)
        )

    for c in BASE_FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return df


transitions = add_engineered_features(pd.read_parquet(TRANSITIONS_PATH))

FEATURE_COLS = [c for c in BASE_FEATURE_COLS if c in transitions.columns]

train_transitions = transitions[transitions["SPLIT"] == "train"].copy()
val_transitions = transitions[transitions["SPLIT"] == "val"].copy()
test_transitions = transitions[transitions["SPLIT"] == "test"].copy()

feature_mean = train_transitions[FEATURE_COLS].mean()
feature_std = train_transitions[FEATURE_COLS].std().replace(0, 1)

print("Loaded transitions:", transitions.shape)
print("Transitions path:", TRANSITIONS_PATH)
print("Transitions by split:")
print(transitions["SPLIT"].value_counts())
print("\nFeature columns:")
print(FEATURE_COLS)
print("\nAblation variants:")
print(pd.DataFrame(ABLATION_VARIANTS)[["VARIANT", "DESCRIPTION"]])
print("\nOutput directory:", OUTPUT_DIR)


# ============================================================
# ENVIRONMENT
# ============================================================

class OptionHedgingAblationEnv(gym.Env):
    """
    Gymnasium environment for short-call deep hedging.

    Two action mappings are supported:

    1. Direct hedge:
        raw_action in [-1, 1]
        desired_hedge = (raw_action + 1) / 2
        desired_hedge in [0, 1]

    2. Residual-delta hedge:
        raw_action in [-1, 1]
        desired_hedge = clip(delta + adjustment_limit * raw_action, 0, 1)

    Optional no-trade band:
        if abs(desired_hedge - prev_hedge) < no_trade_band:
            hedge = prev_hedge
        else:
            hedge = desired_hedge
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        transitions_df: pd.DataFrame,
        feature_cols: list[str],
        feature_mean: pd.Series,
        feature_std: pd.Series,
        residual_delta: bool,
        adjustment_limit: float | None,
        no_trade_band: float,
        downside_penalty: float,
        delta_risk_penalty: float,
        extra_cost_penalty: float,
        hedge_min: float = HEDGE_MIN,
        hedge_max: float = HEDGE_MAX,
        linear_transaction_cost_rate: float = LINEAR_TRANSACTION_COST_RATE,
        quadratic_impact_rate: float = QUADRATIC_IMPACT_RATE,
        reward_scale_mode: str = REWARD_SCALE_MODE,
        min_reward_scale: float = MIN_REWARD_SCALE,
        random_episode: bool = True,
        seed: int = 42,
    ):
        super().__init__()

        self.df = transitions_df.copy()
        self.df = self.df.sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)

        self.feature_cols = feature_cols
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        self.residual_delta = residual_delta
        self.adjustment_limit = adjustment_limit if adjustment_limit is not None else 0.0
        self.no_trade_band = no_trade_band
        self.downside_penalty = downside_penalty
        self.delta_risk_penalty = delta_risk_penalty
        self.extra_cost_penalty = extra_cost_penalty

        self.hedge_min = hedge_min
        self.hedge_max = hedge_max
        self.linear_transaction_cost_rate = linear_transaction_cost_rate
        self.quadratic_impact_rate = quadratic_impact_rate
        self.reward_scale_mode = reward_scale_mode
        self.min_reward_scale = min_reward_scale
        self.random_episode = random_episode

        self.episode_ids = self.df["EPISODE_ID"].unique().tolist()
        self.episode_data = {
            eid: group.reset_index(drop=True)
            for eid, group in self.df.groupby("EPISODE_ID")
        }

        self.rng = np.random.default_rng(seed)

        obs_dim = len(self.feature_cols) + 2  # normalized features + prev_hedge + delta_gap
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.current_episode_id = None
        self.current_data = None
        self.t = 0
        self.prev_hedge = 0.0

    def _get_delta(self, row: pd.Series) -> float:
        delta = float(row["OPTION_DELTA"])
        return float(np.clip(delta, self.hedge_min, self.hedge_max))

    def _scale_action_to_hedge(self, action: np.ndarray, row: pd.Series) -> tuple[float, float, float, float]:
        raw_action = float(np.clip(action[0], -1.0, 1.0))
        delta = self._get_delta(row)

        if self.residual_delta:
            desired_hedge = float(
                np.clip(
                    delta + self.adjustment_limit * raw_action,
                    self.hedge_min,
                    self.hedge_max,
                )
            )
        else:
            desired_hedge = float(
                np.clip(
                    self.hedge_min + (raw_action + 1.0) / 2.0 * (self.hedge_max - self.hedge_min),
                    self.hedge_min,
                    self.hedge_max,
                )
            )

        if self.no_trade_band > 0 and abs(desired_hedge - self.prev_hedge) < self.no_trade_band:
            executed_hedge = float(self.prev_hedge)
        else:
            executed_hedge = desired_hedge

        adjustment = executed_hedge - delta
        return raw_action, desired_hedge, executed_hedge, adjustment

    def _reward_scale(self, row: pd.Series) -> float:
        if self.reward_scale_mode == "option_mid" and "OPTION_MID" in row:
            return max(abs(float(row["OPTION_MID"])), self.min_reward_scale)
        if self.reward_scale_mode == "spy" and "SPY_CLOSE" in row:
            return max(abs(float(row["SPY_CLOSE"])), self.min_reward_scale)
        return 1.0

    def _dt_years(self, row: pd.Series) -> float:
        if "DTE" in row and "NEXT_DTE" in row:
            dte = float(row["DTE"])
            next_dte = float(row["NEXT_DTE"])
            if np.isfinite(dte) and np.isfinite(next_dte):
                return max(dte - next_dte, 1.0) / 252.0
        return 1.0 / 252.0

    def _iv_decimal(self, row: pd.Series) -> float:
        iv = float(row.get("OPTION_IV", 0.20))
        if not np.isfinite(iv) or iv <= 0:
            iv = 0.20
        if iv > 3.0:
            iv = iv / 100.0
        return iv

    def _transaction_cost(self, row: pd.Series, trade_size: float, price_col: str = "SPY_CLOSE") -> float:
        s = float(row[price_col])
        linear_cost = self.linear_transaction_cost_rate * s * abs(trade_size)
        impact_cost = self.quadratic_impact_rate * s * (trade_size ** 2)
        return float(linear_cost + impact_cost)

    def _get_obs(self) -> np.ndarray:
        row = self.current_data.iloc[self.t]

        x = row[self.feature_cols].astype(float)
        x = (x - self.feature_mean[self.feature_cols]) / self.feature_std[self.feature_cols]
        x = x.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)

        delta = self._get_delta(row)
        delta_gap = self.prev_hedge - delta

        obs = np.concatenate([
            x,
            np.array([self.prev_hedge, delta_gap], dtype=np.float32),
        ])

        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if self.random_episode:
            self.current_episode_id = self.rng.choice(self.episode_ids)
        else:
            self.current_episode_id = self.episode_ids[0]

        self.current_data = self.episode_data[self.current_episode_id]
        self.t = 0
        self.prev_hedge = 0.0

        return self._get_obs(), {}

    def step(self, action):
        row = self.current_data.iloc[self.t]

        raw_action, desired_hedge, target_hedge, adjustment = self._scale_action_to_hedge(action, row)
        delta = self._get_delta(row)

        d_stock = float(row["SPY_DS"])
        d_option = float(row["DOPTION"])

        trade_size = target_hedge - self.prev_hedge
        transaction_cost = self._transaction_cost(row, trade_size, price_col="SPY_CLOSE")

        is_last_step = self.t == len(self.current_data) - 1
        final_liquidation_cost = 0.0

        if is_last_step:
            final_liquidation_cost = self._transaction_cost(
                row,
                target_hedge,
                price_col="SPY_NEXT_CLOSE",
            )
            transaction_cost += final_liquidation_cost

        raw_pnl_per_share = (
            -d_option
            + target_hedge * d_stock
            - transaction_cost
        )

        scale = self._reward_scale(row)
        scaled_pnl = raw_pnl_per_share / scale
        scaled_tc = transaction_cost / scale

        downside = min(scaled_pnl, 0.0)

        iv = self._iv_decimal(row)
        dt = self._dt_years(row)
        s_t = float(row["SPY_CLOSE"])

        delta_risk = (((target_hedge - delta) * iv * s_t) ** 2) * dt / (scale ** 2)

        training_reward = (
            scaled_pnl
            - self.downside_penalty * (downside ** 2)
            - self.delta_risk_penalty * delta_risk
            - self.extra_cost_penalty * scaled_tc
        )

        self.prev_hedge = target_hedge
        self.t += 1

        terminated = self.t >= len(self.current_data)
        truncated = False

        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()

        turnover = abs(trade_size) + (abs(target_hedge) if is_last_step else 0.0)

        info = {
            "episode_id": self.current_episode_id,
            "raw_action": raw_action,
            "delta": delta,
            "desired_hedge": desired_hedge,
            "hedge": target_hedge,
            "adjustment": adjustment,
            "raw_pnl_per_share": raw_pnl_per_share,
            "scaled_pnl": scaled_pnl,
            "training_reward": training_reward,
            "transaction_cost_per_share": transaction_cost,
            "final_liquidation_cost_per_share": final_liquidation_cost,
            "delta_risk": delta_risk,
            "turnover": turnover,
            "is_last_step": is_last_step,
            "no_trade": abs(trade_size) < 1e-12,
        }

        return obs, float(training_reward), terminated, truncated, info


# ============================================================
# BASELINE
# ============================================================

def run_baseline_for_current_cost(transitions_df: pd.DataFrame, strategy: str = "delta") -> pd.DataFrame:
    df = transitions_df.copy()
    df = df.sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)

    if strategy == "no_hedge":
        df["HEDGE"] = 0.0
    elif strategy == "delta":
        df["HEDGE"] = df["OPTION_DELTA"].astype(float).clip(HEDGE_MIN, HEDGE_MAX)
    else:
        raise ValueError("strategy must be 'no_hedge' or 'delta'")

    df["PREV_HEDGE"] = df.groupby("EPISODE_ID")["HEDGE"].shift(1).fillna(0.0)
    df["TRADE_SIZE"] = df["HEDGE"] - df["PREV_HEDGE"]

    df["TC_REBALANCE_PER_SHARE"] = (
        LINEAR_TRANSACTION_COST_RATE * df["SPY_CLOSE"] * df["TRADE_SIZE"].abs()
        + QUADRATIC_IMPACT_RATE * df["SPY_CLOSE"] * (df["TRADE_SIZE"] ** 2)
    )

    df["OPTION_PNL_PER_SHARE"] = -df["DOPTION"]
    df["STOCK_PNL_PER_SHARE"] = df["HEDGE"] * df["SPY_DS"]

    df["IS_LAST_TRANSITION"] = (
        df.groupby("EPISODE_ID").cumcount()
        == df.groupby("EPISODE_ID")["EPISODE_ID"].transform("count") - 1
    )

    df["TC_FINAL_LIQ_PER_SHARE"] = 0.0
    last_mask = df["IS_LAST_TRANSITION"]

    df.loc[last_mask, "TC_FINAL_LIQ_PER_SHARE"] = (
        LINEAR_TRANSACTION_COST_RATE
        * df.loc[last_mask, "SPY_NEXT_CLOSE"]
        * df.loc[last_mask, "HEDGE"].abs()
        + QUADRATIC_IMPACT_RATE
        * df.loc[last_mask, "SPY_NEXT_CLOSE"]
        * (df.loc[last_mask, "HEDGE"] ** 2)
    )

    df["STEP_PNL_PER_SHARE"] = (
        df["OPTION_PNL_PER_SHARE"]
        + df["STOCK_PNL_PER_SHARE"]
        - df["TC_REBALANCE_PER_SHARE"]
        - df["TC_FINAL_LIQ_PER_SHARE"]
    )

    df["STEP_PNL"] = df["STEP_PNL_PER_SHARE"] * CONTRACT_MULTIPLIER
    df["TRANSACTION_COST"] = (
        df["TC_REBALANCE_PER_SHARE"] + df["TC_FINAL_LIQ_PER_SHARE"]
    ) * CONTRACT_MULTIPLIER

    df["TURNOVER"] = df["TRADE_SIZE"].abs()
    df.loc[last_mask, "TURNOVER"] += df.loc[last_mask, "HEDGE"].abs()

    episode_result = (
        df.groupby(["EPISODE_ID", "SPLIT"])
        .agg(
            START_DATE=("QUOTE_DATE", "first"),
            END_DATE=("NEXT_QUOTE_DATE", "last"),
            N_STEPS=("STEP_PNL", "count"),
            TERMINAL_PNL=("STEP_PNL", "sum"),
            TOTAL_TC=("TRANSACTION_COST", "sum"),
            TOTAL_TURNOVER=("TURNOVER", "sum"),
            AVG_HEDGE=("HEDGE", "mean"),
            STD_HEDGE=("HEDGE", "std"),
        )
        .reset_index()
    )

    episode_result["STRATEGY"] = strategy
    episode_result["VARIANT"] = strategy
    episode_result["ALGORITHM"] = "baseline"
    episode_result["SEED"] = np.nan
    episode_result["TRAINING_TIME_MIN"] = np.nan

    return episode_result


def make_baseline_results(transitions_df: pd.DataFrame) -> pd.DataFrame:
    nohedge_ep = run_baseline_for_current_cost(transitions_df, strategy="no_hedge")
    delta_ep = run_baseline_for_current_cost(transitions_df, strategy="delta")
    return pd.concat([nohedge_ep, delta_ep], ignore_index=True)


# ============================================================
# EVALUATION
# ============================================================

def make_obs_from_row(
    row: pd.Series,
    prev_hedge: float,
    feature_cols: list[str],
    feature_mean: pd.Series,
    feature_std: pd.Series,
) -> np.ndarray:
    x = row[feature_cols].astype(float)
    x = (x - feature_mean[feature_cols]) / feature_std[feature_cols]
    x = x.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)

    delta = float(np.clip(float(row["OPTION_DELTA"]), HEDGE_MIN, HEDGE_MAX))
    delta_gap = prev_hedge - delta

    obs = np.concatenate([
        x,
        np.array([prev_hedge, delta_gap], dtype=np.float32),
    ])

    return obs.astype(np.float32)


def compute_trade_cost(row: pd.Series, trade_size: float, use_next_price: bool = False) -> float:
    s = float(row["SPY_NEXT_CLOSE"] if use_next_price else row["SPY_CLOSE"])
    linear_cost = LINEAR_TRANSACTION_COST_RATE * s * abs(trade_size)
    impact_cost = QUADRATIC_IMPACT_RATE * s * (trade_size ** 2)
    return float(linear_cost + impact_cost)


def action_to_hedge_for_variant(action, row: pd.Series, prev_hedge: float, variant: dict) -> tuple[float, float, float, float]:
    raw_action = float(np.clip(action[0], -1.0, 1.0))
    delta = float(np.clip(float(row["OPTION_DELTA"]), HEDGE_MIN, HEDGE_MAX))

    if bool(variant["RESIDUAL_DELTA"]):
        adjustment_limit = float(variant["ADJUSTMENT_LIMIT"])
        desired_hedge = float(np.clip(delta + adjustment_limit * raw_action, HEDGE_MIN, HEDGE_MAX))
    else:
        desired_hedge = float(np.clip((raw_action + 1.0) / 2.0, HEDGE_MIN, HEDGE_MAX))

    no_trade_band = float(variant["NO_TRADE_BAND"])
    if no_trade_band > 0 and abs(desired_hedge - prev_hedge) < no_trade_band:
        target_hedge = prev_hedge
    else:
        target_hedge = desired_hedge

    adjustment = target_hedge - delta
    return raw_action, desired_hedge, target_hedge, adjustment


def evaluate_ppo_variant(
    model,
    transitions_df: pd.DataFrame,
    variant: dict,
    seed: int,
    training_time_min: float,
    deterministic: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variant_name = variant["VARIANT"]

    results = []
    step_rows = []

    df = transitions_df.copy()
    df = df.sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)

    for episode_id, ep in df.groupby("EPISODE_ID"):
        ep = ep.reset_index(drop=True)
        split = ep["SPLIT"].iloc[0]
        prev_hedge = 0.0

        rewards_per_share = []
        tcs_per_share = []
        hedges = []
        deltas = []
        raw_actions = []
        adjustments = []
        turnovers = []
        no_trade_flags = []

        for t in range(len(ep)):
            row = ep.iloc[t]

            obs = make_obs_from_row(
                row=row,
                prev_hedge=prev_hedge,
                feature_cols=FEATURE_COLS,
                feature_mean=feature_mean,
                feature_std=feature_std,
            )

            action, _ = model.predict(obs, deterministic=deterministic)
            raw_action, desired_hedge, target_hedge, adjustment = action_to_hedge_for_variant(
                action=action,
                row=row,
                prev_hedge=prev_hedge,
                variant=variant,
            )

            delta = float(np.clip(float(row["OPTION_DELTA"]), HEDGE_MIN, HEDGE_MAX))
            trade_size = target_hedge - prev_hedge
            no_trade = abs(trade_size) < 1e-12

            d_stock = float(row["SPY_DS"])
            d_option = float(row["DOPTION"])

            transaction_cost = compute_trade_cost(row, trade_size, use_next_price=False)

            is_last_step = t == len(ep) - 1
            if is_last_step:
                transaction_cost += compute_trade_cost(row, target_hedge, use_next_price=True)
                turnover = abs(trade_size) + abs(target_hedge)
            else:
                turnover = abs(trade_size)

            reward_per_share = (
                -d_option
                + target_hedge * d_stock
                - transaction_cost
            )

            rewards_per_share.append(reward_per_share)
            tcs_per_share.append(transaction_cost)
            hedges.append(target_hedge)
            deltas.append(delta)
            raw_actions.append(raw_action)
            adjustments.append(adjustment)
            turnovers.append(turnover)
            no_trade_flags.append(no_trade)

            step_rows.append({
                "VARIANT": variant_name,
                "SEED": seed,
                "ALGORITHM": "PPO",
                "EPISODE_ID": episode_id,
                "SPLIT": split,
                "QUOTE_DATE": row["QUOTE_DATE"],
                "STRATEGY": f"ppo_ablation_{variant_name}_seed_{seed}",
                "RAW_ACTION": raw_action,
                "DELTA": delta,
                "DESIRED_HEDGE": desired_hedge,
                "HEDGE": target_hedge,
                "PREV_HEDGE": prev_hedge,
                "ADJUSTMENT_FROM_DELTA": adjustment,
                "NO_TRADE": no_trade,
                "REWARD_PER_SHARE": reward_per_share,
                "REWARD": reward_per_share * CONTRACT_MULTIPLIER,
                "TRANSACTION_COST": transaction_cost * CONTRACT_MULTIPLIER,
                "TURNOVER": turnover,
            })

            prev_hedge = target_hedge

        results.append({
            "VARIANT": variant_name,
            "SEED": seed,
            "ALGORITHM": "PPO",
            "EPISODE_ID": episode_id,
            "SPLIT": split,
            "START_DATE": ep["QUOTE_DATE"].iloc[0],
            "END_DATE": ep["NEXT_QUOTE_DATE"].iloc[-1],
            "N_STEPS": len(ep),
            "TERMINAL_PNL": np.sum(rewards_per_share) * CONTRACT_MULTIPLIER,
            "TOTAL_TC": np.sum(tcs_per_share) * CONTRACT_MULTIPLIER,
            "TOTAL_TURNOVER": np.sum(turnovers),
            "AVG_RAW_ACTION": np.mean(raw_actions),
            "STD_RAW_ACTION": np.std(raw_actions),
            "AVG_DELTA": np.mean(deltas),
            "AVG_HEDGE": np.mean(hedges),
            "STD_HEDGE": np.std(hedges),
            "AVG_ADJUSTMENT_FROM_DELTA": np.mean(adjustments),
            "STD_ADJUSTMENT_FROM_DELTA": np.std(adjustments),
            "NO_TRADE_RATE": np.mean(no_trade_flags),
            "ACTION_NEAR_NEG1_RATE": np.mean(np.array(raw_actions) < -0.95),
            "ACTION_NEAR_POS1_RATE": np.mean(np.array(raw_actions) > 0.95),
            "STRATEGY": f"ppo_ablation_{variant_name}_seed_{seed}",
            "TRAINING_TIME_MIN": training_time_min,
        })

    return pd.DataFrame(step_rows), pd.DataFrame(results)


def cvar_95(x):
    x = pd.Series(x).dropna()
    if x.empty:
        return np.nan
    q = x.quantile(0.05)
    return x[x <= q].mean()


def sharpe_like(x):
    x = pd.Series(x).dropna()
    std = x.std()
    if std == 0 or pd.isna(std):
        return np.nan
    return x.mean() / std


def make_metrics(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    d = df.copy()

    defaults = {
        "TOTAL_TC": np.nan,
        "TOTAL_TURNOVER": np.nan,
        "AVG_HEDGE": np.nan,
        "AVG_DELTA": np.nan,
        "AVG_ADJUSTMENT_FROM_DELTA": np.nan,
        "NO_TRADE_RATE": np.nan,
        "ACTION_NEAR_NEG1_RATE": np.nan,
        "ACTION_NEAR_POS1_RATE": np.nan,
        "TRAINING_TIME_MIN": np.nan,
    }

    for c, v in defaults.items():
        if c not in d.columns:
            d[c] = v

    return (
        d.groupby(group_cols, dropna=False)
        .agg(
            EPISODES=("EPISODE_ID", "nunique"),
            MEAN_PNL=("TERMINAL_PNL", "mean"),
            STD_PNL=("TERMINAL_PNL", "std"),
            MEDIAN_PNL=("TERMINAL_PNL", "median"),
            MIN_PNL=("TERMINAL_PNL", "min"),
            MAX_PNL=("TERMINAL_PNL", "max"),
            CVAR_95=("TERMINAL_PNL", cvar_95),
            SHARPE_LIKE=("TERMINAL_PNL", sharpe_like),
            MEAN_TC=("TOTAL_TC", "mean"),
            MEAN_TURNOVER=("TOTAL_TURNOVER", "mean"),
            AVG_HEDGE=("AVG_HEDGE", "mean"),
            AVG_DELTA=("AVG_DELTA", "mean"),
            AVG_ADJUSTMENT_FROM_DELTA=("AVG_ADJUSTMENT_FROM_DELTA", "mean"),
            NO_TRADE_RATE=("NO_TRADE_RATE", "mean"),
            ACTION_NEAR_NEG1_RATE=("ACTION_NEAR_NEG1_RATE", "mean"),
            ACTION_NEAR_POS1_RATE=("ACTION_NEAR_POS1_RATE", "mean"),
            TRAINING_TIME_MIN=("TRAINING_TIME_MIN", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def make_ablation_summary(metrics_by_seed: pd.DataFrame) -> pd.DataFrame:
    rl = metrics_by_seed.dropna(subset=["SEED"]).copy()

    return (
        rl.groupby(["VARIANT", "SPLIT"], dropna=False)
        .agg(
            N_SEEDS=("SEED", "nunique"),
            MEAN_OF_MEAN_PNL=("MEAN_PNL", "mean"),
            STD_OF_MEAN_PNL=("MEAN_PNL", "std"),
            MEAN_OF_CVAR_95=("CVAR_95", "mean"),
            STD_OF_CVAR_95=("CVAR_95", "std"),
            MEAN_OF_SHARPE_LIKE=("SHARPE_LIKE", "mean"),
            STD_OF_SHARPE_LIKE=("SHARPE_LIKE", "std"),
            MEAN_TC=("MEAN_TC", "mean"),
            MEAN_TURNOVER=("MEAN_TURNOVER", "mean"),
            AVG_HEDGE=("AVG_HEDGE", "mean"),
            AVG_ADJUSTMENT_FROM_DELTA=("AVG_ADJUSTMENT_FROM_DELTA", "mean"),
            NO_TRADE_RATE=("NO_TRADE_RATE", "mean"),
            ACTION_NEAR_NEG1_RATE=("ACTION_NEAR_NEG1_RATE", "mean"),
            ACTION_NEAR_POS1_RATE=("ACTION_NEAR_POS1_RATE", "mean"),
            TRAINING_TIME_MIN=("TRAINING_TIME_MIN", "mean"),
        )
        .reset_index()
        .sort_values(["SPLIT", "VARIANT"])
    )


# ============================================================
# MODEL FACTORY
# ============================================================

def make_env(seed: int, variant: dict) -> Monitor:
    env = OptionHedgingAblationEnv(
        transitions_df=train_transitions,
        feature_cols=FEATURE_COLS,
        feature_mean=feature_mean,
        feature_std=feature_std,
        residual_delta=bool(variant["RESIDUAL_DELTA"]),
        adjustment_limit=variant["ADJUSTMENT_LIMIT"],
        no_trade_band=float(variant["NO_TRADE_BAND"]),
        downside_penalty=float(variant["DOWNSIDE_PENALTY"]),
        delta_risk_penalty=float(variant["DELTA_RISK_PENALTY"]),
        extra_cost_penalty=float(variant["EXTRA_COST_PENALTY"]),
        hedge_min=HEDGE_MIN,
        hedge_max=HEDGE_MAX,
        linear_transaction_cost_rate=LINEAR_TRANSACTION_COST_RATE,
        quadratic_impact_rate=QUADRATIC_IMPACT_RATE,
        reward_scale_mode=REWARD_SCALE_MODE,
        min_reward_scale=MIN_REWARD_SCALE,
        random_episode=True,
        seed=seed,
    )

    return Monitor(env)


def make_model(env: Monitor, seed: int) -> PPO:
    return PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=PPO_POLICY_KWARGS,
        tensorboard_log=str(OUTPUT_DIR / "tb_logs" / "ablation_ppo_v3c"),
        seed=seed,
    )


# ============================================================
# ENV CHECK
# ============================================================

env_check = OptionHedgingAblationEnv(
    transitions_df=train_transitions,
    feature_cols=FEATURE_COLS,
    feature_mean=feature_mean,
    feature_std=feature_std,
    residual_delta=True,
    adjustment_limit=DEFAULT_ADJUSTMENT_LIMIT,
    no_trade_band=DEFAULT_NO_TRADE_BAND,
    downside_penalty=0.50,
    delta_risk_penalty=0.50,
    extra_cost_penalty=0.00,
    seed=42,
)

check_env(env_check, warn=True)
print("\nEnvironment check passed.")


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

all_step_results = []
all_episode_results = []
metrics_by_seed_list = []
training_log_rows = []

for variant in ABLATION_VARIANTS:
    variant_name = variant["VARIANT"]
    variant_dir = MODEL_DIR / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    for seed in SEEDS:
        print("\n" + "=" * 90)
        print(f"Training PPO ablation variant={variant_name}, seed={seed}")
        print("=" * 90)

        train_env = make_env(seed=seed, variant=variant)
        model = make_model(env=train_env, seed=seed)

        print("\nPolicy architecture:")
        print(model.policy)

        start_time = time.time()
        model.learn(total_timesteps=TOTAL_TIMESTEPS)
        elapsed_min = (time.time() - start_time) / 60.0

        model_path = variant_dir / f"ppo_ablation_{variant_name}_seed_{seed}"
        model.save(model_path)

        print("\nSaved model:")
        print(model_path)
        print(f"Training time: {elapsed_min:.2f} minutes")

        step_df, ep_df = evaluate_ppo_variant(
            model=model,
            transitions_df=transitions,
            variant=variant,
            seed=seed,
            training_time_min=elapsed_min,
            deterministic=True,
        )

        seed_metrics = make_metrics(ep_df, ["VARIANT", "ALGORITHM", "SEED", "SPLIT", "STRATEGY"])

        print(f"\nPPO ablation results by split, variant={variant_name}, seed={seed}:")
        print(seed_metrics)

        all_step_results.append(step_df)
        all_episode_results.append(ep_df)
        metrics_by_seed_list.append(seed_metrics)

        training_log_rows.append({
            "VARIANT": variant_name,
            "ALGORITHM": "PPO",
            "SEED": seed,
            "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
            "TRAINING_TIME_MIN": elapsed_min,
            "MODEL_PATH": str(model_path),
        })


rl_step_results = pd.concat(all_step_results, ignore_index=True)
rl_episode_results = pd.concat(all_episode_results, ignore_index=True)
rl_metrics_by_seed = pd.concat(metrics_by_seed_list, ignore_index=True)

baseline_episode_results = make_baseline_results(transitions)

all_episode_results = pd.concat(
    [baseline_episode_results, rl_episode_results],
    ignore_index=True,
    sort=False,
)

comparison_metrics = make_metrics(all_episode_results, ["VARIANT", "ALGORITHM", "SPLIT", "STRATEGY"])
ablation_summary = make_ablation_summary(rl_metrics_by_seed)
training_log = pd.DataFrame(training_log_rows)
variant_config = pd.DataFrame(ABLATION_VARIANTS)

# Save full step results to parquet to avoid Excel row-limit problems.
rl_step_results.to_parquet(STEP_RESULTS_PATH, index=False)

max_excel_rows = 1_000_000
if len(rl_step_results) <= max_excel_rows:
    step_results_for_excel = rl_step_results
else:
    step_results_for_excel = rl_step_results.sample(
        n=max_excel_rows,
        random_state=42,
    ).sort_values(["VARIANT", "SEED", "EPISODE_ID", "QUOTE_DATE"])

print("\nComparison metrics:")
print(comparison_metrics)

print("\nAblation summary:")
print(ablation_summary)

with pd.ExcelWriter(EXCEL_OUTPUT_PATH, engine="openpyxl") as writer:
    variant_config.to_excel(writer, sheet_name="Variant_Config", index=False)
    all_episode_results.to_excel(writer, sheet_name="Episode_Results", index=False)
    comparison_metrics.to_excel(writer, sheet_name="Metrics", index=False)
    rl_metrics_by_seed.to_excel(writer, sheet_name="Metrics_By_Seed", index=False)
    ablation_summary.to_excel(writer, sheet_name="Ablation_Summary", index=False)
    training_log.to_excel(writer, sheet_name="Training_Log", index=False)
    step_results_for_excel.to_excel(writer, sheet_name="Step_Results_Sample", index=False)

print("\nSaved ablation Excel:")
print(EXCEL_OUTPUT_PATH)

print("\nSaved full step results parquet:")
print(STEP_RESULTS_PATH)
