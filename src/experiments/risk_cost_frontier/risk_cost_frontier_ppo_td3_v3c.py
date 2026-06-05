"""
05f_risk_cost_frontier_ppo_td3_v3c.py

Risk-cost frontier experiment for residual-delta V3C deep hedging.

This run compares PPO vs TD3 while sweeping DELTA_RISK_PENALTY.

Main purpose:
    Compare PPO vs TD3 under the same residual-delta V3C environment while sweeping delta-risk penalty.

Environment:
    - Short call hedging with SPY.
    - Daily rebalancing.
    - Proportional transaction cost.
    - Terminal liquidation cost.
    - Residual-delta action:
          h_t = clip(delta_t + adjustment_limit * action_t, 0, 1)
    - Fixed no-trade band.
    - Risk-aware reward:
          scaled_pnl
          - downside_penalty * downside^2
          - delta_risk_penalty * delta_risk

Output:
    outputs/risk_cost_frontier_ppo_td3_v3c_100k_3seeds.xlsx
    outputs/risk_cost_frontier_ppo_td3_v3c_100k_3seeds_step_results.parquet
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
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise


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
# CONFIG: MAIN V3C ENVIRONMENT
# ============================================================

CONTRACT_MULTIPLIER = 100

# Main normal-cost experiment.
LINEAR_TRANSACTION_COST_RATE = 0.0005
QUADRATIC_IMPACT_RATE = 0.0

HEDGE_MIN = 0.0
HEDGE_MAX = 1.0

# V3C residual-delta setup.
ADJUSTMENT_LIMIT = 0.10
NO_TRADE_BAND = 0.02

REWARD_SCALE_MODE = "option_mid"
MIN_REWARD_SCALE = 1.0

DOWNSIDE_PENALTY = 0.50
DELTA_RISK_PENALTY = 0.50
EXTRA_COST_PENALTY = 0.00

TOTAL_TIMESTEPS = 100_000
SEEDS = [1, 2, 3]
ALGORITHMS = ["PPO", "TD3"]

# Risk-cost frontier sweep. Full V3C corresponds to 0.50.
DELTA_RISK_PENALTY_LIST = [0.0, 0.1, 0.3, 0.5, 1.0]

MODEL_DIR = OUTPUT_DIR / "risk_cost_frontier_ppo_td3_v3c_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_OUTPUT_PATH = OUTPUT_DIR / "risk_cost_frontier_ppo_td3_v3c_100k_3seeds.xlsx"
STEP_RESULTS_PATH = OUTPUT_DIR / "risk_cost_frontier_ppo_td3_v3c_100k_3seeds_step_results.parquet"

# PPO uses pi/vf network split.
PPO_POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[256, 256],
        vf=[256, 256],
    ),
    activation_fn=th.nn.Tanh,
)

# SAC/TD3 use pi/qf network split.
OFF_POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[256, 256],
        qf=[256, 256],
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

feature_mean = train_transitions[FEATURE_COLS].mean()
feature_std = train_transitions[FEATURE_COLS].std().replace(0, 1)

print("Loaded transitions:", transitions.shape)
print("Transitions path:", TRANSITIONS_PATH)
print("Transitions by split:")
print(transitions["SPLIT"].value_counts())
print("\nFeature columns:")
print(FEATURE_COLS)
print("\nOutput directory:", OUTPUT_DIR)


# ============================================================
# ENVIRONMENT: RESIDUAL-DELTA V3C
# ============================================================

class OptionHedgingEnvResidualDelta(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        transitions_df: pd.DataFrame,
        feature_cols: list[str],
        feature_mean: pd.Series,
        feature_std: pd.Series,
        adjustment_limit: float = ADJUSTMENT_LIMIT,
        no_trade_band: float = NO_TRADE_BAND,
        hedge_min: float = HEDGE_MIN,
        hedge_max: float = HEDGE_MAX,
        linear_transaction_cost_rate: float = LINEAR_TRANSACTION_COST_RATE,
        quadratic_impact_rate: float = QUADRATIC_IMPACT_RATE,
        downside_penalty: float = DOWNSIDE_PENALTY,
        delta_risk_penalty: float = DELTA_RISK_PENALTY,
        extra_cost_penalty: float = EXTRA_COST_PENALTY,
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

        self.adjustment_limit = adjustment_limit
        self.no_trade_band = no_trade_band
        self.hedge_min = hedge_min
        self.hedge_max = hedge_max

        self.linear_transaction_cost_rate = linear_transaction_cost_rate
        self.quadratic_impact_rate = quadratic_impact_rate
        self.downside_penalty = downside_penalty
        self.delta_risk_penalty = delta_risk_penalty
        self.extra_cost_penalty = extra_cost_penalty
        self.reward_scale_mode = reward_scale_mode
        self.min_reward_scale = min_reward_scale
        self.random_episode = random_episode

        self.episode_ids = self.df["EPISODE_ID"].unique().tolist()
        self.episode_data = {
            eid: group.reset_index(drop=True)
            for eid, group in self.df.groupby("EPISODE_ID")
        }

        self.rng = np.random.default_rng(seed)

        obs_dim = len(self.feature_cols) + 2
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
        return float(np.clip(float(row["OPTION_DELTA"]), self.hedge_min, self.hedge_max))

    def _scale_action_to_hedge(self, action: np.ndarray, row: pd.Series) -> tuple[float, float, float, float]:
        raw_action = float(np.clip(action[0], -1.0, 1.0))
        delta = self._get_delta(row)

        desired_hedge = float(
            np.clip(
                delta + self.adjustment_limit * raw_action,
                self.hedge_min,
                self.hedge_max,
            )
        )

        if abs(desired_hedge - self.prev_hedge) < self.no_trade_band:
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
# BASELINE HELPERS
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
    episode_result["ALGORITHM"] = "baseline"
    episode_result["SEED"] = np.nan
    episode_result["TRAINING_TIME_MIN"] = np.nan

    return episode_result


def make_baseline_results(transitions_df: pd.DataFrame) -> pd.DataFrame:
    nohedge_ep = run_baseline_for_current_cost(transitions_df, strategy="no_hedge")
    delta_ep = run_baseline_for_current_cost(transitions_df, strategy="delta")
    return pd.concat([nohedge_ep, delta_ep], ignore_index=True)


# ============================================================
# EVALUATION HELPERS
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


def evaluate_rl_model_residual_delta(
    model,
    transitions_df: pd.DataFrame,
    algorithm_name: str,
    strategy_name: str,
    seed: int,
    training_time_min: float,
    delta_risk_penalty: float,
    frontier_variant: str,
    deterministic: bool = True,
    contract_multiplier: int = CONTRACT_MULTIPLIER,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
            raw_action = float(np.clip(action[0], -1.0, 1.0))

            delta = float(np.clip(float(row["OPTION_DELTA"]), HEDGE_MIN, HEDGE_MAX))
            desired_hedge = float(np.clip(
                delta + ADJUSTMENT_LIMIT * raw_action,
                HEDGE_MIN,
                HEDGE_MAX,
            ))

            if abs(desired_hedge - prev_hedge) < NO_TRADE_BAND:
                target_hedge = prev_hedge
            else:
                target_hedge = desired_hedge

            adjustment = target_hedge - delta
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
                "FRONTIER_VARIANT": frontier_variant,
                "DELTA_RISK_PENALTY": delta_risk_penalty,
                "SEED": seed,
                "ALGORITHM": algorithm_name,
                "EPISODE_ID": episode_id,
                "SPLIT": split,
                "QUOTE_DATE": row["QUOTE_DATE"],
                "STRATEGY": strategy_name,
                "RAW_ACTION": raw_action,
                "DELTA": delta,
                "DESIRED_HEDGE": desired_hedge,
                "HEDGE": target_hedge,
                "PREV_HEDGE": prev_hedge,
                "ADJUSTMENT_FROM_DELTA": adjustment,
                "NO_TRADE": no_trade,
                "REWARD_PER_SHARE": reward_per_share,
                "REWARD": reward_per_share * contract_multiplier,
                "TRANSACTION_COST": transaction_cost * contract_multiplier,
                "TURNOVER": turnover,
            })

            prev_hedge = target_hedge

        results.append({
            "FRONTIER_VARIANT": frontier_variant,
            "DELTA_RISK_PENALTY": delta_risk_penalty,
            "SEED": seed,
            "ALGORITHM": algorithm_name,
            "EPISODE_ID": episode_id,
            "SPLIT": split,
            "START_DATE": ep["QUOTE_DATE"].iloc[0],
            "END_DATE": ep["NEXT_QUOTE_DATE"].iloc[-1],
            "N_STEPS": len(ep),
            "TERMINAL_PNL": np.sum(rewards_per_share) * contract_multiplier,
            "TOTAL_TC": np.sum(tcs_per_share) * contract_multiplier,
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
            "STRATEGY": strategy_name,
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
    agg_dict = dict(
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
        TRAINING_TIME_MIN=("TRAINING_TIME_MIN", "mean"),
    )

    optional_cols = {
        "AVG_DELTA": ("AVG_DELTA", "mean"),
        "AVG_ADJUSTMENT_FROM_DELTA": ("AVG_ADJUSTMENT_FROM_DELTA", "mean"),
        "NO_TRADE_RATE": ("NO_TRADE_RATE", "mean"),
        "ACTION_NEAR_NEG1_RATE": ("ACTION_NEAR_NEG1_RATE", "mean"),
        "ACTION_NEAR_POS1_RATE": ("ACTION_NEAR_POS1_RATE", "mean"),
    }
    for out_col, agg in optional_cols.items():
        if agg[0] in df.columns:
            agg_dict[out_col] = agg

    return (
        df.groupby(group_cols)
        .agg(**agg_dict)
        .reset_index()
        .sort_values(group_cols)
    )


def make_algorithm_summary(metrics_by_seed: pd.DataFrame) -> pd.DataFrame:
    # Only RL algorithms have seeds.
    rl = metrics_by_seed.dropna(subset=["SEED"]).copy()

    return (
        rl.groupby(["FRONTIER_VARIANT", "DELTA_RISK_PENALTY", "ALGORITHM", "SPLIT"])
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
            TRAINING_TIME_MIN=("TRAINING_TIME_MIN", "mean"),
        )
        .reset_index()
        .sort_values(["SPLIT", "DELTA_RISK_PENALTY", "ALGORITHM"])
    )


# ============================================================
# MODEL FACTORY
# ============================================================

def make_env(seed: int, delta_risk_penalty: float) -> Monitor:
    env = OptionHedgingEnvResidualDelta(
        transitions_df=train_transitions,
        feature_cols=FEATURE_COLS,
        feature_mean=feature_mean,
        feature_std=feature_std,
        adjustment_limit=ADJUSTMENT_LIMIT,
        no_trade_band=NO_TRADE_BAND,
        hedge_min=HEDGE_MIN,
        hedge_max=HEDGE_MAX,
        linear_transaction_cost_rate=LINEAR_TRANSACTION_COST_RATE,
        quadratic_impact_rate=QUADRATIC_IMPACT_RATE,
        downside_penalty=DOWNSIDE_PENALTY,
        delta_risk_penalty=delta_risk_penalty,
        extra_cost_penalty=EXTRA_COST_PENALTY,
        reward_scale_mode=REWARD_SCALE_MODE,
        min_reward_scale=MIN_REWARD_SCALE,
        random_episode=True,
        seed=seed,
    )
    return Monitor(env)


def make_model(algorithm: str, env: Monitor, seed: int):
    algorithm = algorithm.upper()

    if algorithm == "PPO":
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
            tensorboard_log=str(OUTPUT_DIR / "tb_logs" / "risk_cost_frontier_ppo_td3_v3c"),
            seed=seed,
        )

    if algorithm == "SAC":
        return SAC(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            target_update_interval=1,
            policy_kwargs=OFF_POLICY_KWARGS,
            tensorboard_log=str(OUTPUT_DIR / "tb_logs" / "risk_cost_frontier_ppo_td3_v3c"),
            seed=seed,
        )

    if algorithm == "TD3":
        action_noise = NormalActionNoise(
            mean=np.zeros(1),
            sigma=0.10 * np.ones(1),
        )

        return TD3(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            buffer_size=100_000,
            learning_starts=1_000,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=action_noise,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs=OFF_POLICY_KWARGS,
            tensorboard_log=str(OUTPUT_DIR / "tb_logs" / "risk_cost_frontier_ppo_td3_v3c"),
            seed=seed,
        )

    raise ValueError(f"Unsupported algorithm: {algorithm}")


# ============================================================
# ENV CHECK
# ============================================================

env_check = OptionHedgingEnvResidualDelta(
    transitions_df=train_transitions,
    feature_cols=FEATURE_COLS,
    feature_mean=feature_mean,
    feature_std=feature_std,
    seed=42,
)

check_env(env_check, warn=True)
print("\nEnvironment check passed.")


# ============================================================
# TRAIN AND EVALUATE RISK-COST FRONTIER
# ============================================================

all_step_results = []
all_episode_results = []
metrics_by_seed_list = []
training_log_rows = []

for delta_risk_penalty in DELTA_RISK_PENALTY_LIST:
    frontier_variant = f"delta_risk_{delta_risk_penalty:.1f}".replace(".", "p")

    print("\n" + "#" * 90)
    print(f"Risk-cost frontier variant: {frontier_variant}")
    print(f"DELTA_RISK_PENALTY = {delta_risk_penalty}")
    print("#" * 90)

    for algorithm in ALGORITHMS:
        algo_dir = MODEL_DIR / frontier_variant / algorithm.lower()
        algo_dir.mkdir(parents=True, exist_ok=True)

        for seed in SEEDS:
            print("\n" + "=" * 90)
            print(
                f"Training {algorithm} residual-delta V3C, "
                f"frontier_variant={frontier_variant}, seed={seed}"
            )
            print("=" * 90)

            train_env = make_env(seed=seed, delta_risk_penalty=float(delta_risk_penalty))
            model = make_model(algorithm=algorithm, env=train_env, seed=seed)

            print("\nPolicy architecture:")
            print(model.policy)

            start_time = time.time()
            model.learn(total_timesteps=TOTAL_TIMESTEPS)
            elapsed_min = (time.time() - start_time) / 60.0

            model_path = algo_dir / f"{algorithm.lower()}_v3c_{frontier_variant}_seed_{seed}"
            model.save(model_path)

            print("\nSaved model:")
            print(model_path)
            print(f"Training time: {elapsed_min:.2f} minutes")

            strategy_name = f"{algorithm.lower()}_v3c_{frontier_variant}_seed_{seed}"

            step_df, ep_df = evaluate_rl_model_residual_delta(
                model=model,
                transitions_df=transitions,
                algorithm_name=algorithm,
                strategy_name=strategy_name,
                seed=seed,
                training_time_min=elapsed_min,
                delta_risk_penalty=float(delta_risk_penalty),
                frontier_variant=frontier_variant,
                deterministic=True,
                contract_multiplier=CONTRACT_MULTIPLIER,
            )

            seed_metrics = make_metrics(
                ep_df,
                [
                    "FRONTIER_VARIANT",
                    "DELTA_RISK_PENALTY",
                    "ALGORITHM",
                    "SEED",
                    "SPLIT",
                    "STRATEGY",
                ],
            )

            print(
                f"\n{algorithm} results by split, "
                f"variant={frontier_variant}, seed={seed}:"
            )
            print(seed_metrics)

            all_step_results.append(step_df)
            all_episode_results.append(ep_df)
            metrics_by_seed_list.append(seed_metrics)

            training_log_rows.append({
                "FRONTIER_VARIANT": frontier_variant,
                "DELTA_RISK_PENALTY": float(delta_risk_penalty),
                "ALGORITHM": algorithm,
                "SEED": seed,
                "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
                "TRAINING_TIME_MIN": elapsed_min,
                "MODEL_PATH": str(model_path),
            })


rl_step_results = pd.concat(all_step_results, ignore_index=True)
rl_episode_results = pd.concat(all_episode_results, ignore_index=True)
rl_metrics_by_seed = pd.concat(metrics_by_seed_list, ignore_index=True)

baseline_episode_results = make_baseline_results(transitions)
baseline_episode_results["FRONTIER_VARIANT"] = "baseline"
baseline_episode_results["DELTA_RISK_PENALTY"] = np.nan

all_episode_results = pd.concat(
    [baseline_episode_results, rl_episode_results],
    ignore_index=True,
    sort=False,
)

comparison_metrics = make_metrics(
    all_episode_results,
    ["FRONTIER_VARIANT", "DELTA_RISK_PENALTY", "ALGORITHM", "SPLIT", "STRATEGY"],
)

algorithm_summary = make_algorithm_summary(rl_metrics_by_seed)
training_log = pd.DataFrame(training_log_rows)

# Compact frontier table for thesis plots.
test_frontier = algorithm_summary[algorithm_summary["SPLIT"].eq("test")].copy()
test_frontier = test_frontier.sort_values(["ALGORITHM", "DELTA_RISK_PENALTY"])

# Save full step results to parquet to avoid Excel row-limit problems.
rl_step_results.to_parquet(STEP_RESULTS_PATH, index=False)

# If step results are small enough, include them in Excel; otherwise include a sample.
max_excel_rows = 1_000_000
if len(rl_step_results) <= max_excel_rows:
    step_results_for_excel = rl_step_results
else:
    step_results_for_excel = rl_step_results.sample(
        n=max_excel_rows,
        random_state=42,
    ).sort_values(["FRONTIER_VARIANT", "ALGORITHM", "SEED", "EPISODE_ID", "QUOTE_DATE"])

print("\nComparison metrics:")
print(comparison_metrics)

print("\nAlgorithm summary:")
print(algorithm_summary)

print("\nTest frontier:")
print(test_frontier)

frontier_config = pd.DataFrame({
    "DELTA_RISK_PENALTY": DELTA_RISK_PENALTY_LIST,
    "DOWNSIDE_PENALTY": DOWNSIDE_PENALTY,
    "ADJUSTMENT_LIMIT": ADJUSTMENT_LIMIT,
    "NO_TRADE_BAND": NO_TRADE_BAND,
    "LINEAR_TRANSACTION_COST_RATE": LINEAR_TRANSACTION_COST_RATE,
    "QUADRATIC_IMPACT_RATE": QUADRATIC_IMPACT_RATE,
    "ALGORITHMS": ", ".join(ALGORITHMS),
    "SEEDS": ", ".join(map(str, SEEDS)),
    "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS,
})

with pd.ExcelWriter(EXCEL_OUTPUT_PATH, engine="openpyxl") as writer:
    frontier_config.to_excel(writer, sheet_name="Frontier_Config", index=False)
    all_episode_results.to_excel(writer, sheet_name="Episode_Results", index=False)
    comparison_metrics.to_excel(writer, sheet_name="Metrics", index=False)
    rl_metrics_by_seed.to_excel(writer, sheet_name="Metrics_By_Seed", index=False)
    algorithm_summary.to_excel(writer, sheet_name="Algorithm_Summary", index=False)
    test_frontier.to_excel(writer, sheet_name="Test_Frontier", index=False)
    training_log.to_excel(writer, sheet_name="Training_Log", index=False)
    step_results_for_excel.to_excel(writer, sheet_name="Step_Results_Sample", index=False)

print("\nSaved Excel comparison:")
print(EXCEL_OUTPUT_PATH)

print("\nSaved full step results parquet:")
print(STEP_RESULTS_PATH)
