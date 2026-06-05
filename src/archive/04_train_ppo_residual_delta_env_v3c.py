"""
PPO deep hedging environment v2: residual-delta, risk-aware reward.

Main changes versus the earlier PPO risk-aware script:
1. The PPO policy no longer learns the whole hedge ratio from scratch.
   It learns a bounded adjustment around Black-Scholes/market delta:

       hedge_t = clip(delta_t + adjustment_limit * raw_action_t, 0, 1)

2. The reward is scaled by option price to reduce PPO reward-scale instability.

3. Risk penalty is changed from symmetric pnl^2 to more hedging-specific terms:
   - downside-only PnL penalty: penalizes losses, not large gains
   - delta-risk penalty: penalizes large deviations from delta in volatile states

4. Optional no-trade band is added to reduce unnecessary turnover.

5. Actor/critic neural network architecture is explicitly specified.

Recommended first run:
    SEEDS = [42]
    TOTAL_TIMESTEPS = 100_000

For thesis-grade robustness later:
    SEEDS = [1, 2, 3, 4, 5]
    TOTAL_TIMESTEPS = 300_000 or 500_000
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
BASELINE_FILE = "baseline_episode_results_with_spy.parquet"


def _candidate_project_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    candidates = [
        here.parent,
        here.parent.parent,
        Path.cwd(),
        Path.cwd().parent,
        Path("/mnt/data"),
    ]
    # keep order but deduplicate
    out: list[Path] = []
    for p in candidates:
        p = p.resolve()
        if p not in out:
            out.append(p)
    return out


def find_existing_file(filename: str, relative_dirs: Iterable[str]) -> Path:
    """Find a file either in a normal project structure or directly in /mnt/data."""
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
BASELINE_PATH = find_existing_file(
    BASELINE_FILE,
    relative_dirs=["outputs", ""],
)

# If the script is inside scripts/, outputs should go to project/outputs.
# If the script is run from /mnt/data, outputs go to /mnt/data/outputs.
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

# Keep the same linear transaction cost as your baseline for fair first comparison.
LINEAR_TRANSACTION_COST_RATE = 0.0005

# Optional market-impact/friction term. Start with 0.0 for clean comparison to baseline.
# Later try 0.00005, 0.0001, 0.0002 to test high-friction environments.
QUADRATIC_IMPACT_RATE = 0.0

# Short call hedge: long SPY between 0 and 1 share per option share.
HEDGE_MIN = 0.0
HEDGE_MAX = 1.0

# Residual-delta action design.
# raw_action = -1 means delta - adjustment_limit
# raw_action = +1 means delta + adjustment_limit
ADJUSTMENT_LIMIT = 0.10
# If abs(new_target - previous_hedge) is smaller than this band, keep previous hedge.
# Start small. Try 0.00, 0.02, 0.05 in sensitivity analysis.
NO_TRADE_BAND = 0.02

# Reward scaling. This makes the reward more stable across SPY price regimes.
REWARD_SCALE_MODE = "option_mid"  # choices: "option_mid", "spy", "none"
MIN_REWARD_SCALE = 1.0

# Risk-aware reward weights.
# These operate on scaled quantities, so values around 0.05-2.0 are reasonable to test.
DOWNSIDE_PENALTY = 0.50
DELTA_RISK_PENALTY = 0.50
EXTRA_COST_PENALTY = 0.00  # cost is already included in raw PnL; keep 0 first

TOTAL_TIMESTEPS = 100_000
SEEDS = [1, 2, 3, 4, 5]

PPO_MODEL_DIR = OUTPUT_DIR / "ppo_residual_delta_v3c_multiseed_models"
PPO_MODEL_DIR.mkdir(parents=True, exist_ok=True)

PPO_COMPARISON_PATH = OUTPUT_DIR / "comparison_baseline_ppo_residual_delta_v3c_multiseed.xlsx"

# Explicit actor/critic neural network architecture.
# This makes the neural-network role visible for thesis writing.
POLICY_KWARGS = dict(
    net_arch=dict(
        pi=[256, 256],
        vf=[256, 256],
    ),
    activation_fn=th.nn.Tanh,
)

BASE_FEATURE_COLS = [
    # Prefer relative/financial variables over raw price levels.
    "DTE",
    "SPY_LOG_MONEYNESS",
    "SPY_MONEYNESS",
    "OPTION_MID_OVER_SPY",
    "SPY_RET_LAG1",
    # Option risk descriptors.
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

    # Use relative price variables instead of raw price levels.
    if "OPTION_MID" in df.columns and "SPY_CLOSE" in df.columns:
        df["OPTION_MID_OVER_SPY"] = df["OPTION_MID"].astype(float) / df["SPY_CLOSE"].astype(float)

    if "SPY_CLOSE" in df.columns:
        df["SPY_RET_LAG1"] = (
            df.groupby("EPISODE_ID")["SPY_CLOSE"]
            .pct_change()
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

    # If these columns were not precomputed for some reason, create them.
    if "SPY_MONEYNESS" not in df.columns and {"SPY_CLOSE", "STRIKE"}.issubset(df.columns):
        df["SPY_MONEYNESS"] = df["SPY_CLOSE"].astype(float) / df["STRIKE"].astype(float)

    if "SPY_LOG_MONEYNESS" not in df.columns and {"SPY_CLOSE", "STRIKE"}.issubset(df.columns):
        df["SPY_LOG_MONEYNESS"] = np.log(
            df["SPY_CLOSE"].astype(float) / df["STRIKE"].astype(float)
        )

    # Make robust to missing/infinite feature values.
    for c in BASE_FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return df


transitions = add_engineered_features(pd.read_parquet(TRANSITIONS_PATH))
baseline_results = pd.read_parquet(BASELINE_PATH)

FEATURE_COLS = [c for c in BASE_FEATURE_COLS if c in transitions.columns]

train_transitions = transitions[transitions["SPLIT"] == "train"].copy()
val_transitions = transitions[transitions["SPLIT"] == "val"].copy()
test_transitions = transitions[transitions["SPLIT"] == "test"].copy()

feature_mean = train_transitions[FEATURE_COLS].mean()
feature_std = train_transitions[FEATURE_COLS].std().replace(0, 1)

print("Loaded transitions:", transitions.shape)
print("Transitions path:", TRANSITIONS_PATH)
print("Baseline path:", BASELINE_PATH)
print("Transitions by split:")
print(transitions["SPLIT"].value_counts())
print("\nFeature columns:")
print(FEATURE_COLS)
print("\nOutput directory:", OUTPUT_DIR)


# ============================================================
# ENVIRONMENT V2
# ============================================================

class OptionHedgingEnvResidualDelta(gym.Env):
    """
    PPO environment for short call deep hedging.

    Key design:
        raw_action in [-1, 1]
        target hedge = clip(delta + adjustment_limit * raw_action, hedge_min, hedge_max)

    This turns PPO into a residual learner around delta hedge instead of forcing PPO
    to rediscover delta from scratch.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        transitions_df: pd.DataFrame,
        feature_cols: list[str],
        feature_mean: pd.Series,
        feature_std: pd.Series,
        adjustment_limit: float = 0.25,
        no_trade_band: float = 0.02,
        hedge_min: float = 0.0,
        hedge_max: float = 1.0,
        linear_transaction_cost_rate: float = 0.0005,
        quadratic_impact_rate: float = 0.0,
        downside_penalty: float = 0.5,
        delta_risk_penalty: float = 0.1,
        extra_cost_penalty: float = 0.0,
        reward_scale_mode: str = "option_mid",
        min_reward_scale: float = 1.0,
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

        # features + previous hedge + delta gap(prev_hedge - delta)
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
        delta = float(row["OPTION_DELTA"])
        return float(np.clip(delta, self.hedge_min, self.hedge_max))

    def _scale_action_to_hedge(self, action: np.ndarray, row: pd.Series) -> tuple[float, float, float, float]:
        raw_action = float(np.clip(action[0], -1.0, 1.0))
        delta = self._get_delta(row)
        desired_hedge = float(np.clip(
            delta + self.adjustment_limit * raw_action,
            self.hedge_min,
            self.hedge_max,
        ))

        # No-trade band: avoid tiny rebalances.
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
        # Default daily transition. If DTE and NEXT_DTE exist, use their difference.
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
        # Robustness if IV is stored as percentage points, e.g. 25 instead of 0.25.
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
        }

        return obs, float(training_reward), terminated, truncated, info


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
    model: PPO,
    transitions_df: pd.DataFrame,
    strategy_name: str,
    seed: int,
    deterministic: bool = True,
    contract_multiplier: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate with raw accounting PnL so results are comparable to delta baseline."""
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
            desired_hedge = float(np.clip(delta + ADJUSTMENT_LIMIT * raw_action, HEDGE_MIN, HEDGE_MAX))

            if abs(desired_hedge - prev_hedge) < NO_TRADE_BAND:
                target_hedge = prev_hedge
            else:
                target_hedge = desired_hedge

            adjustment = target_hedge - delta
            trade_size = target_hedge - prev_hedge

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

            step_rows.append({
                "SEED": seed,
                "EPISODE_ID": episode_id,
                "SPLIT": split,
                "QUOTE_DATE": row["QUOTE_DATE"],
                "STRATEGY": strategy_name,
                "STRATEGY_FAMILY": "ppo_residual_delta_v2",
                "RAW_ACTION": raw_action,
                "DELTA": delta,
                "DESIRED_HEDGE": desired_hedge,
                "HEDGE": target_hedge,
                "PREV_HEDGE": prev_hedge,
                "ADJUSTMENT_FROM_DELTA": adjustment,
                "REWARD_PER_SHARE": reward_per_share,
                "REWARD": reward_per_share * contract_multiplier,
                "TRANSACTION_COST": transaction_cost * contract_multiplier,
                "TURNOVER": turnover,
            })

            prev_hedge = target_hedge

        results.append({
            "SEED": seed,
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
            "STRATEGY": strategy_name,
            "STRATEGY_FAMILY": "ppo_residual_delta_v2",
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
    return (
        df.groupby(group_cols)
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
        )
        .reset_index()
        .sort_values(group_cols)
    )


# ============================================================
# ENV CHECK
# ============================================================

env_check = OptionHedgingEnvResidualDelta(
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
    delta_risk_penalty=DELTA_RISK_PENALTY,
    extra_cost_penalty=EXTRA_COST_PENALTY,
    reward_scale_mode=REWARD_SCALE_MODE,
    min_reward_scale=MIN_REWARD_SCALE,
    random_episode=True,
    seed=42,
)

check_env(env_check, warn=True)
print("\nEnvironment check passed.")


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

all_ppo_steps = []
all_ppo_episodes = []
metrics_by_seed_list = []

for seed in SEEDS:
    print("\n" + "=" * 80)
    print(f"Training PPO residual-delta v2, seed={seed}")
    print("=" * 80)

    train_env = Monitor(
        OptionHedgingEnvResidualDelta(
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
            delta_risk_penalty=DELTA_RISK_PENALTY,
            extra_cost_penalty=EXTRA_COST_PENALTY,
            reward_scale_mode=REWARD_SCALE_MODE,
            min_reward_scale=MIN_REWARD_SCALE,
            random_episode=True,
            seed=seed,
        )
    )

    start_time = time.time()

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
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
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=str(OUTPUT_DIR / "tb_logs" / "ppo_residual_delta_v3c_multiseed"),
        seed=seed,
    )

    print("\nPolicy architecture:")
    print(model.policy)

    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    elapsed = time.time() - start_time
    model_path = PPO_MODEL_DIR / f"ppo_residual_delta_v2_seed_{seed}"
    model.save(model_path)

    print("\nSaved PPO model:")
    print(model_path)
    print(f"Training time: {elapsed / 60:.2f} minutes")

    strategy_name = f"ppo_residual_delta_v2_seed_{seed}"
    ppo_steps, ppo_ep = evaluate_rl_model_residual_delta(
        model=model,
        transitions_df=transitions,
        strategy_name=strategy_name,
        seed=seed,
        deterministic=True,
        contract_multiplier=CONTRACT_MULTIPLIER,
    )

    seed_metrics = make_metrics(ppo_ep, ["SEED", "SPLIT", "STRATEGY"])
    print("\nPPO residual-delta v2 results by split:")
    print(seed_metrics)

    all_ppo_steps.append(ppo_steps)
    all_ppo_episodes.append(ppo_ep)
    metrics_by_seed_list.append(seed_metrics)


ppo_steps_all = pd.concat(all_ppo_steps, ignore_index=True)
ppo_ep_all = pd.concat(all_ppo_episodes, ignore_index=True)
metrics_by_seed = pd.concat(metrics_by_seed_list, ignore_index=True)

# ============================================================
# COMPARE BASELINE VS PPO V2
# ============================================================

baseline_results_for_export = baseline_results.copy()
baseline_results_for_export["SEED"] = np.nan
baseline_results_for_export["STRATEGY_FAMILY"] = baseline_results_for_export["STRATEGY"]

all_results = pd.concat(
    [baseline_results_for_export, ppo_ep_all],
    ignore_index=True,
    sort=False,
)

comparison_metrics = make_metrics(all_results, ["SPLIT", "STRATEGY"])

# Across-seed summary for PPO only.
seed_summary = (
    metrics_by_seed
    .groupby(["SPLIT"])
    .agg(
        N_SEEDS=("SEED", "nunique"),
        MEAN_OF_MEAN_PNL=("MEAN_PNL", "mean"),
        STD_OF_MEAN_PNL=("MEAN_PNL", "std"),
        MEAN_OF_CVAR_95=("CVAR_95", "mean"),
        STD_OF_CVAR_95=("CVAR_95", "std"),
        MEAN_OF_SHARPE_LIKE=("SHARPE_LIKE", "mean"),
        STD_OF_SHARPE_LIKE=("SHARPE_LIKE", "std"),
        MEAN_TURNOVER=("MEAN_TURNOVER", "mean"),
        MEAN_TC=("MEAN_TC", "mean"),
        AVG_HEDGE=("AVG_HEDGE", "mean"),
    )
    .reset_index()
)

print("\nComparison metrics:")
print(comparison_metrics)

print("\nPPO seed summary:")
print(seed_summary)

with pd.ExcelWriter(PPO_COMPARISON_PATH, engine="openpyxl") as writer:
    all_results.to_excel(writer, sheet_name="Episode_Results", index=False)
    comparison_metrics.to_excel(writer, sheet_name="Metrics", index=False)
    metrics_by_seed.to_excel(writer, sheet_name="Metrics_By_Seed", index=False)
    seed_summary.to_excel(writer, sheet_name="Seed_Summary", index=False)
    ppo_steps_all.to_excel(writer, sheet_name="PPO_Step_Results", index=False)

print("\nSaved comparison:")
print(PPO_COMPARISON_PATH)
