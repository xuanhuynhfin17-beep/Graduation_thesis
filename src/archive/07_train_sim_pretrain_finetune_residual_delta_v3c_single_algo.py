"""
07_train_sim_pretrain_finetune_residual_delta_v3c_single_algo.py

Train one DRL algorithm at a time using simulated pretraining and real-data fine-tuning.

Purpose:
    Run PPO, SAC, or TD3 separately to avoid long all-in-one runs.

Pipeline:
    1. Pretrain on simulated Black-Scholes / GBM transitions.
    2. Fine-tune on real SPY option train split.
    3. Evaluate on real train / validation / test splits using raw accounting PnL.

Example runs:
    py src\07_train_sim_pretrain_finetune_residual_delta_v3c_single_algo.py --algorithm PPO --seeds 1 2 3 --pretrain-steps 200000 --finetune-steps 100000

    py src\07_train_sim_pretrain_finetune_residual_delta_v3c_single_algo.py --algorithm SAC --seeds 1 2 3 --pretrain-steps 200000 --finetune-steps 100000 --sac-ent-coef 0.01

    py src\07_train_sim_pretrain_finetune_residual_delta_v3c_single_algo.py --algorithm TD3 --seeds 1 2 3 --pretrain-steps 200000 --finetune-steps 100000

Quick smoke test:
    py src\07_train_sim_pretrain_finetune_residual_delta_v3c_single_algo.py --algorithm PPO --seeds 1 --pretrain-steps 10000 --finetune-steps 5000

Outputs:
    outputs/sim_pretrain_finetune_v3c_<algorithm>_multiseed.xlsx
    outputs/sim_pretrain_finetune_v3c_<algorithm>_step_results.parquet
"""

from __future__ import annotations

import argparse
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

REAL_TRANSITIONS_FILE = "transitions_daily_top1_final_with_spy_2010_2023.parquet"
SIM_TRANSITIONS_FILE = "simulated_bs_transitions.parquet"


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


def infer_project_dir() -> Path:
    here = Path(__file__).resolve()
    if (here.parent.parent / "data" / "processed").exists():
        return here.parent.parent
    if (Path.cwd() / "data" / "processed").exists():
        return Path.cwd()
    return Path("/mnt/data")


PROJECT_DIR = infer_project_dir()
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# V3C CONFIG
# ============================================================

CONTRACT_MULTIPLIER = 100
LINEAR_TRANSACTION_COST_RATE = 0.0005
QUADRATIC_IMPACT_RATE = 0.0

HEDGE_MIN = 0.0
HEDGE_MAX = 1.0
ADJUSTMENT_LIMIT = 0.10
NO_TRADE_BAND = 0.02

REWARD_SCALE_MODE = "option_mid"
MIN_REWARD_SCALE = 1.0

DOWNSIDE_PENALTY = 0.50
DELTA_RISK_PENALTY = 0.50
EXTRA_COST_PENALTY = 0.00
OBS_CLIP = 10.0

PPO_POLICY_KWARGS = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    activation_fn=th.nn.Tanh,
)

OFF_POLICY_KWARGS = dict(
    net_arch=dict(pi=[256, 256], qf=[256, 256]),
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
# DATA LOADING
# ============================================================


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)

    if "OPTION_MID" in df.columns and "SPY_CLOSE" in df.columns:
        df["OPTION_MID_OVER_SPY"] = df["OPTION_MID"].astype(float) / df["SPY_CLOSE"].astype(float)

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
        df["SPY_LOG_MONEYNESS"] = np.log(df["SPY_CLOSE"].astype(float) / df["STRIKE"].astype(float))

    for c in BASE_FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return df


# ============================================================
# ENVIRONMENT
# ============================================================

class OptionHedgingEnvResidualDelta(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        transitions_df: pd.DataFrame,
        feature_cols: list[str],
        feature_mean: pd.Series,
        feature_std: pd.Series,
        random_episode: bool = True,
        seed: int = 42,
    ):
        super().__init__()
        self.df = transitions_df.copy()
        self.df = self.df.sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)
        self.feature_cols = feature_cols
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.random_episode = random_episode
        self.episode_ids = self.df["EPISODE_ID"].unique().tolist()
        self.episode_data = {
            eid: group.reset_index(drop=True)
            for eid, group in self.df.groupby("EPISODE_ID")
        }
        self.rng = np.random.default_rng(seed)

        obs_dim = len(self.feature_cols) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
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
        return float(np.clip(float(row["OPTION_DELTA"]), HEDGE_MIN, HEDGE_MAX))

    def _get_obs(self) -> np.ndarray:
        row = self.current_data.iloc[self.t]
        x = row[self.feature_cols].astype(float)
        x = (x - self.feature_mean[self.feature_cols]) / self.feature_std[self.feature_cols]
        x = x.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)
        x = np.clip(x, -OBS_CLIP, OBS_CLIP)
        delta = self._get_delta(row)
        delta_gap = self.prev_hedge - delta
        obs = np.concatenate([x, np.array([self.prev_hedge, delta_gap], dtype=np.float32)])
        return obs.astype(np.float32)

    def _reward_scale(self, row: pd.Series) -> float:
        if REWARD_SCALE_MODE == "option_mid" and "OPTION_MID" in row:
            return max(abs(float(row["OPTION_MID"])), MIN_REWARD_SCALE)
        if REWARD_SCALE_MODE == "spy" and "SPY_CLOSE" in row:
            return max(abs(float(row["SPY_CLOSE"])), MIN_REWARD_SCALE)
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
        return float(np.clip(iv, 0.01, 2.00))

    def _transaction_cost(self, row: pd.Series, trade_size: float, price_col: str = "SPY_CLOSE") -> float:
        s = float(row[price_col])
        linear_cost = LINEAR_TRANSACTION_COST_RATE * s * abs(trade_size)
        impact_cost = QUADRATIC_IMPACT_RATE * s * (trade_size ** 2)
        return float(linear_cost + impact_cost)

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
        raw_action = float(np.clip(action[0], -1.0, 1.0))
        delta = self._get_delta(row)
        desired_hedge = float(np.clip(delta + ADJUSTMENT_LIMIT * raw_action, HEDGE_MIN, HEDGE_MAX))
        if abs(desired_hedge - self.prev_hedge) < NO_TRADE_BAND:
            target_hedge = float(self.prev_hedge)
        else:
            target_hedge = desired_hedge

        d_stock = float(row["SPY_DS"])
        d_option = float(row["DOPTION"])
        trade_size = target_hedge - self.prev_hedge
        transaction_cost = self._transaction_cost(row, trade_size, price_col="SPY_CLOSE")

        is_last_step = self.t == len(self.current_data) - 1
        final_liq = 0.0
        if is_last_step:
            final_liq = self._transaction_cost(row, target_hedge, price_col="SPY_NEXT_CLOSE")
            transaction_cost += final_liq

        raw_pnl_per_share = -d_option + target_hedge * d_stock - transaction_cost
        scale = self._reward_scale(row)
        scaled_pnl = raw_pnl_per_share / scale
        downside = min(scaled_pnl, 0.0)
        iv = self._iv_decimal(row)
        dt = self._dt_years(row)
        s_t = float(row["SPY_CLOSE"])
        delta_risk = (((target_hedge - delta) * iv * s_t) ** 2) * dt / (scale ** 2)

        reward = (
            scaled_pnl
            - DOWNSIDE_PENALTY * (downside ** 2)
            - DELTA_RISK_PENALTY * delta_risk
            - EXTRA_COST_PENALTY * (transaction_cost / scale)
        )

        turnover = abs(trade_size) + (abs(target_hedge) if is_last_step else 0.0)
        self.prev_hedge = target_hedge
        self.t += 1
        terminated = self.t >= len(self.current_data)
        truncated = False
        obs = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated else self._get_obs()

        info = {
            "raw_action": raw_action,
            "delta": delta,
            "hedge": target_hedge,
            "adjustment": target_hedge - delta,
            "raw_pnl_per_share": raw_pnl_per_share,
            "transaction_cost_per_share": transaction_cost,
            "turnover": turnover,
        }
        return obs, float(reward), terminated, truncated, info


# ============================================================
# BASELINES AND EVALUATION
# ============================================================

def compute_trade_cost(row: pd.Series, trade_size: float, use_next_price: bool = False) -> float:
    s = float(row["SPY_NEXT_CLOSE"] if use_next_price else row["SPY_CLOSE"])
    return float(
        LINEAR_TRANSACTION_COST_RATE * s * abs(trade_size)
        + QUADRATIC_IMPACT_RATE * s * (trade_size ** 2)
    )


def make_obs_from_row(row, prev_hedge, feature_cols, feature_mean, feature_std):
    x = row[feature_cols].astype(float)
    x = (x - feature_mean[feature_cols]) / feature_std[feature_cols]
    x = x.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)
    x = np.clip(x, -OBS_CLIP, OBS_CLIP)
    delta = float(np.clip(float(row["OPTION_DELTA"]), HEDGE_MIN, HEDGE_MAX))
    delta_gap = prev_hedge - delta
    return np.concatenate([x, np.array([prev_hedge, delta_gap], dtype=np.float32)]).astype(np.float32)


def run_baseline(transitions_df, strategy):
    df = transitions_df.copy().sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)
    if strategy == "no_hedge":
        df["HEDGE"] = 0.0
    elif strategy == "delta":
        df["HEDGE"] = df["OPTION_DELTA"].astype(float).clip(HEDGE_MIN, HEDGE_MAX)
    else:
        raise ValueError(strategy)
    df["PREV_HEDGE"] = df.groupby("EPISODE_ID")["HEDGE"].shift(1).fillna(0.0)
    df["TRADE_SIZE"] = df["HEDGE"] - df["PREV_HEDGE"]
    df["TC"] = LINEAR_TRANSACTION_COST_RATE * df["SPY_CLOSE"] * df["TRADE_SIZE"].abs()
    df["IS_LAST"] = df.groupby("EPISODE_ID").cumcount() == df.groupby("EPISODE_ID")["EPISODE_ID"].transform("count") - 1
    df["TC_FINAL"] = 0.0
    last = df["IS_LAST"]
    df.loc[last, "TC_FINAL"] = LINEAR_TRANSACTION_COST_RATE * df.loc[last, "SPY_NEXT_CLOSE"] * df.loc[last, "HEDGE"].abs()
    df["STEP_PNL"] = (-df["DOPTION"] + df["HEDGE"] * df["SPY_DS"] - df["TC"] - df["TC_FINAL"]) * CONTRACT_MULTIPLIER
    df["TRANSACTION_COST"] = (df["TC"] + df["TC_FINAL"]) * CONTRACT_MULTIPLIER
    df["TURNOVER"] = df["TRADE_SIZE"].abs()
    df.loc[last, "TURNOVER"] += df.loc[last, "HEDGE"].abs()

    ep = df.groupby(["EPISODE_ID", "SPLIT"]).agg(
        START_DATE=("QUOTE_DATE", "first"),
        END_DATE=("NEXT_QUOTE_DATE", "last"),
        N_STEPS=("STEP_PNL", "count"),
        TERMINAL_PNL=("STEP_PNL", "sum"),
        TOTAL_TC=("TRANSACTION_COST", "sum"),
        TOTAL_TURNOVER=("TURNOVER", "sum"),
        AVG_HEDGE=("HEDGE", "mean"),
        STD_HEDGE=("HEDGE", "std"),
    ).reset_index()
    ep["ALGORITHM"] = "baseline"
    ep["STRATEGY"] = strategy
    ep["SEED"] = np.nan
    ep["TRAINING_TIME_MIN"] = np.nan
    return ep


def evaluate_model(model, real_df, algorithm, strategy_name, seed, training_time_min, feature_cols, feature_mean, feature_std):
    results = []
    steps = []
    df = real_df.copy().sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)
    for episode_id, ep in df.groupby("EPISODE_ID"):
        ep = ep.reset_index(drop=True)
        split = ep["SPLIT"].iloc[0]
        prev_hedge = 0.0
        rewards = []
        tcs = []
        turnovers = []
        hedges = []
        deltas = []
        raw_actions = []
        adjustments = []
        no_trades = []
        for t in range(len(ep)):
            row = ep.iloc[t]
            obs = make_obs_from_row(row, prev_hedge, feature_cols, feature_mean, feature_std)
            action, _ = model.predict(obs, deterministic=True)
            raw_action = float(np.clip(action[0], -1.0, 1.0))
            delta = float(np.clip(float(row["OPTION_DELTA"]), HEDGE_MIN, HEDGE_MAX))
            desired = float(np.clip(delta + ADJUSTMENT_LIMIT * raw_action, HEDGE_MIN, HEDGE_MAX))
            if abs(desired - prev_hedge) < NO_TRADE_BAND:
                hedge = prev_hedge
            else:
                hedge = desired
            trade = hedge - prev_hedge
            tc = compute_trade_cost(row, trade, use_next_price=False)
            is_last = t == len(ep) - 1
            if is_last:
                tc += compute_trade_cost(row, hedge, use_next_price=True)
                turnover = abs(trade) + abs(hedge)
            else:
                turnover = abs(trade)
            pnl = -float(row["DOPTION"]) + hedge * float(row["SPY_DS"]) - tc
            rewards.append(pnl)
            tcs.append(tc)
            turnovers.append(turnover)
            hedges.append(hedge)
            deltas.append(delta)
            raw_actions.append(raw_action)
            adjustments.append(hedge - delta)
            no_trades.append(abs(trade) < 1e-12)
            steps.append({
                "SEED": seed,
                "ALGORITHM": algorithm,
                "STRATEGY": strategy_name,
                "EPISODE_ID": episode_id,
                "SPLIT": split,
                "QUOTE_DATE": row["QUOTE_DATE"],
                "RAW_ACTION": raw_action,
                "DELTA": delta,
                "HEDGE": hedge,
                "PREV_HEDGE": prev_hedge,
                "ADJUSTMENT_FROM_DELTA": hedge - delta,
                "NO_TRADE": abs(trade) < 1e-12,
                "REWARD": pnl * CONTRACT_MULTIPLIER,
                "TRANSACTION_COST": tc * CONTRACT_MULTIPLIER,
                "TURNOVER": turnover,
            })
            prev_hedge = hedge
        results.append({
            "SEED": seed,
            "ALGORITHM": algorithm,
            "STRATEGY": strategy_name,
            "EPISODE_ID": episode_id,
            "SPLIT": split,
            "START_DATE": ep["QUOTE_DATE"].iloc[0],
            "END_DATE": ep["NEXT_QUOTE_DATE"].iloc[-1],
            "N_STEPS": len(ep),
            "TERMINAL_PNL": np.sum(rewards) * CONTRACT_MULTIPLIER,
            "TOTAL_TC": np.sum(tcs) * CONTRACT_MULTIPLIER,
            "TOTAL_TURNOVER": np.sum(turnovers),
            "AVG_RAW_ACTION": np.mean(raw_actions),
            "STD_RAW_ACTION": np.std(raw_actions),
            "AVG_DELTA": np.mean(deltas),
            "AVG_HEDGE": np.mean(hedges),
            "STD_HEDGE": np.std(hedges),
            "AVG_ADJUSTMENT_FROM_DELTA": np.mean(adjustments),
            "STD_ADJUSTMENT_FROM_DELTA": np.std(adjustments),
            "NO_TRADE_RATE": np.mean(no_trades),
            "ACTION_NEAR_NEG1_RATE": np.mean(np.array(raw_actions) < -0.95),
            "ACTION_NEAR_POS1_RATE": np.mean(np.array(raw_actions) > 0.95),
            "TRAINING_TIME_MIN": training_time_min,
        })
    return pd.DataFrame(steps), pd.DataFrame(results)


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


def make_metrics(df, group_cols):
    agg = df.groupby(group_cols).agg(
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
    ).reset_index()
    return agg.sort_values(group_cols)


def algorithm_summary(metrics_by_seed):
    return metrics_by_seed.groupby(["ALGORITHM", "SPLIT"]).agg(
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
        TRAINING_TIME_MIN=("TRAINING_TIME_MIN", "mean"),
    ).reset_index().sort_values(["SPLIT", "ALGORITHM"])


# ============================================================
# MODEL FACTORY
# ============================================================


def make_env(df, feature_cols, feature_mean, feature_std, seed):
    return Monitor(OptionHedgingEnvResidualDelta(
        transitions_df=df,
        feature_cols=feature_cols,
        feature_mean=feature_mean,
        feature_std=feature_std,
        random_episode=True,
        seed=seed,
    ))


def make_model(args, env, seed):
    algorithm = args.algorithm.upper()
    if algorithm == "PPO":
        return PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=args.ppo_lr,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=PPO_POLICY_KWARGS,
            tensorboard_log=str(OUTPUT_DIR / "tb_logs" / "sim_pretrain_finetune_v3c"),
            seed=seed,
        )
    if algorithm == "SAC":
        # SAC is made more conservative for transaction-cost hedging:
        # lower learning rate and fixed lower entropy coefficient reduce over-trading.
        return SAC(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=args.sac_lr,
            buffer_size=args.buffer_size,
            learning_starts=args.sac_learning_starts,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=1,
            gradient_steps=1,
            ent_coef=args.sac_ent_coef,
            target_update_interval=1,
            policy_kwargs=OFF_POLICY_KWARGS,
            tensorboard_log=str(OUTPUT_DIR / "tb_logs" / "sim_pretrain_finetune_v3c"),
            seed=seed,
        )
    if algorithm == "TD3":
        noise = NormalActionNoise(mean=np.zeros(1), sigma=args.td3_noise * np.ones(1))
        return TD3(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=args.td3_lr,
            buffer_size=args.buffer_size,
            learning_starts=args.td3_learning_starts,
            batch_size=256,
            gamma=0.99,
            tau=0.005,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=noise,
            policy_delay=2,
            target_policy_noise=0.2,
            target_noise_clip=0.5,
            policy_kwargs=OFF_POLICY_KWARGS,
            tensorboard_log=str(OUTPUT_DIR / "tb_logs" / "sim_pretrain_finetune_v3c"),
            seed=seed,
        )
    raise ValueError(f"Unsupported algorithm: {args.algorithm}")


def reset_replay_buffer_if_possible(model):
    if hasattr(model, "replay_buffer") and model.replay_buffer is not None:
        if hasattr(model.replay_buffer, "reset"):
            model.replay_buffer.reset()
            print("Replay buffer reset before real fine-tuning.")
        else:
            print("Replay buffer has no reset() method; keeping existing buffer.")


# ============================================================
# MAIN
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, required=True, choices=["PPO", "SAC", "TD3"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--pretrain-steps", type=int, default=200_000)
    parser.add_argument("--finetune-steps", type=int, default=100_000)
    parser.add_argument("--sim-path", type=str, default="")
    parser.add_argument("--real-path", type=str, default="")
    parser.add_argument("--clear-replay-before-finetune", action="store_true")

    # Algorithm-specific tuning.
    parser.add_argument("--ppo-lr", type=float, default=3e-4)
    parser.add_argument("--sac-lr", type=float, default=1e-4)
    parser.add_argument("--sac-ent-coef", type=str, default="0.01")
    parser.add_argument("--sac-learning-starts", type=int, default=5_000)
    parser.add_argument("--td3-lr", type=float, default=3e-4)
    parser.add_argument("--td3-learning-starts", type=int, default=5_000)
    parser.add_argument("--td3-noise", type=float, default=0.05)
    parser.add_argument("--buffer-size", type=int, default=300_000)
    return parser.parse_args()


def main():
    args = parse_args()
    algorithm = args.algorithm.upper()

    if args.real_path:
        real_path = Path(args.real_path)
    else:
        real_path = find_existing_file(REAL_TRANSITIONS_FILE, ["data/processed", "processed", ""])

    if args.sim_path:
        sim_path = Path(args.sim_path)
    else:
        sim_path = find_existing_file(SIM_TRANSITIONS_FILE, ["data/processed", "processed", ""])

    real = add_engineered_features(pd.read_parquet(real_path))
    sim = add_engineered_features(pd.read_parquet(sim_path))

    feature_cols = [c for c in BASE_FEATURE_COLS if c in real.columns and c in sim.columns]
    real_train = real[real["SPLIT"] == "train"].copy()
    if real_train.empty:
        raise ValueError("Real data has no SPLIT == 'train'.")

    # Normalize both simulated and real data using real-train statistics.
    feature_mean = real_train[feature_cols].mean()
    feature_std = real_train[feature_cols].std().replace(0, 1)

    print("Real path:", real_path)
    print("Sim path:", sim_path)
    print("Real shape:", real.shape)
    print("Sim shape:", sim.shape)
    print("Feature columns:", feature_cols)
    print("Algorithm:", algorithm)
    print("Seeds:", args.seeds)
    print("Pretrain steps:", args.pretrain_steps)
    print("Finetune steps:", args.finetune_steps)

    env_check = OptionHedgingEnvResidualDelta(sim, feature_cols, feature_mean, feature_std, seed=42)
    check_env(env_check, warn=True)
    print("Environment check passed.")

    model_dir = OUTPUT_DIR / f"sim_pretrain_finetune_v3c_{algorithm.lower()}_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    excel_path = OUTPUT_DIR / f"sim_pretrain_finetune_v3c_{algorithm.lower()}_multiseed.xlsx"
    step_path = OUTPUT_DIR / f"sim_pretrain_finetune_v3c_{algorithm.lower()}_step_results.parquet"

    all_steps = []
    all_eps = []
    metrics_by_seed_list = []
    log_rows = []

    for seed in args.seeds:
        print("\n" + "=" * 90)
        print(f"{algorithm} simulated pretraining + real fine-tuning, seed={seed}")
        print("=" * 90)

        sim_env = make_env(sim, feature_cols, feature_mean, feature_std, seed)
        model = make_model(args, sim_env, seed)
        print("Policy architecture:")
        print(model.policy)

        start = time.time()
        if args.pretrain_steps > 0:
            print(f"Pretraining on simulated data for {args.pretrain_steps:,} steps")
            model.learn(total_timesteps=args.pretrain_steps)

        if args.finetune_steps > 0:
            print(f"Fine-tuning on real train data for {args.finetune_steps:,} steps")
            real_env = make_env(real_train, feature_cols, feature_mean, feature_std, seed + 10_000)
            model.set_env(real_env)
            if args.clear_replay_before_finetune and algorithm in {"SAC", "TD3"}:
                reset_replay_buffer_if_possible(model)
            model.learn(total_timesteps=args.finetune_steps, reset_num_timesteps=False)

        elapsed_min = (time.time() - start) / 60.0
        model_path = model_dir / f"{algorithm.lower()}_sim_pretrain_finetune_v3c_seed_{seed}"
        model.save(model_path)
        print("Saved model:", model_path)
        print(f"Training time: {elapsed_min:.2f} min")

        strategy = f"{algorithm.lower()}_sim_pretrain_finetune_v3c_seed_{seed}"
        step_df, ep_df = evaluate_model(
            model=model,
            real_df=real,
            algorithm=algorithm,
            strategy_name=strategy,
            seed=seed,
            training_time_min=elapsed_min,
            feature_cols=feature_cols,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )
        seed_metrics = make_metrics(ep_df, ["ALGORITHM", "SEED", "SPLIT", "STRATEGY"])
        print(seed_metrics)

        all_steps.append(step_df)
        all_eps.append(ep_df)
        metrics_by_seed_list.append(seed_metrics)
        log_rows.append({
            "ALGORITHM": algorithm,
            "SEED": seed,
            "PRETRAIN_STEPS": args.pretrain_steps,
            "FINETUNE_STEPS": args.finetune_steps,
            "TRAINING_TIME_MIN": elapsed_min,
            "MODEL_PATH": str(model_path),
            "SAC_ENT_COEF": args.sac_ent_coef if algorithm == "SAC" else np.nan,
        })

    rl_steps = pd.concat(all_steps, ignore_index=True)
    rl_eps = pd.concat(all_eps, ignore_index=True)
    metrics_by_seed = pd.concat(metrics_by_seed_list, ignore_index=True)

    baseline = pd.concat([
        run_baseline(real, "no_hedge"),
        run_baseline(real, "delta"),
    ], ignore_index=True)

    all_episode_results = pd.concat([baseline, rl_eps], ignore_index=True, sort=False)
    comparison_metrics = make_metrics(all_episode_results, ["ALGORITHM", "SPLIT", "STRATEGY"])
    summary = algorithm_summary(metrics_by_seed)
    training_log = pd.DataFrame(log_rows)

    rl_steps.to_parquet(step_path, index=False)
    max_excel_rows = 1_000_000
    step_sample = rl_steps if len(rl_steps) <= max_excel_rows else rl_steps.sample(max_excel_rows, random_state=42)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        all_episode_results.to_excel(writer, sheet_name="Episode_Results", index=False)
        comparison_metrics.to_excel(writer, sheet_name="Metrics", index=False)
        metrics_by_seed.to_excel(writer, sheet_name="Metrics_By_Seed", index=False)
        summary.to_excel(writer, sheet_name="Algorithm_Summary", index=False)
        training_log.to_excel(writer, sheet_name="Training_Log", index=False)
        step_sample.to_excel(writer, sheet_name="Step_Results_Sample", index=False)

    print("\nSaved Excel:", excel_path)
    print("Saved full step parquet:", step_path)


if __name__ == "__main__":
    main()
