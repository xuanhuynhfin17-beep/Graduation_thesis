"""
08c_pretrain_finetune_regime_switching_pilot.py

Pilot pretraining transfer experiment:

E0: PPO V3C trained directly on real data, no pretraining.
E1: PPO V3C pretrained on single-regime BS simulated data, then fine-tuned on real data.
E2: PPO V3C pretrained on HMM Markov-switching GBM simulated data, then fine-tuned on real data.
E3: PPO V3C pretrained on HMM Markov-switching GBM simulated data with observable regime proxy state,
    then fine-tuned on real data with the same proxy features.

Inputs:
    data/processed/transitions_daily_top1_final_with_spy_2010_2023.parquet
    data/processed/transitions_daily_top1_final_with_spy_2010_2023_with_regime_proxies.parquet
    data/processed/sim_bs_pretrain_pilot.parquet
    data/processed/sim_ms_gbm_pretrain_pilot.parquet
    data/processed/sim_ms_gbm_proxy_pretrain_pilot.parquet

Output:
    outputs/pretraining_regime_switching_pilot_ppo.xlsx
    outputs/pretraining_regime_switching_pilot_ppo_step_results.parquet

Run:
    py src\\08c_pretrain_finetune_regime_switching_pilot.py
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
# PATHS
# ============================================================

REAL_FILE = "transitions_daily_top1_final_with_spy_2010_2023.parquet"
REAL_PROXY_FILE = "transitions_daily_top1_final_with_spy_2010_2023_with_regime_proxies.parquet"
SIM_BS_FILE = "sim_bs_pretrain_pilot.parquet"
SIM_MS_FILE = "sim_ms_gbm_pretrain_pilot.parquet"
SIM_MS_PROXY_FILE = "sim_ms_gbm_proxy_pretrain_pilot.parquet"


def _candidate_project_dirs() -> list[Path]:
    here = Path(__file__).resolve()
    candidates = [here.parent, here.parent.parent, Path.cwd(), Path.cwd().parent, Path("/mnt/data")]
    out = []
    for p in candidates:
        p = p.resolve()
        if p not in out:
            out.append(p)
    return out


def find_existing_file(filename: str, relative_dirs: Iterable[str]) -> Path:
    checked = []
    for base in _candidate_project_dirs():
        for rel in relative_dirs:
            p = base / rel / filename if rel else base / filename
            checked.append(p)
            if p.exists():
                return p
    raise FileNotFoundError("Could not find " + filename + ". Checked:\n" + "\n".join(map(str, checked)))


REAL_PATH = find_existing_file(REAL_FILE, ["data/processed", "processed", ""])
REAL_PROXY_PATH = find_existing_file(REAL_PROXY_FILE, ["data/processed", "processed", ""])
SIM_BS_PATH = find_existing_file(SIM_BS_FILE, ["data/processed", "processed", ""])
SIM_MS_PATH = find_existing_file(SIM_MS_FILE, ["data/processed", "processed", ""])
SIM_MS_PROXY_PATH = find_existing_file(SIM_MS_PROXY_FILE, ["data/processed", "processed", ""])

if (Path(__file__).resolve().parent.parent / "data" / "processed").exists():
    PROJECT_DIR = Path(__file__).resolve().parent.parent
elif (Path.cwd() / "data" / "processed").exists():
    PROJECT_DIR = Path.cwd()
else:
    PROJECT_DIR = Path("/mnt/data")

OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = OUTPUT_DIR / "pretraining_regime_switching_pilot_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXCEL_OUTPUT = OUTPUT_DIR / "pretraining_regime_switching_pilot_ppo.xlsx"
STEP_OUTPUT = OUTPUT_DIR / "pretraining_regime_switching_pilot_ppo_step_results.parquet"


# ============================================================
# CONFIG
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

SEEDS = [1,2,3]
PRETRAIN_TIMESTEPS = 150_000
FINETUNE_TIMESTEPS = 150_000

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

PROXY_FEATURE_COLS = [
    "ROLLING_RV_10D",
    "ROLLING_RV_20D",
    "ROLLING_ABS_RET_10D",
    "IV_LEVEL",
    "IV_PERCENTILE",
    "SPREAD_PROXY",
    "SPREAD_PERCENTILE",
]

PPO_POLICY_KWARGS = dict(
    net_arch=dict(pi=[256, 256], vf=[256, 256]),
    activation_fn=th.nn.Tanh,
)


# ============================================================
# DATA HELPERS
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)

    if "OPTION_MID" in df.columns and "SPY_CLOSE" in df.columns:
        df["OPTION_MID_OVER_SPY"] = df["OPTION_MID"].astype(float) / df["SPY_CLOSE"].astype(float)

    if "SPY_RET_LAG1" not in df.columns and "SPY_CLOSE" in df.columns:
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

    # Make IV decimal-like if needed.
    if "OPTION_IV" in df.columns:
        df["OPTION_IV"] = pd.to_numeric(df["OPTION_IV"], errors="coerce")
        mask = df["OPTION_IV"] > 3.0
        df.loc[mask, "OPTION_IV"] = df.loc[mask, "OPTION_IV"] / 100.0

    for c in BASE_FEATURE_COLS + PROXY_FEATURE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    return df


real = add_engineered_features(pd.read_parquet(REAL_PATH))
real_proxy = add_engineered_features(pd.read_parquet(REAL_PROXY_PATH))
sim_bs = add_engineered_features(pd.read_parquet(SIM_BS_PATH))
sim_ms = add_engineered_features(pd.read_parquet(SIM_MS_PATH))
sim_ms_proxy = add_engineered_features(pd.read_parquet(SIM_MS_PROXY_PATH))

print("Loaded datasets:")
print("real", real.shape, REAL_PATH)
print("real_proxy", real_proxy.shape, REAL_PROXY_PATH)
print("sim_bs", sim_bs.shape, SIM_BS_PATH)
print("sim_ms", sim_ms.shape, SIM_MS_PATH)
print("sim_ms_proxy", sim_ms_proxy.shape, SIM_MS_PROXY_PATH)


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
        seed: int = 42,
        random_episode: bool = True,
    ):
        super().__init__()

        self.df = transitions_df.copy().sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)
        self.feature_cols = feature_cols
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.random_episode = random_episode

        self.episode_ids = self.df["EPISODE_ID"].unique().tolist()
        self.episode_data = {eid: g.reset_index(drop=True) for eid, g in self.df.groupby("EPISODE_ID")}
        self.rng = np.random.default_rng(seed)

        obs_dim = len(self.feature_cols) + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=np.array([-1.0], dtype=np.float32), high=np.array([1.0], dtype=np.float32), dtype=np.float32)

        self.current_episode_id = None
        self.current_data = None
        self.t = 0
        self.prev_hedge = 0.0

    def _get_delta(self, row: pd.Series) -> float:
        return float(np.clip(float(row["OPTION_DELTA"]), HEDGE_MIN, HEDGE_MAX))

    def _action_to_hedge(self, action: np.ndarray, row: pd.Series):
        raw_action = float(np.clip(action[0], -1.0, 1.0))
        delta = self._get_delta(row)
        desired = float(np.clip(delta + ADJUSTMENT_LIMIT * raw_action, HEDGE_MIN, HEDGE_MAX))
        if abs(desired - self.prev_hedge) < NO_TRADE_BAND:
            hedge = float(self.prev_hedge)
        else:
            hedge = desired
        return raw_action, desired, hedge, hedge - delta

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
        return iv

    def _transaction_cost(self, row: pd.Series, trade_size: float, price_col: str = "SPY_CLOSE") -> float:
        s = float(row[price_col])
        linear = LINEAR_TRANSACTION_COST_RATE * s * abs(trade_size)
        impact = QUADRATIC_IMPACT_RATE * s * (trade_size ** 2)
        return float(linear + impact)

    def _get_obs(self):
        row = self.current_data.iloc[self.t]
        x = row[self.feature_cols].astype(float)
        x = (x - self.feature_mean[self.feature_cols]) / self.feature_std[self.feature_cols]
        x = x.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)
        delta = self._get_delta(row)
        delta_gap = self.prev_hedge - delta
        return np.concatenate([x, np.array([self.prev_hedge, delta_gap], dtype=np.float32)]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.current_episode_id = self.rng.choice(self.episode_ids) if self.random_episode else self.episode_ids[0]
        self.current_data = self.episode_data[self.current_episode_id]
        self.t = 0
        self.prev_hedge = 0.0
        return self._get_obs(), {}

    def step(self, action):
        row = self.current_data.iloc[self.t]

        raw_action, desired, hedge, adjustment = self._action_to_hedge(action, row)
        delta = self._get_delta(row)

        d_stock = float(row["SPY_DS"])
        d_option = float(row["DOPTION"])
        trade_size = hedge - self.prev_hedge

        tc = self._transaction_cost(row, trade_size, "SPY_CLOSE")
        is_last = self.t == len(self.current_data) - 1
        final_tc = 0.0
        if is_last:
            final_tc = self._transaction_cost(row, hedge, "SPY_NEXT_CLOSE")
            tc += final_tc

        raw_pnl = -d_option + hedge * d_stock - tc
        scale = self._reward_scale(row)
        scaled_pnl = raw_pnl / scale
        scaled_tc = tc / scale
        downside = min(scaled_pnl, 0.0)

        iv = self._iv_decimal(row)
        dt = self._dt_years(row)
        s_t = float(row["SPY_CLOSE"])
        delta_risk = (((hedge - delta) * iv * s_t) ** 2) * dt / (scale ** 2)

        reward = (
            scaled_pnl
            - DOWNSIDE_PENALTY * (downside ** 2)
            - DELTA_RISK_PENALTY * delta_risk
            - EXTRA_COST_PENALTY * scaled_tc
        )

        self.prev_hedge = hedge
        self.t += 1

        terminated = self.t >= len(self.current_data)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated else self._get_obs()
        turnover = abs(trade_size) + (abs(hedge) if is_last else 0.0)

        info = {
            "episode_id": self.current_episode_id,
            "raw_action": raw_action,
            "delta": delta,
            "desired_hedge": desired,
            "hedge": hedge,
            "adjustment": adjustment,
            "raw_pnl_per_share": raw_pnl,
            "transaction_cost_per_share": tc,
            "final_liquidation_cost_per_share": final_tc,
            "turnover": turnover,
            "no_trade": abs(trade_size) < 1e-12,
        }

        return obs, float(reward), terminated, False, info


# ============================================================
# EVALUATION
# ============================================================

def make_obs_from_row(row, prev_hedge, feature_cols, feature_mean, feature_std):
    x = row[feature_cols].astype(float)
    x = (x - feature_mean[feature_cols]) / feature_std[feature_cols]
    x = x.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)
    delta = float(np.clip(float(row["OPTION_DELTA"]), HEDGE_MIN, HEDGE_MAX))
    delta_gap = prev_hedge - delta
    return np.concatenate([x, np.array([prev_hedge, delta_gap], dtype=np.float32)]).astype(np.float32)


def compute_trade_cost(row, trade_size, use_next_price=False):
    s = float(row["SPY_NEXT_CLOSE"] if use_next_price else row["SPY_CLOSE"])
    return float(LINEAR_TRANSACTION_COST_RATE * s * abs(trade_size) + QUADRATIC_IMPACT_RATE * s * (trade_size ** 2))


def evaluate_model(model, real_df, experiment, seed, training_time_min, feature_cols, feature_mean, feature_std):
    step_rows = []
    ep_rows = []
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
        actions = []
        adjustments = []
        no_trades = []

        for t in range(len(ep)):
            row = ep.iloc[t]
            obs = make_obs_from_row(row, prev_hedge, feature_cols, feature_mean, feature_std)
            action, _ = model.predict(obs, deterministic=True)
            raw_action = float(np.clip(action[0], -1.0, 1.0))

            delta = float(np.clip(float(row["OPTION_DELTA"]), HEDGE_MIN, HEDGE_MAX))
            desired = float(np.clip(delta + ADJUSTMENT_LIMIT * raw_action, HEDGE_MIN, HEDGE_MAX))
            hedge = prev_hedge if abs(desired - prev_hedge) < NO_TRADE_BAND else desired
            adjustment = hedge - delta
            trade_size = hedge - prev_hedge
            no_trade = abs(trade_size) < 1e-12

            tc = compute_trade_cost(row, trade_size, use_next_price=False)
            is_last = t == len(ep) - 1
            turnover = abs(trade_size)
            if is_last:
                tc += compute_trade_cost(row, hedge, use_next_price=True)
                turnover += abs(hedge)

            reward_per_share = -float(row["DOPTION"]) + hedge * float(row["SPY_DS"]) - tc

            rewards.append(reward_per_share)
            tcs.append(tc)
            turnovers.append(turnover)
            hedges.append(hedge)
            deltas.append(delta)
            actions.append(raw_action)
            adjustments.append(adjustment)
            no_trades.append(no_trade)

            step_rows.append({
                "EXPERIMENT": experiment,
                "SEED": seed,
                "EPISODE_ID": episode_id,
                "SPLIT": split,
                "QUOTE_DATE": row["QUOTE_DATE"],
                "RAW_ACTION": raw_action,
                "DELTA": delta,
                "DESIRED_HEDGE": desired,
                "HEDGE": hedge,
                "PREV_HEDGE": prev_hedge,
                "ADJUSTMENT_FROM_DELTA": adjustment,
                "NO_TRADE": no_trade,
                "REWARD_PER_SHARE": reward_per_share,
                "REWARD": reward_per_share * CONTRACT_MULTIPLIER,
                "TRANSACTION_COST": tc * CONTRACT_MULTIPLIER,
                "TURNOVER": turnover,
            })

            prev_hedge = hedge

        ep_rows.append({
            "EXPERIMENT": experiment,
            "SEED": seed,
            "ALGORITHM": "PPO",
            "EPISODE_ID": episode_id,
            "SPLIT": split,
            "START_DATE": ep["QUOTE_DATE"].iloc[0],
            "END_DATE": ep["NEXT_QUOTE_DATE"].iloc[-1],
            "N_STEPS": len(ep),
            "TERMINAL_PNL": np.sum(rewards) * CONTRACT_MULTIPLIER,
            "TOTAL_TC": np.sum(tcs) * CONTRACT_MULTIPLIER,
            "TOTAL_TURNOVER": np.sum(turnovers),
            "AVG_RAW_ACTION": np.mean(actions),
            "STD_RAW_ACTION": np.std(actions),
            "AVG_DELTA": np.mean(deltas),
            "AVG_HEDGE": np.mean(hedges),
            "STD_HEDGE": np.std(hedges),
            "AVG_ADJUSTMENT_FROM_DELTA": np.mean(adjustments),
            "STD_ADJUSTMENT_FROM_DELTA": np.std(adjustments),
            "NO_TRADE_RATE": np.mean(no_trades),
            "ACTION_NEAR_NEG1_RATE": np.mean(np.array(actions) < -0.95),
            "ACTION_NEAR_POS1_RATE": np.mean(np.array(actions) > 0.95),
            "STRATEGY": f"ppo_{experiment}_seed_{seed}",
            "TRAINING_TIME_MIN": training_time_min,
        })

    return pd.DataFrame(step_rows), pd.DataFrame(ep_rows)


def run_baseline(real_df, strategy):
    df = real_df.copy().sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)
    if strategy == "no_hedge":
        df["HEDGE"] = 0.0
    elif strategy == "delta":
        df["HEDGE"] = df["OPTION_DELTA"].astype(float).clip(HEDGE_MIN, HEDGE_MAX)
    else:
        raise ValueError(strategy)

    df["PREV_HEDGE"] = df.groupby("EPISODE_ID")["HEDGE"].shift(1).fillna(0.0)
    df["TRADE_SIZE"] = df["HEDGE"] - df["PREV_HEDGE"]
    df["TC"] = LINEAR_TRANSACTION_COST_RATE * df["SPY_CLOSE"] * df["TRADE_SIZE"].abs()

    last = df.groupby("EPISODE_ID").cumcount() == df.groupby("EPISODE_ID")["EPISODE_ID"].transform("count") - 1
    df["FINAL_TC"] = 0.0
    df.loc[last, "FINAL_TC"] = LINEAR_TRANSACTION_COST_RATE * df.loc[last, "SPY_NEXT_CLOSE"] * df.loc[last, "HEDGE"].abs()

    df["STEP_PNL"] = (-df["DOPTION"] + df["HEDGE"] * df["SPY_DS"] - df["TC"] - df["FINAL_TC"]) * CONTRACT_MULTIPLIER
    df["TRANSACTION_COST"] = (df["TC"] + df["FINAL_TC"]) * CONTRACT_MULTIPLIER
    df["TURNOVER"] = df["TRADE_SIZE"].abs()
    df.loc[last, "TURNOVER"] += df.loc[last, "HEDGE"].abs()

    out = (
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
    out["EXPERIMENT"] = "baseline"
    out["ALGORITHM"] = "baseline"
    out["SEED"] = np.nan
    out["STRATEGY"] = strategy
    out["TRAINING_TIME_MIN"] = np.nan
    return out


def cvar_95(x):
    x = pd.Series(x).dropna()
    if x.empty:
        return np.nan
    q = x.quantile(0.05)
    return x[x <= q].mean()


def sharpe_like(x):
    x = pd.Series(x).dropna()
    s = x.std()
    if pd.isna(s) or s == 0:
        return np.nan
    return x.mean() / s


def make_metrics(df, group_cols):
    d = df.copy()
    for c in [
        "TOTAL_TC", "TOTAL_TURNOVER", "AVG_HEDGE", "AVG_DELTA", "AVG_ADJUSTMENT_FROM_DELTA",
        "NO_TRADE_RATE", "ACTION_NEAR_NEG1_RATE", "ACTION_NEAR_POS1_RATE", "TRAINING_TIME_MIN"
    ]:
        if c not in d.columns:
            d[c] = np.nan

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
            AVG_ADJUSTMENT_FROM_DELTA=("AVG_ADJUSTMENT_FROM_DELTA", "mean"),
            NO_TRADE_RATE=("NO_TRADE_RATE", "mean"),
            ACTION_NEAR_NEG1_RATE=("ACTION_NEAR_NEG1_RATE", "mean"),
            ACTION_NEAR_POS1_RATE=("ACTION_NEAR_POS1_RATE", "mean"),
            TRAINING_TIME_MIN=("TRAINING_TIME_MIN", "mean"),
        )
        .reset_index()
        .sort_values(group_cols)
    )


def make_env(df, feature_cols, feature_mean, feature_std, seed):
    return Monitor(OptionHedgingEnvResidualDelta(
        transitions_df=df,
        feature_cols=feature_cols,
        feature_mean=feature_mean,
        feature_std=feature_std,
        seed=seed,
    ))


def make_model(env, seed):
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
        tensorboard_log=str(OUTPUT_DIR / "tb_logs" / "pretraining_regime_switching_pilot"),
        seed=seed,
    )


# ============================================================
# EXPERIMENTS
# ============================================================

def build_feature_cols(use_proxy: bool, df: pd.DataFrame):
    cols = [c for c in BASE_FEATURE_COLS if c in df.columns]
    if use_proxy:
        cols += [c for c in PROXY_FEATURE_COLS if c in df.columns]
    return cols


experiments = [
    {
        "EXPERIMENT": "E0_no_pretrain",
        "DESCRIPTION": "No pretraining; PPO V3C trained directly on real train data for 50k steps.",
        "SIM_DF": None,
        "REAL_DF": real,
        "USE_PROXY": False,
    },
    {
        "EXPERIMENT": "E1_bs_pretrain",
        "DESCRIPTION": "Single-regime BS/GBM pretraining, then real fine-tuning.",
        "SIM_DF": sim_bs,
        "REAL_DF": real,
        "USE_PROXY": False,
    },
    {
        "EXPERIMENT": "E2_ms_gbm_pretrain",
        "DESCRIPTION": "HMM Markov-switching GBM pretraining without regime state, then real fine-tuning.",
        "SIM_DF": sim_ms,
        "REAL_DF": real,
        "USE_PROXY": False,
    },
    {
        "EXPERIMENT": "E3_ms_gbm_proxy_pretrain",
        "DESCRIPTION": "HMM Markov-switching GBM pretraining with observable regime proxy state, then real fine-tuning.",
        "SIM_DF": sim_ms_proxy,
        "REAL_DF": real_proxy,
        "USE_PROXY": True,
    },
]

all_steps = []
all_episodes = []
all_seed_metrics = []
training_logs = []
experiment_config = pd.DataFrame([{k: v for k, v in e.items() if k not in ["SIM_DF", "REAL_DF"]} for e in experiments])

# Env check for base experiment.
base_cols = build_feature_cols(False, real)
base_train = real[real["SPLIT"].eq("train")].copy()
base_mean = base_train[base_cols].mean()
base_std = base_train[base_cols].std().replace(0, 1)
check_env(OptionHedgingEnvResidualDelta(base_train, base_cols, base_mean, base_std), warn=True)
print("Environment check passed.")

for exp in experiments:
    exp_name = exp["EXPERIMENT"]
    use_proxy = bool(exp["USE_PROXY"])
    real_df = exp["REAL_DF"].copy()
    real_train = real_df[real_df["SPLIT"].eq("train")].copy()
    feature_cols = build_feature_cols(use_proxy, real_df)

    feature_mean = real_train[feature_cols].mean()
    feature_std = real_train[feature_cols].std().replace(0, 1)

    for seed in SEEDS:
        print("\n" + "=" * 100)
        print(f"Running {exp_name}, seed={seed}, features={len(feature_cols)}")
        print("=" * 100)

        start = time.time()

        if exp["SIM_DF"] is None:
            train_env = make_env(real_train, feature_cols, feature_mean, feature_std, seed)
            model = make_model(train_env, seed)
            model.learn(total_timesteps=FINETUNE_TIMESTEPS)
            pretrain_steps = 0
        else:
            sim_train = exp["SIM_DF"].copy()
            pretrain_env = make_env(sim_train, feature_cols, feature_mean, feature_std, seed)
            model = make_model(pretrain_env, seed)
            model.learn(total_timesteps=PRETRAIN_TIMESTEPS)

            real_env = make_env(real_train, feature_cols, feature_mean, feature_std, seed)
            model.set_env(real_env)
            model.learn(total_timesteps=FINETUNE_TIMESTEPS, reset_num_timesteps=False)
            pretrain_steps = PRETRAIN_TIMESTEPS

        elapsed_min = (time.time() - start) / 60.0

        model_path = MODEL_DIR / exp_name / f"ppo_{exp_name}_seed_{seed}"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(model_path)

        step_df, ep_df = evaluate_model(
            model=model,
            real_df=real_df,
            experiment=exp_name,
            seed=seed,
            training_time_min=elapsed_min,
            feature_cols=feature_cols,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

        seed_metrics = make_metrics(ep_df, ["EXPERIMENT", "ALGORITHM", "SEED", "SPLIT", "STRATEGY"])

        all_steps.append(step_df)
        all_episodes.append(ep_df)
        all_seed_metrics.append(seed_metrics)

        training_logs.append({
            "EXPERIMENT": exp_name,
            "SEED": seed,
            "PRETRAIN_STEPS": pretrain_steps,
            "FINETUNE_STEPS": FINETUNE_TIMESTEPS,
            "FEATURES": ", ".join(feature_cols),
            "N_FEATURES": len(feature_cols),
            "TRAINING_TIME_MIN": elapsed_min,
            "MODEL_PATH": str(model_path),
        })

        print(seed_metrics)


rl_steps = pd.concat(all_steps, ignore_index=True)
rl_episodes = pd.concat(all_episodes, ignore_index=True)
metrics_by_seed = pd.concat(all_seed_metrics, ignore_index=True)

baseline = pd.concat([
    run_baseline(real, "no_hedge"),
    run_baseline(real, "delta"),
], ignore_index=True)

all_episode_results = pd.concat([baseline, rl_episodes], ignore_index=True, sort=False)
metrics = make_metrics(all_episode_results, ["EXPERIMENT", "ALGORITHM", "SPLIT", "STRATEGY"])

experiment_summary = (
    metrics_by_seed
    .groupby(["EXPERIMENT", "SPLIT"], dropna=False)
    .agg(
        N_SEEDS=("SEED", "nunique"),
        MEAN_OF_MEAN_PNL=("MEAN_PNL", "mean"),
        STD_OF_MEAN_PNL=("MEAN_PNL", "std"),
        MEAN_OF_CVAR_95=("CVAR_95", "mean"),
        STD_OF_CVAR_95=("CVAR_95", "std"),
        MEAN_OF_SHARPE_LIKE=("SHARPE_LIKE", "mean"),
        MEAN_TC=("MEAN_TC", "mean"),
        MEAN_TURNOVER=("MEAN_TURNOVER", "mean"),
        AVG_HEDGE=("AVG_HEDGE", "mean"),
        NO_TRADE_RATE=("NO_TRADE_RATE", "mean"),
        TRAINING_TIME_MIN=("TRAINING_TIME_MIN", "mean"),
    )
    .reset_index()
)

training_log = pd.DataFrame(training_logs)

rl_steps.to_parquet(STEP_OUTPUT, index=False)

if len(rl_steps) <= 1_000_000:
    step_sample = rl_steps
else:
    step_sample = rl_steps.sample(1_000_000, random_state=42).sort_values(["EXPERIMENT", "SEED", "EPISODE_ID", "QUOTE_DATE"])

with pd.ExcelWriter(EXCEL_OUTPUT, engine="openpyxl") as writer:
    experiment_config.to_excel(writer, sheet_name="Experiment_Config", index=False)
    all_episode_results.to_excel(writer, sheet_name="Episode_Results", index=False)
    metrics.to_excel(writer, sheet_name="Metrics", index=False)
    metrics_by_seed.to_excel(writer, sheet_name="Metrics_By_Seed", index=False)
    experiment_summary.to_excel(writer, sheet_name="Experiment_Summary", index=False)
    training_log.to_excel(writer, sheet_name="Training_Log", index=False)
    step_sample.to_excel(writer, sheet_name="Step_Results_Sample", index=False)

print("Saved:")
print(EXCEL_OUTPUT)
print(STEP_OUTPUT)
