import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

TRANSITIONS_PATH = DATA_DIR / "transitions_daily_top1_final_with_spy_2010_2023.parquet"
BASELINE_PATH = OUTPUT_DIR / "baseline_episode_results_with_spy.parquet"

PPO_MODEL_PATH = OUTPUT_DIR / "ppo_50k_call_only_risk_aware_010_with_spy"
PPO_COMPARISON_PATH = OUTPUT_DIR / "comparison_baseline_ppo_risk_aware_010_50k_with_spy.xlsx"


# ============================================================
# CONFIG
# ============================================================

CONTRACT_MULTIPLIER = 100
TRANSACTION_COST_RATE = 0.0005

# Main thesis setting:
# short CALL hedge should be long SPY between 0 and 1 share per option share
HEDGE_LIMIT_RL = 1.0
TOTAL_TIMESTEPS = 50_000
RISK_PENALTY = 0.2
    
FEATURE_COLS = [
    "SPY_CLOSE",
    "OPTION_MID",
    "DTE",
    "SPY_LOG_MONEYNESS",
    "OPTION_DELTA",
    "OPTION_GAMMA",
    "OPTION_VEGA",
    "OPTION_THETA",
    "OPTION_IV",
    "OPTION_SPREAD_PCT",
]

# ============================================================
# LOAD DATA
# ============================================================

transitions = pd.read_parquet(TRANSITIONS_PATH)
baseline_results = pd.read_parquet(BASELINE_PATH)

FEATURE_COLS = [c for c in FEATURE_COLS if c in transitions.columns]

train_transitions = transitions[transitions["SPLIT"] == "train"].copy()
val_transitions = transitions[transitions["SPLIT"] == "val"].copy()
test_transitions = transitions[transitions["SPLIT"] == "test"].copy()

feature_mean = train_transitions[FEATURE_COLS].mean()
feature_std = train_transitions[FEATURE_COLS].std().replace(0, 1)

print("Loaded transitions:", transitions.shape)
print("Transitions by split:")
print(transitions["SPLIT"].value_counts())

print("\nFeature columns:")
print(FEATURE_COLS)


# ============================================================
# ENVIRONMENT
# ============================================================

class OptionHedgingEnvCallOnly(gym.Env):
    """
    Hedging-only environment for short CALL option.

    Raw action:
        a_t in [-1, 1]

    Actual hedge:
        h_t = (a_t + 1) / 2 * hedge_limit

    With hedge_limit = 1:
        raw action -1 -> hedge 0
        raw action  0 -> hedge 0.5
        raw action +1 -> hedge 1

    Reward per share:
        -(C_{t+1} - C_t) + h_t * (S_{t+1} - S_t) - transaction cost
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        transitions_df,
        feature_cols,
        feature_mean,
        feature_std,
        hedge_limit=1.0,
        transaction_cost_rate=0.0005,
        risk_penalty=0.05,
        random_episode=True,
        seed=42,
    ):
        super().__init__()
        self.df = transitions_df.copy()
        self.df = self.df.sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)

        self.feature_cols = feature_cols
        self.feature_mean = feature_mean
        self.feature_std = feature_std

        self.hedge_limit = hedge_limit
        self.transaction_cost_rate = transaction_cost_rate
        self.risk_penalty = risk_penalty
        self.random_episode = random_episode

        self.episode_ids = self.df["EPISODE_ID"].unique().tolist()
        self.episode_data = {
            eid: group.reset_index(drop=True)
            for eid, group in self.df.groupby("EPISODE_ID")
        }

        self.rng = np.random.default_rng(seed)

        obs_dim = len(self.feature_cols) + 1  # features + previous hedge

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

    def _scale_action_to_hedge(self, action):
        raw_action = float(np.clip(action[0], -1.0, 1.0))
        target_hedge = (raw_action + 1.0) / 2.0 * self.hedge_limit
        return raw_action, target_hedge

    def _get_obs(self):
        row = self.current_data.iloc[self.t]

        x = row[self.feature_cols].astype(float)
        x = (x - self.feature_mean[self.feature_cols]) / self.feature_std[self.feature_cols]
        x = x.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)

        obs = np.concatenate([
            x,
            np.array([self.prev_hedge], dtype=np.float32),
        ])

        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.random_episode:
            self.current_episode_id = self.rng.choice(self.episode_ids)
        else:
            self.current_episode_id = self.episode_ids[0]

        self.current_data = self.episode_data[self.current_episode_id]
        self.t = 0
        self.prev_hedge = 0.0

        return self._get_obs(), {}

    def step(self, action):
        raw_action, target_hedge = self._scale_action_to_hedge(action)

        row = self.current_data.iloc[self.t]

        s_t = float(row["SPY_CLOSE"])
        s_next = float(row["SPY_NEXT_CLOSE"])
        d_stock = float(row["SPY_DS"])
        d_option = float(row["DOPTION"])

        trade_size = target_hedge - self.prev_hedge

        transaction_cost = (
            self.transaction_cost_rate
            * s_t
            * abs(trade_size)
        )

        is_last_step = self.t == len(self.current_data) - 1

        # Final liquidation: close hedge h_T -> 0
        if is_last_step:
            final_liquidation_cost = (
                self.transaction_cost_rate
                * s_next
                * abs(target_hedge)
            )
            transaction_cost += final_liquidation_cost

        # Accounting PnL per share, exactly following thesis formula:
        # pnl_t = -(C_{t+1} - C_t)
        #         + a_t(S_{t+1} - S_t)
        #         - lambda * S_t * |a_t - h_t|
        raw_pnl_per_share = (
            -d_option
            + target_hedge * d_stock
            - transaction_cost
        )

        # Risk-aware training reward:
        # This keeps the PnL term, but penalizes large hedging errors.
        reward = (
            raw_pnl_per_share
            - self.risk_penalty * (raw_pnl_per_share ** 2)
)

        self.prev_hedge = target_hedge
        self.t += 1

        terminated = self.t >= len(self.current_data)
        truncated = False

        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()

        info = {
            "episode_id": self.current_episode_id,
            "raw_action": raw_action,
            "hedge": target_hedge,
            "raw_pnl_per_share": raw_pnl_per_share,
            "training_reward_per_share": reward,
            "transaction_cost_per_share": transaction_cost,
            "is_last_step": is_last_step,
        }

        return obs, float(reward), terminated, truncated, info


# ============================================================
# ENV CHECK
# ============================================================

env_check = OptionHedgingEnvCallOnly(
    transitions_df=train_transitions,
    feature_cols=FEATURE_COLS,
    feature_mean=feature_mean,
    feature_std=feature_std,
    hedge_limit=HEDGE_LIMIT_RL,
    transaction_cost_rate=TRANSACTION_COST_RATE,
    risk_penalty=RISK_PENALTY,
    random_episode=True,
    seed=42,
)

check_env(env_check, warn=True)
print("\nEnvironment check passed.")


# ============================================================
# TRAIN PPO
# ============================================================

ppo_train_env = Monitor(
    OptionHedgingEnvCallOnly(
        transitions_df=train_transitions,
        feature_cols=FEATURE_COLS,
        feature_mean=feature_mean,
        feature_std=feature_std,
        hedge_limit=HEDGE_LIMIT_RL,
        transaction_cost_rate=TRANSACTION_COST_RATE,
        risk_penalty=RISK_PENALTY,
        random_episode=True,
        seed=42,
    )
)

start_time = time.time()

ppo_model = PPO(
    policy="MlpPolicy",
    env=ppo_train_env,
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
    tensorboard_log=str(OUTPUT_DIR / "tb_logs" / "ppo_call_only"),
)

ppo_model.learn(total_timesteps=TOTAL_TIMESTEPS)

elapsed = time.time() - start_time

ppo_model.save(PPO_MODEL_PATH)

print("\nSaved PPO model:")
print(PPO_MODEL_PATH)

print(f"Training time: {elapsed / 60:.2f} minutes")


# ============================================================
# EVALUATION HELPERS
# ============================================================

def make_obs_from_row(row, prev_hedge, feature_cols, feature_mean, feature_std):
    x = row[feature_cols].astype(float)
    x = (x - feature_mean[feature_cols]) / feature_std[feature_cols]
    x = x.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)

    obs = np.concatenate([
        x,
        np.array([prev_hedge], dtype=np.float32),
    ])

    return obs.astype(np.float32)


def evaluate_rl_model_call_only(
    model,
    transitions_df,
    strategy_name,
    deterministic=True,
    hedge_limit=1.0,
    transaction_cost_rate=0.0005,
    contract_multiplier=100,
):
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
        raw_actions = []
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
            target_hedge = (raw_action + 1.0) / 2.0 * hedge_limit

            s_t = float(row["SPY_CLOSE"])
            s_next = float(row["SPY_NEXT_CLOSE"])
            d_stock = float(row["SPY_DS"])
            d_option = float(row["DOPTION"])

            trade_size = target_hedge - prev_hedge

            transaction_cost = (
                transaction_cost_rate
                * s_t
                * abs(trade_size)
            )

            is_last_step = t == len(ep) - 1

            if is_last_step:
                final_liquidation_cost = (
                    transaction_cost_rate
                    * s_next
                    * abs(target_hedge)
                )
                transaction_cost += final_liquidation_cost
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
            raw_actions.append(raw_action)
            turnovers.append(turnover)

            step_rows.append({
                "EPISODE_ID": episode_id,
                "SPLIT": split,
                "QUOTE_DATE": row["QUOTE_DATE"],
                "STRATEGY": strategy_name,
                "RAW_ACTION": raw_action,
                "HEDGE": target_hedge,
                "PREV_HEDGE": prev_hedge,
                "REWARD_PER_SHARE": reward_per_share,
                "REWARD": reward_per_share * contract_multiplier,
                "TRANSACTION_COST": transaction_cost * contract_multiplier,
                "TURNOVER": turnover,
            })

            prev_hedge = target_hedge

        results.append({
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
            "AVG_HEDGE": np.mean(hedges),
            "STD_HEDGE": np.std(hedges),
            "STRATEGY": strategy_name,
        })

    return pd.DataFrame(step_rows), pd.DataFrame(results)


# ============================================================
# EVALUATE PPO
# ============================================================

ppo_steps, ppo_ep = evaluate_rl_model_call_only(
    model=ppo_model,
    transitions_df=transitions,
    strategy_name="ppo_50k_risk_aware_010",
    deterministic=True,
    hedge_limit=HEDGE_LIMIT_RL,
    transaction_cost_rate=TRANSACTION_COST_RATE,
    contract_multiplier=CONTRACT_MULTIPLIER,
)

print("\nPPO results by split:")
print(
    ppo_ep
    .groupby(["SPLIT", "STRATEGY"])
    .agg(
        EPISODES=("EPISODE_ID", "nunique"),
        MEAN_PNL=("TERMINAL_PNL", "mean"),
        STD_PNL=("TERMINAL_PNL", "std"),
        MEDIAN_PNL=("TERMINAL_PNL", "median"),
        MEAN_TC=("TOTAL_TC", "mean"),
        MEAN_TURNOVER=("TOTAL_TURNOVER", "mean"),
        AVG_HEDGE=("AVG_HEDGE", "mean"),
    )
    .reset_index()
)


# ============================================================
# COMPARE BASELINE VS PPO
# ============================================================

def cvar_95(x):
    x = pd.Series(x).dropna()
    q = x.quantile(0.05)
    return x[x <= q].mean()


def sharpe_like(x):
    x = pd.Series(x).dropna()
    std = x.std()
    if std == 0 or pd.isna(std):
        return np.nan
    return x.mean() / std


all_results = pd.concat(
    [
        baseline_results,
        ppo_ep,
    ],
    ignore_index=True,
)

comparison_metrics = (
    all_results
    .groupby(["SPLIT", "STRATEGY"])
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
    )
    .reset_index()
    .sort_values(["SPLIT", "STRATEGY"])
)

print("\nComparison metrics:")
print(comparison_metrics)

with pd.ExcelWriter(PPO_COMPARISON_PATH, engine="openpyxl") as writer:
    all_results.to_excel(writer, sheet_name="Episode_Results", index=False)
    comparison_metrics.to_excel(writer, sheet_name="Metrics", index=False)
    ppo_steps.to_excel(writer, sheet_name="PPO_Step_Results", index=False)

print("\nSaved comparison:")
print(PPO_COMPARISON_PATH)