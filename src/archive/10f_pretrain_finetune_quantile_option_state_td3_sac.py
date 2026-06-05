"""
10f_pretrain_finetune_quantile_option_state_td3_sac.py

Run E4 quantile option-state pretraining extension for TD3 and SAC.

Purpose
-------
This script tests whether the quantile option-state simulator changes the algorithm ranking.
It keeps the simulator fixed and applies the same pretrain/fine-tune protocol to TD3 and SAC.

Recommended final run:
    py src\10f_pretrain_finetune_quantile_option_state_td3_sac.py --mode final --sim data\simulated\ms_gbm_quantile_option_state_v2_final_n5000.parquet --pretrain_steps 50000 --finetune_steps 100000 --seeds 1 2 3 --algorithms TD3 SAC

Recommended pilot run:
    py src\10f_pretrain_finetune_quantile_option_state_td3_sac.py --mode pilot --sim data\simulated\ms_gbm_quantile_option_state_v2_pilot.parquet --pretrain_steps 50000 --finetune_steps 50000 --seeds 1 --algorithms TD3 SAC

Outputs
-------
    outputs/pretraining_quantile_option_state_td3_sac_results_<mode>.xlsx
    outputs/pretraining_quantile_option_state_td3_sac_step_results_<mode>.parquet

Important design choice
-----------------------
For off-policy algorithms, the replay buffer is reset before real-data fine-tuning by cloning
the pretrained network parameters into a new model with the real environment. This avoids mixing
simulated transitions with real transitions in the fine-tuning replay buffer.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch as th
from gymnasium import spaces

from stable_baselines3 import SAC, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise


# ============================================================
# Paths and constants
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "processed"
SIM_DIR = PROJECT_DIR / "data" / "simulated"
OUT_DIR = PROJECT_DIR / "outputs"
MODEL_DIR = OUT_DIR / "pretraining_quantile_option_state_td3_sac_models"

OUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

REAL_FILE = DATA_DIR / "transitions_daily_top1_final_with_spy_2010_2023.parquet"
SIM_FINAL_DEFAULT = SIM_DIR / "ms_gbm_quantile_option_state_v2_final_n5000.parquet"
SIM_PILOT_DEFAULT = SIM_DIR / "ms_gbm_quantile_option_state_v2_pilot.parquet"

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

FEATURE_COLS = [
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

OFF_POLICY_KWARGS = dict(
    net_arch=[256, 256],
    activation_fn=th.nn.ReLU,
)


# ============================================================
# Feature engineering
# ============================================================

def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)

    if "SPY_MONEYNESS" not in out.columns:
        out["SPY_MONEYNESS"] = out["SPY_CLOSE"].astype(float) / out["STRIKE"].astype(float)
    if "SPY_LOG_MONEYNESS" not in out.columns:
        out["SPY_LOG_MONEYNESS"] = np.log(out["SPY_MONEYNESS"].astype(float))
    if "OPTION_MID_OVER_SPY" not in out.columns:
        out["OPTION_MID_OVER_SPY"] = out["OPTION_MID"].astype(float) / out["SPY_CLOSE"].astype(float)
    if "SPY_RET_LAG1" not in out.columns:
        out["SPY_RET_LAG1"] = out.groupby("EPISODE_ID")["SPY_CLOSE"].pct_change().fillna(0.0)

    if "OPTION_SPREAD_PCT" not in out.columns:
        if "SPREAD_PROXY" in out.columns:
            out["OPTION_SPREAD_PCT"] = out["SPREAD_PROXY"]
        else:
            out["OPTION_SPREAD_PCT"] = 0.01

    out["OPTION_IV"] = pd.to_numeric(out["OPTION_IV"], errors="coerce")
    out.loc[out["OPTION_IV"] > 3.0, "OPTION_IV"] = out.loc[out["OPTION_IV"] > 3.0, "OPTION_IV"] / 100.0

    required = FEATURE_COLS + [
        "SPY_DS", "DOPTION", "SPY_CLOSE", "SPY_NEXT_CLOSE", "OPTION_MID",
        "OPTION_DELTA", "EPISODE_ID", "SPLIT"
    ]
    for c in required:
        if c in out.columns and c not in ["EPISODE_ID", "SPLIT"]:
            out[c] = pd.to_numeric(out[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    numeric_required = [c for c in FEATURE_COLS if c in out.columns] + [
        "SPY_DS", "DOPTION", "SPY_CLOSE", "SPY_NEXT_CLOSE", "OPTION_MID"
    ]
    out = out.dropna(subset=numeric_required).copy()
    return out


# ============================================================
# Environment
# ============================================================

class OptionHedgingEnvResidualDelta(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, transitions_df, feature_cols, feature_mean, feature_std, seed=42, random_episode=True):
        super().__init__()

        self.df = transitions_df.copy().sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)
        self.feature_cols = feature_cols
        self.feature_mean = feature_mean
        self.feature_std = feature_std
        self.random_episode = random_episode

        self.episode_ids = self.df["EPISODE_ID"].unique().tolist()
        self.episode_data = {eid: g.reset_index(drop=True) for eid, g in self.df.groupby("EPISODE_ID")}
        self.rng = np.random.default_rng(seed)

        obs_dim = len(feature_cols) + 2  # normalized features + prev hedge + delta gap
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.current_data = None
        self.current_episode_id = None
        self.t = 0
        self.prev_hedge = 0.0

    def _delta(self, row):
        return float(np.clip(float(row["OPTION_DELTA"]), HEDGE_MIN, HEDGE_MAX))

    def _action_to_hedge(self, action, row):
        raw_action = float(np.clip(action[0], -1.0, 1.0))
        delta = self._delta(row)
        desired = float(np.clip(delta + ADJUSTMENT_LIMIT * raw_action, HEDGE_MIN, HEDGE_MAX))
        hedge = self.prev_hedge if abs(desired - self.prev_hedge) < NO_TRADE_BAND else desired
        return raw_action, desired, hedge, hedge - delta

    def _scale(self, row):
        if REWARD_SCALE_MODE == "option_mid":
            return max(abs(float(row["OPTION_MID"])), MIN_REWARD_SCALE)
        return 1.0

    def _tc(self, row, trade_size, price_col="SPY_CLOSE"):
        s = float(row[price_col])
        return float(
            LINEAR_TRANSACTION_COST_RATE * s * abs(trade_size)
            + QUADRATIC_IMPACT_RATE * s * trade_size ** 2
        )

    def _obs(self):
        row = self.current_data.iloc[self.t]
        x = row[self.feature_cols].astype(float)
        x = (x - self.feature_mean[self.feature_cols]) / self.feature_std[self.feature_cols]
        x = x.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)
        delta = self._delta(row)
        return np.concatenate(
            [x, np.array([self.prev_hedge, self.prev_hedge - delta], dtype=np.float32)]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_episode_id = self.rng.choice(self.episode_ids) if self.random_episode else self.episode_ids[0]
        self.current_data = self.episode_data[self.current_episode_id]
        self.t = 0
        self.prev_hedge = 0.0
        return self._obs(), {}

    def step(self, action):
        row = self.current_data.iloc[self.t]

        raw_action, desired, hedge, adjustment = self._action_to_hedge(action, row)
        delta = self._delta(row)

        trade_size = hedge - self.prev_hedge
        tc = self._tc(row, trade_size, "SPY_CLOSE")

        is_last = self.t == len(self.current_data) - 1
        if is_last:
            tc += self._tc(row, hedge, "SPY_NEXT_CLOSE")

        raw_pnl = -float(row["DOPTION"]) + hedge * float(row["SPY_DS"]) - tc

        scale = self._scale(row)
        scaled = raw_pnl / scale
        downside = min(scaled, 0.0)

        iv = float(row.get("OPTION_IV", 0.20))
        iv = iv / 100.0 if iv > 3.0 else iv
        dte = float(row.get("DTE", 1.0))
        next_dte = float(row.get("NEXT_DTE", max(dte - 1.0, 0.0)))
        dt = max(dte - next_dte, 1.0) / 252.0

        delta_risk = (((hedge - delta) * iv * float(row["SPY_CLOSE"])) ** 2) * dt / (scale ** 2)
        reward = (
            scaled
            - DOWNSIDE_PENALTY * downside ** 2
            - DELTA_RISK_PENALTY * delta_risk
            - EXTRA_COST_PENALTY * (tc / scale)
        )

        turnover = abs(trade_size) + (abs(hedge) if is_last else 0.0)

        self.prev_hedge = hedge
        self.t += 1
        terminated = self.t >= len(self.current_data)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32) if terminated else self._obs()

        info = {
            "raw_action": raw_action,
            "delta": delta,
            "desired_hedge": desired,
            "hedge": hedge,
            "adjustment": adjustment,
            "raw_pnl_per_share": raw_pnl,
            "transaction_cost_per_share": tc,
            "turnover": turnover,
            "no_trade": abs(trade_size) < 1e-12,
        }
        return obs, float(reward), terminated, False, info


# ============================================================
# Model builders
# ============================================================

def make_env(df, feature_cols, feature_mean, feature_std, seed):
    return Monitor(OptionHedgingEnvResidualDelta(df, feature_cols, feature_mean, feature_std, seed=seed))


def make_action_noise():
    return NormalActionNoise(mean=np.zeros(1), sigma=0.10 * np.ones(1))


def make_model(algorithm: str, env, seed: int):
    algorithm = algorithm.upper()

    if algorithm == "SAC":
        return SAC(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            ent_coef="auto",
            buffer_size=100_000,
            learning_starts=1_000,
            train_freq=1,
            gradient_steps=1,
            tau=0.005,
            target_update_interval=1,
            policy_kwargs=OFF_POLICY_KWARGS,
            seed=seed,
            tensorboard_log=str(OUT_DIR / "tb_logs" / "quantile_option_state_e4_sac"),
        )

    if algorithm == "TD3":
        return TD3(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            learning_rate=3e-4,
            batch_size=256,
            gamma=0.99,
            buffer_size=100_000,
            learning_starts=1_000,
            train_freq=(1, "step"),
            gradient_steps=1,
            tau=0.005,
            policy_delay=2,
            target_policy_noise=0.20,
            target_noise_clip=0.50,
            action_noise=make_action_noise(),
            policy_kwargs=OFF_POLICY_KWARGS,
            seed=seed,
            tensorboard_log=str(OUT_DIR / "tb_logs" / "quantile_option_state_e4_td3"),
        )

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def clone_for_real_finetune(pretrained_model, algorithm: str, real_env, seed: int):
    """Clone network parameters into a new model with an empty real-data replay buffer."""
    params = pretrained_model.get_parameters()
    new_model = make_model(algorithm, real_env, seed)
    new_model.set_parameters(params, exact_match=True)
    return new_model


# ============================================================
# Evaluation
# ============================================================

def make_obs(row, prev_hedge, feature_cols, feature_mean, feature_std):
    x = row[feature_cols].astype(float)
    x = (x - feature_mean[feature_cols]) / feature_std[feature_cols]
    x = x.replace([np.inf, -np.inf], 0).fillna(0).values.astype(np.float32)
    delta = float(np.clip(float(row["OPTION_DELTA"]), HEDGE_MIN, HEDGE_MAX))
    return np.concatenate([x, np.array([prev_hedge, prev_hedge - delta], dtype=np.float32)]).astype(np.float32)


def compute_trade_cost(row, trade_size, use_next=False):
    s = float(row["SPY_NEXT_CLOSE"] if use_next else row["SPY_CLOSE"])
    return float(
        LINEAR_TRANSACTION_COST_RATE * s * abs(trade_size)
        + QUADRATIC_IMPACT_RATE * s * trade_size ** 2
    )


def evaluate_model(model, real_df, algorithm, seed, feature_cols, feature_mean, feature_std, training_time_min):
    step_rows, ep_rows = [], []
    df = real_df.sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)
    experiment = f"E4_quantile_option_state_{algorithm.upper()}"

    for episode_id, ep in df.groupby("EPISODE_ID"):
        ep = ep.reset_index(drop=True)
        split = ep["SPLIT"].iloc[0]
        prev_hedge = 0.0

        rewards, tcs, turnovers = [], [], []
        hedges, deltas, actions, adjustments, no_trades = [], [], [], [], []

        for t in range(len(ep)):
            row = ep.iloc[t]
            obs = make_obs(row, prev_hedge, feature_cols, feature_mean, feature_std)
            action, _ = model.predict(obs, deterministic=True)

            raw_action = float(np.clip(action[0], -1.0, 1.0))
            delta = float(np.clip(float(row["OPTION_DELTA"]), HEDGE_MIN, HEDGE_MAX))
            desired = float(np.clip(delta + ADJUSTMENT_LIMIT * raw_action, HEDGE_MIN, HEDGE_MAX))
            hedge = prev_hedge if abs(desired - prev_hedge) < NO_TRADE_BAND else desired
            adjustment = hedge - delta

            trade_size = hedge - prev_hedge
            tc = compute_trade_cost(row, trade_size, use_next=False)
            is_last = t == len(ep) - 1
            turnover = abs(trade_size)
            if is_last:
                tc += compute_trade_cost(row, hedge, use_next=True)
                turnover += abs(hedge)

            pnl = -float(row["DOPTION"]) + hedge * float(row["SPY_DS"]) - tc

            rewards.append(pnl)
            tcs.append(tc)
            turnovers.append(turnover)
            hedges.append(hedge)
            deltas.append(delta)
            actions.append(raw_action)
            adjustments.append(adjustment)
            no_trades.append(abs(trade_size) < 1e-12)

            step_rows.append({
                "EXPERIMENT": experiment,
                "ALGORITHM": algorithm.upper(),
                "SEED": seed,
                "EPISODE_ID": episode_id,
                "SPLIT": split,
                "QUOTE_DATE": row["QUOTE_DATE"],
                "RAW_ACTION": raw_action,
                "DELTA": delta,
                "HEDGE": hedge,
                "ADJUSTMENT_FROM_DELTA": adjustment,
                "NO_TRADE": abs(trade_size) < 1e-12,
                "REWARD": pnl * CONTRACT_MULTIPLIER,
                "TRANSACTION_COST": tc * CONTRACT_MULTIPLIER,
                "TURNOVER": turnover,
            })

            prev_hedge = hedge

        ep_rows.append({
            "EXPERIMENT": experiment,
            "ALGORITHM": algorithm.upper(),
            "SEED": seed,
            "EPISODE_ID": episode_id,
            "SPLIT": split,
            "N_STEPS": len(ep),
            "TERMINAL_PNL": np.sum(rewards) * CONTRACT_MULTIPLIER,
            "TOTAL_TC": np.sum(tcs) * CONTRACT_MULTIPLIER,
            "TOTAL_TURNOVER": np.sum(turnovers),
            "AVG_HEDGE": np.mean(hedges),
            "AVG_DELTA": np.mean(deltas),
            "AVG_ADJUSTMENT_FROM_DELTA": np.mean(adjustments),
            "NO_TRADE_RATE": np.mean(no_trades),
            "ACTION_NEAR_NEG1_RATE": np.mean(np.array(actions) < -0.95),
            "ACTION_NEAR_POS1_RATE": np.mean(np.array(actions) > 0.95),
            "STRATEGY": f"{algorithm.lower()}_quantile_option_state_seed_{seed}",
            "TRAINING_TIME_MIN": training_time_min,
        })

    return pd.DataFrame(step_rows), pd.DataFrame(ep_rows)


def cvar_95(x):
    s = pd.Series(x).dropna()
    if s.empty:
        return np.nan
    q = s.quantile(0.05)
    return s[s <= q].mean()


def sharpe_like(x):
    s = pd.Series(x).dropna()
    std = s.std()
    return s.mean() / std if std and std != 0 else np.nan


def make_metrics(df, group_cols):
    return (
        df.groupby(group_cols, dropna=False)
        .agg(
            EPISODES=("EPISODE_ID", "nunique"),
            MEAN_PNL=("TERMINAL_PNL", "mean"),
            STD_PNL=("TERMINAL_PNL", "std"),
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
    )


# ============================================================
# Main runner
# ============================================================

def run(args):
    if not REAL_FILE.exists():
        raise FileNotFoundError(f"Cannot find real file: {REAL_FILE}")

    sim_path = Path(args.sim)
    if not sim_path.is_absolute():
        sim_path = PROJECT_DIR / sim_path
    if not sim_path.exists():
        raise FileNotFoundError(f"Cannot find sim file: {sim_path}")

    real = add_engineered_features(pd.read_parquet(REAL_FILE))
    sim = add_engineered_features(pd.read_parquet(sim_path))
    real_train = real[real["SPLIT"].astype(str).str.lower().eq("train")].copy()

    feature_cols = [c for c in FEATURE_COLS if c in real.columns and c in sim.columns]
    feature_mean = real_train[feature_cols].mean()
    feature_std = real_train[feature_cols].std().replace(0, 1)

    print("Feature columns:")
    print(feature_cols)
    print(f"Real rows: {len(real):,}; Sim rows: {len(sim):,}; Real train rows: {len(real_train):,}")

    all_steps, all_eps, logs = [], [], []

    for algorithm in [a.upper() for a in args.algorithms]:
        for seed in args.seeds:
            print("=" * 100)
            print(f"E4 quantile option-state pretraining | algorithm={algorithm} | seed={seed}")
            print("=" * 100)

            start = time.time()

            pretrain_env = make_env(sim, feature_cols, feature_mean, feature_std, seed)
            model = make_model(algorithm, pretrain_env, seed)
            model.learn(total_timesteps=args.pretrain_steps)

            pretrain_model_path = MODEL_DIR / f"E4_{algorithm.lower()}_quantile_pretrained_seed_{seed}"
            model.save(pretrain_model_path)

            real_env = make_env(real_train, feature_cols, feature_mean, feature_std, seed)

            if args.clear_replay_before_finetune:
                model = clone_for_real_finetune(model, algorithm, real_env, seed)
            else:
                model.set_env(real_env)

            model.learn(total_timesteps=args.finetune_steps, reset_num_timesteps=False)

            elapsed = (time.time() - start) / 60.0
            final_model_path = MODEL_DIR / f"E4_{algorithm.lower()}_quantile_finetuned_seed_{seed}"
            model.save(final_model_path)

            steps, eps = evaluate_model(
                model=model,
                real_df=real,
                algorithm=algorithm,
                seed=seed,
                feature_cols=feature_cols,
                feature_mean=feature_mean,
                feature_std=feature_std,
                training_time_min=elapsed,
            )

            all_steps.append(steps)
            all_eps.append(eps)

            logs.append({
                "EXPERIMENT": f"E4_quantile_option_state_{algorithm}",
                "ALGORITHM": algorithm,
                "SEED": seed,
                "PRETRAIN_STEPS": args.pretrain_steps,
                "FINETUNE_STEPS": args.finetune_steps,
                "SIM_FILE": str(sim_path),
                "CLEAR_REPLAY_BEFORE_FINETUNE": bool(args.clear_replay_before_finetune),
                "PRETRAIN_MODEL_PATH": str(pretrain_model_path),
                "FINAL_MODEL_PATH": str(final_model_path),
                "TRAINING_TIME_MIN": elapsed,
                "FEATURES": ", ".join(feature_cols),
            })

    step_results = pd.concat(all_steps, ignore_index=True)
    ep_results = pd.concat(all_eps, ignore_index=True)

    metrics_by_seed = make_metrics(ep_results, ["EXPERIMENT", "ALGORITHM", "SEED", "SPLIT", "STRATEGY"])
    metrics = make_metrics(ep_results, ["EXPERIMENT", "ALGORITHM", "SPLIT", "STRATEGY"])
    summary = (
        metrics_by_seed.groupby(["EXPERIMENT", "ALGORITHM", "SPLIT"])
        .agg(
            N_SEEDS=("SEED", "nunique"),
            MEAN_OF_MEAN_PNL=("MEAN_PNL", "mean"),
            STD_OF_MEAN_PNL=("MEAN_PNL", "std"),
            MEAN_OF_CVAR_95=("CVAR_95", "mean"),
            STD_OF_CVAR_95=("CVAR_95", "std"),
            MEAN_OF_SHARPE_LIKE=("SHARPE_LIKE", "mean"),
            MEAN_TC=("MEAN_TC", "mean"),
            MEAN_TURNOVER=("MEAN_TURNOVER", "mean"),
            NO_TRADE_RATE=("NO_TRADE_RATE", "mean"),
            AVG_ADJUSTMENT_FROM_DELTA=("AVG_ADJUSTMENT_FROM_DELTA", "mean"),
            ACTION_NEAR_NEG1_RATE=("ACTION_NEAR_NEG1_RATE", "mean"),
            ACTION_NEAR_POS1_RATE=("ACTION_NEAR_POS1_RATE", "mean"),
            TRAINING_TIME_MIN=("TRAINING_TIME_MIN", "mean"),
        )
        .reset_index()
    )

    suffix = args.mode
    xlsx = OUT_DIR / f"pretraining_quantile_option_state_td3_sac_results_{suffix}.xlsx"
    parquet = OUT_DIR / f"pretraining_quantile_option_state_td3_sac_step_results_{suffix}.parquet"

    step_results.to_parquet(parquet, index=False)
    sample = step_results if len(step_results) <= 500000 else step_results.sample(500000, random_state=42)

    with pd.ExcelWriter(xlsx, engine="openpyxl") as writer:
        ep_results.to_excel(writer, sheet_name="Episode_Results", index=False)
        metrics.to_excel(writer, sheet_name="Metrics", index=False)
        metrics_by_seed.to_excel(writer, sheet_name="Metrics_By_Seed", index=False)
        summary.to_excel(writer, sheet_name="Algorithm_Summary", index=False)
        pd.DataFrame(logs).to_excel(writer, sheet_name="Training_Log", index=False)
        sample.to_excel(writer, sheet_name="Step_Results_Sample", index=False)

    print(f"Saved: {xlsx}")
    print(f"Saved: {parquet}")
    print(summary[summary["SPLIT"].astype(str).str.lower().eq("test")].to_string(index=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["pilot", "final"], default="pilot")
    parser.add_argument("--sim", type=str, default=None)
    parser.add_argument("--pretrain_steps", type=int, default=50_000)
    parser.add_argument("--finetune_steps", type=int, default=100_000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--algorithms", type=str, nargs="+", default=["TD3", "SAC"], choices=["TD3", "SAC", "td3", "sac"])
    parser.add_argument("--keep_sim_replay", action="store_true", help="If set, keep simulated replay buffer during real fine-tuning. Default clears replay.")
    args = parser.parse_args()

    if args.sim is None:
        args.sim = str(SIM_FINAL_DEFAULT if args.mode == "final" else SIM_PILOT_DEFAULT)

    args.clear_replay_before_finetune = not args.keep_sim_replay
    run(args)


if __name__ == "__main__":
    main()
