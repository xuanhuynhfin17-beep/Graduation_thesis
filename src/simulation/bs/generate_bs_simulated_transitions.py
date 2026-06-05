"""
06_generate_bs_simulated_transitions.py

Generate Black-Scholes / GBM simulated option transition data for DRL hedging.

Purpose:
    Create synthetic short-call hedging episodes with the same schema as the
    real SPY option transition dataset used by the residual-delta V3C environment.

Output:
    data/processed/simulated_bs_transitions.parquet

Recommended first run:
    py src\06_generate_bs_simulated_transitions.py --episodes 5000 --dte 30

Larger thesis run:
    py src\06_generate_bs_simulated_transitions.py --episodes 20000 --dte-mix 15 30 45 60

Notes:
    - This simulator is intentionally simple and transparent.
    - It is used for pretraining / data augmentation, not final testing.
    - Final evaluation should remain on real SPY option data.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# PATH HELPERS
# ============================================================

REAL_TRANSITIONS_FILE = "transitions_daily_top1_final_with_spy_2010_2023.parquet"
DEFAULT_OUTPUT_FILE = "simulated_bs_transitions.parquet"


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


def find_optional_file(filename: str, relative_dirs: Iterable[str]) -> Path | None:
    for base in _candidate_project_dirs():
        for rel in relative_dirs:
            p = base / rel / filename if rel else base / filename
            if p.exists():
                return p
    return None


def infer_project_dir() -> Path:
    here = Path(__file__).resolve()
    if (here.parent.parent / "data" / "processed").exists():
        return here.parent.parent
    if (Path.cwd() / "data" / "processed").exists():
        return Path.cwd()
    return Path("/mnt/data")


PROJECT_DIR = infer_project_dir()
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# BLACK-SCHOLES HELPERS
# ============================================================

SQRT_2PI = math.sqrt(2.0 * math.pi)


def norm_pdf(x):
    x = np.asarray(x, dtype=float)
    return np.exp(-0.5 * x * x) / SQRT_2PI


def norm_cdf(x):
    x = np.asarray(x, dtype=float)
    erf_vec = np.vectorize(math.erf)
    return 0.5 * (1.0 + erf_vec(x / math.sqrt(2.0)))


def bs_call_price_delta_gamma_vega_theta(s, k, tau, r, sigma):
    """Return call price and greeks for Black-Scholes.

    theta returned here is per trading day, not annualized.
    """
    s = np.asarray(s, dtype=float)
    tau = np.asarray(tau, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    tau_safe = np.maximum(tau, 1e-8)
    sigma_safe = np.maximum(sigma, 1e-8)

    sqrt_tau = np.sqrt(tau_safe)
    d1 = (np.log(np.maximum(s, 1e-12) / k) + (r + 0.5 * sigma_safe ** 2) * tau_safe) / (sigma_safe * sqrt_tau)
    d2 = d1 - sigma_safe * sqrt_tau

    nd1 = norm_cdf(d1)
    nd2 = norm_cdf(d2)
    pdf_d1 = norm_pdf(d1)

    disc_k = k * np.exp(-r * tau_safe)
    price = s * nd1 - disc_k * nd2
    delta = nd1
    gamma = pdf_d1 / np.maximum(s * sigma_safe * sqrt_tau, 1e-12)
    vega = s * pdf_d1 * sqrt_tau

    theta_annual = -(s * pdf_d1 * sigma_safe) / (2.0 * sqrt_tau) - r * disc_k * nd2
    theta_daily = theta_annual / 252.0

    # For tau close to zero, use intrinsic value and stable greeks.
    near_expiry = tau <= 1e-8
    if np.any(near_expiry):
        intrinsic = np.maximum(s - k, 0.0)
        price = np.where(near_expiry, intrinsic, price)
        delta = np.where(near_expiry, (s > k).astype(float), delta)
        gamma = np.where(near_expiry, 0.0, gamma)
        vega = np.where(near_expiry, 0.0, vega)
        theta_daily = np.where(near_expiry, 0.0, theta_daily)

    return price, delta, gamma, vega, theta_daily


# ============================================================
# PARAMETER SAMPLING
# ============================================================


def load_real_train_stats() -> dict:
    """Use real train distributions when available, otherwise fallback."""
    real_path = find_optional_file(
        REAL_TRANSITIONS_FILE,
        relative_dirs=["data/processed", "processed", ""],
    )
    if real_path is None:
        print("Real transitions not found. Using fallback simulation ranges.")
        return {}

    try:
        real = pd.read_parquet(real_path)
        real_train = real[real["SPLIT"] == "train"].copy()
        if real_train.empty:
            real_train = real.copy()

        stats = {
            "spy_q01": float(real_train["SPY_CLOSE"].quantile(0.01)),
            "spy_q99": float(real_train["SPY_CLOSE"].quantile(0.99)),
            "iv_q05": float(real_train["OPTION_IV"].quantile(0.05)),
            "iv_q95": float(real_train["OPTION_IV"].quantile(0.95)),
        }

        if "SPY_MONEYNESS" in real_train.columns:
            stats["mny_q05"] = float(real_train["SPY_MONEYNESS"].quantile(0.05))
            stats["mny_q95"] = float(real_train["SPY_MONEYNESS"].quantile(0.95))
        elif {"SPY_CLOSE", "STRIKE"}.issubset(real_train.columns):
            m = real_train["SPY_CLOSE"].astype(float) / real_train["STRIKE"].astype(float)
            stats["mny_q05"] = float(m.quantile(0.05))
            stats["mny_q95"] = float(m.quantile(0.95))

        print("Loaded real train stats from:", real_path)
        print(stats)
        return stats
    except Exception as exc:
        print("Could not load real train stats. Using fallback ranges.")
        print("Reason:", repr(exc))
        return {}


def sanitize_iv(iv):
    iv = float(iv)
    if not np.isfinite(iv) or iv <= 0:
        iv = 0.20
    if iv > 3.0:
        iv = iv / 100.0
    return float(np.clip(iv, 0.05, 1.00))


# ============================================================
# SIMULATION CORE
# ============================================================


def simulate_gbm_path(s0: float, mu: float, sigma: float, n_steps: int, dt: float, rng: np.random.Generator):
    z = rng.normal(size=n_steps)
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    log_path = np.empty(n_steps + 1)
    log_path[0] = math.log(s0)
    log_path[1:] = log_path[0] + np.cumsum(log_returns)
    return np.exp(log_path)


def generate_simulated_transitions(
    n_episodes: int,
    dte_choices: list[int],
    seed: int,
    use_real_stats: bool,
    output_path: Path,
):
    rng = np.random.default_rng(seed)
    stats = load_real_train_stats() if use_real_stats else {}

    if stats:
        s_low = max(stats.get("spy_q01", 80.0), 1.0)
        s_high = max(stats.get("spy_q99", 500.0), s_low * 1.10)
        iv_low = sanitize_iv(stats.get("iv_q05", 0.10))
        iv_high = sanitize_iv(stats.get("iv_q95", 0.60))
        if iv_high <= iv_low:
            iv_low, iv_high = 0.10, 0.60
        m_low = max(stats.get("mny_q05", 0.90), 0.70)
        m_high = min(stats.get("mny_q95", 1.10), 1.30)
        if m_high <= m_low:
            m_low, m_high = 0.90, 1.10
    else:
        s_low, s_high = 80.0, 500.0
        iv_low, iv_high = 0.10, 0.60
        m_low, m_high = 0.90, 1.10

    dt = 1.0 / 252.0
    rows = []

    base_date = pd.Timestamp("2000-01-03")

    for i in range(n_episodes):
        n_steps = int(rng.choice(dte_choices))

        # Domain randomization.
        s0 = float(rng.uniform(s_low, s_high))
        sigma = float(rng.uniform(iv_low, iv_high))
        mu = float(rng.uniform(-0.05, 0.12))
        r = float(rng.uniform(0.00, 0.05))
        moneyness0 = float(rng.uniform(m_low, m_high))
        strike = float(s0 / max(moneyness0, 1e-8))

        # Add occasional higher-volatility stress episodes.
        if rng.random() < 0.15:
            sigma = float(rng.uniform(max(iv_high, 0.35), 0.90))

        s_path = simulate_gbm_path(s0=s0, mu=mu, sigma=sigma, n_steps=n_steps, dt=dt, rng=rng)

        dtes = np.arange(n_steps, 0, -1, dtype=float)
        next_dtes = dtes - 1.0
        tau = dtes / 252.0
        next_tau = np.maximum(next_dtes, 0.0) / 252.0

        s_t = s_path[:-1]
        s_next = s_path[1:]

        price, delta, gamma, vega, theta = bs_call_price_delta_gamma_vega_theta(
            s=s_t,
            k=strike,
            tau=tau,
            r=r,
            sigma=sigma,
        )
        next_price, _, _, _, _ = bs_call_price_delta_gamma_vega_theta(
            s=s_next,
            k=strike,
            tau=next_tau,
            r=r,
            sigma=sigma,
        )

        spread_base = rng.uniform(0.001, 0.010)
        spread_noise = rng.uniform(0.0, 0.020, size=n_steps)
        spread_pct = np.clip(spread_base + spread_noise / np.maximum(price, 1.0), 0.001, 0.05)

        episode_start_date = base_date + pd.Timedelta(days=int(i % 3650))
        quote_dates = [episode_start_date + pd.Timedelta(days=int(t)) for t in range(n_steps)]
        next_quote_dates = [episode_start_date + pd.Timedelta(days=int(t + 1)) for t in range(n_steps)]

        episode_id = f"SIM_{i:07d}"

        for t in range(n_steps):
            money = s_t[t] / strike
            rows.append({
                "EPISODE_ID": episode_id,
                "SPLIT": "train",
                "QUOTE_DATE": quote_dates[t],
                "NEXT_QUOTE_DATE": next_quote_dates[t],
                "SPY_CLOSE": float(s_t[t]),
                "SPY_NEXT_CLOSE": float(s_next[t]),
                "SPY_DS": float(s_next[t] - s_t[t]),
                "STRIKE": strike,
                "DTE": float(dtes[t]),
                "NEXT_DTE": float(next_dtes[t]),
                "OPTION_MID": float(max(price[t], 0.0)),
                "NEXT_OPTION_MID": float(max(next_price[t], 0.0)),
                "DOPTION": float(next_price[t] - price[t]),
                "OPTION_DELTA": float(np.clip(delta[t], 0.0, 1.0)),
                "OPTION_GAMMA": float(max(gamma[t], 0.0)),
                "OPTION_VEGA": float(max(vega[t], 0.0)),
                "OPTION_THETA": float(theta[t]),
                "OPTION_IV": float(sigma),
                "OPTION_SPREAD_PCT": float(spread_pct[t]),
                "SPY_MONEYNESS": float(money),
                "SPY_LOG_MONEYNESS": float(math.log(max(money, 1e-12))),
                "SIM_MU": float(mu),
                "SIM_R": float(r),
                "SIM_SIGMA": float(sigma),
            })

        if (i + 1) % max(1, n_episodes // 10) == 0:
            print(f"Generated {i + 1:,}/{n_episodes:,} episodes")

    df = pd.DataFrame(rows)
    df = df.sort_values(["EPISODE_ID", "QUOTE_DATE"]).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print("\nSaved simulated transitions:")
    print(output_path)
    print("Shape:", df.shape)
    print("Rows by split:")
    print(df["SPLIT"].value_counts())
    print("DTE summary:")
    print(df["DTE"].describe())
    print("IV summary:")
    print(df["OPTION_IV"].describe())


# ============================================================
# CLI
# ============================================================


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / DEFAULT_OUTPUT_FILE))
    parser.add_argument("--dte", type=int, default=None, help="Single DTE to use, e.g. 30")
    parser.add_argument("--dte-mix", type=int, nargs="*", default=[15, 30, 45, 60])
    parser.add_argument("--no-real-stats", action="store_true", help="Do not calibrate ranges from real train data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.dte is not None:
        dte_choices = [args.dte]
    else:
        dte_choices = args.dte_mix

    generate_simulated_transitions(
        n_episodes=args.episodes,
        dte_choices=dte_choices,
        seed=args.seed,
        use_real_stats=not args.no_real_stats,
        output_path=Path(args.output),
    )
