# Residual-Delta Reinforcement Learning for Option Hedging

Graduation Thesis

Author: Nguyen Xuan Huynh

---

## Overview

This repository contains the implementation of a Residual-Delta Reinforcement Learning framework for discrete-time option hedging under transaction costs.

The key idea is to parameterize the hedge ratio as

\[
h_t = \Delta_t + e_t,
\]

where

\[
e_t \in [-\varepsilon,\varepsilon]
\]

is a bounded residual correction learned by a reinforcement learning agent.

---

## Repository Structure

src/
    training/
    experiments/
    evaluation/
    simulation/
    pretraining/
    reporting/

supplementary/
    quantile_extension/
    regime_switching_pilot/

tools/

---

## Main Experiments

### Main Comparison

PPO vs SAC vs TD3

File:

src/experiments/main_comparison/...

### Factorial Ablation

Residual Geometry
No-Trade Band
Risk Penalty

File:

src/experiments/ablation/...

### Epsilon Sensitivity

File:

src/experiments/sensitivity/...

---

## Installation

pip install -r requirements.txt

---

## Data

The original SPY option dataset is not included due to size constraints.

Processed data should be placed in:

data/

---
