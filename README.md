# OptiTrade-RL 📉🤖  
**Deep Reinforcement Learning for Optimal Trade Execution**

## Overview

OptiTrade-RL is a Python library and experimental framework for developing and testing deep reinforcement learning agents for optimal trade execution. The project simulates realistic trading environments and benchmarks various reward functions, agent architectures, and market models, focusing on minimizing implementation shortfall and understanding market impact.

## Features

- **Multiple Market Environments:** Almgren-Chriss, GBM, and Heston-Merton models.
- **Custom Reward Functions:** Implementation shortfall, VWAP benchmarking, market-aware rewards, risk-adjusted utilities, smoothness penalties, and more.
- **Agent Architectures:** DDPG, TD3, SAC, and custom agents via Stable-Baselines3.
- **Gymnasium Integration:** Wrappers for RL library compatibility.
- **Runner Script:** Flexible experimentation and training management.
- **Detailed Logging and Analysis:** CSV summaries, reward/shortfall plots.

## Installation

```bash
git clone https://github.com/RayaLiq/optitrade-rl.git
cd optitrade-rl
pip install -r requirements.txt
```

## Usage

### Quickstart: Running Experiments with `runner.py`

The main entry point for experiments is `runner.py`. This script allows you to train and evaluate agents, compare reward functions, and analyze different action transformation methods.

#### Example: Train a TD3 agent with VWAP Benchmarking reward

```bash
python runner.py --agent td3 --reward VWAP_Benchmarking --episodes 10000 --plot
```

#### Example: Compare all agents and reward functions

```bash
python runner.py --compare-agents --compare-reward all --episodes 5000 --plot
```

#### Example: Run action transformation tests

```bash
python runner.py --run-action-test --transform-methods linear square sqrt exp sigmoid clip --test-seeds 10
```

#### Arguments

- `--agent`: Agent to use (`ddpg`, `td3`, `sac`)
- `--reward`: Reward function (see `rewards.py` for options)
- `--compare-agents`: Compare all agents (`ddpg`, `sac`, `td3`)
- `--compare-reward`: Compare multiple reward functions (`all` for every reward)
- `--episodes`: Number of episodes to train/evaluate
- `--seed`: Random seed
- `--csv-dir`: Directory for saving results
- `--plot`: Generate plots after experiments
- `--action`: Action transformation method(s)
- `--compare-action`: Compare multiple action transforms
- `--fee-fixed`, `--fee-prop`: Trading fee configuration

See `runner.py` for full argument list.

## Directory Structure

- `runner.py` – Main experiment and training workflow
- `reward_function/` – Reward function tests and benchmarking
- `gym_wrappers.py` – Gymnasium adapters for environments
- `syntheticChrissAlmgren_extended.py`, `GBM.py`, `Hetson_Merton_Env.py` – Market simulation models
- `ddpg_agent.py`, `td3_custom.py` – RL agent implementations
- `rewards.py` – Reward function definitions

## Output

- Experiment results (CSV summaries)
- Reward and shortfall plots (PNG)
- Detailed logs in console