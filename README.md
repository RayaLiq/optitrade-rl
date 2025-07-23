# OptiTrade-RL 📉🤖

**Deep Reinforcement Learning for Optimal Trade Execution**

---

## Overview

OptiTrade-RL is a modular Python framework for researching and experimenting with deep reinforcement learning (RL) algorithms applied to optimal trade execution problems in finance. The project supports multiple RL agents, flexible environments, customizable reward functions, and scalable experiment workflows.

---

## Features

- **Agents:** Modular implementations of DDPG, TD3, and SAC algorithms.
- **Environments:** Includes Chriss-Almgren, GBM, Hetson-Merton, and Hetson-Merton with fees.
- **Reward Functions:** Easily switch and modify reward functions via a centralized mapping.
- **Action Scaling:** Unified action scaling utilities for all environments.
- **Replay Buffer:** Consistent experience replay implementation for training.
- **Extensible Design:** Add or modify agents, models, environments, and reward functions independently.

---

## Directory Structure

```
agents/
  ddpg_agent.py
  td3_agent.py
  sac_agent.py
actions/
  scalers.py
envs/
  chriss_almgren.py
  gbm.py
  hetson_merton.py
  hetson_metson_fees.py
models/
  ddpg_model.py
  sac_model.py
  td3_model.py
rewards/
  core.py
scripts/
  runner.py
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Recommended: Create a virtual environment

### Installation

```bash
git clone https://github.com/RayaLiq/optitrade-rl.git
cd optitrade-rl
# Install dependencies
pip install -r requirements.txt
```

---

## Running Experiments with `runner.py`

The easiest way to train and evaluate agents is by using the provided `runner.py` script.

### Quick Start

Run an experiment from the command line:
```bash
python runner.py --agent ddpg --env chriss_almgren --reward_fn basic --episodes 100
```

### Supported Arguments

- `--agent` : RL agent to use (`ddpg`, `td3`, `sac`)
- `--env` : Environment (`chriss_almgren`, `gbm`, `hetson_merton`, `hetson_metson_fees`)
- `--reward_fn` : Reward function name (see `rewards/core.py`)
- `--episodes` : Number of training episodes

Additional arguments are available for hyperparameters, logging, and output control—check inside `runner.py` for all options.

### Tips

- `runner.py` automatically sets up the environment, initializes the agent, and handles logging.
- You can extend or modify `runner.py` to support new agents, environments, or custom experiment workflows.
- For batch experiments, consider using shell scripts or modifying `runner.py` to loop over multiple configurations.

---

## Customization

- **Add new agents:** Place them in `agents/` and ensure they follow the modular API.
- **Add new environments:** Implement in `envs/` and support the `reward_fn` parameter.
- **Modify reward logic:** Edit or extend `rewards/core.py`.

---

## Results & Monitoring

- Losses and metrics are tracked for each agent.
- Results and episode rewards are logged automatically.
- For custom logging, extend the agent’s train loop or use external tools.