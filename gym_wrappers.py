# gym_wrappers.py – thin Gymnasium adapter for MarketEnvironment and GBMMarketEnvironment
# -------------------------------------------------------------
# Usage:
#    from gym_wrappers import ACTradingGym
#    env = ACTradingGym(env_type="ac", reward_name="ac_utility", seed=123)
#    or
#    env = ACTradingGym(env_type="gbm", reward_name="ac_utility", seed=123)
#    obs, _ = env.reset()
#    obs, r, terminated, truncated, info = env.step(env.action_space.sample())
#
# This lets you plug Stable‑Baselines3 algos like SAC/TD3 with zero changes.
# -------------------------------------------------------------

from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Any, Tuple

from syntheticChrissAlmgren import MarketEnvironment
from GBM import GBMMarketEnvironment
from Hetson_Merton_Env import HestonMertonEnvironment
from Hetson_Merton_fees import HestonMertonFeesEnvironment

class ACTradingGym(gym.Env):
    """Gym-style wrapper around different trading environments.

    Supports:
    - Almgren-Chriss MarketEnvironment (env_type="ac")
    - GBM MarketEnvironment (env_type="gbm")

    Observation: 1-D float32 vector (length depends on environment)
    Action:      Box(0,1, (1,)) – fraction of remaining inventory to sell this step.
    Reward:      supplied by the underlying environment
    """

    metadata = {"render_modes": []}

    def __init__(self, env_type: str = "ac", reward_fn: str = "ac_utility", seed: int = 0,
                 liquid_time: int = 60, num_trades: int = 60, lamb: float = 1e-6):
        super().__init__()
        self.env_type = env_type
        self.reward_fn = reward_fn 
        # Initialize the appropriate environment
        if env_type == "ac":
            self._ac_env = MarketEnvironment(
                randomSeed=seed,
                lqd_time=liquid_time,
                num_tr=num_trades,
                lambd=lamb,
                reward_fn= reward_fn
            )
        elif env_type == "gbm":
            self._ac_env = GBMMarketEnvironment(
                randomSeed=seed,
                lqd_time=liquid_time,
                num_tr=num_trades,
                lambd=lamb,
                reward_fn= reward_fn
            )
        elif env_type == "hm":
            self._ac_env = HestonMertonEnvironment(
                randomSeed=seed,
                lqd_time=liquid_time,
                num_tr=num_trades,
                lambd=lamb,
                reward_fn= reward_fn
            )
        elif env_type == "hmf":
            self._ac_env = HestonMertonFeesEnvironment(
                randomSeed=seed,
                lqd_time=liquid_time,
                num_tr=num_trades,
                lambd=lamb,
                reward_fn= reward_fn
            )



        else:
            raise ValueError(f"Unknown environment type: {env_type}")

        # ----- Gym spaces -----
        self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_dim = self._ac_env.observation_space_dimension()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # internal episode step counter
        self._steps = 0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            obs = self._ac_env.reset(seed=seed, reward_fn=self.reward_fn)  # Add reward_fn
        else:
            obs = self._ac_env.reset(reward_fn=self.reward_fn)  # Add reward_fn
        self._ac_env.start_transactions()
        self._steps = 0
        return obs.astype(np.float32), {}

    def step(self, action: np.ndarray | float):
        # Handle both environments' step methods
        if self.env_type == "ac":
            obs, reward, done, info = self._ac_env.step(float(action))
        else:  # gbm
            obs, reward, done, info = self._ac_env.step(float(action))
            
        self._steps += 1
        terminated, truncated = done, False  # no truncation logic yet
        return obs.astype(np.float32), float(reward), terminated, truncated, {
            "impl_shortfall": info.implementation_shortfall if hasattr(info, "implementation_shortfall") else None,
            "steps": self._steps,
        }

    # ------------------------------------------------------------------
    # Helpers – not strictly required by Gym, but nice to have
    # ------------------------------------------------------------------
    def render(self):
        pass  # could print remaining shares / price

    def close(self):
        pass