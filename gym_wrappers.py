# gym_wrappers.py – thin Gymnasium adapter for MarketEnvironment
# -------------------------------------------------------------
# Usage:
#    from gym_wrappers import ACTradingGym
#    env = ACTradingGym(reward_name="ac_utility", seed=123)
#    obs, _ = env.reset()
#    obs, r, terminated, truncated, info = env.step(env.action_space.sample())
#
# This lets you plug Stable‑Baselines3 algos like SAC/TD3 with zero changes.
# -------------------------------------------------------------

from __future__ import annotations

import gymnasium as gym
import numpy as np
from typing import Any, Tuple

from syntheticChrissAlmgren_extended import MarketEnvironment

class ACTradingGym(gym.Env):
    """Gym‑style wrapper around the Almgren‑Chriss MarketEnvironment.

    Observation: 1‑D float32 vector (len=8)  – last 5 log‑returns + timeFrac + invFrac
    Action:      Box(0,1, (1,)) – fraction of remaining inventory to sell this step.
    Reward:      supplied by MarketEnvironment via rewards.REWARD_FN_MAP.
    """

    metadata = {"render_modes": []}

    def __init__(self, reward_name: str = "ac_utility", seed: int = 0,
                 liquid_time: int = 60, num_trades: int = 60, lamb: float = 1e-6):
        super().__init__()
        self._ac_env = MarketEnvironment(randomSeed=seed,
                                          lqd_time=liquid_time,
                                          num_tr=num_trades,
                                          lambd=lamb)
        self.reward_name = reward_name

        # ----- Gym spaces -----
        self.action_space      = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
        obs_dim = self._ac_env.observation_space_dimension()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # internal episode step counter
        self._steps = 0

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            obs = self._ac_env.reset(seed)
        else:
            obs = self._ac_env.reset()
        self._ac_env.start_transactions()
        self._steps = 0
        return obs.astype(np.float32), {}

    def step(self, action: np.ndarray | float):
        obs, reward, done, info = self._ac_env.step(float(action), reward_function=self.reward_name)
        self._steps += 1
        terminated, truncated = done, False  # no truncation logic yet
        return obs.astype(np.float32), float(reward), terminated, truncated, {
            "impl_shortfall": info.implementation_shortfall if hasattr(info, "implementation_shortfall") else None,
            "steps": self._steps,
        }

    # ------------------------------------------------------------------
    # Helpers – not strictly required by Gym, but nice to have
    # ------------------------------------------------------------------
    def render(self):
        pass  # could print remaining shares / price

    def close(self):
        pass
