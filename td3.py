import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import DummyVecEnv

class SB3TD3Agent:
    def __init__(self, env, policy_kwargs=None, **kwargs):
        self.env = DummyVecEnv([lambda: env])
        self.model = TD3('MlpPolicy', self.env, policy_kwargs=policy_kwargs, **kwargs)
        self.last_obs = None

    def act(self, state, add_noise=True):
        action, _ = self.model.predict(state, deterministic=not add_noise)
        return action

    def step(self, state, action, reward, next_state, done):
        # SB3 handles training internally; this is a no-op for compatibility
        pass

    def reset(self):
        self.last_obs = None

    def learn(self, total_timesteps):
        self.model.learn(total_timesteps=total_timesteps) 