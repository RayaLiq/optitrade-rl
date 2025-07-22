import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
import gym

class TD3Agent:
    """
    A wrapper for the Stable-Baselines3 (SB3) TD3 agent.

    :param env: The custom market environment instance.
    :param seed: Random seed for reproducibility.
    :param policy_kwargs: Dictionary of arguments for the policy network 
                          (e.g., `dict(net_arch=[400, 300])`).
    :param kwargs: Other TD3 hyperparameters such as learning_rate, buffer_size, etc.
    """

    def __init__(self, env: gym.Env, seed: int, policy_kwargs: dict = None, **kwargs):
        self.vec_env = make_vec_env(lambda: env, n_envs=1)
        
        action_dim = self.vec_env.action_space.shape[0]
        self.action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))

        self.model = TD3(
            policy='MlpPolicy',
            env=self.vec_env,
            action_noise=self.action_noise,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=0,
            **kwargs
        )

    def learn(self, total_timesteps: int):
        """
        Train the TD3 model for a given number of environment steps.
        """
        self.model.learn(total_timesteps=total_timesteps)

    def act(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Get an action from the trained policy for a given state.

        :param state: Current observation from the environment.
        :param deterministic: Whether to use deterministic action selection.
        :return: Action scaled to [0, 1] range for compatibility.
        """
        action, _ = self.model.predict(state, deterministic=deterministic)
        scaled_action = (action + 1.0) / 2.0
        return np.clip(scaled_action, 0, 1)

    def step(self, state, action, reward, next_state, done):
        """
        Dummy method for compatibility with DDPG training loop.
        """
        pass

    def reset(self):
        """
        Dummy method for compatibility with DDPG interface.
        """
        if hasattr(self, 'action_noise'):
            self.action_noise.reset()
