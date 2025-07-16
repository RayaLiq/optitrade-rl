<<<<<<< HEAD
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import DummyVecEnv

class SB3SACAgent:
    def __init__(self, env, policy_kwargs=None, **kwargs):
        self.env = DummyVecEnv([lambda: env])
        self.model = SAC('MlpPolicy', self.env, policy_kwargs=policy_kwargs, **kwargs)
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
=======
# sac.py
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
import gym

class SB3SACAgent:
    """
    A wrapper for the Stable-Baselines3 (SB3) SAC agent.

    :param env: The custom market environment instance.
    :param policy_kwargs: A dictionary of arguments for the policy network. 
                          (e.g., `policy_kwargs=dict(net_arch=[256, 256])`).
    :param kwargs: Other hyperparameters for the SAC model, such as `learning_rate`,
                   `buffer_size`, `gamma`, etc.
    """
    def __init__(self, env: gym.Env, seed: int, policy_kwargs: dict = None, **kwargs):
        # SB3 is designed to work with vectorized environments. `make_vec_env` wraps
        self.vec_env = make_vec_env(lambda: env, n_envs=1)
        self.model = SAC(
            policy='MlpPolicy',          # Standard multi-layer perceptron policy
            env=self.vec_env,
            policy_kwargs=policy_kwargs, # For changing network architecture
            verbose=0,                   # Set to 1 to see SB3 training logs
            seed=seed,
            **kwargs                     # For tuning other hyperparameters
        )

    def learn(self, total_timesteps: int):
        """
        Train the SAC model for a given number of environment steps.
        
        SB3's `.learn()` method encapsulates the entire training loop, including
        data collection (rollouts), replay buffer management, and network updates.
        """
        self.model.learn(total_timesteps=total_timesteps)

    def act(self, state: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """
        Get an action from the trained policy for a given state.

        :param state: The current observation from the environment.
        :param deterministic: If True (for evaluation), return the action with the highest
                              probability. If False (for training/exploration), sample
                              from the policy's action distribution.
        """
        action, _ = self.model.predict(state, deterministic=deterministic)
        
        # *** CRUCIAL STEP FOR COMPATIBILITY ***
        # The SAC policy outputs actions in the range [-1, 1] due to the Tanh squashing function.
        # DDPG agent scales this to [0, 1] before returning. We must do the same
        # here to ensure both agents are interacting with the environment identically.
        # This scales the action from [-1, 1] to [0, 1].
        scaled_action = (action + 1.0) / 2.0
        
        return np.clip(scaled_action, 0, 1)

    def step(self, state, action, reward, next_state, done):
        """
        A dummy 'step' method for compatibility with the DDPG training loop in runner.py.
        This method is called by the runner but is not needed for SAC's training,
        as the `.learn()` method handles all data processing internally.
        """
        pass 

    def reset(self):
        """
        A dummy 'reset' method for compatibility. SAC's internal state (if any)
        is managed within the `.learn()` method.
        """
        pass  

>>>>>>> d3b7565988697e0fc23ab8cbbb17e965ad28ecb8
