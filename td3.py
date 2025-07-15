import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import torch

# reuse our hyper-params
BUFFER_SIZE    = int(1e4)
BATCH_SIZE     = 128
GAMMA          = 0.99
TAU            = 1e-3
LR_ACTOR       = 1e-4   # SB3 uses a single learning_rate; we can choose this
POLICY_NOISE   = 0.2
NOISE_CLIP     = 0.5
POLICY_FREQ    = 2

class TD3AgentSB3:
    """A TD3 agent wrapping stable_baselines3.TD3 to mirror our custom API."""
    def __init__(self,
                 env,
                 seed: int = 0,
                 policy_kwargs: dict = None,
                 action_noise_std: float = POLICY_NOISE,
                 verbose: int = 0):
        """
        Args:
            env            : an instance of our Gym-like env
            seed           : random seed
            policy_kwargs  : dict for net_arch, etc.
            action_noise_std: std for exploration noise
            verbose        : SB3 verbosity
        """
        # vectorize env for SB3
        self.env = DummyVecEnv([lambda: env])
        
        # define action noise
        n_actions = env.action_space.shape[0]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=action_noise_std * np.ones(n_actions)
        )
        
        # build the SB3 TD3 model
        self.model = TD3(
            policy='MlpPolicy',
            env=self.env,
            learning_rate=LR_ACTOR,
            buffer_size=BUFFER_SIZE,
            batch_size=BATCH_SIZE,
            tau=TAU,
            gamma=GAMMA,
            action_noise=action_noise,
            target_policy_noise=POLICY_NOISE,
            target_noise_clip=NOISE_CLIP,
            policy_delay=POLICY_FREQ,
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=verbose,
            device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        )
    
    def act(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Query the policy. If add_noise=False, will act deterministically.
        """
        action, _ = self.model.predict(state, deterministic=not add_noise)
        # match our original scaling from [-1,1]→[0,1]
        action = (action + 1.0) / 2.0
        return np.clip(action, 0, 1)
    
    def step(self, state, action, reward, next_state, done):
        """
        No-op: SB3 handles transitions & learning internally.
        Provided for API-compatibility with our training loop.
        """
        pass
    
    def reset(self):
        """Reset any internal state (e.g., noise)."""
        # SB3’s action_noise resets internally each call to learn(), 
        # but if you had something custom you’d reset here.
        return
    
    def learn(self, total_timesteps: int, **kwargs):
        """
        Kick off the offline training.
        You can pass callbacks, tb_log_name, etc. via kwargs.
        """
        self.model.learn(total_timesteps=total_timesteps, **kwargs)
    
    def save(self, path: str):
        """Save SB3 model to disk."""
        self.model.save(path)
    
    def load(self, path: str):
        """Load SB3 model from disk (keeps self.env)."""
        self.model = TD3.load(path, env=self.env)
