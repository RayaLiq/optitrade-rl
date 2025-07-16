# td3.py
import numpy as np
import random
from stable_baselines3 import TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
from actions import transform_action
import gym
import torch


class TD3Agent:
    """
    A TD3 (Twin Delayed Deep Deterministic Policy Gradient) agent using Stable-Baselines3.
    
    This implementation follows the same interface as the existing DDPG agent while
    leveraging the improvements of TD3 including:
    - Twin critic networks to reduce overestimation bias
    - Delayed policy updates
    - Target policy smoothing
    
    :param state_size: Dimension of each state
    :param action_size: Dimension of each action  
    :param random_seed: Random seed for reproducibility
    :param policy_kwargs: Dictionary of arguments for the policy network
    :param kwargs: Other hyperparameters for TD3 such as learning_rate, buffer_size, etc.
    """
    
    def __init__(self, state_size, action_size, random_seed, env=None, policy_kwargs=None, **kwargs):
        """Initialize TD3 Agent with same interface as DDPG agent."""
        self.state_size = state_size
        self.action_size = action_size
        self.random_seed = random_seed
        self.env = env
        
        # Default TD3 hyperparameters (using your project's values where applicable)
        default_kwargs = {
            'learning_rate': 1e-4,  # Match your LR_ACTOR
            'buffer_size': int(1e4),  # Match your BUFFER_SIZE
            'batch_size': 128,        # Match your BATCH_SIZE
            'gamma': 0.99,            # Match your GAMMA
            'tau': 1e-3,              # Match your TAU (TD3 typically uses 5e-3, but keeping yours)
            'policy_delay': 2,        # TD3 specific: delay policy updates
            'target_policy_noise': 0.2,  # TD3 specific: target policy smoothing
            'target_noise_clip': 0.5,    # TD3 specific: noise clipping
            'learning_starts': 128,      # Start learning when batch_size samples available
            'train_freq': 1,
            'gradient_steps': 1,
            'verbose': 0,
            'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        }
        
        # Update with any provided kwargs
        default_kwargs.update(kwargs)
        
        # Default policy network architecture
        if policy_kwargs is None:
            policy_kwargs = dict(net_arch=[24, 48])
        
        # Initialize noise for exploration (similar to DDPG's OUNoise)
        self.action_noise = NormalActionNoise(
            mean=np.zeros(action_size), 
            sigma=0.1 * np.ones(action_size)
        )
        
        # If environment is provided, create vectorized environment
        if env is not None:
            # Check if it's a gym environment or custom environment
            if hasattr(env, 'observation_space') and hasattr(env, 'action_space'):
                # Standard gym environment
                self.vec_env = make_vec_env(lambda: env, n_envs=1)
            else:
                # Custom environment - create a gym wrapper
                wrapped_env = self._wrap_custom_env(env)
                self.vec_env = make_vec_env(lambda: wrapped_env, n_envs=1)
            
            # Initialize TD3 model
            self.model = TD3(
                policy='MlpPolicy',
                env=self.vec_env,
                action_noise=self.action_noise,
                policy_kwargs=policy_kwargs,
                seed=random_seed,
                **default_kwargs
            )
        else:
            self.vec_env = None
            self.model = None
            
        # Store hyperparameters for potential manual learning
        self.hyperparams = default_kwargs
        self.policy_kwargs = policy_kwargs
    
    def _wrap_custom_env(self, env):
        """Create a gym-compatible wrapper for custom environments."""
        
        class CustomEnvWrapper(gym.Env):
            def __init__(self, custom_env):
                super().__init__()
                self.custom_env = custom_env
                
                # Create gym-compatible spaces
                obs_dim = custom_env.observation_space_dimension()
                act_dim = custom_env.action_space_dimension()
                
                self.observation_space = gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
                )
                self.action_space = gym.spaces.Box(
                    low=-1, high=1, shape=(act_dim,), dtype=np.float32
                )
                
                # Forward relevant attributes
                for attr in ['dtv', 'shares_remaining', 'kappa', 'timeHorizon', 'tau', 
                            'liquidation_time', 'k', 'num_n', 'logReturns']:
                    if hasattr(custom_env, attr):
                        setattr(self, attr, getattr(custom_env, attr))
            
            def reset(self):
                return self.custom_env.reset()
            
            def step(self, action):
                obs, reward, done, info = self.custom_env.step(action)
                
                # Convert custom Info object to dict for compatibility
                if hasattr(info, '__dict__'):
                    info_dict = info.__dict__.copy()
                else:
                    info_dict = {}
                    
                # Add any additional attributes from the info object
                for attr in dir(info):
                    if not attr.startswith('_') and hasattr(info, attr):
                        try:
                            info_dict[attr] = getattr(info, attr)
                        except:
                            pass
                
                return obs, reward, done, info_dict
                
            def seed(self, seed=None):
                """Seed the environment for reproducibility."""
                if seed is not None:
                    np.random.seed(seed)
                    random.seed(seed)
                return [seed]
                
            def start_transactions(self):
                if hasattr(self.custom_env, 'start_transactions'):
                    return self.custom_env.start_transactions()
                    
            def stop_transactions(self):
                if hasattr(self.custom_env, 'stop_transactions'):
                    return self.custom_env.stop_transactions()
        
        return CustomEnvWrapper(env)
    
    def step(self, state, action, reward, next_state, done):
        """
        Save experience in replay memory for learning.
        
        For TD3 with stable-baselines3, this is handled internally by the learn() method.
        This method is kept for compatibility with the existing DDPG training loop.
        """
        pass
    
    def act(self, state, add_noise=True, deterministic=False, transform_method=None):
        """
        Returns actions for given state as per current policy.
        
        :param state: Current state observation
        :param add_noise: Whether to add noise for exploration (ignored if deterministic=True)
        :param deterministic: Whether to use deterministic action selection
        :param transform_method: Method to transform action using transform_action function
        :return: Clipped action in range [0, 1]
        """
        if self.model is None:
            raise ValueError("Model not initialized. Please provide environment in __init__.")
        
        # Deterministic takes precedence over add_noise
        use_deterministic = deterministic or not add_noise
        
        # Get action from TD3 policy
        action, _ = self.model.predict(state, deterministic=use_deterministic)
        
        # TD3 outputs actions in range [-1, 1] due to tanh activation
        # Scale to [0, 1] for compatibility with existing system
        scaled_action = (action + 1.0) / 2.0
        
        # Apply action transformation if specified
        if transform_method and self.env:
            # transform_action returns a float, so we need to convert back to array
            transformed_value = transform_action(scaled_action, self.env, method=transform_method)
            scaled_action = np.array([transformed_value] if np.isscalar(transformed_value) else transformed_value)
        
        return np.clip(scaled_action, 0, 1)
    
    def reset(self):
        """Reset the agent's internal state."""
        # Reset action noise
        if hasattr(self, 'action_noise'):
            self.action_noise.reset()
    
    def learn(self, experiences=None, gamma=None, total_timesteps=1000):
        """
        Learn from experiences using TD3 algorithm.
        
        For compatibility with DDPG interface, this method can be called.
        The actual learning is handled by stable-baselines3's internal training loop.
        
        :param experiences: Not used in SB3 implementation (kept for compatibility)
        :param gamma: Not used in SB3 implementation (kept for compatibility)
        :param total_timesteps: Number of timesteps for training
        """
        if self.model is None:
            raise ValueError("Model not initialized. Please provide environment in __init__.")
        
        # Use stable-baselines3's learn method which handles the full training loop
        self.model.learn(total_timesteps=total_timesteps)
    
    def save(self, path):
        """Save the trained model."""
        if self.model is not None:
            self.model.save(path)
    
    def load(self, path, env=None):
        """Load a trained model."""
        if env is not None:
            self.env = env
            # Check if it's a gym environment or custom environment
            if hasattr(env, 'observation_space') and hasattr(env, 'action_space'):
                # Standard gym environment
                self.vec_env = make_vec_env(lambda: env, n_envs=1)
            else:
                # Custom environment - create a gym wrapper
                wrapped_env = self._wrap_custom_env(env)
                self.vec_env = make_vec_env(lambda: wrapped_env, n_envs=1)
        
        if self.vec_env is not None:
            self.model = TD3.load(path, env=self.vec_env)
        else:
            raise ValueError("Environment must be provided to load model.")
    
    def get_hyperparameters(self):
        """Get current hyperparameters."""
        return self.hyperparams.copy()
    
    def set_hyperparameters(self, **kwargs):
        """Update hyperparameters (only works before model initialization)."""
        if self.model is not None:
            print("Warning: Model already initialized. Hyperparameter changes won't take effect.")
        self.hyperparams.update(kwargs)


# Alias for backward compatibility
Agent = TD3Agent