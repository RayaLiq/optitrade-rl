import numpy as np
import random
import copy
from collections import namedtuple, deque
from actions import transform_action

import torch
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.buffers import ReplayBuffer as SB3ReplayBuffer

BUFFER_SIZE = int(1e4)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 5e-3              # for soft update of target parameters (TD3 uses 5e-3)
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
POLICY_DELAY = 2        # TD3 policy update delay
TARGET_NOISE = 0.2      # TD3 target policy smoothing noise
NOISE_CLIP = 0.5        # TD3 target policy smoothing noise clip

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TD3Agent():
    """TD3 Agent that interacts with and learns from the environment using Twin Delayed Deep Deterministic Policy Gradients."""
    
    def __init__(self, state_size, action_size, random_seed, env=None):
        """Initialize a TD3Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
            env: gym environment (optional, for stable_baselines3)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        
        # If environment is provided, use stable_baselines3 TD3
        if env is not None:
            self.use_sb3 = True
            # Action noise for exploration
            action_noise = NormalActionNoise(mean=np.zeros(action_size), sigma=0.1 * np.ones(action_size))
            
            # Initialize stable_baselines3 TD3 model
            self.model = TD3(
                "MlpPolicy",
                env,
                action_noise=action_noise,
                learning_rate=LR_ACTOR,
                buffer_size=BUFFER_SIZE,
                learning_starts=BATCH_SIZE,
                batch_size=BATCH_SIZE,
                tau=TAU,
                gamma=GAMMA,
                train_freq=1,
                policy_delay=POLICY_DELAY,
                target_policy_noise=TARGET_NOISE,
                target_noise_clip=NOISE_CLIP,
                verbose=1,
                seed=random_seed,
                device=device
            )
        else:
            self.use_sb3 = False
            # Custom TD3 implementation
            from model import Actor, Critic
            
            # Actor Networks (w/ Target Networks)
            self.actor_local = Actor(state_size, action_size, random_seed).to(device)
            self.actor_target = Actor(state_size, action_size, random_seed).to(device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

            # Twin Critic Networks (w/ Target Networks) - TD3 uses two critics
            self.critic_local_1 = Critic(state_size, action_size, random_seed).to(device)
            self.critic_target_1 = Critic(state_size, action_size, random_seed).to(device)
            self.critic_optimizer_1 = optim.Adam(self.critic_local_1.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
            
            self.critic_local_2 = Critic(state_size, action_size, random_seed).to(device)
            self.critic_target_2 = Critic(state_size, action_size, random_seed).to(device)
            self.critic_optimizer_2 = optim.Adam(self.critic_local_2.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

            # Noise process
            self.noise = NormalNoise(action_size, random_seed)

            # Replay memory
            self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
            
            # TD3 specific parameters
            self.policy_update_counter = 0
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        if self.use_sb3:
            # For stable_baselines3, learning is handled internally
            return
        
        # Save experience / reward
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True, transform_method=None):
        """Returns actions for given state as per current policy."""
        if self.use_sb3:
            # Use stable_baselines3 model for action prediction
            action, _ = self.model.predict(state, deterministic=not add_noise)
            
            if transform_method:
                action = transform_action(action, method=transform_method)
            
            return np.clip(action, 0, 1)
        else:
            # Custom TD3 implementation
            state = torch.from_numpy(state).float().to(device)
            self.actor_local.eval()
            with torch.no_grad():
                action = self.actor_local(state).cpu().data.numpy()
            self.actor_local.train()
            
            if add_noise:
                action += self.noise.sample()
            
            action = (action + 1.0) / 2.0
            
            if transform_method:
                action = transform_action(action, method=transform_method)
            
            return np.clip(action, 0, 1)

    def learn_sb3(self, total_timesteps):
        """Learn using stable_baselines3 TD3."""
        if self.use_sb3:
            self.model.learn(total_timesteps=total_timesteps)

    def reset(self):
        """Reset noise process."""
        if not self.use_sb3:
            self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples (TD3 algorithm).
        
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if self.use_sb3:
            return  # Learning is handled internally by stable_baselines3
            
        states, actions, rewards, next_states, dones = experiences
        self.policy_update_counter += 1

        # ---------------------------- update critics ---------------------------- #
        with torch.no_grad():
            # Target policy smoothing: add noise to target actions
            noise = torch.randn_like(actions) * TARGET_NOISE
            noise = torch.clamp(noise, -NOISE_CLIP, NOISE_CLIP)
            
            next_actions = self.actor_target(next_states) + noise
            next_actions = torch.clamp(next_actions, -1, 1)
            
            # Compute target Q-values using the minimum of two critics (clipped double Q-learning)
            target_q1 = self.critic_target_1(next_states, next_actions)
            target_q2 = self.critic_target_2(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Compute Q targets for current states
            Q_targets = rewards + (gamma * target_q * (1 - dones))

        # Update first critic
        current_q1 = self.critic_local_1(states, actions)
        critic_loss_1 = F.mse_loss(current_q1, Q_targets)
        self.critic_optimizer_1.zero_grad()
        critic_loss_1.backward()
        self.critic_optimizer_1.step()

        # Update second critic
        current_q2 = self.critic_local_2(states, actions)
        critic_loss_2 = F.mse_loss(current_q2, Q_targets)
        self.critic_optimizer_2.zero_grad()
        critic_loss_2.backward()
        self.critic_optimizer_2.step()

        # ---------------------------- update actor (delayed) ---------------------------- #
        if self.policy_update_counter % POLICY_DELAY == 0:
            # Compute actor loss using only the first critic
            actions_pred = self.actor_local(states)
            actor_loss = -self.critic_local_1(states, actions_pred).mean()
            
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic_local_1, self.critic_target_1, TAU)
            self.soft_update(self.critic_local_2, self.critic_target_2, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, filepath):
        """Save the model."""
        if self.use_sb3:
            self.model.save(filepath)
        else:
            torch.save({
                'actor_local': self.actor_local.state_dict(),
                'actor_target': self.actor_target.state_dict(),
                'critic_local_1': self.critic_local_1.state_dict(),
                'critic_target_1': self.critic_target_1.state_dict(),
                'critic_local_2': self.critic_local_2.state_dict(),
                'critic_target_2': self.critic_target_2.state_dict(),
            }, filepath)

    def load(self, filepath):
        """Load the model."""
        if self.use_sb3:
            self.model = TD3.load(filepath)
        else:
            checkpoint = torch.load(filepath)
            self.actor_local.load_state_dict(checkpoint['actor_local'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_local_1.load_state_dict(checkpoint['critic_local_1'])
            self.critic_target_1.load_state_dict(checkpoint['critic_target_1'])
            self.critic_local_2.load_state_dict(checkpoint['critic_local_2'])
            self.critic_target_2.load_state_dict(checkpoint['critic_target_2'])

class NormalNoise:
    """Normal noise process for exploration."""

    def __init__(self, size, seed, mu=0., sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu
        self.sigma = sigma
        self.size = size
        self.seed = random.seed(seed)

    def reset(self):
        """Reset noise (no-op for normal noise)."""
        pass

    def sample(self):
        """Sample noise from normal distribution."""
        return np.random.normal(self.mu, self.sigma, self.size)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)