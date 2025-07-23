import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from models.td3_model import Actor,TwinCritic

# Hyperparameters
BUFFER_SIZE = int(1e5)  # Increased buffer size
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 1e-3
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
WEIGHT_DECAY = 0
POLICY_NOISE = 0.2  # Std of Gaussian noise added to target policy
NOISE_CLIP = 0.5    # Range to clip target policy noise
POLICY_FREQ = 2     # Policy update frequency

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class GaussianNoise:
    """Gaussian noise process for exploration."""
    
    def __init__(self, size, seed, mu=0., sigma=0.1, sigma_decay=0.999):
        self.mu = mu
        self.sigma = sigma
        self.sigma_init = sigma
        self.sigma_decay = sigma_decay
        self.size = size
        self.seed = random.seed(seed)
        
    def reset(self):
        """Reset the noise process."""
        self.sigma = self.sigma_init
        
    def sample(self):
        """Generate noise sample."""
        sample = np.random.normal(self.mu, self.sigma, self.size)
        self.sigma = max(0.01, self.sigma * self.sigma_decay)  # Decay noise
        return sample

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples (same as DDPG)."""
    
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
            field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)

class TD3Agent:
    """TD3 Agent implementation."""
    
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.step_count = 0

        # Actor Networks
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Twin Critic Networks
        self.critic_local = TwinCritic(state_size, action_size, random_seed).to(device)
        self.critic_target = TwinCritic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY
        )

        # Noise process for exploration
        self.noise = GaussianNoise(action_size, random_seed, sigma=0.1)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        # Loss tracking
        self.critic_losses = []
        self.actor_losses = []

    def step(self, state, action, reward, next_state, done):
        """Save experience and learn if enough samples."""
        self.memory.add(state, action, reward, next_state, done)
        self.step_count += 1
        
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        # Apply Gaussian noise for exploration
        if add_noise:
            action += self.noise.sample()
        
        # Transform and clip action to [0,1] range
        action = (action + 1.0) / 2.0
        return np.clip(action, 0, 1)

    def reset(self):
        """Reset the noise generator."""
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given experiences."""
        states, actions, rewards, next_states, dones = experiences
        
        # -------------------- Update critics -------------------- #
        # Compute target actions with added noise (target policy smoothing)
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            noise = torch.randn_like(next_actions) * POLICY_NOISE
            noise = torch.clamp(noise, -NOISE_CLIP, NOISE_CLIP)
            next_actions = torch.clamp(next_actions + noise, -1, 1)
            
            # Transform target actions to [0,1] range
            next_actions_env = (next_actions + 1.0) / 2.0
            
            # Compute target Q-values using both critics
            target_Q1, target_Q2 = self.critic_target(next_states, next_actions_env)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + gamma * target_Q * (1 - dones)
        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic_local(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_losses.append(critic_loss.item())
        
        # Minimize critic loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # -------------------- Update actor (delayed) -------------------- #
        if self.step_count % POLICY_FREQ == 0:
            # Compute actor loss
            actions_pred = self.actor_local(states)
            actions_pred_env = (actions_pred + 1.0) / 2.0  # Transform to env range
            actor_loss = -self.critic_local.Q1(states, actions_pred_env).mean()
            self.actor_losses.append(actor_loss.item())
            
            # Minimize actor loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # ----------------- Update target networks ----------------- #
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)