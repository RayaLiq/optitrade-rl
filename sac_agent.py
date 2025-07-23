import numpy as np
import random
import copy
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim
from sac_model import GaussianPolicy, QNetwork

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 5e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
LR_ALPHA = 1e-4         # learning rate for entropy coefficient
TARGET_ENTROPY = -1.0   # target entropy for automatic temperature tuning

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SACAgent():
    """SAC Agent interacting with environment"""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize an Agent object.
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Policy Network
        self.policy = GaussianPolicy(state_size, action_size, random_seed).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=LR_ACTOR)
        
        # Q-Networks (Critics)
        self.q1 = QNetwork(state_size, action_size, random_seed).to(device)
        self.q2 = QNetwork(state_size, action_size, random_seed).to(device)
        self.q1_target = QNetwork(state_size, action_size, random_seed).to(device)
        self.q2_target = QNetwork(state_size, action_size, random_seed).to(device)
        
        # Initialize target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        # Q-Network optimizers
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=LR_CRITIC)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=LR_CRITIC)
        
        # Entropy temperature
        self.target_entropy = TARGET_ENTROPY * action_size
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ALPHA)
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        # Loss tracking
        self.critic_losses = []
        self.actor_losses = []
        self.alpha_losses = []
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory and learn if possible"""
        self.memory.add(state, action, reward, next_state, done)
        
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
    
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy"""
        state = torch.from_numpy(state).float().to(device)
        self.policy.eval()
        with torch.no_grad():
            action, _ = self.policy.sample(state)
        self.policy.train()
        return action.cpu().data.numpy()
    
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experiences"""
        states, actions, rewards, next_states, dones = experiences
        
        # -------------------- Update Q-networks -------------------- #
        with torch.no_grad():
            next_actions, next_log_probs = self.policy.sample(next_states)
            q1_next = self.q1_target(next_states, next_actions)
            q2_next = self.q2_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_targets = rewards + gamma * (1 - dones) * q_next
        
        # Compute Q losses
        q1_current = self.q1(states, actions)
        q2_current = self.q2(states, actions)
        q1_loss = F.mse_loss(q1_current, q_targets)
        q2_loss = F.mse_loss(q2_current, q_targets)
        
        # Update Q-networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # -------------------- Update policy -------------------- #
        new_actions, log_probs = self.policy.sample(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        policy_loss = (self.alpha * log_probs - q_new).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        # -------------------- Update alpha -------------------- #
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()
        
        # -------------------- Update targets -------------------- #
        self.soft_update(self.q1, self.q1_target, TAU)
        self.soft_update(self.q2, self.q2_target, TAU)
        
        # Save losses for tracking
        self.critic_losses.append((q1_loss.item() + q2_loss.item()) / 2)
        self.actor_losses.append(policy_loss.item())
        self.alpha_losses.append(alpha_loss.item())
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples"""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object"""
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
            field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory"""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(
            np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(
            np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(
            np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(
            np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(
            np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory"""
        return len(self.memory)