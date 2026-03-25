"""
agent.py
========
Deep Q-Network (DQN) agent with:
  - Experience replay buffer
  - Target network (updated periodically for stability)
  - Epsilon-greedy exploration with linear decay
"""

import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


# ── Neural Network ───────────────────────────────────────────────────────────

class DQNNetwork(nn.Module):
    """
    Fully-connected network that maps state → Q-values for each action.

    Architecture: state_size → 128 → 128 → 64 → num_actions
    """

    def __init__(self, state_size: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Replay Buffer ────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular buffer that stores (state, action, reward, next_state, done)
    transitions and samples random mini-batches for training.
    """

    def __init__(self, capacity: int = 10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states,      dtype=np.float32),
            np.array(actions,     dtype=np.int64),
            np.array(rewards,     dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones,       dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ── DQN Agent ────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    DQN agent with experience replay and a separate target network.

    Parameters
    ----------
    state_size    : dimensionality of the state vector
    num_actions   : number of discrete actions
    lr            : learning rate
    gamma         : discount factor for future rewards
    epsilon_start : initial exploration rate
    epsilon_end   : minimum exploration rate
    epsilon_decay : linear decay per episode
    batch_size    : mini-batch size for training
    target_update : how often (in episodes) to sync target network
    buffer_cap    : maximum replay buffer size
    """

    def __init__(
        self,
        state_size:    int,
        num_actions:   int,
        lr:            float = 1e-3,
        gamma:         float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end:   float = 0.05,
        epsilon_decay: float = 0.002,
        batch_size:    int   = 64,
        target_update: int   = 10,
        buffer_cap:    int   = 10_000,
    ):
        self.state_size   = state_size
        self.num_actions  = num_actions
        self.gamma        = gamma
        self.epsilon      = epsilon_start
        self.epsilon_end  = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size   = batch_size
        self.target_update = target_update

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Online network (trained every step)
        self.online_net = DQNNetwork(state_size, num_actions).to(self.device)
        # Target network (lags behind for stable Q targets)
        self.target_net = DQNNetwork(state_size, num_actions).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.buffer    = ReplayBuffer(buffer_cap)

        self.episode_count = 0  # track for target-net updates

    # ── Action selection ─────────────────────────────────────────────────────

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Epsilon-greedy: random action with probability ε, greedy otherwise.
        During evaluation (training=False) always act greedily.
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.online_net(state_t)
        return int(q_values.argmax(dim=1).item())

    # ── Learning step ────────────────────────────────────────────────────────

    def learn(self):
        """Sample a mini-batch and perform one gradient-descent step."""
        if len(self.buffer) < self.batch_size:
            return None  # not enough data yet

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Convert to tensors
        s  = torch.tensor(states,      device=self.device)
        a  = torch.tensor(actions,     device=self.device)
        r  = torch.tensor(rewards,     device=self.device)
        ns = torch.tensor(next_states, device=self.device)
        d  = torch.tensor(dones,       device=self.device)

        # Current Q-values for taken actions
        current_q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q-values (Bellman equation)
        with torch.no_grad():
            max_next_q = self.target_net(ns).max(dim=1).values
            target_q   = r + self.gamma * max_next_q * (1 - d)

        loss = nn.MSELoss()(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping prevents exploding gradients
        nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()

    # ── Book-keeping ─────────────────────────────────────────────────────────

    def end_episode(self):
        """Called after each episode: decay ε, maybe sync target network."""
        self.episode_count += 1

        # Linear epsilon decay
        self.epsilon = max(self.epsilon_end, self.epsilon - self.epsilon_decay)

        # Periodically copy weights to target network
        if self.episode_count % self.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        """Save model weights and agent hyper-parameters."""
        torch.save(
            {
                "online_net":  self.online_net.state_dict(),
                "target_net":  self.target_net.state_dict(),
                "epsilon":     self.epsilon,
                "episode":     self.episode_count,
            },
            path,
        )
        print(f"[Agent] Model saved → {path}")

    def load(self, path: str):
        """Load previously saved model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.online_net.load_state_dict(checkpoint["online_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.epsilon       = checkpoint.get("epsilon",  self.epsilon_end)
        self.episode_count = checkpoint.get("episode",  0)
        print(f"[Agent] Model loaded ← {path}  (ε={self.epsilon:.3f})")
