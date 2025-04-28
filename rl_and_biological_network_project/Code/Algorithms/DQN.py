import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# ---- Q-Network ----
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, action_n, inner_dim=128):
        super(DQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_n = action_n
        self.output_dim = action_n ** action_dim  # total discrete actions

        self.net = nn.Sequential(
            nn.Linear(state_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, self.output_dim)
        )

    def forward(self, x):
        return self.net(x)
    
# ---- Replay Buffer ----
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.buffer)

