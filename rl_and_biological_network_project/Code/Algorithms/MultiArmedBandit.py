import numpy as np
import gymnasium as gym
from typing import Optional

class MABAgent:
    """Epsilon-greedy Multi-Armed Bandit agent for Gymnasium environments"""
    
    def __init__(
        self,
        epsilon: float = 0.9,
        epsilon_decay: float = 0.999,
        alpha: Optional[float] = None,
        initial_q: float = 0.0,
        n_actions: int = 25,
    ):
        """
        Args:
            epsilon: Exploration probability (0.0-1.0)
            epsilon_decay: Decay rate for epsilon (0.0-1.0)
            alpha: Learning rate (None for sample-average)
            initial_q: Initial Q-value estimates
            env: Gymnasium environment (optional)
            n_actions: Number of bandit arms (should be multiple of 5, up to 5^5)
        """
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.q_values = np.full(n_actions, initial_q, dtype=np.float32)
        self.action_counts = np.zeros(self.n_actions, dtype=np.int32)
    
    def action_map(self, action: int) -> int:
        """Maps an action number to an actual length 5 action array"""
        action_arr = np.zeros(5, dtype=np.integer)
        
        for i in range(4, -1, -1):
            action_arr[i] = action % 5
            action //= 5

        return action_arr

    def select_action(self) -> int:
        """ε-greedy action selection"""
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else: 
            max_q = np.max(self.q_values)
            max_indices = np.flatnonzero(self.q_values == max_q)
            action = np.random.choice(max_indices)  # Random selection from ties

        return action

    def update(self, action: int, reward: float):
        """Update Q-values using incremental sample-average or constant alpha"""
        self.action_counts[action] += 1
        
        if self.alpha is None:  # Sample-average method
            alpha = 1 / self.action_counts[action]
        else:  # Constant step-size
            alpha = self.alpha
            
        self.q_values[action] += alpha * (reward - self.q_values[action])

        # Decay epsilon
        self.epsilon *= self.epsilon_decay