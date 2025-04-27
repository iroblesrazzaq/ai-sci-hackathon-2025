import numpy as np
import gymnasium as gym
from typing import Optional
from scipy.stats import norm

MAB_STRATEGIES = ["epsilon_greedy", "ucb", "boltzmann", "thompson", "hybrid"]

class MABAgent:
    """Enhanced Multi-Armed Bandit agent with multiple exploration strategies"""
    
    def __init__(
        self,
        strategy: str = "epsilon_greedy",
        epsilon: float = 0.95,
        epsilon_decay: float = 0.9996,
        alpha: Optional[float] = None,
        initial_q: float = 0.0,
        n_actions: int = 25,
        # New parameters for exploration strategies
        temperature: float = 1.0,
        temp_decay: float = 0.9996,
        ucb_beta: float = 2.0,
        thompson_prior: float = 1.0,
        hybrid_ratio: float = 0.5,
    ):
        """
        Args:
            strategy: Exploration strategy (epsilon_greedy, ucb, boltzmann, thompson, hybrid)
            temperature: Initial temperature for softmax exploration
            temp_decay: Temperature decay rate
            ucb_beta: Exploration weight for UCB
            thompson_prior: Prior value for Thompson sampling
            hybrid_ratio: Ratio for hybrid strategies
        """
        self.n_actions = n_actions
        self.strategy = strategy
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.q_values = np.full(n_actions, initial_q, dtype=np.float32)
        self.action_counts = np.zeros(self.n_actions, dtype=np.int32)
        self.total_steps = 0
        
        # Exploration parameters
        self.temperature = temperature
        self.temp_decay = temp_decay
        self.ucb_beta = ucb_beta
        self.thompson_prior = thompson_prior
        self.hybrid_ratio = hybrid_ratio
        
        # Thompson sampling initialization
        self.successes = np.full(n_actions, thompson_prior, dtype=np.float32)
        self.failures = np.full(n_actions, thompson_prior, dtype=np.float32)

    def select_action(self) -> int:
        """Multi-strategy action selection"""
        if self.strategy == "epsilon_greedy":
            return self._epsilon_greedy()
        elif self.strategy == "ucb":
            return self._ucb()
        elif self.strategy == "boltzmann":
            return self._boltzmann()
        elif self.strategy == "thompson":
            return self._thompson()
        elif self.strategy == "hybrid":
            return self._hybrid()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _epsilon_greedy(self) -> int:
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else: 
            return self._greedy_selection()

    def _ucb(self) -> int:
        exploration_bonus = np.sqrt(np.log(self.total_steps + 1) / (self.action_counts + 1e-8))
        ucb_values = self.q_values + self.ucb_beta * exploration_bonus
        return np.argmax(ucb_values)

    def _boltzmann(self) -> int:
        scaled_q = (self.q_values - np.max(self.q_values)) / self.temperature
        exp_q = np.exp(scaled_q)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(self.n_actions, p=probs)

    def _thompson(self) -> int:
        samples = np.random.beta(self.successes, self.failures)
        return np.argmax(samples)

    def _hybrid(self) -> int:
        if np.random.random() < self.hybrid_ratio:
            return self._epsilon_greedy()
        else:
            return self._boltzmann()

    def _greedy_selection(self) -> int:
        max_q = np.max(self.q_values)
        max_indices = np.flatnonzero(self.q_values == max_q)
        return np.random.choice(max_indices)

    def update(self, action: int, reward: float):
        """Update agent parameters and exploration values"""
        self.total_steps += 1
        self.action_counts[action] += 1
        
        # Update Q-values
        alpha = self.alpha or 1 / self.action_counts[action]
        self.q_values[action] += alpha * (reward - self.q_values[action])
        
        # Update Thompson parameters (convert reward to pseudo-success/failure)
        epsilon = 1e-6  # Minimum value to prevent beta distribution errors
        self.successes[action] = max(self.successes[action] + reward, epsilon)
        self.failures[action] = max(self.failures[action] + (1 - reward), epsilon)
        
        # Decay exploration parameters
        self.epsilon *= self.epsilon_decay
        self.temperature *= self.temp_decay

    def action_map(self, action: int) -> int:
        """Maps action number to actual length 5 action array"""
        action_arr = np.zeros(5, dtype=np.int64)
        for i in range(4, -1, -1):
            action_arr[i] = action % 5
            action //= 5
        return action_arr
