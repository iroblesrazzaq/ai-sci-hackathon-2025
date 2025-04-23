import numpy as np
import gymnasium as gym
from typing import Optional

class MABAgent:
    """Epsilon-greedy Multi-Armed Bandit agent for Gymnasium environments"""
    
    def __init__(
        self,
        epsilon: float = 0.1,
        alpha: Optional[float] = None,
        initial_q: float = 0.0,
        env: Optional[gym.Env] = None,
        n_actions: Optional[int] = None
    ):
        """
        Args:
            epsilon: Exploration probability (0.0-1.0)
            alpha: Learning rate (None for sample-average)
            initial_q: Initial Q-value estimates
            env: Gymnasium environment (optional)
            n_actions: Number of bandit arms (required if no env provided)
        """
        if env:
            self.n_actions = env.action_space.n
        elif n_actions:
            self.n_actions = n_actions
        else:
            raise ValueError("Must provide either env or n_actions")
            
        self.epsilon = epsilon
        self.alpha = alpha
        self.q_values = np.full(self.n_actions, initial_q, dtype=np.float32)
        self.action_counts = np.zeros(self.n_actions, dtype=np.int32)

    def select_action(self) -> int:
        """ε-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)  # Explore
        return np.argmax(self.q_values)  # Exploit

    def update(self, action: int, reward: float):
        """Update Q-values using incremental sample-average or constant alpha"""
        self.action_counts[action] += 1
        
        if self.alpha is None:  # Sample-average method
            alpha = 1 / self.action_counts[action]
        else:  # Constant step-size
            alpha = self.alpha
            
        self.q_values[action] += alpha * (reward - self.q_values[action])

# Example usage with Gymnasium-style environment
class BernoulliBandit(gym.Env):
    """Simple Bernoulli bandit environment for demonstration"""
    def __init__(self, success_probs):
        self.success_probs = np.array(success_probs)
        self.action_space = gym.spaces.Discrete(len(success_probs))
        self.observation_space = gym.spaces.Discrete(1)
        
    def step(self, action):
        reward = np.random.binomial(1, self.success_probs[action])
        return (0, reward, False, False, {})  # (obs, reward, terminated, truncated, info)
    
    def reset(self):
        return (0, {})  # (obs, info)

if __name__ == "__main__":
    # Create environment and agent
    env = BernoulliBandit(success_probs=[0.1, 0.5, 0.9])
    agent = MABAgent(epsilon=0.1, env=env)
    
    # Training loop
    obs, info = env.reset()
    for _ in range(1000):
        action = agent.select_action()
        obs, reward, terminated, truncated, info = env.step(action)
        agent.update(action, reward)
    
    print("Final Q-values:", agent.q_values)
    print("Optimal action:", np.argmax(agent.q_values))
