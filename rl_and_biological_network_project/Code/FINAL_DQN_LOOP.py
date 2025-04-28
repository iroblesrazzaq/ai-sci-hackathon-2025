import argparse
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path
current_dir = Path().resolve()
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from Gyms.RealNetworkSync import RealNetworkSync
from Algorithms.DQN import DQN, ReplayBuffer
from Reward.LinearReward import LinearReward
from Reward.TrainingReward import TrainingReward

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Run DQN training with specified circuit ID and reward.')
    parser.add_argument('--circuit_id', type=int, required=True, choices=[0,1,2,3],
                       help='Circuit ID (0-3)')
    parser.add_argument('--reward', required=True, choices=["TrainingReward", "LinearReward"], 
                       help='Reward strategy to use')
    return parser.parse_args()

class TrainingLogger:
    def __init__(self):
        self.steps = []
        self.rewards = []
        self.epsilons = []
        self.losses = []
        self.phases = []

    def log(self, step, reward, epsilon, loss, phase):
        self.steps.append(step)
        self.rewards.append(reward)
        self.epsilons.append(epsilon)
        self.losses.append(loss)
        self.phases.append(phase)

    def save(self, filename_prefix):
        df = pd.DataFrame({
            "step": self.steps,
            "reward": self.rewards,
            "epsilon": self.epsilons,
            "loss": self.losses,
            "phase": self.phases
        })
        df.to_csv(f"{filename_prefix}_training_log.csv", index=False)

    def plot(self, filename_prefix):
        plt.figure(figsize=(14, 10))
        
        # Cumulative rewards
        plt.subplot(3, 1, 1)
        cumulative_rewards = np.cumsum(self.rewards)
        plt.plot(self.steps, cumulative_rewards, label='Cumulative Reward', color='blue')
        plt.ylabel('Cumulative Reward')
        plt.title('Training Performance')
        
        # Epsilon decay
        plt.subplot(3, 1, 2)
        plt.plot(self.steps, self.epsilons, label='Epsilon', color='orange')
        plt.ylabel('Epsilon')
        
        # Loss
        plt.subplot(3, 1, 3)
        plt.plot(self.steps, self.losses, label='Loss', color='green')
        plt.ylabel('Loss')
        plt.xlabel('Step')
        
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_training_plot.png")
        plt.close()

def index_to_action(index, action_dim, action_n):
    action = []
    for _ in range(action_dim):
        action.append(index % action_n)
        index //= action_n
    return np.array(list(reversed(action)))

def action_to_index(action, action_n):
    index = 0
    for a in action:
        index = index * action_n + a
    return index

def select_action(env, state, policy_net, epsilon, action_dim, action_n, phase):
    if phase == 'eval' or random.random() < epsilon:
        action = env.action_space.sample()
        action_idx = action_to_index(action, action_n)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action_idx = torch.argmax(q_values).item()
            action = index_to_action(action_idx, action_dim, action_n)
    return action, action_idx

def train(env, circuit_id, reward_name, 
          batch_size=128, gamma=0.99, epsilon_start=1.0, 
          epsilon_end=0.001, epsilon_decay=0.9992, lr=1e-3, 
          target_update_freq=1000, buffer_capacity=100000,
          update_interval=1800):
    
    session_counter = 1  # Track training sessions
    
    while True:  # Continuous training loop for multiple resets
        # Initialize fresh components for each session
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.nvec.shape[0]
        action_n = env.action_space.nvec[0]

        policy_net = DQN(state_dim, action_dim, action_n, 64).to(device)
        target_net = DQN(state_dim, action_dim, action_n, 64).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()

        optimizer = optim.Adam(policy_net.parameters(), lr=lr)
        replay_buffer = ReplayBuffer(buffer_capacity)
        logger = TrainingLogger()
        
        epsilon = epsilon_start
        total_reward = 0
        episode_loss = 0
        step_count = 0

        # Environment reset for new session
        state, info = env.reset()
        stim_id = info.get('stim_id', 1)
        phase = 'train'

        with tqdm(total=21600+7200, desc=f"Training Session {session_counter}") as pbar:
            while stim_id > 0:
                action, action_idx = select_action(
                    env, state, policy_net, epsilon, action_dim, action_n, phase
                )

                next_state, reward, terminated, truncated, info = env.step(action)
                stim_id = info.get('stim_id', 0)
                
                # Handle phase transitions
                new_phase = 'train' if stim_id < 21600 else 'eval'
                if new_phase != phase:
                    phase = new_phase
                    if phase == 'eval':
                        env.reward_object = LinearReward()
                        target_net.load_state_dict(policy_net.state_dict())

                replay_buffer.push(state, action_idx, reward, next_state, False)
                state = next_state
                total_reward += reward
                step_count += 1

                # Training step
                if phase == 'train' and len(replay_buffer) > batch_size:
                    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                    
                    states = torch.FloatTensor(states).to(device)
                    actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                    next_states = torch.FloatTensor(next_states).to(device)
                    dones = torch.BoolTensor(dones).unsqueeze(1).to(device)

                    q_values = policy_net(states).gather(1, actions)
                    with torch.no_grad():
                        next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                        target_q = rewards + gamma * next_q_values * (~dones)

                    loss = nn.MSELoss()(q_values, target_q)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    episode_loss += loss.item()

                    # Decay epsilon
                    epsilon = max(epsilon_end, epsilon * epsilon_decay)

                    # Update target network
                    if step_count % target_update_freq == 0:
                        target_net.load_state_dict(policy_net.state_dict())

                # Logging
                if step_count % update_interval == 0:
                    avg_loss = episode_loss / update_interval if episode_loss > 0 else 0
                    logger.log(step_count, total_reward, epsilon, avg_loss, phase)
                    episode_loss = 0
                    total_reward = 0
                    
                    tqdm.write(
                        f"Step {step_count} ({phase.upper()}) | "
                        f"Avg Reward: {np.mean(logger.rewards[-update_interval:]):.2f} | "
                        f"ε: {epsilon:.3f} | Loss: {avg_loss:.4f}"
                    )

                pbar.update(1)
                if terminated or truncated or (stim_id % 28800) == 0:
                    break
        
        session_counter += 1

def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize environment
    reward = TrainingReward() if args.reward == "TrainingReward" else LinearReward()
    env = RealNetworkSync(
        action_dim=5,
        state_dim=4,
        circuit_id=args.circuit_id,
        reward_object=reward
    )
    
    # Run training
    trained_model = train(
        env,
        circuit_id=args.circuit_id,
        reward_name=args.reward
    )

if __name__ == "__main__":
    main()
