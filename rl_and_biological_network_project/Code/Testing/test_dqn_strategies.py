import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pandas as pd
import matplotlib.pyplot as plt
import time

# Add parent directory to path
import sys
from pathlib import Path
current_dir = Path().resolve()
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0,str(root_dir))

from Gyms.SimulatedNetworkSync import SimulatedNetworkSync
from Gyms.RealNetworkSync import RealNetworkSync
from Algorithms.DQN import DQN, ReplayBuffer
from Reward.LinearReward import LinearReward
from Reward.TrainingReward import TrainingReward

# Initialize the environment parameters
action_dim = 5 # Number of dimensions in each action (5 time steps)
state_dim = 10 # Number of features in the state representation
circuit_id = 3

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TrainingLogger:
    def __init__(self):
        self.episode_rewards = []
        self.epsilons = []
        self.losses = []

    def log(self, episode, reward, epsilon, loss):
        self.episode_rewards.append(reward)
        self.epsilons.append(epsilon)
        self.losses.append(loss)
        print(f"Episode {episode} | Reward: {reward:.2f} | ε: {epsilon:.3f} | Loss: {loss:.4f}")

    def save(self, filename_prefix):
        df = pd.DataFrame({
            "episode": range(len(self.episode_rewards)),
            "reward": self.episode_rewards,
            "epsilon": self.epsilons,
            "loss": self.losses
        })
        df.to_csv(f"{filename_prefix}_training_log.csv", index=False)

    def plot(self, filename_prefix):
        plt.figure(figsize=(12, 8))
        
        plt.subplot(3, 1, 1)
        plt.plot(self.episode_rewards, label='Reward', color='blue')
        plt.ylabel('Reward')
        plt.title('Training Performance')
        
        plt.subplot(3, 1, 2)
        plt.plot(self.epsilons, label='Epsilon', color='orange')
        plt.ylabel('Epsilon')
        
        plt.subplot(3, 1, 3)
        plt.plot(self.losses, label='Loss', color='green')
        plt.ylabel('Loss')
        plt.xlabel('Episode')
        
        plt.tight_layout()
        plt.savefig(f"{filename_prefix}_training_plot.png")
        plt.close()


# ---- Discretize MultiDiscrete Actions ----
def index_to_action(index, action_dim, action_n):
    # Convert a flat index into MultiDiscrete action
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

def select_action(env, state, policy_net, epsilon, action_dim, action_n):
    # Epsilon-greedy action
    if random.random() < epsilon:
        action = env.action_space.sample()
        action_idx = action_to_index(action, action_n)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action_idx = torch.argmax(q_values).item()
            action = index_to_action(action_idx, action_dim, action_n)
    
    return action, action_idx

# ---- Train Loop ----
def train(env, episodes=200, steps_per_episode=100, batch_size=8, gamma=0.99, 
          epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
          buffer_capacity=100, lr=1e-3, target_update_freq=10, inner_dim=128):

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.nvec.shape[0]
    action_n = env.action_space.nvec[0]

    policy_net = DQN(state_dim, action_dim, action_n, inner_dim).to(device)
    target_net = DQN(state_dim, action_dim, action_n, inner_dim).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    replay_buffer = ReplayBuffer(buffer_capacity)
    logger = TrainingLogger()
    
    epsilon = epsilon_start
    episode_losses = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        episode_loss = 0

        for step in range(steps_per_episode):
            action, action_idx = select_action(env, state, policy_net, epsilon, action_dim, action_n)

            next_state, reward, _, _, info = env.step(action)
            if info['missed_cyc']:
                print("MISSED CYCLE")
                print(info)
                
            replay_buffer.push(state, action_idx, reward, next_state, False)  # done=False always

            state = next_state
            total_reward += reward

            # Train step
            if len(replay_buffer) >= batch_size:
                from time import time
                
                first = time()
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                second = time()

                states = torch.FloatTensor(states).to(device)
                actions = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states = torch.FloatTensor(next_states).to(device)
                dones = torch.BoolTensor(dones).unsqueeze(1).to(device)
                third = time()

                q_values = policy_net(states).gather(1, actions)
                middle = time()

                with torch.no_grad():
                    next_q_values = target_net(next_states).max(1)[0].unsqueeze(1)
                    target_q = rewards + gamma * next_q_values * (~dones)
                fourth = time()

                loss = nn.MSELoss()(q_values, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                fifth = time()
                
                episode_loss += loss.item()
                
                print(f'Sampling {second-first}, torch setup {third-second}, q values {fourth-third}, optimizer {fifth-fourth}')
                print(f'first half {middle-third} second {fourth-middle}')
        
        avg_loss = episode_loss / steps_per_episode if steps_per_episode > 0 else 0
        logger.log(episode, total_reward, epsilon, avg_loss)

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # Periodically update the target network
        if episode % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

    logger.save(f"results/dqn_training_{circuit_id}_LinearReward_buffer100_epsdec97_innerdim16")
    logger.plot(f"results/dqn_performance_{circuit_id}_LinearReward_buffer100_epsdec97_innerdim16")

    return policy_net


def test_policy(env, model, episodes=10, steps_per_episode=100, render=False):
    env.reward_object = LinearReward() # Reset for testing
    print(f"TESTING WITH REWARD FUNCTION {env.reward_object}")
    model.eval()
    
    # Initialize data collection
    test_rewards = []
    action_dim = env.action_space.nvec.shape[0]
    action_n = env.action_space.nvec[0]

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        for step in range(steps_per_episode):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
                action_idx = torch.argmax(q_values).item()
                action = index_to_action(action_idx, action_dim, action_n)

            state, reward, _, _, _ = env.step(action)
            total_reward += reward

            if render:
                env.render()

        print(f"Test Episode {episode} | Total Reward: {total_reward:.2f}")
        test_rewards.append(total_reward)

    # Save results to CSV
    results_df = pd.DataFrame({
        'Episode': range(episodes),
        'Total_Reward': test_rewards,
        'Avg_Reward': [r/steps_per_episode for r in test_rewards]
    })
    results_df.to_csv(f'results/dqn_test_results_{circuit_id}_LinearReward_buffer100_epsdec97_innerdim16.csv', index=False)

    # Generate plots
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(results_df['Episode'], results_df['Total_Reward'], 'b-o')
    plt.title('Total Reward per Test Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(results_df['Episode'], results_df['Avg_Reward'], 'r--s')
    plt.title('Average Reward per Step')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'results/dqn_test_performance_{circuit_id}_LinearReward_buffer100_epsdec97_innerdim16.png')
    plt.close()

    return results_df

# env = SimulatedNetworkSync(action_dim=action_dim, state_dim=state_dim)
# env = SimulatedNetworkSync(action_dim=action_dim, state_dim=state_dim, stim_period=100, reward_object=TrainingReward())
env = RealNetworkSync(action_dim=action_dim, state_dim=state_dim, circuit_id=circuit_id, reward_object=LinearReward())
trained_model = train(env, episodes=4, steps_per_episode=50, epsilon_decay=0.97, inner_dim=16)

test_policy(env, trained_model, episodes=1, steps_per_episode=50)

