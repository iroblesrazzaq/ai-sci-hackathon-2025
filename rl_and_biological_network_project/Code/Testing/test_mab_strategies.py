import argparse
import gymnasium as gym
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path

# Add parent directory to path
current_dir = Path().resolve()
root_dir = current_dir.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from Gyms.RealNetworkSync import RealNetworkSync
from Algorithms.MultiArmedBandit import MABAgent, MAB_STRATEGIES

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Run MAB simulation with specified strategy and circuit ID.')
    parser.add_argument('--strategy', required=True, choices=MAB_STRATEGIES, 
                       help=f'MAB strategy to use. Choices: {MAB_STRATEGIES}')
    parser.add_argument('--circuit_id', type=int, required=True, choices=[0,1,2,3],
                       help='Circuit ID (0-3)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    strategy = args.strategy
    circuit_id = args.circuit_id
    
    # Environment configuration
    state_dim = 4
    action_dim = 5
    
    # Create environment
    env = RealNetworkSync(
        action_dim=action_dim,
        state_dim=state_dim,
        circuit_id=circuit_id
    )
    
    # Run simulation for specified strategy
    print(f"\n{'='*50}")
    print(f"Running {strategy} strategy on circuit {circuit_id}")
    print(f"{'='*50}\n")
    
    train_df, eval_df = run_mab_simulation_full(env, strategy)
    
    # Save results with strategy/circuit identifiers
    train_file = f"train_results_{strategy}_circuit_{circuit_id}.csv"
    eval_file = f"eval_results_{strategy}_circuit_{circuit_id}.csv"
    train_df.to_csv(train_file, index=False)
    eval_df.to_csv(eval_file, index=False)
    print(f"\nSaved results to {train_file} and {eval_file}")
    
    # Generate and save analysis
    results_analysis = analyze_results(train_df, eval_df)
    plot_file = f"cumulative_rewards_{strategy}_circuit_{circuit_id}.png"
    plot_cumulative_rewards(results_analysis, strategy, circuit_id)
    print(f"Saved visualization to {plot_file}")

def run_mab_simulation_full(env, strategy, update_interval=1800):
    """Run MAB simulation for specified strategy"""
    results = {'train': {strategy: []}, 'eval': {strategy: []}}
    
    agent = MABAgent(strategy=strategy, n_actions=25, alpha=0.1)
    state, info = env.reset()
    stim_id = 1  # Initial dummy value
    
    with tqdm(total=21600+7200, desc=f"Running {strategy}") as pbar:
        while stim_id > 0:
            action_idx = agent.select_action()
            state, reward, terminated, truncated, info = env.step(
                agent.action_map(action_idx)
            )
            stim_id = info['stim_id']

            if stim_id == 0:
                tqdm.write("Network reset detected")
                break

            # Phase determination
            phase = 'train' if stim_id < 21600 else 'eval'
            results[phase][strategy].append(reward)

            # Agent updates only during training
            if phase == 'train':
                agent.update(action_idx, reward)

            # Progress updates
            if stim_id % update_interval == 0:
                log_progress(stim_id, phase, agent, results, strategy)
            
            pbar.update(1)
            if terminated or truncated:
                break

    return pd.DataFrame(results['train']), pd.DataFrame(results['eval'])

def log_progress(stim_id, phase, agent, results, strategy):
    """Helper method for logging progress updates"""
    try:
        window = results[phase][strategy][-1800:]
        avg_reward = np.mean(window) if window else 0
        cum_reward = np.sum(window) if window else 0
        
        params = []
        if agent.strategy in ['epsilon_greedy', 'hybrid']:
            params.append(f"ε={agent.epsilon:.3f}")
        if agent.strategy in ['boltzmann', 'hybrid']:
            params.append(f"T={agent.temperature:.3f}")
        if agent.strategy == 'ucb':
            params.append(f"β={agent.ucb_beta:.1f}")
        
        update_msg = (
            f"{strategy} @ {stim_id} ({phase.upper()}): "
            f"Avg={avg_reward:.2f} | Cum={cum_reward:.0f} | "
            f"{', '.join(params)}"
        )
        tqdm.write(update_msg)
        
    except Exception as e:
        print(f"Error logging progress: {e}")

def analyze_results(train_df, eval_df):
    """Generate performance analysis"""
    return {
        'training_stats': train_df.describe().T,
        'evaluation_stats': eval_df.describe().T,
        'cumulative_rewards': {
            'train': train_df.cumsum(),
            'eval': eval_df.cumsum()
        }
    }

def plot_cumulative_rewards(results_analysis, strategy, circuit_id):
    """Visualize results with strategy/circuit context"""
    plt.figure(figsize=(14, 8))
    
    train_cum = results_analysis['cumulative_rewards']['train']
    eval_cum = results_analysis['cumulative_rewards']['eval']

    # Plot training phase
    plt.plot(train_cum.index, train_cum[strategy], 
            label='Training', linewidth=2, color='navy')
    
    # Plot evaluation phase
    eval_start = len(train_cum)
    plt.plot(np.arange(eval_start, eval_start+len(eval_cum)), eval_cum[strategy],
            label='Evaluation', linestyle='--', linewidth=2, color='crimson')
    
    # Formatting
    plt.axvline(eval_start, color='gray', linestyle=':', 
               label='Evaluation Start')
    plt.title(f"{strategy} Performance on Circuit {circuit_id}", pad=20)
    plt.xlabel("Simulation Step", labelpad=15)
    plt.ylabel("Cumulative Reward", labelpad=15)
    plt.legend()
    plt.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f"cumulative_rewards_{strategy}_circuit_{circuit_id}.png")
    plt.close()

if __name__ == "__main__":
    main()
