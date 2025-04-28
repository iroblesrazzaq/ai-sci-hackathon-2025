#!/usr/bin/bash
#SBATCH --account=ai4s-hackathon
#SBATCH --reservation=ai4s-hackathon
#SBATCH -p schmidt-gpu
#SBATCH --qos=schmidt
#SBATCH --gres=gpu:1
#SBATCH --time 20:00


module load python/miniforge-24.1.2 # python 3.10

echo "output of the visible GPU environment"
nvidia-smi

# Use rl and biological networks environment
source activate /project/ai4s-hackathon/ai-sci-hackathon-2025/envs/rl+bnpytorch
source /project/ai4s-hackathon/team-3/henry/venvs/venv/bin/activate

# Run 
python /project/ai4s-hackathon/team-3/henry/ai-sci-hackathon-2025/rl_and_biological_network_project/Code/Testing/test_dqn_strategies.py