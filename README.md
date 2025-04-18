# Getting Started with Midway3

This guide is designed to help you quickly start using the Midway3 system and the hardware provided for this event.

## Accessing Midway3 on RCC

RCC provides a user guide for accessing the shared cluster systems, available [here](https://rcc-uchicago.github.io/user-guide/). We have reserved a private partition of Midway3 for teams that require GPU resources for the challenge.

### Logging In
Use the following command to log into Midway3:

```
ssh cnetid@midway3.rcc.uchicago.edu
```

Log in with your password and confirm the authentication in DUO.

## Checking Permissions

After logging in, check your permissions by running:
```
id
```

Your output should include `11323(ai4s-hackathon)`. If it does not, contact us immediately.

## Workspace Setup

Create a workspace for your team:
```
mkdir /project/ai4s-hackathon/your_team_name
cd /project/ai4s-hackathon/your_team_name
```
Store your data and models here, but keep data sizes and file counts reasonable to avoid impacting others.

### Personal Workspaces
To facilitate collaboration, create a personal space within the team directory:
```
mkdir your_name
cd your_name
```

## Obtaining Hackathon Data

Clone the hackathon data repository:
```
git clone https://github.com/uchicago-dsi/ai-sci-hackathon-2025.git
```

## Environment Setup

We have prepared a tech stack with packages for each project at `material_characterize_project/gnnpytorch_env.yml` and `rl_and_biological_networks/rl_and_biological_networks_env.yml`. 

To use the shared environment for the material characterization project:
```
source setup_material_characterize.sh
```
** please check the material_characterize_project/README.md file for additional details about this environment

To use the shared environment for the RL and biological networks project:
```
source setup_rl_and_biological_network.sh
```

If you need to install packages in addition to those in the shared environment(s), we recommend using an additional python virtual environment.

To create, activate, and install packages to a virtual environment named `your_venv`:
```
mkdir -p /project/ai4s-hackathon/your_team_name/your_name/venvs/your_venv
source <setup_file_for_project.sh>
python3 -m venv /project/ai4s-hackathon/your_team_name/your_name/venvs/your_venv --system-site-packages
source /project/ai4s-hackathon/your_team_name/your_name/venvs/your_venv/bin/activate
python3 -m pip install --upgrade pip ...
```
For example, if you want to run the jupyter notebooks contained in `rl_and_biological_network_project/Code/Examples`, you can install `jupyter` in your virtual environment with `pip install jupyter`.

To activate this environment later:
```
source <setup_file_for_project.sh>
source /project/ai4s-hackathon/your_team_name/your_name/venvs/your_venv/bin/activate
```
To use this environment in jobs submitted to the cluster SLURM scheduler, add the command `source /project/ai4s-hackathon/your_team_name/your_name/venvs/your_venv/bin/activate` after the `source activate ...` command in your job submission script.

## Executing Jobs on GPUs

Use SLURM to schedule jobs on the GPU:
```
sbatch example_submission.sh
```
Check the status of your job:
```
squeue -p schmidt-gpu
```

Results will be available in `slurm-<job_id>.out`.

## Best Practices for Resource Sharing

To ensure fair resource sharing, minimize the use of interactive jobs and Jupyter Notebooks. Thank you for your cooperation.

## Useful Links

 - Invite to [Slack](https://join.slack.com/t/aiscienceuchi-pwb7058/shared_invite/zt-33gwx0qd1-wXO6gryIe6R9h7w04ZPlHw)
 - This [Repo](https://github.com/uchicago-dsi/ai-sci-hackathon-2025)
