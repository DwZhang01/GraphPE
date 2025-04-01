# Graph-based Pursuit-Evasion Environment

A multi-agent reinforcement learning environment implementing a pursuit-evasion game on graphs using the PettingZoo framework.

## Overview

This project implements a pursuit-evasion game where multiple pursuers attempt to capture evaders on a graph structure. The environment is built using the PettingZoo framework and is compatible with popular reinforcement learning libraries such as Stable Baselines3.

### Key Features

- Graph-based environment with configurable number of nodes and edges
- Support for multiple pursuers and evaders
- Customizable capture mechanics (distance and required number of captors)
- Compatible with PettingZoo's ParallelEnv interface
- Integration with Stable Baselines3
- Visualization support for training and evaluation

## Installation

```bash
# Clone the repository
git clone [your-repo-url]
cd [your-repo-name]

# Install required packages
pip install -r requirements.txt
```

### Dependencies

- gymnasium
- pettingzoo
- stable-baselines3
- supersuit
- networkx
- numpy
- opencv-python (for visualization)
- matplotlib (for plotting)

## Environment Configuration

The environment (`GPE`) can be customized with the following parameters:

```python
env = GPE(
    num_nodes=20,          # Number of nodes in the graph
    num_edges=40,          # Number of edges in the graph
    num_pursuers=2,        # Number of pursuing agents
    num_evaders=1,         # Number of evading agents
    capture_distance=1,     # Distance required for capture
    required_captors=1,    # Number of pursuers required for capture
    max_steps=50,          # Maximum steps per episode
    seed=42,              # Random seed for reproducibility
    render_mode="rgb_array" # Rendering mode
)
```

## Usage

### Basic Training Example

```python
import supersuit as ss
from stable_baselines3 import PPO
from GPE.env.graph_pe import GPE

# Create and wrap the environment
env = GPE(num_nodes=20, num_edges=40, num_pursuers=2, num_evaders=1)
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3")

# Initialize PPO model
model = PPO(
    "MlpPolicy",
    env,
    verbose=3,
    gamma=0.95,
    n_steps=256,
    ent_coef=0.0905168,
    learning_rate=0.00062211,
    vf_coef=0.042202,
    max_grad_norm=0.9,
    gae_lambda=0.99,
    n_epochs=5,
    clip_range=0.3,
    batch_size=256
)

# Train the model
model.learn(total_timesteps=200000)
model.save("policy")
```

### Visualization and Monitoring

The environment supports visualization of the pursuit-evasion process and training metrics:

```python
# Enable rendering
env = GPE(..., render_mode="rgb_array")

# Monitor training progress
from stable_baselines3.common.monitor import Monitor
env = Monitor(env, "logs/")

# Create video of episodes
# [See documentation for visualization code examples]
```

## Environment Structure

The environment implements the PettingZoo ParallelEnv interface with the following key components:

- **Observation Space**: Flattened Box space containing agent positions and graph information
- **Action Space**: Discrete space representing possible node movements
- **Reward Structure**: Rewards for successful captures and penalties for movement
- **Terminal Conditions**: Episode ends upon successful capture or maximum steps reached

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License

[Your License Information]

## Citation

If you use this environment in your research, please cite:
