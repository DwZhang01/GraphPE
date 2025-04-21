# Graph Pursuit Evasion (GPE) with GNN Agents

A multi-agent reinforcement learning environment implementing a pursuit-evasion game on graphs using the PettingZoo framework, featuring Graph Neural Network (GNN) based policies trained with Stable Baselines3.

## Overview

This project implements a pursuit-evasion game where multiple pursuers attempt to capture evaders on a graph structure. The environment is built using the PettingZoo framework and is compatible with popular reinforcement learning libraries like Stable Baselines3. It utilizes a custom GNN policy (`GNNPolicy`) and environment wrapper (`GNNEnvWrapper`) to process graph observations effectively.

### Key Features

- Graph-based environment configurable via `config.json`.
- Ability to use dynamically generated grid graphs or load preset graph structures from the config file.
- Support for multiple pursuers and evaders.
- Customizable capture mechanics (distance and required number of captors).
- GNN-based policy (`GNNPolicy` using `GATv2Conv`) for learning graph representations.
- Environment wrapper (`GNNEnvWrapper`) providing graph-structured observations (node features, edge index, action mask) suitable for GNNs.
- Compatible with PettingZoo's `ParallelEnv` interface.
- Integration with Stable Baselines3 (PPO).
- Configurable training loop via `train.py` and `config.json`.
- Visualization support for trained policies, saving results as GIFs.

## Installation

```bash
# Clone the repository
git clone https://github.com/DwZhang01/GraphPE # Replace with your repo URL if different
cd GraphPE

# Create a virtual environment (recommended)
# python -m venv venv
# source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
# OR install the package directly (which reads setup.py)
pip install .
```

### Dependencies

The core dependencies are listed in `setup.py`. Ensure you have compatible versions of:

- gymnasium
- pettingzoo
- stable-baselines3
- supersuit
- networkx
- numpy
- matplotlib
- torch
- torch_geometric (Ensure version matches your PyTorch and CPU/CUDA setup)
- imageio (and `imageio[ffmpeg]`)
- opencv-python (Potentially needed for rendering backends)

## Configuration (`config.json`)

Training, environment, GNN architecture, and visualization parameters are controlled via `config.json`.

```json
{
    "neural_network": { // For GNNPolicy
        "PI_HIDDEN_DIMS": [64, 64],
        "VF_HIDDEN_DIMS": [128, 128],
        "FEATURES_DIM": 128 // Output dim of GNNFeatureExtractor
    },
    "environment": {
        "num_nodes": 50,       // Target number of nodes if generating graph
        "num_pursuers": 2,
        "num_evaders": 1,
        "max_steps": 100,      // Max steps per episode (recommend increasing)
        "layout_algorithm": "kamada_kawai", // "spring", "grid", etc. for rendering
        "p_act": 1.0,          // Probability agent follows chosen action
        "capture_reward_pursuer": 10.0,
        // ... other reward/penalty settings ...
        "use_preset_graph": false, // Set true to load from preset_graph
        "preset_graph": {       // Only used if use_preset_graph is true
            "graph_adj": null   // Replace null with NetworkX adjacency_data dict
        }
    },
    "training": {
        "TOTAL_STEPS": 100000, // Total training steps (recommend increasing)
        "N_STEPS": 2048,       // PPO rollout buffer size
        "BATCH_SIZE": 512,
        // ... other PPO hyperparameters ...
        "N_ENVS": 1,           // Number of parallel environments (increase if possible)
        "MODEL_SAVE_NAME": "gnn_policy_run"
    },
    "visualization": {
        "MAX_STEPS": 100,      // Max steps during visualization
        "NUM_EPISODES": 5,
        "SAVE_ANIMATION": true,
        "RENDER_MODE": "human",  // Must be 'human' for visualization script
        "USE_SHORTEST_PATH": false // Use loaded model instead of shortest path
    }
}
```

## Usage

### Training a GNN Agent

The main training script is `train.py`. It reads `config.json`, sets up the environment, wraps it, initializes the PPO model with the custom `GNNPolicy`, trains, saves the model, and visualizes the result.

```bash
python train.py
```

**Key steps in `train.py`:**

1. Loads `config.json`.
2. Sets up run directory in `models/` and `results/`.
3. Loads preset graph or generates grid graph based on `use_preset_graph`.
4. Saves the final configuration (including used graph structure) to the run directory.
5. Creates the base `GPE` environment.
6. Wraps the environment with `GNNEnvWrapper`.
7. Wraps with `supersuit` wrappers (`pettingzoo_env_to_vec_env_v1`, `concat_vec_envs_v1`) for SB3 compatibility.
8. Defines `policy_kwargs` using `nn_config` from the config file to specify GNN architecture details.
9. Initializes the `PPO` model, passing the `GNNPolicy` class and `policy_kwargs`.
10. Sets up callbacks (e.g., `MARLRewardCallback`).
11. Calls `model.learn()`.
12. Saves the trained model (`.zip`) and metrics plots.
13. Loads the saved model.
14. Runs visualization using the `visualize_policy` utility.

### Running Visualization Separately

You can use `test/test_vis.py` (or adapt it) to load a previously trained model and visualize its performance.

```bash
# Example usage (adapt arguments as needed)
python test/test_vis.py --model_dir models/your_run_directory_name
```

**Key steps in visualization:**

1. Loads the `config.json` from the specified model run directory.
2. Reconstructs the graph using the `graph_adj_saved` data from the config.
3. Loads the saved `PPO` model (`.zip`).
4. Creates the `GPE` environment with `render_mode="human"` and the loaded graph.
5. Wraps the environment with `GNNEnvWrapper`.
6. Calls the `utils.visualize_policy` function, passing the loaded model, environment, and wrapper.

## Environment Structure (GNN Wrapper)

The `GNNEnvWrapper` modifies the observation space for GNN processing:

- **Observation Space (`wrapper.observation_space(agent)`)**: A `gymnasium.spaces.Dict` containing:
  - `node_features`: `Box(num_nodes, feature_dim)` - Features for each node (e.g., type, agent presence). `feature_dim` is determined dynamically in the wrapper.
  - `edge_index`: `Box(2, max_edges)` - Edge list in PyG format (COO), padded to `max_edges`.
  - `action_mask`: `Box(num_nodes,)` - Binary mask indicating valid actions (neighboring nodes + current node).
  - `agent_node_index`: `Box(1,)` - The index of the current agent's node in the graph.
- **Action Space (`wrapper.action_space(agent)`)**: `Discrete(num_nodes)` - Represents choosing the next node to move to (including staying put). Invalid moves are handled by the environment or masked by the policy.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License

[MIT license] (Update if different)

## Citation

If you use this environment in your research, please consider citing:
[GraphPE, Dongwei Zhang, 2025]
