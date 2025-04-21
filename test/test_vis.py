import os
import json
import argparse  # For command-line arguments
import networkx as nx
import torch
from stable_baselines3 import PPO

# Import environment
from GPE.env.graph_pe import GPE

# Import utilities (wrapper, visualization) directly via utils package
from utils import GNNEnvWrapper, visualize_policy

# Import policy (assuming 'policy' is a top-level package/directory)
from policy.GNNPolicy import GNNPolicy


# Function to find the latest run directory (optional helper)
def find_latest_run_dir(base_dir="models"):
    """Finds the latest directory in base_dir based on timestamp naming convention."""
    all_subdirs = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    # Filter based on expected naming convention if needed
    run_dirs = [
        d
        for d in all_subdirs
        if d.startswith("gnn_policy_run_") and len(d.split("_")) >= 3
    ]
    if not run_dirs:
        return None
    try:
        latest_run = max(run_dirs, key=lambda d: d.split("_")[-2] + d.split("_")[-1])
        return os.path.join(base_dir, latest_run)
    except IndexError:
        print(
            "Warning: Could not parse timestamps from directory names. Using alphabetical order."
        )
        return os.path.join(base_dir, sorted(run_dirs)[-1])  # Fallback


# REMOVE the entire visualize_policy function definition from here
# ... (delete the function block) ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize a trained GNN policy for GPE."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to the specific run directory (e.g., models/gnn_policy_run_...). If None, attempts to find the latest run.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,  # Default to value in loaded config
        help="Number of episodes to visualize (overrides config value if set).",
    )
    parser.add_argument(
        "--no_animation",
        action="store_true",
        help="Disable saving animation GIF (overrides config value).",
    )
    parser.add_argument(
        "--use_shortest",
        action="store_true",
        help="Force using shortest path policy instead of the loaded model.",
    )

    args = parser.parse_args()

    # --- Determine Model and Config Path ---
    target_run_dir = args.run_dir
    if target_run_dir is None:
        print(
            "No run directory specified, searching for the latest run in 'models/'..."
        )
        target_run_dir = find_latest_run_dir()
        if target_run_dir is None:
            print("Error: Could not find any run directories in 'models/'.")
            exit(1)
        print(f"Found latest run: {target_run_dir}")
    elif not os.path.isdir(target_run_dir):
        print(f"Error: Specified run directory '{target_run_dir}' does not exist.")
        exit(1)

    model_filename = "trained_model.zip"  # Consistent with saving logic
    config_filename = "config.json"
    model_path_load = os.path.join(target_run_dir, model_filename)
    config_path_load = os.path.join(target_run_dir, config_filename)
    # Create results directory based on the visualized run name
    results_save_dir = os.path.join(
        "results", os.path.basename(target_run_dir) + "_vis"
    )

    if not os.path.exists(model_path_load) and not args.use_shortest:
        print(
            f"Error: Model file not found at {model_path_load} and not using shortest path."
        )
        exit(1)
    if not os.path.exists(config_path_load):
        print(f"Error: Configuration file not found at {config_path_load}.")
        exit(1)

    os.makedirs(results_save_dir, exist_ok=True)
    print(f"Saving visualization results (if enabled) to: {results_save_dir}")

    # --- Load Configuration from the specific run ---
    print(f"Loading configuration from {config_path_load}...")
    try:
        with open(config_path_load, "r") as f:
            config = json.load(f)
        env_config = config["environment"]
        # Use visualization config from loaded file as default
        vis_config_loaded = config.get("visualization", {})
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        exit(1)

    # --- Reconstruct Graph ---
    graph_for_viz = None
    if "graph_adj" in env_config and env_config["graph_adj"] is not None:
        print("Reconstructing graph from saved adjacency data...")
        try:
            graph_for_viz = nx.readwrite.json_graph.adjacency_graph(
                env_config["graph_adj"]
            )
        except Exception as e:
            print(f"Error reconstructing graph: {e}")
            exit(1)
    else:
        print(
            "Error: 'graph_adj' data not found or is null in loaded configuration. Cannot create environment."
        )
        exit(1)

    # --- Load Model (if not using shortest path) ---
    loaded_model = None
    if not args.use_shortest:
        print(f"Loading pre-trained model from '{model_path_load}'...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
    try:
        loaded_model = PPO.load(model_path_load, device=device)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure the GNNPolicy class is imported correctly.")
        exit(1)

    # --- Setup Environment for Visualization ---
    print("\nSetting up environment for visualization...")
    env_config_viz = env_config.copy()
    env_config_viz["graph"] = graph_for_viz  # Pass the reconstructed graph object
    env_config_viz["render_mode"] = "human"
    # Use max_steps from loaded visualization config
    env_config_viz["max_steps"] = vis_config_loaded.get("MAX_STEPS", 50)

    try:
        viz_env_base_gnn = GPE(**env_config_viz)
        viz_env_wrapper_gnn = GNNEnvWrapper(viz_env_base_gnn)
    except Exception as e:
        print(f"Error creating visualization environment: {e}")
        exit(1)

    # --- Determine Visualization Parameters ---
    num_episodes = (
        args.episodes
        if args.episodes is not None
        else vis_config_loaded.get("NUM_EPISODES", 3)
    )
    save_animation = not args.no_animation  # Command line overrides config
    # Use shortest path if forced by arg, otherwise default to False (use model)
    use_shortest_path_policy = args.use_shortest

    # --- Run Visualization using the imported function ---
    print(
        f"\nRunning visualization: Episodes={num_episodes}, Save Animation={save_animation}, Policy={'ShortestPath' if use_shortest_path_policy else 'Loaded Model'}"
    )
    visualize_policy(
        model=loaded_model,  # Will be None if use_shortest_path_policy is True
        env=viz_env_base_gnn,
        wrapper_instance=viz_env_wrapper_gnn,
        num_episodes=num_episodes,
        max_steps=env_config_viz["max_steps"],  # Use max_steps from env config
        save_animation=save_animation,
        use_shortest_path=use_shortest_path_policy,  # Use the determined value
        save_dir=results_save_dir,  # Save to the dedicated results dir
    )

    # --- Clean up ---
    viz_env_base_gnn.close()

    print("\nVisualization script finished.")
