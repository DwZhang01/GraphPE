import os
import json
import argparse
import logging  # Use logging for consistency
import sys
from typing import Optional
from datetime import datetime
import networkx as nx
import torch
from stable_baselines3 import PPO
from networkx.readwrite import json_graph  # For loading graph

# Import environment and wrapper
from GPE.env.graph_pe import GPE
from utils.wrappers.GNNEnvWrapper import GNNEnvWrapper

# Import visualization function
from utils.visualization import visualize_policy

# Import policy for loading the model correctly
from policy.GNNPolicy import GNNPolicy


# --- Helper Function ---
def find_latest_run_dir(
    base_dir: str = "models", prefix: str = "gnn_policy_"
) -> Optional[str]:
    """Finds the latest directory in base_dir matching the prefix and timestamp."""
    if not os.path.isdir(base_dir):
        logging.warning(f"Base directory '{base_dir}' not found.")
        return None

    all_subdirs = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    # Filter based on prefix and expected timestamp format (_YYYYMMDD_HHMMSS)
    run_dirs = [d for d in all_subdirs if d.startswith(prefix)]

    if not run_dirs:
        logging.warning(f"No directories found with prefix '{prefix}' in '{base_dir}'.")
        return None

    try:
        # Sort by timestamp (last two parts of the name)
        latest_run = max(
            run_dirs,
            key=lambda d: datetime.strptime(
                "_".join(d.split("_")[-2:]), "%Y%m%d_%H%M%S"
            ),
        )
        return os.path.join(base_dir, latest_run)
    except (IndexError, ValueError):
        logging.warning(
            f"Could not parse timestamps reliably from directory names matching '{prefix}'. Using simple alphabetical max."
        )
        return os.path.join(base_dir, max(run_dirs))  # Fallback


# --- Setup Logging ---
def setup_basic_logging():
    """Sets up basic logging to console for the test script."""
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    root_logger.handlers.clear()  # Clear existing handlers if any

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
    root_logger.setLevel(logging.INFO)


# --- Main Execution ---
if __name__ == "__main__":
    setup_basic_logging()

    parser = argparse.ArgumentParser(
        description="Visualize a trained GNN policy for GPE."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to the specific run directory containing model and config (e.g., models/gnn_policy_run_...). If None, attempts to find the latest run.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to visualize (overrides config value if set).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Max steps per episode (overrides config value if set).",
    )
    parser.add_argument(
        "--no_animation",
        action="store_true",
        help="Disable saving the animation GIF (forces save_animation=False).",
    )
    parser.add_argument(
        "--use_shortest",
        action="store_true",
        help="Force using shortest path policy instead of the loaded model.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=2,
        help="Frames per second for the saved animation GIF.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,  # Allows --deterministic / --no-deterministic
        default=True,
        help="Use deterministic actions for the loaded model (default: True).",
    )

    args = parser.parse_args()

    # --- Determine Model and Config Path ---
    target_run_dir = args.run_dir
    if target_run_dir is None:
        logging.info(
            "No run directory specified, searching for the latest run in 'models/'..."
        )
        target_run_dir = find_latest_run_dir()
        if target_run_dir is None:
            logging.error(
                "Error: Could not find any valid run directories in 'models/'. Exiting."
            )
            exit(1)
        logging.info(f"Found latest run: {target_run_dir}")
    elif not os.path.isdir(target_run_dir):
        logging.error(
            f"Error: Specified run directory '{target_run_dir}' does not exist."
        )
        exit(1)

    model_filename = "trained_model.zip"
    config_filename = "config.json"
    model_path_load = os.path.join(target_run_dir, model_filename)
    config_path_load = os.path.join(target_run_dir, config_filename)

    # Create a results directory specific to this visualization run
    run_name = os.path.basename(target_run_dir)
    results_save_dir = os.path.join("results", run_name + "_vis")

    # --- Validate File Existence ---
    if not args.use_shortest and not os.path.exists(model_path_load):
        logging.error(
            f"Error: Model file '{model_path_load}' not found and --use_shortest not specified."
        )
        exit(1)
    if not os.path.exists(config_path_load):
        logging.error(f"Error: Configuration file '{config_path_load}' not found.")
        exit(1)

    os.makedirs(results_save_dir, exist_ok=True)
    logging.info(f"Saving visualization results (if enabled) to: {results_save_dir}")

    # --- Load Configuration from the specific run ---
    logging.info(f"Loading configuration from {config_path_load}...")
    try:
        with open(config_path_load, "r") as f:
            config = json.load(f)
        # Extract relevant sections (use .get for safety)
        env_config = config.get("environment", {})
        vis_config_loaded = config.get("visualization", {})  # Get defaults from file
        nn_config = config.get(
            "neural_network", {}
        )  # May not be needed but good practice
    except Exception as e:
        logging.error(
            f"Error loading or parsing configuration file: {e}", exc_info=True
        )
        exit(1)

    # --- Reconstruct Graph ---
    graph_for_viz: Optional[nx.Graph] = None
    if "graph_adj" in env_config and env_config["graph_adj"] is not None:
        logging.info("Reconstructing graph from saved adjacency data...")
        try:
            # graph_for_viz = nx.readwrite.json_graph.adjacency_graph(env_config["graph_adj"]) # Corrected way
            graph_for_viz = json_graph.adjacency_graph(env_config["graph_adj"])
            logging.info(
                f"Graph reconstructed with {graph_for_viz.number_of_nodes()} nodes."
            )
        except Exception as e:
            logging.error(f"Error reconstructing graph: {e}", exc_info=True)
            exit(1)
    else:
        logging.error(
            "Error: 'graph_adj' data not found or is null in loaded configuration. Cannot create environment."
        )
        exit(1)

    # --- Load Model (if not using shortest path) ---
    loaded_model: Optional[PPO] = None
    if not args.use_shortest:
        logging.info(f"Loading pre-trained model from '{model_path_load}'...")
        # Determine device automatically
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        logging.info(f"Using device: {device}")
        try:
            # Ensure GNNPolicy is imported so SB3 can find it
            loaded_model = PPO.load(model_path_load, device=device, policy=GNNPolicy)
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}", exc_info=True)
            logging.error(
                "Ensure the GNNPolicy class is available and matches the saved model's policy."
            )
            exit(1)

    # --- Setup Environment for Visualization ---
    logging.info("Setting up GPE environment for visualization...")
    # Prepare config for the single visualization instance
    viz_env_init_config = env_config.copy()
    viz_env_init_config["graph"] = graph_for_viz
    # Ensure render_mode is 'human' for interactive display
    viz_env_init_config["render_mode"] = "human"
    # Override max_steps if specified in args, else use loaded config or default
    viz_env_init_config["max_steps"] = (
        args.max_steps
        if args.max_steps is not None
        else vis_config_loaded.get("MAX_STEPS", 100)
    )
    # Remove keys not expected by GPE constructor if they exist
    viz_env_init_config.pop("use_preset_graph", None)
    viz_env_init_config.pop("preset_graph", None)
    viz_env_init_config.pop("graph_adj", None)
    # layout hints might be needed if using grid layout
    m_layout = env_config.get("grid_m")  # Get layout hints if they were saved
    n_layout = env_config.get("grid_n")

    # Remove grid dimensions from the dictionary as they will be passed explicitly
    viz_env_init_config.pop(
        "grid_m", None
    )  # Use pop with default None to avoid KeyError if missing
    viz_env_init_config.pop("grid_n", None)

    try:
        # Now grid_m and grid_n are only provided via explicit keyword arguments
        viz_env_base_gnn = GPE(**viz_env_init_config, grid_m=m_layout, grid_n=n_layout)
        viz_env_wrapper_gnn = (
            GNNEnvWrapper(viz_env_base_gnn) if not args.use_shortest else None
        )
        logging.info("Visualization environment and wrapper created.")
    except TypeError as e:  # Keep the error handling
        logging.error(
            f"Error creating visualization GPE instance. Error: {e}", exc_info=True
        )
        # Log the keys being passed AFTER popping
        logging.error(
            f"Arguments passed via **viz_env_init_config (after pop): {list(viz_env_init_config.keys())}"
        )
        logging.error(f"Explicit arguments: grid_m={m_layout}, grid_n={n_layout}")
        exit()
    except Exception as e:
        logging.error(
            f"Unexpected error creating visualization GPE instance: {e}", exc_info=True
        )
        exit()

    # --- Determine Visualization Parameters ---
    num_episodes_to_run = (
        args.episodes
        if args.episodes is not None
        else vis_config_loaded.get("NUM_EPISODES", 3)
    )
    # Command line arg overrides config; default is save=True unless --no_animation is given
    save_animation_flag = not args.no_animation
    # Use shortest path if forced by arg
    use_shortest_path_policy = args.use_shortest

    # --- Run Visualization using the imported function ---
    policy_desc = (
        "ShortestPath"
        if use_shortest_path_policy
        else f"Loaded Model (Deterministic={args.deterministic})"
    )
    logging.info(
        f"\n--- Running Visualization ---"
        f"\n  Run Directory: {target_run_dir}"
        f"\n  Episodes: {num_episodes_to_run}"
        f"\n  Max Steps: {viz_env_init_config['max_steps']}"
        f"\n  Save Animation: {save_animation_flag}"
        f"\n  Animation FPS: {args.fps}"
        f"\n  Policy: {policy_desc}"
        f"\n  Save Directory: {results_save_dir}"
    )

    try:
        visualize_policy(
            model=loaded_model,  # Pass model (or None if using shortest path)
            env=viz_env_base_gnn,  # Pass the base env instance
            wrapper_instance=viz_env_wrapper_gnn,  # Pass wrapper (or None if using shortest path)
            num_episodes=num_episodes_to_run,
            max_steps=viz_env_init_config["max_steps"],
            save_animation=save_animation_flag,
            use_shortest_path=use_shortest_path_policy,
            save_dir=results_save_dir,
            animation_fps=args.fps,
            deterministic_actions=args.deterministic,  # Pass deterministic flag
        )
    except Exception as e:
        logging.error(f"An error occurred during visualization: {e}", exc_info=True)
    finally:
        # --- Clean up ---
        logging.info("Closing visualization environment...")
        viz_env_base_gnn.close()
        logging.info("Visualization script finished.")
