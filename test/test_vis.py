import os
import json
import argparse
import logging
import sys
from typing import Optional
from datetime import datetime
import networkx as nx
import torch
from stable_baselines3 import PPO
from networkx.readwrite import json_graph

# Import environment
from GPE.env.graph_pe import GPE
# from utils.wrappers.GNNEnvWrapper import GNNEnvWrapper # REMOVE: Wrapper is no longer needed here

# Import visualization function
from utils.visualization import visualize_policy # Assuming this is your updated visualize_policy

# Import policy for loading the model correctly
from policy.GNNPolicy import GNNPolicy


# --- Helper Function ---
def find_latest_run_dir(
    base_dir: str = "models", prefix: str = "gnn_policy_"
) -> Optional[str]:
    # ... (您的 find_latest_run_dir 函数保持不变，它看起来很好)
    if not os.path.isdir(base_dir):
        logging.warning(f"Base directory '{base_dir}' not found.")
        return None
    all_subdirs = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    run_dirs = [d for d in all_subdirs if d.startswith(prefix)]
    if not run_dirs:
        logging.warning(f"No directories found with prefix '{prefix}' in '{base_dir}'.")
        return None
    try:
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
        return os.path.join(base_dir, max(run_dirs))


# --- Setup Logging ---
def setup_basic_logging():
    # ... (您的 setup_basic_logging 函数保持不变)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    root_logger.handlers.clear() 
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
    # ... (您的 argparse 参数定义保持不变，它们看起来很全面) ...
    parser.add_argument("--run_dir",type=str,default=None,help="Path to the specific run directory containing model and config (e.g., models/gnn_policy_run_...). If None, attempts to find the latest run.",)
    parser.add_argument("--episodes",type=int,default=None,help="Number of episodes to visualize (overrides config value if set).",)
    parser.add_argument("--max_steps",type=int,default=None,help="Max steps per episode (overrides config value if set).",)
    parser.add_argument("--no_animation",action="store_true",help="Disable saving the animation GIF (forces save_animation=False).",)
    parser.add_argument("--use_shortest",action="store_true",help="Force using shortest path policy instead of the loaded model.",)
    parser.add_argument("--fps",type=int,default=2,help="Frames per second for the saved animation GIF.",)
    parser.add_argument("--deterministic",action=argparse.BooleanOptionalAction,default=True,help="Use deterministic actions for the loaded model (default: True).",)
    parser.add_argument("--pause",type=float,default=0.05,help="Duration (in seconds) to pause after rendering each frame (e.g., 0.05, 0.1).",)
    args = parser.parse_args()

    # --- Determine Model and Config Path ---
    # ... (您的 target_run_dir, model_path_load, config_path_load 逻辑保持不变) ...
    target_run_dir = args.run_dir
    if target_run_dir is None:
        logging.info("No run directory specified, searching for the latest run in 'models/'...")
        target_run_dir = find_latest_run_dir()
        if target_run_dir is None:
            logging.error("Error: Could not find any valid run directories in 'models/'. Exiting.")
            exit(1)
        logging.info(f"Found latest run: {target_run_dir}")
    elif not os.path.isdir(target_run_dir):
        logging.error(f"Error: Specified run directory '{target_run_dir}' does not exist.")
        exit(1)
    model_filename = "trained_model.zip"
    config_filename = "config.json"
    model_path_load = os.path.join(target_run_dir, model_filename)
    config_path_load = os.path.join(target_run_dir, config_filename)
    run_name = os.path.basename(target_run_dir)
    results_save_dir = os.path.join("results", run_name + "_vis")

    # --- Validate File Existence ---
    # ... (您的文件存在性校验逻辑保持不变) ...
    if not args.use_shortest and not os.path.exists(model_path_load):
        logging.error(f"Error: Model file '{model_path_load}' not found and --use_shortest not specified.")
        exit(1)
    if not os.path.exists(config_path_load):
        logging.error(f"Error: Configuration file '{config_path_load}' not found.")
        exit(1)
    os.makedirs(results_save_dir, exist_ok=True)
    logging.info(f"Saving visualization results (if enabled) to: {results_save_dir}")

    # --- Load Configuration from the specific run ---
    # ... (您的配置加载逻辑保持不变) ...
    logging.info(f"Loading configuration from {config_path_load}...")
    try:
        with open(config_path_load, "r") as f:
            config = json.load(f)
        env_config = config.get("environment", {})
        vis_config_loaded = config.get("visualization", {}) 
        nn_config = config.get("neural_network", {}) 
    except Exception as e:
        logging.error(f"Error loading or parsing configuration file: {e}", exc_info=True)
        exit(1)

    # --- Reconstruct Graph ---
    # ... (您的图重建逻辑保持不变) ...
    graph_for_viz: Optional[nx.Graph] = None
    if "graph_adj" in env_config and env_config["graph_adj"] is not None:
        logging.info("Reconstructing graph from saved adjacency data...")
        try:
            graph_for_viz = json_graph.adjacency_graph(env_config["graph_adj"])
            logging.info(f"Graph reconstructed with {graph_for_viz.number_of_nodes()} nodes.")
        except Exception as e:
            logging.error(f"Error reconstructing graph: {e}", exc_info=True)
            exit(1)
    else:
        logging.error("Error: 'graph_adj' data not found or is null in loaded configuration. Cannot create environment.")
        exit(1)

    # --- Load Model (if not using shortest path) ---
    # ... (您的模型加载逻辑保持不变) ...
    loaded_model: Optional[PPO] = None
    if not args.use_shortest:
        logging.info(f"Loading pre-trained model from '{model_path_load}'...")
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available(): # For MacOS
            device = "mps"
        else:
            device = "cpu"
        logging.info(f"Using device: {device}")
        try:
            loaded_model = PPO.load(model_path_load, device=device, policy=GNNPolicy) # custom_objects might be needed if GNNPolicy isn't found by default
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Error loading model: {e}", exc_info=True)
            logging.error("Ensure the GNNPolicy class is available and matches the saved model's policy.")
            exit(1)
            
    # --- Setup Environment for Visualization ---
    logging.info("Setting up GPE environment for visualization...")
    viz_env_init_config = env_config.copy()
    viz_env_init_config["graph"] = graph_for_viz
    viz_env_init_config["render_mode"] = "human" # Essential for visualization
    viz_env_init_config["max_steps"] = (
        args.max_steps
        if args.max_steps is not None
        else vis_config_loaded.get("MAX_STEPS", 200) # Fallback to config then hardcoded default
    )
    viz_env_init_config.pop("use_preset_graph", None)
    viz_env_init_config.pop("preset_graph", None)
    viz_env_init_config.pop("graph_adj", None)
    m_layout = env_config.get("grid_m")
    n_layout = env_config.get("grid_n")
    viz_env_init_config.pop("grid_m", None)
    viz_env_init_config.pop("grid_n", None)

    try:
        # Instantiate GPE directly
        viz_env_base_gpe = GPE(**viz_env_init_config, grid_m=m_layout, grid_n=n_layout)
        # REMOVE: viz_env_wrapper_gnn instantiation
        # viz_env_wrapper_gnn = (
        #     GNNEnvWrapper(viz_env_base_gnn) if not args.use_shortest else None 
        # )
        logging.info("Base GPE visualization environment created.")
    except TypeError as e:
        logging.error(f"Error creating visualization GPE instance. Error: {e}", exc_info=True)
        logging.error(f"Arguments passed via **viz_env_init_config (after pop): {list(viz_env_init_config.keys())}")
        logging.error(f"Explicit arguments: grid_m={m_layout}, grid_n={n_layout}")
        exit()
    except Exception as e:
        logging.error(f"Unexpected error creating visualization GPE instance: {e}", exc_info=True)
        exit()

    # --- Determine Visualization Parameters ---
    # ... (您的可视化参数决定逻辑保持不变) ...
    num_episodes_to_run = (args.episodes if args.episodes is not None else vis_config_loaded.get("NUM_EPISODES", 3))
    save_animation_flag = not args.no_animation
    use_shortest_path_policy = args.use_shortest
    
    policy_desc = ("ShortestPath" if use_shortest_path_policy else f"Loaded Model (Deterministic={args.deterministic})")
    logging.info(
        f"\n--- Running Visualization ---"
        f"\n  Run Directory: {target_run_dir}"
        f"\n  Episodes: {num_episodes_to_run}"
        f"\n  Max Steps per Episode: {viz_env_init_config['max_steps']}" # Use the actual max_steps for this viz run
        f"\n  Save Animation: {save_animation_flag}"
        f"\n  Animation FPS: {args.fps}"
        f"\n  Policy: {policy_desc}"
        f"\n  Render Pause: {args.pause}"
        f"\n  Save Directory: {results_save_dir}"
    )

    try:
        visualize_policy(
            model=loaded_model,
            env=viz_env_base_gpe, # Pass the base GPE environment directly
            # wrapper_instance=None, # REMOVE: wrapper_instance argument no longer exists in visualize_policy
            num_episodes=num_episodes_to_run,
            max_steps_per_episode=viz_env_init_config["max_steps"], # Ensure this matches visualize_policy's param name
            save_animation=save_animation_flag,
            use_shortest_path=use_shortest_path_policy,
            save_dir=results_save_dir,
            animation_fps=args.fps,
            deterministic_actions=args.deterministic,
            pause_duration=args.pause,
        )
    except Exception as e:
        logging.error(f"An error occurred during visualization: {e}", exc_info=True)
    finally:
        logging.info("Closing visualization environment...")
        viz_env_base_gpe.close()
        logging.info("Visualization script finished.")