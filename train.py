import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import json
import torch
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import supersuit as ss
from datetime import datetime
import logging
import sys
from GPE.env.graph_pe import GPE
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
from networkx.readwrite import json_graph
import traceback
from torch_geometric.nn import SAGEConv
from utils.wrappers.GNNEnvWrapper import GNNEnvWrapper
from policy.GNNPolicy import GNNFeatureExtractor, GNNPolicy
from pettingzoo.utils import ParallelEnv
from gymnasium import spaces
from utils.callbacks import (
    MARLRewardCallback,
    CaptureDebugCallback,
    EscapeDebugCallback,
    DetailedDebugCallback,
)
from utils.visualization import visualize_policy


# Main execution
if __name__ == "__main__":

    # === Load and Validate Configuration ===
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        # Use print here initially or setup basic logging just for config loading outside the main logger setup
        print(
            "Configuration loaded from config.json"
        )  # Or use basic logger if preferred
    except FileNotFoundError:
        print("Error: config.json not found. Please create the configuration file.")
        exit()
    except json.JSONDecodeError:
        print("Error: config.json is not valid JSON.")
        exit()

    # Extract config sections for easier access
    env_config = config.get("environment", {})
    nn_config = config.get("neural_network", {})
    train_config = config.get("training", {})
    vis_config = config.get("visualization", {})

    # Validate required keys (example)
    required_env_keys_base = ["num_pursuers", "num_evaders", "max_steps", "allow_stay"]
    required_nn_keys = ["FEATURES_DIM", "PI_HIDDEN_DIMS", "VF_HIDDEN_DIMS"]
    required_train_keys = ["TOTAL_STEPS", "N_STEPS", "BATCH_SIZE"]
    required_vis_keys = ["MAX_STEPS", "NUM_EPISODES"]

    use_preset = env_config.get("use_preset_graph", False)
    if use_preset:
        required_env_keys = required_env_keys_base + ["preset_graph"]
        if "preset_graph" not in env_config or env_config["preset_graph"] is None:
            print(
                "Error: 'use_preset_graph' is true but 'preset_graph' key is missing or null in config.json environment section."
            )
            exit()
        if "graph_adj" not in env_config.get("preset_graph", {}):
            print(
                "Error: 'preset_graph' in config.json environment section must contain a 'graph_adj' key with the graph data."
            )
            exit()
    else:
        required_env_keys = required_env_keys_base + ["num_nodes"]

    if (
        not all(k in env_config for k in required_env_keys)
        or not all(k in nn_config for k in required_nn_keys)
        or not all(k in train_config for k in required_train_keys)
        or not all(k in vis_config for k in required_vis_keys)
    ):
        # Use print here as logging is setup later
        print("Error: Missing required keys in config.json.")
        print(f"  Need in environment: {required_env_keys}")
        print(f"  Need in neural_network: {required_nn_keys}")
        print(f"  Need in training: {required_train_keys}")
        print(f"  Need in visualization: {required_vis_keys}")
        exit()

    # === Setup Run Directory and Timestamp ===
    base_run_name = train_config.get("MODEL_SAVE_NAME", "gnn_policy_run")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{base_run_name}_{timestamp}"

    models_base_dir = "models"
    results_base_dir = "results"
    logs_base_dir = "logs"

    model_save_dir = os.path.join(models_base_dir, run_name)
    results_save_dir = os.path.join(results_base_dir, run_name)
    logs_save_dir = os.path.join(logs_base_dir, run_name)

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(results_save_dir, exist_ok=True)
    os.makedirs(logs_save_dir, exist_ok=True)
    # ==========================================

    # === Configure Logging ===
    log_file_path = os.path.join(logs_save_dir, "run.log")
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    root_logger = logging.getLogger()
    # root_logger.handlers.clear() # Optional: Uncomment if needed

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

    root_logger.setLevel(logging.INFO)
    # ==========================================

    # === Log Run Info ===
    logging.info(f"Run Name: {run_name}")  # This will print to console and file
    logging.info(f"Saving models to: {model_save_dir}")
    logging.info(f"Saving results (plots, animations) to: {results_save_dir}")
    logging.info(f"Saving logs to: {logs_save_dir}")
    logging.info("Using Configuration:")
    logging.info(json.dumps(config, indent=4))

    # --- Start Edit: Generate Grid Graph for Test ---
    # Use grid graph generation matching the environment's new logic
    target_n_nodes = env_config["num_nodes"]
    m = int(np.floor(np.sqrt(target_n_nodes)))
    n = int(np.ceil(target_n_nodes / m))
    actual_num_nodes = m * n

    logging.info(
        f"Test Script: Generating {m}x{n} grid graph ({actual_num_nodes} nodes)."
    )
    base_graph = nx.grid_2d_graph(m, n)
    base_graph = nx.convert_node_labels_to_integers(
        base_graph, first_label=0, ordering="default"
    )

    # Update num_nodes in config to actual grid size
    env_config["num_nodes"] = actual_num_nodes
    # Keep a copy for visualization if needed later (though we'll reload config)
    # graph_for_viz = base_graph # Can be removed if loading from saved config
    # --- End Edit ---

    # === Save Run Configuration ===
    # Store the serializable graph representation in the config dict
    if isinstance(base_graph, nx.Graph):
        try:
            env_config["graph_adj"] = json_graph.adjacency_data(base_graph)
            # Remove the non-serializable graph object if it was added temporarily
            if "graph" in env_config:
                del env_config["graph"]
        except Exception as e:
            logging.warning(
                f"Warning: Could not serialize graph for saving config. Error: {e}"
            )
            env_config["graph_adj"] = None  # Indicate graph couldn't be saved
    else:
        env_config["graph_adj"] = None  # No graph generated or passed directly

    # Save the *final* config used for this run to the model directory
    config_save_path = os.path.join(model_save_dir, "config.json")
    try:
        with open(config_save_path, "w") as f:
            # Use json.dump - ensure all values in config are JSON serializable
            # nx.Graph object was replaced by graph_adj dict above
            json.dump(config, f, indent=4)
        logging.info(f"Run configuration saved to {config_save_path}")
    except TypeError as e:
        logging.warning(
            f"Warning: Could not save run configuration due to non-serializable data. Error: {e}"
        )
    except Exception as e:
        logging.warning(
            f"Warning: Could not save run configuration to {config_save_path}. Error: {e}"
        )
    # ============================

    # --- Re-add graph object for GPE if needed by current training flow ---
    # The GPE init below might expect the actual graph object.
    # If env_config["graph"] was deleted above, add it back from base_graph for the *training* instance.
    # The visualization script will reconstruct it from graph_adj.
    env_config["graph"] = base_graph

    # --- Start Edit: Remove 'graph_adj' before creating GPE instance ---
    # 'graph_adj' was only needed for saving the config, not for initializing GPE
    if "graph_adj" in env_config:
        del env_config["graph_adj"]
    # --- End Edit ---

    logging.info("Creating base GPE environment...")
    # Now env_config only contains arguments expected by GPE.__init__ (plus the actual 'graph')
    base_env = GPE(**env_config, render_mode=None, grid_m=m, grid_n=n)
    logging.info("Base environment created.")

    logging.info("Wrapping environment with GNNEnvWrapper...")
    try:
        env = GNNEnvWrapper(base_env)
        logging.info("GNNEnvWrapper created.")
        logging.info(f"Wrapped env agents (initial): {env.agents}")
        logging.info(f"Wrapped env possible_agents: {env.possible_agents}")
    except Exception as e:
        logging.error(f"Error creating GNNEnvWrapper: {e}", exc_info=True)
        raise

    logging.info("Attempting PettingZoo vectorization with Supersuit...")
    try:
        # Ensure the env passed is the GNNEnvWrapper instance
        vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
        logging.info("pettingzoo_env_to_vec_env_v1 successful.")
        N_ENVS = config["training"]["N_ENVS"]
        N_CPUS = config["training"].get("N_CPUS", N_ENVS)
        vec_env = ss.concat_vec_envs_v1(
            vec_env,
            num_vec_envs=N_ENVS,
            num_cpus=N_CPUS,
            base_class="stable_baselines3",
        )
        logging.info("concat_vec_envs_v1 successful.")
        logging.info("Vectorization successful using Supersuit.")
    except Exception as e:
        logging.error(
            f"!!! PettingZoo/Supersuit vectorization failed: {e} !!!", exc_info=True
        )
        logging.error(
            "Ensure GNNEnvWrapper correctly implements the ParallelEnv interface."
        )
        raise

    # Reset the vectorized environment
    logging.info("Resetting vectorized environment...")
    try:
        obs = vec_env.reset()
        logging.info(f"Reset successful.")
        # Check the type and shape of the observation from the VecEnv
        if isinstance(obs, np.ndarray):
            logging.info(f"Observation shape after reset: {obs.shape}")
        else:
            logging.info(f"Observation type after reset: {type(obs)}")
    except Exception as e:
        logging.error(f"Error resetting vectorized environment: {e}", exc_info=True)
        raise

    policy_kwargs = {
        "features_extractor_class": GNNFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": nn_config["FEATURES_DIM"]},
        "net_arch": dict(
            pi=nn_config["PI_HIDDEN_DIMS"], vf=nn_config["VF_HIDDEN_DIMS"]
        ),
    }

    preferred_device = train_config.get("PREFERRED_DEVICE", "auto").lower()
    if preferred_device == "cuda" and torch.cuda.is_available():
        device = "cuda"
    elif preferred_device == "cpu":
        device = "cpu"
    else:  # Default 'auto' or invalid setting
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    model = PPO(
        GNNPolicy,
        vec_env,
        verbose=train_config.get("PPO_VERBOSE", 1),
        policy_kwargs=policy_kwargs,
        learning_rate=train_config.get("LEARNING_RATE", 3e-4),
        n_steps=train_config["N_STEPS"],
        batch_size=train_config["BATCH_SIZE"],
        n_epochs=train_config.get("N_EPOCHS", 10),
        gamma=train_config.get("GAMMA", 0.99),
        ent_coef=train_config.get("ENT_COEF", 0.01),
        vf_coef=train_config.get("VF_COEF", 1.0),
        gae_lambda=train_config.get("GAE_LAMBDA", 0.95),
        clip_range=train_config.get("CLIP_RANGE", 0.2),
        device=device,
        tensorboard_log=logs_save_dir,
    )

    # Set callbacks
    callback_verbose_level = train_config.get("CALLBACK_VERBOSE", 1)
    reward_callback = MARLRewardCallback(
        num_pursuers=env_config["num_pursuers"], num_evaders=env_config["num_evaders"]
    )
    capture_debug = CaptureDebugCallback(verbose=callback_verbose_level)
    escape_debug = EscapeDebugCallback(verbose=callback_verbose_level)
    detailed_debug = DetailedDebugCallback(verbose=callback_verbose_level)
    callbacks = CallbackList(
        [reward_callback, capture_debug, escape_debug, detailed_debug]
    )

    logging.info(f"Starting GNN PPO training on device: {model.device}")
    total_training_timesteps = train_config["TOTAL_STEPS"]
    logging.info(f"Training for {total_training_timesteps} timesteps...")
    model.learn(total_timesteps=total_training_timesteps, callback=callbacks)
    logging.info("Training finished.")

    # --- Updated Saving Logic ---
    # Use the specific model directory created earlier
    # The filename within the directory can be simpler now
    model_filename = "trained_model.zip"
    model_save_path = os.path.join(model_save_dir, model_filename)
    model.save(model_save_path)
    logging.info(f"Model saved to {model_save_path}")

    # --- Update Metrics Plot Saving ---
    # Assuming plot_metrics can accept a save directory or path prefix
    # You might need to modify the MARLRewardCallback.plot_metrics function
    # Example: Assuming it takes a 'save_dir' argument
    try:
        reward_callback.plot_metrics(save_dir=results_save_dir)
        logging.info(f"Metrics plots saved in {results_save_dir}")
    except TypeError:
        logging.warning(
            f"Warning: reward_callback.plot_metrics might not support 'save_dir'. Saving to default location."
        )
        reward_callback.plot_metrics()  # Fallback to original call
    # ---------------------------------

    vec_env.close()

    model_path_load = (
        model_save_path  # Path already includes the directory and filename
    )
    logging.info(f"Loading pre-trained model from '{model_path_load}'...")
    if not os.path.exists(model_path_load):
        logging.error(f"Error: Model file not found at {model_path_load}")
        exit()
    try:
        loaded_model = PPO.load(model_path_load, device=model.device, policy=GNNPolicy)
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        logging.error(
            "Try ensuring the GNNPolicy class is available in the scope during load."
        )
        exit()

    # --- Update Visualization Saving ---
    logging.info("\nVisualizing trained GNN policy...")
    # Read render mode from config for visualization env
    viz_render_mode = vis_config.get("RENDER_MODE", "human")

    # Create a clean env_config copy for visualization if needed, or ensure 'graph_adj' is removed
    viz_env_config = env_config.copy()  # Use the already cleaned env_config
    # If you re-read the config from file here, you'd need to remove 'graph_adj' again.
    # viz_env_config = config.get("environment", {}).copy() # Example if re-reading
    # viz_env_config["graph"] = base_graph # Add graph object back
    # if "graph_adj" in viz_env_config:
    #    del viz_env_config["graph_adj"]

    # Pass the cleaned config AND grid dimensions to the visualization GPE instance
    viz_env_base_gnn = GPE(
        **viz_env_config, render_mode=viz_render_mode, grid_m=m, grid_n=n
    )
    viz_env_wrapper_gnn = GNNEnvWrapper(viz_env_base_gnn)

    # Call visualize_policy directly, passing the results_save_dir
    visualize_policy(
        model=loaded_model,
        env=viz_env_base_gnn,
        wrapper_instance=viz_env_wrapper_gnn,
        num_episodes=vis_config.get("NUM_EPISODES", 3),
        max_steps=vis_config["MAX_STEPS"],
        save_animation=vis_config.get("SAVE_ANIMATION", True),
        use_shortest_path=vis_config.get("USE_SHORTEST_PATH", False),
        save_dir=results_save_dir,  # Pass the specific run's results directory
    )
    if vis_config.get("SAVE_ANIMATION", True):
        logging.info(f"Visualizations saved in {results_save_dir}")
    # ----------------------------------

    viz_env_base_gnn.close()

    metrics = reward_callback.get_metrics_summary()
    logging.info("\n--- Training Summary ---")
    for key, value in metrics.items():
        logging.info(f"{key}: {value:.4f}")

    logging.info("\n--- Callback Results ---")
    logging.info(f"Total Steps: {reward_callback.total_steps}")
    logging.info(f"Total Episodes: {reward_callback.episodes}")
    logging.info(f"Capture Count (Callback): {capture_debug.capture_count}")
    logging.info(f"Escape Count (Callback): {escape_debug.escape_count}")
    logging.info(
        f"Average Rewards: {np.mean(reward_callback.episode_rewards) if reward_callback.episode_rewards else 0}"
    )
