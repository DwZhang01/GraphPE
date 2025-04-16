import os

# Ensure necessary environment variables are set if needed
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
from datetime import datetime
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio  # For saving GIFs

# Import necessary classes and functions from your project and libraries
from GPE.env.graph_pe import GPE
from GNNEnvWrapper import (
    GNNEnvWrapper,
)  # Assuming GNNEnvWrapper.py is in the same dir or accessible
from GNNPolicy import (
    GNNPolicy,
    GNNFeatureExtractor,
)  # Assuming GNNPolicy.py is accessible
from stable_baselines3 import PPO

# --- Reused Configuration and Function from test_SB3_api.py ---

MAX_STEP = 10  # Or use the value used during training


def visualize_policy(
    model=None,
    env=None,  # Should be the base GPE environment with render_mode='human'
    wrapper_instance=None,  # Add argument for the GNNEnvWrapper instance
    num_episodes=1,
    max_steps=50,
    save_animation=True,
    use_shortest_path=True,  # Kept for consistency, but GNN path needs model
):
    """
    Visualizes the execution of the policy in the provided GPE environment.
    Uses wrapper_instance to get observations if using the trained GNN model.
    """
    print("Starting policy visualization...")
    gif_save_dir = "viz_gif"  # Make sure this directory exists or is created

    if not use_shortest_path and model is None:
        print("Error: Must provide a model if not using shortest path.")
        return
    if not use_shortest_path and wrapper_instance is None:
        print("Error: Must provide wrapper_instance if using GNN model.")
        return

    if save_animation and not os.path.exists(gif_save_dir):
        os.makedirs(gif_save_dir)
        print(f"Created directory: {gif_save_dir}")

    total_cumulative_reward = 0
    total_steps_taken = 0
    total_captures = 0

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode+1}/{num_episodes} ---")
        if save_animation:
            frames = []

        # Reset the BASE environment
        observations_base, _ = env.reset()
        # If using GNN, we need to tell the wrapper about the reset graph (for distance calc etc)
        if wrapper_instance:
            # Call wrapper's reset to update its internal state (graph, etc.)
            # Pass the initial observations from the base env if needed by wrapper logic
            # Assuming wrapper's reset doesn't strictly need obs, but updates internal graph state
            wrapper_instance.reset()

        cumulative_reward = 0
        episode_steps = 0
        captured_in_episode = False

        # Print initial state info using base env
        print(f"Episode start: {len(env.agents)} active agents")
        print(f"Pursuers: {[p for p in env.agents if p.startswith('pursuer')]}")
        print(f"Evaders: {[e for e in env.agents if e.startswith('evader')]}")

        for step in range(max_steps):
            episode_steps += 1
            actions = {}
            current_active_agents = list(
                env.agents
            )  # Get agents active in the base env for this step

            if not current_active_agents:  # Break if no agents left
                print(f"Step {step+1}: No active agents left.")
                break

            if use_shortest_path:
                for agent in current_active_agents:
                    action = env.shortest_path_action(agent)
                    if action is not None:
                        actions[agent] = action
            else:  # Use GNN Model
                wrapped_obs_dict = {
                    agent: wrapper_instance._wrap_observation(agent)
                    for agent in current_active_agents
                }

                for agent in current_active_agents:
                    obs_gnn = wrapped_obs_dict[agent]

                    # Predict action using the NumPy dictionary observation
                    with torch.no_grad():
                        action_tensor, _ = model.predict(obs_gnn, deterministic=True)
                    actions[agent] = action_tensor.item()

            print(
                f"Step {step+1}: Active agents: {len(current_active_agents)}, Actions: {actions}"
            )

            if not actions:
                print(
                    f"Step {step+1}: No actions generated (all agents might be done)."
                )
                break

            # Take a step in the BASE environment using the determined actions
            observations_base, rewards, terminations, truncations, infos = env.step(
                actions
            )

            # Check for capture in this step based on infos from the base env
            # PettingZoo infos are per-agent, check if any agent's info indicates capture
            step_capture = any(
                agent_info.get("capture", False) for agent_info in infos.values()
            )
            if step_capture:
                captured_in_episode = True

            # Render the current state using the base env's render method
            env.render()

            # Save frame for animation
            if save_animation:
                fig = plt.gcf()
                fig.canvas.draw()
                image_rgba = np.array(fig.canvas.buffer_rgba())
                image_rgb = image_rgba[:, :, :3]
                frames.append(image_rgb)

            # Track rewards
            step_reward = sum(
                rewards.values()
            )  # Sum rewards for all agents in this step
            cumulative_reward += step_reward

            print(
                f"Step {step+1}: Reward = {step_reward:.2f}, Cumulative = {cumulative_reward:.2f}"
            )
            plt.pause(0.5)

            # Check if episode is done
            if not env.agents:  # Episode ends if no agents are left
                print(
                    f"Episode ended naturally at step {step+1} - All agents terminated."
                )
                break
            # Also check explicit termination/truncation flags if needed, though agent list check is often sufficient
            # if any(terminations.values()) or any(truncations.values()):
            #    print(f"Episode ended via termination/truncation flag at step {step+1}.")
            #    break

        # --- End of Episode ---
        total_steps_taken += episode_steps
        total_cumulative_reward += cumulative_reward
        if captured_in_episode:
            # Get final count from env attribute after episode ends
            total_captures += len(env.captured_evaders)

        print(
            f"Episode {episode+1} finished after {episode_steps} steps. Cumulative reward: {cumulative_reward:.2f}"
        )

        # Save animation
        if save_animation and frames:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                policy_type = "GNN" if not use_shortest_path else "ShortestPath"
                filename = f"GPE_{policy_type}_{timestamp}_episode{episode+1}.gif"
                filepath = os.path.join(gif_save_dir, filename)
                print(f"Saving animation to {filepath}...")
                imageio.mimsave(filepath, frames, fps=2)  # Adjust fps as needed
                print("Animation saved.")
            except ImportError:
                print("Could not save animation: imageio package not found.")
            except Exception as e:
                print(f"Error saving animation: {e}")

        time.sleep(1)  # Pause slightly between episodes

    print("\n--- Visualization Summary ---")
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Steps per Episode: {total_steps_taken / num_episodes:.2f}")
    print(
        f"Average Cumulative Reward per Episode: {total_cumulative_reward / num_episodes:.2f}"
    )
    print(
        f"Total Captures across all episodes: {total_captures}"
    )  # Note: This counts final captured state
    print("Visualization complete!")


if __name__ == "__main__":
    # --- Environment Configuration (Should match training) ---
    env_config = {
        "num_nodes": 50,
        "num_pursuers": 2,
        "num_evaders": 1,
        "max_steps": 50,  # Use MAX_STEP constant
        "p_act": 1,
        "capture_reward_pursuer": 20.0,
        "capture_reward_evader": -20.0,
        "escape_reward_evader": 100.0,
        "escape_reward_pursuer": -100.0,
        "stay_penalty": -0.1,
    }

    # --- Graph Generation (Consistent with training) ---
    target_n_nodes = env_config["num_nodes"]
    m = int(np.floor(np.sqrt(target_n_nodes)))
    n = int(np.ceil(target_n_nodes / m))
    actual_num_nodes = m * n
    print(f"Visualization: Generating {m}x{n} grid graph ({actual_num_nodes} nodes).")
    base_graph = nx.grid_2d_graph(m, n)
    base_graph = nx.convert_node_labels_to_integers(
        base_graph, first_label=0, ordering="default"
    )
    env_config["num_nodes"] = actual_num_nodes
    env_config["graph"] = base_graph  # Pass the graph instance

    # --- Load Model ---
    model_save_path = "gnn_policy"  # Assuming this is where the model is saved
    model_path_load = f"{model_save_path}.zip"
    print(f"Loading pre-trained model from '{model_path_load}'...")

    if not os.path.exists(model_path_load):
        print(f"Error: Model file not found at {model_path_load}")
        exit()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Need to specify the custom policy when loading
        loaded_model = PPO.load(
            model_path_load,
            device=device,
            custom_objects={
                "policy_class": GNNPolicy
            },  # Make sure GNNPolicy is imported
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Ensure the GNNPolicy class is available in the scope.")
        exit()

    # --- Setup Environment for Visualization ---
    print("\nSetting up environment for GNN Policy visualization...")
    # Create a base GPE env with rendering enabled
    viz_env_base_gnn = GPE(
        **env_config,  # Use the config with the specific graph
        render_mode="human",  # Enable rendering
    )
    # Create the wrapper linked to this base env
    viz_env_wrapper_gnn = GNNEnvWrapper(viz_env_base_gnn)

    # --- Run Visualization ---
    visualize_policy(
        model=loaded_model,
        env=viz_env_base_gnn,  # Base env for rendering/stepping
        wrapper_instance=viz_env_wrapper_gnn,  # Wrapper for observations
        num_episodes=5,  # Visualize more episodes
        max_steps=MAX_STEP,
        save_animation=True,
        use_shortest_path=False,  # Use the loaded GNN model
    )

    # --- Clean up ---
    viz_env_base_gnn.close()

    print("\nVisualization script finished.")
