import os
import time
from datetime import datetime
import numpy as np

import matplotlib

matplotlib.use("Agg")  # Force non-interactive backend for saving files reliably
import matplotlib.pyplot as plt
import torch
import imageio  # Make sure imageio is installed
import traceback

# Type hinting imports
from typing import Optional, Dict, Any, List, Union, Tuple
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv  # For potential future type hinting
from GPE.env.graph_pe import GPE

# Import GNNEnvWrapper using relative path
from .wrappers.GNNEnvWrapper import (
    GNNEnvWrapper,
)  # Assuming wrappers is a subpackage or sibling dir


def visualize_policy(
    model: Optional[BaseAlgorithm] = None,
    env: Optional[GPE] = None,
    wrapper_instance: Optional[GNNEnvWrapper] = None,
    num_episodes: int = 1,
    max_steps: int = 100,
    save_animation: bool = True,
    use_shortest_path: bool = False,  # Default to using model if available
    save_dir: Optional[str] = None,
    animation_fps: int = 2,
    deterministic_actions: bool = True,
) -> None:
    """
    Visualizes the execution of a policy (trained model or shortest path) in the GPE environment.

    Requires the base GPE environment instance ('env') configured with render_mode='human'.
    If using a trained model (`use_shortest_path=False`), both 'model' and the corresponding
    'wrapper_instance' linked to 'env' must be provided.

    Args:
        model: Trained SB3 model (e.g., PPO). Required if use_shortest_path=False.
        env: Base GPE environment instance (must have render_mode="human").
        wrapper_instance: GNNEnvWrapper instance linked to 'env'. Required if use_shortest_path=False.
        num_episodes: Number of episodes to visualize.
        max_steps: Maximum steps per episode.
        save_animation: Whether to save the visualization as a GIF.
        use_shortest_path: If True, uses env.shortest_path_action() for all agents.
                           If False, uses model.predict().
        save_dir: Directory to save the animation GIF. If None, defaults to "results/viz_gifs".
        animation_fps: Frames per second for the saved GIF.
        deterministic_actions: Whether the model should predict actions deterministically.
    """
    print("\n--- Starting Policy Visualization ---")

    # --- Input Validation ---
    if env is None:
        print("Error: Must provide the base 'env' (GPE instance) argument.")
        return
    if not use_shortest_path and model is None:
        print("Error: Must provide a 'model' if 'use_shortest_path' is False.")
        return
    if not use_shortest_path and wrapper_instance is None:
        print(
            "Error: Must provide 'wrapper_instance' if 'use_shortest_path' is False (using GNN model)."
        )
        return
    # Check render mode AFTER env is confirmed not None
    if env.render_mode != "human":
        print(
            "Warning: Environment render_mode is not 'human'. Visualization might not display or save correctly."
        )
        # Decide whether to proceed or return
        # return # Option: exit if render_mode is wrong

    # --- Setup Save Directory ---
    resolved_save_dir: Optional[str] = None
    if save_animation:
        resolved_save_dir = save_dir if save_dir is not None else "results/viz_gifs"
        if not os.path.exists(resolved_save_dir):
            try:
                os.makedirs(resolved_save_dir)
                print(f"Created directory for GIFs: {resolved_save_dir}")
            except OSError as e:
                print(
                    f"Error creating directory {resolved_save_dir}: {e}. Animation saving disabled."
                )
                save_animation = False  # Disable saving

    # --- Statistics Tracking ---
    total_cumulative_reward = 0.0
    total_steps_taken = 0
    total_captures = 0  # Based on environment's captured_evaders set
    total_escapes = 0  # Track escapes if needed

    # --- Episode Loop ---
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode+1}/{num_episodes} ---")
        frames: List[np.ndarray] = []  # Store frames for GIF

        # Reset the base environment
        # PettingZoo ParallelEnv typically returns (obs_dict, info_dict)
        try:
            # Seed argument might be needed depending on Gymnasium/PettingZoo version
            # reset_output = env.reset(seed=...)
            reset_output = env.reset()
            if isinstance(reset_output, tuple) and len(reset_output) == 2:
                observations_base, infos_base = (
                    reset_output  # Initial obs/info from base env
                )
            else:
                # Fallback if reset doesn't return a tuple (older versions?)
                observations_base = reset_output
                infos_base = {agent: {} for agent in env.agents}  # Empty info dict
            print(f"Environment reset. Initial active agents: {env.agents}")
        except Exception as e:
            print(f"Error resetting environment: {e}")
            traceback.print_exc()
            continue  # Skip to next episode

        # If using GNN, reset the wrapper (important for its internal state)
        # Wrapper reset should ideally use the observations from the base env reset
        if wrapper_instance:
            try:
                # Wrapper might need obs or just resets internal state
                # wrapper_instance.reset(seed=..., options=...)
                wrapper_instance.reset()
            except Exception as e:
                print(f"Error resetting wrapper instance: {e}")
                traceback.print_exc()
                # Decide if this is fatal for the episode
                continue

        # Episode specific tracking
        cumulative_reward_episode = 0.0
        episode_steps = 0
        captured_in_episode_set = set()  # Track captures accurately within the episode

        # --- Step Loop ---
        for step in range(max_steps):
            current_active_agents = list(
                env.agents
            )  # Get agents active at the start of the step
            if not current_active_agents:
                print(f"Step {step+1}: No active agents left. Ending episode early.")
                break

            actions: Dict[str, int] = {}  # Actions for the base env step

            # --- Action Selection ---
            try:
                if use_shortest_path:
                    # Use env's built-in shortest path logic
                    for agent in current_active_agents:
                        action = env.shortest_path_action(agent)
                        if action is not None:
                            actions[agent] = action
                        else:
                            # Handle cases where shortest path fails (e.g., stay put)
                            actions[agent] = env.agent_positions[agent]  # Example: stay
                else:
                    # Use the provided GNN model and wrapper
                    # Get observations for ALL currently active agents via the wrapper
                    wrapped_obs_dict: Dict[str, Any] = {}
                    for agent in current_active_agents:
                        try:
                            wrapped_obs_dict[agent] = (
                                wrapper_instance._wrap_observation(agent)
                            )
                        except AttributeError:
                            print(
                                f"Error: Method '_wrap_observation' not found in GNNEnvWrapper for agent {agent}. Check wrapper implementation."
                            )
                            continue  # Skip agent
                        except Exception as e:
                            print(
                                f"Error getting wrapped observation for {agent}: {e}. Skipping agent."
                            )
                            continue  # Skip agent for now

                    # Predict action for each agent using its wrapped observation
                    for agent in current_active_agents:
                        if agent in wrapped_obs_dict:
                            obs_gnn = wrapped_obs_dict[agent]
                            formatted_obs = obs_gnn
                            with torch.no_grad():
                                action_tensor, _ = model.predict(
                                    formatted_obs, deterministic=deterministic_actions
                                )
                            actions[agent] = action_tensor.item()
                        # Else: agent wasn't in wrapped_obs_dict due to error, no action generated

            except Exception as e:
                print(f"Error during action selection at step {step+1}: {e}")
                traceback.print_exc()
                break  # End episode if action selection fails critically

            if len(actions) != len(current_active_agents):
                print(
                    f"Warning: Number of actions ({len(actions)}) does not match number of active agents ({len(current_active_agents)}) at step {step+1}."
                )
                # Need to decide how env.step handles partial actions - often throws error
                # For visualization, maybe end episode here
                # break

            if not actions:
                print(
                    f"Step {step+1}: No valid actions generated for any agent. Ending episode."
                )
                break

            # --- Environment Step ---
            try:
                # Base environment step requires actions for all agents it expects
                observations_base, rewards, terminations, truncations, infos = env.step(
                    actions
                )
                episode_steps += 1  # Count successful steps
            except Exception as e:
                print(
                    f"Error during env.step at step {step+1} with actions {actions}: {e}"
                )
                traceback.print_exc()
                break  # End episode if step fails

            # --- Process Step Results ---
            step_reward = sum(rewards.values())
            cumulative_reward_episode += step_reward

            # Update capture tracking for the episode
            current_captured_this_step = env.captured_evaders
            newly_captured = current_captured_this_step - captured_in_episode_set
            if newly_captured:
                captured_in_episode_set.update(newly_captured)
                print(
                    f"  Capture detected! Newly captured: {newly_captured}. Total captured this ep: {len(captured_in_episode_set)}"
                )

            # Check for escape event (using the flag set in env)
            # Check info for the *first* agent (common pattern for VecEnv info flags)
            # Note: Assumes the 'escape_event' flag is reliably placed in infos[0] by the env logic
            first_agent_info = (
                infos.get(current_active_agents[0], {}) if current_active_agents else {}
            )
            if first_agent_info.get("escape_event", False):
                # Crude escape tracking for visualization summary
                # This might multi-count if flag persists across steps where agent is still active
                total_escapes += 1
                print("  Escape event detected in step info.")

            print(
                f"Step {step+1}/{max_steps}: Agents={len(env.agents)}, Reward={step_reward:.2f}, CumRew={cumulative_reward_episode:.2f}"
            )

            # --- Render and Capture Frame ---
            try:
                env.render()  # Call the environment's render method
                if save_animation:
                    fig = (
                        plt.gcf()
                    )  # Get the current figure created/updated by env.render()
                    # Robust frame capture logic (attempts to handle potential backend inconsistencies)
                    fig.canvas.draw()
                    buf = fig.canvas.buffer_rgba()
                    image_flat = np.frombuffer(buf, dtype=np.uint8)
                    logical_width, logical_height = fig.canvas.get_width_height()
                    actual_size = image_flat.size
                    expected_size = logical_height * logical_width * 4

                    actual_height, actual_width = 0, 0
                    if actual_size == expected_size and expected_size > 0:
                        actual_height, actual_width = logical_height, logical_width
                    elif (
                        actual_size > 0
                        and actual_size % 4 == 0
                        and logical_height > 0
                        and logical_width > 0
                    ):
                        # Attempt inference if sizes mismatch but buffer looks valid
                        num_pixels = actual_size // 4
                        aspect_ratio = logical_width / logical_height
                        inferred_height_f = np.sqrt(num_pixels / aspect_ratio)
                        inferred_width_f = inferred_height_f * aspect_ratio
                        if np.isclose(
                            inferred_height_f, round(inferred_height_f)
                        ) and np.isclose(inferred_width_f, round(inferred_width_f)):
                            inferred_height_i = int(round(inferred_height_f))
                            inferred_width_i = int(round(inferred_width_f))
                            if inferred_height_i * inferred_width_i * 4 == actual_size:
                                actual_height, actual_width = (
                                    inferred_height_i,
                                    inferred_width_i,
                                )
                                # print(f"  Debug: Inferred frame dimensions {actual_height}x{actual_width}") # Optional debug
                            else:  # Mismatch even after inference
                                print(
                                    f"  Warning: Could not match inferred frame dimensions ({inferred_height_i}x{inferred_width_i}) to buffer size {actual_size}. Skipping frame."
                                )
                                continue
                        else:  # Non-integer inference
                            print(
                                f"  Warning: Could not infer integer frame dimensions. Skipping frame."
                            )
                            continue
                    else:  # Invalid buffer or dimensions
                        print(
                            f"  Warning: Invalid frame buffer or dimensions (Size: {actual_size}, Expected: {expected_size}). Skipping frame."
                        )
                        continue

                    # Reshape and store if dimensions are valid
                    image_rgba = image_flat.reshape(actual_height, actual_width, 4)
                    image_rgb = image_rgba[:, :, :3]  # Drop alpha
                    frames.append(image_rgb)

            except Exception as e:
                print(f"Error during rendering or frame capture: {e}")
                traceback.print_exc()
                # Continue without this frame if possible

            plt.pause(0.1)  # Small pause to allow viewing

            # --- Check for Termination/Truncation ---
            # Check if ALL remaining agents are done (terminated or truncated)
            if not env.agents:  # PettingZoo removes done agents
                print(f"Episode ended naturally at step {step+1} (no agents left).")
                break
            # Alternative check if env doesn't remove agents immediately:
            # all_done = all(terminations.get(agent, False) or truncations.get(agent, False) for agent in current_active_agents)
            # if all_done:
            #     print(f"Episode ended at step {step+1} (all agents terminated or truncated).")
            #     break

        # --- End of Episode ---
        total_steps_taken += episode_steps
        total_cumulative_reward += cumulative_reward_episode
        num_captured_this_episode = len(captured_in_episode_set)
        total_captures += num_captured_this_episode  # Use tracked set for accuracy

        print(
            f"Episode {episode+1} finished: Steps={episode_steps}, Reward={cumulative_reward_episode:.2f}, Captures={num_captured_this_episode}"
        )

        # --- Save Animation ---
        if save_animation and frames and resolved_save_dir:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                policy_type = "GNN" if not use_shortest_path else "ShortestPath"
                filename = f"GPE_Viz_{policy_type}_Ep{episode+1}_{timestamp}.gif"
                filepath = os.path.join(resolved_save_dir, filename)
                print(f"Saving animation ({len(frames)} frames) to {filepath}...")
                imageio.mimsave(filepath, frames, fps=animation_fps)
                print("Animation saved.")
            except ImportError:
                print(
                    "Error: Could not save animation - 'imageio' package not found. Please install it (`pip install imageio`)."
                )
            except Exception as e:
                print(f"Error saving animation: {e}")
                traceback.print_exc()

        if num_episodes > 1:
            time.sleep(0.5)  # Short pause between episodes if visualizing multiple

    # --- Final Summary ---
    print("\n--- Visualization Summary ---")
    print(f"Total Episodes Run: {num_episodes}")
    avg_steps = total_steps_taken / num_episodes if num_episodes > 0 else 0
    avg_reward = total_cumulative_reward / num_episodes if num_episodes > 0 else 0
    avg_captures = total_captures / num_episodes if num_episodes > 0 else 0
    print(f"Average Steps per Episode: {avg_steps:.2f}")
    print(f"Average Cumulative Reward per Episode: {avg_reward:.2f}")
    print(f"Average Captures per Episode: {avg_captures:.2f}")
    # print(f"Total Escapes Detected (approx): {total_escapes}") # Optional: Add escape summary
    print("Visualization complete!")
