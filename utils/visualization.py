import os
import time
from datetime import datetime
import numpy as np
import matplotlib
# matplotlib.use("Agg") # Consider uncommenting if running in a headless environment and saving files is primary
import matplotlib.pyplot as plt
import torch
import imageio # Make sure imageio is installed
import traceback

# Type hinting imports
from typing import Optional, Dict, Any, List
from stable_baselines3.common.base_class import BaseAlgorithm
# from stable_baselines3.common.vec_env import VecEnv # Not directly used here for 'env' type
from GPE.env.graph_pe import GPE # Assuming GPE is in this path

def visualize_policy(
    model: Optional[BaseAlgorithm] = None,
    env: Optional[GPE] = None,
    # wrapper_instance parameter is removed
    num_episodes: int = 1,
    max_steps_per_episode: int = 100, # Renamed for clarity
    save_animation: bool = True,
    use_shortest_path: bool = False,
    save_dir: Optional[str] = None,
    animation_fps: int = 2,
    deterministic_actions: bool = True,
    pause_duration: float = 0.2,
) -> None:
    """
    Visualizes the execution of a policy (trained model or shortest path) in the GPE environment.

    Requires the base GPE environment instance ('env') configured with render_mode='human'
    or another mode that its render() method supports for frame capture.
    If using a trained model (`use_shortest_path=False`), 'model' must be provided.

    Args:
        model: Trained SB3 model (e.g., PPO). Required if use_shortest_path=False.
        env: Base GPE environment instance.
        num_episodes: Number of episodes to visualize.
        max_steps_per_episode: Maximum steps per episode.
        save_animation: Whether to save the visualization as a GIF.
        use_shortest_path: If True, uses env.shortest_path_action() for all agents.
                           If False, uses model.predict().
        save_dir: Directory to save the animation GIF. If None, defaults to "results/viz_gifs".
        animation_fps: Frames per second for the saved GIF.
        deterministic_actions: Whether the model should predict actions deterministically.
        pause_duration: Duration (in seconds) for plt.pause() after rendering each frame.
    """
    print("\n--- Starting Policy Visualization ---")

    # --- Input Validation ---
    if env is None:
        print("Error: Must provide the base 'env' (GPE instance) argument.")
        return
    if not use_shortest_path and model is None:
        print("Error: Must provide a 'model' if 'use_shortest_path' is False.")
        return
    
    # It's good practice for env.render() to work for frame capture even if not 'human'
    # but 'human' mode is typical for plt.pause() interaction.
    if env.render_mode is None and save_animation:
         print("Warning: env.render_mode is None. Frame capture for GIF might fail if render() doesn't produce plottable output.")
    elif env.render_mode != "human" and pause_duration > 0:
        print(f"Warning: env.render_mode is '{env.render_mode}', not 'human'. Interactive plt.pause() might not behave as expected.")


    # --- Setup Save Directory ---
    resolved_save_dir: Optional[str] = None
    if save_animation:
        resolved_save_dir = save_dir if save_dir is not None else "results/viz_gifs"
        if not os.path.exists(resolved_save_dir):
            try:
                os.makedirs(resolved_save_dir)
                print(f"Created directory for GIFs: {resolved_save_dir}")
            except OSError as e:
                print(f"Error creating directory {resolved_save_dir}: {e}. Animation saving disabled.")
                save_animation = False

    # --- Statistics Tracking ---
    total_cumulative_reward = 0.0
    total_steps_taken = 0
    total_captures = 0
    total_escapes = 0 # Assuming GPE's info might contain "escape_event"

    # --- Episode Loop ---
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        frames: List[np.ndarray] = []
        episode_reward = 0.0
        episode_steps = 0
        
        # Store initial captured set to find newly captured ones later
        initial_captured_count = len(env.captured_evaders) if hasattr(env, 'captured_evaders') else 0

        try:
            # GPE.reset() returns (observations_dict, infos_dict)
            observations_dict, infos_dict = env.reset()
            print(f"Environment reset. Initial active agents: {env.agents}")
        except Exception as e:
            print(f"Error resetting GPE environment: {e}")
            traceback.print_exc()
            continue # Skip to next episode

        # --- Step Loop ---
        for step_num in range(max_steps_per_episode):
            current_active_agents = list(env.agents) # Agents active at the start of this step
            if not current_active_agents:
                print(f"Step {step_num + 1}: No active agents. Ending episode.")
                break

            actions_to_env: Dict[str, int] = {}

            # --- Action Selection ---
            try:
                if use_shortest_path:
                    if not hasattr(env, 'shortest_path_action'):
                        print("Error: env does not have 'shortest_path_action' method. Cannot use this mode.")
                        break # out of step loop for this episode
                    for agent_id in current_active_agents:
                        action = env.shortest_path_action(agent_id)
                        if action is not None:
                            actions_to_env[agent_id] = action
                        else: # Fallback if shortest path fails (e.g., agent is stuck or at target)
                            actions_to_env[agent_id] = env.agent_positions.get(agent_id, 0) # Stay or default
                else: # Use the SB3 model
                    for agent_id in current_active_agents:
                        if agent_id in observations_dict:
                            agent_observation = observations_dict[agent_id]
                            # GPE's observation is Dict{"node_features", "action_mask"}
                            # SB3 model.predict expects this structure if GNNFeatureExtractor is set up for it.
                            with torch.no_grad():
                                action_tensor, _ = model.predict(
                                    agent_observation, # Pass the dict observation for this agent
                                    deterministic=deterministic_actions
                                )
                            actions_to_env[agent_id] = action_tensor.item()
                        else:
                            print(f"Warning: Agent {agent_id} active but no observation found in observations_dict. Skipping action.")
                
                if not actions_to_env or len(actions_to_env) != len(current_active_agents):
                    # This might happen if an agent was skipped above or if shortest_path failed for all
                    print(f"Step {step_num + 1}: Not all active agents have actions. Active: {len(current_active_agents)}, Actions: {len(actions_to_env)}. Ending episode.")
                    # It's safer to break if actions_to_env is incomplete, as env.step() expects actions for all its current env.agents
                    if len(actions_to_env) < len(current_active_agents):
                         break


            except Exception as e:
                print(f"Error during action selection at step {step_num + 1}: {e}")
                traceback.print_exc()
                break # End episode

            # --- Environment Step ---
            if not actions_to_env: # Should have been caught above, but as a safeguard
                print(f"Step {step_num + 1}: No actions to send to env.step(). Ending episode.")
                break

            try:
                next_observations_dict, rewards_dict, terminations_dict, truncations_dict, infos_dict = env.step(actions_to_env)
                episode_steps += 1
            except Exception as e:
                print(f"Error during env.step at step {step_num + 1} with actions {actions_to_env}: {e}")
                traceback.print_exc()
                break

            # --- Process Step Results ---
            step_reward_sum = sum(rewards_dict.values())
            episode_reward += step_reward_sum
            observations_dict = next_observations_dict # Update for next iteration

            # Track captures for this episode based on change in env's set
            if hasattr(env, 'captured_evaders'):
                current_episode_captures = len(env.captured_evaders) - initial_captured_count
            else:
                current_episode_captures = 0 # Fallback if attribute not present

            # Crude escape tracking (assuming "escape_event" is a global flag in one info dict)
            # A better approach would be for GPE to directly track and expose escaped evaders.
            escaped_this_step = False
            for agent_id in current_active_agents: # Check all infos for this step
                if infos_dict.get(agent_id, {}).get("escape_event", False):
                    escaped_this_step = True
                    break
            if escaped_this_step:
                total_escapes += 1 # Count occurrences of the event flag

            print(f"Step {step_num + 1}/{max_steps_per_episode}: Agents left={len(env.agents)}, StepRew={step_reward_sum:.2f}, EpRew={episode_reward:.2f}, EpCaps={current_episode_captures}")

            # --- Render and Capture Frame ---
            try:
                # GPE.render() should ideally return a figure if render_mode allows frame capture
                fig_or_frame = env.render() 
                
                if fig_or_frame is not None and env.render_mode == "human": # For interactive display with pause
                    plt.pause(pause_duration)
                
                if save_animation:
                    if isinstance(fig_or_frame, np.ndarray): # If render() directly returns an RGB array
                        frames.append(fig_or_frame)
                    elif hasattr(fig_or_frame, 'canvas'): # If it's a matplotlib figure
                        fig_or_frame.canvas.draw()
                        image_rgba = np.array(fig_or_frame.canvas.buffer_rgba())
                        if image_rgba.size > 0:
                            frames.append(image_rgba[:, :, :3]) # RGB
                        else: print(f"Warning: Captured empty Matplotlib frame buffer at step {step_num + 1}.")
                    # else: No valid frame/figure returned for saving
                        
            except Exception as e:
                print(f"Error during rendering/frame capture: {e}")
                traceback.print_exc()

            # --- Check for Termination/Truncation ---
            if not env.agents: # If all agents are done (PettingZoo removes them)
                print(f"Episode ended at step {step_num + 1} (no active agents left).")
                break
            # Or check if all original agents for this step are now terminated/truncated
            all_done_for_step_agents = all(
                terminations_dict.get(aid, False) or truncations_dict.get(aid, False) 
                for aid in current_active_agents
            )
            if all_done_for_step_agents and len(current_active_agents) > 0 : # Ensure not an empty list that all() returns True for
                print(f"Episode ended at step {step_num + 1} (all initially active agents are done).")
                break


        # --- End of Episode ---
        total_steps_taken += episode_steps
        total_cumulative_reward += episode_reward
        if hasattr(env, 'captured_evaders'): # Final capture count for the episode
            final_episode_captures = len(env.captured_evaders) - initial_captured_count
            total_captures += final_episode_captures
        else:
            final_episode_captures = 0


        print(f"Episode {episode + 1} finished: Steps={episode_steps}, Reward={episode_reward:.2f}, Captures this ep={final_episode_captures}")

        if save_animation and frames and resolved_save_dir:
            filepath = None
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                policy_type_str = "GNNModel" if not use_shortest_path else "ShortestPath"
                filename = f"GPE_Viz_{policy_type_str}_{timestamp}_Ep{episode + 1}.gif"
                filepath = os.path.join(resolved_save_dir, filename)
                
                frame_duration_ms = (1.0 / animation_fps) * 1000 # imageio duration is in ms for some writers
                imageio.mimsave(filepath, frames, duration=frame_duration_ms, loop=0) # loop=0 for infinite loop
                print(f"Animation saved to {filepath}")
            except ImportError:
                print("Error: Could not save animation - 'imageio' package not found. Install with `pip install imageio imageio[ffmpeg]`.")
            except Exception as e:
                error_loc = f" at path '{filepath}'" if filepath else ""
                print(f"Error saving animation{error_loc}: {e}")
                traceback.print_exc()
        
        if num_episodes > 1 and episode < num_episodes - 1:
            time.sleep(0.5) # Pause between episodes

    # --- Final Summary ---
    print("\n--- Visualization Summary ---")
    print(f"Total Episodes Run: {num_episodes}")
    avg_steps = total_steps_taken / num_episodes if num_episodes > 0 else 0
    avg_reward = total_cumulative_reward / num_episodes if num_episodes > 0 else 0
    avg_captures = total_captures / num_episodes if num_episodes > 0 else 0
    print(f"Average Steps per Episode: {avg_steps:.2f}")
    print(f"Average Cumulative Reward per Episode: {avg_reward:.2f}")
    print(f"Average Captures per Episode: {avg_captures:.2f}")
    if total_escapes > 0: # Only print if escapes were detected
        print(f"Total Escape Events Detected across episodes (approx): {total_escapes}")
    print("Visualization complete!")