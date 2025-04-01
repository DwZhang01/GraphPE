import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import supersuit as ss
from GPE.env.graph_pe import GPE
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
import time


# Create a callback to track rewards
class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []
        self.episode_rewards = 0
        self.episodes = 0
        self.episode_reward_history = []

    def _on_step(self):
        # Get the reward from the last step
        reward = self.locals["rewards"][0]
        self.episode_rewards += reward
        self.rewards.append(reward)

        # Check if episode is done
        done = self.locals["dones"][0]
        if done:
            self.episodes += 1
            self.episode_reward_history.append(self.episode_rewards)
            self.episode_rewards = 0

        return True

    def plot_rewards(self):
        # Plot cumulative reward
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(self.rewards))
        plt.title("Cumulative Reward")
        plt.xlabel("Steps")
        plt.ylabel("Reward")

        # Plot episode rewards
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_reward_history)
        plt.title("Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")

        plt.tight_layout()
        plt.savefig("reward_history.png")
        plt.show()


def visualize_policy(model, env, num_episodes=3, max_steps=50):
    """
    Visualizes the execution of the trained policy in the provided GPE environment.

    Args:
        model: Trained PPO model
        env: GPE environment instance with render_mode="human"
        num_episodes: Number of episodes to visualize
        max_steps: Maximum steps per episode
    """
    print("Starting policy visualization...")

    # Run several episodes for visualization
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode+1}/{num_episodes} ---")

        # Reset environment
        observations = env.reset()
        cumulative_reward = 0

        for step in range(max_steps):
            # Determine actions for all agents using the trained policy
            actions = {}
            for agent, obs in observations.items():
                # Convert observation to model's expected format
                obs_array = np.array(obs).reshape(1, -1)
                # Get action from model
                action, _ = model.predict(obs_array, deterministic=True)
                actions[agent] = action.item()

            # Take a step in the environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Render the current state (the environment will handle the visualization)
            env.render()

            # Track rewards
            episode_reward = sum(rewards.values())
            cumulative_reward += episode_reward

            # Display step info
            print(
                f"Step {step+1}: Reward = {episode_reward:.2f}, Cumulative = {cumulative_reward:.2f}"
            )

            # Check if episode is done
            if any(terminations.values()) or any(truncations.values()):
                if any(terminations.values()):
                    captured = len(env.captured_evaders)
                    print(
                        f"Episode ended - Evaders captured: {captured}/{env.num_evaders}"
                    )
                else:
                    print(f"Episode ended - Maximum steps reached")
                break

        # Small pause between episodes
        print(
            f"Episode {episode+1} complete. Cumulative reward: {cumulative_reward:.2f}"
        )
        time.sleep(2)  # Pause between episodes

    print("Visualization complete!")


# Main execution
if __name__ == "__main__":
    # Environment configuration
    env_config = {
        "num_nodes": 20,
        "num_edges": 40,
        "num_pursuers": 2,
        "num_evaders": 1,
        "capture_distance": 1,
        "required_captors": 1,
        "seed": 42,
    }

    # Create training environment
    training_env = GPE(
        **env_config,
        max_steps=2000,
        render_mode=None,  # No rendering during training
    )

    # Wrap the training environment for Stable Baselines3
    vec_env = ss.pettingzoo_env_to_vec_env_v1(training_env)
    vec_env = ss.concat_vec_envs_v1(
        vec_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3"
    )
    vec_env.reset()

    # Create the callback to track rewards
    reward_callback = RewardCallback()

    # Create PPO model
    model = PPO(
        MlpPolicy,
        vec_env,
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
        batch_size=256,
        device="cpu",
    )

    # Train the model with the callback to track rewards
    model.learn(total_timesteps=200000, callback=reward_callback)

    # Save the policy
    model.save("policy")

    # Plot the reward history
    reward_callback.plot_rewards()

    # Close the training environment
    vec_env.close()

    # Create a visualization environment (separate from training env)
    # This needs to be a direct GPE instance with human rendering enabled
    viz_env = GPE(
        **env_config,
        max_steps=50,  # Shorter episodes for visualization
        render_mode="human",  # Enable rendering
    )

    # Visualize the policy execution
    print("\nVisualizing trained policy...")
    visualize_policy(model, viz_env, num_episodes=3, max_steps=200)

    # Close the visualization environment
    viz_env.close()
