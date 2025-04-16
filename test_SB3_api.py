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
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EventCallback
import time
import os
from datetime import datetime

from torch_geometric.nn import SAGEConv
import torch

from GNNEnvWrapper import GNNEnvWrapper
from GNNPolicy import GNNFeatureExtractor, GNNPolicy

# Add this import
from pettingzoo.utils import ParallelEnv
from gymnasium import (
    spaces,
)  # Import spaces for GNNEnvWrapper definition below if needed

# aggragation methods: mean, add, max


MAX_STEP = 50


# Create a callback to track rewards
class MARLRewardCallback(BaseCallback):
    def __init__(self, num_pursuers, num_evaders, verbose=0):
        super(MARLRewardCallback, self).__init__(verbose)
        self.num_pursuers = num_pursuers
        self.num_evaders = num_evaders

        # 总体奖励跟踪
        self.total_rewards = []  # 每步的总奖励
        self.episode_total_reward = 0  # 当前episode的累积总奖励

        # Episode 历史数据
        self.episodes = 0
        self.episode_lengths = []
        self.episode_rewards = []  # 每个episode的总奖励
        self.episode_capture_counts = []  # 每个episode的捕获数量

        # 性能指标
        self.total_capture_count = 0
        self.episode_capture_count = 0
        self.total_steps = 0
        self.current_episode_steps = 0

    def _on_step(self):
        self.total_steps += 1
        self.current_episode_steps += 1

        # 在向量化环境中，rewards是一个numpy数组，而不是字典
        # 假设rewards[0]是所有智能体本步的总奖励
        reward = self.locals["rewards"][0]  # 获取标量奖励
        self.total_rewards.append(reward)
        self.episode_total_reward += reward

        # 获取info字典来检查是否有捕获发生
        info = self.locals["infos"][0]  # 获取info
        if "capture" in info and info["capture"]:  # 更明确的检查
            self.episode_capture_count += 1
            self.total_capture_count += 1

        # 检查episode是否结束
        done = self.locals["dones"][
            0
        ]  # 向量化环境使用dones而不是terminations/truncations
        if done:
            self.episodes += 1

            # 记录episode数据
            self.episode_lengths.append(self.current_episode_steps)
            self.episode_rewards.append(self.episode_total_reward)
            self.episode_capture_counts.append(self.episode_capture_count)

            # 重置episode计数器
            self.current_episode_steps = 0
            self.episode_total_reward = 0
            self.episode_capture_count = 0

        return True

    def plot_metrics(self):
        """绘制多个性能指标图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 每个Episode的总奖励
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title("Total Reward per Episode")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].set_ylabel("Reward")
        axes[0, 0].grid(True)

        # 2. 捕获数量
        axes[0, 1].plot(self.episode_capture_counts)
        axes[0, 1].set_title("Captures per Episode")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].set_ylabel("Number of Captures")
        axes[0, 1].grid(True)

        # 3. 累计捕获率 (每个episode的捕获/episode数)
        cumulative_capture_rate = [
            sum(self.episode_capture_counts[: i + 1]) / (i + 1)
            for i in range(len(self.episode_capture_counts))
        ]
        axes[1, 0].plot(cumulative_capture_rate)
        axes[1, 0].set_title("Cumulative Capture Rate")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].set_ylabel("Captures per Episode")
        axes[1, 0].grid(True)

        # 4. Episode长度
        axes[1, 1].plot(self.episode_lengths)
        axes[1, 1].set_title("Episode Lengths")
        axes[1, 1].set_xlabel("Episode")
        axes[1, 1].set_ylabel("Steps")
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.savefig("marl_metrics.png", dpi=300)
        plt.close()

        # 绘制单独的奖励曲线，带有移动平均
        plt.figure(figsize=(10, 6))

        # 计算移动平均
        window_size = min(50, len(self.episode_rewards))
        if window_size > 0:
            moving_avg = np.convolve(
                self.episode_rewards, np.ones(window_size) / window_size, mode="valid"
            )

            # 绘制原始数据
            plt.plot(self.episode_rewards, alpha=0.3, color="blue", label="Raw")

            # 绘制移动平均
            plt.plot(
                np.arange(window_size - 1, len(self.episode_rewards)),
                moving_avg,
                color="red",
                label=f"Moving Avg (window={window_size})",
            )

            plt.title("Episode Rewards with Moving Average")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.legend()
            plt.grid(True)
            plt.savefig("reward_curve.png", dpi=300)
            plt.close()

    def get_metrics_summary(self):
        """返回训练过程的关键指标摘要"""
        return {
            "total_episodes": self.episodes,
            "total_steps": self.total_steps,
            "average_episode_length": (
                np.mean(self.episode_lengths) if self.episode_lengths else 0
            ),
            "average_episode_reward": (
                np.mean(self.episode_rewards) if self.episode_rewards else 0
            ),
            "average_captures_per_episode": (
                np.mean(self.episode_capture_counts)
                if self.episode_capture_counts
                else 0
            ),
            "final_10_episodes_avg_reward": (
                np.mean(self.episode_rewards[-10:])
                if len(self.episode_rewards) >= 10
                else np.mean(self.episode_rewards)
            ),
        }


def visualize_policy(
    model=None,
    env=None,
    num_episodes=1,
    max_steps=50,
    save_animation=True,
    use_shortest_path=True,
):
    """
    Visualizes the execution of the trained policy in the provided GPE environment.

    Args:
        model: Trained PPO model
        env: GPE environment instance with render_mode="human"
        num_episodes: Number of episodes to visualize
        max_steps: Maximum steps per episode
        save_animation: Whether to save the visualization as an animation file
        use_shortest_path: Whether using shortest path actions instead of the model
    """
    print("Starting policy visualization...")
    gif_save_dir = "viz_gif"

    if save_animation and not os.path.exists(gif_save_dir):
        os.makedirs(gif_save_dir)
        print(f"Created directory: {gif_save_dir}")

    for episode in range(num_episodes):
        print(f"\n--- Episode {episode+1}/{num_episodes} ---")
        if save_animation:
            frames = []

        observations, _ = env.reset()
        cumulative_reward = 0

        # 打印初始状态信息
        print(f"Episode start: {len(env.agents)} active agents")
        print(f"Pursuers: {[p for p in env.agents if p.startswith('pursuer')]}")
        print(f"Evaders: {[e for e in env.agents if e.startswith('evader')]}")

        for step in range(max_steps):
            # Determine actions for all ACTIVE agents only
            actions = {}

            for agent in env.agents:  # env.agents 在每一步后都会更新
                if use_shortest_path:
                    action = env.shortest_path_action(agent)
                    if action is not None:  # 确保返回了有效动作
                        actions[agent] = action
                else:
                    obs = observations[agent]  # 这里观察空间只会包含活跃智能体
                    obs_array = np.array(obs).reshape(1, -1)
                    action, _ = model.predict(obs_array, deterministic=False)
                    actions[agent] = action.item()

            print(
                f"Step {step+1}: Active agents: {len(env.agents)}, Actions: {actions}"
            )

            # Take a step in the environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # 在 step 执行后打印状态变化（可选）
            term_agents = [a for a, t in terminations.items() if t]
            trunc_agents = [a for a, t in truncations.items() if t]
            # if term_agents:
            #     print(f"  Terminated agents: {term_agents}")
            # if trunc_agents:
            #     print(f"  Truncated agents: {trunc_agents}")
            # if "capture" in infos.get(list(infos.keys())[0], {}):
            #     print(f"  Capture occurred!")

            # Render the current state
            env.render()

            # 如果要保存动画，获取当前图像
            if save_animation:
                fig = plt.gcf()
                fig.canvas.draw()  # Ensure the canvas is drawn

                # 直接渲染到 RGBA buffer (NumPy array)
                image_rgba = np.array(fig.canvas.buffer_rgba())
                # 转换为 RGB (丢弃 Alpha 通道)
                image_rgb = image_rgba[:, :, :3]
                frames.append(image_rgb)

            # Track rewards
            episode_reward = sum(rewards.values())
            cumulative_reward += episode_reward

            # Display step info
            print(
                f"Step {step+1}: Reward = {episode_reward:.2f}, Cumulative = {cumulative_reward:.2f}"
            )

            # 添加暂停，让观察更清晰
            plt.pause(0.5)  # 增加暂停时间到1秒

            # 添加按键控制（可选）
            # input("Press Enter to continue...")  # 每步都需要按回车继续

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

        print(
            f"Episode {episode+1} complete. Cumulative reward: {cumulative_reward:.2f}"
        )
        # 保存动画
        if save_animation and frames:
            try:
                import imageio

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"GPE_{timestamp}_episode{episode+1}.gif"
                filepath = os.path.join(gif_save_dir, filename)
                print(f"Saving animation for episode {episode+1} to {filepath}...")
                imageio.mimsave(filepath, frames, fps=1)  # 使用 frames 创建 GIF
                print(f"Animation saved as {filepath}")
            except ImportError:
                print("Could not save animation: imageio package not found")
                print("Install it with: pip install imageio")

        time.sleep(2)  # Pause between episodes

    print("Visualization complete!")


class CaptureDebugCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.capture_count = 0

    def _on_step(self):
        info = self.locals["infos"][0]
        if "capture" in info and info["capture"]:
            self.capture_count += 1
            print(f"Step {self.n_calls}: Capture! Total: {self.capture_count})")
            # print(f"当前奖励: {self.locals['rewards'][0]}")
        return True


# 新增详细信息回调
class DetailedDebugCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self):
        if self.n_calls % 1000 == 0:  # 每1000步打印一次
            info = self.locals["infos"][0]
            rewards = self.locals["rewards"][0]
            dones = self.locals["dones"][0]
            print(f"\nStep {self.n_calls}:")
            print(f"Rewards: {rewards}")
            print(f"Info: {info}")
            if dones:
                print("Episode ended!")
        return True


# --- Start Edit: Add EscapeDebugCallback Class ---
class EscapeDebugCallback(BaseCallback):
    """
    A custom callback that detects and prints when an escape event occurs.
    NOTE: Assumes the environment's info dictionary (potentially processed by VecEnv)
          contains an 'escape' flag.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.escape_count = 0
        self.last_escape_step = -1  # Avoid double counting if info persists

    def _on_step(self):
        # In VecEnv, infos is a list (usually size 1 if num_vec_envs=1)
        info = self.locals["infos"][0]

        # Check if the 'escape' key exists and is True
        # The exact structure might depend on VecEnv wrapper, but check top level first
        if "escape" in info and info["escape"]:
            # Avoid counting the same event multiple times if info persists across steps within an episode for done envs
            if self.n_calls != self.last_escape_step:
                self.escape_count += 1
                self.last_escape_step = self.n_calls
                if self.verbose > 0:
                    print(f"Step {self.n_calls}: Escape! Total: {self.escape_count}")
                    # Optionally print reward if helpful
                    # print(f"  当前奖励: {self.locals['rewards'][0]}")

        # Reset last escape step if the episode is done
        done = self.locals["dones"][0]
        if done:
            self.last_escape_step = -1

        return True


# Main execution
if __name__ == "__main__":
    # 环境配置
    env_config = {
        "num_nodes": 50,
        "num_pursuers": 2,
        "num_evaders": 1,
        "max_steps": 50,
        "p_act": 1,
        "capture_reward_pursuer": 20.0,
        "capture_reward_evader": -20.0,
        "escape_reward_evader": 100.0,
        "escape_reward_pursuer": -100.0,
        "stay_penalty": -1.0,
    }

    # --- Start Edit: Generate Grid Graph for Test ---
    # Use grid graph generation matching the environment's new logic
    target_n_nodes = env_config["num_nodes"]
    m = int(np.floor(np.sqrt(target_n_nodes)))
    n = int(np.ceil(target_n_nodes / m))
    actual_num_nodes = m * n

    print(f"Test Script: Generating {m}x{n} grid graph ({actual_num_nodes} nodes).")
    base_graph = nx.grid_2d_graph(m, n)
    base_graph = nx.convert_node_labels_to_integers(
        base_graph, first_label=0, ordering="default"
    )

    # Update num_nodes in config to actual grid size
    env_config["num_nodes"] = actual_num_nodes
    # Add graph to config so the env uses this specific instance
    env_config["graph"] = base_graph
    # Keep a copy for visualization if needed later
    graph_for_viz = base_graph
    # --- End Edit ---

    # 创建基础环境
    print("Creating base GPE environment...")
    # Pass the generated graph via config
    base_env = GPE(**env_config, render_mode=None)
    # graph_for_viz = base_env.graph # Already have graph_for_viz from above
    print("Base environment created.")

    # 包装环境以支持 GNN
    print("Wrapping environment with GNNEnvWrapper...")
    try:
        # Pass the base_env instance
        env = GNNEnvWrapper(base_env)
        print("GNNEnvWrapper created.")
        # Verify it looks like a ParallelEnv
        print(
            f"Wrapped env agents (initial): {env.agents}"
        )  # Should be empty before reset
        print(f"Wrapped env possible_agents: {env.possible_agents}")
    except Exception as e:
        print(f"Error creating GNNEnvWrapper: {e}")
        raise

    # --- Simplified Vectorization - Focus on Supersuit ---
    print("Attempting PettingZoo vectorization with Supersuit...")
    try:
        # Ensure the env passed is the GNNEnvWrapper instance
        vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
        print("pettingzoo_env_to_vec_env_v1 successful.")
        # --- Start Edit: Increase num_vec_envs ---
        N_ENVS = (
            1  # Example: Number of parallel environments (adjust based on CPU cores)
        )
        vec_env = ss.concat_vec_envs_v1(
            vec_env,
            num_vec_envs=N_ENVS,
            num_cpus=N_ENVS,  # Match num_vec_envs or available cores
            base_class="stable_baselines3",
        )
        # --- End Edit ---
        print("concat_vec_envs_v1 successful.")
        print("Vectorization successful using Supersuit.")
    except Exception as e:
        print(f"!!! PettingZoo/Supersuit vectorization failed: {e} !!!")
        print("Ensure GNNEnvWrapper correctly implements the ParallelEnv interface.")
        # --- REMOVED DummyVecEnv Fallback ---
        # It's not compatible with PettingZoo environments.
        raise  # Re-raise the exception to stop execution

    # Reset the vectorized environment
    print("Resetting vectorized environment...")
    try:
        # The observation returned by VecEnv should be a NumPy array (flattened dict)
        obs = vec_env.reset()
        print(f"Reset successful.")
        # Check the type and shape of the observation from the VecEnv
        if isinstance(obs, np.ndarray):
            print(f"Observation shape after reset: {obs.shape}")
        else:
            print(f"Observation type after reset: {type(obs)}")
            # If it's still a dict, concat_vec_envs might not have flattened it correctly
            # Or the base_class='stable_baselines3' might need Dict support in SB3 PPO.
    except Exception as e:
        print(f"Error resetting vectorized environment: {e}")
        raise

    # 创建使用 GNN 的 PPO 模型
    policy_kwargs = {
        "features_extractor_class": GNNFeatureExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
        "net_arch": [dict(pi=[64, 64], vf=[64, 64])],
    }

    # 确保设备配置正确
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = PPO(
        GNNPolicy,
        vec_env,
        verbose=1,
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=1024,
        n_epochs=20,
        gamma=0.99,
        ent_coef=0.01,
        gae_lambda=0.95,
        clip_range=0.2,
        device=device,
    )

    # 设置回调
    reward_callback = MARLRewardCallback(
        num_pursuers=env_config["num_pursuers"], num_evaders=env_config["num_evaders"]
    )
    capture_debug = CaptureDebugCallback(verbose=1)
    escape_debug = EscapeDebugCallback(verbose=1)  # Instantiate the new callback
    detailed_debug = DetailedDebugCallback(verbose=1)
    callbacks = CallbackList(
        [reward_callback, capture_debug, escape_debug, detailed_debug]
    )

    # 训练模型
    print(f"Starting GNN PPO training on device: {model.device}")
    # --- Start Edit: Increase timesteps significantly ---
    total_training_timesteps = 500000  # Example: Increased further
    print(f"Training for {total_training_timesteps} timesteps...")
    # --- End Edit ---
    model.learn(total_timesteps=total_training_timesteps, callback=callbacks)
    print("Training finished.")

    # Save the policy
    model_save_path = "gnn_policy"
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}.zip")

    # Plot the reward history
    reward_callback.plot_metrics()

    # Close the training environment
    vec_env.close()

    # --- 加载 GNN 模型 ---
    model_path_load = f"{model_save_path}.zip"  # Use the correct saved model name
    print(f"Loading pre-trained model from '{model_path_load}'...")

    if not os.path.exists(model_path_load):
        print(f"Error: Model file not found at {model_path_load}")
        exit()

    # IMPORTANT: When loading a model with custom policy/features,
    # specify custom_objects including the policy class and potentially the wrapper.
    # However, for SB3 PPO, usually just loading works if policy is registered,
    # but let's load with the policy specified for safety.
    # We need an env instance with the correct obs space for loading,
    # but SB3 load() often handles this if policy is known.
    # If loading fails, you might need to pass `custom_objects={'policy_class': GNNPolicy}`
    try:
        # Provide the policy class during loading if necessary
        # For VecEnvs with Dict obs space, SB3 load might need help.
        # Let's try without env first, it often works.
        loaded_model = PPO.load(model_path_load, device=model.device, policy=GNNPolicy)
        # If the above fails, try passing a dummy env with the correct spaces:
        # dummy_env = GNNEnvWrapper(GPE(**env_config))
        # dummy_vec_env = ss.pettingzoo_env_to_vec_env_v1(dummy_env)
        # loaded_model = PPO.load(model_path_load, env=dummy_vec_env, device=model.device, custom_objects={'policy_class': GNNPolicy})
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Try ensuring the GNNPolicy class is available in the scope during load.")
        exit()

    # --- 可视化 ---
    # Option 1: Visualize using shortest path (doesn't test GNN model)
    print("\nVisualizing using shortest path (not the trained GNN policy)...")
    # Ensure visualization uses the SAME graph instance
    viz_env_base = GPE(
        **env_config,  # env_config now contains the grid graph
        render_mode="human",
        # graph=graph_for_viz, # Already passed via env_config
    )
    visualize_policy(
        model=None,  # Don't pass model if using shortest path
        env=viz_env_base,
        num_episodes=3,
        max_steps=MAX_STEP,
        save_animation=True,
        use_shortest_path=True,  # Force shortest path
    )
    viz_env_base.close()

    # Option 2: Visualize using the loaded GNN model (Requires wrapped env for predict)
    # print("\nVisualizing trained GNN policy...")
    # # We need to wrap the visualization env for the model's predict step
    # viz_env_base_for_gnn = GPE(
    #     **env_config,
    #     render_mode="human", # Keep human render mode
    #     graph=graph_for_viz,
    # )
    # viz_env_wrapped = GNNEnvWrapper(viz_env_base_for_gnn) # Wrap for observation
    # # Modify visualize_policy to accept the wrapped env for predictions
    # # OR create a temporary wrapped env inside visualize_policy when needed
    # # visualize_policy_gnn(loaded_model, viz_env_wrapped, ...) # Needs modification in visualize_policy
    # print("GNN Policy Visualization requires modifications to visualize_policy function - Skipped.")

    # 打印训练摘要
    metrics = reward_callback.get_metrics_summary()
    print("\nTraining Summary:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # 在训练后添加验证代码
    print("\nCallback 验证:")
    print(f"总步数: {reward_callback.total_steps}")
    print(f"总episode数: {reward_callback.episodes}")
    print(f"捕获次数 (Callback): {capture_debug.capture_count}")
    print(f"逃跑次数 (Callback): {escape_debug.escape_count}")
    print(
        f"平均奖励: {np.mean(reward_callback.episode_rewards) if reward_callback.episode_rewards else 0}"
    )
