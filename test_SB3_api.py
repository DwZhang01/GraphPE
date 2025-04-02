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
    model,
    env,
    num_episodes=3,
    max_steps=50,
    save_animation=True,
    use_shortest_path=False,
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

            # 重要：这里确保我们只为当前活跃的智能体生成动作
            for agent in env.agents:  # env.agents 在每一步后都会更新
                if use_shortest_path:
                    # 使用最短路径策略
                    if agent.startswith("pursuer"):
                        agent_id = int(agent.split("_")[1])
                        action = env.shortest_path_action("pursuer", agent_id)
                        if action is not None:  # 确保返回了有效动作
                            actions[agent] = action
                    elif agent.startswith("evader"):
                        agent_id = int(agent.split("_")[1])
                        action = env.shortest_path_action("evader", agent_id)
                        if action is not None:  # 确保返回了有效动作
                            actions[agent] = action
                else:
                    # 使用训练好的模型策略
                    obs = observations[agent]  # 这里观察空间只会包含活跃智能体
                    obs_array = np.array(obs).reshape(1, -1)
                    action, _ = model.predict(obs_array, deterministic=False)
                    actions[agent] = action.item()

            # 在 step 执行前打印动作信息（可选）
            print(
                f"Step {step+1}: Active agents: {len(env.agents)}, Actions: {actions}"
            )

            # Take a step in the environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # 在 step 执行后打印状态变化（可选）
            term_agents = [a for a, t in terminations.items() if t]
            trunc_agents = [a for a, t in truncations.items() if t]
            if term_agents:
                print(f"  Terminated agents: {term_agents}")
            if trunc_agents:
                print(f"  Truncated agents: {trunc_agents}")
            if "capture" in infos.get(list(infos.keys())[0], {}):
                print(f"  Capture occurred!")

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
                # --- 结束新增 ---

                print(f"Saving animation for episode {episode+1} to {filepath}...")
                imageio.mimsave(filepath, frames, fps=1)  # 使用 frames 创建 GIF
                # --- 添加完成 ---
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
            print(f"Step {self.n_calls}: 捕获事件发生! (总计: {self.capture_count})")
            print(f"当前奖励: {self.locals['rewards'][0]}")
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


# Main execution
if __name__ == "__main__":
    # Environment configuration
    env_config = {
        "num_nodes": 50,
        "num_edges": 100,
        "num_pursuers": 2,
        "num_evaders": 1,
        "capture_distance": 1,
        "required_captors": 1,
        # "seed": 42,
        "capture_reward_pursuer": 20.0,
        "capture_reward_evader": -20.0,
        "escape_reward_evader": 100.0,
        "escape_reward_pursuer": -100.0,
        "stay_penalty": -0.1,
        "max_steps": 50,
        "p_act": 1,
    }

    # Create training environment
    training_env = GPE(
        **env_config,
        render_mode=None,  # No rendering during training
    )
    graph_for_viz = training_env.graph

    # Wrap the training environment for Stable Baselines3
    vec_env = ss.pettingzoo_env_to_vec_env_v1(training_env)
    vec_env = ss.concat_vec_envs_v1(
        vec_env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3"
    )
    vec_env.reset()

    # Create PPO model
    model = PPO(
        MlpPolicy,
        vec_env,
        verbose=1,  # 减少日志输出频率
        gamma=0.99,
        n_steps=256,
        ent_coef=0.2,
        learning_rate=0.1,
        vf_coef=0.05,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=5,
        clip_range=0.3,
        batch_size=256,
        device="cpu",
    )

    # 设置回调
    reward_callback = MARLRewardCallback(num_pursuers=2, num_evaders=1)
    capture_debug = CaptureDebugCallback(verbose=1)
    detailed_debug = DetailedDebugCallback(verbose=1)
    callbacks = CallbackList([reward_callback, capture_debug, detailed_debug])

    # 使用组合的回调进行训练
    model.learn(total_timesteps=200000, callback=callbacks)

    # Save the policy
    model.save("policy")

    # Plot the reward history
    reward_callback.plot_metrics()

    # Close the training environment
    vec_env.close()

    # --- 添加加载模型的代码 ---
    print("Loading pre-trained model from 'policy.zip'...")
    # 假设模型保存在 "policy.zip" (如果文件名不同请修改)
    model_path = "policy.zip"
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure the model was saved correctly after training.")
        exit()  # 或者抛出异常
    # 加载模型时不需要环境，但可视化时需要
    model = PPO.load(model_path)
    print("Model loaded successfully.")

    # Create a visualization environment (separate from training env)
    # This needs to be a direct GPE instance with human rendering enabled
    viz_env = GPE(
        **env_config,
        render_mode="human",  # Enable rendering
        graph=graph_for_viz,
    )

    # Visualize the policy execution
    print("\nVisualizing trained policy...")
    visualize_policy(
        model,
        viz_env,
        num_episodes=3,
        max_steps=MAX_STEP,
        save_animation=True,  # 启用动画保存
        use_shortest_path=True,  # 添加这个参数，测试最短路径移动
    )

    # Close the visualization environment
    viz_env.close()

    # 打印训练摘要
    metrics = reward_callback.get_metrics_summary()
    print("\nTraining Summary:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # 在训练后添加验证代码
    print("\nCallback 验证:")
    print(f"总步数: {reward_callback.total_steps}")
    print(f"总episode数: {reward_callback.episodes}")
    print(f"捕获次数: {capture_debug.capture_count}")
    print(
        f"平均奖励: {np.mean(reward_callback.episode_rewards) if reward_callback.episode_rewards else 0}"
    )
