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


def visualize_policy(model, env, num_episodes=3, max_steps=50, save_animation=True):
    """
    Visualizes the execution of the trained policy in the provided GPE environment.

    Args:
        model: Trained PPO model
        env: GPE environment instance with render_mode="human"
        num_episodes: Number of episodes to visualize
        max_steps: Maximum steps per episode
        save_animation: Whether to save the visualization as an animation file
    """
    print("Starting policy visualization...")

    if save_animation:
        frames = []  # 存储每一帧

    # Run several episodes for visualization
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode+1}/{num_episodes} ---")

        observations, _ = env.reset()
        cumulative_reward = 0

        for step in range(max_steps):
            # Determine actions for all agents using the trained policy
            actions = {}
            for agent, obs in observations.items():
                obs_array = np.array(obs).reshape(1, -1)
                action, _ = model.predict(obs_array, deterministic=True)
                actions[agent] = action.item()

            # Take a step in the environment
            observations, rewards, terminations, truncations, infos = env.step(actions)

            # Render the current state
            env.render()

            # 如果要保存动画，获取当前图像
            if save_animation:
                # 将当前图像添加到帧列表
                fig = plt.gcf()
                # 转换为RGB数组
                fig.canvas.draw()
                image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                frames.append(image)

            # Track rewards
            episode_reward = sum(rewards.values())
            cumulative_reward += episode_reward

            # Display step info
            print(
                f"Step {step+1}: Reward = {episode_reward:.2f}, Cumulative = {cumulative_reward:.2f}"
            )

            # 添加暂停，让观察更清晰
            plt.pause(1.0)  # 增加暂停时间到1秒

            # 添加按键控制（可选）
            input("Press Enter to continue...")  # 每步都需要按回车继续

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
        time.sleep(2)  # Pause between episodes

    print("Visualization complete!")

    # 保存动画
    if save_animation and frames:
        try:
            import imageio

            print("Saving animation...")
            imageio.mimsave(
                "pursuit_evasion.gif", frames, fps=1
            )  # 使用较低的fps使动画更容易观察
            print("Animation saved as 'pursuit_evasion.gif'")
        except ImportError:
            print("Could not save animation: imageio package not found")
            print("Install it with: pip install imageio")


class CaptureDebugCallback(EventCallback):
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
class DetailedDebugCallback(EventCallback):
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
        "num_nodes": 20,
        "num_edges": 40,
        "num_pursuers": 2,
        "num_evaders": 1,
        "capture_distance": 1,
        "required_captors": 1,
        "seed": 42,
        "capture_reward_pursuer": 10.0,
        "capture_reward_evader": -10.0,
        "escape_reward_evader": 20.0,
        "escape_reward_pursuer": -5.0,
    }

    # Create training environment
    training_env = GPE(
        **env_config,
        max_steps=50,
        render_mode=None,  # No rendering during training
    )

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

    # Create a visualization environment (separate from training env)
    # This needs to be a direct GPE instance with human rendering enabled
    viz_env = GPE(
        **env_config,
        max_steps=100,
        render_mode="human",  # Enable rendering
    )

    # Visualize the policy execution
    print("\nVisualizing trained policy...")
    visualize_policy(
        model,
        viz_env,
        num_episodes=3,
        max_steps=100,
        save_animation=True,  # 启用动画保存
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
