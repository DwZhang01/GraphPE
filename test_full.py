import random
import os
import time
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import supersuit as ss
from GPE.env.graph_pe import GPE
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import pandas as pd
import seaborn as sns

# 创建日志目录
log_dir = "logs/"
model_dir = "models/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


# 自定义回调函数，用于保存训练过程中的数据
class SaveTrainingStatsCallback(BaseCallback):
    def __init__(self, check_freq, log_dir, verbose=1):
        super(SaveTrainingStatsCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "training_stats.csv")
        self.stats = {"timesteps": [], "rewards": [], "episode_lengths": []}

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # 获取当前环境的统计数据
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # 记录数据
                self.stats["timesteps"].append(self.num_timesteps)
                self.stats["rewards"].append(
                    np.mean(y[-10:])
                )  # 最近10个episode的平均回报

                # 获取平均episode长度
                if (
                    hasattr(self.model, "ep_info_buffer")
                    and len(self.model.ep_info_buffer) > 0
                ):
                    avg_ep_len = np.mean(
                        [ep_info["l"] for ep_info in self.model.ep_info_buffer[-10:]]
                    )
                    self.stats["episode_lengths"].append(avg_ep_len)
                else:
                    self.stats["episode_lengths"].append(0)

                # 保存为CSV
                df = pd.DataFrame(self.stats)
                df.to_csv(self.save_path, index=False)

                if self.verbose > 0:
                    print(
                        f"Timestep: {self.num_timesteps}, Average Reward: {np.mean(y[-10:]):.2f}"
                    )
        return True

    def _on_training_end(self) -> None:
        # 保存最终的统计数据
        df = pd.DataFrame(self.stats)
        df.to_csv(self.save_path, index=False)


# 创建环境
def make_env(render=False, seed=42):
    env = GPE(
        num_nodes=20,
        num_edges=40,
        num_pursuers=2,
        num_evaders=1,
        capture_distance=1,
        required_captors=1,
        max_steps=50,
        seed=seed,
        render_mode="rgb_array" if render else None,
    )

    # 监控环境，记录训练过程中的回报
    env = Monitor(env, log_dir)

    # 转换为向量环境
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(
        env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3"
    )
    return env


# 训练和评估
def train_and_evaluate():
    # 创建用于训练的环境
    env = make_env(render=False)

    # 创建回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=10000, save_path=model_dir, name_prefix="gpe_model"
    )

    stats_callback = SaveTrainingStatsCallback(check_freq=1000, log_dir=log_dir)

    # 创建并训练模型
    model = PPO(
        MlpPolicy,
        env,
        verbose=1,
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
        tensorboard_log=log_dir,
    )

    print("开始训练...")
    model.learn(total_timesteps=200000, callback=[checkpoint_callback, stats_callback])

    # 保存最终模型
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"模型已保存到 {final_model_path}")

    # 关闭训练环境
    env.close()

    # 可视化训练过程
    plot_training_results()

    # 渲染测试
    render_test(final_model_path)


# 可视化训练结果
def plot_training_results():
    # 设置Seaborn风格
    sns.set(style="darkgrid")

    # 加载训练数据
    stats_path = os.path.join(log_dir, "training_stats.csv")
    if os.path.exists(stats_path):
        df = pd.read_csv(stats_path)

        # 创建一个2x1的子图布局
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # 平滑化曲线的辅助函数
        def smooth_curve(y, box_pts=5):
            box = np.ones(box_pts) / box_pts
            y_smooth = np.convolve(y, box, mode="same")
            return y_smooth

        # 绘制回报曲线
        sns.lineplot(
            x="timesteps",
            y=smooth_curve(df["rewards"]),
            data=df,
            ax=axes[0],
            color="blue",
            label="Smoothed",
        )
        sns.lineplot(
            x="timesteps",
            y="rewards",
            data=df,
            ax=axes[0],
            color="lightblue",
            alpha=0.4,
            label="Raw",
        )
        axes[0].set_title("Average Episode Reward During Training", fontsize=16)
        axes[0].set_ylabel("Average Reward", fontsize=14)
        axes[0].legend()

        # 绘制回合长度曲线
        sns.lineplot(
            x="timesteps",
            y=smooth_curve(df["episode_lengths"]),
            data=df,
            ax=axes[1],
            color="green",
            label="Smoothed",
        )
        sns.lineplot(
            x="timesteps",
            y="episode_lengths",
            data=df,
            ax=axes[1],
            color="lightgreen",
            alpha=0.4,
            label="Raw",
        )
        axes[1].set_title("Average Episode Length During Training", fontsize=16)
        axes[1].set_xlabel("Timesteps", fontsize=14)
        axes[1].set_ylabel("Average Episode Length", fontsize=14)
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "training_visualization.png"), dpi=300)
        plt.show()

        print(
            f"训练可视化结果已保存到 {os.path.join(log_dir, 'training_visualization.png')}"
        )
    else:
        print("找不到训练数据文件，无法绘制图表")


# 渲染并可视化一段测试追逃过程
def render_test(model_path, num_episodes=3):
    # 加载模型
    model = PPO.load(model_path)

    # 创建渲染环境
    env = make_env(render=True, seed=random.randint(0, 1000))

    frames = []
    episode_rewards = []

    for episode in range(num_episodes):
        print(f"渲染 Episode {episode+1}/{num_episodes}...")
        obs = env.reset()
        done = False
        episode_reward = 0
        step = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            step += 1

            # 获取渲染帧
            frame = env.render()
            if frame is not None:
                frames.append(frame)

            time.sleep(0.1)  # 便于观察

        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1} 完成: 回报 = {episode_reward}, 步数 = {step}")

    # 如果有帧，则保存为视频或GIF
    if frames:
        try:
            from moviepy.editor import ImageSequenceClip

            clip = ImageSequenceClip(frames, fps=5)
            clip.write_gif(os.path.join(log_dir, "pursuit_evasion.gif"), fps=5)
            print(
                f"追逃过程动画已保存到 {os.path.join(log_dir, 'pursuit_evasion.gif')}"
            )
        except ImportError:
            print("需要安装moviepy库来保存动画: pip install moviepy")
            # 至少保存一些关键帧
            for i, frame in enumerate(frames[::10]):
                plt.figure(figsize=(8, 8))
                plt.imshow(frame)
                plt.axis("off")
                plt.savefig(os.path.join(log_dir, f"frame_{i}.png"))
                plt.close()
            print(f"已保存关键帧到 {log_dir} 目录")

    env.close()
    print(f"测试完成! 平均回报: {sum(episode_rewards)/len(episode_rewards):.2f}")


# 添加图形追逃环境的可视化
def visualize_graph_environment():
    # 创建环境
    env = GPE(
        num_nodes=20,
        num_edges=40,
        num_pursuers=2,
        num_evaders=1,
        capture_distance=1,
        required_captors=1,
        max_steps=50,
        seed=42,
        render_mode=None,
    )

    # 重要：需要调用reset()来初始化图结构
    env.reset()

    # 确保图结构存在
    if env.graph is None:
        print("无法访问环境的图结构，跳过图可视化")
        return

    # 获取图结构
    G = env.graph

    # 绘制图结构
    plt.figure(figsize=(10, 10))
    pos = nx.spring_layout(G, seed=42)  # 使用布局算法固定节点位置

    # 绘制节点和边
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color="lightblue")
    nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.7)
    nx.draw_networkx_labels(G, pos, font_size=12)

    # 获取追捕者和逃避者的位置
    pursuer_nodes = [
        env.agent_positions[agent]
        for agent in env.pursuers
        if agent in env.agent_positions
    ]
    evader_nodes = [
        env.agent_positions[agent]
        for agent in env.evaders
        if agent in env.agent_positions
    ]

    # 标记安全节点
    if env.safe_node is not None:
        nx.draw_networkx_nodes(
            G,
            pos,
            nodelist=[env.safe_node],
            node_size=700,
            node_color="green",
            label="Safe Node",
        )

    # 强调追捕者和逃避者的位置
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=pursuer_nodes,
        node_size=700,
        node_color="blue",
        label="Pursuers",
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=evader_nodes, node_size=700, node_color="red", label="Evaders"
    )

    plt.title("Graph Pursuit-Evasion Environment", fontsize=18)
    plt.legend(fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "environment_graph.png"), dpi=300)
    plt.show()

    print(f"环境图结构已保存到 {os.path.join(log_dir, 'environment_graph.png')}")


if __name__ == "__main__":
    # 可视化环境图结构
    visualize_graph_environment()

    # 训练并评估模型
    train_and_evaluate()
