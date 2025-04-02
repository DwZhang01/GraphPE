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
from test_SB3_api import visualize_policy


env_config = {
    "num_nodes": 20,
    "num_edges": 40,
    "num_pursuers": 2,
    "num_evaders": 1,
    "capture_distance": 1,
    "required_captors": 1,
    # "seed": 42,
    "capture_reward_pursuer": 10.0,
    "capture_reward_evader": -10.0,
    "escape_reward_evader": 20.0,
    "escape_reward_pursuer": -5.0,
}


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
