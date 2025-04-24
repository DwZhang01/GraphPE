import sys
import os

# 获取当前脚本文件所在的目录 (E:\...GraphPE\test)
script_dir = os.path.dirname(os.path.abspath(__file__))
# 获取项目根目录 (E:\...GraphPE)
project_root = os.path.dirname(script_dir)
# 将项目根目录添加到 sys.path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cProfile
import pstats
import io
import json
import time
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.spaces import Box, Dict as GymDict

# 假设 GNNPolicy.py 和 GPE/env/graph_pe.py 在 Python 路径中
from policy.GNNPolicy import GNNFeatureExtractor
from GPE.env.graph_pe import GPE  # 用于获取观察空间结构

# --- 配置 ---
CONFIG_PATH = "config.json"
NUM_FORWARD_CALLS = 100
BATCH_SIZE = 64  # 模拟的批次大小
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 加载配置 ---
try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    env_config = config.get("environment", {})
    nn_config = config.get("neural_network", {})
    print("Configuration loaded.")
except Exception as e:
    print(f"Error loading config: {e}")
    exit()

# --- 创建模拟的观察空间 (基于 GNNEnvWrapper 的定义) ---
# 注意：这里的 num_nodes 和 feature_dim 需要与你的配置和 Wrapper 匹配
num_nodes = env_config.get("num_nodes", 50)  # 获取节点数，需要确保与实际图匹配
# GNNEnvWrapper 计算 feature_dim 的方式
_node_feature_list = [
    "is_safe_node",
    "pursuer_count",
    "active_evader_count",
    "is_current_agent_pos",
]
feature_dim = len(_node_feature_list)
max_edges = num_nodes * num_nodes

mock_observation_space = GymDict(
    {
        "node_features": Box(
            low=0,
            high=max(num_nodes, 1.0),
            shape=(num_nodes, feature_dim),
            dtype=np.float32,
        ),
        "edge_index": Box(low=0, high=num_nodes, shape=(2, max_edges), dtype=np.int64),
        "action_mask": Box(low=0, high=1, shape=(num_nodes,), dtype=np.float32),
        "agent_node_index": Box(
            low=0, high=max(0, num_nodes - 1), shape=(1,), dtype=np.int64
        ),
    }
)
print(
    f"Mock observation space created with num_nodes={num_nodes}, feature_dim={feature_dim}"
)

# --- 创建 Feature Extractor 实例 ---
print("Creating GNNFeatureExtractor instance...")
feature_extractor = GNNFeatureExtractor(
    observation_space=mock_observation_space, features_dim=nn_config["FEATURES_DIM"]
).to(DEVICE)
feature_extractor.eval()  # 设置为评估模式
print(f"Feature extractor created on device: {DEVICE}")

# --- 创建模拟的输入数据 ---
print(f"Creating dummy observation batch (batch_size={BATCH_SIZE})...")
dummy_obs = {
    # torch.rand 需要浮点类型
    "node_features": torch.rand(
        BATCH_SIZE, num_nodes, feature_dim, dtype=torch.float32
    ).to(DEVICE),
    # edge_index 通常是整数类型，并且值应该在 [0, num_nodes-1] 范围内 (或包含填充值)
    # 这里用随机整数模拟，实际应用中需要更真实的边索引
    "edge_index": torch.randint(
        0, num_nodes + 1, (BATCH_SIZE, 2, max_edges), dtype=torch.int64
    ).to(
        DEVICE
    ),  # +1 模拟可能的填充值
    "action_mask": torch.randint(0, 2, (BATCH_SIZE, num_nodes), dtype=torch.float32).to(
        DEVICE
    ),  # 0 或 1
    "agent_node_index": torch.randint(
        0, num_nodes, (BATCH_SIZE, 1), dtype=torch.int64
    ).to(DEVICE),
}
print("Dummy data created.")

# --- 模拟运行和分析 ---
profiler = cProfile.Profile()
print(f"Profiling {NUM_FORWARD_CALLS} forward calls...")

# 可能需要预热（尤其是在 GPU 上）
print("Warm-up run...")
with torch.no_grad():
    _ = feature_extractor(dummy_obs)
if DEVICE == "cuda":
    torch.cuda.synchronize()
print("Warm-up done.")


profiler.enable()
start_time = time.time()

with torch.no_grad():  # 在分析时不计算梯度
    for _ in range(NUM_FORWARD_CALLS):
        _ = feature_extractor(dummy_obs)
        # 如果在 GPU 上分析，同步很重要，以确保操作完成
        if DEVICE == "cuda":
            torch.cuda.synchronize()

profiler.disable()
end_time = time.time()

print(f"Profiling finished in {end_time - start_time:.2f} seconds.")

# --- 分析结果 ---
print(
    "\n--- cProfile Results for GNNFeatureExtractor.forward (Sorted by Total Time) ---"
)
s = io.StringIO()
# 注意：PyTorch 操作在 profile 中可能显示为 C 扩展或内部函数
stats = pstats.Stats(profiler, stream=s).sort_stats("tottime")
stats.print_stats(40)  # 打印更多行，因为 torch 内部调用可能很多
print(s.getvalue())

print(
    "\n--- cProfile Results for GNNFeatureExtractor.forward (Sorted by Cumulative Time) ---"
)
s = io.StringIO()
stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
stats.print_stats(40)
print(s.getvalue())
