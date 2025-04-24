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
import networkx as nx
from networkx.readwrite import json_graph

from GPE.env.graph_pe import GPE

# 如果 GNNEnvWrapper 需要，也导入它
# from utils.wrappers.GNNEnvWrapper import GNNEnvWrapper

# --- 配置 ---
CONFIG_PATH = "config.json"
NUM_STEPS_TO_PROFILE = 100  # 你想分析多少步

# --- 加载配置和图 ---
try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    env_config = config.get("environment", {})
    train_config = config.get("training", {})  # 可能需要 N_ENVS
    print("Configuration loaded.")
except Exception as e:
    print(f"Error loading config: {e}")
    exit()

base_graph = None
use_preset = env_config.get("use_preset_graph", False)
if use_preset:
    preset_graph_data = env_config.get("preset_graph", {}).get("graph_adj")
    if preset_graph_data:
        base_graph = json_graph.adjacency_graph(preset_graph_data)
        env_config["num_nodes"] = base_graph.number_of_nodes()
        print(f"Preset graph loaded: {env_config['num_nodes']} nodes.")
    else:
        print("Error: Preset graph data not found.")
        exit()
else:
    target_n_nodes = env_config["num_nodes"]
    m = int(np.floor(np.sqrt(target_n_nodes)))
    n = int(np.ceil(target_n_nodes / m))
    actual_num_nodes = m * n
    print(f"Generating {m}x{n} grid graph ({actual_num_nodes} nodes).")
    base_graph = nx.grid_2d_graph(m, n)
    base_graph = nx.convert_node_labels_to_integers(
        base_graph, first_label=0, ordering="default"
    )
    env_config["num_nodes"] = actual_num_nodes

if base_graph is None:
    print("Failed to load or generate graph.")
    exit()

# --- 准备环境 ---
gpe_init_config = env_config.copy()
gpe_init_config["graph"] = base_graph
gpe_init_config.pop("use_preset_graph", None)
gpe_init_config.pop("preset_graph", None)

print("Creating GPE instance...")
env = GPE(**gpe_init_config, render_mode=None)
# 如果你的 SB3 训练使用了 Wrapper，也在这里包裹它
# env = GNNEnvWrapper(env)
print("Environment created.")

# --- 模拟运行和分析 ---
profiler = cProfile.Profile()
print(f"Resetting environment and profiling {NUM_STEPS_TO_PROFILE} steps...")

observations, infos = env.reset()
active_agents = list(observations.keys())  # 获取初始智能体
# print(f"Initial agents: {active_agents}")

profiler.enable()
start_time = time.time()

for step_num in range(NUM_STEPS_TO_PROFILE):
    if not env.agents:  # 如果所有智能体都结束了
        print(f"All agents terminated/truncated at step {step_num}. Stopping profile.")
        break

    # 创建随机/虚拟动作 (你需要确保动作空间匹配)
    actions = {}
    current_agents = env.agents  # 获取当前活跃的智能体
    for agent in current_agents:
        # 假设动作是节点索引
        action_space_size = env.action_space(agent).n
        actions[agent] = np.random.randint(0, action_space_size)

    # 执行一步
    try:
        obs, rewards, terminations, truncations, infos = env.step(actions)
    except ValueError as e:
        print(f"\nError during step {step_num + 1}: {e}")
        print(f"Current env.agents: {env.agents}")
        print(f"Actions provided: {actions.keys()}")
        break  # 出错时停止

profiler.disable()
end_time = time.time()

print(f"Profiling finished in {end_time - start_time:.2f} seconds.")

# --- 分析结果 ---
print("\n--- cProfile Results for GPE.step (Sorted by Total Time) ---")
s = io.StringIO()
stats = pstats.Stats(profiler, stream=s).sort_stats("tottime")
stats.print_stats(30)  # 打印前 30 行
print(s.getvalue())

print("\n--- cProfile Results for GPE.step (Sorted by Cumulative Time) ---")
s = io.StringIO()
stats = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
stats.print_stats(30)  # 打印前 30 行
print(s.getvalue())

env.close()
