import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from networkx.readwrite import json_graph
import time

# 从您的项目目录中导入所需的类
# 请确保这些路径与您的项目结构一致
from GPE.env.graph_pe import GPE
from policy.ACRGPolicy import ActorCriticGNN_RNN_Policy
from torch_geometric.utils import from_networkx

def visualize_episode(config_path="config.json", pursuer_model_path="pursuer_policy.pth", evader_model_path="evader_policy.pth"):
    """
    加载训练好的模型并可视化一个完整的追逃回合。
    """
    print("开始进行可视化...")
    
    # --- 1. 环境和模型设置 ---
    device = torch.device("cpu") # 可视化通常不需要GPU，使用CPU即可
    
    # 加载环境配置
    with open(config_path, "r") as f:
        config = json.load(f)
    env_config = config.get("environment", {})

    # 创建图（与训练时逻辑相同）
    if env_config.get("use_preset_graph", False):
        graph = json_graph.adjacency_graph(env_config["preset_graph"]["graph_adj"])
    else:
        m = int(np.sqrt(env_config["num_nodes"]))
        n = int(np.ceil(env_config["num_nodes"] / m))
        graph = nx.grid_2d_graph(m, n)
        graph = nx.convert_node_labels_to_integers(graph)

    env_config["graph"] = graph
    env = GPE(**env_config)
    edge_index = from_networkx(env.graph).edge_index.to(device)

    # 计算一个固定的图布局，让节点位置在动画中保持不变
    graph_layout = nx.spring_layout(graph, seed=42)

    # --- 2. 加载训练好的策略网络 ---
    pursuer_ids = env.pursuers
    evader_ids = env.evaders

    # 实例化策略网络
    pursuer_policy = ActorCriticGNN_RNN_Policy(env.observation_space(pursuer_ids[0]), env.action_space(pursuer_ids[0])).to(device)
    evader_policy = ActorCriticGNN_RNN_Policy(env.observation_space(evader_ids[0]), env.action_space(evader_ids[0])).to(device)

    # 加载模型权重
    pursuer_policy.load_state_dict(torch.load(pursuer_model_path, map_location=device))
    evader_policy.load_state_dict(torch.load(evader_model_path, map_location=device))

    # 设置为评估模式，这会禁用诸如Dropout等层，并影响梯度计算行为
    pursuer_policy.eval()
    evader_policy.eval()

    print("模型加载成功。")

    # --- 3. 运行并渲染单次回合 ---
    obs, _ = env.reset()
    
    # 初始化RNN隐藏状态
    pursuer_rnn_state = torch.zeros((len(pursuer_ids), pursuer_policy.rnn_hidden_dim)).to(device)
    evader_rnn_state = torch.zeros((len(evader_ids), evader_policy.rnn_hidden_dim)).to(device)
    
    # 开启Matplotlib的交互模式
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))

    max_steps_per_episode = env.max_cycles # 从环境中获取最大步数
    for step in range(max_steps_per_episode):
        # 将观测数据转换为Tensor
        pursuer_obs_tensor = { k: torch.tensor(np.array([obs[agent][k] for agent in pursuer_ids]), dtype=torch.float32).to(device) for k in obs[pursuer_ids[0]].keys() }
        evader_obs_tensor = { k: torch.tensor(np.array([obs[agent][k] for agent in evader_ids]), dtype=torch.float32).to(device) for k in obs[evader_ids[0]].keys() }

        # 使用模型进行决策（不进行梯度计算）
        with torch.no_grad():
            # 追捕者
            pursuer_logits, _, next_pursuer_rnn_state = pursuer_policy(pursuer_obs_tensor, edge_index, pursuer_rnn_state)
            # 选择概率最高的动作（确定性决策）
            pursuer_actions = torch.argmax(pursuer_logits, dim=1)

            # 逃跑者
            evader_logits, _, next_evader_rnn_state = evader_policy(evader_obs_tensor, edge_index, evader_rnn_state)
            evader_actions = torch.argmax(evader_logits, dim=1)

        # 准备提交给环境的动作字典
        actions_to_env = {id: act.item() for id, act in zip(pursuer_ids, pursuer_actions)}
        actions_to_env.update({id: act.item() for id, act in zip(evader_ids, evader_actions)})
        
        # 与环境交互
        next_obs, rewards, dones, truncs, _ = env.step(actions_to_env)

        # --- 渲染当前状态 ---
        ax.clear() # 清除上一帧的图像
        
        # 获取当前所有智能体的位置
        agent_positions = env.agent_positions
        pursuer_nodes = [agent_positions[p] for p in pursuer_ids]
        evader_nodes = [agent_positions[e] for e in evader_ids]

        # 为不同角色的节点设置颜色
        node_colors = []
        for node in graph.nodes():
            if node in pursuer_nodes:
                node_colors.append('red') # 追捕者为红色
            elif node in evader_nodes:
                node_colors.append('blue') # 逃跑者为蓝色
            else:
                node_colors.append('lightgray') # 普通节点为灰色

        # 绘制图形
        nx.draw(graph, pos=graph_layout, with_labels=True, node_color=node_colors, node_size=500, font_size=10, ax=ax)
        ax.set_title(f"Step: {step + 1} / {max_steps_per_episode}")
        
        # 暂停以形成动画效果
        plt.pause(0.5) # 暂停0.5秒，您可以调整这个值来改变动画速度

        # 更新观测和RNN状态
        obs = next_obs
        pursuer_rnn_state = next_pursuer_rnn_state
        evader_rnn_state = next_evader_rnn_state

        # 检查回合是否结束
        if any(dones.values()) or any(truncs.values()):
            print(f"回合在第 {step + 1} 步结束。")
            if any(dones.values()):
                # 确定是谁被抓住了或者逃跑成功
                for agent_id, is_done in dones.items():
                    if is_done:
                        if agent_id in pursuer_ids:
                             print(f"胜利条件满足 (追捕者获胜)!")
                        else:
                             print(f"逃跑者 {agent_id} 被抓住!")
                        break
            if any(truncs.values()):
                print("达到最大步数，回合终止。")
            break

    print("可视化结束。关闭窗口以退出。")
    # 关闭交互模式，让图像窗口保持打开状态直到手动关闭
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    visualize_episode()