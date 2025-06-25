import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import json
from networkx.readwrite import json_graph
import time

from GPE.env.graph_pe import GPE
from policy.ACRGPolicy import ActorCriticGNN_RNN_Policy
from torch_geometric.utils import from_networkx

def visualize_episode(config_path="config.json", pursuer_model_path="pursuer_policy.pth", evader_model_path="evader_policy.pth"):
    print("开始进行可视化...")
    
    device = torch.device("cpu")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    env_config = config.get("environment", {})

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
    graph_layout = nx.spring_layout(graph, seed=42)

    pursuer_ids = env.pursuers
    evader_ids = env.evaders
    pursuer_policy = ActorCriticGNN_RNN_Policy(env.observation_space(pursuer_ids[0]), env.action_space(pursuer_ids[0])).to(device)
    evader_policy = ActorCriticGNN_RNN_Policy(env.observation_space(evader_ids[0]), env.action_space(evader_ids[0])).to(device)

    # 在加载模型时加入 weights_only=True 是更安全的做法，可以避免未来的警告
    pursuer_policy.load_state_dict(torch.load(pursuer_model_path, map_location=device, weights_only=True))
    evader_policy.load_state_dict(torch.load(evader_model_path, map_location=device, weights_only=True))
    pursuer_policy.eval()
    evader_policy.eval()
    print("模型加载成功。")

    obs, _ = env.reset()
    
    # # --- !! 诊断代码开始 !! ---
    # print("\n" + "="*20 + " 初始状态诊断 " + "="*20)
    # initial_positions = env.agent_positions
    # print(f"初始位置: {initial_positions}")
    
    # for p_id in pursuer_ids:
    #     p_pos = initial_positions[p_id]
    #     p_neighbors = list(graph.neighbors(p_pos))
    #     print(f"追捕者 {p_id} 在节点 {p_pos}。它的邻居是: {p_neighbors}")
    #     for e_id in evader_ids:
    #         e_pos = initial_positions[e_id]
    #         if e_pos in p_neighbors:
    #             print(f"--> !!! 发现问题: 追捕者 {p_id} 的邻居中包含了逃跑者 {e_id} (在节点 {e_pos})!")
    # print("="*55 + "\n")
    # # --- !! 诊断代码结束 !! ---

    pursuer_rnn_state = torch.zeros((len(pursuer_ids), pursuer_policy.rnn_hidden_dim)).to(device)
    evader_rnn_state = torch.zeros((len(evader_ids), evader_policy.rnn_hidden_dim)).to(device)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 8))
    # max_steps_per_episode = env.max_cycles
    max_steps_per_episode = 50

    for step in range(max_steps_per_episode):
        pursuer_obs_tensor = { k: torch.tensor(np.array([obs[agent][k] for agent in pursuer_ids]), dtype=torch.float32).to(device) for k in obs[pursuer_ids[0]].keys() }
        evader_obs_tensor = { k: torch.tensor(np.array([obs[agent][k] for agent in evader_ids]), dtype=torch.float32).to(device) for k in obs[evader_ids[0]].keys() }

        with torch.no_grad():
            pursuer_logits, _, next_pursuer_rnn_state = pursuer_policy(pursuer_obs_tensor, edge_index, pursuer_rnn_state)
            pursuer_actions = torch.argmax(pursuer_logits, dim=1)
            evader_logits, _, next_evader_rnn_state = evader_policy(evader_obs_tensor, edge_index, evader_rnn_state)
            evader_actions = torch.argmax(evader_logits, dim=1)

        actions_to_env = {id: act.item() for id, act in zip(pursuer_ids, pursuer_actions)}
        actions_to_env.update({id: act.item() for id, act in zip(evader_ids, evader_actions)})
        
        # 在执行前打印决策
        # print(f"Step {step+1} 决策: {actions_to_env}")

        next_obs, rewards, dones, truncs, _ = env.step(actions_to_env)

        ax.clear()
        agent_positions = env.agent_positions
        pursuer_nodes = [agent_positions[p] for p in pursuer_ids]
        evader_nodes = [agent_positions[e] for e in evader_ids]
        node_colors = ['red' if node in pursuer_nodes else 'blue' if node in evader_nodes else 'lightgray' for node in graph.nodes()]
        nx.draw(graph, pos=graph_layout, with_labels=True, node_color=node_colors, node_size=500, font_size=10, ax=ax)
        ax.set_title(f"Step: {step + 1} / {max_steps_per_episode}")
        plt.pause(0.5)

        obs = next_obs
        pursuer_rnn_state = next_pursuer_rnn_state
        evader_rnn_state = next_evader_rnn_state

        if any(dones.values()) or any(truncs.values()):
            print(f"回合在第 {step + 1} 步结束。")
            # ... (后续结束逻辑保持不变)
            break
            
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    visualize_episode()