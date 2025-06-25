import torch
import numpy as np
from GPE.env.graph_pe import GPE
from policy.ACRGPolicy import ActorCriticGNN_RNN_Policy
from agent.ppo_agent import PPOAgent
from torch_geometric.utils import from_networkx
import json
import networkx as nx
from networkx.readwrite import json_graph

# GAE 辅助函数保持不变
def compute_gae(next_value, rewards, dones, values, gamma, gae_lambda):
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            nextnonterminal = 1.0 - dones[-1]
            nextvalues = next_value
        else:
            nextnonterminal = 1.0 - dones[t]
            nextvalues = values[t+1]
        delta = rewards[t] + gamma * nextvalues * nextnonterminal - values[t]
        advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
    return advantages

if __name__ == "__main__":
    # --- 1. 超参数设置 (保持不变) ---
    total_timesteps = 1_000_000
    num_steps = 256
    learning_rate = 2.5e-4
    gamma = 0.99
    gae_lambda = 0.95
    num_epochs = 4
    
    # --- 2. 环境初始化 (保持不变) ---
    with open("config.json", "r") as f:
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
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = from_networkx(env.graph).edge_index.to(device)

    # --- 3. 初始化 IPPO 智能体 (保持不变) ---
    pursuer_ids = env.pursuers
    evader_ids = env.evaders
    pursuer_policy = ActorCriticGNN_RNN_Policy(env.observation_space(pursuer_ids[0]), env.action_space(pursuer_ids[0])).to(device)
    evader_policy = ActorCriticGNN_RNN_Policy(env.observation_space(evader_ids[0]), env.action_space(evader_ids[0])).to(device)
    pursuer_agent = PPOAgent(pursuer_policy, lr=learning_rate)
    evader_agent = PPOAgent(evader_policy, lr=learning_rate)

    # --- 4. 经验回放缓冲区 (保持不变) ---
    buffer_pursuer = { "obs": [], "actions": [], "logprobs": [], "rewards": [], "dones": [], "values": [], "rnn_states": [] }
    buffer_evader = { "obs": [], "actions": [], "logprobs": [], "rewards": [], "dones": [], "values": [], "rnn_states": [] }

    # --- 5. 训练循环 ---
    global_step = 0
    obs, _ = env.reset()
    
    pursuer_rnn_state = torch.zeros((len(pursuer_ids), pursuer_policy.rnn_hidden_dim)).to(device)
    evader_rnn_state = torch.zeros((len(evader_ids), evader_policy.rnn_hidden_dim)).to(device)

    num_total_agents = len(pursuer_ids) + len(evader_ids)
    num_updates = total_timesteps // (num_steps * num_total_agents)

    for update in range(1, num_updates + 1):
        for k in buffer_pursuer: buffer_pursuer[k] = []
        for k in buffer_evader: buffer_evader[k] = []
        
        # --- Rollout 阶段：收集经验 ---
        for step in range(num_steps):
            global_step += 1
            
            pursuer_obs = { k: torch.tensor(np.array([obs[agent][k] for agent in pursuer_ids]), dtype=torch.float32).to(device) for k in obs[pursuer_ids[0]].keys() }
            evader_obs = { k: torch.tensor(np.array([obs[agent][k] for agent in evader_ids]), dtype=torch.float32).to(device) for k in obs[evader_ids[0]].keys() }
            
            # --- 从策略网络获取动作和价值 (修正) ---
            with torch.no_grad():
                # 追捕者: 获取动作、价值 和 下一步的RNN状态
                pursuer_actions, pursuer_logprobs, _, pursuer_values, next_pursuer_rnn_state = pursuer_agent.get_action_and_value(pursuer_obs, edge_index, pursuer_rnn_state)
                # 逃跑者: 获取动作、价值 和 下一步的RNN状态
                evader_actions, evader_logprobs, _, evader_values, next_evader_rnn_state = evader_agent.get_action_and_value(evader_obs, edge_index, evader_rnn_state)

            # --- 存储当前步的数据 (修正) ---
            # 存储用于计算当前动作的RNN状态
            buffer_pursuer["rnn_states"].append(pursuer_rnn_state)
            buffer_pursuer["obs"].append(pursuer_obs)
            buffer_pursuer["actions"].append(pursuer_actions)
            buffer_pursuer["logprobs"].append(pursuer_logprobs)
            buffer_pursuer["values"].append(pursuer_values.flatten())
            
            buffer_evader["rnn_states"].append(evader_rnn_state)
            buffer_evader["obs"].append(evader_obs)
            buffer_evader["actions"].append(evader_actions)
            buffer_evader["logprobs"].append(evader_logprobs)
            buffer_evader["values"].append(evader_values.flatten())

            # --- 与环境交互 ---
            actions_to_env = {id: act.item() for id, act in zip(pursuer_ids, pursuer_actions)}
            actions_to_env.update({id: act.item() for id, act in zip(evader_ids, evader_actions)})
            next_obs, rewards, dones, truncs, _ = env.step(actions_to_env)
            
            # 存储奖励和完成状态
            buffer_pursuer["rewards"].append(torch.tensor([rewards[agent] for agent in pursuer_ids], dtype=torch.float32).to(device))
            buffer_pursuer["dones"].append(torch.tensor([dones[agent] or truncs[agent] for agent in pursuer_ids], dtype=torch.float32).to(device))
            buffer_evader["rewards"].append(torch.tensor([rewards[agent] for agent in evader_ids], dtype=torch.float32).to(device))
            buffer_evader["dones"].append(torch.tensor([dones[agent] or truncs[agent] for agent in evader_ids], dtype=torch.float32).to(device))

            # --- 更新观测和RNN状态 (修正) ---
            obs = next_obs
            pursuer_rnn_state = next_pursuer_rnn_state
            evader_rnn_state = next_evader_rnn_state

            # 如果回合结束，重置RNN状态
            if any(dones.values()) or any(truncs.values()):
                obs, _ = env.reset()
                pursuer_rnn_state = torch.zeros((len(pursuer_ids), pursuer_policy.rnn_hidden_dim)).to(device)
                evader_rnn_state = torch.zeros((len(evader_ids), evader_policy.rnn_hidden_dim)).to(device)

        # --- GAE 和 Returns 计算 ---
        def process_buffer(agent, buffer, rnn_state):
            with torch.no_grad():
                last_obs = {k: v.to(device) for k,v in buffer["obs"][-1].items()}
                _, _, _, next_value, _ = agent.get_action_and_value(last_obs, edge_index, rnn_state)
            
            # 将列表转换为张量
            for k, v in list(buffer.items()):
                if k == "obs":
                    buffer[k] = {key: torch.stack([step[key] for step in v], dim=0) for key in v[0].keys()}
                else:
                    buffer[k] = torch.stack(v, dim=0)

            advantages = compute_gae(next_value.flatten(), buffer["rewards"], buffer["dones"], buffer["values"], gamma, gae_lambda)
            returns = advantages + buffer["values"]
            
            # 扁平化数据以便训练
            b_obs = { k: v.reshape((-1,) + v.shape[2:]) for k,v in buffer["obs"].items() }
            b_logprobs = buffer["logprobs"].reshape(-1)
            b_actions = buffer["actions"].reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = buffer["values"].reshape(-1)
            b_rnn_states = buffer["rnn_states"].reshape(-1, agent.policy.rnn_hidden_dim)
            
            return b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, b_rnn_states

        b_obs_p, b_log_p, b_act_p, b_adv_p, b_ret_p, b_val_p, b_rnn_p = process_buffer(pursuer_agent, buffer_pursuer, pursuer_rnn_state)
        b_obs_e, b_log_e, b_act_e, b_adv_e, b_ret_e, b_val_e, b_rnn_e = process_buffer(evader_agent, buffer_evader, evader_rnn_state)
        
        # --- 策略更新阶段 ---
        for epoch in range(num_epochs):
            pursuer_agent.train(b_obs_p, b_log_p, b_act_p, b_adv_p, b_ret_p, b_val_p, edge_index, b_rnn_p)
            evader_agent.train(b_obs_e, b_log_e, b_act_e, b_adv_e, b_ret_e, b_val_e, edge_index, b_rnn_e)
        
        print(f"Update {update}/{num_updates}, Global Step: {global_step * num_total_agents}")

    # --- 保存模型 ---
    torch.save(pursuer_policy.state_dict(), "pursuer_policy.pth")
    torch.save(evader_policy.state_dict(), "evader_policy.pth")
    print("Training finished and models saved.")