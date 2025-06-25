import torch
import torch.nn as nn
from typing import Dict, Tuple
from gymnasium import spaces
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Batch
import numpy as np

class ActorCriticGNN_RNN_Policy(nn.Module):
    """
    一个集成了GNN和GRU的Actor-Critic策略网络。
    - GNN部分用于提取每个时间步的图空间特征。
    - GRU部分用于处理时间序列信息，整合历史观测。
    """
    def __init__(self, observation_space: spaces.Dict, action_space: spaces.Discrete, gnn_features_dim: int = 128, rnn_hidden_dim: int = 128):
        super().__init__()

        # --- GNN特征提取器定义 ---
        self.gnn_features_dim = gnn_features_dim
        node_feature_dim = observation_space["node_features"].shape[1]
        
        self.conv1 = GATv2Conv(node_feature_dim, gnn_features_dim // 4, heads=4)
        self.norm1 = nn.LayerNorm(gnn_features_dim)
        self.conv2 = GATv2Conv(gnn_features_dim, gnn_features_dim, heads=1)
        
        # --- GRU记忆模块定义 ---
        self.rnn_hidden_dim = rnn_hidden_dim
        # GRU的输入维度是GNN提取出的特征维度
        self.gru = nn.GRUCell(self.gnn_features_dim, self.rnn_hidden_dim)
        
        # --- Actor和Critic头部网络 ---
        self.action_net = nn.Linear(self.rnn_hidden_dim, action_space.n)
        self.value_net = nn.Linear(self.rnn_hidden_dim, 1)
        
        self.relu = nn.ReLU()

    def _extract_gnn_features(self, obs: Dict[str, torch.Tensor], edge_index: torch.Tensor) -> torch.Tensor:
        """
        从单步观测中提取GNN特征。
        obs["node_features"] 的形状应为 (num_agents, num_nodes, node_feature_dim)
        """
        num_agents = obs["node_features"].shape[0]
        num_nodes = obs["node_features"].shape[1]
        
        # 将批次中的所有节点特征扁平化以进行GNN计算
        node_features_flat = obs["node_features"].reshape(-1, obs["node_features"].shape[-1])
        
        # 构建批处理的 edge_index
        offsets = torch.arange(0, num_agents * num_nodes, num_nodes, device=node_features_flat.device)
        batched_edge_index = edge_index.repeat(1, num_agents) + offsets.repeat_interleave(edge_index.shape[1])

        # GNN前向传播
        x = self.relu(self.norm1(self.conv1(node_features_flat, batched_edge_index)))
        x = self.conv2(x, batched_edge_index) # (num_agents * num_nodes, gnn_features_dim)
        
        # 提取每个智能体自身的节点特征
        is_self_flat = obs["node_features"][:, :, 0].reshape(-1) # 找到 is_self=1 的节点
        agent_node_indices = torch.where(is_self_flat == 1)[0]
        
        agent_features = x[agent_node_indices] # (num_agents, gnn_features_dim)
        
        return agent_features

    def forward(self, obs: Dict[str, torch.Tensor], edge_index: torch.Tensor, rnn_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        网络的前向传播。
        Args:
            obs: 从环境中得到的观测字典。
            edge_index: 图的邻接信息。
            rnn_state: GRU的上一个隐藏状态。
        Returns:
            actions: 动作的 logits。
            values: 状态的评价值。
            next_rnn_state: GRU的下一个隐藏状态。
        """
        # 1. 提取当前时间步的空间特征
        gnn_features = self._extract_gnn_features(obs, edge_index)
        
        # 2. 更新GRU的隐藏状态
        next_rnn_state = self.gru(gnn_features, rnn_state)
        
        # 3. 从GRU的输出计算动作和价值
        # 使用 action_mask 来屏蔽无效动作
        action_mask = obs["action_mask"]
        logits = self.action_net(next_rnn_state)
        
        # 应用掩码：将无效动作的logits设置为一个非常小的值
        logits[action_mask == 0] = -1e8
        
        values = self.value_net(next_rnn_state)
        
        return logits, values, next_rnn_state