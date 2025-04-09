import gymnasium as gym
import numpy as np
import torch
import networkx as nx
from gymnasium.spaces import Dict, Box
from pettingzoo.utils.wrappers import BaseWrapper  # 使用 PettingZoo 的 Wrapper 基类
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from GPE.env.graph_pe import GPE


class GNNEnvWrapper(BaseWrapper):
    """
    Wraps the GPE environment to provide observations suitable for GNNs.

    Observation Space: Dict({
        'node_features': Box(low=0, high=1, shape=(num_nodes, feature_dim)),
        'edge_index': Box(low=0, high=num_nodes-1, shape=(2, max_edges)), # Padded edge index
        'action_mask': Box(low=0, high=1, shape=(num_nodes,)),
        'agent_node_index': Box(low=0, high=num_nodes-1, shape=(1,), dtype=np.int64) # Index of the current agent's node
    })
    """

    def __init__(self, env):
        super().__init__(env)
        self.num_nodes = self.unwrapped.num_nodes
        # Infer max edges (can be approximate or exact if graph is fixed)
        # For dynamic graphs per reset, calculate max possible or use a large buffer
        self.max_edges = self.num_nodes * self.num_nodes  # Simplistic upper bound
        self.feature_dim = (
            6  # [is_safe, is_pursuer, is_evader, is_current, degree, distance_to_safe]
        )

        # Define the new observation space for GNN input
        # Note: SB3's default PPO might struggle with Dict space directly in VecEnv.
        # Custom policy feature extractor is needed.
        self.observation_space = gym.spaces.Dict(
            {
                "node_features": Box(
                    low=0,
                    high=self.num_nodes,
                    shape=(self.num_nodes, self.feature_dim),
                    dtype=np.float32,
                ),
                "edge_index": Box(
                    low=0,
                    high=self.num_nodes - 1,
                    shape=(2, self.max_edges),
                    dtype=np.int64,
                ),
                "action_mask": Box(
                    low=0, high=1, shape=(self.num_nodes,), dtype=np.float32
                ),
                "agent_node_index": Box(
                    low=0, high=self.num_nodes - 1, shape=(1,), dtype=np.int64
                ),
            }
        )

        # Need to redefine observation_space for each agent
        self.observation_spaces = {
            agent: self.observation_space for agent in self.possible_agents
        }

        # Store graph structure (assuming it's somewhat static or accessible)
        self._current_graph_pyg = None
        self._update_graph_pyg()  # Initial graph conversion

    def _update_graph_pyg(self):
        """Converts the networkx graph to PyG Data object and extracts edge_index."""
        # Ensure graph nodes are integers from 0 to num_nodes-1 for PyG
        g_nx = self.unwrapped.graph
        if not all(isinstance(n, int) for n in g_nx.nodes()):
            # Relabel nodes if they are not standard integers
            g_nx = nx.convert_node_labels_to_integers(g_nx, first_label=0)
            print("Warning: Relabeled graph nodes to integers for PyG compatibility.")
            # Potential issue: Need to map original agent positions/actions back if relabeling happens mid-training.
            # Best if the base env always uses integer nodes 0..N-1.

        try:
            self._current_graph_pyg = from_networkx(g_nx)
        except Exception as e:
            print(f"Error converting graph: {e}")
            # Handle cases with isolated nodes or empty graphs if necessary
            num_nodes = self.unwrapped.num_nodes
            self._current_graph_pyg = Data(
                edge_index=torch.empty((2, 0), dtype=torch.long), num_nodes=num_nodes
            )

        # Pad edge_index
        num_edges = self._current_graph_pyg.edge_index.shape[1]
        if num_edges > self.max_edges:
            print(
                f"Warning: Number of edges ({num_edges}) exceeds max_edges ({self.max_edges}). Truncating."
            )
            self.padded_edge_index = self._current_graph_pyg.edge_index[
                :, : self.max_edges
            ]
        else:
            pad_width = self.max_edges - num_edges
            # Pad with a non-existent edge index or repeat last edge? Padding with -1 or num_nodes might be safer
            # Using 0 might connect node 0 incorrectly. Let's pad with num_nodes (invalid index).
            padding_value = self.num_nodes  # Use an invalid node index for padding
            padding = torch.full((2, pad_width), padding_value, dtype=torch.long)
            self.padded_edge_index = torch.cat(
                [self._current_graph_pyg.edge_index, padding], dim=1
            )

    def _create_node_features(self, agent):
        """Creates the node feature matrix X."""
        node_features = np.zeros((self.num_nodes, self.feature_dim), dtype=np.float32)
        safe_node = self.unwrapped.safe_node
        agent_positions = self.unwrapped.agent_positions
        current_agent_pos = agent_positions.get(agent, -1)

        # 1. 基础特征标记
        if safe_node is not None and 0 <= safe_node < self.num_nodes:
            node_features[safe_node, 0] = 1.0  # 安全节点

        # 2. 智能体位置标记
        for other_agent, pos in agent_positions.items():
            if 0 <= pos < self.num_nodes:
                if other_agent.startswith("pursuer"):
                    node_features[pos, 1] = 1.0  # 追捕者位置
                elif (
                    other_agent.startswith("evader")
                    and other_agent not in self.unwrapped.captured_evaders
                ):
                    node_features[pos, 2] = 1.0  # 逃跑者位置

        # 3. 当前智能体位置
        if 0 <= current_agent_pos < self.num_nodes:
            node_features[current_agent_pos, 3] = 1.0

        # 4. 节点度数（归一化）
        degrees = np.array(
            [self.unwrapped.graph.degree(n) for n in range(self.num_nodes)]
        )
        max_degree = max(degrees) if degrees.size > 0 else 1
        node_features[:, 4] = degrees / max_degree

        # 5. 到安全节点的距离（归一化）
        if safe_node is not None:
            for node in range(self.num_nodes):
                try:
                    distance = (
                        len(nx.shortest_path(self.unwrapped.graph, node, safe_node)) - 1
                    )
                    node_features[node, 5] = 1.0 / (distance + 1)  # 归一化距离
                except nx.NetworkXNoPath:
                    node_features[node, 5] = 0.0  # 无路径时设为0

        return node_features

    def _get_action_mask(self, agent):
        """Gets the valid action mask for the agent."""
        # Reuse the logic from the original environment if possible, or recalculate
        # The original observation already calculates it, let's try to extract it
        # This is inefficient; ideally, the base env provides it directly.
        original_obs = self.unwrapped._get_observation(agent)
        # Find where the action mask starts in the flat original_obs
        # Example: pos(1) + pursuers(N_p) + evaders(N_e) + adj(N) + mask(N) ...
        # This structure might vary, making this brittle. Let's recalculate:
        current_pos = self.unwrapped.agent_positions.get(agent, -1)
        action_mask = np.zeros(self.num_nodes, dtype=np.float32)
        if 0 <= current_pos < self.num_nodes:
            neighbors = list(self.unwrapped.graph.neighbors(current_pos))
            valid_action_indices = [current_pos] + neighbors
            # Ensure indices are within bounds
            valid_action_indices = [
                idx for idx in valid_action_indices if 0 <= idx < self.num_nodes
            ]
            if valid_action_indices:
                action_mask[valid_action_indices] = 1.0
        return action_mask

    def observation(self, agent):
        """Returns the processed observation for the agent."""
        # Ensure the graph structure is up-to-date (needed if graph changes on reset)
        # self._update_graph_pyg() # Call this if graph can change

        node_features = self._create_node_features(agent)
        action_mask = self._get_action_mask(agent)
        agent_node_index = self.unwrapped.agent_positions.get(
            agent, -1
        )  # Current position is the node index

        # Ensure agent_node_index is valid, provide a default if agent is not placed?
        if not (0 <= agent_node_index < self.num_nodes):
            agent_node_index = 0  # Default or handle appropriately

        obs = {
            "node_features": node_features,
            "edge_index": self.padded_edge_index.numpy(),  # Convert to numpy for space compatibility
            "action_mask": action_mask,
            "agent_node_index": np.array([agent_node_index], dtype=np.int64),
        }
        # Validate against space (optional but good practice)
        # for key in self.observation_space.spaces:
        #     if not self.observation_space[key].contains(obs[key]):
        #         print(f"Warning: Observation component '{key}' for agent {agent} is outside defined space.")
        #         print(f"Shape expected: {self.observation_space[key].shape}, got: {obs[key].shape}")
        #         print(f"dtype expected: {self.observation_space[key].dtype}, got: {obs[key].dtype}")
        #         # Clamp or correct if necessary
        #         # obs[key] = np.clip(obs[key], self.observation_space[key].low, self.observation_space[key].high)

        return obs

    def reset(self, seed=None, options=None):
        # Reset the underlying environment
        _, infos = self.env.reset(seed=seed, options=options)
        # Update the graph structure after reset
        self._update_graph_pyg()
        # Get observations for all agents using the new wrapper logic
        observations = {agent: self.observation(agent) for agent in self.agents}
        return observations, infos

    def step(self, actions):
        # Step the underlying environment
        observations_orig, rewards, terminations, truncations, infos = self.env.step(
            actions
        )
        # Get the new observations in GNN format for the *next* state
        observations_new = {
            agent: self.observation(agent) for agent in self.env.agents
        }  # Use env.agents for active agents

        # Return observations in the new format
        return observations_new, rewards, terminations, truncations, infos

    # Need to explicitly override observation_space method for PettingZoo wrappers
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    # Action space remains the same
    def action_space(self, agent):
        return self.env.action_space(agent)
