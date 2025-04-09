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
        self.max_edges = self.num_nodes * self.num_nodes  # Keep simple upper bound
        # Define feature_dim ONCE
        self.feature_dim = 8  # [is_safe, is_pursuer, is_evader, is_current, degree, dist_to_safe, dist_to_nearest_pursuer, dist_to_nearest_evader]

        # Define the new observation space using self.feature_dim
        self.observation_space = gym.spaces.Dict(
            {
                "node_features": Box(
                    low=0,
                    high=self.num_nodes,  # Max count for pursuer/evader features could exceed 1
                    shape=(self.num_nodes, self.feature_dim),  # Use self.feature_dim
                    dtype=np.float32,
                ),
                "edge_index": Box(
                    low=0,
                    # Max value should be num_nodes for padding, or num_nodes-1 if no padding
                    high=self.num_nodes,
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
        self.observation_spaces = {
            agent: self.observation_space for agent in self.possible_agents
        }
        self._current_graph_pyg = None
        self._update_graph_pyg()

    def _update_graph_pyg(self):
        """Converts the networkx graph to PyG Data object and extracts edge_index."""
        g_nx = self.unwrapped.graph
        if (
            not all(isinstance(n, int) for n in g_nx.nodes())
            or min(g_nx.nodes()) != 0
            or max(g_nx.nodes()) != len(g_nx) - 1
        ):
            print(
                "Warning: Graph nodes are not contiguous integers from 0. Relabeling for PyG."
            )
            g_nx = nx.convert_node_labels_to_integers(g_nx, first_label=0)
            # If relabeling happens, need to ensure agent_positions etc. are updated/mapped.
            # Best practice: Ensure base GPE env uses contiguous integer node labels 0..N-1.
            self.unwrapped.graph = g_nx  # Update base env graph if relabeled (careful!)

        try:
            # Ensure num_nodes attribute matches the actual graph after potential relabeling
            self._current_graph_pyg = from_networkx(g_nx)
            # Update num_nodes based on the actual graph used for pyg conversion
            current_num_nodes = g_nx.number_of_nodes()
            if current_num_nodes != self.num_nodes:
                print(
                    f"Warning: num_nodes mismatch after potential relabeling. Env: {self.num_nodes}, Graph: {current_num_nodes}. Using graph's node count."
                )
                # This might require resizing observation space features if num_nodes changes dynamically.
                # For simplicity, we assume num_nodes passed to __init__ is reliable.
                # If graphs truly change size dynamically, the observation space definition needs adjustment.

            # Check if self._current_graph_pyg has nodes attribute (it should)
            if hasattr(self._current_graph_pyg, "num_nodes"):
                self._current_graph_pyg.num_nodes = (
                    current_num_nodes  # Set PyG num_nodes explicitly
                )

        except Exception as e:
            print(f"Error converting graph: {e}")
            current_num_nodes = self.unwrapped.num_nodes
            self._current_graph_pyg = Data(
                edge_index=torch.empty((2, 0), dtype=torch.long),
                num_nodes=current_num_nodes,
            )

        # Pad edge_index
        num_edges = self._current_graph_pyg.edge_index.shape[1]
        if num_edges > self.max_edges:
            print(
                f"Warning: Edges ({num_edges}) > max_edges ({self.max_edges}). Truncating."
            )
            self.padded_edge_index = self._current_graph_pyg.edge_index[
                :, : self.max_edges
            ]
        else:
            pad_width = self.max_edges - num_edges
            padding_value = self.num_nodes  # Pad with invalid index
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

        # 2. 智能体位置标记 (Use += if multiple agents can be at the same node)
        for other_agent, pos in agent_positions.items():
            if 0 <= pos < self.num_nodes:
                if other_agent.startswith("pursuer"):
                    node_features[pos, 1] += 1.0  # Count pursuers
                elif (
                    other_agent.startswith("evader")
                    and other_agent not in self.unwrapped.captured_evaders
                ):
                    node_features[pos, 2] += 1.0  # Count active evaders

        # 3. 当前智能体位置
        if 0 <= current_agent_pos < self.num_nodes:
            node_features[current_agent_pos, 3] = 1.0

        # 4. 节点度数（归一化）
        degrees = np.array(
            [self.unwrapped.graph.degree(n) for n in range(self.num_nodes)]
        )
        max_degree = (
            np.max(degrees) if degrees.size > 0 else 1.0
        )  # Use np.max for safety
        node_features[:, 4] = degrees / max_degree if max_degree > 0 else 0.0

        # 5. 到安全节点的距离（归一化）
        if safe_node is not None:
            max_dist_safe = (
                0  # Find max distance for better normalization? Maybe 1/(dist+1) is ok.
            )
            for node in range(self.num_nodes):
                try:
                    # Use all_shortest_paths maybe? No, shortest_path is fine.
                    path = nx.shortest_path(self.unwrapped.graph, node, safe_node)
                    distance = len(path) - 1
                    node_features[node, 5] = 1.0 / (distance + 1)
                    max_dist_safe = max(max_dist_safe, distance)
                except nx.NetworkXNoPath:
                    node_features[node, 5] = 0.0
            # Optional: Normalize by max distance: node_features[:, 5] /= (max_dist_safe + 1)

        # 6. 到最近追捕者和逃跑者的距离（归一化）
        max_dist_pursuer = 0
        max_dist_evader = 0
        all_pursuer_pos = [
            p for a, p in agent_positions.items() if a.startswith("pursuer")
        ]
        all_evader_pos = [
            p
            for a, p in agent_positions.items()
            if a.startswith("evader") and a not in self.unwrapped.captured_evaders
        ]

        for node in range(self.num_nodes):
            min_dist_pursuer = float("inf")
            min_dist_evader = float("inf")

            # Calculate distance to nearest pursuer
            for pursuer_pos in all_pursuer_pos:
                if 0 <= pursuer_pos < self.num_nodes:
                    try:
                        dist = (
                            len(
                                nx.shortest_path(
                                    self.unwrapped.graph, node, pursuer_pos
                                )
                            )
                            - 1
                        )
                        min_dist_pursuer = min(min_dist_pursuer, dist)
                    except nx.NetworkXNoPath:
                        continue

            # Calculate distance to nearest active evader
            for evader_pos in all_evader_pos:
                if 0 <= evader_pos < self.num_nodes:
                    try:
                        dist = (
                            len(
                                nx.shortest_path(self.unwrapped.graph, node, evader_pos)
                            )
                            - 1
                        )
                        min_dist_evader = min(min_dist_evader, dist)
                    except nx.NetworkXNoPath:
                        continue

            node_features[node, 6] = (
                1.0 / (min_dist_pursuer + 1)
                if min_dist_pursuer != float("inf")
                else 0.0
            )
            node_features[node, 7] = (
                1.0 / (min_dist_evader + 1) if min_dist_evader != float("inf") else 0.0
            )
            max_dist_pursuer = max(
                max_dist_pursuer,
                min_dist_pursuer if min_dist_pursuer != float("inf") else 0,
            )
            max_dist_evader = max(
                max_dist_evader,
                min_dist_evader if min_dist_evader != float("inf") else 0,
            )

        # Optional: Normalize by max distance
        # if max_dist_pursuer > 0: node_features[:, 6] /= (max_dist_pursuer + 1)
        # if max_dist_evader > 0: node_features[:, 7] /= (max_dist_evader + 1)

        return node_features

    def _get_action_mask(self, agent):
        """Gets the valid action mask for the agent directly."""
        current_pos = self.unwrapped.agent_positions.get(agent, -1)
        action_mask = np.zeros(self.num_nodes, dtype=np.float32)
        # Check if agent exists and is on the graph
        if (
            agent in self.unwrapped.agent_positions
            and 0 <= current_pos < self.num_nodes
        ):
            # Get neighbors from the graph
            try:
                neighbors = list(self.unwrapped.graph.neighbors(current_pos))
                valid_action_indices = [current_pos] + neighbors
                # Ensure indices are valid
                valid_action_indices = [
                    idx for idx in valid_action_indices if 0 <= idx < self.num_nodes
                ]
                if valid_action_indices:
                    action_mask[valid_action_indices] = 1.0
            except (
                nx.NetworkXError
            ):  # Handle case where current_pos might not be in graph (shouldn't happen ideally)
                print(
                    f"Warning: Agent {agent} position {current_pos} not found in graph for action mask."
                )
                action_mask[current_pos] = 1.0  # Allow staying put if error occurs

        elif agent in self.unwrapped.agent_positions:
            # Agent exists but position is invalid (-1), no valid moves? Or stay?
            # Let's assume no valid moves means mask is all zero? Or allow staying at a virtual pos?
            # For safety, return all zeros if position is invalid.
            pass  # action_mask remains all zeros

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
