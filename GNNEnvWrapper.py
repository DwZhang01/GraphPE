import gymnasium as gym
import numpy as np
import torch
import networkx as nx
from gymnasium.spaces import Box, Dict as GymDict
from typing import Dict, Any, List, Optional, Tuple

# Import ParallelEnv directly for inheritance
from pettingzoo.utils.env import ParallelEnv

# Removed BaseWrapper import
# from pettingzoo.utils.wrappers import BaseWrapper
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from GPE.env.graph_pe import GPE


# Inherit directly from ParallelEnv
class GNNEnvWrapper(ParallelEnv):
    """
    Wraps the GPE environment to provide observations suitable for GNNs.
    Inherits directly from ParallelEnv to ensure type compatibility.

    Observation Space (per agent): Dict({
        'node_features': Box(...),
        'edge_index': Box(...),
        'action_mask': Box(...),
        'agent_node_index': Box(...)
    })
    Action Space (per agent): Inherited from GPE (Discrete)
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "gnn_gpe_wrapper_v0",
        "is_parallelizable": True,
    }

    def __init__(self, env: GPE):  # Specific type hint for clarity
        # Store the original environment
        self.env = env

        # Copy essential attributes from the original environment
        self.num_nodes = self.env.num_nodes
        self.max_edges = self.num_nodes * self.num_nodes  # Keep simple upper bound
        self.feature_dim = 8

        self.render_mode = self.env.render_mode

        # Define the GNN observation space structure (used by all agents)
        self._observation_space = GymDict(
            {
                "node_features": Box(
                    low=0,
                    high=max(self.num_nodes, 1.0),
                    shape=(self.num_nodes, self.feature_dim),
                    dtype=np.float32,
                ),
                "edge_index": Box(
                    low=0,
                    high=self.num_nodes,
                    shape=(2, self.max_edges),
                    dtype=np.int64,
                ),
                "action_mask": Box(
                    low=0,
                    high=1,
                    shape=(self.num_nodes,),
                    dtype=np.float32,
                ),
                "agent_node_index": Box(
                    low=0,
                    high=self.num_nodes - 1,
                    shape=(1,),
                    dtype=np.int64,
                ),
            }
        )

        # Required ParallelEnv properties/attributes
        # Delegate possible_agents
        self.possible_agents = self.env.possible_agents

        # State attributes managed by the wrapper or delegated
        self._current_graph_pyg = None
        self.padded_edge_index = None
        # Agents list will be delegated via @property

        # --- No reset call inside __init__ ---

    # --- Implement Required ParallelEnv Methods ---

    def render(self) -> Any:
        """Delegates rendering to the base environment."""
        return self.env.render()

    def close(self) -> None:
        """Closes the base environment."""
        self.env.close()

    # --- Implement Required Observation/Action Space Methods ---

    # @functools.lru_cache(maxsize=None) # Use lru_cache if spaces are static
    def observation_space(self, agent: str) -> gym.spaces.Space:
        """Returns the observation space for a single agent."""
        # Returns the pre-defined Dict space for GNNs
        return self._observation_space

    # @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.spaces.Space:
        """Returns the action space for a single agent by delegating."""
        return self.env.action_space(agent)

    # --- Delegate agents property ---
    @property
    def agents(self) -> List[str]:
        """Returns the list of currently active agents from the base environment."""
        return self.env.agents

    # --- Helper methods for GNN observation ---
    # (Keep _ensure_graph_updated, _update_graph_pyg, _create_node_features, _get_action_mask, _wrap_observation)
    # Ensure they use self.env consistently to access the base GPE state.

    def _ensure_graph_updated(self):
        """Checks if the graph needs updating and calls _update_graph_pyg."""
        if self.padded_edge_index is None or self.env.graph is None:
            if self.env.graph is None:
                print(
                    "Warning: Base env graph is None. Ensure env is reset before first step."
                )
                if self.env.graph is None:
                    raise RuntimeError("Graph is still None. Cannot proceed.")
            # print("Updating graph structure for PyG...") # Debug
            self._update_graph_pyg()

    def _update_graph_pyg(self):
        """Converts the networkx graph to PyG Data object and extracts edge_index.
        Assumes self.env.graph is not None."""
        g_nx = self.env.graph  # Use self.env directly
        if g_nx is None:
            raise ValueError("_update_graph_pyg called but self.env.graph is None.")

        # Relabeling logic
        nodes = list(g_nx.nodes())
        is_contiguous_int = all(isinstance(n, int) for n in nodes) and nodes == list(
            range(g_nx.number_of_nodes())
        )
        if not is_contiguous_int:
            print("Warning: Relabeling graph nodes for PyG.")
            g_nx = nx.convert_node_labels_to_integers(g_nx, first_label=0)

        current_num_nodes = g_nx.number_of_nodes()
        if current_num_nodes != self.num_nodes:
            print(
                f"Warning: Graph node count {current_num_nodes} != env.num_nodes {self.num_nodes}."
            )

        try:
            self._current_graph_pyg = from_networkx(g_nx)
            self._current_graph_pyg.num_nodes = current_num_nodes
        except Exception as e:
            print(f"Error converting graph: {e}")
            self._current_graph_pyg = Data(
                edge_index=torch.empty((2, 0), dtype=torch.long),
                num_nodes=current_num_nodes,
            )

        # Padding logic
        num_edges = self._current_graph_pyg.edge_index.shape[1]
        max_edges = self.max_edges  # Use instance attribute
        if num_edges > max_edges:
            print(f"Warning: Edges {num_edges} > max_edges {max_edges}. Truncating.")
            self.padded_edge_index = self._current_graph_pyg.edge_index[
                :, :max_edges
            ].clone()
        else:
            pad_width = max_edges - num_edges
            padding_value = current_num_nodes
            padding = torch.full((2, pad_width), padding_value, dtype=torch.long)
            edge_index_long = self._current_graph_pyg.edge_index.long()
            self.padded_edge_index = torch.cat([edge_index_long, padding], dim=1)

    def _create_node_features(self, agent):
        """Creates the node feature matrix X."""
        if self.env.graph is None:
            raise RuntimeError("Cannot create node features: graph is None.")

        node_features = np.zeros((self.num_nodes, self.feature_dim), dtype=np.float32)
        safe_node = self.env.safe_node
        agent_positions = self.env.agent_positions
        current_agent_pos = agent_positions.get(agent, -1)
        captured_evaders = self.env.captured_evaders  # Access base env state

        # 1. Safe node
        if safe_node is not None and 0 <= safe_node < self.num_nodes:
            node_features[safe_node, 0] = 1.0

        # 2. Agent positions
        for other_agent, pos in agent_positions.items():
            if 0 <= pos < self.num_nodes:
                if other_agent.startswith("pursuer"):
                    node_features[pos, 1] += 1.0
                elif (
                    other_agent.startswith("evader")
                    and other_agent not in captured_evaders
                ):
                    node_features[pos, 2] += 1.0

        # 3. Current agent position
        if 0 <= current_agent_pos < self.num_nodes:
            node_features[current_agent_pos, 3] = 1.0

        # 4. Node degree
        try:
            # Use self.env.graph directly
            degrees = np.array(
                [self.env.graph.degree(n) for n in range(self.num_nodes)]
            )
            max_degree = np.max(degrees) if degrees.size > 0 else 1.0
            node_features[:, 4] = degrees / max_degree if max_degree > 0 else 0.0
        except (nx.NetworkXError, KeyError) as e:  # Catch KeyError if node not in graph
            print(f"Warning: Error getting degrees: {e}")
            node_features[:, 4] = 0.0

        # 5. 到安全节点的距离（归一化）
        if safe_node is not None:
            max_dist_safe = (
                0  # Find max distance for better normalization? Maybe 1/(dist+1) is ok.
            )
            for node in range(self.num_nodes):
                try:
                    # Use self.env.graph instead of self.unwrapped.graph
                    path = nx.shortest_path(self.env.graph, node, safe_node)
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
            # Use self.env.captured_evaders
            if a.startswith("evader") and a not in self.env.captured_evaders
        ]

        for node in range(self.num_nodes):
            min_dist_pursuer = float("inf")
            min_dist_evader = float("inf")

            # Calculate distance to nearest pursuer
            for pursuer_pos in all_pursuer_pos:
                if 0 <= pursuer_pos < self.num_nodes:
                    try:
                        # Use self.env.graph instead of self.unwrapped.graph
                        dist = (
                            len(nx.shortest_path(self.env.graph, node, pursuer_pos)) - 1
                        )
                        min_dist_pursuer = min(min_dist_pursuer, dist)
                    except nx.NetworkXNoPath:
                        continue

            # Calculate distance to nearest active evader
            for evader_pos in all_evader_pos:
                if 0 <= evader_pos < self.num_nodes:
                    try:
                        # Use self.env.graph instead of self.unwrapped.graph
                        dist = (
                            len(nx.shortest_path(self.env.graph, node, evader_pos)) - 1
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

        # # Optional: Normalize by max distance
        # # if max_dist_pursuer > 0: node_features[:, 6] /= (max_dist_pursuer + 1)
        # # if max_dist_evader > 0: node_features[:, 7] /= (max_dist_evader + 1)

        return node_features

    def _get_action_mask(self, agent):
        """Gets the valid action mask for the agent directly."""
        action_mask = np.zeros(self.num_nodes, dtype=np.float32)
        if self.env.graph is None:
            return action_mask

        current_pos = self.env.agent_positions.get(agent, -1)

        if agent in self.env.agent_positions and 0 <= current_pos < self.num_nodes:
            try:
                if current_pos in self.env.graph:
                    neighbors = list(self.env.graph.neighbors(current_pos))
                    valid_action_indices = [current_pos] + neighbors
                    valid_action_indices = [
                        idx for idx in valid_action_indices if 0 <= idx < self.num_nodes
                    ]
                    if valid_action_indices:
                        action_mask[valid_action_indices] = 1.0
                else:
                    print(
                        f"Warning: Agent {agent} position {current_pos} not in graph nodes."
                    )
                    action_mask[current_pos] = 1.0  # Allow stay
            except (nx.NetworkXError, KeyError) as e:
                print(
                    f"Warning: Error getting neighbors for {agent} at {current_pos}: {e}"
                )
                if 0 <= current_pos < self.num_nodes:
                    action_mask[current_pos] = 1.0
        return action_mask

    def _wrap_observation(self, agent: str) -> Dict[str, np.ndarray]:
        """Creates the dictionary observation for the GNN."""
        self._ensure_graph_updated()  # Ensure graph/edge_index available

        # Check if padded_edge_index was created successfully
        if self.padded_edge_index is None:
            raise RuntimeError(
                "Padded edge index is None in _wrap_observation. Graph update failed?"
            )

        node_features = self._create_node_features(agent)
        action_mask = self._get_action_mask(agent)
        agent_node_index = self.env.agent_positions.get(agent, -1)

        valid_agent_node_index = (
            agent_node_index if 0 <= agent_node_index < self.num_nodes else 0
        )

        obs_dict = {
            "node_features": node_features.astype(np.float32),  # Ensure float32
            "edge_index": self.padded_edge_index.numpy().astype(
                np.int64
            ),  # Ensure int64
            "action_mask": action_mask.astype(np.float32),  # Ensure float32
            "agent_node_index": np.array(
                [valid_agent_node_index], dtype=np.int64
            ),  # Ensure int64
        }
        return obs_dict

    # --- Override reset and step to return wrapped observations ---

    def reset(
        self, seed: int | None = None, options: dict | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Resets the environment and returns wrapped observations."""
        # print("GNNWrapper: Resetting environment...") # Debug print
        # Reset the underlying parallel environment
        observations_orig, infos = self.env.reset(seed=seed, options=options)
        # Update graph structure after reset
        # print("GNNWrapper: Updating graph after reset...") # Debug print
        self._update_graph_pyg()
        # Wrap observations for all agents returned by the base reset
        # The agents available after reset are in self.agents (property delegates to self.env.agents)
        observations_wrapped = {
            agent: self._wrap_observation(agent) for agent in self.agents
        }
        # print(f"GNNWrapper: Reset complete. Returning obs for {list(observations_wrapped.keys())}") # Debug print
        return observations_wrapped, infos

    def step(self, actions: dict[str, Any]) -> tuple[
        dict[str, Any],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, Any],
    ]:
        """Steps the environment and returns wrapped observations."""
        # print(f"GNNWrapper: Stepping with actions for {list(actions.keys())}") # Debug print
        # Step the underlying parallel environment
        observations_orig, rewards, terminations, truncations, infos = self.env.step(
            actions
        )
        # Graph might change in step? If so, need update logic. Assuming static graph per episode for now.
        # self._ensure_graph_updated() # Uncomment if graph can change during step

        # Wrap observations for all agents returned by the base step
        # The agents available for the *next* step are now in self.agents
        observations_wrapped = {
            agent: self._wrap_observation(agent) for agent in self.agents
        }
        # print(f"GNNWrapper: Step complete. Returning obs for {list(observations_wrapped.keys())}") # Debug print
        return observations_wrapped, rewards, terminations, truncations, infos

    # --- Pass through render and close ---
    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        self.env.close()

    # It can be helpful to explicitly define __getattr__ if BaseWrapper doesn't cover everything
    # def __getattr__(self, name):
    #    """Returns an attribute with ``name``, unless ``name`` starts with an underscore."""
    #    if name.startswith("_"):
    #        raise AttributeError(f"accessing private attribute '{name}' is prohibited")
    #    return getattr(self.env, name)
