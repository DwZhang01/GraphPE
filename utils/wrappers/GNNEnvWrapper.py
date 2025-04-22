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
    Removed shortest path and degree features for efficiency.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "gnn_gpe_wrapper_v0",
        "is_parallelizable": True,
    }

    def __init__(self, env: GPE):
        self.env = env
        self.num_nodes = self.env.num_nodes
        self.max_edges = self.num_nodes * self.num_nodes

        # Define the node features being used
        self._node_feature_list = [
            "is_safe_node",
            "pursuer_count",
            "active_evader_count",
            "is_current_agent_pos",
        ]
        # Calculate feature dimension based on the list
        self.feature_dim = len(self._node_feature_list)
        print(
            f"GNNWrapper: Using {self.feature_dim} node features: {self._node_feature_list}"
        )

        # {{ edit_1: Store allow_stay from base env }}
        self.allow_stay = self.env.allow_stay

        self.render_mode = self.env.render_mode

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
                    high=max(0, self.num_nodes - 1),
                    shape=(1,),
                    dtype=np.int64,
                ),
            }
        )
        self.possible_agents = self.env.possible_agents
        self._current_graph_pyg = None
        self.padded_edge_index = None
        self._graph_cache_valid = False  # Flag to track cache state
        # self.all_pairs_shortest_path_lengths = None # Removed

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
        if not self._graph_cache_valid:  # Update only if cache is invalid
            if self.env.graph is None:
                # This should ideally not happen after reset, but handle defensively
                print(
                    "Error: Base env graph is None when trying to update. Ensure env is reset."
                )
                # Attempt reset if possible, or raise error. For now, raise.
                # self.reset() # Avoid calling reset from here, could cause loops
                raise RuntimeError(
                    "Graph is None in _ensure_graph_updated. Cannot proceed."
                )
            # print("Updating graph structure for PyG...") # Debug
            self._update_graph_pyg()
            self._graph_cache_valid = True  # Mark cache as valid after update
            # print("Graph cache updated and marked as valid.") # Debug

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
            current_edge_index = self._current_graph_pyg.edge_index[
                :, :max_edges
            ].clone()
        else:
            current_edge_index = self._current_graph_pyg.edge_index.clone()

        pad_width = max_edges - current_edge_index.shape[1]
        if pad_width > 0:
            # Use a padding value unlikely to be a real node index, e.g., -1 or num_nodes
            # Using num_nodes might be okay if GNN ignores out-of-bound indices.
            # Let's keep num_nodes for now based on previous logic, but -1 might be safer.
            padding_value = self.num_nodes  # Or potentially -1
            padding = torch.full((2, pad_width), padding_value, dtype=torch.long)
            edge_index_long = current_edge_index.long()
            self.padded_edge_index = torch.cat([edge_index_long, padding], dim=1)
        else:
            self.padded_edge_index = current_edge_index.long()  # No padding needed

    def _create_node_features(self, agent):
        """Creates the node feature matrix X based on self._node_feature_list."""
        if self.env.graph is None:
            raise RuntimeError("Cannot create node features: graph is None.")

        node_features = np.zeros((self.num_nodes, self.feature_dim), dtype=np.float32)
        safe_node = self.env.safe_node
        agent_positions = self.env.agent_positions
        current_agent_pos = agent_positions.get(agent, -1)
        captured_evaders = self.env.captured_evaders

        # Get indices from the defined list
        try:
            safe_node_idx = self._node_feature_list.index("is_safe_node")
            pursuer_count_idx = self._node_feature_list.index("pursuer_count")
            evader_count_idx = self._node_feature_list.index("active_evader_count")
            current_agent_idx = self._node_feature_list.index("is_current_agent_pos")
        except ValueError as e:
            raise RuntimeError(f"Feature name mismatch in _create_node_features: {e}")

        # 0. Is Safe Node
        if safe_node is not None and 0 <= safe_node < self.num_nodes:
            node_features[safe_node, safe_node_idx] = 1.0

        # 1. & 2. Agent positions (Count)
        for other_agent, pos in agent_positions.items():
            if 0 <= pos < self.num_nodes:
                if other_agent.startswith("pursuer"):
                    # node_features[pos, pursuer_count_idx] += 1.0
                    node_features[pos, pursuer_count_idx] = 1.0
                elif (
                    other_agent.startswith("evader")
                    and other_agent not in captured_evaders
                ):
                    # node_features[pos, evader_count_idx] += 1.0
                    node_features[pos, evader_count_idx] = 1.0

        # 3. Is Current Agent Position
        if 0 <= current_agent_pos < self.num_nodes:
            node_features[current_agent_pos, current_agent_idx] = 1.0

        return node_features

    def _get_action_mask(self, agent):
        """Gets the valid action mask for the agent, respecting self.allow_stay."""
        action_mask = np.zeros(self.num_nodes, dtype=np.float32)
        if self.env.graph is None:
            print("Warning: Action mask calculation skipped, graph is None.")
            return action_mask

        current_pos = self.env.agent_positions.get(agent, -1)

        if agent in self.env.agent_positions and 0 <= current_pos < self.num_nodes:
            try:
                if current_pos in self.env.graph:
                    neighbors = list(self.env.graph.neighbors(current_pos))
                    if self.allow_stay:
                        valid_action_indices = [current_pos] + neighbors
                    else:
                        valid_action_indices = neighbors
                        if not valid_action_indices:
                            print(
                                f"Warning: Agent {agent} at {current_pos} has no neighbors and allow_stay=False. Forcing stay."
                            )
                            valid_action_indices = [current_pos]

                    valid_action_indices = [
                        idx for idx in valid_action_indices if 0 <= idx < self.num_nodes
                    ]

                    if valid_action_indices:
                        action_mask[valid_action_indices] = 1.0
                    elif self.allow_stay:
                        action_mask[current_pos] = 1.0
                else:
                    print(
                        f"Warning: Agent {agent} position {current_pos} not in graph nodes."
                    )
                    if self.allow_stay and 0 <= current_pos < self.num_nodes:
                        action_mask[current_pos] = 1.0
            except (nx.NetworkXError, KeyError) as e:
                print(
                    f"Warning: Error getting neighbors for {agent} at {current_pos}: {e}"
                )
                if self.allow_stay and 0 <= current_pos < self.num_nodes:
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
        # {{ edit_7: Invalidate cache before reset }}
        self._graph_cache_valid = False
        # Reset the underlying parallel environment
        observations_orig, infos = self.env.reset(seed=seed, options=options)
        # Update graph structure after reset (will now use _ensure_graph_updated)
        # print("GNNWrapper: Ensuring graph is updated after reset...") # Debug print
        # self._ensure_graph_updated() # Called within _wrap_observation

        # Wrap observations for all agents returned by the base reset
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
        # {{ edit_8: Ensure graph is updated (uses cache) before wrapping obs }}
        # If the graph can change dynamically mid-episode (unlikely here),
        # we would need to invalidate the cache here based on some info flag.
        # Assuming static graph per episode:
        # self._ensure_graph_updated() # Called within _wrap_observation

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
