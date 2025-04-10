import gymnasium as gym
import numpy as np
import torch
import networkx as nx
from gymnasium.spaces import Box, Dict as GymDict  # Use GymDict alias
from typing import Dict, Any

# Use ParallelEnv directly for type hints and checks if needed, BaseWrapper handles implementation
from pettingzoo.utils.env import ParallelEnv
from pettingzoo.utils.wrappers import BaseWrapper
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from GPE.env.graph_pe import GPE


class GNNEnvWrapper(BaseWrapper):  # Still inherit from BaseWrapper
    """
    Wraps the GPE environment to provide observations suitable for GNNs.
    Correctly exposes necessary ParallelEnv attributes/methods via BaseWrapper.
    """

    def __init__(self, env: ParallelEnv):  # Add type hint for clarity
        super().__init__(env)
        # BaseWrapper already makes self.env available
        # Ensure the wrapped env IS a ParallelEnv (runtime check)
        if not isinstance(self.env.unwrapped, ParallelEnv):
            raise TypeError(
                f"GNNEnvWrapper expects a PettingZoo ParallelEnv, but got {type(env)}"
            )

        self.num_nodes = self.env.unwrapped.num_nodes
        # Infer max edges or use a safer calculation if graph structure varies
        # max_degree = max(dict(self.env.unwrapped.graph.degree()).values()) if self.env.unwrapped.graph else self.num_nodes
        # self.max_edges = self.env.unwrapped.graph.number_of_edges() if self.env.unwrapped.graph else self.num_nodes * self.num_nodes
        # Let's stick to the simpler upper bound for now
        self.max_edges = self.num_nodes * self.num_nodes
        self.feature_dim = 8  # Keep consistent

        # Define the GNN observation space structure
        # Use gym.spaces here (Gymnasium is the backend for PettingZoo spaces)
        self._observation_space = GymDict(  # Store the structure
            {
                "node_features": Box(
                    # Bounds should reflect possible feature values (counts, normalized values)
                    low=0,
                    high=max(
                        self.num_nodes, 1.0
                    ),  # Adjust high bound if features can exceed 1.0
                    shape=(self.num_nodes, self.feature_dim),
                    dtype=np.float32,
                ),
                "edge_index": Box(
                    low=0,
                    high=self.num_nodes,  # Max value is padding value
                    shape=(2, self.max_edges),
                    dtype=np.int64,  # Use int64 for indices
                ),
                "action_mask": Box(
                    low=0,
                    high=1,
                    shape=(self.num_nodes,),
                    dtype=np.float32,  # Mask is float for SB3
                ),
                "agent_node_index": Box(
                    # Agent index should be within valid node range
                    low=0,
                    high=self.num_nodes - 1,
                    shape=(1,),
                    dtype=np.int64,  # Index is int
                ),
            }
        )

        # BaseWrapper should correctly handle possible_agents, but let's be explicit
        self.possible_agents = self.env.unwrapped.possible_agents

        # Initialize graph-related attributes AFTER ensuring env exists
        self._current_graph_pyg = None
        self.padded_edge_index = None

        # --- Critical Change: Reset *after* initializing wrapper attributes ---
        # Don't reset here, reset should happen externally *before* training starts
        # self.env.reset() # REMOVE THIS - causes issues if called inside init multiple times

        # --- Defer graph update until first reset/step ---
        # self._update_graph_pyg() # REMOVE THIS - graph might not exist yet

    # --- Explicitly define observation_space and action_space methods ---
    # These are required by PettingZoo API and wrappers expect them.

    def observation_space(self, agent: str) -> gym.spaces.Space:
        """Returns the observation space for a single agent."""
        # All agents share the same GNN dict space in this setup
        return self._observation_space

    def action_space(self, agent: str) -> gym.spaces.Space:
        """Returns the action space for a single agent."""
        # Delegate to the wrapped environment's action space
        return self.env.action_space(agent)

    # --- Properties required/checked by ParallelEnv ---
    # BaseWrapper *should* provide these, but being explicit can help debug
    @property
    def agents(self) -> list[str]:
        return self.env.agents

    @property
    def metadata(self) -> dict[str, Any]:
        # Combine or override metadata if necessary
        meta = self.env.metadata.copy()
        meta["is_parallelizable"] = True  # Indicate compatibility
        meta["name"] = (
            f"gnn_wrapper_v0({self.env.unwrapped.metadata.get('name', 'unknown')})"
        )
        return meta

    def _ensure_graph_updated(self):
        """Checks if the graph needs updating and calls _update_graph_pyg."""
        # Simple check: if padded_edge_index is None, or maybe compare graph instances
        # This needs refinement if the graph can change dynamically mid-episode
        if self.padded_edge_index is None or self.env.unwrapped.graph is None:
            if self.env.unwrapped.graph is None:
                print(
                    "Warning: Wrapped env graph is None. Ensure env is reset before first step."
                )
                # Attempt reset if graph is missing - potential side effect
                # self.env.reset()
                if self.env.unwrapped.graph is None:
                    raise RuntimeError(
                        "Graph is still None after trying to access/reset. Cannot proceed."
                    )
            print("Updating graph structure for PyG...")
            self._update_graph_pyg()

    def _update_graph_pyg(self):
        """Converts the networkx graph to PyG Data object and extracts edge_index.
        Assumes self.env.unwrapped.graph is not None."""
        g_nx = self.env.unwrapped.graph
        # Add robustness check
        if g_nx is None:
            raise ValueError(
                "_update_graph_pyg called but self.env.unwrapped.graph is None."
            )

        # Relabeling logic (ensure nodes are 0 to N-1)
        nodes = list(g_nx.nodes())
        is_contiguous_int = all(isinstance(n, int) for n in nodes) and nodes == list(
            range(g_nx.number_of_nodes())
        )
        if not is_contiguous_int:
            print(
                "Warning: Graph nodes are not contiguous integers from 0. Relabeling for PyG."
            )
            g_nx = nx.convert_node_labels_to_integers(g_nx, first_label=0)
            # Important: If relabeling occurs, agent positions in the base env might become inconsistent
            # with the graph structure used here unless the base env is also updated.
            # This wrapper assumes the base env *maintains* 0..N-1 node labels after reset.
            # self.env.unwrapped.graph = g_nx # Avoid modifying unwrapped env directly if possible

        current_num_nodes = g_nx.number_of_nodes()
        if current_num_nodes != self.num_nodes:
            print(
                f"Warning: Graph node count ({current_num_nodes}) differs from env.num_nodes ({self.num_nodes}). Check consistency."
            )
            # Adjust self.num_nodes? This might break observation space shapes if done mid-training.
            # Best to ensure env_config['num_nodes'] matches the actual graph size used.

        try:
            self._current_graph_pyg = from_networkx(g_nx)
            self._current_graph_pyg.num_nodes = (
                current_num_nodes  # Explicitly set node count
            )
        except Exception as e:
            print(f"Error converting graph: {e}")
            self._current_graph_pyg = Data(
                edge_index=torch.empty((2, 0), dtype=torch.long),
                num_nodes=current_num_nodes,
            )

        # Padding logic
        num_edges = self._current_graph_pyg.edge_index.shape[1]
        if num_edges > self.max_edges:
            print(
                f"Warning: Edges ({num_edges}) > max_edges ({self.max_edges}). Truncating."
            )
            self.padded_edge_index = self._current_graph_pyg.edge_index[
                :, : self.max_edges
            ].clone()  # Use clone
        else:
            pad_width = self.max_edges - num_edges
            padding_value = current_num_nodes  # Use an invalid node index >= num_nodes
            padding = torch.full((2, pad_width), padding_value, dtype=torch.long)
            # Ensure edge_index is long tensor
            edge_index_long = self._current_graph_pyg.edge_index.long()
            self.padded_edge_index = torch.cat([edge_index_long, padding], dim=1)

    def _create_node_features(self, agent):
        """Creates the node feature matrix X."""
        # Ensure graph is available
        if self.env.unwrapped.graph is None:
            raise RuntimeError("Cannot create node features: graph is None.")

        node_features = np.zeros((self.num_nodes, self.feature_dim), dtype=np.float32)
        safe_node = self.env.unwrapped.safe_node
        agent_positions = self.env.unwrapped.agent_positions
        current_agent_pos = agent_positions.get(agent, -1)  # Use -1 if agent not found

        # ... (rest of the feature calculation logic - ensure it uses self.env.unwrapped) ...
        # Example:
        # 1. Safe node
        if safe_node is not None and 0 <= safe_node < self.num_nodes:
            node_features[safe_node, 0] = 1.0

        # 2. Agent positions
        for other_agent, pos in agent_positions.items():
            if 0 <= pos < self.num_nodes:
                if other_agent.startswith("pursuer"):
                    node_features[pos, 1] += 1.0  # Use += for counts
                elif (
                    other_agent.startswith("evader")
                    and other_agent not in self.env.unwrapped.captured_evaders
                ):
                    node_features[pos, 2] += 1.0

        # 3. Current agent position
        if 0 <= current_agent_pos < self.num_nodes:
            node_features[current_agent_pos, 3] = 1.0

        # 4. Node degree
        try:
            degrees = np.array(
                [self.env.unwrapped.graph.degree(n) for n in range(self.num_nodes)]
            )
            max_degree = np.max(degrees) if degrees.size > 0 else 1.0
            node_features[:, 4] = degrees / max_degree if max_degree > 0 else 0.0
        except nx.NetworkXError as e:
            print(f"Warning: Error getting degrees: {e}")
            node_features[:, 4] = 0.0  # Default if error

        # # 5. 到安全节点的距离（归一化）
        # if safe_node is not None:
        #     max_dist_safe = (
        #         0  # Find max distance for better normalization? Maybe 1/(dist+1) is ok.
        #     )
        #     for node in range(self.num_nodes):
        #         try:
        #             # Use all_shortest_paths maybe? No, shortest_path is fine.
        #             path = nx.shortest_path(self.unwrapped.graph, node, safe_node)
        #             distance = len(path) - 1
        #             node_features[node, 5] = 1.0 / (distance + 1)
        #             max_dist_safe = max(max_dist_safe, distance)
        #         except nx.NetworkXNoPath:
        #             node_features[node, 5] = 0.0
        #     # Optional: Normalize by max distance: node_features[:, 5] /= (max_dist_safe + 1)

        # # 6. 到最近追捕者和逃跑者的距离（归一化）
        # max_dist_pursuer = 0
        # max_dist_evader = 0
        # all_pursuer_pos = [
        #     p for a, p in agent_positions.items() if a.startswith("pursuer")
        # ]
        # all_evader_pos = [
        #     p
        #     for a, p in agent_positions.items()
        #     if a.startswith("evader") and a not in self.unwrapped.captured_evaders
        # ]

        # for node in range(self.num_nodes):
        #     min_dist_pursuer = float("inf")
        #     min_dist_evader = float("inf")

        #     # Calculate distance to nearest pursuer
        #     for pursuer_pos in all_pursuer_pos:
        #         if 0 <= pursuer_pos < self.num_nodes:
        #             try:
        #                 dist = (
        #                     len(
        #                         nx.shortest_path(
        #                             self.unwrapped.graph, node, pursuer_pos
        #                         )
        #                     )
        #                     - 1
        #                 )
        #                 min_dist_pursuer = min(min_dist_pursuer, dist)
        #             except nx.NetworkXNoPath:
        #                 continue

        #     # Calculate distance to nearest active evader
        #     for evader_pos in all_evader_pos:
        #         if 0 <= evader_pos < self.num_nodes:
        #             try:
        #                 dist = (
        #                     len(
        #                         nx.shortest_path(self.unwrapped.graph, node, evader_pos)
        #                     )
        #                     - 1
        #                 )
        #                 min_dist_evader = min(min_dist_evader, dist)
        #             except nx.NetworkXNoPath:
        #                 continue

        #     node_features[node, 6] = (
        #         1.0 / (min_dist_pursuer + 1)
        #         if min_dist_pursuer != float("inf")
        #         else 0.0
        #     )
        #     node_features[node, 7] = (
        #         1.0 / (min_dist_evader + 1) if min_dist_evader != float("inf") else 0.0
        #     )
        #     max_dist_pursuer = max(
        #         max_dist_pursuer,
        #         min_dist_pursuer if min_dist_pursuer != float("inf") else 0,
        #     )
        #     max_dist_evader = max(
        #         max_dist_evader,
        #         min_dist_evader if min_dist_evader != float("inf") else 0,
        #     )

        # # Optional: Normalize by max distance
        # # if max_dist_pursuer > 0: node_features[:, 6] /= (max_dist_pursuer + 1)
        # # if max_dist_evader > 0: node_features[:, 7] /= (max_dist_evader + 1)

        return node_features

    def _get_action_mask(self, agent):
        """Gets the valid action mask for the agent directly."""
        action_mask = np.zeros(self.num_nodes, dtype=np.float32)
        if self.env.unwrapped.graph is None:  # Graph might not exist before first reset
            return action_mask  # Return all zeros if no graph

        current_pos = self.env.unwrapped.agent_positions.get(agent, -1)

        # Check if agent is valid and its position is on the graph
        if (
            agent in self.env.unwrapped.agent_positions
            and 0 <= current_pos < self.num_nodes
        ):
            try:
                # Check if node exists in graph before getting neighbors
                if current_pos in self.env.unwrapped.graph:
                    neighbors = list(self.env.unwrapped.graph.neighbors(current_pos))
                    valid_action_indices = [current_pos] + neighbors
                    # Filter indices just in case
                    valid_action_indices = [
                        idx for idx in valid_action_indices if 0 <= idx < self.num_nodes
                    ]
                    if valid_action_indices:
                        action_mask[valid_action_indices] = 1.0
                else:
                    # Position is technically valid index but not in graph (shouldn't happen with contiguous relabeling)
                    print(
                        f"Warning: Agent {agent} position {current_pos} not found in graph nodes."
                    )
                    action_mask[current_pos] = (
                        1.0  # Allow stay? Or zero mask? Let's allow stay.
                    )

            except nx.NetworkXError as e:
                print(
                    f"Warning: NetworkXError getting neighbors for agent {agent} at {current_pos}: {e}"
                )
                if 0 <= current_pos < self.num_nodes:
                    action_mask[current_pos] = 1.0  # Allow staying put if error
        # else: agent not found or position invalid (-1), mask remains zeros

        return action_mask

    def _wrap_observation(self, agent: str) -> Dict[str, np.ndarray]:
        """Creates the dictionary observation for the GNN."""
        # Ensure graph structure is available and potentially updated
        self._ensure_graph_updated()

        node_features = self._create_node_features(agent)
        action_mask = self._get_action_mask(agent)
        agent_node_index = self.env.unwrapped.agent_positions.get(agent, -1)

        # Handle invalid agent index before creating the array
        if not (0 <= agent_node_index < self.num_nodes):
            # print(f"Warning: Agent {agent} has invalid position {agent_node_index}. Using index 0.")
            valid_agent_node_index = 0  # Use a default valid index
        else:
            valid_agent_node_index = agent_node_index

        obs_dict = {
            "node_features": node_features,
            # Ensure edge index is numpy array of correct type for the space
            "edge_index": self.padded_edge_index.numpy().astype(np.int64),
            "action_mask": action_mask,
            "agent_node_index": np.array([valid_agent_node_index], dtype=np.int64),
        }

        # Optional: Validate dict against the space
        # if not self.observation_space(agent).contains(obs_dict):
        #      print(f"Warning: Observation for {agent} does not match space.")
        # Add detailed checks for each key, shape, dtype here if needed

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
