import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict as GymDict
import random
from copy import copy
import networkx as nx
from pettingzoo import ParallelEnv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Optional, Dict, List, Tuple, Any
from gymnasium.utils import EzPickle
from torch_geometric.data import Data 
from torch_geometric.utils import from_networkx 


class GPE(ParallelEnv):
    count = 0 # Static class variable to track instantiations

    metadata = {
        "name": "graph_pe_v0",
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        num_nodes=100,
        num_pursuers=5,
        num_evaders=7,
        capture_distance=1,
        required_captors=1,
        max_steps=200,
        seed=None,
        graph=None,
        render_mode=None,
        p_act=1,
        capture_reward_pursuer=2.0,
        capture_reward_evader=-2.0,
        escape_reward_evader=5.0,
        escape_reward_pursuer=-5.0,
        stay_penalty=-0.05,
        layout_algorithm="grid",
        allow_stay: bool = False,
        grid_m: Optional[int] = None,
        grid_n: Optional[int] = None,
        time_penalty: float = -0.05,
        revisit_penalty: float = -0.05,
    ):
        super().__init__()
        GPE.count += 1 # Increment the static counter
        print(f"GPE instance created. Total instances: {GPE.count}") # Print the count
        self.np_random = np.random.RandomState(seed)
        self.num_pursuers = num_pursuers
        self.num_evaders = num_evaders
        self.capture_distance = capture_distance
        self.required_captors = required_captors
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.p_act = p_act
        self.layout_algorithm = layout_algorithm
        self.allow_stay = allow_stay


        self.grid_m = grid_m
        self.grid_n = grid_n
        self.custom_graph = graph
        if self.custom_graph is None:
            m = int(np.floor(np.sqrt(num_nodes)))
            n = int(np.ceil(num_nodes / m))
            actual_num_nodes = m * n
            self.num_nodes = actual_num_nodes
            if self.grid_m is None:
                self.grid_m = m
            if self.grid_n is None:
                self.grid_n = n
            print(
                f"GPE Init: Grid graph generated ({m}x{n}). Actual num_nodes set to {self.num_nodes}"
            )
        else:
            self.num_nodes = self.custom_graph.number_of_nodes()
            print(
                f"GPE Init: Custom graph provided. Actual num_nodes set to {self.num_nodes}. Grid layout might require manual m, n if not inferred."
            )


        self.pursuers = [f"pursuer_{i}" for i in range(self.num_pursuers)]
        self.evaders = [f"evader_{i}" for i in range(self.num_evaders)]
        self.possible_agents = self.pursuers + self.evaders # All agents.
        self._agents = self.possible_agents.copy()  # All agents active

        self.graph = None
        self.safe_node = None
        self.agent_positions = {}
        self.timestep = 0
        self.rewards = None
        self.terminations = None
        self.truncations = None
        self.infos = None
        self.captured_evaders = set()

        self.capture_reward_pursuer = capture_reward_pursuer
        self.capture_reward_evader = capture_reward_evader
        self.escape_reward_evader = escape_reward_evader
        self.escape_reward_pursuer = escape_reward_pursuer
        self.stay_penalty = stay_penalty
        self.time_penalty = time_penalty
        self.revisit_penalty = revisit_penalty
        self._visited_nodes = {agent: set() for agent in self.possible_agents} # Initialize visited nodes tracker
        
        self.graph = self._generate_graph() # Initialize graph once
        self.adj = {node: list(self.graph.neighbors(node)) for node in self.graph.nodes()}

        #--------add wrapper---------------------------------

        if self.graph.number_of_nodes() > 0:
            self.graph_edge_index = from_networkx(self.graph).edge_index.numpy()
        else:
            self.graph_edge_index = np.empty((2, 0), dtype=np.int64) # Empty graph

        self._node_feature_list = [
            "is_self",
            "is_pursuer",
            "is_evader",
            "is_safe_node",
        ]
        self.feature_dim = len(self._node_feature_list)
        self.single_action_space = Discrete(self.num_nodes) 
        self.single_observation_space = GymDict({
            "node_features": Box(
                low=0,
                high=1,
                shape=(self.num_nodes, self.feature_dim),
                dtype=np.float32,
            ),
            "action_mask": Box(
                low=0,
                high=1,
                shape=(self.num_nodes,),
                dtype=np.float32,
            )
        })

        self._action_spaces = {agent: self.single_action_space for agent in self.possible_agents}
        self._observation_spaces = {agent: self.single_observation_space for agent in self.possible_agents}

        print("GPE environment initialized...")
    
        
    @property
    def agents(self):
        """Return the currently active agents."""
        return self._agents
    
    @property
    def graph_connectivity(self) -> np.ndarray:
        """
        Returns the graph's edge index in COO format (shape: [2, num_edges]).
        This is static information about the graph structure.
        """
        return self.graph_edge_index

    @property
    def action_spaces(self):
        return self._action_spaces
    
    def action_space(self, agent):
        return self.action_spaces[agent]
    
    @property
    def observation_spaces(self):
        return self._observation_spaces

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    
    def _generate_graph(self):
        """Generate a random graph or use the provided one.
        MARK: It should be changed to a certain range of connection for each node.
        """

        if self.custom_graph is not None:
            return copy(self.custom_graph)

        m = int(np.floor(np.sqrt(self.num_nodes)))
        n = int(np.ceil(self.num_nodes / m))
        actual_num_nodes = m * n
        graph = nx.grid_2d_graph(m, n)
        graph = nx.convert_node_labels_to_integers(
            graph, first_label=0, ordering="default"
        )

        return graph

    def _build_node_occupancy_map(self) -> Dict[int, Dict[str, List[str]]]:
        """
        Builds a map from node index to lists of active pursuers and evaders at that node.
        """
        node_occupancy = {i: {'pursuers': [], 'evaders': []} for i in range(self.num_nodes)}

        for p_id in self.pursuers:
            if not self.terminations.get(p_id, False): # Check if pursuer is active
                pos = self.agent_positions.get(p_id)
                if pos is not None and 0 <= pos < self.num_nodes:
                    node_occupancy[pos]['pursuers'].append(p_id)

        for e_id in self.evaders:
            if e_id not in self.captured_evaders and \
            not self.terminations.get(e_id, False): # Check if evader is active and not captured
                pos = self.agent_positions.get(e_id)
                if pos is not None and 0 <= pos < self.num_nodes:
                    node_occupancy[pos]['evaders'].append(e_id)

        return node_occupancy
    
    def reset(self, seed=None, options=None):
        """Reset the environment. Attempts safe initial placement, falls back to random if needed."""

        print("Resetting GPE environment...")
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        self.timestep = 0
        self._agents = self.possible_agents.copy()
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.captured_evaders = set()
        self._visited_nodes = {agent: set() for agent in self.possible_agents}
        all_nodes = list(self.graph.nodes())
        self.np_random.shuffle(all_nodes) 
        self.agent_positions = {}
        infos = {agent: {} for agent in self.agents}

        # Ensure enough nodes total for all agents and the safe node
        required_nodes = self.num_pursuers + self.num_evaders + 1 # +1 for safe_node
        if len(all_nodes) < required_nodes:
            raise ValueError(
                f"Not enough nodes ({len(all_nodes)}) in the graph for all agents ({self.num_pursuers + self.num_evaders}) and a safe node."
            )

        # Place Pursuers by popping from shuffled nodes
        for agent in self.pursuers:
            self.agent_positions[agent] = int(all_nodes.pop())

        # Place Evaders by popping from shuffled nodes
        for agent in self.evaders:
            self.agent_positions[agent] = int(all_nodes.pop())

        # Place Safe Node by popping from remaining shuffled nodes
        self.safe_node = int(all_nodes.pop())

        # Generate initial observations for all agents
        current_node_occupancy = self._build_node_occupancy_map()
        observations = {agent: self._get_observation(agent, node_occupancy=current_node_occupancy) for agent in self.agents}

        # Return standard reset format
        print("GPE environment reset complete.")
        return observations, infos

    
    def _get_action_mask(self, agent_id: str) -> np.ndarray:
            """
            Generate the action mask for the given agent.
            A value of 1 means the action is valid, 0 otherwise.
            The mask corresponds to choosing a target node.
            """
            action_mask = np.zeros(self.num_nodes, dtype=np.float32)

            # If agent is terminated or not active, no actions are valid
            if self.terminations.get(agent_id, False) or agent_id not in self.agent_positions:
                return action_mask

            current_position = self.agent_positions[agent_id]
            
            valid_action_indices = []
            if 0 <= current_position < self.num_nodes: # Check if current_position is a valid node index
                neighbors = self.adj[current_position]
                if self.allow_stay:
                    valid_action_indices = [current_position] + neighbors
                else:
                    valid_action_indices = neighbors
                
                # Filter out invalid node indices just in case (e.g. graph changes, though not expected here)
                valid_action_indices = [
                    idx for idx in valid_action_indices if 0 <= idx < self.num_nodes
                ]

                if valid_action_indices:
                    action_mask[valid_action_indices] = 1.0
                elif self.allow_stay and (0 <= current_position < self.num_nodes): # If no neighbors but stay is allowed
                    action_mask[current_position] = 1.0
            
            return action_mask
    
    def _get_node_features(self, agent_id: str, node_occupancy: Dict[int, Dict[str, List[str]]]) -> np.ndarray:
        """
        Generate node features for the GNN using a precomputed node_occupancy map.
        Features: "is_self", "is_pursuer", "is_evader", "is_safe_node"
        """
        node_features = np.zeros((self.num_nodes, self.feature_dim), dtype=np.float32)
        current_agent_pos = self.agent_positions.get(agent_id, -1) 

        for node_idx in range(self.num_nodes):
            features = np.zeros(self.feature_dim, dtype=np.float32)

            if node_idx == current_agent_pos:
                features[self._node_feature_list.index("is_self")] = 1.0

            if node_occupancy[node_idx]['pursuers']:
                features[self._node_feature_list.index("is_pursuer")] = 1.0

            if node_occupancy[node_idx]['evaders']: 
                features[self._node_feature_list.index("is_evader")] = 1.0

            if node_idx == self.safe_node:
                features[self._node_feature_list.index("is_safe_node")] = 1.0

            node_features[node_idx] = features

        return node_features
    
    def _get_observation(self, agent_id: str, node_occupancy: Dict[int, Dict[str, List[str]]]) -> Dict[str, np.ndarray]: # 添加参数
        
        if self.terminations.get(agent_id, False) or agent_id not in self._agents:
            return self._get_terminal_observation(agent_id)

        node_feats = self._get_node_features(agent_id, node_occupancy) # 传递占用地图
        action_m = self._get_action_mask(agent_id)

        return {
            "node_features": node_feats,
            "action_mask": action_m,
        }

    def _get_terminal_observation(self, agent_id: str) -> Dict[str, np.ndarray]:
        """
        Return a zeroed-out observation dictionary for a terminated or inactive agent.
        """
        term_node_features = np.zeros(
            (self.num_nodes, self.feature_dim), dtype=np.float32
        )
        term_action_mask = np.zeros(self.num_nodes, dtype=np.float32)
        
        return {
            "node_features": term_node_features,
            "action_mask": term_action_mask,
        }

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict],
    ]:
        """Execute actions for all agents and return new observations."""
        
        active_agents_set = set(self.agents) 
        received_actions_set = set(actions.keys())
        
        if not received_actions_set == active_agents_set:
            missing = active_agents_set - received_actions_set
            extra = received_actions_set - active_agents_set
            error_msg = "Actions mismatch. "
            if missing:
                error_msg += f"Missing actions for: {missing}. "
            if extra:
                error_msg += f"Received unexpected actions for: {extra}. "
            error_msg += f"Expected actions for: {list(active_agents_set)}."
            raise ValueError(error_msg)

        rewards = {agent: self.time_penalty for agent in self.agents}
        terminations = {agent: self.terminations.get(agent, False) for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        next_positions = self.agent_positions.copy()
        
        for agent, action in actions.items():
            if agent not in self.agents:
                continue
                
            # 跳过已终止的agents
            if self.terminations.get(agent, False):
                continue

            current_position = self.agent_positions[agent]
            
            # 应用动作噪声
            if self.np_random.random() > self.p_act:
                effective_action = current_position  # 动作失效，保持原位
            else:
                effective_action = action

            # 验证动作有效性
            is_valid_move = (
                effective_action == current_position or 
                effective_action in self.adj[current_position]
            )
            
            if is_valid_move:
                is_staying = effective_action == current_position
                if (not self.allow_stay and is_staying and 
                    self.adj[current_position]):
                    infos[agent]["invalid_action"] = True
                else:
                    next_positions[agent] = effective_action
            else:
                infos[agent]["invalid_action"] = True

            if (self.allow_stay and effective_action == current_position and
                hasattr(self, 'stay_penalty')):
                rewards[agent] += self.stay_penalty

        # Apply revisit penalty and update visited nodes
        for agent, new_pos in next_positions.items():
            if agent in self._visited_nodes and new_pos in self._visited_nodes[agent] and not self.terminations.get(agent, False) and not self.truncations.get(agent, False):
                rewards[agent] += self.revisit_penalty
                infos[agent]["revisit_penalty"] = True
            self._visited_nodes[agent].add(new_pos)

        self.agent_positions = next_positions
        
        rewards,infos = self._check_captures(rewards,infos)
        rewards,infos = self._check_safe_arrivals(rewards,infos)
        self._check_termination()

        self.timestep += 1
        if hasattr(self, 'max_steps') and self.timestep >= self.max_steps:
            for agent in list(rewards.keys()):
                if not terminations.get(agent):
                    truncations[agent] = True

        for agent_id_iter in active_agents_set:
            terminations[agent_id_iter] = self.terminations.get(agent_id_iter, False)
        
        current_node_occupancy = self._build_node_occupancy_map()

        observations = {agent: self._get_observation(agent,current_node_occupancy) for agent in active_agents_set}
        
        # Remove terminated or truncated agents from _agents
        self._agents = [agent for agent in active_agents_set if not (terminations.get(agent,False) or truncations.get(agent,False))]
        
        return observations, rewards, terminations, truncations, infos

    def _check_captures(self,rewards,infos):
        """Check if any evaders have been captured."""

        for evader in self.evaders:
            if evader in self.captured_evaders or self.terminations.get(evader, False):
                continue
            
            evader_pos = self.agent_positions[evader]
            evader_neighbors = self.adj[evader_pos]

            adjacent_pursuers = 0
            for pursuer in self.pursuers:
                if self.terminations.get(pursuer, False):
                    continue

                pursuer_pos = self.agent_positions[pursuer]
                if pursuer_pos == evader_pos or pursuer_pos in evader_neighbors:
                    adjacent_pursuers += 1

            if adjacent_pursuers >= self.required_captors:
                self.captured_evaders.add(evader)
                self.terminations[evader] = True

                for pursuer in self.pursuers:
                    if not self.terminations.get(pursuer, False):
                        rewards[pursuer] += self.capture_reward_pursuer
                rewards[evader] += self.capture_reward_evader

                for agent in self.agents:
                    if agent not in infos:
                        infos[agent] = {}
                    infos[agent]["capture"] = True
        return rewards, infos

    def _check_safe_arrivals(self,rewards, infos):
        """Check if any evaders have reached the safe node."""
        escape_occurred_this_step = False  # Track if any escape happened
        for evader in self.evaders:
            if evader in self.captured_evaders or self.terminations.get(evader, False):
                continue

            if self.agent_positions[evader] == self.safe_node:
                self.terminations[evader] = True
                rewards[evader] += self.escape_reward_evader
                escape_occurred_this_step = True  # Mark that an escape happened

                for pursuer in self.pursuers:
                    if not self.terminations.get(pursuer, False):
                        rewards[pursuer] += self.escape_reward_pursuer
                        

        if escape_occurred_this_step:
            for agent in self.agents:
                if agent not in infos:
                    infos[agent] = {}
                infos[agent]["escape_event"] = True
        return rewards, infos

    def _check_termination(self):
        """Check if the overall game should terminate."""
        active_evaders = []
        for evader in self.evaders:
            if evader not in self.captured_evaders and not self.terminations.get(evader, False):
                active_evaders.append(evader)

        game_over = False
        if len(active_evaders) == 0:
            game_over = True
        elif all(self.agent_positions[e]== self.safe_node for e in active_evaders):
            game_over = True

        if game_over:
            for agent in self.agents:
                if not self.terminations.get(agent, False):
                    self.terminations[agent] = True




    def render(self):
        """Render the environment state with a fixed layout and static legend.
        Returns the matplotlib figure object in 'human' mode.
        """
        print(f"  Debug GPE.render: self.render_mode = {self.render_mode}")

        if self.render_mode is None:
            return None

        print(f"Timestep: {self.timestep}")
        print(f"Safe node: {self.safe_node}")
        print("Pursuer positions:", {p: self.agent_positions[p] for p in self.pursuers})
        print(
            "Evader positions (Active/Captured):",
            {
                e: f"{self.agent_positions[e]}{' (Captured)' if e in self.captured_evaders else ''}"
                for e in self.evaders
            },
        )

        if self.render_mode == "human":
            legend_handles = [
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Pursuer",
                    markerfacecolor="r",
                    markersize=10,
                    linestyle="None",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Evader",
                    markerfacecolor="b",
                    markersize=10,
                    linestyle="None",
                ),
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Captured",
                    markerfacecolor="y",
                    markersize=8,
                    linestyle="None",
                ),  # Smaller marker
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label="Safe",
                    markerfacecolor="g",
                    markersize=12,
                    linestyle="None",
                ),  # Larger marker
            ]
            legend_labels = [h.get_label() for h in legend_handles]

            if not hasattr(self, "fig") or not plt.fignum_exists(self.fig.number):
                self.fig = plt.figure(figsize=(12, 8))
                self.ax = self.fig.add_subplot(111)
                self.fig.subplots_adjust(left=0.05, right=0.80, bottom=0.05, top=0.95)

                if not hasattr(self, "pos_layout") or self.timestep == 0:
                    print(
                        f"Calculating '{self.layout_algorithm}' layout for rendering..."
                    )
                    try:
                        if self.layout_algorithm == "spring":
                            self.pos_layout = nx.spring_layout(
                                self.graph,
                                k=0.5,
                                iterations=50,  # Default spring params
                            )
                        elif self.layout_algorithm == "kamada_kawai":
                            self.pos_layout = nx.kamada_kawai_layout(self.graph)
                        elif self.layout_algorithm == "spectral":
                            self.pos_layout = nx.spectral_layout(self.graph)
                        elif self.layout_algorithm == "grid":
                            if self.grid_m is not None and self.grid_n is not None:
                                self.pos_layout = {
                                    node: (
                                        node % self.grid_n,
                                        self.grid_m - 1 - node // self.grid_n,
                                    )
                                    for node in self.graph.nodes()
                                }
                                print(
                                    f"  Using stored grid dimensions: {self.grid_m}x{self.grid_n}"
                                )
                            else:
                                print(
                                    f"  Warning: 'grid' layout selected but m/n dimensions not available. Falling back to spring layout."
                                )
                                self.pos_layout = nx.spring_layout(
                                    self.graph, k=0.5, iterations=50
                                )
                        else:
                            print(
                                f"  Warning: Unknown layout_algorithm '{self.layout_algorithm}'. Falling back to spring layout."
                            )
                            self.pos_layout = nx.spring_layout(
                                self.graph, k=0.5, iterations=50
                            )
                    except Exception as e:
                        print(
                            f"  Error calculating layout '{self.layout_algorithm}': {e}. Falling back to spring layout."
                        )
                        self.pos_layout = nx.spring_layout(
                            self.graph, k=0.5, iterations=50
                        )

            x_coords, y_coords = zip(*self.pos_layout.values())
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            x_margin = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
            y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
            self.render_xlim = (x_min - x_margin, x_max + x_margin)
            self.render_ylim = (y_min - y_margin, y_max + y_margin)

            plt.figure(self.fig.number)
            self.ax.clear()
            self.ax.set_xlim(self.render_xlim)
            self.ax.set_ylim(self.render_ylim)

            nx.draw_networkx_nodes(
                self.graph,
                self.pos_layout,
                ax=self.ax,
                node_color="lightgray",
                node_size=400,
            )
            nx.draw_networkx_edges(
                self.graph,
                self.pos_layout,
                ax=self.ax,
                edge_color="gray",
                width=0.75,
                alpha=0.6,
            )

            for pursuer in self.pursuers:
                if pursuer in self.agent_positions:
                    position = self.agent_positions[pursuer]
                    self.ax.plot(
                        self.pos_layout[position][0],
                        self.pos_layout[position][1],
                        "ro",
                        markersize=10,
                    )
                    self.ax.annotate(
                        pursuer,
                        (self.pos_layout[position][0], self.pos_layout[position][1]),
                        xytext=(8, 8),
                        textcoords="offset points",
                        color="red",
                        fontsize=7,
                    )

            for evader in self.evaders:
                if evader in self.agent_positions:
                    position = self.agent_positions[evader]
                    is_captured = evader in self.captured_evaders
                    color = "yo" if is_captured else "bo"
                    marker_size = 8 if is_captured else 10

                    self.ax.plot(
                        self.pos_layout[position][0],
                        self.pos_layout[position][1],
                        color,
                        markersize=marker_size,
                    )
                    anno_text = f"{evader}{' (C)' if is_captured else ''}"
                    anno_color = "orange" if is_captured else "blue"
                    self.ax.annotate(
                        anno_text,
                        (self.pos_layout[position][0], self.pos_layout[position][1]),
                        xytext=(8, -12),
                        textcoords="offset points",
                        color=anno_color,
                        fontsize=7,
                        ha="left",
                    )

            if self.safe_node is not None:
                self.ax.plot(
                    self.pos_layout[self.safe_node][0],
                    self.pos_layout[self.safe_node][1],
                    "go",
                    markersize=12,
                )
                self.ax.annotate(
                    "SAFE",
                    (
                        self.pos_layout[self.safe_node][0],
                        self.pos_layout[self.safe_node][1],
                    ),
                    xytext=(0, 15),
                    textcoords="offset points",
                    color="green",
                    fontsize=9,
                    ha="center",
                )

            self.ax.set_title(f"Timestep: {self.timestep}", fontsize=14)
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            self.ax.legend(
                handles=legend_handles,
                labels=legend_labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0.0,
                fontsize="medium",
            )

            return self.fig
        else:
            print(
                f"  Debug GPE.render: Exiting because render_mode is not 'human' (it is '{self.render_mode}')."
            )
            return None

    def close(self):
        """Close the rendering window."""
        if hasattr(self, "fig") and plt.fignum_exists(self.fig.number):
            plt.close(self.fig.number)
            # Reset layout and limits on close
            if hasattr(self, "pos_layout"):
                delattr(self, "pos_layout")
            if hasattr(self, "render_xlim"):
                delattr(self, "render_xlim")
            if hasattr(self, "render_ylim"):
                delattr(self, "render_ylim")

    def shortest_path_action(self, agent):
        """
        Args:
            agent_type: "pursuer" or "evader"
            agent_id: Agent ID (0 to num_agents-1)

        Returns:
            action: Recommended action (node index) for the agent
        """
        current_pos = self.agent_positions[agent]

        if agent.startswith("pursuer"):
            min_distance = float("inf")
            target_pos = None
            target_path = None

            for evader in self.evaders:
                if evader in self.captured_evaders:
                    continue

                evader_pos = self.agent_positions[evader]
                try:
                    path = nx.shortest_path(self.graph, current_pos, evader_pos)
                    distance = len(path) - 1

                    if distance < min_distance:
                        min_distance = distance
                        target_pos = evader_pos
                        target_path = path
                except nx.NetworkXNoPath:
                    continue

            if target_path and len(target_path) > 1:
                return target_path[1]

        elif agent.startswith("evader"):
            try:
                path = nx.shortest_path(self.graph, current_pos, self.safe_node)
                if len(path) > 1:
                    return path[1]
            except nx.NetworkXNoPath:
                pass

            if not path or len(path) <= 1:
                min_distance = float("inf")
                nearest_pursuer_pos = None

                for pursuer in self.pursuers:
                    pursuer_pos = self.agent_positions[pursuer]
                    try:
                        distance = (
                            len(nx.shortest_path(self.graph, current_pos, pursuer_pos))
                            - 1
                        )
                        if distance < min_distance:
                            min_distance = distance
                            nearest_pursuer_pos = pursuer_pos
                    except nx.NetworkXNoPath:
                        continue

                if nearest_pursuer_pos is not None:
                    neighbors = self.adj[current_pos]
                    best_neighbor = None
                    max_escape_distance = -1

                    for neighbor in neighbors:
                        try:
                            escape_distance = (
                                len(
                                    nx.shortest_path(
                                        self.graph, neighbor, nearest_pursuer_pos
                                    )
                                )
                                - 1
                            )
                            if escape_distance > max_escape_distance:
                                max_escape_distance = escape_distance
                                best_neighbor = neighbor
                        except nx.NetworkXNoPath:
                            continue

                    if best_neighbor is not None:
                        return best_neighbor

        return current_pos
