import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import random
from copy import copy
import networkx as nx
from pettingzoo import ParallelEnv
from pettingzoo import AECEnv
import matplotlib.pyplot as plt


class GPE(ParallelEnv):
    """
    Graph Pursuit Evasion (GPE) Environment

    Multi-agent environment where pursuers try to catch evaders on a graph.
    Evaders attempt to reach a safe node while avoiding capture.

    Graph Representation:
    - Nodes: Agent locations
    - Edges: Valid movements
    - Pursuers: Capture evaders by surrounding them
    - Evaders: Reach safe node
    - Capture: Occurs when enough pursuers are adjacent to an evader

    Key Features:
    - Configurable graph size/connectivity
    - Multiple pursuers/evaders
    - Customizable capture mechanics
    - Multiple rendering modes
    - Based on PettingZoo ParallelEnv

    Follows Gymnasium interface:
    - Discrete action space
    - Dict observation space
    - Step-based episodes
    - Customizable termination
    """

    metadata = {
        "name": "graph_pe_v0",
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        num_nodes=100,
        num_edges=200,
        num_pursuers=5,
        num_evaders=7,
        capture_distance=1,
        required_captors=1,
        max_steps=200,
        seed=None,
        graph=None,
        render_mode=None,
        p_act=1,
        capture_reward_pursuer=20.0,
        capture_reward_evader=-20.0,
        escape_reward_evader=100.0,
        escape_reward_pursuer=-100.0,
        stay_penalty=-0.1,
    ):
        """
        Initialize the GPE environment.

        Args:
            num_nodes: Number of nodes in the graph.
            num_edges: Approximate number of edges.
            num_pursuers: Number of pursuer agents.
            num_evaders: Number of evader agents.
            capture_distance: Distance for capture (1=adjacent).
            required_captors: Number of adjacent pursuers for capture.
            max_steps: Max steps per episode.
            seed: Random seed.
            graph: Predefined networkx graph object. If None, generates random graph.
            render_mode: 'human', 'rgb_array', or None.
            p_act: Probability of ignoring action and staying.
            capture_reward_pursuer: Reward for pursuers when capturing an evader.
            capture_reward_evader: Reward for evader when being captured.
            escape_reward_evader: Reward for evader when reaching safe node.
            escape_reward_pursuer: Reward for pursuers when evader reaches safe node.
        """
        super().__init__()
        self.np_random = np.random.RandomState(seed)
        self.p_act = p_act

        self.custom_graph = graph
        if self.custom_graph is None:
            m = int(np.floor(np.sqrt(num_nodes)))
            n = int(np.ceil(num_nodes / m))
            actual_num_nodes = m * n
            self.num_nodes = actual_num_nodes
            print(
                f"GPE Init: Grid graph detected. Actual num_nodes set to {self.num_nodes}"
            )
        else:
            self.num_nodes = self.custom_graph.number_of_nodes()
            print(
                f"GPE Init: Custom graph provided. Actual num_nodes set to {self.num_nodes}"
            )

        self.num_pursuers = num_pursuers
        self.num_evaders = num_evaders
        self.capture_distance = capture_distance
        self.required_captors = required_captors
        self.max_steps = max_steps
        self.render_mode = render_mode

        self.pursuers = [f"pursuer_{i}" for i in range(self.num_pursuers)]
        self.evaders = [f"evader_{i}" for i in range(self.num_evaders)]
        self.possible_agents = self.pursuers + self.evaders

        self._initialize_spaces()

        self.graph = None
        self.safe_node = None
        self.agent_positions = {}
        self.timestep = 0
        self.agents = []
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

    def _initialize_spaces(self):
        """Initialize action and observation spaces for all agents."""
        self.action_spaces = {
            agent: Discrete(self.num_nodes) for agent in self.possible_agents
        }

        pursuer_obs_size = (
            1  # position
            + self.num_pursuers  # pursuers positions
            + self.num_evaders  # evaders positions
            + self.num_nodes  # adjacency
            + self.num_nodes  # action_mask
        )
        evader_obs_size = (
            1  # position
            + 1  # safe_node (extra item for evader)
            + self.num_pursuers
            + self.num_evaders
            + self.num_nodes
            + self.num_nodes
        )
        flat_obs_size = max(pursuer_obs_size, evader_obs_size)

        self.observation_spaces = {
            agent: Box(
                low=-1,
                high=self.num_nodes,
                shape=(flat_obs_size,),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        self._pursuer_padding_size = flat_obs_size - pursuer_obs_size

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

    def reset(self, seed=None, options=None):
        """Reset the environment. Attempts safe initial placement, falls back to random if needed."""

        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        self.graph = self._generate_graph()
        self.timestep = 0
        self.agents = self.possible_agents.copy()
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.captured_evaders = set()
        all_nodes = list(self.graph.nodes())
        all_nodes_set = set(all_nodes)  # Use set for faster operations
        self.agent_positions = {}

        # Ensure enough nodes total
        if len(all_nodes) < self.num_pursuers + self.num_evaders:
            raise ValueError(
                f"Not enough nodes ({len(all_nodes)}) in the graph for all agents ({self.num_pursuers + self.num_evaders})."
            )

        # 1. Place Pursuers
        pursuer_nodes = self.np_random.choice(
            all_nodes, size=self.num_pursuers, replace=False
        )
        for i, agent in enumerate(self.pursuers):
            self.agent_positions[agent] = int(pursuer_nodes[i])

        # Calculate nodes available *after* placing pursuers (needed for fallback)
        pursuer_nodes_set = set(pursuer_nodes)
        remaining_nodes = list(all_nodes_set - pursuer_nodes_set)

        # Calculate Danger Zone
        danger_zone = set(pursuer_nodes)
        for p_node in pursuer_nodes:
            try:
                neighbors = set(self.graph.neighbors(p_node))
                danger_zone.update(neighbors)
            except nx.NetworkXError:
                print(
                    f"Warning: Pursuer node {p_node} not found in graph during danger zone calculation."
                )

        # Calculate Safe Zone for Evaders
        safe_zone_for_evaders = list(all_nodes_set - danger_zone)

        # Attempt to place evaders in the safe zone
        if len(safe_zone_for_evaders) >= self.num_evaders:
            # Place Evaders in the Safe Zone (Ideal Case)
            print("Placing evaders in calculated safe zone.")
            evader_nodes = self.np_random.choice(
                safe_zone_for_evaders, size=self.num_evaders, replace=False
            )
            for i, agent in enumerate(self.evaders):
                self.agent_positions[agent] = int(evader_nodes[i])
        else:
            # Fallback: Place evaders randomly among remaining nodes (Not guaranteed safe)
            print(
                f"Warning: Safe zone ({len(safe_zone_for_evaders)} nodes) is too small for {self.num_evaders} evaders. "
                f"Falling back to random placement in remaining {len(remaining_nodes)} nodes."
            )
            # Double-check if enough remaining nodes exist even for fallback
            if len(remaining_nodes) < self.num_evaders:
                # This should be caught by the initial total node check, but as a safeguard:
                raise ValueError(
                    f"Fallback failed: Not enough remaining nodes ({len(remaining_nodes)}) "
                    f"to place {self.num_evaders} evaders after placing pursuers."
                )
            # Place evaders randomly in the nodes not occupied by pursuers
            evader_nodes = self.np_random.choice(
                remaining_nodes, size=self.num_evaders, replace=False
            )
            for i, agent in enumerate(self.evaders):
                self.agent_positions[agent] = int(evader_nodes[i])

        # Choose a safe node (must not be occupied by any agent)
        occupied_nodes = set(self.agent_positions.values())
        available_nodes_for_safe_node = list(all_nodes_set - occupied_nodes)
        if available_nodes_for_safe_node:
            self.safe_node = self.np_random.choice(available_nodes_for_safe_node)
        else:
            # ... (existing fallback logic for safe node placement remains the same) ...
            pursuer_positions_set = {
                self.agent_positions[p] for p in self.pursuers
            }  # Need to recalculate based on final positions
            evader_positions_set = {
                self.agent_positions[e]
                for e in self.evaders
                if e in self.agent_positions
            }  # Get final evader positions
            non_pursuer_nodes = list(all_nodes_set - pursuer_positions_set)
            if non_pursuer_nodes:
                self.safe_node = self.np_random.choice(non_pursuer_nodes)
            else:  # Should be impossible if checks above work
                print(
                    "Warning: Cannot find a suitable safe node placement. Placing randomly."
                )
                self.safe_node = self.np_random.choice(all_nodes)

        # Generate initial observations for all agents
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        # Return standard reset format
        return observations, self.infos

    def _get_observation(self, agent):
        """Generate observation for an agent, including action mask."""

        position = self.agent_positions[agent]
        neighbors = list(self.graph.neighbors(position))

        adjacency_obs = np.zeros(self.num_nodes, dtype=np.float32)
        adjacency_obs[neighbors] = 1.0

        pursuer_positions = np.array(
            [self.agent_positions[p] for p in self.pursuers],
            dtype=np.float32,
        )
        evader_positions = np.array(
            [
                self.agent_positions[e] if e not in self.captured_evaders else -1.0
                for e in self.evaders
            ],
            dtype=np.float32,
        )

        action_mask = np.zeros(self.num_nodes, dtype=np.float32)
        valid_action_indices = [position] + neighbors
        action_mask[valid_action_indices] = 1.0

        if agent.startswith("evader"):
            observation_vector = np.concatenate(
                [
                    np.array([float(position)], dtype=np.float32),
                    pursuer_positions,
                    evader_positions,
                    adjacency_obs,
                    action_mask,
                    np.array([float(self.safe_node)], dtype=np.float32),
                ]
            ).astype(np.float32)
        else:
            base_vector = np.concatenate(
                [
                    np.array([float(position)], dtype=np.float32),
                    pursuer_positions,
                    evader_positions,
                    adjacency_obs,
                    action_mask,
                ]
            )
            padding = np.zeros(self._pursuer_padding_size, dtype=np.float32)
            observation_vector = np.concatenate([base_vector, padding]).astype(
                np.float32
            )

        return observation_vector

    def step(self, actions):
        """Execute actions for ---all-- agents and return new observations."""

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

        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        next_positions = self.agent_positions.copy()
        for agent, action in actions.items():
            if agent not in self.agents:
                continue

            current_position = self.agent_positions[agent]
            if self.np_random.random() > self.p_act:
                effective_action = current_position
            else:
                effective_action = action

            is_valid_move = (
                effective_action == current_position
                or effective_action in self.graph.neighbors(current_position)
            )
            if is_valid_move:
                next_positions[agent] = effective_action
            else:
                self.infos[agent]["invalid_action"] = True

            if effective_action == current_position:
                self.rewards[agent] += -self.stay_penalty

        self.agent_positions = next_positions

        self._check_captures()
        self._check_safe_arrivals()
        self._check_termination()

        self.timestep += 1

        if self.timestep >= self.max_steps:
            for agent in self.agents:
                if not self.terminations[agent]:
                    self.truncations[agent] = True

        active_agents_next_step = []
        for agent in self.agents:
            if not self.terminations[agent] and not self.truncations[agent]:
                active_agents_next_step.append(agent)
        self.agents = active_agents_next_step

        observations = {agent: self._get_observation(agent) for agent in actions.keys()}

        return (
            observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def _check_captures(self):
        """Check if any evaders have been captured."""
        capture_occurred = False

        for evader in self.evaders:
            if evader in self.captured_evaders:
                continue

            evader_pos = self.agent_positions[evader]
            evader_neighbors = list(self.graph.neighbors(evader_pos))

            adjacent_pursuers = 0
            for pursuer in self.pursuers:
                pursuer_pos = self.agent_positions[pursuer]
                if pursuer_pos == evader_pos or pursuer_pos in evader_neighbors:
                    adjacent_pursuers += 1

            if adjacent_pursuers >= self.required_captors:
                self.captured_evaders.add(evader)
                for pursuer in self.pursuers:
                    self.rewards[pursuer] += self.capture_reward_pursuer
                self.rewards[evader] += self.capture_reward_evader
                self.terminations[evader] = True
                capture_occurred = True

            if capture_occurred:
                for agent in self.agents:
                    self.infos[agent]["capture"] = True

    def _check_safe_arrivals(self):
        """Check if any evaders have reached the safe node."""
        for evader in self.evaders:
            if evader in self.captured_evaders:
                continue

            if self.agent_positions[evader] == self.safe_node:
                if evader in self.rewards:
                    self.rewards[evader] += self.escape_reward_evader
                    self.terminations[evader] = True
                    for pursuer in self.pursuers:
                        if pursuer in self.rewards:
                            self.rewards[pursuer] += self.escape_reward_pursuer

    def _check_termination(self):
        """Check if the overall game should terminate."""
        active_evaders = set(self.evaders) - self.captured_evaders
        evaders_at_safe_node = sum(
            1 for e in active_evaders if self.agent_positions[e] == self.safe_node
        )

        if len(active_evaders) == 0 or evaders_at_safe_node == len(active_evaders):
            for agent in self.agents:
                if not self.terminations[agent]:
                    self.terminations[agent] = True

    def observation_space(self, agent):
        """Return the observation space for a specific agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Return the action space for a specific agent."""
        return self.action_spaces[agent]

    def render(self):
        """Render the environment state.
        Displays the current state in the same figure window, creating animation when called in a loop.
        Captured evaders are shown in yellow.
        """
        if self.render_mode is None:
            return

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
            if not hasattr(self, "fig") or not plt.fignum_exists(self.fig.number):
                self.fig = plt.figure(figsize=(12, 8))
                self.ax = self.fig.add_subplot(111)
                if not hasattr(self, "pos_layout") or self.timestep == 0:
                    try:
                        m, n = self.graph.graph["dim"]
                        self.pos_layout = {
                            node: (data["pos"][1], -data["pos"][0])
                            for node, data in self.graph.nodes(data=True)
                        }
                        print("Using grid layout for rendering.")
                    except KeyError:
                        print("Using spring layout for rendering.")
                        self.pos_layout = nx.spring_layout(
                            self.graph, k=1, iterations=50
                        )

            plt.figure(self.fig.number)
            self.ax.clear()

            nx.draw_networkx_nodes(
                self.graph,
                self.pos_layout,
                ax=self.ax,
                node_color="lightgray",
                node_size=500,
            )
            nx.draw_networkx_edges(
                self.graph, self.pos_layout, ax=self.ax, edge_color="gray", width=1
            )
            nx.draw_networkx_labels(
                self.graph, self.pos_layout, ax=self.ax, font_size=8
            )

            pursuer_handles = []
            for pursuer in self.pursuers:
                if pursuer in self.agent_positions:
                    position = self.agent_positions[pursuer]
                    (handle,) = self.ax.plot(
                        self.pos_layout[position][0],
                        self.pos_layout[position][1],
                        "ro",
                        markersize=15,
                        label=pursuer,
                    )
                    pursuer_handles.append(handle)
                    self.ax.annotate(
                        pursuer,
                        (self.pos_layout[position][0], self.pos_layout[position][1]),
                        xytext=(10, 10),
                        textcoords="offset points",
                        color="red",
                        fontsize=8,
                    )

            evader_handles = []
            captured_evader_handles = []
            for evader in self.evaders:
                if evader in self.agent_positions:
                    position = self.agent_positions[evader]
                    is_captured = evader in self.captured_evaders

                    color = "yo" if is_captured else "bo"
                    label_suffix = " (Captured)" if is_captured else ""
                    marker_size = 12 if is_captured else 15

                    (handle,) = self.ax.plot(
                        self.pos_layout[position][0],
                        self.pos_layout[position][1],
                        color,
                        markersize=marker_size,
                        label=f"{evader}{label_suffix}",
                    )

                    if is_captured:
                        captured_evader_handles.append(handle)
                        self.ax.annotate(
                            f"{evader}\n(Captured)",
                            (
                                self.pos_layout[position][0],
                                self.pos_layout[position][1],
                            ),
                            xytext=(10, -15),
                            textcoords="offset points",
                            color="orange",
                            fontsize=7,
                            ha="left",
                        )
                    else:
                        evader_handles.append(handle)
                        self.ax.annotate(
                            evader,
                            (
                                self.pos_layout[position][0],
                                self.pos_layout[position][1],
                            ),
                            xytext=(10, -10),
                            textcoords="offset points",
                            color="blue",
                            fontsize=8,
                            ha="left",
                        )

            if self.safe_node is not None:
                (safe_handle,) = self.ax.plot(
                    self.pos_layout[self.safe_node][0],
                    self.pos_layout[self.safe_node][1],
                    "go",
                    markersize=20,
                    label="Safe Node",
                )
                self.ax.annotate(
                    "SAFE",
                    (
                        self.pos_layout[self.safe_node][0],
                        self.pos_layout[self.safe_node][1],
                    ),
                    xytext=(0, 20),
                    textcoords="offset points",
                    color="green",
                    fontsize=10,
                    ha="center",
                )

            self.ax.set_title(f"Timestep: {self.timestep}", fontsize=14)
            self.ax.set_xticks([])
            self.ax.set_yticks([])

            all_handles = pursuer_handles + evader_handles + captured_evader_handles
            if hasattr(self, "safe_handle"):
                all_handles.append(safe_handle)
            valid_handles = [
                h
                for h in all_handles
                if h.get_label() and not h.get_label().startswith("_")
            ]
            labels = [h.get_label() for h in valid_handles]

            max_legend_entries = 15
            if len(valid_handles) > max_legend_entries:
                priority_handles = pursuer_handles + evader_handles
                if hasattr(self, "safe_handle"):
                    priority_handles.append(safe_handle)
                priority_handles += captured_evader_handles[
                    : max_legend_entries - len(priority_handles)
                ]
                valid_handles = [
                    h
                    for h in priority_handles
                    if h.get_label() and not h.get_label().startswith("_")
                ]
                labels = [h.get_label() for h in valid_handles]
                if len(all_handles) > len(valid_handles):
                    from matplotlib.lines import Line2D

                    dummy_handle = Line2D(
                        [0], [0], marker="None", linestyle="None", label="..."
                    )
                    valid_handles.append(dummy_handle)
                    labels.append("...")

            self.ax.legend(
                valid_handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.02, 1),
                borderaxespad=0.0,
                fontsize="small",
            )

            self.fig.tight_layout(rect=[0, 0, 0.85, 1])

            plt.pause(0.5)

    def close(self):
        """Close the rendering window."""
        if hasattr(self, "fig") and plt.fignum_exists(self.fig.number):
            plt.close(self.fig.number)
            if hasattr(self, "pos_layout"):
                delattr(self, "pos_layout")

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
                    neighbors = list(self.graph.neighbors(current_pos))
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
