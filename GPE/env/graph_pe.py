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
        # Set random seed
        self.np_random = np.random.RandomState(seed)

        self.p_act = p_act
        # Environment parameters
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_pursuers = num_pursuers
        self.num_evaders = num_evaders
        self.capture_distance = capture_distance
        self.required_captors = required_captors
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Create agent lists
        self.pursuers = [f"pursuer_{i}" for i in range(self.num_pursuers)]
        self.evaders = [f"evader_{i}" for i in range(self.num_evaders)]
        self.possible_agents = self.pursuers + self.evaders

        # Graph structure (optional custom graph)
        self.custom_graph = graph

        # Initialize action and observation spaces
        self._initialize_spaces()

        # State variables initialized in reset()
        self.graph = None
        self.safe_node = None
        self.agent_positions = {}
        self.timestep = 0
        self.agents = []  # List of currently active agents
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

    # TODO: Based on the document, it should be defined as observation_space/action_space with return Discrete(..)
    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized, reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    # @functools.lru_cache(maxsize=None)
    # def observation_space(self, agent):
    #     # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
    #     return Discrete(4)

    # # Action space should be defined here.
    # # If your spaces change over time, remove this line (disable caching).
    # @functools.lru_cache(maxsize=None)
    # def action_space(self, agent):
    #     return Discrete(3)

    def _initialize_spaces(self):
        """Initialize action and observation spaces for all agents."""
        # Action space: Choose any node as target (validity checked in step)
        self.action_spaces = {
            agent: Discrete(self.num_nodes) for agent in self.possible_agents
        }

        # Observation space: Define a flat Box space for all agents
        # Calculate the size needed for the flattened observation vector.
        # Ensure pursuers and evaders have the same final size by padding.
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
        # Use the maximum size for both, we will pad pursuers' observations
        flat_obs_size = max(pursuer_obs_size, evader_obs_size)

        # Define the observation space as a single Box for all agents
        # Note: The bounds need to accommodate all possible values (-1 to num_nodes-1 for positions)
        self.observation_spaces = {
            agent: Box(
                low=-1,
                high=self.num_nodes,
                shape=(flat_obs_size,),
                dtype=np.float32,  # check dtype
            )  # Use float32 for SB3 compatibility
            for agent in self.possible_agents
        }
        # Total Size : n^2m^2
        # Store component sizes for easy concatenation later (optional but helpful)
        self._pursuer_padding_size = flat_obs_size - pursuer_obs_size

    def _generate_graph(self):
        """Generate a random graph or use the provided one.
        MARK: It should be changed to a certain range of connection for each node.
        """

        if self.custom_graph is not None:
            return copy(
                self.custom_graph
            )  # Return a copy to avoid modifying the original

        # Generate random graph (Erdős-Rényi model)
        # Edge probability p calculated for approximate num_edges
        p = 2 * self.num_edges / (self.num_nodes * (self.num_nodes - 1))
        graph = nx.gnp_random_graph(
            self.num_nodes, p, seed=self.np_random.randint(10000)
        )

        # Ensure the graph is connected
        if not nx.is_connected(graph):
            # Take the largest connected component
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc).copy()  # Work with the largest component

            # Add random edges between remaining components to connect them
            nodes = list(graph.nodes())  # Get nodes of the current subgraph
            components = list(nx.connected_components(graph))

            while len(components) > 1:
                # Pick two random components
                comp1 = random.choice(components)
                components.remove(comp1)
                comp2 = random.choice(components)
                # components.remove(comp2) # Bug fix: Don't remove the second component yet

                # Pick one random node from each component and add an edge
                node1 = random.choice(list(comp1))
                node2 = random.choice(list(comp2))
                graph.add_edge(node1, node2)

                # Recalculate components after adding edge
                components = list(nx.connected_components(graph))

        return graph

    def reset(self, seed=None, options=None):
        """Reset the environment and return initial observations."""

        # Reset random seed if provided
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
        self.agent_positions = {}

        # Place pursuers first
        pursuer_nodes = self.np_random.choice(
            all_nodes, size=self.num_pursuers, replace=False
        )
        for i, agent in enumerate(self.pursuers):
            self.agent_positions[agent] = int(pursuer_nodes[i])

        remaining_nodes = list(set(all_nodes) - set(pursuer_nodes))

        # Place evaders next
        evader_nodes = self.np_random.choice(
            remaining_nodes,
            size=min(self.num_evaders, len(remaining_nodes)),
            replace=False,
        )
        for i, agent in enumerate(self.evaders):
            if i < len(evader_nodes):
                self.agent_positions[agent] = int(evader_nodes[i])
            else:
                # Extreme fallback: If no nodes left in remaining_nodes (very unlikely)
                self.agent_positions[agent] = int(
                    self.np_random.choice(list(set(all_nodes) - set(pursuer_nodes)))
                )

        # Choose a safe node that is not occupied by any agent
        occupied_nodes = set(self.agent_positions.values())
        available_nodes = list(set(all_nodes) - occupied_nodes)
        if available_nodes:
            self.safe_node = self.np_random.choice(available_nodes)
        else:
            evader_positions = {self.agent_positions[e] for e in self.evaders}
            all_nodes_set = set(all_nodes)
            non_evader_nodes = list(all_nodes_set - evader_positions)
            if non_evader_nodes:
                self.safe_node = self.np_random.choice(non_evader_nodes)
            else:
                self.safe_node = self.np_random.choice(all_nodes)

        # Generate initial observations for all agents
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        # Return standard reset format
        return observations, self.infos

    def _get_observation(self, agent):
        """Generate observation for an agent, including action mask."""

        position = self.agent_positions[agent]
        neighbors = list(self.graph.neighbors(position))

        # Adjacency observation: 1 if neighbor, 0 otherwise, length=num_nodes
        adjacency_obs = np.zeros(self.num_nodes, dtype=np.float32)  # Use float32
        adjacency_obs[neighbors] = 1.0

        # Get agent positions
        pursuer_positions = np.array(
            [self.agent_positions[p] for p in self.pursuers],
            dtype=np.float32,  # Use float32
        )
        evader_positions = np.array(
            [  # Use -1.0 for captured evaders
                self.agent_positions[e] if e not in self.captured_evaders else -1.0
                for e in self.evaders
            ],
            dtype=np.float32,  # Use float32
        )

        # Calculate action mask
        action_mask = np.zeros(self.num_nodes, dtype=np.float32)  # Use float32
        valid_action_indices = [position] + neighbors  # Current node + neighbor nodes
        action_mask[valid_action_indices] = 1.0

        # Concatenate components into a flat vector
        if agent.startswith("evader"):
            # Evader observation order: pos, safe, pursuers, evaders, adj, mask
            observation_vector = np.concatenate(
                [
                    np.array([float(position)], dtype=np.float32),  # ego position
                    pursuer_positions,
                    evader_positions,
                    adjacency_obs,
                    action_mask,
                    np.array([float(self.safe_node)], dtype=np.float32),  # safe node
                ]  # 4*num_nodes
            ).astype(np.float32)
        else:  # Pursuer
            # Pursuer observation order: pos, pursuers, evaders, adj, mask, [padding]
            base_vector = np.concatenate(
                [
                    np.array([float(position)], dtype=np.float32),
                    pursuer_positions,
                    evader_positions,
                    adjacency_obs,
                    action_mask,
                ]
            )
            # Add padding if necessary to match the evader's flat size
            padding = np.zeros(
                self._pursuer_padding_size, dtype=np.float32
            )  # Use 0.0 for padding
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
                next_positions[agent] = effective_action  # Store intended next position
            else:
                self.infos[agent]["invalid_action"] = True

            if effective_action == current_position:
                self.rewards[agent] += -self.stay_penalty  # Penalize staying put

        self.agent_positions = next_positions

        # !!!Check also before the action!!!
        self._check_captures()
        self._check_safe_arrivals()
        self._check_termination()  # This method updates self.terminations for all agents

        self.timestep += 1

        if self.timestep >= self.max_steps:
            for agent in self.agents:
                if not self.terminations[agent]:
                    self.truncations[agent] = True

        active_agents_next_step = []
        for agent in self.agents:
            if not self.terminations[agent] and not self.truncations[agent]:
                active_agents_next_step.append(agent)
        self.agents = active_agents_next_step  # Update self.agents for the next step's input validation

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
                self.rewards[evader] += self.escape_reward_evader
                self.terminations[evader] = True
                for pursuer in self.pursuers:
                    self.rewards[pursuer] += self.escape_reward_pursuer

    def _check_termination(self):
        """Check if the overall game should terminate."""
        # Game ends if all evaders are either captured or have reached the safe node
        active_evaders = (
            set(self.evaders) - self.captured_evaders
        )  # Evaders not yet captured
        # Count how many of the *active* evaders are at the safe node
        evaders_at_safe_node = sum(
            1 for e in active_evaders if self.agent_positions[e] == self.safe_node
        )

        # Condition: No active evaders left OR all active evaders are at the safe node
        if len(active_evaders) == 0 or evaders_at_safe_node == len(active_evaders):
            for agent in self.agents:  # Use the current self.agents list
                if not self.terminations[agent]:
                    self.terminations[agent] = True  # Set global termination

    def observation_space(self, agent):
        """Return the observation space for a specific agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Return the action space for a specific agent."""
        return self.action_spaces[agent]

    def render(self):
        """Render the environment state.
        Displays the current state in the same figure window, creating animation when called in a loop.
        """
        if self.render_mode is None:
            return

        # Print basic info to console
        print(f"Timestep: {self.timestep}")
        print(f"Safe node: {self.safe_node}")
        print("Pursuer positions:", {p: self.agent_positions[p] for p in self.pursuers})
        print(
            "Active Evader positions:",  # Clarified label
            {
                e: self.agent_positions[e]
                for e in self.evaders
                if e not in self.captured_evaders  # Only show non-captured evaders
            },
        )
        print("Captured evaders:", self.captured_evaders)

        if self.render_mode == "human":
            # Use the same figure window for animation effect
            if not hasattr(self, "fig") or not plt.fignum_exists(
                self.fig.number
            ):  # Check if figure exists/was closed
                self.fig = plt.figure(figsize=(12, 8))
                self.ax = self.fig.add_subplot(111)  # Add axes object
                # Calculate layout once per episode or if graph changes
                # (Assuming graph doesn't change mid-episode for performance)
                if not hasattr(self, "pos_layout") or self.timestep == 0:
                    self.pos_layout = nx.spring_layout(self.graph, k=1, iterations=50)

            # Activate the figure and clear previous drawing
            plt.figure(self.fig.number)
            self.ax.clear()  # Clear axes instead of clf()

            # Draw graph structure using pre-calculated layout
            # Nodes
            nx.draw_networkx_nodes(
                self.graph,
                self.pos_layout,
                ax=self.ax,
                node_color="lightgray",
                node_size=500,
            )
            # Edges
            nx.draw_networkx_edges(
                self.graph, self.pos_layout, ax=self.ax, edge_color="gray", width=1
            )
            # Node labels
            nx.draw_networkx_labels(
                self.graph, self.pos_layout, ax=self.ax, font_size=8
            )

            # Draw agents
            # Pursuers
            pursuer_handles = []
            for pursuer in self.pursuers:
                position = self.agent_positions[pursuer]
                (handle,) = self.ax.plot(  # Use self.ax.plot
                    self.pos_layout[position][0],
                    self.pos_layout[position][1],
                    "ro",
                    markersize=15,
                    label=pursuer,  # Label for legend
                )
                pursuer_handles.append(
                    handle
                )  # Store handle for potential legend filtering
                # Add pursuer annotation
                self.ax.annotate(  # Use self.ax.annotate
                    pursuer,
                    (self.pos_layout[position][0], self.pos_layout[position][1]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    color="red",
                    fontsize=8,
                )

            # Active Evaders
            evader_handles = []
            for evader in self.evaders:
                if evader not in self.captured_evaders:
                    position = self.agent_positions[evader]
                    (handle,) = self.ax.plot(  # Use self.ax.plot
                        self.pos_layout[position][0],
                        self.pos_layout[position][1],
                        "bo",
                        markersize=15,
                        label=evader,  # Label for legend
                    )
                    evader_handles.append(handle)
                    # Add evader annotation
                    self.ax.annotate(  # Use self.ax.annotate
                        evader,
                        (self.pos_layout[position][0], self.pos_layout[position][1]),
                        xytext=(10, -10),
                        textcoords="offset points",
                        color="blue",
                        fontsize=8,
                    )

            # Safe Node
            (safe_handle,) = self.ax.plot(  # Use self.ax.plot
                self.pos_layout[self.safe_node][0],
                self.pos_layout[self.safe_node][1],
                "go",
                markersize=20,
                label="Safe Node",
            )
            self.ax.annotate(  # Use self.ax.annotate
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

            # Add title with timestep
            self.ax.set_title(
                f"Timestep: {self.timestep}", fontsize=14
            )  # Use self.ax.set_title
            self.ax.set_xticks([])  # Hide axes ticks
            self.ax.set_yticks([])

            # Create legend (optional: filter handles if needed)
            # handles = pursuer_handles + evader_handles + [safe_handle]
            # labels = [h.get_label() for h in handles]
            # self.ax.legend(handles, labels, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
            self.ax.legend(
                loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0.0
            )

            # Adjust layout to prevent legend overlap
            self.fig.tight_layout(
                rect=[0, 0, 0.9, 1]
            )  # Adjust rect to make space for legend

            # Pause for animation
            plt.pause(0.5)

    def close(self):
        """Close the rendering window."""
        if hasattr(self, "fig") and plt.fignum_exists(self.fig.number):
            plt.close(self.fig.number)

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

                # Move away from the nearest pursuer
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
