import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import random
from copy import copy
import networkx as nx
from pettingzoo import ParallelEnv
from pettingzoo import AECEnv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from typing import Optional, Dict, List, Tuple, Any
from gymnasium.utils import EzPickle

# EzPickle


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
        escape_reward_evader=50.0,
        escape_reward_pursuer=-50.0,
        stay_penalty=-2.0,
        layout_algorithm="spring",
        allow_stay: bool = False,
        grid_m: Optional[int] = None,
        grid_n: Optional[int] = None,
        delta_distance_reward_pursuer_scale: Optional[float] = None,
        delta_distance_penalty_evader_scale: Optional[float] = None,
        time_penalty: float = 0.0,
    ):
        """
        Initialize the GPE environment.

        Args:
            num_nodes: Number of nodes in the graph.
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
            layout_algorithm (str): Layout algorithm for rendering ('spring', 'kamada_kawai', 'grid', 'spectral'). Defaults to 'spring'.
            allow_stay (bool): If False (default), agents must move to a neighbor. If True, staying in the current node is a valid action.
            delta_distance_reward_pursuer_scale (Optional[float]): Scaling factor for pursuer reward based on change in distance to evader (closer = positive delta). Defaults to None (disabled).
            delta_distance_penalty_evader_scale (Optional[float]): Scaling factor for evader penalty based on change in distance to pursuer (closer = positive delta). Should be negative. Defaults to None (disabled).
            time_penalty (float): Time penalty applied to all agents at each step.
        """
        # EzPickle.__init__(
        #     self,
        #     num_nodes=num_nodes,
        #     num_pursuers=num_pursuers,
        #     num_evaders=num_evaders,
        #     capture_distance=capture_distance,
        #     required_captors=required_captors,
        #     max_steps=max_steps,
        #     seed=seed,
        #     graph=graph,
        #     render_mode=render_mode,
        #     p_act=p_act,
        #     capture_reward_pursuer=capture_reward_pursuer,
        #     capture_reward_evader=capture_reward_evader,
        #     escape_reward_evader=escape_reward_evader,
        #     escape_reward_pursuer=escape_reward_pursuer,
        #     stay_penalty=stay_penalty,
        #     layout_algorithm=layout_algorithm,
        #     allow_stay=allow_stay,
        #     grid_m=grid_m,
        #     grid_n=grid_n,
        #     delta_distance_reward_pursuer_scale=delta_distance_reward_pursuer_scale,
        #     delta_distance_penalty_evader_scale=delta_distance_penalty_evader_scale,
        # )
        super().__init__()
        self.np_random = np.random.RandomState(seed)
        self.p_act = p_act
        self.layout_algorithm = layout_algorithm
        self.allow_stay = allow_stay
        self.grid_m = grid_m
        self.grid_n = grid_n

        # Store delta distance reward parameters
        self.delta_reward_pursuer_scale = delta_distance_reward_pursuer_scale
        self.delta_penalty_evader_scale = delta_distance_penalty_evader_scale
        # Initialize dictionary to store distances from the previous step
        self.last_pursuer_evader_distances: Dict[Tuple[str, str], float] = {}

        self.custom_graph = graph
        if self.custom_graph is None:
            m = int(np.floor(np.sqrt(num_nodes)))
            n = int(np.ceil(num_nodes / m))
            actual_num_nodes = m * n
            self.num_nodes = actual_num_nodes
            if self.grid_m == None:
                self.grid_m = m
            if self.grid_n == None:
                self.grid_n = n
            print(
                f"GPE Init: Grid graph generated ({m}x{n}). Actual num_nodes set to {self.num_nodes}"
            )
        else:
            self.num_nodes = self.custom_graph.number_of_nodes()
            print(
                f"GPE Init: Custom graph provided. Actual num_nodes set to {self.num_nodes}. Grid layout might require manual m, n if not inferred."
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
        self.time_penalty = time_penalty

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
        self.last_pursuer_evader_distances = {}  # Reset the distance cache
        all_nodes = list(self.graph.nodes())
        self.np_random.shuffle(all_nodes)  # Shuffle all nodes
        self.agent_positions = {}

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

        # Calculate and store initial distances after agents are placed
        self._update_last_distances()

        # Generate initial observations for all agents
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        # Return standard reset format
        return observations, self.infos

    def _update_last_distances(self):
        """Calculates and stores the current distances between all active P-E pairs."""
        self.last_pursuer_evader_distances = {}  # Clear previous step's cache
        active_pursuers = [p for p in self.pursuers if p in self.agent_positions]
        active_evaders = [
            e
            for e in self.evaders
            if e in self.agent_positions and e not in self.captured_evaders
        ]

        for p_agent in active_pursuers:
            p_pos = self.agent_positions[p_agent]
            for e_agent in active_evaders:
                e_pos = self.agent_positions[e_agent]
                pair = (p_agent, e_agent)
                try:
                    distance = nx.shortest_path_length(
                        self.graph, source=p_pos, target=e_pos
                    )
                    self.last_pursuer_evader_distances[pair] = distance
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    # Store infinity or a large number if no path, or handle as needed
                    self.last_pursuer_evader_distances[pair] = float("inf")
                    # print(f"Warning: No path or node not found between {p_agent} ({p_pos}) and {e_agent} ({e_pos}) during distance update.")
                    pass  # Silently handle for now

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
        if self.allow_stay:
            valid_action_indices = [position] + neighbors
        else:
            valid_action_indices = neighbors
        valid_action_indices = [
            idx for idx in valid_action_indices if 0 <= idx < self.num_nodes
        ]
        if valid_action_indices:
            action_mask[valid_action_indices] = 1.0
        elif self.allow_stay and (0 <= position < self.num_nodes):
            action_mask[position] = 1.0

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

    def step(self, actions: Dict[str, int]) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, Dict],
    ]:
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

        self.rewards = {agent: self.time_penalty for agent in actions.keys()}
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
                # Check if the intended action was valid based on allow_stay
                # (This adds robustness in case an invalid 'stay' action gets through somehow)
                is_staying = effective_action == current_position
                if (
                    not self.allow_stay
                    and is_staying
                    and list(self.graph.neighbors(current_position))
                ):
                    # If staying is not allowed AND the agent chose to stay AND neighbors exist
                    self.infos[agent]["invalid_action"] = True
                    # Keep agent at current_position implicitly by not updating next_positions[agent] below
                    # Or explicitly: next_positions[agent] = current_position
                else:
                    # Otherwise, the move (or stay if allowed) is fine
                    next_positions[agent] = effective_action
            else:
                # Original invalid move logic (e.g., moving to non-neighbor)
                self.infos[agent]["invalid_action"] = True

            # Apply stay penalty only if staying is allowed and occurred
            if self.allow_stay and effective_action == current_position:
                # Apply penalty only if staying is a valid option and the agent chose it
                if agent in self.rewards:
                    self.rewards[agent] += self.stay_penalty

        self.agent_positions = next_positions

        # --- Calculate Delta Distance Rewards (Optional) ---
        current_step_distances = {}  # Store distances calculated in *this* step
        if (
            self.delta_reward_pursuer_scale is not None
            or self.delta_penalty_evader_scale is not None
        ):
            active_pursuers = [
                p
                for p in self.pursuers
                if p in self.agent_positions and p in self.agents
            ]  # Check agent is still active
            active_evaders = [
                e
                for e in self.evaders
                if e in self.agent_positions
                and e not in self.captured_evaders
                and e in self.agents
            ]  # Check agent is still active

            for p_agent in active_pursuers:
                p_pos = self.agent_positions[p_agent]
                for e_agent in active_evaders:
                    e_pos = self.agent_positions[e_agent]
                    pair = (p_agent, e_agent)

                    try:
                        current_dist = nx.shortest_path_length(
                            self.graph, source=p_pos, target=e_pos
                        )
                        current_step_distances[pair] = (
                            current_dist  # Store current distance
                        )

                        # Get distance from the end of the *previous* step
                        last_dist = self.last_pursuer_evader_distances.get(pair)

                        # Calculate reward only if last distance exists (not first step, both agents were present)
                        if (
                            last_dist is not None
                            and last_dist != float("inf")
                            and current_dist != float("inf")
                        ):
                            delta_distance = (
                                last_dist - current_dist
                            )  # Positive if closer

                            # Apply reward if distance decreased (delta > 0)
                            if delta_distance > 0:
                                if (
                                    self.delta_reward_pursuer_scale is not None
                                    and p_agent in self.rewards
                                ):
                                    self.rewards[p_agent] += (
                                        self.delta_reward_pursuer_scale * delta_distance
                                    )
                                if (
                                    self.delta_penalty_evader_scale is not None
                                    and e_agent in self.rewards
                                ):
                                    # Ensure scale is negative for penalty
                                    self.rewards[e_agent] += (
                                        self.delta_penalty_evader_scale * delta_distance
                                    )

                            # Optional: Apply reward/penalty if distance increased (delta < 0)
                            # elif delta_distance < 0:
                            #    # Add logic here if you want to reward evader for increasing distance
                            #    pass

                    except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
                        # Store inf distance and handle exception (no reward calculation based on this)
                        current_step_distances[pair] = float("inf")
                        # print(f"Warning: No path or node not found between {p_agent} ({p_pos}) and {e_agent} ({e_pos}) during step.")
                        pass  # Silently handle for now
        # --- End Delta Distance Rewards ---

        self._check_captures()
        self._check_safe_arrivals()
        self._check_termination()

        self.timestep += 1

        if self.timestep >= self.max_steps:
            for agent in self.agents:
                if not self.terminations[agent]:
                    self.truncations[agent] = True

        # --- Update the last distances cache *after* all checks and *before* removing agents
        self.last_pursuer_evader_distances = current_step_distances

        # In this version of Supersuit, agents should not be removed from self.agents.
        # Instead, their termination/truncation status is handled, and observations/rewards
        # are provided for all possible agents.
        # The self.agents list should remain constant (self.possible_agents) throughout the episode.
        # The reset function already sets self.agents = self.possible_agents.copy()

        # Generate observations for all possible agents
        observations = {agent: self._get_observation(agent) for agent in self.possible_agents}

        # Ensure rewards, terminations, truncations, infos dictionaries contain entries
        # for all possible agents, as expected by PettingZoo Parallel API when black_death is not used.
        final_rewards = {
            agent: self.rewards.get(agent, 0) for agent in self.possible_agents
        }
        final_terminations = {
            agent: self.terminations.get(agent, False) for agent in self.possible_agents
        }
        final_truncations = {
            agent: self.truncations.get(agent, False) for agent in self.possible_agents
        }
        final_infos = {
            agent: self.infos.get(agent, {}) for agent in self.possible_agents
        }

        return (
            observations,
            final_rewards,  # Return dicts relevant to input agents
            final_terminations,
            final_truncations,
            final_infos,
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
        escape_occurred_this_step = False  # Track if any escape happened
        for evader in self.evaders:
            if evader in self.captured_evaders:
                continue

            if self.agent_positions[evader] == self.safe_node:
                if evader in self.rewards:
                    self.rewards[evader] += self.escape_reward_evader
                    self.terminations[evader] = True
                    # Keep the specific flag if needed for other purposes
                    # self.infos[evader]["escape"] = True
                    escape_occurred_this_step = True  # Mark that an escape happened

                    for pursuer in self.pursuers:
                        if pursuer in self.rewards:
                            self.rewards[pursuer] += self.escape_reward_pursuer

        # --- Start Edit: Add a general flag accessible by VecEnv ---
        # If an escape happened, add a flag that's easier for the VecEnv wrapper to see.
        # We add it to the info dict of the *first* active agent, as VecEnv wrappers
        # often pick one agent's info or merge them.
        if escape_occurred_this_step and self.agents:  # Ensure there are active agents
            first_active_agent = self.agents[0]
            if (
                first_active_agent in self.infos
            ):  # Make sure the agent is in the current infos dict
                self.infos[first_active_agent]["escape_event"] = True
            else:
                # If the first agent somehow isn't in infos (shouldn't happen), create it
                self.infos[first_active_agent] = {"escape_event": True}

            # Optional: Add to all agents' infos if you prefer consistency,
            # but adding to one is usually enough for VecEnv detection.
            # for agent in self.agents:
            #     if agent in self.infos:
            #         self.infos[agent]["escape_event"] = True
            #     else:
            #         self.infos[agent] = {"escape_event": True}
        # --- End Edit ---

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
