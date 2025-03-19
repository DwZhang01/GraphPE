import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import random
from copy import copy
import networkx as nx
from pettingzoo import ParallelEnv


class GPE(ParallelEnv):
    """
    Graph Pursuit Evasion (GPE) Environment

    A multi-agent environment where pursuers try to catch evaders, while evaders
    try to reach a safe node on a graph structure.
    """

    metadata = {
        "name": "graph_pe_v0",
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        num_nodes=100,
        num_edges=200,
        num_pursuers=3,
        num_evaders=2,
        capture_distance=1,
        required_captors=1,
        max_steps=200,
        seed=None,
        graph=None,
        render_mode=None,
    ):
        """
        Initialize the Graph Pursuit Evasion environment.

        Args:
            num_nodes: Number of nodes in the graph
            num_edges: Approximate number of edges in the graph
            num_pursuers: Number of pursuer agents
            num_evaders: Number of evader agents
            capture_distance: Distance required for capture (1 means adjacent nodes)
            required_captors: Number of pursuers required to be adjacent for a capture
            max_steps: Maximum number of steps before the episode ends
            seed: Random seed
            graph: Predefined graph (networkx graph object), if None a random graph is generated
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        # Set random seed if provided
        self.np_random = np.random.RandomState(seed)

        # Environment parameters
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_pursuers = num_pursuers
        self.num_evaders = num_evaders
        self.capture_distance = capture_distance
        self.required_captors = required_captors
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Create agents
        self.pursuers = [f"pursuer_{i}" for i in range(self.num_pursuers)]
        self.evaders = [f"evader_{i}" for i in range(self.num_evaders)]
        self.possible_agents = self.pursuers + self.evaders

        # Graph structure
        self.custom_graph = graph

        # Initialize spaces
        self._initialize_spaces()

        # State variables that will be initialized in reset()
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

    def _initialize_spaces(self):
        """Initialize action and observation spaces for all agents."""
        # Action space: Move to an adjacent node
        # Each agent can move to any connected node or stay at the current node
        self.action_spaces = {
            agent: Discrete(self.num_nodes) for agent in self.possible_agents
        }

        # Observation space:
        # - Current position (node index)
        # - Safe node position (for evaders)
        # - Positions of all other agents
        # - Graph adjacency information (local neighborhood)

        # We'll use a Dict space for more structured observations
        observation_spaces = {}
        for agent in self.possible_agents:
            if agent.startswith("pursuer"):
                observation_spaces[agent] = Dict(
                    {
                        "position": Discrete(self.num_nodes),
                        "pursuers": Box(
                            low=0,
                            high=self.num_nodes - 1,
                            shape=(self.num_pursuers,),
                            dtype=np.int32,
                        ),
                        "evaders": Box(
                            low=0,
                            high=self.num_nodes - 1,
                            shape=(self.num_evaders,),
                            dtype=np.int32,
                        ),
                        "adjacency": Box(
                            low=0, high=1, shape=(self.num_nodes,), dtype=np.int32
                        ),
                    }
                )
            else:  # evader
                observation_spaces[agent] = Dict(
                    {
                        "position": Discrete(self.num_nodes),
                        "safe_node": Discrete(self.num_nodes),
                        "pursuers": Box(
                            low=0,
                            high=self.num_nodes - 1,
                            shape=(self.num_pursuers,),
                            dtype=np.int32,
                        ),
                        "evaders": Box(
                            low=0,
                            high=self.num_nodes - 1,
                            shape=(self.num_evaders,),
                            dtype=np.int32,
                        ),
                        "adjacency": Box(
                            low=0, high=1, shape=(self.num_nodes,), dtype=np.int32
                        ),
                    }
                )

        self.observation_spaces = observation_spaces

    def _generate_graph(self):
        """Generate a random graph or use the provided one."""
        if self.custom_graph is not None:
            return copy(self.custom_graph)

        # Generate a random graph with approximately num_edges edges
        # Using Erdős-Rényi random graph model
        p = 2 * self.num_edges / (self.num_nodes * (self.num_nodes - 1))
        graph = nx.gnp_random_graph(
            self.num_nodes, p, seed=self.np_random.randint(10000)
        )

        # Ensure the graph is connected
        if not nx.is_connected(graph):
            # Get the largest connected component
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc).copy()

            # Add some random edges to ensure connectivity
            nodes = list(graph.nodes())
            components = list(nx.connected_components(graph))

            while len(components) > 1:
                # Connect two random nodes from different components
                comp1 = random.choice(components)
                components.remove(comp1)
                comp2 = random.choice(components)
                components.remove(comp2)

                node1 = random.choice(list(comp1))
                node2 = random.choice(list(comp2))
                graph.add_edge(node1, node2)

                # Recalculate components
                components = list(nx.connected_components(graph))

        return graph

    def reset(self, seed=None, options=None):
        """Reset the environment to start a new episode."""
        # Reset random seed if provided
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        # Generate or use the provided graph
        self.graph = self._generate_graph()

        # Reset episode variables
        self.timestep = 0
        self.agents = self.possible_agents.copy()
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.captured_evaders = set()

        # Select a random safe node
        self.safe_node = self.np_random.choice(list(self.graph.nodes()))

        # Place agents randomly on the graph
        # Ensure pursuers and evaders start at different nodes
        all_nodes = list(self.graph.nodes())
        self.agent_positions = {}

        # Place pursuer agents
        pursuer_nodes = self.np_random.choice(
            all_nodes, size=self.num_pursuers, replace=False
        )
        for i, agent in enumerate(self.pursuers):
            self.agent_positions[agent] = int(pursuer_nodes[i])

        # Place evader agents on different nodes than pursuers
        remaining_nodes = list(set(all_nodes) - set(pursuer_nodes) - {self.safe_node})
        if len(remaining_nodes) < self.num_evaders:
            # Fall back to allowing overlap if there aren't enough nodes
            remaining_nodes = list(set(all_nodes) - {self.safe_node})

        evader_nodes = self.np_random.choice(
            remaining_nodes,
            size=min(self.num_evaders, len(remaining_nodes)),
            replace=False,
        )
        for i, agent in enumerate(self.evaders):
            if i < len(evader_nodes):
                self.agent_positions[agent] = int(evader_nodes[i])
            else:
                # In the unlikely case we run out of nodes, place randomly
                self.agent_positions[agent] = int(
                    self.np_random.choice(list(set(all_nodes) - {self.safe_node}))
                )

        # Generate observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        return observations, self.infos

    def _get_observation(self, agent):
        """Generate observation for an agent."""
        position = self.agent_positions[agent]

        # Get adjacency information (which nodes are connected to the current node)
        adjacency = np.zeros(self.num_nodes, dtype=np.int32)
        neighbors = list(self.graph.neighbors(position))
        for neighbor in neighbors:
            adjacency[neighbor] = 1

        # Get positions of all pursuers and evaders
        pursuer_positions = np.array(
            [self.agent_positions[p] for p in self.pursuers], dtype=np.int32
        )
        evader_positions = np.array(
            [
                self.agent_positions[e] if e not in self.captured_evaders else -1
                for e in self.evaders
            ],
            dtype=np.int32,
        )

        # Create observation dictionary
        observation = {
            "position": position,
            "pursuers": pursuer_positions,
            "evaders": evader_positions,
            "adjacency": adjacency,
        }

        # Add safe node information for evaders
        if agent.startswith("evader"):
            observation["safe_node"] = self.safe_node

        return observation

    def step(self, actions):
        """Execute actions for all agents and return new observations."""
        # Validate that all active agents provided actions
        if not actions.keys() == self.agents:
            raise ValueError(
                f"Actions must be provided for all active agents. Got {actions.keys()}, expected {self.agents}"
            )

        # Initialize rewards, terminations, truncations, and infos
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # Process actions for each agent
        for agent, action in actions.items():
            current_position = self.agent_positions[agent]

            # Check if the action leads to a valid neighbor
            if action == current_position:
                # Stay at current position
                pass
            elif action in list(self.graph.neighbors(current_position)):
                # Move to neighbor
                self.agent_positions[agent] = action
            else:
                # Invalid action, don't move
                self.infos[agent]["invalid_action"] = True

        # Check for captures and safe arrivals
        self._check_captures()
        self._check_safe_arrivals()

        # Check if game is over
        game_over = self._check_termination()

        # Increment timestep
        self.timestep += 1

        # Check for episode truncation (max steps reached)
        if self.timestep >= self.max_steps:
            for agent in self.agents:
                self.truncations[agent] = True

        # Generate new observations
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        return (
            observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def _check_captures(self):
        """Check if any evaders have been captured."""
        for evader in self.evaders:
            if evader in self.captured_evaders:
                continue

            evader_pos = self.agent_positions[evader]
            evader_neighbors = list(self.graph.neighbors(evader_pos))

            # Count how many pursuers are adjacent to this evader
            adjacent_pursuers = 0
            for pursuer in self.pursuers:
                pursuer_pos = self.agent_positions[pursuer]

                # Check if pursuer is adjacent to evader (or at same position)
                if pursuer_pos == evader_pos or pursuer_pos in evader_neighbors:
                    adjacent_pursuers += 1

            # Check if capture conditions are met
            if adjacent_pursuers >= self.required_captors:
                # Evader is captured
                self.captured_evaders.add(evader)

                # Award positive reward to pursuers
                for pursuer in self.pursuers:
                    self.rewards[pursuer] += 10.0

                # Penalty for the captured evader
                self.rewards[evader] -= 10.0
                self.terminations[evader] = True

    def _check_safe_arrivals(self):
        """Check if any evaders have reached the safe node."""
        for evader in self.evaders:
            if evader in self.captured_evaders:
                continue

            # Check if evader reached safe node
            if self.agent_positions[evader] == self.safe_node:
                # Award positive reward to evader
                self.rewards[evader] += 20.0
                self.terminations[evader] = True

                # Penalty for pursuers
                for pursuer in self.pursuers:
                    self.rewards[pursuer] -= 5.0

    def _check_termination(self):
        """Check if the game is over."""
        # Game is over if all evaders are either captured or reached safe node
        active_evaders = set(self.evaders) - self.captured_evaders
        evaders_at_safe_node = sum(
            1 for e in active_evaders if self.agent_positions[e] == self.safe_node
        )

        if len(active_evaders) == 0 or evaders_at_safe_node == len(active_evaders):
            # All evaders have been captured or reached safety
            for agent in self.agents:
                self.terminations[agent] = True
            return True

        return False

    def observation_space(self, agent):
        """Return the observation space for an agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Return the action space for an agent."""
        return self.action_spaces[agent]

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return

        # This is a placeholder for rendering
        # For actual implementation, you could use matplotlib or networkx drawing
        print(f"Timestep: {self.timestep}")
        print(f"Safe node: {self.safe_node}")
        print("Pursuer positions:", {p: self.agent_positions[p] for p in self.pursuers})
        print(
            "Evader positions:",
            {
                e: self.agent_positions[e]
                for e in self.evaders
                if e not in self.captured_evaders
            },
        )
        print("Captured evaders:", self.captured_evaders)

        if self.render_mode == "human":
            # Here you could implement visualization with matplotlib or networkx
            # For example:
            # import matplotlib.pyplot as plt
            # pos = nx.spring_layout(self.graph)
            # nx.draw(self.graph, pos, with_labels=True)
            # for agent, position in self.agent_positions.items():
            #     if agent.startswith("pursuer"):
            #         plt.plot(pos[position][0], pos[position][1], 'ro')
            #     elif agent.startswith("evader") and agent not in self.captured_evaders:
            #         plt.plot(pos[position][0], pos[position][1], 'bo')
            # plt.plot(pos[self.safe_node][0], pos[self.safe_node][1], 'go')
            # plt.pause(0.1)
            # plt.clf()
            pass
