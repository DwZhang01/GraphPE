import numpy as np
import networkx as nx


class ShortestPathStrategy:

    def __init__(self, env):
        self.env = env
        self.graph = env.graph

    def choose_action(self, agent, observation):

        raise NotImplementedError("Subclasses must implement this method")

    def get_shortest_path(self, start, target):

        try:
            path = nx.shortest_path(self.graph, start, target)
            return path
        except nx.NetworkXNoPath:
            return None

    def get_next_node_in_path(self, start, target):

        path = self.get_shortest_path(start, target)

        if path is None or len(path) < 2:
            return start  # Stay in place if no path exists or already at target

        return path[1]  # Return the next node in the path


class PursuerStrategy(ShortestPathStrategy):

    def choose_action(self, agent, observation):
        current_position = observation["position"]
        evader_positions = observation["evaders"]

        # Filter out captured evaders (represented by -1)
        active_evader_positions = [pos for pos in evader_positions if pos >= 0]

        if not active_evader_positions:
            return current_position  # No active evaders, stay in place

        # Find the nearest evader
        nearest_evader = None
        min_distance = float("inf")

        for evader_pos in active_evader_positions:
            try:
                distance = nx.shortest_path_length(
                    self.graph, current_position, evader_pos
                )
                if distance < min_distance:
                    min_distance = distance
                    nearest_evader = evader_pos
            except nx.NetworkXNoPath:
                continue

        if nearest_evader is None:
            return current_position  # No reachable evaders, stay in place

        # Move towards the nearest evader
        return self.get_next_node_in_path(current_position, nearest_evader)


class EvaderStrategy(ShortestPathStrategy):

    def choose_action(self, agent, observation):
        current_position = observation["position"]
        safe_node = observation["safe_node"]
        pursuer_positions = observation["pursuers"]

        if current_position == safe_node:
            return current_position

        path_to_safe = self.get_shortest_path(current_position, safe_node)

        if path_to_safe is None or len(path_to_safe) < 2:
            return current_position  # No path to safe node, stay in place

        next_node = path_to_safe[1]

        # check persuer
        for pursuer_pos in pursuer_positions:
            if next_node == pursuer_pos or pursuer_pos in self.graph.neighbors(
                next_node
            ):
                alternative_neighbors = [
                    n
                    for n in self.graph.neighbors(current_position)
                    if n != next_node and n not in pursuer_positions
                ]

                if alternative_neighbors:
                    # Choose the neighbor that's closest to the safe node
                    best_alternative = None
                    min_distance = float("inf")

                    for neighbor in alternative_neighbors:
                        try:
                            distance = nx.shortest_path_length(
                                self.graph, neighbor, safe_node
                            )
                            if distance < min_distance:
                                min_distance = distance
                                best_alternative = neighbor
                        except nx.NetworkXNoPath:
                            continue

                    if best_alternative is not None:
                        return best_alternative

        # If no better alternative, follow the shortest path
        return next_node
