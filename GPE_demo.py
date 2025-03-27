import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from GPE.env.graph_pe import GPE
from shortest_path_strategies import ShortestPathStrategy


def test_environment():
    """
    Test the Graph Pursuit Evasion environment with random actions.
    """
    # Create the environment with a smaller graph for visualization
    env = GPE(
        num_nodes=200,
        num_edges=400,
        num_pursuers=2,
        num_evaders=1,
        capture_distance=1,
        required_captors=1,
        max_steps=50,
        seed=42,
        render_mode="rgb_array",
    )

    # Reset the environment
    observations, infos = env.reset(seed=42)

    # Visualize the graph
    visualize_graph(env)

    # Run a few episodes with random actions
    episode_rewards = {agent: 0 for agent in env.agents}

    for step in range(env.max_steps):
        # Take random actions
        actions = {
            agent: choose_action(env, agent, observations[agent])
            for agent in env.agents
        }

        # Step the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Update episode rewards
        for agent in env.agents:
            episode_rewards[agent] += rewards[agent]

        # Print step information
        print(f"Step {step}:")
        print(f"Actions: {actions}")
        print(f"Rewards: {rewards}")

        # Visualize the updated state
        visualize_graph(env)

        # Check if episode is done
        if all(terminations.values()) or all(truncations.values()):
            break

    print("\nEpisode Complete!")
    print(f"Total Steps: {step + 1}")
    print(f"Episode Rewards: {episode_rewards}")

    # Display final state
    print("\nFinal State:")
    print(f"Safe Node: {env.safe_node}")
    print(f"Pursuer Positions: {[env.agent_positions[p] for p in env.pursuers]}")
    print(
        f"Evader Positions: {[env.agent_positions[e] if e not in env.captured_evaders else 'CAPTURED' for e in env.evaders]}"
    )
    print(f"Captured Evaders: {env.captured_evaders}")


def choose_action(env, agent, observation):
    """
    Choose a random valid action for the given agent.
    """
    current_position = observation["position"]

    # Get valid neighboring nodes
    valid_neighbors = list(env.graph.neighbors(current_position))

    # Add current position as a valid choice (stay in place)
    # valid_actions = valid_neighbors + [current_position]

    action = ShortestPathStrategy(env, current_position, valid_neighbors)
    # Choose randomly
    # return np.random.choice(valid_actions)
    return action


def visualize_graph(env):
    """
    Visualize the current state of the environment.
    """
    plt.figure(figsize=(10, 8))

    # Create a spring layout for the graph
    pos = nx.spring_layout(env.graph, seed=42)

    # Draw the graph
    nx.draw(
        env.graph,
        pos,
        with_labels=True,
        node_color="lightgray",
        node_size=500,
        font_size=10,
        width=1,
        edge_color="gray",
    )

    # Draw the safe node
    nx.draw_networkx_nodes(
        env.graph, pos, nodelist=[env.safe_node], node_color="green", node_size=700
    )

    # Draw pursuers
    pursuer_positions = [env.agent_positions[p] for p in env.pursuers]
    nx.draw_networkx_nodes(
        env.graph, pos, nodelist=pursuer_positions, node_color="red", node_size=600
    )

    # Draw evaders
    active_evaders = [e for e in env.evaders if e not in env.captured_evaders]
    evader_positions = [env.agent_positions[e] for e in active_evaders]
    nx.draw_networkx_nodes(
        env.graph, pos, nodelist=evader_positions, node_color="blue", node_size=600
    )

    # Add labels for special nodes
    labels = {}
    labels[env.safe_node] = "SAFE"
    for p in env.pursuers:
        labels[env.agent_positions[p]] = f"P{p.split('_')[1]}"
    for e in active_evaders:
        labels[env.agent_positions[e]] = f"E{e.split('_')[1]}"

    nx.draw_networkx_labels(
        env.graph, pos, labels=labels, font_color="white", font_weight="bold"
    )

    plt.title(f"Graph Pursuit Evasion - Step {env.timestep}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    """Problems:
    1. Number of Nodes and Edges, 10,100,1000,10000?
    2. Number of Pursuers and Evaders, and how to initialize their positions
    3. Visualization of the graph
    4. Action policy of the agents. Shortest path ? A* ? RRT ? RL ?
    5. Pursuer cannot observe the position of safe node
    6. Evader can observe the position of safe node

    """

    test_environment()
