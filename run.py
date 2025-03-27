import os
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from GPE.env.graph_pe import GPE
from shortest_path_strategies import PursuerStrategy, EvaderStrategy


def parse_args():
    parser = argparse.ArgumentParser(description="Run Graph Pursuit Evasion simulation")
    parser.add_argument(
        "--num_nodes", type=int, default=20, help="Number of nodes in the graph"
    )
    parser.add_argument(
        "--num_edges", type=int, default=40, help="Number of edges in the graph"
    )
    parser.add_argument(
        "--num_pursuers", type=int, default=2, help="Number of pursuer agents"
    )
    parser.add_argument(
        "--num_evaders", type=int, default=1, help="Number of evader agents"
    )
    parser.add_argument(
        "--max_steps", type=int, default=50, help="Maximum number of steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--interval", type=int, default=500, help="Animation interval in milliseconds"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pursuit_evasion.gif",
        help="Output file for animation",
    )
    parser.add_argument(
        "--fps", type=int, default=2, help="Frames per second for animation"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    env = GPE(
        num_nodes=args.num_nodes,
        num_edges=args.num_edges,
        num_pursuers=args.num_pursuers,
        num_evaders=args.num_evaders,
        capture_distance=1,
        required_captors=1,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    # Create strategies
    pursuer_strategy = PursuerStrategy(env)
    evader_strategy = EvaderStrategy(env)

    # Reset the environment
    observations, _ = env.reset(seed=args.seed)

    # Create a fixed layout for the graph
    pos = nx.spring_layout(env.graph, seed=args.seed)

    # Store the history of the episode
    history = []

    # Initialize episode variables
    episode_rewards = {agent: 0 for agent in env.agents}
    episode_done = False
    current_step = 0

    # Save initial state
    state = {
        "step": current_step,
        "agent_positions": env.agent_positions.copy(),
        "captured_evaders": env.captured_evaders.copy(),
        "safe_node": env.safe_node,
        "rewards": {agent: episode_rewards[agent] for agent in env.agents},
        "done": episode_done,
    }
    history.append(state)

    # Run the episode
    while not episode_done:
        # Choose actions based on strategies
        actions = {}
        for agent in env.agents:
            if agent.startswith("pursuer"):
                actions[agent] = pursuer_strategy.choose_action(
                    agent, observations[agent]
                )
            elif agent.startswith("evader"):
                actions[agent] = evader_strategy.choose_action(
                    agent, observations[agent]
                )

        # Take a step in the environment
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Update state
        for agent in env.agents:
            episode_rewards[agent] += rewards[agent]

        current_step += 1

        # Save current state
        state = {
            "step": current_step,
            "agent_positions": env.agent_positions.copy(),
            "captured_evaders": env.captured_evaders.copy(),
            "safe_node": env.safe_node,
            "rewards": {agent: episode_rewards[agent] for agent in env.agents},
            "done": episode_done,
        }
        history.append(state)

        # Print step information
        print(f"Step {current_step}:")
        print(f"Actions: {actions}")
        print(f"Rewards: {rewards}")
        print(f"Terminations: {terminations}")
        print(f"Truncations: {truncations}")
        print("-" * 50)

        # Check if episode is done
        if (
            all(terminations.values())
            or all(truncations.values())
            or current_step >= args.max_steps
        ):
            episode_done = True
            state["done"] = True

    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))

    def update(frame):
        ax.clear()
        state = history[frame]

        step = state["step"]
        agent_positions = state["agent_positions"]
        captured_evaders = state["captured_evaders"]
        safe_node = state["safe_node"]
        rewards = state["rewards"]
        done = state["done"]

        # Draw the graph
        nx.draw(
            env.graph,
            pos,
            ax=ax,
            with_labels=True,
            node_color="lightgray",
            node_size=500,
            font_size=10,
            width=1,
            edge_color="gray",
        )

        # Draw the safe node
        nx.draw_networkx_nodes(
            env.graph,
            pos,
            ax=ax,
            nodelist=[safe_node],
            node_color="green",
            node_size=700,
        )

        # Draw pursuers
        pursuer_positions = [agent_positions[p] for p in env.pursuers]
        nx.draw_networkx_nodes(
            env.graph,
            pos,
            ax=ax,
            nodelist=pursuer_positions,
            node_color="red",
            node_size=600,
        )

        # Draw active evaders
        active_evaders = [e for e in env.evaders if e not in captured_evaders]
        evader_positions = [agent_positions[e] for e in active_evaders]
        nx.draw_networkx_nodes(
            env.graph,
            pos,
            ax=ax,
            nodelist=evader_positions,
            node_color="blue",
            node_size=600,
        )

        # Add labels for special nodes
        labels = {}
        labels[safe_node] = "SAFE"
        for p in env.pursuers:
            labels[agent_positions[p]] = f"P{p.split('_')[1]}"
        for e in active_evaders:
            labels[agent_positions[e]] = f"E{e.split('_')[1]}"

        nx.draw_networkx_labels(
            env.graph, pos, ax=ax, labels=labels, font_color="white", font_weight="bold"
        )

        # Add a title with current step and rewards
        title = f"Step {step}"
        if done:
            title += " (Episode End)"
        ax.set_title(title)

        # Add reward information
        reward_text = "Rewards:\n"
        for agent, reward in rewards.items():
            reward_text += f"{agent}: {reward:.1f}\n"

        # Add captured evaders information
        if captured_evaders:
            reward_text += "\nCaptured Evaders:\n"
            for e in captured_evaders:
                reward_text += f"{e}\n"

        ax.text(
            0.02,
            0.02,
            reward_text,
            transform=ax.transAxes,
            bbox=dict(facecolor="white", alpha=0.7),
        )

        # Remove axis
        ax.set_axis_off()

        return (ax,)

    ani = FuncAnimation(
        fig,
        update,
        frames=len(history),
        interval=args.interval,
        blit=False,
        repeat=True,
    )

    # Save the animation
    ani.save(args.output, writer="pillow", fps=args.fps)
    print(f"Animation saved to {args.output}")

    # Display the animation
    plt.show()


if __name__ == "__main__":
    main()
