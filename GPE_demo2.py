import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from IPython.display import HTML
from GPE.env.graph_pe import GPE


class AnimatedGPEDemo:
    """
    Animated demonstration of the Graph Pursuit Evasion environment.
    """

    def __init__(
        self,
        strategies=None,
        num_nodes=20,
        num_edges=40,
        num_pursuers=2,
        num_evaders=1,
        max_steps=50,
        seed=42,
        interval=500,
    ):
        """
        Initialize the demo.

        Args:
            strategies: Dictionary mapping agent types to strategy classes
            num_nodes: Number of nodes in the graph
            num_edges: Number of edges in the graph
            num_pursuers: Number of pursuer agents
            num_evaders: Number of evader agents
            max_steps: Maximum number of steps to run
            seed: Random seed
            interval: Animation interval in milliseconds
        """
        self.env = GPE(
            num_nodes=num_nodes,
            num_edges=num_edges,
            num_pursuers=num_pursuers,
            num_evaders=num_evaders,
            capture_distance=1,
            required_captors=1,
            max_steps=max_steps,
            seed=seed,
        )

        self.strategies = strategies or {}
        self.max_steps = max_steps
        self.interval = interval
        self.seed = seed

        # Initialize strategies if not provided
        if "pursuer" not in self.strategies:
            from shortest_path_strategies import PursuerStrategy

            self.strategies["pursuer"] = PursuerStrategy(self.env)

        if "evader" not in self.strategies:
            from shortest_path_strategies import EvaderStrategy

            self.strategies["evader"] = EvaderStrategy(self.env)

        # Initialize visualization
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.pos = None  # Will be set in reset()

        # Animation state
        self.episode_done = False
        self.current_step = 0
        self.observations = None
        self.episode_rewards = None
        self.history = []

    def reset(self):
        """Reset the environment and visualization."""
        self.observations, _ = self.env.reset(seed=self.seed)
        self.episode_rewards = {agent: 0 for agent in self.env.agents}
        self.episode_done = False
        self.current_step = 0
        self.history = []

        # Create a fixed layout for the graph
        self.pos = nx.spring_layout(self.env.graph, seed=self.seed)

        # Save initial state
        self._save_state()

        return self.observations

    def step(self):
        """
        Take a step in the environment using the defined strategies.
        """
        if self.episode_done:
            return None, None, None, None, None

        # Choose actions based on strategies
        actions = {}
        for agent in self.env.agents:
            if agent.startswith("pursuer"):
                actions[agent] = self.strategies["pursuer"].choose_action(
                    agent, self.observations[agent]
                )
            elif agent.startswith("evader"):
                actions[agent] = self.strategies["evader"].choose_action(
                    agent, self.observations[agent]
                )

        # Take a step in the environment
        observations, rewards, terminations, truncations, infos = self.env.step(actions)

        # Update state
        self.observations = observations
        for agent in self.env.agents:
            self.episode_rewards[agent] += rewards[agent]

        self.current_step += 1

        # Check if episode is done
        if (
            all(terminations.values())
            or all(truncations.values())
            or self.current_step >= self.max_steps
        ):
            self.episode_done = True

        # Save current state for animation
        self._save_state()

        return observations, rewards, terminations, truncations, infos

    def _save_state(self):
        """Save the current state for animation."""
        state = {
            "step": self.current_step,
            "agent_positions": self.env.agent_positions.copy(),
            "captured_evaders": self.env.captured_evaders.copy(),
            "safe_node": self.env.safe_node,
            "rewards": {
                agent: self.episode_rewards[agent] for agent in self.env.agents
            },
            "done": self.episode_done,
        }
        self.history.append(state)

    def run_episode(self):
        """Run a complete episode and return the animation."""
        self.reset()

        while not self.episode_done:
            self.step()

        return self.create_animation()

    def create_animation(self):
        """Create an animation of the episode."""
        fig, ax = plt.subplots(figsize=(10, 8))

        def update(frame):
            ax.clear()
            state = self.history[frame]
            self._draw_state(ax, state)
            return (ax,)

        ani = FuncAnimation(
            fig,
            update,
            frames=len(self.history),
            interval=self.interval,
            blit=False,
            repeat=True,
        )

        plt.close()  # Prevent displaying the figure directly
        return ani

    def _draw_state(self, ax, state):
        """Draw the current state on the given axes."""
        step = state["step"]
        agent_positions = state["agent_positions"]
        captured_evaders = state["captured_evaders"]
        safe_node = state["safe_node"]
        rewards = state["rewards"]
        done = state["done"]

        # Draw the graph
        nx.draw(
            self.env.graph,
            self.pos,
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
            self.env.graph,
            self.pos,
            ax=ax,
            nodelist=[safe_node],
            node_color="green",
            node_size=700,
        )

        # Draw pursuers
        pursuer_positions = [agent_positions[p] for p in self.env.pursuers]
        nx.draw_networkx_nodes(
            self.env.graph,
            self.pos,
            ax=ax,
            nodelist=pursuer_positions,
            node_color="red",
            node_size=600,
        )

        # Draw active evaders
        active_evaders = [e for e in self.env.evaders if e not in captured_evaders]
        evader_positions = [agent_positions[e] for e in active_evaders]
        nx.draw_networkx_nodes(
            self.env.graph,
            self.pos,
            ax=ax,
            nodelist=evader_positions,
            node_color="blue",
            node_size=600,
        )

        # Add labels for special nodes
        labels = {}
        labels[safe_node] = "SAFE"
        for p in self.env.pursuers:
            labels[agent_positions[p]] = f"P{p.split('_')[1]}"
        for e in active_evaders:
            labels[agent_positions[e]] = f"E{e.split('_')[1]}"

        nx.draw_networkx_labels(
            self.env.graph,
            self.pos,
            ax=ax,
            labels=labels,
            font_color="white",
            font_weight="bold",
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


def run_demo():
    """Run the animated demo."""
    # Import strategies
    from shortest_path_strategies import PursuerStrategy, EvaderStrategy

    # Create the environment
    env = GPE(
        num_nodes=20,
        num_edges=40,
        num_pursuers=2,
        num_evaders=1,
        capture_distance=1,
        required_captors=1,
        max_steps=50,
        seed=42,
    )

    # Create strategies
    pursuer_strategy = PursuerStrategy(env)
    evader_strategy = EvaderStrategy(env)

    # Create the demo
    demo = AnimatedGPEDemo(
        strategies={"pursuer": pursuer_strategy, "evader": evader_strategy},
        num_nodes=20,
        num_edges=40,
        num_pursuers=2,
        num_evaders=1,
        max_steps=50,
        seed=42,
        interval=500,  # 500ms per frame
    )

    # Run the demo
    animation = demo.run_episode()

    # Display the animation
    plt.rcParams["animation.embed_limit"] = 100
    plt.rcParams["animation.html"] = "jshtml"

    # Save the animation as a GIF
    animation.save("pursuit_evasion.gif", writer="pillow", fps=2)

    # Display the animation (this works in Jupyter notebooks)
    return animation


if __name__ == "__main__":
    animation = run_demo()
    plt.show()
