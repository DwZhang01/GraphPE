import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque


class RLStrategy:
    """
    Base class for reinforcement learning strategies.
    """

    def __init__(self, env, agent_type):
        self.env = env
        self.agent_type = agent_type  # "pursuer" or "evader"

    def choose_action(self, agent, observation):
        """
        Choose an action based on the current observation.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def train(self, episodes):
        """
        Train the strategy.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def save(self, path):
        """
        Save the strategy.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def load(self, path):
        """
        Load the strategy.
        To be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement this method")


class DQNStrategy(RLStrategy):
    """
    Deep Q-Network strategy for graph pursuit evasion.
    """

    def __init__(
        self,
        env,
        agent_type,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        memory_size=10000,
        batch_size=64,
    ):
        super().__init__(env, agent_type)

        # DQN parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory_size = memory_size
        self.batch_size = batch_size

        # Initialize memory buffer
        self.memory = deque(maxlen=memory_size)

        # Initialize neural network
        self._build_model()

    def _build_model(self):
        """Build the neural network model."""
        # We'll implement a simple neural network for demonstration
        # In a real implementation, you'd want to customize this based on your observation space

        # Simple example:
        # - Input: flattened observation
        # - Hidden layers: 128, 64 neurons
        # - Output: Q-values for each possible action

        # This is a placeholder - you'll need to implement the actual neural network
        # based on your specific observation space and action space
        self.model = nn.Sequential(
            nn.Linear(100, 128),  # Adjust input size based on your observation space
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.env.num_nodes),  # Output Q-values for each possible node
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def _preprocess_observation(self, observation):
        """Convert the observation to a format suitable for the neural network."""
        # This is a placeholder - you'll need to implement the actual preprocessing
        # based on your specific observation space

        # Example: Convert the observation dictionary to a flat vector
        # This is just a demonstration and won't work directly with the actual observation space
        return torch.zeros(100)  # Placeholder

    def choose_action(self, agent, observation):
        """Choose an action using epsilon-greedy policy."""
        current_position = observation["position"]

        # With probability epsilon, choose a random action
        if random.random() < self.epsilon:
            # Choose a random neighbor
            neighbors = list(self.env.graph.neighbors(current_position))
            return random.choice(neighbors + [current_position])

        # Otherwise, choose the best action according to the model
        state = self._preprocess_observation(observation)
        with torch.no_grad():
            q_values = self.model(state)

        # Filter q_values to only include valid actions (neighbors + current position)
        valid_actions = list(self.env.graph.neighbors(current_position)) + [
            current_position
        ]
        valid_q_values = {action: q_values[action].item() for action in valid_actions}

        # Choose the action with the highest Q-value
        return max(valid_q_values, key=valid_q_values.get)

    def train(self, episodes):
        """Train the DQN model."""
        # This is a placeholder - you'll need to implement the actual training loop
        # This would typically involve:
        # 1. Running episodes
        # 2. Collecting experiences (state, action, reward, next_state, done)
        # 3. Storing experiences in the memory buffer
        # 4. Sampling from the memory buffer
        # 5. Updating the model using the sampled experiences

        print(f"Training DQN for {episodes} episodes...")

        # Example training loop:
        for episode in range(episodes):
            # Reset the environment
            observations, _ = self.env.reset()

            episode_reward = 0
            done = False

            while not done:
                # Choose actions for all agents
                actions = {}
                for agent in self.env.agents:
                    if agent.startswith(self.agent_type):
                        actions[agent] = self.choose_action(agent, observations[agent])
                    else:
                        # For other agents, use a baseline strategy
                        # This could be a random strategy or a fixed strategy
                        neighbors = list(
                            self.env.graph.neighbors(observations[agent]["position"])
                        )
                        actions[agent] = random.choice(
                            neighbors + [observations[agent]["position"]]
                        )

                # Take a step in the environment
                next_observations, rewards, terminations, truncations, _ = (
                    self.env.step(actions)
                )

                # For each agent of our type, store the experience
                for agent in self.env.agents:
                    if agent.startswith(self.agent_type):
                        state = self._preprocess_observation(observations[agent])
                        action = actions[agent]
                        reward = rewards[agent]
                        next_state = self._preprocess_observation(
                            next_observations[agent]
                        )
                        done = terminations[agent] or truncations[agent]

                        # Store the experience in the memory buffer
                        self.memory.append((state, action, reward, next_state, done))

                # Update observations
                observations = next_observations

                # Check if episode is done
                if all(terminations.values()) or all(truncations.values()):
                    done = True

            # After each episode, update the model using experiences from the memory buffer
            self._update_model()

            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{episodes}, Epsilon: {self.epsilon:.4f}")

    def _update_model(self):
        """Update the model using experiences from the memory buffer."""
        # This is a placeholder - you'll need to implement the actual model update
        # This would typically involve:
        # 1. Sampling a batch of experiences from the memory buffer
        # 2. Computing the
