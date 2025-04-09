import torch
import torch.nn as nn
from gymnasium.spaces import Dict
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import (
    SAGEConv,
    GATv2Conv,
    global_mean_pool,
)  # Or GATv2Conv etc.
from gymnasium import spaces

# (GNNEnvWrapper class defined above)


class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using GraphSAGE for the GPE environment.
    Assumes observation space is the Dict space defined in GNNEnvWrapper.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 128):
        super().__init__(observation_space, features_dim=features_dim)

        # Extract dimensions from the observation space
        node_feature_dim = observation_space["node_features"].shape[1]
        hidden_dim = 128

        # Define GNN layers (Example: 2 layers of GraphSAGE) Multihead attention
        self.conv1 = GATv2Conv(node_feature_dim, hidden_dim // 4, heads=4)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 2, heads=2)
        self.conv3 = GATv2Conv(hidden_dim, features_dim, heads=1)

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        print(f"GNN Feature Extractor Initialized:")
        print(f"  Input node feature dim: {node_feature_dim}")
        print(f"  Output features dim (shared network): {features_dim}")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Observations come in batches from SB3 VecEnv
        # Need to handle the dictionary structure and potential batching
        # Assume tensors are already on the correct device

        node_features = observations[
            "node_features"
        ]  # Shape: [batch_size, num_nodes, node_feature_dim]
        edge_index = observations["edge_index"]  # Shape: [batch_size, 2, max_edges]
        agent_node_idx = observations["agent_node_index"]  # Shape: [batch_size, 1]

        batch_size = node_features.shape[0]
        num_nodes = node_features.shape[1]
        device = node_features.device

        x = node_features.view(batch_size * num_nodes, -1)
        edge_index_flat = edge_index.view(2, -1)

        valid_edge_mask = (edge_index_flat[0] < num_nodes) & (
            edge_index_flat[1] < num_nodes
        )
        edge_index_flat = edge_index_flat[:, valid_edge_mask]

        # Process batch by iterating (less efficient) or using PyG batching (more complex setup)
        # Simple iteration approach:

        # --- GNN forward pass ---
        x = self.conv1(x, edge_index_flat)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index_flat)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, edge_index_flat)

        x = x.view(
            batch_size, num_nodes, -1
        )  # Reshape back to [batch_size, num_nodes, features_dim]
        # Extract the feature vector for the specific agent's node
        agent_node_idx = agent_node_idx.squeeze(1)  # Shape: [batch_size]
        batch_idx = torch.arrange(batch_size, device=device)
        agent_features = x[
            batch_idx, agent_node_idx
        ]  # Shape: [batch_size, features_dim]

        # Stack features back into a batch
        return agent_features


class GNNPolicy(ActorCriticPolicy):
    """
    Custom policy using the GNNFeatureExtractor.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch=None,  # Optional: Define sizes for actor/critic MLP heads
        activation_fn=nn.ReLU,
        features_dim=64,  # Output dim of GNN extractor
        *args,
        **kwargs,
    ):
        # Disable MLP extractor and use our GNN extractor
        kwargs["features_extractor_class"] = GNNFeatureExtractor
        kwargs["features_extractor_kwargs"] = {"features_dim": features_dim}
        # Let ActorCriticPolicy handle shared network setup if needed
        kwargs["share_features_extractor"] = True  # Usually True

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            *args,
            **kwargs,
        )

    def _get_action_dist_from_latent(self, latent_pi, latent_sde=None):
        mean_actions = self.action_net(latent_pi)  # [batch_size, action_dim]
        if hasattr(self, "_last_obs") and "action_mask" in self._last_obs:
            action_masks = self._last_obs["action_mask"]  # [batch_size, num_nodes]
            masked_logits = torch.where(
                action_masks > 0,
                mean_actions,
                torch.tensor(-1e9, device=mean_actions.device),
            )
            return self.action_dist.proba_distribution(action_logits=masked_logits)
        return self.action_dist.proba_distribution(action_logits=mean_actions)

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        """
        Need to store the observation for action masking in _get_action_dist_from_latent.
        """
        # Store the observation before calling the parent's forward method
        self._last_obs = obs
        return super().forward(obs, deterministic)
