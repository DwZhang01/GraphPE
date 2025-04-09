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

        # Define GNN layers (Example: 2 layers of GraphSAGE)
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

        # Process batch by iterating (less efficient) or using PyG batching (more complex setup)
        # Simple iteration approach:
        output_features = []
        for i in range(batch_size):
            x = node_features[i]  # [num_nodes, node_feature_dim]
            ei = edge_index[i]  # [2, max_edges]
            agent_idx = agent_node_idx[i].item()  # Scalar index

            # --- Filter out padded edges ---
            # Padded edges have index >= num_nodes (or the value used for padding)
            # Here we assume padding value was self.num_nodes
            valid_edge_mask = (ei[0, :] < num_nodes) & (ei[1, :] < num_nodes)
            filtered_ei = ei[:, valid_edge_mask]

            # --- GNN forward pass ---
            x = self.conv1(x, filtered_ei)
            x = self.batch_norm1(x)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.conv2(x, filtered_ei)
            x = self.batch_norm2(x)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.conv3(x, filtered_ei)

            # Extract the feature vector for the specific agent's node
            agent_feature = x[agent_idx]  # Shape: [features_dim]
            output_features.append(agent_feature)

        # Stack features back into a batch
        output_batch = torch.stack(output_features)  # Shape: [batch_size, features_dim]
        return output_batch


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
        """
        Overrides the base method to potentially incorporate the action mask.
        Args:
            latent_pi: Features for the policy network.
        """
        mean_actions = self.action_net(latent_pi)  # Logits for all actions

        # --- Action Masking ---
        # The action mask should be part of the observation passed to the policy's forward method.
        # We need access to it here. SB3 policies store the last observation.
        if (
            isinstance(self.features_extractor, GNNFeatureExtractor)
            and hasattr(self, "_last_obs")
            and self._last_obs is not None
        ):
            if isinstance(self._last_obs, dict) and "action_mask" in self._last_obs:
                action_masks = self._last_obs[
                    "action_mask"
                ]  # Shape [batch_size, num_nodes]
                # Apply mask: Set logits of invalid actions to a large negative number
                VERY_LARGE_NEGATIVE_FLOAT = -1e9
                masked_logits = torch.where(
                    action_masks > 0,
                    mean_actions,
                    torch.tensor(VERY_LARGE_NEGATIVE_FLOAT, device=mean_actions.device),
                )
                # masked_logits = mean_actions + (1 - action_masks) * VERY_LARGE_NEGATIVE_FLOAT # Alternative
                action_distribution = self.action_dist.proba_distribution(
                    action_logits=masked_logits
                )
                return action_distribution
            else:
                print(
                    "Warning: Action mask not found in GNNPolicy _last_obs or observation is not dict."
                )
        else:
            print("Warning: Could not access action mask in GNNPolicy.")

        # Fallback to default behavior if mask is unavailable
        return self.action_dist.proba_distribution(action_logits=mean_actions)

    def forward(self, obs: Dict[str, torch.Tensor], deterministic: bool = False):
        """
        Need to store the observation for action masking in _get_action_dist_from_latent.
        """
        # Store the observation before calling the parent's forward method
        self._last_obs = obs
        return super().forward(obs, deterministic)
