import torch
import torch.nn as nn
from typing import Dict
from gymnasium.spaces import Dict as GymDict
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

    def __init__(self, observation_space: GymDict, features_dim: int = 128):
        super().__init__(observation_space, features_dim=features_dim)

        # Extract dimensions from the observation space
        node_feature_dim = observation_space["node_features"].shape[1]
        self.num_nodes = observation_space["node_features"].shape[0]
        hidden_dim = 128

        # Define GNN layers (Example: 3 layers of GATv2)
        self.conv1 = GATv2Conv(node_feature_dim, hidden_dim // 4, heads=4)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 2, heads=2)
        self.conv3 = GATv2Conv(hidden_dim, features_dim, heads=1)

        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        self.last_obs = None

        print(f"GNN Feature Extractor Initialized:")
        print(f"  Input node feature dim: {node_feature_dim}")
        print(f"  Output features dim (shared network): {features_dim}")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.last_obs = observations

        node_features = observations["node_features"]  # [batch, num_nodes, feat_dim]
        edge_index = observations["edge_index"]  # [batch, 2, max_edges]
        agent_node_idx = observations[
            "agent_node_index"
        ]  # [batch, 1] <= Likely float32 here

        batch_size = node_features.shape[0]
        device = node_features.device
        output_features = []

        # Iterate through the batch (simpler for SB3 integration)
        for i in range(batch_size):
            x_i = node_features[i]  # [num_nodes, feat_dim]
            edge_index_i = edge_index[i]  # [2, max_edges]
            agent_idx_i = int(
                agent_node_idx[i].item()
            )  # scalar agent node index, ensure integer

            # Explicitly cast edge_index to long
            edge_index_i = edge_index_i.long()

            # Filter out padded edges for this specific graph instance
            # Assuming padding value is >= num_nodes
            valid_edge_mask_i = (edge_index_i[0, :] < self.num_nodes) & (
                edge_index_i[1, :] < self.num_nodes
            )
            filtered_ei_i = edge_index_i[:, valid_edge_mask_i]  # [2, num_valid_edges]

            # GNN Pass for the i-th item
            x = self.conv1(x_i, filtered_ei_i)
            # Apply BatchNorm correctly (expects [N, C] or [C])
            # If num_nodes is consistent, BatchNorm1d works. Careful if graphs vary size.
            if x.shape[0] > 0:  # Avoid BatchNorm on empty graphs if possible
                # BatchNorm expects [N, C], conv output might be [N, C*heads], need reshape if heads > 1
                # GAT output is [N, heads * out_channels], flatten for BatchNorm? Or apply per head?
                # Let's assume GAT handles head concatenation internally for the next layer
                # Apply BatchNorm on the flattened head dimension if needed, or ensure output dim matches BN dim
                # Simpler: Apply BN after ReLU often works too. Let's try standard order first.
                # Ensure dimensions match BatchNorm layer (hidden_dim)
                x = self.batch_norm1(x)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.conv2(x, filtered_ei_i)
            if x.shape[0] > 0:
                # Ensure dimensions match BatchNorm layer (hidden_dim)
                x = self.batch_norm2(x)
            x = self.relu(x)
            x = self.dropout(x)

            x = self.conv3(x, filtered_ei_i)  # Output: [num_nodes, features_dim]

            # Extract the feature for the agent's node
            # Handle case where agent_idx might be invalid (e.g., -1 if agent not present)
            if 0 <= agent_idx_i < self.num_nodes:
                agent_feature = x[
                    agent_idx_i
                ]  # [features_dim] # Indexing now uses integer
            else:
                # Handle invalid index - perhaps return zeros?
                # Ensure the created zero tensor matches the expected features_dim
                agent_feature = torch.zeros(self.features_dim, device=device)
                print(
                    f"Warning: Invalid agent_node_idx {agent_idx_i} encountered in batch item {i}."
                )

            output_features.append(agent_feature)

        # Stack the features from all items in the batch
        final_features = torch.stack(output_features)  # [batch_size, features_dim]
        return final_features


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
        features_dim=128,  # <<<< Match features_dim here with GNNFeatureExtractor
        *args,
        **kwargs,
    ):
        # Disable MLP extractor and use our GNN extractor
        kwargs["features_extractor_class"] = GNNFeatureExtractor
        # Ensure features_dim passed here matches the one used in GNNFeatureExtractor init
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
        mean_actions = self.action_net(latent_pi)  # Shape [batch_size, num_nodes]

        # {{ edit_4_start: Get masks from features_extractor }}
        # Check if the features extractor and its stored obs are available
        if (
            hasattr(self, "features_extractor")
            and hasattr(self.features_extractor, "last_obs")
            and self.features_extractor.last_obs is not None
            and isinstance(self.features_extractor.last_obs, dict)
            and "action_mask" in self.features_extractor.last_obs
        ):
            # Retrieve the action masks corresponding to the latent_pi batch
            action_masks = self.features_extractor.last_obs[
                "action_mask"
            ]  # Shape [batch_size, num_nodes]

            # Ensure mask is on the same device and has the correct batch size
            action_masks = action_masks.to(mean_actions.device)

            # --- Crucial Check: Compare batch sizes ---
            if action_masks.shape[0] != mean_actions.shape[0]:
                print(
                    f"Warning: Action mask batch size ({action_masks.shape[0]}) does not match "
                    f"latent_pi batch size ({mean_actions.shape[0]}) in _get_action_dist_from_latent. "
                    "Using unmasked actions."
                )
                # Fallback to unmasked actions if sizes mismatch unexpectedly
                return self.action_dist.proba_distribution(action_logits=mean_actions)
            # --- End Check ---

            # Apply the mask
            masked_logits = torch.where(
                action_masks > 0,
                mean_actions,
                torch.tensor(-1e9, device=mean_actions.device),
            )
            return self.action_dist.proba_distribution(action_logits=masked_logits)
        else:
            # Fallback if mask isn't available for some reason
            print(
                "Warning: Action mask not found in features_extractor.last_obs. Using unmasked actions."
            )
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        # {{ edit_4_end }}
