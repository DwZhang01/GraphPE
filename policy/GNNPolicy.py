import torch
import torch.nn as nn
from typing import Dict, List
from gymnasium.spaces import Dict as GymDict
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import (
    SAGEConv,
    GATv2Conv,
    global_mean_pool,
)
from torch_geometric.data import Data, Batch
from gymnasium import spaces

# (GNNEnvWrapper class defined above)


class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using GraphSAGE (or GATv2) for the GPE environment.
    Assumes observation space is the Dict space defined in GNNEnvWrapper.
    Uses PyG Batching for efficient processing.
    """

    def __init__(self, observation_space: GymDict, features_dim: int = 64):
        super().__init__(observation_space, features_dim=features_dim)

        node_feature_dim = observation_space["node_features"].shape[1]
        self.num_nodes = observation_space["node_features"].shape[0]
        hidden_dim = features_dim

        self.conv1 = GATv2Conv(node_feature_dim, hidden_dim // 4, heads=4)
        self.conv2 = GATv2Conv(hidden_dim, hidden_dim // 2, heads=2)
        self.conv3 = GATv2Conv(hidden_dim, features_dim, heads=1)

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.last_obs = None

        print(f"GNN Feature Extractor Initialized (PyG Batching Enabled):")
        print(f"  Input node feature dim: {node_feature_dim}")
        print(f"  Using LayerNorm instead of BatchNorm1d.")
        print(f"  Output features dim (shared network): {features_dim}")

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.last_obs = observations

        node_features = observations["node_features"]
        edge_indices = observations["edge_index"]
        agent_node_indices = observations["agent_node_index"]

        batch_size = node_features.shape[0]
        device = node_features.device

        data_list: List[Data] = []
        original_agent_indices = []

        for i in range(batch_size):
            x_i = node_features[i]
            edge_index_i = edge_indices[i]
            agent_idx_i = int(agent_node_indices[i].item())
            original_agent_indices.append(agent_idx_i)

            edge_index_i = edge_index_i.long()
            valid_edge_mask_i = (edge_index_i[0, :] < self.num_nodes) & (
                edge_index_i[1, :] < self.num_nodes
            )
            filtered_ei_i = edge_index_i[:, valid_edge_mask_i]

            data = Data(x=x_i, edge_index=filtered_ei_i, agent_idx_in_graph=agent_idx_i)
            data_list.append(data)

        try:
            batch_data = Batch.from_data_list(data_list).to(device)
        except RuntimeError as e:
            print(f"Error creating PyG Batch: {e}")
            return torch.zeros((batch_size, self.features_dim), device=device)

        x = self.conv1(batch_data.x, batch_data.edge_index)
        if x.shape[0] > 0:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, batch_data.edge_index)
        if x.shape[0] > 0:
            x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x, batch_data.edge_index)

        absolute_agent_indices = []
        valid_batch_indices = []

        for i in range(batch_size):
            agent_idx_in_graph_i = original_agent_indices[i]
            if 0 <= agent_idx_in_graph_i < self.num_nodes:
                start_node_index = batch_data.ptr[i]
                absolute_index = start_node_index + agent_idx_in_graph_i
                absolute_agent_indices.append(absolute_index)
                valid_batch_indices.append(i)
            else:
                print(
                    f"Warning: Original agent index {agent_idx_in_graph_i} was invalid for batch item {i}. Skipping feature extraction."
                )

        if absolute_agent_indices:
            absolute_agent_indices_tensor = torch.tensor(
                absolute_agent_indices, dtype=torch.long, device=device
            )
            selected_features = x[absolute_agent_indices_tensor]
            # global_features = global_mean_pool(x, batch_data.batch)  # 全局池化
            # selected_features = torch.cat([selected_features, global_features], dim=1)
            # 调整 features_dim 包含全局特征，例如 32（节点特征）+ 32（全局特征）= 64
        else:
            selected_features = torch.empty((0, self.features_dim), device=device)

        final_features = torch.zeros((batch_size, self.features_dim), device=device)
        if selected_features.numel() > 0:
            valid_batch_indices_tensor = torch.tensor(
                valid_batch_indices, dtype=torch.long, device=device
            )
            final_features[valid_batch_indices_tensor] = selected_features

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
        net_arch=None,
        activation_fn=nn.ReLU,
        *args,
        **kwargs,
    ):
        if "features_extractor_class" not in kwargs:
            kwargs["features_extractor_class"] = GNNFeatureExtractor

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
