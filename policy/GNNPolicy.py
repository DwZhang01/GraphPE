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

        # Check for CPU single process
        for i in range(batch_size):
            x_i = node_features[i]
            edge_index_i = edge_indices[i]
            agent_idx_i = int(agent_node_indices[i].item())  ### Original ###
            original_agent_indices.append(agent_idx_i)

            edge_index_i = edge_index_i.long()  ### Original ###

            # {{ edit_7_start: Re-introduce edge filtering inside the loop }}
            # Filter out edges that use the padding index (self.num_nodes)
            # The padding value was set in GNNEnvWrapper._update_graph_pyg
            # Valid node indices are 0 to self.num_nodes - 1.
            valid_edge_mask_i = (edge_index_i[0, :] < self.num_nodes) & (
                edge_index_i[1, :] < self.num_nodes
            )
            filtered_ei_i = edge_index_i[:, valid_edge_mask_i]

            # Use the filtered edge_index
            data = Data(
                x=x_i,
                edge_index=filtered_ei_i,  # Use filtered edge index
                agent_idx_in_graph=agent_idx_i,
            )  ### Original ###
            # {{ edit_7_end }}
            data_list.append(data)

        # Batch.from_data_list will now correctly handle the filtered edge indices
        try:
            batch_data = Batch.from_data_list(data_list).to(device)  ### Original ###
        except RuntimeError as e:
            print(f"Error creating PyG Batch: {e}")
            return torch.zeros((batch_size, self.features_dim), device=device)

        # {{ edit_8: Convolutions now receive edge_index without padding values }}
        # The batch_data.edge_index passed here will be correctly offset but
        # will only contain indices corresponding to actual nodes (0 to num_nodes_in_batch - 1).
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
        # {{ edit_8_end }}

        # {{ edit_6_start: Refine vectorized feature extraction with clamping (This part should still be okay) }}
        # --- Vectorized Agent Feature Extraction (Refined) ---
        final_features = torch.zeros((batch_size, self.features_dim), device=device)

        # 1. Get start indices from batch_data.ptr
        start_node_indices = batch_data.ptr[:-1]  # Shape: [batch_size]

        # 2. Convert original agent indices to tensor
        if not original_agent_indices:
            # print("Warning: original_agent_indices list is empty. Returning zeros.") # Keep if needed
            return final_features
        try:
            original_agent_indices_tensor = torch.tensor(
                original_agent_indices, dtype=torch.long, device=device
            )  # Shape: [batch_size]
        except ValueError:
            print("Error: Could not convert original_agent_indices to tensor.")
            return final_features

        # 3. Create a mask for valid agent indices (within graph bounds)
        valid_mask = (original_agent_indices_tensor >= 0) & (
            original_agent_indices_tensor < self.num_nodes
        )  # Shape: [batch_size]

        # 4. Calculate absolute indices for ALL agents
        if start_node_indices.shape[0] != original_agent_indices_tensor.shape[0]:
            print("Error: Mismatch between ptr size and original indices.")
            return final_features
        absolute_agent_indices_tensor = (
            start_node_indices + original_agent_indices_tensor
        )  # Shape: [batch_size]

        # 5. Check GNN output 'x' and prepare for indexing
        if x.nelement() == 0:
            # if batch_size > 0: print("Warning: GNN output 'x' is empty.")
            return final_features  # Return zeros

        # Check if any originally valid indices exist before proceeding
        if torch.any(valid_mask):
            # *** Optimization: Clamp indices BEFORE indexing x to prevent potential out-of-bounds errors ***
            # Clamp all absolute indices to be within the valid range of x's first dimension.
            # Invalid indices (where valid_mask is False) will also be clamped, but their results won't be used later.
            num_nodes_in_batch = x.shape[0]
            absolute_agent_indices_clamped = torch.clamp(
                absolute_agent_indices_tensor, min=0, max=num_nodes_in_batch - 1
            )

            # Index x using the clamped indices. This is now guaranteed to be safe.
            selected_features = x[
                absolute_agent_indices_clamped
            ]  # Shape: [batch_size, features_dim]

            # 6. Use the original valid_mask to place the correct features into the final result.
            # This ensures that only features corresponding to originally valid and in-bounds
            # agent indices are actually stored. Features from clamped invalid indices are ignored.
            final_features[valid_mask] = selected_features[valid_mask]

        # else: no valid agents, final_features remains zeros
        # --- End Vectorized Extraction ---
        # {{ edit_6_end }} (Keep this vectorized part)

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

        # {{ edit_4_start: Get masks from features_extractor (Review and Confirm) }}
        # This block seems correct for applying the action mask retrieved from the extractor.
        # Ensure features_extractor.last_obs is reliably populated before this call.
        if (
            hasattr(self, "features_extractor")
            and hasattr(self.features_extractor, "last_obs")
            and self.features_extractor.last_obs is not None
            and isinstance(self.features_extractor.last_obs, dict)
            and "action_mask" in self.features_extractor.last_obs
        ):
            action_masks = self.features_extractor.last_obs[
                "action_mask"
            ]  # Shape [batch_size, num_nodes]

            # Ensure mask is on the same device
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

            # Apply the mask: Add a small epsilon for numerical stability if needed,
            # but -1e9 should be sufficient.
            masked_logits = torch.where(
                action_masks > 0,  # Use > 0 for float masks
                mean_actions,
                torch.tensor(
                    -1e9, device=mean_actions.device, dtype=mean_actions.dtype
                ),  # Match dtype
            )
            # Check if all actions are masked (can happen in edge cases)
            if not torch.any(action_masks > 0, dim=1).all():
                print(
                    "Warning: Some samples have all actions masked in _get_action_dist_from_latent!"
                )
                # Optional: Handle this case, e.g., by allowing a default action or raising an error.
                # For now, PPO might handle it, potentially leading to NaN gradients if probabilities are zero.

            return self.action_dist.proba_distribution(action_logits=masked_logits)
        else:
            print(
                "Warning: Action mask not found or invalid in features_extractor.last_obs. Using unmasked actions."
            )
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        # {{ edit_4_end }}
