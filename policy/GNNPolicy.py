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
            agent_idx_i = int(agent_node_indices[i].item())  ### Original ###
            original_agent_indices.append(agent_idx_i)

            edge_index_i = edge_index_i.long()  ### Original ###
            # {{ edit_1_start: Remove redundant edge filtering inside the loop }}
            # valid_edge_mask_i = (edge_index_i[0, :] < self.num_nodes) & (  ### Original ###
            #     edge_index_i[1, :] < self.num_nodes  ### Original ###
            # )
            # filtered_ei_i = edge_index_i[:, valid_edge_mask_i]  ### Original ###

            # Use edge_index_i directly, assuming wrapper padded correctly
            # and convolutions can handle potential disconnected padding nodes/edges.
            data = Data(
                x=x_i,
                edge_index=edge_index_i,
                agent_idx_in_graph=agent_idx_i,  # Use edge_index_i instead of filtered_ei_i
            )  ### Original ###
            # {{ edit_1_end }}
            data_list.append(data)

        try:
            batch_data = Batch.from_data_list(data_list).to(device)  ### Original ###
        except RuntimeError as e:
            print(f"Error creating PyG Batch: {e}")
            # Handle potential errors if removing filter causes issues downstream
            # For now, assume convolutions handle it gracefully.
            # Consider adding specific error handling or alternative padding if needed.
            return torch.zeros((batch_size, self.features_dim), device=device)

        # {{ edit_2: Apply convolutions using batch_data.edge_index directly }}
        x = self.conv1(batch_data.x, batch_data.edge_index)  # Use batch_data.edge_index
        if x.shape[0] > 0:
            x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(
            x, batch_data.edge_index
        )  ### Original ### # Use batch_data.edge_index
        if x.shape[0] > 0:
            x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(
            x, batch_data.edge_index
        )  ### Original ### # Use batch_data.edge_index

        # {{ edit_3_start: Optimize agent feature extraction loop slightly }}
        # Extract agent features: This part is harder to fully vectorize due to batch_data.ptr
        # Try to minimize python operations inside the loop by batching indexing if possible,
        # though significant speedup might be limited here without deeper changes.
        # The current approach is often necessary for heterogeneous batches.

        absolute_agent_indices = []
        valid_batch_indices = []
        invalid_indices_found = False  # Flag to print warning only once

        # Calculate start indices using batch_data.ptr
        start_node_indices = batch_data.ptr[:-1]  # ptr has shape [batch_size + 1]

        for i in range(batch_size):
            agent_idx_in_graph_i = original_agent_indices[i]
            if 0 <= agent_idx_in_graph_i < self.num_nodes:
                # Calculate absolute index without intermediate python variables
                absolute_index = start_node_indices[i] + agent_idx_in_graph_i
                absolute_agent_indices.append(
                    absolute_index.item()
                )  # Still need .item() here for list append
                valid_batch_indices.append(i)
            else:
                if not invalid_indices_found:  # Print only once
                    print(
                        f"Warning: Original agent index {agent_idx_in_graph_i} was invalid for batch item {i}. Skipping feature extraction. (Further warnings suppressed)"
                    )
                    invalid_indices_found = True

        # Indexing after the loop
        final_features = torch.zeros((batch_size, self.features_dim), device=device)
        if absolute_agent_indices:
            absolute_agent_indices_tensor = torch.tensor(  ### Original ###
                absolute_agent_indices, dtype=torch.long, device=device
            )
            # Check bounds before indexing x
            if absolute_agent_indices_tensor.max() < x.shape[0]:
                selected_features = x[absolute_agent_indices_tensor]

                if selected_features.numel() > 0:
                    valid_batch_indices_tensor = torch.tensor(
                        valid_batch_indices, dtype=torch.long, device=device
                    )
                    final_features[valid_batch_indices_tensor] = selected_features
            else:
                print(
                    f"Warning: Calculated absolute agent indices are out of bounds for feature tensor x. Max index: {absolute_agent_indices_tensor.max()}, x shape: {x.shape}. Returning zeros."
                )
        # else: # No valid agents, final_features remains zeros
        #    selected_features = torch.empty((0, self.features_dim), device=device)
        # {{ edit_3_end }}

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
