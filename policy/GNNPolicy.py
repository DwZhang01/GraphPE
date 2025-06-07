import torch
import torch.nn as nn
from typing import Dict,Optional
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import Batch
import numpy as np
import logging
from stable_baselines3.common.policies import ActorCriticPolicy

class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    An optimized feature extractor using GATv2Conv for the GPE environment.
    This implementation avoids Python loops in the forward pass by using vectorized
    PyTorch operations to construct the PyG Batch object directly.
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 64, graph_edge_index: np.ndarray = None):
        super().__init__(observation_space, features_dim=features_dim)

        assert isinstance(observation_space, spaces.Dict), "GNNFeatureExtractor expects a Dict observation space."
        assert graph_edge_index is not None, "GNNFeatureExtractor requires 'graph_edge_index'."

        self.gpe_num_nodes = observation_space["node_features"].shape[0]
        input_node_feature_dim = observation_space["node_features"].shape[1]
        
        # Store static graph edge index as a PyTorch tensor on CPU initially.
        # It will be moved to the correct device in the forward pass.
        self.register_buffer('graph_edge_index_pt', torch.tensor(graph_edge_index, dtype=torch.long))

        # GNN architecture (same as before)
        gnn_hidden_dim_per_head_l1 = features_dim // 4
        gnn_hidden_dim_per_head_l2 = features_dim // 2
        heads_l1, heads_l2, heads_l3 = 4, 2, 1
        effective_hidden_dim_l1 = gnn_hidden_dim_per_head_l1 * heads_l1
        effective_hidden_dim_l2 = gnn_hidden_dim_per_head_l2 * heads_l2
        
        self.conv1 = GATv2Conv(input_node_feature_dim, gnn_hidden_dim_per_head_l1, heads=heads_l1)
        self.norm1 = nn.LayerNorm(effective_hidden_dim_l1)
        self.conv2 = GATv2Conv(effective_hidden_dim_l1, gnn_hidden_dim_per_head_l2, heads=heads_l2)
        self.norm2 = nn.LayerNorm(effective_hidden_dim_l2)
        self.conv3 = GATv2Conv(effective_hidden_dim_l2, features_dim, heads=heads_l3)

        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.last_obs_for_policy: Dict[str, torch.Tensor] = None

        logging.info(f"GNN Feature Extractor Initialized:")
        logging.info(f"  Input node feature dim from GPE: {input_node_feature_dim}")
        logging.info(f"  GPE num_nodes: {self.gpe_num_nodes}")
        logging.info(f"  GNN Layer 1: GATv2Conv({input_node_feature_dim}, {gnn_hidden_dim_per_head_l1}) with {heads_l1} heads -> Dim: {effective_hidden_dim_l1}")
        logging.info(f"  GNN Layer 2: GATv2Conv({effective_hidden_dim_l1}, {gnn_hidden_dim_per_head_l2}) with {heads_l2} heads -> Dim: {effective_hidden_dim_l2}")
        logging.info(f"  GNN Layer 3: GATv2Conv({effective_hidden_dim_l2}, {features_dim}) with {heads_l3} heads -> Output Dim: {features_dim * heads_l3}") # Should be features_dim
        logging.info(f"  Output features_dim (for SB3 policy): {features_dim}")


    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Store observations for the policy to access the action mask
        self.last_obs_for_policy = observations

        node_features_batch = observations["node_features"] # Shape: (B, N, F_in)
        batch_size = node_features_batch.shape[0]
        device = node_features_batch.device

        # --- Vectorized PyG Batch Construction (No Python Loop!) ---
        
        # 1. Concatenate all node features into a single tensor `x`
        # Input shape (B, N, F_in) -> Output shape (B * N, F_in)
        x = node_features_batch.reshape(-1, node_features_batch.shape[-1])

        # 2. Create the batched `edge_index` with correct offsets
        static_edge_index = self.graph_edge_index_pt.to(device) # Shape (2, E)
        num_edges_per_graph = static_edge_index.shape[1]
        
        # Create offsets for each graph in the batch: [0, N, 2N, 3N, ...]
        offsets = torch.arange(0, batch_size * self.gpe_num_nodes, self.gpe_num_nodes, device=device)
        # Add offsets to edge_index for each graph
        # This uses broadcasting to efficiently add the offset to each graph's edges
        edge_index = static_edge_index.repeat(1, batch_size) + offsets.repeat_interleave(num_edges_per_graph)
        #   [[0, 1,  0, 1,  0, 1],      # 部分一
        #    [1, 2,  1, 2,  1, 2]]
        # +
        #   [[0, 0,  4, 4,  8, 8],      # 部分二 (广播后)
        #    [0, 0,  4, 4,  8, 8]]
        # =
        #   [[0, 1,  4, 5,  8, 9],      # 最终结果
        #    [1, 2,  5, 6,  9, 10]]

        
        # --- End of Vectorized Construction ---

        # GNN forward pass using the constructed tensors
        gnn_x = self.relu(self.norm1(self.conv1(x, edge_index)))
        gnn_x = self.dropout(gnn_x)
        gnn_x = self.relu(self.norm2(self.conv2(gnn_x, edge_index)))
        gnn_x = self.dropout(gnn_x)
        gnn_processed_node_features = self.conv3(gnn_x, edge_index) # Shape: (B * N, F_out)
        
        # --- Vectorized Agent-Specific Feature Readout ---
        # Find the node index for each agent in its graph using the "is_self" feature (index 0)
        # This is also done without a Python loop using torch.argmax on the correct dimension.
        # Get is_self feature (index 0) for each node in the batch
        # Shape: (B, N, 1) -> (B,)
        agent_nodes_in_indiv_graphs = torch.argmax(node_features_batch[:, :, 0], dim=1) # Shape: (B,)
        
        # Calculate the absolute index in the large `gnn_processed_node_features` tensor
        # `offsets` is [0, N, 2N, ...], which are the start indices of each graph
        absolute_agent_indices = offsets + agent_nodes_in_indiv_graphs
        
        # Gather the features for each agent's node
        final_agent_features = gnn_processed_node_features[absolute_agent_indices] # Shape: (B, F_out)
            
        # ONLY RETURN THE FEATURE of THE 'is_self' AGENT NODE
        # THAT CAN BE USELESS !
        return final_agent_features
        
    

class GNNPolicy(ActorCriticPolicy):
    """
    Custom policy using the GNNFeatureExtractor.
    Applies action masking using the 'action_mask' from the original observation.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.ReLU,
        # Pass features_extractor_kwargs from PPO model to here,
        # then down to GNNFeatureExtractor
        features_extractor_class=GNNFeatureExtractor, # Default to our GNNFeatureExtractor
        features_extractor_kwargs: Optional[Dict] = None, # Allow passing kwargs
        *args,
        **kwargs,
    ):
        # Ensure default features_extractor_kwargs is an empty dict if None
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {}
            
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs, # Pass it down
            *args,
            **kwargs,
        )
        # self.share_features_extractor is True by default in ActorCriticPolicy if not specified
        # It's good to keep it True for PPO with a shared GNN backbone.

    def _get_action_dist_from_latent(self, latent_pi: torch.Tensor, latent_sde=None):
        mean_actions = self.action_net(latent_pi)  # Logits, Shape [batch_size, num_actions] (num_actions = num_nodes)

        # Retrieve action_mask from the feature extractor's saved observation
        if (
            hasattr(self, "features_extractor")
            and hasattr(self.features_extractor, "last_obs_for_policy") # Check for the renamed attribute
            and self.features_extractor.last_obs_for_policy is not None
            and isinstance(self.features_extractor.last_obs_for_policy, dict)
            and "action_mask" in self.features_extractor.last_obs_for_policy
        ):
            action_masks = self.features_extractor.last_obs_for_policy["action_mask"] # Shape [batch_size, num_nodes]

            if action_masks.shape[0] != mean_actions.shape[0]:
                logging.warning(
                    f"Action mask batch size ({action_masks.shape[0]}) does not match "
                    f"logits batch size ({mean_actions.shape[0]}). Using unmasked actions."
                )
                return self.action_dist.proba_distribution(action_logits=mean_actions)
            
            if action_masks.shape[1] != mean_actions.shape[1]:
                logging.warning(
                    f"Action mask num_actions ({action_masks.shape[1]}) does not match "
                    f"logits num_actions ({mean_actions.shape[1]}). Using unmasked actions."
                )
                return self.action_dist.proba_distribution(action_logits=mean_actions)


            # Ensure mask is on the same device as logits
            action_masks = action_masks.to(mean_actions.device)

            # Apply the mask: Where mask is 0, set logits to a very small number.
            masked_logits = torch.where(
                action_masks.bool(), # Convert float mask (0s and 1s) to boolean
                mean_actions,
                torch.full_like(mean_actions, -1e9), # Use full_like for safety
            )
            
            # # Check for samples where all actions might be masked
            # # (e.g., if an agent has no valid moves, which GPE's _get_action_mask should prevent if possible)
            # if not torch.any(action_masks.bool(), dim=1).all():
            #     # This means for at least one sample in the batch, all actions are masked.
            #     # Iterate and log details for such samples.
            #     for i in range(action_masks.shape[0]):
            #         if not torch.any(action_masks[i].bool()):
            #             logging.warning(
            #                 f"Sample {i} in batch has all actions masked. "
            #                 f"Logits before mask: {mean_actions[i].detach().cpu().numpy()}. "
            #                 f"Mask: {action_masks[i].detach().cpu().numpy()}"
            #             )
            #             # You might want to add a small probability to all actions or a default action
            #             # to prevent potential NaNs if the policy must select an action.
            #             # For now, SB3's CategoricalDistribution might handle it by picking one randomly from the all-low-logit actions.
            
            return self.action_dist.proba_distribution(action_logits=masked_logits)
        else:
            logging.warning(
                "Action mask not found or invalid in features_extractor.last_obs_for_policy. Using unmasked actions."
            )
            return self.action_dist.proba_distribution(action_logits=mean_actions)