import torch
import torch.nn as nn
import torch.nn.functional as F # For ReLU if not using nn.ReLU() instance
from typing import Dict, List, Tuple
from gymnasium.spaces import Dict as GymDict # Keep if observation_space type hint is this
from gymnasium import spaces # For BaseFeaturesExtractor observation_space type hint
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch_geometric.nn import GATv2Conv # Using GATv2Conv as per your code
from torch_geometric.data import Data, Batch
import numpy as np
import logging
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Optional

class GNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using GATv2Conv for the GPE environment.
    - Assumes observation space is Dict{"node_features": Box, "action_mask": Box}.
    - Uses a static graph_edge_index provided during initialization.
    - Extracts features for the agent's specific node after GNN processing.
    - Uses PyG Batching for efficient batch processing.
    """

    def __init__(self, 
                 observation_space: spaces.Dict, 
                 features_dim: int = 64, 
                 graph_edge_index: np.ndarray = None,
                 ):
        super().__init__(observation_space, features_dim=features_dim)

        assert isinstance(observation_space, spaces.Dict), "GNNFeatureExtractor expects a Dict observation space."
        node_feature_dim = observation_space["node_features"].shape[1]
        self.gpe_num_nodes = observation_space["node_features"].shape[0]

        # Store static graph edge index as a PyTorch tensor
        self.graph_edge_index_pt = torch.tensor(graph_edge_index, dtype=torch.long)

        # GNN architecture
        # GATv2Conv concatenates multi-head outputs by default.
        # hidden_dim here is the dimension *before* head concatenation for each layer if heads > 1
        gnn_hidden_dim_per_head_l1 = features_dim // 4 # Or another suitable size
        gnn_hidden_dim_per_head_l2 = features_dim // 2 # Or another suitable size
        heads_l1 = 4
        heads_l2 = 2
        heads_l3 = 1

        # Effective hidden dimensions after head concatenation
        effective_hidden_dim_l1 = gnn_hidden_dim_per_head_l1 * heads_l1
        effective_hidden_dim_l2 = gnn_hidden_dim_per_head_l2 * heads_l2
        
        self.conv1 = GATv2Conv(node_feature_dim, gnn_hidden_dim_per_head_l1, heads=heads_l1)
        self.norm1 = nn.LayerNorm(effective_hidden_dim_l1)
        
        self.conv2 = GATv2Conv(effective_hidden_dim_l1, gnn_hidden_dim_per_head_l2, heads=heads_l2)
        self.norm2 = nn.LayerNorm(effective_hidden_dim_l2)
        
        self.conv3 = GATv2Conv(effective_hidden_dim_l2, features_dim, heads=heads_l3) # Output layer, features_dim after concat

        self.dropout = nn.Dropout(0.1) # Consider making dropout rate configurable
        self.relu = nn.ReLU()
        
        # To store the last observation (primarily for action_mask access by the policy)
        self.last_obs_for_policy: Dict[str, torch.Tensor] = None 

        logging.info(f"GNN Feature Extractor Initialized:")
        logging.info(f"  Input node feature dim from GPE: {node_feature_dim}")
        logging.info(f"  GPE num_nodes: {self.gpe_num_nodes}")
        logging.info(f"  GNN Layer 1: GATv2Conv({node_feature_dim}, {gnn_hidden_dim_per_head_l1}) with {heads_l1} heads -> Dim: {effective_hidden_dim_l1}")
        logging.info(f"  GNN Layer 2: GATv2Conv({effective_hidden_dim_l1}, {gnn_hidden_dim_per_head_l2}) with {heads_l2} heads -> Dim: {effective_hidden_dim_l2}")
        logging.info(f"  GNN Layer 3: GATv2Conv({effective_hidden_dim_l2}, {features_dim}) with {heads_l3} heads -> Output Dim: {features_dim * heads_l3}") # Should be features_dim
        logging.info(f"  Output features_dim (for SB3 policy): {features_dim}")


    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:

        self.last_obs_for_policy = observations

        node_features_batch = observations["node_features"]
        # action_mask_batch = observations["action_mask"] # (如果需要的话)
        

        batch_size = node_features_batch.shape[0]
        device = node_features_batch.device

        # Ensure static edge_index is on the correct device
        static_edge_index = self.graph_edge_index_pt.to(device)

        data_list: List[Data] = []
        agent_node_indices_in_graph: List[int] = [] # Store agent's node index (0 to num_nodes-1) for each item in batch

        # In GPE, "is_self" is feature index 0 (as per _node_feature_list)
        is_self_feature_index = 0 

        for i in range(batch_size):
            x_i = node_features_batch[i] # Shape: (num_nodes, input_node_feature_dim)
            
            # Determine the agent's node index in its graph using the "is_self" feature.
            # This assumes "is_self" is 1.0 for exactly one node for active agents,
            # and 0.0 for all nodes in terminal observations (where features are zeroed out).
            is_self_column = x_i[:, is_self_feature_index]
            
            # Find indices where "is_self" is 1.0
            # Check if any node is marked as 'self'. This handles terminal obs where all features are 0.
            if torch.any(is_self_column > 0.5): # Using > 0.5 for float comparison
                agent_node_idx = torch.argmax(is_self_column).item()
            else:
                # For terminal observations (all zeros) or if "is_self" is missing,
                # default to index 0. The GNN output for this will be based on zero features.
                agent_node_idx = 0 
            agent_node_indices_in_graph.append(agent_node_idx)
            
            # Create PyG Data object for this sample.
            # The edge_index is the same static one for all samples.
            # PyG Batch.from_data_list will handle batching this correctly.
            data_list.append(Data(x=x_i, edge_index=static_edge_index))

        # Create a single PyG Batch object from the list of Data objects
        try:
            pyg_batch = Batch.from_data_list(data_list).to(device)
        except RuntimeError as e:
            logging.error(f"Error creating PyG Batch: {e}. Input shapes: node_features_batch={node_features_batch.shape}", exc_info=True)
            # Return a zero tensor of the expected output shape
            return torch.zeros((batch_size, self.features_dim), device=device)

        # GNN forward pass
        # pyg_batch.x shape: (batch_size * num_nodes, input_node_feature_dim)
        # pyg_batch.edge_index shape: (2, batch_size * num_edges_per_graph_if_all_same) - PyG handles this
        
        x = self.conv1(pyg_batch.x, pyg_batch.edge_index)
        if x.shape[0] > 0: # Ensure not empty batch before norm
            x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, pyg_batch.edge_index)
        if x.shape[0] > 0:
            x = self.norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Final GNN features for all nodes in the batch
        gnn_output_all_nodes = self.conv3(x, pyg_batch.edge_index) # Shape: (batch_size * num_nodes, self.features_dim)

        # --- Extract features for each agent's specific node ---
        # `pyg_batch.ptr` gives the starting index of nodes for each graph in the batch.
        # `agent_node_indices_in_graph` contains the agent's node index *within its original graph*.
        
        final_agent_features = torch.zeros((batch_size, self.features_dim), device=device)
        
        if gnn_output_all_nodes.shape[0] == 0: # Handles cases like empty batch_size or empty graphs
            # logging.warning("GNN output has 0 nodes. Returning zero features.") # Optional warning
            return final_agent_features

        start_indices_of_graphs = pyg_batch.ptr[:-1] # Shape: [batch_size]
        agent_node_indices_tensor = torch.tensor(agent_node_indices_in_graph, dtype=torch.long, device=device) # Shape: [batch_size]

        # Calculate the absolute index in `gnn_output_all_nodes` for each agent's node
        absolute_indices = start_indices_of_graphs + agent_node_indices_tensor
        
        # Safety clamp (though derived indices should be valid if "is_self" logic is robust)
        absolute_indices_clamped = torch.clamp(absolute_indices, 0, gnn_output_all_nodes.shape[0] - 1)
        
        selected_features = gnn_output_all_nodes[absolute_indices_clamped]
        final_agent_features = selected_features # Shape: (batch_size, self.features_dim)
            
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