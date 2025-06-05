import sys
import os

# Get current script directory and project root
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import cProfile
import pstats
import io
import json
import time
import numpy as np
import torch
from gymnasium.spaces import Box, Dict as GymDict

# Import GNNFeatureExtractor and GPE (for reference, not instantiation here)
from policy.GNNPolicy import GNNFeatureExtractor
# from GPE.env.graph_pe import GPE # Not strictly needed if we manually define space

# --- Configuration ---
CONFIG_PATH = "config.json" # Ensure this path is correct relative to where you run the script
NUM_FORWARD_CALLS = 1000 # Increased for potentially more stable profiling
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load Configuration ---
try:
    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)
    env_config = config.get("environment", {})
    nn_config = config.get("neural_network", {})
    print("Configuration loaded.")
except Exception as e:
    print(f"Error loading config from '{CONFIG_PATH}': {e}")
    exit()

# --- Define Observation Space Structure (must match GPE.py) ---
# This should be identical to how it's defined in your GPE environment's __init__
gpe_node_feature_list = [
    "is_self",
    "is_pursuer",
    "is_evader",
    "is_safe_node",
]
gpe_feature_dim = len(gpe_node_feature_list)

# num_nodes for the mock space should ideally come from a consistent source,
# e.g., your config, or a default value if config doesn't specify for this test.
# Using a default if not in config, ensure it's a reasonable value for testing.
num_nodes = env_config.get("num_nodes", 50) # Default to 50 if not in env_config

mock_observation_space = GymDict(
    {
        "node_features": Box(
            low=0,
            high=1.0, # Binary features typically range 0 to 1
            shape=(num_nodes, gpe_feature_dim),
            dtype=np.float32,
        ),
        "action_mask": Box(
            low=0,
            high=1,
            shape=(num_nodes,),
            dtype=np.float32,
        ),
        # "edge_index" and "agent_node_index" are NOT part of GPE's observation_space anymore
    }
)
print(
    f"Mock observation space created with num_nodes={num_nodes}, actual feature_dim={gpe_feature_dim}"
)

# --- Create Dummy Graph Edge Index (as NumPy array) ---
# This simulates the self.graph_edge_index from the GPE environment.
# For profiling, its exact structure might not be critical, but its shape and type are.
# Let's create a simple random graph structure for it.
num_dummy_edges = num_nodes * 5 # Example: average degree of 5, can be adjusted
if num_nodes > 0:
    dummy_graph_edge_index_np = np.random.randint(
        0, num_nodes, size=(2, num_dummy_edges), dtype=np.int64
    )
    # Ensure no self-loops if GPE graph doesn't have them, though for GNN processing it might not matter much
    # dummy_graph_edge_index_np = dummy_graph_edge_index_np[:, dummy_graph_edge_index_np[0, :] != dummy_graph_edge_index_np[1, :]]
else:
    dummy_graph_edge_index_np = np.empty((2,0), dtype=np.int64)

print(f"Dummy graph_edge_index created with shape: {dummy_graph_edge_index_np.shape}")


# --- Create Feature Extractor Instance ---
print("Creating GNNFeatureExtractor instance...")
try:
    feature_extractor = GNNFeatureExtractor(
        observation_space=mock_observation_space,
        features_dim=nn_config.get("FEATURES_DIM", 64), # Get from config or default
        graph_edge_index=dummy_graph_edge_index_np # Pass the dummy graph structure
    ).to(DEVICE)
    feature_extractor.eval()  # Set to evaluation mode
    print(f"Feature extractor created on device: {DEVICE}")
except Exception as e:
    print(f"Error creating GNNFeatureExtractor: {e}")
    print("Ensure 'FEATURES_DIM' is in nn_config of your config.json and GNNFeatureExtractor __init__ is correct.")
    exit()

# --- Create Simulated Input Data (Dummy Observation Batch) ---
print(f"Creating dummy observation batch (batch_size={BATCH_SIZE})...")
dummy_obs_batch = {
    "node_features": torch.rand(
        BATCH_SIZE, num_nodes, gpe_feature_dim, dtype=torch.float32
    ).to(DEVICE),
    "action_mask": torch.randint(
        0, 2, (BATCH_SIZE, num_nodes), dtype=torch.float32
    ).to(DEVICE),
    # "edge_index" and "agent_node_index" are REMOVED from the observation batch
}
print("Dummy observation batch created.")

# --- Warm-up Run ---
# Especially important for GPU to compile kernels, etc.
print("Warm-up run (10 calls)...")
with torch.no_grad():
    for _ in range(10): # Few calls for warm-up
        _ = feature_extractor(dummy_obs_batch)
if DEVICE == "cuda":
    torch.cuda.synchronize() # Wait for CUDA operations to complete
print("Warm-up done.")

# --- Profiling Run ---
profiler = cProfile.Profile()
print(f"Profiling {NUM_FORWARD_CALLS} forward calls...")

profiler.enable()
start_time = time.time()

with torch.no_grad():  # Ensure no gradient calculations during profiling
    for _ in range(NUM_FORWARD_CALLS):
        _ = feature_extractor(dummy_obs_batch)
        if DEVICE == "cuda":
            torch.cuda.synchronize() # Synchronize after each call for accurate GPU timing if calls are fast

profiler.disable()
end_time = time.time()

duration = end_time - start_time
print(f"Profiling finished in {duration:.3f} seconds.")
if NUM_FORWARD_CALLS > 0 and duration > 0:
    calls_per_second = NUM_FORWARD_CALLS / duration
    ms_per_call = (duration * 1000) / NUM_FORWARD_CALLS
    print(f"  Average calls per second: {calls_per_second:.2f}")
    print(f"  Average milliseconds per call: {ms_per_call:.3f} ms")


# --- Analyze and Print Profiling Results ---
print(
    "\n--- cProfile Results for GNNFeatureExtractor.forward (Sorted by Total Time 'tottime') ---"
)
s_tottime = io.StringIO()
stats_tottime = pstats.Stats(profiler, stream=s_tottime).sort_stats("tottime")
stats_tottime.print_stats(40) # Print top 40 functions by total time spent in function
print(s_tottime.getvalue())

print(
    "\n--- cProfile Results for GNNFeatureExtractor.forward (Sorted by Cumulative Time 'cumulative') ---"
)
s_cumulative = io.StringIO()
stats_cumulative = pstats.Stats(profiler, stream=s_cumulative).sort_stats("cumulative")
stats_cumulative.print_stats(40) # Print top 40 functions by cumulative time
print(s_cumulative.getvalue())