import random
import networkx as nx
import numpy as np
import supersuit as ss
from GPE.env.graph_pe import GPE
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

env = GPE(
    num_nodes=20,
    num_edges=40,
    num_pursuers=2,
    num_evaders=1,
    capture_distance=1,
    required_captors=1,
    max_steps=50,
    seed=42,
    render_mode="rgb_array",
)

env = ss.flatten_v0(env)  # 关键：将 Dict 展平成 Box
env = ss.pettingzoo_env_to_vec_env_v1(env)
env = ss.concat_vec_envs_v1(
    env, num_vec_envs=1, num_cpus=1, base_class="stable_baselines3"
)

observations, infos = env.reset(seed=42)
model = PPO(
    MlpPolicy,
    env,
    verbose=3,
    gamma=0.95,
    n_steps=256,
    ent_coef=0.0905168,
    learning_rate=0.00062211,
    vf_coef=0.042202,
    max_grad_norm=0.9,
    gae_lambda=0.99,
    n_epochs=5,
    clip_range=0.3,
    batch_size=256,
)

model.learn(total_timesteps=2000000)
model.save("policy")
