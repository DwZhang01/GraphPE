import numpy
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete
import random
from copy import copy
from pettingzoo import ParallelEnv

import networkx as nx


class GPE(ParallelEnv):

    # 图结构和网格结构的本质区别在于，图结构的节点之间的连接关系是任意的，而网格结构的节点之间的连接关系是固定的。能否添加传送门?
    # 图结构上的动作怎么定义？随机游走？
    # 图的规模
    # 稀疏回报

    metadata = {
        "name": "graph_pe_v0",
    }
    num_pursuit_agents = 0
    num_evade_agents = 0

    def __init__(self):

        self.action_spaces = {agent: Discrete(4) for agent in self.agents}
        self.observation_spaces = {
            agent: MultiDiscrete([4, 4]) for agent in self.agents
        }
        self.escape_y = None
        self.escape_x = None
        self.guard_y = None
        self.guard_x = None
        self.prisoner_y = None
        self.prisoner_x = None
        self.timestep = None
        self.possible_agents = ["prisoner", "guard"]

    def reset(self, seed=None, options=None):
        # 设置随机数种子（如果提供了）
        super().reset(seed=seed)

        # 假设的图结构：一个简单的环形图
        num_nodes = 10  # 假设有10个节点
        self.graph = nx.cycle_graph(num_nodes)

        # 智能体数量（你可以根据需要调整）
        self.num_pursuit_agents = 2
        self.num_evade_agents = 1

        # 创建智能体列表
        self.pursuit_agents = [f"pursuit_{i}" for i in range(self.num_pursuit_agents)]
        self.evade_agents = [f"evade_{i}" for i in range(self.num_evade_agents)]
        self.agents = self.pursuit_agents + self.evade_agents

        # 初始化智能体位置（随机放置在图节点上）
        self.agent_positions = {}
        for agent in self.agents:
            self.agent_positions[agent] = random.choice(list(self.graph.nodes))

        # 初始化观察值
        observations = {agent: self._get_observation(agent) for agent in self.agents}
        # 初始化info
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def _get_observation(self, agent):
        # 在这里定义如何根据智能体位置生成观察值
        # 一个简单的例子：观察值是智能体所在节点和邻居节点
        # 注意：你需要根据你的 MultiDiscrete 空间定义来调整观察值的格式
        position = self.agent_positions[agent]
        neighbors = list(self.graph.neighbors(position))
        # 假设观察值是两个维度的 MultiDiscrete([节点数, 节点数])
        # 第一个维度表示自身位置, 第二个维度表示邻居位置(简化为一个)
        observation = [position, neighbors[0] if neighbors else 0]  # 简化处理
        return observation

    def step(self, actions):
        pass

    def render(self):
        pass

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]
