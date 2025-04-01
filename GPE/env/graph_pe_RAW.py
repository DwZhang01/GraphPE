# This file uses mixed spaces {Box,Discrete}, which is deprecated.
# This file uses mixed language {En,CH}, which is deprecated.

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, Box, Dict
import random
from copy import copy
import networkx as nx
from pettingzoo import ParallelEnv
from pettingzoo import AECEnv
import matplotlib.pyplot as plt


class GPE(ParallelEnv):
    """
    Graph Pursuit Evasion (GPE) Environment

    A multi-agent environment where pursuers try to catch evaders on a graph structure.
    Evaders try to reach a designated safe node while avoiding capture by pursuers.

    The environment uses a graph representation where:
    - Nodes represent locations agents can occupy
    - Edges represent valid movements between nodes
    - Pursuers aim to capture evaders by surrounding them
    - Evaders aim to reach a designated safe node
    - Captures occur when enough pursuers are adjacent to an evader

    Key Features:
    - Configurable graph size and connectivity
    - Multiple pursuers and evaders
    - Customizable capture mechanics
    - Support for different rendering modes
    - Built on PettingZoo's ParallelEnv

    The environment follows the gymnasium interface with:
    - Discrete observation and action spaces
    - Dictionary observations containing agent positions
    - Step-based episode progression
    - Customizable termination conditions
    """

    metadata = {
        "name": "graph_pe_v0",
        "render_modes": ["human", "rgb_array"],
    }

    def __init__(
        self,
        num_nodes=100,
        num_edges=200,
        num_pursuers=3,
        num_evaders=2,
        capture_distance=1,
        required_captors=1,
        max_steps=200,
        seed=None,
        graph=None,
        render_mode=None,
        p_act=0.8,
    ):
        """
        Initialize the Graph Pursuit Evasion environment.

        Args:
            num_nodes: Number of nodes in the graph
            num_edges: `Approximate` number of edges in the graph
            num_pursuers: Number of pursuer agents
            num_evaders: Number of evader agents
            capture_distance: Distance required for capture (1 means adjacent nodes)
            required_captors: Number of pursuers required to be adjacent for a capture
            max_steps: Maximum number of steps before the episode ends
            seed: Random seed
            graph: Predefined graph (networkx graph object), if None a random graph is generated
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        super().__init__()
        # Set random seed if provided
        self.np_random = np.random.RandomState(seed)

        self.p_act = p_act
        # Environment parameters
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.num_pursuers = num_pursuers
        self.num_evaders = num_evaders
        self.capture_distance = capture_distance
        self.required_captors = required_captors
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Create agents
        self.pursuers = [f"pursuer_{i}" for i in range(self.num_pursuers)]
        self.evaders = [f"evader_{i}" for i in range(self.num_evaders)]
        self.possible_agents = self.pursuers + self.evaders

        # Graph structure
        self.custom_graph = graph

        # Initialize spaces
        self._initialize_spaces()

        # State variables that will be initialized in reset()
        self.graph = None
        self.safe_node = None
        self.agent_positions = {}
        self.timestep = 0
        self.agents = []
        self.rewards = None
        self.terminations = None
        self.truncations = None
        self.infos = None
        self.captured_evaders = set()

    def _initialize_spaces(self):
        """Initialize action and observation spaces for all agents."""
        # Action space
        self.action_spaces = {
            agent: Discrete(self.num_nodes) for agent in self.possible_agents
        }

        # Observation space
        observation_spaces = {}
        for agent in self.possible_agents:
            # --- Start defining the base observation dict ---
            obs_dict = {
                "position": Discrete(self.num_nodes),
                "pursuers": Box(
                    low=0,
                    high=self.num_nodes - 1,
                    shape=(self.num_pursuers,),
                    dtype=np.int32,
                ),
                "evaders": Box(
                    low=-1,
                    high=self.num_nodes - 1,
                    shape=(self.num_evaders,),
                    dtype=np.int32,
                ),
                "adjacency": Box(
                    low=0,
                    high=1,
                    shape=(self.num_nodes,),
                    dtype=np.int8,
                ),
                # --- Add the action mask definition ---
                "action_mask": Box(
                    low=0, high=1, shape=(self.num_nodes,), dtype=np.int8
                ),  # Key addition!
            }
            # --- Add evader-specific info if needed ---
            if agent.startswith("evader"):
                obs_dict["safe_node"] = Discrete(self.num_nodes)

            # --- Finalize the Dict space for this agent ---
            observation_spaces[agent] = Dict(obs_dict)

        self.observation_spaces = observation_spaces

    def _generate_graph(self):
        """Generate a random graph or use the provided one."""
        if self.custom_graph is not None:
            return copy(self.custom_graph)

        # Generate a random graph with approximately num_edges edges
        # Using Erdős-Rényi random graph model
        # The probability of connecting two nodes is p = 2 * num_edges / (num_nodes * (num_nodes - 1))
        p = 2 * self.num_edges / (self.num_nodes * (self.num_nodes - 1))
        graph = nx.gnp_random_graph(
            self.num_nodes, p, seed=self.np_random.randint(10000)
        )

        # Ensure the graph is connected
        if not nx.is_connected(graph):
            # Get the largest connected component
            largest_cc = max(nx.connected_components(graph), key=len)
            graph = graph.subgraph(largest_cc).copy()  # check !!!!!!!!!!

            # Add some random edges to ensure connectivity
            nodes = list(graph.nodes())
            components = list(nx.connected_components(graph))

            while len(components) > 1:
                # Connect two random nodes from different components
                comp1 = random.choice(components)
                components.remove(comp1)
                comp2 = random.choice(components)
                components.remove(comp2)

                node1 = random.choice(list(comp1))
                node2 = random.choice(list(comp2))
                graph.add_edge(node1, node2)

                # Recalculate components
                components = list(nx.connected_components(graph))

        return graph

    def reset(self, seed=None, options=None):
        """Resets the environment and returns initial observations."""

        # Reset random seed if provided
        if seed is not None:
            self.np_random = np.random.RandomState(seed)

        # Generate or use the provided graph
        self.graph = self._generate_graph()

        # Reset episode variables
        self.timestep = 0
        self.agents = self.possible_agents.copy()
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.captured_evaders = set()

        # Select a random safe node
        self.safe_node = self.np_random.choice(list(self.graph.nodes()))

        # Place agents randomly on the graph
        # Ensure pursuers and evaders start at different nodes
        all_nodes = list(self.graph.nodes())
        self.agent_positions = {}

        # Place pursuer agents
        pursuer_nodes = self.np_random.choice(
            all_nodes, size=self.num_pursuers, replace=False
        )
        for i, agent in enumerate(self.pursuers):
            self.agent_positions[agent] = int(pursuer_nodes[i])

        # Place evader agents on different nodes than pursuers
        remaining_nodes = list(set(all_nodes) - set(pursuer_nodes) - {self.safe_node})
        if len(remaining_nodes) < self.num_evaders:
            # Fall back to allowing overlap if there aren't enough nodes
            # Ensure safe node is not used
            remaining_nodes = list(set(all_nodes) - {self.safe_node})

        evader_nodes = self.np_random.choice(
            remaining_nodes,
            size=min(self.num_evaders, len(remaining_nodes)),
            replace=False,
        )
        for i, agent in enumerate(self.evaders):
            if i < len(evader_nodes):
                self.agent_positions[agent] = int(evader_nodes[i])
            else:
                # If there are no other nodes, place randomly
                self.agent_positions[agent] = int(
                    self.np_random.choice(list(set(all_nodes) - {self.safe_node}))
                )

        # Generate observations
        # Structure: {agent: observation} where observation is a dictionary
        observations = {agent: self._get_observation(agent) for agent in self.agents}

        return observations, self.infos

    def _get_observation(self, agent):
        """为智能体生成观测信息，包含动作掩码"""
        position = self.agent_positions[agent]

        # --- 获取邻居信息 ---
        # neighbors 列表在后面计算邻接矩阵和动作掩码时都会用到
        neighbors = list(self.graph.neighbors(position))

        # --- 计算邻接矩阵 (用于观测) ---
        # 直接计算一次，使用正确的 dtype (np.int8)
        adjacency_obs = np.zeros(self.num_nodes, dtype=np.int8)
        adjacency_obs[neighbors] = 1  # 将所有邻居位置标记为 1

        # --- 获取其他智能体位置 ---
        pursuer_positions = np.array(
            [self.agent_positions[p] for p in self.pursuers], dtype=np.int32
        )
        evader_positions = np.array(
            [
                self.agent_positions[e] if e not in self.captured_evaders else -1
                for e in self.evaders
            ],
            dtype=np.int32,
        )

        # --- 计算动作掩码 ---
        # 有效动作是停留在当前位置或移动到邻居节点
        action_mask = np.zeros(self.num_nodes, dtype=np.int8)
        valid_action_indices = [position] + neighbors  # 当前位置 + 邻居节点索引
        action_mask[valid_action_indices] = 1  # 将有效动作的索引标记为 1

        # --- 创建观测字典 ---
        # 现在 action_mask 已经定义好了
        observation = {
            "position": position,
            "pursuers": pursuer_positions,
            "evaders": evader_positions,
            "adjacency": adjacency_obs,  # 使用之前计算好的邻接矩阵
            "action_mask": action_mask,  # 添加计算好的动作掩码
        }

        # --- 为逃避者添加安全节点信息 ---
        if agent.startswith("evader"):
            observation["safe_node"] = self.safe_node

        # --- 返回最终观测 ---
        return observation

    def step(self, actions):
        """为所有智能体执行动作并返回新的观测"""

        # 1. 验证动作输入: 确保为所有活动的智能体提供了动作
        #    使用 set() 进行比较，更加健壮
        active_agents_set = set(self.agents)
        received_actions_set = set(actions.keys())
        if not received_actions_set == active_agents_set:
            # 提供更清晰的错误信息
            missing = active_agents_set - received_actions_set
            extra = received_actions_set - active_agents_set
            error_msg = "Actions mismatch. "
            if missing:
                error_msg += f"Missing actions for: {missing}. "
            if extra:
                error_msg += f"Received unexpected actions for: {extra}. "
            error_msg += f"Expected actions for: {list(active_agents_set)}."
            raise ValueError(error_msg)

        # 2. 初始化返回值字典
        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # 3. 处理每个智能体的动作
        next_positions = self.agent_positions.copy()  # 先计算所有智能体的下一步目标位置
        for agent, action in actions.items():
            # 跳过非活动智能体 (虽然理论上不应出现在actions中，但增加一层保险)
            if agent not in self.agents:
                continue

            current_position = self.agent_positions[agent]

            # 3a. 应用随机性: 有 p_act 概率强制停留
            if self.np_random.random() < self.p_act:  # 使用 self.np_random 保证可复现性
                effective_action = current_position  # 强制停留
            else:
                effective_action = action  # 使用智能体选择的动作

            # 3b. 检查动作有效性并确定下一位置
            #    有效的动作是停留在原地或移动到邻居节点
            is_valid_move = (
                effective_action == current_position
                or effective_action in self.graph.neighbors(current_position)
            )

            if is_valid_move:
                next_positions[agent] = effective_action  # 暂存目标位置
            else:
                # 无效动作，记录信息，位置保持不变 (next_positions[agent] 未被修改)
                self.infos[agent]["invalid_action"] = True
                # 无需显式设置 next_positions[agent] = current_position，因为它本来就没变

        # 4. 更新所有智能体的实际位置
        #    将计算好的下一步位置应用到环境中
        self.agent_positions = next_positions

        # 5. 检查游戏状态改变 (捕获、安全到达)
        self._check_captures()
        self._check_safe_arrivals()

        # 6. 检查游戏整体是否结束
        #    直接调用，该方法会更新 self.terminations
        self._check_termination()  # 不再需要接收返回值

        # 7. 增加时间步
        self.timestep += 1

        # 8. 检查回合截断 (最大步数)
        if self.timestep >= self.max_steps:
            for agent in self.agents:
                # 只有当智能体尚未终止时才设置截断
                # (按照Gymnasium最新规范，termination优先于truncation)
                if not self.terminations[agent]:
                    self.truncations[agent] = True

        # 9. 更新活动智能体列表 (移除已终止或截断的智能体)
        # --- 取消注释这部分 ---
        active_agents_next_step = []
        # 遍历此步骤开始时的 agents 列表
        for agent in self.agents:
            # 如果智能体在这一步既没有终止也没有截断，则它在下一步仍然活动
            if not self.terminations[agent] and not self.truncations[agent]:
                active_agents_next_step.append(agent)
        # 更新 self.agents 列表以反映下一步的活动智能体
        self.agents = active_agents_next_step
        # 注意：如果修改了 self.agents, 后续生成观测时要用新的 self.agents

        # 10. 生成新的观测 (为所有初始传入动作的智能体生成，即使它们可能刚终止/截断)
        #     这是 ParallelEnv 的标准做法，总是为请求动作的智能体返回结果
        observations = {agent: self._get_observation(agent) for agent in actions.keys()}

        # 11. 返回结果
        return (
            observations,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def _check_captures(self):
        """Check if any evaders have been captured."""
        # Check if any evaders have been captured by pursuers
        # A capture occurs when an evader is at the same node as a pursuer
        # or when a pursuer moves to a node adjacent to the evader
        for evader in self.evaders:
            if evader in self.captured_evaders:
                continue

            evader_pos = self.agent_positions[evader]
            evader_neighbors = list(self.graph.neighbors(evader_pos))

            # Count how many pursuers are adjacent to this evader
            adjacent_pursuers = 0
            for pursuer in self.pursuers:
                pursuer_pos = self.agent_positions[pursuer]

                # Check if pursuer is adjacent to evader (or at same position)
                if pursuer_pos == evader_pos or pursuer_pos in evader_neighbors:
                    adjacent_pursuers += 1

            # Check if capture conditions are met,
            if adjacent_pursuers >= self.required_captors:
                # Evader is captured
                self.captured_evaders.add(evader)

                # Award positive reward to pursuers
                for pursuer in self.pursuers:
                    self.rewards[pursuer] += 10.0

                # Penalty for the captured evader
                self.rewards[evader] -= 10.0
                self.terminations[evader] = True

    def _check_safe_arrivals(self):
        """Check if any evaders have reached the safe node."""
        # Can be changed to different reward settings
        for evader in self.evaders:
            if evader in self.captured_evaders:
                continue

            # Check if evader reached safe node
            if self.agent_positions[evader] == self.safe_node:
                # Award positive reward to evader
                self.rewards[evader] += 20.0
                self.terminations[evader] = True

                # Penalty for pursuers
                for pursuer in self.pursuers:
                    self.rewards[pursuer] -= 5.0

    def _check_termination(self):
        """Check if the game is over."""
        # Game is over if all evaders are either captured or reached safe node
        active_evaders = set(self.evaders) - self.captured_evaders  # not captured
        evaders_at_safe_node = sum(
            1 for e in active_evaders if self.agent_positions[e] == self.safe_node
        )

        if len(active_evaders) == 0 or evaders_at_safe_node == len(active_evaders):
            # All evaders have been captured or reached safety
            for agent in self.agents:
                self.terminations[agent] = True
            return True

        return False

    def observation_space(self, agent):
        """Return the observation space for an agent."""
        return self.observation_spaces[agent]

    def action_space(self, agent):
        """Return the action space for an agent."""
        return self.action_spaces[agent]

    def render(self):
        """Render the environment.
        在同一个图形窗口中显示当前环境状态，通过循环实现动态

        """
        if self.render_mode is None:
            return

        # 打印基本信息
        print(f"Timestep: {self.timestep}")
        print(f"Safe node: {self.safe_node}")
        print("Pursuer positions:", {p: self.agent_positions[p] for p in self.pursuers})
        print(
            "Evader positions:",
            {
                e: self.agent_positions[e]
                for e in self.evaders
                if e not in self.captured_evaders
            },
        )
        print("Captured evaders:", self.captured_evaders)

        if self.render_mode == "human":
            # 创建并保持同一个图形窗口
            if not hasattr(self, "fig"):
                self.fig = plt.figure(figsize=(12, 8))
            else:
                plt.figure(self.fig.number)  # 使用同一个窗口

            # 清除当前图形内容，但保持窗口
            plt.clf()

            # 使用spring_layout布局，增加节点间距，k越大，节点间距越大
            pos = nx.spring_layout(self.graph, k=1, iterations=50)

            # 绘制图结构
            # 节点
            nx.draw_networkx_nodes(
                self.graph, pos, node_color="lightgray", node_size=500
            )
            # 边
            nx.draw_networkx_edges(self.graph, pos, edge_color="gray", width=1)
            # 节点标签
            nx.draw_networkx_labels(self.graph, pos, font_size=8)

            # 绘制智能体
            # 追捕者
            for pursuer in self.pursuers:
                position = self.agent_positions[pursuer]
                plt.plot(
                    pos[position][0],
                    pos[position][1],
                    "ro",
                    markersize=15,
                    label=pursuer,
                )
                # 添加追捕者标签
                plt.annotate(
                    pursuer,
                    (pos[position][0], pos[position][1]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    color="red",
                    fontsize=8,
                )

            # 逃避者
            for evader in self.evaders:
                if evader not in self.captured_evaders:
                    position = self.agent_positions[evader]
                    plt.plot(
                        pos[position][0],
                        pos[position][1],
                        "bo",
                        markersize=15,
                        label=evader,
                    )
                    # 添加逃避者标签
                    plt.annotate(
                        evader,
                        (pos[position][0], pos[position][1]),
                        xytext=(10, -10),
                        textcoords="offset points",
                        color="blue",
                        fontsize=8,
                    )

            # 安全节点
            plt.plot(
                pos[self.safe_node][0],
                pos[self.safe_node][1],
                "go",
                markersize=20,
                label="Safe Node",
            )
            plt.annotate(
                "SAFE",
                (pos[self.safe_node][0], pos[self.safe_node][1]),
                xytext=(0, 20),
                textcoords="offset points",
                color="green",
                fontsize=10,
                ha="center",
            )

            # 添加标题，显示当前时间步
            plt.title(f"Timestep: {self.timestep}", fontsize=12)

            # 添加图例
            plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

            # 调整布局，确保图例不会被截断
            plt.tight_layout()

            # 显示并暂停
            plt.pause(0.5)  # 暂停0.5秒
            # 避免 plt.close()，让窗口保持打开
