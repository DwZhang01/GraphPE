import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from typing import Dict
from policy.ACRGPolicy import ActorCriticGNN_RNN_Policy

class PPOAgent:
    """
    PPO智能体，封装了策略网络和优化逻辑。
    """
    def __init__(self, policy: ActorCriticGNN_RNN_Policy, lr: float = 3e-4, clip_coef: float = 0.2, vf_coef: float = 0.5, ent_coef: float = 0.01):
        self.policy = policy
        self.optimizer = optim.Adam(policy.parameters(), lr=lr, eps=1e-5)
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef

    def get_action_and_value(self, obs: Dict[str, torch.Tensor], edge_index: torch.Tensor, rnn_state: torch.Tensor, action: torch.Tensor = None):
        """
        根据观测获取动作、动作概率的对数、熵、价值以及下一个RNN状态。
        """
        # 修正: 捕获并返回 next_rnn_state
        logits, value, next_rnn_state = self.policy(obs, edge_index, rnn_state)
        probs = Categorical(logits=logits)
        
        if action is None:
            action = probs.sample()
            
        return action, probs.log_prob(action), probs.entropy(), value, next_rnn_state

    # train 函数保持不变
    def train(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values, edge_index, b_rnn_states):
        device = self.policy.parameters().__next__().device
        b_obs = {k: v.to(device) for k, v in b_obs.items()}
        b_logprobs = b_logprobs.to(device)
        b_actions = b_actions.to(device)
        b_advantages = b_advantages.to(device)
        b_returns = b_returns.to(device)
        b_values = b_values.to(device)
        edge_index = edge_index.to(device)
        b_rnn_states = b_rnn_states.to(device)
        
        logits, new_values, _ = self.policy(b_obs, edge_index, b_rnn_states)
        new_values = new_values.squeeze()
        
        new_probs = Categorical(logits=logits)
        new_logprobs = new_probs.log_prob(b_actions)
        entropy = new_probs.entropy()
        
        logratio = new_logprobs - b_logprobs
        ratio = logratio.exp()
        
        pg_loss1 = -b_advantages * ratio
        pg_loss2 = -b_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
        pg_loss = torch.max(pg_loss1, pg_loss2).mean()

        v_loss = 0.5 * ((new_values - b_returns) ** 2).mean()
        entropy_loss = entropy.mean()
        loss = pg_loss - self.ent_coef * entropy_loss + self.vf_coef * v_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        return pg_loss.item(), v_loss.item(), entropy_loss.item()