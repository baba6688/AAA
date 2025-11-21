"""
强化学习模块

实现多种强化学习算法：
- Q-Learning
- Deep Q-Network (DQN)
- Policy Gradient
- Actor-Critic
- PPO (Proximal Policy Optimization)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
from collections import deque, namedtuple
import random
import logging
from datetime import datetime
from .StrategyLearner import StrategyType, BaseStrategy, LearningContext, StrategyPerformance

logger = logging.getLogger(__name__)

class QNetwork(nn.Module):
    """Q网络实现"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    """策略网络实现"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mean = nn.Linear(hidden_dim, action_dim)
        self.fc_std = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = self.fc_mean(x)
        std = F.softplus(self.fc_std(x)) + 1e-6
        
        return mean, std

class ValueNetwork(nn.Module):
    """价值网络实现"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNLearner(BaseStrategy):
    """深度Q网络学习器"""
    
    def __init__(self, strategy_id: str, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, memory_size: int = 10000):
        super().__init__(strategy_id, StrategyType.REINFORCEMENT)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # 神经网络
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 经验回放
        self.memory = deque(maxlen=memory_size)
        self.Experience = namedtuple('Experience', 
                                   ['state', 'action', 'reward', 'next_state', 'done'])
        
        # 同步目标网络
        self.update_target_network()
        
    def learn(self, context: LearningContext) -> Dict[str, Any]:
        """DQN学习过程"""
        try:
            # 从经验中学习
            if len(self.memory) > 100:
                batch_size = 32
                experiences = random.sample(self.memory, batch_size)
                
                states = torch.FloatTensor([e.state for e in experiences])
                actions = torch.LongTensor([e.action for e in experiences])
                rewards = torch.FloatTensor([e.reward for e in experiences])
                next_states = torch.FloatTensor([e.next_state for e in experiences])
                dones = torch.BoolTensor([e.done for e in experiences])
                
                # 当前Q值
                current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
                
                # 目标Q值
                next_q_values = self.target_network(next_states).max(1)[0].detach()
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
                
                # 计算损失
                loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 更新epsilon
                if self.epsilon > self.epsilon_min:
                    self.epsilon *= self.epsilon_decay
            
            # 更新目标网络
            if random.random() < 0.01:  # 1%的概率更新目标网络
                self.update_target_network()
            
            return {
                'epsilon': self.epsilon,
                'loss': loss.item() if 'loss' in locals() else 0.0,
                'q_values': self.q_network.state_dict(),
                'learning_progress': len(self.memory)
            }
            
        except Exception as e:
            logger.error(f"DQN学习出错: {e}")
            return {'error': str(e)}
    
    def predict(self, state: Dict[str, Any]) -> Any:
        """DQN预测"""
        try:
            state_tensor = torch.FloatTensor(list(state.values())).unsqueeze(0)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                
            # epsilon-greedy策略
            if random.random() < self.epsilon:
                action = random.randint(0, self.action_dim - 1)
                confidence = 0.0
            else:
                action = q_values.argmax().item()
                confidence = torch.softmax(q_values, dim=1).max().item()
            
            return {
                'action': action,
                'confidence': confidence,
                'q_values': q_values.squeeze().tolist(),
                'epsilon': self.epsilon
            }
            
        except Exception as e:
            logger.error(f"DQN预测出错: {e}")
            return {'action': 0, 'confidence': 0.0, 'error': str(e)}
    
    def update_performance(self, performance: StrategyPerformance):
        """更新性能指标"""
        try:
            self.state.performance_metrics.update({
                'return_rate': performance.return_rate,
                'sharpe_ratio': performance.sharpe_ratio,
                'max_drawdown': performance.max_drawdown,
                'win_rate': performance.win_rate
            })
            
            # 基于性能调整epsilon
            if performance.return_rate > 0:
                self.epsilon = max(self.epsilon_min, self.epsilon * 0.99)
            else:
                self.epsilon = min(0.3, self.epsilon * 1.01)
            
            self.state.usage_count += 1
            self.state.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"更新性能指标出错: {e}")
    
    def add_experience(self, state: Dict[str, Any], action: int, reward: float, 
                      next_state: Dict[str, Any], done: bool):
        """添加经验"""
        experience = self.Experience(
            state=list(state.values()),
            action=action,
            reward=reward,
            next_state=list(next_state.values()),
            done=done
        )
        self.memory.append(experience)
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

class PolicyGradientLearner(BaseStrategy):
    """策略梯度学习器"""
    
    def __init__(self, strategy_id: str, state_dim: int, action_dim: int,
                 learning_rate: float = 0.001, gamma: float = 0.99):
        super().__init__(strategy_id, StrategyType.REINFORCEMENT)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        self.trajectory = []
        
    def learn(self, context: LearningContext) -> Dict[str, Any]:
        """策略梯度学习"""
        try:
            if not self.trajectory:
                return {'message': '没有轨迹数据'}
            
            # 计算回报
            rewards = []
            G = 0
            for r in reversed(self.trajectory):
                G = r + self.gamma * G
                rewards.insert(0, G)
            
            # 标准化回报
            rewards = torch.FloatTensor(rewards)
            if rewards.std() > 0:
                rewards = (rewards - rewards.mean()) / rewards.std()
            
            # 计算策略损失
            states = torch.FloatTensor([t['state'] for t in self.trajectory])
            actions = torch.FloatTensor([t['action'] for t in self.trajectory])
            
            mean, std = self.policy_network(states)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=1)
            
            policy_loss = -(log_probs * rewards).mean()
            
            # 反向传播
            self.optimizer.zero_grad()
            policy_loss.backward()
            self.optimizer.step()
            
            # 清空轨迹
            self.trajectory = []
            
            return {
                'policy_loss': policy_loss.item(),
                'avg_reward': rewards.mean().item(),
                'trajectory_length': len(rewards)
            }
            
        except Exception as e:
            logger.error(f"策略梯度学习出错: {e}")
            return {'error': str(e)}
    
    def predict(self, state: Dict[str, Any]) -> Any:
        """策略梯度预测"""
        try:
            state_tensor = torch.FloatTensor(list(state.values())).unsqueeze(0)
            
            with torch.no_grad():
                mean, std = self.policy_network(state_tensor)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                confidence = torch.softmax(mean, dim=1).max().item()
            
            return {
                'action': action.item(),
                'confidence': confidence,
                'mean': mean.squeeze().tolist(),
                'std': std.squeeze().tolist()
            }
            
        except Exception as e:
            logger.error(f"策略梯度预测出错: {e}")
            return {'action': 0, 'confidence': 0.0, 'error': str(e)}
    
    def update_performance(self, performance: StrategyPerformance):
        """更新性能指标"""
        try:
            self.state.performance_metrics.update({
                'return_rate': performance.return_rate,
                'sharpe_ratio': performance.sharpe_ratio,
                'max_drawdown': performance.max_drawdown,
                'win_rate': performance.win_rate
            })
            
            self.state.usage_count += 1
            self.state.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"更新性能指标出错: {e}")
    
    def add_trajectory(self, state: Dict[str, Any], action: float, reward: float):
        """添加轨迹"""
        self.trajectory.append({
            'state': list(state.values()),
            'action': action,
            'reward': reward
        })

class ActorCriticLearner(BaseStrategy):
    """Actor-Critic学习器"""
    
    def __init__(self, strategy_id: str, state_dim: int, action_dim: int,
                 learning_rate: float = 0.001, gamma: float = 0.99):
        super().__init__(strategy_id, StrategyType.REINFORCEMENT)
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = ValueNetwork(state_dim)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        self.trajectory = []
        
    def learn(self, context: LearningContext) -> Dict[str, Any]:
        """Actor-Critic学习"""
        try:
            if not self.trajectory:
                return {'message': '没有轨迹数据'}
            
            # 计算回报和优势
            rewards = []
            G = 0
            for r in reversed(self.trajectory):
                G = r + self.gamma * G
                rewards.insert(0, G)
            
            rewards = torch.FloatTensor(rewards)
            states = torch.FloatTensor([t['state'] for t in self.trajectory])
            actions = torch.FloatTensor([t['action'] for t in self.trajectory])
            
            # 计算价值函数和优势
            with torch.no_grad():
                values = self.critic(states).squeeze()
                next_value = 0
                advantages = []
                for i in reversed(range(len(rewards))):
                    if i == len(rewards) - 1:
                        next_value = 0
                    delta = rewards[i] + self.gamma * next_value - values[i]
                    advantages.insert(0, delta)
                    next_value = values[i]
            
            advantages = torch.FloatTensor(advantages)
            if advantages.std() > 0:
                advantages = (advantages - advantages.mean()) / advantages.std()
            
            # 更新Critic
            values = self.critic(states).squeeze()
            critic_loss = F.mse_loss(values, rewards)
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            # 更新Actor
            mean, std = self.actor(states)
            dist = torch.distributions.Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(dim=1)
            
            actor_loss = -(log_probs * advantages).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 清空轨迹
            self.trajectory = []
            
            return {
                'actor_loss': actor_loss.item(),
                'critic_loss': critic_loss.item(),
                'avg_advantage': advantages.mean().item(),
                'trajectory_length': len(rewards)
            }
            
        except Exception as e:
            logger.error(f"Actor-Critic学习出错: {e}")
            return {'error': str(e)}
    
    def predict(self, state: Dict[str, Any]) -> Any:
        """Actor-Critic预测"""
        try:
            state_tensor = torch.FloatTensor(list(state.values())).unsqueeze(0)
            
            with torch.no_grad():
                mean, std = self.actor(state_tensor)
                value = self.critic(state_tensor).item()
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                confidence = torch.softmax(mean, dim=1).max().item()
            
            return {
                'action': action.item(),
                'confidence': confidence,
                'value': value,
                'mean': mean.squeeze().tolist(),
                'std': std.squeeze().tolist()
            }
            
        except Exception as e:
            logger.error(f"Actor-Critic预测出错: {e}")
            return {'action': 0, 'confidence': 0.0, 'error': str(e)}
    
    def update_performance(self, performance: StrategyPerformance):
        """更新性能指标"""
        try:
            self.state.performance_metrics.update({
                'return_rate': performance.return_rate,
                'sharpe_ratio': performance.sharpe_ratio,
                'max_drawdown': performance.max_drawdown,
                'win_rate': performance.win_rate
            })
            
            self.state.usage_count += 1
            self.state.last_updated = datetime.now()
            
        except Exception as e:
            logger.error(f"更新性能指标出错: {e}")
    
    def add_trajectory(self, state: Dict[str, Any], action: float, reward: float):
        """添加轨迹"""
        self.trajectory.append({
            'state': list(state.values()),
            'action': action,
            'reward': reward
        })

# 工厂函数
def create_rl_learner(algorithm: str, strategy_id: str, state_dim: int, 
                     action_dim: int, **kwargs) -> BaseStrategy:
    """创建强化学习器"""
    algorithms = {
        'dqn': DQNLearner,
        'policy_gradient': PolicyGradientLearner,
        'actor_critic': ActorCriticLearner
    }
    
    if algorithm not in algorithms:
        raise ValueError(f"不支持的算法: {algorithm}")
    
    return algorithms[algorithm](strategy_id, state_dim, action_dim, **kwargs)