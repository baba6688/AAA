"""
强化学习算法库
================

这是一个完整的强化学习算法库，实现了多种主流的强化学习算法：
- Q-Learning 和 Deep Q-Network (DQN)
- Policy Gradient 方法
- Actor-Critic 算法
- Proximal Policy Optimization (PPO)
- Trust Region Policy Optimization (TRPO)
- Deep Deterministic Policy Gradient (DDPG)
- Multi-Agent Reinforcement Learning

包含完整的训练、测试和评估功能。
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
from typing import Dict, List, Tuple, Optional, Any, Union
import random
import copy
import gym
from abc import ABC, abstractmethod


# ==================== 基础环境接口 ====================

class RLEnvironment:
    """强化学习环境接口基类"""
    
    def __init__(self, state_space: gym.Space, action_space: gym.Space):
        """
        初始化环境
        
        Args:
            state_space: 状态空间
            action_space: 动作空间
        """
        self.state_space = state_space
        self.action_space = action_space
        
    def reset(self) -> np.ndarray:
        """重置环境"""
        raise NotImplementedError
        
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        raise NotImplementedError
        
    def render(self) -> None:
        """渲染环境"""
        pass


class GymEnvironment(RLEnvironment):
    """基于OpenAI Gym的环境封装"""
    
    def __init__(self, env_name: str):
        """
        初始化Gym环境
        
        Args:
            env_name: 环境名称
        """
        self.env = gym.make(env_name)
        super().__init__(self.env.observation_space, self.env.action_space)
        self.env_name = env_name
        
    def reset(self) -> np.ndarray:
        """重置环境"""
        return self.env.reset()
        
    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行动作"""
        return self.env.step(action)
        
    def render(self) -> None:
        """渲染环境"""
        self.env.render()
        
    def close(self) -> None:
        """关闭环境"""
        self.env.close()


# ==================== 经验回放缓冲区 ====================

class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        """
        初始化经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        
    def push(self, state: np.ndarray, action: Any, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        添加经验到缓冲区
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        
    def sample(self, batch_size: int) -> List[Tuple]:
        """
        从缓冲区采样
        
        Args:
            batch_size: 批次大小
            
        Returns:
            经验样本列表
        """
        return random.sample(self.buffer, batch_size)
        
    def __len__(self) -> int:
        """返回缓冲区长度"""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """优先级经验回放缓冲区"""
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        初始化优先级经验回放缓冲区
        
        Args:
            capacity: 缓冲区容量
            alpha: 优先级参数
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.ones(capacity)
        self.position = 0
        
    def push(self, state: np.ndarray, action: Any, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        添加经验到缓冲区
        
        Args:
            state: 当前状态
            action: 动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        experience = (state, action, reward, next_state, done)
        
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = max(self.priorities)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int, beta: float = 0.4) -> List[Tuple]:
        """
        从缓冲区采样
        
        Args:
            batch_size: 批次大小
            beta: 重要性采样参数
            
        Returns:
            经验样本列表和重要性权重
        """
        if len(self.buffer) == self.buffer.__len__():
            priorities = self.priorities
        else:
            priorities = self.priorities[:len(self.buffer)]
            
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[i] for i in indices]
        
        weights = (len(self.buffer) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        
        return samples, weights, indices
        
    def update_priorities(self, indices: List[int], td_errors: np.ndarray) -> None:
        """
        更新优先级
        
        Args:
            indices: 样本索引
            td_errors: TD误差
        """
        for i, td_error in zip(indices, td_errors):
            self.priorities[i] = td_error + 1e-6


# ==================== 神经网络模型 ====================

class QNetwork(nn.Module):
    """Q值网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        初始化Q值网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 输入状态
            
        Returns:
            Q值
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DuelingQNetwork(nn.Module):
    """Dueling Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        初始化Dueling Q网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(DuelingQNetwork, self).__init__()
        
        # 共享特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 状态价值函数
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # 优势函数
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 输入状态
            
        Returns:
            Q值
        """
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Q = V + A - mean(A)
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values


class PolicyNetwork(nn.Module):
    """策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        初始化策略网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(PolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 输入状态
            
        Returns:
            动作概率分布
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)
        
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作
        
        Args:
            state: 输入状态
            
        Returns:
            动作和对应的log概率
        """
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob


class ContinuousPolicyNetwork(nn.Module):
    """连续动作策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        初始化连续动作策略网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(ContinuousPolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # 均值网络
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        
        # 标准差网络
        self.std_layer = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 输入状态
            
        Returns:
            动作均值和标准差
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = torch.tanh(self.mean_layer(x))
        std = F.softplus(self.std_layer(x))
        
        return mean, std
        
    def sample(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作
        
        Args:
            state: 输入状态
            
        Returns:
            动作和对应的log概率
        """
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob


class ValueNetwork(nn.Module):
    """价值网络"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        """
        初始化价值网络
        
        Args:
            state_dim: 状态维度
            hidden_dim: 隐藏层维度
        """
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 输入状态
            
        Returns:
            状态价值
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ActorNetwork(nn.Module):
    """Actor网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        初始化Actor网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(ActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 输入状态
            
        Returns:
            动作
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))


class CriticNetwork(nn.Module):
    """Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """
        初始化Critic网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(CriticNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            state: 输入状态
            action: 输入动作
            
        Returns:
            Q值
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# ==================== Q-Learning算法 ====================

class QLearning:
    """Q-Learning算法"""
    
    def __init__(self, state_space: gym.Space, action_space: gym.Space, 
                 learning_rate: float = 0.1, discount_factor: float = 0.95,
                 epsilon: float = 0.1, epsilon_decay: float = 0.995):
        """
        初始化Q-Learning
        
        Args:
            state_space: 状态空间
            action_space: 动作空间
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
        """
        self.state_space = state_space
        self.action_space = action_space
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        # 初始化Q表
        if isinstance(state_space, gym.spaces.Discrete) and isinstance(action_space, gym.spaces.Discrete):
            self.q_table = np.zeros((state_space.n, action_space.n))
            self.use_table = True
        else:
            self.q_table = None
            self.use_table = False
            raise ValueError("Q-Learning只支持离散状态和动作空间")
            
    def get_action(self, state: int) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            
        Returns:
            选择的动作
        """
        if np.random.random() < self.epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.q_table[state])
            
    def update(self, state: int, action: int, reward: float, 
               next_state: int, done: bool) -> None:
        """
        更新Q值
        
        Args:
            state: 当前状态
            action: 当前动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        if done:
            target = reward
        else:
            target = reward + self.discount_factor * np.max(self.q_table[next_state])
            
        self.q_table[state, action] += self.learning_rate * (target - self.q_table[state, action])
        
    def train(self, env: RLEnvironment, episodes: int, max_steps: int = 1000) -> List[float]:
        """
        训练算法
        
        Args:
            env: 环境
            episodes: 训练轮数
            max_steps: 最大步数
            
        Returns:
            每轮奖励列表
        """
        rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            # 衰减探索率
            self.epsilon *= self.epsilon_decay
            
            rewards.append(total_reward)
            
        return rewards
        
    def test(self, env: RLEnvironment, episodes: int, max_steps: int = 1000) -> List[float]:
        """
        测试算法
        
        Args:
            env: 环境
            episodes: 测试轮数
            max_steps: 最大步数
            
        Returns:
            每轮奖励列表
        """
        rewards = []
        old_epsilon = self.epsilon
        self.epsilon = 0  # 测试时不使用探索
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            rewards.append(total_reward)
            
        self.epsilon = old_epsilon
        return rewards


# ==================== Deep Q-Network (DQN) ====================

class DQN:
    """Deep Q-Network算法"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.001, discount_factor: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, buffer_size: int = 10000,
                 batch_size: int = 32, target_update_freq: int = 100,
                 use_dueling: bool = False, use_prioritized: bool = False):
        """
        初始化DQN
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            buffer_size: 经验回放缓冲区大小
            batch_size: 批次大小
            target_update_freq: 目标网络更新频率
            use_dueling: 是否使用Dueling架构
            use_prioritized: 是否使用优先级回放
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0
        
        # 选择网络架构
        if use_dueling:
            self.q_network = DuelingQNetwork(state_dim, action_dim)
            self.target_network = DuelingQNetwork(state_dim, action_dim)
        else:
            self.q_network = QNetwork(state_dim, action_dim)
            self.target_network = QNetwork(state_dim, action_dim)
            
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # 选择经验回放缓冲区
        if use_prioritized:
            self.buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.buffer = ReplayBuffer(buffer_size)
            
        self.use_prioritized = use_prioritized
        
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            training: 是否为训练模式
            
        Returns:
            选择的动作
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
            
    def update(self, state: np.ndarray, action: int, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """
        添加经验到缓冲区
        
        Args:
            state: 当前状态
            action: 当前动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.buffer.push(state, action, reward, next_state, done)
        self.steps += 1
        
        # 更新网络
        if len(self.buffer) >= self.batch_size:
            self._train_step()
            
        # 更新目标网络
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def _train_step(self) -> None:
        """训练一步"""
        if self.use_prioritized:
            experiences, weights, indices = self.buffer.sample(self.batch_size)
            weights = torch.FloatTensor(weights)
        else:
            experiences = self.buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size)
            
        states = torch.FloatTensor([e[0] for e in experiences])
        actions = torch.LongTensor([e[1] for e in experiences]).unsqueeze(1)
        rewards = torch.FloatTensor([e[2] for e in experiences]).unsqueeze(1)
        next_states = torch.FloatTensor([e[3] for e in experiences])
        dones = torch.BoolTensor([e[4] for e in experiences]).unsqueeze(1)
        
        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, actions)
        
        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
            
        # 计算损失
        td_errors = target_q_values - current_q_values
        loss = (weights * td_errors.pow(2)).mean()
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # 更新优先级
        if self.use_prioritized:
            self.buffer.update_priorities(indices, td_errors.abs().detach().numpy())
            
    def train(self, env: RLEnvironment, episodes: int, max_steps: int = 1000) -> List[float]:
        """
        训练算法
        
        Args:
            env: 环境
            episodes: 训练轮数
            max_steps: 最大步数
            
        Returns:
            每轮奖励列表
        """
        rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            rewards.append(total_reward)
            
        return rewards
        
    def test(self, env: RLEnvironment, episodes: int, max_steps: int = 1000) -> List[float]:
        """
        测试算法
        
        Args:
            env: 环境
            episodes: 测试轮数
            max_steps: 最大步数
            
        Returns:
            每轮奖励列表
        """
        rewards = []
        old_epsilon = self.epsilon
        self.epsilon = 0  # 测试时不使用探索
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.get_action(state, training=False)
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            rewards.append(total_reward)
            
        self.epsilon = old_epsilon
        return rewards


# ==================== Policy Gradient算法 ====================

class PolicyGradient:
    """Policy Gradient算法"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.001, discount_factor: float = 0.99):
        """
        初始化Policy Gradient
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            discount_factor: 折扣因子
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        
        self.trajectory = []
        
    def get_action(self, state: np.ndarray) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            
        Returns:
            选择的动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy_network.sample(state_tensor)
        return action.item()
        
    def store_transition(self, state: np.ndarray, action: int, reward: float) -> None:
        """
        存储转移
        
        Args:
            state: 当前状态
            action: 当前动作
            reward: 奖励
        """
        self.trajectory.append((state, action, reward))
        
    def train(self) -> None:
        """训练网络"""
        if len(self.trajectory) == 0:
            return
            
        # 计算折扣奖励
        rewards = [t[2] for t in self.trajectory]
        discounted_rewards = []
        cumulative_reward = 0
        
        for reward in reversed(rewards):
            cumulative_reward = reward + self.discount_factor * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
            
        # 标准化奖励
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-8)
        
        # 计算策略梯度
        states = torch.FloatTensor([t[0] for t in self.trajectory])
        actions = torch.LongTensor([t[1] for t in self.trajectory])
        
        log_probs = []
        for i in range(len(self.trajectory)):
            state = states[i].unsqueeze(0)
            action = actions[i]
            _, log_prob = self.policy_network.sample(state)
            log_probs.append(log_prob)
            
        policy_loss = -torch.stack(log_probs) * discounted_rewards
        policy_loss = policy_loss.sum()
        
        # 更新网络
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # 清空轨迹
        self.trajectory = []
        
    def train_episode(self, env: RLEnvironment, max_steps: int = 1000) -> float:
        """
        训练一轮
        
        Args:
            env: 环境
            max_steps: 最大步数
            
        Returns:
            总奖励
        """
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            action = self.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            self.store_transition(state, action, reward)
            
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        self.train()
        return total_reward
        
    def test(self, env: RLEnvironment, episodes: int, max_steps: int = 1000) -> List[float]:
        """
        测试算法
        
        Args:
            env: 环境
            episodes: 测试轮数
            max_steps: 最大步数
            
        Returns:
            每轮奖励列表
        """
        rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            rewards.append(total_reward)
            
        return rewards


# ==================== Actor-Critic算法 ====================

class ActorCritic:
    """Actor-Critic算法"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.001, discount_factor: float = 0.99):
        """
        初始化Actor-Critic
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            discount_factor: 折扣因子
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.value_network = ValueNetwork(state_dim)
        
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        self.trajectory = []
        
    def get_action(self, state: np.ndarray) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            
        Returns:
            选择的动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy_network.sample(state_tensor)
        return action.item()
        
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        value: float, log_prob: float) -> None:
        """
        存储转移
        
        Args:
            state: 当前状态
            action: 当前动作
            reward: 奖励
            value: 状态价值
            log_prob: 动作log概率
        """
        self.trajectory.append((state, action, reward, value, log_prob))
        
    def train(self) -> None:
        """训练网络"""
        if len(self.trajectory) == 0:
            return
            
        # 计算折扣奖励和优势
        rewards = [t[2] for t in self.trajectory]
        values = [t[3] for t in self.trajectory]
        log_probs = [t[4] for t in self.trajectory]
        
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.discount_factor * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
            
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        values = torch.FloatTensor(values)
        
        # 计算优势
        advantages = discounted_rewards - values
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 计算策略损失
        policy_loss = -torch.stack(log_probs) * advantages
        policy_loss = policy_loss.sum()
        
        # 计算价值损失
        value_loss = F.mse_loss(discounted_rewards, values)
        
        # 更新网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()
        
        # 清空轨迹
        self.trajectory = []
        
    def train_episode(self, env: RLEnvironment, max_steps: int = 1000) -> float:
        """
        训练一轮
        
        Args:
            env: 环境
            max_steps: 最大步数
            
        Returns:
            总奖励
        """
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 选择动作
            action, log_prob = self.policy_network.sample(state_tensor)
            action = action.item()
            
            # 计算状态价值
            value = self.value_network(state_tensor).item()
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            self.store_transition(state, action, reward, value, log_prob.item())
            
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        self.train()
        return total_reward
        
    def test(self, env: RLEnvironment, episodes: int, max_steps: int = 1000) -> List[float]:
        """
        测试算法
        
        Args:
            env: 环境
            episodes: 测试轮数
            max_steps: 最大步数
            
        Returns:
            每轮奖励列表
        """
        rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            rewards.append(total_reward)
            
        return rewards


# ==================== Proximal Policy Optimization (PPO) ====================

class PPO:
    """Proximal Policy Optimization算法"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.0003, discount_factor: float = 0.99,
                 clip_epsilon: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.01, epochs: int = 10):
        """
        初始化PPO
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            discount_factor: 折扣因子
            clip_epsilon: PPO截断参数
            value_coef: 价值函数损失系数
            entropy_coef: 熵正则化系数
            epochs: 每次更新时的训练轮数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.value_network = ValueNetwork(state_dim)
        
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        self.trajectory = []
        
    def get_action(self, state: np.ndarray) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            
        Returns:
            选择的动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy_network.sample(state_tensor)
        return action.item()
        
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        value: float, log_prob: float) -> None:
        """
        存储转移
        
        Args:
            state: 当前状态
            action: 当前动作
            reward: 奖励
            value: 状态价值
            log_prob: 动作log概率
        """
        self.trajectory.append((state, action, reward, value, log_prob))
        
    def train(self) -> None:
        """训练网络"""
        if len(self.trajectory) == 0:
            return
            
        # 准备数据
        states = torch.FloatTensor([t[0] for t in self.trajectory])
        actions = torch.LongTensor([t[1] for t in self.trajectory])
        rewards = [t[2] for t in self.trajectory]
        values = [t[3] for t in self.trajectory]
        old_log_probs = [t[4] for t in self.trajectory]
        
        # 计算折扣奖励
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.discount_factor * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
            
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # 计算优势
        values = torch.FloatTensor(values)
        advantages = discounted_rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多次训练
        for _ in range(self.epochs):
            # 计算新的log概率和价值
            probs = self.policy_network(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            
            new_values = self.value_network(states).squeeze()
            
            # 计算比率
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 计算策略损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            value_loss = F.mse_loss(discounted_rewards, new_values)
            
            # 计算总损失
            total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy.mean()
            
            # 更新网络
            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()
            self.policy_optimizer.step()
            self.value_optimizer.step()
            
        # 清空轨迹
        self.trajectory = []
        
    def train_episode(self, env: RLEnvironment, max_steps: int = 1000) -> float:
        """
        训练一轮
        
        Args:
            env: 环境
            max_steps: 最大步数
            
        Returns:
            总奖励
        """
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 选择动作
            action, log_prob = self.policy_network.sample(state_tensor)
            action = action.item()
            
            # 计算状态价值
            value = self.value_network(state_tensor).item()
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            self.store_transition(state, action, reward, value, log_prob.item())
            
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        self.train()
        return total_reward
        
    def test(self, env: RLEnvironment, episodes: int, max_steps: int = 1000) -> List[float]:
        """
        测试算法
        
        Args:
            env: 环境
            episodes: 测试轮数
            max_steps: 最大步数
            
        Returns:
            每轮奖励列表
        """
        rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            rewards.append(total_reward)
            
        return rewards


# ==================== Trust Region Policy Optimization (TRPO) ====================

class TRPO:
    """Trust Region Policy Optimization算法"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.001, discount_factor: float = 0.99,
                 max_kl: float = 0.01, value_coef: float = 0.5):
        """
        初始化TRPO
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            discount_factor: 折扣因子
            max_kl: 最大KL散度
            value_coef: 价值函数损失系数
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.max_kl = max_kl
        self.value_coef = value_coef
        
        self.policy_network = PolicyNetwork(state_dim, action_dim)
        self.value_network = ValueNetwork(state_dim)
        
        self.policy_optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        self.trajectory = []
        
    def get_action(self, state: np.ndarray) -> int:
        """
        选择动作
        
        Args:
            state: 当前状态
            
        Returns:
            选择的动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action, _ = self.policy_network.sample(state_tensor)
        return action.item()
        
    def store_transition(self, state: np.ndarray, action: int, reward: float, 
                        value: float, log_prob: float) -> None:
        """
        存储转移
        
        Args:
            state: 当前状态
            action: 当前动作
            reward: 奖励
            value: 状态价值
            log_prob: 动作log概率
        """
        self.trajectory.append((state, action, reward, value, log_prob))
        
    def compute_hessian_vector_product(self, states: torch.Tensor, 
                                      old_dist: torch.distributions.Categorical,
                                      vector: torch.Tensor) -> torch.Tensor:
        """
        计算Hessian向量积
        
        Args:
            states: 状态
            old_dist: 旧分布
            vector: 向量
            
        Returns:
            Hessian向量积
        """
        probs = self.policy_network(states)
        dist = torch.distributions.Categorical(probs)
        
        kl = torch.distributions.kl_divergence(old_dist, dist).mean()
        grads = torch.autograd.grad(kl, self.policy_network.parameters(), 
                                   create_graph=True, retain_graph=True)
        grads = torch.cat([grad.view(-1) for grad in grads])
        
        hessian_vector_product = torch.autograd.grad(grads.dot(vector), 
                                                    self.policy_network.parameters(),
                                                    retain_graph=True)
        hessian_vector_product = torch.cat([grad.view(-1) for grad in hessian_vector_product])
        
        return hessian_vector_product
        
    def conjugate_gradient(self, states: torch.Tensor, old_dist: torch.distributions.Categorical,
                          b: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
        """
        共轭梯度法求解线性方程组
        
        Args:
            states: 状态
            old_dist: 旧分布
            b: 向量
            max_iter: 最大迭代次数
            
        Returns:
            求解结果
        """
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rsold = r.dot(r)
        
        for _ in range(max_iter):
            Ap = self.compute_hessian_vector_product(states, old_dist, p)
            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = r.dot(r)
            
            if rsnew < 1e-10:
                break
                
            beta = rsnew / rsold
            p = r + beta * p
            rsold = rsnew
            
        return x
        
    def train(self) -> None:
        """训练网络"""
        if len(self.trajectory) == 0:
            return
            
        # 准备数据
        states = torch.FloatTensor([t[0] for t in self.trajectory])
        actions = torch.LongTensor([t[1] for t in self.trajectory])
        rewards = [t[2] for t in self.trajectory]
        values = [t[3] for t in self.trajectory]
        old_log_probs = [t[4] for t in self.trajectory]
        
        # 计算折扣奖励
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + self.discount_factor * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
            
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        old_log_probs = torch.FloatTensor(old_log_probs)
        
        # 计算优势
        values = torch.FloatTensor(values)
        advantages = discounted_rewards - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 更新价值网络
        for _ in range(5):  # 价值网络训练5次
            new_values = self.value_network(states).squeeze()
            value_loss = F.mse_loss(discounted_rewards, new_values)
            
            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()
            
        # 更新策略网络
        probs = self.policy_network(states)
        old_dist = torch.distributions.Categorical(probs.detach())
        new_dist = torch.distributions.Categorical(probs)
        new_log_probs = new_dist.log_prob(actions)
        
        # 计算比率和优势
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.max_kl, 1 + self.max_kl) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 计算梯度
        grads = torch.autograd.grad(policy_loss, self.policy_network.parameters(),
                                   create_graph=True, retain_graph=True)
        grads = torch.cat([grad.view(-1) for grad in grads])
        
        # 共轭梯度法
        step = self.conjugate_gradient(states, old_dist, grads)
        
        # 线搜索
        old_params = torch.cat([param.view(-1) for param in self.policy_network.parameters()])
        max_step_size = 2 * self.max_kl
        
        for step_size in [0.5**i for i in range(10)]:
            new_params = old_params + step_size * step
            
            # 更新参数
            start = 0
            for param in self.policy_network.parameters():
                end = start + param.numel()
                param.data = new_params[start:end].view(param.shape)
                start = end
                
            # 检查KL散度
            probs = self.policy_network(states)
            new_dist = torch.distributions.Categorical(probs)
            kl = torch.distributions.kl_divergence(old_dist, new_dist).mean()
            
            if kl <= self.max_kl:
                break
                
        # 清空轨迹
        self.trajectory = []
        
    def train_episode(self, env: RLEnvironment, max_steps: int = 1000) -> float:
        """
        训练一轮
        
        Args:
            env: 环境
            max_steps: 最大步数
            
        Returns:
            总奖励
        """
        state = env.reset()
        total_reward = 0
        
        for step in range(max_steps):
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # 选择动作
            action, log_prob = self.policy_network.sample(state_tensor)
            action = action.item()
            
            # 计算状态价值
            value = self.value_network(state_tensor).item()
            
            # 执行动作
            next_state, reward, done, _ = env.step(action)
            
            self.store_transition(state, action, reward, value, log_prob.item())
            
            state = next_state
            total_reward += reward
            
            if done:
                break
                
        self.train()
        return total_reward
        
    def test(self, env: RLEnvironment, episodes: int, max_steps: int = 1000) -> List[float]:
        """
        测试算法
        
        Args:
            env: 环境
            episodes: 测试轮数
            max_steps: 最大步数
            
        Returns:
            每轮奖励列表
        """
        rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            rewards.append(total_reward)
            
        return rewards


# ==================== Deep Deterministic Policy Gradient (DDPG) ====================

class DDPG:
    """Deep Deterministic Policy Gradient算法"""
    
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 0.001, discount_factor: float = 0.99,
                 tau: float = 0.005, buffer_size: int = 10000, batch_size: int = 64):
        """
        初始化DDPG
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            discount_factor: 折扣因子
            tau: 软更新参数
            buffer_size: 经验回放缓冲区大小
            batch_size: 批次大小
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.tau = tau
        self.batch_size = batch_size
        
        # Actor网络
        self.actor = ActorNetwork(state_dim, action_dim)
        self.actor_target = ActorNetwork(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        
        # Critic网络
        self.critic = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # 初始化目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 经验回放缓冲区
        self.buffer = ReplayBuffer(buffer_size)
        
    def get_action(self, state: np.ndarray, noise: bool = True, noise_std: float = 0.1) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 当前状态
            noise: 是否添加噪声
            noise_std: 噪声标准差
            
        Returns:
            选择的动作
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor)
        action = action.numpy().flatten()
        
        if noise:
            action += np.random.normal(0, noise_std, size=action.shape)
            
        # 限制动作范围
        action = np.clip(action, -1, 1)
        return action
        
    def update(self, state: np.ndarray, action: np.ndarray, reward: float, 
               next_state: np.ndarray, done: bool) -> None:
        """
        添加经验到缓冲区并训练
        
        Args:
            state: 当前状态
            action: 当前动作
            reward: 奖励
            next_state: 下一状态
            done: 是否结束
        """
        self.buffer.push(state, action, reward, next_state, done)
        
        # 训练网络
        if len(self.buffer) >= self.batch_size:
            self._train_step()
            
    def _train_step(self) -> None:
        """训练一步"""
        experiences = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor([e[0] for e in experiences])
        actions = torch.FloatTensor([e[1] for e in experiences])
        rewards = torch.FloatTensor([e[2] for e in experiences]).unsqueeze(1)
        next_states = torch.FloatTensor([e[3] for e in experiences])
        dones = torch.BoolTensor([e[4] for e in experiences]).unsqueeze(1)
        
        # 更新Critic网络
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions)
        target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
        
        current_q_values = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q_values, target_q_values.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor网络
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        """
        软更新目标网络
        
        Args:
            source: 源网络
            target: 目标网络
        """
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)
            
    def train(self, env: RLEnvironment, episodes: int, max_steps: int = 1000) -> List[float]:
        """
        训练算法
        
        Args:
            env: 环境
            episodes: 训练轮数
            max_steps: 最大步数
            
        Returns:
            每轮奖励列表
        """
        rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                
                self.update(state, action, reward, next_state, done)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            rewards.append(total_reward)
            
        return rewards
        
    def test(self, env: RLEnvironment, episodes: int, max_steps: int = 1000) -> List[float]:
        """
        测试算法
        
        Args:
            env: 环境
            episodes: 测试轮数
            max_steps: 最大步数
            
        Returns:
            每轮奖励列表
        """
        rewards = []
        
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                action = self.get_action(state, noise=False)
                next_state, reward, done, _ = env.step(action)
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
                    
            rewards.append(total_reward)
            
        return rewards


# ==================== Multi-Agent Reinforcement Learning ====================

class MultiAgentDQN:
    """多智能体DQN算法"""
    
    def __init__(self, state_dims: List[int], action_dims: List[int], 
                 learning_rate: float = 0.001, discount_factor: float = 0.99,
                 epsilon: float = 1.0, epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.01, buffer_size: int = 10000,
                 batch_size: int = 32):
        """
        初始化多智能体DQN
        
        Args:
            state_dims: 各智能体状态维度列表
            action_dims: 各智能体动作维度列表
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            buffer_size: 经验回放缓冲区大小
            batch_size: 批次大小
        """
        self.num_agents = len(state_dims)
        self.state_dims = state_dims
        self.action_dims = action_dims
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        
        # 为每个智能体创建网络
        self.q_networks = []
        self.target_networks = []
        self.optimizers = []
        
        for i in range(self.num_agents):
            q_net = QNetwork(state_dims[i], action_dims[i])
            target_net = QNetwork(state_dims[i], action_dims[i])
            optimizer = optim.Adam(q_net.parameters(), lr=learning_rate)
            
            self.q_networks.append(q_net)
            self.target_networks.append(target_net)
            self.optimizers.append(optimizer)
            
        # 同步目标网络
        for target_net, q_net in zip(self.target_networks, self.q_networks):
            target_net.load_state_dict(q_net.state_dict())
            
        # 经验回放缓冲区
        self.buffer = ReplayBuffer(buffer_size)
        
    def get_actions(self, states: List[np.ndarray], training: bool = True) -> List[int]:
        """
        选择动作
        
        Args:
            states: 各智能体状态列表
            training: 是否为训练模式
            
        Returns:
            各智能体动作列表
        """
        actions = []
        
        for i, state in enumerate(states):
            if training and np.random.random() < self.epsilon:
                action = np.random.randint(self.action_dims[i])
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                with torch.no_grad():
                    q_values = self.q_networks[i](state_tensor)
                action = q_values.argmax().item()
            actions.append(action)
            
        return actions
        
    def update(self, states: List[np.ndarray], actions: List[int], 
               rewards: List[float], next_states: List[np.ndarray], 
               dones: List[bool]) -> None:
        """
        添加经验到缓冲区并训练
        
        Args:
            states: 各智能体状态列表
            actions: 各智能体动作列表
            rewards: 各智能体奖励列表
            next_states: 各智能体下一状态列表
            dones: 各智能体是否结束列表
        """
        # 组合经验（这里简化处理，实际中可能需要更复杂的组合策略）
        combined_state = np.concatenate(states)
        combined_next_state = np.concatenate(next_states)
        combined_reward = sum(rewards)
        combined_done = any(dones)
        
        # 这里简化处理，实际中可能需要更复杂的经验存储方式
        # 为了简化，我们只存储第一个智能体的经验
        self.buffer.push(states[0], actions[0], combined_reward, next_states[0], combined_done)
        
        # 训练网络
        if len(self.buffer) >= self.batch_size:
            self._train_step()
            
        # 衰减探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def _train_step(self) -> None:
        """训练一步"""
        experiences = self.buffer.sample(self.batch_size)
        
        states = torch.FloatTensor([e[0] for e in experiences])
        actions = torch.LongTensor([e[1] for e in experiences]).unsqueeze(1)
        rewards = torch.FloatTensor([e[2] for e in experiences]).unsqueeze(1)
        next_states = torch.FloatTensor([e[3] for e in experiences])
        dones = torch.BoolTensor([e[4] for e in experiences]).unsqueeze(1)
        
        # 这里简化处理，只训练第一个智能体
        # 实际中需要为每个智能体计算损失
        current_q_values = self.q_networks[0](states).gather(1, actions)
        
        with torch.no_grad():
            next_q_values = self.target_networks[0](next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + (self.discount_factor * next_q_values * ~dones)
            
        loss = F.mse_loss(current_q_values, target_q_values)
        
        self.optimizers[0].zero_grad()
        loss.backward()
        self.optimizers[0].step()
        
        # 更新目标网络
        self.target_networks[0].load_state_dict(self.q_networks[0].state_dict())
        
    def train(self, env: 'MultiAgentEnvironment', episodes: int, max_steps: int = 1000) -> List[float]:
        """
        训练算法
        
        Args:
            env: 多智能体环境
            episodes: 训练轮数
            max_steps: 最大步数
            
        Returns:
            每轮奖励列表
        """
        rewards = []
        
        for episode in range(episodes):
            states = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                actions = self.get_actions(states)
                next_states, reward_list, done_list, _ = env.step(actions)
                
                self.update(states, actions, reward_list, next_states, done_list)
                
                states = next_states
                total_reward += sum(reward_list)
                
                if all(done_list):
                    break
                    
            rewards.append(total_reward)
            
        return rewards
        
    def test(self, env: 'MultiAgentEnvironment', episodes: int, max_steps: int = 1000) -> List[float]:
        """
        测试算法
        
        Args:
            env: 多智能体环境
            episodes: 测试轮数
            max_steps: 最大步数
            
        Returns:
            每轮奖励列表
        """
        rewards = []
        old_epsilon = self.epsilon
        self.epsilon = 0  # 测试时不使用探索
        
        for episode in range(episodes):
            states = env.reset()
            total_reward = 0
            
            for step in range(max_steps):
                actions = self.get_actions(states, training=False)
                next_states, reward_list, done_list, _ = env.step(actions)
                
                states = next_states
                total_reward += sum(reward_list)
                
                if all(done_list):
                    break
                    
            rewards.append(total_reward)
            
        self.epsilon = old_epsilon
        return rewards


class MultiAgentEnvironment:
    """多智能体环境接口"""
    
    def __init__(self, env_configs: List[Dict]):
        """
        初始化多智能体环境
        
        Args:
            env_configs: 各智能体环境配置列表
        """
        self.envs = []
        for config in env_configs:
            env = GymEnvironment(config['env_name'])
            self.envs.append(env)
            
        self.num_agents = len(self.envs)
        
    def reset(self) -> List[np.ndarray]:
        """重置所有智能体环境"""
        return [env.reset() for env in self.envs]
        
    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """
        执行所有智能体的动作
        
        Args:
            actions: 各智能体动作列表
            
        Returns:
            (下一状态列表, 奖励列表, 是否结束列表, 信息字典)
        """
        next_states = []
        rewards = []
        dones = []
        
        for env, action in zip(self.envs, actions):
            next_state, reward, done, _ = env.step(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            
        return next_states, rewards, dones, {}
        
    def close(self) -> None:
        """关闭所有环境"""
        for env in self.envs:
            env.close()


# ==================== 强化学习算法库类 ====================

class RLAlgorithmLibrary:
    """强化学习算法库"""
    
    def __init__(self):
        """初始化算法库"""
        self.algorithms = {}
        
    def create_q_learning(self, state_space: gym.Space, action_space: gym.Space,
                         learning_rate: float = 0.1, discount_factor: float = 0.95,
                         epsilon: float = 0.1, epsilon_decay: float = 0.995) -> QLearning:
        """
        创建Q-Learning算法
        
        Args:
            state_space: 状态空间
            action_space: 动作空间
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            
        Returns:
            Q-Learning算法实例
        """
        return QLearning(state_space, action_space, learning_rate, 
                        discount_factor, epsilon, epsilon_decay)
        
    def create_dqn(self, state_dim: int, action_dim: int,
                   learning_rate: float = 0.001, discount_factor: float = 0.99,
                   epsilon: float = 1.0, epsilon_decay: float = 0.995,
                   epsilon_min: float = 0.01, buffer_size: int = 10000,
                   batch_size: int = 32, target_update_freq: int = 100,
                   use_dueling: bool = False, use_prioritized: bool = False) -> DQN:
        """
        创建DQN算法
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            buffer_size: 经验回放缓冲区大小
            batch_size: 批次大小
            target_update_freq: 目标网络更新频率
            use_dueling: 是否使用Dueling架构
            use_prioritized: 是否使用优先级回放
            
        Returns:
            DQN算法实例
        """
        return DQN(state_dim, action_dim, learning_rate, discount_factor,
                  epsilon, epsilon_decay, epsilon_min, buffer_size,
                  batch_size, target_update_freq, use_dueling, use_prioritized)
        
    def create_policy_gradient(self, state_dim: int, action_dim: int,
                              learning_rate: float = 0.001, 
                              discount_factor: float = 0.99) -> PolicyGradient:
        """
        创建Policy Gradient算法
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            discount_factor: 折扣因子
            
        Returns:
            Policy Gradient算法实例
        """
        return PolicyGradient(state_dim, action_dim, learning_rate, discount_factor)
        
    def create_actor_critic(self, state_dim: int, action_dim: int,
                           learning_rate: float = 0.001, 
                           discount_factor: float = 0.99) -> ActorCritic:
        """
        创建Actor-Critic算法
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            discount_factor: 折扣因子
            
        Returns:
            Actor-Critic算法实例
        """
        return ActorCritic(state_dim, action_dim, learning_rate, discount_factor)
        
    def create_ppo(self, state_dim: int, action_dim: int,
                   learning_rate: float = 0.0003, discount_factor: float = 0.99,
                   clip_epsilon: float = 0.2, value_coef: float = 0.5,
                   entropy_coef: float = 0.01, epochs: int = 10) -> PPO:
        """
        创建PPO算法
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            discount_factor: 折扣因子
            clip_epsilon: PPO截断参数
            value_coef: 价值函数损失系数
            entropy_coef: 熵正则化系数
            epochs: 每次更新时的训练轮数
            
        Returns:
            PPO算法实例
        """
        return PPO(state_dim, action_dim, learning_rate, discount_factor,
                  clip_epsilon, value_coef, entropy_coef, epochs)
        
    def create_trpo(self, state_dim: int, action_dim: int,
                    learning_rate: float = 0.001, discount_factor: float = 0.99,
                    max_kl: float = 0.01, value_coef: float = 0.5) -> TRPO:
        """
        创建TRPO算法
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            discount_factor: 折扣因子
            max_kl: 最大KL散度
            value_coef: 价值函数损失系数
            
        Returns:
            TRPO算法实例
        """
        return TRPO(state_dim, action_dim, learning_rate, discount_factor,
                   max_kl, value_coef)
        
    def create_ddpg(self, state_dim: int, action_dim: int,
                    learning_rate: float = 0.001, discount_factor: float = 0.99,
                    tau: float = 0.005, buffer_size: int = 10000,
                    batch_size: int = 64) -> DDPG:
        """
        创建DDPG算法
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            learning_rate: 学习率
            discount_factor: 折扣因子
            tau: 软更新参数
            buffer_size: 经验回放缓冲区大小
            batch_size: 批次大小
            
        Returns:
            DDPG算法实例
        """
        return DDPG(state_dim, action_dim, learning_rate, discount_factor,
                   tau, buffer_size, batch_size)
        
    def create_multi_agent_dqn(self, state_dims: List[int], action_dims: List[int],
                              learning_rate: float = 0.001, discount_factor: float = 0.99,
                              epsilon: float = 1.0, epsilon_decay: float = 0.995,
                              epsilon_min: float = 0.01, buffer_size: int = 10000,
                              batch_size: int = 32) -> MultiAgentDQN:
        """
        创建多智能体DQN算法
        
        Args:
            state_dims: 各智能体状态维度列表
            action_dims: 各智能体动作维度列表
            learning_rate: 学习率
            discount_factor: 折扣因子
            epsilon: 探索率
            epsilon_decay: 探索率衰减
            epsilon_min: 最小探索率
            buffer_size: 经验回放缓冲区大小
            batch_size: 批次大小
            
        Returns:
            多智能体DQN算法实例
        """
        return MultiAgentDQN(state_dims, action_dims, learning_rate, discount_factor,
                           epsilon, epsilon_decay, epsilon_min, buffer_size, batch_size)
        
    def create_gym_environment(self, env_name: str) -> GymEnvironment:
        """
        创建Gym环境
        
        Args:
            env_name: 环境名称
            
        Returns:
            Gym环境实例
        """
        return GymEnvironment(env_name)
        
    def create_multi_agent_environment(self, env_configs: List[Dict]) -> MultiAgentEnvironment:
        """
        创建多智能体环境
        
        Args:
            env_configs: 各智能体环境配置列表
            
        Returns:
            多智能体环境实例
        """
        return MultiAgentEnvironment(env_configs)


# ==================== 测试用例 ====================

def test_q_learning():
    """测试Q-Learning算法"""
    print("测试Q-Learning算法...")
    
    # 创建环境
    env = GymEnvironment('CartPole-v1')
    
    # 创建算法
    library = RLAlgorithmLibrary()
    ql = library.create_q_learning(env.state_space, env.action_space)
    
    # 训练
    rewards = ql.train(env, episodes=100)
    print(f"Q-Learning训练完成，平均奖励: {np.mean(rewards[-10:]):.2f}")
    
    # 测试
    test_rewards = ql.test(env, episodes=10)
    print(f"Q-Learning测试结果，平均奖励: {np.mean(test_rewards):.2f}")


def test_dqn():
    """测试DQN算法"""
    print("\n测试DQN算法...")
    
    # 创建环境
    env = GymEnvironment('CartPole-v1')
    
    # 创建算法
    library = RLAlgorithmLibrary()
    dqn = library.create_dqn(env.state_space.shape[0], env.action_space.n)
    
    # 训练
    rewards = dqn.train(env, episodes=200)
    print(f"DQN训练完成，平均奖励: {np.mean(rewards[-10:]):.2f}")
    
    # 测试
    test_rewards = dqn.test(env, episodes=10)
    print(f"DQN测试结果，平均奖励: {np.mean(test_rewards):.2f}")


def test_policy_gradient():
    """测试Policy Gradient算法"""
    print("\n测试Policy Gradient算法...")
    
    # 创建环境
    env = GymEnvironment('CartPole-v1')
    
    # 创建算法
    library = RLAlgorithmLibrary()
    pg = library.create_policy_gradient(env.state_space.shape[0], env.action_space.n)
    
    # 训练
    rewards = []
    for episode in range(200):
        reward = pg.train_episode(env)
        rewards.append(reward)
        
    print(f"Policy Gradient训练完成，平均奖励: {np.mean(rewards[-10:]):.2f}")
    
    # 测试
    test_rewards = pg.test(env, episodes=10)
    print(f"Policy Gradient测试结果，平均奖励: {np.mean(test_rewards):.2f}")


def test_actor_critic():
    """测试Actor-Critic算法"""
    print("\n测试Actor-Critic算法...")
    
    # 创建环境
    env = GymEnvironment('CartPole-v1')
    
    # 创建算法
    library = RLAlgorithmLibrary()
    ac = library.create_actor_critic(env.state_space.shape[0], env.action_space.n)
    
    # 训练
    rewards = []
    for episode in range(200):
        reward = ac.train_episode(env)
        rewards.append(reward)
        
    print(f"Actor-Critic训练完成，平均奖励: {np.mean(rewards[-10:]):.2f}")
    
    # 测试
    test_rewards = ac.test(env, episodes=10)
    print(f"Actor-Critic测试结果，平均奖励: {np.mean(test_rewards):.2f}")


def test_ppo():
    """测试PPO算法"""
    print("\n测试PPO算法...")
    
    # 创建环境
    env = GymEnvironment('CartPole-v1')
    
    # 创建算法
    library = RLAlgorithmLibrary()
    ppo = library.create_ppo(env.state_space.shape[0], env.action_space.n)
    
    # 训练
    rewards = []
    for episode in range(200):
        reward = ppo.train_episode(env)
        rewards.append(reward)
        
    print(f"PPO训练完成，平均奖励: {np.mean(rewards[-10:]):.2f}")
    
    # 测试
    test_rewards = ppo.test(env, episodes=10)
    print(f"PPO测试结果，平均奖励: {np.mean(test_rewards):.2f}")


def test_trpo():
    """测试TRPO算法"""
    print("\n测试TRPO算法...")
    
    # 创建环境
    env = GymEnvironment('CartPole-v1')
    
    # 创建算法
    library = RLAlgorithmLibrary()
    trpo = library.create_trpo(env.state_space.shape[0], env.action_space.n)
    
    # 训练
    rewards = []
    for episode in range(100):  # TRPO训练较慢，减少轮数
        reward = trpo.train_episode(env)
        rewards.append(reward)
        
    print(f"TRPO训练完成，平均奖励: {np.mean(rewards[-10:]):.2f}")
    
    # 测试
    test_rewards = trpo.test(env, episodes=10)
    print(f"TRPO测试结果，平均奖励: {np.mean(test_rewards):.2f}")


def test_ddpg():
    """测试DDPG算法"""
    print("\n测试DDPG算法...")
    
    # 创建环境
    env = GymEnvironment('Pendulum-v1')
    
    # 创建算法
    library = RLAlgorithmLibrary()
    ddpg = library.create_ddpg(env.state_space.shape[0], env.action_space.shape[0])
    
    # 训练
    rewards = ddpg.train(env, episodes=200)
    print(f"DDPG训练完成，平均奖励: {np.mean(rewards[-10:]):.2f}")
    
    # 测试
    test_rewards = ddpg.test(env, episodes=10)
    print(f"DDPG测试结果，平均奖励: {np.mean(test_rewards):.2f}")


def test_multi_agent_dqn():
    """测试多智能体DQN算法"""
    print("\n测试多智能体DQN算法...")
    
    # 创建多智能体环境
    env_configs = [
        {'env_name': 'CartPole-v1'},
        {'env_name': 'CartPole-v1'}
    ]
    
    library = RLAlgorithmLibrary()
    multi_env = library.create_multi_agent_environment(env_configs)
    
    # 创建算法
    state_dims = [4, 4]  # CartPole状态维度
    action_dims = [2, 2]  # CartPole动作维度
    madqn = library.create_multi_agent_dqn(state_dims, action_dims)
    
    # 训练
    rewards = madqn.train(multi_env, episodes=100)
    print(f"多智能体DQN训练完成，平均奖励: {np.mean(rewards[-10:]):.2f}")
    
    # 测试
    test_rewards = madqn.test(multi_env, episodes=10)
    print(f"多智能体DQN测试结果，平均奖励: {np.mean(test_rewards):.2f}")


def run_all_tests():
    """运行所有测试"""
    print("开始运行强化学习算法库测试...")
    print("=" * 50)
    
    try:
        test_q_learning()
    except Exception as e:
        print(f"Q-Learning测试失败: {e}")
    
    try:
        test_dqn()
    except Exception as e:
        print(f"DQN测试失败: {e}")
    
    try:
        test_policy_gradient()
    except Exception as e:
        print(f"Policy Gradient测试失败: {e}")
    
    try:
        test_actor_critic()
    except Exception as e:
        print(f"Actor-Critic测试失败: {e}")
    
    try:
        test_ppo()
    except Exception as e:
        print(f"PPO测试失败: {e}")
    
    try:
        test_trpo()
    except Exception as e:
        print(f"TRPO测试失败: {e}")
    
    try:
        test_ddpg()
    except Exception as e:
        print(f"DDPG测试失败: {e}")
    
    try:
        test_multi_agent_dqn()
    except Exception as e:
        print(f"多智能体DQN测试失败: {e}")
    
    print("\n所有测试完成!")


if __name__ == "__main__":
    # 运行所有测试
    run_all_tests()