"""
U3模块：强化学习算法库
=====================

这是U区子模块U3的导出接口，提供完整的强化学习算法库功能。

模块包含以下主要组件：
- 基础环境接口（RLEnvironment, GymEnvironment）
- 经验回放缓冲区（ReplayBuffer, PrioritizedReplayBuffer）
- 神经网络模型（QNetwork, DuelingQNetwork, PolicyNetwork, 等）
- 强化学习算法（QLearning, DQN, PolicyGradient, ActorCritic, PPO, TRPO, DDPG, 等）
- 多智能体强化学习（MultiAgentDQN, MultiAgentEnvironment）
- 算法库管理器（RLAlgorithmLibrary）

使用示例：
    from U.U3 import RLAlgorithmLibrary, DQN, PPO
    
    # 创建算法库
    library = RLAlgorithmLibrary()
    
    # 创建DQN算法
    dqn = library.create_dqn(state_dim=4, action_dim=2)
    
    # 创建PPO算法
    ppo = library.create_ppo(state_dim=4, action_dim=2)

作者：U3开发团队
版本：1.0.0
"""

# 导入基础环境接口
from .RLAlgorithmLibrary import (
    RLEnvironment,
    GymEnvironment
)

# 导入经验回放缓冲区
from .RLAlgorithmLibrary import (
    ReplayBuffer,
    PrioritizedReplayBuffer
)

# 导入神经网络模型
from .RLAlgorithmLibrary import (
    QNetwork,
    DuelingQNetwork,
    PolicyNetwork,
    ContinuousPolicyNetwork,
    ValueNetwork,
    ActorNetwork,
    CriticNetwork
)

# 导入强化学习算法
from .RLAlgorithmLibrary import (
    QLearning,
    DQN,
    PolicyGradient,
    ActorCritic,
    PPO,
    TRPO,
    DDPG
)

# 导入多智能体强化学习
from .RLAlgorithmLibrary import (
    MultiAgentDQN,
    MultiAgentEnvironment
)

# 导入算法库管理器
from .RLAlgorithmLibrary import (
    RLAlgorithmLibrary
)

# 定义模块导出列表
__all__ = [
    # 基础环境接口
    'RLEnvironment',
    'GymEnvironment',
    
    # 经验回放缓冲区
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    
    # 神经网络模型
    'QNetwork',
    'DuelingQNetwork',
    'PolicyNetwork',
    'ContinuousPolicyNetwork',
    'ValueNetwork',
    'ActorNetwork',
    'CriticNetwork',
    
    # 强化学习算法
    'QLearning',
    'DQN',
    'PolicyGradient',
    'ActorCritic',
    'PPO',
    'TRPO',
    'DDPG',
    
    # 多智能体强化学习
    'MultiAgentDQN',
    'MultiAgentEnvironment',
    
    # 算法库管理器
    'RLAlgorithmLibrary'
]

# 模块版本信息
__version__ = '1.0.0'
__author__ = 'U3开发团队'

# 模块初始化信息
def _initialize_module():
    """模块初始化函数"""
    print(f"U3模块（强化学习算法库）已成功加载，版本：{__version__}")
    print(f"共导出 {len(__all__)} 个类和接口")

# 在模块导入时自动执行初始化
try:
    _initialize_module()
except ImportError:
    # 在某些环境中可能无法显示输出，静默处理
    pass