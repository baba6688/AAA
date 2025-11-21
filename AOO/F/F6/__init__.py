"""
F6层次学习管理器模块

该模块实现了层次化学习管理系统：
- 多层次学习状态和迁移管理
- 学习冲突检测和解决
- 层次化学习节点管理
- 学习 transferência 和知识传递
- 综合层次学习管理器
"""

from .HierarchicalLearningManager import (
    LearningLevel,
    LearningState,
    ConflictType,
    LearningNode,
    LearningTransfer,
    ConflictInfo,
    HierarchicalLearningManager,
    LevelManager
)

__all__ = [
    'LearningLevel',
    'LearningState',
    'ConflictType',
    'LearningNode',
    'LearningTransfer',
    'ConflictInfo',
    'HierarchicalLearningManager',
    'LevelManager'
]