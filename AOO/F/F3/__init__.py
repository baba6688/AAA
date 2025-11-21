"""
F3元学习器模块
===============

实现元学习功能，包括：
- 学习如何学习的能力
- 元认知监控和分析
- 学习策略自动选择
- 学习效果评估和优化
- 元知识提取和管理
- 学习迁移和适应
- 元学习模型更新

主要类：
- MetaLearner: 主元学习器类
- MAMLAlgorithm: MAML元学习算法
- ReptileAlgorithm: Reptile元学习算法
- MetaCognitionMonitor: 元认知监控器
- LearningStrategySelector: 学习策略选择器
- MetaKnowledgeExtractor: 元知识提取器
- LearningTransferManager: 学习迁移管理器
"""

from .MetaLearner import (
    MetaLearner,
    MAMLAlgorithm,
    ReptileAlgorithm,
    MetaCognitionMonitor,
    LearningStrategySelector,
    MetaKnowledgeExtractor,
    LearningTransferManager,
    BaseLearner,
    SimpleNeuralNetwork,
    LearningStrategy,
    LearningEpisode,
    MetaKnowledge,
    create_sample_task,
    demo_meta_learner
)

__version__ = "1.0.0"
__author__ = "F3元学习器开发团队"

__all__ = [
    'MetaLearner',
    'MAMLAlgorithm', 
    'ReptileAlgorithm',
    'MetaCognitionMonitor',
    'LearningStrategySelector',
    'MetaKnowledgeExtractor',
    'LearningTransferManager',
    'BaseLearner',
    'SimpleNeuralNetwork',
    'LearningStrategy',
    'LearningEpisode',
    'MetaKnowledge',
    'create_sample_task',
    'demo_meta_learner'
]