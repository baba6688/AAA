"""
C9 知识状态聚合器模块
实现多源知识的融合、状态管理、可信度评估等功能

主要类:
- KnowledgeSource: 知识源
- KnowledgeItem: 知识项
- KnowledgeState: 知识状态
- KnowledgeFusionEngine: 知识融合引擎
- KnowledgeStateEvaluator: 知识状态评估器
- KnowledgeCredibilityCalculator: 知识可信度计算器
- KnowledgePriorityRanker: 知识优先级排序器
- KnowledgeStateHistory: 知识状态历史
- KnowledgeStateReporter: 知识状态报告器
- KnowledgeStateAlerter: 知识状态告警器
- KnowledgeStateAggregator: 知识状态聚合器

版本: 1.0.0
作者: AI量化系统
"""

from .KnowledgeStateAggregator import (
    KnowledgeSource,
    KnowledgeItem,
    KnowledgeState,
    KnowledgeFusionEngine,
    KnowledgeStateEvaluator,
    KnowledgeCredibilityCalculator,
    KnowledgePriorityRanker,
    KnowledgeStateHistory,
    KnowledgeStateReporter,
    KnowledgeStateAlerter,
    KnowledgeStateAggregator
)

__version__ = "1.0.0"
__author__ = "AI量化系统"

__all__ = [
    'KnowledgeSource',
    'KnowledgeItem',
    'KnowledgeState',
    'KnowledgeFusionEngine',
    'KnowledgeStateEvaluator',
    'KnowledgeCredibilityCalculator',
    'KnowledgePriorityRanker',
    'KnowledgeStateHistory',
    'KnowledgeStateReporter',
    'KnowledgeStateAlerter',
    'KnowledgeStateAggregator'
]