"""
D9 自我意识状态聚合器模块
实现自我意识状态融合、意识一致性检查、优先级排序等功能

主要类:
- SelfAwarenessLevel: 自我意识水平枚举
- ConsciousnessConsistency: 意识一致性枚举
- AwarenessPriority: 意识优先级枚举
- AwarenessModule: 意识模块
- SelfAwarenessState: 自我意识状态
- AwarenessResult: 意识结果
- SelfAwarenessReport: 自我意识报告
- SelfAwarenessFusionEngine: 自我意识融合引擎
- SelfAwarenessStateAggregator: 自我意识状态聚合器

版本: 1.0.0
作者: AI量化系统
"""

from .SelfAwarenessStateAggregator import (
    SelfAwarenessLevel,
    ConsciousnessConsistency,
    AwarenessPriority,
    AwarenessModule,
    SelfAwarenessState,
    AwarenessResult,
    SelfAwarenessReport,
    SelfAwarenessFusionEngine,
    SelfAwarenessStateAggregator
)

__version__ = "1.0.0"
__author__ = "AI量化系统"

__all__ = [
    'SelfAwarenessLevel',
    'ConsciousnessConsistency',
    'AwarenessPriority',
    'AwarenessModule',
    'SelfAwarenessState',
    'AwarenessResult',
    'SelfAwarenessReport',
    'SelfAwarenessFusionEngine',
    'SelfAwarenessStateAggregator'
]