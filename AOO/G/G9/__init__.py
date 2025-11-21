"""
G9执行状态聚合器模块
实现多模块执行状态融合、评估、检验、排序、历史记录、报告和预警功能
"""

from .ExecutionStateAggregator import (
    ExecutionStateAggregator,
    ExecutionState,
    ExecutionMetrics,
    StateHistory,
    StateReport,
    ExecutionStatus,
    PriorityLevel,
    ConsistencyStatus,
    IntelligentFusionAlgorithm
)

__all__ = [
    'ExecutionStateAggregator',
    'ExecutionState',
    'ExecutionMetrics',
    'StateHistory',
    'StateReport',
    'ExecutionStatus',
    'PriorityLevel',
    'ConsistencyStatus',
    'IntelligentFusionAlgorithm'
]

__version__ = "1.0.0"
__author__ = "AI System"
__description__ = "G9执行状态聚合器 - 智能执行状态融合和管理系统"