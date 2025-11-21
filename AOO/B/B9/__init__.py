"""
B9认知状态聚合器包
Cognition State Aggregator
"""

from .CognitionStateAggregator import (
    CognitiveStateLevel,   # 认知状态级别枚举
    ConsistencyStatus,     # 一致性状态枚举
    PerceptionResult,      # 感知结果
    CognitiveState,        # 认知状态
    CognitiveReport,       # 认知报告
    CognitionStateAggregator # 认知状态聚合器主类
)

__version__ = "1.0.0"
__author__ = "B9 Team"

__all__ = [
    'CognitiveStateLevel',
    'ConsistencyStatus',
    'PerceptionResult',
    'CognitiveState',
    'CognitiveReport',
    'CognitionStateAggregator'
]