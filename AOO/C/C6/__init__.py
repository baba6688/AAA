"""
C6 模块 - 推理链管理器
实现推理链的构建、管理、优化和验证功能
"""

from .ReasoningChainManager import (
    ReasoningChainManager,
    ReasoningType,
    NodeType,
    ConfidenceLevel,
    ReasoningNode,
    ReasoningEdge,
    ReasoningPath,
    PerformanceMetrics
)

__all__ = [
    'ReasoningChainManager',
    'ReasoningType',
    'NodeType', 
    'ConfidenceLevel',
    'ReasoningNode',
    'ReasoningEdge',
    'ReasoningPath',
    'PerformanceMetrics'
]

__version__ = '1.0.0'
__author__ = 'AI Assistant'
__description__ = '推理链管理器 - 支持多种推理算法和可视化功能'