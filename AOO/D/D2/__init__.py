"""
D2 元认知处理器模块
实现元认知监控、认知负荷监控、知识库管理等功能

主要类:
- CognitiveState: 认知状态枚举
- StrategyType: 策略类型枚举
- LoadLevel: 负荷水平枚举
- CognitiveEvent: 认知事件
- CognitiveStrategy: 认知策略
- MetaCognitionMetrics: 元认知指标
- MetaCognitionKnowledgeBase: 元认知知识库
- CognitiveLoadMonitor: 认知负荷监控器
- MetaCognitionProcessor: 元认知处理器

版本: 1.0.0
作者: AI量化系统
"""

from .MetaCognitionProcessor import (
    CognitiveState,
    StrategyType,
    LoadLevel,
    CognitiveEvent,
    CognitiveStrategy,
    MetaCognitionMetrics,
    MetaCognitionKnowledgeBase,
    CognitiveLoadMonitor,
    MetaCognitionProcessor
)

__version__ = "1.0.0"
__author__ = "AI量化系统"

__all__ = [
    'CognitiveState',
    'StrategyType',
    'LoadLevel',
    'CognitiveEvent',
    'CognitiveStrategy',
    'MetaCognitionMetrics',
    'MetaCognitionKnowledgeBase',
    'CognitiveLoadMonitor',
    'MetaCognitionProcessor'
]