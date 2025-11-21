"""
D1 自我认知引擎模块
实现自我认知、认知偏差检测、反思引擎等功能

主要类:
- CognitiveState: 认知状态
- CognitiveBias: 认知偏差
- LearningEvent: 学习事件
- CapabilityModel: 能力模型
- SelfCognitionEngine: 自我认知引擎
- BiasDetector: 偏差检测器
- ReflectionEngine: 反思引擎

版本: 1.0.0
作者: AI量化系统
"""

from .SelfCognitionEngine import (
    CognitiveState,
    CognitiveBias,
    LearningEvent,
    CapabilityModel,
    SelfCognitionEngine,
    BiasDetector,
    ReflectionEngine
)

__version__ = "1.0.0"
__author__ = "AI量化系统"

__all__ = [
    'CognitiveState',
    'CognitiveBias',
    'LearningEvent',
    'CapabilityModel',
    'SelfCognitionEngine',
    'BiasDetector',
    'ReflectionEngine'
]