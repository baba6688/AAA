"""
E2 创意引擎模块
Enhanced Creative Engine - 提供多模态创意生成、评估和优化功能
"""

from .CreativeEngine import (
    CreativeEngine,
    CreativeGenerator,
    QualityEvaluator,
    IdeaFusion,
    EvolutionEngine,
    CreativeLibrary,
    ApplicationFramework
)

__all__ = [
    'CreativeEngine',
    'CreativeGenerator', 
    'QualityEvaluator',
    'IdeaFusion',
    'EvolutionEngine',
    'CreativeLibrary',
    'ApplicationFramework'
]

__version__ = "1.0.0"