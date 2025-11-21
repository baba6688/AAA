"""
F7经验管理器模块
提供多层次经验存储、智能检索、质量评估和应用推荐功能
"""

from .ExperienceManager import (
    ExperienceManager,
    Experience,
    ExperienceLevel,
    ExperienceType,
    QualityLevel,
    ExperienceMetrics,
    ExperienceStorage,
    ExperienceAnalyzer,
    ExperienceRecommender
)

__all__ = [
    'ExperienceManager',
    'Experience',
    'ExperienceLevel',
    'ExperienceType',
    'QualityLevel',
    'ExperienceMetrics',
    'ExperienceStorage',
    'ExperienceAnalyzer',
    'ExperienceRecommender'
]

__version__ = "1.0.0"