"""
D8 适应性评估器模块
实现适应性评估、环境变化分析、适应策略建议等功能

主要类:
- AdaptabilityType: 适应性类型枚举
- EnvironmentType: 环境类型枚举
- AdaptationStrategy: 适应策略枚举
- AdaptabilityMetrics: 适应性指标
- EnvironmentChange: 环境变化
- AdaptationPerformance: 适应性能
- AdaptationSuggestion: 适应建议
- AdaptabilityModel: 适应性模型
- AdaptabilityAssessor: 适应性评估器

版本: 1.0.0
作者: AI量化系统
"""

from .AdaptabilityAssessor import (
    AdaptabilityType,
    EnvironmentType,
    AdaptationStrategy,
    AdaptabilityMetrics,
    EnvironmentChange,
    AdaptationPerformance,
    AdaptationSuggestion,
    AdaptabilityModel,
    AdaptabilityAssessor
)

__version__ = "1.0.0"
__author__ = "AI量化系统"

__all__ = [
    'AdaptabilityType',
    'EnvironmentType',
    'AdaptationStrategy',
    'AdaptabilityMetrics',
    'EnvironmentChange',
    'AdaptationPerformance',
    'AdaptationSuggestion',
    'AdaptabilityModel',
    'AdaptabilityAssessor'
]