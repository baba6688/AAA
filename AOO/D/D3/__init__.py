"""
D3 能力边界评估器模块
实现能力边界评估、能力差距识别、开发潜力分析等功能

主要类:
- CapabilityType: 能力类型枚举
- BoundaryType: 边界类型枚举
- CapabilityMetrics: 能力指标
- BoundaryDefinition: 边界定义
- CapabilityGap: 能力差距
- DevelopmentPotential: 开发潜力
- CapabilityBoundaryAssessor: 能力边界评估器

版本: 1.0.0
作者: AI量化系统
"""

from .CapabilityBoundaryAssessor import (
    CapabilityType,
    BoundaryType,
    CapabilityMetrics,
    BoundaryDefinition,
    CapabilityGap,
    DevelopmentPotential,
    CapabilityBoundaryAssessor
)

__version__ = "1.0.0"
__author__ = "AI量化系统"

__all__ = [
    'CapabilityType',
    'BoundaryType',
    'CapabilityMetrics',
    'BoundaryDefinition',
    'CapabilityGap',
    'DevelopmentPotential',
    'CapabilityBoundaryAssessor'
]