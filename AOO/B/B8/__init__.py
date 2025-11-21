"""
B8成交量分析器包
Volume Analyzer
"""

from .VolumeAnalyzer import (
    VolumePattern,        # 成交量模式枚举
    VolumeAnomaly,        # 成交量异常枚举
    DivergenceType,       # 背离类型枚举
    VolumeSignal,         # 成交量信号
    VolumeAnalysisResult, # 成交量分析结果
    VolumeAnalyzer        # 成交量分析器主类
)

__version__ = "1.0.0"
__author__ = "B8 Team"

__all__ = [
    'VolumePattern',
    'VolumeAnomaly',
    'DivergenceType',
    'VolumeSignal',
    'VolumeAnalysisResult',
    'VolumeAnalyzer'
]