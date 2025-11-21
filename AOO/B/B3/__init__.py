"""
B3趋势分析器包
Trend Analyzer
"""

from .TrendAnalyzer import (
    TrendDirection,      # 趋势方向枚举
    TrendStrength,       # 趋势强度枚举
    SignalType,          # 信号类型枚举
    TrendData,           # 趋势数据类
    TrendAnalysis,       # 趋势分析类
    TradingSignal,       # 交易信号
    TrendAlert,          # 趋势告警
    TrendAnalyzer        # 趋势分析器主类
)

__version__ = "1.0.0"
__author__ = "B3 Team"

__all__ = [
    'TrendDirection',
    'TrendStrength',
    'SignalType',
    'TrendData',
    'TrendAnalysis',
    'TradingSignal',
    'TrendAlert',
    'TrendAnalyzer'
]