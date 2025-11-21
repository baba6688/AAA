"""
J3时间序列工具模块

该模块提供完整的时间序列分析、预测、变换和特征提取功能。
包含异步处理和批处理支持，适用于大规模时间序列数据分析。

主要组件：
- TimeSeriesAnalyzer: 时间序列分析工具
- TimeSeriesForecaster: 时间序列预测工具  
- TimeSeriesTransformer: 时间序列变换工具
- TimeSeriesFeatureExtractor: 时间序列特征提取工具
- MultiTimeScaleProcessor: 多时间尺度处理工具
- AsyncTimeSeriesProcessor: 异步处理工具
- BatchTimeSeriesProcessor: 批处理工具

Author: AI Assistant
Date: 2025-11-06
"""

from .TimeSeriesTools import (
    TimeSeriesAnalyzer,
    TimeSeriesForecaster, 
    TimeSeriesTransformer,
    TimeSeriesFeatureExtractor,
    MultiTimeScaleProcessor,
    AsyncTimeSeriesProcessor,
    BatchTimeSeriesProcessor,
    TimeSeriesError,
    TimeSeriesValidationError,
    TimeSeriesProcessingError
)

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    "TimeSeriesAnalyzer",
    "TimeSeriesForecaster",
    "TimeSeriesTransformer", 
    "TimeSeriesFeatureExtractor",
    "MultiTimeScaleProcessor",
    "AsyncTimeSeriesProcessor",
    "BatchTimeSeriesProcessor",
    "TimeSeriesError",
    "TimeSeriesValidationError",
    "TimeSeriesProcessingError"
]