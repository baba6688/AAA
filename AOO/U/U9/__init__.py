"""
U9算法状态聚合器模块

该模块提供算法状态聚合和管理功能，包括状态收集、性能监控、
使用统计、效果评估、资源监控等功能。

主要类和函数:
- AlgorithmStateAggregator: 算法状态聚合器主类
- AlgorithmMetrics: 算法性能指标数据类
- AlgorithmConfig: 算法配置数据类
- AlgorithmUsageStats: 算法使用统计数据类
- AlgorithmEffectiveness: 算法效果评估数据类
- AlgorithmStatus: 算法状态枚举
- HealthStatus: 健康状态枚举
- run_tests: 测试函数
"""

from .AlgorithmStateAggregator import (
    AlgorithmStateAggregator,
    AlgorithmMetrics,
    AlgorithmConfig,
    AlgorithmUsageStats,
    AlgorithmEffectiveness,
    AlgorithmStatus,
    HealthStatus,
    run_tests
)

__version__ = "1.0.0"
__author__ = "U9系统"
__email__ = "u9-system@company.com"
__description__ = "U9算法状态聚合器模块，提供算法监控和管理功能"

__all__ = [
    "AlgorithmStateAggregator",
    "AlgorithmMetrics", 
    "AlgorithmConfig",
    "AlgorithmUsageStats",
    "AlgorithmEffectiveness",
    "AlgorithmStatus",
    "HealthStatus",
    "run_tests"
]