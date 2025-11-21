"""
I9模块 - 接口状态聚合器

该模块提供接口状态聚合功能，包括：
- 接口状态管理
- 告警级别处理
- 健康状态评估
- 接口指标统计
- 告警规则配置
- 告警管理
- 状态监控
- 性能聚合
- 预测分析
- 可视化引擎
- 报告生成

Author: AI Assistant
Date: 2025-11-05
Version: 1.0.0
"""

from .InterfaceStateAggregator import (
    InterfaceStatus,
    AlertLevel,
    HealthStatus,
    InterfaceMetrics,
    InterfaceInfo,
    AlertRule,
    Alert,
    DatabaseManager,
    HealthCalculator,
    AlertManager,
    StateMonitor,
    PerformanceAggregator,
    PredictiveAnalyzer,
    VisualizationEngine,
    ReportGenerator,
    InterfaceStateAggregator
)

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    "InterfaceStatus",
    "AlertLevel",
    "HealthStatus",
    "InterfaceMetrics",
    "InterfaceInfo",
    "AlertRule",
    "Alert",
    "DatabaseManager",
    "HealthCalculator",
    "AlertManager", 
    "StateMonitor",
    "PerformanceAggregator",
    "PredictiveAnalyzer",
    "VisualizationEngine",
    "ReportGenerator",
    "InterfaceStateAggregator"
]