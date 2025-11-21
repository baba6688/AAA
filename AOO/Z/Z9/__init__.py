"""
Z9扩展状态聚合器模块

该模块提供了完整的扩展状态聚合功能，包括：
- 状态收集器：从各个扩展模块收集状态信息
- 数据聚合：聚合多个扩展模块的结果
- 状态分析：分析扩展状态和趋势
- 报告生成：生成综合扩展状态报告
- 状态监控：实时监控扩展状态
- 预警机制：扩展异常时预警
- 历史记录：保存历史扩展状态
- 仪表板：提供可视化的扩展状态仪表板

主要类：
- ExtensionStatusAggregator: 主要的扩展状态聚合器
- StatusCollector: 状态收集器
- DataAggregator: 数据聚合器
- StatusAnalyzer: 状态分析器
- ReportGenerator: 报告生成器
- StatusMonitor: 状态监控器
- AlertManager: 预警管理器
- HistoryManager: 历史记录管理器
- DashboardManager: 仪表板管理器

版本: 1.0.0
作者: Z9开发团队
"""

from .ExtensionStatusAggregator import (
    ExtensionStatusAggregator,
    StatusCollector,
    DataAggregator,
    StatusAnalyzer,
    ReportGenerator,
    StatusMonitor,
    AlertManager,
    HistoryManager,
    DashboardManager
)

__version__ = "1.0.0"
__author__ = "Z9开发团队"

__all__ = [
    "ExtensionStatusAggregator",
    "StatusCollector",
    "DataAggregator", 
    "StatusAnalyzer",
    "ReportGenerator",
    "StatusMonitor",
    "AlertManager",
    "HistoryManager",
    "DashboardManager"
]