"""
M9监控状态聚合器模块

该模块提供了统一的监控状态聚合和管理功能，包括：
- 聚合M1-M8所有监控模块的状态数据
- 提供统一的监控状态管理接口
- 生成综合监控报告和仪表板
- 管理跨模块的监控协调和告警
- 支持监控历史数据查询和分析

主要类：
- MonitoringStateAggregator: 主聚合器类
- AggregateStatus: 聚合状态枚举
- AlertSeverity: 告警严重级别
- MonitoringMetrics: 监控指标数据
- AggregateAlert: 聚合告警数据
- SystemOverview: 系统概览数据
- MonitoringState: 监控状态数据

版本: 1.0.0
作者: M9监控状态聚合器团队
创建时间: 2025-11-13
"""

from .MonitoringStateAggregator import (
    MonitoringStateAggregator,
    AggregateStatus,
    AlertSeverity,
    MonitoringMetrics,
    AggregateAlert,
    SystemOverview,
    MonitoringState
)

__all__ = [
    'MonitoringStateAggregator',
    'AggregateStatus',
    'AlertSeverity',
    'MonitoringMetrics',
    'AggregateAlert',
    'SystemOverview',
    'MonitoringState'
]

__version__ = '1.0.0'
__author__ = 'M9监控状态聚合器团队'