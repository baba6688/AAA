#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L4性能监控模块

该模块提供了完整的性能监控、日志记录、告警和报告功能。

主要组件:
- PerformanceLogger: 主要的性能日志记录器类
- SystemPerformanceMonitor: 系统性能监控器
- ApplicationPerformanceMonitor: 应用性能监控器
- DatabasePerformanceMonitor: 数据库性能监控器
- NetworkPerformanceMonitor: 网络性能监控器
- AlertManager: 告警管理器
- TrendAnalyzer: 趋势分析器
- AsyncLogProcessor: 异步日志处理器

使用示例:
    from L.L4.PerformanceLogger import PerformanceLogger, PerformanceMetric, AlertLevel
    
    # 创建性能日志记录器
    logger = PerformanceLogger(
        name="MyAppLogger",
        enable_system_monitor=True,
        enable_application_monitor=True
    )
    
    # 启动监控
    logger.start()
    
    # 记录应用请求
    logger.record_application_request(
        response_time=150.0,
        status_code=200,
        method="GET",
        endpoint="/api/users"
    )
    
    # 添加告警规则
    logger.add_alert_rule(
        metric=PerformanceMetric.CPU_USAGE,
        threshold=80.0,
        comparison_operator='>',
        level=AlertLevel.WARNING,
        description="CPU使用率过高"
    )
    
    # 生成报告
    report = logger.generate_summary_report(time_range_hours=24.0)
    logger.export_report(report, "performance_report.json")

"""

from .PerformanceLogger import (
    # 主要类
    PerformanceLogger,
    
    # 监控器类
    SystemPerformanceMonitor,
    ApplicationPerformanceMonitor,
    DatabasePerformanceMonitor,
    NetworkPerformanceMonitor,
    
    # 管理器类
    AlertManager,
    TrendAnalyzer,
    AsyncLogProcessor,
    NotificationManager,
    PerformanceReportGenerator,
    
    # 数据结构
    PerformanceData,
    AlertRule,
    Alert,
    TrendAnalysis,
    
    # 枚举
    PerformanceMetric,
    AlertLevel,
    LogLevel,
    DatabaseType,
    NetworkProtocol
)

__version__ = "1.0.0"
__author__ = "AI系统"

__all__ = [
    # 主要类
    "PerformanceLogger",
    
    # 监控器类
    "SystemPerformanceMonitor",
    "ApplicationPerformanceMonitor", 
    "DatabasePerformanceMonitor",
    "NetworkPerformanceMonitor",
    
    # 管理器类
    "AlertManager",
    "TrendAnalyzer",
    "AsyncLogProcessor",
    "NotificationManager",
    "PerformanceReportGenerator",
    
    # 数据结构
    "PerformanceData",
    "AlertRule",
    "Alert",
    "TrendAnalysis",
    
    # 枚举
    "PerformanceMetric",
    "AlertLevel",
    "LogLevel",
    "DatabaseType",
    "NetworkProtocol"
]