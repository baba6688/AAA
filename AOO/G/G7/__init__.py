#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G7执行监控器模块

实现实时执行监控的核心功能，包括：
1. 执行状态、警报级别、指标类型枚举
2. 执行上下文、性能指标、警报数据类
3. 执行报告、实时监控器
4. 性能评估器、异常检测器
5. 效果分析器、警报管理器
6. 报告生成器、优化建议器
7. 执行监控器主类
"""

from .ExecutionMonitor import (
    # 枚举类
    ExecutionStatus,
    AlertLevel,
    MetricType,
    # 核心数据类
    ExecutionContext,
    PerformanceMetric,
    Alert,
    ExecutionReport,
    # 监控和分析类
    RealTimeMonitor,
    PerformanceEvaluator,
    AnomalyDetector,
    EffectAnalyzer,
    AlertManager,
    ReportGenerator,
    OptimizationAdvisor,
    # 主监控器
    ExecutionMonitor
)

__version__ = "1.0.0"
__author__ = "G7 Team"

__all__ = [
    # 枚举类
    "ExecutionStatus",
    "AlertLevel",
    "MetricType",
    # 核心数据类
    "ExecutionContext",
    "PerformanceMetric",
    "Alert",
    "ExecutionReport",
    # 监控和分析类
    "RealTimeMonitor",
    "PerformanceEvaluator",
    "AnomalyDetector",
    "EffectAnalyzer",
    "AlertManager",
    "ReportGenerator",
    "OptimizationAdvisor",
    # 主监控器
    "ExecutionMonitor"
]