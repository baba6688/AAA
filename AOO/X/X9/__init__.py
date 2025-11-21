#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X9缓存状态聚合器模块

这是一个完整的缓存状态聚合系统，提供了全面的缓存监控和管理功能。

主要功能：
- 状态收集：从各个缓存模块收集状态信息
- 数据聚合：聚合多个缓存模块的结果
- 状态分析：分析缓存状态和趋势
- 预警管理：缓存异常时预警
- 历史记录：保存历史缓存状态
- 报告生成：生成综合缓存状态报告
- 仪表板：提供可视化的缓存状态仪表板

类结构：
- CacheStatus: 缓存状态数据类
- AlertRule: 预警规则数据类  
- Alert: 预警数据类
- StatusCollector: 状态收集器
- DataAggregator: 数据聚合器
- StatusAnalyzer: 状态分析器
- AlertManager: 预警管理器
- HistoryManager: 历史记录管理器
- ReportGenerator: 报告生成器
- Dashboard: 仪表板类
- CacheStatusAggregator: 主聚合器类
- ExampleCacheModule: 示例缓存模块

版本: 1.0.0
作者: X9开发团队
日期: 2025-11-14

使用示例:
    >>> from X9 import CacheStatusAggregator, CacheStatus
    >>> 
    >>> # 创建聚合器实例
    >>> aggregator = CacheStatusAggregator()
    >>> 
    >>> # 获取当前状态
    >>> status = aggregator.get_current_status()
    >>> print(f"整体健康分数: {status['overall_health_score']:.1f}%")
    >>> 
    >>> # 生成报告
    >>> report = aggregator.generate_report('summary')
    >>> print(report)
"""

# 从CacheStatusAggregator模块导入所有主要类
from .CacheStatusAggregator import (
    CacheStatus,
    AlertRule, 
    Alert,
    StatusCollector,
    DataAggregator,
    StatusAnalyzer,
    AlertManager,
    HistoryManager,
    ReportGenerator,
    Dashboard,
    CacheStatusAggregator,
    ExampleCacheModule
)

# 定义模块导出列表
__all__ = [
    # 数据类
    'CacheStatus',
    'AlertRule',
    'Alert',
    
    # 组件类
    'StatusCollector',
    'DataAggregator', 
    'StatusAnalyzer',
    'AlertManager',
    'HistoryManager',
    'ReportGenerator',
    'Dashboard',
    
    # 主类
    'CacheStatusAggregator',
    
    # 示例类
    'ExampleCacheModule'
]

# 模块版本信息
__version__ = '1.0.0'
__author__ = 'X9开发团队'

# 模块级文档
__doc__ = """
X9缓存状态聚合器模块

提供完整的缓存状态监控、聚合、分析和报告功能。
包括实时监控、预警机制、历史数据管理和可视化仪表板。
"""

print(f"X9缓存状态聚合器模块已加载 (v{__version__})")