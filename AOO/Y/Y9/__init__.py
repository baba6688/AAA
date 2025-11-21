#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y9存储状态聚合器包

一个综合的存储状态聚合器，用于收集、分析和监控多个存储模块的状态。
提供实时监控、预警机制、历史记录和可视化仪表板功能。

主要组件:
- StorageStatusAggregator: 主聚合器类
- StorageModule: 存储模块数据模型
- Alert: 预警数据模型
- StorageReport: 报告数据模型

作者: Y9存储团队
版本: 1.0.0
日期: 2025-11-06
"""

from .StorageStatusAggregator import (
    StorageStatusAggregator,
    StorageModule,
    StorageStatus,
    Alert,
    AlertLevel,
    StorageReport,
    DatabaseManager,
    StatusCollector,
    AlertManager,
    TrendAnalyzer,
    ReportGenerator,
    create_sample_collector
)

# 包版本信息
__version__ = "1.0.0"
__author__ = "Y9存储团队"
__email__ = "y9-storage@example.com"
__license__ = "MIT"

# 导出的公共接口
__all__ = [
    "StorageStatusAggregator",
    "StorageModule", 
    "StorageStatus",
    "Alert",
    "AlertLevel",
    "StorageReport",
    "DatabaseManager",
    "StatusCollector",
    "AlertManager",
    "TrendAnalyzer",
    "ReportGenerator",
    "create_sample_collector"
]

# 包级别的便捷函数
def create_aggregator(db_path="storage_status.db"):
    """
    便捷函数：创建存储状态聚合器实例
    
    Args:
        db_path (str): 数据库文件路径，默认为 "storage_status.db"
    
    Returns:
        StorageStatusAggregator: 聚合器实例
    """
    return StorageStatusAggregator(db_path)


def quick_start():
    """
    快速开始示例
    
    Returns:
        StorageStatusAggregator: 配置好的聚合器实例
    """
    aggregator = StorageStatusAggregator()
    
    # 注册示例收集器
    sample_collector = create_sample_collector()
    aggregator.status_collector.register_collector("sample", sample_collector)
    
    # 添加简单的预警处理器
    def simple_alert_handler(alert):
        print(f"预警: [{alert.level.value.upper()}] {alert.message}")
    
    aggregator.alert_manager.add_alert_handler(simple_alert_handler)
    
    return aggregator


# 包信息
package_info = {
    "name": "Y9存储状态聚合器",
    "version": __version__,
    "description": "综合的存储状态聚合和监控系统",
    "author": __author__,
    "license": __license__,
    "components": [
        "状态收集器 (StatusCollector)",
        "预警管理器 (AlertManager)", 
        "趋势分析器 (TrendAnalyzer)",
        "报告生成器 (ReportGenerator)",
        "数据库管理器 (DatabaseManager)"
    ],
    "features": [
        "多源数据收集",
        "实时监控",
        "智能预警",
        "趋势分析",
        "历史记录",
        "可视化仪表板",
        "多格式报告"
    ]
}