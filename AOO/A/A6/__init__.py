#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A6地缘政治监控器包

地缘政治风险监控系统，用于量化金融和投资管理

主要功能：
- 全球政治事件监控
- 贸易政策变化跟踪
- 制裁和关税影响分析
- 地缘政治风险评分
- 市场影响评估
- 风险预警系统

模块：
- GeopoliticalMonitor: 主监控器类
- 数据模型和枚举定义
- 数据源集成
- 风险评估算法

使用示例：
    from A6.GeopoliticalMonitor import GeopoliticalMonitor
    
    monitor = GeopoliticalMonitor()
    # ... 配置和使用监控器
"""

from .GeopoliticalMonitor import (
    GeopoliticalMonitor,
    EventCategory,
    EventSeverity,
    AssetClass,
    GeopoliticalEvent,
    RiskAssessment,
    MarketImpact,
    DataSource,
    NewsAPI,
    OfficialAnnouncements
)

__version__ = "1.0.0"
__author__ = "AI量化系统"
__email__ = "ai-quant-system@example.com"
__description__ = "A6地缘政治风险监控系统"

# 导出的公共接口
__all__ = [
    'GeopoliticalMonitor',
    'EventCategory', 
    'EventSeverity',
    'AssetClass',
    'GeopoliticalEvent',
    'RiskAssessment',
    'MarketImpact',
    'DataSource',
    'NewsAPI',
    'OfficialAnnouncements'
]

# 版本信息
VERSION_INFO = {
    'version': __version__,
    'author': __author__,
    'description': __description__,
    'features': [
        '全球政治事件监控',
        '贸易政策变化跟踪', 
        '制裁和关税影响分析',
        '地缘政治风险评分',
        '市场影响评估',
        '风险预警系统'
    ]
}

def get_version():
    """获取版本信息"""
    return __version__

def get_info():
    """获取系统信息"""
    return VERSION_INFO

def create_monitor(db_path=None):
    """创建监控器实例的便捷函数"""
    return GeopoliticalMonitor(db_path)

# 包初始化完成标识
__package_initialized__ = True