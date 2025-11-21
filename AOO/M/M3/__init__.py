#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M3模块初始化文件

包含资源监控器相关的类和功能
"""

from .ResourceMonitor import (
    ResourceMonitor,
    ResourceMetrics,
    AlertRule,
    AlertLevel,
    ResourceType,
    CostAnalysis,
    create_default_alert_rules,
    alert_callback
)

__version__ = "1.0.0"
__author__ = "M3系统"
__all__ = [
    "ResourceMonitor",
    "ResourceMetrics", 
    "AlertRule",
    "AlertLevel",
    "ResourceType",
    "CostAnalysis",
    "create_default_alert_rules",
    "alert_callback"
]