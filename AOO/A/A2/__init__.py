# -*- coding: utf-8 -*-
"""
A2经济指标监控器包
Economic Indicator Monitor Package

提供经济数据监控、分析和预警功能
"""

from .EconomicIndicatorMonitor import (
    EconomicIndicatorMonitor,
    EconomicIndicator,
    AlertRule
)

__version__ = "1.0.0"
__author__ = "AI量化分析系统"

__all__ = [
    'EconomicIndicatorMonitor',
    'EconomicIndicator', 
    'AlertRule'
]