# -*- coding: utf-8 -*-
"""
宏观经济分析器模块
================

提供宏观经济分析功能，包括经济周期识别、相关性分析、
政策影响评估、风险预警、跨市场分析和预测建模。

主要类:
- MacroEconomicAnalyzer: 宏观经济分析器主类


"""

from .MacroEconomicAnalyzer import MacroEconomicAnalyzer, create_sample_data

__version__ = "1.0.0"
__author__ = "AI量化分析系统"

__all__ = [
    'MacroEconomicAnalyzer',
    'create_sample_data'
]