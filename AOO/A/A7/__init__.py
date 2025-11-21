# -*- coding: utf-8 -*-
"""
A7技术指标计算器包
Technical Indicator Calculator Package

提供全面的技术指标计算和分析功能
"""

from .TechnicalIndicatorCalculator import (
    TechnicalIndicatorCalculator,
    IndicatorSignal,
    IndicatorConfig,
    IndicatorCache,
    BaseIndicator,
    BasicIndicators,
    AdvancedIndicators,
    CustomIndicator,
    IndicatorCombination
)

__version__ = "1.0.0"
__author__ = "AI量化分析系统"

__all__ = [
    'TechnicalIndicatorCalculator',
    'IndicatorSignal',
    'IndicatorConfig',
    'IndicatorCache',
    'BaseIndicator',
    'BasicIndicators',
    'AdvancedIndicators', 
    'CustomIndicator',
    'IndicatorCombination'
]