#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C4 关联分析器模块

实现多种关联分析方法，包括线性、非线性、时间序列等关联分析功能

主要功能:
- 多变量关联分析（Pearson、Spearman、Kendall）
- 时间序列关联分析（滞后相关、动态相关）
- 非线性关联检测（互信息、距离相关、Copula相关）
- 滞后关联分析
- 条件关联分析（偏相关、格兰杰因果）
- 关联强度评估和排序
- 关联模式发现和解释


创建时间: 2025-11-05
"""

from .CorrelationAnalyzer import CorrelationAnalyzer

__version__ = "1.0.0"
__author__ = "AI量化系统"

__all__ = ['CorrelationAnalyzer']