#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H9进化状态聚合器模块

实现进化状态聚合系统的核心功能，包括：
1. 进化状态、优先级级别枚举
2. 模块状态和进化指标数据类
3. 进化报告生成和管理
4. 进化状态聚合器主类
"""

from .EvolutionStateAggregator import (
    # 枚举类
    EvolutionStatus,
    PriorityLevel,
    # 核心数据类
    ModuleState,
    EvolutionMetrics,
    EvolutionReport,
    # 主聚合器
    EvolutionStateAggregator
)

__version__ = "1.0.0"
__author__ = "H9 Team"

__all__ = [
    # 枚举类
    "EvolutionStatus",
    "PriorityLevel",
    # 核心数据类
    "ModuleState",
    "EvolutionMetrics",
    "EvolutionReport",
    # 主聚合器
    "EvolutionStateAggregator"
]