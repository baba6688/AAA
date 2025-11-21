#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G5仓位管理器模块

实现智能仓位管理和优化的核心功能，包括：
1. 仓位计算和优化
2. 仓位监控和调整
3. 仓位风险评估
4. 仓位组合优化
5. 仓位历史跟踪
6. 仓位报告和分析
7. 仓位策略优化
"""

from .PositionManager import (
    PositionManager,
    Position,
    RiskMetrics,
    PositionAdjustment,
    PositionOptimizer
)

__version__ = "1.0.0"
__author__ = "G5 Team"

__all__ = [
    "PositionManager",
    "Position", 
    "RiskMetrics",
    "PositionAdjustment",
    "PositionOptimizer"
]