#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G2动态适配器模块

实现动态环境适配的核心功能，包括：
1. 环境状态监控和检测
2. 多种检测器（波动性、趋势变化、流动性等）
3. 适配策略（保守型、激进型、平衡型等）
4. 参数优化算法（遗传、梯度、贝叶斯等）
5. 学习引擎和效果评估
6. 早期预警系统
"""

from .DynamicAdaptor import (
    # 核心类和状态
    EnvironmentState,
    AdaptationAction,
    PerformanceMetrics,
    EnvironmentMonitor,
    LearningEngine,
    EffectEvaluator,
    HistoryTracker,
    EarlyWarningSystem,
    DynamicAdaptor,
    # 检测器类
    BaseDetector,
    VolatilityDetector,
    TrendChangeDetector,
    LiquidityDetector,
    RiskLevelDetector,
    PerformanceDetector,
    # 策略类
    AdaptationStrategy,
    BaseStrategy,
    ConservativeStrategy,
    AggressiveStrategy,
    BalancedStrategy,
    AdaptiveStrategy,
    EmergencyStrategy,
    # 优化器类
    ParameterOptimizer,
    BaseOptimizer,
    GeneticOptimizer,
    GradientOptimizer,
    BayesianOptimizer,
    GridSearchOptimizer
)

__version__ = "1.0.0"
__author__ = "G2 Team"

__all__ = [
    # 核心类
    "EnvironmentState",
    "AdaptationAction",
    "PerformanceMetrics",
    "EnvironmentMonitor",
    "LearningEngine",
    "EffectEvaluator",
    "HistoryTracker",
    "EarlyWarningSystem",
    "DynamicAdaptor",
    # 检测器类
    "BaseDetector",
    "VolatilityDetector",
    "TrendChangeDetector",
    "LiquidityDetector",
    "RiskLevelDetector",
    "PerformanceDetector",
    # 策略类
    "AdaptationStrategy",
    "BaseStrategy",
    "ConservativeStrategy",
    "AggressiveStrategy",
    "BalancedStrategy",
    "AdaptiveStrategy",
    "EmergencyStrategy",
    # 优化器类
    "ParameterOptimizer",
    "BaseOptimizer",
    "GeneticOptimizer",
    "GradientOptimizer",
    "BayesianOptimizer",
    "GridSearchOptimizer"
]