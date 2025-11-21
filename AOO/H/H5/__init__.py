#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H5进化评估器模块

实现进化评估系统的核心功能，包括：
1. 评估类型、进化阶段、风险级别枚举
2. 进化指标和评估结果数据类
3. 进化报告和历史分析器
4. 效果评估器、质量评估器
5. 适应性评估器、效率评估器
6. 风险评估器和进化评估器主类
"""

from .EvolutionEvaluator import (
    # 枚举类
    EvaluationType,
    EvolutionStage,
    RiskLevel,
    # 核心数据类
    EvolutionMetrics,
    EvaluationResult,
    EvolutionReport,
    # 主评估器
    EvolutionEvaluator,
    # 专项评估器
    EffectivenessEvaluator,
    QualityEvaluator,
    AdaptabilityEvaluator,
    EfficiencyEvaluator,
    RiskEvaluator,
    # 历史分析器
    HistoricalAnalyzer
)

__version__ = "1.0.0"
__author__ = "H5 Team"

__all__ = [
    # 枚举类
    "EvaluationType",
    "EvolutionStage",
    "RiskLevel",
    # 核心数据类
    "EvolutionMetrics",
    "EvaluationResult",
    "EvolutionReport",
    # 主评估器
    "EvolutionEvaluator",
    # 专项评估器
    "EffectivenessEvaluator",
    "QualityEvaluator",
    "AdaptabilityEvaluator",
    "EfficiencyEvaluator",
    "RiskEvaluator",
    # 历史分析器
    "HistoricalAnalyzer"
]