#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H2认知进化模块

实现认知进化系统的核心功能，包括：
1. 进化类型和状态枚举管理
2. 认知模式、能力、结构数据类
3. 进化记录和历史跟踪
4. 认知进化分析器和模式演化器
5. 认知能力优化器和结构重组器
6. 自适应进化引擎和效果评估器
7. 进化策略优化器和历史跟踪器
"""

from .CognitiveEvolution import (
    # 枚举类
    EvolutionType,
    EvolutionStatus,
    # 核心数据类
    CognitivePattern,
    CognitiveCapability,
    CognitiveStructure,
    EvolutionRecord,
    # 分析器和演化器
    CognitiveEvolutionAnalyzer,
    CognitivePatternEvolver,
    CognitiveCapabilityOptimizer,
    CognitiveStructureReorganizer,
    # 进化引擎和评估器
    AdaptiveEvolutionEngine,
    EvolutionEffectivenessEvaluator,
    # 跟踪器和优化器
    EvolutionHistoryTracker,
    EvolutionStrategyOptimizer,
    # 主进化引擎
    CognitiveEvolution
)

__version__ = "1.0.0"
__author__ = "H2 Team"

__all__ = [
    # 枚举类
    "EvolutionType",
    "EvolutionStatus",
    # 核心数据类
    "CognitivePattern",
    "CognitiveCapability",
    "CognitiveStructure",
    "EvolutionRecord",
    # 分析器和演化器
    "CognitiveEvolutionAnalyzer",
    "CognitivePatternEvolver",
    "CognitiveCapabilityOptimizer",
    "CognitiveStructureReorganizer",
    # 进化引擎和评估器
    "AdaptiveEvolutionEngine",
    "EvolutionEffectivenessEvaluator",
    # 跟踪器和优化器
    "EvolutionHistoryTracker",
    "EvolutionStrategyOptimizer",
    # 主进化引擎
    "CognitiveEvolution"
]