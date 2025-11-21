#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G1智能决策引擎模块

实现智能决策系统的核心功能，包括：
1. 决策标准和选项管理
2. 决策结果生成和评估
3. 风险评估和不确定性处理
4. 效果预测和学习引擎
5. 决策解释和说明
"""

from .IntelligentDecisionEngine import (
    DecisionCriteria,
    DecisionOption,
    DecisionResult,
    RiskAssessment,
    UncertaintyHandler,
    RiskAssessmentEngine,
    EffectPredictionModel,
    LearningEngine,
    ExplanationEngine,
    IntelligentDecisionEngine
)

__version__ = "1.0.0"
__author__ = "G1 Team"

__all__ = [
    "DecisionCriteria",
    "DecisionOption", 
    "DecisionResult",
    "RiskAssessment",
    "UncertaintyHandler",
    "RiskAssessmentEngine",
    "EffectPredictionModel",
    "LearningEngine",
    "ExplanationEngine",
    "IntelligentDecisionEngine"
]