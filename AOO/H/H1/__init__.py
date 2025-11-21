#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H1深度反思模块

实现深度反思系统的核心功能，包括：
1. 反思内容收集和管理
2. 反思模式识别和分析
3. 反思洞察生成和评估
4. 反思经验提取和学习
5. 反思效果跟踪和验证
6. 反思历史管理和报告生成
"""

from .DeepReflection import (
    # 反思内容相关
    ReflectionContent,
    ReflectionContentCollector,
    # 反思模式相关
    ReflectionPattern,
    ReflectionPatternRecognizer,
    # 反思洞察相关
    ReflectionInsight,
    ReflectionEvaluator,
    ReflectionValidator,
    # 反思经验相关
    ReflectionExperience,
    ReflectionExperienceExtractor,
    # 反思跟踪相关
    ReflectionEffectTracker,
    # 反思管理相关
    ReflectionHistoryManager,
    ReflectionReportGenerator,
    # 主反思引擎
    DeepReflection
)

__version__ = "1.0.0"
__author__ = "H1 Team"

__all__ = [
    # 反思内容相关
    "ReflectionContent",
    "ReflectionContentCollector",
    # 反思模式相关
    "ReflectionPattern",
    "ReflectionPatternRecognizer",
    # 反思洞察相关
    "ReflectionInsight",
    "ReflectionEvaluator",
    "ReflectionValidator",
    # 反思经验相关
    "ReflectionExperience",
    "ReflectionExperienceExtractor",
    # 反思跟踪相关
    "ReflectionEffectTracker",
    # 反思管理相关
    "ReflectionHistoryManager",
    "ReflectionReportGenerator",
    # 主反思引擎
    "DeepReflection"
]