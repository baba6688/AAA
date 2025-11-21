#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
H4反馈处理器模块
实现反馈信息的收集、分类、分析、处理策略制定、效果评估和历史跟踪
"""

from .FeedbackProcessor import (
    FeedbackProcessor,
    FeedbackCollector,
    FeedbackClassifier,
    FeedbackAnalyzer,
    StrategyManager,
    EffectEvaluator,
    HistoryTracker,
    ReportGenerator,
    FeedbackItem,
    ProcessingStrategy,
    ProcessingResult,
    FeedbackType,
    Priority,
    Status
)

__all__ = [
    'FeedbackProcessor',
    'FeedbackCollector', 
    'FeedbackClassifier',
    'FeedbackAnalyzer',
    'StrategyManager',
    'EffectEvaluator',
    'HistoryTracker',
    'ReportGenerator',
    'FeedbackItem',
    'ProcessingStrategy',
    'ProcessingResult',
    'FeedbackType',
    'Priority',
    'Status'
]

__version__ = '1.0.0'
__author__ = 'H4 Feedback System'
__description__ = 'H4反馈处理器 - 实现反馈信息的智能收集、分类、分析和处理'