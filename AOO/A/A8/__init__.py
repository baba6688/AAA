# -*- coding: utf-8 -*-
"""
A8情绪指标分析器包
Sentiment Analyzer Package

提供市场情绪分析和监控功能
"""

from .SentimentAnalyzer import (
    SentimentAnalyzer,
    SentimentData,
    FearGreedIndex
)

__version__ = "1.0.0"
__author__ = "AI量化分析系统"

__all__ = [
    'SentimentAnalyzer',
    'SentimentData',
    'FearGreedIndex'
]