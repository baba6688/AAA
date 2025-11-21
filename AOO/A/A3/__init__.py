"""
A3新闻事件处理器包
提供新闻数据抓取、情感分析、事件分类和实时处理功能
"""

from .NewsEventProcessor import (
    NewsEventProcessor,
    NewsItem,
    Event,
    NewsAPIClient,
    AlphaVantageNewsClient,
    SentimentAnalyzer,
    KeywordExtractor,
    EventClassifier,
    NewsDeduplicator,
    ImpactAnalyzer,
    TimelineBuilder
)

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    "NewsEventProcessor",
    "NewsItem", 
    "Event",
    "NewsAPIClient",
    "AlphaVantageNewsClient",
    "SentimentAnalyzer",
    "KeywordExtractor",
    "EventClassifier",
    "NewsDeduplicator",
    "ImpactAnalyzer",
    "TimelineBuilder"
]