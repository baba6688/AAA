# -*- coding: utf-8 -*-
"""
A区：外部环境层
External Environment Layer

功能模块：
A1: 市场数据采集器
A2: 经济指标监控器  
A3: 新闻事件处理器
A4: 交易所连接器
A5: 宏观经济分析器
A6: 地缘政治监控器
A7: 技术指标计算器
A8: 情绪指标分析器
A9: 环境状态聚合器
"""

from A.A1 import (
    MarketDataCollector,
    MarketData,
    ExchangeConfig as A1ExchangeConfig,
    DataValidator,
    DataCache,
    DataStorage,
    WebSocketManager
)
from A.A2 import (
    EconomicIndicatorMonitor,
    EconomicIndicator,
    AlertRule
)
from A.A3 import (
    NewsEventProcessor,
    NewsItem,
    Event,
    NewsAPIClient
)
from A.A4 import (
    ExchangeConnector,
    ExchangeType,
    ExchangeStatus,
    APIKey,
    ExchangeConfig,
    HealthStatus,
    RateLimiter,
    ConnectionPool,
    ExchangeConnector as BaseExchangeConnector,
    BinanceConnector,
    OKXConnector,
    HuobiConnector,
    GateConnector,
    ExchangeManager,
    DataAggregator
)
from A.A5 import (
    MacroEconomicAnalyzer,
    create_sample_data
)
from A.A6 import (
    GeopoliticalMonitor,
    EventCategory,
    EventSeverity,
    AssetClass,
    GeopoliticalEvent,
    RiskAssessment
)
from A.A7 import (
    TechnicalIndicatorCalculator,
    IndicatorSignal,
    IndicatorConfig,
    IndicatorCache,
    BaseIndicator,
    BasicIndicators,
    AdvancedIndicators,
    CustomIndicator,
    IndicatorCombination
)
from A.A8 import (
    SentimentAnalyzer,
    SentimentData,
    FearGreedIndex
)
from A.A9 import (
    EnvironmentStateAggregator,
    MarketEnvironment,
    TrendDirection,
    DataSource,
    EnvironmentMetrics,
    EnvironmentState,
    DataFusionEngine
)

# 版本信息
__version__ = "1.0.0"
__author__ = "AI量化分析系统"

# 完整导出接口
__all__ = [
    # A1 - 市场数据采集器
    'MarketDataCollector',
    'MarketData',
    'A1ExchangeConfig',
    'DataValidator',
    'DataCache',
    'DataStorage',
    'WebSocketManager',
    
    # A2 - 经济指标监控器
    'EconomicIndicatorMonitor',
    'EconomicIndicator',
    'AlertRule',
    
    # A3 - 新闻事件处理器
    'NewsEventProcessor',
    'NewsItem',
    'Event',
    'NewsAPIClient',
    
    # A4 - 交易所连接器
    'ExchangeConnector',
    'ExchangeType',
    'ExchangeStatus',
    'APIKey',
    'ExchangeConfig',
    'HealthStatus',
    'RateLimiter',
    'ConnectionPool',
    'BaseExchangeConnector',
    'BinanceConnector',
    'OKXConnector',
    'HuobiConnector',
    'GateConnector',
    'ExchangeManager',
    'DataAggregator',
    
    # A5 - 宏观经济分析器
    'MacroEconomicAnalyzer',
    'create_sample_data',
    
    # A6 - 地缘政治监控器
    'GeopoliticalMonitor',
    'EventCategory',
    'EventSeverity',
    'AssetClass',
    'GeopoliticalEvent',
    'RiskAssessment',
    
    # A7 - 技术指标计算器
    'TechnicalIndicatorCalculator',
    'IndicatorSignal',
    'IndicatorConfig',
    'IndicatorCache',
    'BaseIndicator',
    'BasicIndicators',
    'AdvancedIndicators',
    'CustomIndicator',
    'IndicatorCombination',
    
    # A8 - 情绪指标分析器
    'SentimentAnalyzer',
    'SentimentData',
    'FearGreedIndex',
    
    # A9 - 环境状态聚合器
    'EnvironmentStateAggregator',
    'MarketEnvironment',
    'TrendDirection',
    'DataSource',
    'EnvironmentMetrics',
    'EnvironmentState',
    'DataFusionEngine'
]