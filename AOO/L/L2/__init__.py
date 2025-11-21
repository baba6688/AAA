"""
L2交易日志记录器模块

这个模块提供了完整的L2交易日志记录功能，包括：
- 交易事件日志记录
- 性能监控和分析
- 错误跟踪和处理
- 策略分析和监控
- 市场数据记录
- 统计分析和报告
- 异步处理支持
- 高级分析和告警功能

主要类：
- TradingLogger: 基础交易日志记录器
- ExtendedTradingLogger: 扩展的交易日志记录器
- AdvancedAnalytics: 高级分析模块
- AlertManager: 告警管理器
- DataValidator: 数据验证器
- HealthMonitor: 健康监控器
"""

from .TradingLogger import (
    TradingLogger,
    ExtendedTradingLogger,
    AdvancedAnalytics,
    AlertManager,
    DataValidator,
    ConfigManager,
    HealthMonitor,
    DatabaseManager,
    AsyncLogProcessor,
    LogRotator,
    # 数据类
    TradeEvent,
    PerformanceMetrics,
    ErrorEvent,
    StrategyEvent,
    MarketDataEvent,
    StatisticsSummary,
    # 枚举类
    LogLevel,
    LogType,
    TradeEventType,
    OrderSide,
    OrderType
)

__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "L2交易日志记录器"

__all__ = [
    # 主要类
    "TradingLogger",
    "ExtendedTradingLogger", 
    "AdvancedAnalytics",
    "AlertManager",
    "DataValidator",
    "ConfigManager",
    "HealthMonitor",
    "DatabaseManager",
    "AsyncLogProcessor",
    "LogRotator",
    
    # 数据类
    "TradeEvent",
    "PerformanceMetrics", 
    "ErrorEvent",
    "StrategyEvent",
    "MarketDataEvent",
    "StatisticsSummary",
    
    # 枚举类
    "LogLevel",
    "LogType",
    "TradeEventType",
    "OrderSide",
    "OrderType"
]