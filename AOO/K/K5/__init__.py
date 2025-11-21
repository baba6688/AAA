"""
K5交易配置管理器模块

该模块提供了完整的交易配置管理系统，支持多种交易配置类型。

主要导出的类和功能：
- TradingConfigurationManager: 主要配置管理器
- 配置相关类：TradingMode, TimeRange, VolumeConfig, APIConfig等
- 验证器：ConfigurationValidator
- 备份管理器：ConfigurationBackupManager
- 监控器：ConfigurationMonitor
- 分析类：RiskManager, PerformanceAnalyzer, AdvancedTradingMetrics
- 示例和测试函数
"""

# 导入主要配置类和枚举
from .TradingConfigurationManager import (
    # 主要管理器类
    TradingConfigurationManager,
    
    # 基础配置类
    TradingMode,
    TimeRange,
    VolumeConfig,
    FrequencyConfig,
    APIConfig,
    ExchangeConfig,
    OrderRoutingConfig,
    
    # 限制配置类
    PositionLimit,
    CapitalLimit,
    TimeLimit,
    TradingLimits,
    
    # 成本配置类
    CommissionConfig,
    SlippageConfig,
    SpreadConfig,
    TradingCosts,
    
    # 执行配置类
    ExecutionParameters,
    MonitoringConfig,
    ExecutionConfig,
    ReportConfig,
    LoggingConfig,
    
    # 验证和管理类
    ConfigurationValidator,
    ConfigurationBackupManager,
    ConfigurationMonitor,
    AdvancedTradingMetrics,
    RiskManager,
    PerformanceAnalyzer,
    
    # 示例和测试函数
    sync_example_usage,
    comprehensive_test_suite
)

# 版本信息
__version__ = "1.0.0"
__author__ = "K5交易系统"

# 公开的API列表
__all__ = [
    # 主要类
    "TradingConfigurationManager",
    
    # 基础配置类
    "TradingMode",
    "TimeRange", 
    "VolumeConfig",
    "FrequencyConfig",
    "APIConfig",
    "ExchangeConfig",
    "OrderRoutingConfig",
    
    # 限制配置类
    "PositionLimit",
    "CapitalLimit", 
    "TimeLimit",
    "TradingLimits",
    
    # 成本配置类
    "CommissionConfig",
    "SlippageConfig",
    "SpreadConfig",
    "TradingCosts",
    
    # 执行配置类
    "ExecutionParameters",
    "MonitoringConfig", 
    "ExecutionConfig",
    "ReportConfig",
    "LoggingConfig",
    
    # 验证和管理类
    "ConfigurationValidator",
    "ConfigurationBackupManager",
    "ConfigurationMonitor",
    "AdvancedTradingMetrics",
    "RiskManager",
    "PerformanceAnalyzer",
    
    # 示例和测试函数
    "sync_example_usage",
    "comprehensive_test_suite"
]