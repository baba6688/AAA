"""
M6 交易监控器 (Trading Monitor)

该模块提供全面的交易监控和分析功能，用于实时监控交易活动、
分析交易性能、管理合规性，并生成详细的监控报告。

主要功能包括:
- 交易记录管理和存储
- 实时交易监控和告警
- 交易性能指标分析
- 策略表现监控
- 合规性检查和报告
- 多种格式的报告导出

版本: 1.0.0
创建时间: 2025-11-05
作者: M6 Trading System
"""

from datetime import datetime

# 直接从TradingMonitor模块导入所有需要的类
from .TradingMonitor import (
    # 枚举类
    TradeStatus,
    TradeType,
    ComplianceLevel,
    
    # 数据类
    TradeRecord,
    MonitoringMetrics,
    AlertConfig,
    
    # 主类
    TradingMonitor,
    
    # 测试函数
    create_sample_trades,
    test_trading_monitor
)

# 模块元信息
__version__ = "1.0.0"
__author__ = "M6 Trading System"
__email__ = "support@m6-trading.com"
__license__ = "MIT"
__copyright__ = "Copyright 2025 M6 Trading System"
__description__ = "M6 交易监控器 - 全面的交易监控和分析系统"

# 导出的公共接口
__all__ = [
    # 枚举类
    "TradeStatus",
    "TradeType", 
    "ComplianceLevel",
    
    # 数据类
    "TradeRecord",
    "MonitoringMetrics",
    "AlertConfig",
    
    # 主类
    "TradingMonitor",
    
    # 便捷函数
    "create_monitor",
    "create_trade",
    "create_default_config",
    "create_strict_config",
    "create_lenient_config",
    "quick_demo",
    
    # 工具函数
    "get_version",
    "get_module_info",
    "validate_trade_type",
    "validate_trade_status", 
    "validate_compliance_level",
    "format_trade_summary",
    "calculate_pnl",
    "create_sample_trades",
    "test_trading_monitor",
    
    # 常量
    "SUPPORTED_SYMBOLS",
    "SUPPORTED_STRATEGIES",
    "DEFAULT_THRESHOLDS",
    "LOG_LEVELS",
    
    # 版本信息
    "__version__",
    "__author__",
    "__description__"
]

# ==================== 便捷函数 ====================

def create_monitor(
    max_history_size: int = 10000,
    success_rate_threshold: float = 0.95,
    latency_threshold: float = 1000.0,
    error_rate_threshold: float = 0.05,
    volume_spike_threshold: float = 2.0,
    compliance_threshold: float = 0.9,
    log_level: str = "INFO"
) -> TradingMonitor:
    """
    创建交易监控器的便捷函数
    
    Args:
        max_history_size: 最大历史记录保存数量
        success_rate_threshold: 成功率阈值
        latency_threshold: 延迟阈值（毫秒）
        error_rate_threshold: 错误率阈值
        volume_spike_threshold: 交易量突增阈值
        compliance_threshold: 合规性阈值
        log_level: 日志级别
        
    Returns:
        配置好的交易监控器实例
    """
    alert_config = AlertConfig(
        success_rate_threshold=success_rate_threshold,
        latency_threshold=latency_threshold,
        error_rate_threshold=error_rate_threshold,
        volume_spike_threshold=volume_spike_threshold,
        compliance_threshold=compliance_threshold
    )
    
    return TradingMonitor(
        max_history_size=max_history_size,
        alert_config=alert_config,
        log_level=log_level
    )


def create_trade(
    trade_id: str,
    symbol: str,
    trade_type: TradeType,
    side: str,
    quantity: float,
    price: float,
    timestamp=None,
    status: TradeStatus = TradeStatus.PENDING,
    execution_time: float = None,
    commission: float = 0.0,
    strategy_id: str = None,
    compliance_score: float = 1.0,
    error_message: str = None
) -> TradeRecord:
    """
    创建交易记录的便捷函数
    
    Args:
        trade_id: 交易ID
        symbol: 交易对符号
        trade_type: 交易类型
        side: 买卖方向 (buy/sell)
        quantity: 交易数量
        price: 交易价格
        timestamp: 时间戳（默认当前时间）
        status: 交易状态
        execution_time: 执行时间
        commission: 手续费
        strategy_id: 策略ID
        compliance_score: 合规分数
        error_message: 错误信息
        
    Returns:
        创建的交易记录
    """
    from datetime import datetime
    
    if timestamp is None:
        timestamp = datetime.now()
    
    return TradeRecord(
        trade_id=trade_id,
        symbol=symbol,
        trade_type=trade_type,
        side=side,
        quantity=quantity,
        price=price,
        timestamp=timestamp,
        status=status,
        execution_time=execution_time,
        commission=commission,
        strategy_id=strategy_id,
        compliance_score=compliance_score,
        error_message=error_message
    )


def create_default_config() -> AlertConfig:
    """
    创建默认告警配置
    
    Returns:
        默认配置的告警配置实例
    """
    return AlertConfig()


def create_strict_config() -> AlertConfig:
    """
    创建严格的告警配置（适用于生产环境）
    
    Returns:
        严格配置的告警配置实例
    """
    return AlertConfig(
        success_rate_threshold=0.98,    # 98% 成功率阈值
        latency_threshold=500.0,        # 500ms 延迟阈值
        error_rate_threshold=0.02,      # 2% 错误率阈值
        volume_spike_threshold=1.5,     # 1.5倍交易量突增阈值
        compliance_threshold=0.95       # 95% 合规性阈值
    )


def create_lenient_config() -> AlertConfig:
    """
    创建宽松的告警配置（适用于测试环境）
    
    Returns:
        宽松配置的告警配置实例
    """
    return AlertConfig(
        success_rate_threshold=0.90,    # 90% 成功率阈值
        latency_threshold=2000.0,       # 2000ms 延迟阈值
        error_rate_threshold=0.10,      # 10% 错误率阈值
        volume_spike_threshold=3.0,     # 3倍交易量突增阈值
        compliance_threshold=0.80       # 80% 合规性阈值
    )


def quick_demo(num_trades: int = 50) -> TradingMonitor:
    """
    快速演示函数，创建监控器并添加示例数据
    
    Args:
        num_trades: 要生成的示例交易数量
        
    Returns:
        包含示例数据的交易监控器
    """
    # 创建监控器
    monitor = create_monitor()
    
    # 生成示例数据
    sample_trades = create_sample_trades()[:num_trades]
    
    # 记录交易
    for trade in sample_trades:
        monitor.record_trade(trade)
    
    print(f"✅ 已创建包含 {num_trades} 笔交易示例的监控器")
    print(f"📊 总交易数: {monitor.get_total_trades()}")
    print(f"📈 成功率: {monitor.get_success_rate():.2%}")
    print(f"⏱️  平均延迟: {monitor.get_average_latency():.2f}ms")
    
    return monitor


# ==================== 常量定义 ====================

# 支持的交易对列表
SUPPORTED_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT",
    "BNBUSDT", "XRPUSDT", "LTCUSDT", "BCHUSDT", "EOSUSDT"
]

# 支持的交易策略
SUPPORTED_STRATEGIES = [
    "momentum", "mean_reversion", "arbitrage", "scalping",
    "trend_following", "grid_trading", "dca", "copy_trading"
]

# 默认性能阈值
DEFAULT_THRESHOLDS = {
    "success_rate": 0.95,
    "latency_ms": 1000.0,
    "error_rate": 0.05,
    "volume_spike": 2.0,
    "compliance": 0.9
}

# 日志级别映射
LOG_LEVELS = {
    "DEBUG": "DEBUG",
    "INFO": "INFO", 
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL"
}


# ==================== 工具函数 ====================

def get_version() -> str:
    """获取模块版本信息"""
    return __version__


def get_module_info() -> dict:
    """获取模块详细信息"""
    return {
        "name": "M6 Trading Monitor",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "license": __license__,
        "copyright": __copyright__
    }


def validate_trade_type(trade_type: str) -> bool:
    """
    验证交易类型是否有效
    
    Args:
        trade_type: 交易类型字符串
        
    Returns:
        是否为有效交易类型
    """
    return trade_type in [t.value for t in TradeType]


def validate_trade_status(status: str) -> bool:
    """
    验证交易状态是否有效
    
    Args:
        status: 交易状态字符串
        
    Returns:
        是否为有效交易状态
    """
    return status in [s.value for s in TradeStatus]


def validate_compliance_level(level: str) -> bool:
    """
    验证合规级别是否有效
    
    Args:
        level: 合规级别字符串
        
    Returns:
        是否为有效合规级别
    """
    return level in [l.value for l in ComplianceLevel]


def format_trade_summary(trade: TradeRecord) -> str:
    """
    格式化交易记录摘要
    
    Args:
        trade: 交易记录
        
    Returns:
        格式化的交易摘要字符串
    """
    return (
        f"交易 {trade.trade_id}: {trade.symbol} "
        f"{trade.side.upper()} {trade.quantity} @ {trade.price} "
        f"({trade.trade_type.value}, {trade.status.value})"
    )


def calculate_pnl(trade: TradeRecord, current_price: float = None) -> float:
    """
    计算交易盈亏
    
    Args:
        trade: 交易记录
        current_price: 当前价格（用于未平仓交易）
        
    Returns:
        盈亏金额
    """
    if current_price is None:
        current_price = trade.price
    
    if trade.side.lower() == "buy":
        return (current_price - trade.price) * trade.quantity
    else:
        return (trade.price - current_price) * trade.quantity


# ==================== 导出配置 ====================

# 设置模块级别的日志记录器
import logging

# 创建模块级别的logger
logger = logging.getLogger("M6.TradingMonitor")
logger.setLevel(logging.INFO)

# 添加空处理器以防止"无处理器"警告
if not logger.handlers:
    handler = logging.NullHandler()
    logger.addHandler(handler)

# 便捷导入方式
# from M6 import TradingMonitor, TradeRecord, AlertConfig, create_monitor