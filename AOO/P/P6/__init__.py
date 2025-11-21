"""
P6模拟交易器包

这个包提供了一个完整的模拟交易系统，包括：
- 模拟交易执行
- 风险管理
- 组合管理
- 交易成本模拟
- 市场模拟
- 交易信号生成
- 实时监控
- 交易报告生成

主要类：
- SimulatedTrader: 主要的模拟交易器类
- Order: 订单类
- Position: 持仓类
- Portfolio: 组合类
- RiskManager: 风险管理类
- MarketSimulator: 市场模拟器类
- TradingSignal: 交易信号类
- TradingReport: 交易报告类
"""

from .SimulatedTrader import (
    # 核心枚举
    OrderType,
    OrderSide,
    OrderStatus,
    SignalType,
    
    # 核心数据类
    Order,
    Trade,
    Position,
    Portfolio,
    Signal,
    TransactionCost,
    PerformanceMetrics,
    
    # 核心类
    SimulatedTrader,
    RiskManager,
    MarketSimulator,
    TradingSignal,
    TradingReport,
    
    # 便利函数
    create_simulated_trader
)

__version__ = "1.0.0"
__author__ = "P6 Trading System"

__all__ = [
    # 枚举类型
    'OrderType',
    'OrderSide',
    'OrderStatus', 
    'SignalType',
    
    # 数据类
    'Order',
    'Trade',
    'Position',
    'Portfolio',
    'Signal',
    'TransactionCost',
    'PerformanceMetrics',
    
    # 核心类
    'SimulatedTrader',
    'RiskManager',
    'MarketSimulator',
    'TradingSignal',
    'TradingReport',
    
    # 便利函数
    'create_simulated_trader'
]

# 便利函数
def create_simulated_trader(initial_capital: float = 100000.0, commission_rate: float = 0.001):
    """
    创建模拟交易器实例
    
    Args:
        initial_capital: 初始资金
        commission_rate: 佣金费率
    
    Returns:
        SimulatedTrader: 模拟交易器实例
    
    Examples:
        from P6 import create_simulated_trader
        
        # 创建基本交易器
        trader = create_simulated_trader()
        
        # 创建自定义交易器
        trader = create_simulated_trader(
            initial_capital=500000,
            commission_rate=0.0005
        )
    """
    return SimulatedTrader(initial_capital=initial_capital, commission_rate=commission_rate)

# 交易类型别名
OrderTypes = OrderType
OrderSides = OrderSide
OrderStatuses = OrderStatus
SignalTypes = SignalType

# 快速开始指南
QUICK_START = """
P6模拟交易器快速开始：

1. 创建交易器：
   from P6 import create_simulated_trader
   trader = create_simulated_trader(initial_capital=100000)

2. 下单交易：
   # 创建买单
   order = Order(
       symbol='AAPL',
       side=OrderSide.BUY,
       quantity=100,
       order_type=OrderType.MARKET
   )
   
   # 执行订单
   result = trader.place_order(order)

3. 查看持仓和组合：
   positions = trader.get_positions()
   portfolio = trader.get_portfolio()
   
4. 运行风险管理：
   risk_report = trader.risk_manager.generate_report()

5. 生成交易报告：
   report = trader.generate_trading_report()
   print(report)
"""

# 市场数据模拟器
class MarketDataSimulator:
    """市场数据模拟器便利类"""
    
    @staticmethod
    def generate_sample_prices(symbol: str = 'AAPL', days: int = 30):
        """生成样本价格数据"""
        import random
        from datetime import datetime, timedelta
        
        base_price = 150.0
        prices = []
        current_price = base_price
        
        for i in range(days):
            # 模拟价格波动 (±2%)
            change = random.uniform(-0.02, 0.02)
            current_price *= (1 + change)
            prices.append(current_price)
        
        return prices

# 性能分析装饰器
def analyze_performance(func):
    """性能分析装饰器"""
    def wrapper(*args, **kwargs):
        import time
        from .SimulatedTrader import PerformanceMetrics
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # 创建性能指标
        metrics = PerformanceMetrics()
        metrics.execution_time = end_time - start_time
        
        print(f"函数 {func.__name__} 执行时间: {metrics.execution_time:.3f}秒")
        return result
    
    return wrapper

print("P6模拟交易器已加载")