"""
P5回测引擎 - 量化交易回测系统

一个功能完整的量化交易回测引擎，支持历史数据回测、多时间框架、
交易成本模拟、风险控制、策略优化等功能。

主要组件:
- BacktestEngine: 回测引擎主类
- PerformanceAnalyzer: 性能分析器
- BaseStrategy: 基础策略类
- SimpleMovingAverageStrategy: 简单移动平均策略示例

使用示例:
    from P5 import BacktestEngine, SimpleMovingAverageStrategy, create_sample_data
    
    # 创建数据
    data = create_sample_data('2020-01-01', '2023-12-31', 'D')
    
    # 创建策略
    strategy = SimpleMovingAverageStrategy()
    strategy.set_parameters({'short_window': 5, 'long_window': 20})
    
    # 运行回测
    engine = BacktestEngine(initial_capital=100000)
    engine.set_data(data)
    engine.set_strategy(strategy)
    result = engine.run_backtest()
    
    # 查看结果
    print(result['performance'])
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "P5量化团队"
__email__ = "p5-quant@example.com"

# 导入pandas
import pandas as pd

# 导入核心类
from .BacktestEngine import (
    BacktestEngine,
    PerformanceAnalyzer,
    TradeRecord,
    BaseStrategy,
    SimpleMovingAverageStrategy,
    create_sample_data,
    compare_strategies
)

# 定义公共API
__all__ = [
    'BacktestEngine',
    'PerformanceAnalyzer', 
    'TradeRecord',
    'BaseStrategy',
    'SimpleMovingAverageStrategy',
    'create_sample_data',
    'compare_strategies',
    '__version__'
]

# 快捷函数
def quick_backtest(data, strategy_params, initial_capital=100000):
    """
    快速回测函数
    
    Args:
        data: 历史数据DataFrame
        strategy_params: 策略参数字典
        initial_capital: 初始资金
        
    Returns:
        dict: 回测结果
    """
    from .BacktestEngine import BacktestEngine, SimpleMovingAverageStrategy
    
    # 创建策略
    strategy = SimpleMovingAverageStrategy()
    strategy.set_parameters(strategy_params)
    
    # 创建引擎并运行回测
    engine = BacktestEngine(initial_capital=initial_capital)
    engine.set_data(data)
    engine.set_strategy(strategy)
    
    return engine.run_backtest()


def compare_ma_strategies(data, short_windows, long_windows, initial_capital=100000):
    """
    比较不同移动平均策略
    
    Args:
        data: 历史数据
        short_windows: 短期窗口列表
        long_windows: 长期窗口列表
        initial_capital: 初始资金
        
    Returns:
        dict: 比较结果
    """
    from .BacktestEngine import SimpleMovingAverageStrategy, compare_strategies
    
    strategies = {}
    
    # 生成所有策略组合
    for short_win in short_windows:
        for long_win in long_windows:
            if short_win < long_win:  # 确保短期窗口小于长期窗口
                strategy_name = f"MA{short_win}-{long_win}"
                strategy = SimpleMovingAverageStrategy()
                strategy.set_parameters({
                    'short_window': short_win,
                    'long_window': long_win,
                    'symbol': 'close'
                })
                strategies[strategy_name] = strategy
    
    return compare_strategies(strategies, data, initial_capital)


# 工具函数
def get_sample_data(start_date='2020-01-01', end_date='2023-12-31', freq='D'):
    """
    获取示例数据
    
    Args:
        start_date: 开始日期
        end_date: 结束日期  
        freq: 频率 ('D'=日线, 'H'=小时, 'T'=分钟)
        
    Returns:
        pd.DataFrame: 示例数据
    """
    return create_sample_data(start_date, end_date, freq)


def validate_data(data):
    """
    验证数据格式
    
    Args:
        data: 要验证的数据DataFrame
        
    Returns:
        bool: 数据是否有效
    """
    required_columns = ['timestamp', 'close']
    
    # 检查必需列
    for col in required_columns:
        if col not in data.columns:
            return False
    
    # 检查数据类型
    if not hasattr(data['timestamp'], 'dtype'):
        return False
    
    # 检查时间序列
    if not data['timestamp'].is_monotonic_increasing:
        return False
    
    # 检查数值列
    numeric_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_columns:
        if col in data.columns:
            if not pd.api.types.is_numeric_dtype(data[col]):
                return False
    
    return True


# 性能基准装饰器
def benchmark(func):
    """
    性能基准测试装饰器
    """
    import time
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        print(f"{func.__name__} 执行时间: {end_time - start_time:.3f} 秒")
        return result
    
    return wrapper


# 版本检查
def check_version():
    """检查版本信息"""
    import sys
    import numpy as np
    import pandas as pd
    
    print(f"P5回测引擎版本: {__version__}")
    print(f"Python版本: {sys.version}")
    print(f"NumPy版本: {np.__version__}")
    print(f"Pandas版本: {pd.__version__}")
    
    # 检查依赖
    try:
        import matplotlib
        print(f"Matplotlib版本: {matplotlib.__version__}")
    except ImportError:
        print("Matplotlib: 未安装 (可选)")
    
    print("环境检查完成!")


if __name__ == "__main__":
    # 如果直接运行此文件，显示版本信息
    check_version()
    
    # 运行简单示例
    print("\n运行示例...")
    sample_data = get_sample_data('2020-01-01', '2020-12-31', 'D')
    print(f"创建了 {len(sample_data)} 条示例数据")
    
    # 快速回测示例
    result = quick_backtest(
        sample_data, 
        {'short_window': 5, 'long_window': 20, 'symbol': 'close'}
    )
    
    print("\n回测结果:")
    for key, value in result['performance'].items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")