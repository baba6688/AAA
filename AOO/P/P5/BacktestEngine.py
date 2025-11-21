"""
P5回测引擎 - 核心回测引擎实现
支持历史数据回测、多时间框架、交易成本模拟、风险控制等功能
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import warnings
warnings.filterwarnings('ignore')


class TradeRecord:
    """交易记录类"""
    
    def __init__(self):
        self.trades = []
        self.positions = []
        self.equity_curve = []
        self.drawdown_curve = []
    
    def add_trade(self, timestamp: datetime, symbol: str, action: str, 
                  quantity: float, price: float, commission: float = 0.0):
        """添加交易记录"""
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,  # 'buy' or 'sell'
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'value': quantity * price,
            'net_value': quantity * price + commission if action == 'buy' else quantity * price - commission
        }
        self.trades.append(trade)
    
    def add_position(self, timestamp: datetime, symbol: str, quantity: float, 
                    price: float, market_value: float):
        """添加持仓记录"""
        position = {
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'market_value': market_value
        }
        self.positions.append(position)
    
    def add_equity_point(self, timestamp: datetime, total_value: float, 
                        cash: float, positions_value: float):
        """添加权益曲线点"""
        equity_point = {
            'timestamp': timestamp,
            'total_value': total_value,
            'cash': cash,
            'positions_value': positions_value
        }
        self.equity_curve.append(equity_point)


class PerformanceAnalyzer:
    """性能分析器"""
    
    def __init__(self, equity_curve: List[Dict], trades: List[Dict], 
                 initial_capital: float):
        self.equity_curve = equity_curve
        self.trades = trades
        self.initial_capital = initial_capital
        
    def calculate_returns(self) -> pd.Series:
        """计算收益率序列"""
        if len(self.equity_curve) < 2:
            return pd.Series(dtype=float)
        
        values = [point['total_value'] for point in self.equity_curve]
        returns = pd.Series(values).pct_change().dropna()
        return returns
    
    def calculate_cumulative_returns(self) -> pd.Series:
        """计算累计收益率"""
        if len(self.equity_curve) < 2:
            return pd.Series(dtype=float)
        
        values = [point['total_value'] for point in self.equity_curve]
        cumulative_returns = (pd.Series(values) / self.initial_capital - 1) * 100
        return cumulative_returns
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        returns = self.calculate_returns()
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        
        excess_returns = returns.mean() - risk_free_rate / 252  # 假设252个交易日
        sharpe_ratio = excess_returns / returns.std() * np.sqrt(252)
        return sharpe_ratio
    
    def calculate_max_drawdown(self) -> Tuple[float, datetime, datetime]:
        """计算最大回撤"""
        if len(self.equity_curve) < 2:
            return 0.0, None, None
        
        values = [point['total_value'] for point in self.equity_curve]
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak * 100
        
        max_drawdown = np.min(drawdown)
        max_dd_idx = np.argmin(drawdown)
        
        # 找到回撤开始和结束时间
        start_idx = 0
        for i in range(max_dd_idx, -1, -1):
            if drawdown[i] == 0:
                start_idx = i
                break
        
        start_time = self.equity_curve[start_idx]['timestamp']
        end_time = self.equity_curve[max_dd_idx]['timestamp']
        
        return max_drawdown, start_time, end_time
    
    def calculate_win_rate(self) -> float:
        """计算胜率"""
        if not self.trades:
            return 0.0
        
        profitable_trades = 0
        total_trades = len(self.trades)
        
        # 按交易对分组计算盈亏
        symbol_trades = {}
        for trade in self.trades:
            symbol = trade['symbol']
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)
        
        for symbol, trades_list in symbol_trades.items():
            if len(trades_list) >= 2:
                # 假设买入在前卖出在后
                for i in range(0, len(trades_list) - 1, 2):
                    if i + 1 < len(trades_list):
                        buy_trade = trades_list[i]
                        sell_trade = trades_list[i + 1]
                        
                        if (buy_trade['action'] == 'buy' and 
                            sell_trade['action'] == 'sell'):
                            profit = (sell_trade['price'] - buy_trade['price']) * buy_trade['quantity']
                            if profit > 0:
                                profitable_trades += 1
        
        win_rate = (profitable_trades / (total_trades // 2)) * 100 if total_trades > 0 else 0
        return win_rate
    
    def calculate_profit_factor(self) -> float:
        """计算盈亏比"""
        gross_profit = 0.0
        gross_loss = 0.0
        
        # 按交易对分组
        symbol_trades = {}
        for trade in self.trades:
            symbol = trade['symbol']
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)
        
        for symbol, trades_list in symbol_trades.items():
            if len(trades_list) >= 2:
                for i in range(0, len(trades_list) - 1, 2):
                    if i + 1 < len(trades_list):
                        buy_trade = trades_list[i]
                        sell_trade = trades_list[i + 1]
                        
                        if (buy_trade['action'] == 'buy' and 
                            sell_trade['action'] == 'sell'):
                            profit = (sell_trade['price'] - buy_trade['price']) * buy_trade['quantity']
                            if profit > 0:
                                gross_profit += profit
                            else:
                                gross_loss += abs(profit)
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def calculate_average_trade(self) -> float:
        """计算平均每笔交易收益"""
        if not self.trades:
            return 0.0
        
        total_profit = 0.0
        trade_count = 0
        
        # 按交易对分组
        symbol_trades = {}
        for trade in self.trades:
            symbol = trade['symbol']
            if symbol not in symbol_trades:
                symbol_trades[symbol] = []
            symbol_trades[symbol].append(trade)
        
        for symbol, trades_list in symbol_trades.items():
            if len(trades_list) >= 2:
                for i in range(0, len(trades_list) - 1, 2):
                    if i + 1 < len(trades_list):
                        buy_trade = trades_list[i]
                        sell_trade = trades_list[i + 1]
                        
                        if (buy_trade['action'] == 'buy' and 
                            sell_trade['action'] == 'sell'):
                            profit = (sell_trade['price'] - buy_trade['price']) * buy_trade['quantity']
                            total_profit += profit
                            trade_count += 1
        
        return total_profit / trade_count if trade_count > 0 else 0.0
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        returns = self.calculate_returns()
        cumulative_returns = self.calculate_cumulative_returns()
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown, dd_start, dd_end = self.calculate_max_drawdown()
        win_rate = self.calculate_win_rate()
        profit_factor = self.calculate_profit_factor()
        avg_trade = self.calculate_average_trade()
        
        final_value = self.equity_curve[-1]['total_value'] if self.equity_curve else self.initial_capital
        total_return = (final_value / self.initial_capital - 1) * 100
        
        # 计算年化收益率
        if len(self.equity_curve) >= 2:
            start_date = self.equity_curve[0]['timestamp']
            end_date = self.equity_curve[-1]['timestamp']
            years = (end_date - start_date).days / 365.25
            annualized_return = ((final_value / self.initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
        else:
            annualized_return = 0
        
        report = {
            '总收益率(%)': round(total_return, 2),
            '年化收益率(%)': round(annualized_return, 2),
            '夏普比率': round(sharpe_ratio, 3),
            '最大回撤(%)': round(abs(max_drawdown), 2),
            '胜率(%)': round(win_rate, 2),
            '盈亏比': round(profit_factor, 2),
            '平均每笔交易收益': round(avg_trade, 2),
            '总交易次数': len(self.trades),
            '最大回撤开始时间': dd_start,
            '最大回撤结束时间': dd_end,
            '最终资金': round(final_value, 2),
            '初始资金': self.initial_capital
        }
        
        return report


class BacktestEngine:
    """P5回测引擎主类"""
    
    def __init__(self, initial_capital: float = 100000.0, 
                 commission_rate: float = 0.001, slippage: float = 0.0001):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 手续费率
            slippage: 滑点
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage = slippage
        
        self.cash = initial_capital
        self.positions = {}  # symbol -> quantity
        self.current_time = None
        self.data = None
        self.strategy = None
        self.trade_record = TradeRecord()
        
        # 风险控制参数
        self.max_position_size = 0.95  # 最大仓位比例
        self.stop_loss = None  # 止损比例
        self.take_profit = None  # 止盈比例
        self.max_drawdown_limit = None  # 最大回撤限制
        
    def set_data(self, data: pd.DataFrame):
        """设置历史数据"""
        self.data = data.copy()
        if 'timestamp' not in self.data.columns and 'date' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['date'])
        elif 'timestamp' not in self.data.columns and 'datetime' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['datetime'])
    
    def set_strategy(self, strategy):
        """设置交易策略"""
        self.strategy = strategy
    
    def set_risk_parameters(self, max_position_size: float = None, 
                           stop_loss: float = None, take_profit: float = None,
                           max_drawdown_limit: float = None):
        """设置风险控制参数"""
        if max_position_size is not None:
            self.max_position_size = max_position_size
        if stop_loss is not None:
            self.stop_loss = stop_loss
        if take_profit is not None:
            self.take_profit = take_profit
        if max_drawdown_limit is not None:
            self.max_drawdown_limit = max_drawdown_limit
    
    def execute_order(self, symbol: str, action: str, quantity: float, 
                     price: float, timestamp: datetime) -> bool:
        """
        执行交易订单
        
        Args:
            symbol: 交易品种
            action: 'buy' 或 'sell'
            quantity: 交易数量
            price: 交易价格
            timestamp: 时间戳
            
        Returns:
            bool: 是否成功执行
        """
        # 计算交易成本
        gross_value = quantity * price
        commission = gross_value * self.commission_rate
        slippage_cost = gross_value * self.slippage
        
        if action == 'buy':
            total_cost = gross_value + commission + slippage_cost
            
            # 检查资金是否足够
            if total_cost > self.cash:
                return False
            
            # 检查仓位限制
            current_position_value = sum(self.positions.get(s, 0) * price for s in self.positions)
            new_position_value = current_position_value + gross_value
            total_value = self.cash + current_position_value
            
            if new_position_value / total_value > self.max_position_size:
                return False
            
            # 执行买入
            self.cash -= total_cost
            self.positions[symbol] = self.positions.get(symbol, 0) + quantity
            
            # 记录交易
            self.trade_record.add_trade(timestamp, symbol, action, quantity, 
                                      price + slippage_cost, commission + slippage_cost)
            
        elif action == 'sell':
            # 检查是否有足够持仓
            current_quantity = self.positions.get(symbol, 0)
            if current_quantity < quantity:
                return False
            
            # 计算实际卖出价格（考虑滑点）
            actual_price = price - slippage_cost
            gross_proceeds = quantity * actual_price
            net_proceeds = gross_proceeds - commission
            
            # 执行卖出
            self.cash += net_proceeds
            self.positions[symbol] = current_quantity - quantity
            
            # 记录交易
            self.trade_record.add_trade(timestamp, symbol, action, quantity, 
                                      actual_price, commission)
        
        return True
    
    def check_risk_controls(self, timestamp: datetime) -> bool:
        """检查风险控制条件"""
        # 检查最大回撤
        if self.max_drawdown_limit is not None and self.trade_record.equity_curve:
            current_value = self.trade_record.equity_curve[-1]['total_value']
            peak_value = max(point['total_value'] for point in self.trade_record.equity_curve)
            current_drawdown = (current_value - peak_value) / peak_value * 100
            
            if current_drawdown < -abs(self.max_drawdown_limit):
                return False
        
        return True
    
    def update_portfolio_value(self, timestamp: datetime):
        """更新投资组合价值"""
        total_positions_value = 0.0
        
        for symbol, quantity in self.positions.items():
            if quantity > 0 and symbol in self.data.columns:
                # 获取最新价格
                current_data = self.data[self.data['timestamp'] <= timestamp]
                if not current_data.empty:
                    latest_price = current_data[symbol].iloc[-1]
                    total_positions_value += quantity * latest_price
        
        total_value = self.cash + total_positions_value
        
        # 记录权益曲线
        self.trade_record.add_equity_point(timestamp, total_value, 
                                         self.cash, total_positions_value)
    
    def run_backtest(self, start_date: datetime = None, end_date: datetime = None) -> Dict[str, Any]:
        """
        运行回测
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict: 回测结果
        """
        if self.data is None or self.strategy is None:
            raise ValueError("请先设置数据和策略")
        
        # 过滤数据
        data = self.data.copy()
        if start_date:
            data = data[data['timestamp'] >= start_date]
        if end_date:
            data = data[data['timestamp'] <= end_date]
        
        if data.empty:
            raise ValueError("没有符合条件的数据")
        
        # 重置状态
        self.cash = self.initial_capital
        self.positions = {}
        self.trade_record = TradeRecord()
        
        # 运行回测
        for idx, row in data.iterrows():
            self.current_time = row['timestamp']
            
            # 更新投资组合价值
            self.update_portfolio_value(self.current_time)
            
            # 检查风险控制
            if not self.check_risk_controls(self.current_time):
                break
            
            # 获取策略信号
            signals = self.strategy.generate_signals(data.iloc[:idx+1])
            
            # 执行交易信号
            for signal in signals:
                symbol = signal['symbol']
                action = signal['action']
                quantity = signal['quantity']
                price = signal.get('price', row[symbol])
                
                self.execute_order(symbol, action, quantity, price, self.current_time)
        
        # 最终更新
        self.update_portfolio_value(self.current_time)
        
        # 生成性能报告
        analyzer = PerformanceAnalyzer(
            self.trade_record.equity_curve, 
            self.trade_record.trades, 
            self.initial_capital
        )
        
        performance_report = analyzer.generate_performance_report()
        
        return {
            'performance': performance_report,
            'trades': self.trade_record.trades,
            'equity_curve': self.trade_record.equity_curve,
            'positions': self.positions.copy(),
            'final_cash': self.cash
        }
    
    def run_multi_timeframe_backtest(self, data_dict: Dict[str, pd.DataFrame], 
                                   start_date: datetime = None, 
                                   end_date: datetime = None) -> Dict[str, Any]:
        """
        多时间框架回测
        
        Args:
            data_dict: 不同时间框架的数据字典 {timeframe: DataFrame}
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            Dict: 回测结果
        """
        # 这里可以实现多时间框架的复杂逻辑
        # 简化实现：使用主时间框架数据进行回测
        main_timeframe = list(data_dict.keys())[0]
        self.set_data(data_dict[main_timeframe])
        
        return self.run_backtest(start_date, end_date)
    
    def optimize_strategy(self, param_grid: Dict[str, List], 
                         data: pd.DataFrame = None,
                         metric: str = '夏普比率') -> Dict[str, Any]:
        """
        策略参数优化
        
        Args:
            param_grid: 参数网格 {param_name: [param_values]}
            data: 优化使用的数据
            metric: 优化指标
            
        Returns:
            Dict: 优化结果
        """
        if data is not None:
            self.set_data(data)
        
        best_params = {}
        best_score = float('-inf')
        results = []
        
        # 生成所有参数组合
        import itertools
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        for param_combination in itertools.product(*param_values):
            param_dict = dict(zip(param_names, param_combination))
            
            # 设置策略参数
            self.strategy.set_parameters(param_dict)
            
            # 运行回测
            try:
                result = self.run_backtest()
                score = result['performance'].get(metric, 0)
                
                results.append({
                    'params': param_dict,
                    'score': score,
                    'performance': result['performance']
                })
                
                if score > best_score:
                    best_score = score
                    best_params = param_dict
                    
            except Exception as e:
                print(f"参数组合 {param_dict} 回测失败: {e}")
                continue
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'all_results': results
        }


class BaseStrategy:
    """基础策略类"""
    
    def __init__(self):
        self.parameters = {}
    
    def set_parameters(self, params: Dict[str, Any]):
        """设置策略参数"""
        self.parameters.update(params)
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        生成交易信号
        
        Args:
            data: 历史数据
            
        Returns:
            List[Dict]: 交易信号列表
        """
        raise NotImplementedError("子类必须实现此方法")


class SimpleMovingAverageStrategy(BaseStrategy):
    """简单移动平均策略"""
    
    def generate_signals(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """生成移动平均交易信号"""
        if len(data) < 2:
            return []
        
        signals = []
        
        # 获取参数
        short_window = self.parameters.get('short_window', 5)
        long_window = self.parameters.get('long_window', 20)
        symbol = self.parameters.get('symbol', 'close')
        
        if len(data) < long_window:
            return signals
        
        # 计算移动平均
        short_ma = data[symbol].rolling(window=short_window).mean()
        long_ma = data[symbol].rolling(window=long_window).mean()
        
        # 生成信号
        if short_ma.iloc[-1] > long_ma.iloc[-1] and short_ma.iloc[-2] <= long_ma.iloc[-2]:
            # 金叉买入信号
            signals.append({
                'symbol': symbol,
                'action': 'buy',
                'quantity': 100,  # 固定数量，实际应用中应该基于资金管理
                'price': data[symbol].iloc[-1]
            })
        elif short_ma.iloc[-1] < long_ma.iloc[-1] and short_ma.iloc[-2] >= long_ma.iloc[-2]:
            # 死叉卖出信号
            signals.append({
                'symbol': symbol,
                'action': 'sell',
                'quantity': 100,
                'price': data[symbol].iloc[-1]
            })
        
        return signals


# 辅助函数
def create_sample_data(start_date: str = '2020-01-01', 
                      end_date: str = '2023-12-31',
                      freq: str = 'D') -> pd.DataFrame:
    """
    创建示例数据
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        freq: 频率 ('D'=日线, 'H'=小时, 'T'=分钟)
        
    Returns:
        pd.DataFrame: 示例数据
    """
    date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # 生成模拟价格数据
    np.random.seed(42)
    n_periods = len(date_range)
    
    # 生成随机游走价格
    returns = np.random.normal(0.001, 0.02, n_periods)
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'timestamp': date_range,
        'close': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_periods))),
        'open': prices * (1 + np.random.normal(0, 0.005, n_periods)),
        'volume': np.random.randint(1000, 10000, n_periods)
    })
    
    return data


def compare_strategies(strategies: Dict[str, BaseStrategy], 
                      data: pd.DataFrame,
                      initial_capital: float = 100000) -> Dict[str, Any]:
    """
    比较多个策略的性能
    
    Args:
        strategies: 策略字典 {name: strategy}
        data: 历史数据
        initial_capital: 初始资金
        
    Returns:
        Dict: 比较结果
    """
    results = {}
    
    for name, strategy in strategies.items():
        engine = BacktestEngine(initial_capital=initial_capital)
        engine.set_data(data)
        engine.set_strategy(strategy)
        
        try:
            result = engine.run_backtest()
            results[name] = result
        except Exception as e:
            print(f"策略 {name} 回测失败: {e}")
            continue
    
    return results


if __name__ == "__main__":
    # 示例使用
    print("P5回测引擎示例")
    
    # 创建示例数据
    data = create_sample_data('2020-01-01', '2023-12-31', 'D')
    print(f"创建了 {len(data)} 条数据记录")
    
    # 创建策略
    strategy = SimpleMovingAverageStrategy()
    strategy.set_parameters({
        'short_window': 5,
        'long_window': 20,
        'symbol': 'close'
    })
    
    # 创建回测引擎
    engine = BacktestEngine(initial_capital=100000)
    engine.set_data(data)
    engine.set_strategy(strategy)
    
    # 设置风险控制
    engine.set_risk_parameters(
        max_position_size=0.8,
        stop_loss=0.1,
        take_profit=0.2
    )
    
    # 运行回测
    result = engine.run_backtest()
    
    # 打印结果
    print("\n回测结果:")
    for key, value in result['performance'].items():
        print(f"{key}: {value}")