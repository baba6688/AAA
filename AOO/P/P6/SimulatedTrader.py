"""
P6模拟交易器 - 主要实现

这个模块实现了完整的模拟交易系统，包括所有核心功能：
- 模拟交易执行
- 风险管理
- 组合管理
- 交易成本模拟
- 市场模拟
- 交易信号生成
- 实时监控
- 交易报告生成
"""

import datetime
import logging
import json
import math
import random
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import uuid


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"  # 市价单
    LIMIT = "limit"    # 限价单
    STOP = "stop"      # 止损单
    STOP_LIMIT = "stop_limit"  # 止损限价单


class OrderSide(Enum):
    """订单方向枚举"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"      # 待成交
    FILLED = "filled"        # 已成交
    PARTIAL = "partial"      # 部分成交
    CANCELLED = "cancelled"  # 已取消
    REJECTED = "rejected"    # 已拒绝


class SignalType(Enum):
    """交易信号类型"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass
class Order:
    """订单类"""
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType = OrderType.MARKET
    price: Optional[float] = None
    stop_price: Optional[float] = None
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_price: float = 0.0
    commission: float = 0.0
    
    def __post_init__(self):
        """订单初始化后的处理"""
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("限价单必须指定价格")
        if self.order_type == OrderType.STOP and self.stop_price is None:
            raise ValueError("止损单必须指定止损价格")
    
    def fill(self, quantity: float, price: float, commission: float = 0.0) -> None:
        """订单成交处理"""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            raise ValueError(f"订单状态错误: {self.status}")
        
        if self.filled_quantity + quantity > self.quantity:
            raise ValueError("成交数量超过订单数量")
        
        self.filled_quantity += quantity
        
        # 计算平均成交价格
        total_value = self.average_price * self.filled_quantity + price * quantity
        self.filled_quantity += 0  # 避免重复计算
        self.average_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0
        
        self.commission += commission
        
        # 更新订单状态
        if self.filled_quantity == self.quantity:
            self.status = OrderStatus.FILLED
        elif self.filled_quantity > 0:
            self.status = OrderStatus.PARTIAL
        else:
            self.status = OrderStatus.PENDING
    
    def cancel(self) -> None:
        """取消订单"""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELLED]:
            raise ValueError(f"无法取消已成交或已取消的订单: {self.status}")
        self.status = OrderStatus.CANCELLED
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'order_type': self.order_type.value,
            'price': self.price,
            'stop_price': self.stop_price,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'average_price': self.average_price,
            'commission': self.commission
        }


@dataclass
class Position:
    """持仓类"""
    symbol: str
    quantity: float = 0.0
    average_price: float = 0.0
    market_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    
    def update_market_price(self, price: float) -> None:
        """更新市场价格"""
        self.market_price = price
        if self.quantity != 0:
            self.unrealized_pnl = (price - self.average_price) * self.quantity
    
    def add_position(self, quantity: float, price: float, commission: float = 0.0) -> None:
        """增加持仓"""
        if quantity > 0:  # 买入
            total_cost = self.average_price * self.quantity + price * quantity
            self.quantity += quantity
            self.average_price = total_cost / self.quantity if self.quantity > 0 else 0
        else:  # 卖出
            realized_pnl = (price - self.average_price) * abs(quantity)
            self.realized_pnl += realized_pnl
            self.quantity += quantity  # quantity为负数
        
        self.total_commission += commission
        self.update_market_price(self.market_price)
    
    def get_position_value(self) -> float:
        """获取持仓市值"""
        return self.quantity * self.market_price
    
    def get_total_pnl(self) -> float:
        """获取总盈亏"""
        return self.unrealized_pnl + self.realized_pnl - self.total_commission


@dataclass
class TransactionCost:
    """交易成本类"""
    commission_rate: float = 0.001  # 手续费率 (0.1%)
    min_commission: float = 5.0     # 最小手续费
    slippage_rate: float = 0.0005   # 滑点率 (0.05%)
    stamp_tax_rate: float = 0.001   # 印花税率 (0.1%)
    
    def calculate_commission(self, trade_value: float) -> float:
        """计算手续费"""
        commission = trade_value * self.commission_rate
        return max(commission, self.min_commission)
    
    def calculate_slippage(self, trade_value: float) -> float:
        """计算滑点成本"""
        return trade_value * self.slippage_rate
    
    def calculate_stamp_tax(self, trade_value: float, is_sell: bool = True) -> float:
        """计算印花税（仅卖出时收取）"""
        if not is_sell:
            return 0
        return trade_value * self.stamp_tax_rate
    
    def total_cost(self, trade_value: float, is_sell: bool = True) -> float:
        """计算总交易成本"""
        commission = self.calculate_commission(trade_value)
        slippage = self.calculate_slippage(trade_value)
        stamp_tax = self.calculate_stamp_tax(trade_value, is_sell)
        return commission + slippage + stamp_tax


class Portfolio:
    """投资组合类"""
    
    def __init__(self, initial_cash: float = 1000000.0):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.orders: List[Order] = []
        self.transaction_history: List[Dict] = []
        self.daily_returns: List[float] = []
        self.equity_curve: List[float] = []
        
    def get_position(self, symbol: str) -> Position:
        """获取持仓"""
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol)
        return self.positions[symbol]
    
    def update_market_prices(self, prices: Dict[str, float]) -> None:
        """更新市场价格"""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_market_price(price)
        
        # 更新权益曲线
        total_value = self.cash + sum(pos.get_position_value() for pos in self.positions.values())
        self.equity_curve.append(total_value)
        
        # 计算日收益率
        if len(self.equity_curve) > 1:
            daily_return = (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]
            self.daily_returns.append(daily_return)
    
    def execute_order(self, order: Order, execution_price: float) -> bool:
        """执行订单"""
        try:
            position = self.get_position(order.symbol)
            trade_value = execution_price * order.quantity
            
            # 计算交易成本
            transaction_cost = TransactionCost()
            cost = transaction_cost.total_cost(trade_value, order.side == OrderSide.SELL)
            
            # 检查资金充足性
            if order.side == OrderSide.BUY and self.cash < trade_value + cost:
                logger.warning(f"资金不足，无法执行买入订单: {order.symbol}")
                return False
            
            # 更新持仓
            quantity = order.quantity if order.side == OrderSide.BUY else -order.quantity
            position.add_position(quantity, execution_price, transaction_cost.calculate_commission(trade_value))
            
            # 更新现金
            if order.side == OrderSide.BUY:
                self.cash -= (trade_value + cost)
            else:
                self.cash += (trade_value - cost)
            
            # 记录交易历史
            self.transaction_history.append({
                'timestamp': datetime.datetime.now(),
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': order.quantity,
                'price': execution_price,
                'value': trade_value,
                'commission': transaction_cost.calculate_commission(trade_value),
                'total_cost': cost
            })
            
            # 更新订单状态
            order.fill(order.quantity, execution_price, transaction_cost.calculate_commission(trade_value))
            self.orders.append(order)
            
            logger.info(f"订单执行成功: {order.symbol} {order.side.value} {order.quantity}@{execution_price}")
            return True
            
        except Exception as e:
            logger.error(f"订单执行失败: {e}")
            return False
    
    def get_portfolio_value(self) -> float:
        """获取组合总价值"""
        positions_value = sum(pos.get_position_value() for pos in self.positions.values())
        return self.cash + positions_value
    
    def get_portfolio_pnl(self) -> float:
        """获取组合总盈亏"""
        total_pnl = sum(pos.get_total_pnl() for pos in self.positions.values())
        return total_pnl + (self.cash - self.initial_cash)
    
    def get_positions_summary(self) -> List[Dict]:
        """获取持仓汇总"""
        data = []
        for symbol, position in self.positions.items():
            if position.quantity != 0:
                data.append({
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'average_price': position.average_price,
                    'market_price': position.market_price,
                    'position_value': position.get_position_value(),
                    'unrealized_pnl': position.unrealized_pnl,
                    'realized_pnl': position.realized_pnl,
                    'total_pnl': position.get_total_pnl()
                })
        return data


class RiskManager:
    """风险管理类"""
    
    def __init__(self, 
                 max_position_size: float = 0.1,  # 单个资产最大仓位比例
                 max_portfolio_risk: float = 0.02,  # 组合最大风险
                 stop_loss_pct: float = 0.05,      # 止损比例
                 take_profit_pct: float = 0.15,    # 止盈比例
                 max_drawdown: float = 0.20):      # 最大回撤
        self.max_position_size = max_position_size
        self.max_portfolio_risk = max_portfolio_risk
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_drawdown = max_drawdown
        
        self.peak_equity = 0.0
        self.risk_metrics = {}
    
    def check_position_size(self, portfolio: Portfolio, symbol: str, quantity: float, 
                          price: float) -> Tuple[bool, str]:
        """检查仓位大小"""
        portfolio_value = portfolio.get_portfolio_value()
        trade_value = quantity * price
        position_ratio = trade_value / portfolio_value
        
        if position_ratio > self.max_position_size:
            return False, f"仓位比例 {position_ratio:.2%} 超过最大限制 {self.max_position_size:.2%}"
        
        return True, "仓位大小检查通过"
    
    def check_portfolio_risk(self, portfolio: Portfolio) -> Tuple[bool, str, Dict]:
        """检查组合风险"""
        current_drawdown = self.calculate_drawdown(portfolio)
        
        risk_metrics = {
            'current_drawdown': current_drawdown,
            'max_drawdown_limit': self.max_drawdown,
            'position_count': len([p for p in portfolio.positions.values() if p.quantity != 0]),
            'cash_ratio': portfolio.cash / portfolio.get_portfolio_value()
        }
        
        if current_drawdown > self.max_drawdown:
            return False, f"当前回撤 {current_drawdown:.2%} 超过最大限制 {self.max_drawdown:.2%}", risk_metrics
        
        return True, "组合风险检查通过", risk_metrics
    
    def calculate_drawdown(self, portfolio: Portfolio) -> float:
        """计算当前回撤"""
        current_equity = portfolio.get_portfolio_value()
        
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        
        if self.peak_equity == 0:
            return 0.0
        
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        return max(0.0, drawdown)
    
    def should_stop_loss(self, position: Position) -> bool:
        """检查是否需要止损"""
        if position.quantity == 0:
            return False
        
        if position.market_price == 0:
            return False
        
        loss_pct = (position.average_price - position.market_price) / position.average_price
        return loss_pct >= self.stop_loss_pct
    
    def should_take_profit(self, position: Position) -> bool:
        """检查是否需要止盈"""
        if position.quantity == 0:
            return False
        
        if position.market_price == 0:
            return False
        
        profit_pct = (position.market_price - position.average_price) / position.average_price
        return profit_pct >= self.take_profit_pct
    
    def generate_risk_signals(self, portfolio: Portfolio) -> List[Dict]:
        """生成风险信号"""
        signals = []
        
        for symbol, position in portfolio.positions.items():
            if position.quantity == 0:
                continue
            
            if self.should_stop_loss(position):
                signals.append({
                    'type': 'stop_loss',
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': abs(position.quantity),
                    'reason': f'止损触发，亏损比例: {(position.average_price - position.market_price) / position.average_price:.2%}'
                })
            
            elif self.should_take_profit(position):
                signals.append({
                    'type': 'take_profit',
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': abs(position.quantity),
                    'reason': f'止盈触发，盈利比例: {(position.market_price - position.average_price) / position.average_price:.2%}'
                })
        
        return signals


class MarketSimulator:
    """市场模拟器类"""
    
    def __init__(self, volatility: float = 0.02, trend: float = 0.001):
        self.volatility = volatility  # 市场波动率
        self.trend = trend           # 市场趋势
        self.price_history: Dict[str, List[float]] = {}
        self.current_prices: Dict[str, float] = {}
        
    def set_initial_prices(self, symbols: List[str], initial_prices: Dict[str, float]) -> None:
        """设置初始价格"""
        for symbol in symbols:
            price = initial_prices.get(symbol, 100.0)
            self.current_prices[symbol] = price
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(price)
    
    def simulate_price_movement(self, symbol: str) -> float:
        """模拟价格变动"""
        if symbol not in self.current_prices:
            self.current_prices[symbol] = 100.0
        
        current_price = self.current_prices[symbol]
        
        # 使用几何布朗运动模拟价格变动
        dt = 1/252  # 一天的时间步长
        drift = self.trend * dt
        diffusion = self.volatility * math.sqrt(dt) * random.gauss(0, 1)
        
        # 计算新价格
        new_price = current_price * math.exp(drift + diffusion)
        new_price = max(new_price, 0.01)  # 价格不能为负
        
        # 更新价格
        self.current_prices[symbol] = new_price
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        self.price_history[symbol].append(new_price)
        
        return new_price
    
    def get_market_price(self, symbol: str) -> float:
        """获取市场价格"""
        return self.current_prices.get(symbol, 100.0)
    
    def update_market_prices(self, prices: Dict[str, float]) -> None:
        """更新市场价格（用于组合更新）"""
        for symbol, price in prices.items():
            self.current_prices[symbol] = price
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append(price)
    
    def simulate_order_execution(self, order: Order) -> Tuple[bool, float, str]:
        """模拟订单执行"""
        market_price = self.get_market_price(order.symbol)
        
        if order.order_type == OrderType.MARKET:
            # 市价单立即执行
            execution_price = market_price * (1 + 0.001 if order.side == OrderSide.BUY else 0.999)
            return True, execution_price, "市价单执行"
        
        elif order.order_type == OrderType.LIMIT:
            # 限价单检查价格条件
            if order.side == OrderSide.BUY and market_price <= order.price:
                return True, order.price, "限价买单执行"
            elif order.side == OrderSide.SELL and market_price >= order.price:
                return True, order.price, "限价卖单执行"
            else:
                return False, 0, "限价单未触发"
        
        elif order.order_type == OrderType.STOP:
            # 止损单检查价格条件
            if order.side == OrderSide.BUY and market_price >= order.stop_price:
                execution_price = market_price * 1.001  # 略高于市价
                return True, execution_price, "止损买单执行"
            elif order.side == OrderSide.SELL and market_price <= order.stop_price:
                execution_price = market_price * 0.999  # 略低于市价
                return True, execution_price, "止损卖单执行"
            else:
                return False, 0, "止损单未触发"
        
        return False, 0, "未知订单类型"


class TradingSignal:
    """交易信号类"""
    
    def __init__(self, signal_type: SignalType, symbol: str, strength: float = 1.0):
        self.signal_type = signal_type
        self.symbol = symbol
        self.strength = strength  # 信号强度 0-1
        self.timestamp = datetime.datetime.now()
        self.confidence = 1.0     # 信号置信度 0-1
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'signal_type': self.signal_type.value,
            'symbol': self.symbol,
            'strength': self.strength,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat()
        }


class PerformanceMetrics:
    """绩效指标类"""
    
    def __init__(self, returns: List[float]):
        self.returns = returns
        
    def total_return(self) -> float:
        """总收益率"""
        if not self.returns:
            return 0.0
        total = 1.0
        for r in self.returns:
            total *= (1 + r)
        return total - 1
    
    def annualized_return(self) -> float:
        """年化收益率"""
        total_return = self.total_return()
        periods = len(self.returns)
        if periods == 0:
            return 0.0
        return (1 + total_return) ** (252 / periods) - 1
    
    def volatility(self) -> float:
        """波动率"""
        if not self.returns:
            return 0.0
        mean_return = sum(self.returns) / len(self.returns)
        variance = sum((r - mean_return) ** 2 for r in self.returns) / len(self.returns)
        return math.sqrt(variance) * math.sqrt(252)
    
    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """夏普比率"""
        ann_return = self.annualized_return()
        ann_volatility = self.volatility()
        if ann_volatility == 0:
            return 0.0
        return (ann_return - risk_free_rate) / ann_volatility
    
    def max_drawdown(self) -> float:
        """最大回撤"""
        if not self.returns:
            return 0.0
        
        cumulative = []
        total = 1.0
        for r in self.returns:
            total *= (1 + r)
            cumulative.append(total)
        
        running_max = []
        max_val = cumulative[0]
        for val in cumulative:
            if val > max_val:
                max_val = val
            running_max.append(max_val)
        
        drawdown = []
        for i in range(len(cumulative)):
            dd = (cumulative[i] - running_max[i]) / running_max[i]
            drawdown.append(dd)
        
        return min(drawdown)
    
    def win_rate(self) -> float:
        """胜率"""
        if not self.returns:
            return 0.0
        winning_trades = sum(1 for r in self.returns if r > 0)
        return winning_trades / len(self.returns)
    
    def profit_loss_ratio(self) -> float:
        """盈亏比"""
        if not self.returns:
            return 0.0
        
        profits = [r for r in self.returns if r > 0]
        losses = [r for r in self.returns if r < 0]
        
        if not profits or not losses:
            return 0.0
        
        avg_profit = sum(profits) / len(profits)
        avg_loss = sum(losses) / len(losses)
        return abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf')


class TradingReport:
    """交易报告类"""
    
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.report_data = {}
    
    def generate_report(self) -> Dict:
        """生成交易报告"""
        portfolio_value = self.portfolio.get_portfolio_value()
        total_pnl = self.portfolio.get_portfolio_pnl()
        
        # 绩效指标
        if self.portfolio.daily_returns:
            metrics = PerformanceMetrics(self.portfolio.daily_returns)
            performance = {
                'total_return': metrics.total_return(),
                'annualized_return': metrics.annualized_return(),
                'volatility': metrics.volatility(),
                'sharpe_ratio': metrics.sharpe_ratio(),
                'max_drawdown': metrics.max_drawdown(),
                'win_rate': metrics.win_rate(),
                'profit_loss_ratio': metrics.profit_loss_ratio()
            }
        else:
            performance = {key: 0.0 for key in [
                'total_return', 'annualized_return', 'volatility', 
                'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_loss_ratio'
            ]}
        
        # 持仓信息
        positions_info = self.portfolio.get_positions_summary()
        
        # 交易统计
        total_trades = len(self.portfolio.transaction_history)
        total_commission = sum(t['commission'] for t in self.portfolio.transaction_history)
        
        # 组合信息
        portfolio_info = {
            'initial_cash': self.portfolio.initial_cash,
            'current_cash': self.portfolio.cash,
            'portfolio_value': portfolio_value,
            'total_pnl': total_pnl,
            'total_commission': total_commission,
            'total_trades': total_trades
        }
        
        self.report_data = {
            'portfolio': portfolio_info,
            'positions': positions_info,
            'performance': performance,
            'transactions': self.portfolio.transaction_history[-10:],  # 最近10笔交易
            'equity_curve': self.portfolio.equity_curve[-50:]  # 最近50个权益点
        }
        
        return self.report_data
    
    def save_report(self, filename: str) -> None:
        """保存报告到文件"""
        if not self.report_data:
            self.generate_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.report_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"交易报告已保存到: {filename}")
    
    def print_summary(self) -> None:
        """打印报告摘要"""
        if not self.report_data:
            self.generate_report()
        
        portfolio = self.report_data['portfolio']
        performance = self.report_data['performance']
        
        print("\n" + "="*60)
        print("P6模拟交易器 - 交易报告摘要")
        print("="*60)
        print(f"初始资金:     {portfolio['initial_cash']:,.2f}")
        print(f"当前现金:     {portfolio['current_cash']:,.2f}")
        print(f"组合价值:     {portfolio['portfolio_value']:,.2f}")
        print(f"总盈亏:       {portfolio['total_pnl']:,.2f}")
        print(f"总手续费:     {portfolio['total_commission']:,.2f}")
        print(f"交易次数:     {portfolio['total_trades']}")
        print("-"*60)
        print(f"总收益率:     {performance['total_return']:.2%}")
        print(f"年化收益率:   {performance['annualized_return']:.2%}")
        print(f"波动率:       {performance['volatility']:.2%}")
        print(f"夏普比率:     {performance['sharpe_ratio']:.3f}")
        print(f"最大回撤:     {performance['max_drawdown']:.2%}")
        print(f"胜率:         {performance['win_rate']:.2%}")
        print(f"盈亏比:       {performance['profit_loss_ratio']:.2f}")
        print("="*60)


class SimulatedTrader:
    """主要的模拟交易器类"""
    
    def __init__(self, 
                 initial_cash: float = 1000000.0,
                 volatility: float = 0.02,
                 trend: float = 0.001):
        """初始化模拟交易器"""
        self.portfolio = Portfolio(initial_cash)
        self.risk_manager = RiskManager()
        self.market_simulator = MarketSimulator(volatility, trend)
        self.trading_signals: List[TradingSignal] = []
        self.is_running = False
        self.current_time = datetime.datetime.now()
        
        logger.info(f"P6模拟交易器初始化完成，初始资金: {initial_cash:,.2f}")
    
    def add_symbol(self, symbol: str, initial_price: float = 100.0) -> None:
        """添加交易标的"""
        self.market_simulator.set_initial_prices([symbol], {symbol: initial_price})
        logger.info(f"添加交易标的: {symbol}, 初始价格: {initial_price}")
    
    def submit_order(self, order: Order) -> bool:
        """提交订单"""
        try:
            # 风险检查
            if order.side == OrderSide.BUY:
                can_trade, reason = self.risk_manager.check_position_size(
                    self.portfolio, order.symbol, order.quantity, 
                    self.market_simulator.get_market_price(order.symbol)
                )
                if not can_trade:
                    logger.warning(f"订单被风险检查拒绝: {reason}")
                    order.status = OrderStatus.REJECTED
                    return False
            
            # 市场模拟执行
            can_execute, execution_price, message = self.market_simulator.simulate_order_execution(order)
            
            if can_execute:
                success = self.portfolio.execute_order(order, execution_price)
                if success:
                    logger.info(f"订单执行成功: {message}")
                    return True
                else:
                    logger.error("订单执行失败")
                    return False
            else:
                logger.info(f"订单未触发执行条件: {message}")
                order.status = OrderStatus.REJECTED
                return False
                
        except Exception as e:
            logger.error(f"订单提交失败: {e}")
            order.status = OrderStatus.REJECTED
            return False
    
    def generate_trading_signals(self, strategy_func=None) -> List[TradingSignal]:
        """生成交易信号"""
        signals = []
        
        # 如果提供了策略函数，使用策略函数生成信号
        if strategy_func:
            try:
                strategy_signals = strategy_func(self.market_simulator.current_prices)
                for signal_data in strategy_signals:
                    signal = TradingSignal(
                        signal_type=SignalType(signal_data['type']),
                        symbol=signal_data['symbol'],
                        strength=signal_data.get('strength', 1.0)
                    )
                    signals.append(signal)
            except Exception as e:
                logger.error(f"策略信号生成失败: {e}")
        
        # 检查风险信号
        risk_signals = self.risk_manager.generate_risk_signals(self.portfolio)
        for signal_data in risk_signals:
            signal = TradingSignal(
                signal_type=SignalType(signal_data['action']),
                symbol=signal_data['symbol'],
                strength=0.8  # 风险信号优先级较高
            )
            signals.append(signal)
        
        self.trading_signals.extend(signals)
        return signals
    
    def execute_signals(self, signals: List[TradingSignal]) -> List[bool]:
        """执行交易信号"""
        results = []
        
        for signal in signals:
            try:
                current_price = self.market_simulator.get_market_price(signal.symbol)
                
                if signal.signal_type == SignalType.BUY:
                    # 计算买入数量（基于信号强度和风险限制）
                    portfolio_value = self.portfolio.get_portfolio_value()
                    max_trade_value = portfolio_value * 0.05 * signal.strength  # 最多5%仓位
                    quantity = max_trade_value / current_price
                    
                    order = Order(
                        symbol=signal.symbol,
                        side=OrderSide.BUY,
                        quantity=quantity,
                        order_type=OrderType.MARKET
                    )
                    
                elif signal.signal_type == SignalType.SELL:
                    # 卖出所有持仓
                    position = self.portfolio.get_position(signal.symbol)
                    quantity = abs(position.quantity)
                    
                    if quantity > 0:
                        order = Order(
                            symbol=signal.symbol,
                            side=OrderSide.SELL,
                            quantity=quantity,
                            order_type=OrderType.MARKET
                        )
                    else:
                        results.append(False)
                        continue
                else:  # HOLD
                    results.append(True)
                    continue
                
                success = self.submit_order(order)
                results.append(success)
                
            except Exception as e:
                logger.error(f"信号执行失败: {e}")
                results.append(False)
        
        return results
    
    def update_market(self) -> None:
        """更新市场数据"""
        # 模拟所有标的价格变动
        for symbol in self.market_simulator.current_prices.keys():
            self.market_simulator.simulate_price_movement(symbol)
        
        # 更新组合市值
        self.market_simulator.update_market_prices(self.market_simulator.current_prices)
        self.portfolio.update_market_prices(self.market_simulator.current_prices)
        
        # 更新当前时间
        self.current_time = datetime.datetime.now()
    
    def run_strategy(self, strategy_func=None, periods: int = 100) -> None:
        """运行交易策略"""
        self.is_running = True
        logger.info(f"开始运行交易策略，模拟 {periods} 个周期")
        
        for period in range(periods):
            if not self.is_running:
                break
            
            # 更新市场数据
            self.update_market()
            
            # 生成交易信号
            signals = self.generate_trading_signals(strategy_func)
            
            # 执行交易信号
            if signals:
                results = self.execute_signals(signals)
                successful_trades = sum(results)
                logger.info(f"周期 {period+1}: 生成 {len(signals)} 个信号，成功执行 {successful_trades} 个")
            
            # 风险检查
            can_trade, risk_message, risk_metrics = self.risk_manager.check_portfolio_risk(self.portfolio)
            if not can_trade:
                logger.warning(f"风险检查失败: {risk_message}")
                # 可以在这里添加风险控制措施
            
            # 每10个周期打印一次状态
            if (period + 1) % 10 == 0:
                portfolio_value = self.portfolio.get_portfolio_value()
                total_pnl = self.portfolio.get_portfolio_pnl()
                logger.info(f"周期 {period+1}: 组合价值 {portfolio_value:,.2f}, 总盈亏 {total_pnl:,.2f}")
        
        self.is_running = False
        logger.info("交易策略运行完成")
    
    def stop_strategy(self) -> None:
        """停止交易策略"""
        self.is_running = False
        logger.info("交易策略已停止")
    
    def get_status(self) -> Dict:
        """获取交易器状态"""
        portfolio_value = self.portfolio.get_portfolio_value()
        total_pnl = self.portfolio.get_portfolio_pnl()
        
        return {
            'is_running': self.is_running,
            'current_time': self.current_time.isoformat(),
            'portfolio_value': portfolio_value,
            'total_pnl': total_pnl,
            'cash': self.portfolio.cash,
            'total_trades': len(self.portfolio.transaction_history),
            'open_positions': len([p for p in self.portfolio.positions.values() if p.quantity != 0]),
            'market_prices': self.market_simulator.current_prices.copy()
        }
    
    def generate_report(self) -> TradingReport:
        """生成交易报告"""
        return TradingReport(self.portfolio)
    
    def save_state(self, filename: str) -> None:
        """保存交易器状态"""
        state = {
            'portfolio': {
                'initial_cash': self.portfolio.initial_cash,
                'cash': self.portfolio.cash,
                'positions': {symbol: {
                    'quantity': pos.quantity,
                    'average_price': pos.average_price,
                    'market_price': pos.market_price,
                    'unrealized_pnl': pos.unrealized_pnl,
                    'realized_pnl': pos.realized_pnl,
                    'total_commission': pos.total_commission
                } for symbol, pos in self.portfolio.positions.items()},
                'transaction_history': self.portfolio.transaction_history,
                'equity_curve': self.portfolio.equity_curve,
                'daily_returns': self.portfolio.daily_returns
            },
            'market': {
                'current_prices': self.market_simulator.current_prices,
                'price_history': self.market_simulator.price_history
            },
            'status': self.get_status()
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"交易器状态已保存到: {filename}")
    
    def load_state(self, filename: str) -> None:
        """加载交易器状态"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # 恢复组合状态
            portfolio_data = state['portfolio']
            self.portfolio.initial_cash = portfolio_data['initial_cash']
            self.portfolio.cash = portfolio_data['cash']
            
            for symbol, pos_data in portfolio_data['positions'].items():
                position = Position(symbol)
                position.quantity = pos_data['quantity']
                position.average_price = pos_data['average_price']
                position.market_price = pos_data['market_price']
                position.unrealized_pnl = pos_data['unrealized_pnl']
                position.realized_pnl = pos_data['realized_pnl']
                position.total_commission = pos_data['total_commission']
                self.portfolio.positions[symbol] = position
            
            self.portfolio.transaction_history = portfolio_data['transaction_history']
            self.portfolio.equity_curve = portfolio_data['equity_curve']
            self.portfolio.daily_returns = portfolio_data['daily_returns']
            
            # 恢复市场状态
            market_data = state['market']
            self.market_simulator.current_prices = market_data['current_prices']
            self.market_simulator.price_history = market_data['price_history']
            
            logger.info(f"交易器状态已从 {filename} 加载")
            
        except Exception as e:
            logger.error(f"加载状态失败: {e}")
            raise