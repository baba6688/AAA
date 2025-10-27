"""
OKX模拟交易器
提供完整的模拟交易环境，支持回测和策略测试
包含真实的订单匹配逻辑、资金管理和风险控制
"""

import os
import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from decimal import Decimal, ROUND_DOWN
from datetime import datetime, timedelta
import json
import random
import threading
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, deque
import copy
import hashlib

# 导入配置管理器
try:
    from config_manager import ConfigManager, get_global_config_manager
except ImportError:
    # 备用导入路径
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_manager import ConfigManager, get_global_config_manager


class SimOrderType(Enum):
    """模拟订单类型"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class SimOrderSide(Enum):
    """模拟订单方向"""
    BUY = "buy"
    SELL = "sell"


class SimOrderStatus(Enum):
    """模拟订单状态"""
    PENDING = "pending"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class SimOrder:
    """模拟订单数据类"""
    order_id: str
    client_oid: str
    symbol: str
    side: SimOrderSide
    order_type: SimOrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    status: SimOrderStatus = SimOrderStatus.PENDING
    filled_quantity: Decimal = Decimal('0')
    average_price: Decimal = Decimal('0')
    fee: Decimal = Decimal('0')
    fee_currency: str = ""
    create_time: float = None
    update_time: float = None
    expire_time: float = None
    leverage: str = "1"
    
    def __post_init__(self):
        if self.create_time is None:
            self.create_time = time.time()
        if self.update_time is None:
            self.update_time = self.create_time


@dataclass
class SimPosition:
    """模拟持仓数据类"""
    symbol: str
    quantity: Decimal
    average_price: Decimal
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    margin: Decimal = Decimal('0')
    leverage: str = "1"
    last_update: float = None
    
    def __post_init__(self):
        if self.last_update is None:
            self.last_update = time.time()


@dataclass
class SimBalance:
    """模拟余额数据类"""
    currency: str
    total_balance: Decimal
    available_balance: Decimal
    frozen_balance: Decimal = Decimal('0')


class MarketDataGenerator:
    """市场数据生成器"""
    
    def __init__(self, symbols: List[str], initial_prices: Dict[str, Decimal] = None):
        self.symbols = symbols
        self.initial_prices = initial_prices or {}
        self.current_prices = {}
        self.volatility = defaultdict(lambda: Decimal('0.02'))  # 默认2%波动率
        self.trend = defaultdict(lambda: Decimal('0.0001'))  # 默认微小上涨趋势
        self.spread = defaultdict(lambda: Decimal('0.001'))  # 默认0.1%点差
        
        # 初始化价格
        for symbol in symbols:
            if symbol in initial_prices:
                self.current_prices[symbol] = initial_prices[symbol]
            else:
                # 默认价格
                if 'BTC' in symbol:
                    self.current_prices[symbol] = Decimal('50000')
                elif 'ETH' in symbol:
                    self.current_prices[symbol] = Decimal('3000')
                else:
                    self.current_prices[symbol] = Decimal('1')
    
    def generate_tick(self, symbol: str) -> Dict[str, Any]:
        """生成一个tick数据"""
        if symbol not in self.current_prices:
            return {}
        
        current_price = self.current_prices[symbol]
        
        # 随机波动
        change_percent = Decimal(str(random.gauss(0, float(self.volatility[symbol]) / 10)))
        trend_effect = self.trend[symbol]
        total_change = change_percent + trend_effect
        
        new_price = current_price * (1 + total_change)
        
        # 确保价格为正
        new_price = max(new_price, Decimal('0.0001'))
        
        # 更新当前价格
        self.current_prices[symbol] = new_price
        
        # 计算买卖价
        spread_amount = new_price * self.spread[symbol]
        bid_price = new_price - spread_amount / 2
        ask_price = new_price + spread_amount / 2
        
        # 生成随机成交量
        base_volume = Decimal('1000') if 'BTC' in symbol else Decimal('100')
        volume_variation = Decimal(str(random.uniform(0.5, 2.0)))
        volume = base_volume * volume_variation
        
        return {
            'symbol': symbol,
            'last': new_price,
            'bid': bid_price,
            'ask': ask_price,
            'high': new_price * Decimal('1.01'),  # 简化实现
            'low': new_price * Decimal('0.99'),   # 简化实现
            'volume': volume,
            'timestamp': time.time()
        }
    
    def set_volatility(self, symbol: str, volatility: Decimal):
        """设置波动率"""
        self.volatility[symbol] = volatility
    
    def set_trend(self, symbol: str, trend: Decimal):
        """设置趋势"""
        self.trend[symbol] = trend
    
    def set_spread(self, symbol: str, spread: Decimal):
        """设置点差"""
        self.spread[symbol] = spread


class OKXSimulator:
    """OKX模拟交易器 - 完整版实现"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        
        # 账户状态
        self._balances = defaultdict(lambda: SimBalance('', Decimal('0'), Decimal('0')))
        self._positions = {}
        self._orders = {}  # 活跃订单
        self._order_history = deque(maxlen=10000)  # 订单历史
        
        # 市场数据
        self.trading_pairs = self.config.get('trading_pairs', ['BTC-USDT', 'ETH-USDT', 'ADA-USDT'])
        initial_balance = self.config.get('initial_balance', {
            'USDT': Decimal('10000'),
            'BTC': Decimal('0'),
            'ETH': Decimal('0'),
            'ADA': Decimal('0')
        })
        
        # 初始化余额
        for currency, amount in initial_balance.items():
            self._balances[currency] = SimBalance(
                currency=currency,
                total_balance=amount,
                available_balance=amount
            )
        
        # 市场数据生成器
        initial_prices = {
            'BTC-USDT': Decimal('50000'),
            'ETH-USDT': Decimal('3000'),
            'ADA-USDT': Decimal('0.5')
        }
        self.market_generator = MarketDataGenerator(self.trading_pairs, initial_prices)
        
        # 订单匹配引擎
        self._orderbook = defaultdict(lambda: {'bids': [], 'asks': []})
        self._last_trade_price = {}
        
        # 交易配置
        self.fee_rate = Decimal(str(self.config.get('fee_rate', 0.001)))  # 默认0.1%手续费
        self.slippage = Decimal(str(self.config.get('slippage', 0.001)))  # 默认0.1%滑点
        
        # 性能统计
        self._stats = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_volume': Decimal('0'),
            'total_fees': Decimal('0'),
            'realized_pnl': Decimal('0'),
            'start_time': time.time()
        }
        
        # 线程安全
        self._data_lock = threading.RLock()
        
        # 模拟时间控制
        self._simulation_speed = self.config.get('simulation_speed', 1.0)  # 模拟速度倍数
        self._last_market_update = time.time()
        
        self.logger.info("OKX模拟交易器初始化完成")
        self.logger.info(f"初始余额: {initial_balance}")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('OKXSimulator')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _generate_order_id(self) -> str:
        """生成唯一订单ID"""
        timestamp = int(time.time() * 1000)
        random_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"SIM_{timestamp}_{random_part}"
    
    def _update_market_data(self):
        """更新市场数据"""
        current_time = time.time()
        if current_time - self._last_market_update < (1.0 / self._simulation_speed):
            return
        
        with self._data_lock:
            for symbol in self.trading_pairs:
                # 生成新的市场数据
                tick_data = self.market_generator.generate_tick(symbol)
                if tick_data:
                    self._last_trade_price[symbol] = tick_data['last']
                    
                    # 更新订单簿（简化实现）
                    self._orderbook[symbol] = {
                        'bids': [[tick_data['bid'], Decimal('10')]],
                        'asks': [[tick_data['ask'], Decimal('10')]]
                    }
            
            self._last_market_update = current_time
    
    def _calculate_fee(self, symbol: str, quantity: Decimal, price: Decimal, side: SimOrderSide) -> Decimal:
        """计算手续费"""
        trade_value = quantity * price
        fee = trade_value * self.fee_rate
        
        # 确定手续费币种
        base_currency = symbol.split('-')[0]
        quote_currency = symbol.split('-')[1]
        fee_currency = quote_currency if side == SimOrderSide.BUY else base_currency
        
        return fee, fee_currency
    
    def _apply_slippage(self, symbol: str, side: SimOrderSide, quantity: Decimal) -> Decimal:
        """应用滑点"""
        if symbol not in self._last_trade_price:
            return self._last_trade_price.get(symbol, Decimal('0'))
        
        base_price = self._last_trade_price[symbol]
        slippage_amount = base_price * self.slippage
        
        if side == SimOrderSide.BUY:
            # 买入时价格更高
            execution_price = base_price + slippage_amount
        else:
            # 卖出时价格更低
            execution_price = base_price - slippage_amount
        
        return execution_price
    
    def _check_balance(self, symbol: str, side: SimOrderSide, quantity: Decimal, price: Decimal) -> Tuple[bool, str]:
        """检查余额"""
        base_currency = symbol.split('-')[0]
        quote_currency = symbol.split('-')[1]
        
        if side == SimOrderSide.BUY:
            # 买入：检查报价货币余额
            required_amount = quantity * price
            fee, fee_currency = self._calculate_fee(symbol, quantity, price, side)
            
            if fee_currency == quote_currency:
                total_required = required_amount + fee
            else:
                total_required = required_amount
            
            available_balance = self._balances[quote_currency].available_balance
            
            if available_balance < total_required:
                return False, f"余额不足: 需要 {total_required} {quote_currency}, 可用 {available_balance}"
        
        else:  # SELL
            # 卖出：检查基础货币余额
            available_balance = self._balances[base_currency].available_balance
            
            if available_balance < quantity:
                return False, f"余额不足: 需要 {quantity} {base_currency}, 可用 {available_balance}"
        
        return True, "余额充足"
    
    def _freeze_balance(self, symbol: str, side: SimOrderSide, quantity: Decimal, price: Decimal):
        """冻结资金"""
        base_currency = symbol.split('-')[0]
        quote_currency = symbol.split('-')[1]
        
        if side == SimOrderSide.BUY:
            # 买入：冻结报价货币
            required_amount = quantity * price
            fee, fee_currency = self._calculate_fee(symbol, quantity, price, side)
            
            if fee_currency == quote_currency:
                total_required = required_amount + fee
            else:
                total_required = required_amount
            
            self._balances[quote_currency].available_balance -= total_required
            self._balances[quote_currency].frozen_balance += total_required
        
        else:  # SELL
            # 卖出：冻结基础货币
            self._balances[base_currency].available_balance -= quantity
            self._balances[base_currency].frozen_balance += quantity
    
    def _execute_market_order(self, order: SimOrder) -> bool:
        """执行市价订单"""
        # 更新市场数据
        self._update_market_data()
        
        # 应用滑点获取执行价格
        execution_price = self._apply_slippage(order.symbol, order.side, order.quantity)
        
        # 检查余额
        balance_ok, balance_msg = self._check_balance(order.symbol, order.side, order.quantity, execution_price)
        if not balance_ok:
            order.status = SimOrderStatus.REJECTED
            order.update_time = time.time()
            self.logger.warning(f"订单被拒绝: {balance_msg}")
            return False
        
        # 冻结资金
        self._freeze_balance(order.symbol, order.side, order.quantity, execution_price)
        
        # 计算手续费
        fee, fee_currency = self._calculate_fee(order.symbol, order.quantity, execution_price, order.side)
        
        # 执行交易
        base_currency = order.symbol.split('-')[0]
        quote_currency = order.symbol.split('-')[1]
        
        if order.side == SimOrderSide.BUY:
            # 买入：减少报价货币，增加基础货币
            cost = order.quantity * execution_price + fee
            
            # 解冻并扣除资金
            self._balances[quote_currency].frozen_balance -= cost
            self._balances[quote_currency].total_balance -= cost
            
            # 增加基础货币
            self._balances[base_currency].total_balance += order.quantity
            self._balances[base_currency].available_balance += order.quantity
            
        else:  # SELL
            # 卖出：减少基础货币，增加报价货币
            revenue = order.quantity * execution_price - fee
            
            # 解冻基础货币
            self._balances[base_currency].frozen_balance -= order.quantity
            self._balances[base_currency].total_balance -= order.quantity
            
            # 增加报价货币
            self._balances[quote_currency].total_balance += revenue
            self._balances[quote_currency].available_balance += revenue
        
        # 更新订单状态
        order.status = SimOrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_price = execution_price
        order.fee = fee
        order.fee_currency = fee_currency
        order.update_time = time.time()
        
        # 更新统计
        self._stats['filled_orders'] += 1
        self._stats['total_volume'] += order.quantity
        self._stats['total_fees'] += fee
        
        self.logger.info(
            f"市价订单执行: {order.side.value} {order.quantity} {order.symbol} "
            f"@ {execution_price}, 手续费: {fee} {fee_currency}"
        )
        
        return True
    
    def _execute_limit_order(self, order: SimOrder) -> bool:
        """执行限价订单"""
        # 更新市场数据
        self._update_market_data()
        
        # 检查余额
        balance_ok, balance_msg = self._check_balance(order.symbol, order.side, order.quantity, order.price)
        if not balance_ok:
            order.status = SimOrderStatus.REJECTED
            order.update_time = time.time()
            self.logger.warning(f"订单被拒绝: {balance_msg}")
            return False
        
        # 冻结资金
        self._freeze_balance(order.symbol, order.side, order.quantity, order.price)
        
        # 检查是否可以立即成交
        current_price = self._last_trade_price.get(order.symbol, Decimal('0'))
        
        if order.side == SimOrderSide.BUY:
            # 买入限价单：当前价格 <= 限价时可以成交
            can_fill = current_price <= order.price
        else:
            # 卖出限价单：当前价格 >= 限价时可以成交
            can_fill = current_price >= order.price
        
        if can_fill:
            return self._execute_market_order(order)
        else:
            # 放入订单簿等待成交
            order.status = SimOrderStatus.OPEN
            order.update_time = time.time()
            
            # 添加到订单簿
            if order.side == SimOrderSide.BUY:
                self._orderbook[order.symbol]['bids'].append([order.price, order.quantity])
            else:
                self._orderbook[order.symbol]['asks'].append([order.price, order.quantity])
            
            self.logger.info(
                f"限价订单挂单: {order.side.value} {order.quantity} {order.symbol} @ {order.price}"
            )
            
            return True
    
    def create_market_order(self, symbol: str, side: str, quantity: Decimal, 
                           client_oid: str = None) -> Dict[str, Any]:
        """创建市价订单"""
        with self._data_lock:
            # 生成订单ID
            order_id = self._generate_order_id()
            client_oid = client_oid or order_id
            
            # 创建订单对象
            order = SimOrder(
                order_id=order_id,
                client_oid=client_oid,
                symbol=symbol,
                side=SimOrderSide.BUY if side.lower() == 'buy' else SimOrderSide.SELL,
                order_type=SimOrderType.MARKET,
                quantity=quantity,
                create_time=time.time(),
                update_time=time.time()
            )
            
            # 执行订单
            success = self._execute_market_order(order)
            
            # 记录订单
            self._orders[order_id] = order
            self._order_history.append(copy.deepcopy(order))
            self._stats['total_orders'] += 1
            
            if success:
                return {
                    'order_id': order_id,
                    'client_oid': client_oid,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'status': 'filled',
                    'timestamp': time.time()
                }
            else:
                return {'error': '订单执行失败'}
    
    def create_limit_order(self, symbol: str, side: str, quantity: Decimal, 
                          price: Decimal, client_oid: str = None) -> Dict[str, Any]:
        """创建限价订单"""
        with self._data_lock:
            # 生成订单ID
            order_id = self._generate_order_id()
            client_oid = client_oid or order_id
            
            # 创建订单对象
            order = SimOrder(
                order_id=order_id,
                client_oid=client_oid,
                symbol=symbol,
                side=SimOrderSide.BUY if side.lower() == 'buy' else SimOrderSide.SELL,
                order_type=SimOrderType.LIMIT,
                quantity=quantity,
                price=price,
                create_time=time.time(),
                update_time=time.time()
            )
            
            # 执行订单
            success = self._execute_limit_order(order)
            
            # 记录订单
            self._orders[order_id] = order
            self._order_history.append(copy.deepcopy(order))
            self._stats['total_orders'] += 1
            
            if success:
                status = 'filled' if order.status == SimOrderStatus.FILLED else 'open'
                return {
                    'order_id': order_id,
                    'client_oid': client_oid,
                    'symbol': symbol,
                    'side': side,
                    'quantity': quantity,
                    'price': price,
                    'status': status,
                    'timestamp': time.time()
                }
            else:
                return {'error': '订单创建失败'}
    
    def cancel_order(self, order_id: str, symbol: str) -> bool:
        """取消订单"""
        with self._data_lock:
            if order_id not in self._orders:
                return False
            
            order = self._orders[order_id]
            
            if order.status not in [SimOrderStatus.PENDING, SimOrderStatus.OPEN]:
                return False
            
            # 解冻资金
            base_currency = symbol.split('-')[0]
            quote_currency = symbol.split('-')[1]
            
            if order.side == SimOrderSide.BUY:
                # 买入订单：解冻报价货币
                frozen_amount = order.quantity * (order.price or Decimal('0'))
                self._balances[quote_currency].frozen_balance -= frozen_amount
                self._balances[quote_currency].available_balance += frozen_amount
            else:
                # 卖出订单：解冻基础货币
                self._balances[base_currency].frozen_balance -= order.quantity
                self._balances[base_currency].available_balance += order.quantity
            
            # 更新订单状态
            order.status = SimOrderStatus.CANCELLED
            order.update_time = time.time()
            
            # 从订单簿移除
            if order.order_type == SimOrderType.LIMIT:
                if order.side == SimOrderSide.BUY:
                    self._orderbook[symbol]['bids'] = [
                        [p, q] for p, q in self._orderbook[symbol]['bids'] 
                        if p != order.price or q != order.quantity
                    ]
                else:
                    self._orderbook[symbol]['asks'] = [
                        [p, q] for p, q in self._orderbook[symbol]['asks'] 
                        if p != order.price or q != order.quantity
                    ]
            
            # 更新统计
            self._stats['cancelled_orders'] += 1
            
            self.logger.info(f"订单已取消: {order_id} {symbol}")
            return True
    
    def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """获取订单状态"""
        with self._data_lock:
            if order_id not in self._orders:
                return {}
            
            order = self._orders[order_id]
            
            return {
                'order_id': order.order_id,
                'client_oid': order.client_oid,
                'symbol': order.symbol,
                'side': order.side.value,
                'order_type': order.order_type.value,
                'quantity': order.quantity,
                'filled_quantity': order.filled_quantity,
                'price': order.price,
                'average_price': order.average_price,
                'status': order.status.value,
                'fee': order.fee,
                'fee_currency': order.fee_currency,
                'create_time': order.create_time,
                'update_time': order.update_time
            }
    
    def get_open_orders(self, symbol: str = None) -> List[Dict[str, Any]]:
        """获取未成交订单"""
        with self._data_lock:
            open_orders = []
            
            for order in self._orders.values():
                if order.status in [SimOrderStatus.OPEN, SimOrderStatus.PENDING]:
                    if symbol is None or order.symbol == symbol:
                        open_orders.append({
                            'order_id': order.order_id,
                            'symbol': order.symbol,
                            'side': order.side.value,
                            'order_type': order.order_type.value,
                            'quantity': order.quantity,
                            'filled_quantity': order.filled_quantity,
                            'price': order.price,
                            'status': order.status.value,
                            'create_time': order.create_time
                        })
            
            return open_orders
    
    def get_account_balance(self) -> Dict[str, Any]:
        """获取账户余额"""
        with self._data_lock:
            balance_data = {
                'total': {},
                'free': {},
                'used': {}
            }
            
            for currency, balance in self._balances.items():
                balance_data['total'][currency] = balance.total_balance
                balance_data['free'][currency] = balance.available_balance
                balance_data['used'][currency] = balance.frozen_balance
            
            return balance_data
    
    def get_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """获取持仓信息"""
        # 简化实现 - 基于余额计算持仓
        with self._data_lock:
            positions = []
            
            for currency, balance in self._balances.items():
                if balance.total_balance > Decimal('0') and currency != 'USDT':
                    # 为每种货币创建一个持仓
                    symbol = f"{currency}-USDT"
                    current_price = self._last_trade_price.get(symbol, Decimal('1'))
                    
                    position = {
                        'symbol': symbol,
                        'position_side': 'net',
                        'quantity': balance.total_balance,
                        'available_quantity': balance.available_balance,
                        'average_price': current_price,  # 简化
                        'leverage': '1',
                        'unrealized_pnl': Decimal('0'),  # 简化
                        'margin': Decimal('0'),
                        'liquidation_price': Decimal('0')
                    }
                    
                    if symbol is None or position['symbol'] == symbol:
                        positions.append(position)
            
            return positions
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """获取行情数据"""
        self._update_market_data()
        
        with self._data_lock:
            return self.market_generator.generate_tick(symbol)
    
    def reset_simulator(self, initial_balance: Dict = None):
        """重置模拟器"""
        with self._data_lock:
            # 重置余额
            self._balances.clear()
            
            initial_balance = initial_balance or self.config.get('initial_balance', {
                'USDT': Decimal('10000'),
                'BTC': Decimal('0'),
                'ETH': Decimal('0'),
                'ADA': Decimal('0')
            })
            
            for currency, amount in initial_balance.items():
                self._balances[currency] = SimBalance(
                    currency=currency,
                    total_balance=amount,
                    available_balance=amount
                )
            
            # 重置订单
            self._orders.clear()
            self._order_history.clear()
            
            # 重置统计
            self._stats.update({
                'total_orders': 0,
                'filled_orders': 0,
                'cancelled_orders': 0,
                'rejected_orders': 0,
                'total_volume': Decimal('0'),
                'total_fees': Decimal('0'),
                'realized_pnl': Decimal('0'),
                'start_time': time.time()
            })
            
            self.logger.info("模拟器已重置")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        with self._data_lock:
            stats = self._stats.copy()
            uptime = time.time() - stats['start_time']
            
            stats['uptime'] = uptime
            stats['orders_per_second'] = stats['total_orders'] / uptime if uptime > 0 else 0
            
            if stats['total_orders'] > 0:
                stats['fill_rate'] = stats['filled_orders'] / stats['total_orders']
                stats['cancel_rate'] = stats['cancelled_orders'] / stats['total_orders']
            else:
                stats['fill_rate'] = 0
                stats['cancel_rate'] = 0
            
            return stats
    
    def set_market_conditions(self, symbol: str, volatility: float = None, 
                             trend: float = None, spread: float = None):
        """设置市场条件"""
        if volatility is not None:
            self.market_generator.set_volatility(symbol, Decimal(str(volatility)))
        if trend is not None:
            self.market_generator.set_trend(symbol, Decimal(str(trend)))
        if spread is not None:
            self.market_generator.set_spread(symbol, Decimal(str(spread)))


# 使用示例
if __name__ == "__main__":
    def test_okx_simulator():
        config = {
            'trading_pairs': ['BTC-USDT', 'ETH-USDT'],
            'initial_balance': {
                'USDT': Decimal('10000'),
                'BTC': Decimal('0.1'),
                'ETH': Decimal('5')
            },
            'fee_rate': 0.001,
            'slippage': 0.001
        }
        
        simulator = OKXSimulator(config)
        
        # 测试市价买入
        result = simulator.create_market_order('BTC-USDT', 'buy', Decimal('0.01'))
        print("市价买入结果:", result)
        
        # 测试限价卖出
        result = simulator.create_limit_order('ETH-USDT', 'sell', Decimal('1'), Decimal('3200'))
        print("限价卖出结果:", result)
        
        # 获取余额
        balance = simulator.get_account_balance()
        print("账户余额:", balance)
        
        # 获取未成交订单
        open_orders = simulator.get_open_orders()
        print("未成交订单:", open_orders)
        
        # 获取统计
        stats = simulator.get_performance_stats()
        print("性能统计:", stats)
    
    test_okx_simulator()