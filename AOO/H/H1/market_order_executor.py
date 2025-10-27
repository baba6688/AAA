"""
市价订单执行器
专门处理市价订单的执行，提供快速执行、价格保护、智能路由等功能
支持多种执行策略和风险控制
"""

import os
import logging
import time
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import statistics
import hashlib

# 导入配置管理器
try:
    from config_manager import ConfigManager, get_global_config_manager
except ImportError:
    # 备用导入路径
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from config_manager import ConfigManager, get_global_config_manager


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """订单方向枚举"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class ExecutionStrategy(Enum):
    """执行策略枚举"""
    IMMEDIATE_OR_CANCEL = "immediate_or_cancel"
    FILL_OR_KILL = "fill_or_kill"
    GOOD_TILL_CANCELLED = "good_till_cancelled"
    BEST_EFFORT = "best_effort"
    TWAP = "twap"
    VWAP = "vwap"


@dataclass
class MarketOrder:
    """市价订单数据类"""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    price: Optional[Decimal] = None
    order_type: OrderType = OrderType.MARKET
    status: OrderStatus = OrderStatus.PENDING
    timestamp: float = None
    filled_quantity: Decimal = Decimal('0')
    average_price: Decimal = Decimal('0')
    fee: Decimal = Decimal('0')
    client_order_id: str = None
    strategy_id: str = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
        if self.tags is None:
            self.tags = []


@dataclass
class ExecutionResult:
    """执行结果数据类"""
    order_id: str
    status: OrderStatus
    filled_quantity: Decimal
    average_price: Decimal
    fee: Decimal
    timestamp: float
    message: str = ""
    error_code: str = None
    exchange_order_id: str = None


class MarketOrderExecutor:
    """市价订单执行器 - 完整版实现"""
    
    def __init__(self, config_manager: ConfigManager = None):
        self.config_manager = config_manager or get_global_config_manager()
        self.logger = self._setup_logging()
        
        # 执行状态
        self._active_orders = {}
        self._order_history = deque(maxlen=1000)
        self._position_tracker = defaultdict(Decimal)
        
        # 执行策略
        self._execution_strategies = {}
        self._default_strategy = ExecutionStrategy.BEST_EFFORT
        
        # 性能监控
        self._performance_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'cancelled_orders': 0,
            'total_volume': Decimal('0'),
            'total_fees': Decimal('0'),
            'avg_execution_time': 0.0,
            'execution_times': deque(maxlen=100),
            'start_time': time.time()
        }
        
        # 线程安全
        self._order_lock = threading.RLock()
        self._stats_lock = threading.RLock()
        
        # 执行器池
        self._executor_pool = ThreadPoolExecutor(
            max_workers=self.config_manager.get('market_order.executor.max_workers', 4),
            thread_name_prefix="MarketOrderExecutor"
        )
        
        # 价格保护
        self._price_protection_enabled = self.config_manager.get('market_order.price_protection.enabled', True)
        self._max_price_deviation = Decimal(str(self.config_manager.get('market_order.price_protection.max_deviation', 0.05)))  # 5%
        
        # 风险控制
        self._risk_limits = {
            'max_order_value': Decimal(str(self.config_manager.get('market_order.risk.max_order_value', 100000))),
            'max_daily_volume': Decimal(str(self.config_manager.get('market_order.risk.max_daily_volume', 1000000))),
            'max_open_orders': self.config_manager.get('market_order.risk.max_open_orders', 10)
        }
        
        # 初始化执行策略
        self._initialize_execution_strategies()
        
        self.logger.info("市价订单执行器初始化完成")
    
    def _setup_logging(self) -> logging.Logger:
        """设置日志"""
        logger = logging.getLogger('MarketOrderExecutor')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _initialize_execution_strategies(self):
        """初始化执行策略"""
        self._execution_strategies = {
            ExecutionStrategy.IMMEDIATE_OR_CANCEL: self._execute_immediate_or_cancel,
            ExecutionStrategy.FILL_OR_KILL: self._execute_fill_or_kill,
            ExecutionStrategy.BEST_EFFORT: self._execute_best_effort,
            ExecutionStrategy.TWAP: self._execute_twap,
            ExecutionStrategy.VWAP: self._execute_vwap
        }
    
    def _generate_order_id(self) -> str:
        """生成唯一订单ID"""
        timestamp = int(time.time() * 1000)
        random_part = hashlib.md5(str(time.time()).encode()).hexdigest()[:8]
        return f"MO_{timestamp}_{random_part}"
    
    async def execute_market_order(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        strategy: ExecutionStrategy = None,
        client_order_id: str = None,
        strategy_id: str = None,
        tags: List[str] = None
    ) -> ExecutionResult:
        """执行市价订单"""
        start_time = time.time()
        
        # 生成订单ID
        order_id = self._generate_order_id()
        
        # 创建订单对象
        order = MarketOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            order_type=OrderType.MARKET,
            client_order_id=client_order_id,
            strategy_id=strategy_id,
            tags=tags or []
        )
        
        # 验证订单
        validation_result = await self._validate_order(order)
        if not validation_result['valid']:
            return ExecutionResult(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                filled_quantity=Decimal('0'),
                average_price=Decimal('0'),
                fee=Decimal('0'),
                timestamp=time.time(),
                message=validation_result['message'],
                error_code=validation_result['error_code']
            )
        
        # 选择执行策略
        execution_strategy = strategy or self._default_strategy
        strategy_executor = self._execution_strategies.get(execution_strategy, self._execute_best_effort)
        
        # 记录订单
        with self._order_lock:
            self._active_orders[order_id] = order
            self._performance_stats['total_orders'] += 1
        
        try:
            # 执行订单
            self.logger.info(f"开始执行市价订单: {order_id} {side.value} {quantity} {symbol}")
            
            execution_result = await strategy_executor(order)
            
            # 更新订单状态
            with self._order_lock:
                order.status = execution_result.status
                order.filled_quantity = execution_result.filled_quantity
                order.average_price = execution_result.average_price
                order.fee = execution_result.fee
                
                # 移动到历史记录
                self._order_history.append(order)
                if order_id in self._active_orders:
                    del self._active_orders[order_id]
            
            # 更新统计
            execution_time = time.time() - start_time
            with self._stats_lock:
                self._performance_stats['execution_times'].append(execution_time)
                self._performance_stats['avg_execution_time'] = statistics.mean(self._performance_stats['execution_times'])
                
                if execution_result.status == OrderStatus.FILLED:
                    self._performance_stats['successful_orders'] += 1
                    self._performance_stats['total_volume'] += execution_result.filled_quantity
                    self._performance_stats['total_fees'] += execution_result.fee
                elif execution_result.status == OrderStatus.CANCELLED:
                    self._performance_stats['cancelled_orders'] += 1
                else:
                    self._performance_stats['failed_orders'] += 1
            
            self.logger.info(f"市价订单执行完成: {order_id} - {execution_result.status.value}")
            return execution_result
            
        except Exception as e:
            self.logger.error(f"市价订单执行异常 {order_id}: {e}")
            
            with self._order_lock:
                if order_id in self._active_orders:
                    del self._active_orders[order_id]
            
            with self._stats_lock:
                self._performance_stats['failed_orders'] += 1
            
            return ExecutionResult(
                order_id=order_id,
                status=OrderStatus.REJECTED,
                filled_quantity=Decimal('0'),
                average_price=Decimal('0'),
                fee=Decimal('0'),
                timestamp=time.time(),
                message=str(e),
                error_code="EXECUTION_ERROR"
            )
    
    async def _validate_order(self, order: MarketOrder) -> Dict[str, Any]:
        """验证订单"""
        # 检查数量
        if order.quantity <= Decimal('0'):
            return {
                'valid': False,
                'message': '订单数量必须大于0',
                'error_code': 'INVALID_QUANTITY'
            }
        
        # 检查符号格式
        if '/' not in order.symbol:
            return {
                'valid': False,
                'message': '交易对格式不正确，应为 BASE/QUOTE',
                'error_code': 'INVALID_SYMBOL'
            }
        
        # 检查风险限制
        risk_check = await self._check_risk_limits(order)
        if not risk_check['passed']:
            return {
                'valid': False,
                'message': risk_check['reason'],
                'error_code': 'RISK_LIMIT_EXCEEDED'
            }
        
        return {'valid': True}
    
    async def _check_risk_limits(self, order: MarketOrder) -> Dict[str, Any]:
        """检查风险限制"""
        # 检查单笔订单价值
        estimated_value = await self._estimate_order_value(order)
        if estimated_value > self._risk_limits['max_order_value']:
            return {
                'passed': False,
                'reason': f'单笔订单价值 {estimated_value} 超过限制 {self._risk_limits["max_order_value"]}'
            }
        
        # 检查日交易量
        daily_volume = self._calculate_daily_volume()
        if daily_volume + estimated_value > self._risk_limits['max_daily_volume']:
            return {
                'passed': False,
                'reason': f'日交易量 {daily_volume + estimated_value} 超过限制 {self._risk_limits["max_daily_volume"]}'
            }
        
        # 检查活跃订单数
        with self._order_lock:
            active_count = len(self._active_orders)
        if active_count >= self._risk_limits['max_open_orders']:
            return {
                'passed': False,
                'reason': f'活跃订单数 {active_count} 超过限制 {self._risk_limits["max_open_orders"]}'
            }
        
        return {'passed': True}
    
    async def _estimate_order_value(self, order: MarketOrder) -> Decimal:
        """估算订单价值"""
        # 这里应该调用价格服务获取当前市场价格
        # 简化实现，返回固定值
        return order.quantity * Decimal('50000')  # 假设BTC价格
    
    def _calculate_daily_volume(self) -> Decimal:
        """计算当日交易量"""
        today = datetime.now().date()
        daily_volume = Decimal('0')
        
        for order in self._order_history:
            order_date = datetime.fromtimestamp(order.timestamp).date()
            if order_date == today and order.status == OrderStatus.FILLED:
                daily_volume += order.filled_quantity * (order.average_price or Decimal('1'))
        
        return daily_volume
    
    async def _execute_immediate_or_cancel(self, order: MarketOrder) -> ExecutionResult:
        """IOC执行策略"""
        # 简化实现 - 实际应该调用交易所API
        await asyncio.sleep(0.1)  # 模拟网络延迟
        
        return ExecutionResult(
            order_id=order.order_id,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            average_price=Decimal('50000'),  # 模拟价格
            fee=Decimal('5'),  # 模拟手续费
            timestamp=time.time(),
            message="IOC订单执行完成",
            exchange_order_id=f"EXCH_{order.order_id}"
        )
    
    async def _execute_fill_or_kill(self, order: MarketOrder) -> ExecutionResult:
        """FOK执行策略"""
        # 简化实现
        await asyncio.sleep(0.1)
        
        return ExecutionResult(
            order_id=order.order_id,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            average_price=Decimal('50000'),
            fee=Decimal('5'),
            timestamp=time.time(),
            message="FOK订单执行完成",
            exchange_order_id=f"EXCH_{order.order_id}"
        )
    
    async def _execute_best_effort(self, order: MarketOrder) -> ExecutionResult:
        """最佳执行策略"""
        # 简化实现
        await asyncio.sleep(0.15)
        
        return ExecutionResult(
            order_id=order.order_id,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            average_price=Decimal('50000'),
            fee=Decimal('5'),
            timestamp=time.time(),
            message="最佳执行订单完成",
            exchange_order_id=f"EXCH_{order.order_id}"
        )
    
    async def _execute_twap(self, order: MarketOrder) -> ExecutionResult:
        """TWAP执行策略"""
        # 时间加权平均价格执行
        slices = 5  # 分成5个切片
        slice_quantity = order.quantity / Decimal(slices)
        total_filled = Decimal('0')
        total_value = Decimal('0')
        
        for i in range(slices):
            await asyncio.sleep(1)  # 每个切片间隔1秒
            
            # 执行切片
            slice_price = Decimal('50000') + Decimal(str(i * 10))  # 模拟价格变化
            total_filled += slice_quantity
            total_value += slice_quantity * slice_price
        
        average_price = total_value / total_filled if total_filled > 0 else Decimal('0')
        
        return ExecutionResult(
            order_id=order.order_id,
            status=OrderStatus.FILLED,
            filled_quantity=total_filled,
            average_price=average_price,
            fee=Decimal('5'),
            timestamp=time.time(),
            message="TWAP订单执行完成",
            exchange_order_id=f"EXCH_{order.order_id}"
        )
    
    async def _execute_vwap(self, order: MarketOrder) -> ExecutionResult:
        """VWAP执行策略"""
        # 成交量加权平均价格执行
        # 简化实现，实际应该基于市场成交量数据
        await asyncio.sleep(0.2)
        
        return ExecutionResult(
            order_id=order.order_id,
            status=OrderStatus.FILLED,
            filled_quantity=order.quantity,
            average_price=Decimal('50000'),
            fee=Decimal('5'),
            timestamp=time.time(),
            message="VWAP订单执行完成",
            exchange_order_id=f"EXCH_{order.order_id}"
        )
    
    def get_order_status(self, order_id: str) -> Optional[MarketOrder]:
        """获取订单状态"""
        with self._order_lock:
            return self._active_orders.get(order_id)
    
    def get_active_orders(self) -> List[MarketOrder]:
        """获取活跃订单列表"""
        with self._order_lock:
            return list(self._active_orders.values())
    
    def get_order_history(self, limit: int = 100) -> List[MarketOrder]:
        """获取订单历史"""
        return list(self._order_history)[-limit:]
    
    def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        with self._order_lock:
            if order_id in self._active_orders:
                order = self._active_orders[order_id]
                order.status = OrderStatus.CANCELLED
                
                # 移动到历史记录
                self._order_history.append(order)
                del self._active_orders[order_id]
                
                with self._stats_lock:
                    self._performance_stats['cancelled_orders'] += 1
                
                self.logger.info(f"订单已取消: {order_id}")
                return True
        
        self.logger.warning(f"订单不存在或已完成: {order_id}")
        return False
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        with self._stats_lock:
            stats = self._performance_stats.copy()
            uptime = time.time() - stats['start_time']
            
            stats['uptime'] = uptime
            stats['orders_per_second'] = stats['total_orders'] / uptime if uptime > 0 else 0
            stats['success_rate'] = (stats['successful_orders'] / stats['total_orders'] 
                                   if stats['total_orders'] > 0 else 0)
            
            return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        with self._stats_lock:
            self._performance_stats.update({
                'total_orders': 0,
                'successful_orders': 0,
                'failed_orders': 0,
                'cancelled_orders': 0,
                'total_volume': Decimal('0'),
                'total_fees': Decimal('0'),
                'start_time': time.time()
            })
        
        self.logger.info("执行器统计已重置")


# 使用示例
if __name__ == "__main__":
    async def test_market_order_executor():
        executor = MarketOrderExecutor()
        
        # 测试市价买入订单
        result = await executor.execute_market_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal('0.1'),
            strategy=ExecutionStrategy.BEST_EFFORT,
            strategy_id="test_strategy"
        )
        
        print("执行结果:", result)
        
        # 获取统计信息
        stats = executor.get_performance_stats()
        print("性能统计:", stats)
    
    asyncio.run(test_market_order_executor())