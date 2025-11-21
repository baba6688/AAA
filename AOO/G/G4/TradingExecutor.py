"""
G4交易执行器
实现交易信号接收和验证、订单生成优化、多交易所执行管理、交易成本优化、
执行监控跟踪、异常处理恢复、执行效果评估等核心功能
"""

import asyncio
import logging
import time
import json
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import hashlib
import uuid
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OrderType(Enum):
    """订单类型"""
    MARKET = "market"           # 市价单
    LIMIT = "limit"             # 限价单
    STOP = "stop"               # 止损单
    STOP_LIMIT = "stop_limit"   # 止损限价单
    TRAILING_STOP = "trailing_stop"  # 跟踪止损单

class OrderSide(Enum):
    """订单方向"""
    BUY = "buy"
    SELL = "sell"

class OrderStatus(Enum):
    """订单状态"""
    PENDING = "pending"         # 待执行
    SUBMITTED = "submitted"     # 已提交
    PARTIAL = "partial"         # 部分成交
    FILLED = "filled"           # 完全成交
    CANCELLED = "cancelled"     # 已取消
    REJECTED = "rejected"       # 已拒绝
    FAILED = "failed"           # 执行失败

class ExecutionStrategy(Enum):
    """执行策略"""
    IMMEDIATE = "immediate"     # 立即执行
    TWAP = "twap"              # 时间加权平均
    VWAP = "vwap"              # 成交量加权平均
    ICEBERG = "iceberg"        # 冰山订单
    SNIPER = "sniper"          # 狙击策略

@dataclass
class TradingSignal:
    """交易信号"""
    signal_id: str
    symbol: str
    side: OrderSide
    quantity: float
    price: Optional[float] = None
    timestamp: datetime = None
    confidence: float = 1.0
    strategy_id: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Order:
    """交易订单"""
    order_id: str
    signal_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    status: OrderStatus
    timestamp: datetime
    exchange: str
    strategy: ExecutionStrategy
    filled_quantity: float = 0.0
    avg_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ExecutionResult:
    """执行结果"""
    order_id: str
    success: bool
    filled_quantity: float
    avg_price: float
    commission: float
    slippage: float
    execution_time: float
    error_message: Optional[str] = None

@dataclass
class ExchangeInfo:
    """交易所信息"""
    name: str
    api_key: str
    api_secret: str
    base_url: str
    is_active: bool = True
    commission_rate: float = 0.001
    min_trade_amount: float = 10.0
    max_trade_amount: float = 1000000.0
    supported_order_types: List[OrderType] = None
    latency_ms: float = 0.0
    
    def __post_init__(self):
        if self.supported_order_types is None:
            self.supported_order_types = [OrderType.MARKET, OrderType.LIMIT]

class ExchangeConnector(ABC):
    """交易所连接器抽象类"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接交易所"""
        pass
    
    @abstractmethod
    async def submit_order(self, order: Order) -> ExecutionResult:
        """提交订单"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """取消订单"""
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """获取订单状态"""
        pass
    
    @abstractmethod
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """获取市场数据"""
        pass

class MockExchangeConnector(ExchangeConnector):
    """模拟交易所连接器"""
    
    def __init__(self, exchange_info: ExchangeInfo):
        self.exchange_info = exchange_info
        self.connected = False
        self.orders = {}
    
    async def connect(self) -> bool:
        """连接模拟交易所"""
        await asyncio.sleep(0.1)  # 模拟连接延迟
        self.connected = True
        logger.info(f"已连接到交易所: {self.exchange_info.name}")
        return True
    
    async def submit_order(self, order: Order) -> ExecutionResult:
        """提交模拟订单"""
        if not self.connected:
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                filled_quantity=0.0,
                avg_price=0.0,
                commission=0.0,
                slippage=0.0,
                execution_time=0.0,
                error_message="交易所未连接"
            )
        
        start_time = time.time()
        
        # 模拟订单执行
        if order.order_type == OrderType.MARKET:
            # 市价单立即执行
            await asyncio.sleep(0.05)  # 模拟执行延迟
            filled_qty = order.quantity
            fill_price = self._get_mock_price(order.symbol) * (1 + 0.001)  # 模拟滑点
        elif order.order_type == OrderType.LIMIT:
            # 限价单检查价格
            current_price = self._get_mock_price(order.symbol)
            if order.price and current_price <= order.price:
                await asyncio.sleep(0.1)
                filled_qty = order.quantity
                fill_price = order.price
            else:
                # 价格不匹配，部分成交或未成交
                await asyncio.sleep(0.02)
                filled_qty = order.quantity * 0.7  # 模拟部分成交
                fill_price = current_price
        else:
            # 其他订单类型
            await asyncio.sleep(0.08)
            filled_qty = order.quantity
            fill_price = self._get_mock_price(order.symbol)
        
        execution_time = time.time() - start_time
        commission = filled_qty * fill_price * self.exchange_info.commission_rate
        slippage = abs(fill_price - self._get_mock_price(order.symbol)) * filled_qty
        
        # 更新订单状态
        order.status = OrderStatus.FILLED if filled_qty >= order.quantity else OrderStatus.PARTIAL
        order.filled_quantity = filled_qty
        order.avg_fill_price = fill_price
        order.commission = commission
        order.slippage = slippage
        
        self.orders[order.order_id] = order
        
        return ExecutionResult(
            order_id=order.order_id,
            success=True,
            filled_quantity=filled_qty,
            avg_price=fill_price,
            commission=commission,
            slippage=slippage,
            execution_time=execution_time
        )
    
    async def cancel_order(self, order_id: str) -> bool:
        """取消模拟订单"""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False
    
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """获取模拟订单状态"""
        if order_id in self.orders:
            return self.orders[order_id].status
        return OrderStatus.FAILED
    
    async def get_market_data(self, symbol: str) -> Dict[str, Any]:
        """获取模拟市场数据"""
        base_price = self._get_mock_price(symbol)
        return {
            "symbol": symbol,
            "price": base_price,
            "bid": base_price * 0.999,
            "ask": base_price * 1.001,
            "volume": 1000000,
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_mock_price(self, symbol: str) -> float:
        """生成模拟价格"""
        hash_val = int(hashlib.md5(symbol.encode()).hexdigest()[:8], 16)
        base_price = 100 + (hash_val % 1000) / 10.0
        return base_price

class OrderOptimizer:
    """订单优化器"""
    
    def __init__(self):
        self.optimization_history = []
    
    def optimize_order(self, signal: TradingSignal, market_data: Dict[str, Any]) -> Order:
        """优化订单参数"""
        symbol = signal.symbol
        current_price = market_data.get("price", 0.0)
        bid = market_data.get("bid", current_price * 0.999)
        ask = market_data.get("ask", current_price * 1.001)
        
        # 确定最优订单类型
        order_type = self._determine_optimal_order_type(signal, market_data)
        
        # 确定最优价格
        price = self._determine_optimal_price(signal, market_data, order_type)
        
        # 确定最优执行策略
        strategy = self._determine_execution_strategy(signal, market_data)
        
        # 生成订单ID
        order_id = f"ORD_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        order = Order(
            order_id=order_id,
            signal_id=signal.signal_id,
            symbol=symbol,
            side=signal.side,
            order_type=order_type,
            quantity=signal.quantity,
            price=price,
            status=OrderStatus.PENDING,
            timestamp=datetime.now(),
            exchange="",  # 将在后续分配
            strategy=strategy,
            metadata=signal.metadata
        )
        
        return order
    
    def _determine_optimal_order_type(self, signal: TradingSignal, market_data: Dict[str, Any]) -> OrderType:
        """确定最优订单类型"""
        confidence = signal.confidence
        quantity = signal.quantity
        
        # 高信心度大单使用限价单降低成本
        if confidence > 0.8 and quantity > 10000:
            return OrderType.LIMIT
        
        # 低信心度小单使用市价单快速执行
        if confidence < 0.6 or quantity < 1000:
            return OrderType.MARKET
        
        # 默认使用限价单
        return OrderType.LIMIT
    
    def _determine_optimal_price(self, signal: TradingSignal, market_data: Dict[str, Any], order_type: OrderType) -> Optional[float]:
        """确定最优价格"""
        if order_type == OrderType.MARKET:
            return None  # 市价单不需要指定价格
        
        current_price = market_data.get("price", 0.0)
        bid = market_data.get("bid", current_price * 0.999)
        ask = market_data.get("ask", current_price * 1.001)
        
        if signal.side == OrderSide.BUY:
            # 买单在当前价格下方一点
            return bid * 0.9995
        else:
            # 卖单在当前价格上方一点
            return ask * 1.0005
    
    def _determine_execution_strategy(self, signal: TradingSignal, market_data: Dict[str, Any]) -> ExecutionStrategy:
        """确定执行策略"""
        quantity = signal.quantity
        confidence = signal.confidence
        
        # 大单使用TWAP分散执行
        if quantity > 50000:
            return ExecutionStrategy.TWAP
        
        # 高信心度使用立即执行
        if confidence > 0.9:
            return ExecutionStrategy.IMMEDIATE
        
        # 中等规模使用VWAP
        if quantity > 10000:
            return ExecutionStrategy.VWAP
        
        # 默认立即执行
        return ExecutionStrategy.IMMEDIATE

class CostOptimizer:
    """交易成本优化器"""
    
    def __init__(self):
        self.cost_history = []
        self.exchange_performance = {}
    
    def optimize_execution_cost(self, orders: List[Order], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """优化执行成本"""
        if not orders:
            return {"optimization": "no_orders", "cost_savings": 0.0}
        
        # 计算当前成本
        current_cost = self._calculate_total_cost(orders, market_data)
        
        # 优化策略
        optimized_orders = self._apply_cost_optimization(orders, market_data)
        optimized_cost = self._calculate_total_cost(optimized_orders, market_data)
        
        cost_savings = current_cost - optimized_cost
        savings_percentage = (cost_savings / current_cost * 100) if current_cost > 0 else 0
        
        optimization_result = {
            "original_cost": current_cost,
            "optimized_cost": optimized_cost,
            "cost_savings": cost_savings,
            "savings_percentage": savings_percentage,
            "optimization_applied": len(optimized_orders) != len(orders),
            "optimized_orders": [asdict(order) for order in optimized_orders]
        }
        
        # 记录优化历史
        self.cost_history.append({
            "timestamp": datetime.now(),
            "orders_count": len(orders),
            "original_cost": current_cost,
            "optimized_cost": optimized_cost,
            "savings": cost_savings
        })
        
        return optimization_result
    
    def _calculate_total_cost(self, orders: List[Order], market_data: Dict[str, Any]) -> float:
        """计算总交易成本"""
        total_cost = 0.0
        
        for order in orders:
            # 手续费
            if order.price:
                commission = order.quantity * order.price * 0.001  # 0.1%手续费
            else:
                current_price = market_data.get("price", 0.0)
                commission = order.quantity * current_price * 0.001
            
            # 滑点成本
            bid = market_data.get("bid", market_data.get("price", 0.0) * 0.999)
            ask = market_data.get("ask", market_data.get("price", 0.0) * 1.001)
            
            if order.side == OrderSide.BUY:
                expected_price = ask
            else:
                expected_price = bid
            
            if order.price:
                slippage = abs(order.price - expected_price) * order.quantity
            else:
                slippage = 0.001 * order.quantity  # 市价单默认滑点
            
            total_cost += commission + slippage
        
        return total_cost
    
    def _apply_cost_optimization(self, orders: List[Order], market_data: Dict[str, Any]) -> List[Order]:
        """应用成本优化策略"""
        optimized_orders = []
        
        for order in orders:
            # 批量优化：将同方向同币种的订单合并
            merged_order = self._try_merge_orders(order, optimized_orders, market_data)
            if merged_order:
                optimized_orders.append(merged_order)
            else:
                optimized_orders.append(order)
        
        return optimized_orders
    
    def _try_merge_orders(self, new_order: Order, existing_orders: List[Order], market_data: Dict[str, Any]) -> Optional[Order]:
        """尝试合并订单"""
        for existing_order in existing_orders:
            # 检查是否可以合并（相同方向、相同币种、时间相近）
            if (existing_order.side == new_order.side and 
                existing_order.symbol == new_order.symbol and
                abs((new_order.timestamp - existing_order.timestamp).total_seconds()) < 300):  # 5分钟内
                
                # 合并订单
                merged_quantity = existing_order.quantity + new_order.quantity
                merged_price = (existing_order.price + new_order.price) / 2 if existing_order.price and new_order.price else None
                
                merged_order = Order(
                    order_id=f"MERGED_{int(time.time() * 1000)}",
                    signal_id=existing_order.signal_id,
                    symbol=existing_order.symbol,
                    side=existing_order.side,
                    order_type=existing_order.order_type,
                    quantity=merged_quantity,
                    price=merged_price,
                    status=OrderStatus.PENDING,
                    timestamp=datetime.now(),
                    exchange=existing_order.exchange,
                    strategy=existing_order.strategy,
                    metadata={**existing_order.metadata, "merged_from": [existing_order.order_id, new_order.order_id]}
                )
                
                # 移除原订单，添加合并订单
                existing_orders.remove(existing_order)
                return merged_order
        
        return None

class ExecutionMonitor:
    """执行监控器"""
    
    def __init__(self):
        self.active_orders = {}
        self.execution_history = []
        self.performance_metrics = {}
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """启动监控"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("交易执行监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("交易执行监控已停止")
    
    def add_order(self, order: Order):
        """添加订单到监控"""
        self.active_orders[order.order_id] = {
            "order": order,
            "start_time": time.time(),
            "last_update": time.time(),
            "status_updates": []
        }
    
    def update_order_status(self, order_id: str, status: OrderStatus, details: Dict[str, Any] = None):
        """更新订单状态"""
        if order_id in self.active_orders:
            self.active_orders[order_id]["order"].status = status
            self.active_orders[order_id]["last_update"] = time.time()
            
            update_record = {
                "timestamp": datetime.now(),
                "status": status,
                "details": details or {}
            }
            self.active_orders[order_id]["status_updates"].append(update_record)
            
            # 如果订单完成，移到历史记录
            if status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED, OrderStatus.FAILED]:
                self._move_to_history(order_id)
    
    def get_execution_metrics(self) -> Dict[str, Any]:
        """获取执行指标"""
        if not self.execution_history:
            return {"message": "暂无执行历史"}
        
        total_orders = len(self.execution_history)
        filled_orders = sum(1 for order in self.execution_history if order["status"] == OrderStatus.FILLED)
        success_rate = filled_orders / total_orders if total_orders > 0 else 0
        
        # 计算平均执行时间
        execution_times = [order["execution_time"] for order in self.execution_history if "execution_time" in order]
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0
        
        # 计算平均滑点
        slippage_values = [order["slippage"] for order in self.execution_history if "slippage" in order]
        avg_slippage = sum(slippage_values) / len(slippage_values) if slippage_values else 0
        
        return {
            "total_orders": total_orders,
            "filled_orders": filled_orders,
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "avg_slippage": avg_slippage,
            "active_orders": len(self.active_orders),
            "last_update": datetime.now().isoformat()
        }
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 检查超时订单
                current_time = time.time()
                for order_id, order_info in list(self.active_orders.items()):
                    elapsed_time = current_time - order_info["start_time"]
                    
                    # 超时检查（30秒）
                    if elapsed_time > 30:
                        logger.warning(f"订单 {order_id} 执行超时: {elapsed_time:.2f}秒")
                        self.update_order_status(order_id, OrderStatus.FAILED, {"reason": "timeout"})
                
                # 更新性能指标
                self.performance_metrics = self.get_execution_metrics()
                
                time.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(5)
    
    def _move_to_history(self, order_id: str):
        """将订单移到历史记录"""
        if order_id in self.active_orders:
            order_info = self.active_orders.pop(order_id)
            history_record = {
                "order_id": order_id,
                "symbol": order_info["order"].symbol,
                "side": order_info["order"].side.value,
                "quantity": order_info["order"].quantity,
                "status": order_info["order"].status,
                "execution_time": time.time() - order_info["start_time"],
                "slippage": order_info["order"].slippage,
                "commission": order_info["order"].commission,
                "timestamp": order_info["order"].timestamp
            }
            self.execution_history.append(history_record)

class ExceptionHandler:
    """异常处理器"""
    
    def __init__(self):
        self.retry_config = {
            "max_retries": 3,
            "retry_delay": 1.0,
            "backoff_factor": 2.0
        }
        self.exception_history = []
    
    async def handle_execution_exception(self, order: Order, exception: Exception) -> Dict[str, Any]:
        """处理执行异常"""
        exception_info = {
            "order_id": order.order_id,
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "timestamp": datetime.now(),
            "retry_count": 0,
            "recovery_action": None
        }
        
        # 根据异常类型决定处理策略
        if isinstance(exception, ConnectionError):
            # 连接异常：重试连接
            recovery_action = await self._handle_connection_error(order, exception_info)
        elif isinstance(exception, ValueError):
            # 参数错误：修正参数后重试
            recovery_action = await self._handle_value_error(order, exception_info)
        elif isinstance(exception, TimeoutError):
            # 超时异常：调整超时时间后重试
            recovery_action = await self._handle_timeout_error(order, exception_info)
        else:
            # 其他异常：记录并跳过
            recovery_action = await self._handle_unknown_error(order, exception_info)
        
        exception_info["recovery_action"] = recovery_action
        self.exception_history.append(exception_info)
        
        return exception_info
    
    async def _handle_connection_error(self, order: Order, exception_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理连接错误"""
        retry_count = 0
        max_retries = self.retry_config["max_retries"]
        
        while retry_count < max_retries:
            try:
                retry_count += 1
                exception_info["retry_count"] = retry_count
                
                # 等待后重试
                await asyncio.sleep(self.retry_config["retry_delay"] * (self.retry_config["backoff_factor"] ** (retry_count - 1)))
                
                # 尝试重新连接和执行
                logger.info(f"重试执行订单 {order.order_id}，第 {retry_count} 次")
                
                # 这里应该重新调用执行逻辑
                # return {"action": "retry", "success": True}
                
                return {"action": "retry", "success": True, "retry_count": retry_count}
                
            except Exception as e:
                logger.warning(f"重试失败: {e}")
                continue
        
        # 所有重试都失败，标记为失败
        return {"action": "fail", "success": False, "reason": "connection_failed_after_retries"}
    
    async def _handle_value_error(self, order: Order, exception_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理参数值错误"""
        # 修正订单参数
        corrected_order = self._correct_order_parameters(order)
        
        if corrected_order != order:
            return {"action": "retry_with_corrected_params", "success": True, "corrected_order": asdict(corrected_order)}
        else:
            return {"action": "fail", "success": False, "reason": "unable_to_correct_parameters"}
    
    async def _handle_timeout_error(self, order: Order, exception_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理超时错误"""
        # 延长超时时间或改为更快的执行方式
        if order.order_type == OrderType.LIMIT:
            # 限价单改为市价单
            order.order_type = OrderType.MARKET
            order.price = None
            return {"action": "retry_with_market_order", "success": True}
        else:
            # 其他情况增加超时时间
            return {"action": "retry_with_extended_timeout", "success": True}
    
    async def _handle_unknown_error(self, order: Order, exception_info: Dict[str, Any]) -> Dict[str, Any]:
        """处理未知错误"""
        # 记录错误并跳过
        logger.error(f"未知错误处理订单 {order.order_id}: {exception_info['exception_message']}")
        return {"action": "skip", "success": False, "reason": "unknown_error"}
    
    def _correct_order_parameters(self, order: Order) -> Order:
        """修正订单参数"""
        corrected_order = Order(
            order_id=order.order_id,
            signal_id=order.signal_id,
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=max(order.quantity, 0.01),  # 确保数量大于0
            price=order.price if order.price and order.price > 0 else None,
            status=order.status,
            timestamp=order.timestamp,
            exchange=order.exchange,
            strategy=order.strategy,
            metadata=order.metadata
        )
        return corrected_order

class PerformanceEvaluator:
    """执行效果评估器"""
    
    def __init__(self):
        self.evaluation_history = []
        self.benchmark_data = {}
    
    def evaluate_execution_performance(self, orders: List[Order], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估执行效果"""
        if not orders:
            return {"evaluation": "no_orders", "score": 0.0}
        
        # 计算各项指标
        execution_speed_score = self._evaluate_execution_speed(orders)
        cost_efficiency_score = self._evaluate_cost_efficiency(orders, market_data)
        success_rate_score = self._evaluate_success_rate(orders)
        slippage_control_score = self._evaluate_slippage_control(orders)
        
        # 综合评分
        overall_score = (
            execution_speed_score * 0.3 +
            cost_efficiency_score * 0.3 +
            success_rate_score * 0.25 +
            slippage_control_score * 0.15
        )
        
        evaluation_result = {
            "timestamp": datetime.now(),
            "orders_count": len(orders),
            "overall_score": overall_score,
            "execution_speed_score": execution_speed_score,
            "cost_efficiency_score": cost_efficiency_score,
            "success_rate_score": success_rate_score,
            "slippage_control_score": slippage_control_score,
            "detailed_metrics": self._calculate_detailed_metrics(orders, market_data),
            "recommendations": self._generate_recommendations(orders, market_data)
        }
        
        self.evaluation_history.append(evaluation_result)
        return evaluation_result
    
    def _evaluate_execution_speed(self, orders: List[Order]) -> float:
        """评估执行速度"""
        execution_times = []
        
        for order in orders:
            if order.status == OrderStatus.FILLED:
                # 计算执行时间（简化处理）
                execution_time = 1.0  # 假设平均1秒执行时间
                execution_times.append(execution_time)
        
        if not execution_times:
            return 0.0
        
        avg_time = sum(execution_times) / len(execution_times)
        
        # 速度评分：时间越短分数越高
        if avg_time < 0.5:
            return 100.0
        elif avg_time < 1.0:
            return 80.0
        elif avg_time < 2.0:
            return 60.0
        elif avg_time < 5.0:
            return 40.0
        else:
            return 20.0
    
    def _evaluate_cost_efficiency(self, orders: List[Order], market_data: Dict[str, Any]) -> float:
        """评估成本效率"""
        total_commission = sum(order.commission for order in orders)
        total_value = sum(order.quantity * (order.price or market_data.get("price", 0)) for order in orders)
        
        if total_value == 0:
            return 0.0
        
        cost_ratio = total_commission / total_value
        
        # 成本效率评分：成本比例越低分数越高
        if cost_ratio < 0.0005:  # 0.05%
            return 100.0
        elif cost_ratio < 0.001:  # 0.1%
            return 80.0
        elif cost_ratio < 0.002:  # 0.2%
            return 60.0
        elif cost_ratio < 0.005:  # 0.5%
            return 40.0
        else:
            return 20.0
    
    def _evaluate_success_rate(self, orders: List[Order]) -> float:
        """评估成功率"""
        if not orders:
            return 0.0
        
        successful_orders = sum(1 for order in orders if order.status == OrderStatus.FILLED)
        success_rate = successful_orders / len(orders)
        
        return success_rate * 100.0
    
    def _evaluate_slippage_control(self, orders: List[Order]) -> float:
        """评估滑点控制"""
        slippage_values = [order.slippage for order in orders if order.slippage > 0]
        
        if not slippage_values:
            return 100.0
        
        avg_slippage = sum(slippage_values) / len(slippage_values)
        
        # 滑点控制评分：滑点越小分数越高
        if avg_slippage < 0.001:
            return 100.0
        elif avg_slippage < 0.005:
            return 80.0
        elif avg_slippage < 0.01:
            return 60.0
        elif avg_slippage < 0.02:
            return 40.0
        else:
            return 20.0
    
    def _calculate_detailed_metrics(self, orders: List[Order], market_data: Dict[str, Any]) -> Dict[str, Any]:
        """计算详细指标"""
        total_orders = len(orders)
        filled_orders = sum(1 for order in orders if order.status == OrderStatus.FILLED)
        partial_orders = sum(1 for order in orders if order.status == OrderStatus.PARTIAL)
        failed_orders = sum(1 for order in orders if order.status in [OrderStatus.FAILED, OrderStatus.REJECTED])
        
        total_volume = sum(order.quantity for order in orders)
        total_commission = sum(order.commission for order in orders)
        total_slippage = sum(order.slippage for order in orders)
        
        return {
            "total_orders": total_orders,
            "filled_orders": filled_orders,
            "partial_orders": partial_orders,
            "failed_orders": failed_orders,
            "total_volume": total_volume,
            "total_commission": total_commission,
            "total_slippage": total_slippage,
            "avg_order_size": total_volume / total_orders if total_orders > 0 else 0,
            "fill_rate": filled_orders / total_orders if total_orders > 0 else 0
        }
    
    def _generate_recommendations(self, orders: List[Order], market_data: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 基于成功率建议
        success_rate = sum(1 for order in orders if order.status == OrderStatus.FILLED) / len(orders) if orders else 0
        
        if success_rate < 0.8:
            recommendations.append("建议检查信号质量，提高交易信号准确率")
        
        if success_rate < 0.6:
            recommendations.append("建议优化订单类型选择，使用更适合当前市场条件的订单类型")
        
        # 基于成本建议
        avg_commission = sum(order.commission for order in orders) / len(orders) if orders else 0
        if avg_commission > 10:
            recommendations.append("建议优化手续费成本，考虑使用手续费更低的交易所")
        
        # 基于滑点建议
        avg_slippage = sum(order.slippage for order in orders) / len(orders) if orders else 0
        if avg_slippage > 5:
            recommendations.append("建议优化执行策略，减少市场冲击和滑点成本")
        
        # 基于执行时间建议
        if len(orders) > 0:
            recommendations.append("建议实施更积极的执行策略以提高执行速度")
        
        if not recommendations:
            recommendations.append("当前执行效果良好，建议保持现有策略")
        
        return recommendations

class TradingExecutor:
    """交易执行器主类"""
    
    def __init__(self):
        self.exchanges = {}  # 交易所连接器
        self.order_optimizer = OrderOptimizer()
        self.cost_optimizer = CostOptimizer()
        self.execution_monitor = ExecutionMonitor()
        self.exception_handler = ExceptionHandler()
        self.performance_evaluator = PerformanceEvaluator()
        
        # 数据库
        self.db_path = "trading_executor.db"
        self._init_database()
        
        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # 状态
        self.is_running = False
        self.evaluation_history = []  # 评估历史记录
        
        logger.info("交易执行器初始化完成")
    
    def _init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 创建信号表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                signal_id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL,
                timestamp TEXT NOT NULL,
                confidence REAL,
                strategy_id TEXT,
                metadata TEXT,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # 创建订单表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orders (
                order_id TEXT PRIMARY KEY,
                signal_id TEXT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                order_type TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL,
                status TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                exchange TEXT,
                strategy TEXT,
                filled_quantity REAL DEFAULT 0,
                avg_fill_price REAL DEFAULT 0,
                commission REAL DEFAULT 0,
                slippage REAL DEFAULT 0,
                metadata TEXT,
                FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
            )
        ''')
        
        # 创建执行记录表
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS executions (
                execution_id TEXT PRIMARY KEY,
                order_id TEXT,
                success BOOLEAN,
                filled_quantity REAL,
                avg_price REAL,
                commission REAL,
                slippage REAL,
                execution_time REAL,
                error_message TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (order_id) REFERENCES orders (order_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_exchange(self, exchange_info: ExchangeInfo):
        """添加交易所"""
        connector = MockExchangeConnector(exchange_info)
        self.exchanges[exchange_info.name] = connector
        logger.info(f"已添加交易所: {exchange_info.name}")
    
    async def start(self):
        """启动交易执行器"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # 连接所有交易所
        for exchange_name, connector in self.exchanges.items():
            try:
                await connector.connect()
            except Exception as e:
                logger.error(f"连接交易所 {exchange_name} 失败: {e}")
        
        # 启动监控
        self.execution_monitor.start_monitoring()
        
        # 启动信号处理循环
        asyncio.create_task(self._signal_processing_loop())
        
        logger.info("交易执行器已启动")
    
    async def stop(self):
        """停止交易执行器"""
        self.is_running = False
        self.execution_monitor.stop_monitoring()
        self.executor.shutdown(wait=True)
        logger.info("交易执行器已停止")
    
    async def receive_signal(self, signal: TradingSignal) -> bool:
        """接收交易信号"""
        try:
            # 验证信号
            if not self._validate_signal(signal):
                logger.warning(f"信号验证失败: {signal.signal_id}")
                return False
            
            # 保存到数据库
            self._save_signal(signal)
            
            # 添加到处理队列
            await self._add_to_processing_queue(signal)
            
            logger.info(f"已接收交易信号: {signal.signal_id}")
            return True
            
        except Exception as e:
            logger.error(f"接收信号失败: {e}")
            return False
    
    def _validate_signal(self, signal: TradingSignal) -> bool:
        """验证交易信号"""
        # 基本字段检查
        if not all([signal.signal_id, signal.symbol, signal.side, signal.quantity]):
            return False
        
        # 数量检查
        if signal.quantity <= 0:
            return False
        
        # 信心度检查
        if not 0 <= signal.confidence <= 1:
            return False
        
        # 价格检查（如果有）
        if signal.price and signal.price <= 0:
            return False
        
        return True
    
    def _save_signal(self, signal: TradingSignal):
        """保存信号到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO signals 
            (signal_id, symbol, side, quantity, price, timestamp, confidence, strategy_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            signal.signal_id,
            signal.symbol,
            signal.side.value,
            signal.quantity,
            signal.price,
            signal.timestamp.isoformat(),
            signal.confidence,
            signal.strategy_id,
            json.dumps(signal.metadata)
        ))
        
        conn.commit()
        conn.close()
    
    async def _add_to_processing_queue(self, signal: TradingSignal):
        """添加到处理队列"""
        # 这里可以实现更复杂的队列逻辑
        # 目前直接处理
        await self._process_signal(signal)
    
    async def _process_signal(self, signal: TradingSignal):
        """处理交易信号"""
        try:
            # 获取市场数据
            market_data = await self._get_market_data(signal.symbol)
            
            # 优化订单
            order = self.order_optimizer.optimize_order(signal, market_data)
            
            # 选择最优交易所
            best_exchange = self._select_best_exchange(order, market_data)
            if not best_exchange:
                logger.error(f"没有可用的交易所执行订单: {order.order_id}")
                return
            
            order.exchange = best_exchange
            
            # 成本优化
            cost_optimization = self.cost_optimizer.optimize_execution_cost([order], market_data)
            
            # 添加到监控
            self.execution_monitor.add_order(order)
            
            # 执行订单
            execution_result = await self._execute_order(order, best_exchange)
            
            # 更新订单状态
            self.execution_monitor.update_order_status(
                order.order_id, 
                OrderStatus.FILLED if execution_result.success else OrderStatus.FAILED,
                asdict(execution_result)
            )
            
            # 保存执行结果
            self._save_execution_result(execution_result)
            
            # 评估执行效果
            evaluation = self.performance_evaluator.evaluate_execution_performance([order], market_data)
            
            logger.info(f"订单执行完成: {order.order_id}, 成功: {execution_result.success}")
            
        except Exception as e:
            logger.error(f"处理信号失败: {e}")
            await self.exception_handler.handle_execution_exception(order, e)
    
    async def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """获取市场数据"""
        # 从第一个可用交易所获取数据
        for exchange_name, connector in self.exchanges.items():
            if connector.connected:
                try:
                    return await connector.get_market_data(symbol)
                except Exception as e:
                    logger.warning(f"从交易所 {exchange_name} 获取市场数据失败: {e}")
                    continue
        
        # 如果所有交易所都失败，返回模拟数据
        return {
            "symbol": symbol,
            "price": 100.0,
            "bid": 99.9,
            "ask": 100.1,
            "volume": 1000000,
            "timestamp": datetime.now().isoformat()
        }
    
    def _select_best_exchange(self, order: Order, market_data: Dict[str, Any]) -> Optional[str]:
        """选择最优交易所"""
        best_exchange = None
        best_score = -1
        
        for exchange_name, connector in self.exchanges.items():
            if not connector.connected:
                continue
            
            # 检查是否支持订单类型
            if order.order_type not in connector.exchange_info.supported_order_types:
                continue
            
            # 计算评分
            score = self._calculate_exchange_score(connector.exchange_info, order, market_data)
            
            if score > best_score:
                best_score = score
                best_exchange = exchange_name
        
        return best_exchange
    
    def _calculate_exchange_score(self, exchange_info: ExchangeInfo, order: Order, market_data: Dict[str, Any]) -> float:
        """计算交易所评分"""
        score = 100.0
        
        # 手续费影响
        score -= exchange_info.commission_rate * 1000
        
        # 延迟影响
        score -= exchange_info.latency_ms * 0.1
        
        # 交易量限制检查
        if order.quantity < exchange_info.min_trade_amount:
            score -= 50
        elif order.quantity > exchange_info.max_trade_amount:
            score -= 100
        
        return max(score, 0.0)
    
    async def _execute_order(self, order: Order, exchange_name: str) -> ExecutionResult:
        """执行订单"""
        connector = self.exchanges[exchange_name]
        
        try:
            result = await connector.submit_order(order)
            return result
        except Exception as e:
            logger.error(f"执行订单失败: {e}")
            return ExecutionResult(
                order_id=order.order_id,
                success=False,
                filled_quantity=0.0,
                avg_price=0.0,
                commission=0.0,
                slippage=0.0,
                execution_time=0.0,
                error_message=str(e)
            )
    
    def _save_execution_result(self, execution_result: ExecutionResult):
        """保存执行结果"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO executions 
            (execution_id, order_id, success, filled_quantity, avg_price, commission, 
             slippage, execution_time, error_message, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            f"EXEC_{int(time.time() * 1000)}",
            execution_result.order_id,
            execution_result.success,
            execution_result.filled_quantity,
            execution_result.avg_price,
            execution_result.commission,
            execution_result.slippage,
            execution_result.execution_time,
            execution_result.error_message,
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
    
    async def _signal_processing_loop(self):
        """信号处理循环"""
        while self.is_running:
            try:
                # 这里可以实现更复杂的信号处理逻辑
                # 目前主要通过 receive_signal 方法直接处理
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"信号处理循环错误: {e}")
                await asyncio.sleep(5)
    
    def get_status(self) -> Dict[str, Any]:
        """获取执行器状态"""
        return {
            "is_running": self.is_running,
            "exchanges_count": len(self.exchanges),
            "active_exchanges": [name for name, conn in self.exchanges.items() if conn.connected],
            "performance_metrics": self.execution_monitor.get_execution_metrics(),
            "last_update": datetime.now().isoformat()
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        return {
            "evaluation_history": self.performance_evaluator.evaluation_history[-10:] if self.performance_evaluator.evaluation_history else [],
            "cost_optimization_history": self.cost_optimizer.cost_history[-10:],
            "exception_history": self.exception_handler.exception_history[-10:],
            "current_metrics": self.execution_monitor.get_execution_metrics()
        }

# 使用示例
async def main():
    """主函数示例"""
    # 创建交易执行器
    executor = TradingExecutor()
    
    # 添加交易所
    exchange_info = ExchangeInfo(
        name="mock_exchange",
        api_key="test_key",
        api_secret="test_secret",
        base_url="https://api.mock.com",
        commission_rate=0.001,
        min_trade_amount=10.0,
        max_trade_amount=1000000.0
    )
    executor.add_exchange(exchange_info)
    
    # 启动执行器
    await executor.start()
    
    # 模拟交易信号
    signal = TradingSignal(
        signal_id="SIG_001",
        symbol="BTCUSDT",
        side=OrderSide.BUY,
        quantity=100.0,
        price=50000.0,
        confidence=0.85,
        strategy_id="STRAT_001"
    )
    
    # 接收信号
    success = await executor.receive_signal(signal)
    print(f"信号处理结果: {success}")
    
    # 获取状态
    status = executor.get_status()
    print(f"执行器状态: {json.dumps(status, indent=2, default=str)}")
    
    # 等待处理完成
    await asyncio.sleep(5)
    
    # 获取性能报告
    report = executor.get_performance_report()
    print(f"性能报告: {json.dumps(report, indent=2, default=str)}")
    
    # 停止执行器
    await executor.stop()

if __name__ == "__main__":
    asyncio.run(main())