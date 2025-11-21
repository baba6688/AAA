#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
G6订单管理器
实现订单管理系统的核心功能，包括订单创建、验证、路由、状态跟踪等
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
import json
import threading
from collections import defaultdict, deque
import statistics
import numpy as np


class OrderStatus(Enum):
    """订单状态枚举"""
    PENDING = "pending"           # 待处理
    VALIDATING = "validating"     # 验证中
    ROUTING = "routing"          # 路由中
    QUEUED = "queued"            # 排队中
    EXECUTING = "executing"      # 执行中
    COMPLETED = "completed"      # 已完成
    CANCELLED = "cancelled"      # 已取消
    FAILED = "failed"           # 失败
    TIMEOUT = "timeout"         # 超时
    SUSPENDED = "suspended"     # 暂停


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"           # 市价单
    LIMIT = "limit"             # 限价单
    STOP = "stop"               # 止损单
    STOP_LIMIT = "stop_limit"   # 止损限价单
    TRAILING_STOP = "trailing_stop"  # 追踪止损单
    ICEBERG = "iceberg"         # 冰山单
    ALGO = "algo"               # 算法订单


class Priority(Enum):
    """优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


@dataclass
class OrderValidationResult:
    """订单验证结果"""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    estimated_cost: float = 0.0
    processing_time: float = 0.0


@dataclass
class OrderMetrics:
    """订单指标"""
    order_id: str
    creation_time: datetime
    validation_time: float
    routing_time: float
    execution_time: float
    total_time: float
    status: OrderStatus
    final_price: Optional[float] = None
    filled_quantity: int = 0
    total_quantity: int = 0
    fees: float = 0.0
    slippage: float = 0.0
    error_count: int = 0


@dataclass
class Order:
    """订单对象"""
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    symbol: str = ""
    order_type: OrderType = OrderType.MARKET
    side: str = "buy"  # buy/sell
    quantity: int = 0
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "GTC"  # GTC, IOC, FOK
    priority: Priority = Priority.NORMAL
    status: OrderStatus = OrderStatus.PENDING
    creation_time: datetime = field(default_factory=datetime.now)
    update_time: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)
    execution_history: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        """订单创建后的初始化"""
        self.status = OrderStatus.PENDING
        self.creation_time = datetime.now()
        self.update_time = datetime.now()


class OrderValidator:
    """订单验证器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.validation_rules = self._init_validation_rules()
        
    def _init_validation_rules(self) -> Dict[str, Callable]:
        """初始化验证规则"""
        return {
            'symbol': self._validate_symbol,
            'quantity': self._validate_quantity,
            'price': self._validate_price,
            'order_type': self._validate_order_type,
            'time_in_force': self._validate_time_in_force,
            'risk': self._validate_risk
        }
    
    async def validate_order(self, order: Order) -> OrderValidationResult:
        """验证订单"""
        start_time = time.time()
        errors = []
        warnings = []
        risk_score = 0.0
        
        try:
            # 基础验证
            for rule_name, rule_func in self.validation_rules.items():
                try:
                    result = rule_func(order)
                    if not result['valid']:
                        errors.extend(result.get('errors', []))
                    if result.get('warnings'):
                        warnings.extend(result['warnings'])
                    risk_score += result.get('risk_score', 0.0)
                except Exception as e:
                    errors.append(f"{rule_name}验证失败: {str(e)}")
                    
            # 业务规则验证
            business_errors = await self._validate_business_rules(order)
            errors.extend(business_errors)
            
            validation_time = time.time() - start_time
            
            return OrderValidationResult(
                is_valid=len(errors) == 0,
                errors=errors,
                warnings=warnings,
                risk_score=risk_score,
                estimated_cost=self._estimate_cost(order),
                processing_time=validation_time
            )
            
        except Exception as e:
            self.logger.error(f"订单验证异常: {str(e)}")
            return OrderValidationResult(
                is_valid=False,
                errors=[f"验证过程异常: {str(e)}"],
                processing_time=time.time() - start_time
            )
    
    def _validate_symbol(self, order: Order) -> Dict[str, Any]:
        """验证交易标的"""
        if not order.symbol or not isinstance(order.symbol, str):
            return {'valid': False, 'errors': ['无效的交易标的'], 'risk_score': 0.5}
        
        # 检查标的代码格式
        if len(order.symbol) < 1 or len(order.symbol) > 10:
            return {'valid': False, 'errors': ['交易标的代码格式错误'], 'risk_score': 0.3}
            
        return {'valid': True, 'risk_score': 0.0}
    
    def _validate_quantity(self, order: Order) -> Dict[str, Any]:
        """验证订单数量"""
        if order.quantity <= 0:
            return {'valid': False, 'errors': ['订单数量必须大于0'], 'risk_score': 0.8}
        
        if order.quantity > 1000000:
            return {'valid': False, 'errors': ['订单数量过大'], 'risk_score': 0.6}
            
        return {'valid': True, 'risk_score': 0.0}
    
    def _validate_price(self, order: Order) -> Dict[str, Any]:
        """验证订单价格"""
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT]:
            if order.price is None or order.price <= 0:
                return {'valid': False, 'errors': ['限价订单必须指定有效价格'], 'risk_score': 0.7}
                
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT]:
            if order.stop_price is None or order.stop_price <= 0:
                return {'valid': False, 'errors': ['止损订单必须指定有效止损价格'], 'risk_score': 0.7}
                
        return {'valid': True, 'risk_score': 0.0}
    
    def _validate_order_type(self, order: Order) -> Dict[str, Any]:
        """验证订单类型"""
        valid_types = [t.value for t in OrderType]
        if order.order_type.value not in valid_types:
            return {'valid': False, 'errors': ['无效的订单类型'], 'risk_score': 0.5}
            
        return {'valid': True, 'risk_score': 0.0}
    
    def _validate_time_in_force(self, order: Order) -> Dict[str, Any]:
        """验证订单有效期"""
        valid_tif = ['GTC', 'IOC', 'FOK']
        if order.time_in_force not in valid_tif:
            return {'valid': False, 'errors': ['无效的订单有效期'], 'risk_score': 0.3}
            
        return {'valid': True, 'risk_score': 0.0}
    
    def _validate_risk(self, order: Order) -> Dict[str, Any]:
        """风险验证"""
        risk_score = 0.0
        
        # 高频订单风险
        if order.metadata.get('frequency', 0) > 100:
            risk_score += 0.3
            
        # 大额订单风险
        if order.quantity > 100000:
            risk_score += 0.2
            
        # 特殊订单类型风险
        if order.order_type in [OrderType.ICEBERG, OrderType.ALGO]:
            risk_score += 0.1
            
        return {
            'valid': risk_score < 0.8,
            'warnings': ['高风险订单'] if risk_score > 0.5 else [],
            'risk_score': risk_score
        }
    
    async def _validate_business_rules(self, order: Order) -> List[str]:
        """业务规则验证"""
        errors = []
        
        # 检查交易时间
        current_time = datetime.now()
        if current_time.weekday() >= 5:  # 周末
            errors.append("非交易时间")
            
        # 检查交易限制
        if order.metadata.get('market_closed', False):
            errors.append("市场已关闭")
            
        return errors
    
    def _estimate_cost(self, order: Order) -> float:
        """估算订单成本"""
        base_cost = order.quantity * (order.price or 0) * 0.001  # 基础手续费
        return base_cost


class OrderRouter:
    """订单路由器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.exchanges = {}
        self.routing_rules = self._init_routing_rules()
        self.performance_history = defaultdict(deque)
        
    def _init_routing_rules(self) -> Dict[str, Any]:
        """初始化路由规则"""
        return {
            'load_balancing': True,
            'latency_threshold': 100,  # ms
            'cost_optimization': True,
            'liquidity_optimization': True
        }
    
    async def route_order(self, order: Order, validation_result: OrderValidationResult) -> str:
        """路由订单到最优执行场所"""
        try:
            # 分析可用的执行场所
            venues = await self._analyze_venues(order)
            
            # 计算每个场所的评分
            venue_scores = await self._calculate_venue_scores(venues, order)
            
            # 选择最优场所
            best_venue = max(venue_scores.items(), key=lambda x: x[1])
            
            self.logger.info(f"订单 {order.order_id} 路由到 {best_venue[0]}, 评分: {best_venue[1]:.2f}")
            
            return best_venue[0]
            
        except Exception as e:
            self.logger.error(f"订单路由异常: {str(e)}")
            return "default_venue"
    
    async def _analyze_venues(self, order: Order) -> List[Dict[str, Any]]:
        """分析可用的执行场所"""
        venues = []
        
        # 模拟多个交易场所
        mock_venues = [
            {'name': 'venue_a', 'latency': 50, 'cost': 0.001, 'liquidity': 0.8},
            {'name': 'venue_b', 'latency': 80, 'cost': 0.0008, 'liquidity': 0.6},
            {'name': 'venue_c', 'latency': 120, 'cost': 0.0012, 'liquidity': 0.9}
        ]
        
        for venue in mock_venues:
            # 检查场所是否支持该订单类型
            if self._venue_supports_order_type(venue, order):
                venues.append(venue)
                
        return venues
    
    def _venue_supports_order_type(self, venue: Dict, order: Order) -> bool:
        """检查场所是否支持订单类型"""
        # 模拟支持检查
        return True
    
    async def _calculate_venue_scores(self, venues: List[Dict], order: Order) -> Dict[str, float]:
        """计算场所评分"""
        scores = {}
        
        for venue in venues:
            score = 0.0
            
            # 延迟评分 (越低越好)
            latency_score = max(0, 1 - venue['latency'] / 200)
            score += latency_score * 0.3
            
            # 成本评分 (越低越好)
            cost_score = max(0, 1 - venue['cost'] * 1000)
            score += cost_score * 0.3
            
            # 流动性评分 (越高越好)
            score += venue['liquidity'] * 0.4
            
            scores[venue['name']] = score
            
        return scores


class OrderExecutionMonitor:
    """订单执行监控器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_orders: Dict[str, Order] = {}
        self.execution_callbacks: List[Callable] = []
        self.performance_metrics = defaultdict(list)
        self.monitoring_active = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """启动监控"""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            self.logger.info("订单执行监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("订单执行监控已停止")
    
    def register_order(self, order: Order):
        """注册订单进行监控"""
        self.active_orders[order.order_id] = order
        self.logger.debug(f"订单 {order.order_id} 已注册监控")
    
    def unregister_order(self, order_id: str):
        """取消订单监控"""
        if order_id in self.active_orders:
            del self.active_orders[order_id]
            self.logger.debug(f"订单 {order_id} 已取消监控")
    
    def add_execution_callback(self, callback: Callable):
        """添加执行回调"""
        self.execution_callbacks.append(callback)
    
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                current_time = datetime.now()
                
                # 检查超时订单
                self._check_timeout_orders(current_time)
                
                # 检查异常订单
                self._check_exception_orders()
                
                # 更新订单状态
                self._update_order_statuses()
                
                # 执行回调
                self._execute_callbacks()
                
                time.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                self.logger.error(f"监控循环异常: {str(e)}")
                time.sleep(5)
    
    def _check_timeout_orders(self, current_time: datetime):
        """检查超时订单"""
        timeout_threshold = timedelta(seconds=30)
        
        for order_id, order in list(self.active_orders.items()):
            if current_time - order.creation_time > timeout_threshold:
                if order.status not in [OrderStatus.COMPLETED, OrderStatus.CANCELLED, OrderStatus.FAILED]:
                    order.status = OrderStatus.TIMEOUT
                    order.update_time = current_time
                    self.logger.warning(f"订单 {order_id} 已超时")
    
    def _check_exception_orders(self):
        """检查异常订单"""
        for order_id, order in self.active_orders.items():
            # 检查连续错误
            if order.metadata.get('consecutive_errors', 0) > 3:
                order.status = OrderStatus.FAILED
                self.logger.error(f"订单 {order_id} 因连续错误而失败")
    
    def _update_order_statuses(self):
        """更新订单状态"""
        for order_id, order in self.active_orders.items():
            # 模拟状态更新
            if order.status == OrderStatus.QUEUED:
                if np.random.random() > 0.95:  # 5%概率开始执行
                    order.status = OrderStatus.EXECUTING
                    order.update_time = datetime.now()
            elif order.status == OrderStatus.EXECUTING:
                if np.random.random() > 0.9:  # 10%概率完成
                    order.status = OrderStatus.COMPLETED
                    order.update_time = datetime.now()
    
    def _execute_callbacks(self):
        """执行回调"""
        for callback in self.execution_callbacks:
            try:
                callback(self.active_orders)
            except Exception as e:
                self.logger.error(f"执行回调异常: {str(e)}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        if not self.performance_metrics:
            return {}
            
        return {
            'avg_execution_time': statistics.mean(self.performance_metrics['execution_time']),
            'success_rate': len([s for s in self.performance_metrics['status'] if s == 'completed']) / len(self.performance_metrics['status']),
            'avg_slippage': statistics.mean(self.performance_metrics['slippage']),
            'total_orders': len(self.performance_metrics['order_id'])
        }


class OrderExceptionHandler:
    """订单异常处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.recovery_strategies = self._init_recovery_strategies()
        self.exception_history = defaultdict(list)
        
    def _init_recovery_strategies(self) -> Dict[str, Callable]:
        """初始化恢复策略"""
        return {
            'network_error': self._handle_network_error,
            'timeout': self._handle_timeout,
            'insufficient_funds': self._handle_insufficient_funds,
            'market_closed': self._handle_market_closed,
            'invalid_order': self._handle_invalid_order
        }
    
    async def handle_exception(self, order: Order, exception: Exception) -> Dict[str, Any]:
        """处理订单异常"""
        exception_type = self._classify_exception(exception)
        
        # 记录异常
        self._record_exception(order.order_id, exception_type, str(exception))
        
        # 尝试恢复
        recovery_result = await self._attempt_recovery(order, exception_type, exception)
        
        return {
            'exception_type': exception_type,
            'recovered': recovery_result['success'],
            'action_taken': recovery_result['action'],
            'new_status': recovery_result.get('new_status'),
            'message': recovery_result.get('message', '')
        }
    
    def _classify_exception(self, exception: Exception) -> str:
        """分类异常类型"""
        exception_str = str(exception).lower()
        
        if 'network' in exception_str or 'connection' in exception_str:
            return 'network_error'
        elif 'timeout' in exception_str:
            return 'timeout'
        elif 'funds' in exception_str or 'balance' in exception_str:
            return 'insufficient_funds'
        elif 'market' in exception_str and 'closed' in exception_str:
            return 'market_closed'
        elif 'invalid' in exception_str or 'format' in exception_str:
            return 'invalid_order'
        else:
            return 'unknown_error'
    
    async def _attempt_recovery(self, order: Order, exception_type: str, exception: Exception) -> Dict[str, Any]:
        """尝试恢复订单"""
        strategy = self.recovery_strategies.get(exception_type)
        
        if strategy:
            try:
                return await strategy(order, exception)
            except Exception as e:
                self.logger.error(f"恢复策略执行失败: {str(e)}")
                return {'success': False, 'action': 'none', 'message': '恢复策略失败'}
        else:
            return {'success': False, 'action': 'none', 'message': '未找到对应恢复策略'}
    
    async def _handle_network_error(self, order: Order, exception: Exception) -> Dict[str, Any]:
        """处理网络错误"""
        # 重试机制
        retry_count = order.metadata.get('retry_count', 0)
        max_retries = 3
        
        if retry_count < max_retries:
            order.metadata['retry_count'] = retry_count + 1
            return {
                'success': True,
                'action': 'retry',
                'message': f'网络错误，重试 ({retry_count + 1}/{max_retries})'
            }
        else:
            return {
                'success': False,
                'action': 'fail',
                'new_status': OrderStatus.FAILED,
                'message': '网络错误重试次数超限'
            }
    
    async def _handle_timeout(self, order: Order, exception: Exception) -> Dict[str, Any]:
        """处理超时错误"""
        # 延长超时时间或重新路由
        current_timeout = order.metadata.get('timeout', 30)
        new_timeout = min(current_timeout * 1.5, 120)  # 最多延长到120秒
        
        order.metadata['timeout'] = new_timeout
        order.metadata['timeout_retries'] = order.metadata.get('timeout_retries', 0) + 1
        
        return {
            'success': True,
            'action': 'extend_timeout',
            'message': f'延长超时时间至 {new_timeout} 秒'
        }
    
    async def _handle_insufficient_funds(self, order: Order, exception: Exception) -> Dict[str, Any]:
        """处理资金不足"""
        return {
            'success': False,
            'action': 'cancel',
            'new_status': OrderStatus.FAILED,
            'message': '资金不足，订单取消'
        }
    
    async def _handle_market_closed(self, order: Order, exception: Exception) -> Dict[str, Any]:
        """处理市场关闭"""
        # 延迟到市场开盘
        order.metadata['delayed_execution'] = True
        order.metadata['original_exception'] = str(exception)
        
        return {
            'success': True,
            'action': 'delay',
            'message': '市场已关闭，延迟到开盘执行'
        }
    
    async def _handle_invalid_order(self, order: Order, exception: Exception) -> Dict[str, Any]:
        """处理无效订单"""
        return {
            'success': False,
            'action': 'cancel',
            'new_status': OrderStatus.FAILED,
            'message': '订单无效，取消执行'
        }
    
    def _record_exception(self, order_id: str, exception_type: str, message: str):
        """记录异常"""
        self.exception_history[order_id].append({
            'timestamp': datetime.now(),
            'type': exception_type,
            'message': message
        })


class OrderAnalytics:
    """订单分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.order_history: List[OrderMetrics] = []
        self.performance_data = defaultdict(list)
        
    def record_order_completion(self, metrics: OrderMetrics):
        """记录订单完成"""
        self.order_history.append(metrics)
        self.performance_data['execution_times'].append(metrics.execution_time)
        self.performance_data['success_rates'].append(1 if metrics.status == OrderStatus.COMPLETED else 0)
        
    def analyze_performance(self) -> Dict[str, Any]:
        """分析订单性能"""
        if not self.order_history:
            return {}
            
        recent_orders = [m for m in self.order_history if m.creation_time > datetime.now() - timedelta(hours=24)]
        
        if not recent_orders:
            return {}
            
        return {
            'total_orders_24h': len(recent_orders),
            'success_rate': len([o for o in recent_orders if o.status == OrderStatus.COMPLETED]) / len(recent_orders),
            'avg_execution_time': statistics.mean([o.execution_time for o in recent_orders]),
            'avg_slippage': statistics.mean([o.slippage for o in recent_orders]),
            'total_fees': sum([o.fees for o in recent_orders]),
            'failure_reasons': self._analyze_failure_reasons(recent_orders),
            'performance_trends': self._calculate_trends(recent_orders)
        }
    
    def _analyze_failure_reasons(self, orders: List[OrderMetrics]) -> Dict[str, int]:
        """分析失败原因"""
        failures = [o for o in orders if o.status in [OrderStatus.FAILED, OrderStatus.CANCELLED, OrderStatus.TIMEOUT]]
        reasons = defaultdict(int)
        
        for order in failures:
            # 模拟失败原因分析
            if order.error_count > 0:
                reasons['system_error'] += 1
            elif order.execution_time > 60:
                reasons['timeout'] += 1
            else:
                reasons['unknown'] += 1
                
        return dict(reasons)
    
    def _calculate_trends(self, orders: List[OrderMetrics]) -> Dict[str, float]:
        """计算性能趋势"""
        if len(orders) < 10:
            return {}
            
        # 按时间排序
        sorted_orders = sorted(orders, key=lambda x: x.creation_time)
        
        # 计算滑动平均
        window_size = min(10, len(sorted_orders) // 2)
        execution_times = [o.execution_time for o in sorted_orders]
        
        trends = {}
        if len(execution_times) >= window_size:
            recent_avg = statistics.mean(execution_times[-window_size:])
            earlier_avg = statistics.mean(execution_times[:window_size])
            trends['execution_time_trend'] = (recent_avg - earlier_avg) / earlier_avg if earlier_avg > 0 else 0
            
        return trends
    
    def optimize_routing(self) -> Dict[str, Any]:
        """优化路由策略"""
        # 分析不同场所的性能
        venue_performance = defaultdict(lambda: {'count': 0, 'total_time': 0, 'success_rate': 0})
        
        for metrics in self.order_history:
            venue = metrics.order_id.split('-')[0] if '-' in metrics.order_id else 'default'
            venue_performance[venue]['count'] += 1
            venue_performance[venue]['total_time'] += metrics.execution_time
            if metrics.status == OrderStatus.COMPLETED:
                venue_performance[venue]['success_rate'] += 1
        
        # 计算平均性能
        for venue in venue_performance:
            data = venue_performance[venue]
            if data['count'] > 0:
                data['avg_time'] = data['total_time'] / data['count']
                data['success_rate'] = data['success_rate'] / data['count']
        
        return dict(venue_performance)
    
    def generate_insights(self) -> List[str]:
        """生成优化建议"""
        insights = []
        
        performance = self.analyze_performance()
        
        if performance.get('success_rate', 1) < 0.95:
            insights.append("成功率较低，建议检查订单验证规则和执行环境")
        
        if performance.get('avg_execution_time', 0) > 30:
            insights.append("执行时间较长，建议优化路由策略和执行场所选择")
        
        if performance.get('avg_slippage', 0) > 0.001:
            insights.append("滑点较大，建议改进订单拆分和执行时机")
        
        failure_reasons = performance.get('failure_reasons', {})
        if failure_reasons.get('timeout', 0) > 5:
            insights.append("超时错误较多，建议增加超时时间和重试机制")
        
        if not insights:
            insights.append("系统运行良好，暂无优化建议")
            
        return insights


class OrderManager:
    """订单管理器主类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # 初始化各个组件
        self.validator = OrderValidator()
        self.router = OrderRouter()
        self.monitor = OrderExecutionMonitor()
        self.exception_handler = OrderExceptionHandler()
        self.analytics = OrderAnalytics()
        
        # 订单存储
        self.orders: Dict[str, Order] = {}
        self.order_queue = asyncio.Queue()
        
        # 运行状态
        self.is_running = False
        self.processing_tasks = []
        
        # 配置
        self.config = {
            'max_concurrent_orders': 100,
            'processing_timeout': 60,
            'enable_monitoring': True,
            'enable_analytics': True
        }
    
    async def start(self):
        """启动订单管理器"""
        if self.is_running:
            return
            
        self.is_running = True
        self.logger.info("订单管理器启动中...")
        
        # 启动监控
        if self.config['enable_monitoring']:
            self.monitor.start_monitoring()
        
        # 启动处理任务
        self.processing_tasks = [
            asyncio.create_task(self._process_order_queue()),
            asyncio.create_task(self._cleanup_completed_orders())
        ]
        
        self.logger.info("订单管理器启动完成")
    
    async def stop(self):
        """停止订单管理器"""
        if not self.is_running:
            return
            
        self.is_running = False
        
        # 停止监控
        self.monitor.stop_monitoring()
        
        # 取消处理任务
        for task in self.processing_tasks:
            task.cancel()
        
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        self.logger.info("订单管理器已停止")
    
    async def create_order(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        """创建订单"""
        try:
            # 创建订单对象
            order = Order(**order_data)
            
            # 验证订单
            validation_result = await self.validator.validate_order(order)
            
            if not validation_result.is_valid:
                return {
                    'success': False,
                    'order_id': order.order_id,
                    'errors': validation_result.errors,
                    'status': 'validation_failed'
                }
            
            # 路由订单
            venue = await self.router.route_order(order, validation_result)
            
            # 存储订单
            self.orders[order.order_id] = order
            order.metadata['venue'] = venue
            
            # 添加到处理队列
            await self.order_queue.put(order)
            
            # 注册监控
            if self.config['enable_monitoring']:
                self.monitor.register_order(order)
            
            self.logger.info(f"订单创建成功: {order.order_id}, 场所: {venue}")
            
            return {
                'success': True,
                'order_id': order.order_id,
                'status': order.status.value,
                'venue': venue,
                'estimated_cost': validation_result.estimated_cost,
                'warnings': validation_result.warnings
            }
            
        except Exception as e:
            self.logger.error(f"创建订单异常: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'status': 'creation_failed'
            }
    
    async def modify_order(self, order_id: str, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """修改订单"""
        if order_id not in self.orders:
            return {'success': False, 'error': '订单不存在'}
        
        order = self.orders[order_id]
        
        # 检查订单状态
        if order.status not in [OrderStatus.PENDING, OrderStatus.VALIDATING, OrderStatus.QUEUED]:
            return {'success': False, 'error': '订单状态不允许修改'}
        
        try:
            # 应用修改
            for key, value in modifications.items():
                if hasattr(order, key):
                    setattr(order, key, value)
            
            order.update_time = datetime.now()
            
            # 重新验证
            validation_result = await self.validator.validate_order(order)
            
            if not validation_result.is_valid:
                return {
                    'success': False,
                    'errors': validation_result.errors,
                    'status': 'validation_failed'
                }
            
            self.logger.info(f"订单修改成功: {order_id}")
            
            return {
                'success': True,
                'order_id': order_id,
                'status': order.status.value,
                'modifications_applied': list(modifications.keys())
            }
            
        except Exception as e:
            self.logger.error(f"修改订单异常: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def cancel_order(self, order_id: str, reason: str = "用户取消") -> Dict[str, Any]:
        """取消订单"""
        if order_id not in self.orders:
            return {'success': False, 'error': '订单不存在'}
        
        order = self.orders[order_id]
        
        # 检查订单状态
        if order.status in [OrderStatus.COMPLETED, OrderStatus.CANCELLED, OrderStatus.FAILED]:
            return {'success': False, 'error': '订单已完成或已取消'}
        
        try:
            order.status = OrderStatus.CANCELLED
            order.update_time = datetime.now()
            order.metadata['cancel_reason'] = reason
            
            # 取消监控
            self.monitor.unregister_order(order_id)
            
            self.logger.info(f"订单取消成功: {order_id}, 原因: {reason}")
            
            return {
                'success': True,
                'order_id': order_id,
                'status': order.status.value,
                'cancel_reason': reason
            }
            
        except Exception as e:
            self.logger.error(f"取消订单异常: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """获取订单状态"""
        if order_id not in self.orders:
            return {'success': False, 'error': '订单不存在'}
        
        order = self.orders[order_id]
        
        return {
            'success': True,
            'order_id': order_id,
            'status': order.status.value,
            'creation_time': order.creation_time.isoformat(),
            'update_time': order.update_time.isoformat(),
            'metadata': order.metadata,
            'execution_history': order.execution_history
        }
    
    async def get_orders_by_status(self, status: OrderStatus) -> List[Dict[str, Any]]:
        """按状态获取订单"""
        matching_orders = [
            {
                'order_id': order.order_id,
                'status': order.status.value,
                'symbol': order.symbol,
                'quantity': order.quantity,
                'creation_time': order.creation_time.isoformat()
            }
            for order in self.orders.values()
            if order.status == status
        ]
        
        return matching_orders
    
    async def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.config['enable_analytics']:
            return {'error': '分析功能未启用'}
        
        performance = self.analytics.analyze_performance()
        insights = self.analytics.generate_insights()
        routing_optimization = self.analytics.optimize_routing()
        
        return {
            'performance': performance,
            'insights': insights,
            'routing_optimization': routing_optimization,
            'total_orders': len(self.orders),
            'active_orders': len([o for o in self.orders.values() if o.status not in [OrderStatus.COMPLETED, OrderStatus.CANCELLED, OrderStatus.FAILED]])
        }
    
    async def _process_order_queue(self):
        """处理订单队列"""
        while self.is_running:
            try:
                # 获取订单（超时1秒）
                order = await asyncio.wait_for(self.order_queue.get(), timeout=1.0)
                
                # 处理订单
                await self._process_single_order(order)
                
                # 标记任务完成
                self.order_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"处理订单队列异常: {str(e)}")
    
    async def _process_single_order(self, order: Order):
        """处理单个订单"""
        try:
            # 更新状态
            order.status = OrderStatus.VALIDATING
            order.update_time = datetime.now()
            
            # 验证
            validation_result = await self.validator.validate_order(order)
            if not validation_result.is_valid:
                order.status = OrderStatus.FAILED
                return
            
            # 路由
            order.status = OrderStatus.ROUTING
            venue = await self.router.route_order(order, validation_result)
            
            # 排队
            order.status = OrderStatus.QUEUED
            order.metadata['venue'] = venue
            
            # 模拟执行
            await self._simulate_order_execution(order)
            
        except Exception as e:
            self.logger.error(f"处理订单异常: {order.order_id}, {str(e)}")
            order.status = OrderStatus.FAILED
            order.metadata['error'] = str(e)
    
    async def _simulate_order_execution(self, order: Order):
        """模拟订单执行"""
        try:
            order.status = OrderStatus.EXECUTING
            order.update_time = datetime.now()
            
            # 模拟执行时间
            execution_time = np.random.uniform(1, 10)
            await asyncio.sleep(execution_time)
            
            # 模拟执行结果
            if np.random.random() > 0.05:  # 95%成功率
                order.status = OrderStatus.COMPLETED
                
                # 记录执行历史
                order.execution_history.append({
                    'timestamp': datetime.now(),
                    'action': 'executed',
                    'price': order.price or np.random.uniform(100, 200),
                    'quantity': order.quantity
                })
                
                # 记录指标
                metrics = OrderMetrics(
                    order_id=order.order_id,
                    creation_time=order.creation_time,
                    validation_time=1.0,
                    routing_time=0.5,
                    execution_time=execution_time,
                    total_time=(datetime.now() - order.creation_time).total_seconds(),
                    status=order.status,
                    final_price=order.price or 150.0,
                    filled_quantity=order.quantity,
                    total_quantity=order.quantity,
                    fees=order.quantity * 0.001,
                    slippage=np.random.uniform(0, 0.001)
                )
                
                self.analytics.record_order_completion(metrics)
                
            else:
                order.status = OrderStatus.FAILED
                order.metadata['error'] = '执行失败'
                
        except Exception as e:
            self.logger.error(f"模拟执行异常: {str(e)}")
            order.status = OrderStatus.FAILED
        
        order.update_time = datetime.now()
    
    async def _cleanup_completed_orders(self):
        """清理完成的订单"""
        while self.is_running:
            try:
                current_time = datetime.now()
                cleanup_threshold = timedelta(hours=1)
                
                completed_orders = [
                    order_id for order_id, order in self.orders.items()
                    if order.status in [OrderStatus.COMPLETED, OrderStatus.CANCELLED, OrderStatus.FAILED]
                    and current_time - order.update_time > cleanup_threshold
                ]
                
                for order_id in completed_orders:
                    del self.orders[order_id]
                    self.monitor.unregister_order(order_id)
                
                if completed_orders:
                    self.logger.info(f"清理了 {len(completed_orders)} 个已完成订单")
                
                await asyncio.sleep(300)  # 每5分钟清理一次
                
            except Exception as e:
                self.logger.error(f"清理订单异常: {str(e)}")
                await asyncio.sleep(60)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        status_counts = defaultdict(int)
        for order in self.orders.values():
            status_counts[order.status.value] += 1
        
        return {
            'total_orders': len(self.orders),
            'status_distribution': dict(status_counts),
            'queue_size': self.order_queue.qsize(),
            'is_running': self.is_running,
            'config': self.config
        }


# 使用示例和测试代码
async def main():
    """主函数 - 演示订单管理器使用"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建订单管理器
    manager = OrderManager()
    
    try:
        # 启动管理器
        await manager.start()
        
        # 创建测试订单
        test_orders = [
            {
                'symbol': 'AAPL',
                'order_type': OrderType.LIMIT,
                'side': 'buy',
                'quantity': 100,
                'price': 150.0,
                'priority': Priority.HIGH
            },
            {
                'symbol': 'GOOGL',
                'order_type': OrderType.MARKET,
                'side': 'sell',
                'quantity': 50,
                'priority': Priority.NORMAL
            },
            {
                'symbol': 'TSLA',
                'order_type': OrderType.STOP,
                'side': 'sell',
                'quantity': 25,
                'stop_price': 200.0,
                'priority': Priority.URGENT
            }
        ]
        
        created_orders = []
        
        # 创建订单
        for order_data in test_orders:
            result = await manager.create_order(order_data)
            if result['success']:
                created_orders.append(result['order_id'])
                print(f"订单创建成功: {result['order_id']}")
            else:
                print(f"订单创建失败: {result}")
        
        # 等待订单处理
        await asyncio.sleep(5)
        
        # 检查订单状态
        for order_id in created_orders:
            status = await manager.get_order_status(order_id)
            print(f"订单状态: {order_id} - {status['status']}")
        
        # 获取性能报告
        report = await manager.get_performance_report()
        print(f"性能报告: {json.dumps(report, indent=2, ensure_ascii=False)}")
        
        # 获取统计信息
        stats = manager.get_statistics()
        print(f"统计信息: {json.dumps(stats, indent=2, ensure_ascii=False)}")
        
    except Exception as e:
        print(f"演示过程异常: {str(e)}")
    
    finally:
        # 停止管理器
        await manager.stop()


if __name__ == "__main__":
    asyncio.run(main())