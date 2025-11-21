"""
O9优化状态聚合器模块

该模块提供了完整的优化状态聚合和管理功能，包括：
- 优化状态监控（优化进度、优化效果、优化问题）
- 优化协调管理（优化策略、优化参数、优化结果）
- 优化生命周期管理（优化创建、优化执行、优化完成）
- 优化性能统计（优化次数、优化时间、优化效果）
- 优化健康检查（优化有效性、优化完整性、优化安全性）
- 统一优化接口和API
- 异步优化状态同步和分布式协调
- 优化告警和通知系统
- 完整的错误处理和日志记录
- 详细的文档字符串和使用示例

主要类：
- OptimizationStateAggregator: 主要聚合器类
- OptimizationState: 优化状态数据类
- OptimizationMonitor: 优化状态监控器
- OptimizationCoordinator: 优化协调器
- OptimizationLifecycleManager: 优化生命周期管理器
- OptimizationStatistics: 优化性能统计
- OptimizationHealthChecker: 优化健康检查器
- OptimizationAPI: 统一优化接口
- AsyncStateSynchronizer: 异步状态同步器
- AlertNotificationSystem: 告警通知系统

作者：AI系统
版本：1.0.0
"""

import asyncio
import logging
import time
import json
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union, Tuple, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
import sqlite3
import hashlib
import weakref
from abc import ABC, abstractmethod
import copy
import traceback
import signal
import os
from contextlib import asynccontextmanager, contextmanager
from functools import wraps
import pickle
import gzip
import base64


# ==================== 异常类定义 ====================

class OptimizationError(Exception):
    """优化系统基础异常类"""
    pass


class OptimizationStateError(OptimizationError):
    """优化状态相关异常"""
    pass


class OptimizationCoordinationError(OptimizationError):
    """优化协调相关异常"""
    pass


class OptimizationLifecycleError(OptimizationError):
    """优化生命周期相关异常"""
    pass


class OptimizationStatisticsError(OptimizationError):
    """优化统计相关异常"""
    pass


class OptimizationHealthError(OptimizationError):
    """优化健康检查相关异常"""
    pass


class OptimizationAPIError(OptimizationError):
    """优化API相关异常"""
    pass


class OptimizationSyncError(OptimizationError):
    """优化同步相关异常"""
    pass


class OptimizationAlertError(OptimizationError):
    """优化告警相关异常"""
    pass


# ==================== 枚举类定义 ====================

class OptimizationStatus(Enum):
    """优化状态枚举"""
    CREATED = "created"
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"
    SUSPENDED = "suspended"


class OptimizationType(Enum):
    """优化类型枚举"""
    PARAMETER_TUNING = "parameter_tuning"
    STRATEGY_OPTIMIZATION = "strategy_optimization"
    PORTFOLIO_OPTIMIZATION = "portfolio_optimization"
    RISK_OPTIMIZATION = "risk_optimization"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SYSTEM_OPTIMIZATION = "system_optimization"


class OptimizationPriority(Enum):
    """优化优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


# ==================== 数据类定义 ====================

@dataclass
class OptimizationState:
    """优化状态数据类"""
    optimization_id: str
    name: str
    description: str
    optimization_type: OptimizationType
    status: OptimizationStatus
    priority: OptimizationPriority
    created_at: datetime
    updated_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 进度信息
    progress: float = 0.0  # 0.0 - 1.0
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0
    
    # 效果信息
    current_performance: float = 0.0
    baseline_performance: float = 0.0
    target_performance: float = 0.0
    performance_improvement: float = 0.0
    
    # 参数信息
    parameters: Dict[str, Any] = field(default_factory=dict)
    optimized_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # 结果信息
    results: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    
    # 问题信息
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # 资源信息
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    execution_time: float = 0.0
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        # 处理datetime对象
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, Enum):
                data[key] = value.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationState':
        """从字典创建实例"""
        # 处理datetime对象
        datetime_fields = ['created_at', 'updated_at', 'started_at', 'completed_at']
        for field in datetime_fields:
            if field in data and data[field]:
                if isinstance(data[field], str):
                    data[field] = datetime.fromisoformat(data[field])
        
        # 处理Enum对象
        if 'optimization_type' in data:
            data['optimization_type'] = OptimizationType(data['optimization_type'])
        if 'status' in data:
            data['status'] = OptimizationStatus(data['status'])
        if 'priority' in data:
            data['priority'] = OptimizationPriority(data['priority'])
        
        return cls(**data)


@dataclass
class OptimizationMetrics:
    """优化指标数据类"""
    optimization_id: str
    timestamp: datetime
    
    # 性能指标
    execution_time: float
    cpu_usage: float
    memory_usage: float
    throughput: float
    
    # 优化效果指标
    convergence_rate: float
    improvement_rate: float
    stability_score: float
    
    # 质量指标
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    
    # 业务指标
    profit_improvement: float
    risk_reduction: float
    efficiency_gain: float


@dataclass
class OptimizationAlert:
    """优化告警数据类"""
    alert_id: str
    optimization_id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationHealthReport:
    """优化健康报告数据类"""
    optimization_id: str
    timestamp: datetime
    overall_status: HealthStatus
    
    # 各维度健康状态
    validity_status: HealthStatus
    completeness_status: HealthStatus
    security_status: HealthStatus
    performance_status: HealthStatus
    
    # 健康评分
    validity_score: float  # 0.0 - 1.0
    completeness_score: float  # 0.0 - 1.0
    security_score: float  # 0.0 - 1.0
    performance_score: float  # 0.0 - 1.0
    
    # 问题和建议
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# ==================== 抽象基类 ====================

class OptimizationComponent(ABC):
    """优化组件抽象基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self._running = False
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
    
    @abstractmethod
    async def initialize(self) -> None:
        """初始化组件"""
        pass
    
    @abstractmethod
    async def start(self) -> None:
        """启动组件"""
        pass
    
    @abstractmethod
    async def stop(self) -> None:
        """停止组件"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """清理资源"""
        pass
    
    def register_callback(self, event: str, callback: Callable) -> None:
        """注册事件回调"""
        self._callbacks[event].append(callback)
    
    async def _emit_event(self, event: str, data: Any = None) -> None:
        """触发事件"""
        for callback in self._callbacks[event]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                self.logger.error(f"回调执行失败: {e}")


# ==================== 优化状态监控器 ====================

class OptimizationMonitor(OptimizationComponent):
    """优化状态监控器"""
    
    def __init__(self, check_interval: float = 1.0):
        super().__init__("OptimizationMonitor")
        self.check_interval = check_interval
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._monitored_optimizations: Set[str] = set()
        self._monitoring_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._alerts: List[OptimizationAlert] = []
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """初始化监控器"""
        self.logger.info("初始化优化状态监控器")
        self._running = True
    
    async def start(self) -> None:
        """启动监控器"""
        self.logger.info("启动优化状态监控器")
        # 启动监控循环
        asyncio.create_task(self._monitoring_loop())
    
    async def stop(self) -> None:
        """停止监控器"""
        self.logger.info("停止优化状态监控器")
        self._running = False
        
        # 取消所有监控任务
        for task in self._monitoring_tasks.values():
            task.cancel()
        
        # 等待任务完成
        if self._monitoring_tasks:
            await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)
    
    async def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("清理优化状态监控器资源")
        self._monitoring_tasks.clear()
        self._monitored_optimizations.clear()
        self._monitoring_data.clear()
        self._alerts.clear()
    
    async def start_monitoring(self, optimization_id: str) -> None:
        """开始监控指定优化"""
        async with self._lock:
            if optimization_id not in self._monitored_optimizations:
                self._monitored_optimizations.add(optimization_id)
                task = asyncio.create_task(self._monitor_optimization(optimization_id))
                self._monitoring_tasks[optimization_id] = task
                self.logger.info(f"开始监控优化: {optimization_id}")
    
    async def stop_monitoring(self, optimization_id: str) -> None:
        """停止监控指定优化"""
        async with self._lock:
            if optimization_id in self._monitored_optimizations:
                self._monitored_optimizations.remove(optimization_id)
                if optimization_id in self._monitoring_tasks:
                    task = self._monitoring_tasks[optimization_id]
                    task.cancel()
                    del self._monitoring_tasks[optimization_id]
                self.logger.info(f"停止监控优化: {optimization_id}")
    
    async def get_optimization_status(self, optimization_id: str) -> Optional[Dict[str, Any]]:
        """获取优化状态"""
        if optimization_id in self._monitoring_data:
            data = list(self._monitoring_data[optimization_id])
            if data:
                return data[-1].to_dict() if hasattr(data[-1], 'to_dict') else data[-1]
        return None
    
    async def get_monitoring_data(self, optimization_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """获取监控数据"""
        if optimization_id in self._monitoring_data:
            data = list(self._monitoring_data[optimization_id])[-limit:]
            return [item.to_dict() if hasattr(item, 'to_dict') else item for item in data]
        return []
    
    async def get_alerts(self, optimization_id: Optional[str] = None, 
                        level: Optional[AlertLevel] = None) -> List[OptimizationAlert]:
        """获取告警信息"""
        alerts = self._alerts
        if optimization_id:
            alerts = [alert for alert in alerts if alert.optimization_id == optimization_id]
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    async def _monitoring_loop(self) -> None:
        """监控循环"""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                
                # 检查所有监控的优化
                monitored_ids = list(self._monitored_optimizations)
                for optimization_id in monitored_ids:
                    try:
                        await self._check_optimization_health(optimization_id)
                    except Exception as e:
                        self.logger.error(f"检查优化健康状态失败 {optimization_id}: {e}")
                        
            except Exception as e:
                self.logger.error(f"监控循环异常: {e}")
                await asyncio.sleep(5)  # 异常时等待5秒再继续
    
    async def _monitor_optimization(self, optimization_id: str) -> None:
        """监控单个优化"""
        while self._running and optimization_id in self._monitored_optimizations:
            try:
                # 这里应该从实际的数据源获取优化状态
                # 模拟数据
                monitoring_data = {
                    'optimization_id': optimization_id,
                    'timestamp': datetime.now(),
                    'status': 'running',
                    'progress': 0.5,
                    'cpu_usage': 25.0,
                    'memory_usage': 512.0,
                    'performance': 0.85
                }
                
                self._monitoring_data[optimization_id].append(monitoring_data)
                
                # 检查是否需要告警
                await self._check_alerts(optimization_id, monitoring_data)
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"监控优化失败 {optimization_id}: {e}")
                await asyncio.sleep(5)
    
    async def _check_optimization_health(self, optimization_id: str) -> None:
        """检查优化健康状态"""
        if optimization_id not in self._monitoring_data:
            return
        
        latest_data = None
        for data in reversed(self._monitoring_data[optimization_id]):
            if isinstance(data, dict) and 'timestamp' in data:
                latest_data = data
                break
        
        if not latest_data:
            return
        
        # 检查各种健康指标
        issues = []
        
        # 检查进度
        if latest_data.get('progress', 0) < 0.01:
            issues.append("优化进度异常：进度过低")
        
        # 检查资源使用
        if latest_data.get('cpu_usage', 0) > 90:
            issues.append("CPU使用率过高")
        
        if latest_data.get('memory_usage', 0) > 1024:
            issues.append("内存使用率过高")
        
        # 检查性能
        if latest_data.get('performance', 1) < 0.5:
            issues.append("优化性能异常：效果不佳")
        
        # 生成告警
        if issues:
            alert = OptimizationAlert(
                alert_id=str(uuid.uuid4()),
                optimization_id=optimization_id,
                level=AlertLevel.WARNING,
                title="优化健康检查告警",
                message="; ".join(issues),
                timestamp=datetime.now()
            )
            self._alerts.append(alert)
            await self._emit_event('alert', alert)
    
    async def _check_alerts(self, optimization_id: str, data: Dict[str, Any]) -> None:
        """检查告警条件"""
        # 这里可以实现具体的告警逻辑
        pass


# ==================== 优化协调器 ====================

class OptimizationCoordinator(OptimizationComponent):
    """优化协调器"""
    
    def __init__(self, max_concurrent_optimizations: int = 10):
        super().__init__("OptimizationCoordinator")
        self.max_concurrent_optimizations = max_concurrent_optimizations
        self._active_optimizations: Dict[str, OptimizationState] = {}
        self._optimization_queue: List[OptimizationState] = []
        self._optimization_strategies: Dict[str, Callable] = {}
        self._optimization_policies: Dict[str, Any] = {}
        self._coordination_lock = asyncio.Lock()
        self._resource_manager: Dict[str, Any] = {}
    
    async def initialize(self) -> None:
        """初始化协调器"""
        self.logger.info("初始化优化协调器")
        self._running = True
        
        # 初始化默认策略
        await self._initialize_default_strategies()
    
    async def start(self) -> None:
        """启动协调器"""
        self.logger.info("启动优化协调器")
        # 启动协调循环
        asyncio.create_task(self._coordination_loop())
    
    async def stop(self) -> None:
        """停止协调器"""
        self.logger.info("停止优化协调器")
        self._running = False
    
    async def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("清理优化协调器资源")
        self._active_optimizations.clear()
        self._optimization_queue.clear()
        self._optimization_strategies.clear()
        self._optimization_policies.clear()
        self._resource_manager.clear()
    
    async def submit_optimization(self, optimization: OptimizationState) -> str:
        """提交优化任务"""
        async with self._coordination_lock:
            # 检查并发限制
            if len(self._active_optimizations) >= self.max_concurrent_optimizations:
                self._optimization_queue.append(optimization)
                self.logger.info(f"优化任务加入队列: {optimization.optimization_id}")
            else:
                await self._start_optimization(optimization)
            
            return optimization.optimization_id
    
    async def cancel_optimization(self, optimization_id: str) -> bool:
        """取消优化任务"""
        async with self._coordination_lock:
            # 检查活跃任务
            if optimization_id in self._active_optimizations:
                optimization = self._active_optimizations[optimization_id]
                optimization.status = OptimizationStatus.CANCELLED
                del self._active_optimizations[optimization_id]
                self.logger.info(f"取消活跃优化任务: {optimization_id}")
                return True
            
            # 检查队列
            for i, opt in enumerate(self._optimization_queue):
                if opt.optimization_id == optimization_id:
                    del self._optimization_queue[i]
                    self.logger.info(f"取消队列中的优化任务: {optimization_id}")
                    return True
            
            return False
    
    async def pause_optimization(self, optimization_id: str) -> bool:
        """暂停优化任务"""
        async with self._coordination_lock:
            if optimization_id in self._active_optimizations:
                optimization = self._active_optimizations[optimization_id]
                optimization.status = OptimizationStatus.PAUSED
                self.logger.info(f"暂停优化任务: {optimization_id}")
                return True
            return False
    
    async def resume_optimization(self, optimization_id: str) -> bool:
        """恢复优化任务"""
        async with self._coordination_lock:
            if optimization_id in self._active_optimizations:
                optimization = self._active_optimizations[optimization_id]
                if optimization.status == OptimizationStatus.PAUSED:
                    optimization.status = OptimizationStatus.RUNNING
                    self.logger.info(f"恢复优化任务: {optimization_id}")
                    return True
            return False
    
    async def get_optimization_status(self, optimization_id: str) -> Optional[OptimizationState]:
        """获取优化状态"""
        async with self._coordination_lock:
            # 检查活跃任务
            if optimization_id in self._active_optimizations:
                return self._active_optimizations[optimization_id]
            
            # 检查队列
            for optimization in self._optimization_queue:
                if optimization.optimization_id == optimization_id:
                    return optimization
            
            return None
    
    async def get_all_optimizations(self) -> List[OptimizationState]:
        """获取所有优化任务"""
        async with self._coordination_lock:
            active = list(self._active_optimizations.values())
            queued = list(self._optimization_queue)
            return active + queued
    
    async def register_strategy(self, strategy_name: str, strategy_func: Callable) -> None:
        """注册优化策略"""
        self._optimization_strategies[strategy_name] = strategy_func
        self.logger.info(f"注册优化策略: {strategy_name}")
    
    async def set_optimization_policy(self, policy_name: str, policy_config: Dict[str, Any]) -> None:
        """设置优化策略"""
        self._optimization_policies[policy_name] = policy_config
        self.logger.info(f"设置优化策略: {policy_name}")
    
    async def _coordination_loop(self) -> None:
        """协调循环"""
        while self._running:
            try:
                await asyncio.sleep(1.0)
                
                async with self._coordination_lock:
                    # 检查是否可以启动队列中的任务
                    while (len(self._active_optimizations) < self.max_concurrent_optimizations and 
                           self._optimization_queue):
                        optimization = self._optimization_queue.pop(0)
                        await self._start_optimization(optimization)
                
                # 检查活跃任务状态
                await self._check_active_optimizations()
                
            except Exception as e:
                self.logger.error(f"协调循环异常: {e}")
                await asyncio.sleep(5)
    
    async def _start_optimization(self, optimization: OptimizationState) -> None:
        """启动优化任务"""
        optimization.status = OptimizationStatus.RUNNING
        optimization.started_at = datetime.now()
        self._active_optimizations[optimization.optimization_id] = optimization
        
        # 启动优化执行任务
        asyncio.create_task(self._execute_optimization(optimization))
        
        self.logger.info(f"启动优化任务: {optimization.optimization_id}")
    
    async def _execute_optimization(self, optimization: OptimizationState) -> None:
        """执行优化任务"""
        try:
            # 选择优化策略
            strategy_name = optimization.metadata.get('strategy', 'default')
            strategy_func = self._optimization_strategies.get(strategy_name, self._default_optimization_strategy)
            
            # 执行优化
            await strategy_func(optimization)
            
            # 更新状态
            optimization.status = OptimizationStatus.COMPLETED
            optimization.completed_at = datetime.now()
            optimization.progress = 1.0
            
        except Exception as e:
            self.logger.error(f"优化执行失败 {optimization.optimization_id}: {e}")
            optimization.status = OptimizationStatus.FAILED
            optimization.issues.append(f"执行失败: {str(e)}")
        
        finally:
            # 从活跃任务中移除
            async with self._coordination_lock:
                if optimization.optimization_id in self._active_optimizations:
                    del self._active_optimizations[optimization.optimization_id]
            
            await self._emit_event('optimization_completed', optimization)
    
    async def _check_active_optimizations(self) -> None:
        """检查活跃优化任务"""
        completed_ids = []
        
        for optimization_id, optimization in self._active_optimizations.items():
            # 检查超时
            if (optimization.started_at and 
                datetime.now() - optimization.started_at > timedelta(hours=24)):
                optimization.status = OptimizationStatus.TIMEOUT
                optimization.issues.append("优化超时")
                completed_ids.append(optimization_id)
            
            # 检查资源使用
            if optimization.cpu_usage > 95 or optimization.memory_usage > 2048:
                optimization.status = OptimizationStatus.SUSPENDED
                optimization.warnings.append("资源使用过高，已暂停")
                completed_ids.append(optimization_id)
        
        # 清理已完成的优化
        for optimization_id in completed_ids:
            if optimization_id in self._active_optimizations:
                del self._active_optimizations[optimization_id]
    
    async def _default_optimization_strategy(self, optimization: OptimizationState) -> None:
        """默认优化策略"""
        # 模拟优化过程
        total_steps = optimization.total_steps or 10
        
        for step in range(total_steps):
            if optimization.status != OptimizationStatus.RUNNING:
                break
            
            # 更新进度
            optimization.completed_steps = step + 1
            optimization.progress = (step + 1) / total_steps
            optimization.current_step = f"步骤 {step + 1}/{total_steps}"
            
            # 模拟优化工作
            await asyncio.sleep(0.5)
            
            # 更新性能指标
            optimization.current_performance = 0.5 + (step / total_steps) * 0.4
            optimization.performance_improvement = optimization.current_performance - optimization.baseline_performance
            
            # 随机生成一些问题
            if step == total_steps // 2 and optimization.priority == OptimizationPriority.HIGH:
                optimization.warnings.append("中间检查发现问题")
    
    async def _initialize_default_strategies(self) -> None:
        """初始化默认策略"""
        # 这里可以注册各种默认优化策略
        pass


# ==================== 优化生命周期管理器 ====================

class OptimizationLifecycleManager(OptimizationComponent):
    """优化生命周期管理器"""
    
    def __init__(self):
        super().__init__("OptimizationLifecycleManager")
        self._lifecycle_states: Dict[str, Dict[str, Any]] = {}
        self._state_transitions: Dict[Tuple[str, str], Callable] = {}
        self._lifecycle_hooks: Dict[str, List[Callable]] = defaultdict(list)
        self._lifecycle_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """初始化生命周期管理器"""
        self.logger.info("初始化优化生命周期管理器")
        self._running = True
        
        # 注册默认状态转换
        await self._register_default_transitions()
    
    async def start(self) -> None:
        """启动生命周期管理器"""
        self.logger.info("启动优化生命周期管理器")
    
    async def stop(self) -> None:
        """停止生命周期管理器"""
        self.logger.info("停止优化生命周期管理器")
        self._running = False
    
    async def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("清理优化生命周期管理器资源")
        self._lifecycle_states.clear()
        self._state_transitions.clear()
        self._lifecycle_hooks.clear()
    
    async def create_optimization(self, optimization: OptimizationState) -> str:
        """创建优化任务"""
        async with self._lifecycle_lock:
            # 初始化生命周期状态
            self._lifecycle_states[optimization.optimization_id] = {
                'state': 'created',
                'created_at': datetime.now(),
                'history': [],
                'metadata': optimization.metadata.copy()
            }
            
            # 执行创建钩子
            await self._execute_hooks('on_create', optimization)
            
            self.logger.info(f"创建优化任务: {optimization.optimization_id}")
            return optimization.optimization_id
    
    async def start_optimization(self, optimization_id: str) -> bool:
        """启动优化任务"""
        async with self._lifecycle_lock:
            return await self._transition_state(optimization_id, 'created', 'initializing')
    
    async def initialize_optimization(self, optimization_id: str) -> bool:
        """初始化优化任务"""
        async with self._lifecycle_lock:
            return await self._transition_state(optimization_id, 'initializing', 'running')
    
    async def pause_optimization(self, optimization_id: str) -> bool:
        """暂停优化任务"""
        async with self._lifecycle_lock:
            return await self._transition_state(optimization_id, 'running', 'paused')
    
    async def resume_optimization(self, optimization_id: str) -> bool:
        """恢复优化任务"""
        async with self._lifecycle_lock:
            return await self._transition_state(optimization_id, 'paused', 'running')
    
    async def complete_optimization(self, optimization_id: str) -> bool:
        """完成优化任务"""
        async with self._lifecycle_lock:
            return await self._transition_state(optimization_id, 'running', 'completed')
    
    async def fail_optimization(self, optimization_id: str, error: str) -> bool:
        """失败优化任务"""
        async with self._lifecycle_lock:
            success = await self._transition_state(optimization_id, 'running', 'failed')
            if success and optimization_id in self._lifecycle_states:
                self._lifecycle_states[optimization_id]['error'] = error
            return success
    
    async def cancel_optimization(self, optimization_id: str) -> bool:
        """取消优化任务"""
        async with self._lifecycle_lock:
            return await self._transition_state(optimization_id, 'any', 'cancelled')
    
    async def timeout_optimization(self, optimization_id: str) -> bool:
        """超时优化任务"""
        async with self._lifecycle_lock:
            return await self._transition_state(optimization_id, 'any', 'timeout')
    
    async def get_lifecycle_state(self, optimization_id: str) -> Optional[Dict[str, Any]]:
        """获取生命周期状态"""
        async with self._lifecycle_lock:
            return self._lifecycle_states.get(optimization_id)
    
    async def get_lifecycle_history(self, optimization_id: str) -> List[Dict[str, Any]]:
        """获取生命周期历史"""
        async with self._lifecycle_lock:
            if optimization_id in self._lifecycle_states:
                return self._lifecycle_states[optimization_id]['history']
            return []
    
    async def register_lifecycle_hook(self, event: str, hook_func: Callable) -> None:
        """注册生命周期钩子"""
        self._lifecycle_hooks[event].append(hook_func)
        self.logger.info(f"注册生命周期钩子: {event}")
    
    async def _transition_state(self, optimization_id: str, from_state: str, to_state: str) -> bool:
        """状态转换"""
        if optimization_id not in self._lifecycle_states:
            self.logger.error(f"优化任务不存在: {optimization_id}")
            return False
        
        current_state = self._lifecycle_states[optimization_id]['state']
        
        # 检查转换是否允许
        if from_state != 'any' and current_state != from_state:
            self.logger.error(f"状态转换错误: {current_state} -> {to_state} (期望从 {from_state})")
            return False
        
        # 检查转换规则
        transition_key = (current_state, to_state)
        if transition_key not in self._state_transitions:
            self.logger.error(f"不允许的状态转换: {current_state} -> {to_state}")
            return False
        
        # 执行转换前钩子
        await self._execute_hooks(f'before_{to_state}', optimization_id)
        
        # 执行转换
        transition_func = self._state_transitions[transition_key]
        success = await transition_func(optimization_id)
        
        if success:
            # 更新状态
            self._lifecycle_states[optimization_id]['state'] = to_state
            self._lifecycle_states[optimization_id]['history'].append({
                'from_state': current_state,
                'to_state': to_state,
                'timestamp': datetime.now()
            })
            
            # 执行转换后钩子
            await self._execute_hooks(f'after_{to_state}', optimization_id)
            
            self.logger.info(f"状态转换成功: {optimization_id} {current_state} -> {to_state}")
        
        return success
    
    async def _execute_hooks(self, event: str, *args) -> None:
        """执行钩子"""
        for hook_func in self._lifecycle_hooks[event]:
            try:
                if asyncio.iscoroutinefunction(hook_func):
                    await hook_func(*args)
                else:
                    hook_func(*args)
            except Exception as e:
                self.logger.error(f"执行钩子失败 {event}: {e}")
    
    async def _register_default_transitions(self) -> None:
        """注册默认状态转换"""
        # 创建 -> 初始化
        self._state_transitions[('created', 'initializing')] = self._transition_to_initializing
        
        # 初始化 -> 运行
        self._state_transitions[('initializing', 'running')] = self._transition_to_running
        
        # 运行 -> 暂停
        self._state_transitions[('running', 'paused')] = self._transition_to_paused
        
        # 暂停 -> 运行
        self._state_transitions[('paused', 'running')] = self._transition_to_running
        
        # 运行 -> 完成
        self._state_transitions[('running', 'completed')] = self._transition_to_completed
        
        # 运行 -> 失败
        self._state_transitions[('running', 'failed')] = self._transition_to_failed
        
        # 任意状态 -> 取消
        self._state_transitions[('any', 'cancelled')] = self._transition_to_cancelled
        
        # 任意状态 -> 超时
        self._state_transitions[('any', 'timeout')] = self._transition_to_timeout
    
    async def _transition_to_initializing(self, optimization_id: str) -> bool:
        """转换到初始化状态"""
        # 执行初始化逻辑
        await asyncio.sleep(0.1)  # 模拟初始化时间
        return True
    
    async def _transition_to_running(self, optimization_id: str) -> bool:
        """转换到运行状态"""
        # 执行启动逻辑
        await asyncio.sleep(0.1)  # 模拟启动时间
        return True
    
    async def _transition_to_paused(self, optimization_id: str) -> bool:
        """转换到暂停状态"""
        # 执行暂停逻辑
        return True
    
    async def _transition_to_completed(self, optimization_id: str) -> bool:
        """转换到完成状态"""
        # 执行完成逻辑
        return True
    
    async def _transition_to_failed(self, optimization_id: str) -> bool:
        """转换到失败状态"""
        # 执行失败逻辑
        return True
    
    async def _transition_to_cancelled(self, optimization_id: str) -> bool:
        """转换到取消状态"""
        # 执行取消逻辑
        return True
    
    async def _transition_to_timeout(self, optimization_id: str) -> bool:
        """转换到超时状态"""
        # 执行超时逻辑
        return True


# ==================== 优化性能统计器 ====================

class OptimizationStatistics(OptimizationComponent):
    """优化性能统计器"""
    
    def __init__(self, storage_path: str = "optimization_stats.db"):
        super().__init__("OptimizationStatistics")
        self.storage_path = storage_path
        self._statistics_cache: Dict[str, Any] = {}
        self._metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self._aggregated_stats: Dict[str, Dict[str, float]] = {}
        self._stats_lock = asyncio.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4)
    
    async def initialize(self) -> None:
        """初始化统计器"""
        self.logger.info("初始化优化性能统计器")
        self._running = True
        
        # 初始化数据库
        await self._init_database()
        
        # 启动统计聚合任务
        asyncio.create_task(self._aggregation_loop())
    
    async def start(self) -> None:
        """启动统计器"""
        self.logger.info("启动优化性能统计器")
    
    async def stop(self) -> None:
        """停止统计器"""
        self.logger.info("停止优化性能统计器")
        self._running = False
    
    async def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("清理优化性能统计器资源")
        self._executor.shutdown(wait=True)
        self._statistics_cache.clear()
        self._metrics_history.clear()
        self._aggregated_stats.clear()
    
    async def record_optimization_metric(self, optimization_id: str, metric: OptimizationMetrics) -> None:
        """记录优化指标"""
        async with self._stats_lock:
            # 缓存到内存
            self._metrics_history[optimization_id].append(metric)
            
            # 异步保存到数据库
            asyncio.create_task(self._save_metric_to_db(metric))
            
            # 更新缓存统计
            await self._update_cache_statistics(optimization_id, metric)
    
    async def get_optimization_statistics(self, optimization_id: str) -> Optional[Dict[str, Any]]:
        """获取优化统计信息"""
        async with self._stats_lock:
            return self._statistics_cache.get(optimization_id)
    
    async def get_global_statistics(self) -> Dict[str, Any]:
        """获取全局统计信息"""
        async with self._stats_lock:
            return self._aggregated_stats.copy()
    
    async def get_optimization_history(self, optimization_id: str, 
                                     limit: int = 1000) -> List[OptimizationMetrics]:
        """获取优化历史指标"""
        if optimization_id in self._metrics_history:
            return list(self._metrics_history[optimization_id])[-limit:]
        return []
    
    async def get_performance_trends(self, time_range: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """获取性能趋势"""
        cutoff_time = datetime.now() - time_range
        
        trends = {
            'execution_time_trend': [],
            'cpu_usage_trend': [],
            'memory_usage_trend': [],
            'performance_trend': [],
            'convergence_trend': []
        }
        
        # 收集所有优化的时间范围内的数据
        all_metrics = []
        for metrics_deque in self._metrics_history.values():
            for metric in metrics_deque:
                if metric.timestamp >= cutoff_time:
                    all_metrics.append(metric)
        
        # 按时间排序
        all_metrics.sort(key=lambda x: x.timestamp)
        
        # 生成趋势数据
        for metric in all_metrics:
            trends['execution_time_trend'].append({
                'timestamp': metric.timestamp.isoformat(),
                'value': metric.execution_time
            })
            trends['cpu_usage_trend'].append({
                'timestamp': metric.timestamp.isoformat(),
                'value': metric.cpu_usage
            })
            trends['memory_usage_trend'].append({
                'timestamp': metric.timestamp.isoformat(),
                'value': metric.memory_usage
            })
            trends['performance_trend'].append({
                'timestamp': metric.timestamp.isoformat(),
                'value': metric.throughput
            })
            trends['convergence_trend'].append({
                'timestamp': metric.timestamp.isoformat(),
                'value': metric.convergence_rate
            })
        
        return trends
    
    async def get_optimization_rankings(self, metric_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取优化排名"""
        rankings = []
        
        for optimization_id, stats in self._statistics_cache.items():
            if metric_name in stats:
                rankings.append({
                    'optimization_id': optimization_id,
                    'value': stats[metric_name],
                    'rank': 0  # 将在后面计算
                })
        
        # 按值排序
        rankings.sort(key=lambda x: x['value'], reverse=True)
        
        # 添加排名
        for i, ranking in enumerate(rankings[:limit]):
            ranking['rank'] = i + 1
        
        return rankings[:limit]
    
    async def _init_database(self) -> None:
        """初始化数据库"""
        def _create_tables():
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            # 创建优化指标表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    optimization_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    execution_time REAL,
                    cpu_usage REAL,
                    memory_usage REAL,
                    throughput REAL,
                    convergence_rate REAL,
                    improvement_rate REAL,
                    stability_score REAL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    profit_improvement REAL,
                    risk_reduction REAL,
                    efficiency_gain REAL
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_optimization_id ON optimization_metrics(optimization_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON optimization_metrics(timestamp)')
            
            conn.commit()
            conn.close()
        
        await asyncio.get_event_loop().run_in_executor(self._executor, _create_tables)
    
    async def _save_metric_to_db(self, metric: OptimizationMetrics) -> None:
        """保存指标到数据库"""
        def _save():
            conn = sqlite3.connect(self.storage_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO optimization_metrics (
                    optimization_id, timestamp, execution_time, cpu_usage, memory_usage,
                    throughput, convergence_rate, improvement_rate, stability_score,
                    accuracy, precision, recall, f1_score, profit_improvement,
                    risk_reduction, efficiency_gain
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metric.optimization_id, metric.timestamp.isoformat(),
                metric.execution_time, metric.cpu_usage, metric.memory_usage,
                metric.throughput, metric.convergence_rate, metric.improvement_rate,
                metric.stability_score, metric.accuracy, metric.precision,
                metric.recall, metric.f1_score, metric.profit_improvement,
                metric.risk_reduction, metric.efficiency_gain
            ))
            
            conn.commit()
            conn.close()
        
        await asyncio.get_event_loop().run_in_executor(self._executor, _save)
    
    async def _update_cache_statistics(self, optimization_id: str, metric: OptimizationMetrics) -> None:
        """更新缓存统计"""
        if optimization_id not in self._statistics_cache:
            self._statistics_cache[optimization_id] = {
                'total_executions': 0,
                'total_execution_time': 0.0,
                'avg_execution_time': 0.0,
                'max_execution_time': 0.0,
                'min_execution_time': float('inf'),
                'total_cpu_usage': 0.0,
                'avg_cpu_usage': 0.0,
                'total_memory_usage': 0.0,
                'avg_memory_usage': 0.0,
                'total_throughput': 0.0,
                'avg_throughput': 0.0,
                'total_convergence_rate': 0.0,
                'avg_convergence_rate': 0.0,
                'total_improvement_rate': 0.0,
                'avg_improvement_rate': 0.0,
                'last_updated': datetime.now()
            }
        
        stats = self._statistics_cache[optimization_id]
        stats['total_executions'] += 1
        
        # 更新执行时间统计
        stats['total_execution_time'] += metric.execution_time
        stats['avg_execution_time'] = stats['total_execution_time'] / stats['total_executions']
        stats['max_execution_time'] = max(stats['max_execution_time'], metric.execution_time)
        stats['min_execution_time'] = min(stats['min_execution_time'], metric.execution_time)
        
        # 更新CPU使用率统计
        stats['total_cpu_usage'] += metric.cpu_usage
        stats['avg_cpu_usage'] = stats['total_cpu_usage'] / stats['total_executions']
        
        # 更新内存使用率统计
        stats['total_memory_usage'] += metric.memory_usage
        stats['avg_memory_usage'] = stats['total_memory_usage'] / stats['total_executions']
        
        # 更新吞吐量统计
        stats['total_throughput'] += metric.throughput
        stats['avg_throughput'] = stats['total_throughput'] / stats['total_executions']
        
        # 更新收敛率统计
        stats['total_convergence_rate'] += metric.convergence_rate
        stats['avg_convergence_rate'] = stats['total_convergence_rate'] / stats['total_executions']
        
        # 更新改进率统计
        stats['total_improvement_rate'] += metric.improvement_rate
        stats['avg_improvement_rate'] = stats['total_improvement_rate'] / stats['total_executions']
        
        stats['last_updated'] = datetime.now()
    
    async def _aggregation_loop(self) -> None:
        """统计聚合循环"""
        while self._running:
            try:
                await asyncio.sleep(60)  # 每分钟聚合一次
                
                # 更新全局统计
                await self._update_global_statistics()
                
            except Exception as e:
                self.logger.error(f"统计聚合循环异常: {e}")
                await asyncio.sleep(10)
    
    async def _update_global_statistics(self) -> None:
        """更新全局统计"""
        if not self._statistics_cache:
            return
        
        # 计算全局统计
        total_optimizations = len(self._statistics_cache)
        total_executions = sum(stats['total_executions'] for stats in self._statistics_cache.values())
        
        if total_executions > 0:
            avg_execution_time = sum(stats['total_execution_time'] for stats in self._statistics_cache.values()) / total_executions
            avg_cpu_usage = sum(stats['total_cpu_usage'] for stats in self._statistics_cache.values()) / total_executions
            avg_memory_usage = sum(stats['total_memory_usage'] for stats in self._statistics_cache.values()) / total_executions
            avg_throughput = sum(stats['total_throughput'] for stats in self._statistics_cache.values()) / total_executions
            avg_convergence_rate = sum(stats['total_convergence_rate'] for stats in self._statistics_cache.values()) / total_executions
            avg_improvement_rate = sum(stats['total_improvement_rate'] for stats in self._statistics_cache.values()) / total_executions
        else:
            avg_execution_time = avg_cpu_usage = avg_memory_usage = avg_throughput = avg_convergence_rate = avg_improvement_rate = 0.0
        
        self._aggregated_stats = {
            'total_optimizations': total_optimizations,
            'total_executions': total_executions,
            'avg_execution_time': avg_execution_time,
            'avg_cpu_usage': avg_cpu_usage,
            'avg_memory_usage': avg_memory_usage,
            'avg_throughput': avg_throughput,
            'avg_convergence_rate': avg_convergence_rate,
            'avg_improvement_rate': avg_improvement_rate,
            'last_updated': datetime.now().isoformat()
        }


# ==================== 优化健康检查器 ====================

class OptimizationHealthChecker(OptimizationComponent):
    """优化健康检查器"""
    
    def __init__(self, check_interval: float = 30.0):
        super().__init__("OptimizationHealthChecker")
        self.check_interval = check_interval
        self._health_reports: Dict[str, OptimizationHealthReport] = {}
        self._health_thresholds: Dict[str, float] = {
            'validity_threshold': 0.8,
            'completeness_threshold': 0.9,
            'security_threshold': 0.95,
            'performance_threshold': 0.7
        }
        self._health_rules: List[Callable] = []
        self._health_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """初始化健康检查器"""
        self.logger.info("初始化优化健康检查器")
        self._running = True
        
        # 注册默认健康检查规则
        await self._register_default_health_rules()
    
    async def start(self) -> None:
        """启动健康检查器"""
        self.logger.info("启动优化健康检查器")
        # 启动健康检查循环
        asyncio.create_task(self._health_check_loop())
    
    async def stop(self) -> None:
        """停止健康检查器"""
        self.logger.info("停止优化健康检查器")
        self._running = False
    
    async def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("清理优化健康检查器资源")
        self._health_reports.clear()
        self._health_rules.clear()
    
    async def check_optimization_health(self, optimization_id: str) -> OptimizationHealthReport:
        """检查优化健康状态"""
        async with self._health_lock:
            # 执行健康检查
            validity_status, validity_score = await self._check_validity(optimization_id)
            completeness_status, completeness_score = await self._check_completeness(optimization_id)
            security_status, security_score = await self._check_security(optimization_id)
            performance_status, performance_score = await self._check_performance(optimization_id)
            
            # 计算整体健康状态
            overall_score = (validity_score + completeness_score + security_score + performance_score) / 4
            
            if overall_score >= 0.9:
                overall_status = HealthStatus.HEALTHY
            elif overall_score >= 0.7:
                overall_status = HealthStatus.WARNING
            else:
                overall_status = HealthStatus.CRITICAL
            
            # 生成健康报告
            health_report = OptimizationHealthReport(
                optimization_id=optimization_id,
                timestamp=datetime.now(),
                overall_status=overall_status,
                validity_status=validity_status,
                completeness_status=completeness_status,
                security_status=security_status,
                performance_status=performance_status,
                validity_score=validity_score,
                completeness_score=completeness_score,
                security_score=security_score,
                performance_score=performance_score,
                issues=[],
                recommendations=[]
            )
            
            # 执行自定义健康规则
            await self._apply_health_rules(optimization_id, health_report)
            
            # 缓存报告
            self._health_reports[optimization_id] = health_report
            
            self.logger.info(f"完成健康检查: {optimization_id}, 状态: {overall_status.value}")
            return health_report
    
    async def get_health_report(self, optimization_id: str) -> Optional[OptimizationHealthReport]:
        """获取健康报告"""
        async with self._health_lock:
            return self._health_reports.get(optimization_id)
    
    async def get_all_health_reports(self) -> Dict[str, OptimizationHealthReport]:
        """获取所有健康报告"""
        async with self._health_lock:
            return self._health_reports.copy()
    
    async def set_health_threshold(self, threshold_name: str, value: float) -> None:
        """设置健康阈值"""
        self._health_thresholds[threshold_name] = value
        self.logger.info(f"设置健康阈值: {threshold_name} = {value}")
    
    async def register_health_rule(self, rule_func: Callable) -> None:
        """注册健康检查规则"""
        self._health_rules.append(rule_func)
        self.logger.info(f"注册健康检查规则: {rule_func.__name__}")
    
    async def _health_check_loop(self) -> None:
        """健康检查循环"""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
                
                # 这里可以定期检查所有活跃的优化任务
                # 目前只是示例，实际实现需要从协调器获取活跃任务列表
                
            except Exception as e:
                self.logger.error(f"健康检查循环异常: {e}")
                await asyncio.sleep(10)
    
    async def _check_validity(self, optimization_id: str) -> Tuple[HealthStatus, float]:
        """检查有效性"""
        # 模拟有效性检查
        score = 0.85  # 模拟得分
        
        if score >= self._health_thresholds['validity_threshold']:
            status = HealthStatus.HEALTHY
        elif score >= 0.6:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.CRITICAL
        
        return status, score
    
    async def _check_completeness(self, optimization_id: str) -> Tuple[HealthStatus, float]:
        """检查完整性"""
        # 模拟完整性检查
        score = 0.92  # 模拟得分
        
        if score >= self._health_thresholds['completeness_threshold']:
            status = HealthStatus.HEALTHY
        elif score >= 0.7:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.CRITICAL
        
        return status, score
    
    async def _check_security(self, optimization_id: str) -> Tuple[HealthStatus, float]:
        """检查安全性"""
        # 模拟安全性检查
        score = 0.98  # 模拟得分
        
        if score >= self._health_thresholds['security_threshold']:
            status = HealthStatus.HEALTHY
        elif score >= 0.8:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.CRITICAL
        
        return status, score
    
    async def _check_performance(self, optimization_id: str) -> Tuple[HealthStatus, float]:
        """检查性能"""
        # 模拟性能检查
        score = 0.75  # 模拟得分
        
        if score >= self._health_thresholds['performance_threshold']:
            status = HealthStatus.HEALTHY
        elif score >= 0.5:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.CRITICAL
        
        return status, score
    
    async def _apply_health_rules(self, optimization_id: str, health_report: OptimizationHealthReport) -> None:
        """应用健康检查规则"""
        for rule_func in self._health_rules:
            try:
                if asyncio.iscoroutinefunction(rule_func):
                    await rule_func(optimization_id, health_report)
                else:
                    rule_func(optimization_id, health_report)
            except Exception as e:
                self.logger.error(f"执行健康检查规则失败 {rule_func.__name__}: {e}")
    
    async def _register_default_health_rules(self) -> None:
        """注册默认健康检查规则"""
        # 这里可以注册各种默认的健康检查规则
        pass


# ==================== 异步状态同步器 ====================

class AsyncStateSynchronizer(OptimizationComponent):
    """异步状态同步器"""
    
    def __init__(self, sync_interval: float = 5.0, max_sync_workers: int = 3):
        super().__init__("AsyncStateSynchronizer")
        self.sync_interval = sync_interval
        self.max_sync_workers = max_sync_workers
        self._sync_queue: asyncio.Queue = asyncio.Queue()
        self._sync_workers: List[asyncio.Task] = []
        self._sync_strategies: Dict[str, Callable] = {}
        self._distributed_nodes: Dict[str, Dict[str, Any]] = {}
        self._sync_lock = asyncio.Lock()
        self._sync_stats: Dict[str, int] = defaultdict(int)
    
    async def initialize(self) -> None:
        """初始化同步器"""
        self.logger.info("初始化异步状态同步器")
        self._running = True
        
        # 初始化同步策略
        await self._initialize_sync_strategies()
        
        # 启动同步工作器
        await self._start_sync_workers()
    
    async def start(self) -> None:
        """启动同步器"""
        self.logger.info("启动异步状态同步器")
        # 启动同步循环
        asyncio.create_task(self._sync_loop())
    
    async def stop(self) -> None:
        """停止同步器"""
        self.logger.info("停止异步状态同步器")
        self._running = False
        
        # 取消所有工作器
        for worker in self._sync_workers:
            worker.cancel()
        
        if self._sync_workers:
            await asyncio.gather(*self._sync_workers, return_exceptions=True)
    
    async def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("清理异步状态同步器资源")
        self._sync_queue = asyncio.Queue()
        self._sync_strategies.clear()
        self._distributed_nodes.clear()
        self._sync_stats.clear()
    
    async def register_node(self, node_id: str, node_info: Dict[str, Any]) -> None:
        """注册分布式节点"""
        async with self._sync_lock:
            self._distributed_nodes[node_id] = {
                'info': node_info,
                'last_seen': datetime.now(),
                'status': 'active'
            }
            self.logger.info(f"注册分布式节点: {node_id}")
    
    async def unregister_node(self, node_id: str) -> None:
        """注销分布式节点"""
        async with self._sync_lock:
            if node_id in self._distributed_nodes:
                del self._distributed_nodes[node_id]
                self.logger.info(f"注销分布式节点: {node_id}")
    
    async def sync_optimization_state(self, optimization_id: str, 
                                    target_nodes: Optional[List[str]] = None) -> None:
        """同步优化状态"""
        sync_task = {
            'optimization_id': optimization_id,
            'target_nodes': target_nodes or list(self._distributed_nodes.keys()),
            'timestamp': datetime.now(),
            'sync_type': 'state'
        }
        
        await self._sync_queue.put(sync_task)
        self.logger.info(f"加入同步队列: {optimization_id}")
    
    async def broadcast_optimization_update(self, optimization_id: str, 
                                          update_data: Dict[str, Any]) -> None:
        """广播优化更新"""
        broadcast_task = {
            'optimization_id': optimization_id,
            'update_data': update_data,
            'timestamp': datetime.now(),
            'sync_type': 'broadcast'
        }
        
        await self._sync_queue.put(broadcast_task)
        self.logger.info(f"广播优化更新: {optimization_id}")
    
    async def get_sync_statistics(self) -> Dict[str, Any]:
        """获取同步统计"""
        async with self._sync_lock:
            return {
                'queue_size': self._sync_queue.qsize(),
                'active_nodes': len([n for n in self._distributed_nodes.values() if n['status'] == 'active']),
                'total_nodes': len(self._distributed_nodes),
                'sync_stats': dict(self._sync_stats),
                'last_sync': max(self._sync_stats.keys()) if self._sync_stats else None
            }
    
    async def _start_sync_workers(self) -> None:
        """启动同步工作器"""
        for i in range(self.max_sync_workers):
            worker = asyncio.create_task(self._sync_worker(f"worker-{i}"))
            self._sync_workers.append(worker)
        
        self.logger.info(f"启动 {self.max_sync_workers} 个同步工作器")
    
    async def _sync_loop(self) -> None:
        """同步循环"""
        while self._running:
            try:
                await asyncio.sleep(self.sync_interval)
                
                # 检查节点健康状态
                await self._check_node_health()
                
                # 清理过期任务
                await self._cleanup_expired_tasks()
                
            except Exception as e:
                self.logger.error(f"同步循环异常: {e}")
                await asyncio.sleep(10)
    
    async def _sync_worker(self, worker_id: str) -> None:
        """同步工作器"""
        while self._running:
            try:
                # 获取同步任务
                sync_task = await asyncio.wait_for(self._sync_queue.get(), timeout=1.0)
                
                # 执行同步
                await self._execute_sync_task(worker_id, sync_task)
                
                # 标记任务完成
                self._sync_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"同步工作器异常 {worker_id}: {e}")
                await asyncio.sleep(1)
    
    async def _execute_sync_task(self, worker_id: str, sync_task: Dict[str, Any]) -> None:
        """执行同步任务"""
        try:
            optimization_id = sync_task['optimization_id']
            sync_type = sync_task['sync_type']
            
            if sync_type == 'state':
                await self._sync_optimization_state(optimization_id, sync_task['target_nodes'])
            elif sync_type == 'broadcast':
                await self._broadcast_update(optimization_id, sync_task['update_data'])
            
            # 更新统计
            self._sync_stats[f"sync_{datetime.now().strftime('%Y%m%d_%H%M%S')}"] = 1
            
        except Exception as e:
            self.logger.error(f"执行同步任务失败: {e}")
    
    async def _sync_optimization_state(self, optimization_id: str, target_nodes: List[str]) -> None:
        """同步优化状态"""
        # 选择同步策略
        sync_strategy = self._sync_strategies.get('default', self._default_sync_strategy)
        
        # 执行同步
        await sync_strategy(optimization_id, target_nodes)
    
    async def _broadcast_update(self, optimization_id: str, update_data: Dict[str, Any]) -> None:
        """广播更新"""
        # 广播到所有活跃节点
        active_nodes = [node_id for node_id, node_info in self._distributed_nodes.items() 
                       if node_info['status'] == 'active']
        
        for node_id in active_nodes:
            try:
                # 模拟网络发送
                await asyncio.sleep(0.01)  # 模拟网络延迟
                self.logger.debug(f"发送更新到节点 {node_id}: {optimization_id}")
            except Exception as e:
                self.logger.error(f"发送更新到节点 {node_id} 失败: {e}")
    
    async def _check_node_health(self) -> None:
        """检查节点健康状态"""
        current_time = datetime.now()
        timeout_threshold = timedelta(minutes=5)
        
        for node_id, node_info in self._distributed_nodes.items():
            if current_time - node_info['last_seen'] > timeout_threshold:
                node_info['status'] = 'inactive'
                self.logger.warning(f"节点 {node_id} 标记为不活跃")
    
    async def _cleanup_expired_tasks(self) -> None:
        """清理过期任务"""
        # 这里可以实现任务过期清理逻辑
        pass
    
    async def _default_sync_strategy(self, optimization_id: str, target_nodes: List[str]) -> None:
        """默认同步策略"""
        # 模拟状态同步
        for node_id in target_nodes:
            if node_id in self._distributed_nodes:
                try:
                    # 模拟网络同步
                    await asyncio.sleep(0.05)  # 模拟网络延迟
                    self.logger.debug(f"同步状态到节点 {node_id}: {optimization_id}")
                except Exception as e:
                    self.logger.error(f"同步到节点 {node_id} 失败: {e}")
    
    async def _initialize_sync_strategies(self) -> None:
        """初始化同步策略"""
        # 注册默认同步策略
        self._sync_strategies['default'] = self._default_sync_strategy
        
        # 可以注册其他同步策略，如：
        # self._sync_strategies['priority'] = self._priority_sync_strategy
        # self._sync_strategies['batch'] = self._batch_sync_strategy


# ==================== 告警通知系统 ====================

class AlertNotificationSystem(OptimizationComponent):
    """告警通知系统"""
    
    def __init__(self, notification_channels: Optional[Dict[str, Callable]] = None):
        super().__init__("AlertNotificationSystem")
        self.notification_channels = notification_channels or {}
        self._alert_queue: asyncio.Queue = asyncio.Queue()
        self._alert_rules: List[Callable] = []
        self._notification_history: deque = deque(maxlen=10000)
        self._alert_suppression: Dict[str, datetime] = {}
        self._alert_stats: Dict[str, int] = defaultdict(int)
        self._alert_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """初始化告警通知系统"""
        self.logger.info("初始化告警通知系统")
        self._running = True
        
        # 注册默认通知渠道
        await self._register_default_channels()
        
        # 注册默认告警规则
        await self._register_default_rules()
    
    async def start(self) -> None:
        """启动告警通知系统"""
        self.logger.info("启动告警通知系统")
        # 启动告警处理循环
        asyncio.create_task(self._alert_processing_loop())
    
    async def stop(self) -> None:
        """停止告警通知系统"""
        self.logger.info("停止告警通知系统")
        self._running = False
    
    async def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("清理告警通知系统资源")
        self._alert_queue = asyncio.Queue()
        self._alert_rules.clear()
        self._notification_history.clear()
        self._alert_suppression.clear()
        self._alert_stats.clear()
    
    async def send_alert(self, alert: OptimizationAlert) -> None:
        """发送告警"""
        # 检查告警抑制
        if await self._is_alert_suppressed(alert):
            self.logger.debug(f"告警被抑制: {alert.alert_id}")
            return
        
        # 检查告警规则
        if await self._check_alert_rules(alert):
            await self._alert_queue.put(alert)
            self.logger.info(f"告警加入队列: {alert.alert_id}")
        else:
            self.logger.debug(f"告警被规则过滤: {alert.alert_id}")
    
    async def register_notification_channel(self, channel_name: str, channel_func: Callable) -> None:
        """注册通知渠道"""
        self.notification_channels[channel_name] = channel_func
        self.logger.info(f"注册通知渠道: {channel_name}")
    
    async def register_alert_rule(self, rule_func: Callable) -> None:
        """注册告警规则"""
        self._alert_rules.append(rule_func)
        self.logger.info(f"注册告警规则: {rule_func.__name__}")
    
    async def suppress_alert(self, alert_pattern: str, duration: timedelta) -> None:
        """抑制告警"""
        self._alert_suppression[alert_pattern] = datetime.now() + duration
        self.logger.info(f"抑制告警模式: {alert_pattern}, 持续时间: {duration}")
    
    async def get_notification_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取通知历史"""
        history = list(self._notification_history)[-limit:]
        return [item.to_dict() if hasattr(item, 'to_dict') else item for item in history]
    
    async def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        return {
            'queue_size': self._alert_queue.qsize(),
            'total_alerts': sum(self._alert_stats.values()),
            'alert_stats': dict(self._alert_stats),
            'suppressed_patterns': len(self._alert_suppression),
            'active_channels': len(self.notification_channels)
        }
    
    async def _alert_processing_loop(self) -> None:
        """告警处理循环"""
        while self._running:
            try:
                # 获取告警
                alert = await asyncio.wait_for(self._alert_queue.get(), timeout=1.0)
                
                # 处理告警
                await self._process_alert(alert)
                
                # 标记任务完成
                self._alert_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"告警处理循环异常: {e}")
                await asyncio.sleep(1)
    
    async def _process_alert(self, alert: OptimizationAlert) -> None:
        """处理告警"""
        try:
            # 发送到所有注册的通知渠道
            for channel_name, channel_func in self.notification_channels.items():
                try:
                    if asyncio.iscoroutinefunction(channel_func):
                        await channel_func(alert)
                    else:
                        channel_func(alert)
                    
                    self.logger.debug(f"告警通过渠道 {channel_name} 发送成功")
                    
                except Exception as e:
                    self.logger.error(f"通过渠道 {channel_name} 发送告警失败: {e}")
            
            # 记录到历史
            self._notification_history.append(alert)
            
            # 更新统计
            self._alert_stats[alert.level.value] += 1
            
        except Exception as e:
            self.logger.error(f"处理告警失败: {e}")
    
    async def _is_alert_suppressed(self, alert: OptimizationAlert) -> bool:
        """检查告警是否被抑制"""
        current_time = datetime.now()
        
        # 检查精确匹配
        if alert.alert_id in self._alert_suppression:
            if current_time < self._alert_suppression[alert.alert_id]:
                return True
            else:
                del self._alert_suppression[alert.alert_id]
        
        # 检查模式匹配
        for pattern, expiry_time in list(self._alert_suppression.items()):
            if current_time >= expiry_time:
                del self._alert_suppression[pattern]
            elif self._match_alert_pattern(alert, pattern):
                return True
        
        return False
    
    def _match_alert_pattern(self, alert: OptimizationAlert, pattern: str) -> bool:
        """匹配告警模式"""
        # 简单的模式匹配实现
        # 可以扩展为更复杂的匹配规则
        return (pattern in alert.title or 
                pattern in alert.message or 
                pattern == alert.optimization_id)
    
    async def _check_alert_rules(self, alert: OptimizationAlert) -> bool:
        """检查告警规则"""
        for rule_func in self._alert_rules:
            try:
                if asyncio.iscoroutinefunction(rule_func):
                    result = await rule_func(alert)
                else:
                    result = rule_func(alert)
                
                if not result:
                    return False
                    
            except Exception as e:
                self.logger.error(f"执行告警规则失败 {rule_func.__name__}: {e}")
        
        return True
    
    async def _register_default_channels(self) -> None:
        """注册默认通知渠道"""
        # 日志渠道
        self.notification_channels['log'] = self._log_notification
        
        # 控制台渠道
        self.notification_channels['console'] = self._console_notification
        
        # 文件渠道
        self.notification_channels['file'] = self._file_notification
    
    async def _register_default_rules(self) -> None:
        """注册默认告警规则"""
        # 关键告警规则
        self._alert_rules.append(self._critical_alert_rule)
        
        # 重复告警抑制规则
        self._alert_rules.append(self._duplicate_alert_rule)
        
        # 频率限制规则
        self._alert_rules.append(self._frequency_limit_rule)
    
    async def _log_notification(self, alert: OptimizationAlert) -> None:
        """日志通知"""
        log_message = f"[{alert.level.value.upper()}] {alert.title}: {alert.message}"
        if alert.level == AlertLevel.CRITICAL:
            self.logger.critical(log_message)
        elif alert.level == AlertLevel.ERROR:
            self.logger.error(log_message)
        elif alert.level == AlertLevel.WARNING:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    async def _console_notification(self, alert: OptimizationAlert) -> None:
        """控制台通知"""
        print(f"[{alert.level.value.upper()}] {alert.title}: {alert.message}")
    
    async def _file_notification(self, alert: OptimizationAlert) -> None:
        """文件通知"""
        # 模拟文件写入
        await asyncio.sleep(0.01)  # 模拟文件I/O
    
    async def _critical_alert_rule(self, alert: OptimizationAlert) -> bool:
        """关键告警规则"""
        # 关键告警总是通过
        return True
    
    async def _duplicate_alert_rule(self, alert: OptimizationAlert) -> bool:
        """重复告警抑制规则"""
        # 检查最近是否有相同的告警
        recent_alerts = list(self._notification_history)[-10:]
        for recent_alert in recent_alerts:
            if (recent_alert.optimization_id == alert.optimization_id and
                recent_alert.title == alert.title and
                recent_alert.level == alert.level):
                # 检查时间间隔
                time_diff = alert.timestamp - recent_alert.timestamp
                if time_diff < timedelta(minutes=5):  # 5分钟内不重复发送
                    return False
        
        return True
    
    async def _frequency_limit_rule(self, alert: OptimizationAlert) -> bool:
        """频率限制规则"""
        # 简单的频率限制实现
        # 可以根据告警级别设置不同的频率限制
        return True


# ==================== 统一优化API ====================

class OptimizationAPI(OptimizationComponent):
    """统一优化API接口"""
    
    def __init__(self, api_version: str = "v1"):
        super().__init__("OptimizationAPI")
        self.api_version = api_version
        self._api_handlers: Dict[str, Callable] = {}
        self._rate_limiter: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._api_stats: Dict[str, int] = defaultdict(int)
        self._api_lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """初始化API接口"""
        self.logger.info(f"初始化优化API接口 v{self.api_version}")
        self._running = True
        
        # 注册API处理器
        await self._register_api_handlers()
    
    async def start(self) -> None:
        """启动API接口"""
        self.logger.info("启动优化API接口")
    
    async def stop(self) -> None:
        """停止API接口"""
        self.logger.info("停止优化API接口")
        self._running = False
    
    async def cleanup(self) -> None:
        """清理资源"""
        self.logger.info("清理优化API接口资源")
        self._api_handlers.clear()
        self._rate_limiter.clear()
        self._api_stats.clear()
    
    async def handle_api_request(self, endpoint: str, method: str, 
                               data: Optional[Dict[str, Any]] = None,
                               client_id: Optional[str] = None) -> Dict[str, Any]:
        """处理API请求"""
        try:
            # 速率限制检查
            if not await self._check_rate_limit(client_id or "anonymous"):
                return {
                    'success': False,
                    'error': 'Rate limit exceeded',
                    'code': 429
                }
            
            # 查找处理器
            handler_key = f"{method.upper()}:{endpoint}"
            if handler_key not in self._api_handlers:
                return {
                    'success': False,
                    'error': 'Endpoint not found',
                    'code': 404
                }
            
            # 执行处理器
            handler = self._api_handlers[handler_key]
            result = await handler(data) if asyncio.iscoroutinefunction(handler) else handler(data)
            
            # 更新统计
            self._api_stats[endpoint] += 1
            
            return {
                'success': True,
                'data': result,
                'timestamp': datetime.now().isoformat(),
                'api_version': self.api_version
            }
            
        except Exception as e:
            self.logger.error(f"API请求处理失败 {endpoint}: {e}")
            return {
                'success': False,
                'error': str(e),
                'code': 500
            }
    
    async def register_api_handler(self, endpoint: str, method: str, handler: Callable) -> None:
        """注册API处理器"""
        handler_key = f"{method.upper()}:{endpoint}"
        self._api_handlers[handler_key] = handler
        self.logger.info(f"注册API处理器: {handler_key}")
    
    async def get_api_stats(self) -> Dict[str, Any]:
        """获取API统计"""
        async with self._api_lock:
            return {
                'api_version': self.api_version,
                'registered_endpoints': len(self._api_handlers),
                'endpoint_stats': dict(self._api_stats),
                'total_requests': sum(self._api_stats.values())
            }
    
    async def _check_rate_limit(self, client_id: str) -> bool:
        """检查速率限制"""
        current_time = time.time()
        client_requests = self._rate_limiter[client_id]
        
        # 清理过期请求（1分钟窗口）
        while client_requests and current_time - client_requests[0] > 60:
            client_requests.popleft()
        
        # 检查是否超过限制（每分钟100个请求）
        if len(client_requests) >= 100:
            return False
        
        # 记录当前请求
        client_requests.append(current_time)
        return True
    
    async def _register_api_handlers(self) -> None:
        """注册API处理器"""
        # 优化管理API
        await self.register_api_handler('/optimizations', 'POST', self._create_optimization)
        await self.register_api_handler('/optimizations', 'GET', self._list_optimizations)
        await self.register_api_handler('/optimizations/{id}', 'GET', self._get_optimization)
        await self.register_api_handler('/optimizations/{id}', 'PUT', self._update_optimization)
        await self.register_api_handler('/optimizations/{id}', 'DELETE', self._delete_optimization)
        
        # 优化控制API
        await self.register_api_handler('/optimizations/{id}/start', 'POST', self._start_optimization)
        await self.register_api_handler('/optimizations/{id}/pause', 'POST', self._pause_optimization)
        await self.register_api_handler('/optimizations/{id}/resume', 'POST', self._resume_optimization)
        await self.register_api_handler('/optimizations/{id}/cancel', 'POST', self._cancel_optimization)
        
        # 统计API
        await self.register_api_handler('/statistics', 'GET', self._get_statistics)
        await self.register_api_handler('/statistics/trends', 'GET', self._get_trends)
        await self.register_api_handler('/statistics/rankings', 'GET', self._get_rankings)
        
        # 健康检查API
        await self.register_api_handler('/health', 'GET', self._get_health_status)
        await self.register_api_handler('/health/{id}', 'GET', self._get_optimization_health)
        
        # 告警API
        await self.register_api_handler('/alerts', 'GET', self._get_alerts)
        await self.register_api_handler('/alerts/{id}/resolve', 'POST', self._resolve_alert)
    
    # API处理器实现
    async def _create_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """创建优化任务"""
        # 模拟创建优化任务
        optimization_id = str(uuid.uuid4())
        return {
            'optimization_id': optimization_id,
            'status': 'created',
            'message': 'Optimization created successfully'
        }
    
    async def _list_optimizations(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """列出优化任务"""
        # 模拟返回优化任务列表
        return {
            'optimizations': [],
            'total': 0,
            'page': 1,
            'per_page': 10
        }
    
    async def _get_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """获取优化任务详情"""
        optimization_id = data.get('id', '')
        return {
            'optimization_id': optimization_id,
            'status': 'running',
            'progress': 0.5,
            'message': 'Optimization details retrieved'
        }
    
    async def _update_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """更新优化任务"""
        optimization_id = data.get('id', '')
        return {
            'optimization_id': optimization_id,
            'status': 'updated',
            'message': 'Optimization updated successfully'
        }
    
    async def _delete_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """删除优化任务"""
        optimization_id = data.get('id', '')
        return {
            'optimization_id': optimization_id,
            'status': 'deleted',
            'message': 'Optimization deleted successfully'
        }
    
    async def _start_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """启动优化任务"""
        optimization_id = data.get('id', '')
        return {
            'optimization_id': optimization_id,
            'status': 'started',
            'message': 'Optimization started successfully'
        }
    
    async def _pause_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """暂停优化任务"""
        optimization_id = data.get('id', '')
        return {
            'optimization_id': optimization_id,
            'status': 'paused',
            'message': 'Optimization paused successfully'
        }
    
    async def _resume_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """恢复优化任务"""
        optimization_id = data.get('id', '')
        return {
            'optimization_id': optimization_id,
            'status': 'resumed',
            'message': 'Optimization resumed successfully'
        }
    
    async def _cancel_optimization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """取消优化任务"""
        optimization_id = data.get('id', '')
        return {
            'optimization_id': optimization_id,
            'status': 'cancelled',
            'message': 'Optimization cancelled successfully'
        }
    
    async def _get_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_optimizations': 0,
            'active_optimizations': 0,
            'completed_optimizations': 0,
            'failed_optimizations': 0,
            'avg_execution_time': 0.0,
            'message': 'Statistics retrieved successfully'
        }
    
    async def _get_trends(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """获取趋势数据"""
        return {
            'trends': {},
            'message': 'Trends retrieved successfully'
        }
    
    async def _get_rankings(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """获取排名数据"""
        return {
            'rankings': [],
            'message': 'Rankings retrieved successfully'
        }
    
    async def _get_health_status(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """获取健康状态"""
        return {
            'overall_status': 'healthy',
            'components': {},
            'message': 'Health status retrieved successfully'
        }
    
    async def _get_optimization_health(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """获取优化健康状态"""
        optimization_id = data.get('id', '')
        return {
            'optimization_id': optimization_id,
            'health_status': 'healthy',
            'scores': {},
            'message': 'Optimization health retrieved successfully'
        }
    
    async def _get_alerts(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """获取告警列表"""
        return {
            'alerts': [],
            'total': 0,
            'message': 'Alerts retrieved successfully'
        }
    
    async def _resolve_alert(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """解决告警"""
        alert_id = data.get('id', '')
        return {
            'alert_id': alert_id,
            'status': 'resolved',
            'message': 'Alert resolved successfully'
        }


# ==================== 主要聚合器类 ====================

class OptimizationStateAggregator:
    """O9优化状态聚合器主类
    
    这是O9优化状态聚合器的核心类，集成了所有优化相关功能：
    - 优化状态监控
    - 优化协调管理  
    - 优化生命周期管理
    - 优化性能统计
    - 优化健康检查
    - 统一优化接口和API
    - 异步优化状态同步和分布式协调
    - 优化告警和通知系统
    
    使用示例：
        ```python
        # 创建聚合器实例
        aggregator = OptimizationStateAggregator()
        
        # 初始化和启动
        await aggregator.initialize()
        await aggregator.start()
        
        # 创建优化任务
        optimization = OptimizationState(
            optimization_id="opt_001",
            name="参数优化",
            description="优化模型参数",
            optimization_type=OptimizationType.PARAMETER_TUNING,
            status=OptimizationStatus.CREATED,
            priority=OptimizationPriority.NORMAL,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # 提交优化任务
        await aggregator.submit_optimization(optimization)
        
        # 获取优化状态
        status = await aggregator.get_optimization_status("opt_001")
        
        # 停止聚合器
        await aggregator.stop()
        ```
    """
    
    def __init__(self, 
                 max_concurrent_optimizations: int = 10,
                 monitoring_interval: float = 1.0,
                 health_check_interval: float = 30.0,
                 sync_interval: float = 5.0,
                 storage_path: str = "optimization_state.db"):
        """初始化优化状态聚合器
        
        Args:
            max_concurrent_optimizations: 最大并发优化任务数
            monitoring_interval: 监控间隔（秒）
            health_check_interval: 健康检查间隔（秒）
            sync_interval: 同步间隔（秒）
            storage_path: 数据存储路径
        """
        self.max_concurrent_optimizations = max_concurrent_optimizations
        self.monitoring_interval = monitoring_interval
        self.health_check_interval = health_check_interval
        self.sync_interval = sync_interval
        self.storage_path = storage_path
        
        # 初始化各个组件
        self.monitor = OptimizationMonitor(monitoring_interval)
        self.coordinator = OptimizationCoordinator(max_concurrent_optimizations)
        self.lifecycle_manager = OptimizationLifecycleManager()
        self.statistics = OptimizationStatistics(storage_path)
        self.health_checker = OptimizationHealthChecker(health_check_interval)
        self.api = OptimizationAPI()
        self.synchronizer = AsyncStateSynchronizer(sync_interval)
        self.alert_system = AlertNotificationSystem()
        
        # 组件列表
        self._components = [
            self.monitor,
            self.coordinator,
            self.lifecycle_manager,
            self.statistics,
            self.health_checker,
            self.api,
            self.synchronizer,
            self.alert_system
        ]
        
        # 状态
        self._initialized = False
        self._running = False
        self.logger = logging.getLogger(__name__)
        
        # 注册组件间的事件监听
        self._register_event_listeners()
    
    async def initialize(self) -> None:
        """初始化聚合器"""
        if self._initialized:
            self.logger.warning("聚合器已经初始化")
            return
        
        self.logger.info("初始化O9优化状态聚合器")
        
        try:
            # 初始化所有组件
            for component in self._components:
                await component.initialize()
            
            self._initialized = True
            self.logger.info("O9优化状态聚合器初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化聚合器失败: {e}")
            raise
    
    async def start(self) -> None:
        """启动聚合器"""
        if not self._initialized:
            raise OptimizationError("聚合器尚未初始化，请先调用initialize()")
        
        if self._running:
            self.logger.warning("聚合器已经在运行")
            return
        
        self.logger.info("启动O9优化状态聚合器")
        
        try:
            # 启动所有组件
            for component in self._components:
                await component.start()
            
            self._running = True
            self.logger.info("O9优化状态聚合器启动完成")
            
        except Exception as e:
            self.logger.error(f"启动聚合器失败: {e}")
            raise
    
    async def stop(self) -> None:
        """停止聚合器"""
        if not self._running:
            self.logger.warning("聚合器未在运行")
            return
        
        self.logger.info("停止O9优化状态聚合器")
        
        try:
            # 停止所有组件
            for component in reversed(self._components):
                await component.stop()
            
            self._running = False
            self.logger.info("O9优化状态聚合器已停止")
            
        except Exception as e:
            self.logger.error(f"停止聚合器失败: {e}")
            raise
    
    async def cleanup(self) -> None:
        """清理聚合器资源"""
        self.logger.info("清理O9优化状态聚合器资源")
        
        try:
            # 清理所有组件
            for component in self._components:
                await component.cleanup()
            
            self._initialized = False
            self._running = False
            self.logger.info("O9优化状态聚合器资源清理完成")
            
        except Exception as e:
            self.logger.error(f"清理聚合器资源失败: {e}")
            raise
    
    # ========== 优化管理接口 ==========
    
    async def submit_optimization(self, optimization: OptimizationState) -> str:
        """提交优化任务
        
        Args:
            optimization: 优化任务状态对象
            
        Returns:
            str: 优化任务ID
            
        Raises:
            OptimizationError: 提交失败时抛出
        """
        if not self._running:
            raise OptimizationError("聚合器未在运行")
        
        try:
            # 创建优化任务
            await self.lifecycle_manager.create_optimization(optimization)
            
            # 提交到协调器
            optimization_id = await self.coordinator.submit_optimization(optimization)
            
            # 开始监控
            await self.monitor.start_monitoring(optimization_id)
            
            # 同步状态
            await self.synchronizer.sync_optimization_state(optimization_id)
            
            self.logger.info(f"成功提交优化任务: {optimization_id}")
            return optimization_id
            
        except Exception as e:
            self.logger.error(f"提交优化任务失败: {e}")
            raise OptimizationError(f"提交优化任务失败: {e}")
    
    async def cancel_optimization(self, optimization_id: str) -> bool:
        """取消优化任务
        
        Args:
            optimization_id: 优化任务ID
            
        Returns:
            bool: 是否成功取消
        """
        if not self._running:
            raise OptimizationError("聚合器未在运行")
        
        try:
            # 取消协调器中的任务
            success = await self.coordinator.cancel_optimization(optimization_id)
            
            if success:
                # 更新生命周期状态
                await self.lifecycle_manager.cancel_optimization(optimization_id)
                
                # 停止监控
                await self.monitor.stop_monitoring(optimization_id)
                
                # 同步状态
                await self.synchronizer.sync_optimization_state(optimization_id)
                
                self.logger.info(f"成功取消优化任务: {optimization_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"取消优化任务失败 {optimization_id}: {e}")
            raise OptimizationError(f"取消优化任务失败: {e}")
    
    async def pause_optimization(self, optimization_id: str) -> bool:
        """暂停优化任务
        
        Args:
            optimization_id: 优化任务ID
            
        Returns:
            bool: 是否成功暂停
        """
        if not self._running:
            raise OptimizationError("聚合器未在运行")
        
        try:
            # 暂停协调器中的任务
            success = await self.coordinator.pause_optimization(optimization_id)
            
            if success:
                # 更新生命周期状态
                await self.lifecycle_manager.pause_optimization(optimization_id)
                
                # 同步状态
                await self.synchronizer.sync_optimization_state(optimization_id)
                
                self.logger.info(f"成功暂停优化任务: {optimization_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"暂停优化任务失败 {optimization_id}: {e}")
            raise OptimizationError(f"暂停优化任务失败: {e}")
    
    async def resume_optimization(self, optimization_id: str) -> bool:
        """恢复优化任务
        
        Args:
            optimization_id: 优化任务ID
            
        Returns:
            bool: 是否成功恢复
        """
        if not self._running:
            raise OptimizationError("聚合器未在运行")
        
        try:
            # 恢复协调器中的任务
            success = await self.coordinator.resume_optimization(optimization_id)
            
            if success:
                # 更新生命周期状态
                await self.lifecycle_manager.resume_optimization(optimization_id)
                
                # 同步状态
                await self.synchronizer.sync_optimization_state(optimization_id)
                
                self.logger.info(f"成功恢复优化任务: {optimization_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"恢复优化任务失败 {optimization_id}: {e}")
            raise OptimizationError(f"恢复优化任务失败: {e}")
    
    # ========== 状态查询接口 ==========
    
    async def get_optimization_status(self, optimization_id: str) -> Optional[OptimizationState]:
        """获取优化状态
        
        Args:
            optimization_id: 优化任务ID
            
        Returns:
            Optional[OptimizationState]: 优化状态对象，如果不存在则返回None
        """
        if not self._running:
            raise OptimizationError("聚合器未在运行")
        
        try:
            # 从协调器获取状态
            status = await self.coordinator.get_optimization_status(optimization_id)
            return status
            
        except Exception as e:
            self.logger.error(f"获取优化状态失败 {optimization_id}: {e}")
            raise OptimizationError(f"获取优化状态失败: {e}")
    
    async def get_all_optimizations(self) -> List[OptimizationState]:
        """获取所有优化任务
        
        Returns:
            List[OptimizationState]: 所有优化任务列表
        """
        if not self._running:
            raise OptimizationError("聚合器未在运行")
        
        try:
            # 从协调器获取所有优化任务
            optimizations = await self.coordinator.get_all_optimizations()
            return optimizations
            
        except Exception as e:
            self.logger.error(f"获取所有优化任务失败: {e}")
            raise OptimizationError(f"获取所有优化任务失败: {e}")
    
    async def get_optimization_metrics(self, optimization_id: str) -> List[OptimizationMetrics]:
        """获取优化指标
        
        Args:
            optimization_id: 优化任务ID
            
        Returns:
            List[OptimizationMetrics]: 优化指标列表
        """
        if not self._running:
            raise OptimizationError("聚合器未在运行")
        
        try:
            # 从统计器获取指标
            metrics = await self.statistics.get_optimization_history(optimization_id)
            return metrics
            
        except Exception as e:
            self.logger.error(f"获取优化指标失败 {optimization_id}: {e}")
            raise OptimizationError(f"获取优化指标失败: {e}")
    
    async def get_optimization_health(self, optimization_id: str) -> Optional[OptimizationHealthReport]:
        """获取优化健康状态
        
        Args:
            optimization_id: 优化任务ID
            
        Returns:
            Optional[OptimizationHealthReport]: 健康报告对象
        """
        if not self._running:
            raise OptimizationError("聚合器未在运行")
        
        try:
            # 执行健康检查
            health_report = await self.health_checker.check_optimization_health(optimization_id)
            return health_report
            
        except Exception as e:
            self.logger.error(f"获取优化健康状态失败 {optimization_id}: {e}")
            raise OptimizationError(f"获取优化健康状态失败: {e}")
    
    # ========== 统计和分析接口 ==========
    
    async def get_global_statistics(self) -> Dict[str, Any]:
        """获取全局统计信息
        
        Returns:
            Dict[str, Any]: 全局统计信息
        """
        if not self._running:
            raise OptimizationError("聚合器未在运行")
        
        try:
            # 从统计器获取全局统计
            stats = await self.statistics.get_global_statistics()
            return stats
            
        except Exception as e:
            self.logger.error(f"获取全局统计信息失败: {e}")
            raise OptimizationError(f"获取全局统计信息失败: {e}")
    
    async def get_performance_trends(self, time_range: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """获取性能趋势
        
        Args:
            time_range: 时间范围
            
        Returns:
            Dict[str, Any]: 性能趋势数据
        """
        if not self._running:
            raise OptimizationError("聚合器未在运行")
        
        try:
            # 从统计器获取趋势数据
            trends = await self.statistics.get_performance_trends(time_range)
            return trends
            
        except Exception as e:
            self.logger.error(f"获取性能趋势失败: {e}")
            raise OptimizationError(f"获取性能趋势失败: {e}")
    
    async def get_optimization_rankings(self, metric_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """获取优化排名
        
        Args:
            metric_name: 指标名称
            limit: 返回数量限制
            
        Returns:
            List[Dict[str, Any]]: 排名数据
        """
        if not self._running:
            raise OptimizationError("聚合器未在运行")
        
        try:
            # 从统计器获取排名
            rankings = await self.statistics.get_optimization_rankings(metric_name, limit)
            return rankings
            
        except Exception as e:
            self.logger.error(f"获取优化排名失败: {e}")
            raise OptimizationError(f"获取优化排名失败: {e}")
    
    # ========== 告警和通知接口 ==========
    
    async def send_alert(self, alert: OptimizationAlert) -> None:
        """发送告警
        
        Args:
            alert: 告警对象
        """
        if not self._running:
            raise OptimizationError("聚合器未在运行")
        
        try:
            # 发送到告警系统
            await self.alert_system.send_alert(alert)
            
        except Exception as e:
            self.logger.error(f"发送告警失败: {e}")
            raise OptimizationError(f"发送告警失败: {e}")
    
    async def get_alerts(self, optimization_id: Optional[str] = None, 
                        level: Optional[AlertLevel] = None) -> List[OptimizationAlert]:
        """获取告警列表
        
        Args:
            optimization_id: 优化任务ID（可选）
            level: 告警级别（可选）
            
        Returns:
            List[OptimizationAlert]: 告警列表
        """
        if not self._running:
            raise OptimizationError("聚合器未在运行")
        
        try:
            # 从监控器获取告警
            alerts = await self.monitor.get_alerts(optimization_id, level)
            return alerts
            
        except Exception as e:
            self.logger.error(f"获取告警列表失败: {e}")
            raise OptimizationError(f"获取告警列表失败: {e}")
    
    # ========== API接口 ==========
    
    async def handle_api_request(self, endpoint: str, method: str, 
                               data: Optional[Dict[str, Any]] = None,
                               client_id: Optional[str] = None) -> Dict[str, Any]:
        """处理API请求
        
        Args:
            endpoint: API端点
            method: HTTP方法
            data: 请求数据
            client_id: 客户端ID
            
        Returns:
            Dict[str, Any]: API响应
        """
        if not self._running:
            return {
                'success': False,
                'error': 'Service not running',
                'code': 503
            }
        
        try:
            # 委托给API组件处理
            response = await self.api.handle_api_request(endpoint, method, data, client_id)
            return response
            
        except Exception as e:
            self.logger.error(f"处理API请求失败: {e}")
            return {
                'success': False,
                'error': str(e),
                'code': 500
            }
    
    # ========== 分布式协调接口 ==========
    
    async def register_node(self, node_id: str, node_info: Dict[str, Any]) -> None:
        """注册分布式节点
        
        Args:
            node_id: 节点ID
            node_info: 节点信息
        """
        if not self._running:
            raise OptimizationError("聚合器未在运行")
        
        try:
            await self.synchronizer.register_node(node_id, node_info)
            
        except Exception as e:
            self.logger.error(f"注册节点失败 {node_id}: {e}")
            raise OptimizationError(f"注册节点失败: {e}")
    
    async def sync_optimization_state(self, optimization_id: str, 
                                    target_nodes: Optional[List[str]] = None) -> None:
        """同步优化状态
        
        Args:
            optimization_id: 优化任务ID
            target_nodes: 目标节点列表
        """
        if not self._running:
            raise OptimizationError("聚合器未在运行")
        
        try:
            await self.synchronizer.sync_optimization_state(optimization_id, target_nodes)
            
        except Exception as e:
            self.logger.error(f"同步优化状态失败 {optimization_id}: {e}")
            raise OptimizationError(f"同步优化状态失败: {e}")
    
    # ========== 内部方法 ==========
    
    def _register_event_listeners(self) -> None:
        """注册组件间事件监听"""
        # 监控器告警事件
        self.monitor.register_callback('alert', self._on_alert_received)
        
        # 协调器优化完成事件
        self.coordinator.register_callback('optimization_completed', self._on_optimization_completed)
        
        # 生命周期管理器事件
        self.lifecycle_manager.register_lifecycle_hook('on_create', self._on_optimization_created)
        self.lifecycle_manager.register_lifecycle_hook('after_completed', self._on_optimization_lifecycle_completed)
        
        # 告警系统事件
        self.alert_system.register_alert_rule(self._custom_alert_rule)
    
    async def _on_alert_received(self, alert: OptimizationAlert) -> None:
        """处理接收到的告警"""
        self.logger.info(f"收到告警: {alert.title}")
        
        # 可以在这里添加额外的告警处理逻辑
        # 例如：发送到外部系统、触发自动化响应等
    
    async def _on_optimization_completed(self, optimization: OptimizationState) -> None:
        """处理优化完成事件"""
        self.logger.info(f"优化任务完成: {optimization.optimization_id}")
        
        # 生成完成告警
        if optimization.status == OptimizationStatus.COMPLETED:
            completion_alert = OptimizationAlert(
                alert_id=str(uuid.uuid4()),
                optimization_id=optimization.optimization_id,
                level=AlertLevel.INFO,
                title="优化任务完成",
                message=f"优化任务 {optimization.name} 已成功完成",
                timestamp=datetime.now()
            )
            await self.send_alert(completion_alert)
        
        # 执行健康检查
        await self.get_optimization_health(optimization.optimization_id)
        
        # 记录统计指标
        metric = OptimizationMetrics(
            optimization_id=optimization.optimization_id,
            timestamp=datetime.now(),
            execution_time=optimization.execution_time,
            cpu_usage=optimization.cpu_usage,
            memory_usage=optimization.memory_usage,
            throughput=optimization.current_performance,
            convergence_rate=1.0,  # 优化完成，收敛率为1.0
            improvement_rate=optimization.performance_improvement,
            stability_score=0.9,  # 模拟稳定性评分
            accuracy=0.85,  # 模拟准确率
            precision=0.88,  # 模拟精确率
            recall=0.82,  # 模拟召回率
            f1_score=0.85,  # 模拟F1分数
            profit_improvement=optimization.performance_improvement * 100,
            risk_reduction=5.0,  # 模拟风险降低
            efficiency_gain=optimization.performance_improvement * 50
        )
        
        await self.statistics.record_optimization_metric(optimization.optimization_id, metric)
    
    async def _on_optimization_created(self, optimization: OptimizationState) -> None:
        """处理优化创建事件"""
        self.logger.info(f"创建优化任务: {optimization.optimization_id}")
        
        # 发送创建通知
        creation_alert = OptimizationAlert(
            alert_id=str(uuid.uuid4()),
            optimization_id=optimization.optimization_id,
            level=AlertLevel.INFO,
            title="优化任务创建",
            message=f"新的优化任务 {optimization.name} 已创建",
            timestamp=datetime.now()
        )
        await self.send_alert(creation_alert)
    
    async def _on_optimization_lifecycle_completed(self, optimization_id: str) -> None:
        """处理优化生命周期完成事件"""
        self.logger.info(f"优化生命周期完成: {optimization_id}")
        
        # 清理相关资源
        await self.monitor.stop_monitoring(optimization_id)
    
    async def _custom_alert_rule(self, alert: OptimizationAlert) -> bool:
        """自定义告警规则"""
        # 关键告警总是通过
        if alert.level == AlertLevel.CRITICAL:
            return True
        
        # 高优先级优化任务的告警优先处理
        # 这里可以添加更复杂的规则逻辑
        return True


# ==================== 使用示例 ====================

async def example_usage():
    """使用示例"""
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 创建聚合器实例
    aggregator = OptimizationStateAggregator(
        max_concurrent_optimizations=5,
        monitoring_interval=2.0,
        health_check_interval=60.0
    )
    
    try:
        # 初始化和启动
        await aggregator.initialize()
        await aggregator.start()
        
        print("O9优化状态聚合器已启动")
        
        # 创建优化任务
        optimization = OptimizationState(
            optimization_id="opt_example_001",
            name="机器学习模型参数优化",
            description="优化神经网络的超参数以提高性能",
            optimization_type=OptimizationType.PARAMETER_TUNING,
            status=OptimizationStatus.CREATED,
            priority=OptimizationPriority.HIGH,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            total_steps=10,
            baseline_performance=0.75,
            target_performance=0.90,
            parameters={
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100,
                "hidden_layers": [128, 64, 32]
            }
        )
        
        # 提交优化任务
        optimization_id = await aggregator.submit_optimization(optimization)
        print(f"提交优化任务: {optimization_id}")
        
        # 等待一段时间，观察优化过程
        await asyncio.sleep(5)
        
        # 获取优化状态
        status = await aggregator.get_optimization_status(optimization_id)
        if status:
            print(f"优化状态: {status.status.value}, 进度: {status.progress:.2%}")
        
        # 获取健康状态
        health = await aggregator.get_optimization_health(optimization_id)
        if health:
            print(f"健康状态: {health.overall_status.value}")
        
        # 获取统计信息
        stats = await aggregator.get_global_statistics()
        print(f"全局统计: {stats}")
        
        # 等待优化完成
        print("等待优化任务完成...")
        for i in range(30):  # 最多等待30秒
            status = await aggregator.get_optimization_status(optimization_id)
            if status and status.status in [OptimizationStatus.COMPLETED, 
                                          OptimizationStatus.FAILED, 
                                          OptimizationStatus.CANCELLED]:
                break
            await asyncio.sleep(1)
        
        # 获取最终状态
        final_status = await aggregator.get_optimization_status(optimization_id)
        if final_status:
            print(f"最终状态: {final_status.status.value}")
            print(f"最终性能: {final_status.current_performance:.3f}")
            print(f"性能改进: {final_status.performance_improvement:.3f}")
        
        # 获取告警
        alerts = await aggregator.get_alerts(optimization_id)
        print(f"收到告警数量: {len(alerts)}")
        
        # API调用示例
        api_response = await aggregator.handle_api_request(
            '/statistics', 'GET'
        )
        print(f"API响应: {api_response}")
        
    except Exception as e:
        print(f"示例执行失败: {e}")
        traceback.print_exc()
    
    finally:
        # 清理资源
        await aggregator.stop()
        await aggregator.cleanup()
        print("O9优化状态聚合器已关闭")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())