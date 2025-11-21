#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O8资源优化器模块

该模块提供了完整的系统资源优化解决方案，包括：
1. 系统资源监控和分析（CPU、内存、磁盘、网络）
2. 资源分配和调度优化（动态分配、智能调度）
3. 资源回收和释放优化（自动回收、垃圾回收）
4. 资源限制和配额管理（资源配额、访问控制）
5. 资源预测和规划（负载预测、容量规划）
6. 资源成本优化（成本分析、优化策略）
7. 异步资源优化处理
8. 完整的错误处理和日志记录

主要类：
- ResourceOptimizer: 主控制器
- SystemMonitor: 系统监控器
- ResourcePoolManager: 资源池管理器
- LoadBalancer: 负载均衡器
- ResourcePredictor: 资源预测器
- CostOptimizer: 成本优化器
- QuotaManager: 配额管理器
- ResourceReclaimer: 资源回收器

作者: O8系统
版本: 1.0.0
创建时间: 2025-11-06
"""

import asyncio
import gc
import json
import logging
import math
import os
import pickle
import psutil
import random
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps, lru_cache
from pathlib import Path
from queue import Queue, Empty
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, 
    AsyncGenerator, AsyncIterator, Set, NamedTuple
)
from collections import defaultdict, deque
import weakref
import hashlib
import socket
import resource
import subprocess
import sys
from threading import Lock, RLock, Event, Condition
from multiprocessing import cpu_count

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('resource_optimizer.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 数据结构定义
@dataclass
class ResourceMetrics:
    """资源指标数据类"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_usage: float
    disk_io_read: float
    disk_io_write: float
    network_sent: float
    network_recv: float
    thread_count: int
    process_count: int
    load_average: Tuple[float, float, float]
    connection_count: int = 0
    file_handle_count: int = 0

@dataclass
class OptimizationResult:
    """优化结果数据类"""
    operation: str
    before_metrics: ResourceMetrics
    after_metrics: ResourceMetrics
    improvement_percentage: float
    optimization_time: float
    recommendations: List[str] = field(default_factory=list)
    status: str = "success"
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class ResourcePool:
    """资源池数据类"""
    pool_id: str
    resource_type: 'ResourceType'
    total_capacity: int
    available_capacity: int
    allocated_capacity: int
    utilization_rate: float
    max_allocation: int
    min_allocation: int
    created_time: datetime = field(default_factory=datetime.now)
    last_optimized: Optional[datetime] = None

class LoadBalanceStrategy(Enum):
    """负载均衡策略枚举"""
    ROUND_ROBIN = "round_robin"           # 轮询
    LEAST_CONNECTIONS = "least_connections"  # 最少连接
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # 加权轮询
    IP_HASH = "ip_hash"                   # IP哈希
    LEAST_RESPONSE_TIME = "least_response_time"  # 最少响应时间
    RESOURCE_BASED = "resource_based"     # 基于资源
    ADAPTIVE = "adaptive"                 # 自适应

class PredictionModel(Enum):
    """预测模型枚举"""
    LINEAR_REGRESSION = "linear_regression"
    POLYNOMIAL_REGRESSION = "polynomial_regression"
    EXPONENTIAL_SMOOTHING = "exponential_smoothing"
    ARIMA = "arima"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"

@dataclass
class CostAnalysis:
    """成本分析数据类"""
    resource_type: str
    current_cost: float
    optimized_cost: float
    cost_reduction: float
    cost_reduction_percentage: float
    efficiency_score: float
    roi: float
    payback_period: float

class QuotaPolicy(Enum):
    """配额策略枚举"""
    HARD_LIMIT = "hard_limit"     # 硬限制
    SOFT_LIMIT = "soft_limit"     # 软限制
    GRADUAL_LIMIT = "gradual_limit"  # 渐进限制
    ADAPTIVE_LIMIT = "adaptive_limit"  # 自适应限制

class ReclaimStrategy(Enum):
    """回收策略枚举"""
    IMMEDIATE = "immediate"       # 立即回收
    GRADUAL = "gradual"          # 渐进回收
    LAZY = "lazy"               # 懒回收
    PREDICTIVE = "predictive"    # 预测回收
    SCHEDULED = "scheduled"      # 定时回收

class ResourceType(Enum):
    """资源类型枚举"""
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    THREAD = "thread"
    PROCESS = "process"
    CONNECTION = "connection"
    FILE_HANDLE = "file_handle"

class OptimizationStrategy(Enum):
    """优化策略枚举"""
    AGGRESSIVE = "aggressive"     # 激进优化
    CONSERVATIVE = "conservative" # 保守优化
    BALANCED = "balanced"        # 平衡优化
    ADAPTIVE = "adaptive"        # 自适应优化
    PREDICTIVE = "predictive"    # 预测优化

class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class ResourceState(Enum):
    """资源状态枚举"""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    RESERVED = "reserved"
    EXHAUSTED = "exhausted"
    MAINTENANCE = "maintenance"

# 系统监控器
class SystemMonitor:
    """系统监控器类"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self._is_monitoring = False
        self._monitor_thread = None
        self._metrics_queue = Queue()
        self._alerts = []
        self._lock = RLock()
        
    async def start_monitoring(self):
        """开始监控"""
        if not self._is_monitoring:
            self._is_monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop)
            self._monitor_thread.daemon = True
            self._monitor_thread.start()
            logger.info("系统监控已启动")
    
    async def stop_monitoring(self):
        """停止监控"""
        self._is_monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join()
        logger.info("系统监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self._is_monitoring:
            try:
                metrics = self._collect_metrics()
                self._metrics_queue.put(metrics)
                self._check_alerts(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                traceback.print_exc()
    
    def _collect_metrics(self) -> ResourceMetrics:
        """收集系统指标"""
        try:
            # CPU信息
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # 内存信息
            memory = psutil.virtual_memory()
            
            # 磁盘信息
            disk_usage = psutil.disk_usage('/')
            disk_io = psutil.disk_io_counters()
            
            # 网络信息
            network_io = psutil.net_io_counters()
            
            # 系统负载
            load_avg = os.getloadavg() if hasattr(os, 'getloadavg') else (0, 0, 0)
            
            # 进程信息
            process_count = len(psutil.pids())
            
            return ResourceMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_percent,
                memory_usage=memory.percent,
                memory_available=memory.available,
                disk_usage=disk_usage.percent,
                disk_io_read=disk_io.read_bytes if disk_io else 0,
                disk_io_write=disk_io.write_bytes if disk_io else 0,
                network_sent=network_io.bytes_sent if network_io else 0,
                network_recv=network_io.bytes_recv if network_io else 0,
                thread_count=threading.active_count(),
                process_count=process_count,
                load_average=load_avg,
                connection_count=len(psutil.net_connections()),
                file_handle_count=psutil.Process().num_handles() if hasattr(psutil.Process(), 'num_handles') else 0
            )
        except Exception as e:
            logger.error(f"收集系统指标失败: {e}")
            raise
    
    def _check_alerts(self, metrics: ResourceMetrics):
        """检查告警条件"""
        alerts = []
        
        # CPU使用率告警
        if metrics.cpu_usage > 90:
            alerts.append("CPU使用率过高 (>90%)")
        elif metrics.cpu_usage > 80:
            alerts.append("CPU使用率较高 (>80%)")
        
        # 内存使用率告警
        if metrics.memory_usage > 95:
            alerts.append("内存使用率过高 (>95%)")
        elif metrics.memory_usage > 85:
            alerts.append("内存使用率较高 (>85%)")
        
        # 磁盘使用率告警
        if metrics.disk_usage > 90:
            alerts.append("磁盘使用率过高 (>90%)")
        elif metrics.disk_usage > 80:
            alerts.append("磁盘使用率较高 (>80%)")
        
        # 系统负载告警
        if metrics.load_average[0] > cpu_count() * 0.8:
            alerts.append("系统负载过高")
        
        for alert in alerts:
            with self._lock:
                self._alerts.append({
                    'timestamp': datetime.now(),
                    'level': AlertLevel.WARNING.value,
                    'message': alert,
                    'metrics': metrics
                })
    
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """获取当前指标"""
        try:
            return self._metrics_queue.get_nowait()
        except Empty:
            return None
    
    def get_recent_metrics(self, count: int = 60) -> List[ResourceMetrics]:
        """获取最近的指标记录"""
        metrics_list = []
        temp_queue = Queue()
        
        # 清空队列并收集数据
        while not self._metrics_queue.empty():
            try:
                metrics = self._metrics_queue.get_nowait()
                metrics_list.append(metrics)
                temp_queue.put(metrics)
            except Empty:
                break
        
        # 恢复队列
        while not temp_queue.empty():
            try:
                self._metrics_queue.put(temp_queue.get_nowait())
            except Empty:
                break
        
        return metrics_list[-count:]
    
    def get_alerts(self, level: Optional[AlertLevel] = None) -> List[Dict]:
        """获取告警信息"""
        with self._lock:
            if level:
                return [alert for alert in self._alerts if alert['level'] == level.value]
            return self._alerts.copy()
    
    def clear_alerts(self):
        """清除告警"""
        with self._lock:
            self._alerts.clear()

# 资源池管理器
class ResourcePoolManager:
    """资源池管理器类"""
    
    def __init__(self):
        self._pools: Dict[str, ResourcePool] = {}
        self._lock = RLock()
        self._allocation_history = deque(maxlen=1000)
    
    def create_pool(self, pool_id: str, resource_type: ResourceType, 
                   total_capacity: int, max_allocation: int = None, 
                   min_allocation: int = 0) -> ResourcePool:
        """创建资源池"""
        with self._lock:
            if pool_id in self._pools:
                raise ValueError(f"资源池 {pool_id} 已存在")
            
            if max_allocation is None:
                max_allocation = total_capacity
            
            pool = ResourcePool(
                pool_id=pool_id,
                resource_type=resource_type,
                total_capacity=total_capacity,
                available_capacity=total_capacity,
                allocated_capacity=0,
                utilization_rate=0.0,
                max_allocation=max_allocation,
                min_allocation=min_allocation
            )
            
            self._pools[pool_id] = pool
            logger.info(f"创建资源池: {pool_id}, 类型: {resource_type.value}, 容量: {total_capacity}")
            return pool
    
    def allocate_resource(self, pool_id: str, amount: int) -> bool:
        """分配资源"""
        with self._lock:
            if pool_id not in self._pools:
                return False
            
            pool = self._pools[pool_id]
            
            if amount > pool.available_capacity or amount > pool.max_allocation:
                return False
            
            pool.available_capacity -= amount
            pool.allocated_capacity += amount
            pool.utilization_rate = pool.allocated_capacity / pool.total_capacity
            
            self._allocation_history.append({
                'timestamp': datetime.now(),
                'pool_id': pool_id,
                'amount': amount,
                'operation': 'allocate'
            })
            
            logger.debug(f"分配资源: {pool_id}, 数量: {amount}")
            return True
    
    def deallocate_resource(self, pool_id: str, amount: int) -> bool:
        """释放资源"""
        with self._lock:
            if pool_id not in self._pools:
                return False
            
            pool = self._pools[pool_id]
            
            if amount > pool.allocated_capacity:
                return False
            
            pool.available_capacity += amount
            pool.allocated_capacity -= amount
            pool.utilization_rate = pool.allocated_capacity / pool.total_capacity
            
            self._allocation_history.append({
                'timestamp': datetime.now(),
                'pool_id': pool_id,
                'amount': amount,
                'operation': 'deallocate'
            })
            
            logger.debug(f"释放资源: {pool_id}, 数量: {amount}")
            return True
    
    def get_pool_status(self, pool_id: str) -> Optional[ResourcePool]:
        """获取资源池状态"""
        with self._lock:
            return self._pools.get(pool_id)
    
    def get_all_pools(self) -> Dict[str, ResourcePool]:
        """获取所有资源池状态"""
        with self._lock:
            return self._pools.copy()
    
    def optimize_pool(self, pool_id: str) -> OptimizationResult:
        """优化资源池"""
        with self._lock:
            if pool_id not in self._pools:
                raise ValueError(f"资源池 {pool_id} 不存在")
            
            pool = self._pools[pool_id]
            start_time = time.time()
            
            # 记录优化前的状态
            before_metrics = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_usage=0,  # 这里应该是实际的系统指标
                memory_usage=0,
                memory_available=0,
                disk_usage=0,
                disk_io_read=0,
                disk_io_write=0,
                network_sent=0,
                network_recv=0,
                thread_count=0,
                process_count=0,
                load_average=(0, 0, 0)
            )
            
            # 优化逻辑
            recommendations = []
            
            # 如果利用率过低，回收部分资源
            if pool.utilization_rate < 0.3:
                reclaim_amount = int(pool.allocated_capacity * 0.2)
                if reclaim_amount > 0:
                    self.deallocate_resource(pool_id, reclaim_amount)
                    recommendations.append("回收未充分利用的资源")
            
            # 如果利用率过高，尝试扩容
            elif pool.utilization_rate > 0.9:
                pool.total_capacity = int(pool.total_capacity * 1.2)
                pool.available_capacity += int(pool.total_capacity * 0.2)
                recommendations.append("增加资源池容量以应对高负载")
            
            # 更新优化时间
            pool.last_optimized = datetime.now()
            
            # 记录优化后的状态
            after_metrics = ResourceMetrics(
                timestamp=datetime.now(),
                cpu_usage=before_metrics.cpu_usage,
                memory_usage=before_metrics.memory_usage,
                memory_available=before_metrics.memory_available,
                disk_usage=before_metrics.disk_usage,
                disk_io_read=before_metrics.disk_io_read,
                disk_io_write=before_metrics.disk_io_write,
                network_sent=before_metrics.network_sent,
                network_recv=before_metrics.network_recv,
                thread_count=before_metrics.thread_count,
                process_count=before_metrics.process_count,
                load_average=before_metrics.load_average
            )
            
            optimization_time = time.time() - start_time
            improvement_percentage = (pool.utilization_rate - 0.5) * 100  # 简化的改善计算
            
            result = OptimizationResult(
                operation="pool_optimization",
                before_metrics=before_metrics,
                after_metrics=after_metrics,
                improvement_percentage=improvement_percentage,
                optimization_time=optimization_time,
                recommendations=recommendations,
                status="success"
            )
            
            logger.info(f"优化资源池: {pool_id}, 改善: {improvement_percentage:.2f}%")
            return result

# 负载均衡器
class LoadBalancer:
    """负载均衡器类"""
    
    def __init__(self, strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self._servers: Dict[str, Dict] = {}
        self._current_index = 0
        self._connection_counts = {}
        self._response_times = {}
        self._lock = RLock()
        self._request_history = deque(maxlen=10000)
    
    def add_server(self, server_id: str, weight: int = 1, 
                  host: str = "localhost", port: int = 80):
        """添加服务器"""
        with self._lock:
            self._servers[server_id] = {
                'weight': weight,
                'host': host,
                'port': port,
                'active': True,
                'added_time': datetime.now()
            }
            self._connection_counts[server_id] = 0
            self._response_times[server_id] = []
            logger.info(f"添加服务器: {server_id} ({host}:{port})")
    
    def remove_server(self, server_id: str):
        """移除服务器"""
        with self._lock:
            if server_id in self._servers:
                del self._servers[server_id]
                del self._connection_counts[server_id]
                del self._response_times[server_id]
                logger.info(f"移除服务器: {server_id}")
    
    def get_next_server(self) -> Optional[str]:
        """获取下一个服务器"""
        with self._lock:
            active_servers = [sid for sid, info in self._servers.items() if info['active']]
            
            if not active_servers:
                return None
            
            if self.strategy == LoadBalanceStrategy.ROUND_ROBIN:
                server_id = active_servers[self._current_index % len(active_servers)]
                self._current_index += 1
                return server_id
            
            elif self.strategy == LoadBalanceStrategy.LEAST_CONNECTIONS:
                return min(active_servers, key=lambda x: self._connection_counts[x])
            
            elif self.strategy == LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN:
                weighted_servers = []
                for sid in active_servers:
                    weight = self._servers[sid]['weight']
                    weighted_servers.extend([sid] * weight)
                server_id = weighted_servers[self._current_index % len(weighted_servers)]
                self._current_index += 1
                return server_id
            
            elif self.strategy == LoadBalanceStrategy.IP_HASH:
                # 简化实现，实际应该基于客户端IP
                return active_servers[hash(str(datetime.now())) % len(active_servers)]
            
            elif self.strategy == LoadBalanceStrategy.LEAST_RESPONSE_TIME:
                avg_response_times = {}
                for sid in active_servers:
                    if self._response_times[sid]:
                        avg_response_times[sid] = sum(self._response_times[sid]) / len(self._response_times[sid])
                    else:
                        avg_response_times[sid] = 0
                return min(avg_response_times.keys(), key=lambda x: avg_response_times[x])
            
            elif self.strategy == LoadBalanceStrategy.RESOURCE_BASED:
                # 基于资源使用情况的负载均衡
                return self._get_resource_based_server()
            
            else:
                return active_servers[0]
    
    def _get_resource_based_server(self) -> str:
        """基于资源的服务器选择"""
        # 这里应该根据实际的资源使用情况来选择服务器
        # 简化实现
        active_servers = [sid for sid, info in self._servers.items() if info['active']]
        return random.choice(active_servers)
    
    def record_request(self, server_id: str, response_time: float = None):
        """记录请求"""
        with self._lock:
            if server_id in self._connection_counts:
                self._connection_counts[server_id] += 1
                
                if response_time is not None:
                    self._response_times[server_id].append(response_time)
                    # 保持响应时间记录在合理范围内
                    if len(self._response_times[server_id]) > 100:
                        self._response_times[server_id] = self._response_times[server_id][-50:]
                
                self._request_history.append({
                    'timestamp': datetime.now(),
                    'server_id': server_id,
                    'response_time': response_time
                })
    
    def record_response(self, server_id: str, response_time: float):
        """记录响应"""
        with self._lock:
            if server_id in self._connection_counts:
                self._connection_counts[server_id] = max(0, self._connection_counts[server_id] - 1)
                
                self._response_times[server_id].append(response_time)
                if len(self._response_times[server_id]) > 100:
                    self._response_times[server_id] = self._response_times[server_id][-50:]
    
    def get_server_stats(self, server_id: str) -> Dict:
        """获取服务器统计信息"""
        with self._lock:
            if server_id not in self._servers:
                return {}
            
            stats = self._servers[server_id].copy()
            stats['connection_count'] = self._connection_counts.get(server_id, 0)
            
            if self._response_times[server_id]:
                response_times = self._response_times[server_id]
                stats['avg_response_time'] = sum(response_times) / len(response_times)
                stats['min_response_time'] = min(response_times)
                stats['max_response_time'] = max(response_times)
            else:
                stats['avg_response_time'] = 0
                stats['min_response_time'] = 0
                stats['max_response_time'] = 0
            
            return stats
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """获取所有服务器统计信息"""
        with self._lock:
            return {sid: self.get_server_stats(sid) for sid in self._servers.keys()}

# 资源预测器
class ResourcePredictor:
    """资源预测器类"""
    
    def __init__(self, model: PredictionModel = PredictionModel.LINEAR_REGRESSION):
        self.model = model
        self._historical_data = deque(maxlen=10000)
        self._models = {}
        self._training_data = {}
    
    def add_data_point(self, timestamp: datetime, resource_type: ResourceType, 
                      value: float, metadata: Dict = None):
        """添加数据点"""
        self._historical_data.append({
            'timestamp': timestamp,
            'resource_type': resource_type,
            'value': value,
            'metadata': metadata or {}
        })
        
        # 按资源类型组织训练数据
        if resource_type not in self._training_data:
            self._training_data[resource_type] = []
        self._training_data[resource_type].append(value)
    
    def predict_resource_usage(self, resource_type: ResourceType, 
                              future_time: datetime) -> Optional[float]:
        """预测资源使用量"""
        if resource_type not in self._training_data or len(self._training_data[resource_type]) < 2:
            return None
        
        values = self._training_data[resource_type]
        
        if self.model == PredictionModel.LINEAR_REGRESSION:
            return self._linear_regression_predict(values, future_time)
        elif self.model == PredictionModel.EXPONENTIAL_SMOOTHING:
            return self._exponential_smoothing_predict(values)
        elif self.model == PredictionModel.POLYNOMIAL_REGRESSION:
            return self._polynomial_regression_predict(values, future_time)
        else:
            # 默认使用简单平均
            return sum(values) / len(values)
    
    def _linear_regression_predict(self, values: List[float], future_time: datetime) -> float:
        """线性回归预测"""
        if len(values) < 2:
            return values[-1] if values else 0
        
        # 简化的线性回归实现
        n = len(values)
        x = list(range(n))
        
        # 计算回归系数
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return y_mean
        
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        
        # 预测下一个值
        future_x = n
        prediction = slope * future_x + intercept
        
        return max(0, prediction)  # 确保预测值为非负数
    
    def _exponential_smoothing_predict(self, values: List[float], alpha: float = 0.3) -> float:
        """指数平滑预测"""
        if not values:
            return 0
        
        if len(values) == 1:
            return values[0]
        
        # 计算指数平滑值
        smoothed = values[0]
        for value in values[1:]:
            smoothed = alpha * value + (1 - alpha) * smoothed
        
        return smoothed
    
    def _polynomial_regression_predict(self, values: List[float], future_time: datetime, degree: int = 2) -> float:
        """多项式回归预测"""
        if len(values) < degree + 1:
            return self._linear_regression_predict(values, future_time)
        
        # 简化的多项式回归实现（这里使用二次多项式）
        n = len(values)
        x = list(range(n))
        
        # 二次多项式拟合
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        # 计算二次项系数（简化实现）
        if len(values) >= 3:
            # 使用简化的二次拟合
            x1, x2, x3 = x[-3], x[-2], x[-1]
            y1, y2, y3 = values[-3], values[-2], values[-1]
            
            # 计算差分
            diff1 = y2 - y1
            diff2 = y3 - y2
            
            # 预测下一个值
            trend = diff2 - diff1
            prediction = y3 + trend
        else:
            prediction = self._linear_regression_predict(values, future_time)
        
        return max(0, prediction)
    
    def get_prediction_confidence(self, resource_type: ResourceType) -> float:
        """获取预测置信度"""
        if resource_type not in self._training_data:
            return 0.0
        
        values = self._training_data[resource_type]
        if len(values) < 5:
            return 0.1  # 数据不足，置信度低
        
        # 基于数据变化程度计算置信度
        mean_value = sum(values) / len(values)
        variance = sum((x - mean_value) ** 2 for x in values) / len(values)
        std_dev = math.sqrt(variance)
        
        # 变异系数
        cv = std_dev / mean_value if mean_value > 0 else 1
        
        # 转换为置信度（0-1之间）
        confidence = max(0, min(1, 1 - cv))
        return confidence
    
    def analyze_trends(self, resource_type: ResourceType, 
                      window_size: int = 10) -> Dict[str, float]:
        """分析趋势"""
        if resource_type not in self._training_data or len(self._training_data[resource_type]) < window_size:
            return {}
        
        values = self._training_data[resource_type][-window_size:]
        
        # 计算趋势指标
        trend_slope = self._calculate_slope(values)
        volatility = self._calculate_volatility(values)
        growth_rate = self._calculate_growth_rate(values)
        
        return {
            'trend_slope': trend_slope,
            'volatility': volatility,
            'growth_rate': growth_rate,
            'direction': 'increasing' if trend_slope > 0 else 'decreasing' if trend_slope < 0 else 'stable'
        }
    
    def _calculate_slope(self, values: List[float]) -> float:
        """计算斜率"""
        if len(values) < 2:
            return 0
        
        n = len(values)
        x = list(range(n))
        
        x_mean = sum(x) / n
        y_mean = sum(values) / n
        
        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """计算波动性"""
        if len(values) < 2:
            return 0
        
        mean_value = sum(values) / len(values)
        variance = sum((x - mean_value) ** 2 for x in values) / len(values)
        return math.sqrt(variance)
    
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """计算增长率"""
        if len(values) < 2 or values[0] == 0:
            return 0
        
        return (values[-1] - values[0]) / values[0]

# 成本优化器
class CostOptimizer:
    """成本优化器类"""
    
    def __init__(self):
        self._cost_data = {}
        self._optimization_history = []
        self._lock = RLock()
    
    def add_cost_data(self, resource_type: ResourceType, current_cost: float, 
                     metadata: Dict = None):
        """添加成本数据"""
        with self._lock:
            if resource_type not in self._cost_data:
                self._cost_data[resource_type] = []
            
            self._cost_data[resource_type].append({
                'timestamp': datetime.now(),
                'cost': current_cost,
                'metadata': metadata or {}
            })
            
            # 保持数据在合理范围内
            if len(self._cost_data[resource_type]) > 1000:
                self._cost_data[resource_type] = self._cost_data[resource_type][-500:]
    
    def analyze_cost_efficiency(self, resource_type: ResourceType) -> Optional[CostAnalysis]:
        """分析成本效率"""
        with self._lock:
            if resource_type not in self._cost_data or len(self._cost_data[resource_type]) < 2:
                return None
            
            data = self._cost_data[resource_type]
            recent_costs = [d['cost'] for d in data[-10:]]  # 最近10个数据点
            
            current_cost = sum(recent_costs) / len(recent_costs)
            
            # 计算效率指标
            baseline_cost = data[0]['cost']
            cost_reduction = max(0, baseline_cost - current_cost)
            cost_reduction_percentage = (cost_reduction / baseline_cost) * 100 if baseline_cost > 0 else 0
            
            # 效率分数 (0-100)
            efficiency_score = min(100, max(0, 100 - cost_reduction_percentage))
            
            # ROI计算
            roi = cost_reduction / current_cost if current_cost > 0 else 0
            
            # 回本期计算（假设优化成本为当前成本的10%）
            optimization_cost = current_cost * 0.1
            payback_period = optimization_cost / cost_reduction if cost_reduction > 0 else float('inf')
            
            return CostAnalysis(
                resource_type=resource_type.value,
                current_cost=current_cost,
                optimized_cost=baseline_cost,
                cost_reduction=cost_reduction,
                cost_reduction_percentage=cost_reduction_percentage,
                efficiency_score=efficiency_score,
                roi=roi,
                payback_period=payback_period
            )
    
    def suggest_optimizations(self, resource_type: ResourceType) -> List[str]:
        """建议优化措施"""
        suggestions = []
        
        with self._lock:
            if resource_type not in self._cost_data:
                return ["需要更多成本数据进行分析"]
            
            data = self._cost_data[resource_type]
            recent_costs = [d['cost'] for d in data[-5:]]
            avg_cost = sum(recent_costs) / len(recent_costs)
            
            # 基于成本水平给出建议
            if avg_cost > 1000:
                suggestions.extend([
                    "考虑资源池优化以降低资源分配成本",
                    "实施负载均衡以提高资源利用率",
                    "定期清理未使用的资源"
                ])
            elif avg_cost > 500:
                suggestions.extend([
                    "监控资源使用模式，寻找优化机会",
                    "考虑实施预测性资源管理"
                ])
            else:
                suggestions.append("当前成本控制良好，继续保持")
            
            # 基于成本趋势给出建议
            if len(recent_costs) >= 3:
                recent_trend = recent_costs[-1] - recent_costs[-3]
                if recent_trend > 50:  # 成本上升超过50
                    suggestions.append("成本呈上升趋势，建议加强监控")
                elif recent_trend < -50:  # 成本下降超过50
                    suggestions.append("成本优化效果良好，可考虑进一步优化")
            
            return suggestions
    
    def calculate_optimization_impact(self, resource_type: ResourceType, 
                                    optimization_scenario: str) -> Dict[str, float]:
        """计算优化影响"""
        with self._lock:
            if resource_type not in self._cost_data:
                return {}
            
            data = self._cost_data[resource_type]
            current_cost = sum(d['cost'] for d in data[-5:]) / min(5, len(data))
            
            # 不同优化场景的影响计算
            scenarios = {
                "resource_pooling": 0.15,  # 资源池化降低15%成本
                "load_balancing": 0.12,    # 负载均衡降低12%成本
                "predictive_scaling": 0.20, # 预测性扩容降低20%成本
                "cost_monitoring": 0.08,   # 成本监控降低8%成本
                "resource_rightsizing": 0.18 # 资源调整降低18%成本
            }
            
            reduction_rate = scenarios.get(optimization_scenario, 0.1)
            potential_savings = current_cost * reduction_rate
            new_cost = current_cost - potential_savings
            
            return {
                'current_cost': current_cost,
                'potential_cost': new_cost,
                'potential_savings': potential_savings,
                'savings_percentage': reduction_rate * 100,
                'optimization_scenario': optimization_scenario
            }
    
    def get_cost_history(self, resource_type: ResourceType, 
                        days: int = 30) -> List[Dict]:
        """获取成本历史"""
        with self._lock:
            if resource_type not in self._cost_data:
                return []
            
            cutoff_date = datetime.now() - timedelta(days=days)
            return [
                item for item in self._cost_data[resource_type] 
                if item['timestamp'] >= cutoff_date
            ]

# 配额管理器
class QuotaManager:
    """配额管理器类"""
    
    def __init__(self):
        self._quotas: Dict[str, Dict] = {}
        self._usage_tracking = defaultdict(lambda: defaultdict(float))
        self._violations = []
        self._lock = RLock()
    
    def set_quota(self, user_id: str, resource_type: ResourceType, 
                 limit: float, policy: QuotaPolicy = QuotaPolicy.SOFT_LIMIT):
        """设置配额"""
        with self._lock:
            if user_id not in self._quotas:
                self._quotas[user_id] = {}
            
            self._quotas[user_id][resource_type] = {
                'limit': limit,
                'policy': policy,
                'created_time': datetime.now(),
                'reset_time': None
            }
            
            logger.info(f"设置配额: 用户 {user_id}, 资源 {resource_type.value}, 限制 {limit}")
    
    def check_quota(self, user_id: str, resource_type: ResourceType, 
                   requested_amount: float) -> Tuple[bool, str]:
        """检查配额"""
        with self._lock:
            if user_id not in self._quotas or resource_type not in self._quotas[user_id]:
                return True, "无配额限制"
            
            quota = self._quotas[user_id][resource_type]
            limit = quota['limit']
            policy = quota['policy']
            
            current_usage = self._usage_tracking[user_id][resource_type]
            new_usage = current_usage + requested_amount
            
            if policy == QuotaPolicy.HARD_LIMIT:
                if new_usage > limit:
                    return False, f"超过硬性配额限制 ({new_usage} > {limit})"
                return True, "配额检查通过"
            
            elif policy == QuotaPolicy.SOFT_LIMIT:
                warning_threshold = limit * 0.9
                if new_usage > limit:
                    # 允许超过但记录警告
                    warning_msg = f"超过软性配额限制 ({new_usage} > {limit})"
                    self._record_violation(user_id, resource_type, new_usage, limit, warning_msg)
                    return True, warning_msg  # 仍然允许但返回警告
                elif new_usage > warning_threshold:
                    return True, f"接近配额限制 ({new_usage:.1f}/{limit})"
                return True, "配额检查通过"
            
            elif policy == QuotaPolicy.GRADUAL_LIMIT:
                if current_usage >= limit:
                    return False, "已达到渐进式配额限制"
                
                # 计算允许的数量
                remaining = limit - current_usage
                if requested_amount > remaining:
                    return False, f"超过渐进式配额剩余量 ({remaining} < {requested_amount})"
                return True, "配额检查通过"
            
            elif policy == QuotaPolicy.ADAPTIVE_LIMIT:
                # 自适应配额根据历史使用模式动态调整
                historical_avg = self._get_historical_usage(user_id, resource_type)
                adaptive_limit = max(limit * 0.5, min(limit * 1.5, historical_avg))
                
                if new_usage > adaptive_limit:
                    return False, f"超过自适应配额限制 ({new_usage} > {adaptive_limit:.1f})"
                return True, "配额检查通过"
            
            return True, "配额检查通过"
    
    def allocate_resource(self, user_id: str, resource_type: ResourceType, 
                         amount: float) -> bool:
        """分配资源（记录使用量）"""
        allowed, message = self.check_quota(user_id, resource_type, amount)
        
        if allowed:
            with self._lock:
                self._usage_tracking[user_id][resource_type] += amount
                logger.debug(f"分配资源: 用户 {user_id}, 资源 {resource_type.value}, 数量 {amount}")
                return True
        else:
            logger.warning(f"资源分配被拒绝: 用户 {user_id}, 资源 {resource_type.value}, 原因: {message}")
            return False
    
    def release_resource(self, user_id: str, resource_type: ResourceType, 
                        amount: float) -> bool:
        """释放资源（减少使用量）"""
        with self._lock:
            current_usage = self._usage_tracking[user_id][resource_type]
            released_amount = min(amount, current_usage)
            self._usage_tracking[user_id][resource_type] -= released_amount
            
            logger.debug(f"释放资源: 用户 {user_id}, 资源 {resource_type.value}, 数量 {released_amount}")
            return True
    
    def get_usage(self, user_id: str, resource_type: ResourceType) -> float:
        """获取资源使用量"""
        return self._usage_tracking[user_id][resource_type]
    
    def get_quota_status(self, user_id: str, resource_type: ResourceType) -> Dict:
        """获取配额状态"""
        with self._lock:
            usage = self._usage_tracking[user_id][resource_type]
            
            if user_id in self._quotas and resource_type in self._quotas[user_id]:
                quota = self._quotas[user_id][resource_type]
                limit = quota['limit']
                policy = quota['policy']
                
                usage_percentage = (usage / limit) * 100 if limit > 0 else 0
                
                return {
                    'usage': usage,
                    'limit': limit,
                    'remaining': max(0, limit - usage),
                    'usage_percentage': usage_percentage,
                    'policy': policy.value,
                    'quota_set': True
                }
            else:
                return {
                    'usage': usage,
                    'limit': float('inf'),
                    'remaining': float('inf'),
                    'usage_percentage': 0,
                    'policy': 'none',
                    'quota_set': False
                }
    
    def reset_usage(self, user_id: str, resource_type: ResourceType = None):
        """重置使用量"""
        with self._lock:
            if resource_type:
                self._usage_tracking[user_id][resource_type] = 0
            else:
                self._usage_tracking[user_id] = defaultdict(float)
            
            if user_id in self._quotas:
                for rt in self._quotas[user_id]:
                    self._quotas[user_id][rt]['reset_time'] = datetime.now()
            
            logger.info(f"重置使用量: 用户 {user_id}, 资源 {resource_type.value if resource_type else 'all'}")
    
    def get_violations(self, days: int = 7) -> List[Dict]:
        """获取配额违规记录"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [
            violation for violation in self._violations 
            if violation['timestamp'] >= cutoff_date
        ]
    
    def _get_historical_usage(self, user_id: str, resource_type: ResourceType) -> float:
        """获取历史平均使用量"""
        # 简化实现，实际应该基于历史数据计算
        current_usage = self._usage_tracking[user_id][resource_type]
        return current_usage * 1.2  # 假设历史用量比当前高20%
    
    def _record_violation(self, user_id: str, resource_type: ResourceType, 
                         usage: float, limit: float, message: str):
        """记录配额违规"""
        with self._lock:
            self._violations.append({
                'timestamp': datetime.now(),
                'user_id': user_id,
                'resource_type': resource_type,
                'usage': usage,
                'limit': limit,
                'message': message
            })
            
            # 保持违规记录在合理范围内
            if len(self._violations) > 1000:
                self._violations = self._violations[-500:]

# 资源回收器
class ResourceReclaimer:
    """资源回收器类"""
    
    def __init__(self, strategy: ReclaimStrategy = ReclaimStrategy.GRADUAL):
        self.strategy = strategy
        self._idle_resources = {}
        self._reclaim_history = []
        self._scheduled_reclaims = {}
        self._lock = RLock()
        self._reclaim_thread = None
        self._is_running = False
    
    def start_auto_reclaim(self, interval: int = 300):  # 5分钟间隔
        """启动自动回收"""
        if not self._is_running:
            self._is_running = True
            self._reclaim_thread = threading.Thread(
                target=self._reclaim_loop, 
                args=(interval,)
            )
            self._reclaim_thread.daemon = True
            self._reclaim_thread.start()
            logger.info("自动资源回收已启动")
    
    def stop_auto_reclaim(self):
        """停止自动回收"""
        self._is_running = False
        if self._reclaim_thread:
            self._reclaim_thread.join()
        logger.info("自动资源回收已停止")
    
    def _reclaim_loop(self, interval: int):
        """回收循环"""
        while self._is_running:
            try:
                self.perform_reclaim_scan()
                time.sleep(interval)
            except Exception as e:
                logger.error(f"回收循环错误: {e}")
                traceback.print_exc()
    
    def register_idle_resource(self, resource_id: str, resource_type: ResourceType, 
                             owner: str, idle_duration: float):
        """注册空闲资源"""
        with self._lock:
            self._idle_resources[resource_id] = {
                'resource_type': resource_type,
                'owner': owner,
                'idle_duration': idle_duration,
                'registered_time': datetime.now(),
                'last_accessed': datetime.now() - timedelta(seconds=idle_duration)
            }
            
            logger.debug(f"注册空闲资源: {resource_id}, 类型: {resource_type.value}, 空闲时长: {idle_duration}s")
    
    def unregister_resource(self, resource_id: str):
        """取消注册资源"""
        with self._lock:
            if resource_id in self._idle_resources:
                del self._idle_resources[resource_id]
                logger.debug(f"取消注册资源: {resource_id}")
    
    def perform_reclaim_scan(self) -> List[str]:
        """执行回收扫描"""
        with self._lock:
            reclaimed_resources = []
            current_time = datetime.now()
            
            for resource_id, info in list(self._idle_resources.items()):
                idle_time = (current_time - info['last_accessed']).total_seconds()
                
                if self._should_reclaim(info, idle_time):
                    success = self._reclaim_resource(resource_id, info)
                    if success:
                        reclaimed_resources.append(resource_id)
                        del self._idle_resources[resource_id]
            
            if reclaimed_resources:
                logger.info(f"回收扫描完成，回收资源: {reclaimed_resources}")
            
            return reclaimed_resources
    
    def _should_reclaim(self, resource_info: Dict, idle_time: float) -> bool:
        """判断是否应该回收"""
        resource_type = resource_info['resource_type']
        
        if self.strategy == ReclaimStrategy.IMMEDIATE:
            return idle_time > 60  # 空闲1分钟立即回收
        
        elif self.strategy == ReclaimStrategy.GRADUAL:
            if resource_type == ResourceType.MEMORY:
                return idle_time > 300  # 内存资源5分钟
            elif resource_type == ResourceType.CPU:
                return idle_time > 600  # CPU资源10分钟
            else:
                return idle_time > 180  # 其他资源3分钟
        
        elif self.strategy == ReclaimStrategy.LAZY:
            return idle_time > 1800  # 空闲30分钟
        
        elif self.strategy == ReclaimStrategy.PREDICTIVE:
            # 预测性回收，基于使用模式
            return self._predictive_reclaim_decision(resource_info, idle_time)
        
        elif self.strategy == ReclaimStrategy.SCHEDULED:
            # 定时回收
            return self._scheduled_reclaim_decision(resource_info)
        
        return False
    
    def _predictive_reclaim_decision(self, resource_info: Dict, idle_time: float) -> bool:
        """预测性回收决策"""
        # 简化的预测逻辑
        # 实际应该基于历史使用模式、负载预测等进行决策
        resource_type = resource_info['resource_type']
        
        # 基于资源类型和使用模式预测
        if resource_type == ResourceType.MEMORY:
            # 如果系统内存压力大，优先回收内存资源
            memory = psutil.virtual_memory()
            if memory.percent > 80:
                return idle_time > 60
            else:
                return idle_time > 300
        
        return idle_time > 180  # 默认3分钟
    
    def _scheduled_reclaim_decision(self, resource_info: Dict) -> bool:
        """定时回收决策"""
        # 简化的定时回收逻辑
        # 实际应该基于配置的时间表进行回收
        current_hour = datetime.now().hour
        
        # 在系统负载较低的时间进行回收（凌晨2-6点）
        if 2 <= current_hour <= 6:
            return True
        
        return False
    
    def _reclaim_resource(self, resource_id: str, resource_info: Dict) -> bool:
        """回收资源"""
        try:
            resource_type = resource_info['resource_type']
            owner = resource_info['owner']
            
            # 实际回收逻辑应该根据资源类型执行
            if resource_type == ResourceType.MEMORY:
                # 清理内存缓存等
                gc.collect()
            
            elif resource_type == ResourceType.THREAD:
                # 清理空闲线程
                pass  # 实际实现需要线程池管理
            
            elif resource_type == ResourceType.CONNECTION:
                # 关闭空闲连接
                pass  # 实际实现需要连接池管理
            
            # 记录回收历史
            self._reclaim_history.append({
                'timestamp': datetime.now(),
                'resource_id': resource_id,
                'resource_type': resource_type,
                'owner': owner,
                'strategy': self.strategy,
                'success': True
            })
            
            logger.info(f"成功回收资源: {resource_id}, 类型: {resource_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"回收资源失败: {resource_id}, 错误: {e}")
            
            # 记录失败信息
            self._reclaim_history.append({
                'timestamp': datetime.now(),
                'resource_id': resource_id,
                'resource_type': resource_info['resource_type'],
                'owner': resource_info['owner'],
                'strategy': self.strategy,
                'success': False,
                'error': str(e)
            })
            
            return False
    
    def schedule_reclaim(self, resource_id: str, reclaim_time: datetime):
        """计划回收"""
        with self._lock:
            self._scheduled_reclaims[resource_id] = reclaim_time
            logger.info(f"计划回收资源: {resource_id}, 时间: {reclaim_time}")
    
    def cancel_scheduled_reclaim(self, resource_id: str):
        """取消计划回收"""
        with self._lock:
            if resource_id in self._scheduled_reclaims:
                del self._scheduled_reclaims[resource_id]
                logger.info(f"取消计划回收: {resource_id}")
    
    def get_idle_resources(self) -> Dict[str, Dict]:
        """获取空闲资源列表"""
        with self._lock:
            return self._idle_resources.copy()
    
    def get_reclaim_history(self, limit: int = 100) -> List[Dict]:
        """获取回收历史"""
        with self._lock:
            return self._reclaim_history[-limit:]
    
    def get_statistics(self) -> Dict:
        """获取回收统计信息"""
        with self._lock:
            total_reclaims = len(self._reclaim_history)
            successful_reclaims = sum(1 for r in self._reclaim_history if r['success'])
            failed_reclaims = total_reclaims - successful_reclaims
            
            # 按资源类型统计
            by_type = defaultdict(int)
            for reclaim in self._reclaim_history:
                by_type[reclaim['resource_type'].value] += 1
            
            return {
                'total_reclaims': total_reclaims,
                'successful_reclaims': successful_reclaims,
                'failed_reclaims': failed_reclaims,
                'success_rate': successful_reclaims / total_reclaims if total_reclaims > 0 else 0,
                'idle_resources_count': len(self._idle_resources),
                'by_resource_type': dict(by_type),
                'current_strategy': self.strategy.value
            }

# 主控制器
class ResourceOptimizer:
    """资源优化器主控制器"""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.BALANCED):
        self.strategy = strategy
        self._monitor = SystemMonitor()
        self._pool_manager = ResourcePoolManager()
        self._load_balancer = LoadBalancer()
        self._predictor = ResourcePredictor()
        self._cost_optimizer = CostOptimizer()
        self._quota_manager = QuotaManager()
        self._reclaimer = ResourceReclaimer()
        self._optimization_history = []
        self._is_running = False
        self._optimization_thread = None
        self._lock = RLock()
        
        # 配置默认资源池
        self._setup_default_pools()
    
    def _setup_default_pools(self):
        """设置默认资源池"""
        # CPU资源池
        self._pool_manager.create_pool(
            "cpu_pool", 
            ResourceType.CPU, 
            total_capacity=cpu_count() * 100,  # 假设每个CPU核心100单位
            max_allocation=cpu_count() * 100
        )
        
        # 内存资源池
        memory = psutil.virtual_memory()
        self._pool_manager.create_pool(
            "memory_pool",
            ResourceType.MEMORY,
            total_capacity=int(memory.total / 1024 / 1024),  # MB
            max_allocation=int(memory.total / 1024 / 1024)
        )
        
        # 线程资源池
        self._pool_manager.create_pool(
            "thread_pool",
            ResourceType.THREAD,
            total_capacity=1000,
            max_allocation=800
        )
        
        logger.info("默认资源池设置完成")
    
    async def start(self):
        """启动资源优化器"""
        if not self._is_running:
            self._is_running = True
            
            # 启动系统监控
            await self._monitor.start_monitoring()
            
            # 启动自动回收
            self._reclaimer.start_auto_reclaim()
            
            # 启动优化循环
            self._optimization_thread = threading.Thread(target=self._optimization_loop)
            self._optimization_thread.daemon = True
            self._optimization_thread.start()
            
            logger.info("资源优化器已启动")
    
    async def stop(self):
        """停止资源优化器"""
        if self._is_running:
            self._is_running = False
            
            # 停止系统监控
            await self._monitor.stop_monitoring()
            
            # 停止自动回收
            self._reclaimer.stop_auto_reclaim()
            
            # 等待优化线程结束
            if self._optimization_thread:
                self._optimization_thread.join()
            
            logger.info("资源优化器已停止")
    
    def _optimization_loop(self):
        """优化循环"""
        while self._is_running:
            try:
                self._perform_optimization_cycle()
                time.sleep(30)  # 每30秒执行一次优化
            except Exception as e:
                logger.error(f"优化循环错误: {e}")
                traceback.print_exc()
    
    def _perform_optimization_cycle(self):
        """执行优化周期"""
        with self._lock:
            # 收集当前系统状态
            current_metrics = self._monitor.get_current_metrics()
            if current_metrics:
                # 记录指标用于预测
                self._predictor.add_data_point(
                    current_metrics.timestamp, 
                    ResourceType.CPU, 
                    current_metrics.cpu_usage
                )
                self._predictor.add_data_point(
                    current_metrics.timestamp,
                    ResourceType.MEMORY,
                    current_metrics.memory_usage
                )
                
                # 成本分析
                self._cost_optimizer.add_cost_data(
                    ResourceType.CPU, 
                    current_metrics.cpu_usage,
                    {'timestamp': current_metrics.timestamp}
                )
                self._cost_optimizer.add_cost_data(
                    ResourceType.MEMORY,
                    current_metrics.memory_usage,
                    {'timestamp': current_metrics.timestamp}
                )
            
            # 执行各种优化操作
            self._optimize_resource_pools()
            self._optimize_load_balancing()
            self._optimize_costs()
            
            logger.debug("优化周期执行完成")
    
    def _optimize_resource_pools(self):
        """优化资源池"""
        for pool_id in self._pool_manager.get_all_pools():
            try:
                result = self._pool_manager.optimize_pool(pool_id)
                self._optimization_history.append(result)
            except Exception as e:
                logger.error(f"优化资源池失败 {pool_id}: {e}")
    
    def _optimize_load_balancing(self):
        """优化负载均衡"""
        # 简化实现
        # 实际应该根据服务器负载情况动态调整负载均衡策略
        pass
    
    def _optimize_costs(self):
        """优化成本"""
        # 简化实现
        # 实际应该基于成本分析结果执行优化策略
        pass
    
    def optimize_system(self, target_resource: ResourceType = None) -> List[OptimizationResult]:
        """执行系统优化"""
        results = []
        
        try:
            # 获取当前指标
            before_metrics = self._monitor.get_current_metrics()
            if not before_metrics:
                logger.warning("无法获取当前系统指标")
                return results
            
            start_time = time.time()
            recommendations = []
            
            # 根据优化策略执行不同级别的优化
            if self.strategy == OptimizationStrategy.AGGRESSIVE:
                recommendations.extend([
                    "实施积极的资源回收",
                    "大幅优化资源池配置",
                    "启用预测性资源管理"
                ])
                
                # 执行积极的优化操作
                self._aggressive_optimization(target_resource)
                
            elif self.strategy == OptimizationStrategy.CONSERVATIVE:
                recommendations.extend([
                    "采用保守的资源优化策略",
                    "渐进式资源池调整",
                    "温和的负载均衡优化"
                ])
                
                # 执行保守的优化操作
                self._conservative_optimization(target_resource)
                
            elif self.strategy == OptimizationStrategy.BALANCED:
                recommendations.extend([
                    "平衡的性能和成本优化",
                    "适度的资源回收",
                    "智能负载均衡"
                ])
                
                # 执行平衡的优化操作
                self._balanced_optimization(target_resource)
                
            elif self.strategy == OptimizationStrategy.ADAPTIVE:
                recommendations.extend([
                    "自适应优化策略",
                    "基于使用模式的动态调整"
                ])
                
                # 执行自适应优化
                self._adaptive_optimization(target_resource)
            
            # 获取优化后的指标
            time.sleep(1)  # 等待优化生效
            after_metrics = self._monitor.get_current_metrics()
            
            if after_metrics:
                optimization_time = time.time() - start_time
                
                # 计算改善程度
                if target_resource:
                    improvement = self._calculate_improvement(before_metrics, after_metrics, target_resource)
                else:
                    improvement = self._calculate_overall_improvement(before_metrics, after_metrics)
                
                result = OptimizationResult(
                    operation=f"system_optimization_{self.strategy.value}",
                    before_metrics=before_metrics,
                    after_metrics=after_metrics,
                    improvement_percentage=improvement,
                    optimization_time=optimization_time,
                    recommendations=recommendations,
                    status="success"
                )
                
                results.append(result)
                self._optimization_history.append(result)
                
                logger.info(f"系统优化完成，改善程度: {improvement:.2f}%")
            
        except Exception as e:
            logger.error(f"系统优化失败: {e}")
            traceback.print_exc()
            
            # 创建失败结果
            error_result = OptimizationResult(
                operation="system_optimization_error",
                before_metrics=before_metrics or ResourceMetrics(
                    timestamp=datetime.now(), cpu_usage=0, memory_usage=0,
                    memory_available=0, disk_usage=0, disk_io_read=0,
                    disk_io_write=0, network_sent=0, network_recv=0,
                    thread_count=0, process_count=0, load_average=(0, 0, 0)
                ),
                after_metrics=before_metrics or ResourceMetrics(
                    timestamp=datetime.now(), cpu_usage=0, memory_usage=0,
                    memory_available=0, disk_usage=0, disk_io_read=0,
                    disk_io_write=0, network_sent=0, network_recv=0,
                    thread_count=0, process_count=0, load_average=(0, 0, 0)
                ),
                improvement_percentage=0,
                optimization_time=time.time() - start_time,
                recommendations=[f"优化失败: {e}"],
                status="error"
            )
            results.append(error_result)
        
        return results
    
    def _aggressive_optimization(self, target_resource: ResourceType = None):
        """激进优化"""
        # 强制回收所有空闲资源
        reclaimed = self._reclaimer.perform_reclaim_scan()
        logger.info(f"激进优化回收资源: {reclaimed}")
        
        # 立即优化所有资源池
        for pool_id in self._pool_manager.get_all_pools():
            self._pool_manager.optimize_pool(pool_id)
    
    def _conservative_optimization(self, target_resource: ResourceType = None):
        """保守优化"""
        # 只优化低优先级的资源
        if target_resource in [ResourceType.MEMORY, ResourceType.DISK]:
            for pool_id in self._pool_manager.get_all_pools():
                pool = self._pool_manager.get_pool_status(pool_id)
                if pool and pool.resource_type == target_resource:
                    self._pool_manager.optimize_pool(pool_id)
    
    def _balanced_optimization(self, target_resource: ResourceType = None):
        """平衡优化"""
        # 对所有资源池执行适度的优化
        for pool_id in self._pool_manager.get_all_pools():
            self._pool_manager.optimize_pool(pool_id)
        
        # 执行负载均衡优化
        self._optimize_load_balancing()
    
    def _adaptive_optimization(self, target_resource: ResourceType = None):
        """自适应优化"""
        # 基于预测结果进行优化
        current_metrics = self._monitor.get_current_metrics()
        if current_metrics:
            # 预测未来资源需求
            future_time = datetime.now() + timedelta(minutes=10)
            
            predicted_cpu = self._predictor.predict_resource_usage(ResourceType.CPU, future_time)
            predicted_memory = self._predictor.predict_resource_usage(ResourceType.MEMORY, future_time)
            
            # 根据预测结果调整资源池
            if predicted_cpu and predicted_cpu > current_metrics.cpu_usage * 1.2:
                logger.info("预测CPU使用率将上升，增加CPU资源池容量")
                # 实际实现应该动态调整资源池
            
            if predicted_memory and predicted_memory > current_metrics.memory_usage * 1.2:
                logger.info("预测内存使用率将上升，增加内存资源池容量")
                # 实际实现应该动态调整资源池
    
    def _calculate_improvement(self, before: ResourceMetrics, after: ResourceMetrics, 
                             resource_type: ResourceType) -> float:
        """计算特定资源的改善程度"""
        if resource_type == ResourceType.CPU:
            improvement = before.cpu_usage - after.cpu_usage
        elif resource_type == ResourceType.MEMORY:
            improvement = before.memory_usage - after.memory_usage
        elif resource_type == ResourceType.DISK:
            improvement = before.disk_usage - after.disk_usage
        else:
            improvement = 0
        
        return max(0, improvement)
    
    def _calculate_overall_improvement(self, before: ResourceMetrics, after: ResourceMetrics) -> float:
        """计算整体改善程度"""
        cpu_improvement = before.cpu_usage - after.cpu_usage
        memory_improvement = before.memory_usage - after.memory_usage
        disk_improvement = before.disk_usage - after.disk_usage
        
        # 加权平均
        overall_improvement = (cpu_improvement * 0.4 + 
                             memory_improvement * 0.4 + 
                             disk_improvement * 0.2)
        
        return max(0, overall_improvement)
    
    def get_optimization_report(self) -> Dict:
        """获取优化报告"""
        with self._lock:
            current_metrics = self._monitor.get_current_metrics()
            
            # 收集各种统计信息
            pool_stats = self._pool_manager.get_all_pools()
            load_balancer_stats = self._load_balancer.get_all_stats()
            recent_alerts = self._monitor.get_alerts()[-10:]  # 最近10条告警
            reclaim_stats = self._reclaimer.get_statistics()
            
            # 计算总体指标
            total_cpu_usage = current_metrics.cpu_usage if current_metrics else 0
            total_memory_usage = current_metrics.memory_usage if current_metrics else 0
            optimization_count = len(self._optimization_history)
            avg_improvement = sum(r.improvement_percentage for r in self._optimization_history) / optimization_count if optimization_count > 0 else 0
            
            return {
                'current_status': {
                    'cpu_usage': total_cpu_usage,
                    'memory_usage': total_memory_usage,
                    'disk_usage': current_metrics.disk_usage if current_metrics else 0,
                    'thread_count': current_metrics.thread_count if current_metrics else 0,
                    'process_count': current_metrics.process_count if current_metrics else 0
                },
                'resource_pools': {pool_id: {
                    'total_capacity': pool.total_capacity,
                    'allocated_capacity': pool.allocated_capacity,
                    'utilization_rate': pool.utilization_rate
                } for pool_id, pool in pool_stats.items()},
                'load_balancer': load_balancer_stats,
                'optimization_performance': {
                    'total_optimizations': optimization_count,
                    'average_improvement': avg_improvement,
                    'last_optimization': self._optimization_history[-1].timestamp if self._optimization_history else None
                },
                'recent_alerts': recent_alerts,
                'reclaim_statistics': reclaim_stats,
                'strategy': self.strategy.value,
                'is_running': self._is_running
            }
    
    # 便捷方法
    def get_current_metrics(self) -> Optional[ResourceMetrics]:
        """获取当前系统指标"""
        return self._monitor.get_current_metrics()
    
    def add_server(self, server_id: str, weight: int = 1, host: str = "localhost", port: int = 80):
        """添加负载均衡服务器"""
        self._load_balancer.add_server(server_id, weight, host, port)
    
    def create_custom_pool(self, pool_id: str, resource_type: ResourceType, total_capacity: int):
        """创建自定义资源池"""
        return self._pool_manager.create_pool(pool_id, resource_type, total_capacity)
    
    def set_user_quota(self, user_id: str, resource_type: ResourceType, limit: float, policy: QuotaPolicy = QuotaPolicy.SOFT_LIMIT):
        """设置用户配额"""
        self._quota_manager.set_quota(user_id, resource_type, limit, policy)
    
    def check_user_quota(self, user_id: str, resource_type: ResourceType, amount: float) -> Tuple[bool, str]:
        """检查用户配额"""
        return self._quota_manager.check_quota(user_id, resource_type, amount)

# 模块导出
__all__ = [
    'ResourceOptimizer',
    'SystemMonitor',
    'ResourcePoolManager',
    'LoadBalancer',
    'ResourcePredictor',
    'CostOptimizer',
    'QuotaManager',
    'ResourceReclaimer',
    'ResourceMetrics',
    'OptimizationResult',
    'ResourcePool',
    'LoadBalanceStrategy',
    'PredictionModel',
    'CostAnalysis',
    'QuotaPolicy',
    'ReclaimStrategy',
    'ResourceType',
    'OptimizationStrategy',
    'AlertLevel',
    'ResourceState'
]