#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L4性能日志记录器模块

该模块提供了一个完整的性能监控和日志记录系统，支持：
- 系统性能监控（CPU、内存、磁盘、网络）
- 应用性能监控（响应时间、吞吐量、并发数）
- 数据库性能监控（查询时间、连接数、锁等待）
- 网络性能监控（延迟、带宽、丢包率）
- 性能阈值告警和通知
- 性能趋势分析和报告
- 异步性能日志处理

"""

import asyncio
import logging
import json
import time
import threading
import queue
import sqlite3
import psutil
import statistics
import datetime
import random
import re
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
from pathlib import Path
import warnings
import traceback
import socket
import subprocess
import os
import signal
import weakref
import gc
from contextlib import contextmanager
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


# =============================================================================
# 基础数据结构和枚举定义
# =============================================================================

class PerformanceMetric(Enum):
    """性能指标枚举"""
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_USAGE = "disk_usage"
    NETWORK_IO = "network_io"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    CONCURRENT_USERS = "concurrent_users"
    QUERY_TIME = "query_time"
    DATABASE_CONNECTIONS = "database_connections"
    LOCK_WAIT_TIME = "lock_wait_time"
    NETWORK_LATENCY = "network_latency"
    BANDWIDTH = "bandwidth"
    PACKET_LOSS = "packet_loss"


class AlertLevel(Enum):
    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class DatabaseType(Enum):
    """数据库类型枚举"""
    MYSQL = "mysql"
    POSTGRESQL = "postgresql"
    SQLITE = "sqlite"
    ORACLE = "oracle"
    MONGODB = "mongodb"


class NetworkProtocol(Enum):
    """网络协议枚举"""
    TCP = "tcp"
    UDP = "udp"
    HTTP = "http"
    HTTPS = "https"


@dataclass
class PerformanceData:
    """性能数据基础类"""
    metric: PerformanceMetric
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'metric': self.metric.value,
            'value': self.value,
            'timestamp': self.timestamp,
            'tags': self.tags,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceData':
        """从字典创建实例"""
        return cls(
            metric=PerformanceMetric(data['metric']),
            value=data['value'],
            timestamp=data['timestamp'],
            tags=data.get('tags', {}),
            metadata=data.get('metadata', {})
        )


@dataclass
class AlertRule:
    """告警规则配置"""
    metric: PerformanceMetric
    threshold: float
    comparison_operator: str  # '>', '<', '>=', '<=', '==', '!='
    level: AlertLevel
    enabled: bool = True
    consecutive_count: int = 1  # 连续多少次触发告警
    description: str = ""
    
    def evaluate(self, value: float) -> bool:
        """评估是否触发告警"""
        if not self.enabled:
            return False
        
        try:
            if self.comparison_operator == '>':
                return value > self.threshold
            elif self.comparison_operator == '<':
                return value < self.threshold
            elif self.comparison_operator == '>=':
                return value >= self.threshold
            elif self.comparison_operator == '<=':
                return value <= self.threshold
            elif self.comparison_operator == '==':
                return abs(value - self.threshold) < 1e-10
            elif self.comparison_operator == '!=':
                return abs(value - self.threshold) >= 1e-10
            else:
                logging.error(f"不支持的比较操作符: {self.comparison_operator}")
                return False
        except Exception as e:
            logging.error(f"评估告警规则时发生错误: {e}")
            return False


@dataclass
class Alert:
    """告警信息"""
    rule: AlertRule
    value: float
    timestamp: float
    message: str
    resolved: bool = False
    resolved_timestamp: Optional[float] = None


@dataclass
class TrendAnalysis:
    """趋势分析结果"""
    metric: PerformanceMetric
    time_range: Tuple[float, float]
    trend_direction: str  # 'increasing', 'decreasing', 'stable'
    trend_strength: float  # 0-1之间的值，表示趋势强度
    statistics: Dict[str, float]
    predictions: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'metric': self.metric.value,
            'time_range': self.time_range,
            'trend_direction': self.trend_direction,
            'trend_strength': self.trend_strength,
            'statistics': self.statistics,
            'predictions': self.predictions
        }


# =============================================================================
# 性能监控器基类
# =============================================================================

class BasePerformanceMonitor:
    """性能监控器基类"""
    
    def __init__(self, name: str, logger: Optional[logging.Logger] = None):
        """
        初始化性能监控器
        
        Args:
            name: 监控器名称
            logger: 日志记录器
        """
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self._running = False
        self._thread = None
        self._lock = threading.RLock()
        self._callbacks = []
        
    def start(self, interval: float = 1.0) -> None:
        """开始监控"""
        with self._lock:
            if self._running:
                self.logger.warning(f"监控器 {self.name} 已经在运行")
                return
            
            self._running = True
            self._thread = threading.Thread(
                target=self._monitor_loop,
                args=(interval,),
                daemon=True,
                name=f"Monitor-{self.name}"
            )
            self._thread.start()
            self.logger.info(f"监控器 {self.name} 已启动，监控间隔: {interval}秒")
    
    def stop(self) -> None:
        """停止监控"""
        with self._lock:
            if not self._running:
                return
            
            self._running = False
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=5.0)
            self.logger.info(f"监控器 {self.name} 已停止")
    
    def add_callback(self, callback: Callable[[PerformanceData], None]) -> None:
        """添加数据回调函数"""
        with self._lock:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[PerformanceData], None]) -> None:
        """移除数据回调函数"""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def _notify_callbacks(self, data: PerformanceData) -> None:
        """通知所有回调函数"""
        with self._lock:
            for callback in self._callbacks:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"回调函数执行错误: {e}")
    
    def _monitor_loop(self, interval: float) -> None:
        """监控循环"""
        while self._running:
            try:
                data = self.collect_data()
                if data:
                    self._notify_callbacks(data)
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"监控器 {self.name} 收集数据时发生错误: {e}")
                time.sleep(interval)
    
    def collect_data(self) -> Optional[PerformanceData]:
        """收集性能数据 - 子类需要实现此方法"""
        raise NotImplementedError("子类必须实现 collect_data 方法")
    
    def get_current_value(self) -> Optional[float]:
        """获取当前性能值"""
        try:
            data = self.collect_data()
            return data.value if data else None
        except Exception as e:
            self.logger.error(f"获取当前值时发生错误: {e}")
            return None


# =============================================================================
# 系统性能监控器
# =============================================================================

class SystemPerformanceMonitor(BasePerformanceMonitor):
    """系统性能监控器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        super().__init__("SystemPerformance", logger)
        self._last_disk_io = None
        self._last_network_io = None
        self._last_time = None
    
    def collect_data(self) -> Optional[PerformanceData]:
        """收集系统性能数据"""
        try:
            current_time = time.time()
            
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 磁盘使用率
            disk_usage = psutil.disk_usage('/')
            disk_percent = (disk_usage.used / disk_usage.total) * 100
            
            # 网络I/O
            network_io = psutil.net_io_counters()
            network_data = None
            
            if self._last_network_io and self._last_time:
                time_delta = current_time - self._last_time
                bytes_sent_per_sec = (network_io.bytes_sent - self._last_network_io.bytes_sent) / time_delta
                bytes_recv_per_sec = (network_io.bytes_recv - self._last_network_io.bytes_recv) / time_delta
                network_data = {
                    'bytes_sent_per_sec': bytes_sent_per_sec,
                    'bytes_recv_per_sec': bytes_recv_per_sec,
                    'packets_sent': network_io.packets_sent,
                    'packets_recv': network_io.packets_recv
                }
            
            self._last_network_io = network_io
            self._last_time = current_time
            
            # 返回CPU使用率作为主要指标
            return PerformanceData(
                metric=PerformanceMetric.CPU_USAGE,
                value=cpu_percent,
                timestamp=current_time,
                tags={'component': 'system'},
                metadata={
                    'memory_percent': memory_percent,
                    'disk_percent': disk_percent,
                    'network_io': network_data
                }
            )
            
        except Exception as e:
            self.logger.error(f"收集系统性能数据时发生错误: {e}")
            return None
    
    def get_memory_usage(self) -> Optional[float]:
        """获取内存使用率"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception as e:
            self.logger.error(f"获取内存使用率时发生错误: {e}")
            return None
    
    def get_disk_usage(self) -> Optional[float]:
        """获取磁盘使用率"""
        try:
            disk_usage = psutil.disk_usage('/')
            return (disk_usage.used / disk_usage.total) * 100
        except Exception as e:
            self.logger.error(f"获取磁盘使用率时发生错误: {e}")
            return None
    
    def get_network_stats(self) -> Optional[Dict[str, Any]]:
        """获取网络统计信息"""
        try:
            network_io = psutil.net_io_counters()
            return {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv,
                'errin': network_io.errin,
                'errout': network_io.errout,
                'dropin': network_io.dropin,
                'dropout': network_io.dropout
            }
        except Exception as e:
            self.logger.error(f"获取网络统计信息时发生错误: {e}")
            return None


# =============================================================================
# 应用性能监控器
# =============================================================================

class ApplicationPerformanceMonitor(BasePerformanceMonitor):
    """应用性能监控器"""
    
    def __init__(self, app_name: str = "Application", logger: Optional[logging.Logger] = None):
        super().__init__(f"AppPerformance-{app_name}", logger)
        self.app_name = app_name
        self._request_times = deque(maxlen=1000)
        self._request_count = 0
        self._start_time = time.time()
        self._lock_requests = threading.Lock()
    
    def record_request(self, response_time: float, status_code: int = 200, 
                      method: str = "GET", endpoint: str = "/") -> None:
        """
        记录请求性能数据
        
        Args:
            response_time: 响应时间（毫秒）
            status_code: HTTP状态码
            method: HTTP方法
            endpoint: 请求端点
        """
        with self._lock_requests:
            self._request_times.append({
                'timestamp': time.time(),
                'response_time': response_time,
                'status_code': status_code,
                'method': method,
                'endpoint': endpoint
            })
            self._request_count += 1
    
    def collect_data(self) -> Optional[PerformanceData]:
        """收集应用性能数据"""
        try:
            current_time = time.time()
            
            with self._lock_requests:
                if not self._request_times:
                    return None
                
                # 计算响应时间统计
                response_times = [req['response_time'] for req in self._request_times]
                avg_response_time = statistics.mean(response_times)
                median_response_time = statistics.median(response_times)
                p95_response_time = self._percentile(response_times, 95)
                p99_response_time = self._percentile(response_times, 99)
                
                # 计算吞吐量（请求/秒）
                time_window = current_time - self._start_time
                throughput = self._request_count / time_window if time_window > 0 else 0
                
                # 计算并发数（最近1秒内的请求数）
                recent_requests = [
                    req for req in self._request_times 
                    if current_time - req['timestamp'] <= 1.0
                ]
                concurrent_users = len(recent_requests)
                
                # 成功率
                successful_requests = [
                    req for req in self._request_times 
                    if 200 <= req['status_code'] < 400
                ]
                success_rate = (len(successful_requests) / len(self._request_times)) * 100
                
                metadata = {
                    'avg_response_time': avg_response_time,
                    'median_response_time': median_response_time,
                    'p95_response_time': p95_response_time,
                    'p99_response_time': p99_response_time,
                    'throughput': throughput,
                    'concurrent_users': concurrent_users,
                    'success_rate': success_rate,
                    'total_requests': self._request_count,
                    'time_window': time_window
                }
                
                return PerformanceData(
                    metric=PerformanceMetric.RESPONSE_TIME,
                    value=avg_response_time,
                    timestamp=current_time,
                    tags={
                        'component': 'application',
                        'app_name': self.app_name
                    },
                    metadata=metadata
                )
                
        except Exception as e:
            self.logger.error(f"收集应用性能数据时发生错误: {e}")
            return None
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def get_throughput(self) -> Optional[float]:
        """获取吞吐量"""
        try:
            current_time = time.time()
            time_window = current_time - self._start_time
            return self._request_count / time_window if time_window > 0 else 0
        except Exception as e:
            self.logger.error(f"获取吞吐量时发生错误: {e}")
            return None
    
    def get_success_rate(self) -> Optional[float]:
        """获取成功率"""
        try:
            with self._lock_requests:
                if not self._request_times:
                    return 100.0
                
                successful_requests = [
                    req for req in self._request_times 
                    if 200 <= req['status_code'] < 400
                ]
                return (len(successful_requests) / len(self._request_times)) * 100
        except Exception as e:
            self.logger.error(f"获取成功率时发生错误: {e}")
            return None


# =============================================================================
# 数据库性能监控器
# =============================================================================

class DatabasePerformanceMonitor(BasePerformanceMonitor):
    """数据库性能监控器"""
    
    def __init__(self, db_type: DatabaseType, connection_string: str, 
                 logger: Optional[logging.Logger] = None):
        super().__init__(f"DBPerformance-{db_type.value}", logger)
        self.db_type = db_type
        self.connection_string = connection_string
        self._query_times = deque(maxlen=1000)
        self._connection_pool = {}
        self._lock_queries = threading.Lock()
    
    def record_query(self, query: str, execution_time: float, 
                    rows_affected: int = 0, success: bool = True) -> None:
        """
        记录查询性能数据
        
        Args:
            query: SQL查询语句
            execution_time: 执行时间（毫秒）
            rows_affected: 影响的行数
            success: 是否成功
        """
        with self._lock_queries:
            self._query_times.append({
                'timestamp': time.time(),
                'query': query,
                'execution_time': execution_time,
                'rows_affected': rows_affected,
                'success': success
            })
    
    def collect_data(self) -> Optional[PerformanceData]:
        """收集数据库性能数据"""
        try:
            current_time = time.time()
            
            with self._lock_queries:
                if not self._query_times:
                    return None
                
                # 计算查询时间统计
                execution_times = [q['execution_time'] for q in self._query_times]
                avg_query_time = statistics.mean(execution_times)
                median_query_time = statistics.median(execution_times)
                p95_query_time = self._percentile(execution_times, 95)
                p99_query_time = self._percentile(execution_times, 99)
                
                # 计算连接数
                active_connections = self._get_active_connections()
                
                # 计算锁等待时间
                lock_wait_times = [
                    q['execution_time'] for q in self._query_times 
                    if q['execution_time'] > 1000  # 假设超过1秒的查询可能有锁等待
                ]
                avg_lock_wait = statistics.mean(lock_wait_times) if lock_wait_times else 0
                
                # 成功率
                successful_queries = [q for q in self._query_times if q['success']]
                success_rate = (len(successful_queries) / len(self._query_times)) * 100
                
                metadata = {
                    'avg_query_time': avg_query_time,
                    'median_query_time': median_query_time,
                    'p95_query_time': p95_query_time,
                    'p99_query_time': p99_query_time,
                    'active_connections': active_connections,
                    'avg_lock_wait_time': avg_lock_wait,
                    'success_rate': success_rate,
                    'total_queries': len(self._query_times)
                }
                
                return PerformanceData(
                    metric=PerformanceMetric.QUERY_TIME,
                    value=avg_query_time,
                    timestamp=current_time,
                    tags={
                        'component': 'database',
                        'db_type': self.db_type.value
                    },
                    metadata=metadata
                )
                
        except Exception as e:
            self.logger.error(f"收集数据库性能数据时发生错误: {e}")
            return None
    
    def _get_active_connections(self) -> int:
        """获取活跃连接数"""
        try:
            # 这里应该根据不同的数据库类型实现具体的连接数获取逻辑
            # 简化实现，返回模拟数据
            return len(self._connection_pool)
        except Exception as e:
            self.logger.error(f"获取活跃连接数时发生错误: {e}")
            return 0
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def get_connection_count(self) -> Optional[int]:
        """获取连接数"""
        return self._get_active_connections()
    
    def get_query_success_rate(self) -> Optional[float]:
        """获取查询成功率"""
        try:
            with self._lock_queries:
                if not self._query_times:
                    return 100.0
                
                successful_queries = [q for q in self._query_times if q['success']]
                return (len(successful_queries) / len(self._query_times)) * 100
        except Exception as e:
            self.logger.error(f"获取查询成功率时发生错误: {e}")
            return None


# =============================================================================
# 网络性能监控器
# =============================================================================

class NetworkPerformanceMonitor(BasePerformanceMonitor):
    """网络性能监控器"""
    
    def __init__(self, target_host: str = "8.8.8.8", port: int = 53, 
                 logger: Optional[logging.Logger] = None):
        super().__init__(f"NetworkPerformance-{target_host}", logger)
        self.target_host = target_host
        self.port = port
        self._latency_history = deque(maxlen=100)
        self._packet_loss_count = 0
        self._total_packets = 0
        self._lock_stats = threading.Lock()
    
    def collect_data(self) -> Optional[PerformanceData]:
        """收集网络性能数据"""
        try:
            current_time = time.time()
            
            # 测试延迟
            latency = self._measure_latency()
            
            # 测试丢包率
            packet_loss = self._measure_packet_loss()
            
            # 测试带宽（简化实现）
            bandwidth = self._estimate_bandwidth()
            
            with self._lock_stats:
                if latency is not None:
                    self._latency_history.append(latency)
                
                metadata = {
                    'latency': latency,
                    'packet_loss_rate': packet_loss,
                    'bandwidth_bps': bandwidth,
                    'avg_latency': statistics.mean(self._latency_history) if self._latency_history else None,
                    'min_latency': min(self._latency_history) if self._latency_history else None,
                    'max_latency': max(self._latency_history) if self._latency_history else None,
                    'packet_loss_count': self._packet_loss_count,
                    'total_packets': self._total_packets
                }
                
                return PerformanceData(
                    metric=PerformanceMetric.NETWORK_LATENCY,
                    value=latency or 0,
                    timestamp=current_time,
                    tags={
                        'component': 'network',
                        'target_host': self.target_host,
                        'target_port': self.port
                    },
                    metadata=metadata
                )
                
        except Exception as e:
            self.logger.error(f"收集网络性能数据时发生错误: {e}")
            return None
    
    def _measure_latency(self) -> Optional[float]:
        """测量网络延迟"""
        try:
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5.0)
            result = sock.connect_ex((self.target_host, self.port))
            sock.close()
            
            if result == 0:
                latency = (time.time() - start_time) * 1000  # 转换为毫秒
                return latency
            else:
                return None
                
        except Exception as e:
            self.logger.debug(f"测量延迟时发生错误: {e}")
            return None
    
    def _measure_packet_loss(self) -> float:
        """测量丢包率"""
        try:
            # 使用ping命令测量丢包率
            if os.name == 'nt':  # Windows
                cmd = ['ping', '-n', '4', self.target_host]
            else:  # Unix/Linux
                cmd = ['ping', '-c', '4', self.target_host]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # 解析ping结果
                output = result.stdout
                if '0% packet loss' in output:
                    return 0.0
                elif '100% packet loss' in output:
                    return 100.0
                else:
                    # 尝试提取丢包率
                    match = re.search(r'(\d+)% packet loss', output)
                    if match:
                        loss_rate = float(match.group(1))
                        with self._lock_stats:
                            self._packet_loss_count += 1
                            self._total_packets += 4
                        return loss_rate
            
            return 0.0
            
        except Exception as e:
            self.logger.debug(f"测量丢包率时发生错误: {e}")
            return 0.0
    
    def _estimate_bandwidth(self) -> Optional[float]:
        """估算带宽"""
        try:
            # 简化实现，实际应该通过传输测试数据来测量带宽
            # 这里返回一个基于延迟的估算值
            latency = self._measure_latency()
            if latency and latency > 0:
                # 简单的带宽估算（不准确，仅供参考）
                return 1000000 / latency  # 假设1Mbps基础带宽
            return None
        except Exception as e:
            self.logger.debug(f"估算带宽时发生错误: {e}")
            return None
    
    def get_average_latency(self) -> Optional[float]:
        """获取平均延迟"""
        with self._lock_stats:
            if self._latency_history:
                return statistics.mean(self._latency_history)
            return None
    
    def get_packet_loss_rate(self) -> float:
        """获取丢包率"""
        with self._lock_stats:
            if self._total_packets > 0:
                return (self._packet_loss_count / self._total_packets) * 100
            return 0.0


# =============================================================================
# 告警管理器
# =============================================================================

class AlertManager:
    """告警管理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化告警管理器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self._rules: List[AlertRule] = []
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._callbacks: List[Callable[[Alert], None]] = []
        self._lock = threading.RLock()
        
    def add_rule(self, rule: AlertRule) -> None:
        """添加告警规则"""
        with self._lock:
            self._rules.append(rule)
            self.logger.info(f"添加告警规则: {rule.metric.value} {rule.comparison_operator} {rule.threshold}")
    
    def remove_rule(self, rule: AlertRule) -> None:
        """移除告警规则"""
        with self._lock:
            if rule in self._rules:
                self._rules.remove(rule)
                self.logger.info(f"移除告警规则: {rule.metric.value}")
    
    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """添加告警回调函数"""
        with self._lock:
            self._callbacks.append(callback)
    
    def remove_callback(self, callback: Callable[[Alert], None]) -> None:
        """移除告警回调函数"""
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)
    
    def evaluate_data(self, data: PerformanceData) -> List[Alert]:
        """
        评估性能数据，生成告警
        
        Args:
            data: 性能数据
            
        Returns:
            生成的告警列表
        """
        alerts = []
        
        with self._lock:
            for rule in self._rules:
                if rule.metric == data.metric:
                    try:
                        if rule.evaluate(data.value):
                            alert_key = f"{rule.metric.value}_{rule.threshold}"
                            
                            # 检查是否已经存在相同的活跃告警
                            if alert_key in self._active_alerts:
                                # 更新现有告警
                                existing_alert = self._active_alerts[alert_key]
                                existing_alert.value = data.value
                                existing_alert.timestamp = data.timestamp
                            else:
                                # 创建新告警
                                alert = Alert(
                                    rule=rule,
                                    value=data.value,
                                    timestamp=data.timestamp,
                                    message=self._generate_alert_message(rule, data.value)
                                )
                                self._active_alerts[alert_key] = alert
                                alerts.append(alert)
                                self._alert_history.append(alert)
                                
                                # 限制告警历史记录数量
                                if len(self._alert_history) > 10000:
                                    self._alert_history = self._alert_history[-5000:]
                                
                                self.logger.warning(f"触发告警: {alert.message}")
                                
                    except Exception as e:
                        self.logger.error(f"评估告警规则时发生错误: {e}")
        
        # 通知回调函数
        for alert in alerts:
            self._notify_callbacks(alert)
        
        return alerts
    
    def resolve_alert(self, rule: AlertRule) -> bool:
        """
        解决告警
        
        Args:
            rule: 告警规则
            
        Returns:
            是否成功解决告警
        """
        with self._lock:
            alert_key = f"{rule.metric.value}_{rule.threshold}"
            if alert_key in self._active_alerts:
                alert = self._active_alerts[alert_key]
                alert.resolved = True
                alert.resolved_timestamp = time.time()
                del self._active_alerts[alert_key]
                self.logger.info(f"告警已解决: {rule.metric.value}")
                return True
            return False
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警列表"""
        with self._lock:
            return list(self._active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        with self._lock:
            return self._alert_history[-limit:]
    
    def _generate_alert_message(self, rule: AlertRule, value: float) -> str:
        """生成告警消息"""
        return (
            f"性能告警: {rule.metric.value} {rule.comparison_operator} {rule.threshold}, "
            f"当前值: {value:.2f}, 级别: {rule.level.value}, "
            f"描述: {rule.description}"
        )
    
    def _notify_callbacks(self, alert: Alert) -> None:
        """通知所有回调函数"""
        for callback in self._callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"告警回调函数执行错误: {e}")


# =============================================================================
# 趋势分析器
# =============================================================================

class TrendAnalyzer:
    """性能趋势分析器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化趋势分析器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self._data_history: Dict[PerformanceMetric, deque] = defaultdict(
            lambda: deque(maxlen=10000)
        )
        self._lock = threading.RLock()
    
    def add_data(self, data: PerformanceData) -> None:
        """
        添加性能数据
        
        Args:
            data: 性能数据
        """
        with self._lock:
            self._data_history[data.metric].append(data)
    
    def analyze_trend(self, metric: PerformanceMetric, 
                     time_range_hours: float = 24.0) -> Optional[TrendAnalysis]:
        """
        分析性能趋势
        
        Args:
            metric: 性能指标
            time_range_hours: 时间范围（小时）
            
        Returns:
            趋势分析结果
        """
        try:
            with self._lock:
                data_list = list(self._data_history[metric])
                
                if len(data_list) < 2:
                    return None
                
                # 过滤时间范围内的数据
                current_time = time.time()
                cutoff_time = current_time - (time_range_hours * 3600)
                filtered_data = [
                    d for d in data_list 
                    if d.timestamp >= cutoff_time
                ]
                
                if len(filtered_data) < 2:
                    return None
                
                # 提取数值和时间
                values = [d.value for d in filtered_data]
                timestamps = [d.timestamp for d in filtered_data]
                
                # 计算趋势
                trend_direction, trend_strength = self._calculate_trend(timestamps, values)
                
                # 计算统计信息
                statistics_data = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
                
                # 预测
                predictions = self._predict_future_values(values, timestamps)
                
                return TrendAnalysis(
                    metric=metric,
                    time_range=(cutoff_time, current_time),
                    trend_direction=trend_direction,
                    trend_strength=trend_strength,
                    statistics=statistics_data,
                    predictions=predictions
                )
                
        except Exception as e:
            self.logger.error(f"分析趋势时发生错误: {e}")
            return None
    
    def _calculate_trend(self, timestamps: List[float], 
                        values: List[float]) -> Tuple[str, float]:
        """
        计算趋势方向和强度
        
        Args:
            timestamps: 时间戳列表
            values: 数值列表
            
        Returns:
            (趋势方向, 趋势强度)
        """
        if len(values) < 2:
            return "stable", 0.0
        
        # 使用线性回归计算趋势
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        # 计算斜率
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        # 计算相关系数
        mean_y = sum_y / n
        ss_tot = sum((y - mean_y) ** 2 for y in values)
        ss_res = sum((values[i] - (slope * i + mean_y - slope * (n-1)/2)) ** 2 
                    for i in range(n))
        correlation = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # 确定趋势方向
        if slope > 0.1:
            trend_direction = "increasing"
        elif slope < -0.1:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        # 趋势强度为相关系数的绝对值
        trend_strength = abs(correlation)
        
        return trend_direction, trend_strength
    
    def _predict_future_values(self, values: List[float], 
                              timestamps: List[float]) -> Dict[str, float]:
        """
        预测未来值
        
        Args:
            values: 数值列表
            timestamps: 时间戳列表
            
        Returns:
            预测值字典
        """
        if len(values) < 3:
            return {}
        
        try:
            # 简单线性预测
            n = len(values)
            time_span = timestamps[-1] - timestamps[0]
            if time_span == 0:
                return {}
            
            # 计算平均变化率
            value_changes = [values[i] - values[i-1] for i in range(1, n)]
            avg_change = statistics.mean(value_changes)
            
            # 预测未来值
            last_value = values[-1]
            predictions = {
                'next_hour': last_value + avg_change * (3600 / (time_span / (n-1))),
                'next_day': last_value + avg_change * (86400 / (time_span / (n-1))),
                'next_week': last_value + avg_change * (604800 / (time_span / (n-1)))
            }
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"预测未来值时发生错误: {e}")
            return {}
    
    def get_data_for_metric(self, metric: PerformanceMetric, 
                           limit: int = 1000) -> List[PerformanceData]:
        """获取指定指标的历史数据"""
        with self._lock:
            return list(self._data_history[metric])[-limit:]
    
    def clear_data(self, metric: Optional[PerformanceMetric] = None) -> None:
        """
        清除历史数据
        
        Args:
            metric: 指定指标，为None时清除所有数据
        """
        with self._lock:
            if metric:
                self._data_history[metric].clear()
            else:
                self._data_history.clear()


# =============================================================================
# 异步日志处理器
# =============================================================================

class AsyncLogProcessor:
    """异步日志处理器"""
    
    def __init__(self, db_path: str = "performance_logs.db", 
                 max_workers: int = 4, logger: Optional[logging.Logger] = None):
        """
        初始化异步日志处理器
        
        Args:
            db_path: 数据库文件路径
            max_workers: 最大工作线程数
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self.db_path = db_path
        self.max_workers = max_workers
        self._queue = queue.Queue(maxsize=10000)
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._running = False
        self._thread = None
        self._stats = {
            'processed': 0,
            'errors': 0,
            'queue_size': 0
        }
        self._stats_lock = threading.Lock()
        
        # 初始化数据库
        self._init_database()
    
    def _init_database(self) -> None:
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建性能数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    tags TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metric ON performance_logs(metric)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON performance_logs(timestamp)')
            
            conn.commit()
            conn.close()
            
            self.logger.info("数据库初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化数据库时发生错误: {e}")
    
    def start(self) -> None:
        """启动异步处理器"""
        if self._running:
            self.logger.warning("异步日志处理器已经在运行")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        self.logger.info("异步日志处理器已启动")
    
    def stop(self) -> None:
        """停止异步处理器"""
        if not self._running:
            return
        
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=10.0)
        
        self._executor.shutdown(wait=True)
        self.logger.info("异步日志处理器已停止")
    
    def submit_data(self, data: PerformanceData) -> bool:
        """
        提交性能数据
        
        Args:
            data: 性能数据
            
        Returns:
            是否成功提交
        """
        try:
            self._queue.put_nowait(data)
            with self._stats_lock:
                self._stats['queue_size'] = self._queue.qsize()
            return True
        except queue.Full:
            self.logger.warning("日志处理队列已满，丢弃数据")
            return False
        except Exception as e:
            self.logger.error(f"提交数据时发生错误: {e}")
            return False
    
    def _process_loop(self) -> None:
        """处理循环"""
        while self._running:
            try:
                # 获取数据（阻塞式）
                data = self._queue.get(timeout=1.0)
                
                # 提交到线程池处理
                future = self._executor.submit(self._process_data, data)
                future.add_done_callback(self._process_callback)
                
                with self._stats_lock:
                    self._stats['queue_size'] = self._queue.qsize()
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"处理循环中发生错误: {e}")
    
    def _process_data(self, data: PerformanceData) -> None:
        """处理单个性能数据"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO performance_logs 
                (metric, value, timestamp, tags, metadata)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                data.metric.value,
                data.value,
                data.timestamp,
                json.dumps(data.tags),
                json.dumps(data.metadata)
            ))
            
            conn.commit()
            conn.close()
            
            with self._stats_lock:
                self._stats['processed'] += 1
                
        except Exception as e:
            self.logger.error(f"处理性能数据时发生错误: {e}")
            with self._stats_lock:
                self._stats['errors'] += 1
    
    def _process_callback(self, future) -> None:
        """处理完成回调"""
        try:
            future.result()
        except Exception as e:
            self.logger.error(f"异步处理回调错误: {e}")
    
    def query_logs(self, metric: Optional[PerformanceMetric] = None,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  limit: int = 1000) -> List[Dict[str, Any]]:
        """
        查询日志数据
        
        Args:
            metric: 性能指标
            start_time: 开始时间
            end_time: 结束时间
            limit: 限制数量
            
        Returns:
            查询结果列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 构建查询条件
            conditions = []
            params = []
            
            if metric:
                conditions.append("metric = ?")
                params.append(metric.value)
            
            if start_time:
                conditions.append("timestamp >= ?")
                params.append(start_time)
            
            if end_time:
                conditions.append("timestamp <= ?")
                params.append(end_time)
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f'''
                SELECT metric, value, timestamp, tags, metadata
                FROM performance_logs
                {where_clause}
                ORDER BY timestamp DESC
                LIMIT ?
            '''
            params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            results = []
            for row in rows:
                results.append({
                    'metric': PerformanceMetric(row[0]),
                    'value': row[1],
                    'timestamp': row[2],
                    'tags': json.loads(row[3]) if row[3] else {},
                    'metadata': json.loads(row[4]) if row[4] else {}
                })
            
            return results
            
        except Exception as e:
            self.logger.error(f"查询日志时发生错误: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """获取处理统计信息"""
        with self._stats_lock:
            return self._stats.copy()
    
    def cleanup_old_logs(self, days: int = 30) -> int:
        """
        清理旧日志
        
        Args:
            days: 保留天数
            
        Returns:
            删除的记录数
        """
        try:
            cutoff_time = time.time() - (days * 24 * 3600)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM performance_logs WHERE timestamp < ?', (cutoff_time,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"清理了 {deleted_count} 条旧日志记录")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"清理旧日志时发生错误: {e}")
            return 0


# =============================================================================
# 性能报告生成器
# =============================================================================

class PerformanceReportGenerator:
    """性能报告生成器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化报告生成器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def generate_summary_report(self, trend_analyzer: TrendAnalyzer,
                               time_range_hours: float = 24.0) -> Dict[str, Any]:
        """
        生成性能摘要报告
        
        Args:
            trend_analyzer: 趋势分析器
            time_range_hours: 时间范围（小时）
            
        Returns:
            报告数据
        """
        try:
            report = {
                'generated_at': time.time(),
                'time_range_hours': time_range_hours,
                'metrics_summary': {},
                'alerts_summary': {},
                'trends_summary': {}
            }
            
            # 分析各个指标的趋势
            for metric in PerformanceMetric:
                trend = trend_analyzer.analyze_trend(metric, time_range_hours)
                if trend:
                    report['trends_summary'][metric.value] = trend.to_dict()
                    
                    # 添加指标摘要
                    stats = trend.statistics
                    report['metrics_summary'][metric.value] = {
                        'current_value': stats.get('mean', 0),
                        'min_value': stats.get('min', 0),
                        'max_value': stats.get('max', 0),
                        'trend_direction': trend.trend_direction,
                        'trend_strength': trend.trend_strength
                    }
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成摘要报告时发生错误: {e}")
            return {}
    
    def generate_detailed_report(self, trend_analyzer: TrendAnalyzer,
                               alert_manager: AlertManager,
                               time_range_hours: float = 24.0) -> Dict[str, Any]:
        """
        生成详细性能报告
        
        Args:
            trend_analyzer: 趋势分析器
            alert_manager: 告警管理器
            time_range_hours: 时间范围（小时）
            
        Returns:
            详细报告数据
        """
        try:
            report = {
                'generated_at': time.time(),
                'time_range_hours': time_range_hours,
                'summary': self.generate_summary_report(trend_analyzer, time_range_hours),
                'detailed_trends': {},
                'alert_details': [],
                'recommendations': []
            }
            
            # 详细趋势分析
            for metric in PerformanceMetric:
                trend = trend_analyzer.analyze_trend(metric, time_range_hours)
                if trend:
                    report['detailed_trends'][metric.value] = trend.to_dict()
            
            # 告警详情
            active_alerts = alert_manager.get_active_alerts()
            alert_history = alert_manager.get_alert_history(100)
            
            report['alert_details'] = {
                'active_alerts': [self._alert_to_dict(alert) for alert in active_alerts],
                'recent_alerts': [self._alert_to_dict(alert) for alert in alert_history]
            }
            
            # 生成建议
            report['recommendations'] = self._generate_recommendations(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"生成详细报告时发生错误: {e}")
            return {}
    
    def _alert_to_dict(self, alert: Alert) -> Dict[str, Any]:
        """将告警转换为字典"""
        return {
            'metric': alert.rule.metric.value,
            'value': alert.value,
            'timestamp': alert.timestamp,
            'level': alert.rule.level.value,
            'message': alert.message,
            'resolved': alert.resolved,
            'resolved_timestamp': alert.resolved_timestamp
        }
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        try:
            trends = report.get('trends_summary', {})
            
            for metric_name, trend_data in trends.items():
                trend_direction = trend_data.get('trend_direction', 'stable')
                trend_strength = trend_data.get('trend_strength', 0)
                
                if trend_direction == 'increasing' and trend_strength > 0.7:
                    if 'cpu' in metric_name.lower():
                        recommendations.append("CPU使用率呈上升趋势，建议优化算法或增加CPU资源")
                    elif 'memory' in metric_name.lower():
                        recommendations.append("内存使用率呈上升趋势，建议检查内存泄漏或增加内存容量")
                    elif 'response_time' in metric_name.lower():
                        recommendations.append("响应时间呈上升趋势，建议优化数据库查询或增加服务器资源")
                
                elif trend_direction == 'decreasing' and trend_strength > 0.7:
                    if 'throughput' in metric_name.lower():
                        recommendations.append("吞吐量呈下降趋势，建议检查系统瓶颈或优化配置")
            
            # 检查告警
            alert_details = report.get('alert_details', {})
            active_alerts = alert_details.get('active_alerts', [])
            
            if len(active_alerts) > 5:
                recommendations.append("当前活跃告警较多，建议优先处理关键性能问题")
            
            if not recommendations:
                recommendations.append("系统性能运行良好，建议继续保持当前配置")
            
        except Exception as e:
            self.logger.error(f"生成建议时发生错误: {e}")
            recommendations.append("无法生成具体建议，请检查系统配置")
        
        return recommendations
    
    def export_report(self, report: Dict[str, Any], 
                     output_path: str, format: str = 'json') -> bool:
        """
        导出报告
        
        Args:
            report: 报告数据
            output_path: 输出路径
            format: 格式（json, csv, html）
            
        Returns:
            是否成功导出
        """
        try:
            if format.lower() == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            elif format.lower() == 'csv':
                import csv
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Metric', 'Current Value', 'Min Value', 'Max Value', 
                                   'Trend Direction', 'Trend Strength'])
                    
                    metrics_summary = report.get('metrics_summary', {})
                    for metric, data in metrics_summary.items():
                        writer.writerow([
                            metric,
                            data.get('current_value', 0),
                            data.get('min_value', 0),
                            data.get('max_value', 0),
                            data.get('trend_direction', 'unknown'),
                            data.get('trend_strength', 0)
                        ])
            
            elif format.lower() == 'html':
                self._export_html_report(report, output_path)
            
            else:
                self.logger.error(f"不支持的导出格式: {format}")
                return False
            
            self.logger.info(f"报告已导出到: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出报告时发生错误: {e}")
            return False
    
    def _export_html_report(self, report: Dict[str, Any], output_path: str) -> None:
        """导出HTML格式报告"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>性能监控报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .alert {{ background-color: #ffe6e6; padding: 10px; margin: 5px 0; border-radius: 3px; }}
                .recommendation {{ background-color: #e6f3ff; padding: 10px; margin: 5px 0; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>性能监控报告</h1>
                <p>生成时间: {generated_time}</p>
                <p>时间范围: {time_range} 小时</p>
            </div>
            
            <h2>指标摘要</h2>
            {metrics_html}
            
            <h2>活跃告警</h2>
            {alerts_html}
            
            <h2>优化建议</h2>
            {recommendations_html}
        </body>
        </html>
        """
        
        # 生成指标HTML
        metrics_html = ""
        metrics_summary = report.get('metrics_summary', {})
        for metric, data in metrics_summary.items():
            metrics_html += f"""
            <div class="metric">
                <h3>{metric}</h3>
                <p>当前值: {data.get('current_value', 0):.2f}</p>
                <p>最小值: {data.get('min_value', 0):.2f}</p>
                <p>最大值: {data.get('max_value', 0):.2f}</p>
                <p>趋势: {data.get('trend_direction', 'unknown')} (强度: {data.get('trend_strength', 0):.2f})</p>
            </div>
            """
        
        # 生成告警HTML
        alerts_html = ""
        alert_details = report.get('alert_details', {})
        active_alerts = alert_details.get('active_alerts', [])
        if not active_alerts:
            alerts_html = "<p>当前无活跃告警</p>"
        else:
            for alert in active_alerts:
                alerts_html += f"""
                <div class="alert">
                    <strong>{alert['level'].upper()}</strong>: {alert['message']}
                </div>
                """
        
        # 生成建议HTML
        recommendations_html = ""
        recommendations = report.get('recommendations', [])
        for rec in recommendations:
            recommendations_html += f"""
            <div class="recommendation">
                {rec}
            </div>
            """
        
        # 填充模板
        html_content = html_template.format(
            generated_time=datetime.datetime.fromtimestamp(report['generated_at']).strftime('%Y-%m-%d %H:%M:%S'),
            time_range=report['time_range_hours'],
            metrics_html=metrics_html,
            alerts_html=alerts_html,
            recommendations_html=recommendations_html
        )
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)


# =============================================================================
# 通知管理器
# =============================================================================

class NotificationManager:
    """通知管理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        初始化通知管理器
        
        Args:
            logger: 日志记录器
        """
        self.logger = logger or logging.getLogger(__name__)
        self._email_config = {}
        self._webhook_urls = []
        self._notification_rules = []
    
    def configure_email(self, smtp_server: str, smtp_port: int, 
                       username: str, password: str,
                       from_email: str, to_emails: List[str]) -> None:
        """
        配置邮件通知
        
        Args:
            smtp_server: SMTP服务器
            smtp_port: SMTP端口
            username: 用户名
            password: 密码
            from_email: 发送者邮箱
            to_emails: 接收者邮箱列表
        """
        self._email_config = {
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'username': username,
            'password': password,
            'from_email': from_email,
            'to_emails': to_emails
        }
        self.logger.info("邮件通知配置已更新")
    
    def add_webhook(self, url: str) -> None:
        """添加Webhook URL"""
        self._webhook_urls.append(url)
        self.logger.info(f"添加Webhook URL: {url}")
    
    def remove_webhook(self, url: str) -> None:
        """移除Webhook URL"""
        if url in self._webhook_urls:
            self._webhook_urls.remove(url)
            self.logger.info(f"移除Webhook URL: {url}")
    
    def send_alert_notification(self, alert: Alert) -> bool:
        """
        发送告警通知
        
        Args:
            alert: 告警信息
            
        Returns:
            是否成功发送
        """
        success = True
        
        try:
            # 邮件通知
            if self._email_config:
                if not self._send_email_alert(alert):
                    success = False
            
            # Webhook通知
            if self._webhook_urls:
                if not self._send_webhook_alert(alert):
                    success = False
            
            # 日志记录
            self.logger.info(f"告警通知已发送: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"发送告警通知时发生错误: {e}")
            success = False
        
        return success
    
    def _send_email_alert(self, alert: Alert) -> bool:
        """发送邮件告警"""
        try:
            config = self._email_config
            
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ', '.join(config['to_emails'])
            msg['Subject'] = f"性能告警 - {alert.rule.level.value.upper()}"
            
            body = f"""
            性能告警通知
            
            告警级别: {alert.rule.level.value.upper()}
            告警信息: {alert.message}
            当前值: {alert.value}
            触发时间: {datetime.datetime.fromtimestamp(alert.timestamp).strftime('%Y-%m-%d %H:%M:%S')}
            
            请及时处理此告警。
            
            性能监控系统
            """
            
            msg.attach(MIMEText(body, 'plain', 'utf-8'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['username'], config['password'])
            server.send_message(msg)
            server.quit()
            
            return True
            
        except Exception as e:
            self.logger.error(f"发送邮件告警时发生错误: {e}")
            return False
    
    def _send_webhook_alert(self, alert: Alert) -> bool:
        """发送Webhook告警"""
        try:
            import requests
            
            payload = {
                'alert_level': alert.rule.level.value,
                'message': alert.message,
                'value': alert.value,
                'timestamp': alert.timestamp,
                'metric': alert.rule.metric.value
            }
            
            for url in self._webhook_urls:
                try:
                    response = requests.post(url, json=payload, timeout=10)
                    response.raise_for_status()
                except Exception as e:
                    self.logger.error(f"发送Webhook告警到 {url} 时发生错误: {e}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"发送Webhook告警时发生错误: {e}")
            return False


# =============================================================================
# 主要性能日志记录器类
# =============================================================================

class PerformanceLogger:
    """
    L4性能日志记录器主类
    
    提供完整的性能监控、日志记录、告警和报告功能。
    支持系统性能、应用性能、数据库性能和网络性能的监控。
    """
    
    def __init__(self, 
                 name: str = "PerformanceLogger",
                 log_level: LogLevel = LogLevel.INFO,
                 db_path: str = "performance_logs.db",
                 enable_system_monitor: bool = True,
                 enable_application_monitor: bool = True,
                 enable_database_monitor: bool = False,
                 enable_network_monitor: bool = False,
                 logger: Optional[logging.Logger] = None):
        """
        初始化性能日志记录器
        
        Args:
            name: 日志记录器名称
            log_level: 日志级别
            db_path: 数据库文件路径
            enable_system_monitor: 是否启用系统监控
            enable_application_monitor: 是否启用应用监控
            enable_database_monitor: 是否启用数据库监控
            enable_network_monitor: 是否启用网络监控
            logger: 日志记录器
        """
        self.name = name
        self.logger = logger or self._setup_logger(name, log_level)
        self.db_path = db_path
        
        # 初始化组件
        self._init_components(
            enable_system_monitor,
            enable_application_monitor,
            enable_database_monitor,
            enable_network_monitor
        )
        
        # 运行状态
        self._running = False
        self._start_time = None
        
        # 性能统计
        self._stats = {
            'start_time': None,
            'data_points_collected': 0,
            'alerts_generated': 0,
            'reports_generated': 0,
            'errors_count': 0
        }
        self._stats_lock = threading.RLock()
        
        self.logger.info(f"性能日志记录器 {name} 初始化完成")
    
    def _setup_logger(self, name: str, level: LogLevel) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.value.upper()))
        
        # 避免重复添加handler
        if not logger.handlers:
            # 控制台handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, level.value.upper()))
            
            # 文件handler
            file_handler = logging.FileHandler(f"{name}.log", encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # 格式化器
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
        
        return logger
    
    def _init_components(self, system: bool, app: bool, db: bool, network: bool) -> None:
        """初始化各个组件"""
        try:
            # 异步日志处理器
            self.log_processor = AsyncLogProcessor(self.db_path, logger=self.logger)
            
            # 告警管理器
            self.alert_manager = AlertManager(self.logger)
            self.alert_manager.add_callback(self._handle_alert)
            
            # 趋势分析器
            self.trend_analyzer = TrendAnalyzer(self.logger)
            
            # 通知管理器
            self.notification_manager = NotificationManager(self.logger)
            
            # 报告生成器
            self.report_generator = PerformanceReportGenerator(self.logger)
            
            # 性能监控器
            self.system_monitor = None
            self.application_monitor = None
            self.database_monitor = None
            self.network_monitor = None
            
            if system:
                self.system_monitor = SystemPerformanceMonitor(self.logger)
                self.system_monitor.add_callback(self._handle_performance_data)
            
            if app:
                self.application_monitor = ApplicationPerformanceMonitor(self.logger)
                self.application_monitor.add_callback(self._handle_performance_data)
            
            if db:
                self.database_monitor = DatabasePerformanceMonitor(
                    DatabaseType.SQLITE, "sqlite:///performance.db", self.logger
                )
                self.database_monitor.add_callback(self._handle_performance_data)
            
            if network:
                self.network_monitor = NetworkPerformanceMonitor(logger=self.logger)
                self.network_monitor.add_callback(self._handle_performance_data)
            
            self.logger.info("所有组件初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化组件时发生错误: {e}")
            raise
    
    def start(self, monitoring_interval: float = 1.0) -> None:
        """
        启动性能日志记录器
        
        Args:
            monitoring_interval: 监控间隔（秒）
        """
        if self._running:
            self.logger.warning("性能日志记录器已经在运行")
            return
        
        try:
            self._running = True
            self._start_time = time.time()
            
            with self._stats_lock:
                self._stats['start_time'] = self._start_time
            
            # 启动异步日志处理器
            self.log_processor.start()
            
            # 启动性能监控器
            if self.system_monitor:
                self.system_monitor.start(monitoring_interval)
            
            if self.application_monitor:
                self.application_monitor.start(monitoring_interval)
            
            if self.database_monitor:
                self.database_monitor.start(monitoring_interval)
            
            if self.network_monitor:
                self.network_monitor.start(monitoring_interval * 5)  # 网络监控间隔更长
            
            self.logger.info(f"性能日志记录器 {self.name} 已启动，监控间隔: {monitoring_interval}秒")
            
        except Exception as e:
            self.logger.error(f"启动性能日志记录器时发生错误: {e}")
            self._running = False
            raise
    
    def stop(self) -> None:
        """停止性能日志记录器"""
        if not self._running:
            return
        
        try:
            self._running = False
            
            # 停止性能监控器
            if self.system_monitor:
                self.system_monitor.stop()
            
            if self.application_monitor:
                self.application_monitor.stop()
            
            if self.database_monitor:
                self.database_monitor.stop()
            
            if self.network_monitor:
                self.network_monitor.stop()
            
            # 停止异步日志处理器
            self.log_processor.stop()
            
            self.logger.info(f"性能日志记录器 {self.name} 已停止")
            
        except Exception as e:
            self.logger.error(f"停止性能日志记录器时发生错误: {e}")
    
    def _handle_performance_data(self, data: PerformanceData) -> None:
        """处理性能数据"""
        try:
            # 提交到异步处理器
            self.log_processor.submit_data(data)
            
            # 添加到趋势分析器
            self.trend_analyzer.add_data(data)
            
            # 评估告警
            alerts = self.alert_manager.evaluate_data(data)
            
            with self._stats_lock:
                self._stats['data_points_collected'] += 1
                self._stats['alerts_generated'] += len(alerts)
            
        except Exception as e:
            self.logger.error(f"处理性能数据时发生错误: {e}")
            with self._stats_lock:
                self._stats['errors_count'] += 1
    
    def _handle_alert(self, alert: Alert) -> None:
        """处理告警"""
        try:
            # 发送通知
            self.notification_manager.send_alert_notification(alert)
            
            self.logger.warning(f"处理告警: {alert.message}")
            
        except Exception as e:
            self.logger.error(f"处理告警时发生错误: {e}")
    
    # 应用性能相关方法
    def record_application_request(self, response_time: float, status_code: int = 200,
                                  method: str = "GET", endpoint: str = "/") -> None:
        """
        记录应用请求性能
        
        Args:
            response_time: 响应时间（毫秒）
            status_code: HTTP状态码
            method: HTTP方法
            endpoint: 请求端点
        """
        if self.application_monitor:
            self.application_monitor.record_request(response_time, status_code, method, endpoint)
        else:
            self.logger.warning("应用性能监控器未启用")
    
    def get_application_stats(self) -> Dict[str, Any]:
        """获取应用性能统计"""
        if not self.application_monitor:
            return {}
        
        return {
            'throughput': self.application_monitor.get_throughput(),
            'success_rate': self.application_monitor.get_success_rate(),
            'current_response_time': self.application_monitor.get_current_value()
        }
    
    # 数据库性能相关方法
    def record_database_query(self, query: str, execution_time: float,
                             rows_affected: int = 0, success: bool = True) -> None:
        """
        记录数据库查询性能
        
        Args:
            query: SQL查询语句
            execution_time: 执行时间（毫秒）
            rows_affected: 影响的行数
            success: 是否成功
        """
        if self.database_monitor:
            self.database_monitor.record_query(query, execution_time, rows_affected, success)
        else:
            self.logger.warning("数据库性能监控器未启用")
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库性能统计"""
        if not self.database_monitor:
            return {}
        
        return {
            'connection_count': self.database_monitor.get_connection_count(),
            'query_success_rate': self.database_monitor.get_query_success_rate(),
            'current_query_time': self.database_monitor.get_current_value()
        }
    
    # 系统性能相关方法
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统性能统计"""
        if not self.system_monitor:
            return {}
        
        return {
            'cpu_usage': self.system_monitor.get_current_value(),
            'memory_usage': self.system_monitor.get_memory_usage(),
            'disk_usage': self.system_monitor.get_disk_usage(),
            'network_stats': self.system_monitor.get_network_stats()
        }
    
    # 网络性能相关方法
    def get_network_stats(self) -> Dict[str, Any]:
        """获取网络性能统计"""
        if not self.network_monitor:
            return {}
        
        return {
            'average_latency': self.network_monitor.get_average_latency(),
            'packet_loss_rate': self.network_monitor.get_packet_loss_rate(),
            'current_latency': self.network_monitor.get_current_value()
        }
    
    # 告警管理方法
    def add_alert_rule(self, metric: PerformanceMetric, threshold: float,
                      comparison_operator: str, level: AlertLevel,
                      description: str = "", consecutive_count: int = 1) -> None:
        """
        添加告警规则
        
        Args:
            metric: 性能指标
            threshold: 阈值
            comparison_operator: 比较操作符
            level: 告警级别
            description: 描述
            consecutive_count: 连续触发次数
        """
        rule = AlertRule(
            metric=metric,
            threshold=threshold,
            comparison_operator=comparison_operator,
            level=level,
            description=description,
            consecutive_count=consecutive_count
        )
        self.alert_manager.add_rule(rule)
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return self.alert_manager.get_active_alerts()
    
    def get_alert_history(self, limit: int = 100) -> List[Alert]:
        """获取告警历史"""
        return self.alert_manager.get_alert_history(limit)
    
    # 趋势分析方法
    def analyze_trend(self, metric: PerformanceMetric, 
                     time_range_hours: float = 24.0) -> Optional[TrendAnalysis]:
        """
        分析性能趋势
        
        Args:
            metric: 性能指标
            time_range_hours: 时间范围（小时）
            
        Returns:
            趋势分析结果
        """
        return self.trend_analyzer.analyze_trend(metric, time_range_hours)
    
    def get_performance_data(self, metric: Optional[PerformanceMetric] = None,
                           limit: int = 1000) -> List[PerformanceData]:
        """
        获取性能数据
        
        Args:
            metric: 性能指标，为None时获取所有指标
            limit: 限制数量
            
        Returns:
            性能数据列表
        """
        if metric:
            return self.trend_analyzer.get_data_for_metric(metric, limit)
        else:
            # 获取所有指标的数据
            all_data = []
            for m in PerformanceMetric:
                all_data.extend(self.trend_analyzer.get_data_for_metric(m, limit // len(PerformanceMetric)))
            return sorted(all_data, key=lambda x: x.timestamp, reverse=True)[:limit]
    
    # 报告生成方法
    def generate_summary_report(self, time_range_hours: float = 24.0) -> Dict[str, Any]:
        """
        生成性能摘要报告
        
        Args:
            time_range_hours: 时间范围（小时）
            
        Returns:
            报告数据
        """
        try:
            with self._stats_lock:
                self._stats['reports_generated'] += 1
            
            return self.report_generator.generate_summary_report(
                self.trend_analyzer, time_range_hours
            )
        except Exception as e:
            self.logger.error(f"生成摘要报告时发生错误: {e}")
            return {}
    
    def generate_detailed_report(self, time_range_hours: float = 24.0) -> Dict[str, Any]:
        """
        生成详细性能报告
        
        Args:
            time_range_hours: 时间范围（小时）
            
        Returns:
            详细报告数据
        """
        try:
            with self._stats_lock:
                self._stats['reports_generated'] += 1
            
            return self.report_generator.generate_detailed_report(
                self.trend_analyzer, self.alert_manager, time_range_hours
            )
        except Exception as e:
            self.logger.error(f"生成详细报告时发生错误: {e}")
            return {}
    
    def export_report(self, report: Dict[str, Any], output_path: str,
                     format: str = 'json') -> bool:
        """
        导出报告
        
        Args:
            report: 报告数据
            output_path: 输出路径
            format: 格式（json, csv, html）
            
        Returns:
            是否成功导出
        """
        return self.report_generator.export_report(report, output_path, format)
    
    # 通知配置方法
    def configure_email_notification(self, smtp_server: str, smtp_port: int,
                                   username: str, password: str,
                                   from_email: str, to_emails: List[str]) -> None:
        """配置邮件通知"""
        self.notification_manager.configure_email(
            smtp_server, smtp_port, username, password, from_email, to_emails
        )
    
    def add_webhook_notification(self, url: str) -> None:
        """添加Webhook通知"""
        self.notification_manager.add_webhook(url)
    
    # 查询方法
    def query_logs(self, metric: Optional[PerformanceMetric] = None,
                  start_time: Optional[float] = None,
                  end_time: Optional[float] = None,
                  limit: int = 1000) -> List[Dict[str, Any]]:
        """
        查询日志数据
        
        Args:
            metric: 性能指标
            start_time: 开始时间
            end_time: 结束时间
            limit: 限制数量
            
        Returns:
            查询结果列表
        """
        return self.log_processor.query_logs(metric, start_time, end_time, limit)
    
    # 维护方法
    def cleanup_old_logs(self, days: int = 30) -> int:
        """清理旧日志"""
        return self.log_processor.cleanup_old_logs(days)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._stats_lock:
            stats = self._stats.copy()
            stats['running'] = self._running
            stats['uptime'] = time.time() - self._start_time if self._start_time else 0
            stats['log_processor_stats'] = self.log_processor.get_stats()
            return stats
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health = {
            'status': 'healthy',
            'timestamp': time.time(),
            'components': {},
            'issues': []
        }
        
        try:
            # 检查各个组件状态
            components = {
                'system_monitor': self.system_monitor,
                'application_monitor': self.application_monitor,
                'database_monitor': self.database_monitor,
                'network_monitor': self.network_monitor,
                'log_processor': self.log_processor,
                'alert_manager': self.alert_manager,
                'trend_analyzer': self.trend_analyzer
            }
            
            for name, component in components.items():
                if component:
                    health['components'][name] = 'active'
                else:
                    health['components'][name] = 'disabled'
            
            # 检查活跃告警
            active_alerts = self.get_active_alerts()
            if active_alerts:
                health['issues'].append(f"存在 {len(active_alerts)} 个活跃告警")
            
            # 检查错误统计
            stats = self.get_stats()
            if stats.get('errors_count', 0) > 100:
                health['issues'].append(f"错误数量较多: {stats['errors_count']}")
            
            if health['issues']:
                health['status'] = 'warning' if len(active_alerts) < 5 else 'critical'
            
        except Exception as e:
            health['status'] = 'error'
            health['error'] = str(e)
        
        return health
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop()


# =============================================================================
# 使用示例和测试代码
# =============================================================================

def example_usage():
    """使用示例"""
    # 创建性能日志记录器
    with PerformanceLogger(
        name="ExamplePerformanceLogger",
        log_level=LogLevel.DEBUG,
        enable_system_monitor=True,
        enable_application_monitor=True,
        enable_database_monitor=True,
        enable_network_monitor=True
    ) as logger:
        
        # 配置告警规则
        logger.add_alert_rule(
            metric=PerformanceMetric.CPU_USAGE,
            threshold=80.0,
            comparison_operator='>',
            level=AlertLevel.WARNING,
            description="CPU使用率过高"
        )
        
        logger.add_alert_rule(
            metric=PerformanceMetric.RESPONSE_TIME,
            threshold=1000.0,
            comparison_operator='>',
            level=AlertLevel.ERROR,
            description="响应时间过长"
        )
        
        # 记录应用请求
        for i in range(100):
            # 模拟应用请求
            
            response_time = random.uniform(50, 200)  # 50-200ms
            status_code = 200 if random.random() > 0.05 else 500  # 95%成功率
            
            logger.record_application_request(
                response_time=response_time,
                status_code=status_code,
                method="GET",
                endpoint=f"/api/test/{i}"
            )
            
            # 模拟数据库查询
            query_time = random.uniform(10, 100)  # 10-100ms
            logger.record_database_query(
                query="SELECT * FROM users",
                execution_time=query_time,
                rows_affected=random.randint(1, 100),
                success=random.random() > 0.02  # 98%成功率
            )
            
            time.sleep(0.1)  # 间隔100ms
        
        # 等待一段时间让数据收集
        time.sleep(5)
        
        # 获取统计信息
        print("=== 系统统计 ===")
        system_stats = logger.get_system_stats()
        print(f"CPU使用率: {system_stats.get('cpu_usage', 'N/A')}%")
        print(f"内存使用率: {system_stats.get('memory_usage', 'N/A')}%")
        
        print("\n=== 应用统计 ===")
        app_stats = logger.get_application_stats()
        print(f"吞吐量: {app_stats.get('throughput', 'N/A')} req/s")
        print(f"成功率: {app_stats.get('success_rate', 'N/A')}%")
        
        print("\n=== 数据库统计 ===")
        db_stats = logger.get_database_stats()
        print(f"连接数: {db_stats.get('connection_count', 'N/A')}")
        print(f"查询成功率: {db_stats.get('query_success_rate', 'N/A')}%")
        
        print("\n=== 网络统计 ===")
        network_stats = logger.get_network_stats()
        print(f"平均延迟: {network_stats.get('average_latency', 'N/A')}ms")
        print(f"丢包率: {network_stats.get('packet_loss_rate', 'N/A')}%")
        
        # 趋势分析
        print("\n=== 趋势分析 ===")
        cpu_trend = logger.analyze_trend(PerformanceMetric.CPU_USAGE, time_range_hours=1.0)
        if cpu_trend:
            print(f"CPU趋势: {cpu_trend.trend_direction} (强度: {cpu_trend.trend_strength:.2f})")
            print(f"CPU统计: 均值={cpu_trend.statistics['mean']:.2f}, 最大值={cpu_trend.statistics['max']:.2f}")
        
        # 生成报告
        print("\n=== 生成报告 ===")
        summary_report = logger.generate_summary_report(time_range_hours=1.0)
        logger.export_report(summary_report, "performance_summary.json", "json")
        
        detailed_report = logger.generate_detailed_report(time_range_hours=1.0)
        logger.export_report(detailed_report, "performance_detailed.html", "html")
        
        # 活跃告警
        print("\n=== 活跃告警 ===")
        active_alerts = logger.get_active_alerts()
        for alert in active_alerts:
            print(f"告警: {alert.message}")
        
        # 健康检查
        print("\n=== 健康检查 ===")
        health = logger.health_check()
        print(f"状态: {health['status']}")
        print(f"组件: {health['components']}")
        if health['issues']:
            print(f"问题: {health['issues']}")
        
        # 最终统计
        print("\n=== 最终统计 ===")
        final_stats = logger.get_stats()
        print(f"数据点收集: {final_stats['data_points_collected']}")
        print(f"告警生成: {final_stats['alerts_generated']}")
        print(f"报告生成: {final_stats['reports_generated']}")
        print(f"错误数量: {final_stats['errors_count']}")
        print(f"运行时间: {final_stats['uptime']:.2f}秒")


def stress_test():
    """压力测试"""
    print("开始压力测试...")
    
    logger = PerformanceLogger(
        name="StressTestLogger",
        log_level=LogLevel.WARNING,  # 减少日志输出
        enable_system_monitor=True,
        enable_application_monitor=True,
        enable_database_monitor=False,
        enable_network_monitor=False
    )
    
    try:
        logger.start(monitoring_interval=0.1)  # 高频率监控
        
        # 模拟高并发请求
        import threading
        import queue
        
        def simulate_requests():
            for i in range(1000):
                response_time = random.uniform(10, 500)
                logger.record_application_request(
                    response_time=response_time,
                    status_code=200 if random.random() > 0.1 else 500
                )
        
        # 启动多个线程模拟高并发
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=simulate_requests)
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 等待数据处理
        time.sleep(2)
        
        # 检查性能
        stats = logger.get_stats()
        print(f"压力测试完成:")
        print(f"- 数据点收集: {stats['data_points_collected']}")
        print(f"- 处理错误: {stats['errors_count']}")
        print(f"- 运行时间: {stats['uptime']:.2f}秒")
        
        # 检查日志处理器状态
        log_stats = stats['log_processor_stats']
        print(f"- 处理的数据: {log_stats['processed']}")
        print(f"- 队列大小: {log_stats['queue_size']}")
        
    finally:
        logger.stop()


if __name__ == "__main__":
    print("L4性能日志记录器测试")
    print("=" * 50)
    
    try:
        # 运行使用示例
        print("1. 运行使用示例...")
        example_usage()
        
        print("\n" + "=" * 50)
        print("2. 运行压力测试...")
        stress_test()
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n测试过程中发生错误: {e}")
        traceback.print_exc()
    
    print("\n测试完成")