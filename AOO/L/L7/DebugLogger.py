#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L7调试日志记录器

这是一个全面的L7层调试日志记录系统，提供代码调试、性能分析、内存泄漏检测、
网络调试、数据库调试、调试工具集成等多种调试功能。

主要功能:
1. 代码调试日志（变量值、函数调用、执行路径）
2. 性能分析日志（函数耗时、内存使用、调用次数）
3. 内存泄漏检测日志（内存分配、内存释放、泄漏追踪）
4. 网络调试日志（请求响应、连接状态、错误诊断）
5. 数据库调试日志（SQL执行、查询计划、锁等待）
6. 调试工具集成日志（断点、单步执行、变量检查）
7. 异步调试日志处理
8. 完整的错误处理和日志记录

"""

import asyncio
import functools
import gc
import hashlib
import inspect
import json
import logging
import os
import psutil
import sys
import threading
import time
import traceback
import weakref
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union, Tuple
from urllib.parse import urlparse
import sqlite3
import socket
import ssl
import tracemalloc


# =============================================================================
# 基础异常类定义
# =============================================================================

class DebugLoggerError(Exception):
    """调试日志记录器基础异常类"""
    pass


class MemoryLeakError(DebugLoggerError):
    """内存泄漏异常类"""
    pass


class PerformanceThresholdError(DebugLoggerError):
    """性能阈值异常类"""
    pass


class NetworkDebugError(DebugLoggerError):
    """网络调试异常类"""
    pass


class DatabaseDebugError(DebugLoggerError):
    """数据库调试异常类"""
    pass


# =============================================================================
# 基础数据结构和枚举
# =============================================================================

class LogLevel:
    """日志级别常量"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    TRACE = "TRACE"
    PROFILE = "PROFILE"
    MEMORY = "MEMORY"
    NETWORK = "NETWORK"
    DATABASE = "DATABASE"
    PERFORMANCE = "PERFORMANCE"


class DebugCategory:
    """调试类别常量"""
    CODE = "CODE"
    PERFORMANCE = "PERFORMANCE"
    MEMORY = "MEMORY"
    NETWORK = "NETWORK"
    DATABASE = "DATABASE"
    TOOL = "TOOL"
    ASYNC = "ASYNC"
    GENERAL = "GENERAL"


# =============================================================================
# 基础数据结构类
# =============================================================================

class LogEntry:
    """日志条目数据结构"""
    
    def __init__(self, 
                 level: str,
                 category: str,
                 message: str,
                 timestamp: Optional[datetime] = None,
                 thread_id: Optional[int] = None,
                 process_id: Optional[int] = None,
                 file_path: Optional[str] = None,
                 line_number: Optional[int] = None,
                 function_name: Optional[str] = None,
                 extra_data: Optional[Dict[str, Any]] = None):
        """
        初始化日志条目
        
        Args:
            level: 日志级别
            category: 调试类别
            message: 日志消息
            timestamp: 时间戳
            thread_id: 线程ID
            process_id: 进程ID
            file_path: 文件路径
            line_number: 行号
            function_name: 函数名
            extra_data: 额外数据
        """
        self.level = level
        self.category = category
        self.message = message
        self.timestamp = timestamp or datetime.now()
        self.thread_id = thread_id or threading.get_ident()
        self.process_id = process_id or os.getpid()
        self.file_path = file_path
        self.line_number = line_number
        self.function_name = function_name
        self.extra_data = extra_data or {}
        self.id = hashlib.md5(f"{self.timestamp}{self.level}{self.message}".encode()).hexdigest()[:16]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'level': self.level,
            'category': self.category,
            'message': self.message,
            'timestamp': self.timestamp.isoformat(),
            'thread_id': self.thread_id,
            'process_id': self.process_id,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'function_name': self.function_name,
            'extra_data': self.extra_data
        }
    
    def __str__(self) -> str:
        """字符串表示"""
        timestamp_str = self.timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        return (f"[{timestamp_str}] [{self.level}] [{self.category}] "
                f"[T:{self.thread_id}] [{self.file_path}:{self.line_number}] "
                f"{self.message}")


class PerformanceMetrics:
    """性能指标数据结构"""
    
    def __init__(self, function_name: str, file_path: str, line_number: int):
        """
        初始化性能指标
        
        Args:
            function_name: 函数名
            file_path: 文件路径
            line_number: 行号
        """
        self.function_name = function_name
        self.file_path = file_path
        self.line_number = line_number
        self.call_count = 0
        self.total_time = 0.0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.avg_time = 0.0
        self.memory_usage = []
        self.cpu_usage = []
        self.timestamps = []
    
    def update(self, execution_time: float, memory_usage: float, cpu_usage: float):
        """更新性能指标"""
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.call_count
        self.memory_usage.append(memory_usage)
        self.cpu_usage.append(cpu_usage)
        self.timestamps.append(datetime.now())
    
    def get_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        return {
            'function_name': self.function_name,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'call_count': self.call_count,
            'total_time': self.total_time,
            'min_time': self.min_time if self.min_time != float('inf') else 0,
            'max_time': self.max_time,
            'avg_time': self.avg_time,
            'avg_memory': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
            'avg_cpu': sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
        }


class MemorySnapshot:
    """内存快照数据结构"""
    
    def __init__(self, timestamp: datetime, total_memory: int, allocated_objects: int):
        """
        初始化内存快照
        
        Args:
            timestamp: 时间戳
            total_memory: 总内存使用量（字节）
            allocated_objects: 分配的对象数量
        """
        self.timestamp = timestamp
        self.total_memory = total_memory
        self.allocated_objects = allocated_objects
        self.memory_diff = 0
        self.objects_diff = 0
    
    def calculate_diff(self, previous_snapshot: Optional['MemorySnapshot']):
        """计算与前一个快照的差异"""
        if previous_snapshot:
            self.memory_diff = self.total_memory - previous_snapshot.total_memory
            self.objects_diff = self.allocated_objects - previous_snapshot.allocated_objects


# =============================================================================
# 异步日志处理器
# =============================================================================

class AsyncLogHandler:
    """异步日志处理器"""
    
    def __init__(self, max_queue_size: int = 10000, flush_interval: float = 1.0):
        """
        初始化异步日志处理器
        
        Args:
            max_queue_size: 最大队列大小
            flush_interval: 刷新间隔（秒）
        """
        self.max_queue_size = max_queue_size
        self.flush_interval = flush_interval
        self.log_queue = asyncio.Queue(maxsize=max_queue_size)
        self.is_running = False
        self.flush_task = None
        self.handlers = []
    
    async def start(self):
        """启动异步日志处理器"""
        if self.is_running:
            return
        
        self.is_running = True
        self.flush_task = asyncio.create_task(self._flush_loop())
    
    async def stop(self):
        """停止异步日志处理器"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.flush_task:
            self.flush_task.cancel()
            try:
                await self.flush_task
            except asyncio.CancelledError:
                pass
    
    async def put(self, log_entry: LogEntry):
        """添加日志条目到队列"""
        try:
            self.log_queue.put_nowait(log_entry)
        except asyncio.QueueFull:
            # 队列满时，移除最旧的条目
            try:
                self.log_queue.get_nowait()
                self.log_queue.put_nowait(log_entry)
            except asyncio.QueueEmpty:
                pass
    
    async def _flush_loop(self):
        """刷新循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush()
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"异步日志处理器错误: {e}")
    
    async def flush(self):
        """刷新日志队列"""
        entries = []
        while not self.log_queue.empty():
            try:
                entry = self.log_queue.get_nowait()
                entries.append(entry)
            except asyncio.QueueEmpty:
                break
        
        for handler in self.handlers:
            try:
                await handler.handle(entries)
            except Exception as e:
                print(f"日志处理器错误: {e}")
    
    def add_handler(self, handler):
        """添加日志处理器"""
        self.handlers.append(handler)


# =============================================================================
# 基础日志处理器
# =============================================================================

class BaseLogHandler:
    """基础日志处理器抽象类"""
    
    def __init__(self, name: str):
        """
        初始化处理器
        
        Args:
            name: 处理器名称
        """
        self.name = name
    
    async def handle(self, entries: List[LogEntry]):
        """处理日志条目"""
        raise NotImplementedError


class FileLogHandler(BaseLogHandler):
    """文件日志处理器"""
    
    def __init__(self, file_path: str, encoding: str = 'utf-8', max_size: int = 100 * 1024 * 1024):
        """
        初始化文件日志处理器
        
        Args:
            file_path: 日志文件路径
            encoding: 文件编码
            max_size: 最大文件大小（字节）
        """
        super().__init__("FileLogHandler")
        self.file_path = Path(file_path)
        self.encoding = encoding
        self.max_size = max_size
        self.current_size = 0
        self._ensure_file()
    
    def _ensure_file(self):
        """确保文件存在并检查大小"""
        if not self.file_path.parent.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.file_path.exists():
            self.current_size = self.file_path.stat().st_size
    
    async def handle(self, entries: List[LogEntry]):
        """处理日志条目"""
        if not entries:
            return
        
        # 检查文件大小，必要时轮转
        if self.current_size > self.max_size:
            await self._rotate_file()
        
        # 写入日志
        with open(self.file_path, 'a', encoding=self.encoding) as f:
            for entry in entries:
                line = str(entry) + '\n'
                f.write(line)
                self.current_size += len(line.encode(self.encoding))
    
    async def _rotate_file(self):
        """轮转日志文件"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.file_path.with_suffix(f'.{timestamp}.backup')
        self.file_path.rename(backup_path)
        self.current_size = 0


class ConsoleLogHandler(BaseLogHandler):
    """控制台日志处理器"""
    
    def __init__(self, colored: bool = True):
        """
        初始化控制台日志处理器
        
        Args:
            colored: 是否使用彩色输出
        """
        super().__init__("ConsoleLogHandler")
        self.colored = colored
        self.color_map = {
            LogLevel.DEBUG: '\033[36m',    # 青色
            LogLevel.INFO: '\033[32m',     # 绿色
            LogLevel.WARNING: '\033[33m',  # 黄色
            LogLevel.ERROR: '\033[31m',    # 红色
            LogLevel.CRITICAL: '\033[35m', # 紫色
            LogLevel.TRACE: '\033[37m',    # 白色
            LogLevel.PROFILE: '\033[34m',  # 蓝色
            LogLevel.MEMORY: '\033[35m',   # 紫色
            LogLevel.NETWORK: '\033[36m',  # 青色
            LogLevel.DATABASE: '\033[37m', # 白色
            LogLevel.PERFORMANCE: '\033[34m' # 蓝色
        }
        self.reset_color = '\033[0m'
    
    async def handle(self, entries: List[LogEntry]):
        """处理日志条目"""
        if not entries:
            return
        
        for entry in entries:
            if self.colored and entry.level in self.color_map:
                color = self.color_map[entry.level]
                reset = self.reset_color
                print(f"{color}{entry}{reset}")
            else:
                print(entry)


class DatabaseLogHandler(BaseLogHandler):
    """数据库日志处理器"""
    
    def __init__(self, db_path: str):
        """
        初始化数据库日志处理器
        
        Args:
            db_path: 数据库文件路径
        """
        super().__init__("DatabaseLogHandler")
        self.db_path = Path(db_path)
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        if not self.db_path.parent.exists():
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS debug_logs (
                    id TEXT PRIMARY KEY,
                    level TEXT NOT NULL,
                    category TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    thread_id INTEGER NOT NULL,
                    process_id INTEGER NOT NULL,
                    file_path TEXT,
                    line_number INTEGER,
                    function_name TEXT,
                    extra_data TEXT
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON debug_logs(timestamp)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_level ON debug_logs(level)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_category ON debug_logs(category)
            ''')
    
    async def handle(self, entries: List[LogEntry]):
        """处理日志条目"""
        if not entries:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            for entry in entries:
                conn.execute('''
                    INSERT OR REPLACE INTO debug_logs 
                    (id, level, category, message, timestamp, thread_id, 
                     process_id, file_path, line_number, function_name, extra_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.id, entry.level, entry.category, entry.message,
                    entry.timestamp.isoformat(), entry.thread_id, entry.process_id,
                    entry.file_path, entry.line_number, entry.function_name,
                    json.dumps(entry.extra_data, ensure_ascii=False, default=str)
                ))
            conn.commit()


# =============================================================================
# 内存监控器
# =============================================================================

class MemoryMonitor:
    """内存监控器"""
    
    def __init__(self, debug_logger: 'DebugLogger'):
        """
        初始化内存监控器
        
        Args:
            debug_logger: 调试日志记录器实例
        """
        self.debug_logger = debug_logger
        self.snapshots = deque(maxlen=1000)
        self.allocations = {}
        self.releases = {}
        self.memory_threshold = 100 * 1024 * 1024  # 100MB
        self.is_monitoring = False
        self.monitor_task = None
        self._start_tracemalloc()
    
    def _start_tracemalloc(self):
        """启动内存跟踪"""
        try:
            tracemalloc.start()
        except Exception as e:
            self.debug_logger.log(LogLevel.WARNING, DebugCategory.MEMORY, 
                                f"无法启动内存跟踪: {e}")
    
    async def start_monitoring(self, interval: float = 5.0):
        """开始内存监控"""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop(interval))
    
    async def stop_monitoring(self):
        """停止内存监控"""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self, interval: float):
        """监控循环"""
        while self.is_monitoring:
            try:
                await asyncio.sleep(interval)
                await self.take_snapshot()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.debug_logger.log(LogLevel.ERROR, DebugCategory.MEMORY, 
                                    f"内存监控错误: {e}")
    
    async def take_snapshot(self):
        """获取内存快照"""
        try:
            # 获取系统内存使用情况
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # 获取Python对象数量
            allocated_objects = len(gc.get_objects())
            
            # 获取tracemalloc快照
            if tracemalloc.is_tracing():
                current, peak = tracemalloc.get_traced_memory()
            else:
                current = memory_info.rss
                peak = memory_info.rss
            
            # 创建快照
            snapshot = MemorySnapshot(
                timestamp=datetime.now(),
                total_memory=current,
                allocated_objects=allocated_objects
            )
            
            # 计算差异
            if self.snapshots:
                previous = self.snapshots[-1]
                snapshot.calculate_diff(previous)
            
            self.snapshots.append(snapshot)
            
            # 检查内存阈值
            if current > self.memory_threshold:
                self.debug_logger.log(LogLevel.WARNING, DebugCategory.MEMORY,
                                    f"内存使用量超过阈值: {current / 1024 / 1024:.2f}MB",
                                    extra_data={
                                        'current_memory': current,
                                        'memory_threshold': self.memory_threshold,
                                        'memory_percent': memory_percent
                                    })
            
            # 记录内存快照
            self.debug_logger.log(LogLevel.MEMORY, DebugCategory.MEMORY,
                                f"内存快照: {current / 1024 / 1024:.2f}MB, "
                                f"对象数量: {allocated_objects}",
                                extra_data={
                                    'total_memory': current,
                                    'allocated_objects': allocated_objects,
                                    'memory_diff': snapshot.memory_diff,
                                    'objects_diff': snapshot.objects_diff,
                                    'memory_percent': memory_percent,
                                    'peak_memory': peak
                                })
            
        except Exception as e:
            self.debug_logger.log(LogLevel.ERROR, DebugCategory.MEMORY,
                                f"获取内存快照失败: {e}")
    
    def track_allocation(self, obj_id: int, size: int, source: str):
        """跟踪内存分配"""
        self.allocations[obj_id] = {
            'size': size,
            'source': source,
            'timestamp': datetime.now(),
            'traceback': traceback.format_stack()
        }
    
    def track_release(self, obj_id: int):
        """跟踪内存释放"""
        if obj_id in self.allocations:
            allocation = self.allocations.pop(obj_id)
            self.releases[obj_id] = {
                'allocation': allocation,
                'release_time': datetime.now()
            }
    
    def detect_leaks(self, min_age: timedelta = timedelta(minutes=5)) -> List[Dict[str, Any]]:
        """检测内存泄漏"""
        leaks = []
        current_time = datetime.now()
        
        for obj_id, allocation in self.allocations.items():
            age = current_time - allocation['timestamp']
            if age > min_age:
                leaks.append({
                    'object_id': obj_id,
                    'size': allocation['size'],
                    'source': allocation['source'],
                    'age': age,
                    'allocation_time': allocation['timestamp'],
                    'traceback': allocation['traceback']
                })
        
        return leaks
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        if not self.snapshots:
            return {}
        
        latest = self.snapshots[-1]
        earliest = self.snapshots[0]
        
        # 计算内存增长趋势
        if len(self.snapshots) >= 2:
            memory_trend = (latest.total_memory - self.snapshots[-2].total_memory) / 1024 / 1024
        else:
            memory_trend = 0
        
        return {
            'current_memory_mb': latest.total_memory / 1024 / 1024,
            'memory_growth_mb': memory_trend,
            'total_snapshots': len(self.snapshots),
            'leak_count': len(self.detect_leaks()),
            'allocation_count': len(self.allocations),
            'release_count': len(self.releases)
        }


# =============================================================================
# 性能分析器
# =============================================================================

class PerformanceProfiler:
    """性能分析器"""
    
    def __init__(self, debug_logger: 'DebugLogger'):
        """
        初始化性能分析器
        
        Args:
            debug_logger: 调试日志记录器实例
        """
        self.debug_logger = debug_logger
        self.metrics = {}
        self.active_profiles = {}
        self.thresholds = {}
        self.is_profiling = False
    
    def set_threshold(self, function_name: str, max_time: float, max_memory: float):
        """设置性能阈值"""
        self.thresholds[function_name] = {
            'max_time': max_time,
            'max_memory': max_memory
        }
    
    @contextmanager
    def profile_function(self, function_name: str, file_path: str, line_number: int):
        """函数性能分析上下文管理器"""
        start_time = time.time()
        process = psutil.Process()
        start_memory = process.memory_info().rss
        start_cpu = process.cpu_percent()
        
        profile_id = f"{function_name}_{start_time}"
        self.active_profiles[profile_id] = {
            'function_name': function_name,
            'file_path': file_path,
            'line_number': line_number,
            'start_time': start_time,
            'start_memory': start_memory,
            'start_cpu': start_cpu
        }
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = process.memory_info().rss
            end_cpu = process.cpu_percent()
            
            execution_time = end_time - start_time
            memory_usage = end_memory - start_memory
            cpu_usage = end_cpu - start_cpu
            
            # 记录性能数据
            self._update_metrics(function_name, file_path, line_number,
                               execution_time, memory_usage, cpu_usage)
            
            # 检查阈值
            self._check_thresholds(function_name, execution_time, memory_usage)
            
            # 清理活跃分析
            if profile_id in self.active_profiles:
                del self.active_profiles[profile_id]
    
    def _update_metrics(self, function_name: str, file_path: str, line_number: int,
                       execution_time: float, memory_usage: float, cpu_usage: float):
        """更新性能指标"""
        key = f"{file_path}:{line_number}:{function_name}"
        
        if key not in self.metrics:
            self.metrics[key] = PerformanceMetrics(function_name, file_path, line_number)
        
        self.metrics[key].update(execution_time, memory_usage, cpu_usage)
        
        # 记录性能日志
        self.debug_logger.log(LogLevel.PERFORMANCE, DebugCategory.PERFORMANCE,
                            f"函数 {function_name} 执行完成",
                            file_path=file_path,
                            line_number=line_number,
                            extra_data={
                                'execution_time': execution_time,
                                'memory_usage': memory_usage,
                                'cpu_usage': cpu_usage,
                                'call_count': self.metrics[key].call_count,
                                'avg_time': self.metrics[key].avg_time
                            })
    
    def _check_thresholds(self, function_name: str, execution_time: float, memory_usage: float):
        """检查性能阈值"""
        if function_name not in self.thresholds:
            return
        
        threshold = self.thresholds[function_name]
        
        if execution_time > threshold['max_time']:
            self.debug_logger.log(LogLevel.WARNING, DebugCategory.PERFORMANCE,
                                f"函数 {function_name} 执行时间超过阈值",
                                extra_data={
                                    'execution_time': execution_time,
                                    'threshold': threshold['max_time'],
                                    'overhead': execution_time - threshold['max_time']
                                })
        
        if memory_usage > threshold['max_memory']:
            self.debug_logger.log(LogLevel.WARNING, DebugCategory.PERFORMANCE,
                                f"函数 {function_name} 内存使用超过阈值",
                                extra_data={
                                    'memory_usage': memory_usage,
                                    'threshold': threshold['max_memory'],
                                    'overhead': memory_usage - threshold['max_memory']
                                })
    
    def get_function_metrics(self, function_name: str) -> Optional[Dict[str, Any]]:
        """获取函数性能指标"""
        for key, metrics in self.metrics.items():
            if metrics.function_name == function_name:
                return metrics.get_summary()
        return None
    
    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """获取所有性能指标"""
        return [metrics.get_summary() for metrics in self.metrics.values()]
    
    def get_slow_functions(self, min_time: float = 1.0) -> List[Dict[str, Any]]:
        """获取慢函数列表"""
        slow_functions = []
        for metrics in self.metrics.values():
            if metrics.avg_time > min_time:
                summary = metrics.get_summary()
                summary['is_slow'] = True
                slow_functions.append(summary)
        
        return sorted(slow_functions, key=lambda x: x['avg_time'], reverse=True)


# =============================================================================
# 网络调试器
# =============================================================================

class NetworkDebugger:
    """网络调试器"""
    
    def __init__(self, debug_logger: 'DebugLogger'):
        """
        初始化网络调试器
        
        Args:
            debug_logger: 调试日志记录器实例
        """
        self.debug_logger = debug_logger
        self.connections = {}
        self.requests = {}
        self.response_times = {}
        self.error_counts = defaultdict(int)
    
    async def monitor_connection(self, host: str, port: int, timeout: float = 30.0):
        """监控网络连接"""
        try:
            start_time = time.time()
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), timeout=timeout
            )
            end_time = time.time()
            
            connection_id = f"{host}:{port}"
            self.connections[connection_id] = {
                'host': host,
                'port': port,
                'start_time': start_time,
                'end_time': end_time,
                'response_time': end_time - start_time,
                'status': 'connected',
                'writer': writer,
                'reader': reader
            }
            
            self.debug_logger.log(LogLevel.INFO, DebugCategory.NETWORK,
                                f"成功连接到 {host}:{port}",
                                extra_data={
                                    'connection_id': connection_id,
                                    'response_time': end_time - start_time,
                                    'host': host,
                                    'port': port
                                })
            
            return connection_id
            
        except asyncio.TimeoutError:
            self.debug_logger.log(LogLevel.ERROR, DebugCategory.NETWORK,
                                f"连接 {host}:{port} 超时",
                                extra_data={
                                    'host': host,
                                    'port': port,
                                    'timeout': timeout,
                                    'error_type': 'timeout'
                                })
            raise NetworkDebugError(f"连接超时: {host}:{port}")
        
        except Exception as e:
            self.debug_logger.log(LogLevel.ERROR, DebugCategory.NETWORK,
                                f"连接 {host}:{port} 失败: {e}",
                                extra_data={
                                    'host': host,
                                    'port': port,
                                    'error': str(e),
                                    'error_type': type(e).__name__
                                })
            raise NetworkDebugError(f"连接失败: {host}:{port} - {e}")
    
    async def close_connection(self, connection_id: str):
        """关闭网络连接"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        try:
            if 'writer' in connection:
                connection['writer'].close()
                await connection['writer'].wait_closed()
            
            self.debug_logger.log(LogLevel.INFO, DebugCategory.NETWORK,
                                f"关闭连接 {connection_id}",
                                extra_data={
                                    'connection_id': connection_id,
                                    'duration': time.time() - connection['start_time']
                                })
            
            del self.connections[connection_id]
            
        except Exception as e:
            self.debug_logger.log(LogLevel.ERROR, DebugCategory.NETWORK,
                                f"关闭连接 {connection_id} 失败: {e}",
                                extra_data={
                                    'connection_id': connection_id,
                                    'error': str(e)
                                })
    
    async def make_http_request(self, method: str, url: str, headers: Optional[Dict] = None,
                              data: Optional[bytes] = None, timeout: float = 30.0) -> Dict[str, Any]:
        """发送HTTP请求"""
        try:
            import aiohttp
            
            start_time = time.time()
            request_id = f"{method}_{url}_{start_time}"
            
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, headers=headers, data=data, 
                                         timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    end_time = time.time()
                    response_time = end_time - start_time
                    
                    # 读取响应内容
                    content = await response.text()
                    
                    # 记录请求信息
                    self.requests[request_id] = {
                        'method': method,
                        'url': url,
                        'status_code': response.status,
                        'headers': dict(response.headers),
                        'response_time': response_time,
                        'content_length': len(content),
                        'start_time': start_time,
                        'end_time': end_time
                    }
                    
                    # 记录响应时间
                    domain = urlparse(url).netloc
                    if domain not in self.response_times:
                        self.response_times[domain] = []
                    self.response_times[domain].append(response_time)
                    
                    # 记录日志
                    self.debug_logger.log(LogLevel.INFO, DebugCategory.NETWORK,
                                        f"HTTP {method} {url} - {response.status}",
                                        extra_data={
                                            'request_id': request_id,
                                            'method': method,
                                            'url': url,
                                            'status_code': response.status,
                                            'response_time': response_time,
                                            'content_length': len(content),
                                            'domain': domain
                                        })
                    
                    # 检查响应时间
                    if response_time > 5.0:
                        self.debug_logger.log(LogLevel.WARNING, DebugCategory.NETWORK,
                                            f"HTTP请求响应时间过长: {response_time:.2f}s",
                                            extra_data={
                                                'request_id': request_id,
                                                'response_time': response_time,
                                                'threshold': 5.0
                                            })
                    
                    return {
                        'status': response.status,
                        'headers': dict(response.headers),
                        'content': content,
                        'response_time': response_time
                    }
        
        except asyncio.TimeoutError:
            self.error_counts['timeout'] += 1
            self.debug_logger.log(LogLevel.ERROR, DebugCategory.NETWORK,
                                f"HTTP请求超时: {method} {url}",
                                extra_data={
                                    'method': method,
                                    'url': url,
                                    'timeout': timeout,
                                    'error_type': 'timeout',
                                    'timeout_count': self.error_counts['timeout']
                                })
            raise NetworkDebugError(f"HTTP请求超时: {method} {url}")
        
        except Exception as e:
            self.error_counts['general'] += 1
            self.debug_logger.log(LogLevel.ERROR, DebugCategory.NETWORK,
                                f"HTTP请求失败: {method} {url} - {e}",
                                extra_data={
                                    'method': method,
                                    'url': url,
                                    'error': str(e),
                                    'error_type': type(e).__name__,
                                    'error_count': self.error_counts['general']
                                })
            raise NetworkDebugError(f"HTTP请求失败: {method} {url} - {e}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        if not self.connections:
            return {'active_connections': 0}
        
        total_response_time = sum(conn['response_time'] for conn in self.connections.values())
        avg_response_time = total_response_time / len(self.connections)
        
        return {
            'active_connections': len(self.connections),
            'avg_response_time': avg_response_time,
            'connections': list(self.connections.keys())
        }
    
    def get_request_stats(self) -> Dict[str, Any]:
        """获取请求统计信息"""
        if not self.requests:
            return {'total_requests': 0}
        
        total_requests = len(self.requests)
        status_codes = defaultdict(int)
        total_response_time = 0
        
        for request in self.requests.values():
            status_codes[request['status_code']] += 1
            total_response_time += request['response_time']
        
        avg_response_time = total_response_time / total_requests
        
        return {
            'total_requests': total_requests,
            'avg_response_time': avg_response_time,
            'status_codes': dict(status_codes),
            'error_counts': dict(self.error_counts)
        }


# =============================================================================
# 数据库调试器
# =============================================================================

class DatabaseDebugger:
    """数据库调试器"""
    
    def __init__(self, debug_logger: 'DebugLogger'):
        """
        初始化数据库调试器
        
        Args:
            debug_logger: 调试日志记录器实例
        """
        self.debug_logger = debug_logger
        self.connections = {}
        self.queries = {}
        self.query_times = defaultdict(list)
        self.slow_queries = []
        self.lock_waits = {}
    
    def monitor_connection(self, connection_id: str, db_path: str, connection_params: Dict[str, Any]):
        """监控数据库连接"""
        try:
            conn = sqlite3.connect(db_path)
            self.connections[connection_id] = {
                'connection': conn,
                'db_path': db_path,
                'params': connection_params,
                'created_time': datetime.now(),
                'query_count': 0,
                'total_time': 0.0
            }
            
            # 启用查询计划分析
            conn.execute("PRAGMA analyze")
            
            self.debug_logger.log(LogLevel.INFO, DebugCategory.DATABASE,
                                f"数据库连接已创建: {connection_id}",
                                extra_data={
                                    'connection_id': connection_id,
                                    'db_path': db_path,
                                    'params': connection_params
                                })
            
            return conn
            
        except Exception as e:
            self.debug_logger.log(LogLevel.ERROR, DebugCategory.DATABASE,
                                f"创建数据库连接失败: {connection_id} - {e}",
                                extra_data={
                                    'connection_id': connection_id,
                                    'db_path': db_path,
                                    'error': str(e)
                                })
            raise DatabaseDebugError(f"创建数据库连接失败: {connection_id} - {e}")
    
    def execute_query(self, connection_id: str, sql: str, params: Optional[Tuple] = None,
                     fetch_results: bool = False) -> Any:
        """执行SQL查询并监控"""
        if connection_id not in self.connections:
            raise DatabaseDebugError(f"连接不存在: {connection_id}")
        
        conn_info = self.connections[connection_id]
        conn = conn_info['connection']
        
        query_id = f"{connection_id}_{len(self.queries)}_{time.time()}"
        start_time = time.time()
        
        try:
            # 执行查询
            if params:
                cursor = conn.execute(sql, params)
            else:
                cursor = conn.execute(sql)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            # 获取结果
            results = None
            if fetch_results:
                results = cursor.fetchall()
                column_names = [desc[0] for desc in cursor.description] if cursor.description else []
            
            # 记录查询信息
            query_info = {
                'query_id': query_id,
                'sql': sql,
                'params': params,
                'execution_time': execution_time,
                'start_time': start_time,
                'end_time': end_time,
                'row_count': cursor.rowcount,
                'fetch_results': fetch_results
            }
            
            self.queries[query_id] = query_info
            
            # 更新连接统计
            conn_info['query_count'] += 1
            conn_info['total_time'] += execution_time
            
            # 记录响应时间
            domain = f"db_{connection_id}"
            self.query_times[domain].append(execution_time)
            
            # 检查慢查询
            if execution_time > 1.0:  # 1秒阈值
                self.slow_queries.append({
                    'query_id': query_id,
                    'sql': sql,
                    'execution_time': execution_time,
                    'timestamp': datetime.now()
                })
                
                self.debug_logger.log(LogLevel.WARNING, DebugCategory.DATABASE,
                                    f"慢查询检测: {execution_time:.3f}s",
                                    extra_data={
                                        'query_id': query_id,
                                        'sql': sql,
                                        'execution_time': execution_time,
                                        'threshold': 1.0
                                    })
            
            # 记录查询日志
            self.debug_logger.log(LogLevel.DEBUG, DebugCategory.DATABASE,
                                f"SQL执行完成: {sql[:100]}{'...' if len(sql) > 100 else ''}",
                                extra_data={
                                    'query_id': query_id,
                                    'sql': sql,
                                    'params': params,
                                    'execution_time': execution_time,
                                    'row_count': cursor.rowcount,
                                    'connection_id': connection_id
                                })
            
            return results
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            
            self.debug_logger.log(LogLevel.ERROR, DebugCategory.DATABASE,
                                f"SQL执行失败: {sql[:100]}{'...' if len(sql) > 100 else ''} - {e}",
                                extra_data={
                                    'query_id': query_id,
                                    'sql': sql,
                                    'params': params,
                                    'execution_time': execution_time,
                                    'error': str(e),
                                    'error_type': type(e).__name__,
                                    'connection_id': connection_id
                                })
            
            raise DatabaseDebugError(f"SQL执行失败: {sql[:100]}{'...' if len(sql) > 100 else ''} - {e}")
    
    def get_query_plan(self, connection_id: str, sql: str) -> Dict[str, Any]:
        """获取查询执行计划"""
        if connection_id not in self.connections:
            raise DatabaseDebugError(f"连接不存在: {connection_id}")
        
        conn = self.connections[connection_id]['connection']
        
        try:
            # 执行EXPLAIN QUERY PLAN
            cursor = conn.execute(f"EXPLAIN QUERY PLAN {sql}")
            plan_results = cursor.fetchall()
            
            plan_info = {
                'sql': sql,
                'plan': [],
                'estimated_cost': 0
            }
            
            for row in plan_results:
                plan_info['plan'].append({
                    'select_id': row[0],
                    'order': row[1],
                    'from': row[2],
                    'detail': row[3]
                })
            
            self.debug_logger.log(LogLevel.DEBUG, DebugCategory.DATABASE,
                                f"查询计划分析: {sql[:100]}{'...' if len(sql) > 100 else ''}",
                                extra_data={
                                    'sql': sql,
                                    'plan': plan_info['plan'],
                                    'connection_id': connection_id
                                })
            
            return plan_info
            
        except Exception as e:
            self.debug_logger.log(LogLevel.ERROR, DebugCategory.DATABASE,
                                f"获取查询计划失败: {sql[:100]}{'...' if len(sql) > 100 else ''} - {e}",
                                extra_data={
                                    'sql': sql,
                                    'error': str(e),
                                    'connection_id': connection_id
                                })
            raise DatabaseDebugError(f"获取查询计划失败: {sql[:100]}{'...' if len(sql) > 100 else ''} - {e}")
    
    def close_connection(self, connection_id: str):
        """关闭数据库连接"""
        if connection_id not in self.connections:
            return
        
        conn_info = self.connections[connection_id]
        try:
            conn_info['connection'].close()
            
            self.debug_logger.log(LogLevel.INFO, DebugCategory.DATABASE,
                                f"数据库连接已关闭: {connection_id}",
                                extra_data={
                                    'connection_id': connection_id,
                                    'query_count': conn_info['query_count'],
                                    'total_time': conn_info['total_time'],
                                    'duration': (datetime.now() - conn_info['created_time']).total_seconds()
                                })
            
            del self.connections[connection_id]
            
        except Exception as e:
            self.debug_logger.log(LogLevel.ERROR, DebugCategory.DATABASE,
                                f"关闭数据库连接失败: {connection_id} - {e}",
                                extra_data={
                                    'connection_id': connection_id,
                                    'error': str(e)
                                })
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        stats = {
            'active_connections': len(self.connections),
            'total_queries': len(self.queries),
            'slow_queries': len(self.slow_queries),
            'connections': []
        }
        
        for conn_id, conn_info in self.connections.items():
            avg_query_time = (conn_info['total_time'] / conn_info['query_count'] 
                            if conn_info['query_count'] > 0 else 0)
            
            stats['connections'].append({
                'connection_id': conn_id,
                'query_count': conn_info['query_count'],
                'total_time': conn_info['total_time'],
                'avg_query_time': avg_query_time,
                'duration': (datetime.now() - conn_info['created_time']).total_seconds()
            })
        
        return stats


# =============================================================================
# 调试工具集成器
# =============================================================================

class DebugToolIntegrator:
    """调试工具集成器"""
    
    def __init__(self, debug_logger: 'DebugLogger'):
        """
        初始化调试工具集成器
        
        Args:
            debug_logger: 调试日志记录器实例
        """
        self.debug_logger = debug_logger
        self.breakpoints = {}
        self.watch_variables = {}
        self.step_mode = False
        self.current_line = None
        self.call_stack = []
        self.variable_history = defaultdict(list)
    
    def set_breakpoint(self, file_path: str, line_number: int, condition: Optional[str] = None):
        """设置断点"""
        breakpoint_id = f"{file_path}:{line_number}"
        self.breakpoints[breakpoint_id] = {
            'file_path': file_path,
            'line_number': line_number,
            'condition': condition,
            'hit_count': 0,
            'enabled': True,
            'created_time': datetime.now()
        }
        
        self.debug_logger.log(LogLevel.INFO, DebugCategory.TOOL,
                            f"断点已设置: {file_path}:{line_number}",
                            extra_data={
                                'breakpoint_id': breakpoint_id,
                                'file_path': file_path,
                                'line_number': line_number,
                                'condition': condition
                            })
    
    def remove_breakpoint(self, file_path: str, line_number: int):
        """移除断点"""
        breakpoint_id = f"{file_path}:{line_number}"
        if breakpoint_id in self.breakpoints:
            del self.breakpoints[breakpoint_id]
            
            self.debug_logger.log(LogLevel.INFO, DebugCategory.TOOL,
                                f"断点已移除: {file_path}:{line_number}",
                                extra_data={
                                    'breakpoint_id': breakpoint_id,
                                    'file_path': file_path,
                                    'line_number': line_number
                                })
    
    def check_breakpoint(self, file_path: str, line_number: int, locals_dict: Dict[str, Any]):
        """检查断点"""
        breakpoint_id = f"{file_path}:{line_number}"
        
        if breakpoint_id not in self.breakpoints:
            return False
        
        breakpoint = self.breakpoints[breakpoint_id]
        if not breakpoint['enabled']:
            return False
        
        # 检查条件
        if breakpoint['condition']:
            try:
                # 简单的条件评估（在实际应用中需要更安全的实现）
                result = eval(breakpoint['condition'], {}, locals_dict)
                if not result:
                    return False
            except Exception as e:
                self.debug_logger.log(LogLevel.ERROR, DebugCategory.TOOL,
                                    f"断点条件评估失败: {e}",
                                    extra_data={
                                        'breakpoint_id': breakpoint_id,
                                        'condition': breakpoint['condition'],
                                        'error': str(e)
                                    })
                return False
        
        # 命中断点
        breakpoint['hit_count'] += 1
        
        self.debug_logger.log(LogLevel.INFO, DebugCategory.TOOL,
                            f"断点命中: {file_path}:{line_number}",
                            extra_data={
                                'breakpoint_id': breakpoint_id,
                                'file_path': file_path,
                                'line_number': line_number,
                                'hit_count': breakpoint['hit_count'],
                                'locals': {k: str(v) for k, v in locals_dict.items() if not k.startswith('__')}
                            })
        
        return True
    
    def watch_variable(self, variable_name: str, expression: str):
        """监视变量"""
        self.watch_variables[variable_name] = {
            'expression': expression,
            'history': [],
            'created_time': datetime.now()
        }
        
        self.debug_logger.log(LogLevel.INFO, DebugCategory.TOOL,
                            f"变量监视已设置: {variable_name} = {expression}",
                            extra_data={
                                'variable_name': variable_name,
                                'expression': expression
                            })
    
    def update_watch_variables(self, locals_dict: Dict[str, Any], globals_dict: Dict[str, Any]):
        """更新监视变量"""
        for var_name, watch_info in self.watch_variables.items():
            try:
                value = eval(watch_info['expression'], globals_dict, locals_dict)
                watch_info['history'].append({
                    'value': value,
                    'timestamp': datetime.now()
                })
                
                # 限制历史记录长度
                if len(watch_info['history']) > 100:
                    watch_info['history'] = watch_info['history'][-100:]
                
                # 记录变量变化
                self.debug_logger.log(LogLevel.DEBUG, DebugCategory.TOOL,
                                    f"变量变化: {var_name} = {value}",
                                    extra_data={
                                        'variable_name': var_name,
                                        'expression': watch_info['expression'],
                                        'value': str(value),
                                        'timestamp': datetime.now().isoformat()
                                    })
                
            except Exception as e:
                self.debug_logger.log(LogLevel.ERROR, DebugCategory.TOOL,
                                    f"变量监视失败: {var_name} - {e}",
                                    extra_data={
                                        'variable_name': var_name,
                                        'expression': watch_info['expression'],
                                        'error': str(e)
                                    })
    
    def enable_step_mode(self):
        """启用单步执行模式"""
        self.step_mode = True
        self.debug_logger.log(LogLevel.INFO, DebugCategory.TOOL,
                            "单步执行模式已启用")
    
    def disable_step_mode(self):
        """禁用单步执行模式"""
        self.step_mode = False
        self.debug_logger.log(LogLevel.INFO, DebugCategory.TOOL,
                            "单步执行模式已禁用")
    
    def step_into(self, file_path: str, line_number: int, locals_dict: Dict[str, Any]):
        """单步进入"""
        self.debug_logger.log(LogLevel.INFO, DebugCategory.TOOL,
                            f"单步进入: {file_path}:{line_number}",
                            extra_data={
                                'action': 'step_into',
                                'file_path': file_path,
                                'line_number': line_number,
                                'locals': {k: str(v) for k, v in locals_dict.items() if not k.startswith('__')}
                            })
    
    def step_over(self, file_path: str, line_number: int, locals_dict: Dict[str, Any]):
        """单步跳过"""
        self.debug_logger.log(LogLevel.INFO, DebugCategory.TOOL,
                            f"单步跳过: {file_path}:{line_number}",
                            extra_data={
                                'action': 'step_over',
                                'file_path': file_path,
                                'line_number': line_number,
                                'locals': {k: str(v) for k, v in locals_dict.items() if not k.startswith('__')}
                            })
    
    def step_out(self, file_path: str, line_number: int, locals_dict: Dict[str, Any]):
        """单步跳出"""
        self.debug_logger.log(LogLevel.INFO, DebugCategory.TOOL,
                            f"单步跳出: {file_path}:{line_number}",
                            extra_data={
                                'action': 'step_out',
                                'file_path': file_path,
                                'line_number': line_number,
                                'locals': {k: str(v) for k, v in locals_dict.items() if not k.startswith('__')}
                            })
    
    def get_debug_info(self) -> Dict[str, Any]:
        """获取调试信息"""
        return {
            'breakpoints': self.breakpoints,
            'watch_variables': {k: {
                'expression': v['expression'],
                'history_count': len(v['history']),
                'created_time': v['created_time'].isoformat()
            } for k, v in self.watch_variables.items()},
            'step_mode': self.step_mode,
            'call_stack_depth': len(self.call_stack)
        }


# =============================================================================
# 主要的调试日志记录器类
# =============================================================================

class DebugLogger:
    """L7调试日志记录器主类"""
    
    def __init__(self, 
                 name: str = "DebugLogger",
                 log_level: str = LogLevel.DEBUG,
                 async_mode: bool = True,
                 max_log_entries: int = 10000):
        """
        初始化调试日志记录器
        
        Args:
            name: 日志记录器名称
            log_level: 日志级别
            async_mode: 是否启用异步模式
            max_log_entries: 最大日志条目数
        """
        self.name = name
        self.log_level = log_level
        self.async_mode = async_mode
        self.max_log_entries = max_log_entries
        
        # 日志存储
        self.log_entries = deque(maxlen=max_log_entries)
        self.handlers = []
        
        # 异步处理器
        if async_mode:
            self.async_handler = AsyncLogHandler()
        
        # 子系统
        self.memory_monitor = MemoryMonitor(self)
        self.performance_profiler = PerformanceProfiler(self)
        self.network_debugger = NetworkDebugger(self)
        self.database_debugger = DatabaseDebugger(self)
        self.debug_tool_integrator = DebugToolIntegrator(self)
        
        # 配置
        self.config = {
            'enable_memory_monitoring': True,
            'enable_performance_profiling': True,
            'enable_network_debugging': True,
            'enable_database_debugging': True,
            'enable_debug_tools': True,
            'auto_start_monitors': False
        }
        
        # 状态
        self.is_initialized = False
        self.start_time = datetime.now()
        
        # 添加默认处理器
        self.add_handler(ConsoleLogHandler())
    
    async def initialize(self):
        """初始化日志记录器"""
        if self.is_initialized:
            return
        
        # 启动异步处理器
        if self.async_mode:
            await self.async_handler.start()
            self.async_handler.add_handler(FileLogHandler(f"logs/{self.name}.log"))
            self.async_handler.add_handler(DatabaseLogHandler(f"logs/{self.name}.db"))
        
        # 自动启动监控器
        if self.config['auto_start_monitors']:
            await self.start_monitoring()
        
        self.is_initialized = True
        
        self.log(LogLevel.INFO, DebugCategory.GENERAL,
                f"调试日志记录器 {self.name} 已初始化",
                extra_data={
                    'async_mode': self.async_mode,
                    'start_time': self.start_time.isoformat(),
                    'config': self.config
                })
    
    async def shutdown(self):
        """关闭日志记录器"""
        if not self.is_initialized:
            return
        
        # 停止监控器
        await self.stop_monitoring()
        
        # 停止异步处理器
        if self.async_mode:
            await self.async_handler.stop()
        
        self.is_initialized = False
        
        self.log(LogLevel.INFO, DebugCategory.GENERAL,
                f"调试日志记录器 {self.name} 已关闭")
    
    def configure(self, **kwargs):
        """配置日志记录器"""
        for key, value in kwargs.items():
            if key in self.config:
                old_value = self.config[key]
                self.config[key] = value
                
                self.log(LogLevel.INFO, DebugCategory.GENERAL,
                        f"配置已更新: {key} = {value}",
                        extra_data={
                            'config_key': key,
                            'old_value': old_value,
                            'new_value': value
                        })
    
    def add_handler(self, handler: BaseLogHandler):
        """添加日志处理器"""
        self.handlers.append(handler)
        
        self.log(LogLevel.INFO, DebugCategory.GENERAL,
                f"日志处理器已添加: {handler.name}",
                extra_data={
                    'handler_name': handler.name,
                    'handler_type': type(handler).__name__
                })
    
    def remove_handler(self, handler_name: str):
        """移除日志处理器"""
        self.handlers = [h for h in self.handlers if h.name != handler_name]
        
        self.log(LogLevel.INFO, DebugCategory.GENERAL,
                f"日志处理器已移除: {handler_name}")
    
    def log(self, level: str, category: str, message: str, 
           file_path: Optional[str] = None, line_number: Optional[int] = None,
           function_name: Optional[str] = None, extra_data: Optional[Dict[str, Any]] = None):
        """
        记录日志条目
        
        Args:
            level: 日志级别
            category: 调试类别
            message: 日志消息
            file_path: 文件路径
            line_number: 行号
            function_name: 函数名
            extra_data: 额外数据
        """
        # 获取调用信息
        if not file_path or not line_number or not function_name:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_frame = frame.f_back
                file_path = file_path or caller_frame.f_code.co_filename
                line_number = line_number or caller_frame.f_lineno
                function_name = function_name or caller_frame.f_code.co_name
        
        # 创建日志条目
        entry = LogEntry(
            level=level,
            category=category,
            message=message,
            file_path=file_path,
            line_number=line_number,
            function_name=function_name,
            extra_data=extra_data
        )
        
        # 存储日志条目
        self.log_entries.append(entry)
        
        # 异步处理
        if self.async_mode:
            try:
                asyncio.create_task(self.async_handler.put(entry))
            except RuntimeError:
                # 在没有事件循环的情况下，无法创建异步任务
                # 这种情况在模块初始化时可能发生
                import sys
                print(f"警告：无法创建异步任务，可能没有运行中的事件循环", file=sys.stderr)
        else:
            # 同步处理
            for handler in self.handlers:
                try:
                    if hasattr(handler, 'handle'):
                        # 同步处理器
                        try:
                            loop = asyncio.get_event_loop()
                        except RuntimeError:
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                        loop.run_until_complete(handler.handle([entry]))
                    else:
                        # 传统处理器
                        handler.emit(entry.to_dict() if hasattr(entry, 'to_dict') else str(entry))
                except Exception as e:
                    print(f"日志处理器错误: {e}")
    
    def debug(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """记录调试日志"""
        self.log(LogLevel.DEBUG, DebugCategory.CODE, message, extra_data=extra_data)
    
    def info(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """记录信息日志"""
        self.log(LogLevel.INFO, DebugCategory.GENERAL, message, extra_data=extra_data)
    
    def warning(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """记录警告日志"""
        self.log(LogLevel.WARNING, DebugCategory.GENERAL, message, extra_data=extra_data)
    
    def error(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """记录错误日志"""
        self.log(LogLevel.ERROR, DebugCategory.GENERAL, message, extra_data=extra_data)
    
    def critical(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """记录严重错误日志"""
        self.log(LogLevel.CRITICAL, DebugCategory.GENERAL, message, extra_data=extra_data)
    
    def trace(self, message: str, extra_data: Optional[Dict[str, Any]] = None):
        """记录跟踪日志"""
        self.log(LogLevel.TRACE, DebugCategory.CODE, message, extra_data=extra_data)
    
    @contextmanager
    def code_debug_context(self, context_name: str, log_variables: bool = True):
        """代码调试上下文管理器"""
        frame = inspect.currentframe()
        if frame and frame.f_back:
            caller_frame = frame.f_back
            file_path = caller_frame.f_code.co_filename
            line_number = caller_frame.f_lineno
            function_name = caller_frame.f_code.co_name
            
            self.log(LogLevel.DEBUG, DebugCategory.CODE,
                    f"进入调试上下文: {context_name}",
                    file_path=file_path,
                    line_number=line_number,
                    function_name=function_name,
                    extra_data={
                        'context_name': context_name,
                        'action': 'enter'
                    })
            
            if log_variables and caller_frame.f_locals:
                self.log(LogLevel.TRACE, DebugCategory.CODE,
                        f"上下文变量: {context_name}",
                        file_path=file_path,
                        line_number=line_number,
                        function_name=function_name,
                        extra_data={
                            'context_name': context_name,
                            'variables': {k: str(v) for k, v in caller_frame.f_locals.items() 
                                        if not k.startswith('__')}
                        })
        
        try:
            yield
        finally:
            if frame and frame.f_back:
                caller_frame = frame.f_back
                file_path = caller_frame.f_code.co_filename
                line_number = caller_frame.f_lineno
                function_name = caller_frame.f_code.co_name
                
                self.log(LogLevel.DEBUG, DebugCategory.CODE,
                        f"退出调试上下文: {context_name}",
                        file_path=file_path,
                        line_number=line_number,
                        function_name=function_name,
                        extra_data={
                            'context_name': context_name,
                            'action': 'exit'
                        })
    
    @contextmanager
    def performance_profile(self, function_name: str, file_path: str = None, line_number: int = None):
        """性能分析上下文管理器"""
        if not file_path or not line_number:
            frame = inspect.currentframe()
            if frame and frame.f_back:
                caller_frame = frame.f_back
                file_path = file_path or caller_frame.f_code.co_filename
                line_number = line_number or caller_frame.f_lineno
        
        with self.performance_profiler.profile_function(function_name, file_path, line_number):
            yield
    
    async def start_monitoring(self):
        """开始监控"""
        if self.config['enable_memory_monitoring']:
            await self.memory_monitor.start_monitoring()
        
        self.log(LogLevel.INFO, DebugCategory.GENERAL,
                "调试监控已启动",
                extra_data={
                    'memory_monitoring': self.config['enable_memory_monitoring'],
                    'performance_profiling': self.config['enable_performance_profiling'],
                    'network_debugging': self.config['enable_network_debugging'],
                    'database_debugging': self.config['enable_database_debugging']
                })
    
    async def stop_monitoring(self):
        """停止监控"""
        if self.config['enable_memory_monitoring']:
            await self.memory_monitor.stop_monitoring()
        
        self.log(LogLevel.INFO, DebugCategory.GENERAL,
                "调试监控已停止")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        return self.memory_monitor.get_memory_statistics()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        return {
            'all_metrics': self.performance_profiler.get_all_metrics(),
            'slow_functions': self.performance_profiler.get_slow_functions(),
            'active_profiles': len(self.performance_profiler.active_profiles)
        }
    
    def get_network_stats(self) -> Dict[str, Any]:
        """获取网络统计信息"""
        return {
            'connection_stats': self.network_debugger.get_connection_stats(),
            'request_stats': self.network_debugger.get_request_stats()
        }
    
    def get_database_stats(self) -> Dict[str, Any]:
        """获取数据库统计信息"""
        return self.database_debugger.get_connection_stats()
    
    def get_debug_tool_stats(self) -> Dict[str, Any]:
        """获取调试工具统计信息"""
        return self.debug_tool_integrator.get_debug_info()
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有统计信息"""
        return {
            'memory': self.get_memory_stats(),
            'performance': self.get_performance_stats(),
            'network': self.get_network_stats(),
            'database': self.get_database_stats(),
            'debug_tools': self.get_debug_tool_stats(),
            'system': {
                'uptime': (datetime.now() - self.start_time).total_seconds(),
                'log_entries_count': len(self.log_entries),
                'is_initialized': self.is_initialized,
                'async_mode': self.async_mode
            }
        }
    
    def export_logs(self, file_path: str, format: str = 'json', 
                   level_filter: Optional[str] = None,
                   category_filter: Optional[str] = None,
                   time_range: Optional[Tuple[datetime, datetime]] = None) -> int:
        """
        导出日志
        
        Args:
            file_path: 导出文件路径
            format: 导出格式 ('json', 'csv', 'txt')
            level_filter: 日志级别过滤
            category_filter: 类别过滤
            time_range: 时间范围过滤
            
        Returns:
            导出的日志条目数量
        """
        filtered_entries = []
        
        for entry in self.log_entries:
            # 应用过滤器
            if level_filter and entry.level != level_filter:
                continue
            
            if category_filter and entry.category != category_filter:
                continue
            
            if time_range:
                start_time, end_time = time_range
                if not (start_time <= entry.timestamp <= end_time):
                    continue
            
            filtered_entries.append(entry)
        
        # 导出文件
        file_path = Path(file_path)
        if not file_path.parent.exists():
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump([entry.to_dict() for entry in filtered_entries], 
                         f, ensure_ascii=False, indent=2, default=str)
        
        elif format.lower() == 'csv':
            import csv
            with open(file_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'level', 'category', 'message', 
                               'file_path', 'line_number', 'function_name', 'thread_id'])
                for entry in filtered_entries:
                    writer.writerow([
                        entry.timestamp.isoformat(),
                        entry.level,
                        entry.category,
                        entry.message,
                        entry.file_path,
                        entry.line_number,
                        entry.function_name,
                        entry.thread_id
                    ])
        
        elif format.lower() == 'txt':
            with open(file_path, 'w', encoding='utf-8') as f:
                for entry in filtered_entries:
                    f.write(str(entry) + '\n')
        
        self.log(LogLevel.INFO, DebugCategory.GENERAL,
                f"日志已导出到 {file_path}",
                extra_data={
                    'export_path': str(file_path),
                    'export_format': format,
                    'exported_count': len(filtered_entries),
                    'total_count': len(self.log_entries)
                })
        
        return len(filtered_entries)
    
    def clear_logs(self):
        """清空日志"""
        self.log_entries.clear()
        self.log(LogLevel.INFO, DebugCategory.GENERAL,
                "日志已清空")
    
    def set_memory_threshold(self, threshold_bytes: int):
        """设置内存阈值"""
        self.memory_monitor.memory_threshold = threshold_bytes
        self.log(LogLevel.INFO, DebugCategory.MEMORY,
                f"内存阈值已设置为 {threshold_bytes / 1024 / 1024:.2f}MB",
                extra_data={
                    'threshold_bytes': threshold_bytes,
                    'threshold_mb': threshold_bytes / 1024 / 1024
                })
    
    def set_performance_threshold(self, function_name: str, max_time: float, max_memory: float):
        """设置性能阈值"""
        self.performance_profiler.set_threshold(function_name, max_time, max_memory)
        self.log(LogLevel.INFO, DebugCategory.PERFORMANCE,
                f"性能阈值已设置: {function_name}",
                extra_data={
                    'function_name': function_name,
                    'max_time': max_time,
                    'max_memory': max_memory
                })


# =============================================================================
# 装饰器函数
# =============================================================================

def debug_logger(logger: DebugLogger = None, log_level: str = LogLevel.DEBUG):
    """
    调试日志装饰器
    
    Args:
        logger: 调试日志记录器实例
        log_level: 日志级别
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not logger:
                return await func(*args, **kwargs)
            
            start_time = time.time()
            file_path = inspect.getfile(func)
            line_number = inspect.getsourcelines(func)[1]
            function_name = func.__name__
            
            logger.log(log_level, DebugCategory.CODE,
                      f"开始执行函数: {function_name}",
                      file_path=file_path,
                      line_number=line_number,
                      function_name=function_name,
                      extra_data={
                          'function_name': function_name,
                          'args_count': len(args),
                          'kwargs_count': len(kwargs),
                          'action': 'function_start'
                      })
            
            try:
                result = await func(*args, **kwargs)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                logger.log(log_level, DebugCategory.CODE,
                          f"函数执行成功: {function_name}",
                          file_path=file_path,
                          line_number=line_number,
                          function_name=function_name,
                          extra_data={
                              'function_name': function_name,
                              'execution_time': execution_time,
                              'action': 'function_success',
                              'return_type': type(result).__name__
                          })
                
                return result
                
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                
                logger.log(LogLevel.ERROR, DebugCategory.CODE,
                          f"函数执行失败: {function_name} - {e}",
                          file_path=file_path,
                          line_number=line_number,
                          function_name=function_name,
                          extra_data={
                              'function_name': function_name,
                              'execution_time': execution_time,
                              'error': str(e),
                              'error_type': type(e).__name__,
                              'traceback': traceback.format_exc(),
                              'action': 'function_error'
                          })
                
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not logger:
                return func(*args, **kwargs)
            
            start_time = time.time()
            file_path = inspect.getfile(func)
            line_number = inspect.getsourcelines(func)[1]
            function_name = func.__name__
            
            logger.log(log_level, DebugCategory.CODE,
                      f"开始执行函数: {function_name}",
                      file_path=file_path,
                      line_number=line_number,
                      function_name=function_name,
                      extra_data={
                          'function_name': function_name,
                          'args_count': len(args),
                          'kwargs_count': len(kwargs),
                          'action': 'function_start'
                      })
            
            try:
                result = func(*args, **kwargs)
                
                end_time = time.time()
                execution_time = end_time - start_time
                
                logger.log(log_level, DebugCategory.CODE,
                          f"函数执行成功: {function_name}",
                          file_path=file_path,
                          line_number=line_number,
                          function_name=function_name,
                          extra_data={
                              'function_name': function_name,
                              'execution_time': execution_time,
                              'action': 'function_success',
                              'return_type': type(result).__name__
                          })
                
                return result
                
            except Exception as e:
                end_time = time.time()
                execution_time = end_time - start_time
                
                logger.log(LogLevel.ERROR, DebugCategory.CODE,
                          f"函数执行失败: {function_name} - {e}",
                          file_path=file_path,
                          line_number=line_number,
                          function_name=function_name,
                          extra_data={
                              'function_name': function_name,
                              'execution_time': execution_time,
                              'error': str(e),
                              'error_type': type(e).__name__,
                              'traceback': traceback.format_exc(),
                              'action': 'function_error'
                          })
                
                raise
        
        # 根据函数是否为异步选择包装器
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


def performance_profile(logger: DebugLogger = None, threshold: float = 1.0):
    """
    性能分析装饰器
    
    Args:
        logger: 调试日志记录器实例
        threshold: 性能阈值（秒）
    """
    def decorator(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            if not logger:
                return await func(*args, **kwargs)
            
            with logger.performance_profile(func.__name__):
                result = await func(*args, **kwargs)
                
                # 检查性能阈值
                metrics = logger.performance_profiler.get_function_metrics(func.__name__)
                if metrics and metrics['avg_time'] > threshold:
                    logger.log(LogLevel.WARNING, DebugCategory.PERFORMANCE,
                              f"函数性能超过阈值: {func.__name__}",
                              extra_data={
                                  'function_name': func.__name__,
                                  'avg_time': metrics['avg_time'],
                                  'threshold': threshold,
                                  'overhead': metrics['avg_time'] - threshold
                              })
                
                return result
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            if not logger:
                return func(*args, **kwargs)
            
            with logger.performance_profile(func.__name__):
                result = func(*args, **kwargs)
                
                # 检查性能阈值
                metrics = logger.performance_profiler.get_function_metrics(func.__name__)
                if metrics and metrics['avg_time'] > threshold:
                    logger.log(LogLevel.WARNING, DebugCategory.PERFORMANCE,
                              f"函数性能超过阈值: {func.__name__}",
                              extra_data={
                                  'function_name': func.__name__,
                                  'avg_time': metrics['avg_time'],
                                  'threshold': threshold,
                                  'overhead': metrics['avg_time'] - threshold
                              })
                
                return result
        
        # 根据函数是否为异步选择包装器
        if inspect.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


# =============================================================================
# 使用示例和测试代码
# =============================================================================

async def example_usage():
    """使用示例"""
    # 创建调试日志记录器
    logger = DebugLogger(
        name="ExampleLogger",
        log_level=LogLevel.DEBUG,
        async_mode=True
    )
    
    try:
        # 初始化
        await logger.initialize()
        
        # 配置
        logger.configure(
            enable_memory_monitoring=True,
            enable_performance_profiling=True,
            auto_start_monitors=True
        )
        
        # 设置阈值
        logger.set_memory_threshold(50 * 1024 * 1024)  # 50MB
        logger.set_performance_threshold("example_function", 0.1, 1024 * 1024)  # 0.1秒, 1MB
        
        # 基本日志记录
        logger.debug("这是一条调试信息")
        logger.info("这是一条信息日志")
        logger.warning("这是一条警告信息")
        
        # 代码调试上下文
        with logger.code_debug_context("示例上下文"):
            local_var = "测试变量"
            logger.trace(f"局部变量: {local_var}")
        
        # 性能分析
        with logger.performance_profile("示例函数"):
            time.sleep(0.05)  # 模拟耗时操作
        
        # 使用装饰器
        @debug_logger(logger)
        @performance_profile(logger, threshold=0.01)
        def example_function(x: int, y: int) -> int:
            """示例函数"""
            time.sleep(0.02)  # 模拟计算
            return x + y
        
        result = example_function(10, 20)
        logger.info(f"函数结果: {result}")
        
        # 网络调试示例
        try:
            # 模拟网络请求
            connection_id = await logger.network_debugger.monitor_connection("httpbin.org", 80)
            response = await logger.network_debugger.make_http_request("GET", "http://httpbin.org/get")
            await logger.network_debugger.close_connection(connection_id)
        except Exception as e:
            logger.warning(f"网络请求失败: {e}")
        
        # 数据库调试示例
        try:
            conn_id = logger.database_debugger.monitor_connection("test_db", "test.db", {})
            logger.database_debugger.execute_query(conn_id, "CREATE TABLE IF NOT EXISTS test (id INTEGER, name TEXT)")
            logger.database_debugger.execute_query(conn_id, "INSERT INTO test (id, name) VALUES (1, 'test')")
            results = logger.database_debugger.execute_query(conn_id, "SELECT * FROM test", fetch_results=True)
            logger.database_debugger.close_connection(conn_id)
        except Exception as e:
            logger.warning(f"数据库操作失败: {e}")
        
        # 调试工具示例
        logger.debug_tool_integrator.set_breakpoint(__file__, 1200)
        logger.debug_tool_integrator.watch_variable("result", "result")
        logger.debug_tool_integrator.enable_step_mode()
        
        # 获取统计信息
        stats = logger.get_all_stats()
        print("\n=== 调试统计信息 ===")
        print(json.dumps(stats, indent=2, ensure_ascii=False, default=str))
        
        # 导出日志
        exported_count = logger.export_logs("logs/example_logs.json", format="json")
        print(f"\n已导出 {exported_count} 条日志记录")
        
    finally:
        # 关闭
        await logger.shutdown()


# =============================================================================
# 模块初始化
# =============================================================================

# 创建默认实例（延迟初始化以避免事件循环问题）
_default_logger = None

def get_default_logger() -> DebugLogger:
    """获取默认调试日志记录器实例（延迟初始化）"""
    global _default_logger
    if _default_logger is None:
        _default_logger = DebugLogger(name="DefaultLogger", async_mode=False)  # 默认为同步模式
    return _default_logger

# 保持向后兼容性
default_logger = None  # 将在首次使用时初始化

# 便捷函数
def debug(message: str, extra_data: Optional[Dict[str, Any]] = None):
    """记录调试日志（使用默认实例）"""
    logger = get_default_logger()
    logger.debug(message, extra_data)

def info(message: str, extra_data: Optional[Dict[str, Any]] = None):
    """记录信息日志（使用默认实例）"""
    logger = get_default_logger()
    logger.info(message, extra_data)

def warning(message: str, extra_data: Optional[Dict[str, Any]] = None):
    """记录警告日志（使用默认实例）"""
    logger = get_default_logger()
    logger.warning(message, extra_data)

def error(message: str, extra_data: Optional[Dict[str, Any]] = None):
    """记录错误日志（使用默认实例）"""
    logger = get_default_logger()
    logger.error(message, extra_data)

def critical(message: str, extra_data: Optional[Dict[str, Any]] = None):
    """记录严重错误日志（使用默认实例）"""
    logger = get_default_logger()
    logger.critical(message, extra_data)


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())