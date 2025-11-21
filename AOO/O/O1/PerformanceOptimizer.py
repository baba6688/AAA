#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O1性能优化器模块

该模块提供了全面的性能分析和优化功能，包括：
1. 系统性能分析和瓶颈识别（CPU、内存、I/O、网络）
2. 代码性能分析和优化建议（函数耗时、内存泄漏、热点代码）
3. 数据库性能优化（查询优化、索引优化、连接池优化）
4. 缓存策略优化（缓存命中率、缓存失效策略）
5. 并发性能优化（线程池优化、锁竞争优化）
6. 性能监控和告警系统
7. 异步性能优化处理
8. 完整的错误处理和日志记录

Author: O1 Performance Optimization Team
Version: 1.0.0
Date: 2025-11-06
"""

import asyncio
import gc
import inspect
import logging
import os
import psutil
import sqlite3
import threading
import time
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, 
    AsyncGenerator, AsyncIterator, Set, NamedTuple
)
from collections import defaultdict, deque
from queue import Queue, Empty
import json
import weakref
import tracemalloc
import cProfile
import pstats
import io
import sys
import resource
import socket
import subprocess
import hashlib
import pickle
import threading
from threading import Lock, RLock, Event, Condition
from multiprocessing import cpu_count
import concurrent.futures
import functools
import linecache
import dis
import types
import weakref


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance_optimizer.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# 数据结构定义
@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read: float
    disk_io_write: float
    network_sent: float
    network_recv: float
    thread_count: int
    gc_collections: Dict[int, int] = field(default_factory=dict)
    gc_objects: int = 0


@dataclass
class FunctionProfile:
    """函数性能分析数据类"""
    function_name: str
    call_count: int
    total_time: float
    average_time: float
    min_time: float
    max_time: float
    file_path: str
    line_number: int
    memory_delta: int = 0


@dataclass
class DatabaseMetrics:
    """数据库性能指标数据类"""
    query_count: int
    average_query_time: float
    slow_queries: List[Tuple[str, float]]
    connection_count: int
    cache_hit_rate: float
    index_usage: Dict[str, float]
    lock_wait_time: float


@dataclass
class CacheMetrics:
    """缓存性能指标数据类"""
    hit_count: int
    miss_count: int
    hit_rate: float
    eviction_count: int
    average_access_time: float
    memory_usage: int


@dataclass
class ConcurrencyMetrics:
    """并发性能指标数据类"""
    active_threads: int
    thread_pool_size: int
    queue_size: int
    lock_contention: int
    average_wait_time: float
    throughput: float


@dataclass
class Alert:
    """告警数据类"""
    id: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    timestamp: datetime
    source: str
    details: Dict[str, Any] = field(default_factory=dict)


class SystemProfiler:
    """系统性能分析器"""
    
    def __init__(self, interval: float = 1.0):
        """
        初始化系统性能分析器
        
        Args:
            interval: 采样间隔（秒）
        """
        self.interval = interval
        self.is_monitoring = False
        self.metrics_history: deque = deque(maxlen=1000)
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = Lock()
        
    def start_monitoring(self):
        """开始系统性能监控"""
        if self.is_monitoring:
            logger.warning("系统性能监控已在运行中")
            return
            
        self.is_monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("系统性能监控已启动")
        
    def stop_monitoring(self):
        """停止系统性能监控"""
        self.is_monitoring = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5)
        logger.info("系统性能监控已停止")
        
    def _monitor_loop(self):
        """监控循环"""
        while self.is_monitoring:
            try:
                metrics = self._collect_metrics()
                with self._lock:
                    self.metrics_history.append(metrics)
                time.sleep(self.interval)
            except Exception as e:
                logger.error(f"系统性能监控错误: {e}")
                time.sleep(self.interval)
                
    def _collect_metrics(self) -> PerformanceMetrics:
        """收集系统性能指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=None)
        
        # 内存使用情况
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_mb = memory.used / 1024 / 1024
        
        # 磁盘I/O
        disk_io = psutil.disk_io_counters()
        disk_io_read = disk_io.read_bytes if disk_io else 0
        disk_io_write = disk_io.write_bytes if disk_io else 0
        
        # 网络I/O
        network_io = psutil.net_io_counters()
        network_sent = network_io.bytes_sent if network_io else 0
        network_recv = network_io.bytes_recv if network_io else 0
        
        # 线程数量
        thread_count = threading.active_count()
        
        # 垃圾回收统计
        gc_stats = gc.get_stats()
        gc_collections = {}
        for i, stat in enumerate(gc_stats):
            gc_collections[i] = stat['collections']
            
        gc_objects = len(gc.get_objects())
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_mb=memory_used_mb,
            disk_io_read=disk_io_read,
            disk_io_write=disk_io_write,
            network_sent=network_sent,
            network_recv=network_recv,
            thread_count=thread_count,
            gc_collections=gc_collections,
            gc_objects=gc_objects
        )
        
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """获取当前性能指标"""
        with self._lock:
            if self.metrics_history:
                return self.metrics_history[-1]
            return None
            
    def get_metrics_history(self, duration: timedelta = timedelta(minutes=5)) -> List[PerformanceMetrics]:
        """获取历史性能指标"""
        cutoff_time = datetime.now() - duration
        with self._lock:
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
            
    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """分析系统瓶颈"""
        metrics_history = self.get_metrics_history()
        if not metrics_history:
            return {"error": "没有足够的性能数据进行分析"}
            
        # 分析CPU瓶颈
        cpu_usage = [m.cpu_percent for m in metrics_history]
        avg_cpu = sum(cpu_usage) / len(cpu_usage)
        max_cpu = max(cpu_usage)
        
        # 分析内存瓶颈
        memory_usage = [m.memory_percent for m in metrics_history]
        avg_memory = sum(memory_usage) / len(memory_usage)
        max_memory = max(memory_usage)
        
        # 分析I/O瓶颈
        disk_io_rates = []
        network_io_rates = []
        for i in range(1, len(metrics_history)):
            current = metrics_history[i]
            previous = metrics_history[i-1]
            time_diff = (current.timestamp - previous.timestamp).total_seconds()
            
            if time_diff > 0:
                disk_rate = (current.disk_io_read + current.disk_io_write - 
                           previous.disk_io_read - previous.disk_io_write) / time_diff
                network_rate = (current.network_sent + current.network_recv - 
                              previous.network_sent - previous.network_recv) / time_diff
                disk_io_rates.append(disk_rate)
                network_io_rates.append(network_rate)
                
        avg_disk_rate = sum(disk_io_rates) / len(disk_io_rates) if disk_io_rates else 0
        avg_network_rate = sum(network_io_rates) / len(network_io_rates) if network_io_rates else 0
        
        # 生成瓶颈分析报告
        bottlenecks = []
        
        if avg_cpu > 80:
            bottlenecks.append({
                "type": "CPU",
                "severity": "high" if avg_cpu > 90 else "medium",
                "description": f"CPU使用率过高，平均使用率: {avg_cpu:.1f}%",
                "recommendations": [
                    "考虑优化CPU密集型算法",
                    "使用多进程或异步处理",
                    "检查是否有死循环或无限递归"
                ]
            })
            
        if avg_memory > 85:
            bottlenecks.append({
                "type": "Memory",
                "severity": "high" if avg_memory > 95 else "medium",
                "description": f"内存使用率过高，平均使用率: {avg_memory:.1f}%",
                "recommendations": [
                    "检查内存泄漏",
                    "优化数据结构",
                    "使用对象池或连接池",
                    "考虑分批处理大量数据"
                ]
            })
            
        if avg_disk_rate > 100 * 1024 * 1024:  # 100MB/s
            bottlenecks.append({
                "type": "Disk I/O",
                "severity": "medium",
                "description": f"磁盘I/O负载较高，平均速率: {avg_disk_rate / 1024 / 1024:.1f}MB/s",
                "recommendations": [
                    "使用SSD存储",
                    "优化数据库查询",
                    "考虑使用缓存",
                    "批量写入操作"
                ]
            })
            
        if avg_network_rate > 50 * 1024 * 1024:  # 50MB/s
            bottlenecks.append({
                "type": "Network I/O",
                "severity": "medium",
                "description": f"网络I/O负载较高，平均速率: {avg_network_rate / 1024 / 1024:.1f}MB/s",
                "recommendations": [
                    "优化网络请求",
                    "使用连接池",
                    "压缩传输数据",
                    "考虑CDN或缓存"
                ]
            })
            
        return {
            "analysis_time": datetime.now(),
            "metrics_summary": {
                "avg_cpu_percent": avg_cpu,
                "max_cpu_percent": max_cpu,
                "avg_memory_percent": avg_memory,
                "max_memory_percent": max_memory,
                "avg_disk_rate_mbps": avg_disk_rate / 1024 / 1024,
                "avg_network_rate_mbps": avg_network_rate / 1024 / 1024
            },
            "bottlenecks": bottlenecks,
            "recommendations": [
                "定期监控系统性能指标",
                "建立性能基线和告警机制",
                "进行定期的性能测试",
                "优化代码和数据库查询"
            ]
        }


class CodeProfiler:
    """代码性能分析器"""
    
    def __init__(self):
        """初始化代码性能分析器"""
        self.function_profiles: Dict[str, FunctionProfile] = {}
        self.memory_snapshots: List[Tuple[datetime, Dict[str, int]]] = []
        self.hotspots: List[Dict[str, Any]] = []
        self._lock = Lock()
        
    @contextmanager
    def profile_function(self, func_name: str = None, file_path: str = "", line_number: int = 0):
        """函数性能分析上下文管理器"""
        if not func_name:
            # 从调用栈获取函数名
            frame = inspect.currentframe().f_back
            func_name = frame.f_code.co_name
            file_path = frame.f_code.co_filename
            line_number = frame.f_lineno
            
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            with self._lock:
                if func_name in self.function_profiles:
                    profile = self.function_profiles[func_name]
                    profile.call_count += 1
                    profile.total_time += execution_time
                    profile.average_time = profile.total_time / profile.call_count
                    profile.min_time = min(profile.min_time, execution_time)
                    profile.max_time = max(profile.max_time, execution_time)
                    profile.memory_delta += memory_delta
                else:
                    self.function_profiles[func_name] = FunctionProfile(
                        function_name=func_name,
                        call_count=1,
                        total_time=execution_time,
                        average_time=execution_time,
                        min_time=execution_time,
                        max_time=execution_time,
                        file_path=file_path,
                        line_number=line_number,
                        memory_delta=memory_delta
                    )
                    
    def profile_code(self, code_string: str, globals_dict: Dict = None, locals_dict: Dict = None):
        """分析代码字符串的性能"""
        if globals_dict is None:
            globals_dict = {}
        if locals_dict is None:
            locals_dict = {}
            
        profiler = cProfile.Profile()
        profiler.enable()
        
        try:
            exec(code_string, globals_dict, locals_dict)
        finally:
            profiler.disable()
            
        # 分析结果
        s = io.StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # 显示前20个最耗时的函数
        
        return s.getvalue()
        
    def _get_memory_usage(self) -> int:
        """获取当前内存使用量"""
        if tracemalloc.is_tracing():
            current, peak = tracemalloc.get_traced_memory()
            return current
        else:
            # 使用psutil作为备选
            process = psutil.Process()
            return process.memory_info().rss
            
    def get_memory_leak_detection(self) -> Dict[str, Any]:
        """检测内存泄漏"""
        # 启动内存跟踪
        if not tracemalloc.is_tracing():
            tracemalloc.start()
            
        # 获取初始快照
        snapshot1 = tracemalloc.take_snapshot()
        
        # 强制垃圾回收
        gc.collect()
        
        # 获取垃圾回收后的快照
        snapshot2 = tracemalloc.take_snapshot()
        
        # 分析内存变化
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')
        
        leaks = []
        for stat in top_stats:
            if stat.size_diff > 1024 * 1024:  # 超过1MB的增长
                leaks.append({
                    "filename": stat.traceback.format()[-1],
                    "size_diff_mb": stat.size_diff / 1024 / 1024,
                    "count_diff": stat.count_diff
                })
                
        return {
            "analysis_time": datetime.now(),
            "memory_leaks_detected": len(leaks),
            "potential_leaks": leaks,
            "recommendations": [
                "检查是否有循环引用",
                "确保及时释放大型对象",
                "使用弱引用避免循环引用",
                "定期进行垃圾回收"
            ]
        }
        
    def get_hotspot_analysis(self) -> List[Dict[str, Any]]:
        """分析代码热点"""
        hotspots = []
        
        with self._lock:
            for func_name, profile in self.function_profiles.items():
                # 计算性能分数（基于调用次数和执行时间）
                performance_score = profile.call_count * profile.average_time
                
                hotspots.append({
                    "function_name": func_name,
                    "file_path": profile.file_path,
                    "line_number": profile.line_number,
                    "call_count": profile.call_count,
                    "total_time": profile.total_time,
                    "average_time": profile.average_time,
                    "min_time": profile.min_time,
                    "max_time": profile.max_time,
                    "memory_delta": profile.memory_delta,
                    "performance_score": performance_score,
                    "optimization_priority": "high" if performance_score > 1.0 else "medium" if performance_score > 0.1 else "low"
                })
                
        # 按性能分数排序
        hotspots.sort(key=lambda x: x["performance_score"], reverse=True)
        
        return hotspots
        
    def get_optimization_suggestions(self) -> List[Dict[str, Any]]:
        """获取代码优化建议"""
        suggestions = []
        hotspots = self.get_hotspot_analysis()
        
        for hotspot in hotspots[:10]:  # 只分析前10个热点
            func_name = hotspot["function_name"]
            file_path = hotspot["file_path"]
            
            # 分析源代码获取具体建议
            try:
                lines = linecache.getline(file_path, hotspot["line_number"]).strip()
                
                # 基础优化建议
                if hotspot["average_time"] > 0.1:  # 超过100ms
                    suggestions.append({
                        "function": func_name,
                        "file": file_path,
                        "line": hotspot["line_number"],
                        "issue": "函数执行时间过长",
                        "current_time": f"{hotspot['average_time']:.3f}s",
                        "suggestions": [
                            "考虑使用缓存或记忆化",
                            "优化算法复杂度",
                            "使用异步处理",
                            "分解为更小的函数"
                        ],
                        "priority": "high"
                    })
                    
                if hotspot["memory_delta"] > 10 * 1024 * 1024:  # 超过10MB
                    suggestions.append({
                        "function": func_name,
                        "file": file_path,
                        "line": hotspot["line_number"],
                        "issue": "函数内存使用过多",
                        "memory_delta_mb": hotspot["memory_delta"] / 1024 / 1024,
                        "suggestions": [
                            "使用生成器减少内存占用",
                            "及时释放不需要的对象",
                            "使用对象池",
                            "优化数据结构"
                        ],
                        "priority": "high"
                    })
                    
                # 检查循环和递归
                if "for" in lines or "while" in lines:
                    suggestions.append({
                        "function": func_name,
                        "file": file_path,
                        "line": hotspot["line_number"],
                        "issue": "循环结构可能存在性能问题",
                        "code_snippet": lines,
                        "suggestions": [
                            "检查循环终止条件",
                            "考虑使用列表推导式",
                            "优化循环内部操作",
                            "使用并行处理"
                        ],
                        "priority": "medium"
                    })
                    
            except Exception as e:
                logger.warning(f"分析函数 {func_name} 时出错: {e}")
                
        return suggestions
        
    def clear_profiles(self):
        """清除性能分析数据"""
        with self._lock:
            self.function_profiles.clear()
            self.memory_snapshots.clear()
            self.hotspots.clear()
        logger.info("代码性能分析数据已清除")


class DatabaseOptimizer:
    """数据库性能优化器"""
    
    def __init__(self, db_connection: sqlite3.Connection):
        """
        初始化数据库优化器
        
        Args:
            db_connection: 数据库连接
        """
        self.db_connection = db_connection
        self.query_metrics: Dict[str, DatabaseMetrics] = {}
        self.slow_queries: List[Tuple[str, float]] = []
        self._lock = Lock()
        
    def analyze_query_performance(self, query: str) -> Dict[str, Any]:
        """分析查询性能"""
        start_time = time.perf_counter()
        
        try:
            # 执行查询并获取执行计划
            cursor = self.db_connection.cursor()
            cursor.execute(f"EXPLAIN QUERY PLAN {query}")
            explain_result = cursor.fetchall()
            
            # 执行实际查询
            cursor.execute(query)
            result = cursor.fetchall()
            
            end_time = time.perf_counter()
            execution_time = end_time - start_time
            
            # 分析执行计划
            table_accesses = []
            index_usage = {}
            
            for row in explain_result:
                if len(row) >= 4:
                    detail = row[3]
                    if "SCAN" in detail:
                        table_accesses.append({"type": "table_scan", "table": detail.split()[-1]})
                    elif "SEARCH" in detail:
                        parts = detail.split()
                        table_name = parts[-1]
                        index_name = parts[3] if len(parts) > 3 else None
                        table_accesses.append({"type": "index_search", "table": table_name, "index": index_name})
                        if index_name:
                            index_usage[table_name] = index_name
                            
            return {
                "query": query,
                "execution_time": execution_time,
                "result_count": len(result),
                "explain_plan": [row[3] for row in explain_result],
                "table_accesses": table_accesses,
                "index_usage": index_usage,
                "performance_rating": self._rate_query_performance(execution_time, len(result), table_accesses),
                "suggestions": self._generate_query_suggestions(execution_time, table_accesses, index_usage)
            }
            
        except Exception as e:
            logger.error(f"查询性能分析失败: {e}")
            return {"error": str(e)}
            
    def _rate_query_performance(self, execution_time: float, result_count: int, table_accesses: List[Dict]) -> str:
        """评级查询性能"""
        if execution_time > 5.0:
            return "poor"
        elif execution_time > 1.0:
            return "fair"
        elif execution_time > 0.1:
            return "good"
        else:
            return "excellent"
            
    def _generate_query_suggestions(self, execution_time: float, table_accesses: List[Dict], index_usage: Dict) -> List[str]:
        """生成查询优化建议"""
        suggestions = []
        
        # 检查全表扫描
        table_scans = [access for access in table_accesses if access["type"] == "table_scan"]
        if table_scans:
            suggestions.append("查询存在全表扫描，建议添加适当的索引")
            
        # 检查执行时间
        if execution_time > 1.0:
            suggestions.append("查询执行时间较长，考虑优化查询逻辑或添加索引")
            
        # 检查结果集大小
        if execution_time > 0.1 and len(table_scans) == 0:
            suggestions.append("查询可能返回过多数据，考虑使用LIMIT或优化WHERE条件")
            
        return suggestions
        
    def optimize_index(self, table_name: str, column_name: str, index_type: str = "BTREE") -> Dict[str, Any]:
        """优化数据库索引"""
        try:
            cursor = self.db_connection.cursor()
            
            # 检查索引是否存在
            cursor.execute(f"PRAGMA index_list({table_name})")
            existing_indexes = cursor.fetchall()
            
            index_name = f"idx_{table_name}_{column_name}"
            
            # 检查是否已存在索引
            for idx in existing_indexes:
                if idx[1] == index_name:
                    return {
                        "status": "exists",
                        "message": f"索引 {index_name} 已存在",
                        "index_name": index_name
                    }
                    
            # 创建索引
            cursor.execute(f"CREATE INDEX {index_name} ON {table_name}({column_name})")
            self.db_connection.commit()
            
            # 验证索引
            cursor.execute(f"PRAGMA index_info({index_name})")
            index_info = cursor.fetchall()
            
            return {
                "status": "created",
                "message": f"索引 {index_name} 创建成功",
                "index_name": index_name,
                "index_type": index_type,
                "columns": [info[2] for info in index_info]
            }
            
        except Exception as e:
            logger.error(f"索引优化失败: {e}")
            return {"status": "error", "message": str(e)}
            
    def analyze_connection_pool(self) -> Dict[str, Any]:
        """分析连接池性能"""
        # 模拟连接池分析（实际实现需要根据具体的连接池库）
        try:
            # 获取当前连接状态
            cursor = self.db_connection.cursor()
            
            # 检查数据库连接信息
            cursor.execute("PRAGMA compile_options")
            compile_options = [row[0] for row in cursor.fetchall()]
            
            # 分析连接池配置
            pool_analysis = {
                "connection_count": 1,  # SQLite单连接
                "max_connections": 1,
                "connection_utilization": 100.0,
                "connection_pool_efficiency": "good",
                "recommendations": [
                    "SQLite不支持连接池，考虑使用其他数据库",
                    "对于高并发场景，建议使用PostgreSQL或MySQL"
                ]
            }
            
            return pool_analysis
            
        except Exception as e:
            logger.error(f"连接池分析失败: {e}")
            return {"error": str(e)}
            
    def get_query_optimization_report(self) -> Dict[str, Any]:
        """获取查询优化报告"""
        if not self.slow_queries:
            return {"message": "没有慢查询记录"}
            
        # 分析慢查询
        slow_query_analysis = []
        for query, execution_time in self.slow_queries:
            analysis = self.analyze_query_performance(query)
            if "error" not in analysis:
                slow_query_analysis.append(analysis)
                
        # 生成优化建议
        optimization_suggestions = []
        
        # 索引建议
        tables_needing_indexes = set()
        for analysis in slow_query_analysis:
            for access in analysis.get("table_accesses", []):
                if access["type"] == "table_scan":
                    tables_needing_indexes.add(access["table"])
                    
        for table in tables_needing_indexes:
            optimization_suggestions.append({
                "type": "index_recommendation",
                "table": table,
                "description": f"表 {table} 存在全表扫描，建议添加索引",
                "priority": "high"
            })
            
        # 查询优化建议
        for analysis in slow_query_analysis:
            if analysis.get("performance_rating") in ["poor", "fair"]:
                optimization_suggestions.append({
                    "type": "query_optimization",
                    "query": analysis["query"][:100] + "..." if len(analysis["query"]) > 100 else analysis["query"],
                    "description": f"查询性能评级: {analysis['performance_rating']}",
                    "suggestions": analysis.get("suggestions", []),
                    "priority": "medium"
                })
                
        return {
            "analysis_time": datetime.now(),
            "total_slow_queries": len(self.slow_queries),
            "slow_query_analysis": slow_query_analysis,
            "optimization_suggestions": optimization_suggestions,
            "overall_recommendations": [
                "定期分析慢查询日志",
                "根据查询模式优化索引",
                "考虑查询缓存",
                "优化数据库配置参数"
            ]
        }


class CacheOptimizer:
    """缓存策略优化器"""
    
    def __init__(self, cache_size_limit: int = 1000):
        """
        初始化缓存优化器
        
        Args:
            cache_size_limit: 缓存大小限制
        """
        self.cache: Dict[str, Tuple[Any, datetime]] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.metrics = CacheMetrics(
            hit_count=0,
            miss_count=0,
            hit_rate=0.0,
            eviction_count=0,
            average_access_time=0.0,
            memory_usage=0
        )
        self.size_limit = cache_size_limit
        self._lock = Lock()
        
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        start_time = time.perf_counter()
        
        with self._lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                self.access_count[key] += 1
                self.metrics.hit_count += 1
                self.metrics.hit_rate = self.metrics.hit_count / (self.metrics.hit_count + self.metrics.miss_count)
                return value
            else:
                self.metrics.miss_count += 1
                self.metrics.hit_rate = self.metrics.hit_count / (self.metrics.hit_count + self.metrics.miss_count)
                return None
                
    def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """设置缓存值"""
        with self._lock:
            # 检查缓存大小限制
            if len(self.cache) >= self.size_limit:
                self._evict_lru()
                
            # 设置缓存项
            expiry_time = datetime.now() + timedelta(seconds=ttl)
            self.cache[key] = (value, expiry_time)
            return True
            
    def _evict_lru(self):
        """淘汰最少最近使用的缓存项"""
        if not self.cache:
            return
            
        # 找到最少访问的键
        lru_key = min(self.cache.keys(), key=lambda k: self.access_count[k])
        del self.cache[lru_key]
        del self.access_count[lru_key]
        self.metrics.eviction_count += 1
        
    def clear_expired(self):
        """清除过期缓存项"""
        current_time = datetime.now()
        expired_keys = []
        
        with self._lock:
            for key, (value, expiry_time) in self.cache.items():
                if current_time > expiry_time:
                    expired_keys.append(key)
                    
            for key in expired_keys:
                del self.cache[key]
                if key in self.access_count:
                    del self.access_count[key]
                    
        return len(expired_keys)
        
    def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            total_requests = self.metrics.hit_count + self.metrics.miss_count
            hit_rate = self.metrics.hit_count / total_requests if total_requests > 0 else 0.0
            
            # 计算内存使用量（估算）
            memory_usage = 0
            for key, (value, _) in self.cache.items():
                memory_usage += len(str(key)) + len(str(value))
                
            return {
                "cache_size": len(self.cache),
                "size_limit": self.size_limit,
                "hit_count": self.metrics.hit_count,
                "miss_count": self.metrics.miss_count,
                "hit_rate": hit_rate,
                "eviction_count": self.metrics.eviction_count,
                "memory_usage_bytes": memory_usage,
                "memory_usage_mb": memory_usage / 1024 / 1024,
                "unique_keys": len(self.access_count)
            }
            
    def analyze_cache_performance(self) -> Dict[str, Any]:
        """分析缓存性能"""
        stats = self.get_cache_statistics()
        
        # 分析访问模式
        access_patterns = {}
        with self._lock:
            for key, count in self.access_count.items():
                if count > 0:
                    access_patterns[key] = count
                    
        # 找出热点键
        hot_keys = sorted(access_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 生成优化建议
        recommendations = []
        
        if stats["hit_rate"] < 0.8:
            recommendations.append({
                "type": "hit_rate",
                "issue": "缓存命中率较低",
                "current_hit_rate": f"{stats['hit_rate']:.2%}",
                "suggestions": [
                    "增加缓存大小限制",
                    "优化缓存键的设计",
                    "调整TTL设置",
                    "检查缓存失效策略"
                ]
            })
            
        if stats["cache_size"] / stats["size_limit"] > 0.9:
            recommendations.append({
                "type": "cache_size",
                "issue": "缓存接近满载",
                "utilization": f"{stats['cache_size'] / stats['size_limit']:.2%}",
                "suggestions": [
                    "增加缓存大小限制",
                    "优化缓存淘汰策略",
                    "清理过期缓存项"
                ]
            })
            
        if stats["eviction_count"] > stats["hit_count"]:
            recommendations.append({
                "type": "eviction_rate",
                "issue": "缓存淘汰过于频繁",
                "eviction_rate": f"{stats['eviction_count'] / max(stats['hit_count'], 1):.2f}",
                "suggestions": [
                    "使用更智能的淘汰算法",
                    "增加缓存容量",
                    "优化缓存键策略"
                ]
            })
            
        return {
            "analysis_time": datetime.now(),
            "statistics": stats,
            "hot_keys": hot_keys,
            "recommendations": recommendations,
            "optimization_score": self._calculate_optimization_score(stats)
        }
        
    def _calculate_optimization_score(self, stats: Dict[str, Any]) -> float:
        """计算缓存优化分数"""
        score = 0.0
        
        # 命中率权重：40%
        score += min(stats["hit_rate"], 1.0) * 0.4
        
        # 缓存利用率权重：30%
        utilization = stats["cache_size"] / stats["size_limit"]
        score += (1.0 - abs(utilization - 0.8)) * 0.3  # 理想利用率80%
        
        # 淘汰率权重：20%
        eviction_rate = stats["eviction_count"] / max(stats["hit_count"], 1)
        score += max(0, 1.0 - eviction_rate) * 0.2
        
        # 内存使用权重：10%
        memory_efficiency = 1.0 - min(stats["memory_usage_mb"] / 100, 1.0)  # 假设100MB为基准
        score += memory_efficiency * 0.1
        
        return min(score, 1.0)


class ConcurrencyOptimizer:
    """并发性能优化器"""
    
    def __init__(self, max_workers: int = None):
        """
        初始化并发优化器
        
        Args:
            max_workers: 最大工作线程数
        """
        self.max_workers = max_workers or cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_workers)
        self.active_tasks: Dict[str, concurrent.futures.Future] = {}
        self.metrics_history: deque = deque(maxlen=1000)
        self._lock = Lock()
        
    def submit_task(self, task_func: Callable, task_id: str = None, *args, **kwargs) -> str:
        """提交任务到线程池"""
        if task_id is None:
            task_id = f"task_{len(self.active_tasks)}"
            
        future = self.thread_pool.submit(task_func, *args, **kwargs)
        
        with self._lock:
            self.active_tasks[task_id] = future
            
        return task_id
        
    def get_task_result(self, task_id: str, timeout: float = None) -> Any:
        """获取任务结果"""
        with self._lock:
            if task_id not in self.active_tasks:
                raise ValueError(f"任务 {task_id} 不存在")
                
            future = self.active_tasks[task_id]
            
        try:
            result = future.result(timeout=timeout)
            
            # 任务完成后清理
            with self._lock:
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
                    
            return result
            
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"任务 {task_id} 超时")
        except Exception as e:
            logger.error(f"任务 {task_id} 执行失败: {e}")
            with self._lock:
                if task_id in self.active_tasks:
                    del self.active_tasks[task_id]
            raise
            
    def optimize_thread_pool(self) -> Dict[str, Any]:
        """优化线程池配置"""
        current_metrics = self._collect_concurrency_metrics()
        
        # 分析线程池性能
        analysis = {
            "current_metrics": current_metrics,
            "recommendations": []
        }
        
        # CPU密集型任务建议
        cpu_usage = psutil.cpu_percent(interval=1)
        if cpu_usage < 50:
            analysis["recommendations"].append({
                "type": "thread_count",
                "issue": "CPU使用率较低，可能线程数不足",
                "current_cpu": f"{cpu_usage:.1f}%",
                "suggestion": f"考虑增加线程数，当前: {self.max_workers}, 建议: {min(self.max_workers * 2, cpu_count() * 2)}"
            })
        elif cpu_usage > 90:
            analysis["recommendations"].append({
                "type": "thread_count",
                "issue": "CPU使用率过高，可能线程数过多",
                "current_cpu": f"{cpu_usage:.1f}%",
                "suggestion": f"考虑减少线程数，当前: {self.max_workers}, 建议: {max(1, self.max_workers // 2)}"
            })
            
        # 内存使用分析
        memory = psutil.virtual_memory()
        if memory.percent > 80:
            analysis["recommendations"].append({
                "type": "memory",
                "issue": "内存使用率较高",
                "current_memory": f"{memory.percent:.1f}%",
                "suggestion": "考虑减少线程数或任务数量，避免内存不足"
            })
            
        # 队列长度分析
        if current_metrics.get("queue_size", 0) > self.max_workers * 2:
            analysis["recommendations"].append({
                "type": "queue_size",
                "issue": "任务队列过长",
                "current_queue": current_metrics["queue_size"],
                "suggestion": "考虑增加线程数或优化任务处理速度"
            })
            
        return analysis
        
    def _collect_concurrency_metrics(self) -> ConcurrencyMetrics:
        """收集并发性能指标"""
        # 获取当前活跃线程数
        active_threads = threading.active_count()
        
        # 估算队列大小（基于活跃任务数）
        queue_size = len(self.active_tasks)
        
        # 收集其他指标
        return ConcurrencyMetrics(
            active_threads=active_threads,
            thread_pool_size=self.max_workers,
            queue_size=queue_size,
            lock_contention=0,  # 简化实现
            average_wait_time=0.0,  # 简化实现
            throughput=0.0  # 简化实现
        )
        
    def analyze_lock_contention(self) -> Dict[str, Any]:
        """分析锁竞争情况"""
        # 简化的锁竞争分析
        # 实际实现需要更复杂的监控机制
        
        thread_info = []
        for thread in threading.enumerate():
            thread_info.append({
                "name": thread.name,
                "ident": thread.ident,
                "is_alive": thread.is_alive()
            })
            
        return {
            "analysis_time": datetime.now(),
            "active_threads": len(thread_info),
            "thread_details": thread_info,
            "lock_contention_analysis": {
                "contention_level": "low",  # 简化评级
                "bottlenecks": [],
                "recommendations": [
                    "使用更细粒度的锁",
                    "考虑无锁数据结构",
                    "减少锁的持有时间"
                ]
            }
        }
        
    def get_concurrency_report(self) -> Dict[str, Any]:
        """获取并发性能报告"""
        current_metrics = self._collect_concurrency_metrics()
        thread_pool_analysis = self.optimize_thread_pool()
        lock_analysis = self.analyze_lock_contention()
        
        return {
            "report_time": datetime.now(),
            "current_metrics": current_metrics,
            "thread_pool_analysis": thread_pool_analysis,
            "lock_analysis": lock_analysis,
            "overall_recommendations": [
                "根据CPU核心数调整线程池大小",
                "监控任务执行时间，避免长时间阻塞",
                "使用异步编程模式提高并发性能",
                "定期分析锁竞争情况"
            ]
        }
        
    def shutdown(self):
        """关闭线程池"""
        self.thread_pool.shutdown(wait=True)
        logger.info("并发优化器线程池已关闭")


class PerformanceMonitor:
    """性能监控和告警系统"""
    
    def __init__(self, alert_thresholds: Dict[str, float] = None):
        """
        初始化性能监控器
        
        Args:
            alert_thresholds: 告警阈值配置
        """
        self.alert_thresholds = alert_thresholds or {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0,
            "response_time": 5.0,
            "error_rate": 0.05
        }
        
        self.alerts: List[Alert] = []
        self.monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._alert_handlers: List[Callable] = []
        self._lock = Lock()
        
    async def start_monitoring(self, interval: float = 30.0):
        """开始性能监控"""
        if self.monitoring_active:
            logger.warning("性能监控已在运行中")
            return
            
        self.monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop(interval))
        logger.info("性能监控已启动")
        
    async def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("性能监控已停止")
        
    async def _monitoring_loop(self, interval: float):
        """监控循环"""
        while self.monitoring_active:
            try:
                await self._check_performance_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"性能监控错误: {e}")
                await asyncio.sleep(interval)
                
    async def _check_performance_metrics(self):
        """检查性能指标"""
        # CPU使用率
        cpu_percent = psutil.cpu_percent(interval=1)
        if cpu_percent > self.alert_thresholds["cpu_percent"]:
            await self._create_alert(
                level="WARNING",
                message=f"CPU使用率过高: {cpu_percent:.1f}%",
                source="system",
                details={"cpu_percent": cpu_percent, "threshold": self.alert_thresholds["cpu_percent"]}
            )
            
        # 内存使用率
        memory = psutil.virtual_memory()
        if memory.percent > self.alert_thresholds["memory_percent"]:
            await self._create_alert(
                level="WARNING",
                message=f"内存使用率过高: {memory.percent:.1f}%",
                source="system",
                details={
                    "memory_percent": memory.percent,
                    "threshold": self.alert_thresholds["memory_percent"],
                    "available_gb": memory.available / 1024 / 1024 / 1024
                }
            )
            
        # 磁盘使用率
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        if disk_percent > self.alert_thresholds["disk_usage_percent"]:
            await self._create_alert(
                level="WARNING",
                message=f"磁盘使用率过高: {disk_percent:.1f}%",
                source="system",
                details={
                    "disk_percent": disk_percent,
                    "threshold": self.alert_thresholds["disk_usage_percent"],
                    "free_gb": disk.free / 1024 / 1024 / 1024
                }
            )
            
    async def _create_alert(self, level: str, message: str, source: str, details: Dict[str, Any] = None):
        """创建告警"""
        alert = Alert(
            id=f"alert_{len(self.alerts)}_{int(time.time())}",
            level=level,
            message=message,
            timestamp=datetime.now(),
            source=source,
            details=details or {}
        )
        
        with self._lock:
            self.alerts.append(alert)
            
        # 调用告警处理器
        for handler in self._alert_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(alert)
                else:
                    handler(alert)
            except Exception as e:
                logger.error(f"告警处理器执行失败: {e}")
                
        logger.warning(f"性能告警: {alert.message}")
        
    def add_alert_handler(self, handler: Callable):
        """添加告警处理器"""
        self._alert_handlers.append(handler)
        
    def get_alerts(self, hours: int = 24) -> List[Alert]:
        """获取告警历史"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        with self._lock:
            return [alert for alert in self.alerts if alert.timestamp >= cutoff_time]
            
    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计信息"""
        recent_alerts = self.get_alerts(24)
        
        # 按级别统计
        level_counts = defaultdict(int)
        source_counts = defaultdict(int)
        
        for alert in recent_alerts:
            level_counts[alert.level] += 1
            source_counts[alert.source] += 1
            
        return {
            "total_alerts_24h": len(recent_alerts),
            "level_distribution": dict(level_counts),
            "source_distribution": dict(source_counts),
            "recent_alerts": [
                {
                    "timestamp": alert.timestamp.isoformat(),
                    "level": alert.level,
                    "message": alert.message,
                    "source": alert.source
                }
                for alert in recent_alerts[-10:]  # 最近10条告警
            ]
        }


class AsyncPerformanceOptimizer:
    """异步性能优化处理器"""
    
    def __init__(self):
        """初始化异步性能优化器"""
        self.async_metrics: Dict[str, List[float]] = defaultdict(list)
        self._lock = Lock()
        
    @asynccontextmanager
    async def profile_async_function(self, func_name: str = None):
        """异步函数性能分析上下文管理器"""
        if not func_name:
            # 从调用栈获取函数名
            frame = inspect.currentframe()
            if frame:
                func_name = frame.f_code.co_name
                
        start_time = time.perf_counter()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.perf_counter()
            end_memory = self._get_memory_usage()
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory
            
            with self._lock:
                self.async_metrics[f"{func_name}_execution_time"].append(execution_time)
                self.async_metrics[f"{func_name}_memory_delta"].append(memory_delta)
                
    def _get_memory_usage(self) -> int:
        """获取当前内存使用量"""
        process = psutil.Process()
        return process.memory_info().rss
        
    async def optimize_async_operations(self) -> Dict[str, Any]:
        """优化异步操作"""
        optimization_suggestions = []
        
        with self._lock:
            for metric_name, values in self.async_metrics.items():
                if not values:
                    continue
                    
                avg_time = sum(values) / len(values)
                max_time = max(values)
                
                if "execution_time" in metric_name and avg_time > 0.1:
                    optimization_suggestions.append({
                        "type": "async_optimization",
                        "operation": metric_name.replace("_execution_time", ""),
                        "issue": f"异步操作执行时间过长",
                        "average_time": avg_time,
                        "max_time": max_time,
                        "suggestions": [
                            "使用更高效的异步库",
                            "优化I/O操作",
                            "考虑批量处理",
                            "使用连接池"
                        ]
                    })
                    
        return {
            "optimization_time": datetime.now(),
            "suggestions": optimization_suggestions,
            "metrics_summary": {
                metric: {
                    "count": len(values),
                    "average": sum(values) / len(values) if values else 0,
                    "max": max(values) if values else 0,
                    "min": min(values) if values else 0
                }
                for metric, values in self.async_metrics.items()
            }
        }
        
    def get_async_performance_report(self) -> Dict[str, Any]:
        """获取异步性能报告"""
        with self._lock:
            return {
                "report_time": datetime.now(),
                "metrics": dict(self.async_metrics),
                "total_operations": sum(len(values) for values in self.async_metrics.values())
            }


class PerformanceOptimizer:
    """O1性能优化器主类"""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化性能优化器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        
        # 初始化各个组件
        self.system_profiler = SystemProfiler(
            interval=self.config.get("system_monitor_interval", 1.0)
        )
        self.code_profiler = CodeProfiler()
        self.database_optimizer = None  # 需要时初始化
        self.cache_optimizer = CacheOptimizer(
            cache_size_limit=self.config.get("cache_size_limit", 1000)
        )
        self.concurrency_optimizer = ConcurrencyOptimizer(
            max_workers=self.config.get("max_workers")
        )
        self.performance_monitor = PerformanceMonitor(
            alert_thresholds=self.config.get("alert_thresholds")
        )
        self.async_optimizer = AsyncPerformanceOptimizer()
        
        # 状态管理
        self.is_initialized = False
        self.optimization_history: List[Dict[str, Any]] = []
        
        logger.info("O1性能优化器已初始化")
        
    def initialize_database_optimizer(self, db_connection: sqlite3.Connection):
        """初始化数据库优化器"""
        self.database_optimizer = DatabaseOptimizer(db_connection)
        logger.info("数据库优化器已初始化")
        
    async def start_comprehensive_monitoring(self):
        """启动全面性能监控"""
        if self.is_initialized:
            logger.warning("性能优化器已在运行中")
            return
            
        # 启动系统监控
        self.system_profiler.start_monitoring()
        
        # 启动性能监控
        await self.performance_monitor.start_monitoring(
            interval=self.config.get("monitor_interval", 30.0)
        )
        
        self.is_initialized = True
        logger.info("全面性能监控已启动")
        
    async def stop_comprehensive_monitoring(self):
        """停止全面性能监控"""
        if not self.is_initialized:
            logger.warning("性能优化器未在运行")
            return
            
        # 停止各项监控
        self.system_profiler.stop_monitoring()
        await self.performance_monitor.stop_monitoring()
        self.concurrency_optimizer.shutdown()
        
        self.is_initialized = False
        logger.info("全面性能监控已停止")
        
    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """运行全面的性能分析"""
        logger.info("开始全面性能分析")
        
        analysis_results = {}
        
        try:
            # 1. 系统性能分析
            logger.info("执行系统性能分析")
            system_bottlenecks = self.system_profiler.get_bottleneck_analysis()
            analysis_results["system_analysis"] = system_bottlenecks
            
            # 2. 代码性能分析
            logger.info("执行代码性能分析")
            code_hotspots = self.code_profiler.get_hotspot_analysis()
            code_suggestions = self.code_profiler.get_optimization_suggestions()
            memory_leaks = self.code_profiler.get_memory_leak_detection()
            
            analysis_results["code_analysis"] = {
                "hotspots": code_hotspots,
                "optimization_suggestions": code_suggestions,
                "memory_leak_detection": memory_leaks
            }
            
            # 3. 数据库性能分析
            if self.database_optimizer:
                logger.info("执行数据库性能分析")
                db_report = self.database_optimizer.get_query_optimization_report()
                db_connection_analysis = self.database_optimizer.analyze_connection_pool()
                
                analysis_results["database_analysis"] = {
                    "query_optimization": db_report,
                    "connection_pool": db_connection_analysis
                }
            
            # 4. 缓存性能分析
            logger.info("执行缓存性能分析")
            cache_analysis = self.cache_optimizer.analyze_cache_performance()
            analysis_results["cache_analysis"] = cache_analysis
            
            # 5. 并发性能分析
            logger.info("执行并发性能分析")
            concurrency_report = self.concurrency_optimizer.get_concurrency_report()
            analysis_results["concurrency_analysis"] = concurrency_report
            
            # 6. 异步性能分析
            logger.info("执行异步性能分析")
            async_optimization = await self.async_optimizer.optimize_async_operations()
            async_report = self.async_optimizer.get_async_performance_report()
            
            analysis_results["async_analysis"] = {
                "optimization": async_optimization,
                "performance_report": async_report
            }
            
            # 7. 告警统计
            logger.info("分析告警统计")
            alert_stats = self.performance_monitor.get_alert_statistics()
            analysis_results["alert_statistics"] = alert_stats
            
            # 8. 生成综合优化建议
            logger.info("生成综合优化建议")
            overall_recommendations = self._generate_overall_recommendations(analysis_results)
            analysis_results["overall_recommendations"] = overall_recommendations
            
            # 保存分析结果
            self.optimization_history.append({
                "timestamp": datetime.now(),
                "analysis_results": analysis_results
            })
            
            logger.info("全面性能分析完成")
            return analysis_results
            
        except Exception as e:
            logger.error(f"全面性能分析失败: {e}")
            raise
            
    def _generate_overall_recommendations(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成综合优化建议"""
        recommendations = []
        
        # 基于系统分析的建议
        system_analysis = analysis_results.get("system_analysis", {})
        if "bottlenecks" in system_analysis:
            for bottleneck in system_analysis["bottlenecks"]:
                recommendations.append({
                    "category": "system",
                    "priority": bottleneck.get("severity", "medium"),
                    "issue": bottleneck.get("description", ""),
                    "recommendations": bottleneck.get("recommendations", [])
                })
                
        # 基于代码分析的建议
        code_analysis = analysis_results.get("code_analysis", {})
        if "optimization_suggestions" in code_analysis:
            for suggestion in code_analysis["optimization_suggestions"]:
                recommendations.append({
                    "category": "code",
                    "priority": suggestion.get("priority", "medium"),
                    "issue": suggestion.get("issue", ""),
                    "function": suggestion.get("function", ""),
                    "recommendations": suggestion.get("suggestions", [])
                })
                
        # 基于缓存分析的建议
        cache_analysis = analysis_results.get("cache_analysis", {})
        if "recommendations" in cache_analysis:
            for rec in cache_analysis["recommendations"]:
                recommendations.append({
                    "category": "cache",
                    "priority": "medium",
                    "issue": rec.get("issue", ""),
                    "recommendations": rec.get("suggestions", [])
                })
                
        # 基于并发分析的建议
        concurrency_analysis = analysis_results.get("concurrency_analysis", {})
        if "overall_recommendations" in concurrency_analysis:
            for rec in concurrency_analysis["overall_recommendations"]:
                recommendations.append({
                    "category": "concurrency",
                    "priority": "medium",
                    "recommendation": rec
                })
                
        return recommendations
        
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """获取优化历史"""
        return self.optimization_history
        
    def export_optimization_report(self, file_path: str = None) -> str:
        """导出优化报告"""
        if not file_path:
            file_path = f"performance_optimization_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        report_data = {
            "export_time": datetime.now().isoformat(),
            "config": self.config,
            "optimization_history": self.optimization_history
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
            
        logger.info(f"优化报告已导出到: {file_path}")
        return file_path
        
    @contextmanager
    def profile_function(self, func_name: str = None):
        """函数性能分析装饰器"""
        return self.code_profiler.profile_function(func_name)
        
    def optimize_database_query(self, query: str) -> Dict[str, Any]:
        """优化数据库查询"""
        if not self.database_optimizer:
            return {"error": "数据库优化器未初始化"}
            
        return self.database_optimizer.analyze_query_performance(query)
        
    def get_cache_statistics(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.cache_optimizer.get_cache_statistics()
        
    def get_system_metrics(self) -> Optional[PerformanceMetrics]:
        """获取系统指标"""
        return self.system_profiler.get_current_metrics()
        
    async def get_performance_alerts(self, hours: int = 24) -> List[Alert]:
        """获取性能告警"""
        return self.performance_monitor.get_alerts(hours)
        
    def clear_all_profiles(self):
        """清除所有性能分析数据"""
        self.code_profiler.clear_profiles()
        self.async_metrics = defaultdict(list)
        logger.info("所有性能分析数据已清除")


# 装饰器和工具函数
def profile_performance(func_name: str = None):
    """性能分析装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 获取性能优化器实例（需要全局配置）
            optimizer = globals().get('performance_optimizer_instance')
            if optimizer:
                with optimizer.profile_function(func_name or func.__name__):
                    return func(*args, **kwargs)
            else:
                # 直接使用代码分析器
                code_profiler = CodeProfiler()
                with code_profiler.profile_function(func_name or func.__name__):
                    return func(*args, **kwargs)
        return wrapper
    return decorator


async def profile_async_performance(func_name: str = None):
    """异步性能分析装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            optimizer = globals().get('performance_optimizer_instance')
            if optimizer:
                async with optimizer.async_optimizer.profile_async_function(func_name or func.__name__):
                    return await func(*args, **kwargs)
            else:
                # 直接使用异步优化器
                async_optimizer = AsyncPerformanceOptimizer()
                async with async_optimizer.profile_async_function(func_name or func.__name__):
                    return await func(*args, **kwargs)
        return wrapper
    return decorator


def performance_benchmark(iterations: int = 1000):
    """性能基准测试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            
            for _ in range(iterations):
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
                
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"性能基准测试结果 - {func.__name__}:")
            print(f"  平均执行时间: {avg_time:.6f}秒")
            print(f"  最小执行时间: {min_time:.6f}秒")
            print(f"  最大执行时间: {max_time:.6f}秒")
            print(f"  总迭代次数: {iterations}")
            
            return result
        return wrapper
    return decorator


# 使用示例和测试代码
async def example_usage():
    """使用示例"""
    # 初始化性能优化器
    config = {
        "system_monitor_interval": 1.0,
        "cache_size_limit": 1000,
        "max_workers": 4,
        "monitor_interval": 30.0,
        "alert_thresholds": {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0,
            "response_time": 5.0,
            "error_rate": 0.05
        }
    }
    
    optimizer = PerformanceOptimizer(config)
    
    # 启动全面监控
    await optimizer.start_comprehensive_monitoring()
    
    try:
        # 模拟一些工作负载
        await simulate_workload()
        
        # 运行性能分析
        analysis_results = await optimizer.run_comprehensive_analysis()
        
        # 打印结果
        print("=== 性能分析结果 ===")
        print(f"系统瓶颈: {len(analysis_results.get('system_analysis', {}).get('bottlenecks', []))}个")
        print(f"代码热点: {len(analysis_results.get('code_analysis', {}).get('hotspots', []))}个")
        print(f"优化建议: {len(analysis_results.get('overall_recommendations', []))}条")
        
        # 导出报告
        report_file = optimizer.export_optimization_report()
        print(f"详细报告已保存到: {report_file}")
        
    finally:
        # 停止监控
        await optimizer.stop_comprehensive_monitoring()


async def simulate_workload(optimizer):
    """模拟工作负载"""
    # 模拟CPU密集型任务
    def cpu_intensive_task():
        result = 0
        for i in range(1000000):
            result += i * i
        return result
        
    # 模拟I/O密集型任务
    def io_intensive_task():
        time.sleep(0.1)
        return "I/O任务完成"
        
    # 模拟内存密集型任务
    def memory_intensive_task():
        data = [i for i in range(100000)]
        return len(data)
        
    # 提交任务到线程池
    task_ids = []
    for _ in range(10):
        task_ids.append(optimizer.concurrency_optimizer.submit_task(cpu_intensive_task))
        task_ids.append(optimizer.concurrency_optimizer.submit_task(io_intensive_task))
        task_ids.append(optimizer.concurrency_optimizer.submit_task(memory_intensive_task))
        
    # 等待所有任务完成
    for task_id in task_ids:
        try:
            optimizer.concurrency_optimizer.get_task_result(task_id, timeout=10)
        except (TimeoutError, Exception) as e:
            logger.warning(f"任务 {task_id} 失败: {e}")
            
    # 使用缓存
    for i in range(100):
        key = f"test_key_{i % 50}"  # 重复键来测试缓存命中率
        value = f"test_value_{i}"
        optimizer.cache_optimizer.set(key, value, ttl=60)
        
        # 读取缓存
        cached_value = optimizer.cache_optimizer.get(key)
        if cached_value:
            pass  # 缓存命中
            
    # 模拟数据库操作
    import sqlite3
    db_conn = sqlite3.connect(':memory:')
    optimizer.initialize_database_optimizer(db_conn)
    
    # 创建测试表
    db_conn.execute('CREATE TABLE test_table (id INTEGER, name TEXT, value REAL)')
    
    # 插入测试数据
    for i in range(1000):
        db_conn.execute(f'INSERT INTO test_table VALUES ({i}, "name_{i}", {i * 1.5})')
    db_conn.commit()
    
    # 执行一些查询
    slow_queries = [
        "SELECT * FROM test_table WHERE value > 500",  # 可能较慢的查询
        "SELECT COUNT(*) FROM test_table",  # 简单查询
        "SELECT * FROM test_table ORDER BY value DESC LIMIT 10"  # 排序查询
    ]
    
    for query in slow_queries:
        result = optimizer.optimize_database_query(query)
        if "error" not in result:
            optimizer.database_optimizer.slow_queries.append((query, result.get("execution_time", 0)))


# 异步性能分析示例
@profile_async_performance()
async def async_example_function():
    """异步函数性能分析示例"""
    await asyncio.sleep(0.1)
    return "异步任务完成"


# 性能基准测试示例
@performance_benchmark(iterations=100)
def benchmark_example_function():
    """性能基准测试示例"""
    result = 0
    for i in range(1000):
        result += i ** 2
    return result


if __name__ == "__main__":
    # 运行示例
    print("O1性能优化器演示")
    print("================")
    
    # 运行基本示例
    try:
        asyncio.run(example_usage())
    except KeyboardInterrupt:
        print("\n演示已中断")
    except Exception as e:
        print(f"演示失败: {e}")
        
    # 运行基准测试
    print("\n运行性能基准测试...")
    benchmark_example_function()
    
    # 运行异步示例
    print("\n运行异步性能分析...")
    try:
        result = asyncio.run(async_example_function())
        print(f"异步函数结果: {result}")
    except Exception as e:
        print(f"异步示例失败: {e}")
        
    print("\n演示完成")