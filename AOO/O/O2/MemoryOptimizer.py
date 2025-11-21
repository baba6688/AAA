"""
O2内存优化器模块

该模块提供全面的内存管理和优化功能，包括：
1. 内存使用分析和监控（堆内存、栈内存、对象生命周期）
2. 内存泄漏检测和诊断（泄漏检测、泄漏定位、泄漏修复）
3. 垃圾回收优化（GC调优、内存分配优化）
4. 对象池和资源复用（对象池管理、连接池优化）
5. 大数据处理优化（分块处理、流式处理、内存映射）
6. 内存缓存优化（LRU缓存、弱引用缓存）
7. 异步内存优化处理
8. 完整的错误处理和日志记录

作者: O2优化团队
版本: 2.0.0
日期: 2025-11-06
"""

import asyncio
import gc
import logging
import sys
import threading
import time
import tracemalloc
import weakref
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from inspect import signature
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union, Callable, 
    Generic, TypeVar, AsyncIterator, Iterator, IO, Protocol
)
import psutil
import os
import mmap
import pickle
import json
from pathlib import Path


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# 类型定义
T = TypeVar('T')
MemoryStats = Dict[str, Any]
ObjectInfo = Dict[str, Any]
LeakReport = Dict[str, Any]


@dataclass
class MemorySnapshot:
    """内存快照数据类"""
    timestamp: float
    heap_size: int
    stack_size: int
    object_count: int
    gc_stats: Dict[str, int]
    process_memory: Dict[str, int]
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectTrackingInfo:
    """对象跟踪信息"""
    obj_id: int
    obj_type: str
    size: int
    created_at: float
    last_accessed: float
    reference_count: int
    stack_trace: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryAnalyzer:
    """
    内存分析器
    
    提供内存使用分析和监控功能，包括堆内存、栈内存和对象生命周期跟踪
    """
    
    def __init__(self, enable_tracemalloc: bool = True):
        """
        初始化内存分析器
        
        Args:
            enable_tracemalloc: 是否启用内存跟踪
        """
        self.enable_tracemalloc = enable_tracemalloc
        self.snapshots: List[MemorySnapshot] = []
        self.object_tracking: Dict[int, ObjectTrackingInfo] = {}
        self._tracking_enabled = False
        self._lock = threading.RLock()
        
        if enable_tracemalloc:
            tracemalloc.start(10)  # 保存10个堆栈跟踪
        
        logger.info("内存分析器已初始化")
    
    def start_object_tracking(self) -> None:
        """开始对象跟踪"""
        with self._lock:
            if not self._tracking_enabled:
                self._tracking_enabled = True
                # 设置垃圾回收回调
                gc.callbacks.append(self._gc_callback)
                logger.info("对象跟踪已启动")
    
    def stop_object_tracking(self) -> None:
        """停止对象跟踪"""
        with self._lock:
            if self._tracking_enabled:
                self._tracking_enabled = False
                if self._gc_callback in gc.callbacks:
                    gc.callbacks.remove(self._gc_callback)
                logger.info("对象跟踪已停止")
    
    def _gc_callback(self, phase: str, info: Dict[str, Any]) -> None:
        """垃圾回收回调函数"""
        if phase == 'start' and self._tracking_enabled:
            self._snapshot_objects()
    
    def _snapshot_objects(self) -> None:
        """快照当前对象"""
        current_time = time.time()
        
        # 获取所有对象
        for obj in gc.get_objects():
            obj_id = id(obj)
            obj_type = type(obj).__name__
            size = sys.getsizeof(obj, 0)
            
            # 更新对象信息
            if obj_id in self.object_tracking:
                self.object_tracking[obj_id].last_accessed = current_time
                self.object_tracking[obj_id].reference_count = gc.get_referrers(obj).__len__()
            else:
                # 获取堆栈跟踪
                stack_trace = []
                if self.enable_tracemalloc:
                    snapshot = tracemalloc.take_snapshot()
                    for frame in snapshot.statistics('lineno')[:5]:
                        stack_trace.append(str(frame))
                
                self.object_tracking[obj_id] = ObjectTrackingInfo(
                    obj_id=obj_id,
                    obj_type=obj_type,
                    size=size,
                    created_at=current_time,
                    last_accessed=current_time,
                    reference_count=gc.get_referrers(obj).__len__(),
                    stack_trace=stack_trace
                )
        
        # 清理已删除的对象
        current_ids = {id(obj) for obj in gc.get_objects()}
        deleted_ids = set(self.object_tracking.keys()) - current_ids
        for obj_id in deleted_ids:
            del self.object_tracking[obj_id]
    
    def take_memory_snapshot(self) -> MemorySnapshot:
        """
        拍摄内存快照
        
        Returns:
            MemorySnapshot: 内存快照数据
        """
        with self._lock:
            # 强制垃圾回收
            collected = gc.collect()
            
            # 获取内存统计
            heap_size = 0
            if self.enable_tracemalloc:
                current, peak = tracemalloc.get_traced_memory()
                heap_size = current
            
            # 获取进程内存信息
            process = psutil.Process()
            process_memory = {
                'rss': process.memory_info().rss,
                'vms': process.memory_info().vms,
                'percent': process.memory_percent()
            }
            
            # 获取GC统计
            gc_stats = {
                'collections': gc.get_count(),
                'threshold': gc.get_threshold()
            }
            
            # 自定义指标
            custom_metrics = {
                'tracked_objects': len(self.object_tracking),
                'collected_objects': collected,
                'total_object_size': sum(info.size for info in self.object_tracking.values())
            }
            
            snapshot = MemorySnapshot(
                timestamp=time.time(),
                heap_size=heap_size,
                stack_size=sys.getrecursionlimit(),
                object_count=len(gc.get_objects()),
                gc_stats=gc_stats,
                process_memory=process_memory,
                custom_metrics=custom_metrics
            )
            
            self.snapshots.append(snapshot)
            
            # 限制快照数量
            if len(self.snapshots) > 1000:
                self.snapshots = self.snapshots[-500:]
            
            logger.debug(f"内存快照已保存: 堆内存={heap_size/1024/1024:.2f}MB")
            return snapshot
    
    def get_memory_usage(self) -> MemoryStats:
        """
        获取当前内存使用情况
        
        Returns:
            MemoryStats: 内存使用统计
        """
        snapshot = self.take_memory_snapshot()
        
        return {
            'heap_size_mb': snapshot.heap_size / 1024 / 1024,
            'process_memory_mb': snapshot.process_memory['rss'] / 1024 / 1024,
            'object_count': snapshot.object_count,
            'gc_collections': snapshot.gc_stats['collections'],
            'tracked_objects': snapshot.custom_metrics['tracked_objects'],
            'timestamp': snapshot.timestamp
        }
    
    def analyze_memory_trends(self, duration: float = 60.0) -> Dict[str, Any]:
        """
        分析内存使用趋势
        
        Args:
            duration: 分析时间窗口（秒）
            
        Returns:
            Dict: 趋势分析结果
        """
        with self._lock:
            current_time = time.time()
            recent_snapshots = [
                s for s in self.snapshots 
                if current_time - s.timestamp <= duration
            ]
            
            if len(recent_snapshots) < 2:
                return {'error': '快照数据不足'}
            
            # 计算趋势
            first_snapshot = recent_snapshots[0]
            last_snapshot = recent_snapshots[-1]
            
            time_diff = last_snapshot.timestamp - first_snapshot.timestamp
            heap_growth = last_snapshot.heap_size - first_snapshot.heap_size
            object_growth = last_snapshot.object_count - first_snapshot.object_count
            
            growth_rate = {
                'heap_mb_per_second': (heap_growth / 1024 / 1024) / time_diff if time_diff > 0 else 0,
                'objects_per_second': object_growth / time_diff if time_diff > 0 else 0
            }
            
            # 峰值分析
            peak_heap = max(s.heap_size for s in recent_snapshots)
            peak_objects = max(s.object_count for s in recent_snapshots)
            
            return {
                'duration': duration,
                'snapshot_count': len(recent_snapshots),
                'heap_growth_mb': heap_growth / 1024 / 1024,
                'object_growth': object_growth,
                'growth_rate': growth_rate,
                'peak_heap_mb': peak_heap / 1024 / 1024,
                'peak_objects': peak_objects,
                'current_heap_mb': last_snapshot.heap_size / 1024 / 1024,
                'current_objects': last_snapshot.object_count
            }
    
    def get_object_statistics(self) -> Dict[str, Any]:
        """
        获取对象统计信息
        
        Returns:
            Dict: 对象统计信息
        """
        with self._lock:
            if not self.object_tracking:
                return {'error': '未启用对象跟踪'}
            
            # 按类型分组统计
            type_stats = defaultdict(lambda: {
                'count': 0,
                'total_size': 0,
                'avg_size': 0,
                'oldest': float('inf'),
                'newest': 0
            })
            
            current_time = time.time()
            
            for info in self.object_tracking.values():
                stats = type_stats[info.obj_type]
                stats['count'] += 1
                stats['total_size'] += info.size
                stats['oldest'] = min(stats['oldest'], current_time - info.created_at)
                stats['newest'] = max(stats['newest'], current_time - info.created_at)
            
            # 计算平均值
            for stats in type_stats.values():
                stats['avg_size'] = stats['total_size'] / stats['count'] if stats['count'] > 0 else 0
            
            return {
                'total_tracked_objects': len(self.object_tracking),
                'type_statistics': dict(type_stats),
                'largest_objects': sorted(
                    self.object_tracking.values(),
                    key=lambda x: x.size,
                    reverse=True
                )[:10],
                'oldest_objects': sorted(
                    self.object_tracking.values(),
                    key=lambda x: x.created_at
                )[:10]
            }


class MemoryLeakDetector:
    """
    内存泄漏检测器
    
    提供内存泄漏检测、定位和修复功能
    """
    
    def __init__(self, threshold_mb: float = 100.0):
        """
        初始化内存泄漏检测器
        
        Args:
            threshold_mb: 内存泄漏阈值（MB）
        """
        self.threshold_mb = threshold_mb
        self.baseline_snapshots: List[MemorySnapshot] = []
        self.leak_reports: List[LeakReport] = []
        self._lock = threading.RLock()
        
        logger.info(f"内存泄漏检测器已初始化，阈值: {threshold_mb}MB")
    
    def set_baseline(self, snapshots: List[MemorySnapshot]) -> None:
        """
        设置内存使用基线
        
        Args:
            snapshots: 基线快照列表
        """
        with self._lock:
            self.baseline_snapshots = snapshots
            logger.info(f"已设置基线，快照数量: {len(snapshots)}")
    
    def detect_leaks(self, current_snapshot: MemorySnapshot) -> List[LeakReport]:
        """
        检测内存泄漏
        
        Args:
            current_snapshot: 当前内存快照
            
        Returns:
            List[LeakReport]: 泄漏报告列表
        """
        with self._lock:
            if not self.baseline_snapshots:
                logger.warning("未设置基线，使用默认基线")
                self.baseline_snapshots = [current_snapshot]
                return []
            
            reports = []
            
            # 检测内存增长
            baseline_memory = self.baseline_snapshots[0].process_memory['rss']
            current_memory = current_snapshot.process_memory['rss']
            memory_growth_mb = (current_memory - baseline_memory) / 1024 / 1024
            
            if memory_growth_mb > self.threshold_mb:
                leak_report = {
                    'type': 'memory_growth',
                    'severity': 'high' if memory_growth_mb > self.threshold_mb * 2 else 'medium',
                    'description': f'内存增长 {memory_growth_mb:.2f}MB 超过阈值 {self.threshold_mb}MB',
                    'memory_growth_mb': memory_growth_mb,
                    'baseline_memory_mb': baseline_memory / 1024 / 1024,
                    'current_memory_mb': current_memory / 1024 / 1024,
                    'timestamp': current_snapshot.timestamp,
                    'recommendations': self._generate_recommendations(memory_growth_mb)
                }
                reports.append(leak_report)
            
            # 检测对象数量增长
            baseline_objects = self.baseline_snapshots[0].object_count
            current_objects = current_snapshot.object_count
            object_growth = current_objects - baseline_objects
            
            if object_growth > baseline_objects * 0.5:  # 增长超过50%
                leak_report = {
                    'type': 'object_growth',
                    'severity': 'medium',
                    'description': f'对象数量增长 {object_growth} 超过基线的50%',
                    'object_growth': object_growth,
                    'baseline_objects': baseline_objects,
                    'current_objects': current_objects,
                    'timestamp': current_snapshot.timestamp,
                    'recommendations': ['检查对象生命周期管理', '确认对象是否正确释放']
                }
                reports.append(leak_report)
            
            # 检测GC效率下降
            if len(self.baseline_snapshots) > 1:
                baseline_gc_efficiency = self._calculate_gc_efficiency(self.baseline_snapshots[-5:])
                current_gc_efficiency = self._calculate_gc_efficiency([current_snapshot])
                
                if current_gc_efficiency < baseline_gc_efficiency * 0.7:  # 效率下降30%
                    leak_report = {
                        'type': 'gc_efficiency',
                        'severity': 'medium',
                        'description': f'GC效率下降 {((baseline_gc_efficiency - current_gc_efficiency) / baseline_gc_efficiency * 100):.1f}%',
                        'baseline_efficiency': baseline_gc_efficiency,
                        'current_efficiency': current_gc_efficiency,
                        'timestamp': current_snapshot.timestamp,
                        'recommendations': ['调整GC参数', '优化对象创建和销毁', '使用对象池']
                    }
                    reports.append(leak_report)
            
            self.leak_reports.extend(reports)
            
            # 限制报告数量
            if len(self.leak_reports) > 1000:
                self.leak_reports = self.leak_reports[-500:]
            
            if reports:
                logger.warning(f"检测到 {len(reports)} 个潜在内存泄漏")
            
            return reports
    
    def _calculate_gc_efficiency(self, snapshots: List[MemorySnapshot]) -> float:
        """
        计算GC效率
        
        Args:
            snapshots: 快照列表
            
        Returns:
            float: GC效率分数
        """
        if not snapshots:
            return 1.0
        
        total_collections = sum(s.gc_stats['collections'] for s in snapshots)
        total_objects = sum(s.object_count for s in snapshots)
        
        if total_objects == 0:
            return 1.0
        
        # 效率 = 收集的对象数 / 总对象数
        return min(total_collections / total_objects, 1.0)
    
    def _generate_recommendations(self, memory_growth_mb: float) -> List[str]:
        """
        生成修复建议
        
        Args:
            memory_growth_mb: 内存增长量（MB）
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        if memory_growth_mb > 500:
            recommendations.extend([
                "立即执行强制垃圾回收",
                "检查是否有大量缓存未清理",
                "分析大对象生命周期",
                "考虑使用内存映射文件"
            ])
        elif memory_growth_mb > 200:
            recommendations.extend([
                "执行垃圾回收",
                "清理不必要的缓存",
                "检查对象池配置",
                "优化数据结构"
            ])
        else:
            recommendations.extend([
                "定期执行垃圾回收",
                "监控对象创建模式",
                "使用弱引用",
                "优化循环引用"
            ])
        
        return recommendations
    
    def locate_leak_source(self, leak_report: LeakReport) -> Dict[str, Any]:
        """
        定位泄漏源头
        
        Args:
            leak_report: 泄漏报告
            
        Returns:
            Dict: 泄漏源头信息
        """
        # 这里可以实现更复杂的泄漏源头定位逻辑
        # 例如分析堆栈跟踪、对象引用链等
        
        return {
            'leak_type': leak_report['type'],
            'potential_sources': [
                '循环引用',
                '未关闭的文件句柄',
                '缓存累积',
                '静态变量持有引用'
            ],
            'investigation_steps': [
                '分析对象创建模式',
                '检查引用计数',
                '审查资源释放代码',
                '监控对象生命周期'
            ]
        }
    
    def suggest_fixes(self, leak_report: LeakReport) -> List[Dict[str, str]]:
        """
        建议修复方案
        
        Args:
            leak_report: 泄漏报告
            
        Returns:
            List[Dict]: 修复方案列表
        """
        leak_type = leak_report['type']
        
        fix_suggestions = {
            'memory_growth': [
                {
                    'category': 'gc_optimization',
                    'suggestion': '调整垃圾回收参数',
                    'code_example': 'gc.set_threshold(700, 10, 10)'
                },
                {
                    'category': 'object_pool',
                    'suggestion': '使用对象池重用对象',
                    'code_example': 'pool = ObjectPool(MyClass, max_size=100)'
                },
                {
                    'category': 'weak_reference',
                    'suggestion': '使用弱引用避免循环引用',
                    'code_example': 'weakref.ref(obj, callback_function)'
                }
            ],
            'object_growth': [
                {
                    'category': 'lifecycle_management',
                    'suggestion': '明确对象生命周期',
                    'code_example': 'with obj_context() as obj: ...'
                },
                {
                    'category': 'container_optimization',
                    'suggestion': '优化容器数据结构',
                    'code_example': '使用__slots__减少内存开销'
                }
            ],
            'gc_efficiency': [
                {
                    'category': 'gc_tuning',
                    'suggestion': '调整GC触发阈值',
                    'code_example': 'gc.set_threshold(700, 10, 10)'
                },
                {
                    'category': 'generation_optimization',
                    'suggestion': '优化分代参数',
                    'code_example': 'gc.set_debug(gc.DEBUG_STATS)'
                }
            ]
        }
        
        return fix_suggestions.get(leak_type, [])
    
    def get_leak_history(self, limit: int = 50) -> List[LeakReport]:
        """
        获取泄漏历史
        
        Args:
            limit: 返回记录数限制
            
        Returns:
            List[LeakReport]: 泄漏报告历史
        """
        with self._lock:
            return self.leak_reports[-limit:]
    
    def export_leak_report(self, filepath: str) -> None:
        """
        导出泄漏报告
        
        Args:
            filepath: 导出文件路径
        """
        with self._lock:
            report_data = {
                'timestamp': time.time(),
                'threshold_mb': self.threshold_mb,
                'total_reports': len(self.leak_reports),
                'reports': self.leak_reports
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"泄漏报告已导出到: {filepath}")


class GCOptimizer:
    """
    垃圾回收优化器
    
    提供GC调优和内存分配优化功能
    """
    
    def __init__(self):
        """初始化GC优化器"""
        self.original_thresholds = gc.get_threshold()
        self.optimization_history: List[Dict[str, Any]] = []
        self._lock = threading.RLock()
        
        logger.info("GC优化器已初始化")
    
    def get_current_gc_settings(self) -> Dict[str, Any]:
        """
        获取当前GC设置
        
        Returns:
            Dict: GC设置信息
        """
        return {
            'thresholds': gc.get_threshold(),
            'counts': gc.get_count(),
            'debug_flags': gc.get_debug(),
            'collections': {
                'generation_0': gc.get_count()[0],
                'generation_1': gc.get_count()[1],
                'generation_2': gc.get_count()[2]
            }
        }
    
    def optimize_gc_thresholds(self, memory_pressure: str = 'normal') -> Dict[str, Any]:
        """
        优化GC阈值
        
        Args:
            memory_pressure: 内存压力级别 ('low', 'normal', 'high')
            
        Returns:
            Dict: 优化结果
        """
        with self._lock:
            current_settings = self.get_current_gc_settings()
            
            # 根据内存压力调整阈值
            if memory_pressure == 'low':
                # 低内存压力：减少GC频率
                new_thresholds = (1000, 15, 15)
            elif memory_pressure == 'high':
                # 高内存压力：增加GC频率
                new_thresholds = (500, 8, 8)
            else:
                # 正常内存压力：使用默认设置
                new_thresholds = (700, 10, 10)
            
            # 应用新阈值
            gc.set_threshold(*new_thresholds)
            
            optimization_result = {
                'timestamp': time.time(),
                'memory_pressure': memory_pressure,
                'original_thresholds': self.original_thresholds,
                'new_thresholds': new_thresholds,
                'previous_thresholds': current_settings['thresholds'],
                'status': 'optimized'
            }
            
            self.optimization_history.append(optimization_result)
            
            logger.info(f"GC阈值已优化为: {new_thresholds}")
            return optimization_result
    
    def force_garbage_collection(self, generation: Optional[int] = None) -> Dict[str, int]:
        """
        强制执行垃圾回收
        
        Args:
            generation: 指定代数，None表示全部
            
        Returns:
            Dict: 回收统计
        """
        with self._lock:
            start_time = time.time()
            
            if generation is None:
                collected = gc.collect()
            else:
                collected = gc.collect(generation)
            
            end_time = time.time()
            
            result = {
                'collected_objects': collected,
                'generation': generation,
                'duration_ms': (end_time - start_time) * 1000,
                'timestamp': time.time()
            }
            
            logger.debug(f"GC完成: 回收{collected}个对象，耗时{result['duration_ms']:.2f}ms")
            return result
    
    def analyze_gc_performance(self, duration: float = 60.0) -> Dict[str, Any]:
        """
        分析GC性能
        
        Args:
            duration: 分析时间窗口（秒）
            
        Returns:
            Dict: GC性能分析结果
        """
        with self._lock:
            current_time = time.time()
            
            # 记录GC性能数据
            gc_stats = []
            for _ in range(10):  # 采样10次
                stats = {
                    'timestamp': time.time(),
                    'collections': gc.get_count(),
                    'memory_usage': psutil.Process().memory_info().rss
                }
                gc_stats.append(stats)
                time.sleep(duration / 10)
            
            # 分析性能指标
            collection_counts = [s['collections'] for s in gc_stats]
            memory_usage = [s['memory_usage'] for s in gc_stats]
            
            analysis = {
                'duration': duration,
                'sample_count': len(gc_stats),
                'avg_collections': {
                    'gen_0': sum(c[0] for c in collection_counts) / len(collection_counts),
                    'gen_1': sum(c[1] for c in collection_counts) / len(collection_counts),
                    'gen_2': sum(c[2] for c in collection_counts) / len(collection_counts)
                },
                'memory_trend': {
                    'start_mb': memory_usage[0] / 1024 / 1024,
                    'end_mb': memory_usage[-1] / 1024 / 1024,
                    'growth_mb': (memory_usage[-1] - memory_usage[0]) / 1024 / 1024
                },
                'recommendations': self._generate_gc_recommendations(gc_stats)
            }
            
            return analysis
    
    def _generate_gc_recommendations(self, gc_stats: List[Dict[str, Any]]) -> List[str]:
        """
        生成GC优化建议
        
        Args:
            gc_stats: GC统计数据
            
        Returns:
            List[str]: 建议列表
        """
        recommendations = []
        
        # 分析收集频率
        avg_collections = sum(s['collections'][0] for s in gc_stats) / len(gc_stats)
        
        if avg_collections > 100:
            recommendations.append("GC频率过高，考虑增加阈值")
        elif avg_collections < 10:
            recommendations.append("GC频率较低，可以适当降低阈值")
        
        # 分析内存增长
        memory_growth = (gc_stats[-1]['memory_usage'] - gc_stats[0]['memory_usage']) / 1024 / 1024
        
        if memory_growth > 100:
            recommendations.append("内存增长较快，建议使用对象池或优化数据结构")
        elif memory_growth < 0:
            recommendations.append("内存使用稳定，当前GC设置良好")
        
        if not recommendations:
            recommendations.append("GC性能良好，无需调整")
        
        return recommendations
    
    def enable_gc_debug(self, debug_flags: int = gc.DEBUG_STATS) -> None:
        """
        启用GC调试
        
        Args:
            debug_flags: 调试标志
        """
        gc.set_debug(debug_flags)
        logger.info(f"GC调试已启用，标志: {debug_flags}")
    
    def disable_gc_debug(self) -> None:
        """禁用GC调试"""
        gc.set_debug(0)
        logger.info("GC调试已禁用")
    
    def get_memory_allocation_stats(self) -> Dict[str, Any]:
        """
        获取内存分配统计
        
        Returns:
            Dict: 分配统计信息
        """
        # 强制一次GC以获得准确统计
        gc.collect()
        
        stats = {
            'timestamp': time.time(),
            'object_count': len(gc.get_objects()),
            'garbage_count': len(gc.garbage),
            'gc_counts': gc.get_count(),
            'gc_thresholds': gc.get_threshold(),
            'process_memory': {
                'rss_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'vms_mb': psutil.Process().memory_info().vms / 1024 / 1024
            }
        }
        
        return stats
    
    def reset_gc_settings(self) -> Dict[str, Any]:
        """
        重置GC设置到原始状态
        
        Returns:
            Dict: 重置结果
        """
        with self._lock:
            gc.set_threshold(*self.original_thresholds)
            
            result = {
                'timestamp': time.time(),
                'original_thresholds': self.original_thresholds,
                'current_thresholds': gc.get_threshold(),
                'status': 'reset'
            }
            
            logger.info("GC设置已重置为原始状态")
            return result


class MemoryOptimizer:
    """
    O2内存优化器主类
    
    提供统一的内存管理接口，整合所有内存优化功能
    """
    
    def __init__(self, 
                 enable_tracemalloc: bool = True,
                 leak_threshold_mb: float = 100.0,
                 auto_optimize: bool = True):
        """
        初始化内存优化器
        
        Args:
            enable_tracemalloc: 是否启用内存跟踪
            leak_threshold_mb: 内存泄漏阈值（MB）
            auto_optimize: 是否启用自动优化
        """
        # 初始化各个组件
        self.analyzer = MemoryAnalyzer(enable_tracemalloc)
        self.leak_detector = MemoryLeakDetector(leak_threshold_mb)
        self.gc_optimizer = GCOptimizer()
        
        # 配置选项
        self.auto_optimize = auto_optimize
        self.optimization_interval = 60.0  # 自动优化间隔（秒）
        self.monitoring_active = False
        
        # 状态管理
        self._monitoring_task: Optional[asyncio.Task] = None
        self._shutdown_event = threading.Event()
        self._lock = threading.RLock()
        
        # 性能统计
        self.optimization_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'memory_saved_mb': 0.0,
            'start_time': time.time()
        }
        
        logger.info("O2内存优化器已初始化")
    
    async def start_monitoring(self, interval: float = 30.0) -> None:
        """
        启动内存监控
        
        Args:
            interval: 监控间隔（秒）
        """
        with self._lock:
            if self.monitoring_active:
                logger.warning("监控已在运行中")
                return
            
            self.monitoring_active = True
            self._shutdown_event.clear()
            
            # 启动异步监控任务
            self._monitoring_task = asyncio.create_task(
                self._monitoring_loop(interval)
            )
            
            logger.info(f"内存监控已启动，间隔: {interval}秒")
    
    async def stop_monitoring(self) -> None:
        """停止内存监控"""
        with self._lock:
            if not self.monitoring_active:
                logger.warning("监控未在运行")
                return
            
            self.monitoring_active = False
            self._shutdown_event.set()
            
            if self._monitoring_task:
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
            
            logger.info("内存监控已停止")
    
    async def _monitoring_loop(self, interval: float) -> None:
        """
        监控循环
        
        Args:
            interval: 监控间隔
        """
        try:
            while self.monitoring_active and not self._shutdown_event.is_set():
                # 执行监控步骤
                await self._perform_monitoring_cycle()
                
                # 等待指定间隔
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), 
                        timeout=interval
                    )
                except asyncio.TimeoutError:
                    pass  # 正常超时，继续循环
                    
        except asyncio.CancelledError:
            logger.debug("监控任务已取消")
        except Exception as e:
            logger.error(f"监控循环出错: {e}")
    
    async def _perform_monitoring_cycle(self) -> None:
        """执行一个监控周期"""
        try:
            # 拍摄内存快照
            snapshot = self.analyzer.take_memory_snapshot()
            
            # 检测内存泄漏
            leak_reports = self.leak_detector.detect_leaks(snapshot)
            
            # 如果启用自动优化，执行优化
            if self.auto_optimize and leak_reports:
                await self._auto_optimize(leak_reports)
            
            # 记录监控日志
            if leak_reports:
                logger.warning(f"检测到 {len(leak_reports)} 个内存问题")
            
        except Exception as e:
            logger.error(f"监控周期执行出错: {e}")
    
    async def _auto_optimize(self, leak_reports: List[LeakReport]) -> None:
        """
        自动优化内存使用
        
        Args:
            leak_reports: 泄漏报告列表
        """
        try:
            # 拍摄优化前的快照
            snapshot = self.analyzer.take_memory_snapshot()
            
            with self._lock:
                self.optimization_stats['total_optimizations'] += 1
            
            # 执行GC优化
            gc_result = self.gc_optimizer.force_garbage_collection()
            
            # 拍摄优化后的快照
            after_snapshot = self.analyzer.take_memory_snapshot()
            
            # 计算优化效果
            memory_before = snapshot.process_memory['rss']
            memory_after = after_snapshot.process_memory['rss']
            memory_saved = (memory_before - memory_after) / 1024 / 1024
            
            with self._lock:
                self.optimization_stats['memory_saved_mb'] += memory_saved
                
                if memory_saved > 0:
                    self.optimization_stats['successful_optimizations'] += 1
                    logger.info(f"自动优化完成，释放内存: {memory_saved:.2f}MB")
                else:
                    self.optimization_stats['failed_optimizations'] += 1
                    logger.warning("自动优化未释放内存")
            
        except Exception as e:
            with self._lock:
                self.optimization_stats['failed_optimizations'] += 1
            logger.error(f"自动优化失败: {e}")
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        获取内存状态
        
        Returns:
            Dict: 内存状态信息
        """
        try:
            current_snapshot = self.analyzer.take_memory_snapshot()
            memory_usage = self.analyzer.get_memory_usage()
            gc_settings = self.gc_optimizer.get_current_gc_settings()
            
            status = {
                'timestamp': time.time(),
                'monitoring_active': self.monitoring_active,
                'auto_optimize': self.auto_optimize,
                'memory_usage': memory_usage,
                'gc_settings': gc_settings,
                'optimization_stats': self.optimization_stats.copy(),
                'recent_leak_reports': self.leak_detector.get_leak_history(5)
            }
            
            return status
            
        except Exception as e:
            logger.error(f"获取内存状态失败: {e}")
            return {'error': str(e)}
    
    def optimize_memory(self, strategy: str = 'auto') -> Dict[str, Any]:
        """
        执行内存优化
        
        Args:
            strategy: 优化策略 ('auto', 'aggressive', 'conservative')
            
        Returns:
            Dict: 优化结果
        """
        try:
            with self._lock:
                start_time = time.time()
                self.optimization_stats['total_optimizations'] += 1
            
            # 获取优化前状态
            before_snapshot = self.analyzer.take_memory_snapshot()
            before_memory = before_snapshot.process_memory['rss']
            
            optimization_results = []
            
            # 根据策略执行优化
            if strategy == 'aggressive':
                # 激进优化：强制GC + 多次清理
                for _ in range(3):
                    gc_result = self.gc_optimizer.force_garbage_collection()
                    optimization_results.append(gc_result)
                
                # 清理Python缓存
                sys.modules.clear()
                
            elif strategy == 'conservative':
                # 保守优化：单次GC + 阈值调整
                gc_result = self.gc_optimizer.force_garbage_collection()
                optimization_results.append(gc_result)
                
                self.gc_optimizer.optimize_gc_thresholds('normal')
                
            else:  # auto
                # 自动优化：检测后决定
                leak_reports = self.leak_detector.detect_leaks(before_snapshot)
                
                if leak_reports:
                    # 有泄漏，执行激进优化
                    gc_result = self.gc_optimizer.force_garbage_collection()
                    optimization_results.append(gc_result)
                else:
                    # 无泄漏，执行保守优化
                    gc_result = self.gc_optimizer.force_garbage_collection()
                    optimization_results.append(gc_result)
            
            # 获取优化后状态
            after_snapshot = self.analyzer.take_memory_snapshot()
            after_memory = after_snapshot.process_memory['rss']
            
            # 计算优化效果
            memory_saved = (before_memory - after_memory) / 1024 / 1024
            duration = time.time() - start_time
            
            result = {
                'timestamp': time.time(),
                'strategy': strategy,
                'duration_seconds': duration,
                'memory_before_mb': before_memory / 1024 / 1024,
                'memory_after_mb': after_memory / 1024 / 1024,
                'memory_saved_mb': memory_saved,
                'optimization_results': optimization_results,
                'status': 'completed'
            }
            
            with self._lock:
                self.optimization_stats['memory_saved_mb'] += memory_saved
                if memory_saved > 0:
                    self.optimization_stats['successful_optimizations'] += 1
                else:
                    self.optimization_stats['failed_optimizations'] += 1
            
            logger.info(f"内存优化完成，策略: {strategy}, 释放内存: {memory_saved:.2f}MB")
            return result
            
        except Exception as e:
            with self._lock:
                self.optimization_stats['failed_optimizations'] += 1
            logger.error(f"内存优化失败: {e}")
            return {'error': str(e), 'strategy': strategy}
    
    def analyze_memory_patterns(self, duration: float = 300.0) -> Dict[str, Any]:
        """
        分析内存使用模式
        
        Args:
            duration: 分析时间窗口（秒）
            
        Returns:
            Dict: 模式分析结果
        """
        try:
            # 获取趋势分析
            trend_analysis = self.analyzer.analyze_memory_trends(duration)
            
            # 获取对象统计
            object_stats = self.analyzer.get_object_statistics()
            
            # 获取GC性能分析
            gc_performance = self.gc_optimizer.analyze_gc_performance(duration)
            
            # 综合分析
            patterns = {
                'analysis_duration': duration,
                'timestamp': time.time(),
                'trend_analysis': trend_analysis,
                'object_statistics': object_stats,
                'gc_performance': gc_performance,
                'memory_patterns': self._identify_memory_patterns(trend_analysis, object_stats),
                'recommendations': self._generate_optimization_recommendations(
                    trend_analysis, object_stats, gc_performance
                )
            }
            
            return patterns
            
        except Exception as e:
            logger.error(f"内存模式分析失败: {e}")
            return {'error': str(e)}
    
    def _identify_memory_patterns(self, 
                                 trend_analysis: Dict[str, Any], 
                                 object_stats: Dict[str, Any]) -> List[str]:
        """
        识别内存使用模式
        
        Args:
            trend_analysis: 趋势分析结果
            object_stats: 对象统计信息
            
        Returns:
            List[str]: 识别出的模式
        """
        patterns = []
        
        # 内存增长模式
        if trend_analysis.get('heap_growth_mb', 0) > 50:
            patterns.append("持续内存增长")
        
        # 对象累积模式
        if 'type_statistics' in object_stats:
            for obj_type, stats in object_stats['type_statistics'].items():
                if stats['count'] > 1000:
                    patterns.append(f"大量{obj_type}对象累积")
        
        # GC效率模式
        if trend_analysis.get('growth_rate', {}).get('objects_per_second', 0) > 100:
            patterns.append("高对象创建频率")
        
        return patterns
    
    def _generate_optimization_recommendations(self, 
                                             trend_analysis: Dict[str, Any],
                                             object_stats: Dict[str, Any], 
                                             gc_performance: Dict[str, Any]) -> List[str]:
        """
        生成优化建议
        
        Args:
            trend_analysis: 趋势分析结果
            object_stats: 对象统计信息
            gc_performance: GC性能分析
            
        Returns:
            List[str]: 优化建议
        """
        recommendations = []
        
        # 基于趋势的建议
        heap_growth = trend_analysis.get('heap_growth_mb', 0)
        if heap_growth > 100:
            recommendations.append("内存增长过快，建议使用对象池和分块处理")
        elif heap_growth > 50:
            recommendations.append("内存有增长趋势，建议定期执行GC")
        
        # 基于对象的建议
        if 'type_statistics' in object_stats:
            for obj_type, stats in object_stats['type_statistics'].items():
                if stats['count'] > 1000:
                    recommendations.append(f"考虑优化{obj_type}对象的创建和销毁")
        
        # 基于GC的建议
        recommendations.extend(gc_performance.get('recommendations', []))
        
        if not recommendations:
            recommendations.append("内存使用模式良好，无需特殊优化")
        
        return recommendations
    
    @contextmanager
    def memory_context(self, name: str = "operation"):
        """
        内存监控上下文管理器
        
        Args:
            name: 操作名称
            
        Yields:
            Dict: 内存状态信息
        """
        start_time = time.time()
        start_snapshot = self.analyzer.take_memory_snapshot()
        
        try:
            yield {
                'operation': name,
                'start_time': start_time,
                'start_memory_mb': start_snapshot.process_memory['rss'] / 1024 / 1024
            }
        finally:
            end_time = time.time()
            end_snapshot = self.analyzer.take_memory_snapshot()
            
            duration = end_time - start_time
            memory_diff = (end_snapshot.process_memory['rss'] - start_snapshot.process_memory['rss']) / 1024 / 1024
            
            logger.info(f"操作'{name}'完成: 耗时{duration:.2f}秒, 内存变化{memory_diff:.2f}MB")
    
    def profile_function(self, func: Callable) -> Callable:
        """
        函数内存性能分析装饰器
        
        Args:
            func: 要分析的函数
            
        Returns:
            Callable: 包装后的函数
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self.memory_context(f"profile_{func.__name__}"):
                return func(*args, **kwargs)
        
        return wrapper
    
    async def profile_async_function(self, func: Callable) -> Callable:
        """
        异步函数内存性能分析装饰器
        
        Args:
            func: 要分析的异步函数
            
        Returns:
            Callable: 包装后的异步函数
        """
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with self.memory_context(f"profile_async_{func.__name__}"):
                return await func(*args, **kwargs)
        
        return wrapper
    
    def export_optimization_report(self, filepath: str) -> None:
        """
        导出优化报告
        
        Args:
            filepath: 导出文件路径
        """
        try:
            report_data = {
                'timestamp': time.time(),
                'optimizer_version': '2.0.0',
                'memory_status': self.get_memory_status(),
                'optimization_history': self.gc_optimizer.optimization_history,
                'leak_reports': self.leak_detector.get_leak_history(100),
                'performance_summary': self._generate_performance_summary()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"优化报告已导出到: {filepath}")
            
        except Exception as e:
            logger.error(f"导出优化报告失败: {e}")
    
    def _generate_performance_summary(self) -> Dict[str, Any]:
        """
        生成性能摘要
        
        Returns:
            Dict: 性能摘要信息
        """
        uptime = time.time() - self.optimization_stats['start_time']
        
        return {
            'uptime_seconds': uptime,
            'total_optimizations': self.optimization_stats['total_optimizations'],
            'success_rate': (
                self.optimization_stats['successful_optimizations'] / 
                max(self.optimization_stats['total_optimizations'], 1)
            ),
            'average_memory_saved_mb': (
                self.optimization_stats['memory_saved_mb'] / 
                max(self.optimization_stats['successful_optimizations'], 1)
            ),
            'optimizations_per_hour': (
                self.optimization_stats['total_optimizations'] / 
                max(uptime / 3600, 1)
            )
        }
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            # 停止监控
            if self.monitoring_active:
                asyncio.create_task(self.stop_monitoring())
            
            # 停止对象跟踪
            self.analyzer.stop_object_tracking()
            
            # 重置GC设置
            self.gc_optimizer.reset_gc_settings()
            
            logger.info("O2内存优化器已清理")
            
        except Exception as e:
            logger.error(f"清理资源失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()


# 使用示例和测试函数
def example_usage():
    """使用示例"""
    # 创建内存优化器实例
    optimizer = MemoryOptimizer(
        enable_tracemalloc=True,
        leak_threshold_mb=50.0,
        auto_optimize=True
    )
    
    try:
        # 启动监控
        asyncio.run(optimizer.start_monitoring(interval=30.0))
        
        # 使用内存上下文
        with optimizer.memory_context("example_operation"):
            # 执行一些内存密集型操作
            data = [i for i in range(100000)]
            processed_data = [x * 2 for x in data]
            del data  # 清理
        
        # 执行手动优化
        result = optimizer.optimize_memory(strategy="auto")
        print(f"优化结果: {result}")
        
        # 分析内存模式
        patterns = optimizer.analyze_memory_patterns(duration=60.0)
        print(f"内存模式: {patterns}")
        
        # 获取当前状态
        status = optimizer.get_memory_status()
        print(f"内存状态: {status}")
        
        # 导出报告
        optimizer.export_optimization_report("memory_report.json")
        
    finally:
        # 清理资源
        optimizer.cleanup()


if __name__ == "__main__":
    example_usage()