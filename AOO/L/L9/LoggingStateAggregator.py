#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L9日志状态聚合器模块

该模块提供了一个全面的日志状态聚合和管理系统，包含以下主要功能：
- 日志状态监控（日志记录状态、日志存储状态、日志传输状态）
- 日志协调管理（日志收集、日志过滤、日志聚合）
- 日志生命周期管理（日志创建、日志归档、日志删除）
- 日志性能统计（日志数量、日志大小、处理速度）
- 日志健康检查（日志有效性、日志完整性、日志安全性）
- 统一日志接口和API
- 异步日志状态同步和分布式协调
- 日志告警和通知系统
"""

import asyncio
import json
import time
import logging
import threading
import queue
import hashlib
import gzip
import shutil
import os
import sys
import weakref
import traceback
import signal
import datetime
from typing import Dict, List, Optional, Any, Callable, Union, Set, Tuple, AsyncIterator
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
# 可选依赖库检查
try:
    import aiofiles
    HAS_AIOFILES = True
except ImportError:
    HAS_AIOFILES = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False

try:
    import redis
    HAS_REDIS = True
except ImportError:
    HAS_REDIS = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False


# =============================================================================
# 基础数据结构和枚举定义
# =============================================================================

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogStatus(Enum):
    """日志状态枚举"""
    CREATED = "CREATED"
    PROCESSING = "PROCESSING"
    STORED = "STORED"
    TRANSMITTED = "TRANSMITTED"
    ARCHIVED = "ARCHIVED"
    DELETED = "DELETED"
    FAILED = "FAILED"


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"


class AlertLevel(Enum):
    """告警级别枚举"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AggregationType(Enum):
    """聚合类型枚举"""
    TIME_BASED = "TIME_BASED"
    LEVEL_BASED = "LEVEL_BASED"
    SOURCE_BASED = "SOURCE_BASED"
    CONTENT_BASED = "CONTENT_BASED"


@dataclass
class LogEntry:
    """日志条目数据结构"""
    id: str
    timestamp: float
    level: LogLevel
    message: str
    source: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""
    size: int = 0
    status: LogStatus = LogStatus.CREATED
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.content_hash:
            self.content_hash = hashlib.sha256(
                f"{self.message}{self.metadata}".encode()
            ).hexdigest()
        if not self.size:
            self.size = len(json.dumps(self.to_dict()))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'timestamp': self.timestamp,
            'level': self.level.value,
            'message': self.message,
            'source': self.source,
            'metadata': self.metadata,
            'content_hash': self.content_hash,
            'size': self.size,
            'status': self.status.value
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogEntry':
        """从字典创建日志条目"""
        return cls(
            id=data['id'],
            timestamp=data['timestamp'],
            level=LogLevel(data['level']),
            message=data['message'],
            source=data['source'],
            metadata=data.get('metadata', {}),
            content_hash=data.get('content_hash', ''),
            size=data.get('size', 0),
            status=LogStatus(data.get('status', 'CREATED'))
        )


@dataclass
class LogStatistics:
    """日志统计数据结构"""
    total_logs: int = 0
    total_size: int = 0
    logs_by_level: Dict[LogLevel, int] = field(default_factory=lambda: defaultdict(int))
    logs_by_source: Dict[str, int] = field(default_factory=dict)
    average_size: float = 0.0
    processing_rate: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'total_logs': self.total_logs,
            'total_size': self.total_size,
            'logs_by_level': {k.value: v for k, v in self.logs_by_level.items()},
            'logs_by_source': self.logs_by_source,
            'average_size': self.average_size,
            'processing_rate': self.processing_rate,
            'timestamp': self.timestamp
        }


@dataclass
class HealthCheckResult:
    """健康检查结果数据结构"""
    component: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'component': self.component,
            'status': self.status.value,
            'message': self.message,
            'details': self.details,
            'timestamp': self.timestamp
        }


@dataclass
class Alert:
    """告警数据结构"""
    id: str
    level: AlertLevel
    title: str
    message: str
    component: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'level': self.level.value,
            'title': self.title,
            'message': self.message,
            'component': self.component,
            'timestamp': self.timestamp,
            'resolved': self.resolved,
            'metadata': self.metadata
        }


# =============================================================================
# 异常定义
# =============================================================================

class LoggingStateAggregatorError(Exception):
    """日志状态聚合器基础异常"""
    pass


class LogProcessingError(LoggingStateAggregatorError):
    """日志处理异常"""
    pass


class LogStorageError(LoggingStateAggregatorError):
    """日志存储异常"""
    pass


class LogTransmissionError(LoggingStateAggregatorError):
    """日志传输异常"""
    pass


class LogHealthCheckError(LoggingStateAggregatorError):
    """日志健康检查异常"""
    pass


class LogAggregationError(LoggingStateAggregatorError):
    """日志聚合异常"""
    pass


# =============================================================================
# 核心组件类
# =============================================================================

class LogStateMonitor:
    """日志状态监控器"""
    
    def __init__(self, aggregator: 'LoggingStateAggregator'):
        self.aggregator = weakref.ref(aggregator)
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._monitoring_interval = 5.0  # 监控间隔（秒）
        self._is_monitoring = False
        
    async def start_monitoring(self):
        """启动监控"""
        if self._is_monitoring:
            return
            
        self._is_monitoring = True
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 启动各种监控任务
        self._monitoring_tasks['record_status'] = asyncio.create_task(
            self._monitor_record_status()
        )
        self._monitoring_tasks['storage_status'] = asyncio.create_task(
            self._monitor_storage_status()
        )
        self._monitoring_tasks['transmission_status'] = asyncio.create_task(
            self._monitor_transmission_status()
        )
        
        self.logger.info("日志状态监控已启动")
    
    async def stop_monitoring(self):
        """停止监控"""
        self._is_monitoring = False
        
        # 取消所有监控任务
        for task in self._monitoring_tasks.values():
            task.cancel()
        
        # 等待任务完成
        await asyncio.gather(*self._monitoring_tasks.values(), return_exceptions=True)
        self._monitoring_tasks.clear()
        
        self.logger.info("日志状态监控已停止")
    
    async def _monitor_record_status(self):
        """监控日志记录状态"""
        while self._is_monitoring:
            try:
                stats = self.aggregator().get_statistics()
                if stats.total_logs > 0:
                    self.logger.debug(f"记录状态监控: 总日志数={stats.total_logs}")
                
                await asyncio.sleep(self._monitoring_interval)
            except Exception as e:
                self.logger.error(f"记录状态监控异常: {e}")
                await asyncio.sleep(self._monitoring_interval)
    
    async def _monitor_storage_status(self):
        """监控日志存储状态"""
        while self._is_monitoring:
            try:
                # 检查存储空间
                storage_info = await self._check_storage_status()
                if storage_info['status'] == HealthStatus.CRITICAL:
                    await self.aggregator().alert_system.trigger_alert(
                        AlertLevel.HIGH,
                        "存储空间不足",
                        f"可用空间: {storage_info['free_space']}GB",
                        "LogStateMonitor"
                    )
                
                await asyncio.sleep(self._monitoring_interval)
            except Exception as e:
                self.logger.error(f"存储状态监控异常: {e}")
                await asyncio.sleep(self._monitoring_interval)
    
    async def _monitor_transmission_status(self):
        """监控日志传输状态"""
        while self._is_monitoring:
            try:
                # 检查传输队列
                queue_size = self.aggregator().transmission_queue.qsize()
                if queue_size > 1000:
                    await self.aggregator().alert_system.trigger_alert(
                        AlertLevel.MEDIUM,
                        "传输队列积压",
                        f"队列大小: {queue_size}",
                        "LogStateMonitor"
                    )
                
                await asyncio.sleep(self._monitoring_interval)
            except Exception as e:
                self.logger.error(f"传输状态监控异常: {e}")
                await asyncio.sleep(self._monitoring_interval)
    
    async def _check_storage_status(self) -> Dict[str, Any]:
        """检查存储状态"""
        try:
            if not HAS_PSUTIL:
                return {
                    'status': HealthStatus.UNKNOWN,
                    'free_space': 0,
                    'total_space': 0,
                    'usage_percent': 0,
                    'error': 'psutil库不可用'
                }
            
            # 获取磁盘使用情况
            disk_usage = psutil.disk_usage('/')
            free_space_gb = disk_usage.free / (1024**3)
            total_space_gb = disk_usage.total / (1024**3)
            usage_percent = disk_usage.percent
            
            if free_space_gb < 1:  # 小于1GB
                status = HealthStatus.CRITICAL
            elif free_space_gb < 5 or usage_percent > 90:  # 小于5GB或使用率>90%
                status = HealthStatus.WARNING
            else:
                status = HealthStatus.HEALTHY
            
            return {
                'status': status,
                'free_space': round(free_space_gb, 2),
                'total_space': round(total_space_gb, 2),
                'usage_percent': usage_percent
            }
        except Exception as e:
            self.logger.error(f"存储状态检查异常: {e}")
            return {
                'status': HealthStatus.UNKNOWN,
                'free_space': 0,
                'total_space': 0,
                'usage_percent': 0
            }


class LogLifecycleManager:
    """日志生命周期管理器"""
    
    def __init__(self, aggregator: 'LoggingStateAggregator'):
        self.aggregator = weakref.ref(aggregator)
        self._cleanup_tasks: Dict[str, asyncio.Task] = {}
        self._cleanup_interval = 3600  # 清理间隔（秒）
        self._max_log_age = 30 * 24 * 3600  # 最大日志年龄（30天）
        self._max_log_size = 1024 * 1024 * 1024  # 最大日志大小（1GB）
        self._is_running = False
        
    async def start_lifecycle_management(self):
        """启动生命周期管理"""
        if self._is_running:
            return
            
        self._is_running = True
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 启动生命周期管理任务
        self._cleanup_tasks['auto_archive'] = asyncio.create_task(
            self._auto_archive_logs()
        )
        self._cleanup_tasks['auto_delete'] = asyncio.create_task(
            self._auto_delete_logs()
        )
        self._cleanup_tasks['compact_storage'] = asyncio.create_task(
            self._compact_storage()
        )
        
        self.logger.info("日志生命周期管理已启动")
    
    async def stop_lifecycle_management(self):
        """停止生命周期管理"""
        self._is_running = False
        
        # 取消所有生命周期管理任务
        for task in self._cleanup_tasks.values():
            task.cancel()
        
        # 等待任务完成
        await asyncio.gather(*self._cleanup_tasks.values(), return_exceptions=True)
        self._cleanup_tasks.clear()
        
        self.logger.info("日志生命周期管理已停止")
    
    async def create_log(self, entry: LogEntry) -> bool:
        """创建新日志"""
        try:
            # 验证日志条目
            if not self._validate_log_entry(entry):
                raise LogProcessingError("日志条目验证失败")
            
            # 存储日志
            await self.aggregator().storage_manager.store_log(entry)
            
            # 更新统计信息
            self.aggregator().statistics_manager.update_statistics(entry)
            
            # 添加到传输队列
            await self.aggregator().add_to_transmission_queue(entry)
            
            self.logger.debug(f"日志创建成功: {entry.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"日志创建失败: {e}")
            entry.status = LogStatus.FAILED
            return False
    
    async def archive_logs(self, log_ids: List[str]) -> bool:
        """归档日志"""
        try:
            archived_count = 0
            for log_id in log_ids:
                # 获取日志条目
                entry = await self.aggregator().storage_manager.get_log(log_id)
                if not entry:
                    continue
                
                # 移动到归档存储
                await self._move_to_archive(entry)
                
                # 更新状态
                entry.status = LogStatus.ARCHIVED
                await self.aggregator().storage_manager.update_log(entry)
                
                archived_count += 1
            
            self.logger.info(f"日志归档完成: {archived_count}/{len(log_ids)}")
            return archived_count == len(log_ids)
            
        except Exception as e:
            self.logger.error(f"日志归档失败: {e}")
            return False
    
    async def delete_logs(self, log_ids: List[str]) -> bool:
        """删除日志"""
        try:
            deleted_count = 0
            for log_id in log_ids:
                # 获取日志条目
                entry = await self.aggregator().storage_manager.get_log(log_id)
                if not entry:
                    continue
                
                # 从存储中删除
                await self.aggregator().storage_manager.delete_log(log_id)
                
                # 更新状态
                entry.status = LogStatus.DELETED
                
                deleted_count += 1
            
            self.logger.info(f"日志删除完成: {deleted_count}/{len(log_ids)}")
            return deleted_count == len(log_ids)
            
        except Exception as e:
            self.logger.error(f"日志删除失败: {e}")
            return False
    
    async def _auto_archive_logs(self):
        """自动归档过期日志"""
        while self._is_running:
            try:
                cutoff_time = time.time() - self._max_log_age
                expired_logs = await self.aggregator().storage_manager.get_logs_by_time_range(
                    0, cutoff_time
                )
                
                if expired_logs:
                    log_ids = [log.id for log in expired_logs]
                    await self.archive_logs(log_ids)
                    self.logger.info(f"自动归档过期日志: {len(log_ids)}条")
                
                await asyncio.sleep(self._cleanup_interval)
            except Exception as e:
                self.logger.error(f"自动归档异常: {e}")
                await asyncio.sleep(self._cleanup_interval)
    
    async def _auto_delete_logs(self):
        """自动删除已归档的旧日志"""
        while self._is_running:
            try:
                # 获取已归档的日志
                archived_logs = await self.aggregator().storage_manager.get_logs_by_status(
                    LogStatus.ARCHIVED
                )
                
                # 删除30天前的归档日志
                cutoff_time = time.time() - (60 * 24 * 3600)  # 60天
                old_archived = [
                    log for log in archived_logs 
                    if log.timestamp < cutoff_time
                ]
                
                if old_archived:
                    log_ids = [log.id for log in old_archived]
                    await self.delete_logs(log_ids)
                    self.logger.info(f"自动删除旧归档日志: {len(log_ids)}条")
                
                await asyncio.sleep(self._cleanup_interval)
            except Exception as e:
                self.logger.error(f"自动删除异常: {e}")
                await asyncio.sleep(self._cleanup_interval)
    
    async def _compact_storage(self):
        """压缩存储"""
        while self._is_running:
            try:
                # 检查存储大小
                storage_size = await self.aggregator().storage_manager.get_storage_size()
                
                if storage_size > self._max_log_size:
                    # 执行压缩
                    await self._compress_old_logs()
                    self.logger.info("存储压缩完成")
                
                await asyncio.sleep(self._cleanup_interval)
            except Exception as e:
                self.logger.error(f"存储压缩异常: {e}")
                await asyncio.sleep(self._cleanup_interval)
    
    async def _move_to_archive(self, entry: LogEntry):
        """移动日志到归档存储"""
        archive_path = self.aggregator().config.get('archive_path', './archive')
        os.makedirs(archive_path, exist_ok=True)
        
        archive_file = os.path.join(archive_path, f"{entry.id}.json.gz")
        
        # 压缩并保存
        with gzip.open(archive_file, 'wt', encoding='utf-8') as f:
            json.dump(entry.to_dict(), f, ensure_ascii=False, indent=2)
    
    async def _compress_old_logs(self):
        """压缩旧日志"""
        storage_path = self.aggregator().config.get('storage_path', './logs')
        
        # 查找需要压缩的文件
        cutoff_time = time.time() - (7 * 24 * 3600)  # 7天前
        for filename in os.listdir(storage_path):
            file_path = os.path.join(storage_path, filename)
            if os.path.isfile(file_path):
                file_mtime = os.path.getmtime(file_path)
                if file_mtime < cutoff_time and not filename.endswith('.gz'):
                    # 压缩文件
                    compressed_path = f"{file_path}.gz"
                    with open(file_path, 'rb') as f_in:
                        with gzip.open(compressed_path, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # 删除原文件
                    os.remove(file_path)
    
    def _validate_log_entry(self, entry: LogEntry) -> bool:
        """验证日志条目"""
        if not entry.id or not entry.message or not entry.source:
            return False
        
        if entry.timestamp <= 0:
            return False
        
        if not isinstance(entry.level, LogLevel):
            return False
        
        return True


class LogStatisticsManager:
    """日志统计管理器"""
    
    def __init__(self, aggregator: 'LoggingStateAggregator'):
        self.aggregator = weakref.ref(aggregator)
        self._statistics = LogStatistics()
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=1000)  # 保留最近1000条统计记录
        
    def update_statistics(self, entry: LogEntry):
        """更新统计信息"""
        with self._lock:
            self._statistics.total_logs += 1
            self._statistics.total_size += entry.size
            self._statistics.logs_by_level[entry.level] += 1
            self._statistics.logs_by_source[entry.source] = \
                self._statistics.logs_by_source.get(entry.source, 0) + 1
            
            # 计算平均大小
            if self._statistics.total_logs > 0:
                self._statistics.average_size = \
                    self._statistics.total_size / self._statistics.total_logs
            
            # 计算处理速度（需要基于时间窗口）
            self._calculate_processing_rate()
            
            # 保存历史记录
            self._history.append({
                'timestamp': time.time(),
                'stats': self._statistics.to_dict()
            })
    
    def _calculate_processing_rate(self):
        """计算处理速度"""
        # 基于最近1分钟的数据计算处理速度
        current_time = time.time()
        recent_logs = 0
        
        for record in reversed(self._history):
            if current_time - record['timestamp'] <= 60:
                recent_logs += record['stats']['total_logs']
            else:
                break
        
        self._statistics.processing_rate = recent_logs / 60.0  # 每分钟处理数量
    
    def get_statistics(self) -> LogStatistics:
        """获取当前统计信息"""
        with self._lock:
            return LogStatistics(
                total_logs=self._statistics.total_logs,
                total_size=self._statistics.total_size,
                logs_by_level=dict(self._statistics.logs_by_level),
                logs_by_source=dict(self._statistics.logs_by_source),
                average_size=self._statistics.average_size,
                processing_rate=self._statistics.processing_rate,
                timestamp=self._statistics.timestamp
            )
    
    def get_statistics_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取统计历史"""
        with self._lock:
            return list(self._history)[-limit:]
    
    def reset_statistics(self):
        """重置统计信息"""
        with self._lock:
            self._statistics = LogStatistics()
            self._history.clear()


class LogHealthChecker:
    """日志健康检查器"""
    
    def __init__(self, aggregator: 'LoggingStateAggregator'):
        self.aggregator = weakref.ref(aggregator)
        self._health_checks: Dict[str, Callable] = {}
        self._last_check_results: Dict[str, HealthCheckResult] = {}
        self._check_interval = 30  # 检查间隔（秒）
        self._is_running = False
        
        # 注册健康检查项
        self._register_health_checks()
    
    def _register_health_checks(self):
        """注册健康检查项"""
        self._health_checks = {
            'log_validity': self._check_log_validity,
            'log_integrity': self._check_log_integrity,
            'log_security': self._check_log_security,
            'storage_health': self._check_storage_health,
            'transmission_health': self._check_transmission_health,
            'system_resources': self._check_system_resources
        }
    
    async def start_health_checks(self):
        """启动健康检查"""
        if self._is_running:
            return
            
        self._is_running = True
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 启动健康检查任务
        asyncio.create_task(self._periodic_health_checks())
        
        self.logger.info("日志健康检查已启动")
    
    async def stop_health_checks(self):
        """停止健康检查"""
        self._is_running = False
        self.logger.info("日志健康检查已停止")
    
    async def perform_health_check(self, check_name: str) -> HealthCheckResult:
        """执行指定的健康检查"""
        if check_name not in self._health_checks:
            raise LogHealthCheckError(f"未知的健康检查项: {check_name}")
        
        try:
            result = await self._health_checks[check_name]()
            self._last_check_results[check_name] = result
            return result
        except Exception as e:
            self.logger.error(f"健康检查 {check_name} 执行失败: {e}")
            return HealthCheckResult(
                component=check_name,
                status=HealthStatus.CRITICAL,
                message=f"健康检查执行失败: {str(e)}",
                details={'error': str(e)}
            )
    
    async def perform_all_health_checks(self) -> Dict[str, HealthCheckResult]:
        """执行所有健康检查"""
        results = {}
        for check_name in self._health_checks.keys():
            results[check_name] = await self.perform_health_check(check_name)
        return results
    
    async def _periodic_health_checks(self):
        """定期健康检查"""
        while self._is_running:
            try:
                results = await self.perform_all_health_checks()
                
                # 检查是否有严重问题
                critical_issues = [
                    name for name, result in results.items()
                    if result.status == HealthStatus.CRITICAL
                ]
                
                if critical_issues:
                    await self.aggregator().alert_system.trigger_alert(
                        AlertLevel.CRITICAL,
                        "关键健康检查失败",
                        f"失败的检查项: {', '.join(critical_issues)}",
                        "LogHealthChecker"
                    )
                
                await asyncio.sleep(self._check_interval)
            except Exception as e:
                self.logger.error(f"定期健康检查异常: {e}")
                await asyncio.sleep(self._check_interval)
    
    async def _check_log_validity(self) -> HealthCheckResult:
        """检查日志有效性"""
        try:
            # 检查最近的日志条目是否有效
            recent_logs = await self.aggregator().storage_manager.get_recent_logs(100)
            invalid_count = 0
            
            for log in recent_logs:
                if not self._validate_log_entry(log):
                    invalid_count += 1
            
            if invalid_count > len(recent_logs) * 0.1:  # 超过10%无效
                status = HealthStatus.CRITICAL
                message = f"无效日志比例过高: {invalid_count}/{len(recent_logs)}"
            elif invalid_count > 0:
                status = HealthStatus.WARNING
                message = f"存在无效日志: {invalid_count}条"
            else:
                status = HealthStatus.HEALTHY
                message = "所有日志有效"
            
            return HealthCheckResult(
                component="log_validity",
                status=status,
                message=message,
                details={
                    'total_checked': len(recent_logs),
                    'invalid_count': invalid_count,
                    'validity_rate': (len(recent_logs) - invalid_count) / len(recent_logs) if recent_logs else 1.0
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="log_validity",
                status=HealthStatus.CRITICAL,
                message=f"日志有效性检查异常: {str(e)}",
                details={'error': str(e)}
            )
    
    async def _check_log_integrity(self) -> HealthCheckResult:
        """检查日志完整性"""
        try:
            # 检查日志内容哈希是否匹配
            recent_logs = await self.aggregator().storage_manager.get_recent_logs(50)
            corrupted_count = 0
            
            for log in recent_logs:
                expected_hash = hashlib.sha256(
                    f"{log.message}{log.metadata}".encode()
                ).hexdigest()
                if log.content_hash != expected_hash:
                    corrupted_count += 1
            
            if corrupted_count > 0:
                status = HealthStatus.CRITICAL
                message = f"发现损坏日志: {corrupted_count}条"
            else:
                status = HealthStatus.HEALTHY
                message = "所有日志完整性良好"
            
            return HealthCheckResult(
                component="log_integrity",
                status=status,
                message=message,
                details={
                    'total_checked': len(recent_logs),
                    'corrupted_count': corrupted_count,
                    'integrity_rate': (len(recent_logs) - corrupted_count) / len(recent_logs) if recent_logs else 1.0
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="log_integrity",
                status=HealthStatus.CRITICAL,
                message=f"日志完整性检查异常: {str(e)}",
                details={'error': str(e)}
            )
    
    async def _check_log_security(self) -> HealthCheckResult:
        """检查日志安全性"""
        try:
            # 检查是否有敏感信息泄露
            recent_logs = await self.aggregator().storage_manager.get_recent_logs(100)
            security_issues = 0
            
            sensitive_patterns = [
                'password', 'token', 'key', 'secret', 'private',
                'credit_card', 'ssn', 'api_key'
            ]
            
            for log in recent_logs:
                message_lower = log.message.lower()
                for pattern in sensitive_patterns:
                    if pattern in message_lower:
                        security_issues += 1
                        break
            
            if security_issues > 0:
                status = HealthStatus.WARNING
                message = f"发现潜在安全风险: {security_issues}条日志"
            else:
                status = HealthStatus.HEALTHY
                message = "未发现安全风险"
            
            return HealthCheckResult(
                component="log_security",
                status=status,
                message=message,
                details={
                    'total_checked': len(recent_logs),
                    'security_issues': security_issues,
                    'security_score': (len(recent_logs) - security_issues) / len(recent_logs) if recent_logs else 1.0
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="log_security",
                status=HealthStatus.CRITICAL,
                message=f"日志安全性检查异常: {str(e)}",
                details={'error': str(e)}
            )
    
    async def _check_storage_health(self) -> HealthCheckResult:
        """检查存储健康状况"""
        try:
            if not HAS_PSUTIL:
                return HealthCheckResult(
                    component="storage_health",
                    status=HealthStatus.UNKNOWN,
                    message="psutil库不可用，无法检查存储健康状况",
                    details={'error': 'psutil not available'}
                )
            
            # 检查磁盘空间
            disk_usage = psutil.disk_usage('/')
            free_percent = (disk_usage.free / disk_usage.total) * 100
            
            # 检查存储性能
            storage_response_time = await self._measure_storage_performance()
            
            if free_percent < 5 or storage_response_time > 1000:
                status = HealthStatus.CRITICAL
                message = f"存储健康状况严重: 可用空间{free_percent:.1f}%, 响应时间{storage_response_time}ms"
            elif free_percent < 15 or storage_response_time > 500:
                status = HealthStatus.WARNING
                message = f"存储健康状况一般: 可用空间{free_percent:.1f}%, 响应时间{storage_response_time}ms"
            else:
                status = HealthStatus.HEALTHY
                message = f"存储健康状况良好: 可用空间{free_percent:.1f}%, 响应时间{storage_response_time}ms"
            
            return HealthCheckResult(
                component="storage_health",
                status=status,
                message=message,
                details={
                    'free_space_percent': free_percent,
                    'response_time_ms': storage_response_time,
                    'total_space_gb': disk_usage.total / (1024**3),
                    'free_space_gb': disk_usage.free / (1024**3)
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="storage_health",
                status=HealthStatus.CRITICAL,
                message=f"存储健康检查异常: {str(e)}",
                details={'error': str(e)}
            )
    
    async def _check_transmission_health(self) -> HealthCheckResult:
        """检查传输健康状况"""
        try:
            # 检查传输队列状态
            queue_size = self.aggregator().transmission_queue.qsize()
            queue_max_size = self.aggregator().config.get('max_queue_size', 10000)
            
            # 检查传输成功率
            transmission_stats = await self.aggregator().transmission_manager.get_transmission_stats()
            success_rate = transmission_stats.get('success_rate', 1.0)
            
            if queue_size > queue_max_size * 0.9 or success_rate < 0.8:
                status = HealthStatus.CRITICAL
                message = f"传输健康状况严重: 队列大小{queue_size}, 成功率{success_rate:.2%}"
            elif queue_size > queue_max_size * 0.5 or success_rate < 0.95:
                status = HealthStatus.WARNING
                message = f"传输健康状况一般: 队列大小{queue_size}, 成功率{success_rate:.2%}"
            else:
                status = HealthStatus.HEALTHY
                message = f"传输健康状况良好: 队列大小{queue_size}, 成功率{success_rate:.2%}"
            
            return HealthCheckResult(
                component="transmission_health",
                status=status,
                message=message,
                details={
                    'queue_size': queue_size,
                    'queue_max_size': queue_max_size,
                    'success_rate': success_rate,
                    'queue_utilization': queue_size / queue_max_size
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="transmission_health",
                status=HealthStatus.CRITICAL,
                message=f"传输健康检查异常: {str(e)}",
                details={'error': str(e)}
            )
    
    async def _check_system_resources(self) -> HealthCheckResult:
        """检查系统资源"""
        try:
            if not HAS_PSUTIL:
                return HealthCheckResult(
                    component="system_resources",
                    status=HealthStatus.UNKNOWN,
                    message="psutil库不可用，无法检查系统资源",
                    details={'error': 'psutil not available'}
                )
            
            # 检查CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # 检查内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # 检查磁盘IO
            disk_io = psutil.disk_io_counters()
            if disk_io:
                disk_read_mb = disk_io.read_bytes / (1024 * 1024)
                disk_write_mb = disk_io.write_bytes / (1024 * 1024)
            else:
                disk_read_mb = disk_write_mb = 0
            
            # 综合评估
            if cpu_percent > 90 or memory_percent > 90:
                status = HealthStatus.CRITICAL
                message = f"系统资源严重不足: CPU{cpu_percent}%, 内存{memory_percent}%"
            elif cpu_percent > 80 or memory_percent > 80:
                status = HealthStatus.WARNING
                message = f"系统资源使用较高: CPU{cpu_percent}%, 内存{memory_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = f"系统资源充足: CPU{cpu_percent}%, 内存{memory_percent}%"
            
            return HealthCheckResult(
                component="system_resources",
                status=status,
                message=message,
                details={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory_percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_read_mb': disk_read_mb,
                    'disk_write_mb': disk_write_mb
                }
            )
        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"系统资源检查异常: {str(e)}",
                details={'error': str(e)}
            )
    
    async def _measure_storage_performance(self) -> float:
        """测量存储性能（响应时间）"""
        try:
            start_time = time.time()
            # 执行一个简单的存储操作来测量性能
            test_entry = LogEntry(
                id=f"perf_test_{int(time.time())}",
                timestamp=time.time(),
                level=LogLevel.INFO,
                message="性能测试消息",
                source="performance_test"
            )
            
            await self.aggregator().storage_manager.store_log(test_entry)
            await self.aggregator().storage_manager.get_log(test_entry.id)
            await self.aggregator().storage_manager.delete_log(test_entry.id)
            
            end_time = time.time()
            return (end_time - start_time) * 1000  # 转换为毫秒
        except Exception:
            return 1000.0  # 默认值表示性能较差
    
    def _validate_log_entry(self, entry: LogEntry) -> bool:
        """验证日志条目"""
        if not entry.id or not entry.message or not entry.source:
            return False
        
        if entry.timestamp <= 0:
            return False
        
        if not isinstance(entry.level, LogLevel):
            return False
        
        return True
    
    def get_last_check_results(self) -> Dict[str, HealthCheckResult]:
        """获取最近的检查结果"""
        return dict(self._last_check_results)


class LogAlertSystem:
    """日志告警系统"""
    
    def __init__(self, aggregator: 'LoggingStateAggregator'):
        self.aggregator = weakref.ref(aggregator)
        self._alert_handlers: Dict[AlertLevel, List[Callable]] = defaultdict(list)
        self._alert_history: deque = deque(maxlen=10000)
        self._alert_cooldown: Dict[str, float] = {}
        self._cooldown_duration = 300  # 5分钟冷却时间
        self._is_running = False
        
        # 注册默认告警处理器
        self._register_default_handlers()
    
    def _register_default_handlers(self):
        """注册默认告警处理器"""
        # 日志处理器
        self._alert_handlers[AlertLevel.CRITICAL].append(self._log_critical_alert)
        self._alert_handlers[AlertLevel.HIGH].append(self._log_high_alert)
        self._alert_handlers[AlertLevel.MEDIUM].append(self._log_medium_alert)
        self._alert_handlers[AlertLevel.LOW].append(self._log_low_alert)
        
        # 文件处理器
        self._alert_handlers[AlertLevel.CRITICAL].append(self._file_critical_alert)
        self._alert_handlers[AlertLevel.HIGH].append(self._file_high_alert)
    
    async def start_alert_system(self):
        """启动告警系统"""
        if self._is_running:
            return
            
        self._is_running = True
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 启动告警处理任务
        asyncio.create_task(self._process_alert_queue())
        
        self.logger.info("日志告警系统已启动")
    
    async def stop_alert_system(self):
        """停止告警系统"""
        self._is_running = False
        self.logger.info("日志告警系统已停止")
    
    async def trigger_alert(self, level: AlertLevel, title: str, message: str, 
                          component: str, metadata: Dict[str, Any] = None) -> str:
        """触发告警"""
        alert_id = hashlib.md5(
            f"{level.value}{title}{message}{component}{time.time()}".encode()
        ).hexdigest()
        
        alert = Alert(
            id=alert_id,
            level=level,
            title=title,
            message=message,
            component=component,
            metadata=metadata or {}
        )
        
        # 检查冷却时间
        alert_key = f"{component}:{title}"
        if alert_key in self._alert_cooldown:
            if time.time() - self._alert_cooldown[alert_key] < self._cooldown_duration:
                self.logger.debug(f"告警在冷却期内，跳过: {alert_key}")
                return alert_id
        
        # 添加到历史记录
        self._alert_history.append(alert)
        self._alert_cooldown[alert_key] = time.time()
        
        # 处理告警
        await self._process_alert(alert)
        
        self.logger.info(f"告警已触发: {alert_id} - {title}")
        return alert_id
    
    async def _process_alert(self, alert: Alert):
        """处理告警"""
        try:
            # 调用所有注册的处理器
            for handler in self._alert_handlers[alert.level]:
                try:
                    await handler(alert)
                except Exception as e:
                    self.logger.error(f"告警处理器执行失败: {e}")
        except Exception as e:
            self.logger.error(f"告警处理失败: {e}")
    
    async def _process_alert_queue(self):
        """处理告警队列"""
        while self._is_running:
            try:
                # 清理过期的冷却记录
                current_time = time.time()
                expired_keys = [
                    key for key, timestamp in self._alert_cooldown.items()
                    if current_time - timestamp > self._cooldown_duration * 2
                ]
                for key in expired_keys:
                    del self._alert_cooldown[key]
                
                await asyncio.sleep(60)  # 每分钟清理一次
            except Exception as e:
                self.logger.error(f"告警队列处理异常: {e}")
                await asyncio.sleep(60)
    
    async def _log_critical_alert(self, alert: Alert):
        """记录严重告警"""
        self.logger.critical(
            f"严重告警 [{alert.component}] {alert.title}: {alert.message}"
        )
    
    async def _log_high_alert(self, alert: Alert):
        """记录高级告警"""
        self.logger.error(
            f"高级告警 [{alert.component}] {alert.title}: {alert.message}"
        )
    
    async def _log_medium_alert(self, alert: Alert):
        """记录中级告警"""
        self.logger.warning(
            f"中级告警 [{alert.component}] {alert.title}: {alert.message}"
        )
    
    async def _log_low_alert(self, alert: Alert):
        """记录低级告警"""
        self.logger.info(
            f"低级告警 [{alert.component}] {alert.title}: {alert.message}"
        )
    
    async def _file_critical_alert(self, alert: Alert):
        """文件记录严重告警"""
        try:
            alert_file = self.aggregator().config.get('alert_log_file', './alerts.log')
            alert_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'level': alert.level.value,
                'component': alert.component,
                'title': alert.title,
                'message': alert.message,
                'id': alert.id
            }
            
            if HAS_AIOFILES:
                async with aiofiles.open(alert_file, 'a', encoding='utf-8') as f:
                    await f.write(json.dumps(alert_data, ensure_ascii=False) + '\n')
            else:
                # fallback到标准文件操作
                with open(alert_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(alert_data, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f"严重告警文件记录失败: {e}")
    
    async def _file_high_alert(self, alert: Alert):
        """文件记录高级告警"""
        try:
            alert_file = self.aggregator().config.get('alert_log_file', './alerts.log')
            alert_data = {
                'timestamp': datetime.datetime.now().isoformat(),
                'level': alert.level.value,
                'component': alert.component,
                'title': alert.title,
                'message': alert.message,
                'id': alert.id
            }
            
            if HAS_AIOFILES:
                async with aiofiles.open(alert_file, 'a', encoding='utf-8') as f:
                    await f.write(json.dumps(alert_data, ensure_ascii=False) + '\n')
            else:
                # fallback到标准文件操作
                with open(alert_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(alert_data, ensure_ascii=False) + '\n')
        except Exception as e:
            self.logger.error(f"高级告警文件记录失败: {e}")
    
    def get_alert_history(self, limit: int = 100, level: AlertLevel = None) -> List[Alert]:
        """获取告警历史"""
        alerts = list(self._alert_history)
        
        if level:
            alerts = [alert for alert in alerts if alert.level == level]
        
        return alerts[-limit:]
    
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        for alert in self._alert_history:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False
    
    def add_alert_handler(self, level: AlertLevel, handler: Callable):
        """添加告警处理器"""
        self._alert_handlers[level].append(handler)
    
    def remove_alert_handler(self, level: AlertLevel, handler: Callable):
        """移除告警处理器"""
        if handler in self._alert_handlers[level]:
            self._alert_handlers[level].remove(handler)


class LogCoordinator:
    """日志协调器"""
    
    def __init__(self, aggregator: 'LoggingStateAggregator'):
        self.aggregator = weakref.ref(aggregator)
        self._coordinators: Dict[str, Any] = {}
        self._coordination_lock = asyncio.Lock()
        self._is_running = False
        
    async def start_coordination(self):
        """启动协调"""
        if self._is_running:
            return
            
        self._is_running = True
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 启动协调任务
        asyncio.create_task(self._coordinate_log_flow())
        
        self.logger.info("日志协调已启动")
    
    async def stop_coordination(self):
        """停止协调"""
        self._is_running = False
        self.logger.info("日志协调已停止")
    
    async def coordinate_log_collection(self, logs: List[LogEntry]) -> List[LogEntry]:
        """协调日志收集"""
        async with self._coordination_lock:
            collected_logs = []
            
            for log in logs:
                try:
                    # 收集日志
                    collected = await self.aggregator().lifecycle_manager.create_log(log)
                    if collected:
                        collected_logs.append(log)
                except Exception as e:
                    self.logger.error(f"日志收集协调失败: {e}")
            
            return collected_logs
    
    async def coordinate_log_filtering(self, logs: List[LogEntry], 
                                     filters: Dict[str, Any]) -> List[LogEntry]:
        """协调日志过滤"""
        filtered_logs = []
        
        for log in logs:
            if self._apply_filters(log, filters):
                filtered_logs.append(log)
        
        return filtered_logs
    
    async def coordinate_log_aggregation(self, logs: List[LogEntry], 
                                       aggregation_type: AggregationType) -> Dict[str, Any]:
        """协调日志聚合"""
        if aggregation_type == AggregationType.TIME_BASED:
            return await self._aggregate_by_time(logs)
        elif aggregation_type == AggregationType.LEVEL_BASED:
            return await self._aggregate_by_level(logs)
        elif aggregation_type == AggregationType.SOURCE_BASED:
            return await self._aggregate_by_source(logs)
        elif aggregation_type == AggregationType.CONTENT_BASED:
            return await self._aggregate_by_content(logs)
        else:
            raise LogAggregationError(f"不支持的聚合类型: {aggregation_type}")
    
    async def _coordinate_log_flow(self):
        """协调日志流"""
        while self._is_running:
            try:
                # 定期检查和优化日志流
                await self._optimize_log_flow()
                await asyncio.sleep(30)  # 每30秒优化一次
            except Exception as e:
                self.logger.error(f"日志流协调异常: {e}")
                await asyncio.sleep(30)
    
    async def _optimize_log_flow(self):
        """优化日志流"""
        try:
            # 检查传输队列状态
            queue_size = self.aggregator().transmission_queue.qsize()
            
            if queue_size > 5000:
                # 队列积压，增加处理线程
                try:
                    await self.aggregator().transmission_manager.increase_workers()
                except AttributeError:
                    pass  # transmission_manager可能还没完全初始化
            elif queue_size < 100:
                # 队列空闲，减少处理线程
                try:
                    await self.aggregator().transmission_manager.decrease_workers()
                except AttributeError:
                    pass  # transmission_manager可能还没完全初始化
            
            # 检查存储状态
            try:
                storage_stats = await self.aggregator().storage_manager.get_storage_stats()
                if storage_stats['utilization'] > 0.8:
                    # 存储使用率过高，触发清理
                    await self.aggregator().lifecycle_manager._auto_archive_logs()
                    await self.aggregator().lifecycle_manager._auto_delete_logs()
            except AttributeError:
                pass  # storage_manager可能还没完全初始化
            
        except Exception as e:
            self.logger.error(f"日志流优化异常: {e}")
    
    def _apply_filters(self, log: LogEntry, filters: Dict[str, Any]) -> bool:
        """应用过滤条件"""
        # 级别过滤
        if 'level' in filters:
            if log.level.value not in filters['level']:
                return False
        
        # 时间范围过滤
        if 'start_time' in filters and log.timestamp < filters['start_time']:
            return False
        if 'end_time' in filters and log.timestamp > filters['end_time']:
            return False
        
        # 源过滤
        if 'source' in filters and log.source not in filters['source']:
            return False
        
        # 消息内容过滤
        if 'message_contains' in filters:
            if filters['message_contains'] not in log.message:
                return False
        
        # 元数据过滤
        if 'metadata' in filters:
            for key, value in filters['metadata'].items():
                if key not in log.metadata or log.metadata[key] != value:
                    return False
        
        return True
    
    async def _aggregate_by_time(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """按时间聚合"""
        time_buckets = defaultdict(list)
        
        for log in logs:
            # 按小时分组
            hour_bucket = int(log.timestamp // 3600) * 3600
            time_buckets[hour_bucket].append(log)
        
        result = {}
        for timestamp, bucket_logs in time_buckets.items():
            result[str(timestamp)] = {
                'count': len(bucket_logs),
                'logs': [log.to_dict() for log in bucket_logs],
                'time_range': {
                    'start': timestamp,
                    'end': timestamp + 3600
                }
            }
        
        return result
    
    async def _aggregate_by_level(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """按级别聚合"""
        level_buckets = defaultdict(list)
        
        for log in logs:
            level_buckets[log.level.value].append(log)
        
        result = {}
        for level, bucket_logs in level_buckets.items():
            result[level] = {
                'count': len(bucket_logs),
                'logs': [log.to_dict() for log in bucket_logs],
                'percentage': len(bucket_logs) / len(logs) * 100
            }
        
        return result
    
    async def _aggregate_by_source(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """按源聚合"""
        source_buckets = defaultdict(list)
        
        for log in logs:
            source_buckets[log.source].append(log)
        
        result = {}
        for source, bucket_logs in source_buckets.items():
            result[source] = {
                'count': len(bucket_logs),
                'logs': [log.to_dict() for log in bucket_logs],
                'percentage': len(bucket_logs) / len(logs) * 100
            }
        
        return result
    
    async def _aggregate_by_content(self, logs: List[LogEntry]) -> Dict[str, Any]:
        """按内容聚合"""
        content_buckets = defaultdict(list)
        
        for log in logs:
            # 按消息长度分组
            length_bucket = len(log.message) // 100 * 100  # 每100字符一个区间
            content_buckets[length_bucket].append(log)
        
        result = {}
        for length, bucket_logs in content_buckets.items():
            result[f"{length}-{length+99}"] = {
                'count': len(bucket_logs),
                'logs': [log.to_dict() for log in bucket_logs],
                'length_range': f"{length}-{length+99}"
            }
        
        return result


# =============================================================================
# 存储管理器
# =============================================================================

class LogStorageManager:
    """日志存储管理器"""
    
    def __init__(self, aggregator: 'LoggingStateAggregator'):
        self.aggregator = weakref.ref(aggregator)
        self._storage_path = aggregator.config.get('storage_path', './logs')
        self._index_file = os.path.join(self._storage_path, 'index.json')
        self._lock = asyncio.Lock()
        self._index: Dict[str, Dict[str, Any]] = {}
        
        # 确保存储目录存在
        os.makedirs(self._storage_path, exist_ok=True)
        
        # 加载索引
        self._load_index()
    
    def _load_index(self):
        """加载索引文件"""
        try:
            if os.path.exists(self._index_file):
                with open(self._index_file, 'r', encoding='utf-8') as f:
                    self._index = json.load(f)
        except Exception as e:
            logging.getLogger(__name__).error(f"加载索引文件失败: {e}")
            self._index = {}
    
    async def _save_index(self):
        """保存索引文件"""
        try:
            if HAS_AIOFILES:
                async with aiofiles.open(self._index_file, 'w', encoding='utf-8') as f:
                    await f.write(json.dumps(self._index, ensure_ascii=False, indent=2))
            else:
                # fallback到标准文件操作
                with open(self._index_file, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(self._index, ensure_ascii=False, indent=2))
        except Exception as e:
            logging.getLogger(__name__).error(f"保存索引文件失败: {e}")
    
    async def store_log(self, entry: LogEntry) -> bool:
        """存储日志"""
        async with self._lock:
            try:
                # 生成文件路径
                file_path = os.path.join(self._storage_path, f"{entry.id}.json")
                
                # 存储日志文件
                if HAS_AIOFILES:
                    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                        await f.write(json.dumps(entry.to_dict(), ensure_ascii=False, indent=2))
                else:
                    # fallback到标准文件操作
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(json.dumps(entry.to_dict(), ensure_ascii=False, indent=2))
                
                # 更新索引
                self._index[entry.id] = {
                    'file_path': file_path,
                    'timestamp': entry.timestamp,
                    'level': entry.level.value,
                    'source': entry.source,
                    'size': entry.size,
                    'status': entry.status.value
                }
                
                # 定期保存索引
                if len(self._index) % 100 == 0:
                    await self._save_index()
                
                return True
            except Exception as e:
                logging.getLogger(__name__).error(f"存储日志失败: {e}")
                return False
    
    async def get_log(self, log_id: str) -> Optional[LogEntry]:
        """获取日志"""
        async with self._lock:
            try:
                if log_id not in self._index:
                    return None
                
                file_path = self._index[log_id]['file_path']
                
                if not os.path.exists(file_path):
                    # 文件不存在，从索引中删除
                    del self._index[log_id]
                    return None
                
                if HAS_AIOFILES:
                    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                        data = json.loads(await f.read())
                else:
                    # fallback到标准文件操作
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                
                return LogEntry.from_dict(data)
            except Exception as e:
                logging.getLogger(__name__).error(f"获取日志失败: {e}")
                return None
    
    async def delete_log(self, log_id: str) -> bool:
        """删除日志"""
        async with self._lock:
            try:
                if log_id not in self._index:
                    return True  # 已经不存在，视为删除成功
                
                file_path = self._index[log_id]['file_path']
                
                # 删除文件
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                # 从索引中删除
                del self._index[log_id]
                
                # 定期保存索引
                if len(self._index) % 100 == 0:
                    await self._save_index()
                
                return True
            except Exception as e:
                logging.getLogger(__name__).error(f"删除日志失败: {e}")
                return False
    
    async def update_log(self, entry: LogEntry) -> bool:
        """更新日志"""
        return await self.store_log(entry)
    
    async def get_recent_logs(self, limit: int = 100) -> List[LogEntry]:
        """获取最近的日志"""
        async with self._lock:
            try:
                # 按时间戳排序
                sorted_logs = sorted(
                    self._index.items(),
                    key=lambda x: x[1]['timestamp'],
                    reverse=True
                )
                
                recent_logs = []
                for log_id, _ in sorted_logs[:limit]:
                    log = await self.get_log(log_id)
                    if log:
                        recent_logs.append(log)
                
                return recent_logs
            except Exception as e:
                logging.getLogger(__name__).error(f"获取最近日志失败: {e}")
                return []
    
    async def get_logs_by_time_range(self, start_time: float, end_time: float) -> List[LogEntry]:
        """按时间范围获取日志"""
        async with self._lock:
            try:
                logs = []
                for log_id, info in self._index.items():
                    if start_time <= info['timestamp'] <= end_time:
                        log = await self.get_log(log_id)
                        if log:
                            logs.append(log)
                
                return logs
            except Exception as e:
                logging.getLogger(__name__).error(f"按时间范围获取日志失败: {e}")
                return []
    
    async def get_logs_by_status(self, status: LogStatus) -> List[LogEntry]:
        """按状态获取日志"""
        async with self._lock:
            try:
                logs = []
                for log_id, info in self._index.items():
                    if info['status'] == status.value:
                        log = await self.get_log(log_id)
                        if log:
                            logs.append(log)
                
                return logs
            except Exception as e:
                logging.getLogger(__name__).error(f"按状态获取日志失败: {e}")
                return []
    
    async def get_storage_size(self) -> int:
        """获取存储大小"""
        try:
            total_size = 0
            for info in self._index.values():
                file_path = info['file_path']
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
            return total_size
        except Exception as e:
            logging.getLogger(__name__).error(f"获取存储大小失败: {e}")
            return 0
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            total_logs = len(self._index)
            total_size = await self.get_storage_size()
            
            # 按级别统计
            level_stats = defaultdict(int)
            for info in self._index.values():
                level_stats[info['level']] += 1
            
            # 按状态统计
            status_stats = defaultdict(int)
            for info in self._index.values():
                status_stats[info['status']] += 1
            
            # 存储利用率
            if HAS_PSUTIL:
                disk_usage = psutil.disk_usage(self._storage_path)
                utilization = (disk_usage.used / disk_usage.total) * 100
                free_space_gb = disk_usage.free / (1024**3)
                total_space_gb = disk_usage.total / (1024**3)
            else:
                utilization = 0
                free_space_gb = 0
                total_space_gb = 0
            
            return {
                'total_logs': total_logs,
                'total_size': total_size,
                'level_stats': dict(level_stats),
                'status_stats': dict(status_stats),
                'utilization': utilization,
                'free_space_gb': free_space_gb,
                'total_space_gb': total_space_gb,
                'psutil_available': HAS_PSUTIL
            }
        except Exception as e:
            logging.getLogger(__name__).error(f"获取存储统计失败: {e}")
            return {}


# =============================================================================
# 传输管理器
# =============================================================================

class LogTransmissionManager:
    """日志传输管理器"""
    
    def __init__(self, aggregator: 'LoggingStateAggregator'):
        self.aggregator = weakref.ref(aggregator)
        self._workers = 3
        self._max_workers = 10
        self._min_workers = 1
        self._transmission_stats = {
            'total_sent': 0,
            'total_failed': 0,
            'success_rate': 1.0
        }
        self._transmission_tasks = []
        self._lock = asyncio.Lock()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    async def start_transmission(self):
        """启动传输"""
        # 启动传输工作线程
        self._transmission_tasks = []
        for i in range(self._workers):
            task = asyncio.create_task(self._transmission_worker(f"worker-{i}"))
            self._transmission_tasks.append(task)
        
        self.logger.info(f"日志传输已启动，工作线程数: {self._workers}")
        
        # 不等待任务完成，让它们在后台运行
        # 注意：这里不会返回，传输管理器持续运行
    
    async def stop_transmission(self):
        """停止传输"""
        self.logger.info("日志传输已停止")
    
    async def _transmission_worker(self, worker_id: str):
        """传输工作线程"""
        while True:
            try:
                # 从传输队列获取日志
                try:
                    entry = await asyncio.wait_for(
                        self.aggregator().transmission_queue.get(), 
                        timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                
                # 传输日志
                success = await self._transmit_log(entry)
                
                # 更新统计信息
                async with self._lock:
                    if success:
                        self._transmission_stats['total_sent'] += 1
                    else:
                        self._transmission_stats['total_failed'] += 1
                    
                    total = self._transmission_stats['total_sent'] + self._transmission_stats['total_failed']
                    if total > 0:
                        self._transmission_stats['success_rate'] = self._transmission_stats['total_sent'] / total
                
                # 标记任务完成
                self.aggregator().transmission_queue.task_done()
                
            except Exception as e:
                self.logger.error(f"传输工作线程 {worker_id} 异常: {e}")
                await asyncio.sleep(1)
    
    async def _transmit_log(self, entry: LogEntry) -> bool:
        """传输单个日志"""
        try:
            # 模拟日志传输
            # 实际实现中，这里会连接到远程日志服务器
            
            # 检查传输目标
            transmission_targets = self.aggregator().config.get('transmission_targets', [])
            
            if not transmission_targets:
                # 没有配置传输目标，仅更新状态
                entry.status = LogStatus.TRANSMITTED
                await self.aggregator().storage_manager.update_log(entry)
                return True
            
            # 发送到所有目标
            success_count = 0
            for target in transmission_targets:
                try:
                    if await self._send_to_target(entry, target):
                        success_count += 1
                except Exception as e:
                    self.logger.error(f"发送到目标 {target} 失败: {e}")
            
            # 如果至少有一个目标成功，则认为传输成功
            if success_count > 0:
                entry.status = LogStatus.TRANSMITTED
                await self.aggregator().storage_manager.update_log(entry)
                return True
            else:
                entry.status = LogStatus.FAILED
                await self.aggregator().storage_manager.update_log(entry)
                return False
                
        except Exception as e:
            self.logger.error(f"日志传输异常: {e}")
            entry.status = LogStatus.FAILED
            await self.aggregator().storage_manager.update_log(entry)
            return False
    
    async def _send_to_target(self, entry: LogEntry, target: Dict[str, Any]) -> bool:
        """发送到指定目标"""
        try:
            target_type = target.get('type', 'file')
            
            if target_type == 'file':
                return await self._send_to_file(entry, target)
            elif target_type == 'http':
                return await self._send_to_http(entry, target)
            elif target_type == 'redis':
                return await self._send_to_redis(entry, target)
            else:
                self.logger.warning(f"不支持的传输目标类型: {target_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"发送到目标异常: {e}")
            return False
    
    async def _send_to_file(self, entry: LogEntry, target: Dict[str, Any]) -> bool:
        """发送到文件目标"""
        try:
            file_path = target.get('path', './remote_logs.log')
            
            log_data = {
                'timestamp': datetime.datetime.fromtimestamp(entry.timestamp).isoformat(),
                'level': entry.level.value,
                'message': entry.message,
                'source': entry.source,
                'metadata': entry.metadata
            }
            
            if HAS_AIOFILES:
                async with aiofiles.open(file_path, 'a', encoding='utf-8') as f:
                    await f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
            else:
                # fallback到标准文件操作
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_data, ensure_ascii=False) + '\n')
            
            return True
        except Exception as e:
            self.logger.error(f"文件传输失败: {e}")
            return False
    
    async def _send_to_http(self, entry: LogEntry, target: Dict[str, Any]) -> bool:
        """发送到HTTP目标"""
        try:
            if not HAS_AIOHTTP:
                self.logger.warning("aiohttp库不可用，跳过HTTP传输")
                return False
            
            url = target.get('url')
            if not url:
                return False
            
            headers = target.get('headers', {'Content-Type': 'application/json'})
            
            data = {
                'timestamp': entry.timestamp,
                'level': entry.level.value,
                'message': entry.message,
                'source': entry.source,
                'metadata': entry.metadata
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=data, headers=headers) as response:
                    return response.status == 200
                    
        except Exception as e:
            self.logger.error(f"HTTP传输失败: {e}")
            return False
    
    async def _send_to_redis(self, entry: LogEntry, target: Dict[str, Any]) -> bool:
        """发送到Redis目标"""
        try:
            if not HAS_REDIS:
                self.logger.warning("Redis库不可用，跳过Redis传输")
                return False
            
            redis_config = target.get('redis', {})
            redis_host = redis_config.get('host', 'localhost')
            redis_port = redis_config.get('port', 6379)
            redis_db = redis_config.get('db', 0)
            
            # 创建Redis连接
            r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
            
            # 存储日志
            log_key = f"log:{entry.id}"
            log_data = json.dumps(entry.to_dict(), ensure_ascii=False)
            
            r.set(log_key, log_data, ex=86400)  # 24小时过期
            
            return True
        except Exception as e:
            self.logger.error(f"Redis传输失败: {e}")
            return False
    
    async def increase_workers(self):
        """增加工作线程"""
        if self._workers < self._max_workers:
            self._workers += 1
            self.logger.info(f"增加传输工作线程，当前数量: {self._workers}")
    
    async def decrease_workers(self):
        """减少工作线程"""
        if self._workers > self._min_workers:
            self._workers -= 1
            self.logger.info(f"减少传输工作线程，当前数量: {self._workers}")
    
    async def get_transmission_stats(self) -> Dict[str, Any]:
        """获取传输统计信息"""
        async with self._lock:
            return dict(self._transmission_stats)


# =============================================================================
# 主要的日志状态聚合器类
# =============================================================================

class LoggingStateAggregator:
    """
    L9日志状态聚合器主类
    
    这是一个综合性的日志管理系统，提供以下功能：
    - 日志状态监控和管理
    - 日志生命周期管理
    - 日志统计和分析
    - 日志健康检查
    - 日志告警系统
    - 日志协调和聚合
    - 异步日志处理
    - 分布式日志协调
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化日志状态聚合器
        
        Args:
            config: 配置字典，包含各种设置参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 传输队列
        self.transmission_queue = asyncio.Queue(
            maxsize=self.config.get('max_queue_size', 10000)
        )
        
        # 初始化各个组件
        self.storage_manager = LogStorageManager(self)
        self.statistics_manager = LogStatisticsManager(self)
        self.health_checker = LogHealthChecker(self)
        self.alert_system = LogAlertSystem(self)
        self.lifecycle_manager = LogLifecycleManager(self)
        self.state_monitor = LogStateMonitor(self)
        self.coordinator = LogCoordinator(self)
        self.transmission_manager = LogTransmissionManager(self)
        
        # 运行状态
        self._is_running = False
        self._shutdown_event = asyncio.Event()
        
        # 信号处理
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        try:
            loop = asyncio.get_event_loop()
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._signal_handler)
        except NotImplementedError:
            # Windows不支持add_signal_handler
            pass
    
    def _signal_handler(self):
        """信号处理器"""
        self.logger.info("接收到关闭信号，开始优雅关闭...")
        asyncio.create_task(self.shutdown())
    
    async def start(self):
        """启动日志状态聚合器"""
        if self._is_running:
            self.logger.warning("日志状态聚合器已经在运行")
            return
        
        self.logger.info("启动L9日志状态聚合器...")
        
        try:
            # 启动各个组件
            await self.alert_system.start_alert_system()
            await self.health_checker.start_health_checks()
            await self.lifecycle_manager.start_lifecycle_management()
            await self.state_monitor.start_monitoring()
            await self.coordinator.start_coordination()
            
            # 启动传输管理器
            asyncio.create_task(self.transmission_manager.start_transmission())
            
            self._is_running = True
            self.logger.info("L9日志状态聚合器启动完成")
            
        except Exception as e:
            self.logger.error(f"启动失败: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """关闭日志状态聚合器"""
        if not self._is_running:
            return
        
        self.logger.info("关闭L9日志状态聚合器...")
        
        try:
            # 停止各个组件
            await self.state_monitor.stop_monitoring()
            await self.lifecycle_manager.stop_lifecycle_management()
            await self.health_checker.stop_health_checks()
            await self.coordinator.stop_coordination()
            await self.alert_system.stop_alert_system()
            await self.transmission_manager.stop_transmission()
            
            self._is_running = False
            self._shutdown_event.set()
            
            self.logger.info("L9日志状态聚合器已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭过程中出现异常: {e}")
    
    async def wait_for_shutdown(self):
        """等待关闭完成"""
        await self._shutdown_event.wait()
    
    # =============================================================================
    # 公共API方法
    # =============================================================================
    
    async def log(self, level: Union[LogLevel, str], message: str, 
                  source: str = "default", metadata: Dict[str, Any] = None) -> str:
        """
        记录日志
        
        Args:
            level: 日志级别
            message: 日志消息
            source: 日志源
            metadata: 元数据
            
        Returns:
            日志ID
        """
        if isinstance(level, str):
            level = LogLevel(level.upper())
        
        # 生成日志ID
        log_id = hashlib.md5(
            f"{time.time()}{source}{message}".encode()
        ).hexdigest()
        
        # 创建日志条目
        entry = LogEntry(
            id=log_id,
            timestamp=time.time(),
            level=level,
            message=message,
            source=source,
            metadata=metadata or {}
        )
        
        # 创建日志
        success = await self.lifecycle_manager.create_log(entry)
        
        if success:
            return log_id
        else:
            raise LogProcessingError(f"日志创建失败: {log_id}")
    
    async def get_log(self, log_id: str) -> Optional[LogEntry]:
        """
        获取日志
        
        Args:
            log_id: 日志ID
            
        Returns:
            日志条目，如果不存在则返回None
        """
        return await self.storage_manager.get_log(log_id)
    
    async def get_logs(self, filters: Dict[str, Any] = None, 
                      limit: int = 100) -> List[LogEntry]:
        """
        获取日志列表
        
        Args:
            filters: 过滤条件
            limit: 限制数量
            
        Returns:
            日志条目列表
        """
        if not filters:
            return await self.storage_manager.get_recent_logs(limit)
        
        # 获取所有日志并应用过滤
        all_logs = await self.storage_manager.get_recent_logs(1000)
        filtered_logs = await self.coordinator.coordinate_log_filtering(all_logs, filters)
        
        return filtered_logs[:limit]
    
    async def delete_logs(self, log_ids: List[str]) -> bool:
        """
        删除日志
        
        Args:
            log_ids: 日志ID列表
            
        Returns:
            是否全部删除成功
        """
        return await self.lifecycle_manager.delete_logs(log_ids)
    
    async def archive_logs(self, log_ids: List[str]) -> bool:
        """
        归档日志
        
        Args:
            log_ids: 日志ID列表
            
        Returns:
            是否全部归档成功
        """
        return await self.lifecycle_manager.archive_logs(log_ids)
    
    def get_statistics(self) -> LogStatistics:
        """
        获取统计信息
        
        Returns:
            统计信息
        """
        return self.statistics_manager.get_statistics()
    
    async def perform_health_check(self, check_name: str = None) -> Union[HealthCheckResult, Dict[str, HealthCheckResult]]:
        """
        执行健康检查
        
        Args:
            check_name: 检查名称，如果为None则执行所有检查
            
        Returns:
            健康检查结果
        """
        if check_name:
            return await self.health_checker.perform_health_check(check_name)
        else:
            return await self.health_checker.perform_all_health_checks()
    
    async def trigger_alert(self, level: AlertLevel, title: str, message: str, 
                          component: str, metadata: Dict[str, Any] = None) -> str:
        """
        触发告警
        
        Args:
            level: 告警级别
            title: 告警标题
            message: 告警消息
            component: 组件名称
            metadata: 元数据
            
        Returns:
            告警ID
        """
        return await self.alert_system.trigger_alert(level, title, message, component, metadata)
    
    async def aggregate_logs(self, logs: List[LogEntry], 
                           aggregation_type: AggregationType) -> Dict[str, Any]:
        """
        聚合日志
        
        Args:
            logs: 日志列表
            aggregation_type: 聚合类型
            
        Returns:
            聚合结果
        """
        return await self.coordinator.coordinate_log_aggregation(logs, aggregation_type)
    
    async def add_to_transmission_queue(self, entry: LogEntry):
        """添加到传输队列"""
        await self.transmission_queue.put(entry)
    
    # =============================================================================
    # 便捷方法
    # =============================================================================
    
    async def debug(self, message: str, source: str = "default", metadata: Dict[str, Any] = None):
        """记录调试日志"""
        return await self.log(LogLevel.DEBUG, message, source, metadata)
    
    async def info(self, message: str, source: str = "default", metadata: Dict[str, Any] = None):
        """记录信息日志"""
        return await self.log(LogLevel.INFO, message, source, metadata)
    
    async def warning(self, message: str, source: str = "default", metadata: Dict[str, Any] = None):
        """记录警告日志"""
        return await self.log(LogLevel.WARNING, message, source, metadata)
    
    async def error(self, message: str, source: str = "default", metadata: Dict[str, Any] = None):
        """记录错误日志"""
        return await self.log(LogLevel.ERROR, message, source, metadata)
    
    async def critical(self, message: str, source: str = "default", metadata: Dict[str, Any] = None):
        """记录严重错误日志"""
        return await self.log(LogLevel.CRITICAL, message, source, metadata)


# =============================================================================
# 使用示例和测试代码
# =============================================================================

async def example_usage():
    """使用示例"""
    
    # 配置
    config = {
        'storage_path': './logs',
        'archive_path': './archive',
        'max_queue_size': 5000,
        'transmission_targets': [
            {
                'type': 'file',
                'path': './remote_logs.log'
            },
            {
                'type': 'http',
                'url': 'http://localhost:8080/logs',
                'headers': {'Content-Type': 'application/json'}
            }
        ]
    }
    
    # 创建聚合器
    aggregator = LoggingStateAggregator(config)
    
    try:
        # 启动聚合器
        await aggregator.start()
        
        # 记录一些日志
        await aggregator.info("系统启动", "system")
        await aggregator.warning("内存使用率较高", "system", {"usage": 85})
        await aggregator.error("数据库连接失败", "database", {"host": "localhost", "port": 5432})
        await aggregator.critical("磁盘空间不足", "storage", {"free_space": "0.5GB"})
        
        # 获取统计信息
        stats = aggregator.get_statistics()
        print(f"统计信息: {stats.to_dict()}")
        
        # 执行健康检查
        health_results = await aggregator.perform_health_check()
        print(f"健康检查结果: {[r.to_dict() for r in health_results.values()]}")
        
        # 获取日志
        logs = await aggregator.get_logs(limit=10)
        print(f"最近日志数: {len(logs)}")
        
        # 聚合日志
        if logs:
            aggregated = await aggregator.aggregate_logs(logs, AggregationType.LEVEL_BASED)
            print(f"按级别聚合结果: {aggregated}")
        
        # 等待一段时间
        await asyncio.sleep(10)
        
    except Exception as e:
        print(f"示例执行异常: {e}")
        traceback.print_exc()
    finally:
        # 关闭聚合器
        await aggregator.shutdown()


async def stress_test():
    """压力测试"""
    
    config = {
        'storage_path': './stress_test_logs',
        'max_queue_size': 1000,
    }
    
    aggregator = LoggingStateAggregator(config)
    
    try:
        await aggregator.start()
        
        # 并发记录大量日志
        tasks = []
        for i in range(1000):
            task = asyncio.create_task(
                aggregator.info(f"压力测试日志 {i}", "stress_test", {"index": i})
            )
            tasks.append(task)
        
        # 等待所有任务完成
        await asyncio.gather(*tasks)
        
        # 获取统计信息
        stats = aggregator.get_statistics()
        print(f"压力测试完成，总日志数: {stats.total_logs}")
        
        await asyncio.sleep(5)
        
    except Exception as e:
        print(f"压力测试异常: {e}")
        traceback.print_exc()
    finally:
        await aggregator.shutdown()


async def main():
    """主函数"""
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logging_aggregator.log'),
            logging.StreamHandler()
        ]
    )
    
    print("L9日志状态聚合器示例")
    print("=" * 50)
    
    # 运行示例
    print("1. 运行基本示例...")
    await example_usage()
    
    print("\n2. 运行压力测试...")
    await stress_test()
    
    print("\n示例完成!")


if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main())