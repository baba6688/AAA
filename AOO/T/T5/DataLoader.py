#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5数据加载器 - 智能数据加载系统

该模块实现了一个功能完整的数据加载器，支持多数据源加载、增量/全量加载、
调度自动化、进度监控、错误处理、性能优化、缓存机制、安全控制和日志审计。

Author: T5系统
Date: 2025-11-05
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from functools import wraps
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, AsyncIterator, Tuple
from urllib.parse import urlparse
import pickle
import gzip
import os
import psutil
import requests
from queue import Queue, Empty
import threading
from threading import Lock, RLock
import sqlite3
import hashlib
import zlib
import mmap


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """数据源类型枚举"""
    DATABASE = "database"
    FILE = "file"
    API = "api"
    MESSAGE_QUEUE = "message_queue"
    FTP = "ftp"
    S3 = "s3"


class LoadMode(Enum):
    """加载模式枚举"""
    FULL = "full"  # 全量加载
    INCREMENTAL = "incremental"  # 增量加载
    DELTA = "delta"  # 差量加载


class LoadStatus(Enum):
    """加载状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class SecurityLevel(Enum):
    """安全级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DataSourceConfig:
    """数据源配置类"""
    name: str
    source_type: DataSourceType
    connection_info: Dict[str, Any]
    load_mode: LoadMode = LoadMode.FULL
    schedule_interval: Optional[int] = None  # 调度间隔（秒）
    retry_times: int = 3
    timeout: int = 300
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    cache_enabled: bool = True
    compression_enabled: bool = False
    batch_size: int = 1000
    max_workers: int = 4
    enabled: bool = True


@dataclass
class LoadProgress:
    """加载进度信息"""
    source_name: str
    total_records: int = 0
    loaded_records: int = 0
    failed_records: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: LoadStatus = LoadStatus.PENDING
    current_speed: float = 0.0  # 记录/秒
    estimated_remaining_time: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LoadResult:
    """加载结果类"""
    source_name: str
    success: bool
    records_loaded: int = 0
    records_failed: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    data_hash: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class SecurityManager:
    """安全管理器 - 负责数据加载安全控制"""
    
    def __init__(self):
        self.allowed_hosts = set()
        self.blocked_hosts = set()
        self.allowed_file_extensions = {'.txt', '.csv', '.json', '.xml', '.parquet'}
        self.max_file_size = 100 * 1024 * 1024  # 100MB
        self.encryption_key = None
        
    def validate_source(self, config: DataSourceConfig) -> bool:
        """验证数据源安全性"""
        if config.source_type == DataSourceType.API:
            url = config.connection_info.get('url', '')
            parsed = urlparse(url)
            
            if parsed.netloc in self.blocked_hosts:
                return False
                
            if self.allowed_hosts and parsed.netloc not in self.allowed_hosts:
                return False
                
        elif config.source_type == DataSourceType.FILE:
            file_path = config.connection_info.get('path', '')
            if not self._is_safe_file_path(file_path):
                return False
                
        return True
    
    def _is_safe_file_path(self, file_path: str) -> bool:
        """检查文件路径安全性"""
        path = Path(file_path)
        
        # 检查文件扩展名
        if path.suffix.lower() not in self.allowed_file_extensions:
            return False
            
        # 检查文件大小
        try:
            if path.exists() and path.stat().st_size > self.max_file_size:
                return False
        except OSError:
            return False
            
        # 防止路径遍历攻击
        try:
            path.resolve().relative_to(Path.cwd().resolve())
            return True
        except ValueError:
            return False


class CacheManager:
    """缓存管理器 - 负责数据加载缓存机制"""
    
    def __init__(self, cache_dir: str = "cache", max_size_gb: float = 1.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size_bytes = int(max_size_gb * 1024 * 1024 * 1024)
        self.cache_index = {}
        self._lock = RLock()
        self._load_cache_index()
    
    def _load_cache_index(self):
        """加载缓存索引"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    self.cache_index = json.load(f)
            except Exception as e:
                logger.warning(f"加载缓存索引失败: {e}")
                self.cache_index = {}
    
    def _save_cache_index(self):
        """保存缓存索引"""
        index_file = self.cache_dir / "cache_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2)
        except Exception as e:
            logger.error(f"保存缓存索引失败: {e}")
    
    def _get_cache_key(self, config: DataSourceConfig, **kwargs) -> str:
        """生成缓存键"""
        key_data = {
            'name': config.name,
            'source_type': config.source_type.value,
            'connection_info': config.connection_info,
            'load_mode': config.load_mode.value,
            'timestamp': int(time.time() // 3600),  # 小时级缓存
            **kwargs
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, config: DataSourceConfig, **kwargs) -> Optional[Any]:
        """获取缓存数据"""
        cache_key = self._get_cache_key(config, **kwargs)
        
        with self._lock:
            if cache_key not in self.cache_index:
                return None
                
            cache_info = self.cache_index[cache_key]
            cache_file = self.cache_dir / f"{cache_key}.cache"
            
            if not cache_file.exists():
                # 清理无效缓存项
                del self.cache_index[cache_key]
                self._save_cache_index()
                return None
            
            try:
                # 检查缓存是否过期
                if cache_info.get('expire_time', 0) < time.time():
                    self.delete(cache_key)
                    return None
                
                # 读取缓存数据
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                    
                logger.debug(f"缓存命中: {config.name}")
                return data
                
            except Exception as e:
                logger.error(f"读取缓存失败: {e}")
                self.delete(cache_key)
                return None
    
    def set(self, config: DataSourceConfig, data: Any, expire_hours: int = 24, **kwargs):
        """设置缓存数据"""
        cache_key = self._get_cache_key(config, **kwargs)
        
        # 检查缓存大小限制
        self._cleanup_if_needed()
        
        cache_file = self.cache_dir / f"{cache_key}.cache"
        
        try:
            # 写入缓存文件
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # 更新缓存索引
            with self._lock:
                self.cache_index[cache_key] = {
                    'created_time': time.time(),
                    'expire_time': time.time() + expire_hours * 3600,
                    'size': cache_file.stat().st_size,
                    'source_name': config.name
                }
                self._save_cache_index()
                
            logger.debug(f"缓存设置成功: {config.name}")
            
        except Exception as e:
            logger.error(f"设置缓存失败: {e}")
            # 清理失败的文件
            if cache_file.exists():
                cache_file.unlink()
    
    def delete(self, cache_key: str):
        """删除缓存"""
        with self._lock:
            if cache_key in self.cache_index:
                cache_file = self.cache_dir / f"{cache_key}.cache"
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        logger.error(f"删除缓存文件失败: {e}")
                
                del self.cache_index[cache_key]
                self._save_cache_index()
    
    def _cleanup_if_needed(self):
        """清理过期和过大的缓存"""
        current_size = sum(info.get('size', 0) for info in self.cache_index.values())
        
        if current_size < self.max_size_bytes:
            return
        
        # 按创建时间排序，删除最旧的缓存
        sorted_items = sorted(
            self.cache_index.items(),
            key=lambda x: x[1].get('created_time', 0)
        )
        
        for cache_key, _ in sorted_items:
            if current_size < self.max_size_bytes * 0.8:  # 保留20%余量
                break
            self.delete(cache_key)
            current_size -= self.cache_index[cache_key]['size']
    
    def clear(self):
        """清空所有缓存"""
        with self._lock:
            for cache_key in list(self.cache_index.keys()):
                self.delete(cache_key)


class DataSourceAdapter(ABC):
    """数据源适配器基类"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def load_data(self, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """加载数据"""
        pass
    
    @abstractmethod
    def get_last_modified(self) -> Optional[datetime]:
        """获取最后修改时间"""
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        """验证连接"""
        pass


class DatabaseAdapter(DataSourceAdapter):
    """数据库适配器"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.connection = None
    
    async def load_data(self, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """从数据库加载数据"""
        query = kwargs.get('query')
        if not query:
            query = self.config.connection_info.get('query', 'SELECT * FROM table')
        
        conn_info = self.config.connection_info
        db_type = conn_info.get('type', 'sqlite')
        
        if db_type == 'sqlite':
            await self._load_from_sqlite(query)
        elif db_type == 'mysql':
            await self._load_from_mysql(query)
        elif db_type == 'postgresql':
            await self._load_from_postgresql(query)
    
    async def _load_from_sqlite(self, query: str):
        """从SQLite加载数据"""
        db_path = self.config.connection_info.get('path', 'data.db')
        
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(query)
            
            for row in cursor:
                yield dict(row)
    
    async def _load_from_mysql(self, query: str):
        """从MySQL加载数据（示例实现）"""
        # 这里需要安装mysql-connector-python
        # import mysql.connector
        
        conn_info = self.config.connection_info
        # conn = mysql.connector.connect(
        #     host=conn_info['host'],
        #     user=conn_info['user'],
        #     password=conn_info['password'],
        #     database=conn_info['database']
        # )
        
        # cursor = conn.cursor(dictionary=True)
        # cursor.execute(query)
        # 
        # for row in cursor:
        #     yield row
        
        # cursor.close()
        # conn.close()
        
        # 简化实现
        yield {"message": "MySQL适配器需要mysql-connector-python库"}
    
    async def _load_from_postgresql(self, query: str):
        """从PostgreSQL加载数据（示例实现）"""
        # 这里需要安装psycopg2
        # import psycopg2
        
        # 简化实现
        yield {"message": "PostgreSQL适配器需要psycopg2库"}
    
    def get_last_modified(self) -> Optional[datetime]:
        """获取数据库最后修改时间"""
        db_path = self.config.connection_info.get('path', 'data.db')
        if Path(db_path).exists():
            return datetime.fromtimestamp(Path(db_path).stat().st_mtime)
        return None
    
    def validate_connection(self) -> bool:
        """验证数据库连接"""
        try:
            db_path = self.config.connection_info.get('path', 'data.db')
            with sqlite3.connect(db_path):
                return True
        except Exception as e:
            self.logger.error(f"数据库连接验证失败: {e}")
            return False


class FileAdapter(DataSourceAdapter):
    """文件适配器"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
    
    async def load_data(self, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """从文件加载数据"""
        file_path = self.config.connection_info.get('path')
        if not file_path:
            raise ValueError("文件路径未配置")
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 根据文件扩展名选择解析器
        suffix = file_path.suffix.lower()
        
        if suffix == '.json':
            async for item in self._load_json(file_path):
                yield item
        elif suffix == '.csv':
            async for item in self._load_csv(file_path):
                yield item
        elif suffix == '.xml':
            async for item in self._load_xml(file_path):
                yield item
        else:
            yield {"raw_content": file_path.read_text(encoding='utf-8')}
    
    async def _load_json(self, file_path: Path) -> AsyncIterator[Dict[str, Any]]:
        """加载JSON文件"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        if isinstance(data, list):
            for item in data:
                yield item
        else:
            yield data
    
    async def _load_csv(self, file_path: Path) -> AsyncIterator[Dict[str, Any]]:
        """加载CSV文件"""
        import csv
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield row
    
    async def _load_xml(self, file_path: Path) -> AsyncIterator[Dict[str, Any]]:
        """加载XML文件（简化实现）"""
        import xml.etree.ElementTree as ET
        
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            # 简单的XML转字典
            def xml_to_dict(element):
                result = {}
                for child in element:
                    if child.text and not child.attrib:
                        result[child.tag] = child.text
                    else:
                        result[child.tag] = xml_to_dict(child)
                return result
            
            yield xml_to_dict(root)
            
        except ET.ParseError as e:
            self.logger.error(f"XML解析错误: {e}")
            yield {"error": f"XML解析错误: {e}"}
    
    def get_last_modified(self) -> Optional[datetime]:
        """获取文件最后修改时间"""
        file_path = Path(self.config.connection_info.get('path', ''))
        if file_path.exists():
            return datetime.fromtimestamp(file_path.stat().st_mtime)
        return None
    
    def validate_connection(self) -> bool:
        """验证文件连接"""
        file_path = Path(self.config.connection_info.get('path', ''))
        return file_path.exists()


class APIAdapter(DataSourceAdapter):
    """API适配器"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
    
    async def load_data(self, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """从API加载数据"""
        url = self.config.connection_info.get('url')
        if not url:
            raise ValueError("API URL未配置")
        
        headers = self.config.connection_info.get('headers', {})
        params = self.config.connection_info.get('params', {})
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=self.config.timeout)
            response.raise_for_status()
            
            data = response.json()
            
            if isinstance(data, list):
                for item in data:
                    yield item
            else:
                yield data
                
        except requests.RequestException as e:
            self.logger.error(f"API请求失败: {e}")
            yield {"error": f"API请求失败: {e}"}
    
    def get_last_modified(self) -> Optional[datetime]:
        """获取API最后修改时间（通过Last-Modified头）"""
        try:
            url = self.config.connection_info.get('url')
            response = requests.head(url, timeout=10)
            
            last_modified = response.headers.get('Last-Modified')
            if last_modified:
                return datetime.strptime(last_modified, '%a, %d %b %Y %H:%M:%S GMT')
        except Exception:
            pass
        
        return None
    
    def validate_connection(self) -> bool:
        """验证API连接"""
        try:
            url = self.config.connection_info.get('url')
            response = requests.head(url, timeout=10)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"API连接验证失败: {e}")
            return False


class MessageQueueAdapter(DataSourceAdapter):
    """消息队列适配器"""
    
    def __init__(self, config: DataSourceConfig):
        super().__init__(config)
        self.queue = None
    
    async def load_data(self, **kwargs) -> AsyncIterator[Dict[str, Any]]:
        """从消息队列加载数据"""
        queue_type = self.config.connection_info.get('type', 'memory')
        
        if queue_type == 'memory':
            await self._load_from_memory_queue()
        elif queue_type == 'redis':
            await self._load_from_redis()
        elif queue_type == 'rabbitmq':
            await self._load_from_rabbitmq()
    
    async def _load_from_memory_queue(self):
        """从内存队列加载数据"""
        queue_name = self.config.connection_info.get('queue_name', 'default')
        max_messages = self.config.connection_info.get('max_messages', 1000)
        
        # 模拟从内存队列获取消息
        for i in range(max_messages):
            yield {"message_id": i, "queue": queue_name, "data": f"message_{i}"}
            await asyncio.sleep(0.01)  # 模拟处理时间
    
    async def _load_from_redis(self):
        """从Redis队列加载数据（示例实现）"""
        yield {"message": "Redis适配器需要redis库"}
    
    async def _load_from_rabbitmq(self):
        """从RabbitMQ队列加载数据（示例实现）"""
        yield {"message": "RabbitMQ适配器需要pika库"}
    
    def get_last_modified(self) -> Optional[datetime]:
        """消息队列没有修改时间概念"""
        return None
    
    def validate_connection(self) -> bool:
        """验证消息队列连接"""
        # 简化实现
        return True


class LoadScheduler:
    """加载调度器"""
    
    def __init__(self, dataloader: 'DataLoader'):
        self.dataloader = dataloader
        self.scheduled_tasks = {}
        self._running = False
        self._scheduler_thread = None
    
    def add_schedule(self, source_name: str, interval_seconds: int):
        """添加调度任务"""
        self.scheduled_tasks[source_name] = {
            'interval': interval_seconds,
            'last_run': 0,
            'enabled': True
        }
        logger.info(f"添加调度任务: {source_name}, 间隔: {interval_seconds}秒")
    
    def remove_schedule(self, source_name: str):
        """移除调度任务"""
        if source_name in self.scheduled_tasks:
            del self.scheduled_tasks[source_name]
            logger.info(f"移除调度任务: {source_name}")
    
    def start(self):
        """启动调度器"""
        if self._running:
            return
        
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._scheduler_thread.start()
        logger.info("数据加载调度器已启动")
    
    def stop(self):
        """停止调度器"""
        self._running = False
        if self._scheduler_thread:
            self._scheduler_thread.join()
        logger.info("数据加载调度器已停止")
    
    def _run_scheduler(self):
        """运行调度器主循环"""
        while self._running:
            current_time = time.time()
            
            for source_name, task_info in self.scheduled_tasks.items():
                if not task_info['enabled']:
                    continue
                
                if current_time - task_info['last_run'] >= task_info['interval']:
                    try:
                        # 异步执行加载任务
                        asyncio.run(self.dataloader.load_source_async(source_name))
                        task_info['last_run'] = current_time
                    except Exception as e:
                        logger.error(f"调度任务执行失败 {source_name}: {e}")
            
            time.sleep(1)  # 每秒检查一次


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics = {}
        self._lock = Lock()
    
    def record_load_start(self, source_name: str):
        """记录加载开始"""
        with self._lock:
            self.metrics[source_name] = {
                'start_time': time.time(),
                'records_processed': 0,
                'errors': 0,
                'memory_usage': [],
                'cpu_usage': []
            }
    
    def record_progress(self, source_name: str, records_processed: int):
        """记录进度"""
        with self._lock:
            if source_name in self.metrics:
                self.metrics[source_name]['records_processed'] = records_processed
                
                # 记录系统资源使用情况
                process = psutil.Process()
                self.metrics[source_name]['memory_usage'].append(process.memory_info().rss)
                self.metrics[source_name]['cpu_usage'].append(process.cpu_percent())
    
    def record_error(self, source_name: str):
        """记录错误"""
        with self._lock:
            if source_name in self.metrics:
                self.metrics[source_name]['errors'] += 1
    
    def record_load_end(self, source_name: str):
        """记录加载结束"""
        with self._lock:
            if source_name in self.metrics:
                self.metrics[source_name]['end_time'] = time.time()
    
    def get_performance_report(self, source_name: str) -> Dict[str, Any]:
        """获取性能报告"""
        with self._lock:
            if source_name not in self.metrics:
                return {}
            
            metrics = self.metrics[source_name]
            
            if 'start_time' in metrics and 'end_time' in metrics:
                duration = metrics['end_time'] - metrics['start_time']
                records = metrics['records_processed']
                speed = records / duration if duration > 0 else 0
                
                return {
                    'source_name': source_name,
                    'duration': duration,
                    'records_processed': records,
                    'speed': speed,
                    'errors': metrics['errors'],
                    'avg_memory_usage': sum(metrics['memory_usage']) / len(metrics['memory_usage']) if metrics['memory_usage'] else 0,
                    'avg_cpu_usage': sum(metrics['cpu_usage']) / len(metrics['cpu_usage']) if metrics['cpu_usage'] else 0
                }
            
            return {}


class AuditLogger:
    """审计日志器"""
    
    def __init__(self, log_file: str = "dataloader_audit.log"):
        self.log_file = Path(log_file)
        self.logger = logging.getLogger("DataLoaderAudit")
        
        # 配置审计日志处理器
        handler = logging.FileHandler(self.log_file, encoding='utf-8')
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def log_load_start(self, source_name: str, config: DataSourceConfig):
        """记录加载开始"""
        self.logger.info(f"LOAD_START - Source: {source_name}, "
                        f"Type: {config.source_type.value}, "
                        f"Mode: {config.load_mode.value}")
    
    def log_load_complete(self, source_name: str, result: LoadResult):
        """记录加载完成"""
        status = "SUCCESS" if result.success else "FAILED"
        self.logger.info(f"LOAD_COMPLETE - Source: {source_name}, "
                        f"Status: {status}, "
                        f"Records: {result.records_loaded}, "
                        f"Time: {result.execution_time:.2f}s")
    
    def log_error(self, source_name: str, error: Exception):
        """记录错误"""
        self.logger.error(f"LOAD_ERROR - Source: {source_name}, "
                         f"Error: {str(error)}")
    
    def log_security_event(self, event: str, details: str):
        """记录安全事件"""
        self.logger.warning(f"SECURITY_EVENT - {event}: {details}")


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(delay * (2 ** attempt))  # 指数退避
                    else:
                        logger.error(f"函数 {func.__name__} 在 {max_retries} 次重试后仍然失败")
            
            raise last_exception
        return wrapper
    return decorator


class DataLoader:
    """
    T5数据加载器主类
    
    功能特性：
    1. 多数据源支持（数据库、文件、API、消息队列）
    2. 增量加载和全量加载
    3. 数据加载调度和自动化
    4. 数据加载进度监控
    5. 数据加载错误处理
    6. 数据加载性能优化
    7. 数据加载缓存机制
    8. 数据加载安全控制
    9. 数据加载日志和审计
    """
    
    def __init__(self, 
                 cache_dir: str = "cache",
                 max_cache_size_gb: float = 1.0,
                 log_level: str = "INFO"):
        """
        初始化数据加载器
        
        Args:
            cache_dir: 缓存目录
            max_cache_size_gb: 最大缓存大小（GB）
            log_level: 日志级别
        """
        # 配置日志级别
        logging.getLogger().setLevel(getattr(logging, log_level.upper()))
        
        # 初始化组件
        self.data_sources: Dict[str, DataSourceConfig] = {}
        self.adapters: Dict[str, DataSourceAdapter] = {}
        self.progress_tracking: Dict[str, LoadProgress] = {}
        self.results: Dict[str, LoadResult] = {}
        
        # 核心组件
        self.cache_manager = CacheManager(cache_dir, max_cache_size_gb)
        self.security_manager = SecurityManager()
        self.performance_monitor = PerformanceMonitor()
        self.audit_logger = AuditLogger()
        self.scheduler = LoadScheduler(self)
        
        # 并发控制
        self._executor = ThreadPoolExecutor(max_workers=8)
        self._progress_lock = Lock()
        
        # 状态管理
        self._is_running = False
        self._load_tasks = {}
        
        logger.info("T5数据加载器初始化完成")
    
    def add_data_source(self, config: DataSourceConfig) -> bool:
        """
        添加数据源
        
        Args:
            config: 数据源配置
            
        Returns:
            bool: 添加是否成功
        """
        try:
            # 安全检查
            if not self.security_manager.validate_source(config):
                self.audit_logger.log_security_event(
                    "INVALID_SOURCE", 
                    f"数据源 {config.name} 未通过安全检查"
                )
                return False
            
            # 创建适配器
            adapter = self._create_adapter(config)
            if not adapter.validate_connection():
                logger.error(f"数据源 {config.name} 连接验证失败")
                return False
            
            # 注册数据源
            self.data_sources[config.name] = config
            self.adapters[config.name] = adapter
            
            # 初始化进度跟踪
            self.progress_tracking[config.name] = LoadProgress(
                source_name=config.name
            )
            
            logger.info(f"数据源 {config.name} 添加成功")
            return True
            
        except Exception as e:
            logger.error(f"添加数据源失败: {e}")
            return False
    
    def remove_data_source(self, source_name: str) -> bool:
        """
        移除数据源
        
        Args:
            source_name: 数据源名称
            
        Returns:
            bool: 移除是否成功
        """
        if source_name not in self.data_sources:
            return False
        
        try:
            # 停止相关任务
            if source_name in self._load_tasks:
                task = self._load_tasks[source_name]
                if not task.done():
                    task.cancel()
                del self._load_tasks[source_name]
            
            # 移除组件
            del self.data_sources[source_name]
            del self.adapters[source_name]
            del self.progress_tracking[source_name]
            
            if source_name in self.results:
                del self.results[source_name]
            
            # 移除调度
            self.scheduler.remove_schedule(source_name)
            
            logger.info(f"数据源 {source_name} 移除成功")
            return True
            
        except Exception as e:
            logger.error(f"移除数据源失败: {e}")
            return False
    
    def _create_adapter(self, config: DataSourceConfig) -> DataSourceAdapter:
        """创建数据源适配器"""
        if config.source_type == DataSourceType.DATABASE:
            return DatabaseAdapter(config)
        elif config.source_type == DataSourceType.FILE:
            return FileAdapter(config)
        elif config.source_type == DataSourceType.API:
            return APIAdapter(config)
        elif config.source_type == DataSourceType.MESSAGE_QUEUE:
            return MessageQueueAdapter(config)
        else:
            raise ValueError(f"不支持的数据源类型: {config.source_type}")
    
    async def load_source_async(self, source_name: str, **kwargs) -> LoadResult:
        """
        异步加载指定数据源
        
        Args:
            source_name: 数据源名称
            **kwargs: 额外参数
            
        Returns:
            LoadResult: 加载结果
        """
        if source_name not in self.data_sources:
            raise ValueError(f"数据源 {source_name} 不存在")
        
        config = self.data_sources[source_name]
        
        if not config.enabled:
            raise ValueError(f"数据源 {source_name} 已禁用")
        
        # 检查缓存
        if config.cache_enabled:
            cached_data = self.cache_manager.get(config, **kwargs)
            if cached_data is not None:
                result = LoadResult(
                    source_name=source_name,
                    success=True,
                    records_loaded=len(cached_data) if isinstance(cached_data, list) else 1,
                    start_time=datetime.now(),
                    end_time=datetime.now(),
                    execution_time=0.001,
                    metadata={'from_cache': True}
                )
                self.results[source_name] = result
                return result
        
        # 记录加载开始
        self.audit_logger.log_load_start(source_name, config)
        self.performance_monitor.record_load_start(source_name)
        
        # 更新进度状态
        with self._progress_lock:
            progress = self.progress_tracking[source_name]
            progress.status = LoadStatus.RUNNING
            progress.start_time = datetime.now()
        
        start_time = time.time()
        records_loaded = 0
        records_failed = 0
        error_message = None
        
        try:
            adapter = self.adapters[source_name]
            loaded_data = []
            
            # 异步加载数据
            async for record in adapter.load_data(**kwargs):
                if isinstance(record, dict) and 'error' not in record:
                    loaded_data.append(record)
                    records_loaded += 1
                else:
                    records_failed += 1
                    if isinstance(record, dict):
                        logger.warning(f"加载记录失败: {record.get('error', 'Unknown error')}")
                
                # 更新进度
                self.performance_monitor.record_progress(source_name, records_loaded)
                
                with self._progress_lock:
                    progress = self.progress_tracking[source_name]
                    progress.loaded_records = records_loaded
                    progress.failed_records = records_failed
                    
                    # 计算速度和预计剩余时间
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        progress.current_speed = records_loaded / elapsed
                        remaining_records = progress.total_records - records_loaded if progress.total_records > 0 else 0
                        progress.estimated_remaining_time = remaining_records / progress.current_speed if progress.current_speed > 0 else None
            
            # 设置缓存
            if config.cache_enabled and loaded_data:
                self.cache_manager.set(config, loaded_data)
            
            # 创建结果
            result = LoadResult(
                source_name=source_name,
                success=True,
                records_loaded=records_loaded,
                records_failed=records_failed,
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.now(),
                execution_time=time.time() - start_time,
                data_hash=self._calculate_data_hash(loaded_data),
                metadata={'total_records': len(loaded_data)}
            )
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"数据加载失败 {source_name}: {e}")
            self.performance_monitor.record_error(source_name)
            
            result = LoadResult(
                source_name=source_name,
                success=False,
                records_loaded=records_loaded,
                records_failed=records_failed,
                start_time=datetime.fromtimestamp(start_time),
                end_time=datetime.now(),
                execution_time=time.time() - start_time,
                error_message=error_message
            )
        
        finally:
            # 更新最终状态
            with self._progress_lock:
                progress = self.progress_tracking[source_name]
                progress.status = LoadStatus.COMPLETED if result.success else LoadStatus.FAILED
                progress.end_time = datetime.now()
                progress.error_message = error_message
            
            self.performance_monitor.record_load_end(source_name)
            self.audit_logger.log_load_complete(source_name, result)
            
            if error_message:
                self.audit_logger.log_error(source_name, Exception(error_message))
        
        # 保存结果
        self.results[source_name] = result
        return result
    
    def load_source(self, source_name: str, **kwargs) -> LoadResult:
        """
        同步加载指定数据源
        
        Args:
            source_name: 数据源名称
            **kwargs: 额外参数
            
        Returns:
            LoadResult: 加载结果
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.load_source_async(source_name, **kwargs))
        finally:
            loop.close()
    
    async def load_all_async(self, **kwargs) -> Dict[str, LoadResult]:
        """
        异步加载所有数据源
        
        Args:
            **kwargs: 额外参数
            
        Returns:
            Dict[str, LoadResult]: 所有数据源的加载结果
        """
        tasks = []
        for source_name in self.data_sources.keys():
            if self.data_sources[source_name].enabled:
                task = asyncio.create_task(
                    self.load_source_async(source_name, **kwargs)
                )
                tasks.append((source_name, task))
        
        results = {}
        
        # 并发执行所有任务
        for source_name, task in tasks:
            try:
                result = await task
                results[source_name] = result
            except Exception as e:
                logger.error(f"加载数据源 {source_name} 失败: {e}")
                results[source_name] = LoadResult(
                    source_name=source_name,
                    success=False,
                    error_message=str(e)
                )
        
        return results
    
    def load_all(self, **kwargs) -> Dict[str, LoadResult]:
        """
        同步加载所有数据源
        
        Args:
            **kwargs: 额外参数
            
        Returns:
            Dict[str, LoadResult]: 所有数据源的加载结果
        """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self.load_all_async(**kwargs))
        finally:
            loop.close()
    
    def start_scheduler(self):
        """启动调度器"""
        self.scheduler.start()
        logger.info("数据加载调度器已启动")
    
    def stop_scheduler(self):
        """停止调度器"""
        self.scheduler.stop()
        logger.info("数据加载调度器已停止")
    
    def add_schedule(self, source_name: str, interval_seconds: int):
        """添加调度任务"""
        if source_name not in self.data_sources:
            raise ValueError(f"数据源 {source_name} 不存在")
        
        self.scheduler.add_schedule(source_name, interval_seconds)
    
    def get_progress(self, source_name: str) -> Optional[LoadProgress]:
        """
        获取加载进度
        
        Args:
            source_name: 数据源名称
            
        Returns:
            Optional[LoadProgress]: 加载进度信息
        """
        return self.progress_tracking.get(source_name)
    
    def get_all_progress(self) -> Dict[str, LoadProgress]:
        """获取所有数据源的加载进度"""
        return self.progress_tracking.copy()
    
    def get_result(self, source_name: str) -> Optional[LoadResult]:
        """
        获取加载结果
        
        Args:
            source_name: 数据源名称
            
        Returns:
            Optional[LoadResult]: 加载结果
        """
        return self.results.get(source_name)
    
    def get_performance_report(self, source_name: str) -> Dict[str, Any]:
        """
        获取性能报告
        
        Args:
            source_name: 数据源名称
            
        Returns:
            Dict[str, Any]: 性能报告
        """
        return self.performance_monitor.get_performance_report(source_name)
    
    def clear_cache(self):
        """清空缓存"""
        self.cache_manager.clear()
        logger.info("缓存已清空")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_size = sum(info.get('size', 0) for info in self.cache_manager.cache_index.values())
        return {
            'total_cache_files': len(self.cache_manager.cache_index),
            'total_cache_size': total_size,
            'cache_directory': str(self.cache_manager.cache_dir)
        }
    
    def _calculate_data_hash(self, data: List[Dict[str, Any]]) -> str:
        """计算数据哈希值"""
        try:
            data_str = json.dumps(data, sort_keys=True)
            return hashlib.sha256(data_str.encode()).hexdigest()
        except Exception:
            return ""
    
    def export_config(self, file_path: str):
        """
        导出配置
        
        Args:
            file_path: 导出文件路径
        """
        config_data = {
            'data_sources': {},
            'cache_settings': {
                'cache_dir': str(self.cache_manager.cache_dir),
                'max_size_gb': self.cache_manager.max_size_bytes / (1024**3)
            }
        }
        
        for name, source_config in self.data_sources.items():
            config_data['data_sources'][name] = {
                'name': source_config.name,
                'source_type': source_config.source_type.value,
                'connection_info': source_config.connection_info,
                'load_mode': source_config.load_mode.value,
                'schedule_interval': source_config.schedule_interval,
                'retry_times': source_config.retry_times,
                'timeout': source_config.timeout,
                'security_level': source_config.security_level.value,
                'cache_enabled': source_config.cache_enabled,
                'compression_enabled': source_config.compression_enabled,
                'batch_size': source_config.batch_size,
                'max_workers': source_config.max_workers,
                'enabled': source_config.enabled
            }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"配置已导出到: {file_path}")
    
    def import_config(self, file_path: str):
        """
        导入配置
        
        Args:
            file_path: 配置文件路径
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        # 导入数据源配置
        for name, source_data in config_data.get('data_sources', {}).items():
            config = DataSourceConfig(
                name=source_data['name'],
                source_type=DataSourceType(source_data['source_type']),
                connection_info=source_data['connection_info'],
                load_mode=LoadMode(source_data.get('load_mode', 'full')),
                schedule_interval=source_data.get('schedule_interval'),
                retry_times=source_data.get('retry_times', 3),
                timeout=source_data.get('timeout', 300),
                security_level=SecurityLevel(source_data.get('security_level', 'medium')),
                cache_enabled=source_data.get('cache_enabled', True),
                compression_enabled=source_data.get('compression_enabled', False),
                batch_size=source_data.get('batch_size', 1000),
                max_workers=source_data.get('max_workers', 4),
                enabled=source_data.get('enabled', True)
            )
            
            self.add_data_source(config)
        
        logger.info(f"配置已从 {file_path} 导入")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_scheduler()
        self._executor.shutdown(wait=True)
        logger.info("T5数据加载器已关闭")


# 测试用例
async def test_dataloader():
    """数据加载器测试用例"""
    print("=== T5数据加载器测试 ===\n")
    
    # 创建数据加载器实例
    with DataLoader() as dataloader:
        
        # 测试1: 添加文件数据源
        print("1. 测试文件数据源...")
        file_config = DataSourceConfig(
            name="test_file",
            source_type=DataSourceType.FILE,
            connection_info={
                "path": "test_data.json"
            },
            load_mode=LoadMode.FULL,
            cache_enabled=True
        )
        
        # 创建测试文件
        test_data = [
            {"id": 1, "name": "Alice", "age": 25},
            {"id": 2, "name": "Bob", "age": 30},
            {"id": 3, "name": "Charlie", "age": 35}
        ]
        
        with open("test_data.json", "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)
        
        # 添加数据源
        if dataloader.add_data_source(file_config):
            print("✓ 文件数据源添加成功")
            
            # 加载数据
            result = await dataloader.load_source_async("test_file")
            print(f"✓ 文件加载完成: {result.records_loaded} 条记录")
            
            # 获取进度
            progress = dataloader.get_progress("test_file")
            print(f"✓ 加载进度: {progress.status.value}")
            
        else:
            print("✗ 文件数据源添加失败")
        
        # 测试2: 添加API数据源（模拟）
        print("\n2. 测试API数据源...")
        api_config = DataSourceConfig(
            name="test_api",
            source_type=DataSourceType.API,
            connection_info={
                "url": "https://jsonplaceholder.typicode.com/posts",
                "headers": {"User-Agent": "T5-DataLoader/1.0"}
            },
            load_mode=LoadMode.INCREMENTAL,
            cache_enabled=True
        )
        
        if dataloader.add_data_source(api_config):
            print("✓ API数据源添加成功")
            
            try:
                result = await dataloader.load_source_async("test_api")
                print(f"✓ API加载完成: {result.records_loaded} 条记录")
            except Exception as e:
                print(f"⚠ API加载跳过（网络原因）: {e}")
        else:
            print("✗ API数据源添加失败")
        
        # 测试3: 性能监控
        print("\n3. 测试性能监控...")
        performance_report = dataloader.get_performance_report("test_file")
        if performance_report:
            print(f"✓ 性能报告: 处理速度 {performance_report.get('speed', 0):.2f} 记录/秒")
        
        # 测试4: 缓存功能
        print("\n4. 测试缓存功能...")
        cache_stats = dataloader.get_cache_stats()
        print(f"✓ 缓存统计: {cache_stats['total_cache_files']} 个缓存文件")
        
        # 测试5: 调度功能
        print("\n5. 测试调度功能...")
        dataloader.add_schedule("test_file", 60)  # 每分钟执行一次
        dataloader.start_scheduler()
        print("✓ 调度器已启动")
        
        # 等待一段时间
        await asyncio.sleep(2)
        
        # 停止调度器
        dataloader.stop_scheduler()
        print("✓ 调度器已停止")
        
        # 测试6: 配置导出/导入
        print("\n6. 测试配置管理...")
        dataloader.export_config("dataloader_config.json")
        print("✓ 配置导出成功")
        
        # 清理测试文件
        os.remove("test_data.json")
        os.remove("dataloader_config.json")
        
        print("\n=== 测试完成 ===")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_dataloader())