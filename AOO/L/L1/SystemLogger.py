"""
L1层系统日志记录器模块

该模块提供完整的系统级日志管理功能，包括：
1. 系统级日志管理（系统启动、关闭、配置加载）
2. 多级别日志记录（DEBUG、INFO、WARNING、ERROR、CRITICAL）
3. 多输出目标（文件、控制台、网络、数据库）
4. 日志格式化和模板系统
5. 日志轮转和归档管理
6. 异步日志记录和批处理
7. 日志过滤和搜索功能
8. 完整的错误处理和日志记录
9. 详细的文档字符串和使用示例

"""

import os
import sys
import json
import time
import asyncio
import logging
import threading
import queue
import gzip
import shutil
import sqlite3
import socket
import ssl
import smtplib
import traceback
import datetime
import weakref
import hashlib
import pickle
import io
import re
import glob
import fnmatch
import tempfile
import subprocess
import signal
import atexit
import warnings
from collections import deque, defaultdict, OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Union, Any, Callable, Tuple, Set, Iterator, AsyncIterator
from enum import Enum, auto
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path
import logging.handlers


# =============================================================================
# 枚举和常量定义
# =============================================================================

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    @classmethod
    def from_string(cls, level: str) -> 'LogLevel':
        """从字符串创建日志级别"""
        level_map = {
            'DEBUG': cls.DEBUG,
            'INFO': cls.INFO,
            'WARNING': cls.WARNING,
            'ERROR': cls.ERROR,
            'CRITICAL': cls.CRITICAL
        }
        return level_map.get(level.upper(), cls.INFO)
    
    @classmethod
    def get_priority(cls, level: 'LogLevel') -> int:
        """获取日志级别优先级"""
        priority_map = {
            cls.DEBUG: 10,
            cls.INFO: 20,
            cls.WARNING: 30,
            cls.ERROR: 40,
            cls.CRITICAL: 50
        }
        return priority_map.get(level, 20)


class LogFormat(Enum):
    """日志格式枚举"""
    SIMPLE = "simple"
    DETAILED = "detailed"
    JSON = "json"
    CUSTOM = "custom"


class RotationStrategy(Enum):
    """日志轮转策略枚举"""
    SIZE = "size"
    TIME = "time"
    COUNT = "count"


class CompressionType(Enum):
    """压缩类型枚举"""
    NONE = "none"
    GZIP = "gzip"
    BZIP2 = "bzip2"
    LZMA = "lzma"


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class LogRecord:
    """日志记录数据结构"""
    timestamp: datetime.datetime
    level: LogLevel
    logger_name: str
    message: str
    module: str = ""
    function: str = ""
    line_number: int = 0
    thread_id: int = 0
    process_id: int = 0
    exception_info: Optional[Tuple] = None
    extra_data: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None
    correlation_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    def to_json(self) -> str:
        """转换为JSON格式"""
        return json.dumps(self.to_dict(), default=str, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogRecord':
        """从字典创建日志记录"""
        # 处理日期时间字段
        if isinstance(data.get('timestamp'), str):
            data['timestamp'] = datetime.datetime.fromisoformat(data['timestamp'])
        
        return cls(**data)
    
    def format_message(self, formatter: 'LogFormatter') -> str:
        """格式化日志消息"""
        return formatter.format(self)


@dataclass
class SystemLoggerConfig:
    """系统日志记录器配置"""
    # 基础配置
    name: str = "SystemLogger"
    level: LogLevel = LogLevel.INFO
    enabled: bool = True
    
    # 格式配置
    format_type: LogFormat = LogFormat.DETAILED
    custom_format: Optional[str] = None
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # 输出目标配置
    file_path: Optional[str] = None
    console_output: bool = True
    network_output: bool = False
    database_output: bool = False
    
    # 文件输出配置
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    rotation_strategy: RotationStrategy = RotationStrategy.SIZE
    compression: CompressionType = CompressionType.GZIP
    
    # 网络输出配置
    network_host: str = "localhost"
    network_port: int = 514
    network_protocol: str = "udp"
    network_ssl: bool = False
    network_timeout: float = 5.0
    
    # 数据库输出配置
    database_url: Optional[str] = None
    database_table: str = "system_logs"
    
    # 异步配置
    async_enabled: bool = True
    batch_size: int = 100
    flush_interval: float = 1.0
    max_queue_size: int = 10000
    
    # 过滤配置
    filters: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    include_patterns: List[str] = field(default_factory=list)
    
    # 搜索配置
    search_index_enabled: bool = True
    search_index_path: Optional[str] = None
    
    # 性能配置
    thread_pool_size: int = 4
    buffer_size: int = 8192
    flush_on_write: bool = False
    
    # 监控配置
    performance_monitoring: bool = True
    health_check_enabled: bool = True
    metrics_collection: bool = True


# =============================================================================
# 格式化器
# =============================================================================

class LogFormatter(ABC):
    """日志格式化器抽象基类"""
    
    @abstractmethod
    def format(self, record: LogRecord) -> str:
        """格式化日志记录"""
        pass
    
    @abstractmethod
    def format_exception(self, exc_info) -> str:
        """格式化异常信息"""
        pass


class SimpleFormatter(LogFormatter):
    """简单日志格式化器"""
    
    def __init__(self, date_format: str = "%Y-%m-%d %H:%M:%S"):
        self.date_format = date_format
    
    def format(self, record: LogRecord) -> str:
        """格式化日志记录"""
        timestamp = record.timestamp.strftime(self.date_format)
        return f"[{timestamp}] {record.level.value} {record.message}"
    
    def format_exception(self, exc_info) -> str:
        """格式化异常信息"""
        return ''.join(traceback.format_exception(*exc_info))


class DetailedFormatter(LogFormatter):
    """详细日志格式化器"""
    
    def __init__(self, date_format: str = "%Y-%m-%d %H:%M:%S"):
        self.date_format = date_format
    
    def format(self, record: LogRecord) -> str:
        """格式化日志记录"""
        timestamp = record.timestamp.strftime(self.date_format)
        
        # 基础信息
        parts = [
            f"[{timestamp}]",
            f"{record.level.value}",
            f"[{record.logger_name}]"
        ]
        
        # 位置信息
        if record.module or record.function:
            location = f"{record.module}.{record.function}"
            if record.line_number:
                location += f":{record.line_number}"
            parts.append(f"[{location}]")
        
        # 线程和进程信息
        if record.thread_id or record.process_id:
            thread_info = f"Thread:{record.thread_id}"
            if record.process_id:
                thread_info += f" Process:{record.process_id}"
            parts.append(f"[{thread_info}]")
        
        # 消息内容
        message = record.message
        if record.extra_data:
            extra_str = json.dumps(record.extra_data, ensure_ascii=False, default=str)
            message += f" {extra_str}"
        
        parts.append(message)
        
        # 异常信息
        if record.exception_info:
            exception_str = self.format_exception(record.exception_info)
            parts.append(f"\n{exception_str}")
        
        return ' '.join(parts)
    
    def format_exception(self, exc_info) -> str:
        """格式化异常信息"""
        return ''.join(traceback.format_exception(*exc_info))


class JsonFormatter(LogFormatter):
    """JSON日志格式化器"""
    
    def __init__(self, ensure_ascii: bool = False):
        self.ensure_ascii = ensure_ascii
    
    def format(self, record: LogRecord) -> str:
        """格式化日志记录"""
        return record.to_json()
    
    def format_exception(self, exc_info) -> str:
        """格式化异常信息"""
        return json.dumps({
            'exception_type': exc_info[0].__name__,
            'exception_message': str(exc_info[1]),
            'traceback': traceback.format_exception(*exc_info)
        }, ensure_ascii=self.ensure_ascii)


class CustomFormatter(LogFormatter):
    """自定义日志格式化器"""
    
    def __init__(self, format_template: str, date_format: str = "%Y-%m-%d %H:%M:%S"):
        self.format_template = format_template
        self.date_format = date_format
        
        # 预编译格式模板
        self._compile_template()
    
    def _compile_template(self):
        """编译格式模板"""
        self._compiled_template = self.format_template
        
        # 替换预定义的占位符
        replacements = {
            '%(asctime)s': '{timestamp}',
            '%(levelname)s': '{level}',
            '%(name)s': '{logger_name}',
            '%(message)s': '{message}',
            '%(module)s': '{module}',
            '%(funcName)s': '{function}',
            '%(lineno)d': '{line_number}',
            '%(thread)d': '{thread_id}',
            '%(process)d': '{process_id}',
            '%(correlation_id)s': '{correlation_id}',
            '%(user_id)s': '{user_id}',
            '%(session_id)s': '{session_id}',
            '%(request_id)s': '{request_id}'
        }
        
        for old, new in replacements.items():
            self._compiled_template = self._compiled_template.replace(old, new)
    
    def format(self, record: LogRecord) -> str:
        """格式化日志记录"""
        try:
            # 准备格式化数据
            format_data = {
                'timestamp': record.timestamp.strftime(self.date_format),
                'level': record.level.value,
                'logger_name': record.logger_name,
                'message': record.message,
                'module': record.module,
                'function': record.function,
                'line_number': record.line_number,
                'thread_id': record.thread_id,
                'process_id': record.process_id,
                'correlation_id': record.correlation_id or '',
                'user_id': record.user_id or '',
                'session_id': record.session_id or '',
                'request_id': record.request_id or ''
            }
            
            # 格式化字符串
            formatted = self._compiled_template.format(**format_data)
            
            # 添加额外数据
            if record.extra_data:
                extra_str = json.dumps(record.extra_data, ensure_ascii=False, default=str)
                formatted += f" {extra_str}"
            
            return formatted
            
        except Exception as e:
            # 格式化失败时的备用方案
            return f"[{record.timestamp}] {record.level.value} {record.message}"
    
    def format_exception(self, exc_info) -> str:
        """格式化异常信息"""
        return ''.join(traceback.format_exception(*exc_info))


# =============================================================================
# 日志过滤器
# =============================================================================

class LogFilter(ABC):
    """日志过滤器抽象基类"""
    
    @abstractmethod
    def should_log(self, record: LogRecord) -> bool:
        """判断是否应该记录此日志"""
        pass


class LevelFilter(LogFilter):
    """级别过滤器"""
    
    def __init__(self, min_level: LogLevel = LogLevel.DEBUG):
        self.min_level = min_level
    
    def should_log(self, record: LogRecord) -> bool:
        """判断是否应该记录此日志"""
        return LogLevel.get_priority(record.level) >= LogLevel.get_priority(self.min_level)


class PatternFilter(LogFilter):
    """模式过滤器"""
    
    def __init__(self, include_patterns: List[str] = None, exclude_patterns: List[str] = None):
        self.include_patterns = include_patterns or []
        self.exclude_patterns = exclude_patterns or []
    
    def should_log(self, record: LogRecord) -> bool:
        """判断是否应该记录此日志"""
        message = record.message
        
        # 检查排除模式
        for pattern in self.exclude_patterns:
            if fnmatch.fnmatch(message, pattern):
                return False
        
        # 检查包含模式
        if self.include_patterns:
            for pattern in self.include_patterns:
                if fnmatch.fnmatch(message, pattern):
                    return True
            return False
        
        return True


class ModuleFilter(LogFilter):
    """模块过滤器"""
    
    def __init__(self, include_modules: List[str] = None, exclude_modules: List[str] = None):
        self.include_modules = include_modules or []
        self.exclude_modules = exclude_modules or []
    
    def should_log(self, record: LogRecord) -> bool:
        """判断是否应该记录此日志"""
        module = record.module
        
        # 检查排除模块
        for exclude_module in self.exclude_modules:
            if module.startswith(exclude_module):
                return False
        
        # 检查包含模块
        if self.include_modules:
            for include_module in self.include_modules:
                if module.startswith(include_module):
                    return True
            return False
        
        return True


class CompositeFilter(LogFilter):
    """复合过滤器"""
    
    def __init__(self, filters: List[LogFilter]):
        self.filters = filters
    
    def should_log(self, record: LogRecord) -> bool:
        """判断是否应该记录此日志"""
        for filter_obj in self.filters:
            if not filter_obj.should_log(record):
                return False
        return True


# =============================================================================
# 输出目标
# =============================================================================

class OutputTarget(ABC):
    """输出目标抽象基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.enabled = True
        self.error_count = 0
        self.last_error = None
    
    @abstractmethod
    def write(self, record: LogRecord) -> bool:
        """写入日志记录"""
        pass
    
    @abstractmethod
    def flush(self):
        """刷新输出"""
        pass
    
    @abstractmethod
    def close(self):
        """关闭输出"""
        pass
    
    def is_available(self) -> bool:
        """检查输出是否可用"""
        return self.enabled and self.error_count < 10
    
    def record_error(self, error: Exception):
        """记录错误"""
        self.error_count += 1
        self.last_error = error
        
        # 如果错误过多，禁用输出
        if self.error_count >= 10:
            self.enabled = False


class FileTarget(OutputTarget):
    """文件输出目标"""
    
    def __init__(self, file_path: str, **kwargs):
        super().__init__(f"file_{file_path}")
        self.file_path = Path(file_path)
        self.file_handle = None
        self.buffer = io.StringIO()
        
        # 配置参数
        self.max_size = kwargs.get('max_size', 10 * 1024 * 1024)  # 10MB
        self.backup_count = kwargs.get('backup_count', 5)
        self.compression = kwargs.get('compression', CompressionType.GZIP)
        self.rotation_strategy = kwargs.get('rotation_strategy', RotationStrategy.SIZE)
        
        # 创建目录
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 打开文件
        self._open_file()
    
    def _open_file(self):
        """打开文件"""
        try:
            if self.file_handle:
                self.file_handle.close()
            
            self.file_handle = open(self.file_path, 'a', encoding='utf-8')
        except Exception as e:
            self.record_error(e)
    
    def write(self, record: LogRecord) -> bool:
        """写入日志记录"""
        if not self.is_available():
            return False
        
        try:
            # 检查是否需要轮转
            if self._should_rotate():
                self._rotate_file()
            
            # 写入日志
            formatted_message = record.message + '\n'
            self.file_handle.write(formatted_message)
            
            # 刷新缓冲区
            if hasattr(self, 'flush_on_write') and self.flush_on_write:
                self.file_handle.flush()
            
            return True
            
        except Exception as e:
            self.record_error(e)
            return False
    
    def _should_rotate(self) -> bool:
        """检查是否需要轮转"""
        if self.rotation_strategy == RotationStrategy.SIZE:
            return self.file_path.exists() and self.file_path.stat().st_size >= self.max_size
        return False
    
    def _rotate_file(self):
        """轮转文件"""
        try:
            # 关闭当前文件
            if self.file_handle:
                self.file_handle.close()
                self.file_handle = None
            
            # 轮转现有备份文件
            for i in range(self.backup_count - 1, 0, -1):
                old_file = self.file_path.with_suffix(f'.{i}')
                new_file = self.file_path.with_suffix(f'.{i + 1}')
                
                if old_file.exists():
                    if new_file.exists():
                        new_file.unlink()
                    old_file.rename(new_file)
            
            # 压缩和重命名当前文件
            current_file = self.file_path.with_suffix('.1')
            if current_file.exists():
                current_file.unlink()
            
            self.file_path.rename(current_file)
            
            # 压缩备份文件
            if self.compression != CompressionType.NONE:
                self._compress_file(current_file)
            
            # 重新打开文件
            self._open_file()
            
        except Exception as e:
            self.record_error(e)
    
    def _compress_file(self, file_path: Path):
        """压缩文件"""
        try:
            if self.compression == CompressionType.GZIP:
                compressed_path = file_path.with_suffix(file_path.suffix + '.gz')
                with open(file_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                file_path.unlink()
            
        except Exception as e:
            # 压缩失败不影响主流程
            pass
    
    def flush(self):
        """刷新输出"""
        if self.file_handle:
            self.file_handle.flush()
    
    def close(self):
        """关闭输出"""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None


class ConsoleTarget(OutputTarget):
    """控制台输出目标"""
    
    def __init__(self, **kwargs):
        super().__init__("console")
        self.use_colors = kwargs.get('use_colors', True)
        self.stream = kwargs.get('stream', sys.stdout)
        
        # 颜色映射
        self.colors = {
            LogLevel.DEBUG: '\033[36m',    # 青色
            LogLevel.INFO: '\033[32m',     # 绿色
            LogLevel.WARNING: '\033[33m',  # 黄色
            LogLevel.ERROR: '\033[31m',    # 红色
            LogLevel.CRITICAL: '\033[35m', # 紫色
            'RESET': '\033[0m'             # 重置
        }
    
    def write(self, record: LogRecord) -> bool:
        """写入日志记录"""
        if not self.is_available():
            return False
        
        try:
            # 格式化消息
            message = record.message
            
            # 添加颜色
            if self.use_colors and hasattr(self, 'colors'):
                color = self.colors.get(record.level, '')
                reset = self.colors['RESET']
                message = f"{color}{message}{reset}"
            
            # 写入控制台
            print(message, file=self.stream)
            
            return True
            
        except Exception as e:
            self.record_error(e)
            return False
    
    def flush(self):
        """刷新输出"""
        if self.stream:
            self.stream.flush()
    
    def close(self):
        """关闭输出"""
        # 控制台不需要特殊关闭操作
        pass


class NetworkTarget(OutputTarget):
    """网络输出目标"""
    
    def __init__(self, host: str, port: int, **kwargs):
        super().__init__(f"network_{host}_{port}")
        self.host = host
        self.port = port
        self.protocol = kwargs.get('protocol', 'udp')
        self.ssl = kwargs.get('ssl', False)
        self.timeout = kwargs.get('timeout', 5.0)
        self.socket = None
        
        # 连接池
        self.connection_pool = []
        self.max_connections = kwargs.get('max_connections', 5)
        
        # 建立连接
        self._establish_connection()
    
    def _establish_connection(self):
        """建立网络连接"""
        try:
            if self.protocol.lower() == 'udp':
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            elif self.protocol.lower() == 'tcp':
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                if self.ssl:
                    context = ssl.create_default_context()
                    self.socket = context.wrap_socket(self.socket, server_hostname=self.host)
                
                self.socket.connect((self.host, self.port))
                self.socket.settimeout(self.timeout)
            
        except Exception as e:
            self.record_error(e)
    
    def write(self, record: LogRecord) -> bool:
        """写入日志记录"""
        if not self.is_available():
            return False
        
        try:
            # 序列化日志记录
            message = record.to_json()
            
            # 发送网络消息
            if self.protocol.lower() == 'udp':
                self.socket.sendto(message.encode('utf-8'), (self.host, self.port))
            elif self.protocol.lower() == 'tcp':
                self.socket.sendall(message.encode('utf-8'))
            
            return True
            
        except Exception as e:
            self.record_error(e)
            # 尝试重新连接
            self._establish_connection()
            return False
    
    def flush(self):
        """刷新输出"""
        if self.socket:
            try:
                self.socket.flush()
            except:
                pass
    
    def close(self):
        """关闭输出"""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None


class DatabaseTarget(OutputTarget):
    """数据库输出目标"""
    
    def __init__(self, database_url: str, table_name: str = "system_logs", **kwargs):
        super().__init__(f"database_{database_url}")
        self.database_url = database_url
        self.table_name = table_name
        self.connection = None
        self.batch_size = kwargs.get('batch_size', 100)
        self.buffer = []
        
        # 建立数据库连接
        self._connect_database()
        
        # 创建表
        self._create_table()
    
    def _connect_database(self):
        """连接数据库"""
        try:
            if self.database_url.startswith('sqlite:'):
                db_path = self.database_url.replace('sqlite:', '')
                self.connection = sqlite3.connect(db_path)
            else:
                # 其他数据库类型可以在这里扩展
                raise NotImplementedError("Only SQLite is currently supported")
            
            self.connection.row_factory = sqlite3.Row
            
        except Exception as e:
            self.record_error(e)
    
    def _create_table(self):
        """创建日志表"""
        try:
            cursor = self.connection.cursor()
            
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                level TEXT NOT NULL,
                logger_name TEXT NOT NULL,
                message TEXT NOT NULL,
                module TEXT,
                function TEXT,
                line_number INTEGER,
                thread_id INTEGER,
                process_id INTEGER,
                exception_info TEXT,
                extra_data TEXT,
                stack_trace TEXT,
                correlation_id TEXT,
                user_id TEXT,
                session_id TEXT,
                request_id TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            cursor.execute(create_table_sql)
            self.connection.commit()
            
        except Exception as e:
            self.record_error(e)
    
    def write(self, record: LogRecord) -> bool:
        """写入日志记录"""
        if not self.is_available():
            return False
        
        try:
            # 添加到缓冲区
            self.buffer.append(record)
            
            # 如果缓冲区满了，批量写入
            if len(self.buffer) >= self.batch_size:
                self._flush_buffer()
            
            return True
            
        except Exception as e:
            self.record_error(e)
            return False
    
    def _flush_buffer(self):
        """刷新缓冲区"""
        if not self.buffer:
            return
        
        try:
            cursor = self.connection.cursor()
            
            insert_sql = f"""
            INSERT INTO {self.table_name} (
                timestamp, level, logger_name, message, module, function,
                line_number, thread_id, process_id, exception_info,
                extra_data, stack_trace, correlation_id, user_id,
                session_id, request_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            
            for record in self.buffer:
                cursor.execute(insert_sql, (
                    record.timestamp.isoformat(),
                    record.level.value,
                    record.logger_name,
                    record.message,
                    record.module,
                    record.function,
                    record.line_number,
                    record.thread_id,
                    record.process_id,
                    json.dumps(record.exception_info) if record.exception_info else None,
                    json.dumps(record.extra_data, default=str) if record.extra_data else None,
                    record.stack_trace,
                    record.correlation_id,
                    record.user_id,
                    record.session_id,
                    record.request_id
                ))
            
            self.connection.commit()
            self.buffer.clear()
            
        except Exception as e:
            self.record_error(e)
    
    def flush(self):
        """刷新输出"""
        self._flush_buffer()
    
    def close(self):
        """关闭输出"""
        if self.connection:
            self._flush_buffer()
            self.connection.close()
            self.connection = None


# =============================================================================
# 日志轮转管理器
# =============================================================================

class LogRotationManager:
    """日志轮转管理器"""
    
    def __init__(self, log_dir: str, **kwargs):
        self.log_dir = Path(log_dir)
        self.max_file_size = kwargs.get('max_file_size', 10 * 1024 * 1024)  # 10MB
        self.backup_count = kwargs.get('backup_count', 5)
        self.compression = kwargs.get('compression', CompressionType.GZIP)
        self.rotation_strategy = kwargs.get('rotation_strategy', RotationStrategy.SIZE)
        
        # 创建日志目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 轮转任务管理
        self.rotation_tasks = {}
        self.lock = threading.Lock()
    
    def should_rotate(self, log_file: Path) -> bool:
        """检查是否需要轮转"""
        if not log_file.exists():
            return False
        
        if self.rotation_strategy == RotationStrategy.SIZE:
            return log_file.stat().st_size >= self.max_file_size
        
        return False
    
    def rotate_log(self, log_file: Path) -> bool:
        """轮转日志文件"""
        try:
            with self.lock:
                if not self.should_rotate(log_file):
                    return False
                
                # 轮转现有备份文件
                for i in range(self.backup_count - 1, 0, -1):
                    old_file = log_file.with_suffix(f'.{i}')
                    new_file = log_file.with_suffix(f'.{i + 1}')
                    
                    if old_file.exists():
                        if new_file.exists():
                            new_file.unlink()
                        old_file.rename(new_file)
                
                # 移动当前文件
                backup_file = log_file.with_suffix('.1')
                if backup_file.exists():
                    backup_file.unlink()
                
                log_file.rename(backup_file)
                
                # 压缩备份文件
                if self.compression != CompressionType.NONE:
                    self._compress_backup(backup_file)
                
                return True
                
        except Exception as e:
            print(f"日志轮转失败: {e}")
            return False
    
    def _compress_backup(self, backup_file: Path):
        """压缩备份文件"""
        try:
            if self.compression == CompressionType.GZIP:
                compressed_file = backup_file.with_suffix(backup_file.suffix + '.gz')
                with open(backup_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                backup_file.unlink()
                
        except Exception as e:
            print(f"压缩备份文件失败: {e}")
    
    def cleanup_old_logs(self, log_file: Path, days: int = 30):
        """清理旧日志文件"""
        try:
            cutoff_time = time.time() - (days * 24 * 60 * 60)
            
            # 清理压缩的备份文件
            pattern = f"{log_file.stem}.*{log_file.suffix}*"
            for file_path in self.log_dir.glob(pattern):
                if file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    
        except Exception as e:
            print(f"清理旧日志失败: {e}")
    
    def get_log_files(self, pattern: str = "*.log") -> List[Path]:
        """获取日志文件列表"""
        return list(self.log_dir.glob(pattern))
    
    def archive_logs(self, archive_name: str, pattern: str = "*.log") -> bool:
        """归档日志文件"""
        try:
            archive_path = self.log_dir / f"{archive_name}.tar.gz"
            log_files = self.get_log_files(pattern)
            
            if not log_files:
                return False
            
            # 使用tar命令创建压缩归档
            cmd = ['tar', '-czf', str(archive_path)]
            cmd.extend(str(f) for f in log_files)
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # 删除已归档的文件
                for log_file in log_files:
                    log_file.unlink()
                return True
            
            return False
            
        except Exception as e:
            print(f"归档日志失败: {e}")
            return False


# =============================================================================
# 异步日志处理器
# =============================================================================

class AsyncLogProcessor:
    """异步日志处理器"""
    
    def __init__(self, **kwargs):
        self.batch_size = kwargs.get('batch_size', 100)
        self.flush_interval = kwargs.get('flush_interval', 1.0)
        self.max_queue_size = kwargs.get('max_queue_size', 10000)
        
        # 异步相关
        self.loop = None
        self.running = False
        self.queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.batch_buffer = []
        self.last_flush = time.time()
        
        # 线程池
        self.thread_pool = ThreadPoolExecutor(
            max_workers=kwargs.get('thread_pool_size', 4)
        )
        
        # 任务管理
        self.tasks = []
        self.processors = []
    
    def add_processor(self, processor: Callable[[List[LogRecord]], None]):
        """添加处理器"""
        self.processors.append(processor)
    
    async def start(self):
        """启动异步处理器"""
        if self.running:
            return
        
        self.running = True
        self.loop = asyncio.get_event_loop()
        
        # 创建处理任务
        self.tasks = [
            asyncio.create_task(self._process_queue()),
            asyncio.create_task(self._flush_periodically())
        ]
    
    async def stop(self):
        """停止异步处理器"""
        if not self.running:
            return
        
        self.running = False
        
        # 等待所有任务完成
        for task in self.tasks:
            task.cancel()
        
        # 刷新剩余数据
        await self._flush_batch()
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
    
    async def submit(self, record: LogRecord):
        """提交日志记录"""
        if not self.running:
            return
        
        try:
            await asyncio.wait_for(
                self.queue.put(record),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            # 队列满时丢弃记录
            pass
    
    async def _process_queue(self):
        """处理队列"""
        while self.running:
            try:
                # 获取记录
                record = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=0.1
                )
                
                # 添加到批次
                self.batch_buffer.append(record)
                
                # 如果批次满了，处理批次
                if len(self.batch_buffer) >= self.batch_size:
                    await self._flush_batch()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"处理队列错误: {e}")
    
    async def _flush_periodically(self):
        """定期刷新"""
        while self.running:
            try:
                await asyncio.sleep(self.flush_interval)
                
                current_time = time.time()
                if current_time - self.last_flush >= self.flush_interval:
                    await self._flush_batch()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"定期刷新错误: {e}")
    
    async def _flush_batch(self):
        """刷新批次"""
        if not self.batch_buffer:
            return
        
        batch = self.batch_buffer.copy()
        self.batch_buffer.clear()
        self.last_flush = time.time()
        
        # 在线程池中处理批次
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.thread_pool,
            self._process_batch,
            batch
        )
    
    def _process_batch(self, batch: List[LogRecord]):
        """处理批次"""
        for processor in self.processors:
            try:
                processor(batch)
            except Exception as e:
                print(f"处理器错误: {e}")


# =============================================================================
# 日志搜索引擎
# =============================================================================

class LogSearchEngine:
    """日志搜索引擎"""
    
    def __init__(self, index_path: str = None):
        self.index_path = Path(index_path) if index_path else None
        self.index = {}
        self.lock = threading.Lock()
        
        # 搜索统计
        self.search_stats = defaultdict(int)
        
        # 如果有索引路径，加载索引
        if self.index_path and self.index_path.exists():
            self._load_index()
    
    def index_record(self, record: LogRecord):
        """索引日志记录"""
        with self.lock:
            # 创建索引键
            key = f"{record.timestamp.isoformat()}_{hash(record.message)}"
            
            # 索引内容
            self.index[key] = {
                'timestamp': record.timestamp,
                'level': record.level.value,
                'message': record.message,
                'module': record.module,
                'logger': record.logger_name,
                'extra_data': record.extra_data
            }
            
            # 关键词索引
            words = self._extract_keywords(record.message)
            for word in words:
                if word not in self.index:
                    self.index[word] = []
                self.index[word].append(key)
    
    def search(self, query: str, **filters) -> List[Dict[str, Any]]:
        """搜索日志记录"""
        results = []
        
        with self.lock:
            # 解析查询
            search_terms = self._parse_query(query)
            
            # 获取候选结果
            candidate_keys = self._get_candidate_keys(search_terms)
            
            # 应用过滤器
            for key in candidate_keys:
                if self._matches_filters(self.index.get(key), filters):
                    results.append(self.index[key])
            
            # 按时间排序
            results.sort(key=lambda x: x['timestamp'], reverse=True)
            
        # 记录搜索统计
        self.search_stats[query] += 1
        
        return results
    
    def _extract_keywords(self, text: str) -> Set[str]:
        """提取关键词"""
        # 简单的关键词提取（可以扩展为更复杂的NLP）
        words = re.findall(r'\w+', text.lower())
        return set(words)
    
    def _parse_query(self, query: str) -> List[str]:
        """解析查询字符串"""
        # 支持简单的AND、OR查询
        if ' AND ' in query:
            return query.split(' AND ')
        elif ' OR ' in query:
            return query.split(' OR ')
        else:
            return [query]
    
    def _get_candidate_keys(self, search_terms: List[str]) -> Set[str]:
        """获取候选键"""
        candidate_sets = []
        
        for term in search_terms:
            term_lower = term.lower()
            keys = set()
            
            # 直接匹配
            if term_lower in self.index:
                if isinstance(self.index[term_lower], list):
                    keys.update(self.index[term_lower])
                else:
                    keys.add(term_lower)
            
            # 模糊匹配
            for key in self.index:
                if isinstance(self.index[key], dict):
                    message = self.index[key].get('message', '').lower()
                    if term_lower in message:
                        keys.add(key)
            
            candidate_sets.append(keys)
        
        # 交集（AND查询）
        if len(candidate_sets) > 1:
            return set.intersection(*candidate_sets)
        
        return candidate_sets[0] if candidate_sets else set()
    
    def _matches_filters(self, record: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """检查记录是否匹配过滤器"""
        if not record:
            return False
        
        # 级别过滤
        if 'level' in filters:
            if record.get('level') != filters['level']:
                return False
        
        # 模块过滤
        if 'module' in filters:
            if filters['module'] not in record.get('module', ''):
                return False
        
        # 时间范围过滤
        if 'start_time' in filters:
            if record.get('timestamp') < filters['start_time']:
                return False
        
        if 'end_time' in filters:
            if record.get('timestamp') > filters['end_time']:
                return False
        
        return True
    
    def get_search_stats(self) -> Dict[str, int]:
        """获取搜索统计"""
        return dict(self.search_stats)
    
    def clear_index(self):
        """清空索引"""
        with self.lock:
            self.index.clear()
    
    def save_index(self):
        """保存索引"""
        if not self.index_path:
            return
        
        try:
            with self.index_path.open('wb') as f:
                pickle.dump(dict(self.index), f)
        except Exception as e:
            print(f"保存索引失败: {e}")
    
    def _load_index(self):
        """加载索引"""
        try:
            with self.index_path.open('rb') as f:
                self.index = pickle.load(f)
        except Exception as e:
            print(f"加载索引失败: {e}")
            self.index = {}


# =============================================================================
# 系统日志记录器
# =============================================================================

class SystemLogger:
    """系统日志记录器主类"""
    
    def __init__(self, config: SystemLoggerConfig):
        self.config = config
        self.name = config.name
        self.enabled = config.enabled
        
        # 日志记录器状态
        self.start_time = datetime.datetime.now()
        self.is_running = False
        self.record_count = 0
        self.error_count = 0
        
        # 组件初始化
        self.formatter = self._create_formatter()
        self.filters = self._create_filters()
        self.output_targets = self._create_output_targets()
        self.rotation_manager = None
        self.async_processor = None
        self.search_engine = None
        
        # 性能监控
        self.performance_stats = {
            'total_records': 0,
            'records_per_second': 0.0,
            'average_processing_time': 0.0,
            'memory_usage': 0,
            'disk_usage': 0
        }
        
        # 线程安全
        self.lock = threading.RLock()
        
        # 初始化组件
        self._initialize_components()
        
        # 注册关闭钩子
        atexit.register(self.shutdown)
    
    def _create_formatter(self) -> LogFormatter:
        """创建格式化器"""
        if self.config.format_type == LogFormat.SIMPLE:
            return SimpleFormatter(self.config.date_format)
        elif self.config.format_type == LogFormat.DETAILED:
            return DetailedFormatter(self.config.date_format)
        elif self.config.format_type == LogFormat.JSON:
            return JsonFormatter()
        elif self.config.format_type == LogFormat.CUSTOM:
            return CustomFormatter(
                self.config.custom_format or "%(asctime)s %(levelname)s %(message)s",
                self.config.date_format
            )
        else:
            return DetailedFormatter(self.config.date_format)
    
    def _create_filters(self) -> List[LogFilter]:
        """创建过滤器"""
        filters = []
        
        # 级别过滤器
        filters.append(LevelFilter(self.config.level))
        
        # 模式过滤器
        if self.config.include_patterns or self.config.exclude_patterns:
            filters.append(PatternFilter(
                self.config.include_patterns,
                self.config.exclude_patterns
            ))
        
        return filters
    
    def _create_output_targets(self) -> List[OutputTarget]:
        """创建输出目标"""
        targets = []
        
        # 文件输出
        if self.config.file_path:
            try:
                file_target = FileTarget(
                    self.config.file_path,
                    max_size=self.config.max_file_size,
                    backup_count=self.config.backup_count,
                    compression=self.config.compression,
                    rotation_strategy=self.config.rotation_strategy
                )
                targets.append(file_target)
            except Exception as e:
                print(f"创建文件输出目标失败: {e}")
        
        # 控制台输出
        if self.config.console_output:
            try:
                console_target = ConsoleTarget()
                targets.append(console_target)
            except Exception as e:
                print(f"创建控制台输出目标失败: {e}")
        
        # 网络输出
        if self.config.network_output:
            try:
                network_target = NetworkTarget(
                    self.config.network_host,
                    self.config.network_port,
                    protocol=self.config.network_protocol,
                    ssl=self.config.network_ssl,
                    timeout=self.config.network_timeout
                )
                targets.append(network_target)
            except Exception as e:
                print(f"创建网络输出目标失败: {e}")
        
        # 数据库输出
        if self.config.database_output and self.config.database_url:
            try:
                db_target = DatabaseTarget(
                    self.config.database_url,
                    self.config.database_table,
                    batch_size=self.config.batch_size
                )
                targets.append(db_target)
            except Exception as e:
                print(f"创建数据库输出目标失败: {e}")
        
        return targets
    
    def _initialize_components(self):
        """初始化组件"""
        # 初始化轮转管理器
        if self.config.file_path:
            log_dir = Path(self.config.file_path).parent
            self.rotation_manager = LogRotationManager(
                str(log_dir),
                max_file_size=self.config.max_file_size,
                backup_count=self.config.backup_count,
                compression=self.config.compression,
                rotation_strategy=self.config.rotation_strategy
            )
        
        # 初始化异步处理器
        if self.config.async_enabled:
            self.async_processor = AsyncLogProcessor(
                batch_size=self.config.batch_size,
                flush_interval=self.config.flush_interval,
                max_queue_size=self.config.max_queue_size,
                thread_pool_size=self.config.thread_pool_size
            )
            
            # 添加输出处理器
            self.async_processor.add_processor(self._process_batch_output)
        
        # 初始化搜索引擎
        if self.config.search_index_enabled:
            index_path = self.config.search_index_path or f"{self.name}_index.pkl"
            self.search_engine = LogSearchEngine(index_path)
    
    async def start(self):
        """启动日志记录器"""
        if self.is_running:
            return
        
        with self.lock:
            self.is_running = True
            
            # 启动异步处理器
            if self.async_processor:
                await self.async_processor.start()
            
            # 记录启动日志
            await self.log_system_event("SYSTEM_START", f"日志记录器 {self.name} 已启动")
            
            print(f"系统日志记录器 {self.name} 已启动")
    
    async def shutdown(self):
        """关闭日志记录器"""
        if not self.is_running:
            return
        
        with self.lock:
            self.is_running = False
            
            # 记录关闭日志
            try:
                await self.log_system_event("SYSTEM_SHUTDOWN", f"日志记录器 {self.name} 正在关闭")
            except:
                pass
            
            # 停止异步处理器
            if self.async_processor:
                await self.async_processor.stop()
            
            # 刷新所有输出
            self.flush()
            
            # 保存搜索引擎索引
            if self.search_engine:
                self.search_engine.save_index()
            
            # 关闭输出目标
            for target in self.output_targets:
                try:
                    target.close()
                except:
                    pass
            
            print(f"系统日志记录器 {self.name} 已关闭")
    
    async def log(self, level: LogLevel, message: str, **kwargs):
        """记录日志"""
        if not self.enabled or not self.is_running:
            return
        
        # 创建日志记录
        record = self._create_log_record(level, message, **kwargs)
        
        # 应用过滤器
        if not self._should_log(record):
            return
        
        # 记录统计
        self.record_count += 1
        self.performance_stats['total_records'] += 1
        
        try:
            # 异步处理
            if self.async_processor:
                await self.async_processor.submit(record)
            else:
                # 同步处理
                self._process_record(record)
                
        except Exception as e:
            self.error_count += 1
            print(f"记录日志失败: {e}")
    
    def _create_log_record(self, level: LogLevel, message: str, **kwargs) -> LogRecord:
        """创建日志记录"""
        # 获取调用信息
        frame = sys._getframe(2)
        module = frame.f_globals.get('__name__', '')
        function = frame.f_code.co_name
        line_number = frame.f_lineno
        
        # 提取特殊参数
        exc_info = kwargs.pop('exc_info', None)
        correlation_id = kwargs.pop('correlation_id', None)
        user_id = kwargs.pop('user_id', None)
        session_id = kwargs.pop('session_id', None)
        request_id = kwargs.pop('request_id', None)
        
        # 创建记录
        record = LogRecord(
            timestamp=datetime.datetime.now(),
            level=level,
            logger_name=self.name,
            message=message,
            module=module,
            function=function,
            line_number=line_number,
            thread_id=threading.get_ident(),
            process_id=os.getpid(),
            extra_data=kwargs,
            correlation_id=correlation_id,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id
        )
        
        # 添加异常信息
        if exc_info:
            record.exception_info = exc_info
            record.stack_trace = ''.join(traceback.format_exception(*exc_info))
        
        return record
    
    def _should_log(self, record: LogRecord) -> bool:
        """检查是否应该记录"""
        for filter_obj in self.filters:
            if not filter_obj.should_log(record):
                return False
        return True
    
    def _process_record(self, record: LogRecord):
        """处理日志记录"""
        # 索引到搜索引擎
        if self.search_engine:
            self.search_engine.index_record(record)
        
        # 写入所有输出目标
        for target in self.output_targets:
            if target.is_available():
                target.write(record)
    
    def _process_batch_output(self, records: List[LogRecord]):
        """批量处理输出"""
        for record in records:
            self._process_record(record)
    
    # 便捷方法
    async def debug(self, message: str, **kwargs):
        """记录调试信息"""
        await self.log(LogLevel.DEBUG, message, **kwargs)
    
    async def info(self, message: str, **kwargs):
        """记录信息"""
        await self.log(LogLevel.INFO, message, **kwargs)
    
    async def warning(self, message: str, **kwargs):
        """记录警告"""
        await self.log(LogLevel.WARNING, message, **kwargs)
    
    async def error(self, message: str, **kwargs):
        """记录错误"""
        await self.log(LogLevel.ERROR, message, **kwargs)
    
    async def critical(self, message: str, **kwargs):
        """记录严重错误"""
        await self.log(LogLevel.CRITICAL, message, **kwargs)
    
    async def log_system_event(self, event_type: str, message: str, **kwargs):
        """记录系统事件"""
        await self.info(f"[{event_type}] {message}", **kwargs)
    
    async def log_exception(self, message: str, exc_info=None, **kwargs):
        """记录异常"""
        if exc_info is None:
            exc_info = sys.exc_info()
        
        await self.error(message, exc_info=exc_info, **kwargs)
    
    # 搜索功能
    def search_logs(self, query: str, **filters) -> List[Dict[str, Any]]:
        """搜索日志"""
        if not self.search_engine:
            return []
        
        return self.search_engine.search(query, **filters)
    
    def get_recent_logs(self, count: int = 100) -> List[Dict[str, Any]]:
        """获取最近的日志"""
        # 这里可以实现从数据库或文件读取最近日志的功能
        return []
    
    def get_logs_by_level(self, level: LogLevel, count: int = 100) -> List[Dict[str, Any]]:
        """按级别获取日志"""
        if not self.search_engine:
            return []
        
        return self.search_engine.search("", level=level.value)[:count]
    
    # 管理和监控
    def flush(self):
        """刷新所有输出"""
        for target in self.output_targets:
            try:
                target.flush()
            except:
                pass
    
    def get_status(self) -> Dict[str, Any]:
        """获取状态信息"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'is_running': self.is_running,
            'start_time': self.start_time,
            'record_count': self.record_count,
            'error_count': self.error_count,
            'output_targets': [target.name for target in self.output_targets if target.is_available()],
            'performance_stats': self.performance_stats.copy(),
            'config': asdict(self.config)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        # 更新性能统计
        current_time = time.time()
        uptime = current_time - self.start_time.timestamp()
        
        if uptime > 0:
            self.performance_stats['records_per_second'] = self.record_count / uptime
        
        # 内存使用情况
        try:
            import psutil
            process = psutil.Process()
            self.performance_stats['memory_usage'] = process.memory_info().rss
        except ImportError:
            self.performance_stats['memory_usage'] = 0
        
        # 磁盘使用情况
        if self.config.file_path:
            try:
                log_file = Path(self.config.file_path)
                if log_file.exists():
                    self.performance_stats['disk_usage'] = log_file.stat().st_size
            except:
                self.performance_stats['disk_usage'] = 0
        
        return self.performance_stats.copy()
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            'overall_status': 'healthy',
            'checks': {},
            'issues': []
        }
        
        # 检查输出目标
        for target in self.output_targets:
            status = 'healthy' if target.is_available() else 'unhealthy'
            health_status['checks'][target.name] = status
            
            if status == 'unhealthy':
                health_status['issues'].append(f"输出目标 {target.name} 不可用")
                health_status['overall_status'] = 'degraded'
        
        # 检查错误率
        if self.record_count > 0:
            error_rate = self.error_count / self.record_count
            if error_rate > 0.1:  # 10%错误率
                health_status['overall_status'] = 'degraded'
                health_status['issues'].append(f"错误率过高: {error_rate:.2%}")
        
        # 检查队列状态
        if self.async_processor and hasattr(self.async_processor, 'queue'):
            queue_size = self.async_processor.queue.qsize()
            if queue_size > self.config.max_queue_size * 0.9:
                health_status['overall_status'] = 'degraded'
                health_status['issues'].append(f"异步队列接近满载: {queue_size}/{self.config.max_queue_size}")
        
        return health_status
    
    def rotate_logs(self):
        """手动触发日志轮转"""
        if self.rotation_manager and self.config.file_path:
            log_file = Path(self.config.file_path)
            return self.rotation_manager.rotate_log(log_file)
        return False
    
    def cleanup_old_logs(self, days: int = 30):
        """清理旧日志"""
        if self.rotation_manager and self.config.file_path:
            log_file = Path(self.config.file_path)
            self.rotation_manager.cleanup_old_logs(log_file, days)


# =============================================================================
# 工厂函数和便利函数
# =============================================================================

def create_system_logger(config: SystemLoggerConfig) -> SystemLogger:
    """创建系统日志记录器"""
    return SystemLogger(config)


def setup_system_logging(config_dict: Dict[str, Any]) -> SystemLogger:
    """设置系统日志记录"""
    # 创建配置
    config = SystemLoggerConfig(**config_dict)
    
    # 创建日志记录器
    logger = SystemLogger(config)
    
    return logger


# =============================================================================
# 使用示例和测试代码
# =============================================================================

async def example_usage():
    """使用示例"""
    
    # 1. 基本配置示例
    basic_config = {
        'name': 'MyApp',
        'level': LogLevel.INFO,
        'file_path': 'logs/app.log',
        'console_output': True,
        'format_type': LogFormat.DETAILED
    }
    
    logger = setup_system_logging(basic_config)
    await logger.start()
    
    # 2. 记录不同级别的日志
    await logger.debug("这是调试信息", correlation_id="req-123")
    await logger.info("应用启动成功", user_id="user-456")
    await logger.warning("内存使用率较高", extra_data={'memory_usage': '85%'})
    await logger.error("数据库连接失败", exc_info=sys.exc_info())
    await logger.critical("系统资源耗尽", extra_data={'cpu_usage': '99%'})
    
    # 3. 搜索日志
    results = logger.search_logs("数据库")
    print(f"找到 {len(results)} 条相关日志")
    
    # 4. 获取状态
    status = logger.get_status()
    print(f"日志记录器状态: {status['is_running']}")
    
    # 5. 健康检查
    health = logger.health_check()
    print(f"健康状态: {health['overall_status']}")
    
    # 6. 关闭日志记录器
    await logger.shutdown()


def configuration_examples():
    """配置示例"""
    
    # 1. 开发环境配置
    dev_config = {
        'name': 'DevLogger',
        'level': LogLevel.DEBUG,
        'console_output': True,
        'format_type': LogFormat.DETAILED,
        'async_enabled': False
    }
    
    # 2. 生产环境配置
    prod_config = {
        'name': 'ProdLogger',
        'level': LogLevel.INFO,
        'file_path': '/var/log/app/app.log',
        'console_output': False,
        'format_type': LogFormat.JSON,
        'max_file_size': 50 * 1024 * 1024,  # 50MB
        'backup_count': 10,
        'rotation_strategy': RotationStrategy.SIZE,
        'compression': CompressionType.GZIP,
        'async_enabled': True,
        'search_index_enabled': True,
        'performance_monitoring': True
    }
    
    # 3. 网络日志配置
    network_config = {
        'name': 'NetworkLogger',
        'level': LogLevel.WARNING,
        'console_output': False,
        'network_output': True,
        'network_host': 'log-server.company.com',
        'network_port': 514,
        'network_protocol': 'udp',
        'network_ssl': True
    }
    
    # 4. 数据库日志配置
    db_config = {
        'name': 'DatabaseLogger',
        'level': LogLevel.ERROR,
        'console_output': False,
        'database_output': True,
        'database_url': 'sqlite:/var/log/app/logs.db',
        'database_table': 'application_logs',
        'batch_size': 50
    }
    
    return {
        'dev': dev_config,
        'prod': prod_config,
        'network': network_config,
        'database': db_config
    }


# =============================================================================
# 高级功能扩展
# =============================================================================

class LogAggregationManager:
    """日志聚合管理器"""
    
    def __init__(self, logger: SystemLogger):
        self.logger = logger
        self.aggregators = {}
        self.lock = threading.Lock()
    
    def add_aggregator(self, name: str, aggregator: Callable[[List[LogRecord]], Dict[str, Any]]):
        """添加聚合器"""
        with self.lock:
            self.aggregators[name] = aggregator
    
    def aggregate_logs(self, time_window: datetime.timedelta) -> Dict[str, Any]:
        """聚合日志"""
        end_time = datetime.datetime.now()
        start_time = end_time - time_window
        
        # 获取时间窗口内的日志
        logs = self.logger.search_logs("", start_time=start_time, end_time=end_time)
        
        results = {}
        with self.lock:
            for name, aggregator in self.aggregators.items():
                try:
                    results[name] = aggregator(logs)
                except Exception as e:
                    results[name] = {"error": str(e)}
        
        return results


class LogAnalysisEngine:
    """日志分析引擎"""
    
    def __init__(self, logger: SystemLogger):
        self.logger = logger
        self.patterns = {}
        self.anomalies = []
    
    def add_pattern(self, name: str, pattern: str, action: Callable):
        """添加模式匹配"""
        self.patterns[name] = {
            'pattern': re.compile(pattern),
            'action': action
        }
    
    def analyze_logs(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析日志"""
        analysis = {
            'total_logs': len(logs),
            'level_distribution': defaultdict(int),
            'module_distribution': defaultdict(int),
            'time_distribution': defaultdict(int),
            'patterns_found': defaultdict(list),
            'anomalies': []
        }
        
        for log in logs:
            # 级别分布
            analysis['level_distribution'][log.get('level', 'UNKNOWN')] += 1
            
            # 模块分布
            analysis['module_distribution'][log.get('module', 'UNKNOWN')] += 1
            
            # 时间分布
            if 'timestamp' in log:
                hour = log['timestamp'].hour
                analysis['time_distribution'][hour] += 1
            
            # 模式匹配
            message = log.get('message', '')
            for name, pattern_info in self.patterns.items():
                if pattern_info['pattern'].search(message):
                    analysis['patterns_found'][name].append(log)
        
        return dict(analysis)
    
    def detect_anomalies(self, logs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """检测异常"""
        anomalies = []
        
        # 检测错误激增
        error_count = sum(1 for log in logs if log.get('level') == 'ERROR')
        if error_count > len(logs) * 0.1:  # 错误率超过10%
            anomalies.append({
                'type': 'error_spike',
                'description': f'错误率异常: {error_count}/{len(logs)}',
                'severity': 'high'
            })
        
        # 检测重复错误
        error_messages = defaultdict(int)
        for log in logs:
            if log.get('level') == 'ERROR':
                error_messages[log.get('message', '')] += 1
        
        for message, count in error_messages.items():
            if count > 5:  # 同一错误重复5次以上
                anomalies.append({
                    'type': 'repeated_error',
                    'description': f'重复错误: {message} (出现{count}次)',
                    'severity': 'medium'
                })
        
        return anomalies


class LogMonitoringSystem:
    """日志监控系统"""
    
    def __init__(self, logger: SystemLogger, config: Dict[str, Any]):
        self.logger = logger
        self.config = config
        self.monitors = {}
        self.alerts = []
        self.running = False
        self.monitor_thread = None
        
        # 初始化监控器
        self._initialize_monitors()
    
    def _initialize_monitors(self):
        """初始化监控器"""
        # 错误率监控
        self.monitors['error_rate'] = {
            'threshold': self.config.get('error_rate_threshold', 0.1),
            'window': self.config.get('error_rate_window', 300),  # 5分钟
            'last_check': time.time()
        }
        
        # 日志量监控
        self.monitors['log_volume'] = {
            'threshold': self.config.get('log_volume_threshold', 1000),
            'window': self.config.get('log_volume_window', 60),  # 1分钟
            'last_check': time.time()
        }
        
        # 响应时间监控
        self.monitors['response_time'] = {
            'threshold': self.config.get('response_time_threshold', 5.0),
            'window': self.config.get('response_time_window', 300),
            'last_check': time.time()
        }
    
    def start_monitoring(self):
        """启动监控"""
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """监控循环"""
        while self.running:
            try:
                self._check_all_monitors()
                time.sleep(30)  # 每30秒检查一次
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
    
    def _check_all_monitors(self):
        """检查所有监控器"""
        current_time = time.time()
        
        for monitor_name, monitor_config in self.monitors.items():
            try:
                if current_time - monitor_config['last_check'] >= monitor_config['window']:
                    self._check_monitor(monitor_name, monitor_config)
                    monitor_config['last_check'] = current_time
            except Exception as e:
                self.logger.error(f"检查监控器 {monitor_name} 失败: {e}")
    
    def _check_monitor(self, monitor_name: str, config: Dict[str, Any]):
        """检查单个监控器"""
        if monitor_name == 'error_rate':
            self._check_error_rate(config)
        elif monitor_name == 'log_volume':
            self._check_log_volume(config)
        elif monitor_name == 'response_time':
            self._check_response_time(config)
    
    def _check_error_rate(self, config: Dict[str, Any]):
        """检查错误率"""
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(seconds=config['window'])
        
        logs = self.logger.search_logs("", start_time=start_time, end_time=end_time)
        
        if logs:
            error_count = sum(1 for log in logs if log.get('level') == 'ERROR')
            error_rate = error_count / len(logs)
            
            if error_rate > config['threshold']:
                alert = {
                    'type': 'error_rate_alert',
                    'message': f'错误率过高: {error_rate:.2%}',
                    'severity': 'high',
                    'timestamp': datetime.datetime.now(),
                    'value': error_rate,
                    'threshold': config['threshold']
                }
                self.alerts.append(alert)
                self.logger.warning(f"错误率告警: {alert['message']}")
    
    def _check_log_volume(self, config: Dict[str, Any]):
        """检查日志量"""
        end_time = datetime.datetime.now()
        start_time = end_time - datetime.timedelta(seconds=config['window'])
        
        logs = self.logger.search_logs("", start_time=start_time, end_time=end_time)
        log_count = len(logs)
        
        if log_count > config['threshold']:
            alert = {
                'type': 'log_volume_alert',
                'message': f'日志量异常: {log_count}条/{config["window"]}秒',
                'severity': 'medium',
                'timestamp': datetime.datetime.now(),
                'value': log_count,
                'threshold': config['threshold']
            }
            self.alerts.append(alert)
            self.logger.warning(f"日志量告警: {alert['message']}")
    
    def _check_response_time(self, config: Dict[str, Any]):
        """检查响应时间"""
        # 这里可以实现更复杂的响应时间监控逻辑
        pass
    
    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取告警"""
        return self.alerts[-limit:]
    
    def clear_alerts(self):
        """清除告警"""
        self.alerts.clear()


class LogBackupManager:
    """日志备份管理器"""
    
    def __init__(self, logger: SystemLogger, backup_config: Dict[str, Any]):
        self.logger = logger
        self.config = backup_config
        self.backup_tasks = {}
        self.running = False
        self.backup_thread = None
    
    def add_backup_task(self, name: str, task_config: Dict[str, Any]):
        """添加备份任务"""
        self.backup_tasks[name] = {
            'source': task_config['source'],
            'destination': task_config['destination'],
            'schedule': task_config.get('schedule', 'daily'),
            'retention': task_config.get('retention', 30),
            'last_run': None,
            'enabled': task_config.get('enabled', True)
        }
    
    def start_backup_scheduler(self):
        """启动备份调度器"""
        if self.running:
            return
        
        self.running = True
        self.backup_thread = threading.Thread(target=self._backup_loop)
        self.backup_thread.start()
    
    def stop_backup_scheduler(self):
        """停止备份调度器"""
        self.running = False
        if self.backup_thread:
            self.backup_thread.join()
    
    def _backup_loop(self):
        """备份循环"""
        while self.running:
            try:
                self._check_and_run_backups()
                time.sleep(3600)  # 每小时检查一次
            except Exception as e:
                self.logger.error(f"备份循环错误: {e}")
    
    def _check_and_run_backups(self):
        """检查并运行备份"""
        current_time = datetime.datetime.now()
        
        for name, task_config in self.backup_tasks.items():
            if not task_config['enabled']:
                continue
            
            # 检查是否需要运行备份
            if self._should_run_backup(task_config, current_time):
                self._run_backup(name, task_config)
    
    def _should_run_backup(self, task_config: Dict[str, Any], current_time: datetime.datetime) -> bool:
        """检查是否应该运行备份"""
        last_run = task_config['last_run']
        if not last_run:
            return True
        
        schedule = task_config['schedule']
        if schedule == 'daily':
            return (current_time - last_run).days >= 1
        elif schedule == 'weekly':
            return (current_time - last_run).days >= 7
        elif schedule == 'hourly':
            return (current_time - last_run).total_seconds() >= 3600
        
        return False
    
    def _run_backup(self, name: str, task_config: Dict[str, Any]):
        """运行备份"""
        try:
            source = Path(task_config['source'])
            destination = Path(task_config['destination'])
            
            # 创建目标目录
            destination.parent.mkdir(parents=True, exist_ok=True)
            
            # 执行备份
            if source.is_file():
                shutil.copy2(source, destination)
            elif source.is_dir():
                shutil.copytree(source, destination, dirs_exist_ok=True)
            
            # 更新最后运行时间
            task_config['last_run'] = datetime.datetime.now()
            
            # 清理旧备份
            self._cleanup_old_backups(task_config)
            
            self.logger.info(f"备份任务 {name} 完成")
            
        except Exception as e:
            self.logger.error(f"备份任务 {name} 失败: {e}")
    
    def _cleanup_old_backups(self, task_config: Dict[str, Any]):
        """清理旧备份"""
        retention_days = task_config['retention']
        cutoff_time = time.time() - (retention_days * 24 * 60 * 60)
        
        destination = Path(task_config['destination'])
        if destination.exists():
            for file_path in destination.parent.glob(f"{destination.stem}*"):
                if file_path.stat().st_mtime < cutoff_time:
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)


class LogReportGenerator:
    """日志报告生成器"""
    
    def __init__(self, logger: SystemLogger):
        self.logger = logger
        self.report_templates = {}
    
    def add_template(self, name: str, template: Dict[str, Any]):
        """添加报告模板"""
        self.report_templates[name] = template
    
    def generate_report(self, template_name: str, time_range: Tuple[datetime.datetime, datetime.datetime]) -> str:
        """生成报告"""
        if template_name not in self.report_templates:
            raise ValueError(f"模板 {template_name} 不存在")
        
        template = self.report_templates[template_name]
        start_time, end_time = time_range
        
        # 获取数据
        logs = self.logger.search_logs("", start_time=start_time, end_time=end_time)
        
        # 生成报告
        report = self._format_report(template, logs, start_time, end_time)
        
        return report
    
    def _format_report(self, template: Dict[str, Any], logs: List[Dict[str, Any]], 
                      start_time: datetime.datetime, end_time: datetime.datetime) -> str:
        """格式化报告"""
        report_lines = []
        
        # 报告头部
        report_lines.append(f"# {template.get('title', '日志报告')}")
        report_lines.append(f"生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"时间范围: {start_time} 到 {end_time}")
        report_lines.append("")
        
        # 统计信息
        if template.get('include_statistics', True):
            stats = self._calculate_statistics(logs)
            report_lines.append("## 统计信息")
            report_lines.append(f"- 总日志数: {stats['total_logs']}")
            report_lines.append(f"- 错误数: {stats['error_count']}")
            report_lines.append(f"- 警告数: {stats['warning_count']}")
            report_lines.append(f"- 错误率: {stats['error_rate']:.2%}")
            report_lines.append("")
        
        # 级别分布
        if template.get('include_level_distribution', True):
            level_dist = self._calculate_level_distribution(logs)
            report_lines.append("## 级别分布")
            for level, count in level_dist.items():
                percentage = (count / len(logs)) * 100 if logs else 0
                report_lines.append(f"- {level}: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        # 模块分布
        if template.get('include_module_distribution', True):
            module_dist = self._calculate_module_distribution(logs)
            report_lines.append("## 模块分布")
            for module, count in sorted(module_dist.items(), key=lambda x: x[1], reverse=True)[:10]:
                percentage = (count / len(logs)) * 100 if logs else 0
                report_lines.append(f"- {module}: {count} ({percentage:.1f}%)")
            report_lines.append("")
        
        # 错误日志
        if template.get('include_error_logs', True):
            error_logs = [log for log in logs if log.get('level') == 'ERROR']
            if error_logs:
                report_lines.append("## 错误日志")
                for log in error_logs[-10:]:  # 只显示最近10条
                    timestamp = log.get('timestamp', '').strftime('%Y-%m-%d %H:%M:%S') if hasattr(log.get('timestamp'), 'strftime') else str(log.get('timestamp', ''))
                    report_lines.append(f"- [{timestamp}] {log.get('message', '')}")
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def _calculate_statistics(self, logs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算统计信息"""
        total_logs = len(logs)
        error_count = sum(1 for log in logs if log.get('level') == 'ERROR')
        warning_count = sum(1 for log in logs if log.get('level') == 'WARNING')
        error_rate = error_count / total_logs if total_logs > 0 else 0
        
        return {
            'total_logs': total_logs,
            'error_count': error_count,
            'warning_count': warning_count,
            'error_rate': error_rate
        }
    
    def _calculate_level_distribution(self, logs: List[Dict[str, Any]]) -> Dict[str, int]:
        """计算级别分布"""
        distribution = defaultdict(int)
        for log in logs:
            distribution[log.get('level', 'UNKNOWN')] += 1
        return dict(distribution)
    
    def _calculate_module_distribution(self, logs: List[Dict[str, Any]]) -> Dict[str, int]:
        """计算模块分布"""
        distribution = defaultdict(int)
        for log in logs:
            distribution[log.get('module', 'UNKNOWN')] += 1
        return dict(distribution)


class LogPerformanceProfiler:
    """日志性能分析器"""
    
    def __init__(self, logger: SystemLogger):
        self.logger = logger
        self.profiles = {}
        self.current_profile = None
    
    def start_profiling(self, profile_name: str):
        """开始性能分析"""
        self.current_profile = {
            'name': profile_name,
            'start_time': time.time(),
            'operations': [],
            'memory_usage': [],
            'cpu_usage': []
        }
    
    def end_profiling(self):
        """结束性能分析"""
        if not self.current_profile:
            return None
        
        profile = self.current_profile
        profile['end_time'] = time.time()
        profile['duration'] = profile['end_time'] - profile['start_time']
        
        # 计算统计信息
        profile['stats'] = self._calculate_profile_stats(profile)
        
        # 保存分析结果
        self.profiles[profile['name']] = profile
        
        self.current_profile = None
        return profile
    
    def log_operation(self, operation_name: str, duration: float, **metadata):
        """记录操作"""
        if not self.current_profile:
            return
        
        operation = {
            'name': operation_name,
            'duration': duration,
            'timestamp': time.time(),
            'metadata': metadata
        }
        
        self.current_profile['operations'].append(operation)
    
    def _calculate_profile_stats(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """计算分析统计"""
        operations = profile['operations']
        
        if not operations:
            return {}
        
        durations = [op['duration'] for op in operations]
        
        return {
            'total_operations': len(operations),
            'total_duration': sum(durations),
            'average_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'operations_per_second': len(operations) / profile['duration']
        }
    
    def get_profile_report(self, profile_name: str) -> str:
        """获取分析报告"""
        if profile_name not in self.profiles:
            return f"分析 {profile_name} 不存在"
        
        profile = self.profiles[profile_name]
        stats = profile['stats']
        
        report_lines = [
            f"性能分析报告: {profile_name}",
            f"分析时间: {datetime.datetime.fromtimestamp(profile['start_time'])}",
            f"持续时间: {profile['duration']:.2f}秒",
            "",
            "统计信息:",
            f"- 总操作数: {stats.get('total_operations', 0)}",
            f"- 总耗时: {stats.get('total_duration', 0):.2f}秒",
            f"- 平均耗时: {stats.get('average_duration', 0):.2f}秒",
            f"- 最小耗时: {stats.get('min_duration', 0):.2f}秒",
            f"- 最大耗时: {stats.get('max_duration', 0):.2f}秒",
            f"- 操作频率: {stats.get('operations_per_second', 0):.2f}次/秒",
            ""
        ]
        
        # 操作详情
        report_lines.append("操作详情:")
        for op in profile['operations']:
            report_lines.append(f"- {op['name']}: {op['duration']:.2f}秒")
        
        return "\n".join(report_lines)


# =============================================================================
# 高级系统集成功能
# =============================================================================

class SystemIntegrationManager:
    """系统集成管理器"""
    
    def __init__(self, logger: SystemLogger):
        self.logger = logger
        self.integrations = {}
        self.webhook_handlers = {}
        self.notification_rules = {}
    
    def register_integration(self, name: str, integration: Any):
        """注册集成"""
        self.integrations[name] = integration
    
    def add_webhook_handler(self, event_type: str, handler: Callable):
        """添加Webhook处理器"""
        if event_type not in self.webhook_handlers:
            self.webhook_handlers[event_type] = []
        self.webhook_handlers[event_type].append(handler)
    
    def add_notification_rule(self, rule_name: str, rule_config: Dict[str, Any]):
        """添加通知规则"""
        self.notification_rules[rule_name] = rule_config
    
    def process_log_event(self, event_type: str, log_record: LogRecord):
        """处理日志事件"""
        # 调用Webhook处理器
        if event_type in self.webhook_handlers:
            for handler in self.webhook_handlers[event_type]:
                try:
                    handler(log_record)
                except Exception as e:
                    self.logger.error(f"Webhook处理器错误: {e}")
        
        # 检查通知规则
        self._check_notification_rules(event_type, log_record)
    
    def _check_notification_rules(self, event_type: str, log_record: LogRecord):
        """检查通知规则"""
        for rule_name, rule_config in self.notification_rules.items():
            if self._should_trigger_rule(rule_config, event_type, log_record):
                self._trigger_notification(rule_name, rule_config, log_record)
    
    def _should_trigger_rule(self, rule_config: Dict[str, Any], event_type: str, log_record: LogRecord) -> bool:
        """检查是否应该触发规则"""
        # 检查事件类型
        if 'event_types' in rule_config and event_type not in rule_config['event_types']:
            return False
        
        # 检查级别
        if 'min_level' in rule_config:
            min_level = LogLevel.from_string(rule_config['min_level'])
            if LogLevel.get_priority(log_record.level) < LogLevel.get_priority(min_level):
                return False
        
        # 检查消息模式
        if 'message_pattern' in rule_config:
            pattern = re.compile(rule_config['message_pattern'])
            if not pattern.search(log_record.message):
                return False
        
        return True
    
    def _trigger_notification(self, rule_name: str, rule_config: Dict[str, Any], log_record: LogRecord):
        """触发通知"""
        notification_type = rule_config.get('notification_type', 'log')
        
        if notification_type == 'email':
            self._send_email_notification(rule_config, log_record)
        elif notification_type == 'webhook':
            self._send_webhook_notification(rule_config, log_record)
        elif notification_type == 'slack':
            self._send_slack_notification(rule_config, log_record)
        else:
            # 默认记录到日志
            self.logger.warning(f"通知规则 {rule_name} 触发: {log_record.message}")
    
    def _send_email_notification(self, rule_config: Dict[str, Any], log_record: LogRecord):
        """发送邮件通知"""
        # 实现邮件发送逻辑
        pass
    
    def _send_webhook_notification(self, rule_config: Dict[str, Any], log_record: LogRecord):
        """发送Webhook通知"""
        # 实现Webhook发送逻辑
        pass
    
    def _send_slack_notification(self, rule_config: Dict[str, Any], log_record: LogRecord):
        """发送Slack通知"""
        # 实现Slack通知逻辑
        pass


# =============================================================================
# 扩展的配置管理
# =============================================================================

class ConfigurationManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = None):
        self.config_file = config_file
        self.configs = {}
        self.watchers = []
        
        if config_file:
            self.load_config()
    
    def load_config(self):
        """加载配置"""
        try:
            if self.config_file and Path(self.config_file).exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    self.configs = json.load(f)
        except Exception as e:
            print(f"加载配置文件失败: {e}")
    
    def save_config(self):
        """保存配置"""
        try:
            if self.config_file:
                with open(self.config_file, 'w', encoding='utf-8') as f:
                    json.dump(self.configs, f, indent=2, ensure_ascii=False, default=str)
        except Exception as e:
            print(f"保存配置文件失败: {e}")
    
    def get_config(self, key: str, default=None):
        """获取配置"""
        keys = key.split('.')
        value = self.configs
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_config(self, key: str, value: Any):
        """设置配置"""
        keys = key.split('.')
        config = self.configs
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def add_watcher(self, key: str, callback: Callable):
        """添加配置监听器"""
        self.watchers.append({'key': key, 'callback': callback})
    
    def notify_watchers(self, key: str):
        """通知监听器"""
        for watcher in self.watchers:
            if key.startswith(watcher['key']):
                try:
                    watcher['callback'](key, self.get_config(key))
                except Exception as e:
                    print(f"配置监听器错误: {e}")


# =============================================================================
# 完整的示例和测试代码
# =============================================================================

async def comprehensive_example():
    """完整的示例"""
    
    # 1. 创建高级配置
    config = {
        'name': 'ComprehensiveLogger',
        'level': LogLevel.DEBUG,
        'file_path': 'logs/comprehensive.log',
        'console_output': True,
        'format_type': LogFormat.DETAILED,
        'max_file_size': 20 * 1024 * 1024,  # 20MB
        'backup_count': 10,
        'rotation_strategy': RotationStrategy.SIZE,
        'compression': CompressionType.GZIP,
        'async_enabled': True,
        'batch_size': 50,
        'flush_interval': 0.5,
        'search_index_enabled': True,
        'performance_monitoring': True,
        'health_check_enabled': True
    }
    
    # 2. 创建日志记录器
    logger = setup_system_logging(config)
    
    # 3. 启动日志记录器
    await logger.start()
    
    # 4. 创建日志聚合管理器
    agg_manager = LogAggregationManager(logger)
    
    # 添加聚合器
    def error_summary(records):
        errors = [r for r in records if r.get('level') == 'ERROR']
        return {'error_count': len(errors), 'unique_errors': len(set(r.get('message') for r in errors))}
    
    agg_manager.add_aggregator('error_summary', error_summary)
    
    # 5. 创建日志分析引擎
    analyzer = LogAnalysisEngine(logger)
    
    # 添加模式
    analyzer.add_pattern('database_error', r'database|sql|connection', lambda m: print(f"数据库错误模式匹配: {m}"))
    analyzer.add_pattern('performance_issue', r'slow|timeout|performance', lambda m: print(f"性能问题模式匹配: {m}"))
    
    # 6. 创建监控系统
    monitor_config = {
        'error_rate_threshold': 0.05,
        'error_rate_window': 300,
        'log_volume_threshold': 500,
        'log_volume_window': 60
    }
    
    monitor = LogMonitoringSystem(logger, monitor_config)
    monitor.start_monitoring()
    
    # 7. 创建备份管理器
    backup_config = {
        'retention_days': 30
    }
    
    backup_manager = LogBackupManager(logger, backup_config)
    backup_manager.add_backup_task('daily_backup', {
        'source': 'logs/comprehensive.log',
        'destination': f'backups/comprehensive_{datetime.datetime.now().strftime("%Y%m%d")}.log',
        'schedule': 'daily',
        'retention': 30,
        'enabled': True
    })
    
    # 8. 创建报告生成器
    report_gen = LogReportGenerator(logger)
    report_gen.add_template('daily_report', {
        'title': '每日日志报告',
        'include_statistics': True,
        'include_level_distribution': True,
        'include_module_distribution': True,
        'include_error_logs': True
    })
    
    # 9. 创建性能分析器
    profiler = LogPerformanceProfiler(logger)
    
    # 10. 创建系统集成管理器
    integration_mgr = SystemIntegrationManager(logger)
    
    # 添加通知规则
    integration_mgr.add_notification_rule('critical_errors', {
        'event_types': ['ERROR', 'CRITICAL'],
        'min_level': 'ERROR',
        'notification_type': 'log',
        'message_pattern': r'critical|emergency'
    })
    
    # 11. 生成测试日志
    profiler.start_profiling('test_logging')
    
    for i in range(100):
        start_time = time.time()
        
        # 记录不同类型的日志
        await logger.info(f"处理请求 {i}", 
                         correlation_id=f"req-{i}",
                         user_id=f"user-{i % 10}",
                         extra_data={'request_size': i * 100})
        
        if i % 10 == 0:
            await logger.warning(f"性能警告 {i}", 
                               extra_data={'response_time': i * 0.1})
        
        if i % 20 == 0:
            await logger.error(f"模拟错误 {i}", 
                             exc_info=sys.exc_info(),
                             extra_data={'error_code': i})
        
        duration = time.time() - start_time
        profiler.log_operation(f'log_request_{i}', duration)
    
    profile = profiler.end_profiling()
    
    # 12. 执行分析
    profiler_report = profiler.get_profile_report('test_logging')
    print("\n=== 性能分析报告 ===")
    print(profiler_report)
    
    # 13. 生成报告
    end_time = datetime.datetime.now()
    start_time = end_time - datetime.timedelta(hours=1)
    
    report = report_gen.generate_report('daily_report', (start_time, end_time))
    print("\n=== 日志报告 ===")
    print(report)
    
    # 14. 搜索日志
    search_results = logger.search_logs("错误", level="ERROR")
    print(f"\n=== 搜索结果 ===")
    print(f"找到 {len(search_results)} 条错误日志")
    
    # 15. 健康检查
    health = logger.health_check()
    print(f"\n=== 健康检查 ===")
    print(f"整体状态: {health['overall_status']}")
    for check, status in health['checks'].items():
        print(f"- {check}: {status}")
    
    # 16. 获取告警
    alerts = monitor.get_alerts()
    if alerts:
        print(f"\n=== 告警信息 ===")
        for alert in alerts[-5:]:  # 显示最近5条告警
            print(f"- {alert['type']}: {alert['message']}")
    
    # 17. 性能统计
    perf_stats = logger.get_performance_stats()
    print(f"\n=== 性能统计 ===")
    for key, value in perf_stats.items():
        print(f"- {key}: {value}")
    
    # 18. 清理和关闭
    monitor.stop_monitoring()
    backup_manager.stop_backup_scheduler()
    await logger.shutdown()
    
    print("\n=== 示例完成 ===")


def configuration_examples_extended():
    """扩展的配置示例"""
    
    examples = {
        'development': {
            'name': 'DevLogger',
            'level': LogLevel.DEBUG,
            'console_output': True,
            'file_path': 'logs/dev.log',
            'format_type': LogFormat.DETAILED,
            'async_enabled': False,
            'search_index_enabled': False,
            'performance_monitoring': False
        },
        
        'production': {
            'name': 'ProdLogger',
            'level': LogLevel.INFO,
            'console_output': False,
            'file_path': '/var/log/app/production.log',
            'format_type': LogFormat.JSON,
            'max_file_size': 100 * 1024 * 1024,  # 100MB
            'backup_count': 20,
            'rotation_strategy': RotationStrategy.SIZE,
            'compression': CompressionType.GZIP,
            'async_enabled': True,
            'batch_size': 200,
            'flush_interval': 2.0,
            'search_index_enabled': True,
            'search_index_path': '/var/log/app/search_index.pkl',
            'performance_monitoring': True,
            'health_check_enabled': True
        },
        
        'high_volume': {
            'name': 'HighVolumeLogger',
            'level': LogLevel.WARNING,
            'console_output': False,
            'file_path': '/var/log/app/high_volume.log',
            'format_type': LogFormat.SIMPLE,
            'max_file_size': 500 * 1024 * 1024,  # 500MB
            'backup_count': 50,
            'rotation_strategy': RotationStrategy.SIZE,
            'compression': CompressionType.GZIP,
            'async_enabled': True,
            'batch_size': 1000,
            'flush_interval': 5.0,
            'max_queue_size': 50000,
            'thread_pool_size': 8,
            'network_output': True,
            'network_host': 'log-aggregator.company.com',
            'network_port': 514,
            'network_protocol': 'udp'
        },
        
        'security': {
            'name': 'SecurityLogger',
            'level': LogLevel.INFO,
            'console_output': False,
            'file_path': '/var/log/security/security.log',
            'format_type': LogFormat.JSON,
            'max_file_size': 50 * 1024 * 1024,  # 50MB
            'backup_count': 100,
            'rotation_strategy': RotationStrategy.SIZE,
            'compression': CompressionType.GZIP,
            'async_enabled': True,
            'database_output': True,
            'database_url': 'sqlite:/var/log/security/security.db',
            'database_table': 'security_logs',
            'search_index_enabled': True,
            'filters': ['security', 'auth', 'access'],
            'exclude_patterns': ['debug', 'test'],
            'performance_monitoring': True
        },
        
        'performance': {
            'name': 'PerformanceLogger',
            'level': LogLevel.DEBUG,
            'console_output': False,
            'file_path': '/var/log/performance/performance.log',
            'format_type': LogFormat.JSON,
            'max_file_size': 200 * 1024 * 1024,  # 200MB
            'backup_count': 30,
            'rotation_strategy': RotationStrategy.SIZE,
            'compression': CompressionType.GZIP,
            'async_enabled': True,
            'batch_size': 500,
            'flush_interval': 1.0,
            'search_index_enabled': True,
            'include_patterns': ['performance', 'metric', 'timing'],
            'performance_monitoring': True,
            'metrics_collection': True
        }
    }
    
    return examples


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    """主程序入口"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='系统日志记录器')
    parser.add_argument('--mode', choices=['basic', 'comprehensive', 'config'], 
                       default='basic', help='运行模式')
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    args = parser.parse_args()
    
    try:
        if args.mode == 'basic':
            print("运行基本示例...")
            asyncio.run(example_usage())
        elif args.mode == 'comprehensive':
            print("运行完整示例...")
            asyncio.run(comprehensive_example())
        elif args.mode == 'config':
            print("显示配置示例...")
            examples = configuration_examples_extended()
            for name, config in examples.items():
                print(f"\n=== {name.upper()} 配置 ===")
                print(json.dumps(config, indent=2, ensure_ascii=False))
        
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行错误: {e}")
        traceback.print_exc()