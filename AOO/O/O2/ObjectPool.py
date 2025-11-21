"""
O2对象池模块

提供对象池和连接池管理功能，用于减少对象创建开销和提高内存效率

主要功能：
1. 通用对象池管理
2. 连接池优化
3. 对象生命周期管理
4. 池大小动态调整
5. 线程安全支持
6. 性能监控和统计

作者: O2优化团队
版本: 2.0.0
日期: 2025-11-06
"""

import asyncio
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import deque, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from queue import Queue, Empty, Full
from threading import RLock, Condition
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union, Callable, 
    Generic, TypeVar, Iterator, Protocol, Awaitable
)
import logging
import json
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import socket
import requests


logger = logging.getLogger(__name__)


T = TypeVar('T')
PoolObject = TypeVar('PoolObject')


@dataclass
class PoolConfig:
    """对象池配置"""
    max_size: int = 100
    min_size: int = 5
    max_idle_time: float = 300.0  # 最大空闲时间（秒）
    validation_enabled: bool = True
    validation_interval: float = 60.0  # 验证间隔（秒）
    grow_strategy: str = "dynamic"  # "dynamic", "fixed", "adaptive"
    shrink_strategy: str = "idle"   # "idle", "pressure", "never"
    thread_safe: bool = True
    preallocate: bool = False
    preallocate_count: int = 10


@dataclass
class PoolStats:
    """对象池统计信息"""
    total_created: int = 0
    total_destroyed: int = 0
    total_borrowed: int = 0
    total_returned: int = 0
    current_size: int = 0
    available_count: int = 0
    in_use_count: int = 0
    peak_size: int = 0
    avg_borrow_time: float = 0.0
    avg_return_time: float = 0.0
    validation_failures: int = 0
    last_validation: float = 0.0
    creation_time: float = field(default_factory=time.time)


class ObjectFactory(Protocol):
    """对象工厂协议"""
    
    def create(self) -> PoolObject:
        """创建新对象"""
        ...
    
    def validate(self, obj: PoolObject) -> bool:
        """验证对象是否有效"""
        ...
    
    def reset(self, obj: PoolObject) -> None:
        """重置对象状态"""
        ...
    
    def destroy(self, obj: PoolObject) -> None:
        """销毁对象"""
        ...


class DefaultObjectFactory:
    """默认对象工厂实现"""
    
    def __init__(self, factory_func: Callable[[], PoolObject], 
                 validator: Optional[Callable[[PoolObject], bool]] = None,
                 resetter: Optional[Callable[[PoolObject], None]] = None,
                 destroyer: Optional[Callable[[PoolObject], None]] = None):
        """
        初始化默认对象工厂
        
        Args:
            factory_func: 对象创建函数
            validator: 对象验证函数
            resetter: 对象重置函数
            destroyer: 对象销毁函数
        """
        self.factory_func = factory_func
        self.validator = validator or (lambda obj: obj is not None)
        self.resetter = resetter or (lambda obj: None)
        self.destroyer = destroyer or (lambda obj: None)
    
    def create(self) -> PoolObject:
        """创建新对象"""
        return self.factory_func()
    
    def validate(self, obj: PoolObject) -> bool:
        """验证对象"""
        try:
            return self.validator(obj)
        except Exception as e:
            logger.warning(f"对象验证失败: {e}")
            return False
    
    def reset(self, obj: PoolObject) -> None:
        """重置对象"""
        try:
            self.resetter(obj)
        except Exception as e:
            logger.warning(f"对象重置失败: {e}")
    
    def destroy(self, obj: PoolObject) -> None:
        """销毁对象"""
        try:
            self.destroyer(obj)
        except Exception as e:
            logger.warning(f"对象销毁失败: {e}")


class ObjectPool(Generic[PoolObject]):
    """
    通用对象池
    
    提供线程安全的对象池管理，支持动态调整大小、对象验证和性能监控
    """
    
    def __init__(self, 
                 factory: Union[ObjectFactory, Callable[[], PoolObject]],
                 config: Optional[PoolConfig] = None):
        """
        初始化对象池
        
        Args:
            factory: 对象工厂或创建函数
            config: 池配置
        """
        self.config = config or PoolConfig()
        
        # 处理工厂函数
        if callable(factory) and not hasattr(factory, 'create'):
            self.factory = DefaultObjectFactory(factory)
        else:
            self.factory = factory
        
        # 池存储
        self._available: deque = deque()
        self._in_use: Set[PoolObject] = set()
        self._all_objects: Set[PoolObject] = set()
        
        # 同步控制
        if self.config.thread_safe:
            self._lock = RLock()
            self._condition = Condition(self._lock)
        else:
            self._lock = None
            self._condition = None
        
        # 统计信息
        self.stats = PoolStats()
        
        # 验证相关
        self._last_validation = 0.0
        self._validation_task: Optional[asyncio.Task] = None
        
        # 初始化池
        self._initialize_pool()
        
        # 启动验证任务
        if self.config.validation_enabled:
            self._start_validation_task()
        
        logger.info(f"对象池已初始化，最大大小: {self.config.max_size}")
    
    def _initialize_pool(self) -> None:
        """初始化对象池"""
        with self._get_lock():
            # 预分配对象
            if self.config.preallocate:
                for _ in range(self.config.preallocate_count):
                    obj = self._create_object()
                    if obj:
                        self._available.append(obj)
                        self._all_objects.add(obj)
                        self.stats.current_size += 1
                        self.stats.peak_size = max(self.stats.peak_size, self.stats.current_size)
            
            # 确保最小大小
            while len(self._available) < self.config.min_size and len(self._all_objects) < self.config.max_size:
                obj = self._create_object()
                if obj:
                    self._available.append(obj)
                    self._all_objects.add(obj)
                    self.stats.current_size += 1
                    self.stats.peak_size = max(self.stats.peak_size, self.stats.current_size)
    
    def _create_object(self) -> Optional[PoolObject]:
        """创建新对象"""
        try:
            obj = self.factory.create()
            if obj is not None:
                self.stats.total_created += 1
                logger.debug(f"创建新对象，总数: {self.stats.current_size + 1}")
                return obj
        except Exception as e:
            logger.error(f"创建对象失败: {e}")
        return None
    
    def _validate_object(self, obj: PoolObject) -> bool:
        """验证对象"""
        try:
            return self.factory.validate(obj)
        except Exception as e:
            logger.warning(f"对象验证异常: {e}")
            return False
    
    def _reset_object(self, obj: PoolObject) -> None:
        """重置对象状态"""
        try:
            self.factory.reset(obj)
        except Exception as e:
            logger.warning(f"对象重置异常: {e}")
    
    def _destroy_object(self, obj: PoolObject) -> None:
        """销毁对象"""
        try:
            self.factory.destroy(obj)
            self.stats.total_destroyed += 1
            logger.debug(f"销毁对象，总数: {self.stats.current_size - 1}")
        except Exception as e:
            logger.error(f"对象销毁失败: {e}")
    
    def _get_lock(self):
        """获取锁"""
        return self._lock if self._lock else _DummyLock()
    
    def borrow(self, timeout: Optional[float] = None) -> PoolObject:
        """
        借出对象
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            PoolObject: 借出的对象
            
        Raises:
            TimeoutError: 借出超时
        """
        start_time = time.time()
        
        with self._get_lock():
            while True:
                # 尝试从可用对象中获取
                if self._available:
                    obj = self._available.popleft()
                    
                    # 验证对象
                    if self._validate_object(obj):
                        self._in_use.add(obj)
                        self.stats.total_borrowed += 1
                        self.stats.in_use_count = len(self._in_use)
                        self.stats.available_count = len(self._available)
                        
                        # 记录借用时间
                        if hasattr(obj, '_borrow_time'):
                            self.stats.avg_borrow_time = (
                                (self.stats.avg_borrow_time * (self.stats.total_borrowed - 1) + 
                                 time.time() - obj._borrow_time) / self.stats.total_borrowed
                            )
                        
                        return obj
                    else:
                        # 对象无效，销毁并继续
                        self._destroy_object(obj)
                        self._all_objects.discard(obj)
                        self.stats.current_size -= 1
                        self.stats.validation_failures += 1
                
                # 如果池未满，创建新对象
                if len(self._all_objects) < self.config.max_size:
                    obj = self._create_object()
                    if obj:
                        self._in_use.add(obj)
                        self._all_objects.add(obj)
                        self.stats.total_borrowed += 1
                        self.stats.in_use_count = len(self._in_use)
                        self.stats.available_count = len(self._available)
                        self.stats.current_size = len(self._all_objects)
                        self.stats.peak_size = max(self.stats.peak_size, self.stats.current_size)
                        return obj
                
                # 检查超时
                if timeout is not None and (time.time() - start_time) >= timeout:
                    raise TimeoutError("借出对象超时")
                
                # 等待对象归还
                if self._condition:
                    try:
                        self._condition.wait(timeout=1.0)
                    except:
                        pass
                else:
                    time.sleep(0.001)  # 非线程安全模式下的短暂等待
    
    def return_object(self, obj: PoolObject) -> None:
        """
        归还对象
        
        Args:
            obj: 要归还的对象
        """
        with self._get_lock():
            if obj not in self._in_use:
                logger.warning("尝试归还未借出的对象")
                return
            
            self._in_use.remove(obj)
            
            # 重置对象状态
            self._reset_object(obj)
            
            # 验证对象
            if self._validate_object(obj):
                self._available.append(obj)
                self.stats.total_returned += 1
                self.stats.in_use_count = len(self._in_use)
                self.stats.available_count = len(self._available)
                
                # 记录归还时间
                obj._return_time = time.time()
                if hasattr(obj, '_borrow_time'):
                    borrow_duration = obj._return_time - obj._borrow_time
                    self.stats.avg_return_time = (
                        (self.stats.avg_return_time * (self.stats.total_returned - 1) + 
                         borrow_duration) / self.stats.total_returned
                    )
            else:
                # 对象无效，销毁
                self._destroy_object(obj)
                self._all_objects.discard(obj)
                self.stats.current_size -= 1
                self.stats.validation_failures += 1
            
            # 通知等待线程
            if self._condition:
                self._condition.notify()
    
    @contextmanager
    def get_object(self, timeout: Optional[float] = None):
        """
        获取对象的上下文管理器
        
        Args:
            timeout: 超时时间（秒）
            
        Yields:
            PoolObject: 借出的对象
        """
        obj = self.borrow(timeout)
        try:
            yield obj
        finally:
            self.return_object(obj)
    
    def _start_validation_task(self) -> None:
        """启动验证任务"""
        async def validation_loop():
            while True:
                try:
                    await asyncio.sleep(self.config.validation_interval)
                    await self._validate_pool()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"池验证任务出错: {e}")
        
        self._validation_task = asyncio.create_task(validation_loop())
    
    async def _validate_pool(self) -> None:
        """验证池中的对象"""
        with self._get_lock():
            current_time = time.time()
            self.stats.last_validation = current_time
            
            # 验证可用对象
            validated_objects = deque()
            invalid_objects = []
            
            while self._available:
                obj = self._available.popleft()
                
                # 检查空闲时间
                if hasattr(obj, '_return_time'):
                    idle_time = current_time - obj._return_time
                    if idle_time > self.config.max_idle_time:
                        invalid_objects.append(obj)
                        continue
                
                # 验证对象
                if self._validate_object(obj):
                    validated_objects.append(obj)
                else:
                    invalid_objects.append(obj)
                    self.stats.validation_failures += 1
            
            # 更新可用对象列表
            self._available = validated_objects
            
            # 销毁无效对象
            for obj in invalid_objects:
                self._destroy_object(obj)
                self._all_objects.discard(obj)
                self.stats.current_size -= 1
            
            # 调整池大小
            await self._adjust_pool_size()
    
    async def _adjust_pool_size(self) -> None:
        """调整池大小"""
        current_size = len(self._all_objects)
        
        if self.config.grow_strategy == "dynamic":
            # 根据使用情况动态增长
            if (len(self._in_use) > current_size * 0.8 and 
                current_size < self.config.max_size):
                # 使用率超过80%，增加池大小
                growth_factor = min(10, self.config.max_size - current_size)
                for _ in range(growth_factor):
                    obj = self._create_object()
                    if obj:
                        self._available.append(obj)
                        self._all_objects.add(obj)
                        self.stats.current_size += 1
        
        elif self.config.shrink_strategy == "idle":
            # 根据空闲情况收缩
            if (len(self._available) > current_size * 0.6 and 
                current_size > self.config.min_size):
                # 空闲率超过60%，减少池大小
                shrink_count = min(5, current_size - self.config.min_size)
                for _ in range(shrink_count):
                    if self._available:
                        obj = self._available.popleft()
                        self._destroy_object(obj)
                        self._all_objects.discard(obj)
                        self.stats.current_size -= 1
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取池统计信息
        
        Returns:
            Dict: 统计信息
        """
        with self._get_lock():
            return {
                'config': {
                    'max_size': self.config.max_size,
                    'min_size': self.config.min_size,
                    'max_idle_time': self.config.max_idle_time,
                    'validation_enabled': self.config.validation_enabled,
                    'thread_safe': self.config.thread_safe
                },
                'stats': {
                    'total_created': self.stats.total_created,
                    'total_destroyed': self.stats.total_destroyed,
                    'total_borrowed': self.stats.total_borrowed,
                    'total_returned': self.stats.total_returned,
                    'current_size': self.stats.current_size,
                    'available_count': len(self._available),
                    'in_use_count': len(self._in_use),
                    'peak_size': self.stats.peak_size,
                    'avg_borrow_time': self.stats.avg_borrow_time,
                    'avg_return_time': self.stats.avg_return_time,
                    'validation_failures': self.stats.validation_failures,
                    'utilization_rate': len(self._in_use) / max(self.stats.current_size, 1),
                    'idle_rate': len(self._available) / max(self.stats.current_size, 1)
                },
                'timestamp': time.time()
            }
    
    def clear(self) -> None:
        """清空对象池"""
        with self._get_lock():
            # 销毁所有对象
            for obj in list(self._available):
                self._destroy_object(obj)
            
            for obj in list(self._in_use):
                self._destroy_object(obj)
            
            # 清空集合
            self._available.clear()
            self._in_use.clear()
            self._all_objects.clear()
            
            # 重置统计
            self.stats.current_size = 0
            self.stats.available_count = 0
            self.stats.in_use_count = 0
            
            logger.info("对象池已清空")
    
    def resize(self, new_max_size: int, new_min_size: Optional[int] = None) -> None:
        """
        调整池大小
        
        Args:
            new_max_size: 新的最大大小
            new_min_size: 新的最小大小
        """
        with self._get_lock():
            old_max_size = self.config.max_size
            self.config.max_size = new_max_size
            
            if new_min_size is not None:
                self.config.min_size = new_min_size
            
            # 如果新大小小于当前大小，需要销毁多余对象
            if new_max_size < len(self._all_objects):
                objects_to_remove = len(self._all_objects) - new_max_size
                
                # 从可用对象中移除
                for _ in range(min(objects_to_remove, len(self._available))):
                    if self._available:
                        obj = self._available.popleft()
                        self._destroy_object(obj)
                        self._all_objects.remove(obj)
                        self.stats.current_size -= 1
            
            logger.info(f"池大小调整: {old_max_size} -> {new_max_size}")
    
    def shutdown(self) -> None:
        """关闭对象池"""
        # 停止验证任务
        if self._validation_task:
            self._validation_task.cancel()
        
        # 清空池
        self.clear()
        
        logger.info("对象池已关闭")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()


class _DummyLock:
    """非线程安全模式的虚拟锁"""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class ConnectionPool(ObjectPool):
    """
    连接池
    
    专门用于管理网络连接、数据库连接等资源
    """
    
    def __init__(self, 
                 connection_factory: Callable[[], Any],
                 config: Optional[PoolConfig] = None,
                 connection_validator: Optional[Callable[[Any], bool]] = None):
        """
        初始化连接池
        
        Args:
            connection_factory: 连接创建函数
            config: 池配置
            connection_validator: 连接验证函数
        """
        # 默认连接池配置
        default_config = PoolConfig(
            max_size=50,
            min_size=5,
            max_idle_time=600.0,  # 连接空闲10分钟
            validation_enabled=True,
            validation_interval=30.0,
            thread_safe=True
        )
        
        if config:
            default_config.update(config.__dict__)
        
        # 创建连接工厂
        connection_factory_obj = DefaultObjectFactory(
            factory_func=connection_factory,
            validator=connection_validator or self._default_connection_validator,
            resetter=self._default_connection_resetter,
            destroyer=self._default_connection_destroyer
        )
        
        super().__init__(connection_factory_obj, default_config)
        
        self.connection_stats = {
            'total_connections_created': 0,
            'total_connections_destroyed': 0,
            'connection_errors': 0,
            'last_connection_error': None,
            'avg_connection_time': 0.0
        }
        
        logger.info("连接池已初始化")
    
    def _default_connection_validator(self, conn: Any) -> bool:
        """默认连接验证器"""
        try:
            if hasattr(conn, 'ping'):
                return conn.ping()
            elif hasattr(conn, 'execute'):
                # 尝试执行简单查询
                conn.execute('SELECT 1')
                return True
            elif hasattr(conn, 'is_connected'):
                return conn.is_connected()
            else:
                # 基本存在性检查
                return conn is not None
        except Exception:
            return False
    
    def _default_connection_resetter(self, conn: Any) -> None:
        """默认连接重置器"""
        try:
            if hasattr(conn, 'clear'):
                conn.clear()
            elif hasattr(conn, 'reset'):
                conn.reset()
        except Exception as e:
            logger.warning(f"连接重置失败: {e}")
    
    def _default_connection_destroyer(self, conn: Any) -> None:
        """默认连接销毁器"""
        try:
            if hasattr(conn, 'close'):
                conn.close()
            elif hasattr(conn, 'disconnect'):
                conn.disconnect()
        except Exception as e:
            logger.warning(f"连接销毁失败: {e}")
    
    def get_connection(self, timeout: Optional[float] = None) -> Any:
        """
        获取连接
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            Any: 连接对象
        """
        start_time = time.time()
        
        try:
            conn = self.borrow(timeout)
            
            # 记录连接创建时间
            connection_time = time.time() - start_time
            self.connection_stats['total_connections_created'] += 1
            self.connection_stats['avg_connection_time'] = (
                (self.connection_stats['avg_connection_time'] * 
                 (self.connection_stats['total_connections_created'] - 1) + 
                 connection_time) / self.connection_stats['total_connections_created']
            )
            
            return conn
            
        except Exception as e:
            self.connection_stats['connection_errors'] += 1
            self.connection_stats['last_connection_error'] = {
                'error': str(e),
                'timestamp': time.time()
            }
            raise
    
    def return_connection(self, conn: Any) -> None:
        """
        归还连接
        
        Args:
            conn: 连接对象
        """
        try:
            self.return_object(conn)
            self.connection_stats['total_connections_destroyed'] += 1
        except Exception as e:
            logger.error(f"归还连接失败: {e}")
    
    @contextmanager
    def connection(self, timeout: Optional[float] = None):
        """
        连接上下文管理器
        
        Args:
            timeout: 超时时间（秒）
            
        Yields:
            Any: 连接对象
        """
        conn = self.get_connection(timeout)
        try:
            yield conn
        finally:
            self.return_connection(conn)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        获取连接统计信息
        
        Returns:
            Dict: 连接统计信息
        """
        base_stats = self.get_stats()
        base_stats['connection_stats'] = self.connection_stats.copy()
        
        return base_stats


class DatabaseConnectionPool(ConnectionPool):
    """
    数据库连接池
    """
    
    def __init__(self, 
                 db_config: Dict[str, Any],
                 config: Optional[PoolConfig] = None):
        """
        初始化数据库连接池
        
        Args:
            db_config: 数据库配置
            config: 池配置
        """
        self.db_config = db_config
        
        def create_db_connection():
            if db_config.get('type') == 'sqlite':
                return sqlite3.connect(db_config['database'])
            elif db_config.get('type') == 'postgresql':
                # 这里可以添加PostgreSQL连接逻辑
                import psycopg2
                return psycopg2.connect(**db_config['params'])
            else:
                raise ValueError(f"不支持的数据库类型: {db_config.get('type')}")
        
        super().__init__(create_db_connection, config)
        logger.info(f"数据库连接池已初始化: {db_config.get('type')}")


class HTTPConnectionPool(ConnectionPool):
    """
    HTTP连接池
    """
    
    def __init__(self, 
                 base_url: str,
                 config: Optional[PoolConfig] = None,
                 session_config: Optional[Dict[str, Any]] = None):
        """
        初始化HTTP连接池
        
        Args:
            base_url: 基础URL
            config: 池配置
            session_config: 会话配置
        """
        self.base_url = base_url
        self.session_config = session_config or {}
        
        def create_http_session():
            session = requests.Session()
            
            # 应用会话配置
            for key, value in self.session_config.items():
                if hasattr(session, key):
                    setattr(session, key, value)
            
            return session
        
        super().__init__(create_http_session, config)
        logger.info(f"HTTP连接池已初始化: {base_url}")


class SocketConnectionPool(ConnectionPool):
    """
    Socket连接池
    """
    
    def __init__(self, 
                 host: str,
                 port: int,
                 config: Optional[PoolConfig] = None,
                 socket_config: Optional[Dict[str, Any]] = None):
        """
        初始化Socket连接池
        
        Args:
            host: 主机地址
            port: 端口号
            config: 池配置
            socket_config: Socket配置
        """
        self.host = host
        self.port = port
        self.socket_config = socket_config or {}
        
        def create_socket_connection():
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # 应用Socket配置
            for key, value in self.socket_config.items():
                if hasattr(socket, key.upper()):
                    sock.setsockopt(socket.SOL_SOCKET, getattr(socket, key.upper()), value)
            
            sock.connect((host, port))
            return sock
        
        super().__init__(create_socket_connection, config)
        logger.info(f"Socket连接池已初始化: {host}:{port}")


# 使用示例
def example_usage():
    """使用示例"""
    
    # 1. 基本对象池示例
    class TestObject:
        def __init__(self, value: int = 0):
            self.value = value
            self.created_at = time.time()
    
    def create_test_object():
        return TestObject(value=len(TestObject.__dict__))
    
    pool_config = PoolConfig(
        max_size=20,
        min_size=5,
        validation_enabled=True
    )
    
    with ObjectPool(create_test_object, pool_config) as pool:
        # 使用上下文管理器
        with pool.get_object() as obj:
            print(f"使用对象: {obj.value}")
        
        # 直接借用和归还
        obj = pool.borrow(timeout=5.0)
        try:
            print(f"借用对象: {obj.value}")
        finally:
            pool.return_object(obj)
        
        # 获取统计信息
        stats = pool.get_stats()
        print(f"池统计: {stats}")
    
    # 2. 数据库连接池示例
    db_config = {
        'type': 'sqlite',
        'database': ':memory:'
    }
    
    with DatabaseConnectionPool(db_config) as db_pool:
        with db_pool.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("CREATE TABLE test (id INTEGER, name TEXT)")
            cursor.execute("INSERT INTO test VALUES (1, 'test')")
            conn.commit()
    
    # 3. HTTP连接池示例
    with HTTPConnectionPool("https://httpbin.org") as http_pool:
        with http_pool.connection() as session:
            response = session.get("/get")
            print(f"HTTP响应: {response.status_code}")


if __name__ == "__main__":
    example_usage()