"""
W1网络连接管理器 - 主要实现

提供完整的网络连接管理解决方案，包括连接管理、连接池、监控、重试等功能。
"""

import socket
import ssl
import time
import threading
import logging
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from collections import defaultdict, deque
from contextlib import contextmanager
import queue
import select
import urllib.request
import urllib.error
import urllib.parse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConnectionState(Enum):
    """连接状态枚举"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    FAILED = "failed"
    CLOSING = "closing"
    CLOSED = "closed"


class ConnectionType(Enum):
    """连接类型枚举"""
    TCP = "tcp"
    UDP = "udp"
    HTTP = "http"
    HTTPS = "https"
    SSL = "ssl"


@dataclass
class ConnectionConfig:
    """连接配置类"""
    # 基础配置
    host: str
    port: int
    connection_type: ConnectionType = ConnectionType.TCP
    
    # 超时配置
    connect_timeout: float = 30.0
    read_timeout: float = 60.0
    write_timeout: float = 60.0
    
    # 重试配置
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    
    # 连接池配置
    max_connections: int = 10
    min_connections: int = 2
    connection_idle_timeout: float = 300.0
    pool_check_interval: float = 60.0
    
    # SSL/TLS配置
    ssl_enabled: bool = False
    ssl_verify: bool = True
    ssl_cert_file: Optional[str] = None
    ssl_key_file: Optional[str] = None
    ssl_ca_file: Optional[str] = None
    
    # 安全配置
    enable_security_check: bool = True
    max_connection_time: float = 3600.0
    rate_limit: Optional[int] = None
    
    # 监控配置
    enable_monitoring: bool = True
    health_check_interval: float = 30.0
    stats_enabled: bool = True
    
    # 其他配置
    keep_alive: bool = True
    tcp_nodelay: bool = True
    buffer_size: int = 8192


@dataclass
class ConnectionStats:
    """连接统计信息"""
    total_connections: int = 0
    active_connections: int = 0
    failed_connections: int = 0
    successful_connections: int = 0
    total_bytes_sent: int = 0
    total_bytes_received: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    last_activity_time: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'total_connections': self.total_connections,
            'active_connections': self.active_connections,
            'failed_connections': self.failed_connections,
            'successful_connections': self.successful_connections,
            'total_bytes_sent': self.total_bytes_sent,
            'total_bytes_received': self.total_bytes_received,
            'total_requests': self.total_requests,
            'successful_requests': self.successful_requests,
            'failed_requests': self.failed_requests,
            'average_response_time': self.average_response_time,
            'last_activity_time': self.last_activity_time
        }


class Connection(ABC):
    """连接抽象基类"""
    
    def __init__(self, config: ConnectionConfig, connection_id: str):
        self.config = config
        self.connection_id = connection_id
        self.state = ConnectionState.DISCONNECTED
        self.created_time = time.time()
        self.last_activity_time = time.time()
        self.bytes_sent = 0
        self.bytes_received = 0
        self.request_count = 0
        self.error_count = 0
        self._lock = threading.RLock()
        
    @abstractmethod
    def connect(self) -> bool:
        """建立连接"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass
    
    @abstractmethod
    def send(self, data: Union[str, bytes]) -> bool:
        """发送数据"""
        pass
    
    @abstractmethod
    def receive(self, buffer_size: Optional[int] = None) -> Optional[bytes]:
        """接收数据"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """检查连接状态"""
        pass
    
    def update_activity(self):
        """更新活动状态"""
        with self._lock:
            self.last_activity_time = time.time()
    
    def get_uptime(self) -> float:
        """获取连接运行时间"""
        return time.time() - self.created_time
    
    def get_idle_time(self) -> float:
        """获取空闲时间"""
        return time.time() - self.last_activity_time
    
    def get_stats(self) -> Dict[str, Any]:
        """获取连接统计"""
        with self._lock:
            return {
                'connection_id': self.connection_id,
                'state': self.state.value,
                'created_time': self.created_time,
                'uptime': self.get_uptime(),
                'idle_time': self.get_idle_time(),
                'bytes_sent': self.bytes_sent,
                'bytes_received': self.bytes_received,
                'request_count': self.request_count,
                'error_count': self.error_count,
                'config': {
                    'host': self.config.host,
                    'port': self.config.port,
                    'connection_type': self.config.connection_type.value
                }
            }


class TCPConnection(Connection):
    """TCP连接实现"""
    
    def __init__(self, config: ConnectionConfig, connection_id: str):
        super().__init__(config, connection_id)
        self.socket: Optional[socket.socket] = None
        
    def connect(self) -> bool:
        """建立TCP连接"""
        try:
            self.state = ConnectionState.CONNECTING
            
            # 创建socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # 设置socket选项
            if self.config.keep_alive:
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            
            if self.config.tcp_nodelay:
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            
            # 设置超时
            self.socket.settimeout(self.config.connect_timeout)
            
            # 连接
            self.socket.connect((self.config.host, self.config.port))
            
            # 设置读写超时
            self.socket.settimeout(self.config.read_timeout)
            
            self.state = ConnectionState.CONNECTED
            self.update_activity()
            logger.info(f"TCP连接建立成功: {self.connection_id}")
            return True
            
        except Exception as e:
            self.state = ConnectionState.FAILED
            self.error_count += 1
            logger.error(f"TCP连接失败: {self.connection_id}, 错误: {e}")
            return False
    
    def disconnect(self) -> None:
        """断开TCP连接"""
        try:
            self.state = ConnectionState.CLOSING
            if self.socket:
                self.socket.close()
            self.socket = None
            self.state = ConnectionState.CLOSED
            logger.info(f"TCP连接已断开: {self.connection_id}")
        except Exception as e:
            logger.error(f"断开TCP连接时出错: {self.connection_id}, 错误: {e}")
    
    def send(self, data: Union[str, bytes]) -> bool:
        """发送TCP数据"""
        try:
            if not self.is_connected():
                return False
            
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            sent = self.socket.send(data)
            self.bytes_sent += sent
            self.request_count += 1
            self.update_activity()
            return sent > 0
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"TCP发送数据失败: {self.connection_id}, 错误: {e}")
            return False
    
    def receive(self, buffer_size: Optional[int] = None) -> Optional[bytes]:
        """接收TCP数据"""
        try:
            if not self.is_connected():
                return None
            
            buffer_size = buffer_size or self.config.buffer_size
            data = self.socket.recv(buffer_size)
            
            if data:
                self.bytes_received += len(data)
                self.update_activity()
                return data
            else:
                # 连接已关闭
                self.disconnect()
                return None
                
        except socket.timeout:
            return b""
        except Exception as e:
            self.error_count += 1
            logger.error(f"TCP接收数据失败: {self.connection_id}, 错误: {e}")
            return None
    
    def is_connected(self) -> bool:
        """检查TCP连接状态"""
        return self.state == ConnectionState.CONNECTED and self.socket is not None


class UDPConnection(Connection):
    """UDP连接实现"""
    
    def __init__(self, config: ConnectionConfig, connection_id: str):
        super().__init__(config, connection_id)
        self.socket: Optional[socket.socket] = None
        
    def connect(self) -> bool:
        """建立UDP连接"""
        try:
            self.state = ConnectionState.CONNECTING
            
            # UDP是无连接的，但我们创建socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(self.config.read_timeout)
            
            self.state = ConnectionState.CONNECTED
            self.update_activity()
            logger.info(f"UDP连接建立成功: {self.connection_id}")
            return True
            
        except Exception as e:
            self.state = ConnectionState.FAILED
            self.error_count += 1
            logger.error(f"UDP连接失败: {self.connection_id}, 错误: {e}")
            return False
    
    def disconnect(self) -> None:
        """断开UDP连接"""
        try:
            self.state = ConnectionState.CLOSING
            if self.socket:
                self.socket.close()
            self.socket = None
            self.state = ConnectionState.CLOSED
            logger.info(f"UDP连接已断开: {self.connection_id}")
        except Exception as e:
            logger.error(f"断开UDP连接时出错: {self.connection_id}, 错误: {e}")
    
    def send(self, data: Union[str, bytes]) -> bool:
        """发送UDP数据"""
        try:
            if not self.is_connected():
                return False
            
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            sent = self.socket.sendto(data, (self.config.host, self.config.port))
            self.bytes_sent += sent
            self.request_count += 1
            self.update_activity()
            return sent > 0
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"UDP发送数据失败: {self.connection_id}, 错误: {e}")
            return False
    
    def receive(self, buffer_size: Optional[int] = None) -> Optional[bytes]:
        """接收UDP数据"""
        try:
            if not self.is_connected():
                return None
            
            buffer_size = buffer_size or self.config.buffer_size
            data, addr = self.socket.recvfrom(buffer_size)
            
            if data:
                self.bytes_received += len(data)
                self.update_activity()
                return data
            else:
                return None
                
        except socket.timeout:
            return b""
        except Exception as e:
            self.error_count += 1
            logger.error(f"UDP接收数据失败: {self.connection_id}, 错误: {e}")
            return None
    
    def is_connected(self) -> bool:
        """检查UDP连接状态"""
        return self.state == ConnectionState.CONNECTED and self.socket is not None


class HTTPConnection(Connection):
    """HTTP连接实现"""
    
    def __init__(self, config: ConnectionConfig, connection_id: str):
        super().__init__(config, connection_id)
        self.base_url = f"http://{config.host}:{config.port}"
        self.session: Optional[urllib.request.urlopen] = None
        
    def connect(self) -> bool:
        """建立HTTP连接"""
        try:
            self.state = ConnectionState.CONNECTING
            
            # HTTP连接通过请求建立，这里只是验证连通性
            test_url = f"{self.base_url}/"
            request = Request(test_url, method='HEAD')
            
            with urlopen(request, timeout=self.config.connect_timeout) as response:
                if response.getcode() < 400:
                    self.state = ConnectionState.CONNECTED
                    self.update_activity()
                    logger.info(f"HTTP连接建立成功: {self.connection_id}")
                    return True
            
            self.state = ConnectionState.FAILED
            return False
            
        except Exception as e:
            self.state = ConnectionState.FAILED
            self.error_count += 1
            logger.error(f"HTTP连接失败: {self.connection_id}, 错误: {e}")
            return False
    
    def disconnect(self) -> None:
        """断开HTTP连接"""
        try:
            self.state = ConnectionState.CLOSING
            if self.session:
                self.session.close()
            self.session = None
            self.state = ConnectionState.CLOSED
            logger.info(f"HTTP连接已断开: {self.connection_id}")
        except Exception as e:
            logger.error(f"断开HTTP连接时出错: {self.connection_id}, 错误: {e}")
    
    def send(self, data: Union[str, bytes]) -> bool:
        """发送HTTP请求"""
        try:
            if not self.is_connected():
                return False
            
            if isinstance(data, str):
                # 假设是URL路径
                url = f"{self.base_url}{data}" if not data.startswith('http') else data
            else:
                # 假设是POST数据
                url = self.base_url
                data = data if isinstance(data, bytes) else str(data).encode('utf-8')
            
            request = Request(url, data=data, method='POST' if data else 'GET')
            
            with urlopen(request, timeout=self.config.read_timeout) as response:
                self.bytes_sent += len(data) if data else 0
                self.bytes_received += int(response.headers.get('Content-Length', 0))
                self.request_count += 1
                self.update_activity()
                return response.getcode() < 400
                
        except Exception as e:
            self.error_count += 1
            logger.error(f"HTTP发送请求失败: {self.connection_id}, 错误: {e}")
            return False
    
    def receive(self, buffer_size: Optional[int] = None) -> Optional[bytes]:
        """接收HTTP响应"""
        try:
            if not self.is_connected():
                return None
            
            # HTTP连接通常不主动接收数据，这里返回空
            return b""
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"HTTP接收响应失败: {self.connection_id}, 错误: {e}")
            return None
    
    def is_connected(self) -> bool:
        """检查HTTP连接状态"""
        return self.state == ConnectionState.CONNECTED


class SSLConnection(TCPConnection):
    """SSL/TLS连接实现"""
    
    def __init__(self, config: ConnectionConfig, connection_id: str):
        super().__init__(config, connection_id)
        self.ssl_context = None
        
    def connect(self) -> bool:
        """建立SSL连接"""
        try:
            self.state = ConnectionState.CONNECTING
            
            # 创建SSL上下文
            if self.config.ssl_ca_file:
                self.ssl_context = ssl.create_default_context(cafile=self.config.ssl_ca_file)
            else:
                self.ssl_context = ssl.create_default_context()
            
            if not self.config.ssl_verify:
                self.ssl_context.check_hostname = False
                self.ssl_context.verify_mode = ssl.CERT_NONE
            
            # 创建socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            
            # 设置超时
            self.socket.settimeout(self.config.connect_timeout)
            
            # 建立SSL连接
            self.socket = self.ssl_context.wrap_socket(
                self.socket,
                server_hostname=self.config.host
            )
            self.socket.connect((self.config.host, self.config.port))
            
            # 设置读写超时
            self.socket.settimeout(self.config.read_timeout)
            
            self.state = ConnectionState.CONNECTED
            self.update_activity()
            logger.info(f"SSL连接建立成功: {self.connection_id}")
            return True
            
        except Exception as e:
            self.state = ConnectionState.FAILED
            self.error_count += 1
            logger.error(f"SSL连接失败: {self.connection_id}, 错误: {e}")
            return False


class ConnectionPool:
    """连接池实现"""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connections: Dict[str, Connection] = {}
        self.available_connections: queue.Queue = queue.Queue()
        self.in_use_connections: set = set()
        self._lock = threading.RLock()
        self._shutdown = False
        
    def get_connection(self) -> Optional[Connection]:
        """获取连接"""
        with self._lock:
            # 尝试从可用连接池获取
            try:
                connection = self.available_connections.get_nowait()
                if connection.is_connected():
                    self.in_use_connections.add(connection.connection_id)
                    return connection
                else:
                    # 连接已失效，移除
                    if connection.connection_id in self.connections:
                        del self.connections[connection.connection_id]
            except queue.Empty:
                pass
            
            # 创建新连接
            if len(self.connections) < self.config.max_connections:
                connection = self._create_connection()
                if connection:
                    self.connections[connection.connection_id] = connection
                    self.in_use_connections.add(connection.connection_id)
                    return connection
            
            return None
    
    def return_connection(self, connection: Connection) -> None:
        """归还连接"""
        with self._lock:
            if connection.connection_id in self.in_use_connections:
                self.in_use_connections.remove(connection.connection_id)
                
                # 检查连接是否仍然有效
                if connection.is_connected():
                    self.available_connections.put(connection)
                else:
                    # 连接已失效，移除
                    if connection.connection_id in self.connections:
                        del self.connections[connection.connection_id]
    
    def _create_connection(self) -> Optional[Connection]:
        """创建新连接"""
        connection_id = f"{self.config.connection_type.value}_{self.config.host}_{self.config.port}_{int(time.time() * 1000)}"
        
        try:
            if self.config.connection_type == ConnectionType.TCP:
                connection = TCPConnection(self.config, connection_id)
            elif self.config.connection_type == ConnectionType.UDP:
                connection = UDPConnection(self.config, connection_id)
            elif self.config.connection_type in [ConnectionType.HTTP, ConnectionType.HTTPS]:
                connection = HTTPConnection(self.config, connection_id)
            elif self.config.connection_type == ConnectionType.SSL:
                connection = SSLConnection(self.config, connection_id)
            else:
                raise ValueError(f"不支持的连接类型: {self.config.connection_type}")
            
            if connection.connect():
                return connection
            else:
                return None
                
        except Exception as e:
            logger.error(f"创建连接失败: {connection_id}, 错误: {e}")
            return None
    
    def cleanup_idle_connections(self) -> None:
        """清理空闲连接"""
        with self._lock:
            current_time = time.time()
            idle_connections = []
            
            # 检查可用连接
            temp_queue = queue.Queue()
            while not self.available_connections.empty():
                try:
                    connection = self.available_connections.get_nowait()
                    if current_time - connection.last_activity_time > self.config.connection_idle_timeout:
                        idle_connections.append(connection)
                    else:
                        temp_queue.put(connection)
                except queue.Empty:
                    break
            
            # 恢复剩余连接
            while not temp_queue.empty:
                try:
                    connection = temp_queue.get_nowait()
                    self.available_connections.put(connection)
                except queue.Empty:
                    break
            
            # 关闭空闲连接
            for connection in idle_connections:
                if connection.connection_id in self.connections:
                    del self.connections[connection.connection_id]
                    connection.disconnect()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取连接池统计"""
        with self._lock:
            return {
                'total_connections': len(self.connections),
                'available_connections': self.available_connections.qsize(),
                'in_use_connections': len(self.in_use_connections),
                'max_connections': self.config.max_connections,
                'min_connections': self.config.min_connections
            }
    
    def shutdown(self) -> None:
        """关闭连接池"""
        with self._lock:
            self._shutdown = True
            for connection in self.connections.values():
                connection.disconnect()
            self.connections.clear()
            self.in_use_connections.clear()
            
            # 清空可用连接队列
            while not self.available_connections.empty():
                try:
                    self.available_connections.get_nowait()
                except queue.Empty:
                    break


class ConnectionMonitor:
    """连接监控器"""
    
    def __init__(self, manager: 'NetworkConnectionManager'):
        self.manager = manager
        self.monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self) -> None:
        """开始监控"""
        if not self.monitoring:
            self.monitoring = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.info("连接监控已启动")
    
    def stop_monitoring(self) -> None:
        """停止监控"""
        self.monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("连接监控已停止")
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        while self.monitoring:
            try:
                # 检查所有连接池
                for pool in self.manager.connection_pools.values():
                    pool.cleanup_idle_connections()
                    
                    # 健康检查
                    for connection in list(pool.connections.values()):
                        if not connection.is_connected():
                            # 移除失效连接
                            if connection.connection_id in pool.connections:
                                del pool.connections[connection.connection_id]
                
                time.sleep(self.manager.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"监控循环出错: {e}")
                time.sleep(5)


class ConnectionRetryManager:
    """连接重试管理器"""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.retry_counts: Dict[str, int] = defaultdict(int)
        self.last_retry_time: Dict[str, float] = {}
        self._lock = threading.RLock()
    
    def should_retry(self, connection_id: str) -> bool:
        """检查是否应该重试"""
        with self._lock:
            retry_count = self.retry_counts[connection_id]
            if retry_count >= self.config.max_retries:
                return False
            
            last_retry = self.last_retry_time.get(connection_id, 0)
            current_time = time.time()
            
            # 计算延迟时间（指数退避）
            delay = self.config.retry_delay * (self.config.retry_backoff ** retry_count)
            
            return current_time - last_retry >= delay
    
    def record_retry(self, connection_id: str) -> None:
        """记录重试"""
        with self._lock:
            self.retry_counts[connection_id] += 1
            self.last_retry_time[connection_id] = time.time()
    
    def reset_retry(self, connection_id: str) -> None:
        """重置重试计数"""
        with self._lock:
            self.retry_counts[connection_id] = 0
            if connection_id in self.last_retry_time:
                del self.last_retry_time[connection_id]


class ConnectionSecurityManager:
    """连接安全管理器"""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connection_times: Dict[str, float] = {}
        self.request_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()
    
    def is_connection_safe(self, connection_id: str) -> bool:
        """检查连接是否安全"""
        with self._lock:
            current_time = time.time()
            
            # 检查连接时间
            if connection_id in self.connection_times:
                connection_time = current_time - self.connection_times[connection_id]
                if connection_time > self.config.max_connection_time:
                    return False
            
            # 检查请求频率
            if self.config.rate_limit:
                request_count = self.request_counts[connection_id]
                time_window = 60  # 1分钟窗口
                
                # 重置计数器（简化实现）
                if request_count > self.config.rate_limit * time_window:
                    return False
            
            return True
    
    def register_connection(self, connection_id: str) -> None:
        """注册连接"""
        with self._lock:
            self.connection_times[connection_id] = time.time()
            self.request_counts[connection_id] = 0
    
    def unregister_connection(self, connection_id: str) -> None:
        """注销连接"""
        with self._lock:
            if connection_id in self.connection_times:
                del self.connection_times[connection_id]
            if connection_id in self.request_counts:
                del self.request_counts[connection_id]
    
    def record_request(self, connection_id: str) -> None:
        """记录请求"""
        with self._lock:
            self.request_counts[connection_id] += 1


class NetworkConnectionManager:
    """网络连接管理器主类"""
    
    def __init__(self, config: ConnectionConfig):
        self.config = config
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.global_stats = ConnectionStats()
        self.monitor = ConnectionMonitor(self)
        self.retry_manager = ConnectionRetryManager(config)
        self.security_manager = ConnectionSecurityManager(config)
        self._lock = threading.RLock()
        self._shutdown = False
        
        # 创建连接池
        pool_key = f"{config.connection_type.value}_{config.host}_{config.port}"
        self.connection_pools[pool_key] = ConnectionPool(config)
        
        # 启动监控
        if config.enable_monitoring:
            self.monitor.start_monitoring()
        
        logger.info(f"网络连接管理器已初始化: {pool_key}")
    
    @contextmanager
    def get_connection(self):
        """获取连接的上下文管理器"""
        connection = None
        try:
            pool_key = f"{self.config.connection_type.value}_{self.config.host}_{self.config.port}"
            pool = self.connection_pools.get(pool_key)
            
            if not pool:
                raise ValueError(f"连接池不存在: {pool_key}")
            
            connection = pool.get_connection()
            if not connection:
                raise ConnectionError("无法获取连接")
            
            # 安全检查
            if self.config.enable_security_check:
                if not self.security_manager.is_connection_safe(connection.connection_id):
                    raise SecurityError("连接安全检查失败")
            
            self.security_manager.register_connection(connection.connection_id)
            
            yield connection
            
        finally:
            if connection:
                try:
                    pool_key = f"{self.config.connection_type.value}_{self.config.host}_{self.config.port}"
                    pool = self.connection_pools.get(pool_key)
                    if pool:
                        pool.return_connection(connection)
                    
                    self.security_manager.unregister_connection(connection.connection_id)
                    
                except Exception as e:
                    logger.error(f"归还连接时出错: {e}")
    
    def send_data(self, data: Union[str, bytes], connection_id: Optional[str] = None) -> bool:
        """发送数据"""
        try:
            with self.get_connection() as connection:
                start_time = time.time()
                success = connection.send(data)
                end_time = time.time()
                
                # 更新统计
                with self._lock:
                    self.global_stats.total_requests += 1
                    if success:
                        self.global_stats.successful_requests += 1
                        self.global_stats.total_bytes_sent += len(data) if isinstance(data, (str, bytes)) else 0
                    else:
                        self.global_stats.failed_requests += 1
                    
                    # 更新平均响应时间
                    response_time = end_time - start_time
                    if self.global_stats.total_requests > 0:
                        self.global_stats.average_response_time = (
                            (self.global_stats.average_response_time * (self.global_stats.total_requests - 1) + response_time) 
                            / self.global_stats.total_requests
                        )
                    
                    self.global_stats.last_activity_time = time.time()
                
                # 记录请求
                if connection.connection_id:
                    self.security_manager.record_request(connection.connection_id)
                
                return success
                
        except Exception as e:
            logger.error(f"发送数据失败: {e}")
            with self._lock:
                self.global_stats.failed_requests += 1
            return False
    
    def receive_data(self, buffer_size: Optional[int] = None, connection_id: Optional[str] = None) -> Optional[bytes]:
        """接收数据"""
        try:
            with self.get_connection() as connection:
                data = connection.receive(buffer_size)
                
                if data:
                    with self._lock:
                        self.global_stats.total_bytes_received += len(data)
                        self.global_stats.last_activity_time = time.time()
                
                return data
                
        except Exception as e:
            logger.error(f"接收数据失败: {e}")
            return None
    
    def execute_request(self, method: str, url: str, data: Optional[Union[str, bytes]] = None, 
                       headers: Optional[Dict[str, str]] = None) -> Optional[Dict[str, Any]]:
        """执行HTTP请求"""
        try:
            start_time = time.time()
            
            if self.config.connection_type not in [ConnectionType.HTTP, ConnectionType.HTTPS]:
                raise ValueError("仅支持HTTP连接类型执行请求")
            
            with self.get_connection() as connection:
                if isinstance(connection, HTTPConnection):
                    # 发送请求
                    if data:
                        success = connection.send(data)
                    else:
                        success = connection.send(url)
                    
                    if success:
                        response_data = connection.receive()
                        
                        end_time = time.time()
                        response_time = end_time - start_time
                        
                        # 更新统计
                        with self._lock:
                            self.global_stats.total_requests += 1
                            self.global_stats.successful_requests += 1
                            self.global_stats.average_response_time = (
                                (self.global_stats.average_response_time * (self.global_stats.total_requests - 1) + response_time) 
                                / self.global_stats.total_requests
                            )
                            self.global_stats.last_activity_time = time.time()
                        
                        return {
                            'success': True,
                            'data': response_data,
                            'response_time': response_time
                        }
                    else:
                        with self._lock:
                            self.global_stats.failed_requests += 1
                        return {'success': False, 'error': '请求发送失败'}
                
                return {'success': False, 'error': '连接类型不支持请求'}
                
        except Exception as e:
            logger.error(f"执行请求失败: {e}")
            with self._lock:
                self.global_stats.failed_requests += 1
            return {'success': False, 'error': str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """获取全局统计信息"""
        with self._lock:
            stats = self.global_stats.to_dict()
            
            # 添加连接池统计
            pool_stats = {}
            for pool_key, pool in self.connection_pools.items():
                pool_stats[pool_key] = pool.get_stats()
            
            stats['connection_pools'] = pool_stats
            stats['config'] = {
                'host': self.config.host,
                'port': self.config.port,
                'connection_type': self.config.connection_type.value,
                'max_connections': self.config.max_connections,
                'connect_timeout': self.config.connect_timeout,
                'ssl_enabled': self.config.ssl_enabled
            }
            
            return stats
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            with self.get_connection() as connection:
                is_healthy = connection.is_connected()
                
                return {
                    'healthy': is_healthy,
                    'connection_id': connection.connection_id,
                    'state': connection.state.value,
                    'uptime': connection.get_uptime(),
                    'stats': connection.get_stats()
                }
                
        except Exception as e:
            return {
                'healthy': False,
                'error': str(e)
            }
    
    def shutdown(self) -> None:
        """关闭管理器"""
        with self._lock:
            self._shutdown = True
            
            # 停止监控
            if self.config.enable_monitoring:
                self.monitor.stop_monitoring()
            
            # 关闭所有连接池
            for pool in self.connection_pools.values():
                pool.shutdown()
            
            logger.info("网络连接管理器已关闭")


class SecurityError(Exception):
    """安全异常"""
    pass


# 便捷函数
def create_tcp_manager(host: str, port: int, **kwargs) -> NetworkConnectionManager:
    """创建TCP连接管理器"""
    config = ConnectionConfig(
        host=host,
        port=port,
        connection_type=ConnectionType.TCP,
        **kwargs
    )
    return NetworkConnectionManager(config)


def create_udp_manager(host: str, port: int, **kwargs) -> NetworkConnectionManager:
    """创建UDP连接管理器"""
    config = ConnectionConfig(
        host=host,
        port=port,
        connection_type=ConnectionType.UDP,
        **kwargs
    )
    return NetworkConnectionManager(config)


def create_http_manager(host: str, port: int, **kwargs) -> NetworkConnectionManager:
    """创建HTTP连接管理器"""
    config = ConnectionConfig(
        host=host,
        port=port,
        connection_type=ConnectionType.HTTP,
        **kwargs
    )
    return NetworkConnectionManager(config)


def create_https_manager(host: str, port: int, **kwargs) -> NetworkConnectionManager:
    """创建HTTPS连接管理器"""
    config = ConnectionConfig(
        host=host,
        port=port,
        connection_type=ConnectionType.HTTPS,
        ssl_enabled=True,
        **kwargs
    )
    return NetworkConnectionManager(config)


def create_ssl_manager(host: str, port: int, **kwargs) -> NetworkConnectionManager:
    """创建SSL连接管理器"""
    config = ConnectionConfig(
        host=host,
        port=port,
        connection_type=ConnectionType.SSL,
        ssl_enabled=True,
        **kwargs
    )
    return NetworkConnectionManager(config)