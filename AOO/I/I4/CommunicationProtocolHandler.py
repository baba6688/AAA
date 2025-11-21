#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通信协议处理器

该模块实现了一个综合性的通信协议处理器，支持多种网络协议和通信模式。
包括TCP/UDP、MQTT、HTTP/HTTPS、WebSocket等协议，并提供连接池管理、
消息序列化、协议安全加密和通信质量监控等功能。


版本: 1.0.0
日期: 2025-11-05
"""

import asyncio
import json
import logging
import ssl
import time
import uuid
import hashlib
import hmac
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import (
    Any, AsyncGenerator, Callable, Dict, List, Optional, 
    Tuple, Union, Set, Coroutine, TypeVar, Generic
)
from urllib.parse import urlparse
import base64
import struct
import weakref
import os

# 可选依赖
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 类型定义
T = TypeVar('T')
MessageType = Union[str, bytes, dict, list, Any]
ProtocolVersion = str


class ProtocolType(Enum):
    """协议类型枚举"""
    TCP = "tcp"
    UDP = "udp"
    HTTP = "http"
    HTTPS = "https"
    MQTT = "mqtt"
    WEBSOCKET = "websocket"


class ConnectionState(Enum):
    """连接状态枚举"""
    DISCONNECTED = auto()
    CONNECTING = auto()
    CONNECTED = auto()
    RECONNECTING = auto()
    FAILED = auto()
    CLOSED = auto()


class MessagePriority(Enum):
    """消息优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EncryptionMethod(Enum):
    """加密方法枚举"""
    NONE = "none"
    AES = "aes"
    RSA = "rsa"
    TLS = "tls"


@dataclass
class ConnectionConfig:
    """连接配置数据类"""
    host: str
    port: int
    protocol: ProtocolType
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    keep_alive: bool = True
    ssl_verify: bool = True
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    encryption: EncryptionMethod = EncryptionMethod.NONE
    encryption_key: Optional[str] = None


@dataclass
class Message:
    """消息数据类"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    content: MessageType = ""
    protocol: ProtocolType = ProtocolType.TCP
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    source: str = ""
    destination: str = ""
    headers: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    encrypted: bool = False
    compressed: bool = False
    
    def __post_init__(self):
        """消息初始化后处理"""
        if isinstance(self.content, (dict, list)):
            self.content = json.dumps(self.content, ensure_ascii=False)


@dataclass
class ConnectionMetrics:
    """连接性能指标数据类"""
    connection_id: str
    protocol: ProtocolType
    connected_at: float
    bytes_sent: int = 0
    bytes_received: int = 0
    messages_sent: int = 0
    messages_received: int = 0
    errors_count: int = 0
    last_activity: float = field(default_factory=time.time)
    latency_ms: float = 0.0
    throughput_kbps: float = 0.0
    
    def update_activity(self, bytes_transferred: int, is_sent: bool = True):
        """更新连接活动统计"""
        current_time = time.time()
        if is_sent:
            self.bytes_sent += bytes_transferred
            self.messages_sent += 1
        else:
            self.bytes_received += bytes_transferred
            self.messages_received += 1
        
        self.last_activity = current_time
        
        # 计算吞吐量 (KB/s)
        time_delta = current_time - self.connected_at
        if time_delta > 0:
            total_bytes = self.bytes_sent + self.bytes_received
            self.throughput_kbps = (total_bytes / 1024) / time_delta


@dataclass
class ProtocolVersionInfo:
    """协议版本信息数据类"""
    version: ProtocolVersion
    name: str
    description: str
    supported_features: Set[str] = field(default_factory=set)
    backward_compatible: bool = True
    release_date: str = ""


class SecurityManager:
    """安全管理器类"""
    
    def __init__(self):
        self._encryption_keys: Dict[str, bytes] = {}
        self._fernet_instances: Dict[str, Any] = {}
        self._crypto_available = CRYPTO_AVAILABLE
    
    def generate_key(self, password: str, salt: bytes = None) -> bytes:
        """生成加密密钥"""
        if not self._crypto_available:
            # 简单模拟实现
            return base64.urlsafe_b64encode(password.encode().ljust(32, b'\x00')[:32])
        
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def encrypt_message(self, message: Message, encryption_key: str) -> bytes:
        """加密消息"""
        if not self._crypto_available:
            # 简单模拟实现
            message_data = json.dumps({
                'content': message.content,
                'timestamp': message.timestamp,
                'metadata': message.metadata
            }).encode()
            return base64.b64encode(message_data)
        
        if encryption_key not in self._fernet_instances:
            key = self.generate_key(encryption_key)
            self._fernet_instances[encryption_key] = Fernet(key)
        
        fernet = self._fernet_instances[encryption_key]
        message_data = json.dumps({
            'content': message.content,
            'timestamp': message.timestamp,
            'metadata': message.metadata
        }).encode()
        
        encrypted_data = fernet.encrypt(message_data)
        return encrypted_data
    
    def decrypt_message(self, encrypted_data: bytes, encryption_key: str) -> Message:
        """解密消息"""
        if not self._crypto_available:
            # 简单模拟实现
            message_data = base64.b64decode(encrypted_data)
            data = json.loads(message_data.decode())
            return Message(
                content=data['content'],
                timestamp=data['timestamp'],
                metadata=data['metadata']
            )
        
        if encryption_key not in self._fernet_instances:
            raise ValueError(f"Encryption key {encryption_key} not found")
        
        fernet = self._fernet_instances[encryption_key]
        decrypted_data = fernet.decrypt(encrypted_data)
        data = json.loads(decrypted_data.decode())
        
        return Message(
            content=data['content'],
            timestamp=data['timestamp'],
            metadata=data['metadata']
        )
    
    def create_signature(self, message: Message, secret_key: str) -> str:
        """创建消息签名"""
        message_str = json.dumps({
            'content': message.content,
            'timestamp': message.timestamp,
            'metadata': message.metadata
        }, sort_keys=True)
        
        signature = hmac.new(
            secret_key.encode(),
            message_str.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return signature
    
    def verify_signature(self, message: Message, signature: str, secret_key: str) -> bool:
        """验证消息签名"""
        expected_signature = self.create_signature(message, secret_key)
        return hmac.compare_digest(signature, expected_signature)


class MessageSerializer:
    """消息序列化器类"""
    
    @staticmethod
    def serialize(message: Message, format: str = "json") -> bytes:
        """序列化消息"""
        if format == "json":
            data = {
                'id': message.id,
                'content': message.content,
                'protocol': message.protocol.value,
                'priority': message.priority.value,
                'timestamp': message.timestamp,
                'source': message.source,
                'destination': message.destination,
                'headers': message.headers,
                'metadata': message.metadata,
                'encrypted': message.encrypted,
                'compressed': message.compressed
            }
            return json.dumps(data, ensure_ascii=False).encode()
        
        elif format == "binary":
            # 二进制序列化格式
            header = struct.pack(
                '!16s8s4s4s4s? ?',  # 格式字符串
                message.id.encode(),
                message.protocol.value.encode(),
                message.priority.name.encode(),
                message.source.encode(),
                message.destination.encode(),
                message.encrypted,
                message.compressed
            )
            
            content_data = json.dumps(message.content).encode()
            metadata_data = json.dumps(message.metadata).encode()
            
            # 使用分隔符区分内容和元数据
            separator = b'\x00'
            return header + content_data + separator + metadata_data
        
        else:
            raise ValueError(f"Unsupported serialization format: {format}")
    
    @staticmethod
    def deserialize(data: bytes, format: str = "json") -> Message:
        """反序列化消息"""
        if format == "json":
            data_dict = json.loads(data.decode())
            return Message(
                id=data_dict['id'],
                content=data_dict['content'],
                protocol=ProtocolType(data_dict['protocol']),
                priority=MessagePriority(data_dict['priority']),
                timestamp=data_dict['timestamp'],
                source=data_dict['source'],
                destination=data_dict['destination'],
                headers=data_dict['headers'],
                metadata=data_dict['metadata'],
                encrypted=data_dict['encrypted'],
                compressed=data_dict['compressed']
            )
        
        elif format == "binary":
            # 二进制反序列化
            header_size = 16 + 8 + 4 + 4 + 4 + 1 + 1  # 计算头部大小
            header = data[:header_size]
            
            # 解析头部
            fields = struct.unpack('!16s8s4s4s4s? ?', header)
            message_id = fields[0].decode().rstrip('\x00')
            protocol = fields[1].decode().rstrip('\x00')
            priority = fields[2].decode().rstrip('\x00')
            source = fields[3].decode().rstrip('\x00')
            destination = fields[4].decode().rstrip('\x00')
            encrypted = fields[5]
            compressed = fields[6]
            
            # 解析内容和元数据
            remaining_data = data[header_size:]
            separator = b'\x00'
            null_index = remaining_data.find(separator)
            if null_index != -1:
                content_data = remaining_data[:null_index]
                metadata_data = remaining_data[null_index + 1:]
            else:
                content_data = remaining_data
                metadata_data = b''
            
            content = json.loads(content_data.decode())
            metadata = json.loads(metadata_data.decode()) if metadata_data else {}
            
            return Message(
                id=message_id,
                content=content,
                protocol=ProtocolType(protocol),
                priority=MessagePriority[priority],
                source=source,
                destination=destination,
                headers={},
                metadata=metadata,
                encrypted=encrypted,
                compressed=compressed
            )
        
        else:
            raise ValueError(f"Unsupported deserialization format: {format}")


class ConnectionPool:
    """连接池管理类"""
    
    def __init__(self, max_connections: int = 100):
        self.max_connections = max_connections
        self._pools: Dict[str, List[Any]] = defaultdict(list)
        self._active_connections: Dict[str, Any] = {}
        self._connection_configs: Dict[str, ConnectionConfig] = {}
        self._connection_metrics: Dict[str, ConnectionMetrics] = {}
        self._lock = asyncio.Lock()
    
    async def get_connection(self, config: ConnectionConfig) -> Any:
        """获取连接"""
        pool_key = f"{config.host}:{config.port}:{config.protocol.value}"
        
        async with self._lock:
            # 尝试从池中获取可用连接
            if self._pools[pool_key]:
                connection = self._pools[pool_key].pop()
                if self._is_connection_valid(connection):
                    self._active_connections[pool_key] = connection
                    return connection
            
            # 创建新连接
            connection = await self._create_connection(config)
            self._active_connections[pool_key] = connection
            
            # 记录连接指标
            metrics = ConnectionMetrics(
                connection_id=pool_key,
                protocol=config.protocol,
                connected_at=time.time()
            )
            self._connection_metrics[pool_key] = metrics
            
            return connection
    
    async def return_connection(self, connection: Any, config: ConnectionConfig):
        """归还连接"""
        pool_key = f"{config.host}:{config.port}:{config.protocol.value}"
        
        async with self._lock:
            if pool_key in self._active_connections:
                del self._active_connections[pool_key]
                
                if self._is_connection_valid(connection) and \
                   len(self._pools[pool_key]) < self.max_connections:
                    self._pools[pool_key].append(connection)
    
    def _is_connection_valid(self, connection: Any) -> bool:
        """检查连接是否有效"""
        try:
            if hasattr(connection, 'closed'):
                return not connection.closed
            elif hasattr(connection, 'is_connected'):
                return connection.is_connected()
            return True
        except Exception:
            return False
    
    async def _create_connection(self, config: ConnectionConfig) -> Any:
        """创建新连接"""
        if config.protocol in [ProtocolType.HTTP, ProtocolType.HTTPS]:
            connector = aiohttp.TCPConnector(
                ssl=config.ssl_verify,
                limit=1,
                limit_per_host=10
            )
            timeout = aiohttp.ClientTimeout(total=config.timeout)
            return aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=config.headers
            )
        
        elif config.protocol == ProtocolType.WEBSOCKET:
            # WebSocket连接会在使用时创建
            return None
        
        else:
            # TCP/UDP连接
            return None
    
    async def close_all(self):
        """关闭所有连接"""
        async with self._lock:
            # 关闭活动连接
            for connection in self._active_connections.values():
                try:
                    if hasattr(connection, 'close'):
                        await connection.close()
                except Exception as e:
                    logger.warning(f"Error closing connection: {e}")
            
            # 清空连接池
            for pool in self._pools.values():
                for connection in pool:
                    try:
                        if hasattr(connection, 'close'):
                            await connection.close()
                    except Exception as e:
                        logger.warning(f"Error closing pooled connection: {e}")
            
            self._pools.clear()
            self._active_connections.clear()
    
    def get_metrics(self) -> Dict[str, ConnectionMetrics]:
        """获取连接指标"""
        return self._connection_metrics.copy()


class QualityMonitor:
    """通信质量监控器类"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self._latency_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._throughput_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._error_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self._alerts: List[Dict[str, Any]] = []
        self._monitoring_active = True
    
    async def start_monitoring(self):
        """启动监控"""
        self._monitoring_active = True
        asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self):
        """停止监控"""
        self._monitoring_active = False
    
    async def record_latency(self, connection_id: str, latency_ms: float):
        """记录延迟"""
        self._latency_history[connection_id].append(latency_ms)
        
        # 检查是否触发延迟告警
        if latency_ms > 1000:  # 1秒延迟告警
            await self._create_alert(
                connection_id,
                "HIGH_LATENCY",
                f"High latency detected: {latency_ms}ms"
            )
    
    async def record_throughput(self, connection_id: str, throughput_kbps: float):
        """记录吞吐量"""
        self._throughput_history[connection_id].append(throughput_kbps)
    
    async def record_error(self, connection_id: str, error_type: str, error_message: str):
        """记录错误"""
        self._error_history[connection_id].append({
            'timestamp': time.time(),
            'type': error_type,
            'message': error_message
        })
        
        # 检查错误率
        recent_errors = list(self._error_history[connection_id])[-10:]  # 最近10个错误
        if len(recent_errors) >= 5:
            await self._create_alert(
                connection_id,
                "HIGH_ERROR_RATE",
                f"High error rate detected: {len(recent_errors)} errors in recent samples"
            )
    
    def get_quality_stats(self, connection_id: str) -> Dict[str, Any]:
        """获取质量统计"""
        latencies = list(self._latency_history[connection_id])
        throughputs = list(self._throughput_history[connection_id])
        errors = list(self._error_history[connection_id])
        
        return {
            'connection_id': connection_id,
            'avg_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
            'max_latency_ms': max(latencies) if latencies else 0,
            'min_latency_ms': min(latencies) if latencies else 0,
            'avg_throughput_kbps': sum(throughputs) / len(throughputs) if throughputs else 0,
            'error_count': len(errors),
            'recent_errors': errors[-5:] if errors else []  # 最近5个错误
        }
    
    async def _monitor_loop(self):
        """监控循环"""
        while self._monitoring_active:
            try:
                # 检查所有连接的质量指标
                for connection_id in list(self._latency_history.keys()):
                    stats = self.get_quality_stats(connection_id)
                    
                    # 检查延迟异常
                    if stats['avg_latency_ms'] > 500:
                        await self._create_alert(
                            connection_id,
                            "AVG_LATENCY_HIGH",
                            f"Average latency is high: {stats['avg_latency_ms']}ms"
                        )
                    
                    # 检查错误率
                    if stats['error_count'] > 10:
                        await self._create_alert(
                            connection_id,
                            "HIGH_ERROR_COUNT",
                            f"High error count: {stats['error_count']}"
                        )
                
                await asyncio.sleep(10)  # 每10秒检查一次
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    async def _create_alert(self, connection_id: str, alert_type: str, message: str):
        """创建告警"""
        alert = {
            'id': str(uuid.uuid4()),
            'connection_id': connection_id,
            'type': alert_type,
            'message': message,
            'timestamp': time.time(),
            'acknowledged': False
        }
        
        self._alerts.append(alert)
        logger.warning(f"Quality Alert [{alert_type}]: {message}")
    
    def get_alerts(self, acknowledged: Optional[bool] = None) -> List[Dict[str, Any]]:
        """获取告警列表"""
        if acknowledged is None:
            return self._alerts.copy()
        return [alert for alert in self._alerts if alert['acknowledged'] == acknowledged]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        for alert in self._alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                return True
        return False


class ProtocolHandler(ABC):
    """协议处理器抽象基类"""
    
    @abstractmethod
    async def connect(self, config: ConnectionConfig) -> Any:
        """建立连接"""
        pass
    
    @abstractmethod
    async def disconnect(self, connection: Any, config: ConnectionConfig):
        """断开连接"""
        pass
    
    @abstractmethod
    async def send_message(self, connection: Any, message: Message, config: ConnectionConfig) -> bool:
        """发送消息"""
        pass
    
    @abstractmethod
    async def receive_message(self, connection: Any, config: ConnectionConfig) -> Optional[Message]:
        """接收消息"""
        pass
    
    @abstractmethod
    async def is_connected(self, connection: Any) -> bool:
        """检查连接状态"""
        pass


class TCPHandler(ProtocolHandler):
    """TCP协议处理器"""
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}
    
    async def connect(self, config: ConnectionConfig) -> Any:
        """建立TCP连接"""
        try:
            reader, writer = await asyncio.open_connection(config.host, config.port)
            connection_id = f"{config.host}:{config.port}"
            self.connections[connection_id] = {
                'reader': reader,
                'writer': writer,
                'config': config,
                'connected_at': time.time()
            }
            logger.info(f"TCP connection established to {config.host}:{config.port}")
            return connection_id
        except Exception as e:
            logger.error(f"Failed to establish TCP connection: {e}")
            raise
    
    async def disconnect(self, connection_id: str, config: ConnectionConfig):
        """断开TCP连接"""
        if connection_id in self.connections:
            try:
                writer = self.connections[connection_id]['writer']
                writer.close()
                await writer.wait_closed()
                del self.connections[connection_id]
                logger.info(f"TCP connection closed: {connection_id}")
            except Exception as e:
                logger.error(f"Error closing TCP connection: {e}")
    
    async def send_message(self, connection_id: str, message: Message, config: ConnectionConfig) -> bool:
        """发送TCP消息"""
        if connection_id not in self.connections:
            return False
        
        try:
            writer = self.connections[connection_id]['writer']
            serialized_message = MessageSerializer.serialize(message, "json")
            writer.write(len(serialized_message).to_bytes(4, 'big') + serialized_message)
            await writer.drain()
            return True
        except Exception as e:
            logger.error(f"Error sending TCP message: {e}")
            return False
    
    async def receive_message(self, connection_id: str, config: ConnectionConfig) -> Optional[Message]:
        """接收TCP消息"""
        if connection_id not in self.connections:
            return None
        
        try:
            reader = self.connections[connection_id]['reader']
            
            # 读取消息长度
            length_data = await reader.readexactly(4)
            message_length = int.from_bytes(length_data, 'big')
            
            # 读取消息内容
            message_data = await reader.readexactly(message_length)
            
            return MessageSerializer.deserialize(message_data, "json")
        except asyncio.IncompleteReadError:
            logger.warning(f"TCP connection closed by peer: {connection_id}")
            return None
        except Exception as e:
            logger.error(f"Error receiving TCP message: {e}")
            return None
    
    async def is_connected(self, connection_id: str) -> bool:
        """检查TCP连接状态"""
        if connection_id not in self.connections:
            return False
        
        try:
            writer = self.connections[connection_id]['writer']
            return not writer.is_closing()
        except Exception:
            return False


class UDPHandler(ProtocolHandler):
    """UDP协议处理器"""
    
    def __init__(self):
        self.transport: Optional[asyncio.transports.DatagramTransport] = None
        self.protocol: Optional[asyncio.DatagramProtocol] = None
        self.message_queue: asyncio.Queue = asyncio.Queue()
    
    async def connect(self, config: ConnectionConfig) -> Any:
        """建立UDP连接（实际上是绑定）"""
        loop = asyncio.get_event_loop()
        
        # 创建UDP连接
        self.transport, self.protocol = await loop.create_datagram_endpoint(
            lambda: UDPProtocol(self.message_queue),
            local_addr=(config.host, config.port)
        )
        
        connection_id = f"udp:{config.host}:{config.port}"
        logger.info(f"UDP connection established on {config.host}:{config.port}")
        return connection_id
    
    async def disconnect(self, connection_id: str, config: ConnectionConfig):
        """断开UDP连接"""
        if self.transport:
            self.transport.close()
            self.transport = None
            self.protocol = None
            logger.info(f"UDP connection closed: {connection_id}")
    
    async def send_message(self, connection_id: str, message: Message, config: ConnectionConfig) -> bool:
        """发送UDP消息"""
        if not self.transport:
            return False
        
        try:
            serialized_message = MessageSerializer.serialize(message, "json")
            self.transport.sendto(serialized_message, (config.host, config.port))
            return True
        except Exception as e:
            logger.error(f"Error sending UDP message: {e}")
            return False
    
    async def receive_message(self, connection_id: str, config: ConnectionConfig) -> Optional[Message]:
        """接收UDP消息"""
        try:
            # 非阻塞方式获取消息
            message_data = await asyncio.wait_for(self.message_queue.get(), timeout=0.1)
            return MessageSerializer.deserialize(message_data, "json")
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error receiving UDP message: {e}")
            return None
    
    async def is_connected(self, connection_id: str) -> bool:
        """检查UDP连接状态"""
        return self.transport is not None and not self.transport.is_closing()


class UDPProtocol(asyncio.DatagramProtocol):
    """UDP协议实现"""
    
    def __init__(self, message_queue: asyncio.Queue):
        self.message_queue = message_queue
    
    def datagram_received(self, data: bytes, addr):
        """接收数据报"""
        try:
            self.message_queue.put_nowait(data)
        except Exception as e:
            logger.error(f"Error queuing UDP message: {e}")


class HTTPHandler(ProtocolHandler):
    """HTTP/HTTPS协议处理器"""
    
    def __init__(self):
        self.sessions: Dict[str, Any] = {}
        self._aiohttp_available = AIOHTTP_AVAILABLE
    
    async def connect(self, config: ConnectionConfig) -> Any:
        """建立HTTP连接"""
        if not self._aiohttp_available:
            # 模拟实现
            session_key = f"{config.protocol.value}:{config.host}:{config.port}"
            self.sessions[session_key] = {"mock": True, "config": config}
            logger.warning("aiohttp not available, using mock HTTP handler")
            return session_key
        
        try:
            session_key = f"{config.protocol.value}:{config.host}:{config.port}"
            
            if session_key not in self.sessions:
                connector = aiohttp.TCPConnector(
                    ssl=config.ssl_verify,
                    limit=100,
                    limit_per_host=10
                )
                
                timeout = aiohttp.ClientTimeout(total=config.timeout)
                
                self.sessions[session_key] = aiohttp.ClientSession(
                    connector=connector,
                    timeout=timeout,
                    headers=config.headers
                )
            
            logger.info(f"HTTP session created for {config.host}:{config.port}")
            return session_key
        except Exception as e:
            logger.error(f"Failed to establish HTTP connection: {e}")
            raise
    
    async def disconnect(self, session_key: str, config: ConnectionConfig):
        """断开HTTP连接"""
        if session_key in self.sessions:
            if not self._aiohttp_available:
                # 模拟实现
                del self.sessions[session_key]
                logger.info(f"Mock HTTP session closed: {session_key}")
            else:
                await self.sessions[session_key].close()
                del self.sessions[session_key]
                logger.info(f"HTTP session closed: {session_key}")
    
    async def send_message(self, session_key: str, message: Message, config: ConnectionConfig) -> bool:
        """发送HTTP请求"""
        if session_key not in self.sessions:
            return False
        
        if not self._aiohttp_available:
            # 模拟实现
            logger.info(f"Mock HTTP request: {message.content}")
            return True
        
        try:
            session = self.sessions[session_key]
            url = f"{config.protocol.value}://{config.host}:{config.port}"
            
            # 根据消息内容构建请求
            if isinstance(message.content, dict):
                data = message.content
                method = 'POST'
            else:
                data = {'message': message.content}
                method = 'GET'
            
            async with session.request(
                method=method,
                url=url,
                json=data,
                headers={'Content-Type': 'application/json'}
            ) as response:
                return response.status < 400
        except Exception as e:
            logger.error(f"Error sending HTTP message: {e}")
            return False
    
    async def receive_message(self, session_key: str, config: ConnectionConfig) -> Optional[Message]:
        """HTTP客户端通常不主动接收消息，这里返回None"""
        return None
    
    async def is_connected(self, session_key: str) -> bool:
        """检查HTTP连接状态"""
        if not self._aiohttp_available:
            return session_key in self.sessions
        return session_key in self.sessions and not self.sessions[session_key].closed


class MQTTHandler(ProtocolHandler):
    """MQTT协议处理器"""
    
    def __init__(self):
        self.clients: Dict[str, Any] = {}
        self.message_handlers: Dict[str, Callable] = {}
    
    async def connect(self, config: ConnectionConfig) -> Any:
        """建立MQTT连接"""
        try:
            import asyncio_mqtt as aiomqtt
            
            client_id = f"handler_{uuid.uuid4().hex[:8]}"
            
            client = aiomqtt.Client(
                client_id=client_id,
                protocol=aiomqtt.MQTTv5
            )
            
            if config.username and config.password:
                client.username_pw_set(config.username, config.password)
            
            await client.connect(config.host, config.port)
            
            connection_id = f"mqtt:{config.host}:{config.port}"
            self.clients[connection_id] = client
            
            logger.info(f"MQTT connection established to {config.host}:{config.port}")
            return connection_id
        except ImportError:
            logger.error("asyncio-mqtt not installed. Install with: pip install asyncio-mqtt")
            raise
        except Exception as e:
            logger.error(f"Failed to establish MQTT connection: {e}")
            raise
    
    async def disconnect(self, connection_id: str, config: ConnectionConfig):
        """断开MQTT连接"""
        if connection_id in self.clients:
            try:
                await self.clients[connection_id].disconnect()
                del self.clients[connection_id]
                logger.info(f"MQTT connection closed: {connection_id}")
            except Exception as e:
                logger.error(f"Error closing MQTT connection: {e}")
    
    async def send_message(self, connection_id: str, message: Message, config: ConnectionConfig) -> bool:
        """发送MQTT消息"""
        if connection_id not in self.clients:
            return False
        
        try:
            client = self.clients[connection_id]
            topic = message.destination or "default/topic"
            
            await client.publish(
                topic,
                payload=json.dumps({
                    'content': message.content,
                    'id': message.id,
                    'timestamp': message.timestamp,
                    'metadata': message.metadata
                }),
                qos=1
            )
            return True
        except Exception as e:
            logger.error(f"Error sending MQTT message: {e}")
            return False
    
    async def receive_message(self, connection_id: str, config: ConnectionConfig) -> Optional[Message]:
        """接收MQTT消息"""
        # MQTT是发布/订阅模式，消息接收通过订阅回调处理
        return None
    
    async def subscribe(self, connection_id: str, topic: str, handler: Callable[[Message], None]):
        """订阅MQTT主题"""
        if connection_id not in self.clients:
            return False
        
        try:
            client = self.clients[connection_id]
            
            async def message_handler(topic, payload):
                try:
                    data = json.loads(payload.decode())
                    message = Message(
                        id=data.get('id', str(uuid.uuid4())),
                        content=data.get('content', ''),
                        destination=topic,
                        metadata=data.get('metadata', {}),
                        timestamp=data.get('timestamp', time.time())
                    )
                    handler(message)
                except Exception as e:
                    logger.error(f"Error handling MQTT message: {e}")
            
            await client.subscribe(topic)
            self.message_handlers[f"{connection_id}:{topic}"] = message_handler
            return True
        except Exception as e:
            logger.error(f"Error subscribing to MQTT topic: {e}")
            return False
    
    async def is_connected(self, connection_id: str) -> bool:
        """检查MQTT连接状态"""
        if connection_id not in self.clients:
            return False
        
        try:
            client = self.clients[connection_id]
            return client.is_connected()
        except Exception:
            return False


class WebSocketHandler(ProtocolHandler):
    """WebSocket协议处理器"""
    
    def __init__(self):
        self.connections: Dict[str, Any] = {}
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self._websockets_available = WEBSOCKETS_AVAILABLE
    
    async def connect(self, config: ConnectionConfig) -> Any:
        """建立WebSocket连接"""
        if not self._websockets_available:
            # 模拟实现
            connection_id = f"ws:{config.host}:{config.port}"
            self.connections[connection_id] = {"mock": True, "config": config}
            self.message_queues[connection_id] = asyncio.Queue()
            logger.warning("websockets not available, using mock WebSocket handler")
            return connection_id
        
        try:
            uri = f"{config.protocol.value}://{config.host}:{config.port}"
            
            async with websockets.connect(uri) as websocket:
                connection_id = f"ws:{config.host}:{config.port}"
                self.connections[connection_id] = websocket
                self.message_queues[connection_id] = asyncio.Queue()
                
                # 启动消息接收任务
                asyncio.create_task(self._receive_messages(connection_id, websocket))
                
                logger.info(f"WebSocket connection established to {config.host}:{config.port}")
                return connection_id
        except Exception as e:
            logger.error(f"Failed to establish WebSocket connection: {e}")
            raise
    
    async def disconnect(self, connection_id: str, config: ConnectionConfig):
        """断开WebSocket连接"""
        if connection_id in self.connections:
            try:
                if not self._websockets_available:
                    # 模拟实现
                    del self.connections[connection_id]
                else:
                    websocket = self.connections[connection_id]
                    await websocket.close()
                    del self.connections[connection_id]
                
                if connection_id in self.message_queues:
                    del self.message_queues[connection_id]
                logger.info(f"WebSocket connection closed: {connection_id}")
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")
    
    async def send_message(self, connection_id: str, message: Message, config: ConnectionConfig) -> bool:
        """发送WebSocket消息"""
        if connection_id not in self.connections:
            return False
        
        try:
            if not self._websockets_available:
                # 模拟实现
                logger.info(f"Mock WebSocket send: {message.content}")
                return True
            
            websocket = self.connections[connection_id]
            serialized_message = MessageSerializer.serialize(message, "json")
            await websocket.send(serialized_message)
            return True
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            return False
    
    async def receive_message(self, connection_id: str, config: ConnectionConfig) -> Optional[Message]:
        """接收WebSocket消息"""
        if connection_id not in self.message_queues:
            return None
        
        try:
            message_data = await asyncio.wait_for(
                self.message_queues[connection_id].get(), 
                timeout=0.1
            )
            return MessageSerializer.deserialize(message_data, "json")
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error receiving WebSocket message: {e}")
            return None
    
    async def _receive_messages(self, connection_id: str, websocket):
        """接收消息的协程"""
        if not self._websockets_available:
            return
        
        try:
            async for message_data in websocket:
                await self.message_queues[connection_id].put(message_data)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"Error in WebSocket message receiver: {e}")
    
    async def is_connected(self, connection_id: str) -> bool:
        """检查WebSocket连接状态"""
        if connection_id not in self.connections:
            return False
        
        if not self._websockets_available:
            # 模拟实现
            return True
        
        try:
            websocket = self.connections[connection_id]
            return websocket.open
        except Exception:
            return False


class ProtocolVersionManager:
    """协议版本管理器"""
    
    def __init__(self):
        self.versions: Dict[ProtocolType, List[ProtocolVersionInfo]] = {
            ProtocolType.TCP: [
                ProtocolVersionInfo(
                    version="1.0",
                    name="TCP Basic",
                    description="基础TCP协议",
                    supported_features={"reliable", "stream"},
                    release_date="2020-01-01"
                ),
                ProtocolVersionInfo(
                    version="2.0",
                    name="TCP Enhanced",
                    description="增强TCP协议，支持压缩和加密",
                    supported_features={"reliable", "stream", "compression", "encryption"},
                    backward_compatible=False,
                    release_date="2023-01-01"
                )
            ],
            ProtocolType.MQTT: [
                ProtocolVersionInfo(
                    version="3.1.1",
                    name="MQTT v3.1.1",
                    description="MQTT协议版本3.1.1",
                    supported_features={"publish", "subscribe", "retain", "qos"},
                    release_date="2015-10-29"
                ),
                ProtocolVersionInfo(
                    version="5.0",
                    name="MQTT v5.0",
                    description="MQTT协议版本5.0",
                    supported_features={"publish", "subscribe", "retain", "qos", "user_properties", "shared_subscriptions"},
                    backward_compatible=True,
                    release_date="2019-03-07"
                )
            ],
            ProtocolType.HTTP: [
                ProtocolVersionInfo(
                    version="1.1",
                    name="HTTP/1.1",
                    description="HTTP协议版本1.1",
                    supported_features={"request_response", "persistent", "chunked"},
                    release_date="1999-06-15"
                ),
                ProtocolVersionInfo(
                    version="2",
                    name="HTTP/2",
                    description="HTTP协议版本2",
                    supported_features={"request_response", "multiplexing", "compression", "server_push"},
                    backward_compatible=False,
                    release_date="2015-05-14"
                )
            ]
        }
        self.current_versions: Dict[ProtocolType, ProtocolVersion] = {}
    
    def get_supported_versions(self, protocol: ProtocolType) -> List[ProtocolVersionInfo]:
        """获取协议支持的版本"""
        return self.versions.get(protocol, [])
    
    def set_current_version(self, protocol: ProtocolType, version: ProtocolVersion):
        """设置当前使用的协议版本"""
        supported_versions = [v.version for v in self.get_supported_versions(protocol)]
        if version in supported_versions:
            self.current_versions[protocol] = version
            logger.info(f"Set {protocol.value} current version to {version}")
        else:
            raise ValueError(f"Unsupported {protocol.value} version: {version}")
    
    def get_current_version(self, protocol: ProtocolType) -> Optional[ProtocolVersion]:
        """获取当前协议版本"""
        return self.current_versions.get(protocol)
    
    def is_version_compatible(self, protocol: ProtocolType, version: ProtocolVersion) -> bool:
        """检查版本兼容性"""
        version_info = next(
            (v for v in self.get_supported_versions(protocol) if v.version == version),
            None
        )
        return version_info.backward_compatible if version_info else False


class CommunicationProtocolHandler:
    """
    通信协议处理器主类
    
    这是一个综合性的通信协议处理器，支持多种网络协议和通信模式。
    主要功能包括：
    - 多种协议支持（TCP/UDP、MQTT、HTTP/HTTPS、WebSocket）
    - 连接池管理
    - 消息序列化/反序列化
    - 协议安全加密
    - 通信质量监控
    - 协议版本管理
    """
    
    def __init__(self, max_connections: int = 100, enable_monitoring: bool = True):
        """
        初始化通信协议处理器
        
        Args:
            max_connections: 最大连接数
            enable_monitoring: 是否启用通信质量监控
        """
        # 初始化各个组件
        self.connection_pool = ConnectionPool(max_connections)
        self.security_manager = SecurityManager()
        self.serializer = MessageSerializer()
        self.version_manager = ProtocolVersionManager()
        self.quality_monitor = QualityMonitor() if enable_monitoring else None
        
        # 初始化协议处理器
        self.protocol_handlers: Dict[ProtocolType, ProtocolHandler] = {
            ProtocolType.TCP: TCPHandler(),
            ProtocolType.UDP: UDPHandler(),
            ProtocolType.HTTP: HTTPHandler(),
            ProtocolType.HTTPS: HTTPHandler(),
            ProtocolType.MQTT: MQTTHandler(),
            ProtocolType.WEBSOCKET: WebSocketHandler()
        }
        
        # 连接状态跟踪
        self.active_connections: Dict[str, Tuple[Any, ConnectionConfig]] = {}
        self.message_handlers: Dict[str, Callable] = {}
        
        # 设置默认协议版本
        self.version_manager.set_current_version(ProtocolType.TCP, "1.0")
        self.version_manager.set_current_version(ProtocolType.MQTT, "5.0")
        self.version_manager.set_current_version(ProtocolType.HTTP, "1.1")
        
        # 启动监控
        if self.quality_monitor:
            asyncio.create_task(self.quality_monitor.start_monitoring())
        
        logger.info("Communication Protocol Handler initialized")
    
    async def connect(self, config: ConnectionConfig) -> str:
        """
        建立连接
        
        Args:
            config: 连接配置
            
        Returns:
            连接ID
        """
        connection_id = f"{config.protocol.value}_{uuid.uuid4().hex[:8]}"
        
        try:
            # 获取连接池中的连接
            connection = await self.connection_pool.get_connection(config)
            
            # 使用具体的协议处理器建立连接
            handler = self.protocol_handlers[config.protocol]
            protocol_connection = await handler.connect(config)
            
            # 保存连接信息
            self.active_connections[connection_id] = (protocol_connection, config)
            
            logger.info(f"Connection established: {connection_id} -> {config.host}:{config.port}")
            return connection_id
            
        except Exception as e:
            logger.error(f"Failed to establish connection: {e}")
            raise
    
    async def disconnect(self, connection_id: str):
        """
        断开连接
        
        Args:
            connection_id: 连接ID
        """
        if connection_id not in self.active_connections:
            logger.warning(f"Connection not found: {connection_id}")
            return
        
        try:
            protocol_connection, config = self.active_connections[connection_id]
            handler = self.protocol_handlers[config.protocol]
            
            await handler.disconnect(protocol_connection, config)
            await self.connection_pool.return_connection(protocol_connection, config)
            
            del self.active_connections[connection_id]
            logger.info(f"Connection closed: {connection_id}")
            
        except Exception as e:
            logger.error(f"Error closing connection {connection_id}: {e}")
    
    async def send_message(
        self, 
        connection_id: str, 
        message: Message,
        encrypt: bool = False,
        encryption_key: Optional[str] = None
    ) -> bool:
        """
        发送消息
        
        Args:
            connection_id: 连接ID
            message: 要发送的消息
            encrypt: 是否加密消息
            encryption_key: 加密密钥
            
        Returns:
            发送是否成功
        """
        if connection_id not in self.active_connections:
            logger.error(f"Connection not found: {connection_id}")
            return False
        
        try:
            protocol_connection, config = self.active_connections[connection_id]
            handler = self.protocol_handlers[config.protocol]
            
            # 消息加密
            if encrypt and encryption_key:
                encrypted_content = self.security_manager.encrypt_message(message, encryption_key)
                message.content = base64.b64encode(encrypted_content).decode()
                message.encrypted = True
            
            # 记录发送开始时间
            start_time = time.time()
            
            # 发送消息
            success = await handler.send_message(protocol_connection, message, config)
            
            # 记录延迟
            if self.quality_monitor and success:
                latency_ms = (time.time() - start_time) * 1000
                await self.quality_monitor.record_latency(connection_id, latency_ms)
            
            # 更新连接指标
            if success:
                metrics = self.connection_pool.get_metrics().get(connection_id)
                if metrics:
                    message_size = len(str(message.content).encode())
                    metrics.update_activity(message_size, True)
            
            return success
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            if self.quality_monitor:
                await self.quality_monitor.record_error(
                    connection_id, 
                    "SEND_ERROR", 
                    str(e)
                )
            return False
    
    async def receive_message(self, connection_id: str) -> Optional[Message]:
        """
        接收消息
        
        Args:
            connection_id: 连接ID
            
        Returns:
            接收到的消息，如果无消息则返回None
        """
        if connection_id not in self.active_connections:
            logger.error(f"Connection not found: {connection_id}")
            return None
        
        try:
            protocol_connection, config = self.active_connections[connection_id]
            handler = self.protocol_handlers[config.protocol]
            
            message = await handler.receive_message(protocol_connection, config)
            
            # 更新连接指标
            if message and self.quality_monitor:
                metrics = self.connection_pool.get_metrics().get(connection_id)
                if metrics:
                    message_size = len(str(message.content).encode())
                    metrics.update_activity(message_size, False)
            
            return message
            
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            if self.quality_monitor:
                await self.quality_monitor.record_error(
                    connection_id, 
                    "RECEIVE_ERROR", 
                    str(e)
                )
            return None
    
    async def broadcast_message(
        self, 
        connections: List[str], 
        message: Message,
        encrypt: bool = False,
        encryption_key: Optional[str] = None
    ) -> Dict[str, bool]:
        """
        广播消息到多个连接
        
        Args:
            connections: 连接ID列表
            message: 要广播的消息
            encrypt: 是否加密消息
            encryption_key: 加密密钥
            
        Returns:
            各连接发送结果字典
        """
        results = {}
        
        # 并发发送消息
        tasks = []
        for connection_id in connections:
            task = asyncio.create_task(
                self.send_message(connection_id, message, encrypt, encryption_key)
            )
            tasks.append((connection_id, task))
        
        # 收集结果
        for connection_id, task in tasks:
            try:
                results[connection_id] = await task
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                results[connection_id] = False
        
        return results
    
    async def subscribe_to_messages(
        self, 
        connection_id: str, 
        handler: Callable[[Message], None]
    ) -> bool:
        """
        订阅消息（主要用于MQTT等发布/订阅协议）
        
        Args:
            connection_id: 连接ID
            handler: 消息处理函数
            
        Returns:
            订阅是否成功
        """
        if connection_id not in self.active_connections:
            return False
        
        try:
            protocol_connection, config = self.active_connections[connection_id]
            
            # MQTT订阅
            if config.protocol == ProtocolType.MQTT and isinstance(
                self.protocol_handlers[config.protocol], MQTTHandler
            ):
                mqtt_handler = self.protocol_handlers[config.protocol]
                topic = "default/topic"  # 可以从配置或参数获取
                return await mqtt_handler.subscribe(connection_id, topic, handler)
            
            # WebSocket消息处理
            elif config.protocol == ProtocolType.WEBSOCKET:
                self.message_handlers[connection_id] = handler
                return True
            
            else:
                logger.warning(f"Subscription not supported for protocol: {config.protocol}")
                return False
                
        except Exception as e:
            logger.error(f"Error subscribing to messages: {e}")
            return False
    
    async def get_connection_status(self, connection_id: str) -> Dict[str, Any]:
        """
        获取连接状态
        
        Args:
            connection_id: 连接ID
            
        Returns:
            连接状态信息
        """
        if connection_id not in self.active_connections:
            return {"status": "not_found"}
        
        try:
            protocol_connection, config = self.active_connections[connection_id]
            handler = self.protocol_handlers[config.protocol]
            
            is_connected = await handler.is_connected(protocol_connection)
            
            # 获取连接指标
            metrics = self.connection_pool.get_metrics().get(connection_id)
            quality_stats = self.quality_monitor.get_quality_stats(connection_id) if self.quality_monitor else {}
            
            return {
                "status": "connected" if is_connected else "disconnected",
                "protocol": config.protocol.value,
                "host": config.host,
                "port": config.port,
                "connected_at": metrics.connected_at if metrics else None,
                "bytes_sent": metrics.bytes_sent if metrics else 0,
                "bytes_received": metrics.bytes_received if metrics else 0,
                "messages_sent": metrics.messages_sent if metrics else 0,
                "messages_received": metrics.messages_received if metrics else 0,
                "latency_ms": quality_stats.get('avg_latency_ms', 0),
                "throughput_kbps": quality_stats.get('avg_throughput_kbps', 0),
                "error_count": quality_stats.get('error_count', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting connection status: {e}")
            return {"status": "error", "error": str(e)}
    
    def get_supported_protocols(self) -> List[ProtocolType]:
        """获取支持的协议列表"""
        return list(self.protocol_handlers.keys())
    
    def get_protocol_versions(self, protocol: ProtocolType) -> List[ProtocolVersionInfo]:
        """获取协议支持的版本信息"""
        return self.version_manager.get_supported_versions(protocol)
    
    def set_protocol_version(self, protocol: ProtocolType, version: ProtocolVersion):
        """设置协议版本"""
        self.version_manager.set_current_version(protocol, version)
    
    def get_quality_metrics(self) -> Dict[str, Any]:
        """获取通信质量指标"""
        if not self.quality_monitor:
            return {"error": "Quality monitoring is disabled"}
        
        metrics = {}
        connection_metrics = self.connection_pool.get_metrics()
        
        for connection_id, conn_metrics in connection_metrics.items():
            quality_stats = self.quality_monitor.get_quality_stats(connection_id)
            metrics[connection_id] = {
                "connection": {
                    "protocol": conn_metrics.protocol.value,
                    "connected_at": conn_metrics.connected_at,
                    "bytes_sent": conn_metrics.bytes_sent,
                    "bytes_received": conn_metrics.bytes_received,
                    "messages_sent": conn_metrics.messages_sent,
                    "messages_received": conn_metrics.messages_received,
                    "throughput_kbps": conn_metrics.throughput_kbps
                },
                "quality": quality_stats
            }
        
        return metrics
    
    def get_alerts(self, acknowledged: Optional[bool] = None) -> List[Dict[str, Any]]:
        """获取质量告警"""
        if not self.quality_monitor:
            return []
        return self.quality_monitor.get_alerts(acknowledged)
    
    async def health_check(self) -> Dict[str, Any]:
        """
        健康检查
        
        Returns:
            系统健康状态
        """
        try:
            total_connections = len(self.active_connections)
            active_connections = 0
            protocols_in_use = set()
            
            # 检查所有连接状态
            for connection_id in list(self.active_connections.keys()):
                status = await self.get_connection_status(connection_id)
                if status.get("status") == "connected":
                    active_connections += 1
                protocols_in_use.add(status.get("protocol", "unknown"))
            
            # 获取质量告警
            alerts = self.get_alerts(acknowledged=False)
            critical_alerts = [alert for alert in alerts if alert.get("type", "").startswith("HIGH_")]
            
            return {
                "status": "healthy" if len(critical_alerts) == 0 else "degraded",
                "total_connections": total_connections,
                "active_connections": active_connections,
                "protocols_in_use": list(protocols_in_use),
                "alerts_count": len(alerts),
                "critical_alerts_count": len(critical_alerts),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def cleanup(self):
        """清理资源"""
        logger.info("Starting cleanup of Communication Protocol Handler...")
        
        # 停止监控
        if self.quality_monitor:
            await self.quality_monitor.stop_monitoring()
        
        # 关闭所有连接
        connection_ids = list(self.active_connections.keys())
        for connection_id in connection_ids:
            await self.disconnect(connection_id)
        
        # 关闭连接池
        await self.connection_pool.close_all()
        
        logger.info("Communication Protocol Handler cleanup completed")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await self.cleanup()


# 测试用例
class TestCommunicationProtocolHandler:
    """通信协议处理器测试类"""
    
    @staticmethod
    async def test_tcp_connection():
        """测试TCP连接"""
        print("\n=== 测试TCP连接 ===")
        
        config = ConnectionConfig(
            host="httpbin.org",
            port=80,
            protocol=ProtocolType.TCP
        )
        
        async with CommunicationProtocolHandler() as handler:
            # 建立连接
            connection_id = await handler.connect(config)
            print(f"TCP连接建立: {connection_id}")
            
            # 发送测试消息
            message = Message(
                content="Hello TCP!",
                source="test_client",
                destination="httpbin.org"
            )
            
            success = await handler.send_message(connection_id, message)
            print(f"消息发送结果: {success}")
            
            # 获取连接状态
            status = await handler.get_connection_status(connection_id)
            print(f"连接状态: {status}")
            
            # 清理
            await handler.disconnect(connection_id)
            print("TCP连接已关闭")
    
    @staticmethod
    async def test_http_connection():
        """测试HTTP连接"""
        print("\n=== 测试HTTP连接 ===")
        
        config = ConnectionConfig(
            host="httpbin.org",
            port=80,
            protocol=ProtocolType.HTTP,
            headers={"User-Agent": "CommunicationProtocolHandler/1.0"}
        )
        
        async with CommunicationProtocolHandler() as handler:
            # 建立连接
            connection_id = await handler.connect(config)
            print(f"HTTP连接建立: {connection_id}")
            
            # 发送测试消息
            message = Message(
                content={"test": "data", "message": "Hello HTTP!"},
                source="test_client"
            )
            
            success = await handler.send_message(connection_id, message)
            print(f"HTTP请求发送结果: {success}")
            
            # 获取连接状态
            status = await handler.get_connection_status(connection_id)
            print(f"连接状态: {status}")
            
            # 清理
            await handler.disconnect(connection_id)
            print("HTTP连接已关闭")
    
    @staticmethod
    async def test_websocket_connection():
        """测试WebSocket连接"""
        print("\n=== 测试WebSocket连接 ===")
        
        # 使用公共WebSocket测试服务
        config = ConnectionConfig(
            host="echo.websocket.org",
            port=80,
            protocol=ProtocolType.WEBSOCKET
        )
        
        async with CommunicationProtocolHandler() as handler:
            try:
                # 建立连接
                connection_id = await handler.connect(config)
                print(f"WebSocket连接建立: {connection_id}")
                
                # 发送测试消息
                message = Message(
                    content="Hello WebSocket!",
                    source="test_client"
                )
                
                success = await handler.send_message(connection_id, message)
                print(f"WebSocket消息发送结果: {success}")
                
                # 接收消息
                received_message = await handler.receive_message(connection_id)
                if received_message:
                    print(f"收到消息: {received_message.content}")
                
                # 获取连接状态
                status = await handler.get_connection_status(connection_id)
                print(f"连接状态: {status}")
                
            except Exception as e:
                print(f"WebSocket测试错误: {e}")
    
    @staticmethod
    async def test_message_serialization():
        """测试消息序列化"""
        print("\n=== 测试消息序列化 ===")
        
        # 创建测试消息
        original_message = Message(
            content={"test": "data", "numbers": [1, 2, 3]},
            protocol=ProtocolType.TCP,
            priority=MessagePriority.HIGH,
            source="test_sender",
            destination="test_receiver",
            metadata={"version": "1.0", "encoding": "utf-8"}
        )
        
        # 测试JSON序列化
        serialized_json = MessageSerializer.serialize(original_message, "json")
        print(f"JSON序列化长度: {len(serialized_json)} bytes")
        
        deserialized_json = MessageSerializer.deserialize(serialized_json, "json")
        print(f"JSON反序列化内容: {deserialized_json.content}")
        
        # 测试二进制序列化
        serialized_binary = MessageSerializer.serialize(original_message, "binary")
        print(f"二进制序列化长度: {len(serialized_binary)} bytes")
        
        deserialized_binary = MessageSerializer.deserialize(serialized_binary, "binary")
        print(f"二进制反序列化内容: {deserialized_binary.content}")
        
        # 验证数据一致性
        assert original_message.content == deserialized_json.content
        assert original_message.content == deserialized_binary.content
        print("消息序列化测试通过!")
    
    @staticmethod
    async def test_encryption():
        """测试加密功能"""
        print("\n=== 测试加密功能 ===")
        
        # 创建测试消息
        message = Message(
            content="This is a secret message!",
            source="sender",
            destination="receiver"
        )
        
        security_manager = SecurityManager()
        encryption_key = "test_encryption_key_123"
        
        # 加密消息
        encrypted_data = security_manager.encrypt_message(message, encryption_key)
        print(f"加密后数据长度: {len(encrypted_data)} bytes")
        
        # 解密消息
        decrypted_message = security_manager.decrypt_message(encrypted_data, encryption_key)
        print(f"解密后内容: {decrypted_message.content}")
        
        # 验证数据一致性
        assert message.content == decrypted_message.content
        print("加密测试通过!")
        
        # 测试签名
        signature = security_manager.create_signature(message, "secret_key")
        print(f"消息签名: {signature}")
        
        # 验证签名
        is_valid = security_manager.verify_signature(message, signature, "secret_key")
        print(f"签名验证结果: {is_valid}")
        
        assert is_valid
        print("签名测试通过!")
    
    @staticmethod
    async def test_quality_monitoring():
        """测试通信质量监控"""
        print("\n=== 测试通信质量监控 ===")
        
        async with CommunicationProtocolHandler(enable_monitoring=True) as handler:
            # 创建模拟连接配置
            config = ConnectionConfig(
                host="localhost",
                port=8080,
                protocol=ProtocolType.TCP
            )
            
            # 记录模拟数据
            if handler.quality_monitor:
                await handler.quality_monitor.record_latency("test_connection", 150.5)
                await handler.quality_monitor.record_latency("test_connection", 200.3)
                await handler.quality_monitor.record_latency("test_connection", 100.1)
                
                await handler.quality_monitor.record_throughput("test_connection", 1024.5)
                await handler.quality_monitor.record_throughput("test_connection", 2048.2)
                
                await handler.quality_monitor.record_error("test_connection", "CONNECTION_TIMEOUT", "Connection timeout occurred")
                
                # 获取质量统计
                quality_stats = handler.quality_monitor.get_quality_stats("test_connection")
                print(f"质量统计: {quality_stats}")
                
                # 获取告警
                alerts = handler.get_alerts()
                print(f"当前告警: {alerts}")
                
                print("质量监控测试完成!")
    
    @staticmethod
    async def test_connection_pool():
        """测试连接池管理"""
        print("\n=== 测试连接池管理 ===")
        
        config = ConnectionConfig(
            host="httpbin.org",
            port=80,
            protocol=ProtocolType.HTTP
        )
        
        async with CommunicationProtocolHandler(max_connections=5) as handler:
            # 建立多个连接
            connections = []
            for i in range(3):
                connection_id = await handler.connect(config)
                connections.append(connection_id)
                print(f"建立连接 {i+1}: {connection_id}")
            
            # 获取连接池指标
            pool_metrics = handler.connection_pool.get_metrics()
            print(f"连接池指标: {pool_metrics}")
            
            # 关闭连接
            for connection_id in connections:
                await handler.disconnect(connection_id)
                print(f"关闭连接: {connection_id}")
            
            print("连接池测试完成!")
    
    @staticmethod
    async def test_protocol_versions():
        """测试协议版本管理"""
        print("\n=== 测试协议版本管理 ===")
        
        handler = CommunicationProtocolHandler()
        
        # 获取支持的协议
        protocols = handler.get_supported_protocols()
        print(f"支持的协议: {[p.value for p in protocols]}")
        
        # 获取协议版本信息
        tcp_versions = handler.get_protocol_versions(ProtocolType.TCP)
        print(f"TCP协议版本: {[v.version for v in tcp_versions]}")
        
        mqtt_versions = handler.get_protocol_versions(ProtocolType.MQTT)
        print(f"MQTT协议版本: {[v.version for v in mqtt_versions]}")
        
        # 设置协议版本
        handler.set_protocol_version(ProtocolType.TCP, "2.0")
        current_tcp_version = handler.version_manager.get_current_version(ProtocolType.TCP)
        print(f"当前TCP版本: {current_tcp_version}")
        
        # 检查版本兼容性
        is_compatible = handler.version_manager.is_version_compatible(ProtocolType.TCP, "1.0")
        print(f"TCP 1.0向后兼容性: {is_compatible}")
        
        print("协议版本管理测试完成!")
    
    @staticmethod
    async def test_health_check():
        """测试健康检查"""
        print("\n=== 测试健康检查 ===")
        
        async with CommunicationProtocolHandler(enable_monitoring=True) as handler:
            # 执行健康检查
            health_status = await handler.health_check()
            print(f"健康检查结果: {health_status}")
            
            # 建立一个测试连接
            config = ConnectionConfig(
                host="httpbin.org",
                port=80,
                protocol=ProtocolType.HTTP
            )
            
            try:
                connection_id = await handler.connect(config)
                
                # 再次检查健康状态
                health_status = await handler.health_check()
                print(f"有连接时的健康检查: {health_status}")
                
                await handler.disconnect(connection_id)
                
            except Exception as e:
                print(f"连接测试错误: {e}")
            
            # 最终健康检查
            health_status = await handler.health_check()
            print(f"最终健康检查: {health_status}")
    
    @staticmethod
    async def run_all_tests():
        """运行所有测试"""
        print("开始运行通信协议处理器测试套件...")
        
        test_methods = [
            TestCommunicationProtocolHandler.test_message_serialization,
            TestCommunicationProtocolHandler.test_encryption,
            TestCommunicationProtocolHandler.test_quality_monitoring,
            TestCommunicationProtocolHandler.test_connection_pool,
            TestCommunicationProtocolHandler.test_protocol_versions,
            TestCommunicationProtocolHandler.test_health_check,
            TestCommunicationProtocolHandler.test_tcp_connection,
            TestCommunicationProtocolHandler.test_http_connection,
            TestCommunicationProtocolHandler.test_websocket_connection,
        ]
        
        for test_method in test_methods:
            try:
                await test_method()
            except Exception as e:
                print(f"测试 {test_method.__name__} 失败: {e}")
                continue
        
        print("\n所有测试完成!")


# 示例用法
async def example_usage():
    """示例用法"""
    print("=== 通信协议处理器示例用法 ===")
    
    # 创建处理器实例
    async with CommunicationProtocolHandler(max_connections=50, enable_monitoring=True) as handler:
        
        # 1. TCP连接示例
        print("\n1. TCP连接示例")
        tcp_config = ConnectionConfig(
            host="httpbin.org",
            port=80,
            protocol=ProtocolType.TCP,
            timeout=10.0
        )
        
        tcp_connection = await handler.connect(tcp_config)
        print(f"TCP连接建立: {tcp_connection}")
        
        # 发送消息
        tcp_message = Message(
            content="Hello from TCP client!",
            source="example_client",
            destination="httpbin.org"
        )
        
        success = await handler.send_message(tcp_connection, tcp_message)
        print(f"TCP消息发送: {'成功' if success else '失败'}")
        
        # 2. HTTP连接示例
        print("\n2. HTTP连接示例")
        http_config = ConnectionConfig(
            host="httpbin.org",
            port=80,
            protocol=ProtocolType.HTTP,
            headers={"User-Agent": "ExampleClient/1.0"}
        )
        
        http_connection = await handler.connect(http_config)
        print(f"HTTP连接建立: {http_connection}")
        
        # 发送HTTP请求
        http_message = Message(
            content={
                "method": "GET",
                "url": "/get",
                "headers": {"Accept": "application/json"}
            },
            source="example_client"
        )
        
        success = await handler.send_message(http_connection, http_message)
        print(f"HTTP请求发送: {'成功' if success else '失败'}")
        
        # 3. 加密消息示例
        print("\n3. 加密消息示例")
        secure_message = Message(
            content="This is a secret message!",
            source="secure_client",
            destination="secure_server"
        )
        
        encryption_key = "my_secret_key_123"
        success = await handler.send_message(
            tcp_connection, 
            secure_message, 
            encrypt=True, 
            encryption_key=encryption_key
        )
        print(f"加密消息发送: {'成功' if success else '失败'}")
        
        # 4. 广播消息示例
        print("\n4. 广播消息示例")
        broadcast_message = Message(
            content="Broadcast message to all connections!",
            source="broadcaster",
            priority=MessagePriority.HIGH
        )
        
        connections = [tcp_connection, http_connection]
        results = await handler.broadcast_message(connections, broadcast_message)
        print(f"广播结果: {results}")
        
        # 5. 连接状态监控
        print("\n5. 连接状态监控")
        for conn_id in connections:
            status = await handler.get_connection_status(conn_id)
            print(f"连接 {conn_id} 状态: {status['status']}")
        
        # 6. 通信质量监控
        print("\n6. 通信质量监控")
        quality_metrics = handler.get_quality_metrics()
        print(f"质量指标: {quality_metrics}")
        
        # 7. 告警监控
        print("\n7. 告警监控")
        alerts = handler.get_alerts()
        print(f"当前告警数量: {len(alerts)}")
        
        # 8. 健康检查
        print("\n8. 健康检查")
        health = await handler.health_check()
        print(f"系统健康状态: {health['status']}")
        
        # 清理连接
        print("\n9. 清理资源")
        for conn_id in connections:
            await handler.disconnect(conn_id)
            print(f"连接 {conn_id} 已关闭")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())
    
    # 运行测试
    print("\n" + "="*50)
    asyncio.run(TestCommunicationProtocolHandler.run_all_tests())