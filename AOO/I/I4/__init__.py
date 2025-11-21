"""
I4模块 - 通信协议处理器

该模块提供通信协议处理功能，包括：
- 多种协议类型支持(TCP, UDP, HTTP, MQTT, WebSocket)
- 连接状态管理
- 消息优先级处理
- 加密方法支持
- 连接配置管理
- 消息序列化
- 连接池管理
- 质量监控
- 版本管理

Author: AI Assistant
Date: 2025-11-05
Version: 1.0.0
"""

from .CommunicationProtocolHandler import (
    ProtocolType,
    ConnectionState,
    MessagePriority,
    EncryptionMethod,
    ConnectionConfig,
    Message,
    ConnectionMetrics,
    ProtocolVersionInfo,
    SecurityManager,
    MessageSerializer,
    ConnectionPool,
    QualityMonitor,
    ProtocolHandler,
    TCPHandler,
    UDPHandler,
    UDPProtocol,
    HTTPHandler,
    MQTTHandler,
    WebSocketHandler,
    ProtocolVersionManager
)

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    "ProtocolType",
    "ConnectionState",
    "MessagePriority", 
    "EncryptionMethod",
    "ConnectionConfig",
    "Message",
    "ConnectionMetrics",
    "ProtocolVersionInfo",
    "SecurityManager",
    "MessageSerializer",
    "ConnectionPool",
    "QualityMonitor",
    "ProtocolHandler",
    "TCPHandler",
    "UDPHandler",
    "UDPProtocol", 
    "HTTPHandler",
    "MQTTHandler",
    "WebSocketHandler",
    "ProtocolVersionManager"
]