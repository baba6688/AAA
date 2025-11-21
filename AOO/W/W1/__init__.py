"""
W1网络连接管理器包

提供完整的网络连接管理解决方案，包括：
- 连接管理（TCP、UDP、HTTP连接管理）
- 连接池（连接池管理和复用）
- 连接监控（连接状态监控和健康检查）
- 连接重试（连接失败重试机制）
- 连接超时（连接超时设置和管理）
- 连接加密（SSL/TLS加密支持）
- 连接统计（连接使用统计和分析）
- 连接安全（连接安全检查和防护）
"""

from .NetworkConnectionManager import (
    NetworkConnectionManager,
    ConnectionPool,
    Connection,
    ConnectionConfig,
    ConnectionMonitor,
    ConnectionStats,
    ConnectionRetryManager,
    ConnectionSecurityManager,
    TCPConnection,
    UDPConnection,
    HTTPConnection,
    SSLConnection,
    ConnectionState,
    ConnectionType,
    SecurityError
)

__version__ = "1.0.0"
__author__ = "W1 Network Connection Manager Team"

__all__ = [
    # 核心类
    "NetworkConnectionManager",
    "ConnectionPool", 
    "Connection",
    "ConnectionConfig",
    
    # 监控和统计
    "ConnectionMonitor",
    "ConnectionStats",
    
    # 管理和安全
    "ConnectionRetryManager",
    "ConnectionSecurityManager",
    
    # 连接类型
    "TCPConnection",
    "UDPConnection", 
    "HTTPConnection",
    "SSLConnection",
    
    # 枚举类
    "ConnectionState",
    "ConnectionType",
    
    # 异常类
    "SecurityError"
]