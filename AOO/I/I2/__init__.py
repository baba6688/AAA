"""
I2模块 - Web接口控制器

该模块提供Web接口控制功能，包括：
- Web配置管理
- 连接信息管理
- 会话数据处理
- 限流配置
- 安全管理
- 会话管理
- 负载均衡
- Web接口控制

Author: AI Assistant
Date: 2025-11-05
Version: 1.0.0
"""

from .WebInterfaceController import (
    WebConfig,
    ConnectionInfo,
    SessionData,
    RateLimitEntry,
    SecurityManager,
    SessionManager,
    LoadBalancer,
    WebInterfaceController
)

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    "WebConfig",
    "ConnectionInfo",
    "SessionData", 
    "RateLimitEntry",
    "SecurityManager",
    "SessionManager",
    "LoadBalancer",
    "WebInterfaceController"
]