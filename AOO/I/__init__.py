"""
I区域 - 接口管理模块

该区域提供完整的接口管理功能，包括：
- I1: API接口管理器 - RESTful API管理、GraphQL支持、API版本管理
- I2: Web接口控制器 - Web配置管理、会话管理、负载均衡
- I3: 数据接口适配器 - 多种数据源支持、数据转换、缓存管理
- I4: 通信协议处理器 - TCP/UDP/HTTP/MQTT/WebSocket协议支持
- I5: 消息队列接口 - Redis/RabbitMQ/Kafka消息队列支持
- I6: 事件接口处理器 - 事件发布订阅、过滤路由、流处理
- I7: 回调接口管理器 - 异步回调、回调链、重试机制
- I8: 插件接口控制器 - 插件生命周期管理、沙盒环境
- I9: 接口状态聚合器 - 接口监控、告警管理、性能聚合

Author: MiniMax Agent
Date: 2025-11-13
Version: 1.0.0
"""

# 导入所有子模块
from . import I1  # API接口管理器
from . import I2  # Web接口控制器  
from . import I3  # 数据接口适配器
from . import I4  # 通信协议处理器
from . import I5  # 消息队列接口
from . import I6  # 事件接口处理器
from . import I7  # 回调接口管理器
from . import I8  # 插件接口控制器
from . import I9  # 接口状态聚合器

__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__description__ = "I区域 - 接口管理模块，提供完整的API、数据、消息、事件接口管理功能"