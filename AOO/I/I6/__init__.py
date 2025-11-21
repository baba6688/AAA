#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件接口处理器模块

该模块提供完整的事件驱动架构支持，包括事件发布/订阅、过滤、路由、
异步处理、持久化、重放、监控、版本管理和流处理等功能。

主要组件:
- EventInterfaceHandler: 主要的事件处理器类
- Event: 事件数据类
- EventType: 事件类型枚举
- EventPriority: 事件优先级枚举
- EventStatus: 事件状态枚举
- EventFilter: 事件过滤器
- EventRouter: 事件路由器
- EventPersistence: 事件持久化器
- EventReplay: 事件重放器
- EventMonitor: 事件监控器
- EventVersionManager: 事件版本管理器
- EventStreamProcessor: 事件流处理器

使用示例:
    from I.I6 import EventInterfaceHandler, EventType, EventPriority
    
    handler = EventInterfaceHandler()
    await handler.start()
    
    async def event_handler(event):
        print(f"收到事件: {event.id}")
    
    handler.create_subscriber("my_subscriber", event_handler)
    await handler.publish_system_event("my_service", {"message": "Hello"})
    
    await handler.stop()


版本: 1.0.0
创建时间: 2025-11-05
"""

from .EventInterfaceHandler import (
    # 主要类
    EventInterfaceHandler,
    
    # 事件相关类
    Event,
    EventType,
    EventPriority,
    EventStatus,
    
    # 组件类
    EventFilter,
    EventRouter,
    EventSubscriber,
    EventPublisher,
    EventPersistence,
    EventReplay,
    EventMonitor,
    EventVersionManager,
    EventStreamProcessor,
    
    # 测试类
    TestEventInterfaceHandler
)

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"

__all__ = [
    # 主要类
    "EventInterfaceHandler",
    
    # 事件相关
    "Event",
    "EventType", 
    "EventPriority",
    "EventStatus",
    
    # 组件
    "EventFilter",
    "EventRouter",
    "EventSubscriber",
    "EventPublisher",
    "EventPersistence",
    "EventReplay",
    "EventMonitor",
    "EventVersionManager",
    "EventStreamProcessor",
    
    # 测试
    "TestEventInterfaceHandler"
]

# 模块级配置
DEFAULT_CONFIG = {
    "db_path": "events.db",
    "max_workers": 4,
    "enable_persistence": True,
    "enable_monitoring": True,
    "enable_replay": True,
    "buffer_size": 1000,
    "cleanup_interval": 3600,  # 秒
    "event_retention_days": 30
}

# 事件类型映射
EVENT_TYPE_MAPPING = {
    "SYSTEM": EventType.SYSTEM,
    "BUSINESS": EventType.BUSINESS,
    "USER": EventType.USER,
    "ERROR": EventType.ERROR,
    "PERFORMANCE": EventType.PERFORMANCE,
    "SECURITY": EventType.SECURITY,
    "DATA": EventType.DATA
}

# 事件优先级映射
PRIORITY_MAPPING = {
    "LOW": EventPriority.LOW,
    "NORMAL": EventPriority.NORMAL,
    "HIGH": EventPriority.HIGH,
    "CRITICAL": EventPriority.CRITICAL
}


def create_handler(**kwargs):
    """
    创建事件处理器的便捷函数
    
    Args:
        **kwargs: 传递给EventInterfaceHandler的参数
        
    Returns:
        EventInterfaceHandler: 事件处理器实例
    """
    config = DEFAULT_CONFIG.copy()
    config.update(kwargs)
    return EventInterfaceHandler(**config)


def create_event(event_type: str, 
                priority: str,
                source: str,
                data: dict,
                **kwargs) -> Event:
    """
    创建事件的便捷函数
    
    Args:
        event_type: 事件类型字符串
        priority: 优先级字符串
        source: 事件源
        data: 事件数据
        **kwargs: 其他事件参数
        
    Returns:
        Event: 事件实例
    """
    from .EventInterfaceHandler import Event, datetime
    
    # 转换类型和优先级
    event_type_enum = EVENT_TYPE_MAPPING.get(event_type.upper(), EventType.SYSTEM)
    priority_enum = PRIORITY_MAPPING.get(priority.upper(), EventPriority.NORMAL)
    
    return Event(
        id=str(uuid.uuid4()),
        type=event_type_enum,
        priority=priority_enum,
        source=source,
        timestamp=datetime.now(),
        data=data,
        **kwargs
    )


# 导入uuid用于create_event函数
import uuid


def get_version():
    """获取模块版本"""
    return __version__


def get_config():
    """获取默认配置"""
    return DEFAULT_CONFIG.copy()


# 模块初始化日志
import logging
logger = logging.getLogger(__name__)
logger.info(f"事件接口处理器模块已加载，版本: {__version__}")