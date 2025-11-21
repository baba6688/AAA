#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S4消息服务包
一个功能完整的消息服务系统，支持消息队列、发布订阅、消息路由等功能

主要功能:
- 消息队列管理
- 发布订阅模式
- 消息路由和转发
- 消息持久化存储
- 消息确认机制
- 消息重试机制
- 消息监控和统计

快速入门指南:
===========

1. 基本使用
-----------
    from S4 import MessageService, get_message_service
    
    # 获取全局消息服务实例
    service = get_message_service()
    service.start()
    
    # 定义消息处理函数
    def handle_message(message):
        print(f"收到消息: {message.payload}")
    
    # 订阅主题
    service.subscribe("test.topic", handle_message)
    
    # 发布消息
    message_id = service.publish("test.topic", {"message": "Hello, World!"})
    
    # 获取统计信息
    stats = service.get_stats()
    print(f"已发布 {stats['total_published']} 条消息")

2. 高级功能
-----------
    # 消息路由
    service.add_route("source.topic", "target.topic")
    
    # 消息确认
    service.confirm_message(message_id)
    
    # 获取消息历史
    history = service.get_message_history("test.topic", limit=50)
    
    # 清除队列
    service.clear_queue("test.topic")

3. 自定义配置
------------
    from S4 import init_message_service
    
    # 使用自定义配置初始化
    service = init_message_service(db_path="custom_messages.db")
    service.start()

作者: S4消息服务团队
版本: 1.0.0
文档: https://github.com/your-org/s4-messages
"""

import logging
from typing import Dict, Any, Optional, Callable, List

# 导入核心类
from .MessageService import (
    MessageService,
    Message,
    MessageStatus,
    MessagePersistence,
    get_message_service,
    init_message_service
)

# 版本信息
__version__ = "1.0.0"
__author__ = "S4 Message Service Team"
__description__ = "一个功能完整的消息服务系统"
__email__ = "support@s4-messages.com"
__license__ = "MIT"

# 导出主要类和函数
__all__ = [
    "MessageService",
    "Message", 
    "MessageStatus",
    "MessagePersistence",
    "get_message_service",
    "init_message_service",
    "get_version",
    "get_config",
    "set_config",
    "create_service",
    "quick_publish",
    "quick_subscribe",
    "ServiceConfig",
    "DEFAULT_CONFIG",
    "DEFAULT_DB_PATH",
    "SUPPORTED_STATUSES",
    "validate_message",
    "format_message_stats"
]

# 默认配置常量
DEFAULT_CONFIG = {
    "db_path": "message_service.db",
    "worker_threads": 3,
    "retry_interval": 5,
    "max_retries": 3,
    "queue_timeout": 30,
    "enable_persistence": True,
    "enable_routing": True,
    "enable_monitoring": True
}

# 默认数据库路径
DEFAULT_DB_PATH = "message_service.db"

# 支持的消息状态
SUPPORTED_STATUSES = {
    "pending": MessageStatus.PENDING,
    "published": MessageStatus.PUBLISHED,
    "delivered": MessageStatus.DELIVERED,
    "confirmed": MessageStatus.CONFIRMED,
    "failed": MessageStatus.FAILED,
    "retrying": MessageStatus.RETRYING
}

# 服务配置类
class ServiceConfig:
    """服务配置类"""
    
    def __init__(self, **kwargs):
        self.config = DEFAULT_CONFIG.copy()
        self.config.update(kwargs)
    
    def get(self, key: str, default=None):
        """获取配置值"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """设置配置值"""
        self.config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return self.config.copy()
    
    def __repr__(self):
        return f"ServiceConfig({self.config})"

# 全局默认配置
_DEFAULT_CONFIG = DEFAULT_CONFIG.copy()
_global_service = None

def get_version() -> str:
    """获取版本信息"""
    return __version__

def get_config() -> Dict[str, Any]:
    """获取默认配置"""
    return _DEFAULT_CONFIG.copy()

def set_config(**kwargs):
    """更新默认配置"""
    global _DEFAULT_CONFIG
    _DEFAULT_CONFIG.update(kwargs)
    logging.info(f"配置已更新: {kwargs}")

def create_service(**config_kwargs) -> MessageService:
    """
    创建消息服务实例
    
    Args:
        **config_kwargs: 配置参数
        
    Returns:
        MessageService: 消息服务实例
    """
    if config_kwargs:
        config = DEFAULT_CONFIG.copy()
        config.update(config_kwargs)
        return MessageService(db_path=config['db_path'])
    else:
        return MessageService()

def get_global_service() -> MessageService:
    """获取全局消息服务实例（别名）"""
    return get_message_service()

def quick_publish(topic: str, payload: Any, metadata: Dict[str, Any] = None) -> str:
    """
    快速发布消息
    
    Args:
        topic: 主题名称
        payload: 消息内容
        metadata: 元数据
        
    Returns:
        str: 消息ID
    """
    service = get_message_service()
    return service.publish(topic, payload, metadata)

def quick_subscribe(topic: str, callback: Callable[[Message], None]):
    """
    快速订阅消息
    
    Args:
        topic: 主题名称
        callback: 回调函数
    """
    service = get_message_service()
    service.subscribe(topic, callback)

def validate_message(message_data: Dict[str, Any]) -> bool:
    """
    验证消息数据格式
    
    Args:
        message_data: 消息数据
        
    Returns:
        bool: 是否有效
    """
    required_fields = ['id', 'topic', 'payload', 'timestamp', 'status']
    
    if not isinstance(message_data, dict):
        return False
    
    for field in required_fields:
        if field not in message_data:
            return False
    
    # 验证状态字段
    status = message_data.get('status')
    if isinstance(status, str):
        return status in [s.value for s in MessageStatus]
    elif isinstance(status, MessageStatus):
        return True
    
    return False

def format_message_stats(stats: Dict[str, Any]) -> str:
    """
    格式化消息统计信息
    
    Args:
        stats: 统计信息字典
        
    Returns:
        str: 格式化的统计字符串
    """
    if not stats:
        return "暂无统计数据"
    
    lines = []
    lines.append("=== 消息服务统计 ===")
    lines.append(f"已发布消息: {stats.get('total_published', 0)}")
    lines.append(f"已送达消息: {stats.get('total_delivered', 0)}")
    lines.append(f"失败消息: {stats.get('total_failed', 0)}")
    lines.append(f"重试消息: {stats.get('total_retried', 0)}")
    lines.append(f"平均处理时间: {stats.get('avg_processing_time', 0):.3f}秒")
    
    # 队列统计
    queue_sizes = stats.get('queue_sizes', {})
    if queue_sizes:
        lines.append("\n=== 队列统计 ===")
        for topic, size in queue_sizes.items():
            lines.append(f"{topic}: {size} 条消息")
    
    # 订阅者统计
    subscribers = stats.get('active_subscribers', {})
    if subscribers:
        lines.append("\n=== 活跃订阅者 ===")
        for topic, count in subscribers.items():
            lines.append(f"{topic}: {count} 个订阅者")
    
    return "\n".join(lines)

def get_service_info() -> Dict[str, Any]:
    """
    获取服务信息
    
    Returns:
        Dict[str, Any]: 服务信息
    """
    service = get_message_service()
    
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "license": __license__,
        "is_running": service.is_running if hasattr(service, 'is_running') else False,
        "config": get_config(),
        "stats": service.get_stats() if hasattr(service, 'get_stats') else {}
    }

def health_check() -> Dict[str, Any]:
    """
    健康检查
    
    Returns:
        Dict[str, Any]: 健康状态信息
    """
    service = get_message_service()
    
    try:
        # 检查服务状态
        is_running = service.is_running if hasattr(service, 'is_running') else False
        
        # 检查队列
        queue_sizes = service.get_queue_size() if hasattr(service, 'get_queue_size') else {}
        
        # 检查数据库连接
        db_ok = True
        if hasattr(service, 'persistence') and hasattr(service.persistence, 'load_message'):
            # 尝试加载一个不存在的消息以测试连接
            test_msg = service.persistence.load_message("health_check_test")
        
        # 计算健康状态
        queue_errors = sum(1 for size in queue_sizes.values() if size < 0)
        overall_health = "healthy" if is_running and db_ok and queue_errors == 0 else "unhealthy"
        
        return {
            "status": overall_health,
            "is_running": is_running,
            "database_ok": db_ok,
            "queue_errors": queue_errors,
            "queue_sizes": queue_sizes,
            "timestamp": __import__('time').time()
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "timestamp": __import__('time').time()
        }

# 日志配置
def setup_logging(level: str = "INFO", format_str: str = None):
    """
    设置日志配置
    
    Args:
        level: 日志级别
        format_str: 日志格式字符串
    """
    if format_str is None:
        format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=format_str
    )

# 便利函数：获取消息状态显示名称
def get_status_display_name(status: MessageStatus) -> str:
    """
    获取消息状态显示名称
    
    Args:
        status: 消息状态
        
    Returns:
        str: 显示名称
    """
    display_names = {
        MessageStatus.PENDING: "待处理",
        MessageStatus.PUBLISHED: "已发布", 
        MessageStatus.DELIVERED: "已送达",
        MessageStatus.CONFIRMED: "已确认",
        MessageStatus.FAILED: "失败",
        MessageStatus.RETRYING: "重试中"
    }
    return display_names.get(status, str(status))

# 包初始化日志
logging.info(f"S4消息服务包已加载 (版本: {__version__})")