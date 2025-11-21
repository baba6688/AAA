"""
I5 模块 - 消息队列接口

该模块提供了统一的消息队列接口实现，支持多种消息队列后端：
- Redis
- RabbitMQ  
- Apache Kafka

主要组件：
- MessageQueueInterface: 主要的消息队列接口类
- Message: 消息类
- MessageMetadata: 消息元数据类
- 各种后端实现：RedisBackend, RabbitMQBackend, KafkaBackend
- 消息追踪器：MessageTracker
- 负载均衡器：LoadBalancer


创建时间: 2025-11-05
"""

from .MessageQueueInterface import (
    MessageQueueInterface,
    Message,
    MessageMetadata,
    MessagePriority,
    MessageStatus,
    QueueType,
    MessageQueueBackend,
    RedisBackend,
    RabbitMQBackend,
    KafkaBackend,
    MessageTracker,
    LoadBalancer,
    MockBackend
)

__version__ = "1.0.0"
__author__ = "AI系统"

__all__ = [
    "MessageQueueInterface",
    "Message", 
    "MessageMetadata",
    "MessagePriority",
    "MessageStatus",
    "QueueType",
    "MessageQueueBackend",
    "RedisBackend",
    "RabbitMQBackend", 
    "KafkaBackend",
    "MessageTracker",
    "LoadBalancer",
    "MockBackend"
]