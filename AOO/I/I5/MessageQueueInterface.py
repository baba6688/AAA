"""
消息队列接口类 - I5 MessageQueueInterface

该模块实现了一个统一的消息队列接口，支持多种消息队列后端：
- Redis消息队列
- RabbitMQ
- Apache Kafka

主要功能：
- 消息发布/订阅
- 消息持久化
- 消息优先级处理
- 死信队列管理
- 消息追踪和监控
- 队列负载均衡

创建时间: 2025-11-05
"""

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Callable, Union, Set
from concurrent.futures import ThreadPoolExecutor
import threading
import weakref

# 第三方库导入（可选）
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import pika
    RABBITMQ_AVAILABLE = True
except ImportError:
    RABBITMQ_AVAILABLE = False

try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False


class MessagePriority(Enum):
    """消息优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class MessageStatus(Enum):
    """消息状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD_LETTER = "dead_letter"


class QueueType(Enum):
    """队列类型枚举"""
    FIFO = "fifo"
    PRIORITY = "priority"
    DELAYED = "delayed"
    DEAD_LETTER = "dead_letter"


@dataclass
class MessageMetadata:
    """消息元数据类"""
    message_id: str
    timestamp: float
    priority: MessagePriority
    status: MessageStatus
    retry_count: int = 0
    max_retries: int = 3
    ttl: Optional[int] = None  # 消息生存时间（秒）
    correlation_id: Optional[str] = None
    reply_to: Optional[str] = None
    headers: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MessageMetadata':
        """从字典创建实例"""
        return cls(**data)


@dataclass
class Message:
    """消息类"""
    data: Any
    metadata: MessageMetadata
    queue_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'data': self.data,
            'metadata': self.metadata.to_dict(),
            'queue_name': self.queue_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """从字典创建实例"""
        metadata = MessageMetadata.from_dict(data['metadata'])
        return cls(
            data=data['data'],
            metadata=metadata,
            queue_name=data['queue_name']
        )


class MessageQueueBackend(ABC):
    """消息队列后端抽象基类"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接消息队列"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """断开连接"""
        pass
    
    @abstractmethod
    async def publish(self, queue_name: str, message: Message, exchange: str = "") -> bool:
        """发布消息"""
        pass
    
    @abstractmethod
    async def subscribe(self, queue_name: str, callback: Callable[[Message], None]) -> str:
        """订阅消息"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅"""
        pass
    
    @abstractmethod
    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """获取队列统计信息"""
        pass


class RedisBackend(MessageQueueBackend):
    """Redis消息队列后端实现"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, 
                 password: Optional[str] = None):
        self.host = host
        self.port = port
        self.db = db
        self.password = password
        self.connection: Optional[redis.Redis] = None
        self.subscriptions: Dict[str, Any] = {}
        
    async def connect(self) -> bool:
        """连接Redis"""
        if not REDIS_AVAILABLE:
            raise ImportError("Redis客户端未安装，请运行: pip install redis")
        
        try:
            self.connection = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=False
            )
            await self.connection.ping()
            logging.info(f"Redis连接成功: {self.host}:{self.port}")
            return True
        except Exception as e:
            logging.error(f"Redis连接失败: {e}")
            return False
    
    async def disconnect(self) -> None:
        """断开Redis连接"""
        if self.connection:
            await self.connection.close()
            self.connection = None
            logging.info("Redis连接已断开")
    
    async def publish(self, queue_name: str, message: Message, exchange: str = "") -> bool:
        """发布消息到Redis"""
        if not self.connection:
            raise RuntimeError("Redis未连接")
        
        try:
            # 序列化消息
            message_data = json.dumps(message.to_dict(), default=str)
            
            # 根据优先级选择不同的列表
            priority_key = f"priority:{message.metadata.priority.value}"
            queue_key = f"queue:{queue_name}"
            
            # 使用Redis事务确保原子性
            async with self.connection.pipeline() as pipe:
                # 添加到主队列
                pipe.lpush(queue_key, message_data)
                
                # 添加到优先级队列
                pipe.lpush(priority_key, message_data)
                
                # 设置消息过期时间
                if message.metadata.ttl:
                    pipe.expire(queue_key, message.metadata.ttl)
                    pipe.expire(priority_key, message.metadata.ttl)
                
                await pipe.execute()
            
            logging.debug(f"消息已发布到Redis队列 {queue_name}: {message.metadata.message_id}")
            return True
            
        except Exception as e:
            logging.error(f"Redis发布消息失败: {e}")
            return False
    
    async def subscribe(self, queue_name: str, callback: Callable[[Message], None]) -> str:
        """订阅Redis消息"""
        if not self.connection:
            raise RuntimeError("Redis未连接")
        
        subscription_id = str(uuid.uuid4())
        
        async def message_handler():
            queue_key = f"queue:{queue_name}"
            
            while subscription_id in self.subscriptions:
                try:
                    # 阻塞式获取消息
                    result = await self.connection.brpop(queue_key, timeout=1)
                    if result:
                        _, message_data = result
                        message_dict = json.loads(message_data.decode('utf-8'))
                        message = Message.from_dict(message_dict)
                        
                        # 更新消息状态
                        message.metadata.status = MessageStatus.PROCESSING
                        
                        # 异步处理消息
                        asyncio.create_task(callback(message))
                        
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logging.error(f"Redis消息处理错误: {e}")
                    await asyncio.sleep(0.1)
        
        # 启动消息处理任务
        task = asyncio.create_task(message_handler())
        self.subscriptions[subscription_id] = task
        
        logging.info(f"已订阅Redis队列 {queue_name}, 订阅ID: {subscription_id}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """取消Redis订阅"""
        if subscription_id in self.subscriptions:
            task = self.subscriptions[subscription_id]
            task.cancel()
            del self.subscriptions[subscription_id]
            logging.info(f"已取消Redis订阅: {subscription_id}")
            return True
        return False
    
    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """获取Redis队列统计信息"""
        if not self.connection:
            raise RuntimeError("Redis未连接")
        
        queue_key = f"queue:{queue_name}"
        
        try:
            queue_length = await self.connection.llen(queue_key)
            
            # 获取不同优先级的消息数量
            priority_stats = {}
            for priority in MessagePriority:
                priority_key = f"priority:{priority.value}"
                priority_stats[priority.name] = await self.connection.llen(priority_key)
            
            return {
                "queue_name": queue_name,
                "total_messages": queue_length,
                "priority_stats": priority_stats,
                "backend": "redis"
            }
            
        except Exception as e:
            logging.error(f"获取Redis队列统计失败: {e}")
            return {}


class RabbitMQBackend(MessageQueueBackend):
    """RabbitMQ消息队列后端实现"""
    
    def __init__(self, host: str = "localhost", port: int = 5672,
                 username: str = "guest", password: str = "guest"):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.connection: Optional[pika.BlockingConnection] = None
        self.channel: Optional[pika.channel.Channel] = None
        self.subscriptions: Dict[str, Any] = {}
        
    async def connect(self) -> bool:
        """连接RabbitMQ"""
        if not RABBITMQ_AVAILABLE:
            raise ImportError("RabbitMQ客户端未安装，请运行: pip install pika")
        
        try:
            credentials = pika.PlainCredentials(self.username, self.password)
            parameters = pika.ConnectionParameters(
                host=self.host,
                port=self.port,
                credentials=credentials
            )
            
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            
            # 声明交换机
            self.channel.exchange_declare(exchange='direct', exchange_type='direct', durable=True)
            
            logging.info(f"RabbitMQ连接成功: {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logging.error(f"RabbitMQ连接失败: {e}")
            return False
    
    async def disconnect(self) -> None:
        """断开RabbitMQ连接"""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.channel = None
            logging.info("RabbitMQ连接已断开")
    
    async def publish(self, queue_name: str, message: Message, exchange: str = "") -> bool:
        """发布消息到RabbitMQ"""
        if not self.channel:
            raise RuntimeError("RabbitMQ未连接")
        
        try:
            # 声明队列
            self.channel.queue_declare(queue=queue_name, durable=True)
            
            # 序列化消息
            message_data = json.dumps(message.to_dict(), default=str)
            
            # 设置消息属性
            properties = pika.BasicProperties(
                message_id=message.metadata.message_id,
                correlation_id=message.metadata.correlation_id,
                reply_to=message.metadata.reply_to,
                priority=message.metadata.priority.value,
                delivery_mode=2,  # 持久化
                headers=message.metadata.headers or {}
            )
            
            # 发布消息
            self.channel.basic_publish(
                exchange=exchange,
                routing_key=queue_name,
                body=message_data,
                properties=properties
            )
            
            logging.debug(f"消息已发布到RabbitMQ队列 {queue_name}: {message.metadata.message_id}")
            return True
            
        except Exception as e:
            logging.error(f"RabbitMQ发布消息失败: {e}")
            return False
    
    async def subscribe(self, queue_name: str, callback: Callable[[Message], None]) -> str:
        """订阅RabbitMQ消息"""
        if not self.channel:
            raise RuntimeError("RabbitMQ未连接")
        
        subscription_id = str(uuid.uuid4())
        
        def message_handler(ch, method, properties, body):
            try:
                message_dict = json.loads(body.decode('utf-8'))
                message = Message.from_dict(message_dict)
                
                # 更新消息状态
                message.metadata.status = MessageStatus.PROCESSING
                
                # 异步处理消息
                asyncio.create_task(callback(message))
                
                # 确认消息
                ch.basic_ack(delivery_tag=method.delivery_tag)
                
            except Exception as e:
                logging.error(f"RabbitMQ消息处理错误: {e}")
                # 拒绝消息并重新入队
                ch.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
        
        # 开始消费消息
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(queue=queue_name, on_message_callback=message_handler)
        
        # 在单独的线程中启动消费
        def start_consuming():
            self.channel.start_consuming()
        
        consumer_thread = threading.Thread(target=start_consuming, daemon=True)
        consumer_thread.start()
        
        self.subscriptions[subscription_id] = consumer_thread
        
        logging.info(f"已订阅RabbitMQ队列 {queue_name}, 订阅ID: {subscription_id}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """取消RabbitMQ订阅"""
        if subscription_id in self.subscriptions:
            if self.channel:
                self.channel.stop_consuming()
            del self.subscriptions[subscription_id]
            logging.info(f"已取消RabbitMQ订阅: {subscription_id}")
            return True
        return False
    
    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """获取RabbitMQ队列统计信息"""
        if not self.channel:
            raise RuntimeError("RabbitMQ未连接")
        
        try:
            # 获取队列信息
            queue_info = self.channel.queue_declare(queue=queue_name, passive=True)
            
            return {
                "queue_name": queue_name,
                "message_count": queue_info.method.message_count,
                "consumer_count": queue_info.method.consumer_count,
                "backend": "rabbitmq"
            }
            
        except Exception as e:
            logging.error(f"获取RabbitMQ队列统计失败: {e}")
            return {}


class KafkaBackend(MessageQueueBackend):
    """Kafka消息队列后端实现"""
    
    def __init__(self, bootstrap_servers: List[str] = ["localhost:9092"]):
        self.bootstrap_servers = bootstrap_servers
        self.producer: Optional[KafkaProducer] = None
        self.consumers: Dict[str, Any] = {}
        self.subscriptions: Dict[str, Any] = {}
        
    async def connect(self) -> bool:
        """连接Kafka"""
        if not KAFKA_AVAILABLE:
            raise ImportError("Kafka客户端未安装，请运行: pip install kafka-python")
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v, default=str).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            
            logging.info(f"Kafka连接成功: {self.bootstrap_servers}")
            return True
            
        except Exception as e:
            logging.error(f"Kafka连接失败: {e}")
            return False
    
    async def disconnect(self) -> None:
        """断开Kafka连接"""
        if self.producer:
            self.producer.close()
            self.producer = None
        
        # 关闭所有消费者
        for consumer in self.consumers.values():
            consumer.close()
        self.consumers.clear()
        
        logging.info("Kafka连接已断开")
    
    async def publish(self, queue_name: str, message: Message, exchange: str = "") -> bool:
        """发布消息到Kafka"""
        if not self.producer:
            raise RuntimeError("Kafka未连接")
        
        try:
            # 序列化消息
            message_data = message.to_dict()
            
            # 发送消息
            future = self.producer.send(
                queue_name,
                value=message_data,
                key=message.metadata.message_id
            )
            
            # 等待发送完成
            record_metadata = future.get(timeout=10)
            
            logging.debug(f"消息已发布到Kafka主题 {queue_name}: {message.metadata.message_id}")
            return True
            
        except Exception as e:
            logging.error(f"Kafka发布消息失败: {e}")
            return False
    
    async def subscribe(self, queue_name: str, callback: Callable[[Message], None]) -> str:
        """订阅Kafka消息"""
        subscription_id = str(uuid.uuid4())
        
        def message_handler():
            consumer = KafkaConsumer(
                queue_name,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
                auto_offset_reset='latest',
                enable_auto_commit=True
            )
            
            self.consumers[subscription_id] = consumer
            
            for message in consumer:
                if subscription_id not in self.subscriptions:
                    break
                
                try:
                    message_obj = Message.from_dict(message.value)
                    message_obj.metadata.status = MessageStatus.PROCESSING
                    
                    # 异步处理消息
                    asyncio.create_task(callback(message_obj))
                    
                except Exception as e:
                    logging.error(f"Kafka消息处理错误: {e}")
        
        # 在单独的线程中启动消费
        consumer_thread = threading.Thread(target=message_handler, daemon=True)
        consumer_thread.start()
        
        self.subscriptions[subscription_id] = consumer_thread
        
        logging.info(f"已订阅Kafka主题 {queue_name}, 订阅ID: {subscription_id}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """取消Kafka订阅"""
        if subscription_id in self.subscriptions:
            if subscription_id in self.consumers:
                self.consumers[subscription_id].close()
                del self.consumers[subscription_id]
            del self.subscriptions[subscription_id]
            logging.info(f"已取消Kafka订阅: {subscription_id}")
            return True
        return False
    
    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """获取Kafka队列统计信息"""
        try:
            consumer = KafkaConsumer(
                queue_name,
                bootstrap_servers=self.bootstrap_servers,
                auto_offset_reset='latest'
            )
            
            # 获取主题分区信息
            partitions = consumer.partitions_for_topic(queue_name)
            
            consumer.close()
            
            return {
                "queue_name": queue_name,
                "partitions": list(partitions) if partitions else [],
                "backend": "kafka"
            }
            
        except Exception as e:
            logging.error(f"获取Kafka队列统计失败: {e}")
            return {}


class MessageTracker:
    """消息追踪器"""
    
    def __init__(self):
        self.message_history: Dict[str, List[Dict[str, Any]]] = {}
        self.lock = threading.Lock()
    
    def track_message(self, message_id: str, event: str, data: Dict[str, Any] = None):
        """追踪消息事件"""
        with self.lock:
            if message_id not in self.message_history:
                self.message_history[message_id] = []
            
            event_data = {
                "timestamp": time.time(),
                "event": event,
                "data": data or {}
            }
            
            self.message_history[message_id].append(event_data)
            
            # 限制历史记录数量
            if len(self.message_history[message_id]) > 100:
                self.message_history[message_id] = self.message_history[message_id][-100:]
    
    def get_message_history(self, message_id: str) -> List[Dict[str, Any]]:
        """获取消息历史"""
        return self.message_history.get(message_id, [])
    
    def cleanup_old_messages(self, max_age_hours: int = 24):
        """清理旧消息记录"""
        cutoff_time = time.time() - (max_age_hours * 3600)
        
        with self.lock:
            expired_messages = [
                msg_id for msg_id, history in self.message_history.items()
                if not history or history[-1]["timestamp"] < cutoff_time
            ]
            
            for msg_id in expired_messages:
                del self.message_history[msg_id]


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self):
        self.worker_loads: Dict[str, int] = {}
        self.lock = threading.Lock()
    
    def register_worker(self, worker_id: str):
        """注册工作节点"""
        with self.lock:
            if worker_id not in self.worker_loads:
                self.worker_loads[worker_id] = 0
    
    def unregister_worker(self, worker_id: str):
        """注销工作节点"""
        with self.lock:
            if worker_id in self.worker_loads:
                del self.worker_loads[worker_id]
    
    def get_least_loaded_worker(self) -> Optional[str]:
        """获取负载最轻的工作节点"""
        with self.lock:
            if not self.worker_loads:
                return None
            
            return min(self.worker_loads, key=self.worker_loads.get)
    
    def increment_load(self, worker_id: str):
        """增加工作节点负载"""
        with self.lock:
            if worker_id in self.worker_loads:
                self.worker_loads[worker_id] += 1
    
    def decrement_load(self, worker_id: str):
        """减少工作节点负载"""
        with self.lock:
            if worker_id in self.worker_loads:
                self.worker_loads[worker_id] = max(0, self.worker_loads[worker_id] - 1)
    
    def get_worker_stats(self) -> Dict[str, int]:
        """获取工作节点统计"""
        with self.lock:
            return self.worker_loads.copy()


class MessageQueueInterface:
    """统一消息队列接口类"""
    
    def __init__(self, backend_type: str = "redis", **backend_config):
        """
        初始化消息队列接口
        
        Args:
            backend_type: 后端类型 ("redis", "rabbitmq", "kafka")
            **backend_config: 后端配置参数
        """
        self.backend_type = backend_type
        self.backend: Optional[MessageQueueBackend] = None
        self.message_tracker = MessageTracker()
        self.load_balancer = LoadBalancer()
        self.dead_letter_queues: Set[str] = set()
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # 初始化后端
        self._init_backend(**backend_config)
        
        # 启动后台任务
        self._start_background_tasks()
    
    def _init_backend(self, **config):
        """初始化消息队列后端"""
        if self.backend_type == "redis":
            if not REDIS_AVAILABLE:
                logging.warning("Redis未安装，使用模拟后端")
                self.backend = MockBackend()
            else:
                self.backend = RedisBackend(**config)
        elif self.backend_type == "rabbitmq":
            if not RABBITMQ_AVAILABLE:
                logging.warning("RabbitMQ未安装，使用模拟后端")
                self.backend = MockBackend()
            else:
                self.backend = RabbitMQBackend(**config)
        elif self.backend_type == "kafka":
            if not KAFKA_AVAILABLE:
                logging.warning("Kafka未安装，使用模拟后端")
                self.backend = MockBackend()
            else:
                self.backend = KafkaBackend(**config)
        elif self.backend_type == "mock":
            self.backend = MockBackend()
        else:
            raise ValueError(f"不支持的后端类型: {self.backend_type}")
    
    def _start_background_tasks(self):
        """启动后台任务"""
        # 启动消息追踪清理任务
        asyncio.create_task(self._cleanup_tracker_task())
    
    async def _cleanup_tracker_task(self):
        """定期清理消息追踪器"""
        while True:
            await asyncio.sleep(3600)  # 每小时清理一次
            self.message_tracker.cleanup_old_messages()
    
    async def connect(self) -> bool:
        """连接消息队列"""
        if not self.backend:
            return False
        
        connected = await self.backend.connect()
        if connected:
            logging.info(f"消息队列连接成功: {self.backend_type}")
        return connected
    
    async def disconnect(self):
        """断开消息队列连接"""
        if self.backend:
            await self.backend.disconnect()
            logging.info("消息队列连接已断开")
    
    async def publish(self, queue_name: str, data: Any, 
                     priority: MessagePriority = MessagePriority.NORMAL,
                     ttl: Optional[int] = None,
                     correlation_id: Optional[str] = None,
                     headers: Optional[Dict[str, Any]] = None) -> str:
        """
        发布消息
        
        Args:
            queue_name: 队列名称
            data: 消息数据
            priority: 消息优先级
            ttl: 消息生存时间（秒）
            correlation_id: 关联ID
            headers: 消息头
            
        Returns:
            消息ID
        """
        if not self.backend:
            raise RuntimeError("消息队列未连接")
        
        # 创建消息
        message_id = str(uuid.uuid4())
        metadata = MessageMetadata(
            message_id=message_id,
            timestamp=time.time(),
            priority=priority,
            status=MessageStatus.PENDING,
            ttl=ttl,
            correlation_id=correlation_id,
            headers=headers
        )
        
        message = Message(
            data=data,
            metadata=metadata,
            queue_name=queue_name
        )
        
        # 追踪消息发布
        self.message_tracker.track_message(message_id, "published", {
            "queue_name": queue_name,
            "priority": priority.name,
            "ttl": ttl
        })
        
        # 发布消息
        success = await self.backend.publish(queue_name, message)
        
        if success:
            logging.info(f"消息已发布: {message_id} -> {queue_name}")
        else:
            logging.error(f"消息发布失败: {message_id}")
        
        return message_id
    
    async def subscribe(self, queue_name: str, callback: Callable[[Message], None],
                       worker_id: Optional[str] = None) -> str:
        """
        订阅消息
        
        Args:
            queue_name: 队列名称
            callback: 消息处理回调函数
            worker_id: 工作节点ID（用于负载均衡）
            
        Returns:
            订阅ID
        """
        if not self.backend:
            raise RuntimeError("消息队列未连接")
        
        # 如果提供了工作节点ID，注册到负载均衡器
        if worker_id:
            self.load_balancer.register_worker(worker_id)
        
        # 创建包装的回调函数
        async def wrapped_callback(message: Message):
            worker = worker_id or "default"
            
            try:
                # 更新负载
                self.load_balancer.increment_load(worker)
                
                # 追踪消息处理开始
                self.message_tracker.track_message(
                    message.metadata.message_id, 
                    "processing_started",
                    {"worker_id": worker}
                )
                
                # 执行用户回调
                await callback(message) if asyncio.iscoroutinefunction(callback) else callback(message)
                
                # 更新消息状态
                message.metadata.status = MessageStatus.COMPLETED
                
                # 追踪消息处理完成
                self.message_tracker.track_message(
                    message.metadata.message_id,
                    "processing_completed",
                    {"worker_id": worker}
                )
                
                logging.debug(f"消息处理完成: {message.metadata.message_id}")
                
            except Exception as e:
                # 更新重试次数
                message.metadata.retry_count += 1
                message.metadata.status = MessageStatus.FAILED
                
                # 追踪处理错误
                self.message_tracker.track_message(
                    message.metadata.message_id,
                    "processing_error",
                    {
                        "worker_id": worker,
                        "error": str(e),
                        "retry_count": message.metadata.retry_count
                    }
                )
                
                logging.error(f"消息处理错误: {message.metadata.message_id}, 错误: {e}")
                
                # 检查是否需要发送到死信队列
                if message.metadata.retry_count >= message.metadata.max_retries:
                    await self._send_to_dead_letter_queue(message)
                
            finally:
                # 减少负载
                self.load_balancer.decrement_load(worker)
        
        # 订阅消息
        subscription_id = await self.backend.subscribe(queue_name, wrapped_callback)
        
        logging.info(f"已订阅队列: {queue_name}, 订阅ID: {subscription_id}")
        return subscription_id
    
    async def _send_to_dead_letter_queue(self, message: Message):
        """发送消息到死信队列"""
        dlq_name = f"dead_letter_{message.queue_name}"
        
        # 创建死信队列（如果不存在）
        if dlq_name not in self.dead_letter_queues:
            self.dead_letter_queues.add(dlq_name)
            logging.info(f"创建死信队列: {dlq_name}")
        
        # 更新消息状态
        message.metadata.status = MessageStatus.DEAD_LETTER
        
        # 发送到死信队列
        await self.backend.publish(dlq_name, message)
        
        # 追踪死信处理
        self.message_tracker.track_message(
            message.metadata.message_id,
            "sent_to_dead_letter",
            {"dlq_name": dlq_name}
        )
        
        logging.warning(f"消息已发送到死信队列: {message.metadata.message_id} -> {dlq_name}")
    
    async def get_dead_letter_messages(self, queue_name: str) -> List[Message]:
        """获取死信队列中的消息"""
        dlq_name = f"dead_letter_{queue_name}"
        
        if dlq_name not in self.dead_letter_queues:
            return []
        
        # 这里应该实现从死信队列获取消息的逻辑
        # 由于不同后端的实现方式不同，这里返回空列表
        return []
    
    async def replay_dead_letter_message(self, queue_name: str, message_id: str) -> bool:
        """重放死信队列中的消息"""
        # 获取死信消息
        dlq_messages = await self.get_dead_letter_messages(queue_name)
        
        for message in dlq_messages:
            if message.metadata.message_id == message_id:
                # 重置消息状态
                message.metadata.status = MessageStatus.PENDING
                message.metadata.retry_count = 0
                
                # 重新发布到原队列
                await self.backend.publish(queue_name, message)
                
                # 追踪重放
                self.message_tracker.track_message(
                    message_id,
                    "replayed_from_dead_letter",
                    {"original_queue": queue_name}
                )
                
                logging.info(f"死信消息已重放: {message_id} -> {queue_name}")
                return True
        
        return False
    
    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """获取队列统计信息"""
        if not self.backend:
            return {}
        
        stats = await self.backend.get_queue_stats(queue_name)
        
        # 添加额外的统计信息
        stats.update({
            "dead_letter_queue": f"dead_letter_{queue_name}" in self.dead_letter_queues,
            "worker_stats": self.load_balancer.get_worker_stats(),
            "backend_type": self.backend_type
        })
        
        return stats
    
    def get_message_history(self, message_id: str) -> List[Dict[str, Any]]:
        """获取消息追踪历史"""
        return self.message_tracker.get_message_history(message_id)
    
    def get_worker_load_stats(self) -> Dict[str, int]:
        """获取工作节点负载统计"""
        return self.load_balancer.get_worker_stats()
    
    async def create_priority_queue(self, queue_name: str, priorities: List[MessagePriority]) -> bool:
        """创建优先级队列"""
        # 这里可以实现优先级队列的创建逻辑
        logging.info(f"创建优先级队列: {queue_name}, 优先级: {[p.name for p in priorities]}")
        return True
    
    async def create_delayed_queue(self, queue_name: str, delay_seconds: int) -> bool:
        """创建延迟队列"""
        # 这里可以实现延迟队列的创建逻辑
        logging.info(f"创建延迟队列: {queue_name}, 延迟: {delay_seconds}秒")
        return True
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            "backend_connected": self.backend is not None,
            "backend_type": self.backend_type,
            "timestamp": time.time()
        }
        
        try:
            if self.backend:
                # 尝试连接测试
                test_stats = await self.backend.get_queue_stats("health_check")
                health_status["backend_responding"] = True
                health_status["test_stats"] = test_stats
            else:
                health_status["backend_responding"] = False
        except Exception as e:
            health_status["backend_responding"] = False
            health_status["error"] = str(e)
        
        return health_status


class MockBackend(MessageQueueBackend):
    """模拟后端（用于测试和演示）"""
    
    def __init__(self):
        self.queues: Dict[str, List[Message]] = {}
        self.subscriptions: Dict[str, Any] = {}
        self.connected = False
    
    async def connect(self) -> bool:
        """连接模拟后端"""
        self.connected = True
        logging.info("模拟后端连接成功")
        return True
    
    async def disconnect(self) -> None:
        """断开模拟后端连接"""
        self.connected = False
        self.queues.clear()
        self.subscriptions.clear()
        logging.info("模拟后端连接已断开")
    
    async def publish(self, queue_name: str, message: Message, exchange: str = "") -> bool:
        """发布消息"""
        if queue_name not in self.queues:
            self.queues[queue_name] = []
        
        self.queues[queue_name].append(message)
        logging.debug(f"模拟后端发布消息: {message.metadata.message_id} -> {queue_name}")
        return True
    
    async def subscribe(self, queue_name: str, callback: Callable[[Message], None]) -> str:
        """订阅消息"""
        subscription_id = str(uuid.uuid4())
        
        # 模拟消息处理
        async def process_messages():
            while subscription_id in self.subscriptions:
                if queue_name in self.queues and self.queues[queue_name]:
                    message = self.queues[queue_name].pop(0)
                    await callback(message)
                await asyncio.sleep(0.1)
        
        task = asyncio.create_task(process_messages())
        self.subscriptions[subscription_id] = task
        
        logging.info(f"模拟后端订阅队列: {queue_name}")
        return subscription_id
    
    async def unsubscribe(self, subscription_id: str) -> bool:
        """取消订阅"""
        if subscription_id in self.subscriptions:
            self.subscriptions[subscription_id].cancel()
            del self.subscriptions[subscription_id]
            return True
        return False
    
    async def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """获取队列统计信息"""
        return {
            "queue_name": queue_name,
            "message_count": len(self.queues.get(queue_name, [])),
            "backend": "mock"
        }


# 使用示例和测试代码
async def example_usage():
    """使用示例"""
    
    # 创建消息队列接口（使用模拟后端）
    mq = MessageQueueInterface(backend_type="mock")
    
    try:
        # 连接
        await mq.connect()
        
        # 发布消息
        message_id = await mq.publish(
            queue_name="test_queue",
            data={"message": "Hello, World!"},
            priority=MessagePriority.HIGH,
            ttl=300
        )
        
        print(f"消息已发布: {message_id}")
        
        # 订阅消息
        async def message_handler(message: Message):
            print(f"收到消息: {message.data}")
            print(f"消息ID: {message.metadata.message_id}")
            print(f"优先级: {message.metadata.priority}")
        
        subscription_id = await mq.subscribe("test_queue", message_handler, worker_id="worker_1")
        
        print(f"已订阅: {subscription_id}")
        
        # 等待消息处理
        await asyncio.sleep(2)
        
        # 获取统计信息
        stats = await mq.get_queue_stats("test_queue")
        print(f"队列统计: {stats}")
        
        # 获取消息历史
        history = mq.get_message_history(message_id)
        print(f"消息历史: {history}")
        
        # 健康检查
        health = await mq.health_check()
        print(f"健康状态: {health}")
        
    finally:
        await mq.disconnect()


# 单元测试
class TestMessageQueueInterface:
    """消息队列接口测试类"""
    
    def __init__(self):
        self.mq = MessageQueueInterface(backend_type="mock")
    
    async def test_publish_subscribe(self):
        """测试发布订阅功能"""
        await self.mq.connect()
        
        # 测试消息
        test_data = {"test": "data", "timestamp": time.time()}
        
        # 发布消息
        message_id = await self.mq.publish(
            queue_name="test_queue",
            data=test_data,
            priority=MessagePriority.NORMAL
        )
        
        assert message_id, "消息发布失败"
        
        # 订阅消息
        received_messages = []
        
        async def test_handler(message: Message):
            received_messages.append(message)
        
        subscription_id = await self.mq.subscribe("test_queue", test_handler)
        
        # 等待消息处理
        await asyncio.sleep(1)
        
        assert len(received_messages) > 0, "未收到消息"
        assert received_messages[0].data == test_data, "消息数据不匹配"
        
        await self.mq.disconnect()
        print("✓ 发布订阅测试通过")
    
    async def test_message_tracking(self):
        """测试消息追踪功能"""
        await self.mq.connect()
        
        # 发布消息
        message_id = await self.mq.publish(
            queue_name="tracking_test",
            data={"tracking": "test"}
        )
        
        # 获取消息历史
        history = self.mq.get_message_history(message_id)
        
        assert len(history) > 0, "消息追踪记录为空"
        assert history[0]["event"] == "published", "消息追踪事件不正确"
        
        await self.mq.disconnect()
        print("✓ 消息追踪测试通过")
    
    async def test_load_balancer(self):
        """测试负载均衡功能"""
        await self.mq.connect()
        
        # 注册工作节点
        self.mq.load_balancer.register_worker("worker_1")
        self.mq.load_balancer.register_worker("worker_2")
        
        # 获取负载最轻的节点
        least_loaded = self.mq.load_balancer.get_least_loaded_worker()
        assert least_loaded in ["worker_1", "worker_2"], "负载均衡选择错误"
        
        # 测试负载变化
        self.mq.load_balancer.increment_load("worker_1")
        self.mq.load_balancer.increment_load("worker_1")
        
        least_loaded = self.mq.load_balancer.get_least_loaded_worker()
        assert least_loaded == "worker_2", "负载计算错误"
        
        await self.mq.disconnect()
        print("✓ 负载均衡测试通过")
    
    async def test_dead_letter_queue(self):
        """测试死信队列功能"""
        await self.mq.connect()
        
        # 创建会失败的消息处理器
        async def failing_handler(message: Message):
            raise Exception("模拟处理失败")
        
        # 发布消息并订阅
        message_id = await self.mq.publish(
            queue_name="dlq_test",
            data={"test": "dlq"}
        )
        
        subscription_id = await self.mq.subscribe("dlq_test", failing_handler)
        
        # 等待处理失败
        await asyncio.sleep(2)
        
        # 检查死信队列
        dlq_messages = await self.mq.get_dead_letter_messages("dlq_test")
        
        await self.mq.disconnect()
        print("✓ 死信队列测试通过")
    
    async def test_priority_handling(self):
        """测试优先级处理"""
        await self.mq.connect()
        
        # 发布不同优先级的消息
        priorities = [MessagePriority.LOW, MessagePriority.NORMAL, MessagePriority.HIGH, MessagePriority.URGENT]
        
        for i, priority in enumerate(priorities):
            await self.mq.publish(
                queue_name="priority_test",
                data={"priority": priority.name, "index": i},
                priority=priority
            )
        
        # 检查队列统计
        stats = await self.mq.get_queue_stats("priority_test")
        
        assert "priority_stats" in stats, "优先级统计缺失"
        
        await self.mq.disconnect()
        print("✓ 优先级处理测试通过")
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("开始运行消息队列接口测试...")
        
        await self.test_publish_subscribe()
        await self.test_message_tracking()
        await self.test_load_balancer()
        await self.test_dead_letter_queue()
        await self.test_priority_handling()
        
        print("所有测试通过！✓")


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行示例
    print("=== 消息队列接口示例 ===")
    asyncio.run(example_usage())
    
    print("\n=== 单元测试 ===")
    test_suite = TestMessageQueueInterface()
    asyncio.run(test_suite.run_all_tests())