#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S4消息服务
一个功能完整的消息服务系统，支持消息队列、发布订阅、消息路由等功能
"""

import json
import time
import threading
import uuid
import sqlite3
import logging
from typing import Dict, List, Callable, Optional, Any
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass, asdict, field
from enum import Enum
import queue
import pickle
import os


class MessageStatus(Enum):
    """消息状态枚举"""
    PENDING = "pending"        # 待处理
    PUBLISHED = "published"    # 已发布
    DELIVERED = "delivered"    # 已送达
    CONFIRMED = "confirmed"    # 已确认
    FAILED = "failed"          # 失败
    RETRYING = "retrying"      # 重试中


@dataclass
class Message:
    """消息数据结构"""
    id: str
    topic: str
    payload: Any
    timestamp: float
    status: MessageStatus
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        data = asdict(self)
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Message':
        """从字典创建消息对象"""
        data['status'] = MessageStatus(data['status'])
        return cls(**data)


class MessagePersistence:
    """消息持久化存储"""
    
    def __init__(self, db_path: str = "message_service.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    topic TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    timestamp REAL NOT NULL,
                    status TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    max_retries INTEGER DEFAULT 3,
                    metadata TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_stats (
                    date TEXT PRIMARY KEY,
                    total_published INTEGER DEFAULT 0,
                    total_delivered INTEGER DEFAULT 0,
                    total_failed INTEGER DEFAULT 0,
                    total_retried INTEGER DEFAULT 0,
                    avg_processing_time REAL DEFAULT 0
                )
            """)
    
    def save_message(self, message: Message) -> bool:
        """保存消息到数据库"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO messages 
                    (id, topic, payload, timestamp, status, retry_count, max_retries, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    message.id,
                    message.topic,
                    pickle.dumps(message.payload),
                    message.timestamp,
                    message.status.value,
                    message.retry_count,
                    message.max_retries,
                    json.dumps(message.metadata)
                ))
            return True
        except Exception as e:
            logging.error(f"保存消息失败: {e}")
            return False
    
    def load_message(self, message_id: str) -> Optional[Message]:
        """从数据库加载消息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM messages WHERE id = ?", (message_id,)
                )
                row = cursor.fetchone()
                if row:
                    return Message(
                        id=row[0],
                        topic=row[1],
                        payload=pickle.loads(row[2]),
                        timestamp=row[3],
                        status=MessageStatus(row[4]),
                        retry_count=row[5],
                        max_retries=row[6],
                        metadata=json.loads(row[7] or '{}')
                    )
        except Exception as e:
            logging.error(f"加载消息失败: {e}")
        return None
    
    def update_message_status(self, message_id: str, status: MessageStatus) -> bool:
        """更新消息状态"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "UPDATE messages SET status = ? WHERE id = ?",
                    (status.value, message_id)
                )
            return True
        except Exception as e:
            logging.error(f"更新消息状态失败: {e}")
            return False
    
    def get_messages_by_status(self, status: MessageStatus) -> List[Message]:
        """根据状态获取消息列表"""
        messages = []
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT * FROM messages WHERE status = ?", (status.value,)
                )
                for row in cursor.fetchall():
                    messages.append(Message(
                        id=row[0],
                        topic=row[1],
                        payload=pickle.loads(row[2]),
                        timestamp=row[3],
                        status=MessageStatus(row[4]),
                        retry_count=row[5],
                        max_retries=row[6],
                        metadata=json.loads(row[7] or '{}')
                    ))
        except Exception as e:
            logging.error(f"获取消息列表失败: {e}")
        return messages
    
    def update_stats(self, date: str, published: int = 0, delivered: int = 0, 
                    failed: int = 0, retried: int = 0, processing_time: float = 0) -> bool:
        """更新统计信息"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO message_stats 
                    (date, total_published, total_delivered, total_failed, total_retried, avg_processing_time)
                    VALUES (?, 
                            COALESCE((SELECT total_published FROM message_stats WHERE date = ?), 0) + ?,
                            COALESCE((SELECT total_delivered FROM message_stats WHERE date = ?), 0) + ?,
                            COALESCE((SELECT total_failed FROM message_stats WHERE date = ?), 0) + ?,
                            COALESCE((SELECT total_retried FROM message_stats WHERE date = ?), 0) + ?,
                            ?)
                """, (date, date, published, date, delivered, date, failed, date, retried, processing_time))
            return True
        except Exception as e:
            logging.error(f"更新统计信息失败: {e}")
            return False


class MessageService:
    """消息服务主类"""
    
    def __init__(self, db_path: str = "message_service.db"):
        self.persistence = MessagePersistence(db_path)
        self.message_queues: Dict[str, queue.Queue] = defaultdict(queue.Queue)
        self.subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self.routes: Dict[str, str] = {}  # 主题路由表
        self.message_handlers: Dict[str, Callable] = {}
        self.is_running = False
        self.worker_threads: List[threading.Thread] = []
        self.lock = threading.RLock()
        self.stats = {
            'total_published': 0,
            'total_delivered': 0,
            'total_failed': 0,
            'total_retried': 0,
            'avg_processing_time': 0.0
        }
        
        # 配置日志
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def start(self):
        """启动消息服务"""
        with self.lock:
            if self.is_running:
                logging.warning("消息服务已在运行中")
                return
            
            self.is_running = True
            # 启动消息处理线程
            for i in range(3):  # 启动3个工作线程
                thread = threading.Thread(target=self._message_worker, args=(i,), daemon=True)
                thread.start()
                self.worker_threads.append(thread)
            
            # 启动重试处理线程
            retry_thread = threading.Thread(target=self._retry_worker, daemon=True)
            retry_thread.start()
            self.worker_threads.append(retry_thread)
            
            logging.info("消息服务已启动")
    
    def stop(self):
        """停止消息服务"""
        with self.lock:
            if not self.is_running:
                return
            
            self.is_running = False
            
            # 等待所有线程结束
            for thread in self.worker_threads:
                if thread.is_alive():
                    thread.join(timeout=5)
            
            self.worker_threads.clear()
            logging.info("消息服务已停止")
    
    def publish(self, topic: str, payload: Any, metadata: Dict[str, Any] = None) -> str:
        """发布消息"""
        message_id = str(uuid.uuid4())
        message = Message(
            id=message_id,
            topic=topic,
            payload=payload,
            timestamp=time.time(),
            status=MessageStatus.PENDING,
            metadata=metadata or {}
        )
        
        # 保存消息
        self.persistence.save_message(message)
        
        # 加入队列
        self.message_queues[topic].put(message)
        
        # 更新统计
        self.stats['total_published'] += 1
        date_str = datetime.now().strftime('%Y-%m-%d')
        self.persistence.update_stats(date_str, published=1)
        
        logging.info(f"消息已发布: {message_id} -> {topic}")
        return message_id
    
    def subscribe(self, topic: str, callback: Callable[[Message], None]):
        """订阅主题"""
        with self.lock:
            self.subscribers[topic].append(callback)
            logging.info(f"已订阅主题: {topic}")
    
    def unsubscribe(self, topic: str, callback: Callable[[Message], None]):
        """取消订阅"""
        with self.lock:
            if callback in self.subscribers[topic]:
                self.subscribers[topic].remove(callback)
                logging.info(f"已取消订阅主题: {topic}")
    
    def add_route(self, source_topic: str, target_topic: str):
        """添加消息路由"""
        self.routes[source_topic] = target_topic
        logging.info(f"已添加路由: {source_topic} -> {target_topic}")
    
    def remove_route(self, source_topic: str):
        """移除消息路由"""
        if source_topic in self.routes:
            del self.routes[source_topic]
            logging.info(f"已移除路由: {source_topic}")
    
    def confirm_message(self, message_id: str, message_obj: Message = None) -> bool:
        """确认消息"""
        # 如果提供了消息对象，直接更新其状态
        if message_obj and message_obj.id == message_id:
            message_obj.status = MessageStatus.CONFIRMED
        
        # 从数据库加载消息并更新状态
        message = self.persistence.load_message(message_id)
        if message:
            message.status = MessageStatus.CONFIRMED
            self.persistence.save_message(message)
            logging.info(f"消息已确认: {message_id}")
            return True
        return False
    
    def _message_worker(self, worker_id: int):
        """消息处理工作线程"""
        logging.info(f"消息处理线程 {worker_id} 已启动")
        
        while self.is_running:
            try:
                # 处理所有队列
                topics_to_process = list(self.message_queues.keys())
                for topic in topics_to_process:
                    queue_obj = self.message_queues[topic]
                    
                    try:
                        # 非阻塞获取消息
                        message = queue_obj.get_nowait()
                        self._process_message(message)
                    except queue.Empty:
                        continue
                    except Exception as e:
                        logging.error(f"处理消息队列异常: {e}")
                
                time.sleep(0.1)  # 避免CPU占用过高
                
            except Exception as e:
                logging.error(f"消息处理线程异常: {e}")
                time.sleep(1)
    
    def _process_message(self, message: Message):
        """处理单个消息"""
        start_time = time.time()
        
        try:
            # 更新消息状态
            message.status = MessageStatus.PUBLISHED
            self.persistence.save_message(message)
            
            # 处理路由
            if message.topic in self.routes:
                routed_topic = self.routes[message.topic]
                routed_message = Message(
                    id=str(uuid.uuid4()),
                    topic=routed_topic,
                    payload=message.payload,
                    timestamp=time.time(),
                    status=MessageStatus.PENDING,
                    metadata=message.metadata
                )
                self.message_queues[routed_topic].put(routed_message)
            
            # 通知订阅者
            subscribers = self.subscribers.get(message.topic, [])
            if not subscribers:
                # 如果没有订阅者，标记为已送达
                message.status = MessageStatus.DELIVERED
                self.persistence.save_message(message)
                self.persistence.update_message_status(message.id, MessageStatus.DELIVERED)
                self.stats['total_delivered'] += 1
                
                date_str = datetime.now().strftime('%Y-%m-%d')
                processing_time = time.time() - start_time
                self.persistence.update_stats(date_str, delivered=1, processing_time=processing_time)
            else:
                # 通知所有订阅者
                for callback in subscribers:
                    try:
                        callback(message)
                    except Exception as e:
                        logging.error(f"订阅者回调异常: {e}")
                        message.status = MessageStatus.FAILED
                        self.persistence.save_message(message)
                        self.stats['total_failed'] += 1
                        
                        date_str = datetime.now().strftime('%Y-%m-%d')
                        self.persistence.update_stats(date_str, failed=1)
                        return
                
                # 所有订阅者处理成功
                message.status = MessageStatus.DELIVERED
                self.persistence.save_message(message)
                self.persistence.update_message_status(message.id, MessageStatus.DELIVERED)
                self.stats['total_delivered'] += 1
                
                date_str = datetime.now().strftime('%Y-%m-%d')
                processing_time = time.time() - start_time
                self.persistence.update_stats(date_str, delivered=1, processing_time=processing_time)
            
            logging.info(f"消息处理完成: {message.id}")
            
        except Exception as e:
            logging.error(f"处理消息异常: {e}")
            message.status = MessageStatus.FAILED
            self.persistence.save_message(message)
            self.stats['total_failed'] += 1
            
            date_str = datetime.now().strftime('%Y-%m-%d')
            self.persistence.update_stats(date_str, failed=1)
    
    def _retry_worker(self):
        """重试处理工作线程"""
        logging.info("重试处理线程已启动")
        
        while self.is_running:
            try:
                # 获取需要重试的消息
                pending_messages = self.persistence.get_messages_by_status(MessageStatus.FAILED)
                
                for message in pending_messages:
                    if message.retry_count < message.max_retries:
                        message.retry_count += 1
                        message.status = MessageStatus.RETRYING
                        self.persistence.save_message(message)
                        
                        # 重新加入队列
                        self.message_queues[message.topic].put(message)
                        
                        # 更新统计
                        self.stats['total_retried'] += 1
                        date_str = datetime.now().strftime('%Y-%m-%d')
                        self.persistence.update_stats(date_str, retried=1)
                        
                        logging.info(f"消息重试: {message.id} (第{message.retry_count}次)")
                
                time.sleep(5)  # 每5秒检查一次重试
                
            except Exception as e:
                logging.error(f"重试处理线程异常: {e}")
                time.sleep(5)
    
    def get_queue_size(self, topic: str = None) -> Dict[str, int]:
        """获取队列大小"""
        if topic:
            return {topic: self.message_queues[topic].qsize()}
        
        return {topic: q.qsize() for topic, q in self.message_queues.items()}
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_published': self.stats['total_published'],
            'total_delivered': self.stats['total_delivered'],
            'total_failed': self.stats['total_failed'],
            'total_retried': self.stats['total_retried'],
            'avg_processing_time': self.stats['avg_processing_time'],
            'queue_sizes': self.get_queue_size(),
            'active_subscribers': {topic: len(subs) for topic, subs in self.subscribers.items()},
            'active_routes': len(self.routes)
        }
    
    def get_message_history(self, topic: str = None, limit: int = 100) -> List[Message]:
        """获取消息历史"""
        try:
            with sqlite3.connect(self.persistence.db_path) as conn:
                if topic:
                    cursor = conn.execute("""
                        SELECT * FROM messages 
                        WHERE topic = ? 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (topic, limit))
                else:
                    cursor = conn.execute("""
                        SELECT * FROM messages 
                        ORDER BY timestamp DESC 
                        LIMIT ?
                    """, (limit,))
                
                messages = []
                for row in cursor.fetchall():
                    messages.append(Message(
                        id=row[0],
                        topic=row[1],
                        payload=pickle.loads(row[2]),
                        timestamp=row[3],
                        status=MessageStatus(row[4]),
                        retry_count=row[5],
                        max_retries=row[6],
                        metadata=json.loads(row[7] or '{}')
                    ))
                return messages
        except Exception as e:
            logging.error(f"获取消息历史失败: {e}")
            return []
    
    def clear_queue(self, topic: str):
        """清空指定主题的队列"""
        with self.lock:
            while not self.message_queues[topic].empty():
                try:
                    self.message_queues[topic].get_nowait()
                except queue.Empty:
                    break
            logging.info(f"已清空队列: {topic}")
    
    def clear_all_queues(self):
        """清空所有队列"""
        with self.lock:
            for topic in self.message_queues:
                self.clear_queue(topic)
            logging.info("已清空所有队列")


# 全局消息服务实例
_message_service = None

def get_message_service() -> MessageService:
    """获取全局消息服务实例"""
    global _message_service
    if _message_service is None:
        _message_service = MessageService()
    return _message_service


def init_message_service(db_path: str = "message_service.db") -> MessageService:
    """初始化消息服务"""
    global _message_service
    _message_service = MessageService(db_path)
    return _message_service