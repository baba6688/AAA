#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
事件接口处理器 (Event Interface Handler)

该模块实现了一个完整的事件驱动架构系统，支持事件的发布、订阅、过滤、
路由、异步处理、持久化、重放、监控和版本管理等核心功能。

主要特性：
1. 事件驱动架构支持
2. 事件发布/订阅机制
3. 事件过滤和路由
4. 异步事件处理
5. 事件持久化
6. 事件重放功能
7. 事件监控和告警
8. 事件版本管理
9. 事件流处理

作者：AI Assistant
创建时间：2025-11-05
版本：1.0.0
"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union, Awaitable
from queue import Queue, PriorityQueue
import weakref
import pickle
import gzip


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EventType(Enum):
    """事件类型枚举"""
    SYSTEM = "system"
    BUSINESS = "business"
    USER = "user"
    ERROR = "error"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DATA = "data"


class EventPriority(Enum):
    """事件优先级枚举"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class EventStatus(Enum):
    """事件状态枚举"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"


@dataclass
class Event:
    """事件数据类"""
    id: str
    type: EventType
    priority: EventPriority
    source: str
    timestamp: datetime
    data: Dict[str, Any]
    version: str = "1.0"
    status: EventStatus = EventStatus.PENDING
    correlation_id: Optional[str] = None
    parent_event_id: Optional[str] = None
    tags: Set[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = set()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['type'] = self.type.value
        result['priority'] = self.priority.value
        result['status'] = self.status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Event':
        """从字典创建事件实例"""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['type'] = EventType(data['type'])
        data['priority'] = EventPriority(data['priority'])
        data['status'] = EventStatus(data['status'])
        data['tags'] = set(data.get('tags', []))
        return cls(**data)


class EventFilter:
    """事件过滤器"""
    
    def __init__(self):
        self.filters = []
    
    def add_type_filter(self, event_type: EventType) -> 'EventFilter':
        """添加事件类型过滤"""
        self.filters.append(lambda event: event.type == event_type)
        return self
    
    def add_priority_filter(self, min_priority: EventPriority) -> 'EventFilter':
        """添加优先级过滤"""
        self.filters.append(lambda event: event.priority.value >= min_priority.value)
        return self
    
    def add_source_filter(self, sources: Set[str]) -> 'EventFilter':
        """添加事件源过滤"""
        self.filters.append(lambda event: event.source in sources)
        return self
    
    def add_tag_filter(self, tags: Set[str]) -> 'EventFilter':
        """添加标签过滤"""
        self.filters.append(lambda event: tags.issubset(event.tags))
        return self
    
    def add_custom_filter(self, func: Callable[[Event], bool]) -> 'EventFilter':
        """添加自定义过滤函数"""
        self.filters.append(func)
        return self
    
    def matches(self, event: Event) -> bool:
        """检查事件是否匹配所有过滤器"""
        return all(filter_func(event) for filter_func in self.filters)


class EventRouter:
    """事件路由器"""
    
    def __init__(self):
        self.routes: Dict[str, List[Callable]] = defaultdict(list)
    
    def add_route(self, pattern: str, handler: Callable[[Event], None]) -> None:
        """添加路由规则"""
        self.routes[pattern].append(handler)
    
    def route_event(self, event: Event) -> List[Callable[[Event], None]]:
        """根据事件特征路由到相应的处理器"""
        handlers = []
        
        # 根据事件类型路由
        handlers.extend(self.routes.get(f"type:{event.type.value}", []))
        
        # 根据事件源路由
        handlers.extend(self.routes.get(f"source:{event.source}", []))
        
        # 根据优先级路由
        handlers.extend(self.routes.get(f"priority:{event.priority.value}", []))
        
        # 根据标签路由
        for tag in event.tags:
            handlers.extend(self.routes.get(f"tag:{tag}", []))
        
        return handlers


class EventSubscriber:
    """事件订阅者"""
    
    def __init__(self, name: str, callback: Callable[[Event], Awaitable[None]], 
                 filter_obj: Optional[EventFilter] = None):
        self.name = name
        self.callback = callback
        self.filter_obj = filter_obj
        self.is_active = True
        self.event_count = 0
        self.last_event_time: Optional[datetime] = None


class EventPublisher:
    """事件发布器"""
    
    def __init__(self, name: str):
        self.name = name
        self.published_events = 0
        self.last_publish_time: Optional[datetime] = None


class EventPersistence:
    """事件持久化器"""
    
    def __init__(self, db_path: str = "events.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self) -> None:
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    priority INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    data TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    correlation_id TEXT,
                    parent_event_id TEXT,
                    tags TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON events(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_type ON events(type)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON events(status)
            """)
    
    def save_event(self, event: Event) -> None:
        """保存事件到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO events 
                (id, type, priority, source, timestamp, data, version, 
                 status, correlation_id, parent_event_id, tags, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                event.id,
                event.type.value,
                event.priority.value,
                event.source,
                event.timestamp.isoformat(),
                json.dumps(event.data),
                event.version,
                event.status.value,
                event.correlation_id,
                event.parent_event_id,
                json.dumps(list(event.tags)),
                datetime.now().isoformat()
            ))
    
    def load_events(self, 
                   start_time: Optional[datetime] = None,
                   end_time: Optional[datetime] = None,
                   event_type: Optional[EventType] = None,
                   limit: int = 1000) -> List[Event]:
        """从数据库加载事件"""
        query = "SELECT * FROM events WHERE 1=1"
        params = []
        
        if start_time:
            query += " AND timestamp >= ?"
            params.append(start_time.isoformat())
        
        if end_time:
            query += " AND timestamp <= ?"
            params.append(end_time.isoformat())
        
        if event_type:
            query += " AND type = ?"
            params.append(event_type.value)
        
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            rows = cursor.fetchall()
            
            events = []
            for row in rows:
                event_data = {
                    'id': row[0],
                    'type': row[1],
                    'priority': int(row[2]),
                    'source': row[3],
                    'timestamp': row[4],
                    'data': json.loads(row[5]),
                    'version': row[6],
                    'status': row[7],
                    'correlation_id': row[8],
                    'parent_event_id': row[9],
                    'tags': set(json.loads(row[10])) if row[10] else set()
                }
                events.append(Event.from_dict(event_data))
            
            return events
    
    def delete_old_events(self, days: int = 30) -> int:
        """删除旧事件"""
        cutoff_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM events WHERE timestamp < ?",
                (cutoff_date.isoformat(),)
            )
            deleted_count = cursor.rowcount
        
        logger.info(f"删除了 {deleted_count} 个旧事件")
        return deleted_count


class EventReplay:
    """事件重放器"""
    
    def __init__(self, persistence: EventPersistence):
        self.persistence = persistence
        self.replay_sessions: Dict[str, Dict[str, Any]] = {}
    
    def start_replay(self, 
                    start_time: datetime,
                    end_time: datetime,
                    event_type: Optional[EventType] = None,
                    speed: float = 1.0) -> str:
        """开始事件重放"""
        session_id = str(uuid.uuid4())
        
        events = self.persistence.load_events(
            start_time=start_time,
            end_time=end_time,
            event_type=event_type
        )
        
        self.replay_sessions[session_id] = {
            'events': events,
            'current_index': 0,
            'speed': speed,
            'start_time': start_time,
            'end_time': end_time,
            'is_active': True
        }
        
        logger.info(f"启动事件重放会话 {session_id}，包含 {len(events)} 个事件")
        return session_id
    
    async def replay_events(self, session_id: str, callback: Callable[[Event], Awaitable[None]]) -> None:
        """重放事件"""
        if session_id not in self.replay_sessions:
            raise ValueError(f"重放会话 {session_id} 不存在")
        
        session = self.replay_sessions[session_id]
        if not session['is_active']:
            raise ValueError(f"重放会话 {session_id} 已停止")
        
        events = session['events']
        speed = session['speed']
        
        for i, event in enumerate(events):
            if not session['is_active']:
                break
            
            session['current_index'] = i
            
            # 调整时间戳以反映重放速度
            replay_event = Event(
                id=f"{event.id}_replay_{session_id}",
                type=event.type,
                priority=event.priority,
                source=f"{event.source}_replay",
                timestamp=datetime.now(),
                data=event.data.copy(),
                version=event.version,
                status=EventStatus.PENDING,
                correlation_id=f"{event.correlation_id}_replay" if event.correlation_id else None,
                parent_event_id=event.parent_event_id,
                tags=event.tags.copy()
            )
            
            await callback(replay_event)
            
            # 根据速度控制重放间隔
            if speed > 0:
                await asyncio.sleep(1.0 / speed)
    
    def stop_replay(self, session_id: str) -> None:
        """停止事件重放"""
        if session_id in self.replay_sessions:
            self.replay_sessions[session_id]['is_active'] = False
            logger.info(f"停止事件重放会话 {session_id}")
    
    def get_replay_progress(self, session_id: str) -> Dict[str, Any]:
        """获取重放进度"""
        if session_id not in self.replay_sessions:
            return {}
        
        session = self.replay_sessions[session_id]
        return {
            'session_id': session_id,
            'total_events': len(session['events']),
            'current_index': session['current_index'],
            'progress': session['current_index'] / len(session['events']) if session['events'] else 0,
            'is_active': session['is_active']
        }


class EventMonitor:
    """事件监控器"""
    
    def __init__(self):
        self.metrics: Dict[str, Any] = defaultdict(int)
        self.alerts: List[Dict[str, Any]] = []
        self.alert_rules: List[Callable[[Dict[str, Any]], bool]] = []
        self.thresholds: Dict[str, float] = {
            'events_per_minute': 1000,
            'error_rate': 0.05,
            'processing_time': 5.0
        }
    
    def record_event(self, event: Event, processing_time: float = 0.0) -> None:
        """记录事件处理指标"""
        self.metrics['total_events'] += 1
        self.metrics[f'events_{event.type.value}'] += 1
        
        # 记录处理时间
        if processing_time > 0:
            self.metrics['total_processing_time'] += processing_time
            self.metrics['avg_processing_time'] = (
                self.metrics['total_processing_time'] / 
                self.metrics['total_events']
            )
        
        # 记录错误
        if event.status == EventStatus.FAILED:
            self.metrics['failed_events'] += 1
        
        # 计算错误率
        if self.metrics['total_events'] > 0:
            self.metrics['error_rate'] = (
                self.metrics['failed_events'] / self.metrics['total_events']
            )
        
        # 检查告警规则
        self._check_alerts()
    
    def _check_alerts(self) -> None:
        """检查告警规则"""
        for rule in self.alert_rules:
            if rule(self.metrics):
                alert = {
                    'timestamp': datetime.now(),
                    'metrics': self.metrics.copy(),
                    'message': '告警规则触发'
                }
                self.alerts.append(alert)
                logger.warning(f"触发告警: {alert}")
    
    def add_alert_rule(self, rule: Callable[[Dict[str, Any]], bool]) -> None:
        """添加告警规则"""
        self.alert_rules.append(rule)
    
    def set_threshold(self, metric: str, value: float) -> None:
        """设置告警阈值"""
        self.thresholds[metric] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        return dict(self.metrics)
    
    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取告警列表"""
        return self.alerts[-limit:]


class EventVersionManager:
    """事件版本管理器"""
    
    def __init__(self):
        self.versions: Dict[str, Dict[str, Any]] = {}
        self.migration_rules: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
    
    def register_version(self, version: str, schema: Dict[str, Any]) -> None:
        """注册事件版本"""
        self.versions[version] = {
            'schema': schema,
            'registered_at': datetime.now()
        }
        logger.info(f"注册事件版本: {version}")
    
    def add_migration_rule(self, from_version: str, to_version: str, 
                          migration_func: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        """添加版本迁移规则"""
        key = f"{from_version}->{to_version}"
        self.migration_rules[key] = migration_func
        logger.info(f"添加版本迁移规则: {key}")
    
    def migrate_event(self, event: Event, target_version: str) -> Event:
        """迁移事件到目标版本"""
        if event.version == target_version:
            return event
        
        migration_path = self._find_migration_path(event.version, target_version)
        if not migration_path:
            raise ValueError(f"无法找到从版本 {event.version} 到 {target_version} 的迁移路径")
        
        event_data = event.to_dict()
        
        # 应用迁移规则
        for i in range(len(migration_path) - 1):
            from_ver = migration_path[i]
            to_ver = migration_path[i + 1]
            migration_key = f"{from_ver}->{to_ver}"
            
            if migration_key in self.migration_rules:
                event_data = self.migration_rules[migration_key](event_data)
            else:
                logger.warning(f"缺少迁移规则: {migration_key}")
        
        # 创建新事件
        migrated_event = Event.from_dict(event_data)
        migrated_event.version = target_version
        
        logger.info(f"事件 {event.id} 从版本 {event.version} 迁移到 {target_version}")
        return migrated_event
    
    def _find_migration_path(self, from_version: str, to_version: str) -> Optional[List[str]]:
        """查找迁移路径"""
        # 简单的BFS查找迁移路径
        from collections import deque
        
        queue = deque([(from_version, [from_version])])
        visited = {from_version}
        
        while queue:
            current_version, path = queue.popleft()
            
            if current_version == to_version:
                return path
            
            # 查找所有可能的迁移目标
            for migration_key in self.migration_rules:
                if migration_key.startswith(f"{current_version}->"):
                    next_version = migration_key.split("->")[1]
                    if next_version not in visited:
                        visited.add(next_version)
                        queue.append((next_version, path + [next_version]))
        
        return None


class EventStreamProcessor:
    """事件流处理器"""
    
    def __init__(self, buffer_size: int = 1000):
        self.buffer_size = buffer_size
        self.event_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=buffer_size))
        self.stream_processors: Dict[str, List[Callable]] = defaultdict(list)
        self.is_running = False
        self.processing_tasks: Dict[str, asyncio.Task] = {}
    
    def add_stream_processor(self, stream_name: str, processor: Callable[[Event], Awaitable[None]]) -> None:
        """添加流处理器"""
        self.stream_processors[stream_name].append(processor)
    
    async def start_processing(self) -> None:
        """开始流处理"""
        self.is_running = True
        
        for stream_name, processors in self.stream_processors.items():
            task = asyncio.create_task(self._process_stream(stream_name, processors))
            self.processing_tasks[stream_name] = task
        
        logger.info("开始事件流处理")
    
    async def stop_processing(self) -> None:
        """停止流处理"""
        self.is_running = False
        
        for task in self.processing_tasks.values():
            task.cancel()
        
        # 等待所有任务完成
        await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)
        
        logger.info("停止事件流处理")
    
    async def _process_stream(self, stream_name: str, processors: List[Callable]) -> None:
        """处理事件流"""
        buffer = self.event_buffers[stream_name]
        
        while self.is_running:
            if buffer:
                event = buffer.popleft()
                
                for processor in processors:
                    try:
                        await processor(event)
                    except Exception as e:
                        logger.error(f"流处理器 {processor.__name__} 处理事件失败: {e}")
            else:
                await asyncio.sleep(0.1)  # 避免忙等待
    
    def publish_to_stream(self, stream_name: str, event: Event) -> None:
        """发布事件到流"""
        self.event_buffers[stream_name].append(event)


class EventInterfaceHandler:
    """
    事件接口处理器主类
    
    集成所有事件处理功能，提供统一的事件处理接口。
    """
    
    def __init__(self, 
                 db_path: str = "events.db",
                 max_workers: int = 4,
                 enable_persistence: bool = True,
                 enable_monitoring: bool = True,
                 enable_replay: bool = True):
        """
        初始化事件处理器
        
        Args:
            db_path: 数据库文件路径
            max_workers: 最大工作线程数
            enable_persistence: 是否启用事件持久化
            enable_monitoring: 是否启用事件监控
            enable_replay: 是否启用事件重放
        """
        self.db_path = db_path
        self.max_workers = max_workers
        self.enable_persistence = enable_persistence
        self.enable_monitoring = enable_monitoring
        self.enable_replay = enable_replay
        
        # 初始化组件
        self.router = EventRouter()
        self.filter_builder = EventFilter()
        self.subscribers: Dict[str, EventSubscriber] = {}
        self.publishers: Dict[str, EventPublisher] = {}
        
        # 事件队列
        self.event_queue: "PriorityQueue[Event]" = PriorityQueue()
        self.processing_queue: "Queue[Event]" = Queue()
        
        # 异步处理
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_running = False
        self.processing_tasks: List[asyncio.Task] = []
        
        # 初始化可选组件
        if enable_persistence:
            self.persistence = EventPersistence(db_path)
        else:
            self.persistence = None
        
        if enable_monitoring:
            self.monitor = EventMonitor()
        else:
            self.monitor = None
        
        if enable_replay:
            self.replay = EventReplay(self.persistence) if self.persistence else None
        else:
            self.replay = None
        
        self.version_manager = EventVersionManager()
        self.stream_processor = EventStreamProcessor()
        
        # 线程安全
        self.lock = threading.RLock()
        
        logger.info("事件接口处理器初始化完成")
    
    async def start(self) -> None:
        """启动事件处理器"""
        if self.is_running:
            logger.warning("事件处理器已在运行")
            return
        
        self.is_running = True
        
        # 启动事件处理任务
        self.processing_tasks = [
            asyncio.create_task(self._event_processor()),
            asyncio.create_task(self._cleanup_worker())
        ]
        
        # 启动流处理器
        await self.stream_processor.start_processing()
        
        logger.info("事件处理器已启动")
    
    async def stop(self) -> None:
        """停止事件处理器"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # 取消所有处理任务
        for task in self.processing_tasks:
            task.cancel()
        
        # 等待任务完成
        await asyncio.gather(*self.processing_tasks, return_exceptions=True)
        
        # 停止流处理器
        await self.stream_processor.stop_processing()
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        logger.info("事件处理器已停止")
    
    def create_subscriber(self, 
                         name: str,
                         callback: Callable[[Event], Awaitable[None]],
                         filter_obj: Optional[EventFilter] = None) -> str:
        """
        创建事件订阅者
        
        Args:
            name: 订阅者名称
            callback: 回调函数
            filter_obj: 事件过滤器
            
        Returns:
            订阅者ID
        """
        subscriber = EventSubscriber(name, callback, filter_obj)
        
        with self.lock:
            self.subscribers[name] = subscriber
        
        logger.info(f"创建事件订阅者: {name}")
        return name
    
    def remove_subscriber(self, subscriber_id: str) -> bool:
        """移除事件订阅者"""
        with self.lock:
            if subscriber_id in self.subscribers:
                del self.subscribers[subscriber_id]
                logger.info(f"移除事件订阅者: {subscriber_id}")
                return True
        return False
    
    def create_publisher(self, name: str) -> str:
        """创建事件发布器"""
        publisher = EventPublisher(name)
        
        with self.lock:
            self.publishers[name] = publisher
        
        logger.info(f"创建事件发布器: {name}")
        return name
    
    async def publish_event(self, 
                           event_type: EventType,
                           priority: EventPriority,
                           source: str,
                           data: Dict[str, Any],
                           correlation_id: Optional[str] = None,
                           parent_event_id: Optional[str] = None,
                           tags: Optional[Set[str]] = None) -> str:
        """
        发布事件
        
        Args:
            event_type: 事件类型
            priority: 事件优先级
            source: 事件源
            data: 事件数据
            correlation_id: 关联ID
            parent_event_id: 父事件ID
            tags: 事件标签
            
        Returns:
            事件ID
        """
        event_id = str(uuid.uuid4())
        event = Event(
            id=event_id,
            type=event_type,
            priority=priority,
            source=source,
            timestamp=datetime.now(),
            data=data,
            correlation_id=correlation_id,
            parent_event_id=parent_event_id,
            tags=tags or set()
        )
        
        # 保存事件（如果启用持久化）
        if self.persistence:
            self.persistence.save_event(event)
        
        # 添加到队列
        self.event_queue.put((priority.value, event))
        
        logger.info(f"发布事件: {event_id} (类型: {event_type.value})")
        return event_id
    
    async def _event_processor(self) -> None:
        """事件处理器主循环"""
        while self.is_running:
            try:
                if not self.event_queue.empty():
                    priority, event = self.event_queue.get_nowait()
                    
                    # 更新事件状态
                    event.status = EventStatus.PROCESSING
                    
                    # 路由事件
                    handlers = self.router.route_event(event)
                    
                    # 发送给匹配的订阅者
                    await self._dispatch_to_subscribers(event)
                    
                    # 记录处理时间
                    start_time = time.time()
                    
                    # 执行路由处理器
                    for handler in handlers:
                        try:
                            if asyncio.iscoroutinefunction(handler):
                                await handler(event)
                            else:
                                await asyncio.get_event_loop().run_in_executor(
                                    self.executor, handler, event
                                )
                        except Exception as e:
                            logger.error(f"事件处理器执行失败: {e}")
                            event.status = EventStatus.FAILED
                    
                    # 更新监控指标
                    if self.monitor:
                        processing_time = time.time() - start_time
                        self.monitor.record_event(event, processing_time)
                    
                    # 更新事件状态
                    event.status = EventStatus.COMPLETED
                    
                    # 重新保存事件（如果启用持久化）
                    if self.persistence:
                        self.persistence.save_event(event)
                
                await asyncio.sleep(0.01)  # 避免忙等待
                
            except Exception as e:
                logger.error(f"事件处理器错误: {e}")
                await asyncio.sleep(1)
    
    async def _dispatch_to_subscribers(self, event: Event) -> None:
        """分发事件给订阅者"""
        tasks = []
        
        with self.lock:
            for subscriber in self.subscribers.values():
                if not subscriber.is_active:
                    continue
                
                # 检查过滤器
                if subscriber.filter_obj and not subscriber.filter_obj.matches(event):
                    continue
                
                # 创建处理任务
                task = asyncio.create_task(self._handle_subscriber_event(subscriber, event))
                tasks.append(task)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _handle_subscriber_event(self, subscriber: EventSubscriber, event: Event) -> None:
        """处理订阅者事件"""
        try:
            start_time = time.time()
            
            await subscriber.callback(event)
            
            subscriber.event_count += 1
            subscriber.last_event_time = datetime.now()
            
            processing_time = time.time() - start_time
            logger.debug(f"订阅者 {subscriber.name} 处理事件 {event.id} 耗时 {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"订阅者 {subscriber.name} 处理事件失败: {e}")
            event.status = EventStatus.FAILED
    
    async def _cleanup_worker(self) -> None:
        """清理工作器"""
        while self.is_running:
            try:
                # 清理旧事件（如果启用持久化）
                if self.persistence:
                    self.persistence.delete_old_events(days=30)
                
                # 等待1小时
                await asyncio.sleep(3600)
                
            except Exception as e:
                logger.error(f"清理工作器错误: {e}")
                await asyncio.sleep(300)
    
    # 便捷方法
    async def publish_system_event(self, source: str, data: Dict[str, Any], **kwargs) -> str:
        """发布系统事件"""
        return await self.publish_event(
            event_type=EventType.SYSTEM,
            priority=EventPriority.NORMAL,
            source=source,
            data=data,
            **kwargs
        )
    
    async def publish_business_event(self, source: str, data: Dict[str, Any], **kwargs) -> str:
        """发布业务事件"""
        return await self.publish_event(
            event_type=EventType.BUSINESS,
            priority=EventPriority.NORMAL,
            source=source,
            data=data,
            **kwargs
        )
    
    async def publish_error_event(self, source: str, error_message: str, error_details: Dict[str, Any] = None, **kwargs) -> str:
        """发布错误事件"""
        return await self.publish_event(
            event_type=EventType.ERROR,
            priority=EventPriority.HIGH,
            source=source,
            data={'error_message': error_message, 'error_details': error_details or {}},
            **kwargs
        )
    
    def add_route(self, pattern: str, handler: Callable[[Event], Awaitable[None]]) -> None:
        """添加路由规则"""
        self.router.add_route(pattern, handler)
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取监控指标"""
        if self.monitor:
            return self.monitor.get_metrics()
        return {}
    
    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取告警列表"""
        if self.monitor:
            return self.monitor.get_alerts(limit)
        return []
    
    def get_event_stats(self) -> Dict[str, Any]:
        """获取事件统计信息"""
        stats = {
            'subscribers': len(self.subscribers),
            'publishers': len(self.publishers),
            'queue_size': self.event_queue.qsize(),
            'is_running': self.is_running
        }
        
        if self.monitor:
            stats.update(self.monitor.get_metrics())
        
        return stats


# 测试用例
class TestEventInterfaceHandler:
    """事件接口处理器测试类"""
    
    @staticmethod
    async def test_basic_functionality():
        """测试基本功能"""
        print("=== 测试基本功能 ===")
        
        # 创建事件处理器
        handler = EventInterfaceHandler(
            db_path="test_events.db",
            enable_persistence=True,
            enable_monitoring=True
        )
        
        try:
            # 启动处理器
            await handler.start()
            
            # 创建订阅者
            async def test_subscriber(event: Event):
                print(f"收到事件: {event.id}, 类型: {event.type.value}, 数据: {event.data}")
            
            handler.create_subscriber("test_subscriber", test_subscriber)
            
            # 发布测试事件
            await handler.publish_system_event(
                source="test_system",
                data={"message": "测试系统事件", "value": 123}
            )
            
            await handler.publish_business_event(
                source="test_business",
                data={"action": "user_login", "user_id": "user123"}
            )
            
            await handler.publish_error_event(
                source="test_error",
                error_message="测试错误",
                error_details={"error_code": 500, "stack_trace": "..."}
            )
            
            # 等待处理
            await asyncio.sleep(2)
            
            # 获取统计信息
            stats = handler.get_event_stats()
            print(f"事件统计: {stats}")
            
            # 获取监控指标
            metrics = handler.get_metrics()
            print(f"监控指标: {metrics}")
            
        finally:
            await handler.stop()
    
    @staticmethod
    async def test_event_filtering():
        """测试事件过滤"""
        print("\n=== 测试事件过滤 ===")
        
        handler = EventInterfaceHandler()
        await handler.start()
        
        # 创建过滤器
        filter_obj = EventFilter()
        filter_obj.add_type_filter(EventType.SYSTEM)
        filter_obj.add_priority_filter(EventPriority.HIGH)
        
        # 创建订阅者
        async def filtered_subscriber(event: Event):
            print(f"过滤后收到事件: {event.id}, 类型: {event.type.value}")
        
        handler.create_subscriber("filtered_subscriber", filtered_subscriber, filter_obj)
        
        # 发布不同类型的事件
        await handler.publish_event(
            EventType.SYSTEM, EventPriority.HIGH, "test", {"msg": "高优先级系统事件"}
        )
        
        await handler.publish_event(
            EventType.SYSTEM, EventPriority.LOW, "test", {"msg": "低优先级系统事件"}
        )
        
        await handler.publish_event(
            EventType.BUSINESS, EventPriority.HIGH, "test", {"msg": "高优先级业务事件"}
        )
        
        await asyncio.sleep(1)
        await handler.stop()
    
    @staticmethod
    async def test_event_routing():
        """测试事件路由"""
        print("\n=== 测试事件路由 ===")
        
        handler = EventInterfaceHandler()
        await handler.start()
        
        # 添加路由规则
        async def system_handler(event: Event):
            print(f"系统处理器处理事件: {event.id}")
        
        async def error_handler(event: Event):
            print(f"错误处理器处理事件: {event.id}")
        
        handler.add_route("type:system", system_handler)
        handler.add_route("type:error", error_handler)
        
        # 发布事件
        await handler.publish_system_event("test", {"msg": "路由测试"})
        await handler.publish_error_event("test", "路由错误测试")
        
        await asyncio.sleep(1)
        await handler.stop()
    
    @staticmethod
    async def test_event_persistence_and_replay():
        """测试事件持久化和重放"""
        print("\n=== 测试事件持久化和重放 ===")
        
        db_path = "test_replay_events.db"
        
        # 第一阶段：发布事件
        handler1 = EventInterfaceHandler(db_path=db_path, enable_persistence=True)
        await handler1.start()
        
        # 发布测试事件
        event_ids = []
        for i in range(5):
            event_id = await handler1.publish_system_event(
                source="replay_test",
                data={"sequence": i, "message": f"测试事件 {i}"}
            )
            event_ids.append(event_id)
        
        await asyncio.sleep(1)
        await handler1.stop()
        
        # 第二阶段：重放事件
        handler2 = EventInterfaceHandler(db_path=db_path, enable_replay=True)
        await handler2.start()
        
        # 创建重放订阅者
        replayed_events = []
        
        async def replay_subscriber(event: Event):
            replayed_events.append(event)
            print(f"重放事件: {event.id}, 数据: {event.data}")
        
        handler2.create_subscriber("replay_subscriber", replay_subscriber)
        
        # 开始重放
        start_time = datetime.now() - timedelta(minutes=1)
        end_time = datetime.now()
        
        session_id = handler2.replay.start_replay(start_time, end_time)
        
        await handler2.replay.replay_events(session_id, replay_subscriber)
        
        print(f"重放完成，共重放 {len(replayed_events)} 个事件")
        
        await handler2.stop()
    
    @staticmethod
    async def test_event_stream_processing():
        """测试事件流处理"""
        print("\n=== 测试事件流处理 ===")
        
        handler = EventInterfaceHandler()
        await handler.start()
        
        # 添加流处理器
        async def stream_processor(event: Event):
            print(f"流处理事件: {event.id}, 类型: {event.type.value}")
            # 模拟流处理延迟
            await asyncio.sleep(0.1)
        
        handler.stream_processor.add_stream_processor("test_stream", stream_processor)
        
        # 发布事件到流
        for i in range(3):
            event = Event(
                id=str(uuid.uuid4()),
                type=EventType.DATA,
                priority=EventPriority.NORMAL,
                source="stream_test",
                timestamp=datetime.now(),
                data={"stream_data": f"流数据 {i}"}
            )
            handler.stream_processor.publish_to_stream("test_stream", event)
        
        await asyncio.sleep(2)
        await handler.stop()
    
    @staticmethod
    async def test_event_versioning():
        """测试事件版本管理"""
        print("\n=== 测试事件版本管理 ===")
        
        handler = EventInterfaceHandler()
        
        # 注册版本
        handler.version_manager.register_version("1.0", {
            "required_fields": ["id", "type", "data"],
            "optional_fields": ["tags"]
        })
        
        handler.version_manager.register_version("2.0", {
            "required_fields": ["id", "type", "data", "metadata"],
            "optional_fields": ["tags", "version"]
        })
        
        # 添加迁移规则
        def v1_to_v2_migration(event_data: Dict[str, Any]) -> Dict[str, Any]:
            # 添加metadata字段
            event_data["metadata"] = {"migrated": True}
            return event_data
        
        handler.version_manager.add_migration_rule("1.0", "2.0", v1_to_v2_migration)
        
        # 创建v1.0事件
        v1_event = Event(
            id="test_v1",
            type=EventType.SYSTEM,
            priority=EventPriority.NORMAL,
            source="version_test",
            timestamp=datetime.now(),
            data={"message": "v1.0事件"}
        )
        
        print(f"原始事件版本: {v1_event.version}")
        
        # 迁移到v2.0
        v2_event = handler.version_manager.migrate_event(v1_event, "2.0")
        
        print(f"迁移后事件版本: {v2_event.version}")
        print(f"迁移后事件数据: {v2_event.data}")
    
    @staticmethod
    async def run_all_tests():
        """运行所有测试"""
        print("开始事件接口处理器测试...\n")
        
        tests = [
            TestEventInterfaceHandler.test_basic_functionality,
            TestEventInterfaceHandler.test_event_filtering,
            TestEventInterfaceHandler.test_event_routing,
            TestEventInterfaceHandler.test_event_persistence_and_replay,
            TestEventInterfaceHandler.test_event_stream_processing,
            TestEventInterfaceHandler.test_event_versioning
        ]
        
        for test in tests:
            try:
                await test()
                await asyncio.sleep(1)  # 测试间隔
            except Exception as e:
                print(f"测试失败: {e}")
        
        print("\n所有测试完成!")


# 使用示例
async def main():
    """主函数示例"""
    print("事件接口处理器使用示例")
    
    # 创建事件处理器
    handler = EventInterfaceHandler(
        db_path="example_events.db",
        enable_persistence=True,
        enable_monitoring=True,
        enable_replay=True
    )
    
    try:
        # 启动处理器
        await handler.start()
        
        # 创建订阅者
        async def log_subscriber(event: Event):
            print(f"[日志订阅者] 事件: {event.id}, 类型: {event.type.value}")
        
        async def alert_subscriber(event: Event):
            if event.type == EventType.ERROR:
                print(f"[告警订阅者] 检测到错误事件: {event.data}")
        
        handler.create_subscriber("log_subscriber", log_subscriber)
        handler.create_subscriber("alert_subscriber", alert_subscriber)
        
        # 添加路由规则
        async def critical_handler(event: Event):
            print(f"[关键事件处理器] 处理关键事件: {event.id}")
        
        handler.add_route("priority:4", critical_handler)  # CRITICAL优先级
        
        # 发布各种事件
        await handler.publish_system_event(
            source="user_service",
            data={"action": "user_login", "user_id": "user123"}
        )
        
        await handler.publish_business_event(
            source="order_service",
            data={"action": "create_order", "order_id": "order456"}
        )
        
        await handler.publish_error_event(
            source="payment_service",
            error_message="支付失败",
            error_details={"error_code": "PAYMENT_FAILED", "amount": 100}
        )
        
        # 发布高优先级事件
        await handler.publish_event(
            EventType.SYSTEM,
            EventPriority.CRITICAL,
            "monitor_service",
            {"alert": "系统负载过高", "cpu_usage": 95}
        )
        
        # 等待事件处理
        await asyncio.sleep(2)
        
        # 获取统计信息
        stats = handler.get_event_stats()
        print(f"\n事件统计信息:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        # 获取监控指标
        metrics = handler.get_metrics()
        print(f"\n监控指标:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # 获取告警
        alerts = handler.get_alerts()
        if alerts:
            print(f"\n最近告警:")
            for alert in alerts[-3:]:  # 最近3个告警
                print(f"  {alert}")
        
    finally:
        await handler.stop()


if __name__ == "__main__":
    # 运行测试
    asyncio.run(TestEventInterfaceHandler.run_all_tests())
    
    # 运行示例
    print("\n" + "="*50)
    asyncio.run(main())