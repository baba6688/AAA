"""
L6决策日志记录器模块

该模块提供了全面的决策日志记录功能，包括智能决策过程、策略执行、风险评估、
市场分析、决策解释和效果评估等多个维度的日志记录。支持异步处理、错误处理
和详细的日志管理。

"""

import asyncio
import json
import logging
import threading
import time
import uuid
import hashlib
import pickle
import gzip
import csv
import sqlite3
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager, asynccontextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from queue import Queue, Empty, SimpleQueue
from typing import Any, Dict, List, Optional, Union, Callable, Generator, Set, Tuple, Iterator
from collections import defaultdict, deque
from functools import wraps, lru_cache
from itertools import groupby, combinations
from math import sqrt
from statistics import mean, stdev
import traceback
import weakref
import re
import warnings
from copy import deepcopy
from io import StringIO


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DecisionType(Enum):
    """决策类型枚举"""
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"
    EMERGENCY = "emergency"
    ROUTINE = "routine"


class StrategyType(Enum):
    """策略类型枚举"""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative"
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"
    REACTIVE = "reactive"


class RiskLevel(Enum):
    """风险级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExecutionStatus(Enum):
    """执行状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DecisionInput:
    """决策输入数据类"""
    input_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    input_type: str = ""
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    priority: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class DecisionLogic:
    """决策逻辑数据类"""
    logic_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    algorithm: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.0
    factors: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class DecisionResult:
    """决策结果数据类"""
    result_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    decision_type: DecisionType = DecisionType.OPERATIONAL
    outcome: str = ""
    success: bool = False
    value: Union[float, int, str, Dict[str, Any]] = 0
    confidence: float = 0.0
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['decision_type'] = self.decision_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class StrategyStep:
    """策略执行步骤数据类"""
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_name: str = ""
    description: str = ""
    status: ExecutionStatus = ExecutionStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['status'] = self.status.value
        if self.start_time:
            result['start_time'] = self.start_time.isoformat()
        if self.end_time:
            result['end_time'] = self.end_time.isoformat()
        return result


@dataclass
class RiskAssessment:
    """风险评估数据类"""
    risk_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    risk_type: str = ""
    level: RiskLevel = RiskLevel.LOW
    probability: float = 0.0
    impact: float = 0.0
    score: float = 0.0
    description: str = ""
    mitigation: str = ""
    owner: str = ""
    status: str = "identified"
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['level'] = self.level.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class MarketAnalysis:
    """市场分析数据类"""
    analysis_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    market_segment: str = ""
    trend: str = ""
    trend_strength: float = 0.0
    opportunities: List[str] = field(default_factory=list)
    threats: List[str] = field(default_factory=list)
    indicators: Dict[str, float] = field(default_factory=dict)
    forecast: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class DecisionExplanation:
    """决策解释数据类"""
    explanation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    reason: str = ""
    rationale: str = ""
    evidence: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    impact_analysis: Dict[str, Any] = field(default_factory=dict)
    stakeholder_impact: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class EffectEvaluation:
    """决策效果评估数据类"""
    evaluation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    decision_id: str = ""
    expected_outcome: Any = None
    actual_outcome: Any = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    lessons_learned: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    success_score: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        if self.expected_outcome is not None:
            result['expected_outcome'] = str(self.expected_outcome)
        if self.actual_outcome is not None:
            result['actual_outcome'] = str(self.actual_outcome)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class LogStorage(ABC):
    """日志存储抽象基类"""
    
    @abstractmethod
    async def store(self, log_entry: Dict[str, Any]) -> bool:
        """存储日志条目"""
        pass
    
    @abstractmethod
    async def retrieve(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检索日志条目"""
        pass
    
    @abstractmethod
    async def delete(self, log_id: str) -> bool:
        """删除日志条目"""
        pass


class FileStorage(LogStorage):
    """文件存储实现"""
    
    def __init__(self, base_path: str = "logs"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        self._lock = threading.Lock()
    
    async def store(self, log_entry: Dict[str, Any]) -> bool:
        """存储日志到文件"""
        try:
            with self._lock:
                log_type = log_entry.get('log_type', 'general')
                date_str = datetime.now().strftime('%Y%m%d')
                file_path = self.base_path / f"{log_type}_{date_str}.jsonl"
                
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
                return True
        except Exception as e:
            logging.error(f"文件存储失败: {e}")
            return False
    
    async def retrieve(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从文件检索日志"""
        results = []
        try:
            for file_path in self.base_path.glob("*.jsonl"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            if self._matches_query(entry, query):
                                results.append(entry)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logging.error(f"文件检索失败: {e}")
        return results
    
    async def delete(self, log_id: str) -> bool:
        """删除指定日志"""
        # 文件存储通常不支持按ID删除，这里返回False
        return False
    
    def _matches_query(self, entry: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """检查条目是否匹配查询条件"""
        for key, value in query.items():
            if key not in entry or entry[key] != value:
                return False
        return True


class MemoryStorage(LogStorage):
    """内存存储实现"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.storage: deque = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    async def store(self, log_entry: Dict[str, Any]) -> bool:
        """存储日志到内存"""
        try:
            with self._lock:
                self.storage.append(log_entry)
                return True
        except Exception as e:
            logging.error(f"内存存储失败: {e}")
            return False
    
    async def retrieve(self, query: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从内存检索日志"""
        try:
            with self._lock:
                return [entry for entry in self.storage if self._matches_query(entry, query)]
        except Exception as e:
            logging.error(f"内存检索失败: {e}")
            return []
    
    async def delete(self, log_id: str) -> bool:
        """从内存删除日志"""
        try:
            with self._lock:
                for i, entry in enumerate(self.storage):
                    if entry.get('id') == log_id:
                        del self.storage[i]
                        return True
                return False
        except Exception as e:
            logging.error(f"内存删除失败: {e}")
            return False
    
    def _matches_query(self, entry: Dict[str, Any], query: Dict[str, Any]) -> bool:
        """检查条目是否匹配查询条件"""
        for key, value in query.items():
            if key not in entry or entry[key] != value:
                return False
        return True


class AsyncLogProcessor:
    """异步日志处理器"""
    
    def __init__(self, storage: LogStorage, batch_size: int = 100, flush_interval: float = 5.0):
        self.storage = storage
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.log_queue: Queue = Queue()
        self.batch_queue: List[Dict[str, Any]] = []
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
    
    async def start(self):
        """启动异步处理器"""
        if not self.running:
            self.running = True
            self._task = asyncio.create_task(self._process_logs())
    
    async def stop(self):
        """停止异步处理器"""
        if self.running:
            self.running = False
            self._shutdown_event.set()
            if self._task:
                await self._task
                self._task = None
    
    async def add_log(self, log_entry: Dict[str, Any]):
        """添加日志条目"""
        self.log_queue.put_nowait(log_entry)
    
    async def _process_logs(self):
        """处理日志队列"""
        last_flush = time.time()
        
        while self.running or not self.log_queue.empty():
            try:
                # 获取日志条目
                try:
                    log_entry = self.log_queue.get(timeout=0.1)
                    self.batch_queue.append(log_entry)
                except Empty:
                    pass
                
                # 检查是否需要批量处理
                current_time = time.time()
                should_flush = (
                    len(self.batch_queue) >= self.batch_size or
                    current_time - last_flush >= self.flush_interval
                )
                
                if should_flush and self.batch_queue:
                    await self._flush_batch()
                    last_flush = current_time
                
                # 检查关闭事件
                if self._shutdown_event.is_set():
                    break
                    
            except Exception as e:
                logging.error(f"日志处理错误: {e}")
                await asyncio.sleep(0.1)
        
        # 最后一次刷新
        if self.batch_queue:
            await self._flush_batch()
    
    async def _flush_batch(self):
        """批量刷新日志"""
        if not self.batch_queue:
            return
        
        batch = self.batch_queue.copy()
        self.batch_queue.clear()
        
        # 并发存储日志
        tasks = [self.storage.store(log_entry) for log_entry in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计成功/失败数量
        success_count = sum(1 for result in results if result is True)
        failure_count = len(results) - success_count
        
        if failure_count > 0:
            logging.warning(f"批量存储: {success_count}成功, {failure_count}失败")


class DecisionLogger:
    """
    L6决策日志记录器主类
    
    提供全面的决策日志记录功能，支持多种日志类型、异步处理、
    错误处理和灵活的存储后端。
    """
    
    def __init__(self, 
                 storage_type: str = "memory",
                 storage_config: Optional[Dict[str, Any]] = None,
                 enable_async: bool = True,
                 log_level: LogLevel = LogLevel.INFO,
                 enable_analytics: bool = True):
        """
        初始化决策日志记录器
        
        Args:
            storage_type: 存储类型 ("memory", "file")
            storage_config: 存储配置
            enable_async: 是否启用异步处理
            log_level: 日志级别
            enable_analytics: 是否启用分析功能
        """
        self.storage_type = storage_type
        self.storage_config = storage_config or {}
        self.enable_async = enable_async
        self.log_level = log_level
        self.enable_analytics = enable_analytics
        
        # 初始化存储
        self._init_storage()
        
        # 初始化异步处理器
        if self.enable_async:
            self.async_processor = AsyncLogProcessor(
                self.storage,
                batch_size=self.storage_config.get('batch_size', 100),
                flush_interval=self.storage_config.get('flush_interval', 5.0)
            )
        else:
            self.async_processor = None
        
        # 统计信息
        self.stats = {
            'total_logs': 0,
            'decision_logs': 0,
            'strategy_logs': 0,
            'risk_logs': 0,
            'market_logs': 0,
            'explanation_logs': 0,
            'evaluation_logs': 0,
            'errors': 0
        }
        
        # 线程锁
        self._lock = threading.RLock()
        
        # 注册表
        self._decision_registry: Dict[str, Dict[str, Any]] = {}
        self._strategy_registry: Dict[str, Dict[str, Any]] = {}
        self._risk_registry: Dict[str, Dict[str, Any]] = {}
        
        # 设置日志记录器
        self._setup_logger()
    
    def _init_storage(self):
        """初始化存储后端"""
        if self.storage_type == "memory":
            max_size = self.storage_config.get('max_size', 10000)
            self.storage = MemoryStorage(max_size)
        elif self.storage_type == "file":
            base_path = self.storage_config.get('base_path', 'logs')
            self.storage = FileStorage(base_path)
        else:
            raise ValueError(f"不支持的存储类型: {self.storage_type}")
    
    def _setup_logger(self):
        """设置内部日志记录器"""
        self.logger = logging.getLogger(f"DecisionLogger_{id(self)}")
        self.logger.setLevel(getattr(logging, self.log_level.value))
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    async def start(self):
        """启动日志记录器"""
        if self.async_processor:
            await self.async_processor.start()
        self.logger.info("决策日志记录器已启动")
    
    async def stop(self):
        """停止日志记录器"""
        if self.async_processor:
            await self.async_processor.stop()
        self.logger.info("决策日志记录器已停止")
    
    def _create_log_entry(self, log_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """创建标准日志条目"""
        entry = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'log_type': log_type,
            'data': data,
            'thread_id': threading.get_ident(),
            'process_id': None  # 可以在多进程环境中设置
        }
        return entry
    
    def _update_stats(self, log_type: str):
        """更新统计信息"""
        with self._lock:
            self.stats['total_logs'] += 1
            
            if log_type == 'decision':
                self.stats['decision_logs'] += 1
            elif log_type == 'strategy':
                self.stats['strategy_logs'] += 1
            elif log_type == 'risk':
                self.stats['risk_logs'] += 1
            elif log_type == 'market':
                self.stats['market_logs'] += 1
            elif log_type == 'explanation':
                self.stats['explanation_logs'] += 1
            elif log_type == 'evaluation':
                self.stats['evaluation_logs'] += 1
    
    async def _store_log(self, log_entry: Dict[str, Any]):
        """存储日志条目"""
        try:
            self._update_stats(log_entry['log_type'])
            
            if self.enable_async and self.async_processor:
                await self.async_processor.add_log(log_entry)
            else:
                await self.storage.store(log_entry)
        except Exception as e:
            with self._lock:
                self.stats['errors'] += 1
            self.logger.error(f"存储日志失败: {e}")
            self.logger.error(traceback.format_exc())
    
    # 智能决策过程日志
    async def log_decision_input(self, 
                                input_data: Union[DecisionInput, Dict[str, Any]],
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录决策输入日志
        
        Args:
            input_data: 决策输入数据
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            if isinstance(input_data, dict):
                input_obj = DecisionInput(**input_data)
            else:
                input_obj = input_data
            
            if metadata:
                input_obj.metadata.update(metadata)
            
            log_data = {
                'input': input_obj.to_dict(),
                'input_summary': {
                    'type': input_obj.input_type,
                    'source': input_obj.source,
                    'priority': input_obj.priority,
                    'data_keys': list(input_obj.data.keys())
                }
            }
            
            log_entry = self._create_log_entry('decision_input', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录决策输入日志失败: {e}")
            raise
    
    async def log_decision_logic(self, 
                                logic_data: Union[DecisionLogic, Dict[str, Any]],
                                decision_id: Optional[str] = None,
                                metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录决策逻辑日志
        
        Args:
            logic_data: 决策逻辑数据
            decision_id: 关联的决策ID
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            if isinstance(logic_data, dict):
                logic_obj = DecisionLogic(**logic_data)
            else:
                logic_obj = logic_data
            
            if metadata:
                logic_obj.parameters.update(metadata)
            
            log_data = {
                'logic': logic_obj.to_dict(),
                'decision_id': decision_id,
                'logic_summary': {
                    'algorithm': logic_obj.algorithm,
                    'confidence': logic_obj.confidence,
                    'factors_count': len(logic_obj.factors),
                    'constraints_count': len(logic_obj.constraints)
                }
            }
            
            log_entry = self._create_log_entry('decision_logic', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录决策逻辑日志失败: {e}")
            raise
    
    async def log_decision_result(self, 
                                 result_data: Union[DecisionResult, Dict[str, Any]],
                                 decision_id: Optional[str] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录决策结果日志
        
        Args:
            result_data: 决策结果数据
            decision_id: 关联的决策ID
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            if isinstance(result_data, dict):
                result_obj = DecisionResult(**result_data)
            else:
                result_obj = result_data
            
            if metadata:
                result_obj.metadata.update(metadata)
            
            # 记录到注册表
            if decision_id:
                with self._lock:
                    self._decision_registry[decision_id] = result_obj.to_dict()
            
            log_data = {
                'result': result_obj.to_dict(),
                'decision_id': decision_id,
                'result_summary': {
                    'success': result_obj.success,
                    'confidence': result_obj.confidence,
                    'execution_time': result_obj.execution_time,
                    'value_type': type(result_obj.value).__name__
                }
            }
            
            log_entry = self._create_log_entry('decision_result', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录决策结果日志失败: {e}")
            raise
    
    # 策略执行日志
    async def log_strategy_selection(self,
                                    strategy_id: str,
                                    strategy_type: StrategyType,
                                    selection_reason: str,
                                    alternatives: Optional[List[str]] = None,
                                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录策略选择日志
        
        Args:
            strategy_id: 策略ID
            strategy_type: 策略类型
            selection_reason: 选择原因
            alternatives: 备选策略
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            log_data = {
                'strategy_id': strategy_id,
                'strategy_type': strategy_type.value,
                'selection_reason': selection_reason,
                'alternatives': alternatives or [],
                'selection_timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            log_entry = self._create_log_entry('strategy_selection', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录策略选择日志失败: {e}")
            raise
    
    async def log_strategy_execution(self,
                                    strategy_id: str,
                                    steps: List[Union[StrategyStep, Dict[str, Any]]],
                                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录策略执行日志
        
        Args:
            strategy_id: 策略ID
            steps: 执行步骤列表
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            processed_steps = []
            for step in steps:
                if isinstance(step, dict):
                    step_obj = StrategyStep(**step)
                else:
                    step_obj = step
                processed_steps.append(step_obj.to_dict())
            
            log_data = {
                'strategy_id': strategy_id,
                'steps': processed_steps,
                'steps_count': len(processed_steps),
                'execution_timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            log_entry = self._create_log_entry('strategy_execution', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录策略执行日志失败: {e}")
            raise
    
    async def log_strategy_result(self,
                                 strategy_id: str,
                                 success: bool,
                                 outcome: Any,
                                 performance_metrics: Optional[Dict[str, float]] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录策略执行结果日志
        
        Args:
            strategy_id: 策略ID
            success: 是否成功
            outcome: 执行结果
            performance_metrics: 性能指标
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            log_data = {
                'strategy_id': strategy_id,
                'success': success,
                'outcome': str(outcome) if not isinstance(outcome, (str, int, float)) else outcome,
                'performance_metrics': performance_metrics or {},
                'result_timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            # 记录到注册表
            with self._lock:
                self._strategy_registry[strategy_id] = log_data
            
            log_entry = self._create_log_entry('strategy_result', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录策略结果日志失败: {e}")
            raise
    
    # 风险评估日志
    async def log_risk_identification(self,
                                     risk_data: Union[RiskAssessment, Dict[str, Any]],
                                     detection_method: str,
                                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录风险识别日志
        
        Args:
            risk_data: 风险评估数据
            detection_method: 风险检测方法
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            if isinstance(risk_data, dict):
                risk_obj = RiskAssessment(**risk_data)
            else:
                risk_obj = risk_data
            
            log_data = {
                'risk': risk_obj.to_dict(),
                'detection_method': detection_method,
                'identification_timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            log_entry = self._create_log_entry('risk_identification', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录风险识别日志失败: {e}")
            raise
    
    async def log_risk_evaluation(self,
                                 risk_id: str,
                                 evaluation_criteria: Dict[str, Any],
                                 evaluation_result: Dict[str, float],
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录风险评估日志
        
        Args:
            risk_id: 风险ID
            evaluation_criteria: 评估标准
            evaluation_result: 评估结果
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            log_data = {
                'risk_id': risk_id,
                'evaluation_criteria': evaluation_criteria,
                'evaluation_result': evaluation_result,
                'evaluation_timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            log_entry = self._create_log_entry('risk_evaluation', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录风险评估日志失败: {e}")
            raise
    
    async def log_risk_control(self,
                              risk_id: str,
                              control_measures: List[str],
                              implementation_status: str,
                              effectiveness: Optional[float] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录风险控制日志
        
        Args:
            risk_id: 风险ID
            control_measures: 控制措施
            implementation_status: 实施状态
            effectiveness: 控制效果
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            log_data = {
                'risk_id': risk_id,
                'control_measures': control_measures,
                'implementation_status': implementation_status,
                'effectiveness': effectiveness,
                'control_timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            # 记录到注册表
            with self._lock:
                self._risk_registry[risk_id] = log_data
            
            log_entry = self._create_log_entry('risk_control', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录风险控制日志失败: {e}")
            raise
    
    # 市场分析日志
    async def log_market_trend(self,
                              analysis_data: Union[MarketAnalysis, Dict[str, Any]],
                              trend_indicators: Optional[Dict[str, float]] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录市场趋势分析日志
        
        Args:
            analysis_data: 市场分析数据
            trend_indicators: 趋势指标
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            if isinstance(analysis_data, dict):
                analysis_obj = MarketAnalysis(**analysis_data)
            else:
                analysis_obj = analysis_data
            
            log_data = {
                'analysis': analysis_obj.to_dict(),
                'trend_indicators': trend_indicators or {},
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            log_entry = self._create_log_entry('market_trend', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录市场趋势日志失败: {e}")
            raise
    
    async def log_opportunity_identification(self,
                                            opportunity_id: str,
                                            opportunity_type: str,
                                            description: str,
                                            potential_value: float,
                                            probability: float,
                                            metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录机会识别日志
        
        Args:
            opportunity_id: 机会ID
            opportunity_type: 机会类型
            description: 机会描述
            potential_value: 潜在价值
            probability: 实现概率
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            log_data = {
                'opportunity_id': opportunity_id,
                'opportunity_type': opportunity_type,
                'description': description,
                'potential_value': potential_value,
                'probability': probability,
                'identification_timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            log_entry = self._create_log_entry('opportunity_identification', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录机会识别日志失败: {e}")
            raise
    
    async def log_threat_analysis(self,
                                 threat_id: str,
                                 threat_type: str,
                                 description: str,
                                 severity: float,
                                 likelihood: float,
                                 mitigation_strategies: Optional[List[str]] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录威胁分析日志
        
        Args:
            threat_id: 威胁ID
            threat_type: 威胁类型
            description: 威胁描述
            severity: 严重程度
            likelihood: 发生可能性
            mitigation_strategies: 缓解策略
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            log_data = {
                'threat_id': threat_id,
                'threat_type': threat_type,
                'description': description,
                'severity': severity,
                'likelihood': likelihood,
                'mitigation_strategies': mitigation_strategies or [],
                'analysis_timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            log_entry = self._create_log_entry('threat_analysis', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录威胁分析日志失败: {e}")
            raise
    
    # 决策解释日志
    async def log_decision_explanation(self,
                                      explanation_data: Union[DecisionExplanation, Dict[str, Any]],
                                      decision_id: Optional[str] = None,
                                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录决策解释日志
        
        Args:
            explanation_data: 决策解释数据
            decision_id: 关联的决策ID
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            if isinstance(explanation_data, dict):
                explanation_obj = DecisionExplanation(**explanation_data)
            else:
                explanation_obj = explanation_data
            
            log_data = {
                'explanation': explanation_obj.to_dict(),
                'decision_id': decision_id,
                'explanation_timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            log_entry = self._create_log_entry('decision_explanation', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录决策解释日志失败: {e}")
            raise
    
    async def log_decision_rationale(self,
                                    decision_id: str,
                                    rationale: str,
                                    supporting_evidence: List[str],
                                    decision_criteria: Dict[str, Any],
                                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录决策理由日志
        
        Args:
            decision_id: 决策ID
            rationale: 决策理由
            supporting_evidence: 支持证据
            decision_criteria: 决策标准
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            log_data = {
                'decision_id': decision_id,
                'rationale': rationale,
                'supporting_evidence': supporting_evidence,
                'decision_criteria': decision_criteria,
                'rationale_timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            log_entry = self._create_log_entry('decision_rationale', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录决策理由日志失败: {e}")
            raise
    
    async def log_decision_impact(self,
                                 decision_id: str,
                                 impact_analysis: Dict[str, Any],
                                 stakeholder_impacts: Dict[str, Dict[str, Any]],
                                 long_term_effects: Optional[List[str]] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录决策影响日志
        
        Args:
            decision_id: 决策ID
            impact_analysis: 影响分析
            stakeholder_impacts: 利益相关者影响
            long_term_effects: 长期影响
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            log_data = {
                'decision_id': decision_id,
                'impact_analysis': impact_analysis,
                'stakeholder_impacts': stakeholder_impacts,
                'long_term_effects': long_term_effects or [],
                'impact_timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            log_entry = self._create_log_entry('decision_impact', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录决策影响日志失败: {e}")
            raise
    
    # 决策效果评估日志
    async def log_decision_evaluation(self,
                                     evaluation_data: Union[EffectEvaluation, Dict[str, Any]],
                                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录决策效果评估日志
        
        Args:
            evaluation_data: 效果评估数据
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            if isinstance(evaluation_data, dict):
                evaluation_obj = EffectEvaluation(**evaluation_data)
            else:
                evaluation_obj = evaluation_data
            
            log_data = {
                'evaluation': evaluation_obj.to_dict(),
                'evaluation_timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            log_entry = self._create_log_entry('decision_evaluation', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录决策评估日志失败: {e}")
            raise
    
    async def log_performance_metrics(self,
                                     decision_id: str,
                                     metrics: Dict[str, float],
                                     measurement_period: str,
                                     benchmark_data: Optional[Dict[str, float]] = None,
                                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录性能指标日志
        
        Args:
            decision_id: 决策ID
            metrics: 性能指标
            measurement_period: 测量期间
            benchmark_data: 基准数据
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            log_data = {
                'decision_id': decision_id,
                'metrics': metrics,
                'measurement_period': measurement_period,
                'benchmark_data': benchmark_data or {},
                'metrics_timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            log_entry = self._create_log_entry('performance_metrics', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录性能指标日志失败: {e}")
            raise
    
    async def log_lessons_learned(self,
                                 decision_id: str,
                                 lessons: List[str],
                                 improvement_suggestions: Optional[List[str]] = None,
                                 best_practices: Optional[List[str]] = None,
                                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录经验教训日志
        
        Args:
            decision_id: 决策ID
            lessons: 经验教训
            improvement_suggestions: 改进建议
            best_practices: 最佳实践
            metadata: 额外元数据
            
        Returns:
            日志条目ID
        """
        try:
            log_data = {
                'decision_id': decision_id,
                'lessons': lessons,
                'improvement_suggestions': improvement_suggestions or [],
                'best_practices': best_practices or [],
                'lessons_timestamp': datetime.now(timezone.utc).isoformat(),
                'metadata': metadata or {}
            }
            
            log_entry = self._create_log_entry('lessons_learned', log_data)
            await self._store_log(log_entry)
            
            return log_entry['id']
            
        except Exception as e:
            self.logger.error(f"记录经验教训日志失败: {e}")
            raise
    
    # 批量日志记录方法
    async def log_batch_decisions(self, decisions: List[Dict[str, Any]]) -> List[str]:
        """
        批量记录决策日志
        
        Args:
            decisions: 决策数据列表
            
        Returns:
            日志条目ID列表
        """
        log_ids = []
        for decision in decisions:
            try:
                if 'input' in decision:
                    log_id = await self.log_decision_input(decision['input'])
                    log_ids.append(log_id)
                if 'logic' in decision:
                    log_id = await self.log_decision_logic(decision['logic'])
                    log_ids.append(log_id)
                if 'result' in decision:
                    log_id = await self.log_decision_result(decision['result'])
                    log_ids.append(log_id)
            except Exception as e:
                self.logger.error(f"批量记录决策失败: {e}")
        return log_ids
    
    async def log_batch_risks(self, risks: List[Dict[str, Any]]) -> List[str]:
        """
        批量记录风险日志
        
        Args:
            risks: 风险数据列表
            
        Returns:
            日志条目ID列表
        """
        log_ids = []
        for risk in risks:
            try:
                if 'identification' in risk:
                    log_id = await self.log_risk_identification(
                        risk['identification'], risk.get('detection_method', 'unknown')
                    )
                    log_ids.append(log_id)
                if 'evaluation' in risk:
                    log_id = await self.log_risk_evaluation(
                        risk['risk_id'], risk['evaluation']['criteria'], risk['evaluation']['result']
                    )
                    log_ids.append(log_id)
            except Exception as e:
                self.logger.error(f"批量记录风险失败: {e}")
        return log_ids
    
    # 查询和检索方法
    async def query_logs(self, 
                        query: Dict[str, Any], 
                        log_types: Optional[List[str]] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """
        查询日志条目
        
        Args:
            query: 查询条件
            log_types: 限制的日志类型
            limit: 返回结果限制
            
        Returns:
            匹配的日志条目列表
        """
        try:
            # 添加日志类型过滤
            if log_types:
                query['log_type'] = {'$in': log_types}
            
            results = await self.storage.retrieve(query)
            return results[:limit]
            
        except Exception as e:
            self.logger.error(f"查询日志失败: {e}")
            return []
    
    async def get_decision_history(self, 
                                  decision_id: Optional[str] = None,
                                  time_range: Optional[tuple] = None,
                                  limit: int = 50) -> List[Dict[str, Any]]:
        """
        获取决策历史
        
        Args:
            decision_id: 特定决策ID
            time_range: 时间范围 (start_time, end_time)
            limit: 返回结果限制
            
        Returns:
            决策历史列表
        """
        query = {'log_type': {'$in': ['decision_input', 'decision_logic', 'decision_result']}}
        
        if decision_id:
            query['data.decision_id'] = decision_id
        
        if time_range:
            query['timestamp'] = {
                '$gte': time_range[0].isoformat(),
                '$lte': time_range[1].isoformat()
            }
        
        return await self.query_logs(query, limit=limit)
    
    async def get_strategy_performance(self, 
                                      strategy_id: Optional[str] = None,
                                      time_range: Optional[tuple] = None) -> Dict[str, Any]:
        """
        获取策略性能统计
        
        Args:
            strategy_id: 特定策略ID
            time_range: 时间范围
            
        Returns:
            策略性能统计
        """
        query = {'log_type': {'$in': ['strategy_selection', 'strategy_execution', 'strategy_result']}}
        
        if strategy_id:
            query['data.strategy_id'] = strategy_id
        
        if time_range:
            query['timestamp'] = {
                '$gte': time_range[0].isoformat(),
                '$lte': time_range[1].isoformat()
            }
        
        results = await self.storage.retrieve(query)
        
        # 统计性能指标
        performance = {
            'total_strategies': len([r for r in results if r['log_type'] == 'strategy_selection']),
            'successful_strategies': len([r for r in results if r['log_type'] == 'strategy_result' and r['data']['success']]),
            'failed_strategies': len([r for r in results if r['log_type'] == 'strategy_result' and not r['data']['success']]),
            'average_performance': 0.0,
            'strategy_types': defaultdict(int)
        }
        
        # 计算平均性能
        performance_metrics = []
        for result in results:
            if result['log_type'] == 'strategy_result' and result['data'].get('performance_metrics'):
                metrics = result['data']['performance_metrics']
                if isinstance(metrics, dict):
                    for metric_value in metrics.values():
                        if isinstance(metric_value, (int, float)):
                            performance_metrics.append(metric_value)
        
        if performance_metrics:
            performance['average_performance'] = sum(performance_metrics) / len(performance_metrics)
        
        # 统计策略类型
        for result in results:
            if result['log_type'] == 'strategy_selection':
                strategy_type = result['data'].get('strategy_type', 'unknown')
                performance['strategy_types'][strategy_type] += 1
        
        return performance
    
    async def get_risk_summary(self, 
                              time_range: Optional[tuple] = None) -> Dict[str, Any]:
        """
        获取风险摘要统计
        
        Args:
            time_range: 时间范围
            
        Returns:
            风险摘要统计
        """
        query = {'log_type': {'$in': ['risk_identification', 'risk_evaluation', 'risk_control']}}
        
        if time_range:
            query['timestamp'] = {
                '$gte': time_range[0].isoformat(),
                '$lte': time_range[1].isoformat()
            }
        
        results = await self.storage.retrieve(query)
        
        # 统计风险指标
        risk_summary = {
            'total_risks': 0,
            'high_risks': 0,
            'medium_risks': 0,
            'low_risks': 0,
            'controlled_risks': 0,
            'risk_types': defaultdict(int),
            'average_risk_score': 0.0
        }
        
        risk_scores = []
        for result in results:
            if result['log_type'] == 'risk_identification':
                risk_summary['total_risks'] += 1
                
                risk_data = result['data'].get('risk', {})
                risk_level = risk_data.get('level', 'low')
                risk_type = risk_data.get('risk_type', 'unknown')
                
                risk_summary['risk_types'][risk_type] += 1
                
                if risk_level == 'high':
                    risk_summary['high_risks'] += 1
                elif risk_level == 'medium':
                    risk_summary['medium_risks'] += 1
                else:
                    risk_summary['low_risks'] += 1
                
                risk_score = risk_data.get('score', 0.0)
                if risk_score > 0:
                    risk_scores.append(risk_score)
            
            elif result['log_type'] == 'risk_control':
                risk_summary['controlled_risks'] += 1
        
        if risk_scores:
            risk_summary['average_risk_score'] = sum(risk_scores) / len(risk_scores)
        
        return risk_summary
    
    # 上下文管理器
    @contextmanager
    def decision_context(self, 
                        decision_id: str,
                        context_metadata: Optional[Dict[str, Any]] = None):
        """
        决策上下文管理器
        
        Args:
            decision_id: 决策ID
            context_metadata: 上下文元数据
        """
        start_time = time.time()
        context_data = {
            'decision_id': decision_id,
            'start_time': start_time,
            'metadata': context_metadata or {}
        }
        
        self.logger.info(f"开始决策上下文: {decision_id}")
        
        try:
            yield context_data
        except Exception as e:
            self.logger.error(f"决策上下文异常: {decision_id}, 错误: {e}")
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.logger.info(f"结束决策上下文: {decision_id}, 耗时: {duration:.2f}秒")
    
    @contextmanager
    def strategy_context(self,
                        strategy_id: str,
                        strategy_type: StrategyType,
                        context_metadata: Optional[Dict[str, Any]] = None):
        """
        策略上下文管理器
        
        Args:
            strategy_id: 策略ID
            strategy_type: 策略类型
            context_metadata: 上下文元数据
        """
        start_time = time.time()
        context_data = {
            'strategy_id': strategy_id,
            'strategy_type': strategy_type,
            'start_time': start_time,
            'metadata': context_metadata or {}
        }
        
        self.logger.info(f"开始策略上下文: {strategy_id} ({strategy_type.value})")
        
        try:
            yield context_data
        except Exception as e:
            self.logger.error(f"策略上下文异常: {strategy_id}, 错误: {e}")
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            self.logger.info(f"结束策略上下文: {strategy_id}, 耗时: {duration:.2f}秒")
    
    # 统计和分析方法
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取日志统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            stats = self.stats.copy()
            stats['registries'] = {
                'decisions': len(self._decision_registry),
                'strategies': len(self._strategy_registry),
                'risks': len(self._risk_registry)
            }
            return stats
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要
        
        Returns:
            性能摘要字典
        """
        with self._lock:
            # 计算成功率
            total_decisions = self.stats['decision_logs']
            total_strategies = self.stats['strategy_logs']
            
            # 从注册表计算成功率
            successful_decisions = sum(1 for d in self._decision_registry.values() 
                                     if d.get('success', False))
            successful_strategies = sum(1 for s in self._strategy_registry.values() 
                                      if s.get('success', False))
            
            return {
                'decision_success_rate': successful_decisions / max(total_decisions, 1),
                'strategy_success_rate': successful_strategies / max(total_strategies, 1),
                'total_logs': self.stats['total_logs'],
                'error_rate': self.stats['errors'] / max(self.stats['total_logs'], 1),
                'log_distribution': {
                    'decisions': self.stats['decision_logs'],
                    'strategies': self.stats['strategy_logs'],
                    'risks': self.stats['risk_logs'],
                    'market': self.stats['market_logs'],
                    'explanations': self.stats['explanation_logs'],
                    'evaluations': self.stats['evaluation_logs']
                }
            }
    
    # 清理和维护方法
    async def cleanup_old_logs(self, days: int = 30) -> int:
        """
        清理旧日志
        
        Args:
            days: 保留天数
            
        Returns:
            清理的日志数量
        """
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            query = {'timestamp': {'$lt': cutoff_date.isoformat()}}
            
            # 注意：实际清理逻辑取决于存储后端
            if hasattr(self.storage, 'cleanup_old_entries'):
                return await self.storage.cleanup_old_entries(cutoff_date)
            else:
                self.logger.warning("当前存储后端不支持自动清理")
                return 0
                
        except Exception as e:
            self.logger.error(f"清理旧日志失败: {e}")
            return 0
    
    async def export_logs(self, 
                         query: Optional[Dict[str, Any]] = None,
                         output_path: Optional[str] = None) -> str:
        """
        导出日志到文件
        
        Args:
            query: 查询条件
            output_path: 输出文件路径
            
        Returns:
            导出文件路径
        """
        try:
            if query is None:
                query = {}
            
            logs = await self.storage.retrieve(query)
            
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"decision_logs_export_{timestamp}.json"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"日志导出完成: {output_path}, 条目数: {len(logs)}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"导出日志失败: {e}")
            raise
    
    # 错误处理和恢复
    async def handle_storage_error(self, error: Exception, operation: str):
        """
        处理存储错误
        
        Args:
            error: 错误对象
            operation: 操作名称
        """
        self.logger.error(f"存储错误 [{operation}]: {error}")
        
        # 记录错误统计
        with self._lock:
            self.stats['errors'] += 1
        
        # 尝试恢复策略
        if isinstance(error, (OSError, IOError)):
            self.logger.warning("检测到存储错误，尝试切换到备用存储")
            # 这里可以实现存储后端切换逻辑
        
        # 通知监控或告警系统
        self.logger.critical(f"严重存储错误，操作: {operation}, 错误: {error}")
    
    def enable_debug_mode(self):
        """启用调试模式"""
        self.logger.setLevel(logging.DEBUG)
        self.logger.info("调试模式已启用")
    
    def disable_debug_mode(self):
        """禁用调试模式"""
        self.logger.setLevel(getattr(logging, self.log_level.value))
        self.logger.info("调试模式已禁用")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        if exc_type is not None:
            self.logger.error(f"决策日志记录器异常退出: {exc_val}")
        return self
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'async_processor') and self.async_processor:
            # 尝试优雅关闭异步处理器
            try:
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    loop.create_task(self.async_processor.stop())
            except Exception:
                pass  # 忽略析构时的错误


# 使用示例和测试代码
async def example_usage():
    """使用示例"""
    
    # 创建决策日志记录器
    logger = DecisionLogger(
        storage_type="memory",
        storage_config={"max_size": 1000},
        enable_async=True,
        log_level=LogLevel.INFO
    )
    
    try:
        # 启动日志记录器
        await logger.start()
        
        # 示例1: 记录智能决策过程
        print("=== 智能决策过程日志示例 ===")
        
        # 决策输入
        input_data = DecisionInput(
            input_type="market_data",
            data={"price": 100.5, "volume": 10000},
            source="market_api",
            priority=1
        )
        input_log_id = await logger.log_decision_input(input_data)
        print(f"决策输入日志ID: {input_log_id}")
        
        # 决策逻辑
        logic_data = DecisionLogic(
            algorithm="moving_average_crossover",
            parameters={"period1": 10, "period2": 20},
            reasoning="基于移动平均线交叉信号",
            confidence=0.85,
            factors=["价格趋势", "成交量", "技术指标"]
        )
        logic_log_id = await logger.log_decision_logic(logic_data)
        print(f"决策逻辑日志ID: {logic_log_id}")
        
        # 决策结果
        result_data = DecisionResult(
            decision_type=DecisionType.TACTICAL,
            outcome="买入信号",
            success=True,
            value=100.5,
            confidence=0.85,
            execution_time=0.05
        )
        result_log_id = await logger.log_decision_result(result_data)
        print(f"决策结果日志ID: {result_log_id}")
        
        # 示例2: 记录策略执行
        print("\n=== 策略执行日志示例 ===")
        
        # 策略选择
        strategy_id = "strategy_001"
        selection_log_id = await logger.log_strategy_selection(
            strategy_id=strategy_id,
            strategy_type=StrategyType.AGGRESSIVE,
            selection_reason="市场趋势向好，适合激进策略",
            alternatives=["conservative_strategy", "balanced_strategy"]
        )
        print(f"策略选择日志ID: {selection_log_id}")
        
        # 策略执行步骤
        steps = [
            StrategyStep(
                step_name="市场分析",
                description="分析当前市场状况",
                status=ExecutionStatus.COMPLETED,
                result={"trend": "bullish", "confidence": 0.8}
            ),
            StrategyStep(
                step_name="仓位计算",
                description="计算最优仓位大小",
                status=ExecutionStatus.COMPLETED,
                result={"position_size": 0.1}
            ),
            StrategyStep(
                step_name="执行交易",
                description="执行买入交易",
                status=ExecutionStatus.COMPLETED,
                result={"executed": True, "price": 100.5}
            )
        ]
        execution_log_id = await logger.log_strategy_execution(strategy_id, steps)
        print(f"策略执行日志ID: {execution_log_id}")
        
        # 策略结果
        result_log_id = await logger.log_strategy_result(
            strategy_id=strategy_id,
            success=True,
            outcome="交易成功执行",
            performance_metrics={"return": 0.02, "sharpe_ratio": 1.5}
        )
        print(f"策略结果日志ID: {result_log_id}")
        
        # 示例3: 记录风险评估
        print("\n=== 风险评估日志示例 ===")
        
        # 风险识别
        risk_data = RiskAssessment(
            risk_type="市场风险",
            level=RiskLevel.MEDIUM,
            probability=0.3,
            impact=0.5,
            score=0.15,
            description="市场波动可能导致损失",
            mitigation="设置止损位",
            owner="risk_manager"
        )
        risk_log_id = await logger.log_risk_identification(
            risk_data, 
            detection_method="monte_carlo_simulation"
        )
        print(f"风险识别日志ID: {risk_log_id}")
        
        # 风险评估
        evaluation_log_id = await logger.log_risk_evaluation(
            risk_id=risk_data.risk_id,
            evaluation_criteria={"volatility": 0.2, "correlation": 0.5},
            evaluation_result={"var": 0.05, "expected_shortfall": 0.08}
        )
        print(f"风险评估日志ID: {evaluation_log_id}")
        
        # 风险控制
        control_log_id = await logger.log_risk_control(
            risk_id=risk_data.risk_id,
            control_measures=["止损设置", "仓位限制", "分散投资"],
            implementation_status="implemented",
            effectiveness=0.7
        )
        print(f"风险控制日志ID: {control_log_id}")
        
        # 示例4: 记录市场分析
        print("\n=== 市场分析日志示例 ===")
        
        # 市场趋势分析
        market_data = MarketAnalysis(
            market_segment="科技股",
            trend="上升",
            trend_strength=0.75,
            opportunities=["AI技术发展", "数字化转型"],
            threats=["监管风险", "竞争加剧"],
            indicators={"pe_ratio": 25.5, "growth_rate": 0.15}
        )
        trend_log_id = await logger.log_market_trend(market_data)
        print(f"市场趋势日志ID: {trend_log_id}")
        
        # 机会识别
        opportunity_log_id = await logger.log_opportunity_identification(
            opportunity_id="opp_001",
            opportunity_type="技术突破",
            description="人工智能技术突破带来新机会",
            potential_value=1000000,
            probability=0.6
        )
        print(f"机会识别日志ID: {opportunity_log_id}")
        
        # 威胁分析
        threat_log_id = await logger.log_threat_analysis(
            threat_id="threat_001",
            threat_type="监管变化",
            description="新的监管政策可能影响业务",
            severity=0.7,
            likelihood=0.4,
            mitigation_strategies=["合规调整", "政策跟踪"]
        )
        print(f"威胁分析日志ID: {threat_log_id}")
        
        # 示例5: 记录决策解释
        print("\n=== 决策解释日志示例 ===")
        
        # 决策解释
        explanation_data = DecisionExplanation(
            reason="基于技术分析信号",
            rationale="移动平均线显示强烈买入信号",
            evidence=["MA10上穿MA20", "成交量放大", "RSI超卖"],
            alternatives=["等待更明确信号", "分批建仓"],
            impact_analysis={"预期收益": 0.05, "风险水平": "中等"}
        )
        explanation_log_id = await logger.log_decision_explanation(explanation_data)
        print(f"决策解释日志ID: {explanation_log_id}")
        
        # 决策理由
        rationale_log_id = await logger.log_decision_rationale(
            decision_id=result_data.result_id,
            rationale="技术指标显示强烈买入信号",
            supporting_evidence=["MA交叉", "成交量确认"],
            decision_criteria={"signal_strength": 0.85, "risk_reward": 2.0}
        )
        print(f"决策理由日志ID: {rationale_log_id}")
        
        # 决策影响
        impact_log_id = await logger.log_decision_impact(
            decision_id=result_data.result_id,
            impact_analysis={"portfolio_impact": 0.02, "risk_change": 0.01},
            stakeholder_impacts={
                "traders": {"impact": "positive", "description": "新的交易机会"},
                "risk_managers": {"impact": "neutral", "description": "需要监控风险"}
            }
        )
        print(f"决策影响日志ID: {impact_log_id}")
        
        # 示例6: 记录决策效果评估
        print("\n=== 决策效果评估日志示例 ===")
        
        # 决策评估
        evaluation_data = EffectEvaluation(
            decision_id=result_data.result_id,
            expected_outcome="获得5%收益",
            actual_outcome="获得4.8%收益",
            performance_metrics={"return": 0.048, "max_drawdown": 0.02},
            lessons_learned=["信号准确性较高", "需要更好的时机把握"],
            recommendations=["优化入场时机", "加强风险控制"]
        )
        evaluation_log_id = await logger.log_decision_evaluation(evaluation_data)
        print(f"决策评估日志ID: {evaluation_log_id}")
        
        # 性能指标
        metrics_log_id = await logger.log_performance_metrics(
            decision_id=result_data.result_id,
            metrics={"return": 0.048, "sharpe_ratio": 1.2, "max_drawdown": 0.02},
            measurement_period="30天",
            benchmark_data={"market_return": 0.03, "benchmark_sharpe": 1.0}
        )
        print(f"性能指标日志ID: {metrics_log_id}")
        
        # 经验教训
        lessons_log_id = await logger.log_lessons_learned(
            decision_id=result_data.result_id,
            lessons=["技术分析在当前市场环境下效果良好"],
            improvement_suggestions=["结合基本面分析", "增加确认信号"],
            best_practices=["严格止损", "分批建仓"]
        )
        print(f"经验教训日志ID: {lessons_log_id}")
        
        # 示例7: 使用上下文管理器
        print("\n=== 上下文管理器示例 ===")
        
        with logger.decision_context("context_test_001") as ctx:
            # 在决策上下文中执行操作
            print(f"决策上下文开始: {ctx['decision_id']}")
            await asyncio.sleep(0.1)  # 模拟决策过程
            print("决策过程执行中...")
        
        with logger.strategy_context("strategy_context_001", StrategyType.BALANCED) as ctx:
            print(f"策略上下文开始: {ctx['strategy_id']}")
            await asyncio.sleep(0.1)  # 模拟策略执行
            print("策略执行中...")
        
        # 示例8: 查询和统计
        print("\n=== 查询和统计示例 ===")
        
        # 获取统计信息
        stats = logger.get_statistics()
        print(f"日志统计: {stats}")
        
        # 获取性能摘要
        performance = logger.get_performance_summary()
        print(f"性能摘要: {performance}")
        
        # 查询决策历史
        history = await logger.get_decision_history(limit=10)
        print(f"决策历史条数: {len(history)}")
        
        # 获取策略性能
        strategy_perf = await logger.get_strategy_performance()
        print(f"策略性能: {strategy_perf}")
        
        # 获取风险摘要
        risk_summary = await logger.get_risk_summary()
        print(f"风险摘要: {risk_summary}")
        
        print("\n=== 所有示例执行完成 ===")
        
    except Exception as e:
        print(f"示例执行失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 停止日志记录器
        await logger.stop()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())


# 扩展功能类和方法

class DecisionAnalytics:
    """决策分析器类"""
    
    def __init__(self, logger: DecisionLogger):
        self.logger = logger
        self._cache = {}
    
    async def analyze_decision_patterns(self, 
                                      time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """分析决策模式"""
        try:
            query = {'log_type': {'$in': ['decision_input', 'decision_result']}}
            if time_range:
                query['timestamp'] = {
                    '$gte': time_range[0].isoformat(),
                    '$lte': time_range[1].isoformat()
                }
            
            logs = await self.logger.storage.retrieve(query)
            
            # 按决策类型分组分析
            patterns = defaultdict(lambda: {
                'count': 0,
                'success_rate': 0.0,
                'avg_confidence': 0.0,
                'avg_execution_time': 0.0
            })
            
            for log in logs:
                if log['log_type'] == 'decision_result':
                    result = log['data']['result']
                    decision_type = result['decision_type']
                    patterns[decision_type]['count'] += 1
                    patterns[decision_type]['success_rate'] += 1 if result['success'] else 0
                    patterns[decision_type]['avg_confidence'] += result['confidence']
                    patterns[decision_type]['avg_execution_time'] += result['execution_time']
            
            # 计算平均值
            for pattern in patterns.values():
                if pattern['count'] > 0:
                    pattern['success_rate'] /= pattern['count']
                    pattern['avg_confidence'] /= pattern['count']
                    pattern['avg_execution_time'] /= pattern['count']
            
            return dict(patterns)
            
        except Exception as e:
            self.logger.logger.error(f"决策模式分析失败: {e}")
            return {}
    
    async def analyze_strategy_effectiveness(self, 
                                           strategy_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """分析策略有效性"""
        try:
            query = {'log_type': 'strategy_result'}
            if strategy_types:
                query['data.strategy_type'] = {'$in': strategy_types}
            
            logs = await self.logger.storage.retrieve(query)
            
            effectiveness = defaultdict(lambda: {
                'total_attempts': 0,
                'successful_attempts': 0,
                'avg_performance': 0.0,
                'performance_values': []
            })
            
            for log in logs:
                strategy_type = log['data'].get('strategy_type', 'unknown')
                effectiveness[strategy_type]['total_attempts'] += 1
                
                if log['data']['success']:
                    effectiveness[strategy_type]['successful_attempts'] += 1
                
                if log['data'].get('performance_metrics'):
                    metrics = log['data']['performance_metrics']
                    for value in metrics.values():
                        if isinstance(value, (int, float)):
                            effectiveness[strategy_type]['performance_values'].append(value)
            
            # 计算统计数据
            for stats in effectiveness.values():
                if stats['total_attempts'] > 0:
                    stats['success_rate'] = stats['successful_attempts'] / stats['total_attempts']
                if stats['performance_values']:
                    stats['avg_performance'] = mean(stats['performance_values'])
                    stats['max_performance'] = max(stats['performance_values'])
                    stats['min_performance'] = min(stats['performance_values'])
                    stats['std_performance'] = stdev(stats['performance_values']) if len(stats['performance_values']) > 1 else 0.0
            
            return dict(effectiveness)
            
        except Exception as e:
            self.logger.logger.error(f"策略有效性分析失败: {e}")
            return {}
    
    async def analyze_risk_trends(self, 
                                time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, Any]:
        """分析风险趋势"""
        try:
            query = {'log_type': 'risk_identification'}
            if time_range:
                query['timestamp'] = {
                    '$gte': time_range[0].isoformat(),
                    '$lte': time_range[1].isoformat()
                }
            
            logs = await self.logger.storage.retrieve(query)
            
            # 按风险级别分组
            risk_trends = defaultdict(lambda: {
                'count': 0,
                'total_score': 0.0,
                'avg_probability': 0.0,
                'avg_impact': 0.0,
                'types': defaultdict(int)
            })
            
            probabilities = []
            impacts = []
            
            for log in logs:
                risk = log['data']['risk']
                level = risk['level']
                risk_trends[level]['count'] += 1
                risk_trends[level]['total_score'] += risk['score']
                risk_trends[level]['types'][risk['risk_type']] += 1
                
                probabilities.append(risk['probability'])
                impacts.append(risk['impact'])
            
            # 计算统计数据
            for trends in risk_trends.values():
                if trends['count'] > 0:
                    trends['avg_score'] = trends['total_score'] / trends['count']
            
            if probabilities:
                risk_trends['overall']['avg_probability'] = mean(probabilities)
                risk_trends['overall']['avg_impact'] = mean(impacts)
                risk_trends['overall']['probability_std'] = stdev(probabilities) if len(probabilities) > 1 else 0.0
                risk_trends['overall']['impact_std'] = stdev(impacts) if len(impacts) > 1 else 0.0
            
            return dict(risk_trends)
            
        except Exception as e:
            self.logger.logger.error(f"风险趋势分析失败: {e}")
            return {}
    
    async def generate_decision_report(self, 
                                     time_range: Optional[Tuple[datetime, datetime]] = None,
                                     include_analysis: bool = True) -> Dict[str, Any]:
        """生成决策报告"""
        try:
            report = {
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'time_range': time_range,
                'summary': {},
                'details': {},
                'recommendations': []
            }
            
            # 获取基础统计
            if time_range:
                summary_query = {'timestamp': {
                    '$gte': time_range[0].isoformat(),
                    '$lte': time_range[1].isoformat()
                }}
            else:
                summary_query = {}
            
            all_logs = await self.logger.storage.retrieve(summary_query)
            
            # 基础统计
            report['summary'] = {
                'total_logs': len(all_logs),
                'decision_count': len([log for log in all_logs if 'decision' in log['log_type']]),
                'strategy_count': len([log for log in all_logs if 'strategy' in log['log_type']]),
                'risk_count': len([log for log in all_logs if 'risk' in log['log_type']]),
                'success_rate': 0.0,
                'error_count': len([log for log in all_logs if log['log_type'] == 'error'])
            }
            
            # 计算成功率
            decision_results = [log for log in all_logs if log['log_type'] == 'decision_result']
            if decision_results:
                successful = sum(1 for log in decision_results if log['data']['result']['success'])
                report['summary']['success_rate'] = successful / len(decision_results)
            
            # 详细分析
            if include_analysis:
                report['details'] = {
                    'decision_patterns': await self.analyze_decision_patterns(time_range),
                    'strategy_effectiveness': await self.analyze_strategy_effectiveness(),
                    'risk_trends': await self.analyze_risk_trends(time_range)
                }
                
                # 生成建议
                report['recommendations'] = self._generate_recommendations(report['details'])
            
            return report
            
        except Exception as e:
            self.logger.logger.error(f"生成决策报告失败: {e}")
            return {}
    
    def _generate_recommendations(self, analysis_data: Dict[str, Any]) -> List[str]:
        """基于分析数据生成建议"""
        recommendations = []
        
        # 基于决策模式生成建议
        patterns = analysis_data.get('decision_patterns', {})
        for decision_type, stats in patterns.items():
            if stats['success_rate'] < 0.6:
                recommendations.append(f"优化{decision_type}决策流程，当前成功率仅为{stats['success_rate']:.2%}")
            
            if stats['avg_confidence'] < 0.7:
                recommendations.append(f"提高{decision_type}决策的置信度，当前平均置信度为{stats['avg_confidence']:.2f}")
        
        # 基于策略有效性生成建议
        effectiveness = analysis_data.get('strategy_effectiveness', {})
        for strategy_type, stats in effectiveness.items():
            if stats.get('success_rate', 0) < 0.5:
                recommendations.append(f"重新评估{strategy_type}策略，当前成功率较低")
        
        # 基于风险趋势生成建议
        risk_trends = analysis_data.get('risk_trends', {})
        high_risk_count = risk_trends.get('high', {}).get('count', 0)
        if high_risk_count > 5:
            recommendations.append("高风险事件频发，建议加强风险控制措施")
        
        return recommendations


class LogCompressor:
    """日志压缩器类"""
    
    def __init__(self, compression_level: int = 6):
        self.compression_level = compression_level
    
    def compress_logs(self, logs: List[Dict[str, Any]]) -> bytes:
        """压缩日志数据"""
        try:
            json_data = json.dumps(logs, ensure_ascii=False).encode('utf-8')
            return gzip.compress(json_data, compresslevel=self.compression_level)
        except Exception as e:
            raise ValueError(f"日志压缩失败: {e}")
    
    def decompress_logs(self, compressed_data: bytes) -> List[Dict[str, Any]]:
        """解压缩日志数据"""
        try:
            json_data = gzip.decompress(compressed_data).decode('utf-8')
            return json.loads(json_data)
        except Exception as e:
            raise ValueError(f"日志解压缩失败: {e}")
    
    def compress_to_file(self, logs: List[Dict[str, Any]], file_path: str):
        """压缩日志到文件"""
        try:
            compressed_data = self.compress_logs(logs)
            with open(file_path, 'wb') as f:
                f.write(compressed_data)
        except Exception as e:
            raise ValueError(f"压缩日志到文件失败: {e}")
    
    def decompress_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """从文件解压缩日志"""
        try:
            with open(file_path, 'rb') as f:
                compressed_data = f.read()
            return self.decompress_logs(compressed_data)
        except Exception as e:
            raise ValueError(f"从文件解压缩日志失败: {e}")


class LogValidator:
    """日志验证器类"""
    
    @staticmethod
    def validate_log_entry(log_entry: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证日志条目"""
        errors = []
        
        required_fields = ['id', 'timestamp', 'log_type', 'data']
        for field in required_fields:
            if field not in log_entry:
                errors.append(f"缺少必需字段: {field}")
        
        # 验证时间戳格式
        if 'timestamp' in log_entry:
            try:
                datetime.fromisoformat(log_entry['timestamp'].replace('Z', '+00:00'))
            except ValueError:
                errors.append("时间戳格式无效")
        
        # 验证日志类型
        valid_log_types = [
            'decision_input', 'decision_logic', 'decision_result',
            'strategy_selection', 'strategy_execution', 'strategy_result',
            'risk_identification', 'risk_evaluation', 'risk_control',
            'market_trend', 'opportunity_identification', 'threat_analysis',
            'decision_explanation', 'decision_rationale', 'decision_impact',
            'decision_evaluation', 'performance_metrics', 'lessons_learned'
        ]
        
        if 'log_type' in log_entry and log_entry['log_type'] not in valid_log_types:
            errors.append(f"无效的日志类型: {log_entry['log_type']}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_decision_input(input_data: DecisionInput) -> Tuple[bool, List[str]]:
        """验证决策输入数据"""
        errors = []
        
        if not input_data.input_type:
            errors.append("输入类型不能为空")
        
        if input_data.priority < 0:
            errors.append("优先级不能为负数")
        
        if not isinstance(input_data.data, dict):
            errors.append("数据必须是字典格式")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def validate_decision_result(result_data: DecisionResult) -> Tuple[bool, List[str]]:
        """验证决策结果数据"""
        errors = []
        
        if not 0 <= result_data.confidence <= 1:
            errors.append("置信度必须在0-1之间")
        
        if result_data.execution_time < 0:
            errors.append("执行时间不能为负数")
        
        if not isinstance(result_data.success, bool):
            errors.append("成功标志必须是布尔值")
        
        return len(errors) == 0, errors


class LogExporter:
    """日志导出器类"""
    
    def __init__(self, logger: DecisionLogger):
        self.logger = logger
    
    async def export_to_csv(self, 
                          query: Dict[str, Any],
                          output_path: str,
                          fields: Optional[List[str]] = None) -> str:
        """导出到CSV文件"""
        try:
            logs = await self.logger.storage.retrieve(query)
            
            if not logs:
                raise ValueError("没有找到匹配的日志")
            
            # 确定导出字段
            if fields is None:
                fields = ['id', 'timestamp', 'log_type']
            
            # 提取数据
            rows = []
            for log in logs:
                row = {}
                for field in fields:
                    if field in log:
                        row[field] = log[field]
                    elif field == 'log_type':
                        row[field] = log.get('log_type', '')
                    elif field == 'timestamp':
                        row[field] = log.get('timestamp', '')
                    else:
                        row[field] = log.get('data', {}).get(field, '')
                rows.append(row)
            
            # 写入CSV
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            
            return output_path
            
        except Exception as e:
            self.logger.logger.error(f"导出CSV失败: {e}")
            raise
    
    async def export_to_sqlite(self, 
                             query: Dict[str, Any],
                             db_path: str,
                             table_name: str = "logs") -> str:
        """导出到SQLite数据库"""
        try:
            logs = await self.logger.storage.retrieve(query)
            
            if not logs:
                raise ValueError("没有找到匹配的日志")
            
            # 连接数据库
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # 创建表
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    log_type TEXT,
                    data TEXT
                )
            """)
            
            # 插入数据
            for log in logs:
                cursor.execute(f"""
                    INSERT OR REPLACE INTO {table_name} (id, timestamp, log_type, data)
                    VALUES (?, ?, ?, ?)
                """, (
                    log['id'],
                    log['timestamp'],
                    log['log_type'],
                    json.dumps(log['data'], ensure_ascii=False)
                ))
            
            conn.commit()
            conn.close()
            
            return db_path
            
        except Exception as e:
            self.logger.logger.error(f"导出SQLite失败: {e}")
            raise
    
    async def export_summary_report(self, 
                                  output_path: str,
                                  time_range: Optional[Tuple[datetime, datetime]] = None) -> str:
        """导出摘要报告"""
        try:
            # 生成报告
            analytics = DecisionAnalytics(self.logger)
            report = await analytics.generate_decision_report(time_range)
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=== 决策日志摘要报告 ===\n\n")
                f.write(f"生成时间: {report['timestamp']}\n")
                
                if time_range:
                    f.write(f"时间范围: {time_range[0]} 至 {time_range[1]}\n")
                
                f.write("\n=== 概要统计 ===\n")
                for key, value in report['summary'].items():
                    f.write(f"{key}: {value}\n")
                
                f.write("\n=== 建议 ===\n")
                for i, recommendation in enumerate(report['recommendations'], 1):
                    f.write(f"{i}. {recommendation}\n")
            
            return output_path
            
        except Exception as e:
            self.logger.logger.error(f"导出摘要报告失败: {e}")
            raise


class LogMonitor:
    """日志监控器类"""
    
    def __init__(self, logger: DecisionLogger):
        self.logger = logger
        self._alerts = []
        self._thresholds = {}
        self._monitoring_active = False
    
    def set_alert_threshold(self, metric: str, threshold: float, comparison: str = 'greater'):
        """设置告警阈值"""
        self._thresholds[metric] = {
            'threshold': threshold,
            'comparison': comparison  # 'greater', 'less', 'equal'
        }
    
    async def start_monitoring(self, interval: float = 60.0):
        """开始监控"""
        self._monitoring_active = True
        while self._monitoring_active:
            try:
                await self._check_metrics()
                await asyncio.sleep(interval)
            except Exception as e:
                self.logger.logger.error(f"监控过程出错: {e}")
                await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """停止监控"""
        self._monitoring_active = False
    
    async def _check_metrics(self):
        """检查指标"""
        stats = self.logger.get_statistics()
        performance = self.logger.get_performance_summary()
        
        # 检查错误率
        if 'error_rate' in self._thresholds:
            threshold = self._thresholds['error_rate']
            error_rate = performance.get('error_rate', 0.0)
            
            if self._compare_values(error_rate, threshold['threshold'], threshold['comparison']):
                await self._trigger_alert('error_rate', error_rate, threshold['threshold'])
        
        # 检查成功率
        if 'min_success_rate' in self._thresholds:
            threshold = self._thresholds['min_success_rate']
            success_rate = performance.get('decision_success_rate', 0.0)
            
            if self._compare_values(success_rate, threshold['threshold'], threshold['comparison']):
                await self._trigger_alert('success_rate', success_rate, threshold['threshold'])
    
    def _compare_values(self, value: float, threshold: float, comparison: str) -> bool:
        """比较值与阈值"""
        if comparison == 'greater':
            return value > threshold
        elif comparison == 'less':
            return value < threshold
        elif comparison == 'equal':
            return abs(value - threshold) < 0.001
        return False
    
    async def _trigger_alert(self, metric: str, value: float, threshold: float):
        """触发告警"""
        alert = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metric': metric,
            'value': value,
            'threshold': threshold,
            'message': f"{metric} 告警: 当前值 {value} 超过阈值 {threshold}"
        }
        
        self._alerts.append(alert)
        self.logger.logger.warning(f"告警: {alert['message']}")
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict[str, Any]]:
        """获取最近的告警"""
        return self._alerts[-count:] if self._alerts else []


class LogArchiver:
    """日志归档器类"""
    
    def __init__(self, logger: DecisionLogger):
        self.logger = logger
        self.archive_path = Path("archive")
        self.archive_path.mkdir(exist_ok=True)
    
    async def archive_old_logs(self, days: int = 30) -> int:
        """归档旧日志"""
        try:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            query = {'timestamp': {'$lt': cutoff_date.isoformat()}}
            
            # 获取要归档的日志
            logs_to_archive = await self.logger.storage.retrieve(query)
            
            if not logs_to_archive:
                return 0
            
            # 按月份分组
            logs_by_month = defaultdict(list)
            for log in logs_to_archive:
                timestamp = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
                month_key = timestamp.strftime('%Y_%m')
                logs_by_month[month_key].append(log)
            
            archived_count = 0
            compressor = LogCompressor()
            
            # 归档到压缩文件
            for month, logs in logs_by_month.items():
                archive_file = self.archive_path / f"logs_{month}.json.gz"
                compressor.compress_to_file(logs, str(archive_file))
                archived_count += len(logs)
            
            # 从存储中删除已归档的日志
            if hasattr(self.logger.storage, 'batch_delete'):
                log_ids = [log['id'] for log in logs_to_archive]
                await self.logger.storage.batch_delete(log_ids)
            
            self.logger.logger.info(f"归档完成: {archived_count} 条日志")
            return archived_count
            
        except Exception as e:
            self.logger.logger.error(f"归档旧日志失败: {e}")
            return 0
    
    async def restore_archived_logs(self, month: str) -> int:
        """恢复归档的日志"""
        try:
            archive_file = self.archive_path / f"logs_{month}.json.gz"
            
            if not archive_file.exists():
                raise FileNotFoundError(f"归档文件不存在: {archive_file}")
            
            # 解压缩日志
            compressor = LogCompressor()
            restored_logs = compressor.decompress_from_file(str(archive_file))
            
            # 重新存储日志
            for log in restored_logs:
                await self.logger.storage.store(log)
            
            self.logger.logger.info(f"恢复完成: {len(restored_logs)} 条日志")
            return len(restored_logs)
            
        except Exception as e:
            self.logger.logger.error(f"恢复归档日志失败: {e}")
            return 0


# 装饰器和工具函数

def log_execution_time(func):
    """记录执行时间的装饰器"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.time() - start_time
            if hasattr(args[0], 'logger') and hasattr(args[0].logger, 'logger'):
                args[0].logger.logger.debug(f"{func.__name__} 执行时间: {execution_time:.3f}秒")
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            if hasattr(args[0], 'logger') and hasattr(args[0].logger, 'logger'):
                args[0].logger.logger.error(f"{func.__name__} 执行失败，耗时: {execution_time:.3f}秒，错误: {e}")
            raise
    return async_wrapper


def validate_input(validation_func):
    """输入验证装饰器"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 假设第一个参数是self，第二个是需要验证的数据
            if len(args) > 1:
                is_valid, errors = validation_func(args[1])
                if not is_valid:
                    raise ValueError(f"输入验证失败: {'; '.join(errors)}")
            return await func(*args, **kwargs)
        return async_wrapper
    return decorator


def cache_result(maxsize=128):
    """结果缓存装饰器"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # 简单的缓存实现
            cache_key = str(args) + str(sorted(kwargs.items()))
            if not hasattr(async_wrapper, '_cache'):
                async_wrapper._cache = {}
            
            if cache_key in async_wrapper._cache:
                return async_wrapper._cache[cache_key]
            
            result = await func(*args, **kwargs)
            
            # 管理缓存大小
            if len(async_wrapper._cache) >= maxsize:
                async_wrapper._cache.pop(next(iter(async_wrapper._cache)))
            
            async_wrapper._cache[cache_key] = result
            return result
        return async_wrapper
    return decorator


@lru_cache(maxsize=100)
def get_log_type_stats(log_type: str) -> Dict[str, Any]:
    """获取日志类型统计（缓存版本）"""
    # 这里可以实现更复杂的统计逻辑
    return {
        'log_type': log_type,
        'estimated_size': len(log_type) * 100,
        'complexity_score': len(log_type) % 10
    }


def create_decision_summary(decision_logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """创建决策摘要"""
    if not decision_logs:
        return {}
    
    summary = {
        'total_decisions': len(decision_logs),
        'successful_decisions': 0,
        'failed_decisions': 0,
        'avg_confidence': 0.0,
        'avg_execution_time': 0.0,
        'decision_types': defaultdict(int),
        'time_range': {
            'start': None,
            'end': None
        }
    }
    
    confidences = []
    execution_times = []
    
    for log in decision_logs:
        if log['log_type'] == 'decision_result':
            result = log['data']['result']
            
            if result['success']:
                summary['successful_decisions'] += 1
            else:
                summary['failed_decisions'] += 1
            
            summary['decision_types'][result['decision_type']] += 1
            confidences.append(result['confidence'])
            execution_times.append(result['execution_time'])
            
            # 更新时间范围
            timestamp = datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
            if summary['time_range']['start'] is None or timestamp < summary['time_range']['start']:
                summary['time_range']['start'] = timestamp
            if summary['time_range']['end'] is None or timestamp > summary['time_range']['end']:
                summary['time_range']['end'] = timestamp
    
    if confidences:
        summary['avg_confidence'] = mean(confidences)
    if execution_times:
        summary['avg_execution_time'] = mean(execution_times)
    
    return summary


def calculate_risk_score(probability: float, impact: float) -> float:
    """计算风险评分"""
    return probability * impact


def calculate_performance_score(metrics: Dict[str, float], weights: Optional[Dict[str, float]] = None) -> float:
    """计算性能评分"""
    if not metrics:
        return 0.0
    
    if weights is None:
        weights = {key: 1.0 for key in metrics.keys()}
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    for metric, value in metrics.items():
        weight = weights.get(metric, 1.0)
        weighted_sum += value * weight
        total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def generate_correlation_matrix(data: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """生成相关性矩阵"""
    if not data or not data[0]:
        return {}
    
    metrics = list(data[0].keys())
    n = len(metrics)
    correlation_matrix = {}
    
    for i, metric1 in enumerate(metrics):
        correlation_matrix[metric1] = {}
        for j, metric2 in enumerate(metrics):
            if i == j:
                correlation_matrix[metric1][metric2] = 1.0
            else:
                values1 = [d.get(metric1, 0) for d in data]
                values2 = [d.get(metric2, 0) for d in data]
                
                # 计算皮尔逊相关系数
                correlation = calculate_pearson_correlation(values1, values2)
                correlation_matrix[metric1][metric2] = correlation
    
    return correlation_matrix


def calculate_pearson_correlation(x: List[float], y: List[float]) -> float:
    """计算皮尔逊相关系数"""
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)
    sum_y2 = sum(yi * yi for yi in y)
    
    numerator = n * sum_xy - sum_x * sum_y
    denominator = sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def format_file_size(size_bytes: int) -> str:
    """格式化文件大小"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"


def validate_timestamp(timestamp_str: str) -> bool:
    """验证时间戳格式"""
    try:
        datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False


def sanitize_filename(filename: str) -> str:
    """清理文件名"""
    # 移除或替换非法字符
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # 限制长度
    if len(filename) > 200:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:200-len(ext)-1] + '.' + ext if ext else name[:200]
    return filename


def create_backup_name(original_path: str) -> str:
    """创建备份文件名"""
    path = Path(original_path)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{path.stem}_backup_{timestamp}{path.suffix}"


# 扩展DecisionLogger类的方法
DecisionLogger.analytics = property(lambda self: DecisionAnalytics(self))
DecisionLogger.compressor = property(lambda self: LogCompressor())
DecisionLogger.validator = property(lambda self: LogValidator())
DecisionLogger.exporter = property(lambda self: LogExporter(self))
DecisionLogger.monitor = property(lambda self: LogMonitor(self))
DecisionLogger.archiver = property(lambda self: LogArchiver(self))


# 高级功能扩展

class DecisionPredictor:
    """决策预测器类"""
    
    def __init__(self, logger: DecisionLogger):
        self.logger = logger
        self._prediction_cache = {}
    
    async def predict_decision_outcome(self, 
                                     decision_context: Dict[str, Any]) -> Dict[str, Any]:
        """预测决策结果"""
        try:
            # 基于历史数据预测
            historical_data = await self._get_historical_performance(decision_context)
            
            prediction = {
                'success_probability': 0.0,
                'expected_confidence': 0.0,
                'estimated_execution_time': 0.0,
                'risk_factors': [],
                'confidence_score': 0.0
            }
            
            if historical_data:
                # 计算成功率
                successful = sum(1 for h in historical_data if h.get('success', False))
                prediction['success_probability'] = successful / len(historical_data)
                
                # 计算平均置信度
                confidences = [h.get('confidence', 0) for h in historical_data if h.get('confidence')]
                if confidences:
                    prediction['expected_confidence'] = mean(confidences)
                
                # 计算平均执行时间
                execution_times = [h.get('execution_time', 0) for h in historical_data if h.get('execution_time')]
                if execution_times:
                    prediction['estimated_execution_time'] = mean(execution_times)
                
                # 识别风险因素
                prediction['risk_factors'] = self._identify_risk_factors(historical_data)
                
                # 计算预测置信度
                prediction['confidence_score'] = self._calculate_prediction_confidence(historical_data)
            
            return prediction
            
        except Exception as e:
            self.logger.logger.error(f"决策预测失败: {e}")
            return {}
    
    async def _get_historical_performance(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """获取历史性能数据"""
        # 基于上下文查询历史数据
        query = {'log_type': 'decision_result'}
        
        # 根据上下文类型过滤
        if 'decision_type' in context:
            query['data.result.decision_type'] = context['decision_type']
        
        return await self.logger.storage.retrieve(query)
    
    def _identify_risk_factors(self, historical_data: List[Dict[str, Any]]) -> List[str]:
        """识别风险因素"""
        risk_factors = []
        
        # 分析失败案例
        failed_cases = [h for h in historical_data if not h.get('success', True)]
        
        if len(failed_cases) / len(historical_data) > 0.3:
            risk_factors.append("历史失败率较高")
        
        # 分析执行时间异常
        execution_times = [h.get('execution_time', 0) for h in historical_data]
        if execution_times:
            avg_time = mean(execution_times)
            max_time = max(execution_times)
            if max_time > avg_time * 3:
                risk_factors.append("执行时间波动较大")
        
        # 分析置信度低的情况
        low_confidence_cases = [h for h in historical_data if h.get('confidence', 1) < 0.6]
        if len(low_confidence_cases) / len(historical_data) > 0.2:
            risk_factors.append("历史置信度较低")
        
        return risk_factors
    
    def _calculate_prediction_confidence(self, historical_data: List[Dict[str, Any]]) -> float:
        """计算预测置信度"""
        if not historical_data:
            return 0.0
        
        # 基于数据量和一致性计算置信度
        data_size_score = min(len(historical_data) / 100, 1.0)  # 数据量评分
        
        # 计算一致性评分
        if len(historical_data) > 1:
            success_rates = [1 if h.get('success', False) else 0 for h in historical_data]
            consistency_score = 1.0 - stdev(success_rates) if len(success_rates) > 1 else 1.0
        else:
            consistency_score = 0.5
        
        return (data_size_score + consistency_score) / 2


class DecisionOptimizer:
    """决策优化器类"""
    
    def __init__(self, logger: DecisionLogger):
        self.logger = logger
    
    async def optimize_decision_parameters(self, 
                                         decision_type: str,
                                         target_metric: str = "success_rate") -> Dict[str, Any]:
        """优化决策参数"""
        try:
            # 获取历史决策数据
            query = {
                'log_type': 'decision_result',
                'data.result.decision_type': decision_type
            }
            
            logs = await self.logger.storage.retrieve(query)
            
            if not logs:
                return {'error': '没有足够的历史数据'}
            
            # 分析参数与性能的关系
            parameter_analysis = self._analyze_parameter_impact(logs, target_metric)
            
            # 生成优化建议
            optimization_suggestions = self._generate_optimization_suggestions(parameter_analysis)
            
            return {
                'decision_type': decision_type,
                'target_metric': target_metric,
                'parameter_analysis': parameter_analysis,
                'optimization_suggestions': optimization_suggestions,
                'data_points': len(logs)
            }
            
        except Exception as e:
            self.logger.logger.error(f"决策参数优化失败: {e}")
            return {}
    
    def _analyze_parameter_impact(self, logs: List[Dict[str, Any]], target_metric: str) -> Dict[str, Any]:
        """分析参数影响"""
        analysis = {
            'correlations': {},
            'optimal_ranges': {},
            'performance_by_parameter': {}
        }
        
        # 提取参数和性能数据
        parameter_data = []
        for log in logs:
            if 'decision_logic' in log.get('data', {}):
                logic = log['data']['decision_logic']['logic']
                result = log['data']['result']
                
                data_point = {
                    'parameters': logic.get('parameters', {}),
                    'confidence': result.get('confidence', 0),
                    'success': result.get('success', False),
                    'execution_time': result.get('execution_time', 0)
                }
                parameter_data.append(data_point)
        
        # 分析参数与性能的相关性
        for param_name in set().union(*(d['parameters'].keys() for d in parameter_data if d['parameters'])):
            param_values = [d['parameters'].get(param_name, 0) for d in parameter_data]
            target_values = [d.get(target_metric, 0) for d in parameter_data]
            
            correlation = calculate_pearson_correlation(param_values, target_values)
            analysis['correlations'][param_name] = correlation
        
        return analysis
    
    def _generate_optimization_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        suggestions = []
        
        correlations = analysis.get('correlations', {})
        
        for param, correlation in correlations.items():
            if abs(correlation) > 0.5:
                if correlation > 0:
                    suggestions.append(f"增加{param}参数值可能提高性能 (相关性: {correlation:.3f})")
                else:
                    suggestions.append(f"减少{param}参数值可能提高性能 (相关性: {correlation:.3f})")
        
        if not suggestions:
            suggestions.append("当前参数设置相对合理，建议保持现有配置")
        
        return suggestions


class DecisionSimulator:
    """决策模拟器类"""
    
    def __init__(self, logger: DecisionLogger):
        self.logger = logger
    
    async def simulate_decision_scenario(self, 
                                       scenario: Dict[str, Any],
                                       iterations: int = 100) -> Dict[str, Any]:
        """模拟决策场景"""
        try:
            results = []
            
            for i in range(iterations):
                # 模拟单次决策
                simulation_result = await self._simulate_single_decision(scenario)
                results.append(simulation_result)
            
            # 统计分析结果
            simulation_stats = self._analyze_simulation_results(results)
            
            return {
                'scenario': scenario,
                'iterations': iterations,
                'results': results,
                'statistics': simulation_stats,
                'success_probability': simulation_stats.get('success_rate', 0.0)
            }
            
        except Exception as e:
            self.logger.logger.error(f"决策场景模拟失败: {e}")
            return {}
    
    async def _simulate_single_decision(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """模拟单次决策"""
        # 简化的模拟逻辑
        base_success_rate = scenario.get('base_success_rate', 0.7)
        confidence_factor = scenario.get('confidence_factor', 1.0)
        
        # 添加随机性
        import random
        random.seed(time.time())
        
        success = random.random() < base_success_rate * confidence_factor
        confidence = min(1.0, max(0.0, random.gauss(0.8, 0.1) * confidence_factor))
        execution_time = max(0.01, random.gauss(0.1, 0.05))
        
        return {
            'success': success,
            'confidence': confidence,
            'execution_time': execution_time,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    def _analyze_simulation_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析模拟结果"""
        if not results:
            return {}
        
        successful_results = [r for r in results if r['success']]
        
        stats = {
            'total_simulations': len(results),
            'successful_simulations': len(successful_results),
            'success_rate': len(successful_results) / len(results),
            'avg_confidence': mean([r['confidence'] for r in results]),
            'avg_execution_time': mean([r['execution_time'] for r in results]),
            'confidence_std': stdev([r['confidence'] for r in results]) if len(results) > 1 else 0.0,
            'execution_time_std': stdev([r['execution_time'] for r in results]) if len(results) > 1 else 0.0
        }
        
        return stats


# 高级测试用例

async def advanced_example_usage():
    """高级使用示例"""
    
    # 创建决策日志记录器
    logger = DecisionLogger(
        storage_type="memory",
        storage_config={"max_size": 1000},
        enable_async=True,
        log_level=LogLevel.INFO
    )
    
    try:
        # 启动日志记录器
        await logger.start()
        
        print("=== 高级功能示例 ===")
        
        # 示例1: 批量决策日志记录
        print("\n1. 批量决策日志记录")
        decisions = []
        for i in range(10):
            decision = {
                'input': {
                    'input_type': f'type_{i}',
                    'data': {'value': i * 10},
                    'source': 'batch_test',
                    'priority': i % 3
                },
                'logic': {
                    'algorithm': f'algorithm_{i}',
                    'confidence': 0.7 + i * 0.03,
                    'reasoning': f'批量测试逻辑 {i}'
                },
                'result': {
                    'decision_type': DecisionType.TACTICAL,
                    'outcome': f'结果_{i}',
                    'success': i % 3 != 0,  # 2/3 成功率
                    'value': i * 100,
                    'confidence': 0.7 + i * 0.03,
                    'execution_time': 0.01 * i
                }
            }
            decisions.append(decision)
        
        batch_log_ids = await logger.log_batch_decisions(decisions)
        print(f"批量记录了 {len(batch_log_ids)} 条决策日志")
        
        # 示例2: 风险评估分析
        print("\n2. 风险评估分析")
        risks = []
        for i in range(5):
            risk = {
                'identification': {
                    'risk_type': f'风险类型_{i}',
                    'level': RiskLevel.HIGH if i % 2 == 0 else RiskLevel.MEDIUM,
                    'probability': 0.3 + i * 0.1,
                    'impact': 0.5 + i * 0.1,
                    'score': (0.3 + i * 0.1) * (0.5 + i * 0.1),
                    'description': f'风险描述 {i}',
                    'mitigation': f'缓解措施 {i}'
                },
                'evaluation': {
                    'criteria': {'volatility': 0.2, 'correlation': 0.5},
                    'result': {'var': 0.05 + i * 0.01}
                }
            }
            risks.append(risk)
        
        risk_log_ids = await logger.log_batch_risks(risks)
        print(f"批量记录了 {len(risk_log_ids)} 条风险日志")
        
        # 示例3: 使用分析功能
        print("\n3. 决策分析功能")
        
        # 决策模式分析
        patterns = await logger.analytics.analyze_decision_patterns()
        print(f"决策模式分析: {len(patterns)} 种决策类型")
        
        # 策略有效性分析
        effectiveness = await logger.analytics.analyze_strategy_effectiveness()
        print(f"策略有效性分析: {len(effectiveness)} 种策略类型")
        
        # 风险趋势分析
        risk_trends = await logger.analytics.analyze_risk_trends()
        print(f"风险趋势分析: {len(risk_trends)} 个风险级别")
        
        # 生成决策报告
        report = await logger.analytics.generate_decision_report()
        print(f"决策报告生成完成，包含 {len(report.get('recommendations', []))} 条建议")
        
        # 示例4: 使用预测功能
        print("\n4. 决策预测功能")
        
        predictor = DecisionPredictor(logger)
        prediction = await predictor.predict_decision_outcome({
            'decision_type': 'tactical',
            'context': 'market_analysis'
        })
        print(f"决策预测: 成功率 {prediction.get('success_probability', 0):.2%}")
        
        # 示例5: 使用优化功能
        print("\n5. 决策优化功能")
        
        optimizer = DecisionOptimizer(logger)
        optimization = await optimizer.optimize_decision_parameters('tactical')
        print(f"参数优化: {len(optimization.get('optimization_suggestions', []))} 条建议")
        
        # 示例6: 使用模拟功能
        print("\n6. 决策模拟功能")
        
        simulator = DecisionSimulator(logger)
        simulation = await simulator.simulate_decision_scenario({
            'base_success_rate': 0.8,
            'confidence_factor': 1.0
        }, iterations=50)
        print(f"场景模拟: {simulation.get('success_probability', 0):.2%} 成功率")
        
        # 示例7: 日志验证
        print("\n7. 日志验证功能")
        
        validator = LogValidator()
        
        # 验证决策输入
        test_input = DecisionInput(
            input_type="test",
            data={"test": "data"},
            priority=1
        )
        is_valid, errors = validator.validate_decision_input(test_input)
        print(f"决策输入验证: {'通过' if is_valid else '失败'}")
        
        # 验证决策结果
        test_result = DecisionResult(
            decision_type=DecisionType.TACTICAL,
            outcome="test",
            success=True,
            confidence=0.8,
            execution_time=0.1
        )
        is_valid, errors = validator.validate_decision_result(test_result)
        print(f"决策结果验证: {'通过' if is_valid else '失败'}")
        
        # 示例8: 日志压缩和导出
        print("\n8. 日志压缩和导出功能")
        
        compressor = LogCompressor()
        
        # 获取所有日志进行压缩测试
        all_logs = await logger.storage.retrieve({})
        if all_logs:
            compressed_data = compressor.compress_logs(all_logs)
            print(f"日志压缩: {len(all_logs)} 条日志压缩至 {format_file_size(len(compressed_data))}")
            
            # 解压缩测试
            decompressed_logs = compressor.decompress_logs(compressed_data)
            print(f"日志解压缩: 成功恢复 {len(decompressed_logs)} 条日志")
        
        # 示例9: 监控和告警
        print("\n9. 监控和告警功能")
        
        monitor = LogMonitor(logger)
        monitor.set_alert_threshold('error_rate', 0.1, 'greater')
        monitor.set_alert_threshold('min_success_rate', 0.5, 'less')
        
        # 模拟触发告警
        await monitor._check_metrics()
        alerts = monitor.get_recent_alerts()
        print(f"监控告警: {len(alerts)} 条告警")
        
        # 示例10: 日志归档
        print("\n10. 日志归档功能")
        
        archiver = LogArchiver(logger)
        archived_count = await archiver.archive_old_logs(days=0)  # 归档所有日志用于测试
        print(f"日志归档: 归档了 {archived_count} 条日志")
        
        # 示例11: 性能统计
        print("\n11. 性能统计")
        
        stats = logger.get_statistics()
        performance = logger.get_performance_summary()
        
        print(f"总日志数: {stats['total_logs']}")
        print(f"决策日志数: {stats['decision_logs']}")
        print(f"错误率: {performance['error_rate']:.2%}")
        print(f"决策成功率: {performance['decision_success_rate']:.2%}")
        
        # 示例12: 上下文管理器的嵌套使用
        print("\n12. 嵌套上下文管理器")
        
        with logger.decision_context("nested_test_001") as outer_ctx:
            print(f"外层决策上下文: {outer_ctx['decision_id']}")
            
            # 在外层上下文中创建策略上下文
            with logger.strategy_context("nested_strategy_001", StrategyType.ADAPTIVE) as inner_ctx:
                print(f"内层策略上下文: {inner_ctx['strategy_id']}")
                await asyncio.sleep(0.05)  # 模拟执行时间
        
        print("\n=== 高级功能示例完成 ===")
        
    except Exception as e:
        print(f"高级示例执行失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 停止日志记录器
        await logger.stop()


# 性能测试函数

async def performance_test():
    """性能测试函数"""
    print("=== 性能测试开始 ===")
    
    logger = DecisionLogger(
        storage_type="memory",
        storage_config={"max_size": 10000},
        enable_async=True
    )
    
    await logger.start()
    
    try:
        # 测试1: 单线程写入性能
        print("\n1. 单线程写入性能测试")
        start_time = time.time()
        
        for i in range(1000):
            await logger.log_decision_input(DecisionInput(
                input_type="performance_test",
                data={"iteration": i},
                source="perf_test"
            ))
        
        single_thread_time = time.time() - start_time
        print(f"单线程1000条日志写入耗时: {single_thread_time:.2f}秒")
        print(f"平均每条日志: {single_thread_time/1000*1000:.2f}毫秒")
        
        # 测试2: 批量写入性能
        print("\n2. 批量写入性能测试")
        batch_decisions = []
        for i in range(1000):
            batch_decisions.append({
                'input': {
                    'input_type': 'batch_test',
                    'data': {'iteration': i},
                    'source': 'batch_perf_test'
                }
            })
        
        start_time = time.time()
        log_ids = await logger.log_batch_decisions(batch_decisions)
        batch_time = time.time() - start_time
        
        print(f"批量1000条日志写入耗时: {batch_time:.2f}秒")
        print(f"平均每条日志: {batch_time/1000*1000:.2f}毫秒")
        print(f"批量处理效率提升: {single_thread_time/batch_time:.2f}倍")
        
        # 测试3: 查询性能
        print("\n3. 查询性能测试")
        
        start_time = time.time()
        query_results = await logger.query_logs({'log_type': 'decision_input'}, limit=100)
        query_time = time.time() - start_time
        
        print(f"查询100条记录耗时: {query_time:.2f}秒")
        print(f"找到记录数: {len(query_results)}")
        
        # 测试4: 统计计算性能
        print("\n4. 统计计算性能测试")
        
        start_time = time.time()
        stats = logger.get_statistics()
        performance = logger.get_performance_summary()
        stats_time = time.time() - start_time
        
        print(f"统计计算耗时: {stats_time:.2f}秒")
        
        print("\n=== 性能测试完成 ===")
        
    finally:
        await logger.stop()


# 压力测试函数

async def stress_test():
    """压力测试函数"""
    print("=== 压力测试开始 ===")
    
    logger = DecisionLogger(
        storage_type="memory",
        storage_config={"max_size": 50000},
        enable_async=True
    )
    
    await logger.start()
    
    try:
        # 并发写入测试
        print("\n1. 并发写入压力测试")
        
        async def concurrent_writer(writer_id: int, count: int):
            """并发写入函数"""
            for i in range(count):
                await logger.log_decision_input(DecisionInput(
                    input_type=f"stress_test_writer_{writer_id}",
                    data={"writer_id": writer_id, "iteration": i},
                    source=f"stress_test_{writer_id}"
                ))
        
        # 启动10个并发写入任务
        start_time = time.time()
        tasks = [concurrent_writer(i, 100) for i in range(10)]
        await asyncio.gather(*tasks)
        concurrent_time = time.time() - start_time
        
        print(f"10个并发任务各写入100条日志，总计1000条")
        print(f"并发写入耗时: {concurrent_time:.2f}秒")
        print(f"平均每秒处理: {1000/concurrent_time:.0f} 条日志")
        
        # 内存使用统计
        stats = logger.get_statistics()
        print(f"总日志数: {stats['total_logs']}")
        
        print("\n=== 压力测试完成 ===")
        
    finally:
        await logger.stop()


# 错误处理测试函数

async def error_handling_test():
    """错误处理测试函数"""
    print("=== 错误处理测试开始 ===")
    
    logger = DecisionLogger(
        storage_type="memory",
        storage_config={"max_size": 100}
    )
    
    await logger.start()
    
    try:
        # 测试1: 无效数据处理
        print("\n1. 无效数据处理测试")
        
        try:
            # 尝试记录无效的决策输入
            invalid_input = DecisionInput(
                input_type="",  # 空类型
                data="invalid_data",  # 非字典数据
                priority=-1  # 负优先级
            )
            
            is_valid, errors = LogValidator.validate_decision_input(invalid_input)
            print(f"无效数据验证: {'通过' if is_valid else '失败'}")
            if errors:
                print(f"验证错误: {errors}")
                
        except Exception as e:
            print(f"无效数据处理异常: {e}")
        
        # 测试2: 存储错误处理
        print("\n2. 存储错误处理测试")
        
        # 模拟存储错误
        class FailingStorage(LogStorage):
            async def store(self, log_entry):
                raise IOError("模拟存储失败")
            
            async def retrieve(self, query):
                raise IOError("模拟检索失败")
            
            async def delete(self, log_id):
                raise IOError("模拟删除失败")
        
        # 临时替换存储后端
        original_storage = logger.storage
        logger.storage = FailingStorage()
        
        try:
            await logger.log_decision_input(DecisionInput(input_type="error_test"))
            print("存储错误处理: 成功")
        except Exception as e:
            print(f"存储错误处理: 检测到错误 {e}")
        
        # 恢复原始存储
        logger.storage = original_storage
        
        # 测试3: 网络超时处理
        print("\n3. 异步操作超时处理")
        
        # 模拟慢速存储
        class SlowStorage(LogStorage):
            async def store(self, log_entry):
                await asyncio.sleep(0.1)  # 模拟网络延迟
                return True
            
            async def retrieve(self, query):
                await asyncio.sleep(0.1)
                return []
            
            async def delete(self, log_id):
                await asyncio.sleep(0.1)
                return True
        
        logger.storage = SlowStorage()
        
        start_time = time.time()
        await logger.log_decision_input(DecisionInput(input_type="slow_test"))
        slow_time = time.time() - start_time
        
        print(f"慢速存储测试: 耗时 {slow_time:.2f}秒")
        
        # 恢复原始存储
        logger.storage = original_storage
        
        print("\n=== 错误处理测试完成 ===")
        
    finally:
        await logger.stop()


# 集成测试函数

async def integration_test():
    """集成测试函数"""
    print("=== 集成测试开始 ===")
    
    # 测试完整的决策流程
    logger = DecisionLogger(
        storage_type="memory",
        storage_config={"max_size": 1000},
        enable_async=True
    )
    
    await logger.start()
    
    try:
        print("\n1. 完整决策流程测试")
        
        # 模拟一个完整的投资决策流程
        decision_id = f"investment_decision_{int(time.time())}"
        
        # 步骤1: 市场数据输入
        market_input = DecisionInput(
            input_type="market_data",
            data={
                "stock_price": 100.50,
                "volume": 1000000,
                "pe_ratio": 25.5,
                "market_trend": "bullish"
            },
            source="market_api",
            priority=1
        )
        
        input_log_id = await logger.log_decision_input(market_input)
        print(f"市场数据输入记录: {input_log_id}")
        
        # 步骤2: 分析逻辑
        analysis_logic = DecisionLogic(
            algorithm="technical_fundamental_hybrid",
            parameters={
                "technical_weight": 0.6,
                "fundamental_weight": 0.4,
                "risk_tolerance": 0.3
            },
            reasoning="结合技术分析和基本面分析",
            confidence=0.85,
            factors=["价格趋势", "成交量", "PE比率", "市场情绪"]
        )
        
        logic_log_id = await logger.log_decision_logic(analysis_logic, decision_id)
        print(f"分析逻辑记录: {logic_log_id}")
        
        # 步骤3: 决策结果
        decision_result = DecisionResult(
            decision_type=DecisionType.TACTICAL,
            outcome="买入100股",
            success=True,
            value=10050.0,  # 100股 * 100.50
            confidence=0.85,
            execution_time=0.05
        )
        
        result_log_id = await logger.log_decision_result(decision_result, decision_id)
        print(f"决策结果记录: {result_log_id}")
        
        # 步骤4: 风险评估
        risk_assessment = RiskAssessment(
            risk_type="market_risk",
            level=RiskLevel.MEDIUM,
            probability=0.3,
            impact=0.4,
            score=0.12,
            description="市场波动可能导致短期损失",
            mitigation="设置止损位",
            owner="risk_manager"
        )
        
        risk_log_id = await logger.log_risk_identification(
            risk_assessment, 
            detection_method="monte_carlo_simulation"
        )
        print(f"风险评估记录: {risk_log_id}")
        
        # 步骤5: 策略执行
        strategy_id = "investment_strategy_001"
        
        selection_log_id = await logger.log_strategy_selection(
            strategy_id=strategy_id,
            strategy_type=StrategyType.BALANCED,
            selection_reason="当前市场环境适合平衡策略"
        )
        print(f"策略选择记录: {selection_log_id}")
        
        # 策略执行步骤
        execution_steps = [
            StrategyStep(
                step_name="订单准备",
                description="准备买入订单",
                status=ExecutionStatus.COMPLETED,
                result={"order_size": 100, "price_limit": 101.0}
            ),
            StrategyStep(
                step_name="执行交易",
                description="执行买入交易",
                status=ExecutionStatus.COMPLETED,
                result={"executed": True, "actual_price": 100.50}
            ),
            StrategyStep(
                step_name="风险控制设置",
                description="设置止损和止盈",
                status=ExecutionStatus.COMPLETED,
                result={"stop_loss": 95.0, "take_profit": 110.0}
            )
        ]
        
        execution_log_id = await logger.log_strategy_execution(strategy_id, execution_steps)
        print(f"策略执行记录: {execution_log_id}")
        
        # 步骤6: 策略结果
        strategy_result_log_id = await logger.log_strategy_result(
            strategy_id=strategy_id,
            success=True,
            outcome="投资策略成功执行",
            performance_metrics={
                "return": 0.0,  # 刚执行，暂无收益
                "risk_score": 0.3,
                "execution_quality": 0.95
            }
        )
        print(f"策略结果记录: {strategy_result_log_id}")
        
        # 步骤7: 决策解释
        explanation = DecisionExplanation(
            reason="技术指标显示买入信号",
            rationale="MA10上穿MA20，RSI显示超卖反弹",
            evidence=["技术指标确认", "基本面支撑", "市场情绪积极"],
            alternatives=["等待回调买入", "分批建仓"],
            impact_analysis={
                "portfolio_impact": 0.02,
                "risk_change": 0.01
            }
        )
        
        explanation_log_id = await logger.log_decision_explanation(explanation, decision_id)
        print(f"决策解释记录: {explanation_log_id}")
        
        # 步骤8: 效果评估（模拟后续评估）
        evaluation = EffectEvaluation(
            decision_id=decision_id,
            expected_outcome="获得5%收益",
            actual_outcome="获得3.2%收益",
            performance_metrics={
                "return": 0.032,
                "sharpe_ratio": 1.2,
                "max_drawdown": 0.01
            },
            lessons_learned=[
                "技术分析在当前市场有效",
                "需要更好的入场时机把握"
            ],
            recommendations=[
                "优化技术指标参数",
                "结合更多基本面因素"
            ]
        )
        
        evaluation_log_id = await logger.log_decision_evaluation(evaluation)
        print(f"效果评估记录: {evaluation_log_id}")
        
        # 验证完整流程
        print("\n2. 流程验证")
        
        # 查询完整的决策历史
        history = await logger.get_decision_history(decision_id)
        print(f"决策历史记录数: {len(history)}")
        
        # 生成综合报告
        report = await logger.analytics.generate_decision_report()
        print(f"综合报告建议数: {len(report.get('recommendations', []))}")
        
        # 获取策略性能
        strategy_performance = await logger.get_strategy_performance(strategy_id)
        print(f"策略性能统计: {strategy_performance}")
        
        # 获取风险摘要
        risk_summary = await logger.get_risk_summary()
        print(f"风险摘要: {risk_summary}")
        
        print("\n=== 集成测试完成 ===")
        
    except Exception as e:
        print(f"集成测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await logger.stop()


# 主测试函数

async def run_all_tests():
    """运行所有测试"""
    print("L6决策日志记录器 - 完整测试套件")
    print("=" * 50)
    
    tests = [
        ("基础功能测试", example_usage),
        ("高级功能测试", advanced_example_usage),
        ("性能测试", performance_test),
        ("压力测试", stress_test),
        ("错误处理测试", error_handling_test),
        ("集成测试", integration_test)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            await test_func()
            print(f"✓ {test_name} 通过")
        except Exception as e:
            print(f"✗ {test_name} 失败: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试间隔
        await asyncio.sleep(1)
    
    print("\n" + "=" * 50)
    print("所有测试完成")


if __name__ == "__main__":
    # 运行所有测试
    asyncio.run(run_all_tests())