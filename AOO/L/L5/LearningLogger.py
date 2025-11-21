#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L5学习日志记录器

这是一个全面的机器学习实验日志记录系统，支持：
- 机器学习模型训练日志（训练进度、损失函数、验证指标）
- 模型评估日志（准确率、精确率、召回率、F1分数）
- 超参数调优日志（参数搜索、最优参数、收敛情况）
- 数据处理日志（数据加载、预处理、特征工程）
- 实验跟踪日志（实验配置、实验结果、对比分析）
- 模型版本日志（模型保存、模型加载、模型切换）
- 异步学习日志处理
- 完整的错误处理和日志记录

"""

# 模块版本信息
__version__ = "1.0.0"
__author__ = "L5学习日志记录器开发团队"

import asyncio
import json
import logging
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager, contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, AsyncGenerator
from weakref import WeakValueDictionary
import pickle
import hashlib
import traceback
import warnings
from abc import ABC, abstractmethod


# =============================================================================
# 枚举定义
# =============================================================================

class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ExperimentStatus(Enum):
    """实验状态枚举"""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    PAUSED = "PAUSED"


class ModelVersionStatus(Enum):
    """模型版本状态枚举"""
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    DELETED = "DELETED"
    TESTING = "TESTING"


class DataProcessingStage(Enum):
    """数据处理阶段枚举"""
    LOADING = "LOADING"
    PREPROCESSING = "PREPROCESSING"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    VALIDATION = "VALIDATION"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class HyperparameterSearchStatus(Enum):
    """超参数搜索状态枚举"""
    INITIALIZED = "INITIALIZED"
    SEARCHING = "SEARCHING"
    CONVERGING = "CONVERGING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# =============================================================================
# 数据类定义
# =============================================================================

@dataclass
class TrainingMetrics:
    """训练指标数据类"""
    epoch: int
    batch: int
    iteration: int
    loss: float
    accuracy: Optional[float] = None
    learning_rate: Optional[float] = None
    validation_loss: Optional[float] = None
    validation_accuracy: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """评估指标数据类"""
    accuracy: float
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    confusion_matrix: Optional[List[List[int]]] = None
    classification_report: Optional[Dict[str, Any]] = None
    timestamp: datetime = field(default_factory=datetime.now)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HyperparameterConfig:
    """超参数配置数据类"""
    name: str
    value: Any
    parameter_type: str
    search_space: Optional[Dict[str, Any]] = None
    is_active: bool = True


@dataclass
class HyperparameterTrial:
    """超参数试验数据类"""
    trial_id: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    status: HyperparameterSearchStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    notes: Optional[str] = None


@dataclass
class DataProcessingLog:
    """数据处理日志数据类"""
    stage: DataProcessingStage
    dataset_name: str
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    processing_time: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ExperimentConfig:
    """实验配置数据类"""
    experiment_id: str
    name: str
    model_type: str
    description: Optional[str] = None
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    dataset_info: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None


@dataclass
class ExperimentResult:
    """实验结果数据类"""
    experiment_id: str
    status: ExperimentStatus
    start_time: datetime
    metrics: Dict[str, Any] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class ModelVersion:
    """模型版本数据类"""
    version_id: str
    model_name: str
    version: str
    file_path: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ModelVersionStatus = ModelVersionStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None
    parent_version: Optional[str] = None


@dataclass
class AsyncLogEntry:
    """异步日志条目数据类"""
    entry_id: str
    timestamp: datetime
    level: LogLevel
    category: str
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    experiment_id: Optional[str] = None


# =============================================================================
# 异常类定义
# =============================================================================

class LearningLoggerError(Exception):
    """学习日志记录器基础异常类"""
    pass


class DatabaseError(LearningLoggerError):
    """数据库操作异常"""
    pass


class ExperimentError(LearningLoggerError):
    """实验操作异常"""
    pass


class ModelVersionError(LearningLoggerError):
    """模型版本操作异常"""
    pass


class AsyncProcessingError(LearningLoggerError):
    """异步处理异常"""
    pass


# =============================================================================
# 抽象基类定义
# =============================================================================

class LogStorageBackend(ABC):
    """日志存储后端抽象基类"""
    
    @abstractmethod
    async def store_log_entry(self, entry: AsyncLogEntry) -> None:
        """存储日志条目"""
        pass
    
    @abstractmethod
    async def get_log_entries(self, 
                            filters: Dict[str, Any], 
                            limit: Optional[int] = None) -> List[AsyncLogEntry]:
        """获取日志条目"""
        pass
    
    @abstractmethod
    async def cleanup_old_entries(self, older_than: timedelta) -> int:
        """清理旧日志条目"""
        pass


# =============================================================================
# 数据库存储后端实现
# =============================================================================

class SQLiteStorageBackend(LogStorageBackend):
    """SQLite存储后端实现"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表结构"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS log_entries (
                    entry_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    category TEXT NOT NULL,
                    message TEXT NOT NULL,
                    data TEXT,
                    experiment_id TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON log_entries(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_experiment ON log_entries(experiment_id)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category ON log_entries(category)
            """)
            
            conn.commit()
    
    async def store_log_entry(self, entry: AsyncLogEntry) -> None:
        """存储日志条目"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO log_entries 
                        (entry_id, timestamp, level, category, message, data, experiment_id)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        entry.entry_id,
                        entry.timestamp.isoformat(),
                        entry.level.value,
                        entry.category,
                        entry.message,
                        json.dumps(entry.data),
                        entry.experiment_id
                    ))
                    conn.commit()
        except Exception as e:
            raise DatabaseError(f"Failed to store log entry: {e}")
    
    async def get_log_entries(self, 
                            filters: Dict[str, Any], 
                            limit: Optional[int] = None) -> List[AsyncLogEntry]:
        """获取日志条目"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    query = "SELECT * FROM log_entries WHERE 1=1"
                    params = []
                    
                    if 'start_time' in filters:
                        query += " AND timestamp >= ?"
                        params.append(filters['start_time'].isoformat())
                    
                    if 'end_time' in filters:
                        query += " AND timestamp <= ?"
                        params.append(filters['end_time'].isoformat())
                    
                    if 'level' in filters:
                        query += " AND level = ?"
                        params.append(filters['level'].value)
                    
                    if 'category' in filters:
                        query += " AND category = ?"
                        params.append(filters['category'])
                    
                    if 'experiment_id' in filters:
                        query += " AND experiment_id = ?"
                        params.append(filters['experiment_id'])
                    
                    query += " ORDER BY timestamp DESC"
                    
                    if limit:
                        query += " LIMIT ?"
                        params.append(limit)
                    
                    cursor = conn.execute(query, params)
                    rows = cursor.fetchall()
                    
                    entries = []
                    for row in rows:
                        entries.append(AsyncLogEntry(
                            entry_id=row[0],
                            timestamp=datetime.fromisoformat(row[1]),
                            level=LogLevel(row[2]),
                            category=row[3],
                            message=row[4],
                            data=json.loads(row[5]) if row[5] else {},
                            experiment_id=row[6]
                        ))
                    
                    return entries
        except Exception as e:
            raise DatabaseError(f"Failed to retrieve log entries: {e}")
    
    async def cleanup_old_entries(self, older_than: timedelta) -> int:
        """清理旧日志条目"""
        try:
            cutoff_time = datetime.now() - older_than
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        DELETE FROM log_entries WHERE timestamp < ?
                    """, (cutoff_time.isoformat(),))
                    conn.commit()
                    return cursor.rowcount
        except Exception as e:
            raise DatabaseError(f"Failed to cleanup old entries: {e}")


# =============================================================================
# 核心日志记录器类
# =============================================================================

class LearningLogger:
    """
    L5学习日志记录器主类
    
    这是一个全面的机器学习实验日志记录系统，支持多种日志类型和异步处理。
    
    主要功能：
    - 训练日志记录
    - 模型评估日志
    - 超参数调优日志
    - 数据处理日志
    - 实验跟踪日志
    - 模型版本管理
    - 异步日志处理
    """
    
    def __init__(self, 
                 base_dir: str = "learning_logs",
                 storage_backend: Optional[LogStorageBackend] = None,
                 enable_async: bool = True,
                 max_workers: int = 4,
                 buffer_size: int = 1000,
                 auto_cleanup: bool = True,
                 cleanup_interval_hours: int = 24):
        """
        初始化学习日志记录器
        
        Args:
            base_dir: 日志基础目录
            storage_backend: 存储后端，默认为SQLite
            enable_async: 是否启用异步处理
            max_workers: 异步工作线程数
            buffer_size: 异步缓冲区大小
            auto_cleanup: 是否自动清理旧日志
            cleanup_interval_hours: 清理间隔（小时）
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化存储后端
        if storage_backend is None:
            storage_backend = SQLiteStorageBackend(str(self.base_dir / "logs.db"))
        self.storage_backend = storage_backend
        
        # 异步处理配置
        self.enable_async = enable_async
        self.max_workers = max_workers
        self.buffer_size = buffer_size
        self.auto_cleanup = auto_cleanup
        self.cleanup_interval_hours = cleanup_interval_hours
        
        # 内部状态
        self._active_experiments: Dict[str, Dict[str, Any]] = {}
        self._experiment_locks: Dict[str, threading.Lock] = defaultdict(threading.Lock)
        self._model_versions: WeakValueDictionary = WeakValueDictionary()
        self._async_buffer: deque = deque(maxlen=buffer_size)
        self._shutdown_event = threading.Event()
        
        # 线程池
        if enable_async:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            self._async_worker_thread = threading.Thread(target=self._async_worker, daemon=True)
            self._async_worker_thread.start()
        
        # 初始化数据库
        self._init_database()
        
        # 启动自动清理任务
        if auto_cleanup:
            self._start_auto_cleanup()
        
        # 配置日志记录器
        self._setup_logging()
        
        # 记录初始化信息
        self.info("LearningLogger", "学习日志记录器初始化完成", 
                 extra_data={
                     "base_dir": str(self.base_dir),
                     "async_enabled": enable_async,
                     "storage_backend": type(storage_backend).__name__
                 })
    
    def _init_database(self):
        """初始化数据库表结构"""
        db_path = self.base_dir / "learning_logger.db"
        
        with sqlite3.connect(db_path) as conn:
            # 实验表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    model_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    duration REAL,
                    created_by TEXT,
                    notes TEXT
                )
            """)
            
            # 训练日志表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_logs (
                    log_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    epoch INTEGER,
                    batch INTEGER,
                    iteration INTEGER,
                    loss REAL,
                    accuracy REAL,
                    learning_rate REAL,
                    validation_loss REAL,
                    validation_accuracy REAL,
                    timestamp TEXT NOT NULL,
                    additional_metrics TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)
            
            # 评估日志表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evaluation_logs (
                    eval_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    auc_roc REAL,
                    confusion_matrix TEXT,
                    classification_report TEXT,
                    timestamp TEXT NOT NULL,
                    additional_metrics TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)
            
            # 超参数搜索表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hyperparameter_searches (
                    search_id TEXT PRIMARY KEY,
                    experiment_id TEXT NOT NULL,
                    search_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    best_params TEXT,
                    best_score REAL,
                    total_trials INTEGER,
                    completed_trials INTEGER,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration REAL,
                    notes TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)
            
            # 超参数试验表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS hyperparameter_trials (
                    trial_id TEXT PRIMARY KEY,
                    search_id TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    metrics TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT,
                    duration REAL,
                    notes TEXT,
                    FOREIGN KEY (search_id) REFERENCES hyperparameter_searches (search_id)
                )
            """)
            
            # 数据处理日志表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_processing_logs (
                    process_id TEXT PRIMARY KEY,
                    experiment_id TEXT,
                    stage TEXT NOT NULL,
                    dataset_name TEXT NOT NULL,
                    input_size INTEGER,
                    output_size INTEGER,
                    processing_time REAL,
                    errors TEXT,
                    warnings TEXT,
                    metadata TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)
            
            # 模型版本表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS model_versions (
                    version_id TEXT PRIMARY KEY,
                    model_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    metadata TEXT,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT,
                    parent_version TEXT,
                    experiment_id TEXT,
                    FOREIGN KEY (experiment_id) REFERENCES experiments (experiment_id)
                )
            """)
            
            # 创建索引
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_training_logs_experiment ON training_logs(experiment_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluation_logs_experiment ON evaluation_logs(experiment_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model_versions_name ON model_versions(model_name)")
            
            conn.commit()
    
    def _setup_logging(self):
        """设置日志记录器"""
        log_file = self.base_dir / "learning_logger.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger("LearningLogger")
    
    def _start_auto_cleanup(self):
        """启动自动清理任务"""
        def cleanup_worker():
            while not self._shutdown_event.is_set():
                try:
                    if self.auto_cleanup:
                        older_than = timedelta(hours=self.cleanup_interval_hours)
                        cleaned_count = asyncio.run(
                            self.storage_backend.cleanup_old_entries(older_than)
                        )
                        if cleaned_count > 0:
                            self.debug("LearningLogger", f"自动清理完成，清理了 {cleaned_count} 条旧日志")
                    
                    # 等待清理间隔
                    self._shutdown_event.wait(self.cleanup_interval_hours * 3600)
                except Exception as e:
                    self.error("LearningLogger", f"自动清理任务出错: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()
    
    def _async_worker(self):
        """异步工作线程"""
        while not self._shutdown_event.is_set():
            try:
                if self._async_buffer:
                    entries = []
                    # 批量处理日志条目
                    for _ in range(min(len(self._async_buffer), 100)):
                        if self._async_buffer:
                            entries.append(self._async_buffer.popleft())
                    
                    if entries:
                        # 使用线程池处理
                        futures = []
                        for entry in entries:
                            future = self._executor.submit(
                                lambda e: asyncio.run(self.storage_backend.store_log_entry(e)),
                                entry
                            )
                            futures.append(future)
                        
                        # 等待所有任务完成
                        for future in as_completed(futures):
                            try:
                                future.result()
                            except Exception as e:
                                self.logger.error(f"异步日志处理错误: {e}")
                
                # 短暂休眠
                time.sleep(0.1)
            except Exception as e:
                self.logger.error(f"异步工作线程错误: {e}")
                time.sleep(1)
    
    # =============================================================================
    # 基础日志方法
    # =============================================================================
    
    def _log(self, level: LogLevel, category: str, message: str, 
             extra_data: Optional[Dict[str, Any]] = None, 
             experiment_id: Optional[str] = None):
        """基础日志方法"""
        entry = AsyncLogEntry(
            entry_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            level=level,
            category=category,
            message=message,
            data=extra_data or {},
            experiment_id=experiment_id
        )
        
        if self.enable_async:
            try:
                self._async_buffer.append(entry)
            except Exception as e:
                self.logger.error(f"异步缓冲区添加失败: {e}")
                # 同步处理作为后备
                try:
                    asyncio.run(self.storage_backend.store_log_entry(entry))
                except Exception as sync_error:
                    self.logger.error(f"同步日志处理也失败: {sync_error}")
        else:
            try:
                asyncio.run(self.storage_backend.store_log_entry(entry))
            except Exception as e:
                self.logger.error(f"日志存储失败: {e}")
        
        # 同时输出到标准日志
        log_message = f"[{category}] {message}"
        if extra_data:
            log_message += f" | Data: {extra_data}"
        
        if level == LogLevel.DEBUG:
            self.logger.debug(log_message)
        elif level == LogLevel.INFO:
            self.logger.info(log_message)
        elif level == LogLevel.WARNING:
            self.logger.warning(log_message)
        elif level == LogLevel.ERROR:
            self.logger.error(log_message)
        elif level == LogLevel.CRITICAL:
            self.logger.critical(log_message)
    
    def debug(self, category: str, message: str, extra_data: Optional[Dict[str, Any]] = None,
              experiment_id: Optional[str] = None):
        """调试日志"""
        self._log(LogLevel.DEBUG, category, message, extra_data, experiment_id)
    
    def info(self, category: str, message: str, extra_data: Optional[Dict[str, Any]] = None,
             experiment_id: Optional[str] = None):
        """信息日志"""
        self._log(LogLevel.INFO, category, message, extra_data, experiment_id)
    
    def warning(self, category: str, message: str, extra_data: Optional[Dict[str, Any]] = None,
                experiment_id: Optional[str] = None):
        """警告日志"""
        self._log(LogLevel.WARNING, category, message, extra_data, experiment_id)
    
    def error(self, category: str, message: str, extra_data: Optional[Dict[str, Any]] = None,
              experiment_id: Optional[str] = None):
        """错误日志"""
        self._log(LogLevel.ERROR, category, message, extra_data, experiment_id)
    
    def critical(self, category: str, message: str, extra_data: Optional[Dict[str, Any]] = None,
                 experiment_id: Optional[str] = None):
        """严重错误日志"""
        self._log(LogLevel.CRITICAL, category, message, extra_data, experiment_id)
    
    # =============================================================================
    # 实验管理方法
    # =============================================================================
    
    def _serialize_for_json(self, obj: Any) -> Any:
        """将对象转换为JSON可序列化的格式"""
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {key: self._serialize_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        else:
            return obj

    def create_experiment(self, 
                         name: str,
                         model_type: str,
                         description: Optional[str] = None,
                         hyperparameters: Optional[Dict[str, Any]] = None,
                         dataset_info: Optional[Dict[str, Any]] = None,
                         environment: Optional[Dict[str, Any]] = None,
                         tags: Optional[List[str]] = None,
                         created_by: Optional[str] = None,
                         notes: Optional[str] = None) -> str:
        """
        创建新实验
        
        Args:
            name: 实验名称
            model_type: 模型类型
            description: 实验描述
            hyperparameters: 超参数配置
            dataset_info: 数据集信息
            environment: 环境信息
            tags: 实验标签
            created_by: 创建者
            notes: 备注
            
        Returns:
            实验ID
            
        Raises:
            ExperimentError: 创建实验失败时抛出
        """
        try:
            experiment_id = str(uuid.uuid4())
            config = ExperimentConfig(
                experiment_id=experiment_id,
                name=name,
                description=description,
                model_type=model_type,
                hyperparameters=hyperparameters or {},
                dataset_info=dataset_info or {},
                environment=environment or {},
                tags=tags or [],
                created_by=created_by
            )
            
            # 保存到数据库
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT INTO experiments 
                    (experiment_id, name, description, model_type, status, config, 
                     created_at, created_by, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    experiment_id,
                    name,
                    description,
                    model_type,
                    ExperimentStatus.PENDING.value,
                    json.dumps(self._serialize_for_json(asdict(config))),
                    datetime.now().isoformat(),
                    created_by,
                    notes
                ))
                conn.commit()
            
            # 更新内存状态
            with self._experiment_locks[experiment_id]:
                self._active_experiments[experiment_id] = {
                    'config': config,
                    'status': ExperimentStatus.PENDING,
                    'start_time': None,
                    'metrics': []
                }
            
            self.info("Experiment", f"实验创建成功: {name}", 
                     extra_data={"experiment_id": experiment_id, "model_type": model_type},
                     experiment_id=experiment_id)
            
            return experiment_id
            
        except Exception as e:
            error_msg = f"创建实验失败: {e}"
            self.error("Experiment", error_msg)
            raise ExperimentError(error_msg)
    
    def start_experiment(self, experiment_id: str) -> bool:
        """
        启动实验
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            是否成功启动
            
        Raises:
            ExperimentError: 启动实验失败时抛出
        """
        try:
            with self._experiment_locks[experiment_id]:
                if experiment_id not in self._active_experiments:
                    raise ExperimentError(f"实验不存在: {experiment_id}")
                
                experiment = self._active_experiments[experiment_id]
                if experiment['status'] != ExperimentStatus.PENDING:
                    raise ExperimentError(f"实验状态不正确: {experiment['status']}")
                
                # 更新数据库
                db_path = self.base_dir / "learning_logger.db"
                with sqlite3.connect(db_path) as conn:
                    conn.execute("""
                        UPDATE experiments 
                        SET status = ?, started_at = ?
                        WHERE experiment_id = ?
                    """, (ExperimentStatus.RUNNING.value, datetime.now().isoformat(), experiment_id))
                    conn.commit()
                
                # 更新内存状态
                experiment['status'] = ExperimentStatus.RUNNING
                experiment['start_time'] = datetime.now()
                
                self.info("Experiment", "实验启动", 
                         extra_data={"experiment_id": experiment_id},
                         experiment_id=experiment_id)
                
                return True
                
        except Exception as e:
            error_msg = f"启动实验失败: {e}"
            self.error("Experiment", error_msg, experiment_id=experiment_id)
            raise ExperimentError(error_msg)
    
    def complete_experiment(self, 
                          experiment_id: str, 
                          status: ExperimentStatus = ExperimentStatus.COMPLETED,
                          notes: Optional[str] = None) -> bool:
        """
        完成实验
        
        Args:
            experiment_id: 实验ID
            status: 实验状态
            notes: 完成备注
            
        Returns:
            是否成功完成
            
        Raises:
            ExperimentError: 完成实验失败时抛出
        """
        try:
            with self._experiment_locks[experiment_id]:
                if experiment_id not in self._active_experiments:
                    raise ExperimentError(f"实验不存在: {experiment_id}")
                
                experiment = self._active_experiments[experiment_id]
                if experiment['status'] not in [ExperimentStatus.RUNNING, ExperimentStatus.PAUSED]:
                    raise ExperimentError(f"实验状态不正确: {experiment['status']}")
                
                # 计算持续时间
                start_time = experiment.get('start_time')
                duration = None
                if start_time:
                    duration = (datetime.now() - start_time).total_seconds()
                
                # 更新数据库
                db_path = self.base_dir / "learning_logger.db"
                with sqlite3.connect(db_path) as conn:
                    conn.execute("""
                        UPDATE experiments 
                        SET status = ?, completed_at = ?, duration = ?, notes = ?
                        WHERE experiment_id = ?
                    """, (status.value, datetime.now().isoformat(), duration, notes, experiment_id))
                    conn.commit()
                
                # 更新内存状态
                experiment['status'] = status
                experiment['end_time'] = datetime.now()
                experiment['duration'] = duration
                
                self.info("Experiment", f"实验完成: {status.value}", 
                         extra_data={"experiment_id": experiment_id, "duration": duration},
                         experiment_id=experiment_id)
                
                return True
                
        except Exception as e:
            error_msg = f"完成实验失败: {e}"
            self.error("Experiment", error_msg, experiment_id=experiment_id)
            raise ExperimentError(error_msg)
    
    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        获取实验状态
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            实验状态信息
        """
        try:
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM experiments WHERE experiment_id = ?
                """, (experiment_id,))
                row = cursor.fetchone()
                
                if not row:
                    return None
                
                return {
                    'experiment_id': row[0],
                    'name': row[1],
                    'description': row[2],
                    'model_type': row[3],
                    'status': row[4],
                    'config': json.loads(row[5]) if row[5] else {},
                    'created_at': row[6],
                    'started_at': row[7],
                    'completed_at': row[8],
                    'duration': row[9],
                    'created_by': row[10],
                    'notes': row[11]
                }
                
        except Exception as e:
            self.error("Experiment", f"获取实验状态失败: {e}", experiment_id=experiment_id)
            return None
    
    def list_experiments(self, 
                        status: Optional[ExperimentStatus] = None,
                        limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        列出实验
        
        Args:
            status: 实验状态过滤
            limit: 返回数量限制
            
        Returns:
            实验列表
        """
        try:
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                query = "SELECT * FROM experiments"
                params = []
                
                if status:
                    query += " WHERE status = ?"
                    params.append(status.value)
                
                query += " ORDER BY created_at DESC"
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                experiments = []
                for row in rows:
                    experiments.append({
                        'experiment_id': row[0],
                        'name': row[1],
                        'description': row[2],
                        'model_type': row[3],
                        'status': row[4],
                        'created_at': row[6],
                        'duration': row[9],
                        'created_by': row[10]
                    })
                
                return experiments
                
        except Exception as e:
            self.error("Experiment", f"列出实验失败: {e}")
            return []
    
    # =============================================================================
    # 训练日志方法
    # =============================================================================
    
    def log_training_metrics(self, 
                           experiment_id: str,
                           epoch: int,
                           batch: int,
                           iteration: int,
                           loss: float,
                           accuracy: Optional[float] = None,
                           learning_rate: Optional[float] = None,
                           validation_loss: Optional[float] = None,
                           validation_accuracy: Optional[float] = None,
                           additional_metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        记录训练指标
        
        Args:
            experiment_id: 实验ID
            epoch: 训练轮次
            batch: 批次
            iteration: 迭代次数
            loss: 损失值
            accuracy: 准确率
            learning_rate: 学习率
            validation_loss: 验证损失
            validation_accuracy: 验证准确率
            additional_metrics: 额外指标
            
        Returns:
            是否记录成功
        """
        try:
            log_id = str(uuid.uuid4())
            metrics = TrainingMetrics(
                epoch=epoch,
                batch=batch,
                iteration=iteration,
                loss=loss,
                accuracy=accuracy,
                learning_rate=learning_rate,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                additional_metrics=additional_metrics or {}
            )
            
            # 保存到数据库
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT INTO training_logs 
                    (log_id, experiment_id, epoch, batch, iteration, loss, accuracy, 
                     learning_rate, validation_loss, validation_accuracy, timestamp, additional_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    log_id,
                    experiment_id,
                    epoch,
                    batch,
                    iteration,
                    loss,
                    accuracy,
                    learning_rate,
                    validation_loss,
                    validation_accuracy,
                    datetime.now().isoformat(),
                    json.dumps(metrics.additional_metrics)
                ))
                conn.commit()
            
            # 更新内存状态
            if experiment_id in self._active_experiments:
                self._active_experiments[experiment_id]['metrics'].append(metrics)
            
            # 记录日志
            self.debug("Training", f"Epoch {epoch}, Batch {batch}, Loss: {loss:.4f}", 
                      extra_data={
                          "experiment_id": experiment_id,
                          "accuracy": accuracy,
                          "validation_loss": validation_loss,
                          "learning_rate": learning_rate
                      },
                      experiment_id=experiment_id)
            
            return True
            
        except Exception as e:
            self.error("Training", f"记录训练指标失败: {e}", experiment_id=experiment_id)
            return False
    
    def get_training_metrics(self, 
                           experiment_id: str,
                           limit: Optional[int] = None) -> List[TrainingMetrics]:
        """
        获取训练指标
        
        Args:
            experiment_id: 实验ID
            limit: 返回数量限制
            
        Returns:
            训练指标列表
        """
        try:
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                query = """
                    SELECT epoch, batch, iteration, loss, accuracy, learning_rate,
                           validation_loss, validation_accuracy, timestamp, additional_metrics
                    FROM training_logs 
                    WHERE experiment_id = ?
                    ORDER BY timestamp DESC
                """
                params = [experiment_id]
                
                if limit:
                    query += " LIMIT ?"
                    params.append(limit)
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                metrics = []
                for row in rows:
                    metrics.append(TrainingMetrics(
                        epoch=row[0],
                        batch=row[1],
                        iteration=row[2],
                        loss=row[3],
                        accuracy=row[4],
                        learning_rate=row[5],
                        validation_loss=row[6],
                        validation_accuracy=row[7],
                        timestamp=datetime.fromisoformat(row[8]),
                        additional_metrics=json.loads(row[9]) if row[9] else {}
                    ))
                
                return list(reversed(metrics))  # 按时间正序返回
                
        except Exception as e:
            self.error("Training", f"获取训练指标失败: {e}", experiment_id=experiment_id)
            return []
    
    # =============================================================================
    # 模型评估日志方法
    # =============================================================================
    
    def log_evaluation_metrics(self,
                             experiment_id: str,
                             accuracy: float,
                             precision: Optional[float] = None,
                             recall: Optional[float] = None,
                             f1_score: Optional[float] = None,
                             auc_roc: Optional[float] = None,
                             confusion_matrix: Optional[List[List[int]]] = None,
                             classification_report: Optional[Dict[str, Any]] = None,
                             additional_metrics: Optional[Dict[str, Any]] = None) -> bool:
        """
        记录模型评估指标
        
        Args:
            experiment_id: 实验ID
            accuracy: 准确率
            precision: 精确率
            recall: 召回率
            f1_score: F1分数
            auc_roc: AUC-ROC值
            confusion_matrix: 混淆矩阵
            classification_report: 分类报告
            additional_metrics: 额外指标
            
        Returns:
            是否记录成功
        """
        try:
            eval_id = str(uuid.uuid4())
            metrics = EvaluationMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1_score,
                auc_roc=auc_roc,
                confusion_matrix=confusion_matrix,
                classification_report=classification_report,
                additional_metrics=additional_metrics or {}
            )
            
            # 保存到数据库
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT INTO evaluation_logs 
                    (eval_id, experiment_id, accuracy, precision, recall, f1_score, auc_roc,
                     confusion_matrix, classification_report, timestamp, additional_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    eval_id,
                    experiment_id,
                    accuracy,
                    precision,
                    recall,
                    f1_score,
                    auc_roc,
                    json.dumps(confusion_matrix) if confusion_matrix else None,
                    json.dumps(classification_report) if classification_report else None,
                    datetime.now().isoformat(),
                    json.dumps(metrics.additional_metrics)
                ))
                conn.commit()
            
            # 记录日志
            self.info("Evaluation", f"模型评估完成 - 准确率: {accuracy:.4f}", 
                     extra_data={
                         "experiment_id": experiment_id,
                         "precision": precision,
                         "recall": recall,
                         "f1_score": f1_score,
                         "auc_roc": auc_roc
                     },
                     experiment_id=experiment_id)
            
            return True
            
        except Exception as e:
            self.error("Evaluation", f"记录评估指标失败: {e}", experiment_id=experiment_id)
            return False
    
    def get_evaluation_metrics(self, experiment_id: str) -> List[EvaluationMetrics]:
        """
        获取评估指标
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            评估指标列表
        """
        try:
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT accuracy, precision, recall, f1_score, auc_roc,
                           confusion_matrix, classification_report, timestamp, additional_metrics
                    FROM evaluation_logs 
                    WHERE experiment_id = ?
                    ORDER BY timestamp DESC
                """, (experiment_id,))
                rows = cursor.fetchall()
                
                metrics = []
                for row in rows:
                    metrics.append(EvaluationMetrics(
                        accuracy=row[0],
                        precision=row[1],
                        recall=row[2],
                        f1_score=row[3],
                        auc_roc=row[4],
                        confusion_matrix=json.loads(row[5]) if row[5] else None,
                        classification_report=json.loads(row[6]) if row[6] else None,
                        timestamp=datetime.fromisoformat(row[7]),
                        additional_metrics=json.loads(row[8]) if row[8] else {}
                    ))
                
                return metrics
                
        except Exception as e:
            self.error("Evaluation", f"获取评估指标失败: {e}", experiment_id=experiment_id)
            return []
    
    # =============================================================================
    # 超参数调优日志方法
    # =============================================================================
    
    def start_hyperparameter_search(self,
                                  experiment_id: str,
                                  search_type: str,
                                  search_space: Dict[str, Any],
                                  notes: Optional[str] = None) -> str:
        """
        开始超参数搜索
        
        Args:
            experiment_id: 实验ID
            search_type: 搜索类型 (grid, random, bayesian等)
            search_space: 搜索空间定义
            notes: 备注
            
        Returns:
            搜索ID
            
        Raises:
            ExperimentError: 搜索启动失败时抛出
        """
        try:
            search_id = str(uuid.uuid4())
            
            # 保存到数据库
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT INTO hyperparameter_searches 
                    (search_id, experiment_id, search_type, status, start_time, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    search_id,
                    experiment_id,
                    search_type,
                    HyperparameterSearchStatus.INITIALIZED.value,
                    datetime.now().isoformat(),
                    notes
                ))
                conn.commit()
            
            self.info("Hyperparameter", f"超参数搜索开始: {search_type}", 
                     extra_data={
                         "experiment_id": experiment_id,
                         "search_id": search_id,
                         "search_space": search_space
                     },
                     experiment_id=experiment_id)
            
            return search_id
            
        except Exception as e:
            error_msg = f"启动超参数搜索失败: {e}"
            self.error("Hyperparameter", error_msg, experiment_id=experiment_id)
            raise ExperimentError(error_msg)
    
    def log_hyperparameter_trial(self,
                               search_id: str,
                               parameters: Dict[str, Any],
                               metrics: Dict[str, float],
                               status: HyperparameterSearchStatus = HyperparameterSearchStatus.COMPLETED,
                               notes: Optional[str] = None) -> str:
        """
        记录超参数试验结果
        
        Args:
            search_id: 搜索ID
            parameters: 超参数配置
            metrics: 性能指标
            status: 试验状态
            notes: 备注
            
        Returns:
            试验ID
        """
        try:
            trial_id = str(uuid.uuid4())
            trial = HyperparameterTrial(
                trial_id=trial_id,
                parameters=parameters,
                metrics=metrics,
                status=status,
                start_time=datetime.now(),
                notes=notes
            )
            
            # 计算持续时间
            if status in [HyperparameterSearchStatus.COMPLETED, HyperparameterSearchStatus.FAILED]:
                trial.end_time = datetime.now()
                trial.duration = (trial.end_time - trial.start_time).total_seconds()
            
            # 保存到数据库
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT INTO hyperparameter_trials 
                    (trial_id, search_id, parameters, metrics, status, start_time, end_time, duration, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    trial_id,
                    search_id,
                    json.dumps(parameters),
                    json.dumps(metrics),
                    status.value,
                    trial.start_time.isoformat(),
                    trial.end_time.isoformat() if trial.end_time else None,
                    trial.duration,
                    notes
                ))
                conn.commit()
            
            # 更新搜索统计
            self._update_hyperparameter_search_stats(search_id)
            
            self.debug("Hyperparameter", f"超参数试验完成", 
                      extra_data={
                          "search_id": search_id,
                          "trial_id": trial_id,
                          "parameters": parameters,
                          "metrics": metrics,
                          "status": status.value
                      })
            
            return trial_id
            
        except Exception as e:
            self.error("Hyperparameter", f"记录超参数试验失败: {e}")
            return ""
    
    def _update_hyperparameter_search_stats(self, search_id: str):
        """更新超参数搜索统计信息"""
        try:
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                # 获取试验统计
                cursor = conn.execute("""
                    SELECT COUNT(*), AVG(duration), MAX(metrics)
                    FROM hyperparameter_trials 
                    WHERE search_id = ?
                """, (search_id,))
                
                total_trials, avg_duration, best_score = cursor.fetchone()
                
                # 获取最佳参数
                cursor = conn.execute("""
                    SELECT parameters, metrics 
                    FROM hyperparameter_trials 
                    WHERE search_id = ? AND status = ?
                    ORDER BY metrics DESC
                    LIMIT 1
                """, (search_id, HyperparameterSearchStatus.COMPLETED.value))
                
                best_trial = cursor.fetchone()
                best_params = None
                if best_trial:
                    best_params = json.loads(best_trial[0])
                
                # 更新搜索记录
                conn.execute("""
                    UPDATE hyperparameter_searches 
                    SET total_trials = ?, completed_trials = ?, best_params = ?, best_score = ?
                    WHERE search_id = ?
                """, (
                    total_trials,
                    total_trials,  # 假设所有试验都已完成
                    json.dumps(best_params) if best_params else None,
                    best_score,
                    search_id
                ))
                conn.commit()
                
        except Exception as e:
            self.error("Hyperparameter", f"更新搜索统计失败: {e}")
    
    def get_hyperparameter_trials(self, search_id: str) -> List[HyperparameterTrial]:
        """
        获取超参数试验列表
        
        Args:
            search_id: 搜索ID
            
        Returns:
            试验列表
        """
        try:
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    SELECT trial_id, parameters, metrics, status, start_time, end_time, duration, notes
                    FROM hyperparameter_trials 
                    WHERE search_id = ?
                    ORDER BY start_time DESC
                """, (search_id,))
                rows = cursor.fetchall()
                
                trials = []
                for row in rows:
                    trials.append(HyperparameterTrial(
                        trial_id=row[0],
                        parameters=json.loads(row[1]),
                        metrics=json.loads(row[2]),
                        status=HyperparameterSearchStatus(row[3]),
                        start_time=datetime.fromisoformat(row[4]),
                        end_time=datetime.fromisoformat(row[5]) if row[5] else None,
                        duration=row[6],
                        notes=row[7]
                    ))
                
                return trials
                
        except Exception as e:
            self.error("Hyperparameter", f"获取超参数试验失败: {e}")
            return []
    
    # =============================================================================
    # 数据处理日志方法
    # =============================================================================
    
    def log_data_processing(self,
                          stage: DataProcessingStage,
                          dataset_name: str,
                          experiment_id: Optional[str] = None,
                          input_size: Optional[int] = None,
                          output_size: Optional[int] = None,
                          processing_time: Optional[float] = None,
                          errors: Optional[List[str]] = None,
                          warnings: Optional[List[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        记录数据处理日志
        
        Args:
            stage: 处理阶段
            dataset_name: 数据集名称
            experiment_id: 实验ID
            input_size: 输入数据大小
            output_size: 输出数据大小
            processing_time: 处理时间
            errors: 错误列表
            warnings: 警告列表
            metadata: 元数据
            
        Returns:
            是否记录成功
        """
        try:
            process_id = str(uuid.uuid4())
            log_entry = DataProcessingLog(
                stage=stage,
                dataset_name=dataset_name,
                input_size=input_size,
                output_size=output_size,
                processing_time=processing_time,
                errors=errors or [],
                warnings=warnings or [],
                metadata=metadata or {}
            )
            
            # 保存到数据库
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT INTO data_processing_logs 
                    (process_id, experiment_id, stage, dataset_name, input_size, output_size,
                     processing_time, errors, warnings, metadata, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    process_id,
                    experiment_id,
                    stage.value,
                    dataset_name,
                    input_size,
                    output_size,
                    processing_time,
                    json.dumps(log_entry.errors),
                    json.dumps(log_entry.warnings),
                    json.dumps(log_entry.metadata),
                    datetime.now().isoformat()
                ))
                conn.commit()
            
            # 记录日志
            log_level = LogLevel.ERROR if stage == DataProcessingStage.FAILED else LogLevel.INFO
            message = f"数据处理阶段 {stage.value}: {dataset_name}"
            if errors:
                message += f" - 错误: {len(errors)}"
            if warnings:
                message += f" - 警告: {len(warnings)}"
            
            self._log(log_level, "DataProcessing", message, 
                     extra_data={
                         "experiment_id": experiment_id,
                         "dataset_name": dataset_name,
                         "stage": stage.value,
                         "input_size": input_size,
                         "output_size": output_size,
                         "processing_time": processing_time,
                         "errors_count": len(errors) if errors else 0,
                         "warnings_count": len(warnings) if warnings else 0
                     },
                     experiment_id=experiment_id)
            
            return True
            
        except Exception as e:
            self.error("DataProcessing", f"记录数据处理日志失败: {e}", experiment_id=experiment_id)
            return False
    
    def get_data_processing_logs(self, 
                               experiment_id: Optional[str] = None,
                               dataset_name: Optional[str] = None,
                               stage: Optional[DataProcessingStage] = None) -> List[DataProcessingLog]:
        """
        获取数据处理日志
        
        Args:
            experiment_id: 实验ID过滤
            dataset_name: 数据集名称过滤
            stage: 处理阶段过滤
            
        Returns:
            数据处理日志列表
        """
        try:
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                query = """
                    SELECT stage, dataset_name, input_size, output_size, processing_time,
                           errors, warnings, metadata, timestamp
                    FROM data_processing_logs 
                    WHERE 1=1
                """
                params = []
                
                if experiment_id:
                    query += " AND experiment_id = ?"
                    params.append(experiment_id)
                
                if dataset_name:
                    query += " AND dataset_name = ?"
                    params.append(dataset_name)
                
                if stage:
                    query += " AND stage = ?"
                    params.append(stage.value)
                
                query += " ORDER BY timestamp DESC"
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                logs = []
                for row in rows:
                    logs.append(DataProcessingLog(
                        stage=DataProcessingStage(row[0]),
                        dataset_name=row[1],
                        input_size=row[2],
                        output_size=row[3],
                        processing_time=row[4],
                        errors=json.loads(row[5]) if row[5] else [],
                        warnings=json.loads(row[6]) if row[6] else [],
                        metadata=json.loads(row[7]) if row[7] else {},
                        timestamp=datetime.fromisoformat(row[8])
                    ))
                
                return logs
                
        except Exception as e:
            self.error("DataProcessing", f"获取数据处理日志失败: {e}")
            return []
    
    # =============================================================================
    # 模型版本管理方法
    # =============================================================================
    
    def register_model_version(self,
                             model_name: str,
                             version: str,
                             file_path: str,
                             metadata: Optional[Dict[str, Any]] = None,
                             experiment_id: Optional[str] = None,
                             created_by: Optional[str] = None,
                             parent_version: Optional[str] = None) -> str:
        """
        注册模型版本
        
        Args:
            model_name: 模型名称
            version: 版本号
            file_path: 模型文件路径
            metadata: 模型元数据
            experiment_id: 关联的实验ID
            created_by: 创建者
            parent_version: 父版本
            
        Returns:
            版本ID
            
        Raises:
            ModelVersionError: 注册失败时抛出
        """
        try:
            version_id = str(uuid.uuid4())
            
            # 计算文件哈希
            file_hash = self._calculate_file_hash(file_path)
            
            model_version = ModelVersion(
                version_id=version_id,
                model_name=model_name,
                version=version,
                file_path=file_path,
                metadata=metadata or {},
                created_by=created_by,
                parent_version=parent_version
            )
            
            # 保存到数据库
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                conn.execute("""
                    INSERT INTO model_versions 
                    (version_id, model_name, version, file_path, metadata, status,
                     created_at, created_by, parent_version, experiment_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    version_id,
                    model_name,
                    version,
                    file_path,
                    json.dumps(model_version.metadata),
                    ModelVersionStatus.ACTIVE.value,
                    datetime.now().isoformat(),
                    created_by,
                    parent_version,
                    experiment_id
                ))
                conn.commit()
            
            # 更新内存状态
            self._model_versions[version_id] = model_version
            
            self.info("ModelVersion", f"模型版本注册成功: {model_name} v{version}", 
                     extra_data={
                         "version_id": version_id,
                         "model_name": model_name,
                         "version": version,
                         "file_path": file_path,
                         "file_hash": file_hash,
                         "experiment_id": experiment_id
                     },
                     experiment_id=experiment_id)
            
            return version_id
            
        except Exception as e:
            error_msg = f"注册模型版本失败: {e}"
            self.error("ModelVersion", error_msg, experiment_id=experiment_id)
            raise ModelVersionError(error_msg)
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希值"""
        try:
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception:
            return ""
    
    def get_model_versions(self, 
                         model_name: Optional[str] = None,
                         status: Optional[ModelVersionStatus] = None) -> List[ModelVersion]:
        """
        获取模型版本列表
        
        Args:
            model_name: 模型名称过滤
            status: 版本状态过滤
            
        Returns:
            模型版本列表
        """
        try:
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                query = """
                    SELECT version_id, model_name, version, file_path, metadata, status,
                           created_at, created_by, parent_version, experiment_id
                    FROM model_versions 
                    WHERE 1=1
                """
                params = []
                
                if model_name:
                    query += " AND model_name = ?"
                    params.append(model_name)
                
                if status:
                    query += " AND status = ?"
                    params.append(status.value)
                
                query += " ORDER BY created_at DESC"
                
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                
                versions = []
                for row in rows:
                    version = ModelVersion(
                        version_id=row[0],
                        model_name=row[1],
                        version=row[2],
                        file_path=row[3],
                        metadata=json.loads(row[4]) if row[4] else {},
                        status=ModelVersionStatus(row[5]),
                        created_at=datetime.fromisoformat(row[6]),
                        created_by=row[7],
                        parent_version=row[8]
                    )
                    versions.append(version)
                    self._model_versions[version.version_id] = version
                
                return versions
                
        except Exception as e:
            self.error("ModelVersion", f"获取模型版本失败: {e}")
            return []
    
    def switch_model_version(self, 
                           model_name: str, 
                           target_version: str,
                           notes: Optional[str] = None) -> bool:
        """
        切换模型版本
        
        Args:
            model_name: 模型名称
            target_version: 目标版本
            notes: 切换备注
            
        Returns:
            是否切换成功
        """
        try:
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                # 将所有版本设置为非活跃状态
                conn.execute("""
                    UPDATE model_versions 
                    SET status = ?
                    WHERE model_name = ? AND status = ?
                """, (ModelVersionStatus.ARCHIVED.value, model_name, ModelVersionStatus.ACTIVE.value))
                
                # 设置目标版本为活跃状态
                cursor = conn.execute("""
                    UPDATE model_versions 
                    SET status = ?
                    WHERE model_name = ? AND version = ?
                """, (ModelVersionStatus.ACTIVE.value, model_name, target_version))
                
                if cursor.rowcount == 0:
                    raise ModelVersionError(f"模型版本不存在: {model_name} v{target_version}")
                
                conn.commit()
            
            self.info("ModelVersion", f"模型版本切换成功: {model_name} -> v{target_version}", 
                     extra_data={
                         "model_name": model_name,
                         "target_version": target_version,
                         "notes": notes
                     })
            
            return True
            
        except Exception as e:
            error_msg = f"切换模型版本失败: {e}"
            self.error("ModelVersion", error_msg)
            raise ModelVersionError(error_msg)
    
    def archive_model_version(self, version_id: str, notes: Optional[str] = None) -> bool:
        """
        归档模型版本
        
        Args:
            version_id: 版本ID
            notes: 归档备注
            
        Returns:
            是否归档成功
        """
        try:
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                cursor = conn.execute("""
                    UPDATE model_versions 
                    SET status = ?
                    WHERE version_id = ?
                """, (ModelVersionStatus.ARCHIVED.value, version_id))
                
                if cursor.rowcount == 0:
                    raise ModelVersionError(f"模型版本不存在: {version_id}")
                
                conn.commit()
            
            self.info("ModelVersion", f"模型版本已归档: {version_id}", 
                     extra_data={"version_id": version_id, "notes": notes})
            
            return True
            
        except Exception as e:
            error_msg = f"归档模型版本失败: {e}"
            self.error("ModelVersion", error_msg)
            raise ModelVersionError(error_msg)
    
    # =============================================================================
    # 实验对比分析方法
    # =============================================================================
    
    def compare_experiments(self, 
                          experiment_ids: List[str],
                          metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        对比实验结果
        
        Args:
            experiment_ids: 实验ID列表
            metrics: 要对比的指标列表
            
        Returns:
            对比结果
        """
        try:
            if not experiment_ids:
                raise ValueError("实验ID列表不能为空")
            
            # 默认对比指标
            if metrics is None:
                metrics = ["accuracy", "loss", "f1_score"]
            
            comparison = {
                "experiment_ids": experiment_ids,
                "metrics": metrics,
                "results": {},
                "summary": {}
            }
            
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                for exp_id in experiment_ids:
                    # 获取实验基本信息
                    cursor = conn.execute("""
                        SELECT name, model_type, status, created_at, duration
                        FROM experiments 
                        WHERE experiment_id = ?
                    """, (exp_id,))
                    exp_info = cursor.fetchone()
                    
                    if not exp_info:
                        continue
                    
                    # 获取最新的评估指标
                    cursor = conn.execute("""
                        SELECT accuracy, precision, recall, f1_score, auc_roc, timestamp
                        FROM evaluation_logs 
                        WHERE experiment_id = ?
                        ORDER BY timestamp DESC
                        LIMIT 1
                    """, (exp_id,))
                    eval_result = cursor.fetchone()
                    
                    # 获取训练指标统计
                    cursor = conn.execute("""
                        SELECT AVG(loss), MIN(loss), MAX(loss), AVG(accuracy), MIN(accuracy), MAX(accuracy)
                        FROM training_logs 
                        WHERE experiment_id = ?
                    """, (exp_id,))
                    training_stats = cursor.fetchone()
                    
                    # 整理结果
                    exp_result = {
                        "name": exp_info[0],
                        "model_type": exp_info[1],
                        "status": exp_info[2],
                        "created_at": exp_info[3],
                        "duration": exp_info[4],
                        "evaluation": {},
                        "training": {}
                    }
                    
                    # 添加评估指标
                    if eval_result:
                        exp_result["evaluation"] = {
                            "accuracy": eval_result[0],
                            "precision": eval_result[1],
                            "recall": eval_result[2],
                            "f1_score": eval_result[3],
                            "auc_roc": eval_result[4],
                            "timestamp": eval_result[5]
                        }
                    
                    # 添加训练统计
                    if training_stats and training_stats[0] is not None:
                        exp_result["training"] = {
                            "avg_loss": training_stats[0],
                            "min_loss": training_stats[1],
                            "max_loss": training_stats[2],
                            "avg_accuracy": training_stats[3],
                            "min_accuracy": training_stats[4],
                            "max_accuracy": training_stats[5]
                        }
                    
                    comparison["results"][exp_id] = exp_result
            
            # 计算汇总统计
            for metric in metrics:
                values = []
                for exp_id, result in comparison["results"].items():
                    if "evaluation" in result and metric in result["evaluation"]:
                        value = result["evaluation"][metric]
                        if value is not None:
                            values.append(value)
                
                if values:
                    comparison["summary"][metric] = {
                        "best": max(values),
                        "worst": min(values),
                        "average": sum(values) / len(values),
                        "count": len(values)
                    }
            
            self.info("ExperimentComparison", f"实验对比完成: {len(experiment_ids)} 个实验", 
                     extra_data={
                         "experiment_count": len(experiment_ids),
                         "metrics": metrics,
                         "comparison_summary": comparison["summary"]
                     })
            
            return comparison
            
        except Exception as e:
            self.error("ExperimentComparison", f"实验对比失败: {e}")
            raise ExperimentError(f"实验对比失败: {e}")
    
    def generate_experiment_report(self, experiment_id: str) -> Dict[str, Any]:
        """
        生成实验报告
        
        Args:
            experiment_id: 实验ID
            
        Returns:
            实验报告
        """
        try:
            report = {
                "experiment_id": experiment_id,
                "generated_at": datetime.now().isoformat(),
                "experiment_info": {},
                "training_progress": {},
                "evaluation_results": {},
                "data_processing": {},
                "hyperparameter_search": {},
                "model_versions": [],
                "summary": {}
            }
            
            db_path = self.base_dir / "learning_logger.db"
            with sqlite3.connect(db_path) as conn:
                # 实验基本信息
                cursor = conn.execute("""
                    SELECT name, description, model_type, status, config, created_at, 
                           started_at, completed_at, duration, created_by, notes
                    FROM experiments 
                    WHERE experiment_id = ?
                """, (experiment_id,))
                exp_info = cursor.fetchone()
                
                if not exp_info:
                    raise ExperimentError(f"实验不存在: {experiment_id}")
                
                report["experiment_info"] = {
                    "name": exp_info[0],
                    "description": exp_info[1],
                    "model_type": exp_info[2],
                    "status": exp_info[3],
                    "config": json.loads(exp_info[4]) if exp_info[4] else {},
                    "created_at": exp_info[5],
                    "started_at": exp_info[6],
                    "completed_at": exp_info[7],
                    "duration": exp_info[8],
                    "created_by": exp_info[9],
                    "notes": exp_info[10]
                }
                
                # 训练进度
                cursor = conn.execute("""
                    SELECT COUNT(*), AVG(loss), MIN(loss), MAX(loss), 
                           AVG(accuracy), MIN(accuracy), MAX(accuracy)
                    FROM training_logs 
                    WHERE experiment_id = ?
                """, (experiment_id,))
                training_stats = cursor.fetchone()
                
                if training_stats and training_stats[0] > 0:
                    report["training_progress"] = {
                        "total_logs": training_stats[0],
                        "loss_stats": {
                            "average": training_stats[1],
                            "minimum": training_stats[2],
                            "maximum": training_stats[3]
                        },
                        "accuracy_stats": {
                            "average": training_stats[4],
                            "minimum": training_stats[5],
                            "maximum": training_stats[6]
                        }
                    }
                
                # 评估结果
                cursor = conn.execute("""
                    SELECT accuracy, precision, recall, f1_score, auc_roc, timestamp
                    FROM evaluation_logs 
                    WHERE experiment_id = ?
                    ORDER BY timestamp DESC
                """, (experiment_id,))
                eval_results = cursor.fetchall()
                
                if eval_results:
                    latest_eval = eval_results[0]
                    report["evaluation_results"] = {
                        "latest": {
                            "accuracy": latest_eval[0],
                            "precision": latest_eval[1],
                            "recall": latest_eval[2],
                            "f1_score": latest_eval[3],
                            "auc_roc": latest_eval[4],
                            "timestamp": latest_eval[5]
                        },
                        "history_count": len(eval_results)
                    }
                
                # 数据处理日志
                cursor = conn.execute("""
                    SELECT stage, dataset_name, input_size, output_size, processing_time, timestamp
                    FROM data_processing_logs 
                    WHERE experiment_id = ?
                    ORDER BY timestamp DESC
                """, (experiment_id,))
                data_logs = cursor.fetchall()
                
                if data_logs:
                    report["data_processing"] = {
                        "total_operations": len(data_logs),
                        "stages": list(set([log[0] for log in data_logs])),
                        "datasets": list(set([log[1] for log in data_logs])),
                        "latest": {
                            "stage": data_logs[0][0],
                            "dataset": data_logs[0][1],
                            "timestamp": data_logs[0][5]
                        }
                    }
                
                # 超参数搜索
                cursor = conn.execute("""
                    SELECT search_id, search_type, status, best_params, best_score, 
                           total_trials, completed_trials, start_time, end_time
                    FROM hyperparameter_searches 
                    WHERE experiment_id = ?
                """, (experiment_id,))
                search_results = cursor.fetchall()
                
                if search_results:
                    report["hyperparameter_search"] = {
                        "total_searches": len(search_results),
                        "searches": []
                    }
                    
                    for search in search_results:
                        search_info = {
                            "search_id": search[0],
                            "search_type": search[1],
                            "status": search[2],
                            "best_params": json.loads(search[3]) if search[3] else None,
                            "best_score": search[4],
                            "total_trials": search[5],
                            "completed_trials": search[6],
                            "start_time": search[7],
                            "end_time": search[8]
                        }
                        report["hyperparameter_search"]["searches"].append(search_info)
                
                # 模型版本
                cursor = conn.execute("""
                    SELECT version_id, model_name, version, status, created_at, created_by
                    FROM model_versions 
                    WHERE experiment_id = ?
                    ORDER BY created_at DESC
                """, (experiment_id,))
                version_results = cursor.fetchall()
                
                report["model_versions"] = [
                    {
                        "version_id": version[0],
                        "model_name": version[1],
                        "version": version[2],
                        "status": version[3],
                        "created_at": version[4],
                        "created_by": version[5]
                    }
                    for version in version_results
                ]
                
                # 生成汇总
                report["summary"] = {
                    "experiment_duration": report["experiment_info"].get("duration"),
                    "training_logs_count": report["training_progress"].get("total_logs", 0),
                    "evaluation_count": report["evaluation_results"].get("history_count", 0),
                    "data_processing_operations": report["data_processing"].get("total_operations", 0),
                    "hyperparameter_searches": report["hyperparameter_search"].get("total_searches", 0),
                    "model_versions": len(report["model_versions"]),
                    "final_accuracy": report["evaluation_results"].get("latest", {}).get("accuracy")
                }
            
            self.info("ExperimentReport", f"实验报告生成完成: {experiment_id}", 
                     extra_data={
                         "experiment_id": experiment_id,
                         "report_summary": report["summary"]
                     },
                     experiment_id=experiment_id)
            
            return report
            
        except Exception as e:
            self.error("ExperimentReport", f"生成实验报告失败: {e}", experiment_id=experiment_id)
            raise ExperimentError(f"生成实验报告失败: {e}")
    
    # =============================================================================
    # 异步上下文管理器
    # =============================================================================
    
    @asynccontextmanager
    async def async_experiment_context(self, experiment_config: Dict[str, Any]):
        """
        异步实验上下文管理器
        
        Args:
            experiment_config: 实验配置
            
        Yields:
            实验ID
        """
        experiment_id = None
        try:
            # 创建实验
            experiment_id = self.create_experiment(**experiment_config)
            
            # 启动实验
            self.start_experiment(experiment_id)
            
            yield experiment_id
            
        except Exception as e:
            if experiment_id:
                self.error("AsyncExperiment", f"异步实验上下文异常: {e}", experiment_id=experiment_id)
                self.complete_experiment(experiment_id, ExperimentStatus.FAILED, str(e))
            raise
        finally:
            if experiment_id:
                try:
                    self.complete_experiment(experiment_id)
                except Exception as e:
                    self.error("AsyncExperiment", f"完成异步实验失败: {e}", experiment_id=experiment_id)
    
    @contextmanager
    def experiment_context(self, experiment_config: Dict[str, Any]):
        """
        实验上下文管理器
        
        Args:
            experiment_config: 实验配置
            
        Yields:
            实验ID
        """
        experiment_id = None
        try:
            # 创建实验
            experiment_id = self.create_experiment(**experiment_config)
            
            # 启动实验
            self.start_experiment(experiment_id)
            
            yield experiment_id
            
        except Exception as e:
            if experiment_id:
                self.error("Experiment", f"实验上下文异常: {e}", experiment_id=experiment_id)
                self.complete_experiment(experiment_id, ExperimentStatus.FAILED, str(e))
            raise
        finally:
            if experiment_id:
                try:
                    self.complete_experiment(experiment_id)
                except Exception as e:
                    self.error("Experiment", f"完成实验失败: {e}", experiment_id=experiment_id)
    
    # =============================================================================
    # 清理和关闭方法
    # =============================================================================
    
    def cleanup(self):
        """清理资源"""
        try:
            # 停止异步处理
            self._shutdown_event.set()
            
            if hasattr(self, '_executor'):
                self._executor.shutdown(wait=True)
            
            # 刷新异步缓冲区
            while self._async_buffer:
                try:
                    entry = self._async_buffer.popleft()
                    asyncio.run(self.storage_backend.store_log_entry(entry))
                except Exception as e:
                    self.logger.error(f"清理异步缓冲区失败: {e}")
            
            self.info("LearningLogger", "学习日志记录器清理完成")
            
        except Exception as e:
            self.error("LearningLogger", f"清理资源失败: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()
    
    def __del__(self):
        """析构函数"""
        try:
            self.cleanup()
        except:
            pass


# =============================================================================
# 使用示例和测试代码
# =============================================================================

def example_usage():
    """使用示例"""
    
    # 初始化学习日志记录器
    logger = LearningLogger(
        base_dir="examples/learning_logs",
        enable_async=True,
        max_workers=2
    )
    
    try:
        # 1. 创建和启动实验
        experiment_config = {
            "name": "图像分类模型训练",
            "model_type": "CNN",
            "description": "使用ResNet进行图像分类",
            "hyperparameters": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "epochs": 100
            },
            "dataset_info": {
                "name": "CIFAR-10",
                "train_size": 50000,
                "test_size": 10000
            },
            "tags": ["image-classification", "cnn", "resnet"],
            "created_by": "researcher@example.com"
        }
        
        with logger.experiment_context(experiment_config) as experiment_id:
            print(f"实验ID: {experiment_id}")
            
            # 2. 记录数据处理日志
            logger.log_data_processing(
                stage=DataProcessingStage.LOADING,
                dataset_name="CIFAR-10",
                experiment_id=experiment_id,
                input_size=50000,
                processing_time=120.5
            )
            
            logger.log_data_processing(
                stage=DataProcessingStage.PREPROCESSING,
                dataset_name="CIFAR-10",
                experiment_id=experiment_id,
                input_size=50000,
                output_size=50000,
                processing_time=45.2
            )
            
            # 3. 模拟训练过程
            for epoch in range(1, 11):
                for batch in range(1, 100):
                    iteration = (epoch - 1) * 100 + batch
                    
                    # 模拟损失和准确率
                    loss = 2.0 * (0.95 ** epoch) + 0.1 * (0.99 ** iteration)
                    accuracy = 0.5 + 0.4 * (1 - 0.95 ** epoch)
                    learning_rate = 0.001 * (0.95 ** (epoch // 10))
                    
                    # 记录训练指标
                    logger.log_training_metrics(
                        experiment_id=experiment_id,
                        epoch=epoch,
                        batch=batch,
                        iteration=iteration,
                        loss=loss,
                        accuracy=accuracy,
                        learning_rate=learning_rate
                    )
                    
                    # 每10个批次记录一次
                    if batch % 10 == 0:
                        logger.info("Training", 
                                  f"Epoch {epoch}, Batch {batch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}",
                                  experiment_id=experiment_id)
                
                # 每个epoch结束后记录验证结果
                val_loss = loss * 1.1
                val_accuracy = accuracy * 0.95
                
                logger.log_training_metrics(
                    experiment_id=experiment_id,
                    epoch=epoch,
                    batch=0,
                    iteration=epoch * 100,
                    loss=loss,
                    accuracy=accuracy,
                    validation_loss=val_loss,
                    validation_accuracy=val_accuracy
                )
            
            # 4. 记录模型评估结果
            logger.log_evaluation_metrics(
                experiment_id=experiment_id,
                accuracy=0.89,
                precision=0.87,
                recall=0.91,
                f1_score=0.89,
                auc_roc=0.93,
                additional_metrics={
                    "top_5_accuracy": 0.96,
                    "inference_time": 0.05
                }
            )
            
            # 5. 超参数搜索示例
            search_id = logger.start_hyperparameter_search(
                experiment_id=experiment_id,
                search_type="random",
                search_space={
                    "learning_rate": {"type": "float", "low": 0.0001, "high": 0.01},
                    "batch_size": {"type": "choice", "choices": [16, 32, 64]},
                    "dropout": {"type": "float", "low": 0.1, "high": 0.5}
                }
            )
            
            # 模拟超参数试验
            import random
            for trial in range(5):
                params = {
                    "learning_rate": random.uniform(0.0001, 0.01),
                    "batch_size": random.choice([16, 32, 64]),
                    "dropout": random.uniform(0.1, 0.5)
                }
                
                metrics = {
                    "accuracy": random.uniform(0.8, 0.95),
                    "val_loss": random.uniform(0.2, 0.5)
                }
                
                logger.log_hyperparameter_trial(
                    search_id=search_id,
                    parameters=params,
                    metrics=metrics
                )
            
            # 6. 注册模型版本
            version_id = logger.register_model_version(
                model_name="resnet_cifar10",
                version="1.0.0",
                file_path="/path/to/model.pth",
                metadata={
                    "architecture": "ResNet18",
                    "training_epochs": 100,
                    "final_accuracy": 0.89
                },
                experiment_id=experiment_id
            )
            
            print(f"模型版本ID: {version_id}")
            
            # 7. 生成实验报告
            report = logger.generate_experiment_report(experiment_id)
            print("实验报告生成完成")
            
            # 8. 实验对比
            # 创建第二个实验进行对比
            experiment_config_2 = {
                "name": "图像分类模型训练_v2",
                "model_type": "CNN",
                "description": "使用改进的ResNet进行图像分类",
                "hyperparameters": {
                    "learning_rate": 0.0005,
                    "batch_size": 64,
                    "epochs": 120
                },
                "tags": ["image-classification", "cnn", "resnet-v2"]
            }
            
            experiment_id_2 = logger.create_experiment(**experiment_config_2)
            logger.start_experiment(experiment_id_2)
            
            # 模拟第二个实验的评估结果
            logger.log_evaluation_metrics(
                experiment_id=experiment_id_2,
                accuracy=0.92,
                precision=0.90,
                recall=0.94,
                f1_score=0.92
            )
            
            # 对比实验
            comparison = logger.compare_experiments(
                [experiment_id, experiment_id_2],
                metrics=["accuracy", "f1_score"]
            )
            
            print("实验对比完成")
            print(f"对比结果: {comparison['summary']}")
            
    except Exception as e:
        print(f"示例运行出错: {e}")
        logger.error("Example", f"示例运行出错: {e}")
    
    finally:
        # 清理资源
        logger.cleanup()


def async_example_usage():
    """异步使用示例"""
    
    async def run_async_experiment():
        """运行异步实验"""
        logger = LearningLogger(
            base_dir="examples/async_learning_logs",
            enable_async=True
        )
        
        try:
            experiment_config = {
                "name": "异步图像分类实验",
                "model_type": "CNN",
                "description": "异步处理的图像分类实验",
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": 32
                }
            }
            
            async with logger.async_experiment_context(experiment_config) as experiment_id:
                print(f"异步实验ID: {experiment_id}")
                
                # 异步记录数据处理
                await asyncio.sleep(0.1)  # 模拟异步操作
                logger.log_data_processing(
                    stage=DataProcessingStage.LOADING,
                    dataset_name="AsyncDataset",
                    experiment_id=experiment_id
                )
                
                # 异步训练日志
                for i in range(5):
                    logger.log_training_metrics(
                        experiment_id=experiment_id,
                        epoch=1,
                        batch=i,
                        iteration=i,
                        loss=1.0 - i * 0.1,
                        accuracy=0.5 + i * 0.1
                    )
                    await asyncio.sleep(0.01)  # 模拟异步操作
                
                print("异步实验完成")
                
        except Exception as e:
            print(f"异步实验出错: {e}")
        finally:
            logger.cleanup()
    
    # 运行异步示例
    asyncio.run(async_example_usage())


if __name__ == "__main__":
    print("L5学习日志记录器使用示例")
    print("=" * 50)
    
    # 运行基础示例
    print("1. 运行基础使用示例...")
    example_usage()
    
    print("\n" + "=" * 50)
    
    # 运行异步示例
    print("2. 运行异步使用示例...")
    async_example_usage()
    
    print("\n" + "=" * 50)
    print("所有示例运行完成！")