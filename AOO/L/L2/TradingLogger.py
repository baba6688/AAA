"""
L2交易日志记录器

这是一个功能完整的L2交易日志记录器模块，提供全面的交易系统日志记录功能。
该模块支持异步处理、多种日志类型、统计分析、错误处理等功能。

功能特性：
1. 交易事件日志记录（订单、下单、成交、撤单）
2. 交易性能日志（延迟、成功率、成本分析）
3. 交易错误日志（失败原因、重试记录）
4. 交易策略日志（策略执行、参数变化）
5. 市场数据日志（数据接收、数据质量）
6. 交易统计日志（日汇总、周汇总、月汇总）
7. 异步交易日志处理
8. 完整的错误处理和日志记录
9. 详细的文档字符串和使用示例

"""

import asyncio
import json
import logging
import sqlite3
import threading
import time
import traceback
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from queue import Queue, Empty
from typing import Any, Dict, List, Optional, Union, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import os
import shutil
import weakref


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogType(Enum):
    """日志类型枚举"""
    TRADE_EVENT = "TRADE_EVENT"
    PERFORMANCE = "PERFORMANCE"
    ERROR = "ERROR"
    STRATEGY = "STRATEGY"
    MARKET_DATA = "MARKET_DATA"
    STATISTICS = "STATISTICS"
    SYSTEM = "SYSTEM"


class TradeEventType(Enum):
    """交易事件类型枚举"""
    ORDER_CREATED = "ORDER_CREATED"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    ORDER_PARTIALLY_FILLED = "ORDER_PARTIALLY_FILLED"
    ORDER_REJECTED = "ORDER_REJECTED"


class OrderSide(Enum):
    """订单方向枚举"""
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


@dataclass
class TradeEvent:
    """交易事件数据类"""
    event_id: str
    timestamp: datetime
    event_type: TradeEventType
    symbol: str
    order_id: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    filled_quantity: float = 0.0
    average_price: Optional[float] = None
    status: str = ""
    reason: str = ""
    client_id: str = ""
    strategy_id: str = ""
    exchange: str = ""
    commission: float = 0.0
    latency_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: datetime
    symbol: str
    strategy_id: str
    order_id: str
    latency_ms: float
    success: bool
    error_message: Optional[str] = None
    commission: float = 0.0
    slippage: float = 0.0
    market_impact: float = 0.0
    fill_ratio: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ErrorEvent:
    """错误事件数据类"""
    timestamp: datetime
    error_type: str
    error_message: str
    symbol: str
    order_id: Optional[str]
    strategy_id: str
    retry_count: int = 0
    max_retries: int = 3
    resolved: bool = False
    traceback_info: str = ""
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyEvent:
    """策略事件数据类"""
    timestamp: datetime
    strategy_id: str
    event_type: str
    symbol: str
    parameters: Dict[str, Any]
    execution_time_ms: float
    result: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketDataEvent:
    """市场数据事件数据类"""
    timestamp: datetime
    symbol: str
    data_type: str
    quality_score: float
    latency_ms: float
    source: str
    data_size: int
    missing_fields: List[str] = field(default_factory=list)
    anomalies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StatisticsSummary:
    """统计数据汇总数据类"""
    period_start: datetime
    period_end: datetime
    period_type: str  # "daily", "weekly", "monthly"
    total_trades: int
    successful_trades: int
    failed_trades: int
    total_volume: float
    total_commission: float
    total_pnl: float
    average_latency_ms: float
    max_latency_ms: float
    min_latency_ms: float
    win_rate: float
    profit_factor: float
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatabaseManager:
    """数据库管理器，负责SQLite数据库的创建和管理"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表结构"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 交易事件表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trade_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE,
                    timestamp TEXT,
                    event_type TEXT,
                    symbol TEXT,
                    order_id TEXT,
                    side TEXT,
                    order_type TEXT,
                    quantity REAL,
                    price REAL,
                    filled_quantity REAL,
                    average_price REAL,
                    status TEXT,
                    reason TEXT,
                    client_id TEXT,
                    strategy_id TEXT,
                    exchange TEXT,
                    commission REAL,
                    latency_ms REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 性能指标表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    strategy_id TEXT,
                    order_id TEXT,
                    latency_ms REAL,
                    success BOOLEAN,
                    error_message TEXT,
                    commission REAL,
                    slippage REAL,
                    market_impact REAL,
                    fill_ratio REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 错误事件表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS error_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    error_type TEXT,
                    error_message TEXT,
                    symbol TEXT,
                    order_id TEXT,
                    strategy_id TEXT,
                    retry_count INTEGER,
                    max_retries INTEGER,
                    resolved BOOLEAN,
                    traceback_info TEXT,
                    context TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 策略事件表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategy_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    strategy_id TEXT,
                    event_type TEXT,
                    symbol TEXT,
                    parameters TEXT,
                    execution_time_ms REAL,
                    result TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 市场数据事件表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS market_data_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    symbol TEXT,
                    data_type TEXT,
                    quality_score REAL,
                    latency_ms REAL,
                    source TEXT,
                    data_size INTEGER,
                    missing_fields TEXT,
                    anomalies TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 统计汇总表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS statistics_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    period_start TEXT,
                    period_end TEXT,
                    period_type TEXT,
                    total_trades INTEGER,
                    successful_trades INTEGER,
                    failed_trades INTEGER,
                    total_volume REAL,
                    total_commission REAL,
                    total_pnl REAL,
                    average_latency_ms REAL,
                    max_latency_ms REAL,
                    min_latency_ms REAL,
                    win_rate REAL,
                    profit_factor REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # 创建索引
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_events_timestamp ON trade_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_events_symbol ON trade_events(symbol)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trade_events_strategy ON trade_events(strategy_id)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_performance_timestamp ON performance_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_error_timestamp ON error_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_strategy_timestamp ON strategy_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_data_timestamp ON market_data_events(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_statistics_period ON statistics_summary(period_start, period_end)')
            
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def insert_trade_event(self, event: TradeEvent):
        """插入交易事件"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT OR REPLACE INTO trade_events 
                    (event_id, timestamp, event_type, symbol, order_id, side, order_type, 
                     quantity, price, filled_quantity, average_price, status, reason, 
                     client_id, strategy_id, exchange, commission, latency_ms, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id, event.timestamp.isoformat(), event.event_type.value,
                    event.symbol, event.order_id, event.side.value, event.order_type.value,
                    event.quantity, event.price, event.filled_quantity, event.average_price,
                    event.status, event.reason, event.client_id, event.strategy_id,
                    event.exchange, event.commission, event.latency_ms, json.dumps(event.metadata)
                ))
                conn.commit()
    
    def insert_performance_metric(self, metric: PerformanceMetrics):
        """插入性能指标"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO performance_metrics 
                    (timestamp, symbol, strategy_id, order_id, latency_ms, success, 
                     error_message, commission, slippage, market_impact, fill_ratio, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metric.timestamp.isoformat(), metric.symbol, metric.strategy_id,
                    metric.order_id, metric.latency_ms, metric.success, metric.error_message,
                    metric.commission, metric.slippage, metric.market_impact,
                    metric.fill_ratio, json.dumps(metric.metadata)
                ))
                conn.commit()
    
    def insert_error_event(self, error: ErrorEvent):
        """插入错误事件"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO error_events 
                    (timestamp, error_type, error_message, symbol, order_id, strategy_id,
                     retry_count, max_retries, resolved, traceback_info, context)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    error.timestamp.isoformat(), error.error_type, error.error_message,
                    error.symbol, error.order_id, error.strategy_id, error.retry_count,
                    error.max_retries, error.resolved, error.traceback_info,
                    json.dumps(error.context)
                ))
                conn.commit()
    
    def insert_strategy_event(self, event: StrategyEvent):
        """插入策略事件"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO strategy_events 
                    (timestamp, strategy_id, event_type, symbol, parameters, 
                     execution_time_ms, result, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.timestamp.isoformat(), event.strategy_id, event.event_type,
                    event.symbol, json.dumps(event.parameters), event.execution_time_ms,
                    json.dumps(event.result), json.dumps(event.metadata)
                ))
                conn.commit()
    
    def insert_market_data_event(self, event: MarketDataEvent):
        """插入市场数据事件"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO market_data_events 
                    (timestamp, symbol, data_type, quality_score, latency_ms, source,
                     data_size, missing_fields, anomalies, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.timestamp.isoformat(), event.symbol, event.data_type,
                    event.quality_score, event.latency_ms, event.source, event.data_size,
                    json.dumps(event.missing_fields), json.dumps(event.anomalies),
                    json.dumps(event.metadata)
                ))
                conn.commit()
    
    def insert_statistics_summary(self, summary: StatisticsSummary):
        """插入统计汇总"""
        with self.lock:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO statistics_summary 
                    (period_start, period_end, period_type, total_trades, successful_trades,
                     failed_trades, total_volume, total_commission, total_pnl, average_latency_ms,
                     max_latency_ms, min_latency_ms, win_rate, profit_factor, sharpe_ratio,
                     max_drawdown, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    summary.period_start.isoformat(), summary.period_end.isoformat(),
                    summary.period_type, summary.total_trades, summary.successful_trades,
                    summary.failed_trades, summary.total_volume, summary.total_commission,
                    summary.total_pnl, summary.average_latency_ms, summary.max_latency_ms,
                    summary.min_latency_ms, summary.win_rate, summary.profit_factor,
                    summary.sharpe_ratio, summary.max_drawdown, json.dumps(summary.metadata)
                ))
                conn.commit()


class AsyncLogProcessor:
    """异步日志处理器"""
    
    def __init__(self, max_workers: int = 4, batch_size: int = 100):
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_queue = Queue()
        self.is_processing = False
        self.processors = {
            LogType.TRADE_EVENT: self._process_trade_events,
            LogType.PERFORMANCE: self._process_performance_metrics,
            LogType.ERROR: self._process_error_events,
            LogType.STRATEGY: self._process_strategy_events,
            LogType.MARKET_DATA: self._process_market_data_events,
            LogType.STATISTICS: self._process_statistics_summaries,
        }
    
    def start_processing(self):
        """启动异步处理"""
        if not self.is_processing:
            self.is_processing = True
            self.executor.submit(self._process_queue)
    
    def stop_processing(self):
        """停止异步处理"""
        self.is_processing = False
    
    def add_log_entry(self, log_type: LogType, data: Any):
        """添加日志条目到处理队列"""
        self.processing_queue.put((log_type, data))
    
    def _process_queue(self):
        """处理队列中的日志条目"""
        batch = []
        while self.is_processing:
            try:
                # 批量收集日志条目
                timeout = 1.0  # 1秒超时
                start_time = time.time()
                
                while len(batch) < self.batch_size and (time.time() - start_time) < timeout:
                    try:
                        item = self.processing_queue.get(timeout=0.1)
                        batch.append(item)
                    except Empty:
                        break
                
                if batch:
                    # 批量处理
                    self._process_batch(batch)
                    batch.clear()
                
            except Exception as e:
                logging.error(f"异步日志处理错误: {e}")
                logging.error(traceback.format_exc())
    
    def _process_batch(self, batch: List[Tuple[LogType, Any]]):
        """批量处理日志条目"""
        # 按日志类型分组
        grouped = defaultdict(list)
        for log_type, data in batch:
            grouped[log_type].append(data)
        
        # 并行处理不同类型的日志
        futures = []
        for log_type, data_list in grouped.items():
            if log_type in self.processors:
                future = self.executor.submit(self.processors[log_type], data_list)
                futures.append(future)
        
        # 等待所有处理完成
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"批量处理错误: {e}")
    
    def _process_trade_events(self, events: List[TradeEvent]):
        """处理交易事件"""
        # 这里可以实现具体的处理逻辑
        # 比如发送到外部系统、进行实时分析等
        pass
    
    def _process_performance_metrics(self, metrics: List[PerformanceMetrics]):
        """处理性能指标"""
        # 这里可以实现性能监控和告警逻辑
        pass
    
    def _process_error_events(self, errors: List[ErrorEvent]):
        """处理错误事件"""
        # 这里可以实现错误告警和自动恢复逻辑
        pass
    
    def _process_strategy_events(self, events: List[StrategyEvent]):
        """处理策略事件"""
        # 这里可以实现策略监控和分析逻辑
        pass
    
    def _process_market_data_events(self, events: List[MarketDataEvent]):
        """处理市场数据事件"""
        # 这里可以实现数据质量监控逻辑
        pass
    
    def _process_statistics_summaries(self, summaries: List[StatisticsSummary]):
        """处理统计汇总"""
        # 这里可以实现报表生成和分发逻辑
        pass


class LogRotator:
    """日志轮转器"""
    
    def __init__(self, log_dir: str, max_size_mb: int = 100, backup_count: int = 5):
        self.log_dir = Path(log_dir)
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.backup_count = backup_count
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def rotate_logs(self):
        """轮转日志文件"""
        for log_file in self.log_dir.glob("*.log*"):
            if log_file.stat().st_size > self.max_size_bytes:
                self._rotate_single_file(log_file)
    
    def _rotate_single_file(self, log_file: Path):
        """轮转单个日志文件"""
        # 压缩旧文件
        if log_file.suffix == '.log':
            compressed_file = log_file.with_suffix('.log.gz')
            with open(log_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            log_file.unlink()
        
        # 轮转压缩文件
        if log_file.suffix == '.gz':
            for i in range(self.backup_count - 1, 0, -1):
                old_file = log_file.with_suffix(f'.log.{i}.gz')
                new_file = log_file.with_suffix(f'.log.{i + 1}.gz')
                if old_file.exists():
                    if new_file.exists():
                        new_file.unlink()
                    old_file.rename(new_file)
            
            # 第一个压缩文件
            first_backup = log_file.with_suffix('.log.1.gz')
            if first_backup.exists():
                first_backup.unlink()
            log_file.rename(first_backup)


class TradingLogger:
    """
    L2交易日志记录器主类
    
    提供完整的交易系统日志记录功能，包括：
    - 交易事件记录
    - 性能监控
    - 错误跟踪
    - 策略分析
    - 市场数据监控
    - 统计分析
    - 异步处理
    """
    
    def __init__(self, 
                 log_dir: str = "./logs",
                 db_path: str = "./logs/trading_logger.db",
                 enable_async: bool = True,
                 async_workers: int = 4,
                 batch_size: int = 100,
                 log_level: LogLevel = LogLevel.INFO,
                 enable_rotation: bool = True,
                 max_log_size_mb: int = 100,
                 backup_count: int = 5):
        """
        初始化交易日志记录器
        
        Args:
            log_dir: 日志文件目录
            db_path: 数据库文件路径
            enable_async: 是否启用异步处理
            async_workers: 异步工作线程数
            batch_size: 批处理大小
            log_level: 日志级别
            enable_rotation: 是否启用日志轮转
            max_log_size_mb: 最大日志文件大小(MB)
            backup_count: 备份文件数量
        """
        self.log_dir = Path(log_dir)
        self.db_path = db_path
        self.enable_async = enable_async
        self.batch_size = batch_size
        self.log_level = log_level
        self.enable_rotation = enable_rotation
        
        # 创建目录
        self.log_dir.mkdir(parents=True, exist_ok=True)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.db_manager = DatabaseManager(self.db_path)
        self.async_processor = None
        self.log_rotator = None
        
        if self.enable_async:
            self.async_processor = AsyncLogProcessor(async_workers, batch_size)
            self.async_processor.start_processing()
        
        if self.enable_rotation:
            self.log_rotator = LogRotator(str(self.log_dir), max_log_size_mb, backup_count)
        
        # 初始化日志记录器
        self._setup_logging()
        
        # 内存缓存
        self._recent_events = defaultdict(lambda: deque(maxlen=1000))
        self._performance_cache = defaultdict(list)
        self._error_cache = defaultdict(list)
        
        # 统计缓存
        self._daily_stats = {}
        self._weekly_stats = {}
        self._monthly_stats = {}
        
        # 事件回调
        self._event_callbacks = defaultdict(list)
        
        self.logger.info("交易日志记录器初始化完成")
    
    def _setup_logging(self):
        """设置日志记录器"""
        self.logger = logging.getLogger("TradingLogger")
        self.logger.setLevel(getattr(logging, self.log_level.value))
        
        # 清除现有处理器
        self.logger.handlers.clear()
        
        # 文件处理器
        file_handler = logging.FileHandler(
            self.log_dir / f"trading_logger_{datetime.now().strftime('%Y%m%d')}.log",
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, self.log_level.value))
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.log_level.value))
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def log_trade_event(self, 
                       event_type: TradeEventType,
                       symbol: str,
                       order_id: str,
                       side: OrderSide,
                       order_type: OrderType,
                       quantity: float,
                       price: Optional[float] = None,
                       filled_quantity: float = 0.0,
                       average_price: Optional[float] = None,
                       status: str = "",
                       reason: str = "",
                       client_id: str = "",
                       strategy_id: str = "",
                       exchange: str = "",
                       commission: float = 0.0,
                       latency_ms: float = 0.0,
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        记录交易事件
        
        Args:
            event_type: 事件类型
            symbol: 交易标的
            order_id: 订单ID
            side: 买卖方向
            order_type: 订单类型
            quantity: 数量
            price: 价格
            filled_quantity: 成交数量
            average_price: 平均价格
            status: 状态
            reason: 原因
            client_id: 客户端ID
            strategy_id: 策略ID
            exchange: 交易所
            commission: 手续费
            latency_ms: 延迟(毫秒)
            metadata: 元数据
        
        Returns:
            str: 事件ID
        """
        event_id = f"TE_{int(time.time() * 1000000)}"
        
        event = TradeEvent(
            event_id=event_id,
            timestamp=datetime.now(),
            event_type=event_type,
            symbol=symbol,
            order_id=order_id,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            filled_quantity=filled_quantity,
            average_price=average_price,
            status=status,
            reason=reason,
            client_id=client_id,
            strategy_id=strategy_id,
            exchange=exchange,
            commission=commission,
            latency_ms=latency_ms,
            metadata=metadata or {}
        )
        
        try:
            # 存储到数据库
            self.db_manager.insert_trade_event(event)
            
            # 添加到缓存
            self._recent_events['trade_events'].append(event)
            
            # 异步处理
            if self.enable_async and self.async_processor:
                self.async_processor.add_log_entry(LogType.TRADE_EVENT, event)
            
            # 触发回调
            self._trigger_callbacks('trade_event', event)
            
            self.logger.info(f"交易事件记录成功: {event_type.value} - {symbol} - {order_id}")
            
        except Exception as e:
            self.logger.error(f"记录交易事件失败: {e}")
            self.logger.error(traceback.format_exc())
        
        return event_id
    
    def log_performance_metric(self,
                              symbol: str,
                              strategy_id: str,
                              order_id: str,
                              latency_ms: float,
                              success: bool,
                              error_message: Optional[str] = None,
                              commission: float = 0.0,
                              slippage: float = 0.0,
                              market_impact: float = 0.0,
                              fill_ratio: float = 0.0,
                              metadata: Optional[Dict[str, Any]] = None):
        """
        记录性能指标
        
        Args:
            symbol: 交易标的
            strategy_id: 策略ID
            order_id: 订单ID
            latency_ms: 延迟(毫秒)
            success: 是否成功
            error_message: 错误信息
            commission: 手续费
            slippage: 滑点
            market_impact: 市场冲击
            fill_ratio: 成交比例
            metadata: 元数据
        """
        metric = PerformanceMetrics(
            timestamp=datetime.now(),
            symbol=symbol,
            strategy_id=strategy_id,
            order_id=order_id,
            latency_ms=latency_ms,
            success=success,
            error_message=error_message,
            commission=commission,
            slippage=slippage,
            market_impact=market_impact,
            fill_ratio=fill_ratio,
            metadata=metadata or {}
        )
        
        try:
            # 存储到数据库
            self.db_manager.insert_performance_metric(metric)
            
            # 添加到缓存
            self._performance_cache[f"{symbol}_{strategy_id}"].append(metric)
            
            # 异步处理
            if self.enable_async and self.async_processor:
                self.async_processor.add_log_entry(LogType.PERFORMANCE, metric)
            
            # 触发回调
            self._trigger_callbacks('performance_metric', metric)
            
            if success:
                self.logger.debug(f"性能指标记录成功: {symbol} - 延迟{latency_ms}ms")
            else:
                self.logger.warning(f"性能指标记录失败: {symbol} - {error_message}")
            
        except Exception as e:
            self.logger.error(f"记录性能指标失败: {e}")
            self.logger.error(traceback.format_exc())
    
    def log_error_event(self,
                       error_type: str,
                       error_message: str,
                       symbol: str,
                       strategy_id: str,
                       order_id: Optional[str] = None,
                       retry_count: int = 0,
                       max_retries: int = 3,
                       resolved: bool = False,
                       context: Optional[Dict[str, Any]] = None):
        """
        记录错误事件
        
        Args:
            error_type: 错误类型
            error_message: 错误信息
            symbol: 交易标的
            strategy_id: 策略ID
            order_id: 订单ID
            retry_count: 重试次数
            max_retries: 最大重试次数
            resolved: 是否已解决
            context: 上下文信息
        """
        error = ErrorEvent(
            timestamp=datetime.now(),
            error_type=error_type,
            error_message=error_message,
            symbol=symbol,
            order_id=order_id,
            strategy_id=strategy_id,
            retry_count=retry_count,
            max_retries=max_retries,
            resolved=resolved,
            traceback_info=traceback.format_exc(),
            context=context or {}
        )
        
        try:
            # 存储到数据库
            self.db_manager.insert_error_event(error)
            
            # 添加到缓存
            self._error_cache[symbol].append(error)
            
            # 异步处理
            if self.enable_async and self.async_processor:
                self.async_processor.add_log_entry(LogType.ERROR, error)
            
            # 触发回调
            self._trigger_callbacks('error_event', error)
            
            # 错误告警
            if not resolved and retry_count >= max_retries:
                self.logger.error(f"严重错误: {error_type} - {error_message} - {symbol}")
            else:
                self.logger.warning(f"错误记录: {error_type} - {error_message}")
            
        except Exception as e:
            self.logger.error(f"记录错误事件失败: {e}")
            self.logger.error(traceback.format_exc())
    
    def log_strategy_event(self,
                          strategy_id: str,
                          event_type: str,
                          symbol: str,
                          parameters: Dict[str, Any],
                          execution_time_ms: float,
                          result: Any,
                          metadata: Optional[Dict[str, Any]] = None):
        """
        记录策略事件
        
        Args:
            strategy_id: 策略ID
            event_type: 事件类型
            symbol: 交易标的
            parameters: 参数
            execution_time_ms: 执行时间(毫秒)
            result: 结果
            metadata: 元数据
        """
        event = StrategyEvent(
            timestamp=datetime.now(),
            strategy_id=strategy_id,
            event_type=event_type,
            symbol=symbol,
            parameters=parameters,
            execution_time_ms=execution_time_ms,
            result=result,
            metadata=metadata or {}
        )
        
        try:
            # 存储到数据库
            self.db_manager.insert_strategy_event(event)
            
            # 异步处理
            if self.enable_async and self.async_processor:
                self.async_processor.add_log_entry(LogType.STRATEGY, event)
            
            # 触发回调
            self._trigger_callbacks('strategy_event', event)
            
            self.logger.debug(f"策略事件记录: {strategy_id} - {event_type} - {symbol}")
            
        except Exception as e:
            self.logger.error(f"记录策略事件失败: {e}")
            self.logger.error(traceback.format_exc())
    
    def log_market_data_event(self,
                             symbol: str,
                             data_type: str,
                             quality_score: float,
                             latency_ms: float,
                             source: str,
                             data_size: int,
                             missing_fields: Optional[List[str]] = None,
                             anomalies: Optional[List[str]] = None,
                             metadata: Optional[Dict[str, Any]] = None):
        """
        记录市场数据事件
        
        Args:
            symbol: 交易标的
            data_type: 数据类型
            quality_score: 质量评分
            latency_ms: 延迟(毫秒)
            source: 数据源
            data_size: 数据大小
            missing_fields: 缺失字段
            anomalies: 异常
            metadata: 元数据
        """
        event = MarketDataEvent(
            timestamp=datetime.now(),
            symbol=symbol,
            data_type=data_type,
            quality_score=quality_score,
            latency_ms=latency_ms,
            source=source,
            data_size=data_size,
            missing_fields=missing_fields or [],
            anomalies=anomalies or [],
            metadata=metadata or {}
        )
        
        try:
            # 存储到数据库
            self.db_manager.insert_market_data_event(event)
            
            # 异步处理
            if self.enable_async and self.async_processor:
                self.async_processor.add_log_entry(LogType.MARKET_DATA, event)
            
            # 触发回调
            self._trigger_callbacks('market_data_event', event)
            
            # 数据质量告警
            if quality_score < 0.5:
                self.logger.warning(f"数据质量告警: {symbol} - 质量评分{quality_score}")
            
            self.logger.debug(f"市场数据事件记录: {symbol} - {data_type}")
            
        except Exception as e:
            self.logger.error(f"记录市场数据事件失败: {e}")
            self.logger.error(traceback.format_exc())
    
    def generate_statistics(self, 
                           period_type: str = "daily",
                           start_date: Optional[datetime] = None,
                           end_date: Optional[datetime] = None) -> StatisticsSummary:
        """
        生成统计汇总
        
        Args:
            period_type: 统计周期类型 (daily, weekly, monthly)
            start_date: 开始日期
            end_date: 结束日期
        
        Returns:
            StatisticsSummary: 统计汇总结果
        """
        if start_date is None:
            if period_type == "daily":
                start_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            elif period_type == "weekly":
                start_date = datetime.now() - timedelta(days=datetime.now().weekday())
                start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
            elif period_type == "monthly":
                start_date = datetime.now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        if end_date is None:
            end_date = datetime.now()
        
        try:
            # 从数据库查询数据
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # 查询交易事件
                cursor.execute('''
                    SELECT * FROM trade_events 
                    WHERE timestamp BETWEEN ? AND ?
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                trade_events = [dict(row) for row in cursor.fetchall()]
                
                # 查询性能指标
                cursor.execute('''
                    SELECT * FROM performance_metrics 
                    WHERE timestamp BETWEEN ? AND ?
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                performance_metrics = [dict(row) for row in cursor.fetchall()]
            
            # 计算统计数据
            total_trades = len(trade_events)
            successful_trades = sum(1 for event in trade_events 
                                  if event.get('status') == 'FILLED')
            failed_trades = total_trades - successful_trades
            
            total_volume = sum(event.get('quantity', 0) for event in trade_events)
            total_commission = sum(event.get('commission', 0) for event in trade_events)
            
            # 计算延迟统计
            latencies = [metric.get('latency_ms', 0) for metric in performance_metrics 
                        if metric.get('latency_ms') is not None]
            avg_latency = sum(latencies) / len(latencies) if latencies else 0
            max_latency = max(latencies) if latencies else 0
            min_latency = min(latencies) if latencies else 0
            
            # 计算成功率
            success_rate = successful_trades / total_trades if total_trades > 0 else 0
            
            # 计算盈亏比 (简化计算)
            profits = [event.get('average_price', 0) * event.get('filled_quantity', 0) 
                      for event in trade_events if event.get('status') == 'FILLED']
            total_pnl = sum(profits) - total_commission
            profit_factor = abs(sum(p for p in profits if p > 0) / sum(p for p in profits if p < 0)) if any(p < 0 for p in profits) else float('inf')
            
            # 创建统计汇总
            summary = StatisticsSummary(
                period_start=start_date,
                period_end=end_date,
                period_type=period_type,
                total_trades=total_trades,
                successful_trades=successful_trades,
                failed_trades=failed_trades,
                total_volume=total_volume,
                total_commission=total_commission,
                total_pnl=total_pnl,
                average_latency_ms=avg_latency,
                max_latency_ms=max_latency,
                min_latency_ms=min_latency,
                win_rate=success_rate,
                profit_factor=profit_factor
            )
            
            # 存储到数据库
            self.db_manager.insert_statistics_summary(summary)
            
            # 异步处理
            if self.enable_async and self.async_processor:
                self.async_processor.add_log_entry(LogType.STATISTICS, summary)
            
            self.logger.info(f"统计汇总生成完成: {period_type} - {start_date.date()} 到 {end_date.date()}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"生成统计汇总失败: {e}")
            self.logger.error(traceback.format_exc())
            raise
    
    def get_recent_events(self, event_type: str = "trade_events", limit: int = 100) -> List[Any]:
        """
        获取最近的事件
        
        Args:
            event_type: 事件类型
            limit: 返回数量限制
        
        Returns:
            List[Any]: 事件列表
        """
        events = self._recent_events.get(event_type, [])
        return list(events)[-limit:]
    
    def get_performance_summary(self, 
                               symbol: Optional[str] = None,
                               strategy_id: Optional[str] = None,
                               hours: int = 24) -> Dict[str, Any]:
        """
        获取性能摘要
        
        Args:
            symbol: 交易标的
            strategy_id: 策略ID
            hours: 时间范围(小时)
        
        Returns:
            Dict[str, Any]: 性能摘要
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT * FROM performance_metrics 
                    WHERE timestamp BETWEEN ? AND ?
                '''
                params = [start_time.isoformat(), end_time.isoformat()]
                
                if symbol:
                    query += ' AND symbol = ?'
                    params.append(symbol)
                
                if strategy_id:
                    query += ' AND strategy_id = ?'
                    params.append(strategy_id)
                
                cursor.execute(query, params)
                metrics = [dict(row) for row in cursor.fetchall()]
            
            if not metrics:
                return {}
            
            # 计算统计信息
            latencies = [m.get('latency_ms', 0) for m in metrics]
            success_count = sum(1 for m in metrics if m.get('success', False))
            total_count = len(metrics)
            
            summary = {
                'period_hours': hours,
                'total_orders': total_count,
                'successful_orders': success_count,
                'failed_orders': total_count - success_count,
                'success_rate': success_count / total_count if total_count > 0 else 0,
                'average_latency_ms': sum(latencies) / len(latencies) if latencies else 0,
                'max_latency_ms': max(latencies) if latencies else 0,
                'min_latency_ms': min(latencies) if latencies else 0,
                'total_commission': sum(m.get('commission', 0) for m in metrics),
                'total_slippage': sum(m.get('slippage', 0) for m in metrics),
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取性能摘要失败: {e}")
            self.logger.error(traceback.format_exc())
            return {}
    
    def get_error_summary(self, 
                         symbol: Optional[str] = None,
                         hours: int = 24) -> Dict[str, Any]:
        """
        获取错误摘要
        
        Args:
            symbol: 交易标的
            hours: 时间范围(小时)
        
        Returns:
            Dict[str, Any]: 错误摘要
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=hours)
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT * FROM error_events 
                    WHERE timestamp BETWEEN ? AND ?
                '''
                params = [start_time.isoformat(), end_time.isoformat()]
                
                if symbol:
                    query += ' AND symbol = ?'
                    params.append(symbol)
                
                cursor.execute(query, params)
                errors = [dict(row) for row in cursor.fetchall()]
            
            if not errors:
                return {}
            
            # 统计错误类型
            error_types = defaultdict(int)
            unresolved_errors = []
            
            for error in errors:
                error_types[error.get('error_type', 'UNKNOWN')] += 1
                if not error.get('resolved', False):
                    unresolved_errors.append(error)
            
            summary = {
                'period_hours': hours,
                'total_errors': len(errors),
                'unresolved_errors': len(unresolved_errors),
                'error_types': dict(error_types),
                'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None,
                'resolution_rate': (len(errors) - len(unresolved_errors)) / len(errors) if errors else 0,
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"获取错误摘要失败: {e}")
            self.logger.error(traceback.format_exc())
            return {}
    
    def register_event_callback(self, event_type: str, callback: Callable):
        """
        注册事件回调函数
        
        Args:
            event_type: 事件类型
            callback: 回调函数
        """
        self._event_callbacks[event_type].append(callback)
    
    def _trigger_callbacks(self, event_type: str, event_data: Any):
        """
        触发事件回调
        
        Args:
            event_type: 事件类型
            event_data: 事件数据
        """
        callbacks = self._event_callbacks.get(event_type, [])
        for callback in callbacks:
            try:
                callback(event_data)
            except Exception as e:
                self.logger.error(f"回调函数执行失败: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """
        清理旧数据
        
        Args:
            days_to_keep: 保留天数
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # 清理旧数据
                tables = [
                    'trade_events', 'performance_metrics', 'error_events',
                    'strategy_events', 'market_data_events'
                ]
                
                for table in tables:
                    cursor.execute(f'''
                        DELETE FROM {table} 
                        WHERE timestamp < ?
                    ''', (cutoff_date.isoformat(),))
                    
                    deleted_count = cursor.rowcount
                    self.logger.info(f"从{table}表中清理了{deleted_count}条{days_to_keep}天前的记录")
                
                conn.commit()
            
            # 执行数据库优化
            with self.db_manager.get_connection() as conn:
                conn.execute('VACUUM')
            
            self.logger.info("数据库清理和优化完成")
            
        except Exception as e:
            self.logger.error(f"清理旧数据失败: {e}")
            self.logger.error(traceback.format_exc())
    
    def export_logs(self, 
                   start_date: datetime,
                   end_date: datetime,
                   output_file: str,
                   format: str = "json") -> bool:
        """
        导出日志数据
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            output_file: 输出文件路径
            format: 导出格式 (json, csv)
        
        Returns:
            bool: 是否成功
        """
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # 查询所有相关数据
                cursor.execute('''
                    SELECT 'trade_events' as table_name, * FROM trade_events 
                    WHERE timestamp BETWEEN ? AND ?
                    UNION ALL
                    SELECT 'performance_metrics' as table_name, * FROM performance_metrics 
                    WHERE timestamp BETWEEN ? AND ?
                    UNION ALL
                    SELECT 'error_events' as table_name, * FROM error_events 
                    WHERE timestamp BETWEEN ? AND ?
                    UNION ALL
                    SELECT 'strategy_events' as table_name, * FROM strategy_events 
                    WHERE timestamp BETWEEN ? AND ?
                    UNION ALL
                    SELECT 'market_data_events' as table_name, * FROM market_data_events 
                    WHERE timestamp BETWEEN ? AND ?
                ''', (
                    start_date.isoformat(), end_date.isoformat(),
                    start_date.isoformat(), end_date.isoformat(),
                    start_date.isoformat(), end_date.isoformat(),
                    start_date.isoformat(), end_date.isoformat(),
                    start_date.isoformat(), end_date.isoformat()
                ))
                
                data = [dict(row) for row in cursor.fetchall()]
            
            # 导出数据
            if format.lower() == "json":
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            elif format.lower() == "csv":
                import csv
                with open(output_file, 'w', newline='', encoding='utf-8') as f:
                    if data:
                        writer = csv.DictWriter(f, fieldnames=data[0].keys())
                        writer.writeheader()
                        writer.writerows(data)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
            
            self.logger.info(f"日志数据导出成功: {output_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"导出日志数据失败: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def rotate_logs(self):
        """手动触发日志轮转"""
        if self.log_rotator:
            self.log_rotator.rotate_logs()
            self.logger.info("日志轮转完成")
    
    def shutdown(self):
        """关闭日志记录器"""
        try:
            # 停止异步处理
            if self.async_processor:
                self.async_processor.stop_processing()
            
            # 执行日志轮转
            if self.enable_rotation:
                self.rotate_logs()
            
            # 清理旧数据
            self.cleanup_old_data()
            
            self.logger.info("交易日志记录器已关闭")
            
        except Exception as e:
            self.logger.error(f"关闭日志记录器时发生错误: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()


# 使用示例和测试代码
def example_usage():
    """使用示例"""
    
    # 创建日志记录器实例
    with TradingLogger(
        log_dir="./logs",
        db_path="./logs/trading_logger.db",
        enable_async=True,
        async_workers=4,
        log_level=LogLevel.INFO
    ) as logger:
        
        # 示例1: 记录交易事件
        order_id = logger.log_trade_event(
            event_type=TradeEventType.ORDER_CREATED,
            symbol="AAPL",
            order_id="ORD_001",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100,
            price=150.0,
            strategy_id="STRAT_001",
            exchange="NASDAQ"
        )
        
        # 示例2: 记录性能指标
        logger.log_performance_metric(
            symbol="AAPL",
            strategy_id="STRAT_001",
            order_id="ORD_001",
            latency_ms=25.5,
            success=True,
            commission=1.0,
            slippage=0.5
        )
        
        # 示例3: 记录错误事件
        logger.log_error_event(
            error_type="NETWORK_ERROR",
            error_message="连接超时",
            symbol="AAPL",
            strategy_id="STRAT_001",
            order_id="ORD_001",
            retry_count=1,
            max_retries=3
        )
        
        # 示例4: 记录策略事件
        logger.log_strategy_event(
            strategy_id="STRAT_001",
            event_type="PARAMETER_UPDATE",
            symbol="AAPL",
            parameters={"stop_loss": 0.02, "take_profit": 0.05},
            execution_time_ms=15.2,
            result={"status": "success", "old_params": {"stop_loss": 0.03}}
        )
        
        # 示例5: 记录市场数据事件
        logger.log_market_data_event(
            symbol="AAPL",
            data_type="TICK",
            quality_score=0.95,
            latency_ms=10.0,
            source="REALTIME_FEED",
            data_size=1024,
            missing_fields=[],
            anomalies=[]
        )
        
        # 示例6: 生成统计汇总
        daily_stats = logger.generate_statistics("daily")
        print(f"日统计: 总交易{daily_stats.total_trades}笔, 成功率{daily_stats.win_rate:.2%}")
        
        # 示例7: 获取性能摘要
        perf_summary = logger.get_performance_summary(symbol="AAPL", hours=24)
        print(f"性能摘要: 平均延迟{perf_summary.get('average_latency_ms', 0):.1f}ms")
        
        # 示例8: 获取错误摘要
        error_summary = logger.get_error_summary(symbol="AAPL", hours=24)
        print(f"错误摘要: 总错误{error_summary.get('total_errors', 0)}个")
        
        # 示例9: 注册事件回调
        def on_trade_event(event):
            print(f"交易事件回调: {event.event_type.value} - {event.symbol}")
        
        logger.register_event_callback('trade_event', on_trade_event)
        
        # 示例10: 导出日志数据
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        logger.export_logs(start_date, end_date, "./logs/export_7days.json")


def performance_test():
    """性能测试"""
    import time
    
    with TradingLogger(
        log_dir="./logs",
        db_path="./logs/performance_test.db",
        enable_async=True,
        async_workers=8
    ) as logger:
        
        print("开始性能测试...")
        
        # 测试大量交易事件记录
        start_time = time.time()
        for i in range(1000):
            logger.log_trade_event(
                event_type=TradeEventType.ORDER_FILLED,
                symbol=f"SYM_{i % 10}",
                order_id=f"ORD_{i}",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=100,
                price=100.0 + i * 0.1,
                filled_quantity=100,
                average_price=100.0 + i * 0.1,
                strategy_id=f"STRAT_{i % 5}"
            )
        
        end_time = time.time()
        print(f"记录1000个交易事件耗时: {end_time - start_time:.2f}秒")
        
        # 等待异步处理完成
        time.sleep(2)
        
        # 生成统计
        stats = logger.generate_statistics("daily")
        print(f"统计结果: 总交易{stats.total_trades}笔")


class AdvancedAnalytics:
    """高级分析模块"""
    
    def __init__(self, logger: TradingLogger):
        self.logger = logger
        self.db_manager = logger.db_manager
    
    def analyze_trade_patterns(self, 
                              symbol: str,
                              days: int = 30) -> Dict[str, Any]:
        """分析交易模式"""
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                
                # 查询交易数据
                cursor.execute('''
                    SELECT * FROM trade_events 
                    WHERE symbol = ? AND timestamp BETWEEN ? AND ?
                    ORDER BY timestamp
                ''', (symbol, start_time.isoformat(), end_time.isoformat()))
                
                trades = [dict(row) for row in cursor.fetchall()]
            
            if not trades:
                return {}
            
            # 分析交易模式
            patterns = {
                'total_trades': len(trades),
                'buy_vs_sell': self._analyze_buy_sell_ratio(trades),
                'time_distribution': self._analyze_time_distribution(trades),
                'price_levels': self._analyze_price_levels(trades),
                'volume_patterns': self._analyze_volume_patterns(trades),
                'success_rate_by_hour': self._analyze_success_by_hour(trades),
                'average_fill_time': self._analyze_fill_time(trades),
            }
            
            return patterns
            
        except Exception as e:
            self.logger.logger.error(f"分析交易模式失败: {e}")
            return {}
    
    def _analyze_buy_sell_ratio(self, trades: List[Dict]) -> Dict[str, float]:
        """分析买卖比例"""
        buy_count = sum(1 for t in trades if t.get('side') == 'BUY')
        sell_count = sum(1 for t in trades if t.get('side') == 'SELL')
        total = len(trades)
        
        return {
            'buy_ratio': buy_count / total if total > 0 else 0,
            'sell_ratio': sell_count / total if total > 0 else 0,
            'buy_count': buy_count,
            'sell_count': sell_count
        }
    
    def _analyze_time_distribution(self, trades: List[Dict]) -> Dict[str, int]:
        """分析时间分布"""
        hour_distribution = defaultdict(int)
        
        for trade in trades:
            try:
                timestamp = datetime.fromisoformat(trade.get('timestamp', ''))
                hour = timestamp.hour
                hour_distribution[hour] += 1
            except:
                continue
        
        return dict(hour_distribution)
    
    def _analyze_price_levels(self, trades: List[Dict]) -> Dict[str, float]:
        """分析价格水平"""
        prices = [t.get('price', 0) for t in trades if t.get('price') is not None]
        
        if not prices:
            return {}
        
        return {
            'min_price': min(prices),
            'max_price': max(prices),
            'avg_price': sum(prices) / len(prices),
            'price_volatility': self._calculate_volatility(prices)
        }
    
    def _analyze_volume_patterns(self, trades: List[Dict]) -> Dict[str, float]:
        """分析成交量模式"""
        volumes = [t.get('quantity', 0) for t in trades if t.get('quantity') is not None]
        
        if not volumes:
            return {}
        
        return {
            'total_volume': sum(volumes),
            'avg_volume': sum(volumes) / len(volumes),
            'max_volume': max(volumes),
            'min_volume': min(volumes),
            'volume_volatility': self._calculate_volatility(volumes)
        }
    
    def _analyze_success_by_hour(self, trades: List[Dict]) -> Dict[int, float]:
        """分析按小时的成功率"""
        hour_stats = defaultdict(lambda: {'total': 0, 'successful': 0})
        
        for trade in trades:
            try:
                timestamp = datetime.fromisoformat(trade.get('timestamp', ''))
                hour = timestamp.hour
                hour_stats[hour]['total'] += 1
                if trade.get('status') == 'FILLED':
                    hour_stats[hour]['successful'] += 1
            except:
                continue
        
        return {
            hour: stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            for hour, stats in hour_stats.items()
        }
    
    def _analyze_fill_time(self, trades: List[Dict]) -> Dict[str, float]:
        """分析成交时间"""
        fill_times = [t.get('latency_ms', 0) for t in trades if t.get('latency_ms') is not None]
        
        if not fill_times:
            return {}
        
        return {
            'avg_fill_time_ms': sum(fill_times) / len(fill_times),
            'min_fill_time_ms': min(fill_times),
            'max_fill_time_ms': max(fill_times),
            'fill_time_std': self._calculate_std_deviation(fill_times)
        }
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """计算波动率"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _calculate_std_deviation(self, values: List[float]) -> float:
        """计算标准差"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5


class AlertManager:
    """告警管理器"""
    
    def __init__(self, logger: TradingLogger):
        self.logger = logger
        self.db_manager = logger.db_manager
        self.alert_rules = {}
        self.alert_history = deque(maxlen=1000)
        self.alert_callbacks = []
    
    def add_alert_rule(self, 
                      rule_name: str,
                      condition_func: Callable,
                      alert_message: str,
                      severity: str = "WARNING",
                      cooldown_seconds: int = 300):
        """添加告警规则"""
        self.alert_rules[rule_name] = {
            'condition': condition_func,
            'message': alert_message,
            'severity': severity,
            'cooldown': cooldown_seconds,
            'last_triggered': None
        }
    
    def check_alerts(self):
        """检查告警条件"""
        current_time = datetime.now()
        
        for rule_name, rule in self.alert_rules.items():
            try:
                # 检查冷却时间
                if (rule['last_triggered'] and 
                    (current_time - rule['last_triggered']).total_seconds() < rule['cooldown']):
                    continue
                
                # 检查条件
                if rule['condition']():
                    self._trigger_alert(rule_name, rule)
                    rule['last_triggered'] = current_time
                    
            except Exception as e:
                self.logger.logger.error(f"检查告警规则失败 {rule_name}: {e}")
    
    def _trigger_alert(self, rule_name: str, rule: Dict):
        """触发告警"""
        alert = {
            'timestamp': datetime.now(),
            'rule_name': rule_name,
            'message': rule['message'],
            'severity': rule['severity']
        }
        
        self.alert_history.append(alert)
        
        # 记录日志
        if rule['severity'] == 'CRITICAL':
            self.logger.logger.critical(f"告警 [{rule_name}]: {rule['message']}")
        elif rule['severity'] == 'ERROR':
            self.logger.logger.error(f"告警 [{rule_name}]: {rule['message']}")
        else:
            self.logger.logger.warning(f"告警 [{rule_name}]: {rule['message']}")
        
        # 触发回调
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.logger.error(f"告警回调执行失败: {e}")
    
    def register_alert_callback(self, callback: Callable):
        """注册告警回调"""
        self.alert_callbacks.append(callback)
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """获取最近的告警"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert['timestamp'] > cutoff_time]


class DataValidator:
    """数据验证器"""
    
    def __init__(self, logger: TradingLogger):
        self.logger = logger
        self.validation_rules = {}
        self.validation_stats = defaultdict(int)
    
    def add_validation_rule(self, 
                           data_type: str,
                           field_name: str,
                           validation_func: Callable,
                           error_message: str):
        """添加验证规则"""
        if data_type not in self.validation_rules:
            self.validation_rules[data_type] = {}
        
        self.validation_rules[data_type][field_name] = {
            'func': validation_func,
            'message': error_message
        }
    
    def validate_trade_event(self, event: TradeEvent) -> Tuple[bool, List[str]]:
        """验证交易事件"""
        errors = []
        
        rules = self.validation_rules.get('trade_event', {})
        
        # 验证必填字段
        if 'event_id' in rules and not event.event_id:
            errors.append(rules['event_id']['message'])
        
        if 'symbol' in rules and not event.symbol:
            errors.append(rules['symbol']['message'])
        
        if 'quantity' in rules and event.quantity <= 0:
            errors.append(rules['quantity']['message'])
        
        # 执行自定义验证
        for field, rule in rules.items():
            try:
                value = getattr(event, field, None)
                if not rule['func'](value):
                    errors.append(rule['message'])
            except Exception as e:
                errors.append(f"验证规则执行失败: {e}")
        
        self.validation_stats['trade_events_validated'] += 1
        if errors:
            self.validation_stats['trade_events_failed'] += 1
        
        return len(errors) == 0, errors
    
    def validate_performance_metric(self, metric: PerformanceMetrics) -> Tuple[bool, List[str]]:
        """验证性能指标"""
        errors = []
        
        rules = self.validation_rules.get('performance_metric', {})
        
        # 验证延迟
        if 'latency_ms' in rules and metric.latency_ms < 0:
            errors.append(rules['latency_ms']['message'])
        
        # 执行自定义验证
        for field, rule in rules.items():
            try:
                value = getattr(metric, field, None)
                if not rule['func'](value):
                    errors.append(rule['message'])
            except Exception as e:
                errors.append(f"验证规则执行失败: {e}")
        
        self.validation_stats['performance_metrics_validated'] += 1
        if errors:
            self.validation_stats['performance_metrics_failed'] += 1
        
        return len(errors) == 0, errors
    
    def get_validation_stats(self) -> Dict[str, int]:
        """获取验证统计"""
        return dict(self.validation_stats)


class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "./logs/trading_logger_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.warning(f"加载配置文件失败: {e}")
        
        # 默认配置
        return {
            'log_level': 'INFO',
            'enable_async': True,
            'async_workers': 4,
            'batch_size': 100,
            'enable_rotation': True,
            'max_log_size_mb': 100,
            'backup_count': 5,
            'retention_days': 30,
            'alert_rules': {},
            'validation_rules': {},
            'custom_settings': {}
        }
    
    def save_config(self):
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存配置文件失败: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        self.config[key] = value
        self.save_config()
    
    def update(self, config_dict: Dict[str, Any]):
        """更新配置"""
        self.config.update(config_dict)
        self.save_config()


class HealthMonitor:
    """健康监控器"""
    
    def __init__(self, logger: TradingLogger):
        self.logger = logger
        self.db_manager = logger.db_manager
        self.health_metrics = {}
        self.last_check = None
    
    def check_system_health(self) -> Dict[str, Any]:
        """检查系统健康状态"""
        health_status = {
            'timestamp': datetime.now(),
            'overall_status': 'HEALTHY',
            'checks': {}
        }
        
        try:
            # 检查数据库连接
            health_status['checks']['database'] = self._check_database_health()
            
            # 检查磁盘空间
            health_status['checks']['disk_space'] = self._check_disk_space()
            
            # 检查内存使用
            health_status['checks']['memory'] = self._check_memory_usage()
            
            # 检查异步处理
            health_status['checks']['async_processing'] = self._check_async_health()
            
            # 检查日志文件
            health_status['checks']['log_files'] = self._check_log_files()
            
            # 确定整体状态
            failed_checks = [check for check in health_status['checks'].values() 
                           if not check.get('healthy', False)]
            
            if failed_checks:
                if any(check.get('severity') == 'CRITICAL' for check in failed_checks):
                    health_status['overall_status'] = 'CRITICAL'
                else:
                    health_status['overall_status'] = 'WARNING'
            
            self.health_metrics = health_status
            self.last_check = datetime.now()
            
            return health_status
            
        except Exception as e:
            self.logger.logger.error(f"系统健康检查失败: {e}")
            health_status['overall_status'] = 'ERROR'
            health_status['error'] = str(e)
            return health_status
    
    def _check_database_health(self) -> Dict[str, Any]:
        """检查数据库健康状态"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                cursor.fetchone()
            
            return {
                'healthy': True,
                'message': '数据库连接正常',
                'severity': 'INFO'
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'数据库连接失败: {e}',
                'severity': 'CRITICAL'
            }
    
    def _check_disk_space(self) -> Dict[str, Any]:
        """检查磁盘空间"""
        try:
            log_dir = Path(self.logger.log_dir)
            if not log_dir.exists():
                return {
                    'healthy': False,
                    'message': '日志目录不存在',
                    'severity': 'ERROR'
                }
            
            stat = shutil.disk_usage(log_dir)
            free_gb = stat.free / (1024**3)
            total_gb = stat.total / (1024**3)
            usage_percent = (stat.used / stat.total) * 100
            
            if usage_percent > 90:
                severity = 'CRITICAL'
                healthy = False
                message = f'磁盘空间严重不足: {usage_percent:.1f}% 已使用'
            elif usage_percent > 80:
                severity = 'WARNING'
                healthy = False
                message = f'磁盘空间不足: {usage_percent:.1f}% 已使用'
            else:
                severity = 'INFO'
                healthy = True
                message = f'磁盘空间充足: {free_gb:.1f}GB 可用'
            
            return {
                'healthy': healthy,
                'message': message,
                'severity': severity,
                'free_space_gb': free_gb,
                'total_space_gb': total_gb,
                'usage_percent': usage_percent
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'磁盘空间检查失败: {e}',
                'severity': 'ERROR'
            }
    
    def _check_memory_usage(self) -> Dict[str, Any]:
        """检查内存使用情况"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            usage_percent = memory.percent
            
            if usage_percent > 90:
                severity = 'CRITICAL'
                healthy = False
                message = f'内存使用率过高: {usage_percent}%'
            elif usage_percent > 80:
                severity = 'WARNING'
                healthy = False
                message = f'内存使用率较高: {usage_percent}%'
            else:
                severity = 'INFO'
                healthy = True
                message = f'内存使用正常: {usage_percent}%'
            
            return {
                'healthy': healthy,
                'message': message,
                'severity': severity,
                'usage_percent': usage_percent,
                'available_gb': memory.available / (1024**3)
            }
        except ImportError:
            return {
                'healthy': True,
                'message': '无法检查内存使用情况（psutil未安装）',
                'severity': 'INFO'
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'内存检查失败: {e}',
                'severity': 'ERROR'
            }
    
    def _check_async_health(self) -> Dict[str, Any]:
        """检查异步处理健康状态"""
        try:
            if not self.logger.async_processor:
                return {
                    'healthy': True,
                    'message': '异步处理未启用',
                    'severity': 'INFO'
                }
            
            processor = self.logger.async_processor
            queue_size = processor.processing_queue.qsize()
            
            if queue_size > 1000:
                severity = 'WARNING'
                healthy = False
                message = f'异步处理队列积压过多: {queue_size} 项'
            else:
                severity = 'INFO'
                healthy = True
                message = f'异步处理正常: {queue_size} 项待处理'
            
            return {
                'healthy': healthy,
                'message': message,
                'severity': severity,
                'queue_size': queue_size,
                'is_processing': processor.is_processing
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'异步处理检查失败: {e}',
                'severity': 'ERROR'
            }
    
    def _check_log_files(self) -> Dict[str, Any]:
        """检查日志文件状态"""
        try:
            log_dir = Path(self.logger.log_dir)
            if not log_dir.exists():
                return {
                    'healthy': False,
                    'message': '日志目录不存在',
                    'severity': 'ERROR'
                }
            
            log_files = list(log_dir.glob("*.log*"))
            total_size = sum(f.stat().st_size for f in log_files if f.is_file())
            
            return {
                'healthy': True,
                'message': f'日志文件正常: {len(log_files)} 个文件',
                'severity': 'INFO',
                'file_count': len(log_files),
                'total_size_mb': total_size / (1024**2)
            }
        except Exception as e:
            return {
                'healthy': False,
                'message': f'日志文件检查失败: {e}',
                'severity': 'ERROR'
            }


# 扩展TradingLogger类
class ExtendedTradingLogger(TradingLogger):
    """扩展的交易日志记录器"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 初始化扩展组件
        self.analytics = AdvancedAnalytics(self)
        self.alert_manager = AlertManager(self)
        self.validator = DataValidator(self)
        self.config_manager = ConfigManager()
        self.health_monitor = HealthMonitor(self)
        
        # 设置默认告警规则
        self._setup_default_alerts()
        
        # 设置默认验证规则
        self._setup_default_validations()
    
    def _setup_default_alerts(self):
        """设置默认告警规则"""
        # 高延迟告警
        self.alert_manager.add_alert_rule(
            "high_latency",
            lambda: self._check_high_latency(),
            "交易延迟过高",
            "WARNING",
            300
        )
        
        # 高错误率告警
        self.alert_manager.add_alert_rule(
            "high_error_rate",
            lambda: self._check_high_error_rate(),
            "错误率过高",
            "WARNING",
            600
        )
        
        # 数据库连接告警
        self.alert_manager.add_alert_rule(
            "database_connection",
            lambda: self._check_database_connection(),
            "数据库连接异常",
            "CRITICAL",
            60
        )
    
    def _setup_default_validations(self):
        """设置默认验证规则"""
        # 交易事件验证
        self.validator.add_validation_rule(
            'trade_event', 'event_id',
            lambda x: x and len(x) > 0,
            '事件ID不能为空'
        )
        
        self.validator.add_validation_rule(
            'trade_event', 'symbol',
            lambda x: x and len(x) > 0,
            '交易标的不能为空'
        )
        
        self.validator.add_validation_rule(
            'trade_event', 'quantity',
            lambda x: x > 0,
            '交易数量必须大于0'
        )
        
        # 性能指标验证
        self.validator.add_validation_rule(
            'performance_metric', 'latency_ms',
            lambda x: x >= 0,
            '延迟时间不能为负数'
        )
    
    def _check_high_latency(self) -> bool:
        """检查高延迟"""
        try:
            perf_summary = self.get_performance_summary(hours=1)
            avg_latency = perf_summary.get('average_latency_ms', 0)
            return avg_latency > 1000  # 1秒
        except:
            return False
    
    def _check_high_error_rate(self) -> bool:
        """检查高错误率"""
        try:
            error_summary = self.get_error_summary(hours=1)
            total_errors = error_summary.get('total_errors', 0)
            return total_errors > 10  # 1小时内超过10个错误
        except:
            return False
    
    def _check_database_connection(self) -> bool:
        """检查数据库连接"""
        try:
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT 1')
                return False  # 连接正常，不触发告警
        except:
            return True  # 连接异常，触发告警
    
    def start_monitoring(self):
        """启动监控"""
        # 启动健康检查
        self._start_health_monitoring()
        
        # 启动告警检查
        self._start_alert_monitoring()
        
        self.logger.info("监控服务已启动")
    
    def _start_health_monitoring(self):
        """启动健康监控"""
        def health_check_worker():
            while True:
                try:
                    self.health_monitor.check_system_health()
                    time.sleep(60)  # 每分钟检查一次
                except Exception as e:
                    self.logger.error(f"健康检查异常: {e}")
                    time.sleep(60)
        
        thread = threading.Thread(target=health_check_worker, daemon=True)
        thread.start()
    
    def _start_alert_monitoring(self):
        """启动告警监控"""
        def alert_check_worker():
            while True:
                try:
                    self.alert_manager.check_alerts()
                    time.sleep(30)  # 每30秒检查一次
                except Exception as e:
                    self.logger.error(f"告警检查异常: {e}")
                    time.sleep(30)
        
        thread = threading.Thread(target=alert_check_worker, daemon=True)
        thread.start()
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            'health_check': self.health_monitor.check_system_health(),
            'recent_alerts': self.alert_manager.get_recent_alerts(hours=24),
            'validation_stats': self.validator.get_validation_stats(),
            'performance_summary': self.get_performance_summary(hours=24),
            'error_summary': self.get_error_summary(hours=24)
        }
    
    def advanced_export(self, 
                       start_date: datetime,
                       end_date: datetime,
                       output_dir: str,
                       include_analysis: bool = True) -> Dict[str, str]:
        """高级导出功能"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        try:
            # 导出原始数据
            raw_data_file = output_path / f"raw_data_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
            if self.export_logs(start_date, end_date, str(raw_data_file)):
                exported_files['raw_data'] = str(raw_data_file)
            
            # 生成分析报告
            if include_analysis:
                analysis_file = output_path / f"analysis_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
                analysis_report = self._generate_analysis_report(start_date, end_date)
                
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis_report, f, ensure_ascii=False, indent=2, default=str)
                
                exported_files['analysis'] = str(analysis_file)
            
            # 生成统计报告
            stats_file = output_path / f"statistics_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.json"
            stats_report = self._generate_statistics_report(start_date, end_date)
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_report, f, ensure_ascii=False, indent=2, default=str)
            
            exported_files['statistics'] = str(stats_file)
            
            self.logger.info(f"高级导出完成: {len(exported_files)} 个文件")
            return exported_files
            
        except Exception as e:
            self.logger.error(f"高级导出失败: {e}")
            return {}
    
    def _generate_analysis_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """生成分析报告"""
        report = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat(),
            'analyses': {}
        }
        
        try:
            # 获取所有交易标的
            with self.db_manager.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT DISTINCT symbol FROM trade_events 
                    WHERE timestamp BETWEEN ? AND ?
                ''', (start_date.isoformat(), end_date.isoformat()))
                
                symbols = [row[0] for row in cursor.fetchall()]
            
            # 为每个标的生成分析
            for symbol in symbols:
                report['analyses'][symbol] = self.analytics.analyze_trade_patterns(symbol)
            
        except Exception as e:
            report['error'] = str(e)
        
        return report
    
    def _generate_statistics_report(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """生成统计报告"""
        report = {
            'period': {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            },
            'generated_at': datetime.now().isoformat(),
            'statistics': {}
        }
        
        try:
            # 生成不同周期的统计
            for period_type in ['daily', 'weekly', 'monthly']:
                stats = self.generate_statistics(period_type, start_date, end_date)
                report['statistics'][period_type] = asdict(stats)
            
        except Exception as e:
            report['error'] = str(e)
        
        return report


def comprehensive_example():
    """综合使用示例"""
    
    # 创建扩展日志记录器
    with ExtendedTradingLogger(
        log_dir="./logs",
        db_path="./logs/extended_trading_logger.db",
        enable_async=True,
        async_workers=8,
        log_level=LogLevel.INFO
    ) as logger:
        
        # 启动监控
        logger.start_monitoring()
        
        # 示例1: 记录大量交易事件
        print("记录交易事件...")
        for i in range(100):
            symbol = f"SYM_{i % 5}"
            logger.log_trade_event(
                event_type=TradeEventType.ORDER_FILLED,
                symbol=symbol,
                order_id=f"ORD_{i:04d}",
                side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                order_type=OrderType.LIMIT,
                quantity=100 + i * 10,
                price=100.0 + i * 0.5,
                filled_quantity=100 + i * 10,
                average_price=100.0 + i * 0.5,
                strategy_id=f"STRAT_{i % 3}",
                exchange="NASDAQ",
                commission=1.0 + i * 0.1,
                latency_ms=10 + i * 0.5
            )
        
        # 示例2: 记录性能指标
        print("记录性能指标...")
        for i in range(50):
            logger.log_performance_metric(
                symbol=f"SYM_{i % 5}",
                strategy_id=f"STRAT_{i % 3}",
                order_id=f"ORD_{i:04d}",
                latency_ms=20 + i * 0.3,
                success=i % 10 != 0,  # 10%失败率
                commission=1.0,
                slippage=0.5 + i * 0.02,
                fill_ratio=0.8 + i * 0.01
            )
        
        # 示例3: 记录错误事件
        print("记录错误事件...")
        for i in range(20):
            if i % 5 == 0:  # 20%概率记录错误
                logger.log_error_event(
                    error_type="NETWORK_ERROR" if i % 2 == 0 else "VALIDATION_ERROR",
                    error_message=f"示例错误 {i}",
                    symbol=f"SYM_{i % 5}",
                    strategy_id=f"STRAT_{i % 3}",
                    order_id=f"ORD_{i:04d}",
                    retry_count=i % 3,
                    resolved=i % 4 == 0
                )
        
        # 示例4: 记录策略事件
        print("记录策略事件...")
        for i in range(30):
            logger.log_strategy_event(
                strategy_id=f"STRAT_{i % 3}",
                event_type="SIGNAL_GENERATED" if i % 3 == 0 else "PARAMETER_UPDATE",
                symbol=f"SYM_{i % 5}",
                parameters={
                    "stop_loss": 0.02 + i * 0.001,
                    "take_profit": 0.05 + i * 0.002,
                    "position_size": 0.1 + i * 0.01
                },
                execution_time_ms=5 + i * 0.2,
                result={"signal_strength": 0.7 + i * 0.01, "confidence": 0.8 + i * 0.005}
            )
        
        # 示例5: 记录市场数据事件
        print("记录市场数据事件...")
        for i in range(200):
            logger.log_market_data_event(
                symbol=f"SYM_{i % 5}",
                data_type="TICK" if i % 3 == 0 else "BAR",
                quality_score=0.8 + i * 0.001,
                latency_ms=5 + i * 0.01,
                source="REALTIME_FEED",
                data_size=1024 + i * 10,
                missing_fields=[] if i % 10 != 0 else ["bid_size"],
                anomalies=[] if i % 20 != 0 else ["UNUSUAL_VOLUME"]
            )
        
        # 等待异步处理完成
        time.sleep(3)
        
        # 示例6: 生成统计汇总
        print("生成统计汇总...")
        daily_stats = logger.generate_statistics("daily")
        weekly_stats = logger.generate_statistics("weekly")
        monthly_stats = logger.generate_statistics("monthly")
        
        print(f"日统计: 总交易{daily_stats.total_trades}笔, 成功率{daily_stats.win_rate:.2%}")
        print(f"周统计: 总交易{weekly_stats.total_trades}笔, 成功率{weekly_stats.win_rate:.2%}")
        print(f"月统计: 总交易{monthly_stats.total_trades}笔, 成功率{monthly_stats.win_rate:.2%}")
        
        # 示例7: 获取系统状态
        print("获取系统状态...")
        system_status = logger.get_system_status()
        print(f"系统健康状态: {system_status['health_check']['overall_status']}")
        print(f"最近告警数量: {len(system_status['recent_alerts'])}")
        
        # 示例8: 高级分析
        print("执行高级分析...")
        for symbol in ["SYM_0", "SYM_1", "SYM_2"]:
            patterns = logger.analytics.analyze_trade_patterns(symbol, days=7)
            if patterns:
                print(f"{symbol} 交易模式分析:")
                print(f"  总交易数: {patterns['total_trades']}")
                print(f"  买卖比例: 买入{patterns['buy_vs_sell']['buy_ratio']:.2%}")
        
        # 示例9: 高级导出
        print("执行高级导出...")
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        exported_files = logger.advanced_export(start_date, end_date, "./logs/export")
        print(f"导出文件: {list(exported_files.keys())}")
        
        # 示例10: 数据验证
        print("执行数据验证...")
        test_event = TradeEvent(
            event_id="TEST_001",
            timestamp=datetime.now(),
            event_type=TradeEventType.ORDER_CREATED,
            symbol="TEST",
            order_id="ORD_TEST",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=100
        )
        
        is_valid, errors = logger.validator.validate_trade_event(test_event)
        print(f"验证结果: {'通过' if is_valid else '失败'}")
        if errors:
            print(f"验证错误: {errors}")


def stress_test():
    """压力测试"""
    import threading
    import queue
    
    print("开始压力测试...")
    
    with ExtendedTradingLogger(
        log_dir="./logs",
        db_path="./logs/stress_test.db",
        enable_async=True,
        async_workers=16
    ) as logger:
        
        logger.start_monitoring()
        
        # 多线程并发测试
        def worker_thread(thread_id, num_operations):
            for i in range(num_operations):
                # 随机选择操作类型
                operation_type = i % 6
                
                if operation_type == 0:
                    # 交易事件
                    logger.log_trade_event(
                        event_type=TradeEventType.ORDER_FILLED,
                        symbol=f"SYM_{i % 10}",
                        order_id=f"ORD_{thread_id:03d}_{i:04d}",
                        side=OrderSide.BUY,
                        order_type=OrderType.LIMIT,
                        quantity=100,
                        price=100.0,
                        filled_quantity=100,
                        average_price=100.0,
                        strategy_id=f"STRAT_{i % 5}"
                    )
                elif operation_type == 1:
                    # 性能指标
                    logger.log_performance_metric(
                        symbol=f"SYM_{i % 10}",
                        strategy_id=f"STRAT_{i % 5}",
                        order_id=f"ORD_{thread_id:03d}_{i:04d}",
                        latency_ms=10 + i * 0.1,
                        success=True
                    )
                elif operation_type == 2:
                    # 错误事件
                    if i % 20 == 0:
                        logger.log_error_event(
                            error_type="TEST_ERROR",
                            error_message=f"压力测试错误 {thread_id}_{i}",
                            symbol=f"SYM_{i % 10}",
                            strategy_id=f"STRAT_{i % 5}"
                        )
                elif operation_type == 3:
                    # 策略事件
                    logger.log_strategy_event(
                        strategy_id=f"STRAT_{i % 5}",
                        event_type="STRESS_TEST",
                        symbol=f"SYM_{i % 10}",
                        parameters={"test_param": i},
                        execution_time_ms=5.0,
                        result={"status": "success"}
                    )
                elif operation_type == 4:
                    # 市场数据
                    logger.log_market_data_event(
                        symbol=f"SYM_{i % 10}",
                        data_type="TICK",
                        quality_score=0.9,
                        latency_ms=5.0,
                        source="STRESS_TEST",
                        data_size=1024
                    )
                else:
                    # 查询操作
                    logger.get_performance_summary(symbol=f"SYM_{i % 10}", hours=1)
        
        # 启动多个工作线程
        num_threads = 8
        operations_per_thread = 500
        threads = []
        
        start_time = time.time()
        
        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=worker_thread,
                args=(thread_id, operations_per_thread)
            )
            threads.append(thread)
            thread.start()
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        total_operations = num_threads * operations_per_thread
        
        print(f"压力测试完成:")
        print(f"  总操作数: {total_operations}")
        print(f"  总耗时: {end_time - start_time:.2f}秒")
        print(f"  平均QPS: {total_operations / (end_time - start_time):.1f}")
        
        # 等待异步处理完成
        time.sleep(5)
        
        # 生成最终统计
        final_stats = logger.generate_statistics("daily")
        print(f"最终统计: 总交易{final_stats.total_trades}笔")


if __name__ == "__main__":
    # 运行使用示例
    print("=== 运行综合使用示例 ===")
    comprehensive_example()
    
    print("\n=== 运行性能测试 ===")
    performance_test()
    
    print("\n=== 运行压力测试 ===")
    stress_test()