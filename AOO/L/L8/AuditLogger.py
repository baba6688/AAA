"""
L8审计日志记录器

这是一个全面的企业级审计日志记录系统，支持多种类型的审计日志记录，
包括用户操作、数据访问、系统配置、交易操作、安全事件和合规性审计。

主要功能：
- 用户操作审计日志（登录、登出、权限变更）
- 数据访问审计日志（数据读取、数据修改、数据删除）
- 系统配置审计日志（配置修改、配置删除、配置备份）
- 交易操作审计日志（交易下单、交易修改、交易取消）
- 安全事件审计日志（攻击检测、异常访问、权限绕过）
- 合规性审计日志（监管要求、审计报告、风险评估）
- 异步审计日志处理
- 完整的错误处理和日志记录
- 详细的文档字符串和使用示例

"""

import asyncio
import json
import logging
import os
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Set
from concurrent.futures import ThreadPoolExecutor
import hashlib
import gzip
import pickle
from queue import Queue, Empty
import weakref


# =============================================================================
# 枚举定义
# =============================================================================

class AuditLevel(Enum):
    """审计日志级别枚举"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"


class AuditCategory(Enum):
    """审计类别枚举"""
    USER_OPERATION = "USER_OPERATION"
    DATA_ACCESS = "DATA_ACCESS"
    SYSTEM_CONFIG = "SYSTEM_CONFIG"
    TRANSACTION = "TRANSACTION"
    SECURITY_EVENT = "SECURITY_EVENT"
    COMPLIANCE = "COMPLIANCE"


class UserOperation(Enum):
    """用户操作类型枚举"""
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    LOGIN_FAILED = "LOGIN_FAILED"
    PERMISSION_CHANGE = "PERMISSION_CHANGE"
    USER_CREATE = "USER_CREATE"
    USER_UPDATE = "USER_UPDATE"
    USER_DELETE = "USER_DELETE"
    PASSWORD_CHANGE = "PASSWORD_CHANGE"
    ACCOUNT_LOCK = "ACCOUNT_LOCK"
    ACCOUNT_UNLOCK = "ACCOUNT_UNLOCK"


class DataOperation(Enum):
    """数据操作类型枚举"""
    READ = "READ"
    WRITE = "WRITE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    CREATE = "CREATE"
    EXPORT = "EXPORT"
    IMPORT = "IMPORT"
    BACKUP = "BACKUP"
    RESTORE = "RESTORE"


class ConfigOperation(Enum):
    """配置操作类型枚举"""
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    BACKUP = "BACKUP"
    RESTORE = "RESTORE"
    EXPORT = "EXPORT"
    IMPORT = "IMPORT"


class TransactionOperation(Enum):
    """交易操作类型枚举"""
    ORDER_CREATE = "ORDER_CREATE"
    ORDER_UPDATE = "ORDER_UPDATE"
    ORDER_CANCEL = "ORDER_CANCEL"
    ORDER_EXECUTE = "ORDER_EXECUTE"
    TRADE_SETTLEMENT = "TRADE_SETTLEMENT"
    POSITION_CHANGE = "POSITION_CHANGE"
    MARGIN_CHANGE = "MARGIN_CHANGE"


class SecurityEvent(Enum):
    """安全事件类型枚举"""
    ATTACK_DETECTED = "ATTACK_DETECTED"
    UNAUTHORIZED_ACCESS = "UNAUTHORIZED_ACCESS"
    PRIVILEGE_ESCALATION = "PRIVILEGE_ESCALATION"
    DATA_BREACH = "DATA_BREACH"
    MALWARE_DETECTED = "MALWARE_DETECTED"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"
    POLICY_VIOLATION = "POLICY_VIOLATION"


class ComplianceType(Enum):
    """合规性类型枚举"""
    REGULATORY_REQUIREMENT = "REGULATORY_REQUIREMENT"
    AUDIT_REPORT = "AUDIT_REPORT"
    RISK_ASSESSMENT = "RISK_ASSESSMENT"
    COMPLIANCE_CHECK = "COMPLIANCE_CHECK"
    POLICY_UPDATE = "POLICY_UPDATE"


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class AuditContext:
    """审计上下文信息"""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    tenant_id: Optional[str] = None
    application: Optional[str] = None
    module: Optional[str] = None
    function: Optional[str] = None


@dataclass
class AuditRecord:
    """审计记录数据结构"""
    id: str
    timestamp: datetime
    level: AuditLevel
    category: AuditCategory
    operation: Union[UserOperation, DataOperation, ConfigOperation, 
                     TransactionOperation, SecurityEvent, ComplianceType]
    message: str
    context: AuditContext
    data: Dict[str, Any]
    source: str
    checksum: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理，计算校验和"""
        if self.checksum is None:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """计算记录校验和"""
        record_dict = asdict(self)
        record_dict.pop('checksum', None)  # 移除校验和字段
        content = json.dumps(record_dict, sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)
    
    def to_json(self) -> str:
        """转换为JSON格式"""
        return json.dumps(self.to_dict(), default=str)


@dataclass
class AuditFilter:
    """审计日志过滤器"""
    levels: Optional[Set[AuditLevel]] = None
    categories: Optional[Set[AuditCategory]] = None
    operations: Optional[Set[Any]] = None
    user_ids: Optional[Set[str]] = None
    ip_addresses: Optional[Set[str]] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    search_text: Optional[str] = None
    
    def matches(self, record: AuditRecord) -> bool:
        """检查记录是否匹配过滤条件"""
        if self.levels and record.level not in self.levels:
            return False
        
        if self.categories and record.category not in self.categories:
            return False
        
        if self.operations and record.operation not in self.operations:
            return False
        
        if self.user_ids and record.context.user_id not in self.user_ids:
            return False
        
        if self.ip_addresses and record.context.ip_address not in self.ip_addresses:
            return False
        
        if self.start_time and record.timestamp < self.start_time:
            return False
        
        if self.end_time and record.timestamp > self.end_time:
            return False
        
        if self.search_text and self.search_text.lower() not in record.message.lower():
            return False
        
        return True


# =============================================================================
# 存储接口和实现
# =============================================================================

class AuditStorage(ABC):
    """审计存储抽象基类"""
    
    @abstractmethod
    async def store(self, record: AuditRecord) -> bool:
        """存储审计记录"""
        pass
    
    @abstractmethod
    async def query(self, filter_obj: AuditFilter) -> List[AuditRecord]:
        """查询审计记录"""
        pass
    
    @abstractmethod
    async def delete(self, record_ids: List[str]) -> bool:
        """删除审计记录"""
        pass
    
    @abstractmethod
    async def cleanup(self, before_date: datetime) -> int:
        """清理过期记录"""
        pass


class SQLiteAuditStorage(AuditStorage):
    """SQLite审计存储实现"""
    
    def __init__(self, db_path: str, max_size_mb: int = 100):
        self.db_path = db_path
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.Lock()
        self._init_database()
    
    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_records (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    level TEXT NOT NULL,
                    category TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    message TEXT NOT NULL,
                    context TEXT NOT NULL,
                    data TEXT NOT NULL,
                    source TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp 
                ON audit_records(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_level 
                ON audit_records(level)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_category 
                ON audit_records(category)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_id 
                ON audit_records(context)
            """)
            
            conn.commit()
    
    async def store(self, record: AuditRecord) -> bool:
        """存储审计记录"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT INTO audit_records 
                        (id, timestamp, level, category, operation, message, 
                         context, data, source, checksum)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record.id,
                        record.timestamp.isoformat(),
                        record.level.value,
                        record.category.value,
                        record.operation.value,
                        record.message,
                        json.dumps(asdict(record.context)),
                        json.dumps(record.data),
                        record.source,
                        record.checksum
                    ))
                    conn.commit()
                
                # 检查数据库大小，必要时进行压缩
                await self._check_and_compact()
                return True
                
        except Exception as e:
            logging.error(f"Failed to store audit record: {e}")
            return False
    
    async def _check_and_compact(self):
        """检查数据库大小并进行压缩"""
        try:
            if os.path.exists(self.db_path):
                size = os.path.getsize(self.db_path)
                if size > self.max_size_bytes:
                    # 压缩数据库
                    with sqlite3.connect(self.db_path) as conn:
                        conn.execute("VACUUM")
                    logging.info("Database compacted due to size limit")
        except Exception as e:
            logging.error(f"Failed to compact database: {e}")
    
    async def query(self, filter_obj: AuditFilter) -> List[AuditRecord]:
        """查询审计记录"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    
                    # 构建查询条件
                    conditions = []
                    params = []
                    
                    if filter_obj.levels:
                        placeholders = ','.join(['?' for _ in filter_obj.levels])
                        conditions.append(f"level IN ({placeholders})")
                        params.extend([level.value for level in filter_obj.levels])
                    
                    if filter_obj.categories:
                        placeholders = ','.join(['?' for _ in filter_obj.categories])
                        conditions.append(f"category IN ({placeholders})")
                        params.extend([category.value for category in filter_obj.categories])
                    
                    if filter_obj.start_time:
                        conditions.append("timestamp >= ?")
                        params.append(filter_obj.start_time.isoformat())
                    
                    if filter_obj.end_time:
                        conditions.append("timestamp <= ?")
                        params.append(filter_obj.end_time.isoformat())
                    
                    if filter_obj.user_ids:
                        # 由于context是JSON格式，需要使用LIKE查询
                        for user_id in filter_obj.user_ids:
                            conditions.append("context LIKE ?")
                            params.append(f'%"user_id": "{user_id}"%')
                    
                    where_clause = " AND ".join(conditions) if conditions else "1=1"
                    
                    query = f"""
                        SELECT * FROM audit_records 
                        WHERE {where_clause}
                        ORDER BY timestamp DESC
                        LIMIT 1000
                    """
                    
                    cursor = conn.execute(query, params)
                    records = []
                    
                    for row in cursor:
                        context_data = json.loads(row['context'])
                        data = json.loads(row['data'])
                        
                        record = AuditRecord(
                            id=row['id'],
                            timestamp=datetime.fromisoformat(row['timestamp']),
                            level=AuditLevel(row['level']),
                            category=AuditCategory(row['category']),
                            operation=row['operation'],  # 这里需要根据category来确定具体类型
                            message=row['message'],
                            context=AuditContext(**context_data),
                            data=data,
                            source=row['source'],
                            checksum=row['checksum']
                        )
                        
                        # 应用文本搜索过滤
                        if not filter_obj.search_text or filter_obj.search_text.lower() in record.message.lower():
                            records.append(record)
                    
                    return records
                    
        except Exception as e:
            logging.error(f"Failed to query audit records: {e}")
            return []
    
    async def delete(self, record_ids: List[str]) -> bool:
        """删除审计记录"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    placeholders = ','.join(['?' for _ in record_ids])
                    conn.execute(f"""
                        DELETE FROM audit_records 
                        WHERE id IN ({placeholders})
                    """, record_ids)
                    conn.commit()
                    return True
                    
        except Exception as e:
            logging.error(f"Failed to delete audit records: {e}")
            return False
    
    async def cleanup(self, before_date: datetime) -> int:
        """清理过期记录"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute("""
                        DELETE FROM audit_records 
                        WHERE timestamp < ?
                    """, (before_date.isoformat(),))
                    conn.commit()
                    return cursor.rowcount
                    
        except Exception as e:
            logging.error(f"Failed to cleanup audit records: {e}")
            return 0


class FileAuditStorage(AuditStorage):
    """文件审计存储实现"""
    
    def __init__(self, log_dir: str, max_files: int = 100):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_files = max_files
        self._lock = threading.Lock()
    
    async def store(self, record: AuditRecord) -> bool:
        """存储审计记录到文件"""
        try:
            with self._lock:
                # 按日期创建日志文件
                date_str = record.timestamp.strftime("%Y-%m-%d")
                log_file = self.log_dir / f"audit_{date_str}.log"
                
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write(record.to_json() + '\n')
                
                # 清理旧文件
                await self._cleanup_old_files()
                return True
                
        except Exception as e:
            logging.error(f"Failed to store audit record to file: {e}")
            return False
    
    async def _cleanup_old_files(self):
        """清理旧日志文件"""
        try:
            log_files = list(self.log_dir.glob("audit_*.log"))
            if len(log_files) > self.max_files:
                # 按修改时间排序，删除最旧的文件
                log_files.sort(key=lambda x: x.stat().st_mtime)
                for old_file in log_files[:-self.max_files]:
                    old_file.unlink()
        except Exception as e:
            logging.error(f"Failed to cleanup old log files: {e}")
    
    async def query(self, filter_obj: AuditFilter) -> List[AuditRecord]:
        """从文件查询审计记录"""
        try:
            records = []
            
            # 搜索所有日志文件
            for log_file in self.log_dir.glob("audit_*.log"):
                with open(log_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            record_data = json.loads(line.strip())
                            record = AuditRecord(**record_data)
                            
                            # 转换时间戳
                            if isinstance(record.timestamp, str):
                                record.timestamp = datetime.fromisoformat(record.timestamp)
                            
                            # 转换枚举值
                            record.level = AuditLevel(record.level)
                            record.category = AuditCategory(record.category)
                            
                            if filter_obj.matches(record):
                                records.append(record)
                                
                        except Exception:
                            continue
            
            return records
            
        except Exception as e:
            logging.error(f"Failed to query audit records from files: {e}")
            return []
    
    async def delete(self, record_ids: List[str]) -> bool:
        """从文件删除审计记录"""
        try:
            # 文件存储不支持删除操作
            logging.warning("File storage does not support delete operation")
            return False
        except Exception as e:
            logging.error(f"Failed to delete audit records: {e}")
            return False
    
    async def cleanup(self, before_date: datetime) -> int:
        """清理过期文件"""
        try:
            count = 0
            for log_file in self.log_dir.glob("audit_*.log"):
                try:
                    file_date = datetime.strptime(log_file.stem.split('_')[1], "%Y-%m-%d")
                    if file_date < before_date:
                        log_file.unlink()
                        count += 1
                except Exception:
                    continue
            return count
        except Exception as e:
            logging.error(f"Failed to cleanup old files: {e}")
            return 0


# =============================================================================
# 审计处理器
# =============================================================================

class AuditHandler(ABC):
    """审计处理器抽象基类"""
    
    @abstractmethod
    async def handle(self, record: AuditRecord) -> bool:
        """处理审计记录"""
        pass


class ConsoleAuditHandler(AuditHandler):
    """控制台审计处理器"""
    
    def __init__(self, levels: Optional[Set[AuditLevel]] = None):
        self.levels = levels or {AuditLevel.INFO, AuditLevel.WARNING, 
                                AuditLevel.ERROR, AuditLevel.CRITICAL}
    
    async def handle(self, record: AuditRecord) -> bool:
        """在控制台输出审计记录"""
        try:
            if record.level in self.levels:
                print(f"[{record.timestamp}] [{record.level.value}] "
                      f"[{record.category.value}] {record.message}")
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to handle audit record in console: {e}")
            return False


class FileAuditHandler(AuditHandler):
    """文件审计处理器"""
    
    def __init__(self, log_file: str, levels: Optional[Set[AuditLevel]] = None):
        self.log_file = log_file
        self.levels = levels or {AuditLevel.INFO, AuditLevel.WARNING, 
                                AuditLevel.ERROR, AuditLevel.CRITICAL}
        
        # 配置日志记录器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AuditLogger')
    
    async def handle(self, record: AuditRecord) -> bool:
        """将审计记录写入文件"""
        try:
            if record.level in self.levels:
                log_message = f"[{record.category.value}] {record.message}"
                self.logger.info(log_message)
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to handle audit record in file: {e}")
            return False


class EmailAuditHandler(AuditHandler):
    """邮件审计处理器"""
    
    def __init__(self, smtp_config: Dict[str, Any], 
                 recipients: List[str],
                 levels: Optional[Set[AuditLevel]] = None):
        self.smtp_config = smtp_config
        self.recipients = recipients
        self.levels = levels or {AuditLevel.CRITICAL, AuditLevel.SECURITY}
    
    async def handle(self, record: AuditRecord) -> bool:
        """通过邮件发送审计记录"""
        try:
            if record.level in self.levels:
                # 这里实现邮件发送逻辑
                # 由于复杂性，这里只是模拟
                logging.info(f"Would send email for critical audit: {record.message}")
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to handle audit record via email: {e}")
            return False


class WebhookAuditHandler(AuditHandler):
    """Webhook审计处理器"""
    
    def __init__(self, webhook_url: str, 
                 levels: Optional[Set[AuditLevel]] = None,
                 timeout: int = 30):
        self.webhook_url = webhook_url
        self.levels = levels or {AuditLevel.CRITICAL, AuditLevel.SECURITY}
        self.timeout = timeout
    
    async def handle(self, record: AuditRecord) -> bool:
        """通过Webhook发送审计记录"""
        try:
            if record.level in self.levels:
                # 这里实现Webhook发送逻辑
                # 由于复杂性，这里只是模拟
                logging.info(f"Would send webhook for critical audit: {record.message}")
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to handle audit record via webhook: {e}")
            return False


# =============================================================================
# 异步审计队列
# =============================================================================

class AsyncAuditQueue:
    """异步审计队列"""
    
    def __init__(self, max_size: int = 10000, batch_size: int = 100):
        self.queue = Queue(maxsize=max_size)
        self.batch_size = batch_size
        self.storage: Optional[AuditStorage] = None
        self.handlers: List[AuditHandler] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    def set_storage(self, storage: AuditStorage):
        """设置存储后端"""
        self.storage = storage
    
    def add_handler(self, handler: AuditHandler):
        """添加处理器"""
        self.handlers.append(handler)
    
    async def start(self):
        """启动队列处理器"""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._process_queue())
        logging.info("Audit queue started")
    
    async def stop(self):
        """停止队列处理器"""
        if not self.running:
            return
        
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        
        # 处理队列中剩余的记录
        await self._process_remaining()
        
        self.executor.shutdown(wait=True)
        logging.info("Audit queue stopped")
    
    async def put(self, record: AuditRecord):
        """添加记录到队列"""
        try:
            self.queue.put_nowait(record)
        except:
            # 队列满时，丢弃最旧的记录
            try:
                self.queue.get_nowait()
                self.queue.put_nowait(record)
            except:
                logging.warning("Audit queue is full, dropping record")
    
    async def _process_queue(self):
        """处理队列中的记录"""
        batch = []
        
        while self.running:
            try:
                # 批量获取记录
                record = self.queue.get(timeout=1.0)
                batch.append(record)
                
                # 当批次达到指定大小或超时后处理
                if len(batch) >= self.batch_size:
                    await self._process_batch(batch)
                    batch.clear()
                
            except Empty:
                # 超时后处理剩余记录
                if batch:
                    await self._process_batch(batch)
                    batch.clear()
            except Exception as e:
                logging.error(f"Error processing audit queue: {e}")
    
    async def _process_batch(self, batch: List[AuditRecord]):
        """处理一批记录"""
        if not batch:
            return
        
        try:
            # 并行处理存储和处理器
            tasks = []
            
            # 存储到后端
            if self.storage:
                for record in batch:
                    task = asyncio.create_task(self.storage.store(record))
                    tasks.append(task)
            
            # 处理处理器
            for handler in self.handlers:
                for record in batch:
                    task = asyncio.create_task(handler.handle(record))
                    tasks.append(task)
            
            # 等待所有任务完成
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logging.error(f"Error processing audit batch: {e}")
    
    async def _process_remaining(self):
        """处理队列中剩余的记录"""
        batch = []
        
        while not self.queue.empty():
            try:
                record = self.queue.get_nowait()
                batch.append(record)
            except Empty:
                break
        
        if batch:
            await self._process_batch(batch)


# =============================================================================
# 审计日志记录器主类
# =============================================================================

class AuditLogger:
    """
    L8审计日志记录器主类
    
    这是一个全面的企业级审计日志记录系统，支持多种类型的审计日志记录，
    包括用户操作、数据访问、系统配置、交易操作、安全事件和合规性审计。
    
    主要特性：
    - 异步处理高性能
    - 多种存储后端支持
    - 灵活的过滤和查询
    - 实时监控和告警
    - 数据完整性保证
    - 完整的错误处理
    
    使用示例：
        ```python
        # 创建审计记录器
        logger = AuditLogger()
        
        # 配置存储后端
        storage = SQLiteAuditStorage("audit.db")
        logger.set_storage(storage)
        
        # 添加处理器
        console_handler = ConsoleAuditHandler()
        logger.add_handler(console_handler)
        
        # 记录用户操作
        await logger.log_user_operation(
            operation=UserOperation.LOGIN,
            message="User logged in successfully",
            context=AuditContext(user_id="user123", ip_address="192.168.1.1")
        )
        
        # 记录安全事件
        await logger.log_security_event(
            event=SecurityEvent.ATTACK_DETECTED,
            message="Suspicious login attempt detected",
            level=AuditLevel.CRITICAL,
            context=AuditContext(user_id="user123", ip_address="192.168.1.100")
        )
        ```
    """
    
    def __init__(self, 
                 storage: Optional[AuditStorage] = None,
                 queue_size: int = 10000,
                 batch_size: int = 100):
        """
        初始化审计日志记录器
        
        Args:
            storage: 审计存储后端，如果为None则使用默认的文件存储
            queue_size: 异步队列最大大小
            batch_size: 批处理大小
        """
        self.storage = storage or FileAuditStorage("logs/audit")
        self.queue = AsyncAuditQueue(queue_size, batch_size)
        self.queue.set_storage(self.storage)
        
        # 默认处理器
        self.console_handler = ConsoleAuditHandler()
        self.queue.add_handler(self.console_handler)
        
        # 统计信息
        self.stats = {
            'total_records': 0,
            'records_by_category': defaultdict(int),
            'records_by_level': defaultdict(int),
            'errors': 0,
            'start_time': datetime.now(timezone.utc)
        }
        
        # 监控回调
        self.monitoring_callbacks: List[Callable[[AuditRecord], None]] = []
        
        # 启动队列
        asyncio.create_task(self.queue.start())
    
    def add_handler(self, handler: AuditHandler):
        """添加审计处理器"""
        self.queue.add_handler(handler)
    
    def set_storage(self, storage: AuditStorage):
        """设置存储后端"""
        self.storage = storage
        self.queue.set_storage(storage)
    
    def add_monitoring_callback(self, callback: Callable[[AuditRecord], None]):
        """添加监控回调函数"""
        self.monitoring_callbacks.append(callback)
    
    def _create_record(self,
                      category: AuditCategory,
                      operation: Any,
                      message: str,
                      level: AuditLevel,
                      context: AuditContext,
                      data: Optional[Dict[str, Any]] = None) -> AuditRecord:
        """创建审计记录"""
        record = AuditRecord(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            level=level,
            category=category,
            operation=operation,
            message=message,
            context=context,
            data=data or {},
            source="L8AuditLogger"
        )
        
        # 更新统计信息
        self.stats['total_records'] += 1
        self.stats['records_by_category'][category.value] += 1
        self.stats['records_by_level'][level.value] += 1
        
        return record
    
    async def _log(self, record: AuditRecord):
        """内部日志记录方法"""
        try:
            await self.queue.put(record)
            
            # 调用监控回调
            for callback in self.monitoring_callbacks:
                try:
                    callback(record)
                except Exception as e:
                    logging.error(f"Error in monitoring callback: {e}")
            
        except Exception as e:
            self.stats['errors'] += 1
            logging.error(f"Failed to log audit record: {e}")
    
    # =============================================================================
    # 用户操作审计日志
    # =============================================================================
    
    async def log_user_operation(self,
                                operation: UserOperation,
                                message: str,
                                context: AuditContext,
                                level: AuditLevel = AuditLevel.INFO,
                                data: Optional[Dict[str, Any]] = None):
        """
        记录用户操作审计日志
        
        Args:
            operation: 用户操作类型
            message: 操作消息
            context: 审计上下文
            level: 日志级别
            data: 附加数据
        """
        record = self._create_record(
            category=AuditCategory.USER_OPERATION,
            operation=operation,
            message=message,
            level=level,
            context=context,
            data=data
        )
        
        await self._log(record)
    
    async def log_user_login(self,
                            user_id: str,
                            success: bool,
                            context: AuditContext,
                            data: Optional[Dict[str, Any]] = None):
        """
        记录用户登录审计日志
        
        Args:
            user_id: 用户ID
            success: 登录是否成功
            context: 审计上下文
            data: 附加数据
        """
        operation = UserOperation.LOGIN if success else UserOperation.LOGIN_FAILED
        level = AuditLevel.INFO if success else AuditLevel.WARNING
        message = f"User {user_id} {'logged in successfully' if success else 'login failed'}"
        
        if data is None:
            data = {}
        data['success'] = success
        
        await self.log_user_operation(
            operation=operation,
            message=message,
            context=context,
            level=level,
            data=data
        )
    
    async def log_user_logout(self,
                             user_id: str,
                             context: AuditContext,
                             data: Optional[Dict[str, Any]] = None):
        """
        记录用户登出审计日志
        
        Args:
            user_id: 用户ID
            context: 审计上下文
            data: 附加数据
        """
        message = f"User {user_id} logged out"
        
        await self.log_user_operation(
            operation=UserOperation.LOGOUT,
            message=message,
            context=context,
            data=data
        )
    
    async def log_permission_change(self,
                                   user_id: str,
                                   old_permissions: List[str],
                                   new_permissions: List[str],
                                   context: AuditContext,
                                   data: Optional[Dict[str, Any]] = None):
        """
        记录权限变更审计日志
        
        Args:
            user_id: 用户ID
            old_permissions: 原权限列表
            new_permissions: 新权限列表
            context: 审计上下文
            data: 附加数据
        """
        message = f"Permissions changed for user {user_id}"
        
        if data is None:
            data = {}
        data['old_permissions'] = old_permissions
        data['new_permissions'] = new_permissions
        
        await self.log_user_operation(
            operation=UserOperation.PERMISSION_CHANGE,
            message=message,
            context=context,
            level=AuditLevel.WARNING,
            data=data
        )
    
    # =============================================================================
    # 数据访问审计日志
    # =============================================================================
    
    async def log_data_access(self,
                             operation: DataOperation,
                             resource: str,
                             context: AuditContext,
                             level: AuditLevel = AuditLevel.INFO,
                             data: Optional[Dict[str, Any]] = None):
        """
        记录数据访问审计日志
        
        Args:
            operation: 数据操作类型
            resource: 资源标识
            context: 审计上下文
            level: 日志级别
            data: 附加数据
        """
        message = f"Data {operation.value.lower()} operation on resource {resource}"
        
        record = self._create_record(
            category=AuditCategory.DATA_ACCESS,
            operation=operation,
            message=message,
            level=level,
            context=context,
            data=data
        )
        
        await self._log(record)
    
    async def log_data_read(self,
                           resource: str,
                           context: AuditContext,
                           record_count: Optional[int] = None,
                           data: Optional[Dict[str, Any]] = None):
        """
        记录数据读取审计日志
        
        Args:
            resource: 资源标识
            context: 审计上下文
            record_count: 读取的记录数
            data: 附加数据
        """
        if data is None:
            data = {}
        if record_count is not None:
            data['record_count'] = record_count
        
        await self.log_data_access(
            operation=DataOperation.READ,
            resource=resource,
            context=context,
            data=data
        )
    
    async def log_data_write(self,
                            resource: str,
                            context: AuditContext,
                            record_count: Optional[int] = None,
                            data: Optional[Dict[str, Any]] = None):
        """
        记录数据写入审计日志
        
        Args:
            resource: 资源标识
            context: 审计上下文
            record_count: 写入的记录数
            data: 附加数据
        """
        if data is None:
            data = {}
        if record_count is not None:
            data['record_count'] = record_count
        
        await self.log_data_access(
            operation=DataOperation.WRITE,
            resource=resource,
            context=context,
            level=AuditLevel.WARNING,
            data=data
        )
    
    async def log_data_delete(self,
                             resource: str,
                             context: AuditContext,
                             record_count: Optional[int] = None,
                             data: Optional[Dict[str, Any]] = None):
        """
        记录数据删除审计日志
        
        Args:
            resource: 资源标识
            context: 审计上下文
            record_count: 删除的记录数
            data: 附加数据
        """
        if data is None:
            data = {}
        if record_count is not None:
            data['record_count'] = record_count
        
        await self.log_data_access(
            operation=DataOperation.DELETE,
            resource=resource,
            context=context,
            level=AuditLevel.WARNING,
            data=data
        )
    
    # =============================================================================
    # 系统配置审计日志
    # =============================================================================
    
    async def log_system_config(self,
                               operation: ConfigOperation,
                               config_key: str,
                               context: AuditContext,
                               level: AuditLevel = AuditLevel.INFO,
                               data: Optional[Dict[str, Any]] = None):
        """
        记录系统配置审计日志
        
        Args:
            operation: 配置操作类型
            config_key: 配置键
            context: 审计上下文
            level: 日志级别
            data: 附加数据
        """
        message = f"System config {operation.value.lower()}: {config_key}"
        
        record = self._create_record(
            category=AuditCategory.SYSTEM_CONFIG,
            operation=operation,
            message=message,
            level=level,
            context=context,
            data=data
        )
        
        await self._log(record)
    
    async def log_config_change(self,
                               config_key: str,
                               old_value: Any,
                               new_value: Any,
                               context: AuditContext,
                               data: Optional[Dict[str, Any]] = None):
        """
        记录配置变更审计日志
        
        Args:
            config_key: 配置键
            old_value: 原值
            new_value: 新值
            context: 审计上下文
            data: 附加数据
        """
        if data is None:
            data = {}
        data['old_value'] = old_value
        data['new_value'] = new_value
        
        await self.log_system_config(
            operation=ConfigOperation.UPDATE,
            config_key=config_key,
            context=context,
            level=AuditLevel.WARNING,
            data=data
        )
    
    async def log_config_backup(self,
                               config_keys: List[str],
                               backup_path: str,
                               context: AuditContext,
                               data: Optional[Dict[str, Any]] = None):
        """
        记录配置备份审计日志
        
        Args:
            config_keys: 配置键列表
            backup_path: 备份路径
            context: 审计上下文
            data: 附加数据
        """
        if data is None:
            data = {}
        data['config_keys'] = config_keys
        data['backup_path'] = backup_path
        
        message = f"System config backup created: {len(config_keys)} keys"
        
        await self.log_system_config(
            operation=ConfigOperation.BACKUP,
            config_key="multiple",
            context=context,
            data=data
        )
    
    # =============================================================================
    # 交易操作审计日志
    # =============================================================================
    
    async def log_transaction(self,
                             operation: TransactionOperation,
                             transaction_id: str,
                             context: AuditContext,
                             level: AuditLevel = AuditLevel.INFO,
                             data: Optional[Dict[str, Any]] = None):
        """
        记录交易操作审计日志
        
        Args:
            operation: 交易操作类型
            transaction_id: 交易ID
            context: 审计上下文
            level: 日志级别
            data: 附加数据
        """
        message = f"Transaction {operation.value.lower()}: {transaction_id}"
        
        record = self._create_record(
            category=AuditCategory.TRANSACTION,
            operation=operation,
            message=message,
            level=level,
            context=context,
            data=data
        )
        
        await self._log(record)
    
    async def log_order_create(self,
                              order_id: str,
                              symbol: str,
                              quantity: float,
                              price: Optional[float],
                              context: AuditContext,
                              data: Optional[Dict[str, Any]] = None):
        """
        记录订单创建审计日志
        
        Args:
            order_id: 订单ID
            symbol: 交易标的
            quantity: 数量
            price: 价格
            context: 审计上下文
            data: 附加数据
        """
        if data is None:
            data = {}
        data.update({
            'order_id': order_id,
            'symbol': symbol,
            'quantity': quantity,
            'price': price
        })
        
        await self.log_transaction(
            operation=TransactionOperation.ORDER_CREATE,
            transaction_id=order_id,
            context=context,
            data=data
        )
    
    async def log_order_cancel(self,
                              order_id: str,
                              reason: str,
                              context: AuditContext,
                              data: Optional[Dict[str, Any]] = None):
        """
        记录订单取消审计日志
        
        Args:
            order_id: 订单ID
            reason: 取消原因
            context: 审计上下文
            data: 附加数据
        """
        if data is None:
            data = {}
        data['order_id'] = order_id
        data['cancel_reason'] = reason
        
        await self.log_transaction(
            operation=TransactionOperation.ORDER_CANCEL,
            transaction_id=order_id,
            context=context,
            level=AuditLevel.WARNING,
            data=data
        )
    
    async def log_trade_settlement(self,
                                  trade_id: str,
                                  symbol: str,
                                  quantity: float,
                                  price: float,
                                  context: AuditContext,
                                  data: Optional[Dict[str, Any]] = None):
        """
        记录交易结算审计日志
        
        Args:
            trade_id: 交易ID
            symbol: 交易标的
            quantity: 数量
            price: 价格
            context: 审计上下文
            data: 附加数据
        """
        if data is None:
            data = {}
        data.update({
            'trade_id': trade_id,
            'symbol': symbol,
            'quantity': quantity,
            'price': price
        })
        
        await self.log_transaction(
            operation=TransactionOperation.TRADE_SETTLEMENT,
            transaction_id=trade_id,
            context=context,
            level=AuditLevel.INFO,
            data=data
        )
    
    # =============================================================================
    # 安全事件审计日志
    # =============================================================================
    
    async def log_security_event(self,
                                event: SecurityEvent,
                                message: str,
                                context: AuditContext,
                                level: AuditLevel = AuditLevel.WARNING,
                                data: Optional[Dict[str, Any]] = None):
        """
        记录安全事件审计日志
        
        Args:
            event: 安全事件类型
            message: 事件消息
            context: 审计上下文
            level: 日志级别
            data: 附加数据
        """
        record = self._create_record(
            category=AuditCategory.SECURITY_EVENT,
            operation=event,
            message=message,
            level=level,
            context=context,
            data=data
        )
        
        await self._log(record)
    
    async def log_attack_detected(self,
                                 attack_type: str,
                                 source_ip: str,
                                 context: AuditContext,
                                 data: Optional[Dict[str, Any]] = None):
        """
        记录攻击检测审计日志
        
        Args:
            attack_type: 攻击类型
            source_ip: 攻击源IP
            context: 审计上下文
            data: 附加数据
        """
        if data is None:
            data = {}
        data['attack_type'] = attack_type
        data['source_ip'] = source_ip
        
        message = f"Attack detected: {attack_type} from {source_ip}"
        
        await self.log_security_event(
            event=SecurityEvent.ATTACK_DETECTED,
            message=message,
            context=context,
            level=AuditLevel.CRITICAL,
            data=data
        )
    
    async def log_unauthorized_access(self,
                                     resource: str,
                                     context: AuditContext,
                                     data: Optional[Dict[str, Any]] = None):
        """
        记录未授权访问审计日志
        
        Args:
            resource: 访问的资源
            context: 审计上下文
            data: 附加数据
        """
        if data is None:
            data = {}
        data['resource'] = resource
        
        message = f"Unauthorized access attempt to {resource}"
        
        await self.log_security_event(
            event=SecurityEvent.UNAUTHORIZED_ACCESS,
            message=message,
            context=context,
            level=AuditLevel.WARNING,
            data=data
        )
    
    async def log_privilege_escalation(self,
                                      user_id: str,
                                      from_role: str,
                                      to_role: str,
                                      context: AuditContext,
                                      data: Optional[Dict[str, Any]] = None):
        """
        记录权限提升审计日志
        
        Args:
            user_id: 用户ID
            from_role: 原角色
            to_role: 目标角色
            context: 审计上下文
            data: 附加数据
        """
        if data is None:
            data = {}
        data.update({
            'user_id': user_id,
            'from_role': from_role,
            'to_role': to_role
        })
        
        message = f"Privilege escalation: {user_id} from {from_role} to {to_role}"
        
        await self.log_security_event(
            event=SecurityEvent.PRIVILEGE_ESCALATION,
            message=message,
            context=context,
            level=AuditLevel.CRITICAL,
            data=data
        )
    
    # =============================================================================
    # 合规性审计日志
    # =============================================================================
    
    async def log_compliance_event(self,
                                  compliance_type: ComplianceType,
                                  message: str,
                                  context: AuditContext,
                                  level: AuditLevel = AuditLevel.INFO,
                                  data: Optional[Dict[str, Any]] = None):
        """
        记录合规性审计日志
        
        Args:
            compliance_type: 合规性类型
            message: 合规消息
            context: 审计上下文
            level: 日志级别
            data: 附加数据
        """
        record = self._create_record(
            category=AuditCategory.COMPLIANCE,
            operation=compliance_type,
            message=message,
            level=level,
            context=context,
            data=data
        )
        
        await self._log(record)
    
    async def log_regulatory_requirement(self,
                                        regulation: str,
                                        requirement: str,
                                        context: AuditContext,
                                        data: Optional[Dict[str, Any]] = None):
        """
        记录监管要求审计日志
        
        Args:
            regulation: 监管规定
            requirement: 具体要求
            context: 审计上下文
            data: 附加数据
        """
        if data is None:
            data = {}
        data['regulation'] = regulation
        data['requirement'] = requirement
        
        message = f"Regulatory requirement: {regulation} - {requirement}"
        
        await self.log_compliance_event(
            compliance_type=ComplianceType.REGULATORY_REQUIREMENT,
            message=message,
            context=context,
            level=AuditLevel.INFO,
            data=data
        )
    
    async def log_audit_report(self,
                              report_id: str,
                              report_type: str,
                              context: AuditContext,
                              data: Optional[Dict[str, Any]] = None):
        """
        记录审计报告审计日志
        
        Args:
            report_id: 报告ID
            report_type: 报告类型
            context: 审计上下文
            data: 附加数据
        """
        if data is None:
            data = {}
        data['report_id'] = report_id
        data['report_type'] = report_type
        
        message = f"Audit report generated: {report_type} ({report_id})"
        
        await self.log_compliance_event(
            compliance_type=ComplianceType.AUDIT_REPORT,
            message=message,
            context=context,
            level=AuditLevel.INFO,
            data=data
        )
    
    async def log_risk_assessment(self,
                                 assessment_id: str,
                                 risk_level: str,
                                 context: AuditContext,
                                 data: Optional[Dict[str, Any]] = None):
        """
        记录风险评估审计日志
        
        Args:
            assessment_id: 评估ID
            risk_level: 风险级别
            context: 审计上下文
            data: 附加数据
        """
        if data is None:
            data = {}
        data['assessment_id'] = assessment_id
        data['risk_level'] = risk_level
        
        message = f"Risk assessment completed: {risk_level} risk ({assessment_id})"
        
        await self.log_compliance_event(
            compliance_type=ComplianceType.RISK_ASSESSMENT,
            message=message,
            context=context,
            level=AuditLevel.WARNING if risk_level in ['HIGH', 'CRITICAL'] else AuditLevel.INFO,
            data=data
        )
    
    # =============================================================================
    # 查询和统计方法
    # =============================================================================
    
    async def query_audit_logs(self, 
                              filter_obj: AuditFilter) -> List[AuditRecord]:
        """
        查询审计日志
        
        Args:
            filter_obj: 审计过滤器
            
        Returns:
            匹配的审计记录列表
        """
        if self.storage:
            return await self.storage.query(filter_obj)
        return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        获取审计统计信息
        
        Returns:
            统计信息字典
        """
        runtime = datetime.now(timezone.utc) - self.stats['start_time']
        
        return {
            'total_records': self.stats['total_records'],
            'records_by_category': dict(self.stats['records_by_category']),
            'records_by_level': dict(self.stats['records_by_level']),
            'errors': self.stats['errors'],
            'runtime_seconds': runtime.total_seconds(),
            'records_per_second': self.stats['total_records'] / max(runtime.total_seconds(), 1)
        }
    
    async def cleanup_old_records(self, days: int = 90) -> int:
        """
        清理过期记录
        
        Args:
            days: 保留天数
            
        Returns:
            清理的记录数
        """
        if self.storage:
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
            return await self.storage.cleanup(cutoff_date)
        return 0
    
    # =============================================================================
    # 上下文管理器
    # =============================================================================
    
    @contextmanager
    def audit_context(self, context: AuditContext):
        """审计上下文管理器"""
        # 这里可以将上下文存储在线程本地变量中
        # 供后续的审计日志记录使用
        try:
            yield context
        finally:
            # 清理工作
            pass
    
    # =============================================================================
    # 装饰器
    # =============================================================================
    
    def audit_function(self, 
                      category: AuditCategory,
                      operation: Any,
                      level: AuditLevel = AuditLevel.INFO):
        """
        审计函数装饰器
        
        Args:
            category: 审计类别
            operation: 操作类型
            level: 日志级别
        """
        def decorator(func):
            async def async_wrapper(*args, **kwargs):
                # 创建审计上下文（这里需要根据实际情况获取）
                context = AuditContext(
                    function=func.__name__,
                    module=func.__module__
                )
                
                start_time = time.time()
                success = True
                error_message = None
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    error_message = str(e)
                    level = AuditLevel.ERROR
                    raise
                finally:
                    # 记录审计日志
                    message = f"Function {func.__name__} executed"
                    if not success:
                        message += f" with error: {error_message}"
                    
                    data = {
                        'function': func.__name__,
                        'args': str(args),
                        'kwargs': str(kwargs),
                        'duration': time.time() - start_time,
                        'success': success
                    }
                    
                    await self._log(self._create_record(
                        category=category,
                        operation=operation,
                        message=message,
                        level=level,
                        context=context,
                        data=data
                    ))
            
            def sync_wrapper(*args, **kwargs):
                # 同步版本的装饰器
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(async_wrapper(*args, **kwargs))
                finally:
                    loop.close()
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    # =============================================================================
    # 批量操作方法
    # =============================================================================
    
    async def batch_log_user_operations(self, 
                                       operations: List[Dict[str, Any]]):
        """
        批量记录用户操作审计日志
        
        Args:
            operations: 操作列表，每个元素包含operation, message, context等
        """
        tasks = []
        for op_data in operations:
            task = asyncio.create_task(
                self.log_user_operation(**op_data)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def batch_log_data_access(self, 
                                   accesses: List[Dict[str, Any]]):
        """
        批量记录数据访问审计日志
        
        Args:
            accesses: 访问列表，每个元素包含operation, resource, context等
        """
        tasks = []
        for access_data in accesses:
            task = asyncio.create_task(
                self.log_data_access(**access_data)
            )
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    # =============================================================================
    # 实时监控和告警
    # =============================================================================
    
    def add_real_time_monitor(self, 
                             condition: Callable[[AuditRecord], bool],
                             alert_handler: Callable[[List[AuditRecord]], None],
                             time_window: int = 60):
        """
        添加实时监控规则
        
        Args:
            condition: 触发条件
            alert_handler: 告警处理函数
            time_window: 时间窗口（秒）
        """
        monitor = RealTimeMonitor(condition, alert_handler, time_window)
        self.add_monitoring_callback(monitor.check_record)
    
    # =============================================================================
    # 数据导出和导入
    # =============================================================================
    
    async def export_audit_logs(self, 
                               filter_obj: AuditFilter,
                               export_path: str,
                               format: str = 'json') -> bool:
        """
        导出审计日志
        
        Args:
            filter_obj: 过滤器
            export_path: 导出路径
            format: 导出格式 ('json', 'csv', 'xml')
            
        Returns:
            是否成功
        """
        try:
            records = await self.query_audit_logs(filter_obj)
            
            if format.lower() == 'json':
                with open(export_path, 'w', encoding='utf-8') as f:
                    json.dump([record.to_dict() for record in records], 
                             f, indent=2, default=str)
            elif format.lower() == 'csv':
                import csv
                with open(export_path, 'w', newline='', encoding='utf-8') as f:
                    if records:
                        writer = csv.DictWriter(f, fieldnames=records[0].to_dict().keys())
                        writer.writeheader()
                        for record in records:
                            writer.writerow(record.to_dict())
            else:
                raise ValueError(f"Unsupported export format: {format}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to export audit logs: {e}")
            return False
    
    # =============================================================================
    # 性能优化方法
    # =============================================================================
    
    async def optimize_storage(self):
        """优化存储性能"""
        if hasattr(self.storage, 'optimize'):
            await self.storage.optimize()
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        if hasattr(self.storage, 'get_stats'):
            return await self.storage.get_stats()
        return {}
    
    # =============================================================================
    # 清理和关闭方法
    # =============================================================================
    
    async def close(self):
        """关闭审计记录器"""
        await self.queue.stop()
        logging.info("Audit logger closed")


# =============================================================================
# 实时监控器
# =============================================================================

class RealTimeMonitor:
    """实时监控器"""
    
    def __init__(self, 
                 condition: Callable[[AuditRecord], bool],
                 alert_handler: Callable[[List[AuditRecord]], None],
                 time_window: int = 60):
        self.condition = condition
        self.alert_handler = alert_handler
        self.time_window = time_window
        self.records = deque()
        self.lock = threading.Lock()
    
    def check_record(self, record: AuditRecord):
        """检查记录是否触发监控"""
        with self.lock:
            current_time = time.time()
            
            # 清理过期记录
            while self.records and current_time - self.records[0][0] > self.time_window:
                self.records.popleft()
            
            # 检查是否满足条件
            if self.condition(record):
                self.records.append((current_time, record))
                
                # 检查是否需要告警
                if len(self.records) >= 5:  # 5次触发告警
                    alert_records = [r[1] for r in self.records]
                    try:
                        self.alert_handler(alert_records)
                    except Exception as e:
                        logging.error(f"Error in alert handler: {e}")
                    
                    # 清空记录
                    self.records.clear()


# =============================================================================
# 工厂函数和工具函数
# =============================================================================

def create_audit_logger(storage_type: str = 'sqlite', **kwargs) -> AuditLogger:
    """
    创建审计记录器工厂函数
    
    Args:
        storage_type: 存储类型 ('sqlite', 'file')
        **kwargs: 存储配置参数
        
    Returns:
        配置好的审计记录器实例
    """
    if storage_type.lower() == 'sqlite':
        storage = SQLiteAuditStorage(
            db_path=kwargs.get('db_path', 'audit.db'),
            max_size_mb=kwargs.get('max_size_mb', 100)
        )
    elif storage_type.lower() == 'file':
        storage = FileAuditStorage(
            log_dir=kwargs.get('log_dir', 'logs/audit'),
            max_files=kwargs.get('max_files', 100)
        )
    else:
        raise ValueError(f"Unsupported storage type: {storage_type}")
    
    return AuditLogger(storage=storage)


def create_audit_context(user_id: Optional[str] = None,
                        session_id: Optional[str] = None,
                        ip_address: Optional[str] = None,
                        **kwargs) -> AuditContext:
    """
    创建审计上下文工厂函数
    
    Args:
        user_id: 用户ID
        session_id: 会话ID
        ip_address: IP地址
        **kwargs: 其他上下文参数
        
    Returns:
        审计上下文实例
    """
    return AuditContext(
        user_id=user_id,
        session_id=session_id,
        ip_address=ip_address,
        **kwargs
    )


def create_audit_filter(**kwargs) -> AuditFilter:
    """
    创建审计过滤器工厂函数
    
    Args:
        **kwargs: 过滤条件
        
    Returns:
        审计过滤器实例
    """
    return AuditFilter(**kwargs)


# =============================================================================
# 使用示例
# =============================================================================

async def example_usage():
    """使用示例"""
    
    # 创建审计记录器
    logger = AuditLogger()
    
    # 配置存储后端
    storage = SQLiteAuditStorage("example_audit.db")
    logger.set_storage(storage)
    
    # 添加处理器
    console_handler = ConsoleAuditHandler()
    file_handler = FileAuditHandler("audit.log")
    logger.add_handler(console_handler)
    logger.add_handler(file_handler)
    
    # 创建审计上下文
    context = create_audit_context(
        user_id="user123",
        session_id="session456",
        ip_address="192.168.1.100"
    )
    
    # 记录用户操作
    await logger.log_user_login(
        user_id="user123",
        success=True,
        context=context
    )
    
    # 记录数据访问
    await logger.log_data_read(
        resource="user_database",
        context=context,
        record_count=10
    )
    
    # 记录安全事件
    await logger.log_attack_detected(
        attack_type="SQL注入",
        source_ip="192.168.1.200",
        context=context
    )
    
    # 记录交易操作
    await logger.log_order_create(
        order_id="order789",
        symbol="AAPL",
        quantity=100,
        price=150.0,
        context=context
    )
    
    # 记录合规性事件
    await logger.log_risk_assessment(
        assessment_id="risk001",
        risk_level="MEDIUM",
        context=context
    )
    
    # 查询审计日志
    filter_obj = create_audit_filter(
        levels={AuditLevel.WARNING, AuditLevel.ERROR, AuditLevel.CRITICAL},
        user_ids={"user123"}
    )
    
    records = await logger.query_audit_logs(filter_obj)
    print(f"Found {len(records)} matching records")
    
    # 获取统计信息
    stats = await logger.get_statistics()
    print(f"Statistics: {stats}")
    
    # 导出审计日志
    await logger.export_audit_logs(
        filter_obj=filter_obj,
        export_path="exported_audit.json",
        format="json"
    )
    
    # 关闭审计记录器
    await logger.close()


# =============================================================================
# 主程序入口
# =============================================================================

# =============================================================================
# 高级审计分析器
# =============================================================================

class AuditAnalyzer:
    """审计日志分析器"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.patterns = {}
        self.anomaly_detectors = []
    
    def add_pattern(self, name: str, pattern: Callable[[AuditRecord], bool]):
        """添加模式识别规则"""
        self.patterns[name] = pattern
    
    def add_anomaly_detector(self, detector: Callable[[List[AuditRecord]], List[AuditRecord]]):
        """添加异常检测器"""
        self.anomaly_detectors.append(detector)
    
    async def analyze(self, time_window: int = 3600) -> Dict[str, Any]:
        """分析审计日志"""
        # 获取时间窗口内的记录
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(seconds=time_window)
        
        filter_obj = AuditFilter(start_time=start_time, end_time=end_time)
        records = await self.audit_logger.query_audit_logs(filter_obj)
        
        analysis_result = {
            'time_window': time_window,
            'total_records': len(records),
            'pattern_matches': {},
            'anomalies': [],
            'statistics': self._calculate_statistics(records)
        }
        
        # 模式匹配
        for pattern_name, pattern_func in self.patterns.items():
            matches = [record for record in records if pattern_func(record)]
            analysis_result['pattern_matches'][pattern_name] = len(matches)
        
        # 异常检测
        for detector in self.anomaly_detectors:
            anomalies = detector(records)
            analysis_result['anomalies'].extend(anomalies)
        
        return analysis_result
    
    def _calculate_statistics(self, records: List[AuditRecord]) -> Dict[str, Any]:
        """计算统计信息"""
        if not records:
            return {}
        
        # 按小时统计
        hourly_stats = defaultdict(int)
        user_stats = defaultdict(int)
        ip_stats = defaultdict(int)
        
        for record in records:
            hour = record.timestamp.hour
            hourly_stats[hour] += 1
            
            if record.context.user_id:
                user_stats[record.context.user_id] += 1
            
            if record.context.ip_address:
                ip_stats[record.context.ip_address] += 1
        
        return {
            'hourly_distribution': dict(hourly_stats),
            'top_users': dict(sorted(user_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
            'top_ips': dict(sorted(ip_stats.items(), key=lambda x: x[1], reverse=True)[:10]),
            'level_distribution': {level.value: sum(1 for r in records if r.level == level) 
                                  for level in AuditLevel},
            'category_distribution': {cat.value: sum(1 for r in records if r.category == cat) 
                                    for cat in AuditCategory}
        }


# =============================================================================
# 审计报告生成器
# =============================================================================

class AuditReportGenerator:
    """审计报告生成器"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
    
    async def generate_daily_report(self, date: datetime.date) -> str:
        """生成日报"""
        start_time = datetime.combine(date, datetime.min.time(), tzinfo=timezone.utc)
        end_time = datetime.combine(date, datetime.max.time(), tzinfo=timezone.utc)
        
        filter_obj = AuditFilter(start_time=start_time, end_time=end_time)
        records = await self.audit_logger.query_audit_logs(filter_obj)
        
        report = f"""
# 审计日报 - {date.strftime('%Y-%m-%d')}

## 概述
- 总记录数: {len(records)}
- 时间范围: {start_time.strftime('%H:%M:%S')} - {end_time.strftime('%H:%M:%S')}

## 分类统计
"""
        
        # 按类别统计
        category_stats = defaultdict(int)
        level_stats = defaultdict(int)
        
        for record in records:
            category_stats[record.category.value] += 1
            level_stats[record.level.value] += 1
        
        for category, count in category_stats.items():
            report += f"- {category}: {count}\n"
        
        report += "\n## 级别统计\n"
        for level, count in level_stats.items():
            report += f"- {level}: {count}\n"
        
        # 安全事件
        security_records = [r for r in records if r.category == AuditCategory.SECURITY_EVENT]
        if security_records:
            report += f"\n## 安全事件 ({len(security_records)} 条)\n"
            for record in security_records[:10]:  # 只显示前10条
                report += f"- {record.timestamp.strftime('%H:%M:%S')} {record.message}\n"
        
        # 用户活动
        user_records = [r for r in records if r.category == AuditCategory.USER_OPERATION]
        if user_records:
            report += f"\n## 用户活动 ({len(user_records)} 条)\n"
            for record in user_records[:10]:
                report += f"- {record.timestamp.strftime('%H:%M:%S')} {record.context.user_id} {record.message}\n"
        
        return report
    
    async def generate_compliance_report(self, 
                                       start_date: datetime.date,
                                       end_date: datetime.date) -> str:
        """生成合规报告"""
        start_time = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
        end_time = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc)
        
        filter_obj = AuditFilter(start_time=start_time, end_time=end_time)
        records = await self.audit_logger.query_audit_logs(filter_obj)
        
        # 筛选合规性记录
        compliance_records = [r for r in records if r.category == AuditCategory.COMPLIANCE]
        security_records = [r for r in records if r.category == AuditCategory.SECURITY_EVENT]
        
        report = f"""
# 合规性审计报告

## 报告期间
- 开始日期: {start_date.strftime('%Y-%m-%d')}
- 结束日期: {end_date.strftime('%Y-%m-%d')}
- 总天数: {(end_date - start_date).days + 1}

## 合规性活动
- 合规记录总数: {len(compliance_records)}
- 安全事件总数: {len(security_records)}

## 合规性详情
"""
        
        for record in compliance_records:
            report += f"- {record.timestamp.strftime('%Y-%m-%d %H:%M:%S')} {record.message}\n"
        
        report += "\n## 安全事件详情\n"
        for record in security_records:
            report += f"- {record.timestamp.strftime('%Y-%m-%d %H:%M:%S')} [{record.level.value}] {record.message}\n"
        
        # 风险评估
        high_risk_events = [r for r in security_records if r.level in [AuditLevel.CRITICAL, AuditLevel.ERROR]]
        report += f"\n## 风险评估\n"
        report += f"- 高风险事件: {len(high_risk_events)}\n"
        
        if high_risk_events:
            report += "\n### 高风险事件详情\n"
            for record in high_risk_events:
                report += f"- {record.timestamp.strftime('%Y-%m-%d %H:%M:%S')} {record.message}\n"
        
        return report


# =============================================================================
# 审计数据压缩器
# =============================================================================

class AuditCompressor:
    """审计数据压缩器"""
    
    @staticmethod
    def compress_records(records: List[AuditRecord]) -> bytes:
        """压缩审计记录"""
        data = [record.to_dict() for record in records]
        serialized = json.dumps(data, default=str).encode('utf-8')
        return gzip.compress(serialized)
    
    @staticmethod
    def decompress_records(compressed_data: bytes) -> List[AuditRecord]:
        """解压缩审计记录"""
        decompressed = gzip.decompress(compressed_data).decode('utf-8')
        data = json.loads(decompressed)
        
        records = []
        for record_data in data:
            # 转换时间戳
            if isinstance(record_data['timestamp'], str):
                record_data['timestamp'] = datetime.fromisoformat(record_data['timestamp'])
            
            # 转换枚举值
            record_data['level'] = AuditLevel(record_data['level'])
            record_data['category'] = AuditCategory(record_data['category'])
            
            # 转换上下文
            context_data = record_data.pop('context')
            record_data['context'] = AuditContext(**context_data)
            
            record = AuditRecord(**record_data)
            records.append(record)
        
        return records


# =============================================================================
# 审计数据验证器
# =============================================================================

class AuditValidator:
    """审计数据验证器"""
    
    @staticmethod
    def validate_record(record: AuditRecord) -> bool:
        """验证审计记录"""
        try:
            # 检查必需字段
            if not record.id or not record.timestamp or not record.message:
                return False
            
            # 检查枚举值
            if not isinstance(record.level, AuditLevel):
                return False
            
            if not isinstance(record.category, AuditCategory):
                return False
            
            # 检查校验和
            expected_checksum = record._calculate_checksum()
            if record.checksum != expected_checksum:
                return False
            
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def validate_integrity(records: List[AuditRecord]) -> Dict[str, Any]:
        """验证数据完整性"""
        results = {
            'total_records': len(records),
            'valid_records': 0,
            'invalid_records': 0,
            'corrupted_records': [],
            'checksum_mismatches': []
        }
        
        for i, record in enumerate(records):
            if not AuditValidator.validate_record(record):
                results['invalid_records'] += 1
                
                # 检查校验和
                expected_checksum = record._calculate_checksum()
                if record.checksum != expected_checksum:
                    results['checksum_mismatches'].append({
                        'index': i,
                        'record_id': record.id,
                        'expected': expected_checksum,
                        'actual': record.checksum
                    })
            else:
                results['valid_records'] += 1
        
        return results


# =============================================================================
# 审计缓存管理器
# =============================================================================

class AuditCacheManager:
    """审计缓存管理器"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, AuditRecord] = {}
        self.access_times: Dict[str, float] = {}
        self.lock = threading.Lock()
    
    def get(self, record_id: str) -> Optional[AuditRecord]:
        """获取缓存的记录"""
        with self.lock:
            if record_id in self.cache:
                self.access_times[record_id] = time.time()
                return self.cache[record_id]
            return None
    
    def put(self, record: AuditRecord):
        """缓存记录"""
        with self.lock:
            # 如果缓存已满，删除最少使用的记录
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[record.id] = record
            self.access_times[record.id] = time.time()
    
    def _evict_lru(self):
        """删除最近最少使用的记录"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda x: self.access_times[x])
        del self.cache[lru_key]
        del self.access_times[lru_key]
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            return {
                'size': len(self.cache),
                'max_size': self.max_size,
                'utilization': len(self.cache) / self.max_size if self.max_size > 0 else 0
            }


# =============================================================================
# 审计规则引擎
# =============================================================================

class AuditRuleEngine:
    """审计规则引擎"""
    
    def __init__(self):
        self.rules: List[Callable[[AuditRecord], bool]] = []
        self.actions: Dict[str, Callable[[AuditRecord], None]] = {}
    
    def add_rule(self, rule_name: str, rule_func: Callable[[AuditRecord], bool]):
        """添加规则"""
        self.rules.append(rule_func)
        self.actions[rule_name] = lambda record: None  # 默认动作
    
    def add_rule_with_action(self, 
                           rule_name: str,
                           rule_func: Callable[[AuditRecord], bool],
                           action_func: Callable[[AuditRecord], None]):
        """添加带动作的规则"""
        self.rules.append(rule_func)
        self.actions[rule_name] = action_func
    
    async def evaluate(self, record: AuditRecord) -> List[str]:
        """评估规则"""
        triggered_rules = []
        
        for i, rule in enumerate(self.rules):
            try:
                if rule(record):
                    rule_name = f"rule_{i}"
                    triggered_rules.append(rule_name)
                    
                    # 执行动作
                    if rule_name in self.actions:
                        self.actions[rule_name](record)
                        
            except Exception as e:
                logging.error(f"Error evaluating rule {i}: {e}")
        
        return triggered_rules


# =============================================================================
# 预定义规则
# =============================================================================

class AuditRules:
    """预定义审计规则"""
    
    @staticmethod
    def multiple_failed_logins(threshold: int = 5) -> Callable[[AuditRecord], bool]:
        """多次登录失败规则"""
        return lambda record: (
            record.category == AuditCategory.USER_OPERATION and
            record.operation == UserOperation.LOGIN_FAILED
        )
    
    @staticmethod
    def suspicious_ip_access() -> Callable[[AuditRecord], bool]:
        """可疑IP访问规则"""
        suspicious_ips = {'192.168.1.100', '10.0.0.1', '172.16.0.1'}
        return lambda record: (
            record.category == AuditCategory.SECURITY_EVENT and
            record.context.ip_address in suspicious_ips
        )
    
    @staticmethod
    def high_value_transaction(threshold: float = 100000) -> Callable[[AuditRecord], bool]:
        """高价值交易规则"""
        return lambda record: (
            record.category == AuditCategory.TRANSACTION and
            record.data.get('price', 0) * record.data.get('quantity', 0) > threshold
        )
    
    @staticmethod
    def privileged_operation() -> Callable[[AuditRecord], bool]:
        """特权操作规则"""
        return lambda record: (
            record.category in [AuditCategory.SYSTEM_CONFIG, AuditCategory.USER_OPERATION] and
            record.operation in [UserOperation.PERMISSION_CHANGE, ConfigOperation.DELETE]
        )


# =============================================================================
# 审计数据归档器
# =============================================================================

class AuditArchiver:
    """审计数据归档器"""
    
    def __init__(self, audit_logger: AuditLogger, archive_dir: str):
        self.audit_logger = audit_logger
        self.archive_dir = Path(archive_dir)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
    
    async def archive_old_records(self, days: int = 90) -> Dict[str, Any]:
        """归档旧记录"""
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # 查询要归档的记录
        filter_obj = AuditFilter(end_time=cutoff_date)
        records = await self.audit_logger.query_audit_logs(filter_obj)
        
        if not records:
            return {'archived_count': 0}
        
        # 按日期分组
        date_groups = defaultdict(list)
        for record in records:
            date_key = record.timestamp.strftime('%Y-%m-%d')
            date_groups[date_key].append(record)
        
        archived_count = 0
        
        # 归档每个日期组
        for date_key, date_records in date_groups.items():
            archive_file = self.archive_dir / f"audit_archive_{date_key}.gz"
            
            try:
                compressed_data = AuditCompressor.compress_records(date_records)
                
                with open(archive_file, 'wb') as f:
                    f.write(compressed_data)
                
                archived_count += len(date_records)
                
            except Exception as e:
                logging.error(f"Failed to archive records for {date_key}: {e}")
        
        # 删除已归档的记录
        if archived_count > 0:
            record_ids = [record.id for record in records]
            await self.audit_logger.storage.delete(record_ids)
        
        return {
            'archived_count': archived_count,
            'archive_files': len(date_groups)
        }
    
    async def restore_archive(self, archive_file: str) -> List[AuditRecord]:
        """恢复归档文件"""
        try:
            with open(archive_file, 'rb') as f:
                compressed_data = f.read()
            
            records = AuditCompressor.decompress_records(compressed_data)
            
            # 重新存储记录
            for record in records:
                await self.audit_logger.storage.store(record)
            
            return records
            
        except Exception as e:
            logging.error(f"Failed to restore archive {archive_file}: {e}")
            return []


# =============================================================================
# 审计仪表板
# =============================================================================

class AuditDashboard:
    """审计仪表板"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.real_time_data = defaultdict(int)
        self.alert_thresholds = {
            'failed_logins': 10,
            'security_events': 5,
            'high_risk_transactions': 3
        }
    
    async def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表板数据"""
        # 获取最近1小时的数据
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=1)
        
        filter_obj = AuditFilter(start_time=start_time, end_time=end_time)
        records = await self.audit_logger.query_audit_logs(filter_obj)
        
        # 实时统计
        stats = {
            'last_hour_total': len(records),
            'by_category': defaultdict(int),
            'by_level': defaultdict(int),
            'by_user': defaultdict(int),
            'alerts': []
        }
        
        for record in records:
            stats['by_category'][record.category.value] += 1
            stats['by_level'][record.level.value] += 1
            
            if record.context.user_id:
                stats['by_user'][record.context.user_id] += 1
        
        # 检查告警条件
        failed_logins = sum(1 for r in records if r.operation == UserOperation.LOGIN_FAILED)
        security_events = sum(1 for r in records if r.category == AuditCategory.SECURITY_EVENT)
        
        if failed_logins > self.alert_thresholds['failed_logins']:
            stats['alerts'].append({
                'type': 'failed_logins',
                'message': f'Failed logins exceeded threshold: {failed_logins}',
                'severity': 'warning'
            })
        
        if security_events > self.alert_thresholds['security_events']:
            stats['alerts'].append({
                'type': 'security_events',
                'message': f'Security events exceeded threshold: {security_events}',
                'severity': 'critical'
            })
        
        return stats


# =============================================================================
# 高级审计分析工具
# =============================================================================

class AdvancedAuditAnalyzer:
    """高级审计分析工具"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.ml_models = {}
    
    async def detect_anomalies(self, time_window: int = 3600) -> List[Dict[str, Any]]:
        """异常检测"""
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(seconds=time_window)
        
        filter_obj = AuditFilter(start_time=start_time, end_time=end_time)
        records = await self.audit_logger.query_audit_logs(filter_obj)
        
        anomalies = []
        
        # 基于统计的异常检测
        user_activity = defaultdict(int)
        ip_activity = defaultdict(int)
        
        for record in records:
            if record.context.user_id:
                user_activity[record.context.user_id] += 1
            if record.context.ip_address:
                ip_activity[record.context.ip_address] += 1
        
        # 检测异常用户活动
        if user_activity:
            avg_activity = sum(user_activity.values()) / len(user_activity)
            for user_id, activity in user_activity.items():
                if activity > avg_activity * 3:  # 超过平均值3倍
                    anomalies.append({
                        'type': 'user_anomaly',
                        'user_id': user_id,
                        'activity_count': activity,
                        'severity': 'medium'
                    })
        
        # 检测异常IP活动
        if ip_activity:
            avg_ip_activity = sum(ip_activity.values()) / len(ip_activity)
            for ip_address, activity in ip_activity.items():
                if activity > avg_ip_activity * 5:  # 超过平均值5倍
                    anomalies.append({
                        'type': 'ip_anomaly',
                        'ip_address': ip_address,
                        'activity_count': activity,
                        'severity': 'high'
                    })
        
        return anomalies
    
    async def generate_user_behavior_profile(self, user_id: str) -> Dict[str, Any]:
        """生成用户行为画像"""
        # 获取用户最近30天的数据
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=30)
        
        filter_obj = AuditFilter(
            start_time=start_time,
            end_time=end_time,
            user_ids={user_id}
        )
        
        records = await self.audit_logger.query_audit_logs(filter_obj)
        
        if not records:
            return {'user_id': user_id, 'profile': 'No data available'}
        
        # 分析用户行为模式
        profile = {
            'user_id': user_id,
            'total_activities': len(records),
            'activity_by_category': defaultdict(int),
            'activity_by_hour': defaultdict(int),
            'activity_by_day': defaultdict(int),
            'common_operations': defaultdict(int),
            'access_patterns': defaultdict(int),
            'risk_score': 0
        }
        
        for record in records:
            profile['activity_by_category'][record.category.value] += 1
            profile['activity_by_hour'][record.timestamp.hour] += 1
            profile['activity_by_day'][record.timestamp.weekday()] += 1
            profile['common_operations'][record.operation.value] += 1
            
            # 计算风险分数
            if record.category == AuditCategory.SECURITY_EVENT:
                profile['risk_score'] += 10
            elif record.level in [AuditLevel.ERROR, AuditLevel.CRITICAL]:
                profile['risk_score'] += 5
            elif record.category == AuditCategory.TRANSACTION:
                profile['risk_score'] += 2
        
        return profile
    
    async def predict_security_risks(self) -> List[Dict[str, Any]]:
        """预测安全风险"""
        # 获取最近7天的数据
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(days=7)
        
        filter_obj = AuditFilter(start_time=start_time, end_time=end_time)
        records = await self.audit_logger.query_audit_logs(filter_obj)
        
        risks = []
        
        # 基于规则的预测
        user_risk_scores = defaultdict(int)
        
        for record in records:
            if record.context.user_id:
                # 累计用户风险分数
                if record.category == AuditCategory.SECURITY_EVENT:
                    user_risk_scores[record.context.user_id] += 20
                elif record.level == AuditLevel.ERROR:
                    user_risk_scores[record.context.user_id] += 10
                elif record.operation == UserOperation.LOGIN_FAILED:
                    user_risk_scores[record.context.user_id] += 5
        
        # 生成风险预测
        for user_id, risk_score in user_risk_scores.items():
            if risk_score > 50:
                risks.append({
                    'user_id': user_id,
                    'risk_level': 'HIGH',
                    'risk_score': risk_score,
                    'prediction': 'High probability of security incident'
                })
            elif risk_score > 20:
                risks.append({
                    'user_id': user_id,
                    'risk_level': 'MEDIUM',
                    'risk_score': risk_score,
                    'prediction': 'Moderate risk of security issues'
                })
        
        return sorted(risks, key=lambda x: x['risk_score'], reverse=True)


# =============================================================================
# 审计集成工具
# =============================================================================

class AuditIntegrationTool:
    """审计集成工具"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.integrations = {}
    
    def register_integration(self, name: str, integration_func: Callable):
        """注册外部集成"""
        self.integrations[name] = integration_func
    
    async def sync_with_external_system(self, system_name: str, data: Dict[str, Any]):
        """与外部系统同步"""
        if system_name in self.integrations:
            try:
                result = await self.integrations[system_name](data)
                return result
            except Exception as e:
                logging.error(f"Integration {system_name} failed: {e}")
                return None
        else:
            logging.warning(f"Integration {system_name} not found")
            return None
    
    async def export_to_siem(self, filter_obj: AuditFilter, siem_config: Dict[str, Any]):
        """导出到SIEM系统"""
        records = await self.audit_logger.query_audit_logs(filter_obj)
        
        # 转换为SIEM格式
        siem_events = []
        for record in records:
            siem_event = {
                'timestamp': record.timestamp.isoformat(),
                'severity': record.level.value,
                'event_type': record.category.value,
                'source': record.source,
                'message': record.message,
                'user_id': record.context.user_id,
                'source_ip': record.context.ip_address,
                'additional_data': record.data
            }
            siem_events.append(siem_event)
        
        # 发送到SIEM系统
        siem_endpoint = siem_config.get('endpoint')
        if siem_endpoint:
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.post(siem_endpoint, json=siem_events) as response:
                        return response.status == 200
            except Exception as e:
                logging.error(f"Failed to export to SIEM: {e}")
                return False
        
        return False


# =============================================================================
# 审计性能监控器
# =============================================================================

class AuditPerformanceMonitor:
    """审计性能监控器"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.metrics = {
            'latency_samples': deque(maxlen=1000),
            'throughput_samples': deque(maxlen=1000),
            'error_count': 0,
            'total_operations': 0
        }
        self.start_time = time.time()
    
    def record_operation_latency(self, latency: float):
        """记录操作延迟"""
        self.metrics['latency_samples'].append(latency)
    
    def record_throughput(self, operations_per_second: float):
        """记录吞吐量"""
        self.metrics['throughput_samples'].append(operations_per_second)
    
    def record_error(self):
        """记录错误"""
        self.metrics['error_count'] += 1
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        latency_samples = list(self.metrics['latency_samples'])
        throughput_samples = list(self.metrics['throughput_samples'])
        
        return {
            'avg_latency': sum(latency_samples) / len(latency_samples) if latency_samples else 0,
            'max_latency': max(latency_samples) if latency_samples else 0,
            'min_latency': min(latency_samples) if latency_samples else 0,
            'avg_throughput': sum(throughput_samples) / len(throughput_samples) if throughput_samples else 0,
            'error_rate': self.metrics['error_count'] / max(self.metrics['total_operations'], 1),
            'uptime': time.time() - self.start_time
        }


# =============================================================================
# 审计配置管理器
# =============================================================================

class AuditConfigManager:
    """审计配置管理器"""
    
    def __init__(self, config_file: str = "audit_config.json"):
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logging.error(f"Failed to load config: {e}")
        
        # 默认配置
        return {
            'storage': {
                'type': 'sqlite',
                'db_path': 'audit.db',
                'max_size_mb': 100
            },
            'queue': {
                'max_size': 10000,
                'batch_size': 100
            },
            'handlers': {
                'console': {'enabled': True},
                'file': {'enabled': True, 'path': 'audit.log'},
                'email': {'enabled': False},
                'webhook': {'enabled': False}
            },
            'retention': {
                'days': 90,
                'archive_enabled': True,
                'archive_path': 'archive'
            },
            'monitoring': {
                'real_time_alerts': True,
                'thresholds': {
                    'failed_logins': 10,
                    'security_events': 5
                }
            }
        }
    
    def save_config(self):
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Failed to save config: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value


# =============================================================================
# 审计工具函数集合
# =============================================================================

class AuditUtils:
    """审计工具函数集合"""
    
    @staticmethod
    def generate_audit_id() -> str:
        """生成审计记录ID"""
        return str(uuid.uuid4())
    
    @staticmethod
    def sanitize_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """清理敏感数据"""
        sensitive_keys = {'password', 'token', 'secret', 'key', 'auth'}
        sanitized = {}
        
        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "***REDACTED***"
            elif isinstance(value, dict):
                sanitized[key] = AuditUtils.sanitize_data(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    @staticmethod
    def mask_ip_address(ip: str) -> str:
        """掩码IP地址"""
        if not ip:
            return ip
        
        parts = ip.split('.')
        if len(parts) == 4:
            return f"{parts[0]}.{parts[1]}.***.***"
        
        return "***.***.***.***"
    
    @staticmethod
    def mask_user_id(user_id: str) -> str:
        """掩码用户ID"""
        if not user_id or len(user_id) <= 4:
            return "***"
        return user_id[:2] + "***" + user_id[-2:]
    
    @staticmethod
    def calculate_record_hash(record: AuditRecord) -> str:
        """计算记录哈希值"""
        content = f"{record.timestamp}{record.category.value}{record.operation.value}{record.message}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    @staticmethod
    def format_audit_message(template: str, **kwargs) -> str:
        """格式化审计消息"""
        try:
            return template.format(**kwargs)
        except KeyError as e:
            logging.warning(f"Missing key in audit message template: {e}")
            return template
    
    @staticmethod
    def validate_ip_address(ip: str) -> bool:
        """验证IP地址格式"""
        try:
            parts = ip.split('.')
            return len(parts) == 4 and all(0 <= int(part) <= 255 for part in parts)
        except (ValueError, AttributeError):
            return False
    
    @staticmethod
    def validate_email(email: str) -> bool:
        """验证邮箱格式"""
        import re
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None


# =============================================================================
# 审计测试工具
# =============================================================================

class AuditTestTool:
    """审计测试工具"""
    
    def __init__(self, audit_logger: AuditLogger):
        self.audit_logger = audit_logger
        self.test_data = []
    
    def generate_test_user_operations(self, count: int = 100) -> List[Dict[str, Any]]:
        """生成测试用户操作数据"""
        operations = []
        users = [f"user_{i}" for i in range(1, 21)]
        ips = [f"192.168.1.{i}" for i in range(1, 51)]
        
        for i in range(count):
            user_id = random.choice(users)
            ip_address = random.choice(ips)
            operation = random.choice(list(UserOperation))
            
            context = create_audit_context(
                user_id=user_id,
                ip_address=ip_address,
                session_id=f"session_{i}"
            )
            
            operations.append({
                'operation': operation,
                'message': f"Test {operation.value} by {user_id}",
                'context': context,
                'level': AuditLevel.INFO if operation != UserOperation.LOGIN_FAILED else AuditLevel.WARNING
            })
        
        return operations
    
    def generate_test_security_events(self, count: int = 50) -> List[Dict[str, Any]]:
        """生成测试安全事件数据"""
        events = []
        ips = [f"192.168.1.{i}" for i in range(1, 101)]
        
        for i in range(count):
            event = random.choice(list(SecurityEvent))
            ip_address = random.choice(ips)
            
            context = create_audit_context(
                user_id=f"user_{random.randint(1, 20)}",
                ip_address=ip_address
            )
            
            events.append({
                'event': event,
                'message': f"Test {event.value} from {ip_address}",
                'context': context,
                'level': AuditLevel.CRITICAL if event == SecurityEvent.ATTACK_DETECTED else AuditLevel.WARNING
            })
        
        return events
    
    async def run_performance_test(self, duration: int = 60) -> Dict[str, Any]:
        """运行性能测试"""
        start_time = time.time()
        end_time = start_time + duration
        record_count = 0
        error_count = 0
        
        while time.time() < end_time:
            try:
                context = create_audit_context(
                    user_id=f"user_{random.randint(1, 10)}",
                    ip_address=f"192.168.1.{random.randint(1, 100)}"
                )
                
                await self.audit_logger.log_user_operation(
                    operation=UserOperation.LOGIN,
                    message="Performance test operation",
                    context=context
                )
                
                record_count += 1
                
            except Exception as e:
                error_count += 1
                logging.error(f"Performance test error: {e}")
            
            await asyncio.sleep(0.001)  # 1ms间隔
        
        actual_duration = time.time() - start_time
        
        return {
            'duration': actual_duration,
            'records_written': record_count,
            'errors': error_count,
            'records_per_second': record_count / actual_duration,
            'error_rate': error_count / max(record_count, 1)
        }
    
    async def validate_audit_integrity(self) -> Dict[str, Any]:
        """验证审计完整性"""
        # 查询所有记录
        filter_obj = AuditFilter()
        records = await self.audit_logger.query_audit_logs(filter_obj)
        
        validator = AuditValidator()
        return validator.validate_integrity(records)


# =============================================================================
# 高级使用示例
# =============================================================================

async def advanced_example():
    """高级使用示例"""
    
    # 创建审计记录器
    logger = AuditLogger()
    
    # 配置存储后端
    storage = SQLiteAuditStorage("example_audit.db")
    logger.set_storage(storage)
    
    # 添加处理器
    console_handler = ConsoleAuditHandler()
    file_handler = FileAuditHandler("audit.log")
    logger.add_handler(console_handler)
    logger.add_handler(file_handler)
    
    # 创建分析器
    analyzer = AuditAnalyzer(logger)
    
    # 添加模式识别规则
    analyzer.add_pattern(
        "multiple_failed_logins",
        AuditRules.multiple_failed_logins(threshold=3)
    )
    
    analyzer.add_pattern(
        "high_value_transactions",
        AuditRules.high_value_transaction(50000)
    )
    
    # 创建实时监控器
    def security_alert_handler(records: List[AuditRecord]):
        print(f"Security Alert: {len(records)} suspicious activities detected")
        for record in records:
            print(f"- {record.timestamp}: {record.message}")
    
    logger.add_real_time_monitor(
        condition=lambda r: r.category == AuditCategory.SECURITY_EVENT,
        alert_handler=security_alert_handler,
        time_window=300  # 5分钟窗口
    )
    
    # 创建性能监控器
    perf_monitor = AuditPerformanceMonitor(logger)
    
    # 生成测试数据
    test_tool = AuditTestTool(logger)
    
    print("Generating test data...")
    user_ops = test_tool.generate_test_user_operations(50)
    security_events = test_tool.generate_test_security_events(20)
    
    # 批量记录测试数据
    await logger.batch_log_user_operations(user_ops)
    
    # 记录安全事件
    for event_data in security_events:
        await logger.log_security_event(**event_data)
    
    print("Running performance test...")
    perf_results = await test_tool.run_performance_test(30)
    print(f"Performance results: {perf_results}")
    
    print("Analyzing audit logs...")
    analysis = await analyzer.analyze(time_window=3600)
    print(f"Analysis results: {analysis}")
    
    print("Generating reports...")
    report_gen = AuditReportGenerator(logger)
    daily_report = await report_gen.generate_daily_report(datetime.now().date())
    print("Daily Report:")
    print(daily_report)
    
    print("Running integrity validation...")
    integrity_results = await test_tool.validate_audit_integrity()
    print(f"Integrity validation: {integrity_results}")
    
    print("Getting performance metrics...")
    metrics = perf_monitor.get_performance_metrics()
    print(f"Performance metrics: {metrics}")
    
    # 关闭审计记录器
    await logger.close()


# =============================================================================
# 完整的企业级使用示例
# =============================================================================

async def enterprise_example():
    """完整的企业级使用示例"""
    
    # 1. 初始化配置
    config_manager = AuditConfigManager("enterprise_audit_config.json")
    
    # 2. 创建存储后端
    if config_manager.get('storage.type') == 'sqlite':
        storage = SQLiteAuditStorage(
            config_manager.get('storage.db_path'),
            config_manager.get('storage.max_size_mb')
        )
    else:
        storage = FileAuditStorage(
            config_manager.get('storage.log_dir'),
            config_manager.get('storage.max_files')
        )
    
    # 3. 创建审计记录器
    logger = AuditLogger(
        storage=storage,
        queue_size=config_manager.get('queue.max_size'),
        batch_size=config_manager.get('queue.batch_size')
    )
    
    # 4. 配置处理器
    if config_manager.get('handlers.console.enabled'):
        logger.add_handler(ConsoleAuditHandler())
    
    if config_manager.get('handlers.file.enabled'):
        logger.add_handler(FileAuditHandler(config_manager.get('handlers.file.path')))
    
    # 5. 创建高级分析工具
    advanced_analyzer = AdvancedAuditAnalyzer(logger)
    dashboard = AuditDashboard(logger)
    archiver = AuditArchiver(logger, config_manager.get('retention.archive_path'))
    integration_tool = AuditIntegrationTool(logger)
    
    # 6. 设置监控和告警
    def critical_security_alert(records: List[AuditRecord]):
        print(f"🚨 CRITICAL SECURITY ALERT 🚨")
        print(f"Detected {len(records)} critical security events:")
        for record in records:
            print(f"  - {record.timestamp}: {record.message}")
    
    logger.add_real_time_monitor(
        condition=lambda r: r.level == AuditLevel.CRITICAL,
        alert_handler=critical_security_alert,
        time_window=60
    )
    
    # 7. 模拟企业业务场景
    print("Simulating enterprise business scenarios...")
    
    # 用户登录场景
    for i in range(100):
        user_id = f"employee_{i % 20}"
        context = create_audit_context(
            user_id=user_id,
            ip_address=f"10.0.{i % 10}.{i % 255}",
            session_id=f"session_{uuid.uuid4()}"
        )
        
        # 正常登录
        await logger.log_user_login(
            user_id=user_id,
            success=True,
            context=context
        )
        
        # 偶尔的数据访问
        if i % 5 == 0:
            await logger.log_data_read(
                resource="employee_database",
                context=context,
                record_count=random.randint(1, 100)
            )
        
        # 偶尔的交易操作
        if i % 10 == 0:
            await logger.log_order_create(
                order_id=f"order_{uuid.uuid4()}",
                symbol=f"STOCK_{i % 10}",
                quantity=random.randint(1, 1000),
                price=random.uniform(10, 1000),
                context=context
            )
    
    # 8. 模拟安全事件
    print("Simulating security events...")
    
    # 正常安全事件
    for i in range(10):
        context = create_audit_context(
            user_id=f"suspect_{i}",
            ip_address=f"192.168.1.{i + 100}"
        )
        
        await logger.log_unauthorized_access(
            resource="admin_panel",
            context=context
        )
    
    # 严重安全事件
    for i in range(3):
        context = create_audit_context(
            user_id=f"attacker_{i}",
            ip_address=f"203.0.113.{i + 1}"
        )
        
        await logger.log_attack_detected(
            attack_type="SQL注入攻击",
            source_ip=context.ip_address,
            context=context
        )
    
    # 9. 模拟合规性活动
    print("Simulating compliance activities...")
    
    context = create_audit_context(
        user_id="compliance_officer",
        ip_address="10.0.0.100"
    )
    
    await logger.log_audit_report(
        report_id="compliance_2024_001",
        report_type="季度合规报告",
        context=context
    )
    
    await logger.log_risk_assessment(
        assessment_id="risk_2024_001",
        risk_level="MEDIUM",
        context=context
    )
    
    # 10. 运行分析和报告
    print("Running analysis and generating reports...")
    
    # 异常检测
    anomalies = await advanced_analyzer.detect_anomalies(time_window=3600)
    if anomalies:
        print(f"Detected {len(anomalies)} anomalies:")
        for anomaly in anomalies:
            print(f"  - {anomaly}")
    
    # 用户行为分析
    for user_id in ["employee_1", "employee_2", "employee_3"]:
        profile = await advanced_analyzer.generate_user_behavior_profile(user_id)
        print(f"User profile for {user_id}: {profile['total_activities']} activities")
    
    # 安全风险预测
    risks = await advanced_analyzer.predict_security_risks()
    if risks:
        print(f"Predicted {len(risks)} security risks:")
        for risk in risks[:5]:  # 只显示前5个
            print(f"  - {risk['user_id']}: {risk['risk_level']} (score: {risk['risk_score']})")
    
    # 仪表板数据
    dashboard_data = await dashboard.get_dashboard_data()
    print(f"Dashboard data: {dashboard_data}")
    
    # 生成报告
    report_gen = AuditReportGenerator(logger)
    today = datetime.now().date()
    
    daily_report = await report_gen.generate_daily_report(today)
    print("Daily Report Preview:")
    print(daily_report[:500] + "..." if len(daily_report) > 500 else daily_report)
    
    compliance_report = await report_gen.generate_compliance_report(
        today - timedelta(days=30), today
    )
    print("Compliance Report Preview:")
    print(compliance_report[:500] + "..." if len(compliance_report) > 500 else compliance_report)
    
    # 11. 数据导出
    print("Exporting audit data...")
    
    # 导出安全事件
    security_filter = AuditFilter(categories={AuditCategory.SECURITY_EVENT})
    await logger.export_audit_logs(
        filter_obj=security_filter,
        export_path="security_events_export.json",
        format="json"
    )
    
    # 12. 性能测试
    print("Running performance test...")
    
    test_tool = AuditTestTool(logger)
    perf_results = await test_tool.run_performance_test(60)
    print(f"Performance test results: {perf_results}")
    
    # 13. 数据完整性验证
    print("Validating data integrity...")
    
    integrity_results = await test_tool.validate_audit_integrity()
    print(f"Integrity validation: {integrity_results}")
    
    # 14. 清理和归档
    print("Archiving old records...")
    
    archive_results = await archiver.archive_old_records(days=30)
    print(f"Archive results: {archive_results}")
    
    # 15. 获取最终统计
    print("Getting final statistics...")
    
    stats = await logger.get_statistics()
    print(f"Final statistics: {stats}")
    
    # 16. 清理资源
    print("Cleaning up...")
    
    await logger.close()
    
    print("Enterprise audit example completed successfully!")


# =============================================================================
# 性能基准测试
# =============================================================================

async def benchmark_test():
    """性能基准测试"""
    
    print("Starting audit logger benchmark...")
    
    # 测试配置
    test_configs = [
        {'queue_size': 1000, 'batch_size': 10},
        {'queue_size': 5000, 'batch_size': 50},
        {'queue_size': 10000, 'batch_size': 100}
    ]
    
    results = []
    
    for config in test_configs:
        print(f"Testing with config: {config}")
        
        # 创建测试记录器
        logger = AuditLogger(
            storage=SQLiteAuditStorage(f"benchmark_{config['queue_size']}.db"),
            queue_size=config['queue_size'],
            batch_size=config['batch_size']
        )
        
        # 性能测试
        test_tool = AuditTestTool(logger)
        perf_results = await test_tool.run_performance_test(30)
        
        perf_results['config'] = config
        results.append(perf_results)
        
        await logger.close()
    
    # 输出结果
    print("\nBenchmark Results:")
    for result in results:
        print(f"Config: {result['config']}")
        print(f"  Records/second: {result['records_per_second']:.2f}")
        print(f"  Error rate: {result['error_rate']:.4f}")
        print(f"  Duration: {result['duration']:.2f}s")
        print()
    
    return results


# =============================================================================
# 单元测试
# =============================================================================

class AuditLoggerTest:
    """审计记录器单元测试"""
    
    def __init__(self):
        self.test_logger = None
        self.test_storage = None
    
    async def setup(self):
        """测试设置"""
        self.test_storage = SQLiteAuditStorage("test_audit.db")
        self.test_logger = AuditLogger(storage=self.test_storage)
    
    async def teardown(self):
        """测试清理"""
        if self.test_logger:
            await self.test_logger.close()
        if os.path.exists("test_audit.db"):
            os.remove("test_audit.db")
    
    async def test_user_operation_logging(self):
        """测试用户操作日志记录"""
        context = create_audit_context(user_id="test_user", ip_address="127.0.0.1")
        
        await self.test_logger.log_user_login(
            user_id="test_user",
            success=True,
            context=context
        )
        
        # 验证记录是否被存储
        filter_obj = AuditFilter(user_ids={"test_user"})
        records = await self.test_logger.query_audit_logs(filter_obj)
        
        assert len(records) == 1
        assert records[0].context.user_id == "test_user"
        assert records[0].operation == UserOperation.LOGIN
    
    async def test_security_event_logging(self):
        """测试安全事件日志记录"""
        context = create_audit_context(user_id="test_user", ip_address="127.0.0.1")
        
        await self.test_logger.log_attack_detected(
            attack_type="测试攻击",
            source_ip="127.0.0.1",
            context=context
        )
        
        # 验证记录
        filter_obj = AuditFilter(categories={AuditCategory.SECURITY_EVENT})
        records = await self.test_logger.query_audit_logs(filter_obj)
        
        assert len(records) == 1
        assert records[0].category == AuditCategory.SECURITY_EVENT
        assert records[0].level == AuditLevel.CRITICAL
    
    async def test_data_access_logging(self):
        """测试数据访问日志记录"""
        context = create_audit_context(user_id="test_user", ip_address="127.0.0.1")
        
        await self.test_logger.log_data_read(
            resource="test_table",
            context=context,
            record_count=100
        )
        
        # 验证记录
        filter_obj = AuditFilter(categories={AuditCategory.DATA_ACCESS})
        records = await self.test_logger.query_audit_logs(filter_obj)
        
        assert len(records) == 1
        assert records[0].category == AuditCategory.DATA_ACCESS
        assert records[0].data['record_count'] == 100
    
    async def test_audit_filter(self):
        """测试审计过滤器"""
        context = create_audit_context(user_id="test_user", ip_address="127.0.0.1")
        
        # 记录不同级别的日志
        await self.test_logger.log_user_operation(
            operation=UserOperation.LOGIN,
            message="Info message",
            context=context,
            level=AuditLevel.INFO
        )
        
        await self.test_logger.log_user_operation(
            operation=UserOperation.LOGIN_FAILED,
            message="Warning message",
            context=context,
            level=AuditLevel.WARNING
        )
        
        # 测试过滤器
        info_filter = AuditFilter(levels={AuditLevel.INFO})
        info_records = await self.test_logger.query_audit_logs(info_filter)
        
        warning_filter = AuditFilter(levels={AuditLevel.WARNING})
        warning_records = await self.test_logger.query_audit_logs(warning_filter)
        
        assert len(info_records) == 1
        assert len(warning_records) == 1
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("Running audit logger tests...")
        
        await self.setup()
        
        try:
            await self.test_user_operation_logging()
            print("✓ User operation logging test passed")
            
            await self.test_security_event_logging()
            print("✓ Security event logging test passed")
            
            await self.test_data_access_logging()
            print("✓ Data access logging test passed")
            
            await self.test_audit_filter()
            print("✓ Audit filter test passed")
            
            print("All tests passed! 🎉")
            
        except Exception as e:
            print(f"Test failed: {e}")
            raise
        finally:
            await self.teardown()


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    import random
    
    print("L8 Audit Logger - Enterprise Edition")
    print("=====================================")
    
    # 运行基本示例
    print("\n1. Running basic example...")
    asyncio.run(example_usage())
    
    # 运行高级示例
    print("\n2. Running advanced example...")
    asyncio.run(advanced_example())
    
    # 运行企业级示例
    print("\n3. Running enterprise example...")
    asyncio.run(enterprise_example())
    
    # 运行性能基准测试
    print("\n4. Running benchmark test...")
    asyncio.run(benchmark_test())
    
    # 运行单元测试
    print("\n5. Running unit tests...")
    test_suite = AuditLoggerTest()
    asyncio.run(test_suite.run_all_tests())
    
    print("\nAll examples completed successfully! ✅")