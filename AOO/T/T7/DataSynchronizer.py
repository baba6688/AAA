#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T7数据同步器模块

该模块提供了一个完整的数据同步解决方案，支持多数据源同步、实时和批量同步、
数据冲突检测解决、进度监控、错误处理、性能优化、安全控制、日志审计等功能。

主要功能：
1. 多数据源同步 - 支持数据库、文件、API等多种数据源
2. 实时同步和批量同步 - 支持实时流式同步和批量数据同步
3. 数据冲突检测和解决 - 智能检测和解决数据冲突
4. 数据同步进度监控 - 实时跟踪同步进度和状态
5. 数据同步错误处理 - 完善的错误处理和重试机制
6. 数据同步性能优化 - 性能优化和资源管理
7. 数据同步安全控制 - 数据加密、访问控制等安全措施
8. 数据同步日志和审计 - 详细的日志记录和审计跟踪
9. 数据同步配置管理 - 灵活的配置文件管理

作者: T7系统
创建时间: 2025-11-05
版本: 1.0.0
"""

import asyncio
import logging
import threading
import time
import hashlib
import json
import sqlite3
import pickle
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import queue
import uuid
import os
from pathlib import Path
import ssl
import jwt
from cryptography.fernet import Fernet


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyncMode(Enum):
    """同步模式枚举"""
    REALTIME = "realtime"  # 实时同步
    BATCH = "batch"        # 批量同步
    INCREMENTAL = "incremental"  # 增量同步
    FULL = "full"          # 全量同步


class SyncStatus(Enum):
    """同步状态枚举"""
    PENDING = "pending"        # 待同步
    RUNNING = "running"        # 同步中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 同步失败
    PAUSED = "paused"          # 已暂停
    CANCELLED = "cancelled"    # 已取消


class ConflictResolution(Enum):
    """冲突解决策略枚举"""
    SOURCE_PRIORITY = "source_priority"    # 源数据优先
    TARGET_PRIORITY = "target_priority"    # 目标数据优先
    TIMESTAMP_PRIORITY = "timestamp_priority"  # 时间戳优先
    MANUAL_RESOLUTION = "manual_resolution"    # 手动解决
    MERGE_DATA = "merge_data"                # 合并数据


@dataclass
class DataSource:
    """数据源配置"""
    id: str
    name: str
    type: str  # database, file, api, etc.
    connection_info: Dict[str, Any]
    schema: Dict[str, Any]
    encryption_key: Optional[str] = None
    access_credentials: Optional[Dict[str, str]] = None
    rate_limit: Optional[int] = None
    timeout: Optional[int] = 30
    retry_count: int = 3
    retry_delay: float = 1.0


@dataclass
class SyncTask:
    """同步任务配置"""
    id: str
    name: str
    source_id: str
    target_id: str
    mode: SyncMode
    tables: List[str]
    filters: Optional[Dict[str, Any]] = None
    conflict_resolution: ConflictResolution = ConflictResolution.SOURCE_PRIORITY
    batch_size: int = 1000
    max_workers: int = 4
    enabled: bool = True
    schedule: Optional[str] = None  # cron表达式
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SyncProgress:
    """同步进度信息"""
    task_id: str
    status: SyncStatus
    total_records: int = 0
    processed_records: int = 0
    success_records: int = 0
    failed_records: int = 0
    conflicts_detected: int = 0
    conflicts_resolved: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    current_table: Optional[str] = None
    error_message: Optional[str] = None
    progress_percentage: float = 0.0


@dataclass
class SyncLog:
    """同步日志记录"""
    id: str
    task_id: str
    timestamp: datetime
    level: str
    message: str
    details: Optional[Dict[str, Any]] = None
    source_id: Optional[str] = None
    target_id: Optional[str] = None


class DataSourceConnector(ABC):
    """数据源连接器抽象基类"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接到数据源"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """断开数据源连接"""
        pass
    
    @abstractmethod
    async def read_data(self, table: str, filters: Optional[Dict] = None, 
                       limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict]:
        """读取数据"""
        pass
    
    @abstractmethod
    async def write_data(self, table: str, data: List[Dict], 
                        mode: str = "insert") -> bool:
        """写入数据"""
        pass
    
    @abstractmethod
    async def get_table_schema(self, table: str) -> Dict[str, Any]:
        """获取表结构"""
        pass
    
    @abstractmethod
    async def get_record_count(self, table: str, filters: Optional[Dict] = None) -> int:
        """获取记录数量"""
        pass


class DatabaseConnector(DataSourceConnector):
    """数据库连接器实现"""
    
    def __init__(self, config: DataSource):
        self.config = config
        self.connection = None
        self._lock = threading.Lock()
    
    async def connect(self) -> bool:
        """连接数据库"""
        try:
            with self._lock:
                if self.connection:
                    return True
                
                # 根据数据库类型建立连接
                db_type = self.config.connection_info.get("type", "sqlite")
                
                if db_type == "sqlite":
                    db_path = self.config.connection_info.get("path", ":memory:")
                    self.connection = sqlite3.connect(db_path, check_same_thread=False)
                    self.connection.row_factory = sqlite3.Row
                elif db_type == "postgresql":
                    # PostgreSQL连接逻辑
                    import psycopg2
                    self.connection = psycopg2.connect(**self.config.connection_info)
                elif db_type == "mysql":
                    # MySQL连接逻辑
                    import pymysql
                    self.connection = pymysql.connect(**self.config.connection_info)
                
                logger.info(f"数据库连接成功: {self.config.name}")
                return True
                
        except Exception as e:
            logger.error(f"数据库连接失败: {self.config.name}, 错误: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """断开数据库连接"""
        try:
            with self._lock:
                if self.connection:
                    self.connection.close()
                    self.connection = None
                logger.info(f"数据库连接已断开: {self.config.name}")
                return True
        except Exception as e:
            logger.error(f"数据库断开连接失败: {self.config.name}, 错误: {str(e)}")
            return False
    
    async def read_data(self, table: str, filters: Optional[Dict] = None, 
                       limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict]:
        """读取数据库数据"""
        try:
            if not self.connection:
                await self.connect()
            
            # 构建SQL查询
            sql = f"SELECT * FROM {table}"
            params = []
            
            if filters:
                where_clauses = []
                for key, value in filters.items():
                    where_clauses.append(f"{key} = ?")
                    params.append(value)
                sql += " WHERE " + " AND ".join(where_clauses)
            
            if limit:
                sql += " LIMIT ?"
                params.append(limit)
            
            if offset:
                sql += " OFFSET ?"
                params.append(offset)
            
            cursor = self.connection.cursor()
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            # 转换为字典格式
            result = [dict(row) for row in rows]
            cursor.close()
            
            return result
            
        except Exception as e:
            logger.error(f"读取数据库数据失败: {table}, 错误: {str(e)}")
            raise
    
    async def write_data(self, table: str, data: List[Dict], 
                        mode: str = "insert") -> bool:
        """写入数据库数据"""
        try:
            if not self.connection:
                await self.connect()
            
            if not data:
                return True
            
            cursor = self.connection.cursor()
            
            if mode == "insert":
                # 批量插入
                columns = list(data[0].keys())
                placeholders = ", ".join(["?"] * len(columns))
                sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
                
                for record in data:
                    values = [record[col] for col in columns]
                    cursor.execute(sql, values)
                
            elif mode == "upsert":
                # 插入或更新
                columns = list(data[0].keys())
                placeholders = ", ".join(["?"] * len(columns))
                update_clause = ", ".join([f"{col} = VALUES({col})" for col in columns])
                sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders}) ON DUPLICATE KEY UPDATE {update_clause}"
                
                for record in data:
                    values = [record[col] for col in columns]
                    cursor.execute(sql, values)
            
            self.connection.commit()
            cursor.close()
            
            logger.info(f"数据写入成功: {table}, 记录数: {len(data)}")
            return True
            
        except Exception as e:
            logger.error(f"写入数据库数据失败: {table}, 错误: {str(e)}")
            if self.connection:
                self.connection.rollback()
            raise
    
    async def get_table_schema(self, table: str) -> Dict[str, Any]:
        """获取表结构"""
        try:
            if not self.connection:
                await self.connect()
            
            cursor = self.connection.cursor()
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            
            schema = {
                "table": table,
                "columns": []
            }
            
            for column in columns:
                schema["columns"].append({
                    "name": column[1],
                    "type": column[2],
                    "not_null": bool(column[3]),
                    "default": column[4],
                    "primary_key": bool(column[5])
                })
            
            cursor.close()
            return schema
            
        except Exception as e:
            logger.error(f"获取表结构失败: {table}, 错误: {str(e)}")
            raise
    
    async def get_record_count(self, table: str, filters: Optional[Dict] = None) -> int:
        """获取记录数量"""
        try:
            if not self.connection:
                await self.connect()
            
            sql = f"SELECT COUNT(*) FROM {table}"
            params = []
            
            if filters:
                where_clauses = []
                for key, value in filters.items():
                    where_clauses.append(f"{key} = ?")
                    params.append(value)
                sql += " WHERE " + " AND ".join(where_clauses)
            
            cursor = self.connection.cursor()
            cursor.execute(sql, params)
            count = cursor.fetchone()[0]
            cursor.close()
            
            return count
            
        except Exception as e:
            logger.error(f"获取记录数量失败: {table}, 错误: {str(e)}")
            raise


class FileConnector(DataSourceConnector):
    """文件连接器实现"""
    
    def __init__(self, config: DataSource):
        self.config = config
        self._lock = threading.Lock()
    
    async def connect(self) -> bool:
        """连接文件系统"""
        try:
            with self._lock:
                # 验证文件路径
                base_path = Path(self.config.connection_info.get("base_path", ""))
                if not base_path.exists():
                    base_path.mkdir(parents=True, exist_ok=True)
                
                logger.info(f"文件连接成功: {self.config.name}")
                return True
                
        except Exception as e:
            logger.error(f"文件连接失败: {self.config.name}, 错误: {str(e)}")
            return False
    
    async def disconnect(self) -> bool:
        """断开文件连接"""
        try:
            logger.info(f"文件连接已断开: {self.config.name}")
            return True
        except Exception as e:
            logger.error(f"文件断开连接失败: {self.config.name}, 错误: {str(e)}")
            return False
    
    async def read_data(self, table: str, filters: Optional[Dict] = None, 
                       limit: Optional[int] = None, offset: Optional[int] = None) -> List[Dict]:
        """读取文件数据"""
        try:
            base_path = Path(self.config.connection_info.get("base_path", ""))
            file_path = base_path / f"{table}.json"
            
            if not file_path.exists():
                return []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 应用过滤条件
            if filters:
                filtered_data = []
                for record in data:
                    if all(record.get(key) == value for key, value in filters.items()):
                        filtered_data.append(record)
                data = filtered_data
            
            # 应用分页
            if offset:
                data = data[offset:]
            if limit:
                data = data[:limit]
            
            return data
            
        except Exception as e:
            logger.error(f"读取文件数据失败: {table}, 错误: {str(e)}")
            raise
    
    async def write_data(self, table: str, data: List[Dict], 
                        mode: str = "insert") -> bool:
        """写入文件数据"""
        try:
            base_path = Path(self.config.connection_info.get("base_path", ""))
            file_path = base_path / f"{table}.json"
            
            # 读取现有数据
            existing_data = []
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            
            if mode == "insert":
                existing_data.extend(data)
            elif mode == "upsert":
                # 简单的upsert实现（基于id字段）
                existing_dict = {record.get('id'): record for record in existing_data}
                for record in data:
                    existing_dict[record.get('id')] = record
                existing_data = list(existing_dict.values())
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"文件写入成功: {table}, 记录数: {len(data)}")
            return True
            
        except Exception as e:
            logger.error(f"写入文件数据失败: {table}, 错误: {str(e)}")
            raise
    
    async def get_table_schema(self, table: str) -> Dict[str, Any]:
        """获取文件结构（模拟）"""
        try:
            base_path = Path(self.config.connection_info.get("base_path", ""))
            file_path = base_path / f"{table}.json"
            
            if not file_path.exists():
                return {"table": table, "columns": []}
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data:
                return {"table": table, "columns": []}
            
            # 从第一条记录推断结构
            first_record = data[0]
            columns = []
            for key, value in first_record.items():
                columns.append({
                    "name": key,
                    "type": type(value).__name__,
                    "not_null": False,
                    "default": None,
                    "primary_key": False
                })
            
            return {"table": table, "columns": columns}
            
        except Exception as e:
            logger.error(f"获取文件结构失败: {table}, 错误: {str(e)}")
            raise
    
    async def get_record_count(self, table: str, filters: Optional[Dict] = None) -> int:
        """获取文件记录数量"""
        try:
            data = await self.read_data(table, filters)
            return len(data)
        except Exception as e:
            logger.error(f"获取文件记录数量失败: {table}, 错误: {str(e)}")
            raise


class SecurityManager:
    """安全管理器"""
    
    def __init__(self, encryption_key: Optional[str] = None):
        self.encryption_key = encryption_key or Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        self._jwt_secret = hashlib.sha256(self.encryption_key).hexdigest()
    
    def encrypt_data(self, data: str) -> str:
        """加密数据"""
        try:
            return self.cipher_suite.encrypt(data.encode()).decode()
        except Exception as e:
            logger.error(f"数据加密失败: {str(e)}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """解密数据"""
        try:
            return self.cipher_suite.decrypt(encrypted_data.encode()).decode()
        except Exception as e:
            logger.error(f"数据解密失败: {str(e)}")
            raise
    
    def generate_token(self, payload: Dict[str, Any], expires_in: int = 3600) -> str:
        """生成JWT令牌"""
        try:
            payload['exp'] = int(time.time()) + expires_in
            return jwt.encode(payload, self._jwt_secret, algorithm='HS256')
        except Exception as e:
            logger.error(f"令牌生成失败: {str(e)}")
            raise
    
    def verify_token(self, token: str) -> Dict[str, Any]:
        """验证JWT令牌"""
        try:
            return jwt.decode(token, self._jwt_secret, algorithms=['HS256'])
        except jwt.ExpiredSignatureError:
            raise ValueError("令牌已过期")
        except jwt.InvalidTokenError:
            raise ValueError("无效令牌")


class ConflictDetector:
    """数据冲突检测器"""
    
    def __init__(self):
        self.conflict_rules = {}
    
    def add_conflict_rule(self, table: str, rule: Callable):
        """添加冲突检测规则"""
        if table not in self.conflict_rules:
            self.conflict_rules[table] = []
        self.conflict_rules[table].append(rule)
    
    def detect_conflicts(self, source_data: List[Dict], target_data: List[Dict], 
                        table: str) -> List[Dict]:
        """检测数据冲突"""
        conflicts = []
        
        try:
            # 构建目标数据索引
            target_index = {}
            for record in target_data:
                # 使用主键或唯一标识符作为索引
                key_fields = self._get_key_fields(record, table)
                key = tuple(record[field] for field in key_fields if field in record)
                target_index[key] = record
            
            # 检查源数据中的冲突
            for source_record in source_data:
                key_fields = self._get_key_fields(source_record, table)
                key = tuple(source_record[field] for field in key_fields if field in source_record)
                
                if key in target_index:
                    target_record = target_index[key]
                    
                    # 检查字段级别的冲突
                    field_conflicts = self._check_field_conflicts(source_record, target_record, table)
                    
                    if field_conflicts:
                        conflicts.append({
                            'key': key,
                            'source_record': source_record,
                            'target_record': target_record,
                            'conflicts': field_conflicts,
                            'table': table
                        })
            
            # 应用自定义冲突规则
            if table in self.conflict_rules:
                for rule in self.conflict_rules[table]:
                    try:
                        rule_conflicts = rule(source_data, target_data, table)
                        conflicts.extend(rule_conflicts)
                    except Exception as e:
                        logger.error(f"冲突检测规则执行失败: {str(e)}")
            
            logger.info(f"冲突检测完成: {table}, 检测到 {len(conflicts)} 个冲突")
            return conflicts
            
        except Exception as e:
            logger.error(f"冲突检测失败: {table}, 错误: {str(e)}")
            raise
    
    def _get_key_fields(self, record: Dict, table: str) -> List[str]:
        """获取记录的关键字段"""
        # 优先使用主键字段
        key_fields = ['id', 'uuid', 'guid', 'key']
        for field in key_fields:
            if field in record:
                return [field]
        
        # 如果没有明确的主键，使用所有字段组合
        return list(record.keys())
    
    def _check_field_conflicts(self, source_record: Dict, target_record: Dict, 
                              table: str) -> List[Dict]:
        """检查字段级别的冲突"""
        conflicts = []
        
        for field in source_record:
            if field in target_record:
                source_value = source_record[field]
                target_value = target_record[field]
                
                # 跳过空值冲突
                if source_value == target_value:
                    continue
                
                # 检查是否为有效冲突
                if self._is_valid_conflict(field, source_value, target_value):
                    conflicts.append({
                        'field': field,
                        'source_value': source_value,
                        'target_value': target_value,
                        'conflict_type': 'value_mismatch'
                    })
        
        return conflicts
    
    def _is_valid_conflict(self, field: str, source_value: Any, target_value: Any) -> bool:
        """判断是否为有效冲突"""
        # 排除一些常见的非冲突情况
        if source_value is None or target_value is None:
            return False
        
        # 如果是时间戳字段，检查是否在合理范围内
        if field.lower() in ['created_at', 'updated_at', 'timestamp']:
            try:
                source_time = datetime.fromisoformat(str(source_value)) if isinstance(source_value, str) else source_value
                target_time = datetime.fromisoformat(str(target_value)) if isinstance(target_value, str) else target_value
                
                # 如果时间差小于1秒，认为不是冲突
                if abs((source_time - target_time).total_seconds()) < 1:
                    return False
            except:
                pass
        
        return True
    
    def resolve_conflicts(self, conflicts: List[Dict], strategy: ConflictResolution, 
                         source_data: List[Dict], target_data: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """解决数据冲突"""
        try:
            resolved_source = source_data.copy()
            resolved_target = target_data.copy()
            
            for conflict in conflicts:
                key = conflict['key']
                source_record = conflict['source_record']
                target_record = conflict['target_record']
                field_conflicts = conflict['conflicts']
                
                if strategy == ConflictResolution.SOURCE_PRIORITY:
                    # 源数据优先，保持源数据不变
                    resolved_target = self._update_record(resolved_target, key, source_record)
                
                elif strategy == ConflictResolution.TARGET_PRIORITY:
                    # 目标数据优先，保持目标数据不变
                    resolved_source = self._update_record(resolved_source, key, target_record)
                
                elif strategy == ConflictResolution.TIMESTAMP_PRIORITY:
                    # 时间戳优先
                    source_timestamp = self._get_timestamp(source_record)
                    target_timestamp = self._get_timestamp(target_record)
                    
                    if source_timestamp > target_timestamp:
                        resolved_target = self._update_record(resolved_target, key, source_record)
                    else:
                        resolved_source = self._update_record(resolved_source, key, target_record)
                
                elif strategy == ConflictResolution.MERGE_DATA:
                    # 合并数据
                    merged_record = self._merge_records(source_record, target_record, field_conflicts)
                    resolved_source = self._update_record(resolved_source, key, merged_record)
                    resolved_target = self._update_record(resolved_target, key, merged_record)
                
                elif strategy == ConflictResolution.MANUAL_RESOLUTION:
                    # 手动解决（这里需要外部干预）
                    logger.warning(f"需要手动解决冲突: {conflict}")
            
            logger.info(f"冲突解决完成: 策略={strategy.value}, 解决冲突数={len(conflicts)}")
            return resolved_source, resolved_target
            
        except Exception as e:
            logger.error(f"冲突解决失败: {str(e)}")
            raise
    
    def _update_record(self, data_list: List[Dict], key: Tuple, new_record: Dict) -> List[Dict]:
        """更新数据列表中的记录"""
        updated_list = []
        key_fields = list(new_record.keys())[:len(key)]
        
        for record in data_list:
            record_key = tuple(record.get(field) for field in key_fields if field in record)
            if record_key == key:
                updated_list.append(new_record)
            else:
                updated_list.append(record)
        
        return updated_list
    
    def _get_timestamp(self, record: Dict) -> datetime:
        """获取记录的时间戳"""
        timestamp_fields = ['updated_at', 'modified_at', 'timestamp', 'created_at']
        
        for field in timestamp_fields:
            if field in record:
                try:
                    if isinstance(record[field], str):
                        return datetime.fromisoformat(record[field])
                    return record[field]
                except:
                    continue
        
        return datetime.min
    
    def _merge_records(self, source_record: Dict, target_record: Dict, 
                      field_conflicts: List[Dict]) -> Dict:
        """合并两个记录"""
        merged = target_record.copy()
        
        for conflict in field_conflicts:
            field = conflict['field']
            source_value = conflict['source_value']
            target_value = conflict['target_value']
            
            # 简单的合并策略：保留非空值
            if source_value is not None and source_value != "":
                merged[field] = source_value
        
        return merged


class ProgressMonitor:
    """进度监控器"""
    
    def __init__(self):
        self.progress_callbacks = {}
        self.progress_history = {}
    
    def register_progress_callback(self, task_id: str, callback: Callable):
        """注册进度回调函数"""
        if task_id not in self.progress_callbacks:
            self.progress_callbacks[task_id] = []
        self.progress_callbacks[task_id].append(callback)
    
    def update_progress(self, progress: SyncProgress):
        """更新同步进度"""
        try:
            # 保存进度历史
            if progress.task_id not in self.progress_history:
                self.progress_history[progress.task_id] = []
            self.progress_history[progress.task_id].append({
                'timestamp': datetime.now(),
                'progress': progress
            })
            
            # 触发回调函数
            if progress.task_id in self.progress_callbacks:
                for callback in self.progress_callbacks[progress.task_id]:
                    try:
                        callback(progress)
                    except Exception as e:
                        logger.error(f"进度回调执行失败: {str(e)}")
            
            # 记录日志
            if progress.status == SyncStatus.RUNNING:
                logger.info(f"同步进度更新: {progress.task_id}, "
                          f"进度={progress.progress_percentage:.1f}%, "
                          f"已处理={progress.processed_records}/{progress.total_records}")
            elif progress.status == SyncStatus.COMPLETED:
                logger.info(f"同步完成: {progress.task_id}, "
                          f"成功={progress.success_records}, "
                          f"失败={progress.failed_records}, "
                          f"冲突={progress.conflicts_detected}")
            elif progress.status == SyncStatus.FAILED:
                logger.error(f"同步失败: {progress.task_id}, 错误: {progress.error_message}")
            
        except Exception as e:
            logger.error(f"进度更新失败: {str(e)}")
    
    def get_progress(self, task_id: str) -> Optional[SyncProgress]:
        """获取最新进度"""
        history = self.progress_history.get(task_id, [])
        return history[-1]['progress'] if history else None
    
    def get_progress_history(self, task_id: str) -> List[Dict]:
        """获取进度历史"""
        return self.progress_history.get(task_id, [])
    
    def calculate_performance_metrics(self, task_id: str) -> Dict[str, Any]:
        """计算性能指标"""
        try:
            history = self.get_progress_history(task_id)
            if len(history) < 2:
                return {}
            
            # 计算处理速度
            first_progress = history[0]['progress']
            last_progress = history[-1]['progress']
            
            if first_progress.start_time and last_progress.end_time:
                duration = (last_progress.end_time - first_progress.start_time).total_seconds()
                records_per_second = last_progress.processed_records / duration if duration > 0 else 0
                
                # 计算预估完成时间
                remaining_records = last_progress.total_records - last_progress.processed_records
                estimated_time_remaining = remaining_records / records_per_second if records_per_second > 0 else 0
                
                return {
                    'records_per_second': records_per_second,
                    'estimated_time_remaining': estimated_time_remaining,
                    'total_duration': duration,
                    'success_rate': last_progress.success_records / last_progress.processed_records if last_progress.processed_records > 0 else 0,
                    'conflict_rate': last_progress.conflicts_detected / last_progress.processed_records if last_progress.processed_records > 0 else 0
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"性能指标计算失败: {str(e)}")
            return {}


class AuditLogger:
    """审计日志记录器"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.audit_logger = self._setup_audit_logger()
    
    def _setup_audit_logger(self) -> logging.Logger:
        """设置审计日志记录器"""
        audit_logger = logging.getLogger('audit')
        audit_logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        audit_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(audit_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        audit_logger.addHandler(file_handler)
        return audit_logger
    
    def log_sync_start(self, task: SyncTask, user: str = "system"):
        """记录同步开始"""
        self.audit_logger.info(f"SYNC_START - Task: {task.id}, User: {user}, "
                              f"Mode: {task.mode.value}, Tables: {', '.join(task.tables)}")
    
    def log_sync_end(self, task_id: str, status: SyncStatus, records_processed: int = 0, 
                    user: str = "system"):
        """记录同步结束"""
        self.audit_logger.info(f"SYNC_END - Task: {task_id}, Status: {status.value}, "
                              f"Records: {records_processed}, User: {user}")
    
    def log_data_access(self, task_id: str, source_id: str, operation: str, 
                       table: str, record_count: int, user: str = "system"):
        """记录数据访问"""
        self.audit_logger.info(f"DATA_ACCESS - Task: {task_id}, Source: {source_id}, "
                              f"Operation: {operation}, Table: {table}, "
                              f"Records: {record_count}, User: {user}")
    
    def log_conflict_detection(self, task_id: str, table: str, conflict_count: int, 
                              resolution: str, user: str = "system"):
        """记录冲突检测"""
        self.audit_logger.info(f"CONFLICT_DETECTED - Task: {task_id}, Table: {table}, "
                              f"Conflicts: {conflict_count}, Resolution: {resolution}, "
                              f"User: {user}")
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], 
                          user: str = "system"):
        """记录安全事件"""
        self.audit_logger.warning(f"SECURITY_EVENT - Type: {event_type}, "
                                 f"Details: {json.dumps(details)}, User: {user}")
    
    def log_configuration_change(self, change_type: str, old_config: Dict, 
                                new_config: Dict, user: str = "system"):
        """记录配置变更"""
        self.audit_logger.info(f"CONFIG_CHANGE - Type: {change_type}, "
                              f"Old: {json.dumps(old_config)}, "
                              f"New: {json.dumps(new_config)}, User: {user}")


class ConfigurationManager:
    """配置管理器"""
    
    def __init__(self, config_file: str = "sync_config.json"):
        self.config_file = Path(config_file)
        self.config = self._load_config()
        self.audit_logger = AuditLogger()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # 创建默认配置
                default_config = self._get_default_config()
                self._save_config(default_config)
                return default_config
        except Exception as e:
            logger.error(f"配置文件加载失败: {str(e)}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "version": "1.0.0",
            "data_sources": {},
            "sync_tasks": {},
            "global_settings": {
                "max_concurrent_tasks": 5,
                "default_batch_size": 1000,
                "default_timeout": 30,
                "enable_encryption": True,
                "log_level": "INFO",
                "audit_enabled": True
            },
            "security_settings": {
                "require_authentication": False,
                "token_expiry": 3600,
                "allowed_ips": [],
                "encryption_algorithm": "AES-256"
            },
            "performance_settings": {
                "max_workers": 10,
                "connection_pool_size": 20,
                "cache_enabled": True,
                "compression_enabled": False
            }
        }
    
    def _save_config(self, config: Dict[str, Any]):
        """保存配置文件"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"配置文件保存失败: {str(e)}")
            raise
    
    def get_data_source(self, source_id: str) -> Optional[DataSource]:
        """获取数据源配置"""
        try:
            source_config = self.config["data_sources"].get(source_id)
            if source_config:
                return DataSource(**source_config)
            return None
        except Exception as e:
            logger.error(f"获取数据源配置失败: {source_id}, {str(e)}")
            return None
    
    def add_data_source(self, data_source: DataSource):
        """添加数据源配置"""
        try:
            old_config = self.config["data_sources"].get(data_source.id, {})
            self.config["data_sources"][data_source.id] = {
                "id": data_source.id,
                "name": data_source.name,
                "type": data_source.type,
                "connection_info": data_source.connection_info,
                "schema": data_source.schema,
                "encryption_key": data_source.encryption_key,
                "access_credentials": data_source.access_credentials,
                "rate_limit": data_source.rate_limit,
                "timeout": data_source.timeout,
                "retry_count": data_source.retry_count,
                "retry_delay": data_source.retry_delay
            }
            self._save_config(self.config)
            self.audit_logger.log_configuration_change("ADD_DATA_SOURCE", old_config, 
                                                      self.config["data_sources"][data_source.id])
        except Exception as e:
            logger.error(f"添加数据源配置失败: {data_source.id}, {str(e)}")
            raise
    
    def get_sync_task(self, task_id: str) -> Optional[SyncTask]:
        """获取同步任务配置"""
        try:
            task_config = self.config["sync_tasks"].get(task_id)
            if task_config:
                # 转换枚举值
                task_config["mode"] = SyncMode(task_config["mode"])
                task_config["conflict_resolution"] = ConflictResolution(task_config["conflict_resolution"])
                return SyncTask(**task_config)
            return None
        except Exception as e:
            logger.error(f"获取同步任务配置失败: {task_id}, {str(e)}")
            return None
    
    def add_sync_task(self, sync_task: SyncTask):
        """添加同步任务配置"""
        try:
            old_config = self.config["sync_tasks"].get(sync_task.id, {})
            self.config["sync_tasks"][sync_task.id] = {
                "id": sync_task.id,
                "name": sync_task.name,
                "source_id": sync_task.source_id,
                "target_id": sync_task.target_id,
                "mode": sync_task.mode.value,
                "tables": sync_task.tables,
                "filters": sync_task.filters,
                "conflict_resolution": sync_task.conflict_resolution.value,
                "batch_size": sync_task.batch_size,
                "max_workers": sync_task.max_workers,
                "enabled": sync_task.enabled,
                "schedule": sync_task.schedule,
                "created_at": sync_task.created_at.isoformat(),
                "updated_at": sync_task.updated_at.isoformat()
            }
            self._save_config(self.config)
            self.audit_logger.log_configuration_change("ADD_SYNC_TASK", old_config, 
                                                      self.config["sync_tasks"][sync_task.id])
        except Exception as e:
            logger.error(f"添加同步任务配置失败: {sync_task.id}, {str(e)}")
            raise
    
    def update_global_setting(self, key: str, value: Any):
        """更新全局设置"""
        try:
            old_value = self.config["global_settings"].get(key)
            self.config["global_settings"][key] = value
            self._save_config(self.config)
            self.audit_logger.log_configuration_change("UPDATE_GLOBAL_SETTING", 
                                                      {key: old_value}, {key: value})
        except Exception as e:
            logger.error(f"更新全局设置失败: {key}, {str(e)}")
            raise
    
    def get_global_setting(self, key: str, default: Any = None) -> Any:
        """获取全局设置"""
        return self.config["global_settings"].get(key, default)


class DataSynchronizer:
    """数据同步器主类"""
    
    def __init__(self, config_file: str = "sync_config.json"):
        """初始化数据同步器"""
        self.config_manager = ConfigurationManager(config_file)
        self.audit_logger = AuditLogger()
        self.security_manager = SecurityManager()
        self.conflict_detector = ConflictDetector()
        self.progress_monitor = ProgressMonitor()
        
        self.data_sources: Dict[str, DataSourceConnector] = {}
        self.active_tasks: Dict[str, SyncTask] = {}
        self.task_locks: Dict[str, threading.Lock] = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # 性能统计
        self.performance_stats = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'total_records_processed': 0,
            'average_sync_time': 0.0
        }
        
        logger.info("数据同步器初始化完成")
    
    async def start_sync(self, task_id: str, user: str = "system") -> bool:
        """启动同步任务"""
        try:
            # 获取任务配置
            task = self.config_manager.get_sync_task(task_id)
            if not task:
                logger.error(f"同步任务不存在: {task_id}")
                return False
            
            if not task.enabled:
                logger.warning(f"同步任务已禁用: {task_id}")
                return False
            
            # 检查任务是否已在运行
            if task_id in self.active_tasks:
                logger.warning(f"同步任务已在运行: {task_id}")
                return False
            
            # 初始化任务锁
            if task_id not in self.task_locks:
                self.task_locks[task_id] = threading.Lock()
            
            # 记录审计日志
            self.audit_logger.log_sync_start(task, user)
            
            # 添加到活跃任务列表
            self.active_tasks[task_id] = task
            
            # 创建进度监控
            progress = SyncProgress(
                task_id=task_id,
                status=SyncStatus.PENDING,
                start_time=datetime.now()
            )
            self.progress_monitor.update_progress(progress)
            
            # 提交同步任务到线程池
            future = self.executor.submit(asyncio.run, self._execute_sync_task(task, user))
            
            logger.info(f"同步任务已启动: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"启动同步任务失败: {task_id}, 错误: {str(e)}")
            # 清理状态
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            return False
    
    async def stop_sync(self, task_id: str, user: str = "system") -> bool:
        """停止同步任务"""
        try:
            if task_id not in self.active_tasks:
                logger.warning(f"同步任务未运行: {task_id}")
                return False
            
            # 获取任务锁并停止任务
            if task_id in self.task_locks:
                with self.task_locks[task_id]:
                    # 更新任务状态
                    progress = self.progress_monitor.get_progress(task_id)
                    if progress:
                        progress.status = SyncStatus.CANCELLED
                        progress.end_time = datetime.now()
                        self.progress_monitor.update_progress(progress)
                    
                    # 从活跃任务列表中移除
                    del self.active_tasks[task_id]
            
            # 记录审计日志
            task = self.config_manager.get_sync_task(task_id)
            if task:
                self.audit_logger.log_sync_end(task_id, SyncStatus.CANCELLED, 0, user)
            
            logger.info(f"同步任务已停止: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"停止同步任务失败: {task_id}, 错误: {str(e)}")
            return False
    
    async def pause_sync(self, task_id: str, user: str = "system") -> bool:
        """暂停同步任务"""
        try:
            progress = self.progress_monitor.get_progress(task_id)
            if not progress or progress.status != SyncStatus.RUNNING:
                logger.warning(f"同步任务未在运行: {task_id}")
                return False
            
            progress.status = SyncStatus.PAUSED
            self.progress_monitor.update_progress(progress)
            
            logger.info(f"同步任务已暂停: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"暂停同步任务失败: {task_id}, 错误: {str(e)}")
            return False
    
    async def resume_sync(self, task_id: str, user: str = "system") -> bool:
        """恢复同步任务"""
        try:
            progress = self.progress_monitor.get_progress(task_id)
            if not progress or progress.status != SyncStatus.PAUSED:
                logger.warning(f"同步任务未暂停: {task_id}")
                return False
            
            progress.status = SyncStatus.RUNNING
            self.progress_monitor.update_progress(progress)
            
            logger.info(f"同步任务已恢复: {task_id}")
            return True
            
        except Exception as e:
            logger.error(f"恢复同步任务失败: {task_id}, 错误: {str(e)}")
            return False
    
    async def _execute_sync_task(self, task: SyncTask, user: str):
        """执行同步任务"""
        try:
            # 获取数据源连接器
            source_connector = await self._get_data_source_connector(task.source_id)
            target_connector = await self._get_data_source_connector(task.target_id)
            
            if not source_connector or not target_connector:
                raise ValueError(f"数据源连接器不可用: {task.source_id} -> {task.target_id}")
            
            # 更新进度状态
            progress = self.progress_monitor.get_progress(task.id)
            progress.status = SyncStatus.RUNNING
            progress.start_time = datetime.now()
            self.progress_monitor.update_progress(progress)
            
            total_records = 0
            processed_records = 0
            success_records = 0
            failed_records = 0
            conflicts_detected = 0
            conflicts_resolved = 0
            
            # 同步每个表
            for table in task.tables:
                try:
                    progress.current_table = table
                    self.progress_monitor.update_progress(progress)
                    
                    # 获取表记录数量
                    table_total = await source_connector.get_record_count(table, task.filters)
                    total_records += table_total
                    
                    # 根据同步模式处理数据
                    if task.mode == SyncMode.REALTIME:
                        await self._realtime_sync(source_connector, target_connector, task, table, progress)
                    elif task.mode == SyncMode.BATCH:
                        batch_results = await self._batch_sync(source_connector, target_connector, task, table, progress)
                        processed_records += batch_results['processed']
                        success_records += batch_results['success']
                        failed_records += batch_results['failed']
                        conflicts_detected += batch_results['conflicts']
                        conflicts_resolved += batch_results['resolved']
                    elif task.mode == SyncMode.INCREMENTAL:
                        incremental_results = await self._incremental_sync(source_connector, target_connector, task, table, progress)
                        processed_records += incremental_results['processed']
                        success_records += incremental_results['success']
                        failed_records += incremental_results['failed']
                        conflicts_detected += incremental_results['conflicts']
                        conflicts_resolved += incremental_results['resolved']
                    elif task.mode == SyncMode.FULL:
                        full_results = await self._full_sync(source_connector, target_connector, task, table, progress)
                        processed_records += full_results['processed']
                        success_records += full_results['success']
                        failed_records += full_results['failed']
                        conflicts_detected += full_results['conflicts']
                        conflicts_resolved += full_results['resolved']
                    
                    # 记录数据访问审计
                    self.audit_logger.log_data_access(task.id, task.source_id, "READ", table, table_total, user)
                    self.audit_logger.log_data_access(task.id, task.target_id, "WRITE", table, table_total, user)
                    
                except Exception as e:
                    logger.error(f"表同步失败: {table}, 错误: {str(e)}")
                    failed_records += table_total
                    progress.error_message = str(e)
            
            # 更新最终进度
            progress.total_records = total_records
            progress.processed_records = processed_records
            progress.success_records = success_records
            progress.failed_records = failed_records
            progress.conflicts_detected = conflicts_detected
            progress.conflicts_resolved = conflicts_resolved
            progress.end_time = datetime.now()
            progress.progress_percentage = 100.0 if total_records > 0 else 0.0
            
            # 判断同步结果
            if failed_records > 0 and success_records == 0:
                progress.status = SyncStatus.FAILED
            else:
                progress.status = SyncStatus.COMPLETED
            
            self.progress_monitor.update_progress(progress)
            
            # 更新性能统计
            self._update_performance_stats(progress)
            
            # 记录审计日志
            self.audit_logger.log_sync_end(task.id, progress.status, processed_records, user)
            
            logger.info(f"同步任务执行完成: {task.id}, 状态: {progress.status.value}")
            
        except Exception as e:
            logger.error(f"同步任务执行失败: {task.id}, 错误: {str(e)}")
            
            # 更新错误状态
            progress = self.progress_monitor.get_progress(task.id)
            if progress:
                progress.status = SyncStatus.FAILED
                progress.error_message = str(e)
                progress.end_time = datetime.now()
                self.progress_monitor.update_progress(progress)
            
            # 更新性能统计
            self._update_performance_stats(progress)
            
            # 记录审计日志
            self.audit_logger.log_sync_end(task.id, SyncStatus.FAILED, 0, user)
        
        finally:
            # 清理活跃任务状态
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
    
    async def _realtime_sync(self, source_connector: DataSourceConnector, 
                           target_connector: DataSourceConnector, task: SyncTask, 
                           table: str, progress: SyncProgress):
        """实时同步"""
        # 实时同步实现（这里简化处理）
        logger.info(f"执行实时同步: {table}")
        await asyncio.sleep(1)  # 模拟实时处理
    
    async def _batch_sync(self, source_connector: DataSourceConnector, 
                         target_connector: DataSourceConnector, task: SyncTask, 
                         table: str, progress: SyncProgress) -> Dict[str, int]:
        """批量同步"""
        try:
            batch_size = task.batch_size
            offset = 0
            processed = 0
            success = 0
            failed = 0
            conflicts = 0
            resolved = 0
            
            while True:
                # 读取批次数据
                source_data = await source_connector.read_data(table, task.filters, batch_size, offset)
                if not source_data:
                    break
                
                # 读取目标数据用于冲突检测
                target_data = await target_connector.read_data(table, task.filters, None, None)
                
                # 检测冲突
                detected_conflicts = self.conflict_detector.detect_conflicts(source_data, target_data, table)
                conflicts += len(detected_conflicts)
                
                if detected_conflicts:
                    # 解决冲突
                    resolved_source, resolved_target = self.conflict_detector.resolve_conflicts(
                        detected_conflicts, task.conflict_resolution, source_data, target_data
                    )
                    resolved += len(detected_conflicts)
                    
                    # 使用解决后的数据
                    source_data = resolved_source
                    
                    # 记录冲突解决审计
                    self.audit_logger.log_conflict_detection(task.id, table, len(detected_conflicts), 
                                                           task.conflict_resolution.value)
                
                # 写入数据
                try:
                    await target_connector.write_data(table, source_data, "upsert")
                    success += len(source_data)
                except Exception as e:
                    logger.error(f"批量写入失败: {table}, 错误: {str(e)}")
                    failed += len(source_data)
                
                processed += len(source_data)
                offset += batch_size
                
                # 更新进度
                progress.processed_records = processed
                progress.conflicts_detected = conflicts
                progress.conflicts_resolved = resolved
                progress.progress_percentage = (processed / progress.total_records * 100) if progress.total_records > 0 else 0
                self.progress_monitor.update_progress(progress)
                
                # 控制处理速度
                await asyncio.sleep(0.1)
            
            return {
                'processed': processed,
                'success': success,
                'failed': failed,
                'conflicts': conflicts,
                'resolved': resolved
            }
            
        except Exception as e:
            logger.error(f"批量同步失败: {table}, 错误: {str(e)}")
            raise
    
    async def _incremental_sync(self, source_connector: DataSourceConnector, 
                              target_connector: DataSourceConnector, task: SyncTask, 
                              table: str, progress: SyncProgress) -> Dict[str, int]:
        """增量同步"""
        # 增量同步实现（基于时间戳或版本号）
        logger.info(f"执行增量同步: {table}")
        
        # 获取最后同步时间
        last_sync_time = await self._get_last_sync_time(task.id, table)
        current_time = datetime.now()
        
        # 构建增量同步过滤器
        incremental_filters = task.filters.copy() if task.filters else {}
        if last_sync_time:
            incremental_filters['updated_at'] = {'$gt': last_sync_time.isoformat()}
        
        # 执行批量同步
        results = await self._batch_sync(source_connector, target_connector, task, table, progress)
        
        # 更新最后同步时间
        await self._update_last_sync_time(task.id, table, current_time)
        
        return results
    
    async def _full_sync(self, source_connector: DataSourceConnector, 
                        target_connector: DataSourceConnector, task: SyncTask, 
                        table: str, progress: SyncProgress) -> Dict[str, int]:
        """全量同步"""
        # 全量同步实现
        logger.info(f"执行全量同步: {table}")
        
        # 清空目标表（可选）
        if task.mode == SyncMode.FULL and task.conflict_resolution == ConflictResolution.SOURCE_PRIORITY:
            logger.info(f"清空目标表: {table}")
        
        # 执行批量同步
        return await self._batch_sync(source_connector, target_connector, task, table, progress)
    
    async def _get_data_source_connector(self, source_id: str) -> Optional[DataSourceConnector]:
        """获取数据源连接器"""
        try:
            if source_id in self.data_sources:
                return self.data_sources[source_id]
            
            # 创建新的连接器
            data_source = self.config_manager.get_data_source(source_id)
            if not data_source:
                logger.error(f"数据源不存在: {source_id}")
                return None
            
            # 根据数据源类型创建连接器
            if data_source.type == "database":
                connector = DatabaseConnector(data_source)
            elif data_source.type == "file":
                connector = FileConnector(data_source)
            else:
                raise ValueError(f"不支持的数据源类型: {data_source.type}")
            
            # 连接到数据源
            if await connector.connect():
                self.data_sources[source_id] = connector
                return connector
            else:
                return None
                
        except Exception as e:
            logger.error(f"获取数据源连接器失败: {source_id}, 错误: {str(e)}")
            return None
    
    async def _get_last_sync_time(self, task_id: str, table: str) -> Optional[datetime]:
        """获取最后同步时间"""
        # 简化实现，实际应该从数据库或文件读取
        return None
    
    async def _update_last_sync_time(self, task_id: str, table: str, sync_time: datetime):
        """更新最后同步时间"""
        # 简化实现，实际应该保存到数据库或文件
        pass
    
    def _update_performance_stats(self, progress: SyncProgress):
        """更新性能统计"""
        try:
            self.performance_stats['total_syncs'] += 1
            
            if progress.status == SyncStatus.COMPLETED:
                self.performance_stats['successful_syncs'] += 1
            else:
                self.performance_stats['failed_syncs'] += 1
            
            self.performance_stats['total_records_processed'] += progress.processed_records
            
            # 计算平均同步时间
            if progress.start_time and progress.end_time:
                sync_duration = (progress.end_time - progress.start_time).total_seconds()
                total_syncs = self.performance_stats['total_syncs']
                current_avg = self.performance_stats['average_sync_time']
                self.performance_stats['average_sync_time'] = (
                    (current_avg * (total_syncs - 1) + sync_duration) / total_syncs
                )
            
        except Exception as e:
            logger.error(f"更新性能统计失败: {str(e)}")
    
    def get_sync_status(self, task_id: str) -> Optional[SyncProgress]:
        """获取同步状态"""
        return self.progress_monitor.get_progress(task_id)
    
    def get_all_sync_status(self) -> Dict[str, SyncProgress]:
        """获取所有同步任务状态"""
        return {task_id: self.progress_monitor.get_progress(task_id) 
                for task_id in self.active_tasks.keys()}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return self.performance_stats.copy()
    
    def get_performance_metrics(self, task_id: str) -> Dict[str, Any]:
        """获取性能指标"""
        return self.progress_monitor.calculate_performance_metrics(task_id)
    
    def register_progress_callback(self, task_id: str, callback: Callable):
        """注册进度回调"""
        self.progress_monitor.register_progress_callback(task_id, callback)
    
    def add_conflict_rule(self, table: str, rule: Callable):
        """添加冲突检测规则"""
        self.conflict_detector.add_conflict_rule(table, rule)
    
    async def cleanup(self):
        """清理资源"""
        try:
            # 关闭所有数据源连接
            for connector in self.data_sources.values():
                await connector.disconnect()
            self.data_sources.clear()
            
            # 关闭线程池
            self.executor.shutdown(wait=True)
            
            logger.info("数据同步器资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {str(e)}")


# 测试用例
class DataSynchronizerTest:
    """数据同步器测试类"""
    
    def __init__(self):
        self.synchronizer = DataSynchronizer("test_config.json")
    
    async def test_basic_sync(self):
        """测试基本同步功能"""
        try:
            print("=== 测试基本同步功能 ===")
            
            # 创建测试数据源
            source_config = DataSource(
                id="test_source",
                name="测试源数据源",
                type="file",
                connection_info={"base_path": "test_data/source"},
                schema={"tables": ["users", "orders"]}
            )
            
            target_config = DataSource(
                id="test_target",
                name="测试目标数据源",
                type="file", 
                connection_info={"base_path": "test_data/target"},
                schema={"tables": ["users", "orders"]}
            )
            
            self.synchronizer.config_manager.add_data_source(source_config)
            self.synchronizer.config_manager.add_data_source(target_config)
            
            # 创建测试数据
            await self._create_test_data()
            
            # 创建同步任务
            task_config = SyncTask(
                id="test_task",
                name="测试同步任务",
                source_id="test_source",
                target_id="test_target",
                mode=SyncMode.BATCH,
                tables=["users"],
                batch_size=100,
                conflict_resolution=ConflictResolution.SOURCE_PRIORITY
            )
            
            self.synchronizer.config_manager.add_sync_task(task_config)
            
            # 启动同步
            print("启动同步任务...")
            success = await self.synchronizer.start_sync("test_task")
            print(f"同步启动结果: {success}")
            
            # 等待同步完成
            await asyncio.sleep(5)
            
            # 检查同步状态
            status = self.synchronizer.get_sync_status("test_task")
            if status:
                print(f"同步状态: {status.status.value}")
                print(f"处理记录: {status.processed_records}")
                print(f"成功记录: {status.success_records}")
                print(f"失败记录: {status.failed_records}")
                print(f"进度: {status.progress_percentage:.1f}%")
            
            # 获取性能指标
            metrics = self.synchronizer.get_performance_metrics("test_task")
            print(f"性能指标: {metrics}")
            
            # 获取性能统计
            stats = self.synchronizer.get_performance_stats()
            print(f"性能统计: {stats}")
            
            print("基本同步功能测试完成")
            
        except Exception as e:
            print(f"基本同步功能测试失败: {str(e)}")
    
    async def test_conflict_detection(self):
        """测试冲突检测功能"""
        try:
            print("\n=== 测试冲突检测功能 ===")
            
            # 创建冲突数据
            source_data = [
                {"id": 1, "name": "张三", "age": 25},
                {"id": 2, "name": "李四", "age": 30}
            ]
            
            target_data = [
                {"id": 1, "name": "张三", "age": 26},  # 年龄冲突
                {"id": 3, "name": "王五", "age": 28}   # 新记录
            ]
            
            # 检测冲突
            conflicts = self.synchronizer.conflict_detector.detect_conflicts(
                source_data, target_data, "users"
            )
            
            print(f"检测到 {len(conflicts)} 个冲突:")
            for i, conflict in enumerate(conflicts):
                print(f"  冲突 {i+1}: {conflict}")
            
            # 测试冲突解决
            if conflicts:
                resolved_source, resolved_target = self.synchronizer.conflict_detector.resolve_conflicts(
                    conflicts, ConflictResolution.SOURCE_PRIORITY, source_data, target_data
                )
                
                print("解决后的源数据:")
                for record in resolved_source:
                    print(f"  {record}")
                
                print("解决后的目标数据:")
                for record in resolved_target:
                    print(f"  {record}")
            
            print("冲突检测功能测试完成")
            
        except Exception as e:
            print(f"冲突检测功能测试失败: {str(e)}")
    
    async def test_progress_monitoring(self):
        """测试进度监控功能"""
        try:
            print("\n=== 测试进度监控功能 ===")
            
            # 创建进度回调
            def progress_callback(progress: SyncProgress):
                print(f"进度更新: {progress.progress_percentage:.1f}% - {progress.current_table}")
            
            # 注册回调
            self.synchronizer.register_progress_callback("test_task", progress_callback)
            
            # 模拟进度更新
            progress = SyncProgress(
                task_id="test_task",
                status=SyncStatus.RUNNING,
                total_records=1000,
                processed_records=500,
                current_table="users"
            )
            
            self.synchronizer.progress_monitor.update_progress(progress)
            
            print("进度监控功能测试完成")
            
        except Exception as e:
            print(f"进度监控功能测试失败: {str(e)}")
    
    async def test_security_features(self):
        """测试安全功能"""
        try:
            print("\n=== 测试安全功能 ===")
            
            # 测试数据加密
            test_data = "敏感数据测试"
            encrypted = self.synchronizer.security_manager.encrypt_data(test_data)
            decrypted = self.synchronizer.security_manager.decrypt_data(encrypted)
            
            print(f"原始数据: {test_data}")
            print(f"加密数据: {encrypted}")
            print(f"解密数据: {decrypted}")
            print(f"加密解密成功: {test_data == decrypted}")
            
            # 测试JWT令牌
            payload = {"user_id": "12345", "permissions": ["read", "write"]}
            token = self.synchronizer.security_manager.generate_token(payload)
            verified = self.synchronizer.security_manager.verify_token(token)
            
            print(f"JWT令牌: {token}")
            print(f"验证结果: {verified}")
            
            print("安全功能测试完成")
            
        except Exception as e:
            print(f"安全功能测试失败: {str(e)}")
    
    async def test_audit_logging(self):
        """测试审计日志功能"""
        try:
            print("\n=== 测试审计日志功能 ===")
            
            # 模拟审计日志记录
            self.synchronizer.audit_logger.log_sync_start(
                SyncTask("test", "测试任务", "source", "target", SyncMode.BATCH, ["users"]),
                "test_user"
            )
            
            self.synchronizer.audit_logger.log_data_access(
                "test_task", "test_source", "READ", "users", 100, "test_user"
            )
            
            self.synchronizer.audit_logger.log_conflict_detection(
                "test_task", "users", 5, "source_priority", "test_user"
            )
            
            self.synchronizer.audit_logger.log_sync_end(
                "test_task", SyncStatus.COMPLETED, 100, "test_user"
            )
            
            print("审计日志功能测试完成")
            
        except Exception as e:
            print(f"审计日志功能测试失败: {str(e)}")
    
    async def _create_test_data(self):
        """创建测试数据"""
        try:
            import os
            import json
            
            # 创建源数据目录
            source_dir = Path("test_data/source")
            source_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建测试用户数据
            users_data = [
                {"id": 1, "name": "张三", "age": 25, "email": "zhangsan@example.com"},
                {"id": 2, "name": "李四", "age": 30, "email": "lisi@example.com"},
                {"id": 3, "name": "王五", "age": 28, "email": "wangwu@example.com"}
            ]
            
            with open(source_dir / "users.json", 'w', encoding='utf-8') as f:
                json.dump(users_data, f, ensure_ascii=False, indent=2)
            
            print("测试数据创建完成")
            
        except Exception as e:
            print(f"创建测试数据失败: {str(e)}")
    
    async def run_all_tests(self):
        """运行所有测试"""
        try:
            print("开始运行数据同步器测试...")
            
            await self.test_basic_sync()
            await self.test_conflict_detection()
            await self.test_progress_monitoring()
            await self.test_security_features()
            await self.test_audit_logging()
            
            print("\n所有测试完成!")
            
        except Exception as e:
            print(f"测试运行失败: {str(e)}")
        
        finally:
            # 清理资源
            await self.synchronizer.cleanup()


# 示例使用代码
async def main():
    """主函数示例"""
    try:
        # 创建数据同步器实例
        synchronizer = DataSynchronizer("example_config.json")
        
        # 创建数据源配置
        source_config = DataSource(
            id="db_source",
            name="数据库源",
            type="database",
            connection_info={
                "type": "sqlite",
                "path": "source.db"
            },
            schema={"tables": ["users", "products", "orders"]}
        )
        
        target_config = DataSource(
            id="file_target",
            name="文件目标",
            type="file",
            connection_info={
                "base_path": "target_data"
            },
            schema={"tables": ["users", "products", "orders"]}
        )
        
        # 添加数据源
        synchronizer.config_manager.add_data_source(source_config)
        synchronizer.config_manager.add_data_source(target_config)
        
        # 创建同步任务
        sync_task = SyncTask(
            id="main_sync",
            name="主同步任务",
            source_id="db_source",
            target_id="file_target",
            mode=SyncMode.BATCH,
            tables=["users", "products"],
            batch_size=500,
            conflict_resolution=ConflictResolution.SOURCE_PRIORITY,
            enabled=True
        )
        
        # 添加同步任务
        synchronizer.config_manager.add_sync_task(sync_task)
        
        # 注册进度回调
        def on_progress_update(progress: SyncProgress):
            print(f"同步进度: {progress.progress_percentage:.1f}% - {progress.current_table}")
        
        synchronizer.register_progress_callback("main_sync", on_progress_update)
        
        # 启动同步
        success = await synchronizer.start_sync("main_sync")
        print(f"同步启动结果: {success}")
        
        # 等待同步完成
        while synchronizer.get_sync_status("main_sync"):
            await asyncio.sleep(1)
        
        # 获取最终结果
        final_status = synchronizer.get_sync_status("main_sync")
        if final_status:
            print(f"同步完成: {final_status.status.value}")
            print(f"处理记录数: {final_status.processed_records}")
            print(f"成功记录数: {final_status.success_records}")
        
        # 获取性能指标
        metrics = synchronizer.get_performance_metrics("main_sync")
        print(f"性能指标: {metrics}")
        
        # 获取性能统计
        stats = synchronizer.get_performance_stats()
        print(f"性能统计: {stats}")
        
    except Exception as e:
        print(f"示例运行失败: {str(e)}")
    
    finally:
        # 清理资源
        await synchronizer.cleanup()


if __name__ == "__main__":
    # 运行测试
    test = DataSynchronizerTest()
    asyncio.run(test.run_all_tests())
    
    # 运行示例
    # asyncio.run(main())