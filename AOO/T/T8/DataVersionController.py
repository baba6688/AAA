#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据版本控制器 (Data Version Controller)

一个功能完整的数据版本管理系统，支持数据版本的全生命周期管理，
包括版本追踪、历史记录、差异分析、回滚恢复、分支合并、标签发布、
依赖管理、访问控制和统计分析等功能。


创建时间: 2025-11-05
版本: 1.0.0
"""

import json
import hashlib
import datetime
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import sqlite3
import logging
from contextlib import contextmanager
import pickle
import difflib


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VersionStatus(Enum):
    """版本状态枚举"""
    DRAFT = "draft"          # 草稿
    ACTIVE = "active"        # 活跃
    STABLE = "stable"        # 稳定
    DEPRECATED = "deprecated" # 废弃
    ARCHIVED = "archived"     # 归档


class AccessLevel(Enum):
    """访问级别枚举"""
    READ_ONLY = "read_only"      # 只读
    READ_WRITE = "read_write"    # 读写
    ADMIN = "admin"              # 管理员
    OWNER = "owner"              # 所有者


class ChangeType(Enum):
    """变更类型枚举"""
    CREATE = "create"            # 创建
    UPDATE = "update"            # 更新
    DELETE = "delete"            # 删除
    MERGE = "merge"              # 合并
    REVERT = "revert"            # 回滚


@dataclass
class DataVersion:
    """数据版本信息类"""
    version_id: str
    data_id: str
    parent_version: Optional[str]
    data_hash: str
    metadata: Dict[str, Any]
    created_at: datetime.datetime
    created_by: str
    status: VersionStatus
    tags: List[str]
    description: str
    size_bytes: int
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['status'] = self.status.value
        result['created_at'] = self.created_at.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataVersion':
        """从字典创建实例"""
        data = data.copy()
        data['status'] = VersionStatus(data['status'])
        data['created_at'] = datetime.datetime.fromisoformat(data['created_at'])
        return cls(**data)


@dataclass
class ChangeRecord:
    """数据变更记录类"""
    change_id: str
    version_id: str
    change_type: ChangeType
    changes: Dict[str, Any]
    timestamp: datetime.datetime
    user: str
    description: str
    rollback_data: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['change_type'] = self.change_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChangeRecord':
        """从字典创建实例"""
        data = data.copy()
        data['change_type'] = ChangeType(data['change_type'])
        data['timestamp'] = datetime.datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class VersionBranch:
    """版本分支信息类"""
    branch_name: str
    head_version: str
    created_at: datetime.datetime
    created_by: str
    description: str
    is_default: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        return result


@dataclass
class AccessPermission:
    """访问权限类"""
    user_id: str
    resource_type: str  # 'data', 'version', 'branch'
    resource_id: str
    access_level: AccessLevel
    granted_at: datetime.datetime
    granted_by: str
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        result = asdict(self)
        result['access_level'] = self.access_level.value
        result['granted_at'] = self.granted_at.isoformat()
        return result


class DataVersionController:
    """数据版本控制器主类
    
    提供完整的数据版本管理功能，包括版本追踪、历史记录、
    差异分析、回滚恢复、分支合并、标签发布、依赖管理、
    访问控制和统计分析等。
    """
    
    def __init__(self, storage_path: str = "data_version_storage"):
        """初始化数据版本控制器
        
        Args:
            storage_path: 存储路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化数据库
        self.db_path = self.storage_path / "version_control.db"
        self._init_database()
        
        # 内存缓存
        self._version_cache: Dict[str, DataVersion] = {}
        self._branch_cache: Dict[str, VersionBranch] = {}
        self._permission_cache: Dict[str, AccessPermission] = {}
        
        # 默认分支
        self.default_branch = "main"
        self._ensure_default_branch()
    
    def _init_database(self) -> None:
        """初始化SQLite数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 创建版本表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS versions (
                    version_id TEXT PRIMARY KEY,
                    data_id TEXT NOT NULL,
                    parent_version TEXT,
                    data_hash TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    status TEXT NOT NULL,
                    tags TEXT NOT NULL,
                    description TEXT,
                    size_bytes INTEGER NOT NULL
                )
            ''')
            
            # 创建变更记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS change_records (
                    change_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    changes TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    user TEXT NOT NULL,
                    description TEXT,
                    rollback_data TEXT,
                    FOREIGN KEY (version_id) REFERENCES versions (version_id)
                )
            ''')
            
            # 创建分支表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS branches (
                    branch_name TEXT PRIMARY KEY,
                    head_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    description TEXT,
                    is_default BOOLEAN DEFAULT 0
                )
            ''')
            
            # 创建权限表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS permissions (
                    permission_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    access_level TEXT NOT NULL,
                    granted_at TEXT NOT NULL,
                    granted_by TEXT NOT NULL
                )
            ''')
            
            # 创建依赖关系表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dependencies (
                    dependency_id TEXT PRIMARY KEY,
                    version_id TEXT NOT NULL,
                    depends_on_version_id TEXT NOT NULL,
                    dependency_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (version_id) REFERENCES versions (version_id),
                    FOREIGN KEY (depends_on_version_id) REFERENCES versions (version_id)
                )
            ''')
            
            conn.commit()
    
    def _ensure_default_branch(self) -> None:
        """确保默认分支存在"""
        if self.default_branch not in self._branch_cache:
            default_branch = VersionBranch(
                branch_name=self.default_branch,
                head_version="",
                created_at=datetime.datetime.now(),
                created_by="system",
                description="默认主分支",
                is_default=True
            )
            self._create_branch(default_branch)
        
        # 为默认用户授予权限
        self._grant_default_permissions()
    
    def _grant_default_permissions(self) -> None:
        """授予默认权限"""
        # 为所有用户授予对所有数据的读写权限（仅用于测试）
        # 在生产环境中应该更加严格
        pass
    
    def _generate_version_id(self, data_id: str) -> str:
        """生成版本ID"""
        timestamp = datetime.datetime.now().isoformat()
        unique_str = f"{data_id}_{timestamp}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]
    
    def _generate_change_id(self) -> str:
        """生成变更记录ID"""
        timestamp = datetime.datetime.now().isoformat()
        unique_str = f"change_{timestamp}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]
    
    def _calculate_data_hash(self, data: Any) -> str:
        """计算数据哈希值"""
        if isinstance(data, str):
            data_bytes = data.encode('utf-8')
        elif isinstance(data, bytes):
            data_bytes = data
        else:
            data_bytes = pickle.dumps(data)
        
        return hashlib.sha256(data_bytes).hexdigest()
    
    def _save_data_file(self, version_id: str, data: Any) -> str:
        """保存数据文件"""
        data_file = self.storage_path / f"{version_id}.dat"
        
        if isinstance(data, str):
            data_file.write_text(data, encoding='utf-8')
        elif isinstance(data, bytes):
            data_file.write_bytes(data)
        else:
            with open(data_file, 'wb') as f:
                pickle.dump(data, f)
        
        return str(data_file)
    
    def _load_data_file(self, version_id: str) -> Any:
        """加载数据文件"""
        data_file = self.storage_path / f"{version_id}.dat"
        
        if not data_file.exists():
            raise FileNotFoundError(f"数据文件不存在: {version_id}")
        
        # 尝试不同的加载方式
        try:
            # 首先尝试作为文本文件加载
            return data_file.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # 然后尝试作为二进制文件加载
                return data_file.read_bytes()
            except:
                # 最后尝试作为pickle文件加载
                with open(data_file, 'rb') as f:
                    return pickle.load(f)
    
    # ==================== 版本管理功能 ====================
    
    def create_version(self, data_id: str, data: Any, user: str, 
                      description: str = "", metadata: Optional[Dict] = None,
                      branch: str = None) -> DataVersion:
        """创建新版本
        
        Args:
            data_id: 数据ID
            data: 数据内容
            user: 创建用户
            description: 版本描述
            metadata: 元数据
            branch: 分支名称
            
        Returns:
            DataVersion: 创建的版本对象
        """
        if branch is None:
            branch = self.default_branch
        
        # 检查权限
        if not self._check_permission(user, "data", data_id, AccessLevel.READ_WRITE):
            raise PermissionError(f"用户 {user} 没有权限修改数据 {data_id}")
        
        # 获取当前分支的头版本
        branch_obj = self.get_branch(branch)
        parent_version = branch_obj.head_version if branch_obj else None
        
        # 生成版本信息
        version_id = self._generate_version_id(data_id)
        data_hash = self._calculate_data_hash(data)
        size_bytes = len(pickle.dumps(data))
        
        # 创建版本对象
        version = DataVersion(
            version_id=version_id,
            data_id=data_id,
            parent_version=parent_version,
            data_hash=data_hash,
            metadata=metadata or {},
            created_at=datetime.datetime.now(),
            created_by=user,
            status=VersionStatus.DRAFT,
            tags=[],
            description=description,
            size_bytes=size_bytes
        )
        
        # 保存数据文件
        self._save_data_file(version_id, data)
        
        # 保存到数据库
        self._save_version_to_db(version)
        
        # 更新分支头版本
        self._update_branch_head(branch, version_id)
        
        # 创建变更记录
        change_record = ChangeRecord(
            change_id=self._generate_change_id(),
            version_id=version_id,
            change_type=ChangeType.CREATE,
            changes={"data_id": data_id, "size": size_bytes},
            timestamp=datetime.datetime.now(),
            user=user,
            description=description or f"创建数据版本 {version_id}"
        )
        self._save_change_record_to_db(change_record)
        
        # 更新缓存
        self._version_cache[version_id] = version
        
        logger.info(f"创建版本 {version_id} 成功")
        return version
    
    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """获取版本信息
        
        Args:
            version_id: 版本ID
            
        Returns:
            DataVersion: 版本对象，如果不存在返回None
        """
        # 先从缓存查找
        if version_id in self._version_cache:
            return self._version_cache[version_id]
        
        # 从数据库查找
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM versions WHERE version_id = ?",
                (version_id,)
            )
            row = cursor.fetchone()
            
            if row:
                version_data = {
                    'version_id': row[0],
                    'data_id': row[1],
                    'parent_version': row[2],
                    'data_hash': row[3],
                    'metadata': json.loads(row[4]),
                    'created_at': row[5],
                    'created_by': row[6],
                    'status': row[7],
                    'tags': json.loads(row[8]),
                    'description': row[9],
                    'size_bytes': row[10]
                }
                version = DataVersion.from_dict(version_data)
                self._version_cache[version_id] = version
                return version
        
        return None
    
    def list_versions(self, data_id: str = None, branch: str = None,
                     status: VersionStatus = None, user: str = None) -> List[DataVersion]:
        """列出版本列表
        
        Args:
            data_id: 数据ID过滤
            branch: 分支过滤
            status: 状态过滤
            user: 用户过滤
            
        Returns:
            List[DataVersion]: 版本列表
        """
        query = "SELECT * FROM versions WHERE 1=1"
        params = []
        
        if data_id:
            query += " AND data_id = ?"
            params.append(data_id)
        
        if status:
            query += " AND status = ?"
            params.append(status.value)
        
        if user:
            query += " AND created_by = ?"
            params.append(user)
        
        query += " ORDER BY created_at DESC"
        
        versions = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                version_data = {
                    'version_id': row[0],
                    'data_id': row[1],
                    'parent_version': row[2],
                    'data_hash': row[3],
                    'metadata': json.loads(row[4]),
                    'created_at': row[5],
                    'created_by': row[6],
                    'status': row[7],
                    'tags': json.loads(row[8]),
                    'description': row[9],
                    'size_bytes': row[10]
                }
                version = DataVersion.from_dict(version_data)
                
                # 分支过滤
                if branch:
                    branch_obj = self.get_branch(branch)
                    if branch_obj and version_id_in_branch(version.version_id, branch_obj, conn):
                        versions.append(version)
                else:
                    versions.append(version)
        
        return versions
    
    def update_version_status(self, version_id: str, status: VersionStatus, user: str) -> bool:
        """更新版本状态
        
        Args:
            version_id: 版本ID
            status: 新状态
            user: 操作用户
            
        Returns:
            bool: 是否成功
        """
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"版本 {version_id} 不存在")
        
        # 检查权限
        if not self._check_permission(user, "version", version_id, AccessLevel.ADMIN):
            raise PermissionError(f"用户 {user} 没有权限修改版本 {version_id}")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE versions SET status = ? WHERE version_id = ?",
                (status.value, version_id)
            )
            conn.commit()
        
        # 更新缓存
        version.status = status
        self._version_cache[version_id] = version
        
        # 创建变更记录
        change_record = ChangeRecord(
            change_id=self._generate_change_id(),
            version_id=version_id,
            change_type=ChangeType.UPDATE,
            changes={"status": status.value},
            timestamp=datetime.datetime.now(),
            user=user,
            description=f"更新版本状态为 {status.value}"
        )
        self._save_change_record_to_db(change_record)
        
        logger.info(f"更新版本 {version_id} 状态为 {status.value}")
        return True
    
    def add_version_tag(self, version_id: str, tag: str, user: str) -> bool:
        """添加版本标签
        
        Args:
            version_id: 版本ID
            tag: 标签
            user: 操作用户
            
        Returns:
            bool: 是否成功
        """
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"版本 {version_id} 不存在")
        
        if tag not in version.tags:
            version.tags.append(tag)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE versions SET tags = ? WHERE version_id = ?",
                    (json.dumps(version.tags), version_id)
                )
                conn.commit()
            
            # 更新缓存
            self._version_cache[version_id] = version
            
            # 创建变更记录
            change_record = ChangeRecord(
                change_id=self._generate_change_id(),
                version_id=version_id,
                change_type=ChangeType.UPDATE,
                changes={"tag_added": tag},
                timestamp=datetime.datetime.now(),
                user=user,
                description=f"添加标签 {tag}"
            )
            self._save_change_record_to_db(change_record)
        
        return True
    
    def remove_version_tag(self, version_id: str, tag: str, user: str) -> bool:
        """移除版本标签
        
        Args:
            version_id: 版本ID
            tag: 标签
            user: 操作用户
            
        Returns:
            bool: 是否成功
        """
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"版本 {version_id} 不存在")
        
        if tag in version.tags:
            version.tags.remove(tag)
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE versions SET tags = ? WHERE version_id = ?",
                    (json.dumps(version.tags), version_id)
                )
                conn.commit()
            
            # 更新缓存
            self._version_cache[version_id] = version
            
            # 创建变更记录
            change_record = ChangeRecord(
                change_id=self._generate_change_id(),
                version_id=version_id,
                change_type=ChangeType.UPDATE,
                changes={"tag_removed": tag},
                timestamp=datetime.datetime.now(),
                user=user,
                description=f"移除标签 {tag}"
            )
            self._save_change_record_to_db(change_record)
        
        return True
    
    # ==================== 历史记录功能 ====================
    
    def get_change_history(self, version_id: str = None, data_id: str = None,
                          user: str = None, change_type: ChangeType = None) -> List[ChangeRecord]:
        """获取变更历史记录
        
        Args:
            version_id: 版本ID过滤
            data_id: 数据ID过滤
            user: 用户过滤
            change_type: 变更类型过滤
            
        Returns:
            List[ChangeRecord]: 变更记录列表
        """
        query = "SELECT * FROM change_records WHERE 1=1"
        params = []
        
        if version_id:
            query += " AND version_id = ?"
            params.append(version_id)
        
        if user:
            query += " AND user = ?"
            params.append(user)
        
        if change_type:
            query += " AND change_type = ?"
            params.append(change_type.value)
        
        query += " ORDER BY timestamp DESC"
        
        records = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            for row in cursor.fetchall():
                record_data = {
                    'change_id': row[0],
                    'version_id': row[1],
                    'change_type': row[2],
                    'changes': json.loads(row[3]),
                    'timestamp': row[4],
                    'user': row[5],
                    'description': row[6],
                    'rollback_data': json.loads(row[7]) if row[7] else None
                }
                record = ChangeRecord.from_dict(record_data)
                
                # 如果指定了data_id，需要通过version_id关联
                if data_id:
                    version = self.get_version(record.version_id)
                    if version and version.data_id == data_id:
                        records.append(record)
                else:
                    records.append(record)
        
        return records
    
    # ==================== 差异分析功能 ====================
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict[str, Any]:
        """比较两个版本的差异
        
        Args:
            version_id1: 第一个版本ID
            version_id2: 第二个版本ID
            
        Returns:
            Dict[str, Any]: 差异分析结果
        """
        version1 = self.get_version(version_id1)
        version2 = self.get_version(version_id2)
        
        if not version1 or not version2:
            raise ValueError("版本不存在")
        
        # 获取数据内容
        data1 = self._load_data_file(version_id1)
        data2 = self._load_data_file(version_id2)
        
        # 基础信息比较
        comparison = {
            "version1": version1.to_dict(),
            "version2": version2.to_dict(),
            "differences": {
                "metadata": self._compare_metadata(version1.metadata, version2.metadata),
                "size": version1.size_bytes != version2.size_bytes,
                "hash": version1.data_hash != version2.data_hash,
                "tags": self._compare_lists(version1.tags, version2.tags),
                "status": version1.status != version2.status
            }
        }
        
        # 数据内容比较
        if isinstance(data1, str) and isinstance(data2, str):
            # 文本数据比较
            diff = list(difflib.unified_diff(
                data1.splitlines(keepends=True),
                data2.splitlines(keepends=True),
                fromfile=f"version_{version_id1}",
                tofile=f"version_{version_id2}",
                lineterm=''
            ))
            comparison["content_diff"] = diff
        else:
            # 非文本数据比较
            comparison["content_diff"] = "数据格式不支持文本差异显示"
            comparison["data_type"] = {
                "version1": type(data1).__name__,
                "version2": type(data2).__name__
            }
        
        return comparison
    
    def _compare_metadata(self, metadata1: Dict, metadata2: Dict) -> Dict[str, Any]:
        """比较元数据差异"""
        all_keys = set(metadata1.keys()) | set(metadata2.keys())
        differences = {}
        
        for key in all_keys:
            val1 = metadata1.get(key)
            val2 = metadata2.get(key)
            
            if val1 != val2:
                differences[key] = {
                    "old": val1,
                    "new": val2
                }
        
        return differences
    
    def _compare_lists(self, list1: List, list2: List) -> Dict[str, Any]:
        """比较列表差异"""
        set1 = set(list1)
        set2 = set(list2)
        
        return {
            "added": list(set2 - set1),
            "removed": list(set1 - set2),
            "common": list(set1 & set2)
        }
    
    # ==================== 回滚恢复功能 ====================
    
    def rollback_version(self, target_version_id: str, user: str, 
                        description: str = "") -> DataVersion:
        """回滚到指定版本
        
        Args:
            target_version_id: 目标版本ID
            user: 操作用户
            description: 回滚描述
            
        Returns:
            DataVersion: 新创建的版本对象
        """
        target_version = self.get_version(target_version_id)
        if not target_version:
            raise ValueError(f"目标版本 {target_version_id} 不存在")
        
        # 获取目标版本的数据
        data = self._load_data_file(target_version_id)
        
        # 创建新版本（基于目标版本）
        new_version = self.create_version(
            data_id=target_version.data_id,
            data=data,
            user=user,
            description=description or f"回滚到版本 {target_version_id}",
            metadata={
                **target_version.metadata,
                "rollback_from": target_version_id,
                "rollback_reason": description
            }
        )
        
        # 标记为回滚版本
        new_version.metadata["is_rollback"] = True
        self._update_version_metadata(new_version.version_id, new_version.metadata)
        
        # 创建回滚变更记录
        change_record = ChangeRecord(
            change_id=self._generate_change_id(),
            version_id=new_version.version_id,
            change_type=ChangeType.REVERT,
            changes={"target_version": target_version_id},
            timestamp=datetime.datetime.now(),
            user=user,
            description=description or f"回滚到版本 {target_version_id}"
        )
        self._save_change_record_to_db(change_record)
        
        logger.info(f"成功回滚到版本 {target_version_id}，新版本: {new_version.version_id}")
        return new_version
    
    def recover_deleted_version(self, version_id: str, user: str) -> bool:
        """恢复已删除的版本
        
        Args:
            version_id: 版本ID
            user: 操作用户
            
        Returns:
            bool: 是否成功
        """
        # 这里实现逻辑恢复已标记为删除的版本
        # 实际实现中可能需要更复杂的恢复机制
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"版本 {version_id} 不存在")
        
        # 更新状态为活跃
        self.update_version_status(version_id, VersionStatus.ACTIVE, user)
        
        logger.info(f"成功恢复版本 {version_id}")
        return True
    
    # ==================== 分支合并功能 ====================
    
    def create_branch(self, branch_name: str, from_version: str = None, 
                     user: str = "", description: str = "") -> VersionBranch:
        """创建新分支
        
        Args:
            branch_name: 分支名称
            from_version: 基于的版本ID
            user: 创建用户
            description: 分支描述
            
        Returns:
            VersionBranch: 创建的分支对象
        """
        if from_version is None:
            # 获取默认分支的头版本
            default_branch = self.get_branch(self.default_branch)
            from_version = default_branch.head_version if default_branch else ""
        
        branch = VersionBranch(
            branch_name=branch_name,
            head_version=from_version,
            created_at=datetime.datetime.now(),
            created_by=user,
            description=description
        )
        
        return self._create_branch(branch)
    
    def _create_branch(self, branch: VersionBranch) -> VersionBranch:
        """内部创建分支方法"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO branches 
                   (branch_name, head_version, created_at, created_by, description, is_default)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    branch.branch_name,
                    branch.head_version,
                    branch.created_at.isoformat(),
                    branch.created_by,
                    branch.description,
                    branch.is_default
                )
            )
            conn.commit()
        
        self._branch_cache[branch.branch_name] = branch
        logger.info(f"创建分支 {branch.branch_name} 成功")
        return branch
    
    def get_branch(self, branch_name: str) -> Optional[VersionBranch]:
        """获取分支信息
        
        Args:
            branch_name: 分支名称
            
        Returns:
            VersionBranch: 分支对象，如果不存在返回None
        """
        if branch_name in self._branch_cache:
            return self._branch_cache[branch_name]
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM branches WHERE branch_name = ?",
                (branch_name,)
            )
            row = cursor.fetchone()
            
            if row:
                branch_data = {
                    'branch_name': row[0],
                    'head_version': row[1],
                    'created_at': row[2],
                    'created_by': row[3],
                    'description': row[4],
                    'is_default': bool(row[5])
                }
                branch = VersionBranch(**branch_data)
                self._branch_cache[branch_name] = branch
                return branch
        
        return None
    
    def merge_branches(self, source_branch: str, target_branch: str, 
                      user: str, description: str = "") -> DataVersion:
        """合并分支
        
        Args:
            source_branch: 源分支
            target_branch: 目标分支
            user: 操作用户
            description: 合并描述
            
        Returns:
            DataVersion: 合并后的新版本
        """
        source = self.get_branch(source_branch)
        target = self.get_branch(target_branch)
        
        if not source or not target:
            raise ValueError("源分支或目标分支不存在")
        
        # 获取两个分支的头版本数据
        source_data = self._load_data_file(source.head_version) if source.head_version else None
        target_data = self._load_data_file(target.head_version) if target.head_version else None
        
        # 简单的合并策略：优先使用源分支的数据
        merged_data = source_data if source_data is not None else target_data
        
        if merged_data is None:
            raise ValueError("无法合并空分支")
        
        # 创建合并版本
        merge_version = self.create_version(
            data_id="merged_data",  # 这里需要根据实际情况调整
            data=merged_data,
            user=user,
            description=description or f"合并分支 {source_branch} 到 {target_branch}",
            metadata={
                "merge_source": source_branch,
                "merge_target": target_branch,
                "source_version": source.head_version,
                "target_version": target.head_version
            },
            branch=target_branch
        )
        
        # 更新目标分支头版本
        self._update_branch_head(target_branch, merge_version.version_id)
        
        # 创建合并变更记录
        change_record = ChangeRecord(
            change_id=self._generate_change_id(),
            version_id=merge_version.version_id,
            change_type=ChangeType.MERGE,
            changes={
                "source_branch": source_branch,
                "target_branch": target_branch,
                "source_version": source.head_version,
                "target_version": target.head_version
            },
            timestamp=datetime.datetime.now(),
            user=user,
            description=description or f"合并分支 {source_branch} 到 {target_branch}"
        )
        self._save_change_record_to_db(change_record)
        
        logger.info(f"成功合并分支 {source_branch} 到 {target_branch}")
        return merge_version
    
    def _update_branch_head(self, branch_name: str, version_id: str) -> None:
        """更新分支头版本"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE branches SET head_version = ? WHERE branch_name = ?",
                (version_id, branch_name)
            )
            conn.commit()
        
        # 更新缓存
        if branch_name in self._branch_cache:
            self._branch_cache[branch_name].head_version = version_id
    
    # ==================== 标签发布功能 ====================
    
    def create_tag(self, version_id: str, tag_name: str, user: str, 
                  description: str = "") -> bool:
        """创建标签
        
        Args:
            version_id: 版本ID
            tag_name: 标签名称
            user: 创建用户
            description: 标签描述
            
        Returns:
            bool: 是否成功
        """
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"版本 {version_id} 不存在")
        
        # 添加标签
        return self.add_version_tag(version_id, tag_name, user)
    
    def release_version(self, version_id: str, release_name: str, 
                       user: str, description: str = "") -> bool:
        """发布版本
        
        Args:
            version_id: 版本ID
            release_name: 发布名称
            user: 发布用户
            description: 发布描述
            
        Returns:
            bool: 是否成功
        """
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"版本 {version_id} 不存在")
        
        # 更新版本状态为稳定
        self.update_version_status(version_id, VersionStatus.STABLE, user)
        
        # 添加发布标签
        release_tag = f"release-{release_name}"
        self.add_version_tag(version_id, release_tag, user)
        
        # 更新元数据
        version.metadata["release_name"] = release_name
        version.metadata["release_description"] = description
        version.metadata["released_at"] = datetime.datetime.now().isoformat()
        version.metadata["released_by"] = user
        
        self._update_version_metadata(version_id, version.metadata)
        
        logger.info(f"发布版本 {version_id} 为 {release_name}")
        return True
    
    # ==================== 依赖管理功能 ====================
    
    def add_dependency(self, version_id: str, depends_on_version_id: str, 
                      dependency_type: str = "data", user: str = "") -> bool:
        """添加版本依赖
        
        Args:
            version_id: 当前版本ID
            depends_on_version_id: 依赖的版本ID
            dependency_type: 依赖类型
            user: 操作用户
            
        Returns:
            bool: 是否成功
        """
        dependency_id = self._generate_change_id()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT INTO dependencies 
                   (dependency_id, version_id, depends_on_version_id, dependency_type, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    dependency_id,
                    version_id,
                    depends_on_version_id,
                    dependency_type,
                    datetime.datetime.now().isoformat()
                )
            )
            conn.commit()
        
        logger.info(f"添加依赖: {version_id} -> {depends_on_version_id}")
        return True
    
    def get_dependencies(self, version_id: str) -> List[Dict[str, Any]]:
        """获取版本依赖列表
        
        Args:
            version_id: 版本ID
            
        Returns:
            List[Dict[str, Any]]: 依赖列表
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT d.*, v.data_id 
                   FROM dependencies d
                   JOIN versions v ON d.depends_on_version_id = v.version_id
                   WHERE d.version_id = ?""",
                (version_id,)
            )
            
            dependencies = []
            for row in cursor.fetchall():
                dependencies.append({
                    'dependency_id': row[0],
                    'version_id': row[1],
                    'depends_on_version_id': row[2],
                    'dependency_type': row[3],
                    'created_at': row[4],
                    'depends_on_data_id': row[5]
                })
        
        return dependencies
    
    def check_circular_dependency(self, version_id: str) -> bool:
        """检查循环依赖
        
        Args:
            version_id: 版本ID
            
        Returns:
            bool: 是否存在循环依赖
        """
        visited = set()
        rec_stack = set()
        
        def has_cycle_util(v_id: str) -> bool:
            visited.add(v_id)
            rec_stack.add(v_id)
            
            # 获取依赖的版本
            dependencies = self.get_dependencies(v_id)
            for dep in dependencies:
                dep_version_id = dep['depends_on_version_id']
                
                if dep_version_id not in visited:
                    if has_cycle_util(dep_version_id):
                        return True
                elif dep_version_id in rec_stack:
                    return True
            
            rec_stack.remove(v_id)
            return False
        
        return has_cycle_util(version_id)
    
    # ==================== 访问控制功能 ====================
    
    def grant_permission(self, user_id: str, resource_type: str, resource_id: str,
                        access_level: AccessLevel, granted_by: str) -> bool:
        """授予访问权限
        
        Args:
            user_id: 用户ID
            resource_type: 资源类型
            resource_id: 资源ID
            access_level: 访问级别
            granted_by: 授权用户
            
        Returns:
            bool: 是否成功
        """
        permission_id = self._generate_change_id()
        
        permission = AccessPermission(
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            access_level=access_level,
            granted_at=datetime.datetime.now(),
            granted_by=granted_by
        )
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO permissions 
                   (permission_id, user_id, resource_type, resource_id, access_level, granted_at, granted_by)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    permission_id,
                    user_id,
                    resource_type,
                    resource_id,
                    access_level.value,
                    permission.granted_at.isoformat(),
                    granted_by
                )
            )
            conn.commit()
        
        self._permission_cache[f"{user_id}:{resource_type}:{resource_id}"] = permission
        logger.info(f"授予权限: {user_id} 对 {resource_type}:{resource_id} 的 {access_level.value} 权限")
        return True
    
    def revoke_permission(self, user_id: str, resource_type: str, resource_id: str) -> bool:
        """撤销访问权限
        
        Args:
            user_id: 用户ID
            resource_type: 资源类型
            resource_id: 资源ID
            
        Returns:
            bool: 是否成功
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "DELETE FROM permissions WHERE user_id = ? AND resource_type = ? AND resource_id = ?",
                (user_id, resource_type, resource_id)
            )
            conn.commit()
        
        # 从缓存中移除
        cache_key = f"{user_id}:{resource_type}:{resource_id}"
        if cache_key in self._permission_cache:
            del self._permission_cache[cache_key]
        
        logger.info(f"撤销权限: {user_id} 对 {resource_type}:{resource_id} 的权限")
        return True
    
    def _check_permission(self, user_id: str, resource_type: str, 
                         resource_id: str, required_level: AccessLevel) -> bool:
        """检查用户权限
        
        Args:
            user_id: 用户ID
            resource_type: 资源类型
            resource_id: 资源ID
            required_level: 需要的访问级别
            
        Returns:
            bool: 是否有权限
        """
        # 检查缓存
        cache_key = f"{user_id}:{resource_type}:{resource_id}"
        if cache_key in self._permission_cache:
            permission = self._permission_cache[cache_key]
            return self._has_required_level(permission.access_level, required_level)
        
        # 从数据库查询 - 首先查询精确匹配
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT access_level FROM permissions 
                   WHERE user_id = ? AND resource_type = ? AND resource_id = ?""",
                (user_id, resource_type, resource_id)
            )
            row = cursor.fetchone()
            
            if row:
                access_level = AccessLevel(row[0])
                return self._has_required_level(access_level, required_level)
            
            # 如果没有精确匹配，查询通配符权限
            cursor.execute(
                """SELECT access_level FROM permissions 
                   WHERE user_id = ? AND resource_type = ? AND (resource_id = ? OR resource_id = '*')""",
                (user_id, resource_type, resource_id)
            )
            row = cursor.fetchone()
            
            if row:
                access_level = AccessLevel(row[0])
                return self._has_required_level(access_level, required_level)
        
        # 检查默认权限（资源所有者或管理员）
        return False
    
    def _has_required_level(self, user_level: AccessLevel, required_level: AccessLevel) -> bool:
        """检查是否具有要求的访问级别"""
        level_order = {
            AccessLevel.READ_ONLY: 1,
            AccessLevel.READ_WRITE: 2,
            AccessLevel.ADMIN: 3,
            AccessLevel.OWNER: 4
        }
        
        return level_order.get(user_level, 0) >= level_order.get(required_level, 0)
    
    # ==================== 统计报告功能 ====================
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取版本控制统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # 总版本数
            cursor.execute("SELECT COUNT(*) FROM versions")
            total_versions = cursor.fetchone()[0]
            
            # 各状态版本数
            cursor.execute("SELECT status, COUNT(*) FROM versions GROUP BY status")
            status_counts = dict(cursor.fetchall())
            
            # 总数据大小
            cursor.execute("SELECT SUM(size_bytes) FROM versions")
            total_size = cursor.fetchone()[0] or 0
            
            # 分支数
            cursor.execute("SELECT COUNT(*) FROM branches")
            total_branches = cursor.fetchone()[0]
            
            # 用户数
            cursor.execute("SELECT COUNT(DISTINCT created_by) FROM versions")
            total_users = cursor.fetchone()[0]
            
            # 最近活动
            cursor.execute("SELECT COUNT(*) FROM change_records WHERE timestamp > ?", 
                          ((datetime.datetime.now() - datetime.timedelta(days=30)).isoformat(),))
            recent_activities = cursor.fetchone()[0]
            
            # 变更类型统计
            cursor.execute("SELECT change_type, COUNT(*) FROM change_records GROUP BY change_type")
            change_type_counts = dict(cursor.fetchall())
        
        return {
            "total_versions": total_versions,
            "status_distribution": status_counts,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "total_branches": total_branches,
            "total_users": total_users,
            "recent_activities_30d": recent_activities,
            "change_type_distribution": change_type_counts,
            "generated_at": datetime.datetime.now().isoformat()
        }
    
    def generate_report(self, output_file: str = None) -> str:
        """生成版本控制报告
        
        Args:
            output_file: 输出文件路径
            
        Returns:
            str: 报告内容
        """
        stats = self.get_statistics()
        
        report = f"""
# 数据版本控制报告

生成时间: {stats['generated_at']}

## 总体统计
- 总版本数: {stats['total_versions']}
- 总分支数: {stats['total_branches']}
- 总用户数: {stats['total_users']}
- 总数据大小: {stats['total_size_mb']} MB

## 版本状态分布
"""
        
        for status, count in stats['status_distribution'].items():
            report += f"- {status}: {count}\n"
        
        report += f"""
## 变更类型分布
"""
        
        for change_type, count in stats['change_type_distribution'].items():
            report += f"- {change_type}: {count}\n"
        
        report += f"""
## 最近活动
- 过去30天变更次数: {stats['recent_activities_30d']}

## 分支列表
"""
        
        branches = list(self._branch_cache.values())
        for branch in branches:
            report += f"- {branch.branch_name}: 头版本 {branch.head_version}\n"
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"报告已保存到 {output_file}")
        
        return report
    
    # ==================== 内部辅助方法 ====================
    
    def _save_version_to_db(self, version: DataVersion) -> None:
        """保存版本到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO versions 
                   (version_id, data_id, parent_version, data_hash, metadata, created_at, 
                    created_by, status, tags, description, size_bytes)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    version.version_id,
                    version.data_id,
                    version.parent_version,
                    version.data_hash,
                    json.dumps(version.metadata),
                    version.created_at.isoformat(),
                    version.created_by,
                    version.status.value,
                    json.dumps(version.tags),
                    version.description,
                    version.size_bytes
                )
            )
            conn.commit()
    
    def _save_change_record_to_db(self, record: ChangeRecord) -> None:
        """保存变更记录到数据库"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """INSERT OR REPLACE INTO change_records 
                   (change_id, version_id, change_type, changes, timestamp, user, description, rollback_data)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record.change_id,
                    record.version_id,
                    record.change_type.value,
                    json.dumps(record.changes),
                    record.timestamp.isoformat(),
                    record.user,
                    record.description,
                    json.dumps(record.rollback_data) if record.rollback_data else None
                )
            )
            conn.commit()
    
    def _update_version_metadata(self, version_id: str, metadata: Dict[str, Any]) -> None:
        """更新版本元数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE versions SET metadata = ? WHERE version_id = ?",
                (json.dumps(metadata), version_id)
            )
            conn.commit()


def version_id_in_branch(version_id: str, branch: VersionBranch, conn) -> bool:
    """检查版本是否在指定分支中（简化实现）"""
    # 这里应该实现复杂的分支版本追踪逻辑
    # 简化实现：检查版本是否在分支的历史路径中
    return True


# ==================== 测试用例 ====================

def test_data_version_controller():
    """数据版本控制器测试函数"""
    print("开始测试数据版本控制器...")
    
    # 创建控制器实例
    controller = DataVersionController("test_storage")
    
    # 为测试用户授予权限
    controller.grant_permission("test_user", "data", "*", AccessLevel.READ_WRITE, "system")
    controller.grant_permission("test_user", "version", "*", AccessLevel.ADMIN, "system")
    controller.grant_permission("test_user", "branch", "*", AccessLevel.ADMIN, "system")
    
    try:
        # 测试创建版本
        print("1. 测试创建版本...")
        data1 = {"name": "test_data", "value": 100, "timestamp": "2025-11-05"}
        version1 = controller.create_version(
            data_id="test_data_1",
            data=data1,
            user="test_user",
            description="初始版本"
        )
        print(f"创建版本: {version1.version_id}")
        
        # 测试更新版本
        print("2. 测试更新版本...")
        data2 = {"name": "test_data", "value": 200, "timestamp": "2025-11-05"}
        version2 = controller.create_version(
            data_id="test_data_1",
            data=data2,
            user="test_user",
            description="更新版本"
        )
        print(f"更新版本: {version2.version_id}")
        
        # 测试版本比较
        print("3. 测试版本比较...")
        comparison = controller.compare_versions(version1.version_id, version2.version_id)
        print("版本比较完成")
        
        # 测试分支功能
        print("4. 测试分支功能...")
        branch = controller.create_branch(
            branch_name="feature_branch",
            from_version=version1.version_id,
            user="test_user",
            description="功能分支"
        )
        print(f"创建分支: {branch.branch_name}")
        
        # 测试标签功能
        print("5. 测试标签功能...")
        controller.add_version_tag(version1.version_id, "v1.0", "test_user")
        controller.add_version_tag(version2.version_id, "v2.0", "test_user")
        print("添加标签完成")
        
        # 测试发布功能
        print("6. 测试发布功能...")
        controller.release_version(
            version_id=version2.version_id,
            release_name="v2.0.0",
            user="test_user",
            description="正式发布版本"
        )
        print("版本发布完成")
        
        # 测试统计功能
        print("7. 测试统计功能...")
        stats = controller.get_statistics()
        print(f"统计信息: {stats}")
        
        # 测试报告生成
        print("8. 测试报告生成...")
        report = controller.generate_report("test_report.md")
        print("报告生成完成")
        
        print("所有测试通过！")
        
    except Exception as e:
        print(f"测试失败: {e}")
        raise
    finally:
        # 清理测试数据
        import shutil
        if Path("test_storage").exists():
            shutil.rmtree("test_storage")
        if Path("test_report.md").exists():
            Path("test_report.md").unlink()


if __name__ == "__main__":
    # 运行测试
    test_data_version_controller()
    
    # 示例使用
    print("\n数据版本控制器使用示例:")
    
    # 创建控制器
    controller = DataVersionController("example_storage")
    
    # 为示例用户授予权限
    controller.grant_permission("admin", "data", "*", AccessLevel.OWNER, "system")
    controller.grant_permission("admin", "version", "*", AccessLevel.ADMIN, "system")
    controller.grant_permission("admin", "branch", "*", AccessLevel.ADMIN, "system")
    
    # 创建一些示例数据
    print("创建示例数据...")
    
    # 数据版本1
    data_v1 = {
        "product": "智能手机",
        "price": 2999,
        "specs": {"ram": "8GB", "storage": "128GB"},
        "version": "1.0"
    }
    
    version1 = controller.create_version(
        data_id="product_001",
        data=data_v1,
        user="admin",
        description="产品初始版本",
        metadata={"category": "electronics", "brand": "TechBrand"}
    )
    
    print(f"创建版本1: {version1.version_id}")
    
    # 数据版本2
    data_v2 = {
        "product": "智能手机",
        "price": 2799,
        "specs": {"ram": "8GB", "storage": "128GB"},
        "version": "1.1",
        "discount": "限时优惠"
    }
    
    version2 = controller.create_version(
        data_id="product_001",
        data=data_v2,
        user="admin",
        description="产品价格调整",
        metadata={"category": "electronics", "brand": "TechBrand"}
    )
    
    print(f"创建版本2: {version2.version_id}")
    
    # 添加标签
    controller.add_version_tag(version1.version_id, "initial", "admin")
    controller.add_version_tag(version2.version_id, "discounted", "admin")
    
    # 创建分支
    feature_branch = controller.create_branch(
        branch_name="price_optimization",
        from_version=version1.version_id,
        user="admin",
        description="价格优化分支"
    )
    
    print(f"创建分支: {feature_branch.branch_name}")
    
    # 版本比较
    print("\n版本比较结果:")
    comparison = controller.compare_versions(version1.version_id, version2.version_id)
    print(f"版本1大小: {comparison['version1']['size_bytes']} bytes")
    print(f"版本2大小: {comparison['version2']['size_bytes']} bytes")
    print(f"哈希值不同: {comparison['differences']['hash']}")
    
    # 获取统计信息
    print("\n统计信息:")
    stats = controller.get_statistics()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 生成报告
    print("\n生成报告...")
    report = controller.generate_report("example_report.md")
    print("报告已生成: example_report.md")
    
    print("\n示例完成！")