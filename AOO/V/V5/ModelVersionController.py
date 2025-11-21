"""
V5模型版本控制器
==================

这个模块实现了完整的模型版本控制系统，用于管理机器学习模型的版本控制、
元数据存储、分支管理、变更历史追踪等功能。

主要功能:
- 模型版本管理和追踪
- 模型元数据存储
- 版本分支和合并
- 模型变更历史
- 版本比较和差异分析
- 模型回滚和恢复
- 版本发布和标签
- 版本依赖管理
- 版本统计和报告


版本: V5.0
创建时间: 2025-11-05
"""

import json
import sqlite3
import hashlib
import datetime
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging
from contextlib import contextmanager


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VersionStatus(Enum):
    """版本状态枚举"""
    DRAFT = "draft"           # 草稿
    DEVELOPMENT = "development"  # 开发中
    TESTING = "testing"       # 测试中
    STABLE = "stable"         # 稳定版
    DEPRECATED = "deprecated"  # 已废弃
    ARCHIVED = "archived"     # 已归档


class MergeStrategy(Enum):
    """合并策略枚举"""
    MANUAL = "manual"         # 手动合并
    AUTO = "auto"            # 自动合并
    CONFLICT = "conflict"    # 冲突


@dataclass
class ModelMetadata:
    """模型元数据类"""
    model_id: str
    name: str
    version: str
    description: str
    author: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    status: VersionStatus
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    dependencies: List[str]
    tags: List[str]
    parent_version: Optional[str] = None
    branch: str = "main"
    size_bytes: int = 0
    hash_sha256: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """从字典创建实例"""
        data['status'] = VersionStatus(data['status'])
        data['created_at'] = datetime.datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.datetime.fromisoformat(data['updated_at'])
        return cls(**data)


@dataclass
class ChangeRecord:
    """变更记录类"""
    change_id: str
    model_id: str
    version: str
    author: str
    timestamp: datetime.datetime
    change_type: str  # 'add', 'modify', 'delete', 'merge'
    description: str
    changes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChangeRecord':
        """从字典创建实例"""
        data['timestamp'] = datetime.datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class VersionComparison:
    """版本比较结果类"""
    version1: str
    version2: str
    added_parameters: Dict[str, Any]
    removed_parameters: Dict[str, Any]
    modified_parameters: Dict[str, Tuple[Any, Any]]
    metrics_changes: Dict[str, Tuple[float, float]]
    compatibility_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


class ModelVersionController:
    """
    V5模型版本控制器
    
    提供完整的模型版本管理功能，包括版本控制、元数据管理、
    分支管理、变更追踪、版本比较等功能。
    """
    
    def __init__(self, storage_path: str = "model_versions.db"):
        """
        初始化模型版本控制器
        
        Args:
            storage_path: SQLite数据库文件路径
        """
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
        logger.info(f"模型版本控制器初始化完成，存储路径: {storage_path}")
    
    def _init_database(self):
        """初始化数据库表结构"""
        with self._get_db_connection() as conn:
            cursor = conn.cursor()
            
            # 创建模型元数据表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_metadata (
                    model_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT,
                    author TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    status TEXT NOT NULL,
                    parameters TEXT,
                    metrics TEXT,
                    dependencies TEXT,
                    tags TEXT,
                    parent_version TEXT,
                    branch TEXT NOT NULL,
                    size_bytes INTEGER DEFAULT 0,
                    hash_sha256 TEXT,
                    PRIMARY KEY (model_id, version)
                )
            ''')
            
            # 创建变更历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS change_history (
                    change_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    author TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    description TEXT,
                    changes TEXT,
                    FOREIGN KEY (model_id, version) REFERENCES model_metadata (model_id, version)
                )
            ''')
            
            # 创建分支信息表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS branches (
                    branch_name TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    head_version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    description TEXT,
                    FOREIGN KEY (model_id, head_version) REFERENCES model_metadata (model_id, version)
                )
            ''')
            
            # 创建标签表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tags (
                    tag_name TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    description TEXT,
                    FOREIGN KEY (model_id, version) REFERENCES model_metadata (model_id, version)
                )
            ''')
            
            # 创建依赖关系表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS dependencies (
                    dependency_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    depends_on_model_id TEXT NOT NULL,
                    depends_on_version TEXT NOT NULL,
                    dependency_type TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (model_id, version) REFERENCES model_metadata (model_id, version)
                )
            ''')
            
            conn.commit()
            logger.info("数据库表结构初始化完成")
    
    @contextmanager
    def _get_db_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = sqlite3.connect(self.storage_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def create_model_version(self, 
                           model_id: str,
                           name: str,
                           version: str,
                           description: str,
                           author: str,
                           parameters: Dict[str, Any],
                           metrics: Dict[str, float],
                           dependencies: List[str] = None,
                           tags: List[str] = None,
                           parent_version: str = None,
                           branch: str = "main",
                           status: VersionStatus = VersionStatus.DRAFT) -> bool:
        """
        创建新的模型版本
        
        Args:
            model_id: 模型ID
            name: 模型名称
            version: 版本号
            description: 版本描述
            author: 作者
            parameters: 模型参数
            metrics: 模型指标
            dependencies: 依赖列表
            tags: 标签列表
            parent_version: 父版本
            branch: 分支名称
            status: 版本状态
            
        Returns:
            bool: 创建是否成功
        """
        try:
            now = datetime.datetime.now()
            
            # 计算模型大小和哈希值
            model_content = json.dumps(parameters, sort_keys=True)
            size_bytes = len(model_content.encode('utf-8'))
            hash_sha256 = hashlib.sha256(model_content.encode('utf-8')).hexdigest()
            
            metadata = ModelMetadata(
                model_id=model_id,
                name=name,
                version=version,
                description=description,
                author=author,
                created_at=now,
                updated_at=now,
                status=status,
                parameters=parameters,
                metrics=metrics,
                dependencies=dependencies or [],
                tags=tags or [],
                parent_version=parent_version,
                branch=branch,
                size_bytes=size_bytes,
                hash_sha256=hash_sha256
            )
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 插入模型元数据
                cursor.execute('''
                    INSERT INTO model_metadata 
                    (model_id, name, version, description, author, created_at, updated_at, 
                     status, parameters, metrics, dependencies, tags, parent_version, 
                     branch, size_bytes, hash_sha256)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metadata.model_id, metadata.name, metadata.version, metadata.description,
                    metadata.author, metadata.created_at.isoformat(), metadata.updated_at.isoformat(),
                    metadata.status.value, json.dumps(metadata.parameters), 
                    json.dumps(metadata.metrics), json.dumps(metadata.dependencies),
                    json.dumps(metadata.tags), metadata.parent_version, metadata.branch,
                    metadata.size_bytes, metadata.hash_sha256
                ))
                
                # 创建变更记录
                change_record = ChangeRecord(
                    change_id=f"change_{model_id}_{version}_{now.timestamp()}",
                    model_id=model_id,
                    version=version,
                    author=author,
                    timestamp=now,
                    change_type="add",
                    description=f"创建模型版本 {version}",
                    changes={"action": "create", "parameters": parameters, "metrics": metrics}
                )
                
                cursor.execute('''
                    INSERT INTO change_history 
                    (change_id, model_id, version, author, timestamp, change_type, description, changes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    change_record.change_id, model_id, version, author,
                    change_record.timestamp.isoformat(), change_record.change_type,
                    change_record.description, json.dumps(change_record.changes)
                ))
                
                # 更新分支头指针
                self._update_branch_head(conn, branch, model_id, version, author)
                
                conn.commit()
                
            logger.info(f"成功创建模型版本: {model_id} v{version}")
            return True
            
        except Exception as e:
            logger.error(f"创建模型版本失败: {e}")
            return False
    
    def get_model_metadata(self, model_id: str, version: str) -> Optional[ModelMetadata]:
        """
        获取模型元数据
        
        Args:
            model_id: 模型ID
            version: 版本号
            
        Returns:
            ModelMetadata: 模型元数据对象，如果不存在则返回None
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM model_metadata WHERE model_id = ? AND version = ?
                ''', (model_id, version))
                
                row = cursor.fetchone()
                if row:
                    data = dict(row)
                    data['parameters'] = json.loads(data['parameters'])
                    data['metrics'] = json.loads(data['metrics'])
                    data['dependencies'] = json.loads(data['dependencies'])
                    data['tags'] = json.loads(data['tags'])
                    return ModelMetadata.from_dict(data)
                return None
                
        except Exception as e:
            logger.error(f"获取模型元数据失败: {e}")
            return None
    
    def list_model_versions(self, model_id: str = None, branch: str = None) -> List[ModelMetadata]:
        """
        列出模型版本
        
        Args:
            model_id: 模型ID，如果为None则列出所有模型
            branch: 分支名称，如果为None则列出所有分支
            
        Returns:
            List[ModelMetadata]: 模型元数据列表
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM model_metadata WHERE 1=1"
                params = []
                
                if model_id:
                    query += " AND model_id = ?"
                    params.append(model_id)
                
                if branch:
                    query += " AND branch = ?"
                    params.append(branch)
                
                query += " ORDER BY created_at DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                versions = []
                for row in rows:
                    data = dict(row)
                    data['parameters'] = json.loads(data['parameters'])
                    data['metrics'] = json.loads(data['metrics'])
                    data['dependencies'] = json.loads(data['dependencies'])
                    data['tags'] = json.loads(data['tags'])
                    versions.append(ModelMetadata.from_dict(data))
                
                return versions
                
        except Exception as e:
            logger.error(f"列出模型版本失败: {e}")
            return []
    
    def create_branch(self, branch_name: str, model_id: str, from_version: str, 
                     author: str, description: str = "") -> bool:
        """
        创建版本分支
        
        Args:
            branch_name: 分支名称
            model_id: 模型ID
            from_version: 基于的版本
            author: 创建者
            description: 分支描述
            
        Returns:
            bool: 创建是否成功
        """
        try:
            now = datetime.datetime.now()
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 检查分支是否已存在
                cursor.execute("SELECT * FROM branches WHERE branch_name = ?", (branch_name,))
                if cursor.fetchone():
                    logger.warning(f"分支 {branch_name} 已存在")
                    return False
                
                # 创建分支记录
                cursor.execute('''
                    INSERT INTO branches (branch_name, model_id, head_version, created_at, created_by, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (branch_name, model_id, from_version, now.isoformat(), author, description))
                
                conn.commit()
                
            logger.info(f"成功创建分支: {branch_name}")
            return True
            
        except Exception as e:
            logger.error(f"创建分支失败: {e}")
            return False
    
    def merge_branches(self, source_branch: str, target_branch: str, author: str,
                      merge_strategy: MergeStrategy = MergeStrategy.MANUAL) -> Tuple[bool, str]:
        """
        合并分支
        
        Args:
            source_branch: 源分支
            target_branch: 目标分支
            author: 执行者
            merge_strategy: 合并策略
            
        Returns:
            Tuple[bool, str]: (是否成功, 结果消息)
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 获取分支信息
                cursor.execute("SELECT * FROM branches WHERE branch_name = ?", (source_branch,))
                source_branch_info = cursor.fetchone()
                
                cursor.execute("SELECT * FROM branches WHERE branch_name = ?", (target_branch,))
                target_branch_info = cursor.fetchone()
                
                if not source_branch_info or not target_branch_info:
                    return False, "源分支或目标分支不存在"
                
                source_version = source_branch_info['head_version']
                target_version = target_branch_info['head_version']
                
                # 获取两个版本的元数据
                source_metadata = self.get_model_metadata(source_branch_info['model_id'], source_version)
                target_metadata = self.get_model_metadata(target_branch_info['model_id'], target_version)
                
                if not source_metadata or not target_metadata:
                    return False, "无法获取分支头版本信息"
                
                # 执行合并逻辑
                merged_parameters = self._merge_parameters(
                    source_metadata.parameters, target_metadata.parameters, merge_strategy
                )
                
                merged_metrics = self._merge_metrics(
                    source_metadata.metrics, target_metadata.metrics, merge_strategy
                )
                
                # 创建合并后的新版本
                new_version = self._generate_merge_version(target_metadata.version)
                
                now = datetime.datetime.now()
                
                # 插入合并后的版本
                cursor.execute('''
                    INSERT INTO model_metadata 
                    (model_id, name, version, description, author, created_at, updated_at,
                     status, parameters, metrics, dependencies, tags, parent_version, branch)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    target_metadata.model_id, target_metadata.name, new_version,
                    f"合并分支 {source_branch} 到 {target_branch}", author, now.isoformat(),
                    now.isoformat(), VersionStatus.STABLE.value,
                    json.dumps(merged_parameters), json.dumps(merged_metrics),
                    json.dumps(target_metadata.dependencies), json.dumps(target_metadata.tags),
                    target_version, target_branch
                ))
                
                # 更新目标分支头指针
                self._update_branch_head(conn, target_branch, target_metadata.model_id, new_version, author)
                
                # 记录变更历史
                change_record = ChangeRecord(
                    change_id=f"merge_{source_branch}_{target_branch}_{now.timestamp()}",
                    model_id=target_metadata.model_id,
                    version=new_version,
                    author=author,
                    timestamp=now,
                    change_type="merge",
                    description=f"合并分支 {source_branch} 到 {target_branch}",
                    changes={
                        "source_branch": source_branch,
                        "target_branch": target_branch,
                        "source_version": source_version,
                        "target_version": target_version,
                        "strategy": merge_strategy.value
                    }
                )
                
                cursor.execute('''
                    INSERT INTO change_history 
                    (change_id, model_id, version, author, timestamp, change_type, description, changes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    change_record.change_id, target_metadata.model_id, new_version, author,
                    change_record.timestamp.isoformat(), change_record.change_type,
                    change_record.description, json.dumps(change_record.changes)
                ))
                
                conn.commit()
                
            logger.info(f"成功合并分支 {source_branch} 到 {target_branch}")
            return True, f"成功合并分支，创建新版本 {new_version}"
            
        except Exception as e:
            logger.error(f"合并分支失败: {e}")
            return False, f"合并失败: {e}"
    
    def _merge_parameters(self, source_params: Dict[str, Any], 
                         target_params: Dict[str, Any], 
                         strategy: MergeStrategy) -> Dict[str, Any]:
        """合并参数"""
        if strategy == MergeStrategy.AUTO:
            # 简单策略：目标分支优先，源分支的新参数添加到目标分支
            merged = target_params.copy()
            for key, value in source_params.items():
                if key not in merged:
                    merged[key] = value
            return merged
        else:
            # 手动合并需要用户干预，这里返回目标分支参数
            return target_params.copy()
    
    def _merge_metrics(self, source_metrics: Dict[str, float], 
                      target_metrics: Dict[str, float], 
                      strategy: MergeStrategy) -> Dict[str, float]:
        """合并指标"""
        if strategy == MergeStrategy.AUTO:
            # 简单策略：取两个指标的平均值
            merged = {}
            all_keys = set(source_metrics.keys()) | set(target_metrics.keys())
            
            for key in all_keys:
                if key in source_metrics and key in target_metrics:
                    merged[key] = (source_metrics[key] + target_metrics[key]) / 2
                elif key in source_metrics:
                    merged[key] = source_metrics[key]
                else:
                    merged[key] = target_metrics[key]
            
            return merged
        else:
            return target_metrics.copy()
    
    def _generate_merge_version(self, base_version: str) -> str:
        """生成合并后的版本号"""
        # 简单的版本号生成策略：在基础版本后添加 .merge.时间戳
        timestamp = int(datetime.datetime.now().timestamp())
        return f"{base_version}.merge.{timestamp}"
    
    def _update_branch_head(self, conn, branch_name: str, model_id: str, 
                           version: str, author: str):
        """更新分支头指针"""
        cursor = conn.cursor()
        now = datetime.datetime.now().isoformat()
        
        cursor.execute('''
            UPDATE branches SET head_version = ?, model_id = ? WHERE branch_name = ?
        ''', (version, model_id, branch_name))
        
        if cursor.rowcount == 0:
            # 如果分支不存在，创建新分支
            cursor.execute('''
                INSERT INTO branches (branch_name, model_id, head_version, created_at, created_by, description)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (branch_name, model_id, version, now, author, "自动创建"))
    
    def get_change_history(self, model_id: str, version: str = None, 
                          limit: int = 50) -> List[ChangeRecord]:
        """
        获取变更历史
        
        Args:
            model_id: 模型ID
            version: 版本号，如果为None则获取所有版本的历史
            limit: 限制返回数量
            
        Returns:
            List[ChangeRecord]: 变更记录列表
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                if version:
                    cursor.execute('''
                        SELECT * FROM change_history 
                        WHERE model_id = ? AND version = ?
                        ORDER BY timestamp DESC LIMIT ?
                    ''', (model_id, version, limit))
                else:
                    cursor.execute('''
                        SELECT * FROM change_history 
                        WHERE model_id = ?
                        ORDER BY timestamp DESC LIMIT ?
                    ''', (model_id, limit))
                
                rows = cursor.fetchall()
                
                history = []
                for row in rows:
                    data = dict(row)
                    data['changes'] = json.loads(data['changes'])
                    history.append(ChangeRecord.from_dict(data))
                
                return history
                
        except Exception as e:
            logger.error(f"获取变更历史失败: {e}")
            return []
    
    def compare_versions(self, model_id: str, version1: str, version2: str) -> VersionComparison:
        """
        比较两个版本
        
        Args:
            model_id: 模型ID
            version1: 版本1
            version2: 版本2
            
        Returns:
            VersionComparison: 比较结果
        """
        try:
            metadata1 = self.get_model_metadata(model_id, version1)
            metadata2 = self.get_model_metadata(model_id, version2)
            
            if not metadata1 or not metadata2:
                raise ValueError("版本不存在")
            
            # 比较参数
            params1 = metadata1.parameters
            params2 = metadata2.parameters
            
            added_params = {k: v for k, v in params2.items() if k not in params1}
            removed_params = {k: v for k, v in params1.items() if k not in params2}
            modified_params = {}
            
            for k in set(params1.keys()) & set(params2.keys()):
                if params1[k] != params2[k]:
                    modified_params[k] = (params1[k], params2[k])
            
            # 比较指标
            metrics1 = metadata1.metrics
            metrics2 = metadata2.metrics
            
            metrics_changes = {}
            all_metric_keys = set(metrics1.keys()) | set(metrics2.keys())
            
            for key in all_metric_keys:
                val1 = metrics1.get(key, 0.0)
                val2 = metrics2.get(key, 0.0)
                if val1 != val2:
                    metrics_changes[key] = (val1, val2)
            
            # 计算兼容性分数
            compatibility_score = self._calculate_compatibility_score(
                added_params, removed_params, modified_params, metrics_changes
            )
            
            return VersionComparison(
                version1=version1,
                version2=version2,
                added_parameters=added_params,
                removed_parameters=removed_params,
                modified_parameters=modified_params,
                metrics_changes=metrics_changes,
                compatibility_score=compatibility_score
            )
            
        except Exception as e:
            logger.error(f"比较版本失败: {e}")
            raise
    
    def _calculate_compatibility_score(self, added_params: Dict[str, Any],
                                     removed_params: Dict[str, Any],
                                     modified_params: Dict[str, Tuple[Any, Any]],
                                     metrics_changes: Dict[str, Tuple[float, float]]) -> float:
        """计算兼容性分数"""
        base_score = 1.0
        
        # 参数移除惩罚
        removal_penalty = len(removed_params) * 0.1
        base_score -= removal_penalty
        
        # 参数修改惩罚
        modification_penalty = len(modified_params) * 0.05
        base_score -= modification_penalty
        
        # 参数添加奖励
        addition_bonus = len(added_params) * 0.02
        base_score += addition_bonus
        
        # 指标改善奖励
        improvement_bonus = 0
        for old_val, new_val in metrics_changes.values():
            if new_val > old_val:
                improvement_bonus += 0.01
        
        base_score += improvement_bonus
        
        return max(0.0, min(1.0, base_score))
    
    def rollback_version(self, model_id: str, target_version: str, author: str,
                        reason: str = "") -> bool:
        """
        回滚到指定版本
        
        Args:
            model_id: 模型ID
            target_version: 目标版本
            author: 执行者
            reason: 回滚原因
            
        Returns:
            bool: 是否成功
        """
        try:
            target_metadata = self.get_model_metadata(model_id, target_version)
            if not target_metadata:
                logger.error(f"目标版本 {target_version} 不存在")
                return False
            
            # 创建新的回滚版本
            rollback_version = f"{target_version}.rollback.{int(datetime.datetime.now().timestamp())}"
            
            now = datetime.datetime.now()
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 插入回滚版本
                cursor.execute('''
                    INSERT INTO model_metadata 
                    (model_id, name, version, description, author, created_at, updated_at,
                     status, parameters, metrics, dependencies, tags, parent_version, branch)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_id, target_metadata.name, rollback_version,
                    f"回滚到版本 {target_version}: {reason}", author, now.isoformat(),
                    now.isoformat(), VersionStatus.STABLE.value,
                    json.dumps(target_metadata.parameters), json.dumps(target_metadata.metrics),
                    json.dumps(target_metadata.dependencies), json.dumps(target_metadata.tags),
                    target_version, target_metadata.branch
                ))
                
                # 更新分支头指针
                self._update_branch_head(conn, target_metadata.branch, model_id, rollback_version, author)
                
                # 记录变更历史
                change_record = ChangeRecord(
                    change_id=f"rollback_{model_id}_{target_version}_{now.timestamp()}",
                    model_id=model_id,
                    version=rollback_version,
                    author=author,
                    timestamp=now,
                    change_type="rollback",
                    description=f"回滚到版本 {target_version}",
                    changes={
                        "target_version": target_version,
                        "reason": reason,
                        "original_parameters": target_metadata.parameters,
                        "original_metrics": target_metadata.metrics
                    }
                )
                
                cursor.execute('''
                    INSERT INTO change_history 
                    (change_id, model_id, version, author, timestamp, change_type, description, changes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    change_record.change_id, model_id, rollback_version, author,
                    change_record.timestamp.isoformat(), change_record.change_type,
                    change_record.description, json.dumps(change_record.changes)
                ))
                
                conn.commit()
                
            logger.info(f"成功回滚到版本 {target_version}")
            return True
            
        except Exception as e:
            logger.error(f"回滚版本失败: {e}")
            return False
    
    def create_tag(self, tag_name: str, model_id: str, version: str, 
                  author: str, description: str = "") -> bool:
        """
        创建版本标签
        
        Args:
            tag_name: 标签名称
            model_id: 模型ID
            version: 版本号
            author: 创建者
            description: 标签描述
            
        Returns:
            bool: 是否成功
        """
        try:
            now = datetime.datetime.now()
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 检查标签是否已存在
                cursor.execute("SELECT * FROM tags WHERE tag_name = ?", (tag_name,))
                if cursor.fetchone():
                    logger.warning(f"标签 {tag_name} 已存在")
                    return False
                
                # 创建标签
                cursor.execute('''
                    INSERT INTO tags (tag_name, model_id, version, created_at, created_by, description)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (tag_name, model_id, version, now.isoformat(), author, description))
                
                conn.commit()
                
            logger.info(f"成功创建标签: {tag_name}")
            return True
            
        except Exception as e:
            logger.error(f"创建标签失败: {e}")
            return False
    
    def get_tag(self, tag_name: str) -> Optional[Dict[str, Any]]:
        """
        获取标签信息
        
        Args:
            tag_name: 标签名称
            
        Returns:
            Dict[str, Any]: 标签信息，如果不存在则返回None
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM tags WHERE tag_name = ?
                ''', (tag_name,))
                
                row = cursor.fetchone()
                if row:
                    return dict(row)
                return None
                
        except Exception as e:
            logger.error(f"获取标签信息失败: {e}")
            return None
    
    def list_tags(self, model_id: str = None) -> List[Dict[str, Any]]:
        """
        列出标签
        
        Args:
            model_id: 模型ID，如果为None则列出所有标签
            
        Returns:
            List[Dict[str, Any]]: 标签信息列表
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                if model_id:
                    cursor.execute('''
                        SELECT * FROM tags WHERE model_id = ? ORDER BY created_at DESC
                    ''', (model_id,))
                else:
                    cursor.execute('''
                        SELECT * FROM tags ORDER BY created_at DESC
                    ''')
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"列出标签失败: {e}")
            return []
    
    def add_dependency(self, model_id: str, version: str, depends_on_model_id: str,
                      depends_on_version: str, dependency_type: str = "model") -> bool:
        """
        添加版本依赖
        
        Args:
            model_id: 模型ID
            version: 版本号
            depends_on_model_id: 依赖的模型ID
            depends_on_version: 依赖的版本号
            dependency_type: 依赖类型
            
        Returns:
            bool: 是否成功
        """
        try:
            dependency_id = f"{model_id}_{version}_{depends_on_model_id}_{depends_on_version}"
            now = datetime.datetime.now()
            
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 检查依赖是否已存在
                cursor.execute('''
                    SELECT * FROM dependencies WHERE dependency_id = ?
                ''', (dependency_id,))
                
                if cursor.fetchone():
                    logger.warning(f"依赖关系已存在: {dependency_id}")
                    return False
                
                # 添加依赖
                cursor.execute('''
                    INSERT INTO dependencies 
                    (dependency_id, model_id, version, depends_on_model_id, depends_on_version, dependency_type, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (dependency_id, model_id, version, depends_on_model_id, 
                     depends_on_version, dependency_type, now.isoformat()))
                
                conn.commit()
                
            logger.info(f"成功添加依赖: {model_id} v{version} -> {depends_on_model_id} v{depends_on_version}")
            return True
            
        except Exception as e:
            logger.error(f"添加依赖失败: {e}")
            return False
    
    def get_dependencies(self, model_id: str, version: str) -> List[Dict[str, Any]]:
        """
        获取版本依赖
        
        Args:
            model_id: 模型ID
            version: 版本号
            
        Returns:
            List[Dict[str, Any]]: 依赖信息列表
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM dependencies WHERE model_id = ? AND version = ?
                ''', (model_id, version))
                
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
                
        except Exception as e:
            logger.error(f"获取依赖失败: {e}")
            return []
    
    def get_version_statistics(self, model_id: str = None) -> Dict[str, Any]:
        """
        获取版本统计信息
        
        Args:
            model_id: 模型ID，如果为None则统计所有模型
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                # 基础统计
                if model_id:
                    cursor.execute('''
                        SELECT COUNT(*) as total_versions,
                               COUNT(DISTINCT branch) as total_branches,
                               AVG(size_bytes) as avg_size,
                               MAX(size_bytes) as max_size,
                               MIN(size_bytes) as min_size
                        FROM model_metadata WHERE model_id = ?
                    ''', (model_id,))
                else:
                    cursor.execute('''
                        SELECT COUNT(*) as total_versions,
                               COUNT(DISTINCT branch) as total_branches,
                               AVG(size_bytes) as avg_size,
                               MAX(size_bytes) as max_size,
                               MIN(size_bytes) as min_size
                        FROM model_metadata
                    ''')
                
                basic_stats = dict(cursor.fetchone())
                
                # 状态分布
                if model_id:
                    cursor.execute('''
                        SELECT status, COUNT(*) as count
                        FROM model_metadata WHERE model_id = ?
                        GROUP BY status
                    ''', (model_id,))
                else:
                    cursor.execute('''
                        SELECT status, COUNT(*) as count
                        FROM model_metadata
                        GROUP BY status
                    ''')
                
                status_distribution = {row['status']: row['count'] for row in cursor.fetchall()}
                
                # 分支统计
                if model_id:
                    cursor.execute('''
                        SELECT branch, COUNT(*) as version_count
                        FROM model_metadata WHERE model_id = ?
                        GROUP BY branch
                    ''', (model_id,))
                else:
                    cursor.execute('''
                        SELECT branch, COUNT(*) as version_count
                        FROM model_metadata
                        GROUP BY branch
                    ''')
                
                branch_stats = {row['branch']: row['version_count'] for row in cursor.fetchall()}
                
                # 最近活动
                if model_id:
                    cursor.execute('''
                        SELECT version, updated_at, author
                        FROM model_metadata WHERE model_id = ?
                        ORDER BY updated_at DESC LIMIT 10
                    ''', (model_id,))
                else:
                    cursor.execute('''
                        SELECT model_id, version, updated_at, author
                        FROM model_metadata
                        ORDER BY updated_at DESC LIMIT 10
                    ''')
                
                recent_activity = [dict(row) for row in cursor.fetchall()]
                
                return {
                    "basic_statistics": basic_stats,
                    "status_distribution": status_distribution,
                    "branch_statistics": branch_stats,
                    "recent_activity": recent_activity,
                    "generated_at": datetime.datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"获取版本统计失败: {e}")
            return {}
    
    def generate_version_report(self, model_id: str, version: str) -> Dict[str, Any]:
        """
        生成版本报告
        
        Args:
            model_id: 模型ID
            version: 版本号
            
        Returns:
            Dict[str, Any]: 版本报告
        """
        try:
            metadata = self.get_model_metadata(model_id, version)
            if not metadata:
                return {"error": "版本不存在"}
            
            # 获取变更历史
            change_history = self.get_change_history(model_id, version)
            
            # 获取依赖
            dependencies = self.get_dependencies(model_id, version)
            
            # 获取标签
            tags = self.list_tags(model_id)
            version_tags = [tag for tag in tags if tag['version'] == version]
            
            # 生成报告
            report = {
                "model_info": metadata.to_dict(),
                "change_history": [record.to_dict() for record in change_history],
                "dependencies": dependencies,
                "tags": version_tags,
                "report_generated_at": datetime.datetime.now().isoformat(),
                "summary": {
                    "total_changes": len(change_history),
                    "has_dependencies": len(dependencies) > 0,
                    "has_tags": len(version_tags) > 0,
                    "is_latest": self._is_latest_version(model_id, version),
                    "compatibility_level": self._assess_compatibility_level(metadata)
                }
            }
            
            return report
            
        except Exception as e:
            logger.error(f"生成版本报告失败: {e}")
            return {"error": str(e)}
    
    def _is_latest_version(self, model_id: str, version: str) -> bool:
        """检查是否为最新版本"""
        try:
            versions = self.list_model_versions(model_id)
            if not versions:
                return False
            
            # 简单的版本比较逻辑（假设版本号格式为 x.y.z）
            latest_version = max(versions, key=lambda v: v.version)
            return latest_version.version == version
            
        except Exception:
            return False
    
    def _assess_compatibility_level(self, metadata: ModelMetadata) -> str:
        """评估兼容性等级"""
        if metadata.status == VersionStatus.STABLE:
            return "high"
        elif metadata.status == VersionStatus.TESTING:
            return "medium"
        elif metadata.status == VersionStatus.DEPRECATED:
            return "low"
        else:
            return "unknown"
    
    def export_version_data(self, model_id: str, version: str, output_path: str) -> bool:
        """
        导出版本数据
        
        Args:
            model_id: 模型ID
            version: 版本号
            output_path: 输出文件路径
            
        Returns:
            bool: 是否成功
        """
        try:
            report = self.generate_version_report(model_id, version)
            
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"成功导出版本数据到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出版本数据失败: {e}")
            return False
    
    def cleanup_old_versions(self, model_id: str, keep_count: int = 10, 
                           status_filter: List[VersionStatus] = None) -> int:
        """
        清理旧版本
        
        Args:
            model_id: 模型ID
            keep_count: 保留版本数量
            status_filter: 状态过滤器，只清理指定状态的版本
            
        Returns:
            int: 清理的版本数量
        """
        try:
            versions = self.list_model_versions(model_id)
            
            # 按创建时间排序
            versions.sort(key=lambda v: v.created_at)
            
            # 过滤要清理的版本
            versions_to_cleanup = versions[:-keep_count] if len(versions) > keep_count else []
            
            if status_filter:
                versions_to_cleanup = [
                    v for v in versions_to_cleanup 
                    if v.status in status_filter
                ]
            
            cleaned_count = 0
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                
                for version in versions_to_cleanup:
                    try:
                        # 删除相关数据
                        cursor.execute('''
                            DELETE FROM dependencies WHERE model_id = ? AND version = ?
                        ''', (model_id, version.version))
                        
                        cursor.execute('''
                            DELETE FROM change_history WHERE model_id = ? AND version = ?
                        ''', (model_id, version.version))
                        
                        cursor.execute('''
                            DELETE FROM model_metadata WHERE model_id = ? AND version = ?
                        ''', (model_id, version.version))
                        
                        cleaned_count += 1
                        
                    except Exception as e:
                        logger.warning(f"清理版本 {version.version} 时出错: {e}")
                
                conn.commit()
            
            logger.info(f"清理完成，删除了 {cleaned_count} 个旧版本")
            return cleaned_count
            
        except Exception as e:
            logger.error(f"清理旧版本失败: {e}")
            return 0


# 测试用例
def test_model_version_controller():
    """测试模型版本控制器的功能"""
    print("开始测试模型版本控制器...")
    
    # 创建控制器实例
    controller = ModelVersionController("test_versions.db")
    
    # 测试创建模型版本
    print("\n1. 测试创建模型版本")
    success = controller.create_model_version(
        model_id="test_model",
        name="测试模型",
        version="1.0.0",
        description="这是一个测试模型",
        author="测试用户",
        parameters={"learning_rate": 0.01, "batch_size": 32, "epochs": 100},
        metrics={"accuracy": 0.95, "loss": 0.05},
        tags=["test", "v1"],
        status=VersionStatus.STABLE
    )
    print(f"创建版本结果: {success}")
    
    # 测试获取模型元数据
    print("\n2. 测试获取模型元数据")
    metadata = controller.get_model_metadata("test_model", "1.0.0")
    if metadata:
        print(f"模型名称: {metadata.name}")
        print(f"版本: {metadata.version}")
        print(f"参数: {metadata.parameters}")
        print(f"指标: {metadata.metrics}")
    
    # 测试创建分支
    print("\n3. 测试创建分支")
    branch_success = controller.create_branch(
        branch_name="feature/new_algorithm",
        model_id="test_model",
        from_version="1.0.0",
        author="测试用户",
        description="新算法功能分支"
    )
    print(f"创建分支结果: {branch_success}")
    
    # 测试版本比较
    print("\n4. 测试版本比较")
    controller.create_model_version(
        model_id="test_model",
        name="测试模型",
        version="1.1.0",
        description="更新版本的模型",
        author="测试用户",
        parameters={"learning_rate": 0.02, "batch_size": 64, "epochs": 150},
        metrics={"accuracy": 0.97, "loss": 0.03},
        tags=["test", "v2"],
        status=VersionStatus.TESTING
    )
    
    comparison = controller.compare_versions("test_model", "1.0.0", "1.1.0")
    print(f"兼容性分数: {comparison.compatibility_score}")
    print(f"新增参数: {comparison.added_parameters}")
    print(f"修改参数: {list(comparison.modified_parameters.keys())}")
    
    # 测试创建标签
    print("\n5. 测试创建标签")
    tag_success = controller.create_tag(
        tag_name="v1.0.0-stable",
        model_id="test_model",
        version="1.0.0",
        author="测试用户",
        description="稳定版本标签"
    )
    print(f"创建标签结果: {tag_success}")
    
    # 测试添加依赖
    print("\n6. 测试添加依赖")
    dep_success = controller.add_dependency(
        model_id="test_model",
        version="1.0.0",
        depends_on_model_id="base_model",
        depends_on_version="2.0.0",
        dependency_type="model"
    )
    print(f"添加依赖结果: {dep_success}")
    
    # 测试版本统计
    print("\n7. 测试版本统计")
    stats = controller.get_version_statistics("test_model")
    print(f"版本统计: {stats}")
    
    # 测试生成报告
    print("\n8. 测试生成版本报告")
    report = controller.generate_version_report("test_model", "1.0.0")
    print(f"报告概要: {report['summary']}")
    
    # 测试导出数据
    print("\n9. 测试导出数据")
    export_success = controller.export_version_data("test_model", "1.0.0", "test_export.json")
    print(f"导出结果: {export_success}")
    
    print("\n测试完成!")


if __name__ == "__main__":
    # 运行测试
    test_model_version_controller()