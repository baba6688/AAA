"""
K9配置状态聚合器模块

这个模块提供了完整的配置状态聚合和管理功能，包括：
- 配置状态监控和管理
- 配置协调和同步
- 配置生命周期管理
- 配置性能统计
- 配置健康检查
- 统一配置接口和API
- 异步配置状态同步
- 配置告警和通知系统

作者: K9系统
版本: 1.0.0
日期: 2025-11-06
"""

import asyncio
import logging
import json
import time
import hashlib
import threading
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Callable, Union, Tuple, AsyncGenerator
from uuid import uuid4, UUID
import sqlite3
import copy
import re
import socket
from urllib.parse import urlparse
import ssl
import aiohttp
import websockets
import yaml
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import psutil


# ==================== 基础异常类定义 ====================

class ConfigurationError(Exception):
    """配置相关的基础异常类"""
    pass


class ConfigurationStateError(ConfigurationError):
    """配置状态异常"""
    pass


class ConfigurationSyncError(ConfigurationError):
    """配置同步异常"""
    pass


class ConfigurationValidationError(ConfigurationError):
    """配置验证异常"""
    pass


class ConfigurationSecurityError(ConfigurationError):
    """配置安全异常"""
    pass


class ConfigurationPerformanceError(ConfigurationError):
    """配置性能异常"""
    pass


class ConfigurationHealthError(ConfigurationError):
    """配置健康检查异常"""
    pass


class ConfigurationAlertError(ConfigurationError):
    """配置告警异常"""
    pass


# ==================== 枚举类定义 ====================

class ConfigurationStatus(Enum):
    """配置状态枚举"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    ERROR = "error"
    DEPRECATED = "deprecated"
    MAINTENANCE = "maintenance"


class ConfigurationType(Enum):
    """配置类型枚举"""
    SYSTEM = "system"
    APPLICATION = "application"
    DATABASE = "database"
    NETWORK = "network"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USER = "user"
    CUSTOM = "custom"


class ConfigurationChangeType(Enum):
    """配置变更类型枚举"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    ROLLBACK = "rollback"
    MIGRATE = "migrate"


class AlertSeverity(Enum):
    """告警严重级别枚举"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class SyncStatus(Enum):
    """同步状态枚举"""
    SYNCED = "synced"
    PENDING = "pending"
    FAILED = "failed"
    CONFLICT = "conflict"


# ==================== 数据类定义 ====================

@dataclass
class ConfigurationMetadata:
    """配置元数据类"""
    id: str
    name: str
    description: str
    config_type: ConfigurationType
    version: str
    created_at: datetime
    updated_at: datetime
    created_by: str
    updated_by: str
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    checksum: str = ""
    encrypted: bool = False
    
    def __post_init__(self):
        """初始化后处理"""
        if not self.checksum:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """计算配置校验和"""
        data = f"{self.id}{self.name}{self.version}{self.created_at}{self.updated_at}"
        return hashlib.sha256(data.encode()).hexdigest()


@dataclass
class ConfigurationValue:
    """配置值类"""
    key: str
    value: Any
    data_type: str
    description: str = ""
    default_value: Any = None
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    sensitive: bool = False
    encrypted_value: str = ""
    
    def __post_init__(self):
        """初始化后处理"""
        if self.sensitive and not self.encrypted_value:
            self.encrypted_value = self._encrypt_value()
    
    def _encrypt_value(self) -> str:
        """加密敏感值"""
        if not self.sensitive:
            return ""
        
        # 简单的加密实现（实际应用中应使用更安全的方法）
        key = os.environ.get('CONFIG_ENCRYPTION_KEY', 'default_key_32_chars_long!!')
        f = Fernet(Fernet.generate_key())
        encrypted = f.encrypt(str(self.value).encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_value(self) -> Any:
        """解密值"""
        if not self.sensitive or not self.encrypted_value:
            return self.value
        
        try:
            key = os.environ.get('CONFIG_ENCRYPTION_KEY', 'default_key_32_chars_long!!')
            f = Fernet(Fernet.generate_key())
            encrypted = base64.b64decode(self.encrypted_value.encode())
            decrypted = f.decrypt(encrypted)
            return decrypted.decode()
        except Exception:
            return self.value


@dataclass
class Configuration:
    """配置类"""
    metadata: ConfigurationMetadata
    values: Dict[str, ConfigurationValue]
    status: ConfigurationStatus = ConfigurationStatus.ACTIVE
    environment: str = "default"
    scope: str = "global"
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        if key in self.values:
            return self.values[key].decrypt_value() if self.values[key].sensitive else self.values[key].value
        return default
    
    def set_value(self, key: str, value: Any, data_type: str = "string", description: str = "", sensitive: bool = False):
        """设置配置值"""
        self.values[key] = ConfigurationValue(
            key=key,
            value=value,
            data_type=data_type,
            description=description,
            sensitive=sensitive
        )
    
    def validate(self) -> Tuple[bool, List[str]]:
        """验证配置"""
        errors = []
        
        for key, config_value in self.values.items():
            # 基本验证
            if not config_value.key:
                errors.append(f"配置键 '{key}' 不能为空")
            
            # 数据类型验证
            try:
                if config_value.data_type == "int":
                    int(config_value.value)
                elif config_value.data_type == "float":
                    float(config_value.value)
                elif config_value.data_type == "bool":
                    bool(config_value.value)
                elif config_value.data_type == "list":
                    if not isinstance(config_value.value, list):
                        errors.append(f"配置键 '{key}' 应该是列表类型")
                elif config_value.data_type == "dict":
                    if not isinstance(config_value.value, dict):
                        errors.append(f"配置键 '{key}' 应该是字典类型")
            except (ValueError, TypeError):
                errors.append(f"配置键 '{key}' 的值类型不正确")
            
            # 验证规则验证
            for rule_name, rule_value in config_value.validation_rules.items():
                if rule_name == "min" and isinstance(config_value.value, (int, float)):
                    if config_value.value < rule_value:
                        errors.append(f"配置键 '{key}' 的值不能小于 {rule_value}")
                elif rule_name == "max" and isinstance(config_value.value, (int, float)):
                    if config_value.value > rule_value:
                        errors.append(f"配置键 '{key}' 的值不能大于 {rule_value}")
                elif rule_name == "pattern" and isinstance(config_value.value, str):
                    if not re.match(rule_value, config_value.value):
                        errors.append(f"配置键 '{key}' 的值不符合模式 '{rule_value}'")
                elif rule_name == "choices" and config_value.value not in rule_value:
                    errors.append(f"配置键 '{key}' 的值必须是 {rule_value} 中的一个")
        
        return len(errors) == 0, errors


@dataclass
class ConfigurationChange:
    """配置变更类"""
    id: str
    configuration_id: str
    change_type: ConfigurationChangeType
    changes: Dict[str, Any]
    timestamp: datetime
    user: str
    reason: str = ""
    rollback_data: Dict[str, Any] = field(default_factory=dict)
    approved: bool = False
    approved_by: str = ""
    approved_at: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """性能指标类"""
    configuration_id: str
    access_count: int = 0
    success_count: int = 0
    error_count: int = 0
    total_response_time: float = 0.0
    last_access_time: Optional[datetime] = None
    average_response_time: float = 0.0
    success_rate: float = 100.0
    
    def update_response_time(self, response_time: float, success: bool):
        """更新响应时间统计"""
        self.access_count += 1
        self.total_response_time += response_time
        self.last_access_time = datetime.now()
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        self.average_response_time = self.total_response_time / self.access_count
        if self.access_count > 0:
            self.success_rate = (self.success_count / self.access_count) * 100


@dataclass
class HealthCheckResult:
    """健康检查结果类"""
    configuration_id: str
    status: HealthStatus
    checks: Dict[str, Any]
    timestamp: datetime
    message: str = ""
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """告警类"""
    id: str
    configuration_id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    resolved_by: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SyncOperation:
    """同步操作类"""
    id: str
    configuration_id: str
    source: str
    target: str
    status: SyncStatus
    timestamp: datetime
    completed_at: Optional[datetime] = None
    error_message: str = ""
    retry_count: int = 0
    max_retries: int = 3


# ==================== 事件系统 ====================

class ConfigurationEvent:
    """配置事件基类"""
    
    def __init__(self, event_type: str, data: Dict[str, Any]):
        self.event_type = event_type
        self.data = data
        self.timestamp = datetime.now()
        self.id = str(uuid4())


class ConfigurationEventHandler(ABC):
    """配置事件处理器抽象基类"""
    
    @abstractmethod
    async def handle_event(self, event: ConfigurationEvent):
        """处理事件"""
        pass


class EventBus:
    """事件总线"""
    
    def __init__(self):
        self._handlers: Dict[str, List[ConfigurationEventHandler]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def subscribe(self, event_type: str, handler: ConfigurationEventHandler):
        """订阅事件"""
        with self._lock:
            self._handlers[event_type].append(handler)
    
    def unsubscribe(self, event_type: str, handler: ConfigurationEventHandler):
        """取消订阅事件"""
        with self._lock:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)
    
    async def publish(self, event: ConfigurationEvent):
        """发布事件"""
        handlers = self._handlers.get(event.event_type, [])
        
        # 并发处理所有处理器
        tasks = [handler.handle_event(event) for handler in handlers]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


# ==================== 配置监控器 ====================

class ConfigurationMonitor:
    """配置状态监控器"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._configurations: Dict[str, Configuration] = {}
        self._change_history: Dict[str, List[ConfigurationChange]] = defaultdict(list)
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_interval = 30  # 30秒监控间隔
        self.logger = logging.getLogger(__name__)
    
    def add_configuration(self, config: Configuration):
        """添加配置"""
        self._configurations[config.metadata.id] = config
        self.logger.info(f"添加配置: {config.metadata.name} ({config.metadata.id})")
    
    def remove_configuration(self, config_id: str):
        """移除配置"""
        if config_id in self._configurations:
            config = self._configurations.pop(config_id)
            self.logger.info(f"移除配置: {config.metadata.name} ({config_id})")
    
    def get_configuration(self, config_id: str) -> Optional[Configuration]:
        """获取配置"""
        return self._configurations.get(config_id)
    
    def list_configurations(self) -> List[Configuration]:
        """列出所有配置"""
        return list(self._configurations.values())
    
    def start_monitoring(self):
        """开始监控"""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            self.logger.info("配置监控已启动")
    
    def stop_monitoring(self):
        """停止监控"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join()
        self.logger.info("配置监控已停止")
    
    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring_active:
            try:
                self._check_configuration_health()
                self._detect_configuration_changes()
                time.sleep(self._monitor_interval)
            except Exception as e:
                self.logger.error(f"监控循环错误: {e}")
    
    def _check_configuration_health(self):
        """检查配置健康状态"""
        for config_id, config in self._configurations.items():
            try:
                is_valid, errors = config.validate()
                if not is_valid:
                    # 发布配置验证失败事件
                    event = ConfigurationEvent(
                        "configuration_validation_failed",
                        {
                            "config_id": config_id,
                            "errors": errors,
                            "config_name": config.metadata.name
                        }
                    )
                    asyncio.create_task(self.event_bus.publish(event))
            except Exception as e:
                self.logger.error(f"检查配置 {config_id} 健康状态时出错: {e}")
    
    def _detect_configuration_changes(self):
        """检测配置变更"""
        # 这里可以实现变更检测逻辑
        # 例如：检查配置文件修改时间、版本变化等
        pass
    
    async def track_configuration_access(self, config_id: str, operation: str, success: bool, response_time: float):
        """跟踪配置访问"""
        event = ConfigurationEvent(
            "configuration_accessed",
            {
                "config_id": config_id,
                "operation": operation,
                "success": success,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat()
            }
        )
        await self.event_bus.publish(event)
    
    def get_configuration_statistics(self) -> Dict[str, Any]:
        """获取配置统计信息"""
        total_configs = len(self._configurations)
        active_configs = sum(1 for config in self._configurations.values() 
                           if config.status == ConfigurationStatus.ACTIVE)
        error_configs = sum(1 for config in self._configurations.values() 
                          if config.status == ConfigurationStatus.ERROR)
        
        return {
            "total_configurations": total_configs,
            "active_configurations": active_configs,
            "error_configurations": error_configs,
            "inactive_configurations": total_configs - active_configs - error_configs,
            "monitoring_active": self._monitoring_active
        }


# ==================== 配置协调管理器 ====================

class ConfigurationCoordinator:
    """配置协调管理器"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._sync_operations: Dict[str, SyncOperation] = {}
        self._sync_strategies: Dict[str, Callable] = {}
        self._conflict_resolvers: Dict[str, Callable] = {}
        self._distributed_nodes: Set[str] = set()
        self.logger = logging.getLogger(__name__)
        
        # 注册默认同步策略
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """注册默认同步策略"""
        self._sync_strategies["last_write_wins"] = self._last_write_wins_strategy
        self._sync_strategies["version_based"] = self._version_based_strategy
        self._sync_strategies["priority_based"] = self._priority_based_strategy
        self._sync_strategies["manual_merge"] = self._manual_merge_strategy
    
    def register_sync_strategy(self, name: str, strategy: Callable):
        """注册同步策略"""
        self._sync_strategies[name] = strategy
    
    def register_conflict_resolver(self, name: str, resolver: Callable):
        """注册冲突解决器"""
        self._conflict_resolvers[name] = resolver
    
    def add_distributed_node(self, node_id: str):
        """添加分布式节点"""
        self._distributed_nodes.add(node_id)
        self.logger.info(f"添加分布式节点: {node_id}")
    
    def remove_distributed_node(self, node_id: str):
        """移除分布式节点"""
        self._distributed_nodes.discard(node_id)
        self.logger.info(f"移除分布式节点: {node_id}")
    
    async def synchronize_configuration(self, config_id: str, strategy: str = "last_write_wins") -> SyncOperation:
        """同步配置"""
        sync_id = str(uuid4())
        
        # 创建同步操作记录
        sync_op = SyncOperation(
            id=sync_id,
            configuration_id=config_id,
            source="local",
            target="distributed",
            status=SyncStatus.PENDING,
            timestamp=datetime.now()
        )
        
        self._sync_operations[sync_id] = sync_op
        
        try:
            # 执行同步策略
            if strategy in self._sync_strategies:
                await self._sync_strategies[strategy](config_id)
                sync_op.status = SyncStatus.SYNCED
                sync_op.completed_at = datetime.now()
            else:
                raise ConfigurationSyncError(f"未知的同步策略: {strategy}")
            
            # 发布同步成功事件
            event = ConfigurationEvent(
                "configuration_synced",
                {
                    "sync_id": sync_id,
                    "config_id": config_id,
                    "strategy": strategy,
                    "status": sync_op.status.value
                }
            )
            await self.event_bus.publish(event)
            
        except Exception as e:
            sync_op.status = SyncStatus.FAILED
            sync_op.error_message = str(e)
            self.logger.error(f"配置同步失败: {e}")
            
            # 发布同步失败事件
            event = ConfigurationEvent(
                "configuration_sync_failed",
                {
                    "sync_id": sync_id,
                    "config_id": config_id,
                    "error": str(e)
                }
            )
            await self.event_bus.publish(event)
        
        return sync_op
    
    async def _last_write_wins_strategy(self, config_id: str):
        """最后写入优先策略"""
        # 模拟分布式同步过程
        await asyncio.sleep(0.1)  # 模拟网络延迟
        
        for node in self._distributed_nodes:
            # 模拟向每个节点同步配置
            await self._sync_to_node(config_id, node)
    
    async def _version_based_strategy(self, config_id: str):
        """基于版本的同步策略"""
        # 检查版本冲突并解决
        conflicts = await self._detect_version_conflicts(config_id)
        if conflicts:
            await self._resolve_version_conflicts(config_id, conflicts)
        
        await self._last_write_wins_strategy(config_id)
    
    async def _priority_based_strategy(self, config_id: str):
        """基于优先级的同步策略"""
        # 根据节点优先级进行同步
        sorted_nodes = sorted(self._distributed_nodes)
        for node in sorted_nodes:
            await self._sync_to_node(config_id, node)
    
    async def _manual_merge_strategy(self, config_id: str):
        """手动合并策略"""
        # 等待手动确认
        event = ConfigurationEvent(
            "configuration_sync_manual_merge_required",
            {
                "config_id": config_id,
                "nodes": list(self._distributed_nodes)
            }
        )
        await self.event_bus.publish(event)
        
        # 这里可以实现等待手动确认的逻辑
        await asyncio.sleep(1)  # 模拟等待时间
    
    async def _sync_to_node(self, config_id: str, node_id: str):
        """向指定节点同步配置"""
        # 模拟网络同步
        await asyncio.sleep(0.05)
        self.logger.debug(f"向节点 {node_id} 同步配置 {config_id}")
    
    async def _detect_version_conflicts(self, config_id: str) -> List[Dict[str, Any]]:
        """检测版本冲突"""
        # 模拟冲突检测
        conflicts = []
        if len(self._distributed_nodes) > 1:
            conflicts.append({
                "config_id": config_id,
                "conflict_type": "version_mismatch",
                "nodes": list(self._distributed_nodes)
            })
        return conflicts
    
    async def _resolve_version_conflicts(self, config_id: str, conflicts: List[Dict[str, Any]]):
        """解决版本冲突"""
        for conflict in conflicts:
            self.logger.warning(f"检测到版本冲突: {conflict}")
            
            # 发布冲突事件
            event = ConfigurationEvent(
                "configuration_conflict_detected",
                {
                    "config_id": config_id,
                    "conflict": conflict
                }
            )
            await self.event_bus.publish(event)
    
    def get_sync_status(self, sync_id: str) -> Optional[SyncOperation]:
        """获取同步状态"""
        return self._sync_operations.get(sync_id)
    
    def list_sync_operations(self) -> List[SyncOperation]:
        """列出所有同步操作"""
        return list(self._sync_operations.values())
    
    def get_coordination_statistics(self) -> Dict[str, Any]:
        """获取协调统计信息"""
        total_syncs = len(self._sync_operations)
        successful_syncs = sum(1 for op in self._sync_operations.values() 
                             if op.status == SyncStatus.SYNCED)
        failed_syncs = sum(1 for op in self._sync_operations.values() 
                         if op.status == SyncStatus.FAILED)
        pending_syncs = sum(1 for op in self._sync_operations.values() 
                          if op.status == SyncStatus.PENDING)
        
        return {
            "total_sync_operations": total_syncs,
            "successful_syncs": successful_syncs,
            "failed_syncs": failed_syncs,
            "pending_syncs": pending_syncs,
            "distributed_nodes": len(self._distributed_nodes),
            "sync_strategies": list(self._sync_strategies.keys())
        }


# ==================== 配置生命周期管理器 ====================

class ConfigurationLifecycleManager:
    """配置生命周期管理器"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._configurations: Dict[str, Configuration] = {}
        self._change_history: Dict[str, List[ConfigurationChange]] = defaultdict(list)
        self._rollback_stack: Dict[str, List[Configuration]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)
    
    def create_configuration(self, config: Configuration, user: str) -> ConfigurationChange:
        """创建配置"""
        config.metadata.created_by = user
        config.metadata.updated_by = user
        config.metadata.created_at = datetime.now()
        config.metadata.updated_at = datetime.now()
        
        # 验证配置
        is_valid, errors = config.validate()
        if not is_valid:
            raise ConfigurationValidationError(f"配置验证失败: {errors}")
        
        # 创建变更记录
        change = ConfigurationChange(
            id=str(uuid4()),
            configuration_id=config.metadata.id,
            change_type=ConfigurationChangeType.CREATE,
            changes={"config": asdict(config)},
            timestamp=datetime.now(),
            user=user,
            reason="初始创建"
        )
        
        self._configurations[config.metadata.id] = config
        self._change_history[config.metadata.id].append(change)
        
        # 发布创建事件
        event = ConfigurationEvent(
            "configuration_created",
            {
                "config_id": config.metadata.id,
                "config_name": config.metadata.name,
                "user": user,
                "change_id": change.id
            }
        )
        asyncio.create_task(self.event_bus.publish(event))
        
        self.logger.info(f"创建配置: {config.metadata.name} ({config.metadata.id})")
        return change
    
    def update_configuration(self, config_id: str, updates: Dict[str, Any], user: str, reason: str = "") -> ConfigurationChange:
        """更新配置"""
        if config_id not in self._configurations:
            raise ConfigurationError(f"配置不存在: {config_id}")
        
        config = self._configurations[config_id]
        
        # 保存当前状态用于回滚
        self._rollback_stack[config_id].append(copy.deepcopy(config))
        
        # 应用更新
        old_values = {}
        for key, value in updates.items():
            if key in config.values:
                old_values[key] = config.values[key].value
            config.set_value(key, value)
        
        # 更新元数据
        config.metadata.updated_by = user
        config.metadata.updated_at = datetime.now()
        
        # 创建变更记录
        change = ConfigurationChange(
            id=str(uuid4()),
            configuration_id=config_id,
            change_type=ConfigurationChangeType.UPDATE,
            changes={"updates": updates, "old_values": old_values},
            timestamp=datetime.now(),
            user=user,
            reason=reason,
            rollback_data={"old_config": asdict(copy.deepcopy(config))}
        )
        
        self._change_history[config_id].append(change)
        
        # 发布更新事件
        event = ConfigurationEvent(
            "configuration_updated",
            {
                "config_id": config_id,
                "config_name": config.metadata.name,
                "user": user,
                "changes": list(updates.keys()),
                "reason": reason,
                "change_id": change.id
            }
        )
        asyncio.create_task(self.event_bus.publish(event))
        
        self.logger.info(f"更新配置: {config.metadata.name} ({config_id})")
        return change
    
    def delete_configuration(self, config_id: str, user: str, reason: str = "") -> ConfigurationChange:
        """删除配置"""
        if config_id not in self._configurations:
            raise ConfigurationError(f"配置不存在: {config_id}")
        
        config = self._configurations[config_id]
        
        # 保存配置用于回滚
        self._rollback_stack[config_id].append(copy.deepcopy(config))
        
        # 创建变更记录
        change = ConfigurationChange(
            id=str(uuid4()),
            configuration_id=config_id,
            change_type=ConfigurationChangeType.DELETE,
            changes={"deleted_config": asdict(config)},
            timestamp=datetime.now(),
            user=user,
            reason=reason,
            rollback_data={"deleted_config": asdict(config)}
        )
        
        # 标记为已删除（软删除）
        config.status = ConfigurationStatus.DEPRECATED
        config.metadata.updated_by = user
        config.metadata.updated_at = datetime.now()
        
        self._change_history[config_id].append(change)
        
        # 发布删除事件
        event = ConfigurationEvent(
            "configuration_deleted",
            {
                "config_id": config_id,
                "config_name": config.metadata.name,
                "user": user,
                "reason": reason,
                "change_id": change.id
            }
        )
        asyncio.create_task(self.event_bus.publish(event))
        
        self.logger.info(f"删除配置: {config.metadata.name} ({config_id})")
        return change
    
    def rollback_configuration(self, config_id: str, change_id: str, user: str) -> bool:
        """回滚配置"""
        if config_id not in self._configurations:
            raise ConfigurationError(f"配置不存在: {config_id}")
        
        # 查找变更记录
        change = None
        for c in self._change_history[config_id]:
            if c.id == change_id:
                change = c
                break
        
        if not change:
            raise ConfigurationError(f"变更记录不存在: {change_id}")
        
        try:
            # 恢复配置
            if change.change_type == ConfigurationChangeType.DELETE:
                # 恢复已删除的配置
                config_data = change.rollback_data["deleted_config"]
                config = Configuration(**config_data)
                self._configurations[config_id] = config
            else:
                # 恢复其他类型的变更
                if "old_config" in change.rollback_data:
                    config_data = change.rollback_data["old_config"]
                    config = Configuration(**config_data)
                    self._configurations[config_id] = config
            
            # 创建回滚变更记录
            rollback_change = ConfigurationChange(
                id=str(uuid4()),
                configuration_id=config_id,
                change_type=ConfigurationChangeType.ROLLBACK,
                changes={"rolled_back_change": change_id},
                timestamp=datetime.now(),
                user=user,
                reason=f"回滚到变更 {change_id}"
            )
            
            self._change_history[config_id].append(rollback_change)
            
            # 发布回滚事件
            event = ConfigurationEvent(
                "configuration_rolled_back",
                {
                    "config_id": config_id,
                    "original_change_id": change_id,
                    "rollback_change_id": rollback_change.id,
                    "user": user
                }
            )
            asyncio.create_task(self.event_bus.publish(event))
            
            self.logger.info(f"回滚配置: {config_id} 到变更 {change_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"回滚配置失败: {e}")
            return False
    
    def get_configuration_history(self, config_id: str) -> List[ConfigurationChange]:
        """获取配置变更历史"""
        return self._change_history.get(config_id, [])
    
    def get_rollback_stack(self, config_id: str) -> List[Configuration]:
        """获取回滚栈"""
        return self._rollback_stack.get(config_id, [])
    
    def cleanup_old_changes(self, config_id: str, keep_count: int = 100):
        """清理旧的变更记录"""
        if config_id in self._change_history:
            changes = self._change_history[config_id]
            if len(changes) > keep_count:
                # 保留最近的变更记录
                self._change_history[config_id] = changes[-keep_count:]
                self.logger.info(f"清理配置 {config_id} 的旧变更记录，保留 {keep_count} 条")
    
    def get_lifecycle_statistics(self) -> Dict[str, Any]:
        """获取生命周期统计信息"""
        total_configs = len(self._configurations)
        total_changes = sum(len(changes) for changes in self._change_history.values())
        
        change_types = defaultdict(int)
        for changes in self._change_history.values():
            for change in changes:
                change_types[change.change_type.value] += 1
        
        return {
            "total_configurations": total_configs,
            "total_changes": total_changes,
            "change_types": dict(change_types),
            "average_changes_per_config": total_changes / total_configs if total_configs > 0 else 0
        }


# ==================== 配置性能统计器 ====================

class ConfigurationPerformanceTracker:
    """配置性能统计器"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._response_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
    
    def track_access(self, config_id: str, operation: str, response_time: float, success: bool):
        """跟踪配置访问"""
        with self._lock:
            if config_id not in self._metrics:
                self._metrics[config_id] = PerformanceMetrics(configuration_id=config_id)
            
            self._metrics[config_id].update_response_time(response_time, success)
            self._response_times[config_id].append(response_time)
        
        # 发布性能事件
        event = ConfigurationEvent(
            "configuration_performance_tracked",
            {
                "config_id": config_id,
                "operation": operation,
                "response_time": response_time,
                "success": success,
                "timestamp": datetime.now().isoformat()
            }
        )
        asyncio.create_task(self.event_bus.publish(event))
    
    def get_performance_metrics(self, config_id: str) -> Optional[PerformanceMetrics]:
        """获取性能指标"""
        return self._metrics.get(config_id)
    
    def get_all_performance_metrics(self) -> Dict[str, PerformanceMetrics]:
        """获取所有性能指标"""
        return self._metrics.copy()
    
    def get_response_time_statistics(self, config_id: str) -> Dict[str, float]:
        """获取响应时间统计"""
        if config_id not in self._response_times:
            return {}
        
        response_times = list(self._response_times[config_id])
        if not response_times:
            return {}
        
        response_times.sort()
        count = len(response_times)
        
        return {
            "min": min(response_times),
            "max": max(response_times),
            "mean": sum(response_times) / count,
            "median": response_times[count // 2],
            "p95": response_times[int(count * 0.95)],
            "p99": response_times[int(count * 0.99)],
            "samples": count
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        if not self._metrics:
            return {}
        
        total_accesses = sum(metrics.access_count for metrics in self._metrics.values())
        total_successes = sum(metrics.success_count for metrics in self._metrics.values())
        total_errors = sum(metrics.error_count for metrics in self._metrics.values())
        
        avg_response_time = sum(metrics.average_response_time for metrics in self._metrics.values()) / len(self._metrics)
        overall_success_rate = (total_successes / total_accesses * 100) if total_accesses > 0 else 0
        
        # 找出性能最差的配置
        worst_performers = sorted(
            self._metrics.items(),
            key=lambda x: x[1].average_response_time,
            reverse=True
        )[:5]
        
        return {
            "total_configurations_tracked": len(self._metrics),
            "total_accesses": total_accesses,
            "total_successes": total_successes,
            "total_errors": total_errors,
            "average_response_time": avg_response_time,
            "overall_success_rate": overall_success_rate,
            "worst_performers": [
                {
                    "config_id": config_id,
                    "average_response_time": metrics.average_response_time,
                    "success_rate": metrics.success_rate,
                    "access_count": metrics.access_count
                }
                for config_id, metrics in worst_performers
            ]
        }
    
    def detect_performance_issues(self) -> List[Dict[str, Any]]:
        """检测性能问题"""
        issues = []
        
        for config_id, metrics in self._metrics.items():
            # 检查响应时间
            if metrics.average_response_time > 5.0:  # 超过5秒
                issues.append({
                    "config_id": config_id,
                    "issue_type": "high_response_time",
                    "severity": "high" if metrics.average_response_time > 10.0 else "medium",
                    "value": metrics.average_response_time,
                    "threshold": 5.0,
                    "message": f"平均响应时间过高: {metrics.average_response_time:.2f}秒"
                })
            
            # 检查成功率
            if metrics.success_rate < 95.0:  # 成功率低于95%
                issues.append({
                    "config_id": config_id,
                    "issue_type": "low_success_rate",
                    "severity": "high" if metrics.success_rate < 90.0 else "medium",
                    "value": metrics.success_rate,
                    "threshold": 95.0,
                    "message": f"成功率过低: {metrics.success_rate:.2f}%"
                })
            
            # 检查访问频率
            if metrics.access_count > 1000:  # 高频访问
                issues.append({
                    "config_id": config_id,
                    "issue_type": "high_frequency",
                    "severity": "low",
                    "value": metrics.access_count,
                    "threshold": 1000,
                    "message": f"高频访问配置: {metrics.access_count} 次"
                })
        
        return issues
    
    def cleanup_old_metrics(self, max_age_hours: int = 24):
        """清理旧的性能指标"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        with self._lock:
            configs_to_remove = []
            for config_id, metrics in self._metrics.items():
                if metrics.last_access_time and metrics.last_access_time < cutoff_time:
                    configs_to_remove.append(config_id)
            
            for config_id in configs_to_remove:
                del self._metrics[config_id]
                del self._response_times[config_id]
        
        if configs_to_remove:
            self.logger.info(f"清理了 {len(configs_to_remove)} 个配置的旧性能指标")


# ==================== 配置健康检查器 ====================

class ConfigurationHealthChecker:
    """配置健康检查器"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._health_checks: Dict[str, Callable] = {}
        self._check_results: Dict[str, HealthCheckResult] = {}
        self._check_interval = 300  # 5分钟检查间隔
        self._health_thresholds = {
            "max_response_time": 5.0,
            "min_success_rate": 95.0,
            "max_error_rate": 5.0,
            "max_config_age_days": 30
        }
        self.logger = logging.getLogger(__name__)
        
        # 注册默认健康检查
        self._register_default_checks()
    
    def _register_default_checks(self):
        """注册默认健康检查"""
        self._health_checks["validity"] = self._check_validity
        self._health_checks["completeness"] = self._check_completeness
        self._health_checks["security"] = self._check_security
        self._health_checks["performance"] = self._check_performance
        self._health_checks["consistency"] = self._check_consistency
        self._health_checks["freshness"] = self._check_freshness
    
    def register_health_check(self, name: str, check_function: Callable):
        """注册健康检查函数"""
        self._health_checks[name] = check_function
    
    async def perform_health_check(self, config_id: str, config: Configuration) -> HealthCheckResult:
        """执行健康检查"""
        checks = {}
        issues = []
        recommendations = []
        
        # 执行所有注册的检查
        for check_name, check_function in self._health_checks.items():
            try:
                result = await check_function(config_id, config)
                checks[check_name] = result
                
                if not result.get("passed", True):
                    issues.extend(result.get("issues", []))
                    recommendations.extend(result.get("recommendations", []))
                    
            except Exception as e:
                self.logger.error(f"健康检查 {check_name} 执行失败: {e}")
                checks[check_name] = {
                    "passed": False,
                    "error": str(e),
                    "issues": [f"健康检查执行失败: {e}"],
                    "recommendations": ["检查配置健康检查函数"]
                }
        
        # 确定整体健康状态
        overall_status = HealthStatus.HEALTHY
        if any(not check.get("passed", True) for check in checks.values()):
            critical_issues = sum(1 for check in checks.values() 
                                if check.get("severity") == "critical")
            if critical_issues > 0:
                overall_status = HealthStatus.CRITICAL
            else:
                overall_status = HealthStatus.WARNING
        
        # 创建健康检查结果
        result = HealthCheckResult(
            configuration_id=config_id,
            status=overall_status,
            checks=checks,
            timestamp=datetime.now(),
            message=f"健康检查完成，发现 {len(issues)} 个问题",
            issues=issues,
            recommendations=recommendations
        )
        
        self._check_results[config_id] = result
        
        # 发布健康检查事件
        event = ConfigurationEvent(
            "configuration_health_checked",
            {
                "config_id": config_id,
                "status": overall_status.value,
                "issues_count": len(issues),
                "checks_performed": list(checks.keys())
            }
        )
        await self.event_bus.publish(event)
        
        return result
    
    async def _check_validity(self, config_id: str, config: Configuration) -> Dict[str, Any]:
        """检查配置有效性"""
        try:
            is_valid, errors = config.validate()
            return {
                "passed": is_valid,
                "severity": "critical" if not is_valid else "info",
                "issues": errors if not is_valid else [],
                "recommendations": ["修复配置验证错误"] if not is_valid else []
            }
        except Exception as e:
            return {
                "passed": False,
                "severity": "critical",
                "issues": [f"配置验证异常: {e}"],
                "recommendations": ["检查配置格式和内容"]
            }
    
    async def _check_completeness(self, config_id: str, config: Configuration) -> Dict[str, Any]:
        """检查配置完整性"""
        issues = []
        recommendations = []
        
        # 检查必需的配置项
        required_keys = ["name", "version", "description"]
        for key in required_keys:
            if not getattr(config.metadata, key):
                issues.append(f"缺少必需的配置项: {key}")
                recommendations.append(f"添加配置项: {key}")
        
        # 检查配置值
        if not config.values:
            issues.append("配置没有任何值")
            recommendations.append("添加必要的配置值")
        
        return {
            "passed": len(issues) == 0,
            "severity": "high" if len(issues) > 2 else "medium",
            "issues": issues,
            "recommendations": recommendations
        }
    
    async def _check_security(self, config_id: str, config: Configuration) -> Dict[str, Any]:
        """检查配置安全性"""
        issues = []
        recommendations = []
        
        # 检查敏感信息
        sensitive_count = sum(1 for value in config.values.values() if value.sensitive)
        if sensitive_count > 0:
            # 检查敏感信息是否已加密
            unencrypted_sensitive = sum(1 for value in config.values.values() 
                                      if value.sensitive and not value.encrypted_value)
            if unencrypted_sensitive > 0:
                issues.append(f"有 {unencrypted_sensitive} 个敏感配置项未加密")
                recommendations.append("对敏感配置项进行加密")
        
        # 检查配置权限
        if config.metadata.created_by == config.metadata.updated_by:
            issues.append("配置缺少审核流程")
            recommendations.append("实施配置变更审核流程")
        
        return {
            "passed": len(issues) == 0,
            "severity": "high" if sensitive_count > 0 else "medium",
            "issues": issues,
            "recommendations": recommendations
        }
    
    async def _check_performance(self, config_id: str, config: Configuration) -> Dict[str, Any]:
        """检查配置性能"""
        # 这里可以集成性能统计器来获取实际性能数据
        issues = []
        recommendations = []
        
        # 模拟性能检查
        avg_response_time = 0.1  # 模拟值
        success_rate = 99.5      # 模拟值
        
        if avg_response_time > self._health_thresholds["max_response_time"]:
            issues.append(f"平均响应时间过高: {avg_response_time:.2f}秒")
            recommendations.append("优化配置访问性能")
        
        if success_rate < self._health_thresholds["min_success_rate"]:
            issues.append(f"成功率过低: {success_rate:.2f}%")
            recommendations.append("检查配置访问错误原因")
        
        return {
            "passed": len(issues) == 0,
            "severity": "high" if avg_response_time > 10.0 else "medium",
            "issues": issues,
            "recommendations": recommendations,
            "metrics": {
                "average_response_time": avg_response_time,
                "success_rate": success_rate
            }
        }
    
    async def _check_consistency(self, config_id: str, config: Configuration) -> Dict[str, Any]:
        """检查配置一致性"""
        issues = []
        recommendations = []
        
        # 检查配置内部一致性
        for key, value in config.values.items():
            # 检查数据类型一致性
            if value.data_type == "int":
                try:
                    int(value.value)
                except ValueError:
                    issues.append(f"配置项 {key} 的数据类型不匹配")
                    recommendations.append(f"修正配置项 {key} 的数据类型")
            
            elif value.data_type == "bool":
                if value.value not in [True, False, "true", "false", "1", "0"]:
                    issues.append(f"配置项 {key} 的布尔值格式不正确")
                    recommendations.append(f"修正配置项 {key} 的布尔值")
        
        return {
            "passed": len(issues) == 0,
            "severity": "medium",
            "issues": issues,
            "recommendations": recommendations
        }
    
    async def _check_freshness(self, config_id: str, config: Configuration) -> Dict[str, Any]:
        """检查配置新鲜度"""
        issues = []
        recommendations = []
        
        age_days = (datetime.now() - config.metadata.updated_at).days
        max_age = self._health_thresholds["max_config_age_days"]
        
        if age_days > max_age:
            issues.append(f"配置过于陈旧，已更新于 {age_days} 天前")
            recommendations.append("更新配置内容")
        
        return {
            "passed": len(issues) == 0,
            "severity": "low" if age_days > max_age else "info",
            "issues": issues,
            "recommendations": recommendations,
            "metrics": {
                "age_days": age_days,
                "last_updated": config.metadata.updated_at.isoformat()
            }
        }
    
    def get_health_status(self, config_id: str) -> Optional[HealthCheckResult]:
        """获取配置健康状态"""
        return self._check_results.get(config_id)
    
    def get_all_health_statuses(self) -> Dict[str, HealthCheckResult]:
        """获取所有配置健康状态"""
        return self._check_results.copy()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """获取健康状态摘要"""
        if not self._check_results:
            return {}
        
        status_counts = defaultdict(int)
        total_issues = 0
        total_recommendations = 0
        
        for result in self._check_results.values():
            status_counts[result.status.value] += 1
            total_issues += len(result.issues)
            total_recommendations += len(result.recommendations)
        
        return {
            "total_configurations": len(self._check_results),
            "status_distribution": dict(status_counts),
            "total_issues": total_issues,
            "total_recommendations": total_recommendations,
            "average_issues_per_config": total_issues / len(self._check_results),
            "average_recommendations_per_config": total_recommendations / len(self._check_results)
        }


# ==================== 配置告警系统 ====================

class ConfigurationAlertSystem:
    """配置告警和通知系统"""
    
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self._alerts: Dict[str, Alert] = {}
        self._alert_rules: Dict[str, Dict[str, Any]] = {}
        self._notification_channels: Dict[str, Callable] = {}
        self._alert_history: Dict[str, List[Alert]] = defaultdict(list)
        self._suppression_rules: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
        # 注册默认通知渠道
        self._register_default_channels()
    
    def _register_default_channels(self):
        """注册默认通知渠道"""
        self._notification_channels["console"] = self._send_console_notification
        self._notification_channels["log"] = self._send_log_notification
        self._notification_channels["webhook"] = self._send_webhook_notification
        # 可以添加更多通知渠道，如邮件、短信等
    
    def register_notification_channel(self, name: str, channel_function: Callable):
        """注册通知渠道"""
        self._notification_channels[name] = channel_function
    
    def add_alert_rule(self, rule_id: str, rule_config: Dict[str, Any]):
        """添加告警规则"""
        self._alert_rules[rule_id] = rule_config
        self.logger.info(f"添加告警规则: {rule_id}")
    
    def remove_alert_rule(self, rule_id: str):
        """移除告警规则"""
        if rule_id in self._alert_rules:
            del self._alert_rules[rule_id]
            self.logger.info(f"移除告警规则: {rule_id}")
    
    def add_suppression_rule(self, rule_id: str, rule_config: Dict[str, Any]):
        """添加抑制规则"""
        self._suppression_rules[rule_id] = rule_config
        self.logger.info(f"添加抑制规则: {rule_id}")
    
    def remove_suppression_rule(self, rule_id: str):
        """移除抑制规则"""
        if rule_id in self._suppression_rules:
            del self._suppression_rules[rule_id]
            self.logger.info(f"移除抑制规则: {rule_id}")
    
    async def create_alert(self, configuration_id: str, severity: AlertSeverity, 
                          title: str, message: str, metadata: Dict[str, Any] = None) -> Alert:
        """创建告警"""
        alert_id = str(uuid4())
        
        # 检查是否应该抑制此告警
        if self._should_suppress_alert(configuration_id, severity, title):
            self.logger.info(f"告警被抑制: {title}")
            return None
        
        alert = Alert(
            id=alert_id,
            configuration_id=configuration_id,
            severity=severity,
            title=title,
            message=message,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        
        with self._lock:
            self._alerts[alert_id] = alert
            self._alert_history[configuration_id].append(alert)
        
        # 发送通知
        await self._send_notifications(alert)
        
        # 发布告警事件
        event = ConfigurationEvent(
            "alert_created",
            {
                "alert_id": alert_id,
                "configuration_id": configuration_id,
                "severity": severity.value,
                "title": title,
                "message": message
            }
        )
        await self.event_bus.publish(event)
        
        self.logger.warning(f"创建告警 [{severity.value}]: {title} - {message}")
        return alert
    
    def _should_suppress_alert(self, configuration_id: str, severity: AlertSeverity, title: str) -> bool:
        """检查是否应该抑制告警"""
        current_time = datetime.now()
        
        for rule_id, rule in self._suppression_rules.items():
            # 检查配置ID匹配
            if rule.get("configuration_id") and rule["configuration_id"] != configuration_id:
                continue
            
            # 检查严重级别匹配
            if rule.get("severity") and rule["severity"] != severity.value:
                continue
            
            # 检查标题匹配
            if rule.get("title_pattern"):
                if not re.search(rule["title_pattern"], title):
                    continue
            
            # 检查时间窗口
            time_window = rule.get("time_window_minutes", 5)
            cutoff_time = current_time - timedelta(minutes=time_window)
            
            # 检查最近的告警
            recent_alerts = [
                alert for alert in self._alert_history.get(configuration_id, [])
                if alert.timestamp > cutoff_time and alert.title == title
            ]
            
            if len(recent_alerts) >= rule.get("max_count", 1):
                return True
        
        return False
    
    async def _send_notifications(self, alert: Alert):
        """发送通知"""
        notification_tasks = []
        
        for channel_name, channel_function in self._notification_channels.items():
            try:
                task = asyncio.create_task(channel_function(alert))
                notification_tasks.append(task)
            except Exception as e:
                self.logger.error(f"通知渠道 {channel_name} 初始化失败: {e}")
        
        if notification_tasks:
            await asyncio.gather(*notification_tasks, return_exceptions=True)
    
    async def _send_console_notification(self, alert: Alert):
        """发送控制台通知"""
        print(f"\n🚨 配置告警 [{alert.severity.value.upper()}]")
        print(f"配置ID: {alert.configuration_id}")
        print(f"标题: {alert.title}")
        print(f"消息: {alert.message}")
        print(f"时间: {alert.timestamp}")
        print("-" * 50)
    
    async def _send_log_notification(self, alert: Alert):
        """发送日志通知"""
        log_message = f"配置告警 [{alert.severity.value}] - 配置: {alert.configuration_id} - {alert.title}: {alert.message}"
        
        if alert.severity == AlertSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif alert.severity == AlertSeverity.HIGH:
            self.logger.error(log_message)
        elif alert.severity == AlertSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)
    
    async def _send_webhook_notification(self, alert: Alert):
        """发送Webhook通知"""
        webhook_url = os.environ.get("ALERT_WEBHOOK_URL")
        if not webhook_url:
            return
        
        payload = {
            "alert_id": alert.id,
            "configuration_id": alert.configuration_id,
            "severity": alert.severity.value,
            "title": alert.title,
            "message": alert.message,
            "timestamp": alert.timestamp.isoformat(),
            "metadata": alert.metadata
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, timeout=10) as response:
                    if response.status == 200:
                        self.logger.info(f"Webhook通知发送成功: {alert.id}")
                    else:
                        self.logger.error(f"Webhook通知发送失败: {response.status}")
        except Exception as e:
            self.logger.error(f"Webhook通知发送异常: {e}")
    
    def resolve_alert(self, alert_id: str, resolved_by: str) -> bool:
        """解决告警"""
        if alert_id not in self._alerts:
            return False
        
        alert = self._alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now()
        alert.resolved_by = resolved_by
        
        self.logger.info(f"告警已解决: {alert_id} by {resolved_by}")
        return True
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """获取告警"""
        return self._alerts.get(alert_id)
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        return [alert for alert in self._alerts.values() if not alert.resolved]
    
    def get_alerts_by_configuration(self, config_id: str) -> List[Alert]:
        """获取指定配置的告警"""
        return self._alert_history.get(config_id, [])
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        total_alerts = len(self._alerts)
        active_alerts = len(self.get_active_alerts())
        
        severity_counts = defaultdict(int)
        for alert in self._alerts.values():
            severity_counts[alert.severity.value] += 1
        
        resolved_count = sum(1 for alert in self._alerts.values() if alert.resolved)
        resolution_rate = (resolved_count / total_alerts * 100) if total_alerts > 0 else 0
        
        return {
            "total_alerts": total_alerts,
            "active_alerts": active_alerts,
            "resolved_alerts": resolved_count,
            "resolution_rate": resolution_rate,
            "severity_distribution": dict(severity_counts),
            "alert_rules": len(self._alert_rules),
            "suppression_rules": len(self._suppression_rules),
            "notification_channels": list(self._notification_channels.keys())
        }


# ==================== 分布式配置同步 ====================

class DistributedConfigurationSync:
    """分布式配置同步器"""
    
    def __init__(self, event_bus: EventBus, node_id: str):
        self.event_bus = event_bus
        self.node_id = node_id
        self._peer_nodes: Dict[str, Dict[str, Any]] = {}
        self._sync_queue: asyncio.Queue = asyncio.Queue()
        self._sync_workers: List[asyncio.Task] = []
        self._sync_interval = 60  # 60秒同步间隔
        self._max_workers = 5
        self._running = False
        self.logger = logging.getLogger(__name__)
    
    async def start(self):
        """启动分布式同步"""
        if self._running:
            return
        
        self._running = True
        
        # 启动同步工作器
        for i in range(self._max_workers):
            worker = asyncio.create_task(self._sync_worker(f"worker-{i}"))
            self._sync_workers.append(worker)
        
        # 启动定期同步任务
        sync_task = asyncio.create_task(self._periodic_sync())
        self._sync_workers.append(sync_task)
        
        self.logger.info(f"分布式配置同步已启动，节点ID: {self.node_id}")
    
    async def stop(self):
        """停止分布式同步"""
        self._running = False
        
        # 取消所有工作器
        for worker in self._sync_workers:
            worker.cancel()
        
        # 等待工作器结束
        if self._sync_workers:
            await asyncio.gather(*self._sync_workers, return_exceptions=True)
        
        self._sync_workers.clear()
        self.logger.info("分布式配置同步已停止")
    
    def add_peer_node(self, node_id: str, node_info: Dict[str, Any]):
        """添加对等节点"""
        self._peer_nodes[node_id] = {
            **node_info,
            "last_seen": datetime.now(),
            "status": "online"
        }
        self.logger.info(f"添加对等节点: {node_id}")
    
    def remove_peer_node(self, node_id: str):
        """移除对等节点"""
        if node_id in self._peer_nodes:
            del self._peer_nodes[node_id]
            self.logger.info(f"移除对等节点: {node_id}")
    
    async def queue_sync_operation(self, operation: Dict[str, Any]):
        """排队同步操作"""
        await self._sync_queue.put(operation)
    
    async def _sync_worker(self, worker_id: str):
        """同步工作器"""
        while self._running:
            try:
                # 从队列获取同步操作
                operation = await asyncio.wait_for(self._sync_queue.get(), timeout=1.0)
                
                # 执行同步
                await self._execute_sync_operation(operation)
                
                # 标记任务完成
                self._sync_queue.task_done()
                
            except asyncio.TimeoutError:
                # 正常超时，继续循环
                continue
            except Exception as e:
                self.logger.error(f"同步工作器 {worker_id} 错误: {e}")
    
    async def _execute_sync_operation(self, operation: Dict[str, Any]):
        """执行同步操作"""
        operation_type = operation.get("type")
        
        if operation_type == "config_update":
            await self._sync_config_update(operation)
        elif operation_type == "config_delete":
            await self._sync_config_delete(operation)
        elif operation_type == "heartbeat":
            await self._process_heartbeat(operation)
        else:
            self.logger.warning(f"未知的同步操作类型: {operation_type}")
    
    async def _sync_config_update(self, operation: Dict[str, Any]):
        """同步配置更新"""
        config_data = operation.get("config_data")
        target_nodes = operation.get("target_nodes", [])
        
        for node_id in target_nodes:
            if node_id in self._peer_nodes:
                try:
                    await self._send_to_node(node_id, {
                        "type": "config_update",
                        "config_data": config_data,
                        "source_node": self.node_id,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    self.logger.error(f"向节点 {node_id} 同步配置更新失败: {e}")
    
    async def _sync_config_delete(self, operation: Dict[str, Any]):
        """同步配置删除"""
        config_id = operation.get("config_id")
        target_nodes = operation.get("target_nodes", [])
        
        for node_id in target_nodes:
            if node_id in self._peer_nodes:
                try:
                    await self._send_to_node(node_id, {
                        "type": "config_delete",
                        "config_id": config_id,
                        "source_node": self.node_id,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    self.logger.error(f"向节点 {node_id} 同步配置删除失败: {e}")
    
    async def _process_heartbeat(self, operation: Dict[str, Any]):
        """处理心跳"""
        source_node = operation.get("source_node")
        if source_node in self._peer_nodes:
            self._peer_nodes[source_node]["last_seen"] = datetime.now()
            self._peer_nodes[source_node]["status"] = "online"
    
    async def _send_to_node(self, node_id: str, message: Dict[str, Any]):
        """向节点发送消息"""
        node_info = self._peer_nodes[node_id]
        
        # 这里应该实现实际的节点间通信
        # 可以使用WebSocket、gRPC、HTTP等协议
        
        # 模拟网络延迟
        await asyncio.sleep(0.1)
        
        self.logger.debug(f"向节点 {node_id} 发送消息: {message['type']}")
    
    async def _periodic_sync(self):
        """定期同步"""
        while self._running:
            try:
                # 发送心跳
                await self._send_heartbeat()
                
                # 检查节点状态
                await self._check_node_health()
                
                # 等待下次同步
                await asyncio.sleep(self._sync_interval)
                
            except Exception as e:
                self.logger.error(f"定期同步错误: {e}")
    
    async def _send_heartbeat(self):
        """发送心跳"""
        heartbeat_message = {
            "type": "heartbeat",
            "node_id": self.node_id,
            "timestamp": datetime.now().isoformat(),
            "status": "online"
        }
        
        for node_id in self._peer_nodes:
            try:
                await self._send_to_node(node_id, heartbeat_message)
            except Exception as e:
                self.logger.warning(f"向节点 {node_id} 发送心跳失败: {e}")
    
    async def _check_node_health(self):
        """检查节点健康状态"""
        current_time = datetime.now()
        offline_threshold = timedelta(minutes=5)
        
        for node_id, node_info in self._peer_nodes.items():
            last_seen = node_info.get("last_seen", datetime.min)
            if current_time - last_seen > offline_threshold:
                node_info["status"] = "offline"
                self.logger.warning(f"节点 {node_id} 离线")
    
    def get_sync_status(self) -> Dict[str, Any]:
        """获取同步状态"""
        online_nodes = sum(1 for node in self._peer_nodes.values() if node["status"] == "online")
        
        return {
            "node_id": self.node_id,
            "running": self._running,
            "total_peer_nodes": len(self._peer_nodes),
            "online_nodes": online_nodes,
            "offline_nodes": len(self._peer_nodes) - online_nodes,
            "queue_size": self._sync_queue.qsize(),
            "active_workers": len(self._sync_workers)
        }


# ==================== 统一配置API ====================

class ConfigurationAPI:
    """统一配置接口和API"""
    
    def __init__(self, aggregator: 'ConfigurationStateAggregator'):
        self.aggregator = aggregator
        self.logger = logging.getLogger(__name__)
    
    async def get_configuration(self, config_id: str) -> Optional[Configuration]:
        """获取配置"""
        start_time = time.time()
        try:
            config = self.aggregator.monitor.get_configuration(config_id)
            response_time = time.time() - start_time
            
            # 跟踪性能
            self.aggregator.performance_tracker.track_access(
                config_id, "get", response_time, config is not None
            )
            
            return config
            
        except Exception as e:
            response_time = time.time() - start_time
            self.aggregator.performance_tracker.track_access(
                config_id, "get", response_time, False
            )
            self.logger.error(f"获取配置失败: {e}")
            raise
    
    async def create_configuration(self, config: Configuration, user: str) -> Configuration:
        """创建配置"""
        start_time = time.time()
        try:
            change = self.aggregator.lifecycle_manager.create_configuration(config, user)
            response_time = time.time() - start_time
            
            # 跟踪性能
            self.aggregator.performance_tracker.track_access(
                config.metadata.id, "create", response_time, True
            )
            
            # 执行健康检查
            await self.aggregator.health_checker.perform_health_check(config.metadata.id, config)
            
            return config
            
        except Exception as e:
            response_time = time.time() - start_time
            self.aggregator.performance_tracker.track_access(
                config.metadata.id, "create", response_time, False
            )
            self.logger.error(f"创建配置失败: {e}")
            raise
    
    async def update_configuration(self, config_id: str, updates: Dict[str, Any], 
                                  user: str, reason: str = "") -> Configuration:
        """更新配置"""
        start_time = time.time()
        try:
            change = self.aggregator.lifecycle_manager.update_configuration(
                config_id, updates, user, reason
            )
            response_time = time.time() - start_time
            
            # 跟踪性能
            self.aggregator.performance_tracker.track_access(
                config_id, "update", response_time, True
            )
            
            # 获取更新后的配置
            config = self.aggregator.monitor.get_configuration(config_id)
            
            # 执行健康检查
            if config:
                await self.aggregator.health_checker.perform_health_check(config_id, config)
            
            return config
            
        except Exception as e:
            response_time = time.time() - start_time
            self.aggregator.performance_tracker.track_access(
                config_id, "update", response_time, False
            )
            self.logger.error(f"更新配置失败: {e}")
            raise
    
    async def delete_configuration(self, config_id: str, user: str, reason: str = "") -> bool:
        """删除配置"""
        start_time = time.time()
        try:
            change = self.aggregator.lifecycle_manager.delete_configuration(
                config_id, user, reason
            )
            response_time = time.time() - start_time
            
            # 跟踪性能
            self.aggregator.performance_tracker.track_access(
                config_id, "delete", response_time, True
            )
            
            return True
            
        except Exception as e:
            response_time = time.time() - start_time
            self.aggregator.performance_tracker.track_access(
                config_id, "delete", response_time, False
            )
            self.logger.error(f"删除配置失败: {e}")
            raise
    
    async def list_configurations(self, filters: Dict[str, Any] = None) -> List[Configuration]:
        """列出配置"""
        start_time = time.time()
        try:
            all_configs = self.aggregator.monitor.list_configurations()
            
            # 应用过滤器
            if filters:
                filtered_configs = []
                for config in all_configs:
                    if self._apply_filters(config, filters):
                        filtered_configs.append(config)
                all_configs = filtered_configs
            
            response_time = time.time() - start_time
            
            # 跟踪性能（不针对特定配置）
            self.aggregator.performance_tracker.track_access(
                "list_configurations", "list", response_time, True
            )
            
            return all_configs
            
        except Exception as e:
            response_time = time.time() - start_time
            self.aggregator.performance_tracker.track_access(
                "list_configurations", "list", response_time, False
            )
            self.logger.error(f"列出配置失败: {e}")
            raise
    
    def _apply_filters(self, config: Configuration, filters: Dict[str, Any]) -> bool:
        """应用过滤器"""
        for key, value in filters.items():
            if key == "status":
                if config.status.value != value:
                    return False
            elif key == "type":
                if config.metadata.config_type.value != value:
                    return False
            elif key == "environment":
                if config.environment != value:
                    return False
            elif key == "created_after":
                if config.metadata.created_at < value:
                    return False
            elif key == "created_before":
                if config.metadata.created_at > value:
                    return False
            elif key == "tag":
                if value not in config.metadata.tags:
                    return False
            # 可以添加更多过滤条件
        
        return True
    
    async def get_configuration_history(self, config_id: str) -> List[ConfigurationChange]:
        """获取配置历史"""
        return self.aggregator.lifecycle_manager.get_configuration_history(config_id)
    
    async def rollback_configuration(self, config_id: str, change_id: str, user: str) -> bool:
        """回滚配置"""
        return self.aggregator.lifecycle_manager.rollback_configuration(config_id, change_id, user)
    
    async def synchronize_configuration(self, config_id: str, strategy: str = "last_write_wins") -> SyncOperation:
        """同步配置"""
        return await self.aggregator.coordinator.synchronize_configuration(config_id, strategy)
    
    async def get_health_status(self, config_id: str) -> Optional[HealthCheckResult]:
        """获取健康状态"""
        return self.aggregator.health_checker.get_health_status(config_id)
    
    async def get_performance_metrics(self, config_id: str) -> Optional[PerformanceMetrics]:
        """获取性能指标"""
        return self.aggregator.performance_tracker.get_performance_metrics(config_id)
    
    async def get_alerts(self, config_id: str = None, active_only: bool = False) -> List[Alert]:
        """获取告警"""
        if config_id:
            alerts = self.aggregator.alert_system.get_alerts_by_configuration(config_id)
        else:
            alerts = list(self.aggregator.alert_system._alerts.values())
        
        if active_only:
            alerts = [alert for alert in alerts if not alert.resolved]
        
        return alerts
    
    async def resolve_alert(self, alert_id: str, user: str) -> bool:
        """解决告警"""
        return self.aggregator.alert_system.resolve_alert(alert_id, user)
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """获取系统统计"""
        return {
            "monitor": self.aggregator.monitor.get_configuration_statistics(),
            "lifecycle": self.aggregator.lifecycle_manager.get_lifecycle_statistics(),
            "coordinator": self.aggregator.coordinator.get_coordination_statistics(),
            "performance": self.aggregator.performance_tracker.get_performance_summary(),
            "health": self.aggregator.health_checker.get_health_summary(),
            "alerts": self.aggregator.alert_system.get_alert_statistics()
        }


# ==================== 主配置状态聚合器 ====================

class ConfigurationStateAggregator:
    """
    K9配置状态聚合器
    
    这是系统的核心组件，集成了所有配置管理功能：
    - 配置状态监控
    - 配置协调管理
    - 配置生命周期管理
    - 配置性能统计
    - 配置健康检查
    - 配置告警系统
    - 统一配置API
    - 分布式配置同步
    
    示例用法：
        ```python
        # 创建聚合器实例
        aggregator = ConfigurationStateAggregator()
        
        # 启动聚合器
        await aggregator.start()
        
        # 创建配置
        config = Configuration(
            metadata=ConfigurationMetadata(...),
            values={...}
        )
        
        # 通过API创建配置
        created_config = await aggregator.api.create_configuration(config, "admin")
        
        # 获取配置
        retrieved_config = await aggregator.api.get_configuration(config.metadata.id)
        
        # 停止聚合器
        await aggregator.stop()
        ```
    """
    
    def __init__(self, 
                 config_dir: str = "./config",
                 db_path: str = "./config_state.db",
                 log_level: str = "INFO",
                 enable_distributed_sync: bool = True,
                 node_id: str = None):
        """
        初始化配置状态聚合器
        
        Args:
            config_dir: 配置文件目录
            db_path: 数据库文件路径
            log_level: 日志级别
            enable_distributed_sync: 是否启用分布式同步
            node_id: 节点ID（用于分布式同步）
        """
        
        # 设置日志
        self._setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        # 初始化事件总线
        self.event_bus = EventBus()
        
        # 初始化各个组件
        self.monitor = ConfigurationMonitor(self.event_bus)
        self.coordinator = ConfigurationCoordinator(self.event_bus)
        self.lifecycle_manager = ConfigurationLifecycleManager(self.event_bus)
        self.performance_tracker = ConfigurationPerformanceTracker(self.event_bus)
        self.health_checker = ConfigurationHealthChecker(self.event_bus)
        self.alert_system = ConfigurationAlertSystem(self.event_bus)
        self.api = ConfigurationAPI(self)
        
        # 初始化分布式同步（可选）
        self.distributed_sync = None
        if enable_distributed_sync:
            self.distributed_sync = DistributedConfigurationSync(
                self.event_bus, 
                node_id or socket.gethostname()
            )
        
        # 配置设置
        self.config_dir = Path(config_dir)
        self.db_path = db_path
        self.enable_distributed_sync = enable_distributed_sync
        
        # 状态
        self._running = False
        self._start_time: Optional[datetime] = None
        
        # 初始化数据库
        self._init_database()
        
        # 注册事件处理器
        self._register_event_handlers()
        
        self.logger.info("配置状态聚合器初始化完成")
    
    def _setup_logging(self, log_level: str):
        """设置日志"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('configuration_state.log')
            ]
        )
    
    def _init_database(self):
        """初始化数据库"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建配置表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS configurations (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config_type TEXT NOT NULL,
                    version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    environment TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    created_by TEXT NOT NULL,
                    updated_by TEXT NOT NULL,
                    data TEXT NOT NULL,
                    checksum TEXT NOT NULL
                )
            ''')
            
            # 创建变更历史表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS configuration_changes (
                    id TEXT PRIMARY KEY,
                    configuration_id TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    changes TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    user TEXT NOT NULL,
                    reason TEXT,
                    rollback_data TEXT,
                    approved BOOLEAN DEFAULT FALSE,
                    approved_by TEXT,
                    approved_at TIMESTAMP,
                    FOREIGN KEY (configuration_id) REFERENCES configurations (id)
                )
            ''')
            
            # 创建性能指标表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    configuration_id TEXT PRIMARY KEY,
                    access_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    total_response_time REAL DEFAULT 0.0,
                    last_access_time TIMESTAMP,
                    average_response_time REAL DEFAULT 0.0,
                    success_rate REAL DEFAULT 100.0,
                    FOREIGN KEY (configuration_id) REFERENCES configurations (id)
                )
            ''')
            
            # 创建健康检查结果表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS health_check_results (
                    id TEXT PRIMARY KEY,
                    configuration_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    checks TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    message TEXT,
                    issues TEXT,
                    recommendations TEXT,
                    FOREIGN KEY (configuration_id) REFERENCES configurations (id)
                )
            ''')
            
            # 创建告警表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    configuration_id TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_at TIMESTAMP,
                    resolved_by TEXT,
                    metadata TEXT,
                    FOREIGN KEY (configuration_id) REFERENCES configurations (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
            self.logger.info("数据库初始化完成")
            
        except Exception as e:
            self.logger.error(f"数据库初始化失败: {e}")
            raise ConfigurationError(f"数据库初始化失败: {e}")
    
    def _register_event_handlers(self):
        """注册事件处理器"""
        # 配置验证失败事件
        self.event_bus.subscribe("configuration_validation_failed", 
                                ConfigurationValidationHandler(self))
        
        # 配置同步事件
        self.event_bus.subscribe("configuration_synced", 
                                ConfigurationSyncHandler(self))
        
        # 配置健康检查事件
        self.event_bus.subscribe("configuration_health_checked", 
                                ConfigurationHealthHandler(self))
        
        # 配置性能问题事件
        self.event_bus.subscribe("configuration_performance_tracked", 
                                ConfigurationPerformanceHandler(self))
        
        # 告警事件
        self.event_bus.subscribe("alert_created", 
                                ConfigurationAlertHandler(self))
    
    async def start(self):
        """启动配置状态聚合器"""
        if self._running:
            self.logger.warning("配置状态聚合器已经在运行")
            return
        
        self._running = True
        self._start_time = datetime.now()
        
        try:
            # 启动监控
            self.monitor.start_monitoring()
            
            # 启动分布式同步
            if self.distributed_sync:
                await self.distributed_sync.start()
            
            # 加载现有配置
            await self._load_configurations()
            
            self.logger.info("配置状态聚合器启动完成")
            
        except Exception as e:
            self._running = False
            self.logger.error(f"启动配置状态聚合器失败: {e}")
            raise
    
    async def stop(self):
        """停止配置状态聚合器"""
        if not self._running:
            return
        
        try:
            # 停止监控
            self.monitor.stop_monitoring()
            
            # 停止分布式同步
            if self.distributed_sync:
                await self.distributed_sync.stop()
            
            # 保存配置
            await self._save_configurations()
            
            self._running = False
            
            self.logger.info("配置状态聚合器已停止")
            
        except Exception as e:
            self.logger.error(f"停止配置状态聚合器失败: {e}")
            raise
    
    async def _load_configurations(self):
        """加载配置"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT data FROM configurations")
            rows = cursor.fetchall()
            
            for row in rows:
                config_data = json.loads(row[0])
                config = Configuration(**config_data)
                self.monitor.add_configuration(config)
            
            conn.close()
            
            self.logger.info(f"加载了 {len(rows)} 个配置")
            
        except Exception as e:
            self.logger.error(f"加载配置失败: {e}")
    
    async def _save_configurations(self):
        """保存配置"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for config in self.monitor.list_configurations():
                config_json = json.dumps(asdict(config), default=str)
                cursor.execute('''
                    INSERT OR REPLACE INTO configurations 
                    (id, name, config_type, version, status, environment, scope,
                     created_at, updated_at, created_by, updated_by, data, checksum)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    config.metadata.id,
                    config.metadata.name,
                    config.metadata.config_type.value,
                    config.metadata.version,
                    config.status.value,
                    config.environment,
                    config.scope,
                    config.metadata.created_at,
                    config.metadata.updated_at,
                    config.metadata.created_by,
                    config.metadata.updated_by,
                    config_json,
                    config.metadata.checksum
                ))
            
            conn.commit()
            conn.close()
            
            self.logger.info("配置保存完成")
            
        except Exception as e:
            self.logger.error(f"保存配置失败: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """获取聚合器状态"""
        uptime = datetime.now() - self._start_time if self._start_time else timedelta(0)
        
        return {
            "running": self._running,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "uptime_seconds": uptime.total_seconds(),
            "components": {
                "monitor": self.monitor.get_configuration_statistics(),
                "coordinator": self.coordinator.get_coordination_statistics(),
                "lifecycle": self.lifecycle_manager.get_lifecycle_statistics(),
                "performance": self.performance_tracker.get_performance_summary(),
                "health": self.health_checker.get_health_summary(),
                "alerts": self.alert_system.get_alert_statistics()
            },
            "distributed_sync": self.distributed_sync.get_sync_status() if self.distributed_sync else None
        }
    
    @asynccontextmanager
    async def lifespan(self):
        """生命周期管理器"""
        await self.start()
        try:
            yield self
        finally:
            await self.stop()


# ==================== 事件处理器实现 ====================

class ConfigurationValidationHandler(ConfigurationEventHandler):
    """配置验证失败事件处理器"""
    
    def __init__(self, aggregator: ConfigurationStateAggregator):
        self.aggregator = aggregator
    
    async def handle_event(self, event: ConfigurationEvent):
        """处理配置验证失败事件"""
        config_id = event.data.get("config_id")
        errors = event.data.get("errors", [])
        
        # 创建告警
        await self.aggregator.alert_system.create_alert(
            configuration_id=config_id,
            severity=AlertSeverity.HIGH,
            title="配置验证失败",
            message=f"配置 {config_id} 验证失败: {', '.join(errors)}",
            metadata={"errors": errors, "event_id": event.id}
        )


class ConfigurationSyncHandler(ConfigurationEventHandler):
    """配置同步事件处理器"""
    
    def __init__(self, aggregator: ConfigurationStateAggregator):
        self.aggregator = aggregator
    
    async def handle_event(self, event: ConfigurationEvent):
        """处理配置同步事件"""
        sync_id = event.data.get("sync_id")
        config_id = event.data.get("config_id")
        status = event.data.get("status")
        
        if status == "failed":
            # 同步失败，创建告警
            await self.aggregator.alert_system.create_alert(
                configuration_id=config_id,
                severity=AlertSeverity.MEDIUM,
                title="配置同步失败",
                message=f"配置 {config_id} 同步失败，同步ID: {sync_id}",
                metadata={"sync_id": sync_id, "event_id": event.id}
            )


class ConfigurationHealthHandler(ConfigurationEventHandler):
    """配置健康检查事件处理器"""
    
    def __init__(self, aggregator: ConfigurationStateAggregator):
        self.aggregator = aggregator
    
    async def handle_event(self, event: ConfigurationEvent):
        """处理配置健康检查事件"""
        config_id = event.data.get("config_id")
        status = event.data.get("status")
        issues_count = event.data.get("issues_count", 0)
        
        if status == "critical" and issues_count > 0:
            # 健康状态严重，创建告警
            await self.aggregator.alert_system.create_alert(
                configuration_id=config_id,
                severity=AlertSeverity.CRITICAL,
                title="配置健康状态严重",
                message=f"配置 {config_id} 健康状态严重，发现 {issues_count} 个问题",
                metadata={"status": status, "issues_count": issues_count, "event_id": event.id}
            )


class ConfigurationPerformanceHandler(ConfigurationEventHandler):
    """配置性能事件处理器"""
    
    def __init__(self, aggregator: ConfigurationStateAggregator):
        self.aggregator = aggregator
    
    async def handle_event(self, event: ConfigurationEvent):
        """处理配置性能事件"""
        config_id = event.data.get("config_id")
        response_time = event.data.get("response_time", 0)
        success = event.data.get("success", True)
        
        if not success or response_time > 5.0:
            # 性能问题，创建告警
            severity = AlertSeverity.HIGH if not success else AlertSeverity.MEDIUM
            message = f"配置 {config_id} 访问失败" if not success else f"配置 {config_id} 响应时间过长: {response_time:.2f}秒"
            
            await self.aggregator.alert_system.create_alert(
                configuration_id=config_id,
                severity=severity,
                title="配置性能问题",
                message=message,
                metadata={
                    "response_time": response_time,
                    "success": success,
                    "event_id": event.id
                }
            )


class ConfigurationAlertHandler(ConfigurationEventHandler):
    """配置告警事件处理器"""
    
    def __init__(self, aggregator: ConfigurationStateAggregator):
        self.aggregator = aggregator
    
    async def handle_event(self, event: ConfigurationEvent):
        """处理告警创建事件"""
        alert_id = event.data.get("alert_id")
        severity = event.data.get("severity")
        title = event.data.get("title")
        
        # 可以在这里添加额外的告警处理逻辑
        # 例如：发送邮件、短信、Slack通知等
        
        self.aggregator.logger.info(f"告警已创建 [{severity}]: {title} (ID: {alert_id})")


# ==================== 使用示例和测试函数 ====================

async def example_usage():
    """使用示例"""
    print("=== K9配置状态聚合器使用示例 ===\n")
    
    # 创建聚合器实例
    aggregator = ConfigurationStateAggregator(
        config_dir="./configs",
        db_path="./config_state.db",
        log_level="INFO",
        enable_distributed_sync=True
    )
    
    try:
        # 启动聚合器
        print("1. 启动配置状态聚合器...")
        await aggregator.start()
        
        # 创建示例配置
        print("\n2. 创建示例配置...")
        config_metadata = ConfigurationMetadata(
            id="app-config-001",
            name="应用程序配置",
            description="主应用程序的配置",
            config_type=ConfigurationType.APPLICATION,
            version="1.0.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            created_by="admin",
            updated_by="admin",
            tags=["production", "critical"]
        )
        
        config = Configuration(
            metadata=config_metadata,
            values={
                "database_url": ConfigurationValue(
                    key="database_url",
                    value="postgresql://localhost:5432/mydb",
                    data_type="string",
                    description="数据库连接URL"
                ),
                "max_connections": ConfigurationValue(
                    key="max_connections",
                    value=100,
                    data_type="int",
                    description="最大连接数",
                    validation_rules={"min": 1, "max": 1000}
                ),
                "debug_mode": ConfigurationValue(
                    key="debug_mode",
                    value=False,
                    data_type="bool",
                    description="调试模式"
                ),
                "api_keys": ConfigurationValue(
                    key="api_keys",
                    value=["key1", "key2"],
                    data_type="list",
                    description="API密钥列表"
                )
            },
            status=ConfigurationStatus.ACTIVE,
            environment="production",
            scope="global"
        )
        
        # 通过API创建配置
        created_config = await aggregator.api.create_configuration(config, "admin")
        print(f"   ✓ 配置已创建: {created_config.metadata.name}")
        
        # 获取配置
        print("\n3. 获取配置...")
        retrieved_config = await aggregator.api.get_configuration(config.metadata.id)
        print(f"   ✓ 配置已获取: {retrieved_config.metadata.name}")
        print(f"   - 数据库URL: {retrieved_config.get_value('database_url')}")
        print(f"   - 最大连接数: {retrieved_config.get_value('max_connections')}")
        print(f"   - 调试模式: {retrieved_config.get_value('debug_mode')}")
        
        # 更新配置
        print("\n4. 更新配置...")
        updates = {
            "max_connections": 200,
            "debug_mode": True
        }
        await aggregator.api.update_configuration(
            config.metadata.id, 
            updates, 
            "admin", 
            "增加连接数并启用调试模式"
        )
        print("   ✓ 配置已更新")
        
        # 执行健康检查
        print("\n5. 执行健康检查...")
        health_result = await aggregator.api.get_health_status(config.metadata.id)
        if health_result:
            print(f"   - 健康状态: {health_result.status.value}")
            print(f"   - 问题数量: {len(health_result.issues)}")
            print(f"   - 建议数量: {len(health_result.recommendations)}")
        
        # 获取性能指标
        print("\n6. 获取性能指标...")
        perf_metrics = await aggregator.api.get_performance_metrics(config.metadata.id)
        if perf_metrics:
            print(f"   - 访问次数: {perf_metrics.access_count}")
            print(f"   - 平均响应时间: {perf_metrics.average_response_time:.3f}秒")
            print(f"   - 成功率: {perf_metrics.success_rate:.1f}%")
        
        # 同步配置
        print("\n7. 同步配置...")
        sync_result = await aggregator.api.synchronize_configuration(
            config.metadata.id, 
            "last_write_wins"
        )
        print(f"   - 同步状态: {sync_result.status.value}")
        
        # 获取系统统计
        print("\n8. 获取系统统计...")
        stats = aggregator.api.get_system_statistics()
        print(f"   - 总配置数: {stats['monitor']['total_configurations']}")
        print(f"   - 活跃配置数: {stats['monitor']['active_configurations']}")
        print(f"   - 总变更数: {stats['lifecycle']['total_changes']}")
        
        # 模拟一些访问以生成性能数据
        print("\n9. 模拟配置访问...")
        for i in range(5):
            await aggregator.api.get_configuration(config.metadata.id)
        
        # 获取更新后的性能指标
        updated_perf = await aggregator.api.get_performance_metrics(config.metadata.id)
        if updated_perf:
            print(f"   - 更新后访问次数: {updated_perf.access_count}")
            print(f"   - 更新后平均响应时间: {updated_perf.average_response_time:.3f}秒")
        
        print("\n=== 示例运行完成 ===")
        
    except Exception as e:
        print(f"示例运行出错: {e}")
        
    finally:
        # 停止聚合器
        print("\n停止配置状态聚合器...")
        await aggregator.stop()
        print("聚合器已停止")


async def comprehensive_test():
    """综合测试"""
    print("=== K9配置状态聚合器综合测试 ===\n")
    
    aggregator = ConfigurationStateAggregator(
        config_dir="./test_configs",
        db_path="./test_config_state.db",
        log_level="WARNING"  # 减少测试时的日志输出
    )
    
    try:
        await aggregator.start()
        
        # 测试1: 创建多个配置
        print("测试1: 创建多个配置")
        configs_to_create = []
        
        for i in range(5):
            config_metadata = ConfigurationMetadata(
                id=f"test-config-{i:03d}",
                name=f"测试配置 {i+1}",
                description=f"用于测试的配置 {i+1}",
                config_type=ConfigurationType.CUSTOM,
                version="1.0.0",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                created_by="test_user",
                updated_by="test_user",
                tags=[f"test", f"config-{i}"]
            )
            
            config = Configuration(
                metadata=config_metadata,
                values={
                    "setting_1": ConfigurationValue(
                        key="setting_1",
                        value=f"value_{i}",
                        data_type="string"
                    ),
                    "setting_2": ConfigurationValue(
                        key="setting_2",
                        value=i * 10,
                        data_type="int",
                        validation_rules={"min": 0, "max": 100}
                    ),
                    "setting_3": ConfigurationValue(
                        key="setting_3",
                        value=i % 2 == 0,
                        data_type="bool"
                    )
                },
                status=ConfigurationStatus.ACTIVE,
                environment="test",
                scope="test"
            )
            
            configs_to_create.append(config)
        
        # 批量创建配置
        for config in configs_to_create:
            await aggregator.api.create_configuration(config, "test_user")
        
        print(f"   ✓ 创建了 {len(configs_to_create)} 个配置")
        
        # 测试2: 配置过滤和查询
        print("\n测试2: 配置过滤和查询")
        
        # 按状态过滤
        active_configs = await aggregator.api.list_configurations({"status": "active"})
        print(f"   - 活跃配置数: {len(active_configs)}")
        
        # 按环境过滤
        test_configs = await aggregator.api.list_configurations({"environment": "test"})
        print(f"   - 测试环境配置数: {len(test_configs)}")
        
        # 测试3: 配置更新和历史
        print("\n测试3: 配置更新和历史")
        
        test_config = configs_to_create[0]
        updates = {
            "setting_1": "updated_value",
            "setting_2": 999
        }
        
        await aggregator.api.update_configuration(
            test_config.metadata.id,
            updates,
            "test_user",
            "测试更新操作"
        )
        
        # 获取变更历史
        history = await aggregator.api.get_configuration_history(test_config.metadata.id)
        print(f"   - 变更历史记录数: {len(history)}")
        
        # 测试4: 健康检查
        print("\n测试4: 健康检查")
        
        for config in configs_to_create[:3]:  # 只检查前3个配置
            health_result = await aggregator.api.get_health_status(config.metadata.id)
            if health_result:
                print(f"   - {config.metadata.name}: {health_result.status.value}")
        
        # 测试5: 性能跟踪
        print("\n测试5: 性能跟踪")
        
        # 模拟大量访问
        for _ in range(20):
            await aggregator.api.get_configuration(test_config.metadata.id)
        
        perf_metrics = await aggregator.api.get_performance_metrics(test_config.metadata.id)
        if perf_metrics:
            print(f"   - 访问次数: {perf_metrics.access_count}")
            print(f"   - 平均响应时间: {perf_metrics.average_response_time:.4f}秒")
            print(f"   - 成功率: {perf_metrics.success_rate:.1f}%")
        
        # 测试6: 告警系统
        print("\n测试6: 告警系统")
        
        # 创建一些告警
        await aggregator.alert_system.create_alert(
            test_config.metadata.id,
            AlertSeverity.MEDIUM,
            "测试告警",
            "这是一个测试告警"
        )
        
        await aggregator.alert_system.create_alert(
            configs_to_create[1].metadata.id,
            AlertSeverity.HIGH,
            "高优先级测试告警",
            "这是一个高优先级测试告警"
        )
        
        alerts = await aggregator.api.get_alerts()
        print(f"   - 总告警数: {len(alerts)}")
        
        active_alerts = await aggregator.api.get_alerts(active_only=True)
        print(f"   - 活跃告警数: {len(active_alerts)}")
        
        # 测试7: 系统统计
        print("\n测试7: 系统统计")
        
        stats = aggregator.api.get_system_statistics()
        print(f"   - 配置统计: {stats['monitor']}")
        print(f"   - 生命周期统计: {stats['lifecycle']}")
        print(f"   - 性能统计: {stats['performance']}")
        print(f"   - 健康统计: {stats['health']}")
        print(f"   - 告警统计: {stats['alerts']}")
        
        # 测试8: 聚合器状态
        print("\n测试8: 聚合器状态")
        
        status = aggregator.get_status()
        print(f"   - 运行状态: {status['running']}")
        print(f"   - 启动时间: {status['start_time']}")
        print(f"   - 运行时间: {status['uptime_seconds']:.1f}秒")
        
        print("\n=== 综合测试完成 ===")
        
    except Exception as e:
        print(f"测试出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        await aggregator.stop()


if __name__ == "__main__":
    # 运行使用示例
    print("开始运行K9配置状态聚合器示例...\n")
    
    # 运行基本示例
    asyncio.run(example_usage())
    
    print("\n" + "="*60 + "\n")
    
    # 运行综合测试
    print("开始运行综合测试...\n")
    asyncio.run(comprehensive_test())
    
    print("\n所有测试完成！")