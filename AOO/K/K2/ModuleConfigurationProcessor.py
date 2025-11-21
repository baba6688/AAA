#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
K2模块配置处理器

该模块提供了完整的模块配置管理功能，包括：
- 模块级配置管理（模块参数、模块依赖、模块接口）
- 模块配置模板和默认配置
- 模块配置继承和覆盖机制
- 模块配置动态加载和卸载
- 模块配置依赖解析和验证
- 模块配置热更新和实时生效
- 异步配置处理和批处理
- 完整的错误处理和日志记录

作者: 智能量化系统
版本: 1.0.0
创建时间: 2025-11-06
"""

import asyncio
import json
import logging
import os
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Set, Union, Callable, 
    AsyncGenerator, Generator, TypeVar, Generic,
    Awaitable, Tuple, Protocol, NamedTuple
)
import yaml
import jsonschema
from jsonschema import validate, ValidationError as JsonSchemaValidationError
import pickle
import hashlib
from functools import wraps, lru_cache
import signal
import sys


# 类型别名
ConfigDict = Dict[str, Any]
ModuleId = str
ConfigId = str
TemplateId = str
DependencyId = str
HandlerId = str


class ConfigType(Enum):
    """配置类型枚举"""
    SYSTEM = "system"           # 系统配置
    MODULE = "module"           # 模块配置
    TEMPLATE = "template"       # 模板配置
    USER = "user"              # 用户配置
    DYNAMIC = "dynamic"        # 动态配置
    INHERITED = "inherited"    # 继承配置
    OVERRIDE = "override"      # 覆盖配置


class ConfigStatus(Enum):
    """配置状态枚举"""
    LOADING = "loading"        # 加载中
    LOADED = "loaded"          # 已加载
    ACTIVE = "active"          # 激活状态
    INACTIVE = "inactive"      # 非激活状态
    UPDATING = "updating"      # 更新中
    ERROR = "error"            # 错误状态
    DEPRECATED = "deprecated"  # 已废弃


class ConfigEvent(Enum):
    """配置事件类型"""
    LOAD_START = "load_start"
    LOAD_SUCCESS = "load_success"
    LOAD_FAILURE = "load_failure"
    UPDATE_START = "update_start"
    UPDATE_SUCCESS = "update_success"
    UPDATE_FAILURE = "update_failure"
    VALIDATION_START = "validation_start"
    VALIDATION_SUCCESS = "validation_success"
    VALIDATION_FAILURE = "validation_failure"
    DEPENDENCY_RESOLVED = "dependency_resolved"
    DEPENDENCY_FAILED = "dependency_failed"
    HOT_UPDATE = "hot_update"
    TEMPLATE_APPLIED = "template_applied"
    INHERITANCE_CHAIN_UPDATED = "inheritance_chain_updated"


class ConfigPriority(Enum):
    """配置优先级枚举"""
    LOWEST = 0
    LOW = 1
    NORMAL = 2
    HIGH = 3
    HIGHEST = 4
    SYSTEM = 5


class ConfigScope(Enum):
    """配置作用域枚举"""
    GLOBAL = "global"          # 全局配置
    MODULE = "module"          # 模块配置
    INSTANCE = "instance"      # 实例配置
    SESSION = "session"        # 会话配置
    TEMPORARY = "temporary"    # 临时配置


class ConfigError(Exception):
    """配置处理器基础异常类"""
    
    def __init__(self, message: str, config_id: Optional[ConfigId] = None, 
                 cause: Optional[Exception] = None):
        self.message = message
        self.config_id = config_id
        self.cause = cause
        super().__init__(self.message)
    
    def __str__(self):
        base_msg = self.message
        if self.config_id:
            base_msg += f" (配置ID: {self.config_id})"
        if self.cause:
            base_msg += f" (原因: {self.cause})"
        return base_msg


class ConfigValidationError(ConfigError):
    """配置验证错误"""
    pass


class ConfigDependencyError(ConfigError):
    """配置依赖错误"""
    pass


class ConfigLoadError(ConfigError):
    """配置加载错误"""
    pass


class ConfigUpdateError(ConfigError):
    """配置更新错误"""
    pass


class ConfigNotFoundError(ConfigError):
    """配置未找到错误"""
    pass


class ConfigConflictError(ConfigError):
    """配置冲突错误"""
    pass


class ConfigTimeoutError(ConfigError):
    """配置超时错误"""
    pass


class CircularDependencyError(ConfigDependencyError):
    """循环依赖错误"""
    pass


@dataclass
class ConfigMetadata:
    """配置元数据"""
    config_id: ConfigId
    module_id: ModuleId
    config_type: ConfigType
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    version: str = "1.0.0"
    priority: ConfigPriority = ConfigPriority.NORMAL
    scope: ConfigScope = ConfigScope.MODULE
    status: ConfigStatus = ConfigStatus.LOADING
    checksum: Optional[str] = None
    dependencies: Set[DependencyId] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    description: str = ""
    author: str = "系统"
    ttl: Optional[int] = None  # 生存时间（秒）
    last_accessed: Optional[datetime] = None
    access_count: int = 0
    
    def __post_init__(self):
        """初始化后处理"""
        if self.checksum is None:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """计算配置校验和"""
        data = f"{self.config_id}{self.module_id}{self.config_type.value}{self.version}"
        return hashlib.md5(data.encode()).hexdigest()
    
    def update_access_info(self):
        """更新访问信息"""
        self.last_accessed = datetime.now()
        self.access_count += 1


@dataclass
class ModuleConfig:
    """模块配置数据类"""
    config_id: ConfigId
    module_id: ModuleId
    config_data: ConfigDict
    metadata: ConfigMetadata
    template_id: Optional[TemplateId] = None
    parent_config_id: Optional[ConfigId] = None
    override_configs: Dict[ConfigId, ConfigDict] = field(default_factory=dict)
    inherited_configs: Set[ConfigId] = field(default_factory=set)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    default_values: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保配置数据不为空
        if not self.config_data:
            self.config_data = {}
        
        # 合并默认配置
        self._merge_default_values()
        
        # 应用验证规则
        self._apply_validation_rules()
    
    def _merge_default_values(self):
        """合并默认配置值"""
        for key, default_value in self.default_values.items():
            if key not in self.config_data:
                self.config_data[key] = default_value
    
    def _apply_validation_rules(self):
        """应用验证规则"""
        # 这里可以添加具体的验证逻辑
        pass
    
    def get_value(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        return self.config_data.get(key, default)
    
    def set_value(self, key: str, value: Any):
        """设置配置值"""
        self.config_data[key] = value
        self.metadata.updated_at = datetime.now()
    
    def get_effective_config(self) -> ConfigDict:
        """获取生效配置（包含继承和覆盖）"""
        effective_config = {}
        
        # 从继承配置开始
        for inherited_config_id in self.inherited_configs:
            # 这里需要从外部获取继承的配置
            # 暂时使用空字典，实际实现中需要从配置管理器获取
            pass
        
        # 应用当前配置
        effective_config.update(self.config_data)
        
        # 应用覆盖配置
        for override_config_id, override_data in self.override_configs.items():
            effective_config.update(override_data)
        
        return effective_config
    
    def validate(self) -> bool:
        """验证配置"""
        try:
            # 使用JSON Schema验证
            if self.validation_rules:
                validate(instance=self.config_data, schema=self.validation_rules)
            return True
        except JsonSchemaValidationError as e:
            raise ConfigValidationError(f"配置验证失败: {e.message}", self.config_id, e)
    
    def to_dict(self) -> ConfigDict:
        """转换为字典"""
        return {
            "config_id": self.config_id,
            "module_id": self.module_id,
            "config_data": self.config_data,
            "metadata": asdict(self.metadata),
            "template_id": self.template_id,
            "parent_config_id": self.parent_config_id,
            "override_configs": self.override_configs,
            "inherited_configs": list(self.inherited_configs),
            "validation_rules": self.validation_rules,
            "default_values": self.default_values
        }
    
    @classmethod
    def from_dict(cls, data: ConfigDict) -> 'ModuleConfig':
        """从字典创建实例"""
        metadata = ConfigMetadata(**data["metadata"])
        return cls(
            config_id=data["config_id"],
            module_id=data["module_id"],
            config_data=data["config_data"],
            metadata=metadata,
            template_id=data.get("template_id"),
            parent_config_id=data.get("parent_config_id"),
            override_configs=data.get("override_configs", {}),
            inherited_configs=set(data.get("inherited_configs", [])),
            validation_rules=data.get("validation_rules", {}),
            default_values=data.get("default_values", {})
        )


@dataclass
class ConfigTemplate:
    """配置模板"""
    template_id: TemplateId
    name: str
    description: str
    template_data: ConfigDict
    validation_schema: Optional[Dict[str, Any]] = None
    default_values: Dict[str, Any] = field(default_factory=dict)
    required_fields: List[str] = field(default_factory=list)
    optional_fields: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    version: str = "1.0.0"
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: Optional[datetime] = None
    author: str = "系统"
    is_system_template: bool = False
    parent_template_id: Optional[TemplateId] = None
    
    def apply_template(self, config_data: ConfigDict) -> ConfigDict:
        """应用模板到配置数据"""
        result = self.template_data.copy()
        result.update(config_data)
        
        # 验证必需字段
        for field_name in self.required_fields:
            if field_name not in result:
                raise ConfigValidationError(f"模板 {self.template_id} 缺少必需字段: {field_name}")
        
        # 应用验证规则
        if self.validation_schema:
            try:
                validate(instance=result, schema=self.validation_schema)
            except JsonSchemaValidationError as e:
                raise ConfigValidationError(f"模板验证失败: {e.message}", self.template_id, e)
        
        return result
    
    def validate_template(self) -> bool:
        """验证模板本身"""
        if self.validation_schema:
            try:
                validate(instance=self.template_data, schema=self.validation_schema)
                return True
            except JsonSchemaValidationError as e:
                raise ConfigValidationError(f"模板验证失败: {e.message}", self.template_id, e)
        return True


@dataclass
class ConfigDependency:
    """配置依赖关系"""
    dependency_id: DependencyId
    source_config_id: ConfigId
    target_config_id: ConfigId
    dependency_type: str = "requires"  # requires, conflicts, enhances, overrides
    priority: ConfigPriority = ConfigPriority.NORMAL
    conditions: Dict[str, Any] = field(default_factory=dict)
    timeout: Optional[int] = None
    retry_count: int = 3
    created_at: datetime = field(default_factory=datetime.now)
    
    def is_satisfied(self, source_config: ModuleConfig, target_config: ModuleConfig) -> bool:
        """检查依赖是否满足"""
        # 检查条件
        for condition_key, condition_value in self.conditions.items():
            source_value = source_config.get_value(condition_key)
            target_value = target_config.get_value(condition_key)
            
            if source_value != condition_value and target_value != condition_value:
                return False
        
        # 检查状态
        if target_config.metadata.status != ConfigStatus.ACTIVE:
            return False
        
        return True


class ConfigEventHandler(ABC):
    """配置事件处理器抽象基类"""
    
    @abstractmethod
    async def handle_event(self, event_type: ConfigEvent, config_id: ConfigId, 
                          data: Dict[str, Any]) -> None:
        """处理配置事件"""
        pass


class LoggingConfigEventHandler(ConfigEventHandler):
    """日志配置事件处理器"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
    
    async def handle_event(self, event_type: ConfigEvent, config_id: ConfigId, 
                          data: Dict[str, Any]) -> None:
        """记录配置事件日志"""
        self.logger.info(f"配置事件: {event_type.value}, 配置ID: {config_id}, 数据: {data}")


class ConfigValidator:
    """配置验证器"""
    
    def __init__(self):
        self.validation_rules: Dict[str, Dict[str, Any]] = {}
        self.custom_validators: Dict[str, Callable] = {}
    
    def add_validation_rule(self, config_type: str, rules: Dict[str, Any]):
        """添加验证规则"""
        self.validation_rules[config_type] = rules
    
    def add_custom_validator(self, name: str, validator_func: Callable):
        """添加自定义验证器"""
        self.custom_validators[name] = validator_func
    
    def validate_config(self, config: ModuleConfig) -> bool:
        """验证配置"""
        try:
            # 基础验证
            config.validate()
            
            # 类型特定验证
            config_type = config.metadata.config_type.value
            if config_type in self.validation_rules:
                validate(instance=config.config_data, schema=self.validation_rules[config_type])
            
            # 自定义验证
            for validator_name, validator_func in self.custom_validators.items():
                validator_func(config)
            
            return True
        except Exception as e:
            raise ConfigValidationError(f"配置验证失败: {e}", config.config_id, e)
    
    def validate_dependency(self, dependency: ConfigDependency, 
                          source_config: ModuleConfig, target_config: ModuleConfig) -> bool:
        """验证依赖关系"""
        if not dependency.is_satisfied(source_config, target_config):
            raise ConfigDependencyError(
                f"依赖关系不满足: {dependency.source_config_id} -> {dependency.target_config_id}"
            )
        return True


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, processor: 'ModuleConfigurationProcessor'):
        self.processor = processor
        self.load_strategies: Dict[str, Callable] = {
            "json": self._load_json,
            "yaml": self._load_yaml,
            "pickle": self._load_pickle,
            "env": self._load_env
        }
    
    async def load_config(self, config_path: str, config_id: ConfigId, 
                         module_id: ModuleId, config_type: ConfigType = ConfigType.MODULE) -> ModuleConfig:
        """异步加载配置"""
        try:
            # 确定加载策略
            file_ext = Path(config_path).suffix.lower()
            load_strategy = self.load_strategies.get(file_ext[1:], self._load_json)
            
            # 加载配置数据
            config_data = await load_strategy(config_path)
            
            # 创建配置对象
            metadata = ConfigMetadata(
                config_id=config_id,
                module_id=module_id,
                config_type=config_type
            )
            
            config = ModuleConfig(
                config_id=config_id,
                module_id=module_id,
                config_data=config_data,
                metadata=metadata
            )
            
            # 验证配置
            validator = ConfigValidator()
            validator.validate_config(config)
            
            return config
            
        except Exception as e:
            raise ConfigLoadError(f"加载配置失败: {config_path}", config_id, e)
    
    async def _load_json(self, config_path: str) -> ConfigDict:
        """加载JSON配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    async def _load_yaml(self, config_path: str) -> ConfigDict:
        """加载YAML配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    async def _load_pickle(self, config_path: str) -> ConfigDict:
        """加载Pickle配置"""
        with open(config_path, 'rb') as f:
            return pickle.load(f)
    
    async def _load_env(self, config_path: str) -> ConfigDict:
        """加载环境变量配置"""
        # 简单的环境变量加载实现
        return {key: value for key, value in os.environ.items() 
                if key.startswith('CONFIG_')}


class ConfigInheritance:
    """配置继承管理器"""
    
    def __init__(self, processor: 'ModuleConfigurationProcessor'):
        self.processor = processor
        self.inheritance_chains: Dict[ConfigId, List[ConfigId]] = {}
        self.inheritance_graph: Dict[ConfigId, Set[ConfigId]] = defaultdict(set)
    
    def add_inheritance(self, child_config_id: ConfigId, parent_config_id: ConfigId):
        """添加继承关系"""
        self.inheritance_graph[child_config_id].add(parent_config_id)
        self._update_inheritance_chain(child_config_id)
    
    def remove_inheritance(self, child_config_id: ConfigId, parent_config_id: ConfigId):
        """移除继承关系"""
        if child_config_id in self.inheritance_graph:
            self.inheritance_graph[child_config_id].discard(parent_config_id)
            self._update_inheritance_chain(child_config_id)
    
    def _update_inheritance_chain(self, config_id: ConfigId):
        """更新继承链"""
        chain = self._build_inheritance_chain(config_id)
        self.inheritance_chains[config_id] = chain
    
    def _build_inheritance_chain(self, config_id: ConfigId) -> List[ConfigId]:
        """构建继承链"""
        chain = []
        visited = set()
        
        def _dfs(current_id: ConfigId):
            if current_id in visited:
                return
            visited.add(current_id)
            
            # 添加父配置
            if current_id in self.inheritance_graph:
                for parent_id in self.inheritance_graph[current_id]:
                    _dfs(parent_id)
                    chain.append(parent_id)
        
        _dfs(config_id)
        return chain
    
    def get_inherited_config(self, config_id: ConfigId) -> ConfigDict:
        """获取继承的配置"""
        inherited_config = {}
        
        if config_id in self.inheritance_chains:
            for parent_config_id in self.inheritance_chains[config_id]:
                try:
                    parent_config = self.processor.get_config(parent_config_id)
                    inherited_config.update(parent_config.get_effective_config())
                except ConfigNotFoundError:
                    continue
        
        return inherited_config
    
    def detect_circular_dependency(self) -> List[List[ConfigId]]:
        """检测循环依赖"""
        cycles = []
        visited = set()
        rec_stack = set()
        
        def _dfs(node: ConfigId, path: List[ConfigId]):
            if node in rec_stack:
                # 找到循环
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            if node in self.inheritance_graph:
                for neighbor in self.inheritance_graph[node]:
                    _dfs(neighbor, path.copy())
            
            rec_stack.remove(node)
        
        for config_id in self.inheritance_graph:
            if config_id not in visited:
                _dfs(config_id, [])
        
        return cycles


class ConfigHotUpdate:
    """配置热更新管理器"""
    
    def __init__(self, processor: 'ModuleConfigurationProcessor'):
        self.processor = processor
        self.update_listeners: Dict[str, List[Callable]] = defaultdict(list)
        self.update_history: deque = deque(maxlen=1000)
        self.update_lock = threading.RLock()
    
    def add_update_listener(self, config_id: ConfigId, callback: Callable):
        """添加更新监听器"""
        self.update_listeners[config_id].append(callback)
    
    def remove_update_listener(self, config_id: ConfigId, callback: Callable):
        """移除更新监听器"""
        if config_id in self.update_listeners:
            try:
                self.update_listeners[config_id].remove(callback)
            except ValueError:
                pass
    
    async def hot_update_config(self, config_id: ConfigId, new_config_data: ConfigDict) -> bool:
        """热更新配置"""
        with self.update_lock:
            try:
                # 获取旧配置
                old_config = self.processor.get_config(config_id)
                
                # 记录更新历史
                update_record = {
                    "config_id": config_id,
                    "old_config": old_config.config_data.copy(),
                    "new_config": new_config_data.copy(),
                    "timestamp": datetime.now(),
                    "update_id": str(uuid.uuid4())
                }
                self.update_history.append(update_record)
                
                # 更新配置
                old_config.config_data.update(new_config_data)
                old_config.metadata.updated_at = datetime.now()
                old_config.metadata.status = ConfigStatus.ACTIVE
                
                # 通知监听器
                await self._notify_update_listeners(config_id, new_config_data)
                
                return True
                
            except Exception as e:
                raise ConfigUpdateError(f"热更新配置失败: {config_id}", config_id, e)
    
    async def _notify_update_listeners(self, config_id: ConfigId, new_config_data: ConfigDict):
        """通知更新监听器"""
        tasks = []
        for callback in self.update_listeners.get(config_id, []):
            if asyncio.iscoroutinefunction(callback):
                tasks.append(callback(config_id, new_config_data))
            else:
                callback(config_id, new_config_data)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def rollback_update(self, update_id: str) -> bool:
        """回滚更新"""
        with self.update_lock:
            for record in reversed(list(self.update_history)):
                if record["update_id"] == update_id:
                    try:
                        config_id = record["config_id"]
                        config = self.processor.get_config(config_id)
                        config.config_data = record["old_config"].copy()
                        config.metadata.updated_at = datetime.now()
                        return True
                    except Exception:
                        continue
            return False
    
    def get_update_history(self, config_id: Optional[ConfigId] = None, 
                          limit: int = 10) -> List[Dict[str, Any]]:
        """获取更新历史"""
        history = list(self.update_history)
        if config_id:
            history = [record for record in history if record["config_id"] == config_id]
        return history[-limit:]


class AsyncConfigProcessor:
    """异步配置处理器"""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.task_results: Dict[str, Any] = {}
        self.task_errors: Dict[str, Exception] = {}
    
    async def process_configs_async(self, configs: List[ModuleConfig], 
                                  process_func: Callable[[ModuleConfig], Any]) -> Dict[ConfigId, Any]:
        """异步处理配置列表"""
        tasks = {}
        
        for config in configs:
            task_id = f"{config.config_id}_{uuid.uuid4().hex[:8]}"
            task = asyncio.create_task(self._process_single_config(config, process_func))
            tasks[task_id] = task
            self.processing_tasks[task_id] = task
        
        # 等待所有任务完成
        results = {}
        for task_id, task in tasks.items():
            try:
                result = await task
                results[task_id] = result
                self.task_results[task_id] = result
            except Exception as e:
                self.task_errors[task_id] = e
                results[task_id] = None
        
        # 清理完成的任务
        for task_id in tasks:
            self.processing_tasks.pop(task_id, None)
        
        return results
    
    async def _process_single_config(self, config: ModuleConfig, 
                                   process_func: Callable[[ModuleConfig], Any]) -> Any:
        """异步处理单个配置"""
        loop = asyncio.get_event_loop()
        
        if asyncio.iscoroutinefunction(process_func):
            return await process_func(config)
        else:
            return await loop.run_in_executor(self.executor, process_func, config)
    
    async def process_batch_configs(self, configs: List[ModuleConfig], 
                                  batch_size: int = 5,
                                  process_func: Callable[[List[ModuleConfig]], Any] = None) -> List[Any]:
        """批处理配置"""
        results = []
        
        for i in range(0, len(configs), batch_size):
            batch = configs[i:i + batch_size]
            
            if process_func:
                if asyncio.iscoroutinefunction(process_func):
                    batch_result = await process_func(batch)
                else:
                    batch_result = await asyncio.get_event_loop().run_in_executor(
                        self.executor, process_func, batch
                    )
                results.append(batch_result)
            else:
                # 默认处理：并行处理批次中的每个配置
                tasks = [self._process_single_config(config, lambda c: c) for config in batch]
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                results.extend([r for r in batch_results if not isinstance(r, Exception)])
        
        return results
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        if task_id in self.processing_tasks:
            task = self.processing_tasks[task_id]
            task.cancel()
            return True
        return False
    
    def get_task_status(self, task_id: str) -> str:
        """获取任务状态"""
        if task_id not in self.processing_tasks:
            return "not_found"
        
        task = self.processing_tasks[task_id]
        if task.done():
            if task.exception():
                return "failed"
            else:
                return "completed"
        else:
            return "running"
    
    async def shutdown(self):
        """关闭处理器"""
        # 取消所有正在运行的任务
        for task in self.processing_tasks.values():
            task.cancel()
        
        # 等待任务完成
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)
        
        # 关闭线程池
        self.executor.shutdown(wait=True)


class BatchConfigProcessor:
    """批处理配置处理器"""
    
    def __init__(self, processor: 'ModuleConfigurationProcessor'):
        self.processor = processor
        self.batch_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self.batch_results: Dict[str, Any] = {}
        self.batch_errors: Dict[str, Exception] = {}
        self.is_processing = False
        self.batch_size = 10
        self.batch_timeout = 30.0  # 秒
    
    async def add_to_batch(self, operation: str, config_id: ConfigId, 
                          data: Any = None) -> str:
        """添加操作到批处理队列"""
        batch_id = f"batch_{uuid.uuid4().hex[:8]}"
        batch_item = {
            "batch_id": batch_id,
            "operation": operation,
            "config_id": config_id,
            "data": data,
            "timestamp": datetime.now()
        }
        
        await self.batch_queue.put(batch_item)
        return batch_id
    
    async def start_batch_processing(self):
        """启动批处理"""
        if self.is_processing:
            return
        
        self.is_processing = True
        asyncio.create_task(self._batch_processing_loop())
    
    async def stop_batch_processing(self):
        """停止批处理"""
        self.is_processing = False
    
    async def _batch_processing_loop(self):
        """批处理循环"""
        batch_items = []
        
        while self.is_processing:
            try:
                # 尝试获取批次项目
                try:
                    item = await asyncio.wait_for(
                        self.batch_queue.get(), timeout=self.batch_timeout
                    )
                    batch_items.append(item)
                except asyncio.TimeoutError:
                    # 超时，处理当前批次
                    if batch_items:
                        await self._process_batch(batch_items)
                        batch_items = []
                    continue
                
                # 如果达到批次大小，处理批次
                if len(batch_items) >= self.batch_size:
                    await self._process_batch(batch_items)
                    batch_items = []
                    
            except Exception as e:
                logging.error(f"批处理错误: {e}")
                break
        
        # 处理剩余项目
        if batch_items:
            await self._process_batch(batch_items)
    
    async def _process_batch(self, batch_items: List[Dict[str, Any]]):
        """处理批次"""
        if not batch_items:
            return
        
        # 按操作类型分组
        operations = defaultdict(list)
        for item in batch_items:
            operations[item["operation"]].append(item)
        
        # 处理不同类型的操作
        for operation, items in operations.items():
            try:
                if operation == "load":
                    await self._process_load_batch(items)
                elif operation == "update":
                    await self._process_update_batch(items)
                elif operation == "validate":
                    await self._process_validate_batch(items)
                elif operation == "delete":
                    await self._process_delete_batch(items)
            except Exception as e:
                for item in items:
                    self.batch_errors[item["batch_id"]] = e
    
    async def _process_load_batch(self, items: List[Dict[str, Any]]):
        """处理加载批次"""
        load_tasks = []
        for item in items:
            # 这里需要实际的加载逻辑
            load_tasks.append(self.processor.load_config_from_source(item["data"]))
        
        results = await asyncio.gather(*load_tasks, return_exceptions=True)
        
        for item, result in zip(items, results):
            if isinstance(result, Exception):
                self.batch_errors[item["batch_id"]] = result
            else:
                self.batch_results[item["batch_id"]] = result
    
    async def _process_update_batch(self, items: List[Dict[str, Any]]):
        """处理更新批次"""
        update_tasks = []
        for item in items:
            update_tasks.append(
                self.processor.update_config(item["config_id"], item["data"])
            )
        
        results = await asyncio.gather(*update_tasks, return_exceptions=True)
        
        for item, result in zip(items, results):
            if isinstance(result, Exception):
                self.batch_errors[item["batch_id"]] = result
    
    async def _process_validate_batch(self, items: List[Dict[str, Any]]):
        """处理验证批次"""
        for item in items:
            try:
                config = self.processor.get_config(item["config_id"])
                validator = ConfigValidator()
                validator.validate_config(config)
                self.batch_results[item["batch_id"]] = True
            except Exception as e:
                self.batch_errors[item["batch_id"]] = e
    
    async def _process_delete_batch(self, items: List[Dict[str, Any]]):
        """处理删除批次"""
        for item in items:
            try:
                await self.processor.delete_config(item["config_id"])
                self.batch_results[item["batch_id"]] = True
            except Exception as e:
                self.batch_errors[item["batch_id"]] = e
    
    def get_batch_result(self, batch_id: str) -> Tuple[bool, Any]:
        """获取批次结果"""
        if batch_id in self.batch_results:
            return True, self.batch_results[batch_id]
        elif batch_id in self.batch_errors:
            return False, self.batch_errors[batch_id]
        else:
            return False, None
    
    def get_batch_status(self) -> Dict[str, Any]:
        """获取批处理状态"""
        return {
            "is_processing": self.is_processing,
            "queue_size": self.batch_queue.qsize(),
            "pending_batches": len(self.batch_results) + len(self.batch_errors)
        }


class ModuleConfigurationProcessor:
    """
    模块配置处理器主类
    
    提供完整的模块配置管理功能，包括：
    - 配置的增删改查
    - 配置验证和依赖管理
    - 配置继承和覆盖
    - 热更新和实时生效
    - 异步处理和批处理
    - 完整的错误处理和日志记录
    """
    
    def __init__(self, config_dir: str = "configs", 
                 enable_hot_update: bool = True,
                 enable_async_processing: bool = True,
                 enable_batch_processing: bool = True,
                 max_workers: int = 10,
                 log_level: str = "INFO"):
        """
        初始化模块配置处理器
        
        Args:
            config_dir: 配置目录路径
            enable_hot_update: 是否启用热更新
            enable_async_processing: 是否启用异步处理
            enable_batch_processing: 是否启用批处理
            max_workers: 最大工作线程数
            log_level: 日志级别
        """
        # 基本配置
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        # 日志配置
        self._setup_logging(log_level)
        
        # 核心组件
        self.configs: Dict[ConfigId, ModuleConfig] = {}
        self.templates: Dict[TemplateId, ConfigTemplate] = {}
        self.dependencies: Dict[DependencyId, ConfigDependency] = {}
        
        # 组件初始化
        self.validator = ConfigValidator()
        self.loader = ConfigLoader(self)
        self.inheritance = ConfigInheritance(self)
        self.hot_update = ConfigHotUpdate(self) if enable_hot_update else None
        self.async_processor = AsyncConfigProcessor(max_workers) if enable_async_processing else None
        self.batch_processor = BatchConfigProcessor(self) if enable_batch_processing else None
        
        # 事件处理器
        self.event_handlers: List[ConfigEventHandler] = [
            LoggingConfigEventHandler(self.logger)
        ]
        
        # 缓存和索引
        self.config_cache: Dict[ConfigId, ModuleConfig] = {}
        self.module_index: Dict[ModuleId, Set[ConfigId]] = defaultdict(set)
        self.tag_index: Dict[str, Set[ConfigId]] = defaultdict(set)
        
        # 状态管理
        self.is_initialized = False
        self.initialization_lock = asyncio.Lock()
        
        # 统计信息
        self.stats = {
            "configs_loaded": 0,
            "configs_updated": 0,
            "configs_deleted": 0,
            "validations_performed": 0,
            "dependencies_resolved": 0,
            "hot_updates_performed": 0,
            "async_operations": 0,
            "batch_operations": 0,
            "errors_encountered": 0
        }
        
        self.logger.info("模块配置处理器初始化完成")
    
    def _setup_logging(self, log_level: str):
        """设置日志配置"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # 创建文件处理器
        log_file = self.config_dir / "config_processor.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, log_level.upper()))
        
        # 设置格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    async def initialize(self):
        """异步初始化处理器"""
        async with self.initialization_lock:
            if self.is_initialized:
                return
            
            try:
                self.logger.info("开始初始化模块配置处理器...")
                
                # 加载内置模板
                await self._load_builtin_templates()
                
                # 扫描配置文件
                await self._scan_config_files()
                
                # 验证依赖关系
                await self._validate_dependencies()
                
                # 启动批处理
                if self.batch_processor:
                    await self.batch_processor.start_batch_processing()
                
                self.is_initialized = True
                self.logger.info("模块配置处理器初始化完成")
                
            except Exception as e:
                self.logger.error(f"初始化失败: {e}")
                raise ConfigError(f"处理器初始化失败: {e}") from e
    
    async def _load_builtin_templates(self):
        """加载内置模板"""
        # 基础模块模板
        base_template = ConfigTemplate(
            template_id="base_module",
            name="基础模块模板",
            description="所有模块的基础配置模板",
            template_data={
                "version": "1.0.0",
                "enabled": True,
                "debug": False,
                "timeout": 30,
                "retry_count": 3,
                "log_level": "INFO"
            },
            validation_schema={
                "type": "object",
                "properties": {
                    "version": {"type": "string"},
                    "enabled": {"type": "boolean"},
                    "debug": {"type": "boolean"},
                    "timeout": {"type": "integer", "minimum": 1},
                    "retry_count": {"type": "integer", "minimum": 0},
                    "log_level": {"type": "string", "enum": ["DEBUG", "INFO", "WARNING", "ERROR"]}
                },
                "required": ["version", "enabled"]
            }
        )
        
        self.templates[base_template.template_id] = base_template
        
        # 数据处理模块模板
        data_template = ConfigTemplate(
            template_id="data_module",
            name="数据处理模块模板",
            description="数据处理相关模块的配置模板",
            template_data={
                "batch_size": 1000,
                "max_workers": 4,
                "buffer_size": 10000,
                "compression": "gzip",
                "format": "json"
            },
            parent_template_id="base_module",
            validation_schema={
                "type": "object",
                "properties": {
                    "batch_size": {"type": "integer", "minimum": 1},
                    "max_workers": {"type": "integer", "minimum": 1},
                    "buffer_size": {"type": "integer", "minimum": 1},
                    "compression": {"type": "string", "enum": ["gzip", "bz2", "lzma", "none"]},
                    "format": {"type": "string", "enum": ["json", "xml", "csv", "parquet"]}
                }
            }
        )
        
        self.templates[data_template.template_id] = data_template
        
        self.logger.info(f"已加载 {len(self.templates)} 个内置模板")
    
    async def _scan_config_files(self):
        """扫描配置文件"""
        config_files = []
        
        # 扫描JSON文件
        config_files.extend(self.config_dir.glob("**/*.json"))
        
        # 扫描YAML文件
        config_files.extend(self.config_dir.glob("**/*.yaml"))
        config_files.extend(self.config_dir.glob("**/*.yml"))
        
        # 扫描环境变量文件
        env_files = self.config_dir.glob("**/.env")
        config_files.extend(env_files)
        
        self.logger.info(f"发现 {len(config_files)} 个配置文件")
        
        # 异步加载所有配置文件
        load_tasks = []
        for config_file in config_files:
            config_id = config_file.stem
            module_id = config_file.parent.name
            load_tasks.append(
                self.loader.load_config(str(config_file), config_id, module_id)
            )
        
        if load_tasks:
            configs = await asyncio.gather(*load_tasks, return_exceptions=True)
            
            for config in configs:
                if isinstance(config, Exception):
                    self.logger.error(f"加载配置失败: {config}")
                    self.stats["errors_encountered"] += 1
                else:
                    self.configs[config.config_id] = config
                    self._index_config(config)
                    self.stats["configs_loaded"] += 1
    
    def _index_config(self, config: ModuleConfig):
        """索引配置"""
        # 按模块索引
        self.module_index[config.module_id].add(config.config_id)
        
        # 按标签索引
        for tag in config.metadata.tags:
            self.tag_index[tag].add(config.config_id)
    
    async def _validate_dependencies(self):
        """验证依赖关系"""
        # 检测循环依赖
        cycles = self.inheritance.detect_circular_dependency()
        if cycles:
            raise CircularDependencyError(f"检测到循环依赖: {cycles}")
        
        # 验证所有依赖
        for dependency in self.dependencies.values():
            try:
                source_config = self.configs.get(dependency.source_config_id)
                target_config = self.configs.get(dependency.target_config_id)
                
                if source_config and target_config:
                    self.validator.validate_dependency(dependency, source_config, target_config)
                    self.stats["dependencies_resolved"] += 1
                    
            except Exception as e:
                self.logger.warning(f"依赖验证失败 {dependency.dependency_id}: {e}")
                self.stats["errors_encountered"] += 1
    
    # ==================== 配置管理核心方法 ====================
    
    def get_config(self, config_id: ConfigId) -> ModuleConfig:
        """
        获取配置
        
        Args:
            config_id: 配置ID
            
        Returns:
            模块配置对象
            
        Raises:
            ConfigNotFoundError: 配置不存在
        """
        if config_id not in self.configs:
            raise ConfigNotFoundError(f"配置不存在: {config_id}", config_id)
        
        config = self.configs[config_id]
        config.metadata.update_access_info()
        
        return config
    
    def get_configs_by_module(self, module_id: ModuleId) -> List[ModuleConfig]:
        """
        获取指定模块的所有配置
        
        Args:
            module_id: 模块ID
            
        Returns:
            配置列表
        """
        config_ids = self.module_index.get(module_id, set())
        return [self.configs[config_id] for config_id in config_ids if config_id in self.configs]
    
    def get_configs_by_tag(self, tag: str) -> List[ModuleConfig]:
        """
        获取指定标签的所有配置
        
        Args:
            tag: 标签
            
        Returns:
            配置列表
        """
        config_ids = self.tag_index.get(tag, set())
        return [self.configs[config_id] for config_id in config_ids if config_id in self.configs]
    
    def get_configs_by_type(self, config_type: ConfigType) -> List[ModuleConfig]:
        """
        获取指定类型的所有配置
        
        Args:
            config_type: 配置类型
            
        Returns:
            配置列表
        """
        return [config for config in self.configs.values() 
                if config.metadata.config_type == config_type]
    
    async def create_config(self, config_id: ConfigId, module_id: ModuleId,
                          config_data: ConfigDict, template_id: Optional[TemplateId] = None,
                          config_type: ConfigType = ConfigType.MODULE,
                          validation_rules: Optional[Dict[str, Any]] = None,
                          default_values: Optional[Dict[str, Any]] = None) -> ModuleConfig:
        """
        创建新配置
        
        Args:
            config_id: 配置ID
            module_id: 模块ID
            config_data: 配置数据
            template_id: 模板ID
            config_type: 配置类型
            validation_rules: 验证规则
            default_values: 默认值
            
        Returns:
            创建的配置对象
            
        Raises:
            ConfigConflictError: 配置已存在
            ConfigValidationError: 配置验证失败
        """
        if config_id in self.configs:
            raise ConfigConflictError(f"配置已存在: {config_id}", config_id)
        
        try:
            # 应用模板
            if template_id and template_id in self.templates:
                template = self.templates[template_id]
                config_data = template.apply_template(config_data)
            
            # 创建元数据
            metadata = ConfigMetadata(
                config_id=config_id,
                module_id=module_id,
                config_type=config_type
            )
            
            # 创建配置对象
            config = ModuleConfig(
                config_id=config_id,
                module_id=module_id,
                config_data=config_data,
                metadata=metadata,
                template_id=template_id,
                validation_rules=validation_rules or {},
                default_values=default_values or {}
            )
            
            # 验证配置
            self.validator.validate_config(config)
            
            # 存储配置
            self.configs[config_id] = config
            self._index_config(config)
            
            # 发送事件
            await self._emit_event(ConfigEvent.LOAD_SUCCESS, config_id, {"action": "create"})
            
            self.logger.info(f"创建配置成功: {config_id}")
            return config
            
        except Exception as e:
            self.logger.error(f"创建配置失败 {config_id}: {e}")
            self.stats["errors_encountered"] += 1
            raise
    
    async def update_config(self, config_id: ConfigId, config_data: ConfigDict,
                          validate: bool = True) -> ModuleConfig:
        """
        更新配置
        
        Args:
            config_id: 配置ID
            config_data: 新的配置数据
            validate: 是否验证
            
        Returns:
            更新后的配置对象
            
        Raises:
            ConfigNotFoundError: 配置不存在
            ConfigValidationError: 配置验证失败
        """
        if config_id not in self.configs:
            raise ConfigNotFoundError(f"配置不存在: {config_id}", config_id)
        
        config = self.configs[config_id]
        
        try:
            # 记录更新前状态
            old_data = config.config_data.copy()
            
            # 更新配置数据
            config.config_data.update(config_data)
            config.metadata.updated_at = datetime.now()
            
            # 验证配置
            if validate:
                self.validator.validate_config(config)
            
            # 发送更新事件
            await self._emit_event(ConfigEvent.UPDATE_SUCCESS, config_id, {
                "action": "update",
                "old_data": old_data,
                "new_data": config_data
            })
            
            self.stats["configs_updated"] += 1
            self.logger.info(f"更新配置成功: {config_id}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"更新配置失败 {config_id}: {e}")
            self.stats["errors_encountered"] += 1
            raise
    
    async def delete_config(self, config_id: ConfigId, force: bool = False) -> bool:
        """
        删除配置
        
        Args:
            config_id: 配置ID
            force: 是否强制删除（有依赖时也删除）
            
        Returns:
            是否删除成功
            
        Raises:
            ConfigNotFoundError: 配置不存在
            ConfigDependencyError: 存在依赖且未强制删除
        """
        if config_id not in self.configs:
            raise ConfigNotFoundError(f"配置不存在: {config_id}", config_id)
        
        config = self.configs[config_id]
        
        # 检查依赖
        if not force:
            dependent_configs = self._find_dependent_configs(config_id)
            if dependent_configs:
                raise ConfigDependencyError(
                    f"配置 {config_id} 被其他配置依赖: {dependent_configs}",
                    config_id
                )
        
        try:
            # 删除依赖关系
            self._remove_config_dependencies(config_id)
            
            # 删除继承关系
            self.inheritance.remove_inheritance(config_id, "")
            
            # 从索引中移除
            self._remove_from_index(config)
            
            # 删除配置
            del self.configs[config_id]
            
            # 发送删除事件
            await self._emit_event(ConfigEvent.LOAD_FAILURE, config_id, {"action": "delete"})
            
            self.stats["configs_deleted"] += 1
            self.logger.info(f"删除配置成功: {config_id}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"删除配置失败 {config_id}: {e}")
            self.stats["errors_encountered"] += 1
            raise
    
    def _find_dependent_configs(self, config_id: ConfigId) -> List[ConfigId]:
        """查找依赖指定配置的其它配置"""
        dependent = []
        for dep in self.dependencies.values():
            if dep.target_config_id == config_id:
                dependent.append(dep.source_config_id)
        return dependent
    
    def _remove_config_dependencies(self, config_id: ConfigId):
        """移除配置的依赖关系"""
        to_remove = []
        for dep_id, dep in self.dependencies.items():
            if dep.source_config_id == config_id or dep.target_config_id == config_id:
                to_remove.append(dep_id)
        
        for dep_id in to_remove:
            del self.dependencies[dep_id]
    
    def _remove_from_index(self, config: ModuleConfig):
        """从索引中移除配置"""
        # 从模块索引中移除
        if config.module_id in self.module_index:
            self.module_index[config.module_id].discard(config.config_id)
        
        # 从标签索引中移除
        for tag in config.metadata.tags:
            if tag in self.tag_index:
                self.tag_index[tag].discard(config.config_id)
    
    # ==================== 模板管理 ====================
    
    def get_template(self, template_id: TemplateId) -> ConfigTemplate:
        """获取模板"""
        if template_id not in self.templates:
            raise ConfigNotFoundError(f"模板不存在: {template_id}", template_id)
        return self.templates[template_id]
    
    def get_all_templates(self) -> List[ConfigTemplate]:
        """获取所有模板"""
        return list(self.templates.values())
    
    async def create_template(self, template: ConfigTemplate) -> ConfigTemplate:
        """创建模板"""
        if template.template_id in self.templates:
            raise ConfigConflictError(f"模板已存在: {template.template_id}", template.template_id)
        
        # 验证模板
        template.validate_template()
        
        # 存储模板
        self.templates[template.template_id] = template
        
        self.logger.info(f"创建模板成功: {template.template_id}")
        return template
    
    async def update_template(self, template_id: TemplateId, 
                            template_data: ConfigDict) -> ConfigTemplate:
        """更新模板"""
        if template_id not in self.templates:
            raise ConfigNotFoundError(f"模板不存在: {template_id}", template_id)
        
        template = self.templates[template_id]
        
        # 更新模板数据
        template.template_data.update(template_data)
        template.updated_at = datetime.now()
        
        # 验证模板
        template.validate_template()
        
        self.logger.info(f"更新模板成功: {template_id}")
        return template
    
    async def delete_template(self, template_id: TemplateId) -> bool:
        """删除模板"""
        if template_id not in self.templates:
            raise ConfigNotFoundError(f"模板不存在: {template_id}", template_id)
        
        # 检查是否有配置使用此模板
        using_configs = [config for config in self.configs.values() 
                        if config.template_id == template_id]
        
        if using_configs:
            raise ConfigDependencyError(
                f"模板 {template_id} 被 {len(using_configs)} 个配置使用",
                template_id
            )
        
        del self.templates[template_id]
        self.logger.info(f"删除模板成功: {template_id}")
        return True
    
    # ==================== 依赖管理 ====================
    
    async def add_dependency(self, dependency: ConfigDependency) -> bool:
        """添加依赖关系"""
        if dependency.dependency_id in self.dependencies:
            raise ConfigConflictError(f"依赖关系已存在: {dependency.dependency_id}", 
                                    dependency.dependency_id)
        
        # 验证依赖关系
        source_config = self.configs.get(dependency.source_config_id)
        target_config = self.configs.get(dependency.target_config_id)
        
        if not source_config or not target_config:
            raise ConfigNotFoundError("依赖的配置文件不存在")
        
        self.validator.validate_dependency(dependency, source_config, target_config)
        
        # 添加依赖
        self.dependencies[dependency.dependency_id] = dependency
        
        # 更新配置元数据
        source_config.metadata.dependencies.add(dependency.dependency_id)
        
        self.logger.info(f"添加依赖关系成功: {dependency.dependency_id}")
        return True
    
    async def remove_dependency(self, dependency_id: DependencyId) -> bool:
        """移除依赖关系"""
        if dependency_id not in self.dependencies:
            raise ConfigNotFoundError(f"依赖关系不存在: {dependency_id}", dependency_id)
        
        dependency = self.dependencies[dependency_id]
        
        # 从源配置中移除依赖引用
        source_config = self.configs.get(dependency.source_config_id)
        if source_config:
            source_config.metadata.dependencies.discard(dependency_id)
        
        del self.dependencies[dependency_id]
        
        self.logger.info(f"移除依赖关系成功: {dependency_id}")
        return True
    
    def get_dependencies(self, config_id: ConfigId, 
                        as_source: bool = True, as_target: bool = True) -> List[ConfigDependency]:
        """获取配置的依赖关系"""
        result = []
        
        for dependency in self.dependencies.values():
            if as_source and dependency.source_config_id == config_id:
                result.append(dependency)
            if as_target and dependency.target_config_id == config_id:
                result.append(dependency)
        
        return result
    
    async def resolve_dependencies(self, config_id: ConfigId) -> Dict[ConfigId, bool]:
        """解析配置依赖"""
        dependencies = self.get_dependencies(config_id)
        resolution_results = {}
        
        for dependency in dependencies:
            try:
                source_config = self.configs.get(dependency.source_config_id)
                target_config = self.configs.get(dependency.target_config_id)
                
                if source_config and target_config:
                    is_satisfied = dependency.is_satisfied(source_config, target_config)
                    resolution_results[dependency.dependency_id] = is_satisfied
                    
                    if is_satisfied:
                        await self._emit_event(ConfigEvent.DEPENDENCY_RESOLVED, config_id, {
                            "dependency_id": dependency.dependency_id,
                            "target_config_id": dependency.target_config_id
                        })
                    else:
                        await self._emit_event(ConfigEvent.DEPENDENCY_FAILED, config_id, {
                            "dependency_id": dependency.dependency_id,
                            "target_config_id": dependency.target_config_id
                        })
                        
            except Exception as e:
                self.logger.error(f"解析依赖失败 {dependency.dependency_id}: {e}")
                resolution_results[dependency.dependency_id] = False
        
        return resolution_results
    
    # ==================== 继承管理 ====================
    
    async def add_inheritance(self, child_config_id: ConfigId, parent_config_id: ConfigId):
        """添加继承关系"""
        if child_config_id not in self.configs:
            raise ConfigNotFoundError(f"子配置不存在: {child_config_id}", child_config_id)
        
        if parent_config_id not in self.configs:
            raise ConfigNotFoundError(f"父配置不存在: {parent_config_id}", parent_config_id)
        
        self.inheritance.add_inheritance(child_config_id, parent_config_id)
        
        # 更新子配置
        child_config = self.configs[child_config_id]
        child_config.inherited_configs.add(parent_config_id)
        
        await self._emit_event(ConfigEvent.INHERITANCE_CHAIN_UPDATED, child_config_id, {
            "parent_config_id": parent_config_id,
            "action": "add"
        })
        
        self.logger.info(f"添加继承关系: {child_config_id} -> {parent_config_id}")
    
    async def remove_inheritance(self, child_config_id: ConfigId, parent_config_id: ConfigId):
        """移除继承关系"""
        self.inheritance.remove_inheritance(child_config_id, parent_config_id)
        
        # 更新子配置
        if child_config_id in self.configs:
            child_config = self.configs[child_config_id]
            child_config.inherited_configs.discard(parent_config_id)
        
        await self._emit_event(ConfigEvent.INHERITANCE_CHAIN_UPDATED, child_config_id, {
            "parent_config_id": parent_config_id,
            "action": "remove"
        })
        
        self.logger.info(f"移除继承关系: {child_config_id} -> {parent_config_id}")
    
    def get_inheritance_chain(self, config_id: ConfigId) -> List[ConfigId]:
        """获取继承链"""
        return self.inheritance.inheritance_chains.get(config_id, [])
    
    def get_effective_config(self, config_id: ConfigId) -> ConfigDict:
        """获取生效配置（包含继承和覆盖）"""
        if config_id not in self.configs:
            raise ConfigNotFoundError(f"配置不存在: {config_id}", config_id)
        
        config = self.configs[config_id]
        return config.get_effective_config()
    
    # ==================== 热更新 ====================
    
    async def hot_update(self, config_id: ConfigId, new_config_data: ConfigDict) -> bool:
        """热更新配置"""
        if not self.hot_update:
            raise ConfigError("热更新功能未启用")
        
        if config_id not in self.configs:
            raise ConfigNotFoundError(f"配置不存在: {config_id}", config_id)
        
        try:
            success = await self.hot_update.hot_update_config(config_id, new_config_data)
            
            if success:
                self.stats["hot_updates_performed"] += 1
                await self._emit_event(ConfigEvent.HOT_UPDATE, config_id, {
                    "new_config_data": new_config_data
                })
            
            return success
            
        except Exception as e:
            self.logger.error(f"热更新失败 {config_id}: {e}")
            self.stats["errors_encountered"] += 1
            raise
    
    def add_hot_update_listener(self, config_id: ConfigId, callback: Callable):
        """添加热更新监听器"""
        if self.hot_update:
            self.hot_update.add_update_listener(config_id, callback)
    
    def rollback_hot_update(self, update_id: str) -> bool:
        """回滚热更新"""
        if not self.hot_update:
            raise ConfigError("热更新功能未启用")
        
        return self.hot_update.rollback_update(update_id)
    
    def get_hot_update_history(self, config_id: Optional[ConfigId] = None, 
                             limit: int = 10) -> List[Dict[str, Any]]:
        """获取热更新历史"""
        if not self.hot_update:
            return []
        
        return self.hot_update.get_update_history(config_id, limit)
    
    # ==================== 异步处理 ====================
    
    async def process_configs_async(self, config_ids: List[ConfigId],
                                  process_func: Callable[[ModuleConfig], Any]) -> Dict[ConfigId, Any]:
        """异步处理配置"""
        if not self.async_processor:
            raise ConfigError("异步处理功能未启用")
        
        # 获取配置对象
        configs = []
        for config_id in config_ids:
            if config_id in self.configs:
                configs.append(self.configs[config_id])
            else:
                self.logger.warning(f"配置不存在: {config_id}")
        
        if not configs:
            return {}
        
        results = await self.async_processor.process_configs_async(configs, process_func)
        
        self.stats["async_operations"] += 1
        return results
    
    async def batch_process_configs(self, config_ids: List[ConfigId],
                                  batch_size: int = 5,
                                  process_func: Callable[[List[ModuleConfig]], Any] = None) -> List[Any]:
        """批处理配置"""
        if not self.async_processor:
            raise ConfigError("异步处理功能未启用")
        
        # 获取配置对象
        configs = []
        for config_id in config_ids:
            if config_id in self.configs:
                configs.append(self.configs[config_id])
        
        if not configs:
            return []
        
        results = await self.async_processor.process_batch_configs(configs, batch_size, process_func)
        
        self.stats["batch_operations"] += 1
        return results
    
    def cancel_async_task(self, task_id: str) -> bool:
        """取消异步任务"""
        if not self.async_processor:
            return False
        
        return self.async_processor.cancel_task(task_id)
    
    def get_async_task_status(self, task_id: str) -> str:
        """获取异步任务状态"""
        if not self.async_processor:
            return "not_supported"
        
        return self.async_processor.get_task_status(task_id)
    
    # ==================== 批处理 ====================
    
    async def add_batch_operation(self, operation: str, config_id: ConfigId,
                                data: Any = None) -> str:
        """添加批处理操作"""
        if not self.batch_processor:
            raise ConfigError("批处理功能未启用")
        
        return await self.batch_processor.add_to_batch(operation, config_id, data)
    
    def get_batch_result(self, batch_id: str) -> Tuple[bool, Any]:
        """获取批处理结果"""
        if not self.batch_processor:
            raise ConfigError("批处理功能未启用")
        
        return self.batch_processor.get_batch_result(batch_id)
    
    def get_batch_status(self) -> Dict[str, Any]:
        """获取批处理状态"""
        if not self.batch_processor:
            return {"error": "批处理功能未启用"}
        
        return self.batch_processor.get_batch_status()
    
    # ==================== 验证和检查 ====================
    
    async def validate_all_configs(self) -> Dict[ConfigId, bool]:
        """验证所有配置"""
        validation_results = {}
        
        for config_id, config in self.configs.items():
            try:
                self.validator.validate_config(config)
                validation_results[config_id] = True
                self.stats["validations_performed"] += 1
            except Exception as e:
                validation_results[config_id] = False
                self.logger.error(f"配置验证失败 {config_id}: {e}")
                self.stats["errors_encountered"] += 1
        
        return validation_results
    
    async def validate_config(self, config_id: ConfigId) -> bool:
        """验证指定配置"""
        config = self.get_config(config_id)
        self.validator.validate_config(config)
        self.stats["validations_performed"] += 1
        return True
    
    def check_config_health(self, config_id: ConfigId) -> Dict[str, Any]:
        """检查配置健康状态"""
        config = self.get_config(config_id)
        
        health_status = {
            "config_id": config_id,
            "status": "healthy",
            "checks": {},
            "issues": [],
            "last_check": datetime.now()
        }
        
        # 检查基本属性
        if not config.config_data:
            health_status["issues"].append("配置数据为空")
            health_status["status"] = "warning"
        
        # 检查依赖状态
        dependencies = self.get_dependencies(config_id)
        for dep in dependencies:
            if dep.target_config_id not in self.configs:
                health_status["issues"].append(f"依赖的配置不存在: {dep.target_config_id}")
                health_status["status"] = "error"
        
        # 检查继承链
        try:
            inheritance_chain = self.get_inheritance_chain(config_id)
            # 检查循环依赖
            if config_id in inheritance_chain:
                health_status["issues"].append("存在循环继承")
                health_status["status"] = "error"
        except Exception as e:
            health_status["issues"].append(f"继承链检查失败: {e}")
            health_status["status"] = "warning"
        
        # 检查TTL
        if config.metadata.ttl:
            if config.metadata.updated_at:
                age = datetime.now() - config.metadata.updated_at
                if age.total_seconds() > config.metadata.ttl:
                    health_status["issues"].append("配置已过期")
                    health_status["status"] = "warning"
        
        health_status["checks"] = {
            "has_data": bool(config.config_data),
            "dependencies_valid": len([dep for dep in dependencies 
                                     if dep.target_config_id in self.configs]) == len(dependencies),
            "no_circular_inheritance": config_id not in self.get_inheritance_chain(config_id),
            "not_expired": True  # 简化检查
        }
        
        return health_status
    
    def get_system_health(self) -> Dict[str, Any]:
        """获取系统整体健康状态"""
        total_configs = len(self.configs)
        total_templates = len(self.templates)
        total_dependencies = len(self.dependencies)
        
        # 计算健康配置数量
        healthy_configs = 0
        for config_id in self.configs:
            try:
                health = self.check_config_health(config_id)
                if health["status"] == "healthy":
                    healthy_configs += 1
            except Exception:
                pass
        
        return {
            "timestamp": datetime.now(),
            "total_configs": total_configs,
            "healthy_configs": healthy_configs,
            "total_templates": total_templates,
            "total_dependencies": total_dependencies,
            "health_ratio": healthy_configs / total_configs if total_configs > 0 else 1.0,
            "statistics": self.stats.copy(),
            "features": {
                "hot_update_enabled": self.hot_update is not None,
                "async_processing_enabled": self.async_processor is not None,
                "batch_processing_enabled": self.batch_processor is not None
            }
        }
    
    # ==================== 事件处理 ====================
    
    async def _emit_event(self, event_type: ConfigEvent, config_id: ConfigId, data: Dict[str, Any]):
        """发送配置事件"""
        for handler in self.event_handlers:
            try:
                await handler.handle_event(event_type, config_id, data)
            except Exception as e:
                self.logger.error(f"事件处理器执行失败: {e}")
    
    def add_event_handler(self, handler: ConfigEventHandler):
        """添加事件处理器"""
        self.event_handlers.append(handler)
    
    def remove_event_handler(self, handler: ConfigEventHandler):
        """移除事件处理器"""
        try:
            self.event_handlers.remove(handler)
        except ValueError:
            pass
    
    # ==================== 工具方法 ====================
    
    def list_configs(self, filter_func: Optional[Callable[[ModuleConfig], bool]] = None) -> List[ModuleConfig]:
        """列出配置"""
        configs = list(self.configs.values())
        if filter_func:
            configs = [config for config in configs if filter_func(config)]
        return configs
    
    def search_configs(self, query: str) -> List[ModuleConfig]:
        """搜索配置"""
        results = []
        query_lower = query.lower()
        
        for config in self.configs.values():
            # 在配置ID中搜索
            if query_lower in config.config_id.lower():
                results.append(config)
                continue
            
            # 在模块ID中搜索
            if query_lower in config.module_id.lower():
                results.append(config)
                continue
            
            # 在配置数据中搜索
            config_str = json.dumps(config.config_data, ensure_ascii=False).lower()
            if query_lower in config_str:
                results.append(config)
                continue
            
            # 在标签中搜索
            for tag in config.metadata.tags:
                if query_lower in tag.lower():
                    results.append(config)
                    break
        
        return results
    
    def export_config(self, config_id: ConfigId, format: str = "json") -> str:
        """导出配置"""
        config = self.get_config(config_id)
        
        if format.lower() == "json":
            return json.dumps(config.to_dict(), ensure_ascii=False, indent=2, default=str)
        elif format.lower() == "yaml":
            return yaml.dump(config.to_dict(), allow_unicode=True, default_flow_style=False)
        else:
            raise ValueError(f"不支持的导出格式: {format}")
    
    async def import_config(self, config_data: str, format: str = "json") -> ModuleConfig:
        """导入配置"""
        try:
            if format.lower() == "json":
                data = json.loads(config_data)
            elif format.lower() == "yaml":
                data = yaml.safe_load(config_data)
            else:
                raise ValueError(f"不支持的导入格式: {format}")
            
            config = ModuleConfig.from_dict(data)
            
            # 验证配置
            self.validator.validate_config(config)
            
            # 检查冲突
            if config.config_id in self.configs:
                raise ConfigConflictError(f"配置已存在: {config.config_id}", config.config_id)
            
            # 存储配置
            self.configs[config.config_id] = config
            self._index_config(config)
            
            self.logger.info(f"导入配置成功: {config.config_id}")
            return config
            
        except Exception as e:
            self.logger.error(f"导入配置失败: {e}")
            raise ConfigError(f"导入配置失败: {e}") from e
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            "total_configs": len(self.configs),
            "total_templates": len(self.templates),
            "total_dependencies": len(self.dependencies),
            "modules_count": len(self.module_index),
            "tags_count": len(self.tag_index),
            "cache_size": len(self.config_cache)
        }
    
    # ==================== 生命周期管理 ====================
    
    async def shutdown(self):
        """关闭处理器"""
        self.logger.info("开始关闭模块配置处理器...")
        
        # 停止批处理
        if self.batch_processor:
            await self.batch_processor.stop_batch_processing()
        
        # 关闭异步处理器
        if self.async_processor:
            await self.async_processor.shutdown()
        
        # 保存配置到文件
        await self._save_configs_to_files()
        
        self.is_initialized = False
        self.logger.info("模块配置处理器已关闭")
    
    async def _save_configs_to_files(self):
        """保存配置到文件"""
        for config_id, config in self.configs.items():
            try:
                config_file = self.config_dir / f"{config_id}.json"
                with open(config_file, 'w', encoding='utf-8') as f:
                    json.dump(config.to_dict(), f, ensure_ascii=False, indent=2, default=str)
            except Exception as e:
                self.logger.error(f"保存配置失败 {config_id}: {e}")
    
    async def reload(self):
        """重新加载配置"""
        self.logger.info("开始重新加载配置...")
        
        # 清空现有数据
        self.configs.clear()
        self.templates.clear()
        self.dependencies.clear()
        self.module_index.clear()
        self.tag_index.clear()
        
        # 重新初始化
        await self.initialize()
        
        self.logger.info("配置重新加载完成")
    
    def clear_cache(self):
        """清空缓存"""
        self.config_cache.clear()
        self.logger.info("配置缓存已清空")
    
    def cleanup_expired_configs(self) -> int:
        """清理过期配置"""
        expired_count = 0
        current_time = datetime.now()
        
        to_remove = []
        for config_id, config in self.configs.items():
            if config.metadata.ttl and config.metadata.updated_at:
                age = current_time - config.metadata.updated_at
                if age.total_seconds() > config.metadata.ttl:
                    to_remove.append(config_id)
        
        for config_id in to_remove:
            try:
                asyncio.create_task(self.delete_config(config_id, force=True))
                expired_count += 1
            except Exception as e:
                self.logger.error(f"清理过期配置失败 {config_id}: {e}")
        
        self.logger.info(f"清理了 {expired_count} 个过期配置")
        return expired_count
    
    # ==================== 调试和诊断 ====================
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """获取依赖关系图"""
        graph = {}
        for config_id in self.configs:
            dependencies = self.get_dependencies(config_id, as_source=True, as_target=False)
            graph[config_id] = [dep.target_config_id for dep in dependencies]
        return graph
    
    def detect_issues(self) -> List[Dict[str, Any]]:
        """检测配置问题"""
        issues = []
        
        # 检测循环依赖
        cycles = self.inheritance.detect_circular_dependency()
        for cycle in cycles:
            issues.append({
                "type": "circular_dependency",
                "severity": "error",
                "description": f"发现循环依赖: {' -> '.join(cycle)}",
                "configs_involved": cycle
            })
        
        # 检测孤立的配置
        for config_id, config in self.configs.items():
            if not config.metadata.dependencies and not config.inherited_configs:
                issues.append({
                    "type": "isolated_config",
                    "severity": "warning",
                    "description": f"配置 {config_id} 没有依赖关系",
                    "config_id": config_id
                })
        
        # 检测无效的依赖
        for dependency in self.dependencies.values():
            if dependency.target_config_id not in self.configs:
                issues.append({
                    "type": "invalid_dependency",
                    "severity": "error",
                    "description": f"依赖的目标配置不存在: {dependency.target_config_id}",
                    "dependency_id": dependency.dependency_id
                })
        
        return issues
    
    def export_system_state(self) -> Dict[str, Any]:
        """导出系统状态"""
        return {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.get_statistics(),
            "health": self.get_system_health(),
            "issues": self.detect_issues(),
            "dependency_graph": self.get_dependency_graph(),
            "configurations": {
                config_id: config.to_dict() 
                for config_id, config in self.configs.items()
            },
            "templates": {
                template_id: asdict(template)
                for template_id, template in self.templates.items()
            },
            "dependencies": {
                dep_id: asdict(dep)
                for dep_id, dep in self.dependencies.items()
            }
        }


# ==================== 使用示例 ====================

async def example_usage():
    """使用示例"""
    
    # 创建配置处理器
    processor = ModuleConfigurationProcessor(
        config_dir="./configs",
        enable_hot_update=True,
        enable_async_processing=True,
        enable_batch_processing=True
    )
    
    try:
        # 初始化处理器
        await processor.initialize()
        
        # 创建配置
        config = await processor.create_config(
            config_id="test_config",
            module_id="test_module",
            config_data={
                "name": "测试配置",
                "value": 100,
                "enabled": True
            },
            template_id="base_module"
        )
        
        print(f"创建配置: {config.config_id}")
        
        # 更新配置
        updated_config = await processor.update_config(
            "test_config",
            {"value": 200}
        )
        
        print(f"更新配置: {updated_config.get_value('value')}")
        
        # 热更新
        await processor.hot_update("test_config", {"value": 300})
        print("执行热更新")
        
        # 异步处理配置
        results = await processor.process_configs_async(
            ["test_config"],
            lambda config: config.get_value("value")
        )
        print(f"异步处理结果: {results}")
        
        # 检查健康状态
        health = processor.check_config_health("test_config")
        print(f"健康状态: {health}")
        
        # 获取系统统计
        stats = processor.get_statistics()
        print(f"系统统计: {stats}")
        
    finally:
        # 关闭处理器
        await processor.shutdown()


if __name__ == "__main__":
    # 运行示例
    asyncio.run(example_usage())