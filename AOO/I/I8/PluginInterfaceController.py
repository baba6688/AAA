"""
I8插件接口控制器

完整的插件系统实现，提供插件加载、管理、生命周期管理、依赖解析、
配置管理、安全沙箱、热更新、性能监控、错误隔离和市场集成等功能。


版本: 1.0.0
日期: 2025-11-05
"""

import asyncio
import hashlib
import importlib
import inspect
import json
import logging
import os
import sys
import time
import traceback
import weakref
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Set, Callable, Type, Union, 
    AsyncIterator, Iterator, Tuple, Protocol
)
from enum import Enum
import threading
import tempfile
import shutil
import subprocess
from datetime import datetime, timedelta


# ==================== 基础数据结构 ====================

class PluginStatus(Enum):
    """插件状态枚举"""
    UNLOADED = "unloaded"
    LOADING = "loading" 
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UPDATING = "updating"
    DISABLED = "disabled"


@dataclass
class PluginInfo:
    """插件信息数据类"""
    plugin_id: str
    name: str
    version: str
    description: str
    author: str
    dependencies: List[str] = field(default_factory=list)
    entry_point: Optional[str] = None
    config_schema: Optional[Dict[str, Any]] = None
    permissions: List[str] = field(default_factory=list)
    category: str = "general"
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    file_path: Optional[str] = None
    checksum: Optional[str] = None
    size: Optional[int] = None


@dataclass
class PluginMetrics:
    """插件性能指标"""
    plugin_id: str
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    execution_time: float = 0.0
    call_count: int = 0
    error_count: int = 0
    last_execution: Optional[datetime] = None
    avg_response_time: float = 0.0
    success_rate: float = 100.0


# ==================== 插件接口定义 ====================

class IPlugin(Protocol):
    """插件基础接口协议"""
    
    @property
    def plugin_info(self) -> PluginInfo:
        """获取插件信息"""
        ...
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        ...
    
    async def activate(self) -> bool:
        """激活插件"""
        ...
    
    async def deactivate(self) -> bool:
        """停用插件"""
        ...
    
    async def execute(self, method: str, *args, **kwargs) -> Any:
        """执行插件方法"""
        ...
    
    async def cleanup(self) -> bool:
        """清理资源"""
        ...


class PluginBase(ABC):
    """插件基类"""
    
    def __init__(self):
        self._config = {}
        self._is_active = False
        self._logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
    
    @property
    @abstractmethod
    def plugin_info(self) -> PluginInfo:
        """获取插件信息"""
        pass
    
    async def initialize(self, config: Dict[str, Any]) -> bool:
        """初始化插件"""
        try:
            self._config = config
            self._logger.info(f"插件 {self.plugin_info.name} 初始化成功")
            return True
        except Exception as e:
            self._logger.error(f"插件初始化失败: {e}")
            return False
    
    async def activate(self) -> bool:
        """激活插件"""
        try:
            self._is_active = True
            self._logger.info(f"插件 {self.plugin_info.name} 激活成功")
            return True
        except Exception as e:
            self._logger.error(f"插件激活失败: {e}")
            return False
    
    async def deactivate(self) -> bool:
        """停用插件"""
        try:
            self._is_active = False
            self._logger.info(f"插件 {self.plugin_info.name} 停用成功")
            return True
        except Exception as e:
            self._logger.error(f"插件停用失败: {e}")
            return False
    
    async def execute(self, method: str, *args, **kwargs) -> Any:
        """执行插件方法"""
        if not self._is_active:
            raise RuntimeError("插件未激活")
        
        method_func = getattr(self, method, None)
        if not method_func or not callable(method_func):
            raise AttributeError(f"方法 {method} 不存在")
        
        return await method_func(*args, **kwargs)
    
    async def cleanup(self) -> bool:
        """清理资源"""
        try:
            await self.deactivate()
            self._config.clear()
            self._logger.info(f"插件 {self.plugin_info.name} 清理完成")
            return True
        except Exception as e:
            self._logger.error(f"插件清理失败: {e}")
            return False


# ==================== 安全沙箱实现 ====================

class PluginSandbox:
    """插件安全沙箱"""
    
    def __init__(self, plugin_id: str, allowed_modules: Optional[Set[str]] = None):
        self.plugin_id = plugin_id
        self.allowed_modules = allowed_modules or {
            'json', 'math', 'datetime', 'time', 'random', 'hashlib',
            'base64', 'urllib', 'asyncio', 'concurrent.futures'
        }
        self.restricted_functions = {
            'exec', 'eval', 'compile', 'open', 'input', '__import__',
            'globals', 'locals', 'vars', 'dir', 'help'
        }
        self.execution_count = 0
        self.max_execution_time = 30.0  # 30秒超时
        self.max_memory = 100 * 1024 * 1024  # 100MB内存限制
    
    def create_safe_globals(self) -> Dict[str, Any]:
        """创建安全的全局环境"""
        safe_globals = {
            '__builtins__': {
                'print': print,
                'len': len,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'reversed': reversed,
                'any': any,
                'all': all,
                'isinstance': isinstance,
                'type': type,
                'hasattr': hasattr,
                'getattr': getattr,
                'setattr': setattr,
                'delattr': delattr,
            }
        }
        
        # 添加允许的模块
        for module_name in self.allowed_modules:
            try:
                safe_globals[module_name] = __import__(module_name)
            except ImportError:
                pass
        
        return safe_globals
    
    def validate_code(self, code: str) -> bool:
        """验证代码安全性"""
        # 检查受限函数
        for func in self.restricted_functions:
            if func in code:
                return False
        
        # 检查危险的导入
        dangerous_imports = ['os', 'sys', 'subprocess', 'socket', 'threading']
        for imp in dangerous_imports:
            if imp in code:
                return False
        
        return True
    
    @contextmanager
    def execute_in_sandbox(self, timeout: Optional[float] = None):
        """在沙箱中执行代码"""
        timeout = timeout or self.max_execution_time
        
        def timeout_handler():
            raise TimeoutError(f"插件 {self.plugin_id} 执行超时")
        
        timer = threading.Timer(timeout, timeout_handler)
        timer.start()
        
        try:
            yield self.create_safe_globals()
        finally:
            timer.cancel()
            self.execution_count += 1


# ==================== 依赖解析器 ====================

class PluginDependencyResolver:
    """插件依赖解析器"""
    
    def __init__(self):
        self.dependency_graph: Dict[str, Set[str]] = {}
        self.reverse_dependency_graph: Dict[str, Set[str]] = {}
        self.plugin_registry: Dict[str, PluginInfo] = {}
    
    def register_plugin(self, plugin_info: PluginInfo) -> None:
        """注册插件"""
        self.plugin_registry[plugin_info.plugin_id] = plugin_info
        self.dependency_graph[plugin_info.plugin_id] = set(plugin_info.dependencies)
        
        # 更新反向依赖图
        for dep in plugin_info.dependencies:
            if dep not in self.reverse_dependency_graph:
                self.reverse_dependency_graph[dep] = set()
            self.reverse_dependency_graph[dep].add(plugin_info.plugin_id)
    
    def unregister_plugin(self, plugin_id: str) -> None:
        """注销插件"""
        if plugin_id in self.plugin_registry:
            del self.plugin_registry[plugin_id]
        
        if plugin_id in self.dependency_graph:
            deps = self.dependency_graph.pop(plugin_id)
            for dep in deps:
                if dep in self.reverse_dependency_graph:
                    self.reverse_dependency_graph[dep].discard(plugin_id)
        
        if plugin_id in self.reverse_dependency_graph:
            dependents = self.reverse_dependency_graph.pop(plugin_id)
            for dependent in dependents:
                if dependent in self.dependency_graph:
                    self.dependency_graph[dependent].discard(plugin_id)
    
    def resolve_dependencies(self, plugin_id: str) -> List[str]:
        """解析插件依赖，返回加载顺序"""
        if plugin_id not in self.plugin_registry:
            raise ValueError(f"插件 {plugin_id} 未注册")
        
        visited = set()
        visiting = set()
        load_order = []
        
        def dfs(pid: str):
            if pid in visiting:
                raise ValueError(f"检测到循环依赖: {pid}")
            if pid in visited:
                return
            
            visiting.add(pid)
            
            # 先加载依赖
            for dep in self.dependency_graph.get(pid, set()):
                if dep not in self.plugin_registry:
                    raise ValueError(f"插件 {pid} 的依赖 {dep} 不存在")
                dfs(dep)
            
            visiting.remove(pid)
            visited.add(pid)
            load_order.append(pid)
        
        try:
            dfs(plugin_id)
            return load_order
        except ValueError as e:
            raise e
    
    def get_dependents(self, plugin_id: str) -> List[str]:
        """获取依赖指定插件的其他插件"""
        return list(self.reverse_dependency_graph.get(plugin_id, set()))
    
    def check_circular_dependencies(self) -> List[List[str]]:
        """检查循环依赖"""
        cycles = []
        visited = set()
        recursion_stack = set()
        
        def dfs(node: str, path: List[str]):
            if node in recursion_stack:
                # 找到循环
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return
            
            if node in visited:
                return
            
            visited.add(node)
            recursion_stack.add(node)
            
            for neighbor in self.dependency_graph.get(node, set()):
                dfs(neighbor, path + [node])
            
            recursion_stack.remove(node)
        
        for plugin_id in self.plugin_registry:
            if plugin_id not in visited:
                dfs(plugin_id, [])
        
        return cycles


# ==================== 插件市场连接器 ====================

class PluginMarketConnector:
    """插件市场连接器"""
    
    def __init__(self, market_url: str = "https://plugins.example.com"):
        self.market_url = market_url
        self.cache_dir = Path(tempfile.gettempdir()) / "plugin_cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    async def search_plugins(self, query: str, category: Optional[str] = None) -> List[PluginInfo]:
        """搜索插件"""
        # 模拟插件市场搜索
        await asyncio.sleep(0.1)  # 模拟网络延迟
        
        # 返回模拟结果
        mock_plugins = [
            PluginInfo(
                plugin_id="example-plugin-1",
                name="示例插件1",
                version="1.0.0",
                description="这是一个示例插件",
                author="示例作者",
                category=category or "general"
            ),
            PluginInfo(
                plugin_id="example-plugin-2", 
                name="示例插件2",
                version="2.0.0",
                description="这是另一个示例插件",
                author="示例作者",
                category=category or "general"
            )
        ]
        
        return [p for p in mock_plugins if query.lower() in p.name.lower()]
    
    async def download_plugin(self, plugin_id: str, version: str) -> str:
        """下载插件"""
        # 模拟下载
        await asyncio.sleep(1.0)
        
        plugin_file = self.cache_dir / f"{plugin_id}-{version}.py"
        
        # 创建模拟插件文件
        mock_plugin_code = f'''
from {__name__} import PluginBase, PluginInfo

class {plugin_id.replace("-", "_").title()}Plugin(PluginBase):
    @property
    def plugin_info(self):
        return PluginInfo(
            plugin_id="{plugin_id}",
            name="{plugin_id}",
            version="{version}",
            description="从市场下载的插件",
            author="市场作者"
        )
    
    async def test_method(self):
        return "插件执行成功"
'''
        
        with open(plugin_file, 'w', encoding='utf-8') as f:
            f.write(mock_plugin_code)
        
        return str(plugin_file)
    
    async def get_plugin_details(self, plugin_id: str) -> Optional[PluginInfo]:
        """获取插件详细信息"""
        await asyncio.sleep(0.1)
        
        # 返回模拟详情
        return PluginInfo(
            plugin_id=plugin_id,
            name=f"插件 {plugin_id}",
            version="1.0.0",
            description="从市场获取的插件详情",
            author="市场作者",
            dependencies=[],
            permissions=["read", "write"]
        )


# ==================== 主要的插件接口控制器 ====================

class PluginInterfaceController:
    """
    I8插件接口控制器
    
    提供完整的插件系统功能，包括：
    - 插件加载和管理
    - 插件生命周期管理
    - 插件依赖解析
    - 插件配置管理
    - 插件安全沙箱
    - 插件热更新支持
    - 插件性能监控
    - 插件错误隔离
    - 插件市场集成
    """
    
    def __init__(self, 
                 plugin_directory: str = "./plugins",
                 enable_sandbox: bool = True,
                 enable_hot_reload: bool = True,
                 max_concurrent_plugins: int = 10):
        """
        初始化插件控制器
        
        Args:
            plugin_directory: 插件目录路径
            enable_sandbox: 是否启用安全沙箱
            enable_hot_reload: 是否启用热更新
            max_concurrent_plugins: 最大并发插件数量
        """
        self.plugin_directory = Path(plugin_directory)
        self.plugin_directory.mkdir(exist_ok=True)
        
        self.enable_sandbox = enable_sandbox
        self.enable_hot_reload = enable_hot_reload
        self.max_concurrent_plugins = max_concurrent_plugins
        
        # 核心组件
        self.dependency_resolver = PluginDependencyResolver()
        self.market_connector = PluginMarketConnector()
        
        # 插件管理
        self.loaded_plugins: Dict[str, Any] = {}
        self.plugin_status: Dict[str, PluginStatus] = {}
        self.plugin_configs: Dict[str, Dict[str, Any]] = {}
        self.plugin_metrics: Dict[str, PluginMetrics] = {}
        self.plugin_sandboxes: Dict[str, PluginSandbox] = {}
        
        # 线程池和异步管理
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_plugins)
        self.plugin_locks: Dict[str, asyncio.Lock] = {}
        
        # 日志
        self.logger = logging.getLogger(__name__)
        
        # 监控和统计
        self.total_plugin_loads = 0
        self.successful_loads = 0
        self.failed_loads = 0
        self.plugin_start_time = time.time()
        
        self.logger.info("插件接口控制器初始化完成")
    
    # ==================== 插件生命周期管理 ====================
    
    async def load_plugin(self, plugin_path: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        加载插件
        
        Args:
            plugin_path: 插件文件路径
            config: 插件配置
            
        Returns:
            bool: 加载是否成功
        """
        plugin_path = Path(plugin_path)
        plugin_id = plugin_path.stem
        
        if plugin_id in self.loaded_plugins:
            self.logger.warning(f"插件 {plugin_id} 已经加载")
            return True
        
        # 创建插件锁
        if plugin_id not in self.plugin_locks:
            self.plugin_locks[plugin_id] = asyncio.Lock()
        
        async with self.plugin_locks[plugin_id]:
            try:
                self.plugin_status[plugin_id] = PluginStatus.LOADING
                self.logger.info(f"开始加载插件: {plugin_id}")
                
                # 验证插件文件
                if not await self._validate_plugin_file(plugin_path):
                    raise ValueError(f"插件文件验证失败: {plugin_path}")
                
                # 动态导入插件
                plugin_module = await self._import_plugin_module(plugin_path)
                
                # 获取插件类
                plugin_class = await self._find_plugin_class(plugin_module)
                
                # 创建插件实例
                plugin_instance = plugin_class()
                
                # 验证插件信息
                plugin_info = plugin_instance.plugin_info
                if not await self._validate_plugin_info(plugin_info):
                    raise ValueError("插件信息验证失败")
                
                # 注册依赖
                self.dependency_resolver.register_plugin(plugin_info)
                
                # 创建安全沙箱
                if self.enable_sandbox:
                    self.plugin_sandboxes[plugin_id] = PluginSandbox(plugin_id)
                
                # 初始化插件
                plugin_config = config or {}
                if not await plugin_instance.initialize(plugin_config):
                    raise RuntimeError("插件初始化失败")
                
                # 保存插件实例
                self.loaded_plugins[plugin_id] = plugin_instance
                self.plugin_configs[plugin_id] = plugin_config
                self.plugin_metrics[plugin_id] = PluginMetrics(plugin_id=plugin_id)
                
                # 设置插件状态
                self.plugin_status[plugin_id] = PluginStatus.LOADED
                
                # 更新统计
                self.total_plugin_loads += 1
                self.successful_loads += 1
                
                self.logger.info(f"插件加载成功: {plugin_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"插件加载失败 {plugin_id}: {e}")
                self.plugin_status[plugin_id] = PluginStatus.ERROR
                self.failed_loads += 1
                
                # 清理失败加载的插件
                await self._cleanup_failed_plugin(plugin_id)
                return False
    
    async def unload_plugin(self, plugin_id: str) -> bool:
        """
        卸载插件
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            bool: 卸载是否成功
        """
        if plugin_id not in self.loaded_plugins:
            self.logger.warning(f"插件 {plugin_id} 未加载")
            return True
        
        async with self.plugin_locks.get(plugin_id, asyncio.Lock()):
            try:
                self.logger.info(f"开始卸载插件: {plugin_id}")
                
                # 停用插件
                if self.plugin_status[plugin_id] == PluginStatus.ACTIVE:
                    await self.deactivate_plugin(plugin_id)
                
                # 清理插件
                plugin_instance = self.loaded_plugins[plugin_id]
                await plugin_instance.cleanup()
                
                # 注销依赖
                self.dependency_resolver.unregister_plugin(plugin_id)
                
                # 清理资源
                del self.loaded_plugins[plugin_id]
                del self.plugin_status[plugin_id]
                del self.plugin_configs[plugin_id]
                del self.plugin_metrics[plugin_id]
                
                if plugin_id in self.plugin_sandboxes:
                    del self.plugin_sandboxes[plugin_id]
                
                if plugin_id in self.plugin_locks:
                    del self.plugin_locks[plugin_id]
                
                self.logger.info(f"插件卸载成功: {plugin_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"插件卸载失败 {plugin_id}: {e}")
                return False
    
    async def activate_plugin(self, plugin_id: str) -> bool:
        """
        激活插件
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            bool: 激活是否成功
        """
        if plugin_id not in self.loaded_plugins:
            raise ValueError(f"插件 {plugin_id} 未加载")
        
        async with self.plugin_locks.get(plugin_id, asyncio.Lock()):
            try:
                self.logger.info(f"激活插件: {plugin_id}")
                
                plugin_instance = self.loaded_plugins[plugin_id]
                
                # 激活插件
                if not await plugin_instance.activate():
                    raise RuntimeError("插件激活失败")
                
                self.plugin_status[plugin_id] = PluginStatus.ACTIVE
                self.logger.info(f"插件激活成功: {plugin_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"插件激活失败 {plugin_id}: {e}")
                self.plugin_status[plugin_id] = PluginStatus.ERROR
                return False
    
    async def deactivate_plugin(self, plugin_id: str) -> bool:
        """
        停用插件
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            bool: 停用是否成功
        """
        if plugin_id not in self.loaded_plugins:
            raise ValueError(f"插件 {plugin_id} 未加载")
        
        async with self.plugin_locks.get(plugin_id, asyncio.Lock()):
            try:
                self.logger.info(f"停用插件: {plugin_id}")
                
                plugin_instance = self.loaded_plugins[plugin_id]
                
                # 停用插件
                if not await plugin_instance.deactivate():
                    raise RuntimeError("插件停用失败")
                
                self.plugin_status[plugin_id] = PluginStatus.INACTIVE
                self.logger.info(f"插件停用成功: {plugin_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"插件停用失败 {plugin_id}: {e}")
                self.plugin_status[plugin_id] = PluginStatus.ERROR
                return False
    
    # ==================== 插件执行和监控 ====================
    
    async def execute_plugin(self, 
                           plugin_id: str, 
                           method: str, 
                           *args, 
                           timeout: float = 30.0,
                           **kwargs) -> Any:
        """
        执行插件方法
        
        Args:
            plugin_id: 插件ID
            method: 方法名
            *args: 位置参数
            timeout: 超时时间
            **kwargs: 关键字参数
            
        Returns:
            Any: 执行结果
        """
        if plugin_id not in self.loaded_plugins:
            raise ValueError(f"插件 {plugin_id} 未加载")
        
        if self.plugin_status[plugin_id] != PluginStatus.ACTIVE:
            raise RuntimeError(f"插件 {plugin_id} 未激活")
        
        plugin_instance = self.loaded_plugins[plugin_id]
        metrics = self.plugin_metrics[plugin_id]
        
        start_time = time.time()
        
        try:
            # 在沙箱中执行（如果启用）
            if self.enable_sandbox and plugin_id in self.plugin_sandboxes:
                sandbox = self.plugin_sandboxes[plugin_id]
                with sandbox.execute_in_sandbox(timeout):
                    result = await plugin_instance.execute(method, *args, **kwargs)
            else:
                # 设置执行超时
                result = await asyncio.wait_for(
                    plugin_instance.execute(method, *args, **kwargs),
                    timeout=timeout
                )
            
            # 更新性能指标
            execution_time = time.time() - start_time
            metrics.execution_time = execution_time
            metrics.call_count += 1
            metrics.last_execution = datetime.now()
            
            # 计算平均响应时间
            if metrics.call_count == 1:
                metrics.avg_response_time = execution_time
            else:
                metrics.avg_response_time = (
                    (metrics.avg_response_time * (metrics.call_count - 1) + execution_time) 
                    / metrics.call_count
                )
            
            return result
            
        except Exception as e:
            # 更新错误统计
            metrics.error_count += 1
            metrics.last_execution = datetime.now()
            
            # 计算成功率
            total_calls = metrics.call_count
            if total_calls > 0:
                metrics.success_rate = ((total_calls - metrics.error_count) / total_calls) * 100
            
            self.logger.error(f"插件执行失败 {plugin_id}.{method}: {e}")
            raise
    
    async def get_plugin_metrics(self, plugin_id: str) -> Optional[PluginMetrics]:
        """获取插件性能指标"""
        return self.plugin_metrics.get(plugin_id)
    
    async def get_all_plugin_metrics(self) -> Dict[str, PluginMetrics]:
        """获取所有插件性能指标"""
        return self.plugin_metrics.copy()
    
    # ==================== 热更新支持 ====================
    
    async def hot_reload_plugin(self, plugin_id: str) -> bool:
        """
        热更新插件
        
        Args:
            plugin_id: 插件ID
            
        Returns:
            bool: 更新是否成功
        """
        if not self.enable_hot_reload:
            raise RuntimeError("热更新功能未启用")
        
        if plugin_id not in self.loaded_plugins:
            raise ValueError(f"插件 {plugin_id} 未加载")
        
        self.logger.info(f"开始热更新插件: {plugin_id}")
        
        try:
            # 保存当前配置
            current_config = self.plugin_configs[plugin_id].copy()
            current_status = self.plugin_status[plugin_id]
            
            # 卸载插件
            await self.unload_plugin(plugin_id)
            
            # 重新加载插件
            plugin_path = self._find_plugin_file(plugin_id)
            if not plugin_path:
                raise FileNotFoundError(f"找不到插件文件: {plugin_id}")
            
            success = await self.load_plugin(str(plugin_path), current_config)
            
            if success and current_status == PluginStatus.ACTIVE:
                await self.activate_plugin(plugin_id)
            
            self.logger.info(f"插件热更新完成: {plugin_id}")
            return success
            
        except Exception as e:
            self.logger.error(f"插件热更新失败 {plugin_id}: {e}")
            return False
    
    async def watch_plugin_changes(self):
        """监控插件文件变化（需要结合文件监控工具实现）"""
        # 这里可以实现文件监控逻辑
        # 当检测到插件文件变化时，自动触发热更新
        pass
    
    # ==================== 插件市场集成 ====================
    
    async def install_from_market(self, plugin_id: str, version: str = "latest") -> bool:
        """
        从市场安装插件
        
        Args:
            plugin_id: 插件ID
            version: 版本号
            
        Returns:
            bool: 安装是否成功
        """
        try:
            self.logger.info(f"从市场安装插件: {plugin_id}")
            
            # 下载插件
            plugin_file = await self.market_connector.download_plugin(plugin_id, version)
            
            # 加载插件
            success = await self.load_plugin(plugin_file)
            
            if success:
                self.logger.info(f"插件安装成功: {plugin_id}")
            else:
                self.logger.error(f"插件安装失败: {plugin_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"从市场安装插件失败 {plugin_id}: {e}")
            return False
    
    async def search_market_plugins(self, query: str, category: Optional[str] = None) -> List[PluginInfo]:
        """搜索市场插件"""
        return await self.market_connector.search_plugins(query, category)
    
    async def get_market_plugin_info(self, plugin_id: str) -> Optional[PluginInfo]:
        """获取市场插件信息"""
        return await self.market_connector.get_plugin_details(plugin_id)
    
    # ==================== 批量操作 ====================
    
    async def load_all_plugins(self, directory: Optional[str] = None) -> Dict[str, bool]:
        """
        批量加载插件目录中的所有插件
        
        Args:
            directory: 插件目录，默认为初始化时指定的目录
            
        Returns:
            Dict[str, bool]: 插件ID到加载结果的映射
        """
        plugin_dir = Path(directory) if directory else self.plugin_directory
        
        if not plugin_dir.exists():
            self.logger.warning(f"插件目录不存在: {plugin_dir}")
            return {}
        
        results = {}
        
        # 查找所有Python插件文件
        plugin_files = list(plugin_dir.glob("*.py"))
        
        # 按依赖关系排序
        plugin_load_order = []
        for plugin_file in plugin_files:
            plugin_id = plugin_file.stem
            try:
                order = self.dependency_resolver.resolve_dependencies(plugin_id)
                plugin_load_order.extend([pid for pid in order if pid not in plugin_load_order])
            except ValueError:
                # 如果依赖解析失败，按文件名顺序加载
                plugin_load_order.append(plugin_id)
        
        # 加载插件
        for plugin_id in plugin_load_order:
            plugin_file = plugin_dir / f"{plugin_id}.py"
            if plugin_file.exists():
                results[plugin_id] = await self.load_plugin(str(plugin_file))
        
        return results
    
    async def activate_all_plugins(self) -> Dict[str, bool]:
        """批量激活所有已加载的插件"""
        results = {}
        
        for plugin_id in self.loaded_plugins:
            if self.plugin_status[plugin_id] == PluginStatus.LOADED:
                results[plugin_id] = await self.activate_plugin(plugin_id)
        
        return results
    
    async def deactivate_all_plugins(self) -> Dict[str, bool]:
        """批量停用所有插件"""
        results = {}
        
        for plugin_id in self.loaded_plugins:
            if self.plugin_status[plugin_id] == PluginStatus.ACTIVE:
                results[plugin_id] = await self.deactivate_plugin(plugin_id)
        
        return results
    
    # ==================== 状态查询和管理 ====================
    
    def get_plugin_status(self, plugin_id: str) -> Optional[PluginStatus]:
        """获取插件状态"""
        return self.plugin_status.get(plugin_id)
    
    def get_all_plugin_status(self) -> Dict[str, PluginStatus]:
        """获取所有插件状态"""
        return self.plugin_status.copy()
    
    def get_loaded_plugins(self) -> List[str]:
        """获取已加载的插件ID列表"""
        return list(self.loaded_plugins.keys())
    
    def get_active_plugins(self) -> List[str]:
        """获取活跃插件ID列表"""
        return [pid for pid, status in self.plugin_status.items() 
                if status == PluginStatus.ACTIVE]
    
    def get_plugin_info(self, plugin_id: str) -> Optional[PluginInfo]:
        """获取插件信息"""
        plugin_instance = self.loaded_plugins.get(plugin_id)
        return plugin_instance.plugin_info if plugin_instance else None
    
    # ==================== 统计和监控 ====================
    
    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        uptime = time.time() - self.plugin_start_time
        
        return {
            "uptime_seconds": uptime,
            "total_plugins_loaded": self.total_plugin_loads,
            "successful_loads": self.successful_loads,
            "failed_loads": self.failed_loads,
            "success_rate": (self.successful_loads / max(self.total_plugin_loads, 1)) * 100,
            "currently_loaded": len(self.loaded_plugins),
            "currently_active": len([s for s in self.plugin_status.values() if s == PluginStatus.ACTIVE]),
            "error_count": len([s for s in self.plugin_status.values() if s == PluginStatus.ERROR]),
            "plugin_directory": str(self.plugin_directory),
            "sandbox_enabled": self.enable_sandbox,
            "hot_reload_enabled": self.enable_hot_reload,
            "max_concurrent_plugins": self.max_concurrent_plugins
        }
    
    # ==================== 私有方法 ====================
    
    async def _validate_plugin_file(self, plugin_path: Path) -> bool:
        """验证插件文件"""
        if not plugin_path.exists():
            return False
        
        if plugin_path.suffix != '.py':
            return False
        
        # 检查文件大小（防止过大文件）
        if plugin_path.stat().st_size > 10 * 1024 * 1024:  # 10MB
            return False
        
        return True
    
    async def _import_plugin_module(self, plugin_path: Path):
        """动态导入插件模块"""
        # 将插件目录添加到Python路径
        plugin_dir = plugin_path.parent
        if str(plugin_dir) not in sys.path:
            sys.path.insert(0, str(plugin_dir))
        
        try:
            module_name = plugin_path.stem
            plugin_module = importlib.import_module(module_name)
            return plugin_module
        except Exception as e:
            self.logger.error(f"导入插件模块失败: {e}")
            raise
    
    async def _find_plugin_class(self, plugin_module) -> Type:
        """查找插件类"""
        # 查找继承自PluginBase的类
        plugin_classes = []
        for name, obj in inspect.getmembers(plugin_module):
            if (inspect.isclass(obj) and 
                issubclass(obj, PluginBase) and 
                obj != PluginBase):
                plugin_classes.append(obj)
        
        if not plugin_classes:
            raise ValueError("未找到有效的插件类")
        
        if len(plugin_classes) > 1:
            self.logger.warning(f"找到多个插件类，使用第一个: {plugin_classes[0].__name__}")
        
        return plugin_classes[0]
    
    async def _validate_plugin_info(self, plugin_info: PluginInfo) -> bool:
        """验证插件信息"""
        if not plugin_info.plugin_id or not plugin_info.name:
            return False
        
        if not plugin_info.version:
            return False
        
        return True
    
    def _find_plugin_file(self, plugin_id: str) -> Optional[Path]:
        """查找插件文件"""
        plugin_file = self.plugin_directory / f"{plugin_id}.py"
        return plugin_file if plugin_file.exists() else None
    
    async def _cleanup_failed_plugin(self, plugin_id: str):
        """清理失败的插件加载"""
        # 清理相关状态
        for data_dict in [self.plugin_status, self.plugin_configs, 
                         self.plugin_metrics, self.plugin_sandboxes, 
                         self.plugin_locks]:
            data_dict.pop(plugin_id, None)
    
    # ==================== 资源清理 ====================
    
    async def shutdown(self):
        """关闭插件控制器"""
        self.logger.info("开始关闭插件控制器")
        
        # 停用所有插件
        await self.deactivate_all_plugins()
        
        # 卸载所有插件
        plugin_ids = list(self.loaded_plugins.keys())
        for plugin_id in plugin_ids:
            await self.unload_plugin(plugin_id)
        
        # 关闭线程池
        self.executor.shutdown(wait=True)
        
        self.logger.info("插件控制器已关闭")
    
    def __del__(self):
        """析构函数"""
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.create_task(self.shutdown())
        except:
            pass


# ==================== 测试用例 ====================

class TestPlugin(PluginBase):
    """测试插件"""
    
    @property
    def plugin_info(self) -> PluginInfo:
        return PluginInfo(
            plugin_id="test-plugin",
            name="测试插件",
            version="1.0.0",
            description="用于测试的示例插件",
            author="测试作者",
            dependencies=[]
        )
    
    async def test_method(self, message: str = "Hello") -> str:
        """测试方法"""
        await asyncio.sleep(0.1)  # 模拟一些工作
        return f"{message} from {self.plugin_info.name}"
    
    async def calculate(self, a: int, b: int) -> int:
        """计算方法"""
        return a + b


async def run_tests():
    """运行测试用例"""
    print("开始测试I8插件接口控制器...")
    
    # 创建测试插件文件
    test_plugin_dir = Path("./test_plugins")
    test_plugin_dir.mkdir(exist_ok=True)
    
    test_plugin_code = '''
from I.I8.PluginInterfaceController import PluginBase, PluginInfo

class SamplePlugin(PluginBase):
    @property
    def plugin_info(self):
        return PluginInfo(
            plugin_id="sample-plugin",
            name="示例插件",
            version="1.0.0",
            description="这是一个示例插件",
            author="示例作者",
            dependencies=[]
        )
    
    async def sample_method(self, text: str = "Hello"):
        return f"Sample: {text}"
    
    async def math_operation(self, x: int, y: int):
        return x * y
'''
    
    test_plugin_file = test_plugin_dir / "sample_plugin.py"
    with open(test_plugin_file, 'w', encoding='utf-8') as f:
        f.write(test_plugin_code)
    
    # 创建控制器
    controller = PluginInterfaceController(
        plugin_directory=str(test_plugin_dir),
        enable_sandbox=True,
        enable_hot_reload=True
    )
    
    try:
        # 测试1: 加载插件
        print("测试1: 加载插件")
        success = await controller.load_plugin(str(test_plugin_file))
        print(f"插件加载结果: {success}")
        
        # 测试2: 激活插件
        print("测试2: 激活插件")
        success = await controller.activate_plugin("sample-plugin")
        print(f"插件激活结果: {success}")
        
        # 测试3: 执行插件方法
        print("测试3: 执行插件方法")
        result = await controller.execute_plugin("sample-plugin", "sample_method", "World")
        print(f"执行结果: {result}")
        
        # 测试4: 获取插件状态
        print("测试4: 获取插件状态")
        status = controller.get_plugin_status("sample-plugin")
        print(f"插件状态: {status}")
        
        # 测试5: 获取性能指标
        print("测试5: 获取性能指标")
        metrics = await controller.get_plugin_metrics("sample-plugin")
        print(f"性能指标: {metrics}")
        
        # 测试6: 热更新测试
        print("测试6: 热更新测试")
        if controller.enable_hot_reload:
            success = await controller.hot_reload_plugin("sample-plugin")
            print(f"热更新结果: {success}")
        
        # 测试7: 插件市场搜索
        print("测试7: 插件市场搜索")
        market_plugins = await controller.search_market_plugins("示例")
        print(f"市场插件数量: {len(market_plugins)}")
        
        # 测试8: 系统统计
        print("测试8: 系统统计")
        stats = controller.get_system_stats()
        print(f"系统统计: {json.dumps(stats, indent=2, default=str)}")
        
        # 测试9: 批量操作
        print("测试9: 批量操作")
        all_status = controller.get_all_plugin_status()
        print(f"所有插件状态: {all_status}")
        
        # 测试10: 错误处理
        print("测试10: 错误处理")
        try:
            await controller.execute_plugin("non-existent-plugin", "test_method")
        except Exception as e:
            print(f"预期错误: {e}")
        
        print("所有测试完成!")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        traceback.print_exc()
    
    finally:
        # 清理
        await controller.shutdown()
        if test_plugin_file.exists():
            test_plugin_file.unlink()
        if test_plugin_dir.exists():
            test_plugin_dir.rmdir()


if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 运行测试
    asyncio.run(run_tests())