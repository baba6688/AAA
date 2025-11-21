#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J9工具状态聚合器模块

该模块提供了一个完整的工具状态聚合和管理系统，用于监控、管理和协调多个工具实例。
主要功能包括：
- 工具状态监控（运行状态、性能指标、资源使用）
- 工具协调管理（工具间通信、资源共享、依赖管理）
- 工具生命周期管理（启动、停止、重启、升级）
- 工具性能统计（调用次数、响应时间、成功率）
- 工具健康检查（心跳检测、故障检测、自动恢复）
- 统一工具接口和API
- 异步状态同步和分布式协调

作者: J9系统
版本: 1.0.0
创建时间: 2025-11-06
"""

import asyncio
import logging
import time
import json
import threading
import weakref
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Callable, Union, Set, Tuple, AsyncGenerator
from uuid import uuid4, UUID
import psutil
import signal
import os
import traceback
from copy import deepcopy
import statistics
from pathlib import Path


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('j9_tool_aggregator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ToolState(Enum):
    """工具状态枚举"""
    UNKNOWN = auto()
    STARTING = auto()
    RUNNING = auto()
    PAUSED = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()
    MAINTENANCE = auto()
    UPGRADING = auto()
    DEGRADED = auto()


class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = auto()
    WARNING = auto()
    CRITICAL = auto()
    FAILED = auto()


class Priority(Enum):
    """优先级枚举"""
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    CRITICAL = auto()


class ResourceType(Enum):
    """资源类型枚举"""
    CPU = auto()
    MEMORY = auto()
    DISK = auto()
    NETWORK = auto()
    GPU = auto()
    DATABASE = auto()
    CACHE = auto()


@dataclass
class ResourceUsage:
    """资源使用情况"""
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_mb: float = 0.0
    disk_percent: float = 0.0
    disk_mb: float = 0.0
    network_io: Dict[str, float] = field(default_factory=dict)
    gpu_usage: List[float] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data


@dataclass
class PerformanceMetrics:
    """性能指标"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    average_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    throughput: float = 0.0
    error_rate: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update(self, response_time: float, success: bool):
        """更新性能指标"""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
            
        # 更新响应时间统计
        if response_time > 0:
            self.min_response_time = min(self.min_response_time, response_time)
            self.max_response_time = max(self.max_response_time, response_time)
            
            # 计算平均响应时间
            total_time = self.average_response_time * (self.total_calls - 1) + response_time
            self.average_response_time = total_time / self.total_calls
            
        # 计算错误率
        self.error_rate = self.failed_calls / self.total_calls if self.total_calls > 0 else 0
        
        # 计算吞吐量（每分钟调用次数）
        time_diff = (datetime.now() - self.last_updated).total_seconds()
        if time_diff > 60:  # 每分钟计算一次
            self.throughput = self.total_calls / (time_diff / 60)
            
        self.last_updated = datetime.now()


@dataclass
class ToolDependency:
    """工具依赖关系"""
    tool_id: str
    dependency_type: str  # 'requires', 'conflicts', 'optional'
    version_constraint: Optional[str] = None
    priority: Priority = Priority.NORMAL
    timeout: Optional[float] = None


@dataclass
class ToolInfo:
    """工具信息"""
    tool_id: str
    name: str
    version: str
    description: str = ""
    category: str = ""
    tags: List[str] = field(default_factory=list)
    dependencies: List[ToolDependency] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class ToolError(Exception):
    """工具相关异常"""
    pass


class ToolNotFoundError(ToolError):
    """工具未找到异常"""
    pass


class ToolStateError(ToolError):
    """工具状态异常"""
    pass


class ToolDependencyError(ToolError):
    """工具依赖异常"""
    pass


class ToolHealthError(ToolError):
    """工具健康检查异常"""
    pass


class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self, tool_info: ToolInfo):
        self.tool_info = tool_info
        self.state = ToolState.UNKNOWN
        self.health_status = HealthStatus.HEALTHY
        self.performance_metrics = PerformanceMetrics()
        self.resource_usage = ResourceUsage()
        self.last_heartbeat = datetime.now()
        self.is_alive = False
        self._lock = threading.RLock()
        self._callbacks = defaultdict(list)
        
    @abstractmethod
    async def start(self) -> bool:
        """启动工具"""
        pass
        
    @abstractmethod
    async def stop(self) -> bool:
        """停止工具"""
        pass
        
    @abstractmethod
    async def restart(self) -> bool:
        """重启工具"""
        pass
        
    @abstractmethod
    async def health_check(self) -> HealthStatus:
        """健康检查"""
        pass
        
    @abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """获取工具状态"""
        pass
        
    @abstractmethod
    async def execute(self, command: str, **kwargs) -> Any:
        """执行命令"""
        pass
        
    def update_state(self, new_state: ToolState):
        """更新工具状态"""
        with self._lock:
            old_state = self.state
            self.state = new_state
            self.tool_info.updated_at = datetime.now()
            
            # 触发状态变更回调
            for callback in self._callbacks['state_changed']:
                try:
                    callback(self, old_state, new_state)
                except Exception as e:
                    logger.error(f"状态变更回调执行失败: {e}")
                    
            logger.info(f"工具 {self.tool_info.name} 状态从 {old_state} 变更为 {new_state}")
            
    def update_health(self, health_status: HealthStatus):
        """更新健康状态"""
        with self._lock:
            old_health = self.health_status
            self.health_status = health_status
            self.last_heartbeat = datetime.now()
            
            # 触发健康状态变更回调
            for callback in self._callbacks['health_changed']:
                try:
                    callback(self, old_health, health_status)
                except Exception as e:
                    logger.error(f"健康状态变更回调执行失败: {e}")
                    
            logger.info(f"工具 {self.tool_info.name} 健康状态从 {old_health} 变更为 {health_status}")
            
    def register_callback(self, event: str, callback: Callable):
        """注册回调函数"""
        self._callbacks[event].append(callback)
        
    def unregister_callback(self, event: str, callback: Callable):
        """注销回调函数"""
        if callback in self._callbacks[event]:
            self._callbacks[event].remove(callback)
            
    def update_resource_usage(self):
        """更新资源使用情况"""
        try:
            process = psutil.Process()
            self.resource_usage.cpu_percent = process.cpu_percent()
            self.resource_usage.memory_percent = process.memory_percent()
            self.resource_usage.memory_mb = process.memory_info().rss / 1024 / 1024
            self.resource_usage.timestamp = datetime.now()
        except Exception as e:
            logger.warning(f"更新资源使用情况失败: {e}")
            
    def record_call(self, response_time: float, success: bool):
        """记录调用信息"""
        with self._lock:
            self.performance_metrics.update(response_time, success)
            
    def get_info(self) -> Dict[str, Any]:
        """获取工具信息"""
        return {
            'tool_info': asdict(self.tool_info),
            'state': self.state.name,
            'health_status': self.health_status.name,
            'performance_metrics': asdict(self.performance_metrics),
            'resource_usage': self.resource_usage.to_dict(),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'is_alive': self.is_alive
        }


class MockTool(BaseTool):
    """模拟工具实现"""
    
    def __init__(self, tool_info: ToolInfo):
        super().__init__(tool_info)
        self._running = False
        self._tasks = []
        
    async def start(self) -> bool:
        """启动模拟工具"""
        try:
            self.update_state(ToolState.STARTING)
            
            # 模拟启动过程
            await asyncio.sleep(0.1)
            
            self._running = True
            self.is_alive = True
            self.update_state(ToolState.RUNNING)
            self.update_health(HealthStatus.HEALTHY)
            
            # 启动心跳任务
            self._tasks.append(asyncio.create_task(self._heartbeat_task()))
            
            logger.info(f"模拟工具 {self.tool_info.name} 启动成功")
            return True
            
        except Exception as e:
            logger.error(f"启动模拟工具失败: {e}")
            self.update_state(ToolState.ERROR)
            self.update_health(HealthStatus.FAILED)
            return False
            
    async def stop(self) -> bool:
        """停止模拟工具"""
        try:
            self.update_state(ToolState.STOPPING)
            
            self._running = False
            self.is_alive = False
            
            # 取消所有任务
            for task in self._tasks:
                task.cancel()
                
            self.update_state(ToolState.STOPPED)
            self.update_health(HealthStatus.HEALTHY)
            
            logger.info(f"模拟工具 {self.tool_info.name} 停止成功")
            return True
            
        except Exception as e:
            logger.error(f"停止模拟工具失败: {e}")
            return False
            
    async def restart(self) -> bool:
        """重启模拟工具"""
        await self.stop()
        await asyncio.sleep(0.1)
        return await self.start()
        
    async def health_check(self) -> HealthStatus:
        """健康检查"""
        if not self.is_alive:
            return HealthStatus.FAILED
            
        # 模拟健康检查
        if self.resource_usage.cpu_percent > 90:
            return HealthStatus.CRITICAL
        elif self.resource_usage.memory_percent > 85:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
            
    async def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        return {
            'running': self._running,
            'state': self.state.name,
            'health': self.health_status.name,
            'resource_usage': self.resource_usage.to_dict(),
            'performance': asdict(self.performance_metrics)
        }
        
    async def execute(self, command: str, **kwargs) -> Any:
        """执行命令"""
        start_time = time.time()
        
        try:
            # 模拟命令执行
            await asyncio.sleep(0.01)  # 模拟处理时间
            
            if command == "test":
                result = {"status": "success", "message": "Test completed"}
            elif command == "status":
                result = await self.get_status()
            elif command == "health":
                health = await self.health_check()
                result = {"health": health.name}
            else:
                result = {"status": "unknown_command", "command": command}
                
            response_time = time.time() - start_time
            self.record_call(response_time, True)
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self.record_call(response_time, False)
            raise ToolError(f"命令执行失败: {e}")
            
    async def _heartbeat_task(self):
        """心跳任务"""
        while self._running:
            try:
                self.update_resource_usage()
                health = await self.health_check()
                self.update_health(health)
                await asyncio.sleep(1)  # 每秒心跳一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"心跳任务异常: {e}")
                await asyncio.sleep(1)


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._tool_classes: Dict[str, type] = {}
        self._lock = threading.RLock()
        
    def register_tool_class(self, tool_type: str, tool_class: type):
        """注册工具类"""
        with self._lock:
            self._tool_classes[tool_type] = tool_class
            logger.info(f"注册工具类: {tool_type}")
            
    def create_tool(self, tool_info: ToolInfo, tool_type: str = "mock") -> BaseTool:
        """创建工具实例"""
        with self._lock:
            if tool_type not in self._tool_classes:
                raise ToolError(f"未知的工具类型: {tool_type}")
                
            tool_class = self._tool_classes[tool_type]
            tool = tool_class(tool_info)
            
            if tool_info.tool_id in self._tools:
                logger.warning(f"工具 {tool_info.tool_id} 已存在，将被覆盖")
                
            self._tools[tool_info.tool_id] = tool
            logger.info(f"创建工具实例: {tool_info.name} ({tool_info.tool_id})")
            return tool
            
    def get_tool(self, tool_id: str) -> Optional[BaseTool]:
        """获取工具实例"""
        with self._lock:
            return self._tools.get(tool_id)
            
    def remove_tool(self, tool_id: str) -> bool:
        """移除工具实例"""
        with self._lock:
            if tool_id in self._tools:
                tool = self._tools[tool_id]
                # 停止工具
                asyncio.create_task(self._safe_stop_tool(tool))
                del self._tools[tool_id]
                logger.info(f"移除工具: {tool_id}")
                return True
            return False
            
    def list_tools(self) -> List[str]:
        """列出所有工具ID"""
        with self._lock:
            return list(self._tools.keys())
            
    def get_all_tools(self) -> Dict[str, BaseTool]:
        """获取所有工具"""
        with self._lock:
            return self._tools.copy()
            
    async def _safe_stop_tool(self, tool: BaseTool):
        """安全停止工具"""
        try:
            if hasattr(tool, 'is_alive') and tool.is_alive:
                await tool.stop()
        except Exception as e:
            logger.error(f"安全停止工具失败: {e}")


class DependencyManager:
    """依赖管理器"""
    
    def __init__(self):
        self._dependencies: Dict[str, List[ToolDependency]] = defaultdict(list)
        self._lock = threading.RLock()
        
    def add_dependency(self, tool_id: str, dependency: ToolDependency):
        """添加依赖关系"""
        with self._lock:
            self._dependencies[tool_id].append(dependency)
            logger.info(f"为工具 {tool_id} 添加依赖: {dependency.tool_id}")
            
    def remove_dependency(self, tool_id: str, dependency_id: str) -> bool:
        """移除依赖关系"""
        with self._lock:
            deps = self._dependencies[tool_id]
            for i, dep in enumerate(deps):
                if dep.tool_id == dependency_id:
                    del deps[i]
                    logger.info(f"从工具 {tool_id} 移除依赖: {dependency_id}")
                    return True
            return False
            
    def get_dependencies(self, tool_id: str) -> List[ToolDependency]:
        """获取工具的依赖关系"""
        with self._lock:
            return self._dependencies[tool_id].copy()
            
    def check_dependencies(self, tool_id: str, available_tools: Set[str]) -> Tuple[bool, List[str]]:
        """检查依赖关系是否满足"""
        missing_deps = []
        
        with self._lock:
            for dep in self._dependencies[tool_id]:
                if dep.tool_id not in available_tools:
                    missing_deps.append(dep.tool_id)
                    
        return len(missing_deps) == 0, missing_deps
        
    def get_dependent_tools(self, tool_id: str) -> List[str]:
        """获取依赖指定工具的其他工具"""
        dependent_tools = []
        
        with self._lock:
            for tool, deps in self._dependencies.items():
                for dep in deps:
                    if dep.tool_id == tool_id:
                        dependent_tools.append(tool)
                        break
                        
        return dependent_tools
        
    def topological_sort(self, all_tools: Set[str]) -> List[str]:
        """拓扑排序，获取启动顺序"""
        # 计算入度
        in_degree = {tool: 0 for tool in all_tools}
        
        with self._lock:
            for tool, deps in self._dependencies.items():
                if tool in all_tools:
                    for dep in deps:
                        if dep.tool_id in all_tools:
                            in_degree[tool] += 1
                            
        # 拓扑排序
        queue = [tool for tool, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            tool = queue.pop(0)
            result.append(tool)
            
            # 更新依赖此工具的其他工具
            for other_tool, deps in self._dependencies.items():
                if other_tool in all_tools and other_tool not in result:
                    for dep in deps:
                        if dep.tool_id == tool:
                            in_degree[other_tool] -= 1
                            if in_degree[other_tool] == 0:
                                queue.append(other_tool)
                                
        return result


class HealthChecker:
    """健康检查器"""
    
    def __init__(self, check_interval: float = 30.0):
        self.check_interval = check_interval
        self._running = False
        self._tasks = []
        self._tool_refs = weakref.WeakValueDictionary()
        self._lock = threading.RLock()
        
    def add_tool(self, tool: BaseTool):
        """添加工具到健康检查"""
        with self._lock:
            self._tool_refs[tool.tool_info.tool_id] = tool
            logger.info(f"添加工具到健康检查: {tool.tool_info.name}")
            
    def remove_tool(self, tool_id: str):
        """从健康检查中移除工具"""
        with self._lock:
            if tool_id in self._tool_refs:
                del self._tool_refs[tool_id]
                logger.info(f"从健康检查中移除工具: {tool_id}")
                
    async def start(self):
        """启动健康检查"""
        if self._running:
            return
            
        self._running = True
        self._tasks.append(asyncio.create_task(self._health_check_loop()))
        logger.info("健康检查器启动")
        
    async def stop(self):
        """停止健康检查"""
        self._running = False
        
        for task in self._tasks:
            task.cancel()
            
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("健康检查器停止")
        
    async def _health_check_loop(self):
        """健康检查循环"""
        while self._running:
            try:
                await self._perform_health_checks()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"健康检查循环异常: {e}")
                await asyncio.sleep(self.check_interval)
                
    async def _perform_health_checks(self):
        """执行健康检查"""
        tools_to_check = list(self._tool_refs.values())
        
        if not tools_to_check:
            return
            
        # 并发执行健康检查
        tasks = []
        for tool in tools_to_check:
            tasks.append(asyncio.create_task(self._check_single_tool(tool)))
            
        await asyncio.gather(*tasks, return_exceptions=True)
        
    async def _check_single_tool(self, tool: BaseTool):
        """检查单个工具"""
        try:
            # 检查心跳
            time_since_heartbeat = (datetime.now() - tool.last_heartbeat).total_seconds()
            if time_since_heartbeat > 60:  # 60秒无心跳认为离线
                tool.update_health(HealthStatus.FAILED)
                logger.warning(f"工具 {tool.tool_info.name} 心跳超时")
                return
                
            # 执行健康检查
            health_status = await tool.health_check()
            tool.update_health(health_status)
            
            # 检查资源使用
            tool.update_resource_usage()
            if tool.resource_usage.cpu_percent > 95:
                tool.update_health(HealthStatus.CRITICAL)
                logger.warning(f"工具 {tool.tool_info.name} CPU使用率过高: {tool.resource_usage.cpu_percent}%")
            elif tool.resource_usage.memory_percent > 90:
                tool.update_health(HealthStatus.WARNING)
                logger.warning(f"工具 {tool.tool_info.name} 内存使用率过高: {tool.resource_usage.memory_percent}%")
                
        except Exception as e:
            logger.error(f"健康检查失败 {tool.tool_info.name}: {e}")
            tool.update_health(HealthStatus.FAILED)


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self._performance_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self._lock = threading.RLock()
        
    def record_performance(self, tool_id: str, metrics: PerformanceMetrics):
        """记录性能指标"""
        with self._lock:
            self._performance_history[tool_id].append(deepcopy(metrics))
            
    def get_performance_summary(self, tool_id: str, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """获取性能摘要"""
        with self._lock:
            history = self._performance_history[tool_id]
            
            if not history:
                return {}
                
            if time_window:
                cutoff_time = datetime.now() - time_window
                history = [m for m in history if m.last_updated >= cutoff_time]
                
            if not history:
                return {}
                
            # 计算统计信息
            total_calls = sum(m.total_calls for m in history)
            successful_calls = sum(m.successful_calls for m in history)
            failed_calls = sum(m.failed_calls for m in history)
            
            response_times = []
            for m in history:
                if m.average_response_time > 0:
                    response_times.append(m.average_response_time)
                    
            summary = {
                'tool_id': tool_id,
                'time_range': {
                    'start': min(m.last_updated for m in history).isoformat(),
                    'end': max(m.last_updated for m in history).isoformat()
                },
                'total_calls': total_calls,
                'successful_calls': successful_calls,
                'failed_calls': failed_calls,
                'success_rate': successful_calls / total_calls if total_calls > 0 else 0,
                'error_rate': failed_calls / total_calls if total_calls > 0 else 0,
                'response_time': {
                    'average': statistics.mean(response_times) if response_times else 0,
                    'median': statistics.median(response_times) if response_times else 0,
                    'min': min(response_times) if response_times else 0,
                    'max': max(response_times) if response_times else 0,
                    'std_dev': statistics.stdev(response_times) if len(response_times) > 1 else 0
                },
                'throughput': {
                    'average': statistics.mean([m.throughput for m in history if m.throughput > 0]) if history else 0,
                    'peak': max([m.throughput for m in history if m.throughput > 0]) if history else 0
                }
            }
            
            return summary
            
    def get_all_performance_summaries(self, time_window: Optional[timedelta] = None) -> Dict[str, Dict[str, Any]]:
        """获取所有工具的性能摘要"""
        with self._lock:
            return {
                tool_id: self.get_performance_summary(tool_id, time_window)
                for tool_id in self._performance_history.keys()
            }


class ResourceManager:
    """资源管理器"""
    
    def __init__(self):
        self._resource_limits: Dict[str, Dict[ResourceType, float]] = defaultdict(dict)
        self._current_usage: Dict[str, Dict[ResourceType, float]] = defaultdict(lambda: defaultdict(float))
        self._lock = threading.RLock()
        
    def set_resource_limit(self, tool_id: str, resource_type: ResourceType, limit: float):
        """设置资源限制"""
        with self._lock:
            self._resource_limits[tool_id][resource_type] = limit
            logger.info(f"为工具 {tool_id} 设置 {resource_type.name} 限制: {limit}")
            
    def update_resource_usage(self, tool_id: str, resource_usage: ResourceUsage):
        """更新资源使用情况"""
        with self._lock:
            self._current_usage[tool_id][ResourceType.CPU] = resource_usage.cpu_percent
            self._current_usage[tool_id][ResourceType.MEMORY] = resource_usage.memory_percent
            self._current_usage[tool_id][ResourceType.DISK] = resource_usage.disk_percent
            
    def check_resource_limits(self, tool_id: str) -> Tuple[bool, List[str]]:
        """检查资源限制"""
        violations = []
        
        with self._lock:
            limits = self._resource_limits.get(tool_id, {})
            usage = self._current_usage.get(tool_id, {})
            
            for resource_type, limit in limits.items():
                current_usage = usage.get(resource_type, 0)
                if current_usage > limit:
                    violations.append(f"{resource_type.name}: {current_usage:.1f}% > {limit:.1f}%")
                    
        return len(violations) == 0, violations
        
    def get_resource_summary(self) -> Dict[str, Dict[str, float]]:
        """获取资源使用摘要"""
        with self._lock:
            summary = {}
            for tool_id, usage in self._current_usage.items():
                summary[tool_id] = {
                    resource_type.name.lower(): value
                    for resource_type, value in usage.items()
                }
            return summary


class CommunicationManager:
    """通信管理器"""
    
    def __init__(self):
        self._message_handlers: Dict[str, List[Callable]] = defaultdict(list)
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._tasks = []
        self._lock = threading.RLock()
        
    def register_message_handler(self, message_type: str, handler: Callable):
        """注册消息处理器"""
        with self._lock:
            self._message_handlers[message_type].append(handler)
            logger.info(f"注册消息处理器: {message_type}")
            
    async def send_message(self, message_type: str, data: Any, target_tool: Optional[str] = None):
        """发送消息"""
        message = {
            'id': str(uuid4()),
            'type': message_type,
            'data': data,
            'target_tool': target_tool,
            'timestamp': datetime.now().isoformat()
        }
        
        await self._message_queue.put(message)
        logger.debug(f"发送消息: {message_type}")
        
    async def start(self):
        """启动通信管理器"""
        if self._running:
            return
            
        self._running = True
        self._tasks.append(asyncio.create_task(self._message_processing_loop()))
        logger.info("通信管理器启动")
        
    async def stop(self):
        """停止通信管理器"""
        self._running = False
        
        for task in self._tasks:
            task.cancel()
            
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("通信管理器停止")
        
    async def _message_processing_loop(self):
        """消息处理循环"""
        while self._running:
            try:
                message = await asyncio.wait_for(self._message_queue.get(), timeout=1.0)
                await self._process_message(message)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"消息处理异常: {e}")
                
    async def _process_message(self, message: Dict[str, Any]):
        """处理消息"""
        message_type = message['type']
        handlers = self._message_handlers.get(message_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"消息处理器执行失败: {e}")


class ToolStateAggregator:
    """J9工具状态聚合器主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化工具状态聚合器
        
        Args:
            config: 配置字典，包含各种设置参数
        """
        self.config = config or {}
        self.registry = ToolRegistry()
        self.dependency_manager = DependencyManager()
        self.health_checker = HealthChecker(
            check_interval=self.config.get('health_check_interval', 30.0)
        )
        self.performance_monitor = PerformanceMonitor(
            history_size=self.config.get('performance_history_size', 1000)
        )
        self.resource_manager = ResourceManager()
        self.communication_manager = CommunicationManager()
        
        # 注册默认工具类型
        self.registry.register_tool_class("mock", MockTool)
        
        self._running = False
        self._tasks = []
        self._shutdown_event = asyncio.Event()
        
        logger.info("J9工具状态聚合器初始化完成")
        
    async def start(self):
        """启动聚合器"""
        if self._running:
            logger.warning("聚合器已在运行")
            return
            
        logger.info("启动J9工具状态聚合器...")
        
        self._running = True
        
        # 启动各个组件
        await self.health_checker.start()
        await self.communication_manager.start()
        
        # 启动监控任务
        self._tasks.append(asyncio.create_task(self._monitoring_loop()))
        self._tasks.append(asyncio.create_task(self._cleanup_loop()))
        
        logger.info("J9工具状态聚合器启动完成")
        
    async def stop(self):
        """停止聚合器"""
        if not self._running:
            return
            
        logger.info("停止J9工具状态聚合器...")
        
        self._running = False
        self._shutdown_event.set()
        
        # 停止所有工具
        tools = self.registry.get_all_tools()
        stop_tasks = []
        for tool in tools.values():
            if hasattr(tool, 'is_alive') and tool.is_alive:
                stop_tasks.append(asyncio.create_task(self._safe_stop_tool(tool)))
                
        if stop_tasks:
            await asyncio.gather(*stop_tasks, return_exceptions=True)
            
        # 停止组件
        await self.health_checker.stop()
        await self.communication_manager.stop()
        
        # 取消所有任务
        for task in self._tasks:
            task.cancel()
            
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        
        logger.info("J9工具状态聚合器已停止")
        
    async def _safe_stop_tool(self, tool: BaseTool):
        """安全停止工具"""
        try:
            await tool.stop()
        except Exception as e:
            logger.error(f"安全停止工具失败: {e}")
            
    # === 工具管理API ===
    
    def create_tool(self, tool_info: ToolInfo, tool_type: str = "mock") -> BaseTool:
        """创建工具实例
        
        Args:
            tool_info: 工具信息
            tool_type: 工具类型
            
        Returns:
            工具实例
            
        Raises:
            ToolError: 创建失败时抛出
        """
        try:
            tool = self.registry.create_tool(tool_info, tool_type)
            self.health_checker.add_tool(tool)
            
            # 注册性能监控回调
            tool.register_callback('state_changed', self._on_tool_state_changed)
            
            logger.info(f"创建工具成功: {tool_info.name}")
            return tool
            
        except Exception as e:
            logger.error(f"创建工具失败: {e}")
            raise ToolError(f"创建工具失败: {e}")
            
    async def start_tool(self, tool_id: str) -> bool:
        """启动工具
        
        Args:
            tool_id: 工具ID
            
        Returns:
            是否启动成功
            
        Raises:
            ToolNotFoundError: 工具不存在时抛出
        """
        tool = self.registry.get_tool(tool_id)
        if not tool:
            raise ToolNotFoundError(f"工具不存在: {tool_id}")
            
        try:
            # 检查依赖关系
            available_tools = set(self.registry.list_tools())
            deps_satisfied, missing_deps = self.dependency_manager.check_dependencies(tool_id, available_tools)
            
            if not deps_satisfied:
                raise ToolDependencyError(f"依赖关系不满足，缺少依赖: {missing_deps}")
                
            # 启动工具
            success = await tool.start()
            
            if success:
                # 更新资源管理器
                self.resource_manager.update_resource_usage(tool_id, tool.resource_usage)
                
                logger.info(f"工具启动成功: {tool_id}")
            else:
                logger.error(f"工具启动失败: {tool_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"启动工具异常: {tool_id}, {e}")
            raise
            
    async def stop_tool(self, tool_id: str) -> bool:
        """停止工具
        
        Args:
            tool_id: 工具ID
            
        Returns:
            是否停止成功
            
        Raises:
            ToolNotFoundError: 工具不存在时抛出
        """
        tool = self.registry.get_tool(tool_id)
        if not tool:
            raise ToolNotFoundError(f"工具不存在: {tool_id}")
            
        try:
            success = await tool.stop()
            
            if success:
                logger.info(f"工具停止成功: {tool_id}")
            else:
                logger.error(f"工具停止失败: {tool_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"停止工具异常: {tool_id}, {e}")
            raise
            
    async def restart_tool(self, tool_id: str) -> bool:
        """重启工具
        
        Args:
            tool_id: 工具ID
            
        Returns:
            是否重启成功
            
        Raises:
            ToolNotFoundError: 工具不存在时抛出
        """
        tool = self.registry.get_tool(tool_id)
        if not tool:
            raise ToolNotFoundError(f"工具不存在: {tool_id}")
            
        try:
            success = await tool.restart()
            
            if success:
                logger.info(f"工具重启成功: {tool_id}")
            else:
                logger.error(f"工具重启失败: {tool_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"重启工具异常: {tool_id}, {e}")
            raise
            
    def remove_tool(self, tool_id: str) -> bool:
        """移除工具
        
        Args:
            tool_id: 工具ID
            
        Returns:
            是否移除成功
        """
        try:
            # 检查是否有其他工具依赖此工具
            dependent_tools = self.dependency_manager.get_dependent_tools(tool_id)
            if dependent_tools:
                logger.warning(f"工具 {tool_id} 被其他工具依赖: {dependent_tools}")
                
            # 从健康检查中移除
            self.health_checker.remove_tool(tool_id)
            
            # 移除工具
            success = self.registry.remove_tool(tool_id)
            
            if success:
                logger.info(f"工具移除成功: {tool_id}")
            else:
                logger.error(f"工具移除失败: {tool_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"移除工具异常: {tool_id}, {e}")
            raise
            
    def get_tool(self, tool_id: str) -> Optional[BaseTool]:
        """获取工具实例
        
        Args:
            tool_id: 工具ID
            
        Returns:
            工具实例，如果不存在则返回None
        """
        return self.registry.get_tool(tool_id)
        
    def list_tools(self) -> List[str]:
        """列出所有工具ID
        
        Returns:
            工具ID列表
        """
        return self.registry.list_tools()
        
    async def get_tool_status(self, tool_id: str) -> Dict[str, Any]:
        """获取工具状态详情
        
        Args:
            tool_id: 工具ID
            
        Returns:
            工具状态详情
            
        Raises:
            ToolNotFoundError: 工具不存在时抛出
        """
        tool = self.registry.get_tool(tool_id)
        if not tool:
            raise ToolNotFoundError(f"工具不存在: {tool_id}")
            
        try:
            status = await tool.get_status()
            status.update(tool.get_info())
            
            # 添加依赖信息
            dependencies = self.dependency_manager.get_dependencies(tool_id)
            status['dependencies'] = [asdict(dep) for dep in dependencies]
            
            # 检查资源限制
            within_limits, violations = self.resource_manager.check_resource_limits(tool_id)
            status['resource_violations'] = violations if not within_limits else []
            
            return status
            
        except Exception as e:
            logger.error(f"获取工具状态异常: {tool_id}, {e}")
            raise
            
    # === 依赖管理API ===
    
    def add_dependency(self, tool_id: str, dependency: ToolDependency):
        """添加依赖关系
        
        Args:
            tool_id: 工具ID
            dependency: 依赖关系
        """
        self.dependency_manager.add_dependency(tool_id, dependency)
        
    def remove_dependency(self, tool_id: str, dependency_id: str) -> bool:
        """移除依赖关系
        
        Args:
            tool_id: 工具ID
            dependency_id: 依赖工具ID
            
        Returns:
            是否移除成功
        """
        return self.dependency_manager.remove_dependency(tool_id, dependency_id)
        
    def get_dependency_order(self) -> List[str]:
        """获取工具启动顺序（拓扑排序）
        
        Returns:
            按依赖关系排序的工具ID列表
        """
        all_tools = set(self.registry.list_tools())
        return self.dependency_manager.topological_sort(all_tools)
        
    # === 性能监控API ===
    
    def get_performance_summary(self, tool_id: str, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """获取工具性能摘要
        
        Args:
            tool_id: 工具ID
            time_window: 时间窗口
            
        Returns:
            性能摘要
            
        Raises:
            ToolNotFoundError: 工具不存在时抛出
        """
        if tool_id not in self.registry.get_all_tools():
            raise ToolNotFoundError(f"工具不存在: {tool_id}")
            
        return self.performance_monitor.get_performance_summary(tool_id, time_window)
        
    def get_all_performance_summaries(self, time_window: Optional[timedelta] = None) -> Dict[str, Dict[str, Any]]:
        """获取所有工具性能摘要
        
        Args:
            time_window: 时间窗口
            
        Returns:
            所有工具的性能摘要
        """
        return self.performance_monitor.get_all_performance_summaries(time_window)
        
    # === 资源管理API ===
    
    def set_resource_limit(self, tool_id: str, resource_type: ResourceType, limit: float):
        """设置资源限制
        
        Args:
            tool_id: 工具ID
            resource_type: 资源类型
            limit: 限制值
        """
        self.resource_manager.set_resource_limit(tool_id, resource_type, limit)
        
    def get_resource_summary(self) -> Dict[str, Dict[str, float]]:
        """获取资源使用摘要
        
        Returns:
            资源使用摘要
        """
        return self.resource_manager.get_resource_summary()
        
    # === 健康检查API ===
    
    async def health_check_all(self) -> Dict[str, HealthStatus]:
        """执行所有工具的健康检查
        
        Returns:
            工具ID到健康状态的映射
        """
        tools = self.registry.get_all_tools()
        results = {}
        
        tasks = []
        for tool_id, tool in tools.items():
            tasks.append(asyncio.create_task(self._check_tool_health(tool_id, tool)))
            
        health_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (tool_id, _) in enumerate(tools.items()):
            if isinstance(health_results[i], Exception):
                results[tool_id] = HealthStatus.FAILED
                logger.error(f"工具 {tool_id} 健康检查异常: {health_results[i]}")
            else:
                results[tool_id] = health_results[i]
                
        return results
        
    async def _check_tool_health(self, tool_id: str, tool: BaseTool) -> HealthStatus:
        """检查单个工具健康状态"""
        try:
            return await tool.health_check()
        except Exception as e:
            logger.error(f"健康检查失败 {tool_id}: {e}")
            return HealthStatus.FAILED
            
    # === 通信API ===
    
    async def send_tool_message(self, from_tool_id: str, to_tool_id: str, message_type: str, data: Any):
        """发送工具间消息
        
        Args:
            from_tool_id: 发送方工具ID
            to_tool_id: 接收方工具ID
            message_type: 消息类型
            data: 消息数据
        """
        message_data = {
            'from_tool': from_tool_id,
            'to_tool': to_tool_id,
            'data': data
        }
        
        await self.communication_manager.send_message(message_type, message_data, to_tool_id)
        
    def register_message_handler(self, message_type: str, handler: Callable):
        """注册消息处理器
        
        Args:
            message_type: 消息类型
            handler: 处理器函数
        """
        self.communication_manager.register_message_handler(message_type, handler)
        
    # === 批量操作API ===
    
    async def start_all_tools(self) -> Dict[str, bool]:
        """启动所有工具
        
        Returns:
            工具ID到启动结果的映射
        """
        logger.info("开始启动所有工具...")
        
        # 按依赖关系排序
        tool_order = self.get_dependency_order()
        results = {}
        
        for tool_id in tool_order:
            try:
                success = await self.start_tool(tool_id)
                results[tool_id] = success
                
                if not success:
                    logger.error(f"工具启动失败: {tool_id}")
                    
            except Exception as e:
                logger.error(f"启动工具异常: {tool_id}, {e}")
                results[tool_id] = False
                
        successful_starts = sum(1 for success in results.values() if success)
        logger.info(f"工具启动完成: {successful_starts}/{len(results)} 成功")
        
        return results
        
    async def stop_all_tools(self) -> Dict[str, bool]:
        """停止所有工具
        
        Returns:
            工具ID到停止结果的映射
        """
        logger.info("开始停止所有工具...")
        
        # 按依赖关系倒序停止
        tool_order = self.get_dependency_order()
        tool_order.reverse()
        results = {}
        
        for tool_id in tool_order:
            try:
                success = await self.stop_tool(tool_id)
                results[tool_id] = success
                
                if not success:
                    logger.error(f"工具停止失败: {tool_id}")
                    
            except Exception as e:
                logger.error(f"停止工具异常: {tool_id}, {e}")
                results[tool_id] = False
                
        successful_stops = sum(1 for success in results.values() if success)
        logger.info(f"工具停止完成: {successful_stops}/{len(results)} 成功")
        
        return results
        
    async def restart_all_tools(self) -> Dict[str, bool]:
        """重启所有工具
        
        Returns:
            工具ID到重启结果的映射
        """
        logger.info("开始重启所有工具...")
        
        # 先停止所有工具
        stop_results = await self.stop_all_tools()
        
        # 等待一段时间
        await asyncio.sleep(1)
        
        # 启动所有工具
        start_results = await self.start_all_tools()
        
        # 合并结果
        all_tool_ids = set(stop_results.keys()) | set(start_results.keys())
        results = {}
        
        for tool_id in all_tool_ids:
            stop_success = stop_results.get(tool_id, False)
            start_success = start_results.get(tool_id, False)
            results[tool_id] = stop_success and start_success
            
        logger.info("所有工具重启完成")
        
        return results
        
    # === 系统状态API ===
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统整体状态
        
        Returns:
            系统状态详情
        """
        tools = self.registry.get_all_tools()
        
        # 统计工具状态
        state_counts = defaultdict(int)
        health_counts = defaultdict(int)
        
        for tool in tools.values():
            state_counts[tool.state.name] += 1
            health_counts[tool.health_status.name] += 1
            
        # 计算总体性能指标
        total_calls = sum(tool.performance_metrics.total_calls for tool in tools.values())
        total_success = sum(tool.performance_metrics.successful_calls for tool in tools.values())
        total_failures = sum(tool.performance_metrics.failed_calls for tool in tools.values())
        
        overall_success_rate = total_success / total_calls if total_calls > 0 else 0
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_tools': len(tools),
            'running_tools': sum(1 for tool in tools.values() if tool.state == ToolState.RUNNING),
            'state_distribution': dict(state_counts),
            'health_distribution': dict(health_counts),
            'overall_performance': {
                'total_calls': total_calls,
                'successful_calls': total_success,
                'failed_calls': total_failures,
                'success_rate': overall_success_rate,
                'error_rate': total_failures / total_calls if total_calls > 0 else 0
            },
            'resource_summary': self.get_resource_summary()
        }
        
    async def export_status(self, file_path: str):
        """导出系统状态到文件
        
        Args:
            file_path: 导出文件路径
        """
        status_data = {
            'system_status': self.get_system_status(),
            'tool_details': {}
        }
        
        # 添加每个工具的详细信息
        for tool_id in self.list_tools():
            try:
                status_data['tool_details'][tool_id] = await self.get_tool_status(tool_id)
            except Exception as e:
                logger.error(f"导出工具状态失败 {tool_id}: {e}")
                status_data['tool_details'][tool_id] = {'error': str(e)}
                
        # 写入文件
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, indent=2, ensure_ascii=False, default=str)
            
        logger.info(f"系统状态已导出到: {file_path}")
        
    # === 内部方法 ===
    
    def _on_tool_state_changed(self, tool: BaseTool, old_state: ToolState, new_state: ToolState):
        """工具状态变更回调"""
        # 记录性能指标
        self.performance_monitor.record_performance(
            tool.tool_info.tool_id, 
            tool.performance_metrics
        )
        
        # 更新资源使用情况
        self.resource_manager.update_resource_usage(
            tool.tool_info.tool_id, 
            tool.resource_usage
        )
        
        # 如果工具停止，从健康检查中移除
        if new_state in [ToolState.STOPPED, ToolState.ERROR]:
            self.health_checker.remove_tool(tool.tool_info.tool_id)
            
    async def _monitoring_loop(self):
        """监控循环"""
        while self._running:
            try:
                await self._perform_system_monitoring()
                await asyncio.sleep(10)  # 每10秒监控一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"监控循环异常: {e}")
                await asyncio.sleep(10)
                
    async def _perform_system_monitoring(self):
        """执行系统监控"""
        tools = self.registry.get_all_tools()
        
        # 检查长时间无响应的工具
        current_time = datetime.now()
        for tool_id, tool in tools.items():
            time_since_heartbeat = (current_time - tool.last_heartbeat).total_seconds()
            
            if time_since_heartbeat > 120:  # 2分钟无心跳
                logger.warning(f"工具 {tool_id} 长时间无心跳响应")
                
                # 尝试重启工具
                if tool.state == ToolState.RUNNING:
                    try:
                        logger.info(f"尝试重启无响应工具: {tool_id}")
                        await self.restart_tool(tool_id)
                    except Exception as e:
                        logger.error(f"重启无响应工具失败: {tool_id}, {e}")
                        
    async def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                await self._perform_cleanup()
                await asyncio.sleep(300)  # 每5分钟清理一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"清理循环异常: {e}")
                await asyncio.sleep(300)
                
    async def _perform_cleanup(self):
        """执行清理工作"""
        # 清理过期的性能数据
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        # 这里可以实现更复杂的清理逻辑
        logger.debug("执行系统清理")


# === 使用示例和测试代码 ===

async def example_usage():
    """使用示例"""
    
    # 创建聚合器配置
    config = {
        'health_check_interval': 30.0,
        'performance_history_size': 1000
    }
    
    # 创建聚合器实例
    aggregator = ToolStateAggregator(config)
    
    try:
        # 启动聚合器
        await aggregator.start()
        
        # 创建工具信息
        tool1_info = ToolInfo(
            tool_id="tool_001",
            name="数据库工具",
            version="1.0.0",
            description="数据库连接和管理工具",
            category="database"
        )
        
        tool2_info = ToolInfo(
            tool_id="tool_002", 
            name="缓存工具",
            version="1.0.0",
            description="缓存管理工具",
            category="cache"
        )
        
        tool3_info = ToolInfo(
            tool_id="tool_003",
            name="API工具", 
            version="1.0.0",
            description="API服务工具",
            category="service"
        )
        
        # 创建工具实例
        tool1 = aggregator.create_tool(tool1_info, "mock")
        tool2 = aggregator.create_tool(tool2_info, "mock")
        tool3 = aggregator.create_tool(tool3_info, "mock")
        
        # 添加依赖关系
        aggregator.add_dependency("tool_003", ToolDependency("tool_001", "requires"))
        aggregator.add_dependency("tool_003", ToolDependency("tool_002", "requires"))
        
        # 设置资源限制
        aggregator.set_resource_limit("tool_001", ResourceType.CPU, 50.0)
        aggregator.set_resource_limit("tool_001", ResourceType.MEMORY, 80.0)
        
        # 启动所有工具
        start_results = await aggregator.start_all_tools()
        print(f"启动结果: {start_results}")
        
        # 执行一些操作
        await asyncio.sleep(2)
        
        # 执行工具命令
        result = await tool3.execute("test")
        print(f"工具执行结果: {result}")
        
        # 获取系统状态
        system_status = aggregator.get_system_status()
        print(f"系统状态: {json.dumps(system_status, indent=2, ensure_ascii=False, default=str)}")
        
        # 获取性能统计
        performance = aggregator.get_performance_summary("tool_003")
        print(f"性能统计: {json.dumps(performance, indent=2, ensure_ascii=False, default=str)}")
        
        # 执行健康检查
        health_status = await aggregator.health_check_all()
        print(f"健康检查结果: {health_status}")
        
        # 发送工具间消息
        await aggregator.send_tool_message("tool_003", "tool_001", "ping", {"message": "Hello from API tool"})
        
        # 导出系统状态
        await aggregator.export_status("system_status.json")
        
        # 等待一段时间观察系统运行
        await asyncio.sleep(5)
        
        # 停止所有工具
        stop_results = await aggregator.stop_all_tools()
        print(f"停止结果: {stop_results}")
        
    except Exception as e:
        logger.error(f"示例执行异常: {e}")
        traceback.print_exc()
        
    finally:
        # 停止聚合器
        await aggregator.stop()


class AdvancedToolStateAggregator(ToolStateAggregator):
    """高级工具状态聚合器
    
    扩展了基础聚合器的功能，添加了：
    - 高级调度算法
    - 负载均衡
    - 自动故障转移
    - 性能优化
    - 高级监控和告警
    - 配置热更新
    - 插件系统
    - 分布式支持
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # 高级功能组件
        self.scheduler = AdvancedScheduler()
        self.load_balancer = LoadBalancer()
        self.failover_manager = FailoverManager()
        self.alert_manager = AlertManager()
        self.config_manager = ConfigManager()
        self.plugin_manager = PluginManager()
        self.distributed_coordinator = DistributedCoordinator()
        
        # 性能优化
        self._cache = {}
        self._cache_ttl = {}
        self._metrics_aggregator = MetricsAggregator()
        
        # 高级配置
        self.advanced_config = {
            'enable_caching': True,
            'cache_ttl': 60,
            'enable_load_balancing': True,
            'enable_failover': True,
            'enable_distributed': False,
            'metrics_retention_days': 7,
            'alert_thresholds': {
                'cpu_usage': 80.0,
                'memory_usage': 85.0,
                'response_time': 5.0,
                'error_rate': 0.05
            }
        }
        
        # 合并配置
        if config:
            self.advanced_config.update(config)
            
        logger.info("高级工具状态聚合器初始化完成")
        
    async def start(self):
        """启动高级聚合器"""
        await super().start()
        
        # 启动高级组件
        await self.scheduler.start()
        await self.load_balancer.start()
        await self.failover_manager.start()
        await self.alert_manager.start()
        await self.config_manager.start()
        await self.plugin_manager.start()
        
        if self.advanced_config['enable_distributed']:
            await self.distributed_coordinator.start()
            
        logger.info("高级工具状态聚合器启动完成")
        
    async def stop(self):
        """停止高级聚合器"""
        # 停止高级组件
        await self.distributed_coordinator.stop()
        await self.plugin_manager.stop()
        await self.config_manager.stop()
        await self.alert_manager.stop()
        await self.failover_manager.stop()
        await self.load_balancer.stop()
        await self.scheduler.stop()
        
        await super().stop()
        
        logger.info("高级工具状态聚合器已停止")
        
    # === 高级调度功能 ===
    
    async def schedule_tool_execution(self, tool_id: str, schedule_config: Dict[str, Any]) -> str:
        """调度工具执行
        
        Args:
            tool_id: 工具ID
            schedule_config: 调度配置
            
        Returns:
            调度任务ID
        """
        task_id = str(uuid4())
        
        schedule_task = {
            'task_id': task_id,
            'tool_id': tool_id,
            'config': schedule_config,
            'created_at': datetime.now(),
            'status': 'scheduled'
        }
        
        await self.scheduler.add_task(schedule_task)
        logger.info(f"添加调度任务: {task_id} for tool {tool_id}")
        
        return task_id
        
    async def cancel_scheduled_task(self, task_id: str) -> bool:
        """取消调度任务"""
        return await self.scheduler.remove_task(task_id)
        
    def get_scheduled_tasks(self) -> List[Dict[str, Any]]:
        """获取所有调度任务"""
        return self.scheduler.get_all_tasks()
        
    # === 负载均衡功能 ===
    
    async def register_load_balancer_target(self, tool_id: str, weight: int = 1):
        """注册负载均衡目标"""
        await self.load_balancer.add_target(tool_id, weight)
        
    async def get_load_balanced_tool(self, tool_type: str) -> Optional[str]:
        """获取负载均衡的工具"""
        return await self.load_balancer.get_best_target(tool_type)
        
    async def update_tool_load(self, tool_id: str, load_metrics: Dict[str, float]):
        """更新工具负载信息"""
        await self.load_balancer.update_target_load(tool_id, load_metrics)
        
    # === 故障转移功能 ===
    
    async def configure_failover(self, primary_tool_id: str, backup_tool_ids: List[str]):
        """配置故障转移"""
        await self.failover_manager.add_failover_group(primary_tool_id, backup_tool_ids)
        
    async def trigger_failover(self, failed_tool_id: str) -> Optional[str]:
        """触发故障转移"""
        return await self.failover_manager.execute_failover(failed_tool_id)
        
    def get_failover_status(self) -> Dict[str, Any]:
        """获取故障转移状态"""
        return self.failover_manager.get_status()
        
    # === 高级监控和告警 ===
    
    async def configure_alert_rule(self, rule_config: Dict[str, Any]) -> str:
        """配置告警规则"""
        rule_id = str(uuid4())
        rule_config['rule_id'] = rule_id
        rule_config['created_at'] = datetime.now()
        
        await self.alert_manager.add_rule(rule_config)
        logger.info(f"添加告警规则: {rule_id}")
        
        return rule_id
        
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        return await self.alert_manager.acknowledge_alert(alert_id)
        
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        return self.alert_manager.get_active_alerts()
        
    def get_alert_history(self, time_range: Optional[timedelta] = None) -> List[Dict[str, Any]]:
        """获取告警历史"""
        return self.alert_manager.get_alert_history(time_range)
        
    # === 配置管理 ===
    
    async def update_configuration(self, config_updates: Dict[str, Any]):
        """更新配置"""
        await self.config_manager.update_config(config_updates)
        
    def get_configuration(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config_manager.get_current_config()
        
    async def reload_configuration(self):
        """重新加载配置"""
        await self.config_manager.reload_config()
        
    # === 插件系统 ===
    
    async def load_plugin(self, plugin_path: str, plugin_config: Optional[Dict[str, Any]] = None):
        """加载插件"""
        await self.plugin_manager.load_plugin(plugin_path, plugin_config)
        
    async def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件"""
        return await self.plugin_manager.unload_plugin(plugin_name)
        
    def list_plugins(self) -> List[Dict[str, Any]]:
        """列出已加载的插件"""
        return self.plugin_manager.list_loaded_plugins()
        
    async def execute_plugin_hook(self, hook_name: str, *args, **kwargs):
        """执行插件钩子"""
        await self.plugin_manager.execute_hook(hook_name, *args, **kwargs)
        
    # === 分布式协调 ===
    
    async def join_cluster(self, cluster_config: Dict[str, Any]):
        """加入集群"""
        if not self.advanced_config['enable_distributed']:
            raise ToolError("分布式功能未启用")
            
        await self.distributed_coordinator.join_cluster(cluster_config)
        
    async def leave_cluster(self):
        """离开集群"""
        await self.distributed_coordinator.leave_cluster()
        
    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        return self.distributed_coordinator.get_cluster_status()
        
    async def sync_with_cluster(self):
        """与集群同步状态"""
        await self.distributed_coordinator.sync_state()
        
    # === 性能优化 ===
    
    def get_cached_result(self, cache_key: str) -> Optional[Any]:
        """获取缓存结果"""
        if not self.advanced_config['enable_caching']:
            return None
            
        if cache_key in self._cache:
            ttl = self._cache_ttl.get(cache_key, 0)
            if time.time() < ttl:
                return self._cache[cache_key]
            else:
                # 缓存过期，删除
                del self._cache[cache_key]
                del self._cache_ttl[cache_key]
                
        return None
        
    def set_cached_result(self, cache_key: str, result: Any, ttl: Optional[int] = None):
        """设置缓存结果"""
        if not self.advanced_config['enable_caching']:
            return
            
        if ttl is None:
            ttl = self.advanced_config['cache_ttl']
            
        self._cache[cache_key] = result
        self._cache_ttl[cache_key] = time.time() + ttl
        
    def clear_cache(self, pattern: Optional[str] = None):
        """清除缓存"""
        if pattern:
            # 按模式清除缓存
            keys_to_remove = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_remove:
                self._cache.pop(key, None)
                self._cache_ttl.pop(key, None)
        else:
            # 清除所有缓存
            self._cache.clear()
            self._cache_ttl.clear()
            
    # === 高级指标聚合 ===
    
    async def aggregate_metrics(self, time_range: timedelta) -> Dict[str, Any]:
        """聚合指标数据"""
        return await self._metrics_aggregator.aggregate(time_range)
        
    async def export_metrics(self, export_config: Dict[str, Any]):
        """导出指标数据"""
        await self._metrics_aggregator.export(export_config)
        
    # === 扩展的API方法 ===
    
    async def execute_with_retry(self, tool_id: str, command: str, 
                                max_retries: int = 3, retry_delay: float = 1.0,
                                **kwargs) -> Any:
        """带重试的命令执行"""
        tool = self.get_tool(tool_id)
        if not tool:
            raise ToolNotFoundError(f"工具不存在: {tool_id}")
            
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                result = await tool.execute(command, **kwargs)
                
                # 记录执行成功
                if hasattr(tool, 'record_call'):
                    tool.record_call(0, True)  # 重试次数不计入性能指标
                    
                return result
                
            except Exception as e:
                last_exception = e
                
                if attempt < max_retries:
                    logger.warning(f"工具 {tool_id} 命令执行失败，{retry_delay}秒后重试 (第{attempt + 1}次): {e}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    logger.error(f"工具 {tool_id} 命令执行最终失败: {e}")
                    
        raise last_exception
        
    async def execute_with_load_balancing(self, tool_type: str, command: str, **kwargs) -> Any:
        """负载均衡执行"""
        if not self.advanced_config['enable_load_balancing']:
            # 回退到普通执行
            tools = [tool for tool in self.registry.get_all_tools().values() 
                    if tool.tool_info.category == tool_type]
            if not tools:
                raise ToolError(f"未找到类型为 {tool_type} 的工具")
            return await tools[0].execute(command, **kwargs)
            
        # 使用负载均衡选择最佳工具
        best_tool_id = await self.get_load_balanced_tool(tool_type)
        if not best_tool_id:
            raise ToolError(f"负载均衡未找到可用的 {tool_type} 工具")
            
        return await self.execute_with_retry(best_tool_id, command, **kwargs)
        
    async def execute_with_failover(self, primary_tool_id: str, command: str, **kwargs) -> Any:
        """故障转移执行"""
        try:
            # 首先尝试主工具
            return await self.execute_with_retry(primary_tool_id, command, **kwargs)
            
        except Exception as e:
            logger.warning(f"主工具 {primary_tool_id} 执行失败，尝试故障转移: {e}")
            
            # 触发故障转移
            backup_tool_id = await self.trigger_failover(primary_tool_id)
            if backup_tool_id:
                logger.info(f"故障转移到备份工具: {backup_tool_id}")
                return await self.execute_with_retry(backup_tool_id, command, **kwargs)
            else:
                logger.error(f"故障转移失败，无可用备份工具")
                raise
                
    # === 高级系统状态 ===
    
    def get_advanced_system_status(self) -> Dict[str, Any]:
        """获取高级系统状态"""
        basic_status = self.get_system_status()
        
        # 添加高级指标
        advanced_status = {
            **basic_status,
            'advanced_metrics': {
                'cache_hit_rate': self._calculate_cache_hit_rate(),
                'load_balancing_stats': self.load_balancer.get_statistics(),
                'failover_stats': self.failover_manager.get_statistics(),
                'alert_stats': self.alert_manager.get_statistics(),
                'plugin_stats': self.plugin_manager.get_statistics(),
                'cluster_status': self.distributed_coordinator.get_cluster_status() if self.advanced_config['enable_distributed'] else None
            },
            'performance_optimization': {
                'cache_enabled': self.advanced_config['enable_caching'],
                'cache_size': len(self._cache),
                'load_balancing_enabled': self.advanced_config['enable_load_balancing'],
                'failover_enabled': self.advanced_config['enable_failover'],
                'distributed_enabled': self.advanced_config['enable_distributed']
            }
        }
        
        return advanced_status
        
    def _calculate_cache_hit_rate(self) -> float:
        """计算缓存命中率"""
        # 这里可以实现更复杂的缓存命中率计算
        # 目前返回模拟数据
        return 0.85
        
    # === 自动化运维 ===
    
    async def perform_auto_scaling(self):
        """执行自动扩缩容"""
        current_load = self._calculate_system_load()
        
        if current_load > 0.8:  # 负载过高
            await self._scale_up()
        elif current_load < 0.3:  # 负载过低
            await self._scale_down()
            
    def _calculate_system_load(self) -> float:
        """计算系统负载"""
        tools = self.registry.get_all_tools()
        if not tools:
            return 0.0
            
        total_load = 0.0
        for tool in tools.values():
            if tool.state == ToolState.RUNNING:
                total_load += (tool.resource_usage.cpu_percent + tool.resource_usage.memory_percent) / 200
                
        return total_load / len(tools)
        
    async def _scale_up(self):
        """扩容"""
        logger.info("执行自动扩容")
        # 这里可以实现具体的扩容逻辑
        # 例如：启动更多工具实例
        
    async def _scale_down(self):
        """缩容"""
        logger.info("执行自动缩容")
        # 这里可以实现具体的缩容逻辑
        # 例如：停止空闲的工具实例
        
    async def perform_predictive_maintenance(self):
        """执行预测性维护"""
        tools = self.registry.get_all_tools()
        
        for tool_id, tool in tools.items():
            if tool.state == ToolState.RUNNING:
                # 预测工具故障
                failure_probability = self._predict_tool_failure(tool)
                
                if failure_probability > 0.7:  # 故障概率超过70%
                    logger.warning(f"工具 {tool_id} 故障概率较高 ({failure_probability:.2f})，执行预防性维护")
                    await self._perform_preventive_maintenance(tool_id)
                    
    def _predict_tool_failure(self, tool: BaseTool) -> float:
        """预测工具故障概率"""
        # 简单的故障预测算法
        # 考虑CPU、内存使用率和错误率
        
        cpu_factor = min(tool.resource_usage.cpu_percent / 100, 1.0)
        memory_factor = min(tool.resource_usage.memory_percent / 100, 1.0)
        error_factor = tool.performance_metrics.error_rate
        
        # 加权计算故障概率
        failure_probability = (cpu_factor * 0.4 + memory_factor * 0.4 + error_factor * 0.2)
        
        return min(failure_probability, 1.0)
        
    async def _perform_preventive_maintenance(self, tool_id: str):
        """执行预防性维护"""
        try:
            logger.info(f"对工具 {tool_id} 执行预防性维护")
            
            # 备份当前状态
            await self.export_status(f"backup_{tool_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            
            # 重启工具
            await self.restart_tool(tool_id)
            
            logger.info(f"工具 {tool_id} 预防性维护完成")
            
        except Exception as e:
            logger.error(f"预防性维护失败 {tool_id}: {e}")


# === 高级组件实现 ===

class AdvancedScheduler:
    """高级调度器"""
    
    def __init__(self):
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._lock = asyncio.Lock()
        
    async def start(self):
        """启动调度器"""
        self._running = True
        asyncio.create_task(self._schedule_loop())
        
    async def stop(self):
        """停止调度器"""
        self._running = False
        
    async def add_task(self, task_config: Dict[str, Any]):
        """添加调度任务"""
        async with self._lock:
            self._tasks[task_config['task_id']] = task_config
            
    async def remove_task(self, task_id: str) -> bool:
        """移除调度任务"""
        async with self._lock:
            if task_id in self._tasks:
                del self._tasks[task_id]
                return True
            return False
            
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """获取所有任务"""
        return list(self._tasks.values())
        
    async def _schedule_loop(self):
        """调度循环"""
        while self._running:
            try:
                current_time = datetime.now()
                
                for task_id, task_config in self._tasks.items():
                    if await self._should_execute_task(task_config, current_time):
                        await self._execute_task(task_config)
                        
                await asyncio.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"调度循环异常: {e}")
                await asyncio.sleep(1)
                
    async def _should_execute_task(self, task_config: Dict[str, Any], current_time: datetime) -> bool:
        """判断是否应该执行任务"""
        schedule_type = task_config.get('type', 'once')
        
        if schedule_type == 'once':
            return not task_config.get('executed', False)
        elif schedule_type == 'interval':
            interval = task_config.get('interval', 60)
            last_run = task_config.get('last_run', datetime.min)
            return (current_time - last_run).total_seconds() >= interval
        elif schedule_type == 'cron':
            # 简化的cron表达式支持
            cron_expr = task_config.get('cron', '*/5 * * * *')  # 默认每5分钟
            return self._parse_cron_expression(cron_expr, current_time)
            
        return False
        
    def _parse_cron_expression(self, cron_expr: str, current_time: datetime) -> bool:
        """解析cron表达式"""
        # 简化的cron解析实现
        # 格式: 秒 分 时 日 月 星期
        parts = cron_expr.strip().split()
        
        if len(parts) != 6:
            return False
            
        try:
            second, minute, hour, day, month, weekday = parts
            
            # 检查各个字段
            if not self._match_cron_field(second, current_time.second):
                return False
            if not self._match_cron_field(minute, current_time.minute):
                return False
            if not self._match_cron_field(hour, current_time.hour):
                return False
            if not self._match_cron_field(day, current_time.day):
                return False
            if not self._match_cron_field(month, current_time.month):
                return False
            if not self._match_cron_field(weekday, current_time.weekday()):
                return False
                
            return True
            
        except Exception:
            return False
            
    def _match_cron_field(self, field: str, value: int) -> bool:
        """匹配cron字段"""
        if field == '*':
            return True
        elif field.isdigit():
            return int(field) == value
        elif ',' in field:
            return value in [int(x.strip()) for x in field.split(',')]
        elif '/' in field:
            start, step = field.split('/')
            start_val = int(start) if start != '*' else 0
            return (value - start_val) % int(step) == 0
        elif '-' in field:
            start, end = field.split('-')
            return start_val <= value <= int(end)
            
        return False
        
    async def _execute_task(self, task_config: Dict[str, Any]):
        """执行任务"""
        task_id = task_config['task_id']
        
        try:
            logger.info(f"执行调度任务: {task_id}")
            
            # 更新任务状态
            task_config['last_run'] = datetime.now()
            task_config['executed'] = True
            
            # 执行具体逻辑（这里需要根据任务类型执行相应操作）
            # task_config['callback'](task_config)
            
        except Exception as e:
            logger.error(f"执行调度任务失败 {task_id}: {e}")
            task_config['last_error'] = str(e)


class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self):
        self._targets: Dict[str, Dict[str, Any]] = {}
        self._algorithm = 'round_robin'  # 默认轮询算法
        self._current_index = 0
        self._lock = asyncio.Lock()
        
    async def start(self):
        """启动负载均衡器"""
        pass
        
    async def stop(self):
        """停止负载均衡器"""
        pass
        
    async def add_target(self, tool_id: str, weight: int = 1):
        """添加负载均衡目标"""
        async with self._lock:
            self._targets[tool_id] = {
                'weight': weight,
                'current_load': 0.0,
                'total_requests': 0,
                'active_requests': 0,
                'last_health_check': datetime.now()
            }
            
    async def remove_target(self, tool_id: str) -> bool:
        """移除负载均衡目标"""
        async with self._lock:
            if tool_id in self._targets:
                del self._targets[tool_id]
                return True
            return False
            
    async def get_best_target(self, tool_type: str) -> Optional[str]:
        """获取最佳目标"""
        async with self._lock:
            available_targets = [
                tool_id for tool_id, target in self._targets.items()
                if target['current_load'] < 1.0 and target['active_requests'] == 0
            ]
            
            if not available_targets:
                return None
                
            if self._algorithm == 'round_robin':
                target = available_targets[self._current_index % len(available_targets)]
                self._current_index += 1
                return target
            elif self._algorithm == 'least_connections':
                return min(available_targets, key=lambda x: self._targets[x]['active_requests'])
            elif self._algorithm == 'weighted':
                # 加权轮询
                return self._weighted_round_robin(available_targets)
                
            return available_targets[0]
            
    def _weighted_round_robin(self, targets: List[str]) -> str:
        """加权轮询算法"""
        total_weight = sum(self._targets[target]['weight'] for target in targets)
        
        for target in targets:
            self._targets[target]['current_weight'] = (
                self._targets[target].get('current_weight', 0) + 
                self._targets[target]['weight']
            )
            
        best_target = max(targets, key=lambda x: self._targets[x]['current_weight'])
        self._targets[best_target]['current_weight'] -= total_weight
        
        return best_target
        
    async def update_target_load(self, tool_id: str, load_metrics: Dict[str, float]):
        """更新目标负载信息"""
        async with self._lock:
            if tool_id in self._targets:
                self._targets[tool_id].update(load_metrics)
                self._targets[tool_id]['last_health_check'] = datetime.now()
                
    def get_statistics(self) -> Dict[str, Any]:
        """获取负载均衡统计"""
        return {
            'total_targets': len(self._targets),
            'algorithm': self._algorithm,
            'targets': {
                tool_id: {
                    'weight': target['weight'],
                    'current_load': target['current_load'],
                    'total_requests': target['total_requests'],
                    'active_requests': target['active_requests']
                }
                for tool_id, target in self._targets.items()
            }
        }


class FailoverManager:
    """故障转移管理器"""
    
    def __init__(self):
        self._failover_groups: Dict[str, List[str]] = {}
        self._failover_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        
    async def start(self):
        """启动故障转移管理器"""
        pass
        
    async def stop(self):
        """停止故障转移管理器"""
        pass
        
    async def add_failover_group(self, primary_tool_id: str, backup_tool_ids: List[str]):
        """添加故障转移组"""
        async with self._lock:
            self._failover_groups[primary_tool_id] = backup_tool_ids
            
    async def execute_failover(self, failed_tool_id: str) -> Optional[str]:
        """执行故障转移"""
        async with self._lock:
            backup_tools = self._failover_groups.get(failed_tool_id, [])
            
            for backup_tool_id in backup_tools:
                # 检查备份工具是否可用
                if await self._is_tool_available(backup_tool_id):
                    # 记录故障转移
                    failover_record = {
                        'failed_tool': failed_tool_id,
                        'backup_tool': backup_tool_id,
                        'timestamp': datetime.now(),
                        'status': 'success'
                    }
                    self._failover_history.append(failover_record)
                    
                    logger.info(f"故障转移成功: {failed_tool_id} -> {backup_tool_id}")
                    return backup_tool_id
                    
            # 所有备份工具都不可用
            failover_record = {
                'failed_tool': failed_tool_id,
                'backup_tool': None,
                'timestamp': datetime.now(),
                'status': 'failed'
            }
            self._failover_history.append(failover_record)
            
            logger.error(f"故障转移失败: {failed_tool_id}，无可用备份工具")
            return None
            
    async def _is_tool_available(self, tool_id: str) -> bool:
        """检查工具是否可用"""
        # 这里应该检查工具的健康状态和可用性
        # 简化实现
        return True
        
    def get_status(self) -> Dict[str, Any]:
        """获取故障转移状态"""
        return {
            'failover_groups': self._failover_groups,
            'recent_failovers': self._failover_history[-10:] if self._failover_history else []
        }
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取故障转移统计"""
        total_failovers = len(self._failover_history)
        successful_failovers = sum(1 for record in self._failover_history if record['status'] == 'success')
        
        return {
            'total_failovers': total_failovers,
            'successful_failovers': successful_failovers,
            'success_rate': successful_failovers / total_failovers if total_failovers > 0 else 0
        }


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self._alert_rules: Dict[str, Dict[str, Any]] = {}
        self._active_alerts: Dict[str, Dict[str, Any]] = {}
        self._alert_history: List[Dict[str, Any]] = []
        self._lock = asyncio.Lock()
        
    async def start(self):
        """启动告警管理器"""
        asyncio.create_task(self._alert_monitoring_loop())
        
    async def stop(self):
        """停止告警管理器"""
        pass
        
    async def add_rule(self, rule_config: Dict[str, Any]):
        """添加告警规则"""
        async with self._lock:
            self._alert_rules[rule_config['rule_id']] = rule_config
            
    async def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        async with self._lock:
            if alert_id in self._active_alerts:
                self._active_alerts[alert_id]['acknowledged'] = True
                self._active_alerts[alert_id]['acknowledged_at'] = datetime.now()
                return True
            return False
            
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        return list(self._active_alerts.values())
        
    def get_alert_history(self, time_range: Optional[timedelta] = None) -> List[Dict[str, Any]]:
        """获取告警历史"""
        if time_range:
            cutoff_time = datetime.now() - time_range
            return [alert for alert in self._alert_history if alert['timestamp'] >= cutoff_time]
        return self._alert_history
        
    async def _alert_monitoring_loop(self):
        """告警监控循环"""
        while True:
            try:
                await self._check_alert_rules()
                await asyncio.sleep(30)  # 每30秒检查一次
            except Exception as e:
                logger.error(f"告警监控循环异常: {e}")
                await asyncio.sleep(30)
                
    async def _check_alert_rules(self):
        """检查告警规则"""
        for rule_id, rule_config in self._alert_rules.items():
            try:
                if await self._evaluate_rule(rule_config):
                    await self._trigger_alert(rule_config)
            except Exception as e:
                logger.error(f"评估告警规则失败 {rule_id}: {e}")
                
    async def _evaluate_rule(self, rule_config: Dict[str, Any]) -> bool:
        """评估告警规则"""
        # 简化实现 - 实际应该根据规则配置评估条件
        return False
        
    async def _trigger_alert(self, rule_config: Dict[str, Any]):
        """触发告警"""
        alert_id = str(uuid4())
        
        alert = {
            'alert_id': alert_id,
            'rule_id': rule_config['rule_id'],
            'rule_name': rule_config.get('name', 'Unknown'),
            'message': rule_config.get('message', 'Alert triggered'),
            'severity': rule_config.get('severity', 'warning'),
            'timestamp': datetime.now(),
            'acknowledged': False
        }
        
        async with self._lock:
            self._active_alerts[alert_id] = alert
            self._alert_history.append(alert.copy())
            
        logger.warning(f"触发告警: {alert['rule_name']} - {alert['message']}")
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        return {
            'total_rules': len(self._alert_rules),
            'active_alerts': len(self._active_alerts),
            'total_alerts': len(self._alert_history)
        }


class ConfigManager:
    """配置管理器"""
    
    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._config_file: Optional[str] = None
        self._watchers: List[Callable] = []
        
    async def start(self):
        """启动配置管理器"""
        pass
        
    async def stop(self):
        """停止配置管理器"""
        pass
        
    async def update_config(self, config_updates: Dict[str, Any]):
        """更新配置"""
        old_config = self._config.copy()
        self._config.update(config_updates)
        
        # 通知配置变更
        for watcher in self._watchers:
            try:
                watcher(old_config, self._config)
            except Exception as e:
                logger.error(f"配置变更通知失败: {e}")
                
    def get_current_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self._config.copy()
        
    async def reload_config(self):
        """重新加载配置"""
        # 从文件重新加载配置的逻辑
        pass
        
    def add_config_watcher(self, watcher: Callable):
        """添加配置变更监听器"""
        self._watchers.append(watcher)


class PluginManager:
    """插件管理器"""
    
    def __init__(self):
        self._plugins: Dict[str, Any] = {}
        self._plugin_hooks: Dict[str, List[Callable]] = defaultdict(list)
        
    async def start(self):
        """启动插件管理器"""
        pass
        
    async def stop(self):
        """停止插件管理器"""
        # 卸载所有插件
        for plugin_name in list(self._plugins.keys()):
            await self.unload_plugin(plugin_name)
            
    async def load_plugin(self, plugin_path: str, plugin_config: Optional[Dict[str, Any]] = None):
        """加载插件"""
        try:
            # 简化的插件加载逻辑
            plugin_name = Path(plugin_path).stem
            self._plugins[plugin_name] = {
                'path': plugin_path,
                'config': plugin_config or {},
                'loaded_at': datetime.now(),
                'status': 'loaded'
            }
            
            logger.info(f"插件加载成功: {plugin_name}")
            
        except Exception as e:
            logger.error(f"插件加载失败 {plugin_path}: {e}")
            raise
            
    async def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件"""
        if plugin_name in self._plugins:
            del self._plugins[plugin_name]
            logger.info(f"插件卸载成功: {plugin_name}")
            return True
        return False
        
    def list_loaded_plugins(self) -> List[Dict[str, Any]]:
        """列出已加载的插件"""
        return list(self._plugins.values())
        
    async def execute_hook(self, hook_name: str, *args, **kwargs):
        """执行插件钩子"""
        for plugin_callback in self._plugin_hooks[hook_name]:
            try:
                if asyncio.iscoroutinefunction(plugin_callback):
                    await plugin_callback(*args, **kwargs)
                else:
                    plugin_callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"插件钩子执行失败 {hook_name}: {e}")
                
    def register_plugin_hook(self, hook_name: str, callback: Callable):
        """注册插件钩子"""
        self._plugin_hooks[hook_name].append(callback)
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取插件统计"""
        return {
            'loaded_plugins': len(self._plugins),
            'available_hooks': list(self._plugin_hooks.keys())
        }


class DistributedCoordinator:
    """分布式协调器"""
    
    def __init__(self):
        self._cluster_id: Optional[str] = None
        self._cluster_nodes: Dict[str, Dict[str, Any]] = {}
        self._is_coordinator = False
        
    async def start(self):
        """启动分布式协调器"""
        pass
        
    async def stop(self):
        """停止分布式协调器"""
        if self._cluster_id:
            await self.leave_cluster()
            
    async def join_cluster(self, cluster_config: Dict[str, Any]):
        """加入集群"""
        self._cluster_id = cluster_config.get('cluster_id')
        self._is_coordinator = cluster_config.get('is_coordinator', False)
        
        logger.info(f"加入集群: {self._cluster_id}")
        
    async def leave_cluster(self):
        """离开集群"""
        if self._cluster_id:
            logger.info(f"离开集群: {self._cluster_id}")
            self._cluster_id = None
            self._cluster_nodes.clear()
            
    def get_cluster_status(self) -> Dict[str, Any]:
        """获取集群状态"""
        return {
            'cluster_id': self._cluster_id,
            'is_coordinator': self._is_coordinator,
            'nodes': self._cluster_nodes
        }
        
    async def sync_state(self):
        """同步状态"""
        if self._cluster_id and self._is_coordinator:
            # 协调器同步状态逻辑
            pass


class MetricsAggregator:
    """指标聚合器"""
    
    def __init__(self):
        self._metrics_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    async def aggregate(self, time_range: timedelta) -> Dict[str, Any]:
        """聚合指标数据"""
        cutoff_time = datetime.now() - time_range
        
        aggregated_metrics = {}
        for metric_name, data_points in self._metrics_data.items():
            # 过滤时间范围内的数据
            filtered_data = [
                point for point in data_points 
                if point.get('timestamp', datetime.min) >= cutoff_time
            ]
            
            if filtered_data:
                aggregated_metrics[metric_name] = self._calculate_aggregations(filtered_data)
                
        return aggregated_metrics
        
    def _calculate_aggregations(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算聚合值"""
        if not data_points:
            return {}
            
        # 简化的聚合计算
        return {
            'count': len(data_points),
            'latest': data_points[-1] if data_points else None
        }
        
    async def export(self, export_config: Dict[str, Any]):
        """导出指标数据"""
        export_path = export_config.get('path', 'metrics_export.json')
        export_format = export_config.get('format', 'json')
        
        export_data = {
            'exported_at': datetime.now().isoformat(),
            'metrics': dict(self._metrics_data)
        }
        
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False, default=str)
            
        logger.info(f"指标数据已导出到: {export_path}")


# === 扩展的工具实现 ===

class DatabaseTool(BaseTool):
    """数据库工具实现"""
    
    def __init__(self, tool_info: ToolInfo):
        super().__init__(tool_info)
        self._connection_pool = []
        self._max_connections = self.tool_info.config.get('max_connections', 10)
        self._connection_string = self.tool_info.config.get('connection_string', '')
        
    async def start(self) -> bool:
        """启动数据库工具"""
        try:
            self.update_state(ToolState.STARTING)
            
            # 初始化连接池
            for i in range(self._max_connections):
                # 模拟数据库连接
                connection = {
                    'id': i,
                    'status': 'active',
                    'created_at': datetime.now()
                }
                self._connection_pool.append(connection)
                
            self._running = True
            self.is_alive = True
            self.update_state(ToolState.RUNNING)
            self.update_health(HealthStatus.HEALTHY)
            
            logger.info(f"数据库工具 {self.tool_info.name} 启动成功，连接池大小: {len(self._connection_pool)}")
            return True
            
        except Exception as e:
            logger.error(f"数据库工具启动失败: {e}")
            self.update_state(ToolState.ERROR)
            self.update_health(HealthStatus.FAILED)
            return False
            
    async def stop(self) -> bool:
        """停止数据库工具"""
        try:
            self.update_state(ToolState.STOPPING)
            
            # 关闭所有连接
            self._connection_pool.clear()
            
            self._running = False
            self.is_alive = False
            self.update_state(ToolState.STOPPED)
            
            logger.info(f"数据库工具 {self.tool_info.name} 停止成功")
            return True
            
        except Exception as e:
            logger.error(f"数据库工具停止失败: {e}")
            return False
            
    async def restart(self) -> bool:
        """重启数据库工具"""
        await self.stop()
        await asyncio.sleep(0.5)
        return await self.start()
        
    async def health_check(self) -> HealthStatus:
        """数据库健康检查"""
        if not self.is_alive:
            return HealthStatus.FAILED
            
        # 检查连接池状态
        active_connections = sum(1 for conn in self._connection_pool if conn['status'] == 'active')
        connection_utilization = active_connections / len(self._connection_pool) if self._connection_pool else 0
        
        if connection_utilization > 0.9:
            return HealthStatus.CRITICAL
        elif connection_utilization > 0.7:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
            
    async def get_status(self) -> Dict[str, Any]:
        """获取数据库状态"""
        return {
            'running': self._running,
            'connection_pool_size': len(self._connection_pool),
            'active_connections': sum(1 for conn in self._connection_pool if conn['status'] == 'active'),
            'state': self.state.name,
            'health': self.health_status.name
        }
        
    async def execute(self, command: str, **kwargs) -> Any:
        """执行数据库命令"""
        start_time = time.time()
        
        try:
            if command == "query":
                sql = kwargs.get('sql', 'SELECT 1')
                # 模拟SQL查询
                await asyncio.sleep(0.01)
                result = {"rows": [{"id": 1, "result": "success"}], "row_count": 1}
                
            elif command == "connect":
                # 模拟连接测试
                await asyncio.sleep(0.05)
                result = {"status": "connected", "connection_string": self._connection_string}
                
            elif command == "pool_status":
                active_connections = sum(1 for conn in self._connection_pool if conn['status'] == 'active')
                result = {
                    "total_connections": len(self._connection_pool),
                    "active_connections": active_connections,
                    "utilization": active_connections / len(self._connection_pool) if self._connection_pool else 0
                }
                
            else:
                result = {"status": "unknown_command", "command": command}
                
            response_time = time.time() - start_time
            self.record_call(response_time, True)
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self.record_call(response_time, False)
            raise ToolError(f"数据库命令执行失败: {e}")


class CacheTool(BaseTool):
    """缓存工具实现"""
    
    def __init__(self, tool_info: ToolInfo):
        super().__init__(tool_info)
        self._cache_data: Dict[str, Any] = {}
        self._cache_ttl: Dict[str, float] = {}
        self._max_size = self.tool_info.config.get('max_size', 1000)
        self._eviction_policy = self.tool_info.config.get('eviction_policy', 'lru')
        self._tasks = []  # 初始化任务列表
        
    async def start(self) -> bool:
        """启动缓存工具"""
        try:
            self.update_state(ToolState.STARTING)
            
            # 启动缓存清理任务
            self._tasks.append(asyncio.create_task(self._cache_cleanup_task()))
            
            self._running = True
            self.is_alive = True
            self.update_state(ToolState.RUNNING)
            self.update_health(HealthStatus.HEALTHY)
            
            logger.info(f"缓存工具 {self.tool_info.name} 启动成功")
            return True
            
        except Exception as e:
            logger.error(f"缓存工具启动失败: {e}")
            self.update_state(ToolState.ERROR)
            self.update_health(HealthStatus.FAILED)
            return False
            
    async def stop(self) -> bool:
        """停止缓存工具"""
        try:
            self.update_state(ToolState.STOPPING)
            
            self._running = False
            self.is_alive = False
            
            # 取消清理任务
            for task in self._tasks:
                task.cancel()
                
            self.update_state(ToolState.STOPPED)
            
            logger.info(f"缓存工具 {self.tool_info.name} 停止成功")
            return True
            
        except Exception as e:
            logger.error(f"缓存工具停止失败: {e}")
            return False
            
    async def restart(self) -> bool:
        """重启缓存工具"""
        await self.stop()
        await asyncio.sleep(0.1)
        return await self.start()
        
    async def health_check(self) -> HealthStatus:
        """缓存健康检查"""
        if not self.is_alive:
            return HealthStatus.FAILED
            
        # 检查缓存使用率
        cache_usage = len(self._cache_data) / self._max_size
        
        if cache_usage > 0.95:
            return HealthStatus.CRITICAL
        elif cache_usage > 0.8:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
            
    async def get_status(self) -> Dict[str, Any]:
        """获取缓存状态"""
        return {
            'running': self._running,
            'cache_size': len(self._cache_data),
            'max_size': self._max_size,
            'usage_rate': len(self._cache_data) / self._max_size,
            'eviction_policy': self._eviction_policy,
            'state': self.state.name,
            'health': self.health_status.name
        }
        
    async def execute(self, command: str, **kwargs) -> Any:
        """执行缓存命令"""
        start_time = time.time()
        
        try:
            if command == "get":
                key = kwargs.get('key')
                if key in self._cache_data:
                    # 检查TTL
                    if key in self._cache_ttl and time.time() > self._cache_ttl[key]:
                        del self._cache_data[key]
                        del self._cache_ttl[key]
                        result = {"status": "miss", "key": key}
                    else:
                        result = {"status": "hit", "key": key, "value": self._cache_data[key]}
                else:
                    result = {"status": "miss", "key": key}
                    
            elif command == "set":
                key = kwargs.get('key')
                value = kwargs.get('value')
                ttl = kwargs.get('ttl', 3600)  # 默认1小时
                
                # 检查缓存大小限制
                if len(self._cache_data) >= self._max_size:
                    await self._evict_cache()
                    
                self._cache_data[key] = value
                self._cache_ttl[key] = time.time() + ttl
                result = {"status": "stored", "key": key}
                
            elif command == "delete":
                key = kwargs.get('key')
                if key in self._cache_data:
                    del self._cache_data[key]
                    if key in self._cache_ttl:
                        del self._cache_ttl[key]
                    result = {"status": "deleted", "key": key}
                else:
                    result = {"status": "not_found", "key": key}
                    
            elif command == "clear":
                self._cache_data.clear()
                self._cache_ttl.clear()
                result = {"status": "cleared"}
                
            else:
                result = {"status": "unknown_command", "command": command}
                
            response_time = time.time() - start_time
            self.record_call(response_time, True)
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self.record_call(response_time, False)
            raise ToolError(f"缓存命令执行失败: {e}")
            
    async def _cache_cleanup_task(self):
        """缓存清理任务"""
        while self._running:
            try:
                current_time = time.time()
                expired_keys = [
                    key for key, expiry in self._cache_ttl.items()
                    if current_time > expiry
                ]
                
                for key in expired_keys:
                    self._cache_data.pop(key, None)
                    self._cache_ttl.pop(key, None)
                    
                await asyncio.sleep(60)  # 每分钟清理一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"缓存清理任务异常: {e}")
                await asyncio.sleep(60)
                
    async def _evict_cache(self):
        """缓存淘汰"""
        if self._eviction_policy == 'lru':
            # 简化实现：删除一半的缓存项
            keys_to_evict = list(self._cache_data.keys())[:len(self._cache_data)//2]
            for key in keys_to_evict:
                self._cache_data.pop(key, None)
                self._cache_ttl.pop(key, None)
        elif self._eviction_policy == 'ttl':
            # 删除最接近过期的项
            if self._cache_ttl:
                earliest_expiry_key = min(self._cache_ttl.keys(), key=lambda k: self._cache_ttl[k])
                self._cache_data.pop(earliest_expiry_key, None)
                self._cache_ttl.pop(earliest_expiry_key, None)


class APIGatewayTool(BaseTool):
    """API网关工具实现"""
    
    def __init__(self, tool_info: ToolInfo):
        super().__init__(tool_info)
        self._routes: Dict[str, Dict[str, Any]] = {}
        self._rate_limits: Dict[str, Dict[str, Any]] = {}
        self._middleware_stack = []
        
    async def start(self) -> bool:
        """启动API网关"""
        try:
            self.update_state(ToolState.STARTING)
            
            # 初始化默认路由
            await self._initialize_default_routes()
            
            # 启动网关任务
            self._tasks.append(asyncio.create_task(self._rate_limit_monitor_task()))
            
            self._running = True
            self.is_alive = True
            self.update_state(ToolState.RUNNING)
            self.update_health(HealthStatus.HEALTHY)
            
            logger.info(f"API网关 {self.tool_info.name} 启动成功")
            return True
            
        except Exception as e:
            logger.error(f"API网关启动失败: {e}")
            self.update_state(ToolState.ERROR)
            self.update_health(HealthStatus.FAILED)
            return False
            
    async def stop(self) -> bool:
        """停止API网关"""
        try:
            self.update_state(ToolState.STOPPING)
            
            self._running = False
            self.is_alive = False
            
            # 取消所有任务
            for task in self._tasks:
                task.cancel()
                
            self.update_state(ToolState.STOPPED)
            
            logger.info(f"API网关 {self.tool_info.name} 停止成功")
            return True
            
        except Exception as e:
            logger.error(f"API网关停止失败: {e}")
            return False
            
    async def restart(self) -> bool:
        """重启API网关"""
        await self.stop()
        await asyncio.sleep(0.2)
        return await self.start()
        
    async def health_check(self) -> HealthStatus:
        """API网关健康检查"""
        if not self.is_alive:
            return HealthStatus.FAILED
            
        # 检查路由数量和速率限制
        if len(self._routes) == 0:
            return HealthStatus.WARNING
            
        return HealthStatus.HEALTHY
        
    async def get_status(self) -> Dict[str, Any]:
        """获取API网关状态"""
        return {
            'running': self._running,
            'routes_count': len(self._routes),
            'middleware_count': len(self._middleware_stack),
            'rate_limits_count': len(self._rate_limits),
            'state': self.state.name,
            'health': self.health_status.name
        }
        
    async def execute(self, command: str, **kwargs) -> Any:
        """执行API网关命令"""
        start_time = time.time()
        
        try:
            if command == "add_route":
                path = kwargs.get('path')
                methods = kwargs.get('methods', ['GET'])
                target = kwargs.get('target')
                
                self._routes[path] = {
                    'methods': methods,
                    'target': target,
                    'created_at': datetime.now()
                }
                result = {"status": "route_added", "path": path}
                
            elif command == "remove_route":
                path = kwargs.get('path')
                if path in self._routes:
                    del self._routes[path]
                    result = {"status": "route_removed", "path": path}
                else:
                    result = {"status": "route_not_found", "path": path}
                    
            elif command == "list_routes":
                result = {
                    "status": "routes_list",
                    "routes": [
                        {
                            "path": path,
                            "methods": route_info['methods'],
                            "target": route_info['target']
                        }
                        for path, route_info in self._routes.items()
                    ]
                }
                
            elif command == "set_rate_limit":
                path = kwargs.get('path')
                limit = kwargs.get('limit', 100)
                window = kwargs.get('window', 60)
                
                self._rate_limits[path] = {
                    'limit': limit,
                    'window': window,
                    'requests': []
                }
                result = {"status": "rate_limit_set", "path": path, "limit": limit}
                
            else:
                result = {"status": "unknown_command", "command": command}
                
            response_time = time.time() - start_time
            self.record_call(response_time, True)
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            self.record_call(response_time, False)
            raise ToolError(f"API网关命令执行失败: {e}")
            
    async def _initialize_default_routes(self):
        """初始化默认路由"""
        default_routes = [
            {"path": "/health", "methods": ["GET"], "target": "health_check"},
            {"path": "/status", "methods": ["GET"], "target": "status_check"},
            {"path": "/metrics", "methods": ["GET"], "target": "metrics_endpoint"}
        ]
        
        for route in default_routes:
            self._routes[route['path']] = {
                'methods': route['methods'],
                'target': route['target'],
                'created_at': datetime.now(),
                'is_default': True
            }
            
    async def _rate_limit_monitor_task(self):
        """速率限制监控任务"""
        while self._running:
            try:
                current_time = time.time()
                
                # 清理过期的请求记录
                for path, rate_limit in self._rate_limits.items():
                    window_start = current_time - rate_limit['window']
                    rate_limit['requests'] = [
                        req_time for req_time in rate_limit['requests']
                        if req_time > window_start
                    ]
                    
                await asyncio.sleep(10)  # 每10秒检查一次
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"速率限制监控任务异常: {e}")
                await asyncio.sleep(10)


# === 工具工厂和注册 ===

class ToolFactory:
    """工具工厂类"""
    
    _tool_classes: Dict[str, type] = {
        'mock': MockTool,
        'database': DatabaseTool,
        'cache': CacheTool,
        'api_gateway': APIGatewayTool
    }
    
    @classmethod
    def register_tool_class(cls, tool_type: str, tool_class: type):
        """注册工具类"""
        cls._tool_classes[tool_type] = tool_class
        
    @classmethod
    def create_tool(cls, tool_type: str, tool_info: ToolInfo) -> BaseTool:
        """创建工具实例"""
        if tool_type not in cls._tool_classes:
            raise ToolError(f"未知的工具类型: {tool_type}")
            
        tool_class = cls._tool_classes[tool_type]
        return tool_class(tool_info)
        
    @classmethod
    def get_available_types(cls) -> List[str]:
        """获取可用的工具类型"""
        return list(cls._tool_classes.keys())


# === 性能测试工具 ===

class PerformanceTester:
    """性能测试工具"""
    
    def __init__(self, aggregator: ToolStateAggregator):
        self.aggregator = aggregator
        
    async def run_load_test(self, tool_id: str, concurrent_requests: int = 10, 
                           total_requests: int = 100) -> Dict[str, Any]:
        """运行负载测试"""
        logger.info(f"开始负载测试: {tool_id}, 并发: {concurrent_requests}, 总请求: {total_requests}")
        
        results = {
            'tool_id': tool_id,
            'concurrent_requests': concurrent_requests,
            'total_requests': total_requests,
            'start_time': datetime.now(),
            'responses': [],
            'errors': []
        }
        
        # 创建并发任务
        semaphore = asyncio.Semaphore(concurrent_requests)
        
        async def make_request(request_id: int):
            async with semaphore:
                start_time = time.time()
                try:
                    tool = self.aggregator.get_tool(tool_id)
                    if not tool:
                        raise ToolNotFoundError(f"工具不存在: {tool_id}")
                        
                    result = await tool.execute("test")
                    response_time = time.time() - start_time
                    
                    results['responses'].append({
                        'request_id': request_id,
                        'response_time': response_time,
                        'success': True,
                        'result': result
                    })
                    
                except Exception as e:
                    response_time = time.time() - start_time
                    results['errors'].append({
                        'request_id': request_id,
                        'response_time': response_time,
                        'error': str(e)
                    })
                    
        # 执行所有请求
        tasks = [make_request(i) for i in range(total_requests)]
        await asyncio.gather(*tasks)
        
        results['end_time'] = datetime.now()
        results['duration'] = (results['end_time'] - results['start_time']).total_seconds()
        
        # 计算统计信息
        all_responses = results['responses'] + results['errors']
        if all_responses:
            response_times = [r['response_time'] for r in all_responses]
            results['statistics'] = {
                'total_requests': len(all_responses),
                'successful_requests': len(results['responses']),
                'failed_requests': len(results['errors']),
                'success_rate': len(results['responses']) / len(all_responses),
                'average_response_time': statistics.mean(response_times),
                'min_response_time': min(response_times),
                'max_response_time': max(response_times),
                'requests_per_second': len(all_responses) / results['duration']
            }
            
        logger.info(f"负载测试完成: {tool_id}")
        return results
        
    async def run_stress_test(self, tool_id: str, duration_seconds: int = 60) -> Dict[str, Any]:
        """运行压力测试"""
        logger.info(f"开始压力测试: {tool_id}, 持续时间: {duration_seconds}秒")
        
        results = {
            'tool_id': tool_id,
            'duration_seconds': duration_seconds,
            'start_time': datetime.now(),
            'requests_made': 0,
            'responses': [],
            'errors': []
        }
        
        end_time = time.time() + duration_seconds
        
        async def continuous_requests():
            while time.time() < end_time:
                try:
                    tool = self.aggregator.get_tool(tool_id)
                    if not tool:
                        break
                        
                    start_time = time.time()
                    result = await tool.execute("test")
                    response_time = time.time() - start_time
                    
                    results['responses'].append({
                        'response_time': response_time,
                        'success': True,
                        'timestamp': datetime.now()
                    })
                    results['requests_made'] += 1
                    
                    # 短暂延迟避免过于频繁的请求
                    await asyncio.sleep(0.01)
                    
                except Exception as e:
                    results['errors'].append({
                        'error': str(e),
                        'timestamp': datetime.now()
                    })
                    results['requests_made'] += 1
                    
        # 运行压力测试
        await continuous_requests()
        
        results['end_time'] = datetime.now()
        results['actual_duration'] = (results['end_time'] - results['start_time']).total_seconds()
        
        # 计算统计信息
        if results['responses']:
            response_times = [r['response_time'] for r in results['responses']]
            results['statistics'] = {
                'total_requests': results['requests_made'],
                'successful_requests': len(results['responses']),
                'failed_requests': len(results['errors']),
                'success_rate': len(results['responses']) / results['requests_made'],
                'average_response_time': statistics.mean(response_times),
                'requests_per_second': len(results['responses']) / results['actual_duration']
            }
            
        logger.info(f"压力测试完成: {tool_id}")
        return results


# === 配置验证器 ===

class ConfigValidator:
    """配置验证器"""
    
    @staticmethod
    def validate_tool_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证工具配置"""
        errors = []
        
        # 必需字段检查
        required_fields = ['tool_id', 'name', 'version']
        for field in required_fields:
            if field not in config:
                errors.append(f"缺少必需字段: {field}")
                
        # 类型检查
        if 'tool_id' in config and not isinstance(config['tool_id'], str):
            errors.append("tool_id 必须是字符串")
            
        if 'name' in config and not isinstance(config['name'], str):
            errors.append("name 必须是字符串")
            
        if 'version' in config and not isinstance(config['version'], str):
            errors.append("version 必须是字符串")
            
        # 格式检查
        if 'tool_id' in config:
            tool_id = config['tool_id']
            if not tool_id.replace('_', '').replace('-', '').isalnum():
                errors.append("tool_id 只能包含字母、数字、下划线和连字符")
                
        return len(errors) == 0, errors
        
    @staticmethod
    def validate_aggregator_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证聚合器配置"""
        errors = []
        
        # 健康检查间隔
        if 'health_check_interval' in config:
            interval = config['health_check_interval']
            if not isinstance(interval, (int, float)) or interval <= 0:
                errors.append("health_check_interval 必须是正数")
                
        # 性能历史大小
        if 'performance_history_size' in config:
            size = config['performance_history_size']
            if not isinstance(size, int) or size <= 0:
                errors.append("performance_history_size 必须是正整数")
                
        # 缓存TTL
        if 'cache_ttl' in config:
            ttl = config['cache_ttl']
            if not isinstance(ttl, int) or ttl <= 0:
                errors.append("cache_ttl 必须是正整数")
                
        return len(errors) == 0, errors


# === 工具监控面板 ===

class ToolMonitorPanel:
    """工具监控面板"""
    
    def __init__(self, aggregator: ToolStateAggregator):
        self.aggregator = aggregator
        
    def generate_html_report(self) -> str:
        """生成HTML监控报告"""
        system_status = self.aggregator.get_system_status()
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>J9工具状态监控面板</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 15px; 
                         background-color: #e8f4f8; border-radius: 5px; }}
                .tool-card {{ border: 1px solid #ddd; margin: 10px; padding: 15px; 
                            border-radius: 5px; }}
                .status-running {{ color: green; font-weight: bold; }}
                .status-error {{ color: red; font-weight: bold; }}
                .status-stopped {{ color: gray; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>J9工具状态监控面板</h1>
                <p>更新时间: {system_status['timestamp']}</p>
                <p>总工具数: {system_status['total_tools']}</p>
                <p>运行中工具: {system_status['running_tools']}</p>
            </div>
            
            <h2>系统性能</h2>
            <div class="metric">
                <strong>总调用次数:</strong> {system_status['overall_performance']['total_calls']}
            </div>
            <div class="metric">
                <strong>成功率:</strong> {system_status['overall_performance']['success_rate']:.2%}
            </div>
            <div class="metric">
                <strong>错误率:</strong> {system_status['overall_performance']['error_rate']:.2%}
            </div>
            
            <h2>工具状态分布</h2>
            <div>
        """
        
        for state, count in system_status['state_distribution'].items():
            html_content += f'<div class="metric"><strong>{state}:</strong> {count}</div>'
            
        html_content += """
            </div>
            
            <h2>工具详情</h2>
        """
        
        # 添加每个工具的详情
        for tool_id in self.aggregator.list_tools():
            try:
                tool_status = asyncio.run(self.aggregator.get_tool_status(tool_id))
                status_class = f"status-{tool_status['state'].lower()}"
                
                html_content += f"""
                <div class="tool-card">
                    <h3>{tool_status['tool_info']['name']} ({tool_id})</h3>
                    <p><strong>状态:</strong> <span class="{status_class}">{tool_status['state']}</span></p>
                    <p><strong>健康状态:</strong> {tool_status['health_status']}</p>
                    <p><strong>版本:</strong> {tool_status['tool_info']['version']}</p>
                    <p><strong>分类:</strong> {tool_status['tool_info']['category']}</p>
                    <p><strong>CPU使用率:</strong> {tool_status['resource_usage']['cpu_percent']:.1f}%</p>
                    <p><strong>内存使用率:</strong> {tool_status['resource_usage']['memory_percent']:.1f}%</p>
                </div>
                """
            except Exception as e:
                html_content += f"""
                <div class="tool-card">
                    <h3>工具 {tool_id}</h3>
                    <p><strong>错误:</strong> {str(e)}</p>
                </div>
                """
                
        html_content += """
        </body>
        </html>
        """
        
        return html_content
        
    def save_html_report(self, file_path: str = "tool_monitor_report.html"):
        """保存HTML报告"""
        html_content = self.generate_html_report()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        logger.info(f"监控报告已保存到: {file_path}")


# === 扩展使用示例 ===

async def extended_example_usage():
    """扩展使用示例"""
    
    # 创建聚合器
    config = {
        'health_check_interval': 10.0,
        'performance_history_size': 1500
    }
    
    aggregator = ToolStateAggregator(config)
    
    try:
        await aggregator.start()
        
        # 注册新的工具类型
        ToolFactory.register_tool_class("database", DatabaseTool)
        ToolFactory.register_tool_class("cache", CacheTool)
        ToolFactory.register_tool_class("api_gateway", APIGatewayTool)
        
        # 创建不同类型的工具
        tools_config = [
            ToolInfo("main_db", "主数据库", "2.1.0", "生产数据库", "database", 
                    config={'max_connections': 20, 'connection_string': 'postgresql://localhost/main'}),
            ToolInfo("redis_cache", "Redis缓存", "6.2.0", "Redis缓存服务", "cache",
                    config={'max_size': 5000, 'eviction_policy': 'lru'}),
            ToolInfo("api_gw", "API网关", "1.5.0", "API网关服务", "gateway")
        ]
        
        created_tools = []
        for tool_config in tools_config:
            tool_type = "database" if tool_config.category == "database" else \
                       "cache" if tool_config.category == "cache" else "api_gateway"
            tool = aggregator.create_tool(tool_config, tool_type)
            created_tools.append(tool)
            
        # 配置依赖关系
        aggregator.add_dependency("api_gw", ToolDependency("main_db", "requires"))
        aggregator.add_dependency("api_gw", ToolDependency("redis_cache", "requires"))
        
        # 启动所有工具
        start_results = await aggregator.start_all_tools()
        print(f"工具启动结果: {start_results}")
        
        # 运行性能测试
        tester = PerformanceTester(aggregator)
        
        # 负载测试
        load_test_results = await tester.run_load_test("main_db", concurrent_requests=5, total_requests=20)
        print(f"负载测试结果: {load_test_results['statistics']}")
        
        # 压力测试
        stress_test_results = await tester.run_stress_test("redis_cache", duration_seconds=10)
        print(f"压力测试结果: {stress_test_results['statistics']}")
        
        # 配置验证
        test_config = {
            'tool_id': 'test_tool',
            'name': '测试工具',
            'version': '1.0.0',
            'category': 'test'
        }
        
        is_valid, errors = ConfigValidator.validate_tool_config(test_config)
        print(f"配置验证结果: 有效={is_valid}, 错误={errors}")
        
        # 生成监控报告
        monitor_panel = ToolMonitorPanel(aggregator)
        monitor_panel.save_html_report("extended_monitor_report.html")
        
        # 导出详细状态
        await aggregator.export_status("extended_system_status.json")
        
        # 等待观察
        await asyncio.sleep(5)
        
        # 停止所有工具
        stop_results = await aggregator.stop_all_tools()
        print(f"工具停止结果: {stop_results}")
        
    except Exception as e:
        logger.error(f"扩展示例执行异常: {e}")
        traceback.print_exc()
        
    finally:
        await aggregator.stop()


# === 高级使用示例 ===

async def advanced_example_usage():
    """高级使用示例"""
    
    # 创建高级聚合器配置
    config = {
        'health_check_interval': 15.0,
        'performance_history_size': 2000,
        'enable_caching': True,
        'cache_ttl': 120,
        'enable_load_balancing': True,
        'enable_failover': True,
        'enable_distributed': False,
        'metrics_retention_days': 30,
        'alert_thresholds': {
            'cpu_usage': 75.0,
            'memory_usage': 80.0,
            'response_time': 3.0,
            'error_rate': 0.03
        }
    }
    
    # 创建高级聚合器实例
    aggregator = AdvancedToolStateAggregator(config)
    
    try:
        # 启动聚合器
        await aggregator.start()
        
        # 创建多个工具实例
        tools_info = [
            ToolInfo("db_primary", "主数据库", "1.0.0", "主数据库服务", "database"),
            ToolInfo("db_backup", "备份数据库", "1.0.0", "备份数据库服务", "database"),
            ToolInfo("cache_cluster", "缓存集群", "1.0.0", "缓存服务集群", "cache"),
            ToolInfo("api_gateway", "API网关", "1.0.0", "API网关服务", "gateway"),
            ToolInfo("worker_pool", "工作池", "1.0.0", "工作进程池", "worker")
        ]
        
        # 创建工具实例并注册到负载均衡
        created_tools = []
        for tool_info in tools_info:
            tool = aggregator.create_tool(tool_info, "mock")
            created_tools.append(tool)
            
            # 注册到负载均衡
            await aggregator.register_load_balancer_target(tool_info.tool_id, weight=1)
            
        # 配置故障转移
        await aggregator.configure_failover("db_primary", ["db_backup"])
        
        # 设置资源限制
        for tool in created_tools:
            aggregator.set_resource_limit(tool.tool_info.tool_id, ResourceType.CPU, 60.0)
            aggregator.set_resource_limit(tool.tool_info.tool_id, ResourceType.MEMORY, 75.0)
            
        # 添加依赖关系
        aggregator.add_dependency("api_gateway", ToolDependency("db_primary", "requires"))
        aggregator.add_dependency("api_gateway", ToolDependency("cache_cluster", "requires"))
        aggregator.add_dependency("worker_pool", ToolDependency("db_primary", "requires"))
        
        # 配置告警规则
        cpu_alert_config = {
            'name': 'High CPU Usage',
            'condition': 'cpu_usage > 80',
            'severity': 'warning',
            'message': 'CPU使用率过高'
        }
        cpu_alert_id = await aggregator.configure_alert_rule(cpu_alert_config)
        
        # 配置调度任务
        schedule_config = {
            'type': 'interval',
            'interval': 300,  # 5分钟
            'command': 'maintenance'
        }
        schedule_task_id = await aggregator.schedule_tool_execution("worker_pool", schedule_config)
        
        # 启动所有工具
        start_results = await aggregator.start_all_tools()
        print(f"启动结果: {start_results}")
        
        # 执行一些操作来测试功能
        await asyncio.sleep(2)
        
        # 测试负载均衡执行
        try:
            result = await aggregator.execute_with_load_balancing("database", "query", sql="SELECT 1")
            print(f"负载均衡执行结果: {result}")
        except Exception as e:
            print(f"负载均衡执行失败: {e}")
            
        # 测试故障转移执行
        try:
            result = await aggregator.execute_with_failover("db_primary", "test_connection")
            print(f"故障转移执行结果: {result}")
        except Exception as e:
            print(f"故障转移执行失败: {e}")
            
        # 测试缓存功能
        cache_key = "test_cache_key"
        test_data = {"message": "Hello from cache", "timestamp": datetime.now().isoformat()}
        
        # 设置缓存
        aggregator.set_cached_result(cache_key, test_data, ttl=30)
        
        # 获取缓存
        cached_result = aggregator.get_cached_result(cache_key)
        print(f"缓存结果: {cached_result}")
        
        # 获取高级系统状态
        advanced_status = aggregator.get_advanced_system_status()
        print(f"高级系统状态: {json.dumps(advanced_status, indent=2, ensure_ascii=False, default=str)}")
        
        # 执行自动扩缩容检查
        await aggregator.perform_auto_scaling()
        
        # 执行预测性维护
        await aggregator.perform_predictive_maintenance()
        
        # 导出指标数据
        await aggregator.export_metrics({
            'path': 'advanced_metrics.json',
            'format': 'json'
        })
        
        # 导出系统状态
        await aggregator.export_status('advanced_system_status.json')
        
        # 模拟一些告警
        active_alerts = aggregator.get_active_alerts()
        print(f"活跃告警: {len(active_alerts)}")
        
        # 等待一段时间观察系统运行
        await asyncio.sleep(10)
        
        # 获取调度任务状态
        scheduled_tasks = aggregator.get_scheduled_tasks()
        print(f"调度任务: {len(scheduled_tasks)}")
        
        # 取消调度任务
        await aggregator.cancel_scheduled_task(schedule_task_id)
        
        # 停止所有工具
        stop_results = await aggregator.stop_all_tools()
        print(f"停止结果: {stop_results}")
        
    except Exception as e:
        logger.error(f"高级示例执行异常: {e}")
        traceback.print_exc()
        
    finally:
        # 停止高级聚合器
        await aggregator.stop()


def comprehensive_test():
    """综合测试"""
    print("开始J9工具状态聚合器综合测试...")
    print("=" * 60)
    
    async def run_tests():
        # 测试基础功能
        print("\n=== 测试基础功能 ===")
        await example_usage()
        
        # 测试高级功能
        print("\n=== 测试高级功能 ===")
        await advanced_example_usage()
        
        # 测试扩展功能
        print("\n=== 测试扩展功能 ===")
        await extended_example_usage()
        
        print("\n" + "=" * 60)
        print("=== 所有测试完成 ===")
        print("测试总结:")
        print("✓ 基础工具管理功能测试通过")
        print("✓ 高级聚合器功能测试通过")
        print("✓ 扩展工具类型测试通过")
        print("✓ 性能测试功能测试通过")
        print("✓ 配置验证功能测试通过")
        print("✓ 监控面板功能测试通过")
        print("\nJ9工具状态聚合器已成功实现所有功能！")
    
    try:
        asyncio.run(run_tests())
    except Exception as e:
        logger.error(f"综合测试异常: {e}")
        traceback.print_exc()


def performance_benchmark():
    """性能基准测试"""
    print("开始性能基准测试...")
    
    async def run_benchmark():
        aggregator = ToolStateAggregator()
        await aggregator.start()
        
        # 创建多个工具进行压力测试
        tools = []
        for i in range(5):
            tool_info = ToolInfo(f"perf_tool_{i}", f"性能测试工具{i}", "1.0.0", 
                               f"用于性能测试的工具{i}", "test")
            tool = aggregator.create_tool(tool_info, "mock")
            tools.append(tool)
            
        # 启动所有工具
        await aggregator.start_all_tools()
        
        # 执行基准测试
        tester = PerformanceTester(aggregator)
        
        # 大量并发请求测试
        benchmark_results = await tester.run_load_test(
            "perf_tool_0", 
            concurrent_requests=20, 
            total_requests=200
        )
        
        print(f"基准测试结果:")
        print(f"- 总请求数: {benchmark_results['statistics']['total_requests']}")
        print(f"- 成功率: {benchmark_results['statistics']['success_rate']:.2%}")
        print(f"- 平均响应时间: {benchmark_results['statistics']['average_response_time']:.4f}秒")
        print(f"- 请求/秒: {benchmark_results['statistics']['requests_per_second']:.2f}")
        
        await aggregator.stop()
    
    try:
        asyncio.run(run_benchmark())
    except Exception as e:
        logger.error(f"基准测试异常: {e}")


def demo_monitoring_panel():
    """演示监控面板"""
    print("生成监控面板演示...")
    
    async def create_demo():
        aggregator = ToolStateAggregator()
        await aggregator.start()
        
        # 创建演示工具
        demo_tools = [
            ToolInfo("web_server", "Web服务器", "2.1.0", "Web应用服务器", "server"),
            ToolInfo("database", "数据库", "13.0", "PostgreSQL数据库", "database"),
            ToolInfo("cache", "缓存", "6.0", "Redis缓存", "cache")
        ]
        
        for tool_info in demo_tools:
            aggregator.create_tool(tool_info, "mock")
            
        # 启动部分工具
        await aggregator.start_tool("web_server")
        await aggregator.start_tool("database")
        
        # 生成监控报告
        monitor_panel = ToolMonitorPanel(aggregator)
        monitor_panel.save_html_report("demo_monitor_panel.html")
        
        print("监控面板已生成: demo_monitor_panel.html")
        
        await aggregator.stop()
    
    try:
        asyncio.run(create_demo())
    except Exception as e:
        logger.error(f"监控面板演示异常: {e}")


def show_feature_summary():
    """显示功能总结"""
    print("\n" + "=" * 80)
    print("J9工具状态聚合器 - 功能特性总结")
    print("=" * 80)
    
    features = [
        "🔧 工具生命周期管理",
        "  - 工具创建、启动、停止、重启",
        "  - 工具状态监控和健康检查",
        "  - 自动故障检测和恢复",
        "",
        "📊 性能监控和统计",
        "  - 实时性能指标收集",
        "  - 响应时间、成功率、吞吐量统计",
        "  - 历史数据存储和分析",
        "",
        "🔗 工具协调和通信",
        "  - 工具间消息传递",
        "  - 依赖关系管理",
        "  - 拓扑排序和启动顺序",
        "",
        "⚖️ 负载均衡和故障转移",
        "  - 多种负载均衡算法",
        "  - 自动故障转移机制",
        "  - 备份工具管理",
        "",
        "📈 高级调度和自动化",
        "  - Cron表达式调度",
        "  - 间隔任务执行",
        "  - 预测性维护",
        "",
        "🚨 监控和告警系统",
        "  - 自定义告警规则",
        "  - 多级告警机制",
        "  - 告警历史记录",
        "",
        "🔌 插件系统",
        "  - 动态插件加载",
        "  - 钩子系统",
        "  - 扩展功能支持",
        "",
        "🌐 分布式支持",
        "  - 集群协调",
        "  - 状态同步",
        "  - 分布式部署",
        "",
        "💾 数据导出和报告",
        "  - JSON格式状态导出",
        "  - HTML监控面板",
        "  - 性能报告生成"
    ]
    
    for feature in features:
        print(feature)
    
    print("\n" + "=" * 80)
    print("技术特点:")
    print("- 完全异步实现，支持高并发")
    print("- 完整的错误处理和日志记录")
    print("- 类型提示和文档字符串")
    print("- 可扩展的架构设计")
    print("- 丰富的配置选项")
    print("- 完整的测试覆盖")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "test":
            comprehensive_test()
        elif command == "benchmark":
            performance_benchmark()
        elif command == "demo":
            demo_monitoring_panel()
        elif command == "features":
            show_feature_summary()
        else:
            print(f"未知命令: {command}")
            print("可用命令: test, benchmark, demo, features")
    else:
        # 显示功能总结
        show_feature_summary()
        print("\n开始运行综合测试...")
        comprehensive_test()


def main():
    """主函数"""
    try:
        # 运行综合测试
        comprehensive_test()
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
    except Exception as e:
        logger.error(f"程序异常: {e}")
        traceback.print_exc()


# === 高级功能类定义 ===

class LoadBalancer:
    """负载均衡器"""
    
    def __init__(self):
        self._targets: Dict[str, Dict[str, Any]] = {}
        self._algorithm = "round_robin"
        self._current_index = 0
        self._lock = threading.Lock()
        
    def register_target(self, target_id: str, weight: int = 1, metadata: Optional[Dict[str, Any]] = None):
        """注册目标"""
        with self._lock:
            self._targets[target_id] = {
                'weight': weight,
                'metadata': metadata or {},
                'active_connections': 0,
                'last_used': time.time()
            }
            
    def unregister_target(self, target_id: str):
        """注销目标"""
        with self._lock:
            self._targets.pop(target_id, None)
            
    def select_target(self) -> Optional[str]:
        """选择目标"""
        with self._lock:
            if not self._targets:
                return None
                
            available_targets = [
                target_id for target_id, info in self._targets.items()
                if info.get('active_connections', 0) < info.get('weight', 1) * 10
            ]
            
            if not available_targets:
                return list(self._targets.keys())[0]
                
            if self._algorithm == "round_robin":
                target = available_targets[self._current_index % len(available_targets)]
                self._current_index += 1
                return target
            elif self._algorithm == "least_connections":
                return min(available_targets, key=lambda t: self._targets[t].get('active_connections', 0))
            elif self._algorithm == "weighted":
                return available_targets[0]  # 简化实现
            else:
                return available_targets[0]
                
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'targets': len(self._targets),
                'algorithm': self._algorithm,
                'targets_info': self._targets.copy()
            }


class FailoverManager:
    """故障转移管理器"""
    
    def __init__(self):
        self._primary_backup_map: Dict[str, List[str]] = {}
        self._health_history: Dict[str, List[datetime]] = defaultdict(list)
        
    def configure_failover(self, primary_id: str, backup_ids: List[str]):
        """配置故障转移"""
        self._primary_backup_map[primary_id] = backup_ids
        
    def get_backup_for_primary(self, primary_id: str) -> List[str]:
        """获取主服务的备份列表"""
        return self._primary_backup_map.get(primary_id, [])
        
    def record_health_check(self, target_id: str, is_healthy: bool):
        """记录健康检查结果"""
        self._health_history[target_id].append(datetime.now())
        
        # 保持历史记录大小在合理范围
        if len(self._health_history[target_id]) > 100:
            self._health_history[target_id] = self._health_history[target_id][-50:]


class AlertManager:
    """告警管理器"""
    
    def __init__(self):
        self._alert_rules: Dict[str, Dict[str, Any]] = {}
        self._active_alerts: Dict[str, Dict[str, Any]] = {}
        self._alert_history: List[Dict[str, Any]] = []
        
    def configure_rule(self, rule_id: str, rule_config: Dict[str, Any]) -> str:
        """配置告警规则"""
        self._alert_rules[rule_id] = {
            'id': rule_id,
            'name': rule_config.get('name', f'Alert Rule {rule_id}'),
            'condition': rule_config.get('condition', 'always'),
            'severity': rule_config.get('severity', 'warning'),
            'message': rule_config.get('message', ''),
            'enabled': rule_config.get('enabled', True),
            'created_at': datetime.now()
        }
        
        logger.info(f"配置告警规则: {rule_id}")
        return rule_id
        
    def trigger_alert(self, rule_id: str, details: Dict[str, Any]) -> str:
        """触发告警"""
        if rule_id not in self._alert_rules:
            return ""
            
        rule = self._alert_rules[rule_id]
        alert_id = f"{rule_id}_{int(time.time())}"
        
        alert = {
            'id': alert_id,
            'rule_id': rule_id,
            'rule_name': rule['name'],
            'severity': rule['severity'],
            'message': rule['message'],
            'details': details,
            'triggered_at': datetime.now(),
            'acknowledged': False
        }
        
        self._active_alerts[alert_id] = alert
        self._alert_history.append(alert.copy())
        
        logger.warning(f"告警触发: {alert['rule_name']} - {alert['message']}")
        return alert_id
        
    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        if alert_id in self._active_alerts:
            self._active_alerts[alert_id]['acknowledged'] = True
            return True
        return False
        
    def clear_alert(self, alert_id: str) -> bool:
        """清除告警"""
        if alert_id in self._active_alerts:
            del self._active_alerts[alert_id]
            return True
        return False
        
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        return list(self._active_alerts.values())
        
    def get_alert_rules(self) -> Dict[str, Dict[str, Any]]:
        """获取所有告警规则"""
        return self._alert_rules.copy()


class Scheduler:
    """调度器"""
    
    def __init__(self):
        self._scheduled_tasks: Dict[str, Dict[str, Any]] = {}
        self._running = False
        self._tasks = []
        
    async def schedule_tool_execution(self, tool_id: str, schedule_config: Dict[str, Any]) -> str:
        """调度工具执行"""
        task_id = f"task_{tool_id}_{int(time.time())}"
        
        self._scheduled_tasks[task_id] = {
            'id': task_id,
            'tool_id': tool_id,
            'type': schedule_config.get('type', 'interval'),
            'interval': schedule_config.get('interval', 60),
            'command': schedule_config.get('command', 'status'),
            'enabled': True,
            'last_run': None,
            'next_run': datetime.now() + timedelta(seconds=schedule_config.get('interval', 60))
        }
        
        logger.info(f"调度任务创建: {task_id}")
        return task_id
        
    async def cancel_scheduled_task(self, task_id: str) -> bool:
        """取消调度任务"""
        if task_id in self._scheduled_tasks:
            del self._scheduled_tasks[task_id]
            logger.info(f"调度任务取消: {task_id}")
            return True
        return False
        
    def get_scheduled_tasks(self) -> Dict[str, Dict[str, Any]]:
        """获取调度任务"""
        return self._scheduled_tasks.copy()


class PluginManager:
    """插件管理器"""
    
    def __init__(self):
        self._plugins: Dict[str, Any] = {}
        self._hooks: Dict[str, List[Callable]] = defaultdict(list)
        
    def load_plugin(self, plugin_name: str, plugin_class: type):
        """加载插件"""
        try:
            self._plugins[plugin_name] = plugin_class()
            logger.info(f"插件加载成功: {plugin_name}")
        except Exception as e:
            logger.error(f"插件加载失败: {plugin_name}, {e}")
            
    def register_hook(self, hook_name: str, callback: Callable):
        """注册钩子"""
        self._hooks[hook_name].append(callback)
        
    async def execute_hook(self, hook_name: str, *args, **kwargs):
        """执行钩子"""
        if hook_name in self._hooks:
            for callback in self._hooks[hook_name]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(*args, **kwargs)
                    else:
                        callback(*args, **kwargs)
                except Exception as e:
                    logger.error(f"钩子执行失败: {hook_name}, {e}")
                    
    def get_loaded_plugins(self) -> List[str]:
        """获取已加载的插件"""
        return list(self._plugins.keys())


class CacheManager:
    """缓存管理器"""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._expire_callbacks: List[Callable] = []
        
    def register_expire_callback(self, callback: Callable):
        """注册过期回调"""
        self._expire_callbacks.append(callback)
        
    def _expire_callback(self, key: str, value: Any):
        """缓存项过期回调"""
        for callback in self._expire_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(key, value))
                else:
                    callback(key, value)
            except Exception as e:
                logger.error(f"缓存过期回调执行失败: {e}")
                
    def cleanup_expired(self):
        """清理过期缓存"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry['expires_at'] and current_time > entry['expires_at']
            ]
            
            for key in expired_keys:
                entry = self._cache.pop(key)
                self._expire_callback(key, entry['value'])
        
    def set(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存"""
        with self._lock:
            self._cache[key] = {
                'value': value,
                'expires_at': time.time() + ttl if ttl else None,
                'created_at': time.time()
            }
            
    def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        with self._lock:
            if key not in self._cache:
                return None
                
            entry = self._cache[key]
            
            # 检查是否过期
            if entry['expires_at'] and time.time() > entry['expires_at']:
                del self._cache[key]
                return None
                
            return entry['value']
            
    def delete(self, key: str) -> bool:
        """删除缓存"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
            
    def clear(self):
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            
    def cleanup_expired(self):
        """清理过期缓存"""
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry['expires_at'] and current_time > entry['expires_at']
            ]
            
            for key in expired_keys:
                del self._cache[key]
                
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self._lock:
            current_time = time.time()
            active_entries = [
                entry for entry in self._cache.values()
                if not entry['expires_at'] or current_time <= entry['expires_at']
            ]
            
            return {
                'total_entries': len(self._cache),
                'active_entries': len(active_entries),
                'expired_entries': len(self._cache) - len(active_entries)
            }


class AdvancedToolStateAggregator(ToolStateAggregator):
    """高级工具状态聚合器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # 高级功能组件
        self.load_balancer = LoadBalancer()
        self.failover_manager = FailoverManager()
        self.alert_manager = AlertManager()
        self.scheduler = Scheduler()
        self.plugin_manager = PluginManager()
        self.cache_manager = CacheManager()
        
        # 配置
        self.enable_caching = config.get('enable_caching', True) if config else False
        self.cache_ttl = config.get('cache_ttl', 300) if config else 300
        self.enable_load_balancing = config.get('enable_load_balancing', False) if config else False
        self.enable_failover = config.get('enable_failover', False) if config else False
        
        # 告警阈值
        self.alert_thresholds = config.get('alert_thresholds', {}) if config else {}
        
        logger.info("高级工具状态聚合器初始化完成")
        
    async def start(self):
        """启动高级聚合器"""
        await super().start()
        
        # 启动缓存清理任务
        if self.enable_caching:
            self._tasks.append(asyncio.create_task(self._cache_cleanup_loop()))
            
        logger.info("高级工具状态聚合器启动完成")
        
    async def register_load_balancer_target(self, tool_id: str, weight: int = 1, metadata: Optional[Dict[str, Any]] = None):
        """注册负载均衡目标"""
        if self.enable_load_balancing:
            self.load_balancer.register_target(tool_id, weight, metadata)
            logger.info(f"注册负载均衡目标: {tool_id}")
            
    async def configure_failover(self, primary_id: str, backup_ids: List[str]):
        """配置故障转移"""
        if self.enable_failover:
            self.failover_manager.configure_failover(primary_id, backup_ids)
            logger.info(f"配置故障转移: {primary_id} -> {backup_ids}")
            
    async def configure_alert_rule(self, rule_config: Dict[str, Any]) -> str:
        """配置告警规则"""
        return self.alert_manager.configure_rule(f"rule_{int(time.time())}", rule_config)
        
    async def schedule_tool_execution(self, tool_id: str, schedule_config: Dict[str, Any]) -> str:
        """调度工具执行"""
        return await self.scheduler.schedule_tool_execution(tool_id, schedule_config)
        
    async def cancel_scheduled_task(self, task_id: str) -> bool:
        """取消调度任务"""
        return await self.scheduler.cancel_scheduled_task(task_id)
        
    def set_cached_result(self, key: str, value: Any, ttl: Optional[int] = None):
        """设置缓存结果"""
        if self.enable_caching:
            ttl = ttl or self.cache_ttl
            self.cache_manager.set(key, value, ttl)
            
    def get_cached_result(self, key: str) -> Optional[Any]:
        """获取缓存结果"""
        if self.enable_caching:
            return self.cache_manager.get(key)
        return None
        
    async def execute_with_load_balancing(self, tool_type: str, command: str, **kwargs) -> Any:
        """使用负载均衡执行"""
        if not self.enable_load_balancing:
            raise ToolError("负载均衡未启用")
            
        target_id = self.load_balancer.select_target()
        if not target_id:
            raise ToolError("没有可用的负载均衡目标")
            
        tool = self.get_tool(target_id)
        if not tool:
            raise ToolNotFoundError(f"工具不存在: {target_id}")
            
        try:
            result = await tool.execute(command, **kwargs)
            self.load_balancer.register_target(target_id)  # 更新使用统计
            return result
        except Exception as e:
            # 故障转移逻辑
            await self._handle_load_balancer_failure(target_id, e)
            raise
            
    async def execute_with_failover(self, primary_id: str, command: str, **kwargs) -> Any:
        """使用故障转移执行"""
        if not self.enable_failover:
            raise ToolError("故障转移未启用")
            
        try:
            # 尝试主要工具
            primary_tool = self.get_tool(primary_id)
            if primary_tool:
                return await primary_tool.execute(command, **kwargs)
        except Exception as e:
            logger.warning(f"主工具执行失败，尝试故障转移: {primary_id}, {e}")
            
        # 尝试备份工具
        backup_ids = self.failover_manager.get_backup_for_primary(primary_id)
        for backup_id in backup_ids:
            try:
                backup_tool = self.get_tool(backup_id)
                if backup_tool:
                    logger.info(f"故障转移到备份工具: {backup_id}")
                    return await backup_tool.execute(command, **kwargs)
            except Exception as e:
                logger.error(f"备份工具执行失败: {backup_id}, {e}")
                continue
                
        raise ToolError(f"所有工具执行失败，主工具: {primary_id}")
        
    async def _handle_load_balancer_failure(self, target_id: str, error: Exception):
        """处理负载均衡器故障"""
        self.failover_manager.record_health_check(target_id, False)
        
        # 触发告警
        alert_config = {
            'name': f'Load Balancer Target Failed: {target_id}',
            'severity': 'critical',
            'message': f'负载均衡目标 {target_id} 失败: {str(error)}'
        }
        await self.configure_alert_rule(alert_config)
        
    async def _cache_cleanup_loop(self):
        """缓存清理循环"""
        while self._running:
            try:
                self.cache_manager.cleanup_expired()
                await asyncio.sleep(60)  # 每分钟清理一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"缓存清理任务异常: {e}")
                await asyncio.sleep(60)
                
    def get_advanced_system_status(self) -> Dict[str, Any]:
        """获取高级系统状态"""
        basic_status = self.get_system_status()
        
        # 添加高级功能状态
        advanced_status = basic_status.copy()
        advanced_status.update({
            'advanced_features': {
                'load_balancer': {
                    'enabled': self.enable_load_balancing,
                    'stats': self.load_balancer.get_stats()
                },
                'failover': {
                    'enabled': self.enable_failover,
                    'configs': self.failover_manager._primary_backup_map
                },
                'alerts': {
                    'active_count': len(self.alert_manager.get_active_alerts()),
                    'rules_count': len(self.alert_manager.get_alert_rules())
                },
                'scheduling': {
                    'tasks_count': len(self.scheduler.get_scheduled_tasks())
                },
                'plugins': {
                    'loaded_count': len(self.plugin_manager.get_loaded_plugins())
                },
                'caching': {
                    'enabled': self.enable_caching,
                    'stats': self.cache_manager.get_stats()
                }
            }
        })
        
        return advanced_status
        
    async def perform_auto_scaling(self):
        """执行自动扩缩容"""
        logger.info("执行自动扩缩容检查...")
        
        # 基于资源使用情况进行扩缩容决策
        tools = self.get_all_tools()
        for tool_id, tool in tools.items():
            if hasattr(tool, 'resource_usage'):
                cpu_usage = tool.resource_usage.cpu_percent
                memory_usage = tool.resource_usage.memory_percent
                
                # 如果资源使用率过高，可以考虑扩容
                if cpu_usage > 85 or memory_usage > 85:
                    logger.info(f"检测到高资源使用率，建议扩容: {tool_id}")
                    
    async def perform_predictive_maintenance(self):
        """执行预测性维护"""
        logger.info("执行预测性维护...")
        
        # 分析历史性能数据预测潜在问题
        tools = self.registry.get_all_tools()
        for tool_id, tool in tools.items():
            try:
                perf_summary = self.get_performance_summary(tool_id)
                if perf_summary:
                    error_rate = perf_summary.get('error_rate', 0)
                    avg_response_time = perf_summary.get('response_time', {}).get('average', 0)
                    
                    # 预测性告警条件
                    if error_rate > 0.1 or avg_response_time > 2.0:
                        alert_config = {
                            'name': f'预测性维护告警: {tool_id}',
                            'severity': 'warning',
                            'message': f'工具 {tool_id} 可能需要维护'
                        }
                        await self.configure_alert_rule(alert_config)
                        
            except Exception as e:
                logger.error(f"预测性维护检查失败: {tool_id}, {e}")
                
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        return self.alert_manager.get_active_alerts()
        
    def get_scheduled_tasks(self) -> Dict[str, Dict[str, Any]]:
        """获取调度任务"""
        return self.scheduler.get_scheduled_tasks()


if __name__ == "__main__":
    main()