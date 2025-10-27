"""
生命周期管理器
负责管理所有模块的生命周期状态，包括初始化、启动、停止和销毁
提供状态监控、健康检查和优雅关闭功能
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Set, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, Future
import signal
import sys

class LifecycleState(Enum):
    """生命周期状态枚举"""
    UNINITIALIZED = "uninitialized"  # 未初始化
    INITIALIZING = "initializing"    # 初始化中
    INITIALIZED = "initialized"      # 已初始化
    STARTING = "starting"            # 启动中
    RUNNING = "running"              # 运行中
    STOPPING = "stopping"            # 停止中
    STOPPED = "stopped"              # 已停止
    DESTROYING = "destroying"        # 销毁中
    DESTROYED = "destroyed"          # 已销毁
    ERROR = "error"                  # 错误

class HealthStatus(Enum):
    """健康状态枚举"""
    HEALTHY = "healthy"              # 健康
    WARNING = "warning"              # 警告
    UNHEALTHY = "unhealthy"          # 不健康
    UNKNOWN = "unknown"              # 未知

@dataclass
class ModuleLifecycleInfo:
    """模块生命周期信息数据类"""
    module_id: str
    module_name: str
    zone: str
    current_state: LifecycleState
    previous_state: LifecycleState
    state_history: List[Dict[str, Any]]
    health_status: HealthStatus
    health_check_timestamp: float
    initialization_time: Optional[float]
    start_time: Optional[float]
    stop_time: Optional[float]
    destroy_time: Optional[float]
    error_count: int
    last_error: Optional[str]
    dependencies: List[str]
    metadata: Dict[str, Any]

@dataclass
class SystemHealthReport:
    """系统健康报告数据类"""
    timestamp: float
    overall_status: HealthStatus
    module_health: Dict[str, HealthStatus]
    healthy_modules: int
    total_modules: int
    health_checks_passed: int
    health_checks_failed: int
    details: Dict[str, Any]

class LifecycleManager:
    """生命周期管理器 - 生产环境级别实现"""
    
    def __init__(self, registry, factory, config_manager=None):
        self.registry = registry
        self.factory = factory
        self.config_manager = config_manager
        
        # 生命周期状态跟踪
        self._module_lifecycles = OrderedDict()  # 模块ID -> ModuleLifecycleInfo
        self._state_listeners = defaultdict(list)  # 状态 -> 监听器列表
        self._health_checkers = {}  # 模块ID -> 健康检查函数
        
        # 线程安全
        self._lifecycle_lock = threading.RLock()
        self._health_check_lock = threading.RLock()
        
        # 配置
        self.auto_initialize = True
        self.auto_start = False
        self.shutdown_timeout = 30.0
        self.health_check_interval = 60.0
        self.enable_health_monitoring = True
        self.parallel_initialization = True
        self.parallel_shutdown = True
        
        # 状态管理
        self._system_state = LifecycleState.UNINITIALIZED
        self._shutdown_event = threading.Event()
        self._health_check_thread = None
        
        # 性能统计
        self._lifecycle_stats = {
            'total_initializations': 0,
            'total_starts': 0,
            'total_stops': 0,
            'total_destructions': 0,
            'total_errors': 0,
            'start_time': time.time()
        }
        
        # 事件回调
        self._pre_initialization_callbacks = []
        self._post_initialization_callbacks = []
        self._pre_start_callbacks = []
        self._post_start_callbacks = []
        self._pre_stop_callbacks = []
        self._post_stop_callbacks = []
        self._pre_destruction_callbacks = []
        self._post_destruction_callbacks = []
        
        # 日志
        self.logger = logging.getLogger('AOO.Lifecycle')
        
        # 初始化
        self._load_config()
        self._setup_signal_handlers()
    
    def _load_config(self):
        """加载配置"""
        if self.config_manager:
            try:
                lifecycle_config = self.config_manager.get_section('lifecycle') or {}
                self.auto_initialize = lifecycle_config.get('auto_initialize', True)
                self.auto_start = lifecycle_config.get('auto_start', False)
                self.shutdown_timeout = lifecycle_config.get('shutdown_timeout', 30.0)
                self.health_check_interval = lifecycle_config.get('health_check_interval', 60.0)
                self.enable_health_monitoring = lifecycle_config.get('enable_health_monitoring', True)
                self.parallel_initialization = lifecycle_config.get('parallel_initialization', True)
                self.parallel_shutdown = lifecycle_config.get('parallel_shutdown', True)
            except Exception as e:
                self.logger.warning(f"加载生命周期管理器配置失败: {e}")
    
    def _setup_signal_handlers(self):
        """设置信号处理器"""
        try:
            signal.signal(signal.SIGINT, self._handle_shutdown_signal)
            signal.signal(signal.SIGTERM, self._handle_shutdown_signal)
        except Exception as e:
            self.logger.warning(f"设置信号处理器失败: {e}")
    
    def _handle_shutdown_signal(self, signum, frame):
        """处理关闭信号"""
        self.logger.info(f"接收到关闭信号: {signum}")
        self.shutdown()
    
    def register_module(self, module_id: str, module_name: str, zone: str, dependencies: List[str] = None):
        """注册模块到生命周期管理器"""
        with self._lifecycle_lock:
            if module_id in self._module_lifecycles:
                self.logger.warning(f"模块已注册: {module_id}")
                return
            
            lifecycle_info = ModuleLifecycleInfo(
                module_id=module_id,
                module_name=module_name,
                zone=zone,
                current_state=LifecycleState.UNINITIALIZED,
                previous_state=LifecycleState.UNINITIALIZED,
                state_history=[],
                health_status=HealthStatus.UNKNOWN,
                health_check_timestamp=0.0,
                initialization_time=None,
                start_time=None,
                stop_time=None,
                destroy_time=None,
                error_count=0,
                last_error=None,
                dependencies=dependencies or [],
                metadata={}
            )
            
            self._module_lifecycles[module_id] = lifecycle_info
            self._update_state_history(module_id, LifecycleState.UNINITIALIZED, "registered")
            
            self.logger.debug(f"模块已注册到生命周期管理器: {module_id}")
    
    def initialize_module(self, module_id: str) -> bool:
        """初始化单个模块"""
        with self._lifecycle_lock:
            if module_id not in self._module_lifecycles:
                self.logger.error(f"模块未注册: {module_id}")
                return False
            
            lifecycle_info = self._module_lifecycles[module_id]
            
            # 检查当前状态
            if lifecycle_info.current_state != LifecycleState.UNINITIALIZED:
                self.logger.warning(f"模块状态不正确，无法初始化: {module_id} (当前状态: {lifecycle_info.current_state.value})")
                return False
            
            # 检查依赖是否已初始化
            if not self._check_dependencies_initialized(module_id):
                self.logger.error(f"模块依赖未满足，无法初始化: {module_id}")
                return False
            
            try:
                # 更新状态
                self._update_module_state(module_id, LifecycleState.INITIALIZING, "initialization_started")
                
                # 获取模块实例
                instance = self.factory.get_instance(lifecycle_info.module_name)
                if not instance:
                    raise ValueError(f"无法获取模块实例: {lifecycle_info.module_name}")
                
                # 执行预初始化回调
                self._execute_pre_initialization_callbacks(module_id, instance)
                
                # 调用模块的初始化方法
                if hasattr(instance, 'initialize'):
                    if asyncio.iscoroutinefunction(instance.initialize):
                        # 异步初始化
                        asyncio.create_task(self._async_initialize_module(module_id, instance))
                    else:
                        # 同步初始化
                        if instance.initialize():
                            self._update_module_state(module_id, LifecycleState.INITIALIZED, "initialization_completed")
                            lifecycle_info.initialization_time = time.time()
                            self._lifecycle_stats['total_initializations'] += 1
                            
                            # 执行后初始化回调
                            self._execute_post_initialization_callbacks(module_id, instance)
                            
                            self.logger.info(f"模块初始化成功: {module_id}")
                            return True
                        else:
                            raise RuntimeError("模块初始化方法返回False")
                else:
                    # 模块没有初始化方法，直接标记为已初始化
                    self._update_module_state(module_id, LifecycleState.INITIALIZED, "no_initialize_method")
                    lifecycle_info.initialization_time = time.time()
                    self._lifecycle_stats['total_initializations'] += 1
                    self.logger.debug(f"模块无初始化方法，标记为已初始化: {module_id}")
                    return True
                
            except Exception as e:
                self._handle_module_error(module_id, f"初始化失败: {str(e)}")
                return False
        
        return False
    
    async def _async_initialize_module(self, module_id: str, instance: Any):
        """异步初始化模块"""
        try:
            if await instance.initialize():
                with self._lifecycle_lock:
                    lifecycle_info = self._module_lifecycles[module_id]
                    self._update_module_state(module_id, LifecycleState.INITIALIZED, "async_initialization_completed")
                    lifecycle_info.initialization_time = time.time()
                    self._lifecycle_stats['total_initializations'] += 1
                    
                    # 执行后初始化回调
                    self._execute_post_initialization_callbacks(module_id, instance)
                    
                    self.logger.info(f"模块异步初始化成功: {module_id}")
            else:
                self._handle_module_error(module_id, "异步初始化方法返回False")
        except Exception as e:
            self._handle_module_error(module_id, f"异步初始化失败: {str(e)}")
    
    def initialize_all_modules(self) -> bool:
        """初始化所有模块"""
        with self._lifecycle_lock:
            self._system_state = LifecycleState.INITIALIZING
            
            # 执行预初始化回调
            self._execute_pre_initialization_callbacks_system()
            
            # 获取所有模块ID
            module_ids = list(self._module_lifecycles.keys())
            
            # 根据依赖关系排序
            ordered_modules = self._get_initialization_order(module_ids)
            
            success = True
            
            if self.parallel_initialization and len(ordered_modules) > 1:
                # 并行初始化
                success = self._initialize_modules_parallel(ordered_modules)
            else:
                # 串行初始化
                success = self._initialize_modules_serial(ordered_modules)
            
            if success:
                self._system_state = LifecycleState.INITIALIZED
                self._execute_post_initialization_callbacks_system()
                self.logger.info("所有模块初始化完成")
            else:
                self._system_state = LifecycleState.ERROR
                self.logger.error("模块初始化失败")
            
            return success
    
    def _get_initialization_order(self, module_ids: List[str]) -> List[str]:
        """根据依赖关系获取初始化顺序"""
        # 构建依赖图
        dependency_graph = {}
        for module_id in module_ids:
            lifecycle_info = self._module_lifecycles[module_id]
            dependency_graph[module_id] = set(lifecycle_info.dependencies)
        
        # 拓扑排序
        try:
            from graphlib import TopologicalSorter
            sorter = TopologicalSorter(dependency_graph)
            ordered = list(sorter.static_order())
            return ordered
        except Exception as e:
            self.logger.warning(f"拓扑排序失败，使用原始顺序: {e}")
            return module_ids
    
    def _initialize_modules_serial(self, module_ids: List[str]) -> bool:
        """串行初始化模块"""
        success_count = 0
        
        for module_id in module_ids:
            if self.initialize_module(module_id):
                success_count += 1
            else:
                self.logger.error(f"模块初始化失败: {module_id}")
                # 继续初始化其他模块
        
        return success_count == len(module_ids)
    
    def _initialize_modules_parallel(self, module_ids: List[str]) -> bool:
        """并行初始化模块"""
        success_count = 0
        futures = {}
        
        with ThreadPoolExecutor(max_workers=min(len(module_ids), 4)) as executor:
            # 提交初始化任务
            for module_id in module_ids:
                future = executor.submit(self.initialize_module, module_id)
                futures[future] = module_id
            
            # 收集结果
            for future in futures:
                module_id = futures[future]
                try:
                    if future.result(timeout=30.0):  # 30秒超时
                        success_count += 1
                    else:
                        self.logger.error(f"模块并行初始化失败: {module_id}")
                except Exception as e:
                    self.logger.error(f"模块并行初始化异常: {module_id}, {e}")
        
        return success_count == len(module_ids)
    
    def start_module(self, module_id: str) -> bool:
        """启动单个模块"""
        with self._lifecycle_lock:
            if module_id not in self._module_lifecycles:
                self.logger.error(f"模块未注册: {module_id}")
                return False
            
            lifecycle_info = self._module_lifecycles[module_id]
            
            # 检查当前状态
            if lifecycle_info.current_state != LifecycleState.INITIALIZED:
                self.logger.warning(f"模块状态不正确，无法启动: {module_id} (当前状态: {lifecycle_info.current_state.value})")
                return False
            
            try:
                # 更新状态
                self._update_module_state(module_id, LifecycleState.STARTING, "startup_started")
                
                # 获取模块实例
                instance = self.factory.get_instance(lifecycle_info.module_name)
                if not instance:
                    raise ValueError(f"无法获取模块实例: {lifecycle_info.module_name}")
                
                # 执行预启动回调
                self._execute_pre_start_callbacks(module_id, instance)
                
                # 调用模块的启动方法
                if hasattr(instance, 'start'):
                    if asyncio.iscoroutinefunction(instance.start):
                        # 异步启动
                        asyncio.create_task(self._async_start_module(module_id, instance))
                    else:
                        # 同步启动
                        if instance.start():
                            self._update_module_state(module_id, LifecycleState.RUNNING, "startup_completed")
                            lifecycle_info.start_time = time.time()
                            self._lifecycle_stats['total_starts'] += 1
                            
                            # 执行后启动回调
                            self._execute_post_start_callbacks(module_id, instance)
                            
                            self.logger.info(f"模块启动成功: {module_id}")
                            return True
                        else:
                            raise RuntimeError("模块启动方法返回False")
                else:
                    # 模块没有启动方法，直接标记为运行中
                    self._update_module_state(module_id, LifecycleState.RUNNING, "no_start_method")
                    lifecycle_info.start_time = time.time()
                    self._lifecycle_stats['total_starts'] += 1
                    self.logger.debug(f"模块无启动方法，标记为运行中: {module_id}")
                    return True
                
            except Exception as e:
                self._handle_module_error(module_id, f"启动失败: {str(e)}")
                return False
        
        return False
    
    async def _async_start_module(self, module_id: str, instance: Any):
        """异步启动模块"""
        try:
            if await instance.start():
                with self._lifecycle_lock:
                    lifecycle_info = self._module_lifecycles[module_id]
                    self._update_module_state(module_id, LifecycleState.RUNNING, "async_startup_completed")
                    lifecycle_info.start_time = time.time()
                    self._lifecycle_stats['total_starts'] += 1
                    
                    # 执行后启动回调
                    self._execute_post_start_callbacks(module_id, instance)
                    
                    self.logger.info(f"模块异步启动成功: {module_id}")
            else:
                self._handle_module_error(module_id, "异步启动方法返回False")
        except Exception as e:
            self._handle_module_error(module_id, f"异步启动失败: {str(e)}")
    
    def start_all_modules(self) -> bool:
        """启动所有模块"""
        with self._lifecycle_lock:
            if self._system_state != LifecycleState.INITIALIZED:
                self.logger.error(f"系统状态不正确，无法启动: {self._system_state.value}")
                return False
            
            self._system_state = LifecycleState.STARTING
            
            # 执行预启动回调
            self._execute_pre_start_callbacks_system()
            
            # 获取所有已初始化的模块
            module_ids = [
                module_id for module_id, info in self._module_lifecycles.items()
                if info.current_state == LifecycleState.INITIALIZED
            ]
            
            success_count = 0
            
            for module_id in module_ids:
                if self.start_module(module_id):
                    success_count += 1
                else:
                    self.logger.error(f"模块启动失败: {module_id}")
                    # 继续启动其他模块
            
            if success_count == len(module_ids):
                self._system_state = LifecycleState.RUNNING
                self._execute_post_start_callbacks_system()
                self.logger.info("所有模块启动完成")
                
                # 启动健康检查（如果启用）
                if self.enable_health_monitoring:
                    self._start_health_monitoring()
                
                return True
            else:
                self._system_state = LifecycleState.ERROR
                self.logger.error("模块启动失败")
                return False
    
    def stop_module(self, module_id: str) -> bool:
        """停止单个模块"""
        with self._lifecycle_lock:
            if module_id not in self._module_lifecycles:
                self.logger.error(f"模块未注册: {module_id}")
                return False
            
            lifecycle_info = self._module_lifecycles[module_id]
            
            # 检查当前状态
            if lifecycle_info.current_state not in [LifecycleState.RUNNING, LifecycleState.ERROR]:
                self.logger.warning(f"模块状态不正确，无法停止: {module_id} (当前状态: {lifecycle_info.current_state.value})")
                return False
            
            try:
                # 更新状态
                self._update_module_state(module_id, LifecycleState.STOPPING, "shutdown_started")
                
                # 获取模块实例
                instance = self.factory.get_instance(lifecycle_info.module_name)
                if not instance:
                    raise ValueError(f"无法获取模块实例: {lifecycle_info.module_name}")
                
                # 执行预停止回调
                self._execute_pre_stop_callbacks(module_id, instance)
                
                # 调用模块的停止方法
                if hasattr(instance, 'stop'):
                    if asyncio.iscoroutinefunction(instance.stop):
                        # 异步停止
                        asyncio.create_task(self._async_stop_module(module_id, instance))
                    else:
                        # 同步停止
                        if instance.stop():
                            self._update_module_state(module_id, LifecycleState.STOPPED, "shutdown_completed")
                            lifecycle_info.stop_time = time.time()
                            self._lifecycle_stats['total_stops'] += 1
                            
                            # 执行后停止回调
                            self._execute_post_stop_callbacks(module_id, instance)
                            
                            self.logger.info(f"模块停止成功: {module_id}")
                            return True
                        else:
                            raise RuntimeError("模块停止方法返回False")
                else:
                    # 模块没有停止方法，直接标记为已停止
                    self._update_module_state(module_id, LifecycleState.STOPPED, "no_stop_method")
                    lifecycle_info.stop_time = time.time()
                    self._lifecycle_stats['total_stops'] += 1
                    self.logger.debug(f"模块无停止方法，标记为已停止: {module_id}")
                    return True
                
            except Exception as e:
                self._handle_module_error(module_id, f"停止失败: {str(e)}")
                return False
        
        return False
    
    async def _async_stop_module(self, module_id: str, instance: Any):
        """异步停止模块"""
        try:
            if await instance.stop():
                with self._lifecycle_lock:
                    lifecycle_info = self._module_lifecycles[module_id]
                    self._update_module_state(module_id, LifecycleState.STOPPED, "async_shutdown_completed")
                    lifecycle_info.stop_time = time.time()
                    self._lifecycle_stats['total_stops'] += 1
                    
                    # 执行后停止回调
                    self._execute_post_stop_callbacks(module_id, instance)
                    
                    self.logger.info(f"模块异步停止成功: {module_id}")
            else:
                self._handle_module_error(module_id, "异步停止方法返回False")
        except Exception as e:
            self._handle_module_error(module_id, f"异步停止失败: {str(e)}")
    
    def stop_all_modules(self) -> bool:
        """停止所有模块"""
        with self._lifecycle_lock:
            if self._system_state != LifecycleState.RUNNING:
                self.logger.warning(f"系统状态不正确，无法停止: {self._system_state.value}")
                return False
            
            self._system_state = LifecycleState.STOPPING
            
            # 停止健康检查
            self._stop_health_monitoring()
            
            # 执行预停止回调
            self._execute_pre_stop_callbacks_system()
            
            # 获取所有运行中的模块（按依赖关系的逆序）
            module_ids = [
                module_id for module_id, info in self._module_lifecycles.items()
                if info.current_state in [LifecycleState.RUNNING, LifecycleState.ERROR]
            ]
            
            # 逆序停止（依赖关系的反向）
            module_ids.reverse()
            
            success_count = 0
            
            for module_id in module_ids:
                if self.stop_module(module_id):
                    success_count += 1
                else:
                    self.logger.error(f"模块停止失败: {module_id}")
                    # 继续停止其他模块
            
            if success_count == len(module_ids):
                self._system_state = LifecycleState.STOPPED
                self._execute_post_stop_callbacks_system()
                self.logger.info("所有模块停止完成")
                return True
            else:
                self._system_state = LifecycleState.ERROR
                self.logger.error("模块停止失败")
                return False
    
    def destroy_module(self, module_id: str) -> bool:
        """销毁单个模块"""
        with self._lifecycle_lock:
            if module_id not in self._module_lifecycles:
                self.logger.error(f"模块未注册: {module_id}")
                return False
            
            lifecycle_info = self._module_lifecycles[module_id]
            
            # 检查当前状态
            if lifecycle_info.current_state not in [LifecycleState.STOPPED, LifecycleState.ERROR, LifecycleState.INITIALIZED]:
                self.logger.warning(f"模块状态不正确，无法销毁: {module_id} (当前状态: {lifecycle_info.current_state.value})")
                return False
            
            try:
                # 更新状态
                self._update_module_state(module_id, LifecycleState.DESTROYING, "destruction_started")
                
                # 获取模块实例
                instance = self.factory.get_instance(lifecycle_info.module_name)
                if not instance:
                    raise ValueError(f"无法获取模块实例: {lifecycle_info.module_name}")
                
                # 执行预销毁回调
                self._execute_pre_destruction_callbacks(module_id, instance)
                
                # 调用模块的销毁方法
                if hasattr(instance, 'destroy'):
                    if asyncio.iscoroutinefunction(instance.destroy):
                        # 异步销毁
                        asyncio.create_task(self._async_destroy_module(module_id, instance))
                    else:
                        # 同步销毁
                        instance.destroy()
                        self._update_module_state(module_id, LifecycleState.DESTROYED, "destruction_completed")
                        lifecycle_info.destroy_time = time.time()
                        self._lifecycle_stats['total_destructions'] += 1
                        
                        # 执行后销毁回调
                        self._execute_post_destruction_callbacks(module_id, instance)
                        
                        self.logger.info(f"模块销毁成功: {module_id}")
                        return True
                else:
                    # 模块没有销毁方法，直接标记为已销毁
                    self._update_module_state(module_id, LifecycleState.DESTROYED, "no_destroy_method")
                    lifecycle_info.destroy_time = time.time()
                    self._lifecycle_stats['total_destructions'] += 1
                    self.logger.debug(f"模块无销毁方法，标记为已销毁: {module_id}")
                    return True
                
            except Exception as e:
                self._handle_module_error(module_id, f"销毁失败: {str(e)}")
                return False
        
        return False
    
    async def _async_destroy_module(self, module_id: str, instance: Any):
        """异步销毁模块"""
        try:
            await instance.destroy()
            with self._lifecycle_lock:
                lifecycle_info = self._module_lifecycles[module_id]
                self._update_module_state(module_id, LifecycleState.DESTROYED, "async_destruction_completed")
                lifecycle_info.destroy_time = time.time()
                self._lifecycle_stats['total_destructions'] += 1
                
                # 执行后销毁回调
                self._execute_post_destruction_callbacks(module_id, instance)
                
                self.logger.info(f"模块异步销毁成功: {module_id}")
        except Exception as e:
            self._handle_module_error(module_id, f"异步销毁失败: {str(e)}")
    
    def destroy_all_modules(self) -> bool:
        """销毁所有模块"""
        with self._lifecycle_lock:
            if self._system_state not in [LifecycleState.STOPPED, LifecycleState.ERROR, LifecycleState.INITIALIZED]:
                self.logger.error(f"系统状态不正确，无法销毁: {self._system_state.value}")
                return False
            
            self._system_state = LifecycleState.DESTROYING
            
            # 执行预销毁回调
            self._execute_pre_destruction_callbacks_system()
            
            # 获取所有需要销毁的模块（按依赖关系的逆序）
            module_ids = list(self._module_lifecycles.keys())
            module_ids.reverse()
            
            success_count = 0
            
            for module_id in module_ids:
                if self.destroy_module(module_id):
                    success_count += 1
                else:
                    self.logger.error(f"模块销毁失败: {module_id}")
                    # 继续销毁其他模块
            
            if success_count == len(module_ids):
                self._system_state = LifecycleState.DESTROYED
                self._execute_post_destruction_callbacks_system()
                self.logger.info("所有模块销毁完成")
                return True
            else:
                self._system_state = LifecycleState.ERROR
                self.logger.error("模块销毁失败")
                return False
    
    def shutdown(self):
        """关闭系统（停止并销毁所有模块）"""
        self.logger.info("开始关闭系统...")
        
        # 停止所有模块
        if not self.stop_all_modules():
            self.logger.warning("停止模块时发生错误，继续销毁...")
        
        # 销毁所有模块
        if not self.destroy_all_modules():
            self.logger.error("销毁模块时发生错误")
        
        self.logger.info("系统关闭完成")
    
    def _check_dependencies_initialized(self, module_id: str) -> bool:
        """检查模块的依赖是否已初始化"""
        lifecycle_info = self._module_lifecycles[module_id]
        
        for dep_id in lifecycle_info.dependencies:
            if dep_id not in self._module_lifecycles:
                self.logger.warning(f"依赖模块未注册: {dep_id}")
                continue
            
            dep_state = self._module_lifecycles[dep_id].current_state
            if dep_state not in [LifecycleState.INITIALIZED, LifecycleState.RUNNING, LifecycleState.STOPPED]:
                self.logger.warning(f"依赖模块未初始化: {dep_id} (状态: {dep_state.value})")
                return False
        
        return True
    
    def _update_module_state(self, module_id: str, new_state: LifecycleState, reason: str):
        """更新模块状态"""
        with self._lifecycle_lock:
            if module_id not in self._module_lifecycles:
                return
            
            lifecycle_info = self._module_lifecycles[module_id]
            old_state = lifecycle_info.current_state
            
            # 更新状态
            lifecycle_info.previous_state = old_state
            lifecycle_info.current_state = new_state
            
            # 更新状态历史
            self._update_state_history(module_id, new_state, reason)
            
            # 触发状态监听器
            self._trigger_state_listeners(module_id, old_state, new_state, reason)
            
            self.logger.debug(f"模块状态变更: {module_id} {old_state.value} -> {new_state.value} ({reason})")
    
    def _update_state_history(self, module_id: str, state: LifecycleState, reason: str):
        """更新状态历史"""
        lifecycle_info = self._module_lifecycles[module_id]
        
        state_record = {
            'state': state,
            'timestamp': time.time(),
            'reason': reason
        }
        
        lifecycle_info.state_history.append(state_record)
        
        # 保持历史记录在合理范围内
        if len(lifecycle_info.state_history) > 100:
            lifecycle_info.state_history = lifecycle_info.state_history[-50:]
    
    def _handle_module_error(self, module_id: str, error_msg: str):
        """处理模块错误"""
        with self._lifecycle_lock:
            if module_id not in self._module_lifecycles:
                return
            
            lifecycle_info = self._module_lifecycles[module_id]
            
            # 更新错误信息
            lifecycle_info.error_count += 1
            lifecycle_info.last_error = error_msg
            lifecycle_info.health_status = HealthStatus.UNHEALTHY
            
            # 更新状态
            self._update_module_state(module_id, LifecycleState.ERROR, error_msg)
            
            # 更新统计
            self._lifecycle_stats['total_errors'] += 1
            
            self.logger.error(f"模块错误: {module_id} - {error_msg}")
    
    def register_health_checker(self, module_id: str, health_checker: Callable[[Any], HealthStatus]):
        """注册健康检查器"""
        with self._health_check_lock:
            self._health_checkers[module_id] = health_checker
            self.logger.debug(f"注册健康检查器: {module_id}")
    
    def _start_health_monitoring(self):
        """启动健康监控"""
        if self._health_check_thread is not None and self._health_check_thread.is_alive():
            return
        
        self._health_check_thread = threading.Thread(
            target=self._health_monitoring_worker,
            daemon=True,
            name="HealthMonitor"
        )
        self._health_check_thread.start()
        self.logger.info("健康监控已启动")
    
    def _stop_health_monitoring(self):
        """停止健康监控"""
        self._shutdown_event.set()
        
        if self._health_check_thread and self._health_check_thread.is_alive():
            self._health_check_thread.join(timeout=5.0)
        
        self._shutdown_event.clear()
        self.logger.info("健康监控已停止")
    
    def _health_monitoring_worker(self):
        """健康监控工作线程"""
        while not self._shutdown_event.is_set():
            try:
                # 执行健康检查
                self._perform_health_checks()
                
                # 等待下一次检查
                self._shutdown_event.wait(self.health_check_interval)
            except Exception as e:
                self.logger.error(f"健康监控工作线程异常: {e}")
                time.sleep(self.health_check_interval)  # 发生异常时等待一段时间再继续
    
    def _perform_health_checks(self):
        """执行健康检查"""
        with self._health_check_lock:
            for module_id, health_checker in self._health_checkers.items():
                try:
                    # 获取模块实例
                    lifecycle_info = self._module_lifecycles.get(module_id)
                    if not lifecycle_info:
                        continue
                    
                    instance = self.factory.get_instance(lifecycle_info.module_name)
                    if not instance:
                        continue
                    
                    # 执行健康检查
                    health_status = health_checker(instance)
                    
                    # 更新健康状态
                    lifecycle_info.health_status = health_status
                    lifecycle_info.health_check_timestamp = time.time()
                    
                    # 记录不健康状态
                    if health_status != HealthStatus.HEALTHY:
                        self.logger.warning(f"模块健康状态异常: {module_id} - {health_status.value}")
                
                except Exception as e:
                    self.logger.error(f"健康检查执行失败 {module_id}: {e}")
                    lifecycle_info.health_status = HealthStatus.UNKNOWN
    
    def get_health_report(self) -> SystemHealthReport:
        """获取系统健康报告"""
        with self._lifecycle_lock:
            module_health = {}
            healthy_count = 0
            total_count = len(self._module_lifecycles)
            checks_passed = 0
            checks_failed = 0
            
            for module_id, lifecycle_info in self._module_lifecycles.items():
                module_health[module_id] = lifecycle_info.health_status
                
                if lifecycle_info.health_status == HealthStatus.HEALTHY:
                    healthy_count += 1
                    checks_passed += 1
                elif lifecycle_info.health_status == HealthStatus.UNHEALTHY:
                    checks_failed += 1
            
            # 计算整体状态
            if healthy_count == total_count:
                overall_status = HealthStatus.HEALTHY
            elif checks_failed > 0:
                overall_status = HealthStatus.UNHEALTHY
            else:
                overall_status = HealthStatus.WARNING
            
            report = SystemHealthReport(
                timestamp=time.time(),
                overall_status=overall_status,
                module_health=module_health,
                healthy_modules=healthy_count,
                total_modules=total_count,
                health_checks_passed=checks_passed,
                health_checks_failed=checks_failed,
                details={
                    'system_state': self._system_state.value,
                    'error_count': self._lifecycle_stats['total_errors']
                }
            )
            
            return report
    
    def add_state_listener(self, state: LifecycleState, listener: Callable):
        """添加状态监听器"""
        with self._lifecycle_lock:
            self._state_listeners[state].append(listener)
    
    def _trigger_state_listeners(self, module_id: str, old_state: LifecycleState, new_state: LifecycleState, reason: str):
        """触发状态监听器"""
        listeners = self._state_listeners.get(new_state, [])
        
        for listener in listeners:
            try:
                listener(module_id, old_state, new_state, reason)
            except Exception as e:
                self.logger.error(f"状态监听器执行失败: {e}")
    
    # 回调方法
    def add_pre_initialization_callback(self, callback: Callable):
        """添加预初始化回调"""
        self._pre_initialization_callbacks.append(callback)
    
    def add_post_initialization_callback(self, callback: Callable):
        """添加后初始化回调"""
        self._post_initialization_callbacks.append(callback)
    
    def add_pre_start_callback(self, callback: Callable):
        """添加预启动回调"""
        self._pre_start_callbacks.append(callback)
    
    def add_post_start_callback(self, callback: Callable):
        """添加后启动回调"""
        self._post_start_callbacks.append(callback)
    
    def add_pre_stop_callback(self, callback: Callable):
        """添加预停止回调"""
        self._pre_stop_callbacks.append(callback)
    
    def add_post_stop_callback(self, callback: Callable):
        """添加后停止回调"""
        self._post_stop_callbacks.append(callback)
    
    def add_pre_destruction_callback(self, callback: Callable):
        """添加预销毁回调"""
        self._pre_destruction_callbacks.append(callback)
    
    def add_post_destruction_callback(self, callback: Callable):
        """添加后销毁回调"""
        self._post_destruction_callbacks.append(callback)
    
    def _execute_pre_initialization_callbacks(self, module_id: str, instance: Any):
        """执行预初始化回调"""
        for callback in self._pre_initialization_callbacks:
            try:
                callback(module_id, instance)
            except Exception as e:
                self.logger.error(f"预初始化回调执行失败: {e}")
    
    def _execute_post_initialization_callbacks(self, module_id: str, instance: Any):
        """执行后初始化回调"""
        for callback in self._post_initialization_callbacks:
            try:
                callback(module_id, instance)
            except Exception as e:
                self.logger.error(f"后初始化回调执行失败: {e}")
    
    def _execute_pre_start_callbacks(self, module_id: str, instance: Any):
        """执行预启动回调"""
        for callback in self._pre_start_callbacks:
            try:
                callback(module_id, instance)
            except Exception as e:
                self.logger.error(f"预启动回调执行失败: {e}")
    
    def _execute_post_start_callbacks(self, module_id: str, instance: Any):
        """执行后启动回调"""
        for callback in self._post_start_callbacks:
            try:
                callback(module_id, instance)
            except Exception as e:
                self.logger.error(f"后启动回调执行失败: {e}")
    
    def _execute_pre_stop_callbacks(self, module_id: str, instance: Any):
        """执行预停止回调"""
        for callback in self._pre_stop_callbacks:
            try:
                callback(module_id, instance)
            except Exception as e:
                self.logger.error(f"预停止回调执行失败: {e}")
    
    def _execute_post_stop_callbacks(self, module_id: str, instance: Any):
        """执行后停止回调"""
        for callback in self._post_stop_callbacks:
            try:
                callback(module_id, instance)
            except Exception as e:
                self.logger.error(f"后停止回调执行失败: {e}")
    
    def _execute_pre_destruction_callbacks(self, module_id: str, instance: Any):
        """执行预销毁回调"""
        for callback in self._pre_destruction_callbacks:
            try:
                callback(module_id, instance)
            except Exception as e:
                self.logger.error(f"预销毁回调执行失败: {e}")
    
    def _execute_post_destruction_callbacks(self, module_id: str, instance: Any):
        """执行后销毁回调"""
        for callback in self._post_destruction_callbacks:
            try:
                callback(module_id, instance)
            except Exception as e:
                self.logger.error(f"后销毁回调执行失败: {e}")
    
    def _execute_pre_initialization_callbacks_system(self):
        """执行系统级预初始化回调"""
        for callback in self._pre_initialization_callbacks:
            try:
                callback(None, None)  # 系统级回调，模块ID和实例为None
            except Exception as e:
                self.logger.error(f"系统级预初始化回调执行失败: {e}")
    
    def _execute_post_initialization_callbacks_system(self):
        """执行系统级后初始化回调"""
        for callback in self._post_initialization_callbacks:
            try:
                callback(None, None)
            except Exception as e:
                self.logger.error(f"系统级后初始化回调执行失败: {e}")
    
    def _execute_pre_start_callbacks_system(self):
        """执行系统级预启动回调"""
        for callback in self._pre_start_callbacks:
            try:
                callback(None, None)
            except Exception as e:
                self.logger.error(f"系统级预启动回调执行失败: {e}")
    
    def _execute_post_start_callbacks_system(self):
        """执行系统级后启动回调"""
        for callback in self._post_start_callbacks:
            try:
                callback(None, None)
            except Exception as e:
                self.logger.error(f"系统级后启动回调执行失败: {e}")
    
    def _execute_pre_stop_callbacks_system(self):
        """执行系统级预停止回调"""
        for callback in self._pre_stop_callbacks:
            try:
                callback(None, None)
            except Exception as e:
                self.logger.error(f"系统级预停止回调执行失败: {e}")
    
    def _execute_post_stop_callbacks_system(self):
        """执行系统级后停止回调"""
        for callback in self._post_stop_callbacks:
            try:
                callback(None, None)
            except Exception as e:
                self.logger.error(f"系统级后停止回调执行失败: {e}")
    
    def _execute_pre_destruction_callbacks_system(self):
        """执行系统级预销毁回调"""
        for callback in self._pre_destruction_callbacks:
            try:
                callback(None, None)
            except Exception as e:
                self.logger.error(f"系统级预销毁回调执行失败: {e}")
    
    def _execute_post_destruction_callbacks_system(self):
        """执行系统级后销毁回调"""
        for callback in self._post_destruction_callbacks:
            try:
                callback(None, None)
            except Exception as e:
                self.logger.error(f"系统级后销毁回调执行失败: {e}")
    
    def get_module_lifecycle_info(self, module_id: str) -> Optional[ModuleLifecycleInfo]:
        """获取模块生命周期信息"""
        with self._lifecycle_lock:
            return self._module_lifecycles.get(module_id)
    
    def get_all_lifecycle_info(self) -> Dict[str, ModuleLifecycleInfo]:
        """获取所有模块的生命周期信息"""
        with self._lifecycle_lock:
            return dict(self._module_lifecycles)
    
    def get_system_state(self) -> LifecycleState:
        """获取系统状态"""
        return self._system_state
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        uptime = time.time() - self._lifecycle_stats['start_time']
        
        return {
            'system_state': self._system_state.value,
            'total_modules': len(self._module_lifecycles),
            'initialized_modules': len([m for m in self._module_lifecycles.values() if m.current_state == LifecycleState.INITIALIZED]),
            'running_modules': len([m for m in self._module_lifecycles.values() if m.current_state == LifecycleState.RUNNING]),
            'stopped_modules': len([m for m in self._module_lifecycles.values() if m.current_state == LifecycleState.STOPPED]),
            'destroyed_modules': len([m for m in self._module_lifecycles.values() if m.current_state == LifecycleState.DESTROYED]),
            'error_modules': len([m for m in self._module_lifecycles.values() if m.current_state == LifecycleState.ERROR]),
            'operations': self._lifecycle_stats.copy(),
            'performance': {
                'uptime': uptime,
                'initializations_per_second': self._lifecycle_stats['total_initializations'] / uptime if uptime > 0 else 0,
                'starts_per_second': self._lifecycle_stats['total_starts'] / uptime if uptime > 0 else 0,
                'stops_per_second': self._lifecycle_stats['total_stops'] / uptime if uptime > 0 else 0,
                'destructions_per_second': self._lifecycle_stats['total_destructions'] / uptime if uptime > 0 else 0
            },
            'health_monitoring': {
                'enabled': self.enable_health_monitoring,
                'interval': self.health_check_interval,
                'active': self._health_check_thread is not None and self._health_check_thread.is_alive(),
                'health_checkers_registered': len(self._health_checkers)
            }
        }


class LifecycleBuilder:
    """生命周期管理器构建器"""
    
    def __init__(self):
        self._config = {
            'auto_initialize': True,
            'auto_start': False,
            'shutdown_timeout': 30.0,
            'health_check_interval': 60.0,
            'enable_health_monitoring': True,
            'parallel_initialization': True,
            'parallel_shutdown': True
        }
        self._callbacks = {
            'pre_initialization': [],
            'post_initialization': [],
            'pre_start': [],
            'post_start': [],
            'pre_stop': [],
            'post_stop': [],
            'pre_destruction': [],
            'post_destruction': []
        }
        self._state_listeners = defaultdict(list)
    
    def set_auto_initialize(self, enabled: bool) -> 'LifecycleBuilder':
        """设置自动初始化"""
        self._config['auto_initialize'] = enabled
        return self
    
    def set_auto_start(self, enabled: bool) -> 'LifecycleBuilder':
        """设置自动启动"""
        self._config['auto_start'] = enabled
        return self
    
    def set_health_monitoring(self, enabled: bool, interval: float = None) -> 'LifecycleBuilder':
        """设置健康监控"""
        self._config['enable_health_monitoring'] = enabled
        if interval is not None:
            self._config['health_check_interval'] = interval
        return self
    
    def set_parallel_operations(self, initialization: bool = None, shutdown: bool = None) -> 'LifecycleBuilder':
        """设置并行操作"""
        if initialization is not None:
            self._config['parallel_initialization'] = initialization
        if shutdown is not None:
            self._config['parallel_shutdown'] = shutdown
        return self
    
    def add_pre_initialization_callback(self, callback: Callable) -> 'LifecycleBuilder':
        """添加预初始化回调"""
        self._callbacks['pre_initialization'].append(callback)
        return self
    
    def add_post_initialization_callback(self, callback: Callable) -> 'LifecycleBuilder':
        """添加后初始化回调"""
        self._callbacks['post_initialization'].append(callback)
        return self
    
    def add_pre_start_callback(self, callback: Callable) -> 'LifecycleBuilder':
        """添加预启动回调"""
        self._callbacks['pre_start'].append(callback)
        return self
    
    def add_post_start_callback(self, callback: Callable) -> 'LifecycleBuilder':
        """添加后启动回调"""
        self._callbacks['post_start'].append(callback)
        return self
    
    def add_pre_stop_callback(self, callback: Callable) -> 'LifecycleBuilder':
        """添加预停止回调"""
        self._callbacks['pre_stop'].append(callback)
        return self
    
    def add_post_stop_callback(self, callback: Callable) -> 'LifecycleBuilder':
        """添加后停止回调"""
        self._callbacks['post_stop'].append(callback)
        return self
    
    def add_pre_destruction_callback(self, callback: Callable) -> 'LifecycleBuilder':
        """添加预销毁回调"""
        self._callbacks['pre_destruction'].append(callback)
        return self
    
    def add_post_destruction_callback(self, callback: Callable) -> 'LifecycleBuilder':
        """添加后销毁回调"""
        self._callbacks['post_destruction'].append(callback)
        return self
    
    def add_state_listener(self, state: LifecycleState, listener: Callable) -> 'LifecycleBuilder':
        """添加状态监听器"""
        self._state_listeners[state].append(listener)
        return self
    
    def build(self, registry, factory, config_manager=None) -> LifecycleManager:
        """构建生命周期管理器实例"""
        manager = LifecycleManager(registry, factory, config_manager)
        
        # 应用配置
        for key, value in self._config.items():
            if hasattr(manager, key):
                setattr(manager, key, value)
        
        # 注册回调
        for callback in self._callbacks['pre_initialization']:
            manager.add_pre_initialization_callback(callback)
        
        for callback in self._callbacks['post_initialization']:
            manager.add_post_initialization_callback(callback)
        
        for callback in self._callbacks['pre_start']:
            manager.add_pre_start_callback(callback)
        
        for callback in self._callbacks['post_start']:
            manager.add_post_start_callback(callback)
        
        for callback in self._callbacks['pre_stop']:
            manager.add_pre_stop_callback(callback)
        
        for callback in self._callbacks['post_stop']:
            manager.add_post_stop_callback(callback)
        
        for callback in self._callbacks['pre_destruction']:
            manager.add_pre_destruction_callback(callback)
        
        for callback in self._callbacks['post_destruction']:
            manager.add_post_destruction_callback(callback)
        
        # 注册状态监听器
        for state, listeners in self._state_listeners.items():
            for listener in listeners:
                manager.add_state_listener(state, listener)
        
        return manager