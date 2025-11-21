"""
I7 回调接口管理器

提供完整的回调接口管理功能，包括异步回调管理、回调链处理、超时处理、
重试机制、结果缓存、监控日志、性能统计、安全验证和版本管理。


版本: 1.0.0
创建时间: 2025-11-05
"""

import asyncio
import logging
import time
import hashlib
import json
import weakref
from typing import (
    Any, Callable, Dict, List, Optional, Union, 
    Awaitable, Tuple, Set, TypeVar, Generic
)
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
import functools
import traceback


# 类型定义
T = TypeVar('T')
CallbackResult = TypeVar('CallbackResult')
CallbackFunc = Callable[..., Awaitable[Any]]
CallbackContext = Dict[str, Any]


class CallbackStatus(Enum):
    """回调状态枚举"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class SecurityLevel(Enum):
    """安全级别枚举"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CallbackMetrics:
    """回调性能指标"""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    timeout_calls: int = 0
    retry_calls: int = 0
    average_response_time: float = 0.0
    min_response_time: float = float('inf')
    max_response_time: float = 0.0
    total_response_time: float = 0.0
    last_call_time: Optional[datetime] = None
    error_rate: float = 0.0


@dataclass
class CallbackRequest:
    """回调请求"""
    id: str
    callback_func: CallbackFunc
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    timeout: float = 30.0
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    status: CallbackStatus = CallbackStatus.PENDING
    result: Optional[Any] = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    version: str = "1.0.0"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CallbackChain:
    """回调链"""
    id: str
    name: str
    callbacks: List[CallbackRequest] = field(default_factory=list)
    parallel: bool = False
    timeout: float = 60.0
    continue_on_error: bool = True
    created_at: datetime = field(default_factory=datetime.now)


class CallbackCache:
    """回调结果缓存"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        """
        初始化缓存
        
        Args:
            max_size: 最大缓存条目数
            ttl: 缓存生存时间（秒）
        """
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._lock = threading.RLock()
    
    def _generate_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        func_id = f"{func.__module__}.{func.__name__}"
        args_str = json.dumps(args, sort_keys=True, default=str)
        kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
        key_data = f"{func_id}:{args_str}:{kwargs_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, func: Callable, args: tuple, kwargs: dict) -> Optional[Any]:
        """获取缓存结果"""
        key = self._generate_key(func, args, kwargs)
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                    self._access_times[key] = datetime.now()
                    return value
                else:
                    del self._cache[key]
                    del self._access_times[key]
        return None
    
    def set(self, func: Callable, args: tuple, kwargs: dict, value: Any) -> None:
        """设置缓存结果"""
        key = self._generate_key(func, args, kwargs)
        with self._lock:
            # 清理过期缓存
            self._cleanup_expired()
            
            # 如果超过最大大小，删除最久未访问的条目
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            self._cache[key] = (value, datetime.now())
            self._access_times[key] = datetime.now()
    
    def _cleanup_expired(self) -> None:
        """清理过期缓存"""
        now = datetime.now()
        expired_keys = [
            key for key, (_, timestamp) in self._cache.items()
            if now - timestamp >= timedelta(seconds=self.ttl)
        ]
        for key in expired_keys:
            del self._cache[key]
            del self._access_times[key]
    
    def _evict_oldest(self) -> None:
        """删除最久未访问的缓存条目"""
        if not self._access_times:
            return
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[oldest_key]
        del self._access_times[oldest_key]
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()


class SecurityValidator:
    """回调安全验证器"""
    
    def __init__(self):
        self._trusted_modules: Set[str] = set()
        self._blocked_patterns: List[str] = []
        self._allowed_functions: Dict[str, Set[str]] = defaultdict(set)
    
    def add_trusted_module(self, module_name: str) -> None:
        """添加信任模块"""
        self._trusted_modules.add(module_name)
    
    def add_blocked_pattern(self, pattern: str) -> None:
        """添加阻止模式"""
        self._blocked_patterns.append(pattern)
    
    def add_allowed_function(self, module: str, func_name: str) -> None:
        """添加允许的函数"""
        self._allowed_functions[module].add(func_name)
    
    def validate_callback(self, func: Callable, args: tuple, kwargs: dict) -> Tuple[bool, str]:
        """验证回调安全性"""
        try:
            # 检查模块是否信任
            module_name = func.__module__
            if self._trusted_modules and module_name not in self._trusted_modules:
                return False, f"模块 {module_name} 不在信任列表中"
            
            # 检查阻止模式
            func_name = func.__qualname__
            for pattern in self._blocked_patterns:
                if pattern in func_name:
                    return False, f"函数名包含阻止模式: {pattern}"
            
            # 检查允许的函数列表
            if self._allowed_functions:
                if module_name not in self._allowed_functions:
                    return False, f"模块 {module_name} 不在允许列表中"
                if func.__name__ not in self._allowed_functions[module_name]:
                    return False, f"函数 {func.__name__} 不在允许列表中"
            
            # 检查参数安全性
            if not self._validate_arguments(args, kwargs):
                return False, "参数包含不安全内容"
            
            return True, "安全验证通过"
        
        except Exception as e:
            return False, f"安全验证异常: {str(e)}"
    
    def _validate_arguments(self, args: tuple, kwargs: dict) -> bool:
        """验证参数安全性"""
        dangerous_patterns = ['eval', 'exec', 'import', 'open', '__']
        
        def check_value(value: Any) -> bool:
            if isinstance(value, str):
                value_lower = value.lower()
                for pattern in dangerous_patterns:
                    if pattern in value_lower:
                        return False
            elif isinstance(value, (list, tuple)):
                return all(check_value(item) for item in value)
            elif isinstance(value, dict):
                return all(check_value(k) and check_value(v) for k, v in value.items())
            return True
        
        return check_value(args) and check_value(kwargs)


class CallbackInterfaceManager:
    """
    回调接口管理器
    
    提供完整的回调接口管理功能，包括：
    - 异步回调管理
    - 回调链处理
    - 超时回调处理
    - 回调重试机制
    - 回调结果缓存
    - 回调监控和日志
    - 回调性能统计
    - 回调安全验证
    - 回调接口版本管理
    """
    
    def __init__(self, 
                 max_workers: int = 10,
                 enable_cache: bool = True,
                 cache_size: int = 1000,
                 cache_ttl: int = 300,
                 enable_security: bool = True,
                 enable_metrics: bool = True,
                 log_level: int = logging.INFO):
        """
        初始化回调接口管理器
        
        Args:
            max_workers: 最大工作线程数
            enable_cache: 是否启用缓存
            cache_size: 缓存大小
            cache_ttl: 缓存生存时间
            enable_security: 是否启用安全验证
            enable_metrics: 是否启用性能统计
            log_level: 日志级别
        """
        self.max_workers = max_workers
        self.enable_cache = enable_cache
        self.enable_security = enable_security
        self.enable_metrics = enable_metrics
        
        # 初始化组件
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache = CallbackCache(cache_size, cache_ttl) if enable_cache else None
        self.security_validator = SecurityValidator() if enable_security else None
        
        # 回调管理
        self._callbacks: Dict[str, CallbackRequest] = {}
        self._callback_chains: Dict[str, CallbackChain] = {}
        self._metrics: Dict[str, CallbackMetrics] = defaultdict(CallbackMetrics)
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 日志配置
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # 性能统计
        self._performance_data: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        # 版本管理
        self._versions: Dict[str, str] = {}
        
        self.logger.info("回调接口管理器初始化完成")
    
    async def register_callback(self, 
                               name: str,
                               callback_func: CallbackFunc,
                               version: str = "1.0.0",
                               timeout: float = 30.0,
                               max_retries: int = 3,
                               retry_delay: float = 1.0,
                               security_level: SecurityLevel = SecurityLevel.MEDIUM) -> bool:
        """
        注册回调函数
        
        Args:
            name: 回调名称
            callback_func: 回调函数
            version: 版本号
            timeout: 超时时间
            max_retries: 最大重试次数
            security_level: 安全级别
            
        Returns:
            bool: 注册是否成功
        """
        try:
            with self._lock:
                # 安全验证
                if self.enable_security:
                    is_safe, message = self.security_validator.validate_callback(
                        callback_func, (), {}
                    )
                    if not is_safe:
                        self.logger.error(f"回调 {name} 安全验证失败: {message}")
                        return False
                
                # 创建回调请求
                callback_request = CallbackRequest(
                    id=str(uuid.uuid4()),
                    callback_func=callback_func,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_delay=retry_delay,
                    security_level=security_level,
                    version=version,
                    metadata={"name": name}
                )
                
                self._callbacks[name] = callback_request
                self._versions[name] = version
                
                self.logger.info(f"回调 {name} (版本: {version}) 注册成功")
                return True
                
        except Exception as e:
            self.logger.error(f"注册回调 {name} 失败: {str(e)}")
            return False
    
    async def execute_callback(self, 
                              name: str,
                              *args,
                              timeout: Optional[float] = None,
                              use_cache: bool = True,
                              **kwargs) -> Any:
        """
        执行回调函数
        
        Args:
            name: 回调名称
            *args: 位置参数
            timeout: 超时时间（覆盖默认值）
            use_cache: 是否使用缓存
            **kwargs: 关键字参数
            
        Returns:
            Any: 回调结果
            
        Raises:
            ValueError: 回调不存在
            TimeoutError: 执行超时
            Exception: 执行失败
        """
        if name not in self._callbacks:
            raise ValueError(f"回调 {name} 不存在")
        
        callback_request = self._callbacks[name]
        callback_func = callback_request.callback_func
        
        # 缓存检查
        if self.enable_cache and use_cache and self.cache:
            cached_result = self.cache.get(callback_func, args, kwargs)
            if cached_result is not None:
                self.logger.debug(f"回调 {name} 缓存命中")
                return cached_result
        
        # 设置超时
        exec_timeout = timeout or callback_request.timeout
        
        # 创建异步任务
        task = asyncio.create_task(
            self._execute_with_retry(callback_request, args, kwargs)
        )
        
        try:
            # 等待执行结果
            result = await asyncio.wait_for(task, timeout=exec_timeout)
            
            # 缓存结果
            if self.enable_cache and use_cache and self.cache:
                self.cache.set(callback_func, args, kwargs, result)
            
            return result
            
        except asyncio.TimeoutError:
            callback_request.status = CallbackStatus.TIMEOUT
            self.logger.warning(f"回调 {name} 执行超时")
            raise TimeoutError(f"回调 {name} 执行超时 ({exec_timeout}秒)")
    
    async def _execute_with_retry(self, 
                                 callback_request: CallbackRequest,
                                 args: tuple,
                                 kwargs: dict) -> Any:
        """带重试机制的执行"""
        callback_func = callback_request.callback_func
        max_retries = callback_request.max_retries
        retry_delay = callback_request.retry_delay
        
        for attempt in range(max_retries + 1):
            try:
                # 更新状态
                callback_request.status = CallbackStatus.RUNNING
                callback_request.start_time = time.time()
                
                # 记录性能数据
                if self.enable_metrics:
                    self._record_performance_start(callback_request.id)
                
                # 执行回调
                if asyncio.iscoroutinefunction(callback_func):
                    result = await callback_func(*args, **kwargs)
                else:
                    # 在线程池中执行同步函数
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(
                        self.executor, 
                        lambda: callback_func(*args, **kwargs)
                    )
                
                # 更新状态和指标
                callback_request.status = CallbackStatus.SUCCESS
                callback_request.result = result
                callback_request.end_time = time.time()
                
                if self.enable_metrics:
                    self._record_performance_end(callback_request.id, True)
                    self._update_metrics(callback_request.id, True)
                
                self.logger.debug(f"回调执行成功: {callback_request.metadata.get('name', 'unknown')}")
                return result
                
            except Exception as e:
                callback_request.retry_count = attempt
                callback_request.error = e
                
                if attempt < max_retries:
                    callback_request.status = CallbackStatus.RETRYING
                    self.logger.warning(
                        f"回调执行失败，{retry_delay}秒后重试 (第{attempt + 1}次): {str(e)}"
                    )
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    callback_request.status = CallbackStatus.FAILED
                    callback_request.end_time = time.time()
                    
                    if self.enable_metrics:
                        self._record_performance_end(callback_request.id, False)
                        self._update_metrics(callback_request.id, False)
                    
                    self.logger.error(f"回调执行最终失败: {str(e)}")
                    raise e
    
    async def execute_callback_chain(self, 
                                    chain_name: str,
                                    inputs: Dict[str, Any] = None,
                                    parallel: bool = False) -> Dict[str, Any]:
        """
        执行回调链
        
        Args:
            chain_name: 链名称
            inputs: 输入参数
            parallel: 是否并行执行
            
        Returns:
            Dict[str, Any]: 执行结果
        """
        if chain_name not in self._callback_chains:
            raise ValueError(f"回调链 {chain_name} 不存在")
        
        chain = self._callback_chains[chain_name]
        inputs = inputs or {}
        
        self.logger.info(f"开始执行回调链: {chain_name} (并行: {parallel})")
        
        if parallel:
            # 并行执行
            tasks = []
            for callback in chain.callbacks:
                task = asyncio.create_task(
                    self._execute_chain_callback(callback, inputs)
                )
                tasks.append((callback.metadata.get('name', 'unknown'), task))
            
            results = {}
            for name, task in tasks:
                try:
                    results[name] = await task
                except Exception as e:
                    if chain.continue_on_error:
                        self.logger.error(f"链中回调 {name} 执行失败，继续执行: {str(e)}")
                        results[name] = None
                    else:
                        raise e
            
            return results
        else:
            # 串行执行
            results = {}
            for callback in chain.callbacks:
                try:
                    name = callback.metadata.get('name', 'unknown')
                    result = await self._execute_chain_callback(callback, inputs)
                    results[name] = result
                    
                    # 将结果传递给下一个回调
                    if hasattr(callback, 'output_key'):
                        inputs[callback.output_key] = result
                        
                except Exception as e:
                    if chain.continue_on_error:
                        self.logger.error(f"链中回调执行失败，继续执行: {str(e)}")
                        results[name] = None
                    else:
                        raise e
            
            return results
    
    async def _execute_chain_callback(self, 
                                     callback: CallbackRequest,
                                     inputs: Dict[str, Any]) -> Any:
        """执行链中的单个回调"""
        # 准备参数
        args = ()
        kwargs = inputs.copy()
        
        # 如果回调有定义的参数映射，使用映射
        if hasattr(callback, 'arg_mapping'):
            args = tuple(inputs.get(key) for key in callback.arg_mapping.get('args', []))
            for key, value in callback.arg_mapping.get('kwargs', {}).items():
                kwargs[key] = inputs.get(value)
        
        return await self.execute_callback(
            callback.metadata.get('name', 'unknown'),
            *args,
            **kwargs
        )
    
    async def create_callback_chain(self,
                                   name: str,
                                   callback_names: List[str],
                                   parallel: bool = False,
                                   timeout: float = 60.0,
                                   continue_on_error: bool = True) -> bool:
        """
        创建回调链
        
        Args:
            name: 链名称
            callback_names: 回调名称列表
            parallel: 是否并行执行
            timeout: 超时时间
            continue_on_error: 错误时是否继续
            
        Returns:
            bool: 创建是否成功
        """
        try:
            with self._lock:
                # 验证回调是否存在
                callbacks = []
                for callback_name in callback_names:
                    if callback_name not in self._callbacks:
                        raise ValueError(f"回调 {callback_name} 不存在")
                    callbacks.append(self._callbacks[callback_name])
                
                # 创建回调链
                chain = CallbackChain(
                    id=str(uuid.uuid4()),
                    name=name,
                    callbacks=callbacks,
                    parallel=parallel,
                    timeout=timeout,
                    continue_on_error=continue_on_error
                )
                
                self._callback_chains[name] = chain
                
                self.logger.info(f"回调链 {name} 创建成功")
                return True
                
        except Exception as e:
            self.logger.error(f"创建回调链 {name} 失败: {str(e)}")
            return False
    
    def get_callback_metrics(self, name: str) -> CallbackMetrics:
        """
        获取回调性能指标
        
        Args:
            name: 回调名称
            
        Returns:
            CallbackMetrics: 性能指标
        """
        return self._metrics.get(name, CallbackMetrics())
    
    def get_all_metrics(self) -> Dict[str, CallbackMetrics]:
        """获取所有回调的性能指标"""
        return dict(self._metrics)
    
    def _record_performance_start(self, callback_id: str) -> None:
        """记录性能开始"""
        self._performance_data[f"{callback_id}_start"].append(time.time())
    
    def _record_performance_end(self, callback_id: str, success: bool) -> None:
        """记录性能结束"""
        end_time = time.time()
        start_time = self._performance_data[f"{callback_id}_start"].pop() if self._performance_data[f"{callback_id}_start"] else end_time
        duration = end_time - start_time
        
        self._performance_data[f"{callback_id}_duration"].append(duration)
        self._performance_data[f"{callback_id}_success"].append(success)
    
    def _update_metrics(self, callback_id: str, success: bool) -> None:
        """更新性能指标"""
        # 找到对应的回调请求
        callback_request = None
        for req in self._callbacks.values():
            if req.id == callback_id:
                callback_request = req
                break
        
        if not callback_request:
            return
        
        name = callback_request.metadata.get('name', 'unknown')
        metrics = self._metrics[name]
        
        # 更新基本统计
        metrics.total_calls += 1
        if success:
            metrics.successful_calls += 1
        else:
            metrics.failed_calls += 1
        
        # 更新响应时间
        if callback_request.start_time and callback_request.end_time:
            duration = callback_request.end_time - callback_request.start_time
            metrics.total_response_time += duration
            metrics.average_response_time = metrics.total_response_time / metrics.total_calls
            metrics.min_response_time = min(metrics.min_response_time, duration)
            metrics.max_response_time = max(metrics.max_response_time, duration)
        
        metrics.last_call_time = datetime.now()
        metrics.error_rate = metrics.failed_calls / metrics.total_calls if metrics.total_calls > 0 else 0.0
        
        # 更新重试统计
        if callback_request.retry_count > 0:
            metrics.retry_calls += callback_request.retry_count
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_callbacks": len(self._callbacks),
            "total_chains": len(self._callback_chains),
            "cache_enabled": self.enable_cache,
            "security_enabled": self.enable_security,
            "metrics_enabled": self.enable_metrics,
            "callback_metrics": {},
            "overall_stats": {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "average_response_time": 0.0,
                "overall_error_rate": 0.0
            }
        }
        
        total_calls = 0
        total_successful = 0
        total_failed = 0
        total_response_time = 0.0
        
        for name, metrics in self._metrics.items():
            report["callback_metrics"][name] = {
                "total_calls": metrics.total_calls,
                "successful_calls": metrics.successful_calls,
                "failed_calls": metrics.failed_calls,
                "timeout_calls": metrics.timeout_calls,
                "retry_calls": metrics.retry_calls,
                "average_response_time": round(metrics.average_response_time, 4),
                "min_response_time": round(metrics.min_response_time, 4) if metrics.min_response_time != float('inf') else 0,
                "max_response_time": round(metrics.max_response_time, 4),
                "error_rate": round(metrics.error_rate, 4),
                "last_call_time": metrics.last_call_time.isoformat() if metrics.last_call_time else None
            }
            
            total_calls += metrics.total_calls
            total_successful += metrics.successful_calls
            total_failed += metrics.failed_calls
            total_response_time += metrics.total_response_time
        
        # 计算总体统计
        if total_calls > 0:
            report["overall_stats"]["total_calls"] = total_calls
            report["overall_stats"]["successful_calls"] = total_successful
            report["overall_stats"]["failed_calls"] = total_failed
            report["overall_stats"]["average_response_time"] = round(total_response_time / total_calls, 4)
            report["overall_stats"]["overall_error_rate"] = round(total_failed / total_calls, 4)
        
        return report
    
    def clear_cache(self) -> None:
        """清空缓存"""
        if self.cache:
            self.cache.clear()
            self.logger.info("缓存已清空")
    
    def clear_metrics(self) -> None:
        """清空性能指标"""
        self._metrics.clear()
        self._performance_data.clear()
        self.logger.info("性能指标已清空")
    
    def get_callback_status(self, name: str) -> Optional[CallbackStatus]:
        """获取回调状态"""
        if name in self._callbacks:
            return self._callbacks[name].status
        return None
    
    def cancel_callback(self, name: str) -> bool:
        """取消回调"""
        if name in self._callbacks:
            self._callbacks[name].status = CallbackStatus.CANCELLED
            self.logger.info(f"回调 {name} 已取消")
            return True
        return False
    
    def update_callback_version(self, name: str, version: str) -> bool:
        """更新回调版本"""
        if name in self._callbacks:
            self._callbacks[name].version = version
            self._versions[name] = version
            self.logger.info(f"回调 {name} 版本更新为 {version}")
            return True
        return False
    
    def get_version_info(self) -> Dict[str, str]:
        """获取版本信息"""
        return dict(self._versions)
    
    def add_security_rule(self, 
                         module: str = None,
                         function: str = None,
                         pattern: str = None,
                         action: str = "allow") -> bool:
        """
        添加安全规则
        
        Args:
            module: 模块名
            function: 函数名
            pattern: 模式
            action: 操作（allow/deny）
            
        Returns:
            bool: 添加是否成功
        """
        try:
            if self.security_validator:
                if module and function:
                    if action == "allow":
                        self.security_validator.add_allowed_function(module, function)
                    else:
                        self.security_validator.add_blocked_pattern(f"{module}.{function}")
                elif pattern:
                    if action == "allow":
                        # 这里可以实现更复杂的模式匹配
                        pass
                    else:
                        self.security_validator.add_blocked_pattern(pattern)
                
                self.logger.info(f"安全规则添加成功: {action} {module or pattern}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"添加安全规则失败: {str(e)}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "executor": "running" if not self.executor._shutdown else "stopped",
                "cache": "enabled" if self.cache else "disabled",
                "security": "enabled" if self.security_validator else "disabled",
                "metrics": "enabled" if self.enable_metrics else "disabled"
            },
            "statistics": {
                "registered_callbacks": len(self._callbacks),
                "registered_chains": len(self._callback_chains),
                "total_calls": sum(m.total_calls for m in self._metrics.values()),
                "cache_size": len(self.cache._cache) if self.cache else 0
            }
        }
        
        # 检查是否有失败的回调
        failed_callbacks = [
            name for name, req in self._callbacks.items() 
            if req.status == CallbackStatus.FAILED
        ]
        
        if failed_callbacks:
            health["status"] = "degraded"
            health["issues"] = f"存在 {len(failed_callbacks)} 个失败的回调"
        
        return health
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()
    
    def shutdown(self) -> None:
        """关闭管理器"""
        try:
            self.executor.shutdown(wait=True)
            if self.cache:
                self.cache.clear()
            self.logger.info("回调接口管理器已关闭")
        except Exception as e:
            self.logger.error(f"关闭管理器时发生错误: {str(e)}")
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        await asyncio.get_event_loop().run_in_executor(self.executor, self.shutdown)
    
    def add_security_pattern(self, pattern: str) -> bool:
        """添加安全模式（测试用）"""
        try:
            if self.security_validator:
                self.security_validator.add_blocked_pattern(pattern)
                self.logger.info(f"安全模式添加成功: {pattern}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"添加安全模式失败: {str(e)}")
            return False
        except Exception as e:
            self.logger.error(f"关闭管理器时发生错误: {str(e)}")


# 测试用例
async def test_callback_interface_manager():
    """测试回调接口管理器"""
    
    # 创建管理器
    manager = CallbackInterfaceManager(
        max_workers=5,
        enable_cache=True,
        enable_security=True,
        enable_metrics=True
    )
    
    # 定义测试回调函数
    async def simple_callback(data: str) -> str:
        """简单回调函数"""
        await asyncio.sleep(0.1)  # 模拟异步操作
        return f"处理完成: {data}"
    
    async def error_callback() -> None:
        """会出错的回调函数"""
        await asyncio.sleep(0.05)
        raise ValueError("测试错误")
    
    def sync_callback(value: int) -> int:
        """同步回调函数"""
        time.sleep(0.1)  # 模拟同步操作
        return value * 2
    
    try:
        # 注册回调
        print("=== 测试回调注册 ===")
        assert await manager.register_callback("simple", simple_callback, timeout=5.0)
        assert await manager.register_callback("error", error_callback, max_retries=2)
        assert await manager.register_callback("sync", sync_callback)
        print("回调注册成功")
        
        # 执行简单回调
        print("\n=== 测试简单回调 ===")
        result = await manager.execute_callback("simple", "测试数据")
        print(f"简单回调结果: {result}")
        assert result == "处理完成: 测试数据"
        
        # 测试缓存
        print("\n=== 测试缓存 ===")
        result1 = await manager.execute_callback("simple", "缓存测试")
        result2 = await manager.execute_callback("simple", "缓存测试")
        print(f"缓存测试结果: {result1 == result2}")
        
        # 测试错误回调和重试
        print("\n=== 测试错误回调和重试 ===")
        try:
            await manager.execute_callback("error")
        except Exception as e:
            print(f"错误回调捕获异常: {type(e).__name__}: {e}")
        
        # 测试同步回调
        print("\n=== 测试同步回调 ===")
        result = await manager.execute_callback("sync", 5)
        print(f"同步回调结果: {result}")
        assert result == 10
        
        # 创建回调链
        print("\n=== 测试回调链 ===")
        await manager.create_callback_chain(
            "test_chain",
            ["simple", "sync"],
            parallel=False
        )
        
        chain_result = await manager.execute_callback_chain("test_chain", {"data": "链测试"})
        print(f"回调链结果: {chain_result}")
        
        # 获取性能指标
        print("\n=== 测试性能指标 ===")
        metrics = manager.get_callback_metrics("simple")
        print(f"简单回调指标: {metrics}")
        
        # 获取性能报告
        print("\n=== 测试性能报告 ===")
        report = manager.get_performance_report()
        print(f"性能报告: {json.dumps(report, indent=2, ensure_ascii=False)}")
        
        # 健康检查
        print("\n=== 测试健康检查 ===")
        health = manager.health_check()
        print(f"健康状态: {health}")
        
        print("\n=== 所有测试通过 ===")
        
    finally:
        # 关闭管理器
        manager.shutdown()


if __name__ == "__main__":
    # 运行测试
    asyncio.run(test_callback_interface_manager())