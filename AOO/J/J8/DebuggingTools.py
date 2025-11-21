"""
J8调试工具模块

这个模块提供了全面的调试工具集，包括代码调试、性能调试、日志调试、
错误诊断、实时监控、调试报告生成等功能。支持异步和分布式调试。

主要组件：
- CodeDebugger: 代码调试工具（断点、变量检查、调用栈）
- PerformanceDebugger: 性能调试工具（性能分析、内存检测、CPU分析）
- LogDebugger: 日志调试工具（级别控制、过滤、聚合）
- ErrorDiagnostician: 错误诊断工具（异常捕获、分类、定位）
- RealtimeDebugger: 实时调试监控
- DebugReporter: 调试报告生成
- DistributedDebugger: 分布式调试
- J8DebuggingTools: 主要调试工具管理器

作者: J8 Team
版本: 1.0.0
创建时间: 2025-11-06
"""

import asyncio
import threading
import time
import traceback
import sys
import os
import json
import logging
import inspect
import functools
import weakref
import gc
import psutil
import cProfile
import pstats
import io
import pickle
import hashlib
import datetime
from typing import Any, Dict, List, Optional, Callable, Union, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager, asynccontextmanager
from enum import Enum
import warnings


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DebugLevel(Enum):
    """调试级别枚举"""
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"
    DEBUG = "DEBUG"
    TRACE = "TRACE"


class ErrorType(Enum):
    """错误类型枚举"""
    SYNTAX_ERROR = "SYNTAX_ERROR"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    LOGIC_ERROR = "LOGIC_ERROR"
    PERFORMANCE_ERROR = "PERFORMANCE_ERROR"
    MEMORY_ERROR = "MEMORY_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    DATABASE_ERROR = "DATABASE_ERROR"
    UNKNOWN_ERROR = "UNKNOWN_ERROR"


@dataclass
class DebugInfo:
    """调试信息数据类"""
    timestamp: float
    level: DebugLevel
    message: str
    file_path: str
    line_number: int
    function_name: str
    variables: Dict[str, Any] = field(default_factory=dict)
    stack_trace: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    cpu_usage: float
    memory_usage: float
    execution_time: float
    function_calls: int
    timestamp: float
    function_name: str = ""
    file_path: str = ""


@dataclass
class ErrorInfo:
    """错误信息数据类"""
    error_type: ErrorType
    error_message: str
    stack_trace: str
    timestamp: float
    context: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)


class Breakpoint:
    """断点类"""
    
    def __init__(self, file_path: str, line_number: int, condition: Optional[Callable] = None):
        self.file_path = file_path
        self.line_number = line_number
        self.condition = condition
        self.enabled = True
        self.hit_count = 0
        self.creation_time = time.time()
        self.variables_to_watch: List[str] = []
        
    def should_break(self, context: Dict[str, Any]) -> bool:
        """检查是否应该触发断点"""
        if not self.enabled:
            return False
            
        if self.condition:
            try:
                return self.condition(context)
            except Exception as e:
                logger.error(f"断点条件检查失败: {e}")
                return False
                
        return True
        
    def add_watched_variable(self, var_name: str):
        """添加要监视的变量"""
        if var_name not in self.variables_to_watch:
            self.variables_to_watch.append(var_name)


class CodeDebugger:
    """代码调试工具类"""
    
    def __init__(self):
        self.breakpoints: Dict[str, Breakpoint] = {}
        self.call_stack: List[Dict[str, Any]] = []
        self.variable_snapshots: Dict[str, Dict[str, Any]] = {}
        self.debug_mode = False
        self._lock = threading.Lock()
        
    def add_breakpoint(self, file_path: str, line_number: int, 
                      condition: Optional[Callable] = None) -> str:
        """添加断点"""
        breakpoint_id = f"{file_path}:{line_number}"
        
        with self._lock:
            self.breakpoints[breakpoint_id] = Breakpoint(file_path, line_number, condition)
            
        logger.info(f"添加断点: {breakpoint_id}")
        return breakpoint_id
        
    def remove_breakpoint(self, breakpoint_id: str) -> bool:
        """移除断点"""
        with self._lock:
            if breakpoint_id in self.breakpoints:
                del self.breakpoints[breakpoint_id]
                logger.info(f"移除断点: {breakpoint_id}")
                return True
        return False
        
    def enable_breakpoint(self, breakpoint_id: str) -> bool:
        """启用断点"""
        with self._lock:
            if breakpoint_id in self.breakpoints:
                self.breakpoints[breakpoint_id].enabled = True
                return True
        return False
        
    def disable_breakpoint(self, breakpoint_id: str) -> bool:
        """禁用断点"""
        with self._lock:
            if breakpoint_id in self.breakpoints:
                self.breakpoints[breakpoint_id].enabled = False
                return True
        return False
        
    def check_breakpoint(self, file_path: str, line_number: int, 
                        context: Dict[str, Any]) -> Optional[Breakpoint]:
        """检查断点"""
        breakpoint_id = f"{file_path}:{line_number}"
        
        with self._lock:
            breakpoint = self.breakpoints.get(breakpoint_id)
            if breakpoint and breakpoint.should_break(context):
                breakpoint.hit_count += 1
                logger.debug(f"断点触发: {breakpoint_id}")
                return breakpoint
        return None
        
    def get_call_stack(self) -> List[Dict[str, Any]]:
        """获取调用栈"""
        return self.call_stack.copy()
        
    def push_call_stack(self, frame_info: Dict[str, Any]):
        """推入调用栈"""
        with self._lock:
            self.call_stack.append({
                **frame_info,
                'timestamp': time.time()
            })
            
    def pop_call_stack(self):
        """弹出调用栈"""
        with self._lock:
            if self.call_stack:
                self.call_stack.pop()
                
    def snapshot_variables(self, scope: str, variables: Dict[str, Any]):
        """变量快照"""
        with self._lock:
            self.variable_snapshots[scope] = {
                'timestamp': time.time(),
                'variables': variables.copy()
            }
            
    def get_variable_snapshot(self, scope: str) -> Optional[Dict[str, Any]]:
        """获取变量快照"""
        with self._lock:
            return self.variable_snapshots.get(scope)
            
    def inspect_variable(self, var_name: str, frame_locals: Dict[str, Any], 
                        frame_globals: Dict[str, Any]) -> Any:
        """检查变量"""
        # 先检查局部变量
        if var_name in frame_locals:
            return frame_locals[var_name]
            
        # 再检查全局变量
        if var_name in frame_globals:
            return frame_globals[var_name]
            
        # 检查内置变量
        if var_name in __builtins__:
            return __builtins__[var_name]
            
        return None
        
    @contextmanager
    def debug_context(self, function_name: str, file_path: str):
        """调试上下文管理器"""
        frame = inspect.currentframe()
        try:
            frame_info = {
                'function_name': function_name,
                'file_path': file_path,
                'line_number': frame.f_lineno,
                'locals': frame.f_locals.copy(),
                'globals': {k: v for k, v in frame.f_globals.items() 
                           if not k.startswith('__')}
            }
            
            self.push_call_stack(frame_info)
            
            yield frame_info
            
        finally:
            self.pop_call_stack()
            del frame


class PerformanceDebugger:
    """性能调试工具类"""
    
    def __init__(self):
        self.profiles: Dict[str, cProfile.Profile] = {}
        self.performance_data: Dict[str, List[PerformanceMetrics]] = defaultdict(list)
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.cpu_samples: deque = deque(maxlen=1000)
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
    def start_profiling(self, profile_name: str = "default") -> bool:
        """开始性能分析"""
        try:
            with self._lock:
                if profile_name not in self.profiles:
                    self.profiles[profile_name] = cProfile.Profile()
                    self.profiles[profile_name].enable()
                    logger.info(f"开始性能分析: {profile_name}")
                    return True
                else:
                    logger.warning(f"性能分析已存在: {profile_name}")
                    return False
        except Exception as e:
            logger.error(f"开始性能分析失败: {e}")
            return False
            
    def stop_profiling(self, profile_name: str = "default") -> Optional[str]:
        """停止性能分析并获取结果"""
        try:
            with self._lock:
                if profile_name in self.profiles:
                    self.profiles[profile_name].disable()
                    s = io.StringIO()
                    ps = pstats.Stats(self.profiles[profile_name], stream=s)
                    ps.sort_stats('cumulative')
                    ps.print_stats(20)  # 显示前20个函数
                    
                    result = s.getvalue()
                    logger.info(f"停止性能分析: {profile_name}")
                    return result
                else:
                    logger.warning(f"性能分析不存在: {profile_name}")
                    return None
        except Exception as e:
            logger.error(f"停止性能分析失败: {e}")
            return None
            
    def record_performance(self, function_name: str, execution_time: float, 
                          file_path: str = ""):
        """记录性能指标"""
        try:
            cpu_usage = psutil.cpu_percent(interval=0.1)
            memory_info = psutil.virtual_memory()
            
            metrics = PerformanceMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory_info.percent,
                execution_time=execution_time,
                function_calls=1,
                timestamp=time.time(),
                function_name=function_name,
                file_path=file_path
            )
            
            with self._lock:
                self.performance_data[function_name].append(metrics)
                
        except Exception as e:
            logger.error(f"记录性能指标失败: {e}")
            
    def get_performance_summary(self, function_name: str = None) -> Dict[str, Any]:
        """获取性能摘要"""
        try:
            with self._lock:
                if function_name:
                    data = self.performance_data.get(function_name, [])
                else:
                    data = []
                    for metrics_list in self.performance_data.values():
                        data.extend(metrics_list)
                        
                if not data:
                    return {}
                    
                total_calls = len(data)
                avg_execution_time = sum(m.execution_time for m in data) / total_calls
                avg_cpu_usage = sum(m.cpu_usage for m in data) / total_calls
                avg_memory_usage = sum(m.memory_usage for m in data) / total_calls
                max_execution_time = max(m.execution_time for m in data)
                
                return {
                    'total_calls': total_calls,
                    'avg_execution_time': avg_execution_time,
                    'max_execution_time': max_execution_time,
                    'avg_cpu_usage': avg_cpu_usage,
                    'avg_memory_usage': avg_memory_usage,
                    'function_name': function_name or 'all'
                }
                
        except Exception as e:
            logger.error(f"获取性能摘要失败: {e}")
            return {}
            
    def start_monitoring(self, interval: float = 1.0):
        """开始实时监控"""
        if self._monitoring:
            logger.warning("性能监控已在运行")
            return
            
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("开始性能监控")
        
    def stop_monitoring(self):
        """停止实时监控"""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("停止性能监控")
        
    def _monitor_loop(self, interval: float):
        """监控循环"""
        while self._monitoring:
            try:
                # CPU使用率
                cpu_percent = psutil.cpu_percent(interval=None)
                self.cpu_samples.append({
                    'timestamp': time.time(),
                    'cpu_percent': cpu_percent
                })
                
                # 内存使用情况
                memory_info = psutil.virtual_memory()
                self.memory_snapshots.append({
                    'timestamp': time.time(),
                    'total': memory_info.total,
                    'available': memory_info.available,
                    'percent': memory_info.percent,
                    'used': memory_info.used,
                    'free': memory_info.free
                })
                
                # 保持最近1000个快照
                if len(self.memory_snapshots) > 1000:
                    self.memory_snapshots.pop(0)
                    
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"性能监控循环错误: {e}")
                time.sleep(interval)
                
    def detect_memory_leaks(self) -> Dict[str, Any]:
        """检测内存泄漏"""
        try:
            if len(self.memory_snapshots) < 10:
                return {'error': '内存快照数据不足'}
                
            # 分析内存使用趋势
            recent_snapshots = self.memory_snapshots[-10:]
            memory_trend = []
            
            for i in range(1, len(recent_snapshots)):
                prev = recent_snapshots[i-1]
                curr = recent_snapshots[i]
                growth = curr['used'] - prev['used']
                memory_trend.append(growth)
                
            avg_growth = sum(memory_trend) / len(memory_trend)
            max_growth = max(memory_trend)
            
            # 检测异常增长
            leak_detected = avg_growth > 1024 * 1024 * 10  # 10MB增长阈值
            
            return {
                'leak_detected': leak_detected,
                'avg_memory_growth': avg_growth,
                'max_memory_growth': max_growth,
                'trend_analysis': memory_trend,
                'recommendations': self._generate_memory_recommendations(leak_detected, avg_growth)
            }
            
        except Exception as e:
            logger.error(f"内存泄漏检测失败: {e}")
            return {'error': str(e)}
            
    def _generate_memory_recommendations(self, leak_detected: bool, avg_growth: float) -> List[str]:
        """生成内存优化建议"""
        recommendations = []
        
        if leak_detected:
            recommendations.append("检测到可能的内存泄漏，建议检查对象引用和清理逻辑")
            recommendations.append("考虑使用弱引用(weakref)来避免循环引用")
            recommendations.append("检查是否有未关闭的文件句柄或网络连接")
            
        if avg_growth > 1024 * 1024 * 100:  # 100MB
            recommendations.append("内存使用量增长较快，建议优化数据结构")
            recommendations.append("考虑使用生成器或迭代器来减少内存占用")
            
        if not recommendations:
            recommendations.append("内存使用正常")
            
        return recommendations
        
    @contextmanager
    def performance_timer(self, function_name: str = "", file_path: str = ""):
        """性能计时上下文管理器"""
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            self.record_performance(function_name, execution_time, file_path)


class LogDebugger:
    """日志调试工具类"""
    
    def __init__(self):
        self.log_handlers: Dict[str, logging.Handler] = {}
        self.log_filters: List[Callable] = []
        self.log_aggregator: Dict[str, List[DebugInfo]] = defaultdict(list)
        self.log_level = DebugLevel.DEBUG
        self.max_log_entries = 10000
        self._lock = threading.Lock()
        
    def set_log_level(self, level: DebugLevel):
        """设置日志级别"""
        self.log_level = level
        logger.setLevel(getattr(logging, level.value))
        
    def add_log_filter(self, filter_func: Callable[[DebugInfo], bool]):
        """添加日志过滤器"""
        with self._lock:
            self.log_filters.append(filter_func)
            
    def remove_log_filter(self, filter_func: Callable):
        """移除日志过滤器"""
        with self._lock:
            if filter_func in self.log_filters:
                self.log_filters.remove(filter_func)
                
    def add_log_handler(self, name: str, handler: logging.Handler):
        """添加日志处理器"""
        with self._lock:
            self.log_handlers[name] = handler
            logger.addHandler(handler)
            
    def remove_log_handler(self, name: str):
        """移除日志处理器"""
        with self._lock:
            if name in self.log_handlers:
                handler = self.log_handlers[name]
                logger.removeHandler(handler)
                del self.log_handlers[name]
                
    def log_debug_info(self, level: DebugLevel, message: str, 
                      file_path: str = "", line_number: int = 0,
                      function_name: str = "", variables: Dict[str, Any] = None,
                      context: Dict[str, Any] = None):
        """记录调试信息"""
        try:
            debug_info = DebugInfo(
                timestamp=time.time(),
                level=level,
                message=message,
                file_path=file_path,
                line_number=line_number,
                function_name=function_name,
                variables=variables or {},
                context=context or {}
            )
            
            # 应用过滤器
            if not self._should_log(debug_info):
                return
                
            # 添加到聚合器
            with self._lock:
                log_key = f"{debug_info.file_path}:{debug_info.function_name}"
                self.log_aggregator[log_key].append(debug_info)
                
                # 限制日志条目数量
                if len(self.log_aggregator[log_key]) > self.max_log_entries:
                    self.log_aggregator[log_key].pop(0)
                    
            # 记录到标准日志
            log_message = f"[{level.value}] {file_path}:{line_number} - {message}"
            if variables:
                log_message += f" | Variables: {variables}"
                
            if level == DebugLevel.CRITICAL:
                logger.critical(log_message)
            elif level == DebugLevel.ERROR:
                logger.error(log_message)
            elif level == DebugLevel.WARNING:
                logger.warning(log_message)
            elif level == DebugLevel.INFO:
                logger.info(log_message)
            elif level == DebugLevel.DEBUG:
                logger.debug(log_message)
            else:
                logger.debug(log_message)
                
        except Exception as e:
            logger.error(f"记录调试信息失败: {e}")
            
    def _should_log(self, debug_info: DebugInfo) -> bool:
        """检查是否应该记录日志"""
        # 检查日志级别
        level_order = [DebugLevel.TRACE, DebugLevel.DEBUG, DebugLevel.INFO, 
                      DebugLevel.WARNING, DebugLevel.ERROR, DebugLevel.CRITICAL]
        
        if level_order.index(debug_info.level) < level_order.index(self.log_level):
            return False
            
        # 应用过滤器
        for filter_func in self.log_filters:
            try:
                if not filter_func(debug_info):
                    return False
            except Exception as e:
                logger.error(f"日志过滤器执行失败: {e}")
                
        return True
        
    def get_logs(self, filter_func: Callable[[DebugInfo], bool] = None, 
                limit: int = 100) -> List[DebugInfo]:
        """获取日志"""
        with self._lock:
            all_logs = []
            for logs in self.log_aggregator.values():
                all_logs.extend(logs)
                
            # 按时间戳排序
            all_logs.sort(key=lambda x: x.timestamp, reverse=True)
            
            # 应用过滤器
            if filter_func:
                all_logs = [log for log in all_logs if filter_func(log)]
                
            return all_logs[:limit]
            
    def aggregate_logs_by_function(self, function_name: str) -> List[DebugInfo]:
        """按函数聚合日志"""
        with self._lock:
            return self.log_aggregator.get(function_name, [])
            
    def aggregate_logs_by_file(self, file_path: str) -> List[DebugInfo]:
        """按文件聚合日志"""
        with self._lock:
            result = []
            for key, logs in self.log_aggregator.items():
                if key.startswith(file_path + ":"):
                    result.extend(logs)
            return result
            
    def export_logs(self, file_path: str, format: str = "json") -> bool:
        """导出日志"""
        try:
            logs = self.get_logs()
            
            if format.lower() == "json":
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([{
                        'timestamp': log.timestamp,
                        'level': log.level.value,
                        'message': log.message,
                        'file_path': log.file_path,
                        'line_number': log.line_number,
                        'function_name': log.function_name,
                        'variables': log.variables,
                        'context': log.context
                    } for log in logs], f, indent=2, ensure_ascii=False)
            else:
                # 文本格式
                with open(file_path, 'w', encoding='utf-8') as f:
                    for log in logs:
                        timestamp = datetime.datetime.fromtimestamp(log.timestamp).isoformat()
                        f.write(f"[{timestamp}] {log.level.value} {log.file_path}:{log.line_number} "
                               f"{log.function_name} - {log.message}\n")
                              
            logger.info(f"日志已导出到: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出日志失败: {e}")
            return False
            
    def clear_logs(self):
        """清除日志"""
        with self._lock:
            self.log_aggregator.clear()
            logger.info("日志已清除")


class ErrorDiagnostician:
    """错误诊断工具类"""
    
    def __init__(self):
        self.error_history: List[ErrorInfo] = []
        self.error_patterns: Dict[str, int] = defaultdict(int)
        self.error_suggestions: Dict[ErrorType, List[str]] = {
            ErrorType.SYNTAX_ERROR: [
                "检查语法错误，特别注意缩进和括号匹配",
                "使用代码检查工具如flake8或pylint",
                "查看Python语法文档确认正确语法"
            ],
            ErrorType.RUNTIME_ERROR: [
                "检查变量是否已定义",
                "确认函数调用参数正确",
                "检查类型转换是否有效"
            ],
            ErrorType.LOGIC_ERROR: [
                "检查算法逻辑是否正确",
                "验证业务规则实现",
                "添加调试输出来跟踪执行流程"
            ],
            ErrorType.PERFORMANCE_ERROR: [
                "检查是否有不必要的循环",
                "优化数据结构选择",
                "考虑使用缓存机制"
            ],
            ErrorType.MEMORY_ERROR: [
                "检查是否有内存泄漏",
                "优化对象创建和销毁",
                "使用生成器减少内存占用"
            ]
        }
        self._lock = threading.Lock()
        
    def diagnose_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """诊断错误"""
        try:
            error_type = self._classify_error(exception)
            error_message = str(exception)
            stack_trace = traceback.format_exc()
            timestamp = time.time()
            
            # 生成建议
            suggestions = self._generate_suggestions(error_type, exception, context)
            
            error_info = ErrorInfo(
                error_type=error_type,
                error_message=error_message,
                stack_trace=stack_trace,
                timestamp=timestamp,
                context=context or {},
                suggestions=suggestions
            )
            
            # 记录错误历史
            with self._lock:
                self.error_history.append(error_info)
                
                # 限制历史记录数量
                if len(self.error_history) > 1000:
                    self.error_history.pop(0)
                    
                # 更新错误模式统计
                error_pattern = self._extract_error_pattern(exception)
                self.error_patterns[error_pattern] += 1
                
            logger.error(f"错误诊断: {error_type.value} - {error_message}")
            return error_info
            
        except Exception as e:
            logger.error(f"错误诊断失败: {e}")
            return ErrorInfo(
                error_type=ErrorType.UNKNOWN_ERROR,
                error_message=str(e),
                stack_trace=traceback.format_exc(),
                timestamp=time.time()
            )
            
    def _classify_error(self, exception: Exception) -> ErrorType:
        """分类错误类型"""
        exception_type = type(exception).__name__
        
        if isinstance(exception, SyntaxError):
            return ErrorType.SYNTAX_ERROR
        elif isinstance(exception, (NameError, TypeError, ValueError, AttributeError)):
            return ErrorType.RUNTIME_ERROR
        elif isinstance(exception, (MemoryError, RecursionError)):
            return ErrorType.MEMORY_ERROR
        elif isinstance(exception, (TimeoutError, ConnectionError)):
            return ErrorType.NETWORK_ERROR
        else:
            return ErrorType.LOGIC_ERROR
            
    def _extract_error_pattern(self, exception: Exception) -> str:
        """提取错误模式"""
        exception_type = type(exception).__name__
        error_message = str(exception)
        
        # 提取关键信息作为模式
        if "NameError" in exception_type:
            return "NameError"
        elif "TypeError" in exception_type:
            return "TypeError"
        elif "ValueError" in exception_type:
            return "ValueError"
        elif "AttributeError" in exception_type:
            return "AttributeError"
        else:
            # 使用错误消息的哈希作为模式
            return hashlib.md5(error_message.encode()).hexdigest()[:8]
            
    def _generate_suggestions(self, error_type: ErrorType, exception: Exception, 
                            context: Dict[str, Any]) -> List[str]:
        """生成错误建议"""
        suggestions = self.error_suggestions.get(error_type, []).copy()
        
        # 根据具体异常添加特定建议
        exception_type = type(exception).__name__
        
        if "NameError" in exception_type:
            suggestions.append("检查变量名是否拼写正确")
            suggestions.append("确认变量是否在作用域内定义")
        elif "TypeError" in exception_type:
            suggestions.append("检查变量类型是否正确")
            suggestions.append("确认函数参数类型匹配")
        elif "ValueError" in exception_type:
            suggestions.append("检查输入值是否在有效范围内")
            suggestions.append("确认数据格式是否符合要求")
            
        # 根据上下文添加建议
        if context:
            if 'function_name' in context:
                suggestions.append(f"检查函数 {context['function_name']} 的实现")
            if 'file_path' in context:
                suggestions.append(f"检查文件 {context['file_path']} 中的相关代码")
                
        return suggestions
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计"""
        with self._lock:
            if not self.error_history:
                return {}
                
            # 按错误类型统计
            error_type_counts = defaultdict(int)
            for error_info in self.error_history:
                error_type_counts[error_info.error_type.value] += 1
                
            # 最近错误
            recent_errors = sorted(self.error_history, 
                                 key=lambda x: x.timestamp, reverse=True)[:10]
            
            # 最常见的错误模式
            top_patterns = sorted(self.error_patterns.items(), 
                                key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'total_errors': len(self.error_history),
                'error_type_distribution': dict(error_type_counts),
                'recent_errors': [{
                    'type': error.error_type.value,
                    'message': error.error_message,
                    'timestamp': error.timestamp
                } for error in recent_errors],
                'top_error_patterns': [{'pattern': pattern, 'count': count} 
                                     for pattern, count in top_patterns]
            }
            
    def get_similar_errors(self, error_message: str, limit: int = 5) -> List[ErrorInfo]:
        """查找相似错误"""
        with self._lock:
            similar_errors = []
            error_words = set(error_message.lower().split())
            
            for error_info in self.error_history:
                if error_words & set(error_info.error_message.lower().split()):
                    similar_errors.append(error_info)
                    
            return similar_errors[:limit]
            
    def clear_error_history(self):
        """清除错误历史"""
        with self._lock:
            self.error_history.clear()
            self.error_patterns.clear()
            logger.info("错误历史已清除")


class RealtimeDebugger:
    """实时调试监控类"""
    
    def __init__(self):
        self.monitoring_active = False
        self.watchers: Dict[str, Callable] = {}
        self.event_queue: deque = deque(maxlen=10000)
        self.subscribers: Set[Callable] = set()
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
    def start_monitoring(self):
        """开始实时监控"""
        if self.monitoring_active:
            logger.warning("实时监控已在运行")
            return
            
        self.monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("开始实时调试监控")
        
    def stop_monitoring(self):
        """停止实时监控"""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("停止实时调试监控")
        
    def add_watcher(self, name: str, watcher_func: Callable[[], Any]):
        """添加监控器"""
        with self._lock:
            self.watchers[name] = watcher_func
            logger.info(f"添加监控器: {name}")
            
    def remove_watcher(self, name: str):
        """移除监控器"""
        with self._lock:
            if name in self.watchers:
                del self.watchers[name]
                logger.info(f"移除监控器: {name}")
                
    def subscribe(self, callback: Callable):
        """订阅调试事件"""
        with self._lock:
            self.subscribers.add(callback)
            logger.info(f"添加调试事件订阅者: {callback.__name__}")
            
    def unsubscribe(self, callback: Callable):
        """取消订阅调试事件"""
        with self._lock:
            self.subscribers.discard(callback)
            logger.info(f"移除调试事件订阅者: {callback.__name__}")
            
    def emit_event(self, event_type: str, data: Dict[str, Any]):
        """发出调试事件"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': time.time(),
            'thread_id': threading.get_ident()
        }
        
        with self._lock:
            self.event_queue.append(event)
            
        # 通知订阅者
        self._notify_subscribers(event)
        
    def _notify_subscribers(self, event: Dict[str, Any]):
        """通知订阅者"""
        for callback in self.subscribers.copy():
            try:
                callback(event)
            except Exception as e:
                logger.error(f"调试事件通知失败: {e}")
                
    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 执行所有监控器
                for name, watcher_func in self.watchers.items():
                    try:
                        result = watcher_func()
                        if result:
                            self.emit_event('watcher_result', {
                                'watcher_name': name,
                                'result': result
                            })
                    except Exception as e:
                        self.emit_event('watcher_error', {
                            'watcher_name': name,
                            'error': str(e)
                        })
                        
                time.sleep(1)  # 每秒检查一次
                
            except Exception as e:
                logger.error(f"实时监控循环错误: {e}")
                time.sleep(1)
                
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取最近事件"""
        with self._lock:
            events = list(self.event_queue)
            return events[-limit:] if len(events) > limit else events
            
    def get_events_by_type(self, event_type: str, limit: int = 100) -> List[Dict[str, Any]]:
        """按类型获取事件"""
        with self._lock:
            events = [event for event in self.event_queue if event['type'] == event_type]
            return events[-limit:] if len(events) > limit else events


class DebugReporter:
    """调试报告生成器"""
    
    def __init__(self):
        self.report_templates = {
            'summary': self._generate_summary_report,
            'performance': self._generate_performance_report,
            'errors': self._generate_error_report,
            'logs': self._generate_log_report,
            'full': self._generate_full_report
        }
        
    def generate_report(self, report_type: str, debug_data: Dict[str, Any], 
                       output_path: str = None) -> str:
        """生成调试报告"""
        try:
            if report_type not in self.report_templates:
                raise ValueError(f"不支持的报告类型: {report_type}")
                
            report_generator = self.report_templates[report_type]
            report_content = report_generator(debug_data)
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                logger.info(f"调试报告已生成: {output_path}")
                
            return report_content
            
        except Exception as e:
            logger.error(f"生成调试报告失败: {e}")
            return f"报告生成失败: {e}"
            
    def _generate_summary_report(self, debug_data: Dict[str, Any]) -> str:
        """生成摘要报告"""
        lines = [
            "# 调试摘要报告",
            f"生成时间: {datetime.datetime.now().isoformat()}",
            ""
        ]
        
        # 基本统计
        if 'performance' in debug_data:
            perf_data = debug_data['performance']
            lines.extend([
                "## 性能摘要",
                f"- 总函数调用: {perf_data.get('total_calls', 0)}",
                f"- 平均执行时间: {perf_data.get('avg_execution_time', 0):.4f}s",
                f"- 最大执行时间: {perf_data.get('max_execution_time', 0):.4f}s",
                f"- 平均CPU使用率: {perf_data.get('avg_cpu_usage', 0):.1f}%",
                f"- 平均内存使用率: {perf_data.get('avg_memory_usage', 0):.1f}%",
                ""
            ])
            
        # 错误统计
        if 'errors' in debug_data:
            error_data = debug_data['errors']
            lines.extend([
                "## 错误摘要",
                f"- 总错误数: {error_data.get('total_errors', 0)}",
                f"- 错误类型分布: {error_data.get('error_type_distribution', {})}",
                ""
            ])
            
        # 日志统计
        if 'logs' in debug_data:
            log_data = debug_data['logs']
            lines.extend([
                "## 日志摘要",
                f"- 总日志条数: {len(log_data.get('logs', []))}",
                f"- 日志级别分布: {self._count_log_levels(log_data.get('logs', []))}",
                ""
            ])
            
        return "\n".join(lines)
        
    def _generate_performance_report(self, debug_data: Dict[str, Any]) -> str:
        """生成性能报告"""
        lines = [
            "# 性能分析报告",
            f"生成时间: {datetime.datetime.now().isoformat()}",
            ""
        ]
        
        if 'performance' in debug_data:
            perf_data = debug_data['performance']
            
            lines.extend([
                "## 性能指标",
                f"- 总函数调用次数: {perf_data.get('total_calls', 0)}",
                f"- 平均执行时间: {perf_data.get('avg_execution_time', 0):.6f}s",
                f"- 最大执行时间: {perf_data.get('max_execution_time', 0):.6f}s",
                f"- 平均CPU使用率: {perf_data.get('avg_cpu_usage', 0):.2f}%",
                f"- 平均内存使用率: {perf_data.get('avg_memory_usage', 0):.2f}%",
                ""
            ])
            
            # 性能建议
            lines.extend([
                "## 性能建议",
                self._generate_performance_suggestions(perf_data),
                ""
            ])
            
        return "\n".join(lines)
        
    def _generate_error_report(self, debug_data: Dict[str, Any]) -> str:
        """生成错误报告"""
        lines = [
            "# 错误分析报告",
            f"生成时间: {datetime.datetime.now().isoformat()}",
            ""
        ]
        
        if 'errors' in debug_data:
            error_data = debug_data['errors']
            
            lines.extend([
                "## 错误统计",
                f"- 总错误数: {error_data.get('total_errors', 0)}",
                ""
            ])
            
            # 错误类型分布
            error_dist = error_data.get('error_type_distribution', {})
            if error_dist:
                lines.extend(["### 错误类型分布", ""])
                for error_type, count in error_dist.items():
                    lines.append(f"- {error_type}: {count}")
                lines.append("")
                
            # 最近错误
            recent_errors = error_data.get('recent_errors', [])
            if recent_errors:
                lines.extend(["### 最近错误", ""])
                for error in recent_errors[:10]:
                    timestamp = datetime.datetime.fromtimestamp(error['timestamp']).isoformat()
                    lines.append(f"- [{timestamp}] {error['type']}: {error['message']}")
                lines.append("")
                
        return "\n".join(lines)
        
    def _generate_log_report(self, debug_data: Dict[str, Any]) -> str:
        """生成日志报告"""
        lines = [
            "# 日志分析报告",
            f"生成时间: {datetime.datetime.now().isoformat()}",
            ""
        ]
        
        if 'logs' in debug_data:
            logs = debug_data['logs']
            
            lines.extend([
                "## 日志统计",
                f"- 总日志条数: {len(logs)}",
                f"- 日志级别分布: {self._count_log_levels(logs)}",
                ""
            ])
            
            # 最近日志
            lines.extend(["### 最近日志", ""])
            for log in logs[-20:]:  # 最近20条
                timestamp = datetime.datetime.fromtimestamp(log['timestamp']).isoformat()
                lines.append(f"- [{timestamp}] {log['level']}: {log['message']}")
            lines.append("")
            
        return "\n".join(lines)
        
    def _generate_full_report(self, debug_data: Dict[str, Any]) -> str:
        """生成完整报告"""
        lines = [
            "# 完整调试报告",
            f"生成时间: {datetime.datetime.now().isoformat()}",
            ""
        ]
        
        # 包含所有其他报告
        lines.extend([
            self._generate_summary_report(debug_data),
            self._generate_performance_report(debug_data),
            self._generate_error_report(debug_data),
            self._generate_log_report(debug_data),
            ""
        ])
        
        return "\n".join(lines)
        
    def _count_log_levels(self, logs: List[Dict[str, Any]]) -> Dict[str, int]:
        """统计日志级别"""
        level_counts = defaultdict(int)
        for log in logs:
            level_counts[log.get('level', 'UNKNOWN')] += 1
        return dict(level_counts)
        
    def _generate_performance_suggestions(self, perf_data: Dict[str, Any]) -> str:
        """生成性能建议"""
        suggestions = []
        
        avg_time = perf_data.get('avg_execution_time', 0)
        if avg_time > 1.0:
            suggestions.append("- 执行时间较长，建议优化算法效率")
            
        avg_cpu = perf_data.get('avg_cpu_usage', 0)
        if avg_cpu > 80:
            suggestions.append("- CPU使用率较高，建议检查计算密集型操作")
            
        avg_memory = perf_data.get('avg_memory_usage', 0)
        if avg_memory > 80:
            suggestions.append("- 内存使用率较高，建议检查内存泄漏")
            
        if not suggestions:
            suggestions.append("- 性能表现良好")
            
        return "\n".join(suggestions)


class DistributedDebugger:
    """分布式调试工具类"""
    
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.debug_sessions: Dict[str, Dict[str, Any]] = {}
        self.message_queue: deque = deque(maxlen=10000)
        self.coordination_lock = threading.Lock()
        
    def register_node(self, node_id: str, node_info: Dict[str, Any]) -> bool:
        """注册调试节点"""
        try:
            with self.coordination_lock:
                self.nodes[node_id] = {
                    **node_info,
                    'registration_time': time.time(),
                    'last_heartbeat': time.time(),
                    'status': 'active'
                }
                logger.info(f"注册调试节点: {node_id}")
                return True
        except Exception as e:
            logger.error(f"注册节点失败: {e}")
            return False
            
    def unregister_node(self, node_id: str) -> bool:
        """注销调试节点"""
        try:
            with self.coordination_lock:
                if node_id in self.nodes:
                    del self.nodes[node_id]
                    logger.info(f"注销调试节点: {node_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"注销节点失败: {e}")
            return False
            
    def create_debug_session(self, session_id: str, participating_nodes: List[str]) -> bool:
        """创建调试会话"""
        try:
            with self.coordination_lock:
                # 验证所有节点都存在
                for node_id in participating_nodes:
                    if node_id not in self.nodes:
                        logger.error(f"节点不存在: {node_id}")
                        return False
                        
                self.debug_sessions[session_id] = {
                    'session_id': session_id,
                    'nodes': participating_nodes,
                    'creation_time': time.time(),
                    'status': 'active',
                    'debug_data': defaultdict(list)
                }
                
                logger.info(f"创建调试会话: {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"创建调试会话失败: {e}")
            return False
            
    def send_debug_command(self, session_id: str, command: Dict[str, Any]) -> bool:
        """发送调试命令"""
        try:
            with self.coordination_lock:
                if session_id not in self.debug_sessions:
                    logger.error(f"调试会话不存在: {session_id}")
                    return False
                    
                session = self.debug_sessions[session_id]
                
                # 将命令发送到所有参与的节点
                for node_id in session['nodes']:
                    if node_id in self.nodes:
                        command_message = {
                            'session_id': session_id,
                            'node_id': node_id,
                            'command': command,
                            'timestamp': time.time()
                        }
                        self.message_queue.append(command_message)
                        
                logger.debug(f"发送调试命令到会话 {session_id}")
                return True
                
        except Exception as e:
            logger.error(f"发送调试命令失败: {e}")
            return False
            
    def collect_debug_data(self, session_id: str, node_id: str, 
                          debug_data: Dict[str, Any]) -> bool:
        """收集调试数据"""
        try:
            with self.coordination_lock:
                if session_id in self.debug_sessions:
                    session = self.debug_sessions[session_id]
                    session['debug_data'][node_id].append({
                        'data': debug_data,
                        'timestamp': time.time()
                    })
                    return True
                return False
        except Exception as e:
            logger.error(f"收集调试数据失败: {e}")
            return False
            
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """获取会话状态"""
        with self.coordination_lock:
            if session_id in self.debug_sessions:
                session = self.debug_sessions[session_id]
                return {
                    'session_id': session_id,
                    'nodes': session['nodes'],
                    'creation_time': session['creation_time'],
                    'status': session['status'],
                    'data_count': sum(len(data_list) for data_list in session['debug_data'].values())
                }
            return {}
            
    def close_session(self, session_id: str) -> bool:
        """关闭调试会话"""
        try:
            with self.coordination_lock:
                if session_id in self.debug_sessions:
                    del self.debug_sessions[session_id]
                    logger.info(f"关闭调试会话: {session_id}")
                    return True
                return False
        except Exception as e:
            logger.error(f"关闭调试会话失败: {e}")
            return False
            
    def get_all_nodes(self) -> Dict[str, Dict[str, Any]]:
        """获取所有节点"""
        with self.coordination_lock:
            return self.nodes.copy()
            
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """获取所有会话"""
        with self.coordination_lock:
            return {sid: {
                'session_id': data['session_id'],
                'nodes': data['nodes'],
                'creation_time': data['creation_time'],
                'status': data['status']
            } for sid, data in self.debug_sessions.items()}


class J8DebuggingTools:
    """J8调试工具主类"""
    
    def __init__(self, enable_async: bool = True):
        """初始化J8调试工具"""
        self.enable_async = enable_async
        
        # 初始化各个调试组件
        self.code_debugger = CodeDebugger()
        self.performance_debugger = PerformanceDebugger()
        self.log_debugger = LogDebugger()
        self.error_diagnostician = ErrorDiagnostician()
        self.realtime_debugger = RealtimeDebugger()
        self.debug_reporter = DebugReporter()
        self.distributed_debugger = DistributedDebugger()
        
        # 配置默认设置
        self._setup_default_configuration()
        
        logger.info("J8调试工具初始化完成")
        
    def _setup_default_configuration(self):
        """设置默认配置"""
        # 设置默认日志级别
        self.log_debugger.set_log_level(DebugLevel.DEBUG)
        
        # 添加默认日志过滤器
        self.log_debugger.add_log_filter(self._default_log_filter)
        
        # 启动实时监控
        self.realtime_debugger.start_monitoring()
        
    def _default_log_filter(self, debug_info: DebugInfo) -> bool:
        """默认日志过滤器"""
        # 过滤掉过于频繁的日志
        return True
        
    # 代码调试接口
    def debug_function(self, func: Callable = None, *, function_name: str = None):
        """函数调试装饰器"""
        def decorator(f):
            func_name = function_name or f.__name__
            file_path = inspect.getfile(f)
            
            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                # 进入调试上下文
                with self.code_debugger.debug_context(func_name, file_path):
                    try:
                        # 记录函数调用
                        self.log_debugger.log_debug_info(
                            DebugLevel.DEBUG,
                            f"调用函数: {func_name}",
                            file_path=file_path,
                            function_name=func_name,
                            variables={'args': args, 'kwargs': kwargs}
                        )
                        
                        # 性能计时
                        with self.performance_debugger.performance_timer(func_name, file_path):
                            result = f(*args, **kwargs)
                            
                        return result
                        
                    except Exception as e:
                        # 错误诊断
                        error_info = self.error_diagnostician.diagnose_error(
                            e,
                            {'function_name': func_name, 'file_path': file_path}
                        )
                        
                        # 重新抛出异常
                        raise
                        
            return wrapper
            
        if func is None:
            return decorator
        else:
            return decorator(func)
            
    def set_breakpoint(self, file_path: str, line_number: int, 
                      condition: Callable = None) -> str:
        """设置断点"""
        return self.code_debugger.add_breakpoint(file_path, line_number, condition)
        
    def remove_breakpoint(self, breakpoint_id: str) -> bool:
        """移除断点"""
        return self.code_debugger.remove_breakpoint(breakpoint_id)
        
    def get_call_stack(self) -> List[Dict[str, Any]]:
        """获取调用栈"""
        return self.code_debugger.get_call_stack()
        
    # 性能调试接口
    def start_profiling(self, profile_name: str = "default") -> bool:
        """开始性能分析"""
        return self.performance_debugger.start_profiling(profile_name)
        
    def stop_profiling(self, profile_name: str = "default") -> Optional[str]:
        """停止性能分析"""
        return self.performance_debugger.stop_profiling(profile_name)
        
    def get_performance_summary(self, function_name: str = None) -> Dict[str, Any]:
        """获取性能摘要"""
        return self.performance_debugger.get_performance_summary(function_name)
        
    def detect_memory_leaks(self) -> Dict[str, Any]:
        """检测内存泄漏"""
        return self.performance_debugger.detect_memory_leaks()
        
    # 日志调试接口
    def log(self, level: DebugLevel, message: str, **kwargs):
        """记录日志"""
        frame = inspect.currentframe()
        try:
            file_path = frame.f_code.co_filename
            line_number = frame.f_lineno
            function_name = frame.f_code.co_name
            
            self.log_debugger.log_debug_info(
                level, message, file_path, line_number, function_name, **kwargs
            )
        finally:
            del frame
            
    def get_logs(self, filter_func: Callable = None, limit: int = 100) -> List[DebugInfo]:
        """获取日志"""
        return self.log_debugger.get_logs(filter_func, limit)
        
    def export_logs(self, file_path: str, format: str = "json") -> bool:
        """导出日志"""
        return self.log_debugger.export_logs(file_path, format)
        
    # 错误诊断接口
    def diagnose_error(self, exception: Exception, context: Dict[str, Any] = None) -> ErrorInfo:
        """诊断错误"""
        return self.error_diagnostician.diagnose_error(exception, context)
        
    def get_error_statistics(self) -> Dict[str, Any]:
        """获取错误统计"""
        return self.error_diagnostician.get_error_statistics()
        
    # 实时监控接口
    def add_watcher(self, name: str, watcher_func: Callable):
        """添加监控器"""
        self.realtime_debugger.add_watcher(name, watcher_func)
        
    def get_recent_events(self, limit: int = 100) -> List[Dict[str, Any]]:
        """获取最近事件"""
        return self.realtime_debugger.get_recent_events(limit)
        
    # 调试报告接口
    def generate_report(self, report_type: str, output_path: str = None) -> str:
        """生成调试报告"""
        # 收集调试数据
        debug_data = self._collect_debug_data()
        return self.debug_reporter.generate_report(report_type, debug_data, output_path)
        
    def _collect_debug_data(self) -> Dict[str, Any]:
        """收集所有调试数据"""
        return {
            'performance': self.get_performance_summary(),
            'errors': self.get_error_statistics(),
            'logs': {'logs': [log.__dict__ for log in self.get_logs(limit=1000)]},
            'realtime_events': self.get_recent_events(limit=100)
        }
        
    # 分布式调试接口
    def register_debug_node(self, node_id: str, node_info: Dict[str, Any]) -> bool:
        """注册调试节点"""
        return self.distributed_debugger.register_node(node_id, node_info)
        
    def create_distributed_session(self, session_id: str, nodes: List[str]) -> bool:
        """创建分布式调试会话"""
        return self.distributed_debugger.create_debug_session(session_id, nodes)
        
    # 异步支持
    @asynccontextmanager
    async def async_debug_context(self, function_name: str, file_path: str):
        """异步调试上下文"""
        # 记录异步函数调用
        self.log_debugger.log_debug_info(
            DebugLevel.DEBUG,
            f"异步调用函数: {function_name}",
            file_path=file_path,
            function_name=function_name
        )
        
        # 性能计时
        start_time = time.time()
        try:
            yield
        finally:
            execution_time = time.time() - start_time
            self.performance_debugger.record_performance(function_name, execution_time, file_path)
            
    async def async_debug_function(self, func: Callable):
        """异步函数调试装饰器"""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = func.__name__
            file_path = inspect.getfile(func)
            
            async with self.async_debug_context(func_name, file_path):
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    self.error_diagnostician.diagnose_error(
                        e, {'function_name': func_name, 'file_path': file_path}
                    )
                    raise
                    
        return wrapper
        
    # 清理和关闭
    def cleanup(self):
        """清理资源"""
        try:
            # 停止性能监控
            self.performance_debugger.stop_monitoring()
            
            # 停止实时监控
            self.realtime_debugger.stop_monitoring()
            
            logger.info("J8调试工具清理完成")
            
        except Exception as e:
            logger.error(f"清理资源失败: {e}")
            
    def __enter__(self):
        """上下文管理器入口"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()


# 全局调试工具实例
_global_debugger = None

def get_debugger() -> J8DebuggingTools:
    """获取全局调试工具实例"""
    global _global_debugger
    if _global_debugger is None:
        _global_debugger = J8DebuggingTools()
    return _global_debugger


# 便捷函数
def debug_function(func: Callable = None, **kwargs):
    """便捷的函数调试装饰器"""
    return get_debugger().debug_function(func, **kwargs)


def log_debug(level: DebugLevel, message: str, **kwargs):
    """便捷的日志记录函数"""
    get_debugger().log(level, message, **kwargs)


def set_breakpoint(file_path: str, line_number: int, condition: Callable = None) -> str:
    """便捷的断点设置函数"""
    return get_debugger().set_breakpoint(file_path, line_number, condition)


def start_profiling(profile_name: str = "default") -> bool:
    """便捷的性能分析启动函数"""
    return get_debugger().start_profiling(profile_name)


def stop_profiling(profile_name: str = "default") -> Optional[str]:
    """便捷的性能分析停止函数"""
    return get_debugger().stop_profiling(profile_name)


def generate_debug_report(report_type: str, output_path: str = None) -> str:
    """便捷的调试报告生成函数"""
    return get_debugger().generate_report(report_type, output_path)


# 高级调试工具类
class AdvancedDebugger:
    """高级调试工具类"""
    
    def __init__(self, base_debugger: J8DebuggingTools):
        self.base_debugger = base_debugger
        self.breakpoint_conditions: Dict[str, Callable] = {}
        self.conditional_breakpoints: Dict[str, Breakpoint] = {}
        self.debug_scripts: Dict[str, str] = {}
        self.automation_rules: List[Dict[str, Any]] = []
        self.debug_history: List[Dict[str, Any]] = []
        self.custom_watchers: Dict[str, Callable] = {}
        self.debug_session_data: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
    def add_conditional_breakpoint(self, breakpoint_id: str, condition_func: Callable):
        """添加条件断点"""
        with self._lock:
            self.breakpoint_conditions[breakpoint_id] = condition_func
            logger.info(f"添加条件断点: {breakpoint_id}")
            
    def create_debug_script(self, script_name: str, script_content: str):
        """创建调试脚本"""
        with self._lock:
            self.debug_scripts[script_name] = script_content
            logger.info(f"创建调试脚本: {script_name}")
            
    def execute_debug_script(self, script_name: str, context: Dict[str, Any] = None) -> Any:
        """执行调试脚本"""
        try:
            if script_name not in self.debug_scripts:
                raise ValueError(f"调试脚本不存在: {script_name}")
                
            script_content = self.debug_scripts[script_name]
            context = context or {}
            
            # 创建安全的执行环境
            safe_globals = {
                '__builtins__': __builtins__,
                'debugger': self.base_debugger,
                'context': context,
                'time': time,
                'datetime': datetime,
                'json': json,
                'logging': logging
            }
            
            # 执行脚本
            exec(script_content, safe_globals)
            logger.info(f"执行调试脚本: {script_name}")
            return True
            
        except Exception as e:
            logger.error(f"执行调试脚本失败: {e}")
            return None
            
    def add_automation_rule(self, rule_name: str, trigger_condition: Callable, 
                          action: Callable, description: str = ""):
        """添加自动化规则"""
        rule = {
            'name': rule_name,
            'trigger': trigger_condition,
            'action': action,
            'description': description,
            'creation_time': time.time(),
            'execution_count': 0,
            'enabled': True
        }
        
        with self._lock:
            self.automation_rules.append(rule)
            logger.info(f"添加自动化规则: {rule_name}")
            
    def process_automation_rules(self, debug_context: Dict[str, Any]):
        """处理自动化规则"""
        with self._lock:
            for rule in self.automation_rules:
                if not rule['enabled']:
                    continue
                    
                try:
                    if rule['trigger'](debug_context):
                        result = rule['action'](debug_context)
                        rule['execution_count'] += 1
                        logger.debug(f"执行自动化规则: {rule['name']}")
                        
                except Exception as e:
                    logger.error(f"自动化规则执行失败 {rule['name']}: {e}")
                    
    def add_custom_watcher(self, watcher_name: str, watcher_func: Callable):
        """添加自定义监控器"""
        with self._lock:
            self.custom_watchers[watcher_name] = watcher_func
            logger.info(f"添加自定义监控器: {watcher_name}")
            
    def start_debug_session(self, session_name: str, session_config: Dict[str, Any] = None):
        """开始调试会话"""
        with self._lock:
            self.debug_session_data[session_name] = {
                'config': session_config or {},
                'start_time': time.time(),
                'events': [],
                'breakpoints_hit': [],
                'errors_occurred': [],
                'performance_metrics': []
            }
            logger.info(f"开始调试会话: {session_name}")
            
    def end_debug_session(self, session_name: str) -> Dict[str, Any]:
        """结束调试会话"""
        with self._lock:
            if session_name in self.debug_session_data:
                session_data = self.debug_session_data[session_name]
                session_data['end_time'] = time.time()
                session_data['duration'] = session_data['end_time'] - session_data['start_time']
                
                # 生成会话报告
                report = self._generate_session_report(session_name, session_data)
                logger.info(f"结束调试会话: {session_name}")
                return report
            return {}
            
    def _generate_session_report(self, session_name: str, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成会话报告"""
        return {
            'session_name': session_name,
            'duration': session_data.get('duration', 0),
            'events_count': len(session_data.get('events', [])),
            'breakpoints_hit': len(session_data.get('breakpoints_hit', [])),
            'errors_count': len(session_data.get('errors_occurred', [])),
            'performance_summary': self._analyze_session_performance(session_data)
        }
        
    def _analyze_session_performance(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """分析会话性能"""
        metrics = session_data.get('performance_metrics', [])
        if not metrics:
            return {}
            
        return {
            'avg_execution_time': sum(m.get('execution_time', 0) for m in metrics) / len(metrics),
            'total_function_calls': len(metrics),
            'slowest_function': max(metrics, key=lambda x: x.get('execution_time', 0), default={})
        }
        
    def record_debug_history(self, event_type: str, data: Dict[str, Any]):
        """记录调试历史"""
        with self._lock:
            self.debug_history.append({
                'timestamp': time.time(),
                'event_type': event_type,
                'data': data
            })
            
            # 限制历史记录数量
            if len(self.debug_history) > 5000:
                self.debug_history.pop(0)
                
    def get_debug_history(self, event_type: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """获取调试历史"""
        with self._lock:
            history = self.debug_history
            if event_type:
                history = [event for event in history if event['event_type'] == event_type]
            return history[-limit:] if len(history) > limit else history


class NetworkDebugger:
    """网络调试工具类"""
    
    def __init__(self):
        self.network_requests: List[Dict[str, Any]] = []
        self.response_times: Dict[str, List[float]] = defaultdict(list)
        self.error_requests: List[Dict[str, Any]] = []
        self.request_patterns: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        
    def log_request(self, method: str, url: str, headers: Dict[str, Any] = None,
                   data: Any = None, response_time: float = None, status_code: int = None,
                   error: str = None):
        """记录网络请求"""
        request_info = {
            'timestamp': time.time(),
            'method': method,
            'url': url,
            'headers': headers or {},
            'data': data,
            'response_time': response_time,
            'status_code': status_code,
            'error': error
        }
        
        with self._lock:
            self.network_requests.append(request_info)
            
            # 记录响应时间
            if response_time:
                self.response_times[url].append(response_time)
                
            # 记录错误请求
            if error or (status_code and status_code >= 400):
                self.error_requests.append(request_info)
                
            # 记录请求模式
            self.request_patterns[f"{method} {url}"] += 1
            
            # 限制记录数量
            if len(self.network_requests) > 10000:
                self.network_requests.pop(0)
                if self.error_requests:
                    self.error_requests.pop(0)
                    
    def get_request_statistics(self) -> Dict[str, Any]:
        """获取请求统计"""
        with self._lock:
            total_requests = len(self.network_requests)
            error_requests = len(self.error_requests)
            error_rate = (error_requests / total_requests * 100) if total_requests > 0 else 0
            
            # 平均响应时间
            all_response_times = []
            for times in self.response_times.values():
                all_response_times.extend(times)
            avg_response_time = sum(all_response_times) / len(all_response_times) if all_response_times else 0
            
            # 最慢的请求
            slowest_requests = sorted(
                [(req['url'], req['response_time']) for req in self.network_requests if req['response_time']],
                key=lambda x: x[1], reverse=True
            )[:10]
            
            return {
                'total_requests': total_requests,
                'error_requests': error_requests,
                'error_rate': error_rate,
                'avg_response_time': avg_response_time,
                'slowest_requests': slowest_requests,
                'request_patterns': dict(self.request_patterns)
            }
            
    def detect_slow_requests(self, threshold: float = 5.0) -> List[Dict[str, Any]]:
        """检测慢请求"""
        with self._lock:
            return [req for req in self.network_requests 
                   if req.get('response_time', 0) > threshold]
                   
    def detect_failed_requests(self) -> List[Dict[str, Any]]:
        """检测失败请求"""
        with self._lock:
            return self.error_requests.copy()


class DatabaseDebugger:
    """数据库调试工具类"""
    
    def __init__(self):
        self.sql_queries: List[Dict[str, Any]] = []
        self.query_performance: Dict[str, List[float]] = defaultdict(list)
        self.slow_queries: List[Dict[str, Any]] = []
        self.error_queries: List[Dict[str, Any]] = []
        self.query_patterns: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        
    def log_query(self, query: str, parameters: Dict[str, Any] = None,
                 execution_time: float = None, row_count: int = None,
                 error: str = None, connection_id: str = None):
        """记录SQL查询"""
        query_info = {
            'timestamp': time.time(),
            'query': query,
            'parameters': parameters or {},
            'execution_time': execution_time,
            'row_count': row_count,
            'error': error,
            'connection_id': connection_id
        }
        
        with self._lock:
            self.sql_queries.append(query_info)
            
            # 记录查询性能
            if execution_time:
                query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
                self.query_performance[query_hash].append(execution_time)
                
            # 记录慢查询
            if execution_time and execution_time > 1.0:  # 1秒阈值
                self.slow_queries.append(query_info)
                
            # 记录错误查询
            if error:
                self.error_queries.append(query_info)
                
            # 记录查询模式
            self.query_patterns[query.strip()] += 1
            
            # 限制记录数量
            if len(self.sql_queries) > 10000:
                self.sql_queries.pop(0)
                if self.slow_queries:
                    self.slow_queries.pop(0)
                if self.error_queries:
                    self.error_queries.pop(0)
                    
    def get_query_statistics(self) -> Dict[str, Any]:
        """获取查询统计"""
        with self._lock:
            total_queries = len(self.sql_queries)
            slow_queries = len(self.slow_queries)
            error_queries = len(self.error_queries)
            
            # 平均执行时间
            all_times = [q.get('execution_time', 0) for q in self.sql_queries if q.get('execution_time')]
            avg_execution_time = sum(all_times) / len(all_times) if all_times else 0
            
            # 最慢的查询
            slowest_queries = sorted(
                [(q['query'], q['execution_time']) for q in self.sql_queries if q.get('execution_time')],
                key=lambda x: x[1], reverse=True
            )[:10]
            
            return {
                'total_queries': total_queries,
                'slow_queries': slow_queries,
                'error_queries': error_queries,
                'avg_execution_time': avg_execution_time,
                'slowest_queries': slowest_queries,
                'query_patterns': dict(self.query_patterns)
            }
            
    def detect_n_plus_one_queries(self) -> List[Dict[str, Any]]:
        """检测N+1查询问题"""
        with self._lock:
            # 简单的N+1检测：查找相同模式但参数不同的查询
            potential_n_plus_one = []
            query_groups = defaultdict(list)
            
            for query_info in self.sql_queries:
                query = query_info['query']
                # 简化查询模式（移除具体参数）
                pattern = ' '.join(query.split()[:3])  # 取前3个词作为模式
                query_groups[pattern].append(query_info)
                
            for pattern, queries in query_groups.items():
                if len(queries) > 5:  # 相同模式查询超过5次
                    potential_n_plus_one.extend(queries)
                    
            return potential_n_plus_one
            
    def optimize_query_suggestions(self) -> List[str]:
        """查询优化建议"""
        suggestions = []
        
        with self._lock:
            # 基于统计信息生成建议
            stats = self.get_query_statistics()
            
            if stats['slow_queries'] > 0:
                suggestions.append("发现慢查询，建议添加索引或优化查询语句")
                
            if stats['error_queries'] > 0:
                suggestions.append("存在错误查询，检查SQL语法和参数")
                
            if len(self.detect_n_plus_one_queries()) > 0:
                suggestions.append("检测到可能的N+1查询问题，考虑使用JOIN或预加载")
                
            # 检查是否有全表扫描
            for query_info in self.sql_queries:
                query = query_info['query'].upper()
                if 'SELECT *' in query and 'WHERE' not in query:
                    suggestions.append("发现全表扫描查询，建议指定具体字段和添加WHERE条件")
                    break
                    
        if not suggestions:
            suggestions.append("数据库查询性能良好")
            
        return suggestions


class MemoryDebugger:
    """内存调试工具类"""
    
    def __init__(self):
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.object_tracking: Dict[int, Dict[str, Any]] = {}
        self.memory_leaks: List[Dict[str, Any]] = []
        self.gc_statistics: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
    def take_memory_snapshot(self, description: str = ""):
        """拍摄内存快照"""
        try:
            import tracemalloc
            
            if not tracemalloc.is_tracing():
                tracemalloc.start()
                
            # 获取当前内存使用情况
            current, peak = tracemalloc.get_traced_memory()
            
            # 获取对象统计
            object_counts = self._get_object_counts()
            
            snapshot = {
                'timestamp': time.time(),
                'description': description,
                'current_memory': current,
                'peak_memory': peak,
                'object_counts': object_counts,
                'gc_stats': gc.get_stats()
            }
            
            with self._lock:
                self.memory_snapshots.append(snapshot)
                
                # 限制快照数量
                if len(self.memory_snapshots) > 100:
                    self.memory_snapshots.pop(0)
                    
            logger.debug(f"内存快照已保存: {description}")
            return snapshot
            
        except Exception as e:
            logger.error(f"拍摄内存快照失败: {e}")
            return None
            
    def _get_object_counts(self) -> Dict[str, int]:
        """获取对象统计"""
        object_counts = {}
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            object_counts[obj_type] = object_counts.get(obj_type, 0) + 1
        return object_counts
        
    def track_object(self, obj: Any, object_id: str = None):
        """跟踪对象"""
        object_id = object_id or str(id(obj))
        
        with self._lock:
            self.object_tracking[id(obj)] = {
                'object_id': object_id,
                'object_type': type(obj).__name__,
                'creation_time': time.time(),
                'last_access_time': time.time(),
                'access_count': 1
            }
            
    def update_object_access(self, obj: Any):
        """更新对象访问信息"""
        obj_id = id(obj)
        with self._lock:
            if obj_id in self.object_tracking:
                self.object_tracking[obj_id]['last_access_time'] = time.time()
                self.object_tracking[obj_id]['access_count'] += 1
                
    def detect_memory_leaks(self, threshold_growth: float = 10.0) -> List[Dict[str, Any]]:
        """检测内存泄漏"""
        leaks = []
        
        with self._lock:
            if len(self.memory_snapshots) < 2:
                return leaks
                
            # 分析内存增长趋势
            for i in range(1, len(self.memory_snapshots)):
                prev_snapshot = self.memory_snapshots[i-1]
                curr_snapshot = self.memory_snapshots[i]
                
                memory_growth = curr_snapshot['current_memory'] - prev_snapshot['current_memory']
                growth_mb = memory_growth / (1024 * 1024)
                
                if growth_mb > threshold_growth:
                    leak_info = {
                        'timestamp': curr_snapshot['timestamp'],
                        'memory_growth_mb': growth_mb,
                        'description': curr_snapshot['description'],
                        'object_changes': self._compare_object_counts(
                            prev_snapshot['object_counts'],
                            curr_snapshot['object_counts']
                        )
                    }
                    leaks.append(leak_info)
                    
        return leaks
        
    def _compare_object_counts(self, prev_counts: Dict[str, int], 
                             curr_counts: Dict[str, int]) -> Dict[str, int]:
        """比较对象数量变化"""
        changes = {}
        all_types = set(prev_counts.keys()) | set(curr_counts.keys())
        
        for obj_type in all_types:
            prev_count = prev_counts.get(obj_type, 0)
            curr_count = curr_counts.get(obj_type, 0)
            change = curr_count - prev_count
            
            if change != 0:
                changes[obj_type] = change
                
        return changes
        
    def get_memory_statistics(self) -> Dict[str, Any]:
        """获取内存统计"""
        with self._lock:
            if not self.memory_snapshots:
                return {}
                
            latest_snapshot = self.memory_snapshots[-1]
            
            # 计算内存增长
            memory_growth = 0
            if len(self.memory_snapshots) > 1:
                first_snapshot = self.memory_snapshots[0]
                memory_growth = latest_snapshot['current_memory'] - first_snapshot['current_memory']
                
            # 对象类型统计
            top_object_types = sorted(
                latest_snapshot['object_counts'].items(),
                key=lambda x: x[1], reverse=True
            )[:10]
            
            return {
                'current_memory_mb': latest_snapshot['current_memory'] / (1024 * 1024),
                'peak_memory_mb': latest_snapshot['peak_memory'] / (1024 * 1024),
                'memory_growth_mb': memory_growth / (1024 * 1024),
                'total_snapshots': len(self.memory_snapshots),
                'tracked_objects': len(self.object_tracking),
                'top_object_types': top_object_types,
                'detected_leaks': len(self.memory_leaks)
            }
            
    def force_garbage_collection(self):
        """强制垃圾回收"""
        collected = gc.collect()
        
        with self._lock:
            self.gc_statistics.append({
                'timestamp': time.time(),
                'collected_objects': collected,
                'gc_stats': gc.get_stats()
            })
            
        logger.debug(f"强制垃圾回收，回收了 {collected} 个对象")
        return collected


class SecurityDebugger:
    """安全调试工具类"""
    
    def __init__(self):
        self.security_events: List[Dict[str, Any]] = []
        self.suspicious_activities: List[Dict[str, Any]] = []
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.security_rules: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
    def log_security_event(self, event_type: str, severity: str, 
                          description: str, source: str = "", context: Dict[str, Any] = None):
        """记录安全事件"""
        event = {
            'timestamp': time.time(),
            'event_type': event_type,
            'severity': severity,
            'description': description,
            'source': source,
            'context': context or {}
        }
        
        with self._lock:
            self.security_events.append(event)
            
            # 检查是否为可疑活动
            if severity in ['HIGH', 'CRITICAL']:
                self.suspicious_activities.append(event)
                
            # 限制记录数量
            if len(self.security_events) > 5000:
                self.security_events.pop(0)
                if self.suspicious_activities:
                    self.suspicious_activities.pop(0)
                    
    def add_security_rule(self, rule_name: str, condition: Callable, 
                         action: Callable, description: str = ""):
        """添加安全规则"""
        rule = {
            'name': rule_name,
            'condition': condition,
            'action': action,
            'description': description,
            'creation_time': time.time(),
            'trigger_count': 0
        }
        
        with self._lock:
            self.security_rules.append(rule)
            logger.info(f"添加安全规则: {rule_name}")
            
    def check_security_rules(self, context: Dict[str, Any]):
        """检查安全规则"""
        with self._lock:
            for rule in self.security_rules:
                try:
                    if rule['condition'](context):
                        rule['trigger_count'] += 1
                        result = rule['action'](context)
                        logger.warning(f"安全规则触发: {rule['name']}")
                        
                except Exception as e:
                    logger.error(f"安全规则执行失败 {rule['name']}: {e}")
                    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测异常行为"""
        anomalies = []
        
        with self._lock:
            # 检测访问频率异常
            current_time = time.time()
            recent_events = [
                event for event in self.security_events
                if current_time - event['timestamp'] < 300  # 最近5分钟
            ]
            
            # 按源分组统计
            source_counts = defaultdict(int)
            for event in recent_events:
                source = event.get('source', 'unknown')
                source_counts[source] += 1
                
            # 检测访问频率过高的源
            for source, count in source_counts.items():
                if count > 50:  # 5分钟内超过50次访问
                    anomalies.append({
                        'type': 'high_frequency_access',
                        'source': source,
                        'count': count,
                        'time_window': '5 minutes',
                        'severity': 'MEDIUM'
                    })
                    
        return anomalies
        
    def get_security_summary(self) -> Dict[str, Any]:
        """获取安全摘要"""
        with self._lock:
            total_events = len(self.security_events)
            high_severity_events = len([
                event for event in self.security_events 
                if event['severity'] in ['HIGH', 'CRITICAL']
            ])
            
            # 最近24小时的事件
            day_ago = time.time() - 86400
            recent_events = [
                event for event in self.security_events 
                if event['timestamp'] > day_ago
            ]
            
            # 事件类型分布
            event_types = defaultdict(int)
            for event in self.security_events:
                event_types[event['event_type']] += 1
                
            return {
                'total_events': total_events,
                'high_severity_events': high_severity_events,
                'recent_events_24h': len(recent_events),
                'suspicious_activities': len(self.suspicious_activities),
                'event_type_distribution': dict(event_types),
                'security_rules_count': len(self.security_rules),
                'anomalies_detected': len(self.detect_anomalies())
            }


# 扩展J8DebuggingTools类
class ExtendedJ8DebuggingTools(J8DebuggingTools):
    """扩展的J8调试工具类"""
    
    def __init__(self, enable_async: bool = True):
        super().__init__(enable_async)
        
        # 添加高级调试组件
        self.advanced_debugger = AdvancedDebugger(self)
        self.network_debugger = NetworkDebugger()
        self.database_debugger = DatabaseDebugger()
        self.memory_debugger = MemoryDebugger()
        self.security_debugger = SecurityDebugger()
        
        # 扩展的调试配置
        self._setup_extended_configuration()
        
    def _setup_extended_configuration(self):
        """设置扩展配置"""
        # 添加默认安全规则
        self.security_debugger.add_security_rule(
            "sql_injection_detection",
            lambda ctx: self._detect_sql_injection(ctx),
            lambda ctx: self._handle_security_alert("SQL注入检测", ctx),
            "检测可能的SQL注入攻击"
        )
        
        # 添加默认自动化规则
        self.advanced_debugger.add_automation_rule(
            "performance_threshold_alert",
            lambda ctx: ctx.get('execution_time', 0) > 5.0,
            lambda ctx: self._handle_performance_alert(ctx),
            "性能阈值告警"
        )
        
    def _detect_sql_injection(self, context: Dict[str, Any]) -> bool:
        """检测SQL注入"""
        query = context.get('query', '')
        if not query:
            return False
            
        # 简单的SQL注入检测模式
        injection_patterns = [
            "union select",
            "drop table",
            "delete from",
            "insert into",
            "update set",
            "--",
            "/*",
            "*/",
            "xp_cmdshell",
            "sp_executesql"
        ]
        
        query_lower = query.lower()
        return any(pattern in query_lower for pattern in injection_patterns)
        
    def _handle_security_alert(self, alert_type: str, context: Dict[str, Any]):
        """处理安全告警"""
        self.security_debugger.log_security_event(
            event_type=alert_type,
            severity="HIGH",
            description=f"检测到安全威胁: {alert_type}",
            context=context
        )
        
    def _handle_performance_alert(self, context: Dict[str, Any]):
        """处理性能告警"""
        self.log_debugger.log_debug_info(
            DebugLevel.WARNING,
            f"性能告警: 执行时间过长",
            function_name=context.get('function_name', ''),
            variables={'execution_time': context.get('execution_time', 0)}
        )
        
    # 扩展的调试方法
    def debug_network_request(self, method: str, url: str, **kwargs):
        """调试网络请求"""
        start_time = time.time()
        
        try:
            # 记录请求开始
            self.network_debugger.log_request(method, url, **kwargs)
            
            # 检查安全规则
            self.security_debugger.check_security_rules({
                'method': method,
                'url': url,
                'context': kwargs
            })
            
        except Exception as e:
            # 记录错误
            execution_time = time.time() - start_time
            self.network_debugger.log_request(
                method, url, error=str(e), response_time=execution_time
            )
            raise
            
    def debug_database_query(self, query: str, **kwargs):
        """调试数据库查询"""
        start_time = time.time()
        
        try:
            # 检查安全规则
            self.security_debugger.check_security_rules({
                'query': query,
                'context': kwargs
            })
            
            return query
            
        except Exception as e:
            # 记录错误
            execution_time = time.time() - start_time
            self.database_debugger.log_query(
                query, execution_time=execution_time, error=str(e), **kwargs
            )
            raise
        finally:
            # 记录成功查询
            execution_time = time.time() - start_time
            self.database_debugger.log_query(
                query, execution_time=execution_time, **kwargs
            )
            
    def take_memory_snapshot(self, description: str = ""):
        """拍摄内存快照"""
        return self.memory_debugger.take_memory_snapshot(description)
        
    def get_comprehensive_report(self) -> str:
        """生成综合调试报告"""
        debug_data = self._collect_debug_data()
        
        # 添加扩展数据
        debug_data.update({
            'network': self.network_debugger.get_request_statistics(),
            'database': self.database_debugger.get_query_statistics(),
            'memory': self.memory_debugger.get_memory_statistics(),
            'security': self.security_debugger.get_security_summary(),
            'advanced': {
                'debug_history_count': len(self.advanced_debugger.debug_history),
                'automation_rules': len(self.advanced_debugger.automation_rules),
                'custom_watchers': len(self.advanced_debugger.custom_watchers)
            }
        })
        
        return self.debug_reporter.generate_report('full', debug_data)
        
    def start_comprehensive_debug_session(self, session_name: str):
        """开始综合调试会话"""
        # 拍摄初始内存快照
        self.take_memory_snapshot(f"会话开始 - {session_name}")
        
        # 开始调试会话
        self.advanced_debugger.start_debug_session(session_name)
        
        # 开始性能监控
        self.performance_debugger.start_monitoring()
        
    def end_comprehensive_debug_session(self, session_name: str) -> Dict[str, Any]:
        """结束综合调试会话"""
        # 拍摄结束内存快照
        self.take_memory_snapshot(f"会话结束 - {session_name}")
        
        # 结束调试会话
        session_report = self.advanced_debugger.end_debug_session(session_name)
        
        # 生成综合报告
        comprehensive_report = self.get_comprehensive_report()
        
        # 检测内存泄漏
        memory_leaks = self.memory_debugger.detect_memory_leaks()
        
        # 检测安全异常
        security_anomalies = self.security_debugger.detect_anomalies()
        
        return {
            'session_report': session_report,
            'comprehensive_report': comprehensive_report,
            'memory_leaks': memory_leaks,
            'security_anomalies': security_anomalies,
            'network_stats': self.network_debugger.get_request_statistics(),
            'database_stats': self.database_debugger.get_query_statistics(),
            'memory_stats': self.memory_debugger.get_memory_statistics(),
            'security_stats': self.security_debugger.get_security_summary()
        }


# 工具函数和装饰器
def debug_async_function(debugger: ExtendedJ8DebuggingTools = None):
    """异步函数调试装饰器"""
    if debugger is None:
        debugger = get_debugger()
        
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            func_name = func.__name__
            file_path = inspect.getfile(func)
            
            async with debugger.async_debug_context(func_name, file_path):
                try:
                    # 记录异步函数调用
                    debugger.log_debugger.log_debug_info(
                        DebugLevel.DEBUG,
                        f"异步调用函数: {func_name}",
                        file_path=file_path,
                        function_name=func_name,
                        variables={'args': args, 'kwargs': kwargs}
                    )
                    
                    result = await func(*args, **kwargs)
                    
                    # 记录成功结果
                    debugger.log_debugger.log_debug_info(
                        DebugLevel.DEBUG,
                        f"异步函数执行成功: {func_name}",
                        file_path=file_path,
                        function_name=func_name,
                        variables={'result': str(result)[:100]}  # 限制结果长度
                    )
                    
                    return result
                    
                except Exception as e:
                    # 错误诊断
                    error_info = debugger.error_diagnostician.diagnose_error(
                        e, {'function_name': func_name, 'file_path': file_path}
                    )
                    
                    debugger.log_debugger.log_debug_info(
                        DebugLevel.ERROR,
                        f"异步函数执行失败: {func_name}",
                        file_path=file_path,
                        function_name=func_name,
                        variables={'error': str(e)}
                    )
                    
                    raise
                    
        return wrapper
    return decorator


def debug_class_methods(debugger: ExtendedJ8DebuggingTools = None):
    """类方法调试装饰器"""
    if debugger is None:
        debugger = get_debugger()
        
    def decorator(cls):
        for name, method in inspect.getmembers(cls, inspect.isfunction):
            if not name.startswith('_'):  # 不调试私有方法
                setattr(cls, name, debugger.debug_function(method))
        return cls
    return decorator


def performance_benchmark(iterations: int = 1000):
    """性能基准测试装饰器"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            
            for i in range(iterations):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                times.append(end_time - start_time)
                
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            print(f"性能基准测试结果 - {func.__name__}:")
            print(f"  平均时间: {avg_time:.6f}s")
            print(f"  最快时间: {min_time:.6f}s")
            print(f"  最慢时间: {max_time:.6f}s")
            print(f"  迭代次数: {iterations}")
            
            return result
        return wrapper
    return decorator


def memory_profiler(debugger: ExtendedJ8DebuggingTools = None):
    """内存分析装饰器"""
    if debugger is None:
        debugger = get_debugger()
        
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 拍摄内存快照
            debugger.take_memory_snapshot(f"函数开始 - {func.__name__}")
            
            # 执行函数
            result = func(*args, **kwargs)
            
            # 拍摄内存快照
            debugger.take_memory_snapshot(f"函数结束 - {func.__name__}")
            
            return result
        return wrapper
    return decorator


# 使用示例和测试代码
if __name__ == "__main__":
    # 创建扩展调试工具实例
    debugger = ExtendedJ8DebuggingTools()
    
    print("=== J8调试工具使用示例 ===\n")
    
    # 1. 基本函数调试
    @debugger.debug_function
    def example_calculation(x, y):
        debugger.log(DebugLevel.INFO, f"开始计算: {x} + {y}")
        result = x + y
        time.sleep(0.1)  # 模拟计算时间
        debugger.log(DebugLevel.DEBUG, f"计算结果: {result}")
        return result
    
    # 2. 异步函数调试
    @debug_async_function(debugger)
    async def async_example():
        debugger.log(DebugLevel.INFO, "异步函数开始执行")
        await asyncio.sleep(0.1)
        debugger.log(DebugLevel.INFO, "异步函数执行完成")
        return "async_result"
    
    # 3. 性能基准测试
    @performance_benchmark(100)
    def benchmark_function():
        return sum(range(1000))
    
    # 4. 内存分析
    @memory_profiler(debugger)
    def memory_intensive_function():
        data = [i for i in range(10000)]
        return len(data)
    
    # 5. 网络请求调试
    def debug_network_example():
        debugger.debug_network_request("GET", "https://api.example.com/data")
    
    # 6. 数据库查询调试
    def debug_database_example():
        query = "SELECT * FROM users WHERE id = ?"
        for result in debugger.debug_database_query(query, parameters={'id': 1}):
            pass
    
    # 7. 安全检测示例
    def security_detection_example():
        debugger.security_debugger.log_security_event(
            event_type="login_attempt",
            severity="MEDIUM",
            description="用户登录尝试",
            source="192.168.1.100"
        )
    
    # 执行示例
    print("1. 执行基本函数调试:")
    result = example_calculation(10, 20)
    print(f"   结果: {result}\n")
    
    print("2. 执行异步函数调试:")
    async_result = asyncio.run(async_example())
    print(f"   结果: {async_result}\n")
    
    print("3. 执行性能基准测试:")
    benchmark_result = benchmark_function()
    print(f"   结果: {benchmark_result}\n")
    
    print("4. 执行内存分析:")
    memory_result = memory_intensive_function()
    print(f"   结果: {memory_result}\n")
    
    print("5. 网络请求调试示例:")
    debug_network_example()
    
    print("6. 数据库查询调试示例:")
    debug_database_example()
    
    print("7. 安全检测示例:")
    security_detection_example()
    
    # 8. 综合调试会话
    print("8. 开始综合调试会话:")
    debugger.start_comprehensive_debug_session("example_session")
    
    # 模拟一些调试活动
    for i in range(5):
        example_calculation(i, i * 2)
    
    # 结束调试会话
    session_report = debugger.end_comprehensive_debug_session("example_session")
    print("   调试会话报告已生成\n")
    
    # 9. 生成调试报告
    print("9. 生成调试报告:")
    summary_report = debugger.generate_report("summary")
    print("   摘要报告:")
    print(summary_report[:500] + "..." if len(summary_report) > 500 else summary_report)
    
    comprehensive_report = debugger.get_comprehensive_report()
    print("\n   综合报告已生成")
    
    # 10. 获取统计信息
    print("\n10. 统计信息:")
    perf_stats = debugger.get_performance_summary()
    print(f"   性能统计: {perf_stats}")
    
    error_stats = debugger.get_error_statistics()
    print(f"   错误统计: {error_stats}")
    
    network_stats = debugger.network_debugger.get_request_statistics()
    print(f"   网络统计: {network_stats}")
    
    memory_stats = debugger.memory_debugger.get_memory_statistics()
    print(f"   内存统计: {memory_stats}")
    
    security_stats = debugger.security_debugger.get_security_summary()
    print(f"   安全统计: {security_stats}")
    
    # 11. 清理资源
    print("\n11. 清理调试资源:")
    debugger.cleanup()
    print("   清理完成")
    
    print("\n=== J8调试工具示例执行完成 ===")