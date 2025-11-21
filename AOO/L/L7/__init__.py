#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L7调试日志记录器模块

这是一个全面的L7层调试日志记录系统，提供代码调试、性能分析、内存泄漏检测、
网络调试、数据库调试、调试工具集成等多种调试功能。

主要导出:
- DebugLogger: 主调试日志记录器类
- 异常类: DebugLoggerError, MemoryLeakError, PerformanceThresholdError等
- 常量类: LogLevel, DebugCategory
- 工具函数: debug_logger装饰器, performance_profile装饰器
- 便捷函数: debug(), info(), warning(), error(), critical()

"""

# 导入主要类和常量
from .DebugLogger import (
    # 主要类
    DebugLogger,
    
    # 异常类
    DebugLoggerError,
    MemoryLeakError,
    PerformanceThresholdError,
    NetworkDebugError,
    DatabaseDebugError,
    
    # 常量类
    LogLevel,
    DebugCategory,
    
    # 装饰器
    debug_logger,
    performance_profile,
    
    # 便捷函数
    get_default_logger,
    debug,
    info,
    warning,
    error,
    critical,
    
    # 便捷方法示例
    AsyncLogHandler,
    FileLogHandler,
    ConsoleLogHandler,
    DatabaseLogHandler,
    MemoryMonitor,
    PerformanceProfiler,
    NetworkDebugger,
    DatabaseDebugger,
    DebugToolIntegrator
)

# 模块元信息
__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "L7调试日志记录器 - 全面的L7层调试日志记录系统"

# 公开API列表
__all__ = [
    # 主要类
    "DebugLogger",
    
    # 异常类
    "DebugLoggerError",
    "MemoryLeakError", 
    "PerformanceThresholdError",
    "NetworkDebugError",
    "DatabaseDebugError",
    
    # 常量类
    "LogLevel",
    "DebugCategory",
    
    # 装饰器
    "debug_logger",
    "performance_profile",
    
    # 便捷函数
    "get_default_logger",
    "debug",
    "info",
    "warning",
    "error",
    "critical",
    
    # 基础类
    "BaseLogHandler",
    
    # 数据结构类
    "LogEntry",
    "MemorySnapshot", 
    "PerformanceMetrics",
    
    # 工具类
    "AsyncLogHandler",
    "FileLogHandler", 
    "ConsoleLogHandler",
    "DatabaseLogHandler",
    "MemoryMonitor",
    "PerformanceProfiler",
    "NetworkDebugger",
    "DatabaseDebugger",
    "DebugToolIntegrator",
    
    # 模块信息
    "__version__",
    "__author__",
    "__description__"
]