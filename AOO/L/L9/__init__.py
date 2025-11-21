#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L9日志状态聚合器模块

该模块提供了一个全面的日志状态聚合和管理系统。

依赖库检查:
- aiofiles: 异步文件操作 (可选)
- aiohttp: HTTP客户端 (可选)
- redis: Redis客户端 (可选)
- psutil: 系统资源监控 (可选)
- cryptography: 加密功能 (可选)

如果某些依赖不可用，相关功能将自动降级或禁用。
"""

# 可选依赖检查
try:
    from .LoggingStateAggregator import (
        # 主要类
        LoggingStateAggregator,
        
        # 数据结构
        LogEntry,
        LogStatistics,
        HealthCheckResult,
        Alert,
        
        # 枚举
        LogLevel,
        LogStatus,
        HealthStatus,
        AlertLevel,
        AggregationType,
        
        # 异常
        LoggingStateAggregatorError,
        LogProcessingError,
        LogStorageError,
        LogTransmissionError,
        LogHealthCheckError,
        LogAggregationError,
        
        # 组件类
        LogStateMonitor,
        LogLifecycleManager,
        LogStatisticsManager,
        LogHealthChecker,
        LogAlertSystem,
        LogCoordinator,
        LogStorageManager,
        LogTransmissionManager,
    )
    
    # 依赖可用性检查
    import sys
    
    def check_dependencies():
        """检查可选依赖的可用性"""
        dependencies = {
            'aiofiles': '异步文件操作',
            'aiohttp': 'HTTP客户端',
            'redis': 'Redis客户端',
            'psutil': '系统资源监控',
            'cryptography': '加密功能'
        }
        
        available = []
        missing = []
        
        for dep_name, description in dependencies.items():
            try:
                if dep_name == 'aiofiles':
                    import aiofiles
                elif dep_name == 'aiohttp':
                    import aiohttp
                elif dep_name == 'redis':
                    import redis
                elif dep_name == 'psutil':
                    import psutil
                elif dep_name == 'cryptography':
                    import cryptography
                
                available.append((dep_name, description))
            except ImportError:
                missing.append((dep_name, description))
        
        return available, missing
    
    # 初始化时检查依赖
    _AVAILABLE_DEPS, _MISSING_DEPS = check_dependencies()
    
    if _MISSING_DEPS:
        import logging
        logger = logging.getLogger(__name__)
        missing_str = ', '.join([f"{name} ({desc})" for name, desc in _MISSING_DEPS])
        logger.info(f"L9模块部分功能将不可用，缺少依赖: {missing_str}")

except ImportError as e:
    raise ImportError(f"L9模块加载失败: {e}. 请确保所有必需的Python包已安装。") from e

__version__ = "1.0.0"
__author__ = "L9系统"

# 导出所有公共接口
__all__ = [
    "LoggingStateAggregator",
    "LogEntry",
    "LogStatistics", 
    "HealthCheckResult",
    "Alert",
    "LogLevel",
    "LogStatus",
    "HealthStatus",
    "AlertLevel",
    "AggregationType",
    "LoggingStateAggregatorError",
    "LogProcessingError",
    "LogStorageError",
    "LogTransmissionError",
    "LogHealthCheckError",
    "LogAggregationError",
    "LogStateMonitor",
    "LogLifecycleManager",
    "LogStatisticsManager",
    "LogHealthChecker",
    "LogAlertSystem",
    "LogCoordinator",
    "LogStorageManager",
    "LogTransmissionManager",
    "check_dependencies"
]