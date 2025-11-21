#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J9工具状态聚合器模块

该模块提供了一个完整的工具状态聚合和管理系统，用于监控、管理和协调多个工具实例。

主要组件:
- ToolStateAggregator: 主要的聚合器类
- ToolInfo: 工具信息数据类
- ToolDependency: 工具依赖数据类
- BaseTool: 工具基类
- MockTool: 模拟工具实现
- DatabaseTool: 数据库工具实现
- CacheTool: 缓存工具实现
- APIGatewayTool: API网关工具实现

异常类:
- ToolError: 基础工具异常
- ToolNotFoundError: 工具未找到异常
- ToolStateError: 工具状态异常
- ToolDependencyError: 工具依赖异常
- ToolHealthError: 工具健康检查异常

枚举类:
- ToolState: 工具状态
- HealthStatus: 健康状态
- Priority: 优先级
- ResourceType: 资源类型

数据类:
- ResourceUsage: 资源使用情况
- PerformanceMetrics: 性能指标

作者: J9系统
版本: 1.0.1
创建时间: 2025-11-13
"""

from .ToolStateAggregator import (
    # 主要类
    ToolStateAggregator,
    
    # 数据类
    ToolInfo,
    ToolDependency,
    ResourceUsage,
    PerformanceMetrics,
    
    # 工具基类和实现
    BaseTool,
    MockTool,
    DatabaseTool,
    CacheTool,
    APIGatewayTool,
    
    # 工具工厂和注册
    ToolFactory,
    
    # 工具注册表和依赖管理器
    ToolRegistry,
    DependencyManager,
    
    # 监控组件
    HealthChecker,
    PerformanceMonitor,
    ResourceManager,
    CommunicationManager,
    
    # 异常类
    ToolError,
    ToolNotFoundError,
    ToolStateError,
    ToolDependencyError,
    ToolHealthError,
    
    # 枚举类
    ToolState,
    HealthStatus,
    Priority,
    ResourceType,
    
    # 工具和验证
    PerformanceTester,
    ConfigValidator,
    ToolMonitorPanel,
    
    # 高级功能
    LoadBalancer,
    FailoverManager,
    AlertManager,
    Scheduler,
    PluginManager,
    CacheManager,
    
    # 扩展聚合器
    AdvancedToolStateAggregator
)

# 版本信息
__version__ = "1.0.1"
__author__ = "J9系统"
__description__ = "J9工具状态聚合和管理系统"

# 导出的公共接口
__all__ = [
    # 主要类
    'ToolStateAggregator',
    'AdvancedToolStateAggregator',
    
    # 数据类
    'ToolInfo',
    'ToolDependency',
    'ResourceUsage',
    'PerformanceMetrics',
    
    # 工具类
    'BaseTool',
    'MockTool',
    'DatabaseTool',
    'CacheTool',
    'APIGatewayTool',
    
    # 管理器类
    'ToolRegistry',
    'DependencyManager',
    'HealthChecker',
    'PerformanceMonitor',
    'ResourceManager',
    'CommunicationManager',
    
    # 高级功能管理器
    'LoadBalancer',
    'FailoverManager',
    'AlertManager',
    'Scheduler',
    'PluginManager',
    'CacheManager',
    
    # 工厂和工具类
    'ToolFactory',
    'PerformanceTester',
    'ConfigValidator',
    'ToolMonitorPanel',
    
    # 异常类
    'ToolError',
    'ToolNotFoundError',
    'ToolStateError',
    'ToolDependencyError',
    'ToolHealthError',
    
    # 枚举类
    'ToolState',
    'HealthStatus',
    'Priority',
    'ResourceType',
    
    # 版本信息
    '__version__',
    '__author__',
    '__description__'
]

def get_version():
    """获取版本信息"""
    return __version__

def get_system_info():
    """获取系统信息"""
    import platform
    import sys
    
    return {
        'version': __version__,
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture(),
        'processor': platform.processor()
    }

# 确保所有导出的名称在模块级别可用
__all__ = __all__