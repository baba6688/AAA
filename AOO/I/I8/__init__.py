"""
I8模块 - 插件接口控制器

该模块提供插件接口控制功能，包括：
- 插件状态管理
- 插件信息处理
- 插件指标监控
- 插件基础接口
- 插件沙盒环境
- 依赖解析器
- 插件市场连接器
- 插件接口控制
- 测试和示例插件

Author: AI Assistant
Date: 2025-11-05
Version: 1.0.0
"""

from .PluginInterfaceController import (
    PluginStatus,
    PluginInfo,
    PluginMetrics,
    IPlugin,
    PluginBase,
    PluginSandbox,
    PluginDependencyResolver,
    PluginMarketConnector,
    PluginInterfaceController,
    TestPlugin,
    SamplePlugin
)

__version__ = "1.0.0"
__author__ = "AI Assistant"

__all__ = [
    "PluginStatus",
    "PluginInfo",
    "PluginMetrics",
    "IPlugin", 
    "PluginBase",
    "PluginSandbox",
    "PluginDependencyResolver",
    "PluginMarketConnector",
    "PluginInterfaceController",
    "TestPlugin",
    "SamplePlugin"
]