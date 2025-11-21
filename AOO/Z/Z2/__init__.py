"""
Z2 扩展接口系统

Z2扩展接口系统是一个功能完整的扩展接口管理框架，
提供了接口定义、扩展点管理、接口适配、接口验证、
接口文档生成、版本管理、统计分析和性能优化等功能。

主要模块:
- ExtensionInterface: 主要的扩展接口实现
- ExtensionPoint: 扩展点管理
- InterfaceAdapter: 接口适配器
- InterfaceValidator: 接口验证器
- InterfaceDocumenter: 接口文档生成器
- VersionManager: 版本管理器
- InterfaceStats: 统计分析
- InterfaceOptimizer: 性能优化器

作者: Z2开发团队
版本: 1.0.0
日期: 2025-11-06
"""

from .ExtensionInterface import (
    ExtensionInterface,
    ExtensionPoint,
    InterfaceAdapter,
    InterfaceValidator,
    InterfaceDocumenter,
    VersionManager,
    InterfaceStats,
    InterfaceOptimizer,
    InterfaceDefinition,
    ExtensionRegistry,
    InterfaceMetadata
)

__version__ = "1.0.0"
__author__ = "Z2开发团队"
__email__ = "dev@z2.com"

__all__ = [
    "ExtensionInterface",
    "ExtensionPoint", 
    "InterfaceAdapter",
    "InterfaceValidator",
    "InterfaceDocumenter",
    "VersionManager",
    "InterfaceStats",
    "InterfaceOptimizer",
    "InterfaceDefinition",
    "ExtensionRegistry",
    "InterfaceMetadata"
]