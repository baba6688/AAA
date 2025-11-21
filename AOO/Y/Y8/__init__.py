"""
Y8存储清理器包

这是一个功能完整的存储清理器，支持自动清理、手动清理、
多种清理策略、清理统计、监控等功能。

主要组件：
- StorageCleaner: 主要的存储清理器类
- CleanStrategy: 清理策略基类
- CleanMonitor: 清理监控器
- CleanConfig: 清理配置管理
- CleanReporter: 清理报告生成器
"""

from .StorageCleaner import (
    StorageCleaner,
    CleanStrategy,
    CleanMonitor,
    CleanConfig,
    CleanReporter,
    FileCleaner,
    CacheCleaner,
    TempCleaner,
    LogCleaner,
    AutoCleaner,
    ManualCleaner
)

__version__ = "1.0.0"
__author__ = "Y8存储清理器开发团队"
__description__ = "Y8存储清理器 - 功能完整的存储清理解决方案"

__all__ = [
    "StorageCleaner",
    "CleanStrategy", 
    "CleanMonitor",
    "CleanConfig",
    "CleanReporter",
    "FileCleaner",
    "CacheCleaner", 
    "TempCleaner",
    "LogCleaner",
    "AutoCleaner",
    "ManualCleaner"
]