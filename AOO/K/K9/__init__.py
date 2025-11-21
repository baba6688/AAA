"""
K9配置状态聚合器模块

这个模块提供了完整的配置状态聚合和管理功能，包括：
- 配置状态监控和管理
- 配置协调和同步
- 配置生命周期管理
- 配置性能统计
- 配置健康检查
- 统一配置接口和API
- 异步配置状态同步
- 配置告警和通知系统

主要类：
- ConfigurationStateAggregator: 主要的配置状态聚合器类
- ConfigurationMonitor: 配置状态监控器
- ConfigurationCoordinator: 配置协调管理器
- ConfigurationLifecycleManager: 配置生命周期管理器
- ConfigurationPerformanceTracker: 配置性能统计器
- ConfigurationHealthChecker: 配置健康检查器
- ConfigurationAlertSystem: 配置告警和通知系统

作者: K9系统
版本: 1.0.0
"""

from .ConfigurationStateAggregator import (
    ConfigurationStateAggregator,
    ConfigurationMonitor,
    ConfigurationCoordinator,
    ConfigurationLifecycleManager,
    ConfigurationPerformanceTracker,
    ConfigurationHealthChecker,
    ConfigurationAlertSystem,
    ConfigurationAPI,
    ConfigurationEventHandler,
    DistributedConfigurationSync
)

__version__ = "1.0.0"
__author__ = "K9系统"
__email__ = "k9-system@example.com"

__all__ = [
    "ConfigurationStateAggregator",
    "ConfigurationMonitor", 
    "ConfigurationCoordinator",
    "ConfigurationLifecycleManager",
    "ConfigurationPerformanceTracker",
    "ConfigurationHealthChecker",
    "ConfigurationAlertSystem",
    "ConfigurationAPI",
    "ConfigurationEventHandler",
    "DistributedConfigurationSync"
]