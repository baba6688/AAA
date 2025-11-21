#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M8健康检查器包

该包提供了全面的系统健康监控和评估功能，包括：
- 系统健康检查 (CPU、内存、磁盘等)
- 服务健康检查 (端口连通性、HTTP响应等)
- 数据库健康检查 (连接状态、查询性能等)
- 网络健康检查 (连通性、延迟等)
- 组件健康检查 (模块状态、进程状态等)
- 健康报告生成和导出
- 实时告警通知
- 定时健康检查调度

主要类:
- HealthChecker: 主要的健康检查器类
- HealthStatus: 健康状态枚举
- HealthMetric: 健康指标数据类
- HealthCheckResult: 健康检查结果数据类
- HealthReport: 健康报告数据类

使用示例:
    from D.AO.AOO.M.M8.HealthChecker import HealthChecker
    
    checker = HealthChecker()
    report = checker.perform_comprehensive_health_check()
    print(f"整体状态: {report.overall_status.value}")


版本: 1.0.0
创建时间: 2025-11-05
"""

from .HealthChecker import (
    HealthChecker,
    HealthStatus,
    CheckType,
    HealthMetric,
    HealthCheckResult,
    HealthReport,
    AlertManager
)

__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "ai@example.com"
__license__ = "MIT"

__all__ = [
    "HealthChecker",
    "HealthStatus", 
    "CheckType",
    "HealthMetric",
    "HealthCheckResult",
    "HealthReport",
    "AlertManager"
]

# 包级别常量
DEFAULT_CONFIG = {
    "thresholds": {
        "cpu_warning": 70.0,
        "cpu_critical": 90.0,
        "memory_warning": 80.0,
        "memory_critical": 95.0,
        "disk_warning": 80.0,
        "disk_critical": 90.0,
        "network_latency_warning": 100.0,
        "network_latency_critical": 500.0,
        "service_response_time_warning": 5.0,
        "service_response_time_critical": 10.0,
        "db_connection_timeout_warning": 5.0,
        "db_connection_timeout_critical": 10.0
    },
    "check_intervals": {
        "system": 60,
        "service": 30,
        "database": 120,
        "network": 30,
        "component": 300
    },
    "reports": {
        "save_history": True,
        "max_history": 100,
        "export_formats": ["json", "text"]
    }
}

def create_health_checker(config=None):
    """
    创建健康检查器实例的便捷函数
    
    Args:
        config: 可选的配置字典，如果为None则使用默认配置
        
    Returns:
        HealthChecker: 健康检查器实例
    """
    return HealthChecker(config or DEFAULT_CONFIG)

def quick_health_check():
    """
    执行快速健康检查的便捷函数
    
    Returns:
        HealthReport: 快速健康检查报告
    """
    checker = create_health_checker()
    return checker.perform_comprehensive_health_check()

# 包初始化时的提示信息
def _init_message():
    """显示包初始化信息"""
    print(f"M8健康检查器 v{__version__} 已加载")
    print("使用 help(D.AO.AOO.M.M8.HealthChecker) 查看详细文档")

# 在包导入时显示信息（可选）
# _init_message()