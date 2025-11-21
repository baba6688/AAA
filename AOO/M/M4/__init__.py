#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M4模块 - 网络监控器 (NetworkMonitor)

提供全面的网络监控功能，包括连接监控、延迟监控、带宽监控、错误监控、
安全监控、拓扑发现、流量分析、性能优化和监控报告。

模块信息:
- 创建时间: 2025-11-05
- 版本: 1.0.0
- 作者: 系统开发者
- 许可: MIT License

主要功能:
- NetworkConnection: 网络连接数据管理
- NetworkMetrics: 网络性能指标收集
- SecurityAlert: 安全警报检测与处理
- NetworkTopology: 网络拓扑发现与映射
- TrafficAnalysis: 网络流量分析与统计
- NetworkMonitor: 核心监控器类

示例使用:
    from M4 import NetworkMonitor, NetworkConnection, SecurityAlert
    
    # 创建监控器实例
    monitor = NetworkMonitor()
    
    # 启动监控
    monitor.start_monitoring()
    
    # 获取网络连接
    connections = monitor.get_active_connections()
    
    # 获取安全警报
    alerts = monitor.get_security_alerts()

依赖项:
- psutil: 系统和进程监控
- asyncio: 异步编程支持
- threading: 多线程支持
- datetime: 时间处理
- ipaddress: IP地址处理
- statistics: 统计计算
- ssl: SSL/TLS支持
- requests: HTTP请求支持
- urllib.parse: URL解析
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "系统开发者"
__email__ = "dev@example.com"
__license__ = "MIT"
__copyright__ = "Copyright 2025"

# 模块路径
__module_path__ = "M4"

# 导入所有主要类和函数
from .NetworkMonitor import (
    NetworkConnection,
    NetworkMetrics,
    SecurityAlert,
    NetworkTopology,
    TrafficAnalysis,
    NetworkMonitor
)

# 定义公共API - 指定从模块中导出的所有公共符号
__all__ = [
    # 版本信息
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    "__copyright__",
    "__module_path__",
    
    # 主要数据类
    "NetworkConnection",
    "NetworkMetrics", 
    "SecurityAlert",
    "NetworkTopology",
    "TrafficAnalysis",
    
    # 主类
    "NetworkMonitor",
    
    # 便捷函数
    "create_monitor",
    "quick_network_check",
    "get_default_config",
    "validate_network_config",
    "format_network_metrics"
]

# 模块元数据
__all_metadata__ = {
    "version": __version__,
    "author": __author__,
    "description": "全面的网络监控解决方案",
    "keywords": ["网络监控", "网络安全", "流量分析", "拓扑发现", "性能监控"],
    "classifiers": [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: System Administrators",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Monitoring",
        "Topic :: Security"
    ]
}

# ============================================================================
# 便捷函数
# ============================================================================

def create_monitor(config: dict = None, auto_start: bool = False) -> NetworkMonitor:
    """
    创建并可选启动网络监控器的便捷函数
    
    Args:
        config: 配置字典，包含监控参数
        auto_start: 是否自动启动监控
        
    Returns:
        NetworkMonitor: 配置好的网络监控器实例
        
    Example:
        # 创建基本监控器
        monitor = create_monitor()
        
        # 创建带配置的监控器并启动
        monitor = create_monitor(
            config={"monitor_interval": 0.5},
            auto_start=True
        )
    """
    monitor = NetworkMonitor(config=config)
    if auto_start:
        monitor.start_monitoring()
    return monitor


def quick_network_check(target_host: str = "8.8.8.8", timeout: float = 5.0) -> dict:
    """
    快速网络连通性检查
    
    Args:
        target_host: 目标主机地址
        timeout: 超时时间（秒）
        
    Returns:
        dict: 包含检查结果的字典
        
    Example:
        result = quick_network_check("www.google.com")
        print(f"网络状态: {'正常' if result['reachable'] else '异常'}")
        print(f"延迟: {result['latency']}ms")
    """
    import socket
    import time
    
    result = {
        "target": target_host,
        "reachable": False,
        "latency": None,
        "error": None
    }
    
    try:
        # 测试连接
        start_time = time.time()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result_code = sock.connect_ex((target_host, 80))
        sock.close()
        end_time = time.time()
        
        if result_code == 0:
            result["reachable"] = True
            result["latency"] = round((end_time - start_time) * 1000, 2)
        else:
            result["error"] = f"连接失败，错误码: {result_code}"
            
    except Exception as e:
        result["error"] = str(e)
        
    return result


def get_default_config() -> dict:
    """
    获取默认的监控配置
    
    Returns:
        dict: 默认配置字典
        
    Example:
        config = get_default_config()
        print("默认监控间隔:", config["monitor_interval"])
    """
    return {
        "monitor_interval": 1.0,
        "connection_timeout": 5.0,
        "latency_threshold": 100.0,
        "bandwidth_threshold": 80.0,
        "packet_loss_threshold": 5.0,
        "error_threshold": 10,
        "security_scan_interval": 300,
        "topology_discovery_interval": 600,
        "traffic_analysis_interval": 60,
        "max_connections": 1000,
        "log_level": "INFO",
        "enable_notifications": True,
        "enable_reports": True
    }


def validate_network_config(config: dict) -> tuple[bool, list]:
    """
    验证网络监控配置
    
    Args:
        config: 待验证的配置字典
        
    Returns:
        tuple: (是否有效, 错误列表)
        
    Example:
        config = {"monitor_interval": 0.5}
        is_valid, errors = validate_network_config(config)
        if not is_valid:
            print("配置错误:", errors)
    """
    errors = []
    
    # 验证必需字段和类型
    numeric_fields = {
        "monitor_interval": (float, 0.1, 60.0),
        "connection_timeout": (float, 1.0, 300.0),
        "latency_threshold": (float, 1.0, 10000.0),
        "bandwidth_threshold": (float, 0.0, 100.0),
        "packet_loss_threshold": (float, 0.0, 100.0)
    }
    
    for field, (field_type, min_val, max_val) in numeric_fields.items():
        if field in config:
            if not isinstance(config[field], field_type):
                errors.append(f"字段 {field} 必须是 {field_type.__name__} 类型")
            elif not (min_val <= config[field] <= max_val):
                errors.append(f"字段 {field} 值必须在 {min_val} 到 {max_val} 之间")
    
    # 验证字符串字段
    string_fields = {
        "log_level": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        "enable_notifications": bool,
        "enable_reports": bool
    }
    
    for field, valid_values in string_fields.items():
        if field in config:
            if isinstance(valid_values, list):
                if config[field] not in valid_values:
                    errors.append(f"字段 {field} 必须是 {valid_values} 中的一个")
            elif not isinstance(config[field], valid_values):
                errors.append(f"字段 {field} 必须是 {valid_values.__name__} 类型")
    
    return len(errors) == 0, errors


def format_network_metrics(metrics: NetworkMetrics) -> str:
    """
    格式化网络指标为易读的字符串
    
    Args:
        metrics: 网络指标对象
        
    Returns:
        str: 格式化的指标字符串
        
    Example:
        # 获取指标后格式化显示
        metrics = monitor.get_current_metrics()
        print(format_network_metrics(metrics))
    """
    if not isinstance(metrics, NetworkMetrics):
        raise TypeError("metrics 必须是 NetworkMetrics 类型")
    
    io_info = []
    for interface, stats in metrics.network_io.items():
        io_info.append(f"  {interface}: 发送 {stats.get('bytes_sent', 0):,} 字节, 接收 {stats.get('bytes_recv', 0):,} 字节")
    
    io_info_str = "\n".join(io_info) if io_info else "  暂无网络IO数据"
    
    return f"""
网络性能指标报告
====================
时间戳: {metrics.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
CPU使用率: {metrics.cpu_percent:.2f}%
内存使用率: {metrics.memory_percent:.2f}%
网络连接数: {metrics.connections_count}
带宽使用率: {metrics.bandwidth_usage:.2f}%
网络延迟: {metrics.latency:.2f}ms
丢包率: {metrics.packet_loss:.2f}%
错误计数: {metrics.errors_count}

网络IO统计:
----------
{io_info_str}
"""

# ============================================================================
# 模块初始化
# ============================================================================

def _initialize_module():
    """模块初始化函数"""
    try:
        # 验证依赖项
        import psutil
        import socket
        import threading
        import datetime
        import ipaddress
        import statistics
        import ssl
        import requests
        
        # 设置模块就绪标志
        globals()["_module_ready"] = True
        
        # 记录初始化信息
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"M4模块初始化完成 - 版本 {__version__}")
        
    except ImportError as e:
        # 依赖项缺失时设置错误标志
        globals()["_module_ready"] = False
        globals()["_import_error"] = str(e)
        
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"M4模块依赖项缺失: {e}")

# 执行模块初始化
_initialize_module()

# 导出便捷属性
module_info = {
    "name": "M4",
    "version": __version__,
    "description": __doc__.split('\n')[0] if __doc__ else "",
    "classes": [name for name in __all__ if name[0].isupper() and name != "__version__"],
    "functions": [name for name in __all__ if callable(globals().get(name))],
    "ready": globals().get("_module_ready", False),
    "import_error": globals().get("_import_error")
}