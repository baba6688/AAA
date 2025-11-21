#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
W4网络监控器包 - 全面的网络监控解决方案

该包提供了完整的网络监控功能，包括：
- 网络流量监控
- 网络性能监控  
- 网络故障检测
- 网络拓扑监控
- 网络告警处理
- 网络报告生成

模块主要类：
- NetworkMetrics: 网络指标数据类，存储网络流量统计信息
- NetworkPerformance: 网络性能数据类，存储延迟、丢包率等性能指标
- NetworkAlert: 网络告警数据类，存储各种网络告警信息
- NetworkMonitor: 网络监控器主类，提供完整的监控功能

使用示例：
    from W4 import NetworkMonitor, NetworkMetrics, NetworkPerformance, NetworkAlert
    
    # 创建监控器实例
    monitor = NetworkMonitor()
    
    # 启动监控
    monitor.start_monitoring()
    
    # 获取监控数据
    traffic_stats = monitor.get_network_traffic_stats()
    performance_stats = monitor.get_network_performance_stats()
    alerts = monitor.get_network_alerts()

版本要求：
    Python >= 3.6
    psutil >= 5.7.0
    ping3 >= 2.6.0
    requests >= 2.25.0
"""

from .NetworkMonitor import NetworkMonitor, NetworkMetrics, NetworkPerformance, NetworkAlert

__version__ = "1.0.0"
__author__ = "W4 Network Monitor Team"
__description__ = "W4网络监控器 - 全面的网络监控解决方案"
__email__ = "network-monitor@team.local"
__license__ = "MIT License"
__maintainer__ = "W4 Network Monitor Team"

# 包级别的配置常量
DEFAULT_CONFIG_FILE = "network_monitor_config.json"
DEFAULT_LOG_DIR = "logs"
DEFAULT_REPORT_DIR = "reports"
DEFAULT_HISTORY_SIZE = 1000
DEFAULT_MONITORING_INTERVAL = 5

# 告警严重级别常量
ALERT_SEVERITY_INFO = "info"
ALERT_SEVERITY_WARNING = "warning"
ALERT_SEVERITY_CRITICAL = "critical"

# 数据类型常量  
DATA_TYPE_METRICS = "metrics"
DATA_TYPE_PERFORMANCE = "performance"
DATA_TYPE_ALERTS = "alerts"
DATA_TYPE_ALL = "all"

__all__ = [
    'NetworkMonitor',
    'NetworkMetrics', 
    'NetworkPerformance',
    'NetworkAlert',
    # 常量
    'DEFAULT_CONFIG_FILE',
    'DEFAULT_LOG_DIR', 
    'DEFAULT_REPORT_DIR',
    'DEFAULT_HISTORY_SIZE',
    'DEFAULT_MONITORING_INTERVAL',
    'ALERT_SEVERITY_INFO',
    'ALERT_SEVERITY_WARNING', 
    'ALERT_SEVERITY_CRITICAL',
    'DATA_TYPE_METRICS',
    'DATA_TYPE_PERFORMANCE',
    'DATA_TYPE_ALERTS',
    'DATA_TYPE_ALL',
    # 版本信息
    '__version__',
    '__author__',
    '__description__',
    '__email__',
    '__license__',
    '__maintainer__'
]