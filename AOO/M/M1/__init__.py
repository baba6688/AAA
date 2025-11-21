#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1系统监控器模块
===============

M1系统监控器是一个全面的系统监控解决方案，提供以下功能：
1. 系统运行状态监控
2. 系统服务状态监控
3. 系统进程监控
4. 系统事件监控
5. 系统告警管理
6. 系统日志收集
7. 系统性能指标
8. 系统健康评估
9. 系统监控报告

主要组件:
- DatabaseManager: 数据库管理器，负责监控数据的存储和查询
- AlertManager: 告警管理器，处理系统告警的创建、确认和解决
- SystemMonitor: 系统监控器主类，提供完整的监控功能
- SystemMonitorTest: 系统监控器测试类，提供测试用例

版本: 1.0.0
创建时间: 2025-11-05
模块时间: 2025-11-13
"""

# 导入所有组件
try:
    # 尝试相对导入
    from .SystemMonitor import (
        # 枚举类
        AlertLevel,
        ServiceStatus,
        HealthStatus,
        
        # 数据类
        SystemMetrics,
        ProcessInfo,
        ServiceInfo,
        Alert,
        SystemEvent,
        
        # 主要类
        DatabaseManager,
        AlertManager,
        SystemMonitor,
        SystemMonitorTest,
    )
except ImportError:
    # 如果相对导入失败，使用绝对导入
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    
    from SystemMonitor import (
        # 枚举类
        AlertLevel,
        ServiceStatus,
        HealthStatus,
        
        # 数据类
        SystemMetrics,
        ProcessInfo,
        ServiceInfo,
        Alert,
        SystemEvent,
        
        # 主要类
        DatabaseManager,
        AlertManager,
        SystemMonitor,
        SystemMonitorTest,
    )

# 定义模块的公开接口
__all__ = [
    # 枚举类
    'AlertLevel',
    'ServiceStatus', 
    'HealthStatus',
    
    # 数据类
    'SystemMetrics',
    'ProcessInfo',
    'ServiceInfo',
    'Alert',
    'SystemEvent',
    
    # 主要类
    'DatabaseManager',
    'AlertManager',
    'SystemMonitor',
    'SystemMonitorTest',
    
    # 便捷函数
    'create_simple_monitor',
    'quick_system_check',
    'get_system_info',
]

# 模块版本信息
__version__ = "1.0.0"
__author__ = "M1 System Monitor Team"
__email__ = "system-monitor@m1.com"
__license__ = "MIT"
__copyright__ = "Copyright 2025 M1 System Monitor Team"

# 模块元数据
__title__ = "M1 System Monitor"
__description__ = "Comprehensive system monitoring solution for M1"
__url__ = "https://github.com/m1/system-monitor"
__version_info__ = (1, 0, 0)


def create_simple_monitor(monitor_interval: int = 60, 
                         db_path: str = "system_monitor.db") -> SystemMonitor:
    """
    创建简单配置的系统监控器
    
    Args:
        monitor_interval: 监控间隔（秒），默认60秒
        db_path: 数据库路径，默认"system_monitor.db"
    
    Returns:
        SystemMonitor: 配置好的系统监控器实例
    
    Example:
        >>> monitor = create_simple_monitor()
        >>> monitor.start_monitoring()
        >>> # 监控将在后台运行
    """
    config = {
        'monitor_interval': monitor_interval,
        'db_path': db_path,
        'cpu_warning_threshold': 70.0,
        'cpu_critical_threshold': 90.0,
        'memory_warning_threshold': 80.0,
        'memory_critical_threshold': 95.0,
        'disk_warning_threshold': 85.0,
        'disk_critical_threshold': 95.0,
        'log_level': 'INFO',
    }
    
    return SystemMonitor(config)


def quick_system_check() -> dict:
    """
    快速系统检查，返回当前系统状态摘要
    
    Returns:
        dict: 包含系统基本信息的字典
    
    Example:
        >>> status = quick_system_check()
        >>> print(f"CPU: {status['cpu_percent']:.1f}%")
        >>> print(f"内存: {status['memory_percent']:.1f}%")
    """
    monitor = create_simple_monitor()
    
    try:
        # 收集一次系统指标
        monitor._collect_system_metrics()
        
        if monitor.current_metrics:
            metrics = monitor.current_metrics
            health = monitor.get_system_health()
            
            return {
                'timestamp': metrics.timestamp.isoformat(),
                'cpu_percent': round(metrics.cpu_percent, 1),
                'memory_percent': round(metrics.memory_percent, 1),
                'memory_used_gb': round(metrics.memory_used / (1024**3), 2),
                'memory_total_gb': round(metrics.memory_total / (1024**3), 2),
                'disk_usage_percent': round(metrics.disk_usage, 1),
                'load_average': metrics.load_average,
                'uptime_hours': round(metrics.uptime / 3600, 1),
                'health_status': health.value,
                'status': 'healthy' if health == HealthStatus.HEALTHY else 'warning' if health == HealthStatus.WARNING else 'critical'
            }
        else:
            return {
                'status': 'error',
                'message': '无法获取系统指标'
            }
            
    except Exception as e:
        return {
            'status': 'error',
            'message': f'系统检查失败: {str(e)}'
        }
    finally:
        monitor.stop_monitoring()


def get_system_info() -> dict:
    """
    获取详细的系统信息
    
    Returns:
        dict: 包含系统详细信息的字典
    
    Example:
        >>> info = get_system_info()
        >>> print(f"系统健康状态: {info['health_status']}")
        >>> print(f"进程数量: {len(info['processes'])}")
    """
    monitor = create_simple_monitor()
    
    try:
        # 收集所有监控数据
        monitor._collect_system_metrics()
        monitor._monitor_processes()
        monitor._monitor_services()
        
        health = monitor.get_system_health()
        current_metrics = monitor.current_metrics
        
        if current_metrics:
            return {
                'system_metrics': {
                    'timestamp': current_metrics.timestamp.isoformat(),
                    'cpu_percent': round(current_metrics.cpu_percent, 1),
                    'memory_percent': round(current_metrics.memory_percent, 1),
                    'memory_used_gb': round(current_metrics.memory_used / (1024**3), 2),
                    'memory_total_gb': round(current_metrics.memory_total / (1024**3), 2),
                    'disk_usage_percent': round(current_metrics.disk_usage, 1),
                    'network_sent_mb': round(current_metrics.network_sent / (1024**2), 2),
                    'network_recv_mb': round(current_metrics.network_recv / (1024**2), 2),
                    'load_average': current_metrics.load_average,
                    'uptime_hours': round(current_metrics.uptime / 3600, 1),
                },
                'health_status': health.value,
                'processes': [
                    {
                        'pid': p.pid,
                        'name': p.name,
                        'status': p.status,
                        'cpu_percent': round(p.cpu_percent, 1),
                        'memory_percent': round(p.memory_percent, 1),
                        'memory_used_mb': round(p.memory_used / (1024**2), 2),
                        'num_threads': p.num_threads
                    }
                    for p in list(monitor.process_cache.values())[:10]  # 只显示前10个进程
                ],
                'services': [
                    {
                        'name': s.name,
                        'status': s.status.value,
                        'pid': s.pid,
                        'last_check': s.last_check.isoformat()
                    }
                    for s in monitor.service_cache.values()
                ],
                'alerts': [
                    {
                        'id': a.id,
                        'level': a.level.value,
                        'title': a.title,
                        'message': a.message,
                        'timestamp': a.timestamp.isoformat(),
                        'source': a.source,
                        'acknowledged': a.acknowledged,
                        'resolved': a.resolved
                    }
                    for a in monitor.get_active_alerts()[:5]  # 只显示前5个活跃告警
                ],
                'summary': {
                    'total_processes': len(monitor.process_cache),
                    'monitored_services': len(monitor.service_cache),
                    'active_alerts': len(monitor.get_active_alerts()),
                    'system_status': 'healthy' if health == HealthStatus.HEALTHY else 'warning' if health == HealthStatus.WARNING else 'critical'
                }
            }
        else:
            return {
                'error': '无法获取系统指标'
            }
            
    except Exception as e:
        return {
            'error': f'获取系统信息失败: {str(e)}'
        }
    finally:
        monitor.stop_monitoring()


# 模块初始化完成标志
_MODULE_INITIALIZED = True

# 导出便捷访问器
def get_version():
    """获取模块版本信息"""
    return __version__

def get_all_exports():
    """获取模块所有导出内容"""
    return __all__

def print_module_info():
    """打印模块信息"""
    print(f"{__title__} v{__version__}")
    print(f"{__description__}")
    print(f"作者: {__author__}")
    print(f"许可证: {__license__}")
    print(f"可用导出: {len(__all__)} 个组件")


# 在模块导入时显示基本信息（可选）
# print(f"已加载 {__title__} v{__version__}")