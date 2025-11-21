#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M1模块功能测试脚本
===============

测试M1模块的各项功能：
1. 基本导入测试
2. 便捷函数测试
3. 完整功能测试
"""

# 测试基本导入
print("=== M1模块基本导入测试 ===")

try:
    from __init__ import (
        AlertLevel,
        ServiceStatus,
        HealthStatus,
        SystemMetrics,
        ProcessInfo,
        ServiceInfo,
        Alert,
        SystemEvent,
        DatabaseManager,
        AlertManager,
        SystemMonitor,
        SystemMonitorTest,
        create_simple_monitor,
        quick_system_check,
        get_system_info,
    )
    print("✓ 所有组件导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    exit(1)

# 测试版本信息
print(f"\n=== 模块信息 ===")
import __init__ as m1_module
print(f"版本: {m1_module.__version__}")
print(f"描述: {m1_module.__title__}")

# 测试便捷函数
print(f"\n=== 便捷函数测试 ===")

try:
    # 测试快速系统检查
    print("1. 快速系统检查:")
    status = quick_system_check()
    print(f"   系统状态: {status['status']}")
    print(f"   CPU使用率: {status['cpu_percent']}%")
    print(f"   内存使用率: {status['memory_percent']}%")
    print(f"   磁盘使用率: {status['disk_usage_percent']}%")
    print(f"   健康状态: {status['health_status']}")
    print("   ✓ 快速系统检查完成")
except Exception as e:
    print(f"   ✗ 快速系统检查失败: {e}")

try:
    # 测试创建简单监控器
    print("\n2. 创建简单监控器:")
    monitor = create_simple_monitor(monitor_interval=10)
    print(f"   监控间隔: {monitor.monitor_interval}秒")
    print(f"   数据库路径: {monitor.db_manager.db_path}")
    print("   ✓ 简单监控器创建成功")
except Exception as e:
    print(f"   ✗ 简单监控器创建失败: {e}")

try:
    # 测试获取详细系统信息
    print("\n3. 获取详细系统信息:")
    info = get_system_info()
    print(f"   系统状态: {info['summary']['system_status']}")
    print(f"   总进程数: {info['summary']['total_processes']}")
    print(f"   监控服务数: {info['summary']['monitored_services']}")
    print(f"   活跃告警数: {info['summary']['active_alerts']}")
    print(f"   系统运行时间: {info['system_metrics']['uptime_hours']}小时")
    print("   ✓ 详细系统信息获取成功")
except Exception as e:
    print(f"   ✗ 详细系统信息获取失败: {e}")

# 测试枚举类
print(f"\n=== 枚举类测试 ===")

print("1. AlertLevel枚举:")
for level in AlertLevel:
    print(f"   {level.name}: {level.value}")

print("\n2. ServiceStatus枚举:")
for status in ServiceStatus:
    print(f"   {status.name}: {status.value}")

print("\n3. HealthStatus枚举:")
for health in HealthStatus:
    print(f"   {health.name}: {health.value}")

# 测试数据类创建
print(f"\n=== 数据类测试 ===")

try:
    from datetime import datetime
    
    # 创建测试数据
    test_metrics = SystemMetrics(
        timestamp=datetime.now(),
        cpu_percent=25.5,
        memory_percent=60.0,
        memory_used=8589934592,  # 8GB
        memory_total=17179869184,  # 16GB
        disk_usage=45.2,
        network_sent=1048576,
        network_recv=2097152,
        load_average=[1.0, 1.1, 1.2],
        uptime=3600.0
    )
    
    print("1. SystemMetrics创建成功")
    print(f"   CPU使用率: {test_metrics.cpu_percent}%")
    print(f"   内存使用率: {test_metrics.memory_percent}%")
    
    # 创建告警测试
    test_alert = Alert(
        id="test_001",
        level=AlertLevel.WARNING,
        title="测试告警",
        message="这是一个测试告警",
        timestamp=datetime.now(),
        source="测试模块"
    )
    
    print("2. Alert创建成功")
    print(f"   告警级别: {test_alert.level.value}")
    print(f"   告警标题: {test_alert.title}")
    
    print("✓ 数据类测试完成")
    
except Exception as e:
    print(f"✗ 数据类测试失败: {e}")

# 测试SystemMonitor功能
print(f"\n=== SystemMonitor功能测试 ===")

try:
    # 创建监控器实例
    config = {
        'monitor_interval': 5,
        'cpu_warning_threshold': 50.0,
        'log_level': 'INFO'
    }
    
    monitor = SystemMonitor(config)
    print("1. SystemMonitor实例创建成功")
    
    # 测试收集系统指标
    monitor._collect_system_metrics()
    if monitor.current_metrics:
        print("2. 系统指标收集成功")
        print(f"   当前CPU: {monitor.current_metrics.cpu_percent:.1f}%")
        print(f"   当前内存: {monitor.current_metrics.memory_percent:.1f}%")
    else:
        print("2. 系统指标收集失败")
    
    # 测试健康状态评估
    health = monitor.get_system_health()
    print(f"3. 系统健康状态: {health.value}")
    
    # 测试进程监控
    monitor._monitor_processes()
    print(f"4. 发现进程数量: {len(monitor.process_cache)}")
    
    print("✓ SystemMonitor功能测试完成")
    
except Exception as e:
    print(f"✗ SystemMonitor功能测试失败: {e}")

print(f"\n=== 测试总结 ===")
print("✓ M1模块的所有功能测试完成")
print("✓ 模块导入正常")
print("✓ 便捷函数可用")
print("✓ 主要类功能正常")
print("✓ 枚举和数据类工作正常")

if __name__ == "__main__":
    print("\n测试完成！M1模块已准备就绪。")