#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N9安全状态聚合器包

一个完整的企业级安全监控系统，提供安全状态收集、聚合、分析等核心功能。

主要模块:
- SecurityStateAggregator: 核心安全状态聚合器类
- SecurityEvent: 安全事件类
- SecurityThreat: 安全威胁类
- SecurityAlert: 安全告警类
- SecurityReport: 安全报告类
- SecurityMetric: 安全指标类
- SecurityModuleInterface: 安全模块接口类

使用示例:
    from N9 import SecurityStateAggregator, EventType, AlertSeverity
    
    aggregator = SecurityStateAggregator()
    threat = SecurityThreat(
        threat_type="malware",
        severity=AlertSeverity.HIGH,
        description="检测到恶意软件活动"
    )
    aggregator.process_threat(threat)

版本: 1.0.0
日期: 2025-11-06
"""

from .SecurityStateAggregator import (
    SecurityLevel,
    AlertSeverity,
    EventType,
    SecurityMetric,
    SecurityThreat,
    SecurityEvent,
    SecurityAlert,
    SecurityReport,
    SecurityModuleInterface,
    MockSecurityModule,
    SecurityStateAggregator
)

__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__email__ = "security@example.com"
__license__ = "MIT"

# 包级别的公共接口
__all__ = [
    # 核心类
    "SecurityStateAggregator",
    
    # 枚举类型
    "SecurityLevel",
    "AlertSeverity",
    "EventType",
    
    # 数据结构
    "SecurityMetric",
    "SecurityThreat",
    "SecurityEvent",
    "SecurityAlert",
    "SecurityReport",
    
    # 接口类
    "SecurityModuleInterface",
    "MockSecurityModule",
]

# 包初始化信息
def get_version():
    """获取版本信息"""
    return __version__

def get_author():
    """获取作者信息"""
    return __author__

def get_license():
    """获取许可证信息"""
    return __license__

# 便捷函数
def create_aggregator(config_file=None):
    """创建安全状态聚合器实例的便捷函数
    
    Args:
        config_file: 配置文件路径，可选
        
    Returns:
        SecurityStateAggregator实例
    """
    return SecurityStateAggregator(config_file)

def create_security_threat(threat_type, severity, description, **kwargs):
    """创建安全威胁实例的便捷函数
    
    Args:
        threat_type: 威胁类型
        severity: 严重级别
        description: 威胁描述
        **kwargs: 其他威胁属性
        
    Returns:
        SecurityThreat实例
    """
    return SecurityThreat(
        threat_type=threat_type,
        severity=severity,
        description=description,
        **kwargs
    )

def create_security_alert(alert_type, severity, message, **kwargs):
    """创建安全告警实例的便捷函数
    
    Args:
        alert_type: 告警类型
        severity: 严重级别
        message: 告警消息
        **kwargs: 其他告警属性
        
    Returns:
        SecurityAlert实例
    """
    return SecurityAlert(
        alert_type=alert_type,
        severity=severity,
        message=message,
        **kwargs
    )

def create_security_metric(name, value, metric_type, **kwargs):
    """创建安全指标实例的便捷函数
    
    Args:
        name: 指标名称
        value: 指标值
        metric_type: 指标类型
        **kwargs: 其他指标属性
        
    Returns:
        SecurityMetric实例
    """
    return SecurityMetric(
        name=name,
        value=value,
        metric_type=metric_type,
        **kwargs
    )

# 使用示例
EXAMPLE_USAGE = """
使用示例:

1. 安全状态监控:
   from N9 import SecurityStateAggregator, SecurityEvent, EventType
   
   aggregator = SecurityStateAggregator()
   event = SecurityEvent(
       event_type=EventType.LOGIN_ATTEMPT,
       source_ip="192.168.1.100",
       user_id="user123",
       timestamp=datetime.now()
   )
   aggregator.process_event(event)

2. 威胁聚合处理:
   from N9 import SecurityStateAggregator, SecurityThreat, AlertSeverity
   
   aggregator = SecurityStateAggregator()
   threat = SecurityThreat(
       threat_type="malware",
       severity=AlertSeverity.HIGH,
       source="N6_threat_detector",
       description="检测到恶意软件文件"
   )
   aggregator.process_threat(threat)

3. 安全指标收集:
   from N9 import SecurityStateAggregator, SecurityMetric
   
   aggregator = SecurityStateAggregator()
   metric = SecurityMetric(
       name="failed_login_attempts",
       value=15,
       metric_type="count",
       threshold=10,
       timestamp=datetime.now()
   )
   aggregator.record_metric(metric)

4. 安全告警生成:
   from N9 import SecurityStateAggregator, SecurityAlert, AlertSeverity
   
   aggregator = SecurityStateAggregator()
   alert = SecurityAlert(
       alert_type="threshold_exceeded",
       severity=AlertSeverity.MEDIUM,
       message="连续登录失败次数超过阈值",
       source="login_monitor"
   )
   aggregator.generate_alert(alert)

5. 安全报告生成:
   from N9 import SecurityStateAggregator
   
   aggregator = SecurityStateAggregator()
   await aggregator.start_collection()
   
   # 生成报告
   report = aggregator.generate_report(
       start_time=datetime.now() - timedelta(days=7),
       end_time=datetime.now()
   )
   print(f"安全报告生成完成:")
   print(f"- 总事件数: {report['total_events']}")
   print(f"- 高危威胁数: {report['high_severity_threats']}")
   print(f"- 平均安全级别: {report['average_security_level']}")

6. 模块集成:
   from N9 import SecurityStateAggregator, MockSecurityModule
   
   aggregator = SecurityStateAggregator()
   mock_module = MockSecurityModule("test_module")
   aggregator.register_module(mock_module)
   
   # 获取模块状态
   module_status = aggregator.get_module_status("test_module")
   print(f"模块状态: {module_status.status}")
"""

if __name__ == "__main__":
    print("N9安全状态聚合器包")
    print(f"版本: {__version__}")
    print(f"作者: {__author__}")
    print(f"许可证: {__license__}")
    print("\n" + "="*50)
    print(EXAMPLE_USAGE)