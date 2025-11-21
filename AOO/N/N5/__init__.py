#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N5安全审计器包

一个完整的企业级安全审计系统，提供安全审计、合规检查、威胁评估等核心功能。

主要模块:
- SecurityAuditor: 核心安全审计器类
- SecurityEvent: 安全事件类
- SecurityMetric: 安全指标类
- ComplianceCheck: 合规性检查类
- ThreatLevel: 威胁级别枚举
- ComplianceStatus: 合规状态枚举

使用示例:
    from N5 import SecurityAuditor, SecurityEvent, ThreatLevel
    
    auditor = SecurityAuditor()
    event = SecurityEvent(
        event_type=SecurityEventType.LOGIN,
        severity=ThreatLevel.MEDIUM,
        description="用户登录事件"
    )
    auditor.record_event(event)

版本: 1.0.0
日期: 2025-11-06
"""

from .SecurityAuditor import (
    SecurityEventType,
    ThreatLevel,
    ComplianceStatus,
    SecurityEvent,
    SecurityMetric,
    ComplianceCheck,
    SecurityAuditor
)

__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__email__ = "security@example.com"
__license__ = "MIT"

# 包级别的公共接口
__all__ = [
    # 核心类
    "SecurityAuditor",
    
    # 枚举类型
    "SecurityEventType",
    "ThreatLevel",
    "ComplianceStatus",
    
    # 数据结构
    "SecurityEvent",
    "SecurityMetric",
    "ComplianceCheck",
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
def create_auditor(config_file=None):
    """创建安全审计器实例的便捷函数
    
    Args:
        config_file: 配置文件路径，可选
        
    Returns:
        SecurityAuditor实例
    """
    return SecurityAuditor(config_file)

def create_security_event(event_type, description, **kwargs):
    """创建安全事件实例的便捷函数
    
    Args:
        event_type: 事件类型
        description: 事件描述
        **kwargs: 其他事件属性
        
    Returns:
        SecurityEvent实例
    """
    return SecurityEvent(
        event_type=event_type,
        description=description,
        **kwargs
    )

def create_compliance_check(standard, checklist=None):
    """创建合规性检查实例的便捷函数
    
    Args:
        standard: 合规性标准
        checklist: 检查清单，可选
        
    Returns:
        ComplianceCheck实例
    """
    return ComplianceCheck(
        standard=standard,
        checklist=checklist or []
    )

# 使用示例
EXAMPLE_USAGE = """
使用示例:

1. 记录安全事件:
   from N5 import SecurityAuditor, SecurityEvent, SecurityEventType, ThreatLevel
   
   auditor = SecurityAuditor()
   event = SecurityEvent(
       event_type=SecurityEventType.LOGIN,
       severity=ThreatLevel.LOW,
       description="用户正常登录",
       source_ip="192.168.1.100"
   )
   auditor.record_event(event)

2. 合规性检查:
   from N5 import SecurityAuditor, ComplianceStatus
   
   auditor = SecurityAuditor()
   compliance = auditor.check_compliance("GDPR", "data_protection")
   if compliance.status == ComplianceStatus.COMPLIANT:
       print("符合GDPR要求")

3. 生成安全报告:
   from N5 import SecurityAuditor, SecurityEventType
   
   auditor = SecurityAuditor()
   report = auditor.generate_report(
       start_date=datetime.now() - timedelta(days=30),
       end_date=datetime.now()
   )
   print(f"报告生成完成: {report['summary']}")

4. 威胁评估:
   from N5 import SecurityAuditor, ThreatLevel
   
   auditor = SecurityAuditor()
   threat_assessment = auditor.assess_threat("unusual_activity")
   if threat_assessment.level >= ThreatLevel.MEDIUM:
       print("检测到潜在威胁")

5. 安全指标监控:
   from N5 import SecurityAuditor
   
   auditor = SecurityAuditor()
   metric = SecurityMetric(
       name="failed_logins",
       value=5,
       threshold=10,
       timestamp=datetime.now()
   )
   auditor.record_metric(metric)
   
   alert = auditor.check_thresholds()
   if alert:
       print(f"安全告警: {alert.message}")
"""

if __name__ == "__main__":
    print("N5安全审计器包")
    print(f"版本: {__version__}")
    print(f"作者: {__author__}")
    print(f"许可证: {__license__}")
    print("\n" + "="*50)
    print(EXAMPLE_USAGE)