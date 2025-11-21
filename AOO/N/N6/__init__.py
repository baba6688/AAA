#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
N6威胁检测器包

一个完整的企业级威胁检测系统，提供恶意软件检测、网络攻击检测、内部威胁检测等核心功能。

主要模块:
- ThreatDetector: 核心威胁检测器类
- ThreatEvent: 威胁事件类
- ThreatReport: 威胁报告类
- MalwareDetector: 恶意软件检测器
- NetworkAttackDetector: 网络攻击检测器
- InsiderThreatDetector: 内部威胁检测器
- AnomalyDetector: 异常检测器

使用示例:
    from N6 import ThreatDetector, ThreatLevel, ThreatType
    
    detector = ThreatDetector()
    threat = detector.detect_threat("suspicious_file.exe")
    if threat.level >= ThreatLevel.HIGH:
        print(f"检测到威胁: {threat.description}")

版本: 1.0.0
日期: 2025-11-06
"""

from .ThreatDetector import (
    ThreatLevel,
    ThreatType,
    ResponseAction,
    ThreatIndicator,
    NetworkEvent,
    ThreatEvent,
    ThreatReport,
    MalwareDetector,
    NetworkAttackDetector,
    InsiderThreatDetector,
    AnomalyDetector,
    ThreatIntelligenceIntegrator,
    RealTimeMonitor,
    ThreatResponseEngine,
    ThreatClassifier,
    ThreatScorer,
    ThreatReporter,
    ThreatDetector
)

__version__ = "1.0.0"
__author__ = "MiniMax Agent"
__email__ = "security@example.com"
__license__ = "MIT"

# 包级别的公共接口
__all__ = [
    # 核心类
    "ThreatDetector",
    
    # 枚举类型
    "ThreatLevel",
    "ThreatType",
    "ResponseAction",
    
    # 数据结构
    "ThreatIndicator",
    "NetworkEvent",
    "ThreatEvent",
    "ThreatReport",
    
    # 检测器组件
    "MalwareDetector",
    "NetworkAttackDetector",
    "InsiderThreatDetector",
    "AnomalyDetector",
    "ThreatIntelligenceIntegrator",
    
    # 监控和响应
    "RealTimeMonitor",
    "ThreatResponseEngine",
    "ThreatClassifier",
    "ThreatScorer",
    "ThreatReporter",
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
def create_detector(config_file=None):
    """创建威胁检测器实例的便捷函数
    
    Args:
        config_file: 配置文件路径，可选
        
    Returns:
        ThreatDetector实例
    """
    return ThreatDetector(config_file)

def create_threat_event(threat_type, level, description, **kwargs):
    """创建威胁事件实例的便捷函数
    
    Args:
        threat_type: 威胁类型
        level: 威胁级别
        description: 威胁描述
        **kwargs: 其他事件属性
        
    Returns:
        ThreatEvent实例
    """
    return ThreatEvent(
        threat_type=threat_type,
        level=level,
        description=description,
        **kwargs
    )

def create_threat_indicator(name, pattern, source, **kwargs):
    """创建威胁指示器实例的便捷函数
    
    Args:
        name: 指示器名称
        pattern: 匹配模式
        source: 数据源
        **kwargs: 其他属性
        
    Returns:
        ThreatIndicator实例
    """
    return ThreatIndicator(
        name=name,
        pattern=pattern,
        source=source,
        **kwargs
    )

# 使用示例
EXAMPLE_USAGE = """
使用示例:

1. 恶意软件检测:
   from N6 import ThreatDetector, MalwareDetector, ThreatLevel
   
   detector = ThreatDetector()
   malware_result = detector.detect_malware("suspicious_file.exe")
   if malware_result.threat_level >= ThreatLevel.HIGH:
       print(f"检测到恶意软件: {malware_result.file_name}")

2. 网络攻击检测:
   from N6 import NetworkAttackDetector, NetworkEvent, ThreatType
   
   network_detector = NetworkAttackDetector()
   network_event = NetworkEvent(
       source_ip="192.168.1.100",
       dest_ip="192.168.1.200",
       protocol="TCP",
       port=80,
       payload_size=1500
   )
   attack_result = network_detector.analyze_event(network_event)
   if attack_result.threat_type == ThreatType.DOS:
       print("检测到拒绝服务攻击")

3. 内部威胁检测:
   from N6 import InsiderThreatDetector, ThreatLevel
   
   insider_detector = InsiderThreatDetector()
   insider_threat = insider_detector.detect_insider_threat(
       user_id="employee123",
       activity_pattern="unusual_access"
   )
   if insider_threat.level >= ThreatLevel.MEDIUM:
       print(f"检测到内部威胁: {insider_threat.description}")

4. 异常检测:
   from N6 import AnomalyDetector
   
   anomaly_detector = AnomalyDetector()
   anomalies = anomaly_detector.detect_anomalies("user_behavior_data.csv")
   for anomaly in anomalies:
       print(f"检测到异常: {anomaly.description}")

5. 威胁响应:
   from N6 import ThreatDetector, ResponseAction, ThreatType
   
   detector = ThreatDetector()
   threat_event = ThreatEvent(
       threat_type=ThreatType.MALWARE,
       level=ThreatLevel.CRITICAL,
       description="检测到恶意软件活动"
   )
   response = detector.respond_to_threat(threat_event)
   if response.action == ResponseAction.QUARANTINE:
       print("已隔离威胁源")
"""

if __name__ == "__main__":
    print("N6威胁检测器包")
    print(f"版本: {__version__}")
    print(f"作者: {__author__}")
    print(f"许可证: {__license__}")
    print("\n" + "="*50)
    print(EXAMPLE_USAGE)