#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R7灾难恢复器 - 包初始化文件

提供全面的灾难恢复功能，包括：
- 灾难检测和监控
- 自动恢复流程
- 应急响应管理
- 备用系统切换
- 数据同步
- 实时监控告警
- 恢复验证
- 报告生成
- 灾难演练
- 业务连续性规划

作者: R7灾难恢复器团队
版本: 1.0.0
"""

# 核心类
from .DisasterRecovery import (
    # 主要类
    DisasterRecovery,
    
    # 检测系统
    DisasterDetector,
    
    # 恢复管理
    RecoveryManager,
    
    # 应急响应
    EmergencyResponse,
    
    # 备份系统
    BackupSystem,
    
    # 数据同步
    DataSynchronizer,
    
    # 监控告警
    MonitoringAlert,
    
    # 验证系统
    RecoveryValidator,
    
    # 报告生成
    ReportGenerator,
    
    # 数据结构
    DisasterEvent,
    RecoveryStep,
    
    # 枚举类型
    DisasterType,
    RecoveryStatus,
    
    # 便利函数
    detect_disaster,
    trigger_recovery,
    validate_recovery,
    monitor_system_health,
    generate_recovery_report,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "R7灾难恢复器团队"
__email__ = "support@r7disaster.com"
__description__ = "R7灾难恢复器 - 全面的灾难恢复和业务连续性解决方案"

# 包级别的默认配置
DEFAULT_CONFIG = {
    'dr_dir': './disaster_recovery',
    'backup_dir': './dr_backups',
    'log_dir': './dr_logs',
    'monitoring': {
        'real_time_monitoring': True,
        'health_check_interval': 60,
        'threat_detection': True,
        'alert_threshold': 'MEDIUM'
    },
    'recovery': {
        'auto_recovery': False,
        'recovery_timeout': 7200,  # 2小时
        'validation_required': True,
        'rollback_on_failure': True
    },
    'backup': {
        'backup_frequency': 'daily',
        'retention_period': 365,  # 天
        'verification': True,
        'compression': 'gzip'
    },
    'communication': {
        'alert_channels': ['email', 'sms', 'webhook'],
        'escalation_levels': 3,
        'response_time_target': 300  # 5分钟
    },
    'compliance': {
        'rto_target': 3600,  # 1小时恢复时间目标
        'rpo_target': 300,   # 5分钟恢复点目标
        'audit_logging': True,
        'compliance_reports': True
    }
}

# 灾难类型
DISASTER_TYPES = [
    'HARDWARE_FAILURE',    # 硬件故障
    'SOFTWARE_CORRUPTION', # 软件损坏
    'NETWORK_OUTAGE',      # 网络中断
    'POWER_FAILURE',       # 电源故障
    'NATURAL_DISASTER',    # 自然灾害
    'CYBER_ATTACK',        # 网络攻击
    'HUMAN_ERROR',         # 人为错误
    'VIRUS_INFECTION',     # 病毒感染
    'FIRE',                # 火灾
    'FLOOD',               # 洪水
    'EARTHQUAKE',          # 地震
    'PANDEMIC',            # 疫情
]

# 恢复优先级
RECOVERY_PRIORITY = [
    'CRITICAL',        # 关键业务
    'HIGH',            # 高优先级
    'MEDIUM',          # 中优先级
    'LOW',             # 低优先级
]

# 监控级别
MONITORING_LEVELS = [
    'BASIC',           # 基础监控
    'STANDARD',        # 标准监控
    'ADVANCED',        # 高级监控
    'COMPREHENSIVE',   # 全面监控
]

# 告警级别
ALERT_LEVELS = [
    'INFO',            # 信息
    'WARNING',         # 警告
    'MEDIUM',          # 中等
    'HIGH',            # 高
    'CRITICAL',        # 严重
]

# 恢复状态
RECOVERY_STATUS = [
    'DETECTED',        # 已检测
    'INVESTIGATING',   # 调查中
    'PLANNING',        # 计划中
    'EXECUTING',       # 执行中
    'VALIDATING',      # 验证中
    'COMPLETED',       # 已完成
    'FAILED',          # 失败
]

# 系统健康状态
HEALTH_STATUS = [
    'HEALTHY',         # 健康
    'DEGRADED',        # 降级
    'UNHEALTHY',       # 不健康
    'CRITICAL',        # 严重
    'UNKNOWN',         # 未知
]

# 通知渠道
NOTIFICATION_CHANNELS = [
    'EMAIL',           # 邮件
    'SMS',             # 短信
    'WEBHOOK',         # Webhook
    'SLACK',           # Slack
    'TEAMS',           # Teams
    'PAGERDUTY',       # PagerDuty
]

# 验证级别
VALIDATION_LEVELS = [
    'BASIC',           # 基础验证
    'FUNCTIONAL',      # 功能验证
    'PERFORMANCE',     # 性能验证
    'COMPREHENSIVE',   # 全面验证
]

# 公开的API函数
__all__ = [
    # 核心类
    'DisasterRecovery',
    
    # 检测系统
    'DisasterDetector',
    
    # 恢复管理
    'RecoveryManager',
    
    # 应急响应
    'EmergencyResponse',
    
    # 备份系统
    'BackupSystem',
    
    # 数据同步
    'DataSynchronizer',
    
    # 监控告警
    'MonitoringAlert',
    
    # 验证系统
    'RecoveryValidator',
    
    # 报告生成
    'ReportGenerator',
    
    # 数据结构
    'DisasterEvent',
    'RecoveryStep',
    
    # 枚举类型
    'DisasterType',
    'RecoveryStatus',
    
    # 便利函数
    'detect_disaster',
    'trigger_recovery',
    'validate_recovery',
    'monitor_system_health',
    'generate_recovery_report',
    
    # 常量
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    'DEFAULT_CONFIG',
    'DISASTER_TYPES',
    'RECOVERY_PRIORITY',
    'MONITORING_LEVELS',
    'ALERT_LEVELS',
    'RECOVERY_STATUS',
    'HEALTH_STATUS',
    'NOTIFICATION_CHANNELS',
    'VALIDATION_LEVELS',
]

# 快速入门指南
def quick_start():
    """
    快速入门指南
    
    返回一个包含基本使用示例的字符串。
    """
    return """
    R7灾难恢复器快速入门
    ====================
    
    1. 检测灾难:
       ```python
       from R7 import detect_disaster
       
       disaster = detect_disaster(
           system_health="/path/to/health.json",
           monitoring_data="/path/to/metrics.json"
       )
       ```
    
    2. 触发恢复:
       ```python
       from R7 import trigger_recovery
       
       recovery = trigger_recovery(
           disaster_type="HARDWARE_FAILURE",
           priority="CRITICAL",
           recovery_plan="primary_system_recovery"
       )
       ```
    
    3. 验证恢复:
       ```python
       from R7 import validate_recovery
       
       validation = validate_recovery(
           recovery_id="recovery_001",
           validation_level="COMPREHENSIVE"
       )
       ```
    
    4. 监控健康状态:
       ```python
       from R7 import monitor_system_health
       
       health_status = monitor_system_health(
           components=["database", "web_server", "cache"],
           check_frequency="5m"
       )
       ```
    
    5. 生成恢复报告:
       ```python
       from R7 import generate_recovery_report
       
       report = generate_recovery_report(
           recovery_id="recovery_001",
           report_format="pdf",
           include_metrics=True
       )
       ```
    
    更多信息请查看文档或运行测试文件。
    """

# 便利函数实现
def detect_disaster(system_health=None, monitoring_data=None, **kwargs):
    """
    检测灾难的便利函数
    """
    disaster_recovery = DisasterRecovery()
    return disaster_recovery.detect(system_health, monitoring_data, **kwargs)

def trigger_recovery(disaster_type, priority='MEDIUM', **kwargs):
    """
    触发恢复的便利函数
    """
    disaster_recovery = DisasterRecovery()
    return disaster_recovery.trigger_recovery(disaster_type, priority, **kwargs)

def validate_recovery(recovery_id, validation_level='FUNCTIONAL', **kwargs):
    """
    验证恢复的便利函数
    """
    recovery_validator = RecoveryValidator()
    return recovery_validator.validate(recovery_id, validation_level, **kwargs)

def monitor_system_health(components=None, **kwargs):
    """
    监控系统健康的便利函数
    """
    disaster_recovery = DisasterRecovery()
    return disaster_recovery.monitor_health(components, **kwargs)

def generate_recovery_report(recovery_id, **kwargs):
    """
    生成恢复报告的便利函数
    """
    report_generator = ReportGenerator()
    return report_generator.generate(recovery_id, **kwargs)

# 版本检查
def check_version():
    """检查版本信息"""
    import sys
    print(f"R7灾难恢复器版本: {__version__}")
    print(f"Python版本: {sys.version}")
    print(f"作者: {__author__}")
    print(f"描述: {__description__}")

if __name__ == "__main__":
    check_version()
    print(quick_start())