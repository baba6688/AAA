#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S区系统监控模块 - 包初始化文件

提供全面的系统监控和管理功能，包括：
- 任务调度和定时执行
- 通知和消息发送
- 邮件服务管理
- 报告生成和发送
- 系统监控和告警
- 数据分析和统计
- 服务维护管理
- 状态聚合和监控

主要子模块：
- S1: 调度服务 (SchedulerService)
- S2: 通知服务 (NotificationService)  
- S3: 邮件服务 (EmailService)
- S4: 消息服务 (MessageService)
- S5: 报告服务 (ReportService)
- S6: 分析服务 (AnalysisService)
- S7: 监控服务 (MonitoringService)
- S8: 维护服务 (MaintenanceService)
- S9: 状态聚合器 (ServiceStatusAggregator)

作者: S区系统监控团队
版本: 1.0.0
"""

# 核心服务导入
from .S1.SchedulerService import (
    # 调度服务
    SchedulerService,
    ScheduledTask,
    TaskQueue,
    TaskLogger,
    AlertManager,
    
    # 数据结构
    TaskStatus,
    TaskPriority,
    TaskResult,
)

from .S2.NotificationService import (
    # 通知服务
    NotificationService,
    NotificationQueue,
    NotificationHistory,
    
    # 通知类型
    NotificationType,
    NotificationStatus,
    
    # 通知对象
    Notification,
    NotificationTemplate,
    
    # 通知渠道
    EmailService,
    SMSService,
    PushService,
)

from .S3.EmailService import (
    # 邮件服务
    EmailService as EmailManager,
    EmailQueue,
    EmailFilter,
    EmailSecurity,
    EmailStats,
    
    # 邮件对象
    EmailConfig,
    EmailAttachment,
    EmailMessage,
    EmailTemplate,
)

from .S4.MessageService import (
    # 消息服务
    MessageService,
    MessagePersistence,
    
    # 消息对象
    MessageStatus,
    Message,
)

from .S5.ReportService import (
    # 报告服务
    ReportService,
    ReportTemplateManager,
    ReportSchedule,
    ReportSender,
    ReportStorage,
    ReportPermission,
    ReportStatistics,
    
    # 报告对象
    ReportStatus,
    ReportType,
    ReportConfig,
    ReportTemplate,
    ReportVersion,
)

from .S6.AnalysisService import (
    # 分析服务
    AnalysisService,
)

from .S7.MonitoringService import (
    # 监控服务
    MonitoringService,
    SystemMonitor,
    PerformanceMonitor,
    AlertManager,
    Dashboard,
    MonitoringData,
    
    # 监控对象
    AlertLevel,
    MetricData,
    AlertRule,
    Alert,
)

from .S8.MaintenanceService import (
    # 维护服务
    MaintenanceService,
    MaintenancePlan,
    
    # 维护对象
    TaskStatus as MaintenanceTaskStatus,
    TaskType,
    MaintenanceTask,
)

from .S9.ServiceStatusAggregator import (
    # 状态聚合器
    ServiceStatusAggregator,
    StatusCollector,
    DataAggregator,
    StatusAnalyzer,
    AlertSystem,
    HistoryManager,
    ReportGenerator,
    StatusMonitor,
    Dashboard as StatusDashboard,
    
    # 状态对象
    ServiceStatus,
    AlertLevel as StatusAlertLevel,
    ServiceInfo,
    AlertInfo,
    StatusReport,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "S区系统监控团队"
__email__ = "support@ssystem.com"
__description__ = "S区系统监控模块 - 全面的系统监控和管理解决方案"

# 包级别的默认配置
DEFAULT_CONFIG = {
    'scheduler': {
        'max_workers': 10,
        'task_timeout': 3600,
        'retry_attempts': 3,
        'retry_delay': 60
    },
    'notification': {
        'max_queue_size': 1000,
        'batch_size': 50,
        'retry_attempts': 3,
        'channels': ['email', 'sms', 'push']
    },
    'email': {
        'smtp_host': 'localhost',
        'smtp_port': 587,
        'timeout': 30,
        'max_retries': 3
    },
    'monitoring': {
        'collection_interval': 60,
        'retention_days': 30,
        'alert_threshold': 0.9,
        'dashboard_refresh': 30
    },
    'maintenance': {
        'auto_maintenance': False,
        'maintenance_window': '02:00-04:00',
        'notification_advance': 3600
    }
}

# 服务类型
SERVICE_TYPES = [
    'SCHEDULER',       # 调度服务
    'NOTIFICATION',    # 通知服务
    'EMAIL',          # 邮件服务
    'MESSAGE',        # 消息服务
    'REPORT',         # 报告服务
    'ANALYSIS',       # 分析服务
    'MONITORING',     # 监控服务
    'MAINTENANCE',    # 维护服务
    'AGGREGATOR',     # 状态聚合器
]

# 监控级别
MONITORING_LEVELS = [
    'INFO',           # 信息
    'WARNING',        # 警告
    'CRITICAL',       # 严重
    'EMERGENCY',      # 紧急
]

# 任务状态
TASK_STATUS = [
    'PENDING',        # 等待执行
    'RUNNING',        # 正在执行
    'COMPLETED',      # 执行完成
    'FAILED',         # 执行失败
    'CANCELLED',      # 已取消
    'RETRYING',       # 重试中
]

# 通知状态
NOTIFICATION_STATUS = [
    'PENDING',        # 待发送
    'SENDING',        # 发送中
    'SENT',           # 已发送
    'FAILED',         # 发送失败
    'DELIVERED',      # 已送达
    'READ',           # 已阅读
]

# 报告状态
REPORT_STATUS = [
    'DRAFT',          # 草稿
    'GENERATING',     # 生成中
    'COMPLETED',      # 已完成
    'FAILED',         # 生成失败
    'SENT',           # 已发送
    'ARCHIVED',       # 已归档
]

# 公开的API函数
__all__ = [
    # S1: 调度服务
    'SchedulerService',
    'ScheduledTask',
    'TaskQueue',
    'TaskLogger',
    'AlertManager',
    'TaskStatus',
    'TaskPriority',
    'TaskResult',
    
    # S2: 通知服务
    'NotificationService',
    'NotificationQueue',
    'NotificationHistory',
    'NotificationType',
    'NotificationStatus',
    'Notification',
    'NotificationTemplate',
    'EmailService',
    'SMSService',
    'PushService',
    
    # S3: 邮件服务
    'EmailManager',
    'EmailQueue',
    'EmailFilter',
    'EmailSecurity',
    'EmailStats',
    'EmailConfig',
    'EmailAttachment',
    'EmailMessage',
    'EmailTemplate',
    
    # S4: 消息服务
    'MessageService',
    'MessagePersistence',
    'MessageStatus',
    'Message',
    
    # S5: 报告服务
    'ReportService',
    'ReportTemplateManager',
    'ReportSchedule',
    'ReportSender',
    'ReportStorage',
    'ReportPermission',
    'ReportStatistics',
    'ReportStatus',
    'ReportType',
    'ReportConfig',
    'ReportTemplate',
    'ReportVersion',
    
    # S6: 分析服务
    'AnalysisService',
    
    # S7: 监控服务
    'MonitoringService',
    'SystemMonitor',
    'PerformanceMonitor',
    'AlertManager',
    'Dashboard',
    'MonitoringData',
    'AlertLevel',
    'MetricData',
    'AlertRule',
    'Alert',
    
    # S8: 维护服务
    'MaintenanceService',
    'MaintenancePlan',
    'MaintenanceTaskStatus',
    'TaskType',
    'MaintenanceTask',
    
    # S9: 状态聚合器
    'ServiceStatusAggregator',
    'StatusCollector',
    'DataAggregator',
    'StatusAnalyzer',
    'AlertSystem',
    'HistoryManager',
    'ReportGenerator',
    'StatusMonitor',
    'StatusDashboard',
    'ServiceStatus',
    'StatusAlertLevel',
    'ServiceInfo',
    'AlertInfo',
    'StatusReport',
    
    # 常量
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    'DEFAULT_CONFIG',
    'SERVICE_TYPES',
    'MONITORING_LEVELS',
    'TASK_STATUS',
    'NOTIFICATION_STATUS',
    'REPORT_STATUS',
]

# 快速入门指南
def quick_start():
    """
    快速入门指南
    
    返回一个包含基本使用示例的字符串。
    """
    return """
    S区系统监控模块快速入门
    =======================
    
    1. 启动调度服务:
       ```python
       from S import SchedulerService
       
       scheduler = SchedulerService()
       scheduler.start()
       ```
    
    2. 发送通知:
       ```python
       from S import NotificationService, NotificationType
       
       notification_service = NotificationService()
       notification = notification_service.create_notification(
           recipient="user@example.com",
           type=NotificationType.EMAIL,
           subject="系统告警",
           content="系统发现异常情况"
       )
       notification_service.send(notification)
       ```
    
    3. 监控系统状态:
       ```python
       from S import MonitoringService, AlertLevel
       
       monitoring = MonitoringService()
       monitoring.add_alert_rule(
           name="high_cpu",
           metric="cpu_usage",
           threshold=90,
           level=AlertLevel.WARNING
       )
       monitoring.start_monitoring()
       ```
    
    4. 生成报告:
       ```python
       from S import ReportService, ReportType
       
       report_service = ReportService()
       report = report_service.create_report(
           title="系统状态日报",
           type=ReportType.DAILY,
           config={"include_metrics": True}
       )
       report_service.generate(report)
       ```
    
    5. 执行维护任务:
       ```python
       from S import MaintenanceService, TaskType
       
       maintenance = MaintenanceService()
       task = maintenance.create_task(
           type=TaskType.CLEANUP,
           description="清理临时文件",
           schedule="0 2 * * *"
       )
       maintenance.execute(task)
       ```
    
    更多信息请查看文档或运行测试文件。
    """

# 便利函数
def create_scheduler():
    """创建调度器实例"""
    return SchedulerService()

def create_notification_service():
    """创建通知服务实例"""
    return NotificationService()

def create_monitoring_service():
    """创建监控服务实例"""
    return MonitoringService()

def create_report_service():
    """创建报告服务实例"""
    return ReportService()

def create_maintenance_service():
    """创建维护服务实例"""
    return MaintenanceService()

def create_status_aggregator():
    """创建状态聚合器实例"""
    return ServiceStatusAggregator()

# 版本检查
def check_version():
    """检查版本信息"""
    import sys
    print(f"S区系统监控模块版本: {__version__}")
    print(f"Python版本: {sys.version}")
    print(f"作者: {__author__}")
    print(f"描述: {__description__}")

if __name__ == "__main__":
    check_version()
    print(quick_start())