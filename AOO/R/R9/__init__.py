#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R9备份状态聚合器 - 包初始化文件

提供备份状态的收集、聚合、分析、报告生成和监控功能，包括：
- 多模块备份状态聚合
- 实时监控和预警机制
- 数据分析和趋势预测
- 报告生成和可视化
- 仪表板和仪表盘
- 性能指标监控
- 历史数据分析
- 自动化告警

作者: R9备份状态聚合器团队
版本: 1.0.0
"""

# 核心类
from .BackupStatusAggregator import (
    # 主要类
    BackupStatusAggregator,
    
    # 状态管理
    BackupStatus,
    AlertLevel,
    BackupTaskInfo,
    BackupModuleStatus,
    AlertInfo,
    
    # 数据库管理
    DatabaseManager,
    
    # 收集器
    StatusCollector,
    
    # 数据聚合
    DataAggregator,
    
    # 告警系统
    AlertManager,
    
    # 报告生成
    ReportGenerator,
    
    # 可视化
    Dashboard,
    
    # 监控
    StatusMonitor,
    
    # 便利函数
    create_aggregator,
    start_monitoring,
    generate_status_report,
    check_backup_health,
    list_alerts,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "R9备份状态聚合器团队"
__email__ = "support@r9backup.com"
__description__ = "R9备份状态聚合器 - 全面的备份状态监控和分析解决方案"

# 包级别的默认配置
DEFAULT_CONFIG = {
    'data_dir': './backup_status_data',
    'database': {
        'type': 'sqlite',
        'path': './backup_status.db',
        'backup_interval': 3600
    },
    'monitoring': {
        'enabled': True,
        'interval': 60,
        'real_time': True,
        'history_retention': 2592000  # 30天
    },
    'alerting': {
        'enabled': True,
        'escalation_levels': 3,
        'notification_channels': ['email', 'log'],
        'alert_threshold': {
            'backup_failure_rate': 0.05,  # 5%
            'average_backup_time': 3600,  # 1小时
            'disk_usage': 0.90,           # 90%
        }
    },
    'reporting': {
        'auto_generate': True,
        'report_interval': 86400,  # 24小时
        'formats': ['html', 'pdf', 'json'],
        'include_charts': True
    },
    'dashboard': {
        'auto_refresh': True,
        'refresh_interval': 30,
        'max_data_points': 1000,
        'chart_types': ['line', 'bar', 'pie']
    }
}

# 备份状态
BACKUP_STATUS = [
    'PENDING',         # 待执行
    'RUNNING',         # 执行中
    'SUCCESS',         # 成功
    'FAILED',          # 失败
    'PARTIAL',         # 部分成功
    'CANCELLED',       # 已取消
    'TIMEOUT',         # 超时
    'RETRYING',        # 重试中
]

# 告警级别
ALERT_LEVELS = [
    'INFO',            # 信息
    'WARNING',         # 警告
    'MEDIUM',          # 中等
    'HIGH',            # 高
    'CRITICAL',        # 严重
]

# 模块类型
MODULE_TYPES = [
    'DATA_BACKUP',     # 数据备份
    'CONFIG_BACKUP',   # 配置备份
    'MODEL_BACKUP',    # 模型备份
    'LOG_BACKUP',      # 日志备份
    'SYSTEM_BACKUP',   # 系统备份
    'DISASTER_RECOVERY', # 灾难恢复
    'ARCHIVE_BACKUP',  # 归档备份
]

# 指标类型
METRIC_TYPES = [
    'BACKUP_DURATION',     # 备份持续时间
    'BACKUP_SIZE',         # 备份大小
    'SUCCESS_RATE',        # 成功率
    'FAILURE_RATE',        # 失败率
    'THROUGHPUT',          # 吞吐量
    'ERROR_COUNT',         # 错误数量
    'RETRY_COUNT',         # 重试次数
    'DISK_USAGE',          # 磁盘使用率
]

# 报告类型
REPORT_TYPES = [
    'SUMMARY',         # 摘要报告
    'DETAILED',        # 详细报告
    'TREND',           # 趋势报告
    'PERFORMANCE',     # 性能报告
    'COMPLIANCE',      # 合规报告
    'ALERT_SUMMARY',   # 告警摘要
]

# 时间聚合粒度
TIME_GRANULARITY = [
    'MINUTE',          # 分钟
    'HOUR',            # 小时
    'DAY',             # 天
    'WEEK',            # 周
    'MONTH',           # 月
]

# 通知渠道
NOTIFICATION_CHANNELS = [
    'EMAIL',           # 邮件
    'SMS',             # 短信
    'WEBHOOK',         # Webhook
    'SLACK',           # Slack
    'TEAMS',           # Teams
    'LOG',             # 日志
    'FILE',            # 文件
]

# 公开的API函数
__all__ = [
    # 核心类
    'BackupStatusAggregator',
    
    # 状态管理
    'BackupStatus',
    'AlertLevel',
    'BackupTaskInfo',
    'BackupModuleStatus',
    'AlertInfo',
    
    # 数据库管理
    'DatabaseManager',
    
    # 收集器
    'StatusCollector',
    
    # 数据聚合
    'DataAggregator',
    
    # 告警系统
    'AlertManager',
    
    # 报告生成
    'ReportGenerator',
    
    # 可视化
    'Dashboard',
    
    # 监控
    'StatusMonitor',
    
    # 便利函数
    'create_aggregator',
    'start_monitoring',
    'generate_status_report',
    'check_backup_health',
    'list_alerts',
    
    # 常量
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    'DEFAULT_CONFIG',
    'BACKUP_STATUS',
    'ALERT_LEVELS',
    'MODULE_TYPES',
    'METRIC_TYPES',
    'REPORT_TYPES',
    'TIME_GRANULARITY',
    'NOTIFICATION_CHANNELS',
]

# 快速入门指南
def quick_start():
    """
    快速入门指南
    
    返回一个包含基本使用示例的字符串。
    """
    return """
    R9备份状态聚合器快速入门
    ========================
    
    1. 创建聚合器:
       ```python
       from R9 import create_aggregator
       
       aggregator = create_aggregator(
           config={
               'monitoring': {'enabled': True},
               'alerting': {'enabled': True}
           }
       )
       ```
    
    2. 启动监控:
       ```python
       from R9 import start_monitoring
       
       monitoring = start_monitoring(
           modules=['R1', 'R2', 'R3'],
           interval=60
       )
       ```
    
    3. 生成状态报告:
       ```python
       from R9 import generate_status_report
       
       report = generate_status_report(
           report_type='SUMMARY',
           time_range='24h',
           format='html'
       )
       ```
    
    4. 检查备份健康:
       ```python
       from R9 import check_backup_health
       
       health = check_backup_health(
           module_name='R1',
           include_metrics=True
       )
       ```
    
    5. 列出告警:
       ```python
       from R9 import list_alerts
       
       alerts = list_alerts(
           status='active',
           level='HIGH'
       )
       ```
    
    更多信息请查看文档或运行测试文件。
    """

# 便利函数实现
def create_aggregator(config=None, **kwargs):
    """
    创建聚合器的便利函数
    """
    aggregator = BackupStatusAggregator()
    if config:
        aggregator.configure(config)
    return aggregator

def start_monitoring(modules=None, interval=60, **kwargs):
    """
    启动监控的便利函数
    """
    aggregator = BackupStatusAggregator()
    return aggregator.start_monitoring(modules, interval, **kwargs)

def generate_status_report(report_type='SUMMARY', **kwargs):
    """
    生成状态报告的便利函数
    """
    aggregator = BackupStatusAggregator()
    return aggregator.generate_report(report_type, **kwargs)

def check_backup_health(module_name, **kwargs):
    """
    检查备份健康的便利函数
    """
    aggregator = BackupStatusAggregator()
    return aggregator.check_health(module_name, **kwargs)

def list_alerts(status='all', **kwargs):
    """
    列出告警的便利函数
    """
    aggregator = BackupStatusAggregator()
    return aggregator.get_alerts(status, **kwargs)

# 版本检查
def check_version():
    """检查版本信息"""
    import sys
    print(f"R9备份状态聚合器版本: {__version__}")
    print(f"Python版本: {sys.version}")
    print(f"作者: {__author__}")
    print(f"描述: {__description__}")

if __name__ == "__main__":
    check_version()
    print(quick_start())