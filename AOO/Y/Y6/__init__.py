#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y6存储监控器包

这是一个功能完整的存储系统监控解决方案，提供：
- 存储状态监控
- 性能监控
- 告警管理
- 监控数据管理
- 监控报告生成
- 监控配置管理
- 监控历史管理
- 监控性能优化

主要组件：
- StorageMonitor: 主监控器类
- DatabaseManager: 数据库管理器
- AlertManager: 告警管理器
- PerformanceMonitor: 性能监控器
- EmailAlertHandler: 邮件告警处理器
- LogAlertHandler: 日志告警处理器

数据类：
- StorageInfo: 存储信息
- PerformanceMetrics: 性能指标
- Alert: 告警信息

使用示例：
    from StorageMonitor import StorageMonitor
    
    # 创建监控器
    monitor = StorageMonitor()
    
    # 开始监控
    monitor.start_monitoring()
    
    # 获取状态
    status = monitor.get_status()
    
    # 生成报告
    report = monitor.get_storage_report('/home', 24)

版本: 1.0.0
作者: Y6开发团队
许可证: MIT
"""

__version__ = "1.0.0"
__author__ = "Y6开发团队"
__email__ = "dev@y6team.com"
__license__ = "MIT"

# 导入主要类和函数
from .StorageMonitor import (
    StorageMonitor,
    DatabaseManager,
    AlertManager,
    PerformanceMonitor,
    EmailAlertHandler,
    LogAlertHandler,
    StorageInfo,
    PerformanceMetrics,
    Alert
)

# 包的公共接口
__all__ = [
    # 主要类
    'StorageMonitor',
    'DatabaseManager', 
    'AlertManager',
    'PerformanceMonitor',
    
    # 告警处理器
    'EmailAlertHandler',
    'LogAlertHandler',
    
    # 数据类
    'StorageInfo',
    'PerformanceMetrics',
    'Alert',
    
    # 版本信息
    '__version__',
    '__author__',
    '__email__',
    '__license__'
]

# 包级别的配置
DEFAULT_CONFIG = {
    'monitor_interval': 60,
    'database_path': 'storage_monitor.db',
    'log_level': 'INFO',
    'log_file': 'storage_monitor.log',
    'alert_cooldown_minutes': 5,
    'usage_thresholds': {
        'warning': 85,
        'critical': 95
    },
    'free_space_thresholds': {
        'warning': 5,
        'critical': 1
    },
    'monitored_mounts': ['/'],
    'cleanup_old_data_days': 30,
    'report_generation_time': '09:00'
}

def create_monitor(config_path=None, **kwargs):
    """
    创建一个新的存储监控器实例
    
    Args:
        config_path: 配置文件路径，如果为None则使用默认配置
        **kwargs: 其他配置参数，会覆盖配置文件中的设置
    
    Returns:
        StorageMonitor: 监控器实例
    
    Example:
        # 使用默认配置
        monitor = create_monitor()
        
        # 使用自定义配置文件
        monitor = create_monitor('/path/to/config.yaml')
        
        # 使用参数配置
        monitor = create_monitor(monitor_interval=30, usage_thresholds={'warning': 80})
    """
    if config_path:
        monitor = StorageMonitor(config_path)
    else:
        monitor = StorageMonitor()
    
    # 应用额外的配置参数
    if kwargs:
        monitor.update_config(kwargs)
    
    return monitor

def quick_start(mount_points=None, interval=60):
    """
    快速启动存储监控
    
    Args:
        mount_points: 要监控的挂载点列表，默认为['/']
        interval: 监控间隔（秒），默认为60秒
    
    Returns:
        StorageMonitor: 监控器实例（已开始监控）
    
    Example:
        # 快速监控根目录
        monitor = quick_start()
        
        # 监控多个挂载点
        monitor = quick_start(['/', '/home', '/var'], interval=30)
    """
    if mount_points is None:
        mount_points = ['/']
    
    config = {
        'monitor_interval': interval,
        'monitored_mounts': mount_points
    }
    
    monitor = create_monitor(**config)
    monitor.start_monitoring()
    
    return monitor

def get_default_config():
    """
    获取默认配置
    
    Returns:
        dict: 默认配置字典
    """
    return DEFAULT_CONFIG.copy()

def validate_config(config):
    """
    验证配置参数
    
    Args:
        config: 配置字典
    
    Returns:
        bool: 配置是否有效
    
    Raises:
        ValueError: 配置无效时抛出异常
    """
    required_keys = [
        'monitor_interval', 'usage_thresholds', 'free_space_thresholds'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"缺少必需的配置项: {key}")
    
    # 验证监控间隔
    interval = config['monitor_interval']
    if not isinstance(interval, int) or interval < 1:
        raise ValueError("监控间隔必须是大于0的整数")
    
    # 验证阈值
    usage_thresholds = config['usage_thresholds']
    if 'warning' not in usage_thresholds or 'critical' not in usage_thresholds:
        raise ValueError("使用率阈值必须包含warning和critical")
    
    if usage_thresholds['warning'] >= usage_thresholds['critical']:
        raise ValueError("警告阈值必须小于严重阈值")
    
    free_thresholds = config['free_space_thresholds']
    if 'warning' not in free_thresholds or 'critical' not in free_thresholds:
        raise ValueError("剩余空间阈值必须包含warning和critical")
    
    if free_thresholds['warning'] <= free_thresholds['critical']:
        raise ValueError("警告阈值必须大于严重阈值")
    
    return True

# 包初始化时的检查
def _check_dependencies():
    """检查依赖包是否安装"""
    missing_deps = []
    
    try:
        import psutil
    except ImportError:
        missing_deps.append('psutil')
    
    try:
        import yaml
    except ImportError:
        missing_deps.append('pyyaml')
    
    try:
        import schedule
    except ImportError:
        missing_deps.append('schedule')
    
    if missing_deps:
        import warnings
        warnings.warn(
            f"缺少以下依赖包: {', '.join(missing_deps)}\n"
            f"请使用以下命令安装: pip install {' '.join(missing_deps)}",
            UserWarning
        )

# 执行依赖检查
_check_dependencies()

# 包级别的工具函数
def list_supported_filesystems():
    """
    获取支持的文件系统列表
    
    Returns:
        list: 支持的文件系统类型列表
    """
    return [
        'ext2', 'ext3', 'ext4', 'xfs', 'btrfs', 'ntfs', 
        'fat32', 'exfat', 'tmpfs', 'ramfs', 'nfs', 'cifs'
    ]

def estimate_resource_usage(mount_count=1, interval=60):
    """
    估算资源使用情况
    
    Args:
        mount_count: 监控的挂载点数量
        interval: 监控间隔（秒）
    
    Returns:
        dict: 资源使用估算
    """
    # 基于测试数据的估算
    base_memory = 20  # MB
    memory_per_mount = 5  # MB per mount
    base_cpu = 1  # %
    cpu_per_mount = 0.5  # % per mount
    base_disk = 1  # MB per day
    disk_per_mount = 0.5  # MB per day per mount
    
    # 监控间隔对资源使用的影响
    interval_factor = max(0.5, min(2.0, 60 / interval))
    
    return {
        'estimated_memory_mb': base_memory + (mount_count * memory_per_mount * interval_factor),
        'estimated_cpu_percent': base_cpu + (mount_count * cpu_per_mount * interval_factor),
        'estimated_disk_mb_per_day': base_disk + (mount_count * disk_per_mount),
        'monitoring_interval_seconds': interval,
        'mount_points_count': mount_count
    }

def create_sample_config(output_path='storage_monitor_sample.yaml'):
    """
    创建示例配置文件
    
    Args:
        output_path: 输出文件路径
    
    Example:
        create_sample_config('/path/to/sample_config.yaml')
    """
    sample_config = """
# Y6存储监控器示例配置文件
# 请根据实际情况修改配置参数

# 监控配置
monitor_interval: 60                    # 监控间隔（秒）
database_path: storage_monitor.db       # 数据库文件路径
log_level: INFO                        # 日志级别 (DEBUG, INFO, WARNING, ERROR)
log_file: storage_monitor.log          # 日志文件路径

# 告警配置
alert_cooldown_minutes: 5              # 告警冷却时间（分钟）

# 存储使用率阈值（百分比）
usage_thresholds:
  warning: 85                          # 警告阈值
  critical: 95                         # 严重阈值

# 剩余空间阈值（GB）
free_space_thresholds:
  warning: 5                           # 警告阈值
  critical: 1                          # 严重阈值

# 邮件告警配置
smtp_config:
  smtp_server: smtp.example.com        # SMTP服务器地址
  smtp_port: 587                       # SMTP端口
  use_tls: true                        # 是否使用TLS
  from_email: monitor@example.com      # 发件人邮箱
  to_emails:                           # 收件人邮箱列表
    - admin@example.com
    - ops@example.com
  username: your_username              # SMTP用户名（可选）
  password: your_password              # SMTP密码（可选）

# 要监控的挂载点
monitored_mounts:
  - /                                  # 根文件系统
  - /home                              # 用户数据目录
  - /var                               # 变量数据目录
  - /opt                               # 应用程序目录

# 数据清理配置
cleanup_old_data_days: 30              # 清理多少天前的数据

# 报告生成时间
report_generation_time: "09:00"        # 每日报告生成时间
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample_config.strip())
    
    return output_path

# 包的元数据
PACKAGE_INFO = {
    'name': 'Y6存储监控器',
    'version': __version__,
    'description': '功能完整的存储系统监控解决方案',
    'author': __author__,
    'license': __license__,
    'python_requires': '>=3.7',
    'keywords': ['storage', 'monitor', 'disk', 'alert', 'performance'],
    'classifiers': [
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: System Administrators',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: System :: Systems Administration',
        'Topic :: System :: Monitoring',
    ],
    'dependencies': [
        'psutil>=5.6.0',
        'pyyaml>=5.3.0',
        'schedule>=1.0.0'
    ]
}

def get_package_info():
    """
    获取包信息
    
    Returns:
        dict: 包信息字典
    """
    return PACKAGE_INFO.copy()

# 便捷函数
def get_version():
    """获取版本号"""
    return __version__

def get_author():
    """获取作者信息"""
    return f"{__author__} <{__email__}>"

def print_banner():
    """打印启动横幅"""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                     Y6存储监控器 v{__version__:<10}                    ║
║                                                              ║
║  功能完整的存储系统监控解决方案                              ║
║  - 存储状态监控    - 性能监控                                ║
║  - 告警管理        - 监控报告                                ║
║  - 历史数据        - 性能优化                                ║
║                                                              ║
║  作者: {get_author():<50} ║
║  许可证: {__license__:<49} ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

# 包初始化完成标识
_PACKAGE_INITIALIZED = True