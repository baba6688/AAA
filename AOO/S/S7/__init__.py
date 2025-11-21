"""
S7监控服务模块
===============

一个功能完整的系统监控解决方案，提供实时监控、性能分析、告警管理和可视化仪表板功能。

主要功能：
- 系统资源监控（CPU、内存、磁盘、网络）
- 性能指标收集和分析
- 智能告警规则管理
- 历史数据存储和查询
- 可视化监控仪表板
- 自动故障检测
- 自定义指标支持

版本信息
--------
"""

__version__ = "1.0.0"
__author__ = "S7 Team"
__email__ = "s7-team@example.com"
__description__ = "S7监控服务 - 完整的系统监控解决方案"

# 版本兼容性标识
__all__ = [
    # 版本信息
    "__version__",
    "__author__",
    "__email__", 
    "__description__",
    
    # 核心类
    "MonitoringService",
    "SystemMonitor", 
    "PerformanceMonitor",
    "AlertManager",
    "MonitoringData",
    "Dashboard",
    
    # 数据结构
    "AlertLevel",
    "MetricData",
    "AlertRule",
    "Alert",
    
    # 便利函数
    "create_monitoring_service",
    "quick_start",
    "get_default_config",
    "setup_logging",
    "health_check",
    "detect_faults",
    "generate_report",
    
    # 常量
    "DEFAULT_CONFIG",
    "DEFAULT_ALERT_RULES",
    "METRIC_CATEGORIES",
    "MONITORING_INTERVALS"
]


# ==================== 常量定义 ====================

# 默认配置
DEFAULT_CONFIG = {
    # 数据库配置
    'db_path': 'monitoring_data.db',
    'db_backup_interval': 3600,  # 备份间隔（秒）
    
    # 监控配置
    'monitoring_interval': 60,  # 监控间隔（秒）
    'retention_days': 30,  # 数据保留天数
    'max_metrics_per_category': 1000,
    
    # 告警配置
    'alert': {
        'email_notifications': False,
        'smtp_server': '',
        'smtp_port': 587,
        'smtp_username': '',
        'smtp_password': '',
        'alert_cooldown': 300,  # 告警冷却时间（秒）
        'max_alerts_per_hour': 100
    },
    
    # 性能配置
    'performance': {
        'max_threads': 4,
        'cache_size': 1000,
        'batch_size': 100,
        'async_processing': True
    },
    
    # 日志配置
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_path': 'monitoring.log',
        'max_file_size': 10485760,  # 10MB
        'backup_count': 5
    },
    
    # 仪表板配置
    'dashboard': {
        'refresh_interval': 30,
        'chart_width': 1200,
        'chart_height': 600,
        'dpi': 150,
        'theme': 'default'
    }
}

# 监控指标类别
METRIC_CATEGORIES = {
    'system': '系统指标',
    'performance': '性能指标', 
    'network': '网络指标',
    'application': '应用指标',
    'custom': '自定义指标'
}

# 默认监控间隔
MONITORING_INTERVALS = {
    'realtime': 1,      # 实时监控（1秒）
    'frequent': 10,     # 频繁监控（10秒）
    'normal': 60,       # 正常监控（1分钟）
    'extended': 300,    # 扩展监控（5分钟）
    'minimal': 1800     # 最小监控（30分钟）
}

# 默认告警规则
DEFAULT_ALERT_RULES = {
    'high_cpu': {
        'name': '高CPU使用率',
        'metric': 'cpu_usage',
        'operator': '>',
        'threshold': 80.0,
        'level': 'warning',
        'description': 'CPU使用率超过80%'
    },
    'high_memory': {
        'name': '高内存使用率', 
        'metric': 'memory_usage',
        'operator': '>',
        'threshold': 85.0,
        'level': 'warning',
        'description': '内存使用率超过85%'
    },
    'low_disk_space': {
        'name': '磁盘空间不足',
        'metric': 'disk_usage', 
        'operator': '>',
        'threshold': 90.0,
        'level': 'critical',
        'description': '磁盘使用率超过90%'
    },
    'high_system_load': {
        'name': '系统负载过高',
        'metric': 'system_load_1min',
        'operator': '>', 
        'threshold': 2.0,
        'level': 'warning',
        'description': '系统1分钟负载超过2.0'
    },
    'network_errors': {
        'name': '网络错误',
        'metric': 'network_error_rate',
        'operator': '>',
        'threshold': 5.0,
        'level': 'error',
        'description': '网络错误率超过5%'
    }
}

# 系统健康状态
HEALTH_STATUS = {
    'healthy': '健康',
    'warning': '警告', 
    'error': '错误',
    'critical': '严重',
    'unknown': '未知'
}

# 告警级别权重（用于排序和优先级）
ALERT_LEVEL_WEIGHTS = {
    'info': 1,
    'warning': 2,
    'error': 3, 
    'critical': 4
}


# ==================== 便利函数 ====================

def create_monitoring_service(config=None):
    """
    创建监控服务实例
    
    Args:
        config (dict, optional): 配置字典，如果为None则使用默认配置
        
    Returns:
        MonitoringService: 监控服务实例
        
    Example:
        >>> # 使用默认配置
        >>> service = create_monitoring_service()
        
        >>> # 使用自定义配置  
        >>> config = {
        ...     'monitoring_interval': 30,
        ...     'db_path': '/var/log/monitoring.db'
        ... }
        >>> service = create_monitoring_service(config)
    """
    from .MonitoringService import MonitoringService
    merged_config = {**DEFAULT_CONFIG, **(config or {})}
    return MonitoringService(merged_config)


def quick_start(config=None, auto_start=True):
    """
    快速启动监控服务
    
    Args:
        config (dict, optional): 配置字典
        auto_start (bool): 是否自动开始监控
        
    Returns:
        MonitoringService: 监控服务实例
        
    Example:
        >>> # 最简单的启动方式
        >>> service = quick_start()
        
        >>> # 自定义配置并启动
        >>> config = {
        ...     'monitoring_interval': 30,
        ...     'alert': {'email_notifications': True}
        ... }
        >>> service = quick_start(config=config, auto_start=True)
    """
    service = create_monitoring_service(config)
    
    if auto_start:
        service.start_monitoring()
        print("✅ S7监控服务已启动")
        print(f"📊 监控间隔: {service.monitoring_interval}秒")
        print(f"🗄️ 数据库路径: {service.data_manager.db_path}")
        print(f"⚠️ 活跃告警规则: {len(service.alert_manager.rules)}个")
    
    return service


def get_default_config():
    """
    获取默认配置
    
    Returns:
        dict: 默认配置字典
    """
    return DEFAULT_CONFIG.copy()


def setup_logging(level='INFO', file_path=None):
    """
    设置监控服务日志
    
    Args:
        level (str): 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_path (str, optional): 日志文件路径
        
    Example:
        >>> setup_logging('DEBUG', '/var/log/monitoring.log')
        >>> setup_logging('INFO')  # 控制台输出
    """
    import logging
    
    logger = logging.getLogger('S7MonitoringService')
    logger.setLevel(getattr(logging, level.upper()))
    
    # 清除现有处理器
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # 文件处理器
    if file_path:
        try:
            file_handler = logging.FileHandler(file_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            print(f"📝 日志文件: {file_path}")
        except Exception as e:
            print(f"⚠️ 无法创建日志文件: {e}")
    
    print(f"📋 日志级别: {level}")
    return logger


def health_check():
    """
    快速系统健康检查
    
    Returns:
        dict: 健康检查结果
        
    Example:
        >>> result = health_check()
        >>> print(f"系统状态: {result['status']}")
        >>> if result['issues']:
        ...     print("发现的问题:")
        ...     for issue in result['issues']:
        ...         print(f"  - {issue}")
    """
    service = create_monitoring_service()
    return service.health_check()


def detect_faults():
    """
    快速故障检测
    
    Returns:
        list: 检测到的故障列表
        
    Example:
        >>> faults = detect_faults()
        >>> if faults:
        ...     print(f"检测到 {len(faults)} 个故障:")
        ...     for fault in faults:
        ...         print(f"  [{fault['severity']}] {fault['description']}")
    """
    service = create_monitoring_service()
    return service.detect_faults()


def generate_report(hours=24):
    """
    生成监控报告
    
    Args:
        hours (int): 报告时间范围（小时）
        
    Returns:
        dict: 监控报告数据
        
    Example:
        >>> report = generate_report(24)  # 24小时报告
        >>> print(f"报告时间范围: {hours}小时")
        >>> print(f"总告警数: {report['total_alerts']}")
        >>> print(f"活跃告警: {report['active_alerts']}")
    """
    service = create_monitoring_service()
    # 临时创建仪表板来生成报告
    dashboard = service.dashboard
    return dashboard.generate_report()


def get_system_status():
    """
    获取当前系统状态
    
    Returns:
        dict: 系统状态信息
        
    Example:
        >>> status = get_system_status()
        >>> print(f"CPU使用率: {status['cpu_usage']:.1f}%")
        >>> print(f"内存使用率: {status['memory_usage']:.1f}%") 
        >>> print(f"磁盘使用率: {status['disk_usage']:.1f}%")
        >>> print(f"活跃告警: {status['active_alerts']}个")
    """
    service = create_monitoring_service()
    return service.get_system_status()


def add_custom_metric(name, value, unit, category='custom'):
    """
    添加自定义监控指标
    
    Args:
        name (str): 指标名称
        value (float): 指标值
        unit (str): 单位
        category (str): 类别
        
    Example:
        >>> add_custom_metric('应用响应时间', 0.25, '秒', 'application')
        >>> add_custom_metric('用户会话数', 150, '个', 'application')
        >>> add_custom_metric('数据库连接池', 0.75, '%', 'database')
    """
    service = create_monitoring_service()
    service.add_custom_metric(name, value, unit, category)


def create_alert_rule(name, metric, operator, threshold, level='warning', description=''):
    """
    创建告警规则
    
    Args:
        name (str): 规则名称
        metric (str): 监控指标名
        operator (str): 操作符 (>, <, >=, <=, ==)
        threshold (float): 阈值
        level (str): 告警级别 (info, warning, error, critical)
        description (str): 描述
        
    Returns:
        AlertRule: 告警规则对象
        
    Example:
        >>> rule = create_alert_rule(
        ...     name='高磁盘I/O',
        ...     metric='disk_io_usage', 
        ...     operator='>',
        ...     threshold=70.0,
        ...     level='warning',
        ...     description='磁盘I/O使用率过高'
        ... )
    """
    from .MonitoringService import AlertRule, AlertLevel
    
    level_map = {
        'info': AlertLevel.INFO,
        'warning': AlertLevel.WARNING, 
        'error': AlertLevel.ERROR,
        'critical': AlertLevel.CRITICAL
    }
    
    return AlertRule(
        name=name,
        metric=metric,
        operator=operator,
        threshold=threshold,
        level=level_map.get(level.lower(), AlertLevel.WARNING),
        description=description
    )


# ==================== 快速入门指南 ====================

def print_quick_start_guide():
    """
    打印快速入门指南
    
    Example:
        >>> print_quick_start_guide()
    """
    guide = """
🚀 S7监控服务 - 快速入门指南
================================

📚 基础使用
-----------

1. 快速启动（推荐新手）
   >>> from S7 import quick_start
   >>> service = quick_start()

2. 使用自定义配置
   >>> from S7 import create_monitoring_service
   >>> config = {
   ...     'monitoring_interval': 30,  # 30秒监控间隔
   ...     'db_path': '/var/log/monitoring.db'  # 自定义数据库路径
   ... }
   >>> service = create_monitoring_service(config)

3. 启动/停止监控
   >>> service.start_monitoring()   # 开始监控
   >>> service.stop_monitoring()    # 停止监控

📊 监控数据
-----------

1. 查看系统状态
   >>> from S7 import get_system_status
   >>> status = get_system_status()
   >>> print(f"CPU: {status['cpu_usage']:.1f}%")

2. 添加自定义指标
   >>> from S7 import add_custom_metric
   >>> add_custom_metric('响应时间', 0.25, '秒')

3. 生成监控报告
   >>> from S7 import generate_report
   >>> report = generate_report(24)  # 24小时报告

⚠️ 告警管理
------------

1. 创建告警规则
   >>> from S7 import create_alert_rule
   >>> rule = create_alert_rule(
   ...     name='高CPU使用率',
   ...     metric='cpu_usage',
   ...     operator='>', 
   ...     threshold=80.0,
   ...     level='warning'
   ... )

2. 查看活跃告警
   >>> active_alerts = service.alert_manager.get_active_alerts()
   >>> for alert in active_alerts:
   ...     print(f"[{alert.level.value}] {alert.message}")

🔧 高级配置
-----------

1. 数据库配置
   >>> config = {
   ...     'db_path': 'custom_monitoring.db',
   ...     'retention_days': 7  # 保留7天数据
   ... }

2. 告警配置
   >>> config = {
   ...     'alert': {
   ...         'email_notifications': True,
   ...         'smtp_server': 'smtp.gmail.com',
   ...         'alert_cooldown': 300  # 5分钟冷却时间
   ...     }
   ... }

3. 性能配置
   >>> config = {
   ...     'performance': {
   ...         'max_threads': 8,
   ...         'cache_size': 2000,
   ...         'async_processing': True
   ...     }
   ... }

📈 数据可视化
-------------

1. 生成性能图表
   >>> chart = service.generate_performance_chart('cpu_usage', hours=24)
   >>> # chart是base64编码的PNG图像

2. 获取仪表板数据
   >>> dashboard_data = service.get_dashboard_data()

🩺 健康检查
-----------

1. 系统健康检查
   >>> from S7 import health_check
   >>> result = health_check()

2. 故障检测
   >>> from S7 import detect_faults
   >>> faults = detect_faults()

📋 主要类和对象
---------------

- MonitoringService: 监控服务主类
- SystemMonitor: 系统监控器  
- PerformanceMonitor: 性能监控器
- AlertManager: 告警管理器
- MonitoringData: 数据管理器
- Dashboard: 仪表板类

🔗 有用的链接
------------

- 文档: https://docs.s7-monitoring.com
- 示例: https://github.com/s7-team/examples
- 问题反馈: https://github.com/s7-team/issues
- 邮箱支持: s7-support@example.com

✨ 开始你的监控之旅吧！
    """
    print(guide)


# ==================== 模块导入便捷性 ====================

# 从MonitoringService模块导入所有核心类
from .MonitoringService import (
    AlertLevel,
    MetricData, 
    AlertRule,
    Alert,
    SystemMonitor,
    PerformanceMonitor,
    AlertManager,
    MonitoringData,
    Dashboard,
    MonitoringService
)

# 设置默认日志
try:
    setup_logging(DEFAULT_CONFIG['logging']['level'])
except Exception:
    pass  # 忽略日志设置错误

# 模块初始化完成提示
def _init_message():
    """显示模块初始化信息"""
    print(f"✅ S7监控服务 v{__version__} 已加载")
    print(f"📚 文档: https://docs.s7-monitoring.com")
    print(f"💡 快速开始: S7.print_quick_start_guide()")
    print(f"🆘 获取帮助: help(S7)")

# 自动显示初始化信息（仅在交互式环境中）
try:
    import sys
    if sys.flags.interactive:
        _init_message()
except Exception:
    pass