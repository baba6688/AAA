"""
S9状态聚合器模块
================

这是S区S9子模块的完整导出接口，提供了服务状态聚合、监控、分析和报告生成的完整功能。

版本信息
--------
- 版本: 1.0.0
- 最后更新: 2025-11-13
- 作者: S9开发团队
- 描述: 企业级服务状态聚合和监控系统

主要功能
--------
- 服务状态收集和聚合
- 实时监控和预警
- 状态分析和异常检测
- 历史数据管理
- 报告生成和导出
- 可视化仪表板
- 趋势分析

依赖项
------
- Python 3.7+
- matplotlib (可选，用于图表生成)
- sqlite3 (内置)
- threading (内置)
- json (内置)
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "S9开发团队"
__description__ = "企业级服务状态聚合和监控系统"
__email__ = "s9-team@company.com"
__license__ = "MIT"

# 导入所有核心类和枚举
from .ServiceStatusAggregator import (
    # 枚举类
    ServiceStatus,
    AlertLevel,
    
    # 数据类
    ServiceInfo,
    AlertInfo,
    StatusReport,
    
    # 核心组件
    StatusCollector,
    DataAggregator,
    StatusAnalyzer,
    AlertSystem,
    HistoryManager,
    ReportGenerator,
    StatusMonitor,
    Dashboard,
    ServiceStatusAggregator,
    
    # 示例函数
    example_service_collector
)

# 默认配置
DEFAULT_CONFIG = {
    # 数据库配置
    "database": {
        "path": "service_status_history.db",
        "backup_enabled": True,
        "cleanup_days": 30
    },
    
    # 监控配置
    "monitoring": {
        "check_interval": 30,  # 秒
        "concurrent_workers": 5,
        "timeout": 10,  # 秒
        "retry_count": 3
    },
    
    # 阈值配置
    "thresholds": {
        "response_time_warning": 2.0,  # 秒
        "response_time_critical": 5.0,
        "error_count_warning": 3,
        "error_count_critical": 5,
        "availability_warning": 95.0,  # 百分比
        "availability_critical": 90.0
    },
    
    # 预警配置
    "alerting": {
        "enabled": True,
        "max_alerts_per_service": 10,
        "alert_cooldown": 300,  # 秒
        "notification_channels": ["console", "log"]
    },
    
    # 报告配置
    "reporting": {
        "auto_generate": True,
        "interval": 3600,  # 秒
        "formats": ["json", "html"],
        "include_charts": True
    },
    
    # 可视化配置
    "dashboard": {
        "chart_theme": "default",
        "chart_width": 1200,
        "chart_height": 800,
        "dpi": 300,
        "colors": {
            "healthy": "#4CAF50",
            "warning": "#FF9800", 
            "critical": "#F44336",
            "offline": "#9E9E9E",
            "unknown": "#2196F3"
        }
    }
}

# 常量定义
class Constants:
    """系统常量定义"""
    
    # 状态常量
    STATUS_HEALTHY = "healthy"
    STATUS_WARNING = "warning"
    STATUS_CRITICAL = "critical"
    STATUS_OFFLINE = "offline"
    STATUS_UNKNOWN = "unknown"
    
    # 预警级别常量
    ALERT_INFO = "info"
    ALERT_WARNING = "warning"
    ALERT_CRITICAL = "critical"
    
    # 文件扩展名
    EXT_JSON = ".json"
    EXT_HTML = ".html"
    EXT_CSV = ".csv"
    EXT_PNG = ".png"
    EXT_DB = ".db"
    
    # 时间格式
    TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    
    # 数据库表名
    TABLE_SERVICE_HISTORY = "service_status_history"
    TABLE_ALERT_HISTORY = "alert_history"
    
    # 默认值
    DEFAULT_CHECK_INTERVAL = 30  # 秒
    DEFAULT_RESPONSE_TIMEOUT = 10  # 秒
    DEFAULT_HISTORY_DAYS = 30  # 天
    DEFAULT_MAX_ALERTS = 100
    
    # API端点
    ENDPOINT_STATUS = "/api/status"
    ENDPOINT_HEALTH = "/health"
    ENDPOINT_METRICS = "/metrics"

# 便利函数
def create_aggregator(config: dict = None, db_path: str = None) -> ServiceStatusAggregator:
    """
    创建状态聚合器实例的便利函数
    
    Args:
        config: 自定义配置字典，如果不提供则使用默认配置
        db_path: 数据库文件路径
    
    Returns:
        ServiceStatusAggregator: 配置好的聚合器实例
    """
    # 合并配置
    final_config = DEFAULT_CONFIG.copy()
    if config:
        final_config.update(config)
    
    # 创建聚合器
    db_path = db_path or final_config["database"]["path"]
    aggregator = ServiceStatusAggregator(db_path)
    
    return aggregator

def quick_monitor(service_configs: list, interval: int = 30) -> ServiceStatusAggregator:
    """
    快速设置监控的便利函数
    
    Args:
        service_configs: 服务配置列表，每个配置包含service_id, name, endpoint, collector_func
        interval: 检查间隔（秒）
    
    Returns:
        ServiceStatusAggregator: 配置好的聚合器实例
    
    Example:
        >>> def my_service_check():
        ...     return {"status": ServiceStatus.HEALTHY, "response_time": 0.5}
        >>> 
        >>> configs = [
        ...     {
        ...         "service_id": "web_service",
        ...         "name": "Web服务", 
        ...         "endpoint": "http://localhost:8080",
        ...         "collector_func": my_service_check
        ...     }
        ... ]
        >>> 
        >>> aggregator = quick_monitor(configs, interval=30)
        >>> aggregator.start_monitoring()
    """
    aggregator = create_aggregator()
    
    # 注册服务
    for config in service_configs:
        aggregator.register_service(
            service_id=config["service_id"],
            name=config["name"],
            endpoint=config["endpoint"],
            collector_func=config["collector_func"],
            metadata=config.get("metadata", {})
        )
    
    # 开始监控
    aggregator.start_monitoring(interval)
    
    return aggregator

def generate_sample_report(output_file: str = "sample_report.html") -> StatusReport:
    """
    生成示例报告的便利函数
    
    Args:
        output_file: 输出文件路径
    
    Returns:
        StatusReport: 生成的状态报告
    
    Example:
        >>> report = generate_sample_report("my_report.html")
        >>> print(report.summary)
    """
    aggregator = create_aggregator()
    
    # 注册示例服务
    aggregator.register_service(
        service_id="example_service",
        name="示例服务",
        endpoint="http://localhost:8080",
        collector_func=example_service_collector
    )
    
    # 收集状态并生成报告
    aggregator.collect_status()
    report = aggregator.generate_report()
    
    # 导出报告
    if output_file.endswith('.html'):
        html_content = aggregator.report_generator.export_report_html(report)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
    elif output_file.endswith('.json'):
        json_content = aggregator.report_generator.export_report_json(report)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(json_content)
    
    return report

def health_check_summary(aggregator: ServiceStatusAggregator) -> dict:
    """
    获取健康检查摘要的便利函数
    
    Args:
        aggregator: 状态聚合器实例
    
    Returns:
        dict: 健康检查摘要信息
    
    Example:
        >>> aggregator = create_aggregator()
        >>> # ... 注册并收集服务状态
        >>> summary = health_check_summary(aggregator)
        >>> print(f"系统健康度: {summary['health_rate']:.1f}%")
    """
    metrics = aggregator.get_aggregated_metrics()
    
    if not metrics:
        return {"status": "no_data", "message": "暂无服务数据"}
    
    total = metrics.get("total_services", 0)
    if total == 0:
        return {"status": "no_services", "message": "未注册任何服务"}
    
    # 计算健康度（基于状态分布）
    status_dist = metrics.get("status_distribution", {})
    healthy_count = status_dist.get("healthy", 0)
    health_rate = (healthy_count / total) * 100
    
    # 确定整体状态
    if health_rate >= 90:
        overall_status = "excellent"
    elif health_rate >= 70:
        overall_status = "good"
    elif health_rate >= 50:
        overall_status = "warning"
    else:
        overall_status = "critical"
    
    return {
        "overall_status": overall_status,
        "health_rate": health_rate,
        "total_services": total,
        "healthy_services": healthy_count,
        "warning_services": status_dist.get("warning", 0),
        "critical_services": status_dist.get("critical", 0),
        "offline_services": status_dist.get("offline", 0),
        "avg_response_time": metrics.get("avg_response_time", 0),
        "timestamp": metrics.get("timestamp").isoformat() if metrics.get("timestamp") else None
    }

# 快速入门指南
QUICK_START_GUIDE = """
快速入门指南
============

1. 基本使用
-----------

```python
from S9 import create_aggregator, ServiceStatus, example_service_collector

# 创建聚合器
aggregator = create_aggregator()

# 注册服务
aggregator.register_service(
    service_id="my_service",
    name="我的服务",
    endpoint="http://localhost:8080",
    collector_func=example_service_collector
)

# 收集状态
services = aggregator.collect_status()
print("服务状态:", {sid: service.status.value for sid, service in services.items()})

# 生成报告
report = aggregator.generate_report()
print("系统摘要:", report.summary)
```

2. 监控设置
-----------

```python
from S9 import quick_monitor

# 定义服务检查函数
def check_my_service():
    import requests
    try:
        response = requests.get("http://localhost:8080/health", timeout=5)
        return {
            "status": ServiceStatus.HEALTHY if response.status_code == 200 else ServiceStatus.CRITICAL,
            "response_time": response.elapsed.total_seconds()
        }
    except:
        return {
            "status": ServiceStatus.OFFLINE,
            "response_time": 0
        }

# 快速设置监控
configs = [{
    "service_id": "my_service",
    "name": "我的服务",
    "endpoint": "http://localhost:8080", 
    "collector_func": check_my_service
}]

aggregator = quick_monitor(configs, interval=30)
```

3. 预警配置
-----------

```python
from S9 import AlertLevel

# 添加自定义预警回调
def my_alert_callback(alert):
    print(f"🚨 自定义预警: {alert.level.value} - {alert.message}")
    # 这里可以添加邮件通知、Slack消息等

aggregator.alert_system.add_alert_callback(my_alert_callback)

# 添加预警规则
def check_response_time(service_info):
    if service_info.response_time > 5.0:
        return True
    return False

aggregator.alert_system.add_alert_rule("response_time", check_response_time)
```

4. 数据导出
-----------

```python
# 导出仪表板数据
aggregator.export_dashboard_json("dashboard.json")

# 生成图表（需要matplotlib）
charts = aggregator.generate_charts("./charts")
print("生成的图表:", charts)

# 导出HTML报告
report = aggregator.generate_report()
html_content = aggregator.report_generator.export_report_html(report)
with open("status_report.html", "w", encoding="utf-8") as f:
    f.write(html_content)
```

5. 历史数据分析
---------------

```python
# 获取服务历史数据
history = aggregator.history_manager.get_service_history("my_service", hours=24)
print("过去24小时记录数:", len(history))

# 获取趋势分析
trend = aggregator.data_aggregator.get_trend_analysis("response_time", hours=24)
print("响应时间趋势:", trend["trend"])
```

6. 配置自定义
------------

```python
custom_config = {
    "monitoring": {
        "check_interval": 15,  # 15秒检查一次
        "timeout": 5
    },
    "thresholds": {
        "response_time_warning": 1.0,
        "response_time_critical": 3.0
    }
}

aggregator = create_aggregator(config=custom_config)
```

注意事项
--------
- 确保已安装所有依赖项（特别是matplotlib用于图表生成）
- 数据库文件会自动创建，注意文件权限
- 大量服务监控时考虑性能影响
- 生产环境建议配置日志级别为WARNING或ERROR
- 定期清理历史数据以避免数据库过大
"""

# 模块元数据
__all__ = [
    # 版本信息
    "__version__",
    "__author__", 
    "__description__",
    
    # 核心类
    "ServiceStatus",
    "AlertLevel", 
    "ServiceInfo",
    "AlertInfo",
    "StatusReport",
    "StatusCollector",
    "DataAggregator",
    "StatusAnalyzer", 
    "AlertSystem",
    "HistoryManager",
    "ReportGenerator",
    "StatusMonitor",
    "Dashboard",
    "ServiceStatusAggregator",
    
    # 便利函数
    "create_aggregator",
    "quick_monitor", 
    "generate_sample_report",
    "health_check_summary",
    
    # 配置和常量
    "DEFAULT_CONFIG",
    "Constants",
    "QUICK_START_GUIDE",
    
    # 示例函数
    "example_service_collector"
]

# 模块初始化日志
import logging
logger = logging.getLogger(__name__)
logger.info(f"S9状态聚合器模块已加载 - 版本: {__version__}")

# 模块检查
def check_dependencies():
    """检查依赖项是否满足"""
    missing_deps = []
    
    try:
        import matplotlib
        logger.info("✓ matplotlib已安装")
    except ImportError:
        missing_deps.append("matplotlib")
        logger.warning("⚠ matplotlib未安装，图表功能将不可用")
    
    try:
        import sqlite3
        logger.info("✓ sqlite3可用")
    except ImportError:
        missing_deps.append("sqlite3")
    
    try:
        import threading
        logger.info("✓ threading可用")
    except ImportError:
        missing_deps.append("threading")
    
    if missing_deps:
        logger.warning(f"缺少依赖项: {', '.join(missing_deps)}")
        return False
    else:
        logger.info("所有依赖项检查通过")
        return True

# 启动时自动检查依赖
if __name__ != "__main__":
    check_dependencies()