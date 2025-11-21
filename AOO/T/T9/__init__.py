"""
T9数据状态聚合器模块 - 统一导出接口

该模块提供了完整的数据状态聚合和监控功能，包括：
- 数据模块状态收集和监控
- 数据质量指标聚合和分析
- 数据使用统计和性能监控
- 数据健康度评估和告警管理
- 数据资源消耗监控和状态变更跟踪
- 数据状态报告生成和导出功能

Author: T9系统开发团队
Date: 2025-11-13
Version: 1.0.0
License: MIT
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "T9系统开发团队"
__license__ = "MIT"
__email__ = "dev@T9-system.com"
__description__ = "T9数据状态聚合器 - 完整的数据监控和状态管理系统"

# 类型导入
from typing import Dict, Any, List, Optional, Union, Tuple, Callable

# 核心类导出
from .DataStateAggregator import (
    # 枚举类
    DataStatus,
    DataQualityLevel,
    AlertLevel,
    
    # 数据结构类
    DataModuleInfo,
    DataQualityMetrics,
    DataUsageStats,
    DataPerformanceMetrics,
    ResourceConsumption,
    StateChangeEvent,
    DataHealthScore,
    Alert,
    
    # 收集器类
    DataModuleCollector,
    MockDataModuleCollector,
    
    # 管理器类
    AlertManager,
    
    # 主要功能类
    DataStateAggregator,
    
    # 测试类
    TestDataStateAggregator
)

# 版本元数据
__all__ = [
    # 版本信息
    "__version__",
    "__author__", 
    "__license__",
    "__email__",
    "__description__",
    
    # 枚举类
    "DataStatus",
    "DataQualityLevel", 
    "AlertLevel",
    
    # 数据类
    "DataModuleInfo",
    "DataQualityMetrics",
    "DataUsageStats", 
    "DataPerformanceMetrics",
    "ResourceConsumption",
    "StateChangeEvent",
    "DataHealthScore",
    "Alert",
    
    # 核心功能类
    "DataModuleCollector",
    "MockDataModuleCollector",
    "AlertManager",
    "DataStateAggregator",
    "TestDataStateAggregator",
    
    # 便利函数
    "create_aggregator",
    "quick_start_demo",
    "generate_sample_data",
    "get_system_health",
    "export_monitoring_data"
]

# =============================================================================
# 默认配置常量
# =============================================================================

# 默认配置
DEFAULT_CONFIG = {
    # 数据收集配置
    "collection_interval": 60,  # 数据收集间隔（秒）
    "max_history_records": 1000,  # 最大历史记录数
    "alert_retention_days": 30,  # 告警保留天数
    
    # 性能配置
    "max_concurrent_collectors": 10,  # 最大并发收集器数量
    "collection_timeout": 30,  # 单次收集超时时间（秒）
    
    # 告警配置
    "cpu_warning_threshold": 80.0,  # CPU告警阈值
    "memory_warning_threshold": 90.0,  # 内存告警阈值
    "error_rate_threshold": 5.0,  # 错误率告警阈值
    "quality_score_threshold": 0.7,  # 质量分数告警阈值
    
    # 报告配置
    "report_formats": ["json", "text", "html"],  # 支持的报告格式
    "include_metadata": True,  # 报告中包含元数据
    "max_recent_events": 10,  # 报告中包含的最大最近事件数
}

# 健康度阈值配置
HEALTH_THRESHOLDS = {
    "excellent": 90.0,  # 优秀阈值
    "good": 75.0,       # 良好阈值  
    "fair": 60.0,       # 一般阈值
    "poor": 40.0,       # 较差阈值
    "unacceptable": 0.0 # 不可接受阈值
}

# 资源使用阈值
RESOURCE_THRESHOLDS = {
    "cpu_cores": {"max": 16.0, "warning": 12.0},
    "memory_mb": {"max": 16384.0, "warning": 12288.0},
    "disk_gb": {"max": 1000.0, "warning": 800.0},
    "network_mbps": {"max": 1000.0, "warning": 800.0},
    "storage_gb": {"max": 5000.0, "warning": 4000.0},
}

# 性能指标配置
PERFORMANCE_CONFIG = {
    "latency_percentiles": [50, 95, 99],  # 延迟百分位数
    "throughput_unit": "qps",  # 吞吐量单位
    "error_rate_unit": "percent",  # 错误率单位
    "availability_target": 99.9,  # 可用性目标（百分比）
}

# 告警配置
ALERT_CONFIG = {
    "auto_escalation": True,  # 自动升级告警
    "escalation_timeout": 300,  # 升级超时时间（秒）
    "notification_channels": ["log", "email"],  # 通知渠道
    "rate_limiting": {
        "max_alerts_per_hour": 100,
        "cooldown_period": 60  # 冷却期（秒）
    }
}

# =============================================================================
# 常量定义
# =============================================================================

# 状态常量
STATUS_LABELS = {
    "healthy": "健康",
    "warning": "警告", 
    "critical": "严重",
    "unknown": "未知",
    "maintenance": "维护中"
}

# 质量等级标签
QUALITY_LABELS = {
    "excellent": "优秀",
    "good": "良好",
    "fair": "一般", 
    "poor": "较差",
    "unacceptable": "不可接受"
}

# 告警级别标签
ALERT_LABELS = {
    "info": "信息",
    "warning": "警告",
    "error": "错误", 
    "critical": "严重"
}

# 模块类型标签
MODULE_TYPE_LABELS = {
    "user_data": "用户数据",
    "order_data": "订单数据",
    "product_data": "产品数据",
    "transaction_data": "交易数据",
    "log_data": "日志数据",
    "cache_data": "缓存数据"
}

# 单位常量
UNITS = {
    "cpu": "%",
    "memory": "MB", 
    "disk": "GB",
    "network": "Mbps",
    "storage": "GB",
    "latency": "ms",
    "throughput": "QPS",
    "cost": "元/小时"
}

# 收集器类型
COLLECTOR_TYPES = {
    "mock": "模拟收集器",
    "prometheus": "Prometheus收集器",
    "statsd": "StatsD收集器",
    "custom": "自定义收集器"
}

# =============================================================================
# 便利函数
# =============================================================================

def create_aggregator(
    collector_type: str = "mock",
    collection_interval: int = DEFAULT_CONFIG["collection_interval"],
    **kwargs
) -> DataStateAggregator:
    """
    创建数据状态聚合器实例的便利函数
    
    Args:
        collector_type: 收集器类型 ("mock", "prometheus", "statsd", "custom")
        collection_interval: 数据收集间隔（秒）
        **kwargs: 其他初始化参数
        
    Returns:
        DataStateAggregator: 数据状态聚合器实例
        
    Example:
        # 创建模拟数据聚合器
        aggregator = create_aggregator("mock", collection_interval=30)
        
        # 创建自定义收集器聚合器
        custom_collector = MyCustomCollector()
        aggregator = create_aggregator("custom", collector=custom_collector)
    """
    from .DataStateAggregator import MockDataModuleCollector, DataModuleCollector
    
    if collector_type == "mock":
        collector = MockDataModuleCollector()
        return DataStateAggregator(collector=collector, collection_interval=collection_interval)
    elif collector_type == "custom":
        collector = kwargs.get("collector")
        if not collector or not isinstance(collector, DataModuleCollector):
            raise ValueError("自定义收集器必须是DataModuleCollector的实例")
        return DataStateAggregator(collector=collector, collection_interval=collection_interval)
    else:
        raise ValueError(f"不支持的收集器类型: {collector_type}")


async def quick_start_demo(
    duration: int = 10,
    collection_interval: int = 3
) -> None:
    """
    快速开始演示函数
    
    Args:
        duration: 演示运行时间（秒）
        collection_interval: 数据收集间隔（秒）
        
    Example:
        # 运行10秒演示
        await quick_start_demo(duration=10)
    """
    from .DataStateAggregator import DataStateAggregator, MockDataModuleCollector
    
    print("🚀 T9数据状态聚合器快速演示")
    print("=" * 40)
    
    # 创建聚合器
    collector = MockDataModuleCollector()
    aggregator = DataStateAggregator(collector=collector, collection_interval=collection_interval)
    
    try:
        # 启动数据收集
        print("📊 启动数据收集...")
        await aggregator.start_collection()
        
        # 运行指定时间
        print(f"⏱️  运行数据收集（{duration}秒）...")
        await asyncio.sleep(duration)
        
        # 生成并显示报告
        print("\n📈 生成系统状态报告...")
        report = aggregator.generate_status_report("text")
        print(report)
        
        # 显示系统概览
        print("\n📋 系统概览信息:")
        overview = aggregator.get_system_overview()
        for key, value in overview.items():
            print(f"  • {key}: {value}")
        
        # 显示活跃告警
        active_alerts = aggregator.alert_manager.get_active_alerts()
        print(f"\n🚨 活跃告警数量: {len(active_alerts)}")
        for alert in active_alerts:
            print(f"  • [{ALERT_LABELS[alert.alert_level.value]}] {alert.title}")
        
    except Exception as e:
        print(f"❌ 演示过程中发生错误: {e}")
    finally:
        # 停止数据收集
        print("\n⏹️  停止数据收集...")
        await aggregator.stop_collection()
        print("✅ 演示完成!")


def generate_sample_data(
    module_count: int = 3,
    include_metrics: bool = True
) -> Dict[str, Any]:
    """
    生成示例数据
    
    Args:
        module_count: 模块数量
        include_metrics: 是否包含指标数据
        
    Returns:
        Dict: 示例数据字典
    """
    from .DataStateAggregator import (
        DataModuleInfo, DataStatus, DataQualityMetrics, DataPerformanceMetrics,
        ResourceConsumption, DataHealthScore
    )
    from datetime import datetime
    import random
    
    modules = []
    for i in range(module_count):
        module = DataModuleInfo(
            module_id=f"module_{i+1}",
            module_name=f"示例模块_{i+1}",
            module_type=f"type_{i+1}",
            status=random.choice(list(DataStatus)),
            last_updated=datetime.now(),
            metadata={"description": f"这是示例模块 {i+1}"}
        )
        modules.append(module)
    
    sample_data = {
        "modules": [m.__dict__ for m in modules],
        "timestamp": datetime.now().isoformat(),
        "collection_interval": DEFAULT_CONFIG["collection_interval"],
    }
    
    if include_metrics:
        # 添加示例指标
        for module in modules:
            quality = DataQualityMetrics(
                completeness=random.uniform(0.7, 1.0),
                accuracy=random.uniform(0.8, 1.0),
                consistency=random.uniform(0.7, 0.95),
                timeliness=random.uniform(0.6, 0.9),
                validity=random.uniform(0.75, 1.0)
            )
            
            performance = DataPerformanceMetrics(
                cpu_usage=random.uniform(10, 90),
                memory_usage=random.uniform(20, 95),
                disk_io=random.uniform(5, 50),
                network_io=random.uniform(1, 30),
                query_latency_p50=random.uniform(10, 100),
                query_latency_p95=random.uniform(50, 300),
                query_latency_p99=random.uniform(100, 600),
                throughput_qps=random.uniform(100, 2000),
                error_rate=random.uniform(0, 10)
            )
            
            resource = ResourceConsumption(
                cpu_cores=random.uniform(0.5, 8.0),
                memory_mb=random.uniform(512, 8192),
                disk_gb=random.uniform(10, 500),
                network_mbps=random.uniform(1, 200),
                storage_gb=random.uniform(50, 2000),
                cost_per_hour=random.uniform(0.1, 10.0)
            )
            
            health = DataHealthScore(
                overall_score=random.uniform(60, 95),
                availability=random.uniform(80, 100),
                performance=random.uniform(70, 95),
                quality=quality.overall_score * 100,
                security=random.uniform(80, 95),
                compliance=random.uniform(85, 100)
            )
            
            module.metrics = {
                "quality": quality.__dict__,
                "performance": performance.__dict__, 
                "resource": resource.__dict__,
                "health": health.__dict__
            }
    
    return sample_data


def get_system_health(aggregator: DataStateAggregator) -> Dict[str, Any]:
    """
    获取系统健康状况摘要
    
    Args:
        aggregator: 数据状态聚合器实例
        
    Returns:
        Dict: 系统健康状况摘要
    """
    with aggregator._lock:
        overview = aggregator.get_system_overview()
        active_alerts = aggregator.alert_manager.get_active_alerts()
        
        # 计算健康等级
        total_modules = overview["total_modules"]
        if total_modules == 0:
            health_level = "unknown"
        else:
            critical_count = overview["modules_by_status"].get("critical", 0)
            warning_count = overview["modules_by_status"].get("warning", 0)
            
            if critical_count > 0:
                health_level = "critical"
            elif warning_count > total_modules * 0.3:
                health_level = "warning"
            else:
                health_level = "healthy"
        
        return {
            "health_level": health_level,
            "health_score": 100 - (critical_count * 20 + warning_count * 10),
            "total_modules": total_modules,
            "healthy_modules": overview["modules_by_status"].get("healthy", 0),
            "warning_modules": overview["modules_by_status"].get("warning", 0),
            "critical_modules": overview["modules_by_status"].get("critical", 0),
            "active_alerts": len(active_alerts),
            "average_quality": overview["average_quality_score"],
            "collection_success_rate": (
                overview["collection_stats"]["successful_collections"] / 
                max(overview["collection_stats"]["total_collections"], 1) * 100
            )
        }


def export_monitoring_data(
    aggregator: DataStateAggregator,
    format_type: str = "json",
    include_history: bool = True
) -> str:
    """
    导出监控数据
    
    Args:
        aggregator: 数据状态聚合器实例
        format_type: 导出格式 ("json", "csv")
        include_history: 是否包含历史数据
        
    Returns:
        str: 导出的数据字符串
    """
    if format_type.lower() == "json":
        return aggregator.export_data()
    else:
        raise ValueError(f"不支持的导出格式: {format_type}")


# =============================================================================
# 快速入门指南
# =============================================================================

def print_quick_start_guide():
    """打印快速入门指南"""
    guide = """
🎯 T9数据状态聚合器 - 快速入门指南
=======================================

📚 基本概念:
-----------
• DataStateAggregator: 核心聚合器类，负责数据收集和分析
• DataModuleCollector: 数据收集器抽象基类
• MockDataModuleCollector: 模拟数据收集器实现
• AlertManager: 告警管理器
• DataStatus: 数据状态枚举 (healthy, warning, critical, unknown, maintenance)
• DataQualityLevel: 数据质量等级 (excellent, good, fair, poor, unacceptable)

🚀 快速开始:
-----------
1. 创建聚合器实例:
   from T9 import create_aggregator, DataStateAggregator
   
   # 使用默认配置
   aggregator = create_aggregator("mock")
   
   # 自定义配置
   aggregator = DataStateAggregator(collection_interval=30)

2. 启动数据收集:
   import asyncio
   
   async def main():
       await aggregator.start_collection()
       # ... 等待数据收集
       await aggregator.stop_collection()
   
   asyncio.run(main())

3. 生成状态报告:
   # JSON格式报告
   json_report = aggregator.generate_status_report("json")
   
   # 文本格式报告  
   text_report = aggregator.generate_status_report("text")
   
   # HTML格式报告
   html_report = aggregator.generate_status_report("html")

4. 获取系统信息:
   # 获取系统概览
   overview = aggregator.get_system_overview()
   
   # 获取模块详细信息
   module_info = aggregator.get_module_status("module_1")
   
   # 获取系统健康状况
   health = get_system_health(aggregator)

5. 处理告警:
   # 获取活跃告警
   active_alerts = aggregator.alert_manager.get_active_alerts()
   
   # 创建自定义告警
   alert = aggregator.alert_manager.create_alert(
       AlertLevel.WARNING,
       "自定义告警标题",
       "告警描述内容",
       "module_1"
   )
   
   # 解决告警
   aggregator.alert_manager.resolve_alert(alert.alert_id)

6. 导出数据:
   # 导出所有数据
   export_data = aggregator.export_data()
   
   # 便利函数导出
   exported_data = export_monitoring_data(aggregator, "json")

💡 高级用法:
-----------
• 自定义数据收集器: 继承DataModuleCollector实现自定义收集逻辑
• 自定义告警处理器: 使用AlertManager.add_alert_handler()添加处理器
• 批量数据导出: 使用export_monitoring_data()批量导出监控数据
• 健康度评估: 使用DataHealthScore.get_health_level()获取健康等级
• 性能监控: 监控CPU、内存、磁盘、网络等资源使用情况
• 质量评估: 评估数据完整性、准确性、一致性、及时性、有效性

🔧 配置选项:
-----------
• collection_interval: 数据收集间隔（默认60秒）
• max_history_records: 最大历史记录数（默认1000条）
• alert_retention_days: 告警保留天数（默认30天）
• cpu_warning_threshold: CPU告警阈值（默认80%）
• memory_warning_threshold: 内存告警阈值（默认90%）
• error_rate_threshold: 错误率告警阈值（默认5%）

📊 监控指标:
-----------
• 性能指标: CPU使用率、内存使用率、磁盘IO、网络IO、查询延迟、吞吐量、错误率
• 质量指标: 完整性、准确性、一致性、及时性、有效性
• 使用统计: 读操作、写操作、查询次数、独特用户、响应时间、并发用户、数据量
• 资源消耗: CPU核心、内存、磁盘、网络、存储、成本
• 健康评分: 可用性、性能、质量、安全性、合规性

🎓 示例代码:
-----------
# 基本使用示例
from T9 import create_aggregator
import asyncio

async def example():
    # 创建聚合器
    aggregator = create_aggregator("mock", collection_interval=5)
    
    # 启动收集
    await aggregator.start_collection()
    
    # 运行一段时间
    await asyncio.sleep(10)
    
    # 生成报告
    report = aggregator.generate_status_report("text")
    print(report)
    
    # 停止收集
    await aggregator.stop_collection()

# 运行示例
asyncio.run(example())

# 快速演示
from T9 import quick_start_demo
await quick_start_demo(duration=15)

❓ 常见问题:
-----------
Q: 如何添加自定义告警？
A: 使用aggregator.alert_manager.create_alert()方法

Q: 如何监控多个模块？
A: DataStateAggregator会自动监控所有注册的模块

Q: 如何自定义数据收集器？
A: 继承DataModuleCollector并实现所有抽象方法

Q: 如何调整收集频率？
A: 在创建聚合器时设置collection_interval参数

📞 技术支持:
-----------
• 邮箱: dev@T9-system.com
• 文档: https://docs.T9-system.com
• GitHub: https://github.com/T9-system/monitoring
• 问题反馈: https://github.com/T9-system/issues

🎉 开始使用:
-----------
现在您可以开始使用T9数据状态聚合器了！
建议先运行quick_start_demo()进行快速体验。
"""
    print(guide)


def print_api_reference():
    """打印API参考"""
    api_doc = """
📖 T9数据状态聚合器 API 参考
============================

🏗️ 核心类:
---------
• DataStateAggregator: 主要聚合器类
• DataModuleCollector: 数据收集器基类
• MockDataModuleCollector: 模拟收集器实现
• AlertManager: 告警管理器
• TestDataStateAggregator: 测试类

📊 数据结构:
-----------
• DataStatus: 数据状态枚举
• DataQualityLevel: 数据质量等级枚举
• AlertLevel: 告警级别枚举
• DataModuleInfo: 模块信息数据类
• DataQualityMetrics: 质量指标数据类
• DataUsageStats: 使用统计数据类
• DataPerformanceMetrics: 性能指标数据类
• ResourceConsumption: 资源消耗数据类
• StateChangeEvent: 状态变更事件数据类
• DataHealthScore: 健康度评分数据类
• Alert: 告警信息数据类

🔧 主要方法:
-----------
• start_collection(): 启动数据收集
• stop_collection(): 停止数据收集
• collect_all_data(): 收集所有数据
• generate_status_report(): 生成状态报告
• get_system_overview(): 获取系统概览
• get_module_status(): 获取模块状态
• export_data(): 导出数据
• calculate_health_scores(): 计算健康度分数
• check_alert_conditions(): 检查告警条件

📝 详细文档请参考模块源代码注释
"""
    print(api_doc)


# 在模块加载时打印欢迎信息
def _welcome_message():
    """显示欢迎信息"""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                  🚀 T9数据状态聚合器 {__version__}                          ║
║                                                              ║
║    完整的数据监控和状态管理系统                              ║
║    专注于数据质量、性能监控和健康度评估                      ║
║                                                              ║
║    📚 快速入门: print_quick_start_guide()                   ║
║    📖 API参考: print_api_reference()                        ║
║    🎯 快速演示: quick_start_demo()                          ║
║    ⚙️  创建实例: create_aggregator()                        ║
║                                                              ║
║    📧 技术支持: dev@T9-system.com                           ║
╚══════════════════════════════════════════════════════════════╝
    """)

# 显示欢迎信息
_welcome_message()

# 导入必要的模块
import asyncio