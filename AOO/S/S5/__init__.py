"""
S5报告服务 - 完整导出接口

提供报告生成、调度、发送、存储、权限管理、版本控制和统计等功能的完整解决方案。

主要模块:
- ReportService: 主服务类
- ReportTemplateManager: 模板管理
- ReportSchedule: 调度管理
- ReportSender: 发送服务
- ReportStorage: 存储管理
- ReportPermission: 权限控制
- ReportVersion: 版本管理
- ReportStatistics: 统计分析

版本: 1.0.0
作者: S区开发团队
更新: 2025-11-13
"""

import os
import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

# 版本信息
__version__ = "1.0.0"
__author__ = "S区开发团队"
__email__ = "dev@company.com"
__description__ = "S5报告服务 - 智能报告生成与管理系统"

# 导入所有核心类
try:
    # 相对导入（当作为模块使用时）
    from .ReportService import (
        ReportStatus,
        ReportType,
        ReportConfig,
        ReportTemplate,
        ReportTemplateManager,
        ReportSchedule,
        ReportSender,
        ReportStorage,
        ReportPermission,
        ReportVersion,
        ReportStatistics,
        ReportService
    )
except ImportError:
    # 绝对导入（当直接运行时）
    from ReportService import (
        ReportStatus,
        ReportType,
        ReportConfig,
        ReportTemplate,
        ReportTemplateManager,
        ReportSchedule,
        ReportSender,
        ReportStorage,
        ReportPermission,
        ReportVersion,
        ReportStatistics,
        ReportService
    )

# 默认配置
DEFAULT_CONFIG = {
    "template_path": "templates",
    "storage_path": "reports", 
    "version_path": "versions",
    "permission_db": "permissions.db",
    "statistics_db": "statistics.db",
    "smtp": {
        "server": "smtp.gmail.com",
        "port": 587,
        "username": "",
        "password": ""
    }
}

# 系统常量
MAX_REPORT_SIZE = 50 * 1024 * 1024  # 50MB
MAX_RECIPIENTS = 100
TEMPLATE_CACHE_SIZE = 100
DEFAULT_TEMPLATE_VARIABLES = ["generated_at", "user_id", "session_id"]

# 权限类型常量
PERMISSION_TYPES = {
    "READ": "read",
    "WRITE": "write", 
    "ADMIN": "admin",
    "FULL": "full"
}

# 报告状态常量
STATUS_MAP = {
    "pending": ReportStatus.PENDING,
    "generating": ReportStatus.GENERATING,
    "completed": ReportStatus.COMPLETED,
    "failed": ReportStatus.FAILED,
    "sent": ReportStatus.SENT
}

# 报告类型常量
TYPE_MAP = {
    "daily": ReportType.DAILY,
    "weekly": ReportType.WEEKLY,
    "monthly": ReportType.MONTHLY,
    "custom": ReportType.CUSTOM
}

# 便利函数
def create_default_config(**overrides) -> Dict[str, Any]:
    """创建默认配置，可选择覆盖特定参数"""
    config = DEFAULT_CONFIG.copy()
    config.update(overrides)
    return config

def get_report_service(config: Optional[Dict[str, Any]] = None) -> ReportService:
    """获取ReportService实例（便利函数）"""
    if config is None:
        config = create_default_config()
    return ReportService(config)

def quick_start_config(template_path: Optional[str] = None,
                      storage_path: Optional[str] = None,
                      smtp_server: Optional[str] = None,
                      smtp_username: Optional[str] = None,
                      smtp_password: Optional[str] = None) -> Dict[str, Any]:
    """快速配置向导 - 简化配置创建"""
    config = create_default_config()
    
    if template_path:
        config["template_path"] = template_path
    if storage_path:
        config["storage_path"] = storage_path
    if smtp_server or smtp_username or smtp_password:
        config["smtp"].update({
            "server": smtp_server or config["smtp"]["server"],
            "username": smtp_username or "",
            "password": smtp_password or ""
        })
    
    return config

def create_daily_report_config(name: str, template_id: str, recipients: List[str],
                             cron_expression: str = "0 9 * * *", enabled: bool = True) -> ReportConfig:
    """创建日报配置（便利函数）"""
    return ReportConfig(
        name=name,
        description=f"{name} - 日报配置",
        report_type=ReportType.DAILY,
        template_id=template_id,
        recipients=recipients,
        schedule_cron=cron_expression,
        enabled=enabled
    )

def create_weekly_report_config(name: str, template_id: str, recipients: List[str],
                              cron_expression: str = "0 9 * * 1", enabled: bool = True) -> ReportConfig:
    """创建周报配置（便利函数）"""
    return ReportConfig(
        name=name,
        description=f"{name} - 周报配置",
        report_type=ReportType.WEEKLY,
        template_id=template_id,
        recipients=recipients,
        schedule_cron=cron_expression,
        enabled=enabled
    )

def create_monthly_report_config(name: str, template_id: str, recipients: List[str],
                               cron_expression: str = "0 9 1 * *", enabled: bool = True) -> ReportConfig:
    """创建月报配置（便利函数）"""
    return ReportConfig(
        name=name,
        description=f"{name} - 月报配置",
        report_type=ReportType.MONTHLY,
        template_id=template_id,
        recipients=recipients,
        schedule_cron=cron_expression,
        enabled=enabled
    )

def validate_config(config: Dict[str, Any]) -> List[str]:
    """验证配置完整性，返回错误信息列表"""
    errors = []
    
    # 检查必要字段
    required_fields = ["template_path", "storage_path", "smtp"]
    for field in required_fields:
        if field not in config:
            errors.append(f"缺少必要配置字段: {field}")
    
    # 检查SMTP配置
    if "smtp" in config:
        smtp = config["smtp"]
        if not smtp.get("server"):
            errors.append("SMTP服务器地址不能为空")
        if not smtp.get("username"):
            errors.append("SMTP用户名不能为空")
    
    # 检查路径
    if "template_path" in config:
        try:
            Path(config["template_path"]).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"模板路径创建失败: {e}")
    
    if "storage_path" in config:
        try:
            Path(config["storage_path"]).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"存储路径创建失败: {e}")
    
    return errors

def get_system_info() -> Dict[str, Any]:
    """获取系统信息"""
    return {
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "python_version": "3.7+",
        "dependencies": [
            "pathlib",
            "sqlite3", 
            "threading",
            "datetime",
            "email",
            "smtplib"
        ],
        "optional_dependencies": [
            "schedule (用于调度功能)"
        ],
        "supported_formats": ["html", "pdf", "excel"],
        "max_report_size": f"{MAX_REPORT_SIZE // (1024*1024)}MB",
        "max_recipients": MAX_RECIPIENTS,
        "supported_statuses": list(STATUS_MAP.keys()),
        "supported_types": list(TYPE_MAP.keys())
    }

def quick_demo():
    """快速演示函数"""
    print("🚀 S5报告服务快速演示")
    print("=" * 50)
    
    # 1. 显示系统信息
    info = get_system_info()
    print(f"📦 版本: {info['version']}")
    print(f"👨‍💻 作者: {info['author']}")
    print(f"📝 描述: {info['description']}")
    print(f"🔧 Python版本: {info['python_version']}")
    print()
    
    # 2. 显示配置示例
    print("⚙️ 默认配置:")
    print(f"  模板路径: {DEFAULT_CONFIG['template_path']}")
    print(f"  存储路径: {DEFAULT_CONFIG['storage_path']}")
    print(f"  SMTP服务器: {DEFAULT_CONFIG['smtp']['server']}")
    print(f"  SMTP端口: {DEFAULT_CONFIG['smtp']['port']}")
    print()
    
    # 3. 显示使用指南
    print("📚 快速使用指南:")
    print("1. 创建配置: config = create_default_config()")
    print("2. 初始化服务: service = get_report_service(config)")
    print("3. 创建模板: template = ReportTemplate(...)")
    print("4. 配置报告: config = create_daily_report_config(...)")
    print("5. 生成报告: report_path = service.generate_report(config, data)")
    print("6. 发送报告: service.send_report(report_path, config)")
    print("7. 启动调度: service.start_scheduler()")
    print()
    
    # 4. 状态和类型
    print("📊 支持的报告状态:")
    for status in ReportStatus:
        print(f"  - {status.name}: {status.value}")
    
    print("\n📅 支持的报告类型:")
    for report_type in ReportType:
        print(f"  - {report_type.name}: {report_type.value}")
    
    print("\n🔐 支持的权限类型:")
    for perm_type, value in PERMISSION_TYPES.items():
        print(f"  - {perm_type}: {value}")
    
    print("\n✅ 演示完成！请参考完整文档进行开发。")

def create_sample_templates() -> List[ReportTemplate]:
    """创建示例模板"""
    return [
        ReportTemplate(
            id="business_daily",
            name="业务日报",
            description="日常业务数据汇总报告",
            content="""
            <html>
            <head>
                <title>业务日报 - {{date}}</title>
                <meta charset="UTF-8">
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    .header { background-color: #f0f8ff; padding: 20px; border-radius: 5px; }
                    .metric { display: inline-block; margin: 10px; padding: 15px; 
                             background-color: #f9f9f9; border-radius: 5px; text-align: center; }
                    .metric-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
                    .metric-label { font-size: 12px; color: #7f8c8d; }
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>业务日报</h1>
                    <p>日期: {{date}}</p>
                    <p>生成时间: {{generated_at}}</p>
                </div>
                
                <h2>核心指标</h2>
                <div class="metric">
                    <div class="metric-value">{{user_count}}</div>
                    <div class="metric-label">活跃用户</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{order_count}}</div>
                    <div class="metric-label">订单数量</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{{revenue}}</div>
                    <div class="metric-label">营收 (元)</div>
                </div>
                
                <h2>业务详情</h2>
                <p>今日业务表现: {{business_summary}}</p>
                
                <h2>重要提醒</h2>
                <ul>
                    {{#events}}
                    <li>{{.}}</li>
                    {{/events}}
                </ul>
                
                <p><small>本报告由S5报告服务自动生成</small></p>
            </body>
            </html>
            """,
            variables=["date", "user_count", "order_count", "revenue", "business_summary", "events"]
        ),
        ReportTemplate(
            id="weekly_summary", 
            name="周度总结",
            description="周度业务数据总结报告",
            content="""
            <html>
            <head>
                <title>周度总结 - {{week_range}}</title>
                <meta charset="UTF-8">
            </head>
            <body>
                <h1>周度业务总结</h1>
                <p>周期: {{week_range}}</p>
                <p>生成时间: {{generated_at}}</p>
                
                <h2>本周亮点</h2>
                <ul>
                    {{#highlights}}
                    <li>{{.}}</li>
                    {{/highlights}}
                </ul>
                
                <h2>数据分析</h2>
                <table border="1" style="border-collapse: collapse; width: 100%;">
                    <tr style="background-color: #f2f2f2;">
                        <th>指标</th>
                        <th>本周</th>
                        <th>上周</th>
                        <th>变化</th>
                    </tr>
                    <tr>
                        <td>用户增长</td>
                        <td>{{current_users}}</td>
                        <td>{{previous_users}}</td>
                        <td>{{user_growth}}%</td>
                    </tr>
                    <tr>
                        <td>订单量</td>
                        <td>{{current_orders}}</td>
                        <td>{{previous_orders}}</td>
                        <td>{{order_growth}}%</td>
                    </tr>
                    <tr>
                        <td>营收</td>
                        <td>{{current_revenue}}元</td>
                        <td>{{previous_revenue}}元</td>
                        <td>{{revenue_growth}}%</td>
                    </tr>
                </table>
                
                <h2>下周计划</h2>
                <p>{{next_week_plan}}</p>
            </body>
            </html>
            """,
            variables=["week_range", "highlights", "current_users", "previous_users", "user_growth",
                      "current_orders", "previous_orders", "order_growth", 
                      "current_revenue", "previous_revenue", "revenue_growth", "next_week_plan"]
        )
    ]

# 导出所有可用组件
__all__ = [
    # 版本信息
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
    
    # 核心类
    "ReportStatus",
    "ReportType", 
    "ReportConfig",
    "ReportTemplate",
    "ReportTemplateManager",
    "ReportSchedule",
    "ReportSender",
    "ReportStorage",
    "ReportPermission",
    "ReportVersion",
    "ReportStatistics",
    "ReportService",
    
    # 默认配置
    "DEFAULT_CONFIG",
    
    # 常量
    "MAX_REPORT_SIZE",
    "MAX_RECIPIENTS", 
    "TEMPLATE_CACHE_SIZE",
    "DEFAULT_TEMPLATE_VARIABLES",
    "PERMISSION_TYPES",
    "STATUS_MAP",
    "TYPE_MAP",
    
    # 便利函数
    "create_default_config",
    "get_report_service", 
    "quick_start_config",
    "create_daily_report_config",
    "create_weekly_report_config",
    "create_monthly_report_config",
    "validate_config",
    "get_system_info",
    "quick_demo",
    "create_sample_templates"
]

# 模块初始化时执行的代码
def _initialize_module():
    """模块初始化"""
    try:
        # 创建必要的目录
        Path(DEFAULT_CONFIG["template_path"]).mkdir(parents=True, exist_ok=True)
        Path(DEFAULT_CONFIG["storage_path"]).mkdir(parents=True, exist_ok=True) 
        Path(DEFAULT_CONFIG["version_path"]).mkdir(parents=True, exist_ok=True)
        
        print(f"✅ S5报告服务 {__version__} 初始化完成")
        print(f"📁 模板目录: {Path(DEFAULT_CONFIG['template_path']).absolute()}")
        print(f"📁 存储目录: {Path(DEFAULT_CONFIG['storage_path']).absolute()}")
        print(f"📁 版本目录: {Path(DEFAULT_CONFIG['version_path']).absolute()}")
        
        # 检查可选依赖
        try:
            import schedule
            print("✅ 调度功能可用")
        except ImportError:
            print("⚠️ 调度功能不可用 (需要安装: pip install schedule)")
            
    except Exception as e:
        print(f"❌ 模块初始化失败: {e}")

# 执行初始化
_initialize_module()

# 快速入门指南
QUICK_START_GUIDE = """
🚀 S5报告服务快速入门指南
================================

1. 基础设置
-----------
```python
from S5 import get_report_service, create_default_config

# 创建配置
config = create_default_config()
config['smtp']['username'] = 'your_email@gmail.com'
config['smtp']['password'] = 'your_password'

# 初始化服务
service = get_report_service(config)
```

2. 创建报告模板
--------------
```python
from S5 import ReportTemplate

template = ReportTemplate(
    id="my_report",
    name="我的报告",
    description="自定义报告模板", 
    content="<h1>{{title}}</h1><p>{{content}}</p>",
    variables=["title", "content"]
)

service.template_manager.create_template(template)
```

3. 配置报告调度
--------------
```python
from S5 import create_daily_report_config

config = create_daily_report_config(
    name="每日业务汇总",
    template_id="my_report", 
    recipients=["user1@company.com", "user2@company.com"]
)

service.add_schedule(config)
service.start_scheduler()
```

4. 生成和发送报告
----------------
```python
# 准备数据
data = {
    "title": "2025-11-13 业务汇总",
    "content": "今日业绩良好，订单量增长15%"
}

# 生成报告
report_path = service.generate_report(config, data)

# 发送报告
success = service.send_report(report_path, config, 
                             subject="每日业务汇总", 
                             body="请查看附件中的详细报告")
```

5. 监控和统计
------------
```python
# 获取统计信息
stats = service.get_statistics()
print(f"总报告数: {stats['total_reports']}")
print(f"成功率: {stats['success_rate']:.1f}%")

# 获取最近活动
activities = service.get_recent_activity(limit=5)
for activity in activities:
    print(f"- {activity['report_name']}: {activity['status']}")
```

6. 权限管理
-----------
```python
# 授予用户权限
service.grant_permission("user123", "每日业务汇总", "read")

# 检查权限
if service.check_permission("user123", "每日业务汇总"):
    print("用户有权限查看报告")
```

7. 版本控制
-----------
```python
# 获取报告版本
versions = service.get_report_versions("每日业务汇总")
for version in versions:
    print(f"版本 {version['version_number']}: {version['description']}")
```

💡 提示:
- 默认配置路径可以使用 quick_start_config() 快速修改
- 使用 validate_config() 检查配置完整性
- 运行 quick_demo() 查看完整演示
- 查看 get_system_info() 了解系统能力

📚 更多信息:
- 完整文档: https://docs.company.com/S5
- 示例代码: https://github.com/company/S5-examples
- 问题反馈: https://github.com/company/S5/issues
"""

# 在导入时提供便捷访问
def help():
    """显示帮助信息"""
    print(QUICK_START_GUIDE)

# 将帮助函数也加入导出
__all__.append("help")