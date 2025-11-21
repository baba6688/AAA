"""
L8审计日志记录器模块

这是一个全面的企业级审计日志记录系统，支持多种类型的审计日志记录，
包括用户操作、数据访问、系统配置、交易操作、安全事件和合规性审计。

主要功能：
- 用户操作审计日志（登录、登出、权限变更）
- 数据访问审计日志（数据读取、数据修改、数据删除）
- 系统配置审计日志（配置修改、配置删除、配置备份）
- 交易操作审计日志（交易下单、交易修改、交易取消）
- 安全事件审计日志（攻击检测、异常访问、权限绕过）
- 合规性审计日志（监管要求、审计报告、风险评估）
- 异步审计日志处理
- 完整的错误处理和日志记录
- 详细的文档字符串和使用示例

主要类：
- AuditLogger: 主要的审计日志记录器类
- AuditRecord: 审计记录数据结构
- AuditContext: 审计上下文信息
- SQLiteAuditStorage: SQLite存储实现
- FileAuditStorage: 文件存储实现
- ConsoleAuditHandler: 控制台处理器
- FileAuditHandler: 文件处理器
- EmailAuditHandler: 邮件处理器
- WebhookAuditHandler: Webhook处理器
- AuditValidator: 审计数据验证器
- AuditRules: 预定义审计规则
- AdvancedAuditAnalyzer: 高级分析工具
- AuditDashboard: 审计仪表板
- AuditArchiver: 审计数据归档器
- AuditIntegrationTool: 审计集成工具
- AuditConfigManager: 审计配置管理器
- AuditUtils: 审计工具函数集合

"""

from .AuditLogger import (
    # 枚举类
    AuditLevel,
    AuditCategory,
    UserOperation,
    DataOperation,
    ConfigOperation,
    TransactionOperation,
    SecurityEvent,
    ComplianceType,
    
    # 数据结构
    AuditContext,
    AuditRecord,
    AuditFilter,
    
    # 存储接口和实现
    AuditStorage,
    SQLiteAuditStorage,
    FileAuditStorage,
    
    # 处理器
    AuditHandler,
    ConsoleAuditHandler,
    FileAuditHandler,
    EmailAuditHandler,
    WebhookAuditHandler,
    
    # 异步处理
    AsyncAuditQueue,
    
    # 主要记录器
    AuditLogger,
    
    # 分析和验证
    AuditAnalyzer,
    AuditValidator,
    AuditCacheManager,
    AuditRuleEngine,
    AuditRules,
    AdvancedAuditAnalyzer,
    
    # 报告和显示
    AuditReportGenerator,
    AuditDashboard,
    
    # 压缩和归档
    AuditCompressor,
    AuditArchiver,
    
    # 集成和工具
    AuditIntegrationTool,
    AuditPerformanceMonitor,
    AuditConfigManager,
    AuditUtils,
    AuditTestTool,
    
    # 创建函数
    create_audit_context,
    
    # 单元测试
    AuditLoggerTest
)

__version__ = "1.0.0"
__author__ = "L8系统"
__description__ = "企业级审计日志记录器"

__all__ = [
    # 枚举类
    "AuditLevel",
    "AuditCategory",
    "UserOperation",
    "DataOperation",
    "ConfigOperation",
    "TransactionOperation",
    "SecurityEvent",
    "ComplianceType",
    
    # 数据结构
    "AuditContext",
    "AuditRecord",
    "AuditFilter",
    
    # 存储接口和实现
    "AuditStorage",
    "SQLiteAuditStorage",
    "FileAuditStorage",
    
    # 处理器
    "AuditHandler",
    "ConsoleAuditHandler",
    "FileAuditHandler",
    "EmailAuditHandler",
    "WebhookAuditHandler",
    
    # 异步处理
    "AsyncAuditQueue",
    
    # 主要记录器
    "AuditLogger",
    
    # 分析和验证
    "AuditAnalyzer",
    "AuditValidator",
    "AuditCacheManager",
    "AuditRuleEngine",
    "AuditRules",
    "AdvancedAuditAnalyzer",
    
    # 报告和显示
    "AuditReportGenerator",
    "AuditDashboard",
    
    # 压缩和归档
    "AuditCompressor",
    "AuditArchiver",
    
    # 集成和工具
    "AuditIntegrationTool",
    "AuditPerformanceMonitor",
    "AuditConfigManager",
    "AuditUtils",
    "AuditTestTool",
    
    # 创建函数
    "create_audit_context",
    
    # 单元测试
    "AuditLoggerTest"
]