#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T7数据同步器模块 - 完整导出接口

T7数据同步器是一个强大的数据同步解决方案，支持多数据源同步、实时和批量同步、
数据冲突检测解决、进度监控、错误处理、性能优化、安全控制、日志审计等功能。

作者: T7系统
创建时间: 2025-11-13
版本: 1.0.0
许可证: MIT
文档: https://github.com/t7/datasync
"""

# ================================
# 类型导入
# ================================

from typing import Tuple, List, Any, Optional, Dict, Union, Callable

# ================================
# 版本信息
# ================================

__version__ = "1.0.0"
__author__ = "T7系统"
__email__ = "support@t7-system.com"
__license__ = "MIT"
__description__ = "T7数据同步器 - 强大的多数据源同步解决方案"
__url__ = "https://github.com/t7/datasync"

# 版本信息字典
VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "version": __version__,
    "release_date": "2025-11-13",
    "author": __author__,
    "license": __license__
}

# ================================
# 导入核心类
# ================================

# 定义导入状态和错误变量
_import_success = True
_import_error = None

try:
    # 枚举类
    from .DataSynchronizer import (
        SyncMode,
        SyncStatus, 
        ConflictResolution
    )

    # 数据模型类
    from .DataSynchronizer import (
        DataSource,
        SyncTask,
        SyncProgress,
        SyncLog
    )

    # 连接器类
    from .DataSynchronizer import (
        DataSourceConnector,
        DatabaseConnector,
        FileConnector
    )

    # 管理器类
    from .DataSynchronizer import (
        SecurityManager,
        ConflictDetector,
        ProgressMonitor,
        AuditLogger,
        ConfigurationManager
    )

    # 核心同步器类
    from .DataSynchronizer import (
        DataSynchronizer,
        DataSynchronizerTest
    )
    
except ImportError as e:
    # 如果导入失败，创建一个警告并提供基本信息
    _import_error = str(e)
    _import_success = False
    
    # 创建占位类
    class _ImportWarning:
        def __init__(self, error_msg):
            self._error_msg = error_msg
        
        def __getattr__(self, name):
            raise ImportError(f"无法导入T7模块: {self._error_msg}")
    
    # 设置所有导入的类为警告类
    SyncMode = SyncStatus = ConflictResolution = _ImportWarning(_import_error)
    DataSource = SyncTask = SyncProgress = SyncLog = _ImportWarning(_import_error)
    DataSourceConnector = DatabaseConnector = FileConnector = _ImportWarning(_import_error)
    SecurityManager = ConflictDetector = ProgressMonitor = AuditLogger = ConfigurationManager = _ImportWarning(_import_error)
    DataSynchronizer = DataSynchronizerTest = _ImportWarning(_import_error)

# ================================
# 常量定义
# ================================

# 同步模式常量
if _import_success:
    SYNC_MODES = {
        "REALTIME": SyncMode.REALTIME.value,
        "BATCH": SyncMode.BATCH.value, 
        "INCREMENTAL": SyncMode.INCREMENTAL.value,
        "FULL": SyncMode.FULL.value
    }
else:
    SYNC_MODES = {
        "REALTIME": "realtime",
        "BATCH": "batch", 
        "INCREMENTAL": "incremental",
        "FULL": "full"
    }

# 同步状态常量
if _import_success:
    SYNC_STATUSES = {
        "PENDING": SyncStatus.PENDING.value,
        "RUNNING": SyncStatus.RUNNING.value,
        "COMPLETED": SyncStatus.COMPLETED.value,
        "FAILED": SyncStatus.FAILED.value,
        "PAUSED": SyncStatus.PAUSED.value,
        "CANCELLED": SyncStatus.CANCELLED.value
    }
else:
    SYNC_STATUSES = {
        "PENDING": "pending",
        "RUNNING": "running",
        "COMPLETED": "completed",
        "FAILED": "failed",
        "PAUSED": "paused",
        "CANCELLED": "cancelled"
    }

# 冲突解决策略常量
if _import_success:
    CONFLICT_RESOLUTIONS = {
        "SOURCE_PRIORITY": ConflictResolution.SOURCE_PRIORITY.value,
        "TARGET_PRIORITY": ConflictResolution.TARGET_PRIORITY.value,
        "TIMESTAMP_PRIORITY": ConflictResolution.TIMESTAMP_PRIORITY.value,
        "MANUAL_RESOLUTION": ConflictResolution.MANUAL_RESOLUTION.value,
        "MERGE_DATA": ConflictResolution.MERGE_DATA.value
    }
else:
    CONFLICT_RESOLUTIONS = {
        "SOURCE_PRIORITY": "source_priority",
        "TARGET_PRIORITY": "target_priority",
        "TIMESTAMP_PRIORITY": "timestamp_priority",
        "MANUAL_RESOLUTION": "manual_resolution",
        "MERGE_DATA": "merge_data"
    }

# 数据源类型常量
DATA_SOURCE_TYPES = [
    "database",
    "file", 
    "api",
    "cloud_storage",
    "message_queue"
]

# 数据库类型常量
DATABASE_TYPES = [
    "sqlite",
    "postgresql",
    "mysql",
    "oracle",
    "sqlserver"
]

# 操作系统常量
PLATFORM_CONSTANTS = {
    "WINDOWS": "win32",
    "LINUX": "linux",
    "DARWIN": "darwin",
    "JAVA": "java"
}

# ================================
# 默认配置
# ================================

DEFAULT_CONFIG = {
    "version": "1.0.0",
    "data_sources": {},
    "sync_tasks": {},
    "global_settings": {
        "max_concurrent_tasks": 5,
        "default_batch_size": 1000,
        "default_timeout": 30,
        "enable_encryption": True,
        "log_level": "INFO",
        "audit_enabled": True,
        "auto_retry": True,
        "retry_max_attempts": 3,
        "retry_delay": 1.0,
        "compression_enabled": False,
        "cache_enabled": True,
        "connection_pool_size": 10,
        "max_workers": 4
    },
    "security_settings": {
        "require_authentication": False,
        "token_expiry": 3600,
        "allowed_ips": [],
        "encryption_algorithm": "AES-256",
        "ssl_verify": True,
        "max_login_attempts": 5,
        "lockout_duration": 300
    },
    "performance_settings": {
        "max_workers": 10,
        "connection_pool_size": 20,
        "cache_enabled": True,
        "compression_enabled": False,
        "buffer_size": 8192,
        "chunk_size": 1000,
        "memory_limit": "512MB",
        "disk_cache_size": "100MB"
    },
    "logging_settings": {
        "log_level": "INFO",
        "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "log_file": "sync.log",
        "log_max_size": "10MB",
        "log_backup_count": 5,
        "enable_console_output": True,
        "enable_file_output": True
    },
    "audit_settings": {
        "audit_enabled": True,
        "audit_log_dir": "logs",
        "audit_retention_days": 30,
        "audit_include_data": False,
        "audit_sensitive_fields": ["password", "token", "secret"]
    }
}

# 默认数据源配置模板
DEFAULT_DATA_SOURCE_TEMPLATES = {
    "sqlite": {
        "type": "database",
        "connection_info": {
            "type": "sqlite",
            "path": "database.db"
        },
        "schema": {
            "tables": []
        },
        "timeout": 30,
        "retry_count": 3,
        "retry_delay": 1.0
    },
    "postgresql": {
        "type": "database", 
        "connection_info": {
            "type": "postgresql",
            "host": "localhost",
            "port": 5432,
            "database": "mydb",
            "user": "user",
            "password": "password"
        },
        "schema": {
            "tables": []
        },
        "timeout": 30,
        "retry_count": 3,
        "retry_delay": 1.0
    },
    "mysql": {
        "type": "database",
        "connection_info": {
            "type": "mysql", 
            "host": "localhost",
            "port": 3306,
            "database": "mydb",
            "user": "user",
            "password": "password"
        },
        "schema": {
            "tables": []
        },
        "timeout": 30,
        "retry_count": 3,
        "retry_delay": 1.0
    },
    "file": {
        "type": "file",
        "connection_info": {
            "base_path": "./data"
        },
        "schema": {
            "tables": []
        },
        "timeout": 30,
        "retry_count": 3,
        "retry_delay": 1.0
    }
}

# 默认同步任务配置模板
DEFAULT_SYNC_TASK_TEMPLATES = {
    "full_sync": {
        "mode": "full",
        "batch_size": 1000,
        "max_workers": 4,
        "conflict_resolution": "source_priority",
        "enabled": True,
        "filters": None,
        "schedule": None
    },
    "incremental_sync": {
        "mode": "incremental",
        "batch_size": 500,
        "max_workers": 2,
        "conflict_resolution": "timestamp_priority",
        "enabled": True,
        "filters": None,
        "schedule": "0 */6 * * *"  # 每6小时执行一次
    },
    "realtime_sync": {
        "mode": "realtime",
        "batch_size": 100,
        "max_workers": 1,
        "conflict_resolution": "source_priority",
        "enabled": True,
        "filters": None,
        "schedule": None
    }
}

# ================================
# 便利函数
# ================================

def create_synchronizer(config_file: str = "sync_config.json") -> Any:
    """
    创建数据同步器实例的便利函数
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        DataSynchronizer: 数据同步器实例
    """
    if not _import_success:
        raise ImportError(f"无法创建同步器: {_import_error}")
    return DataSynchronizer(config_file)


def create_data_source(source_id: str, name: str, source_type: str, 
                      connection_info: dict, schema: dict = None) -> Any:
    """
    创建数据源配置的便利函数
    
    Args:
        source_id: 数据源ID
        name: 数据源名称
        source_type: 数据源类型
        connection_info: 连接信息
        schema: 表结构信息
        
    Returns:
        DataSource: 数据源配置对象
    """
    if not _import_success:
        raise ImportError(f"无法创建数据源: {_import_error}")
    if schema is None:
        schema = {"tables": []}
    
    return DataSource(
        id=source_id,
        name=name,
        type=source_type,
        connection_info=connection_info,
        schema=schema
    )


def create_sync_task(task_id: str, name: str, source_id: str, target_id: str,
                    mode: str, tables: list, **kwargs) -> Any:
    """
    创建同步任务配置的便利函数
    
    Args:
        task_id: 任务ID
        name: 任务名称
        source_id: 源数据源ID
        target_id: 目标数据源ID
        mode: 同步模式
        tables: 要同步的表列表
        **kwargs: 其他配置参数
        
    Returns:
        SyncTask: 同步任务配置对象
    """
    if not _import_success:
        raise ImportError(f"无法创建同步任务: {_import_error}")
    # 设置默认值
    default_kwargs = {
        "filters": None,
        "conflict_resolution": "source_priority",
        "batch_size": 1000,
        "max_workers": 4,
        "enabled": True,
        "schedule": None
    }
    default_kwargs.update(kwargs)
    
    return SyncTask(
        id=task_id,
        name=name,
        source_id=source_id,
        target_id=target_id,
        mode=SyncMode(mode),
        tables=tables,
        **default_kwargs
    )


def get_default_config() -> dict:
    """
    获取默认配置
    
    Returns:
        dict: 默认配置字典
    """
    return DEFAULT_CONFIG.copy()


def get_config_template(source_type: str) -> dict:
    """
    获取数据源配置模板
    
    Args:
        source_type: 数据源类型
        
    Returns:
        dict: 配置模板
    """
    return DEFAULT_DATA_SOURCE_TEMPLATES.get(source_type, {}).copy()


def get_sync_task_template(task_type: str) -> dict:
    """
    获取同步任务配置模板
    
    Args:
        task_type: 任务类型
        
    Returns:
        dict: 任务配置模板
    """
    return DEFAULT_SYNC_TASK_TEMPLATES.get(task_type, {}).copy()


def validate_config(config: dict) -> Tuple[bool, List[str]]:
    """
    验证配置是否有效
    
    Args:
        config: 配置字典
        
    Returns:
        tuple: (是否有效, 错误信息列表)
    """
    errors = []
    
    # 验证必需字段
    required_fields = ["version", "global_settings", "security_settings"]
    for field in required_fields:
        if field not in config:
            errors.append(f"缺少必需字段: {field}")
    
    # 验证全局设置
    if "global_settings" in config:
        global_settings = config["global_settings"]
        if "max_concurrent_tasks" in global_settings:
            if not isinstance(global_settings["max_concurrent_tasks"], int) or global_settings["max_concurrent_tasks"] <= 0:
                errors.append("max_concurrent_tasks必须是正整数")
        
        if "default_batch_size" in global_settings:
            if not isinstance(global_settings["default_batch_size"], int) or global_settings["default_batch_size"] <= 0:
                errors.append("default_batch_size必须是正整数")
    
    return len(errors) == 0, errors


def quick_start_demo() -> str:
    """
    快速开始演示
    
    Returns:
        str: 演示代码示例
    """
    demo_code = '''
# T7数据同步器快速开始示例

import asyncio
from T7 import (
    DataSynchronizer, DataSource, SyncTask, 
    SyncMode, ConflictResolution, create_synchronizer
)

async def main():
    # 1. 创建数据同步器
    synchronizer = DataSynchronizer("quickstart_config.json")
    
    # 2. 创建源数据源（SQLite数据库）
    source = DataSource(
        id="db_source",
        name="SQLite源数据库",
        type="database",
        connection_info={
            "type": "sqlite",
            "path": "source.db"
        },
        schema={"tables": ["users", "products"]}
    )
    
    # 3. 创建目标数据源（文件）
    target = DataSource(
        id="file_target",
        name="文件目标",
        type="file",
        connection_info={
            "base_path": "target_data"
        },
        schema={"tables": ["users", "products"]}
    )
    
    # 4. 添加数据源到配置管理器
    synchronizer.config_manager.add_data_source(source)
    synchronizer.config_manager.add_data_source(target)
    
    # 5. 创建同步任务
    sync_task = SyncTask(
        id="demo_sync",
        name="演示同步任务",
        source_id="db_source",
        target_id="file_target",
        mode=SyncMode.BATCH,
        tables=["users", "products"],
        batch_size=100,
        conflict_resolution=ConflictResolution.SOURCE_PRIORITY
    )
    
    # 6. 添加同步任务
    synchronizer.config_manager.add_sync_task(sync_task)
    
    # 7. 注册进度回调
    def on_progress(progress):
        print(f"同步进度: {progress.progress_percentage:.1f}% - {progress.current_table}")
    
    synchronizer.register_progress_callback("demo_sync", on_progress)
    
    # 8. 启动同步
    success = await synchronizer.start_sync("demo_sync")
    print(f"同步启动: {'成功' if success else '失败'}")
    
    # 9. 等待同步完成
    while synchronizer.get_sync_status("demo_sync"):
        await asyncio.sleep(1)
    
    # 10. 获取结果
    status = synchronizer.get_sync_status("demo_sync")
    if status:
        print(f"同步状态: {status.status.value}")
        print(f"处理记录: {status.processed_records}")
        print(f"成功记录: {status.success_records}")
    
    # 11. 清理资源
    await synchronizer.cleanup()

# 运行演示
if __name__ == "__main__":
    asyncio.run(main())
'''
    return demo_code


# ================================
# 工具函数
# ================================

def format_sync_stats(stats: dict) -> str:
    """
    格式化同步统计信息
    
    Args:
        stats: 性能统计字典
        
    Returns:
        str: 格式化的统计信息
    """
    if not stats:
        return "无统计数据"
    
    lines = []
    lines.append("=== 同步性能统计 ===")
    lines.append(f"总同步次数: {stats.get('total_syncs', 0)}")
    lines.append(f"成功同步: {stats.get('successful_syncs', 0)}")
    lines.append(f"失败同步: {stats.get('failed_syncs', 0)}")
    lines.append(f"总处理记录: {stats.get('total_records_processed', 0)}")
    
    avg_time = stats.get('average_sync_time', 0)
    if avg_time > 0:
        lines.append(f"平均同步时间: {avg_time:.2f}秒")
    
    # 计算成功率
    total = stats.get('total_syncs', 0)
    success = stats.get('successful_syncs', 0)
    if total > 0:
        success_rate = (success / total) * 100
        lines.append(f"成功率: {success_rate:.1f}%")
    
    return "\n".join(lines)


def validate_sync_task(task: Any) -> Tuple[bool, List[str]]:
    """
    验证同步任务配置
    
    Args:
        task: 同步任务对象
        
    Returns:
        tuple: (是否有效, 错误信息列表)
    """
    errors = []
    
    if not task.id:
        errors.append("任务ID不能为空")
    
    if not task.name:
        errors.append("任务名称不能为空")
    
    if not task.source_id:
        errors.append("源数据源ID不能为空")
    
    if not task.target_id:
        errors.append("目标数据源ID不能为空")
    
    if not task.tables:
        errors.append("必须指定要同步的表")
    
    if not isinstance(task.batch_size, int) or task.batch_size <= 0:
        errors.append("批次大小必须是正整数")
    
    if not isinstance(task.max_workers, int) or task.max_workers <= 0:
        errors.append("最大工作线程数必须是正整数")
    
    return len(errors) == 0, errors


def estimate_sync_time(record_count: int, records_per_second: float = 100.0) -> int:
    """
    估算同步时间
    
    Args:
        record_count: 记录数量
        records_per_second: 每秒处理记录数
        
    Returns:
        int: 预计秒数
    """
    if records_per_second <= 0:
        return 0
    
    return int(record_count / records_per_second)


# ================================
# 异常类定义
# ================================

class T7SyncError(Exception):
    """T7同步器基础异常类"""
    pass


class ConfigurationError(T7SyncError):
    """配置错误异常"""
    pass


class SyncTaskError(T7SyncError):
    """同步任务错误异常"""
    pass


class DataSourceError(T7SyncError):
    """数据源错误异常"""
    pass


class ConflictResolutionError(T7SyncError):
    """冲突解决错误异常"""
    pass


# ================================
# 快速入门指南
# ================================

QUICK_START_GUIDE = """
=== T7数据同步器快速入门指南 ===

一、安装和导入
--------------
from T7 import (
    DataSynchronizer, 
    DataSource, 
    SyncTask,
    SyncMode,
    ConflictResolution,
    create_synchronizer
)

二、基本使用流程
--------------
1. 创建数据同步器实例
2. 配置数据源（源和目标）
3. 创建同步任务
4. 注册进度回调（可选）
5. 启动同步
6. 监控进度
7. 获取结果
8. 清理资源

三、支持的同步模式
----------------
• REALTIME: 实时同步，适合小量数据的实时更新
• BATCH: 批量同步，适合大量数据的定期同步
• INCREMENTAL: 增量同步，只同步变更的数据
• FULL: 全量同步，同步所有数据

四、支持的冲突解决策略
--------------------
• SOURCE_PRIORITY: 源数据优先
• TARGET_PRIORITY: 目标数据优先  
• TIMESTAMP_PRIORITY: 时间戳优先
• MANUAL_RESOLUTION: 手动解决
• MERGE_DATA: 合并数据

五、支持的数据库类型
------------------
• SQLite
• PostgreSQL
• MySQL
• Oracle
• SQL Server

六、关键特性
----------
✓ 多数据源同步支持
✓ 实时和批量同步模式
✓ 智能数据冲突检测和解决
✓ 实时进度监控
✓ 完善的错误处理和重试机制
✓ 数据加密和安全控制
✓ 详细的审计日志
✓ 性能优化和资源管理
✓ 灵活的配置管理
✓ 丰富的API接口

七、示例代码
-----------
请运行: from T7 import quick_start_demo; print(quick_start_demo())

八、更多信息
----------
• 文档: https://github.com/t7/datasync
• 示例: https://github.com/t7/datasync/examples
• 报告问题: https://github.com/t7/datasync/issues
• 许可证: MIT
"""

# ================================
# 模块信息
# ================================

# 模块元信息
__all__ = [
    # 核心类
    'DataSynchronizer',
    'DataSource',
    'SyncTask',
    'SyncProgress',
    'SyncLog',
    
    # 连接器
    'DataSourceConnector',
    'DatabaseConnector', 
    'FileConnector',
    
    # 管理器
    'SecurityManager',
    'ConflictDetector',
    'ProgressMonitor',
    'AuditLogger',
    'ConfigurationManager',
    
    # 枚举
    'SyncMode',
    'SyncStatus',
    'ConflictResolution',
    
    # 测试
    'DataSynchronizerTest',
    
    # 便利函数
    'create_synchronizer',
    'create_data_source',
    'create_sync_task',
    'get_default_config',
    'get_config_template',
    'get_sync_task_template',
    'validate_config',
    'validate_sync_task',
    'format_sync_stats',
    'estimate_sync_time',
    'quick_start_demo',
    
    # 常量
    'SYNC_MODES',
    'SYNC_STATUSES', 
    'CONFLICT_RESOLUTIONS',
    'DATA_SOURCE_TYPES',
    'DATABASE_TYPES',
    'DEFAULT_CONFIG',
    'DEFAULT_DATA_SOURCE_TEMPLATES',
    'DEFAULT_SYNC_TASK_TEMPLATES',
    
    # 异常
    'T7SyncError',
    'ConfigurationError',
    'SyncTaskError', 
    'DataSourceError',
    'ConflictResolutionError',
    
    # 版本信息
    'VERSION_INFO',
    
    # 文档
    'QUICK_START_GUIDE'
]

# 模块简介
__doc__ = f"""
T7数据同步器 v{__version__}

{__description__}

主要功能：
• 多数据源同步 - 支持数据库、文件、API等多种数据源
• 实时同步和批量同步 - 支持实时流式同步和批量数据同步
• 数据冲突检测和解决 - 智能检测和解决数据冲突
• 数据同步进度监控 - 实时跟踪同步进度和状态
• 数据同步错误处理 - 完善的错误处理和重试机制
• 数据同步性能优化 - 性能优化和资源管理
• 数据同步安全控制 - 数据加密、访问控制等安全措施
• 数据同步日志和审计 - 详细的日志记录和审计跟踪
• 数据同步配置管理 - 灵活的配置文件管理

更多信息请查看 QUICK_START_GUIDE 或访问: {__url__}
"""

# ================================
# 模块初始化完成提示
# ================================

def _show_welcome():
    """显示欢迎信息"""
    print(f"✓ T7数据同步器 v{__version__} 导入成功")
    print(f"  文档: {__url__}")
    print(f"  快速开始: from T7 import QUICK_START_GUIDE; print(QUICK_START_GUIDE)")
    print(f"  演示代码: from T7 import quick_start_demo; print(quick_start_demo())")

# 在模块导入时显示欢迎信息（可选）
# _show_welcome()