"""
S区S8维护服务模块 - 完整导出接口

提供完整的系统维护、数据清理、性能优化等功能

版本信息:
    S8-MaintenanceService: 1.0.0
    作者: S区开发团队
    更新时间: 2025-11-13

模块包含:
    - TaskStatus: 任务状态枚举
    - TaskType: 任务类型枚举
    - MaintenanceTask: 维护任务数据类
    - MaintenancePlan: 维护计划数据类
    - MaintenanceService: S8维护服务主类

快速开始:
    from S.S8 import MaintenanceService, TaskType
    
    # 创建维护服务
    service = MaintenanceService()
    
    # 创建清理任务
    task_id = service.create_maintenance_task(
        TaskType.SYSTEM_CLEANUP,
        "每日系统清理",
        "清理临时文件和系统缓存"
    )
    
    # 执行任务
    service.execute_task(task_id)
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "S区开发团队"
__last_update__ = "2025-11-13"

# 主要类导入
from .MaintenanceService import (
    TaskStatus,
    TaskType, 
    MaintenanceTask,
    MaintenancePlan,
    MaintenanceService
)

# 便利函数
from typing import Dict, List, Optional, Any
import os
import json
import time
from pathlib import Path

def create_maintenance_service(config_path: str = "maintenance_config.json") -> MaintenanceService:
    """
    创建维护服务实例的便利函数
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        MaintenanceService实例
    """
    return MaintenanceService(config_path)

def quick_system_cleanup(service: Optional[MaintenanceService] = None) -> str:
    """
    快速系统清理的便利函数
    
    Args:
        service: 维护服务实例，如果为None则创建新的实例
        
    Returns:
        执行的任务ID
    """
    if service is None:
        service = create_maintenance_service()
    
    task_id = service.create_maintenance_task(
        TaskType.SYSTEM_CLEANUP,
        "快速系统清理",
        "执行快速系统清理，清理临时文件和缓存"
    )
    service.execute_task(task_id)
    return task_id

def quick_data_cleanup(service: Optional[MaintenanceService] = None) -> str:
    """
    快速数据清理的便利函数
    
    Args:
        service: 维护服务实例，如果为None则创建新的实例
        
    Returns:
        执行的任务ID
    """
    if service is None:
        service = create_maintenance_service()
    
    task_id = service.create_maintenance_task(
        TaskType.DATA_CLEANUP,
        "快速数据清理",
        "执行快速数据清理，清理过期数据"
    )
    service.execute_task(task_id)
    return task_id

def quick_performance_optimization(service: Optional[MaintenanceService] = None) -> str:
    """
    快速性能优化的便利函数
    
    Args:
        service: 维护服务实例，如果为None则创建新的实例
        
    Returns:
        执行的任务ID
    """
    if service is None:
        service = create_maintenance_service()
    
    task_id = service.create_maintenance_task(
        TaskType.PERFORMANCE_OPTIMIZATION,
        "快速性能优化",
        "执行快速性能优化，包括内存清理和数据库优化"
    )
    service.execute_task(task_id)
    return task_id

def full_maintenance_sequence(service: Optional[MaintenanceService] = None, 
                            include_backup: bool = True) -> List[str]:
    """
    执行完整维护序列的便利函数
    
    Args:
        service: 维护服务实例，如果为None则创建新的实例
        include_backup: 是否包含备份维护
        
    Returns:
        执行的任务ID列表
    """
    if service is None:
        service = create_maintenance_service()
    
    task_ids = []
    
    # 数据清理
    task_ids.append(quick_data_cleanup(service))
    
    # 系统清理
    task_ids.append(quick_system_cleanup(service))
    
    # 性能优化
    task_ids.append(quick_performance_optimization(service))
    
    # 数据库优化
    task_id = service.create_maintenance_task(
        TaskType.DATABASE_OPTIMIZATION,
        "数据库优化",
        "执行数据库VACUUM和ANALYZE优化"
    )
    service.execute_task(task_id)
    task_ids.append(task_id)
    
    # 日志清理
    task_id = service.create_maintenance_task(
        TaskType.LOG_CLEANUP,
        "日志清理",
        "清理过期的日志文件"
    )
    service.execute_task(task_id)
    task_ids.append(task_id)
    
    # 备份维护（可选）
    if include_backup:
        task_id = service.create_maintenance_task(
            TaskType.BACKUP_MAINTENANCE,
            "备份维护",
            "执行备份维护和清理过期备份"
        )
        service.execute_task(task_id)
        task_ids.append(task_id)
    
    return task_ids

# 默认配置常量
DEFAULT_CONFIG = {
    "log_level": "INFO",
    "log_file": "maintenance.log",
    "max_log_size": 10485760,  # 10MB
    "backup_dir": "./backups",
    "temp_dir": "./temp",
    "log_dir": "./logs",
    "db_optimization": {
        "vacuum_interval": 86400,  # 24小时
        "analyze_interval": 3600   # 1小时
    },
    "cleanup": {
        "temp_file_age": 86400,    # 24小时
        "log_file_age": 604800,    # 7天
        "backup_retention": 30     # 30天
    },
    "performance": {
        "memory_threshold": 80,    # 80%
        "disk_threshold": 85,      # 85%
        "cpu_threshold": 90        # 90%
    }
}

# 任务类型描述映射
TASK_TYPE_DESCRIPTIONS = {
    TaskType.SYSTEM_CLEANUP: "系统清理 - 清理临时文件和系统缓存",
    TaskType.DATA_CLEANUP: "数据清理 - 清理过期数据和记录",
    TaskType.PERFORMANCE_OPTIMIZATION: "性能优化 - 内存、磁盘和数据库优化",
    TaskType.BACKUP_MAINTENANCE: "备份维护 - 创建备份和清理过期备份",
    TaskType.UPDATE_MANAGEMENT: "更新管理 - 检查和更新系统组件",
    TaskType.LOG_CLEANUP: "日志清理 - 清理过期的日志文件",
    TaskType.DATABASE_OPTIMIZATION: "数据库优化 - VACUUM、ANALYZE和索引重建"
}

# 任务状态描述映射
STATUS_DESCRIPTIONS = {
    TaskStatus.PENDING: "等待中",
    TaskStatus.RUNNING: "执行中",
    TaskStatus.COMPLETED: "已完成",
    TaskStatus.FAILED: "执行失败",
    TaskStatus.CANCELLED: "已取消"
}

def get_task_type_description(task_type: TaskType) -> str:
    """
    获取任务类型描述
    
    Args:
        task_type: 任务类型
        
    Returns:
        任务类型描述字符串
    """
    return TASK_TYPE_DESCRIPTIONS.get(task_type, "未知任务类型")

def get_status_description(status: TaskStatus) -> str:
    """
    获取任务状态描述
    
    Args:
        status: 任务状态
        
    Returns:
        任务状态描述字符串
    """
    return STATUS_DESCRIPTIONS.get(status, "未知状态")

def create_default_config_file(config_path: str = "maintenance_config.json"):
    """
    创建默认配置文件
    
    Args:
        config_path: 配置文件路径
    """
    config_dir = os.path.dirname(config_path)
    if config_dir and not os.path.exists(config_dir):
        os.makedirs(config_dir, exist_ok=True)
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_CONFIG, f, indent=4, ensure_ascii=False)

def get_system_info(service: Optional[MaintenanceService] = None) -> Dict[str, Any]:
    """
    获取系统维护信息的便利函数
    
    Args:
        service: 维护服务实例，如果为None则创建新的实例
        
    Returns:
        系统信息字典
    """
    if service is None:
        service = create_maintenance_service()
    
    import platform
    import sys
    
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": sys.version
        },
        "service_status": service.get_service_status(),
        "config_path": service.config_path,
        "db_path": service.tasks_db_path
    }

def export_maintenance_report(service: Optional[MaintenanceService] = None, 
                            days: int = 7, 
                            output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    导出维护报告的便利函数
    
    Args:
        service: 维护服务实例，如果为None则创建新的实例
        days: 报告天数
        output_file: 输出文件路径，如果为None则不保存文件
        
    Returns:
        维护报告字典
    """
    if service is None:
        service = create_maintenance_service()
    
    report = service.generate_maintenance_report(days)
    
    if output_file:
        report_dir = os.path.dirname(output_file)
        if report_dir and not os.path.exists(report_dir):
            os.makedirs(report_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=4, ensure_ascii=False)
    
    return report

# 预定义的维护计划
PREDEFINED_PLANS = {
    "daily_maintenance": {
        "name": "日常维护",
        "description": "每日执行的常规维护任务",
        "schedule": "0 2 * * *",  # 每天凌晨2点
        "tasks": [
            {
                "task_type": "data_cleanup",
                "name": "数据清理",
                "description": "清理过期数据"
            },
            {
                "task_type": "log_cleanup", 
                "name": "日志清理",
                "description": "清理过期日志"
            }
        ]
    },
    "weekly_maintenance": {
        "name": "周度维护",
        "description": "每周执行的深度维护任务",
        "schedule": "0 3 * * 0",  # 每周日凌晨3点
        "tasks": [
            {
                "task_type": "system_cleanup",
                "name": "系统清理",
                "description": "清理系统临时文件"
            },
            {
                "task_type": "performance_optimization",
                "name": "性能优化", 
                "description": "执行性能优化"
            },
            {
                "task_type": "database_optimization",
                "name": "数据库优化",
                "description": "执行数据库优化"
            },
            {
                "task_type": "backup_maintenance",
                "name": "备份维护",
                "description": "创建和维护备份"
            }
        ]
    },
    "monthly_maintenance": {
        "name": "月度维护",
        "description": "每月执行的全面维护任务",
        "schedule": "0 4 1 * *",  # 每月1日凌晨4点
        "tasks": [
            {
                "task_type": "system_cleanup",
                "name": "全面系统清理",
                "description": "执行全面的系统清理"
            },
            {
                "task_type": "performance_optimization",
                "name": "全面性能优化",
                "description": "执行全面的性能优化"
            },
            {
                "task_type": "database_optimization",
                "name": "数据库深度优化",
                "description": "执行数据库深度优化"
            },
            {
                "task_type": "backup_maintenance",
                "name": "月度备份",
                "description": "创建月度备份"
            },
            {
                "task_type": "update_management",
                "name": "更新检查",
                "description": "检查系统更新"
            }
        ]
    }
}

def create_predefined_plan(service: MaintenanceService, plan_name: str, plan_id: str = None) -> str:
    """
    创建预定义维护计划的便利函数
    
    Args:
        service: 维护服务实例
        plan_name: 预定义计划名称（daily_maintenance, weekly_maintenance, monthly_maintenance）
        plan_id: 计划ID，如果为None则自动生成
        
    Returns:
        创建的计划ID
    """
    if plan_name not in PREDEFINED_PLANS:
        raise ValueError(f"未知的预定义计划: {plan_name}")
    
    plan_config = PREDEFINED_PLANS[plan_name]
    
    if plan_id is None:
        plan_id = f"{plan_name}_{int(time.time())}"
    
    task_configs = plan_config["tasks"]
    
    return service.create_maintenance_plan(
        plan_config["name"],
        plan_config["description"],
        task_configs,
        plan_config["schedule"]
    )

# 常用常量
MAX_LOG_SIZE = 10 * 1024 * 1024  # 10MB
DEFAULT_BACKUP_RETENTION_DAYS = 30
DEFAULT_TEMP_FILE_AGE_DAYS = 1
DEFAULT_LOG_FILE_AGE_DAYS = 7
DEFAULT_MEMORY_THRESHOLD = 80
DEFAULT_DISK_THRESHOLD = 85
DEFAULT_CPU_THRESHOLD = 90

# 错误消息常量
ERROR_MESSAGES = {
    "TASK_NOT_FOUND": "任务不存在",
    "TASK_ALREADY_RUNNING": "任务已在运行中", 
    "TASK_EXECUTION_FAILED": "任务执行失败",
    "PLAN_NOT_FOUND": "维护计划不存在",
    "CONFIG_LOAD_FAILED": "配置文件加载失败",
    "DATABASE_ERROR": "数据库操作错误",
    "INSUFFICIENT_SPACE": "磁盘空间不足",
    "PERMISSION_DENIED": "权限不足"
}

# 快速入门指南
QUICK_START_GUIDE = """
=== S8维护服务快速入门 ===

1. 基本使用:
   from S.S8 import MaintenanceService, TaskType
   
   # 创建服务实例
   service = MaintenanceService()
   
   # 创建任务
   task_id = service.create_maintenance_task(
       TaskType.SYSTEM_CLEANUP,
       "系统清理",
       "清理临时文件"
   )
   
   # 执行任务
   service.execute_task(task_id)

2. 使用便利函数:
   from S.S8 import quick_system_cleanup, full_maintenance_sequence
   
   # 快速系统清理
   task_id = quick_system_cleanup()
   
   # 完整维护序列
   task_ids = full_maintenance_sequence()

3. 创建维护计划:
   service = MaintenanceService()
   plan_id = service.create_maintenance_plan(
       "每日清理",
       "每日系统维护",
       [{"task_type": "system_cleanup", "name": "清理", "description": "清理文件"}],
       "0 2 * * *"  # 每天凌晨2点
   )

4. 生成维护报告:
   from S.S8 import export_maintenance_report
   
   report = export_maintenance_report(days=7, output_file="maintenance_report.json")

5. 配置管理:
   from S.S8 import create_default_config_file
   
   # 创建默认配置文件
   create_default_config_file("my_config.json")
   
   # 使用自定义配置
   service = MaintenanceService("my_config.json")

更多信息请查看各类的文档字符串。
"""

def print_quick_start_guide():
    """打印快速入门指南"""
    print(QUICK_START_GUIDE)

# 模块级便捷访问
get_maintenance_service = create_maintenance_service

# 导出的公共接口
__all__ = [
    # 版本信息
    "__version__",
    "__author__", 
    "__last_update__",
    
    # 主要类
    "TaskStatus",
    "TaskType",
    "MaintenanceTask", 
    "MaintenancePlan",
    "MaintenanceService",
    
    # 便利函数
    "create_maintenance_service",
    "quick_system_cleanup",
    "quick_data_cleanup", 
    "quick_performance_optimization",
    "full_maintenance_sequence",
    "create_default_config_file",
    "get_system_info",
    "export_maintenance_report",
    "create_predefined_plan",
    "get_maintenance_service",
    
    # 常量
    "DEFAULT_CONFIG",
    "TASK_TYPE_DESCRIPTIONS",
    "STATUS_DESCRIPTIONS",
    "PREDEFINED_PLANS",
    "ERROR_MESSAGES",
    "MAX_LOG_SIZE",
    "DEFAULT_BACKUP_RETENTION_DAYS",
    "DEFAULT_TEMP_FILE_AGE_DAYS",
    "DEFAULT_LOG_FILE_AGE_DAYS", 
    "DEFAULT_MEMORY_THRESHOLD",
    "DEFAULT_DISK_THRESHOLD",
    "DEFAULT_CPU_THRESHOLD",
    
    # 工具函数
    "get_task_type_description",
    "get_status_description",
    "print_quick_start_guide"
]

# 模块初始化日志
import logging

logger = logging.getLogger(__name__)
logger.info(f"S8维护服务模块已加载 - 版本: {__version__}")
logger.info("使用 print_quick_start_guide() 查看快速入门指南")