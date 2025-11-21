#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
S1调度服务 - 包初始化文件

提供全面的任务调度和执行管理功能，包括：
- 任务调度和定时执行
- 任务队列管理
- 并发控制
- 任务状态监控
- 告警管理
- 任务日志记录
- 任务重试机制
- 优先级调度

作者: S1调度服务团队
版本: 1.0.0
"""

# 核心类导入
from .SchedulerService import (
    # 主要服务类
    SchedulerService,
    
    # 任务相关类
    ScheduledTask,
    TaskResult,
    TaskStatus,
    TaskPriority,
    
    # 组件类
    TaskLogger,
    AlertManager,
    TaskQueue,
    
    # 便利函数
    create_simple_task,
    demo_task
)

# 版本信息
__version__ = "1.0.0"
__author__ = "S1调度服务团队"
__email__ = "support@s1scheduler.com"
__description__ = "S1调度服务 - 全面的任务调度和执行管理解决方案"

# 包级别的默认配置
DEFAULT_CONFIG = {
    'max_workers': 10,
    'task_timeout': 3600,
    'retry_attempts': 3,
    'retry_delay': 60,
    'queue_size': 1000,
    'log_level': 'INFO',
    'alert_enabled': True,
    'cleanup_interval': 3600,
    'max_concurrent_tasks': 5,
    'default_priority': 'NORMAL'
}

# 任务状态
TASK_STATUS = [
    'PENDING',         # 等待执行
    'RUNNING',         # 正在执行
    'COMPLETED',       # 执行完成
    'FAILED',          # 执行失败
    'CANCELLED',       # 已取消
    'RETRYING',        # 重试中
]

# 任务优先级
TASK_PRIORITY = [
    'LOW',             # 低优先级
    'NORMAL',          # 普通优先级
    'HIGH',            # 高优先级
    'CRITICAL',        # 关键优先级
]

# 调度类型
SCHEDULE_TYPES = [
    'ONCE',            # 单次执行
    'INTERVAL',        # 间隔执行
    'CRON',            # Cron表达式
    'DELAYED',         # 延迟执行
]

# 告警级别
ALERT_LEVELS = [
    'INFO',            # 信息
    'WARNING',         # 警告
    'ERROR',           # 错误
    'CRITICAL',        # 严重
]

# 公开的API函数
__all__ = [
    # 主要服务类
    'SchedulerService',
    
    # 任务相关类
    'ScheduledTask',
    'TaskResult',
    'TaskStatus',
    'TaskPriority',
    
    # 组件类
    'TaskLogger',
    'AlertManager',
    'TaskQueue',
    
    # 便利函数
    'create_simple_task',
    'demo_task',
    
    # 常量
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    'DEFAULT_CONFIG',
    'TASK_STATUS',
    'TASK_PRIORITY',
    'SCHEDULE_TYPES',
    'ALERT_LEVELS',
]

# 快速入门指南
def quick_start():
    """
    快速入门指南
    
    返回一个包含基本使用示例的字符串。
    """
    return """
    S1调度服务快速入门
    =================
    
    1. 创建简单任务:
       ```python
       from S1 import create_simple_task, TaskPriority
       
       def my_task():
           print("任务执行中...")
       
       task = create_simple_task(
           name="my_task",
           func=my_task,
           interval=60,  # 每60秒执行一次
           priority=TaskPriority.NORMAL
       )
       ```
    
    2. 使用调度器:
       ```python
       from S1 import SchedulerService, TaskStatus
       
       scheduler = SchedulerService()
       scheduler.add_task(task)
       scheduler.start()
       
       # 检查任务状态
       status = scheduler.get_task_status(task.task_id)
       ```
    
    3. 任务队列管理:
       ```python
       from S1 import TaskQueue, TaskPriority
       
       queue = TaskQueue(max_size=100)
       queue.add_task(task, priority=TaskPriority.HIGH)
       ```
    
    4. 告警管理:
       ```python
       from S1 import AlertManager, ALERT_LEVELS
       
       alert_manager = AlertManager()
       alert_manager.add_rule(
           name="task_failed",
           condition="status == 'FAILED'",
           level=ALERT_LEVELS['ERROR']
       )
       ```
    
    更多信息请查看文档或运行测试文件。
    """

# 便利函数
def create_scheduler(max_workers=10, **kwargs):
    """
    创建调度器的便利函数
    """
    return SchedulerService(max_workers=max_workers, **kwargs)

def create_task(name, func, **kwargs):
    """
    创建任务的便利函数
    """
    return create_simple_task(name, func, **kwargs)

# 版本检查
def check_version():
    """检查版本信息"""
    import sys
    print(f"S1调度服务版本: {__version__}")
    print(f"Python版本: {sys.version}")
    print(f"作者: {__author__}")
    print(f"描述: {__description__}")

if __name__ == "__main__":
    check_version()
    print(quick_start())