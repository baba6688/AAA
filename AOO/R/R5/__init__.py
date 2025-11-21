#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R5恢复管理器 - 包初始化文件

提供全面的系统恢复解决方案，包括：
- 数据恢复：文件、数据库、配置恢复
- 系统恢复：操作系统、应用系统恢复
- 灾难恢复：灾难情况下的快速恢复
- 恢复计划：恢复流程和步骤计划
- 恢复验证：恢复后数据完整性验证
- 恢复监控：恢复进度和状态监控
- 恢复报告：恢复结果和统计报告
- 恢复测试：定期恢复演练和测试

作者: R5恢复管理器团队
版本: 1.0.0
"""

# 核心类
from .RecoveryManager import (
    # 主要类
    RecoveryManager,
    
    # 恢复模块
    DataRecovery,
    SystemRecovery,
    DisasterRecovery,
    
    # 恢复管理
    RecoveryPlan,
    RecoveryValidator,
    RecoveryMonitor,
    RecoveryReporter,
    RecoveryTester,
    
    # 恢复任务
    RecoveryTask,
    
    # 枚举类型
    RecoveryType,
    RecoveryStatus,
    Priority,
    
    # 便利函数
    create_recovery_plan,
    execute_recovery,
    validate_recovery,
    monitor_recovery_progress,
    run_recovery_test,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "R5恢复管理器团队"
__email__ = "support@r5recovery.com"
__description__ = "R5恢复管理器 - 全面的系统恢复解决方案"

# 包级别的默认配置
DEFAULT_CONFIG = {
    'recovery_dir': './recoveries',
    'backup_dir': './backups',
    'log_dir': './recovery_logs',
    'temp_dir': './recovery_temp',
    'max_concurrent_recoveries': 5,
    'validation': {
        'auto_validate': True,
        'deep_validation': True,
        'consistency_check': True
    },
    'monitoring': {
        'real_time_monitoring': True,
        'alert_on_failure': True,
        'progress_tracking': True
    },
    'testing': {
        'auto_test_scheduled': True,
        'test_frequency': 'weekly',
        'backup_before_test': True
    },
    'disaster_recovery': {
        'rto_target': 3600,  # 1小时
        'rpo_target': 300,   # 5分钟
        'failover_auto': False,
        'backup_validation': True
    }
}

# 恢复类型
RECOVERY_TYPES = [
    'DATA',          # 数据恢复
    'SYSTEM',        # 系统恢复
    'DISASTER',      # 灾难恢复
    'CONFIG',        # 配置恢复
    'APPLICATION',   # 应用恢复
    'DATABASE',      # 数据库恢复
    'FILE',          # 文件恢复
    'FULL',          # 完整恢复
]

# 恢复状态
RECOVERY_STATUS = [
    'PENDING',       # 待执行
    'RUNNING',       # 执行中
    'PAUSED',        # 暂停
    'COMPLETED',     # 已完成
    'FAILED',        # 执行失败
    'CANCELLED',     # 已取消
    'VALIDATED',     # 已验证
    'ROLLED_BACK',   # 已回滚
]

# 优先级级别
PRIORITY_LEVELS = [
    'CRITICAL',      # 紧急
    'HIGH',          # 高
    'MEDIUM',        # 中
    'LOW',           # 低
]

# 验证级别
VALIDATION_LEVELS = [
    'BASIC',         # 基础验证
    'STANDARD',      # 标准验证
    'DEEP',          # 深度验证
    'COMPREHENSIVE', # 全面验证
]

# 恢复模式
RECOVERY_MODES = [
    'AUTOMATIC',     # 自动恢复
    'MANUAL',        # 手动恢复
    'SEMI_AUTO',     # 半自动恢复
    'INTERACTIVE',   # 交互式恢复
]

# 灾难类型
DISASTER_TYPES = [
    'HARDWARE',      # 硬件故障
    'SOFTWARE',      # 软件故障
    'NETWORK',       # 网络故障
    'POWER',         # 电源故障
    'FIRE',          # 火灾
    'FLOOD',         # 水灾
    'EARTHQUAKE',    # 地震
    'CYBER_ATTACK',  # 网络攻击
    'HUMAN_ERROR',   # 人为错误
]

# 测试状态
TEST_STATUS = [
    'SCHEDULED',     # 已计划
    'RUNNING',       # 正在运行
    'PASSED',        # 通过
    'FAILED',        # 失败
    'SKIPPED',       # 跳过
    'CANCELLED',     # 取消
]

# 公开的API函数
__all__ = [
    # 核心类
    'RecoveryManager',
    
    # 恢复模块
    'DataRecovery',
    'SystemRecovery',
    'DisasterRecovery',
    
    # 恢复管理
    'RecoveryPlan',
    'RecoveryValidator',
    'RecoveryMonitor',
    'RecoveryReporter',
    'RecoveryTester',
    
    # 恢复任务
    'RecoveryTask',
    
    # 枚举类型
    'RecoveryType',
    'RecoveryStatus',
    'Priority',
    
    # 便利函数
    'create_recovery_plan',
    'execute_recovery',
    'validate_recovery',
    'monitor_recovery_progress',
    'run_recovery_test',
    
    # 常量
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    'DEFAULT_CONFIG',
    'RECOVERY_TYPES',
    'RECOVERY_STATUS',
    'PRIORITY_LEVELS',
    'VALIDATION_LEVELS',
    'RECOVERY_MODES',
    'DISASTER_TYPES',
    'TEST_STATUS',
]

# 快速入门指南
def quick_start():
    """
    快速入门指南
    
    返回一个包含基本使用示例的字符串。
    """
    return """
    R5恢复管理器快速入门
    ====================
    
    1. 创建恢复计划:
       ```python
       from R5 import create_recovery_plan
       
       plan = create_recovery_plan(
           recovery_type="DATA",
           priority="HIGH",
           steps=[
               {"action": "backup_restore", "target": "/data"},
               {"action": "validation", "level": "DEEP"}
           ]
       )
       ```
    
    2. 执行恢复:
       ```python
       from R5 import execute_recovery
       
       result = execute_recovery(
           recovery_id="recovery_001",
           backup_source="/backup/latest",
           target_location="/data"
       )
       ```
    
    3. 恢复验证:
       ```python
       from R5 import validate_recovery
       
       validation = validate_recovery(
           recovery_id="recovery_001",
           validation_level="COMPREHENSIVE"
       )
       ```
    
    4. 监控恢复进度:
       ```python
       from R5 import monitor_recovery_progress
       
       progress = monitor_recovery_progress(
           recovery_id="recovery_001"
       )
       ```
    
    5. 运行恢复测试:
       ```python
       from R5 import run_recovery_test
       
       test_result = run_recovery_test(
           test_scenario="disaster_recovery",
           backup_source="/backup/test"
       )
       ```
    
    更多信息请查看文档或运行测试文件。
    """

# 便利函数实现
def create_recovery_plan(recovery_type, priority='MEDIUM', steps=None, **kwargs):
    """
    创建恢复计划的便利函数
    """
    recovery_manager = RecoveryManager()
    return recovery_manager.create_plan(recovery_type, priority, steps, **kwargs)

def execute_recovery(recovery_id, backup_source, target_location, **kwargs):
    """
    执行恢复的便利函数
    """
    recovery_manager = RecoveryManager()
    return recovery_manager.execute(recovery_id, backup_source, target_location, **kwargs)

def validate_recovery(recovery_id, validation_level='STANDARD', **kwargs):
    """
    验证恢复的便利函数
    """
    recovery_validator = RecoveryValidator()
    return recovery_validator.validate(recovery_id, validation_level, **kwargs)

def monitor_recovery_progress(recovery_id):
    """
    监控恢复进度的便利函数
    """
    recovery_monitor = RecoveryMonitor()
    return recovery_monitor.get_progress(recovery_id)

def run_recovery_test(test_scenario, **kwargs):
    """
    运行恢复测试的便利函数
    """
    recovery_tester = RecoveryTester()
    return recovery_tester.run_test(test_scenario, **kwargs)

# 版本检查
def check_version():
    """检查版本信息"""
    import sys
    print(f"R5恢复管理器版本: {__version__}")
    print(f"Python版本: {sys.version}")
    print(f"作者: {__author__}")
    print(f"描述: {__description__}")

if __name__ == "__main__":
    check_version()
    print(quick_start())