#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R区运行时管理模块 - 完整导出接口
================================

这是一个全面的运行时管理系统，提供以下核心功能：

## 核心模块
- **R1 数据备份器 (DataBackup)**: 完整的数据备份解决方案
- **R2 配置备份器 (ConfigBackup)**: 系统配置备份和版本管理
- **R3 模型备份器 (ModelBackup)**: 机器学习模型备份和部署管理
- **R4 日志备份器 (LogBackup)**: 多类型日志备份、归档和检索
- **R5 恢复管理器 (RecoveryManager)**: 综合系统恢复解决方案
- **R6 版本控制器 (VersionController)**: 版本控制和分支管理
- **R7 灾难恢复器 (DisasterRecovery)**: 灾难检测和自动恢复
- **R8 归档管理器 (ArchiveManager)**: 智能数据归档和压缩存储
- **R9 备份状态聚合器 (BackupStatusAggregator)**: 备份状态监控和报告

## 主要特性
- 🗂️ **多类型备份**: 文件、数据库、配置、模型、日志
- 🔄 **智能恢复**: 自动故障检测和数据恢复
- 📊 **状态监控**: 实时备份状态监控和预警
- 📈 **版本控制**: 完整的版本管理和分支操作
- 🏗️ **灾难恢复**: 自动化灾难检测和应急响应
- 🗜️ **归档管理**: 高效的数据归档和压缩存储
- 📋 **报告系统**: 全面的备份状态和性能报告

## 使用方式
```python
# 方式1: 直接导入需要的模块
from R.R1 import DataBackup
from R.R5 import RecoveryManager

# 方式2: 导入所有功能
from R import *

# 方式3: 使用便捷函数
from R import create_data_backup, create_recovery_manager
```

作者: R区运行时管理团队
版本: 1.0.0
创建时间: 2025-11-06
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# 版本信息
__version__ = "1.0.0"
__author__ = "R区运行时管理团队"
__email__ = "runtime@backup-system.com"

# 导出的公共API
__all__ = [
    # 数据备份相关
    'DataBackup', 'BackupConfig', 'BackupStatus',
    'create_file_backup', 'create_database_backup', 'create_cloud_backup',
    
    # 配置备份相关
    'ConfigBackup', 'ConfigType', 'BackupRecord', 'ConfigMetadata',
    
    # 模型备份相关
    'ModelBackup', 'ModelStatus', 'ModelMetadata', 'DeploymentRecord',
    
    # 日志备份相关
    'LogBackup', 'LogType', 'LogBackup',
    
    # 恢复管理相关
    'RecoveryManager', 'RecoveryType', 'RecoveryStatus', 'Priority',
    'create_recovery_manager', 'quick_recover_file', 'quick_recover_database',
    
    # 版本控制相关
    'VersionController', 'PermissionLevel', 'Version', 'Branch',
    
    # 灾难恢复相关
    'DisasterRecovery', 'DisasterType', 'RecoveryStatus',
    'DEFAULT_CONFIG',
    
    # 归档管理相关
    'ArchiveManager', 'CompressionType', 'ArchiveStatus', 'ArchiveEntry',
    'create_archive_manager',
    
    # 状态聚合相关
    'BackupStatusAggregator', 'AlertLevel', 'BackupTaskInfo',
    
    # 便捷函数
    'quick_backup_and_recovery',
    'create_runtime_management_system',
    
    # 数据类
    'BackupTaskInfo', 'BackupModuleStatus', 'AlertInfo',
    'RecoveryTask', 'ArchiveRule',
    
    # 枚举类型
    'BackupStatus', 'ArchiveStatus', 'AlertLevel'
]

# ============================================================================
# R1 数据备份器导入和重新导出
# ============================================================================
try:
    from R1.DataBackup import (
        DataBackup as _DataBackup,
        BackupConfig, 
        BackupStatus,
        create_file_backup as _create_file_backup,
        create_database_backup as _create_database_backup,
        create_cloud_backup as _create_cloud_backup
    )
    
    # 重命名以避免命名冲突
    DataBackup = _DataBackup
    create_file_backup = _create_file_backup
    create_database_backup = _create_database_backup
    create_cloud_backup = _create_cloud_backup
    
except ImportError as e:
    print(f"警告: 导入R1数据备份器失败: {e}")
    DataBackup = None
    create_file_backup = None
    create_database_backup = None
    create_cloud_backup = None

# ============================================================================
# R2 配置备份器导入和重新导出
# ============================================================================
try:
    from R2.ConfigBackup import (
        ConfigBackup as _ConfigBackup,
        ConfigType,
        BackupRecord,
        ConfigMetadata
    )
    
    ConfigBackup = _ConfigBackup
    
except ImportError as e:
    print(f"警告: 导入R2配置备份器失败: {e}")
    ConfigBackup = None

# ============================================================================
# R3 模型备份器导入和重新导出
# ============================================================================
try:
    from R3.ModelBackup import (
        ModelBackup as _ModelBackup,
        ModelStatus,
        ModelMetadata,
        DeploymentRecord
    )
    
    ModelBackup = _ModelBackup
    
except ImportError as e:
    print(f"警告: 导入R3模型备份器失败: {e}")
    ModelBackup = None

# ============================================================================
# R4 日志备份器导入和重新导出
# ============================================================================
try:
    from R4.LogBackup import (
        LogBackup as _LogBackup,
        LogType
    )
    
    LogBackup = _LogBackup
    
except ImportError as e:
    print(f"警告: 导入R4日志备份器失败: {e}")
    LogBackup = None

# ============================================================================
# R5 恢复管理器导入和重新导出
# ============================================================================
try:
    from R5.RecoveryManager import (
        RecoveryManager as _RecoveryManager,
        RecoveryType,
        RecoveryStatus,
        Priority,
        create_recovery_manager as _create_recovery_manager,
        quick_recover_file as _quick_recover_file,
        quick_recover_database as _quick_recover_database
    )
    
    RecoveryManager = _RecoveryManager
    create_recovery_manager = _create_recovery_manager
    quick_recover_file = _quick_recover_file
    quick_recover_database = _quick_recover_database
    
except ImportError as e:
    print(f"警告: 导入R5恢复管理器失败: {e}")
    RecoveryManager = None
    create_recovery_manager = None
    quick_recover_file = None
    quick_recover_database = None

# ============================================================================
# R6 版本控制器导入和重新导出
# ============================================================================
try:
    from R6.VersionController import (
        VersionController as _VersionController,
        PermissionLevel,
        Version,
        Branch
    )
    
    VersionController = _VersionController
    
except ImportError as e:
    print(f"警告: 导入R6版本控制器失败: {e}")
    VersionController = None

# ============================================================================
# R7 灾难恢复器导入和重新导出
# ============================================================================
try:
    from R7.DisasterRecovery import (
        DisasterRecovery as _DisasterRecovery,
        DisasterType,
        RecoveryStatus as _DRRecoveryStatus,
        DEFAULT_CONFIG
    )
    
    # 避免与R5的RecoveryStatus冲突
    from R5.RecoveryManager import RecoveryStatus as R5RecoveryStatus
    DisasterRecovery = _DisasterRecovery
    DRRecoveryStatus = _DRRecoveryStatus
    
except ImportError as e:
    print(f"警告: 导入R7灾难恢复器失败: {e}")
    DisasterRecovery = None
    DRRecoveryStatus = None

# ============================================================================
# R8 归档管理器导入和重新导出
# ============================================================================
try:
    from R8.ArchiveManager import (
        ArchiveManager as _ArchiveManager,
        CompressionType,
        ArchiveStatus,
        ArchiveEntry,
        create_archive_manager as _create_archive_manager
    )
    
    ArchiveManager = _ArchiveManager
    create_archive_manager = _create_archive_manager
    
except ImportError as e:
    print(f"警告: 导入R8归档管理器失败: {e}")
    ArchiveManager = None
    create_archive_manager = None

# ============================================================================
# R9 备份状态聚合器导入和重新导出
# ============================================================================
try:
    from R9.BackupStatusAggregator import (
        BackupStatusAggregator as _BackupStatusAggregator,
        AlertLevel,
        BackupTaskInfo,
        BackupModuleStatus,
        AlertInfo,
        BackupStatus as _AggregatorBackupStatus
    )
    
    # 避免与R1的BackupStatus冲突
    BackupStatusAggregator = _BackupStatusAggregator
    AggregatorBackupStatus = _AggregatorBackupStatus
    
except ImportError as e:
    print(f"警告: 导入R9备份状态聚合器失败: {e}")
    BackupStatusAggregator = None
    AggregatorBackupStatus = None

# ============================================================================
# 数据类导入
# ============================================================================
try:
    from R5.RecoveryManager import RecoveryTask
except ImportError:
    RecoveryTask = None

try:
    from R8.ArchiveManager import ArchiveRule
except ImportError:
    ArchiveRule = None

# ============================================================================
# 便捷函数
# ============================================================================

def quick_backup_and_recovery(source_path: str, backup_path: str, 
                             recovery_path: str = None, 
                             config: dict = None) -> bool:
    """
    快速备份和恢复的便捷函数
    
    Args:
        source_path: 源文件路径
        backup_path: 备份存储路径
        recovery_path: 恢复目标路径
        config: 配置参数
    
    Returns:
        bool: 是否成功
    """
    try:
        # 执行备份
        if DataBackup and create_file_backup:
            backup_config = BackupConfig(
                backup_id=f"quick_backup_{int(__import__('time').time())}",
                source_path=source_path,
                backup_path=backup_path,
                compression="gzip",
                encryption=False
            )
            
            backup_system = DataBackup(config or {})
            backup_result = backup_system.create_backup(backup_config)
            
            if backup_result.status != 'success':
                print(f"备份失败: {backup_result.error_message}")
                return False
            
            # 执行恢复
            if recovery_path and RecoveryManager and create_recovery_manager:
                recovery_manager = create_recovery_manager(config)
                success = recovery_manager.recover_file(source_path, recovery_path)
                recovery_manager.shutdown()
                return success
            
            return True
            
    except Exception as e:
        print(f"快速备份恢复失败: {e}")
        return False
    
    return False


def create_runtime_management_system(base_path: str = "./runtime_system",
                                   config: dict = None) -> dict:
    """
    创建完整的运行时管理系统
    
    Args:
        base_path: 系统根目录
        config: 配置参数
    
    Returns:
        dict: 包含所有组件的系统字典
    """
    system = {
        'base_path': base_path,
        'data_backup': None,
        'config_backup': None,
        'model_backup': None,
        'log_backup': None,
        'recovery_manager': None,
        'version_controller': None,
        'disaster_recovery': None,
        'archive_manager': None,
        'status_aggregator': None,
        'initialized': False
    }
    
    try:
        # 创建目录结构
        os.makedirs(base_path, exist_ok=True)
        
        # 初始化各个组件
        if DataBackup:
            system['data_backup'] = DataBackup(config or {})
        
        if ConfigBackup:
            system['config_backup'] = ConfigBackup(os.path.join(base_path, 'config_backups'))
        
        if ModelBackup:
            system['model_backup'] = ModelBackup(os.path.join(base_path, 'model_backups'))
        
        if LogBackup:
            system['log_backup'] = LogBackup(os.path.join(base_path, 'log_backup_config.json'))
        
        if RecoveryManager:
            system['recovery_manager'] = create_recovery_manager(config)
        
        if VersionController:
            system['version_controller'] = VersionController(os.path.join(base_path, 'repository'))
        
        if DisasterRecovery:
            system['disaster_recovery'] = DisasterRecovery(DEFAULT_CONFIG or {})
        
        if ArchiveManager:
            system['archive_manager'] = create_archive_manager(os.path.join(base_path, 'archives'))
        
        if BackupStatusAggregator:
            system['status_aggregator'] = BackupStatusAggregator(os.path.join(base_path, 'status.db'))
        
        system['initialized'] = True
        
    except Exception as e:
        print(f"创建运行时管理系统失败: {e}")
    
    return system


def get_system_info() -> dict:
    """获取R区系统信息"""
    return {
        'name': 'R区运行时管理模块',
        'version': __version__,
        'author': __author__,
        'components': {
            'data_backup': DataBackup is not None,
            'config_backup': ConfigBackup is not None,
            'model_backup': ModelBackup is not None,
            'log_backup': LogBackup is not None,
            'recovery_manager': RecoveryManager is not None,
            'version_controller': VersionController is not None,
            'disaster_recovery': DisasterRecovery is not None,
            'archive_manager': ArchiveManager is not None,
            'status_aggregator': BackupStatusAggregator is not None
        },
        'total_components': 9,
        'available_components': sum([
            DataBackup is not None,
            ConfigBackup is not None,
            ModelBackup is not None,
            LogBackup is not None,
            RecoveryManager is not None,
            VersionController is not None,
            DisasterRecovery is not None,
            ArchiveManager is not None,
            BackupStatusAggregator is not None
        ])
    }


def print_system_status():
    """打印系统状态"""
    info = get_system_info()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    R区运行时管理系统                          ║
║                    版本 {info['version']} ({info['author']})                    ║
╠══════════════════════════════════════════════════════════════╣""")
    
    for component, available in info['components'].items():
        status = "✅ 已加载" if available else "❌ 未加载"
        component_name = {
            'data_backup': 'R1 数据备份器',
            'config_backup': 'R2 配置备份器',
            'model_backup': 'R3 模型备份器',
            'log_backup': 'R4 日志备份器',
            'recovery_manager': 'R5 恢复管理器',
            'version_controller': 'R6 版本控制器',
            'disaster_recovery': 'R7 灾难恢复器',
            'archive_manager': 'R8 归档管理器',
            'status_aggregator': 'R9 状态聚合器'
        }.get(component, component)
        
        print(f"║ {component_name:<35} {status:<15} ║")
    
    print(f"╠══════════════════════════════════════════════════════════════╣")
    print(f"║ 可用组件: {info['available_components']}/{info['total_components']}                                       ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    
    if info['available_components'] == info['total_components']:
        print("🎉 所有组件已成功加载！")
    else:
        print(f"⚠️  还有 {info['total_components'] - info['available_components']} 个组件未加载")


# ============================================================================
# 模块初始化
# ============================================================================

# 打印系统状态（仅在首次导入时）
if not hasattr(sys, '_r_module_loaded'):
    sys._r_module_loaded = True
    print_system_status()

# 导出便捷函数到__all__
__all__.extend([
    'quick_backup_and_recovery',
    'create_runtime_management_system',
    'get_system_info',
    'print_system_status',
    'RecoveryTask',
    'ArchiveRule'
])

# 清理导入过程中的临时变量
try:
    del _DataBackup, _create_file_backup, _create_database_backup, _create_cloud_backup
    del _ConfigBackup
    del _ModelBackup
    del _LogBackup
    del _RecoveryManager, _create_recovery_manager, _quick_recover_file, _quick_recover_database
    del _VersionController
    del _DisasterRecovery, _DRRecoveryStatus
    del _ArchiveManager, _create_archive_manager
    del _BackupStatusAggregator, _AggregatorBackupStatus
except NameError:
    pass

# 文档化模块
__doc__ = """
R区运行时管理模块 - 完整的企业级备份恢复解决方案

该模块提供了一个全面的运行时管理系统，包括：

核心功能：
- 多类型数据备份（文件、数据库、配置、模型、日志）
- 智能恢复管理（自动故障检测和数据恢复）
- 版本控制和分支管理
- 灾难检测和自动恢复
- 数据归档和压缩存储
- 备份状态监控和报告

使用方法：
1. 直接导入需要的组件
2. 使用便捷函数快速操作
3. 创建完整的运行时管理系统

示例：
```python
from R import DataBackup, RecoveryManager, quick_backup_and_recovery

# 快速备份和恢复
success = quick_backup_and_recovery(
    source_path="/path/to/data",
    backup_path="/backup/location",
    recovery_path="/recovery/location"
)

# 创建完整系统
system = create_runtime_management_system("./my_runtime_system")
if system['initialized']:
    print("系统创建成功！")
```
"""