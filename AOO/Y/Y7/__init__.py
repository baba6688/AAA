"""
Y7存储备份器包
一个功能完整的存储备份系统，支持数据备份、恢复、管理等功能
"""

from .StorageBackup import (
    StorageBackup,
    BackupConfig,
    BackupTask,
    BackupManager,
    BackupRecovery,
    BackupStatistics,
    BackupValidator,
    BackupReport
)

__version__ = "1.0.0"
__author__ = "Y7 Storage Backup Team"

__all__ = [
    'StorageBackup',
    'BackupConfig', 
    'BackupTask',
    'BackupManager',
    'BackupRecovery',
    'BackupStatistics',
    'BackupValidator',
    'BackupReport'
]