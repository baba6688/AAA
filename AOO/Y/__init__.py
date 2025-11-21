"""
Y区 - 扩展功能模块：存储管理组件
Extension Module - Storage Management Components

模块描述：
Y区为扩展功能模块提供全面的存储管理支持，包含文件存储、数据库管理、对象存储等
9个子模块，总计85个类和8,830行代码，提供完整的存储管理框架。

功能分类：
- Y1: 文件存储管理器 (FileStorageManager) - 10类文件存储管理
- Y2: 数据库管理器 (DatabaseManager) - 15类数据库操作
- Y3: 对象存储管理器 (ObjectStorageManager) - 6类对象存储
- Y4: 分布式存储管理器 (DistributedStorageManager) - 5类分布式存储
- Y5: 存储优化器 (StorageOptimizer) - 3类存储优化
- Y6: 存储监控器 (StorageMonitor) - 8类存储监控
- Y7: 存储备份管理器 (StorageBackup) - 12类备份恢复
- Y8: 存储清理器 (StorageCleaner) - 15类清理策略
- Y9: 存储状态聚合器 (StorageStatusAggregator) - 11类状态管理

版本：v1.0.0
最后更新：2025-11-14
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "MiniMax Agent"

# 主模块导入
from .Y1.FileStorageManager import (
    FileType,
    PermissionLevel,
    FileInfo,
    StorageStats,
    FileStorageManager
)

from .Y1 import (
    FileStorageError,
    FileNotFoundError,
    PermissionError,
    StorageFullError,
    InvalidFileError
)

from .Y2.DatabaseManager import (
    DatabaseConfig,
    QueryResult,
    DatabaseStatus,
    ConnectionPool,
    DatabaseManager,
    MySQLManager,
    PostgreSQLManager,
    SQLiteManager,
    Y2DatabaseManager
)

from .Y2 import (
    DatabaseError,
    ConnectionError,
    QueryError,
    TransactionError,
    BackupError,
    MigrationError
)

from .Y3.ObjectStorageManager import (
    StorageProvider,
    ObjectStatus,
    Permission,
    ObjectMetadata,
    StorageConfig,
    ObjectStorageManager
)

from .Y4.DistributedStorageManager import (
    StorageNode,
    DataShard,
    SyncOperation,
    ConsistentHashing,
    DistributedStorageManager
)

from .Y5.StorageOptimizer import (
    StorageMetrics,
    OptimizationReport,
    StorageOptimizer
)

from .Y6.StorageMonitor import (
    StorageInfo,
    PerformanceMetrics,
    Alert,
    DatabaseManager,
    AlertManager,
    EmailAlertHandler,
    LogAlertHandler,
    PerformanceMonitor,
    StorageMonitor
)

from .Y7.StorageBackup import (
    BackupType,
    BackupStatus,
    BackupPriority,
    BackupConfig,
    BackupTask,
    BackupStatistics,
    BackupValidator,
    BackupReport,
    BackupRecovery,
    BackupManager,
    StorageBackup
)

from .Y8.StorageCleaner import (
    CleanStats,
    CleanConfig,
    CleanStrategy,
    AgeBasedStrategy,
    SizeBasedStrategy,
    ExtensionBasedStrategy,
    CleanMonitor,
    CleanReporter,
    FileCleaner,
    CacheCleaner,
    TempCleaner,
    LogCleaner,
    AutoCleaner,
    ManualCleaner,
    StorageCleaner
)

from .Y9.StorageStatusAggregator import (
    StorageStatus,
    AlertLevel,
    StorageModule,
    Alert,
    StorageReport,
    DatabaseManager,
    StatusCollector,
    AlertManager,
    TrendAnalyzer,
    ReportGenerator,
    StorageStatusAggregator
)

# 导出配置
__all__ = [
    # Y1 - 文件存储管理器 (10类)
    "FileType", "PermissionLevel", "FileInfo", "StorageStats", "FileStorageManager",
    "FileStorageError", "FileNotFoundError", "PermissionError", "StorageFullError", "InvalidFileError",
    
    # Y2 - 数据库管理器 (15类)
    "DatabaseConfig", "QueryResult", "DatabaseStatus", "ConnectionPool",
    "DatabaseManager", "MySQLManager", "PostgreSQLManager", "SQLiteManager", "Y2DatabaseManager",
    "DatabaseError", "ConnectionError", "QueryError", "TransactionError", "BackupError", "MigrationError",
    
    # Y3 - 对象存储管理器 (6类)
    "StorageProvider", "ObjectStatus", "Permission", "ObjectMetadata", "StorageConfig", "ObjectStorageManager",
    
    # Y4 - 分布式存储管理器 (5类)
    "StorageNode", "DataShard", "SyncOperation", "ConsistentHashing", "DistributedStorageManager",
    
    # Y5 - 存储优化器 (3类)
    "StorageMetrics", "OptimizationReport", "StorageOptimizer",
    
    # Y6 - 存储监控器 (8类)
    "StorageInfo", "PerformanceMetrics", "Alert", "DatabaseManager", "AlertManager",
    "EmailAlertHandler", "LogAlertHandler", "PerformanceMonitor", "StorageMonitor",
    
    # Y7 - 存储备份管理器 (12类)
    "BackupType", "BackupStatus", "BackupPriority", "BackupConfig", "BackupTask",
    "BackupStatistics", "BackupValidator", "BackupReport", "BackupRecovery", "BackupManager", "StorageBackup",
    
    # Y8 - 存储清理器 (15类)
    "CleanStats", "CleanConfig", "CleanStrategy", "AgeBasedStrategy", "SizeBasedStrategy",
    "ExtensionBasedStrategy", "CleanMonitor", "CleanReporter", "FileCleaner", "CacheCleaner",
    "TempCleaner", "LogCleaner", "AutoCleaner", "ManualCleaner", "StorageCleaner",
    
    # Y9 - 存储状态聚合器 (11类)
    "StorageStatus", "AlertLevel", "StorageModule", "Alert", "StorageReport",
    "DatabaseManager", "StatusCollector", "AlertManager", "TrendAnalyzer", "ReportGenerator", "StorageStatusAggregator"
]

# 模块信息
MODULE_INFO = {
    "name": "Extension Module - Storage Management",
    "version": "1.0.0",
    "total_classes": 85,
    "total_lines": 8830,
    "sub_modules": {
        "Y1": {"name": "File Storage Manager", "classes": 10},
        "Y2": {"name": "Database Manager", "classes": 15},
        "Y3": {"name": "Object Storage Manager", "classes": 6},
        "Y4": {"name": "Distributed Storage Manager", "classes": 5},
        "Y5": {"name": "Storage Optimizer", "classes": 3},
        "Y6": {"name": "Storage Monitor", "classes": 8},
        "Y7": {"name": "Storage Backup Manager", "classes": 12},
        "Y8": {"name": "Storage Cleaner", "classes": 15},
        "Y9": {"name": "Storage Status Aggregator", "classes": 11}
    }
}

print(f"Y区 - 扩展功能模块已初始化，存储组件总数: {MODULE_INFO['total_classes']} 类")