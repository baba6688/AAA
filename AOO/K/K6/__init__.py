"""
K6数据配置处理器模块

这是一个全面的数据配置管理模块，用于处理K6系统中的各种数据配置需求。

主要组件:
- DataConfigurationProcessor: 主配置处理器
- 数据源管理: DataSourceManager
- 存储管理: StorageManager  
- 数据更新管理: DataUpdateManager
- 数据质量管理: DataQualityManager
- 安全管理: SecurityManager
- 备份管理: BackupManager
- 异步处理: AsyncDataProcessor

示例使用:
    from K.K6 import DataConfigurationProcessor
    
    processor = DataConfigurationProcessor()
    await processor.initialize()
"""

# 导入主要类和异常
from .DataConfigurationProcessor import (
    # 主要处理器类
    DataConfigurationProcessor,
    
    # 异常类
    ConfigError,
    DataSourceError, 
    StorageError,
    SecurityError,
    QualityError,
    BackupError,
    AsyncProcessingError,
    
    # 数据格式和压缩枚举
    DataFormat,
    CompressionType,
    StorageType,
    UpdateStrategy,
    SecurityLevel,
    LogLevel,
    
    # 数据结构类
    DataSourceParameters,
    AuthenticationInfo,
    DataFormatConfig,
    StoragePathConfig,
    StorageStrategyConfig,
    UpdateFrequency,
    QualityValidationRule,
    EncryptionConfig,
    AccessControlConfig,
    AuditLogConfig,
    BackupConfig,
    RecoveryConfig,
    
    # 管理器类
    DataSourceManager,
    DataFormatProcessor,
    StorageManager,
    DataUpdateManager,
    DataQualityManager,
    SecurityManager,
    BackupManager,
    AsyncDataProcessor,
    
    # 类型别名
    T,
    ConfigDict,
    DataSourceConfig,
    StorageConfig,
    SecurityConfig
)

# 模块级文档和元信息
__version__ = "1.0.0"
__author__ = "K6开发团队"
__all__ = [
    # 主要类
    "DataConfigurationProcessor",
    
    # 异常类
    "ConfigError",
    "DataSourceError", 
    "StorageError",
    "SecurityError", 
    "QualityError",
    "BackupError",
    "AsyncProcessingError",
    
    # 枚举
    "DataFormat",
    "CompressionType",
    "StorageType", 
    "UpdateStrategy",
    "SecurityLevel",
    "LogLevel",
    
    # 数据类
    "DataSourceParameters",
    "AuthenticationInfo",
    "DataFormatConfig",
    "StoragePathConfig",
    "StorageStrategyConfig",
    "UpdateFrequency",
    "QualityValidationRule",
    "EncryptionConfig",
    "AccessControlConfig",
    "AuditLogConfig", 
    "BackupConfig",
    "RecoveryConfig",
    
    # 管理器
    "DataSourceManager",
    "DataFormatProcessor",
    "StorageManager",
    "DataUpdateManager",
    "DataQualityManager",
    "SecurityManager",
    "BackupManager",
    "AsyncDataProcessor",
    
    # 类型别名
    "T",
    "ConfigDict",
    "DataSourceConfig",
    "StorageConfig",
    "SecurityConfig"
]