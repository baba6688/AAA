#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R1数据备份器 - 包初始化文件

提供完整的数据备份解决方案，包括：
- 文件、数据库、对象存储备份
- 多种压缩算法支持
- 数据加密保护
- 增量备份
- 备份验证
- 存储管理
- 备份调度
- 备份报告

作者: R1数据备份器团队
版本: 1.0.0
"""

# 核心类
from .DataBackup import (
    # 主要类
    DataBackup,
    
    # 配置类
    BackupConfig,
    BackupStatus,
    
    # 压缩器
    Compressor,
    
    # 加密器
    Encryptor,
    
    # 验证器
    BackupVerifier,
    
    # 调度器
    BackupScheduler,
    
    # 报告器
    BackupReporter,
    
    # 备份源
    BackupSource,
    FileBackupSource,
    DatabaseBackupSource,
    CloudBackupSource,
    
    # 存储
    BackupStorage,
    LocalStorage,
    CloudStorage,
    
    # 便利函数
    create_file_backup,
    create_database_backup,
    create_cloud_backup,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "R1数据备份器团队"
__email__ = "support@r1backup.com"
__description__ = "R1数据备份器 - 完整的数据备份解决方案"

# 包级别的默认配置
DEFAULT_CONFIG = {
    'data_dir': './backup_data',
    'log_level': 'INFO',
    'compression': 'gzip',
    'encryption': False,
    'verify_backup': True,
    'retention_days': 30,
    'max_concurrent_backups': 3,
    'report_config': {
        'email': {
            'enabled': False,
            'smtp_server': '',
            'smtp_port': 587,
            'use_tls': True,
            'username': '',
            'password': '',
            'from_email': '',
            'to_email': ''
        }
    }
}

# 支持的压缩算法
SUPPORTED_COMPRESSIONS = [
    'none',      # 无压缩
    'gzip',      # GZIP压缩
    'bz2',       # BZ2压缩
    'lzma',      # LZMA压缩
    'zip',       # ZIP压缩
    'tar',       # TAR压缩
    'tar.gz',    # TAR.GZ压缩
    'tar.bz2',   # TAR.BZ2压缩
]

# 支持的数据库类型
SUPPORTED_DATABASES = [
    'sqlite',      # SQLite
    'mysql',       # MySQL
    'postgresql',  # PostgreSQL
]

# 支持的云存储提供商
SUPPORTED_CLOUD_PROVIDERS = [
    's3',          # Amazon S3
    'azure',       # Azure Blob Storage
    'gcs',         # Google Cloud Storage
]

# 备份类型
BACKUP_TYPES = [
    'full',        # 完整备份
    'incremental', # 增量备份
    'differential', # 差异备份
]

# 调度类型
SCHEDULE_TYPES = [
    'interval',    # 间隔执行
    'daily',       # 每日执行
    'weekly',      # 每周执行
    'monthly',     # 每月执行
]

# 备份状态
BACKUP_STATUS = [
    'pending',     # 待执行
    'running',     # 执行中
    'success',     # 成功
    'failed',      # 失败
    'cancelled',   # 已取消
]

# 公开的API函数
__all__ = [
    # 核心类
    'DataBackup',
    
    # 配置类
    'BackupConfig',
    'BackupStatus',
    
    # 工具类
    'Compressor',
    'Encryptor',
    'BackupVerifier',
    'BackupScheduler',
    'BackupReporter',
    
    # 备份源
    'BackupSource',
    'FileBackupSource',
    'DatabaseBackupSource',
    'CloudBackupSource',
    
    # 存储
    'BackupStorage',
    'LocalStorage',
    'CloudStorage',
    
    # 便利函数
    'create_file_backup',
    'create_database_backup',
    'create_cloud_backup',
    
    # 常量
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    'DEFAULT_CONFIG',
    'SUPPORTED_COMPRESSIONS',
    'SUPPORTED_DATABASES',
    'SUPPORTED_CLOUD_PROVIDERS',
    'BACKUP_TYPES',
    'SCHEDULE_TYPES',
    'BACKUP_STATUS',
]

# 包初始化时的检查
def _check_dependencies():
    """检查依赖项"""
    missing_deps = []
    
    try:
        import cryptography
    except ImportError:
        missing_deps.append('cryptography')
    
    try:
        import boto3
    except ImportError:
        missing_deps.append('boto3')
    
    try:
        import mysql.connector
    except ImportError:
        missing_deps.append('mysql-connector-python')
    
    try:
        import psycopg2
    except ImportError:
        missing_deps.append('psycopg2-binary')
    
    try:
        import schedule
    except ImportError:
        missing_deps.append('schedule')
    
    if missing_deps:
        import warnings
        warnings.warn(
            f"缺少以下可选依赖项: {', '.join(missing_deps)}"
            f"某些功能可能无法使用。请使用 'pip install {' '.join(missing_deps)}' 安装。",
            UserWarning
        )

# 执行依赖检查
_check_dependencies()

# 快速入门指南
def quick_start():
    """
    快速入门指南
    
    返回一个包含基本使用示例的字符串。
    """
    return """
    R1数据备份器快速入门
    ====================
    
    1. 基本文件备份:
       ```python
       from R1 import create_file_backup
       
       result = create_file_backup(
           source_path="/path/to/source",
           backup_path="/path/to/backup",
           backup_id="my_backup",
           compression="gzip"
       )
       print(f"备份状态: {result.status}")
       ```
    
    2. 带加密的备份:
       ```python
       from R1 import BackupConfig, DataBackup
       
       config = BackupConfig(
           backup_id="encrypted_backup",
           source_path="/path/to/source",
           backup_path="/path/to/backup",
           encryption=True,
           encryption_key="my_secret_key"
       )
       
       backup_system = DataBackup()
       result = backup_system.create_backup(config)
       ```
    
    3. 数据库备份:
       ```python
       from R1 import create_database_backup
       
       db_config = {
           'type': 'sqlite',
           'database_path': '/path/to/database.db'
       }
       
       result = create_database_backup(
           db_config=db_config,
           backup_path="/path/to/backup",
           backup_id="db_backup"
       )
       ```
    
    4. 定时备份:
       ```python
       from R1 import BackupConfig, DataBackup
       
       backup_system = DataBackup()
       
       config = BackupConfig(
           backup_id="scheduled_backup",
           source_path="/path/to/source",
           backup_path="/path/to/backup"
       )
       
       schedule_config = {
           'type': 'daily',
           'time': '02:00'
       }
       
       job_id = backup_system.schedule_backup(config, schedule_config)
       backup_system.start_scheduler()
       ```
    
    5. 恢复备份:
       ```python
       backup_system = DataBackup()
       result = backup_system.restore_backup(
           backup_id="my_backup",
           restore_path="/path/to/restore",
           encryption_key="my_secret_key"  # 如果备份是加密的
       )
       ```
    
    6. 生成报告:
       ```python
       backup_system = DataBackup()
       
       # 生成报告
       report = backup_system.generate_report()
       print(report)
       
       # 保存报告
       backup_system.save_report(report, "/path/to/report.txt")
       ```
    
    更多信息请查看文档或运行测试文件。
    """

# 版本检查
def check_version():
    """检查版本信息"""
    import sys
    print(f"R1数据备份器版本: {__version__}")
    print(f"Python版本: {sys.version}")
    print(f"作者: {__author__}")
    print(f"描述: {__description__}")

if __name__ == "__main__":
    check_version()
    print(quick_start())