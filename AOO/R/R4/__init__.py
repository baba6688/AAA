#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R4日志备份器 - Python包初始化文件

这是一个功能完整的企业级日志备份管理系统，支持：
- 多类型日志备份（系统、应用、交易、错误、访问日志）
- 智能归档和压缩存储
- 自动清理和空间管理
- 快速索引和搜索
- 数据分析和报告
- 灵活恢复机制
- 实时监控和告警

主要模块：
- LogBackup: 核心日志备份器类
- LogType: 日志类型枚举
- BackupStatus: 备份状态枚举
- LogEntry: 日志条目数据结构
- BackupRecord: 备份记录数据结构

使用示例：
    from LogBackup import LogBackup, LogType
    
    # 初始化备份器
    backup = LogBackup()
    
    # 备份所有日志
    records = backup.backup_logs()
    
    # 搜索日志
    results = backup.search_logs("ERROR")
    
    # 分析日志
    analysis = backup.analyze_logs()

版本：1.0.0
作者：AI Assistant
创建时间：2025-11-06
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "AI Assistant"
__email__ = "support@example.com"
__description__ = "R4企业级日志备份管理系统"

# 导入主要类和枚举
from .LogBackup import (
    LogBackup,
    LogType,
    BackupStatus,
    LogEntry,
    BackupRecord
)

# 公开的API接口
__all__ = [
    "LogBackup",
    "LogType", 
    "BackupStatus",
    "LogEntry",
    "BackupRecord",
    "__version__",
    "__author__",
    "__email__",
    "__description__"
]

# 包级别的便捷函数
def create_backup(config_file="log_backup_config.json"):
    """
    创建日志备份器实例的便捷函数
    
    Args:
        config_file (str): 配置文件路径
        
    Returns:
        LogBackup: 日志备份器实例
    """
    return LogBackup(config_file)


def quick_backup(log_type=None, config_file="log_backup_config.json"):
    """
    快速备份日志的便捷函数
    
    Args:
        log_type (LogType, optional): 要备份的日志类型
        config_file (str): 配置文件路径
        
    Returns:
        List[BackupRecord]: 备份记录列表
    """
    backup = LogBackup(config_file)
    return backup.backup_logs(log_type)


def quick_search(query, log_type=None, config_file="log_backup_config.json"):
    """
    快速搜索日志的便捷函数
    
    Args:
        query (str): 搜索关键词
        log_type (LogType, optional): 日志类型过滤
        config_file (str): 配置文件路径
        
    Returns:
        List[Dict]: 搜索结果列表
    """
    backup = LogBackup(config_file)
    return backup.search_logs(query, log_type)


def quick_status(config_file="log_backup_config.json"):
    """
    快速获取备份状态的便捷函数
    
    Args:
        config_file (str): 配置文件路径
        
    Returns:
        Dict: 备份状态信息
    """
    backup = LogBackup(config_file)
    return backup.get_backup_status()


# 添加便捷函数到__all__
__all__.extend([
    "create_backup",
    "quick_backup", 
    "quick_search",
    "quick_status"
])

# 包信息
package_info = {
    "name": "r4-log-backup",
    "version": __version__,
    "description": __description__,
    "author": __author__,
    "author_email": __email__,
    "license": "MIT",
    "keywords": ["log", "backup", "archive", "management", "system"],
    "classifiers": [
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8", 
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: System :: Systems Administration",
        "Topic :: System :: Logging",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    "requirements": [
        "python >= 3.7",
    ],
    "features": [
        "多类型日志备份",
        "智能归档压缩",
        "自动清理管理",
        "快速索引搜索",
        "数据分析报告",
        "灵活恢复机制",
        "实时监控告警",
    ]
}

def get_package_info():
    """
    获取包信息
    
    Returns:
        Dict: 包详细信息
    """
    return package_info.copy()


def print_banner():
    """打印欢迎横幅"""
    banner = f"""
╔══════════════════════════════════════════════════════════════╗
║                    R4日志备份器 v{__version__}                          ║
║                                                              ║
║              企业级日志备份管理系统                           ║
║                                                              ║
║  功能：备份 | 归档 | 清理 | 压缩 | 索引 | 搜索 | 分析 | 恢复 | 监控  ║
║                                                              ║
║  作者：{__author__:<20}                                    ║
║  邮箱：{__email__:<20}                           ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_requirements():
    """
    检查系统要求
    
    Returns:
        bool: 是否满足要求
    """
    import sys
    import sqlite3
    
    # 检查Python版本
    if sys.version_info < (3, 7):
        print("错误：需要Python 3.7或更高版本")
        return False
    
    # 检查SQLite支持
    try:
        sqlite3.connect(':memory:')
    except Exception as e:
        print(f"错误：SQLite支持检查失败: {e}")
        return False
    
    # 检查必要的模块
    required_modules = [
        'os', 'sys', 'json', 'time', 'shutil', 'gzip', 
        'sqlite3', 'threading', 'logging', 'hashlib',
        'zipfile', 'tarfile', 'glob', 're', 'datetime',
        'pathlib', 'typing', 'dataclasses', 'enum',
        'subprocess', 'tempfile'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"错误：缺少必要的模块: {', '.join(missing_modules)}")
        return False
    
    return True


def initialize_package():
    """初始化包"""
    # 打印横幅
    print_banner()
    
    # 检查系统要求
    if not check_requirements():
        print("系统要求检查失败，请确保满足所有依赖")
        return False
    
    print("✅ 系统要求检查通过")
    print("✅ R4日志备份器已就绪")
    print()
    print("快速开始：")
    print("  from LogBackup import LogBackup, LogType")
    print("  backup = LogBackup()")
    print("  records = backup.backup_logs()")
    print()
    
    return True


# 包初始化时自动检查和显示信息
if __name__ == "__main__":
    # 直接运行包时的初始化
    initialize_package()
else:
    # 作为包导入时的初始化
    # 可以在这里添加包级别的初始化逻辑
    pass