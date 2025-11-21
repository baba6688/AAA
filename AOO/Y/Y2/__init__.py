#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y2数据库管理器包初始化文件
提供数据库管理器的公共接口
"""

from .DatabaseManager import (
    # 主要类和配置
    Y2DatabaseManager,
    DatabaseConfig,
    QueryResult,
    DatabaseStatus,
    
    # 数据库管理器类
    MySQLManager,
    PostgreSQLManager,
    SQLiteManager,
    
    # 连接池类
    ConnectionPool,
    
    # 异常类
    DatabaseError,
    ConnectionError,
    QueryError,
    TransactionError,
    BackupError,
    MigrationError,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "Y2 Database Manager Team"
__email__ = "support@y2db.com"
__description__ = "Y2数据库管理器 - 功能完整的数据库管理系统"

# 包级别的公共接口
__all__ = [
    # 主要类
    "Y2DatabaseManager",
    "DatabaseConfig", 
    "QueryResult",
    "DatabaseStatus",
    
    # 数据库管理器
    "MySQLManager",
    "PostgreSQLManager", 
    "SQLiteManager",
    
    # 工具类
    "ConnectionPool",
    
    # 异常类
    "DatabaseError",
    "ConnectionError",
    "QueryError", 
    "TransactionError",
    "BackupError",
    "MigrationError",
]

# 便捷函数
def create_manager():
    """
    创建Y2数据库管理器实例的便捷函数
    
    Returns:
        Y2DatabaseManager: 数据库管理器实例
    """
    return Y2DatabaseManager()

def create_mysql_config(host="localhost", port=3306, database="", 
                       username="", password="", **kwargs):
    """
    创建MySQL数据库配置的便捷函数
    
    Args:
        host: 数据库主机地址
        port: 数据库端口
        database: 数据库名称
        username: 用户名
        password: 密码
        **kwargs: 其他配置参数
    
    Returns:
        DatabaseConfig: MySQL数据库配置
    """
    return DatabaseConfig(
        db_type="mysql",
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        **kwargs
    )

def create_postgresql_config(host="localhost", port=5432, database="",
                           username="", password="", **kwargs):
    """
    创建PostgreSQL数据库配置的便捷函数
    
    Args:
        host: 数据库主机地址
        port: 数据库端口
        database: 数据库名称
        username: 用户名
        password: 密码
        **kwargs: 其他配置参数
    
    Returns:
        DatabaseConfig: PostgreSQL数据库配置
    """
    return DatabaseConfig(
        db_type="postgresql",
        host=host,
        port=port,
        database=database,
        username=username,
        password=password,
        **kwargs
    )

def create_sqlite_config(database="", **kwargs):
    """
    创建SQLite数据库配置的便捷函数
    
    Args:
        database: 数据库文件路径
        **kwargs: 其他配置参数
    
    Returns:
        DatabaseConfig: SQLite数据库配置
    """
    return DatabaseConfig(
        db_type="sqlite",
        database=database,
        **kwargs
    )

# 异常类定义
class DatabaseError(Exception):
    """数据库操作基础异常"""
    pass

class ConnectionError(DatabaseError):
    """数据库连接异常"""
    pass

class QueryError(DatabaseError):
    """数据库查询异常"""
    pass

class TransactionError(DatabaseError):
    """数据库事务异常"""
    pass

class BackupError(DatabaseError):
    """数据库备份异常"""
    pass

class MigrationError(DatabaseError):
    """数据库迁移异常"""
    pass

# 包初始化日志
import logging
logger = logging.getLogger(__name__)
logger.info(f"Y2数据库管理器 v{__version__} 已加载")

# 使用示例
EXAMPLE_USAGE = """
使用示例:

from Y2 import create_manager, create_mysql_config

# 创建管理器
manager = create_manager()

# 添加MySQL数据库
config = create_mysql_config(
    host="localhost",
    database="myapp",
    username="root", 
    password="password"
)
manager.add_database("my_mysql", config)

# 连接并查询
manager.connect_database("my_mysql")
result = manager.execute_query("my_mysql", "SELECT VERSION()")
print(result.data)
"""

if __name__ == "__main__":
    print(__description__)
    print(f"版本: {__version__}")
    print(f"作者: {__author__}")
    print("\n" + "="*50)
    print("使用示例:")
    print(EXAMPLE_USAGE)