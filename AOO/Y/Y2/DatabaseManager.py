#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Y2数据库管理器
一个功能完整的数据库管理系统，支持多数据库类型连接、查询、事务管理、备份、优化等功能
"""

import sqlite3

# 条件导入MySQL和PostgreSQL驱动
try:
    import mysql.connector
    MYSQL_AVAILABLE = True
except ImportError:
    MYSQL_AVAILABLE = False

try:
    import psycopg2
    import psycopg2.extras
    POSTGRESQL_AVAILABLE = True
except ImportError:
    POSTGRESQL_AVAILABLE = False
import json
import os
import time
import hashlib
import logging
from datetime import datetime, timedelta
from contextlib import contextmanager
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import threading
from abc import ABC, abstractmethod


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('y2_database_manager.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    """数据库配置类"""
    db_type: str  # 数据库类型: mysql, postgresql, sqlite
    host: str = "localhost"
    port: int = 3306
    database: str = ""
    username: str = ""
    password: str = ""
    charset: str = "utf8mb4"
    connection_timeout: int = 30
    max_connections: int = 20
    pool_name: str = "default"


@dataclass
class QueryResult:
    """查询结果类"""
    data: List[Dict[str, Any]]
    row_count: int
    execution_time: float
    query: str
    timestamp: datetime


@dataclass
class DatabaseStatus:
    """数据库状态类"""
    is_connected: bool
    connection_count: int
    active_queries: int
    uptime: float
    last_backup: Optional[datetime]
    database_size: Optional[float]


class ConnectionPool:
    """数据库连接池"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connections = []
        self.in_use = set()
        self.max_connections = config.max_connections
        self.lock = threading.Lock()
        
    def get_connection(self):
        """获取连接"""
        with self.lock:
            # 尝试复用现有连接
            for conn in self.connections:
                if id(conn) not in self.in_use:
                    self.in_use.add(id(conn))
                    return conn
            
            # 创建新连接
            if len(self.connections) < self.max_connections:
                conn = self._create_connection()
                self.connections.append(conn)
                self.in_use.add(id(conn))
                return conn
            
            raise Exception("连接池已满")
    
    def return_connection(self, conn):
        """归还连接"""
        with self.lock:
            if id(conn) in self.in_use:
                self.in_use.remove(id(conn))
    
    def _create_connection(self):
        """创建连接"""
        if self.config.db_type == "mysql":
            return mysql.connector.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password,
                charset=self.config.charset,
                connection_timeout=self.config.connection_timeout
            )
        elif self.config.db_type == "postgresql":
            return psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.username,
                password=self.config.password
            )
        elif self.config.db_type == "sqlite":
            return sqlite3.connect(self.config.database)
        else:
            raise ValueError(f"不支持的数据库类型: {self.config.db_type}")
    
    def close_all(self):
        """关闭所有连接"""
        with self.lock:
            for conn in self.connections:
                try:
                    conn.close()
                except:
                    pass
            self.connections.clear()
            self.in_use.clear()


class DatabaseManager(ABC):
    """数据库管理器基类"""
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection_pool = ConnectionPool(config)
        self.is_connected = False
        self.start_time = time.time()
        self.query_count = 0
        self.successful_queries = 0
        self.failed_queries = 0
        
    @abstractmethod
    def connect(self) -> bool:
        """连接数据库"""
        pass
    
    @abstractmethod
    def disconnect(self):
        """断开数据库连接"""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> QueryResult:
        """执行查询"""
        pass
    
    @abstractmethod
    def execute_update(self, query: str, params: Optional[Tuple] = None) -> int:
        """执行更新操作"""
        pass
    
    @abstractmethod
    def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        pass


class MySQLManager(DatabaseManager):
    """MySQL数据库管理器"""
    
    def __init__(self, config: DatabaseConfig):
        if not MYSQL_AVAILABLE:
            raise ImportError("MySQL驱动未安装，请运行: pip install mysql-connector-python")
        super().__init__(config)
    
    def connect(self) -> bool:
        """连接MySQL数据库"""
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            self.connection_pool.return_connection(conn)
            self.is_connected = True
            logger.info("MySQL数据库连接成功")
            return True
        except Exception as e:
            logger.error(f"MySQL数据库连接失败: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """断开MySQL数据库连接"""
        try:
            self.connection_pool.close_all()
            self.is_connected = False
            logger.info("MySQL数据库连接已断开")
        except Exception as e:
            logger.error(f"断开MySQL数据库连接失败: {e}")
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> QueryResult:
        """执行MySQL查询"""
        start_time = time.time()
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor(dictionary=True)
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            result = cursor.fetchall()
            row_count = cursor.rowcount
            execution_time = time.time() - start_time
            
            cursor.close()
            self.connection_pool.return_connection(conn)
            
            self.query_count += 1
            self.successful_queries += 1
            
            logger.info(f"MySQL查询执行成功: {query[:50]}...")
            
            return QueryResult(
                data=result,
                row_count=row_count,
                execution_time=execution_time,
                query=query,
                timestamp=datetime.now()
            )
        except Exception as e:
            self.query_count += 1
            self.failed_queries += 1
            logger.error(f"MySQL查询执行失败: {e}")
            raise
    
    def execute_update(self, query: str, params: Optional[Tuple] = None) -> int:
        """执行MySQL更新操作"""
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            affected_rows = cursor.rowcount
            conn.commit()
            
            cursor.close()
            self.connection_pool.return_connection(conn)
            
            logger.info(f"MySQL更新操作执行成功，影响行数: {affected_rows}")
            
            return affected_rows
        except Exception as e:
            logger.error(f"MySQL更新操作执行失败: {e}")
            raise
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取MySQL数据库信息"""
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor()
            
            # 获取数据库版本
            cursor.execute("SELECT VERSION()")
            version = cursor.fetchone()[0]
            
            # 获取连接数
            cursor.execute("SHOW STATUS LIKE 'Threads_connected'")
            connections = cursor.fetchone()[1]
            
            # 获取查询数
            cursor.execute("SHOW STATUS LIKE 'Queries'")
            queries = cursor.fetchone()[1]
            
            cursor.close()
            self.connection_pool.return_connection(conn)
            
            return {
                "type": "MySQL",
                "version": version,
                "connections": int(connections),
                "queries": int(queries),
                "uptime": time.time() - self.start_time
            }
        except Exception as e:
            logger.error(f"获取MySQL数据库信息失败: {e}")
            return {}


class PostgreSQLManager(DatabaseManager):
    """PostgreSQL数据库管理器"""
    
    def __init__(self, config: DatabaseConfig):
        if not POSTGRESQL_AVAILABLE:
            raise ImportError("PostgreSQL驱动未安装，请运行: pip install psycopg2-binary")
        super().__init__(config)
    
    def connect(self) -> bool:
        """连接PostgreSQL数据库"""
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            self.connection_pool.return_connection(conn)
            self.is_connected = True
            logger.info("PostgreSQL数据库连接成功")
            return True
        except Exception as e:
            logger.error(f"PostgreSQL数据库连接失败: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """断开PostgreSQL数据库连接"""
        try:
            self.connection_pool.close_all()
            self.is_connected = False
            logger.info("PostgreSQL数据库连接已断开")
        except Exception as e:
            logger.error(f"断开PostgreSQL数据库连接失败: {e}")
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> QueryResult:
        """执行PostgreSQL查询"""
        start_time = time.time()
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            result = cursor.fetchall()
            row_count = cursor.rowcount
            execution_time = time.time() - start_time
            
            cursor.close()
            self.connection_pool.return_connection(conn)
            
            self.query_count += 1
            self.successful_queries += 1
            
            logger.info(f"PostgreSQL查询执行成功: {query[:50]}...")
            
            return QueryResult(
                data=[dict(row) for row in result],
                row_count=row_count,
                execution_time=execution_time,
                query=query,
                timestamp=datetime.now()
            )
        except Exception as e:
            self.query_count += 1
            self.failed_queries += 1
            logger.error(f"PostgreSQL查询执行失败: {e}")
            raise
    
    def execute_update(self, query: str, params: Optional[Tuple] = None) -> int:
        """执行PostgreSQL更新操作"""
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            affected_rows = cursor.rowcount
            conn.commit()
            
            cursor.close()
            self.connection_pool.return_connection(conn)
            
            logger.info(f"PostgreSQL更新操作执行成功，影响行数: {affected_rows}")
            
            return affected_rows
        except Exception as e:
            logger.error(f"PostgreSQL更新操作执行失败: {e}")
            raise
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取PostgreSQL数据库信息"""
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor()
            
            # 获取数据库版本
            cursor.execute("SELECT version()")
            version = cursor.fetchone()[0]
            
            # 获取连接数
            cursor.execute("SELECT count(*) FROM pg_stat_activity")
            connections = cursor.fetchone()[0]
            
            cursor.close()
            self.connection_pool.return_connection(conn)
            
            return {
                "type": "PostgreSQL",
                "version": version,
                "connections": connections,
                "uptime": time.time() - self.start_time
            }
        except Exception as e:
            logger.error(f"获取PostgreSQL数据库信息失败: {e}")
            return {}


class SQLiteManager(DatabaseManager):
    """SQLite数据库管理器"""
    
    def connect(self) -> bool:
        """连接SQLite数据库"""
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            self.connection_pool.return_connection(conn)
            self.is_connected = True
            logger.info("SQLite数据库连接成功")
            return True
        except Exception as e:
            logger.error(f"SQLite数据库连接失败: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """断开SQLite数据库连接"""
        try:
            self.connection_pool.close_all()
            self.is_connected = False
            logger.info("SQLite数据库连接已断开")
        except Exception as e:
            logger.error(f"断开SQLite数据库连接失败: {e}")
    
    def execute_query(self, query: str, params: Optional[Tuple] = None) -> QueryResult:
        """执行SQLite查询"""
        start_time = time.time()
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            result = cursor.fetchall()
            # SQLite的cursor.rowcount对于SELECT查询返回-1，需要使用len(result)
            row_count = len(result) if result else 0
            execution_time = time.time() - start_time
            
            # 获取列名
            column_names = [description[0] for description in cursor.description]
            result_dicts = [dict(zip(column_names, row)) for row in result]
            
            cursor.close()
            self.connection_pool.return_connection(conn)
            
            self.query_count += 1
            self.successful_queries += 1
            
            logger.info(f"SQLite查询执行成功: {query[:50]}...")
            
            return QueryResult(
                data=result_dicts,
                row_count=row_count,
                execution_time=execution_time,
                query=query,
                timestamp=datetime.now()
            )
        except Exception as e:
            self.query_count += 1
            self.failed_queries += 1
            logger.error(f"SQLite查询执行失败: {e}")
            raise
    
    def execute_update(self, query: str, params: Optional[Tuple] = None) -> int:
        """执行SQLite更新操作"""
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            affected_rows = cursor.rowcount
            conn.commit()
            
            cursor.close()
            self.connection_pool.return_connection(conn)
            
            logger.info(f"SQLite更新操作执行成功，影响行数: {affected_rows}")
            
            return affected_rows
        except Exception as e:
            logger.error(f"SQLite更新操作执行失败: {e}")
            raise
    
    def get_database_info(self) -> Dict[str, Any]:
        """获取SQLite数据库信息"""
        try:
            conn = self.connection_pool.get_connection()
            cursor = conn.cursor()
            
            # 获取SQLite版本
            cursor.execute("SELECT sqlite_version()")
            version = cursor.fetchone()[0]
            
            # 获取数据库大小
            cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
            size = cursor.fetchone()[0]
            
            cursor.close()
            self.connection_pool.return_connection(conn)
            
            return {
                "type": "SQLite",
                "version": version,
                "size_bytes": size,
                "uptime": time.time() - self.start_time
            }
        except Exception as e:
            logger.error(f"获取SQLite数据库信息失败: {e}")
            return {}


class Y2DatabaseManager:
    """Y2数据库管理器主类"""
    
    def __init__(self):
        self.managers: Dict[str, DatabaseManager] = {}
        self.configs: Dict[str, DatabaseConfig] = {}
        self.transaction_history: List[Dict[str, Any]] = []
        self.backup_history: List[Dict[str, Any]] = []
        self.migration_history: List[Dict[str, Any]] = []
        self.monitoring_data: Dict[str, Any] = {}
        self.security_settings = {
            "enable_encryption": False,
            "allowed_ips": [],
            "max_query_time": 30,
            "query_logging": True
        }
        
    def add_database(self, name: str, config: DatabaseConfig) -> bool:
        """添加数据库配置"""
        try:
            if config.db_type == "mysql":
                if not MYSQL_AVAILABLE:
                    raise ImportError("MySQL驱动未安装，请运行: pip install mysql-connector-python")
                manager = MySQLManager(config)
            elif config.db_type == "postgresql":
                if not POSTGRESQL_AVAILABLE:
                    raise ImportError("PostgreSQL驱动未安装，请运行: pip install psycopg2-binary")
                manager = PostgreSQLManager(config)
            elif config.db_type == "sqlite":
                manager = SQLiteManager(config)
            else:
                raise ValueError(f"不支持的数据库类型: {config.db_type}")
            
            self.managers[name] = manager
            self.configs[name] = config
            
            logger.info(f"数据库配置已添加: {name}")
            return True
        except Exception as e:
            logger.error(f"添加数据库配置失败: {e}")
            return False
    
    def connect_database(self, name: str) -> bool:
        """连接数据库"""
        if name not in self.managers:
            logger.error(f"数据库配置不存在: {name}")
            return False
        
        return self.managers[name].connect()
    
    def disconnect_database(self, name: str):
        """断开数据库连接"""
        if name in self.managers:
            self.managers[name].disconnect()
    
    def execute_query(self, name: str, query: str, params: Optional[Tuple] = None) -> QueryResult:
        """执行查询"""
        if name not in self.managers:
            raise ValueError(f"数据库配置不存在: {name}")
        
        # 安全检查
        if self.security_settings["query_logging"]:
            self._log_query(name, query, params)
        
        # 检查查询时间限制
        start_time = time.time()
        result = self.managers[name].execute_query(query, params)
        execution_time = time.time() - start_time
        
        if execution_time > self.security_settings["max_query_time"]:
            logger.warning(f"查询执行时间过长: {execution_time:.2f}秒")
        
        return result
    
    def execute_update(self, name: str, query: str, params: Optional[Tuple] = None) -> int:
        """执行更新操作"""
        if name not in self.managers:
            raise ValueError(f"数据库配置不存在: {name}")
        
        return self.managers[name].execute_update(query, params)
    
    @contextmanager
    def transaction(self, name: str):
        """事务管理上下文"""
        if name not in self.managers:
            raise ValueError(f"数据库配置不存在: {name}")
        
        manager = self.managers[name]
        conn = manager.connection_pool.get_connection()
        
        try:
            yield conn
            conn.commit()
            self._record_transaction(name, "commit", True)
            logger.info(f"事务提交成功: {name}")
        except Exception as e:
            conn.rollback()
            self._record_transaction(name, "rollback", False, str(e))
            logger.error(f"事务回滚: {e}")
            raise
        finally:
            manager.connection_pool.return_connection(conn)
    
    def backup_database(self, name: str, backup_path: str) -> bool:
        """备份数据库"""
        try:
            if name not in self.managers:
                raise ValueError(f"数据库配置不存在: {name}")
            
            config = self.configs[name]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if config.db_type == "mysql":
                return self._backup_mysql(config, backup_path, timestamp)
            elif config.db_type == "postgresql":
                return self._backup_postgresql(config, backup_path, timestamp)
            elif config.db_type == "sqlite":
                return self._backup_sqlite(config, backup_path, timestamp)
            
            return False
        except Exception as e:
            logger.error(f"数据库备份失败: {e}")
            return False
    
    def restore_database(self, name: str, backup_path: str) -> bool:
        """恢复数据库"""
        try:
            if name not in self.managers:
                raise ValueError(f"数据库配置不存在: {name}")
            
            config = self.configs[name]
            
            if config.db_type == "mysql":
                return self._restore_mysql(config, backup_path)
            elif config.db_type == "postgresql":
                return self._restore_postgresql(config, backup_path)
            elif config.db_type == "sqlite":
                return self._restore_sqlite(config, backup_path)
            
            return False
        except Exception as e:
            logger.error(f"数据库恢复失败: {e}")
            return False
    
    def optimize_database(self, name: str) -> Dict[str, Any]:
        """优化数据库"""
        try:
            if name not in self.managers:
                raise ValueError(f"数据库配置不存在: {name}")
            
            config = self.configs[name]
            results = {}
            
            if config.db_type == "mysql":
                results = self._optimize_mysql(name)
            elif config.db_type == "postgresql":
                results = self._optimize_postgresql(name)
            elif config.db_type == "sqlite":
                results = self._optimize_sqlite(name)
            
            logger.info(f"数据库优化完成: {name}")
            return results
        except Exception as e:
            logger.error(f"数据库优化失败: {e}")
            return {}
    
    def monitor_database(self, name: str) -> DatabaseStatus:
        """监控数据库状态"""
        try:
            if name not in self.managers:
                raise ValueError(f"数据库配置不存在: {name}")
            
            manager = self.managers[name]
            info = manager.get_database_info()
            
            # 计算连接数
            connection_count = 0
            if hasattr(manager, 'connection_pool'):
                connection_count = len(manager.connection_pool.connections)
            
            # 获取最后备份时间
            last_backup = None
            for backup in reversed(self.backup_history):
                if backup['database'] == name:
                    last_backup = backup['timestamp']
                    break
            
            # 获取数据库大小
            database_size = info.get('size_bytes', 0) if info else 0
            
            status = DatabaseStatus(
                is_connected=manager.is_connected,
                connection_count=connection_count,
                active_queries=manager.query_count,
                uptime=time.time() - manager.start_time,
                last_backup=last_backup,
                database_size=database_size
            )
            
            return status
        except Exception as e:
            logger.error(f"数据库监控失败: {e}")
            return DatabaseStatus(False, 0, 0, 0, None, 0)
    
    def migrate_database(self, name: str, migration_file: str) -> bool:
        """数据库迁移"""
        try:
            if name not in self.managers:
                raise ValueError(f"数据库配置不存在: {name}")
            
            # 读取迁移文件
            with open(migration_file, 'r', encoding='utf-8') as f:
                migrations = json.load(f)
            
            # 执行迁移
            for migration in migrations:
                sql = migration['sql']
                description = migration.get('description', '')
                
                self.execute_update(name, sql)
                
                # 记录迁移历史
                self.migration_history.append({
                    'database': name,
                    'description': description,
                    'sql': sql,
                    'timestamp': datetime.now(),
                    'migration_file': migration_file
                })
            
            logger.info(f"数据库迁移完成: {name}")
            return True
        except Exception as e:
            logger.error(f"数据库迁移失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = {
            'total_databases': len(self.managers),
            'connected_databases': sum(1 for m in self.managers.values() if m.is_connected),
            'total_queries': sum(m.query_count for m in self.managers.values()),
            'successful_queries': sum(m.successful_queries for m in self.managers.values()),
            'failed_queries': sum(m.failed_queries for m in self.managers.values()),
            'transaction_count': len(self.transaction_history),
            'backup_count': len(self.backup_history),
            'migration_count': len(self.migration_history),
            'uptime': time.time() - min(m.start_time for m in self.managers.values()) if self.managers else 0
        }
        
        return stats
    
    def _log_query(self, database: str, query: str, params: Optional[Tuple]):
        """记录查询日志"""
        query_hash = hashlib.md5(f"{query}{params}".encode()).hexdigest()
        self.monitoring_data[query_hash] = {
            'database': database,
            'query': query,
            'params': params,
            'timestamp': datetime.now()
        }
    
    def _record_transaction(self, database: str, action: str, success: bool, error: str = ""):
        """记录事务历史"""
        self.transaction_history.append({
            'database': database,
            'action': action,
            'success': success,
            'error': error,
            'timestamp': datetime.now()
        })
    
    def _backup_mysql(self, config: DatabaseConfig, backup_path: str, timestamp: str) -> bool:
        """MySQL备份"""
        try:
            backup_file = f"{backup_path}/mysql_{config.database}_{timestamp}.sql"
            os.makedirs(os.path.dirname(backup_file), exist_ok=True)
            
            # 使用mysqldump命令备份
            cmd = f"mysqldump -h{config.host} -P{config.port} -u{config.username} -p{config.password} {config.database} > {backup_file}"
            os.system(cmd)
            
            self.backup_history.append({
                'database': config.database,
                'backup_file': backup_file,
                'timestamp': datetime.now(),
                'type': 'mysql'
            })
            
            return os.path.exists(backup_file)
        except Exception as e:
            logger.error(f"MySQL备份失败: {e}")
            return False
    
    def _backup_postgresql(self, config: DatabaseConfig, backup_path: str, timestamp: str) -> bool:
        """PostgreSQL备份"""
        try:
            backup_file = f"{backup_path}/postgresql_{config.database}_{timestamp}.sql"
            os.makedirs(os.path.dirname(backup_file), exist_ok=True)
            
            # 使用pg_dump命令备份
            cmd = f"PGPASSWORD={config.password} pg_dump -h{config.host} -p{config.port} -U{config.username} {config.database} > {backup_file}"
            os.system(cmd)
            
            self.backup_history.append({
                'database': config.database,
                'backup_file': backup_file,
                'timestamp': datetime.now(),
                'type': 'postgresql'
            })
            
            return os.path.exists(backup_file)
        except Exception as e:
            logger.error(f"PostgreSQL备份失败: {e}")
            return False
    
    def _backup_sqlite(self, config: DatabaseConfig, backup_path: str, timestamp: str) -> bool:
        """SQLite备份"""
        try:
            backup_file = f"{backup_path}/sqlite_{config.database}_{timestamp}.db"
            os.makedirs(os.path.dirname(backup_file), exist_ok=True)
            
            # 复制数据库文件
            import shutil
            shutil.copy2(config.database, backup_file)
            
            self.backup_history.append({
                'database': config.database,
                'backup_file': backup_file,
                'timestamp': datetime.now(),
                'type': 'sqlite'
            })
            
            return os.path.exists(backup_file)
        except Exception as e:
            logger.error(f"SQLite备份失败: {e}")
            return False
    
    def _restore_mysql(self, config: DatabaseConfig, backup_path: str) -> bool:
        """MySQL恢复"""
        try:
            # 使用mysql命令恢复
            cmd = f"mysql -h{config.host} -P{config.port} -u{config.username} -p{config.password} {config.database} < {backup_path}"
            result = os.system(cmd)
            return result == 0
        except Exception as e:
            logger.error(f"MySQL恢复失败: {e}")
            return False
    
    def _restore_postgresql(self, config: DatabaseConfig, backup_path: str) -> bool:
        """PostgreSQL恢复"""
        try:
            # 使用psql命令恢复
            cmd = f"PGPASSWORD={config.password} psql -h{config.host} -p{config.port} -U{config.username} {config.database} < {backup_path}"
            result = os.system(cmd)
            return result == 0
        except Exception as e:
            logger.error(f"PostgreSQL恢复失败: {e}")
            return False
    
    def _restore_sqlite(self, config: DatabaseConfig, backup_path: str) -> bool:
        """SQLite恢复"""
        try:
            import shutil
            shutil.copy2(backup_path, config.database)
            return True
        except Exception as e:
            logger.error(f"SQLite恢复失败: {e}")
            return False
    
    def _optimize_mysql(self, name: str) -> Dict[str, Any]:
        """MySQL优化"""
        results = {}
        
        # 优化表
        result = self.execute_query(name, "SHOW TABLES")
        tables = [list(row.values())[0] for row in result.data]
        
        for table in tables:
            try:
                self.execute_update(name, f"OPTIMIZE TABLE {table}")
                results[f"optimized_table_{table}"] = "success"
            except:
                results[f"optimized_table_{table}"] = "failed"
        
        return results
    
    def _optimize_postgresql(self, name: str) -> Dict[str, Any]:
        """PostgreSQL优化"""
        results = {}
        
        # 清理数据库
        try:
            self.execute_update(name, "VACUUM")
            results["vacuum"] = "success"
        except:
            results["vacuum"] = "failed"
        
        # 分析数据库
        try:
            self.execute_update(name, "ANALYZE")
            results["analyze"] = "success"
        except:
            results["analyze"] = "failed"
        
        return results
    
    def _optimize_sqlite(self, name: str) -> Dict[str, Any]:
        """SQLite优化"""
        results = {}
        
        try:
            self.execute_update(name, "VACUUM")
            results["vacuum"] = "success"
        except:
            results["vacuum"] = "failed"
        
        try:
            self.execute_update(name, "ANALYZE")
            results["analyze"] = "success"
        except:
            results["analyze"] = "failed"
        
        return results
    
    def close_all_connections(self):
        """关闭所有数据库连接"""
        for name, manager in self.managers.items():
            try:
                manager.disconnect()
            except Exception as e:
                logger.error(f"关闭数据库连接失败 {name}: {e}")
        
        logger.info("所有数据库连接已关闭")