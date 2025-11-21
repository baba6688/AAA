"""
T1数据仓库管理器导出接口

提供完整的企业级数据仓库管理功能，支持多数据源集成、智能分区、
性能监控、备份恢复等核心能力。

Version: 1.0.0
Author: T1系统团队
Date: 2025-11-13
License: MIT
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "T1系统团队"
__email__ = "t1-team@company.com"
__license__ = "MIT"
__copyright__ = "Copyright 2025 T1系统团队"
__description__ = "T1企业级数据仓库管理器"

# 核心组件导出
from .DataWarehouseManager import (
    # 枚举类型
    DataSourceType,
    PartitionType, 
    IndexType,
    CompressionType,
    
    # 数据结构类
    DataSource,
    DataPartition,
    DataIndex,
    DataSchema,
    PerformanceMetrics,
    BackupInfo,
    
    # 主管理器类
    DataWarehouseManager
)

# 默认配置常量
DEFAULT_CONFIG = {
    # 基础配置
    "base_path": "./warehouse",
    "max_connections": 100,
    "cache_size": 1000,
    
    # 压缩配置
    "enable_compression": True,
    "default_compression_type": "gzip",
    
    # 安全配置
    "enable_encryption": False,
    
    # 备份配置
    "backup_retention_days": 30,
    "backup_interval_hours": 24,
    
    # 性能配置
    "monitoring_interval_seconds": 60,
    "performance_history_hours": 24,
    "max_performance_records": 1440,
    
    # 分区配置
    "default_partition_type": "time",
    "max_partition_size_mb": 1000,
    "auto_optimize_partitions": True,
    
    # 索引配置
    "auto_create_indexes": True,
    "index_rebuild_threshold_mb": 100,
    "index_rebuild_days": 7,
    
    # 缓存配置
    "cache_ttl_seconds": 3600,
    "cache_eviction_policy": "LRU",
    
    # 存储配置
    "storage_format": "parquet",
    "compression_level": 6,
    "chunk_size": 10000
}

# 数据类型映射
DATA_TYPE_MAPPING = {
    "int": "int64",
    "integer": "int64", 
    "bigint": "int64",
    "long": "int64",
    "float": "float64",
    "double": "float64",
    "decimal": "float64",
    "string": "object",
    "varchar": "object",
    "text": "object",
    "char": "object",
    "boolean": "bool",
    "bool": "bool",
    "datetime": "datetime64[ns]",
    "timestamp": "datetime64[ns]",
    "date": "datetime64[ns]",
    "time": "datetime64[ns]"
}

# 系统限制常量
SYSTEM_LIMITS = {
    "max_data_sources": 100,
    "max_tables_per_source": 1000,
    "max_partitions_per_table": 10000,
    "max_indexes_per_table": 100,
    "max_backup_size_gb": 1000,
    "max_concurrent_connections": 1000,
    "max_cache_size_mb": 10240,  # 10GB
    "max_metadata_file_size_mb": 100,
    "max_audit_logs": 10000
}

# 性能阈值常量
PERFORMANCE_THRESHOLDS = {
    "query_time_warn_ms": 1000,    # 查询时间警告阈值(毫秒)
    "query_time_crit_ms": 5000,    # 查询时间严重阈值(毫秒)
    "memory_usage_warn_percent": 80,    # 内存使用警告阈值(%)
    "memory_usage_crit_percent": 95,    # 内存使用严重阈值(%)
    "disk_usage_warn_percent": 85,     # 磁盘使用警告阈值(%)
    "disk_usage_crit_percent": 95,     # 磁盘使用严重阈值(%)
    "cache_hit_rate_min_percent": 70,  # 缓存命中率最小值(%)
    "error_rate_max_percent": 5,       # 错误率最大值(%)
    "throughput_min_qps": 10           # 最小吞吐量(QPS)
}

# 日志级别配置
LOG_LEVELS = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50
}

# 备份类型常量
BACKUP_TYPES = {
    "FULL": "full",
    "INCREMENTAL": "incremental", 
    "DIFFERENTIAL": "differential"
}

# 存储格式常量
STORAGE_FORMATS = {
    "PARQUET": "parquet",
    "CSV": "csv",
    "JSON": "json",
    "AVRO": "avro",
    "ORC": "orc"
}

# 便利函数

def create_data_source(name: str, 
                      source_type: str,
                      connection_string: str,
                      schema: str = "public",
                      tables: list = None,
                      credentials: dict = None,
                      metadata: dict = None) -> DataSource:
    """
    创建数据源的便利函数
    
    Args:
        name: 数据源名称
        source_type: 数据源类型 (database/file/api/stream/cache)
        connection_string: 连接字符串
        schema: 数据库模式
        tables: 表列表
        credentials: 凭据字典
        metadata: 元数据字典
        
    Returns:
        DataSource: 数据源对象
    """
    from datetime import datetime
    
    if tables is None:
        tables = []
    if credentials is None:
        credentials = {}
    if metadata is None:
        metadata = {}
    
    return DataSource(
        name=name,
        type=DataSourceType(source_type),
        connection_string=connection_string,
        schema=schema,
        tables=tables,
        credentials=credentials,
        metadata=metadata,
        last_updated=datetime.now()
    )


def create_data_schema(table_name: str,
                      columns: dict,
                      primary_key: list = None,
                      foreign_keys: dict = None,
                      constraints: list = None,
                      indexes: list = None) -> DataSchema:
    """
    创建数据模式的便利函数
    
    Args:
        table_name: 表名
        columns: 列定义字典 {column_name: data_type}
        primary_key: 主键列列表
        foreign_keys: 外键字典 {column: reference_table.column}
        constraints: 约束列表
        indexes: 索引列表
        
    Returns:
        DataSchema: 数据模式对象
    """
    if primary_key is None:
        primary_key = []
    if foreign_keys is None:
        foreign_keys = {}
    if constraints is None:
        constraints = []
    if indexes is None:
        indexes = []
    
    return DataSchema(
        table_name=table_name,
        columns=columns,
        primary_key=primary_key,
        foreign_keys=foreign_keys,
        constraints=constraints,
        indexes=indexes
    )


def create_data_index(name: str,
                     table: str,
                     columns: list,
                     index_type: str = "btree",
                     unique: bool = False,
                     size: int = 0) -> DataIndex:
    """
    创建数据索引的便利函数
    
    Args:
        name: 索引名称
        table: 表名
        columns: 索引列列表
        index_type: 索引类型 (btree/hash/bitmap/fulltext)
        unique: 是否唯一索引
        size: 索引大小(字节)
        
    Returns:
        DataIndex: 数据索引对象
    """
    from datetime import datetime
    
    return DataIndex(
        name=name,
        table=table,
        columns=columns,
        type=IndexType(index_type),
        unique=unique,
        size=size,
        created_at=datetime.now()
    )


def create_partition(name: str,
                    partition_type: str,
                    column: str,
                    value,
                    size: int = 0,
                    record_count: int = 0,
                    compressed: bool = True,
                    compression_type: str = "gzip") -> DataPartition:
    """
    创建数据分区的便利函数
    
    Args:
        name: 分区名称
        partition_type: 分区类型 (time/hash/range/list)
        column: 分区列
        value: 分区值
        size: 分区大小(字节)
        record_count: 记录数
        compressed: 是否压缩
        compression_type: 压缩类型
        
    Returns:
        DataPartition: 数据分区对象
    """
    from datetime import datetime
    
    return DataPartition(
        name=name,
        type=PartitionType(partition_type),
        column=column,
        value=value,
        size=size,
        record_count=record_count,
        created_at=datetime.now(),
        compressed=compressed,
        compression_type=CompressionType(compression_type)
    )


def create_manager(warehouse_id: str,
                  base_path: str = None,
                  config: dict = None) -> DataWarehouseManager:
    """
    创建数据仓库管理器的便利函数
    
    Args:
        warehouse_id: 仓库唯一标识符
        base_path: 基础路径
        config: 配置字典
        
    Returns:
        DataWarehouseManager: 数据仓库管理器对象
    """
    if config is None:
        config = {}
    if base_path is None:
        base_path = DEFAULT_CONFIG["base_path"]
    
    # 合并默认配置和用户配置
    merged_config = {**DEFAULT_CONFIG, **config}
    
    return DataWarehouseManager(
        warehouse_id=warehouse_id,
        base_path=base_path,
        max_connections=merged_config.get("max_connections", 100),
        cache_size=merged_config.get("cache_size", 1000),
        enable_compression=merged_config.get("enable_compression", True),
        compression_type=CompressionType(merged_config.get("default_compression_type", "gzip")),
        enable_encryption=merged_config.get("enable_encryption", False),
        backup_retention_days=merged_config.get("backup_retention_days", 30)
    )


def validate_connection_string(connection_string: str, source_type: str) -> bool:
    """
    验证连接字符串格式
    
    Args:
        connection_string: 连接字符串
        source_type: 数据源类型
        
    Returns:
        bool: 验证结果
    """
    import re
    
    patterns = {
        "database": r"^(mysql|postgresql|sqlite|oracle|sqlserver)://.+",
        "file": r"^(file|ftp|s3|hdfs)://.+", 
        "api": r"^https?://.+",
        "stream": r"^(kafka|redis|mqtt)://.+",
        "cache": r"^(redis|memcached)://.+"
    }
    
    pattern = patterns.get(source_type.lower())
    if not pattern:
        return False
        
    return bool(re.match(pattern, connection_string))


def format_bytes(bytes_value: int) -> str:
    """
    格式化字节数显示
    
    Args:
        bytes_value: 字节数
        
    Returns:
        str: 格式化后的字符串
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} PB"


def format_duration(seconds: float) -> str:
    """
    格式化持续时间显示
    
    Args:
        seconds: 秒数
        
    Returns:
        str: 格式化后的字符串
    """
    if seconds < 1:
        return f"{seconds*1000:.2f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"


# 快速入门指南
QUICK_START_GUIDE = """
=== T1数据仓库管理器快速入门指南 ===

1. 基本使用流程：

   # 导入模块
   from T.T1 import create_manager, create_data_source, create_data_schema
   
   # 创建管理器
   manager = create_manager("my_warehouse", "./warehouse_data")
   
   # 注册数据源
   source = create_data_source(
       name="my_db",
       source_type="database", 
       connection_string="sqlite:///mydb.db"
   )
   manager.register_data_source(source)
   
   # 定义数据模式
   schema = create_data_schema(
       table_name="users",
       columns={"id": "int", "name": "string", "email": "string"},
       primary_key=["id"]
   )
   manager.register_schema(schema)
   
   # 存储数据
   import pandas as pd
   data = pd.DataFrame({
       "id": [1, 2, 3],
       "name": ["Alice", "Bob", "Charlie"],
       "email": ["alice@email.com", "bob@email.com", "charlie@email.com"]
   })
   manager.store_data("users", data)

2. 核心功能模块：

   - 数据源管理：register_data_source(), get_data_source(), list_data_sources()
   - 数据模式：register_schema(), get_schema(), validate_data()
   - 数据分区：create_partition(), get_partitions(), optimize_partitions()
   - 索引管理：create_index(), get_indexes(), optimize_indexes()
   - 数据存储：store_data(), retrieve_data()
   - 性能监控：get_performance_metrics(), get_performance_summary()
   - 备份恢复：create_backup(), restore_backup(), list_backups()
   - 安全控制：set_access_control(), check_permission(), audit_access()
   - 系统状态：get_system_status(), add_storage_node(), redistribute_data()

3. 配置选项：

   可以通过config参数自定义配置：
   
   config = {
       "max_connections": 200,
       "cache_size": 2000,
       "enable_compression": True,
       "backup_retention_days": 7
   }
   manager = create_manager("my_warehouse", config=config)

4. 最佳实践：

   - 使用上下文管理器自动管理资源：
     with create_manager("my_warehouse") as manager:
         # 你的代码
         pass
         
   - 定期创建备份：
     backup_id = manager.create_backup("incremental")
     
   - 监控性能指标：
     summary = manager.get_performance_summary(24)
     
   - 优化分区和索引：
     manager.optimize_partitions("users")
     manager.optimize_indexes("users")

5. 错误处理：

   所有主要方法都返回布尔值表示成功/失败，建议检查返回值：
   
   if not manager.store_data("users", data):
       print("数据存储失败")

6. 日志和调试：

   日志文件位于仓库目录的logs子目录中，可以通过get_system_status()查看系统状态。
"""

# 导出所有公共接口
__all__ = [
    # 版本信息
    "__version__",
    "__author__", 
    "__email__",
    "__license__",
    "__copyright__",
    "__description__",
    
    # 核心类
    "DataSourceType",
    "PartitionType",
    "IndexType", 
    "CompressionType",
    "DataSource",
    "DataPartition",
    "DataIndex",
    "DataSchema",
    "PerformanceMetrics",
    "BackupInfo",
    "DataWarehouseManager",
    
    # 默认配置
    "DEFAULT_CONFIG",
    "DATA_TYPE_MAPPING",
    "SYSTEM_LIMITS",
    "PERFORMANCE_THRESHOLDS",
    "LOG_LEVELS",
    "BACKUP_TYPES",
    "STORAGE_FORMATS",
    
    # 便利函数
    "create_data_source",
    "create_data_schema",
    "create_data_index", 
    "create_partition",
    "create_manager",
    "validate_connection_string",
    "format_bytes",
    "format_duration",
    
    # 指南
    "QUICK_START_GUIDE"
]

# 模块初始化日志
def _log_init():
    """模块初始化日志"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"T1数据仓库管理器 v{__version__} 初始化完成")
    logger.info(f"默认配置: {len(DEFAULT_CONFIG)} 项配置参数")
    logger.info(f"支持的数据类型: {len(DATA_TYPE_MAPPING)} 种")
    logger.info(f"系统限制: {len(SYSTEM_LIMITS)} 项")
    logger.info(f"性能阈值: {len(PERFORMANCE_THRESHOLDS)} 项")

# 在模块导入时执行初始化
_log_init()