#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
T5数据加载器模块 - 智能数据加载系统

该模块实现了一个功能完整的数据加载器，支持多数据源加载、增量/全量加载、
调度自动化、进度监控、错误处理、性能优化、缓存机制、安全控制和日志审计。

Author: T5系统
Date: 2025-11-05
License: MIT
Version: 1.0.0
"""

# 类型导入
from typing import Dict, Any, Union, List, Tuple
from pathlib import Path

# 版本信息
__version__ = "1.0.0"
__author__ = "T5系统"
__email__ = "support@t5.com"
__license__ = "MIT"
__copyright__ = "Copyright 2025 T5系统"

# 导出所有核心类和枚举
from .DataLoader import (
    # 枚举类型
    DataSourceType,
    LoadMode,
    LoadStatus,
    SecurityLevel,
    
    # 数据类
    DataSourceConfig,
    LoadProgress,
    LoadResult,
    
    # 管理器类
    SecurityManager,
    CacheManager,
    LoadScheduler,
    PerformanceMonitor,
    AuditLogger,
    
    # 适配器基类
    DataSourceAdapter,
    
    # 具体适配器类
    DatabaseAdapter,
    FileAdapter,
    APIAdapter,
    MessageQueueAdapter,
    
    # 主类
    DataLoader,
    
    # 装饰器
    retry_on_failure
)

# 定义包的公共API
__all__ = [
    # 版本信息
    '__version__',
    '__author__',
    '__email__',
    '__license__',
    '__copyright__',
    
    # 枚举类型
    'DataSourceType',
    'LoadMode', 
    'LoadStatus',
    'SecurityLevel',
    
    # 数据类
    'DataSourceConfig',
    'LoadProgress',
    'LoadResult',
    
    # 管理器类
    'SecurityManager',
    'CacheManager',
    'LoadScheduler',
    'PerformanceMonitor',
    'AuditLogger',
    
    # 适配器
    'DataSourceAdapter',
    'DatabaseAdapter',
    'FileAdapter',
    'APIAdapter',
    'MessageQueueAdapter',
    
    # 主类
    'DataLoader',
    
    # 装饰器
    'retry_on_failure',
    
    # 便利函数
    'create_file_source',
    'create_database_source',
    'create_api_source',
    'create_message_queue_source',
    'quick_load',
    'batch_load_sources',
    'get_source_template',
    'validate_config',
    'create_dataloader_with_defaults'
]

# 默认配置常量
DEFAULT_CONFIG = {
    # 缓存配置
    'cache_dir': 'cache',
    'max_cache_size_gb': 1.0,
    'cache_expire_hours': 24,
    
    # 安全配置
    'security_level': SecurityLevel.MEDIUM,
    'allowed_file_extensions': {'.txt', '.csv', '.json', '.xml', '.parquet'},
    'max_file_size': 100 * 1024 * 1024,  # 100MB
    
    # 性能配置
    'max_workers': 4,
    'batch_size': 1000,
    'timeout': 300,
    'retry_times': 3,
    
    # 日志配置
    'log_level': 'INFO',
    'audit_log_file': 'dataloader_audit.log',
    
    # 调度配置
    'scheduler_check_interval': 1,  # 秒
}

# 数据源类型常量
DATA_SOURCE_TYPES = {
    'DATABASE': DataSourceType.DATABASE,
    'FILE': DataSourceType.FILE,
    'API': DataSourceType.API,
    'MESSAGE_QUEUE': DataSourceType.MESSAGE_QUEUE,
    'FTP': DataSourceType.FTP,
    'S3': DataSourceType.S3
}

# 加载模式常量
LOAD_MODES = {
    'FULL': LoadMode.FULL,
    'INCREMENTAL': LoadMode.INCREMENTAL,
    'DELTA': LoadMode.DELTA
}

# 安全级别常量
SECURITY_LEVELS = {
    'LOW': SecurityLevel.LOW,
    'MEDIUM': SecurityLevel.MEDIUM,
    'HIGH': SecurityLevel.HIGH,
    'CRITICAL': SecurityLevel.CRITICAL
}

# 加载状态常量
LOAD_STATUS = {
    'PENDING': LoadStatus.PENDING,
    'RUNNING': LoadStatus.RUNNING,
    'COMPLETED': LoadStatus.COMPLETED,
    'FAILED': LoadStatus.FAILED,
    'CANCELLED': LoadStatus.CANCELLED,
    'PAUSED': LoadStatus.PAUSED
}

# 便利函数 - 快速创建数据源配置
def create_file_source(
    name: str,
    file_path: str,
    load_mode: LoadMode = LoadMode.FULL,
    **kwargs
) -> DataSourceConfig:
    """
    快速创建文件数据源配置
    
    Args:
        name: 数据源名称
        file_path: 文件路径
        load_mode: 加载模式
        **kwargs: 其他配置参数
        
    Returns:
        DataSourceConfig: 数据源配置对象
    """
    return DataSourceConfig(
        name=name,
        source_type=DataSourceType.FILE,
        connection_info={'path': file_path},
        load_mode=load_mode,
        **{k: v for k, v in kwargs.items() if k != 'connection_info'}
    )


def create_database_source(
    name: str,
    db_type: str,
    connection_params: Dict[str, Any],
    query: str = "SELECT * FROM table",
    load_mode: LoadMode = LoadMode.FULL,
    **kwargs
) -> DataSourceConfig:
    """
    快速创建数据库数据源配置
    
    Args:
        name: 数据源名称
        db_type: 数据库类型 (sqlite, mysql, postgresql)
        connection_params: 连接参数字典
        query: SQL查询语句
        load_mode: 加载模式
        **kwargs: 其他配置参数
        
    Returns:
        DataSourceConfig: 数据源配置对象
    """
    connection_info = {
        'type': db_type,
        'query': query,
        **connection_params
    }
    
    return DataSourceConfig(
        name=name,
        source_type=DataSourceType.DATABASE,
        connection_info=connection_info,
        load_mode=load_mode,
        **{k: v for k, v in kwargs.items() if k != 'connection_info'}
    )


def create_api_source(
    name: str,
    url: str,
    headers: Dict[str, str] = None,
    params: Dict[str, Any] = None,
    load_mode: LoadMode = LoadMode.INCREMENTAL,
    **kwargs
) -> DataSourceConfig:
    """
    快速创建API数据源配置
    
    Args:
        name: 数据源名称
        url: API URL
        headers: 请求头
        params: 请求参数
        load_mode: 加载模式
        **kwargs: 其他配置参数
        
    Returns:
        DataSourceConfig: 数据源配置对象
    """
    connection_info = {'url': url}
    if headers:
        connection_info['headers'] = headers
    if params:
        connection_info['params'] = params
    
    return DataSourceConfig(
        name=name,
        source_type=DataSourceType.API,
        connection_info=connection_info,
        load_mode=load_mode,
        **{k: v for k, v in kwargs.items() if k != 'connection_info'}
    )


def create_message_queue_source(
    name: str,
    queue_type: str = 'memory',
    queue_name: str = 'default',
    max_messages: int = 1000,
    **kwargs
) -> DataSourceConfig:
    """
    快速创建消息队列数据源配置
    
    Args:
        name: 数据源名称
        queue_type: 队列类型 (memory, redis, rabbitmq)
        queue_name: 队列名称
        max_messages: 最大消息数
        **kwargs: 其他配置参数
        
    Returns:
        DataSourceConfig: 数据源配置对象
    """
    connection_info = {
        'type': queue_type,
        'queue_name': queue_name,
        'max_messages': max_messages
    }
    
    return DataSourceConfig(
        name=name,
        source_type=DataSourceType.MESSAGE_QUEUE,
        connection_info=connection_info,
        load_mode=LoadMode.INCREMENTAL,
        **{k: v for k, v in kwargs.items() if k != 'connection_info'}
    )

# 便利函数 - 快速加载数据
def quick_load(
    source_type: str,
    source_config: Union[str, Dict[str, Any]],
    **kwargs
) -> LoadResult:
    """
    快速加载数据（无需预注册数据源）
    
    Args:
        source_type: 数据源类型 ('file', 'api', 'database')
        source_config: 数据源配置（文件路径或配置字典）
        **kwargs: 加载参数
        
    Returns:
        LoadResult: 加载结果
        
    Examples:
        >>> # 快速加载文件
        >>> result = quick_load('file', 'data.json')
        >>> 
        >>> # 快速加载API
        >>> result = quick_load('api', {'url': 'https://api.example.com/data'})
    """
    # 创建临时数据加载器
    temp_dataloader = DataLoader()
    
    try:
        if source_type == 'file':
            config = create_file_source('temp_file', source_config)
        elif source_type == 'api':
            config = create_api_source('temp_api', **source_config)
        elif source_type == 'database':
            config = create_database_source('temp_db', **source_config)
        else:
            raise ValueError(f"不支持的数据源类型: {source_type}")
        
        # 添加数据源
        if temp_dataloader.add_data_source(config):
            # 加载数据
            result = temp_dataloader.load_source(config.name, **kwargs)
            return result
        else:
            raise ValueError("数据源添加失败")
            
    finally:
        # 清理资源
        temp_dataloader._executor.shutdown(wait=False)


def batch_load_sources(
    configs: List[DataSourceConfig],
    **kwargs
) -> Dict[str, LoadResult]:
    """
    批量加载多个数据源
    
    Args:
        configs: 数据源配置列表
        **kwargs: 加载参数
        
    Returns:
        Dict[str, LoadResult]: 各数据源的加载结果
        
    Examples:
        >>> configs = [
        ...     create_file_source('file1', 'data1.json'),
        ...     create_api_source('api1', 'https://api.example.com/data')
        ... ]
        >>> results = batch_load_sources(configs)
    """
    with DataLoader() as dataloader:
        # 添加所有数据源
        for config in configs:
            dataloader.add_data_source(config)
        
        # 批量加载
        return dataloader.load_all(**kwargs)

# 工具函数
def get_source_template(source_type: str) -> Dict[str, Any]:
    """
    获取数据源配置模板
    
    Args:
        source_type: 数据源类型
        
    Returns:
        Dict[str, Any]: 配置模板
        
    Examples:
        >>> template = get_source_template('file')
        >>> print(template)
    """
    templates = {
        'file': {
            'name': '示例文件数据源',
            'source_type': 'file',
            'connection_info': {
                'path': '/path/to/your/file.json'
            },
            'load_mode': 'full',
            'cache_enabled': True,
            'batch_size': 1000
        },
        'database': {
            'name': '示例数据库数据源',
            'source_type': 'database',
            'connection_info': {
                'type': 'sqlite',
                'path': 'database.db',
                'query': 'SELECT * FROM your_table'
            },
            'load_mode': 'full',
            'cache_enabled': True,
            'timeout': 300
        },
        'api': {
            'name': '示例API数据源',
            'source_type': 'api',
            'connection_info': {
                'url': 'https://api.example.com/data',
                'headers': {'User-Agent': 'T5-DataLoader/1.0'},
                'params': {}
            },
            'load_mode': 'incremental',
            'cache_enabled': True,
            'timeout': 30
        },
        'message_queue': {
            'name': '示例消息队列数据源',
            'source_type': 'message_queue',
            'connection_info': {
                'type': 'memory',
                'queue_name': 'default',
                'max_messages': 1000
            },
            'load_mode': 'incremental',
            'cache_enabled': False
        }
    }
    
    return templates.get(source_type, {})


def validate_config(config: DataSourceConfig) -> Tuple[bool, List[str]]:
    """
    验证数据源配置
    
    Args:
        config: 数据源配置
        
    Returns:
        Tuple[bool, List[str]]: (是否有效, 错误信息列表)
    """
    errors = []
    
    # 基本字段检查
    if not config.name or not config.name.strip():
        errors.append("数据源名称不能为空")
    
    if not config.connection_info:
        errors.append("连接信息不能为空")
    
    # 类型特定验证
    if config.source_type == DataSourceType.FILE:
        file_path = config.connection_info.get('path')
        if not file_path:
            errors.append("文件路径不能为空")
        elif not Path(file_path).exists():
            errors.append(f"文件不存在: {file_path}")
    
    elif config.source_type == DataSourceType.API:
        url = config.connection_info.get('url')
        if not url:
            errors.append("API URL不能为空")
        elif not url.startswith(('http://', 'https://')):
            errors.append("API URL格式不正确")
    
    elif config.source_type == DataSourceType.DATABASE:
        if 'type' not in config.connection_info:
            errors.append("数据库类型未指定")
    
    return len(errors) == 0, errors


def create_dataloader_with_defaults(**kwargs) -> DataLoader:
    """
    使用默认配置创建数据加载器
    
    Args:
        **kwargs: 自定义配置参数
        
    Returns:
        DataLoader: 配置好的数据加载器实例
    """
    config = DEFAULT_CONFIG.copy()
    config.update(kwargs)
    
    return DataLoader(
        cache_dir=config['cache_dir'],
        max_cache_size_gb=config['max_cache_size_gb'],
        log_level=config['log_level']
    )

# 快速入门指南
QUICK_START_GUIDE = """
=== T5数据加载器快速入门 ===

1. 基本使用
-----------
from T5 import DataLoader, DataSourceType, LoadMode, DataSourceConfig

# 创建数据加载器
dataloader = DataLoader()

# 创建数据源配置
config = DataSourceConfig(
    name="my_source",
    source_type=DataSourceType.FILE,
    connection_info={"path": "data.json"},
    load_mode=LoadMode.FULL
)

# 添加数据源
dataloader.add_data_source(config)

# 加载数据
result = dataloader.load_source("my_source")

2. 便利函数使用
--------------
from T5 import create_file_source, quick_load

# 快速创建文件数据源
config = create_file_source("file1", "data.json")

# 快速加载数据
result = quick_load("file", "data.json")

3. 批量加载
----------
from T5 import create_api_source, batch_load_sources

configs = [
    create_file_source("file1", "data1.json"),
    create_api_source("api1", "https://api.example.com/data")
]

results = batch_load_sources(configs)

4. 调度功能
----------
from T5 import create_dataloader_with_defaults

dataloader = create_dataloader_with_defaults()

# 添加调度任务（每60秒执行一次）
dataloader.add_schedule("my_source", 60)
dataloader.start_scheduler()

5. 性能监控
----------
# 获取加载进度
progress = dataloader.get_progress("my_source")
print(f"已加载: {progress.loaded_records} 记录")

# 获取性能报告
report = dataloader.get_performance_report("my_source")
print(f"处理速度: {report['speed']:.2f} 记录/秒")

6. 配置管理
----------
# 导出配置
dataloader.export_config("my_config.json")

# 导入配置
dataloader.import_config("my_config.json")

更多详细信息请参考官方文档。
"""

# 模块初始化日志
import logging
logger = logging.getLogger(__name__)
logger.info(f"T5数据加载器模块已加载 - 版本: {__version__}")

# 导出便捷访问
class T5Loader:
    """T5数据加载器便捷访问类"""
    
    def __init__(self, **kwargs):
        self._dataloader = create_dataloader_with_defaults(**kwargs)
    
    def load_file(self, file_path: str, name: str = None) -> LoadResult:
        """快速加载文件"""
        if name is None:
            name = f"file_{Path(file_path).stem}"
        config = create_file_source(name, file_path)
        self._dataloader.add_data_source(config)
        return self._dataloader.load_source(name)
    
    def load_api(self, url: str, name: str = None, **kwargs) -> LoadResult:
        """快速加载API"""
        if name is None:
            name = f"api_{hash(url) % 10000}"
        config = create_api_source(name, url, **kwargs)
        self._dataloader.add_data_source(config)
        return self._dataloader.load_source(name)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._dataloader.__exit__(exc_type, exc_val, exc_tb)

# 默认导出实例
default_loader = T5Loader()

# 为向后兼容，导出便捷函数到模块级别
def load_file(file_path: str, name: str = None, **kwargs) -> LoadResult:
    """便捷的文件加载函数"""
    return default_loader.load_file(file_path, name, **kwargs)

def load_api(url: str, name: str = None, **kwargs) -> LoadResult:
    """便捷的API加载函数"""
    return default_loader.load_api(url, name, **kwargs)

# 添加到__all__
__all__.extend(['T5Loader', 'default_loader', 'load_file', 'load_api'])