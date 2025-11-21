#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
R8归档管理器 - 包初始化文件

提供全面的数据归档管理功能，包括：
- 数据归档和存储
- 多算法压缩支持
- 灵活归档策略
- 快速检索索引
- 完整数据验证
- 生命周期管理
- 存储优化
- 并发安全
- 性能监控
- 归档规则管理
- 归档恢复
- 存储迁移

作者: R8归档管理器团队
版本: 1.0.0
"""

# 核心类
from .ArchiveManager import (
    # 主要类
    ArchiveManager,
    create_archive_manager,
    
    # 归档条目
    ArchiveEntry,
    
    # 归档规则
    ArchiveRule,
    
    # 归档策略
    ArchiveStrategy,
    
    # 压缩系统
    CompressionType,
    
    # 状态管理
    ArchiveStatus,
    
    # 便利函数
    create_archive,
    extract_archive,
    list_archives,
    validate_archive,
    optimize_storage,
    search_archive,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "R8归档管理器团队"
__email__ = "support@r8archive.com"
__description__ = "R8归档管理器 - 企业级数据归档管理系统"

# 包级别的默认配置
DEFAULT_CONFIG = {
    'archive_dir': './archives',
    'temp_dir': './archive_temp',
    'index_dir': './archive_index',
    'max_archive_size': '10GB',
    'compression': {
        'default_type': 'gzip',
        'compression_level': 6,
        'auto_compress': True,
        'threshold_size': '100MB'
    },
    'indexing': {
        'auto_index': True,
        'index_interval': 3600,
        'full_text_search': True
    },
    'validation': {
        'auto_validate': True,
        'checksum_validation': True,
        'integrity_check': True
    },
    'lifecycle': {
        'auto_cleanup': True,
        'retention_days': 365,
        'cleanup_interval': 86400
    },
    'performance': {
        'max_concurrent_operations': 4,
        'buffer_size': '64MB',
        'cache_size': '256MB'
    }
}

# 压缩类型
COMPRESSION_TYPES = [
    'NONE',        # 无压缩
    'GZIP',        # GZIP压缩
    'BZ2',         # BZ2压缩
    'LZMA',        # LZMA压缩
    'ZLIB',        # ZLIB压缩
    'LZ4',         # LZ4压缩
    'SNAPPY',      # Snappy压缩
    'ZSTD',        # Zstandard压缩
]

# 归档状态
ARCHIVE_STATUS = [
    'CREATED',         # 已创建
    'COMPRESSING',     # 压缩中
    'COMPRESSED',      # 已压缩
    'VALIDATING',      # 验证中
    'VALIDATED',       # 已验证
    'STORED',          # 已存储
    'ARCHIVED',        # 已归档
    'RESTORING',       # 恢复中
    'EXPIRED',         # 已过期
    'DELETED',         # 已删除
]

# 归档策略
ARCHIVE_STRATEGIES = [
    'SIZE_BASED',      # 基于大小
    'TIME_BASED',      # 基于时间
    'ACCESS_BASED',    # 基于访问
    'CUSTOM',          # 自定义
    'HYBRID',          # 混合策略
]

# 压缩级别
COMPRESSION_LEVELS = [
    'FASTEST',     # 最快
    'FAST',        # 快速
    'DEFAULT',     # 默认
    'MAXIMUM',     # 最大压缩
]

# 验证级别
VALIDATION_LEVELS = [
    'BASIC',       # 基础验证
    'STANDARD',    # 标准验证
    'STRICT',      # 严格验证
    'COMPREHENSIVE', # 全面验证
]

# 检索类型
SEARCH_TYPES = [
    'EXACT',       # 精确匹配
    'FUZZY',       # 模糊匹配
    'REGEX',       # 正则表达式
    'FULL_TEXT',   # 全文搜索
]

# 存储类型
STORAGE_TYPES = [
    'LOCAL',       # 本地存储
    'NETWORK',     # 网络存储
    'CLOUD',       # 云存储
    'HYBRID',      # 混合存储
]

# 生命周期状态
LIFECYCLE_STATES = [
    'ACTIVE',      # 活跃
    'ARCHIVED',    # 已归档
    'EXPIRED',     # 已过期
    'DELETED',     # 已删除
]

# 公开的API函数
__all__ = [
    # 核心类
    'ArchiveManager',
    'create_archive_manager',
    
    # 归档条目
    'ArchiveEntry',
    
    # 归档规则
    'ArchiveRule',
    
    # 归档策略
    'ArchiveStrategy',
    
    # 压缩系统
    'CompressionType',
    
    # 状态管理
    'ArchiveStatus',
    
    # 便利函数
    'create_archive',
    'extract_archive',
    'list_archives',
    'validate_archive',
    'optimize_storage',
    'search_archive',
    
    # 常量
    '__version__',
    '__author__',
    '__email__',
    '__description__',
    'DEFAULT_CONFIG',
    'COMPRESSION_TYPES',
    'ARCHIVE_STATUS',
    'ARCHIVE_STRATEGIES',
    'COMPRESSION_LEVELS',
    'VALIDATION_LEVELS',
    'SEARCH_TYPES',
    'STORAGE_TYPES',
    'LIFECYCLE_STATES',
]

# 包信息
PACKAGE_INFO = {
    'name': 'R8归档管理器',
    'version': __version__,
    'description': '企业级数据归档管理系统',
    'author': __author__,
    'email': __email__,
    'features': [
        '多算法压缩存储',
        '灵活归档策略',
        '快速检索索引',
        '完整数据验证',
        '生命周期管理',
        '存储优化',
        '并发安全',
        '性能监控'
    ]
}

def get_package_info():
    """获取包信息"""
    return PACKAGE_INFO.copy()

# 快速入门指南
def quick_start():
    """
    快速入门指南
    
    返回一个包含基本使用示例的字符串。
    """
    return """
    R8归档管理器快速入门
    ====================
    
    1. 创建归档:
       ```python
       from R8 import create_archive
       
       archive = create_archive(
           source_path="/path/to/data",
           archive_path="/path/to/archive.tar.gz",
           compression_type="gzip"
       )
       ```
    
    2. 提取归档:
       ```python
       from R8 import extract_archive
       
       result = extract_archive(
           archive_path="/path/to/archive.tar.gz",
           extract_path="/path/to/extract"
       )
       ```
    
    3. 列出归档:
       ```python
       from R8 import list_archives
       
       archives = list_archives(
           archive_dir="/path/to/archives",
           include_metadata=True
       )
       ```
    
    4. 验证归档:
       ```python
       from R8 import validate_archive
       
       is_valid = validate_archive(
           archive_path="/path/to/archive.tar.gz",
           validation_level="STRICT"
       )
       ```
    
    5. 搜索归档:
       ```python
       from R8 import search_archive
       
       results = search_archive(
           archive_dir="/path/to/archives",
           query="*.txt",
           search_type="REGEX"
       )
       ```
    
    6. 优化存储:
       ```python
       from R8 import optimize_storage
       
       optimization = optimize_storage(
           archive_dir="/path/to/archives",
           strategy="size_based",
           compression_level="MAXIMUM"
       )
       ```
    
    更多信息请查看文档或运行测试文件。
    """

# 便利函数实现
def create_archive(source_path, archive_path, compression_type='gzip', **kwargs):
    """
    创建归档的便利函数
    """
    archive_manager = ArchiveManager()
    return archive_manager.create_archive(source_path, archive_path, compression_type, **kwargs)

def extract_archive(archive_path, extract_path, **kwargs):
    """
    提取归档的便利函数
    """
    archive_manager = ArchiveManager()
    return archive_manager.extract_archive(archive_path, extract_path, **kwargs)

def list_archives(archive_dir, include_metadata=False, **kwargs):
    """
    列出归档的便利函数
    """
    archive_manager = ArchiveManager()
    return archive_manager.list_archives(archive_dir, include_metadata, **kwargs)

def validate_archive(archive_path, validation_level='STANDARD', **kwargs):
    """
    验证归档的便利函数
    """
    archive_manager = ArchiveManager()
    return archive_manager.validate_archive(archive_path, validation_level, **kwargs)

def optimize_storage(archive_dir, **kwargs):
    """
    优化存储的便利函数
    """
    archive_manager = ArchiveManager()
    return archive_manager.optimize_storage(archive_dir, **kwargs)

def search_archive(archive_dir, query, **kwargs):
    """
    搜索归档的便利函数
    """
    archive_manager = ArchiveManager()
    return archive_manager.search_archive(archive_dir, query, **kwargs)

def check_dependencies():
    """检查依赖项"""
    import sys
    if sys.version_info < (3, 7):
        raise RuntimeError("R8归档管理器需要Python 3.7或更高版本")
    
    # 检查可选依赖
    optional_deps = {
        'sqlite3': '内置模块',
        'hashlib': '内置模块',
        'threading': '内置模块',
        'datetime': '内置模块',
        'pathlib': '内置模块',
        'gzip': '内置模块',
        'bz2': '内置模块',
        'lzma': '内置模块',
        'zlib': '内置模块'
    }
    
    missing_deps = []
    for dep, desc in optional_deps.items():
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(f"{dep} ({desc})")
    
    if missing_deps:
        print(f"警告: 缺少以下依赖项: {', '.join(missing_deps)}")
    
    return len(missing_deps) == 0

# 版本检查
def check_version():
    """检查版本信息"""
    import sys
    print(f"R8归档管理器版本: {__version__}")
    print(f"Python版本: {sys.version}")
    print(f"作者: {__author__}")
    print(f"描述: {__description__}")

# 初始化时检查依赖
try:
    check_dependencies()
except Exception as e:
    print(f"依赖检查失败: {e}")

# 包初始化完成标志
_PACKAGE_INITIALIZED = True

if __name__ == "__main__":
    check_version()
    print(quick_start())