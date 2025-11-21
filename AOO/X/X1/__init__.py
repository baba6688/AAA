#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X1内存缓存管理器包

这是一个功能完整的内存缓存管理器，支持：
- LRU和TTL缓存清理策略
- 线程安全访问
- 内存使用限制和监控
- 缓存统计和监控
- 缓存持久化存储
- 缓存键过期管理

主要类和接口：
- MemoryCacheManager: 主要的内存缓存管理器
- CacheEntry: 缓存条目数据结构
- CacheStatistics: 缓存统计信息
- EvictionStrategy: 缓存清理策略基类
- LRUEvictionStrategy: LRU清理策略
- TTLEvictionStrategy: TTL清理策略
- SizeEvictionStrategy: 大小限制清理策略
- CacheConfigs: 预定义的缓存配置类

便捷函数：
- create_cache_manager: 创建缓存管理器
- get_cache: 获取全局缓存管理器
- clear_global_cache: 清空全局缓存

配置管理：
- CacheConfigs: 提供预定义的缓存配置(SMALL, MEDIUM, LARGE, PERSISTENT, UNLIMITED)
- quick_cache: 快速创建简单缓存管理器的便捷函数

版本信息：
- Version: 1.0.0
- Author: X1 Team
- Email: team@x1.com
"""

# 导入所有核心类
from .MemoryCacheManager import (
    # 主要类
    MemoryCacheManager,
    CacheEntry,
    CacheStatistics,
    
    # 清理策略
    EvictionStrategy,
    LRUEvictionStrategy,
    TTLEvictionStrategy,
    SizeEvictionStrategy,
    
    # 便捷函数
    create_cache_manager,
    get_global_cache,
    set_global_cache,
    get_cache,
    clear_global_cache
)

# 导入配置类（已在本文件中定义）
# CacheConfigs 类已在文件底部定义

__version__ = "1.0.0"
__author__ = "X1 Team"
__email__ = "team@x1.com"

# 定义完整的导出接口
__all__ = [
    # 核心类 (8个主要类)
    "MemoryCacheManager",     # 主要的内存缓存管理器
    "CacheEntry",             # 缓存条目数据结构
    "CacheStatistics",        # 缓存统计信息
    "EvictionStrategy",       # 缓存清理策略基类
    "LRUEvictionStrategy",    # LRU清理策略
    "TTLEvictionStrategy",    # TTL清理策略
    "SizeEvictionStrategy",   # 大小限制清理策略
    "CacheConfigs",           # 预定义的缓存配置类
    
    # 便捷函数
    "create_cache_manager",   # 创建缓存管理器
    "get_global_cache",       # 获取全局缓存管理器实例
    "set_global_cache",       # 设置全局缓存管理器实例
    "get_cache",             # 获取或创建全局缓存管理器
    "clear_global_cache",    # 清空全局缓存
    "quick_cache",           # 快速创建简单缓存管理器
    
    # 元信息
    "__version__",           # 版本信息
    "__author__",           # 作者信息
    "__email__",           # 邮箱信息
    "__all__"              # 导出列表本身
]

# 包级别的便捷访问
def quick_cache(max_size=1000, default_ttl=None):
    """
    快速创建简单缓存管理器的便捷函数
    
    Args:
        max_size: 最大缓存条目数量
        default_ttl: 默认生存时间（秒）
        
    Returns:
        MemoryCacheManager: 缓存管理器实例
    """
    return MemoryCacheManager(
        max_size=max_size,
        default_ttl=default_ttl,
        enable_monitoring=True
    )

# 预定义的缓存配置
class CacheConfigs:
    """预定义的缓存配置"""
    
    # 小型缓存（适合会话数据）
    SMALL = {
        'max_size': 100,
        'max_memory_bytes': 10 * 1024 * 1024,  # 10MB
        'default_ttl': 300,  # 5分钟
        'cleanup_interval': 60  # 1分钟清理一次
    }
    
    # 中型缓存（适合应用缓存）
    MEDIUM = {
        'max_size': 1000,
        'max_memory_bytes': 100 * 1024 * 1024,  # 100MB
        'default_ttl': 1800,  # 30分钟
        'cleanup_interval': 300  # 5分钟清理一次
    }
    
    # 大型缓存（适合数据缓存）
    LARGE = {
        'max_size': 10000,
        'max_memory_bytes': 1024 * 1024 * 1024,  # 1GB
        'default_ttl': 3600,  # 1小时
        'cleanup_interval': 600  # 10分钟清理一次
    }
    
    # 持久化缓存（适合重要数据）
    PERSISTENT = {
        'max_size': 5000,
        'max_memory_bytes': 500 * 1024 * 1024,  # 500MB
        'default_ttl': 7200,  # 2小时
        'cleanup_interval': 900,  # 15分钟清理一次
        'enable_persistence': True
    }
    
    # 无限制缓存（适合测试）
    UNLIMITED = {
        'max_size': 0,  # 无限制
        'max_memory_bytes': 0,  # 无限制
        'default_ttl': None,  # 永不过期
        'cleanup_interval': 0  # 不自动清理
    }


# 导出便捷函数到模块级别
quick_cache = quick_cache  # 别名

# 包初始化信息
def _init_package():
    """包初始化时的处理"""
    import sys
    if 'X.X1' not in sys.modules:
        return
    # 可以在这里添加包级别的初始化逻辑
    pass

# 包级别的元数据
_PACKAGE_INFO = {
    'name': 'X.X1',
    'description': 'X1内存缓存管理器 - 功能完整的内存缓存解决方案',
    'version': __version__,
    'author': __author__,
    'email': __email__,
    'classes': [
        'MemoryCacheManager', 'CacheEntry', 'CacheStatistics',
        'EvictionStrategy', 'LRUEvictionStrategy', 'TTLEvictionStrategy', 
        'SizeEvictionStrategy', 'CacheConfigs'
    ],
    'functions': [
        'create_cache_manager', 'get_global_cache', 'set_global_cache',
        'get_cache', 'clear_global_cache', 'quick_cache'
    ]
}

# 导出包信息
__all__.append("_PACKAGE_INFO")