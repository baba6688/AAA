#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X2磁盘缓存管理器模块

该模块提供完整的磁盘缓存管理功能，包括文件存储、缓存优化、空间管理等功能。
支持压缩、备份、统计、清理等高级特性。

主要类和功能：
- DiskCacheManager: 磁盘缓存管理器核心类，提供完整的缓存管理功能
- CacheConfig: 缓存配置类，用于设置缓存参数
- CacheEntry: 缓存条目类，表示单个缓存项的元数据
- CacheStats: 缓存统计类，用于收集和分析缓存性能数据

模块特性：
- 支持数据压缩以节省存储空间
- 自动清理过期和无效的缓存项
- 支持备份和恢复功能
- 完整的性能统计和监控
- 线程安全的操作
- 上下文管理器支持
- 自动垃圾回收机制

使用示例：
    from X2 import DiskCacheManager, CacheConfig
    
    # 使用默认配置创建缓存管理器
    cache_manager = DiskCacheManager("/tmp/cache")
    
    # 或使用自定义配置
    config = CacheConfig(
        max_cache_size=1024*1024*200,  # 200MB
        compression_enabled=True
    )
    cache_manager = DiskCacheManager("/tmp/cache", config)
    
    # 存储数据
    cache_manager.set("user_123", {"name": "张三", "age": 25})
    
    # 获取数据
    user_data = cache_manager.get("user_123")
    
    # 检查是否存在
    if cache_manager.exists("user_123"):
        print("用户数据存在")
    
    # 批量操作示例
    for i in range(100):
        cache_manager.set(f"key_{i}", f"value_{i}")
    
    # 获取统计信息
    stats = cache_manager.get_stats()
    print(f"缓存命中率: {stats.cache_hits / max(stats.cache_hits + stats.cache_misses, 1) * 100:.2f}%")
    
    # 获取缓存信息
    info = cache_manager.get_cache_info()
    print(f"缓存详情: {info}")
    
    # 清理缓存
    cache_manager.cleanup()
    
    # 使用上下文管理器
    with DiskCacheManager("/tmp/cache") as cache:
        cache.set("session", "data")
        value = cache.get("session")
    
    # 备份和恢复
    cache_manager.backup()
    cache_manager.restore("backup_20231114_120000")

模块版本信息：
- 版本: 1.0.0
- 作者: X2开发团队
- 最后更新: 2025-11-14

依赖要求：
- Python 3.6+
- 标准库模块: os, json, time, gzip, shutil, hashlib, threading, pickle, logging
- 第三方模块: 无

性能特性：
- 缓存命中率监控
- 内存使用优化
- 自动压缩和解压
- 异步后台清理任务
- 文件锁定机制
- 数据完整性校验
"""

# 从DiskCacheManager模块导入所有核心类
from .DiskCacheManager import (
    DiskCacheManager,
    CacheConfig,
    CacheEntry,
    CacheStats
)

# 模块元数据
__version__ = "1.0.0"
__author__ = "X2开发团队"
__email__ = "dev@x2.example.com"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025 X2 Team"
__status__ = "Production"

# 模块初始化信息
def _init_module_info():
    """初始化模块信息并打印版本信息"""
    import sys
    if hasattr(sys, 'version_info') and sys.version_info >= (3, 6):
        # 只有在直接运行时才打印版本信息
        import __main__
        if not hasattr(__main__, '__file__'):
            # 交互式模式，延迟打印
            return
    
    # 打印模块版本信息
    print(f"X2磁盘缓存管理器 v{__version__} - {__copyright__}")

# 模块加载时的初始化
_init_module_info()

# 定义模块级别的公共接口
# 控制 from X2 import * 时的导入行为
__all__ = [
    # 核心缓存管理器
    "DiskCacheManager",
    
    # 配置和元数据类
    "CacheConfig",
    "CacheEntry", 
    "CacheStats",
    
    # 模块版本信息
    "__version__",
    "__author__",
    "__license__"
]

# 模块级别的便利函数（可选）
def create_cache_manager(cache_dir: str, **kwargs) -> DiskCacheManager:
    """
    便捷函数：创建并配置缓存管理器
    
    Args:
        cache_dir: 缓存目录路径
        **kwargs: 传递给CacheConfig的其他参数
    
    Returns:
        DiskCacheManager: 配置好的缓存管理器实例
    
    Example:
        cache = create_cache_manager("/tmp/cache", max_cache_size=200*1024*1024)
    """
    config_kwargs = {}
    
    # 提取CacheConfig相关参数
    config_params = {
        'max_cache_size', 'max_file_size', 'cleanup_interval', 
        'compression_enabled', 'backup_enabled', 'index_file',
        'backup_dir', 'temp_dir', 'gc_threshold'
    }
    
    for key, value in kwargs.items():
        if key in config_params:
            config_kwargs[key] = value
    
    # 创建配置对象
    config = CacheConfig(**config_kwargs)
    
    # 返回缓存管理器实例
    return DiskCacheManager(cache_dir, config)

# 扩展__all__以包含便利函数
__all__.append("create_cache_manager")

# 模块卸载时的清理操作
def _cleanup_module():
    """模块卸载时的清理操作"""
    # 这里可以添加模块级别的清理代码
    pass

# 注册清理函数（如果需要在模块卸载时执行）
import atexit
atexit.register(_cleanup_module)

# 导入完成后的模块信息提示
try:
    from .DiskCacheManager import DiskCacheManager, CacheConfig, CacheEntry, CacheStats
    # 验证导入是否成功
    assert issubclass(DiskCacheManager, object)
    assert issubclass(CacheConfig, object)
    assert issubclass(CacheEntry, object) 
    assert issubclass(CacheStats, object)
    
    # 导入成功的标记（用于测试）
    _import_success = True
    
except ImportError as e:
    _import_success = False
    print(f"警告: X2模块导入失败 - {e}")