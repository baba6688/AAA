#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X8缓存优化器 - 高性能缓存优化解决方案

该模块提供了全面的缓存优化功能，包括性能优化、内存优化、
策略优化、分析监控等多个方面。

主要功能:
- 多种缓存策略 (LRU, LFU, TTL, FIFO)
- 智能数据压缩
- 性能监控与分析
- 自动优化与清理
- 内存管理
- 缓存数据导入导出
- 性能报告与优化建议

Author: X8 Team
Date: 2025-11-14
Version: 1.0.0

使用示例:
    from X8 import CacheOptimizer, OptimizationConfig, create_cache_optimizer
    
    # 使用默认配置
    optimizer = CacheOptimizer()
    
    # 自定义配置
    config = OptimizationConfig(
        max_memory_mb=100,
        compression_enabled=True,
        strategy="lru"
    )
    optimizer = CacheOptimizer(config)
    
    # 使用便捷函数
    optimizer = create_cache_optimizer()
    
    # 设置和获取缓存
    optimizer.set("key", "value")
    value = optimizer.get("key")
    
    # 获取性能报告
    report = optimizer.get_performance_report()
"""

from .CacheOptimizer import (
    CacheOptimizer,
    OptimizationConfig,
    CacheMetrics,
    CacheStrategy,
    LRUStrategy,
    LFUStrategy,
    TTLStrategy,
    Compressor,
    CacheAnalyzer,
    create_cache_optimizer,
    get_default_config
)

__version__ = "1.0.0"
__author__ = "X8 Team"
__email__ = "x8-team@example.com"
__license__ = "MIT"

# 定义完整的导出列表
__all__ = [
    # 数据类
    "CacheMetrics",
    "OptimizationConfig",
    
    # 策略类
    "CacheStrategy",
    "LRUStrategy", 
    "LFUStrategy",
    "TTLStrategy",
    
    # 核心类
    "CacheOptimizer",
    "Compressor",
    "CacheAnalyzer",
    
    # 便捷函数
    "create_cache_optimizer",
    "get_default_config",
    "quick_start",
    "get_performance_monitor"
]

# 包级别的便捷函数
def quick_start(max_memory_mb=100, strategy="lru"):
    """
    快速创建缓存优化器实例
    
    Args:
        max_memory_mb: 最大内存使用量(MB)
        strategy: 缓存策略 ("lru", "lfu", "ttl", "fifo")
    
    Returns:
        CacheOptimizer: 缓存优化器实例
    """
    config = OptimizationConfig(
        max_memory_mb=max_memory_mb,
        strategy=strategy,
        enable_monitoring=True
    )
    return CacheOptimizer(config)


def get_performance_monitor(optimizer):
    """
    获取性能监控器
    
    Args:
        optimizer: CacheOptimizer实例
    
    Returns:
        CacheAnalyzer: 缓存分析器实例
    """
    return CacheAnalyzer(optimizer)


# 模块级别的便捷访问
default_config = get_default_config
create_optimizer = create_cache_optimizer

# 版本信息
VERSION_INFO = {
    "version": __version__,
    "author": __author__,
    "description": "X8缓存优化器 - 高性能缓存优化解决方案",
    "features": [
        "缓存性能优化",
        "内存优化与压缩",
        "多种缓存策略",
        "智能分析监控",
        "优化建议系统",
        "详细性能报告"
    ],
    "exported_classes": [cls for cls in __all__ if cls not in ["create_cache_optimizer", "get_default_config", "quick_start", "get_performance_monitor"]],
    "convenience_functions": ["create_cache_optimizer", "get_default_config", "quick_start", "get_performance_monitor"]
}

def get_version():
    """获取版本信息"""
    return __version__

def get_module_info():
    """获取模块信息"""
    return {
        "name": "X8缓存优化器",
        "version": __version__,
        "author": __author__,
        "description": "高性能缓存优化解决方案",
        "classes": VERSION_INFO["exported_classes"],
        "functions": VERSION_INFO["convenience_functions"]
    }

# 模块初始化信息
import logging

# 设置模块日志
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 添加模块信息到日志
logger.info(f"X8缓存优化器 v{__version__} 初始化完成")
logger.info(f"可用的类: {', '.join(VERSION_INFO['exported_classes'])}")
logger.info(f"便捷函数: {', '.join(VERSION_INFO['convenience_functions'])}")

# 提供向后兼容的属性访问
def __getattr__(name):
    """提供向后兼容的属性访问"""
    if name == "default_config":
        return get_default_config
    elif name == "create_optimizer":
        return create_cache_optimizer
    raise AttributeError(f"模块 '{__name__}' 没有属性 '{name}'")

# 使用指南
_USAGE_GUIDE = """
X8缓存优化器使用指南:

1. 基础使用:
   from X8 import CacheOptimizer, OptimizationConfig
   
   # 创建缓存优化器
   optimizer = CacheOptimizer()
   
   # 设置和获取缓存
   optimizer.set("key", "value")
   value = optimizer.get("key")

2. 自定义配置:
   config = OptimizationConfig(
       max_memory_mb=200,
       compression_enabled=True,
       strategy="lru",
       ttl_default=1800
   )
   optimizer = CacheOptimizer(config)

3. 性能监控:
   # 获取指标
   metrics = optimizer.get_metrics()
   
   # 获取性能报告
   report = optimizer.get_performance_report()
   
   # 获取优化建议
   suggestions = optimizer.get_optimization_suggestions()

4. 数据分析:
   from X8 import CacheAnalyzer
   
   analyzer = CacheAnalyzer(optimizer)
   patterns = analyzer.analyze_cache_patterns()

5. 便捷函数:
   from X8 import create_cache_optimizer, get_default_config, quick_start
   
   # 快速创建
   optimizer = create_cache_optimizer()
   
   # 获取默认配置
   config = get_default_config()
   
   # 一键启动
   optimizer = quick_start(max_memory_mb=100, strategy="lru")
"""

def print_usage_guide():
    """打印使用指南"""
    print(_USAGE_GUIDE)

# 将新增函数添加到导出列表
__all__.extend(["get_version", "get_module_info", "print_usage_guide", "VERSION_INFO"])