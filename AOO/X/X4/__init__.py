"""
X4缓存策略管理器
================

一个功能完整、高性能、可扩展的缓存策略管理系统。

支持多种缓存策略：
- LRU (最近最少使用)
- LFU (最少使用频率)  
- FIFO (先进先出)
- TTL (基于时间过期)

主要功能：
- 策略动态切换
- 性能监控与分析
- 自动优化
- 策略测试与评估
- 配置管理

Author: X4 Cache Strategy Manager
Date: 2025-11-06
Version: 1.0.0
"""

from .CacheStrategyManager import (
    # 核心类
    CacheStrategyManager,
    CacheStrategyBase,
    
    # 策略实现
    LRUStrategy,
    LFUStrategy,
    TTLStrategy,
    FIFOStrategy,
    
    # 配置和指标
    CacheConfig,
    StrategyMetrics,
    StrategyPerformance,
    
    # 策略枚举
    CacheStrategy,
    
    # 工厂函数
    create_cache_manager,
    
    # 装饰器
    cached,
)

__version__ = "1.0.0"
__author__ = "X4 Cache Strategy Manager"
__email__ = "support@x4-cache.com"

# 公开的API
__all__ = [
    # 核心类
    "CacheStrategyManager",
    "CacheStrategyBase",
    
    # 策略实现
    "LRUStrategy",
    "LFUStrategy", 
    "TTLStrategy",
    "FIFOStrategy",
    
    # 配置和指标
    "CacheConfig",
    "StrategyMetrics",
    "StrategyPerformance",
    
    # 枚举
    "CacheStrategy",
    
    # 工具函数
    "create_cache_manager",
    "cached",
]

# 版本信息
VERSION_INFO = {
    "major": 1,
    "minor": 0,
    "patch": 0,
    "release": "stable"
}

def get_version():
    """获取版本字符串"""
    return __version__

def get_version_info():
    """获取详细版本信息"""
    return VERSION_INFO.copy()

# 快速开始示例
def quick_start():
    """
    快速开始示例
    
    Returns:
        CacheStrategyManager: 配置好的缓存管理器实例
    """
    from .CacheStrategyManager import create_cache_manager
    
    # 创建默认配置的缓存管理器
    manager = create_cache_manager()
    
    # 基本使用
    manager.put("key", "value")
    value = manager.get("key")
    
    return manager

# 性能基准测试
def run_benchmark():
    """
    运行性能基准测试
    
    Returns:
        dict: 测试结果
    """
    from .CacheStrategyManager import CacheStrategyManager, CacheStrategy, CacheConfig
    import time
    
    results = {}
    strategies = [CacheStrategy.LRU, CacheStrategy.LFU, CacheStrategy.FIFO, CacheStrategy.TTL]
    
    for strategy in strategies:
        config = CacheConfig(max_size=1000)
        manager = CacheStrategyManager(config)
        manager.switch_strategy(strategy)
        
        # 插入测试
        start_time = time.time()
        for i in range(1000):
            manager.put(f"key_{i}", f"value_{i}")
        insert_time = time.time() - start_time
        
        # 读取测试
        start_time = time.time()
        hit_count = 0
        for i in range(1000):
            if manager.get(f"key_{i}") is not None:
                hit_count += 1
        read_time = time.time() - start_time
        
        metrics = manager.get_strategy_metrics()
        
        results[strategy.value] = {
            "insert_time": insert_time,
            "read_time": read_time,
            "hit_rate": metrics.hit_rate,
            "hits": metrics.hits,
            "misses": metrics.misses
        }
        
        manager.cleanup()
    
    return results

# 默认配置
DEFAULT_CONFIG = {
    "max_size": 1000,
    "ttl": 3600,
    "cleanup_interval": 300,
    "enable_monitoring": True,
    "enable_optimization": True
}

def get_default_config():
    """获取默认配置"""
    return DEFAULT_CONFIG.copy()

# 常用配置模板
CONFIG_TEMPLATES = {
    "web_cache": {
        "max_size": 2000,
        "ttl": 300,
        "cleanup_interval": 60,
        "enable_monitoring": True,
        "enable_optimization": True
    },
    "database_cache": {
        "max_size": 5000,
        "ttl": 1800,
        "cleanup_interval": 300,
        "enable_monitoring": True,
        "enable_optimization": True
    },
    "session_cache": {
        "max_size": 10000,
        "ttl": 3600,
        "cleanup_interval": 600,
        "enable_monitoring": True,
        "enable_optimization": False
    },
    "high_throughput": {
        "max_size": 100000,
        "ttl": 7200,
        "cleanup_interval": 1200,
        "enable_monitoring": False,
        "enable_optimization": False
    }
}

def get_config_template(template_name):
    """获取配置模板"""
    return CONFIG_TEMPLATES.get(template_name, DEFAULT_CONFIG).copy()

# 错误代码
ERROR_CODES = {
    "INVALID_STRATEGY": "001",
    "CONFIG_ERROR": "002", 
    "CACHE_FULL": "003",
    "TIMEOUT": "004",
    "MEMORY_ERROR": "005"
}

def get_error_message(error_code):
    """获取错误消息"""
    error_messages = {
        ERROR_CODES["INVALID_STRATEGY"]: "无效的缓存策略",
        ERROR_CODES["CONFIG_ERROR"]: "配置错误",
        ERROR_CODES["CACHE_FULL"]: "缓存已满",
        ERROR_CODES["TIMEOUT"]: "操作超时",
        ERROR_CODES["MEMORY_ERROR"]: "内存错误"
    }
    return error_messages.get(error_code, "未知错误")

# 导出所有公共接口
__all__.extend([
    "quick_start",
    "run_benchmark", 
    "get_version",
    "get_version_info",
    "get_default_config",
    "get_config_template",
    "get_error_message",
    "ERROR_CODES",
    "CONFIG_TEMPLATES",
    "DEFAULT_CONFIG",
    "VERSION_INFO"
])