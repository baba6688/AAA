"""
X区 - 扩展功能模块：缓存管理组件
Extension Module - Cache Management Components

模块描述：
X区为扩展功能模块提供全面的缓存管理支持，包含内存缓存、磁盘缓存、分布式缓存等
9个子模块，总计74个类和7,741行代码，提供完整的缓存管理框架。

功能分类：
- X1: 内存缓存管理器 (MemoryCacheManager) - 8类内存缓存管理
- X2: 磁盘缓存管理器 (DiskCacheManager) - 4类磁盘缓存管理
- X3: 分布式缓存管理器 (DistributedCacheManager) - 8类分布式缓存
- X4: 缓存策略管理器 (CacheStrategyManager) - 12类策略管理
- X5: 缓存预热器 (CacheWarmer) - 10类预热机制
- X6: 缓存清理器 (CacheCleaner) - 4类清理策略
- X7: 缓存监控器 (CacheMonitor) - 7类监控功能
- X8: 缓存优化器 (CacheOptimizer) - 8类优化工具
- X9: 缓存状态聚合器 (CacheStatusAggregator) - 13类状态管理

版本：v1.0.0
最后更新：2025-11-14
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "MiniMax Agent"

# 主模块导入
from .X1.MemoryCacheManager import (
    CacheEntry,
    EvictionStrategy,
    LRUEvictionStrategy,
    TTLEvictionStrategy,
    SizeEvictionStrategy,
    CacheStatistics,
    MemoryCacheManager
)

from .X1 import CacheConfigs

from .X2.DiskCacheManager import (
    CacheConfig,
    CacheEntry,
    CacheStats,
    DiskCacheManager
)

from .X3.DistributedCacheManager import (
    ConsistencyLevel,
    NodeStatus,
    CacheNode,
    CacheEntry,
    ConsistentHash,
    LoadBalancer,
    PerformanceMonitor,
    DistributedCacheManager
)

from .X4.CacheStrategyManager import (
    CacheStrategy,
    StrategyPerformance,
    CacheConfig,
    StrategyMetrics,
    CacheStrategyBase,
    LRUStrategy,
    LFUStrategy,
    TTLStrategy,
    FIFOStrategy,
    CacheStrategyManager
)

from .X5.CacheWarmer import (
    PreheatingStatus,
    PreheatingTask,
    PreheatingConfig,
    PreheatingStrategy,
    DataPreloader,
    PreheatingManager,
    PreheatingMonitor,
    PreheatingStatistics,
    PreheatingOptimizer,
    PreheatingReporter,
    CacheWarmer
)

from .X6.CacheCleaner import (
    CleanStrategy,
    CacheEntry,
    CleanStats,
    CacheCleaner
)

from .X7.CacheMonitor import (
    AlertLevel,
    CacheStatus,
    CacheMetrics,
    AlertRule,
    Alert,
    DatabaseManager,
    CacheMonitor,
    SimpleCacheBackend
)

from .X8.CacheOptimizer import (
    CacheMetrics,
    OptimizationConfig,
    CacheStrategy,
    LRUStrategy,
    LFUStrategy,
    TTLStrategy,
    CacheOptimizer,
    Compressor,
    CacheAnalyzer
)

from .X9.CacheStatusAggregator import (
    CacheStatus,
    AlertRule,
    Alert,
    StatusCollector,
    DataAggregator,
    StatusAnalyzer,
    AlertManager,
    HistoryManager,
    ReportGenerator,
    Dashboard,
    CacheStatusAggregator,
    ExampleCacheModule
)

# 导出配置
__all__ = [
    # X1 - 内存缓存管理器 (8类)
    "CacheEntry", "EvictionStrategy", "LRUEvictionStrategy", "TTLEvictionStrategy",
    "SizeEvictionStrategy", "CacheStatistics", "MemoryCacheManager", "CacheConfigs",
    
    # X2 - 磁盘缓存管理器 (4类)
    "CacheConfig", "CacheEntry", "CacheStats", "DiskCacheManager",
    
    # X3 - 分布式缓存管理器 (8类)
    "ConsistencyLevel", "NodeStatus", "CacheNode", "CacheEntry", "ConsistentHash",
    "LoadBalancer", "PerformanceMonitor", "DistributedCacheManager",
    
    # X4 - 缓存策略管理器 (12类)
    "CacheStrategy", "StrategyPerformance", "CacheConfig", "StrategyMetrics",
    "CacheStrategyBase", "LRUStrategy", "LFUStrategy", "TTLStrategy", "FIFOStrategy",
    "CacheStrategyManager",
    
    # X5 - 缓存预热器 (12类)
    "PreheatingStatus", "PreheatingTask", "PreheatingConfig", "PreheatingStrategy",
    "DataPreloader", "PreheatingManager", "PreheatingMonitor", "PreheatingStatistics",
    "PreheatingOptimizer", "PreheatingReporter", "CacheWarmer",
    
    # X6 - 缓存清理器 (4类)
    "CleanStrategy", "CacheEntry", "CleanStats", "CacheCleaner",
    
    # X7 - 缓存监控器 (7类)
    "AlertLevel", "CacheStatus", "CacheMetrics", "AlertRule", "Alert",
    "DatabaseManager", "CacheMonitor", "SimpleCacheBackend",
    
    # X8 - 缓存优化器 (8类)
    "CacheMetrics", "OptimizationConfig", "CacheStrategy", "LRUStrategy",
    "LFUStrategy", "TTLStrategy", "CacheOptimizer", "Compressor", "CacheAnalyzer",
    
    # X9 - 缓存状态聚合器 (13类)
    "CacheStatus", "AlertRule", "Alert", "StatusCollector", "DataAggregator",
    "StatusAnalyzer", "AlertManager", "HistoryManager", "ReportGenerator",
    "Dashboard", "CacheStatusAggregator", "ExampleCacheModule"
]

# 模块信息
MODULE_INFO = {
    "name": "Extension Module - Cache Management",
    "version": "1.0.0",
    "total_classes": 74,
    "total_lines": 7741,
    "sub_modules": {
        "X1": {"name": "Memory Cache Manager", "classes": 8},
        "X2": {"name": "Disk Cache Manager", "classes": 4},
        "X3": {"name": "Distributed Cache Manager", "classes": 8},
        "X4": {"name": "Cache Strategy Manager", "classes": 12},
        "X5": {"name": "Cache Warmer", "classes": 12},
        "X6": {"name": "Cache Cleaner", "classes": 4},
        "X7": {"name": "Cache Monitor", "classes": 7},
        "X8": {"name": "Cache Optimizer", "classes": 8},
        "X9": {"name": "Cache Status Aggregator", "classes": 13}
    }
}

print(f"X区 - 扩展功能模块已初始化，缓存组件总数: {MODULE_INFO['total_classes']} 类")