"""
X6缓存清理器包

提供完整的缓存清理解决方案，包括：
- 自动清理（定时自动清理过期缓存）
- 手动清理（手动触发缓存清理）
- 清理策略（LRU、LFU、TTL等清理策略）
- 清理统计（清理效果统计和分析）
- 清理监控（清理进度和状态监控）
- 清理配置（清理参数配置）
- 清理报告（清理结果报告）
- 清理优化（清理性能优化）

主要类：
- CacheCleaner: 主要的缓存清理器类
- CleanStrategy: 清理策略枚举
- CacheEntry: 缓存条目类
- CleanStats: 清理统计类

使用示例：
    from X.X6 import CacheCleaner, CleanStrategy
    
    # 创建缓存清理器
    cleaner = CacheCleaner(
        cache_dir="/tmp/x6_cache",
        auto_clean_interval=3600,
        max_cache_size=1024*1024*1024
    )
    
    # 添加缓存
    cleaner.add_cache_entry("key", "value", ttl=3600)
    
    # 获取缓存
    value = cleaner.get_cache_entry("key")
    
    # 手动清理
    result = cleaner.manual_clean(CleanStrategy.LRU, target_count=100)
    
    # 获取统计
    stats = cleaner.get_clean_stats()
    
    # 关闭清理器
    cleaner.shutdown()

    # 或者直接导入所有需要的类
    from X.X6 import CacheCleaner, CleanStrategy, CacheEntry, CleanStats

版本: 1.0.0
作者: X6开发团队
日期: 2025-11-06
"""

from .CacheCleaner import (
    CacheCleaner,
    CleanStrategy,
    CacheEntry,
    CleanStats
)

__version__ = "1.0.0"
__author__ = "X6开发团队"
__email__ = "x6-dev@example.com"
__license__ = "MIT"

__all__ = [
    "CacheCleaner",
    "CleanStrategy", 
    "CacheEntry",
    "CleanStats"
]