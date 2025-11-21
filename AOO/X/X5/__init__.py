"""
X5缓存预热器模块

这是一个高效的缓存预热系统，提供智能的数据预热、策略管理、
监控统计等功能，帮助提升系统性能和用户体验。

主要功能：
- 数据预热：系统启动时预加载关键数据
- 预热策略：智能预热算法和策略管理
- 预热管理：预热任务管理和调度
- 预热监控：实时监控预热进度和状态
- 预热统计：预热效果统计和分析
- 预热优化：性能优化和资源管理
- 预热配置：灵活的配置参数管理
- 预热报告：详细的预热结果报告

快速开始：
    from X import X5
    from X5 import CacheWarmer, create_cache_warmer
    
    # 使用主类
    warmer = CacheWarmer()
    task_id = await warmer.create_preheating_task("数据预热", "data_source", ["key1", "key2"])
    await warmer.start()
    
    # 使用便捷函数
    warmer = await create_cache_warmer()

作者：X5开发团队
版本：1.0.0
日期：2025-11-06
"""

from .CacheWarmer import (
    CacheWarmer,
    PreheatingStrategy,
    PreheatingManager,
    PreheatingMonitor,
    PreheatingStatistics,
    PreheatingOptimizer,
    PreheatingConfig,
    PreheatingReporter,
    DataPreloader,
    PreheatingTask,
    PreheatingStatus,
    create_cache_warmer,
    quick_preload
)

__version__ = "1.0.0"
__author__ = "X5开发团队"

# 模块初始化信息
__all__ = [
    "CacheWarmer",
    "PreheatingStrategy",
    "PreheatingManager", 
    "PreheatingMonitor",
    "PreheatingStatistics",
    "PreheatingOptimizer",
    "PreheatingConfig",
    "PreheatingReporter",
    "DataPreloader",
    "PreheatingTask",
    "PreheatingStatus",
    "create_cache_warmer",
    "quick_preload"
]

# 模块级常量
DEFAULT_CONFIG_FILE = None
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_CONCURRENT_TASKS = 5
DEFAULT_PREHEATING_TIMEOUT = 300

def get_module_info():
    """获取模块信息"""
    return {
        "name": "X5缓存预热器",
        "version": __version__,
        "author": __author__,
        "description": "高效的缓存预热系统",
        "classes": len(__all__) - 2,  # 减去2个函数
        "functions": 2,
        "total_exports": len(__all__)
    }

# 模块加载完成提示
def _init_module():
    """模块初始化"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"X5缓存预热器模块 v{__version__} 已加载")
    logger.info(f"可用导出: {len(__all__)} 个类和函数")

# 自动执行模块初始化
_init_module()