#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X7缓存监控模块
==============

X7模块提供完整的缓存监控解决方案，包括：

功能特性：
- 实时缓存性能监控
- 缓存指标收集与分析
- 灵活的告警规则配置
- 告警管理通知
- 数据持久化存储
- 监控报告生成
- 性能优化建议

核心类说明：
- AlertLevel: 告警级别枚举 (INFO, WARNING, ERROR, CRITICAL)
- CacheStatus: 缓存状态枚举 (ACTIVE, INACTIVE, FULL, ERROR)  
- CacheMetrics: 缓存指标数据结构
- AlertRule: 告警规则配置
- Alert: 告警信息对象
- DatabaseManager: 数据库管理器
- CacheMonitor: 缓存监控主类
- SimpleCacheBackend: 示例缓存后端实现

使用示例：
    from X7 import CacheMonitor, AlertLevel, CacheMetrics
    
    # 创建缓存监控器
    monitor = CacheMonitor(
        cache_backend=my_cache,
        max_size_mb=100,
        monitor_interval=30
    )
    
    # 开始监控
    monitor.start_monitoring()
    
    # 获取当前指标
    metrics = monitor.get_current_metrics()
    print(f"缓存命中率: {metrics.hit_rate:.2%}")

版本信息：
- 版本: 1.0.0
- 作者: X7开发团队
- 更新: 2025-11-14
"""

# 版本信息
__version__ = "1.0.0"
__author__ = "X7开发团队"
__email__ = "dev@x7.com"
__description__ = "X7缓存监控器 - 专业的缓存监控解决方案"

# 从CacheMonitor模块导入所有核心类和函数
from .CacheMonitor import (
    # 核心数据类和枚举
    AlertLevel,
    CacheStatus, 
    CacheMetrics,
    AlertRule,
    Alert,
    
    # 管理器和主类
    DatabaseManager,
    CacheMonitor,
    SimpleCacheBackend,
    
    # 工具函数
    monitor_cache_operation
)

# 定义模块导出列表 - 用于 from X7 import * 的情况
__all__ = [
    # 核心数据类
    "AlertLevel",
    "CacheStatus", 
    "CacheMetrics",
    "AlertRule",
    "Alert",
    
    # 管理器类
    "DatabaseManager",
    "CacheMonitor",
    "SimpleCacheBackend",
    
    # 工具函数
    "monitor_cache_operation",
    
    # 模块信息
    "__version__",
    "__author__", 
    "__email__",
    "__description__"
]

# 模块初始化信息
def _init_module_info():
    """模块初始化信息"""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(f"X7缓存监控模块 v{__version__} 初始化完成")
    logger.info(f"可用的导出: {len(__all__)} 个项目")
    
# 自动执行模块初始化
_init_module_info()