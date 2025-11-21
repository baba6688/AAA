#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X3分布式缓存管理器模块

X3模块提供完整的分布式缓存解决方案，具有以下核心功能：

- 分布式存储：支持数据在多个节点间的分布式存储
- 数据分片：采用一致性哈希算法实现数据均匀分片
- 缓存同步：支持多节点数据同步和一致性保证
- 负载均衡：提供多种负载均衡策略
- 容错处理：自动检测和恢复节点故障
- 缓存一致性：支持多种一致性级别配置
- 集群管理：动态添加/移除节点，集群状态监控
- 性能监控：实时监控缓存性能和集群状态

主要组件：
- DistributedCacheManager：分布式缓存管理器主类
- ConsistencyLevel：一致性级别枚举
- NodeStatus：节点状态枚举
- CacheNode：缓存节点数据类
- CacheEntry：缓存条目数据类
- ConsistentHash：一致性哈希算法实现
- LoadBalancer：负载均衡器
- PerformanceMonitor：性能监控器

使用示例：
    >>> from X3 import DistributedCacheManager, ConsistencyLevel
    >>> 
    >>> # 创建缓存管理器
    >>> cache = DistributedCacheManager(
    ...     consistency_level=ConsistencyLevel.QUORUM,
    ...     replication_factor=2
    ... )
    >>> 
    >>> # 添加节点
    >>> cache.add_node("node1", "192.168.1.1", 6379)
    >>> cache.add_node("node2", "192.168.1.2", 6379)
    >>> 
    >>> # 使用缓存
    >>> cache.set("key1", "value1", ttl=3600)
    >>> value = cache.get("key1")
    >>> 
    >>> # 获取统计信息
    >>> stats = cache.get_stats()
    >>> print(f"命中率: {stats['hit_rate']:.2%}")

版本信息：
- 当前版本: 1.0.0
- Python要求: >= 3.6
- 作者: X3开发团队
- 许可证: MIT License
"""

# 主要组件导入
from .DistributedCacheManager import (
    # 核心管理器
    DistributedCacheManager,
    
    # 配置枚举
    ConsistencyLevel,
    NodeStatus,
    
    # 数据结构
    CacheNode,
    CacheEntry,
    
    # 算法组件
    ConsistentHash,
    LoadBalancer,
    PerformanceMonitor,
)

# 版本信息
__version__ = "1.0.0"
__author__ = "X3开发团队"
__license__ = "MIT License"
__email__ = "x3-team@example.com"

# 导出的公共接口
__all__ = [
    # 核心管理器
    "DistributedCacheManager",
    
    # 配置枚举
    "ConsistencyLevel",
    "NodeStatus",
    
    # 数据结构
    "CacheNode",
    "CacheEntry",
    
    # 算法组件
    "ConsistentHash",
    "LoadBalancer",
    "PerformanceMonitor",
    
    # 模块信息
    "__version__",
    "__author__",
    "__license__",
    
    # 便捷函数
    "create_cache_manager",
    "check_compatibility",
]
# 快速入门函数
def create_cache_manager(consistency_level='QUORUM', replication_factor=2, **kwargs):
    """
    快速创建分布式缓存管理器的便捷函数
    
    Args:
        consistency_level (str): 一致性级别 ('ONE', 'QUORUM', 'ALL')
        replication_factor (int): 复制因子，默认为2
        **kwargs: 其他分布式缓存管理器参数
        
    Returns:
        DistributedCacheManager: 配置好的缓存管理器实例
        
    使用示例:
        >>> cache = create_cache_manager('QUORUM', 3)
        >>> cache.add_node("node1", "localhost", 6379)
    """
    # 处理一致性级别参数
    if isinstance(consistency_level, str):
        try:
            consistency_level = ConsistencyLevel[consistency_level.upper()]
        except KeyError:
            consistency_level = ConsistencyLevel.QUORUM
    
    # 创建缓存管理器
    return DistributedCacheManager(
        consistency_level=consistency_level,
        replication_factor=replication_factor,
        **kwargs
    )

# 模块配置检查
def check_compatibility():
    """
    检查模块兼容性
    
    Returns:
        dict: 兼容性检查结果
    """
    import sys
    
    result = {
        'python_version': sys.version,
        'python_version_ok': sys.version_info >= (3, 6),
        'required_modules': {
            'hashlib': True,      # 内置模块
            'json': True,         # 内置模块
            'threading': True,    # 内置模块
            'time': True,         # 内置模块
            'logging': True,      # 内置模块
            'random': True,       # 内置模块
            'typing': True,       # 内置模块
            'collections': True,  # 内置模块
            'dataclasses': sys.version_info >= (3, 6),  # Python 3.6+
            'enum': True,         # 内置模块
        }
    }
    
    result['all_compatible'] = all(result['required_modules'].values())
    return result

# 模块初始化信息
def _init_module_info():
    """初始化模块信息"""
    import logging
    
    logger = logging.getLogger(__name__)
    logger.info(f"X3分布式缓存管理器模块已加载 (版本: {__version__})")
    logger.info(f"作者: {__author__}")
    logger.info(f"许可证: {__license__}")
    
    return {
        'module': __name__,
        'version': __version__,
        'author': __author__,
        'license': __license__,
        'components': 8,  # 8个核心组件
    }

# 模块初始化
try:
    _module_info = _init_module_info()
    
    # 导出模块信息到模块级别
    _module_info['all_exports'] = __all__
    
except Exception as e:
    # 如果初始化失败，记录错误但不阻止模块加载
    import logging
    logger = logging.getLogger(__name__)
    logger.error(f"X3模块初始化时出现错误: {e}")

# 模块文档补充
"""
核心类和功能详细说明：

1. DistributedCacheManager
   - 分布式缓存的核心管理器
   - 提供缓存的增删改查功能
   - 支持集群管理和监控

2. ConsistencyLevel (一致性级别)
   - ONE: 单节点确认，写入速度快但一致性弱
   - QUORUM: 大多数节点确认，平衡一致性和性能
   - ALL: 所有节点确认，一致性最强但性能较低

3. NodeStatus (节点状态)
   - HEALTHY: 节点健康正常
   - UNHEALTHY: 节点不健康
   - SUSPECTED: 节点疑似故障
   - DOWN: 节点确认故障

4. CacheNode (缓存节点)
   - 存储节点的基本信息
   - 包含节点ID、地址、端口、状态等
   - 维护节点的实时状态和统计信息

5. CacheEntry (缓存条目)
   - 表示单个缓存数据项
   - 包含键值、创建时间、过期时间等
   - 支持版本控制和过期检查

6. ConsistentHash (一致性哈希)
   - 实现数据分片算法
   - 确保数据在节点间均匀分布
   - 支持节点的动态增删

7. LoadBalancer (负载均衡器)
   - 提供多种负载均衡策略
   - 包括轮询、最少连接、加权轮询、随机等
   - 智能选择最优节点

8. PerformanceMonitor (性能监控器)
   - 实时监控缓存性能指标
   - 记录操作延迟、错误率等统计信息
   - 支持单节点和集群级别的性能分析

性能优化建议：
- 根据业务需求选择合适的一致性级别
- 合理设置复制因子，平衡可用性和性能
- 定期监控集群状态和性能指标
- 根据负载情况调整负载均衡策略
"""