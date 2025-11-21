"""
O2内存缓存模块

提供高效的内存缓存实现，包括LRU缓存、弱引用缓存等多种缓存策略

主要功能：
1. LRU缓存（最近最少使用）
2. 弱引用缓存
3. TTL缓存（超时缓存）
4. 多级缓存
5. 缓存预热和预取
6. 缓存统计和监控
7. 线程安全支持
8. 异步缓存操作

作者: O2优化团队
版本: 2.0.0
日期: 2025-11-06
"""

import asyncio
import threading
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps, lru_cache
from threading import RLock
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Union, Callable, 
    Generic, TypeVar, Iterator, Protocol, Awaitable, Hashable
)
import logging
import json
import pickle
import hashlib
from concurrent.futures import ThreadPoolExecutor


logger = logging.getLogger(__name__)


K = TypeVar('K')
V = TypeVar('V')
T = TypeVar('T')


@dataclass
class CacheConfig:
    """缓存配置"""
    max_size: int = 1000
    ttl: Optional[float] = None  # 生存时间（秒）
    cleanup_interval: float = 60.0  # 清理间隔（秒）
    thread_safe: bool = True
    weak_reference: bool = False
    enable_stats: bool = True
    preload_keys: Optional[List[K]] = None
    eviction_policy: str = "lru"  # "lru", "lfu", "fifo", "adaptive"
    compression: bool = False
    serializer: str = "pickle"  # "pickle", "json", "none"


@dataclass
class CacheStats:
    """缓存统计信息"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    inserts: int = 0
    updates: int = 0
    deletes: int = 0
    current_size: int = 0
    hit_rate: float = 0.0
    avg_access_time: float = 0.0
    memory_usage: int = 0
    start_time: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)


class CacheEntry:
    """缓存条目"""
    
    def __init__(self, key: K, value: V, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.expires_at = self.created_at + ttl if ttl else None
        self.size = self._calculate_size()
    
    def _calculate_size(self) -> int:
        """计算条目大小"""
        try:
            return len(pickle.dumps((self.key, self.value)))
        except Exception:
            return 64  # 默认大小估算
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at
    
    def access(self) -> V:
        """访问值并更新统计"""
        self.last_accessed = time.time()
        self.access_count += 1
        return self.value
    
    def update_access_time(self) -> None:
        """更新访问时间"""
        self.last_accessed = time.time()


class CacheEvictionPolicy(ABC, Generic[K]):
    """缓存淘汰策略抽象基类"""
    
    @abstractmethod
    def on_access(self, key: K, entry: CacheEntry) -> None:
        """访问时调用"""
        pass
    
    @abstractmethod
    def on_insert(self, key: K, entry: CacheEntry) -> None:
        """插入时调用"""
        pass
    
    @abstractmethod
    def on_evict(self, key: K) -> None:
        """淘汰时调用"""
        pass
    
    @abstractmethod
    def get_eviction_order(self) -> List[K]:
        """获取淘汰顺序"""
        pass


class LRUEvictionPolicy(CacheEvictionPolicy[K]):
    """LRU淘汰策略"""
    
    def __init__(self):
        self.access_order = OrderedDict()
    
    def on_access(self, key: K, entry: CacheEntry) -> None:
        """访问时更新顺序"""
        if key in self.access_order:
            self.access_order.move_to_end(key)
        else:
            self.access_order[key] = entry
    
    def on_insert(self, key: K, entry: CacheEntry) -> None:
        """插入时添加到末尾"""
        self.access_order[key] = entry
    
    def on_evict(self, key: K) -> None:
        """淘汰时移除"""
        self.access_order.pop(key, None)
    
    def get_eviction_order(self) -> List[K]:
        """获取淘汰顺序（最少使用的在前）"""
        return list(self.access_order.keys())


class LFUEvictionPolicy(CacheEvictionPolicy[K]):
    """LFU淘汰策略"""
    
    def __init__(self):
        self.frequency = defaultdict(int)
        self.frequency_groups = defaultdict(set)
        self.min_frequency = 0
    
    def on_access(self, key: K, entry: CacheEntry) -> None:
        """访问时增加频率"""
        old_freq = self.frequency[key]
        self.frequency_groups[old_freq].discard(key)
        
        if not self.frequency_groups[old_freq] and old_freq == self.min_frequency:
            self.min_frequency += 1
        
        new_freq = old_freq + 1
        self.frequency[key] = new_freq
        self.frequency_groups[new_freq].add(key)
    
    def on_insert(self, key: K, entry: CacheEntry) -> None:
        """插入时设置初始频率"""
        self.frequency[key] = 0
        self.frequency_groups[0].add(key)
        self.min_frequency = 0
    
    def on_evict(self, key: K) -> None:
        """淘汰时清理"""
        freq = self.frequency[key]
        self.frequency_groups[freq].discard(key)
        del self.frequency[key]
        
        if not self.frequency_groups[freq] and freq == self.min_frequency:
            self.min_frequency += 1
    
    def get_eviction_order(self) -> List[K]:
        """获取淘汰顺序（频率最低的在前）"""
        candidates = list(self.frequency_groups[self.min_frequency])
        return sorted(candidates, key=lambda k: self.frequency[k])


class FIFOEvictionPolicy(CacheEvictionPolicy[K]):
    """FIFO淘汰策略"""
    
    def __init__(self):
        self.insertion_order = []
        self.position = defaultdict(int)
        self.counter = 0
    
    def on_access(self, key: K, entry: CacheEntry) -> None:
        """访问时不改变顺序"""
        pass
    
    def on_insert(self, key: K, entry: CacheEntry) -> None:
        """插入时记录顺序"""
        self.position[key] = self.counter
        self.insertion_order.append(key)
        self.counter += 1
    
    def on_evict(self, key: K) -> None:
        """淘汰时清理位置记录"""
        self.position.pop(key, None)
    
    def get_eviction_order(self) -> List[K]:
        """获取淘汰顺序（先进先出）"""
        return self.insertion_order.copy()


class AdaptiveEvictionPolicy(CacheEvictionPolicy[K]):
    """自适应淘汰策略"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.access_history = []
        self.hit_miss_ratio = 1.0
        self.current_policy = "lru"
        self.lru_policy = LRUEvictionPolicy()
        self.lfu_policy = LFUEvictionPolicy()
    
    def _update_hit_miss_ratio(self, hit: bool) -> None:
        """更新命中率"""
        self.access_history.append(hit)
        if len(self.access_history) > self.window_size:
            self.access_history.pop(0)
        
        if self.access_history:
            hits = sum(self.access_history)
            self.hit_miss_ratio = hits / len(self.access_history)
        
        # 根据命中率选择策略
        if self.hit_miss_ratio > 0.8:
            self.current_policy = "lfu"
        else:
            self.current_policy = "lru"
    
    def on_access(self, key: K, entry: CacheEntry) -> None:
        """访问时更新策略"""
        self._update_hit_miss_ratio(True)
        
        if self.current_policy == "lru":
            self.lru_policy.on_access(key, entry)
        else:
            self.lfu_policy.on_access(key, entry)
    
    def on_insert(self, key: K, entry: CacheEntry) -> None:
        """插入时调用"""
        self.lru_policy.on_insert(key, entry)
        self.lfu_policy.on_insert(key, entry)
    
    def on_evict(self, key: K) -> None:
        """淘汰时调用"""
        self.lru_policy.on_evict(key)
        self.lfu_policy.on_evict(key)
    
    def get_eviction_order(self) -> List[K]:
        """获取淘汰顺序"""
        if self.current_policy == "lru":
            return self.lru_policy.get_eviction_order()
        else:
            return self.lfu_policy.get_eviction_order()


class BaseCache(ABC, Generic[K, V]):
    """缓存基类"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.stats = CacheStats()
        
        # 同步控制
        if config.thread_safe:
            self._lock = RLock()
        else:
            self._lock = None
        
        # 缓存存储
        self._cache: Dict[K, CacheEntry] = {}
        
        # 淘汰策略
        self._eviction_policy = self._create_eviction_policy()
        
        # 统计
        if config.enable_stats:
            self._access_times = []
        
        logger.info(f"缓存已初始化: {config.max_size} 项")
    
    def _create_eviction_policy(self) -> CacheEvictionPolicy[K]:
        """创建淘汰策略"""
        policy_map = {
            "lru": LRUEvictionPolicy,
            "lfu": LFUEvictionPolicy,
            "fifo": FIFOEvictionPolicy,
            "adaptive": AdaptiveEvictionPolicy
        }
        
        policy_class = policy_map.get(self.config.eviction_policy, LRUEvictionPolicy)
        return policy_class()
    
    def _get_lock(self):
        """获取锁"""
        return self._lock if self._lock else _DummyLock()
    
    def _update_stats(self, operation: str, success: bool = True) -> None:
        """更新统计信息"""
        if not self.config.enable_stats:
            return
        
        with self._get_lock():
            if operation == "hit":
                self.stats.hits += 1
            elif operation == "miss":
                self.stats.misses += 1
            elif operation == "eviction":
                self.stats.evictions += 1
            elif operation == "expiration":
                self.stats.expirations += 1
            elif operation == "insert":
                self.stats.inserts += 1
            elif operation == "update":
                self.stats.updates += 1
            elif operation == "delete":
                self.stats.deletes += 1
            
            self.stats.current_size = len(self._cache)
            self.stats.last_access = time.time()
            
            # 计算命中率
            total_requests = self.stats.hits + self.stats.misses
            if total_requests > 0:
                self.stats.hit_rate = self.stats.hits / total_requests
            
            # 计算平均访问时间
            if self._access_times:
                self.stats.avg_access_time = sum(self._access_times) / len(self._access_times)
    
    def _should_evict(self) -> bool:
        """检查是否需要淘汰"""
        return len(self._cache) >= self.config.max_size
    
    def _evict_one(self) -> Optional[K]:
        """淘汰一个条目"""
        if not self._cache:
            return None
        
        eviction_order = self._eviction_policy.get_eviction_order()
        
        for key in eviction_order:
            if key in self._cache:
                self._evict(key)
                return key
        
        return None
    
    def _evict(self, key: K) -> None:
        """淘汰指定键"""
        if key in self._cache:
            entry = self._cache.pop(key)
            self._eviction_policy.on_evict(key)
            self._update_stats("eviction")
            
            # 如果是弱引用缓存，清理引用
            if self.config.weak_reference:
                weakref.finalize(entry.value, lambda: None)
    
    def _cleanup_expired(self) -> List[K]:
        """清理过期条目"""
        expired_keys = []
        current_time = time.time()
        
        for key, entry in list(self._cache.items()):
            if entry.is_expired():
                expired_keys.append(key)
                self._evict(key)
        
        if expired_keys:
            self._update_stats("expiration")
        
        return expired_keys
    
    @abstractmethod
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """获取缓存值"""
        pass
    
    @abstractmethod
    def set(self, key: K, value: V, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        pass
    
    @abstractmethod
    def delete(self, key: K) -> bool:
        """删除缓存值"""
        pass
    
    def get_or_set(self, key: K, factory: Callable[[], V], ttl: Optional[float] = None) -> V:
        """
        获取或设置缓存值
        
        Args:
            key: 缓存键
            factory: 值工厂函数
            ttl: 生存时间
            
        Returns:
            V: 缓存值
        """
        value = self.get(key)
        if value is not None:
            return value
        
        # 创建新值
        value = factory()
        self.set(key, value, ttl)
        return value
    
    def has(self, key: K) -> bool:
        """检查键是否存在"""
        return self.get(key) is not None
    
    def clear(self) -> None:
        """清空缓存"""
        with self._get_lock():
            self._cache.clear()
            self._eviction_policy = self._create_eviction_policy()
            
            if self.config.enable_stats:
                self.stats.current_size = 0
            
            logger.info("缓存已清空")
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self._cache)
    
    def is_full(self) -> bool:
        """检查缓存是否已满"""
        return len(self._cache) >= self.config.max_size
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._get_lock():
            return {
                'config': {
                    'max_size': self.config.max_size,
                    'ttl': self.config.ttl,
                    'eviction_policy': self.config.eviction_policy,
                    'weak_reference': self.config.weak_reference,
                    'thread_safe': self.config.thread_safe
                },
                'stats': {
                    'hits': self.stats.hits,
                    'misses': self.stats.misses,
                    'evictions': self.stats.evictions,
                    'expirations': self.stats.expirations,
                    'inserts': self.stats.inserts,
                    'updates': self.stats.updates,
                    'deletes': self.stats.deletes,
                    'current_size': self.stats.current_size,
                    'hit_rate': self.stats.hit_rate,
                    'avg_access_time': self.stats.avg_access_time,
                    'uptime': time.time() - self.stats.start_time
                },
                'timestamp': time.time()
            }
    
    def keys(self) -> List[K]:
        """获取所有键"""
        return list(self._cache.keys())
    
    def values(self) -> List[V]:
        """获取所有值"""
        return [entry.value for entry in self._cache.values()]
    
    def items(self) -> List[Tuple[K, V]]:
        """获取所有键值对"""
        return [(entry.key, entry.value) for entry in self._cache.values()]


class _DummyLock:
    """非线程安全模式的虚拟锁"""
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class LRUCache(BaseCache[K, V]):
    """
    LRU缓存实现
    
    基于最近最少使用算法的缓存，支持TTL、线程安全和统计功能
    """
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        
        # 使用OrderedDict实现LRU
        self._access_order = OrderedDict()
        
        # 启动清理任务
        if config.cleanup_interval > 0:
            self._start_cleanup_task()
    
    def _start_cleanup_task(self) -> None:
        """启动清理任务"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(self.config.cleanup_interval)
                    with self._get_lock():
                        self._cleanup_expired()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"缓存清理任务出错: {e}")
        
        self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            Optional[V]: 缓存值或默认值
        """
        start_time = time.time()
        
        with self._get_lock():
            # 清理过期条目
            self._cleanup_expired()
            
            if key in self._cache:
                entry = self._cache[key]
                
                # 检查是否过期
                if entry.is_expired():
                    self._evict(key)
                    self._update_stats("miss")
                    return default
                
                # 更新访问顺序
                self._access_order.move_to_end(key)
                self._eviction_policy.on_access(key, entry)
                
                value = entry.access()
                self._update_stats("hit")
                
                # 记录访问时间
                if self.config.enable_stats:
                    access_time = time.time() - start_time
                    self._access_times.append(access_time)
                    if len(self._access_times) > 100:
                        self._access_times.pop(0)
                
                return value
            else:
                self._update_stats("miss")
                return default
    
    def set(self, key: K, value: V, ttl: Optional[float] = None) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间
            
        Returns:
            bool: 是否设置成功
        """
        with self._get_lock():
            # 清理过期条目
            self._cleanup_expired()
            
            # 处理弱引用
            if self.config.weak_reference:
                value = weakref.ref(value) if hasattr(value, '__dict__') else value
            
            # 计算TTL
            effective_ttl = ttl or self.config.ttl
            
            # 检查键是否存在
            if key in self._cache:
                # 更新现有条目
                entry = self._cache[key]
                entry.value = value
                entry.expires_at = time.time() + effective_ttl if effective_ttl else None
                entry.update_access_time()
                entry.size = entry._calculate_size()
                
                self._access_order.move_to_end(key)
                self._eviction_policy.on_access(key, entry)
                
                self._update_stats("update")
            else:
                # 创建新条目
                entry = CacheEntry(key, value, effective_ttl)
                self._cache[key] = entry
                self._access_order[key] = entry
                self._eviction_policy.on_insert(key, entry)
                
                self._update_stats("insert")
            
            # 检查是否需要淘汰
            while self._should_evict():
                evicted_key = self._evict_one()
                if not evicted_key:
                    break
            
            return True
    
    def delete(self, key: K) -> bool:
        """
        删除缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否删除成功
        """
        with self._get_lock():
            if key in self._cache:
                self._evict(key)
                self._access_order.pop(key, None)
                self._update_stats("delete")
                return True
            return False
    
    def peek(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        偷看缓存值（不更新访问顺序）
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            Optional[V]: 缓存值或默认值
        """
        with self._get_lock():
            if key in self._cache:
                entry = self._cache[key]
                if not entry.is_expired():
                    return entry.value
                else:
                    self._evict(key)
            return default
    
    def pop(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """
        获取并删除缓存值
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            Optional[V]: 缓存值或默认值
        """
        with self._get_lock():
            value = self.get(key, default)
            if value != default:
                self.delete(key)
            return value
    
    def update(self, items: Dict[K, V], ttl: Optional[float] = None) -> None:
        """
        批量更新缓存
        
        Args:
            items: 键值对字典
            ttl: 生存时间
        """
        with self._get_lock():
            for key, value in items.items():
                self.set(key, value, ttl)
    
    def get_many(self, keys: List[K]) -> Dict[K, V]:
        """
        批量获取缓存值
        
        Args:
            keys: 键列表
            
        Returns:
            Dict[K, V]: 缓存值字典
        """
        result = {}
        for key in keys:
            value = self.get(key)
            if value is not None:
                result[key] = value
        return result
    
    def set_many(self, items: Dict[K, V], ttl: Optional[float] = None) -> None:
        """
        批量设置缓存值
        
        Args:
            items: 键值对字典
            ttl: 生存时间
        """
        with self._get_lock():
            for key, value in items.items():
                self.set(key, value, ttl)
    
    def expire(self, key: K, ttl: float) -> bool:
        """
        设置键的过期时间
        
        Args:
            key: 缓存键
            ttl: 生存时间
            
        Returns:
            bool: 是否设置成功
        """
        with self._get_lock():
            if key in self._cache:
                entry = self._cache[key]
                entry.expires_at = time.time() + ttl
                return True
            return False
    
    def ttl(self, key: K) -> Optional[float]:
        """
        获取键的剩余生存时间
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[float]: 剩余生存时间（秒）
        """
        with self._get_lock():
            if key in self._cache:
                entry = self._cache[key]
                if entry.expires_at:
                    remaining = entry.expires_at - time.time()
                    return max(0, remaining)
            return None
    
    def cleanup(self) -> Dict[str, int]:
        """
        手动清理过期条目
        
        Returns:
            Dict[str, int]: 清理统计
        """
        with self._get_lock():
            expired_keys = self._cleanup_expired()
            
            return {
                'expired_count': len(expired_keys),
                'remaining_size': len(self._cache)
            }
    
    def shutdown(self) -> None:
        """关闭缓存"""
        if hasattr(self, '_cleanup_task'):
            self._cleanup_task.cancel()
        
        self.clear()
        logger.info("LRU缓存已关闭")


class WeakReferenceCache(BaseCache[K, V]):
    """
    弱引用缓存
    
    使用弱引用的缓存，当对象没有强引用时自动清理
    """
    
    def __init__(self, config: CacheConfig):
        # 弱引用缓存强制使用弱引用
        config.weak_reference = True
        super().__init__(config)
        
        # 使用弱引用字典
        self._weak_cache = weakref.WeakValueDictionary()
        
        logger.info("弱引用缓存已初始化")
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """获取缓存值"""
        start_time = time.time()
        
        with self._get_lock():
            # 清理过期条目
            self._cleanup_expired()
            
            if key in self._cache:
                entry = self._cache[key]
                
                # 检查是否过期
                if entry.is_expired():
                    self._evict(key)
                    self._update_stats("miss")
                    return default
                
                # 获取弱引用值
                value_ref = self._weak_cache.get(key)
                if value_ref is not None:
                    value = value_ref()
                    if value is not None:
                        entry.update_access_time()
                        self._eviction_policy.on_access(key, entry)
                        self._update_stats("hit")
                        
                        if self.config.enable_stats:
                            access_time = time.time() - start_time
                            self._access_times.append(access_time)
                            if len(self._access_times) > 100:
                                self._access_times.pop(0)
                        
                        return value
                
                # 弱引用已失效，清理
                self._evict(key)
                self._update_stats("miss")
                return default
            else:
                self._update_stats("miss")
                return default
    
    def set(self, key: K, value: V, ttl: Optional[float] = None) -> bool:
        """设置缓存值"""
        with self._get_lock():
            # 清理过期条目
            self._cleanup_expired()
            
            # 计算TTL
            effective_ttl = ttl or self.config.ttl
            
            # 检查键是否存在
            if key in self._cache:
                # 更新现有条目
                entry = self._cache[key]
                entry.value = value
                entry.expires_at = time.time() + effective_ttl if effective_ttl else None
                entry.update_access_time()
                entry.size = entry._calculate_size()
                
                self._weak_cache[key] = value
                self._eviction_policy.on_access(key, entry)
                
                self._update_stats("update")
            else:
                # 创建新条目
                entry = CacheEntry(key, value, effective_ttl)
                self._cache[key] = entry
                self._weak_cache[key] = value
                self._eviction_policy.on_insert(key, entry)
                
                self._update_stats("insert")
            
            # 检查是否需要淘汰
            while self._should_evict():
                evicted_key = self._evict_one()
                if not evicted_key:
                    break
            
            return True
    
    def delete(self, key: K) -> bool:
        """删除缓存值"""
        with self._get_lock():
            if key in self._cache:
                self._evict(key)
                self._weak_cache.pop(key, None)
                self._update_stats("delete")
                return True
            return False


class TTLCache(LRUCache[K, V]):
    """
    TTL缓存
    
    基于TTL（生存时间）的缓存，自动清理过期条目
    """
    
    def __init__(self, ttl: float, max_size: int = 1000):
        config = CacheConfig(
            max_size=max_size,
            ttl=ttl,
            cleanup_interval=min(ttl / 4, 60.0),
            eviction_policy="lru"
        )
        super().__init__(config)
        
        logger.info(f"TTL缓存已初始化: TTL={ttl}秒, 最大大小={max_size}")


class MultiLevelCache(BaseCache[K, V]):
    """
    多级缓存
    
    实现多级缓存系统，支持L1/L2/L3等多层缓存
    """
    
    def __init__(self, levels: List[CacheConfig]):
        """
        初始化多级缓存
        
        Args:
            levels: 各级缓存配置列表
        """
        self.levels = []
        for i, config in enumerate(levels):
            config = CacheConfig(**config.__dict__) if hasattr(config, '__dict__') else config
            level_cache = LRUCache(config)
            self.levels.append(level_cache)
        
        self.num_levels = len(self.levels)
        
        # 统计信息
        self.level_stats = [CacheStats() for _ in range(self.num_levels)]
        
        logger.info(f"多级缓存已初始化: {self.num_levels}级")
    
    def get(self, key: K, default: Optional[V] = None) -> Optional[V]:
        """从多级缓存中获取值"""
        # 从最高级开始查找
        for level in range(self.num_levels):
            cache = self.levels[level]
            value = cache.get(key)
            
            if value is not None:
                # 缓存命中，更新低级缓存
                self._propagate_to_lower_levels(key, value, level)
                
                # 更新统计
                self.level_stats[level].hits += 1
                return value
        
        # 所有级别都未命中
        for i in range(self.num_levels):
            self.level_stats[i].misses += 1
        
        return default
    
    def set(self, key: K, value: V, ttl: Optional[float] = None) -> bool:
        """设置多级缓存值"""
        # 设置所有级别
        success = True
        for cache in self.levels:
            if not cache.set(key, value, ttl):
                success = False
        
        return success
    
    def delete(self, key: K) -> bool:
        """从多级缓存中删除值"""
        success = True
        for cache in self.levels:
            if not cache.delete(key):
                success = False
        return success
    
    def _propagate_to_lower_levels(self, key: K, value: V, hit_level: int) -> None:
        """向低级缓存传播值"""
        # 向所有低级缓存传播值
        for level in range(hit_level + 1, self.num_levels):
            self.levels[level].set(key, value)
    
    def get_level_stats(self, level: int) -> Dict[str, Any]:
        """获取指定级别的统计信息"""
        if 0 <= level < self.num_levels:
            return self.levels[level].get_stats()
        else:
            return {'error': '无效的缓存级别'}
    
    def get_all_stats(self) -> Dict[str, Any]:
        """获取所有级别的统计信息"""
        stats = {}
        for i, cache in enumerate(self.levels):
            stats[f'level_{i}'] = cache.get_stats()
        
        return stats
    
    def clear_level(self, level: int) -> None:
        """清空指定级别"""
        if 0 <= level < self.num_levels:
            self.levels[level].clear()
    
    def clear_all(self) -> None:
        """清空所有级别"""
        for cache in self.levels:
            cache.clear()


# 缓存装饰器
def cached(cache_instance: BaseCache, key_func: Optional[Callable] = None):
    """
    缓存装饰器
    
    Args:
        cache_instance: 缓存实例
        key_func: 键生成函数
        
    Returns:
        Callable: 装饰器函数
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 生成缓存键
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 尝试从缓存获取
            result = cache_instance.get(cache_key)
            if result is not None:
                return result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache_instance.set(cache_key, result)
            
            return result
        
        return wrapper
    return decorator


# 使用示例
def example_usage():
    """使用示例"""
    
    # 1. LRU缓存示例
    config = CacheConfig(
        max_size=100,
        ttl=300.0,
        enable_stats=True,
        eviction_policy="lru"
    )
    
    lru_cache = LRUCache(config)
    
    # 设置缓存
    lru_cache.set("key1", "value1", ttl=60.0)
    lru_cache.set("key2", "value2")
    
    # 获取缓存
    value = lru_cache.get("key1")
    print(f"缓存值: {value}")
    
    # 批量操作
    lru_cache.set_many({
        "key3": "value3",
        "key4": "value4",
        "key5": "value5"
    })
    
    values = lru_cache.get_many(["key1", "key2", "key3"])
    print(f"批量获取: {values}")
    
    # 获取统计信息
    stats = lru_cache.get_stats()
    print(f"缓存统计: {stats}")
    
    # 2. 弱引用缓存示例
    weak_cache = WeakReferenceCache(CacheConfig(max_size=50))
    
    class TestObject:
        def __init__(self, value):
            self.value = value
        
        def __repr__(self):
            return f"TestObject({self.value})"
    
    obj = TestObject("test")
    weak_cache.set("obj1", obj)
    
    # 获取缓存
    cached_obj = weak_cache.get("obj1")
    print(f"弱引用缓存: {cached_obj}")
    
    # 3. TTL缓存示例
    ttl_cache = TTLCache(ttl=5.0, max_size=10)
    
    ttl_cache.set("temp1", "temporary_data")
    
    # 5秒后数据将过期
    import time
    time.sleep(6)
    
    temp_value = ttl_cache.get("temp1")
    print(f"TTL缓存(过期后): {temp_value}")
    
    # 4. 缓存装饰器示例
    @cached(lru_cache)
    def expensive_function(x, y):
        """模拟耗时函数"""
        time.sleep(0.1)  # 模拟耗时操作
        return x + y
    
    # 第一次调用（计算）
    result1 = expensive_function(1, 2)
    print(f"第一次调用: {result1}")
    
    # 第二次调用（缓存命中）
    result2 = expensive_function(1, 2)
    print(f"第二次调用: {result2}")
    
    # 5. 多级缓存示例
    level_configs = [
        CacheConfig(max_size=10, ttl=60.0),   # L1: 小容量，短TTL
        CacheConfig(max_size=50, ttl=300.0),  # L2: 中等容量，中TTL
        CacheConfig(max_size=200, ttl=1800.0) # L3: 大容量，长TTL
    ]
    
    multi_cache = MultiLevelCache(level_configs)
    
    multi_cache.set("multi_key", "multi_value")
    
    # 获取缓存
    multi_value = multi_cache.get("multi_key")
    print(f"多级缓存: {multi_value}")
    
    # 获取各级统计
    all_stats = multi_cache.get_all_stats()
    print(f"多级缓存统计: {all_stats}")


if __name__ == "__main__":
    example_usage()