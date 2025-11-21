#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X1内存缓存管理器
一个功能完整的内存缓存管理器，支持多种缓存策略、线程安全、持久化等功能

主要特性：
- LRU和TTL缓存清理策略
- 线程安全访问
- 内存使用限制和监控
- 缓存统计和监控
- 缓存持久化存储
- 缓存键过期管理
"""

import threading
import time
import json
import pickle
import os
import weakref
from typing import Any, Dict, Optional, Callable, Union, List, Tuple
from dataclasses import dataclass, asdict
from collections import OrderedDict
from datetime import datetime, timedelta
from abc import ABC, abstractmethod


@dataclass
class CacheEntry:
    """缓存条目数据结构"""
    key: str
    value: Any
    created_time: float
    last_accessed: float
    access_count: int = 0
    ttl: Optional[float] = None  # 生存时间（秒）
    size: int = 0  # 缓存条目大小（字节）
    
    def is_expired(self) -> bool:
        """检查缓存条目是否过期"""
        if self.ttl is None:
            return False
        return time.time() - self.created_time > self.ttl
    
    def update_access(self):
        """更新访问信息"""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def get_age(self) -> float:
        """获取缓存条目年龄（秒）"""
        return time.time() - self.created_time


class EvictionStrategy(ABC):
    """缓存清理策略抽象基类"""
    
    @abstractmethod
    def select_eviction_candidates(self, cache_entries: Dict[str, CacheEntry], count: int) -> List[str]:
        """选择需要清理的缓存条目"""
        pass


class LRUEvictionStrategy(EvictionStrategy):
    """最近最少使用清理策略"""
    
    def select_eviction_candidates(self, cache_entries: Dict[str, CacheEntry], count: int) -> List[str]:
        """选择最少使用的缓存条目进行清理"""
        # 按最后访问时间排序，选择最早访问的条目
        sorted_entries = sorted(cache_entries.items(), key=lambda x: x[1].last_accessed)
        return [key for key, _ in sorted_entries[:count]]


class TTLEvictionStrategy(EvictionStrategy):
    """TTL清理策略"""
    
    def select_eviction_candidates(self, cache_entries: Dict[str, CacheEntry], count: int) -> List[str]:
        """选择过期时间最早的缓存条目进行清理"""
        expired_entries = []
        current_time = time.time()
        
        for key, entry in cache_entries.items():
            if entry.is_expired():
                expired_entries.append((key, entry.created_time + (entry.ttl or 0)))
        
        if len(expired_entries) >= count:
            # 如果过期条目足够多，按过期时间排序
            expired_entries.sort(key=lambda x: x[1])
            return [key for key, _ in expired_entries[:count]]
        else:
            # 如果过期条目不够，清理所有过期条目
            return [key for key, _ in expired_entries]


class SizeEvictionStrategy(EvictionStrategy):
    """大小限制清理策略"""
    
    def __init__(self, max_size_bytes: int):
        self.max_size_bytes = max_size_bytes
    
    def select_eviction_candidates(self, cache_entries: Dict[str, CacheEntry], count: int) -> List[str]:
        """选择占用内存最大的缓存条目进行清理"""
        # 按大小排序，选择最大的条目
        sorted_entries = sorted(cache_entries.items(), key=lambda x: x[1].size, reverse=True)
        return [key for key, _ in sorted_entries[:count]]


class CacheStatistics:
    """缓存统计信息"""
    
    def __init__(self):
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        self.total_requests = 0
        self.start_time = time.time()
        self.last_reset_time = time.time()
        self.peak_memory_usage = 0
        self.current_memory_usage = 0
    
    @property
    def hit_rate(self) -> float:
        """缓存命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.hits / self.total_requests
    
    @property
    def uptime(self) -> float:
        """缓存运行时间（秒）"""
        return time.time() - self.start_time
    
    def reset(self):
        """重置统计信息"""
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.expirations = 0
        self.total_requests = 0
        self.last_reset_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
            'evictions': self.evictions,
            'expirations': self.expirations,
            'total_requests': self.total_requests,
            'uptime': self.uptime,
            'peak_memory_usage': self.peak_memory_usage,
            'current_memory_usage': self.current_memory_usage,
            'last_reset_time': self.last_reset_time
        }


class MemoryCacheManager:
    """X1内存缓存管理器主类"""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_bytes: int = 100 * 1024 * 1024,  # 100MB
        default_ttl: Optional[float] = None,
        eviction_strategy: str = 'lru',
        enable_persistence: bool = False,
        persistence_file: Optional[str] = None,
        enable_monitoring: bool = True,
        cleanup_interval: float = 60.0  # 清理间隔（秒）
    ):
        """
        初始化内存缓存管理器
        
        Args:
            max_size: 最大缓存条目数量
            max_memory_bytes: 最大内存使用量（字节）
            default_ttl: 默认生存时间（秒）
            eviction_strategy: 清理策略 ('lru', 'ttl', 'size')
            enable_persistence: 是否启用持久化
            persistence_file: 持久化文件路径
            enable_monitoring: 是否启用监控
            cleanup_interval: 自动清理间隔（秒）
        """
        self.max_size = max_size
        self.max_memory_bytes = max_memory_bytes
        self.default_ttl = default_ttl
        self.enable_persistence = enable_persistence
        self.persistence_file = persistence_file or 'cache_data.pkl'
        self.enable_monitoring = enable_monitoring
        self.cleanup_interval = cleanup_interval
        
        # 缓存存储
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()  # 可重入锁
        
        # 统计信息
        self._statistics = CacheStatistics()
        
        # 清理策略
        self._eviction_strategies = {
            'lru': LRUEvictionStrategy(),
            'ttl': TTLEvictionStrategy(),
            'size': SizeEvictionStrategy(max_memory_bytes)
        }
        self._eviction_strategy = self._eviction_strategies.get(eviction_strategy, LRUEvictionStrategy())
        
        # 监控相关
        self._monitoring_enabled = enable_monitoring
        self._last_cleanup_time = time.time()
        self._cleanup_thread = None
        self._shutdown_event = threading.Event()
        
        # 启动自动清理
        if self.cleanup_interval > 0:
            self._start_cleanup_thread()
        
        # 加载持久化数据
        if self.enable_persistence and os.path.exists(self.persistence_file):
            self._load_from_disk()
    
    def _start_cleanup_thread(self):
        """启动自动清理线程"""
        def cleanup_worker():
            while not self._shutdown_event.wait(self.cleanup_interval):
                try:
                    self._perform_cleanup()
                except Exception as e:
                    if self._monitoring_enabled:
                        print(f"缓存清理异常: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _stop_cleanup_thread(self):
        """停止自动清理线程"""
        if self._cleanup_thread:
            self._shutdown_event.set()
            self._cleanup_thread.join(timeout=1.0)
    
    def _calculate_size(self, obj: Any) -> int:
        """计算对象大小（字节）"""
        try:
            return len(pickle.dumps(obj))
        except Exception:
            # 如果无法序列化，返回估算值
            return 64  # 默认估算值
    
    def _perform_cleanup(self):
        """执行缓存清理"""
        current_time = time.time()
        
        # 清理过期条目
        expired_keys = []
        for key, entry in self._cache.items():
            if entry.is_expired():
                expired_keys.append(key)
        
        # 移除过期条目
        for key in expired_keys:
            self._remove_entry(key)
            self._statistics.expirations += 1
        
        # 检查内存使用情况
        current_memory = self._get_current_memory_usage()
        if current_memory > self.max_memory_bytes:
            # 需要清理内存
            excess_memory = current_memory - self.max_memory_bytes
            self._evict_by_memory(excess_memory)
        
        # 检查条目数量
        if len(self._cache) > self.max_size:
            excess_count = len(self._cache) - self.max_size
            self._evict_by_count(excess_count)
        
        self._last_cleanup_time = current_time
        self._update_memory_stats()
    
    def _evict_by_memory(self, excess_memory: int):
        """根据内存使用量清理缓存"""
        candidates = []
        current_memory = 0
        
        for key, entry in self._cache.items():
            candidates.append((key, entry))
            current_memory += entry.size
        
        # 按大小排序，优先清理大条目
        candidates.sort(key=lambda x: x[1].size, reverse=True)
        
        evicted_count = 0
        for key, entry in candidates:
            if current_memory <= self.max_memory_bytes:
                break
            
            self._remove_entry(key)
            current_memory -= entry.size
            self._statistics.evictions += 1
            evicted_count += 1
    
    def _evict_by_count(self, excess_count: int):
        """根据条目数量清理缓存"""
        candidates = self._eviction_strategy.select_eviction_candidates(self._cache, excess_count)
        
        for key in candidates:
            self._remove_entry(key)
            self._statistics.evictions += 1
    
    def _remove_entry(self, key: str) -> bool:
        """移除缓存条目"""
        if key in self._cache:
            del self._cache[key]
            return True
        return False
    
    def _get_current_memory_usage(self) -> int:
        """获取当前内存使用量"""
        return sum(entry.size for entry in self._cache.values())
    
    def _update_memory_stats(self):
        """更新内存统计信息"""
        current_memory = self._get_current_memory_usage()
        self._statistics.current_memory_usage = current_memory
        if current_memory > self._statistics.peak_memory_usage:
            self._statistics.peak_memory_usage = current_memory
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """
        存储缓存数据
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒），None表示使用默认值
            
        Returns:
            bool: 是否成功存储
        """
        with self._lock:
            # 计算大小
            size = self._calculate_size(value)
            
            # 创建缓存条目
            current_time = time.time()
            entry = CacheEntry(
                key=key,
                value=value,
                created_time=current_time,
                last_accessed=current_time,
                ttl=ttl or self.default_ttl,
                size=size
            )
            
            # 如果键已存在，移除旧条目
            if key in self._cache:
                del self._cache[key]
            
            # 添加新条目到末尾（LRU）
            self._cache[key] = entry
            
            # 检查是否需要清理
            self._perform_cleanup()
            
            # 更新统计
            self._update_memory_stats()
            
            return True
    
    def get(self, key: str, default: Any = None, update_access: bool = True) -> Any:
        """
        获取缓存数据
        
        Args:
            key: 缓存键
            default: 默认值
            update_access: 是否更新访问信息
            
        Returns:
            Any: 缓存值或默认值
        """
        with self._lock:
            self._statistics.total_requests += 1
            
            if key not in self._cache:
                self._statistics.misses += 1
                return default
            
            entry = self._cache[key]
            
            # 检查是否过期
            if entry.is_expired():
                self._remove_entry(key)
                self._statistics.misses += 1
                self._statistics.expirations += 1
                return default
            
            # 更新访问信息
            if update_access:
                entry.update_access()
                # 将条目移动到末尾（LRU）
                self._cache.move_to_end(key)
            
            self._statistics.hits += 1
            return entry.value
    
    def delete(self, key: str) -> bool:
        """
        删除缓存条目
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否成功删除
        """
        with self._lock:
            return self._remove_entry(key)
    
    def clear(self):
        """清空所有缓存"""
        with self._lock:
            self._cache.clear()
            self._statistics.reset()
    
    def contains(self, key: str) -> bool:
        """
        检查缓存是否包含指定键
        
        Args:
            key: 缓存键
            
        Returns:
            bool: 是否包含该键
        """
        with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            if entry.is_expired():
                self._remove_entry(key)
                return False
            
            return True
    
    def get_or_set(self, key: str, factory: Callable[[], Any], ttl: Optional[float] = None) -> Any:
        """
        获取缓存值，如果不存在则设置并返回
        
        Args:
            key: 缓存键
            factory: 值生成函数
            ttl: 生存时间（秒）
            
        Returns:
            Any: 缓存值
        """
        # 尝试获取现有值
        value = self.get(key, update_access=False)
        if value is not None:
            return value
        
        # 生成新值
        try:
            new_value = factory()
        except Exception as e:
            if self._monitoring_enabled:
                print(f"缓存值生成异常: {e}")
            raise
        
        # 存储新值
        self.put(key, new_value, ttl)
        return new_value
    
    def get_ttl(self, key: str) -> Optional[float]:
        """
        获取缓存条目的剩余生存时间
        
        Args:
            key: 缓存键
            
        Returns:
            Optional[float]: 剩余生存时间（秒），None表示永不过期
        """
        with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            if entry.ttl is None:
                return None
            
            elapsed = time.time() - entry.created_time
            remaining = entry.ttl - elapsed
            return max(0, remaining)
    
    def set_ttl(self, key: str, ttl: Optional[float]) -> bool:
        """
        设置缓存条目的生存时间
        
        Args:
            key: 缓存键
            ttl: 生存时间（秒），None表示永不过期
            
        Returns:
            bool: 是否成功设置
        """
        with self._lock:
            if key not in self._cache:
                return False
            
            entry = self._cache[key]
            entry.ttl = ttl
            return True
    
    def get_keys(self, pattern: Optional[str] = None) -> List[str]:
        """
        获取所有缓存键
        
        Args:
            pattern: 键匹配模式，支持通配符
            
        Returns:
            List[str]: 缓存键列表
        """
        with self._lock:
            keys = list(self._cache.keys())
            
            if pattern is None:
                return keys
            
            # 简单的模式匹配
            import fnmatch
            return [key for key in keys if fnmatch.fnmatch(key, pattern)]
    
    def get_entries(self) -> List[Tuple[str, Any, float, float, int]]:
        """
        获取所有缓存条目信息
        
        Returns:
            List[Tuple[str, Any, float, float, int]]: 
            (键, 值, 创建时间, 最后访问时间, 访问次数)
        """
        with self._lock:
            return [
                (entry.key, entry.value, entry.created_time, entry.last_accessed, entry.access_count)
                for entry in self._cache.values()
            ]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        with self._lock:
            stats = self._statistics.to_dict()
            stats.update({
                'current_size': len(self._cache),
                'max_size': self.max_size,
                'current_memory_usage': self._get_current_memory_usage(),
                'max_memory_bytes': self.max_memory_bytes,
                'default_ttl': self.default_ttl,
                'cleanup_interval': self.cleanup_interval,
                'last_cleanup_time': self._last_cleanup_time
            })
            return stats
    
    def get_memory_usage(self) -> Dict[str, Union[int, float]]:
        """
        获取内存使用情况
        
        Returns:
            Dict[str, Union[int, float]]: 内存使用信息
        """
        with self._lock:
            current_memory = self._get_current_memory_usage()
            return {
                'current_bytes': current_memory,
                'max_bytes': self.max_memory_bytes,
                'usage_percentage': (current_memory / self.max_memory_bytes) * 100 if self.max_memory_bytes > 0 else 0,
                'entry_count': len(self._cache),
                'average_entry_size': current_memory / len(self._cache) if len(self._cache) > 0 else 0
            }
    
    def set_eviction_strategy(self, strategy: str):
        """
        设置清理策略
        
        Args:
            strategy: 清理策略 ('lru', 'ttl', 'size')
        """
        with self._lock:
            if strategy in self._eviction_strategies:
                self._eviction_strategy = self._eviction_strategies[strategy]
            else:
                raise ValueError(f"不支持的清理策略: {strategy}")
    
    def save_to_disk(self):
        """保存缓存数据到磁盘"""
        if not self.enable_persistence:
            return
        
        with self._lock:
            try:
                data = {
                    'cache': {key: asdict(entry) for key, entry in self._cache.items()},
                    'statistics': self._statistics.to_dict(),
                    'config': {
                        'max_size': self.max_size,
                        'max_memory_bytes': self.max_memory_bytes,
                        'default_ttl': self.default_ttl,
                        'cleanup_interval': self.cleanup_interval
                    }
                }
                
                with open(self.persistence_file, 'wb') as f:
                    pickle.dump(data, f)
                    
            except Exception as e:
                if self._monitoring_enabled:
                    print(f"保存缓存数据异常: {e}")
    
    def _load_from_disk(self):
        """从磁盘加载缓存数据"""
        try:
            with open(self.persistence_file, 'rb') as f:
                data = pickle.load(f)
            
            # 恢复缓存数据
            cache_data = data.get('cache', {})
            for key, entry_data in cache_data.items():
                entry = CacheEntry(**entry_data)
                self._cache[key] = entry
            
            # 恢复统计信息
            stats_data = data.get('statistics', {})
            for key, value in stats_data.items():
                if hasattr(self._statistics, key):
                    setattr(self._statistics, key, value)
            
            self._update_memory_stats()
            
        except Exception as e:
            if self._monitoring_enabled:
                print(f"加载缓存数据异常: {e}")
    
    def export_data(self, format: str = 'json') -> str:
        """
        导出缓存数据
        
        Args:
            format: 导出格式 ('json' 或 'pickle')
            
        Returns:
            str: 导出的数据字符串
        """
        with self._lock:
            data = {
                'cache': {key: asdict(entry) for key, entry in self._cache.items()},
                'statistics': self._statistics.to_dict(),
                'export_time': time.time()
            }
            
            if format.lower() == 'json':
                return json.dumps(data, indent=2, default=str)
            elif format.lower() == 'pickle':
                return pickle.dumps(data)
            else:
                raise ValueError(f"不支持的导出格式: {format}")
    
    def import_data(self, data: str, format: str = 'json'):
        """
        导入缓存数据
        
        Args:
            data: 数据字符串
            format: 数据格式 ('json' 或 'pickle')
        """
        with self._lock:
            try:
                if format.lower() == 'json':
                    parsed_data = json.loads(data)
                elif format.lower() == 'pickle':
                    parsed_data = pickle.loads(data.encode('latin1'))
                else:
                    raise ValueError(f"不支持的导入格式: {format}")
                
                # 恢复缓存数据
                cache_data = parsed_data.get('cache', {})
                self._cache.clear()
                
                for key, entry_data in cache_data.items():
                    try:
                        entry = CacheEntry(**entry_data)
                        self._cache[key] = entry
                    except Exception as e:
                        if self._monitoring_enabled:
                            print(f"导入缓存条目异常 {key}: {e}")
                
                self._update_memory_stats()
                
            except Exception as e:
                if self._monitoring_enabled:
                    print(f"导入缓存数据异常: {e}")
                raise
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self._stop_cleanup_thread()
        if self.enable_persistence:
            self.save_to_disk()
    
    def __len__(self):
        """返回缓存条目数量"""
        with self._lock:
            return len(self._cache)
    
    def __contains__(self, key):
        """检查是否包含指定键"""
        return self.contains(key)
    
    def __getitem__(self, key):
        """获取缓存值（字典式访问）"""
        return self.get(key)
    
    def __setitem__(self, key, value):
        """设置缓存值（字典式访问）"""
        self.put(key, value)
    
    def __delitem__(self, key):
        """删除缓存值（字典式访问）"""
        self.delete(key)
    
    def __iter__(self):
        """迭代缓存键"""
        return iter(self.get_keys())
    
    def close(self):
        """关闭缓存管理器"""
        self._stop_cleanup_thread()
        if self.enable_persistence:
            self.save_to_disk()


# 便捷函数
def create_cache_manager(**kwargs) -> MemoryCacheManager:
    """
    创建内存缓存管理器的便捷函数
    
    Args:
        **kwargs: MemoryCacheManager的初始化参数
        
    Returns:
        MemoryCacheManager: 缓存管理器实例
    """
    return MemoryCacheManager(**kwargs)


def get_global_cache() -> Optional[MemoryCacheManager]:
    """获取全局缓存管理器实例"""
    return getattr(get_global_cache, '_instance', None)


def set_global_cache(cache_manager: MemoryCacheManager):
    """设置全局缓存管理器实例"""
    get_global_cache._instance = cache_manager


# 全局缓存实例
_global_cache = None


def get_cache() -> MemoryCacheManager:
    """获取或创建全局缓存管理器"""
    global _global_cache
    if _global_cache is None:
        _global_cache = MemoryCacheManager()
    return _global_cache


def clear_global_cache():
    """清空全局缓存"""
    global _global_cache
    if _global_cache:
        _global_cache.clear()


if __name__ == "__main__":
    # 简单使用示例
    cache = MemoryCacheManager(max_size=100, default_ttl=300)
    
    # 存储数据
    cache.put("user:1", {"name": "张三", "age": 25})
    cache.put("user:2", {"name": "李四", "age": 30}, ttl=60)
    
    # 获取数据
    user1 = cache.get("user:1")
    print(f"用户1: {user1}")
    
    # 检查缓存统计
    stats = cache.get_statistics()
    print(f"缓存统计: {stats}")
    
    # 关闭缓存
    cache.close()