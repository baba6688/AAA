"""
O7缓存优化器模块

提供企业级缓存优化解决方案，包括：
1. 多种缓存策略（LRU、LFU、TTL、缓存分层）
2. 缓存预热和加载策略（预加载、懒加载、批量加载）
3. 缓存失效和更新策略（主动失效、被动失效、版本控制）
4. 分布式缓存优化（缓存一致性、缓存同步、缓存分片）
5. 缓存监控和性能分析（命中率、延迟、内存使用）
6. 缓存安全性和可靠性（缓存穿透、缓存击穿、缓存雪崩）
7. 异步缓存优化处理
8. 完整的错误处理和日志记录

作者: O7 Cache Optimizer Team
版本: 1.0.0
"""

import asyncio
import time
import threading
import hashlib
import json
import pickle
import logging
import weakref
import random
import math
import psutil
import gc
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from functools import wraps, lru_cache
from typing import (
    Any, Dict, List, Optional, Union, Callable, Tuple, Set, 
    Generic, TypeVar, Protocol, Awaitable, NamedTuple
)
import zlib
import gzip
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 类型定义
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class CacheStrategy(Enum):
    """缓存策略枚举"""
    LRU = "lru"           # 最近最少使用
    LFU = "lfu"           # 最少使用频率
    TTL = "ttl"           # 时间过期
    FIFO = "fifo"         # 先进先出
    ADAPTIVE = "adaptive" # 自适应策略
    HYBRID = "hybrid"     # 混合策略

class CacheLevel(Enum):
    """缓存层级枚举"""
    L1 = "l1"     # 内存缓存
    L2 = "l2"     # 本地磁盘缓存
    L3 = "l3"     # 分布式缓存

class InvalidationType(Enum):
    """失效类型枚举"""
    TIME_BASED = "time_based"      # 基于时间
    ACCESS_BASED = "access_based"  # 基于访问次数
    SIZE_BASED = "size_based"      # 基于大小
    MANUAL = "manual"              # 手动失效
    VERSION_BASED = "version_based" # 基于版本

class ConsistencyLevel(Enum):
    """一致性级别枚举"""
    EVENTUAL = "eventual"    # 最终一致性
    STRONG = "strong"        # 强一致性
    WEAK = "weak"            # 弱一致性

class SecurityLevel(Enum):
    """安全级别枚举"""
    NONE = "none"
    BASIC = "basic"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_time: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    ttl: Optional[int] = None  # 生存时间（秒）
    version: str = "1.0"
    compressed: bool = False
    size: int = 0
    tags: Set[str] = field(default_factory=set)
    
    def is_expired(self) -> bool:
        """检查是否过期"""
        if self.ttl is None:
            return False
        return (datetime.now() - self.created_time).total_seconds() > self.ttl
    
    def update_access(self):
        """更新访问信息"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def calculate_score(self, strategy: CacheStrategy) -> float:
        """根据策略计算分数"""
        if strategy == CacheStrategy.LRU:
            return (datetime.now() - self.last_accessed).total_seconds()
        elif strategy == CacheStrategy.LFU:
            return 1.0 / (self.access_count + 1)
        elif strategy == CacheStrategy.FIFO:
            return (datetime.now() - self.created_time).total_seconds()
        else:
            return 0.0

@dataclass
class CacheMetrics:
    """缓存性能指标"""
    hit_count: int = 0
    miss_count: int = 0
    total_requests: int = 0
    avg_latency: float = 0.0
    memory_usage: int = 0
    disk_usage: int = 0
    evictions: int = 0
    expired_entries: int = 0
    compression_ratio: float = 0.0
    load_factor: float = 0.0
    
    @property
    def hit_ratio(self) -> float:
        """命中率"""
        if self.total_requests == 0:
            return 0.0
        return self.hit_count / self.total_requests
    
    @property
    def miss_ratio(self) -> float:
        """未命中率"""
        return 1.0 - self.hit_ratio
    
    def update_latency(self, latency: float):
        """更新延迟指标"""
        if self.total_requests == 0:
            self.avg_latency = latency
        else:
            self.avg_latency = (self.avg_latency * (self.total_requests - 1) + latency) / self.total_requests

@dataclass
class CacheConfig:
    """缓存配置"""
    max_size: int = 1000
    max_memory: int = 100 * 1024 * 1024  # 100MB
    default_ttl: int = 3600  # 1小时
    strategy: CacheStrategy = CacheStrategy.LRU
    enable_compression: bool = True
    compression_level: int = 6
    enable_persistence: bool = False
    persistence_path: Optional[str] = None
    consistency_level: ConsistencyLevel = ConsistencyLevel.EVENTUAL
    security_level: SecurityLevel = SecurityLevel.BASIC
    enable_monitoring: bool = True
    monitoring_interval: int = 60  # 秒
    enable_prefetching: bool = True
    prefetch_count: int = 10
    enable_batch_loading: bool = True
    batch_size: int = 50
    enable_distributed: bool = False
    nodes: List[str] = field(default_factory=list)
    replication_factor: int = 1
    enable_security: bool = True
    encryption_key: Optional[str] = None

class CachePolicy(ABC):
    """缓存策略抽象基类"""
    
    @abstractmethod
    def should_evict(self, entry: CacheEntry) -> bool:
        """判断是否应该淘汰该条目"""
        pass
    
    @abstractmethod
    def select_eviction_candidates(self, entries: Dict[str, CacheEntry], count: int) -> List[str]:
        """选择淘汰候选条目"""
        pass

class LRUCachePolicy(CachePolicy):
    """LRU缓存策略"""
    
    def should_evict(self, entry: CacheEntry) -> bool:
        return False  # LRU通过访问时间判断
    
    def select_eviction_candidates(self, entries: Dict[str, CacheEntry], count: int) -> List[str]:
        # 按最后访问时间排序，选择最久未访问的
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].last_accessed)
        return [key for key, _ in sorted_entries[:count]]

class LFUCachePolicy(CachePolicy):
    """LFU缓存策略"""
    
    def should_evict(self, entry: CacheEntry) -> bool:
        return False  # LFU通过访问频率判断
    
    def select_eviction_candidates(self, entries: Dict[str, CacheEntry], count: int) -> List[str]:
        # 按访问频率排序，选择访问次数最少的
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].access_count)
        return [key for key, _ in sorted_entries[:count]]

class TTLCachePolicy(CachePolicy):
    """TTL缓存策略"""
    
    def should_evict(self, entry: CacheEntry) -> bool:
        return entry.is_expired()
    
    def select_eviction_candidates(self, entries: Dict[str, CacheEntry], count: int) -> List[str]:
        # 优先淘汰过期的条目
        expired = [key for key, entry in entries.items() if entry.is_expired()]
        if len(expired) >= count:
            return expired[:count]
        
        # 如果过期条目不足，再淘汰即将过期的
        remaining = count - len(expired)
        soon_to_expire = sorted(
            [(key, entry) for key, entry in entries.items() if not entry.is_expired()],
            key=lambda x: x[1].created_time + timedelta(seconds=x[1].ttl or 0)
        )
        return expired + [key for key, _ in soon_to_expire[:remaining]]

class AdaptiveCachePolicy(CachePolicy):
    """自适应缓存策略"""
    
    def __init__(self):
        self.hit_ratio_history = deque(maxlen=100)
        self.strategy_weights = {
            CacheStrategy.LRU: 0.4,
            CacheStrategy.LFU: 0.3,
            CacheStrategy.TTL: 0.3
        }
    
    def should_evict(self, entry: CacheEntry) -> bool:
        return False
    
    def select_eviction_candidates(self, entries: Dict[str, CacheEntry], count: int) -> List[str]:
        # 根据历史命中率调整策略权重
        if self.hit_ratio_history:
            recent_hit_ratio = sum(self.hit_ratio_history) / len(self.hit_ratio_history)
            if recent_hit_ratio < 0.7:  # 命中率低，增加TTL权重
                self.strategy_weights[CacheStrategy.TTL] = min(0.6, self.strategy_weights[CacheStrategy.TTL] + 0.1)
            elif recent_hit_ratio > 0.9:  # 命中率高，增加LRU权重
                self.strategy_weights[CacheStrategy.LRU] = min(0.6, self.strategy_weights[CacheStrategy.LRU] + 0.1)
        
        # 综合多种策略选择淘汰候选
        candidates = []
        
        # LRU候选
        lru_candidates = LRUCachePolicy().select_eviction_candidates(entries, count // 2)
        candidates.extend(lru_candidates)
        
        # LFU候选
        lfu_candidates = LFUCachePolicy().select_eviction_candidates(entries, count // 2)
        candidates.extend(lfu_candidates)
        
        # 去重并限制数量
        unique_candidates = list(dict.fromkeys(candidates))[:count]
        
        # 如果候选不足，用TTL策略补充
        if len(unique_candidates) < count:
            ttl_candidates = TTLCachePolicy().select_eviction_candidates(entries, count - len(unique_candidates))
            for candidate in ttl_candidates:
                if candidate not in unique_candidates:
                    unique_candidates.append(candidate)
        
        return unique_candidates

class CacheLoadBalancer:
    """缓存负载均衡器"""
    
    def __init__(self, nodes: List[str]):
        self.nodes = nodes
        self.node_weights = {node: 1.0 for node in nodes}
        self.node_loads = {node: 0 for node in nodes}
        self.node_failures = {node: 0 for node in nodes}
        self.lock = threading.RLock()
    
    def select_node(self, key: str) -> str:
        """选择节点"""
        with self.lock:
            # 过滤掉故障节点
            available_nodes = [node for node in self.nodes if self.node_failures[node] < 3]
            
            if not available_nodes:
                # 如果所有节点都故障，回退到所有节点
                available_nodes = self.nodes
            
            # 根据权重和负载选择节点
            weights = []
            for node in available_nodes:
                weight = self.node_weights[node] / (self.node_loads[node] + 1)
                weights.append(weight)
            
            # 加权随机选择
            total_weight = sum(weights)
            if total_weight == 0:
                return random.choice(available_nodes)
            
            r = random.uniform(0, total_weight)
            cumulative = 0
            for i, node in enumerate(available_nodes):
                cumulative += weights[i]
                if r <= cumulative:
                    return node
            
            return available_nodes[-1]
    
    def update_node_load(self, node: str, load_delta: int):
        """更新节点负载"""
        with self.lock:
            self.node_loads[node] = max(0, self.node_loads[node] + load_delta)
    
    def report_node_failure(self, node: str):
        """报告节点故障"""
        with self.lock:
            self.node_failures[node] += 1
            if self.node_failures[node] >= 3:
                logger.warning(f"Node {node} marked as failed")
    
    def report_node_recovery(self, node: str):
        """报告节点恢复"""
        with self.lock:
            self.node_failures[node] = 0
            logger.info(f"Node {node} recovered")

class DistributedCacheNode:
    """分布式缓存节点"""
    
    def __init__(self, node_id: str, host: str, port: int, config: CacheConfig):
        self.node_id = node_id
        self.host = host
        self.port = port
        self.config = config
        self.cache = {}
        self.metrics = CacheMetrics()
        self.is_healthy = True
        self.last_heartbeat = datetime.now()
        self.replication_factor = config.replication_factor
        self.consistency_level = config.consistency_level
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        start_time = time.time()
        
        with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                self.metrics.miss_count += 1
                return None
            
            if entry.is_expired():
                del self.cache[key]
                self.metrics.expired_entries += 1
                self.metrics.miss_count += 1
                return None
            
            entry.update_access()
            self.metrics.hit_count += 1
            self.metrics.total_requests += 1
            
            latency = time.time() - start_time
            self.metrics.update_latency(latency)
            
            # 解压缩
            if entry.compressed:
                return self._decompress(entry.value)
            return entry.value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            version: str = "1.0", tags: Optional[Set[str]] = None) -> bool:
        """设置缓存值"""
        try:
            with self.lock:
                # 压缩
                compressed_value = value
                compressed = False
                if self.config.enable_compression:
                    compressed_value, compressed = self._compress(value)
                
                # 计算大小
                size = len(pickle.dumps(compressed_value))
                
                entry = CacheEntry(
                    key=key,
                    value=compressed_value,
                    ttl=ttl or self.config.default_ttl,
                    version=version,
                    compressed=compressed,
                    size=size,
                    tags=tags or set()
                )
                
                # 检查内存限制
                if self._check_memory_limit():
                    self._evict_entries(1)
                
                self.cache[key] = entry
                self.metrics.total_requests += 1
                self._update_memory_usage()
                
                return True
                
        except Exception as e:
            logger.error(f"Failed to set cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """删除缓存值"""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                self._update_memory_usage()
                return True
            return False
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.metrics = CacheMetrics()
    
    def _compress(self, data: Any) -> Tuple[Any, bool]:
        """压缩数据"""
        try:
            serialized = pickle.dumps(data)
            compressed = zlib.compress(serialized, self.config.compression_level)
            return compressed, True
        except Exception:
            return data, False
    
    def _decompress(self, data: Any) -> Any:
        """解压缩数据"""
        try:
            decompressed = zlib.decompress(data)
            return pickle.loads(decompressed)
        except Exception:
            return data
    
    def _check_memory_limit(self) -> bool:
        """检查内存限制"""
        current_usage = sum(entry.size for entry in self.cache.values())
        return current_usage > self.config.max_memory
    
    def _evict_entries(self, count: int):
        """淘汰条目"""
        if not self.cache:
            return
        
        policy = self._get_policy()
        candidates = policy.select_eviction_candidates(self.cache, count)
        
        for key in candidates:
            if key in self.cache:
                del self.cache[key]
                self.metrics.evictions += 1
        
        self._update_memory_usage()
    
    def _get_policy(self) -> CachePolicy:
        """获取缓存策略"""
        if self.config.strategy == CacheStrategy.LRU:
            return LRUCachePolicy()
        elif self.config.strategy == CacheStrategy.LFU:
            return LFUCachePolicy()
        elif self.config.strategy == CacheStrategy.TTL:
            return TTLCachePolicy()
        else:
            return AdaptiveCachePolicy()
    
    def _update_memory_usage(self):
        """更新内存使用量"""
        self.metrics.memory_usage = sum(entry.size for entry in self.cache.values())
    
    def get_metrics(self) -> CacheMetrics:
        """获取性能指标"""
        return self.metrics
    
    def is_healthy_check(self) -> bool:
        """健康检查"""
        return self.is_healthy

class CacheSecurityManager:
    """缓存安全管理器"""
    
    def __init__(self, security_level: SecurityLevel, encryption_key: Optional[str] = None):
        self.security_level = security_level
        self.encryption_key = encryption_key
        self.blocked_keys = set()
        self.rate_limits = defaultdict(int)
        self.suspicious_patterns = []
    
    def validate_key(self, key: str) -> bool:
        """验证缓存键"""
        if self.security_level == SecurityLevel.NONE:
            return True
        
        # 检查是否被阻止
        if key in self.blocked_keys:
            return False
        
        # 检查键的长度
        if len(key) > 1000:
            return False
        
        # 检查是否包含可疑模式
        for pattern in self.suspicious_patterns:
            if pattern in key:
                return False
        
        return True
    
    def validate_value(self, value: Any) -> bool:
        """验证缓存值"""
        if self.security_level == SecurityLevel.NONE:
            return True
        
        # 检查值的大小
        value_size = len(pickle.dumps(value))
        if value_size > 100 * 1024 * 1024:  # 100MB
            return False
        
        # 检查值的类型
        if not isinstance(value, (str, int, float, bool, list, dict, tuple, set)):
            return False
        
        return True
    
    def encrypt_data(self, data: Any) -> bytes:
        """加密数据"""
        if self.security_level in [SecurityLevel.NONE, SecurityLevel.BASIC]:
            return pickle.dumps(data)
        
        # 简单的XOR加密（实际应用中应使用更强的加密算法）
        if self.encryption_key:
            serialized = pickle.dumps(data)
            key_bytes = self.encryption_key.encode()[:len(serialized)]
            encrypted = bytes(a ^ b for a, b in zip(serialized, key_bytes))
            return encrypted
        
        return pickle.dumps(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> Any:
        """解密数据"""
        if self.security_level in [SecurityLevel.NONE, SecurityLevel.BASIC]:
            return pickle.loads(encrypted_data)
        
        if self.encryption_key:
            key_bytes = self.encryption_key.encode()[:len(encrypted_data)]
            decrypted = bytes(a ^ b for a, b in zip(encrypted_data, key_bytes))
            return pickle.loads(decrypted)
        
        return pickle.loads(encrypted_data)
    
    def check_rate_limit(self, client_id: str, limit: int = 100) -> bool:
        """检查速率限制"""
        current_time = int(time.time())
        window_start = current_time - (current_time % 60)  # 1分钟窗口
        
        key = f"{client_id}:{window_start}"
        self.rate_limits[key] += 1
        
        return self.rate_limits[key] <= limit
    
    def add_blocked_key(self, key: str):
        """添加被阻止的键"""
        self.blocked_keys.add(key)
    
    def remove_blocked_key(self, key: str):
        """移除被阻止的键"""
        self.blocked_keys.discard(key)
    
    def add_suspicious_pattern(self, pattern: str):
        """添加可疑模式"""
        self.suspicious_patterns.append(pattern)

class AsyncCacheProcessor:
    """异步缓存处理器"""
    
    def __init__(self, max_workers: int = 10):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_tasks = {}
        self.task_results = {}
        self.lock = threading.RLock()
    
    async def async_get(self, cache: 'CacheOptimizer', key: str) -> Optional[Any]:
        """异步获取缓存"""
        loop = asyncio.get_event_loop()
        
        def _get():
            return cache.get(key)
        
        return await loop.run_in_executor(self.executor, _get)
    
    async def async_set(self, cache: 'CacheOptimizer', key: str, value: Any, 
                       ttl: Optional[int] = None) -> bool:
        """异步设置缓存"""
        loop = asyncio.get_event_loop()
        
        def _set():
            return cache.set(key, value, ttl)
        
        return await loop.run_in_executor(self.executor, _set)
    
    async def async_batch_get(self, cache: 'CacheOptimizer', keys: List[str]) -> Dict[str, Any]:
        """异步批量获取"""
        loop = asyncio.get_event_loop()
        
        async def _batch_get():
            return cache.batch_get(keys)
        
        return await loop.run_in_executor(self.executor, _batch_get)
    
    async def async_batch_set(self, cache: 'CacheOptimizer', data: Dict[str, Any],
                             ttl: Optional[int] = None) -> Dict[str, bool]:
        """异步批量设置"""
        loop = asyncio.get_event_loop()
        
        def _batch_set():
            return cache.batch_set(data, ttl)
        
        return await loop.run_in_executor(self.executor, _batch_set)
    
    async def async_preload(self, cache: 'CacheOptimizer', keys: List[str],
                           loader_func: Callable[[str], Any]) -> Dict[str, Any]:
        """异步预加载"""
        loop = asyncio.get_event_loop()
        
        async def _preload():
            return cache.preload(keys, loader_func)
        
        return await loop.run_in_executor(self.executor, _preload)
    
    def submit_background_task(self, task_id: str, func: Callable, *args, **kwargs):
        """提交后台任务"""
        with self.lock:
            future = self.executor.submit(func, *args, **kwargs)
            self.pending_tasks[task_id] = future
            return future
    
    def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """获取任务结果"""
        with self.lock:
            if task_id not in self.pending_tasks:
                raise ValueError(f"Task {task_id} not found")
            
            future = self.pending_tasks[task_id]
            try:
                result = future.result(timeout=timeout)
                self.task_results[task_id] = result
                del self.pending_tasks[task_id]
                return result
            except Exception as e:
                logger.error(f"Task {task_id} failed: {e}")
                del self.pending_tasks[task_id]
                raise
    
    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        with self.lock:
            if task_id in self.pending_tasks:
                future = self.pending_tasks[task_id]
                cancelled = future.cancel()
                if cancelled:
                    del self.pending_tasks[task_id]
                return cancelled
            return False
    
    def shutdown(self, wait: bool = True):
        """关闭处理器"""
        self.executor.shutdown(wait=wait)

class CacheOptimizer:
    """
    O7缓存优化器主类
    
    提供企业级缓存解决方案，支持多种缓存策略、分布式缓存、
    性能监控、安全保护等功能。
    """
    
    def __init__(self, config: Optional[CacheConfig] = None):
        """
        初始化缓存优化器
        
        Args:
            config: 缓存配置，如果为None则使用默认配置
        """
        self.config = config or CacheConfig()
        self.cache = {}
        self.metrics = CacheMetrics()
        self.security_manager = CacheSecurityManager(
            self.config.security_level,
            self.config.encryption_key
        )
        self.async_processor = AsyncCacheProcessor()
        
        # 分布式缓存相关
        self.distributed_enabled = self.config.enable_distributed
        self.distributed_nodes = {}
        self.load_balancer = None
        if self.distributed_enabled:
            self._init_distributed_cache()
        
        # 缓存策略
        self.policy = self._get_policy()
        
        # 监控相关
        self.monitoring_enabled = self.config.enable_monitoring
        self.monitoring_task = None
        if self.monitoring_enabled:
            self._start_monitoring()
        
        # 预热和批量加载
        self.prefetch_enabled = self.config.enable_prefetching
        self.batch_loading_enabled = self.config.enable_batch_loading
        
        # 线程锁
        self.lock = threading.RLock()
        
        logger.info(f"CacheOptimizer initialized with strategy: {self.config.strategy}")
    
    def _init_distributed_cache(self):
        """初始化分布式缓存"""
        for i, node_info in enumerate(self.config.nodes):
            node_id = f"node_{i}"
            # 解析节点信息（host:port格式）
            if ':' in node_info:
                host, port = node_info.split(':')
                port = int(port)
            else:
                host, port = node_info, 6379  # 默认Redis端口
            
            node = DistributedCacheNode(node_id, host, port, self.config)
            self.distributed_nodes[node_id] = node
        
        if self.distributed_nodes:
            node_list = list(self.distributed_nodes.keys())
            self.load_balancer = CacheLoadBalancer(node_list)
    
    def _get_policy(self) -> CachePolicy:
        """获取缓存策略"""
        if self.config.strategy == CacheStrategy.LRU:
            return LRUCachePolicy()
        elif self.config.strategy == CacheStrategy.LFU:
            return LFUCachePolicy()
        elif self.config.strategy == CacheStrategy.TTL:
            return TTLCachePolicy()
        elif self.config.strategy == CacheStrategy.FIFO:
            return FIFO
        else:
            return AdaptiveCachePolicy()
    
    def _start_monitoring(self):
        """启动监控任务"""
        def monitor_task():
            while True:
                try:
                    self._collect_metrics()
                    self._check_health()
                    time.sleep(self.config.monitoring_interval)
                except Exception as e:
                    logger.error(f"Monitoring task error: {e}")
        
        monitor_thread = threading.Thread(target=monitor_task, daemon=True)
        monitor_thread.start()
    
    def _collect_metrics(self):
        """收集性能指标"""
        with self.lock:
            # 更新内存使用量
            self.metrics.memory_usage = sum(entry.size for entry in self.cache.values())
            
            # 统计过期条目
            expired_count = sum(1 for entry in self.cache.values() if entry.is_expired())
            self.metrics.expired_entries = expired_count
            
            # 清理过期条目
            expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]
            for key in expired_keys:
                del self.cache[key]
            
            # 更新负载因子
            if self.config.max_size > 0:
                self.metrics.load_factor = len(self.cache) / self.config.max_size
    
    def _check_health(self):
        """健康检查"""
        if self.distributed_enabled:
            for node_id, node in self.distributed_nodes.items():
                if not node.is_healthy_check():
                    logger.warning(f"Node {node_id} is unhealthy")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取缓存值
        
        Args:
            key: 缓存键
            default: 默认值
            
        Returns:
            缓存值或默认值
        """
        start_time = time.time()
        
        try:
            # 安全检查
            if not self.security_manager.validate_key(key):
                logger.warning(f"Invalid cache key: {key}")
                return default
            
            # 分布式缓存处理
            if self.distributed_enabled and self.load_balancer:
                node = self.load_balancer.select_node(key)
                result = self.distributed_nodes[node].get(key)
                if result is not None:
                    self.metrics.hit_count += 1
                    self.metrics.total_requests += 1
                    return result
                self.metrics.miss_count += 1
                self.metrics.total_requests += 1
                return default
            
            # 本地缓存处理
            with self.lock:
                entry = self.cache.get(key)
                if entry is None:
                    self.metrics.miss_count += 1
                    self.metrics.total_requests += 1
                    return default
                
                if entry.is_expired():
                    del self.cache[key]
                    self.metrics.expired_entries += 1
                    self.metrics.miss_count += 1
                    self.metrics.total_requests += 1
                    return default
                
                entry.update_access()
                self.metrics.hit_count += 1
                self.metrics.total_requests += 1
                
                latency = time.time() - start_time
                self.metrics.update_latency(latency)
                
                # 解压缩
                if entry.compressed:
                    return self._decompress(entry.value)
                return entry.value
                
        except Exception as e:
            logger.error(f"Error getting cache key {key}: {e}")
            self.metrics.miss_count += 1
            self.metrics.total_requests += 1
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            version: str = "1.0", tags: Optional[Set[str]] = None) -> bool:
        """
        设置缓存值
        
        Args:
            key: 缓存键
            value: 缓存值
            ttl: 生存时间（秒）
            version: 版本号
            tags: 标签集合
            
        Returns:
            是否设置成功
        """
        try:
            # 安全检查
            if not self.security_manager.validate_key(key):
                logger.warning(f"Invalid cache key: {key}")
                return False
            
            if not self.security_manager.validate_value(value):
                logger.warning(f"Invalid cache value for key: {key}")
                return False
            
            # 分布式缓存处理
            if self.distributed_enabled and self.load_balancer:
                node = self.load_balancer.select_node(key)
                success = self.distributed_nodes[node].set(key, value, ttl, version, tags)
                if success:
                    self.load_balancer.update_node_load(node, 1)
                return success
            
            # 本地缓存处理
            with self.lock:
                # 检查缓存大小限制
                if len(self.cache) >= self.config.max_size:
                    self._evict_entries(1)
                
                # 压缩
                compressed_value = value
                compressed = False
                if self.config.enable_compression:
                    compressed_value, compressed = self._compress(value)
                
                # 计算大小
                size = len(pickle.dumps(compressed_value))
                
                entry = CacheEntry(
                    key=key,
                    value=compressed_value,
                    ttl=ttl or self.config.default_ttl,
                    version=version,
                    compressed=compressed,
                    size=size,
                    tags=tags or set()
                )
                
                self.cache[key] = entry
                self._update_memory_usage()
                
                # 异步预热相关键
                if self.prefetch_enabled:
                    asyncio.create_task(self._prefetch_related_keys(key))
                
                return True
                
        except Exception as e:
            logger.error(f"Error setting cache key {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """
        删除缓存值
        
        Args:
            key: 缓存键
            
        Returns:
            是否删除成功
        """
        try:
            # 分布式缓存处理
            if self.distributed_enabled and self.load_balancer:
                node = self.load_balancer.select_node(key)
                return self.distributed_nodes[node].delete(key)
            
            # 本地缓存处理
            with self.lock:
                if key in self.cache:
                    del self.cache[key]
                    self._update_memory_usage()
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        检查缓存键是否存在
        
        Args:
            key: 缓存键
            
        Returns:
            是否存在
        """
        try:
            # 分布式缓存处理
            if self.distributed_enabled and self.load_balancer:
                node = self.load_balancer.select_node(key)
                return self.distributed_nodes[node].get(key) is not None
            
            # 本地缓存处理
            with self.lock:
                entry = self.cache.get(key)
                if entry is None or entry.is_expired():
                    return False
                return True
                
        except Exception as e:
            logger.error(f"Error checking cache key {key}: {e}")
            return False
    
    def clear(self):
        """清空所有缓存"""
        try:
            with self.lock:
                self.cache.clear()
                self.metrics = CacheMetrics()
                
                if self.distributed_enabled:
                    for node in self.distributed_nodes.values():
                        node.clear()
                
                logger.info("Cache cleared")
                
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def get_many(self, keys: List[str]) -> Dict[str, Any]:
        """
        批量获取缓存值
        
        Args:
            keys: 缓存键列表
            
        Returns:
            键值对字典
        """
        result = {}
        for key in keys:
            result[key] = self.get(key)
        return result
    
    def set_many(self, data: Dict[str, Any], ttl: Optional[int] = None) -> Dict[str, bool]:
        """
        批量设置缓存值
        
        Args:
            data: 键值对字典
            ttl: 生存时间
            
        Returns:
            设置结果字典
        """
        result = {}
        for key, value in data.items():
            result[key] = self.set(key, value, ttl)
        return result
    
    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取缓存值（优化版本）"""
        if not self.batch_loading_enabled:
            return self.get_many(keys)
        
        # 如果启用批量加载，尝试优化查询
        result = {}
        
        # 分布式缓存批量处理
        if self.distributed_enabled and self.load_balancer:
            # 按节点分组键
            node_keys = defaultdict(list)
            for key in keys:
                if self.security_manager.validate_key(key):
                    node = self.load_balancer.select_node(key)
                    node_keys[node].append(key)
            
            # 并行从各节点获取
            with ThreadPoolExecutor(max_workers=len(node_keys)) as executor:
                futures = {}
                for node_id, key_list in node_keys.items():
                    future = executor.submit(
                        self._batch_get_from_node, node_id, key_list
                    )
                    futures[future] = node_id
                
                for future in as_completed(futures):
                    node_id = futures[future]
                    try:
                        node_result = future.result()
                        result.update(node_result)
                    except Exception as e:
                        logger.error(f"Batch get failed from node {node_id}: {e}")
        else:
            # 本地缓存批量处理
            with self.lock:
                for key in keys:
                    if key in self.cache:
                        entry = self.cache[key]
                        if not entry.is_expired():
                            if entry.compressed:
                                result[key] = self._decompress(entry.value)
                            else:
                                result[key] = entry.value
        
        return result
    
    def batch_set(self, data: Dict[str, Any], ttl: Optional[int] = None) -> Dict[str, bool]:
        """批量设置缓存值（优化版本）"""
        if not self.batch_loading_enabled:
            return self.set_many(data, ttl)
        
        # 分布式缓存批量处理
        if self.distributed_enabled and self.load_balancer:
            # 按节点分组键
            node_data = defaultdict(dict)
            for key, value in data.items():
                if self.security_manager.validate_key(key) and self.security_manager.validate_value(value):
                    node = self.load_balancer.select_node(key)
                    node_data[node][key] = value
            
            # 并行向各节点设置
            with ThreadPoolExecutor(max_workers=len(node_data)) as executor:
                futures = {}
                for node_id, node_batch_data in node_data.items():
                    future = executor.submit(
                        self._batch_set_to_node, node_id, node_batch_data, ttl
                    )
                    futures[future] = node_id
                
                result = {}
                for future in as_completed(futures):
                    node_id = futures[future]
                    try:
                        node_result = future.result()
                        result.update(node_result)
                    except Exception as e:
                        logger.error(f"Batch set failed to node {node_id}: {e}")
                        # 标记该节点的所有键为失败
                        for key in node_data[node_id]:
                            result[key] = False
        else:
            # 本地缓存批量处理
            result = {}
            with self.lock:
                # 检查是否有足够空间
                current_size = len(self.cache)
                available_space = max(0, self.config.max_size - current_size)
                
                if available_space < len(data):
                    # 需要淘汰一些条目
                    evict_count = len(data) - available_space
                    self._evict_entries(evict_count)
                
                for key, value in data.items():
                    if self.security_manager.validate_key(key) and self.security_manager.validate_value(value):
                        result[key] = self.set(key, value, ttl)
                    else:
                        result[key] = False
        
        return result
    
    def _batch_get_from_node(self, node_id: str, keys: List[str]) -> Dict[str, Any]:
        """从指定节点批量获取"""
        result = {}
        node = self.distributed_nodes[node_id]
        
        for key in keys:
            value = node.get(key)
            if value is not None:
                result[key] = value
        
        return result
    
    def _batch_set_to_node(self, node_id: str, data: Dict[str, Any], 
                          ttl: Optional[int] = None) -> Dict[str, bool]:
        """向指定节点批量设置"""
        result = {}
        node = self.distributed_nodes[node_id]
        
        for key, value in data.items():
            result[key] = node.set(key, value, ttl)
        
        return result
    
    def preload(self, keys: List[str], loader_func: Callable[[str], Any]) -> Dict[str, Any]:
        """
        预加载缓存
        
        Args:
            keys: 要预加载的键列表
            loader_func: 数据加载函数
            
        Returns:
            预加载结果字典
        """
        result = {}
        
        # 找出未缓存的键
        missing_keys = []
        for key in keys:
            if not self.exists(key):
                missing_keys.append(key)
        
        if not missing_keys:
            logger.info("All keys already cached")
            return {key: self.get(key) for key in keys}
        
        try:
            # 批量加载数据
            if self.batch_loading_enabled:
                # 尝试批量加载
                loaded_data = self._batch_load_data(missing_keys, loader_func)
            else:
                # 单个加载
                loaded_data = {}
                for key in missing_keys:
                    try:
                        loaded_data[key] = loader_func(key)
                    except Exception as e:
                        logger.error(f"Failed to load data for key {key}: {e}")
            
            # 批量设置到缓存
            set_results = self.batch_set(loaded_data)
            
            # 合并结果
            for key in keys:
                if key in loaded_data and set_results.get(key, False):
                    result[key] = loaded_data[key]
                else:
                    result[key] = self.get(key)
            
            logger.info(f"Preloaded {len(result)} keys")
            
        except Exception as e:
            logger.error(f"Preload failed: {e}")
            # 回退到逐个加载
            for key in keys:
                result[key] = self.get(key)
        
        return result
    
    def _batch_load_data(self, keys: List[str], loader_func: Callable[[str], Any]) -> Dict[str, Any]:
        """批量加载数据"""
        loaded_data = {}
        
        # 尝试批量加载
        try:
            # 如果加载函数支持批量
            if hasattr(loader_func, '__code__') and loader_func.__code__.co_argcount > 1:
                # 批量加载
                batch_result = loader_func(keys)
                if isinstance(batch_result, dict):
                    loaded_data.update(batch_result)
                else:
                    # 如果返回的不是字典，尝试映射
                    for i, key in enumerate(keys):
                        if i < len(batch_result):
                            loaded_data[key] = batch_result[i]
            else:
                # 单个加载
                for key in keys:
                    try:
                        loaded_data[key] = loader_func(key)
                    except Exception as e:
                        logger.error(f"Failed to load data for key {key}: {e}")
        except Exception as e:
            logger.error(f"Batch loading failed, falling back to individual loading: {e}")
            # 回退到单个加载
            for key in keys:
                try:
                    loaded_data[key] = loader_func(key)
                except Exception as e:
                    logger.error(f"Failed to load data for key {key}: {e}")
        
        return loaded_data
    
    async def _prefetch_related_keys(self, key: str):
        """预取相关键"""
        if not self.prefetch_enabled:
            return
        
        try:
            # 根据键的模式预取相关键
            related_keys = self._generate_related_keys(key)
            
            # 只预取少量键，避免过度预取
            prefetch_keys = related_keys[:self.config.prefetch_count]
            
            # 异步预取
            await self.async_processor.async_preload(
                self, prefetch_keys, lambda k: self._default_loader(k)
            )
            
        except Exception as e:
            logger.error(f"Prefetch failed for key {key}: {e}")
    
    def _generate_related_keys(self, key: str) -> List[str]:
        """生成相关键"""
        related_keys = []
        
        # 基于键的前缀生成相关键
        if ':' in key:
            prefix = key.rsplit(':', 1)[0]
            related_keys.append(f"{prefix}:*")
        
        # 基于键的后缀生成相关键
        if ':' in key:
            suffix = key.rsplit(':', 1)[1]
            related_keys.append(f"*:{suffix}")
        
        # 添加一些常见的相关模式
        patterns = [
            f"{key}:metadata",
            f"{key}:stats",
            f"{key}:config",
            f"related:{key}",
            f"parent:{key}",
            f"children:{key}"
        ]
        
        related_keys.extend(patterns)
        
        return related_keys
    
    def _default_loader(self, key: str) -> Any:
        """默认数据加载器"""
        # 这是一个示例加载器，实际应用中应该根据具体需求实现
        return f"default_value_for_{key}"
    
    def invalidate_by_tag(self, tag: str) -> int:
        """
        根据标签失效缓存
        
        Args:
            tag: 标签
            
        Returns:
            失效的条目数量
        """
        count = 0
        try:
            with self.lock:
                keys_to_delete = []
                for key, entry in self.cache.items():
                    if tag in entry.tags:
                        keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    del self.cache[key]
                    count += 1
                
                self._update_memory_usage()
            
            logger.info(f"Invalidated {count} entries with tag: {tag}")
            
        except Exception as e:
            logger.error(f"Error invalidating by tag {tag}: {e}")
        
        return count
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """
        根据模式失效缓存
        
        Args:
            pattern: 匹配模式（支持*通配符）
            
        Returns:
            失效的条目数量
        """
        count = 0
        try:
            # 转换模式为正则表达式
            import re
            regex_pattern = pattern.replace('*', '.*')
            compiled_pattern = re.compile(regex_pattern)
            
            with self.lock:
                keys_to_delete = []
                for key in self.cache.keys():
                    if compiled_pattern.match(key):
                        keys_to_delete.append(key)
                
                for key in keys_to_delete:
                    del self.cache[key]
                    count += 1
                
                self._update_memory_usage()
            
            logger.info(f"Invalidated {count} entries matching pattern: {pattern}")
            
        except Exception as e:
            logger.error(f"Error invalidating by pattern {pattern}: {e}")
        
        return count
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        with self.lock:
            stats = {
                'total_entries': len(self.cache),
                'hit_ratio': self.metrics.hit_ratio,
                'miss_ratio': self.metrics.miss_ratio,
                'total_requests': self.metrics.total_requests,
                'hit_count': self.metrics.hit_count,
                'miss_count': self.metrics.miss_count,
                'evictions': self.metrics.evictions,
                'expired_entries': self.metrics.expired_entries,
                'memory_usage': self.metrics.memory_usage,
                'load_factor': self.metrics.load_factor,
                'avg_latency': self.metrics.avg_latency,
                'compression_ratio': self.metrics.compression_ratio,
                'strategy': self.config.strategy.value,
                'max_size': self.config.max_size,
                'max_memory': self.config.max_memory,
                'distributed_enabled': self.distributed_enabled,
                'security_level': self.config.security_level.value
            }
            
            # 添加分布式节点信息
            if self.distributed_enabled:
                stats['distributed_nodes'] = {
                    node_id: {
                        'total_entries': len(node.cache),
                        'memory_usage': node.metrics.memory_usage,
                        'hit_ratio': node.metrics.hit_ratio,
                        'is_healthy': node.is_healthy_check()
                    }
                    for node_id, node in self.distributed_nodes.items()
                }
            
            return stats
    
    def optimize(self):
        """优化缓存性能"""
        try:
            logger.info("Starting cache optimization...")
            
            # 清理过期条目
            self._cleanup_expired_entries()
            
            # 重新计算压缩比
            self._recalculate_compression_ratio()
            
            # 调整缓存策略参数
            self._adjust_strategy_parameters()
            
            # 负载均衡优化
            if self.distributed_enabled:
                self._optimize_load_balancing()
            
            logger.info("Cache optimization completed")
            
        except Exception as e:
            logger.error(f"Cache optimization failed: {e}")
    
    def _cleanup_expired_entries(self):
        """清理过期条目"""
        with self.lock:
            expired_keys = [key for key, entry in self.cache.items() if entry.is_expired()]
            for key in expired_keys:
                del self.cache[key]
            
            logger.info(f"Cleaned up {len(expired_keys)} expired entries")
    
    def _recalculate_compression_ratio(self):
        """重新计算压缩比"""
        if not self.config.enable_compression:
            return
        
        total_original_size = 0
        total_compressed_size = 0
        
        with self.lock:
            for entry in self.cache.values():
                if entry.compressed:
                    original_size = len(pickle.dumps(self._decompress(entry.value)))
                    compressed_size = entry.size
                    total_original_size += original_size
                    total_compressed_size += compressed_size
        
        if total_original_size > 0:
            self.metrics.compression_ratio = total_compressed_size / total_original_size
    
    def _adjust_strategy_parameters(self):
        """调整策略参数"""
        if isinstance(self.policy, AdaptiveCachePolicy):
            # 更新自适应策略的历史数据
            self.policy.hit_ratio_history.append(self.metrics.hit_ratio)
            
            # 如果命中率过低，增加预取数量
            if self.metrics.hit_ratio < 0.7:
                self.config.prefetch_count = min(20, self.config.prefetch_count + 2)
            elif self.metrics.hit_ratio > 0.9:
                self.config.prefetch_count = max(5, self.config.prefetch_count - 1)
    
    def _optimize_load_balancing(self):
        """优化负载均衡"""
        if not self.load_balancer:
            return
        
        # 重新计算节点权重
        for node_id, node in self.distributed_nodes.items():
            metrics = node.get_metrics()
            # 基于命中率调整权重
            if metrics.hit_ratio > 0.9:
                self.load_balancer.node_weights[node_id] = min(2.0, 
                    self.load_balancer.node_weights[node_id] + 0.1)
            elif metrics.hit_ratio < 0.5:
                self.load_balancer.node_weights[node_id] = max(0.5,
                    self.load_balancer.node_weights[node_id] - 0.1)
    
    def _evict_entries(self, count: int):
        """淘汰缓存条目"""
        if not self.cache or count <= 0:
            return
        
        # 选择淘汰候选
        candidates = self.policy.select_eviction_candidates(self.cache, count)
        
        # 执行淘汰
        for key in candidates:
            if key in self.cache:
                del self.cache[key]
                self.metrics.evictions += 1
        
        self._update_memory_usage()
    
    def _update_memory_usage(self):
        """更新内存使用量"""
        self.metrics.memory_usage = sum(entry.size for entry in self.cache.values())
    
    def _compress(self, data: Any) -> Tuple[Any, bool]:
        """压缩数据"""
        try:
            serialized = pickle.dumps(data)
            compressed = zlib.compress(serialized, self.config.compression_level)
            return compressed, True
        except Exception:
            return data, False
    
    def _decompress(self, data: Any) -> Any:
        """解压缩数据"""
        try:
            decompressed = zlib.decompress(data)
            return pickle.loads(decompressed)
        except Exception:
            return data
    
    def export_cache(self, file_path: str) -> bool:
        """
        导出缓存数据
        
        Args:
            file_path: 导出文件路径
            
        Returns:
            是否导出成功
        """
        try:
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'strategy': self.config.strategy.value,
                    'max_size': self.config.max_size,
                    'max_memory': self.config.max_memory,
                    'default_ttl': self.config.default_ttl
                },
                'metrics': {
                    'hit_count': self.metrics.hit_count,
                    'miss_count': self.metrics.miss_count,
                    'total_requests': self.metrics.total_requests,
                    'evictions': self.metrics.evictions
                },
                'cache_data': {}
            }
            
            with self.lock:
                for key, entry in self.cache.items():
                    export_data['cache_data'][key] = {
                        'value': entry.value,
                        'created_time': entry.created_time.isoformat(),
                        'last_accessed': entry.last_accessed.isoformat(),
                        'access_count': entry.access_count,
                        'ttl': entry.ttl,
                        'version': entry.version,
                        'compressed': entry.compressed,
                        'size': entry.size,
                        'tags': list(entry.tags)
                    }
            
            # 保存到文件
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Cache exported to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export cache: {e}")
            return False
    
    def import_cache(self, file_path: str, merge: bool = True) -> bool:
        """
        导入缓存数据
        
        Args:
            file_path: 导入文件路径
            merge: 是否合并现有数据
            
        Returns:
            是否导入成功
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            cache_data = import_data.get('cache_data', {})
            
            if not merge:
                self.clear()
            
            imported_count = 0
            with self.lock:
                for key, entry_data in cache_data.items():
                    try:
                        # 验证键
                        if not self.security_manager.validate_key(key):
                            continue
                        
                        # 重建缓存条目
                        entry = CacheEntry(
                            key=key,
                            value=entry_data['value'],
                            created_time=datetime.fromisoformat(entry_data['created_time']),
                            last_accessed=datetime.fromisoformat(entry_data['last_accessed']),
                            access_count=entry_data['access_count'],
                            ttl=entry_data['ttl'],
                            version=entry_data['version'],
                            compressed=entry_data['compressed'],
                            size=entry_data['size'],
                            tags=set(entry_data['tags'])
                        )
                        
                        self.cache[key] = entry
                        imported_count += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to import entry {key}: {e}")
            
            self._update_memory_usage()
            logger.info(f"Imported {imported_count} cache entries from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to import cache: {e}")
            return False
    
    def add_node(self, host: str, port: int) -> str:
        """
        添加分布式缓存节点
        
        Args:
            host: 主机地址
            port: 端口号
            
        Returns:
            节点ID
        """
        if not self.distributed_enabled:
            logger.warning("Distributed cache not enabled")
            return ""
        
        try:
            node_id = f"node_{len(self.distributed_nodes)}"
            node = DistributedCacheNode(node_id, host, port, self.config)
            
            self.distributed_nodes[node_id] = node
            
            # 更新负载均衡器
            if self.load_balancer:
                self.load_balancer.nodes.append(node_id)
                self.load_balancer.node_weights[node_id] = 1.0
                self.load_balancer.node_loads[node_id] = 0
                self.load_balancer.node_failures[node_id] = 0
            
            logger.info(f"Added cache node: {node_id} ({host}:{port})")
            return node_id
            
        except Exception as e:
            logger.error(f"Failed to add cache node: {e}")
            return ""
    
    def remove_node(self, node_id: str) -> bool:
        """
        移除分布式缓存节点
        
        Args:
            node_id: 节点ID
            
        Returns:
            是否移除成功
        """
        if not self.distributed_enabled or node_id not in self.distributed_nodes:
            return False
        
        try:
            del self.distributed_nodes[node_id]
            
            # 更新负载均衡器
            if self.load_balancer:
                if node_id in self.load_balancer.nodes:
                    self.load_balancer.nodes.remove(node_id)
                self.load_balancer.node_weights.pop(node_id, None)
                self.load_balancer.node_loads.pop(node_id, None)
                self.load_balancer.node_failures.pop(node_id, None)
            
            logger.info(f"Removed cache node: {node_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to remove cache node: {e}")
            return False
    
    def get_node_stats(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        获取节点统计信息
        
        Args:
            node_id: 节点ID
            
        Returns:
            节点统计信息
        """
        if node_id not in self.distributed_nodes:
            return None
        
        try:
            node = self.distributed_nodes[node_id]
            metrics = node.get_metrics()
            
            return {
                'node_id': node_id,
                'host': node.host,
                'port': node.port,
                'is_healthy': node.is_healthy_check(),
                'total_entries': len(node.cache),
                'memory_usage': metrics.memory_usage,
                'hit_count': metrics.hit_count,
                'miss_count': metrics.miss_count,
                'hit_ratio': metrics.hit_ratio,
                'evictions': metrics.evictions,
                'avg_latency': metrics.avg_latency
            }
            
        except Exception as e:
            logger.error(f"Failed to get node stats for {node_id}: {e}")
            return None
    
    def shutdown(self):
        """关闭缓存优化器"""
        try:
            logger.info("Shutting down CacheOptimizer...")
            
            # 停止监控
            if self.monitoring_task:
                self.monitoring_task = None
            
            # 关闭异步处理器
            self.async_processor.shutdown()
            
            # 保存缓存（如果启用持久化）
            if self.config.enable_persistence and self.config.persistence_path:
                self.export_cache(self.config.persistence_path)
            
            logger.info("CacheOptimizer shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# FIFO缓存策略实现
class FIFO(CachePolicy):
    """FIFO缓存策略"""
    
    def should_evict(self, entry: CacheEntry) -> bool:
        return False  # FIFO通过创建时间判断
    
    def select_eviction_candidates(self, entries: Dict[str, CacheEntry], count: int) -> List[str]:
        # 按创建时间排序，选择最早的
        sorted_entries = sorted(entries.items(), key=lambda x: x[1].created_time)
        return [key for key, _ in sorted_entries[:count]]

# 使用示例和测试代码
def example_usage():
    """使用示例"""
    
    # 1. 基本使用
    print("=== 基本使用示例 ===")
    
    config = CacheConfig(
        max_size=1000,
        max_memory=50 * 1024 * 1024,  # 50MB
        default_ttl=1800,  # 30分钟
        strategy=CacheStrategy.LRU,
        enable_compression=True,
        enable_monitoring=True
    )
    
    cache = CacheOptimizer(config)
    
    # 设置缓存
    cache.set("user:123", {"name": "张三", "age": 25})
    cache.set("product:456", {"name": "iPhone 15", "price": 7999}, ttl=3600)
    
    # 获取缓存
    user = cache.get("user:123")
    product = cache.get("product:456")
    
    print(f"用户信息: {user}")
    print(f"产品信息: {product}")
    
    # 2. 批量操作
    print("\n=== 批量操作示例 ===")
    
    user_data = {
        "user:124": {"name": "李四", "age": 30},
        "user:125": {"name": "王五", "age": 28},
        "user:126": {"name": "赵六", "age": 35}
    }
    
    # 批量设置
    results = cache.batch_set(user_data, ttl=3600)
    print(f"批量设置结果: {results}")
    
    # 批量获取
    users = cache.batch_get(list(user_data.keys()))
    print(f"批量获取用户: {users}")
    
    # 3. 分布式缓存
    print("\n=== 分布式缓存示例 ===")
    
    distributed_config = CacheConfig(
        max_size=500,
        enable_distributed=True,
        nodes=["localhost:6379", "localhost:6380"],
        replication_factor=2,
        consistency_level=ConsistencyLevel.EVENTUAL
    )
    
    distributed_cache = CacheOptimizer(distributed_config)
    
    # 添加节点
    node_id = distributed_cache.add_node("localhost", 6379)
    print(f"添加节点: {node_id}")
    
    # 分布式缓存操作
    distributed_cache.set("session:abc123", {"user_id": 123, "login_time": "2024-01-01"})
    session = distributed_cache.get("session:abc123")
    print(f"分布式会话信息: {session}")
    
    # 4. 异步操作
    print("\n=== 异步操作示例 ===")
    
    async def async_example():
        async_processor = AsyncCacheProcessor()
        
        # 异步设置
        await async_processor.async_set(cache, "async_key", "async_value", ttl=300)
        
        # 异步获取
        value = await async_processor.async_get(cache, "async_key")
        print(f"异步获取的值: {value}")
        
        # 异步批量操作
        await async_processor.async_batch_set(cache, {"key1": "value1", "key2": "value2"})
        batch_result = await async_processor.async_batch_get(cache, ["key1", "key2"])
        print(f"异步批量结果: {batch_result}")
        
        async_processor.shutdown()
    
    # 运行异步示例
    # asyncio.run(async_example())
    
    # 5. 缓存预热
    print("\n=== 缓存预热示例 ===")
    
    def data_loader(key: str):
        # 模拟数据加载
        time.sleep(0.1)  # 模拟数据库查询延迟
        return f"loaded_data_for_{key}"
    
    # 预热缓存
    keys_to_preload = ["warm:1", "warm:2", "warm:3"]
    preloaded_data = cache.preload(keys_to_preload, data_loader)
    print(f"预热的数据: {preloaded_data}")
    
    # 6. 缓存失效
    print("\n=== 缓存失效示例 ===")
    
    # 设置带标签的缓存
    cache.set("article:1", {"title": "缓存优化技术", "author": "张三"}, tags={"tech", "cache"})
    cache.set("article:2", {"title": "分布式系统设计", "author": "李四"}, tags={"tech", "distributed"})
    
    # 根据标签失效
    invalidated_count = cache.invalidate_by_tag("tech")
    print(f"失效的条目数量: {invalidated_count}")
    
    # 根据模式失效
    cache.set("temp:1", "temporary1")
    cache.set("temp:2", "temporary2")
    cache.set("permanent:1", "permanent1")
    
    pattern_invalidated = cache.invalidate_by_pattern("temp:*")
    print(f"模式失效的条目数量: {pattern_invalidated}")
    
    # 7. 性能监控
    print("\n=== 性能监控示例 ===")
    
    # 执行一些操作
    for i in range(100):
        cache.set(f"perf:test:{i}", f"value_{i}")
        cache.get(f"perf:test:{i}")
    
    # 获取统计信息
    stats = cache.get_stats()
    print(f"缓存统计信息:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # 8. 缓存优化
    print("\n=== 缓存优化示例 ===")
    
    cache.optimize()
    print("缓存优化完成")
    
    # 9. 数据导入导出
    print("\n=== 数据导入导出示例 ===")
    
    # 导出缓存
    export_success = cache.export_cache("cache_export.json")
    print(f"缓存导出: {'成功' if export_success else '失败'}")
    
    # 清空缓存
    cache.clear()
    print("缓存已清空")
    
    # 导入缓存
    import_success = cache.import_cache("cache_export.json")
    print(f"缓存导入: {'成功' if import_success else '失败'}")
    
    # 10. 安全功能
    print("\n=== 安全功能示例 ===")
    
    # 添加被阻止的键
    cache.security_manager.add_blocked_key("blocked_key")
    
    # 尝试设置被阻止的键
    blocked_result = cache.set("blocked_key", "should_not_be_set")
    print(f"被阻止键设置结果: {blocked_result}")
    
    # 添加可疑模式
    cache.security_manager.add_suspicious_pattern("admin:")
    
    # 尝试设置包含可疑模式的键
    suspicious_result = cache.set("admin:secret", "secret_data")
    print(f"可疑模式键设置结果: {suspicious_result}")
    
    print("\n=== 示例完成 ===")

def performance_test():
    """性能测试"""
    print("=== 性能测试 ===")
    
    config = CacheConfig(
        max_size=10000,
        max_memory=100 * 1024 * 1024,
        strategy=CacheStrategy.LRU,
        enable_compression=True
    )
    
    cache = CacheOptimizer(config)
    
    # 测试数据
    test_data = {f"key_{i}": f"value_{i}" * 10 for i in range(5000)}
    
    # 批量写入测试
    start_time = time.time()
    results = cache.batch_set(test_data)
    write_time = time.time() - start_time
    write_throughput = len(test_data) / write_time
    
    print(f"批量写入: {len(test_data)} 条记录")
    print(f"写入时间: {write_time:.3f} 秒")
    print(f"写入吞吐量: {write_throughput:.0f} ops/sec")
    
    # 批量读取测试
    start_time = time.time()
    results = cache.batch_get(list(test_data.keys()))
    read_time = time.time() - start_time
    read_throughput = len(test_data) / read_time
    
    print(f"批量读取: {len(test_data)} 条记录")
    print(f"读取时间: {read_time:.3f} 秒")
    print(f"读取吞吐量: {read_throughput:.0f} ops/sec")
    
    # 随机访问测试
    import random
    random_keys = random.sample(list(test_data.keys()), 1000)
    
    start_time = time.time()
    for key in random_keys:
        cache.get(key)
    random_access_time = time.time() - start_time
    random_access_throughput = 1000 / random_access_time
    
    print(f"随机访问: 1000 次访问")
    print(f"访问时间: {random_access_time:.3f} 秒")
    print(f"访问吞吐量: {random_access_throughput:.0f} ops/sec")
    
    # 命中率测试
    hit_count = 0
    total_requests = 2000
    
    for i in range(total_requests):
        if i < 1000:
            # 前1000次访问已缓存的键
            key = random.choice(list(test_data.keys()))
        else:
            # 后1000次访问未缓存的键
            key = f"non_existent_key_{i}"
        
        result = cache.get(key)
        if result is not None:
            hit_count += 1
    
    hit_ratio = hit_count / total_requests
    print(f"命中率: {hit_ratio:.3f} ({hit_count}/{total_requests})")
    
    # 最终统计
    final_stats = cache.get_stats()
    print(f"\n最终统计信息:")
    print(f"缓存条目数: {final_stats['total_entries']}")
    print(f"内存使用: {final_stats['memory_usage'] / 1024 / 1024:.2f} MB")
    print(f"总请求数: {final_stats['total_requests']}")
    print(f"淘汰次数: {final_stats['evictions']}")

if __name__ == "__main__":
    # 运行示例
    example_usage()
    
    print("\n" + "="*50 + "\n")
    
    # 运行性能测试
    performance_test()