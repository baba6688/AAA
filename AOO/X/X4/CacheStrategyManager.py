"""
X4缓存策略管理器
================

提供完整的缓存策略管理功能，包括策略配置、切换、优化、监控等功能。

Author: X4 Cache Strategy Manager
Date: 2025-11-06
"""

import time
import threading
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from collections import OrderedDict
from enum import Enum
import statistics
import weakref


class CacheStrategy(Enum):
    """缓存策略枚举"""
    LRU = "lru"           # 最近最少使用
    LFU = "lfu"           # 最少使用频率
    FIFO = "fifo"         # 先进先出
    TTL = "ttl"           # 基于时间过期
    ARC = "arc"           # 自适应缓存替换
    CAR = "car"           # 循环替换算法


class StrategyPerformance(Enum):
    """策略性能等级"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


@dataclass
class CacheConfig:
    """缓存配置"""
    max_size: int = 1000
    ttl: int = 3600  # 默认1小时
    cleanup_interval: int = 300  # 清理间隔5分钟
    enable_monitoring: bool = True
    enable_optimization: bool = True


@dataclass
class StrategyMetrics:
    """策略性能指标"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    hit_rate: float = 0.0
    avg_response_time: float = 0.0
    memory_usage: int = 0
    timestamp: float = 0.0


class CacheStrategyBase(ABC):
    """缓存策略基类"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.metrics = StrategyMetrics()
        self._lock = threading.RLock()
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any) -> None:
        """放入缓存项"""
        pass
    
    @abstractmethod
    def remove(self, key: str) -> bool:
        """移除缓存项"""
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """清空缓存"""
        pass
    
    def get_metrics(self) -> StrategyMetrics:
        """获取性能指标"""
        return self.metrics
    
    def _update_hit_rate(self) -> None:
        """更新命中率"""
        total = self.metrics.hits + self.metrics.misses
        if total > 0:
            self.metrics.hit_rate = self.metrics.hits / total
    
    def _update_response_time(self, response_time: float) -> None:
        """更新响应时间"""
        if self.metrics.avg_response_time == 0:
            self.metrics.avg_response_time = response_time
        else:
            self.metrics.avg_response_time = (self.metrics.avg_response_time + response_time) / 2


class LRUStrategy(CacheStrategyBase):
    """最近最少使用策略"""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self._cache = OrderedDict()
        self._timestamps = {}
    
    def get(self, key: str) -> Optional[Any]:
        start_time = time.time()
        
        with self._lock:
            if key in self._cache:
                # 移动到末尾（最近使用）
                value = self._cache.pop(key)
                self._cache[key] = value
                self._timestamps[key] = time.time()
                
                self.metrics.hits += 1
                self._update_hit_rate()
                response_time = time.time() - start_time
                self._update_response_time(response_time)
                return value
            else:
                self.metrics.misses += 1
                self._update_hit_rate()
                return None
    
    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                # 更新现有项
                self._cache[key] = value
                self._timestamps[key] = time.time()
            else:
                # 添加新项
                if len(self._cache) >= self.config.max_size:
                    # 移除最久未使用的项（最早的时间戳）
                    if self._timestamps:
                        oldest_key = min(self._timestamps.keys(), key=lambda k: self._timestamps[k])
                        self._cache.pop(oldest_key)
                        self._timestamps.pop(oldest_key)
                        self.metrics.evictions += 1
                
                self._cache[key] = value
                self._timestamps[key] = time.time()
    
    def remove(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
                self._timestamps.pop(key)
                return True
            return False
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._timestamps.clear()
    
    def get_lru_order(self) -> List[str]:
        """获取LRU顺序"""
        return list(self._cache.keys())


class LFUStrategy(CacheStrategyBase):
    """最少使用频率策略"""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self._cache = {}
        self._frequencies = {}
        self._access_times = {}
    
    def get(self, key: str) -> Optional[Any]:
        start_time = time.time()
        
        with self._lock:
            if key in self._cache:
                # 增加使用频率
                self._frequencies[key] = self._frequencies.get(key, 0) + 1
                self._access_times[key] = time.time()
                
                self.metrics.hits += 1
                self._update_hit_rate()
                response_time = time.time() - start_time
                self._update_response_time(response_time)
                return self._cache[key]
            else:
                self.metrics.misses += 1
                self._update_hit_rate()
                return None
    
    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                # 更新现有项
                self._cache[key] = value
                # 更新访问时间
                self._access_times[key] = time.time()
            else:
                # 添加新项
                if len(self._cache) >= self.config.max_size:
                    # 移除使用频率最低的项（频率相同时按访问时间）
                    if self._frequencies:
                        lfu_key = min(self._frequencies.keys(), key=lambda k: (self._frequencies[k], self._access_times[k]))
                        self._cache.pop(lfu_key)
                        self._frequencies.pop(lfu_key)
                        self._access_times.pop(lfu_key)
                        self.metrics.evictions += 1
                
                self._cache[key] = value
                self._frequencies[key] = 0
                self._access_times[key] = time.time()
    
    def remove(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
                self._frequencies.pop(key)
                self._access_times.pop(key)
                return True
            return False
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._frequencies.clear()
            self._access_times.clear()
    
    def get_frequency_stats(self) -> Dict[str, int]:
        """获取频率统计"""
        return self._frequencies.copy()


class TTLStrategy(CacheStrategyBase):
    """基于时间的过期策略"""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self._cache = {}
        self._expiry_times = {}
        self._cleanup_thread = None
        self._running = False
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self) -> None:
        """启动清理线程"""
        if self._cleanup_thread is None:
            self._running = True
            self._cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
            self._cleanup_thread.start()
    
    def _cleanup_expired(self) -> None:
        """清理过期项"""
        while self._running:
            try:
                current_time = time.time()
                expired_keys = [
                    key for key, expiry in self._expiry_times.items()
                    if current_time > expiry
                ]
                
                with self._lock:
                    for key in expired_keys:
                        self._cache.pop(key, None)
                        self._expiry_times.pop(key, None)
                        self.metrics.evictions += 1
                
                time.sleep(self.config.cleanup_interval)
            except Exception as e:
                logging.error(f"清理过期项时发生错误: {e}")
                time.sleep(self.config.cleanup_interval)
    
    def get(self, key: str) -> Optional[Any]:
        start_time = time.time()
        
        with self._lock:
            if key in self._cache:
                # 检查是否过期
                if time.time() > self._expiry_times.get(key, 0):
                    # 已过期，移除
                    self._cache.pop(key)
                    self._expiry_times.pop(key)
                    self.metrics.misses += 1
                    self._update_hit_rate()
                    return None
                
                self.metrics.hits += 1
                self._update_hit_rate()
                response_time = time.time() - start_time
                self._update_response_time(response_time)
                return self._cache[key]
            else:
                self.metrics.misses += 1
                self._update_hit_rate()
                return None
    
    def put(self, key: str, value: Any) -> None:
        with self._lock:
            # 计算过期时间
            expiry_time = time.time() + self.config.ttl
            
            if key in self._cache:
                # 更新现有项
                self._cache[key] = value
                self._expiry_times[key] = expiry_time
            else:
                # 添加新项
                if len(self._cache) >= self.config.max_size:
                    # 移除最接近过期的项
                    if self._expiry_times:
                        oldest_key = min(self._expiry_times.keys(), key=lambda k: self._expiry_times[k])
                        self._cache.pop(oldest_key)
                        self._expiry_times.pop(oldest_key)
                        self.metrics.evictions += 1
                
                self._cache[key] = value
                self._expiry_times[key] = expiry_time
    
    def remove(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
                self._expiry_times.pop(key)
                return True
            return False
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._expiry_times.clear()
    
    def stop(self) -> None:
        """停止清理线程"""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=1)


class FIFOStrategy(CacheStrategyBase):
    """先进先出策略"""
    
    def __init__(self, config: CacheConfig):
        super().__init__(config)
        self._cache = {}
        self._queue = []
    
    def get(self, key: str) -> Optional[Any]:
        start_time = time.time()
        
        with self._lock:
            if key in self._cache:
                self.metrics.hits += 1
                self._update_hit_rate()
                response_time = time.time() - start_time
                self._update_response_time(response_time)
                return self._cache[key]
            else:
                self.metrics.misses += 1
                self._update_hit_rate()
                return None
    
    def put(self, key: str, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                # 更新现有项
                self._cache[key] = value
            else:
                # 添加新项
                if len(self._cache) >= self.config.max_size:
                    # 移除最早的项
                    if self._queue:
                        oldest_key = self._queue.pop(0)
                        self._cache.pop(oldest_key)
                        self.metrics.evictions += 1
                
                self._cache[key] = value
                self._queue.append(key)
    
    def remove(self, key: str) -> bool:
        with self._lock:
            if key in self._cache:
                self._cache.pop(key)
                if key in self._queue:
                    self._queue.remove(key)
                return True
            return False
    
    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._queue.clear()


class CacheStrategyManager:
    """缓存策略管理器主类"""
    
    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._strategies: Dict[CacheStrategy, CacheStrategyBase] = {}
        self._current_strategy = CacheStrategy.LRU
        self._strategy_history: List[Dict[str, Any]] = []
        self._optimization_callbacks: List[Callable] = []
        self._monitoring_enabled = True
        self._lock = threading.RLock()
        self._initialize_strategies()
    
    def _initialize_strategies(self) -> None:
        """初始化所有策略"""
        self._strategies = {
            CacheStrategy.LRU: LRUStrategy(self.config),
            CacheStrategy.LFU: LFUStrategy(self.config),
            CacheStrategy.TTL: TTLStrategy(self.config),
            CacheStrategy.FIFO: FIFOStrategy(self.config),
        }
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        strategy = self._get_current_strategy()
        return strategy.get(key)
    
    def put(self, key: str, value: Any) -> None:
        """放入缓存项"""
        strategy = self._get_current_strategy()
        strategy.put(key, value)
    
    def remove(self, key: str) -> bool:
        """移除缓存项"""
        strategy = self._get_current_strategy()
        return strategy.remove(key)
    
    def clear(self) -> None:
        """清空缓存"""
        strategy = self._get_current_strategy()
        strategy.clear()
    
    def switch_strategy(self, strategy: CacheStrategy) -> bool:
        """切换策略"""
        with self._lock:
            if strategy not in self._strategies:
                return False
            
            # 记录切换历史
            self._strategy_history.append({
                'from': self._current_strategy.value,
                'to': strategy.value,
                'timestamp': time.time(),
                'metrics': asdict(self._get_current_strategy().get_metrics())
            })
            
            self._current_strategy = strategy
            return True
    
    def _get_current_strategy(self) -> CacheStrategyBase:
        """获取当前策略"""
        return self._strategies[self._current_strategy]
    
    def get_available_strategies(self) -> List[CacheStrategy]:
        """获取可用策略列表"""
        return list(self._strategies.keys())
    
    def get_current_strategy(self) -> CacheStrategy:
        """获取当前策略类型"""
        return self._current_strategy
    
    def get_strategy_metrics(self, strategy: Optional[CacheStrategy] = None) -> StrategyMetrics:
        """获取策略性能指标"""
        if strategy is None:
            strategy = self._current_strategy
        
        if strategy in self._strategies:
            return self._strategies[strategy].get_metrics()
        else:
            return StrategyMetrics()
    
    def get_all_metrics(self) -> Dict[str, StrategyMetrics]:
        """获取所有策略的性能指标"""
        return {strategy.value: self._strategies[strategy].get_metrics() 
                for strategy in self._strategies}
    
    def optimize_strategy(self, strategy: CacheStrategy) -> StrategyPerformance:
        """优化策略性能"""
        if strategy not in self._strategies:
            return StrategyPerformance.POOR
        
        metrics = self._strategies[strategy].get_metrics()
        
        # 基于命中率、响应时间等指标评估性能
        if metrics.hit_rate >= 0.9 and metrics.avg_response_time <= 0.001:
            return StrategyPerformance.EXCELLENT
        elif metrics.hit_rate >= 0.7 and metrics.avg_response_time <= 0.005:
            return StrategyPerformance.GOOD
        elif metrics.hit_rate >= 0.5:
            return StrategyPerformance.FAIR
        else:
            return StrategyPerformance.POOR
    
    def auto_optimize(self) -> Optional[CacheStrategy]:
        """自动优化策略选择"""
        if not self.config.enable_optimization:
            return None
        
        best_strategy = None
        best_performance = StrategyPerformance.POOR
        
        for strategy in self._strategies:
            performance = self.optimize_strategy(strategy)
            if performance.value in ['excellent', 'good'] and performance != best_performance:
                if performance == StrategyPerformance.EXCELLENT:
                    best_strategy = strategy
                    best_performance = performance
                elif best_performance == StrategyPerformance.POOR:
                    best_strategy = strategy
                    best_performance = performance
        
        if best_strategy and best_strategy != self._current_strategy:
            self.switch_strategy(best_strategy)
            return best_strategy
        
        return None
    
    def add_optimization_callback(self, callback: Callable) -> None:
        """添加优化回调函数"""
        self._optimization_callbacks.append(callback)
    
    def remove_optimization_callback(self, callback: Callable) -> None:
        """移除优化回调函数"""
        if callback in self._optimization_callbacks:
            self._optimization_callbacks.remove(callback)
    
    def get_strategy_comparison(self) -> Dict[str, Dict[str, Any]]:
        """获取策略对比分析"""
        comparison = {}
        
        for strategy in self._strategies:
            metrics = self._strategies[strategy].get_metrics()
            performance = self.optimize_strategy(strategy)
            
            comparison[strategy.value] = {
                'metrics': asdict(metrics),
                'performance': performance.value,
                'is_current': strategy == self._current_strategy
            }
        
        return comparison
    
    def get_usage_statistics(self) -> Dict[str, Any]:
        """获取使用统计信息"""
        total_operations = 0
        total_hits = 0
        total_misses = 0
        total_evictions = 0
        
        for strategy in self._strategies:
            metrics = self._strategies[strategy].get_metrics()
            total_operations += metrics.hits + metrics.misses
            total_hits += metrics.hits
            total_misses += metrics.misses
            total_evictions += metrics.evictions
        
        overall_hit_rate = total_hits / total_operations if total_operations > 0 else 0.0
        
        return {
            'total_operations': total_operations,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'total_evictions': total_evictions,
            'overall_hit_rate': overall_hit_rate,
            'strategy_switches': len(self._strategy_history),
            'current_strategy': self._current_strategy.value
        }
    
    def test_strategy(self, strategy: CacheStrategy, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """测试策略性能"""
        if strategy not in self._strategies:
            return {'error': '策略不存在'}
        
        # 备份当前策略
        original_strategy = self._current_strategy
        
        try:
            # 切换到测试策略
            self.switch_strategy(strategy)
            
            # 执行测试
            test_strategy = self._strategies[strategy]
            start_time = time.time()
            
            # 执行读写操作测试
            for key, value in test_data.items():
                test_strategy.put(key, value)
            
            hit_count = 0
            for key in test_data.keys():
                if test_strategy.get(key) is not None:
                    hit_count += 1
            
            test_time = time.time() - start_time
            hit_rate = hit_count / len(test_data) if test_data else 0.0
            
            return {
                'strategy': strategy.value,
                'test_duration': test_time,
                'hit_rate': hit_rate,
                'operations_count': len(test_data),
                'performance': self.optimize_strategy(strategy).value
            }
        
        finally:
            # 恢复原始策略
            self.switch_strategy(original_strategy)
    
    def export_configuration(self) -> str:
        """导出配置为JSON格式"""
        config_data = {
            'current_strategy': self._current_strategy.value,
            'config': asdict(self.config),
            'strategy_history': self._strategy_history[-10:],  # 只保留最近10次切换
            'metrics': {k.value: asdict(v.get_metrics()) for k, v in self._strategies.items()}
        }
        return json.dumps(config_data, indent=2, ensure_ascii=False)
    
    def import_configuration(self, config_json: str) -> bool:
        """从JSON配置导入"""
        try:
            config_data = json.loads(config_json)
            
            # 恢复策略
            if 'current_strategy' in config_data:
                strategy = CacheStrategy(config_data['current_strategy'])
                if strategy in self._strategies:
                    self._current_strategy = strategy
            
            # 恢复配置
            if 'config' in config_data:
                config_dict = config_data['config']
                self.config = CacheConfig(**config_dict)
                # 重新初始化策略
                self._initialize_strategies()
            
            return True
        
        except Exception as e:
            logging.error(f"导入配置失败: {e}")
            return False
    
    def cleanup(self) -> None:
        """清理资源"""
        # 停止所有策略的清理线程
        for strategy in self._strategies.values():
            if isinstance(strategy, TTLStrategy):
                strategy.stop()
        
        # 清空历史记录
        self._strategy_history.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


# 工厂函数
def create_cache_manager(strategy: CacheStrategy = CacheStrategy.LRU, 
                        max_size: int = 1000, 
                        ttl: int = 3600) -> CacheStrategyManager:
    """创建缓存策略管理器"""
    config = CacheConfig(max_size=max_size, ttl=ttl)
    manager = CacheStrategyManager(config)
    manager.switch_strategy(strategy)
    return manager


# 装饰器
def cached(cache_manager: CacheStrategyManager, key_prefix: str = ""):
    """缓存装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 生成缓存键
            cache_key = f"{key_prefix}{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # 尝试从缓存获取
            result = cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # 执行函数并缓存结果
            result = func(*args, **kwargs)
            cache_manager.put(cache_key, result)
            return result
        
        return wrapper
    return decorator