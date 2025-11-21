#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
X8缓存优化器 - 高性能缓存优化解决方案

该模块提供了全面的缓存优化功能，包括性能优化、内存优化、
策略优化、分析监控等多个方面。

Author: X8 Team
Date: 2025-11-06
Version: 1.0.0
"""

import time
import threading
import json
import gzip
import pickle
import hashlib
import psutil
import weakref
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from abc import ABC, abstractmethod
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CacheMetrics:
    """缓存指标数据类"""
    hit_rate: float = 0.0
    miss_rate: float = 0.0
    total_requests: int = 0
    hits: int = 0
    misses: int = 0
    avg_response_time: float = 0.0
    memory_usage: int = 0
    compression_ratio: float = 0.0
    timestamp: str = ""


@dataclass
class OptimizationConfig:
    """优化配置数据类"""
    max_memory_mb: int = 100
    compression_enabled: bool = True
    compression_threshold: int = 1024
    cleanup_interval: int = 300
    optimization_interval: int = 60
    max_cache_size: int = 10000
    ttl_default: int = 3600
    strategy: str = "lru"  # lru, lfu, fifo, ttl
    enable_monitoring: bool = True
    monitoring_interval: int = 30


class CacheStrategy(ABC):
    """缓存策略抽象基类"""
    
    @abstractmethod
    def select_eviction_candidates(self, cache_data: Dict) -> List[str]:
        """选择要驱逐的缓存项"""
        pass


class LRUStrategy(CacheStrategy):
    """最近最少使用策略"""
    
    def __init__(self):
        self.access_order = OrderedDict()
    
    def select_eviction_candidates(self, cache_data: Dict) -> List[str]:
        candidates = []
        for key in self.access_order:
            if key not in cache_data:
                continue
            # 找到最少使用的项
            candidates.append(key)
            break
        return candidates
    
    def update_access(self, key: str):
        if key in self.access_order:
            self.access_order.move_to_end(key)
        else:
            self.access_order[key] = time.time()


class LFUStrategy(CacheStrategy):
    """最不经常使用策略"""
    
    def __init__(self):
        self.frequency = defaultdict(int)
        self.access_times = {}
    
    def select_eviction_candidates(self, cache_data: Dict) -> List[str]:
        if not cache_data:
            return []
        
        min_freq = min(self.frequency[key] for key in cache_data.keys() if key in self.frequency)
        candidates = [key for key, freq in self.frequency.items() 
                     if freq == min_freq and key in cache_data]
        return candidates[:1] if candidates else []
    
    def update_access(self, key: str):
        self.frequency[key] += 1
        self.access_times[key] = time.time()


class TTLStrategy(CacheStrategy):
    """TTL策略"""
    
    def __init__(self, default_ttl: int = 3600):
        self.default_ttl = default_ttl
        self.expiry_times = {}
    
    def select_eviction_candidates(self, cache_data: Dict) -> List[str]:
        current_time = time.time()
        expired_keys = []
        
        for key, expiry_time in self.expiry_times.items():
            if current_time > expiry_time and key in cache_data:
                expired_keys.append(key)
        
        return expired_keys[:1] if expired_keys else []
    
    def update_access(self, key: str):
        if key not in self.expiry_times:
            self.expiry_times[key] = time.time() + self.default_ttl


class CacheOptimizer:
    """X8缓存优化器主类"""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.cache_data = {}
        self.metadata = {}
        self.metrics = CacheMetrics()
        self.lock = threading.RLock()
        
        # 初始化策略
        self._init_strategy()
        
        # 初始化监控
        if self.config.enable_monitoring:
            self._init_monitoring()
        
        # 初始化压缩器
        self.compressor = Compressor()
        
        # 启动后台任务
        self._start_background_tasks()
    
    def _init_strategy(self):
        """初始化缓存策略"""
        strategies = {
            "lru": LRUStrategy,
            "lfu": LFUStrategy,
            "ttl": TTLStrategy,
            "fifo": lambda: LRUStrategy()  # 简化的FIFO实现
        }
        
        strategy_class = strategies.get(self.config.strategy, LRUStrategy)
        if self.config.strategy == "ttl":
            self.strategy = strategy_class(self.config.ttl_default)
        else:
            self.strategy = strategy_class()
    
    def _init_monitoring(self):
        """初始化监控"""
        self.monitoring_active = True
        self.optimization_history = []
        self.performance_data = defaultdict(list)
    
    def _start_background_tasks(self):
        """启动后台任务"""
        def cleanup_task():
            while True:
                time.sleep(self.config.cleanup_interval)
                self._cleanup_expired_items()
        
        def optimization_task():
            while True:
                time.sleep(self.config.optimization_interval)
                self._perform_optimization()
        
        def monitoring_task():
            while self.monitoring_active:
                time.sleep(self.config.monitoring_interval)
                self._update_metrics()
        
        # 启动后台线程
        threading.Thread(target=cleanup_task, daemon=True).start()
        threading.Thread(target=optimization_task, daemon=True).start()
        
        if self.config.enable_monitoring:
            threading.Thread(target=monitoring_task, daemon=True).start()
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        with self.lock:
            start_time = time.time()
            
            if key not in self.cache_data:
                self.metrics.misses += 1
                self.metrics.total_requests += 1
                return None
            
            # 检查是否过期
            if self._is_expired(key):
                self._remove(key)
                self.metrics.misses += 1
                self.metrics.total_requests += 1
                return None
            
            # 获取数据
            value = self.cache_data[key]
            
            # 解压缩
            if isinstance(value, dict) and value.get('_compressed'):
                value = self.compressor.decompress(value['data'])
            
            # 更新访问信息
            self.strategy.update_access(key)
            self.metadata[key]['last_access'] = time.time()
            self.metadata[key]['access_count'] += 1
            
            # 更新指标
            self.metrics.hits += 1
            self.metrics.total_requests += 1
            self.metrics.avg_response_time = (time.time() - start_time)
            
            return value
    
    def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """设置缓存值"""
        with self.lock:
            try:
                # 检查内存限制
                if not self._check_memory_limit():
                    self._evict_items()
                
                # 设置TTL
                if ttl is None:
                    ttl = self.config.ttl_default
                
                # 压缩大对象
                original_value = value
                if self.config.compression_enabled and self._should_compress(value):
                    compressed_data = self.compressor.compress(value)
                    if compressed_data:
                        value = {'_compressed': True, 'data': compressed_data}
                
                # 存储数据
                self.cache_data[key] = value
                self.metadata[key] = {
                    'created': time.time(),
                    'last_access': time.time(),
                    'access_count': 1,
                    'ttl': ttl,
                    'size': self._calculate_size(value),
                    'compressed': isinstance(value, dict) and value.get('_compressed', False)
                }
                
                # 更新策略
                self.strategy.update_access(key)
                
                # 更新指标
                self._update_memory_usage()
                
                return True
                
            except Exception as e:
                logger.error(f"设置缓存失败: {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """删除缓存项"""
        with self.lock:
            if key in self.cache_data:
                del self.cache_data[key]
                del self.metadata[key]
                self._update_memory_usage()
                return True
            return False
    
    def clear(self):
        """清空缓存"""
        with self.lock:
            self.cache_data.clear()
            self.metadata.clear()
            self._update_memory_usage()
    
    def _is_expired(self, key: str) -> bool:
        """检查是否过期"""
        if key not in self.metadata:
            return True
        
        meta = self.metadata[key]
        current_time = time.time()
        return current_time > (meta['created'] + meta['ttl'])
    
    def _should_compress(self, value: Any) -> bool:
        """判断是否应该压缩"""
        if not isinstance(value, (str, bytes, dict, list)):
            return False
        
        size = self._calculate_size(value)
        return size > self.config.compression_threshold
    
    def _calculate_size(self, value: Any) -> int:
        """计算值的大小"""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode('utf-8'))
    
    def _check_memory_limit(self) -> bool:
        """检查内存限制"""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        return current_memory < self.config.max_memory_mb
    
    def _evict_items(self):
        """驱逐缓存项"""
        candidates = self.strategy.select_eviction_candidates(self.cache_data)
        
        for key in candidates:
            if len(self.cache_data) <= 1:  # 保留至少一项
                break
            self._remove(key)
    
    def _remove(self, key: str):
        """移除缓存项"""
        if key in self.cache_data:
            del self.cache_data[key]
        if key in self.metadata:
            del self.metadata[key]
    
    def _cleanup_expired_items(self):
        """清理过期项"""
        with self.lock:
            expired_keys = [key for key in self.cache_data.keys() if self._is_expired(key)]
            for key in expired_keys:
                self._remove(key)
            
            if expired_keys:
                logger.info(f"清理了 {len(expired_keys)} 个过期缓存项")
    
    def _perform_optimization(self):
        """执行优化"""
        with self.lock:
            try:
                # 内存优化
                self._optimize_memory()
                
                # 压缩优化
                self._optimize_compression()
                
                # 策略优化
                self._optimize_strategy()
                
                # 记录优化结果
                self._record_optimization()
                
            except Exception as e:
                logger.error(f"优化过程出错: {e}")
    
    def _optimize_memory(self):
        """内存优化"""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        if current_memory > self.config.max_memory_mb * 0.8:
            # 强制清理过期项
            self._cleanup_expired_items()
            
            # 驱逐更多项
            target_size = len(self.cache_data) // 2
            while len(self.cache_data) > target_size:
                candidates = self.strategy.select_eviction_candidates(self.cache_data)
                if not candidates:
                    break
                self._remove(candidates[0])
    
    def _optimize_compression(self):
        """压缩优化"""
        if not self.config.compression_enabled:
            return
        
        for key, value in list(self.cache_data.items()):
            if isinstance(value, dict) and not value.get('_compressed'):
                # 检查是否值得压缩
                if self._should_compress(value):
                    compressed_data = self.compressor.compress(value)
                    if compressed_data:
                        self.cache_data[key] = {'_compressed': True, 'data': compressed_data}
                        self.metadata[key]['compressed'] = True
                        self.metadata[key]['size'] = len(compressed_data)
    
    def _optimize_strategy(self):
        """策略优化"""
        # 根据访问模式调整策略参数
        if hasattr(self.strategy, 'default_ttl'):
            # 动态调整TTL
            avg_access_interval = self._calculate_avg_access_interval()
            if avg_access_interval > 0:
                self.strategy.default_ttl = min(max(avg_access_interval * 2, 300), 7200)
    
    def _calculate_avg_access_interval(self) -> float:
        """计算平均访问间隔"""
        access_times = [meta['last_access'] for meta in self.metadata.values()]
        if len(access_times) < 2:
            return 0
        
        access_times.sort()
        intervals = [access_times[i] - access_times[i-1] for i in range(1, len(access_times))]
        return sum(intervals) / len(intervals)
    
    def _record_optimization(self):
        """记录优化结果"""
        optimization_record = {
            'timestamp': datetime.now().isoformat(),
            'cache_size': len(self.cache_data),
            'memory_usage': psutil.Process().memory_info().rss / 1024 / 1024,
            'hit_rate': self.metrics.hit_rate,
            'compression_ratio': self._calculate_compression_ratio()
        }
        self.optimization_history.append(optimization_record)
        
        # 保留最近100条记录
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
    
    def _calculate_compression_ratio(self) -> float:
        """计算压缩比"""
        total_original_size = sum(meta['size'] for meta in self.metadata.values())
        total_compressed_size = 0
        
        for key, meta in self.metadata.items():
            if meta.get('compressed') and key in self.cache_data:
                cached_value = self.cache_data[key]
                if isinstance(cached_value, dict) and cached_value.get('_compressed'):
                    total_compressed_size += len(cached_value['data'])
        
        if total_original_size > 0:
            return (total_original_size - total_compressed_size) / total_original_size
        return 0.0
    
    def _update_memory_usage(self):
        """更新内存使用量"""
        self.metrics.memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
    
    def _update_metrics(self):
        """更新指标"""
        total = self.metrics.total_requests
        if total > 0:
            self.metrics.hit_rate = self.metrics.hits / total
            self.metrics.miss_rate = self.metrics.misses / total
        
        self.metrics.timestamp = datetime.now().isoformat()
        self.metrics.compression_ratio = self._calculate_compression_ratio()
    
    def get_metrics(self) -> CacheMetrics:
        """获取缓存指标"""
        self._update_metrics()
        return self.metrics
    
    def get_optimization_suggestions(self) -> List[str]:
        """获取优化建议"""
        suggestions = []
        
        # 基于命中率建议
        if self.metrics.hit_rate < 0.8:
            suggestions.append("命中率较低，建议增加缓存容量或调整TTL设置")
        
        # 基于内存使用建议
        if self.metrics.memory_usage > self.config.max_memory_mb * 0.9:
            suggestions.append("内存使用率过高，建议启用压缩或减少缓存容量")
        
        # 基于压缩效果建议
        if self.metrics.compression_ratio < 0.1:
            suggestions.append("压缩效果不明显，建议调整压缩阈值或禁用压缩")
        
        # 基于响应时间建议
        if self.metrics.avg_response_time > 0.1:
            suggestions.append("响应时间较长，建议优化缓存策略或增加内存")
        
        # 基于缓存大小建议
        if len(self.cache_data) > self.config.max_cache_size * 0.9:
            suggestions.append("缓存项数量接近上限，建议清理过期项或增加容量")
        
        return suggestions
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        report = {
            'metrics': asdict(self.get_metrics()),
            'configuration': asdict(self.config),
            'optimization_suggestions': self.get_optimization_suggestions(),
            'optimization_history': self.optimization_history[-10:],  # 最近10条记录
            'cache_statistics': {
                'total_items': len(self.cache_data),
                'compressed_items': sum(1 for meta in self.metadata.values() if meta.get('compressed')),
                'expired_items': sum(1 for key in self.cache_data.keys() if self._is_expired(key)),
                'avg_item_size': sum(meta['size'] for meta in self.metadata.values()) / max(len(self.metadata), 1)
            }
        }
        
        return report
    
    def export_cache(self, filepath: str) -> bool:
        """导出缓存数据"""
        try:
            export_data = {
                'cache_data': self.cache_data,
                'metadata': self.metadata,
                'config': asdict(self.config),
                'export_time': datetime.now().isoformat()
            }
            
            with gzip.open(filepath, 'wb') as f:
                pickle.dump(export_data, f)
            
            logger.info(f"缓存数据已导出到: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"导出缓存失败: {e}")
            return False
    
    def import_cache(self, filepath: str) -> bool:
        """导入缓存数据"""
        try:
            with gzip.open(filepath, 'rb') as f:
                import_data = pickle.load(f)
            
            self.cache_data = import_data['cache_data']
            self.metadata = import_data['metadata']
            
            logger.info(f"缓存数据已从 {filepath} 导入")
            return True
            
        except Exception as e:
            logger.error(f"导入缓存失败: {e}")
            return False


class Compressor:
    """数据压缩器"""
    
    @staticmethod
    def compress(data: Any) -> Optional[bytes]:
        """压缩数据"""
        try:
            serialized_data = pickle.dumps(data)
            compressed_data = gzip.compress(serialized_data)
            
            # 只在压缩有效率的情况下使用压缩
            if len(compressed_data) < len(serialized_data) * 0.9:
                return compressed_data
            return None
            
        except Exception as e:
            logger.error(f"压缩失败: {e}")
            return None
    
    @staticmethod
    def decompress(compressed_data: bytes) -> Any:
        """解压缩数据"""
        try:
            serialized_data = gzip.decompress(compressed_data)
            return pickle.loads(serialized_data)
        except Exception as e:
            logger.error(f"解压缩失败: {e}")
            return None


class CacheAnalyzer:
    """缓存分析器"""
    
    def __init__(self, optimizer: CacheOptimizer):
        self.optimizer = optimizer
    
    def analyze_cache_patterns(self) -> Dict[str, Any]:
        """分析缓存模式"""
        with self.optimizer.lock:
            patterns = {
                'access_distribution': self._analyze_access_distribution(),
                'size_distribution': self._analyze_size_distribution(),
                'temporal_patterns': self._analyze_temporal_patterns(),
                'hot_keys': self._identify_hot_keys(),
                'cold_keys': self._identify_cold_keys()
            }
        
        return patterns
    
    def _analyze_access_distribution(self) -> Dict[str, Any]:
        """分析访问分布"""
        access_counts = [meta['access_count'] for meta in self.optimizer.metadata.values()]
        
        if not access_counts:
            return {}
        
        return {
            'mean': sum(access_counts) / len(access_counts),
            'max': max(access_counts),
            'min': min(access_counts),
            'distribution': self._calculate_distribution(access_counts)
        }
    
    def _analyze_size_distribution(self) -> Dict[str, Any]:
        """分析大小分布"""
        sizes = [meta['size'] for meta in self.optimizer.metadata.values()]
        
        if not sizes:
            return {}
        
        return {
            'mean': sum(sizes) / len(sizes),
            'max': max(sizes),
            'min': min(sizes),
            'total': sum(sizes)
        }
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """分析时间模式"""
        access_times = [meta['last_access'] for meta in self.optimizer.metadata.values()]
        
        if len(access_times) < 2:
            return {}
        
        access_times.sort()
        intervals = [access_times[i] - access_times[i-1] for i in range(1, len(access_times))]
        
        return {
            'avg_interval': sum(intervals) / len(intervals),
            'access_rate': len(access_times) / (max(access_times) - min(access_times)) if max(access_times) > min(access_times) else 0
        }
    
    def _identify_hot_keys(self, threshold: int = 10) -> List[str]:
        """识别热门键"""
        hot_keys = []
        for key, meta in self.optimizer.metadata.items():
            if meta['access_count'] >= threshold:
                hot_keys.append(key)
        
        return sorted(hot_keys, key=lambda k: self.optimizer.metadata[k]['access_count'], reverse=True)
    
    def _identify_cold_keys(self, threshold: int = 2) -> List[str]:
        """识别冷门键"""
        cold_keys = []
        for key, meta in self.optimizer.metadata.items():
            if meta['access_count'] <= threshold:
                cold_keys.append(key)
        
        return cold_keys
    
    def _calculate_distribution(self, values: List[float]) -> Dict[str, int]:
        """计算分布"""
        if not values:
            return {}
        
        min_val, max_val = min(values), max(values)
        range_size = (max_val - min_val) / 10 if max_val > min_val else 1
        
        distribution = defaultdict(int)
        for value in values:
            bin_index = int((value - min_val) / range_size)
            bin_index = min(bin_index, 9)  # 确保在0-9范围内
            distribution[bin_index] += 1
        
        return dict(distribution)


# 便捷函数
def create_cache_optimizer(config: OptimizationConfig = None) -> CacheOptimizer:
    """创建缓存优化器实例"""
    return CacheOptimizer(config)


def get_default_config() -> OptimizationConfig:
    """获取默认配置"""
    return OptimizationConfig()


if __name__ == "__main__":
    # 示例使用
    config = OptimizationConfig(
        max_memory_mb=50,
        compression_enabled=True,
        compression_threshold=512,
        strategy="lru",
        enable_monitoring=True
    )
    
    optimizer = CacheOptimizer(config)
    
    # 测试缓存操作
    optimizer.set("test_key", "test_value", ttl=60)
    value = optimizer.get("test_key")
    print(f"获取到的值: {value}")
    
    # 获取性能报告
    report = optimizer.get_performance_report()
    print(f"性能报告: {json.dumps(report, indent=2, ensure_ascii=False)}")