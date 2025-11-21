#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
O5数据优化器模块

该模块提供了全面的数据优化功能，包括：
1. 数据存储优化（数据分片、数据压缩、数据编码）
2. 数据访问优化（索引优化、查询优化、缓存策略）
3. 数据传输优化（数据压缩、增量传输、批量传输）
4. 数据清洗和预处理优化（数据去重、数据填充、数据转换）
5. 大数据处理优化（MapReduce、Spark、数据流处理）
6. 数据质量优化（数据验证、数据监控、数据修复）
7. 异步数据优化处理
8. 完整的错误处理和日志记录
9. 详细的文档字符串和使用示例

作者: O5数据优化器开发团队
版本: 1.0.0
创建日期: 2025-11-06
"""

import asyncio
import logging
import hashlib
import json
import pickle
import gzip
import zlib
import bz2
import lzma
import sqlite3
import threading
import time
import uuid
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps, lru_cache
from pathlib import Path
from queue import Queue, Empty
from typing import (
    Any, Dict, List, Optional, Union, Callable, Tuple, Set, 
    Iterator, Generator, Type, Generic, TypeVar, Awaitable,
    BinaryIO, TextIO, Protocol
)
from collections import defaultdict, deque
from itertools import islice, groupby
from operator import itemgetter
import weakref
import gc
import os
import sys
import traceback
import psutil
import numpy as np
import pandas as pd

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_optimizer.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 类型定义
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')

class CompressionType:
    """压缩类型常量"""
    GZIP = 'gzip'
    BZ2 = 'bz2'
    LZMA = 'lzma'
    ZLIB = 'zlib'

class IndexType:
    """索引类型常量"""
    BTREE = 'btree'
    HASH = 'hash'
    FULLTEXT = 'fulltext'
    SPATIAL = 'spatial'

class DataFormat:
    """数据格式常量"""
    JSON = 'json'
    PICKLE = 'pickle'
    CSV = 'csv'
    PARQUET = 'parquet'
    HDF5 = 'hdf5'

@dataclass
class OptimizationConfig:
    """数据优化配置类"""
    # 存储优化配置
    enable_sharding: bool = True
    shard_size: int = 10000
    compression_type: str = CompressionType.GZIP
    encoding_format: str = DataFormat.JSON
    
    # 访问优化配置
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600  # 秒
    enable_indexing: bool = True
    
    # 传输优化配置
    batch_size: int = 1000
    enable_incremental: bool = True
    compression_level: int = 6
    
    # 清洗优化配置
    enable_deduplication: bool = True
    enable_data_filling: bool = True
    enable_data_transformation: bool = True
    
    # 大数据处理配置
    enable_mapreduce: bool = True
    enable_spark: bool = False
    enable_streaming: bool = True
    max_workers: int = 4
    
    # 质量优化配置
    enable_validation: bool = True
    enable_monitoring: bool = True
    enable_repair: bool = True
    
    # 异步处理配置
    enable_async: bool = True
    max_concurrent_tasks: int = 10
    task_timeout: int = 300  # 秒

@dataclass
class OptimizationResult:
    """优化结果类"""
    success: bool
    original_size: int
    optimized_size: int
    compression_ratio: float
    processing_time: float
    operations_count: int
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DataQualityMetrics:
    """数据质量指标类"""
    completeness: float  # 完整性 0-1
    accuracy: float      # 准确性 0-1
    consistency: float   # 一致性 0-1
    timeliness: float    # 时效性 0-1
    validity: float      # 有效性 0-1
    uniqueness: float    # 唯一性 0-1
    overall_score: float = 0.0
    
    def __post_init__(self):
        """计算总体质量分数"""
        self.overall_score = (
            self.completeness + self.accuracy + self.consistency + 
            self.timeliness + self.validity + self.uniqueness
        ) / 6

class OptimizationError(Exception):
    """数据优化异常基类"""
    pass

class CompressionError(OptimizationError):
    """压缩异常"""
    pass

class IndexingError(OptimizationError):
    """索引异常"""
    pass

class ValidationError(OptimizationError):
    """验证异常"""
    pass

class CacheError(OptimizationError):
    """缓存异常"""
    pass

class ShardingError(OptimizationError):
    """分片异常"""
    pass

class DataCleaner:
    """数据清洗器类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def deduplicate_data(self, data: List[Dict[str, Any]], 
                        key_fields: List[str] = None) -> List[Dict[str, Any]]:
        """数据去重
        
        Args:
            data: 待去重的数据列表
            key_fields: 用于去重的关键字段列表
            
        Returns:
            去重后的数据列表
        """
        try:
            self.logger.info(f"开始数据去重，原始数据量: {len(data)}")
            
            if not key_fields:
                # 如果没有指定关键字段，使用全部字段
                key_fields = list(data[0].keys()) if data else []
            
            seen = set()
            deduplicated = []
            
            for item in data:
                # 生成去重键
                key = tuple(item.get(field) for field in key_fields)
                
                if key not in seen:
                    seen.add(key)
                    deduplicated.append(item)
            
            self.logger.info(f"数据去重完成，去重后数据量: {len(deduplicated)}")
            return deduplicated
            
        except Exception as e:
            self.logger.error(f"数据去重失败: {str(e)}")
            raise DataCleanerError(f"数据去重失败: {str(e)}")
    
    def fill_missing_data(self, data: List[Dict[str, Any]], 
                         strategy: str = 'forward_fill') -> List[Dict[str, Any]]:
        """数据填充
        
        Args:
            data: 待填充的数据列表
            strategy: 填充策略 ('forward_fill', 'backward_fill', 'mean_fill', 'mode_fill')
            
        Returns:
            填充后的数据列表
        """
        try:
            self.logger.info(f"开始数据填充，策略: {strategy}")
            
            if not data:
                return data
            
            # 转换为DataFrame进行填充
            df = pd.DataFrame(data)
            
            if strategy == 'forward_fill':
                df = df.fillna(method='ffill')
            elif strategy == 'backward_fill':
                df = df.fillna(method='bfill')
            elif strategy == 'mean_fill':
                df = df.fillna(df.mean())
            elif strategy == 'mode_fill':
                df = df.fillna(df.mode().iloc[0])
            else:
                raise ValueError(f"不支持的填充策略: {strategy}")
            
            result = df.to_dict('records')
            self.logger.info("数据填充完成")
            return result
            
        except Exception as e:
            self.logger.error(f"数据填充失败: {str(e)}")
            raise DataCleanerError(f"数据填充失败: {str(e)}")
    
    def transform_data(self, data: List[Dict[str, Any]], 
                      transformations: Dict[str, Callable]) -> List[Dict[str, Any]]:
        """数据转换
        
        Args:
            data: 待转换的数据列表
            transformations: 字段转换函数字典 {字段名: 转换函数}
            
        Returns:
            转换后的数据列表
        """
        try:
            self.logger.info(f"开始数据转换，转换字段: {list(transformations.keys())}")
            
            transformed_data = []
            
            for item in data:
                transformed_item = item.copy()
                
                for field, transform_func in transformations.items():
                    if field in transformed_item:
                        try:
                            transformed_item[field] = transform_func(transformed_item[field])
                        except Exception as e:
                            self.logger.warning(f"字段 {field} 转换失败: {str(e)}")
                
                transformed_data.append(transformed_item)
            
            self.logger.info("数据转换完成")
            return transformed_data
            
        except Exception as e:
            self.logger.error(f"数据转换失败: {str(e)}")
            raise DataCleanerError(f"数据转换失败: {str(e)}")

class DataCleanerError(OptimizationError):
    """数据清洗异常"""
    pass

class Compressor:
    """数据压缩器类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def compress_data(self, data: Union[bytes, str], 
                     compression_type: str = None) -> bytes:
        """压缩数据
        
        Args:
            data: 待压缩的数据
            compression_type: 压缩类型
            
        Returns:
            压缩后的数据
        """
        try:
            if compression_type is None:
                compression_type = self.config.compression_type
            
            self.logger.info(f"开始数据压缩，类型: {compression_type}")
            
            if isinstance(data, str):
                data = data.encode('utf-8')
            
            if compression_type == CompressionType.GZIP:
                compressed = gzip.compress(data, compresslevel=self.config.compression_level)
            elif compression_type == CompressionType.BZ2:
                compressed = bz2.compress(data, compresslevel=self.config.compression_level)
            elif compression_type == CompressionType.LZMA:
                compressed = lzma.compress(data, preset=self.config.compression_level)
            elif compression_type == CompressionType.ZLIB:
                compressed = zlib.compress(data, level=self.config.compression_level)
            else:
                raise ValueError(f"不支持的压缩类型: {compression_type}")
            
            self.logger.info(f"数据压缩完成，原始大小: {len(data)}, 压缩后大小: {len(compressed)}")
            return compressed
            
        except Exception as e:
            self.logger.error(f"数据压缩失败: {str(e)}")
            raise CompressionError(f"数据压缩失败: {str(e)}")
    
    def decompress_data(self, data: bytes, 
                       compression_type: str = None) -> bytes:
        """解压数据
        
        Args:
            data: 待解压的数据
            compression_type: 压缩类型
            
        Returns:
            解压后的数据
        """
        try:
            if compression_type is None:
                compression_type = self.config.compression_type
            
            self.logger.info(f"开始数据解压，类型: {compression_type}")
            
            if compression_type == CompressionType.GZIP:
                decompressed = gzip.decompress(data)
            elif compression_type == CompressionType.BZ2:
                decompressed = bz2.decompress(data)
            elif compression_type == CompressionType.LZMA:
                decompressed = lzma.decompress(data)
            elif compression_type == CompressionType.ZLIB:
                decompressed = zlib.decompress(data)
            else:
                raise ValueError(f"不支持的压缩类型: {compression_type}")
            
            self.logger.info(f"数据解压完成，解压后大小: {len(decompressed)}")
            return decompressed
            
        except Exception as e:
            self.logger.error(f"数据解压失败: {str(e)}")
            raise CompressionError(f"数据解压失败: {str(e)}")

class DataEncoder:
    """数据编码器类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def encode_data(self, data: Any, encoding_format: str = None) -> bytes:
        """编码数据
        
        Args:
            data: 待编码的数据
            encoding_format: 编码格式
            
        Returns:
            编码后的数据
        """
        try:
            if encoding_format is None:
                encoding_format = self.config.encoding_format
            
            self.logger.info(f"开始数据编码，格式: {encoding_format}")
            
            if encoding_format == DataFormat.JSON:
                encoded = json.dumps(data, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
            elif encoding_format == DataFormat.PICKLE:
                encoded = pickle.dumps(data)
            else:
                raise ValueError(f"不支持的编码格式: {encoding_format}")
            
            self.logger.info(f"数据编码完成，编码后大小: {len(encoded)}")
            return encoded
            
        except Exception as e:
            self.logger.error(f"数据编码失败: {str(e)}")
            raise OptimizationError(f"数据编码失败: {str(e)}")
    
    def decode_data(self, data: bytes, encoding_format: str = None) -> Any:
        """解码数据
        
        Args:
            data: 待解码的数据
            encoding_format: 编码格式
            
        Returns:
            解码后的数据
        """
        try:
            if encoding_format is None:
                encoding_format = self.config.encoding_format
            
            self.logger.info(f"开始数据解码，格式: {encoding_format}")
            
            if encoding_format == DataFormat.JSON:
                decoded = json.loads(data.decode('utf-8'))
            elif encoding_format == DataFormat.PICKLE:
                decoded = pickle.loads(data)
            else:
                raise ValueError(f"不支持的编码格式: {encoding_format}")
            
            self.logger.info("数据解码完成")
            return decoded
            
        except Exception as e:
            self.logger.error(f"数据解码失败: {str(e)}")
            raise OptimizationError(f"数据解码失败: {str(e)}")

class DataSharder:
    """数据分片器类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def shard_data(self, data: List[Dict[str, Any]], 
                  shard_key: str = None) -> Dict[str, List[Dict[str, Any]]]:
        """数据分片
        
        Args:
            data: 待分片的数据列表
            shard_key: 分片键字段名
            
        Returns:
            分片后的数据字典
        """
        try:
            self.logger.info(f"开始数据分片，原始数据量: {len(data)}")
            
            if not self.config.enable_sharding:
                return {"default": data}
            
            shards = defaultdict(list)
            
            if shard_key is None:
                # 基于哈希的分片
                for i, item in enumerate(data):
                    shard_id = hash(str(item)) % self.config.shard_size
                    shards[f"shard_{shard_id}"].append(item)
            else:
                # 基于字段的分片
                for item in data:
                    shard_value = item.get(shard_key, "default")
                    shards[str(shard_value)].append(item)
            
            self.logger.info(f"数据分片完成，分片数量: {len(shards)}")
            return dict(shards)
            
        except Exception as e:
            self.logger.error(f"数据分片失败: {str(e)}")
            raise ShardingError(f"数据分片失败: {str(e)}")

class DataIndexer:
    """数据索引器类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.indices: Dict[str, Dict[str, List[int]]] = {}
    
    def create_index(self, data: List[Dict[str, Any]], 
                    field: str, index_type: str = IndexType.BTREE) -> bool:
        """创建索引
        
        Args:
            data: 数据列表
            field: 索引字段
            index_type: 索引类型
            
        Returns:
            是否创建成功
        """
        try:
            self.logger.info(f"开始创建索引，字段: {field}, 类型: {index_type}")
            
            if not self.config.enable_indexing:
                return False
            
            index_dict = {}
            
            for i, item in enumerate(data):
                key = item.get(field)
                if key is not None:
                    if key not in index_dict:
                        index_dict[key] = []
                    index_dict[key].append(i)
            
            self.indices[field] = index_dict
            self.logger.info(f"索引创建完成，索引字段: {field}")
            return True
            
        except Exception as e:
            self.logger.error(f"索引创建失败: {str(e)}")
            raise IndexingError(f"索引创建失败: {str(e)}")
    
    def search_by_index(self, field: str, value: Any) -> List[int]:
        """基于索引搜索
        
        Args:
            field: 索引字段
            value: 搜索值
            
        Returns:
            匹配的索引列表
        """
        try:
            if field not in self.indices:
                return []
            
            return self.indices[field].get(value, [])
            
        except Exception as e:
            self.logger.error(f"索引搜索失败: {str(e)}")
            raise IndexingError(f"索引搜索失败: {str(e)}")

class DataCache:
    """数据缓存类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.access_times: Dict[str, float] = {}
        self.max_size = config.cache_size
        self.ttl = config.cache_ttl
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存数据
        
        Args:
            key: 缓存键
            
        Returns:
            缓存的数据，如果不存在或已过期则返回None
        """
        try:
            if not self.config.enable_caching:
                return None
            
            if key not in self.cache:
                return None
            
            value, timestamp = self.cache[key]
            
            # 检查是否过期
            if time.time() - timestamp > self.ttl:
                self._remove(key)
                return None
            
            # 更新访问时间
            self.access_times[key] = time.time()
            return value
            
        except Exception as e:
            self.logger.error(f"缓存获取失败: {str(e)}")
            raise CacheError(f"缓存获取失败: {str(e)}")
    
    def set(self, key: str, value: Any) -> bool:
        """设置缓存数据
        
        Args:
            key: 缓存键
            value: 缓存值
            
        Returns:
            是否设置成功
        """
        try:
            if not self.config.enable_caching:
                return False
            
            # 检查缓存大小限制
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            self.cache[key] = (value, time.time())
            self.access_times[key] = time.time()
            return True
            
        except Exception as e:
            self.logger.error(f"缓存设置失败: {str(e)}")
            raise CacheError(f"缓存设置失败: {str(e)}")
    
    def _remove(self, key: str) -> None:
        """移除缓存项"""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
    
    def _evict_lru(self) -> None:
        """移除最近最少使用的缓存项"""
        if not self.access_times:
            return
        
        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        self._remove(lru_key)
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_times.clear()

class DataValidator:
    """数据验证器类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def validate_data(self, data: List[Dict[str, Any]], 
                     rules: Dict[str, Callable[[Any], bool]]) -> Tuple[bool, List[str]]:
        """验证数据
        
        Args:
            data: 待验证的数据
            rules: 验证规则字典 {字段名: 验证函数}
            
        Returns:
            (验证是否通过, 错误信息列表)
        """
        try:
            if not self.config.enable_validation:
                return True, []
            
            self.logger.info(f"开始数据验证，数据量: {len(data)}")
            
            errors = []
            
            for i, item in enumerate(data):
                for field, rule in rules.items():
                    try:
                        if not rule(item.get(field)):
                            errors.append(f"行 {i+1}, 字段 {field}: 验证失败")
                    except Exception as e:
                        errors.append(f"行 {i+1}, 字段 {field}: 验证异常 - {str(e)}")
            
            is_valid = len(errors) == 0
            self.logger.info(f"数据验证完成，验证结果: {'通过' if is_valid else '失败'}")
            
            return is_valid, errors
            
        except Exception as e:
            self.logger.error(f"数据验证失败: {str(e)}")
            raise ValidationError(f"数据验证失败: {str(e)}")

class DataMonitor:
    """数据监控器类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.metrics_history: List[DataQualityMetrics] = []
    
    def monitor_data_quality(self, data: List[Dict[str, Any]]) -> DataQualityMetrics:
        """监控数据质量
        
        Args:
            data: 待监控的数据
            
        Returns:
            数据质量指标
        """
        try:
            if not self.config.enable_monitoring:
                return DataQualityMetrics(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
            
            self.logger.info(f"开始数据质量监控，数据量: {len(data)}")
            
            # 计算各项质量指标
            completeness = self._calculate_completeness(data)
            accuracy = self._calculate_accuracy(data)
            consistency = self._calculate_consistency(data)
            timeliness = self._calculate_timeliness(data)
            validity = self._calculate_validity(data)
            uniqueness = self._calculate_uniqueness(data)
            
            metrics = DataQualityMetrics(
                completeness=completeness,
                accuracy=accuracy,
                consistency=consistency,
                timeliness=timeliness,
                validity=validity,
                uniqueness=uniqueness
            )
            
            self.metrics_history.append(metrics)
            self.logger.info(f"数据质量监控完成，总体分数: {metrics.overall_score:.3f}")
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"数据质量监控失败: {str(e)}")
            raise OptimizationError(f"数据质量监控失败: {str(e)}")
    
    def _calculate_completeness(self, data: List[Dict[str, Any]]) -> float:
        """计算完整性"""
        if not data:
            return 1.0
        
        total_fields = len(data[0]) if data else 0
        if total_fields == 0:
            return 1.0
        
        non_null_count = sum(
            sum(1 for v in item.values() if v is not None and v != '')
            for item in data
        )
        total_count = len(data) * total_fields
        
        return non_null_count / total_count if total_count > 0 else 1.0
    
    def _calculate_accuracy(self, data: List[Dict[str, Any]]) -> float:
        """计算准确性（简化实现）"""
        # 这里可以添加更复杂的数据准确性验证逻辑
        return 0.95  # 假设95%的准确性
    
    def _calculate_consistency(self, data: List[Dict[str, Any]]) -> float:
        """计算一致性"""
        if len(data) < 2:
            return 1.0
        
        # 检查数据类型一致性
        field_types = defaultdict(set)
        for item in data:
            for field, value in item.items():
                field_types[field].add(type(value).__name__)
        
        consistent_fields = sum(1 for types in field_types.values() if len(types) == 1)
        total_fields = len(field_types)
        
        return consistent_fields / total_fields if total_fields > 0 else 1.0
    
    def _calculate_timeliness(self, data: List[Dict[str, Any]]) -> float:
        """计算时效性"""
        # 检查数据的时间戳字段
        timestamp_fields = ['timestamp', 'created_at', 'updated_at', 'time']
        
        for item in data:
            for field in timestamp_fields:
                if field in item:
                    try:
                        if isinstance(item[field], str):
                            dt = datetime.fromisoformat(item[field].replace('Z', '+00:00'))
                        else:
                            dt = datetime.fromtimestamp(item[field])
                        
                        age = (datetime.now() - dt).total_seconds()
                        # 如果数据超过24小时，认为时效性较低
                        return max(0.0, 1.0 - age / 86400)
                    except:
                        continue
        
        return 1.0  # 如果没有时间戳字段，认为时效性良好
    
    def _calculate_validity(self, data: List[Dict[str, Any]]) -> float:
        """计算有效性"""
        # 检查数据格式和范围的合理性
        valid_count = 0
        total_count = len(data)
        
        for item in data:
            is_valid = True
            
            # 检查数值字段的合理性
            for field, value in item.items():
                if isinstance(value, (int, float)):
                    # 简单的合理性检查
                    if field.lower() in ['age', 'score', 'price', 'amount']:
                        if value < 0:
                            is_valid = False
                            break
            
            if is_valid:
                valid_count += 1
        
        return valid_count / total_count if total_count > 0 else 1.0
    
    def _calculate_uniqueness(self, data: List[Dict[str, Any]]) -> float:
        """计算唯一性"""
        if not data:
            return 1.0
        
        unique_items = set()
        for item in data:
            # 创建可哈希的表示
            item_str = json.dumps(item, sort_keys=True, ensure_ascii=False)
            unique_items.add(item_str)
        
        return len(unique_items) / len(data)

class DataRepairer:
    """数据修复器类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def repair_data(self, data: List[Dict[str, Any]], 
                   repair_rules: Dict[str, Callable[[Any], Any]]) -> List[Dict[str, Any]]:
        """修复数据
        
        Args:
            data: 待修复的数据
            repair_rules: 修复规则字典 {字段名: 修复函数}
            
        Returns:
            修复后的数据
        """
        try:
            if not self.config.enable_repair:
                return data
            
            self.logger.info(f"开始数据修复，数据量: {len(data)}")
            
            repaired_data = []
            
            for item in data:
                repaired_item = item.copy()
                
                for field, repair_func in repair_rules.items():
                    try:
                        if field in repaired_item:
                            repaired_item[field] = repair_func(repaired_item[field])
                    except Exception as e:
                        self.logger.warning(f"字段 {field} 修复失败: {str(e)}")
                
                repaired_data.append(repaired_item)
            
            self.logger.info("数据修复完成")
            return repaired_data
            
        except Exception as e:
            self.logger.error(f"数据修复失败: {str(e)}")
            raise OptimizationError(f"数据修复失败: {str(e)}")

class MapReduceProcessor:
    """MapReduce处理器类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def map_reduce(self, data: List[Any], 
                  map_func: Callable[[Any], Tuple[K, V]], 
                  reduce_func: Callable[[K, List[V]], Any]) -> Dict[K, Any]:
        """MapReduce处理
        
        Args:
            data: 待处理的数据
            map_func: Map函数
            reduce_func: Reduce函数
            
        Returns:
            处理结果字典
        """
        try:
            if not self.config.enable_mapreduce:
                # 简化处理
                mapped = [map_func(item) for item in data]
                grouped = defaultdict(list)
                for key, value in mapped:
                    grouped[key].append(value)
                
                return {key: reduce_func(key, values) for key, values in grouped.items()}
            
            self.logger.info(f"开始MapReduce处理，数据量: {len(data)}")
            
            # Map阶段
            mapped_results = []
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [executor.submit(map_func, item) for item in data]
                for future in as_completed(futures):
                    mapped_results.append(future.result())
            
            # Shuffle阶段
            grouped = defaultdict(list)
            for key, value in mapped_results:
                grouped[key].append(value)
            
            # Reduce阶段
            result = {}
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {executor.submit(reduce_func, key, values): key 
                          for key, values in grouped.items()}
                for future in as_completed(futures):
                    key = futures[future]
                    result[key] = future.result()
            
            self.logger.info(f"MapReduce处理完成，结果数量: {len(result)}")
            return result
            
        except Exception as e:
            self.logger.error(f"MapReduce处理失败: {str(e)}")
            raise OptimizationError(f"MapReduce处理失败: {str(e)}")

class StreamingProcessor:
    """流数据处理器类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.processors: List[Callable[[Any], Any]] = []
    
    def add_processor(self, processor: Callable[[Any], Any]) -> None:
        """添加流处理器"""
        self.processors.append(processor)
    
    def process_stream(self, data_stream: Iterator[Any]) -> Iterator[Any]:
        """处理数据流
        
        Args:
            data_stream: 数据流
            
        Yields:
            处理后的数据
        """
        try:
            if not self.config.enable_streaming:
                # 简化处理
                for item in data_stream:
                    processed = item
                    for processor in self.processors:
                        processed = processor(processed)
                    yield processed
                return
            
            self.logger.info("开始流数据处理")
            
            for item in data_stream:
                try:
                    processed = item
                    for processor in self.processors:
                        processed = processor(processed)
                    yield processed
                except Exception as e:
                    self.logger.warning(f"流数据处理项失败: {str(e)}")
                    continue
            
            self.logger.info("流数据处理完成")
            
        except Exception as e:
            self.logger.error(f"流数据处理失败: {str(e)}")
            raise OptimizationError(f"流数据处理失败: {str(e)}")

class AsyncDataOptimizer:
    """异步数据优化器类"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.semaphore = asyncio.Semaphore(config.max_concurrent_tasks)
        self.tasks: Dict[str, asyncio.Task] = {}
    
    async def optimize_data_async(self, data: List[Dict[str, Any]], 
                                 operations: List[str]) -> OptimizationResult:
        """异步优化数据
        
        Args:
            data: 待优化的数据
            operations: 操作列表
            
        Returns:
            优化结果
        """
        try:
            if not self.config.enable_async:
                # 同步执行
                return await self._optimize_sync(data, operations)
            
            self.logger.info(f"开始异步数据优化，操作: {operations}")
            
            start_time = time.time()
            original_size = len(json.dumps(data, ensure_ascii=False))
            
            # 创建异步任务
            tasks = []
            for operation in operations:
                task = asyncio.create_task(self._execute_operation_async(data, operation))
                tasks.append(task)
            
            # 等待所有任务完成
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            processing_time = time.time() - start_time
            optimized_size = len(json.dumps(data, ensure_ascii=False))
            
            # 统计成功操作数
            successful_ops = sum(1 for result in results if not isinstance(result, Exception))
            
            result = OptimizationResult(
                success=True,
                original_size=original_size,
                optimized_size=optimized_size,
                compression_ratio=original_size / optimized_size if optimized_size > 0 else 1.0,
                processing_time=processing_time,
                operations_count=successful_ops,
                metadata={"operations": operations, "async": True}
            )
            
            self.logger.info(f"异步数据优化完成，耗时: {processing_time:.2f}秒")
            return result
            
        except Exception as e:
            self.logger.error(f"异步数据优化失败: {str(e)}")
            return OptimizationResult(
                success=False,
                original_size=0,
                optimized_size=0,
                compression_ratio=0.0,
                processing_time=0.0,
                operations_count=0,
                error_message=str(e)
            )
    
    async def _execute_operation_async(self, data: List[Dict[str, Any]], 
                                      operation: str) -> Any:
        """异步执行单个操作"""
        async with self.semaphore:
            # 模拟异步操作
            await asyncio.sleep(0.1)
            
            if operation == "compress":
                compressor = Compressor(self.config)
                return compressor.compress_data(json.dumps(data, ensure_ascii=False))
            elif operation == "deduplicate":
                cleaner = DataCleaner(self.config)
                return cleaner.deduplicate_data(data)
            elif operation == "validate":
                validator = DataValidator(self.config)
                return validator.validate_data(data, {})
            else:
                return data
    
    async def _optimize_sync(self, data: List[Dict[str, Any]], 
                           operations: List[str]) -> OptimizationResult:
        """同步优化数据"""
        start_time = time.time()
        original_size = len(json.dumps(data, ensure_ascii=False))
        
        # 同步执行操作
        for operation in operations:
            if operation == "compress":
                compressor = Compressor(self.config)
                compressed_data = compressor.compress_data(json.dumps(data, ensure_ascii=False))
                data = json.loads(compressor.decompress_data(compressed_data))
            elif operation == "deduplicate":
                cleaner = DataCleaner(self.config)
                data = cleaner.deduplicate_data(data)
        
        processing_time = time.time() - start_time
        optimized_size = len(json.dumps(data, ensure_ascii=False))
        
        return OptimizationResult(
            success=True,
            original_size=original_size,
            optimized_size=optimized_size,
            compression_ratio=original_size / optimized_size if optimized_size > 0 else 1.0,
            processing_time=processing_time,
            operations_count=len(operations),
            metadata={"operations": operations, "async": False}
        )

class DataOptimizer:
    """O5数据优化器主类
    
    提供全面的数据优化功能，包括存储优化、访问优化、传输优化、
    清洗优化、大数据处理优化、质量优化和异步处理。
    
    Attributes:
        config: 优化配置
        compressor: 数据压缩器
        encoder: 数据编码器
        sharder: 数据分片器
        indexer: 数据索引器
        cache: 数据缓存
        validator: 数据验证器
        monitor: 数据监控器
        repairer: 数据修复器
        map_reduce: MapReduce处理器
        streaming: 流数据处理器
        async_optimizer: 异步优化器
        logger: 日志记录器
    """
    
    def __init__(self, config: OptimizationConfig = None):
        """初始化数据优化器
        
        Args:
            config: 优化配置，如果为None则使用默认配置
        """
        self.config = config or OptimizationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 初始化各个组件
        self.compressor = Compressor(self.config)
        self.encoder = DataEncoder(self.config)
        self.sharder = DataSharder(self.config)
        self.indexer = DataIndexer(self.config)
        self.cache = DataCache(self.config)
        self.validator = DataValidator(self.config)
        self.monitor = DataMonitor(self.config)
        self.repairer = DataRepairer(self.config)
        self.map_reduce = MapReduceProcessor(self.config)
        self.streaming = StreamingProcessor(self.config)
        self.async_optimizer = AsyncDataOptimizer(self.config)
        
        self.logger.info("O5数据优化器初始化完成")
    
    def optimize_storage(self, data: List[Dict[str, Any]], 
                        shard_key: str = None) -> Dict[str, bytes]:
        """优化数据存储
        
        Args:
            data: 待优化的数据
            shard_key: 分片键
            
        Returns:
            优化后的数据分片字典
        """
        try:
            self.logger.info(f"开始存储优化，数据量: {len(data)}")
            start_time = time.time()
            
            # 1. 数据分片
            if self.config.enable_sharding:
                shards = self.sharder.shard_data(data, shard_key)
            else:
                shards = {"default": data}
            
            # 2. 数据压缩和编码
            optimized_shards = {}
            for shard_id, shard_data in shards.items():
                # 编码数据
                encoded_data = self.encoder.encode_data(shard_data)
                
                # 压缩数据
                compressed_data = self.compressor.compress_data(encoded_data)
                
                optimized_shards[shard_id] = compressed_data
            
            processing_time = time.time() - start_time
            original_size = len(json.dumps(data, ensure_ascii=False))
            optimized_size = sum(len(shard) for shard in optimized_shards.values())
            
            self.logger.info(f"存储优化完成，原始大小: {original_size}, "
                           f"优化后大小: {optimized_size}, "
                           f"压缩比: {original_size/optimized_size:.2f}, "
                           f"耗时: {processing_time:.2f}秒")
            
            return optimized_shards
            
        except Exception as e:
            self.logger.error(f"存储优化失败: {str(e)}")
            raise OptimizationError(f"存储优化失败: {str(e)}")
    
    def optimize_access(self, data: List[Dict[str, Any]], 
                       index_fields: List[str] = None) -> bool:
        """优化数据访问
        
        Args:
            data: 待优化的数据
            index_fields: 索引字段列表
            
        Returns:
            是否优化成功
        """
        try:
            self.logger.info(f"开始访问优化，数据量: {len(data)}")
            
            # 创建索引
            if self.config.enable_indexing and index_fields:
                for field in index_fields:
                    self.indexer.create_index(data, field)
            
            # 清空缓存
            self.cache.clear()
            
            self.logger.info("访问优化完成")
            return True
            
        except Exception as e:
            self.logger.error(f"访问优化失败: {str(e)}")
            raise OptimizationError(f"访问优化失败: {str(e)}")
    
    def optimize_transmission(self, data: List[Dict[str, Any]], 
                            batch_size: int = None) -> List[bytes]:
        """优化数据传输
        
        Args:
            data: 待传输的数据
            batch_size: 批处理大小
            
        Returns:
            传输批次列表
        """
        try:
            if batch_size is None:
                batch_size = self.config.batch_size
            
            self.logger.info(f"开始传输优化，批处理大小: {batch_size}")
            
            # 批量处理数据
            batches = []
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                # 编码和压缩批次数据
                encoded_batch = self.encoder.encode_data(batch)
                compressed_batch = self.compressor.compress_data(encoded_batch)
                
                batches.append(compressed_batch)
            
            self.logger.info(f"传输优化完成，批次数: {len(batches)}")
            return batches
            
        except Exception as e:
            self.logger.error(f"传输优化失败: {str(e)}")
            raise OptimizationError(f"传输优化失败: {str(e)}")
    
    def optimize_cleaning(self, data: List[Dict[str, Any]], 
                         clean_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """优化数据清洗
        
        Args:
            data: 待清洗的数据
            clean_config: 清洗配置
            
        Returns:
            清洗后的数据
        """
        try:
            if clean_config is None:
                clean_config = {}
            
            self.logger.info(f"开始清洗优化，数据量: {len(data)}")
            
            cleaner = DataCleaner(self.config)
            cleaned_data = data
            
            # 数据去重
            if self.config.enable_deduplication:
                key_fields = clean_config.get('key_fields')
                cleaned_data = cleaner.deduplicate_data(cleaned_data, key_fields)
            
            # 数据填充
            if self.config.enable_data_filling:
                fill_strategy = clean_config.get('fill_strategy', 'forward_fill')
                cleaned_data = cleaner.fill_missing_data(cleaned_data, fill_strategy)
            
            # 数据转换
            if self.config.enable_data_transformation:
                transformations = clean_config.get('transformations', {})
                if transformations:
                    cleaned_data = cleaner.transform_data(cleaned_data, transformations)
            
            self.logger.info(f"清洗优化完成，原始数据量: {len(data)}, "
                           f"清洗后数据量: {len(cleaned_data)}")
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"清洗优化失败: {str(e)}")
            raise OptimizationError(f"清洗优化失败: {str(e)}")
    
    def optimize_big_data(self, data: List[Any], 
                         operation: str, **kwargs) -> Any:
        """优化大数据处理
        
        Args:
            data: 待处理的数据
            operation: 操作类型 ('map_reduce', 'streaming')
            **kwargs: 操作参数
            
        Returns:
            处理结果
        """
        try:
            self.logger.info(f"开始大数据处理，操作: {operation}")
            
            if operation == 'map_reduce':
                map_func = kwargs.get('map_func')
                reduce_func = kwargs.get('reduce_func')
                
                if not map_func or not reduce_func:
                    raise ValueError("MapReduce操作需要map_func和reduce_func参数")
                
                return self.map_reduce.map_reduce(data, map_func, reduce_func)
            
            elif operation == 'streaming':
                # 创建数据流
                data_stream = iter(data)
                
                # 添加流处理器
                processors = kwargs.get('processors', [])
                for processor in processors:
                    self.streaming.add_processor(processor)
                
                # 处理流数据
                return list(self.streaming.process_stream(data_stream))
            
            else:
                raise ValueError(f"不支持的大数据操作: {operation}")
            
        except Exception as e:
            self.logger.error(f"大数据处理失败: {str(e)}")
            raise OptimizationError(f"大数据处理失败: {str(e)}")
    
    def optimize_quality(self, data: List[Dict[str, Any]], 
                        quality_config: Dict[str, Any] = None) -> DataQualityMetrics:
        """优化数据质量
        
        Args:
            data: 待处理的数据
            quality_config: 质量配置
            
        Returns:
            数据质量指标
        """
        try:
            if quality_config is None:
                quality_config = {}
            
            self.logger.info(f"开始质量优化，数据量: {len(data)}")
            
            # 数据验证
            validation_rules = quality_config.get('validation_rules', {})
            if validation_rules:
                is_valid, errors = self.validator.validate_data(data, validation_rules)
                if not is_valid:
                    self.logger.warning(f"数据验证失败: {errors}")
            
            # 数据质量监控
            metrics = self.monitor.monitor_data_quality(data)
            
            # 数据修复
            repair_rules = quality_config.get('repair_rules', {})
            if repair_rules and metrics.overall_score < 0.8:  # 如果质量分数低于0.8，进行修复
                data = self.repairer.repair_data(data, repair_rules)
                # 重新监控质量
                metrics = self.monitor.monitor_data_quality(data)
            
            self.logger.info(f"质量优化完成，总体质量分数: {metrics.overall_score:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"质量优化失败: {str(e)}")
            raise OptimizationError(f"质量优化失败: {str(e)}")
    
    async def optimize_async(self, data: List[Dict[str, Any]], 
                           operations: List[str]) -> OptimizationResult:
        """异步优化数据
        
        Args:
            data: 待优化的数据
            operations: 操作列表
            
        Returns:
            优化结果
        """
        try:
            return await self.async_optimizer.optimize_data_async(data, operations)
            
        except Exception as e:
            self.logger.error(f"异步优化失败: {str(e)}")
            raise OptimizationError(f"异步优化失败: {str(e)}")
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告
        
        Returns:
            优化报告字典
        """
        try:
            report = {
                "optimizer_info": {
                    "version": "1.0.0",
                    "config": self.config.__dict__,
                    "components": {
                        "compressor": self.compressor.__class__.__name__,
                        "encoder": self.encoder.__class__.__name__,
                        "sharder": self.sharder.__class__.__name__,
                        "indexer": self.indexer.__class__.__name__,
                        "cache": self.cache.__class__.__name__,
                        "validator": self.validator.__class__.__name__,
                        "monitor": self.monitor.__class__.__name__,
                        "repairer": self.repairer.__class__.__name__,
                        "map_reduce": self.map_reduce.__class__.__name__,
                        "streaming": self.streaming.__class__.__name__,
                        "async_optimizer": self.async_optimizer.__class__.__name__
                    }
                },
                "performance_metrics": {
                    "cache_size": len(self.cache.cache),
                    "indices_count": len(self.indexer.indices),
                    "metrics_history_count": len(self.monitor.metrics_history)
                },
                "timestamp": datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"获取优化报告失败: {str(e)}")
            raise OptimizationError(f"获取优化报告失败: {str(e)}")
    
    def cleanup(self) -> None:
        """清理资源"""
        try:
            self.logger.info("开始清理资源")
            
            # 清空缓存
            self.cache.clear()
            
            # 清空索引
            self.indexer.indices.clear()
            
            # 清空监控历史
            self.monitor.metrics_history.clear()
            
            # 强制垃圾回收
            gc.collect()
            
            self.logger.info("资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {str(e)}")

# 使用示例和测试代码
def create_sample_data(count: int = 1000) -> List[Dict[str, Any]]:
    """创建示例数据
    
    Args:
        count: 数据条数
        
    Returns:
        示例数据列表
    """
    import random
    import string
    
    data = []
    for i in range(count):
        item = {
            "id": i,
            "name": f"用户_{i}",
            "age": random.randint(18, 80),
            "email": f"user{i}@example.com",
            "score": random.uniform(0, 100),
            "timestamp": datetime.now().isoformat(),
            "category": random.choice(["A", "B", "C"]),
            "is_active": random.choice([True, False])
        }
        data.append(item)
    
    return data

def example_usage():
    """使用示例"""
    print("=== O5数据优化器使用示例 ===\n")
    
    # 1. 创建配置
    config = OptimizationConfig(
        enable_sharding=True,
        shard_size=100,
        compression_type=CompressionType.GZIP,
        encoding_format=DataFormat.JSON,
        enable_caching=True,
        cache_size=500,
        enable_indexing=True,
        batch_size=50,
        enable_deduplication=True,
        enable_data_filling=True,
        enable_async=True,
        max_workers=2
    )
    
    # 2. 创建优化器
    optimizer = DataOptimizer(config)
    
    # 3. 创建示例数据
    print("1. 创建示例数据...")
    sample_data = create_sample_data(500)
    print(f"   创建了 {len(sample_data)} 条数据")
    
    # 4. 存储优化
    print("\n2. 执行存储优化...")
    storage_result = optimizer.optimize_storage(sample_data, "category")
    print(f"   分片数量: {len(storage_result)}")
    for shard_id, shard_data in storage_result.items():
        print(f"   分片 {shard_id}: {len(shard_data)} 字节")
    
    # 5. 访问优化
    print("\n3. 执行访问优化...")
    access_result = optimizer.optimize_access(sample_data, ["id", "category"])
    print(f"   索引创建: {'成功' if access_result else '失败'}")
    
    # 6. 传输优化
    print("\n4. 执行传输优化...")
    transmission_batches = optimizer.optimize_transmission(sample_data, 100)
    print(f"   传输批次: {len(transmission_batches)}")
    
    # 7. 清洗优化
    print("\n5. 执行清洗优化...")
    clean_config = {
        "key_fields": ["id", "email"],
        "fill_strategy": "forward_fill",
        "transformations": {
            "age": lambda x: int(x) if x else 0,
            "score": lambda x: round(x, 2) if x else 0.0
        }
    }
    cleaned_data = optimizer.optimize_cleaning(sample_data, clean_config)
    print(f"   清洗后数据量: {len(cleaned_data)}")
    
    # 8. 大数据处理
    print("\n6. 执行大数据处理...")
    
    # MapReduce示例
    map_func = lambda x: (x["category"], 1)
    reduce_func = lambda key, values: sum(values)
    map_reduce_result = optimizer.optimize_big_data(
        sample_data, "map_reduce", 
        map_func=map_func, reduce_func=reduce_func
    )
    print(f"   MapReduce结果: {map_reduce_result}")
    
    # 流处理示例
    def uppercase_name(item):
        item["name"] = item["name"].upper()
        return item
    
    streaming_result = optimizer.optimize_big_data(
        sample_data[:10], "streaming", 
        processors=[uppercase_name]
    )
    print(f"   流处理结果: {streaming_result[0]['name'] if streaming_result else '无结果'}")
    
    # 9. 质量优化
    print("\n7. 执行质量优化...")
    quality_config = {
        "validation_rules": {
            "age": lambda x: 0 <= x <= 150,
            "score": lambda x: 0 <= x <= 100,
            "email": lambda x: "@" in str(x)
        },
        "repair_rules": {
            "age": lambda x: max(0, min(150, x)) if x is not None else 0,
            "score": lambda x: max(0, min(100, x)) if x is not None else 0.0
        }
    }
    quality_metrics = optimizer.optimize_quality(sample_data, quality_config)
    print(f"   质量指标:")
    print(f"     完整性: {quality_metrics.completeness:.3f}")
    print(f"     准确性: {quality_metrics.accuracy:.3f}")
    print(f"     一致性: {quality_metrics.consistency:.3f}")
    print(f"     时效性: {quality_metrics.timeliness:.3f}")
    print(f"     有效性: {quality_metrics.validity:.3f}")
    print(f"     唯一性: {quality_metrics.uniqueness:.3f}")
    print(f"     总体分数: {quality_metrics.overall_score:.3f}")
    
    # 10. 异步优化
    print("\n8. 执行异步优化...")
    
    async def async_example():
        operations = ["compress", "deduplicate", "validate"]
        async_result = await optimizer.optimize_async(sample_data[:100], operations)
        print(f"   异步优化结果:")
        print(f"     成功: {async_result.success}")
        print(f"     原始大小: {async_result.original_size}")
        print(f"     优化后大小: {async_result.optimized_size}")
        print(f"     压缩比: {async_result.compression_ratio:.2f}")
        print(f"     处理时间: {async_result.processing_time:.2f}秒")
        print(f"     操作数量: {async_result.operations_count}")
    
    asyncio.run(async_example())
    
    # 11. 获取优化报告
    print("\n9. 获取优化报告...")
    report = optimizer.get_optimization_report()
    print(f"   优化器版本: {report['optimizer_info']['version']}")
    print(f"   缓存大小: {report['performance_metrics']['cache_size']}")
    print(f"   索引数量: {report['performance_metrics']['indices_count']}")
    
    # 12. 清理资源
    print("\n10. 清理资源...")
    optimizer.cleanup()
    print("    资源清理完成")
    
    print("\n=== 示例执行完成 ===")

if __name__ == "__main__":
    # 运行示例
    example_usage()