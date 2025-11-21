#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
I3 数据接口适配器

实现多数据源适配、数据格式转换、数据验证清洗、批量处理、
增量同步、缓存机制、错误重试、数据血缘追踪和性能监控等功能。

Author: AI Assistant
Date: 2025-11-05
Version: 1.0.0
"""

import asyncio
import json
import csv
import xml.etree.ElementTree as ET
import sqlite3
import aiohttp
import aiofiles
import hashlib
import time
import logging
import threading
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd
import numpy as np
from functools import wraps
import pickle
import gzip
import weakref
import psutil
import gc


# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataSourceType(Enum):
    """数据源类型枚举"""
    DATABASE = "database"
    API = "api"
    FILE = "file"
    CACHE = "cache"


class DataFormat(Enum):
    """数据格式枚举"""
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    PARQUET = "parquet"
    EXCEL = "excel"


class DataQuality(Enum):
    """数据质量等级"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class DataRecord:
    """数据记录结构"""
    id: str
    data: Any
    timestamp: datetime
    source: str
    format: DataFormat
    quality: DataQuality = DataQuality.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)
    lineage: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'id': self.id,
            'data': self.data,
            'timestamp': self.timestamp.isoformat(),
            'source': self.source,
            'format': self.format.value,
            'quality': self.quality.value,
            'metadata': self.metadata,
            'lineage': self.lineage
        }


@dataclass
class PerformanceMetrics:
    """性能指标"""
    operation: str
    start_time: float
    end_time: Optional[float] = None
    records_processed: int = 0
    records_successful: int = 0
    records_failed: int = 0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    
    @property
    def duration(self) -> float:
        """操作持续时间"""
        if self.end_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.records_processed == 0:
            return 0.0
        return self.records_successful / self.records_processed
    
    @property
    def throughput(self) -> float:
        """吞吐量（记录/秒）"""
        if self.duration == 0:
            return 0.0
        return self.records_processed / self.duration


class DataSourceAdapter(ABC):
    """数据源适配器基类"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """连接数据源"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """断开数据源连接"""
        pass
    
    @abstractmethod
    async def read(self, query: str = None, **kwargs) -> List[DataRecord]:
        """读取数据"""
        pass
    
    @abstractmethod
    async def write(self, records: List[DataRecord]) -> bool:
        """写入数据"""
        pass


class DatabaseAdapter(DataSourceAdapter):
    """数据库适配器"""
    
    def __init__(self, connection_string: str, table_name: str = None):
        self.connection_string = connection_string
        self.table_name = table_name
        self.connection = None
    
    async def connect(self) -> bool:
        """连接数据库"""
        try:
            self.connection = sqlite3.connect(self.connection_string)
            return True
        except Exception as e:
            logger.error(f"数据库连接失败: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """断开数据库连接"""
        try:
            if self.connection:
                self.connection.close()
            return True
        except Exception as e:
            logger.error(f"数据库断开连接失败: {e}")
            return False
    
    async def read(self, query: str = None, **kwargs) -> List[DataRecord]:
        """从数据库读取数据"""
        if not self.connection:
            await self.connect()
        
        query = query or f"SELECT * FROM {self.table_name}"
        cursor = self.connection.cursor()
        
        try:
            cursor.execute(query)
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description]
            
            records = []
            for row in rows:
                data = dict(zip(columns, row))
                record = DataRecord(
                    id=hashlib.md5(str(data).encode()).hexdigest(),
                    data=data,
                    timestamp=datetime.now(),
                    source=f"database:{self.connection_string}",
                    format=DataFormat.JSON
                )
                records.append(record)
            
            return records
        except Exception as e:
            logger.error(f"数据库读取失败: {e}")
            return []
    
    async def write(self, records: List[DataRecord]) -> bool:
        """写入数据库"""
        if not self.connection:
            await self.connect()
        
        cursor = self.connection.cursor()
        
        try:
            for record in records:
                if isinstance(record.data, dict):
                    columns = list(record.data.keys())
                    placeholders = ','.join(['?' for _ in columns])
                    values = list(record.data.values())
                    
                    query = f"INSERT OR REPLACE INTO {self.table_name} ({','.join(columns)}) VALUES ({placeholders})"
                    cursor.execute(query, values)
            
            self.connection.commit()
            return True
        except Exception as e:
            logger.error(f"数据库写入失败: {e}")
            self.connection.rollback()
            return False


class APIAdapter(DataSourceAdapter):
    """API适配器"""
    
    def __init__(self, base_url: str, headers: Dict[str, str] = None, timeout: int = 30):
        self.base_url = base_url
        self.headers = headers or {}
        self.timeout = timeout
        self.session = None
    
    async def connect(self) -> bool:
        """初始化API会话"""
        try:
            self.session = aiohttp.ClientSession(
                headers=self.headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            return True
        except Exception as e:
            logger.error(f"API连接失败: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """关闭API会话"""
        try:
            if self.session:
                await self.session.close()
            return True
        except Exception as e:
            logger.error(f"API断开连接失败: {e}")
            return False
    
    async def read(self, endpoint: str = "", params: Dict = None, **kwargs) -> List[DataRecord]:
        """从API读取数据"""
        if not self.session:
            await self.connect()
        
        url = f"{self.base_url}/{endpoint}" if endpoint else self.base_url
        
        try:
            async with self.session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    # 处理不同的API响应格式
                    if isinstance(data, list):
                        records_data = data
                    elif isinstance(data, dict) and 'data' in data:
                        records_data = data['data']
                    else:
                        records_data = [data]
                    
                    records = []
                    for item in records_data:
                        record = DataRecord(
                            id=hashlib.md5(str(item).encode()).hexdigest(),
                            data=item,
                            timestamp=datetime.now(),
                            source=f"api:{self.base_url}",
                            format=DataFormat.JSON
                        )
                        records.append(record)
                    
                    return records
                else:
                    logger.error(f"API请求失败，状态码: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"API读取失败: {e}")
            return []
    
    async def write(self, records: List[DataRecord], endpoint: str = "", **kwargs) -> bool:
        """向API写入数据"""
        if not self.session:
            await self.connect()
        
        url = f"{self.base_url}/{endpoint}" if endpoint else self.base_url
        
        try:
            for record in records:
                async with self.session.post(url, json=record.data) as response:
                    if response.status not in [200, 201]:
                        logger.error(f"API写入失败，状态码: {response.status}")
                        return False
            return True
        except Exception as e:
            logger.error(f"API写入失败: {e}")
            return False


class FileAdapter(DataSourceAdapter):
    """文件适配器"""
    
    def __init__(self, file_path: str, encoding: str = 'utf-8'):
        self.file_path = Path(file_path)
        self.encoding = encoding
    
    async def connect(self) -> bool:
        """检查文件是否存在"""
        return self.file_path.exists()
    
    async def disconnect(self) -> bool:
        """文件适配器无需断开连接"""
        return True
    
    async def read(self, format_type: DataFormat = DataFormat.JSON, **kwargs) -> List[DataRecord]:
        """从文件读取数据"""
        if not await self.connect():
            logger.error(f"文件不存在: {self.file_path}")
            return []
        
        records = []
        
        try:
            async with aiofiles.open(self.file_path, 'r', encoding=self.encoding) as f:
                content = await f.read()
                
                if format_type == DataFormat.JSON:
                    data = json.loads(content)
                    if isinstance(data, list):
                        records_data = data
                    else:
                        records_data = [data]
                elif format_type == DataFormat.CSV:
                    # 简化的CSV处理
                    lines = content.strip().split('\n')
                    if lines:
                        headers = lines[0].split(',')
                        records_data = []
                        for line in lines[1:]:
                            values = line.split(',')
                            if len(values) == len(headers):
                                records_data.append(dict(zip(headers, values)))
                else:
                    logger.error(f"不支持的文件格式: {format_type}")
                    return []
                
                for item in records_data:
                    record = DataRecord(
                        id=hashlib.md5(str(item).encode()).hexdigest(),
                        data=item,
                        timestamp=datetime.now(),
                        source=f"file:{self.file_path}",
                        format=format_type
                    )
                    records.append(record)
                
                return records
        except Exception as e:
            logger.error(f"文件读取失败: {e}")
            return []
    
    async def write(self, records: List[DataRecord], format_type: DataFormat = DataFormat.JSON) -> bool:
        """写入文件"""
        try:
            content = ""
            
            if format_type == DataFormat.JSON:
                data = [record.data for record in records]
                content = json.dumps(data, ensure_ascii=False, indent=2)
            elif format_type == DataFormat.CSV:
                if records:
                    headers = list(records[0].data.keys()) if isinstance(records[0].data, dict) else []
                    content = ','.join(headers) + '\n'
                    for record in records:
                        if isinstance(record.data, dict):
                            values = [str(record.data.get(h, '')) for h in headers]
                            content += ','.join(values) + '\n'
            else:
                logger.error(f"不支持的文件格式: {format_type}")
                return False
            
            async with aiofiles.open(self.file_path, 'w', encoding=self.encoding) as f:
                await f.write(content)
            
            return True
        except Exception as e:
            logger.error(f"文件写入失败: {e}")
            return False


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_schema(data: Any, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """验证数据模式"""
        errors = []
        
        if not isinstance(data, dict):
            return False, ["数据必须是字典格式"]
        
        for field, rules in schema.items():
            if field not in data:
                if rules.get('required', False):
                    errors.append(f"缺少必需字段: {field}")
                continue
            
            value = data[field]
            
            # 类型验证
            expected_type = rules.get('type')
            if expected_type and not isinstance(value, expected_type):
                errors.append(f"字段 {field} 类型错误，期望: {expected_type}，实际: {type(value)}")
            
            # 范围验证
            if isinstance(value, (int, float)):
                min_val = rules.get('min')
                max_val = rules.get('max')
                if min_val is not None and value < min_val:
                    errors.append(f"字段 {field} 值 {value} 小于最小值 {min_val}")
                if max_val is not None and value > max_val:
                    errors.append(f"字段 {field} 值 {value} 大于最大值 {max_val}")
            
            # 长度验证
            if isinstance(value, (str, list, dict)):
                min_len = rules.get('min_length')
                max_len = rules.get('max_length')
                length = len(value)
                if min_len is not None and length < min_len:
                    errors.append(f"字段 {field} 长度 {length} 小于最小长度 {min_len}")
                if max_len is not None and length > max_len:
                    errors.append(f"字段 {field} 长度 {length} 大于最大长度 {max_len}")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def clean_data(data: Any) -> Any:
        """数据清洗"""
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                if value is not None:
                    if isinstance(value, str):
                        # 清理字符串
                        cleaned_value = value.strip()
                        if cleaned_value:  # 只保留非空字符串
                            cleaned[key] = cleaned_value
                    else:
                        cleaned[key] = DataValidator.clean_data(value)
            return cleaned
        elif isinstance(data, list):
            return [DataValidator.clean_data(item) for item in data if item is not None]
        else:
            return data


class DataConverter:
    """数据格式转换器"""
    
    @staticmethod
    def convert_format(data: Any, from_format: DataFormat, to_format: DataFormat) -> Any:
        """转换数据格式"""
        if from_format == to_format:
            return data
        
        # 先转换为标准格式（字典列表）
        if from_format == DataFormat.JSON:
            standard_data = data if isinstance(data, list) else [data]
        elif from_format == DataFormat.XML:
            standard_data = DataConverter._xml_to_dict(data)
        elif from_format == DataFormat.CSV:
            standard_data = DataConverter._csv_to_dict(data)
        else:
            raise ValueError(f"不支持的源格式: {from_format}")
        
        # 转换为目标格式
        if to_format == DataFormat.JSON:
            return standard_data
        elif to_format == DataFormat.XML:
            return DataConverter._dict_to_xml(standard_data)
        elif to_format == DataFormat.CSV:
            return DataConverter._dict_to_csv(standard_data)
        else:
            raise ValueError(f"不支持的目标格式: {to_format}")
    
    @staticmethod
    def _xml_to_dict(xml_data: str) -> List[Dict[str, Any]]:
        """XML转字典"""
        try:
            root = ET.fromstring(xml_data)
            return [DataConverter._element_to_dict(root)]
        except Exception as e:
            logger.error(f"XML解析失败: {e}")
            return []
    
    @staticmethod
    def _element_to_dict(element: ET.Element) -> Dict[str, Any]:
        """XML元素转字典"""
        result = {}
        
        # 处理属性
        if element.attrib:
            result['@attributes'] = element.attrib
        
        # 处理文本内容
        if element.text and element.text.strip():
            if len(element) == 0:  # 叶子节点
                return element.text.strip()
            else:
                result['#text'] = element.text.strip()
        
        # 处理子元素
        for child in element:
            child_data = DataConverter._element_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_data)
            else:
                result[child.tag] = child_data
        
        return result
    
    @staticmethod
    def _dict_to_xml(data: List[Dict[str, Any]]) -> str:
        """字典转XML"""
        root = ET.Element("data")
        for item in data:
            item_element = ET.SubElement(root, "item")
            DataConverter._dict_to_element(item, item_element)
        
        return ET.tostring(root, encoding='unicode')
    
    @staticmethod
    def _dict_to_element(data: Dict[str, Any], parent: ET.Element):
        """字典转XML元素"""
        for key, value in data.items():
            if key == '@attributes':
                parent.attrib.update(value)
            elif key == '#text':
                parent.text = str(value)
            elif isinstance(value, dict):
                child = ET.SubElement(parent, key)
                DataConverter._dict_to_element(value, child)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        child = ET.SubElement(parent, key)
                        DataConverter._dict_to_element(item, child)
                    else:
                        child = ET.SubElement(parent, key)
                        child.text = str(item)
            else:
                child = ET.SubElement(parent, key)
                child.text = str(value)
    
    @staticmethod
    def _csv_to_dict(csv_data: str) -> List[Dict[str, Any]]:
        """CSV转字典"""
        try:
            lines = csv_data.strip().split('\n')
            if len(lines) < 2:
                return []
            
            reader = csv.DictReader(lines)
            return list(reader)
        except Exception as e:
            logger.error(f"CSV解析失败: {e}")
            return []
    
    @staticmethod
    def _dict_to_csv(data: List[Dict[str, Any]]) -> str:
        """字典转CSV"""
        if not data:
            return ""
        
        import io
        output = io.StringIO()
        fieldnames = list(data[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            writer.writerow(row)
        
        return output.getvalue()


class DataCache:
    """数据缓存管理器"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl  # 缓存生存时间（秒）
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._access_times: Dict[str, datetime] = {}
        self._lock = threading.RLock()
    
    def _generate_key(self, source: str, query: str, params: Dict) -> str:
        """生成缓存键"""
        key_data = f"{source}:{query}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, source: str, query: str = "", params: Dict = None) -> Optional[Any]:
        """获取缓存数据"""
        if params is None:
            params = {}
        
        key = self._generate_key(source, query, params)
        
        with self._lock:
            if key in self._cache:
                data, timestamp = self._cache[key]
                
                # 检查是否过期
                if datetime.now() - timestamp < timedelta(seconds=self.ttl):
                    self._access_times[key] = datetime.now()
                    return data
                else:
                    # 清理过期缓存
                    del self._cache[key]
                    del self._access_times[key]
        
        return None
    
    def set(self, source: str, data: Any, query: str = "", params: Dict = None) -> None:
        """设置缓存数据"""
        if params is None:
            params = {}
        
        key = self._generate_key(source, query, params)
        
        with self._lock:
            # 如果缓存已满，删除最久未访问的项
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            
            self._cache[key] = (data, datetime.now())
            self._access_times[key] = datetime.now()
    
    def _evict_oldest(self) -> None:
        """清理最久未访问的缓存项"""
        if not self._access_times:
            return
        
        oldest_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[oldest_key]
        del self._access_times[oldest_key]
    
    def clear(self) -> None:
        """清空缓存"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self._lock:
            return {
                'size': len(self._cache),
                'max_size': self.max_size,
                'hit_rate': getattr(self, '_hit_rate', 0.0),
                'ttl': self.ttl
            }


class RetryManager:
    """错误重试管理器"""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.0, 
                 max_backoff: float = 60.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.max_backoff = max_backoff
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """执行函数并处理重试"""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt == self.max_retries:
                    logger.error(f"函数执行失败，已达到最大重试次数: {func.__name__}")
                    break
                
                # 计算等待时间
                wait_time = min(
                    self.backoff_factor * (2 ** attempt),
                    self.max_backoff
                )
                
                logger.warning(f"函数执行失败，{wait_time}秒后重试 (第{attempt + 1}次): {e}")
                await asyncio.sleep(wait_time)
        
        raise last_exception


class DataLineageTracker:
    """数据血缘追踪器"""
    
    def __init__(self):
        self.lineage_graph: Dict[str, List[str]] = {}
        self.transformations: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
    
    def add_lineage(self, source_id: str, target_id: str, 
                   transformation: str = "", metadata: Dict = None):
        """添加血缘关系"""
        with self._lock:
            if source_id not in self.lineage_graph:
                self.lineage_graph[source_id] = []
            
            self.lineage_graph[source_id].append(target_id)
            
            if metadata is None:
                metadata = {}
            
            self.transformations[target_id] = {
                'source': source_id,
                'transformation': transformation,
                'timestamp': datetime.now().isoformat(),
                'metadata': metadata
            }
    
    def get_lineage(self, record_id: str) -> List[str]:
        """获取数据血缘链"""
        with self._lock:
            lineage = []
            visited = set()
            
            def dfs(node: str):
                if node in visited:
                    return
                visited.add(node)
                lineage.append(node)
                
                if node in self.lineage_graph:
                    for child in self.lineage_graph[node]:
                        dfs(child)
            
            dfs(record_id)
            return lineage
    
    def get_impact_analysis(self, record_id: str) -> Dict[str, Any]:
        """影响分析"""
        with self._lock:
            downstream = []
            upstream = []
            
            # 查找下游影响
            def find_downstream(node: str, visited: set = None):
                if visited is None:
                    visited = set()
                if node in visited:
                    return
                visited.add(node)
                
                if node in self.lineage_graph:
                    for child in self.lineage_graph[node]:
                        downstream.append(child)
                        find_downstream(child, visited.copy())
            
            # 查找上游依赖
            for source, targets in self.lineage_graph.items():
                if record_id in targets:
                    upstream.append(source)
            
            find_downstream(record_id)
            
            return {
                'record_id': record_id,
                'upstream_dependencies': upstream,
                'downstream_impacts': downstream,
                'risk_level': 'high' if downstream else 'low'
            }


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.current_metrics: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.RLock()
    
    def start_operation(self, operation: str) -> str:
        """开始操作监控"""
        metric_id = f"{operation}_{int(time.time() * 1000000)}"
        
        with self._lock:
            self.current_metrics[metric_id] = PerformanceMetrics(
                operation=operation,
                start_time=time.time(),
                memory_usage=psutil.Process().memory_info().rss / 1024 / 1024,  # MB
                cpu_usage=psutil.Process().cpu_percent()
            )
        
        return metric_id
    
    def end_operation(self, metric_id: str, records_processed: int = 0,
                     records_successful: int = 0, records_failed: int = 0):
        """结束操作监控"""
        with self._lock:
            if metric_id in self.current_metrics:
                metric = self.current_metrics[metric_id]
                metric.end_time = time.time()
                metric.records_processed = records_processed
                metric.records_successful = records_successful
                metric.records_failed = records_failed
                metric.memory_usage = psutil.Process().memory_info().rss / 1024 / 1024
                metric.cpu_usage = psutil.Process().cpu_percent()
                
                self.metrics_history.append(metric)
                del self.current_metrics[metric_id]
    
    def get_performance_summary(self, operation: str = None, 
                               time_range: int = 3600) -> Dict[str, Any]:
        """获取性能摘要"""
        with self._lock:
            cutoff_time = time.time() - time_range
            
            relevant_metrics = [
                m for m in self.metrics_history
                if m.end_time and m.end_time > cutoff_time
                and (operation is None or m.operation == operation)
            ]
            
            if not relevant_metrics:
                return {'message': '没有找到相关性能数据'}
            
            total_operations = len(relevant_metrics)
            total_duration = sum(m.duration for m in relevant_metrics)
            total_records = sum(m.records_processed for m in relevant_metrics)
            total_successful = sum(m.records_successful for m in relevant_metrics)
            
            return {
                'operation': operation or 'all',
                'time_range': time_range,
                'total_operations': total_operations,
                'avg_duration': total_duration / total_operations,
                'total_records_processed': total_records,
                'overall_success_rate': total_successful / total_records if total_records > 0 else 0,
                'avg_throughput': total_records / total_duration if total_duration > 0 else 0,
                'avg_memory_usage': sum(m.memory_usage for m in relevant_metrics) / total_operations,
                'avg_cpu_usage': sum(m.cpu_usage for m in relevant_metrics) / total_operations
            }


def performance_monitor_decorator(operation_name: str = None):
    """性能监控装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(self, *args, **kwargs):
            monitor = getattr(self, 'performance_monitor', None)
            if not monitor:
                return await func(self, *args, **kwargs)
            
            op_name = operation_name or f"{self.__class__.__name__}.{func.__name__}"
            metric_id = monitor.start_operation(op_name)
            
            try:
                result = await func(self, *args, **kwargs)
                # 假设处理了1条记录（可根据实际情况调整）
                monitor.end_operation(metric_id, records_processed=1, records_successful=1)
                return result
            except Exception as e:
                monitor.end_operation(metric_id, records_processed=1, records_failed=1)
                raise e
        
        return wrapper
    return decorator


class DataInterfaceAdapter:
    """
    数据接口适配器主类
    
    提供统一的数据访问接口，支持多种数据源、格式转换、
    数据验证、批量处理、增量同步、缓存、重试、血缘追踪和性能监控。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据接口适配器
        
        Args:
            config: 配置字典，包含各种适配器配置
        """
        self.config = config
        self.adapters: Dict[str, DataSourceAdapter] = {}
        self.validator = DataValidator()
        self.converter = DataConverter()
        self.cache = DataCache(
            max_size=config.get('cache_max_size', 1000),
            ttl=config.get('cache_ttl', 3600)
        )
        self.retry_manager = RetryManager(
            max_retries=config.get('max_retries', 3),
            backoff_factor=config.get('backoff_factor', 1.0)
        )
        self.lineage_tracker = DataLineageTracker()
        self.performance_monitor = PerformanceMonitor()
        
        self._initialize_adapters()
        logger.info("数据接口适配器初始化完成")
    
    def _initialize_adapters(self):
        """初始化数据源适配器"""
        # 数据库适配器
        if 'database' in self.config:
            db_config = self.config['database']
            self.adapters['database'] = DatabaseAdapter(
                connection_string=db_config.get('connection_string', ':memory:'),
                table_name=db_config.get('table_name', 'data')
            )
        
        # API适配器
        if 'api' in self.config:
            api_config = self.config['api']
            self.adapters['api'] = APIAdapter(
                base_url=api_config.get('base_url', ''),
                headers=api_config.get('headers', {}),
                timeout=api_config.get('timeout', 30)
            )
        
        # 文件适配器
        if 'file' in self.config:
            file_config = self.config['file']
            self.adapters['file'] = FileAdapter(
                file_path=file_config.get('file_path', ''),
                encoding=file_config.get('encoding', 'utf-8')
            )
    
    @performance_monitor_decorator("read_data")
    async def read_data(self, source: str, query: str = "", 
                       format_type: DataFormat = DataFormat.JSON,
                       use_cache: bool = True, **kwargs) -> List[DataRecord]:
        """
        读取数据
        
        Args:
            source: 数据源名称
            query: 查询语句或参数
            format_type: 数据格式
            use_cache: 是否使用缓存
            **kwargs: 其他参数
            
        Returns:
            数据记录列表
        """
        if source not in self.adapters:
            raise ValueError(f"不支持的数据源: {source}")
        
        # 尝试从缓存获取
        if use_cache:
            cached_data = self.cache.get(source, query, kwargs)
            if cached_data:
                logger.info(f"从缓存获取数据: {source}")
                return cached_data
        
        # 从数据源读取
        adapter = self.adapters[source]
        
        try:
            records = await self.retry_manager.execute_with_retry(
                adapter.read, query, **kwargs
            )
            
            # 数据验证和清洗
            validated_records = []
            for record in records:
                # 格式转换
                if record.format != format_type:
                    converted_data = self.converter.convert_format(
                        record.data, record.format, format_type
                    )
                    record.data = converted_data
                    record.format = format_type
                
                # 数据质量评估
                record.quality = self._assess_data_quality(record)
                
                # 添加血缘信息
                record.lineage.append(f"{source}:{datetime.now().isoformat()}")
                
                validated_records.append(record)
            
            # 缓存数据
            if use_cache:
                self.cache.set(source, validated_records, query, kwargs)
            
            logger.info(f"成功读取 {len(validated_records)} 条记录")
            return validated_records
            
        except Exception as e:
            logger.error(f"读取数据失败: {e}")
            raise
    
    @performance_monitor_decorator("write_data")
    async def write_data(self, source: str, records: List[DataRecord], 
                        format_type: DataFormat = DataFormat.JSON, **kwargs) -> bool:
        """
        写入数据
        
        Args:
            source: 数据源名称
            records: 数据记录列表
            format_type: 数据格式
            **kwargs: 其他参数
            
        Returns:
            写入是否成功
        """
        if source not in self.adapters:
            raise ValueError(f"不支持的数据源: {source}")
        
        adapter = self.adapters[source]
        
        try:
            # 数据预处理
            processed_records = []
            for record in records:
                # 格式转换
                if record.format != format_type:
                    converted_data = self.converter.convert_format(
                        record.data, record.format, format_type
                    )
                    record.data = converted_data
                    record.format = format_type
                
                # 数据验证
                is_valid, errors = self.validator.validate_schema(
                    record.data, self.config.get('validation_schema', {})
                )
                
                if not is_valid:
                    logger.warning(f"数据验证失败: {errors}")
                    continue
                
                # 数据清洗
                record.data = self.validator.clean_data(record.data)
                
                # 更新血缘信息
                record.lineage.append(f"write_to_{source}:{datetime.now().isoformat()}")
                
                processed_records.append(record)
            
            # 写入数据源
            result = await self.retry_manager.execute_with_retry(
                adapter.write, processed_records, **kwargs
            )
            
            # 清除相关缓存
            if result:
                self.cache.clear()
                logger.info(f"成功写入 {len(processed_records)} 条记录")
            
            return result
            
        except Exception as e:
            logger.error(f"写入数据失败: {e}")
            return False
    
    @performance_monitor_decorator("batch_process")
    async def batch_process(self, operations: List[Dict[str, Any]]) -> List[Any]:
        """
        批量数据处理
        
        Args:
            operations: 操作列表，每个操作包含type、source、query等参数
            
        Returns:
            处理结果列表
        """
        results = []
        
        # 创建任务组
        tasks = []
        for i, operation in enumerate(operations):
            if operation['type'] == 'read':
                task = asyncio.create_task(
                    self.read_data(
                        operation['source'],
                        operation.get('query', ''),
                        DataFormat(operation.get('format', 'json')),
                        operation.get('use_cache', True),
                        **operation.get('params', {})
                    )
                )
            elif operation['type'] == 'write':
                task = asyncio.create_task(
                    self.write_data(
                        operation['source'],
                        operation['records'],
                        DataFormat(operation.get('format', 'json')),
                        **operation.get('params', {})
                    )
                )
            else:
                logger.warning(f"不支持的操作类型: {operation['type']}")
                continue
            
            tasks.append((i, task))
        
        # 等待所有任务完成
        for i, task in tasks:
            try:
                result = await task
                results.append((i, 'success', result))
            except Exception as e:
                logger.error(f"批量处理任务 {i} 失败: {e}")
                results.append((i, 'error', str(e)))
        
        # 按原始顺序排序结果
        results.sort(key=lambda x: x[0])
        return [result[2] for result in results]
    
    @performance_monitor_decorator("incremental_sync")
    async def incremental_sync(self, source: str, target: str, 
                              sync_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        增量数据同步
        
        Args:
            source: 源数据源
            target: 目标数据源
            sync_config: 同步配置
            
        Returns:
            同步结果统计
        """
        sync_stats = {
            'total_records': 0,
            'synced_records': 0,
            'failed_records': 0,
            'start_time': datetime.now(),
            'end_time': None
        }
        
        try:
            # 获取增量条件
            last_sync_time = sync_config.get('last_sync_time')
            if last_sync_time:
                query = f"timestamp > '{last_sync_time}'"
            else:
                query = ""
            
            # 读取增量数据
            records = await self.read_data(source, query)
            sync_stats['total_records'] = len(records)
            
            # 批量同步到目标
            if records:
                # 分批处理
                batch_size = sync_config.get('batch_size', 100)
                for i in range(0, len(records), batch_size):
                    batch = records[i:i + batch_size]
                    
                    try:
                        success = await self.write_data(target, batch)
                        if success:
                            sync_stats['synced_records'] += len(batch)
                        else:
                            sync_stats['failed_records'] += len(batch)
                    except Exception as e:
                        logger.error(f"批量同步失败: {e}")
                        sync_stats['failed_records'] += len(batch)
                
                # 更新血缘关系
                for record in records:
                    self.lineage_tracker.add_lineage(
                        record.id,
                        f"sync_{target}_{int(time.time())}",
                        transformation="incremental_sync",
                        metadata={'source': source, 'target': target}
                    )
            
            sync_stats['end_time'] = datetime.now()
            sync_stats['duration'] = (
                sync_stats['end_time'] - sync_stats['start_time']
            ).total_seconds()
            
            logger.info(f"增量同步完成: {sync_stats}")
            return sync_stats
            
        except Exception as e:
            logger.error(f"增量同步失败: {e}")
            sync_stats['end_time'] = datetime.now()
            return sync_stats
    
    def _assess_data_quality(self, record: DataRecord) -> DataQuality:
        """评估数据质量"""
        quality_score = 0
        total_checks = 5
        
        # 检查数据完整性
        if record.data and isinstance(record.data, dict):
            quality_score += 1
        
        # 检查数据格式
        if record.format in [DataFormat.JSON, DataFormat.XML, DataFormat.CSV]:
            quality_score += 1
        
        # 检查血缘信息
        if len(record.lineage) > 0:
            quality_score += 1
        
        # 检查元数据
        if record.metadata:
            quality_score += 1
        
        # 检查时间戳
        if record.timestamp:
            quality_score += 1
        
        # 根据质量分数评级
        ratio = quality_score / total_checks
        if ratio >= 0.8:
            return DataQuality.HIGH
        elif ratio >= 0.6:
            return DataQuality.MEDIUM
        elif ratio >= 0.4:
            return DataQuality.LOW
        else:
            return DataQuality.UNKNOWN
    
    def get_data_lineage(self, record_id: str) -> List[str]:
        """获取数据血缘链"""
        return self.lineage_tracker.get_lineage(record_id)
    
    def analyze_data_impact(self, record_id: str) -> Dict[str, Any]:
        """分析数据影响"""
        return self.lineage_tracker.get_impact_analysis(record_id)
    
    def get_performance_report(self, operation: str = None, 
                              time_range: int = 3600) -> Dict[str, Any]:
        """获取性能报告"""
        return self.performance_monitor.get_performance_summary(operation, time_range)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        return self.cache.get_stats()
    
    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': {}
        }
        
        # 检查各适配器连接状态
        for name, adapter in self.adapters.items():
            try:
                if hasattr(adapter, 'connect'):
                    connected = await adapter.connect()
                    health_status['components'][name] = {
                        'status': 'connected' if connected else 'disconnected',
                        'type': type(adapter).__name__
                    }
                else:
                    health_status['components'][name] = {
                        'status': 'available',
                        'type': type(adapter).__name__
                    }
            except Exception as e:
                health_status['components'][name] = {
                    'status': 'error',
                    'error': str(e),
                    'type': type(adapter).__name__
                }
                health_status['overall_status'] = 'degraded'
        
        # 检查系统资源
        try:
            health_status['system'] = {
                'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
                'cpu_percent': psutil.Process().cpu_percent(),
                'cache_size': len(self.cache._cache)
            }
        except Exception as e:
            health_status['system'] = {'error': str(e)}
            health_status['overall_status'] = 'degraded'
        
        return health_status
    
    async def cleanup(self):
        """清理资源"""
        logger.info("开始清理数据接口适配器资源...")
        
        # 断开所有适配器连接
        for name, adapter in self.adapters.items():
            try:
                if hasattr(adapter, 'disconnect'):
                    await adapter.disconnect()
            except Exception as e:
                logger.warning(f"断开适配器 {name} 连接失败: {e}")
        
        # 清空缓存
        self.cache.clear()
        
        # 清理性能监控数据
        self.performance_monitor.metrics_history.clear()
        
        logger.info("数据接口适配器资源清理完成")


# 测试用例
class TestDataInterfaceAdapter:
    """数据接口适配器测试类"""
    
    @staticmethod
    async def test_basic_operations():
        """测试基本操作"""
        print("=== 测试基本操作 ===")
        
        # 配置
        config = {
            'database': {
                'connection_string': ':memory:',
                'table_name': 'test_data'
            },
            'cache_max_size': 100,
            'cache_ttl': 300,
            'max_retries': 2,
            'validation_schema': {
                'id': {'type': str, 'required': True},
                'name': {'type': str, 'required': True},
                'age': {'type': int, 'min': 0, 'max': 150}
            }
        }
        
        # 创建适配器
        adapter = DataInterfaceAdapter(config)
        
        try:
            # 测试健康检查
            health = await adapter.health_check()
            print(f"健康检查: {health['overall_status']}")
            
            # 创建测试数据
            test_records = [
                DataRecord(
                    id="1",
                    data={"id": "1", "name": "张三", "age": 25},
                    timestamp=datetime.now(),
                    source="test",
                    format=DataFormat.JSON
                ),
                DataRecord(
                    id="2",
                    data={"id": "2", "name": "李四", "age": 30},
                    timestamp=datetime.now(),
                    source="test",
                    format=DataFormat.JSON
                )
            ]
            
            # 测试写入数据
            print("测试写入数据...")
            success = await adapter.write_data('database', test_records)
            print(f"写入结果: {success}")
            
            # 测试读取数据
            print("测试读取数据...")
            records = await adapter.read_data('database')
            print(f"读取到 {len(records)} 条记录")
            
            # 测试数据验证
            print("测试数据验证...")
            invalid_record = DataRecord(
                id="3",
                data={"id": "3", "name": "", "age": -5},  # 无效数据
                timestamp=datetime.now(),
                source="test",
                format=DataFormat.JSON
            )
            
            is_valid, errors = adapter.validator.validate_schema(
                invalid_record.data, config['validation_schema']
            )
            print(f"数据验证结果: 有效={is_valid}, 错误={errors}")
            
            # 测试格式转换
            print("测试格式转换...")
            json_data = [{"name": "测试", "value": 123}]
            xml_data = adapter.converter.convert_format(
                json_data, DataFormat.JSON, DataFormat.XML
            )
            print(f"JSON转XML结果: {xml_data[:100]}...")
            
            # 测试缓存
            print("测试缓存...")
            cache_stats = adapter.get_cache_stats()
            print(f"缓存统计: {cache_stats}")
            
            # 测试性能监控
            print("测试性能监控...")
            perf_report = adapter.get_performance_report()
            print(f"性能报告: {perf_report}")
            
        finally:
            await adapter.cleanup()
        
        print("基本操作测试完成\n")
    
    @staticmethod
    async def test_batch_processing():
        """测试批量处理"""
        print("=== 测试批量处理 ===")
        
        config = {
            'database': {
                'connection_string': ':memory:',
                'table_name': 'batch_test'
            },
            'cache_max_size': 200,
            'cache_ttl': 600
        }
        
        adapter = DataInterfaceAdapter(config)
        
        try:
            # 创建批量操作
            operations = [
                {
                    'type': 'write',
                    'source': 'database',
                    'records': [
                        DataRecord(
                            id=f"batch_{i}",
                            data={"id": f"batch_{i}", "value": i},
                            timestamp=datetime.now(),
                            source="batch_test",
                            format=DataFormat.JSON
                        ) for i in range(5)
                    ]
                },
                {
                    'type': 'read',
                    'source': 'database',
                    'format': 'json'
                }
            ]
            
            print("执行批量处理...")
            results = await adapter.batch_process(operations)
            print(f"批量处理完成，结果数量: {len(results)}")
            
            if results:
                print(f"读取结果: {len(results[1])} 条记录")
            
        finally:
            await adapter.cleanup()
        
        print("批量处理测试完成\n")
    
    @staticmethod
    async def test_incremental_sync():
        """测试增量同步"""
        print("=== 测试增量同步 ===")
        
        config = {
            'database': {
                'connection_string': ':memory:',
                'table_name': 'sync_source'
            },
            'cache_max_size': 100,
            'cache_ttl': 300
        }
        
        adapter = DataInterfaceAdapter(config)
        
        try:
            # 初始数据
            initial_records = [
                DataRecord(
                    id="sync_1",
                    data={"id": "sync_1", "name": "初始数据1"},
                    timestamp=datetime.now(),
                    source="sync_test",
                    format=DataFormat.JSON
                )
            ]
            
            await adapter.write_data('database', initial_records)
            print("写入初始数据完成")
            
            # 模拟增量数据（延迟1秒）
            await asyncio.sleep(1)
            incremental_records = [
                DataRecord(
                    id="sync_2",
                    data={"id": "sync_2", "name": "增量数据1"},
                    timestamp=datetime.now(),
                    source="sync_test",
                    format=DataFormat.JSON
                )
            ]
            
            # 执行增量同步
            sync_config = {
                'last_sync_time': datetime.now() - timedelta(seconds=0.5),
                'batch_size': 10
            }
            
            print("执行增量同步...")
            sync_result = await adapter.incremental_sync(
                'database', 'database', sync_config
            )
            print(f"同步结果: {sync_result}")
            
        finally:
            await adapter.cleanup()
        
        print("增量同步测试完成\n")
    
    @staticmethod
    async def test_data_lineage():
        """测试数据血缘追踪"""
        print("=== 测试数据血缘追踪 ===")
        
        config = {
            'database': {
                'connection_string': ':memory:',
                'table_name': 'lineage_test'
            }
        }
        
        adapter = DataInterfaceAdapter(config)
        
        try:
            # 创建测试记录
            record = DataRecord(
                id="lineage_test_1",
                data={"id": "lineage_test_1", "name": "血缘测试"},
                timestamp=datetime.now(),
                source="lineage_test",
                format=DataFormat.JSON
            )
            
            # 添加血缘关系
            adapter.lineage_tracker.add_lineage(
                record.id,
                "transformed_1",
                transformation="data_conversion",
                metadata={"from_format": "json", "to_format": "xml"}
            )
            
            adapter.lineage_tracker.add_lineage(
                "transformed_1",
                "final_1",
                transformation="data_validation",
                metadata={"validation_rules": "schema_check"}
            )
            
            # 获取血缘链
            lineage = adapter.get_data_lineage(record.id)
            print(f"血缘链: {lineage}")
            
            # 影响分析
            impact = adapter.analyze_data_impact(record.id)
            print(f"影响分析: {impact}")
            
        finally:
            await adapter.cleanup()
        
        print("数据血缘追踪测试完成\n")


async def main():
    """主函数"""
    print("I3 数据接口适配器测试开始\n")
    
    # 运行所有测试
    await TestDataInterfaceAdapter.test_basic_operations()
    await TestDataInterfaceAdapter.test_batch_processing()
    await TestDataInterfaceAdapter.test_incremental_sync()
    await TestDataInterfaceAdapter.test_data_lineage()
    
    print("所有测试完成！")


if __name__ == "__main__":
    # 运行测试
    asyncio.run(main())