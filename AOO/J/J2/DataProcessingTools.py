#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
J2数据处理工具模块

该模块提供了全面的数据处理功能，包括数据清洗、转换、聚合、验证等。
支持异步处理、内存优化和大数据处理。

主要功能：
1. 数据清洗工具（缺失值处理、异常值检测、数据标准化）
2. 数据转换工具（格式转换、编码处理、数据类型转换）
3. 数据聚合工具（分组统计、时间窗口聚合、多维度聚合）
4. 数据验证工具（数据质量检查、完整性验证、一致性检查）
5. 异步数据处理管道
6. 内存优化和大数据处理

作者：J2数据处理团队
版本：1.0.0
创建日期：2025-11-06
"""

import asyncio
import logging
import warnings
import hashlib
import json
import pickle
import csv
import xml.etree.ElementTree as ET
import re
import math
import statistics
from abc import ABC, abstractmethod
from collections import defaultdict, Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import wraps, reduce
from io import StringIO, BytesIO
from itertools import groupby, combinations
from typing import (
    Any, Dict, List, Optional, Union, Callable, Generator, 
    Tuple, Set, Iterator, Type, TypeVar, Generic, Protocol,
    overload, Literal, NamedTuple
)
from urllib.parse import urlparse
import weakref

import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# 类型定义
DataFrameLike = Union[pd.DataFrame, pd.Series]
NumericType = Union[int, float, np.number]
AnyDataFrame = Union[pd.DataFrame, pd.Series, np.ndarray]
ProcessingResult = Dict[str, Any]
ValidationResult = Dict[str, Union[bool, List[str], Dict[str, Any]]]


class ProcessingError(Exception):
    """数据处理错误基类"""
    pass


class ValidationError(Exception):
    """数据验证错误基类"""
    pass


class MemoryError(Exception):
    """内存处理错误基类"""
    pass


@dataclass
class ProcessingStats:
    """处理统计信息"""
    total_records: int = 0
    processed_records: int = 0
    error_records: int = 0
    processing_time: float = 0.0
    memory_usage: float = 0.0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        """计算成功率"""
        if self.total_records == 0:
            return 0.0
        return (self.processed_records / self.total_records) * 100
    
    @property
    def error_rate(self) -> float:
        """计算错误率"""
        if self.total_records == 0:
            return 0.0
        return (self.error_records / self.total_records) * 100


@dataclass
class ValidationRule:
    """验证规则定义"""
    name: str
    description: str
    validation_func: Callable
    severity: Literal['error', 'warning', 'info'] = 'error'
    parameters: Dict[str, Any] = field(default_factory=dict)


class BaseProcessor(ABC):
    """数据处理器基类"""
    
    def __init__(self, name: str = "BaseProcessor"):
        self.name = name
        self.stats = ProcessingStats()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def process(self, data: AnyDataFrame, **kwargs) -> AnyDataFrame:
        """处理数据"""
        pass
    
    def _update_stats(self, success: bool = True, records: int = 1):
        """更新处理统计信息"""
        self.stats.processed_records += 1 if success else 0
        self.stats.error_records += 0 if success else 1
        self.stats.total_records += records
    
    def _log_processing(self, operation: str, data_shape: Optional[Tuple[int, ...]] = None):
        """记录处理日志"""
        self.logger.info(f"{self.name} - {operation} - 数据形状: {data_shape}")


class DataCleaner(BaseProcessor):
    """数据清洗工具类"""
    
    def __init__(self):
        super().__init__("DataCleaner")
        self.missing_strategies = {
            'mean': self._fill_mean,
            'median': self._fill_median,
            'mode': self._fill_mode,
            'constant': self._fill_constant,
            'forward_fill': self._forward_fill,
            'backward_fill': self._backward_fill,
            'interpolate': self._interpolate,
            'drop': self._drop_missing
        }
        self.outlier_methods = {
            'zscore': self._detect_outliers_zscore,
            'iqr': self._detect_outliers_iqr,
            'isolation_forest': self._detect_outliers_isolation_forest,
            'dbscan': self._detect_outliers_dbscan,
            'lof': self._detect_outliers_lof
        }
    
    async def process(self, data: DataFrameLike, **kwargs) -> DataFrameLike:
        """执行数据清洗"""
        if data is None:
            raise ValueError("输入数据不能为None")
        
        # 数据验证
        if hasattr(data, 'empty') and data.empty:
            raise ValueError("输入数据为空")
        
        if not hasattr(data, 'shape') and not hasattr(data, '__len__'):
            raise ValueError("不支持的数据类型")
        
        self.stats.start_time = datetime.now()
        
        try:
            # 缺失值处理
            if kwargs.get('handle_missing', True):
                data = await self.handle_missing_values(
                    data, 
                    strategy=kwargs.get('missing_strategy', 'drop'),
                    columns=kwargs.get('missing_columns')
                )
            
            # 异常值检测和处理
            if kwargs.get('handle_outliers', False):
                data = await self.handle_outliers(
                    data,
                    method=kwargs.get('outlier_method', 'iqr'),
                    action=kwargs.get('outlier_action', 'remove')
                )
            
            # 数据标准化
            if kwargs.get('normalize', False):
                data = await self.normalize_data(
                    data,
                    method=kwargs.get('normalization_method', 'standard')
                )
            
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            
            return data
            
        except Exception as e:
            self.logger.error(f"数据清洗失败: {str(e)}")
            raise ProcessingError(f"数据清洗失败: {str(e)}")
    
    async def handle_missing_values(
        self, 
        data: DataFrameLike, 
        strategy: str = 'drop',
        columns: Optional[List[str]] = None,
        **kwargs
    ) -> DataFrameLike:
        """处理缺失值"""
        self._log_processing("缺失值处理", data.shape if hasattr(data, 'shape') else None)
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        if columns is None:
            columns = data.columns.tolist()
        
        if strategy not in self.missing_strategies:
            raise ValueError(f"不支持的缺失值处理策略: {strategy}")
        
        # 检查缺失值情况
        missing_info = data.isnull().sum()
        missing_cols = missing_info[missing_info > 0].index.tolist()
        
        if not missing_cols:
            self.logger.info("未发现缺失值")
            return data
        
        self.logger.info(f"发现缺失值的列: {missing_cols}")
        
        # 应用处理策略
        result = self.missing_strategies[strategy](data, missing_cols, **kwargs)
        
        return result
    
    def _fill_mean(self, data: DataFrameLike, columns: List[str], **kwargs) -> DataFrameLike:
        """用均值填充缺失值"""
        # 内存优化：如果数据集很大且不需要保留原数据，直接在原数据上操作
        inplace = kwargs.get('inplace', False)
        if inplace and hasattr(data, 'copy'):
            result = data
        else:
            result = data.copy()
            
        for col in columns:
            if col in result.columns:
                try:
                    mean_value = result[col].mean()
                    if pd.isna(mean_value):
                        self.logger.warning(f"列 '{col}' 均值为空，跳过填充")
                        continue
                    result.loc[:, col] = result[col].fillna(mean_value)
                    self.logger.info(f"列 '{col}' 用均值 {mean_value:.4f} 填充")
                except Exception as e:
                    self.logger.warning(f"列 '{col}' 填充失败: {str(e)}")
        return result
    
    def _fill_median(self, data: DataFrameLike, columns: List[str], **kwargs) -> DataFrameLike:
        """用中位数填充缺失值"""
        result = data.copy()
        for col in columns:
            if col in result.columns:
                median_value = result[col].median()
                if pd.isna(median_value):
                    self.logger.warning(f"列 '{col}' 中位数为空，跳过填充")
                    continue
                result.loc[:, col] = result[col].fillna(median_value)
                self.logger.info(f"列 '{col}' 用中位数 {median_value:.4f} 填充")
        return result
    
    def _fill_mode(self, data: DataFrameLike, columns: List[str], **kwargs) -> DataFrameLike:
        """用众数填充缺失值"""
        result = data.copy()
        for col in columns:
            if col in result.columns:
                mode_value = result[col].mode()
                if len(mode_value) > 0:
                    fill_value = mode_value.iloc[0]
                    if pd.isna(fill_value):
                        self.logger.warning(f"列 '{col}' 众数为空，跳过填充")
                        continue
                    result.loc[:, col] = result[col].fillna(fill_value)
                    self.logger.info(f"列 '{col}' 用众数 {fill_value} 填充")
                else:
                    self.logger.warning(f"列 '{col}' 没有众数，跳过填充")
        return result
    
    def _fill_constant(self, data: DataFrameLike, columns: List[str], **kwargs) -> DataFrameLike:
        """用常数填充缺失值"""
        result = data.copy()
        constant_value = kwargs.get('constant_value', 0)
        for col in columns:
            if col in result.columns:
                result.loc[:, col] = result[col].fillna(constant_value)
                self.logger.info(f"列 '{col}' 用常数 {constant_value} 填充")
        return result
    
    def _forward_fill(self, data: DataFrameLike, columns: List[str], **kwargs) -> DataFrameLike:
        """前向填充"""
        result = data.copy()
        result = result.ffill()
        self.logger.info("执行前向填充")
        return result
    
    def _backward_fill(self, data: DataFrameLike, columns: List[str], **kwargs) -> DataFrameLike:
        """后向填充"""
        result = data.copy()
        result = result.bfill()
        self.logger.info("执行后向填充")
        return result
    
    def _interpolate(self, data: DataFrameLike, columns: List[str], **kwargs) -> DataFrameLike:
        """插值填充"""
        result = data.copy()
        method = kwargs.get('interpolate_method', 'linear')
        for col in columns:
            if col in result.columns:
                try:
                    result.loc[:, col] = result[col].interpolate(method=method)
                    self.logger.info(f"列 '{col}' 使用 {method} 方法执行插值填充")
                except Exception as e:
                    self.logger.warning(f"列 '{col}' 插值失败: {str(e)}")
        return result
    
    def _drop_missing(self, data: DataFrameLike, columns: List[str], **kwargs) -> DataFrameLike:
        """删除缺失值"""
        if not columns:
            self.logger.warning("没有指定要处理的列")
            return data
        
        try:
            result = data.dropna(subset=columns)
            dropped_count = len(data) - len(result)
            self.logger.info(f"删除了 {dropped_count} 行缺失值")
            return result
        except Exception as e:
            self.logger.error(f"删除缺失值失败: {str(e)}")
            raise ProcessingError(f"删除缺失值失败: {str(e)}")
    
    async def handle_outliers(
        self, 
        data: DataFrameLike, 
        method: str = 'iqr',
        action: str = 'remove',
        **kwargs
    ) -> DataFrameLike:
        """处理异常值"""
        self._log_processing("异常值处理", data.shape if hasattr(data, 'shape') else None)
        
        if method not in self.outlier_methods:
            raise ValueError(f"不支持的异常值检测方法: {method}")
        
        # 只对数值列进行异常值检测
        numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            self.logger.warning("未发现数值列，跳过异常值处理")
            return data
        
        # 检测异常值
        outlier_indices = await self.outlier_methods[method](data, numeric_columns, **kwargs)
        
        if not outlier_indices:
            self.logger.info("未发现异常值")
            return data
        
        self.logger.info(f"发现 {len(outlier_indices)} 个异常值")
        
        # 处理异常值
        if action == 'remove':
            result = data.drop(outlier_indices)
            self.logger.info(f"移除了 {len(outlier_indices)} 个异常值")
        elif action == 'cap':
            result = await self._cap_outliers(data, outlier_indices, numeric_columns)
        else:
            result = data
        
        return result
    
    def _detect_outliers_zscore(
        self, 
        data: DataFrameLike, 
        columns: List[str], 
        threshold: float = 3.0
    ) -> Set[int]:
        """使用Z-score方法检测异常值"""
        if threshold <= 0:
            raise ValueError("阈值必须大于0")
        
        outlier_indices = set()
        
        for col in columns:
            if col in data.columns:
                try:
                    series_data = data[col].dropna()
                    if len(series_data) == 0:
                        continue
                    
                    z_scores = np.abs(stats.zscore(series_data))
                    outlier_mask = z_scores > threshold
                    outlier_indices.update(series_data[outlier_mask].index.tolist())
                    
                except Exception as e:
                    self.logger.warning(f"列 '{col}' Z-score检测失败: {str(e)}")
        
        return outlier_indices
    
    def _detect_outliers_iqr(
        self, 
        data: DataFrameLike, 
        columns: List[str],
        factor: float = 1.5
    ) -> Set[int]:
        """使用IQR方法检测异常值"""
        if factor <= 0:
            raise ValueError("因子必须大于0")
        
        outlier_indices = set()
        
        for col in columns:
            if col in data.columns:
                try:
                    series_data = data[col].dropna()
                    if len(series_data) == 0:
                        continue
                    
                    Q1 = series_data.quantile(0.25)
                    Q3 = series_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR == 0:
                        self.logger.warning(f"列 '{col}' IQR为0，跳过异常值检测")
                        continue
                    
                    lower_bound = Q1 - factor * IQR
                    upper_bound = Q3 + factor * IQR
                    
                    outliers = series_data[(series_data < lower_bound) | (series_data > upper_bound)]
                    outlier_indices.update(outliers.index.tolist())
                    
                except Exception as e:
                    self.logger.warning(f"列 '{col}' IQR检测失败: {str(e)}")
        
        return outlier_indices
    
    def _detect_outliers_isolation_forest(
        self, 
        data: DataFrameLike, 
        columns: List[str],
        contamination: float = 0.1
    ) -> Set[int]:
        """使用Isolation Forest检测异常值"""
        if not (0 < contamination < 0.5):
            raise ValueError("contamination必须在(0, 0.5)范围内")
        
        outlier_indices = set()
        
        # 准备数据
        try:
            numeric_data = data[columns].dropna()
            
            if len(numeric_data) == 0:
                self.logger.warning("没有有效数据用于异常值检测")
                return outlier_indices
            
            # 训练模型
            iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
            predictions = iso_forest.fit_predict(numeric_data)
            
            # 获取异常值索引
            outlier_mask = predictions == -1
            outlier_indices.update(numeric_data[outlier_mask].index.tolist())
            
        except Exception as e:
            self.logger.error(f"Isolation Forest检测失败: {str(e)}")
        
        return outlier_indices
    
    def _detect_outliers_dbscan(
        self, 
        data: DataFrameLike, 
        columns: List[str],
        eps: float = 0.5,
        min_samples: int = 5
    ) -> Set[int]:
        """使用DBSCAN检测异常值"""
        if eps <= 0:
            raise ValueError("eps必须大于0")
        if min_samples <= 1:
            raise ValueError("min_samples必须大于1")
        
        outlier_indices = set()
        
        # 准备数据
        try:
            numeric_data = data[columns].dropna()
            
            if len(numeric_data) == 0:
                self.logger.warning("没有有效数据用于异常值检测")
                return outlier_indices
            
            if len(numeric_data) < min_samples:
                self.logger.warning(f"数据量({len(numeric_data)})小于最小样本数({min_samples})")
                return outlier_indices
            
            # 训练模型
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=-1)
            cluster_labels = dbscan.fit_predict(numeric_data)
            
            # 标记噪声点为异常值
            outlier_mask = cluster_labels == -1
            outlier_indices.update(numeric_data[outlier_mask].index.tolist())
            
        except Exception as e:
            self.logger.error(f"DBSCAN检测失败: {str(e)}")
        
        return outlier_indices
    
    def _detect_outliers_lof(
        self, 
        data: DataFrameLike, 
        columns: List[str],
        n_neighbors: int = 20,
        contamination: float = 0.1
    ) -> Set[int]:
        """使用局部异常因子检测异常值"""
        if n_neighbors <= 0:
            raise ValueError("n_neighbors必须大于0")
        if not (0 < contamination < 0.5):
            raise ValueError("contamination必须在(0, 0.5)范围内")
        
        outlier_indices = set()
        
        # 准备数据
        try:
            numeric_data = data[columns].dropna()
            
            if len(numeric_data) == 0:
                self.logger.warning("没有有效数据用于异常值检测")
                return outlier_indices
            
            if len(numeric_data) <= n_neighbors:
                self.logger.warning(f"数据量({len(numeric_data)})小于邻居数({n_neighbors})")
                return outlier_indices
            
            # 训练模型
            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination, n_jobs=-1)
            predictions = lof.fit_predict(numeric_data)
            
            # 获取异常值索引
            outlier_mask = predictions == -1
            outlier_indices.update(numeric_data[outlier_mask].index.tolist())
            
        except Exception as e:
            self.logger.error(f"LOF检测失败: {str(e)}")
        
        return outlier_indices
    
    def _cap_outliers(
        self, 
        data: DataFrameLike, 
        outlier_indices: Set[int], 
        columns: List[str]
    ) -> DataFrameLike:
        """截断异常值"""
        if not outlier_indices or not columns:
            return data.copy()
        
        result = data.copy()
        
        for col in columns:
            if col in result.columns:
                try:
                    series_data = result[col].dropna()
                    if len(series_data) == 0:
                        continue
                    
                    Q1 = series_data.quantile(0.25)
                    Q3 = series_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR == 0:
                        self.logger.warning(f"列 '{col}' IQR为0，跳过截断")
                        continue
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # 截断异常值
                    before_count = len(result[(result[col] < lower_bound) | (result[col] > upper_bound)])
                    result.loc[result[col] < lower_bound, col] = lower_bound
                    result.loc[result[col] > upper_bound, col] = upper_bound
                    after_count = len(result[(result[col] < lower_bound) | (result[col] > upper_bound)])
                    
                    if before_count > 0:
                        self.logger.info(f"列 '{col}' 截断了 {before_count - after_count} 个异常值")
                    
                except Exception as e:
                    self.logger.warning(f"列 '{col}' 截断失败: {str(e)}")
        
        return result
    
    async def normalize_data(
        self, 
        data: DataFrameLike, 
        method: str = 'standard',
        columns: Optional[List[str]] = None
    ) -> DataFrameLike:
        """数据标准化"""
        self._log_processing("数据标准化", data.shape if hasattr(data, 'shape') else None)
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        # 选择要标准化的列
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            self.logger.warning("未发现数值列，跳过标准化")
            return data
        
        result = data.copy()
        
        try:
            if method == 'standard':
                scaler = StandardScaler()
                result[columns] = scaler.fit_transform(result[columns])
                self.logger.info("执行标准标准化")
            elif method == 'minmax':
                scaler = MinMaxScaler()
                result[columns] = scaler.fit_transform(result[columns])
                self.logger.info("执行MinMax标准化")
            elif method == 'robust':
                scaler = RobustScaler()
                result[columns] = scaler.fit_transform(result[columns])
                self.logger.info("执行鲁棒标准化")
            else:
                raise ValueError(f"不支持的标准化方法: {method}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"标准化失败: {str(e)}")
            raise ProcessingError(f"标准化失败: {str(e)}")


class DataTransformer(BaseProcessor):
    """数据转换工具类"""
    
    def __init__(self):
        super().__init__("DataTransformer")
        self.format_converters = {
            'json': self._convert_to_json,
            'xml': self._convert_to_xml,
            'csv': self._convert_to_csv,
            'excel': self._convert_to_excel,
            'parquet': self._convert_to_parquet,
            'pickle': self._convert_to_pickle
        }
        self.encoding_converters = {
            'utf-8': 'utf-8',
            'gbk': 'gbk',
            'gb2312': 'gb2312',
            'ascii': 'ascii',
            'latin1': 'latin1'
        }
    
    async def process(self, data: DataFrameLike, **kwargs) -> AnyDataFrame:
        """执行数据转换"""
        self.stats.start_time = datetime.now()
        
        try:
            # 格式转换
            if kwargs.get('convert_format'):
                target_format = kwargs.get('target_format', 'json')
                data = await self.convert_format(data, target_format)
            
            # 编码转换
            if kwargs.get('convert_encoding'):
                target_encoding = kwargs.get('target_encoding', 'utf-8')
                data = await self.convert_encoding(data, target_encoding)
            
            # 数据类型转换
            if kwargs.get('convert_dtype'):
                dtype_mapping = kwargs.get('dtype_mapping', {})
                data = await self.convert_data_types(data, dtype_mapping)
            
            # 特征工程
            if kwargs.get('feature_engineering'):
                feature_config = kwargs.get('feature_config', {})
                data = await self.feature_engineering(data, feature_config)
            
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            
            return data
            
        except Exception as e:
            self.logger.error(f"数据转换失败: {str(e)}")
            raise ProcessingError(f"数据转换失败: {str(e)}")
    
    async def convert_format(self, data: DataFrameLike, target_format: str, **kwargs) -> Any:
        """转换数据格式"""
        self._log_processing(f"格式转换到{target_format}", data.shape if hasattr(data, 'shape') else None)
        
        if target_format not in self.format_converters:
            raise ValueError(f"不支持的目标格式: {target_format}")
        
        return await self.format_converters[target_format](data, **kwargs)
    
    async def _convert_to_json(self, data: DataFrameLike, **kwargs) -> str:
        """转换为JSON格式"""
        if isinstance(data, pd.DataFrame):
            return data.to_json(orient='records', force_ascii=False, indent=2)
        elif isinstance(data, pd.Series):
            return data.to_json(force_ascii=False, indent=2)
        else:
            return json.dumps(data, ensure_ascii=False, indent=2)
    
    async def _convert_to_xml(self, data: DataFrameLike, **kwargs) -> str:
        """转换为XML格式"""
        root = ET.Element("data")
        
        if isinstance(data, pd.DataFrame):
            for _, row in data.iterrows():
                record = ET.SubElement(root, "record")
                for col, value in row.items():
                    field = ET.SubElement(record, col)
                    field.text = str(value)
        elif isinstance(data, pd.Series):
            for key, value in data.items():
                field = ET.SubElement(root, str(key))
                field.text = str(value)
        
        return ET.tostring(root, encoding='unicode')
    
    async def _convert_to_csv(self, data: DataFrameLike, **kwargs) -> str:
        """转换为CSV格式"""
        output = StringIO()
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(output, index=False, **kwargs)
        elif isinstance(data, pd.Series):
            data.to_csv(output, **kwargs)
        
        return output.getvalue()
    
    async def _convert_to_excel(self, data: DataFrameLike, **kwargs) -> bytes:
        """转换为Excel格式"""
        output = BytesIO()
        
        if isinstance(data, pd.DataFrame):
            data.to_excel(output, index=False, **kwargs)
        elif isinstance(data, pd.Series):
            data.to_excel(output, **kwargs)
        
        return output.getvalue()
    
    async def _convert_to_parquet(self, data: DataFrameLike, **kwargs) -> bytes:
        """转换为Parquet格式"""
        output = BytesIO()
        
        if isinstance(data, pd.DataFrame):
            data.to_parquet(output, **kwargs)
        elif isinstance(data, pd.Series):
            data.to_frame().to_parquet(output, **kwargs)
        
        return output.getvalue()
    
    async def _convert_to_pickle(self, data: DataFrameLike, **kwargs) -> bytes:
        """转换为Pickle格式"""
        output = BytesIO()
        pickle.dump(data, output, **kwargs)
        return output.getvalue()
    
    async def convert_encoding(self, data: DataFrameLike, target_encoding: str) -> DataFrameLike:
        """转换数据编码"""
        self._log_processing(f"编码转换到{target_encoding}", data.shape if hasattr(data, 'shape') else None)
        
        if target_encoding not in self.encoding_converters:
            raise ValueError(f"不支持的目标编码: {target_encoding}")
        
        result = data.copy()
        
        # 转换字符串列的编码
        for col in result.columns:
            if result[col].dtype == 'object':
                try:
                    result[col] = result[col].str.encode('utf-8').str.decode(target_encoding)
                except (UnicodeDecodeError, UnicodeEncodeError) as e:
                    self.logger.warning(f"列 {col} 编码转换失败: {str(e)}")
        
        return result
    
    async def convert_data_types(
        self, 
        data: DataFrameLike, 
        dtype_mapping: Dict[str, Union[str, Type]]
    ) -> DataFrameLike:
        """转换数据类型"""
        self._log_processing("数据类型转换", data.shape if hasattr(data, 'shape') else None)
        
        result = data.copy()
        
        for col, target_dtype in dtype_mapping.items():
            if col in result.columns:
                try:
                    if isinstance(target_dtype, str):
                        result[col] = result[col].astype(target_dtype)
                    else:
                        result[col] = result[col].astype(target_dtype)
                    self.logger.info(f"列 {col} 转换为 {target_dtype}")
                except Exception as e:
                    self.logger.warning(f"列 {col} 类型转换失败: {str(e)}")
        
        return result
    
    async def feature_engineering(
        self, 
        data: DataFrameLike, 
        config: Dict[str, Any]
    ) -> DataFrameLike:
        """特征工程"""
        self._log_processing("特征工程", data.shape if hasattr(data, 'shape') else None)
        
        result = data.copy()
        
        # 创建新特征
        if 'new_features' in config:
            for feature_name, feature_expr in config['new_features'].items():
                try:
                    result[feature_name] = result.eval(feature_expr)
                    self.logger.info(f"创建特征: {feature_name}")
                except Exception as e:
                    self.logger.warning(f"特征 {feature_name} 创建失败: {str(e)}")
        
        # 时间特征提取
        if config.get('extract_time_features', False):
            datetime_cols = result.select_dtypes(include=['datetime64', 'object']).columns
            for col in datetime_cols:
                try:
                    # 尝试解析为日期时间
                    dt_series = pd.to_datetime(result[col], errors='coerce')
                    
                    if not dt_series.isna().all():
                        result[f"{col}_year"] = dt_series.dt.year
                        result[f"{col}_month"] = dt_series.dt.month
                        result[f"{col}_day"] = dt_series.dt.day
                        result[f"{col}_weekday"] = dt_series.dt.weekday
                        result[f"{col}_hour"] = dt_series.dt.hour
                        self.logger.info(f"从 {col} 提取时间特征")
                except Exception as e:
                    self.logger.warning(f"时间特征提取失败 {col}: {str(e)}")
        
        # 数值特征变换
        if config.get('numeric_transformations', False):
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                try:
                    # 对数变换
                    if result[col].min() > 0:
                        result[f"{col}_log"] = np.log(result[col])
                    
                    # 平方根变换
                    if result[col].min() >= 0:
                        result[f"{col}_sqrt"] = np.sqrt(result[col])
                    
                    # 标准化
                    result[f"{col}_standardized"] = (result[col] - result[col].mean()) / result[col].std()
                    
                    self.logger.info(f"对 {col} 执行数值变换")
                except Exception as e:
                    self.logger.warning(f"数值变换失败 {col}: {str(e)}")
        
        return result


class DataAggregator(BaseProcessor):
    """数据聚合工具类"""
    
    def __init__(self):
        super().__init__("DataAggregator")
        self.aggregation_functions = {
            'sum': 'sum',
            'mean': 'mean',
            'median': 'median',
            'std': 'std',
            'var': 'var',
            'min': 'min',
            'max': 'max',
            'count': 'count',
            'nunique': 'nunique',
            'first': 'first',
            'last': 'last'
        }
    
    async def process(self, data: DataFrameLike, **kwargs) -> DataFrameLike:
        """执行数据聚合"""
        self.stats.start_time = datetime.now()
        
        try:
            # 分组统计
            if kwargs.get('groupby'):
                group_config = kwargs.get('group_config', {})
                data = await self.groupby_aggregation(data, **group_config)
            
            # 时间窗口聚合
            if kwargs.get('time_window'):
                window_config = kwargs.get('window_config', {})
                data = await self.time_window_aggregation(data, **window_config)
            
            # 多维度聚合
            if kwargs.get('multi_dimensional'):
                multi_config = kwargs.get('multi_config', {})
                data = await self.multi_dimensional_aggregation(data, **multi_config)
            
            # 滚动统计
            if kwargs.get('rolling_stats'):
                rolling_config = kwargs.get('rolling_config', {})
                data = await self.rolling_statistics(data, **rolling_config)
            
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            
            return data
            
        except Exception as e:
            self.logger.error(f"数据聚合失败: {str(e)}")
            raise ProcessingError(f"数据聚合失败: {str(e)}")
    
    async def groupby_aggregation(
        self, 
        data: DataFrameLike, 
        group_columns: List[str],
        agg_functions: Dict[str, List[str]],
        **kwargs
    ) -> DataFrameLike:
        """分组聚合"""
        self._log_processing("分组聚合", data.shape if hasattr(data, 'shape') else None)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("分组聚合需要DataFrame格式的数据")
        
        # 验证分组列
        missing_cols = [col for col in group_columns if col not in data.columns]
        if missing_cols:
            raise ValueError(f"分组列不存在: {missing_cols}")
        
        # 执行分组聚合
        result = data.groupby(group_columns).agg(agg_functions).reset_index()
        
        # 简化列名
        result.columns = ['_'.join(col).strip() if col[1] else col[0] for col in result.columns]
        
        self.logger.info(f"分组聚合完成，结果形状: {result.shape}")
        return result
    
    async def time_window_aggregation(
        self, 
        data: DataFrameLike,
        time_column: str,
        window_size: str,
        agg_functions: Dict[str, List[str]],
        **kwargs
    ) -> DataFrameLike:
        """时间窗口聚合"""
        self._log_processing("时间窗口聚合", data.shape if hasattr(data, 'shape') else None)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("时间窗口聚合需要DataFrame格式的数据")
        
        if time_column not in data.columns:
            raise ValueError(f"时间列不存在: {time_column}")
        
        # 确保时间列是datetime类型
        data[time_column] = pd.to_datetime(data[time_column])
        
        # 设置时间列为索引
        data_indexed = data.set_index(time_column)
        
        # 执行时间窗口聚合
        result = data_indexed.resample(window_size).agg(agg_functions)
        
        # 简化列名
        result.columns = ['_'.join(col).strip() if col[1] else col[0] for col in result.columns]
        
        # 重置索引
        result = result.reset_index()
        
        self.logger.info(f"时间窗口聚合完成，结果形状: {result.shape}")
        return result
    
    async def multi_dimensional_aggregation(
        self, 
        data: DataFrameLike,
        dimensions: List[List[str]],
        metrics: Dict[str, List[str]],
        **kwargs
    ) -> DataFrameLike:
        """多维度聚合"""
        self._log_processing("多维度聚合", data.shape if hasattr(data, 'shape') else None)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("多维度聚合需要DataFrame格式的数据")
        
        results = []
        
        for dimension in dimensions:
            # 验证维度列
            missing_cols = [col for col in dimension if col not in data.columns]
            if missing_cols:
                self.logger.warning(f"维度列不存在，跳过: {missing_cols}")
                continue
            
            # 执行分组聚合
            result = data.groupby(dimension).agg(metrics).reset_index()
            result['dimension'] = '_'.join(dimension)
            results.append(result)
        
        # 合并结果
        if results:
            final_result = pd.concat(results, ignore_index=True)
            self.logger.info(f"多维度聚合完成，结果形状: {final_result.shape}")
            return final_result
        else:
            self.logger.warning("没有有效的维度进行聚合")
            return data
    
    async def rolling_statistics(
        self, 
        data: DataFrameLike,
        columns: List[str],
        window: int,
        functions: List[str],
        **kwargs
    ) -> DataFrameLike:
        """滚动统计"""
        self._log_processing("滚动统计", data.shape if hasattr(data, 'shape') else None)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("滚动统计需要DataFrame格式的数据")
        
        result = data.copy()
        
        for col in columns:
            if col not in data.columns:
                self.logger.warning(f"列不存在，跳过: {col}")
                continue
            
            for func in functions:
                if func not in self.aggregation_functions:
                    self.logger.warning(f"不支持的聚合函数，跳过: {func}")
                    continue
                
                try:
                    result[f"{col}_rolling_{func}_{window}"] = (
                        data[col].rolling(window=window).agg(func)
                    )
                except Exception as e:
                    self.logger.warning(f"滚动统计失败 {col}_{func}: {str(e)}")
        
        self.logger.info(f"滚动统计完成，添加了 {len(columns) * len(functions)} 个特征")
        return result


class DataValidator(BaseProcessor):
    """数据验证工具类"""
    
    def __init__(self):
        super().__init__("DataValidator")
        self.validation_rules = []
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """设置默认验证规则"""
        # 数据完整性规则
        self.add_rule(ValidationRule(
            name="missing_values",
            description="检查缺失值",
            validation_func=self._check_missing_values,
            severity="warning"
        ))
        
        # 数据类型规则
        self.add_rule(ValidationRule(
            name="data_types",
            description="检查数据类型",
            validation_func=self._check_data_types,
            severity="error"
        ))
        
        # 数据范围规则
        self.add_rule(ValidationRule(
            name="data_range",
            description="检查数据范围",
            validation_func=self._check_data_range,
            severity="warning"
        ))
        
        # 重复值检查
        self.add_rule(ValidationRule(
            name="duplicates",
            description="检查重复值",
            validation_func=self._check_duplicates,
            severity="warning"
        ))
        
        # 唯一性检查
        self.add_rule(ValidationRule(
            name="uniqueness",
            description="检查唯一性",
            validation_func=self._check_uniqueness,
            severity="error"
        ))
    
    def add_rule(self, rule: ValidationRule):
        """添加验证规则"""
        self.validation_rules.append(rule)
    
    def remove_rule(self, rule_name: str):
        """移除验证规则"""
        self.validation_rules = [rule for rule in self.validation_rules if rule.name != rule_name]
    
    async def process(self, data: DataFrameLike, **kwargs) -> ValidationResult:
        """执行数据验证"""
        self.stats.start_time = datetime.now()
        
        try:
            validation_results = await self.validate_data(data, **kwargs)
            
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"数据验证失败: {str(e)}")
            raise ValidationError(f"数据验证失败: {str(e)}")
    
    async def validate_data(self, data: DataFrameLike, **kwargs) -> ValidationResult:
        """验证数据"""
        self._log_processing("数据验证", data.shape if hasattr(data, 'shape') else None)
        
        results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'info': [],
            'details': {}
        }
        
        for rule in self.validation_rules:
            try:
                rule_result = await self._run_validation_rule(rule, data, **kwargs)
                results['details'][rule.name] = rule_result
                
                if not rule_result['passed']:
                    if rule.severity == 'error':
                        results['errors'].append(f"{rule.name}: {rule_result['message']}")
                        results['is_valid'] = False
                    elif rule.severity == 'warning':
                        results['warnings'].append(f"{rule.name}: {rule_result['message']}")
                    else:
                        results['info'].append(f"{rule.name}: {rule_result['message']}")
                        
            except Exception as e:
                error_msg = f"规则 {rule.name} 执行失败: {str(e)}"
                results['errors'].append(error_msg)
                results['is_valid'] = False
                self.logger.error(error_msg)
        
        return results
    
    async def _run_validation_rule(
        self, 
        rule: ValidationRule, 
        data: DataFrameLike, 
        **kwargs
    ) -> Dict[str, Any]:
        """运行单个验证规则"""
        # 直接调用验证函数，不使用asyncio.to_thread
        return rule.validation_func(data, **kwargs)
    
    def _check_missing_values(self, data: DataFrameLike, **kwargs) -> Dict[str, Any]:
        """检查缺失值"""
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        missing_counts = data.isnull().sum()
        total_missing = missing_counts.sum()
        
        return {
            'passed': total_missing == 0,
            'message': f"发现 {total_missing} 个缺失值",
            'details': {
                'total_missing': int(total_missing),
                'missing_by_column': missing_counts.to_dict(),
                'missing_percentage': (total_missing / (len(data) * len(data.columns))) * 100
            }
        }
    
    def _check_data_types(self, data: DataFrameLike, **kwargs) -> Dict[str, Any]:
        """检查数据类型"""
        expected_types = kwargs.get('expected_types', {})
        type_errors = []
        
        for col, expected_type in expected_types.items():
            if col in data.columns:
                actual_type = str(data[col].dtype)
                if expected_type.lower() not in actual_type.lower():
                    type_errors.append(f"列 {col}: 期望 {expected_type}, 实际 {actual_type}")
        
        return {
            'passed': len(type_errors) == 0,
            'message': f"发现 {len(type_errors)} 个类型错误" if type_errors else "数据类型检查通过",
            'details': {
                'type_errors': type_errors,
                'actual_types': {col: str(dtype) for col, dtype in data.dtypes.items()}
            }
        }
    
    def _check_data_range(self, data: DataFrameLike, **kwargs) -> Dict[str, Any]:
        """检查数据范围"""
        range_rules = kwargs.get('range_rules', {})
        range_errors = []
        
        for col, rule in range_rules.items():
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                values = data[col].dropna()
                
                if 'min' in rule and values.min() < rule['min']:
                    range_errors.append(f"列 {col}: 最小值 {values.min()} 小于期望的 {rule['min']}")
                
                if 'max' in rule and values.max() > rule['max']:
                    range_errors.append(f"列 {col}: 最大值 {values.max()} 大于期望的 {rule['max']}")
        
        return {
            'passed': len(range_errors) == 0,
            'message': f"发现 {len(range_errors)} 个范围错误" if range_errors else "数据范围检查通过",
            'details': {
                'range_errors': range_errors,
                'numeric_summary': data.select_dtypes(include=[np.number]).describe().to_dict()
            }
        }
    
    def _check_duplicates(self, data: DataFrameLike, **kwargs) -> Dict[str, Any]:
        """检查重复值"""
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        duplicate_rows = data.duplicated().sum()
        duplicate_percentage = (duplicate_rows / len(data)) * 100
        
        return {
            'passed': duplicate_rows == 0,
            'message': f"发现 {duplicate_rows} 行重复数据 ({duplicate_percentage:.2f}%)",
            'details': {
                'duplicate_rows': int(duplicate_rows),
                'duplicate_percentage': duplicate_percentage,
                'duplicate_indices': data[data.duplicated()].index.tolist()
            }
        }
    
    def _check_uniqueness(self, data: DataFrameLike, **kwargs) -> Dict[str, Any]:
        """检查唯一性"""
        uniqueness_rules = kwargs.get('uniqueness_rules', [])
        uniqueness_errors = []
        
        for col in uniqueness_rules:
            if col in data.columns:
                unique_count = data[col].nunique()
                total_count = len(data[col].dropna())
                
                if unique_count != total_count:
                    uniqueness_errors.append(f"列 {col}: 唯一值数量 {unique_count} 不等于总数量 {total_count}")
        
        return {
            'passed': len(uniqueness_errors) == 0,
            'message': f"发现 {len(uniqueness_errors)} 个唯一性错误" if uniqueness_errors else "唯一性检查通过",
            'details': {
                'uniqueness_errors': uniqueness_errors,
                'uniqueness_stats': {col: data[col].nunique() for col in data.columns}
            }
        }


class AsyncDataPipeline:
    """异步数据处理管道"""
    
    def __init__(self, name: str = "AsyncPipeline"):
        self.name = name
        self.processors = []
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.stats = ProcessingStats()
    
    def add_processor(self, processor: BaseProcessor, **kwargs):
        """添加处理器到管道"""
        self.processors.append((processor, kwargs))
        self.logger.info(f"添加处理器: {processor.name}")
    
    def remove_processor(self, processor_name: str):
        """从管道中移除处理器"""
        self.processors = [
            (proc, cfg) for proc, cfg in self.processors 
            if proc.name != processor_name
        ]
        self.logger.info(f"移除处理器: {processor_name}")
    
    async def execute(self, data: AnyDataFrame, **kwargs) -> AnyDataFrame:
        """执行异步管道"""
        self.stats.start_time = datetime.now()
        self.logger.info(f"开始执行管道: {self.name}")
        
        try:
            result = data
            
            for processor, config in self.processors:
                self.logger.info(f"执行处理器: {processor.name}")
                result = await processor.process(result, **{**config, **kwargs})
                
                # 更新管道统计信息
                self.stats.total_records += processor.stats.total_records
                self.stats.processed_records += processor.stats.processed_records
                self.stats.error_records += processor.stats.error_records
            
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            
            self.logger.info(f"管道执行完成，成功率: {self.stats.success_rate:.2f}%")
            return result
            
        except Exception as e:
            self.logger.error(f"管道执行失败: {str(e)}")
            raise ProcessingError(f"管道执行失败: {str(e)}")
    
    def get_stats(self) -> ProcessingStats:
        """获取管道统计信息"""
        return self.stats
    
    def clear(self):
        """清空管道"""
        self.processors.clear()
        self.logger.info("清空管道")


class MemoryOptimizedProcessor(BaseProcessor):
    """内存优化处理器"""
    
    def __init__(self, chunk_size: int = 10000, memory_limit_mb: int = 512):
        super().__init__("MemoryOptimizedProcessor")
        self.chunk_size = chunk_size
        self.memory_limit_mb = memory_limit_mb
        self.temp_files = []
    
    async def process(self, data: AnyDataFrame, **kwargs) -> AnyDataFrame:
        """内存优化处理"""
        self.stats.start_time = datetime.now()
        
        try:
            if isinstance(data, pd.DataFrame):
                result = await self._process_dataframe(data, **kwargs)
            elif isinstance(data, pd.Series):
                result = await self._process_series(data, **kwargs)
            elif isinstance(data, np.ndarray):
                result = await self._process_array(data, **kwargs)
            else:
                result = data
            
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            
            return result
            
        except Exception as e:
            self.logger.error(f"内存优化处理失败: {str(e)}")
            raise MemoryError(f"内存优化处理失败: {str(e)}")
        finally:
            self._cleanup_temp_files()
    
    async def _process_dataframe(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """处理DataFrame"""
        total_rows = len(data)
        
        if total_rows <= self.chunk_size:
            # 数据量小，直接处理
            return await self._process_chunk(data, **kwargs)
        
        # 大数据量，分块处理
        results = []
        
        for i in range(0, total_rows, self.chunk_size):
            chunk = data.iloc[i:i + self.chunk_size]
            processed_chunk = await self._process_chunk(chunk, **kwargs)
            results.append(processed_chunk)
            
            # 检查内存使用
            self._check_memory_usage()
        
        # 合并结果
        final_result = pd.concat(results, ignore_index=True)
        return final_result
    
    async def _process_series(self, data: pd.Series, **kwargs) -> pd.Series:
        """处理Series"""
        return await self._process_dataframe(data.to_frame(), **kwargs).iloc[:, 0]
    
    async def _process_array(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """处理numpy数组"""
        # 转换为DataFrame进行处理
        df = pd.DataFrame(data)
        processed_df = await self._process_dataframe(df, **kwargs)
        return processed_df.values
    
    async def _process_chunk(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """处理数据块"""
        # 这里可以应用各种数据处理操作
        result = chunk.copy()
        
        # 示例：数据清洗
        if kwargs.get('clean_data', False):
            # 处理缺失值
            result = result.ffill().fillna(method='bfill')
            
            # 移除异常值
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                Q1 = result[col].quantile(0.25)
                Q3 = result[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                result = result[(result[col] >= lower_bound) & (result[col] <= upper_bound)]
        
        return result
    
    def _check_memory_usage(self):
        """检查内存使用情况"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        if memory_mb > self.memory_limit_mb:
            self.logger.warning(f"内存使用 ({memory_mb:.2f}MB) 超过限制 ({self.memory_limit_mb}MB)")
            self._trigger_garbage_collection()
    
    def _trigger_garbage_collection(self):
        """触发垃圾回收"""
        import gc
        gc.collect()
        self.logger.info("触发垃圾回收")
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        import os
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                self.logger.warning(f"清理临时文件失败 {temp_file}: {str(e)}")
        self.temp_files.clear()


class BigDataProcessor(BaseProcessor):
    """大数据处理器"""
    
    def __init__(self, use_multiprocessing: bool = True, max_workers: int = None):
        super().__init__("BigDataProcessor")
        self.use_multiprocessing = use_multiprocessing
        self.max_workers = max_workers or min(4, (os.cpu_count() or 1))
    
    async def process(self, data: AnyDataFrame, **kwargs) -> AnyDataFrame:
        """大数据处理"""
        self.stats.start_time = datetime.now()
        
        try:
            if isinstance(data, pd.DataFrame):
                result = await self._process_large_dataframe(data, **kwargs)
            else:
                result = await self._process_data_parallel(data, **kwargs)
            
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            
            return result
            
        except Exception as e:
            self.logger.error(f"大数据处理失败: {str(e)}")
            raise ProcessingError(f"大数据处理失败: {str(e)}")
    
    async def _process_large_dataframe(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """处理大型DataFrame"""
        operation = kwargs.get('operation', 'transform')
        
        if operation == 'transform':
            return await self._parallel_transform(data, **kwargs)
        elif operation == 'aggregate':
            return await self._parallel_aggregate(data, **kwargs)
        elif operation == 'filter':
            return await self._parallel_filter(data, **kwargs)
        else:
            raise ValueError(f"不支持的操作: {operation}")
    
    async def _parallel_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """并行转换"""
        transform_func = kwargs.get('transform_func')
        if not transform_func:
            raise ValueError("需要提供转换函数")
        
        # 分块处理
        chunk_size = len(data) // self.max_workers
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        if self.use_multiprocessing:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._apply_transform, chunk, transform_func) for chunk in chunks]
                results = [future.result() for future in futures]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._apply_transform, chunk, transform_func) for chunk in chunks]
                results = [future.result() for future in futures]
        
        return pd.concat(results, ignore_index=True)
    
    def _apply_transform(self, chunk: pd.DataFrame, transform_func: Callable) -> pd.DataFrame:
        """应用转换函数"""
        try:
            return transform_func(chunk)
        except Exception as e:
            self.logger.error(f"转换函数执行失败: {str(e)}")
            return chunk
    
    async def _parallel_aggregate(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """并行聚合"""
        group_columns = kwargs.get('group_columns', [])
        agg_functions = kwargs.get('agg_functions', {})
        
        if not group_columns:
            raise ValueError("需要提供分组列")
        
        # 分组并并行处理
        groups = data.groupby(group_columns)
        
        if self.use_multiprocessing:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = []
                for name, group in groups:
                    future = executor.submit(self._aggregate_group, group, agg_functions)
                    futures.append((name, future))
                
                results = []
                for name, future in futures:
                    result = future.result()
                    if isinstance(result, pd.DataFrame):
                        results.append(result)
                    else:
                        # 如果结果是标量，转换为DataFrame
                        result_df = pd.DataFrame([result])
                        result_df.index = [name]
                        results.append(result_df)
        else:
            results = []
            for name, group in groups:
                result = self._aggregate_group(group, agg_functions)
                if isinstance(result, pd.DataFrame):
                    results.append(result)
                else:
                    result_df = pd.DataFrame([result])
                    result_df.index = [name]
                    results.append(result_df)
        
        return pd.concat(results, ignore_index=False)
    
    def _aggregate_group(self, group: pd.DataFrame, agg_functions: Dict[str, List[str]]) -> pd.DataFrame:
        """聚合数据组"""
        return group.agg(agg_functions)
    
    async def _parallel_filter(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """并行过滤"""
        filter_func = kwargs.get('filter_func')
        if not filter_func:
            raise ValueError("需要提供过滤函数")
        
        # 分块处理
        chunk_size = len(data) // self.max_workers
        chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        if self.use_multiprocessing:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._apply_filter, chunk, filter_func) for chunk in chunks]
                results = [future.result() for future in futures]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(self._apply_filter, chunk, filter_func) for chunk in chunks]
                results = [future.result() for future in futures]
        
        # 合并过滤结果
        filtered_results = [result for result in results if result is not None]
        if filtered_results:
            return pd.concat(filtered_results, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _apply_filter(self, chunk: pd.DataFrame, filter_func: Callable) -> pd.DataFrame:
        """应用过滤函数"""
        try:
            return filter_func(chunk)
        except Exception as e:
            self.logger.error(f"过滤函数执行失败: {str(e)}")
            return None
    
    async def _process_data_parallel(self, data: AnyDataFrame, **kwargs) -> AnyDataFrame:
        """并行处理数据"""
        # 对于非DataFrame数据，转换为DataFrame处理
        if isinstance(data, pd.Series):
            df = data.to_frame()
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(data)
        
        processed_df = await self._process_large_dataframe(df, **kwargs)
        
        # 转换回原始格式
        if isinstance(data, pd.Series):
            return processed_df.iloc[:, 0]
        elif isinstance(data, np.ndarray):
            return processed_df.values
        else:
            return processed_df


class DataProcessingTools:
    """J2数据处理工具主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化数据处理工具
        
        Args:
            config: 配置字典，包含各种处理器的配置参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # 初始化各个处理器
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.aggregator = DataAggregator()
        self.validator = DataValidator()
        self.memory_optimizer = MemoryOptimizedProcessor(
            chunk_size=self.config.get('chunk_size', 10000),
            memory_limit_mb=self.config.get('memory_limit_mb', 512)
        )
        self.big_data_processor = BigDataProcessor(
            use_multiprocessing=self.config.get('use_multiprocessing', True),
            max_workers=self.config.get('max_workers', None)
        )
        
        self.logger.info("J2数据处理工具初始化完成")
    
    async def process_data(
        self, 
        data: AnyDataFrame, 
        operations: List[str],
        **kwargs
    ) -> Tuple[AnyDataFrame, Dict[str, ProcessingStats]]:
        """
        执行数据处理操作
        
        Args:
            data: 输入数据
            operations: 操作列表 ['clean', 'transform', 'aggregate', 'validate']
            **kwargs: 各种操作的参数
        
        Returns:
            处理后的数据和统计信息
        """
        self.logger.info(f"开始数据处理，操作: {operations}")
        
        pipeline = AsyncDataPipeline("MainPipeline")
        stats = {}
        
        try:
            # 根据操作类型添加相应的处理器
            for operation in operations:
                if operation == 'clean':
                    pipeline.add_processor(self.cleaner, **kwargs.get('clean_config', {}))
                    stats['clean'] = self.cleaner.stats
                elif operation == 'transform':
                    pipeline.add_processor(self.transformer, **kwargs.get('transform_config', {}))
                    stats['transform'] = self.transformer.stats
                elif operation == 'aggregate':
                    pipeline.add_processor(self.aggregator, **kwargs.get('aggregate_config', {}))
                    stats['aggregate'] = self.aggregator.stats
                elif operation == 'validate':
                    pipeline.add_processor(self.validator, **kwargs.get('validate_config', {}))
                    stats['validate'] = self.validator.stats
                elif operation == 'optimize_memory':
                    pipeline.add_processor(self.memory_optimizer, **kwargs.get('memory_config', {}))
                    stats['memory_optimize'] = self.memory_optimizer.stats
                elif operation == 'big_data':
                    pipeline.add_processor(self.big_data_processor, **kwargs.get('big_data_config', {}))
                    stats['big_data'] = self.big_data_processor.stats
                else:
                    self.logger.warning(f"未知的操作类型: {operation}")
            
            # 执行管道
            result = await pipeline.execute(data, **kwargs)
            
            # 获取管道统计信息
            stats['pipeline'] = pipeline.get_stats()
            
            self.logger.info("数据处理完成")
            return result, stats
            
        except Exception as e:
            self.logger.error(f"数据处理失败: {str(e)}")
            raise ProcessingError(f"数据处理失败: {str(e)}")
    
    async def quick_clean(self, data: DataFrameLike, **kwargs) -> DataFrameLike:
        """
        快速数据清洗
        
        Args:
            data: 输入数据
            **kwargs: 清洗参数
        
        Returns:
            清洗后的数据
        """
        self.logger.info("执行快速数据清洗")
        return await self.cleaner.process(data, **kwargs)
    
    async def quick_transform(self, data: DataFrameLike, **kwargs) -> DataFrameLike:
        """
        快速数据转换
        
        Args:
            data: 输入数据
            **kwargs: 转换参数
        
        Returns:
            转换后的数据
        """
        self.logger.info("执行快速数据转换")
        return await self.transformer.process(data, **kwargs)
    
    async def quick_aggregate(self, data: DataFrameLike, **kwargs) -> DataFrameLike:
        """
        快速数据聚合
        
        Args:
            data: 输入数据
            **kwargs: 聚合参数
        
        Returns:
            聚合后的数据
        """
        self.logger.info("执行快速数据聚合")
        return await self.aggregator.process(data, **kwargs)
    
    async def quick_validate(self, data: DataFrameLike, **kwargs) -> ValidationResult:
        """
        快速数据验证
        
        Args:
            data: 输入数据
            **kwargs: 验证参数
        
        Returns:
            验证结果
        """
        self.logger.info("执行快速数据验证")
        return await self.validator.process(data, **kwargs)
    
    async def batch_process(
        self, 
        data_list: List[AnyDataFrame], 
        operations: List[str],
        **kwargs
    ) -> List[Tuple[AnyDataFrame, Dict[str, ProcessingStats]]]:
        """
        批量处理数据
        
        Args:
            data_list: 数据列表
            operations: 操作列表
            **kwargs: 处理参数
        
        Returns:
            处理结果列表
        """
        self.logger.info(f"开始批量处理 {len(data_list)} 个数据集")
        
        results = []
        
        for i, data in enumerate(data_list):
            try:
                self.logger.info(f"处理数据集 {i+1}/{len(data_list)}")
                result, stats = await self.process_data(data, operations, **kwargs)
                results.append((result, stats))
            except Exception as e:
                self.logger.error(f"数据集 {i+1} 处理失败: {str(e)}")
                results.append((None, {'error': str(e)}))
        
        self.logger.info("批量处理完成")
        return results
    
    def get_processing_report(self, stats: Dict[str, ProcessingStats]) -> str:
        """生成处理报告"""
        report = ["=== J2数据处理工具报告 ===\n"]
        
        for name, stat in stats.items():
            report.append(f"处理器: {name}")
            report.append(f"  总记录数: {stat.total_records}")
            report.append(f"  处理记录数: {stat.processed_records}")
            report.append(f"  错误记录数: {stat.error_records}")
            report.append(f"  成功率: {stat.success_rate:.2f}%")
            report.append(f"  处理时间: {stat.processing_time:.2f}秒")
            report.append("")
        
        return "\n".join(report)
    
    def save_config(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
        self.logger.info(f"配置已保存到: {filepath}")
    
    def load_config(self, filepath: str):
        """从文件加载配置"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        self.logger.info(f"配置已从文件加载: {filepath}")
        
        # 更新处理器配置
        self.memory_optimizer.chunk_size = self.config.get('chunk_size', 10000)
        self.memory_optimizer.memory_limit_mb = self.config.get('memory_limit_mb', 512)
        self.big_data_processor.use_multiprocessing = self.config.get('use_multiprocessing', True)
        self.big_data_processor.max_workers = self.config.get('max_workers', None)


# 使用示例和测试代码
async def main():
    """主函数 - 使用示例"""
    
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'id': range(n_samples),
        'name': [f'User_{i}' for i in range(n_samples)],
        'age': np.random.normal(30, 10, n_samples),
        'salary': np.random.lognormal(10, 1, n_samples),
        'department': np.random.choice(['IT', 'Sales', 'HR', 'Finance'], n_samples),
        'join_date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'performance_score': np.random.beta(2, 5, n_samples) * 100
    })
    
    # 人工添加一些缺失值和异常值
    data.loc[np.random.choice(n_samples, 50, replace=False), 'age'] = np.nan
    data.loc[np.random.choice(n_samples, 30, replace=False), 'salary'] = np.nan
    data.loc[np.random.choice(n_samples, 10, replace=False), 'salary'] = 999999  # 异常值
    
    print("原始数据形状:", data.shape)
    print("\n原始数据前5行:")
    print(data.head())
    
    # 初始化数据处理工具
    config = {
        'chunk_size': 5000,
        'memory_limit_mb': 256,
        'use_multiprocessing': True,
        'max_workers': 4
    }
    
    processor = DataProcessingTools(config)
    
    # 执行完整的数据处理流程
    operations = ['clean', 'transform', 'aggregate', 'validate']
    
    try:
        # 处理数据
        result, stats = await processor.process_data(
            data, 
            operations,
            # 清洗配置
            clean_config={
                'handle_missing': True,
                'missing_strategy': 'mean',
                'handle_outliers': True,
                'outlier_method': 'iqr',
                'outlier_action': 'remove',
                'normalize': True,
                'normalization_method': 'standard'
            },
            # 转换配置
            transform_config={
                'convert_dtype': True,
                'dtype_mapping': {
                    'age': 'int32',
                    'salary': 'float64'
                },
                'feature_engineering': True,
                'feature_config': {
                    'new_features': {
                        'age_group': "pd.cut(data['age'], bins=[0, 25, 35, 45, 100], labels=['young', 'adult', 'middle', 'senior'])",
                        'salary_log': "np.log(data['salary'])"
                    },
                    'extract_time_features': True,
                    'numeric_transformations': True
                }
            },
            # 聚合配置
            aggregate_config={
                'groupby': True,
                'group_config': {
                    'group_columns': ['department'],
                    'agg_functions': {
                        'age': ['mean', 'std'],
                        'salary': ['mean', 'median', 'std'],
                        'performance_score': ['mean', 'std']
                    }
                },
                'rolling_stats': True,
                'rolling_config': {
                    'columns': ['performance_score'],
                    'window': 5,
                    'functions': ['mean', 'std']
                }
            },
            # 验证配置
            validate_config={
                'expected_types': {
                    'age': 'numeric',
                    'salary': 'numeric'
                },
                'range_rules': {
                    'age': {'min': 0, 'max': 100},
                    'salary': {'min': 0, 'max': 1000000}
                },
                'uniqueness_rules': ['id']
            }
        )
        
        print("\n处理后数据形状:", result.shape)
        print("\n处理后数据前5行:")
        print(result.head())
        
        # 生成处理报告
        report = processor.get_processing_report(stats)
        print("\n" + report)
        
        # 快速操作示例
        print("\n=== 快速操作示例 ===")
        
        # 快速清洗
        cleaned_data = await processor.quick_clean(data, missing_strategy='median')
        print(f"快速清洗完成，数据形状: {cleaned_data.shape}")
        
        # 快速转换
        transformed_data = await processor.quick_transform(
            cleaned_data,
            convert_dtype=True,
            dtype_mapping={'age': 'int32'}
        )
        print(f"快速转换完成，数据形状: {transformed_data.shape}")
        
        # 快速聚合
        aggregated_data = await processor.quick_aggregate(
            transformed_data,
            groupby=True,
            group_config={
                'group_columns': ['department'],
                'agg_functions': {
                    'age': ['mean'],
                    'salary': ['mean']
                }
            }
        )
        print(f"快速聚合完成，数据形状: {aggregated_data.shape}")
        
        # 快速验证
        validation_result = await processor.quick_validate(transformed_data)
        print(f"快速验证完成，有效性: {validation_result['is_valid']}")
        if validation_result['errors']:
            print("错误:", validation_result['errors'])
        if validation_result['warnings']:
            print("警告:", validation_result['warnings'])
        
        # 批量处理示例
        print("\n=== 批量处理示例 ===")
        data_list = [data[:500], data[500:]]
        batch_results = await processor.batch_process(
            data_list,
            ['clean', 'transform'],
            clean_config={'missing_strategy': 'drop'},
            transform_config={'convert_dtype': True}
        )
        
        print(f"批量处理完成，处理了 {len(batch_results)} 个数据集")
        for i, (result, stats) in enumerate(batch_results):
            print(f"数据集 {i+1}: 形状 {result.shape if result is not None else 'None'}")
        
    except Exception as e:
        print(f"处理失败: {str(e)}")
        raise


class AdvancedAnalytics(BaseProcessor):
    """高级分析工具类"""
    
    def __init__(self):
        super().__init__("AdvancedAnalytics")
        self.statistical_tests = {
            'ttest': self._ttest_analysis,
            'chisquare': self._chi_square_analysis,
            'anova': self._anova_analysis,
            'correlation': self._correlation_analysis,
            'regression': self._regression_analysis
        }
    
    async def process(self, data: DataFrameLike, **kwargs) -> AnyDataFrame:
        """执行高级分析"""
        self.stats.start_time = datetime.now()
        
        try:
            # 统计分析
            if kwargs.get('statistical_analysis'):
                analysis_config = kwargs.get('analysis_config', {})
                data = await self.statistical_analysis(data, **analysis_config)
            
            # 相关性分析
            if kwargs.get('correlation_analysis'):
                data = await self.correlation_analysis(data)
            
            # 聚类分析
            if kwargs.get('clustering_analysis'):
                cluster_config = kwargs.get('cluster_config', {})
                data = await self.clustering_analysis(data, **cluster_config)
            
            # 趋势分析
            if kwargs.get('trend_analysis'):
                trend_config = kwargs.get('trend_config', {})
                data = await self.trend_analysis(data, **trend_config)
            
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            
            return data
            
        except Exception as e:
            self.logger.error(f"高级分析失败: {str(e)}")
            raise ProcessingError(f"高级分析失败: {str(e)}")
    
    async def statistical_analysis(self, data: DataFrameLike, **kwargs) -> DataFrameLike:
        """统计分析"""
        self._log_processing("统计分析", data.shape if hasattr(data, 'shape') else None)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("统计分析需要DataFrame格式的数据")
        
        result = data.copy()
        
        # 描述性统计
        if kwargs.get('descriptive_stats', True):
            numeric_cols = result.select_dtypes(include=[np.number]).columns
            desc_stats = result[numeric_cols].describe()
            
            # 添加到结果中
            for col in numeric_cols:
                result[f"{col}_skewness"] = result[col].skew()
                result[f"{col}_kurtosis"] = result[col].kurtosis()
                result[f"{col}_cv"] = result[col].std() / result[col].mean()  # 变异系数
        
        # 分组统计
        if kwargs.get('group_statistics'):
            group_cols = kwargs.get('group_columns', [])
            if group_cols:
                for col in result.select_dtypes(include=[np.number]).columns:
                    result[f"{col}_group_mean"] = result.groupby(group_cols)[col].transform('mean')
                    result[f"{col}_group_std"] = result.groupby(group_cols)[col].transform('std')
        
        return result
    
    async def correlation_analysis(self, data: DataFrameLike) -> DataFrameLike:
        """相关性分析"""
        self._log_processing("相关性分析", data.shape if hasattr(data, 'shape') else None)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("相关性分析需要DataFrame格式的数据")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            self.logger.warning("数值列少于2个，跳过相关性分析")
            return data
        
        # 计算相关性矩阵
        corr_matrix = data[numeric_cols].corr()
        
        # 创建相关性特征
        for i, col1 in enumerate(numeric_cols):
            for j, col2 in enumerate(numeric_cols):
                if i < j:  # 避免重复
                    data[f"{col1}_{col2}_corr"] = data[col1] * data[col2]
        
        return data
    
    async def clustering_analysis(
        self, 
        data: DataFrameLike, 
        n_clusters: int = 3,
        algorithm: str = 'kmeans',
        **kwargs
    ) -> DataFrameLike:
        """聚类分析"""
        self._log_processing("聚类分析", data.shape if hasattr(data, 'shape') else None)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("聚类分析需要DataFrame格式的数据")
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            self.logger.warning("未发现数值列，跳过聚类分析")
            return data
        
        # 准备数据
        cluster_data = data[numeric_cols].fillna(data[numeric_cols].mean())
        
        if algorithm == 'kmeans':
            from sklearn.cluster import KMeans
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, **kwargs)
        elif algorithm == 'dbscan':
            clusterer = DBSCAN(**kwargs)
        elif algorithm == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            clusterer = AgglomerativeClustering(n_clusters=n_clusters, **kwargs)
        else:
            raise ValueError(f"不支持的聚类算法: {algorithm}")
        
        # 执行聚类
        cluster_labels = clusterer.fit_predict(cluster_data)
        
        # 添加聚类标签
        data['cluster_label'] = cluster_labels
        
        # 计算聚类统计
        cluster_stats = {}
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_stats[f'cluster_{cluster_id}_size'] = np.sum(cluster_mask)
        
        # 添加聚类大小特征
        for cluster_id, size in cluster_stats.items():
            data[cluster_id] = (data['cluster_label'] == int(cluster_id.split('_')[1])).astype(int)
        
        return data
    
    async def trend_analysis(
        self, 
        data: DataFrameLike, 
        time_column: str,
        value_column: str,
        **kwargs
    ) -> DataFrameLike:
        """趋势分析"""
        self._log_processing("趋势分析", data.shape if hasattr(data, 'shape') else None)
        
        if not isinstance(data, pd.DataFrame):
            raise ValueError("趋势分析需要DataFrame格式的数据")
        
        if time_column not in data.columns or value_column not in data.columns:
            raise ValueError(f"时间列 {time_column} 或值列 {value_column} 不存在")
        
        # 确保时间列是datetime类型
        data[time_column] = pd.to_datetime(data[time_column])
        
        # 按时间排序
        data = data.sort_values(time_column)
        
        # 计算移动平均
        window_sizes = kwargs.get('moving_windows', [3, 7, 14, 30])
        for window in window_sizes:
            data[f"{value_column}_ma_{window}"] = data[value_column].rolling(window=window).mean()
        
        # 计算趋势
        data[f"{value_column}_trend"] = np.polyfit(
            range(len(data)), data[value_column], 1
        )[0]
        
        # 计算变化率
        data[f"{value_column}_pct_change"] = data[value_column].pct_change()
        
        return data


class DataProfiler(BaseProcessor):
    """数据画像工具类"""
    
    def __init__(self):
        super().__init__("DataProfiler")
        self.profile_cache = {}
    
    async def process(self, data: AnyDataFrame, **kwargs) -> Dict[str, Any]:
        """生成数据画像"""
        self.stats.start_time = datetime.now()
        
        try:
            profile = await self.generate_profile(data, **kwargs)
            
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            
            return profile
            
        except Exception as e:
            self.logger.error(f"数据画像生成失败: {str(e)}")
            raise ProcessingError(f"数据画像生成失败: {str(e)}")
    
    async def generate_profile(
        self, 
        data: AnyDataFrame, 
        include_correlations: bool = True,
        include_distributions: bool = True,
        include_missing_analysis: bool = True
    ) -> Dict[str, Any]:
        """生成完整的数据画像"""
        self._log_processing("数据画像生成", data.shape if hasattr(data, 'shape') else None)
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        profile = {
            'basic_info': await self._get_basic_info(data),
            'column_info': await self._get_column_info(data),
            'data_quality': await self._get_data_quality_info(data),
            'statistical_summary': await self._get_statistical_summary(data),
            'missing_data_analysis': {},
            'correlation_analysis': {},
            'distribution_analysis': {}
        }
        
        if include_missing_analysis:
            profile['missing_data_analysis'] = await self._analyze_missing_data(data)
        
        if include_correlations and isinstance(data, pd.DataFrame):
            profile['correlation_analysis'] = await self._analyze_correlations(data)
        
        if include_distributions:
            profile['distribution_analysis'] = await self._analyze_distributions(data)
        
        return profile
    
    async def _get_basic_info(self, data: DataFrameLike) -> Dict[str, Any]:
        """获取基本信息"""
        return {
            'shape': data.shape if hasattr(data, 'shape') else (len(data), 1),
            'size': data.size if hasattr(data, 'size') else len(data),
            'memory_usage': data.memory_usage(deep=True).sum() if isinstance(data, pd.DataFrame) else data.memory_usage(),
            'dtypes': data.dtypes.to_dict() if isinstance(data, pd.DataFrame) else {data.name: data.dtype},
            'index_info': {
                'type': type(data.index).__name__,
                'is_unique': data.index.is_unique,
                'has_duplicates': data.index.duplicated().any()
            }
        }
    
    async def _get_column_info(self, data: DataFrameLike) -> Dict[str, Any]:
        """获取列信息"""
        if isinstance(data, pd.Series):
            return {
                data.name: {
                    'dtype': str(data.dtype),
                    'non_null_count': data.count(),
                    'null_count': data.isnull().sum(),
                    'unique_count': data.nunique(),
                    'sample_values': data.dropna().head(5).tolist()
                }
            }
        
        column_info = {}
        for col in data.columns:
            column_info[col] = {
                'dtype': str(data[col].dtype),
                'non_null_count': data[col].count(),
                'null_count': data[col].isnull().sum(),
                'unique_count': data[col].nunique(),
                'sample_values': data[col].dropna().head(5).tolist()
            }
        
        return column_info
    
    async def _get_data_quality_info(self, data: DataFrameLike) -> Dict[str, Any]:
        """获取数据质量信息"""
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        duplicate_rows = data.duplicated().sum()
        
        return {
            'completeness': ((total_cells - missing_cells) / total_cells) * 100,
            'missing_cells': missing_cells,
            'missing_percentage': (missing_cells / total_cells) * 100,
            'duplicate_rows': duplicate_rows,
            'duplicate_percentage': (duplicate_rows / len(data)) * 100,
            'data_quality_score': self._calculate_quality_score(data)
        }
    
    def _calculate_quality_score(self, data: DataFrameLike) -> float:
        """计算数据质量分数"""
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        # 基础质量指标
        completeness = 1 - (data.isnull().sum().sum() / data.size)
        uniqueness = 1 - (data.duplicated().sum() / len(data))
        
        # 数值列的一致性
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        consistency = 1.0
        if len(numeric_cols) > 0:
            # 检查数值列是否有异常值
            outlier_counts = 0
            total_numeric_cells = 0
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))).sum()
                outlier_counts += outliers
                total_numeric_cells += data[col].count()
            
            consistency = 1 - (outlier_counts / max(total_numeric_cells, 1))
        
        # 综合质量分数
        quality_score = (completeness + uniqueness + consistency) / 3 * 100
        return max(0, min(100, quality_score))
    
    async def _get_statistical_summary(self, data: DataFrameLike) -> Dict[str, Any]:
        """获取统计摘要"""
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        summary = {}
        
        # 数值列统计
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['numeric'] = data[numeric_cols].describe().to_dict()
            
            # 添加额外的统计信息
            for col in numeric_cols:
                summary['numeric'][col].update({
                    'skewness': float(data[col].skew()),
                    'kurtosis': float(data[col].kurtosis()),
                    'variance': float(data[col].var()),
                    'cv': float(data[col].std() / data[col].mean()) if data[col].mean() != 0 else 0
                })
        
        # 分类列统计
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            summary['categorical'] = {}
            for col in categorical_cols:
                value_counts = data[col].value_counts()
                summary['categorical'][col] = {
                    'value_counts': value_counts.head(10).to_dict(),
                    'unique_count': int(data[col].nunique()),
                    'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                    'frequency_of_most_frequent': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0
                }
        
        return summary
    
    async def _analyze_missing_data(self, data: DataFrameLike) -> Dict[str, Any]:
        """分析缺失数据"""
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        missing_info = data.isnull().sum()
        missing_percentage = (missing_info / len(data)) * 100
        
        # 缺失模式分析
        missing_patterns = {}
        for col in data.columns:
            if data[col].isnull().any():
                # 找到与其他列同时缺失的模式
                pattern = data.isnull().sum(axis=1) > 0
                pattern_counts = pattern.value_counts()
                missing_patterns[col] = {
                    'missing_count': int(missing_info[col]),
                    'missing_percentage': float(missing_percentage[col]),
                    'pattern_distribution': pattern_counts.to_dict()
                }
        
        return {
            'missing_by_column': missing_info.to_dict(),
            'missing_percentage_by_column': missing_percentage.to_dict(),
            'missing_patterns': missing_patterns,
            'missing_completeness': (1 - data.isnull().sum().sum() / data.size) * 100
        }
    
    async def _analyze_correlations(self, data: DataFrameLike) -> Dict[str, Any]:
        """分析相关性"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {'message': '数值列少于2个，无法计算相关性'}
        
        corr_matrix = data[numeric_cols].corr()
        
        # 找出强相关性
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # 强相关性阈值
                    strong_correlations.append({
                        'variable1': corr_matrix.columns[i],
                        'variable2': corr_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        
        return {
            'correlation_matrix': corr_matrix.to_dict(),
            'strong_correlations': strong_correlations,
            'highly_correlated_pairs': len(strong_correlations)
        }
    
    async def _analyze_distributions(self, data: DataFrameLike) -> Dict[str, Any]:
        """分析数据分布"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        distributions = {}
        
        for col in numeric_cols:
            series = data[col].dropna()
            if len(series) == 0:
                continue
            
            # 正态性检验
            try:
                from scipy.stats import normaltest
                stat, p_value = normaltest(series)
                is_normal = p_value > 0.05
            except:
                is_normal = None
            
            distributions[col] = {
                'normality_test': {
                    'is_normal': is_normal,
                    'statistic': float(stat) if 'stat' in locals() else None,
                    'p_value': float(p_value) if 'p_value' in locals() else None
                },
                'distribution_shape': self._analyze_distribution_shape(series)
            }
        
        return distributions
    
    def _analyze_distribution_shape(self, series: pd.Series) -> Dict[str, Any]:
        """分析分布形状"""
        skewness = series.skew()
        kurtosis = series.kurtosis()
        
        # 判断分布类型
        if abs(skewness) < 0.5:
            shape = "approximately symmetric"
        elif skewness > 0.5:
            shape = "right-skewed"
        else:
            shape = "left-skewed"
        
        if abs(kurtosis) < 3:
            tail_type = "normal-like tails"
        elif kurtosis > 3:
            tail_type = "heavy tails"
        else:
            tail_type = "light tails"
        
        return {
            'skewness': float(skewness),
            'kurtosis': float(kurtosis),
            'shape_description': shape,
            'tail_description': tail_type
        }


class DataExporter(BaseProcessor):
    """数据导出工具类"""
    
    def __init__(self):
        super().__init__("DataExporter")
        self.export_formats = {
            'csv': self._export_to_csv,
            'excel': self._export_to_excel,
            'json': self._export_to_json,
            'parquet': self._export_to_parquet,
            'pickle': self._export_to_pickle,
            'hdf5': self._export_to_hdf5,
            'sql': self._export_to_sql
        }
    
    async def process(self, data: AnyDataFrame, **kwargs) -> Any:
        """执行数据导出"""
        self.stats.start_time = datetime.now()
        
        try:
            export_format = kwargs.get('format', 'csv')
            output_path = kwargs.get('output_path')
            
            if export_format not in self.export_formats:
                raise ValueError(f"不支持的导出格式: {export_format}")
            
            result = await self.export_formats[export_format](data, **kwargs)
            
            if output_path:
                await self._save_to_file(result, output_path, export_format)
            
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            
            return result
            
        except Exception as e:
            self.logger.error(f"数据导出失败: {str(e)}")
            raise ProcessingError(f"数据导出失败: {str(e)}")
    
    async def _export_to_csv(self, data: AnyDataFrame, **kwargs) -> str:
        """导出为CSV格式"""
        if isinstance(data, pd.DataFrame):
            return data.to_csv(index=False, **kwargs)
        elif isinstance(data, pd.Series):
            return data.to_csv(**kwargs)
        else:
            # 转换为DataFrame再导出
            df = pd.DataFrame(data)
            return df.to_csv(index=False, **kwargs)
    
    async def _export_to_excel(self, data: AnyDataFrame, **kwargs) -> bytes:
        """导出为Excel格式"""
        output = BytesIO()
        
        if isinstance(data, pd.DataFrame):
            data.to_excel(output, index=False, **kwargs)
        elif isinstance(data, pd.Series):
            data.to_excel(output, **kwargs)
        else:
            df = pd.DataFrame(data)
            df.to_excel(output, index=False, **kwargs)
        
        return output.getvalue()
    
    async def _export_to_json(self, data: AnyDataFrame, **kwargs) -> str:
        """导出为JSON格式"""
        if isinstance(data, pd.DataFrame):
            return data.to_json(orient='records', force_ascii=False, indent=2, **kwargs)
        elif isinstance(data, pd.Series):
            return data.to_json(force_ascii=False, indent=2, **kwargs)
        else:
            return json.dumps(data, ensure_ascii=False, indent=2, **kwargs)
    
    async def _export_to_parquet(self, data: AnyDataFrame, **kwargs) -> bytes:
        """导出为Parquet格式"""
        output = BytesIO()
        
        if isinstance(data, pd.DataFrame):
            data.to_parquet(output, **kwargs)
        elif isinstance(data, pd.Series):
            data.to_frame().to_parquet(output, **kwargs)
        else:
            df = pd.DataFrame(data)
            df.to_parquet(output, **kwargs)
        
        return output.getvalue()
    
    async def _export_to_pickle(self, data: AnyDataFrame, **kwargs) -> bytes:
        """导出为Pickle格式"""
        output = BytesIO()
        pickle.dump(data, output, **kwargs)
        return output.getvalue()
    
    async def _export_to_hdf5(self, data: AnyDataFrame, **kwargs) -> bytes:
        """导出为HDF5格式"""
        output = BytesIO()
        
        key = kwargs.get('key', 'data')
        
        if isinstance(data, pd.DataFrame):
            data.to_hdf(output, key=key, mode='w', **kwargs)
        elif isinstance(data, pd.Series):
            data.to_frame().to_hdf(output, key=key, mode='w', **kwargs)
        else:
            df = pd.DataFrame(data)
            df.to_hdf(output, key=key, mode='w', **kwargs)
        
        return output.getvalue()
    
    async def _export_to_sql(self, data: AnyDataFrame, **kwargs) -> str:
        """导出为SQL格式"""
        table_name = kwargs.get('table_name', 'data_table')
        
        if isinstance(data, pd.DataFrame):
            # 生成CREATE TABLE语句
            columns = []
            for col, dtype in data.dtypes.items():
                sql_type = self._pandas_dtype_to_sql(dtype)
                columns.append(f"{col} {sql_type}")
            
            create_table = f"CREATE TABLE {table_name} (\n    " + ",\n    ".join(columns) + "\n);"
            
            # 生成INSERT语句
            insert_statements = []
            for _, row in data.iterrows():
                values = []
                for value in row:
                    if pd.isna(value):
                        values.append("NULL")
                    elif isinstance(value, str):
                        values.append(f"'{value.replace('\'', '\'\'')}'")
                    else:
                        values.append(str(value))
                
                insert_statements.append(
                    f"INSERT INTO {table_name} ({', '.join(data.columns)}) VALUES ({', '.join(values)});"
                )
            
            return create_table + "\n\n" + "\n".join(insert_statements)
        else:
            return "-- SQL导出需要DataFrame格式"
    
    def _pandas_dtype_to_sql(self, dtype) -> str:
        """将pandas数据类型转换为SQL类型"""
        dtype_str = str(dtype)
        
        if 'int' in dtype_str:
            return "INTEGER"
        elif 'float' in dtype_str:
            return "REAL"
        elif 'datetime' in dtype_str:
            return "DATETIME"
        elif 'bool' in dtype_str:
            return "BOOLEAN"
        else:
            return "TEXT"
    
    async def _save_to_file(self, data: Any, output_path: str, format_type: str):
        """保存数据到文件"""
        import os
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        if format_type in ['csv', 'json', 'sql']:
            # 文本格式
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(data)
        else:
            # 二进制格式
            with open(output_path, 'wb') as f:
                f.write(data)
        
        self.logger.info(f"数据已导出到: {output_path}")


class PerformanceMonitor:
    """性能监控类"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def start_timer(self, operation: str):
        """开始计时"""
        import time
        self.start_times[operation] = time.time()
        self.logger.debug(f"开始计时: {operation}")
    
    def end_timer(self, operation: str, additional_metrics: Optional[Dict[str, Any]] = None):
        """结束计时"""
        import time
        if operation not in self.start_times:
            self.logger.warning(f"未找到操作 {operation} 的开始时间")
            return
        
        end_time = time.time()
        duration = end_time - self.start_times[operation]
        
        self.metrics['duration'].append({
            'operation': operation,
            'duration': duration,
            'timestamp': datetime.now().isoformat()
        })
        
        if additional_metrics:
            for key, value in additional_metrics.items():
                self.metrics[key].append({
                    'operation': operation,
                    'value': value,
                    'timestamp': datetime.now().isoformat()
                })
        
        self.logger.info(f"操作 {operation} 耗时: {duration:.4f}秒")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """获取性能摘要"""
        summary = {}
        
        for metric_name, metric_data in self.metrics.items():
            if metric_name == 'duration':
                durations = [item['duration'] for item in metric_data]
                summary[metric_name] = {
                    'total_operations': len(durations),
                    'total_time': sum(durations),
                    'average_time': statistics.mean(durations),
                    'min_time': min(durations),
                    'max_time': max(durations),
                    'operations': metric_data
                }
            else:
                values = [item['value'] for item in metric_data]
                summary[metric_name] = {
                    'count': len(values),
                    'total': sum(values),
                    'average': statistics.mean(values),
                    'min': min(values),
                    'max': max(values),
                    'data': metric_data
                }
        
        return summary
    
    def clear_metrics(self):
        """清除指标"""
        self.metrics.clear()
        self.start_times.clear()
        self.logger.info("性能指标已清除")


# 工具函数
def create_sample_data(n_samples: int = 1000, n_features: int = 10) -> pd.DataFrame:
    """创建示例数据"""
    np.random.seed(42)
    
    # 生成数值特征
    numeric_data = np.random.randn(n_samples, n_features)
    numeric_cols = [f'feature_{i}' for i in range(n_features)]
    
    # 生成分类特征
    categorical_data = {
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'status': np.random.choice(['active', 'inactive', 'pending'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    }
    
    # 生成时间特征
    start_date = datetime(2020, 1, 1)
    time_data = {
        'created_date': [start_date + timedelta(days=i) for i in range(n_samples)],
        'updated_date': [start_date + timedelta(days=i, hours=np.random.randint(0, 24)) 
                        for i in range(n_samples)]
    }
    
    # 组合所有数据
    data = pd.DataFrame(numeric_data, columns=numeric_cols)
    for col, values in categorical_data.items():
        data[col] = values
    for col, values in time_data.items():
        data[col] = values
    
    # 添加一些缺失值和异常值
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    data.loc[missing_indices, 'feature_0'] = np.nan
    
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    data.loc[outlier_indices, 'feature_1'] = np.random.randn(len(outlier_indices)) * 10
    
    return data


def benchmark_processor(processor_class: Type[BaseProcessor], data: pd.DataFrame, **kwargs) -> Dict[str, float]:
    """基准测试处理器性能"""
    import time
    
    processor = processor_class()
    monitor = PerformanceMonitor()
    
    # 预热
    warmup_data = data.sample(min(100, len(data)))
    asyncio.run(processor.process(warmup_data))
    
    # 正式测试
    monitor.start_timer('processing')
    result = asyncio.run(processor.process(data, **kwargs))
    monitor.end_timer('processing', {
        'data_size': len(data),
        'memory_usage': result.memory_usage(deep=True).sum() if hasattr(result, 'memory_usage') else 0
    })
    
    return monitor.get_performance_summary()


# 扩展主类以包含新功能
class ExtendedDataProcessingTools(DataProcessingTools):
    """扩展的数据处理工具类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # 添加新的处理器
        self.analytics = AdvancedAnalytics()
        self.profiler = DataProfiler()
        self.exporter = DataExporter()
        self.performance_monitor = PerformanceMonitor()
    
    async def advanced_analytics(self, data: DataFrameLike, **kwargs) -> DataFrameLike:
        """执行高级分析"""
        self.logger.info("执行高级分析")
        return await self.analytics.process(data, **kwargs)
    
    async def generate_profile(self, data: AnyDataFrame, **kwargs) -> Dict[str, Any]:
        """生成数据画像"""
        self.logger.info("生成数据画像")
        return await self.profiler.process(data, **kwargs)
    
    async def export_data(self, data: AnyDataFrame, **kwargs) -> Any:
        """导出数据"""
        self.logger.info("导出数据")
        return await self.exporter.process(data, **kwargs)
    
    async def comprehensive_analysis(
        self, 
        data: AnyDataFrame, 
        **kwargs
    ) -> Tuple[AnyDataFrame, Dict[str, Any], Dict[str, Any]]:
        """综合分析"""
        self.logger.info("开始综合分析")
        
        # 数据清洗
        cleaned_data = await self.quick_clean(data, **kwargs.get('clean_config', {}))
        
        # 数据画像
        profile = await self.generate_profile(cleaned_data, **kwargs.get('profile_config', {}))
        
        # 高级分析
        analyzed_data = await self.advanced_analytics(cleaned_data, **kwargs.get('analytics_config', {}))
        
        # 性能监控
        perf_summary = self.performance_monitor.get_performance_summary()
        
        return analyzed_data, profile, perf_summary
    
    async def end_to_end_pipeline(
        self, 
        data: AnyDataFrame,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """端到端处理管道"""
        self.logger.info("开始端到端处理管道")
        
        results = {}
        
        try:
            # 步骤1: 数据画像
            self.performance_monitor.start_timer('profiling')
            profile = await self.generate_profile(data)
            self.performance_monitor.end_timer('profiling')
            results['profile'] = profile
            
            # 步骤2: 数据清洗
            self.performance_monitor.start_timer('cleaning')
            cleaned_data = await self.quick_clean(data, **kwargs.get('clean_config', {}))
            self.performance_monitor.end_timer('cleaning')
            results['cleaned_data'] = cleaned_data
            
            # 步骤3: 数据转换
            self.performance_monitor.start_timer('transforming')
            transformed_data = await self.quick_transform(cleaned_data, **kwargs.get('transform_config', {}))
            self.performance_monitor.end_timer('transforming')
            results['transformed_data'] = transformed_data
            
            # 步骤4: 高级分析
            self.performance_monitor.start_timer('analytics')
            analyzed_data = await self.advanced_analytics(transformed_data, **kwargs.get('analytics_config', {}))
            self.performance_monitor.end_timer('analytics')
            results['analyzed_data'] = analyzed_data
            
            # 步骤5: 数据验证
            self.performance_monitor.start_timer('validation')
            validation_result = await self.quick_validate(analyzed_data, **kwargs.get('validate_config', {}))
            self.performance_monitor.end_timer('validation')
            results['validation'] = validation_result
            
            # 步骤6: 数据导出
            if output_path:
                self.performance_monitor.start_timer('exporting')
                export_result = await self.export_data(
                    analyzed_data, 
                    output_path=output_path,
                    **kwargs.get('export_config', {})
                )
                self.performance_monitor.end_timer('exporting')
                results['export'] = export_result
            
            # 性能摘要
            results['performance'] = self.performance_monitor.get_performance_summary()
            
            self.logger.info("端到端处理管道完成")
            return results
            
        except Exception as e:
            self.logger.error(f"端到端处理管道失败: {str(e)}")
            raise ProcessingError(f"端到端处理管道失败: {str(e)}")


# 扩展主函数以展示新功能
async def extended_main():
    """扩展的主函数 - 展示所有功能"""
    
    print("=== J2数据处理工具 - 完整功能演示 ===\n")
    
    # 创建示例数据
    print("1. 创建示例数据...")
    data = create_sample_data(n_samples=2000, n_features=8)
    print(f"数据形状: {data.shape}")
    print(f"列名: {list(data.columns)}")
    print()
    
    # 初始化扩展的数据处理工具
    print("2. 初始化扩展数据处理工具...")
    config = {
        'chunk_size': 10000,
        'memory_limit_mb': 1024,
        'use_multiprocessing': True,
        'max_workers': 4
    }
    
    processor = ExtendedDataProcessingTools(config)
    print("扩展数据处理工具初始化完成")
    print()
    
    # 端到端处理管道
    print("3. 执行端到端处理管道...")
    try:
        results = await processor.end_to_end_pipeline(
            data,
            output_path='/tmp/processed_data.csv',
            clean_config={
                'missing_strategy': 'mean',
                'handle_outliers': True,
                'outlier_method': 'iqr',
                'normalize': True
            },
            transform_config={
                'convert_dtype': True,
                'feature_engineering': True,
                'feature_config': {
                    'new_features': {
                        'feature_sum': 'data[["feature_0", "feature_1", "feature_2"]].sum(axis=1)',
                        'feature_ratio': 'data["feature_0"] / (data["feature_1"] + 1e-8)'
                    },
                    'extract_time_features': True,
                    'numeric_transformations': True
                }
            },
            analytics_config={
                'statistical_analysis': True,
                'correlation_analysis': True,
                'clustering_analysis': True,
                'cluster_config': {
                    'n_clusters': 3,
                    'algorithm': 'kmeans'
                },
                'trend_analysis': True,
                'trend_config': {
                    'time_column': 'created_date',
                    'value_column': 'feature_0',
                    'moving_windows': [3, 7, 14]
                }
            },
            validate_config={
                'expected_types': {
                    'feature_0': 'numeric',
                    'feature_1': 'numeric'
                },
                'range_rules': {
                    'feature_0': {'min': -5, 'max': 5},
                    'feature_1': {'min': -5, 'max': 5}
                }
            },
            export_config={
                'format': 'csv',
                'index': False
            }
        )
        
        print("端到端处理管道成功完成!")
        print()
        
        # 显示结果摘要
        print("4. 结果摘要:")
        print(f"  - 数据画像: {len(results['profile'])} 个部分")
        print(f"  - 清洗后数据形状: {results['cleaned_data'].shape}")
        print(f"  - 转换后数据形状: {results['transformed_data'].shape}")
        print(f"  - 分析后数据形状: {results['analyzed_data'].shape}")
        print(f"  - 数据验证结果: {'通过' if results['validation']['is_valid'] else '失败'}")
        print(f"  - 性能监控: {len(results['performance'])} 个指标")
        print()
        
        # 显示数据画像摘要
        print("5. 数据画像摘要:")
        profile = results['profile']
        print(f"  - 数据完整性: {profile['data_quality']['completeness']:.2f}%")
        print(f"  - 数据质量分数: {profile['data_quality']['data_quality_score']:.2f}")
        print(f"  - 重复行数: {profile['data_quality']['duplicate_rows']}")
        print()
        
        # 显示性能摘要
        print("6. 性能摘要:")
        perf = results['performance']
        for metric_name, metric_data in perf.items():
            if metric_name == 'duration':
                print(f"  - 总操作数: {metric_data['total_operations']}")
                print(f"  - 总处理时间: {metric_data['total_time']:.4f}秒")
                print(f"  - 平均处理时间: {metric_data['average_time']:.4f}秒")
            else:
                print(f"  - {metric_name}: {metric_data['average']:.4f}")
        print()
        
        # 单独功能演示
        print("7. 单独功能演示:")
        
        # 快速分析
        print("  a) 快速数据分析...")
        analysis_data = await processor.advanced_analytics(
            data.head(500),
            statistical_analysis=True,
            clustering_analysis=True,
            cluster_config={'n_clusters': 2}
        )
        print(f"     分析后数据形状: {analysis_data.shape}")
        
        # 数据画像
        print("  b) 详细数据画像...")
        detailed_profile = await processor.generate_profile(
            data,
            include_correlations=True,
            include_distributions=True
        )
        print(f"     画像包含 {len(detailed_profile)} 个分析维度")
        
        # 数据导出
        print("  c) 数据导出...")
        export_result = await processor.export_data(
            analysis_data,
            format='json',
            output_path='/tmp/exported_data.json'
        )
        print(f"     数据已导出，格式: JSON")
        
        print()
        print("=== 所有功能演示完成 ===")
        
    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


class DataQualityAssessment(BaseProcessor):
    """数据质量评估工具类"""
    
    def __init__(self):
        super().__init__("DataQualityAssessment")
        self.quality_dimensions = {
            'completeness': self._assess_completeness,
            'accuracy': self._assess_accuracy,
            'consistency': self._assess_consistency,
            'timeliness': self._assess_timeliness,
            'validity': self._assess_validity,
            'uniqueness': self._assess_uniqueness
        }
    
    async def process(self, data: AnyDataFrame, **kwargs) -> Dict[str, Any]:
        """执行数据质量评估"""
        self.stats.start_time = datetime.now()
        
        try:
            assessment = await self.assess_data_quality(data, **kwargs)
            
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            
            return assessment
            
        except Exception as e:
            self.logger.error(f"数据质量评估失败: {str(e)}")
            raise ProcessingError(f"数据质量评估失败: {str(e)}")
    
    async def assess_data_quality(
        self, 
        data: AnyDataFrame, 
        dimensions: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """评估数据质量"""
        self._log_processing("数据质量评估", data.shape if hasattr(data, 'shape') else None)
        
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        if dimensions is None:
            dimensions = list(self.quality_dimensions.keys())
        
        assessment = {
            'overall_score': 0.0,
            'dimension_scores': {},
            'recommendations': [],
            'assessment_details': {}
        }
        
        total_score = 0
        valid_dimensions = 0
        
        for dimension in dimensions:
            if dimension in self.quality_dimensions:
                try:
                    score, details = await self.quality_dimensions[dimension](data, **kwargs)
                    assessment['dimension_scores'][dimension] = score
                    assessment['assessment_details'][dimension] = details
                    total_score += score
                    valid_dimensions += 1
                except Exception as e:
                    self.logger.warning(f"维度 {dimension} 评估失败: {str(e)}")
        
        if valid_dimensions > 0:
            assessment['overall_score'] = total_score / valid_dimensions
        
        # 生成建议
        assessment['recommendations'] = self._generate_recommendations(assessment)
        
        return assessment
    
    async def _assess_completeness(self, data: DataFrameLike, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """评估完整性"""
        total_cells = data.size
        missing_cells = data.isnull().sum().sum()
        completeness_score = ((total_cells - missing_cells) / total_cells) * 100
        
        details = {
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'missing_percentage': (missing_cells / total_cells) * 100,
            'column_completeness': {}
        }
        
        for col in data.columns:
            col_missing = data[col].isnull().sum()
            details['column_completeness'][col] = {
                'missing_count': int(col_missing),
                'completeness_percentage': ((len(data) - col_missing) / len(data)) * 100
            }
        
        return completeness_score, details
    
    async def _assess_accuracy(self, data: DataFrameLike, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """评估准确性"""
        accuracy_score = 100.0  # 基础分数
        details = {
            'data_type_accuracy': {},
            'range_accuracy': {},
            'format_accuracy': {}
        }
        
        # 数据类型准确性检查
        for col in data.columns:
            expected_type = kwargs.get('expected_types', {}).get(col)
            if expected_type:
                actual_type = str(data[col].dtype)
                if expected_type.lower() in actual_type.lower():
                    details['data_type_accuracy'][col] = {'status': 'correct', 'type': actual_type}
                else:
                    details['data_type_accuracy'][col] = {'status': 'incorrect', 'type': actual_type}
                    accuracy_score -= 5
        
        # 范围准确性检查
        range_rules = kwargs.get('range_rules', {})
        for col, rule in range_rules.items():
            if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                values = data[col].dropna()
                violations = 0
                if 'min' in rule:
                    violations += (values < rule['min']).sum()
                if 'max' in rule:
                    violations += (values > rule['max']).sum()
                
                details['range_accuracy'][col] = {
                    'violations': int(violations),
                    'violation_percentage': (violations / len(values)) * 100 if len(values) > 0 else 0
                }
                
                if violations > 0:
                    accuracy_score -= min(10, violations * 0.1)
        
        return max(0, accuracy_score), details
    
    async def _assess_consistency(self, data: DataFrameLike, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """评估一致性"""
        consistency_score = 100.0
        details = {
            'format_consistency': {},
            'value_consistency': {},
            'structural_consistency': {}
        }
        
        # 格式一致性检查
        for col in data.columns:
            if data[col].dtype == 'object':
                # 检查字符串格式一致性
                sample_values = data[col].dropna().head(100)
                if len(sample_values) > 0:
                    # 简单的格式检查示例
                    has_numbers = sample_values.str.contains(r'\d').any()
                    has_letters = sample_values.str.contains(r'[a-zA-Z]').any()
                    
                    details['format_consistency'][col] = {
                        'has_numbers': bool(has_numbers),
                        'has_letters': bool(has_letters),
                        'sample_formats': sample_values.head(5).tolist()
                    }
        
        # 值一致性检查
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            unique_values = data[col].nunique()
            total_values = len(data[col].dropna())
            
            details['value_consistency'][col] = {
                'unique_count': int(unique_values),
                'total_count': int(total_values),
                'uniqueness_ratio': unique_values / total_values if total_values > 0 else 0
            }
            
            # 如果唯一值过多，可能存在一致性问题
            if unique_values > total_values * 0.8:
                consistency_score -= 5
        
        return max(0, consistency_score), details
    
    async def _assess_timeliness(self, data: DataFrameLike, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """评估时效性"""
        timeliness_score = 100.0
        details = {
            'data_freshness': {},
            'update_frequency': {},
            'time_gaps': {}
        }
        
        # 检查时间列
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_cols:
            if col in data.columns:
                time_diff = datetime.now() - data[col].max()
                details['data_freshness'][col] = {
                    'latest_timestamp': data[col].max().isoformat(),
                    'hours_old': time_diff.total_seconds() / 3600,
                    'is_recent': time_diff.total_seconds() < 86400  # 24小时
                }
                
                # 如果数据太旧，降低分数
                if time_diff.total_seconds() > 86400 * 7:  # 7天
                    timeliness_score -= 10
        
        return timeliness_score, details
    
    async def _assess_validity(self, data: DataFrameLike, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """评估有效性"""
        validity_score = 100.0
        details = {
            'data_type_validity': {},
            'pattern_validity': {},
            'business_rule_validity': {}
        }
        
        # 数据类型有效性
        for col in data.columns:
            if data[col].dtype == 'object':
                # 检查是否可以转换为数值
                try:
                    pd.to_numeric(data[col], errors='raise')
                    details['data_type_validity'][col] = {'status': 'valid_numeric', 'convertible': True}
                except:
                    details['data_type_validity'][col] = {'status': 'non_numeric', 'convertible': False}
        
        # 业务规则有效性
        business_rules = kwargs.get('business_rules', {})
        for col, rule in business_rules.items():
            if col in data.columns:
                valid_count = 0
                total_count = len(data[col].dropna())
                
                for value in data[col].dropna():
                    if rule.get('allowed_values') and value in rule['allowed_values']:
                        valid_count += 1
                    elif rule.get('pattern') and re.match(rule['pattern'], str(value)):
                        valid_count += 1
                
                validity_percentage = (valid_count / total_count) * 100 if total_count > 0 else 0
                details['business_rule_validity'][col] = {
                    'valid_count': valid_count,
                    'total_count': total_count,
                    'validity_percentage': validity_percentage
                }
                
                if validity_percentage < 95:
                    validity_score -= 5
        
        return max(0, validity_score), details
    
    async def _assess_uniqueness(self, data: DataFrameLike, **kwargs) -> Tuple[float, Dict[str, Any]]:
        """评估唯一性"""
        uniqueness_score = 100.0
        details = {
            'duplicate_rows': {},
            'duplicate_values': {},
            'primary_key_uniqueness': {}
        }
        
        # 行级重复检查
        total_rows = len(data)
        duplicate_rows = data.duplicated().sum()
        uniqueness_percentage = ((total_rows - duplicate_rows) / total_rows) * 100
        
        details['duplicate_rows'] = {
            'total_rows': total_rows,
            'duplicate_rows': int(duplicate_rows),
            'uniqueness_percentage': uniqueness_percentage
        }
        
        if duplicate_rows > 0:
            uniqueness_score -= min(20, duplicate_rows * 0.1)
        
        # 列级重复检查
        for col in data.columns:
            unique_count = data[col].nunique()
            total_count = len(data[col].dropna())
            col_uniqueness = (unique_count / total_count) * 100 if total_count > 0 else 0
            
            details['duplicate_values'][col] = {
                'unique_count': int(unique_count),
                'total_count': int(total_count),
                'uniqueness_percentage': col_uniqueness
            }
        
        return max(0, uniqueness_score), details
    
    def _generate_recommendations(self, assessment: Dict[str, Any]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        dimension_scores = assessment['dimension_scores']
        
        # 基于各维度分数生成建议
        if dimension_scores.get('completeness', 0) < 90:
            recommendations.append("数据完整性较低，建议加强数据收集和验证流程")
        
        if dimension_scores.get('accuracy', 0) < 90:
            recommendations.append("数据准确性有待提升，建议实施更严格的数据验证规则")
        
        if dimension_scores.get('consistency', 0) < 90:
            recommendations.append("数据一致性存在问题，建议统一数据格式和标准")
        
        if dimension_scores.get('uniqueness', 0) < 90:
            recommendations.append("数据唯一性不足，建议去除重复记录并建立唯一性约束")
        
        if assessment['overall_score'] < 70:
            recommendations.append("整体数据质量需要全面改进，建议制定数据质量管理策略")
        elif assessment['overall_score'] < 85:
            recommendations.append("数据质量良好，但仍有改进空间")
        else:
            recommendations.append("数据质量优秀，继续保持当前的数据管理标准")
        
        return recommendations


class DataMonitoring(BaseProcessor):
    """数据监控工具类"""
    
    def __init__(self, monitoring_rules: Optional[Dict[str, Any]] = None):
        super().__init__("DataMonitoring")
        self.monitoring_rules = monitoring_rules or {}
        self.alert_thresholds = {
            'data_volume_change': 0.2,  # 20%变化
            'quality_score_drop': 0.1,  # 10%下降
            'missing_data_increase': 0.05  # 5%增加
        }
        self.baseline_metrics = {}
        self.monitoring_history = []
    
    async def process(self, data: AnyDataFrame, **kwargs) -> Dict[str, Any]:
        """执行数据监控"""
        self.stats.start_time = datetime.now()
        
        try:
            monitoring_result = await self.monitor_data(data, **kwargs)
            
            self.stats.end_time = datetime.now()
            if self.stats.start_time:
                self.stats.processing_time = (self.stats.end_time - self.stats.start_time).total_seconds()
            
            return monitoring_result
            
        except Exception as e:
            self.logger.error(f"数据监控失败: {str(e)}")
            raise ProcessingError(f"数据监控失败: {str(e)}")
    
    async def monitor_data(
        self, 
        data: AnyDataFrame, 
        establish_baseline: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """监控数据"""
        self._log_processing("数据监控", data.shape if hasattr(data, 'shape') else None)
        
        current_metrics = await self._calculate_current_metrics(data)
        
        monitoring_result = {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': current_metrics,
            'alerts': [],
            'trends': {},
            'recommendations': []
        }
        
        # 建立基线
        if establish_baseline or not self.baseline_metrics:
            self.baseline_metrics = current_metrics.copy()
            monitoring_result['baseline_established'] = True
            self.logger.info("数据监控基线已建立")
        else:
            # 比较当前指标与基线
            comparison = self._compare_with_baseline(current_metrics)
            monitoring_result['comparison_with_baseline'] = comparison
            
            # 生成警报
            alerts = self._generate_alerts(comparison)
            monitoring_result['alerts'] = alerts
            
            # 分析趋势
            trends = self._analyze_trends(current_metrics)
            monitoring_result['trends'] = trends
        
        # 保存监控历史
        self.monitoring_history.append({
            'timestamp': monitoring_result['timestamp'],
            'metrics': current_metrics
        })
        
        # 限制历史记录大小
        if len(self.monitoring_history) > 100:
            self.monitoring_history = self.monitoring_history[-100:]
        
        return monitoring_result
    
    async def _calculate_current_metrics(self, data: AnyDataFrame) -> Dict[str, Any]:
        """计算当前指标"""
        if isinstance(data, pd.Series):
            data = data.to_frame()
        
        metrics = {
            'data_volume': {
                'total_rows': len(data),
                'total_columns': len(data.columns),
                'total_cells': data.size,
                'memory_usage': data.memory_usage(deep=True).sum()
            },
            'data_quality': {
                'completeness': ((data.size - data.isnull().sum().sum()) / data.size) * 100,
                'duplicate_rows': data.duplicated().sum(),
                'unique_columns': len(data.columns[data.nunique() == len(data)])
            },
            'data_distribution': {},
            'column_statistics': {}
        }
        
        # 数值列统计
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            metrics['data_distribution']['numeric_columns'] = len(numeric_cols)
            metrics['data_distribution']['numeric_stats'] = data[numeric_cols].describe().to_dict()
        
        # 分类列统计
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            metrics['data_distribution']['categorical_columns'] = len(categorical_cols)
            metrics['data_distribution']['categorical_stats'] = {}
            for col in categorical_cols:
                metrics['data_distribution']['categorical_stats'][col] = {
                    'unique_count': data[col].nunique(),
                    'most_frequent': data[col].mode().iloc[0] if len(data[col].mode()) > 0 else None,
                    'frequency_distribution': data[col].value_counts().head(5).to_dict()
                }
        
        # 列级统计
        for col in data.columns:
            metrics['column_statistics'][col] = {
                'dtype': str(data[col].dtype),
                'non_null_count': data[col].count(),
                'null_count': data[col].isnull().sum(),
                'unique_count': data[col].nunique(),
                'missing_percentage': (data[col].isnull().sum() / len(data)) * 100
            }
        
        return metrics
    
    def _compare_with_baseline(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """与基线比较"""
        comparison = {}
        
        # 数据量变化
        volume_change = (
            (current_metrics['data_volume']['total_rows'] - self.baseline_metrics['data_volume']['total_rows']) /
            self.baseline_metrics['data_volume']['total_rows']
        )
        comparison['volume_change'] = volume_change
        
        # 质量变化
        quality_change = (
            current_metrics['data_quality']['completeness'] - 
            self.baseline_metrics['data_quality']['completeness']
        )
        comparison['quality_change'] = quality_change
        
        # 缺失数据变化
        missing_change = (
            current_metrics['data_quality']['completeness'] - 
            self.baseline_metrics['data_quality']['completeness']
        )
        comparison['missing_data_change'] = -missing_change  # 负值表示缺失增加
        
        return comparison
    
    def _generate_alerts(self, comparison: Dict[str, Any]) -> List[Dict[str, Any]]:
        """生成警报"""
        alerts = []
        
        # 数据量警报
        if abs(comparison['volume_change']) > self.alert_thresholds['data_volume_change']:
            alert_type = "increase" if comparison['volume_change'] > 0 else "decrease"
            alerts.append({
                'type': 'data_volume_change',
                'severity': 'warning',
                'message': f"数据量{alert_type} {abs(comparison['volume_change']*100):.1f}%",
                'threshold': self.alert_thresholds['data_volume_change'],
                'actual_change': comparison['volume_change']
            })
        
        # 质量下降警报
        if comparison['quality_change'] < -self.alert_thresholds['quality_score_drop']:
            alerts.append({
                'type': 'quality_degradation',
                'severity': 'error',
                'message': f"数据质量下降 {abs(comparison['quality_change']):.1f}%",
                'threshold': self.alert_thresholds['quality_score_drop'],
                'actual_change': comparison['quality_change']
            })
        
        # 缺失数据增加警报
        if comparison['missing_data_change'] > self.alert_thresholds['missing_data_increase']:
            alerts.append({
                'type': 'missing_data_increase',
                'severity': 'warning',
                'message': f"缺失数据增加 {comparison['missing_data_change']*100:.1f}%",
                'threshold': self.alert_thresholds['missing_data_increase'],
                'actual_change': comparison['missing_data_change']
            })
        
        return alerts
    
    def _analyze_trends(self, current_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """分析趋势"""
        trends = {}
        
        if len(self.monitoring_history) >= 2:
            # 数据量趋势
            recent_volumes = [h['metrics']['data_volume']['total_rows'] for h in self.monitoring_history[-5:]]
            if len(recent_volumes) >= 3:
                # 简单线性趋势
                x = list(range(len(recent_volumes)))
                slope = np.polyfit(x, recent_volumes, 1)[0]
                trends['volume_trend'] = {
                    'direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
                    'slope': slope
                }
            
            # 质量趋势
            recent_quality = [h['metrics']['data_quality']['completeness'] for h in self.monitoring_history[-5:]]
            if len(recent_quality) >= 3:
                quality_slope = np.polyfit(range(len(recent_quality)), recent_quality, 1)[0]
                trends['quality_trend'] = {
                    'direction': 'improving' if quality_slope > 0 else 'declining' if quality_slope < 0 else 'stable',
                    'slope': quality_slope
                }
        
        return trends
    
    def set_alert_threshold(self, alert_type: str, threshold: float):
        """设置警报阈值"""
        if alert_type in self.alert_thresholds:
            self.alert_thresholds[alert_type] = threshold
            self.logger.info(f"警报阈值已更新: {alert_type} = {threshold}")
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """获取监控摘要"""
        if not self.monitoring_history:
            return {'message': '暂无监控历史数据'}
        
        latest = self.monitoring_history[-1]
        summary = {
            'latest_timestamp': latest['timestamp'],
            'monitoring_period_days': len(self.monitoring_history),
            'current_metrics': latest['metrics'],
            'baseline_established': bool(self.baseline_metrics),
            'recent_alerts': len([h for h in self.monitoring_history[-10:] if 'alerts' in h])
        }
        
        return summary


# 扩展工具函数
def validate_data_schema(data: DataFrameLike, schema: Dict[str, Any]) -> ValidationResult:
    """验证数据模式"""
    result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'details': {}
    }
    
    if isinstance(data, pd.Series):
        data = data.to_frame()
    
    # 检查必需的列
    required_columns = schema.get('required_columns', [])
    missing_columns = [col for col in required_columns if col not in data.columns]
    if missing_columns:
        result['errors'].append(f"缺少必需的列: {missing_columns}")
        result['is_valid'] = False
    
    # 检查列类型
    column_types = schema.get('column_types', {})
    for col, expected_type in column_types.items():
        if col in data.columns:
            actual_type = str(data[col].dtype)
            if expected_type.lower() not in actual_type.lower():
                result['warnings'].append(f"列 {col} 类型不匹配: 期望 {expected_type}, 实际 {actual_type}")
    
    # 检查数值范围
    value_ranges = schema.get('value_ranges', {})
    for col, range_config in value_ranges.items():
        if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
            values = data[col].dropna()
            violations = []
            if 'min' in range_config and (values < range_config['min']).any():
                violations.append(f"小于最小值 {range_config['min']}")
            if 'max' in range_config and (values > range_config['max']).any():
                violations.append(f"大于最大值 {range_config['max']}")
            
            if violations:
                result['errors'].append(f"列 {col} 值范围违规: {', '.join(violations)}")
                result['is_valid'] = False
    
    result['details'] = {
        'total_columns': len(data.columns),
        'required_columns_count': len(required_columns),
        'validated_columns': len([col for col in column_types.keys() if col in data.columns])
    }
    
    return result


def create_data_pipeline(
    steps: List[Tuple[str, BaseProcessor, Dict[str, Any]]]
) -> AsyncDataPipeline:
    """创建数据处理管道"""
    pipeline = AsyncDataPipeline("CustomPipeline")
    
    for step_name, processor, config in steps:
        pipeline.add_processor(processor, **config)
        logger.info(f"添加管道步骤: {step_name}")
    
    return pipeline


def parallel_process_data(
    data_list: List[AnyDataFrame],
    processor: BaseProcessor,
    **kwargs
) -> List[Tuple[AnyDataFrame, ProcessingStats]]:
    """并行处理多个数据集"""
    async def process_single_dataset(data):
        stats = ProcessingStats()
        try:
            result = await processor.process(data, **kwargs)
            stats.processed_records = len(data) if hasattr(data, '__len__') else 1
            return result, stats
        except Exception as e:
            stats.error_records = len(data) if hasattr(data, '__len__') else 1
            logger.error(f"数据集处理失败: {str(e)}")
            return None, stats
    
    async def run_parallel():
        tasks = [process_single_dataset(data) for data in data_list]
        return await asyncio.gather(*tasks)
    
    return asyncio.run(run_parallel())


# 扩展主类
class CompleteDataProcessingTools(ExtendedDataProcessingTools):
    """完整的数据处理工具类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # 添加新的处理器
        self.quality_assessor = DataQualityAssessment()
        self.monitor = DataMonitoring()
    
    async def assess_quality(self, data: AnyDataFrame, **kwargs) -> Dict[str, Any]:
        """评估数据质量"""
        self.logger.info("评估数据质量")
        return await self.quality_assessor.process(data, **kwargs)
    
    async def monitor_data(
        self, 
        data: AnyDataFrame, 
        establish_baseline: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """监控数据"""
        self.logger.info("监控数据")
        return await self.monitor.process(data, establish_baseline=establish_baseline, **kwargs)
    
    async def full_analysis_pipeline(
        self, 
        data: AnyDataFrame,
        output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """完整分析管道"""
        self.logger.info("开始完整分析管道")
        
        results = {}
        
        try:
            # 步骤1: 数据画像
            self.performance_monitor.start_timer('profiling')
            profile = await self.generate_profile(data)
            self.performance_monitor.end_timer('profiling')
            results['profile'] = profile
            
            # 步骤2: 质量评估
            self.performance_monitor.start_timer('quality_assessment')
            quality_assessment = await self.assess_quality(data)
            self.performance_monitor.end_timer('quality_assessment')
            results['quality_assessment'] = quality_assessment
            
            # 步骤3: 数据监控
            self.performance_monitor.start_timer('monitoring')
            monitoring_result = await self.monitor_data(data, establish_baseline=True)
            self.performance_monitor.end_timer('monitoring')
            results['monitoring'] = monitoring_result
            
            # 步骤4: 数据清洗
            self.performance_monitor.start_timer('cleaning')
            cleaned_data = await self.quick_clean(data, **kwargs.get('clean_config', {}))
            self.performance_monitor.end_timer('cleaning')
            results['cleaned_data'] = cleaned_data
            
            # 步骤5: 数据转换
            self.performance_monitor.start_timer('transforming')
            transformed_data = await self.quick_transform(cleaned_data, **kwargs.get('transform_config', {}))
            self.performance_monitor.end_timer('transforming')
            results['transformed_data'] = transformed_data
            
            # 步骤6: 高级分析
            self.performance_monitor.start_timer('analytics')
            analyzed_data = await self.advanced_analytics(transformed_data, **kwargs.get('analytics_config', {}))
            self.performance_monitor.end_timer('analytics')
            results['analyzed_data'] = analyzed_data
            
            # 步骤7: 数据验证
            self.performance_monitor.start_timer('validation')
            validation_result = await self.quick_validate(analyzed_data, **kwargs.get('validate_config', {}))
            self.performance_monitor.end_timer('validation')
            results['validation'] = validation_result
            
            # 步骤8: 数据导出
            if output_path:
                self.performance_monitor.start_timer('exporting')
                export_result = await self.export_data(
                    analyzed_data, 
                    output_path=output_path,
                    **kwargs.get('export_config', {})
                )
                self.performance_monitor.end_timer('exporting')
                results['export'] = export_result
            
            # 性能摘要
            results['performance'] = self.performance_monitor.get_performance_summary()
            
            # 生成综合报告
            results['comprehensive_report'] = self._generate_comprehensive_report(results)
            
            self.logger.info("完整分析管道完成")
            return results
            
        except Exception as e:
            self.logger.error(f"完整分析管道失败: {str(e)}")
            raise ProcessingError(f"完整分析管道失败: {str(e)}")
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合报告"""
        report = {
            'summary': {},
            'key_findings': [],
            'recommendations': [],
            'next_steps': []
        }
        
        # 摘要信息
        if 'profile' in results:
            profile = results['profile']
            report['summary']['data_quality_score'] = profile.get('data_quality', {}).get('data_quality_score', 0)
            report['summary']['data_completeness'] = profile.get('data_quality', {}).get('completeness', 0)
        
        if 'quality_assessment' in results:
            assessment = results['quality_assessment']
            report['summary']['overall_quality_score'] = assessment.get('overall_score', 0)
            report['summary']['quality_dimensions'] = assessment.get('dimension_scores', {})
        
        # 关键发现
        if 'validation' in results:
            validation = results['validation']
            if not validation.get('is_valid', True):
                report['key_findings'].append("数据验证发现问题，需要进一步处理")
        
        if 'monitoring' in results:
            monitoring = results['monitoring']
            if monitoring.get('alerts'):
                report['key_findings'].append(f"监控系统检测到 {len(monitoring['alerts'])} 个警报")
        
        # 建议和下一步
        if 'quality_assessment' in results:
            recommendations = results['quality_assessment'].get('recommendations', [])
            report['recommendations'].extend(recommendations)
        
        report['next_steps'] = [
            "根据质量评估结果优化数据收集流程",
            "建立定期数据质量监控机制",
            "实施数据治理最佳实践",
            "持续改进数据处理管道"
        ]
        
        return report


if __name__ == "__main__":
    # 运行扩展示例
    asyncio.run(extended_main())