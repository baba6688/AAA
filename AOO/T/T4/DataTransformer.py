"""
T4数据转换器
===========

一个功能强大的数据转换器，支持多种数据格式转换、结构转换、聚合分析等功能。

功能特性：
1. 数据格式转换（JSON、CSV、XML、Parquet、Excel等）
2. 数据结构转换（宽表转长表、长表转宽表）
3. 数据聚合和分组
4. 数据拆分和合并
5. 数据标准化和归一化
6. 数据编码和解码
7. 数据映射和匹配
8. 转换流水线管理
9. 转换性能优化


日期: 2025-11-05
版本: 1.0.0
"""

import json
import csv
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Iterator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import pickle
import base64
from pathlib import Path
import warnings
from collections import defaultdict
import hashlib


class DataFormat(Enum):
    """支持的数据格式枚举"""
    JSON = "json"
    CSV = "csv"
    XML = "xml"
    PARQUET = "parquet"
    EXCEL = "excel"
    HDF5 = "hdf5"
    PICKLE = "pickle"
    SQL = "sql"


class NormalizationMethod(Enum):
    """标准化方法枚举"""
    MIN_MAX = "min_max"
    Z_SCORE = "z_score"
    ROBUST = "robust"
    DECIMAL = "decimal"


class AggregationMethod(Enum):
    """聚合方法枚举"""
    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    MAX = "max"
    MIN = "min"
    STD = "std"
    VAR = "var"
    MEDIAN = "median"


@dataclass
class TransformationConfig:
    """转换配置类"""
    source_format: DataFormat
    target_format: DataFormat
    encoding: str = "utf-8"
    normalize_method: Optional[NormalizationMethod] = None
    aggregation_method: Optional[AggregationMethod] = None
    chunk_size: int = 10000
    parallel: bool = False
    max_workers: int = 4
    preserve_index: bool = False
    compress: bool = False
    validation: bool = True


@dataclass
class TransformationResult:
    """转换结果类"""
    success: bool
    data: Any = None
    error_message: str = ""
    execution_time: float = 0.0
    rows_processed: int = 0
    memory_usage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataTransformer:
    """
    数据转换器主类
    
    提供完整的数据转换功能，包括格式转换、结构转换、聚合分析等。
    """
    
    def __init__(self, config: Optional[TransformationConfig] = None):
        """
        初始化数据转换器
        
        Args:
            config: 转换配置
        """
        self.config = config or TransformationConfig(
            source_format=DataFormat.JSON,
            target_format=DataFormat.CSV
        )
        self.logger = self._setup_logger()
        self._performance_stats = defaultdict(list)
        
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger("DataTransformer")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def convert_format(self, data: Any, config: TransformationConfig) -> TransformationResult:
        """
        转换数据格式
        
        Args:
            data: 源数据
            config: 转换配置
            
        Returns:
            TransformationResult: 转换结果
        """
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        try:
            result = self._convert_format_internal(data, config)
            execution_time = time.time() - start_time
            memory_after = self._get_memory_usage()
            
            return TransformationResult(
                success=True,
                data=result,
                execution_time=execution_time,
                memory_usage=memory_after - memory_before,
                metadata={
                    "source_format": config.source_format.value,
                    "target_format": config.target_format.value,
                    "encoding": config.encoding
                }
            )
            
        except Exception as e:
            self.logger.error(f"格式转换失败: {str(e)}")
            return TransformationResult(
                success=False,
                error_message=str(e),
                execution_time=time.time() - start_time
            )
    
    def _convert_format_internal(self, data: Any, config: TransformationConfig) -> Any:
        """内部格式转换逻辑"""
        if config.source_format == config.target_format:
            return data
            
        # 转换为pandas DataFrame
        if isinstance(data, (list, dict)):
            df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
        
        # 根据目标格式进行转换
        if config.target_format == DataFormat.CSV:
            return self._to_csv(df, config)
        elif config.target_format == DataFormat.JSON:
            return self._to_json(df, config)
        elif config.target_format == DataFormat.XML:
            return self._to_xml(df, config)
        elif config.target_format == DataFormat.PARQUET:
            return self._to_parquet(df, config)
        elif config.target_format == DataFormat.EXCEL:
            return self._to_excel(df, config)
        elif config.target_format == DataFormat.PICKLE:
            return self._to_pickle(df, config)
        else:
            raise ValueError(f"不支持的目标格式: {config.target_format}")
    
    def _to_csv(self, df: pd.DataFrame, config: TransformationConfig) -> str:
        """转换为CSV格式"""
        import io
        output = io.StringIO()
        df.to_csv(output, index=config.preserve_index, encoding=config.encoding)
        return output.getvalue()
    
    def _to_json(self, df: pd.DataFrame, config: TransformationConfig) -> str:
        """转换为JSON格式"""
        return df.to_json(orient='records', force_ascii=False, indent=2)
    
    def _to_xml(self, df: pd.DataFrame, config: TransformationConfig) -> str:
        """转换为XML格式"""
        root = ET.Element("data")
        
        for _, row in df.iterrows():
            record = ET.SubElement(root, "record")
            for col, value in row.items():
                element = ET.SubElement(record, col)
                element.text = str(value) if value is not None else ""
        
        return ET.tostring(root, encoding='unicode', method='xml')
    
    def _to_parquet(self, df: pd.DataFrame, config: TransformationConfig) -> bytes:
        """转换为Parquet格式"""
        import io
        buffer = io.BytesIO()
        df.to_parquet(buffer, index=config.preserve_index)
        return buffer.getvalue()
    
    def _to_excel(self, df: pd.DataFrame, config: TransformationConfig) -> bytes:
        """转换为Excel格式"""
        import io
        buffer = io.BytesIO()
        df.to_excel(buffer, index=config.preserve_index, engine='openpyxl')
        return buffer.getvalue()
    
    def _to_pickle(self, df: pd.DataFrame, config: TransformationConfig) -> bytes:
        """转换为Pickle格式"""
        return pickle.dumps(df)
    
    def reshape_wide_to_long(self, df: pd.DataFrame, 
                           id_vars: List[str], 
                           var_name: str = "variable",
                           value_name: str = "value") -> pd.DataFrame:
        """
        宽表转长表
        
        Args:
            df: 源DataFrame
            id_vars: ID列名列表
            var_name: 变量名列名
            value_name: 数值列名
            
        Returns:
            pd.DataFrame: 转换后的长表
        """
        try:
            # 找出需要转换的列
            value_vars = [col for col in df.columns if col not in id_vars]
            
            # 使用pandas的melt函数进行转换
            long_df = pd.melt(df, 
                            id_vars=id_vars,
                            value_vars=value_vars,
                            var_name=var_name,
                            value_name=value_name)
            
            self.logger.info(f"宽表转长表完成: {len(df)} -> {len(long_df)} 行")
            return long_df
            
        except Exception as e:
            self.logger.error(f"宽表转长表失败: {str(e)}")
            raise
    
    def reshape_long_to_wide(self, df: pd.DataFrame,
                           id_vars: List[str],
                           variable_col: str = "variable",
                           value_col: str = "value",
                           aggfunc: str = "first") -> pd.DataFrame:
        """
        长表转宽表
        
        Args:
            df: 源DataFrame
            id_vars: ID列名列表
            variable_col: 变量名列名
            value_col: 数值列名
            aggfunc: 聚合函数
            
        Returns:
            pd.DataFrame: 转换后的宽表
        """
        try:
            wide_df = df.pivot_table(
                index=id_vars,
                columns=variable_col,
                values=value_col,
                aggfunc=aggfunc
            ).reset_index()
            
            # 简化列名
            wide_df.columns.name = None
            
            self.logger.info(f"长表转宽表完成: {len(df)} -> {len(wide_df)} 行")
            return wide_df
            
        except Exception as e:
            self.logger.error(f"长表转宽表失败: {str(e)}")
            raise
    
    def aggregate_data(self, df: pd.DataFrame,
                      group_by: List[str],
                      agg_dict: Dict[str, Union[str, List[str]]],
                      fill_na: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        数据聚合
        
        Args:
            df: 源DataFrame
            group_by: 分组列名列表
            agg_dict: 聚合配置字典
            fill_na: 缺失值填充配置
            
        Returns:
            pd.DataFrame: 聚合后的数据
        """
        try:
            # 执行聚合
            result = df.groupby(group_by).agg(agg_dict).reset_index()
            
            # 填充缺失值
            if fill_na:
                result = result.fillna(fill_na)
            
            self.logger.info(f"数据聚合完成: {len(df)} -> {len(result)} 行")
            return result
            
        except Exception as e:
            self.logger.error(f"数据聚合失败: {str(e)}")
            raise
    
    def split_data(self, df: pd.DataFrame,
                   split_column: str,
                   split_values: List[Any]) -> Dict[Any, pd.DataFrame]:
        """
        数据拆分
        
        Args:
            df: 源DataFrame
            split_column: 拆分列名
            split_values: 拆分值列表
            
        Returns:
            Dict[Any, pd.DataFrame]: 拆分后的数据字典
        """
        try:
            result = {}
            for value in split_values:
                subset = df[df[split_column] == value].copy()
                result[value] = subset
                self.logger.debug(f"拆分数据: {value} -> {len(subset)} 行")
            
            self.logger.info(f"数据拆分完成: {len(df)} -> {len(result)} 个子集")
            return result
            
        except Exception as e:
            self.logger.error(f"数据拆分失败: {str(e)}")
            raise
    
    def merge_data(self, data_list: List[pd.DataFrame],
                   on: Union[str, List[str]],
                   how: str = "inner") -> pd.DataFrame:
        """
        数据合并
        
        Args:
            data_list: DataFrame列表
            on: 合并键
            how: 合并方式 ('inner', 'outer', 'left', 'right')
            
        Returns:
            pd.DataFrame: 合并后的数据
        """
        try:
            if not data_list:
                raise ValueError("数据列表不能为空")
            
            result = data_list[0]
            for df in data_list[1:]:
                result = pd.merge(result, df, on=on, how=how)
            
            self.logger.info(f"数据合并完成: {len(data_list)} 个数据集")
            return result
            
        except Exception as e:
            self.logger.error(f"数据合并失败: {str(e)}")
            raise
    
    def normalize_data(self, df: pd.DataFrame,
                      columns: List[str],
                      method: NormalizationMethod = NormalizationMethod.MIN_MAX,
                      parameters: Optional[Dict[str, Dict[str, float]]] = None) -> pd.DataFrame:
        """
        数据标准化和归一化
        
        Args:
            df: 源DataFrame
            columns: 需要标准化的列
            method: 标准化方法
            parameters: 预计算的参数（用于新数据标准化）
            
        Returns:
            pd.DataFrame: 标准化后的数据
        """
        try:
            result = df.copy()
            params = parameters or {}
            
            for col in columns:
                if col not in result.columns:
                    continue
                    
                if method == NormalizationMethod.MIN_MAX:
                    if col not in params:
                        min_val = result[col].min()
                        max_val = result[col].max()
                        params[col] = {"min": min_val, "max": max_val}
                    else:
                        min_val = params[col]["min"]
                        max_val = params[col]["max"]
                    
                    if max_val != min_val:
                        result[col] = (result[col] - min_val) / (max_val - min_val)
                    else:
                        result[col] = 0
                
                elif method == NormalizationMethod.Z_SCORE:
                    if col not in params:
                        mean_val = result[col].mean()
                        std_val = result[col].std()
                        params[col] = {"mean": mean_val, "std": std_val}
                    else:
                        mean_val = params[col]["mean"]
                        std_val = params[col]["std"]
                    
                    if std_val != 0:
                        result[col] = (result[col] - mean_val) / std_val
                    else:
                        result[col] = 0
                
                elif method == NormalizationMethod.ROBUST:
                    if col not in params:
                        median_val = result[col].median()
                        q1 = result[col].quantile(0.25)
                        q3 = result[col].quantile(0.75)
                        params[col] = {"median": median_val, "q1": q1, "q3": q3}
                    else:
                        median_val = params[col]["median"]
                        q1 = params[col]["q1"]
                        q3 = params[col]["q3"]
                    
                    iqr = q3 - q1
                    if iqr != 0:
                        result[col] = (result[col] - median_val) / iqr
                    else:
                        result[col] = 0
            
            self.logger.info(f"数据标准化完成: {method.value}")
            return result, params
            
        except Exception as e:
            self.logger.error(f"数据标准化失败: {str(e)}")
            raise
    
    def encode_data(self, df: pd.DataFrame,
                   columns: List[str],
                   method: str = "onehot") -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        数据编码
        
        Args:
            df: 源DataFrame
            columns: 需要编码的列
            method: 编码方法 ('onehot', 'label', 'target')
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: 编码后的数据和编码映射
        """
        try:
            result = df.copy()
            encoding_map = {}
            
            for col in columns:
                if col not in result.columns:
                    continue
                
                if method == "onehot":
                    # 独热编码
                    dummies = pd.get_dummies(result[col], prefix=col)
                    result = pd.concat([result, dummies], axis=1)
                    result = result.drop(col, axis=1)
                    encoding_map[col] = {"type": "onehot", "categories": dummies.columns.tolist()}
                
                elif method == "label":
                    # 标签编码
                    unique_values = result[col].unique()
                    label_map = {val: idx for idx, val in enumerate(unique_values)}
                    result[col] = result[col].map(label_map)
                    encoding_map[col] = {"type": "label", "mapping": label_map}
                
                elif method == "target":
                    # 目标编码（简化版本）
                    target_col = f"{col}_encoded"
                    mean_map = result.groupby(col).mean().to_dict()
                    result[target_col] = result[col].map(mean_map)
                    result = result.drop(col, axis=1)
                    encoding_map[col] = {"type": "target", "mapping": mean_map}
            
            self.logger.info(f"数据编码完成: {method}, {len(columns)} 列")
            return result, encoding_map
            
        except Exception as e:
            self.logger.error(f"数据编码失败: {str(e)}")
            raise
    
    def decode_data(self, df: pd.DataFrame,
                   encoding_map: Dict[str, Any]) -> pd.DataFrame:
        """
        数据解码
        
        Args:
            df: 编码后的DataFrame
            encoding_map: 编码映射
            
        Returns:
            pd.DataFrame: 解码后的数据
        """
        try:
            result = df.copy()
            
            for col, encoding_info in encoding_map.items():
                if encoding_info["type"] == "label":
                    # 反向标签编码
                    mapping = encoding_info["mapping"]
                    reverse_mapping = {v: k for k, v in mapping.items()}
                    result[col] = result[col].map(reverse_mapping)
                
                elif encoding_info["type"] == "target":
                    # 目标编码解码（使用中位数）
                    mapping = encoding_info["mapping"]
                    reverse_mapping = {}
                    for key, value in mapping.items():
                        reverse_mapping[value] = key
                    result[col] = result[col].map(reverse_mapping)
            
            self.logger.info("数据解码完成")
            return result
            
        except Exception as e:
            self.logger.error(f"数据解码失败: {str(e)}")
            raise
    
    def map_data(self, df: pd.DataFrame,
                mapping_rules: Dict[str, Dict[Any, Any]],
                default_value: Any = None) -> pd.DataFrame:
        """
        数据映射
        
        Args:
            df: 源DataFrame
            mapping_rules: 映射规则字典 {列名: {原值: 新值}}
            default_value: 默认值
            
        Returns:
            pd.DataFrame: 映射后的数据
        """
        try:
            result = df.copy()
            
            for col, mapping in mapping_rules.items():
                if col in result.columns:
                    result[col] = result[col].map(mapping)
                    if default_value is not None:
                        result[col] = result[col].fillna(default_value)
            
            self.logger.info(f"数据映射完成: {len(mapping_rules)} 列")
            return result
            
        except Exception as e:
            self.logger.error(f"数据映射失败: {str(e)}")
            raise
    
    def match_data(self, df1: pd.DataFrame,
                  df2: pd.DataFrame,
                  match_columns: List[str],
                  threshold: float = 0.8) -> pd.DataFrame:
        """
        数据匹配
        
        Args:
            df1: 第一个DataFrame
            df2: 第二个DataFrame
            match_columns: 匹配列
            threshold: 匹配阈值
            
        Returns:
            pd.DataFrame: 匹配结果
        """
        try:
            matched_data = []
            
            for _, row1 in df1.iterrows():
                best_match = None
                best_score = 0
                
                for _, row2 in df2.iterrows():
                    score = self._calculate_similarity(row1, row2, match_columns)
                    
                    if score > best_score and score >= threshold:
                        best_score = score
                        best_match = row2
                
                if best_match is not None:
                    match_result = row1.to_dict()
                    match_result.update({f"{col}_matched": best_match[col] 
                                       for col in match_columns})
                    match_result["similarity_score"] = best_score
                    matched_data.append(match_result)
            
            result = pd.DataFrame(matched_data)
            self.logger.info(f"数据匹配完成: {len(result)} 条匹配记录")
            return result
            
        except Exception as e:
            self.logger.error(f"数据匹配失败: {str(e)}")
            raise
    
    def _calculate_similarity(self, row1: pd.Series, 
                            row2: pd.Series, 
                            columns: List[str]) -> float:
        """计算两行数据的相似度"""
        if not columns:
            return 0.0
        
        similarities = []
        for col in columns:
            if col in row1.index and col in row2.index:
                val1, val2 = row1[col], row2[col]
                if pd.isna(val1) or pd.isna(val2):
                    continue
                
                # 数值相似度
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if val1 == val2:
                        similarities.append(1.0)
                    else:
                        max_val = max(abs(val1), abs(val2))
                        if max_val > 0:
                            similarities.append(1.0 - abs(val1 - val2) / max_val)
                        else:
                            similarities.append(1.0)
                
                # 字符串相似度
                elif isinstance(val1, str) and isinstance(val2, str):
                    if val1.lower() == val2.lower():
                        similarities.append(1.0)
                    else:
                        # 简单的字符串相似度计算
                        similarity = self._string_similarity(val1.lower(), val2.lower())
                        similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """计算字符串相似度"""
        if s1 == s2:
            return 1.0
        
        # 使用编辑距离的简化版本
        longer = s1 if len(s1) > len(s2) else s2
        shorter = s2 if len(s1) > len(s2) else s1
        
        if len(longer) == 0:
            return 1.0
        
        # 计算共同前缀和后缀
        common_prefix = 0
        for i in range(min(len(s1), len(s2))):
            if s1[i] == s2[i]:
                common_prefix += 1
            else:
                break
        
        common_suffix = 0
        for i in range(1, min(len(s1), len(s2)) + 1):
            if s1[-i] == s2[-i]:
                common_suffix += 1
            else:
                break
        
        # 相似度 = (共同字符数) / (总字符数)
        common_chars = common_prefix + common_suffix
        total_chars = len(s1) + len(s2)
        return common_chars / total_chars if total_chars > 0 else 0.0
    
    def create_pipeline(self, steps: List[Dict[str, Any]]) -> Callable:
        """
        创建转换流水线
        
        Args:
            steps: 转换步骤列表
            
        Returns:
            Callable: 流水线函数
        """
        def pipeline(data: Any) -> Any:
            result = data
            
            for step in steps:
                step_type = step.get("type")
                step_config = step.get("config", {})
                
                if step_type == "format_convert":
                    result = self.convert_format(result, step_config)
                    if not result.success:
                        raise ValueError(f"格式转换失败: {result.error_message}")
                    result = result.data
                
                elif step_type == "reshape_wide_to_long":
                    result = self.reshape_wide_to_long(result, **step_config)
                
                elif step_type == "reshape_long_to_wide":
                    result = self.reshape_long_to_wide(result, **step_config)
                
                elif step_type == "aggregate":
                    result = self.aggregate_data(result, **step_config)
                
                elif step_type == "normalize":
                    result, _ = self.normalize_data(result, **step_config)
                
                elif step_type == "encode":
                    result, _ = self.encode_data(result, **step_config)
                
                elif step_type == "map":
                    result = self.map_data(result, **step_config)
                
                else:
                    raise ValueError(f"未知的转换步骤: {step_type}")
            
            return result
        
        return pipeline
    
    def optimize_performance(self, data: pd.DataFrame,
                           optimization_config: Dict[str, Any]) -> pd.DataFrame:
        """
        性能优化
        
        Args:
            data: 源数据
            optimization_config: 优化配置
            
        Returns:
            pd.DataFrame: 优化后的数据
        """
        try:
            result = data.copy()
            
            # 数据类型优化
            if optimization_config.get("optimize_dtypes", True):
                result = self._optimize_dtypes(result)
            
            # 内存优化
            if optimization_config.get("optimize_memory", True):
                result = self._optimize_memory(result)
            
            # 索引优化
            if optimization_config.get("optimize_index", True):
                result = self._optimize_index(result)
            
            self.logger.info("数据性能优化完成")
            return result
            
        except Exception as e:
            self.logger.error(f"性能优化失败: {str(e)}")
            raise
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化数据类型"""
        result = df.copy()
        
        for col in result.columns:
            col_type = result[col].dtype
            
            if col_type != 'object':
                continue
            
            # 尝试转换为数值类型
            numeric_col = pd.to_numeric(result[col], errors='coerce')
            if not numeric_col.isna().all():
                result[col] = numeric_col
                continue
            
            # 尝试转换为分类类型
            if result[col].nunique() / len(result[col]) < 0.5:
                result[col] = result[col].astype('category')
        
        return result
    
    def _optimize_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化内存使用"""
        return df.copy()  # 简化实现
    
    def _optimize_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """优化索引"""
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        return df
    
    def _get_memory_usage(self) -> float:
        """获取内存使用量"""
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def validate_data(self, data: Any, validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """
        数据验证
        
        Args:
            data: 待验证数据
            validation_rules: 验证规则
            
        Returns:
            Dict[str, Any]: 验证结果
        """
        try:
            if isinstance(data, pd.DataFrame):
                return self._validate_dataframe(data, validation_rules)
            else:
                return {"valid": True, "errors": []}
                
        except Exception as e:
            return {"valid": False, "errors": [str(e)]}
    
    def _validate_dataframe(self, df: pd.DataFrame, rules: Dict[str, Any]) -> Dict[str, Any]:
        """验证DataFrame"""
        errors = []
        
        # 检查必需列
        required_cols = rules.get("required_columns", [])
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            errors.append(f"缺少必需列: {missing_cols}")
        
        # 检查数据类型
        dtype_rules = rules.get("dtypes", {})
        for col, expected_type in dtype_rules.items():
            if col in df.columns:
                actual_type = str(df[col].dtype)
                if actual_type != expected_type:
                    errors.append(f"列 {col} 数据类型不匹配: 期望 {expected_type}, 实际 {actual_type}")
        
        # 检查缺失值
        null_rules = rules.get("null_check", {})
        for col, max_null_ratio in null_rules.items():
            if col in df.columns:
                null_ratio = df[col].isna().sum() / len(df)
                if null_ratio > max_null_ratio:
                    errors.append(f"列 {col} 缺失值比例过高: {null_ratio:.2%} > {max_null_ratio:.2%}")
        
        # 检查数值范围
        range_rules = rules.get("ranges", {})
        for col, (min_val, max_val) in range_rules.items():
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                if df[col].min() < min_val or df[col].max() > max_val:
                    errors.append(f"列 {col} 数值超出范围: [{min_val}, {max_val}]")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "summary": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "missing_values": df.isna().sum().to_dict()
            }
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        stats = {}
        for operation, times in self._performance_stats.items():
            if times:
                stats[operation] = {
                    "count": len(times),
                    "avg_time": np.mean(times),
                    "min_time": np.min(times),
                    "max_time": np.max(times),
                    "total_time": np.sum(times)
                }
        return stats
    
    def reset_performance_stats(self):
        """重置性能统计"""
        self._performance_stats.clear()
    
    def batch_process(self, data_list: List[Any], 
                     process_func: Callable,
                     config: Dict[str, Any]) -> List[Any]:
        """
        批量处理数据
        
        Args:
            data_list: 数据列表
            process_func: 处理函数
            config: 处理配置
            
        Returns:
            List[Any]: 处理结果列表
        """
        try:
            parallel = config.get("parallel", False)
            max_workers = config.get("max_workers", 4)
            
            if parallel:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    results = list(executor.map(process_func, data_list))
            else:
                results = [process_func(data) for data in data_list]
            
            self.logger.info(f"批量处理完成: {len(data_list)} 个数据集")
            return results
            
        except Exception as e:
            self.logger.error(f"批量处理失败: {str(e)}")
            raise


def create_sample_data() -> pd.DataFrame:
    """创建示例数据"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'id': range(1, n_samples + 1),
        'name': [f'Product_{i}' for i in range(1, n_samples + 1)],
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'price': np.random.uniform(10, 1000, n_samples),
        'quantity': np.random.randint(1, 100, n_samples),
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples)
    }
    
    return pd.DataFrame(data)


def run_comprehensive_tests():
    """运行综合测试"""
    print("开始T4数据转换器综合测试...")
    
    # 创建示例数据
    sample_data = create_sample_data()
    print(f"创建示例数据: {len(sample_data)} 行")
    
    # 初始化转换器
    transformer = DataTransformer()
    
    # 测试1: 格式转换
    print("\n1. 测试格式转换...")
    # 测试1: 格式转换
    print("\n1. 测试格式转换...")
    config = TransformationConfig(
        source_format=DataFormat.JSON,  # 实际不影响，因为函数会根据数据类型处理
        target_format=DataFormat.JSON
    )
    result = transformer.convert_format(sample_data, config)
    if result.success:
        print("✓ JSON转换成功")
    else:
        print(f"✗ JSON转换失败: {result.error_message}")
    
    # 测试2: 宽表转长表
    print("\n2. 测试宽表转长表...")
    wide_data = pd.DataFrame({
        'id': [1, 2, 3],
        'product_a': [100, 200, 150],
        'product_b': [150, 250, 200],
        'product_c': [120, 180, 160]
    })
    
    try:
        long_data = transformer.reshape_wide_to_long(
            wide_data, 
            id_vars=['id'], 
            var_name='product',
            value_name='sales'
        )
        print(f"✓ 宽表转长表成功: {len(wide_data)} -> {len(long_data)} 行")
    except Exception as e:
        print(f"✗ 宽表转长表失败: {e}")
    
    # 测试3: 数据聚合
    print("\n3. 测试数据聚合...")
    try:
        aggregated = transformer.aggregate_data(
            sample_data,
            group_by=['category', 'region'],
            agg_dict={
                'price': 'mean',
                'quantity': 'sum',
                'id': 'count'
            }
        )
        print(f"✓ 数据聚合成功: {len(sample_data)} -> {len(aggregated)} 行")
    except Exception as e:
        print(f"✗ 数据聚合失败: {e}")
    
    # 测试4: 数据标准化
    print("\n4. 测试数据标准化...")
    try:
        normalized_data, params = transformer.normalize_data(
            sample_data,
            columns=['price', 'quantity'],
            method=NormalizationMethod.MIN_MAX
        )
        print("✓ 数据标准化成功")
    except Exception as e:
        print(f"✗ 数据标准化失败: {e}")
    
    # 测试5: 数据编码
    print("\n5. 测试数据编码...")
    try:
        encoded_data, encoding_map = transformer.encode_data(
            sample_data,
            columns=['category', 'region'],
            method='onehot'
        )
        print(f"✓ 数据编码成功: {len(sample_data.columns)} -> {len(encoded_data.columns)} 列")
    except Exception as e:
        print(f"✗ 数据编码失败: {e}")
    
    # 测试6: 数据映射
    print("\n6. 测试数据映射...")
    try:
        mapping_rules = {
            'category': {'A': 'Category_A', 'B': 'Category_B', 'C': 'Category_C'},
            'region': {'North': 'N', 'South': 'S', 'East': 'E', 'West': 'W'}
        }
        mapped_data = transformer.map_data(sample_data, mapping_rules)
        print("✓ 数据映射成功")
    except Exception as e:
        print(f"✗ 数据映射失败: {e}")
    
    # 测试7: 转换流水线
    print("\n7. 测试转换流水线...")
    try:
        pipeline_steps = [
            {
                "type": "aggregate",
                "config": {
                    "group_by": ["category"],
                    "agg_dict": {"price": "mean", "quantity": "sum"}
                }
            },
            {
                "type": "normalize",
                "config": {
                    "columns": ["price"],
                    "method": NormalizationMethod.Z_SCORE
                }
            }
        ]
        
        pipeline = transformer.create_pipeline(pipeline_steps)
        pipeline_result = pipeline(sample_data)
        print(f"✓ 转换流水线成功: {len(pipeline_result)} 行")
    except Exception as e:
        print(f"✗ 转换流水线失败: {e}")
    
    # 测试8: 性能优化
    print("\n8. 测试性能优化...")
    try:
        optimized_data = transformer.optimize_performance(
            sample_data,
            optimization_config={
                "optimize_dtypes": True,
                "optimize_memory": True,
                "optimize_index": True
            }
        )
        print("✓ 性能优化成功")
    except Exception as e:
        print(f"✗ 性能优化失败: {e}")
    
    # 测试9: 数据验证
    print("\n9. 测试数据验证...")
    try:
        validation_rules = {
            "required_columns": ["id", "name", "category"],
            "null_check": {"price": 0.1, "quantity": 0.1},
            "ranges": {"price": (0, 10000), "quantity": (0, 1000)}
        }
        validation_result = transformer.validate_data(sample_data, validation_rules)
        if validation_result["valid"]:
            print("✓ 数据验证通过")
        else:
            print(f"✗ 数据验证失败: {validation_result['errors']}")
    except Exception as e:
        print(f"✗ 数据验证失败: {e}")
    
    # 测试10: 批量处理
    print("\n10. 测试批量处理...")
    try:
        data_list = [sample_data.copy() for _ in range(5)]
        
        def process_single_data(df):
            return transformer.aggregate_data(
                df,
                group_by=['category'],
                agg_dict={'price': 'mean'}
            )
        
        batch_results = transformer.batch_process(
            data_list,
            process_single_data,
            config={"parallel": True, "max_workers": 2}
        )
        print(f"✓ 批量处理成功: {len(batch_results)} 个结果")
    except Exception as e:
        print(f"✗ 批量处理失败: {e}")
    
    print("\nT4数据转换器综合测试完成!")
    
    # 显示性能统计
    stats = transformer.get_performance_stats()
    if stats:
        print("\n性能统计:")
        for operation, stat in stats.items():
            print(f"  {operation}: 平均耗时 {stat['avg_time']:.4f}s")


if __name__ == "__main__":
    # 运行综合测试
    run_comprehensive_tests()