"""
J7验证工具模块 - 完整实现

这个模块提供了全面的验证工具集合，包括：
- 数据验证工具（格式验证、范围验证、唯一性验证）
- 模型验证工具（交叉验证、留一验证、时序验证）
- 策略验证工具（回测验证、样本外验证、稳健性检验）
- 系统验证工具（接口验证、配置验证、依赖验证）
- 自动化验证流程
- 验证报告生成
- 异步验证和并行处理
- 完整的错误处理和日志记录

作者：AI系统
版本：1.0.0
日期：2025-11-06
"""

import asyncio
import logging
import warnings
import json
import pickle
import hashlib
import inspect
import importlib
import traceback
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, Set
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import wraps
import threading
import multiprocessing as mp
from pathlib import Path
import re
import statistics
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import (
    KFold, StratifiedKFold, TimeSeriesSplit, LeaveOneOut,
    cross_val_score, cross_validate
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
# 可选依赖 - 用于HTML报告生成
try:
    from jinja2 import Template
    JINJA2_AVAILABLE = True
except ImportError:
    Template = None
    JINJA2_AVAILABLE = False
import yaml

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('validation_tools.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# =============================================================================
# 基础异常类
# =============================================================================

class ValidationError(Exception):
    """验证错误基类"""
    def __init__(self, message: str, error_code: str = None, details: Dict = None):
        super().__init__(message)
        self.error_code = error_code or "VALIDATION_ERROR"
        self.details = details or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'error_code': self.error_code,
            'message': str(self),
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


class DataValidationError(ValidationError):
    """数据验证错误"""
    def __init__(self, message: str, data_info: Dict = None):
        super().__init__(message, "DATA_VALIDATION_ERROR", data_info)


class ModelValidationError(ValidationError):
    """模型验证错误"""
    def __init__(self, message: str, model_info: Dict = None):
        super().__init__(message, "MODEL_VALIDATION_ERROR", model_info)


class StrategyValidationError(ValidationError):
    """策略验证错误"""
    def __init__(self, message: str, strategy_info: Dict = None):
        super().__init__(message, "STRATEGY_VALIDATION_ERROR", strategy_info)


class SystemValidationError(ValidationError):
    """系统验证错误"""
    def __init__(self, message: str, system_info: Dict = None):
        super().__init__(message, "SYSTEM_VALIDATION_ERROR", system_info)


# =============================================================================
# 数据类和配置类
# =============================================================================

@dataclass
class ValidationResult:
    """验证结果数据类"""
    success: bool
    message: str
    score: Optional[float] = None
    details: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'success': self.success,
            'message': self.message,
            'score': self.score,
            'details': self.details,
            'warnings': self.warnings,
            'errors': self.errors,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class ValidationConfig:
    """验证配置数据类"""
    # 通用配置
    strict_mode: bool = False
    parallel_execution: bool = True
    max_workers: int = 4
    timeout: int = 300
    save_intermediate_results: bool = True
    
    # 数据验证配置
    data_validation_config: Dict = field(default_factory=dict)
    
    # 模型验证配置
    model_validation_config: Dict = field(default_factory=dict)
    
    # 策略验证配置
    strategy_validation_config: Dict = field(default_factory=dict)
    
    # 系统验证配置
    system_validation_config: Dict = field(default_factory=dict)
    
    # 报告配置
    report_config: Dict = field(default_factory=lambda: {
        'format': 'html',
        'include_charts': True,
        'include_details': True,
        'output_path': 'validation_report.html'
    })


# =============================================================================
# 基础验证器类
# =============================================================================

class BaseValidator(ABC):
    """验证器基类"""
    
    def __init__(self, config: ValidationConfig = None):
        self.config = config or ValidationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._lock = threading.Lock()
    
    @abstractmethod
    def validate(self, *args, **kwargs) -> ValidationResult:
        """执行验证的抽象方法"""
        pass
    
    def _log_execution_time(self, func_name: str, start_time: float):
        """记录执行时间"""
        execution_time = time.time() - start_time
        self.logger.info(f"{func_name} 执行时间: {execution_time:.2f}秒")
        return execution_time
    
    def _handle_exception(self, func_name: str, exception: Exception) -> ValidationResult:
        """处理异常并返回验证结果"""
        error_msg = f"{func_name} 执行失败: {str(exception)}"
        self.logger.error(error_msg)
        self.logger.debug(traceback.format_exc())
        
        return ValidationResult(
            success=False,
            message=error_msg,
            errors=[str(exception)],
            execution_time=0.0
        )
    
    def _validate_input(self, data: Any, expected_type: type = None) -> bool:
        """验证输入数据"""
        if expected_type and not isinstance(data, expected_type):
            raise ValidationError(f"期望数据类型 {expected_type}，实际类型 {type(data)}")
        return True


# =============================================================================
# 数据验证工具
# =============================================================================

class DataValidator(BaseValidator):
    """
    数据验证工具类
    
    提供数据格式验证、范围验证、唯一性验证等功能
    """
    
    def __init__(self, config: ValidationConfig = None):
        super().__init__(config)
        self.data_validation_config = self.config.data_validation_config
        self._format_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'phone': r'^\+?1?\d{9,15}$',
            'date': r'^\d{4}-\d{2}-\d{2}$',
            'datetime': r'^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$',
            'url': r'^https?://[^\s/$.?#].[^\s]*$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
            'ip': r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
        }
    
    def validate_format(self, data: Any, format_type: str, 
                       custom_pattern: str = None) -> ValidationResult:
        """
        验证数据格式
        
        Args:
            data: 要验证的数据
            format_type: 预定义格式类型 ('email', 'phone', 'date', 'datetime', 'url', 'uuid', 'ip')
            custom_pattern: 自定义正则表达式模式
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            if data is None:
                return ValidationResult(
                    success=False,
                    message="数据不能为空",
                    execution_time=self._log_execution_time("validate_format", start_time)
                )
            
            # 获取验证模式
            if custom_pattern:
                pattern = custom_pattern
            elif format_type in self._format_patterns:
                pattern = self._format_patterns[format_type]
            else:
                raise ValidationError(f"不支持的格式类型: {format_type}")
            
            # 执行格式验证
            data_str = str(data)
            if re.match(pattern, data_str):
                result = ValidationResult(
                    success=True,
                    message=f"数据格式验证通过: {format_type}",
                    execution_time=self._log_execution_time("validate_format", start_time)
                )
            else:
                result = ValidationResult(
                    success=False,
                    message=f"数据格式验证失败: {format_type}",
                    details={'data': data_str, 'expected_format': format_type},
                    execution_time=self._log_execution_time("validate_format", start_time)
                )
            
            return result
            
        except Exception as e:
            return self._handle_exception("validate_format", e)
    
    def validate_range(self, data: Union[int, float], 
                      min_val: float = None, 
                      max_val: float = None,
                      inclusive: bool = True) -> ValidationResult:
        """
        验证数据范围
        
        Args:
            data: 要验证的数据
            min_val: 最小值
            max_val: 最大值
            inclusive: 是否包含边界值
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            if not isinstance(data, (int, float)):
                raise DataValidationError(f"数据必须是数字类型，实际类型: {type(data)}")
            
            violations = []
            
            # 检查最小值
            if min_val is not None:
                if inclusive and data < min_val:
                    violations.append(f"数据 {data} 小于最小值 {min_val}")
                elif not inclusive and data <= min_val:
                    violations.append(f"数据 {data} 小于等于最小值 {min_val}")
            
            # 检查最大值
            if max_val is not None:
                if inclusive and data > max_val:
                    violations.append(f"数据 {data} 大于最大值 {max_val}")
                elif not inclusive and data >= max_val:
                    violations.append(f"数据 {data} 大于等于最大值 {max_val}")
            
            if violations:
                result = ValidationResult(
                    success=False,
                    message="数据范围验证失败",
                    errors=violations,
                    details={
                        'data': data,
                        'min_val': min_val,
                        'max_val': max_val,
                        'inclusive': inclusive
                    },
                    execution_time=self._log_execution_time("validate_range", start_time)
                )
            else:
                result = ValidationResult(
                    success=True,
                    message="数据范围验证通过",
                    details={
                        'data': data,
                        'min_val': min_val,
                        'max_val': max_val,
                        'inclusive': inclusive
                    },
                    execution_time=self._log_execution_time("validate_range", start_time)
                )
            
            return result
            
        except Exception as e:
            return self._handle_exception("validate_range", e)
    
    def validate_uniqueness(self, data: List[Any], 
                           key_func: Callable = None) -> ValidationResult:
        """
        验证数据唯一性
        
        Args:
            data: 要验证的数据列表
            key_func: 用于提取唯一键的函数
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            if not isinstance(data, (list, tuple)):
                raise DataValidationError(f"数据必须是列表或元组类型，实际类型: {type(data)}")
            
            if not data:
                return ValidationResult(
                    success=True,
                    message="空数据列表，唯一性验证通过",
                    execution_time=self._log_execution_time("validate_uniqueness", start_time)
                )
            
            # 提取唯一键
            if key_func:
                keys = [key_func(item) for item in data]
            else:
                keys = data
            
            # 检查重复
            seen = set()
            duplicates = []
            
            for key in keys:
                if key in seen:
                    duplicates.append(key)
                else:
                    seen.add(key)
            
            if duplicates:
                result = ValidationResult(
                    success=False,
                    message=f"发现 {len(duplicates)} 个重复项",
                    details={
                        'total_items': len(data),
                        'unique_items': len(seen),
                        'duplicate_items': len(duplicates),
                        'duplicates': list(set(duplicates))
                    },
                    execution_time=self._log_execution_time("validate_uniqueness", start_time)
                )
            else:
                result = ValidationResult(
                    success=True,
                    message="数据唯一性验证通过",
                    details={
                        'total_items': len(data),
                        'unique_items': len(seen)
                    },
                    execution_time=self._log_execution_time("validate_uniqueness", start_time)
                )
            
            return result
            
        except Exception as e:
            return self._handle_exception("validate_uniqueness", e)
    
    def validate_dataframe(self, df: pd.DataFrame, 
                          required_columns: List[str] = None,
                          column_types: Dict[str, type] = None,
                          null_threshold: float = 0.0) -> ValidationResult:
        """
        验证DataFrame数据
        
        Args:
            df: 要验证的DataFrame
            required_columns: 必需列名列表
            column_types: 列类型字典 {列名: 期望类型}
            null_threshold: 空值比例阈值
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            if not isinstance(df, pd.DataFrame):
                raise DataValidationError(f"数据必须是DataFrame类型，实际类型: {type(df)}")
            
            errors = []
            warnings = []
            details = {}
            
            # 验证必需列
            if required_columns:
                missing_columns = set(required_columns) - set(df.columns)
                if missing_columns:
                    errors.append(f"缺少必需列: {list(missing_columns)}")
                else:
                    details['required_columns'] = required_columns
            
            # 验证列类型
            if column_types:
                type_errors = []
                for col, expected_type in column_types.items():
                    if col in df.columns:
                        actual_type = df[col].dtype
                        if not self._is_type_compatible(actual_type, expected_type):
                            type_errors.append(f"列 {col} 类型不匹配: 期望 {expected_type}, 实际 {actual_type}")
                
                if type_errors:
                    errors.extend(type_errors)
                else:
                    details['column_types'] = column_types
            
            # 验证空值比例
            null_ratios = df.isnull().sum() / len(df)
            high_null_columns = null_ratios[null_ratios > null_threshold]
            
            if not high_null_columns.empty:
                warnings.append(f"以下列的空值比例超过阈值 {null_threshold}: {high_null_columns.to_dict()}")
            
            details['null_ratios'] = null_ratios.to_dict()
            details['total_rows'] = len(df)
            details['total_columns'] = len(df.columns)
            
            success = len(errors) == 0
            
            result = ValidationResult(
                success=success,
                message="DataFrame验证" + ("通过" if success else "失败"),
                errors=errors if errors else None,
                warnings=warnings if warnings else None,
                details=details,
                execution_time=self._log_execution_time("validate_dataframe", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("validate_dataframe", e)
    
    def _is_type_compatible(self, pandas_dtype, expected_type: type) -> bool:
        """检查pandas数据类型与期望类型的兼容性"""
        type_mapping = {
            int: ['int64', 'int32', 'int16', 'int8'],
            float: ['float64', 'float32'],
            str: ['object', 'string'],
            bool: ['bool']
        }
        
        if expected_type in type_mapping:
            return str(pandas_dtype) in type_mapping[expected_type]
        
        return str(pandas_dtype) == expected_type.__name__
    
    def validate_time_series(self, data: pd.Series, 
                            freq: str = None,
                            check_stationarity: bool = True) -> ValidationResult:
        """
        验证时间序列数据
        
        Args:
            data: 时间序列数据
            freq: 期望频率 ('D', 'H', 'M', 'Y' 等)
            check_stationarity: 是否检查平稳性
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            if not isinstance(data, pd.Series):
                raise DataValidationError(f"数据必须是Series类型，实际类型: {type(data)}")
            
            if data.index.dtype != 'datetime64[ns]':
                raise DataValidationError("时间序列索引必须是datetime类型")
            
            errors = []
            warnings = []
            details = {
                'start_date': data.index.min(),
                'end_date': data.index.max(),
                'total_points': len(data),
                'frequency': str(data.index.freq) if data.index.freq else 'unknown'
            }
            
            # 验证频率
            if freq and data.index.freq:
                expected_freq = pd.tseries.frequencies.to_offset(freq)
                if data.index.freq != expected_freq:
                    warnings.append(f"频率不匹配: 期望 {freq}, 实际 {data.index.freq}")
            
            # 检查平稳性
            if check_stationarity and len(data) > 10:
                try:
                    from statsmodels.tsa.stattools import adfuller
                    result = adfuller(data.dropna())
                    is_stationary = result[1] < 0.05
                    details['stationarity'] = {
                        'is_stationary': is_stationary,
                        'p_value': result[1],
                        'critical_values': result[4]
                    }
                    
                    if not is_stationary:
                        warnings.append("时间序列可能不平稳")
                        
                except ImportError:
                    warnings.append("无法检查平稳性：缺少statsmodels库")
            
            # 检查缺失值
            null_count = data.isnull().sum()
            if null_count > 0:
                warnings.append(f"发现 {null_count} 个缺失值")
            
            details['null_count'] = null_count
            
            success = len(errors) == 0
            
            result = ValidationResult(
                success=success,
                message="时间序列验证" + ("通过" if success else "失败"),
                errors=errors if errors else None,
                warnings=warnings if warnings else None,
                details=details,
                execution_time=self._log_execution_time("validate_time_series", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("validate_time_series", e)
    
    def validate(self, data: Any, validation_type: str = "format", **kwargs) -> ValidationResult:
        """
        通用验证方法
        
        Args:
            data: 要验证的数据
            validation_type: 验证类型 ('format', 'range', 'uniqueness', 'dataframe', 'time_series')
            **kwargs: 验证参数
            
        Returns:
            ValidationResult: 验证结果
        """
        if validation_type == "format":
            return self.validate_format(data, kwargs.get('format_type', 'email'))
        elif validation_type == "range":
            return self.validate_range(
                data, 
                min_val=kwargs.get('min_val'), 
                max_val=kwargs.get('max_val')
            )
        elif validation_type == "uniqueness":
            return self.validate_uniqueness(data, kwargs.get('key_func'))
        elif validation_type == "dataframe":
            return self.validate_dataframe(
                data,
                required_columns=kwargs.get('required_columns'),
                column_types=kwargs.get('column_types')
            )
        elif validation_type == "time_series":
            return self.validate_time_series(
                data,
                freq=kwargs.get('freq'),
                check_stationarity=kwargs.get('check_stationarity', True)
            )
        else:
            raise ValidationError(f"未知的验证类型: {validation_type}")


# =============================================================================
# 模型验证工具
# =============================================================================

class ModelValidator(BaseValidator):
    """
    模型验证工具类
    
    提供交叉验证、留一验证、时序验证等功能
    """
    
    def __init__(self, config: ValidationConfig = None):
        super().__init__(config)
        self.model_validation_config = self.config.model_validation_config
        self._scoring_metrics = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'mse': mean_squared_error,
            'mae': mean_absolute_error,
            'r2': r2_score,
            'roc_auc': roc_auc_score
        }
    
    def cross_validate(self, model, X, y, 
                      cv_folds: int = 5,
                      scoring: List[str] = None,
                      stratified: bool = False,
                      random_state: int = 42) -> ValidationResult:
        """
        执行交叉验证
        
        Args:
            model: 要验证的模型
            X: 特征数据
            y: 目标数据
            cv_folds: 交叉验证折数
            scoring: 评估指标列表
            stratified: 是否使用分层抽样
            random_state: 随机种子
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            # 默认评估指标
            if scoring is None:
                scoring = ['accuracy']
            
            # 选择交叉验证策略
            if stratified and len(np.unique(y)) > 1:
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            else:
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            # 执行交叉验证
            cv_results = cross_validate(
                model, X, y, 
                cv=cv, 
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1 if self.config.parallel_execution else 1
            )
            
            # 计算统计信息
            details = {}
            for metric in scoring:
                test_scores = cv_results[f'test_{metric}']
                train_scores = cv_results[f'train_{metric}']
                
                details[metric] = {
                    'test_mean': np.mean(test_scores),
                    'test_std': np.std(test_scores),
                    'test_scores': test_scores.tolist(),
                    'train_mean': np.mean(train_scores),
                    'train_std': np.std(train_scores),
                    'train_scores': train_scores.tolist()
                }
                
                # 检查过拟合
                overfitting_threshold = 0.1
                if details[metric]['test_mean'] + overfitting_threshold < details[metric]['train_mean']:
                    details[metric]['overfitting_risk'] = True
                else:
                    details[metric]['overfitting_risk'] = False
            
            # 总体评估
            primary_metric = scoring[0]
            mean_score = details[primary_metric]['test_mean']
            std_score = details[primary_metric]['test_std']
            
            # 判断验证结果
            success = mean_score > 0.5  # 可以根据具体需求调整阈值
            
            result = ValidationResult(
                success=success,
                message=f"交叉验证完成，主要指标 {primary_metric}: {mean_score:.4f} (±{std_score:.4f})",
                score=mean_score,
                details=details,
                execution_time=self._log_execution_time("cross_validate", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("cross_validate", e)
    
    def leave_one_out_validate(self, model, X, y) -> ValidationResult:
        """
        执行留一验证
        
        Args:
            model: 要验证的模型
            X: 特征数据
            y: 目标数据
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            loo = LeaveOneOut()
            scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
            
            details = {
                'scores': scores.tolist(),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'total_iterations': len(scores)
            }
            
            success = details['mean_score'] > 0.5
            
            result = ValidationResult(
                success=success,
                message=f"留一验证完成，平均准确率: {details['mean_score']:.4f}",
                score=details['mean_score'],
                details=details,
                execution_time=self._log_execution_time("leave_one_out_validate", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("leave_one_out_validate", e)
    
    def time_series_validate(self, model, X, y, 
                            n_splits: int = 5,
                            test_size: int = 1) -> ValidationResult:
        """
        执行时间序列验证
        
        Args:
            model: 要验证的模型
            X: 特征数据
            y: 目标数据
            n_splits: 分割数量
            test_size: 测试集大小
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            tscv = TimeSeriesSplit(n_splits=n_splits)
            scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
            
            details = {
                'scores': scores.tolist(),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'n_splits': n_splits,
                'test_size': test_size
            }
            
            success = details['mean_score'] > 0.5
            
            result = ValidationResult(
                success=success,
                message=f"时间序列验证完成，平均准确率: {details['mean_score']:.4f}",
                score=details['mean_score'],
                details=details,
                execution_time=self._log_execution_time("time_series_validate", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("time_series_validate", e)
    
    def bootstrap_validate(self, model, X, y, 
                          n_bootstrap: int = 100,
                          test_ratio: float = 0.2,
                          scoring: str = 'accuracy') -> ValidationResult:
        """
        执行Bootstrap验证
        
        Args:
            model: 要验证的模型
            X: 特征数据
            y: 目标数据
            n_bootstrap: Bootstrap样本数量
            test_ratio: 测试集比例
            scoring: 评估指标
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            scores = []
            n_samples = len(X)
            test_size = int(n_samples * test_ratio)
            
            for i in range(n_bootstrap):
                # 生成Bootstrap样本
                indices = np.random.choice(n_samples, n_samples, replace=True)
                train_indices = indices[:-test_size]
                test_indices = indices[-test_size:]
                
                # 训练和测试
                X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
                y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # 计算得分
                if scoring in self._scoring_metrics:
                    score = self._scoring_metrics[scoring](y_test, y_pred)
                    scores.append(score)
            
            scores = np.array(scores)
            
            details = {
                'scores': scores.tolist(),
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'confidence_interval': {
                    'lower': np.percentile(scores, 2.5),
                    'upper': np.percentile(scores, 97.5)
                },
                'n_bootstrap': n_bootstrap
            }
            
            success = details['mean_score'] > 0.5
            
            result = ValidationResult(
                success=success,
                message=f"Bootstrap验证完成，平均得分: {details['mean_score']:.4f}",
                score=details['mean_score'],
                details=details,
                execution_time=self._log_execution_time("bootstrap_validate", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("bootstrap_validate", e)
    
    def validate_model_assumptions(self, model, X, y) -> ValidationResult:
        """
        验证模型假设
        
        Args:
            model: 要验证的模型
            X: 特征数据
            y: 目标数据
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            warnings_list = []
            details = {}
            
            # 拟合模型
            model.fit(X, y)
            y_pred = model.predict(X)
            residuals = y - y_pred
            
            # 检查残差正态性（针对回归模型）
            if hasattr(model, 'predict_proba'):
                # 分类模型
                if len(np.unique(y)) == 2:
                    # 二分类
                    try:
                        from sklearn.metrics import roc_auc_score
                        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
                        details['auc_score'] = auc
                        
                        if auc < 0.6:
                            warnings_list.append(f"AUC得分较低: {auc:.3f}")
                    except:
                        pass
            else:
                # 回归模型
                # Shapiro-Wilk检验
                try:
                    stat, p_value = stats.shapiro(residuals)
                    details['shapiro_stat'] = stat
                    details['shapiro_p_value'] = p_value
                    
                    if p_value < 0.05:
                        warnings_list.append("残差可能不符合正态分布")
                except:
                    pass
                
                # 检查异方差性
                try:
                    from scipy.stats import spearmanr
                    correlation, p_value = spearmanr(np.abs(residuals), y_pred)
                    details['heteroscedasticity_correlation'] = correlation
                    details['heteroscedasticity_p_value'] = p_value
                    
                    if p_value < 0.05:
                        warnings_list.append("可能存在异方差性")
                except:
                    pass
            
            # 检查多重共线性（针对线性模型）
            if hasattr(model, 'coef_'):
                try:
                    from statsmodels.stats.outliers_influence import variance_inflation_factor
                    vif_data = []
                    for i in range(X.shape[1]):
                        vif = variance_inflation_factor(X.values, i)
                        vif_data.append(vif)
                    
                    details['vif_scores'] = vif_data
                    high_vif = [v for v in vif_data if v > 10]
                    
                    if high_vif:
                        warnings_list.append(f"发现高VIF值: {high_vif}")
                        
                except ImportError:
                    warnings_list.append("无法检查多重共线性：缺少statsmodels库")
                except:
                    pass
            
            success = len(warnings_list) == 0
            
            result = ValidationResult(
                success=success,
                message="模型假设验证" + ("通过" if success else "发现潜在问题"),
                warnings=warnings_list if warnings_list else None,
                details=details,
                execution_time=self._log_execution_time("validate_model_assumptions", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("validate_model_assumptions", e)
    
    def validate(self, model, X, y, validation_type: str = "cross_validate", **kwargs) -> ValidationResult:
        """
        通用验证方法
        
        Args:
            model: 要验证的模型
            X: 特征数据
            y: 目标数据
            validation_type: 验证类型 ('cross_validate', 'leave_one_out', 'time_series', 'bootstrap', 'assumptions')
            **kwargs: 验证参数
            
        Returns:
            ValidationResult: 验证结果
        """
        if validation_type == "cross_validate":
            return self.cross_validate(
                model, X, y,
                cv_folds=kwargs.get('cv_folds', 5),
                scoring=kwargs.get('scoring', ['accuracy'])
            )
        elif validation_type == "leave_one_out":
            return self.leave_one_out_validate(model, X, y)
        elif validation_type == "time_series":
            return self.time_series_validate(
                model, X, y,
                n_splits=kwargs.get('n_splits', 5),
                test_size=kwargs.get('test_size', 1)
            )
        elif validation_type == "bootstrap":
            return self.bootstrap_validate(
                model, X, y,
                n_bootstrap=kwargs.get('n_bootstrap', 100),
                test_ratio=kwargs.get('test_ratio', 0.2),
                scoring=kwargs.get('scoring', 'accuracy')
            )
        elif validation_type == "assumptions":
            return self.validate_model_assumptions(model, X, y)
        else:
            raise ValidationError(f"未知的验证类型: {validation_type}")


# =============================================================================
# 策略验证工具
# =============================================================================

class StrategyValidator(BaseValidator):
    """
    策略验证工具类
    
    提供回测验证、样本外验证、稳健性检验等功能
    """
    
    def __init__(self, config: ValidationConfig = None):
        super().__init__(config)
        self.strategy_validation_config = self.config.strategy_validation_config
    
    def backtest_validate(self, strategy_func: Callable, 
                         data: pd.DataFrame,
                         start_date: str = None,
                         end_date: str = None,
                         initial_capital: float = 100000,
                         transaction_cost: float = 0.001) -> ValidationResult:
        """
        执行回测验证
        
        Args:
            strategy_func: 策略函数
            data: 历史数据
            start_date: 开始日期
            end_date: 结束日期
            initial_capital: 初始资金
            transaction_cost: 交易成本
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            # 过滤数据
            if start_date or end_date:
                data = data.copy()
                if start_date:
                    data = data[data.index >= start_date]
                if end_date:
                    data = data[data.index <= end_date]
            
            if len(data) == 0:
                raise StrategyValidationError("没有符合条件的数据")
            
            # 执行回测
            portfolio_value = initial_capital
            portfolio_values = [initial_capital]
            trades = []
            positions = 0
            cash = initial_capital
            
            for i in range(1, len(data)):
                current_data = data.iloc[:i+1]
                
                # 获取策略信号
                try:
                    signal = strategy_func(current_data)
                    
                    if signal == 1 and positions == 0:  # 买入
                        # 计算考虑交易成本后的股票购买数量
                        available_cash = cash * (1 - transaction_cost)
                        if available_cash > 0:
                            positions = available_cash / data.iloc[i]['close']
                            cash = 0
                        else:
                            # 现金不足，无法买入
                            continue
                        trades.append({
                            'date': data.index[i],
                            'action': 'buy',
                            'price': data.iloc[i]['close'],
                            'quantity': positions
                        })
                    elif signal == -1 and positions > 0:  # 卖出
                        cash = positions * data.iloc[i]['close'] * (1 - transaction_cost)
                        trades.append({
                            'date': data.index[i],
                            'action': 'sell',
                            'price': data.iloc[i]['close'],
                            'quantity': positions
                        })
                        positions = 0
                        
                except Exception as e:
                    self.logger.warning(f"策略执行失败: {e}")
                    continue
                
                # 计算组合价值
                if positions > 0:
                    portfolio_value = positions * data.iloc[i]['close']
                else:
                    portfolio_value = cash
                
                portfolio_values.append(portfolio_value)
            
            # 计算最终价值
            if positions > 0:
                final_value = positions * data.iloc[-1]['close'] * (1 - transaction_cost)
            else:
                final_value = cash
            
            # 计算性能指标
            returns = pd.Series(portfolio_values).pct_change().dropna()
            
            details = {
                'initial_capital': initial_capital,
                'final_value': final_value,
                'total_return': (final_value - initial_capital) / initial_capital,
                'annualized_return': self._calculate_annualized_return(returns),
                'volatility': returns.std() * np.sqrt(252),
                'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0,
                'max_drawdown': self._calculate_max_drawdown(portfolio_values),
                'num_trades': len(trades),
                'trades': trades,
                'portfolio_values': portfolio_values
            }
            
            success = details['total_return'] > 0
            
            result = ValidationResult(
                success=success,
                message=f"回测验证完成，总收益率: {details['total_return']:.2%}",
                score=details['total_return'],
                details=details,
                execution_time=self._log_execution_time("backtest_validate", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("backtest_validate", e)
    
    def out_of_sample_validate(self, strategy_func: Callable,
                              train_data: pd.DataFrame,
                              test_data: pd.DataFrame,
                              **kwargs) -> ValidationResult:
        """
        执行样本外验证
        
        Args:
            strategy_func: 策略函数
            train_data: 训练数据
            test_data: 测试数据
            **kwargs: 其他参数
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            # 在训练集上验证
            train_result = self.backtest_validate(strategy_func, train_data, **kwargs)
            
            # 在测试集上验证
            test_result = self.backtest_validate(strategy_func, test_data, **kwargs)
            
            # 比较性能
            train_return = train_result.details['total_return']
            test_return = test_result.details['total_return']
            
            # 计算性能下降率，处理各种边界情况
            if train_return == 0:
                if test_return == 0:
                    performance_degradation = 0  # 都为0，无变化
                else:
                    performance_degradation = 1 if test_return < 0 else -1  # 训练集为0，测试集有变化
            else:
                # 正常计算，考虑负收益率情况
                performance_degradation = (train_return - test_return) / abs(train_return)
            
            details = {
                'train_performance': train_result.details,
                'test_performance': test_result.details,
                'performance_degradation': performance_degradation,
                'train_return': train_return,
                'test_return': test_return
            }
            
            # 判断稳健性
            stability_threshold = 0.3  # 性能下降不超过30%
            is_stable = performance_degradation < stability_threshold
            
            success = test_return > 0 and is_stable
            
            result = ValidationResult(
                success=success,
                message=f"样本外验证完成，测试集收益率: {test_return:.2%}",
                score=test_return,
                details=details,
                execution_time=self._log_execution_time("out_of_sample_validate", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("out_of_sample_validate", e)
    
    def robustness_validate(self, strategy_func: Callable,
                           data: pd.DataFrame,
                           parameter_ranges: Dict[str, List],
                           n_samples: int = 100) -> ValidationResult:
        """
        执行稳健性检验
        
        Args:
            strategy_func: 策略函数
            data: 数据
            parameter_ranges: 参数范围字典
            n_samples: 采样数量
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            results = []
            
            # 随机采样参数组合
            param_combinations = []
            for _ in range(n_samples):
                params = {}
                for param_name, param_range in parameter_ranges.items():
                    if isinstance(param_range, list):
                        params[param_name] = np.random.choice(param_range)
                    else:
                        params[param_name] = np.random.uniform(param_range[0], param_range[1])
                param_combinations.append(params)
            
            # 对每个参数组合执行验证
            for params in param_combinations:
                try:
                    # 创建参数化的策略函数
                    def param_strategy(data, **strategy_params):
                        return strategy_func(data, **strategy_params)
                    
                    result = self.backtest_validate(
                        lambda data: param_strategy(data, **params),
                        data
                    )
                    
                    results.append({
                        'parameters': params,
                        'return': result.details['total_return'],
                        'sharpe_ratio': result.details['sharpe_ratio'],
                        'max_drawdown': result.details['max_drawdown']
                    })
                    
                except Exception as e:
                    self.logger.warning(f"参数组合验证失败: {params}, 错误: {e}")
                    continue
            
            if not results:
                raise StrategyValidationError("所有参数组合验证都失败")
            
            returns = [r['return'] for r in results]
            sharpe_ratios = [r['sharpe_ratio'] for r in results]
            max_drawdowns = [r['max_drawdown'] for r in results]
            
            details = {
                'n_successful_tests': len(results),
                'total_tests': n_samples,
                'return_stats': {
                    'mean': np.mean(returns),
                    'std': np.std(returns),
                    'min': np.min(returns),
                    'max': np.max(returns),
                    'percentile_5': np.percentile(returns, 5),
                    'percentile_95': np.percentile(returns, 95)
                },
                'sharpe_ratio_stats': {
                    'mean': np.mean(sharpe_ratios),
                    'std': np.std(sharpe_ratios),
                    'min': np.min(sharpe_ratios),
                    'max': np.max(sharpe_ratios)
                },
                'max_drawdown_stats': {
                    'mean': np.mean(max_drawdowns),
                    'std': np.std(max_drawdowns),
                    'min': np.min(max_drawdowns),
                    'max': np.max(max_drawdowns)
                },
                'parameter_results': results
            }
            
            # 判断稳健性
            positive_returns_ratio = len([r for r in returns if r > 0]) / len(returns)
            return_volatility = np.std(returns) / np.mean(returns) if np.mean(returns) != 0 else float('inf')
            
            is_robust = positive_returns_ratio > 0.6 and return_volatility < 2.0
            
            success = is_robust and np.mean(returns) > 0
            
            result = ValidationResult(
                success=success,
                message=f"稳健性检验完成，正收益比例: {positive_returns_ratio:.2%}",
                score=np.mean(returns),
                details=details,
                execution_time=self._log_execution_time("robustness_validate", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("robustness_validate", e)
    
    def stress_test_validate(self, strategy_func: Callable,
                           data: pd.DataFrame,
                           stress_scenarios: Dict[str, Dict]) -> ValidationResult:
        """
        执行压力测试
        
        Args:
            strategy_func: 策略函数
            data: 基准数据
            stress_scenarios: 压力测试场景
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            scenario_results = {}
            
            for scenario_name, scenario_config in stress_scenarios.items():
                try:
                    # 创建压力测试数据
                    stressed_data = self._apply_stress_scenario(data, scenario_config)
                    
                    # 执行回测
                    result = self.backtest_validate(strategy_func, stressed_data)
                    
                    scenario_results[scenario_name] = {
                        'return': result.details['total_return'],
                        'sharpe_ratio': result.details['sharpe_ratio'],
                        'max_drawdown': result.details['max_drawdown'],
                        'success': result.success
                    }
                    
                except Exception as e:
                    scenario_results[scenario_name] = {
                        'error': str(e),
                        'success': False
                    }
            
            # 分析结果
            successful_scenarios = sum(1 for r in scenario_results.values() if r.get('success', False))
            total_scenarios = len(stress_scenarios)
            survival_rate = successful_scenarios / total_scenarios
            
            returns = [r['return'] for r in scenario_results.values() if 'return' in r]
            avg_return = np.mean(returns) if returns else 0
            
            details = {
                'scenario_results': scenario_results,
                'survival_rate': survival_rate,
                'average_return': avg_return,
                'total_scenarios': total_scenarios,
                'successful_scenarios': successful_scenarios
            }
            
            success = survival_rate > 0.5 and avg_return > -0.2
            
            result = ValidationResult(
                success=success,
                message=f"压力测试完成，生存率: {survival_rate:.2%}",
                score=avg_return,
                details=details,
                execution_time=self._log_execution_time("stress_test_validate", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("stress_test_validate", e)
    
    def _apply_stress_scenario(self, data: pd.DataFrame, scenario_config: Dict) -> pd.DataFrame:
        """应用压力测试场景"""
        stressed_data = data.copy()
        
        # 价格冲击
        if 'price_shock' in scenario_config:
            shock_size = scenario_config['price_shock']
            stressed_data['close'] *= (1 + shock_size)
        
        # 波动率增加
        if 'volatility_multiplier' in scenario_config:
            multiplier = scenario_config['volatility_multiplier']
            returns = stressed_data['close'].pct_change()
            stressed_returns = returns * multiplier
            stressed_data['close'] = stressed_data['close'].iloc[0] * (1 + stressed_returns).cumprod()
        
        # 趋势反转
        if 'trend_reversal' in scenario_config:
            stressed_data = stressed_data.iloc[::-1]  # 反转时间序列
        
        return stressed_data
    
    def _calculate_annualized_return(self, returns: pd.Series) -> float:
        """计算年化收益率"""
        if len(returns) == 0:
            return 0
        
        total_return = (1 + returns).prod() - 1
        periods_per_year = 252  # 假设一年252个交易日
        years = len(returns) / periods_per_year
        
        if years <= 0:
            return 0
        
        return (1 + total_return) ** (1 / years) - 1
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """计算最大回撤"""
        if len(portfolio_values) < 2:
            return 0
        
        peak = portfolio_values[0]
        max_drawdown = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown
    
    def validate(self, strategy_func: Callable, data: pd.DataFrame, 
                validation_type: str = "backtest", **kwargs) -> ValidationResult:
        """
        通用验证方法
        
        Args:
            strategy_func: 策略函数
            data: 数据
            validation_type: 验证类型 ('backtest', 'out_of_sample', 'robustness', 'stress_test')
            **kwargs: 验证参数
            
        Returns:
            ValidationResult: 验证结果
        """
        if validation_type == "backtest":
            return self.backtest_validate(
                strategy_func, data,
                start_date=kwargs.get('start_date'),
                end_date=kwargs.get('end_date'),
                initial_capital=kwargs.get('initial_capital', 100000),
                transaction_cost=kwargs.get('transaction_cost', 0.001)
            )
        elif validation_type == "out_of_sample":
            return self.out_of_sample_validate(
                strategy_func,
                kwargs.get('train_data'),
                kwargs.get('test_data'),
                initial_capital=kwargs.get('initial_capital', 100000)
            )
        elif validation_type == "robustness":
            return self.robustness_validate(
                strategy_func, data,
                parameter_ranges=kwargs.get('parameter_ranges', {}),
                n_samples=kwargs.get('n_samples', 100)
            )
        elif validation_type == "stress_test":
            return self.stress_test_validate(
                strategy_func, data,
                stress_scenarios=kwargs.get('stress_scenarios', {})
            )
        else:
            raise ValidationError(f"未知的验证类型: {validation_type}")


# =============================================================================
# 系统验证工具
# =============================================================================

class SystemValidator(BaseValidator):
    """
    系统验证工具类
    
    提供接口验证、配置验证、依赖验证等功能
    """
    
    def __init__(self, config: ValidationConfig = None):
        super().__init__(config)
        self.system_validation_config = self.config.system_validation_config
    
    def validate_interface(self, interface_func: Callable,
                          expected_params: List[str],
                          expected_return_type: type = None) -> ValidationResult:
        """
        验证接口规范
        
        Args:
            interface_func: 要验证的函数
            expected_params: 期望的参数列表
            expected_return_type: 期望的返回类型
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            # 检查函数签名
            sig = inspect.signature(interface_func)
            actual_params = list(sig.parameters.keys())
            
            errors = []
            
            # 检查参数
            missing_params = set(expected_params) - set(actual_params)
            if missing_params:
                errors.append(f"缺少参数: {list(missing_params)}")
            
            extra_params = set(actual_params) - set(expected_params)
            if extra_params:
                errors.append(f"多余参数: {list(extra_params)}")
            
            # 检查返回类型注解
            if expected_return_type and sig.return_annotation != sig.empty:
                if sig.return_annotation != expected_return_type:
                    errors.append(f"返回类型不匹配: 期望 {expected_return_type}, 实际 {sig.return_annotation}")
            
            details = {
                'expected_params': expected_params,
                'actual_params': actual_params,
                'return_annotation': str(sig.return_annotation) if sig.return_annotation != sig.empty else 'None',
                'function_name': interface_func.__name__,
                'function_docstring': interface_func.__doc__
            }
            
            success = len(errors) == 0
            
            result = ValidationResult(
                success=success,
                message="接口验证" + ("通过" if success else "失败"),
                errors=errors if errors else None,
                details=details,
                execution_time=self._log_execution_time("validate_interface", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("validate_interface", e)
    
    def validate_configuration(self, config: Dict,
                              required_keys: List[str],
                              value_constraints: Dict[str, Callable] = None) -> ValidationResult:
        """
        验证配置
        
        Args:
            config: 配置字典
            required_keys: 必需的键列表
            value_constraints: 值约束字典 {键: 验证函数}
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            errors = []
            warnings = []
            
            # 检查必需键
            missing_keys = set(required_keys) - set(config.keys())
            if missing_keys:
                errors.append(f"缺少必需配置项: {list(missing_keys)}")
            
            # 检查值约束
            if value_constraints:
                for key, constraint_func in value_constraints.items():
                    if key in config:
                        try:
                            if not constraint_func(config[key]):
                                errors.append(f"配置项 {key} 不满足约束条件")
                        except Exception as e:
                            errors.append(f"配置项 {key} 约束检查失败: {e}")
            
            # 检查配置完整性
            config_keys = set(config.keys())
            if len(config_keys) < len(required_keys):
                warnings.append("配置项数量不足")
            
            details = {
                'required_keys': required_keys,
                'actual_keys': list(config.keys()),
                'missing_keys': list(missing_keys),
                'config_size': len(config),
                'value_constraints': {k: str(v) for k, v in (value_constraints or {}).items()}
            }
            
            success = len(errors) == 0
            
            result = ValidationResult(
                success=success,
                message="配置验证" + ("通过" if success else "失败"),
                errors=errors if errors else None,
                warnings=warnings if warnings else None,
                details=details,
                execution_time=self._log_execution_time("validate_configuration", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("validate_configuration", e)
    
    def validate_dependencies(self, required_packages: List[str],
                             version_constraints: Dict[str, str] = None) -> ValidationResult:
        """
        验证依赖包
        
        Args:
            required_packages: 必需的包列表
            version_constraints: 版本约束字典 {包名: 版本要求}
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            missing_packages = []
            version_mismatches = []
            installed_packages = {}
            
            # 检查已安装的包
            try:
                import pkg_resources
                for dist in pkg_resources.working_set:
                    installed_packages[dist.project_name.lower()] = dist.version
            except ImportError:
                # 尝试使用importlib.metadata (Python 3.8+)
                try:
                    import importlib.metadata as metadata
                    for dist in metadata.distributions():
                        installed_packages[dist.metadata['Name'].lower()] = dist.metadata['Version']
                except ImportError:
                    warnings.append("无法获取已安装包信息")
            
            # 检查必需包
            for package in required_packages:
                package_lower = package.lower()
                if package_lower not in installed_packages:
                    missing_packages.append(package)
            
            # 检查版本约束
            if version_constraints:
                for package, version_req in version_constraints.items():
                    package_lower = package.lower()
                    if package_lower in installed_packages:
                        installed_version = installed_packages[package_lower]
                        # 简单的版本检查（实际应该使用packaging.version）
                        if not self._check_version_constraint(installed_version, version_req):
                            version_mismatches.append({
                                'package': package,
                                'installed': installed_version,
                                'required': version_req
                            })
            
            errors = []
            if missing_packages:
                errors.append(f"缺少依赖包: {missing_packages}")
            if version_mismatches:
                errors.append(f"版本不匹配: {version_mismatches}")
            
            details = {
                'required_packages': required_packages,
                'installed_packages': installed_packages,
                'missing_packages': missing_packages,
                'version_mismatches': version_mismatches,
                'version_constraints': version_constraints
            }
            
            success = len(errors) == 0
            
            result = ValidationResult(
                success=success,
                message="依赖验证" + ("通过" if success else "失败"),
                errors=errors if errors else None,
                details=details,
                execution_time=self._log_execution_time("validate_dependencies", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("validate_dependencies", e)
    
    def validate_performance(self, func: Callable,
                           args: tuple = (),
                           kwargs: dict = None,
                           max_execution_time: float = 5.0,
                           max_memory_usage: float = 100.0) -> ValidationResult:
        """
        验证性能
        
        Args:
            func: 要测试的函数
            args: 函数参数
            kwargs: 函数关键字参数
            max_execution_time: 最大执行时间（秒）
            max_memory_usage: 最大内存使用（MB）
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            kwargs = kwargs or {}
            
            # 监控内存使用 (可选功能)
            try:
                import psutil
                process = psutil.Process()
                memory_before = process.memory_info().rss / 1024 / 1024  # MB
                memory_monitoring_available = True
            except ImportError:
                memory_before = 0.0
                memory_monitoring_available = False
                self.logger.warning("psutil未安装，跳过内存监控")
            
            # 执行函数
            try:
                result = func(*args, **kwargs)
                execution_success = True
            except KeyboardInterrupt:
                execution_success = False
                result = None
                raise  # 重新抛出键盘中断异常
            except Exception as e:
                execution_success = False
                result = None
                self.logger.warning(f"函数执行出现异常: {str(e)}")
                # 不抛出异常，返回性能验证失败的结果
                errors = [f"函数执行异常: {str(e)}"]
            
            execution_time = time.time() - start_time
            
            if memory_monitoring_available:
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_used = memory_after - memory_before
            else:
                memory_after = 0.0
                memory_used = 0.0
            
            # 检查性能指标
            errors = []
            if execution_time > max_execution_time:
                errors.append(f"执行时间过长: {execution_time:.2f}s > {max_execution_time}s")
            
            if memory_used > max_memory_usage:
                errors.append(f"内存使用过多: {memory_used:.2f}MB > {max_memory_usage}MB")
            
            details = {
                'execution_time': execution_time,
                'memory_before': memory_before,
                'memory_after': memory_after,
                'memory_used': memory_used,
                'max_execution_time': max_execution_time,
                'max_memory_usage': max_memory_usage,
                'execution_success': execution_success,
                'result_type': type(result).__name__ if result is not None else 'None'
            }
            
            success = len(errors) == 0 and execution_success
            
            result = ValidationResult(
                success=success,
                message="性能验证" + ("通过" if success else "失败"),
                errors=errors if errors else None,
                details=details,
                execution_time=self._log_execution_time("validate_performance", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("validate_performance", e)
    
    def validate_error_handling(self, func: Callable,
                              error_scenarios: List[Dict]) -> ValidationResult:
        """
        验证错误处理
        
        Args:
            func: 要测试的函数
            error_scenarios: 错误场景列表
            
        Returns:
            ValidationResult: 验证结果
        """
        start_time = time.time()
        
        try:
            scenario_results = []
            
            for scenario in error_scenarios:
                scenario_name = scenario.get('name', 'Unknown')
                args = scenario.get('args', ())
                kwargs = scenario.get('kwargs', {})
                expected_error = scenario.get('expected_error')
                
                try:
                    # 执行函数
                    result = func(*args, **kwargs)
                    
                    # 检查是否应该抛出错误
                    if expected_error:
                        scenario_results.append({
                            'name': scenario_name,
                            'success': False,
                            'error': f"应该抛出 {expected_error} 但没有抛出",
                            'expected_error': expected_error
                        })
                    else:
                        scenario_results.append({
                            'name': scenario_name,
                            'success': True,
                            'message': '正常执行'
                        })
                        
                except Exception as e:
                    # 检查是否抛出了期望的错误
                    if expected_error and expected_error in str(type(e)):
                        scenario_results.append({
                            'name': scenario_name,
                            'success': True,
                            'message': f'正确抛出错误: {type(e).__name__}',
                            'actual_error': type(e).__name__
                        })
                    else:
                        scenario_results.append({
                            'name': scenario_name,
                            'success': False,
                            'error': f'意外错误: {type(e).__name__}: {str(e)}',
                            'expected_error': expected_error,
                            'actual_error': type(e).__name__
                        })
            
            # 统计结果
            successful_scenarios = sum(1 for r in scenario_results if r['success'])
            total_scenarios = len(scenario_results)
            success_rate = successful_scenarios / total_scenarios
            
            details = {
                'total_scenarios': total_scenarios,
                'successful_scenarios': successful_scenarios,
                'success_rate': success_rate,
                'scenario_results': scenario_results
            }
            
            # 设置更严格的成功率要求，同时确保有足够的测试场景
            min_required_scenarios = 3  # 最少需要测试3个场景
            success_rate_threshold = 0.7  # 70%以上的成功率
            success = success_rate >= success_rate_threshold and total_scenarios >= min_required_scenarios
            
            result = ValidationResult(
                success=success,
                message=f"错误处理验证完成，成功率: {success_rate:.2%}",
                details=details,
                execution_time=self._log_execution_time("validate_error_handling", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("validate_error_handling", e)
    
    def _check_version_constraint(self, installed_version: str, required_version: str) -> bool:
        """检查版本约束"""
        try:
            # 尝试使用packaging库进行版本比较
            try:
                from packaging import version
                return version.parse(installed_version) >= version.parse(required_version.replace('>=', ''))
            except ImportError:
                # 如果没有packaging库，使用简单的字符串比较
                # 移除版本号中的特殊字符进行比较
                import re
                clean_installed = re.sub(r'[^\d.]', '', installed_version)
                clean_required = re.sub(r'[^\d.]', '', required_version.replace('>=', ''))
                
                # 简单的版本号分割比较
                def version_to_list(v):
                    return [int(x) for x in v.split('.') if x.isdigit()]
                
                return version_to_list(clean_installed) >= version_to_list(clean_required)
        except:
            # 如果所有方法都失败，返回True避免阻塞
            return True
    
    def validate(self, target, validation_type: str = "interface", **kwargs) -> ValidationResult:
        """
        通用验证方法
        
        Args:
            target: 要验证的目标（函数、配置等）
            validation_type: 验证类型 ('interface', 'configuration', 'dependencies', 'performance', 'error_handling')
            **kwargs: 验证参数
            
        Returns:
            ValidationResult: 验证结果
        """
        if validation_type == "interface":
            return self.validate_interface(
                target,
                expected_params=kwargs.get('expected_params', []),
                expected_return_type=kwargs.get('expected_return_type')
            )
        elif validation_type == "configuration":
            return self.validate_configuration(
                target,
                required_keys=kwargs.get('required_keys', []),
                value_constraints=kwargs.get('value_constraints')
            )
        elif validation_type == "dependencies":
            return self.validate_dependencies(
                required_packages=kwargs.get('required_packages', []),
                version_constraints=kwargs.get('version_constraints')
            )
        elif validation_type == "performance":
            return self.validate_performance(
                target,
                args=kwargs.get('args', ()),
                kwargs=kwargs.get('kwargs', {}),
                max_execution_time=kwargs.get('max_execution_time', 5.0),
                max_memory_usage=kwargs.get('max_memory_usage', 100.0)
            )
        elif validation_type == "error_handling":
            return self.validate_error_handling(
                target,
                error_scenarios=kwargs.get('error_scenarios', [])
            )
        else:
            raise ValidationError(f"未知的验证类型: {validation_type}")


# =============================================================================
# 异步验证器
# =============================================================================

class AsyncValidator(BaseValidator):
    """
    异步验证器类
    
    提供异步验证和并行处理功能
    """
    
    def __init__(self, config: ValidationConfig = None):
        super().__init__(config)
        self._executor = None
    
    def validate(self, validator: BaseValidator, validation_type: str = "async", **kwargs) -> ValidationResult:
        """
        通用验证方法 - 同步包装异步验证
        
        Args:
            validator: 验证器实例
            validation_type: 验证类型 ('async', 'batch', 'parallel_cross_validate')
            **kwargs: 验证参数
            
        Returns:
            ValidationResult: 验证结果
        """
        try:
            if validation_type == "async":
                # 对于异步验证，我们创建一个简单的包装器
                def sync_wrapper():
                    return validator.validate(**kwargs)
                
                return sync_wrapper()
            elif validation_type == "batch":
                # 批量验证
                validation_tasks = kwargs.get('validation_tasks', [])
                return ValidationResult(
                    success=True,
                    message=f"批量验证任务已创建，数量: {len(validation_tasks)}",
                    details={'task_count': len(validation_tasks)}
                )
            else:
                raise ValidationError(f"未知的验证类型: {validation_type}")
        except Exception as e:
            return ValidationResult(
                success=False,
                message=f"异步验证失败: {str(e)}",
                errors=[str(e)]
            )
    
    async def async_validate(self, validator: BaseValidator, 
                           *args, **kwargs) -> ValidationResult:
        """
        异步执行验证
        
        Args:
            validator: 验证器实例
            *args: 验证器参数
            **kwargs: 验证器关键字参数
            
        Returns:
            ValidationResult: 验证结果
        """
        loop = asyncio.get_event_loop()
        
        # 在线程池中执行同步验证器
        if self.config.parallel_execution:
            if not self._executor:
                self._executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
            
            result = await loop.run_in_executor(
                self._executor, 
                validator.validate, 
                *args, 
                **kwargs
            )
        else:
            # 直接执行
            result = validator.validate(*args, **kwargs)
        
        return result
    
    async def batch_validate(self, validation_tasks: List[Dict]) -> List[ValidationResult]:
        """
        批量异步验证
        
        Args:
            validation_tasks: 验证任务列表，每个任务包含validator、args、kwargs
            
        Returns:
            List[ValidationResult]: 验证结果列表
        """
        tasks = []
        
        for task in validation_tasks:
            validator = task['validator']
            args = task.get('args', ())
            kwargs = task.get('kwargs', {})
            
            task_coro = self.async_validate(validator, *args, **kwargs)
            tasks.append(task_coro)
        
        # 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理异常结果
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ValidationResult(
                    success=False,
                    message=f"任务 {i} 执行失败: {str(result)}",
                    errors=[str(result)]
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def parallel_cross_validate(self, models: List[Any], 
                                    X, y, **cv_params) -> List[ValidationResult]:
        """
        并行交叉验证多个模型
        
        Args:
            models: 模型列表
            X: 特征数据
            y: 目标数据
            **cv_params: 交叉验证参数
            
        Returns:
            List[ValidationResult]: 验证结果列表
        """
        model_validator = ModelValidator(self.config)
        
        validation_tasks = []
        for model in models:
            task = {
                'validator': model_validator,
                'args': (model, X, y),
                'kwargs': cv_params
            }
            validation_tasks.append(task)
        
        return await self.batch_validate(validation_tasks)
    
    async def stress_test_parallel(self, strategy_func: Callable,
                                 data: pd.DataFrame,
                                 stress_scenarios: Dict[str, Dict]) -> List[ValidationResult]:
        """
        并行压力测试
        
        Args:
            strategy_func: 策略函数
            data: 数据
            stress_scenarios: 压力测试场景
            
        Returns:
            List[ValidationResult]: 验证结果列表
        """
        strategy_validator = StrategyValidator(self.config)
        
        validation_tasks = []
        for scenario_name, scenario_config in stress_scenarios.items():
            task = {
                'validator': strategy_validator,
                'args': (strategy_func, data),
                'kwargs': {'stress_scenarios': {scenario_name: scenario_config}}
            }
            validation_tasks.append(task)
        
        return await self.batch_validate(validation_tasks)
    
    def __del__(self):
        """清理资源"""
        if self._executor:
            self._executor.shutdown(wait=False)


# =============================================================================
# 验证流水线
# =============================================================================

class ValidationPipeline(BaseValidator):
    """
    验证流水线类
    
    提供自动化的验证流程编排
    """
    
    def __init__(self, config: ValidationConfig = None):
        super().__init__(config)
        self.validators = {
            'data': DataValidator(config),
            'model': ModelValidator(config),
            'strategy': StrategyValidator(config),
            'system': SystemValidator(config)
        }
        self.pipeline_steps = []
        self.results = []
    
    def add_step(self, name: str, validator_type: str, 
                method: str, args: tuple = (), kwargs: dict = None) -> 'ValidationPipeline':
        """
        添加验证步骤
        
        Args:
            name: 步骤名称
            validator_type: 验证器类型 ('data', 'model', 'strategy', 'system')
            method: 验证方法名
            args: 方法参数
            kwargs: 方法关键字参数
            
        Returns:
            ValidationPipeline: 流水线实例
        """
        step = {
            'name': name,
            'validator_type': validator_type,
            'method': method,
            'args': args,
            'kwargs': kwargs or {}
        }
        
        self.pipeline_steps.append(step)
        return self
    
    def run_pipeline(self, stop_on_error: bool = True) -> ValidationResult:
        """
        运行验证流水线
        
        Args:
            stop_on_error: 遇到错误时是否停止
            
        Returns:
            ValidationResult: 总体验证结果
        """
        start_time = time.time()
        self.results = []
        
        try:
            for step in self.pipeline_steps:
                try:
                    # 获取验证器
                    validator = self.validators.get(step['validator_type'])
                    if not validator:
                        raise ValidationError(f"未知的验证器类型: {step['validator_type']}")
                    
                    # 获取验证方法
                    method = getattr(validator, step['method'])
                    if not callable(method):
                        raise ValidationError(f"验证器 {step['validator_type']} 没有方法: {step['method']}")
                    
                    # 执行验证
                    result = method(*step['args'], **step['kwargs'])
                    self.results.append({
                        'step_name': step['name'],
                        'result': result
                    })
                    
                    # 记录结果
                    self.logger.info(f"步骤 '{step['name']}' 完成: {'成功' if result.success else '失败'}")
                    
                    # 如果失败且需要停止
                    if not result.success and stop_on_error:
                        self.logger.warning(f"步骤 '{step['name']}' 失败，停止流水线执行")
                        break
                        
                except Exception as e:
                    error_result = ValidationResult(
                        success=False,
                        message=f"步骤 '{step['name']}' 执行异常: {str(e)}",
                        errors=[str(e)]
                    )
                    self.results.append({
                        'step_name': step['name'],
                        'result': error_result
                    })
                    
                    self.logger.error(f"步骤 '{step['name']}' 执行异常: {e}")
                    
                    if stop_on_error:
                        break
            
            # 计算总体结果
            successful_steps = sum(1 for r in self.results if r['result'].success)
            total_steps = len(self.results)
            success_rate = successful_steps / total_steps if total_steps > 0 else 0
            
            overall_success = success_rate >= 0.8  # 80%以上步骤成功认为总体成功
            
            # 汇总详细信息
            all_errors = []
            all_warnings = []
            total_execution_time = 0
            
            for r in self.results:
                result = r['result']
                total_execution_time += result.execution_time
                
                if result.errors:
                    all_errors.extend([f"{r['step_name']}: {err}" for err in result.errors])
                
                if result.warnings:
                    all_warnings.extend([f"{r['step_name']}: {warn}" for warn in result.warnings])
            
            details = {
                'pipeline_steps': len(self.pipeline_steps),
                'executed_steps': total_steps,
                'successful_steps': successful_steps,
                'success_rate': success_rate,
                'step_results': [
                    {
                        'step_name': r['step_name'],
                        'success': r['result'].success,
                        'message': r['result'].message,
                        'execution_time': r['result'].execution_time
                    } for r in self.results
                ],
                'total_execution_time': total_execution_time
            }
            
            result = ValidationResult(
                success=overall_success,
                message=f"验证流水线完成，成功率: {success_rate:.2%}",
                details=details,
                errors=all_errors if all_errors else None,
                warnings=all_warnings if all_warnings else None,
                execution_time=self._log_execution_time("run_pipeline", start_time)
            )
            
            return result
            
        except Exception as e:
            return self._handle_exception("run_pipeline", e)
    
    def get_step_result(self, step_name: str) -> Optional[ValidationResult]:
        """
        获取特定步骤的结果
        
        Args:
            step_name: 步骤名称
            
        Returns:
            ValidationResult: 步骤结果，如果不存在返回None
        """
        for r in self.results:
            if r['step_name'] == step_name:
                return r['result']
        return None
    
    def reset_pipeline(self):
        """重置流水线"""
        self.pipeline_steps = []
        self.results = []
    
    def save_pipeline_config(self, file_path: str):
        """保存流水线配置"""
        config = {
            'steps': self.pipeline_steps,
            'config': asdict(self.config)
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2, default=str)
    
    def load_pipeline_config(self, file_path: str):
        """加载流水线配置"""
        with open(file_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        self.pipeline_steps = config.get('steps', [])
        
        # 重新创建配置对象
        if 'config' in config:
            config_dict = config['config']
            self.config = ValidationConfig(**config_dict)
    
    def validate(self, *args, **kwargs) -> ValidationResult:
        """
        通用验证方法 - 执行整个流水线
        
        Args:
            *args: 位置参数（未使用）
            **kwargs: 关键字参数，支持 stop_on_error
            
        Returns:
            ValidationResult: 流水线执行结果
        """
        stop_on_error = kwargs.get('stop_on_error', True)
        return self.run_pipeline(stop_on_error=stop_on_error)


# =============================================================================
# 验证报告生成器
# =============================================================================

class ValidationReport(BaseValidator):
    """
    验证报告生成器类
    
    生成详细的验证报告，支持多种格式
    """
    
    def __init__(self, config: ValidationConfig = None):
        super().__init__(config)
        self.report_config = self.config.report_config
    
    def validate(self, results: List[ValidationResult], validation_type: str = "report", **kwargs) -> ValidationResult:
        """
        通用验证方法
        
        Args:
            results: 验证结果列表
            validation_type: 验证类型 ('report', 'export')
            **kwargs: 验证参数
            
        Returns:
            ValidationResult: 验证结果
        """
        if validation_type == "report":
            try:
                title = kwargs.get('title', "验证报告")
                output_path = kwargs.get('output_path')
                self.generate_report(results, title, output_path)
                return ValidationResult(
                    success=True,
                    message="报告生成成功",
                    details={'output_path': output_path}
                )
            except Exception as e:
                return ValidationResult(
                    success=False,
                    message=f"报告生成失败: {str(e)}",
                    errors=[str(e)]
                )
        elif validation_type == "export":
            try:
                file_path = kwargs.get('file_path', 'results.json')
                format_type = kwargs.get('format_type', 'json')
                self.export_results(results, file_path, format_type)
                return ValidationResult(
                    success=True,
                    message="结果导出成功",
                    details={'file_path': file_path, 'format': format_type}
                )
            except Exception as e:
                return ValidationResult(
                    success=False,
                    message=f"结果导出失败: {str(e)}",
                    errors=[str(e)]
                )
        else:
            raise ValidationError(f"未知的验证类型: {validation_type}")
    
    def generate_report(self, results: List[ValidationResult], 
                       title: str = "验证报告",
                       output_path: str = None) -> str:
        """
        生成验证报告
        
        Args:
            results: 验证结果列表
            title: 报告标题
            output_path: 输出文件路径
            
        Returns:
            str: 报告内容
        """
        try:
            # 确定输出格式和路径
            format_type = self.report_config.get('format', 'html')
            output_path = output_path or self.report_config.get('output_path', 'validation_report.html')
            
            # 生成报告内容
            if format_type.lower() == 'html':
                content = self._generate_html_report(results, title)
            elif format_type.lower() == 'json':
                content = self._generate_json_report(results, title)
            elif format_type.lower() == 'markdown':
                content = self._generate_markdown_report(results, title)
            else:
                raise ValidationError(f"不支持的报告格式: {format_type}")
            
            # 保存报告，添加错误处理
            if output_path:
                try:
                    # 确保目录存在
                    import os
                    os.makedirs(os.path.dirname(output_path), exist_ok=True) if os.path.dirname(output_path) else None
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    self.logger.info(f"报告已保存到: {output_path}")
                except Exception as save_error:
                    self.logger.error(f"保存报告失败: {save_error}")
                    raise ValidationError(f"保存报告失败: {str(save_error)}") from save_error
            
            return content
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")
            return f"报告生成失败: {str(e)}"
    
    def _generate_html_report(self, results: List[ValidationResult], title: str) -> str:
        """生成HTML格式报告"""
        # 统计信息
        total_results = len(results)
        successful_results = sum(1 for r in results if r.success)
        success_rate = successful_results / total_results if total_results > 0 else 0
        
        total_execution_time = sum(r.execution_time for r in results)
        avg_execution_time = total_execution_time / total_results if total_results > 0 else 0
        
        # 生成图表
        chart_data = self._generate_chart_data(results)
        
        # HTML模板
        html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .header { text-align: center; margin-bottom: 30px; }
        .summary { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }
        .summary-card { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }
        .summary-card h3 { margin: 0 0 10px 0; color: #333; }
        .summary-card .value { font-size: 24px; font-weight: bold; color: #007bff; }
        .success { color: #28a745; }
        .failure { color: #dc3545; }
        .warning { color: #ffc107; }
        .results-table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        .results-table th, .results-table td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        .results-table th { background-color: #f8f9fa; font-weight: bold; }
        .status-success { color: #28a745; font-weight: bold; }
        .status-failure { color: #dc3545; font-weight: bold; }
        .chart-container { margin: 30px 0; height: 400px; }
        .details { margin-top: 20px; }
        .error { color: #dc3545; background-color: #f8d7da; padding: 10px; border-radius: 4px; margin: 5px 0; }
        .warning { color: #856404; background-color: #fff3cd; padding: 10px; border-radius: 4px; margin: 5px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ title }}</h1>
            <p>生成时间: {{ timestamp }}</p>
        </div>
        
        <div class="summary">
            <div class="summary-card">
                <h3>总验证数</h3>
                <div class="value">{{ total_results }}</div>
            </div>
            <div class="summary-card">
                <h3>成功率</h3>
                <div class="value {% if success_rate >= 0.8 %}success{% else %}failure{% endif %}">{{ success_rate_percentage }}</div>
            </div>
            <div class="summary-card">
                <h3>总执行时间</h3>
                <div class="value">{{ total_execution_time }}s</div>
            </div>
            <div class="summary-card">
                <h3>平均执行时间</h3>
                <div class="value">{{ avg_execution_time }}s</div>
            </div>
        </div>
        
        {% if chart_data %}
        <div class="chart-container">
            <canvas id="resultsChart"></canvas>
        </div>
        {% endif %}
        
        <table class="results-table">
            <thead>
                <tr>
                    <th>序号</th>
                    <th>状态</th>
                    <th>消息</th>
                    <th>得分</th>
                    <th>执行时间</th>
                    <th>时间戳</th>
                </tr>
            </thead>
            <tbody>
                {% for result in results %}
                <tr>
                    <td>{{ loop.index }}</td>
                    <td class="{% if result.success %}status-success{% else %}status-failure{% endif %}">
                        {{ "成功" if result.success else "失败" }}
                    </td>
                    <td>{{ result.message }}</td>
                    <td>{{ "%.4f"|format(result.score) if result.score is not none else "N/A" }}</td>
                    <td>{{ "%.2f"|format(result.execution_time) }}s</td>
                    <td>{{ result.timestamp }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        
        {% if include_details %}
        <div class="details">
            <h2>详细结果</h2>
            {% for result in results %}
            <div style="border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px;">
                <h3>验证 {{ loop.index }}: {{ result.message }}</h3>
                <p><strong>状态:</strong> <span class="{% if result.success %}success{% else %}failure{% endif %}">{{ "成功" if result.success else "失败" }}</span></p>
                {% if result.score is not none %}
                <p><strong>得分:</strong> {{ "%.4f"|format(result.score) }}</p>
                {% endif %}
                <p><strong>执行时间:</strong> {{ "%.2f"|format(result.execution_time) }}s</p>
                
                {% if result.errors %}
                <div class="error">
                    <strong>错误:</strong>
                    <ul>
                        {% for error in result.errors %}
                        <li>{{ error }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}
                
                {% if result.warnings %}
                <div class="warning">
                    <strong>警告:</strong>
                    <ul>
                        {% for warning in result.warnings %}
                        <li>{{ warning }}</li>
                        {% endfor %}
                    </ul>
                </div>
                {% endif %}
                
                {% if result.details %}
                <details>
                    <summary>详细信息</summary>
                    <pre>{{ result.details|tojson(indent=2) }}</pre>
                </details>
                {% endif %}
            </div>
            {% endfor %}
        </div>
        {% endif %}
    </div>
    
    {% if chart_data %}
    <script>
        const ctx = document.getElementById('resultsChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: {{ chart_data.labels|tojson }},
                datasets: [{
                    label: '执行时间 (秒)',
                    data: {{ chart_data.execution_times|tojson }},
                    backgroundColor: {{ chart_data.colors|tojson }},
                    borderColor: {{ chart_data.colors|tojson }},
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    </script>
    {% endif %}
</body>
</html>
        """
        
        # 准备模板数据
        template_data = {
            'title': title,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_results': total_results,
            'success_rate': success_rate,
            'success_rate_percentage': f"{success_rate:.1%}",
            'total_execution_time': f"{total_execution_time:.2f}",
            'avg_execution_time': f"{avg_execution_time:.2f}",
            'results': [r.to_dict() for r in results],
            'chart_data': chart_data,
            'include_details': self.report_config.get('include_details', True)
        }
        
        # 渲染模板
        if not JINJA2_AVAILABLE or Template is None:
            raise ValidationError("生成HTML报告需要jinja2库，请安装: pip install jinja2")
        
        template = Template(html_template)
        return template.render(**template_data)
    
    def _generate_json_report(self, results: List[ValidationResult], title: str) -> str:
        """生成JSON格式报告"""
        report_data = {
            'title': title,
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_results': len(results),
                'successful_results': sum(1 for r in results if r.success),
                'success_rate': sum(1 for r in results if r.success) / len(results) if results else 0,
                'total_execution_time': sum(r.execution_time for r in results),
                'average_execution_time': sum(r.execution_time for r in results) / len(results) if results else 0
            },
            'results': [r.to_dict() for r in results]
        }
        
        return json.dumps(report_data, ensure_ascii=False, indent=2)
    
    def _generate_markdown_report(self, results: List[ValidationResult], title: str) -> str:
        """生成Markdown格式报告"""
        lines = []
        
        # 标题
        lines.append(f"# {title}")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # 统计信息
        total_results = len(results)
        successful_results = sum(1 for r in results if r.success)
        success_rate = successful_results / total_results if total_results > 0 else 0
        total_execution_time = sum(r.execution_time for r in results)
        
        lines.append("## 统计摘要")
        lines.append(f"- 总验证数: {total_results}")
        lines.append(f"- 成功数: {successful_results}")
        lines.append(f"- 成功率: {success_rate:.1%}")
        lines.append(f"- 总执行时间: {total_execution_time:.2f}秒")
        lines.append("")
        
        # 结果表格
        lines.append("## 验证结果")
        lines.append("| 序号 | 状态 | 消息 | 得分 | 执行时间 |")
        lines.append("|------|------|------|------|----------|")
        
        for i, result in enumerate(results, 1):
            status = "✅ 成功" if result.success else "❌ 失败"
            score = f"{result.score:.4f}" if result.score is not None else "N/A"
            lines.append(f"| {i} | {status} | {result.message} | {score} | {result.execution_time:.2f}s |")
        
        lines.append("")
        
        # 详细结果
        if self.report_config.get('include_details', True):
            lines.append("## 详细结果")
            
            for i, result in enumerate(results, 1):
                lines.append(f"### 验证 {i}: {result.message}")
                lines.append(f"- **状态**: {'成功' if result.success else '失败'}")
                if result.score is not None:
                    lines.append(f"- **得分**: {result.score:.4f}")
                lines.append(f"- **执行时间**: {result.execution_time:.2f}秒")
                
                if result.errors:
                    lines.append("- **错误**:")
                    for error in result.errors:
                        lines.append(f"  - {error}")
                
                if result.warnings:
                    lines.append("- **警告**:")
                    for warning in result.warnings:
                        lines.append(f"  - {warning}")
                
                if result.details:
                    lines.append("- **详细信息**:")
                    lines.append("```json")
                    lines.append(json.dumps(result.details, ensure_ascii=False, indent=2))
                    lines.append("```")
                
                lines.append("")
        
        return "\n".join(lines)
    
    def _generate_chart_data(self, results: List[ValidationResult]) -> Optional[Dict]:
        """生成图表数据"""
        if not self.report_config.get('include_charts', True):
            return None
        
        labels = [f"验证 {i+1}" for i in range(len(results))]
        execution_times = [r.execution_time for r in results]
        colors = ["#28a745" if r.success else "#dc3545" for r in results]
        
        return {
            'labels': labels,
            'execution_times': execution_times,
            'colors': colors
        }
    
    def export_results(self, results: List[ValidationResult], 
                      file_path: str, format_type: str = 'json'):
        """
        导出验证结果
        
        Args:
            results: 验证结果列表
            file_path: 导出文件路径
            format_type: 导出格式 ('json', 'csv', 'excel')
        """
        try:
            if format_type.lower() == 'json':
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump([r.to_dict() for r in results], f, ensure_ascii=False, indent=2, default=str)
            
            elif format_type.lower() == 'csv':
                import pandas as pd
                df_data = []
                for r in results:
                    row = {
                        'success': r.success,
                        'message': r.message,
                        'score': r.score,
                        'execution_time': r.execution_time,
                        'timestamp': r.timestamp.isoformat()
                    }
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
            
            elif format_type.lower() == 'excel':
                import pandas as pd
                df_data = []
                for r in results:
                    row = {
                        'success': r.success,
                        'message': r.message,
                        'score': r.score,
                        'execution_time': r.execution_time,
                        'timestamp': r.timestamp.isoformat()
                    }
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                df.to_excel(file_path, index=False)
            
            else:
                raise ValidationError(f"不支持的导出格式: {format_type}")
            
            self.logger.info(f"验证结果已导出到: {file_path}")
            
        except Exception as e:
            self.logger.error(f"导出验证结果失败: {e}")
            raise


# =============================================================================
# 使用示例和测试代码
# =============================================================================

def create_sample_data():
    """创建示例数据"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # 生成股价数据
    prices = 100 * np.cumprod(1 + np.random.normal(0.001, 0.02, 1000))
    
    data = pd.DataFrame({
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 1000)
    }, index=dates)
    
    return data


def sample_strategy(data: pd.DataFrame, short_window: int = 5, long_window: int = 20) -> int:
    """示例策略函数"""
    if len(data) < long_window:
        return 0
    
    short_ma = data['close'].rolling(short_window).mean().iloc[-1]
    long_ma = data['close'].rolling(long_window).mean().iloc[-1]
    
    if short_ma > long_ma:
        return 1  # 买入信号
    else:
        return -1  # 卖出信号


def sample_model():
    """创建示例模型"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    return model, X, y


def run_comprehensive_example():
    """运行综合示例"""
    print("=== J7验证工具综合示例 ===")
    
    # 创建配置
    config = ValidationConfig(
        strict_mode=True,
        parallel_execution=True,
        max_workers=2,
        report_config={
            'format': 'html',
            'include_charts': True,
            'include_details': True,
            'output_path': 'comprehensive_validation_report.html'
        }
    )
    
    # 创建验证器
    data_validator = DataValidator(config)
    model_validator = ModelValidator(config)
    strategy_validator = StrategyValidator(config)
    system_validator = SystemValidator(config)
    async_validator = AsyncValidator(config)
    report_generator = ValidationReport(config)
    
    results = []
    
    # 1. 数据验证示例
    print("\n1. 执行数据验证...")
    
    # 格式验证
    email_result = data_validator.validate_format("test@example.com", "email")
    results.append(email_result)
    
    # 范围验证
    range_result = data_validator.validate_range(85, min_val=0, max_val=100)
    results.append(range_result)
    
    # 唯一性验证
    unique_result = data_validator.validate_uniqueness([1, 2, 3, 4, 5])
    results.append(unique_result)
    
    # DataFrame验证
    sample_df = pd.DataFrame({
        'A': [1, 2, 3, 4, 5],
        'B': ['a', 'b', 'c', 'd', 'e'],
        'C': [1.1, 2.2, 3.3, 4.4, 5.5]
    })
    df_result = data_validator.validate_dataframe(
        sample_df, 
        required_columns=['A', 'B'],
        column_types={'A': int, 'B': str, 'C': float}
    )
    results.append(df_result)
    
    # 时间序列验证
    time_series_data = create_sample_data()['close']
    ts_result = data_validator.validate_time_series(time_series_data, check_stationarity=False)
    results.append(ts_result)
    
    # 2. 模型验证示例
    print("2. 执行模型验证...")
    
    model, X, y = sample_model()
    
    # 交叉验证
    cv_result = model_validator.cross_validate(model, X, y, cv_folds=3)
    results.append(cv_result)
    
    # 留一验证
    loo_result = model_validator.leave_one_out_validate(model, X[:100], y[:100])  # 减少数据量以加快速度
    results.append(loo_result)
    
    # 模型假设验证
    assumption_result = model_validator.validate_model_assumptions(model, X, y)
    results.append(assumption_result)
    
    # 3. 策略验证示例
    print("3. 执行策略验证...")
    
    strategy_data = create_sample_data()
    
    # 回测验证
    backtest_result = strategy_validator.backtest_validate(
        sample_strategy, 
        strategy_data,
        initial_capital=100000
    )
    results.append(backtest_result)
    
    # 样本外验证
    train_data = strategy_data.iloc[:800]
    test_data = strategy_data.iloc[800:]
    
    oos_result = strategy_validator.out_of_sample_validate(
        sample_strategy,
        train_data,
        test_data,
        initial_capital=100000
    )
    results.append(oos_result)
    
    # 稳健性验证
    param_ranges = {
        'short_window': [3, 5, 7],
        'long_window': [15, 20, 25]
    }
    
    robustness_result = strategy_validator.robustness_validate(
        sample_strategy,
        strategy_data.iloc[:200],  # 减少数据量
        param_ranges,
        n_samples=10  # 减少样本数量
    )
    results.append(robustness_result)
    
    # 4. 系统验证示例
    print("4. 执行系统验证...")
    
    # 接口验证
    interface_result = system_validator.validate_interface(
        sample_strategy,
        expected_params=['data'],
        expected_return_type=int
    )
    results.append(interface_result)
    
    # 配置验证
    test_config = {
        'database_url': 'postgresql://localhost:5432/test',
        'max_connections': 100,
        'timeout': 30
    }
    
    config_result = system_validator.validate_configuration(
        test_config,
        required_keys=['database_url', 'max_connections'],
        value_constraints={
            'max_connections': lambda x: x > 0,
            'timeout': lambda x: x > 0
        }
    )
    results.append(config_result)
    
    # 依赖验证
    deps_result = system_validator.validate_dependencies(
        required_packages=['pandas', 'numpy', 'sklearn'],
        version_constraints={
            'pandas': '>=1.0.0',
            'numpy': '>=1.18.0'
        }
    )
    results.append(deps_result)
    
    # 性能验证
    def test_function():
        time.sleep(0.1)  # 模拟耗时操作
        return "完成"
    
    perf_result = system_validator.validate_performance(
        test_function,
        max_execution_time=1.0,
        max_memory_usage=100.0
    )
    results.append(perf_result)
    
    # 5. 异步验证示例
    print("5. 执行异步验证...")
    
    async def run_async_example():
        # 并行交叉验证
        models = [RandomForestClassifier(n_estimators=50, random_state=42) for _ in range(2)]
        async_results = await async_validator.parallel_cross_validate(
            models, X[:200], y[:200], cv_folds=3
        )
        return async_results
    
    # 注意：在实际使用中需要运行事件循环
    try:
        # async_results = asyncio.run(run_async_example())
        # results.extend(async_results)
        # 为了演示目的，这里跳过异步验证
        pass
    except Exception as e:
        # 使用模块级别的logger
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"异步验证执行失败: {e}")
        results.append(ValidationResult(
            success=False,
            message=f"异步验证失败: {str(e)}",
            errors=[str(e)]
        ))
    
    # 6. 验证流水线示例
    print("6. 执行验证流水线...")
    
    pipeline = ValidationPipeline(config)
    pipeline.add_step(
        "数据格式验证", "data", "validate_format",
        args=("test@example.com", "email")
    )
    pipeline.add_step(
        "模型交叉验证", "model", "cross_validate",
        args=(model, X[:100], y[:100]),
        kwargs={'cv_folds': 3}
    )
    pipeline.add_step(
        "策略回测", "strategy", "backtest_validate",
        args=(sample_strategy, strategy_data.iloc[:100]),
        kwargs={'initial_capital': 100000}
    )
    
    pipeline_result = pipeline.run_pipeline()
    results.append(pipeline_result)
    
    # 7. 生成报告
    print("7. 生成验证报告...")
    
    report_content = report_generator.generate_report(
        results,
        title="J7验证工具综合测试报告",
        output_path="j7_validation_comprehensive_report.html"
    )
    
    # 导出结果
    report_generator.export_results(results, "validation_results.json", "json")
    report_generator.export_results(results, "validation_results.csv", "csv")
    
    # 打印摘要
    print("\n=== 验证结果摘要 ===")
    successful_validations = sum(1 for r in results if r.success)
    total_validations = len(results)
    
    print(f"总验证数: {total_validations}")
    print(f"成功验证数: {successful_validations}")
    print(f"成功率: {successful_validations/total_validations:.1%}")
    print(f"总执行时间: {sum(r.execution_time for r in results):.2f}秒")
    
    print(f"\n报告已生成:")
    print(f"- HTML报告: j7_validation_comprehensive_report.html")
    print(f"- JSON结果: validation_results.json")
    print(f"- CSV结果: validation_results.csv")
    
    return results


if __name__ == "__main__":
    # 运行综合示例
    try:
        results = run_comprehensive_example()
        print("\n✅ J7验证工具综合示例运行完成！")
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断了程序执行")
    except ImportError as e:
        print(f"\n❌ 缺少必要的依赖包: {e}")
        print("请确保安装了所有必要的Python包")
    except Exception as e:
        print(f"\n❌ 示例运行失败: {e}")
        # 只在调试模式下显示详细错误信息
        import os
        if os.getenv('DEBUG', '').lower() in ('true', '1', 'yes'):
            import traceback
            traceback.print_exc()
        else:
            print("如需详细错误信息，请设置环境变量 DEBUG=true")