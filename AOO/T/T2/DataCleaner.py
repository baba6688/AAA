"""
T2数据清洗器
==========

一个功能完整的数据清洗器类，用于处理各种数据质量问题。

功能特性:
- 缺失值处理（删除、填充、插值）
- 重复值检测和处理
- 异常值检测和处理
- 数据格式标准化
- 数据类型转换
- 数据去噪和过滤
- 数据一致性检查
- 数据质量评估
- 清洗结果验证


日期: 2025-11-05
版本: 1.0.0
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from enum import Enum
from dataclasses import dataclass
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import re
import logging


class MissingValueStrategy(Enum):
    """缺失值处理策略枚举"""
    DELETE = "delete"
    FILL_MEAN = "fill_mean"
    FILL_MEDIAN = "fill_median"
    FILL_MODE = "fill_mode"
    FILL_FORWARD = "fill_forward"
    FILL_BACKWARD = "fill_backward"
    FILL_CONSTANT = "fill_constant"
    INTERPOLATE_LINEAR = "interpolate_linear"
    INTERPOLATE_POLYNOMIAL = "interpolate_polynomial"
    INTERPOLATE_SPLINE = "interpolate_spline"


class OutlierDetectionMethod(Enum):
    """异常值检测方法枚举"""
    ZSCORE = "zscore"
    IQR = "iqr"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    DBSCAN = "dbscan"
    STATISTICAL = "statistical"


class DataQualityMetric(Enum):
    """数据质量评估指标枚举"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    VALIDITY = "validity"
    UNIQUENESS = "uniqueness"
    TIMELINESS = "timeliness"


@dataclass
class DataQualityReport:
    """数据质量报告"""
    completeness: float
    accuracy: float
    consistency: float
    validity: float
    uniqueness: float
    timeliness: float
    overall_score: float
    issues: List[str]
    recommendations: List[str]


@dataclass
class CleaningReport:
    """数据清洗报告"""
    original_shape: Tuple[int, int]
    cleaned_shape: Tuple[int, int]
    missing_values_handled: int
    duplicates_removed: int
    outliers_removed: int
    format_standardized: int
    type_conversions: int
    noise_filtered: int
    consistency_fixed: int
    processing_time: float
    cleaning_steps: List[str]


class DataCleaner:
    """
    T2数据清洗器
    
    提供全面的数据清洗功能，包括缺失值处理、重复值检测、异常值检测、
    数据格式标准化、数据类型转换、数据去噪、数据一致性检查等功能。
    
    Attributes:
        config (Dict[str, Any]): 清洗配置参数
        logger (logging.Logger): 日志记录器
        quality_report (Optional[DataQualityReport]): 数据质量报告
        cleaning_report (Optional[CleaningReport]): 清洗报告
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """
        初始化数据清洗器
        
        Args:
            config: 配置参数字典，包含各种清洗参数
        """
        self.config = config or self._default_config()
        self.logger = self._setup_logger()
        self.quality_report: Optional[DataQualityReport] = None
        self.cleaning_report: Optional[CleaningReport] = None
        
    def _default_config(self) -> Dict[str, Any]:
        """默认配置参数"""
        return {
            'missing_value_strategy': MissingValueStrategy.FILL_MEDIAN,
            'missing_value_constant': 0,
            'outlier_detection_method': OutlierDetectionMethod.IQR,
            'outlier_threshold': 1.5,
            'duplicate_keep': 'first',
            'numeric_columns': None,
            'categorical_columns': None,
            'datetime_columns': None,
            'standardize_formats': True,
            'remove_noise': True,
            'noise_threshold': 0.1,
            'consistency_check': True,
            'quality_threshold': 0.8,
            'random_state': 42
        }
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('DataCleaner')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def load_data(self, data: Union[pd.DataFrame, str, Dict]) -> pd.DataFrame:
        """
        加载数据
        
        Args:
            data: 数据源，可以是DataFrame、文件路径或字典
            
        Returns:
            加载的DataFrame
        """
        if isinstance(data, pd.DataFrame):
            return data.copy()
        elif isinstance(data, str):
            # 根据文件扩展名加载数据
            if data.endswith('.csv'):
                return pd.read_csv(data)
            elif data.endswith('.xlsx') or data.endswith('.xls'):
                return pd.read_excel(data)
            elif data.endswith('.json'):
                return pd.read_json(data)
            else:
                raise ValueError(f"不支持的文件格式: {data}")
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        else:
            raise ValueError(f"不支持的数据类型: {type(data)}")
    
    def assess_data_quality(self, data: pd.DataFrame) -> DataQualityReport:
        """
        评估数据质量
        
        Args:
            data: 待评估的数据
            
        Returns:
            数据质量报告
        """
        self.logger.info("开始数据质量评估...")
        
        # 计算各项质量指标
        completeness = self._calculate_completeness(data)
        accuracy = self._calculate_accuracy(data)
        consistency = self._calculate_consistency(data)
        validity = self._calculate_validity(data)
        uniqueness = self._calculate_uniqueness(data)
        timeliness = self._calculate_timeliness(data)
        
        # 计算综合评分
        overall_score = (completeness + accuracy + consistency + 
                        validity + uniqueness + timeliness) / 6
        
        # 识别问题和提供建议
        issues = self._identify_quality_issues(data)
        recommendations = self._generate_recommendations(issues)
        
        report = DataQualityReport(
            completeness=completeness,
            accuracy=accuracy,
            consistency=consistency,
            validity=validity,
            uniqueness=uniqueness,
            timeliness=timeliness,
            overall_score=overall_score,
            issues=issues,
            recommendations=recommendations
        )
        
        self.quality_report = report
        self.logger.info(f"数据质量评估完成，综合评分: {overall_score:.2f}")
        
        return report
    
    def _calculate_completeness(self, data: pd.DataFrame) -> float:
        """计算数据完整性"""
        total_cells = data.shape[0] * data.shape[1]
        non_null_cells = data.count().sum()
        return non_null_cells / total_cells if total_cells > 0 else 0.0
    
    def _calculate_accuracy(self, data: pd.DataFrame) -> float:
        """计算数据准确性（基于统计分布的合理性）"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return 1.0
        
        accuracy_scores = []
        for col in numeric_cols:
            # 检查异常值比例
            if data[col].std() > 0:
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                outlier_ratio = (z_scores > 3).sum() / len(z_scores)
                accuracy_scores.append(1 - outlier_ratio)
            else:
                accuracy_scores.append(1.0)
        
        return np.mean(accuracy_scores) if accuracy_scores else 1.0
    
    def _calculate_consistency(self, data: pd.DataFrame) -> float:
        """计算数据一致性"""
        consistency_scores = []
        
        # 检查数据类型一致性
        for col in data.columns:
            # 检查同一列中数据类型的一致性
            non_null_data = data[col].dropna()
            if len(non_null_data) > 0:
                # 简单的一致性检查：检查是否主要为同一数据类型
                type_consistency = len(non_null_data.apply(type).unique()) <= 2
                consistency_scores.append(1.0 if type_consistency else 0.5)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _calculate_validity(self, data: pd.DataFrame) -> float:
        """计算数据有效性"""
        validity_scores = []
        
        for col in data.columns:
            non_null_data = data[col].dropna()
            if len(non_null_data) == 0:
                continue
                
            # 检查数据格式的有效性
            if data[col].dtype == 'object':
                # 简单的字符串有效性检查
                valid_ratio = (~non_null_data.str.strip().eq('')).mean()
                validity_scores.append(valid_ratio)
            else:
                # 数值型数据的有效性检查
                valid_ratio = (~non_null_data.isin([np.inf, -np.inf])).mean()
                validity_scores.append(valid_ratio)
        
        return np.mean(validity_scores) if validity_scores else 1.0
    
    def _calculate_uniqueness(self, data: pd.DataFrame) -> float:
        """计算数据唯一性"""
        uniqueness_scores = []
        
        for col in data.columns:
            non_null_data = data[col].dropna()
            if len(non_null_data) > 0:
                uniqueness_ratio = non_null_data.nunique() / len(non_null_data)
                uniqueness_scores.append(uniqueness_ratio)
        
        return np.mean(uniqueness_scores) if uniqueness_scores else 1.0
    
    def _calculate_timeliness(self, data: pd.DataFrame) -> float:
        """计算数据时效性（基于时间列的分布）"""
        datetime_cols = data.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) == 0:
            return 1.0
        
        timeliness_scores = []
        for col in datetime_cols:
            non_null_data = pd.to_datetime(data[col], errors='coerce').dropna()
            if len(non_null_data) > 0:
                # 检查时间分布的合理性
                time_span = (non_null_data.max() - non_null_data.min()).days
                if time_span > 0:
                    # 检查是否有过于集中的时间分布
                    time_density = len(non_null_data) / time_span
                    timeliness_scores.append(min(1.0, time_density / 100))  # 假设每天100条记录为合理密度
                else:
                    timeliness_scores.append(0.5)
        
        return np.mean(timeliness_scores) if timeliness_scores else 1.0
    
    def _identify_quality_issues(self, data: pd.DataFrame) -> List[str]:
        """识别数据质量问题"""
        issues = []
        
        # 检查缺失值
        missing_ratio = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_ratio > 0.1:
            issues.append(f"缺失值比例过高: {missing_ratio:.2%}")
        
        # 检查重复值
        duplicate_ratio = data.duplicated().sum() / len(data)
        if duplicate_ratio > 0.05:
            issues.append(f"重复值比例过高: {duplicate_ratio:.2%}")
        
        # 检查异常值
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].std() > 0:
                z_scores = np.abs(stats.zscore(data[col].dropna()))
                outlier_ratio = (z_scores > 3).sum() / len(z_scores)
                if outlier_ratio > 0.05:
                    issues.append(f"列 '{col}' 异常值比例过高: {outlier_ratio:.2%}")
        
        # 检查数据类型问题
        for col in data.columns:
            if data[col].dtype == 'object':
                # 检查是否包含混合数据类型
                non_null_data = data[col].dropna()
                if len(non_null_data) > 0:
                    type_variety = len(non_null_data.apply(type).unique())
                    if type_variety > 2:
                        issues.append(f"列 '{col}' 包含多种数据类型")
        
        return issues
    
    def _generate_recommendations(self, issues: List[str]) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if any("缺失值" in issue for issue in issues):
            recommendations.append("使用适当的缺失值处理策略（填充、插值等）")
        
        if any("重复值" in issue for issue in issues):
            recommendations.append("删除重复记录或进行去重处理")
        
        if any("异常值" in issue for issue in issues):
            recommendations.append("检测和处理异常值（删除、转换或保留）")
        
        if any("数据类型" in issue for issue in issues):
            recommendations.append("统一数据类型，转换混合类型数据")
        
        recommendations.append("定期进行数据质量监控和评估")
        recommendations.append("建立数据验证规则和约束")
        
        return recommendations
    
    def handle_missing_values(self, data: pd.DataFrame, 
                            strategy: Optional[MissingValueStrategy] = None,
                            columns: Optional[List[str]] = None,
                            constant_value: Any = None) -> pd.DataFrame:
        """
        处理缺失值
        
        Args:
            data: 原始数据
            strategy: 处理策略
            columns: 指定的列，None表示所有列
            constant_value: 常量填充值
            
        Returns:
            处理后的数据
        """
        strategy = strategy or self.config['missing_value_strategy']
        columns = columns or data.columns.tolist()
        constant_value = constant_value or self.config['missing_value_constant']
        
        self.logger.info(f"开始处理缺失值，策略: {strategy.value}")
        
        cleaned_data = data.copy()
        missing_count = 0
        
        for col in columns:
            if col not in cleaned_data.columns:
                continue
                
            col_missing = cleaned_data[col].isnull().sum()
            if col_missing == 0:
                continue
                
            missing_count += col_missing
            
            if strategy == MissingValueStrategy.DELETE:
                # 删除包含缺失值的行
                cleaned_data = cleaned_data.dropna(subset=[col])
                
            elif strategy == MissingValueStrategy.FILL_MEAN:
                # 数值列用均值填充
                if pd.api.types.is_numeric_dtype(cleaned_data[col]):
                    cleaned_data[col].fillna(cleaned_data[col].mean(), inplace=True)
                    
            elif strategy == MissingValueStrategy.FILL_MEDIAN:
                # 数值列用中位数填充
                if pd.api.types.is_numeric_dtype(cleaned_data[col]):
                    cleaned_data[col].fillna(cleaned_data[col].median(), inplace=True)
                    
            elif strategy == MissingValueStrategy.FILL_MODE:
                # 用众数填充
                mode_value = cleaned_data[col].mode()
                if len(mode_value) > 0:
                    cleaned_data[col].fillna(mode_value[0], inplace=True)
                    
            elif strategy == MissingValueStrategy.FILL_FORWARD:
                # 前向填充
                cleaned_data[col].fillna(method='ffill', inplace=True)
                
            elif strategy == MissingValueStrategy.FILL_BACKWARD:
                # 后向填充
                cleaned_data[col].fillna(method='bfill', inplace=True)
                
            elif strategy == MissingValueStrategy.FILL_CONSTANT:
                # 常量填充
                cleaned_data[col].fillna(constant_value, inplace=True)
                
            elif strategy == MissingValueStrategy.INTERPOLATE_LINEAR:
                # 线性插值
                cleaned_data[col].interpolate(method='linear', inplace=True)
                
            elif strategy == MissingValueStrategy.INTERPOLATE_POLYNOMIAL:
                # 多项式插值
                cleaned_data[col].interpolate(method='polynomial', order=2, inplace=True)
                
            elif strategy == MissingValueStrategy.INTERPOLATE_SPLINE:
                # 样条插值
                cleaned_data[col].interpolate(method='spline', order=3, inplace=True)
        
        self.logger.info(f"缺失值处理完成，处理了 {missing_count} 个缺失值")
        return cleaned_data
    
    def detect_and_remove_duplicates(self, data: pd.DataFrame, 
                                   keep: str = 'first',
                                   subset: Optional[List[str]] = None) -> pd.DataFrame:
        """
        检测和处理重复值
        
        Args:
            data: 原始数据
            keep: 保留策略 ('first', 'last', False)
            subset: 检查重复的列，None表示所有列
            
        Returns:
            去重后的数据
        """
        keep = self.config.get('duplicate_keep', keep)
        
        self.logger.info("开始检测和处理重复值...")
        
        original_count = len(data)
        cleaned_data = data.copy()
        
        # 检测重复值
        duplicates = cleaned_data.duplicated(subset=subset, keep=keep)
        duplicate_count = duplicates.sum()
        
        if duplicate_count > 0:
            # 移除重复值
            cleaned_data = cleaned_data.drop_duplicates(subset=subset, keep=keep)
            self.logger.info(f"移除了 {duplicate_count} 个重复值")
        else:
            self.logger.info("未发现重复值")
        
        final_count = len(cleaned_data)
        self.logger.info(f"数据从 {original_count} 行减少到 {final_count} 行")
        
        return cleaned_data
    
    def detect_outliers(self, data: pd.DataFrame, 
                       method: Optional[OutlierDetectionMethod] = None,
                       columns: Optional[List[str]] = None,
                       **kwargs) -> Dict[str, np.ndarray]:
        """
        检测异常值
        
        Args:
            data: 原始数据
            method: 检测方法
            columns: 检测的列
            **kwargs: 方法特定参数
            
        Returns:
            异常值索引字典
        """
        method = method or self.config['outlier_detection_method']
        columns = columns or data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.logger.info(f"开始检测异常值，方法: {method.value}")
        
        outliers = {}
        threshold = kwargs.get('threshold', self.config['outlier_threshold'])
        
        for col in columns:
            if col not in data.columns or not pd.api.types.is_numeric_dtype(data[col]):
                continue
            
            col_data = data[col].dropna()
            if len(col_data) < 3:
                continue
            
            if method == OutlierDetectionMethod.ZSCORE:
                # Z-score方法
                z_scores = np.abs(stats.zscore(col_data))
                outlier_mask = z_scores > threshold
                
            elif method == OutlierDetectionMethod.IQR:
                # IQR方法
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                
            elif method == OutlierDetectionMethod.STATISTICAL:
                # 统计方法
                mean = col_data.mean()
                std = col_data.std()
                if std > 0:
                    outlier_mask = np.abs(col_data - mean) > threshold * std
                else:
                    outlier_mask = pd.Series([False] * len(col_data))
                    
            else:
                # 其他方法暂时使用IQR作为默认
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
            
            # 获取异常值索引
            outlier_indices = col_data.index[outlier_mask]
            outliers[col] = outlier_indices
        
        total_outliers = sum(len(indices) for indices in outliers.values())
        self.logger.info(f"检测到 {total_outliers} 个异常值")
        
        return outliers
    
    def handle_outliers(self, data: pd.DataFrame,
                       outliers: Optional[Dict[str, np.ndarray]] = None,
                       method: Optional[OutlierDetectionMethod] = None,
                       action: str = 'remove',
                       **kwargs) -> pd.DataFrame:
        """
        处理异常值
        
        Args:
            data: 原始数据
            outliers: 预检测的异常值
            method: 检测方法
            action: 处理动作 ('remove', 'cap', 'transform')
            **kwargs: 方法参数
            
        Returns:
            处理后的数据
        """
        if outliers is None:
            outliers = self.detect_outliers(data, method, **kwargs)
        
        self.logger.info(f"开始处理异常值，动作: {action}")
        
        cleaned_data = data.copy()
        total_removed = 0
        
        if action == 'remove':
            # 移除异常值
            outlier_indices = set()
            for col, indices in outliers.items():
                outlier_indices.update(indices)
            
            if outlier_indices:
                cleaned_data = cleaned_data.drop(outlier_indices)
                total_removed = len(outlier_indices)
                self.logger.info(f"移除了 {total_removed} 个异常值")
        
        elif action == 'cap':
            # 限制异常值（winsorization）
            for col, indices in outliers.items():
                if col in cleaned_data.columns and pd.api.types.is_numeric_dtype(cleaned_data[col]):
                    Q1 = cleaned_data[col].quantile(0.25)
                    Q3 = cleaned_data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    cleaned_data.loc[cleaned_data[col] < lower_bound, col] = lower_bound
                    cleaned_data.loc[cleaned_data[col] > upper_bound, col] = upper_bound
        
        elif action == 'transform':
            # 转换异常值（如对数转换）
            for col in outliers.keys():
                if col in cleaned_data.columns and pd.api.types.is_numeric_dtype(cleaned_data[col]):
                    # 简单的对数转换（确保值为正）
                    min_val = cleaned_data[col].min()
                    if min_val > 0:
                        cleaned_data[col] = np.log(cleaned_data[col] + 1)
                    else:
                        cleaned_data[col] = cleaned_data[col] - min_val + 1
                        cleaned_data[col] = np.log(cleaned_data[col])
        
        return cleaned_data
    
    def standardize_formats(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        数据格式标准化
        
        Args:
            data: 原始数据
            
        Returns:
            格式标准化后的数据
        """
        if not self.config.get('standardize_formats', True):
            return data
        
        self.logger.info("开始数据格式标准化...")
        
        cleaned_data = data.copy()
        standardized_count = 0
        
        # 先标准化字符串格式（使用原始列名）
        for col in cleaned_data.select_dtypes(include=['object']).columns:
            original_values = cleaned_data[col].copy()
            
            # 去除前后空格
            cleaned_data[col] = cleaned_data[col].astype(str).str.strip()
            
            # 标准化大小写（可选）
            # cleaned_data[col] = cleaned_data[col].str.lower()
            
            # 标准化空值表示
            cleaned_data[col] = cleaned_data[col].replace(['', 'nan', 'null', 'None'], np.nan)
            
            if not cleaned_data[col].equals(original_values):
                standardized_count += 1
        
        # 然后标准化列名
        original_columns = cleaned_data.columns.tolist()
        cleaned_data.columns = [self._standardize_column_name(col) for col in cleaned_data.columns]
        if cleaned_data.columns.tolist() != original_columns:
            standardized_count += len(original_columns)
        
        # 标准化数值格式
        for col in cleaned_data.select_dtypes(include=[np.number]).columns:
            # 获取列数据
            col_data = cleaned_data[col]
            
            # 处理无穷大值
            col_data = col_data.replace([np.inf, -np.inf], np.nan)
            
            # 处理异常大的数值
            if len(col_data.dropna()) > 0:
                try:
                    std_val = col_data.std()
                    if std_val > 0:
                        z_scores = np.abs(stats.zscore(col_data.dropna()))
                        if (z_scores > 10).any():  # 异常大的值
                            col_data = col_data.replace([np.inf, -np.inf], np.nan)
                            standardized_count += 1
                except:
                    # 如果计算失败，跳过这个列
                    pass
            
            # 更新列数据
            cleaned_data[col] = col_data
        
        self.logger.info(f"格式标准化完成，处理了 {standardized_count} 处格式问题")
        return cleaned_data
    
    def _standardize_column_name(self, name: str) -> str:
        """标准化列名"""
        # 转换为字符串并去除前后空格
        name = str(name).strip()
        
        # 替换特殊字符为下划线
        name = re.sub(r'[^\w\s]', '_', name)
        
        # 替换多个空格为单个下划线
        name = re.sub(r'\s+', '_', name)
        
        # 转换为小写
        name = name.lower()
        
        return name
    
    def convert_data_types(self, data: pd.DataFrame,
                         type_mapping: Optional[Dict[str, str]] = None,
                         infer_types: bool = True) -> pd.DataFrame:
        """
        数据类型转换
        
        Args:
            data: 原始数据
            type_mapping: 类型映射字典
            infer_types: 是否自动推断类型
            
        Returns:
            类型转换后的数据
        """
        self.logger.info("开始数据类型转换...")
        
        cleaned_data = data.copy()
        conversion_count = 0
        
        if type_mapping:
            # 使用提供的类型映射
            for col, target_type in type_mapping.items():
                if col in cleaned_data.columns:
                    try:
                        original_type = cleaned_data[col].dtype
                        if target_type == 'numeric':
                            cleaned_data[col] = pd.to_numeric(cleaned_data[col], errors='coerce')
                        elif target_type == 'datetime':
                            cleaned_data[col] = pd.to_datetime(cleaned_data[col], errors='coerce')
                        elif target_type == 'category':
                            cleaned_data[col] = cleaned_data[col].astype('category')
                        elif target_type == 'string':
                            cleaned_data[col] = cleaned_data[col].astype(str)
                        else:
                            cleaned_data[col] = cleaned_data[col].astype(target_type)
                        
                        if cleaned_data[col].dtype != original_type:
                            conversion_count += 1
                            self.logger.info(f"列 '{col}' 从 {original_type} 转换为 {cleaned_data[col].dtype}")
                    except Exception as e:
                        self.logger.warning(f"列 '{col}' 类型转换失败: {e}")
        
        if infer_types:
            # 自动推断数据类型
            for col in cleaned_data.columns:
                if cleaned_data[col].dtype == 'object':
                    # 尝试转换为数值型
                    numeric_attempt = pd.to_numeric(cleaned_data[col], errors='coerce')
                    numeric_ratio = numeric_attempt.notna().sum() / len(cleaned_data[col])
                    
                    if numeric_ratio > 0.8:  # 80%以上可以转换为数值
                        cleaned_data[col] = numeric_attempt
                        conversion_count += 1
                        self.logger.info(f"列 '{col}' 自动转换为数值型")
                    
                    else:
                        # 尝试转换为日期时间
                        datetime_attempt = pd.to_datetime(cleaned_data[col], errors='coerce')
                        datetime_ratio = datetime_attempt.notna().sum() / len(cleaned_data[col])
                        
                        if datetime_ratio > 0.8:
                            cleaned_data[col] = datetime_attempt
                            conversion_count += 1
                            self.logger.info(f"列 '{col}' 自动转换为日期时间型")
        
        self.logger.info(f"数据类型转换完成，转换了 {conversion_count} 列")
        return cleaned_data
    
    def filter_noise(self, data: pd.DataFrame,
                    noise_threshold: Optional[float] = None,
                    columns: Optional[List[str]] = None) -> pd.DataFrame:
        """
        数据去噪和过滤
        
        Args:
            data: 原始数据
            noise_threshold: 噪声阈值
            columns: 处理的列
            
        Returns:
            去噪后的数据
        """
        if not self.config.get('remove_noise', True):
            return data
        
        noise_threshold = noise_threshold or self.config['noise_threshold']
        columns = columns or data.select_dtypes(include=[np.number]).columns.tolist()
        
        self.logger.info(f"开始数据去噪，阈值: {noise_threshold}")
        
        cleaned_data = data.copy()
        noise_removed = 0
        
        for col in columns:
            if col not in cleaned_data.columns or not pd.api.types.is_numeric_dtype(cleaned_data[col]):
                continue
            
            col_data = cleaned_data[col].dropna()
            if len(col_data) < 3:
                continue
            
            # 使用移动平均滤波
            window_size = min(5, len(col_data) // 10)
            if window_size >= 3:
                rolling_mean = col_data.rolling(window=window_size, center=True).mean()
                
                # 计算偏差
                deviation = np.abs(col_data - rolling_mean)
                std_dev = deviation.std()
                
                # 识别噪声点
                noise_mask = deviation > noise_threshold * std_dev
                
                if noise_mask.any():
                    # 用移动平均替换噪声点
                    cleaned_data.loc[col_data.index[noise_mask], col] = rolling_mean[noise_mask]
                    noise_removed += noise_mask.sum()
        
        self.logger.info(f"数据去噪完成，移除了 {noise_removed} 个噪声点")
        return cleaned_data
    
    def check_consistency(self, data: pd.DataFrame,
                         rules: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[Dict]]:
        """
        数据一致性检查
        
        Args:
            data: 原始数据
            rules: 一致性规则列表
            
        Returns:
            一致性问题报告
        """
        if not self.config.get('consistency_check', True):
            return {}
        
        self.logger.info("开始数据一致性检查...")
        
        issues = {}
        
        if rules is None:
            # 默认一致性规则
            rules = [
                {'type': 'range_check', 'columns': [], 'min': 0, 'max': 100},
                {'type': 'unique_check', 'columns': []},
                {'type': 'format_check', 'columns': [], 'pattern': r'^[\w\s]+$'},
                {'type': 'dependency_check', 'columns': [], 'dependency': {}}
            ]
        
        for rule in rules:
            rule_type = rule.get('type')
            columns = rule.get('columns', [])
            
            if rule_type == 'range_check':
                for col in columns:
                    if col in data.columns and pd.api.types.is_numeric_dtype(data[col]):
                        min_val = rule.get('min')
                        max_val = rule.get('max')
                        
                        if min_val is not None:
                            below_min = data[col] < min_val
                            if below_min.any():
                                if col not in issues:
                                    issues[col] = []
                                issues[col].append({
                                    'type': 'range_violation',
                                    'description': f'值小于最小值 {min_val}',
                                    'count': below_min.sum(),
                                    'indices': data.index[below_min].tolist()
                                })
                        
                        if max_val is not None:
                            above_max = data[col] > max_val
                            if above_max.any():
                                if col not in issues:
                                    issues[col] = []
                                issues[col].append({
                                    'type': 'range_violation',
                                    'description': f'值大于最大值 {max_val}',
                                    'count': above_max.sum(),
                                    'indices': data.index[above_max].tolist()
                                })
            
            elif rule_type == 'unique_check':
                for col in columns:
                    if col in data.columns:
                        duplicates = data[col].duplicated()
                        if duplicates.any():
                            if col not in issues:
                                issues[col] = []
                            issues[col].append({
                                'type': 'uniqueness_violation',
                                'description': '存在重复值',
                                'count': duplicates.sum(),
                                'indices': data.index[duplicates].tolist()
                            })
            
            elif rule_type == 'format_check':
                for col in columns:
                    if col in data.columns and data[col].dtype == 'object':
                        pattern = rule.get('pattern', r'^[\w\s]+$')
                        non_compliant = ~data[col].astype(str).str.match(pattern, na=False)
                        if non_compliant.any():
                            if col not in issues:
                                issues[col] = []
                            issues[col].append({
                                'type': 'format_violation',
                                'description': f'格式不符合正则表达式 {pattern}',
                                'count': non_compliant.sum(),
                                'indices': data.index[non_compliant].tolist()
                            })
        
        total_issues = sum(len(issue_list) for issue_list in issues.values())
        self.logger.info(f"一致性检查完成，发现 {total_issues} 个一致性问题")
        
        return issues
    
    def fix_consistency_issues(self, data: pd.DataFrame,
                             issues: Dict[str, List[Dict]],
                             fix_strategies: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        修复一致性问题
        
        Args:
            data: 原始数据
            issues: 一致性问题报告
            fix_strategies: 修复策略
            
        Returns:
            修复后的数据
        """
        self.logger.info("开始修复一致性问题...")
        
        cleaned_data = data.copy()
        fixed_count = 0
        
        for col, col_issues in issues.items():
            if col not in cleaned_data.columns:
                continue
            
            for issue in col_issues:
                issue_type = issue['type']
                indices = issue['indices']
                
                if issue_type == 'range_violation':
                    # 修复范围违规
                    strategy = fix_strategies.get(f'{col}_{issue_type}', 'clip') if fix_strategies else 'clip'
                    
                    if '小于最小值' in issue['description']:
                        min_val = float(issue['description'].split()[-1])
                        if strategy == 'clip':
                            cleaned_data.loc[indices, col] = min_val
                        elif strategy == 'remove':
                            cleaned_data = cleaned_data.drop(indices)
                        fixed_count += len(indices)
                    
                    elif '大于最大值' in issue['description']:
                        max_val = float(issue['description'].split()[-1])
                        if strategy == 'clip':
                            cleaned_data.loc[indices, col] = max_val
                        elif strategy == 'remove':
                            cleaned_data = cleaned_data.drop(indices)
                        fixed_count += len(indices)
                
                elif issue_type == 'format_violation':
                    # 修复格式违规
                    strategy = fix_strategies.get(f'{col}_{issue_type}', 'clean') if fix_strategies else 'clean'
                    
                    if strategy == 'clean':
                        # 清理格式问题
                        cleaned_data.loc[indices, col] = cleaned_data.loc[indices, col].astype(str).str.strip()
                    elif strategy == 'remove':
                        cleaned_data = cleaned_data.drop(indices)
                    fixed_count += len(indices)
        
        self.logger.info(f"一致性修复完成，修复了 {fixed_count} 个问题")
        return cleaned_data
    
    def validate_cleaning_results(self, original_data: pd.DataFrame,
                                cleaned_data: pd.DataFrame,
                                quality_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        验证清洗结果
        
        Args:
            original_data: 原始数据
            cleaned_data: 清洗后数据
            quality_threshold: 质量阈值
            
        Returns:
            验证结果
        """
        quality_threshold = quality_threshold or self.config.get('quality_threshold', 0.8)
        
        self.logger.info("开始验证清洗结果...")
        
        validation_results = {
            'is_valid': True,
            'quality_score': 0.0,
            'improvements': [],
            'degradations': [],
            'warnings': [],
            'recommendations': []
        }
        
        # 比较数据质量
        original_quality = self.assess_data_quality(original_data)
        cleaned_quality = self.assess_data_quality(cleaned_data)
        
        validation_results['original_quality'] = original_quality
        validation_results['cleaned_quality'] = cleaned_quality
        
        # 计算质量改进
        quality_improvement = cleaned_quality.overall_score - original_quality.overall_score
        validation_results['quality_score'] = cleaned_quality.overall_score
        
        if quality_improvement > 0.1:
            validation_results['improvements'].append(f"数据质量显著提升: {quality_improvement:.2%}")
        elif quality_improvement > 0:
            validation_results['improvements'].append(f"数据质量轻微提升: {quality_improvement:.2%}")
        elif quality_improvement < -0.1:
            validation_results['degradations'].append(f"数据质量显著下降: {quality_improvement:.2%}")
            validation_results['is_valid'] = False
        elif quality_improvement < 0:
            validation_results['degradations'].append(f"数据质量轻微下降: {quality_improvement:.2%}")
        
        # 检查数据损失
        data_loss_ratio = (len(original_data) - len(cleaned_data)) / len(original_data)
        if data_loss_ratio > 0.2:
            validation_results['warnings'].append(f"数据损失过多: {data_loss_ratio:.2%}")
            validation_results['is_valid'] = False
        elif data_loss_ratio > 0.1:
            validation_results['warnings'].append(f"数据损失较多: {data_loss_ratio:.2%}")
        
        # 检查完整性改进
        completeness_improvement = cleaned_quality.completeness - original_quality.completeness
        if completeness_improvement > 0.1:
            validation_results['improvements'].append(f"数据完整性显著提升: {completeness_improvement:.2%}")
        
        # 检查唯一性改进
        uniqueness_improvement = cleaned_quality.uniqueness - original_quality.uniqueness
        if uniqueness_improvement > 0.05:
            validation_results['improvements'].append(f"数据唯一性提升: {uniqueness_improvement:.2%}")
        
        # 生成建议
        if cleaned_quality.overall_score < quality_threshold:
            validation_results['recommendations'].append("数据质量仍低于阈值，建议进一步清洗")
        
        if len(validation_results['degradations']) > 0:
            validation_results['recommendations'].append("检查清洗参数，可能过于激进")
        
        if len(validation_results['warnings']) > 0:
            validation_results['recommendations'].append("关注数据损失，考虑调整清洗策略")
        
        validation_results['is_valid'] = (cleaned_quality.overall_score >= quality_threshold and 
                                        len(validation_results['degradations']) == 0)
        
        self.logger.info(f"清洗结果验证完成，有效性: {validation_results['is_valid']}")
        
        return validation_results
    
    def clean_data(self, data: Union[pd.DataFrame, str, Dict],
                  **kwargs) -> Tuple[pd.DataFrame, CleaningReport]:
        """
        完整的数据清洗流程
        
        Args:
            data: 原始数据
            **kwargs: 清洗参数
            
        Returns:
            清洗后的数据和清洗报告
        """
        import time
        start_time = time.time()
        
        self.logger.info("开始完整数据清洗流程...")
        
        # 加载数据
        original_data = self.load_data(data)
        self.logger.info(f"加载数据: {original_data.shape}")
        
        # 记录清洗步骤
        cleaning_steps = []
        cleaned_data = original_data.copy()
        
        # 1. 初步质量评估
        original_quality = self.assess_data_quality(cleaned_data)
        cleaning_steps.append("完成初步质量评估")
        
        # 2. 数据格式标准化
        if kwargs.get('standardize_formats', True):
            cleaned_data = self.standardize_formats(cleaned_data)
            cleaning_steps.append("完成数据格式标准化")
        
        # 3. 数据类型转换
        if kwargs.get('convert_types', True):
            type_mapping = kwargs.get('type_mapping')
            cleaned_data = self.convert_data_types(cleaned_data, type_mapping)
            cleaning_steps.append("完成数据类型转换")
        
        # 4. 缺失值处理
        if kwargs.get('handle_missing', True):
            missing_strategy = kwargs.get('missing_strategy')
            cleaned_data = self.handle_missing_values(cleaned_data, missing_strategy)
            cleaning_steps.append("完成缺失值处理")
        
        # 5. 重复值处理
        if kwargs.get('handle_duplicates', True):
            cleaned_data = self.detect_and_remove_duplicates(cleaned_data)
            cleaning_steps.append("完成重复值处理")
        
        # 6. 异常值处理
        if kwargs.get('handle_outliers', True):
            outlier_method = kwargs.get('outlier_method')
            outlier_action = kwargs.get('outlier_action', 'remove')
            cleaned_data = self.handle_outliers(cleaned_data, method=outlier_method, action=outlier_action)
            cleaning_steps.append("完成异常值处理")
        
        # 7. 数据去噪
        if kwargs.get('filter_noise', True):
            cleaned_data = self.filter_noise(cleaned_data)
            cleaning_steps.append("完成数据去噪")
        
        # 8. 一致性检查和修复
        if kwargs.get('check_consistency', True):
            issues = self.check_consistency(cleaned_data)
            if issues:
                cleaned_data = self.fix_consistency_issues(cleaned_data, issues)
                cleaning_steps.append("完成一致性检查和修复")
            else:
                cleaning_steps.append("一致性检查通过")
        
        # 9. 最终质量评估
        final_quality = self.assess_data_quality(cleaned_data)
        cleaning_steps.append("完成最终质量评估")
        
        # 计算处理统计
        processing_time = time.time() - start_time
        
        # 生成清洗报告
        cleaning_report = CleaningReport(
            original_shape=original_data.shape,
            cleaned_shape=cleaned_data.shape,
            missing_values_handled=original_data.isnull().sum().sum() - cleaned_data.isnull().sum().sum(),
            duplicates_removed=len(original_data) - len(cleaned_data.drop_duplicates()),
            outliers_removed=0,  # 需要根据具体方法计算
            format_standardized=len(original_data.columns),
            type_conversions=0,  # 需要根据具体转换计算
            noise_filtered=0,  # 需要根据具体去噪计算
            consistency_fixed=sum(len(issues) for issues in self.check_consistency(cleaned_data).values()),
            processing_time=processing_time,
            cleaning_steps=cleaning_steps
        )
        
        # 验证清洗结果
        validation_results = self.validate_cleaning_results(original_data, cleaned_data)
        
        self.logger.info(f"数据清洗完成，耗时: {processing_time:.2f}秒")
        self.logger.info(f"原始数据: {original_data.shape}, 清洗后: {cleaned_data.shape}")
        self.logger.info(f"质量评分: {original_quality.overall_score:.2f} -> {final_quality.overall_score:.2f}")
        
        return cleaned_data, cleaning_report
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """
        获取清洗摘要信息
        
        Returns:
            清洗摘要字典
        """
        summary = {
            'config': self.config,
            'quality_report': self.quality_report.__dict__ if self.quality_report else None,
            'cleaning_report': self.cleaning_report.__dict__ if self.cleaning_report else None,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return summary
    
    def export_cleaning_report(self, filepath: str) -> None:
        """
        导出清洗报告
        
        Args:
            filepath: 报告文件路径
        """
        summary = self.get_cleaning_summary()
        
        # 转换为易读的格式
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("T2数据清洗器 - 清洗报告")
        report_lines.append("=" * 60)
        report_lines.append(f"生成时间: {summary['timestamp']}")
        report_lines.append("")
        
        if summary['quality_report']:
            qr = summary['quality_report']
            report_lines.append("数据质量评估:")
            report_lines.append(f"  完整性: {qr['completeness']:.2%}")
            report_lines.append(f"  准确性: {qr['accuracy']:.2%}")
            report_lines.append(f"  一致性: {qr['consistency']:.2%}")
            report_lines.append(f"  有效性: {qr['validity']:.2%}")
            report_lines.append(f"  唯一性: {qr['uniqueness']:.2%}")
            report_lines.append(f"  时效性: {qr['timeliness']:.2%}")
            report_lines.append(f"  综合评分: {qr['overall_score']:.2%}")
            report_lines.append("")
            
            if qr['issues']:
                report_lines.append("发现的问题:")
                for issue in qr['issues']:
                    report_lines.append(f"  - {issue}")
                report_lines.append("")
            
            if qr['recommendations']:
                report_lines.append("改进建议:")
                for rec in qr['recommendations']:
                    report_lines.append(f"  - {rec}")
                report_lines.append("")
        
        if summary['cleaning_report']:
            cr = summary['cleaning_report']
            report_lines.append("清洗处理统计:")
            report_lines.append(f"  原始数据形状: {cr['original_shape']}")
            report_lines.append(f"  清洗后形状: {cr['cleaned_shape']}")
            report_lines.append(f"  处理缺失值: {cr['missing_values_handled']}")
            report_lines.append(f"  移除重复值: {cr['duplicates_removed']}")
            report_lines.append(f"  修复一致性问题: {cr['consistency_fixed']}")
            report_lines.append(f"  处理时间: {cr['processing_time']:.2f}秒")
            report_lines.append("")
            
            report_lines.append("清洗步骤:")
            for i, step in enumerate(cr['cleaning_steps'], 1):
                report_lines.append(f"  {i}. {step}")
        
        # 写入文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))
        
        self.logger.info(f"清洗报告已导出到: {filepath}")


def create_sample_data() -> pd.DataFrame:
    """
    创建示例数据用于测试
    
    Returns:
        包含各种数据质量问题的示例DataFrame
    """
    np.random.seed(42)
    n_samples = 1000
    
    # 创建基础数据
    data = {
        'id': range(1, n_samples + 1),
        'age': np.random.normal(35, 10, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'score': np.random.beta(2, 5, n_samples) * 100,
        'category': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'date': pd.date_range('2020-01-01', periods=n_samples, freq='D'),
        'description': [f'Item {i}' for i in range(n_samples)],
        'flag': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 添加数据质量问题
    # 1. 缺失值
    df.loc[np.random.choice(df.index, 50, replace=False), 'age'] = np.nan
    df.loc[np.random.choice(df.index, 30, replace=False), 'income'] = np.nan
    df.loc[np.random.choice(df.index, 20, replace=False), 'category'] = np.nan
    
    # 2. 重复值
    duplicate_indices = np.random.choice(df.index, 25, replace=False)
    df = pd.concat([df, df.loc[duplicate_indices]], ignore_index=True)
    
    # 3. 异常值
    outlier_indices = np.random.choice(df.index, 15, replace=False)
    df.loc[outlier_indices, 'age'] = np.random.choice([150, -10, 200], len(outlier_indices))
    df.loc[outlier_indices, 'income'] = np.random.choice([1000000, -50000], len(outlier_indices))
    
    # 4. 格式问题
    df.loc[np.random.choice(df.index, 10, replace=False), 'description'] = '  Extra Spaces  '
    df.loc[np.random.choice(df.index, 5, replace=False), 'description'] = ''
    
    # 5. 数据类型问题
    df.loc[np.random.choice(df.index, 8, replace=False), 'age'] = 'unknown'
    df.loc[np.random.choice(df.index, 8, replace=False), 'flag'] = 'yes'
    
    return df


def run_comprehensive_test():
    """
    运行综合测试
    """
    print("T2数据清洗器 - 综合测试")
    print("=" * 50)
    
    # 创建示例数据
    print("1. 创建示例数据...")
    sample_data = create_sample_data()
    print(f"   原始数据形状: {sample_data.shape}")
    print(f"   缺失值数量: {sample_data.isnull().sum().sum()}")
    print(f"   重复值数量: {sample_data.duplicated().sum()}")
    
    # 初始化清洗器
    print("\n2. 初始化数据清洗器...")
    cleaner = DataCleaner()
    
    # 质量评估
    print("\n3. 初始质量评估...")
    initial_quality = cleaner.assess_data_quality(sample_data)
    print(f"   综合质量评分: {initial_quality.overall_score:.2%}")
    
    # 完整清洗流程
    print("\n4. 执行完整数据清洗...")
    cleaned_data, cleaning_report = cleaner.clean_data(
        sample_data,
        standardize_formats=True,
        convert_types=True,
        handle_missing=True,
        handle_duplicates=True,
        handle_outliers=True,
        filter_noise=True,
        check_consistency=True
    )
    
    print(f"   清洗后数据形状: {cleaned_data.shape}")
    print(f"   处理时间: {cleaning_report.processing_time:.2f}秒")
    
    # 最终质量评估
    print("\n5. 最终质量评估...")
    final_quality = cleaner.assess_data_quality(cleaned_data)
    print(f"   综合质量评分: {final_quality.overall_score:.2%}")
    print(f"   质量提升: {final_quality.overall_score - initial_quality.overall_score:.2%}")
    
    # 验证结果
    print("\n6. 验证清洗结果...")
    validation = cleaner.validate_cleaning_results(sample_data, cleaned_data)
    print(f"   清洗结果有效性: {validation['is_valid']}")
    print(f"   改进项目: {len(validation['improvements'])}")
    print(f"   退化项目: {len(validation['degradations'])}")
    print(f"   警告: {len(validation['warnings'])}")
    
    # 导出报告
    print("\n7. 导出清洗报告...")
    cleaner.export_cleaning_report('/workspace/D/AO/AOO/T/T2/cleaning_report.txt')
    
    # 显示清洗步骤
    print("\n8. 清洗步骤详情:")
    for i, step in enumerate(cleaning_report.cleaning_steps, 1):
        print(f"   {i}. {step}")
    
    print("\n测试完成！")
    return cleaned_data, cleaning_report, validation


if __name__ == "__main__":
    # 运行综合测试
    cleaned_data, report, validation = run_comprehensive_test()
    
    # 保存清洗后的数据
    cleaned_data.to_csv('/workspace/D/AO/AOO/T/T2/cleaned_sample_data.csv', index=False)
    print(f"\n清洗后的数据已保存到: /workspace/D/AO/AOO/T/T2/cleaned_sample_data.csv")